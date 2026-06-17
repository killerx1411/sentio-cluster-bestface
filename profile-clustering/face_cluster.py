"""
face_cluster.py  —  Steps 1–3 of the profile-extraction pipeline
====================================================================
CAPABILITIES (active features depend on installed optional packages):

  Core (always available):
    • Frame sampling + temporal spread selection
    • Seated-environment seat-map extraction (YOLO/GroundingDINO or fallback)
    • Camera homography / ground-plane calibration
    • Zoom-level calibration per seat (front-row vs back-row face sizes)
    • Compression-aware enhancement:
        - clean frames  → CLAHE + unsharp mask
        - blocky frames → bilateral filter (no artifact amplification)
    • Face detection: InsightFace → MTCNN → Haar cascade fallback stack
    • Torso ROI extraction derived from head bounding box
    • DUAL-REGION FEATURE TOWERS:v
        Face tower:
          - AdaFace IR-101 → InsightFace → histogram fallback embeddings
          - Gaze direction vector (yaw/pitch)
          - Head silhouette radial histogram
          - Forehead LBP micro-texture
          - Hair color histogram + hair presence flag
          - Glasses presence flag
        Torso tower:
          - OSNet re-ID embedding → CLIP-ReID → HOG fallback
          - Clothing HSV color histogram (3×32 bins)
          - Body proportion descriptor (shoulder/neck/head ratios)
          - Seated posture keypoints (10-point upper-body fingerprint)
          - HOG upper-chest texture patch
          - Writing/typing hand laterality
    • Adaptive confidence-weighted late fusion (α/β per-session calibration)
    • Seat-anchored tracklets (replace IoU tracking)
    • Seat-prior Bayesian post-processing (fragmentation/swap detection)
    • Illumination change tracking (frame weighting)
    • Per-seat occlusion masking from desk geometry
    • Bag/backpack color descriptor (seat-anchored)
    • Batched embedding extraction (EMBED_BATCH_SIZE=64, AdaFace only)
    • FAISS kNN graph + Chinese Whispers clustering
      (falls back to HDBSCAN if faiss/networkx unavailable)
    • KNOWN_N_PERSONS agglomerative merge (Stage 2)
    • Noise rescue, profile matching, full evaluation metrics
    • Structured logging (stderr, optional JSON-lines via LOG_JSON=true)
    • Pydantic settings model with FACE_* env-var overrides
    • JSON + NumPy directory cache (auto-migrates old .pkl files)

  Optional (enabled automatically when packages are installed):
    • Pose-aware quality gate      — pip install mediapipe
    • OSNet re-ID embeddings       — pip install torchreid
    • CLIP-ReID embeddings         — pip install clip (OpenAI CLIP)
    • Super-resolution upscaling   — pip install basicsr realesrgan
    • ByteTrack tracklet sampling  — pip install boxmot
    • YOLO seat detection          — pip install ultralytics

====================================================================
SEATED-ENVIRONMENT DESIGN:
  The core insight: in classrooms/offices people remain anchored to one
  seat for the entire session. Spatial position is used as a prior
  BEFORE examining embeddings, transforming the clustering problem.

  Seat Map → Seat-Anchored Tracklets → Dual-Tower Features →
  Confidence-Weighted Adaptive Fusion → Seat-Prior Bayesian Correction
====================================================================
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────
import gc
import json
import logging
import math
import os
import re
import time
import warnings
import networkx as nx
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ══════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════

logger = logging.getLogger("face_cluster")
_logging_configured = False


def setup_logging(level: int = logging.INFO, json_output: bool = False) -> None:
    global _logging_configured
    if _logging_configured:
        return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    if json_output:
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level":     record.levelname,
                    "event":     record.getMessage(),
                }
                if record.exc_info:
                    payload["exc_info"] = self.formatException(record.exc_info)
                return json.dumps(payload)
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logging.getLogger("face_cluster").addHandler(handler)
    logging.getLogger("face_cluster").setLevel(level)
    _logging_configured = True


# ══════════════════════════════════════════════════════════════════
#  LIBRARY DETECTION
# ══════════════════════════════════════════════════════════════════

# ── InsightFace ───────────────────────────────────────────────────
INSIGHTFACE = False
_insight_app = None
_INSIGHT_MODEL = None

try:
    from insightface.app import FaceAnalysis as _InsightFaceApp
    for _model_name in ["antelopev2", "buffalo_l", "buffalo_sc"]:
        try:
            _app = _InsightFaceApp(
                name=_model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            _app.prepare(ctx_id=0, det_size=(640, 640))
            _insight_app = _app
            _INSIGHT_MODEL = _model_name
            INSIGHTFACE = True
            logger.info(f"InsightFace ({_model_name}) loaded")
            break
        except Exception as _e:
            logger.warning(f"InsightFace {_model_name} not available: {_e}")
except Exception as e:
    logger.warning(f"InsightFace not available ({e})")

# ── AdaFace ───────────────────────────────────────────────────────
ADAFACE = False
_adaface_model = None
try:
    import torch
    import sys as _sys

    _adaface_dir = Path(__file__).parent / "AdaFace"
    _root_dir = Path(__file__).parent
    for _p in [str(_adaface_dir), str(_root_dir)]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    from net import build_model as _adaface_build

    _w1 = _adaface_dir / "weights" / "adaface_ir101_webface12m.ckpt"
    _w2 = Path(__file__).parent / "weights" / "adaface_ir101_webface12m.ckpt"
    _ADAFACE_WEIGHTS = _w1 if _w1.exists() else _w2

    if _ADAFACE_WEIGHTS.exists():
        _adaface_model = _adaface_build("ir_101")
        ckpt = torch.load(str(_ADAFACE_WEIGHTS), map_location="cpu")
        _statedict = {
            k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")
        }
        _adaface_model.load_state_dict(_statedict)
        _adaface_model.eval()
        if torch.cuda.is_available():
            _adaface_model = _adaface_model.cuda()
        ADAFACE = True
        logger.info("AdaFace IR-101 (WebFace12M) loaded")
    else:
        logger.warning(f"AdaFace weights not found at {_w1} or {_w2} — skipping")
except Exception as e:
    logger.warning(f"AdaFace not available ({e})")

# ── MTCNN ─────────────────────────────────────────────────────────
MTCNN = False
_mtcnn_det = None
try:
    from mtcnn import MTCNN as _MTCNNCls
    _mtcnn_det = _MTCNNCls()
    MTCNN = True
    logger.info("MTCNN loaded")
except Exception as e:
    logger.warning(f"MTCNN not available ({e})")

# ── MediaPipe Face Mesh (pose + gaze estimation) ──────────────────
MEDIAPIPE = False
_mp_face_mesh = None
_mp_pose = None
try:
    import mediapipe as _mp
    _mp_face_mesh_module = _mp.solutions.face_mesh
    _mp_face_mesh = _mp_face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )
    # Also load Pose for upper-body keypoints
    try:
        _mp_pose_module = _mp.solutions.pose
        _mp_pose = _mp_pose_module.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
    except Exception:
        pass
    MEDIAPIPE = True
    logger.info("MediaPipe Face Mesh + Pose loaded (gaze + posture enabled)")
except Exception as _e:
    logger.warning(f"MediaPipe not available ({_e}) — pose/gaze gates bypassed")

# ── OSNet (torchreid) — torso re-ID ──────────────────────────────
OSNET = False
_osnet_model = None
try:
    import torchreid
    _osnet_model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        pretrained=True,
    )
    _osnet_model.eval()
    try:
        if torch.cuda.is_available():
            _osnet_model = _osnet_model.cuda()
    except Exception:
        pass
    OSNET = True
    logger.info("OSNet x1.0 (torchreid) loaded — torso re-ID enabled")
except Exception as _e:
    logger.warning(f"OSNet not available ({_e}) — pip install torchreid")

# ── CLIP-ReID fallback ────────────────────────────────────────────
CLIP_REID = False
_clip_model = None
_clip_preprocess = None
try:
    import clip as _clip_lib
    _clip_model, _clip_preprocess = _clip_lib.load("ViT-B/32", device="cpu")
    _clip_model.eval()
    CLIP_REID = True
    logger.info("CLIP ViT-B/32 loaded — CLIP-ReID torso fallback enabled")
except Exception as _e:
    logger.warning(f"CLIP not available ({_e}) — pip install clip")

# ── YOLO seat detection ───────────────────────────────────────────
YOLO = False
_yolo_model = None
try:
    from ultralytics import YOLO as _YOLOCls
    _yolo_model = _YOLOCls("yolov8n.pt")
    YOLO = True
    logger.info("YOLOv8n loaded — seat detection enabled")
except Exception as _e:
    logger.warning(f"YOLO not available ({_e}) — pip install ultralytics")

# ── RealESRGAN super-resolution ───────────────────────────────────
REALESRGAN = False
_realesrgan_upsampler = None
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet as _RRDBNet
    from realesrgan import RealESRGANer as _RealESRGANer
    _sr_weights = Path(__file__).parent / "weights" / "RealESRGAN_x4plus.pth"
    if _sr_weights.exists():
        _sr_model = _RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        _realesrgan_upsampler = _RealESRGANer(
            scale=4, model_path=str(_sr_weights), model=_sr_model,
            tile=128, tile_pad=10, pre_pad=0, half=False,
        )
        REALESRGAN = True
        logger.info("RealESRGAN x4 loaded (super-resolution enabled)")
    else:
        logger.warning(f"RealESRGAN weights not found at {_sr_weights}")
except Exception as _e:
    logger.warning(f"RealESRGAN not available ({_e}) — using Lanczos upscaling")

# ── ByteTrack tracklet sampling ───────────────────────────────────
BYTETRACK = False
try:
    from boxmot import ByteTrack as _ByteTrackCls
    BYTETRACK = True
    logger.info("boxmot ByteTrack loaded (tracklet-based sampling enabled)")
except Exception as _e:
    logger.warning(f"boxmot not available ({_e}) — using temporal sampling")

# ── Active embedding engine ───────────────────────────────────────
if ADAFACE:
    ACTIVE_ENGINE = "adaface"
elif INSIGHTFACE:
    ACTIVE_ENGINE = "insightface"
else:
    ACTIVE_ENGINE = "histogram"
    logger.warning("All neural face engines unavailable — clustering accuracy will be poor.")

if OSNET:
    ACTIVE_TORSO_ENGINE = "osnet"
elif CLIP_REID:
    ACTIVE_TORSO_ENGINE = "clip"
else:
    ACTIVE_TORSO_ENGINE = "hog"
    logger.warning("No neural torso engine — falling back to HOG descriptor.")

logger.info(f"Active face engine: {ACTIVE_ENGINE}  |  Active torso engine: {ACTIVE_TORSO_ENGINE}")


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field as _Field
    _PYDANTIC_AVAILABLE = True
    _PYDANTIC_V2 = True
    print("pydantic avail")
except ImportError:
    try:
        from pydantic import BaseSettings, Field as _Field
        SettingsConfigDict = None
        _PYDANTIC_AVAILABLE = True
        _PYDANTIC_V2 = False
    except ImportError:
        _PYDANTIC_AVAILABLE = False
        _PYDANTIC_V2 = False

if _PYDANTIC_AVAILABLE:
    class PipelineConfig(BaseSettings):
        if _PYDANTIC_V2 and SettingsConfigDict is not None:
            model_config = SettingsConfigDict(env_prefix="FACE_")
        else:
            class Config:
                env_prefix = "FACE_"

        # Sampling
        sample_interval_sec:              float         = 1.5
        min_face_quality:                 float         = 4.0
        absolute_max_frames:              int           = 600
        # Clustering
        hdbscan_min_cluster_size:         int           = 3
        hdbscan_min_samples:              int           = 2
        hdbscan_epsilon_adaface:          float         = 0.44   # tighter for 1024-d fused
        hdbscan_epsilon_insightface:      float         = 0.44
        hdbscan_epsilon_deepface:         float         = 0.45
        hdbscan_epsilon_face_recognition: float         = 0.42
        hdbscan_epsilon_histogram:        float         = 0.50
        known_n_persons:                  Optional[int] = None
        merge_jump_factor:                float         = 1.4
        illum_drop_threshold:             float = 0.05
        centroid_merge_thresh:            float = 0.30
        # Noise rescue
        noise_rescue_dist_adaface:          float       = 0.56
        noise_rescue_dist_insightface:      float       = 0.62
        noise_rescue_dist_deepface:         float       = 0.65
        noise_rescue_dist_face_recognition: float       = 0.60
        noise_rescue_dist_histogram:        float       = 0.70
        # kNN
        knn_k:                            int           = 30    # slightly larger for 1024-d
        # SR
        sr_threshold_px:                  int           = 60
        # Tracking
        tracking_fps:                     int           = 6
        embed_batch_size:                 int           = 64
        artifact_threshold:               float         = 0.4
        # Pose
        max_yaw_deg:                      float         = 45.0
        max_pitch_deg:                    float         = 30.0
        # Logging
        log_json:                         bool          = False
        # Profile matching
        profile_match_threshold:          float         = 0.45
        # ── NEW: seated-environment settings ──────────────────────
        # Fusion weights (calibrated per-session if calibrate_fusion=True)
        face_alpha:                       float         = 0.6    # face tower weight
        torso_beta:                       float         = 0.4    # torso tower weight
        calibrate_fusion:                 bool          = True   # auto-calibrate α/β
        calibration_window_sec:           float         = 60.0   # first N secs for calib
        # Seat map
        seat_iou_thresh:                  float         = 0.40
        seat_grid_tolerance_px:           int           = 40     # px radius for seat anchor
        min_seat_size_px:                 int           = 30
        # Torso extraction
        torso_height_factor:              float         = 2.8    # head_heights below chin
        torso_width_factor:               float         = 2.3    # head_widths centered
        min_torso_quality:                float         = 3.0
        # Outlier rejection within tracklet
        tracklet_outlier_dist:            float         = 0.50
        # Seat-prior Bayesian parameters
        same_seat_merge_prior:            float         = 0.85   # boosts merge if same seat
        diff_seat_dist_min_px:            int           = 120    # seats this far get no prior
        # Illumination
        illum_change_thresh:              float         = 40.0   # mean LAB-L diff threshold
        # Occlusion / desk masking
        desk_mask_bottom_frac:            float         = 0.45   # bottom N% of frame = desk
        # Profile dir (CLI override)
        profile_dir:                      Optional[str] = None

    config = PipelineConfig()

else:
    from dataclasses import dataclass as _dc
    print("Plainconfig")
    @_dc
    class _PlainConfig:
        sample_interval_sec:              float         = 1.5
        min_face_quality:                 float         = 4.0
        absolute_max_frames:              int           = 600
        hdbscan_min_cluster_size:         int           = 3
        hdbscan_min_samples:              int           = 2
        hdbscan_epsilon_adaface:          float         = 0.44
        hdbscan_epsilon_insightface:      float         = 0.44
        hdbscan_epsilon_deepface:         float         = 0.45
        hdbscan_epsilon_face_recognition: float         = 0.42
        hdbscan_epsilon_histogram:        float         = 0.50
        known_n_persons:                  Optional[int] = None
        merge_jump_factor:                float         = 1.4
        noise_rescue_dist_adaface:          float       = 0.56
        noise_rescue_dist_insightface:      float       = 0.62
        noise_rescue_dist_deepface:         float       = 0.65
        noise_rescue_dist_face_recognition: float       = 0.60
        noise_rescue_dist_histogram:        float       = 0.70
        knn_k:                            int           = 30
        sr_threshold_px:                  int           = 60
        tracking_fps:                     int           = 6
        embed_batch_size:                 int           = 64
        artifact_threshold:               float         = 0.4
        max_yaw_deg:                      float         = 45.0
        max_pitch_deg:                    float         = 30.0
        log_json:                         bool          = False
        profile_match_threshold:          float         = 0.45
        face_alpha:                       float         = 0.6
        torso_beta:                       float         = 0.4
        calibrate_fusion:                 bool          = True
        calibration_window_sec:           float         = 60.0
        seat_iou_thresh:                  float         = 0.40
        seat_grid_tolerance_px:           int           = 40
        min_seat_size_px:                 int           = 30
        torso_height_factor:              float         = 2.8
        torso_width_factor:               float         = 2.3
        min_torso_quality:                float         = 3.0
        tracklet_outlier_dist:            float         = 0.50
        same_seat_merge_prior:            float         = 0.85
        diff_seat_dist_min_px:            int           = 120
        illum_change_thresh:              float         = 40.0
        desk_mask_bottom_frac:            float         = 0.45
        profile_dir:                      Optional[str] = None
        illum_drop_threshold:             float = 0.05
        centroid_merge_thresh:            float = 0.30

    _config_raw = _PlainConfig()
    import os as _os
    for _fname in _PlainConfig.__dataclass_fields__:
        _evar = "FACE_" + _fname.upper()
        _eval = _os.environ.get(_evar)
        if _eval is not None:
            _ftype = type(getattr(_config_raw, _fname))
            try:
                if _ftype is bool:
                    setattr(_config_raw, _fname, _eval.lower() in ("1", "true", "yes"))
                elif _ftype is int:
                    setattr(_config_raw, _fname, int(_eval))
                elif _ftype is float:
                    setattr(_config_raw, _fname, float(_eval))
                else:
                    setattr(_config_raw, _fname, _eval)
            except Exception:
                pass
    config = _config_raw

# ── Backwards-compatible module-level aliases ─────────────────────
SAMPLE_INTERVAL_SEC      = config.sample_interval_sec
MIN_FACE_QUALITY         = config.min_face_quality
ABSOLUTE_MAX_FRAMES      = config.absolute_max_frames
HDBSCAN_MIN_CLUSTER_SIZE = config.hdbscan_min_cluster_size
HDBSCAN_MIN_SAMPLES      = config.hdbscan_min_samples
HDBSCAN_EPSILON = {
    "adaface":          config.hdbscan_epsilon_adaface,
    "insightface":      config.hdbscan_epsilon_insightface,
    "deepface":         config.hdbscan_epsilon_deepface,
    "face_recognition": config.hdbscan_epsilon_face_recognition,
    "histogram":        config.hdbscan_epsilon_histogram,
}
KNOWN_N_PERSONS   = config.known_n_persons
MERGE_JUMP_FACTOR = config.merge_jump_factor
NOISE_RESCUE_DIST = {
    "adaface":          config.noise_rescue_dist_adaface,
    "insightface":      config.noise_rescue_dist_insightface,
    "deepface":         config.noise_rescue_dist_deepface,
    "face_recognition": config.noise_rescue_dist_face_recognition,
    "histogram":        config.noise_rescue_dist_histogram,
}
KNN_K              = config.knn_k
SR_THRESHOLD_PX    = config.sr_threshold_px
TRACKING_FPS       = config.tracking_fps
EMBED_BATCH_SIZE   = config.embed_batch_size
ARTIFACT_THRESHOLD = config.artifact_threshold
MAX_YAW_DEG        = config.max_yaw_deg
MAX_PITCH_DEG      = config.max_pitch_deg
LOG_JSON           = config.log_json
PROFILE_MATCH_THRESHOLD = config.profile_match_threshold


# ══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class FaceDetection:
    frame_idx:          int
    timestamp_sec:      float
    embedding:          np.ndarray          # fused 1024-d (face 512 + torso 512) or face-only 512
    crop_bgr:           np.ndarray          # face crop
    quality_score:      float
    bbox:               tuple               # (x, y, w, h) face bbox
    confidence:         float
    landmarks:          Optional[list]
    # ── New fields ─────────────────────────────────────────────
    face_embedding:     Optional[np.ndarray] = None   # raw 512-d face embedding
    torso_embedding:    Optional[np.ndarray] = None   # raw 512-d torso embedding
    torso_crop_bgr:     Optional[np.ndarray] = None   # torso crop
    torso_quality:      float                = 0.0
    face_conf:          float                = 0.0    # 0–1 face confidence weight
    torso_conf:         float                = 0.0    # 0–1 torso confidence weight
    seat_id:            Optional[str]        = None   # "row_col" seat label
    seat_centroid_px:   Optional[tuple]      = None   # (cx, cy) image coords
    gaze_vector:        Optional[tuple]      = None   # (yaw_deg, pitch_deg)
    head_silhouette:    Optional[np.ndarray] = None   # 32-bin radial histogram
    lbp_texture:        Optional[np.ndarray] = None   # 256-bin LBP on forehead
    hair_color_hist:    Optional[np.ndarray] = None   # 48-bin HSV on hair region
    hair_present:       Optional[bool]       = None
    has_glasses:        Optional[bool]       = None
    clothing_hist:      Optional[np.ndarray] = None   # 96-bin HSV torso histogram
    body_proportions:   Optional[np.ndarray] = None   # 5-value ratio vector
    posture_keypoints:  Optional[np.ndarray] = None   # 20-value normalized KP
    chest_hog:          Optional[np.ndarray] = None   # HOG descriptor
    writing_hand:       Optional[str]        = None   # "left" / "right" / None
    bag_color_hist:     Optional[np.ndarray] = None   # 48-bin HSV from seat periphery
    illum_score:        float                = 1.0    # illumination quality weight


@dataclass
class ClusterResult:
    id:    int
    faces: list = field(default_factory=list)   # list[FaceDetection]

    @property
    def size(self) -> int:
        return len(self.faces)

    @property
    def mean_quality(self) -> float:
        if not self.faces:
            return 0.0
        return float(np.mean([f.quality_score for f in self.faces]))

    @property
    def centroid(self) -> np.ndarray:
        embeddings = np.stack([f.embedding for f in self.faces])
        c = embeddings.mean(axis=0)
        n = np.linalg.norm(c)
        return c / n if n > 0 else c

    @property
    def dominant_seat(self) -> Optional[str]:
        seats = [f.seat_id for f in self.faces if f.seat_id is not None]
        if not seats:
            return None
        return Counter(seats).most_common(1)[0][0]


# ── Seat map entry ────────────────────────────────────────────────
@dataclass
class SeatEntry:
    seat_id:       str            # "row_col"
    centroid_px:   tuple          # (cx, cy) in image coords
    centroid_world: Optional[tuple] = None   # (x_m, y_m) after homography
    bbox_px:       Optional[tuple] = None    # (x, y, w, h)
    face_size_est_px: float = 0.0            # estimated face pixel size from this seat
    occlusion_mask: Optional[np.ndarray] = None  # desk occlusion mask (frame-sized)
# ── Per-cluster profile match record ─────────────────────────────
@dataclass
class MatchResult:
    """
    Full top-2 profile match record for one cluster.
    Always populated regardless of threshold, so fragmentation
    detection and evaluation can use the complete picture.
    """
    cluster_id:    int
    best_name:     str
    best_dist:     float
    second_name:   str
    second_dist:   float
    is_confident:  bool   # best_dist < threshold used at match time

# ── Session-level state (populated during scene initialisation) ───
@dataclass
class SessionState:
    seat_map:          Dict[str, SeatEntry] = field(default_factory=dict)
    homography:        Optional[np.ndarray] = None  # 3×3 ground-plane H
    fusion_alpha:      float = 0.6                  # face weight (calibrated)
    fusion_beta:       float = 0.4                  # torso weight (calibrated)
    illum_baseline:    Optional[np.ndarray] = None  # reference LAB-L channel
    desk_mask:         Optional[np.ndarray] = None  # binary mask: True = desk area
    calibration_done:  bool = False

    # Seat-anchored tracklets: seat_id → list of FaceDetection
    seat_tracklets:    Dict[str, List] = field(default_factory=lambda: defaultdict(list))


# Module-level session state (reset per video)
_session = SessionState()


# ══════════════════════════════════════════════════════════════════
#  FRAME SAMPLING
# ══════════════════════════════════════════════════════════════════

def _compute_sampling_budget(duration_s: float) -> int:
    duration_min = duration_s / 60.0
    budget = int(50 + 35 * math.sqrt(duration_min))
    return min(budget, config.absolute_max_frames)


def _select_frames_temporal(
    candidates: list,
    budget: int,
) -> list:
    if not candidates:
        return []
    max_idx = max(c[0] for c in candidates)
    min_idx = min(c[0] for c in candidates)
    span = max(max_idx - min_idx, 1)
    bucket_size = span / budget
    buckets: Dict[int, list] = {}
    for item in candidates:
        b = min(int((item[0] - min_idx) / bucket_size), budget - 1)
        buckets.setdefault(b, []).append(item)
    selected = []
    for b in range(budget):
        group = buckets.get(b)
        if group:
            selected.append(max(group, key=lambda x: x[2]))
    selected.sort(key=lambda x: x[0])
    return selected


# ══════════════════════════════════════════════════════════════════
#  SCENE INITIALISATION — SEAT MAP EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _detect_persons_yolo(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """Returns list of (x1, y1, x2, y2, conf) person bboxes via YOLO."""
    if not YOLO or _yolo_model is None:
        return []
    try:
        results = _yolo_model(frame_bgr, classes=[0], verbose=False)  # class 0 = person
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                if conf > 0.55:
                    boxes.append((x1, y1, x2, y2, conf))
        return boxes
    except Exception as _e:
        logger.warning(f"YOLO detection failed ({_e})")
        return []


def _detect_persons_haar_fallback(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """Fallback: use upper-body Haar cascade."""
    try:
        xml = "haarcascade_upperbody.xml"
        cas = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
        if cas.empty():
            # Further fallback: face cascade + expand bbox down
            cas = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if cas.empty():
                return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dets = cas.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
        boxes = []
        for x, y, w, h in dets:
            # Expand to approximate torso
            x2 = min(frame_bgr.shape[1], x + w)
            y2 = min(frame_bgr.shape[0], y + int(h * 2.5))
            boxes.append((x, y, x2, y2, 0.5))
        return boxes
    except Exception:
        return []


def build_seat_map(init_frame: np.ndarray) -> Dict[str, SeatEntry]:
    """
    Detects all occupied seats in a wide-angle initialisation frame.
    Uses YOLO if available, falls back to Haar cascade.
    Assigns grid row/col labels, estimates face size per seat.
    Also builds a desk occlusion mask.
    Returns a dict of seat_id → SeatEntry.
    """
    h_frame, w_frame = init_frame.shape[:2]

    # Detect persons
    persons = _detect_persons_yolo(init_frame)
    if not persons:
        persons = _detect_persons_haar_fallback(init_frame)
    if not persons:
        logger.warning("Seat map: no persons detected — spatial prior disabled")
        return {}

    logger.info(f"Seat map: {len(persons)} persons detected in init frame")
    for p in persons[:10]:
        logger.info(f"  bbox={p[:4]} conf={p[4]:.2f}")
    # Compute seat centroids (use top-third of person bbox ≈ head/torso area)
    centroids = []
    bboxes = []
    for (x1, y1, x2, y2, conf) in persons:
        cx = (x1 + x2) // 2
        cy = y1 + (y2 - y1) // 4  # upper quarter = head region
        centroids.append((cx, cy))
        bboxes.append((x1, y1, x2 - x1, y2 - y1))

    # Sort centroids into grid: first by row (y), then by col (x) within each row
    if not centroids:
        return {}

    ys = np.array([c[1] for c in centroids], dtype=float)
    xs = np.array([c[0] for c in centroids], dtype=float)

    # Cluster y-values into rows using simple threshold
    row_thresh = h_frame * 0.08  # 8% of frame height = same row
    sorted_y_idx = np.argsort(ys)
    rows: List[List[int]] = []
    current_row: List[int] = [sorted_y_idx[0]]
    for idx in sorted_y_idx[1:]:
        if ys[idx] - ys[current_row[-1]] < row_thresh:
            current_row.append(idx)
        else:
            rows.append(sorted(current_row, key=lambda i: xs[i]))
            current_row = [idx]
    rows.append(sorted(current_row, key=lambda i: xs[i]))

    seat_map: Dict[str, SeatEntry] = {}
    for row_idx, row_members in enumerate(rows):
        for col_idx, member_idx in enumerate(row_members):
            seat_id = f"r{row_idx}_c{col_idx}"
            cx, cy = centroids[member_idx]
            bx, by, bw, bh = bboxes[member_idx]
            # Estimate face size: person bbox height / 6 (rough empirical ratio)
            face_size_est = max(bh / 6.0, 20.0)
            seat_map[seat_id] = SeatEntry(
                seat_id=seat_id,
                centroid_px=(cx, cy),
                bbox_px=(bx, by, bw, bh),
                face_size_est_px=face_size_est,
            )

    logger.info(f"Seat map: {len(seat_map)} seats assigned across {len(rows)} rows")

    MAX_SEATS = 50
    if len(seat_map) > MAX_SEATS:
        logger.warning(f"Seat map: {len(seat_map)} seats exceeds cap {MAX_SEATS} — spatial prior disabled")
        return {}

    return seat_map


def estimate_homography(seat_map: Dict[str, SeatEntry], frame_shape: tuple) -> Optional[np.ndarray]:
    """
    Estimate a simple ground-plane homography from seat centroids.
    Assumes a roughly rectangular grid layout in the real world.
    Returns 3×3 H matrix (image → world) or None if insufficient seats.
    """
    if len(seat_map) < 4:
        return None
    try:
        # Group seats by row/col
        rows: Dict[int, List[SeatEntry]] = defaultdict(list)
        for entry in seat_map.values():
            r = int(entry.seat_id.split("_")[0][1:])
            rows[r].append(entry)

        # Build correspondences: image points ↔ world points (row×seat_spacing)
        SEAT_SPACING_M = 0.9  # assumed seat pitch in meters
        img_pts, world_pts = [], []
        for r, seats in sorted(rows.items()):
            for c_idx, seat in enumerate(sorted(seats, key=lambda s: s.centroid_px[0])):
                img_pts.append(seat.centroid_px)
                world_pts.append((c_idx * SEAT_SPACING_M, r * SEAT_SPACING_M))

        if len(img_pts) < 4:
            return None
        src = np.array(img_pts, dtype=np.float32)
        dst = np.array(world_pts, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H
    except Exception as _e:
        logger.warning(f"Homography estimation failed ({_e})")
        return None


def build_desk_occlusion_mask(frame_bgr: np.ndarray, seat_map: Dict[str, SeatEntry]) -> np.ndarray:
    """
    Builds a binary mask where True = likely desk/occlusion area.
    Simple heuristic: bottom desk_mask_bottom_frac of the frame,
    refined by seat bounding box bottom edges.
    """
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    # Conservative bottom-fraction mask
    desk_y = int(h * (1.0 - config.desk_mask_bottom_frac))
    mask[desk_y:, :] = True

    # Refine: mark below each person bbox's lower third as desk
    for entry in seat_map.values():
        if entry.bbox_px is None:
            continue
        bx, by, bw, bh = entry.bbox_px
        desk_start = by + int(bh * 0.6)
        desk_end   = min(h, by + bh)
        mask[desk_start:desk_end, bx:bx + bw] = True

    return mask


# ══════════════════════════════════════════════════════════════════
#  IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════

def detect_compression_artifacts(img_bgr: np.ndarray) -> float:
    try:
        h, w = img_bgr.shape[:2]
        if h > 320:
            img_bgr = cv2.resize(img_bgr, (int(w * 320 / h), 320), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bh, bw = gray.shape
        bh8 = (bh // 8) * 8
        bw8 = (bw // 8) * 8
        if bh8 < 8 or bw8 < 8:
            return 0.0
        blocks     = gray[:bh8, :bw8].reshape(bh8 // 8, 8, bw8 // 8, 8)
        block_means = blocks.mean(axis=(1, 3))
        within_var  = float(blocks.var(axis=(1, 3)).mean())
        between_var = float(block_means.var())
        if between_var < 1e-6:
            return 0.0
        ratio = within_var / (between_var + 1e-6)
        return float(np.clip(ratio / 5.0, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_illumination_score(frame_bgr: np.ndarray, baseline_l: Optional[np.ndarray]) -> float:
    """
    Returns 0–1 weight for this frame's illumination quality.
    Frames with large LAB-L deviation from the session baseline are down-weighted.
    """
    try:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0].astype(np.float32)
        if baseline_l is None:
            return 1.0
        # Resize baseline if shapes differ (e.g. frame was resized)
        if l_ch.shape != baseline_l.shape:
            bl = cv2.resize(baseline_l, (l_ch.shape[1], l_ch.shape[0]))
        else:
            bl = baseline_l
        mean_diff = float(np.abs(l_ch - bl).mean())
        # Sigmoid-like penalty: weight=1 if diff≈0, weight→0 as diff→illum_thresh
        weight = max(0.1, 1.0 - mean_diff / max(config.illum_change_thresh, 1.0))
        return float(np.clip(weight, 0.0, 1.0))
    except Exception:
        return 1.0


def enhance_frame(img_bgr: np.ndarray) -> np.ndarray:
    try:
        artifact_score = detect_compression_artifacts(img_bgr)
        if artifact_score > config.artifact_threshold:
            return cv2.bilateralFilter(img_bgr, d=5, sigmaColor=30, sigmaSpace=30)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    except Exception:
        return img_bgr


def upscale_face(face_bgr: np.ndarray, target: int = 112) -> np.ndarray:
    h, w = face_bgr.shape[:2]
    if max(h, w) >= target:
        return face_bgr
    scale = target / max(h, w)
    interp = cv2.INTER_LANCZOS4 if scale < 4 else cv2.INTER_CUBIC
    return cv2.resize(face_bgr, (int(w * scale), int(h * scale)), interpolation=interp)


def super_resolve_face(face_bgr: np.ndarray, target: int = 112) -> np.ndarray:
    h, w = face_bgr.shape[:2]
    if min(h, w) >= config.sr_threshold_px:
        return upscale_face(face_bgr, target)
    if REALESRGAN and _realesrgan_upsampler is not None:
        try:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            output, _ = _realesrgan_upsampler.enhance(rgb, outscale=4)
            sr_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            return cv2.resize(sr_bgr, (target, target), interpolation=cv2.INTER_LANCZOS4)
        except Exception as _e:
            logger.warning(f"RealESRGAN inference failed ({_e}), falling back to Lanczos")
    return upscale_face(face_bgr, target)


def align_face_affine(img_bgr: np.ndarray, landmarks) -> np.ndarray:
    try:
        dst = np.array(
            [[38.29, 51.69], [73.53, 51.50], [56.02, 71.73],
             [41.55, 92.37], [70.73, 92.38]], dtype=np.float32,
        )
        src = np.array(landmarks[:5], dtype=np.float32)
        M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
        if M is not None:
            return cv2.warpAffine(img_bgr, M, (112, 112), flags=cv2.INTER_LINEAR)
    except Exception:
        pass
    return cv2.resize(img_bgr, (112, 112))


# ══════════════════════════════════════════════════════════════════
#  SEAT ASSIGNMENT
# ══════════════════════════════════════════════════════════════════

def assign_seat(
    head_cx: float, head_cy: float,
    seat_map: Dict[str, SeatEntry],
) -> Optional[str]:
    """
    Returns the seat_id whose centroid is closest to (head_cx, head_cy),
    if within config.seat_grid_tolerance_px pixels. Otherwise None.
    """
    if not seat_map:
        return None
    best_id, best_dist = None, float("inf")
    for sid, entry in seat_map.items():
        scx, scy = entry.centroid_px
        dist = math.hypot(head_cx - scx, head_cy - scy)
        if dist < best_dist:
            best_dist = dist
            best_id = sid
    if best_dist <= config.seat_grid_tolerance_px:
        return best_id
    return None


# ══════════════════════════════════════════════════════════════════
#  TORSO ROI EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_torso_roi(
    frame_bgr: np.ndarray,
    face_bbox: tuple,  # (x, y, w, h)
    desk_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Derives a torso crop from the face bounding box deterministically.
    Top:   chin y + small pad (= face_y + face_h)
    Bottom: chin_y + torso_height_factor * face_h
    Width:  torso_width_factor * face_w, centered on face_cx
    Applies desk occlusion mask: zero-fills masked rows.
    Returns None if crop area is too small or fully occluded.
    """
    fx, fy, fw, fh = face_bbox
    if fw < 8 or fh < 8:
        return None

    face_cx = fx + fw // 2
    chin_y  = fy + fh

    torso_h = int(fh * config.torso_height_factor)
    torso_w = int(fw * config.torso_width_factor)

    tx1 = max(0, face_cx - torso_w // 2)
    ty1 = min(frame_bgr.shape[0] - 1, chin_y)
    tx2 = min(frame_bgr.shape[1], tx1 + torso_w)
    ty2 = min(frame_bgr.shape[0], ty1 + torso_h)

    if tx2 - tx1 < 20 or ty2 - ty1 < 20:
        return None

    torso_crop = frame_bgr[ty1:ty2, tx1:tx2].copy()

    # Apply desk occlusion mask: zero-fill desk pixels so they don't pollute histograms
    if desk_mask is not None:
        try:
            desk_region = desk_mask[ty1:ty2, tx1:tx2]
            torso_crop[desk_region] = 0
        except Exception:
            pass

    return torso_crop


def torso_quality_score(torso_bgr: np.ndarray, desk_mask_local: Optional[np.ndarray] = None) -> float:
    """
    Scores 0–∞ torso crop quality.
    Penalises: too small, mostly zero (desk-occluded), blurry.
    """
    if torso_bgr is None:
        return 0.0
    h, w = torso_bgr.shape[:2]
    if w < 30 or h < 30:
        return 0.0
    # Occlusion check: fraction of non-zero pixels
    nonzero_frac = float(np.count_nonzero(torso_bgr.sum(axis=2))) / max(h * w, 1)
    if nonzero_frac < 0.25:   # >75% occluded by desk
        return 0.0
    gray = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    size_score = min(1.0, min(h, w) / 60.0)
    return sharpness * size_score * nonzero_frac


# ══════════════════════════════════════════════════════════════════
#  FACE QUALITY & POSE
# ══════════════════════════════════════════════════════════════════

def estimate_yaw_pitch(face_bgr: np.ndarray) -> Tuple[float, float]:
    if not MEDIAPIPE or _mp_face_mesh is None:
        return 0.0, 0.0
    try:
        h, w = face_bgr.shape[:2]
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        results = _mp_face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return 0.0, 0.0
        lm = results.multi_face_landmarks[0].landmark

        def _xy(idx: int) -> Tuple[float, float]:
            return lm[idx].x * w, lm[idx].y * h

        nose_x, nose_y  = _xy(1)
        leye_x, _       = _xy(33)
        reye_x, _       = _xy(263)
        eye_width = reye_x - leye_x
        if eye_width < 1e-3:
            return 0.0, 0.0
        eye_mid_x = (leye_x + reye_x) / 2.0
        yaw_norm  = (nose_x - eye_mid_x) / (eye_width / 2.0)
        yaw_deg   = float(np.clip(yaw_norm * 90.0, -90.0, 90.0))
        forehead_y = _xy(10)[1]
        chin_y     = _xy(152)[1]
        face_height = abs(chin_y - forehead_y)
        if face_height < 1e-3:
            return yaw_deg, 0.0
        nose_frac  = (nose_y - forehead_y) / face_height
        pitch_norm = (nose_frac - 0.60) / 0.20
        pitch_deg  = float(np.clip(pitch_norm * 90.0, -90.0, 90.0))
        return yaw_deg, pitch_deg
    except Exception:
        return 0.0, 0.0


def face_quality_score(face_bgr: np.ndarray) -> float:
    h, w = face_bgr.shape[:2]
    if h < 5 or w < 5:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness    = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    size_score   = min(1.0, min(h, w) / 80.0)
    mean_bright  = float(gray.mean())
    bright_mul   = 1.0 if 35 < mean_bright < 225 else 0.25
    contrast_mul = min(1.0, float(gray.std()) / 20.0)
    base_score   = float(sharpness * size_score * bright_mul * contrast_mul)
    if MEDIAPIPE:
        yaw_deg, pitch_deg = estimate_yaw_pitch(face_bgr)
        if abs(yaw_deg) > config.max_yaw_deg or abs(pitch_deg) > config.max_pitch_deg:
            return 0.0
        pose_penalty = max(0.1, 1.0 - abs(yaw_deg) / 90.0)
        base_score  *= pose_penalty
    return base_score


def laplacian_sharpness(img_bgr: np.ndarray) -> float:
    h, w = img_bgr.shape[:2]
    if h > 320:
        img_bgr = cv2.resize(img_bgr, (int(w * 320 / h), 320), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


# ══════════════════════════════════════════════════════════════════
#  AUXILIARY DESCRIPTORS — FACE TOWER
# ══════════════════════════════════════════════════════════════════

def compute_gaze_vector(face_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
    """Returns (yaw_deg, pitch_deg) gaze estimate."""
    yaw, pitch = estimate_yaw_pitch(face_bgr)
    return (yaw, pitch)


def compute_head_silhouette(face_bgr: np.ndarray, n_bins: int = 32) -> Optional[np.ndarray]:
    """
    Radial histogram of the head silhouette.
    Converts to grayscale, thresholds, finds largest contour (head outline),
    samples radial distances from centroid at n_bins angles.
    Returns normalized n_bins float32 array.
    """
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] < 1:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        h, w = gray.shape
        angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        radii = []
        for angle in angles:
            # Ray-cast from centroid
            for r in np.linspace(0, max(h, w) / 2.0, 200):
                px = int(cx + r * math.cos(angle))
                py = int(cy + r * math.sin(angle))
                if 0 <= px < w and 0 <= py < h:
                    if bw[py, px] == 0:
                        radii.append(r)
                        break
            else:
                radii.append(0.0)
        out = np.array(radii, dtype=np.float32)
        norm = out.max()
        return out / norm if norm > 0 else out
    except Exception:
        return None


def compute_lbp_texture(face_bgr: np.ndarray, n_points: int = 8, radius: int = 1) -> Optional[np.ndarray]:
    """
    Local Binary Pattern on the forehead region (top 30% of face crop).
    Returns 256-bin normalized histogram.
    """
    try:
        h, w = face_bgr.shape[:2]
        forehead = face_bgr[:max(1, int(h * 0.30)), :]
        gray = cv2.cvtColor(forehead, cv2.COLOR_BGR2GRAY)
        # Manual LBP (no skimage dependency)
        fh, fw = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                lbp |= ((gray >= shifted).astype(np.uint8)) << max(0, min(7, (dy + radius) * 3 + (dx + radius)))
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        norm = hist.sum()
        return hist / norm if norm > 0 else hist
    except Exception:
        return None


def compute_hair_descriptor(face_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[bool]]:
    """
    Extracts hair region (top 20% of face crop) HSV color histogram + presence flag.
    Returns (48-bin hist [16×H + 16×S + 16×V], hair_present bool).
    """
    try:
        h, w = face_bgr.shape[:2]
        hair_region = face_bgr[:max(1, int(h * 0.20)), :]
        if hair_region.size == 0:
            return None, None
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        combined = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = combined.sum()
        hist = combined / norm if norm > 0 else combined
        # Hair presence: if mean saturation is high and value is not skin-tone
        mean_s = float(hsv[:, :, 1].mean())
        mean_v = float(hsv[:, :, 2].mean())
        hair_present = mean_s > 30 or mean_v < 120
        return hist, hair_present
    except Exception:
        return None, None


def detect_glasses(face_bgr: np.ndarray) -> Optional[bool]:
    """
    Simple glasses detector based on horizontal edge density in the eye region.
    The frame of glasses creates strong horizontal edges in the mid-face area.
    Returns True/False/None.
    """
    try:
        h, w = face_bgr.shape[:2]
        # Eye region: rows 30–55% of face height
        eye_region = face_bgr[int(h * 0.30):int(h * 0.55), :]
        if eye_region.size == 0:
            return None
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        # Horizontal Sobel to detect frame edges
        sobel_h = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_v = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        h_energy = float(np.abs(sobel_h).mean())
        v_energy = float(np.abs(sobel_v).mean())
        # Glasses: high horizontal edge energy relative to vertical
        ratio = h_energy / max(v_energy, 1.0)
        return ratio > 1.4
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
#  AUXILIARY DESCRIPTORS — TORSO TOWER
# ══════════════════════════════════════════════════════════════════

def compute_clothing_histogram(torso_bgr: np.ndarray, n_bins: int = 32) -> Optional[np.ndarray]:
    """
    3-channel HSV histogram on upper 60% of torso crop (avoids desk area).
    Returns 96-bin (3×32) normalized float32 array.
    """
    try:
        if torso_bgr is None or torso_bgr.size == 0:
            return None
        h = torso_bgr.shape[0]
        upper = torso_bgr[:max(1, int(h * 0.60)), :]
        # Exclude zero pixels (desk-masked)
        mask = (upper.sum(axis=2) > 0).astype(np.uint8)
        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        hists = []
        for ch in range(3):
            ranges = [0, 180] if ch == 0 else [0, 256]
            hist = cv2.calcHist([hsv], [ch], mask, [n_bins], ranges).flatten()
            hists.append(hist)
        combined = np.concatenate(hists).astype(np.float32)
        norm = combined.sum()
        return combined / norm if norm > 0 else combined
    except Exception:
        return None


def compute_body_proportions(torso_bgr: np.ndarray, face_bbox: tuple) -> Optional[np.ndarray]:
    """
    Extracts upper-body pose from MediaPipe Pose if available.
    Returns a 5-value normalized ratio vector:
      [shoulder_width/face_w, neck_len/face_h, shoulder_tilt,
       head_to_shoulder_ratio, left_right_symmetry]
    Falls back to simple image-based heuristics if MediaPipe unavailable.
    """
    fx, fy, fw, fh = face_bbox
    if fw < 1 or fh < 1:
        return None

    if MEDIAPIPE and _mp_pose is not None and torso_bgr is not None:
        try:
            rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)
            results = _mp_pose.process(rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                th, tw = torso_bgr.shape[:2]
                def _lm_xy(idx):
                    return lm[idx].x * tw, lm[idx].y * th
                # MediaPipe Pose landmark indices for upper body
                L_SHOULDER, R_SHOULDER = 11, 12
                L_EAR, R_EAR = 7, 8
                try:
                    ls_x, ls_y = _lm_xy(L_SHOULDER)
                    rs_x, rs_y = _lm_xy(R_SHOULDER)
                    le_x, le_y = _lm_xy(L_EAR)
                    re_x, re_y = _lm_xy(R_EAR)
                    shoulder_w = abs(rs_x - ls_x)
                    neck_y_approx = (le_y + re_y) / 2.0
                    shoulder_y_avg = (ls_y + rs_y) / 2.0
                    neck_len = abs(shoulder_y_avg - neck_y_approx)
                    shoulder_tilt = (rs_y - ls_y) / max(shoulder_w, 1.0)
                    head_w_approx = abs(re_x - le_x)
                    head_to_shoulder = head_w_approx / max(shoulder_w, 1.0)
                    symmetry = 1.0 - abs(ls_y - rs_y) / max(shoulder_w, 1.0)
                    props = np.array([
                        shoulder_w / max(fw, 1.0),
                        neck_len / max(fh, 1.0),
                        float(np.clip(shoulder_tilt, -1.0, 1.0)),
                        float(np.clip(head_to_shoulder, 0.0, 2.0)),
                        float(np.clip(symmetry, 0.0, 1.0)),
                    ], dtype=np.float32)
                    return props
                except Exception:
                    pass
        except Exception:
            pass

    # Image-based heuristic fallback
    try:
        if torso_bgr is None or torso_bgr.size == 0:
            return None
        th, tw = torso_bgr.shape[:2]
        gray = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2GRAY)
        # Horizontal profile to estimate shoulder width
        row_means = gray[:int(th * 0.3), :].mean(axis=0)
        nonzero_cols = np.where(row_means > 15)[0]
        shoulder_w = float(len(nonzero_cols))
        props = np.array([
            shoulder_w / max(tw, 1.0),
            float(th) / max(fh, 1.0),
            0.0,  # tilt unknown
            float(fw) / max(shoulder_w, 1.0),
            1.0,  # symmetry unknown
        ], dtype=np.float32)
        return props
    except Exception:
        return None


def compute_posture_keypoints(torso_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Extracts 10 upper-body landmarks from MediaPipe Pose.
    Normalises to shoulder width and outputs a 20-value vector (x,y per keypoint).
    Keypoints: nose, L/R ears, L/R shoulders, L/R elbows, L/R wrists, neck midpoint.
    Returns None if pose estimation fails.
    """
    if not MEDIAPIPE or _mp_pose is None or torso_bgr is None:
        return None
    try:
        h, w = torso_bgr.shape[:2]
        rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)
        results = _mp_pose.process(rgb)
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        # Landmark indices
        KP_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 0]  # nose,L/R ear, L/R shoulder, L/R elbow, L/R wrist, nose again
        KP_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23]  # + L hip as anchor
        pts = []
        for idx in KP_INDICES:
            pts.append([lm[idx].x * w, lm[idx].y * h])
        pts = np.array(pts, dtype=np.float32)

        # Normalize by shoulder width
        l_sh = pts[3]
        r_sh = pts[4]
        shoulder_w = max(np.linalg.norm(r_sh - l_sh), 1.0)
        shoulder_center = (l_sh + r_sh) / 2.0

        pts_norm = (pts - shoulder_center) / shoulder_w
        return pts_norm.flatten()
    except Exception:
        return None


def compute_chest_hog(torso_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    HOG descriptor on the upper-chest patch (top 40% of torso crop, center 60% width).
    Returns a compact float32 HOG feature vector.
    """
    try:
        if torso_bgr is None or torso_bgr.size == 0:
            return None
        h, w = torso_bgr.shape[:2]
        chest = torso_bgr[:max(1, int(h * 0.40)),
                          int(w * 0.20):int(w * 0.80)]
        if chest.size == 0:
            return None
        chest_resized = cv2.resize(chest, (64, 64))
        gray = cv2.cvtColor(chest_resized, cv2.COLOR_BGR2GRAY)
        # Manual HOG: 4×4 cells, 9 orientation bins
        cell_size = 16
        n_bins = 9
        rows = gray.shape[0] // cell_size
        cols = gray.shape[1] // cell_size
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hog_features = []
        for r in range(rows):
            for c in range(cols):
                cell_mag = mag[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
                cell_ang = ang[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
                bin_idx = (cell_ang / (180.0 / n_bins)).astype(int) % n_bins
                hist = np.zeros(n_bins, dtype=np.float32)
                for i in range(n_bins):
                    hist[i] = cell_mag[bin_idx == i].sum()
                norm = np.linalg.norm(hist)
                hog_features.append(hist / norm if norm > 0 else hist)
        feat = np.concatenate(hog_features)
        return feat.astype(np.float32)
    except Exception:
        return None


def detect_writing_hand(torso_bgr: np.ndarray) -> Optional[str]:
    """
    Estimates dominant writing hand from wrist movement asymmetry.
    Uses MediaPipe Pose wrist keypoints; returns "left" / "right" / None.
    """
    if not MEDIAPIPE or _mp_pose is None or torso_bgr is None:
        return None
    try:
        rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)
        results = _mp_pose.process(rgb)
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        h, w = torso_bgr.shape[:2]
        L_WRIST, R_WRIST = 15, 16
        L_SHOULDER, R_SHOULDER = 11, 12
        lw_vis = lm[L_WRIST].visibility
        rw_vis = lm[R_WRIST].visibility
        if max(lw_vis, rw_vis) < 0.5:
            return None
        # Wrist lower than shoulder = actively writing/typing
        l_wrist_y = lm[L_WRIST].y * h
        r_wrist_y = lm[R_WRIST].y * h
        l_shoulder_y = lm[L_SHOULDER].y * h
        r_shoulder_y = lm[R_SHOULDER].y * h
        l_active = l_wrist_y > l_shoulder_y and lw_vis > 0.5
        r_active = r_wrist_y > r_shoulder_y and rw_vis > 0.5
        if l_active and not r_active:
            return "left"
        if r_active and not l_active:
            return "right"
        return None
    except Exception:
        return None


def compute_bag_descriptor(
    frame_bgr: np.ndarray,
    seat_centroid: Tuple[int, int],
    face_bbox: tuple,
) -> Optional[np.ndarray]:
    """
    Samples the region adjacent to the seat (to the side and below face),
    which often contains a bag/backpack. Returns a 48-bin HSV histogram.
    """
    try:
        fx, fy, fw, fh = face_bbox
        h_frame, w_frame = frame_bgr.shape[:2]
        cx, cy = seat_centroid
        # Sample a region to the left or right of the seat, below the torso
        bag_w = int(fw * 1.5)
        bag_h = int(fh * 2.0)
        # Try left side first
        bx1 = max(0, cx - fw * 2)
        by1 = max(0, fy + fh)
        bx2 = min(w_frame, bx1 + bag_w)
        by2 = min(h_frame, by1 + bag_h)
        if bx2 - bx1 < 10 or by2 - by1 < 10:
            return None
        bag_crop = frame_bgr[by1:by2, bx1:bx2]
        hsv = cv2.cvtColor(bag_crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        combined = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = combined.sum()
        return combined / norm if norm > 0 else combined
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
#  FACE DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_faces(img_bgr: np.ndarray) -> list:
    results = []

    if INSIGHTFACE:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            faces = _insight_app.get(rgb)
            for f in faces:
                b = f.bbox.astype(int)
                x1 = max(0, b[0]); y1 = max(0, b[1])
                x2 = min(img_bgr.shape[1], b[2]); y2 = min(img_bgr.shape[0], b[3])
                w, h = x2 - x1, y2 - y1
                if w < 8 or h < 8:
                    continue
                results.append({
                    "box":        (x1, y1, w, h),
                    "crop":       img_bgr[y1:y2, x1:x2].copy(),
                    "landmarks":  f.kps.tolist() if f.kps is not None else None,
                    "confidence": float(f.det_score),
                    "embedding":  f.embedding,
                })
            if results:
                return results
        except Exception:
            pass

    if MTCNN and _mtcnn_det:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            dets = _mtcnn_det.detect_faces(rgb)
            for d in dets:
                if d["confidence"] < 0.70:
                    continue
                x, y, w, h = d["box"]
                x = max(0, x); y = max(0, y)
                w = min(w, img_bgr.shape[1] - x); h = min(h, img_bgr.shape[0] - y)
                if w < 8 or h < 8:
                    continue
                kp = d.get("keypoints", {})
                lmk = ([
                    list(kp.get("left_eye",    [0, 0])),
                    list(kp.get("right_eye",   [0, 0])),
                    list(kp.get("nose",        [0, 0])),
                    list(kp.get("mouth_left",  [0, 0])),
                    list(kp.get("mouth_right", [0, 0])),
                ] if kp else None)
                results.append({
                    "box": (x, y, w, h), "crop": img_bgr[y:y+h, x:x+w].copy(),
                    "landmarks": lmk, "confidence": d["confidence"], "embedding": None,
                })
            if results:
                return results
        except Exception:
            pass

    for xml, sf, mn in [
        ("haarcascade_frontalface_default.xml", 1.05, 4),
        ("haarcascade_frontalface_alt2.xml",    1.04, 3),
    ]:
        cas = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
        if cas.empty():
            continue
        h, w = img_bgr.shape[:2]
        small = (cv2.resize(img_bgr, (int(w * 360 / h), 360), interpolation=cv2.INTER_AREA)
                 if h > 360 else img_bgr)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        scale = h / small.shape[0]
        dets_small = cas.detectMultiScale(gray, sf, mn, minSize=(12, 12))
        dets = ([(int(x * scale), int(y * scale), int(w2 * scale), int(h2 * scale))
                 for x, y, w2, h2 in dets_small]
                if len(dets_small) else [])
        for x, y, w, h in dets:
            x = max(0, x); y = max(0, y)
            w = min(w, img_bgr.shape[1] - x); h = min(h, img_bgr.shape[0] - y)
            if w < 8 or h < 8:
                continue
            results.append({
                "box": (x, y, w, h), "crop": img_bgr[y:y+h, x:x+w].copy(),
                "landmarks": None, "confidence": 0.40, "embedding": None,
            })
        if results:
            return results

    return results


def deduplicate_faces(faces: list, iou_thresh: float = 0.35) -> list:
    if not faces:
        return []
    faces = sorted(faces, key=lambda f: -f["confidence"])
    kept = []
    for f in faces:
        x, y, w, h = f["box"]
        dup = False
        for k in kept:
            kx, ky, kw, kh = k["box"]
            ix1 = max(x, kx); iy1 = max(y, ky)
            ix2 = min(x + w, kx + kw); iy2 = min(y + h, ky + kh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = w * h + kw * kh - inter
                if inter / max(union, 1) > iou_thresh:
                    dup = True
                    break
        if not dup:
            kept.append(f)
    return kept


# ══════════════════════════════════════════════════════════════════
#  EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def extract_face_embedding(face_bgr: np.ndarray, precomputed_insight=None) -> Optional[np.ndarray]:
    """512-d L2-normalised face embedding (AdaFace → InsightFace → histogram)."""
    if ADAFACE and _adaface_model is not None:
        try:
            import torch
            import torch.nn.functional as TF
            face_up = upscale_face(face_bgr, 112)
            face = cv2.resize(face_up, (112, 112)).astype(np.float32)
            face = (face / 255.0 - 0.5) / 0.5
            face = face[:, :, ::-1].copy()
            tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)
            if next(_adaface_model.parameters()).is_cuda:
                tensor = tensor.cuda()
            with torch.no_grad():
                emb, _ = _adaface_model(tensor)
            import torch.nn.functional as TF
            emb = TF.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy().flatten().astype(np.float64)
        except Exception:
            pass

    if INSIGHTFACE:
        if precomputed_insight is not None:
            emb = np.array(precomputed_insight, dtype=np.float64).flatten()
            if emb.size > 0:
                return _l2_norm(emb)
        try:
            face_up = upscale_face(face_bgr, 112)
            rgb = cv2.cvtColor(face_up, cv2.COLOR_BGR2RGB)
            faces = _insight_app.get(rgb)
            if faces and faces[0].embedding is not None:
                return _l2_norm(np.array(faces[0].embedding, dtype=np.float64).flatten())
        except Exception:
            pass

    # Histogram fallback
    try:
        face_r = cv2.resize(face_bgr, (64, 64))
        gray   = cv2.cvtColor(face_r, cv2.COLOR_BGR2GRAY)
        gx     = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy     = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        _, ang = cv2.cartToPolar(gx, gy)
        hist_g = cv2.calcHist([ang], [0], None, [64], [0, 2 * np.pi]).flatten()
        hists  = [hist_g]
        for ch in cv2.split(face_r):
            hists.append(cv2.calcHist([ch], [0], None, [32], [0, 256]).flatten())
        return _l2_norm(np.concatenate(hists).astype(np.float64))
    except Exception:
        pass
    return None


# Alias for backwards compatibility
extract_embedding = extract_face_embedding


def extract_torso_embedding(torso_bgr: np.ndarray) -> Optional[np.ndarray]:
    """512-d L2-normalised torso embedding (OSNet → CLIP → HOG fallback)."""
    if OSNET and _osnet_model is not None:
        try:
            import torch
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0)
            if next(_osnet_model.parameters()).is_cuda:
                tensor = tensor.cuda()
            with torch.no_grad():
                feat = _osnet_model(tensor)
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            emb = feat.cpu().numpy().flatten().astype(np.float64)
            # Pad or trim to 512
            if emb.shape[0] > 512:
                emb = emb[:512]
            elif emb.shape[0] < 512:
                emb = np.pad(emb, (0, 512 - emb.shape[0]))
            return _l2_norm(emb)
        except Exception as _e:
            logger.warning(f"OSNet inference failed ({_e})")

    if CLIP_REID and _clip_model is not None:
        try:
            import torch
            from PIL import Image as _PILImage
            rgb = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB)
            pil_img = _PILImage.fromarray(rgb)
            inp = _clip_preprocess(pil_img).unsqueeze(0)
            with torch.no_grad():
                feat = _clip_model.encode_image(inp)
            emb = feat.cpu().numpy().flatten().astype(np.float64)
            # CLIP ViT-B/32 → 512-d already
            return _l2_norm(emb)
        except Exception as _e:
            logger.warning(f"CLIP inference failed ({_e})")

    # HOG fallback: compute HOG on resized torso
    try:
        torso_r = cv2.resize(torso_bgr, (128, 256))
        gray    = cv2.cvtColor(torso_r, cv2.COLOR_BGR2GRAY)
        gx      = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy      = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # 8×4 cells, 9 bins
        cell_h, cell_w = gray.shape[0] // 8, gray.shape[1] // 4
        feats = []
        for r in range(8):
            for c in range(4):
                cell_mag = mag[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                cell_ang = ang[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                bin_idx = (cell_ang / 20.0).astype(int) % 9
                hist = np.zeros(9, dtype=np.float32)
                for i in range(9):
                    hist[i] = cell_mag[bin_idx == i].sum()
                n = np.linalg.norm(hist)
                feats.append(hist / n if n > 0 else hist)
        hog_feat = np.concatenate(feats).astype(np.float64)
        # 8×4×9 = 288 dims; pad to 512
        if hog_feat.shape[0] < 512:
            hog_feat = np.pad(hog_feat, (0, 512 - hog_feat.shape[0]))
        return _l2_norm(hog_feat[:512])
    except Exception:
        pass
    return None


def fuse_embeddings(
    face_emb: Optional[np.ndarray],
    torso_emb: Optional[np.ndarray],
    face_conf: float,
    torso_conf: float,
    alpha: float,
    beta: float,
) -> Optional[np.ndarray]:
    """
    Confidence-weighted late fusion of face + torso embeddings.
    Output: L2-normalised 1024-d vector (face 512 || torso 512).
    Hard routing:
      - face_conf < 0.2 → torso only (zeroed face half)
      - torso_conf < 0.2 → face only (zeroed torso half)
      - both poor → None
    """
    face_ok  = face_emb is not None and face_conf >= 0.2
    torso_ok = torso_emb is not None and torso_conf >= 0.2

    if not face_ok and not torso_ok:
        return None

    dim = 512
    face_part  = np.zeros(dim, dtype=np.float64)
    torso_part = np.zeros(dim, dtype=np.float64)

    if face_ok:
        fe = face_emb.flatten()[:dim]
        if fe.shape[0] < dim:
            fe = np.pad(fe, (0, dim - fe.shape[0]))
        face_part = fe * face_conf * alpha

    if torso_ok:
        te = torso_emb.flatten()[:dim]
        if te.shape[0] < dim:
            te = np.pad(te, (0, dim - te.shape[0]))
        torso_part = te * torso_conf * beta

    fused = np.concatenate([face_part, torso_part])
    return _l2_norm(fused.astype(np.float64))


# ══════════════════════════════════════════════════════════════════
#  PER-SESSION FUSION CALIBRATION
# ══════════════════════════════════════════════════════════════════

def calibrate_fusion(
    calibration_detections: List[FaceDetection],
) -> Tuple[float, float]:
    """
    Calibrates the α/β fusion weights using the first N seconds of detections.
    Measures within-seat intra-class cosine distance for face-only vs torso-only
    embeddings. Weights the tower with tighter within-seat consistency more.

    Returns (alpha, beta) normalised to sum to 1.0.
    """
    if not calibration_detections:
        return config.face_alpha, config.torso_beta

    seat_groups: Dict[str, List[FaceDetection]] = defaultdict(list)
    for det in calibration_detections:
        if det.seat_id is not None:
            seat_groups[det.seat_id].append(det)

    # Need at least some seats with ≥2 detections
    valid_seats = {sid: dets for sid, dets in seat_groups.items() if len(dets) >= 2}
    if not valid_seats:
        logger.info("Fusion calibration: insufficient multi-detection seats — using defaults")
        return config.face_alpha, config.torso_beta

    face_distances, torso_distances = [], []

    for sid, dets in valid_seats.items():
        face_embs  = [d.face_embedding  for d in dets if d.face_embedding  is not None]
        torso_embs = [d.torso_embedding for d in dets if d.torso_embedding is not None]

        def _mean_intra_dist(embs):
            if len(embs) < 2:
                return None
            mat = np.stack(embs)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat_n = mat / np.maximum(norms, 1e-10)
            sim = mat_n @ mat_n.T
            n = len(embs)
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            dists = [1.0 - float(sim[i, j]) for i, j in pairs]
            return float(np.mean(dists))

        fd = _mean_intra_dist(face_embs)
        td = _mean_intra_dist(torso_embs)
        if fd is not None:
            face_distances.append(fd)
        if td is not None:
            torso_distances.append(td)

    if not face_distances or not torso_distances:
        return config.face_alpha, config.torso_beta

    mean_face_dist  = float(np.mean(face_distances))
    mean_torso_dist = float(np.mean(torso_distances))

    # Tighter (smaller) distance → more reliable → higher weight
    # Weight inversely proportional to mean intra-class distance
    face_reliability  = 1.0 / max(mean_face_dist, 1e-4)
    torso_reliability = 1.0 / max(mean_torso_dist, 1e-4)
    total = face_reliability + torso_reliability
    alpha = float(face_reliability  / total)
    beta  = float(torso_reliability / total)

    logger.info(
        f"Fusion calibrated: face_dist={mean_face_dist:.4f} "
        f"torso_dist={mean_torso_dist:.4f} "
        f"→ α={alpha:.3f}  β={beta:.3f}"
    )
    return alpha, beta


# ══════════════════════════════════════════════════════════════════
#  BATCH EMBEDDING EXTRACTION (AdaFace — batched; others — per-image)
# ══════════════════════════════════════════════════════════════════

def extract_embeddings_batch(
    face_list: List[np.ndarray],
) -> List[Optional[np.ndarray]]:
    """
    Batch face embedding extraction.
    AdaFace: true batched forward pass.
    Others: per-image loop.
    """
    if not face_list:
        return []

    results: List[Optional[np.ndarray]] = [None] * len(face_list)

    if ADAFACE and _adaface_model is not None:
        try:
            import torch
            import torch.nn.functional as TF

            def _prep(f: np.ndarray):
                f2 = cv2.resize(upscale_face(f, 112), (112, 112)).astype(np.float32)
                f2 = (f2 / 255.0 - 0.5) / 0.5
                f2 = f2[:, :, ::-1].copy()
                return torch.from_numpy(f2).permute(2, 0, 1)

            batch_size = config.embed_batch_size
            for start in range(0, len(face_list), batch_size):
                batch_faces = face_list[start: start + batch_size]
                try:
                    tensors = torch.stack([_prep(f) for f in batch_faces])
                    if next(_adaface_model.parameters()).is_cuda:
                        tensors = tensors.cuda()
                    with torch.no_grad():
                        embs, _ = _adaface_model(tensors)
                    embs = TF.normalize(embs, p=2, dim=1).cpu().numpy()
                    for i, emb in enumerate(embs):
                        results[start + i] = emb.flatten().astype(np.float64)
                except Exception as _be:
                    logger.warning(
                        f"AdaFace batch [{start}:{start + len(batch_faces)}] "
                        f"failed ({_be}), retrying per-image"
                    )
                    for i, face in enumerate(batch_faces):
                        results[start + i] = extract_face_embedding(face)
            return results
        except Exception as _e:
            logger.warning(f"extract_embeddings_batch AdaFace failed ({_e}), per-image")
            return [extract_face_embedding(f) for f in face_list]

    return [extract_face_embedding(f) for f in face_list]


# ══════════════════════════════════════════════════════════════════
#  FULL PER-DETECTION FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_full_features(
    frame_bgr: np.ndarray,
    face_info: dict,
    frame_idx: int,
    fps: float,
    face_crop: np.ndarray,
    session: SessionState,
) -> Optional[FaceDetection]:
    """
    Extracts ALL features for a single face detection:
      - Assigns seat from seat map
      - Extracts torso ROI
      - Computes face + torso embeddings
      - Computes all auxiliary descriptors
      - Fuses embeddings with session-calibrated α/β
      - Returns a fully-populated FaceDetection (or None if both towers fail)
    """
    fx, fy, fw, fh = face_info["box"]
    face_cx = fx + fw // 2
    face_cy = fy + fh // 2

    # ── Seat assignment ───────────────────────────────────────────
    seat_id = assign_seat(face_cx, face_cy, session.seat_map)
    seat_entry = session.seat_map.get(seat_id) if seat_id else None
    seat_centroid = seat_entry.centroid_px if seat_entry else (face_cx, face_cy)

    # ── Face crop + embedding ────────────────────────────────────
    if face_info["landmarks"]:
        aligned_crop = align_face_affine(frame_bgr, face_info["landmarks"])
    else:
        aligned_crop = face_crop
    aligned_crop = super_resolve_face(aligned_crop, 112)

    quality = face_quality_score(aligned_crop)
    face_emb = extract_face_embedding(
        aligned_crop,
        precomputed_insight=face_info.get("embedding"),
    )

    # Normalised face confidence 0–1
    raw_q = quality
    face_conf_norm = float(np.clip(raw_q / 200.0, 0.0, 1.0))  # 200 ~ typical good score

    # ── Torso ROI + embedding ────────────────────────────────────
    torso_crop = extract_torso_roi(frame_bgr, face_info["box"], desk_mask=session.desk_mask)
    torso_emb  = None
    torso_q    = 0.0
    torso_conf_norm = 0.0

    if torso_crop is not None:
        torso_q = torso_quality_score(torso_crop)
        if torso_q >= config.min_torso_quality:
            torso_emb = extract_torso_embedding(torso_crop)
            torso_conf_norm = float(np.clip(torso_q / 500.0, 0.0, 1.0))

    # ── Fused embedding ──────────────────────────────────────────
    fused = fuse_embeddings(
        face_emb, torso_emb,
        face_conf_norm, torso_conf_norm,
        session.fusion_alpha, session.fusion_beta,
    )
    if fused is None:
        return None

    # ── Auxiliary face descriptors ───────────────────────────────
    gaze_vec      = compute_gaze_vector(aligned_crop) if MEDIAPIPE else None
    head_silh     = compute_head_silhouette(aligned_crop)
    lbp_tex       = compute_lbp_texture(aligned_crop)
    hair_hist, hair_pres = compute_hair_descriptor(aligned_crop)
    glasses_flag  = detect_glasses(aligned_crop)

    # ── Auxiliary torso descriptors ──────────────────────────────
    clothing_hist  = None
    body_props     = None
    posture_kps    = None
    chest_hog_feat = None
    writing_hand   = None
    if torso_crop is not None and torso_q >= config.min_torso_quality:
        clothing_hist  = compute_clothing_histogram(torso_crop)
        body_props     = compute_body_proportions(torso_crop, face_info["box"])
        posture_kps    = compute_posture_keypoints(torso_crop)
        chest_hog_feat = compute_chest_hog(torso_crop)
        writing_hand   = detect_writing_hand(torso_crop)

    # ── Bag/backpack descriptor (seat-anchored) ──────────────────
    bag_hist = None
    if seat_centroid is not None:
        bag_hist = compute_bag_descriptor(frame_bgr, seat_centroid, face_info["box"])

    # ── Illumination score ───────────────────────────────────────
    illum_sc = compute_illumination_score(frame_bgr, session.illum_baseline)

    return FaceDetection(
        frame_idx=frame_idx,
        timestamp_sec=round(frame_idx / fps, 3),
        embedding=fused,
        crop_bgr=aligned_crop,
        quality_score=quality,
        bbox=face_info["box"],
        confidence=face_info["confidence"],
        landmarks=face_info.get("landmarks"),
        face_embedding=face_emb,
        torso_embedding=torso_emb,
        torso_crop_bgr=torso_crop,
        torso_quality=torso_q,
        face_conf=face_conf_norm,
        torso_conf=torso_conf_norm,
        seat_id=seat_id,
        seat_centroid_px=seat_centroid,
        gaze_vector=gaze_vec,
        head_silhouette=head_silh,
        lbp_texture=lbp_tex,
        hair_color_hist=hair_hist,
        hair_present=hair_pres,
        has_glasses=glasses_flag,
        clothing_hist=clothing_hist,
        body_proportions=body_props,
        posture_keypoints=posture_kps,
        chest_hog=chest_hog_feat,
        writing_hand=writing_hand,
        bag_color_hist=bag_hist,
        illum_score=illum_sc,
    )


# ══════════════════════════════════════════════════════════════════
#  STEP 1 — FRAME SAMPLING + DETECTION
# ══════════════════════════════════════════════════════════════════

def track_and_sample(
    video_path: Path,
    budget: int,
    session: SessionState,
) -> List[FaceDetection]:
    """ByteTrack-based sampling. Returns [] on any error."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step  = max(1, int(fps / config.tracking_fps))
        tracker = _ByteTrackCls()

        track_crops: Dict[int, list] = {}
        idx = 0
        while idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if h > 480:
                frame = cv2.resize(frame, (int(w * 480 / h), 480), interpolation=cv2.INTER_AREA)
            frame_enh = enhance_frame(frame)
            raw_faces = deduplicate_faces(detect_faces(frame_enh))

            if raw_faces:
                dets_arr = np.array(
                    [[f["box"][0], f["box"][1],
                      f["box"][0] + f["box"][2],
                      f["box"][1] + f["box"][3],
                      f["confidence"], 0]
                     for f in raw_faces], dtype=np.float32,
                )
                tracks = tracker.update(dets_arr, frame_enh)
                for track in tracks:
                    x1, y1 = int(track[0]), int(track[1])
                    x2, y2 = int(track[2]), int(track[3])
                    tid    = int(track[4])
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame_enh.shape[1], x2)
                    y2 = min(frame_enh.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame_enh[y1:y2, x1:x2].copy()
                    crop = super_resolve_face(crop, 112)
                    q    = face_quality_score(crop)
                    # Build minimal face_info for full extraction
                    face_info = {
                        "box": (x1, y1, x2 - x1, y2 - y1),
                        "crop": crop,
                        "landmarks": None,
                        "confidence": float(track[5]) if len(track) > 5 else 0.9,
                        "embedding": None,
                    }
                    fd = extract_full_features(
                        frame_enh, face_info, idx, fps, crop, session
                    )
                    if fd is not None:
                        track_crops.setdefault(tid, []).append((q, fd))

            del frame, frame_enh
            idx += step

        cap.release()

        kept: List[FaceDetection] = []
        for tid, items in track_crops.items():
            items.sort(key=lambda x: -x[0])
            for q, fd in items[:3]:
                if q >= config.min_face_quality:
                    kept.append(fd)

        logger.info(f"ByteTrack: {len(track_crops)} tracks → {len(kept)} kept detections")
        return kept

    except Exception as _e:
        logger.warning(f"track_and_sample failed ({_e}) — falling back to temporal sampling")
        return []


def sample_and_detect(video_path: str | Path, session: SessionState) -> List[FaceDetection]:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_s   = total_frames / fps
    step         = max(1, int(fps * config.sample_interval_sec))

    logger.info(f"Video : {video_path.name}")
    logger.info(f"Frames: {total_frames}   FPS: {fps:.1f}   Duration: {duration_s:.1f}s")

    budget = _compute_sampling_budget(duration_s)
    logger.info(f"Sampling budget: {budget} frames  (duration={duration_s / 60:.1f} min)")

    # ── Scene initialisation (first frame) ───────────────────────
    ret, init_frame = cap.read()
    if ret:
        h, w = init_frame.shape[:2]
        if h > 480:
            init_frame = cv2.resize(init_frame, (int(w * 480 / h), 480), interpolation=cv2.INTER_AREA)
        logger.info("INIT — Building seat map from first frame")
        session.seat_map  = build_seat_map(init_frame)
        session.homography = estimate_homography(session.seat_map, init_frame.shape)
        session.desk_mask = build_desk_occlusion_mask(init_frame, session.seat_map)
        # Set illumination baseline
        lab = cv2.cvtColor(init_frame, cv2.COLOR_BGR2LAB)
        session.illum_baseline = lab[:, :, 0].astype(np.float32)
        if session.homography is not None:
            logger.info("Ground-plane homography calibrated")

    # ── ByteTrack path ────────────────────────────────────────────
    if BYTETRACK:
        cap.release()
        bt_result = track_and_sample(video_path, budget, session)
        if bt_result:
            logger.info(f"Total face detections collected: {len(bt_result)}")
            _run_fusion_calibration(bt_result, session, duration_s, fps)
            return bt_result
        logger.warning("ByteTrack returned empty — falling back to temporal sampling")
        cap = cv2.VideoCapture(str(video_path))

    # ── Temporal sampling path ────────────────────────────────────
    candidates = []
    idx = 0
    while idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if h > 480:
            frame = cv2.resize(frame, (int(w * 480 / h), 480), interpolation=cv2.INTER_AREA)
        candidates.append((idx, frame, laplacian_sharpness(frame)))
        idx += step
    cap.release()

    selected = _select_frames_temporal(candidates, budget)
    logger.info(f"Sampled {len(candidates)} candidates → kept {len(selected)} (temporal spread)")

    calibration_cutoff_frame = int(config.calibration_window_sec * fps)
    calibration_detections: List[FaceDetection] = []

    all_detections: List[FaceDetection] = []
    for processed, (frame_idx, frame, _) in enumerate(selected):
        h, w = frame.shape[:2]
        if h > 480:
            frame = cv2.resize(frame, (int(w * 480 / h), 480), interpolation=cv2.INTER_AREA)
        frame_enh = enhance_frame(frame)
        del frame
        faces = deduplicate_faces(detect_faces(frame_enh))

        for face_info in faces:
            crop = face_info["crop"]
            # Pre-quality check before expensive extraction
            pre_q = face_quality_score(super_resolve_face(crop, 112))
            if pre_q < config.min_face_quality:
                continue

            fd = extract_full_features(frame_enh, face_info, frame_idx, fps, crop, session)
            if fd is None:
                continue
            all_detections.append(fd)
            if frame_idx <= calibration_cutoff_frame:
                calibration_detections.append(fd)

        del frame_enh, faces

        if (processed + 1) % 10 == 0:
            gc.collect()
            logger.info(
                f"Processed {processed + 1}/{len(selected)} frames "
                f"| Detections so far: {len(all_detections)}"
            )

    # ── Per-session fusion calibration ───────────────────────────
    _run_fusion_calibration(calibration_detections, session, duration_s, fps)

    # ── Re-embed with calibrated α/β if needed ───────────────────
    if session.calibration_done and config.calibrate_fusion:
        _reembed_with_calibrated_fusion(all_detections, session)

    logger.info(f"Total face detections collected: {len(all_detections)}")
    return all_detections


def _run_fusion_calibration(
    detections: List[FaceDetection],
    session: SessionState,
    duration_s: float,
    fps: float,
) -> None:
    """Calibrate session fusion weights if enabled and not already done."""
    if session.calibration_done or not config.calibrate_fusion:
        return
    if not detections:
        return
    calib_dets = [d for d in detections
                  if d.timestamp_sec <= config.calibration_window_sec]
    if not calib_dets:
        calib_dets = detections  # use everything if session is shorter than window
    alpha, beta = calibrate_fusion(calib_dets)
    session.fusion_alpha = alpha
    session.fusion_beta  = beta
    session.calibration_done = True
    logger.info(f"Session fusion weights set: α={alpha:.3f}  β={beta:.3f}")


def _reembed_with_calibrated_fusion(
    detections: List[FaceDetection],
    session: SessionState,
) -> None:
    """Re-fuse embeddings with calibrated α/β weights in-place."""
    updated = 0
    for det in detections:
        new_fused = fuse_embeddings(
            det.face_embedding, det.torso_embedding,
            det.face_conf, det.torso_conf,
            session.fusion_alpha, session.fusion_beta,
        )
        if new_fused is not None:
            det.embedding = new_fused
            updated += 1
    if updated:
        logger.info(f"Re-fused {updated} embeddings with calibrated α/β")


# ══════════════════════════════════════════════════════════════════
#  SEAT-ANCHORED TRACKLET MANAGEMENT
# ══════════════════════════════════════════════════════════════════

def build_seat_tracklets(
    detections: List[FaceDetection],
    session: SessionState,
) -> Dict[str, List[FaceDetection]]:
    """
    Groups detections into seat-anchored tracklets.
    Applies per-tracklet outlier rejection: any detection whose embedding
    is > tracklet_outlier_dist from the running tracklet mean is discarded.
    """
    tracklets: Dict[str, List[FaceDetection]] = defaultdict(list)
    tracklet_means: Dict[str, np.ndarray] = {}

    for det in detections:
        sid = det.seat_id if det.seat_id else f"noseat_{det.bbox[0]}_{det.bbox[1]}"
        existing = tracklets[sid]
        if not existing:
            tracklets[sid].append(det)
            tracklet_means[sid] = det.embedding.copy()
        else:
            mean = tracklet_means[sid]
            dist = 1.0 - float(np.dot(det.embedding, mean) /
                               max(np.linalg.norm(det.embedding) * np.linalg.norm(mean), 1e-10))
            if dist <= config.tracklet_outlier_dist:
                tracklets[sid].append(det)
                # Update running mean
                n = len(tracklets[sid])
                tracklet_means[sid] = _l2_norm(
                    mean * (n - 1) / n + det.embedding / n
                )
            else:
                logger.debug(f"Seat {sid}: outlier rejected (dist={dist:.3f})")

    logger.info(
        f"Seat tracklets: {len(tracklets)} seats, "
        f"{sum(len(v) for v in tracklets.values())} kept detections"
    )
    session.seat_tracklets = tracklets
    return tracklets


# ══════════════════════════════════════════════════════════════════
#  STEP 2 — FACE CLUSTERING
# ══════════════════════════════════════════════════════════════════

def _build_knn_graph(
    embs_n: np.ndarray,
    k: int,
    sim_threshold: float,
) -> "nx.Graph":
    import faiss

    n, d   = embs_n.shape
    embs_f = np.ascontiguousarray(embs_n, dtype=np.float32)
    index  = faiss.IndexFlatIP(d)
    index.add(embs_f)

    real_k = min(k + 1, n)
    sims, idxs = index.search(embs_f, real_k)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j_pos in range(real_k):
            j   = int(idxs[i, j_pos])
            sim = float(sims[i, j_pos])
            if j == i:
                continue
            if sim > sim_threshold:
                if G.has_edge(i, j):
                    if G[i][j]["weight"] < sim:
                        G[i][j]["weight"] = sim
                else:
                    G.add_edge(i, j, weight=sim)
    return G


def _chinese_whispers(G: "nx.Graph", iterations: int = 20) -> dict:
    import random
    labels: dict = {node: node for node in G.nodes()}
    for _ in range(iterations):
        nodes = list(G.nodes())
        random.shuffle(nodes)
        changed = False
        for node in nodes:
            neighbours = list(G.neighbors(node))
            if not neighbours:
                continue
            votes: dict = {}
            for nb in neighbours:
                lbl = labels[nb]
                w   = G[node][nb].get("weight", 1.0)
                votes[lbl] = votes.get(lbl, 0.0) + w
            best_label = max(votes, key=votes.__getitem__)
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True
        if not changed:
            break
    return labels


def _rescue_noise(
    detections: List[FaceDetection],
    labels: np.ndarray,
    dist_matrix: np.ndarray,
    rescue_dist: float,
) -> np.ndarray:
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels

    labels      = labels.copy()
    real_labels = sorted(set(labels) - {-1})
    if not real_labels:
        return labels

    emb_matrix = np.stack([d.embedding for d in detections])
    centroids: dict = {}
    for rl in real_labels:
        members = emb_matrix[labels == rl]
        c = members.mean(axis=0)
        n = np.linalg.norm(c)
        centroids[rl] = c / n if n > 0 else c

    rescued = 0
    for i in np.where(noise_mask)[0]:
        best_label, best_dist = -1, float("inf")
        for rl, centroid in centroids.items():
            d = 1.0 - float(np.dot(detections[i].embedding, centroid))
            if d < best_dist:
                best_dist = d
                best_label = rl
        if best_dist <= rescue_dist:
            labels[i] = best_label
            rescued += 1

    n_noise = int(noise_mask.sum())
    if rescued > 0:
        logger.info(f"Rescued {rescued}/{n_noise} noise points into existing clusters")
    return labels


def _stage2_merge_subclusters(
    labels: np.ndarray,
    embs_n: np.ndarray,
    n_target: Optional[int],
) -> np.ndarray:
    from scipy.cluster.hierarchy import linkage as _sp_linkage, fcluster as _sp_fcluster
    from scipy.spatial.distance import squareform as _sp_squareform
    from sklearn.cluster import AgglomerativeClustering

    TRUST_THRESHOLD  = 15
    GAP_RATIO_THRESH = 0.30

    real_labels = sorted(set(labels) - {-1})
    n_sub = len(real_labels)
    if n_sub <= 1:
        return labels

    centroids = []
    for rl in real_labels:
        members = embs_n[labels == rl]
        c = members.mean(axis=0)
        nc = np.linalg.norm(c)
        centroids.append(c / nc if nc > 0 else c)
    cent_arr  = np.stack(centroids)
    cent_dist = np.maximum(1.0 - np.clip(cent_arr @ cent_arr.T, -1.0, 1.0), 0.0)
    np.fill_diagonal(cent_dist, 0.0)

    if n_target is not None:
        n_clusters = min(n_target, n_sub)
        if n_clusters >= n_sub:
            logger.info(f"Stage 2: {n_sub} sub-clusters == target ({n_target}), no merge needed")
            return labels
        logger.info(f"Stage 2: merging {n_sub} sub-clusters → {n_clusters} (KNOWN_N_PERSONS={n_target})")
        agg = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="complete")
        meta_labels   = agg.fit_predict(cent_dist)
        label_to_meta = {rl: int(meta_labels[i]) for i, rl in enumerate(real_labels)}
        new_labels    = labels.copy()
        for i, orig in enumerate(labels):
            if orig in label_to_meta:
                new_labels[i] = label_to_meta[orig]
        return new_labels

    upper = cent_dist[np.triu_indices(n_sub, k=1)]
    if len(upper) == 0:
        return labels

    sorted_dists = np.sort(upper)
    gaps         = np.diff(sorted_dists) if len(sorted_dists) > 1 else np.array([0.0])
    max_gap      = float(gaps.max())
    gap_ratio    = max_gap / max(float(sorted_dists.mean()), 1e-6)

    logger.info(
        f"Stage 2 (auto): {n_sub} sub-clusters  "
        f"dist=[{sorted_dists[0]:.3f}..{sorted_dists[-1]:.3f}]  "
        f"gap_ratio={gap_ratio:.3f}"
    )

    if n_sub <= TRUST_THRESHOLD and gap_ratio < GAP_RATIO_THRESH:
        logger.info(f"Stage 2: no bimodal gap detected → trusting {n_sub} sub-clusters")
        label_remap = {rl: i for i, rl in enumerate(real_labels)}
        new_labels  = labels.copy()
        for i, orig in enumerate(labels):
            if orig in label_remap:
                new_labels[i] = label_remap[orig]
        return new_labels

    gap_idx   = int(np.argmax(gaps))
    threshold = (sorted_dists[gap_idx] + sorted_dists[gap_idx + 1]) / 2.0
    logger.info(
        f"Stage 2: gap at {sorted_dists[gap_idx]:.3f}→{sorted_dists[gap_idx+1]:.3f}, "
        f"threshold={threshold:.3f}"
    )

    condensed  = _sp_squareform(cent_dist, checks=False)
    Z          = _sp_linkage(condensed, method="complete")
    meta_array = _sp_fcluster(Z, t=threshold, criterion="distance")
    n_clusters = max(2, len(set(meta_array)))
    logger.info(f"Stage 2: merging {n_sub} sub-clusters → {n_clusters} (auto, complete linkage)")

    label_to_meta = {rl: int(meta_array[i]) - 1 for i, rl in enumerate(real_labels)}
    new_labels    = labels.copy()
    for i, orig in enumerate(labels):
        if orig in label_to_meta:
            new_labels[i] = label_to_meta[orig]
    return new_labels


# ══════════════════════════════════════════════════════════════════
#  SEAT-PRIOR BAYESIAN POST-PROCESSING
# ══════════════════════════════════════════════════════════════════

def _seat_prior_correction(
    clusters: List[ClusterResult],
    session: SessionState,
) -> List[ClusterResult]:
    """
    Post-processing pass that uses seat geometry to correct clustering errors.

    Pass 1 — Fragmentation detection:
      If all detections from a given seat belong to 2+ different clusters
      AND there's no temporal gap (no identity swap), merge them.

    Pass 2 — Identity swap detection:
      If a cluster has detections from two seats with a large temporal gap
      between the seat occupancies, flag as potential swap (log warning).

    Pass 3 — Seat consistency:
      If a cluster has detections from seats that are geographically far apart
      (> diff_seat_dist_min_px), flag as likely merge error (log warning).
    """
    if not session.seat_map:
        return clusters

    # Build seat → cluster mapping
    seat_to_clusters: Dict[str, set] = defaultdict(set)
    cluster_to_seats: Dict[int, list] = defaultdict(list)
    for c in clusters:
        if c.id == -1:
            continue
        for det in c.faces:
            if det.seat_id:
                seat_to_clusters[det.seat_id].add(c.id)
                cluster_to_seats[c.id].append(det.seat_id)

    # Pass 1: fragmentation — same seat, multiple clusters, no temporal gap
    merge_pairs: List[Tuple[int, int]] = []
    for seat_id, cluster_ids in seat_to_clusters.items():
        if len(cluster_ids) <= 1:
            continue
        cluster_ids = sorted(cluster_ids)
        # Check for temporal gap between the clusters at this seat
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                ci_id = cluster_ids[i]
                cj_id = cluster_ids[j]
                ci = next((c for c in clusters if c.id == ci_id), None)
                cj = next((c for c in clusters if c.id == cj_id), None)
                if ci is None or cj is None:
                    continue
                # Check if time ranges overlap or are adjacent (< 5 min gap = same person)
                ci_times = [f.timestamp_sec for f in ci.faces if f.seat_id == seat_id]
                cj_times = [f.timestamp_sec for f in cj.faces if f.seat_id == seat_id]
                if not ci_times or not cj_times:
                    continue
                gap = min(
                    abs(max(ci_times) - min(cj_times)),
                    abs(max(cj_times) - min(ci_times)),
                )
                if gap < 600.0:  # < 5 minutes = likely fragmentation
                    merge_pairs.append((ci_id, cj_id))
                    logger.info(
                        f"Seat-prior: seat {seat_id} → merging clusters "
                        f"{ci_id} + {cj_id} (gap={gap:.1f}s < 300s)"
                    )

    # Apply merges (union-find style)
    if merge_pairs:
        clusters = _apply_cluster_merges(clusters, merge_pairs)

    # Pass 3: geographic consistency check (warning only, no auto-merge)
    for c in clusters:
        if c.id == -1 or not c.faces:
            continue
        seat_ids = set(f.seat_id for f in c.faces if f.seat_id)
        if len(seat_ids) > 1:
            # Check if any pair of seats is geographically distant
            for sa in seat_ids:
                for sb in seat_ids:
                    if sa >= sb:
                        continue
                    ea = session.seat_map.get(sa)
                    eb = session.seat_map.get(sb)
                    if ea is None or eb is None:
                        continue
                    dist = math.hypot(
                        ea.centroid_px[0] - eb.centroid_px[0],
                        ea.centroid_px[1] - eb.centroid_px[1],
                    )
                    if dist > config.diff_seat_dist_min_px:
                        logger.warning(
                            f"Cluster {c.id}: detections from seats "
                            f"{sa} and {sb} are {dist:.0f}px apart "
                            f"(possible merge error)"
                        )

    return clusters


def _apply_cluster_merges(
    clusters: List[ClusterResult],
    merge_pairs: List[Tuple[int, int]],
) -> List[ClusterResult]:
    """Merges cluster pairs using union-find, returns a new cluster list."""
    # Build union-find
    parent: Dict[int, int] = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for ci, cj in merge_pairs:
        union(ci, cj)

    # Group clusters by root
    root_to_faces: Dict[int, list] = defaultdict(list)
    noise_faces = []
    for c in clusters:
        if c.id == -1:
            noise_faces.extend(c.faces)
            continue
        root = find(c.id)
        root_to_faces[root].extend(c.faces)

    new_clusters = []
    for i, (root, faces) in enumerate(sorted(root_to_faces.items())):
        new_clusters.append(ClusterResult(id=i, faces=faces))
    if noise_faces:
        new_clusters.append(ClusterResult(id=-1, faces=noise_faces))
    return new_clusters


def cluster_faces(
    detections: List[FaceDetection],
    session: Optional[SessionState] = None,
) -> List[ClusterResult]:
    """
    Cluster face detections into per-person groups.

    KEY CHANGE FROM PREVIOUS VERSION:
      Clustering now runs on the FACE-ONLY embedding (512-d), not the
      fused 1024-d face+torso vector. The fused vector's zero-padding
      behaviour (when torso confidence is low) creates two structurally
      different embedding "modes" for the same person, which fragments
      clusters. Torso/posture/clothing descriptors are still used, but
      only in the seat-prior and centroid-merge passes below, as
      corroborating evidence — not as part of the primary distance metric.

    Pipeline:
      Stage 1:  FAISS kNN graph + Chinese Whispers on face embeddings
                (fallback: HDBSCAN)
      Stage 1.5: Same-seat fragment merge (pre-KNOWN_N_PERSONS)
      Stage 2:  KNOWN_N_PERSONS agglomerative merge
      Stage 3:  Seat-prior Bayesian correction
      Stage 4:  Centroid safety merge (catches remaining duplicates
                regardless of seat)
    """
    if session is None:
        session = _session

    try:
        import faiss
        import networkx
        _FAISS_NX = True
    except ImportError:
        logger.warning("faiss or networkx not installed — pip install faiss-cpu networkx")
        _FAISS_NX = False

    if not detections:
        logger.info("No detections to cluster.")
        return []

    # ── Quality gate: drop detections with very low illumination ────────
    # (replaces the previous no-op illumination renormalization)
    usable = [d for d in detections if d.illum_score >= config.illum_drop_threshold]
    dropped = len(detections) - len(usable)
    if dropped:
        logger.info(f"Dropped {dropped} detections below illum_score={config.illum_drop_threshold}")
    detections = usable
    if not detections:
        return []

    # ── Build face-only embedding matrix (the actual clustering basis) ──
    def _face_vec(d: FaceDetection) -> Optional[np.ndarray]:
        if d.face_embedding is not None and d.face_embedding.shape[0] == 512:
            return d.face_embedding
        # Fall back to first 512 dims of fused vector if face_embedding missing
        if d.embedding.shape[0] >= 512:
            return d.embedding[:512]
        return None

    face_vecs = [_face_vec(d) for d in detections]
    keep_idx  = [i for i, v in enumerate(face_vecs) if v is not None]
    if len(keep_idx) != len(detections):
        logger.info(f"Dropped {len(detections) - len(keep_idx)} detections with no face embedding")
    detections = [detections[i] for i in keep_idx]
    face_vecs  = [face_vecs[i] for i in keep_idx]
    if not detections:
        return []

    n        = len(detections)
    embs_raw = np.stack(face_vecs)
    norms    = np.linalg.norm(embs_raw, axis=1, keepdims=True)
    embs_n   = embs_raw / np.maximum(norms, 1e-10)

    # Epsilon is now correctly applied to 512-d face-only cosine distances,
    # matching the values it was originally calibrated for.
    epsilon       = HDBSCAN_EPSILON.get(ACTIVE_ENGINE, 0.36)
    sim_threshold = 1.0 - epsilon

    labels: np.ndarray

    if _FAISS_NX:
        try:
            logger.info(
                f"Stage 1: FAISS kNN graph on {n} detections "
                f"(face-only 512-d, k={config.knn_k}, sim_thresh={sim_threshold:.3f})"
            )
            G      = _build_knn_graph(embs_n, k=config.knn_k, sim_threshold=sim_threshold)
            cw_map = _chinese_whispers(G, iterations=20)

            unique_communities = sorted(set(cw_map.values()))
            remap  = {c: i for i, c in enumerate(unique_communities)}
            labels = np.array([remap[cw_map[i]] for i in range(n)], dtype=np.int64)

            n_sub = len(set(labels))
            logger.info(f"Chinese Whispers → {n_sub} communities")

            label_counts = Counter(labels.tolist())
            noise_mask   = np.array([label_counts[l] < config.hdbscan_min_cluster_size for l in labels])
            labels[noise_mask] = -1
            n_noise_raw = int(noise_mask.sum())
            logger.info(f"→ {len(set(labels) - {-1})} sub-clusters, {n_noise_raw} noise points")

            rescue_dist = NOISE_RESCUE_DIST.get(ACTIVE_ENGINE, 0.50)
            if n_noise_raw > 0:
                labels = _rescue_noise(detections, labels, np.empty((0, 0)), rescue_dist)

        except Exception as _e:
            logger.warning(f"FAISS/CW clustering failed ({_e}) — falling back to HDBSCAN")
            _FAISS_NX = False

    if not _FAISS_NX:
        try:
            import hdbscan as _hdbscan
        except ImportError:
            raise ImportError("Neither faiss/networkx nor hdbscan is available.")

        sim_mat     = np.clip(embs_n @ embs_n.T, -1.0, 1.0)
        dist_matrix = np.maximum(1.0 - sim_mat, 0.0)
        np.fill_diagonal(dist_matrix, 0.0)

        logger.info(f"Stage 1 (HDBSCAN fallback) on {n} detections (face-only 512-d)")
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=config.hdbscan_min_cluster_size,
            min_samples=config.hdbscan_min_samples,
            cluster_selection_epsilon=epsilon,
            cluster_selection_method="eom",
            metric="precomputed",
            core_dist_n_jobs=-1,
        )
        labels      = clusterer.fit_predict(dist_matrix)
        rescue_dist = NOISE_RESCUE_DIST.get(ACTIVE_ENGINE, 0.50)
        n_noise_raw = int((labels == -1).sum())
        logger.info(f"→ {len(set(labels) - {-1})} sub-clusters, {n_noise_raw} noise points")
        if n_noise_raw > 0:
            labels = _rescue_noise(detections, labels, dist_matrix, rescue_dist)

    # ── Stage 1.5: same-seat fragment merge (BEFORE Stage 2) ─────────────
    # If two sub-clusters are dominated by the same seat, and that seat
    # was only assigned to ≤2 distinct identities historically (i.e. no
    # evidence of a seat swap), merge them now. This collapses the most
    # common cause of "same person, multiple clusters": temporal/lighting
    # drift splitting one seat's tracklet into several sub-clusters.
    labels = _merge_same_seat_subclusters(labels, detections)

    # ── Stage 2: merge to KNOWN_N_PERSONS ─────────────────────────
    labels = _stage2_merge_subclusters(labels, embs_n, config.known_n_persons)

    n_real        = len(set(labels) - {-1})
    n_noise_final = int((labels == -1).sum())
    logger.info(f"→ {n_real} unique persons identified   {n_noise_final} noise detections")

    cluster_map: Dict[int, ClusterResult] = {}
    for det, label in zip(detections, labels):
        label = int(label)
        if label not in cluster_map:
            cluster_map[label] = ClusterResult(id=label)
        cluster_map[label].faces.append(det)

    clusters = sorted(cluster_map.values(), key=lambda c: (c.id == -1, -c.size))

    real_id = 0
    for c in clusters:
        if c.id != -1:
            c.id = real_id
            real_id += 1

    for c in clusters:
        if c.id == -1:
            logger.info(f"[noise]  {c.size:4d} detections  (outliers)")
        else:
            ts = [f.timestamp_sec for f in c.faces]
            logger.info(
                f"Person {c.id:03d}  {c.size:4d} detections  "
                f"t={min(ts):.1f}s–{max(ts):.1f}s  "
                f"avg_quality={c.mean_quality:.1f}  "
                f"dominant_seat={c.dominant_seat}"
            )

    # ── Stage 3: seat-prior Bayesian correction ────────────────────
    if session.seat_map:
        logger.info("Stage 3: applying seat-prior Bayesian correction")
        clusters = _seat_prior_correction(clusters, session)
        real_id = 0
        for c in clusters:
            if c.id != -1:
                c.id = real_id
                real_id += 1
        n_after = len([c for c in clusters if c.id != -1])
        logger.info(f"After seat-prior correction: {n_after} clusters")

    # ── Stage 4: centroid safety merge ─────────────────────────────
    # Catches remaining duplicates regardless of seat (e.g. someone moved
    # seats once, or seat assignment was noisy for some frames). Uses a
    # TIGHT threshold — this is a "obviously the same face" merge, not a
    # general clustering decision, so it should rarely fire on distinct
    # people.
    clusters = _centroid_safety_merge(clusters, dist_thresh=config.centroid_merge_thresh)
    real_id = 0
    for c in clusters:
        if c.id != -1:
            c.id = real_id
            real_id += 1
    n_final = len([c for c in clusters if c.id != -1])
    logger.info(f"After centroid safety merge: {n_final} clusters")

    return clusters


def _merge_same_seat_subclusters(
    labels: np.ndarray,
    detections: List[FaceDetection],
) -> np.ndarray:
    """
    Stage 1.5: if a seat's detections span multiple sub-cluster labels,
    AND the seat has not previously been seen with a *different, already
    confidently-distinct* identity (we don't know identity yet at this
    stage — so we use simple majority assignment), reassign all of that
    seat's detections to the majority label.

    Rationale: in seated environments, the dominant prior is "same seat
    = same person for the whole session". Sub-cluster splits within a
    seat are almost always representation drift (lighting, pose, partial
    occlusion across the session), not identity swaps. True swaps are
    rare and handled separately if you have swap-detection logic — this
    pass intentionally trades a small risk of merging a rare seat-swap
    for a large reduction in over-segmentation.
    """
    labels = labels.copy()
    seat_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, det in enumerate(detections):
        if det.seat_id:
            seat_to_indices[det.seat_id].append(i)

    merges = 0
    for seat_id, idxs in seat_to_indices.items():
        seat_labels = [labels[i] for i in idxs if labels[i] != -1]
        if len(set(seat_labels)) <= 1:
            continue
        majority_label = Counter(seat_labels).most_common(1)[0][0]
        for i in idxs:
            if labels[i] != -1 and labels[i] != majority_label:
                labels[i] = majority_label
                merges += 1

    if merges:
        logger.info(f"Stage 1.5: reassigned {merges} detections via same-seat majority merge")
    return labels


def _centroid_safety_merge(
    clusters: List[ClusterResult],
    dist_thresh: float = 0.30,
) -> List[ClusterResult]:
    """
    Stage 4: merges any two final clusters whose face-embedding centroids
    are closer than `dist_thresh`. This is deliberately tight — it is a
    safety net for "this is unmistakably the same face", not a general
    re-clustering step. Run AFTER seat-prior correction so it only needs
    to catch cross-seat duplicates.
    """
    real_clusters = [c for c in clusters if c.id != -1]
    if len(real_clusters) <= 1:
        return clusters
    def _face_centroid(c: ClusterResult) -> np.ndarray:
        embs = [f.face_embedding for f in c.faces if f.face_embedding is not None]
        if not embs:
            embs = [f.embedding[:512] for f in c.faces]
        mean = np.stack(embs).mean(axis=0)
        n = np.linalg.norm(mean)
        return mean / n if n > 0 else mean

    centroids = {c.id: _face_centroid(c) for c in real_clusters}

    merge_pairs: List[Tuple[int, int]] = []
    ids = [c.id for c in real_clusters]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ci, cj = ids[i], ids[j]
            dist = float(1.0 - np.clip(np.dot(centroids[ci], centroids[cj]), -1.0, 1.0))
            if dist < dist_thresh:
                merge_pairs.append((ci, cj))
                logger.info(
                    f"Stage 4: merging clusters {ci} + {cj} "
                    f"(centroid_dist={dist:.4f} < {dist_thresh})"
                )

    if merge_pairs:
        clusters = _apply_cluster_merges(clusters, merge_pairs)
    return clusters


# ══════════════════════════════════════════════════════════════════
#  STEP 3 — PROFILE MATCHING & EVALUATION
# ══════════════════════════════════════════════════════════════════

PROFILE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_profile_embeddings(profile_dir: str | Path) -> Dict[str, np.ndarray]:
    """
    Load profile images, extract one face embedding per image.
    Returns {person_name: embedding_array}.
    """
    profile_dir = Path(profile_dir)
    if not profile_dir.exists():
        raise FileNotFoundError(
            f"Profile directory not found: {profile_dir}\n"
            f"Create it and add one image per person named <person_name>.jpg"
        )

    profiles: Dict[str, np.ndarray] = {}
    image_files = sorted([
        f for f in profile_dir.iterdir()
        if f.suffix.lower() in PROFILE_IMAGE_EXTS
    ])

    if not image_files:
        raise ValueError(f"No images found in {profile_dir}. Supported: {PROFILE_IMAGE_EXTS}")

    logger.info(f"Loading {len(image_files)} profile images from {profile_dir}/")

    for img_path in image_files:
        person_name = img_path.stem
        img_bgr     = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning(f"Could not read {img_path.name} — skipping")
            continue

        detected = deduplicate_faces(detect_faces(img_bgr))

        if detected:
            best = max(detected, key=lambda d: d["confidence"])
            crop = best["crop"]
            if best["landmarks"]:
                crop = align_face_affine(img_bgr, best["landmarks"])
            crop = super_resolve_face(crop, 112)
            emb  = extract_face_embedding(crop, precomputed_insight=best.get("embedding"))
        else:
            logger.warning(f"No face detected in {img_path.name} — embedding whole image")
            crop = super_resolve_face(img_bgr, 112)
            emb  = extract_face_embedding(crop)

        if emb is None:
            logger.warning(f"Could not extract embedding for {img_path.name} — skipping")
            continue

        profiles[person_name] = emb
        logger.info(f"{img_path.name:30s}  →  \"{person_name}\"")

    logger.info(f"Loaded {len(profiles)}/{len(image_files)} profile embeddings")
    return profiles


def load_profile_embeddings_multi(profile_dir: str | Path) -> Dict[str, np.ndarray]:
    """Multi-image per person — averages + renormalises embeddings."""
    raw: Dict[str, list] = {}
    profile_dir = Path(profile_dir)
    for img_path in sorted(profile_dir.iterdir()):
        if img_path.suffix.lower() not in PROFILE_IMAGE_EXTS:
            continue
        stem = img_path.stem
        base = re.sub(r'[_\-]\w+$', '', stem) or stem
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        detected = deduplicate_faces(detect_faces(img_bgr))
        if detected:
            best = max(detected, key=lambda d: d["confidence"])
            crop = best["crop"]
            if best["landmarks"]:
                crop = align_face_affine(img_bgr, best["landmarks"])
            crop = super_resolve_face(crop, 112)
            emb  = extract_face_embedding(crop, precomputed_insight=best.get("embedding"))
        else:
            emb = extract_face_embedding(super_resolve_face(img_bgr, 112))
        if emb is not None:
            raw.setdefault(base, []).append(emb)

    profiles: Dict[str, np.ndarray] = {}
    for name, embs in raw.items():
        avg = np.mean(embs, axis=0)
        n   = np.linalg.norm(avg)
        profiles[name] = avg / n if n > 0 else avg
    return profiles


def match_clusters_to_profiles(
    clusters: List[ClusterResult],
    profile_embeddings: Dict[str, np.ndarray],
    threshold: float = PROFILE_MATCH_THRESHOLD,
) -> Tuple[Dict[int, str], Dict[int, "MatchResult"]]:
    """
    Matches each cluster centroid to the nearest profile embedding.
    Uses only the face half of fused embeddings for profile matching
    (profiles are face-only images).

    Returns
    -------
    cluster_labels : Dict[int, str]
        cluster_id → matched profile name (or "unknown_{id}" if below threshold).
        Backwards-compatible with all existing callers.
    match_records : Dict[int, MatchResult]
        cluster_id → full MatchResult with top-2 candidates + confidence flag.
        Always populated for every real cluster regardless of threshold.
    """
    if not profile_embeddings:
        raise ValueError("profile_embeddings is empty")

    profile_names  = list(profile_embeddings.keys())
    profile_matrix = np.stack([profile_embeddings[n] for n in profile_names])
    pnorms         = np.linalg.norm(profile_matrix, axis=1, keepdims=True)
    profile_normed = profile_matrix / np.maximum(pnorms, 1e-10)

    real_clusters = [c for c in clusters if c.id != -1]

    def _cluster_face_centroid(c: ClusterResult) -> np.ndarray:
        """Returns face-only centroid (first 512 dims of fused, or full if 512-d)."""
        embs = []
        for f in c.faces:
            if f.face_embedding is not None:
                embs.append(f.face_embedding)
            else:
                fe = f.embedding[:512]
                embs.append(fe)
        if not embs:
            return c.centroid[:512]
        stack = np.stack(embs)
        mean  = stack.mean(axis=0)
        n     = np.linalg.norm(mean)
        return mean / n if n > 0 else mean

    def _normed_centroid(c: ClusterResult) -> np.ndarray:
        """Returns profile-dimension-aligned, L2-normed face centroid."""
        centroid = _cluster_face_centroid(c)
        prof_dim = profile_normed.shape[1]
        if centroid.shape[0] != prof_dim:
            pad = prof_dim - centroid.shape[0]
            centroid = np.pad(centroid, (0, pad)) if pad > 0 else centroid[:prof_dim]
        n = np.linalg.norm(centroid)
        return centroid / n if n > 0 else centroid

    # ── Pass 0: compute all raw distances for threshold auto-calibration ──
    all_best_dists = []
    for c in real_clusters:
        cnormed = _normed_centroid(c)
        dists   = np.maximum(1.0 - np.clip(profile_normed @ cnormed, -1.0, 1.0), 0.0)
        all_best_dists.append(float(dists.min()))

    logger.info(
        f"Distance diagnostic: min={min(all_best_dists):.4f}  "
        f"median={float(np.median(all_best_dists)):.4f}  "
        f"max={max(all_best_dists):.4f}  threshold={threshold:.4f}"
    )

    n_would_match = sum(1 for d in all_best_dists if d < threshold)
    effective_threshold = threshold
    if n_would_match < len(real_clusters) * 0.5:
        effective_threshold = float(np.percentile(all_best_dists, 80))
        logger.warning(
            f"Only {n_would_match}/{len(real_clusters)} clusters matched at "
            f"threshold={threshold:.4f}. Auto-raising to {effective_threshold:.4f}"
        )

    # ── Pass 1: build full match records (top-2 always) ──────────────────
    cluster_labels: Dict[int, str]           = {}
    match_records:  Dict[int, "MatchResult"] = {}

    logger.info(
        f"{'ClusterID':>9}  {'Best Match':>20}  {'BestDist':>9}  "
        f"{'2ndMatch':>20}  {'2ndDist':>9}  gap={'':<6}  {'Status':>10}"
    )

    for c in real_clusters:
        cnormed = _normed_centroid(c)
        dists   = np.maximum(1.0 - np.clip(profile_normed @ cnormed, -1.0, 1.0), 0.0)

        sorted_idx  = np.argsort(dists)
        best_idx    = int(sorted_idx[0])
        best_dist   = float(dists[best_idx])
        best_name   = profile_names[best_idx]
        second_idx  = int(sorted_idx[1]) if len(sorted_idx) > 1 else best_idx
        second_dist = float(dists[second_idx]) if len(sorted_idx) > 1 else 1.0
        second_name = profile_names[second_idx] if len(sorted_idx) > 1 else best_name
        gap         = second_dist - best_dist
        is_confident = best_dist < effective_threshold

        if is_confident:
            cluster_labels[c.id] = best_name
            status = "matched"
        else:
            cluster_labels[c.id] = f"unknown_{c.id}"
            status = "unknown"

        # Always store full record — threshold never hides data here
        match_records[c.id] = MatchResult(
            cluster_id   = c.id,
            best_name    = best_name,
            best_dist    = best_dist,
            second_name  = second_name,
            second_dist  = second_dist,
            is_confident = is_confident,
        )

        logger.info(
            f"{c.id:>9}  {best_name:>20}  {best_dist:>9.4f}  "
            f"{second_name:>20}  {second_dist:>9.4f}  gap={gap:.4f}  {status:>10}"
        )

    n_matched = sum(1 for v in cluster_labels.values() if not v.startswith("unknown_"))
    logger.info(
        f"Matched: {n_matched}/{len(cluster_labels)} "
        f"({n_matched / max(len(cluster_labels), 1) * 100:.0f}%)   "
        f"Threshold used: {effective_threshold:.4f}"
    )
    return cluster_labels, match_records
def detect_cross_cluster_duplicates(
    clusters: List[ClusterResult],
    match_records: Dict[int, "MatchResult"],
    dup_threshold: float = 0.55,
) -> Dict[str, List[int]]:
    """
    Detects likely identity fragmentation: the same real person split
    across multiple clusters.

    Two complementary passes are run:

    Pass A — Profile-name collision
        Groups all clusters by their best-matching profile name (from
        match_records, ignoring the confident/unknown split). Any profile
        name with 2+ clusters assigned is a fragmentation candidate.

    Pass B — Pairwise centroid proximity among unknowns
        For every pair of unknown clusters (no confident profile match),
        computes the cosine distance between their embedding centroids.
        Pairs below dup_threshold are flagged as likely the same person
        even without a profile reference.

    Parameters
    ----------
    clusters       : output of cluster_faces()
    match_records  : Dict[int, MatchResult] from match_clusters_to_profiles()
    dup_threshold  : centroid cosine-distance below which two clusters are
                     considered the same identity (default 0.55)

    Returns
    -------
    fragmentation_groups : Dict[str, List[int]]
        profile_name (or "unknown_pair_X_Y") → sorted list of cluster IDs
        that likely belong to the same real person.
    """
    real_clusters  = {c.id: c for c in clusters if c.id != -1}
    fragmentation_groups: Dict[str, List[int]] = {}

    # ── Pass A: profile-name collision ───────────────────────────────────
    profile_to_clusters: Dict[str, List[int]] = defaultdict(list)
    for cid, rec in match_records.items():
        profile_to_clusters[rec.best_name].append(cid)

    logger.info("── Fragmentation Detection: Pass A (profile-name collision) ──")
    pass_a_found = False
    for profile_name, cids in sorted(profile_to_clusters.items()):
        if len(cids) < 2:
            continue
        pass_a_found = True
        fragmentation_groups[profile_name] = sorted(cids)
        # Sort sub-list by best_dist ascending so closest match leads
        cids_sorted = sorted(cids, key=lambda cid: match_records[cid].best_dist)
        logger.warning(
            f"  FRAGMENTATION: profile '{profile_name}' claimed by "
            f"{len(cids)} clusters → {cids_sorted}"
        )
        total_dets = sum(real_clusters[cid].size for cid in cids_sorted if cid in real_clusters)
        for cid in cids_sorted:
            c   = real_clusters.get(cid)
            rec = match_records[cid]
            if c is None:
                continue
            confident_tag = "✓ confident" if rec.is_confident else "✗ below-threshold"
            logger.warning(
                f"    cluster {cid:>4}  dist={rec.best_dist:.4f}  "
                f"size={c.size:>4}  seat={str(c.dominant_seat or '—'):>8}  "
                f"{confident_tag}"
            )
        logger.warning(f"    → combined detection count: {total_dets}")
    if not pass_a_found:
        logger.info("  No profile-name collisions detected.")

    # ── Pass B: pairwise centroid proximity among unknown clusters ────────
    logger.info("── Fragmentation Detection: Pass B (pairwise unknown proximity) ──")

    # Collect unknown cluster centroids (full fused embedding centroid)
    unknown_ids = [
        cid for cid, rec in match_records.items()
        if not rec.is_confident and cid in real_clusters
    ]

    if len(unknown_ids) >= 2:
        # Also compare unknowns against matched clusters for cross-type proximity
        all_ids_for_B = list(real_clusters.keys())  # every real cluster
        centroids_B: Dict[int, np.ndarray] = {}
        for cid in all_ids_for_B:
            c = real_clusters[cid]
            embs = np.stack([f.embedding for f in c.faces])
            mean = embs.mean(axis=0)
            n    = np.linalg.norm(mean)
            centroids_B[cid] = mean / n if n > 0 else mean

        pass_b_found = False
        reported_pairs: set = set()
        for i, cid_a in enumerate(unknown_ids):
            for cid_b in list(real_clusters.keys()):
                if cid_b == cid_a:
                    continue
                pair_key = tuple(sorted([cid_a, cid_b]))
                if pair_key in reported_pairs:
                    continue
                ca = centroids_B[cid_a]
                cb = centroids_B[cid_b]
                dist = float(1.0 - np.clip(np.dot(ca, cb) /
                             max(np.linalg.norm(ca) * np.linalg.norm(cb), 1e-10), -1.0, 1.0))
                if dist < dup_threshold:
                    reported_pairs.add(pair_key)
                    pass_b_found = True
                    group_key = f"unknown_pair_{pair_key[0]}_{pair_key[1]}"
                    # Avoid double-reporting what Pass A already caught
                    rec_a = match_records.get(cid_a)
                    rec_b = match_records.get(cid_b)
                    if (rec_a and rec_b and
                            rec_a.best_name == rec_b.best_name and
                            rec_a.best_name in fragmentation_groups):
                        continue  # already reported in Pass A
                    fragmentation_groups.setdefault(group_key, sorted(pair_key))
                    c_a_obj = real_clusters[cid_a]
                    c_b_obj = real_clusters[cid_b]
                    b_tag = "matched" if (rec_b and rec_b.is_confident) else "unknown"
                    logger.warning(
                        f"  PROXIMITY DUP: cluster {cid_a} (unknown) ↔ "
                        f"cluster {cid_b} ({b_tag})  "
                        f"centroid_dist={dist:.4f}  "
                        f"sizes={c_a_obj.size}/{c_b_obj.size}  "
                        f"seats={str(c_a_obj.dominant_seat or '—')}/"
                        f"{str(c_b_obj.dominant_seat or '—')}"
                    )
        if not pass_b_found:
            logger.info("  No pairwise unknown proximity duplicates detected.")
    else:
        logger.info(f"  Skipped (fewer than 2 unknown clusters: {len(unknown_ids)} found).")

    total_frags = sum(
        1 for v in fragmentation_groups.values() if len(v) >= 2
    )
    logger.info(
        f"Cross-cluster duplicate check complete: "
        f"{total_frags} fragmentation group(s) found."
    )
    return fragmentation_groups

def evaluate_clusters(
    clusters: List[ClusterResult],
    cluster_labels: Dict[int, str],
    save_confusion_png: Optional[str | Path] = "cluster_confusion.png",
    match_records: Optional[Dict[int, "MatchResult"]] = None,
) -> dict:
    """
    Full accuracy evaluation: NMI, ARI, Homogeneity, Completeness,
    V-measure, Purity, Recall, fragmentation_rate.

    cluster_labels : the threshold-split dict (cluster_id → name or "unknown_N").
                     Used for backwards-compatible confident-match reporting.
    match_records  : if provided, every cluster is assigned its best-match
                     profile name for NMI/ARI/Purity/Recall/fragmentation
                     regardless of threshold, so low-confidence clusters
                     are not silently excluded from metrics.
    """
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        homogeneity_completeness_v_measure,
    )

    # ── Build full-coverage assignment ──────────────────────────────────
    # If match_records available, every real cluster gets its best-match name
    # (could still be "unknown_N" if match_records is missing for that cluster).
    def _assigned_name(c: ClusterResult) -> str:
        if match_records and c.id in match_records:
            return match_records[c.id].best_name
        return cluster_labels.get(c.id, f"unknown_{c.id}")

    def _is_confident(c: ClusterResult) -> bool:
        if match_records and c.id in match_records:
            return match_records[c.id].is_confident
        return not cluster_labels.get(c.id, f"unknown_{c.id}").startswith("unknown_")

    # pred_labels / true_labels now include ALL real clusters
    pred_labels_full: List[int]  = []
    true_labels_full: List[str]  = []
    confidence_flags: List[bool] = []   # per detection

    cluster_to_true: Dict[int, list] = defaultdict(list)

    for c in clusters:
        if c.id == -1:
            continue
        name       = _assigned_name(c)
        confident  = _is_confident(c)
        for _ in c.faces:
            pred_labels_full.append(c.id)
            true_labels_full.append(name)
            confidence_flags.append(confident)
            cluster_to_true[c.id].append(name)

    if not pred_labels_full:
        logger.warning("evaluate_clusters: no clusters to evaluate.")
        return {}

    n_eval = len(pred_labels_full)
    logger.info(
        f"CLUSTER EVALUATION  ({n_eval} detections, "
        f"{len([c for c in clusters if c.id != -1])} clusters, "
        f"full-coverage assignment)"
    )

    nmi  = normalized_mutual_info_score(true_labels_full, pred_labels_full)
    ari  = adjusted_rand_score(true_labels_full, pred_labels_full)
    hom, comp_sk, vmeas = homogeneity_completeness_v_measure(
        true_labels_full, pred_labels_full
    )

    # ── Purity (all clusters) ─────────────────────────────────────────────
    purity_scores, fp_rates = [], []
    conf_purity_scores = []   # only confident clusters
    for cid, true_list in cluster_to_true.items():
        dom_count = Counter(true_list).most_common(1)[0][1]
        p = dom_count / len(true_list)
        purity_scores.append(p)
        fp_rates.append(1.0 - p)
        c_obj = next((c for c in clusters if c.id == cid), None)
        if c_obj and _is_confident(c_obj):
            conf_purity_scores.append(p)

    purity         = float(np.mean(purity_scores))
    fp_rate        = float(np.mean(fp_rates))
    purity_conf    = float(np.mean(conf_purity_scores)) if conf_purity_scores else float("nan")

    # ── Recall (all persons) ─────────────────────────────────────────────
    person_to_pred: Dict[str, list] = defaultdict(list)
    for p, t in zip(pred_labels_full, true_labels_full):
        person_to_pred[t].append(p)

    recall_scores, fn_rates = [], []
    for pid, pred_list in person_to_pred.items():
        dom_count = Counter(pred_list).most_common(1)[0][1]
        recall_scores.append(dom_count / len(pred_list))
        fn_rates.append(1.0 - dom_count / len(pred_list))
    recall  = float(np.mean(recall_scores))
    fn_rate = float(np.mean(fn_rates))

    # ── Fragmentation ────────────────────────────────────────────────────
    person_to_clusters: Dict[str, set] = defaultdict(set)
    for cid, name in [(c.id, _assigned_name(c)) for c in clusters if c.id != -1]:
        person_to_clusters[name].add(cid)

    n_true_persons = len(
        [p for p in person_to_clusters if not p.startswith("unknown_")]
    )
    fragmented_persons = {
        p: sorted(cids)
        for p, cids in person_to_clusters.items()
        if len(cids) > 1 and not p.startswith("unknown_")
    }
    n_fragmented       = len(fragmented_persons)
    fragmentation_rate = n_fragmented / max(n_true_persons, 1)

    n_pred = len(set(pred_labels_full))
    n_true = len(set(t for t in true_labels_full if not t.startswith("unknown_")))

    logger.info(f"Cluster count   : predicted={n_pred}   true persons={n_true}")
    logger.info(f"Fragmented      : {n_fragmented} persons split across 2+ clusters")
    logger.info(f"NMI             : {nmi:.4f}")
    logger.info(f"ARI             : {ari:.4f}")
    logger.info(f"Homogeneity     : {hom:.4f}")
    logger.info(f"Completeness    : {comp_sk:.4f}")
    logger.info(f"V-measure       : {vmeas:.4f}")
    logger.info(f"Purity (all)    : {purity:.4f}")
    logger.info(f"Purity (conf.)  : {purity_conf:.4f}")
    logger.info(f"Recall          : {recall:.4f}")
    logger.info(f"FP rate         : {fp_rate:.4f}")
    logger.info(f"FN rate         : {fn_rate:.4f}")
    logger.info(f"Fragmentation%  : {fragmentation_rate * 100:.1f}%  "
                f"({n_fragmented}/{n_true_persons} persons fragmented)")

    # ── Per-person breakdown ─────────────────────────────────────────────
    logger.info("Per-Person Breakdown:")
    for pid, pred_list in sorted(person_to_pred.items(), key=lambda x: str(x[0])):
        if str(pid).startswith("unknown_"):
            continue
        dom_cid       = Counter(pred_list).most_common(1)[0][0]
        dom_count     = Counter(pred_list).most_common(1)[0][1]
        fn            = len(pred_list) - dom_count
        recall_p      = dom_count / len(pred_list) * 100
        all_cids      = sorted(set(pred_list))
        dom_true_list = cluster_to_true[dom_cid]
        fp            = sum(1 for l in dom_true_list if l != pid)
        purity_p      = dom_count / len(dom_true_list) * 100
        cids_str      = str(all_cids) if len(all_cids) <= 4 else f"{all_cids[:4]}…"
        # Confidence tag
        conf_tag = ""
        if match_records:
            confident_cids   = [cid for cid in all_cids
                                if match_records.get(cid, MatchResult(0,"",1.0,"",1.0,False)).is_confident]
            conf_tag = f"  confident_clusters={confident_cids}"
        logger.info(
            f"{str(pid):>16}  {cids_str:>24}  "
            f"recall={recall_p:.1f}%  purity={purity_p:.1f}%  fn={fn}  fp={fp}{conf_tag}"
        )

    # ── Fragmentation detail ─────────────────────────────────────────────
    if fragmented_persons:
        logger.info("Fragmentation detail:")
        for person, cids in sorted(fragmented_persons.items()):
            sizes = [next((c.size for c in clusters if c.id == cid), 0) for cid in cids]
            logger.info(f"  {person}: clusters={cids}  sizes={sizes}")

    # ── Fragmentation Report (sorted by total detection count desc) ──────
    logger.info("=" * 72)
    logger.info("FRAGMENTATION REPORT")
    logger.info("=" * 72)
    if not fragmented_persons:
        logger.info("  No fragmented persons detected — all profiles map to exactly one cluster.")
    else:
        sorted_frags = sorted(
            fragmented_persons.items(),
            key=lambda kv: sum(
                next((c.size for c in clusters if c.id == cid), 0) for cid in kv[1]
            ),
            reverse=True,
        )
        for rank, (person, cids) in enumerate(sorted_frags, start=1):
            total_dets = sum(
                next((c.size for c in clusters if c.id == cid), 0) for cid in cids
            )
            logger.info(
                f"  #{rank}  '{person}'  →  {len(cids)} clusters  "
                f"(total {total_dets} detections)"
            )
            for cid in cids:
                c_obj = next((c for c in clusters if c.id == cid), None)
                if c_obj is None:
                    continue
                rec = match_records.get(cid) if match_records else None
                dist_str = f"dist={rec.best_dist:.4f}" if rec else "dist=n/a"
                conf_str = ("✓ confident" if rec and rec.is_confident else "✗ below-thresh") if rec else ""
                seat_str = str(c_obj.dominant_seat or "—")
                ts       = [f.timestamp_sec for f in c_obj.faces]
                t_range  = f"{min(ts):.1f}s–{max(ts):.1f}s"
                logger.info(
                    f"       cluster {cid:>4}  {dist_str}  size={c_obj.size:>4}  "
                    f"seat={seat_str:>8}  {t_range}  {conf_str}"
                )
            logger.info(
                f"       → To fix: set KNOWN_N_PERSONS to a lower value, "
                f"or merge clusters {cids} manually."
            )
        logger.info("=" * 72)

    if save_confusion_png:
        _plot_confusion_matrix(
            true_labels_full, pred_labels_full,
            save_path=save_confusion_png,
        )

    return {
        "nmi": nmi, "ari": ari, "homogeneity": hom,
        "completeness_sklearn": comp_sk, "v_measure": vmeas,
        "purity": purity, "purity_confident": purity_conf,
        "recall": recall,
        "fp_rate": fp_rate, "fn_rate": fn_rate,
        "n_pred": n_pred, "n_true": n_true,
        "n_fragmented": n_fragmented,
        "fragmentation_rate": fragmentation_rate,
        "fragmented_persons": {k: list(v) for k, v in fragmented_persons.items()},
    }

def _plot_confusion_matrix(true_labels, pred_labels, save_path="cluster_confusion.png") -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("Confusion matrix skipped — pip install seaborn matplotlib pandas")
        return

    all_true = sorted(set(true_labels), key=str)
    all_pred = sorted(set(pred_labels))
    df = pd.crosstab(
        pd.Categorical(true_labels, categories=all_true),
        pd.Categorical(pred_labels, categories=all_pred),
    )
    df.index   = [f"Person:{t}" for t in df.index]
    df.columns = ["noise" if p == -1 else f"Cluster:{p}" for p in df.columns]

    fig_w = max(8, len(all_pred) * 0.6)
    fig_h = max(5, len(all_true) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, linecolor="lightgrey", ax=ax)
    ax.set_title(
        "Cluster Confusion Matrix\nRows = True Person  |  Cols = Predicted Cluster",
        fontsize=10, pad=12,
    )
    ax.set_xlabel("Predicted Cluster", fontsize=9)
    ax.set_ylabel("True Person", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {save_path}")


# ══════════════════════════════════════════════════════════════════
#  CACHE HELPERS  (JSON + NumPy)
# ══════════════════════════════════════════════════════════════════

def save_detections(detections: List[FaceDetection], cache_path: str | Path) -> None:
    """
    Save detections to a directory-based cache.
    Layout:  <stem>_cache/
               manifest.json     — scalar fields
               embeddings.npy    — stacked fused embedding matrix (N × D)
               face_embs.npy     — face-only embeddings (N × 512), -1 if missing
               torso_embs.npy    — torso-only embeddings (N × 512), -1 if missing
               crops/NNNNNN.jpg  — face crops
               torso_crops/NNNNNN.jpg
    """
    cache_path = Path(cache_path)
    if cache_path.suffix == ".pkl":
        cache_path = cache_path.parent / (cache_path.stem + "_cache")

    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "crops").mkdir(exist_ok=True)
    (cache_path / "torso_crops").mkdir(exist_ok=True)

    manifest: list = []
    embeddings_list: list = []
    face_emb_list:   list = []
    torso_emb_list:  list = []

    def _to_python(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, list): return [_to_python(i) for i in v]
        return v

    for i, d in enumerate(detections):
        manifest.append({
            "frame_idx":     int(d.frame_idx),
            "timestamp_sec": float(d.timestamp_sec),
            "quality_score": float(d.quality_score),
            "bbox":          [int(x) for x in d.bbox],
            "confidence":    float(d.confidence),
            "landmarks":     _to_python(d.landmarks),
            "torso_quality": float(d.torso_quality),
            "face_conf":     float(d.face_conf),
            "torso_conf":    float(d.torso_conf),
            "seat_id":       d.seat_id,
            "gaze_vector":   list(d.gaze_vector) if d.gaze_vector else None,
            "hair_present":  d.hair_present,
            "has_glasses":   d.has_glasses,
            "writing_hand":  d.writing_hand,
            "illum_score":   float(d.illum_score),
        })
        embeddings_list.append(d.embedding)
        _dim512 = np.zeros(512, dtype=np.float64)
        face_emb_list.append(d.face_embedding  if d.face_embedding  is not None else _dim512)
        torso_emb_list.append(d.torso_embedding if d.torso_embedding is not None else _dim512)
        if d.crop_bgr is not None and d.crop_bgr.size > 0:
            cv2.imwrite(str(cache_path / "crops" / f"{i:06d}.jpg"), d.crop_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        if d.torso_crop_bgr is not None and d.torso_crop_bgr.size > 0:
            cv2.imwrite(str(cache_path / "torso_crops" / f"{i:06d}.jpg"), d.torso_crop_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

    with open(cache_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    np.save(str(cache_path / "embeddings.npy"),  np.stack(embeddings_list))
    np.save(str(cache_path / "face_embs.npy"),   np.stack(face_emb_list))
    np.save(str(cache_path / "torso_embs.npy"),  np.stack(torso_emb_list))
    logger.info(f"Saved {len(detections)} detections → {cache_path}/")


def load_detections(cache_path: str | Path) -> Optional[List[FaceDetection]]:
    """Load detections from directory-based cache. Migrates legacy .pkl files."""
    cache_path = Path(cache_path)

    # Legacy .pkl migration
    pkl_candidate: Optional[Path] = None
    if cache_path.suffix == ".pkl":
        pkl_candidate = cache_path
    else:
        alt = cache_path.parent / (
            cache_path.stem.removesuffix("_cache") + "_detections.pkl"
        )
        if alt.exists():
            pkl_candidate = alt

    if pkl_candidate is not None and pkl_candidate.exists():
        logger.warning(f"Loading legacy pickle cache: {pkl_candidate} (migrating to JSON+npy)")
        try:
            import pickle as _pickle
            with open(pkl_candidate, "rb") as f:
                detections = _pickle.load(f)
            new_path = pkl_candidate.parent / (pkl_candidate.stem + "_cache")
            save_detections(detections, new_path)
            pkl_candidate.unlink()
            logger.info(f"Migrated pickle → {new_path}/  (pkl deleted)")
            return detections
        except Exception as _e:
            logger.warning(f"Pickle migration failed ({_e}) — ignoring old cache")
            return None

    new_dir = cache_path if cache_path.suffix != ".pkl" else (
        cache_path.parent / (cache_path.stem + "_cache")
    )
    manifest_path   = new_dir / "manifest.json"
    embeddings_path = new_dir / "embeddings.npy"
    crops_dir       = new_dir / "crops"
    torso_dir       = new_dir / "torso_crops"
    face_emb_path   = new_dir / "face_embs.npy"
    torso_emb_path  = new_dir / "torso_embs.npy"

    if not manifest_path.exists() or not embeddings_path.exists():
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        emb_matrix   = np.load(str(embeddings_path))
        face_embs    = np.load(str(face_emb_path))   if face_emb_path.exists()  else None
        torso_embs   = np.load(str(torso_emb_path))  if torso_emb_path.exists() else None

        detections: List[FaceDetection] = []
        for i, m in enumerate(manifest):
            crop_path  = crops_dir / f"{i:06d}.jpg"
            torso_path = torso_dir / f"{i:06d}.jpg"
            crop_bgr   = cv2.imread(str(crop_path))  if crop_path.exists()  else None
            torso_bgr  = cv2.imread(str(torso_path)) if torso_path.exists() else None
            gv = m.get("gaze_vector")
            detections.append(FaceDetection(
                frame_idx=m["frame_idx"],
                timestamp_sec=m["timestamp_sec"],
                embedding=emb_matrix[i],
                crop_bgr=(crop_bgr if crop_bgr is not None else np.zeros((112, 112, 3), np.uint8)),
                quality_score=m["quality_score"],
                bbox=tuple(m["bbox"]),
                confidence=m["confidence"],
                landmarks=m.get("landmarks"),
                face_embedding=(face_embs[i]  if face_embs  is not None else None),
                torso_embedding=(torso_embs[i] if torso_embs is not None else None),
                torso_crop_bgr=torso_bgr,
                torso_quality=m.get("torso_quality", 0.0),
                face_conf=m.get("face_conf", 0.0),
                torso_conf=m.get("torso_conf", 0.0),
                seat_id=m.get("seat_id"),
                gaze_vector=(tuple(gv) if gv else None),
                hair_present=m.get("hair_present"),
                has_glasses=m.get("has_glasses"),
                writing_hand=m.get("writing_hand"),
                illum_score=m.get("illum_score", 1.0),
            ))
        logger.info(f"Loaded {len(detections)} detections from {new_dir}/")
        return detections
    except Exception as _e:
        logger.warning(f"Failed to load cache from {new_dir}: {_e}")
        return None
def save_session_meta(session: SessionState, cache_path: str | Path) -> None:
    cache_path = Path(cache_path)
    if cache_path.suffix == ".pkl":
        cache_path = cache_path.parent / (cache_path.stem + "_cache")
    meta = {
        "seat_map": {
            sid: {
                "seat_id": e.seat_id,
                "centroid_px": [int(x) for x in e.centroid_px],
                "bbox_px": [int(x) for x in e.bbox_px] if e.bbox_px else None,
                "face_size_est_px": e.face_size_est_px,
            }
            for sid, e in session.seat_map.items()
        }
    }
    with open(cache_path / "session_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_session_meta(cache_path: str | Path) -> Dict[str, SeatEntry]:
    cache_path = Path(cache_path)
    if cache_path.suffix == ".pkl":
        cache_path = cache_path.parent / (cache_path.stem + "_cache")
    meta_path = cache_path / "session_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        sid: SeatEntry(
            seat_id=e["seat_id"],
            centroid_px=tuple(e["centroid_px"]),
            bbox_px=tuple(e["bbox_px"]) if e["bbox_px"] else None,
            face_size_est_px=e["face_size_est_px"],
        )
        for sid, e in meta.get("seat_map", {}).items()
    }

# ══════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def run_pipeline(
    video_path: str | Path,
    cache_path: str | Path | None = None,
    force_redetect: bool = False,
    profile_dir: str | Path | None = None,
    profile_match_threshold: float = PROFILE_MATCH_THRESHOLD,
    seated_mode: bool = True,
) -> List[ClusterResult]:
    """
    Run the full detect → cluster pipeline.

    seated_mode:
        If True (default), enables all seated-environment features:
        seat map extraction, seat-anchored tracklets, dual-tower embeddings,
        adaptive fusion calibration, and Bayesian seat-prior correction.

    profile_dir (optional):
        Path to folder with one profile image per person (<name>.jpg).
        If provided, runs matching + evaluation automatically.

    Example:
        clusters = run_pipeline("lecture.mp4", profile_dir="ch_profile")
    """
    setup_logging(json_output=config.log_json)
    video_path = Path(video_path)

    if cache_path is None:
        cache_path = video_path.parent / f"{video_path.stem}_detections.pkl"
    cache_path = Path(cache_path)

    # Reset session state for this video
    global _session
    _session = SessionState(
        fusion_alpha=config.face_alpha,
        fusion_beta=config.torso_beta,
    )
    if not seated_mode:
        _session.seat_map = {}

    logger.info("STEP 1 — Frame sampling & face detection")
    detections = None
    if not force_redetect:
        detections = load_detections(cache_path)

    if detections is None:
        detections = sample_and_detect(video_path, _session)
        save_detections(detections, cache_path)
        save_session_meta(_session, cache_path)
    else:
        logger.info("Skipped detection — using cache")
        _session.seat_map = load_session_meta(cache_path)
        if not _session.seat_map:
            logger.warning(
                "session_meta.json not found — seat_map empty, "
                "Stage 3 seat-prior correction will be skipped. "
                "Re-run with force_redetect=True once to populate it."
            )
        if detections and config.calibrate_fusion:
            _run_fusion_calibration(detections, _session, 0, 25.0)
            _reembed_with_calibrated_fusion(detections, _session)

    if seated_mode and detections:
        logger.info("Building seat-anchored tracklets")
        build_seat_tracklets(detections, _session)

    logger.info("STEP 2 — Face clustering")
    clusters = cluster_faces(detections, _session)

    n_persons = len([c for c in clusters if c.id != -1])
    logger.info(f"Pipeline complete — {n_persons} unique persons identified")

    if profile_dir is not None:
            logger.info("STEP 3 — Profile matching & evaluation")
            profile_embeddings = load_profile_embeddings(profile_dir)
            cluster_labels, match_records = match_clusters_to_profiles(
                clusters, profile_embeddings, threshold=profile_match_threshold
            )
            detect_cross_cluster_duplicates(clusters, match_records)
            confusion_png = video_path.parent / f"{video_path.stem}_confusion.png"
            evaluate_clusters(
                clusters, cluster_labels,
                save_confusion_png=confusion_png,
                match_records=match_records,
            )

    return clusters


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    setup_logging(json_output=config.log_json)

    if len(sys.argv) < 2:
        vdir = Path("input_videos")
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        videos = [f for f in vdir.iterdir() if f.suffix.lower() in exts]
        if not videos:
            logger.info("Usage: python face_cluster.py <video_path> [--no-seated]")
            sys.exit(1)
        video = sorted(videos)[0]
    else:
        video = Path(sys.argv[1])

    seated_mode = "--no-seated" not in sys.argv

    PROFILE_DIR = os.environ.get("FACE_PROFILE_DIR") or None
    if "--profile" in sys.argv:
        pidx = sys.argv.index("--profile")
        PROFILE_DIR = sys.argv[pidx + 1] if pidx + 1 < len(sys.argv) else PROFILE_DIR

    clusters = run_pipeline(video, profile_dir=PROFILE_DIR, seated_mode=seated_mode)

    _cluster_name:   Dict[int, str]            = {}
    _match_records:  Dict[int, "MatchResult"]  = {}
    if PROFILE_DIR is not None:
        try:
            _pembs                      = load_profile_embeddings(PROFILE_DIR)
            _cluster_name, _match_records = match_clusters_to_profiles(
                clusters, _pembs, threshold=config.profile_match_threshold)
            detect_cross_cluster_duplicates(clusters, _match_records)
        except Exception:
            pass

    logger.info("FINAL CLUSTER SUMMARY")
    logger.info(
        f"{'ID':>5}  {'Person':>16}  {'Det':>5}  {'AvgQ':>7}  "
        f"{'Recall%':>8}  {'Purity%':>8}  {'Seat':>8}  {'Time Range':>22}"
    )

    _c2true: Dict = defaultdict(list)
    _p2pred: Dict = defaultdict(list)
    for c in clusters:
        if c.id == -1 or c.id not in _cluster_name:
            continue
        name = _cluster_name[c.id]
        for _ in c.faces:
            _c2true[c.id].append(name)
            _p2pred[name].append(c.id)

    for c in clusters:
        ts         = [f.timestamp_sec for f in c.faces]
        time_range = f"{min(ts):.1f}s–{max(ts):.1f}s"

        if c.id == -1:
            logger.info(
                f"{'noise':>5}  {'':>16}  {c.size:>5}  "
                f"{c.mean_quality:>7.1f}  {'':>8}  {'':>8}  {'':>8}  {time_range:>22}"
            )
            continue

        name      = _cluster_name.get(c.id, f"cluster_{c.id}")
        seat_str  = str(c.dominant_seat or "—")

        true_list = _c2true.get(c.id, [])
        purity_p  = (Counter(true_list).most_common(1)[0][1] / len(true_list) * 100
                     if true_list else 0.0)

        pred_list = _p2pred.get(name, [])
        recall_p  = (Counter(pred_list).most_common(1)[0][1] / len(pred_list) * 100
                     if pred_list else 0.0)

        purity_str = f"{purity_p:.1f}%" if true_list else "  —"
        recall_str = f"{recall_p:.1f}%" if pred_list else "  —"

        # Dominant auxiliary features summary
        glasses_votes = [f.has_glasses for f in c.faces if f.has_glasses is not None]
        glasses_str   = ("👓" if Counter(glasses_votes).most_common(1)[0][0] else "")  if glasses_votes else ""
        hand_votes    = [f.writing_hand for f in c.faces if f.writing_hand is not None]
        hand_str      = (Counter(hand_votes).most_common(1)[0][0][0].upper() if hand_votes else "")

        logger.info(
            f"{c.id:>5}  {name:>16}  {c.size:>5}  "
            f"{c.mean_quality:>7.1f}  {recall_str:>8}  {purity_str:>8}  "
            f"{seat_str:>8}  {time_range:>22}  {glasses_str}{hand_str}"
        )

    n_real          = len([c for c in clusters if c.id != -1])
    n_matched_final = sum(1 for v in _cluster_name.values() if not v.startswith("unknown_"))
    logger.info(
        f"Total: {n_real} clusters  "
        f"{n_matched_final} matched to profiles  "
        f"{len([c for c in clusters if c.id == -1])} noise  "
        f"fusion α={_session.fusion_alpha:.3f} β={_session.fusion_beta:.3f}"
    )
