"""
face_cluster.py  —  Step 1 & 2 of the profile-extraction pipeline
====================================================================
WHAT THIS FILE DOES:
  1. Face Detection  — samples frames from a video, detects every face
                       using a stacked multi-detector pipeline
  2. Face Clustering — groups detections of the same person together
                       using HDBSCAN on cosine distance with AdaFace embeddings

CLUSTERING APPROACH:
  Uses hdbscan.HDBSCAN with:
    - Precomputed cosine distance matrix
    - min_cluster_size=3, min_samples=2  (avoids over-splitting)
    - cluster_selection_epsilon=0.35     (merges nearby sub-clusters)
    - cluster_selection_method="eom"     (excess of mass — fewer, larger clusters)
    - NO post-merge step

OUTPUT (Step 3 — Best-Face registration via combined pipeline):
  A list of Cluster objects, each containing:

  Combined run (detect → cluster → best face):
    python ../run_face_pipeline.py input_videos/myvideo.mp4
  Or API: POST /api/v1/register/video on the Best-Face server.

  Cluster fields:
    cluster.id          : int  (0, 1, 2 … N-1;  -1 = noise/outliers)
    cluster.faces       : list of FaceDetection namedtuples
      .frame_idx        : int
      .timestamp_sec    : float
      .embedding        : np.ndarray (512-d float64, L2-normed)
      .crop_bgr         : np.ndarray
      .quality_score    : float
      .bbox             : (x, y, w, h)
      .confidence       : float
      .landmarks        : list or None

HOW TO USE:
  from face_cluster import run_pipeline, ClusterResult

  clusters = run_pipeline("input_videos/myvideo.mp4")

  for c in clusters:
      print(f"Person {c.id}  →  {len(c.faces)} detections")

INSTALL:
  pip install hdbscan insightface onnxruntime-gpu deepface
  pip install face_recognition mtcnn
  pip install scikit-learn numpy opencv-python
"""

from __future__ import annotations

import os
import gc
import math
import pickle
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ══════════════════════════════════════════════════════════════════
#  LIBRARY DETECTION
# ══════════════════════════════════════════════════════════════════

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
            print(f"   InsightFace ({_model_name}) loaded")
            break
        except Exception as _e:
            print(f"   InsightFace {_model_name} not available: {_e}")
except Exception as e:
    print(f"✗  InsightFace not available ({e})")

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
        print("   AdaFace IR-101 (WebFace12M) loaded")
    else:
        print(f"   AdaFace weights not found at {_w1} or {_w2} — skipping")
except Exception as e:
    print(f"✗  AdaFace not available ({e})")

# DEEPFACE = False
# _DF_MODEL = "ArcFace"
# _DF_DETECTOR = "retinaface"
# try:
#     from deepface import DeepFace
#     DeepFace.build_model("ArcFace")
#     DEEPFACE = True
#     print(f"   DeepFace ({_DF_MODEL} + {_DF_DETECTOR}) loaded")
# except Exception as e:
#     print(f"✗  DeepFace not available ({e})")

# FACE_REC = False
# try:
#     import face_recognition as _fr
#     FACE_REC = True
#     print("   face_recognition (dlib) loaded")
# except Exception as e:
#     print(f"✗  face_recognition not available ({e})")

MTCNN = False
_mtcnn_det = None
try:
    from mtcnn import MTCNN as _MTCNNCls
    _mtcnn_det = _MTCNNCls()
    MTCNN = True
    print("   MTCNN loaded")
except Exception as e:
    print(f"✗  MTCNN not available ({e})")

if ADAFACE:
    ACTIVE_ENGINE = "adaface"
elif INSIGHTFACE:
    ACTIVE_ENGINE = "insightface"
# elif DEEPFACE:
#     ACTIVE_ENGINE = "deepface"
# elif FACE_REC:
#     ACTIVE_ENGINE = "face_recognition"
else:
    ACTIVE_ENGINE = "histogram"
    print("⚠  WARNING: All neural engines unavailable — clustering accuracy will be poor.")

print(f"\n   🔬 Active engine: {ACTIVE_ENGINE}\n")


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

SAMPLE_INTERVAL_SEC = 1.5
MIN_FACE_QUALITY = 4.0
ABSOLUTE_MAX_FRAMES = 600

# ── Stage 1: HDBSCAN (tight — intentionally over-splits) ──────────
# epsilon deliberately kept tight (0.36) so HDBSCAN finds every
# plausible sub-cluster. Stage 2 then merges them correctly.
# Do NOT raise epsilon to fight over-splitting — that causes the
# mega-cluster collapse seen at eps=0.48+. Let Stage 2 handle merging.
HDBSCAN_MIN_CLUSTER_SIZE = 3   # min detections to form a sub-cluster
HDBSCAN_MIN_SAMPLES = 2        # lower = fewer noise points
HDBSCAN_EPSILON = {
    "adaface":          0.36,
    "insightface":      0.38,
    "deepface":         0.45,
    "face_recognition": 0.42,
    "histogram":        0.50,
}

# ── Stage 2: Agglomerative merge on sub-cluster centroids ─────────
# After HDBSCAN produces ~99 sub-clusters, we run AgglomerativeClustering
# with complete linkage on the 99×99 centroid distance matrix to collapse
# them into the correct number of people.
#
# KNOWN_N_PERSONS: set this to the number of people you know are in the
# video. The algorithm will produce exactly this many clusters.
# Set to None to use AUTO mode (finds the elbow in the merge tree).
KNOWN_N_PERSONS = 4   # set to None to auto-detect

# AUTO mode: when KNOWN_N_PERSONS is None, we walk down the merge tree
# and stop where the next merge would jump by more than MERGE_JUMP_FACTOR
# times the previous merge distance. Tune this if auto gives wrong count.
MERGE_JUMP_FACTOR = 1.8  # larger = fewer final clusters

# Noise rescue: assign HDBSCAN noise points to nearest cluster centroid
# if within this cosine distance.
NOISE_RESCUE_DIST = {
    "adaface":          0.50,
    "insightface":      0.52,
    "deepface":         0.65,
    "face_recognition": 0.60,
    "histogram":        0.70,
}


# ══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class FaceDetection:
    frame_idx:     int
    timestamp_sec: float
    embedding:     np.ndarray
    crop_bgr:      np.ndarray
    quality_score: float
    bbox:          tuple
    confidence:    float
    landmarks:     Optional[list]


@dataclass
class ClusterResult:
    id:    int
    faces: list[FaceDetection] = field(default_factory=list)

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


# ══════════════════════════════════════════════════════════════════
#  FRAME SAMPLING
# ══════════════════════════════════════════════════════════════════

def _compute_sampling_budget(duration_s: float) -> int:
    duration_min = duration_s / 60.0
    budget = int(50 + 35 * math.sqrt(duration_min))
    return min(budget, ABSOLUTE_MAX_FRAMES)


def _select_frames_temporal(
    candidates: list[tuple[int, np.ndarray, float]],
    budget: int,
) -> list[tuple[int, np.ndarray, float]]:
    if not candidates:
        return []
    max_idx = max(c[0] for c in candidates)
    min_idx = min(c[0] for c in candidates)
    span = max(max_idx - min_idx, 1)
    bucket_size = span / budget
    buckets: dict[int, list] = {}
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
#  IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════

def enhance_frame(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def upscale_face(face_bgr: np.ndarray, target: int = 112) -> np.ndarray:
    h, w = face_bgr.shape[:2]
    if max(h, w) >= target:
        return face_bgr
    scale = target / max(h, w)
    interp = cv2.INTER_LANCZOS4 if scale < 4 else cv2.INTER_CUBIC
    return cv2.resize(face_bgr, (int(w * scale), int(h * scale)), interpolation=interp)


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


def face_quality_score(face_bgr: np.ndarray) -> float:
    h, w = face_bgr.shape[:2]
    if h < 5 or w < 5:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    size_score = min(1.0, min(h, w) / 80.0)
    mean_bright = float(gray.mean())
    bright_mul = 1.0 if 35 < mean_bright < 225 else 0.25
    contrast_mul = min(1.0, float(gray.std()) / 20.0)
    return float(sharpness * size_score * bright_mul * contrast_mul)


def laplacian_sharpness(img_bgr: np.ndarray) -> float:
    h, w = img_bgr.shape[:2]
    if h > 320:
        img_bgr = cv2.resize(img_bgr, (int(w * 320 / h), 320), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


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
                    list(kp.get("left_eye", [0, 0])), list(kp.get("right_eye", [0, 0])),
                    list(kp.get("nose", [0, 0])), list(kp.get("mouth_left", [0, 0])),
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

    # if FACE_REC:
    #     try:
    #         rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    #         locs = _fr.face_locations(rgb, number_of_times_to_upsample=2, model="hog")
    #         for top, right, bottom, left in locs:
    #             x = max(0, left); y = max(0, top)
    #             w = right - left; h = bottom - top
    #             if w < 8 or h < 8:
    #                 continue
    #             results.append({
    #                 "box": (x, y, w, h), "crop": img_bgr[y:y+h, x:x+w].copy(),
    #                 "landmarks": None, "confidence": 0.70, "embedding": None,
    #             })
    #         if results:
    #             return results
    #     except Exception:
    #         pass

    for xml, sf, mn in [
        ("haarcascade_frontalface_default.xml", 1.05, 4),
        ("haarcascade_frontalface_alt2.xml", 1.04, 3),
    ]:
        cas = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
        if cas.empty():
            continue
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (int(w * 360 / h), 360), interpolation=cv2.INTER_AREA) if h > 360 else img_bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        scale = h / small.shape[0]
        dets_small = cas.detectMultiScale(gray, sf, mn, minSize=(12, 12))
        dets = [(int(x*scale), int(y*scale), int(w2*scale), int(h2*scale))
                for x, y, w2, h2 in dets_small] if len(dets_small) else []
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


def extract_embedding(face_bgr: np.ndarray, precomputed_insight=None) -> Optional[np.ndarray]:
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

    # if DEEPFACE:
    #     try:
    #         face_up = upscale_face(face_bgr, 112)
    #         face_rgb = cv2.cvtColor(face_up, cv2.COLOR_BGR2RGB)
    #         rep = DeepFace.represent(
    #             img_path=face_rgb, model_name=_DF_MODEL,
    #             detector_backend="skip", enforce_detection=False, align=True,
    #         )
    #         if rep:
    #             return _l2_norm(np.array(rep[0]["embedding"], dtype=np.float64))
    #     except Exception:
    #         pass

    # if FACE_REC:
    #     try:
    #         face_up = upscale_face(face_bgr, 150)
    #         face_rgb = cv2.cvtColor(face_up, cv2.COLOR_BGR2RGB)
    #         locs = _fr.face_locations(face_rgb, number_of_times_to_upsample=2)
    #         if not locs:
    #             h, w = face_rgb.shape[:2]
    #             locs = [(0, w, h, 0)]
    #         encs = _fr.face_encodings(face_rgb, locs)
    #         if encs:
    #             return _l2_norm(np.array(encs[0], dtype=np.float64))
    #     except Exception:
    #         pass

    # Histogram fallback
    try:
        face_r = cv2.resize(face_bgr, (64, 64))
        gray = cv2.cvtColor(face_r, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        _, ang = cv2.cartToPolar(gx, gy)
        hist_g = cv2.calcHist([ang], [0], None, [64], [0, 2 * np.pi]).flatten()
        hists = [hist_g]
        for ch in cv2.split(face_r):
            hists.append(cv2.calcHist([ch], [0], None, [32], [0, 256]).flatten())
        return _l2_norm(np.concatenate(hists).astype(np.float64))
    except Exception:
        pass

    return None


# ══════════════════════════════════════════════════════════════════
#  STEP 1 — FRAME SAMPLING + FACE DETECTION
# ══════════════════════════════════════════════════════════════════

def sample_and_detect(video_path: str | Path) -> list[FaceDetection]:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_s = total_frames / fps
    step = max(1, int(fps * SAMPLE_INTERVAL_SEC))

    print(f"\n  Video : {video_path.name}")
    print(f"  Frames: {total_frames}   FPS: {fps:.1f}   Duration: {duration_s:.1f}s")

    budget = _compute_sampling_budget(duration_s)
    print(f"  Sampling budget: {budget} frames  (duration={duration_s/60:.1f} min)")

    # Phase 1a: collect candidates scored by sharpness
    candidates: list[tuple[int, np.ndarray, float]] = []
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

    # Phase 1b: temporally-spread selection
    selected = _select_frames_temporal(candidates, budget)
    print(f"  Sampled {len(candidates)} candidates → kept {len(selected)} (temporal spread)")

    # Phase 1c: detect + embed
    all_detections: list[FaceDetection] = []
    for processed, (frame_idx, frame, _) in enumerate(selected):
        h, w = frame.shape[:2]
        if h > 480:
            frame = cv2.resize(frame, (int(w * 480 / h), 480), interpolation=cv2.INTER_AREA)
        frame_enh = enhance_frame(frame)
        del frame
        faces = deduplicate_faces(detect_faces(frame_enh))

        for face_info in faces:
            crop = face_info["crop"]
            if face_info["landmarks"]:
                crop = align_face_affine(frame_enh, face_info["landmarks"])
            crop = upscale_face(crop, 112)

            quality = face_quality_score(crop)
            if quality < MIN_FACE_QUALITY:
                continue

            emb = extract_embedding(crop, precomputed_insight=face_info.get("embedding"))
            if emb is None:
                continue

            all_detections.append(FaceDetection(
                frame_idx=frame_idx,
                timestamp_sec=round(frame_idx / fps, 3),
                embedding=emb,
                crop_bgr=crop,
                quality_score=quality,
                bbox=face_info["box"],
                confidence=face_info["confidence"],
                landmarks=face_info.get("landmarks"),
            ))

        del frame_enh, faces

        if (processed + 1) % 10 == 0:
            gc.collect()
            print(f"  Processed {processed+1}/{len(selected)} frames  "
                  f"| Detections so far: {len(all_detections)}")

    print(f"\n  Total face detections collected: {len(all_detections)}")
    return all_detections


# ══════════════════════════════════════════════════════════════════
#  STEP 2 — FACE CLUSTERING  (HDBSCAN + cosine distance)
# ══════════════════════════════════════════════════════════════════
#
#  WHY HDBSCAN with cluster_selection_epsilon instead of silhouette sweep:
#
#  The previous silhouette sweep over AgglomerativeClustering thresholds
#  optimises for *separation quality*, not *correct person count*. When
#  AdaFace embeddings have tight within-person variance but the video has
#  many people, silhouette score keeps rewarding finer splits, leading to
#  500+ clusters for 26 people.
#
#  HDBSCAN with cluster_selection_epsilon works differently:
#    - Builds a full density hierarchy of the embedding space
#    - cluster_selection_epsilon sets a minimum cosine distance between
#      distinct clusters (merges anything closer than this)
#    - For AdaFace, same-person distances are typically 0.10–0.40 and
#      different-person distances are typically 0.50–0.90.
#      epsilon=0.35 sits in the gap → collapses multiple detections of
#      the same person without merging different people.
#    - cluster_selection_method="eom" extracts fewer, larger clusters
#      (opposite of "leaf" which extracts many small ones)
#    - NO post-merge step needed — epsilon handles it at clustering time.
#
# ══════════════════════════════════════════════════════════════════

def _cosine_dist_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance for L2-normed embeddings: dist = 1 - dot(a,b)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-10)
    sim = np.clip(normed @ normed.T, -1.0, 1.0)
    dist = np.maximum(1.0 - sim, 0.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _rescue_noise(
    detections: list[FaceDetection],
    labels: np.ndarray,
    dist_matrix: np.ndarray,
    rescue_dist: float,
) -> np.ndarray:
    """
    Assign noise points (label == -1) to the nearest cluster centroid
    if within rescue_dist. Uses centroid distance for robustness.
    """
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels

    labels = labels.copy()
    real_labels = sorted(set(labels) - {-1})
    if not real_labels:
        return labels

    # Precompute cluster centroids (mean of member embeddings, L2-normed)
    emb_matrix = np.stack([d.embedding for d in detections])
    centroids = {}
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
        print(f"  Rescued {rescued}/{n_noise} noise points into existing clusters")
    return labels


def _stage2_merge_subclusters(
    labels: np.ndarray,
    embs_n: np.ndarray,
    n_target: Optional[int],
) -> np.ndarray:
    """
    Stage 2: Agglomerative merge of HDBSCAN sub-clusters.

    Takes the over-split HDBSCAN labels (~99 sub-clusters) and merges
    them down to the correct number of people using AgglomerativeClustering
    with COMPLETE linkage on the centroid distance matrix.

    Complete linkage: two groups merge only when the farthest pair of
    their centroids is within threshold. This prevents chain-linking
    (the mega-cluster collapse that greedy single-linkage caused).

    n_target:
      - int  → produce exactly n_target clusters (set KNOWN_N_PERSONS)
      - None → auto-detect using elbow in merge distances
    """
    from sklearn.cluster import AgglomerativeClustering

    real_labels = sorted(set(labels) - {-1})
    n_sub = len(real_labels)
    if n_sub == 0:
        return labels
    if n_sub == 1:
        return labels

    # Build centroid matrix: one row per sub-cluster
    centroids = []
    for rl in real_labels:
        members = embs_n[labels == rl]
        c = members.mean(axis=0)
        nc = np.linalg.norm(c)
        centroids.append(c / nc if nc > 0 else c)
    cent_arr = np.stack(centroids)                   # (n_sub, 512)

    # Pairwise cosine distance between centroids
    cent_dist = np.maximum(
        1.0 - np.clip(cent_arr @ cent_arr.T, -1.0, 1.0), 0.0
    )
    np.fill_diagonal(cent_dist, 0.0)

    # Determine n_clusters to request
    if n_target is not None:
        n_clusters = min(n_target, n_sub)
        print(f"  Stage 2: merging {n_sub} sub-clusters → {n_clusters} "
              f"(KNOWN_N_PERSONS={n_target}, complete linkage)")
    else:
        # Auto: find elbow in the linkage tree
        # Run with n_clusters=2 to expose the full merge distance sequence,
        # then pick the cut where the jump ratio is largest.
        agg_full = AgglomerativeClustering(
            n_clusters=2, metric="precomputed", linkage="complete"
        )
        agg_full.fit(cent_dist)
        # distances_ is sorted ascending — last entries are the largest merges
        merge_dists = agg_full.distances_
        jumps = merge_dists[1:] / np.maximum(merge_dists[:-1], 1e-6)
        # Only consider merges in the upper half of the tree
        half = len(jumps) // 2
        candidate_jumps = jumps[half:]
        if candidate_jumps.max() >= MERGE_JUMP_FACTOR:
            # Cut at the largest jump in the upper half
            cut_idx = half + int(candidate_jumps.argmax())
            n_clusters = n_sub - cut_idx - 1
            n_clusters = max(2, min(n_clusters, n_sub - 1))
        else:
            # No clear elbow — fall back to a reasonable fraction
            n_clusters = max(2, n_sub // 4)
        print(f"  Stage 2: merging {n_sub} sub-clusters → {n_clusters} "
              f"(auto-detected, jump_factor={MERGE_JUMP_FACTOR})")

    agg = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="complete"
    )
    meta_labels = agg.fit_predict(cent_dist)  # maps sub-cluster index → person id

    # Map back to original detection labels
    label_to_meta = {rl: int(meta_labels[i]) for i, rl in enumerate(real_labels)}
    new_labels = labels.copy()
    for i, orig in enumerate(labels):
        if orig in label_to_meta:
            new_labels[i] = label_to_meta[orig]
        # noise stays -1

    return new_labels


def cluster_faces(detections: list[FaceDetection]) -> list[ClusterResult]:
    """
    Two-stage clustering pipeline:

    Stage 1 — HDBSCAN (tight epsilon, intentionally over-splits):
      Finds dense sub-clusters in embedding space. With eps=0.36 this
      gives ~99 sub-clusters for a typical multi-person video. We want
      over-splitting here — Stage 2 will correct it.

    Stage 2 — Agglomerative merge on sub-cluster centroids:
      Runs AgglomerativeClustering(n_clusters=N, linkage="complete") on
      the 99×99 centroid distance matrix to produce exactly N people.
      Complete linkage prevents chain-linking (the greedy single-linkage
      mega-cluster collapse). Set KNOWN_N_PERSONS or let auto-detect run.

    Returns list sorted by cluster size (largest first). id==-1 is noise.
    """
    try:
        import hdbscan as _hdbscan
    except ImportError:
        raise ImportError("hdbscan not installed. Run: pip install hdbscan")

    if not detections:
        print("  No detections to cluster.")
        return []

    # Drop embeddings with non-standard dimension
    emb_sizes = [d.embedding.shape[0] for d in detections]
    majority_dim = max(set(emb_sizes), key=emb_sizes.count)
    detections = [d for d in detections if d.embedding.shape[0] == majority_dim]
    if not detections:
        return []

    n = len(detections)
    embs_raw = np.stack([d.embedding for d in detections])
    norms = np.linalg.norm(embs_raw, axis=1, keepdims=True)
    embs_n = embs_raw / np.maximum(norms, 1e-10)

    # ── Stage 1: HDBSCAN ─────────────────────────────────────────
    print(f"\n  Computing {n}×{n} cosine distance matrix ...")
    dist_matrix = _cosine_dist_matrix(embs_raw)

    epsilon = HDBSCAN_EPSILON.get(ACTIVE_ENGINE, 0.36)
    print(f"\n  Stage 1: HDBSCAN on {n} detections")
    print(f"  Engine: {ACTIVE_ENGINE}   eps={epsilon}   "
          f"mcs={HDBSCAN_MIN_CLUSTER_SIZE}   ms={HDBSCAN_MIN_SAMPLES}")

    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="eom",
        metric="precomputed",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(dist_matrix)

    n_sub = len(set(labels) - {-1})
    n_noise_raw = int((labels == -1).sum())
    print(f"  → {n_sub} sub-clusters, {n_noise_raw} noise points")

    # Rescue noise points before Stage 2
    rescue_dist = NOISE_RESCUE_DIST.get(ACTIVE_ENGINE, 0.50)
    if n_noise_raw > 0:
        labels = _rescue_noise(detections, labels, dist_matrix, rescue_dist)

    # ── Stage 2: Agglomerative merge on centroids ─────────────────
    labels = _stage2_merge_subclusters(labels, embs_n, KNOWN_N_PERSONS)

    # Final counts
    n_real = len(set(labels) - {-1})
    n_noise_final = int((labels == -1).sum())
    print(f"\n  → {n_real} unique persons identified   "
          f"{n_noise_final} noise detections")

    # Build ClusterResult objects
    cluster_map: dict[int, ClusterResult] = {}
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
            print(f"  [noise]  {c.size:4d} detections  (outliers)")
        else:
            ts = [f.timestamp_sec for f in c.faces]
            print(f"  Person {c.id:03d}  {c.size:4d} detections  "
                  f"t={min(ts):.1f}s–{max(ts):.1f}s  "
                  f"avg_quality={c.mean_quality:.1f}")

    return clusters


# ══════════════════════════════════════════════════════════════════
#  TUNING GUIDE
# ══════════════════════════════════════════════════════════════════
#
#  RECOMMENDED: If you know the number of people in the video, set:
#    KNOWN_N_PERSONS = 26   (or whatever the count is)
#  This gives exact results every time. No other tuning needed.
#
#  If KNOWN_N_PERSONS = None (auto mode):
#    Too many clusters  → decrease MERGE_JUMP_FACTOR (e.g. 1.8 → 1.5)
#    Too few clusters   → increase MERGE_JUMP_FACTOR (e.g. 1.8 → 2.2)
#
#  If Stage 1 (HDBSCAN) produces very few sub-clusters (<30):
#    → Decrease HDBSCAN_EPSILON[engine] by 0.02 steps
#    → Stage 2 needs enough sub-clusters to work from
#
#  If too many detections are noise:
#    → Lower HDBSCAN_MIN_CLUSTER_SIZE from 3 to 2
#    → Lower HDBSCAN_MIN_SAMPLES from 2 to 1
#    → Increase NOISE_RESCUE_DIST[engine] by 0.05 steps
#
# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
#  CACHE HELPERS
# ══════════════════════════════════════════════════════════════════

def save_detections(detections: list[FaceDetection], cache_path: str | Path) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(detections, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ Saved {len(detections)} detections → {cache_path}")


def load_detections(cache_path: str | Path) -> list[FaceDetection] | None:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        detections = pickle.load(f)
    print(f"  ✓ Loaded {len(detections)} detections from cache → {cache_path}")
    return detections


# ══════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def run_pipeline(
    video_path: str | Path,
    cache_path: str | Path | None = None,
    force_redetect: bool = False,
) -> list[ClusterResult]:
    video_path = Path(video_path)

    if cache_path is None:
        cache_path = video_path.parent / f"{video_path.stem}_detections.pkl"
    cache_path = Path(cache_path)

    print("\n" + "=" * 60)
    print("   STEP 1 — Frame sampling & face detection")
    print("=" * 60)

    detections = None
    if not force_redetect:
        detections = load_detections(cache_path)

    if detections is None:
        detections = sample_and_detect(video_path)
        save_detections(detections, cache_path)
    else:
        print("  (Skipped detection — using cache)")

    print("\n" + "=" * 60)
    print("   STEP 2 — Face clustering")
    print("=" * 60)
    clusters = cluster_faces(detections)

    n_persons = len([c for c in clusters if c.id != -1])
    print("\n" + "=" * 60)
    print(f"   Pipeline complete — {n_persons} unique persons identified")
    print("=" * 60 + "\n")
    return clusters


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        vdir = Path("input_videos")
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        videos = [f for f in vdir.iterdir() if f.suffix.lower() in exts]
        if not videos:
            print("Usage: python face_cluster.py <video_path>")
            sys.exit(1)
        video = sorted(videos)[0]
    else:
        video = Path(sys.argv[1])

    clusters = run_pipeline(video)

    print("\n  CLUSTER SUMMARY")
    print("  " + "-" * 55)
    print(f"  {'ID':>6}  {'Detections':>10}  {'Avg Quality':>11}  {'Time Range':>18}")
    print("  " + "-" * 55)
    for c in clusters:
        label = "noise" if c.id == -1 else str(c.id)
        ts = [f.timestamp_sec for f in c.faces]
        time_range = f"{min(ts):.1f}s – {max(ts):.1f}s"
        print(f"  {label:>6}  {c.size:>10}  {c.mean_quality:>11.1f}  {time_range:>18}")
    print("  " + "-" * 55)
    print(f"\n  → Pass `clusters` list to teammate for steps 3 & 4.\n")