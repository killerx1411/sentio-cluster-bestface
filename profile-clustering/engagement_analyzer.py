"""
engagement_analyzer.py — Step 3b: per-person engagement signal extraction
=========================================================================
Reads cluster results from face_cluster.run_pipeline() plus the original video,
extracts gaze / arousal / posture / texture signals per frame, and produces
an engagement profile per person (cluster id).

Pure signal-extraction library — no web server, no Flask, no person_database.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — Library detection (graceful degradation)
# ══════════════════════════════════════════════════════════════════

DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace

    try:
        DeepFace.build_model("Emotion")
    except Exception:
        # Some DeepFace versions load the emotion model lazily on first analyze().
        pass
    DEEPFACE_AVAILABLE = True
    print("   DeepFace (Emotion) loaded")
except Exception as e:
    print(f"   DeepFace not available ({e}) — using texture fallback")

MEDIAPIPE_AVAILABLE = False
_mp_face_mesh = None
_mp_pose = None
try:
    import mediapipe as mp

    _mp_face_mesh = mp.solutions.face_mesh
    _mp_pose = mp.solutions.pose
    MEDIAPIPE_AVAILABLE = True
    print("   MediaPipe (FaceMesh + Pose) loaded")
except Exception as e:
    print(f"   MediaPipe not available ({e}) — gaze/posture signals disabled")

# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — Configuration
# ══════════════════════════════════════════════════════════════════

WEIGHT_ATTENTION = 0.35
WEIGHT_AROUSAL = 0.25
WEIGHT_PARTICIPATION = 0.25
WEIGHT_VITALITY = 0.15

PENALTY_GAZE_AWAY = 0.80
PENALTY_HIGH_STRESS = 0.88
PENALTY_POOR_POSTURE = 0.82
PENALTY_DROWSY = 0.72

EMA_ALPHA = 0.3
EMA_HISTORY_LEN = 8

MIN_CROP_PX = 32
MIN_MODEL_PX = 112
MIN_IRIS_PX = 64
MIN_POSE_ROI_PX = 80

POSE_BODY_EXPAND_FACTOR = 4.0

# MediaPipe FaceMesh landmark indices
_LEFT_IRIS = 468
_RIGHT_IRIS = 473
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263
_NOSE_TIP = 1
_CHIN = 152
_FOREHEAD = 10

# MediaPipe Pose landmark indices
_POSE_NOSE = 0
_POSE_L_SHOULDER = 11
_POSE_R_SHOULDER = 12
_POSE_L_HIP = 23
_POSE_R_HIP = 24
_POSE_L_WRIST = 15
_POSE_R_WRIST = 16
_POSE_L_ELBOW = 13
_POSE_R_ELBOW = 14

# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(np.clip(v, lo, hi))


def _upscale_crop(crop_bgr: np.ndarray, min_side: int = MIN_MODEL_PX) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if h >= min_side and w >= min_side:
        return crop_bgr
    scale = min_side / min(h, w)
    new_w = max(min_side, int(round(w * scale)))
    new_h = max(min_side, int(round(h * scale)))
    return cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def _apply_clahe(crop_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    return _apply_clahe(_upscale_crop(crop_bgr))


def _default_gaze(attention_score: int = 50) -> dict:
    return {
        "gaze_direction": "forward",
        "gaze_horizontal": 0.0,
        "gaze_vertical": 0.0,
        "head_yaw": 0.0,
        "head_pitch": 0.0,
        "eye_openness": 50,
        "eye_contact": False,
        "attention_score": attention_score,
    }


def _default_arousal() -> dict:
    return {
        "dominant_emotion": "unknown",
        "stress_level": "low",
        "emotion_scores": {},
        "arousal_score": 50,
    }


def _default_posture() -> dict:
    return {
        "posture_score": 50.0,
        "posture_quality": "unknown",
        "body_openness": 50.0,
        "energy_level": 50.0,
        "shoulder_symmetry": 50.0,
        "head_forward_tilt": 50.0,
        "spine_alignment": 50.0,
        "movement_energy": 50.0,
    }


def _default_texture() -> dict:
    return {
        "face_brightness": 50.0,
        "facial_tension": 50.0,
        "eye_openness_proxy": 50.0,
        "skin_uniformity": 50.0,
        "vitality_score": 50,
    }


def _head_pose_from_landmarks(
    landmarks: Optional[list], crop_w: int, crop_h: int
) -> tuple[float, float, int]:
    """Estimate yaw (deg), pitch (deg), eye_openness (0-100) from 5-point landmarks."""
    if not landmarks or len(landmarks) < 3:
        return 0.0, 0.0, 50

    le = landmarks[0]
    re = landmarks[1]
    nose = landmarks[2]

    eye_mid_x = (le[0] + re[0]) / 2.0
    eye_mid_y = (le[1] + re[1]) / 2.0
    ied = max(abs(re[0] - le[0]), 1.0)

    yaw = float(np.clip((nose[0] - eye_mid_x) / ied * 35.0, -45.0, 45.0))
    pitch = float(np.clip((nose[1] - eye_mid_y) / ied * 30.0, -35.0, 35.0))

    eye_openness = 50
    if len(landmarks) >= 5:
        mouth_y = (landmarks[3][1] + landmarks[4][1]) / 2.0
        face_h = max(mouth_y - eye_mid_y, 1.0)
        eye_to_nose = abs(nose[1] - eye_mid_y)
        eye_openness = int(_clip(eye_to_nose / face_h * 120.0))

    return yaw, pitch, eye_openness


def _gaze_direction_from_angles(
    gaze_h: float, gaze_v: float, head_yaw: float, head_pitch: float
) -> str:
    combined_h = gaze_h * 0.6 + np.sign(head_yaw) * min(abs(head_yaw) / 30.0, 1.0) * 0.4
    combined_v = gaze_v * 0.6 + np.sign(head_pitch) * min(abs(head_pitch) / 25.0, 1.0) * 0.4

    if abs(combined_h) < 0.25 and abs(combined_v) < 0.25:
        return "forward"
    if abs(combined_h) >= abs(combined_v):
        return "left" if combined_h < 0 else "right"
    return "up" if combined_v < 0 else "down"


def _compute_attention_score(
    direction: str,
    head_yaw: float,
    head_pitch: float,
    eye_openness: int,
    eye_contact: bool,
) -> int:
    forward_bonus = 40 if direction == "forward" else 5
    head_bonus = max(0.0, 30.0 - abs(head_yaw) * 0.8 - abs(head_pitch) * 0.6)
    eye_bonus = min(30.0, eye_openness * 0.30)
    attention = _clip(forward_bonus + head_bonus + eye_bonus)
    gaze_bonus = 12 if eye_contact else 0
    gaze_penalty = 0 if direction in ("forward", "camera") else -18
    return int(round(_clip(attention + gaze_bonus + gaze_penalty)))


def _head_pose_from_mesh(lms, w: int, h: int) -> tuple[float, float]:
    """Estimate head yaw/pitch from FaceMesh 3D-ish landmarks via solvePnP."""
    try:
        image_points = np.array(
            [
                (lms[_NOSE_TIP].x * w, lms[_NOSE_TIP].y * h),
                (lms[_CHIN].x * w, lms[_CHIN].y * h),
                (lms[_LEFT_EYE_OUTER].x * w, lms[_LEFT_EYE_OUTER].y * h),
                (lms[_RIGHT_EYE_OUTER].x * w, lms[_RIGHT_EYE_OUTER].y * h),
                (lms[_FOREHEAD].x * w, lms[_FOREHEAD].y * h),
            ],
            dtype=np.float64,
        )
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (0.0, 33.0, -25.0),
            ],
            dtype=np.float64,
        )
        focal = w
        cam_matrix = np.array(
            [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1.0]], dtype=np.float64
        )
        dist = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, _tvec = cv2.solvePnP(
            model_points, image_points, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        if sy < 1e-6:
            pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        else:
            pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
            yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        return float(yaw), float(pitch)
    except Exception:
        return 0.0, 0.0


def _eye_openness_from_mesh(lms) -> int:
    try:
        left_h = abs(lms[159].y - lms[145].y)
        right_h = abs(lms[386].y - lms[374].y)
        left_w = abs(lms[133].x - lms[33].x) + 1e-6
        right_w = abs(lms[362].x - lms[263].x) + 1e-6
        left_ratio = left_h / left_w
        right_ratio = right_h / right_w
        ratio = (left_ratio + right_ratio) / 2.0
        return int(_clip(ratio * 400.0))
    except Exception:
        return 50


def _iris_gaze_from_mesh(lms) -> tuple[float, float, bool]:
    """Return gaze_horizontal, gaze_vertical (-1..1), eye_contact."""
    try:
        def _iris_offset(inner_idx, outer_idx, iris_idx):
            inner = lms[inner_idx]
            outer = lms[outer_idx]
            iris = lms[iris_idx]
            eye_w = outer.x - inner.x
            if abs(eye_w) < 1e-6:
                return 0.0
            center = (inner.x + outer.x) / 2.0
            return float(np.clip((iris.x - center) / (eye_w / 2.0), -1.0, 1.0))

        def _iris_v_offset(top_idx, bottom_idx, iris_idx):
            top = lms[top_idx]
            bottom = lms[bottom_idx]
            iris = lms[iris_idx]
            eye_h = bottom.y - top.y
            if abs(eye_h) < 1e-6:
                return 0.0
            center = (top.y + bottom.y) / 2.0
            return float(np.clip((iris.y - center) / (eye_h / 2.0), -1.0, 1.0))

        gh_l = _iris_offset(_LEFT_EYE_INNER, _LEFT_EYE_OUTER, _LEFT_IRIS)
        gh_r = _iris_offset(_RIGHT_EYE_INNER, _RIGHT_EYE_OUTER, _RIGHT_IRIS)
        gv_l = _iris_v_offset(159, 145, _LEFT_IRIS)
        gv_r = _iris_v_offset(386, 374, _RIGHT_IRIS)

        gaze_h = (gh_l + gh_r) / 2.0
        gaze_v = (gv_l + gv_r) / 2.0
        eye_contact = abs(gaze_h) < 0.35 and abs(gaze_v) < 0.35
        return gaze_h, gaze_v, eye_contact
    except Exception:
        return 0.0, 0.0, False


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — Signal extractors
# ══════════════════════════════════════════════════════════════════


def extract_gaze_signal(
    face_crop_bgr: np.ndarray,
    landmarks: Optional[list] = None,
) -> dict:
    """Gaze / head-pose / eye-openness from MediaPipe FaceMesh (or landmark fallback)."""
    try:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return _default_gaze()

        orig_h, orig_w = face_crop_bgr.shape[:2]
        if orig_h < MIN_CROP_PX or orig_w < MIN_CROP_PX:
            return _default_gaze()

        crop = _preprocess_crop(face_crop_bgr)
        h, w = crop.shape[:2]
        use_iris = orig_h >= MIN_IRIS_PX and orig_w >= MIN_IRIS_PX

        if not MEDIAPIPE_AVAILABLE or _mp_face_mesh is None:
            yaw, pitch, eye_open = _head_pose_from_landmarks(landmarks, w, h)
            direction = _gaze_direction_from_angles(0.0, 0.0, yaw, pitch)
            attention = _compute_attention_score(direction, yaw, pitch, eye_open, False)
            return {
                "gaze_direction": direction,
                "gaze_horizontal": 0.0,
                "gaze_vertical": 0.0,
                "head_yaw": yaw,
                "head_pitch": pitch,
                "eye_openness": eye_open,
                "eye_contact": direction == "forward",
                "attention_score": attention,
            }

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with _mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=use_iris,
            min_detection_confidence=0.4,
        ) as mesh:
            result = mesh.process(rgb)

        if not result.multi_face_landmarks:
            yaw, pitch, eye_open = _head_pose_from_landmarks(landmarks, w, h)
            direction = _gaze_direction_from_angles(0.0, 0.0, yaw, pitch)
            attention = _compute_attention_score(direction, yaw, pitch, eye_open, False)
            return {
                "gaze_direction": direction,
                "gaze_horizontal": 0.0,
                "gaze_vertical": 0.0,
                "head_yaw": yaw,
                "head_pitch": pitch,
                "eye_openness": eye_open,
                "eye_contact": False,
                "attention_score": attention,
            }

        lms = result.multi_face_landmarks[0].landmark
        head_yaw, head_pitch = _head_pose_from_mesh(lms, w, h)

        if use_iris:
            gaze_h, gaze_v, eye_contact = _iris_gaze_from_mesh(lms)
            eye_open = _eye_openness_from_mesh(lms)
        else:
            yaw, pitch, eye_open = _head_pose_from_landmarks(landmarks, w, h)
            if abs(yaw) < 0.01 and abs(pitch) < 0.01:
                head_yaw, head_pitch = yaw, pitch
            gaze_h, gaze_v = 0.0, 0.0
            eye_contact = abs(head_yaw) < 15 and abs(head_pitch) < 15

        direction = _gaze_direction_from_angles(gaze_h, gaze_v, head_yaw, head_pitch)
        attention = _compute_attention_score(
            direction, head_yaw, head_pitch, eye_open, eye_contact
        )

        return {
            "gaze_direction": direction,
            "gaze_horizontal": float(gaze_h),
            "gaze_vertical": float(gaze_v),
            "head_yaw": float(head_yaw),
            "head_pitch": float(head_pitch),
            "eye_openness": int(eye_open),
            "eye_contact": bool(eye_contact),
            "attention_score": attention,
        }
    except Exception:
        return _default_gaze()


def extract_arousal_signal(face_crop_bgr: np.ndarray) -> dict:
    """Cognitive arousal from DeepFace emotions (or texture fallback)."""
    try:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return _default_arousal()

        h, w = face_crop_bgr.shape[:2]
        if h < MIN_CROP_PX or w < MIN_CROP_PX:
            return _default_arousal()

        crop = _preprocess_crop(face_crop_bgr)

        if DEEPFACE_AVAILABLE:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(
                rgb,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            scores = {k: float(v) for k, v in result.get("emotion", {}).items()}
            happy = scores.get("happy", 0.0)
            surprise = scores.get("surprise", 0.0)
            angry = scores.get("angry", 0.0)
            sad = scores.get("sad", 0.0)
            fear = scores.get("fear", 0.0)
            neutral = scores.get("neutral", 0.0)
            disgust = scores.get("disgust", 0.0)

            active = happy + surprise * 0.8 + angry * 0.3
            passive = sad * 0.7 + fear * 0.5 + neutral * 0.4
            disgust_pen = disgust * 0.6
            arousal_raw = _clip(50.0 + (active - passive - disgust_pen) * 0.4)

            dominant = max(scores, key=scores.get) if scores else "unknown"
            stress_sum = angry + fear + disgust
            if stress_sum > 45:
                stress_level = "high"
            elif stress_sum > 20:
                stress_level = "moderate"
            else:
                stress_level = "low"

            return {
                "dominant_emotion": dominant,
                "stress_level": stress_level,
                "emotion_scores": scores,
                "arousal_score": int(round(arousal_raw)),
            }

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
        brightness = gray.mean()
        arousal_raw = _clip(lap_var / 8.0 * 0.4 + brightness / 2.55 * 0.6)
        return {
            "dominant_emotion": "unknown",
            "stress_level": "low",
            "emotion_scores": {},
            "arousal_score": int(round(arousal_raw)),
        }
    except Exception:
        return _default_arousal()


def extract_posture_signal(
    full_frame_bgr: np.ndarray, face_bbox: tuple
) -> dict:
    """Body posture / openness from MediaPipe Pose on an expanded torso ROI."""
    try:
        if not MEDIAPIPE_AVAILABLE or _mp_pose is None:
            return _default_posture()

        if full_frame_bgr is None or full_frame_bgr.size == 0:
            return _default_posture()

        fh, fw = full_frame_bgr.shape[:2]
        x, y, w, h = [int(v) for v in face_bbox]
        if w < 1 or h < 1:
            return _default_posture()

        expand = int(POSE_BODY_EXPAND_FACTOR * h)
        x1 = max(0, x - w // 2)
        y1 = max(0, y)
        x2 = min(fw, x + w + w // 2)
        y2 = min(fh, y + h + expand)

        roi_w = x2 - x1
        roi_h = y2 - y1
        if roi_w < MIN_POSE_ROI_PX or roi_h < MIN_POSE_ROI_PX:
            return _default_posture()

        roi = full_frame_bgr[y1:y2, x1:x2]
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        with _mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.4,
        ) as pose:
            result = pose.process(rgb)

        if not result.pose_landmarks:
            return _default_posture()

        lms = result.pose_landmarks.landmark

        def _lm(idx):
            p = lms[idx]
            return p.x, p.y, p.visibility

        nose = _lm(_POSE_NOSE)
        ls = _lm(_POSE_L_SHOULDER)
        rs = _lm(_POSE_R_SHOULDER)
        lh = _lm(_POSE_L_HIP)
        rh = _lm(_POSE_R_HIP)
        lw = _lm(_POSE_L_WRIST)
        rw = _lm(_POSE_R_WRIST)
        le = _lm(_POSE_L_ELBOW)
        re = _lm(_POSE_R_ELBOW)

        key_pts = [nose, ls, rs, lh, rh]
        if any(p[2] < 0.3 for p in key_pts):
            return _default_posture()

        shoulder_symmetry = _clip(100.0 - abs(ls[1] - rs[1]) * 400.0)
        mid_shoulder_x = (ls[0] + rs[0]) / 2.0
        mid_hip_x = (lh[0] + rh[0]) / 2.0
        head_forward_tilt = _clip(100.0 - abs(nose[0] - mid_shoulder_x) * 300.0)
        spine_alignment = _clip(100.0 - abs(mid_shoulder_x - mid_hip_x) * 350.0)

        shoulder_width = abs(ls[0] - rs[0])
        hip_width = abs(lh[0] - rh[0])
        body_openness = _clip(shoulder_width / (hip_width + 1e-6) * 60.0)

        arm_vals = [1.0 - lw[1], 1.0 - rw[1], 1.0 - le[1], 1.0 - re[1]]
        arm_activity = float(np.mean(arm_vals)) * 100.0
        movement_energy = _clip(arm_activity * 1.4)

        posture_score = _clip(
            shoulder_symmetry * 0.25
            + head_forward_tilt * 0.20
            + spine_alignment * 0.25
            + body_openness * 0.15
            + movement_energy * 0.15
        )

        if posture_score >= 78:
            posture_quality = "excellent"
            energy_base = 75.0
        elif posture_score >= 60:
            posture_quality = "good"
            energy_base = 60.0
        elif posture_score >= 42:
            posture_quality = "fair"
            energy_base = 45.0
        else:
            posture_quality = "poor"
            energy_base = 25.0

        energy_level = _clip(energy_base + movement_energy * 0.20)

        return {
            "posture_score": float(posture_score),
            "posture_quality": posture_quality,
            "body_openness": float(body_openness),
            "energy_level": float(energy_level),
            "shoulder_symmetry": float(shoulder_symmetry),
            "head_forward_tilt": float(head_forward_tilt),
            "spine_alignment": float(spine_alignment),
            "movement_energy": float(movement_energy),
        }
    except Exception:
        return _default_posture()


def extract_texture_signal(face_crop_bgr: np.ndarray) -> dict:
    """Vitality proxies from OpenCV texture / brightness analysis."""
    try:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return _default_texture()

        h, w = face_crop_bgr.shape[:2]
        if h < MIN_CROP_PX or w < MIN_CROP_PX:
            return _default_texture()

        crop = cv2.resize(face_crop_bgr, (96, 96), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        face_brightness = _clip(gray.mean() / 2.55)
        facial_tension = _clip(np.std(cv2.Laplacian(gray, cv2.CV_32F)) / 3.0)

        y0 = int(96 * 0.20)
        y1 = int(96 * 0.55)
        eye_band = gray[y0:y1, :]
        eye_openness_proxy = _clip(eye_band.mean() / 2.55) if eye_band.size else 50.0

        skin_uniformity = 100.0 - _clip(np.std(hsv[:, :, 1]) / 1.28)

        vitality_score = int(
            round(
                _clip(
                    face_brightness * 0.40
                    + skin_uniformity * 0.25
                    + (100.0 - facial_tension) * 0.35
                )
            )
        )

        return {
            "face_brightness": float(face_brightness),
            "facial_tension": float(facial_tension),
            "eye_openness_proxy": float(eye_openness_proxy),
            "skin_uniformity": float(skin_uniformity),
            "vitality_score": vitality_score,
        }
    except Exception:
        return _default_texture()


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — Fusion
# ══════════════════════════════════════════════════════════════════


def compute_engagement_score(
    gaze: dict,
    arousal: dict,
    posture: dict,
    texture: dict,
    ema_history: list[float] | None = None,
) -> dict:
    attention_component = float(gaze.get("attention_score", 50))
    arousal_component = float(arousal.get("arousal_score", 50))
    participation_component = float(
        np.clip(
            posture.get("body_openness", 50) * 0.30
            + posture.get("energy_level", 50) * 0.40
            + posture.get("head_forward_tilt", 50) * 0.30,
            0,
            100,
        )
    )
    vitality_component = float(texture.get("vitality_score", 50))

    raw = (
        attention_component * WEIGHT_ATTENTION
        + arousal_component * WEIGHT_AROUSAL
        + participation_component * WEIGHT_PARTICIPATION
        + vitality_component * WEIGHT_VITALITY
    )

    penalties: list[str] = []
    multiplier = 1.0

    gaze_dir = gaze.get("gaze_direction", "forward")
    head_yaw = abs(gaze.get("head_yaw", 0))
    if gaze_dir not in ("forward", "camera") and head_yaw > 20:
        penalties.append("gaze_away")
        multiplier *= PENALTY_GAZE_AWAY

    if arousal.get("stress_level") == "high":
        penalties.append("high_stress")
        multiplier *= PENALTY_HIGH_STRESS

    if posture.get("posture_quality") == "poor":
        penalties.append("poor_posture")
        multiplier *= PENALTY_POOR_POSTURE

    if gaze.get("eye_openness", 50) < 25:
        penalties.append("drowsy")
        multiplier *= PENALTY_DROWSY

    raw_penalized = float(np.clip(raw * multiplier, 0, 100))

    if ema_history:
        last_ema = ema_history[-1]
        smoothed = EMA_ALPHA * raw_penalized + (1.0 - EMA_ALPHA) * last_ema
    else:
        smoothed = raw_penalized

    score = int(round(np.clip(smoothed, 0, 100)))
    level = "high" if score >= 70 else "moderate" if score >= 45 else "low"

    return {
        "engagement_score": score,
        "engagement_raw": int(round(raw_penalized)),
        "engagement_level": level,
        "engagement_components": {
            "attention": round(attention_component, 1),
            "arousal": round(arousal_component, 1),
            "participation": round(participation_component, 1),
            "vitality": round(vitality_component, 1),
        },
        "engagement_penalties": penalties,
        "gaze": gaze,
        "arousal": arousal,
        "posture": posture,
        "texture": texture,
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — Per-person aggregator
# ══════════════════════════════════════════════════════════════════


def analyze_person_engagement(
    cluster,
    video_path: str,
    verbose: bool = True,
) -> dict:
    """
    Run the four-signal engagement pipeline over every FaceDetection in a cluster.
    """
    faces = sorted(cluster.faces, key=lambda f: f.frame_idx)
    ema_history: list[float] = []
    frame_series: list[dict] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose:
            print(f"   Warning: cannot open video {video_path} — posture signals disabled")

    for i, face in enumerate(faces):
        full_frame = None
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, face.frame_idx)
            ok, full_frame = cap.read()
            if not ok:
                full_frame = None

        crop = face.crop_bgr
        if crop is None or crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        if ch < MIN_CROP_PX or cw < MIN_CROP_PX:
            continue

        gaze = extract_gaze_signal(crop, landmarks=getattr(face, "landmarks", None))
        arousal = extract_arousal_signal(crop)

        if full_frame is not None:
            posture = extract_posture_signal(full_frame, face.bbox)
        else:
            posture = _default_posture()

        texture = extract_texture_signal(crop)
        result = compute_engagement_score(gaze, arousal, posture, texture, ema_history)

        ema_history.append(float(result["engagement_raw"]))
        if len(ema_history) > EMA_HISTORY_LEN:
            ema_history.pop(0)

        frame_series.append(
            {
                "frame_idx": int(face.frame_idx),
                "timestamp_sec": float(face.timestamp_sec),
                "engagement_score": result["engagement_score"],
                "engagement_level": result["engagement_level"],
                "engagement_components": result["engagement_components"],
                "engagement_penalties": result["engagement_penalties"],
            }
        )

        if verbose and (i + 1) % 10 == 0:
            print(f"      cluster {cluster.id:03d}: {i + 1}/{len(faces)} frames processed")

    if cap.isOpened():
        cap.release()

    if not frame_series:
        return {
            "cluster_id": int(cluster.id),
            "total_frames": 0,
            "avg_engagement": 0,
            "peak_engagement": 0,
            "low_engagement": 0,
            "engagement_level": "low",
            "trend": "stable",
            "component_means": {
                "attention": 0.0,
                "arousal": 0.0,
                "participation": 0.0,
                "vitality": 0.0,
            },
            "penalty_frequency": {},
            "frame_series": [],
        }

    scores = [f["engagement_score"] for f in frame_series]
    avg_engagement = int(round(np.mean(scores)))
    peak_engagement = int(max(scores))
    low_engagement = int(min(scores))
    engagement_level = (
        "high" if avg_engagement >= 70 else "moderate" if avg_engagement >= 45 else "low"
    )

    n = len(scores)
    third = max(n // 3, 1)
    first_third_avg = float(np.mean(scores[:third]))
    last_third_avg = float(np.mean(scores[-third:]))
    if last_third_avg > first_third_avg + 5:
        trend = "improving"
    elif first_third_avg > last_third_avg + 5:
        trend = "declining"
    else:
        trend = "stable"

    component_means = {}
    for key in ("attention", "arousal", "participation", "vitality"):
        vals = [f["engagement_components"][key] for f in frame_series]
        component_means[key] = round(float(np.mean(vals)), 1)

    penalty_counts: dict[str, int] = {}
    for f in frame_series:
        for p in f["engagement_penalties"]:
            penalty_counts[p] = penalty_counts.get(p, 0) + 1
    penalty_frequency = {
        k: round(v / n, 3) for k, v in penalty_counts.items()
    }

    return {
        "cluster_id": int(cluster.id),
        "total_frames": n,
        "avg_engagement": avg_engagement,
        "peak_engagement": peak_engagement,
        "low_engagement": low_engagement,
        "engagement_level": engagement_level,
        "trend": trend,
        "component_means": component_means,
        "penalty_frequency": penalty_frequency,
        "frame_series": frame_series,
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 — Public entry point
# ══════════════════════════════════════════════════════════════════


def _top_penalty(penalty_frequency: dict) -> str:
    if not penalty_frequency:
        return "—"
    top = max(penalty_frequency, key=penalty_frequency.get)
    pct = int(round(penalty_frequency[top] * 100))
    return f"{top} ({pct}%)"


def _print_summary(results: dict[int, dict]) -> None:
    print("\n        ENGAGEMENT SUMMARY")
    print("        " + "─" * 62)
    print(f"        {'Person':<8}{'Avg%':<6}{'Peak%':<7}{'Level':<10}{'Trend':<11}{'Top Penalty'}")
    print("        " + "─" * 62)
    for cid in sorted(results):
        p = results[cid]
        print(
            f"        {cid:03d}     "
            f"{p['avg_engagement']:<6}"
            f"{p['peak_engagement']:<7}"
            f"{p['engagement_level']:<10}"
            f"{p['trend']:<11}"
            f"{_top_penalty(p.get('penalty_frequency', {}))}"
        )
    print("        " + "─" * 62)


def _save_engagement_json(results: dict[int, dict], video_path: str) -> Path:
    out_path = Path(video_path).with_name(f"{Path(video_path).stem}_engagement.json")
    serializable = {str(k): v for k, v in results.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return out_path


def run_engagement_pipeline(
    clusters: list,
    video_path: str,
    exclude_noise: bool = True,
    verbose: bool = True,
) -> dict[int, dict]:
    """
    Run analyze_person_engagement() for every real cluster.

    Returns dict mapping cluster_id -> engagement profile.
    Saves <video_stem>_engagement.json alongside the video.
    """
    video_path = str(video_path)
    results: dict[int, dict] = {}

    eligible = [
        c for c in clusters if (not exclude_noise or c.id != -1)
    ]

    if verbose:
        print(f"\n=== Engagement analysis: {Path(video_path).name} ===")
        print(f"    {len(eligible)} person(s) to analyze")

    for cluster in eligible:
        if verbose:
            print(f"\n   Person {cluster.id:03d}  ({len(cluster.faces)} detections)")
        try:
            profile = analyze_person_engagement(cluster, video_path, verbose=verbose)
            results[int(cluster.id)] = profile
        except Exception as e:
            if verbose:
                print(f"   Error analyzing cluster {cluster.id}: {e}")
                traceback.print_exc()

    if verbose:
        _print_summary(results)

    try:
        out_path = _save_engagement_json(results, video_path)
        if verbose:
            print(f"\n  Engagement JSON → {out_path}\n")
    except Exception as e:
        if verbose:
            print(f"   Warning: failed to save engagement JSON: {e}")

    return results
