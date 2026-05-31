import uuid
import numpy as np
import cv2
import base64
from app.core.config import settings
from app.models.request import FaceDetection


def score_frontality(face: FaceDetection) -> float:
    lx, ly = face.landmarks.left_eye
    rx, ry = face.landmarks.right_eye
    nx, ny = face.landmarks.nose
    eye_span = abs(rx - lx) + 1e-5
    eye_y_diff = abs(ly - ry)
    nose_dev = abs(nx - (lx + rx) / 2)
    return max(0.0, min(1.0, 1.0 - (eye_y_diff + nose_dev) / eye_span))


def score_size(face: FaceDetection) -> float:
    x1, y1, x2, y2 = face.bbox
    face_area = (x2 - x1) * (y2 - y1)
    return max(0.0, min(1.0, face_area / settings.REF_AREA))


def score_illumination(face: FaceDetection) -> float:
    crop = _decode_crop(face.crop_bgr)
    if crop is None:
        return 0.0
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    mu = float(np.mean(l_channel))
    sigma = float(np.std(l_channel))
    brightness_score = 1.0 - abs(mu - 127.0) / 127.0
    contrast_score = max(0.0, min(1.0, sigma / 60.0))
    return 0.6 * brightness_score + 0.4 * contrast_score


def score_composite(face: FaceDetection) -> float:
    f_q = face.quality_score
    f_f = score_frontality(face)
    f_s = score_size(face)
    f_c = face.confidence
    f_l = score_illumination(face)
    return (
        settings.W_QUALITY * f_q
        + settings.W_FRONT * f_f
        + settings.W_SIZE * f_s
        + settings.W_CONF * f_c
        + settings.W_ILLUM * f_l
    )


def score_all(faces: list[FaceDetection]) -> list[FaceDetection]:
    for face in faces:
        face._score = score_composite(face)
    return faces


def select_best(faces: list[FaceDetection]) -> FaceDetection | None:
    if not faces:
        return None
    return max(faces, key=lambda f: (getattr(f, "_score", 0.0), -f.frame_idx))


def generate_identity(cluster_id: int, person_counter: int, total_clusters: int) -> tuple[str, str]:
    pad_width = max(4, len(str(total_clusters)))
    person_id = f"person_{str(person_counter).zfill(pad_width)}"
    return person_id, str(uuid.uuid4())


def _decode_crop(crop_bgr_b64: str) -> np.ndarray | None:
    try:
        img_bytes = base64.b64decode(crop_bgr_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
