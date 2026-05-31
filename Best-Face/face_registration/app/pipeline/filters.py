import numpy as np
from app.core.config import settings
from app.models.request import Cluster, FaceDetection


def drop_noise_clusters(clusters: list[Cluster]) -> list[Cluster]:
    return [c for c in clusters if c.id != -1]


def pre_filter_face(face: FaceDetection) -> bool:
    x1, y1, x2, y2 = face.bbox
    min_side = min(x2 - x1, y2 - y1)
    if min_side < settings.MIN_FACE_PX:
        return False
    if face.confidence < settings.MIN_CONF:
        return False
    if face.quality_score < settings.MIN_QUALITY:
        return False
    landmarks = face.landmarks
    lx, ly = landmarks.left_eye
    rx, ry = landmarks.right_eye
    nx, ny = landmarks.nose
    eye_span = abs(rx - lx) + 1e-5
    eye_y_diff = abs(ly - ry)
    nose_dev = abs(nx - (lx + rx) / 2)
    frontality = max(0.0, min(1.0, 1.0 - (eye_y_diff + nose_dev) / eye_span))
    if frontality < settings.MIN_FRONT:
        return False
    crop = _decode_crop(face.crop_bgr)
    if crop is not None:
        illumination = _compute_illumination(crop)
        if illumination < settings.MIN_ILLUM:
            return False
    return True


def pre_filter_faces(faces: list[FaceDetection]) -> list[FaceDetection]:
    filtered = [f for f in faces if pre_filter_face(f)]
    if not filtered:
        return faces
    return filtered


def _decode_crop(crop_bgr_b64: str) -> np.ndarray | None:
    try:
        import base64
        import cv2
        img_bytes = base64.b64decode(crop_bgr_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _compute_illumination(crop_bgr: np.ndarray) -> float:
    import cv2
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    mu = float(np.mean(l_channel))
    sigma = float(np.std(l_channel))
    brightness_score = 1.0 - abs(mu - 127.0) / 127.0
    contrast_score = max(0.0, min(1.0, sigma / 60.0))
    return 0.6 * brightness_score + 0.4 * contrast_score
