import base64
import cv2
import numpy as np
import pytest
from app.models.request import FaceDetection, Landmarks, Cluster, ClusterPayload


_DETERMINISTIC_IMAGE_CACHE: str | None = None


def _make_test_image(width=250, height=250) -> str:
    global _DETERMINISTIC_IMAGE_CACHE
    if _DETERMINISTIC_IMAGE_CACHE is not None:
        return _DETERMINISTIC_IMAGE_CACHE
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    _DETERMINISTIC_IMAGE_CACHE = base64.b64encode(buf).decode()
    return _DETERMINISTIC_IMAGE_CACHE


def make_face(
    frame_idx=0,
    timestamp_sec=0.0,
    quality_score=0.8,
    confidence=0.9,
    bbox=None,
    left_eye=None,
    right_eye=None,
    nose=None,
    crop=None,
):
    if bbox is None:
        bbox = [100, 80, 200, 180]
    if left_eye is None:
        left_eye = [130, 110]
    if right_eye is None:
        right_eye = [170, 110]
    if nose is None:
        nose = [150, 130]
    if crop is None:
        crop = _make_test_image()
    return FaceDetection(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        embedding=[0.1] * 512,
        crop_bgr=crop,
        quality_score=quality_score,
        bbox=bbox,
        confidence=confidence,
        landmarks=Landmarks(
            left_eye=left_eye,
            right_eye=right_eye,
            nose=nose,
            mouth_left=[135, 150],
            mouth_right=[165, 150],
        ),
    )


def make_cluster(cluster_id=0, faces=None):
    if faces is None:
        faces = [make_face()]
    return Cluster(id=cluster_id, faces=faces)


def make_payload(clusters=None, job_id="job_001", video_id="vid_001"):
    if clusters is None:
        clusters = [make_cluster()]
    return ClusterPayload(job_id=job_id, video_id=video_id, clusters=clusters)
