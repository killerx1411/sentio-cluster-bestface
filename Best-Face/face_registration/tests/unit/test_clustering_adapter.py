import base64

import cv2
import numpy as np
import pytest

from app.adapters.clustering_adapter import (
    bbox_xywh_to_xyxy,
    clusters_to_payload,
    encode_crop_bgr,
    face_detection_to_model,
    landmarks_to_model,
    normalize_quality_score,
)
from app.models.request import ClusterPayload


class _FakeFace:
    def __init__(self):
        self.frame_idx = 10
        self.timestamp_sec = 1.5
        self.embedding = np.ones(512, dtype=np.float64) * 0.01
        self.crop_bgr = np.full((112, 112, 3), 128, dtype=np.uint8)
        self.quality_score = 40.0
        self.bbox = (20, 30, 80, 90)
        self.confidence = 0.92
        self.landmarks = [
            [40, 50], [70, 50], [55, 65], [45, 80], [65, 80],
        ]


class _FakeCluster:
    def __init__(self, cid=0):
        self.id = cid
        self.faces = [_FakeFace()]


def test_bbox_xywh_to_xyxy():
    assert bbox_xywh_to_xyxy((10, 20, 30, 40)) == [10, 20, 40, 60]


def test_normalize_quality():
    assert normalize_quality_score(0) == 0.0
    assert normalize_quality_score(80) == 1.0
    assert normalize_quality_score(160) == 1.0


def test_encode_crop_roundtrip():
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    crop[10:50, 10:50] = 200
    b64 = encode_crop_bgr(crop)
    raw = base64.b64decode(b64)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape[:2] == (64, 64)


def test_landmarks_from_insightface_list():
    lmk = landmarks_to_model(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        (112, 112, 3),
    )
    assert lmk.left_eye == [1, 2]
    assert lmk.mouth_right == [9, 10]


def test_face_detection_to_model_fields():
    face = face_detection_to_model(_FakeFace())
    assert face.frame_idx == 10
    assert len(face.embedding) == 512
    assert face.bbox == [20, 30, 100, 120]
    assert 0.0 <= face.quality_score <= 1.0
    assert face.landmarks.left_eye == [40, 50]


def test_clusters_to_payload():
    payload = clusters_to_payload([_FakeCluster(0), _FakeCluster(-1)], "j1", "v1")
    assert isinstance(payload, ClusterPayload)
    assert payload.job_id == "j1"
    assert len(payload.clusters) == 2
    assert payload.clusters[0].id == 0
