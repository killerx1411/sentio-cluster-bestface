"""
Convert profile-clustering output (FaceDetection / ClusterResult dataclasses)
into Best-Face API models (Pydantic ClusterPayload).
"""
from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from app.models.request import Cluster, ClusterPayload, FaceDetection, Landmarks

# Clustering quality scores are Laplacian-based and often >> 1.0; Best-Face expects [0, 1].
QUALITY_NORM_DIVISOR = 80.0


def normalize_quality_score(raw: float) -> float:
    return max(0.0, min(1.0, float(raw) / QUALITY_NORM_DIVISOR))


def bbox_xywh_to_xyxy(bbox: tuple | list) -> list[int]:
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]


def encode_crop_bgr(crop: np.ndarray, jpeg_quality: int = 90) -> str:
    if crop is None or crop.size == 0:
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("failed to encode face crop")
    return base64.b64encode(buf).decode("ascii")


def landmarks_to_model(
    landmarks: Any,
    crop_shape: tuple[int, ...],
) -> Landmarks:
    h, w = crop_shape[:2]

    if landmarks and len(landmarks) >= 5:
        pts = landmarks[:5]
        return Landmarks(
            left_eye=[int(pts[0][0]), int(pts[0][1])],
            right_eye=[int(pts[1][0]), int(pts[1][1])],
            nose=[int(pts[2][0]), int(pts[2][1])],
            mouth_left=[int(pts[3][0]), int(pts[3][1])],
            mouth_right=[int(pts[4][0]), int(pts[4][1])],
        )

    return Landmarks(
        left_eye=[w // 3, h // 3],
        right_eye=[2 * w // 3, h // 3],
        nose=[w // 2, h // 2],
        mouth_left=[w // 3, 2 * h // 3],
        mouth_right=[2 * w // 3, 2 * h // 3],
    )


def face_detection_to_model(face: Any) -> FaceDetection:
    crop = face.crop_bgr
    if not isinstance(crop, np.ndarray):
        crop = np.asarray(crop)

    emb = face.embedding
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    else:
        emb = list(emb)

    return FaceDetection(
        frame_idx=int(face.frame_idx),
        timestamp_sec=float(face.timestamp_sec),
        embedding=[float(x) for x in emb],
        crop_bgr=encode_crop_bgr(crop),
        quality_score=normalize_quality_score(face.quality_score),
        bbox=bbox_xywh_to_xyxy(face.bbox),
        confidence=float(face.confidence),
        landmarks=landmarks_to_model(face.landmarks, crop.shape),
    )


def cluster_result_to_cluster(cluster: Any) -> Cluster:
    return Cluster(
        id=int(cluster.id),
        faces=[face_detection_to_model(f) for f in cluster.faces],
    )


def clusters_to_payload(
    clusters: list,
    job_id: str,
    video_id: str,
) -> ClusterPayload:
    return ClusterPayload(
        job_id=job_id,
        video_id=video_id,
        clusters=[cluster_result_to_cluster(c) for c in clusters],
    )
