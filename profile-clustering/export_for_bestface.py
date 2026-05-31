"""
Steps 1+2 only: load detection cache → cluster → export JSON for Best-Face.

Run inside profile-clustering venv (clustervenv):

  python export_for_bestface.py input_videos/longvid.mp4 \\
      --cache input_videos/longvid_detections.pkl \\
      --out input_videos/longvid_clusters.json
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from face_cluster import run_pipeline

QUALITY_NORM_DIVISOR = 80.0
from face_cluster import FaceDetection
import sys

sys.modules["__main__"].FaceDetection = FaceDetection

def _normalize_quality(raw: float) -> float:
    return max(0.0, min(1.0, float(raw) / QUALITY_NORM_DIVISOR))


def _bbox_xywh_to_xyxy(bbox) -> list[int]:
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]


def _encode_crop(crop: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("failed to encode crop")
    return base64.b64encode(buf).decode("ascii")


def _landmarks(face) -> dict:
    h, w = face.crop_bgr.shape[:2]
    lmk = face.landmarks
    if lmk and len(lmk) >= 5:
        return {
            "left_eye": [int(lmk[0][0]), int(lmk[0][1])],
            "right_eye": [int(lmk[1][0]), int(lmk[1][1])],
            "nose": [int(lmk[2][0]), int(lmk[2][1])],
            "mouth_left": [int(lmk[3][0]), int(lmk[3][1])],
            "mouth_right": [int(lmk[4][0]), int(lmk[4][1])],
        }
    return {
        "left_eye": [w // 3, h // 3],
        "right_eye": [2 * w // 3, h // 3],
        "nose": [w // 2, h // 2],
        "mouth_left": [w // 3, 2 * h // 3],
        "mouth_right": [2 * w // 3, 2 * h // 3],
    }


def _face_to_dict(face) -> dict:
    emb = face.embedding.tolist() if hasattr(face.embedding, "tolist") else list(face.embedding)
    return {
        "frame_idx": int(face.frame_idx),
        "timestamp_sec": float(face.timestamp_sec),
        "embedding": [float(x) for x in emb],
        "crop_bgr": _encode_crop(face.crop_bgr),
        "quality_score": _normalize_quality(face.quality_score),
        "bbox": _bbox_xywh_to_xyxy(face.bbox),
        "confidence": float(face.confidence),
        "landmarks": _landmarks(face),
    }


def clusters_to_payload(clusters, job_id: str, video_id: str) -> dict:
    return {
        "job_id": job_id,
        "video_id": video_id,
        "clusters": [
            {"id": int(c.id), "faces": [_face_to_dict(f) for f in c.faces]}
            for c in clusters
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cluster faces and export JSON for Best-Face registration"
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument(
        "--cache",
        default=None,
        help="Detections .pkl cache (e.g. input_videos/longvid_detections.pkl)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <video_stem>_clusters.json next to video)",
    )
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--force-redetect", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else video_path.with_name(f"{video_path.stem}_clusters.json")
    job_id = args.job_id or f"job_{video_path.stem}"

    print(f"\n=== Clustering: {video_path.name} ===")
    if args.cache:
        print(f"    Cache: {args.cache}")

    clusters = run_pipeline(
        video_path,
        cache_path=args.cache,
        force_redetect=args.force_redetect,
    )

    payload = clusters_to_payload(clusters, job_id, video_path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    n_persons = len([c for c in clusters if c.id != -1])
    print(f"\n  {n_persons} persons → {out_path}")
    print("  Next: run register_from_json.py in the Best-Face venv.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
