#!/usr/bin/env python3
"""
Combined face pipeline (repo root CLI):

  1. Face detection   — profile-clustering/face_cluster.py
  2. Face clustering  — HDBSCAN + agglomerative merge
  3. Best-face pick   — Best-Face/face_registration

Usage:
  python run_face_pipeline.py path/to/video.mp4
  python run_face_pipeline.py path/to/video.mp4 --known-n 26
  python run_face_pipeline.py path/to/video.mp4 --force-redetect
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACE_REG = ROOT / "Best-Face" / "face_registration"
sys.path.insert(0, str(FACE_REG))
os.chdir(FACE_REG)

from app.pipeline.video_pipeline import run_video_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Face detect → cluster → best face")
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument("--job-id", default=None, help="Job id (default: job_<stem>)")
    parser.add_argument("--known-n", type=int, default=None, help="Known person count for clustering")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Detections .pkl cache (e.g. profile-clustering/input_videos/longvid_detections.pkl)",
    )
    parser.add_argument("--force-redetect", action="store_true", help="Ignore detection cache")
    parser.add_argument("--json", action="store_true", help="Print registration report as JSON")
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 1

    print(f"\n=== Face pipeline: {video.name} ===\n")
    report = run_video_pipeline(
        video,
        job_id=args.job_id,
        cache_path=str(args.cache) if args.cache else None,
        force_redetect=args.force_redetect,
        known_n_persons=args.known_n,
    )

    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
    else:
        print(f"Status:     {report.status}")
        print(f"Registered: {report.registered}")
        print(f"Skipped:    {report.skipped}")
        print(f"Noise drop: {report.noise_dropped}")
        print(f"Time:       {report.processing_time_ms:.1f} ms\n")
        for r in report.results:
            print(
                f"  cluster {r.cluster_id:3d} → {r.person_id}  "
                f"score={r.composite_score:.3f}  {r.image_path}  [{r.status}]"
            )

    return 0 if report.registered > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
