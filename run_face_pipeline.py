#!/usr/bin/env python3
"""
Combined face pipeline — all 3 steps in one Python process.

  1. Face detection   — profile-clustering/face_cluster.py
  2. Face clustering  — HDBSCAN + agglomerative merge
  3. Best-face pick   — Best-Face/face_registration

IMPORTANT — two venvs (recommended setup):
  profile-clustering uses clustervenv; Best-Face uses bestfaceharsh.
  Those envs have different dependencies. Do NOT use this script unless
  BOTH dependency sets are installed in the SAME activated venv.

  Use instead (from repo root):
    .\\run_face_pipeline.ps1

  Or run manually:
    # Step 1+2 (clustervenv):
    cd profile-clustering
    .\\clustervenv\\Scripts\\Activate.ps1
    python export_for_bestface.py input_videos/longvid.mp4 \\
        --cache input_videos/longvid_detections.pkl \\
        --out input_videos/longvid_clusters.json

    # Step 3 (bestfaceharsh):
    cd ..\\Best-Face\\face_registration
    ..\\..\\Best-Face\\bestfaceharsh\\Scripts\\Activate.ps1
    python scripts/register_from_json.py \\
        ..\\..\\profile-clustering\\input_videos\\longvid_clusters.json

Single-venv usage (only when one env has all deps):
  cd checkidk
  python run_face_pipeline.py profile-clustering/input_videos/longvid.mp4 \\
      --cache profile-clustering/input_videos/longvid_detections.pkl

  python run_face_pipeline.py profile-clustering/input_videos/longvid.mp4 \\
      --cache profile-clustering/input_videos/longvid_detections.pkl \\
      --known-n 26 --json

Output:
  Best-Face/face_registration/database/person_0001/best_face.jpg
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
    parser = argparse.ArgumentParser(
        description="Face detect → cluster → best face (single-process; needs one venv with all deps)",
        epilog="For separate venvs use: .\\run_face_pipeline.ps1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument("--job-id", default=None, help="Job id (default: job_<stem>)")
    parser.add_argument(
        "--known-n",
        type=int,
        default=None,
        help="Known person count for clustering (overrides face_cluster.KNOWN_N_PERSONS)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Detections .pkl cache, e.g. profile-clustering/input_videos/longvid_detections.pkl",
    )
    parser.add_argument(
        "--force-redetect",
        action="store_true",
        help="Ignore detection cache and re-run face detection",
    )
    parser.add_argument("--json", action="store_true", help="Print registration report as JSON")
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 1

    cache = args.cache.resolve() if args.cache else None
    if cache and not cache.is_file() and not args.force_redetect:
        print(f"Warning: cache not found ({cache}); detection will run from scratch.", file=sys.stderr)

    print(f"\n=== Face pipeline (single-process): {video.name} ===")
    if cache:
        print(f"    Cache: {cache}\n")

    report = run_video_pipeline(
        video,
        job_id=args.job_id,
        cache_path=str(cache) if cache else None,
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
