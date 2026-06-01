#!/usr/bin/env python3
"""
End-to-end test: face_cluster.run_pipeline → engagement_analyzer.run_engagement_pipeline.

Run from profile-clustering/:
  python test_engagement.py input_videos/longvid.mp4
  python test_engagement.py input_videos/longvid.mp4 --cache input_videos/longvid_detections.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Pickle caches saved by `python face_cluster.py` reference __main__.FaceDetection.
import face_cluster

sys.modules["__main__"].FaceDetection = face_cluster.FaceDetection

from engagement_analyzer import run_engagement_pipeline  # noqa: E402
from face_cluster import run_pipeline  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run face clustering then per-person engagement analysis",
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=None,
        help="Input video (default: first .mp4 in input_videos/)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Detections .pkl cache (default: <video_stem>_detections.pkl next to video)",
    )
    parser.add_argument(
        "--force-redetect",
        action="store_true",
        help="Ignore detection cache and re-run face detection",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less console output from engagement step",
    )
    args = parser.parse_args()

    video = args.video
    if video is None:
        vdir = SCRIPT_DIR / "input_videos"
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        if not vdir.is_dir():
            parser.error(f"No video given and {vdir} does not exist")
        candidates = sorted(f for f in vdir.iterdir() if f.suffix.lower() in exts)
        if not candidates:
            parser.error(f"No videos found in {vdir}")
        video = candidates[0]
    else:
        video = Path(video)
        if not video.is_absolute():
            video = (Path.cwd() / video).resolve()

    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 1

    cache = args.cache
    if cache is not None and not cache.is_absolute():
        cache = (Path.cwd() / cache).resolve()

    print(f"\n=== Step 1–2: Face detection & clustering ===")
    print(f"    Video: {video.name}\n")
    clusters = run_pipeline(
        video,
        cache_path=cache,
        force_redetect=args.force_redetect,
    )

    n_persons = len([c for c in clusters if c.id != -1])
    if n_persons == 0:
        print("No person clusters found — engagement analysis skipped.", file=sys.stderr)
        return 1

    print(f"\n=== Step 3: Engagement analysis ({n_persons} person(s)) ===\n")
    results = run_engagement_pipeline(
        clusters,
        str(video),
        verbose=not args.quiet,
    )

    if not results:
        print("Engagement pipeline produced no profiles.", file=sys.stderr)
        return 1

    print(f"Done — {len(results)} engagement profile(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
