"""
Step 3 only: Best-face registration from clustering JSON.

Run inside Best-Face venv (bestfaceharsh):

  python scripts/register_from_json.py ../../profile-clustering/input_videos/longvid_clusters.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models.request import ClusterPayload  # noqa: E402
from app.pipeline.orchestrator import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Register best faces from clustering JSON")
    parser.add_argument("json_path", type=Path, help="Path to *_clusters.json")
    parser.add_argument("--json-out", action="store_true", help="Print full report as JSON")
    args = parser.parse_args()

    json_path = args.json_path.resolve()
    if not json_path.is_file():
        print(f"File not found: {json_path}", file=sys.stderr)
        return 1

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    payload = ClusterPayload.model_validate(data)
    report = run_pipeline(payload)

    if args.json_out:
        print(json.dumps(report.model_dump(), indent=2))
    else:
        print(f"\nStatus:     {report.status}")
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
