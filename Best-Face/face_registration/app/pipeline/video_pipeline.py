"""
End-to-end pipeline: video → face detection → clustering → best-face registration.

Step 1–2: profile-clustering/face_cluster.py (detect + HDBSCAN cluster)
Step 3:   Best-Face registration orchestrator (filter, score, select best)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from app.adapters.clustering_adapter import clusters_to_payload
from app.models.request import ClusterPayload
from app.models.response import RegistrationReport
from app.pipeline.orchestrator import run_pipeline as run_registration_pipeline

# checkidk/profile-clustering (sibling of Best-Face/)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3].parent
_DEFAULT_CLUSTERING_ROOT = _WORKSPACE_ROOT / "profile-clustering"


def _ensure_clustering_import(clustering_root: Path | None = None) -> None:
    root = Path(clustering_root or _DEFAULT_CLUSTERING_ROOT).resolve()
    root_str = str(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"profile-clustering not found at {root}. "
            "Set PROFILE_CLUSTERING_ROOT or place it next to Best-Face/."
        )
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def run_clustering(
    video_path: str | Path,
    *,
    clustering_root: Path | None = None,
    cache_path: str | Path | None = None,
    force_redetect: bool = False,
    known_n_persons: int | None = None,
) -> list:
    _ensure_clustering_import(clustering_root)
    import face_cluster as fc  # noqa: WPS433 — runtime import after path setup

    old_n = fc.KNOWN_N_PERSONS
    if known_n_persons is not None:
        fc.KNOWN_N_PERSONS = known_n_persons
    try:
        return fc.run_pipeline(
            video_path,
            cache_path=cache_path,
            force_redetect=force_redetect,
        )
    finally:
        fc.KNOWN_N_PERSONS = old_n


def run_video_pipeline(
    video_path: str | Path,
    *,
    job_id: str | None = None,
    video_id: str | None = None,
    clustering_root: Path | None = None,
    cache_path: str | Path | None = None,
    force_redetect: bool = False,
    known_n_persons: int | None = None,
) -> RegistrationReport:
    """
    Full pipeline for one video file.

    1. Face detection + clustering (profile-clustering)
    2. Best-face selection + registration (Best-Face)
    """
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    job_id = job_id or f"job_{video_path.stem}"
    video_id = video_id or video_path.stem

    t0 = time.perf_counter()
    clusters = run_clustering(
        video_path,
        clustering_root=clustering_root,
        cache_path=cache_path,
        force_redetect=force_redetect,
        known_n_persons=known_n_persons,
    )
    cluster_ms = (time.perf_counter() - t0) * 1000

    if not clusters:
        return RegistrationReport(
            job_id=job_id,
            video_id=video_id,
            status="failed",
            registered=0,
            skipped=0,
            noise_dropped=0,
            results=[],
            processing_time_ms=round(cluster_ms, 2),
        )

    payload: ClusterPayload = clusters_to_payload(clusters, job_id, video_id)
    report = run_registration_pipeline(payload)
    report.processing_time_ms = round(report.processing_time_ms + cluster_ms, 2)
    return report
