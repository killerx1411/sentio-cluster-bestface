import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.config import settings
from app.models.request import ClusterPayload, FaceDetection
from app.models.response import RegistrationReport, ClusterResult
from app.pipeline.filters import drop_noise_clusters, pre_filter_faces
from app.pipeline.dedup import temporal_dedup
from app.pipeline.scoring import score_all, select_best, generate_identity
from app.storage.local import LocalStorage

storage = LocalStorage()


def run_pipeline(payload: ClusterPayload) -> RegistrationReport:
    start = time.perf_counter()
    noise_dropped = len(payload.clusters) - len(drop_noise_clusters(payload.clusters))
    valid_clusters = drop_noise_clusters(payload.clusters)
    if not valid_clusters:
        return RegistrationReport(
            job_id=payload.job_id,
            video_id=payload.video_id,
            status="failed",
            registered=0,
            skipped=0,
            noise_dropped=noise_dropped,
            results=[],
            processing_time_ms=0.0,
        )
    total = len(valid_clusters)
    results: list[ClusterResult] = []
    registered = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
        futures = {
            executor.submit(_process_cluster, c, idx + 1, total, payload): c
            for idx, c in enumerate(valid_clusters)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.status == "skipped":
                skipped += 1
            else:
                registered += 1
    results.sort(key=lambda r: r.cluster_id)
    elapsed_ms = (time.perf_counter() - start) * 1000
    status = "success" if registered > 0 else "partial" if skipped > 0 else "failed"
    return RegistrationReport(
        job_id=payload.job_id,
        video_id=payload.video_id,
        status=status,
        registered=registered,
        skipped=skipped,
        noise_dropped=noise_dropped,
        results=results,
        processing_time_ms=round(elapsed_ms, 2),
    )


def _process_cluster(cluster, person_counter: int, total_clusters: int, payload) -> ClusterResult:
    warnings: list[str] = []
    person_id, uid = generate_identity(cluster.id, person_counter, total_clusters)
    person_dir = os.path.join(settings.DATABASE_ROOT, person_id)
    image_path = os.path.join(person_dir, "best_face.jpg")
    if os.path.exists(image_path):
        meta_path = os.path.join(person_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return ClusterResult(
                cluster_id=cluster.id,
                person_id=meta["person_id"],
                uuid=meta["uuid"],
                composite_score=meta["composite_score"],
                frame_idx=meta["frame_idx"],
                timestamp_sec=meta["timestamp_sec"],
                image_path=image_path,
                fallback_used=meta.get("fallback_used", False),
                status="skipped",
                warnings=[],
            )
    filtered_faces = pre_filter_faces(cluster.faces)
    fallback_used = len(filtered_faces) < len(cluster.faces)
    scored = score_all(filtered_faces)
    deduped = temporal_dedup(scored)
    best = select_best(deduped)
    if best is None:
        warnings.append("no_faces_after_processing")
        best = cluster.faces[0] if cluster.faces else None
        if best is None:
            return ClusterResult(
                cluster_id=cluster.id,
                person_id=person_id,
                uuid=uid,
                composite_score=0.0,
                frame_idx=0,
                timestamp_sec=0.0,
                image_path="",
                fallback_used=True,
                warnings=["empty_cluster"],
            )
        fallback_used = True
    storage.write_face(best, person_id)
    storage.write_metadata(person_id, {
        "person_id": person_id,
        "uuid": uid,
        "cluster_id": cluster.id,
        "job_id": payload.job_id,
        "video_id": payload.video_id,
        "composite_score": round(getattr(best, "_score", 0.0), 4),
        "sub_scores": {
            "quality": best.quality_score,
            "frontality": round(best.landmarks.left_eye[0] and 0.0 or 0.0, 4),
            "size": 0.0,
            "confidence": best.confidence,
            "illumination": 0.0,
        },
        "frame_idx": best.frame_idx,
        "timestamp_sec": best.timestamp_sec,
        "bbox": best.bbox,
        "confidence": best.confidence,
        "quality_score": best.quality_score,
        "fallback_used": fallback_used,
        "embedding": best.embedding,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return ClusterResult(
        cluster_id=cluster.id,
        person_id=person_id,
        uuid=uid,
        composite_score=round(getattr(best, "_score", 0.0), 4),
        frame_idx=best.frame_idx,
        timestamp_sec=best.timestamp_sec,
        image_path=image_path,
        fallback_used=fallback_used,
        warnings=warnings,
    )
