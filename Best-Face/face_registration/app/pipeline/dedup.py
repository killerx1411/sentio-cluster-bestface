from app.core.config import settings
from app.models.request import FaceDetection


def temporal_dedup(faces: list[FaceDetection]) -> list[FaceDetection]:
    if len(faces) <= 1:
        return list(faces)
    sorted_faces = sorted(faces, key=lambda f: f.timestamp_sec)
    kept: list[FaceDetection] = [sorted_faces[0]]
    for f in sorted_faces[1:]:
        if f.timestamp_sec - kept[-1].timestamp_sec >= settings.DEDUP_GAP_SEC:
            kept.append(f)
        else:
            score_f = getattr(f, "_score", 0.0)
            score_kept = getattr(kept[-1], "_score", 0.0)
            if score_f > score_kept:
                kept[-1] = f
    return kept
