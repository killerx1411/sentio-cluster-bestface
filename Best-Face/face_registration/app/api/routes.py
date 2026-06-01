from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Header
from app.models.request import ClusterPayload, VideoRegisterRequest
from app.models.response import RegistrationReport
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.video_pipeline import run_video_pipeline
from app.core.config import settings

router = APIRouter()


def verify_auth(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    token = authorization.replace("Bearer ", "")
    if token != settings.AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


@router.post("/api/v1/register", response_model=RegistrationReport, dependencies=[Depends(verify_auth)])
def register(payload: ClusterPayload) -> RegistrationReport:
    if len(payload.clusters) == 0:
        raise HTTPException(status_code=400, detail="no_valid_clusters")
    report = run_pipeline(payload)
    if report.registered == 0 and report.skipped == 0:
        raise HTTPException(status_code=400, detail="no_valid_clusters")
    return report


@router.post(
    "/api/v1/register/video",
    response_model=RegistrationReport,
    dependencies=[Depends(verify_auth)],
)
def register_video(body: VideoRegisterRequest) -> RegistrationReport:
    """
    Full pipeline: face detection → clustering → best-face registration.
    """
    video_path = Path(body.video_path)
    if not video_path.is_file():
        raise HTTPException(status_code=400, detail="video_not_found")

    clustering_root = (
        Path(settings.PROFILE_CLUSTERING_ROOT)
        if settings.PROFILE_CLUSTERING_ROOT
        else None
    )
    try:
        report = run_video_pipeline(
            video_path,
            job_id=body.job_id,
            video_id=body.video_id or video_path.stem,
            clustering_root=clustering_root,
            force_redetect=body.force_redetect,
            known_n_persons=body.known_n_persons,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if report.registered == 0 and report.skipped == 0:
        raise HTTPException(status_code=400, detail="no_valid_clusters")
    return report
