from pydantic import BaseModel


class ClusterResult(BaseModel):
    cluster_id: int
    person_id: str
    uuid: str
    composite_score: float
    frame_idx: int
    timestamp_sec: float
    image_path: str
    fallback_used: bool
    status: str = "registered"
    warnings: list[str] = []


class RegistrationReport(BaseModel):
    job_id: str
    video_id: str
    status: str
    registered: int
    skipped: int
    noise_dropped: int
    results: list[ClusterResult]
    processing_time_ms: float
