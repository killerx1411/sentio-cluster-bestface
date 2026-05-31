from pydantic import BaseModel, Field


class Landmarks(BaseModel):
    left_eye: list[int]
    right_eye: list[int]
    nose: list[int]
    mouth_left: list[int]
    mouth_right: list[int]


class FaceDetection(BaseModel):
    frame_idx: int
    timestamp_sec: float
    embedding: list[float]
    crop_bgr: str
    quality_score: float = Field(ge=0.0, le=1.0)
    bbox: list[int] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    landmarks: Landmarks


class Cluster(BaseModel):
    id: int
    faces: list[FaceDetection]


class ClusterPayload(BaseModel):
    job_id: str = Field(max_length=128)
    video_id: str = Field(max_length=256)
    clusters: list[Cluster]


class VideoRegisterRequest(BaseModel):
    """Run detection → clustering → best-face on a local video path."""

    job_id: str = Field(max_length=128)
    video_path: str = Field(max_length=1024)
    video_id: str | None = Field(default=None, max_length=256)
    force_redetect: bool = False
    known_n_persons: int | None = Field(default=None, ge=1)
