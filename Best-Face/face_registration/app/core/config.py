import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MIN_FACE_PX: int = 60
    MIN_CONF: float = 0.70
    MIN_QUALITY: float = 0.20
    MIN_FRONT: float = 0.40
    MIN_ILLUM: float = 0.25
    DEDUP_GAP_SEC: float = 1.0
    JPEG_QUALITY: int = 95
    REF_AREA: int = 62500
    W_QUALITY: float = 0.30
    W_FRONT: float = 0.25
    W_SIZE: float = 0.20
    W_CONF: float = 0.15
    W_ILLUM: float = 0.10
    MAX_WORKERS: int = min(32, (os.cpu_count() or 4) * 2)
    DATABASE_ROOT: str = "database/"
    MAX_PAYLOAD_MB: int = 100
    STORAGE_BACKEND: str = "local"
    AUTH_TOKEN: str = "your-secret-token-here"
    PROFILE_CLUSTERING_ROOT: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
