import os
import json
import tempfile
import cv2
import numpy as np
import base64
from app.core.config import settings


class LocalStorage:
    def __init__(self):
        os.makedirs(settings.DATABASE_ROOT, exist_ok=True)

    def write_face(self, face, person_id: str) -> str:
        person_dir = os.path.join(settings.DATABASE_ROOT, person_id)
        os.makedirs(person_dir, exist_ok=True)
        image_path = os.path.join(person_dir, "best_face.jpg")
        crop = self._decode_crop(face.crop_bgr)
        if crop is not None:
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=person_dir)
            os.close(fd)
            try:
                cv2.imwrite(tmp_path, crop, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY])
                os.replace(tmp_path, image_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        return image_path

    def write_metadata(self, person_id: str, metadata: dict) -> str:
        person_dir = os.path.join(settings.DATABASE_ROOT, person_id)
        os.makedirs(person_dir, exist_ok=True)
        meta_path = os.path.join(person_dir, "metadata.json")
        fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=person_dir)
        os.close(fd)
        try:
            with open(tmp_path, "w") as f:
                json.dump(metadata, f, indent=2)
            os.replace(tmp_path, meta_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return meta_path

    def _decode_crop(self, crop_bgr_b64: str) -> np.ndarray | None:
        try:
            img_bytes = base64.b64decode(crop_bgr_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None
