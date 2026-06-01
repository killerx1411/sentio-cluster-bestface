import pytest
import os
import shutil
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from tests.conftest import make_face, make_cluster, make_payload

client = TestClient(app)
HEADERS = {"Authorization": f"Bearer {settings.AUTH_TOKEN}"}


@pytest.fixture(autouse=True)
def cleanup_db():
    yield
    if os.path.exists(settings.DATABASE_ROOT):
        shutil.rmtree(settings.DATABASE_ROOT)


class TestRegisterEndpoint:
    def test_valid_payload_registers(self):
        payload = make_payload(clusters=[make_cluster(i) for i in range(5)])
        resp = client.post("/api/v1/register", json=payload.model_dump(), headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["registered"] == 5
        assert data["status"] == "success"

    def test_noise_only_payload(self):
        payload = make_payload(clusters=[make_cluster(-1)])
        resp = client.post("/api/v1/register", json=payload.model_dump(), headers=HEADERS)
        assert resp.status_code == 400

    def test_missing_embedding_field(self):
        payload = make_payload()
        bad_data = payload.model_dump()
        bad_data["clusters"][0]["faces"][0].pop("embedding")
        resp = client.post("/api/v1/register", json=bad_data, headers=HEADERS)
        assert resp.status_code == 422

    def test_idempotent_resubmission(self):
        payload = make_payload(clusters=[make_cluster(i) for i in range(3)])
        resp1 = client.post("/api/v1/register", json=payload.model_dump(), headers=HEADERS)
        assert resp1.status_code == 200
        resp2 = client.post("/api/v1/register", json=payload.model_dump(), headers=HEADERS)
        assert resp2.status_code == 200
        assert resp2.json()["skipped"] == 3

    def test_empty_clusters_list(self):
        payload = make_payload(clusters=[])
        resp = client.post("/api/v1/register", json=payload.model_dump(), headers=HEADERS)
        assert resp.status_code == 400

    def test_invalid_bearer_token(self):
        payload = make_payload()
        resp = client.post("/api/v1/register", json=payload.model_dump(), headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_response_contains_image_path(self):
        payload = make_payload(clusters=[make_cluster(0)])
        resp = client.post("/api/v1/register", json=payload.model_dump(), headers=HEADERS)
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert "image_path" in results[0]
        assert os.path.exists(results[0]["image_path"])

    def test_response_time_100_clusters(self):
        import time
        payload = make_payload(clusters=[make_cluster(i) for i in range(100)])
        start = time.perf_counter()
        resp = client.post("/api/v1/register", json=payload.model_dump(), headers=HEADERS)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        assert elapsed_ms < 1000
