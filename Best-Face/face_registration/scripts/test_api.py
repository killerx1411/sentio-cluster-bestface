import requests
import base64
import numpy as np
import cv2
import json
import time

BASE_URL = "http://localhost:8000"
AUTH_HEADER = {"Authorization": "Bearer your-secret-token-here"}


def make_test_image(width=250, height=250, brightness=128) -> str:
    img = np.full((height, width, 3), brightness, dtype=np.uint8)
    img[50:200, 50:200] = np.random.randint(80, 200, (150, 150, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


def make_face(frame_idx=0, timestamp_sec=0.0, quality=0.8, confidence=0.9,
              bbox=None, left_eye=None, right_eye=None, nose=None, brightness=128):
    if bbox is None:
        bbox = [100, 80, 200, 180]
    if left_eye is None:
        left_eye = [130, 110]
    if right_eye is None:
        right_eye = [170, 110]
    if nose is None:
        nose = [150, 130]
    return {
        "frame_idx": frame_idx,
        "timestamp_sec": timestamp_sec,
        "embedding": np.random.uniform(-1, 1, 512).tolist(),
        "crop_bgr": make_test_image(brightness=brightness),
        "quality_score": quality,
        "bbox": bbox,
        "confidence": confidence,
        "landmarks": {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "nose": nose,
            "mouth_left": [135, 150],
            "mouth_right": [165, 150],
        },
    }


def make_cluster(cluster_id, num_faces=5):
    faces = []
    for i in range(num_faces):
        faces.append(make_face(
            frame_idx=i * 100,
            timestamp_sec=i * 2.5,
            quality=np.random.uniform(0.3, 0.95),
            confidence=np.random.uniform(0.7, 0.99),
            brightness=np.random.randint(50, 200),
        ))
    return {"id": cluster_id, "faces": faces}


def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    print(f"[HEALTH] {resp.status_code} -> {resp.json()}")


def test_single_cluster():
    payload = {
        "job_id": "test_job_001",
        "video_id": "test_video_001",
        "clusters": [make_cluster(0, num_faces=3)],
    }
    resp = requests.post(f"{BASE_URL}/api/v1/register", json=payload, headers=AUTH_HEADER)
    print(f"[SINGLE CLUSTER] {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


def test_multi_cluster():
    payload = {
        "job_id": "test_job_002",
        "video_id": "test_video_002",
        "clusters": [make_cluster(i, num_faces=5) for i in range(10)],
    }
    resp = requests.post(f"{BASE_URL}/api/v1/register", json=payload, headers=AUTH_HEADER)
    data = resp.json()
    print(f"[MULTI CLUSTER] {resp.status_code} | registered={data['registered']} | time={data['processing_time_ms']}ms")


def test_idempotency():
    payload = {
        "job_id": "test_job_003",
        "video_id": "test_video_003",
        "clusters": [make_cluster(0, num_faces=3)],
    }
    resp1 = requests.post(f"{BASE_URL}/api/v1/register", json=payload, headers=AUTH_HEADER)
    resp2 = requests.post(f"{BASE_URL}/api/v1/register", json=payload, headers=AUTH_HEADER)
    d1, d2 = resp1.json(), resp2.json()
    print(f"[IDEMPOTENCY] 1st: registered={d1['registered']} | 2nd: skipped={d2['skipped']}")


def test_noise_clusters():
    payload = {
        "job_id": "test_job_004",
        "video_id": "test_video_004",
        "clusters": [
            {"id": -1, "faces": [make_face()]},
            {"id": -1, "faces": [make_face()]},
            {"id": 0, "faces": [make_face()]},
        ],
    }
    resp = requests.post(f"{BASE_URL}/api/v1/register", json=payload, headers=AUTH_HEADER)
    data = resp.json()
    print(f"[NOISE FILTER] {resp.status_code} | noise_dropped={data['noise_dropped']} | registered={data['registered']}")


def test_bad_payload():
    resp = requests.post(f"{BASE_URL}/api/v1/register",
                         json={"job_id": "x"},
                         headers=AUTH_HEADER)
    print(f"[BAD PAYLOAD] {resp.status_code} -> {resp.json().get('detail', resp.json())}")


def test_no_auth():
    payload = {"job_id": "x", "video_id": "y", "clusters": []}
    resp = requests.post(f"{BASE_URL}/api/v1/register", json=payload)
    print(f"[NO AUTH] {resp.status_code}")


def test_benchmark(num_clusters=100):
    payload = {
        "job_id": "benchmark_job",
        "video_id": "benchmark_video",
        "clusters": [make_cluster(i, num_faces=5) for i in range(num_clusters)],
    }
    start = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/api/v1/register", json=payload, headers=AUTH_HEADER)
    elapsed = (time.perf_counter() - start) * 1000
    data = resp.json()
    print(f"[BENCHMARK {num_clusters} clusters] {resp.status_code} | registered={data['registered']} | wall_time={elapsed:.0f}ms | server_time={data['processing_time_ms']}ms")


if __name__ == "__main__":
    print("=" * 60)
    print("FACE REGISTRATION API — TEST SUITE")
    print("=" * 60)

    test_health()
    print()
    test_single_cluster()
    print()
    test_multi_cluster()
    print()
    test_idempotency()
    print()
    test_noise_clusters()
    print()
    test_bad_payload()
    print()
    test_no_auth()
    print()
    test_benchmark(100)

    print()
    print("Done. Check database/ folder for output files.")
