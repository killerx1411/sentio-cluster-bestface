# Face Registration API

Best Face Selection & Identity Registration Module — Stage 5 of the Face Attendance Pipeline.

## Quick Start

```bash
cd face_registration

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v
```

Server starts at `http://localhost:8000`. Auto-generated API docs at `http://localhost:8000/docs`.

## API Usage

### Register Faces

```bash
curl -X POST http://localhost:8000/api/v1/register \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{
    "job_id": "job_001",
    "video_id": "lecture_001",
    "clusters": [
      {
        "id": 0,
        "faces": [
          {
            "frame_idx": 100,
            "timestamp_sec": 3.3,
            "embedding": [0.1, 0.2, "...512 floats..."],
            "crop_bgr": "<base64-encoded-image>",
            "quality_score": 0.85,
            "bbox": [120, 80, 220, 180],
            "confidence": 0.95,
            "landmarks": {
              "left_eye": [145, 110],
              "right_eye": [195, 110],
              "nose": [170, 130],
              "mouth_left": [150, 150],
              "mouth_right": [190, 150]
            }
          }
        ]
      }
    ]
  }'
```

### Response (HTTP 200)

```json
{
  "job_id": "job_001",
  "video_id": "lecture_001",
  "status": "success",
  "registered": 1,
  "skipped": 0,
  "noise_dropped": 0,
  "results": [
    {
      "cluster_id": 0,
      "person_id": "person_0001",
      "uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "composite_score": 0.847,
      "frame_idx": 100,
      "timestamp_sec": 3.3,
      "image_path": "database/person_0001/best_face.jpg",
      "fallback_used": false,
      "status": "registered",
      "warnings": []
    }
  ],
  "processing_time_ms": 312.5
}
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Configuration

All settings via environment variables or `.env` file:

| Key | Default | Description |
|-----|---------|-------------|
| `AUTH_TOKEN` | `your-secret-token-here` | Bearer token for auth |
| `DATABASE_ROOT` | `database/` | Output directory |
| `MIN_FACE_PX` | `60` | Min face side in pixels |
| `MIN_CONF` | `0.70` | Min detection confidence |
| `MIN_QUALITY` | `0.20` | Min pipeline quality score |
| `MIN_FRONT` | `0.40` | Min frontality score |
| `MIN_ILLUM` | `0.25` | Min illumination score |
| `DEDUP_GAP_SEC` | `1.0` | Temporal dedup window (sec) |
| `JPEG_QUALITY` | `95` | JPEG compression quality |
| `MAX_WORKERS` | `cpu_count * 2` | Thread pool size |

## Docker

```bash
docker-compose up --build
```

## Test Without Real Data

Generate fake face data and test the full pipeline:

```bash
# Start server first
uvicorn app.main:app --reload --port 8000

# In another terminal, run the test script
python scripts/test_api.py
```

This runs 8 test scenarios: health check, single cluster, multi-cluster, idempotency, noise filtering, bad payload, auth check, and a 100-cluster benchmark. Output goes to `database/` folder.

## Project Structure

```
face_registration/
  app/
    main.py              # FastAPI entry
    api/routes.py        # REST endpoint + auth
    core/config.py       # All config in one place
    models/              # Pydantic request/response schemas
    pipeline/            # filters → dedup → scoring → orchestrator
    storage/local.py     # Atomic JPEG + metadata writes
  tests/
    unit/test_all.py     # 28 unit tests
    integration/test_api.py  # 8 integration tests
```
