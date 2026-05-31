# Best-Face + profile-clustering

Combined face pipeline for classroom / CCTV video:

| Step | Project | Module | What it does |
|------|---------|--------|--------------|
| 1 | `profile-clustering` | `face_cluster.py` | Sample frames, detect faces, AdaFace embeddings |
| 2 | `profile-clustering` | `face_cluster.py` | HDBSCAN + agglomerative merge → person clusters |
| 3 | `Best-Face` | `face_registration` | Filter, score, dedup, save best face per person |

```
video.mp4
    │
    ▼  (optional cache: longvid_detections.pkl)
┌─────────────────────────────┐
│  profile-clustering venv    │  Steps 1 + 2
│  clustervenv                │
└─────────────┬───────────────┘
              │  longvid_clusters.json
              ▼
┌─────────────────────────────┐
│  Best-Face venv             │  Step 3
│  bestfaceharsh              │
└─────────────┬───────────────┘
              ▼
   face_registration/database/person_0001/best_face.jpg
```

---

## Repo layout

Both projects live under the same repo root (`checkidk/`):

```
checkidk/
├── run_face_pipeline.ps1          ← recommended (two venvs)
├── run_face_pipeline.py             ← single-process (one venv with all deps)
├── profile-clustering/
│   ├── clustervenv/                 ← clustering Python env
│   ├── face_cluster.py
│   ├── export_for_bestface.py       ← step 1+2 → JSON
│   ├── export_cluster_profiles.py   ← contact sheets only (optional)
│   └── input_videos/
│       ├── longvid.mp4
│       ├── longvid_detections.pkl   ← detection cache (reuse)
│       └── longvid_clusters.json    ← generated bridge file
└── Best-Face/
    ├── bestfaceharsh/               ← Best-Face Python env
    ├── requirements.txt
    └── face_registration/
        ├── scripts/register_from_json.py   ← step 3
        └── database/                         ← output
```

---

## First-time setup

### 1. profile-clustering venv

```powershell
cd c:\Users\Kulkarni\checkidk\profile-clustering
python -m venv clustervenv
.\clustervenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place AdaFace weights under `profile-clustering/AdaFace/weights/` if not already present.

### 2. Best-Face venv

```powershell
cd c:\Users\Kulkarni\checkidk\Best-Face
python -m venv bestfaceharsh
.\bestfaceharsh\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## How to run (recommended — two venvs)

The two projects use **different virtual environments**. Use the PowerShell orchestrator from repo root:

```powershell
cd c:\Users\Kulkarni\checkidk
.\run_face_pipeline.ps1
```

Defaults (matches your existing `longvid` workflow):

| Parameter | Default |
|-----------|---------|
| Video | `profile-clustering\input_videos\longvid.mp4` |
| Detection cache | `profile-clustering\input_videos\longvid_detections.pkl` |
| Cluster JSON | `profile-clustering\input_videos\longvid_clusters.json` |

Custom paths:

```powershell
.\run_face_pipeline.ps1 `
  -Video "profile-clustering\input_videos\longvid.mp4" `
  -Cache "profile-clustering\input_videos\longvid_detections.pkl" `
  -JsonOut "profile-clustering\input_videos\longvid_clusters.json"
```

**Output:** `Best-Face\face_registration\database\person_0001\best_face.jpg`, `metadata.json`, etc.

---

## How to run (manual — two venvs)

### Step 1+2 — clustering (profile-clustering venv)

Reuses your existing detection cache — **no re-detection** unless you pass `--force-redetect`:

```powershell
cd c:\Users\Kulkarni\checkidk\profile-clustering
.\clustervenv\Scripts\Activate.ps1

python export_for_bestface.py "input_videos/longvid.mp4" `
  --cache "input_videos/longvid_detections.pkl" `
  --out "input_videos/longvid_clusters.json"
```

This is the same cache you already use with:

```powershell
python export_cluster_profiles.py "input_videos/longvid.mp4" --cache "input_videos/longvid_detections.pkl"
```

`export_cluster_profiles.py` only exports contact-sheet images to `cluster_profiles/`.  
`export_for_bestface.py` runs the same clustering step but writes JSON for Best-Face.

Person count for clustering is controlled in `face_cluster.py`:

```python
KNOWN_N_PERSONS = 26   # set to None for auto-detect
```

### Step 3 — best-face registration (Best-Face venv)

```powershell
cd c:\Users\Kulkarni\checkidk\Best-Face\face_registration
..\..\Best-Face\bestfaceharsh\Scripts\Activate.ps1

python scripts/register_from_json.py `
  "..\..\profile-clustering\input_videos\longvid_clusters.json"
```

Print full JSON report:

```powershell
python scripts/register_from_json.py `
  "..\..\profile-clustering\input_videos\longvid_clusters.json" --json-out
```

---

## How to run (`run_face_pipeline.py` — single venv only)

`run_face_pipeline.py` runs all three steps in **one Python process**.  
It only works if **both** dependency sets are installed in the **same** environment (InsightFace, hdbscan, FastAPI, etc.).

```powershell
cd c:\Users\Kulkarni\checkidk

# Activate ONE venv that has profile-clustering + Best-Face deps installed, then:
python run_face_pipeline.py profile-clustering/input_videos/longvid.mp4 `
  --cache profile-clustering/input_videos/longvid_detections.pkl

python run_face_pipeline.py profile-clustering/input_videos/longvid.mp4 `
  --cache profile-clustering/input_videos/longvid_detections.pkl `
  --known-n 26 `
  --json
```

| Flag | Description |
|------|-------------|
| `video` | Path to input video (required) |
| `--cache` | Detection `.pkl` cache (skips step 1 if present) |
| `--known-n` | Override `KNOWN_N_PERSONS` for clustering |
| `--force-redetect` | Ignore cache and re-run face detection |
| `--job-id` | Job id for registration report (default: `job_<stem>`) |
| `--json` | Print registration report as JSON |

**If you use separate venvs (normal setup), use `run_face_pipeline.ps1` instead.**

---

## Best-Face API (standalone step 3)

Start the server (Best-Face venv):

```powershell
cd c:\Users\Kulkarni\checkidk\Best-Face\face_registration
..\..\Best-Face\bestfaceharsh\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Docs: http://localhost:8000/docs

### Register from pre-clustered JSON (step 3 only)

After running `export_for_bestface.py`, POST the JSON:

```powershell
curl -X POST http://localhost:8000/api/v1/register `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-token-here" `
  -d "@..\..\profile-clustering\input_videos\longvid_clusters.json"
```

### Full video path (single venv with all ML deps)

Requires InsightFace/hdbscan in the same env as the API:

```powershell
curl -X POST http://localhost:8000/api/v1/register/video `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-token-here" `
  -d '{
    "job_id": "job_longvid",
    "video_path": "C:/Users/Kulkarni/checkidk/profile-clustering/input_videos/longvid.mp4",
    "known_n_persons": 26
  }'
```

Set `PROFILE_CLUSTERING_ROOT` in `.env` if `profile-clustering` is not next to `Best-Face/`.

### Health check

```powershell
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Configuration

Best-Face settings via `.env` in `face_registration/` (or environment variables):

| Key | Default | Description |
|-----|---------|-------------|
| `AUTH_TOKEN` | `your-secret-token-here` | Bearer token for API auth |
| `DATABASE_ROOT` | `database/` | Best-face output directory |
| `PROFILE_CLUSTERING_ROOT` | *(auto)* | Path to `profile-clustering/` if non-standard |
| `MIN_FACE_PX` | `60` | Min face side in pixels |
| `MIN_CONF` | `0.70` | Min detection confidence |
| `MIN_QUALITY` | `0.20` | Min pipeline quality score |
| `MIN_FRONT` | `0.40` | Min frontality score |
| `MIN_ILLUM` | `0.25` | Min illumination score |
| `DEDUP_GAP_SEC` | `1.0` | Temporal dedup window (seconds) |
| `JPEG_QUALITY` | `95` | Saved JPEG quality |
| `MAX_WORKERS` | `cpu_count * 2` | Thread pool size |

Clustering tuning in `profile-clustering/face_cluster.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `KNOWN_N_PERSONS` | `26` | Expected person count (`None` = auto) |
| `MIN_FACE_QUALITY` | `4.0` | Min quality to keep a detection |
| `HDBSCAN_EPSILON` | per engine | Cluster merge epsilon |

---

## Tests

Best-Face unit / integration tests (Best-Face venv):

```powershell
cd c:\Users\Kulkarni\checkidk\Best-Face\face_registration
..\..\Best-Face\bestfaceharsh\Scripts\Activate.ps1
pytest tests/ -v
```

Fake-data API smoke test:

```powershell
uvicorn app.main:app --reload --port 8000
# separate terminal:
python scripts/test_api.py
```

---

## Project structure

```
Best-Face/
  face_registration/
    app/
      main.py                    # FastAPI entry
      api/routes.py              # /register, /register/video
      adapters/clustering_adapter.py   # clustering → API model bridge
      pipeline/
        orchestrator.py          # step 3: filter → score → save
        video_pipeline.py          # single-process detect→cluster→register
      storage/local.py           # atomic JPEG + metadata writes
    scripts/
      register_from_json.py      # step 3 CLI (two-venv workflow)
    database/                    # output (gitignored)
  bestfaceharsh/                 # venv (gitignored)

profile-clustering/
  face_cluster.py                # steps 1 + 2
  export_for_bestface.py         # step 1+2 → JSON bridge
  export_cluster_profiles.py     # optional contact sheets
  clustervenv/                   # venv (gitignored)
  input_videos/                  # videos + .pkl cache (gitignored)
```
