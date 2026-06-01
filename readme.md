# Sentio Cluster + Best-Face Pipeline

End-to-end pipeline for classroom / CCTV footage:

1. **Face clustering** (`profile-clustering`) — detect faces, AdaFace embeddings, HDBSCAN + merge into person clusters  
2. **Best-face registration** (`Best-Face`) — score, filter, and save one best face per person  
3. **Engagement analysis** (`profile-clustering`) — per-person engagement signals and JSON report  

```
your_video.mp4  →  input_videos/
        │
        ▼  Step 1+2 (profile-clustering venv)
   *_detections.pkl  (optional cache)
   *_clusters.json
        │
        ▼  Step 3 (Best-Face venv)
   Best-Face/face_registration/database/person_0001/best_face.jpg
        │
        ▼  Step 4 (profile-clustering venv)
   *_engagement.json
```

## Repository layout

```
.
├── README.md

├── profile-clustering/
│   ├── face_cluster.py          # detection + clustering
│   ├── export_for_bestface.py   # steps 1+2 → JSON for Best-Face
│   ├── engagement_analyzer.py
│   ├── test_engagement.py       # clustering + engagement (step 4)
│   ├── net.py                   # AdaFace IR model definition
│   ├── requirements.txt
│   ├── AdaFace/                 # cloned locally (not in git)
│   │   └── weights/
│   │       └── adaface_ir101_webface12m.ckpt
│   └── input_videos/            # you create this; put CCTV .mp4 here
└── Best-Face/
    ├── requirements.txt
    └── face_registration/
        ├── scripts/register_from_json.py
        └── database/            # output (gitignored)
```

---

## Prerequisites

- **Windows** (PowerShell scripts tested on Windows; Python commands work on Linux/macOS with path adjustments)
- **Python 3.10+** recommended
- **CUDA GPU** optional but strongly recommended for detection and AdaFace
- Enough disk space for model weights (~600 MB for AdaFace checkpoint) and video caches

---

## First-time setup

### 1. Clone this repository

```powershell
git clone https://github.com/killerx1411/sentio-cluster-bestface.git
cd sentio-cluster-bestface
```

### 2. Create `input_videos` and add your CCTV footage

```powershell
mkdir profile-clustering\input_videos
```

Copy your classroom / CCTV `.mp4` (or `.avi`, `.mov`, etc.) into that folder, for example:

```
profile-clustering/input_videos/my_classroom.mp4
```

All pipeline commands below assume paths under `profile-clustering/input_videos/`. Replace `my_classroom` with your file stem.

### 3. Set `KNOWN_N_PERSONS` (required for accurate clustering)

Open `profile-clustering/face_cluster.py` and set the number of distinct people you expect in the video:

```python
# profile-clustering/face_cluster.py (around line 208)
KNOWN_N_PERSONS = 26   # your headcount; use None for auto-detect
```

| Value | Behavior |
|-------|----------|
| **Integer** (e.g. `26`) | Stage-2 agglomerative merge produces **exactly** that many person clusters |
| **`None`** | Auto mode — estimates count from the merge tree (may need tuning via `MERGE_JUMP_FACTOR`) |

Re-run clustering whenever you change this value.



### 4. AdaFace setup (required for embedding quality)

Clustering uses **AdaFace IR-101** weights. They are not stored in git.

**4a. Clone AdaFace inside `profile-clustering`:**

```powershell
cd profile-clustering
git clone https://github.com/mk-minchul/AdaFace.git AdaFace
```

**4b. Download the checkpoint**

Download **adaface_ir101_webface12m.ckpt** from Google Drive:

[adaface_ir101_webface12m.ckpt](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view)

**4c. Place the file in the weights folder**

```powershell
mkdir AdaFace\weights
# Move the downloaded file to:
# profile-clustering/AdaFace/weights/adaface_ir101_webface12m.ckpt
```

Expected path:

```
profile-clustering/AdaFace/weights/adaface_ir101_webface12m.ckpt
```

Alternate location (also supported): `profile-clustering/weights/adaface_ir101_webface12m.ckpt`

On first run, you should see: `AdaFace IR-101 (WebFace12M) loaded`. If weights are missing, clustering falls back without AdaFace (lower quality).

### 5. Python virtual environments (two venvs)

The projects use **different dependency sets**. Use two separate venvs.

**profile-clustering (`clustervenv`):**

```powershell
cd profile-clustering
python -m venv clustervenv
.\clustervenv\Scripts\Activate.ps1
pip install -r requirements.txt
deactivate
cd ..
```

**Best-Face (`bestfaceharsh`):**

```powershell
cd Best-Face
python -m venv bestfaceharsh
.\bestfaceharsh\Scripts\Activate.ps1
pip install -r requirements.txt
deactivate
cd ..
```

---



manually in the **profile-clustering** venv:

```powershell
cd profile-clustering
.\clustervenv\Scripts\Activate.ps1

python export_for_bestface.py "input_videos/my_classroom.mp4" `
  --cache "input_videos/my_classroom_detections.pkl" `
  --out "input_videos/my_classroom_clusters.json"
```

- First run **without** a `.pkl` cache runs full face detection (slow).  
- Later runs reuse `*_detections.pkl` unless you pass `--force-redetect`.  
- Optional contact sheets: `python export_cluster_profiles.py input_videos/my_classroom.mp4 --cache input_videos/my_classroom_detections.pkl`

### Step 3 — Best-face registration

 in the **Best-Face** venv:

```powershell
cd Best-Face\face_registration
..\..\Best-Face\bestfaceharsh\Scripts\python.exe scripts\register_from_json.py `
  "..\..\profile-clustering\input_videos\my_classroom_clusters.json"
```

**Output:** `Best-Face/face_registration/database/person_0001/best_face.jpg`, `metadata.json`, …

### Step 4 — Engagement analysis

Still in **profile-clustering** venv (reuses detection cache; re-runs clustering with current `KNOWN_N_PERSONS`):

```powershell
cd profile-clustering
.\clustervenv\Scripts\Activate.ps1

python test_engagement.py "input_videos/my_classroom.mp4" `
  --cache "input_videos/my_classroom_detections.pkl"
```

**Output:** `profile-clustering/input_videos/my_classroom_engagement.json` (per-person engagement scores, levels, trends).

---

 

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Wrong number of people | `KNOWN_N_PERSONS` in `face_cluster.py` |
| `AdaFace weights not found` | File at `AdaFace/weights/adaface_ir101_webface12m.ckpt` |
| `Missing: ...\clustervenv\...` | Create venvs (setup step 5) |
| Empty `database/` | Step 3 failed or all clusters were noise — inspect `*_clusters.json` |
| Engagement skipped | No valid person clusters after step 1+2 |

---

## License and third-party

- [AdaFace](https://github.com/mk-minchul/AdaFace) — clone locally; weights from [Google Drive](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view)
- [InsightFace](https://github.com/deepinsight/insightface) — face detection (via `requirements.txt`)

See `Best-Face/README.md` for API and configuration details for the registration service.
