**FACE-BASED ATTENDANCE SYSTEM**

**Best Face Selection & Identity Registration Module**

Production Architecture \| E2E Pipeline \| Integration Guide

REST Input Contract \| Test Cases \| Repository Structure

**Table of Contents**

  -----------------------------------------------------------------------
  **1. SYSTEM OVERVIEW**

  -----------------------------------------------------------------------

**1.1 Purpose**

This module operates as Stage 5 in the Face Attendance Pipeline. It
receives clustered face data from the upstream pipeline (Stages 1-4) via
a REST API, selects the single highest-quality representative face per
individual, assigns a permanent identity, and writes the result to a
structured attendance database.

**1.2 Position in the Full Pipeline**

  ------------------------------------------------------------------------------
  **Stage**   **Component**         **Owner**     **Output**
  ----------- --------------------- ------------- ------------------------------
  1           Video Ingestion &     Upstream Team Frame images (JPEG/PNG)
              Frame Extraction                    

  2           Face Detection (MTCNN Upstream Team Bounding boxes + landmarks
              / RetinaFace)                       

  3           Embedding Generation  Upstream Team Embedding vectors (512-d)
              (ArcFace/FaceNet)                   

  4           Face Clustering       Upstream Team Cluster objects with face list
              (DBSCAN / AHC)                      

  5 ← YOU ARE Best Face Selection & Your Module   Registered DB: person_XXXX/
  HERE        Identity Registration               

  6           Attendance            Downstream    Attendance records
              Recognition &                       
              Matching                            
  ------------------------------------------------------------------------------

**1.3 Design Goals**

-   Latency: process 1 000 clusters in \< 5 seconds on a single 8-core
    machine

-   Correctness: always select the most frontal, sharp, well-lit face
    per cluster

-   Resilience: graceful degradation when all faces fail quality
    thresholds

-   Integration: single REST endpoint as entry point; no shared
    file-system dependency

-   Observability: structured JSON logs + Prometheus metrics at every
    stage

-   Idempotency: re-running on same cluster set must produce identical
    output

  -----------------------------------------------------------------------
  **2. END-TO-END PIPELINE FLOW**

  -----------------------------------------------------------------------

**2.1 High-Level Architecture Diagram**

  -----------------------------------------------------------------------
  ┌─────────────────────────────────────────────────────────────────┐

  │ UPSTREAM PIPELINE (Other Team) │

  │ Video ─► Frame Extract ─► Face Detect ─► Embed ─► Cluster │

  └──────────────────────────────┬──────────────────────────────────┘

  │ POST /api/v1/register

  │ Content-Type: application/json

  ▼

  ┌─────────────────────────────────────────────────────────────────┐

  │ YOUR MODULE (Stage 5) │

  │ │

  │ REST Gateway │

  │ └─► Input Validator ──► Deserialiser │

  │ │ │

  │ ▼ │

  │ Noise Filter (drop cluster.id == -1) │

  │ │ │

  │ ▼ │

  │ Pre-Filter (size / conf / quality floor checks) │

  │ │ │

  │ ▼ │

  │ Temporal Deduplicator (collapse near-duplicate frames) │

  │ │ │

  │ ▼ │

  │ Multi-Factor Scorer (composite quality score) │

  │ │ │

  │ ▼ │

  │ Best Face Selector (argmax over scored candidates) │

  │ │ │

  │ ▼ │

  │ Identity Generator (person_XXXX + UUID) │

  │ │ │

  │ ▼ │

  │ Database Writer (image + metadata.json) │

  │ │ │

  │ ▼ │

  │ Response Builder (registration report JSON) │

  └──────────────────────────────┬──────────────────────────────────┘

  │ HTTP 200 / 422 / 500

  ▼

  ┌─────────────────────────────────────────────────────────────────┐

  │ DOWNSTREAM (Attendance Recognition) │

  │ Reads database/ + embedding vectors for face matching │

  └─────────────────────────────────────────────────────────────────┘
  -----------------------------------------------------------------------

**2.2 Detailed Stage-by-Stage Flow**

**Stage A --- Input Validation & Deserialisation**

-   Receive JSON payload from upstream team via POST /api/v1/register

-   Validate schema using Pydantic model (strict mode)

-   Base64-decode crop_bgr images → numpy uint8 arrays

-   Deserialise embedding vectors from list\[float\] → np.ndarray
    float32

-   Fail fast: return HTTP 422 with per-field error detail on schema
    violation

**Stage B --- Noise Cluster Filtering**

-   Drop any cluster where cluster.id == -1 (DBSCAN noise label)

-   Log count of dropped noise clusters for observability

**Stage C --- Per-Face Pre-Filtering (Hard Thresholds)**

-   For each face in each valid cluster, apply hard threshold checks:

    -   min face dimension \>= MIN_FACE_PX (default: 60 px)

    -   detection confidence \>= MIN_CONF (default: 0.70)

    -   quality_score \>= MIN_QUALITY (default: 0.20)

    -   frontality_score \>= MIN_FRONT (default: 0.40)

    -   illumination_score \>= MIN_ILLUM (default: 0.25)

-   If no faces pass → keep all faces as fallback pool (never discard a
    real person)

**Stage D --- Temporal Deduplication**

-   Sort candidate faces by timestamp_sec ascending

-   Slide a 1-second window: if two faces fall within the window, keep
    higher-scored one

-   Reduces redundant near-duplicate video frames before scoring

**Stage E --- Multi-Factor Quality Scoring**

-   Compute five sub-scores per face (see Section 4 for all formulas)

-   Combine into composite_score with weighted sum

-   Attach \_score to each face object

**Stage F --- Best Face Selection**

-   Select face with maximum composite_score from deduplicated candidate
    pool

-   Deterministic tie-break: earlier frame_idx wins

**Stage G --- Identity Generation**

-   Assign sequential person_XXXX ID (zero-padded to max cluster count
    width)

-   Generate UUID v4 for stable cross-system reference

-   Check for existing registration to enforce idempotency

**Stage H --- Database Write**

-   Save best_face.jpg with JPEG quality=95 using OpenCV

-   Save metadata.json with all scores, bbox, embedding, timestamps

-   Atomic write: write to .tmp file first, then os.rename() for crash
    safety

**Stage I --- Response**

-   Return JSON report with per-cluster registration summary

-   Include: person_id, composite_score, timestamp, cluster_id, warnings

  -----------------------------------------------------------------------
  **3. REST INTEGRATION CONTRACT**

  -----------------------------------------------------------------------

This section defines the exact API contract. The upstream team must send
data matching this schema. All integration happens through this single
endpoint --- no shared filesystem, no shared database, no direct
function calls required.

**3.1 Endpoint Definition**

  -----------------------------------------------------------------------
  **Property**       **Value**
  ------------------ ----------------------------------------------------
  Method             POST

  URL                /api/v1/register

  Content-Type       application/json

  Auth               Bearer token via Authorization header (configurable)

  Idempotency        Supported --- same cluster_id re-submission is a
                     no-op

  Max Payload        100 MB (configurable in settings.py)
  -----------------------------------------------------------------------

**3.2 Request Schema**

  -----------------------------------------------------------------------
  POST /api/v1/register

  Content-Type: application/json

  {

  \"job_id\": \"video_job_2024_001\", // str --- upstream batch
  identifier

  \"video_id\": \"lecture_hall_2024_10_01\", // str --- source video
  reference

  \"clusters\": \[ // list\[Cluster\]

  {

  \"id\": 3, // int --- cluster ID (-1 = noise)

  \"faces\": \[ // list\[FaceDetection\]

  {

  \"frame_idx\": 1420, // int

  \"timestamp_sec\": 47.3, // float

  \"embedding\": \[0.12, -0.34, \...\],// list\[float\] 512-d or 128-d

  \"crop_bgr\": \"\<base64_string\>\", // str --- base64-encoded JPEG/PNG

  \"quality_score\": 0.81, // float \[0.0, 1.0\]

  \"bbox\": \[120, 85, 220, 185\],// list\[int\] \[x1,y1,x2,y2\]

  \"confidence\": 0.96, // float \[0.0, 1.0\]

  \"landmarks\": {

  \"left_eye\": \[145, 105\], // list\[int\] \[x, y\]

  \"right_eye\": \[195, 105\],

  \"nose\": \[170, 125\],

  \"mouth_left\": \[150, 150\],

  \"mouth_right\": \[190, 150\]

  }

  }

  \]

  }

  \]

  }
  -----------------------------------------------------------------------

**3.3 Response Schema**

  -----------------------------------------------------------------------
  HTTP 200 OK

  Content-Type: application/json

  {

  \"job_id\": \"video_job_2024_001\",

  \"video_id\": \"lecture_hall_2024_10_01\",

  \"status\": \"success\", // success \| partial \| failed

  \"registered\": 42, // int --- new registrations

  \"skipped\": 3, // int --- already registered (idempotent)

  \"noise_dropped\": 5, // int --- cluster.id == -1 discarded

  \"results\": \[

  {

  \"cluster_id\": 3,

  \"person_id\": \"person_0001\",

  \"uuid\": \"f47ac10b-58cc-4372-a567-0e02b2c3d479\",

  \"composite_score\": 0.847,

  \"frame_idx\": 1420,

  \"timestamp_sec\": 47.3,

  \"image_path\": \"database/person_0001/best_face.jpg\",

  \"fallback_used\": false, // true if all faces failed pre-filter

  \"warnings\": \[\] // list of warning strings if any

  }

  \],

  \"processing_time_ms\": 312

  }
  -----------------------------------------------------------------------

**3.4 Error Responses**

  ----------------------------------------------------------------------------
  **HTTP Code**   **Reason**                 **Response Body Key**
  --------------- -------------------------- ---------------------------------
  422             Schema validation failure  \"detail\": \[{ \"loc\":
  Unprocessable   (missing field, wrong      \[\...\], \"msg\": \"\...\" }\]
  Entity          type)                      

  400 Bad Request No valid clusters after    \"error\": \"no_valid_clusters\"
                  noise filtering            

  413 Payload Too Payload exceeds            \"error\": \"payload_too_large\"
  Large           MAX_PAYLOAD_MB             

  401             Missing or invalid Bearer  \"error\": \"unauthorized\"
  Unauthorized    token                      

  500 Internal    Unexpected error during    \"error\": \"internal\",
  Server Error    processing                 \"trace_id\": \"\...\"
  ----------------------------------------------------------------------------

  -----------------------------------------------------------------------
  **4. QUALITY SCORING --- ALL FORMULAS**

  -----------------------------------------------------------------------

**4.1 Sub-Score Definitions**

**Sub-Score 1 --- Frontality Score F(face)**

Measures how directly the face is pointing at the camera. Derived from
2D landmark geometry.

  -----------------------------------------------------------------------
  Inputs: left_eye = (lx, ly)

  right_eye = (rx, ry)

  nose = (nx, ny)

  eye_span = \|rx - lx\| + ε (ε = 1e-5, avoids div-by-zero)

  eye_y_diff = \|ly - ry\| (tilt: should be 0 for frontal)

  nose_dev = \|nx - (lx + rx) / 2\| (should be 0 for frontal)

  F(face) = clip( 1 - (eye_y_diff + nose_dev) / eye_span , 0, 1)

  Range: \[0.0 = pure side-face, 1.0 = perfect frontal\]
  -----------------------------------------------------------------------

**Sub-Score 2 --- Size Score S(face)**

Rewards larger detected faces, which carry more detail and are more
reliable for matching.

  -----------------------------------------------------------------------
  Inputs: bbox = \[x1, y1, x2, y2\]

  REF_AREA = 250 \* 250 = 62500 px² (tunable constant)

  face_area = (x2 - x1) \* (y2 - y1)

  S(face) = clip( face_area / REF_AREA , 0, 1)

  Range: \[0.0, 1.0\] --- capped at 1.0 for very large faces
  -----------------------------------------------------------------------

**Sub-Score 3 --- Illumination Score L(face)**

Measures whether the face is properly and evenly lit. Uses the L\*
(perceptual lightness) channel of the CIE LAB colour space.

  -----------------------------------------------------------------------
  Inputs: crop_bgr (H x W x 3 uint8 numpy array)

  1\. Convert: lab = BGR → LAB

  2\. Extract: L\* channel (0--255 scale in OpenCV)

  3\. Compute: μ = mean(L\*)

  σ = std(L\*)

  brightness_score = 1 - \|μ - 127\| / 127 (penalty for dark or
  overexposed)

  contrast_score = clip( σ / 60, 0, 1 ) (higher σ = more texture)

  L(face) = 0.6 × brightness_score + 0.4 × contrast_score

  Range: \[0.0, 1.0\]
  -----------------------------------------------------------------------

**Sub-Score 4 --- Pipeline Quality Score Q(face)**

The quality_score already provided by the upstream pipeline (sharpness /
blur detection). Used directly.

  -----------------------------------------------------------------------
  Q(face) = face.quality_score ∈ \[0.0, 1.0\]

  -----------------------------------------------------------------------

**Sub-Score 5 --- Detection Confidence C(face)**

The confidence score from the face detector (MTCNN / RetinaFace). Higher
confidence means the face region is more reliably detected.

  -----------------------------------------------------------------------
  C(face) = face.confidence ∈ \[0.0, 1.0\]

  -----------------------------------------------------------------------

**4.2 Composite Score**

All five sub-scores are combined into a single composite score using a
weighted sum. Weights are configurable in settings.py.

  -----------------------------------------------------------------------
  composite_score(face) =

  w_Q × Q(face) + // pipeline quality (sharpness/blur)

  w_F × F(face) + // frontality (landmark geometry)

  w_S × S(face) + // face size (bbox area)

  w_C × C(face) + // detection confidence

  w_L × L(face) // illumination (LAB lightness)

  Default weights (sum = 1.0):

  w_Q = 0.30

  w_F = 0.25

  w_S = 0.20

  w_C = 0.15

  w_L = 0.10

  Range: \[0.0, 1.0\]

  Best face = argmax_f composite_score(f) for f in candidate_pool

  Tie-break: min(frame_idx) --- earlier frame wins
  -----------------------------------------------------------------------

**4.3 Pre-Filter Hard Thresholds**

A face must pass ALL of the following checks to enter the candidate
pool. These are not scored --- they are binary pass/fail gates applied
before scoring.

  ----------------------------------------------------------------------------
  **Check**         **Formula**             **Default**   **Config Key**
  ----------------- ----------------------- ------------- --------------------
  Minimum face size min(x2-x1, y2-y1) \>=   60 px         MIN_FACE_PX
                    MIN_FACE_PX                           

  Detection         C(face) \>= MIN_CONF    0.70          MIN_CONF
  confidence                                              

  Pipeline quality  Q(face) \>= MIN_QUALITY 0.20          MIN_QUALITY

  Frontality        F(face) \>= MIN_FRONT   0.40          MIN_FRONT

  Illumination      L(face) \>= MIN_ILLUM   0.25          MIN_ILLUM
  ----------------------------------------------------------------------------

**4.4 Temporal Deduplication Window**

  -----------------------------------------------------------------------
  Algorithm:

  1\. Sort faces by timestamp_sec ascending

  2\. Initialise kept = \[faces\[0\]\]

  3\. For each subsequent face f:

  if f.timestamp_sec - kept\[-1\].timestamp_sec \> DEDUP_GAP_SEC:

  kept.append(f) // new time window → keep

  else:

  if f.\_score \> kept\[-1\].\_score:

  kept\[-1\] = f // same window → replace if better

  Default: DEDUP_GAP_SEC = 1.0 // configurable in settings.py
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  **5. INPUTS & EXPECTED OUTPUTS**

  -----------------------------------------------------------------------

**5.1 Module Inputs**

  ---------------------------------------------------------------------------------------------------------
  **Input Field**                        **Type**        **Required**   **Valid Range / Notes**
  -------------------------------------- --------------- -------------- -----------------------------------
  job_id                                 string          Yes            Non-empty, max 128 chars

  video_id                               string          Yes            Non-empty, max 256 chars

  clusters\[\].id                        int             Yes            -1 for noise, ≥0 for valid clusters

  clusters\[\].faces\[\].frame_idx       int             Yes            ≥ 0

  clusters\[\].faces\[\].timestamp_sec   float           Yes            ≥ 0.0

  clusters\[\].faces\[\].embedding       list\[float\]   Yes            Length 128 or 512, values ∈ \[-1,
                                                                        1\]

  clusters\[\].faces\[\].crop_bgr        string          Yes            Base64-encoded JPEG or PNG

  clusters\[\].faces\[\].quality_score   float           Yes            \[0.0, 1.0\]

  clusters\[\].faces\[\].bbox            list\[int\]     Yes            \[x1, y1, x2, y2\], all ≥ 0

  clusters\[\].faces\[\].confidence      float           Yes            \[0.0, 1.0\]

  clusters\[\].faces\[\].landmarks       object          Yes            5 keys: left_eye, right_eye, nose,
                                                                        mouth_left, mouth_right

  clusters\[\].faces\[\].landmarks.\*    list\[int\]     Yes            \[x, y\] coordinates ≥ 0
  ---------------------------------------------------------------------------------------------------------

**5.2 Module Outputs**

**Filesystem Output**

  -----------------------------------------------------------------------
  database/

  person_0001/

  best_face.jpg ← JPEG quality=95, BGR colour space

  metadata.json ← see schema below

  person_0002/

  best_face.jpg

  metadata.json

  \...

  metadata.json schema:

  {

  \"person_id\": \"person_0001\",

  \"uuid\": \"f47ac10b-\...\",

  \"cluster_id\": 3,

  \"job_id\": \"video_job_2024_001\",

  \"video_id\": \"lecture_hall_2024_10_01\",

  \"composite_score\": 0.847,

  \"sub_scores\": {

  \"quality\": 0.81,

  \"frontality\": 0.89,

  \"size\": 0.76,

  \"confidence\": 0.96,

  \"illumination\":0.72

  },

  \"frame_idx\": 1420,

  \"timestamp_sec\": 47.3,

  \"bbox\": \[120, 85, 220, 185\],

  \"confidence\": 0.96,

  \"quality_score\": 0.81,

  \"fallback_used\": false,

  \"embedding\": \[0.12, -0.34, \...\],

  \"registered_at\": \"2024-10-01T09:15:33Z\"

  }
  -----------------------------------------------------------------------

**5.3 Performance Targets**

  ------------------------------------------------------------------------
  **Metric**           **Target**         **Notes**
  -------------------- ------------------ --------------------------------
  Throughput           ≥ 200 clusters /   On 8-core CPU with
                       second             ThreadPoolExecutor

  End-to-end latency   \< 5 seconds       Excluding network I/O
  (1000 clusters)                         

  Image write time per \< 5 ms            JPEG quality=95, OpenCV
  face                                    cv2.imwrite

  Memory peak          \< 2 GB            For 1000 clusters with 50 faces
                                          each

  Cold start time      \< 500 ms          FastAPI + module import

  REST response time   \< 1 second for    Observed p99 target
                       100 clusters       
  ------------------------------------------------------------------------

  -----------------------------------------------------------------------
  **6. PRODUCTION ARCHITECTURE**

  -----------------------------------------------------------------------

**6.1 Component Stack**

  ---------------------------------------------------------------------------------------------
  **Layer**          **Technology**                          **Why**
  ------------------ --------------------------------------- ----------------------------------
  REST API Framework FastAPI + Uvicorn                       Async, auto OpenAPI docs, Pydantic
                                                             validation built-in

  Input Validation   Pydantic v2 (strict mode)               Zero-boilerplate schema
                                                             enforcement, fast C core

  Image Processing   OpenCV (cv2)                            Fastest CPU-side BGR ops; no PIL
                                                             overhead

  Numerical          NumPy                                   Vectorised scoring; no Python
                                                             loops on arrays

  Parallelism        concurrent.futures.ThreadPoolExecutor   I/O-bound disk writes benefit from
                                                             threads

  Persistence        Local filesystem (POSIX) /              Swap via StorageBackend interface
                     S3-compatible                           

  Structured Logging structlog + JSON formatter              Machine-parseable logs for ELK /
                                                             CloudWatch

  Metrics            Prometheus client                       Latency histograms, counters per
                                                             pipeline stage

  Config             Pydantic BaseSettings + .env            All thresholds and weights
                                                             externally configurable

  Containerisation   Docker + docker-compose                 Reproducible build; single
                                                             Dockerfile
  ---------------------------------------------------------------------------------------------

**6.2 Concurrency Model**

Processing of clusters is parallelised using a ThreadPoolExecutor
because the bottleneck is disk I/O (image writes), not CPU. Worker count
scales with machine core count.

  -----------------------------------------------------------------------
  max_workers = min(32, os.cpu_count() \* 2) // I/O bound → 2× CPU count

  with ThreadPoolExecutor(max_workers=max_workers) as executor:

  futures = {executor.submit(process_cluster, c, idx): c

  for idx, c in enumerate(valid_clusters)}

  for future in as_completed(futures):

  result = future.result() // exceptions bubble up here
  -----------------------------------------------------------------------

**6.3 Atomic File Writes**

To prevent partial writes on crash or power loss, all files are written
atomically:

  -----------------------------------------------------------------------
  \# Write to temp file first, then atomic rename

  tmp_path = image_path + \".tmp\"

  cv2.imwrite(tmp_path, crop_bgr, \[cv2.IMWRITE_JPEG_QUALITY, 95\])

  os.rename(tmp_path, image_path) // POSIX atomic on same filesystem
  -----------------------------------------------------------------------

**6.4 Idempotency**

Re-sending the same cluster data must produce no side effects. The
module checks for existing person directories before processing:

  -----------------------------------------------------------------------
  if os.path.exists(os.path.join(out_root, person_id,
  \"best_face.jpg\")):

  return RegistrationResult(person_id=person_id, status=\"skipped\")
  -----------------------------------------------------------------------

**6.5 Configuration Reference**

  ----------------------------------------------------------------------------
  **Config Key**    **Default**     **Description**
  ----------------- --------------- ------------------------------------------
  MIN_FACE_PX       60              Minimum face side in pixels

  MIN_CONF          0.70            Minimum detector confidence

  MIN_QUALITY       0.20            Minimum pipeline quality score

  MIN_FRONT         0.40            Minimum frontality score

  MIN_ILLUM         0.25            Minimum illumination score

  DEDUP_GAP_SEC     1.0             Temporal dedup window (seconds)

  JPEG_QUALITY      95              JPEG compression quality (0-100)

  REF_AREA          62500           Reference face area for size score (px²)

  W_QUALITY         0.30            Weight for pipeline quality score

  W_FRONT           0.25            Weight for frontality score

  W_SIZE            0.20            Weight for size score

  W_CONF            0.15            Weight for detection confidence

  W_ILLUM           0.10            Weight for illumination score

  MAX_WORKERS       cpu_count \* 2  ThreadPoolExecutor worker count

  DATABASE_ROOT     \"database/\"   Root path for output storage

  MAX_PAYLOAD_MB    100             Maximum accepted REST payload size

  STORAGE_BACKEND   \"local\"       \"local\" or \"s3\"
  ----------------------------------------------------------------------------

  -----------------------------------------------------------------------
  **7. REPOSITORY STRUCTURE**

  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  face_registration/

  │

  ├── app/ \# Application source

  │ ├── \_\_init\_\_.py

  │ ├── main.py \# FastAPI app entry point

  │ ├── api/

  │ │ ├── \_\_init\_\_.py

  │ │ ├── routes.py \# POST /api/v1/register endpoint

  │ │ └── dependencies.py \# Auth, rate-limit dependencies

  │ ├── core/

  │ │ ├── \_\_init\_\_.py

  │ │ ├── config.py \# Pydantic BaseSettings + all config keys

  │ │ └── logging.py \# structlog JSON setup

  │ ├── models/

  │ │ ├── \_\_init\_\_.py

  │ │ ├── request.py \# Pydantic input schema (ClusterPayload)

  │ │ └── response.py \# Pydantic output schema (RegistrationReport)

  │ ├── pipeline/

  │ │ ├── \_\_init\_\_.py

  │ │ ├── deserialiser.py \# base64 → numpy, list → np.ndarray

  │ │ ├── noise_filter.py \# drop cluster.id == -1

  │ │ ├── pre_filter.py \# hard threshold gates

  │ │ ├── deduplicator.py \# temporal dedup window

  │ │ ├── scorer.py \# all 5 sub-scores + composite

  │ │ ├── selector.py \# argmax best face

  │ │ ├── identity.py \# person_XXXX + UUID generation

  │ │ └── orchestrator.py \# wires all stages, ThreadPoolExecutor

  │ └── storage/

  │ ├── \_\_init\_\_.py

  │ ├── base.py \# StorageBackend ABC

  │ ├── local.py \# LocalFileStorage implementation

  │ └── s3.py \# S3Storage implementation (optional)

  │

  ├── tests/

  │ ├── \_\_init\_\_.py

  │ ├── conftest.py \# shared fixtures (mock clusters, mock faces)

  │ ├── unit/

  │ │ ├── test_noise_filter.py

  │ │ ├── test_pre_filter.py

  │ │ ├── test_deduplicator.py

  │ │ ├── test_scorer.py \# one test per formula

  │ │ ├── test_selector.py

  │ │ └── test_identity.py

  │ └── integration/

  │ ├── test_api_register.py \# full REST round-trip tests

  │ └── test_pipeline_e2e.py \# end-to-end with real image fixtures

  │

  ├── fixtures/ \# Test image assets

  │ ├── frontal_good.jpg

  │ ├── side_face.jpg

  │ ├── blurry.jpg

  │ ├── dark.jpg

  │ └── sample_payload.json \# Example valid REST payload

  │

  ├── scripts/

  │ ├── benchmark.py \# latency benchmark (1000 cluster test)

  │ └── generate_sample_payload.py \# generate test fixtures

  │

  ├── database/ \# Output --- git-ignored

  │ └── .gitkeep

  │

  ├── Dockerfile

  ├── docker-compose.yml

  ├── requirements.txt

  ├── .env.example \# All config keys with defaults

  ├── pytest.ini

  └── README.md
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  **8. TEST CASES**

  -----------------------------------------------------------------------

**8.1 Unit Tests --- Scorer**

  --------------------------------------------------------------------------------------------
  **Test   **Test Name**                   **Input**       **Expected    **Validates**
  ID**                                                     Output**      
  -------- ------------------------------- --------------- ------------- ---------------------
  U-S-01   test_frontality_perfect         Symmetric       F ≈ 1.0       F formula
                                           landmarks, nose               
                                           centred                       

  U-S-02   test_frontality_side_face       Large eye       F \< 0.4      F penalty
                                           Y-diff, nose                  
                                           far from                      
                                           midpoint                      

  U-S-03   test_frontality_no_eye_span     left_eye ==     F = 0.0 (no   ε guard
                                           right_eye       crash)        
                                           (degenerate)                  

  U-S-04   test_size_large_face            bbox 300×300    S = 1.0       S clip
                                                           (capped)      

  U-S-05   test_size_tiny_face             bbox 30×30      S ≈ 0.014     S scale

  U-S-06   test_illumination_normal        LAB mean≈127,   L ≈ 0.9+      L happy path
                                           std≈55                        

  U-S-07   test_illumination_dark          LAB mean≈20,    L \< 0.4      L dark penalty
                                           std≈10                        

  U-S-08   test_illumination_overexposed   LAB mean≈250,   L \< 0.3      L bright penalty
                                           std≈5                         

  U-S-09   test_composite_weights_sum      Mock sub-scores composite =   Weight sum = 1
                                           all = 1.0       1.0           

  U-S-10   test_composite_deterministic    Same face twice Same score    Determinism
  --------------------------------------------------------------------------------------------

**8.2 Unit Tests --- Pre-Filter**

  ---------------------------------------------------------------------------------------
  **Test   **Test Name**                **Input**        **Expected Output**
  ID**                                                   
  -------- ---------------------------- ---------------- --------------------------------
  U-F-01   test_passes_all_thresholds   Good face, all   passes_filter = True
                                        metrics above    
                                        min              

  U-F-02   test_fails_tiny_face         bbox 40×40       passes_filter = False

  U-F-03   test_fails_low_confidence    confidence =     passes_filter = False
                                        0.50             

  U-F-04   test_fails_low_quality       quality_score =  passes_filter = False
                                        0.10             

  U-F-05   test_fails_side_face         frontality =     passes_filter = False
                                        0.20             

  U-F-06   test_fallback_all_fail       All faces fail   Returns all faces (fallback
                                        filter           pool)
  ---------------------------------------------------------------------------------------

**8.3 Unit Tests --- Deduplicator**

  ---------------------------------------------------------------------------------------------
  **Test   **Test Name**                      **Input**          **Expected Output**
  ID**                                                           
  -------- ---------------------------------- ------------------ ------------------------------
  U-D-01   test_dedup_keeps_distant_frames    Faces at t=0.0,    All 3 kept
                                              2.0, 4.0 sec       

  U-D-02   test_dedup_collapses_same_window   Faces at t=0.0,    1 kept (highest score)
                                              0.3, 0.7 sec       

  U-D-03   test_dedup_selects_better_score    t=0.2 score=0.8,   Face at t=0.5 kept
                                              t=0.5 score=0.9    

  U-D-04   test_dedup_single_face             Only one face      That face returned

  U-D-05   test_dedup_boundary_exact          Faces exactly 1.0  Both kept (boundary inclusive)
                                              sec apart          
  ---------------------------------------------------------------------------------------------

**8.4 Unit Tests --- Noise Filter & Identity**

  ------------------------------------------------------------------------------------------
  **Test   **Test Name**                   **Input**          **Expected Output**
  ID**                                                        
  -------- ------------------------------- ------------------ ------------------------------
  U-N-01   test_noise_cluster_dropped      Cluster with id=-1 Not in valid_clusters

  U-N-02   test_valid_cluster_kept         Cluster with id=0  In valid_clusters

  U-N-03   test_mixed_clusters             3 valid + 2 noise  valid_clusters length = 3
                                           clusters           

  U-I-01   test_identity_padding_4digits   \< 1000 clusters   person_0001 format

  U-I-02   test_identity_padding_5digits   \> 9999 clusters   person_00001 format

  U-I-03   test_uuid_is_unique             Two separate calls Different UUIDs
  ------------------------------------------------------------------------------------------

**8.5 Integration Tests --- REST API**

  --------------------------------------------------------------------------------------------------------
  **Test   **Test Name**                       **Input**       **Expected HTTP**   **Verifies**
  ID**                                                                             
  -------- ----------------------------------- --------------- ------------------- -----------------------
  I-A-01   test_valid_payload_registers        Valid JSON with 200 + registered=5  Happy path
                                               5 clusters                          

  I-A-02   test_noise_only_payload             All clusters    400                 Noise-only rejection
                                               id=-1           no_valid_clusters   

  I-A-03   test_missing_embedding_field        Face missing    422 + field detail  Schema validation
                                               embedding key                       

  I-A-04   test_idempotent_resubmission        Same payload    200 + skipped=5 on  Idempotency
                                               twice           2nd                 

  I-A-05   test_empty_clusters_list            \"clusters\":   400                 Empty input
                                               \[\]            no_valid_clusters   

  I-A-06   test_invalid_bearer_token           Wrong           401 unauthorized    Auth
                                               Authorization                       
                                               header                              

  I-A-07   test_response_contains_image_path   Valid payload   image_path field    File written
                                                               present + file      
                                                               exists              

  I-A-08   test_response_time_100_clusters     100 valid       p99 \< 1000 ms      Latency SLA
                                               clusters                            
  --------------------------------------------------------------------------------------------------------

**8.6 End-to-End Tests**

  -----------------------------------------------------------------------------------------------------
  **Test   **Test Name**                          **Scenario**          **Assertions**
  ID**                                                                  
  -------- -------------------------------------- --------------------- -------------------------------
  E-01     test_e2e_best_frontal_selected         Cluster with 1 good   Frontal image saved;
                                                  frontal + 3 side      metadata.json has correct
                                                  faces                 composite score

  E-02     test_e2e_all_faces_filtered_fallback   Cluster where every   A face is still selected;
                                                  face fails pre-filter fallback_used=true in metadata

  E-03     test_e2e_1000_cluster_benchmark        1000 auto-generated   Completes in \< 5 s; 1000
                                                  clusters              directories created

  E-04     test_e2e_embedding_saved               Valid cluster         metadata.json embedding list ==
                                                                        input embedding

  E-05     test_e2e_database_structure            10 valid clusters     database/person_0001/ through
                                                                        person_0010/ each have
                                                                        best_face.jpg + metadata.json

  E-06     test_e2e_crash_recovery                Kill process          No partial/corrupt files;
                                                  mid-write; restart    incomplete .tmp files cleaned
                                                                        up
  -----------------------------------------------------------------------------------------------------

  -----------------------------------------------------------------------
  **9. LATENCY OPTIMISATION GUIDE**

  -----------------------------------------------------------------------

  --------------------------------------------------------------------------
  **Optimisation**     **Technique**            **Impact**
  -------------------- ------------------------ ----------------------------
  Pre-filter before    Hard thresholds reject   Avoids LAB conversion +
  scoring              bad faces before any     landmark math on garbage
                       math                     

  Dedup before scoring Collapse temporal        Reduces scored candidates by
                       duplicates before        \~40-60% in typical video
                       scoring loop             

  JPEG not PNG         cv2.imwrite with JPEG    5× smaller file, 3× faster
                       quality=95               write

  ThreadPoolExecutor   Parallelize disk writes  Near-linear speedup on
                       across clusters          I/O-bound workload

  NumPy vectorisation  No Python loops on array Scoring is O(1) per face
                       operations               with no interpreter overhead

  Pydantic strict mode Fail fast on bad input,  Validation is free for valid
                       no extra coercion passes payloads

  Idempotency skip     Check file existence     Re-runs are near-instant
                       before re-processing     

  base64 decode once   Decode crop_bgr at       Prevents repeated decode of
                       deserialisation, not     same bytes
                       per-stage                

  Embedding as list    Store in metadata.json;  Downstream matching avoids
                       no re-encode on          re-running encoder
                       downstream read          

  Cold start           Import scipy/cv2 at      First request is not
                       module level not inside  penalised
                       handler                  
  --------------------------------------------------------------------------

  -----------------------------------------------------------------------
  **10. QUICK INTEGRATION CHECKLIST (For Upstream Team)**

  -----------------------------------------------------------------------

The upstream team only needs to make ONE change to integrate: send a
POST request to the registration endpoint instead of writing cluster
data to a file. The checklist below covers everything required.

1.  Confirm REST endpoint URL and port with DevOps (default:
    http://localhost:8000/api/v1/register)

2.  Obtain Bearer token from the deployment configuration (.env
    AUTH_TOKEN)

3.  Serialise each Cluster object to the JSON schema defined in Section
    3.2

4.  Base64-encode each crop_bgr numpy array before sending
    (cv2.imencode + base64.b64encode)

5.  Ensure landmarks dict contains exactly these 5 keys: left_eye,
    right_eye, nose, mouth_left, mouth_right

6.  Ensure embedding is sent as list\[float\], not numpy array (call
    .tolist() on ndarray)

7.  Handle HTTP 422 by inspecting the detail field for per-field
    validation errors

8.  On HTTP 200, read results\[\].person_id to map cluster IDs to
    registered person identities

9.  On HTTP 200, results\[\].image_path contains the relative path to
    the saved face image

+-----------------------------------------------------------------------+
| IMPORTANT NOTE FOR UPSTREAM TEAM:                                     |
|                                                                       |
| The registration module is stateless with respect to your pipeline.   |
| You may send one job at a time                                        |
|                                                                       |
| or batch multiple jobs. Re-sending the same cluster IDs is safe ---   |
| they will be skipped (idempotent).                                    |
|                                                                       |
| The only shared contract is the JSON schema. No shared filesystem or  |
| database is needed.                                                   |
+-----------------------------------------------------------------------+

  -----------------------------------------------------------------------
  **APPENDIX --- DEPENDENCY LIST & DOCKER**

  -----------------------------------------------------------------------

**Requirements**

  -----------------------------------------------------------------------
  \# requirements.txt

  fastapi\>=0.111.0

  uvicorn\[standard\]\>=0.29.0

  pydantic\>=2.6.0

  pydantic-settings\>=2.2.0

  opencv-python-headless\>=4.9.0

  numpy\>=1.26.0

  structlog\>=24.1.0

  prometheus-client\>=0.20.0

  python-multipart\>=0.0.9

  boto3\>=1.34.0 \# only if STORAGE_BACKEND=s3

  \# dev / test

  pytest\>=8.0.0

  pytest-asyncio\>=0.23.0

  httpx\>=0.27.0 \# async test client for FastAPI

  faker\>=24.0.0 \# generate test payloads
  -----------------------------------------------------------------------

**Dockerfile**

  -----------------------------------------------------------------------
  FROM python:3.11-slim

  WORKDIR /app

  RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf
  /var/lib/apt/lists/\*

  COPY requirements.txt .

  RUN pip install \--no-cache-dir -r requirements.txt

  COPY app/ ./app/

  VOLUME \[\"/app/database\"\]

  EXPOSE 8000

  CMD \[\"uvicorn\", \"app.main:app\", \"\--host\", \"0.0.0.0\",
  \"\--port\", \"8000\", \"\--workers\", \"4\"\]
  -----------------------------------------------------------------------
