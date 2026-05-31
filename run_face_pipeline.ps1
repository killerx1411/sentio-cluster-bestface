# Two-venv face pipeline: detect (cached) → cluster → best face
#
# Usage:
#   .\run_face_pipeline.ps1
#   .\run_face_pipeline.ps1 -Video "profile-clustering\input_videos\longvid.mp4" `
#       -Cache "profile-clustering\input_videos\longvid_detections.pkl"

param(
    [string]$Video = "profile-clustering\input_videos\longvid.mp4",
    [string]$Cache = "profile-clustering\input_videos\longvid_detections.pkl",
    [string]$JsonOut = "profile-clustering\input_videos\longvid_clusters.json",
    [int]$KnownN = 26
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

$ClusterPy = Join-Path $Root "profile-clustering\clustervenv\Scripts\python.exe"
$BestFacePy = Join-Path $Root "Best-Face\bestfaceharsh\Scripts\python.exe"
$ExportScript = Join-Path $Root "profile-clustering\export_for_bestface.py"
$RegisterScript = Join-Path $Root "Best-Face\face_registration\scripts\register_from_json.py"

foreach ($path in @($ClusterPy, $BestFacePy, $ExportScript, $RegisterScript)) {
    if (-not (Test-Path $path)) {
        Write-Error "Missing: $path"
    }
}

$videoPath = Join-Path $Root $Video
$cachePath = Join-Path $Root $Cache
$jsonPath = Join-Path $Root $JsonOut

Write-Host "`n=== Step 1+2: profile-clustering venv (cache + cluster) ===`n" -ForegroundColor Cyan
Push-Location (Join-Path $Root "profile-clustering")
try {
    & $ClusterPy $ExportScript $videoPath --cache $cachePath --out $jsonPath --job-id "job_longvid"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

Write-Host "`n=== Step 3: Best-Face venv (best-face registration) ===`n" -ForegroundColor Cyan
Push-Location (Join-Path $Root "Best-Face\face_registration")
try {
    & $BestFacePy $RegisterScript $jsonPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
