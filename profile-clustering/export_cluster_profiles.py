"""
export_cluster_profiles.py  —  Cluster visualization & export
==============================================================
Takes the output of face_cluster.run_pipeline() and saves a contact-sheet
image for each detected person inside  cluster_profiles/

LABELLING SCHEME:
  Persons 0-25   → A, B, C … Z
  Persons 26-51  → A1, B1 … Z1
  Persons 52-77  → A2, B2 … Z2
  … and so on.

OUTPUT STRUCTURE:
  cluster_profiles/
    A/
      A_detection_001.jpg   ← best face crop
      A_detection_002.jpg
      …
      A_sheet.jpg           ← contact-sheet of all detections (with timestamps)
    B/
      …
    noise/
      noise_detection_001.jpg
      noise_sheet.jpg

USAGE (standalone — runs the full pipeline and exports):
  python export_cluster_profiles.py <video_path>

USAGE (from another script — export only):
  from export_cluster_profiles import export_clusters
  clusters = run_pipeline("myvideo.mp4")   # from face_cluster
  export_clusters(clusters, out_dir="cluster_profiles")
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import face_cluster
# Make FaceDetection & ClusterResult findable under __main__ namespace
# so pickle can deserialize the .pkl that was saved from app.py/__main__
for _name in ('FaceDetection', 'ClusterResult'):
    if hasattr(face_cluster, _name):
        setattr(sys.modules['__main__'], _name, getattr(face_cluster, _name))
# ─── label generation ────────────────────────────────────────────────────────

def _person_label(person_index: int) -> str:
    """
    Maps a 0-based person index to an alphabetical label:
      0  → A
      25 → Z
      26 → A1
      51 → Z1
      52 → A2  … etc.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cycle, offset = divmod(person_index, 26)
    letter = letters[offset]
    return letter if cycle == 0 else f"{letter}{cycle}"


# ─── image helpers ────────────────────────────────────────────────────────────

def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((112, 112, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _thumb(img: np.ndarray, size: int = 160) -> np.ndarray:
    img = _ensure_bgr(img)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    # Pad to square
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _make_contact_sheet(
    faces,              # list of FaceDetection
    label: str,
    cols: int = 8,
    thumb_size: int = 160,
    padding: int = 6,
    header_h: int = 48,
) -> np.ndarray:
    """
    Builds a dark contact sheet of all face crops with timestamp captions.
    Sorted by quality_score descending so the best shots come first.
    """
    faces_sorted = sorted(faces, key=lambda f: -f.quality_score)
    n = len(faces_sorted)
    rows = math.ceil(n / cols)

    cell_w = thumb_size + padding
    cell_h = thumb_size + padding + 20          # 20px caption area
    total_w = cols * cell_w + padding
    total_h = header_h + rows * cell_h + padding

    # Dark background
    sheet = np.full((total_h, total_w, 3), 22, dtype=np.uint8)

    # Header
    cv2.putText(
        sheet, f"Person  {label}   ({n} detections)",
        (padding, header_h - 14),
        cv2.FONT_HERSHEY_DUPLEX, 0.7, (230, 230, 230), 1, cv2.LINE_AA,
    )

    # Cells
    for idx, face in enumerate(faces_sorted):
        row, col = divmod(idx, cols)
        x0 = padding + col * cell_w
        y0 = header_h + padding + row * cell_h

        thumb = _thumb(face.crop_bgr, thumb_size)
        sheet[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb

        # Quality badge (top-right corner)
        q_text = f"Q{face.quality_score:.2f}"
        q_color = (80, 220, 80) if face.quality_score > 0.5 else (220, 180, 80)
        cv2.putText(
            sheet, q_text,
            (x0 + thumb_size - 48, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, q_color, 1, cv2.LINE_AA,
        )

        # Timestamp caption
        ts = f"{face.timestamp_sec:.1f}s"
        cv2.putText(
            sheet, ts,
            (x0, y0 + thumb_size + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA,
        )

    return sheet


# ─── main export function ──────────────────────────────────────────────────────

def export_clusters(
    clusters,
    out_dir: str | Path = "cluster_profiles",
    thumb_size: int = 160,
    save_individual: bool = True,
    save_sheet: bool = True,
    jpeg_quality: int = 92,
) -> Path:
    """
    Export all clusters to out_dir/

    Parameters
    ----------
    clusters        : list[ClusterResult] from face_cluster.run_pipeline()
    out_dir         : root output directory (created if absent)
    thumb_size      : pixel size for thumbnail squares in the contact sheet
    save_individual : save each face crop as an individual JPEG
    save_sheet      : save a contact-sheet image per cluster
    jpeg_quality    : JPEG compression quality (0-100)

    Returns
    -------
    Path to the output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    person_idx = 0   # counter for label assignment (noise gets its own label)

    print(f"\n  Exporting clusters → {out_dir.resolve()}")

    for cluster in clusters:
        is_noise = cluster.id == -1

        if is_noise:
            label = "noise"
        else:
            label = _person_label(person_idx)
            person_idx += 1

        person_dir = out_dir / label
        person_dir.mkdir(parents=True, exist_ok=True)

        faces = cluster.faces
        faces_sorted = sorted(faces, key=lambda f: -f.quality_score)

        # ── individual crops ──────────────────────────────────────
        if save_individual:
            for i, face in enumerate(faces_sorted, start=1):
                crop = _ensure_bgr(face.crop_bgr)
                fname = person_dir / f"{label}_detection_{i:03d}.jpg"
                cv2.imwrite(str(fname), crop, encode_params)

        # ── contact sheet ─────────────────────────────────────────
        if save_sheet and faces:
            sheet = _make_contact_sheet(
                faces_sorted, label, thumb_size=thumb_size
            )
            sheet_path = person_dir / f"{label}_sheet.jpg"
            cv2.imwrite(str(sheet_path), sheet, encode_params)

        # ── console summary ───────────────────────────────────────
        ts_list = [f.timestamp_sec for f in faces]
        q_vals  = [f.quality_score for f in faces]
        label_str = "noise" if is_noise else f"Person {label}"
        print(
            f"  [{label_str:>10}]  "
            f"{len(faces):4d} detections  "
            f"t={min(ts_list):.1f}s–{max(ts_list):.1f}s  "
            f"avg_q={sum(q_vals)/len(q_vals):.1f}  "
            f"→ {person_dir.relative_to(out_dir.parent)}"
        )

    print(f"\n  ✓ Exported {person_idx} persons + {'1 noise' if any(c.id==-1 for c in clusters) else 'no noise'} cluster")
    print(f"  ✓ Output: {out_dir.resolve()}\n")
    return out_dir


# ─── summary index sheet ──────────────────────────────────────────────────────

def build_index_sheet(
    clusters,
    out_dir: str | Path = "cluster_profiles",
    thumb_size: int = 120,
) -> Path:
    """
    Builds a single  cluster_profiles/INDEX.jpg  showing one best-crop
    per person, labelled, in a grid. Useful for a quick overview.
    """
    out_dir = Path(out_dir)

    real_clusters = [c for c in clusters if c.id != -1]
    n = len(real_clusters)
    if n == 0:
        print("  No real clusters to index.")
        return out_dir

    cols = min(10, n)
    rows = math.ceil(n / cols)
    padding = 8
    caption_h = 22
    cell_w = thumb_size + padding
    cell_h = thumb_size + padding + caption_h
    header_h = 56

    total_w = cols * cell_w + padding
    total_h = header_h + rows * cell_h + padding

    index = np.full((total_h, total_w, 3), 18, dtype=np.uint8)
    cv2.putText(
        index, f"Cluster Index  —  {n} unique persons",
        (padding, header_h - 16),
        cv2.FONT_HERSHEY_DUPLEX, 0.75, (230, 230, 230), 1, cv2.LINE_AA,
    )

    person_idx = 0
    for cluster in clusters:
        if cluster.id == -1:
            continue
        label = _person_label(person_idx)
        person_idx += 1

        best_face = max(cluster.faces, key=lambda f: f.quality_score)
        thumb = _thumb(best_face.crop_bgr, thumb_size)

        row, col = divmod(person_idx - 1, cols)
        x0 = padding + col * cell_w
        y0 = header_h + padding + row * cell_h

        index[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb

        # Label below thumb
        cv2.putText(
            index, label,
            (x0 + thumb_size // 2 - 10, y0 + thumb_size + caption_h - 4),
            cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )
        # Detection count (small, top-left)
        cv2.putText(
            index, str(len(cluster.faces)),
            (x0 + 4, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1, cv2.LINE_AA,
        )

    index_path = out_dir / "INDEX.jpg"
    cv2.imwrite(str(index_path), index, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  ✓ Index sheet → {index_path}")
    return index_path


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running as:
    #   python export_cluster_profiles.py <video_path>
    # Or it can load an existing detection cache:
    #   python export_cluster_profiles.py <video_path.mp4> --cache <pkl_path>

    import argparse

    parser = argparse.ArgumentParser(
        description="Run face clustering and export profiles to cluster_profiles/"
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--cache", default=None,
        help="Path to detections .pkl cache (optional; auto-detected if absent)"
    )
    parser.add_argument(
        "--out", default="cluster_profiles",
        help="Output directory (default: cluster_profiles/)"
    )
    parser.add_argument(
        "--no-individual", action="store_true",
        help="Skip saving individual face crop images"
    )
    parser.add_argument(
        "--no-sheet", action="store_true",
        help="Skip saving contact-sheet images"
    )
    parser.add_argument(
        "--thumb-size", type=int, default=160,
        help="Thumbnail size in pixels (default: 160)"
    )
    args = parser.parse_args()

    # Import pipeline from face_cluster (must be in the same directory)
    try:
        from face_cluster import run_pipeline
    except ImportError as e:
        print(f"\n✗  Could not import facecluster: {e}")
        print("   Make sure app.py is in the same directory as this script.")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"\n✗  Video not found: {video_path}")
        sys.exit(1)

    clusters = run_pipeline(
        video_path,
        cache_path=args.cache,
    )

    out_dir = export_clusters(
        clusters,
        out_dir=args.out,
        thumb_size=args.thumb_size,
        save_individual=not args.no_individual,
        save_sheet=not args.no_sheet,
    )

    build_index_sheet(clusters, out_dir=out_dir, thumb_size=120)

    print("\nDone! Open  cluster_profiles/INDEX.jpg  for a quick overview.\n")
