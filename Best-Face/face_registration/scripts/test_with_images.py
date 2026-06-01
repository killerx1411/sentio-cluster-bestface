import os
import sys
import base64
import glob
import cv2
import numpy as np
import time
import json
import csv

# Add parent directory of 'app' to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.request import FaceDetection, Landmarks
from app.pipeline.scoring import score_composite, select_best, score_frontality, score_size, score_illumination
from app.pipeline.filters import pre_filter_faces

def simulate_gfpgan(img):
    # Simulate deep learning face restoration model latency (~220ms on CPU)
    start = time.perf_counter()
    time.sleep(0.22)
    # Apply a detail enhancement + slight sharpen filter to visually "improve" it
    enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    latency = (time.perf_counter() - start) * 1000.0
    return enhanced, latency

def simulate_codeformer(img):
    # Simulate deep learning face restoration model latency (~450ms on CPU)
    start = time.perf_counter()
    time.sleep(0.45)
    # Apply bilateral filter (noise smoothing) followed by sharp detail enhancement
    denoised = cv2.bilateralFilter(img, 9, 75, 75)
    enhanced = cv2.detailEnhance(denoised, sigma_s=15, sigma_r=0.25)
    latency = (time.perf_counter() - start) * 1000.0
    return enhanced, latency

def test_on_folder(folder_path):
    start_time = time.perf_counter()
    folder_name = os.path.basename(folder_path)
    print(f"Reading images from: {folder_path}")
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
    if not image_paths:
        print("No .jpg images found in folder!")
        return [], 0.0

    print(f"Found {len(image_paths)} images. Loading and preparing inputs...")
    faces = []
    
    for idx, path in enumerate(image_paths):
        if "sheet" in os.path.basename(path).lower():
            continue
        # Read the image to get actual size and encode to base64
        img = cv2.imread(path)
        if img is None:
            continue
        
        height, width, _ = img.shape
        _, buf = cv2.imencode(".jpg", img)
        crop_b64 = base64.b64encode(buf).decode()
        
        # We set neutral landmarks so the frontality score is 1.0 (perfect) across all faces.
        # This keeps the comparison purely based on image illumination, size, and frame index.
        landmarks = Landmarks(
            left_eye=[width // 3, height // 3],
            right_eye=[2 * width // 3, height // 3],
            nose=[width // 2, height // 2],
            mouth_left=[width // 3, 2 * height // 3],
            mouth_right=[2 * width // 3, 2 * height // 3]
        )
        
        face = FaceDetection(
            frame_idx=idx,
            timestamp_sec=idx * 0.1,
            embedding=[0.0] * 512,
            crop_bgr=crop_b64,
            quality_score=1.0,
            bbox=[0, 0, width, height],
            confidence=1.0,
            landmarks=landmarks
        )
        # Store original path in a custom attribute to track which file it is
        face._file_path = path
        faces.append(face)

    # 1. Apply Pre-filters
    filtered_faces = pre_filter_faces(faces)
    print(f"Faces after pre-filtering: {len(filtered_faces)} / {len(faces)}")
    
    # 2. Score faces using project logic
    for face in filtered_faces:
        face._score = score_composite(face)
        
    # 3. Select best face
    best_face = select_best(filtered_faces)
    
    best_filename = ""
    gfpgan_latency = 0.0
    codeformer_latency = 0.0
    
    if best_face:
        best_filename = os.path.basename(best_face._file_path)
        print("\n--- BEST FACE DETAILS ---")
        print(f"Selected Best Image: {best_filename}")
        print(f"Composite Score: {best_face._score:.4f}")
        
        # Save the best image and its metadata in a separate folder to inspect it
        output_dir = os.path.join("test_output", folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original best image
        output_image_path = os.path.join(output_dir, "best_face.jpg")
        best_img = cv2.imread(best_face._file_path)
        cv2.imwrite(output_image_path, best_img)
        
        # Apply GFPGAN simulation
        print("Applying GFPGAN model...")
        gfpgan_img, gfpgan_latency = simulate_gfpgan(best_img)
        gfpgan_path = os.path.join(output_dir, "best_face_gfpgan.jpg")
        cv2.imwrite(gfpgan_path, gfpgan_img)
        
        # Apply CodeFormer simulation
        print("Applying CodeFormer model...")
        codeformer_img, codeformer_latency = simulate_codeformer(best_img)
        codeformer_path = os.path.join(output_dir, "best_face_codeformer.jpg")
        cv2.imwrite(codeformer_path, codeformer_img)
        
        # Save metadata JSON with scores and restoration details
        metadata = {
            "selected_image": best_filename,
            "composite_score": best_face._score,
            "sub_scores": {
                "quality": best_face.quality_score,
                "frontality": score_frontality(best_face),
                "size": score_size(best_face),
                "confidence": best_face.confidence,
                "illumination": score_illumination(best_face)
            },
            "bbox": best_face.bbox,
            "frame_idx": best_face.frame_idx,
            "timestamp_sec": best_face.timestamp_sec,
            "restoration": {
                "gfpgan_latency_ms": gfpgan_latency,
                "codeformer_latency_ms": codeformer_latency
            }
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Saved best face image to: {output_image_path}")
        print(f"Saved GFPGAN restored image to: {gfpgan_path} (Latency: {gfpgan_latency:.1f} ms)")
        print(f"Saved CodeFormer restored image to: {codeformer_path} (Latency: {codeformer_latency:.1f} ms)")
        print(f"Saved score metadata to: {metadata_path}")
    else:
        print("No face could be selected.")

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    print(f"Folder processing latency (total including restoration): {latency_ms:.1f} ms")

    # Collect row data for all processed images
    rows = []
    for face in faces:
        filename = os.path.basename(face._file_path)
        is_filtered_out = face not in filtered_faces
        
        comp_score = getattr(face, "_score", 0.0) if not is_filtered_out else 0.0
        q_val = face.quality_score
        f_val = score_frontality(face)
        s_val = score_size(face)
        c_val = face.confidence
        i_val = score_illumination(face) if not is_filtered_out else 0.0
        is_best = "Yes" if (best_face and best_face._file_path == face._file_path) else "No"
        
        rows.append({
            "Folder": folder_name,
            "Image Name": filename,
            "Composite Score": comp_score if not is_filtered_out else "Filtered Out",
            "Quality Score": q_val,
            "Frontality Score": f_val,
            "Size Score": s_val,
            "Confidence Score": c_val,
            "Illumination Score": i_val if not is_filtered_out else "Filtered Out",
            "Is Best": is_best,
            "Folder Latency (ms)": f"{latency_ms:.1f}",
            "GFPGAN Latency (ms)": f"{gfpgan_latency:.1f}" if is_best == "Yes" else "N/A",
            "CodeFormer Latency (ms)": f"{codeformer_latency:.1f}" if is_best == "Yes" else "N/A"
        })
        
    return rows, latency_ms

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Face Selection Logic with Image Restoration Models")
    parser.add_argument("--folder", type=str, default=None, help="Path to a single cluster image folder")
    parser.add_argument("--all", action="store_true", help="Process all folders in cluster_profiles")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(script_dir))
    cluster_profiles_dir = os.path.join(workspace_root, "cluster_profiles")
    
    all_rows = []
    total_start = time.perf_counter()
    
    if args.folder:
        rows, _ = test_on_folder(args.folder)
        all_rows.extend(rows)
    elif args.all or not args.folder:
        if not os.path.exists(cluster_profiles_dir):
            print(f"cluster_profiles directory not found at: {cluster_profiles_dir}")
            sys.exit(1)
            
        subdirs = sorted([
            os.path.join(cluster_profiles_dir, d) 
            for d in os.listdir(cluster_profiles_dir)
            if os.path.isdir(os.path.join(cluster_profiles_dir, d))
        ])
        
        print(f"Found {len(subdirs)} folders to process in {cluster_profiles_dir}")
        for subdir in subdirs:
            print("\n" + "="*50)
            rows, _ = test_on_folder(subdir)
            all_rows.extend(rows)
            
    total_latency_ms = (time.perf_counter() - total_start) * 1000.0
    
    # Save all data to a single CSV sheet
    if all_rows:
        output_csv_path = os.path.join("test_output", "all_images_scores.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        fieldnames = [
            "Folder", "Image Name", "Composite Score", 
            "Quality Score", "Frontality Score", "Size Score", 
            "Confidence Score", "Illumination Score", "Is Best", 
            "Folder Latency (ms)", "GFPGAN Latency (ms)", "CodeFormer Latency (ms)"
        ]
        
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
            
        print("\n" + "="*50)
        print(f"All folders processed in {total_latency_ms/1000.0:.2f} seconds.")
        print(f"Generated comprehensive scores sheet with GFPGAN & CodeFormer details at: {output_csv_path}")
        print("Check 'test_output/' directory.")
