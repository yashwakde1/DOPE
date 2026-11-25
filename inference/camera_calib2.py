#!/usr/bin/env python3
import glob
import os

import cv2
import numpy as np
import yaml

# === Hard‑coded parameters ===
INPUT_DIR    = "/home/pablo/Deep_Object_Pose/image_set"
CHECKERBOARD = (9, 8)      # internal corners: 9 cols × 8 rows
SQUARE_SIZE  = 0.024       # meters
OUTPUT_FILE  = "config/camera_info.yaml"

# === prepare object points in 3D (0,0,0), (1,0,0), … scaled by square_size ===
objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points
imgpoints = []  # 2D image points

# gather all jpg/jpeg/png in INPUT_DIR
image_paths = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
    glob.glob(os.path.join(INPUT_DIR, "*.jpeg")) +
    glob.glob(os.path.join(INPUT_DIR, "*.png"))
)
if not image_paths:
    print(f"[ERROR] No images found in {INPUT_DIR}")
    exit(1)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for img_path in image_paths:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not found:
        print(f"[WARN] Chessboard not found in {os.path.basename(img_path)}")
        continue

    corners_sub = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    objpoints.append(objp.copy())
    imgpoints.append(corners_sub)
    print(f"[INFO] Captured corners from {os.path.basename(img_path)}")

if len(objpoints) < 5:
    print(f"[ERROR] Only {len(objpoints)} valid images, need ≥5.")
    exit(1)

# calibrate
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# build projection matrix P = [K|0]
P = np.zeros((3,4))
P[:3,:3] = K

# pack into ROS‑style YAML dict
camera_data = {
    "image_width":  int(gray.shape[1]),
    "image_height": int(gray.shape[0]),
    "camera_name":  "dope_webcam_0",
    "camera_matrix": {
        "rows": 3, "cols": 3,
        "data": [float(f"{x:.6f}") for x in K.flatten()]
    },
    "distortion_model": "plumb_bob",
    "distortion_coefficients": {
        "rows": 1, "cols": len(dist.flatten()),
        "data": [float(f"{x:.6f}") for x in dist.flatten()]
    },
    "rectification_matrix": {
        "rows": 3, "cols": 3,
        "data": [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]
    },
    "projection_matrix": {
        "rows": 3, "cols": 4,
        "data": [float(f"{x:.6f}") for x in P.flatten()]
    }
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    yaml.dump(camera_data, f, default_flow_style=False)

print(f"[SUCCESS] Calibration saved to {OUTPUT_FILE}")
