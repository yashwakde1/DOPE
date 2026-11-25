import cv2
import yaml
import numpy as np
import os

# === Calibration Settings ===
CHECKERBOARD = (9, 6)               # (cols, rows) → internal corners, not squares!
square_size = 0.024                 # in meters
output_file = "config/camera_info.yaml"

# === Criteria for cornerSubPix ===
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# === Prepare object points for 3D chessboard coordinates ===
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

cap = cv2.VideoCapture(0)
print("[INFO] Press 'c' to capture chessboard. Press 'q' to calibrate and save.")

gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Couldn't read from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if found:
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)

    cv2.imshow("Calibration", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and found:
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_sub)
        objpoints.append(objp.copy())
        print(f"[INFO] Captured {len(objpoints)} frame(s)")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("[ERROR] Not enough valid captures. Need at least 5.")
    exit(1)

# === Calibrate ===
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# === Projection Matrix (P = K | 0) ===
P = np.zeros((3, 4))
P[:3, :3] = K

# === Save YAML ===
camera_data = {
    "image_width": gray.shape[1],
    "image_height": gray.shape[0],
    "camera_name": "dope_webcam_0",
    "camera_matrix": {
        "rows": 3, "cols": 3,
        "data": [round(float(x), 6) for x in K.flatten()]
    },
    "distortion_model": "plumb_bob",
    "distortion_coefficients": {
        "rows": 1, "cols": len(dist.flatten()),
        "data": [round(float(x), 6) for x in dist.flatten()]
    },
    "rectification_matrix": {
        "rows": 3, "cols": 3,
        "data": [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
    },
    "projection_matrix": {
        "rows": 3, "cols": 4,
        "data": [round(float(x), 6) for x in P.flatten()]
    }
}

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as f:
    yaml.dump(camera_data, f, default_flow_style=False)

print(f"[SUCCESS] Calibration saved to: {output_file}")
