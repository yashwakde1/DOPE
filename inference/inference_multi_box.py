
#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import os
import sys
import yaml
from datetime import datetime

sys.path.append("../common/")
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
from utils import loadweights

def quaternion_to_rotation_matrix(x, y, z, w):
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R

def get_bounding_box_points(center, R, dims):
    l, w, h = dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0
    corners = np.array([
        [ l,  w,  h], [-l,  w,  h], [-l, -w,  h], [ l, -w,  h],
        [ l,  w, -h], [-l,  w, -h], [-l, -w, -h], [ l, -w, -h]
    ]).T
    rotated = R @ corners
    translated = rotated + center.reshape(3, 1)
    return translated.T

def draw_projected_box3d(img, pts2d):
    pts2d = pts2d.astype(int)
    for i, j in zip([0,1,2,3,0], [1,2,3,0,1]):
        cv2.line(img, tuple(pts2d[i]), tuple(pts2d[j]), (0,255,255), 2)
    for i, j in zip([4,5,6,7,4], [5,6,7,4,5]):
        cv2.line(img, tuple(pts2d[i]), tuple(pts2d[j]), (255,0,255), 2)
    for i, j in zip(range(4), range(4,8)):
        cv2.line(img, tuple(pts2d[i]), tuple(pts2d[j]), (255,255,0), 2)
    return img

class DopeNode:
    def __init__(self, config, weight_path, parallel, class_name):
        self.config = config
        self.class_name = class_name
        self.input_is_rectified = config["input_is_rectified"]
        self.downscale_height = config["downscale_height"]

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config["thresh_angle"]
        self.config_detect.thresh_map = config["thresh_map"]
        self.config_detect.sigma = config["sigma"]
        self.config_detect.thresh_points = config["thresh_points"]

        self.model = ModelData(name=class_name, net_path=weight_path, parallel=parallel)
        self.model.load_net_model()
        self.dimension = tuple(config["dimensions"][class_name])
        self.pnp_solver = CuboidPNPSolver(class_name, cuboid3d=Cuboid3d(self.dimension))

    def process(self, img, camera_matrix, dist_coeffs):
        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(dist_coeffs)
        results, _ = ObjectDetector.detect_object_in_image(
            self.model.net, self.pnp_solver, img, self.config_detect
        )
        for r in results:
            loc = r.get("location")
            quat = r.get("quaternion_xyzw") or r.get("quaternion")
            if loc is None or quat is None:
                continue
            t = np.array(loc, dtype=np.float64).reshape(3, 1)
            xq, yq, zq, wq = quat
            R = quaternion_to_rotation_matrix(xq, yq, zq, wq)

            # Draw axes
            axis_length = max(self.dimension) * 0.5
            x_end = t + axis_length * R[:, 0].reshape(3, 1)
            y_end = t + axis_length * R[:, 2].reshape(3, 1)
            z_end = t - axis_length * R[:, 1].reshape(3, 1)
            pts3d_axes = np.vstack([t.T, x_end.T, y_end.T, z_end.T]).reshape(-1, 3)
            pts2d_axes, _ = cv2.projectPoints(pts3d_axes, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
            pts2d_axes = pts2d_axes.reshape(-1, 2).astype(int)
            origin = tuple(pts2d_axes[0])
            cv2.line(img, origin, tuple(pts2d_axes[1]), (0, 0, 255), 2)
            cv2.line(img, origin, tuple(pts2d_axes[2]), (0, 255, 0), 2)
            cv2.line(img, origin, tuple(pts2d_axes[3]), (255, 0, 0), 2)

            # Draw bounding box
            bbox_corners = get_bounding_box_points(t.flatten(), R, self.dimension)
            pts2d_box, _ = cv2.projectPoints(bbox_corners, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
            draw_projected_box3d(img, pts2d_box.reshape(-1, 2))
        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="../config/camera_info.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parallel", action='store_true')
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)

    if opt.weights.endswith(".pth"):
        weights_dict = {list(config["weights"].keys())[0]: opt.weights}
    else:
        weights_dict = config["weights"]

    nodes = []
    for obj_name, weight_path in weights_dict.items():
        weight_path = weight_path.replace("package://dope/weights", opt.weights) if "package://" in weight_path else weight_path
        node = DopeNode(config, weight_path, opt.parallel, obj_name)
        nodes.append(node)

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        exit()

    P = np.matrix(camera_info["projection_matrix"]["data"], dtype=np.float64)
    P.resize((3, 4))
    camera_matrix = P[:, :3]
    dist_coeffs = np.zeros((4, 1))

    save_dir = "/home/pablo/Deep_Object_Pose/output_webcam"
    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for node in nodes:
            frame = node.process(frame, camera_matrix, dist_coeffs)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("DOPE Multi-Object Pose", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('s'), ord('S')]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/frame_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")

    cap.release()
    cv2.destroyAllWindows()