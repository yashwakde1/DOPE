#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import os
import sys
import yaml
from PIL import Image, ImageDraw
import simplejson as json

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

class DopeNode(object):
    def __init__(self, config, weight, parallel, class_name):
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
        self.model = ModelData(name=class_name, net_path=weight, parallel=parallel)
        self.model.load_net_model()
        self.dimension = tuple(config["dimensions"][class_name])
        self.pnp_solver = CuboidPNPSolver(class_name, cuboid3d=Cuboid3d(config["dimensions"][class_name]))
        self.class_name = class_name

    def process_frame(self, img, camera_info):
        if self.input_is_rectified:
            P = np.matrix(camera_info["projection_matrix"]["data"], dtype=np.float64)
            P.resize((3, 4))
            camera_matrix = P[:, :3]
            dist_coeffs = np.zeros((4, 1))
        else:
            camera_matrix = np.matrix(camera_info["K"], dtype=np.float64)
            camera_matrix.resize((3, 3))
            dist_coeffs = np.matrix(camera_info["D"], dtype=np.float64)
            dist_coeffs.resize((len(camera_info["D"]), 1))

        h, w, _ = img.shape
        scale = float(self.downscale_height) / h
        if scale < 1.0:
            camera_matrix[:2] *= scale
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

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
            axis_length = max(self.dimension) * 0.5
            x_end = t + axis_length * R[:, 0].reshape(3, 1)
            y_end = t + axis_length * R[:, 2].reshape(3, 1)
            z_end = t - axis_length * R[:, 1].reshape(3, 1)
            pts3d = np.vstack([t.T, x_end.T, y_end.T, z_end.T]).reshape(-1, 3)
            pts2d, _ = cv2.projectPoints(
                pts3d,
                rvec=np.zeros((3, 1), dtype=np.float64),
                tvec=np.zeros((3, 1), dtype=np.float64),
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs
            )
            pts2d = pts2d.reshape(-1, 2).astype(int)
            origin = tuple(pts2d[0])
            x_p, y_p, z_p = tuple(pts2d[1]), tuple(pts2d[2]), tuple(pts2d[3])
            cv2.line(img, origin, x_p, (0, 0, 255), 2)
            cv2.line(img, origin, y_p, (0, 255, 0), 2)
            cv2.line(img, origin, z_p, (255, 0, 0), 2)
        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="../config/camera_info.yaml", help="Path to camera info file")
    parser.add_argument("--weights", required=True, help="Path to weights.")
    parser.add_argument("--object", required=True, help="Name of object to detect")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--parallel", action='store_true', help="Parallel weights flag")
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)

    dope_node = DopeNode(config, opt.weights, opt.parallel, opt.object)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = dope_node.process_frame(frame, camera_info)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("DOPE Live Pose", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
