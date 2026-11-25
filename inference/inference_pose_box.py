
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
from utils import loadimages_inference, loadweights

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

def generate_3d_bounding_box(center, dimensions, rotation_matrix):
    x_len, y_len, z_len = [d / 2.0 for d in dimensions]
    corners = np.array([
        [ x_len,  y_len,  z_len],
        [-x_len,  y_len,  z_len],
        [-x_len, -y_len,  z_len],
        [ x_len, -y_len,  z_len],
        [ x_len,  y_len, -z_len],
        [-x_len,  y_len, -z_len],
        [-x_len, -y_len, -z_len],
        [ x_len, -y_len, -z_len]
    ])
    world_corners = (rotation_matrix @ corners.T).T + center.reshape(1, 3)
    return world_corners

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

    def image_callback(self, img, camera_info, img_name, output_folder, weight, debug=False):
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

        results, belief_imgs = ObjectDetector.detect_object_in_image(
            self.model.net, self.pnp_solver, img, self.config_detect, grid_belief_debug=debug
        )

        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)

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

            # Draw axes
            pts3d_axes = np.vstack([t.T, x_end.T, y_end.T, z_end.T]).reshape(-1, 3)
            pts2d_axes, _ = cv2.projectPoints(pts3d_axes, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
            pts2d_axes = pts2d_axes.reshape(-1, 2).astype(int)
            origin = tuple(pts2d_axes[0])
            x_p, y_p, z_p = tuple(pts2d_axes[1]), tuple(pts2d_axes[2]), tuple(pts2d_axes[3])
            draw.line([origin, x_p], fill=(255, 0, 0), width=3)
            draw.line([origin, y_p], fill=(0, 255, 0), width=3)
            draw.line([origin, z_p], fill=(0, 0, 255), width=3)

            # Draw bounding box
            box_corners_3d = generate_3d_bounding_box(t.flatten(), self.dimension, R)
            box_corners_2d, _ = cv2.projectPoints(box_corners_3d, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
            box_corners_2d = box_corners_2d.reshape(-1, 2).astype(int)
            box_lines = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for i, j in box_lines:
                draw.line([tuple(box_corners_2d[i]), tuple(box_corners_2d[j])], fill=(255, 255, 0), width=2)

        img_rel = img_name.lstrip(os.sep)
        img_base = os.path.basename(img_rel)
        subdir = os.path.dirname(img_rel)
        weight_name = os.path.splitext(os.path.basename(weight))[0]
        save_dir = os.path.join(output_folder, weight_name, subdir)
        os.makedirs(save_dir, exist_ok=True)
        im.save(os.path.join(save_dir, img_base))
        if belief_imgs is not None:
            belief_imgs.save(os.path.join(save_dir, "belief_maps.png"))
        print(f"→ wrote image + json to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outf", default="output", help="Where to store the output images and inference results.")
    parser.add_argument("--data", required=True, help="folder for data images to load.")
    parser.add_argument("--config", default="../config/config_pose.yaml", help="Path to inference config file")
    parser.add_argument("--camera", default="../config/camera_info.yaml", help="Path to camera info file")
    parser.add_argument("--weights", "--weight", "-w", required=True, help="Path to weights or folder containing weights.")
    parser.add_argument("--parallel", action='store_true', help="Set if weights were trained using DDP")
    parser.add_argument("--exts", nargs="+", type=str, default=["png", "jpg", "jpeg"], help="Image file extensions to use")
    parser.add_argument("--object", required=True, help="Name of class to run detections on.")
    parser.add_argument('--debug', action='store_true', help="Save debug belief maps and visuals")
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(opt.outf, exist_ok=True)
    weights = loadweights(opt.weights)
    imgs, imgsname = loadimages_inference(opt.data, extensions=opt.exts)

    for w_i, weight in enumerate(weights):
        dope_node = DopeNode(config, weight, opt.parallel, opt.object)
        for i, img_path in enumerate(imgs):
            print(f"({w_i+1}/{len(weights)}) Frame {i+1}/{len(imgs)}: {imgsname[i]}")
            frame = cv2.imread(img_path)[..., ::-1].copy()
            dope_node.image_callback(
                img=frame,
                camera_info=camera_info,
                img_name=imgsname[i],
                output_folder=opt.outf,
                weight=weight,
                debug=opt.debug
            )
        print("------")