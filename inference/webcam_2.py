import argparse
import cv2
import numpy as np
from PIL import Image
import os
import simplejson as json
import yaml
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
from utils import Draw

def draw_bounding_box(image, projected_points, color=(0, 255, 0), thickness=2):
    points = np.array(projected_points, dtype=np.int32)
    if points.shape != (8, 2):
        return image
    x, y, w, h = cv2.boundingRect(points)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image

class DopeNode:
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
        print("Model Loaded")

        self.draw_color = tuple(config.get("draw_colors", {}).get(class_name, (0, 255, 0)))
        self.dimension = tuple(config["dimensions"][class_name])
        self.class_id = config["class_ids"][class_name]

        self.pnp_solver = CuboidPNPSolver(class_name, cuboid3d=Cuboid3d(self.dimension))
        self.class_name = class_name

    def image_callback(self, img, camera_info, img_name, output_folder, weight, save=False):
        P = np.matrix(camera_info["projection_matrix"]["data"], dtype="float64").copy()
        P.resize((3, 4))
        camera_matrix = P[:, :3]
        dist_coeffs = np.zeros((4, 1))

        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)))

        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(dist_coeffs)

        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)

        dict_out = {"camera_data": {}, "objects": []}

        results, belief_imgs = ObjectDetector.detect_object_in_image(
            self.model.net, self.pnp_solver, img, self.config_detect, grid_belief_debug=False
        )

        for result in results:
            if result["location"] is None:
                continue

            loc = result["location"]
            ori = result["quaternion"]

            dict_out["objects"].append({
                "class": self.class_name,
                "location": np.array(loc).tolist(),
                "quaternion_xyzw": np.array(ori).tolist(),
                "projected_cuboid": np.array(result["projected_points"]).tolist(),
            })

            if None not in result["projected_points"]:
                points2d = [tuple(pair) for pair in result["projected_points"]]
                draw.draw_cube(points2d, self.draw_color)
                img = draw_bounding_box(img, points2d, color=self.draw_color)

        if save:
            output_path = os.path.join(output_folder, weight.split("/")[-1].replace(".pth", ""))
            os.makedirs(output_path, exist_ok=True)

            im.save(os.path.join(output_path, img_name + ".png"))
            with open(os.path.join(output_path, img_name + ".json"), "w") as fp:
                json.dump(dict_out, fp, indent=2)

        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--config", default="config/config_pose.yaml")
    parser.add_argument("--camera", default="config/camera_info.yaml")
    parser.add_argument("--outf", default="output_webcam")
    parser.add_argument("--parallel", action='store_true')
    parser.add_argument("--save", action='store_true')
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(opt.outf, exist_ok=True)
    dope_node = DopeNode(config, opt.weights, opt.parallel, opt.object)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    frame_id = 0
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_name = f"frame_{frame_id:04d}"
        processed_frame = dope_node.image_callback(frame_rgb, camera_info, img_name, opt.outf, opt.weights, save=opt.save)

        frame_bgr = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)
        cv2.imshow("DOPE Webcam", frame_bgr)

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
