import os

import torch
import json
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.parameter import Parameter

try:
    from .worldgt import WorldGT
    from .line_fit import lane_fit, final_viz, perspective_transform, closest_point_on_polynomial
    from .model_utils import load_model, inference
except Exception:
    from worldgt import WorldGT
    from line_fit import lane_fit, final_viz, perspective_transform, closest_point_on_polynomial
    from model_utils import load_model, inference
import rich
import cv2
from scipy.spatial.transform import Rotation as R


class LaneVisualizer(Node):
    def __init__(self):
        super().__init__("lane_visualizer")

        sim_time_param = Parameter('use_sim_time', Parameter.Type.BOOL, True)
        self.set_parameters([sim_time_param])

        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._model = load_model()
            if self._model is not None:
                self._model = self._model.to(self._dev)
                self._model = self._model.eval()
                rich.print("[green]loaded SimpleEnet :o")
            else:
                self.get_logger().error(f"could not load SimpleEnet model x_X: {e}")
                exit(1)
        except Exception as e:
            self.get_logger().error(f"could not load SimpleEnet model x_X: {e}")
            exit(1)
        
        try: 
            with open(os.path.join("data", "bev_config.json")) as f:
                self._bev_cfg = json.load(f)
        except FileNotFoundError:
            self.get_logger().error(f"could not load bev config x_X: {e}")
            exit(1)

        self._world = WorldGT("Silverstone")
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)
        
        self._image_msg = None
        self._cv_bridge = CvBridge()
        
        self.create_subscription(
            Image,
            "/camera/image_raw",
            self._on_image,
            10
        )
    
    def _on_image(self, msg) -> None:
        self._image_msg = msg
        if self._model is None:
            return
        
        image = self._cv_bridge.imgmsg_to_cv2(self._image_msg, "bgr8")
        mask = inference(self._model, image, self._dev)
        m = mask.astype(np.uint8) * 255
        combine_fit_img, binary_BEV, ret = self.fit_poly_lanes(image, m)

        binary_BEV = np.pad(binary_BEV, ((0, 100), (0, 0)))
        binary_BEV = cv2.cvtColor(binary_BEV, cv2.COLOR_GRAY2BGR)
        
        if ret:                
            poly_px = (np.add(ret["left_fit"], ret["right_fit"]) / 2)
            XTE, HE, camera_px, closest_px = self.compute_error(poly_px)
            
            # draw lane lines
            ploty = ret['ploty']
            left_fitx = np.polyval(ret["left_fit"], ploty)
            center_fitx = np.polyval(poly_px, ploty)
            right_fitx = np.polyval(ret["right_fit"], ploty)
            
            pts_left = np.stack((left_fitx, ploty), axis=1).astype(np.int32)
            pts_center = np.stack((center_fitx, ploty), axis=1).astype(np.int32)
            pts_right = np.stack((right_fitx, ploty), axis=1).astype(np.int32)

            cv2.polylines(binary_BEV, [pts_center], isClosed=False, color=(0, 255, 255), thickness=4)                
            cv2.polylines(binary_BEV, [pts_left], isClosed=False, color=(255, 0, 0), thickness=4)
            cv2.polylines(binary_BEV, [pts_right], isClosed=False, color=(0, 0, 255), thickness=4)

            # draw closest point and bridge line
            cv2.circle(binary_BEV, (int(closest_px[0]), int(closest_px[1])), 8, (0, 255, 0), -1)
            cv2.line(
                binary_BEV,
                (int(camera_px[0]), int(camera_px[1])),
                (int(closest_px[0]), int(closest_px[1])),
                (0, 255, 0),
                4
            )

            # draw camera chevron
            cv2.line(
                binary_BEV,
                (int(camera_px[0]), int(camera_px[1])),
                (int(camera_px[0] - 20), int(camera_px[1] + 20)),
                (255, 0, 255),
                4
            )
            cv2.line(
                binary_BEV,
                (int(camera_px[0]), int(camera_px[1])),
                (int(camera_px[0] + 20), int(camera_px[1] + 20)),
                (255, 0, 255),
                4
            )

            XTE = f"{XTE:.2f}"
            HE = f"{np.degrees(HE):.2f}"
        else:
            XTE = "N/A"
            HE = "N/A"

        try:
            trans = self._tf_buf.lookup_transform("silverstone", "stereo_camera_link", msg.header.stamp)
            pos = trans.transform.translation
            q = trans.transform.rotation
            rotation = R.from_quat([q.x, q.y, q.z, q.w])
            euler_angles = rotation.as_euler('xyz', degrees=False)
            yaw = euler_angles[2]
            lane, _, gt_XTE, gt_HE = self._world.get_metrics(pos.x, pos.y, yaw)
            gt_XTE = f"{gt_XTE:.2f}"
            gt_HE = f"{np.degrees(gt_HE):.2f}"
        except:
            lane = "unknown"
            gt_XTE = "N/A"
            gt_HE = "N/A"
            
        print(f"EST XTE: {XTE} m - HE: {HE}° -- GT XTE: {gt_XTE} m HE: {gt_HE}° - lane: {lane}")

        if combine_fit_img is None:
            combine_fit_img = image
            
        cv2.imshow("render_view", combine_fit_img)
        cv2.imshow("binary_BEV", binary_BEV)
        cv2.waitKey(1)
    
    def compute_error(self, poly_px):
        """
        Calculates Cross-Track Error (XTE) and Heading Error.

        poly_px:    polynomial coefficients defined in pixels
                    ex for 2nd order: (A, B, C) where x = Ay^2 + By + C
        """
        bev_height_m, bev_width_m = self._bev_cfg["bev_world_dim"]
        Sy, Sx = self._bev_cfg["unit_conversion_factor"]
        scale = np.array([Sx, Sy])

        camera_m = np.array([(bev_width_m / 2), bev_height_m])
        camera_px = camera_m / scale
        closest_px = closest_point_on_polynomial(camera_px, poly_px)
        closest_m = closest_px * scale

        ##### YOUR CODE STARTS HERE #####

        # calculate cross track error
        # hint: |XTE| = distance between camera and closest point
        #       on ploly_px however XTE is not a strictly positive value
        delta_m = camera_m - closest_m
        XTE = float(np.linalg.norm(delta_m))
        if closest_m[0] < camera_m[0]:
            XTE = -XTE

        # hint: find derivative of the polynomial at the closest point
        #       then use arctan on the scaled slope
        dpoly_dy_px = float(np.polyder(poly_px)(closest_px[1]))
        dpoly_dx_m = dpoly_dy_px * (Sx / Sy)
        HE = float(np.arctan(dpoly_dx_m))

        ##### YOUR CODE ENDS HERE #####

        return XTE, HE, camera_px, closest_px

    def fit_poly_lanes(self, raw_img, binary_img):
        binary_warped, M, Minv = perspective_transform(binary_img, np.float32(self._bev_cfg["src"]))
        ret = lane_fit(binary_warped)
        if ret is None:
            self.get_logger().debug("ret is None; returning None for both.")
            return None, binary_warped, None
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        
        combine_fit_img = None
        if ret is not None:
            self.get_logger().debug("Model detected lanes")
            combine_fit_img = final_viz(raw_img, left_fit, right_fit, Minv)
        else:
            self.get_logger().debug("Model unable to detect lanes")
        return combine_fit_img, binary_warped, ret


def main(args=None):
    rclpy.init(args=args)
    node = LaneVisualizer()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
