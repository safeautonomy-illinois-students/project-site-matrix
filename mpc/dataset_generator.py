#!/usr/bin/env python3
"""
Generate fine-tuning dataset pairs (Xi, Yi) from live runs.

Xi: natural-language prompt describing the exact environment scene.
Yi: MPC next waypoint label for that same scene timestamp.
"""

import json
import os
from collections import deque

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from llm_drone.verifier.cli_goal import parse_goal_overrides

try:
    from px4_msgs.msg import VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


class LLMFineTuneDatasetGenerator(Node):
    """Collect synchronized scene->MPC label examples as JSONL."""

    def __init__(self, goal_override=None):
        super().__init__('llm_finetune_dataset_generator')

        self.declare_parameter('goal_x', 35.0)
        self.declare_parameter('goal_y', 3.0)
        self.declare_parameter('goal_z', 2.5)
        self.declare_parameter('goal_frame', 'ned')  # gazebo/map means ENU
        self.declare_parameter('update_rate', 2.0)
        self.declare_parameter('obstacle_threshold', 2.5)
        self.declare_parameter('obstacle_sample_step', 20)
        self.declare_parameter('max_scene_time_diff_s', 0.2)
        self.declare_parameter('max_label_time_diff_s', 0.2)
        self.declare_parameter('output_path', '/tmp/llm_finetune_dataset.jsonl')
        self.declare_parameter('max_samples', 0)  # 0 => unlimited
        self.declare_parameter('variations_per_scene', 4)

        goal_input = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value,
        ])
        goal_frame = str(self.get_parameter('goal_frame').value).strip().lower()
        self.goal = self.convert_goal_to_ned(goal_input, goal_frame)
        if goal_override is not None:
            self.goal = np.array(goal_override, dtype=float)

        self.obs_threshold = float(self.get_parameter('obstacle_threshold').value)
        self.obstacle_sample_step = int(self.get_parameter('obstacle_sample_step').value)
        self.max_scene_dt = float(self.get_parameter('max_scene_time_diff_s').value)
        self.max_label_dt = float(self.get_parameter('max_label_time_diff_s').value)
        self.output_path = str(self.get_parameter('output_path').value)
        self.max_samples = int(self.get_parameter('max_samples').value)
        self.variations_per_scene = int(self.get_parameter('variations_per_scene').value)

        prompt_file = self._resolve_prompt_file()
        try:
            with open(prompt_file, 'r') as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            self.system_prompt = "You are a drone motion planning system. Generate safe trajectories."
            self.get_logger().warn(f'Prompt file not found: {prompt_file}')

        self.bridge = CvBridge()
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.rotation_ned_body = np.eye(3)
        self.depth_image = None
        self.obstacles_ned = []
        self.depth_stamp_s = None
        self.odom_stamp_s = None
        self.scene_counter = 0
        self.sample_counter = 0

        # Small trajectory history for nearest-time label lookup.
        self.mpc_buffer = deque(maxlen=2000)  # (stamp_s, np.array([x,y,z]))

        # Match MPC camera projection.
        self.camera_K = np.array([
            [432.496042035043, 0.0, 319.5],
            [0.0, 432.496042035043, 239.5],
            [0, 0, 1]
        ], dtype=float)
        self.rotation_body_camera = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)
        self.translation_body_camera_m = np.array([0.13233, 0.0, 0.26078], dtype=float)

        if HAS_PX4_MSGS:
            self.odom_sub = self.create_subscription(
                VehicleOdometry,
                '/fmu/out/vehicle_odometry',
                self.vehicle_odometry_callback,
                qos_profile_sensor_data,
            )
        else:
            self.odom_sub = self.create_subscription(
                Odometry,
                '/fmu/out/vehicle_odometry',
                self.odometry_callback,
                qos_profile_sensor_data,
            )
        self.depth_sub = self.create_subscription(
            Image,
            '/depth_camera',
            self.depth_image_callback,
            qos_profile_sensor_data,
        )
        self.mpc_sub = self.create_subscription(
            PoseStamped,
            '/mpc/trajectory',
            self.mpc_trajectory_callback,
            10,
        )

        period = 1.0 / float(self.get_parameter('update_rate').value)
        self.timer = self.create_timer(period, self.sample_once)

        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        self.get_logger().info(f'Dataset generator started. output={self.output_path}')
        self.get_logger().info(f'Goal (NED): {self.goal}')

    def _resolve_prompt_file(self):
        local_path = os.path.join(os.path.dirname(__file__), '../config/llm_prompt.txt')
        local_path = os.path.abspath(local_path)
        if os.path.exists(local_path):
            return local_path
        share_dir = get_package_share_directory('llm_drone')
        return os.path.join(share_dir, 'config', 'llm_prompt.txt')

    @staticmethod
    def convert_goal_to_ned(goal, goal_frame):
        g = np.array(goal, dtype=float)
        if goal_frame in ('gazebo', 'gazebo_enu', 'enu', 'map'):
            return np.array([g[1], g[0], -g[2]], dtype=float)
        return g

    @staticmethod
    def quaternion_to_rotation_matrix(w, x, y, z):
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-9:
            return np.eye(3)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=float)

    @staticmethod
    def stamp_to_sec(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def depth_image_callback(self, msg):
        try:
            if msg.encoding in ('16UC1', 'mono16'):
                depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self.depth_image = depth_mm.astype(np.float32) / 1000.0
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.depth_stamp_s = self.stamp_to_sec(msg.header.stamp)
            self.obstacles_ned = self.detect_obstacles_in_ned()
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    def odometry_callback(self, msg):
        self.current_position = np.array([
            msg.pose.pose.position.y,
            msg.pose.pose.position.x,
            -msg.pose.pose.position.z,
        ])
        self.current_velocity = np.array([
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.x,
            -msg.twist.twist.linear.z,
        ])
        q = msg.pose.pose.orientation
        rotation_enu_body = self.quaternion_to_rotation_matrix(q.w, q.x, q.y, q.z)
        rotation_enu_to_ned = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=float,
        )
        self.rotation_ned_body = rotation_enu_to_ned @ rotation_enu_body
        self.odom_stamp_s = self.stamp_to_sec(msg.header.stamp)

    def vehicle_odometry_callback(self, msg):
        self.current_position = np.array([
            float(msg.position[0]),
            float(msg.position[1]),
            float(msg.position[2]),
        ])
        self.current_velocity = np.array([
            float(msg.velocity[0]),
            float(msg.velocity[1]),
            float(msg.velocity[2]),
        ])
        self.rotation_ned_body = self.quaternion_to_rotation_matrix(
            float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])
        )
        stamp_us = float(msg.timestamp_sample if hasattr(msg, 'timestamp_sample') else msg.timestamp)
        self.odom_stamp_s = stamp_us * 1e-6

    def mpc_trajectory_callback(self, msg):
        t = self.stamp_to_sec(msg.header.stamp)
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self.mpc_buffer.append((t, p))

    def detect_obstacles_in_ned(self):
        if self.depth_image is None:
            return []
        h, w = self.depth_image.shape
        step = max(1, self.obstacle_sample_step)
        points = []
        for v in range(0, h, step):
            for u in range(0, w, step):
                depth = self.depth_image[v, u]
                if depth < 0.5 or depth > self.obs_threshold or not np.isfinite(depth):
                    continue
                x_cam = (u - self.camera_K[0, 2]) * depth / self.camera_K[0, 0]
                y_cam = (v - self.camera_K[1, 2]) * depth / self.camera_K[1, 1]
                z_cam = depth
                point_cam = np.array([x_cam, y_cam, z_cam], dtype=float)
                point_body = self.rotation_body_camera @ point_cam + self.translation_body_camera_m
                point_ned = self.current_position + self.rotation_ned_body @ point_body
                points.append(point_ned)
        return points

    def build_environment_vector(self):
        if self.depth_image is None:
            return None
        depth = self.depth_image
        valid = depth[np.isfinite(depth)]
        valid = valid[(valid > 0.2) & (valid < 20.0)]

        h, w = depth.shape
        bands = {
            'far_left': (0, int(0.2 * w)),
            'left': (int(0.2 * w), int(0.4 * w)),
            'center': (int(0.4 * w), int(0.6 * w)),
            'right': (int(0.6 * w), int(0.8 * w)),
            'far_right': (int(0.8 * w), w),
        }
        sector_min = {}
        for name, (s, e) in bands.items():
            patch = depth[:, s:e]
            pv = patch[np.isfinite(patch)]
            pv = pv[(pv > 0.2) & (pv < 20.0)]
            sector_min[name] = float(np.min(pv)) if pv.size > 0 else 20.0

        goal_delta = self.goal - self.current_position
        dist_to_goal = float(np.linalg.norm(goal_delta))
        speed = float(np.linalg.norm(self.current_velocity))

        if self.obstacles_ned:
            obs = np.array(self.obstacles_ned, dtype=float)
            rel = obs - self.current_position
            d = np.linalg.norm(rel, axis=1)
            i = int(np.argmin(d))
            nearest_d = float(d[i])
            nearest_pos = [float(x) for x in obs[i]]
            nearest_rel = [float(x) for x in rel[i]]
        else:
            nearest_d = 20.0
            nearest_pos = [float(self.current_position[0]), float(self.current_position[1]), float(self.current_position[2])]
            nearest_rel = [20.0, 0.0, 0.0]

        return {
            'current_position_ned_m': [float(x) for x in self.current_position],
            'current_velocity_mps': [float(v) for v in self.current_velocity],
            'goal_position_ned_m': [float(g) for g in self.goal],
            'goal_delta_m': [float(x) for x in goal_delta],
            'distance_to_goal_m': dist_to_goal,
            'speed_mps': speed,
            'obstacle_features_ned': {
                'sample_count': int(len(self.obstacles_ned)),
                'nearest_obstacle_distance_ned_m': nearest_d,
                'nearest_obstacle_position_ned_m': nearest_pos,
                'nearest_obstacle_relative_ned_m': nearest_rel,
            },
            'depth_features': {
                'valid_fraction': float(valid.size / depth.size) if depth.size > 0 else 0.0,
                'global_min_m': float(np.min(valid)) if valid.size > 0 else 20.0,
                'global_mean_m': float(np.mean(valid)) if valid.size > 0 else 20.0,
                'sector_min_m': sector_min,
            },
        }

    def translate_vector_to_nlp(self, vector):
        p = vector['current_position_ned_m']
        v = vector['current_velocity_mps']
        g = vector['goal_position_ned_m']
        d = vector['goal_delta_m']
        obs = vector['obstacle_features_ned']
        mins = vector['depth_features']['sector_min_m']
        return (
            f"Current NED position=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) m. "
            f"Velocity=({v[0]:.2f},{v[1]:.2f},{v[2]:.2f}) m/s. "
            f"Goal NED=({g[0]:.2f},{g[1]:.2f},{g[2]:.2f}) m. "
            f"Goal delta=({d[0]:.2f},{d[1]:.2f},{d[2]:.2f}) m, dist={vector['distance_to_goal_m']:.2f} m. "
            f"Nearest obstacle NED=({obs['nearest_obstacle_position_ned_m'][0]:.2f},"
            f"{obs['nearest_obstacle_position_ned_m'][1]:.2f},{obs['nearest_obstacle_position_ned_m'][2]:.2f}) m, "
            f"relative=({obs['nearest_obstacle_relative_ned_m'][0]:.2f},"
            f"{obs['nearest_obstacle_relative_ned_m'][1]:.2f},{obs['nearest_obstacle_relative_ned_m'][2]:.2f}) m, "
            f"distance={obs['nearest_obstacle_distance_ned_m']:.2f} m. "
            f"Sector mins [far_left,left,center,right,far_right]="
            f"[{mins['far_left']:.2f},{mins['left']:.2f},{mins['center']:.2f},{mins['right']:.2f},{mins['far_right']:.2f}] m."
        )

    def build_prompt_variations(self, env_vector, env_text):
        base = (
            "Use the fixed planning policy exactly as defined in the system prompt.\n\n"
            f"Dynamic Environment Section:\n{env_text}\n\n"
            f"Numerical Environment Vector v:\n{json.dumps(env_vector, indent=2)}\n\n"
            "Return only one JSON object with keys: waypoints, selected_waypoint_index, reasoning."
        )
        concise = (
            f"Scene summary: {env_text}\n"
            "Output the next best waypoint(s) in NED as strict JSON."
        )
        obstacle_focus = (
            f"Obstacle-centric planning scene: {env_text}\n"
            "Prioritize collision avoidance with smooth progress to goal; output strict JSON waypoint selection."
        )
        goal_focus = (
            f"Goal-progress scene in NED: goal={env_vector['goal_position_ned_m']}, "
            f"delta={env_vector['goal_delta_m']}, nearest_obstacle_dist="
            f"{env_vector['obstacle_features_ned']['nearest_obstacle_distance_ned_m']:.2f}m.\n"
            "Return strict JSON next waypoint decision."
        )
        variants = [base, concise, obstacle_focus, goal_focus]
        return variants[:max(1, self.variations_per_scene)]

    def nearest_mpc_label(self, scene_stamp_s):
        if not self.mpc_buffer:
            return None, None, None
        best_t, best_p = min(self.mpc_buffer, key=lambda tp: abs(tp[0] - scene_stamp_s))
        dt = abs(best_t - scene_stamp_s)
        if dt > self.max_label_dt:
            return None, best_t, dt
        return best_p, best_t, dt

    def sample_once(self):
        if self.depth_image is None or self.depth_stamp_s is None or self.odom_stamp_s is None:
            self.get_logger().warn('Waiting for depth+odometry for dataset sampling...', throttle_duration_sec=2.0)
            return

        scene_dt = abs(self.depth_stamp_s - self.odom_stamp_s)
        if scene_dt > self.max_scene_dt:
            self.get_logger().warn(
                f'Skipping unsynced scene (|depth-odom|={scene_dt:.3f}s)',
                throttle_duration_sec=1.0
            )
            return

        scene_stamp_s = max(self.depth_stamp_s, self.odom_stamp_s)
        label, label_time_s, label_dt = self.nearest_mpc_label(scene_stamp_s)
        if label is None:
            self.get_logger().warn(
                f'No near-time MPC label for scene (dt={label_dt if label_dt is not None else -1:.3f}s)',
                throttle_duration_sec=1.0
            )
            return

        env_vector = self.build_environment_vector()
        env_text = self.translate_vector_to_nlp(env_vector)
        prompts = self.build_prompt_variations(env_vector, env_text)
        self.scene_counter += 1

        for vidx, xi in enumerate(prompts):
            yi = {
                'mpc_next_waypoint_ned_m': [float(label[0]), float(label[1]), float(label[2])]
            }
            row = {
                'scene_id': self.scene_counter,
                'variation_id': vidx,
                'timestamp_scene_s': float(scene_stamp_s),
                'timestamp_depth_s': float(self.depth_stamp_s),
                'timestamp_odom_s': float(self.odom_stamp_s),
                'timestamp_mpc_label_s': float(label_time_s),
                'scene_to_label_dt_s': float(label_dt if label_dt is not None else 0.0),
                'Xi': xi,
                'Yi': yi,
                'environment_vector': env_vector,
            }
            with open(self.output_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
            self.sample_counter += 1

            if self.max_samples > 0 and self.sample_counter >= self.max_samples:
                self.get_logger().info(
                    f'Max samples reached ({self.sample_counter}). Dataset saved to {self.output_path}'
                )
                rclpy.shutdown()
                return

        self.get_logger().info(
            f'Wrote scene {self.scene_counter} with {len(prompts)} variations '
            f'(total samples={self.sample_counter}, label_dt={label_dt:.3f}s)',
            throttle_duration_sec=0.5
        )


def main(args=None):
    goal_override, ros_args = parse_goal_overrides(args)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = LLMFineTuneDatasetGenerator(goal_override=goal_override)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
