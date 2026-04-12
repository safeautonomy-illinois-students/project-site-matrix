#!/usr/bin/env python3
"""
Model Predictive Control (MPC) Vision-Based Obstacle Avoidance
Uses depth camera for obstacle detection and avoidance
Ready for deployment to real hardware (Starling 2)

Dependencies:
pip install rclpy opencv-python numpy cvxpy scipy

Author: Vision MPC Controller
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
import cvxpy as cp
from scipy.spatial import distance
import matplotlib.pyplot as plt

try:
    from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


class MPCVisionController(Node):
    """
    MPC Controller with Vision-based Obstacle Avoidance
    """
    
    def __init__(self):
        super().__init__('mpc_vision_controller')
        
        # Parameters
        self.declare_parameter('goal_x', 35.00)
        self.declare_parameter('goal_y', 0.00)
        self.declare_parameter('goal_z', 2.0)  # Gazebo frame (negative = up)
        self.declare_parameter('pointcloud_topic', '/depth_camera/points')
        self.declare_parameter('goal_frame', 'gazebo')  # gazebo(ENU) or ned
        self.declare_parameter('prediction_horizon', 10)
        self.declare_parameter('control_horizon', 5)
        self.declare_parameter('dt', 0.55)
        self.declare_parameter('max_velocity', 5)
        self.declare_parameter('max_control_slew', 0.5)  # m/s change per MPC step
        self.declare_parameter('obstacle_threshold', 2.5)  # meters
        self.declare_parameter('safety_distance', 1.5)  # meters
        self.declare_parameter('position_setpoint_lookahead_base_s', 0.5)
        self.declare_parameter('position_setpoint_lookahead_gain', 0.05)
        self.declare_parameter('position_setpoint_lookahead_min_s', 0.2)
        self.declare_parameter('position_setpoint_lookahead_max_s', 0.8)
        self.declare_parameter('latched_setpoint_acceptance_radius', 0.6)
        self.declare_parameter('enable_infeasible_yaw_recovery', True)
        self.declare_parameter('infeasible_yaw_rate_rad_s', 0.35)
        self.declare_parameter('infeasible_yaw_scan_duration_s', 2.0)
        self.declare_parameter('infeasible_recovery_hold_position', True)
        self.declare_parameter('solver_status_log_throttle_s', 0.5)
        self.declare_parameter('obstacle_log_throttle_s', 1.0)
        self.declare_parameter('obstacle_log_include_min_dist', True)
        
        self.goal_input = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value
        ])
        self.pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        self.goal_frame = str(self.get_parameter('goal_frame').value).strip().lower()
        self.goal = self.convert_goal_to_ned(self.goal_input, self.goal_frame)
        
        self.N = self.get_parameter('prediction_horizon').value
        self.M = self.get_parameter('control_horizon').value
        self.dt = self.get_parameter('dt').value
        self.v_max = self.get_parameter('max_velocity').value
        self.max_control_slew = self.get_parameter('max_control_slew').value
        self.obs_threshold = self.get_parameter('obstacle_threshold').value
        self.safety_dist = self.get_parameter('safety_distance').value
        self.enable_obstacle_avoidance = True
        self.lookahead_base = self.get_parameter('position_setpoint_lookahead_base_s').value
        self.lookahead_gain = self.get_parameter('position_setpoint_lookahead_gain').value
        self.lookahead_min = self.get_parameter('position_setpoint_lookahead_min_s').value
        self.lookahead_max = self.get_parameter('position_setpoint_lookahead_max_s').value
        self.latched_sp_accept_radius = self.get_parameter(
            'latched_setpoint_acceptance_radius'
        ).value
        self.enable_infeasible_yaw_recovery = bool(
            self.get_parameter('enable_infeasible_yaw_recovery').value
        )
        self.infeasible_yaw_rate_rad_s = float(
            self.get_parameter('infeasible_yaw_rate_rad_s').value
        )
        self.infeasible_yaw_scan_duration_s = float(
            self.get_parameter('infeasible_yaw_scan_duration_s').value
        )
        self.infeasible_recovery_hold_position = bool(
            self.get_parameter('infeasible_recovery_hold_position').value
        )
        self.solver_status_log_throttle_s = float(
            self.get_parameter('solver_status_log_throttle_s').value
        )
        self.obstacle_log_throttle_s = float(
            self.get_parameter('obstacle_log_throttle_s').value
        )
        self.obstacle_log_include_min_dist = bool(
            self.get_parameter('obstacle_log_include_min_dist').value
        )
        
        # State
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_yaw = 0.0
        self.rotation_ned_body = np.eye(3)
        self.obstacles = []  # List of (x, y, z, radius) in NED frame
        self.warned_fallback_pub = False
        self.warned_solver = False
        self.prev_velocity_cmd = np.zeros(3)
        # PX4 setpoint latch: hold one target until close, then advance.
        self.latched_target_position = None
        self.latched_target_yaw = None
        self.recovery_active = False
        self.recovery_start_time = None
        self.recovery_yaw_sign = 1.0
        self.recovery_yaw = 0.0
        self.recovery_yaw_center = 0.0
        self.recovery_hold_position = None
        self.start_time = self.get_clock().now()
        self.history_t = []
        self.history_x = []
        self.history_y = []
        self.history_error = []
        self.history_effort = []
        self.plot_ready = False
        self.fig = None
        self.ax_xy = None
        self.ax_err = None
        self.ax_u = None
        self.path_line = None
        self.err_line = None
        self.effort_line = None

        # Prefer conic solvers for SOC constraints (norm <= vmax)
        self.available_solvers = cp.installed_solvers()
        if 'ECOS' in self.available_solvers:
            self.conic_solver = cp.ECOS
        elif 'SCS' in self.available_solvers:
            self.conic_solver = cp.SCS
        else:
            self.conic_solver = None
            self.get_logger().error(
                'No conic solver available to CVXPY. '
                'Install ECOS or SCS (python3-ecos or python3-scs).'
            )

        # PX4 uORB -> ROS 2 typically uses best-effort, volatile QoS
        self.odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # Subscribers
        self.depth_sub = self.create_subscription(
            PointCloud2, self.pointcloud_topic, self.pointcloud_callback, 10)
        if HAS_PX4_MSGS:
            self.odom_sub = self.create_subscription(
                VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, self.odom_qos)
        else:
            self.odom_sub = self.create_subscription(
                Odometry, '/fmu/out/vehicle_odometry', self.odometry_callback, self.odom_qos)
        
        # Publishers
        self.mpc_traj_pub = self.create_publisher(
            PoseStamped, '/mpc/trajectory', 10)
        if HAS_PX4_MSGS:
            self.traj_setpoint_pub = self.create_publisher(
                TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
            self.offboard_mode_pub = self.create_publisher(
                OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
            self.cmd_vel_pub = None
        else:
            self.traj_setpoint_pub = None
            self.offboard_mode_pub = None
            self.cmd_vel_pub = self.create_publisher(
                TwistStamped, '/fmu/in/setpoint_velocity', 10)
        
        # Vision data (PointCloud2 in camera frame)
        # Camera optical -> body (FRD/NED-aligned body axes).
        # Preserves previous mapping: x_body=z_cam, y_body=-x_cam, z_body=-y_cam
        self.rotation_body_camera = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)
        # x500_depth SDF: base_link -> camera_link (0.12, 0.03, 0.242)
        # OakD-Lite SDF: camera_link -> StereoOV7251 (0.01233, -0.03, 0.01878)
        self.translation_body_camera_m = np.array([0.13233, 0.0, 0.26078], dtype=float)
        
        # MPC Timer
        self.mpc_timer = self.create_timer(self.dt, self.mpc_control_loop)
        if HAS_PX4_MSGS:
            self.offboard_timer = self.create_timer(0.1, self.publish_offboard_heartbeat)
        
        self.get_logger().info('MPC Vision Controller initialized')
        self.get_logger().info(
            f'Goal input ({self.goal_frame}): {self.goal_input} -> NED goal: {self.goal}'
        )
        self.get_logger().info(
            f'Aggressiveness params: v_max={self.v_max:.2f}, dt={self.dt:.2f}, '
            f'max_slew={self.max_control_slew:.2f}'
        )
        self.get_logger().info(
            f'Obstacle avoidance enabled: {self.enable_obstacle_avoidance}'
        )
        self.get_logger().info(f'Pointcloud topic: {self.pointcloud_topic}')
        self.setup_live_plot()
        if HAS_PX4_MSGS:
            self.get_logger().info('Publishing PX4 trajectory setpoints to /fmu/in/trajectory_setpoint')
        else:
            self.get_logger().warn(
                'px4_msgs not found in Python env; falling back to /fmu/in/setpoint_velocity'
            )
    
    def pointcloud_callback(self, msg):
        """Update obstacle list from depth pointcloud when enabled."""
        try:
            self.detect_obstacles_from_pointcloud(msg)
        except Exception as e:
            self.get_logger().error(f'Pointcloud processing error: {e}')
    
    def detect_obstacles_from_pointcloud(self, msg):
        """Transform cloud points camera->body->NED and cluster occupied points."""
        obstacles = []
        valid_points = 0

        for x_cam, y_cam, z_cam in pc2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True
        ):
            valid_points += 1
            if not (np.isfinite(x_cam) and np.isfinite(y_cam) and np.isfinite(z_cam)):
                continue
            point_cam = np.array([x_cam, y_cam, z_cam], dtype=float)
            point_body = self.rotation_body_camera @ point_cam + self.translation_body_camera_m
            if not np.all(np.isfinite(point_body)):
                continue
            point_ned = self.current_position + self.rotation_ned_body @ point_body
            if not np.all(np.isfinite(point_ned)):
                continue
            if np.linalg.norm(point_ned - self.current_position) > self.obs_threshold:
                continue
            obstacles.append(point_ned.tolist())
        
        if len(obstacles) > 0:
            obstacles = np.array(obstacles)
            self.obstacles = self.cluster_obstacles(obstacles)
        else:
            self.obstacles = []
        if self.obstacles:
            formatted = ', '.join(
                f'({ox:.2f},{oy:.2f},{oz:.2f},r={orad:.2f})'
                for ox, oy, oz, orad in self.obstacles
            )
            self.get_logger().info(
                f'Obstacles: {formatted}',
                throttle_duration_sec=self.obstacle_log_throttle_s
            )
            if self.obstacle_log_include_min_dist:
                centers = np.array(
                    [[ox, oy, oz] for ox, oy, oz, _ in self.obstacles],
                    dtype=float
                )
                min_dist = float(np.min(np.linalg.norm(centers - self.current_position, axis=1)))
                self.get_logger().info(
                    f'Obstacle summary: count={len(self.obstacles)}, min_dist={min_dist:.2f} m',
                    throttle_duration_sec=self.obstacle_log_throttle_s
                )
        else:
            self.get_logger().info(
                'Obstacles: none',
                throttle_duration_sec=self.obstacle_log_throttle_s
            )
        self.get_logger().info(
            f'Pointcloud processed: points={valid_points}, clusters={len(self.obstacles)}',
            throttle_duration_sec=self.obstacle_log_throttle_s
        )
    
    def cluster_obstacles(self, points, radius=0.5):
        """Cluster obstacle points into spheres."""
        if len(points) == 0:
            return []
        points = np.asarray(points, dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) == 0:
            return []
        
        clusters = []
        used = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if used[i]:
                continue
            
            distances = distance.cdist([points[i]], points)[0]
            distances = np.where(np.isfinite(distances), distances, np.inf)
            cluster_mask = distances < radius
            cluster_points = points[cluster_mask]
            if len(cluster_points) == 0:
                used[i] = True
                continue
            
            used[cluster_mask] = True
            
            center = np.mean(cluster_points, axis=0)
            if not np.all(np.isfinite(center)):
                continue
            max_dist = np.max(distance.cdist([center], cluster_points)[0])
            
            clusters.append((center[0], center[1], center[2], max_dist + 0.3))
        
        return clusters
    
    def odometry_callback(self, msg):
        """Update current position and velocity"""
        self.current_position = np.array([
            msg.pose.pose.position.y,
            msg.pose.pose.position.x,
            -msg.pose.pose.position.z
        ])
        
        self.current_velocity = np.array([
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.x,
            -msg.twist.twist.linear.z
        ])
        q = msg.pose.pose.orientation
        rotation_enu_body = self.quaternion_to_rotation_matrix(
            q.w, q.x, q.y, q.z
        )
        rotation_enu_to_ned = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=float,
        )
        self.rotation_ned_body = rotation_enu_to_ned @ rotation_enu_body
        self.current_yaw = float(np.arctan2(self.rotation_ned_body[1, 0], self.rotation_ned_body[0, 0]))

    def vehicle_odometry_callback(self, msg):
        """Update current state from px4_msgs/VehicleOdometry (NED)."""
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
            float(msg.q[0]),
            float(msg.q[1]),
            float(msg.q[2]),
            float(msg.q[3]),
        )
        self.current_yaw = self.quaternion_to_yaw(
            float(msg.q[0]),
            float(msg.q[1]),
            float(msg.q[2]),
            float(msg.q[3]),
        )

    @staticmethod
    def quaternion_to_rotation_matrix(w, x, y, z):
        """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
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
    def quaternion_to_yaw(w, x, y, z):
        """Extract yaw angle from quaternion (w, x, y, z)."""
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-9:
            return 0.0
        w, x, y, z = w / n, x / n, y / n, z / n
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    
    def mpc_control_loop(self):
        """Main MPC control loop"""
        if np.linalg.norm(self.current_position) < 0.1:
            # Not initialized yet
            return
        
        # Check if goal reached
        dist_to_goal = np.linalg.norm(self.goal - self.current_position)
        if dist_to_goal < 0.5:
            self.get_logger().info('Goal reached!', throttle_duration_sec=1.0)
            self.update_live_plot(np.zeros(3))
            self.prev_velocity_cmd = np.zeros(3)
            self.latched_target_position = None
            self.latched_target_yaw = None
            self.stop_infeasible_recovery()
            self.publish_zero_velocity()
            return
        
        # Solve MPC
        solve_result = self.solve_mpc()
        mode = solve_result['mode']

        if mode == 'infeasible' and self.enable_infeasible_yaw_recovery:
            if HAS_PX4_MSGS:
                target_position, target_yaw = self.get_infeasible_recovery_setpoint()
                self.latched_target_position = np.array(target_position, dtype=float)
                self.latched_target_yaw = float(target_yaw)
                self.publish_trajectory_setpoint(target_position, yaw=target_yaw)
                self.publish_mpc_trajectory(target_position)
            else:
                self.publish_velocity(np.zeros(3))
            self.update_live_plot(np.zeros(3))
            self.get_logger().warn(
                'MPC infeasible: holding position and yaw-scanning for fresh depth view',
                throttle_duration_sec=1.0
            )
            return

        if mode != 'infeasible':
            self.stop_infeasible_recovery()

        optimal_velocity = solve_result.get('u')
        if optimal_velocity is not None:
            # Publish MPC trajectory point for comparison/visualization.
            target_position = self.compute_target_position(optimal_velocity, dist_to_goal)

            # Send command on the same PX4 topics as llm_planner.
            if HAS_PX4_MSGS:
                self.latched_target_yaw = None
                target_position = self.update_latched_target(target_position)
                self.publish_trajectory_setpoint(target_position)
            else:
                self.publish_velocity(optimal_velocity)
            self.publish_mpc_trajectory(target_position)
            
            self.get_logger().info(
                f'Pos: [{self.current_position[0]:.2f}, '
                f'{self.current_position[1]:.2f}, '
                f'{self.current_position[2]:.2f}], '
                f'Vel: [{optimal_velocity[0]:.2f}, '
                f'{optimal_velocity[1]:.2f}, '
                f'{optimal_velocity[2]:.2f}], '
                f'Dist: {dist_to_goal:.2f}m',
                throttle_duration_sec=0.5
            )
            self.update_live_plot(optimal_velocity)
            self.prev_velocity_cmd = np.array(optimal_velocity, dtype=float)

    def convert_goal_to_ned(self, goal, goal_frame):
        """
        Convert user-provided goal coordinates to local NED.

        Gazebo world is typically ENU:
          x = East, y = North, z = Up
        PX4 local frame is NED:
          x = North, y = East, z = Down
        ENU -> NED: [x_n, y_e, z_d] = [y_enu, x_enu, -z_enu]
        """
        g = np.array(goal, dtype=float)
        if goal_frame in ('gazebo', 'gazebo_enu', 'enu', 'map'):
            return np.array([g[1], g[0], -g[2]], dtype=float)
        if goal_frame in ('ned', 'px4', 'px4_ned'):
            return g

        self.get_logger().warn(
            f"Unknown goal_frame='{goal_frame}'. Expected 'gazebo' or 'ned'; assuming NED."
        )
        return g

    def compute_target_position(self, optimal_velocity, tracking_error):
        """Adaptive lookahead position setpoint based on tracking error magnitude."""
        lookahead = self.lookahead_base + self.lookahead_gain * tracking_error
        lookahead = float(np.clip(lookahead, self.lookahead_min, self.lookahead_max))
        target = self.current_position + optimal_velocity * lookahead

        # Clamp to goal when step would overshoot.
        to_goal = self.goal - self.current_position
        step = target - self.current_position
        if np.linalg.norm(step) > np.linalg.norm(to_goal):
            return self.goal.copy()
        return target

    def update_latched_target(self, candidate_target):
        """Hold one PX4 setpoint until close; then switch to the next MPC target."""
        if self.latched_target_position is None:
            self.latched_target_position = np.array(candidate_target, dtype=float)
            return self.latched_target_position
        if np.linalg.norm(self.current_position - self.latched_target_position) <= self.latched_sp_accept_radius:
            self.latched_target_position = np.array(candidate_target, dtype=float)
        return self.latched_target_position
    
    def solve_mpc(self):
        """
        Solve MPC optimization problem
        Minimize: distance to goal + control effort
        Subject to: velocity limits, obstacle avoidance
        """
        x = cp.Variable((3, self.N + 1))  # States
        u = cp.Variable((3, self.M))      # Control inputs
        
        cost = 0
        Q = np.diag([8.0, 8.0, 8.0])      # State cost
        R = np.diag([0.4, 0.4, 0.4])      # Higher effort penalty for smoother commands
        
        for k in range(self.N):
            cost += cp.quad_form(x[:, k] - self.goal, Q)
            
            if k < self.M:
                cost += cp.quad_form(u[:, k], R)
        
        cost += cp.quad_form(x[:, self.N] - self.goal, Q * 10)
        
        constraints = []
        
        constraints.append(x[:, 0] == self.current_position)
        
        for k in range(self.M):
            constraints.append(x[:, k+1] == x[:, k] + u[:, k] * self.dt)
        
        for k in range(self.M, self.N):
            constraints.append(
                x[:, k+1] == x[:, k] + u[:, self.M-1] * self.dt
            )
        
        for k in range(self.M):
            constraints.append(cp.norm(u[:, k], 2) <= self.v_max)
            if k == 0:
                constraints.append(
                    cp.norm(u[:, k] - self.prev_velocity_cmd, 2) <= self.max_control_slew
                )
            else:
                constraints.append(
                    cp.norm(u[:, k] - u[:, k-1], 2) <= self.max_control_slew
                )
        
        self.add_obstacle_constraints(constraints, x)
        
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            if self.conic_solver is None:
                if not self.warned_solver:
                    self.get_logger().error(
                        f'CVXPY conic solver missing. Installed solvers: {self.available_solvers}'
                    )
                    self.warned_solver = True
                return {
                    'mode': 'fallback',
                    'u': self.compute_goal_fallback_velocity(),
                    'status': 'missing_solver'
                }

            problem.solve(solver=self.conic_solver, verbose=False, warm_start=True)

            self.get_logger().info(
                f'MPC status={problem.status}, solver={self.conic_solver}, obstacles={len(self.obstacles)}',
                throttle_duration_sec=self.solver_status_log_throttle_s
            )

            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return {'mode': 'mpc', 'u': u[:, 0].value, 'status': problem.status}
            if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
                return {'mode': 'infeasible', 'u': None, 'status': problem.status}
            return {
                'mode': 'fallback',
                'u': self.compute_goal_fallback_velocity(),
                'status': problem.status
            }
        except Exception as e:
            self.get_logger().error(
                f'MPC solve error: {e}',
                throttle_duration_sec=self.solver_status_log_throttle_s
            )
            return {
                'mode': 'fallback',
                'u': self.compute_goal_fallback_velocity(),
                'status': 'solve_error'
            }

    def compute_goal_fallback_velocity(self):
        """Fallback translational command when MPC is unavailable."""
        direction = self.goal - self.current_position
        dist = np.linalg.norm(direction)
        if dist <= 1e-9:
            return np.zeros(3)
        fallback = (direction / dist) * min(self.v_max * 0.4, dist)
        delta = fallback - self.prev_velocity_cmd
        delta_norm = np.linalg.norm(delta)
        if delta_norm > self.max_control_slew and delta_norm > 1e-9:
            fallback = self.prev_velocity_cmd + delta / delta_norm * self.max_control_slew
        return fallback

    def stop_infeasible_recovery(self):
        """Exit yaw-recovery mode and clear its state."""
        if self.recovery_active:
            self.get_logger().info('Exited infeasible yaw-recovery mode')
        self.recovery_active = False
        self.recovery_start_time = None
        self.recovery_hold_position = None
        self.recovery_yaw_sign = 1.0
        self.latched_target_yaw = None

    def get_infeasible_recovery_setpoint(self):
        """Hold position and yaw-scan to refresh depth observations."""
        now = self.get_clock().now()
        if not self.recovery_active:
            self.recovery_active = True
            self.recovery_start_time = now
            self.recovery_yaw_sign = 1.0
            self.recovery_yaw_center = float(self.current_yaw)
            self.recovery_yaw = float(self.current_yaw)
            self.recovery_hold_position = np.array(self.current_position, dtype=float)
            self.get_logger().warn('Entered infeasible yaw-recovery mode')

        elapsed = (now - self.recovery_start_time).nanoseconds * 1e-9
        if elapsed >= self.infeasible_yaw_scan_duration_s:
            self.stop_infeasible_recovery()
            return np.array(self.current_position, dtype=float), float(self.current_yaw)

        yaw_limit = np.deg2rad(35.0)
        self.recovery_yaw += self.recovery_yaw_sign * self.infeasible_yaw_rate_rad_s * self.dt
        yaw_max = self.recovery_yaw_center + yaw_limit
        yaw_min = self.recovery_yaw_center - yaw_limit
        if self.recovery_yaw >= yaw_max:
            self.recovery_yaw = yaw_max
            self.recovery_yaw_sign = -1.0
        elif self.recovery_yaw <= yaw_min:
            self.recovery_yaw = yaw_min
            self.recovery_yaw_sign = 1.0

        if self.infeasible_recovery_hold_position and self.recovery_hold_position is not None:
            target_position = self.recovery_hold_position.copy()
        else:
            target_position = np.array(self.current_position, dtype=float)
        return target_position, float(self.recovery_yaw)

    def add_obstacle_constraints(self, constraints, x):
        """Add obstacle avoidance constraints only when enabled."""
        for obs_x, obs_y, obs_z, obs_r in self.obstacles:
            obs_pos = np.array([obs_x, obs_y, obs_z])
            current_to_obs = obs_pos - self.current_position
            dist = np.linalg.norm(current_to_obs)
            if dist >= self.obs_threshold:
                continue

            safe_dist = obs_r + self.safety_dist
            for k in range(1, self.N + 1):
                constraints.append(
                    current_to_obs @ (x[:, k] - obs_pos) >= safe_dist * dist
                )
    
    def publish_velocity(self, velocity):
        """Publish velocity command"""
        if self.cmd_vel_pub is None:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(velocity[0])
        msg.twist.linear.y = float(velocity[1])
        msg.twist.linear.z = float(velocity[2])
        self.cmd_vel_pub.publish(msg)

        if not self.warned_fallback_pub:
            self.get_logger().warn(
                'Using /fmu/in/setpoint_velocity fallback. '
                'Install px4_msgs to publish /fmu/in/trajectory_setpoint.'
            )
            self.warned_fallback_pub = True

    def publish_offboard_heartbeat(self):
        """Publish OffboardControlMode heartbeat and resend latched PX4 setpoint."""
        if self.offboard_mode_pub is None:
            return
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        self.offboard_mode_pub.publish(msg)
        # Keep feeding the same target at 10 Hz so PX4 can converge before switching.
        if self.latched_target_position is not None:
            self.publish_trajectory_setpoint(
                self.latched_target_position,
                yaw=self.latched_target_yaw
            )

    def publish_trajectory_setpoint(self, target_position, yaw=None):
        """Publish position setpoint directly to PX4 trajectory setpoint topic."""
        if self.traj_setpoint_pub is None:
            return
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [
            float(target_position[0]),
            float(target_position[1]),
            float(target_position[2]),
        ]
        msg.yaw = 0.0 if yaw is None else float(yaw)
        self.traj_setpoint_pub.publish(msg)

    def publish_mpc_trajectory(self, position):
        """Publish MPC next position for comparison with LLM trajectory."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        self.mpc_traj_pub.publish(msg)
    
    def publish_zero_velocity(self):
        """Stop the drone"""
        if HAS_PX4_MSGS:
            self.latched_target_yaw = None
            self.publish_trajectory_setpoint(self.current_position)
        else:
            self.publish_velocity(np.zeros(3))

    def setup_live_plot(self):
        """Initialize real-time diagnostic plots."""
        try:
            plt.ion()
            self.fig, (self.ax_xy, self.ax_err, self.ax_u) = plt.subplots(3, 1, figsize=(9, 10))

            self.path_line, = self.ax_xy.plot([], [], 'b-', linewidth=2, label='Drone position')
            self.ax_xy.scatter(
                [self.goal[0]], [self.goal[1]], c='red', s=80, marker='*', label='Goal'
            )
            self.ax_xy.set_title('MPC Position (XY)')
            self.ax_xy.set_xlabel('X (m)')
            self.ax_xy.set_ylabel('Y (m)')
            self.ax_xy.grid(True, alpha=0.3)
            self.ax_xy.legend(loc='best')
            self.ax_xy.axis('equal')

            self.err_line, = self.ax_err.plot([], [], 'tab:orange', linewidth=2, label='MPC error')
            self.ax_err.set_title('Tracking Error vs Time')
            self.ax_err.set_xlabel('Time (s)')
            self.ax_err.set_ylabel('||goal - position|| (m)')
            self.ax_err.grid(True, alpha=0.3)
            self.ax_err.legend(loc='best')

            self.effort_line, = self.ax_u.plot([], [], 'tab:green', linewidth=2, label='Control effort')
            self.ax_u.set_title('Control Effort vs Time')
            self.ax_u.set_xlabel('Time (s)')
            self.ax_u.set_ylabel('||u|| (m/s)')
            self.ax_u.grid(True, alpha=0.3)
            self.ax_u.legend(loc='best')

            self.fig.tight_layout()
            self.plot_ready = True
            self.get_logger().info('Real-time matplotlib plots enabled')
        except Exception as exc:
            self.plot_ready = False
            self.get_logger().warn(f'Could not initialize matplotlib plot: {exc}')

    def update_live_plot(self, optimal_velocity):
        """Update position, error, and control-effort plots."""
        if not self.plot_ready:
            return

        t = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        self.history_t.append(float(t))
        self.history_x.append(float(self.current_position[0]))
        self.history_y.append(float(self.current_position[1]))
        self.history_error.append(float(np.linalg.norm(self.goal - self.current_position)))
        self.history_effort.append(float(np.linalg.norm(optimal_velocity)))

        self.path_line.set_data(self.history_x, self.history_y)
        self.err_line.set_data(self.history_t, self.history_error)
        self.effort_line.set_data(self.history_t, self.history_effort)

        self.ax_xy.relim()
        self.ax_xy.autoscale_view()
        self.ax_err.relim()
        self.ax_err.autoscale_view()
        self.ax_u.relim()
        self.ax_u.autoscale_view()

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    controller = MPCVisionController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
