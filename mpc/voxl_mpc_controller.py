#!/usr/bin/env python3
"""
VOXL2 HITL MPC Controller with Vision-based Obstacle Avoidance
Tailored for voxl-mpa-to-ros2 sensors

Subscribes to:
  - /tracking (tracking camera)
  - /tof_depth (ToF depth)
  - /qvio (VIO odometry)

Publishes to:
  - /fmu/in/trajectory_setpoint
  - /mpc/trajectory (for comparison)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode
from cv_bridge import CvBridge
import cv2
import numpy as np
import cvxpy as cp
from scipy.spatial import distance


class VoxlMPCController(Node):
    
    def __init__(self):
        super().__init__('voxl_mpc_controller')
        
        # Parameters
        self.declare_parameter('goal_x', 15.0)
        self.declare_parameter('goal_y', 10.0)
        self.declare_parameter('goal_z', 2.0)
        self.declare_parameter('prediction_horizon', 10)
        self.declare_parameter('control_horizon', 5)
        self.declare_parameter('dt', 0.2)
        self.declare_parameter('max_velocity', 2.0)
        self.declare_parameter('obstacle_threshold', 2.5)
        self.declare_parameter('safety_distance', 1.5)
        
        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value
        ])
        
        self.N = self.get_parameter('prediction_horizon').value
        self.M = self.get_parameter('control_horizon').value
        self.dt = self.get_parameter('dt').value
        self.v_max = self.get_parameter('max_velocity').value
        self.obs_threshold = self.get_parameter('obstacle_threshold').value
        self.safety_dist = self.get_parameter('safety_distance').value
        
        # State
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.obstacles = []
        self.bridge = CvBridge()
        self.tracking_image = None
        self.depth_image = None
        
        # VOXL MPA Subscribers
        self.tracking_sub = self.create_subscription(
            Image, '/tracking', self.tracking_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/tof_depth', self.depth_callback, 10)
        self.qvio_sub = self.create_subscription(
            Odometry, '/qvio', self.qvio_callback, 10)
        
        # PX4 Publishers
        self.traj_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.mpc_traj_pub = self.create_publisher(
            PoseStamped, '/mpc/trajectory', 10)
        
        # Timers
        self.mpc_timer = self.create_timer(self.dt, self.mpc_control_loop)
        self.offboard_timer = self.create_timer(0.1, self.publish_offboard_heartbeat)
        
        self.get_logger().info('VOXL2 HITL MPC Controller initialized')
        self.get_logger().info(f'Goal: {self.goal}')
    
    def tracking_callback(self, msg):
        try:
            self.tracking_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except Exception as e:
            self.get_logger().error(f'Tracking error: {e}')
    
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            self.detect_obstacles_from_depth()
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')
    
    def qvio_callback(self, msg):
        self.current_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.current_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
    
    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        self.offboard_mode_pub.publish(msg)
    
    def detect_obstacles_from_depth(self):
        """Detect obstacles from VOXL ToF depth"""
        if self.depth_image is None:
            return
        
        h, w = self.depth_image.shape
        obstacles = []
        
        # Sample depth image
        step = 20
        for v in range(0, h, step):
            for u in range(0, w, step):
                depth = self.depth_image[v, u]
                
                if depth < 0.5 or depth > self.obs_threshold or np.isnan(depth):
                    continue
                
                # Simple pinhole projection (adjust for VOXL ToF calibration)
                fx = 320.0  # Approximate focal length
                fy = 320.0
                cx = 320.0
                cy = 240.0
                
                x_cam = (u - cx) * depth / fx
                y_cam = (v - cy) * depth / fy
                z_cam = depth
                
                # Transform to NED body frame
                x_body = z_cam
                y_body = -x_cam
                z_body = -y_cam
                
                # Transform to NED world frame
                x_ned = self.current_position[0] + x_body
                y_ned = self.current_position[1] + y_body
                z_ned = self.current_position[2] + z_body
                
                obstacles.append([x_ned, y_ned, z_ned])
        
        if len(obstacles) > 0:
            obstacles = np.array(obstacles)
            self.obstacles = self.cluster_obstacles(obstacles)
            
            self.get_logger().info(
                f'Detected {len(self.obstacles)} obstacles',
                throttle_duration_sec=2.0
            )
    
    def cluster_obstacles(self, points, radius=0.5):
        """Cluster obstacle points"""
        if len(points) == 0:
            return []
        
        clusters = []
        used = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if used[i]:
                continue
            
            distances = distance.cdist([points[i]], points)[0]
            cluster_mask = distances < radius
            cluster_points = points[cluster_mask]
            used[cluster_mask] = True
            
            center = np.mean(cluster_points, axis=0)
            max_dist = np.max(distance.cdist([center], cluster_points)[0])
            
            clusters.append((center[0], center[1], center[2], max_dist + 0.3))
        
        return clusters
    
    def mpc_control_loop(self):
        """Main MPC control loop"""
        if np.linalg.norm(self.current_position) < 0.1:
            return
        
        dist_to_goal = np.linalg.norm(self.goal - self.current_position)
        if dist_to_goal < 0.5:
            self.get_logger().info('Goal reached!', throttle_duration_sec=1.0)
            return
        
        optimal_velocity = self.solve_mpc()
        
        if optimal_velocity is not None:
            self.publish_trajectory_setpoint(optimal_velocity)
            
            # Publish for comparison
            traj_msg = PoseStamped()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.pose.position.x = self.current_position[0] + optimal_velocity[0] * self.dt
            traj_msg.pose.position.y = self.current_position[1] + optimal_velocity[1] * self.dt
            traj_msg.pose.position.z = self.current_position[2] + optimal_velocity[2] * self.dt
            self.mpc_traj_pub.publish(traj_msg)
    
    def solve_mpc(self):
        """Solve MPC optimization"""
        x = cp.Variable((3, self.N + 1))
        u = cp.Variable((3, self.M))
        
        cost = 0
        Q = np.diag([10.0, 10.0, 10.0])
        R = np.diag([0.1, 0.1, 0.1])
        
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
            constraints.append(x[:, k+1] == x[:, k] + u[:, self.M-1] * self.dt)
        
        for k in range(self.M):
            constraints.append(cp.norm(u[:, k], 2) <= self.v_max)
        
        for obs_x, obs_y, obs_z, obs_r in self.obstacles:
            obs_pos = np.array([obs_x, obs_y, obs_z])
            for k in range(1, self.N + 1):
                current_to_obs = obs_pos - self.current_position
                dist = np.linalg.norm(current_to_obs)
                
                if dist < self.obs_threshold:
                    safe_dist = obs_r + self.safety_dist
                    constraints.append(
                        current_to_obs @ (x[:, k] - obs_pos) >= safe_dist * dist
                    )
        
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            
            if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
                return u[:, 0].value
            else:
                direction = self.goal - self.current_position
                dist = np.linalg.norm(direction)
                if dist > 0:
                    return (direction / dist) * min(self.v_max * 0.5, dist)
                
        except Exception as e:
            self.get_logger().error(f'MPC error: {e}')
            return None
    
    def publish_trajectory_setpoint(self, velocity):
        """Publish to PX4"""
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.velocity = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
        msg.yaw = 0.0
        self.traj_setpoint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VoxlMPCController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()