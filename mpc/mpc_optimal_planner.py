#!/usr/bin/env python3
"""
MPC-based Ground Vehicle Trajectory Planner for ROS2 (2D)
==========================================================

System:
  - GEM sensor  : /depth_camera/image_raw and /scan/points
  - GEM output  : /ackermann_cmd (ackermann_msgs/AckermannDrive)
  - GEM state   : /odom (nav_msgs/Odometry)

MPC Formulation (discrete-time, horizon T steps)
-------------------------------------------------
Solver state  : [x, y, yaw, vx, vy] in the horizontal plane
Solver control: [ax, ay, yaw_rate] planar accelerations + heading-rate command
GEM output    : speed + steering via AckermannDrive
Dynamics      : x(k+1) = x(k) + Ts * v(k),  v(k+1) = v(k) + Ts * a(k)

Optimisation:
  MATLAB-style sequential convex programming (SCP):
    - real cost   phi(X,U),
    - convexified phi_hat(X,U) around current iterate,
    - trust-region accept/reject updates,
    - obstacle linearisation d_safe*||d_now|| - d_now^T d(X) <= 0.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from concurrent.futures import Future, ThreadPoolExecutor
import time
import numpy as np
from numpy.linalg import norm
import cv2
from cv_bridge import CvBridge
from collections import deque
import os
import json
try:
    import cvxpy as cp
except Exception:
    cp = None
from native_mpc_backend import (
    native_backend_available,
    native_osqp_available,
    solve_osqp as solve_native_osqp,
    solve_subgradient as solve_native_subgradient,
)

# ROS2 message types
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDrive
try:
    from sensor_msgs_py import point_cloud2
except Exception:
    point_cloud2 = None
try:
    import torch
except Exception:
    torch = None

# ─────────────────────────────────────────────────────────────────────────────
# MPC PARAMETERS  (tune these)
# ─────────────────────────────────────────────────────────────────────────────
Ts               = 0.1    # sampling time [s]
T_horizon        = 50     # MPC prediction horizon [steps]
V_MAX            = 15.0   # per-axis max commanded velocity [m/s]
A_MAX            = 8.0    # per-axis max commanded acceleration [m/s^2]
N_OBS            = 1      # nearest-point method: N=1 (Section III-A)
SCP_MAX_OUTER_ITERS = 4   # sequential convexification iterations per MPC cycle
SCP_INNER_ITERS  = 6      # projected-gradient iterations per convex sub-problem
SCP_EPS          = 1e-3   # stop when trajectory update is small
SCP_LR           = 0.003  # projected-gradient base step size (stability-critical)
SCP_LR_BACKTRACK = 0.5    # multiplicative backtracking factor for failed inner steps
SCP_LS_MAX_STEPS = 6      # max line-search reductions per inner iteration
SCP_LAMBDA       = 100.0  # penalty multiplier (constraints)
SCP_ALPHA        = 0.1    # trust-region acceptance parameter
SCP_BETA_GROW    = 1.1    # trust-region growth factor after accepted step
SCP_BETA_SHRINK  = 0.5    # trust-region shrink factor after rejected step
TRUST_POS0       = 1.0    # initial trust region on position trajectory [m]
TRUST_U0         = 1.0    # initial trust region on control [m/s^2]
MIN_CLEARANCE    = 1.5    # desired clearance [m] from nearest obstacle point

# GEM simulator depth sensor from gem_description/urdf/gem.urdf.xacro:
#   horizontal_fov=1.047 rad, 800x600, near=0.1 m, far=300 m.
DEPTH_HFOV = 59.989  # GEM depth camera horizontal FOV [deg]
DEPTH_VFOV = 46.798  # GEM depth camera vertical FOV [deg] from 800x600 + HFOV
DEPTH_MAX  = 300.0   # depth camera far clip [m]
DEPTH_MIN  = 0.1     # depth camera near clip [m]

RGB_HFOV_DEG   = 80.0
RGB_IMG_WIDTH  = 800
RGB_IMG_HEIGHT = 600

LANE_REFERENCE_TOPIC         = '/mpc/lane_reference'
LANE_MASK_TOPIC              = '/mpc/lane_mask'
DEFAULT_GEM_RGB_TOPIC        = '/camera/image_raw'
DEFAULT_ENABLE_LANE_TRACKING = True
DEFAULT_LANE_MODEL_PATH      = os.path.join('lane_Segmentation', 'data', 'checkpoints', 'epoch100.pth')
DEFAULT_BEV_CONFIG_PATH      = os.path.join('lane_Segmentation', 'data', 'bev_config.json')
LANE_CENTER_WEIGHT           = 4.0
LANE_PROGRESS_STEP_M         = 0.8
LANE_LOOKAHEAD_MIN_M         = 4.0
LANE_LOOKAHEAD_MAX_M         = 20.0
LANE_MASK_THRESHOLD          = 127
LANE_MIN_POINTS              = 8
LANE_REFERENCE_TIMEOUT_SEC   = 0.5
LANE_BLEND_FAR_METERS        = 8.0
LANE_HEADING_REF_SPEED_MPS   = 3.0
LANE_HEADING_ERROR_WEIGHT    = 1.5
YAW_ERROR_WEIGHT            = 3.0
YAW_RATE_WEIGHT             = 0.35
MAX_YAW_RATE_RADPS          = 0.8

FIFO_LEN              = 200   # local obstacle map FIFO size (Section III-B)
SETPOINT_LOG_EVERY_N  = 10    # print debug every N control ticks
DEBUG_PLOT_ENABLED    = True
DEBUG_PLOT_RATE_HZ    = 5.0
DEBUG_PLOT_OBS_MAX    = 200
DEBUG_PATH_HISTORY_LEN = 300

DEPTH_FILTER_MIN_FORWARD_M  = 0.35
DEPTH_FILTER_MAX_FORWARD_M  = 12.0
DEPTH_FILTER_MAX_LATERAL_M  = 4.0

MPC_LABEL_WAYPOINT_COUNT = 5

LOCAL_PREDICTION_SEQUENCE_TOPIC    = '/mpc/local_prediction_sequence'
FRESH_COMMITTED_WAYPOINT_TOPIC     = '/mpc/committed_waypoint_fresh'
EXECUTED_COMMITTED_WAYPOINT_TOPIC  = '/mpc/committed_waypoint_executed'
PUBLISH_PERIOD_SEC  = 0.05
SOLVE_WARN_SEC      = 0.10
RATE_WINDOW_LEN     = 40
GOAL_REACHED_THRESHOLD_M = 0.5

ENABLE_DELAY_COMPENSATION  = False
DELAY_COMPENSATION_SEC     = 0.15

DEFAULT_MPC_SOLVER_BACKEND       = 'python'
DEFAULT_ENABLE_DEPTH_CAMERA      = True
DEFAULT_ENABLE_LIDAR             = True
DEFAULT_GEM_DEPTH_TOPIC          = '/depth_camera/image_raw'
DEFAULT_GEM_LIDAR_TOPIC          = '/scan/points'
DEFAULT_GEM_ODOM_TOPIC           = '/odom'
DEFAULT_GEM_ACKERMANN_TOPIC      = '/ackermann_cmd'
DEFAULT_GEM_MAX_SPEED_MPS        = 6.0
DEFAULT_GEM_WHEELBASE_M          = 1.75
DEFAULT_GEM_MAX_STEERING_RAD     = 0.61
DEFAULT_GEM_STEERING_KP          = 1.8
DEFAULT_OBSTACLE_CONSTRAINTS_DISABLED = False

U_LIMS   = np.array([A_MAX, A_MAX, MAX_YAW_RATE_RADPS], dtype=float)
VEL_LIMS = np.array([V_MAX, V_MAX], dtype=float)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from lane_Segmentation.model_utils import load_model as load_lane_model, inference as lane_inference
    from lane_Segmentation.line_fit import lane_fit, perspective_transform, closest_point_on_polynomial
except Exception:
    load_lane_model = None
    lane_inference = None
    lane_fit = None
    perspective_transform = None
    closest_point_on_polynomial = None


def _default_bev_config() -> dict:
    return {
        'bev_world_dim': [15.0, 20.0],
        'unit_conversion_factor': [15.0 / 600.0, 20.0 / 800.0],
        'src': [
            [43.54545454545453, 351.3446494845361],
            [165.68520578420465, 602.3177445323929],
            [634.3147942157953, 602.3177445323929],
            [756.4545454545454, 351.3446494845361],
        ],
    }


class LaneCenterDetector:
    """SimpleEnet + BEV + polynomial fit wrapper returning a lane centerline in body/world frames."""

    def __init__(self, node: Node, bridge: CvBridge):
        self._node = node
        self._bridge = bridge
        self._enabled = False
        self._model = None
        self._dev = None
        self._bev_cfg = _default_bev_config()

        if (torch is None or load_lane_model is None or lane_inference is None
                or lane_fit is None or perspective_transform is None
                or closest_point_on_polynomial is None):
            self._node.get_logger().warn(
                'Lane tracking disabled: lane segmentation dependencies are unavailable.'
            )
            return

        bev_path = str(self._node.get_parameter('lane_bev_config_path').value)
        if bev_path and os.path.exists(bev_path):
            try:
                with open(bev_path, 'r', encoding='utf-8') as f:
                    self._bev_cfg = json.load(f)
            except Exception as exc:
                self._node.get_logger().warn(
                    f'Failed to load BEV config {bev_path}: {exc}; using built-in defaults.'
                )

        model_path = str(self._node.get_parameter('lane_model_path').value)
        if model_path:
            os.environ.setdefault('LANE_MODEL_PATH', model_path)
        try:
            self._dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model = load_lane_model()
            self._model = self._model.to(self._dev)
            self._model.eval()
            self._enabled = True
        except Exception as exc:
            self._node.get_logger().warn(
                f'Lane tracking disabled: unable to load lane model ({exc}).'
            )

    def is_enabled(self) -> bool:
        return self._enabled

    def process(self, msg: Image, yaw_world: float, pos_world: np.ndarray) -> dict | None:
        """
        Run lane segmentation on an RGB image.

        yaw_world   : current vehicle heading [rad] in world frame (used only to
                      rotate body-frame points into world frame)
        pos_world   : (2,) vehicle XY position in world frame [m]
        Returns a dict with 'path_body' and 'path_world' as (N,2) arrays, or None.
        """
        if not self._enabled:
            return None

        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self._node.get_logger().warn(f'RGB decode error: {exc}')
            return None

        mask = lane_inference(self._model, image, self._dev)
        if mask is None:
            return None

        mask_u8 = np.asarray(mask, dtype=np.uint8)
        if mask_u8.max() <= 1:
            mask_u8 = mask_u8 * 255
        _, binary = cv2.threshold(mask_u8, LANE_MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

        warped, _, _ = perspective_transform(binary, np.float32(self._bev_cfg['src']))
        fit = lane_fit(warped)
        if fit is None:
            return {
                'stamp': msg.header.stamp,
                'mask': binary,
                'path_body': np.zeros((0, 2), dtype=float),
                'path_world': np.zeros((0, 2), dtype=float),
                'confidence': 0.0,
                'xte_m': None,
                'heading_error_rad': None,
                'path_heading_world_rad': None,
            }

        center_fit = 0.5 * (
            np.asarray(fit['left_fit'], dtype=float)
            + np.asarray(fit['right_fit'], dtype=float)
        )
        path_body = self._centerline_poly_to_body_points(center_fit)
        if path_body.shape[0] < LANE_MIN_POINTS:
            return None

        xte_m, heading_error_rad = self._compute_error(center_fit)

        c = float(np.cos(yaw_world))
        s = float(np.sin(yaw_world))
        rot = np.array([[c, -s], [s, c]], dtype=float)
        path_world = (rot @ path_body.T).T + np.asarray(pos_world[:2], dtype=float)[None, :]
        path_heading_body_rad = self._path_heading_from_poly(center_fit)
        path_heading_world_rad = float(yaw_world + path_heading_body_rad)

        confidence = float(min(1.0, path_body.shape[0] / float(T_horizon)))
        return {
            'stamp': msg.header.stamp,
            'mask': binary,
            'path_body': path_body,
            'path_world': path_world,
            'confidence': confidence,
            'xte_m': xte_m,
            'heading_error_rad': heading_error_rad,
            'path_heading_world_rad': path_heading_world_rad,
        }

    def _centerline_poly_to_body_points(self, poly_px: np.ndarray) -> np.ndarray:
        bev_height_m, bev_width_m = [float(v) for v in self._bev_cfg['bev_world_dim']]
        meters_per_pixel_y, meters_per_pixel_x = [float(v) for v in self._bev_cfg['unit_conversion_factor']]
        bev_height_px = int(round(bev_height_m / meters_per_pixel_y))
        bev_width_px  = int(round(bev_width_m  / meters_per_pixel_x))

        lookahead_m   = min(max(LANE_LOOKAHEAD_MIN_M, T_horizon * LANE_PROGRESS_STEP_M), LANE_LOOKAHEAD_MAX_M)
        forward_samples = np.linspace(0.5, lookahead_m, T_horizon)
        pixel_y = bev_height_px - (forward_samples / meters_per_pixel_y)
        pixel_x = np.polyval(poly_px, pixel_y)

        valid  = np.isfinite(pixel_x) & np.isfinite(pixel_y)
        valid &= (pixel_y >= 0.0) & (pixel_y < bev_height_px)
        valid &= (pixel_x >= 0.0) & (pixel_x < bev_width_px)
        if not np.any(valid):
            return np.zeros((0, 2), dtype=float)

        forward_m      = forward_samples[valid]
        lateral_right_m = (pixel_x[valid] - (0.5 * bev_width_px)) * meters_per_pixel_x
        return np.column_stack([forward_m, -lateral_right_m])

    def _compute_error(self, poly_px: np.ndarray) -> tuple[float, float]:
        bev_height_m, bev_width_m = [float(v) for v in self._bev_cfg['bev_world_dim']]
        meters_per_pixel_y, meters_per_pixel_x = [float(v) for v in self._bev_cfg['unit_conversion_factor']]
        scale = np.array([meters_per_pixel_x, meters_per_pixel_y], dtype=float)

        camera_m  = np.array([bev_width_m / 2.0, bev_height_m], dtype=float)
        camera_px = camera_m / scale
        closest_px = closest_point_on_polynomial(camera_px, poly_px)
        closest_m  = closest_px * scale

        delta_m = camera_m - closest_m
        xte_m   = float(np.linalg.norm(delta_m))
        if closest_m[0] < camera_m[0]:
            xte_m = -xte_m

        slope_px = float(np.polyder(poly_px)(closest_px[1]))
        heading_error_rad = float(np.arctan(slope_px * (meters_per_pixel_x / meters_per_pixel_y)))
        return xte_m, heading_error_rad

    def _path_heading_from_poly(self, poly_px: np.ndarray) -> float:
        _, heading_error_rad = self._compute_error(poly_px)
        return float(-heading_error_rad)


class LocalObstacleMap:
    """
    FIFO-based local obstacle map (Section III-B of paper).

    Stores recent depth/lidar points in the world XY frame and returns the
    nearest obstacle point to a query position.
    """

    def __init__(self, maxlen: int = FIFO_LEN):
        self._buf: list[np.ndarray] = []  # list of (2,) world-frame XY points
        self._maxlen = maxlen

    def update(self, points_xy: np.ndarray) -> None:
        """
        Add new world-frame obstacle points to the map.
        points_xy : (N, 2) array of XY positions [m]
        """
        if points_xy.shape[0] == 0:
            return
        for pt in points_xy:
            self._buf.append(pt.copy())
        if len(self._buf) > self._maxlen:
            self._buf = self._buf[-self._maxlen:]

    def nearest(self, x_ref: np.ndarray) -> np.ndarray | None:
        """
        Return the point in the map closest to x_ref (2-vector, world XY).
        Implements Eq.(11):  X_O^min = argmin_{X_O in S_obs} ||X_O - X_ref||_2
        Returns None if map is empty.
        """
        if not self._buf:
            return None
        pts   = np.array(self._buf)
        dists = norm(pts - np.asarray(x_ref[:2])[None, :], axis=1)
        return pts[np.argmin(dists)]

    def nearest_k(self, x_ref: np.ndarray, k: int) -> np.ndarray:
        """Return up to k nearest obstacle points to x_ref, sorted nearest-first."""
        if not self._buf or k <= 0:
            return np.zeros((0, 2), dtype=float)
        pts   = np.array(self._buf, dtype=float)
        dists = norm(pts - np.asarray(x_ref[:2], dtype=float)[None, :], axis=1)
        order = np.argsort(dists)[:int(k)]
        return pts[order]

    def clear(self):
        self._buf.clear()

    def snapshot(self, max_points: int | None = None) -> np.ndarray:
        """Return a copy of cached obstacle points for debug/visualisation."""
        if not self._buf:
            return np.zeros((0, 2), dtype=float)
        pts = np.array(self._buf, dtype=float)
        if max_points is not None and pts.shape[0] > max_points:
            pts = pts[-max_points:]
        return pts


class MPCDebugPlotter2D:
    """
    Optional async matplotlib plotter for the 2D MPC planner.

    Rendering is triggered by a lightweight ROS timer on the main thread.
    """

    def __init__(self,
                 enabled: bool = DEBUG_PLOT_ENABLED,
                 rate_hz: float = DEBUG_PLOT_RATE_HZ,
                 path_history_len: int = DEBUG_PATH_HISTORY_LEN):
        self._enabled = bool(enabled)
        self._latest: dict | None = None
        self._path_hist = deque(maxlen=path_history_len)
        self._plt = None
        self._fig = None
        self._ax_map = None
        self._ax_err = None
        self._ax_3d = None
        self._artists: dict[str, object] = {}
        self._time_hist = deque(maxlen=path_history_len)
        self._err_hist = deque(maxlen=path_history_len)
        self._yaw_err_hist = deque(maxlen=path_history_len)

        if not self._enabled:
            return

        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            self._plt = plt
            self._plt.ion()
            self._fig = self._plt.figure(figsize=(12, 9))
            gs = self._fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
            self._ax_map = self._fig.add_subplot(gs[0, 0])
            self._ax_err = self._fig.add_subplot(gs[1, 0])
            self._ax_3d = self._fig.add_subplot(gs[:, 1], projection='3d')
            try:
                self._fig.canvas.manager.set_window_title('MPC Local Map + Yaw Debug')
            except Exception:
                pass
            self._init_figure()
            self._plt.show(block=False)
        except Exception as e:
            self._enabled = False
            print(f'[MPCDebugPlotter2D] Disabled (matplotlib init failed): {e}')

    def _init_figure(self) -> None:
        ax = self._ax_map
        ax.set_title('Top view (world XY)')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        self._artists['obs']    = ax.scatter([], [], s=10, c='0.7', alpha=0.6, label='obstacles')
        self._artists['path'],  = ax.plot([], [], 'k-',  lw=1.5, alpha=0.8, label='actual path')
        self._artists['pred'],  = ax.plot([], [], 'b.-', lw=1.5, ms=5, label='MPC prediction')
        self._artists['cur'],   = ax.plot([], [], 'go',  ms=7, label='current')
        self._artists['goal'],  = ax.plot([], [], 'r*',  ms=12, label='goal')
        self._artists['near'],  = ax.plot([], [], 'mx',  ms=8, mew=2, label='nearest obs')
        self._artists['msg']    = ax.text(
            0.03, 0.97, '', transform=ax.transAxes, va='top', ha='left',
            fontsize=11, color='tab:green', fontweight='bold'
        )
        ax.legend(loc='best', fontsize=8)

        err_ax = self._ax_err
        err_ax.set_title('Tracking Error vs Time')
        err_ax.set_xlabel('Time [s]')
        err_ax.set_ylabel('Error')
        err_ax.grid(True, alpha=0.3)
        self._artists['err'], = err_ax.plot([], [], color='tab:orange', lw=2, label='position error [m]')
        self._artists['yaw_err'], = err_ax.plot([], [], color='tab:purple', lw=2, label='yaw error [rad]')
        err_ax.legend(loc='best', fontsize=8)

        ax3 = self._ax_3d
        ax3.set_title('3D view (x, y, yaw)')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_zlabel('Yaw [rad]')
        self._artists['path3d'], = ax3.plot([], [], [], 'k-', lw=1.5, alpha=0.8)
        self._artists['pred3d'], = ax3.plot([], [], [], 'b.-', lw=1.5, ms=5)
        self._artists['cur3d'], = ax3.plot([], [], [], 'go', ms=7)
        self._artists['goal3d'], = ax3.plot([], [], [], 'r*', ms=12)
        self._fig.tight_layout()

    def submit(self,
               x0: np.ndarray,
               x_ref: np.ndarray,
               X_opt: np.ndarray,
               x_min: np.ndarray | None,
               obs_pts: np.ndarray | None = None,
               t_now: float | None = None,
               yaw_ref: float | None = None,
               goal_reached: bool = False,
               dist_goal: float | None = None) -> None:
        if not self._enabled:
            return
        self._path_hist.append(np.array(x0[:2], dtype=float, copy=True))
        self._latest = {
            'x0':           np.array(x0[:2],    dtype=float, copy=True),
            'x_ref':        np.array(x_ref[:2],  dtype=float, copy=True),
            'x_pred':       np.array(X_opt[:, :2], dtype=float, copy=True),
            'yaw0':         float(x0[2]),
            'yaw_ref':      float(x_ref[2] if yaw_ref is None else yaw_ref),
            'yaw_pred':     np.array(X_opt[:, 2], dtype=float, copy=True),
            'x_min':        None if x_min is None else np.array(x_min[:2], dtype=float, copy=True),
            'obs':          None if obs_pts is None else np.array(obs_pts[:, :2], dtype=float, copy=True),
            'goal_reached': bool(goal_reached),
            'dist_goal':    None if dist_goal is None else float(dist_goal),
        }
        if t_now is not None and dist_goal is not None:
            self._time_hist.append(float(t_now))
            self._err_hist.append(float(dist_goal))
            self._yaw_err_hist.append(float(abs(self._wrap_pi(self._latest['yaw_ref'] - self._latest['yaw0']))))

    def draw_latest(self) -> None:
        if not self._enabled or self._latest is None:
            return
        try:
            self._draw(self._latest.copy(), np.array(self._path_hist, dtype=float) if self._path_hist else None)
        except Exception as e:
            print(f'[MPCDebugPlotter2D] Plot error: {e}')
            self._enabled = False

    def _draw(self, snap: dict, path_hist: np.ndarray | None) -> None:
        self._artists['pred'].set_data(snap['x_pred'][:, 0], snap['x_pred'][:, 1])
        self._artists['cur'].set_data([snap['x0'][0]], [snap['x0'][1]])
        self._artists['goal'].set_data([snap['x_ref'][0]], [snap['x_ref'][1]])
        if path_hist is not None and path_hist.size:
            self._artists['path'].set_data(path_hist[:, 0], path_hist[:, 1])
        if snap['x_min'] is not None:
            self._artists['near'].set_data([snap['x_min'][0]], [snap['x_min'][1]])
        else:
            self._artists['near'].set_data([], [])
        if snap['obs'] is not None and snap['obs'].size:
            self._artists['obs'].set_offsets(snap['obs'])
        else:
            self._artists['obs'].set_offsets(np.zeros((0, 2)))
        if self._time_hist:
            t = np.array(self._time_hist, dtype=float)
            self._artists['err'].set_data(t, np.array(self._err_hist, dtype=float))
            self._artists['yaw_err'].set_data(t, np.array(self._yaw_err_hist, dtype=float))
            self._ax_err.relim()
            self._ax_err.autoscale_view()
        if path_hist is not None and path_hist.size:
            z_hist = np.zeros(path_hist.shape[0], dtype=float)
            if len(self._yaw_err_hist) == path_hist.shape[0]:
                z_hist = np.array([snap['yaw0']] * path_hist.shape[0], dtype=float)
            self._artists['path3d'].set_data(path_hist[:, 0], path_hist[:, 1])
            self._artists['path3d'].set_3d_properties(z_hist)
        self._artists['pred3d'].set_data(snap['x_pred'][:, 0], snap['x_pred'][:, 1])
        self._artists['pred3d'].set_3d_properties(snap['yaw_pred'])
        self._artists['cur3d'].set_data([snap['x0'][0]], [snap['x0'][1]])
        self._artists['cur3d'].set_3d_properties([snap['yaw0']])
        self._artists['goal3d'].set_data([snap['x_ref'][0]], [snap['x_ref'][1]])
        self._artists['goal3d'].set_3d_properties([snap['yaw_ref']])
        if snap['goal_reached']:
            d = snap['dist_goal']
            self._artists['msg'].set_text(
                'GOAL REACHED' if d is None else f'GOAL REACHED (err={d:.2f} m)'
            )
        else:
            self._artists['msg'].set_text('')
        self._ax_map.relim()
        self._ax_map.autoscale_view()
        self._ax_3d.relim()
        self._ax_3d.autoscale_view()
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def close(self) -> None:
        if self._enabled and self._plt is not None and self._fig is not None:
            try:
                self._plt.close(self._fig)
            except Exception:
                pass


class MPCPlanner:
    """
    MATLAB-style SCP planner (2D):
      - penalty objective phi (real) and phi_hat (convexified),
      - linearised obstacle constraints around previous trajectory,
      - trust-region accept/reject update using delta and delta_hat.

    State  : [x, y, yaw, vx, vy]
    Control: [ax, ay, yaw_rate]
    """

    def __init__(self, backend: str = DEFAULT_MPC_SOLVER_BACKEND):
        requested = str(backend).strip().lower()
        if requested == 'auto':
            if native_osqp_available():
                resolved = 'cpp_osqp'
            elif native_backend_available():
                resolved = 'cpp_subgradient'
            else:
                resolved = 'python'
        elif requested == 'cpp_osqp':
            if not native_osqp_available():
                raise RuntimeError('mpc_solver_backend=cpp_osqp requested but not available')
            resolved = 'cpp_osqp'
        elif requested == 'cpp_subgradient':
            if not native_backend_available():
                raise RuntimeError('mpc_solver_backend=cpp_subgradient requested but not available')
            resolved = 'cpp_subgradient'
        elif requested == 'python':
            resolved = 'python'
        else:
            raise ValueError(
                f'Unsupported mpc_solver_backend={backend!r}; '
                "expected 'python', 'cpp_osqp', 'cpp_subgradient', or 'auto'"
            )
        self.backend_name = resolved
        self.last_solver_status = 'not_run'

    @staticmethod
    def _native_solver_config() -> dict:
        return {
            'Ts': Ts, 'T_horizon': T_horizon,
            'V_MAX': V_MAX, 'A_MAX': A_MAX,
            'SCP_MAX_OUTER_ITERS': SCP_MAX_OUTER_ITERS,
            'SCP_INNER_ITERS': SCP_INNER_ITERS,
            'SCP_EPS': SCP_EPS, 'SCP_LR': SCP_LR,
            'SCP_LAMBDA': SCP_LAMBDA, 'SCP_ALPHA': SCP_ALPHA,
            'SCP_BETA_GROW': SCP_BETA_GROW, 'SCP_BETA_SHRINK': SCP_BETA_SHRINK,
            'TRUST_POS0': TRUST_POS0, 'TRUST_U0': TRUST_U0,
            'MIN_CLEARANCE': MIN_CLEARANCE,
        }

    @staticmethod
    def _rollout(x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Rollout dynamics. State: [x, y, yaw, vx, vy]. Control: [ax, ay, yaw_rate]."""
        X = np.zeros((T_horizon + 1, 5), dtype=float)
        X[0] = x0
        for k in range(T_horizon):
            X[k + 1, :2] = X[k, :2] + Ts * X[k, 3:5]
            X[k + 1, 2] = X[k, 2] + Ts * U[k, 2]
            X[k + 1, 3:5] = X[k, 3:5] + Ts * U[k, :2]
        return X

    @staticmethod
    def _project_speed_limits(U: np.ndarray) -> None:
        np.clip(U, -U_LIMS[None, :], U_LIMS[None, :], out=U)

    @staticmethod
    def _hinge(v: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, v)

    @staticmethod
    def _sum_abs(v: np.ndarray) -> float:
        return float(np.sum(np.abs(v)))

    @staticmethod
    def _normalize_x_obs(x_obs: np.ndarray | None) -> np.ndarray | None:
        if x_obs is None:
            return None
        arr = np.asarray(x_obs, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError(f'Expected x_obs shape (2,), got {arr.shape}')
            return arr[None, :]
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        raise ValueError(f'Expected x_obs shape (K,2), got {arr.shape}')

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _phi_real(
        self,
        X: np.ndarray,
        U: np.ndarray,
        x_ref: np.ndarray,
        x_obs: np.ndarray | None,
        x_ref_path: np.ndarray | None = None,
    ) -> float:
        """Real (non-convex) penalty objective phi(X, U)."""
        x_obs = self._normalize_x_obs(x_obs)
        # Terminal tracking (L1 in position and velocity)
        eqns  = self._sum_abs(X[-1, :2] - x_ref[:2])
        eqns += YAW_ERROR_WEIGHT * abs(self._wrap_pi(float(X[-1, 2] - x_ref[2])))
        eqns += self._sum_abs(X[-1, 3:5] - x_ref[3:5])
        # Control and velocity bound violations
        ineq_u = np.sum(self._hinge(np.abs(U) - U_LIMS[None, :]))
        ineq_v = np.sum(self._hinge(np.abs(X[1:, 3:5]) - VEL_LIMS[None, :]))
        # Obstacle clearance
        ineq_obs = 0.0
        if x_obs is not None:
            d = X[1:, None, :2] - x_obs[None, :, :]
            dist = np.linalg.norm(d, axis=2)
            ineq_obs = float(np.sum(self._hinge(MIN_CLEARANCE - dist)))
        # Control effort
        obj = float(Ts * (np.sum(U[:, :2] * U[:, :2]) + YAW_RATE_WEIGHT * np.sum(U[:, 2] * U[:, 2])))
        # Optional lane-centring term
        if x_ref_path is not None and x_ref_path.size:
            lane_err = X[1:, :2] - np.asarray(x_ref_path, dtype=float)
            obj += float(LANE_CENTER_WEIGHT * Ts * np.sum(lane_err * lane_err))
        yaw_path = np.arctan2(X[1:, 4], X[1:, 3])
        yaw_err = np.arctan2(np.sin(yaw_path - X[1:, 2]), np.cos(yaw_path - X[1:, 2]))
        obj += float(YAW_ERROR_WEIGHT * Ts * np.sum(yaw_err * yaw_err))
        return obj + SCP_LAMBDA * (eqns + float(ineq_u) + float(ineq_v) + ineq_obs)

    def _phi_hat(
        self,
        X: np.ndarray,
        U: np.ndarray,
        x_ref: np.ndarray,
        x_obs: np.ndarray | None,
        X_now: np.ndarray,
        U_now: np.ndarray,
        l_pos: float,
        l_u: float,
        x_ref_path: np.ndarray | None = None,
    ) -> float:
        """Convexified objective phi_hat(X, U) around (X_now, U_now)."""
        x_obs = self._normalize_x_obs(x_obs)
        eqns  = self._sum_abs(X[-1, :2] - x_ref[:2])
        eqns += YAW_ERROR_WEIGHT * abs(self._wrap_pi(float(X[-1, 2] - x_ref[2])))
        eqns += self._sum_abs(X[-1, 3:5] - x_ref[3:5])
        ineq_u    = np.sum(self._hinge(np.abs(U) - U_LIMS[None, :]))
        ineq_v    = np.sum(self._hinge(np.abs(X[1:, 3:5]) - VEL_LIMS[None, :]))
        ineq_tr_u = np.sum(self._hinge(np.abs(U - U_now) - l_u))
        ineq_tr_x = np.sum(self._hinge(np.abs(X[1:, :2] - X_now[1:, :2]) - l_pos))
        # Linearised obstacle term
        ineq_obs = 0.0
        if x_obs is not None:
            d_now      = X_now[1:, None, :2] - x_obs[None, :, :]
            d_now_norm = np.linalg.norm(d_now, axis=2)
            d          = X[1:, None, :2] - x_obs[None, :, :]
            lin_obs    = MIN_CLEARANCE * d_now_norm - np.sum(d_now * d, axis=2)
            ineq_obs   = float(np.sum(self._hinge(lin_obs)))
        obj = float(Ts * (np.sum(U[:, :2] * U[:, :2]) + YAW_RATE_WEIGHT * np.sum(U[:, 2] * U[:, 2])))
        if x_ref_path is not None and x_ref_path.size:
            lane_err = X[1:, :2] - np.asarray(x_ref_path, dtype=float)
            obj += float(LANE_CENTER_WEIGHT * Ts * np.sum(lane_err * lane_err))
        yaw_path = np.arctan2(X[1:, 4], X[1:, 3])
        yaw_err = np.arctan2(np.sin(yaw_path - X[1:, 2]), np.cos(yaw_path - X[1:, 2]))
        obj += float(YAW_ERROR_WEIGHT * Ts * np.sum(yaw_err * yaw_err))
        ineq = float(ineq_u + ineq_v + ineq_tr_u + ineq_tr_x) + ineq_obs
        return obj + SCP_LAMBDA * (eqns + ineq)

    def _phi_hat_gradient(
        self,
        x0: np.ndarray,
        U: np.ndarray,
        x_ref: np.ndarray,
        x_obs: np.ndarray | None,
        X_now: np.ndarray,
        U_now: np.ndarray,
        l_pos: float,
        l_u: float,
        x_ref_path: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Analytical subgradient of phi_hat wrt U via chain rule through rollout.
        Returns (grad, X(U)).
        """
        x_obs = self._normalize_x_obs(x_obs)
        X    = self._rollout(x0, U)
        grad = np.zeros_like(U)
        grad[:, :2] = 2.0 * Ts * U[:, :2]
        grad[:, 2] = 2.0 * Ts * YAW_RATE_WEIGHT * U[:, 2]

        # Lane-centring gradient
        if x_ref_path is not None and np.size(x_ref_path):
            x_ref_path = np.asarray(x_ref_path, dtype=float)
            for k in range(1, T_horizon + 1):
                err_xy = X[k, :2] - x_ref_path[k - 1]
                for j in range(k - 1):
                    w = Ts * Ts * float(k - 1 - j)
                    grad[j] += 2.0 * LANE_CENTER_WEIGHT * Ts * w * err_xy

        # Terminal position L1
        sign_pos = np.sign(X[-1, :2] - x_ref[:2])
        for j in range(T_horizon - 1):
            w = Ts * Ts * float(T_horizon - 1 - j)
            grad[j] += SCP_LAMBDA * w * sign_pos

        # Terminal velocity L1
        yaw_terminal_err = self._wrap_pi(float(X[-1, 2] - x_ref[2]))
        grad[:, 2] += SCP_LAMBDA * YAW_ERROR_WEIGHT * Ts * np.sign(yaw_terminal_err)

        sign_vel = np.sign(X[-1, 3:5] - x_ref[3:5])
        grad[:, :2] += SCP_LAMBDA * Ts * sign_vel[None, :]

        # Control bound hinge
        viol_u = np.abs(U) - U_LIMS[None, :]
        grad  += SCP_LAMBDA * ((viol_u > 0.0) * np.sign(U))

        # Velocity bound hinge — propagate back through dynamics
        for k in range(1, T_horizon + 1):
            v_k    = X[k, 3:5]
            mask_v = (np.abs(v_k) - VEL_LIMS) > 0.0
            if not np.any(mask_v):
                continue
            g_v = np.zeros(2, dtype=float)
            g_v[mask_v] = np.sign(v_k[mask_v])
            for j in range(k):
                grad[j, :2] += SCP_LAMBDA * Ts * g_v

        # Control trust-region hinge
        viol_tr_u = np.abs(U - U_now) - l_u
        grad += SCP_LAMBDA * ((viol_tr_u > 0.0) * np.sign(U - U_now))

        # State trust-region hinge — propagate back through double integrator
        for k in range(1, T_horizon + 1):
            dx   = X[k, :2] - X_now[k, :2]
            mask = (np.abs(dx) - l_pos) > 0.0
            if not np.any(mask):
                continue
            g_x = np.zeros(2, dtype=float)
            g_x[mask] = np.sign(dx[mask])
            for j in range(k - 1):
                w = Ts * Ts * float(k - 1 - j)
                grad[j] += SCP_LAMBDA * w * g_x

        # Convexified obstacle constraint gradient
        if x_obs is not None:
            d_now      = X_now[1:, None, :2] - x_obs[None, :, :]
            d_now_norm = np.linalg.norm(d_now, axis=2)
            d          = X[1:, None, :2] - x_obs[None, :, :]
            lin_obs    = MIN_CLEARANCE * d_now_norm - np.sum(d_now * d, axis=2)
            for k in range(T_horizon):
                for obs_idx in range(x_obs.shape[0]):
                    if lin_obs[k, obs_idx] <= 0.0:
                        continue
                    g_x = -d_now[k, obs_idx]
                    for j in range(k):
                        w = Ts * Ts * float(k - j)
                        grad[j] += SCP_LAMBDA * w * g_x

        return grad, X

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        x_obs: np.ndarray | None,
        U_warm: np.ndarray,
        x_ref_path: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run SCP optimisation.

        Parameters
        ----------
        x0         : (5,) current state [x, y, yaw, vx, vy]
        x_ref      : (5,) terminal reference [x_goal, y_goal, yaw_ref, vx_ref, vy_ref]
        x_obs      : (K, 2) nearest obstacle XY positions, or None
        U_warm     : (T, 3) warm-start control sequence [ax, ay, yaw_rate]
        x_ref_path : (T, 2) per-step lane-centre reference, or None

        Returns
        -------
        U_opt : (T, 3) optimised controls
        X_opt : (T+1, 5) optimised state trajectory
        """
        # Native backends still implement the old 4-state model, so the yaw-augmented
        # planner always runs through the Python path for now.
        if self.backend_name != 'python':
            self.last_solver_status = f'{self.backend_name}|yaw_python_fallback'

        # Python SCP loop
        U_now = U_warm.copy()
        self._project_speed_limits(U_now)
        X_now = self._rollout(x0, U_now)
        l_pos, l_u = TRUST_POS0, TRUST_U0

        for _ in range(SCP_MAX_OUTER_ITERS):
            U_cand = self._solve_convex_subproblem(
                x0=x0, x_ref=x_ref, x_obs=x_obs,
                X_now=X_now, U_now=U_now, l_pos=l_pos, l_u=l_u,
                x_ref_path=x_ref_path,
            )
            if U_cand is None:
                self.last_solver_status = f'{self.last_solver_status}|fallback_subgradient'
                U_cand = U_now.copy()
                for _ in range(SCP_INNER_ITERS):
                    grad, _ = self._phi_hat_gradient(
                        x0=x0, U=U_cand, x_ref=x_ref, x_obs=x_obs,
                        X_now=X_now, U_now=U_now, l_pos=l_pos, l_u=l_u,
                        x_ref_path=x_ref_path,
                    )
                    U_cand -= SCP_LR * grad
                    self._project_speed_limits(U_cand)

            X_cand = self._rollout(x0, U_cand)

            phi_now     = self._phi_real(X_now,  U_now,  x_ref, x_obs, x_ref_path)
            phi_hat_new = self._phi_hat(X_cand, U_cand, x_ref, x_obs, X_now, U_now, l_pos, l_u, x_ref_path)
            phi_new     = self._phi_real(X_cand, U_cand, x_ref, x_obs, x_ref_path)

            delta_hat = phi_now - phi_hat_new
            delta     = phi_now - phi_new

            if delta > SCP_ALPHA * delta_hat:
                l_pos *= SCP_BETA_GROW
                l_u   *= SCP_BETA_GROW
                step_norm = float(np.max(np.abs(X_cand - X_now)))
                X_now, U_now = X_cand, U_cand
                if step_norm < SCP_EPS:
                    break
            else:
                l_pos *= SCP_BETA_SHRINK
                l_u   *= SCP_BETA_SHRINK

        return U_now, X_now

    def _solve_convex_subproblem(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        x_obs: np.ndarray | None,
        X_now: np.ndarray,
        U_now: np.ndarray,
        l_pos: float,
        l_u: float,
        x_ref_path: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """
        Solve one SCP convex subproblem via CVXPY (OSQP/SCS).
        All bounds are soft penalties — no explicit inequality constraints.
        Returns U or None on failure.
        """
        if cp is None:
            self.last_solver_status = 'no_cvxpy'
            return None
        x_obs = self._normalize_x_obs(x_obs)

        U     = cp.Variable((T_horizon, 2))
        X_pos = cp.Variable((T_horizon + 1, 2))
        X_vel = cp.Variable((T_horizon + 1, 2))

        X_yaw = cp.Variable(T_horizon + 1)

        constraints = [X_pos[0] == x0[:2], X_yaw[0] == x0[2], X_vel[0] == x0[3:5]]
        for k in range(T_horizon):
            constraints += [
                X_pos[k + 1] == X_pos[k] + Ts * X_vel[k],
                X_yaw[k + 1] == X_yaw[k] + Ts * U[k, 2],
                X_vel[k + 1] == X_vel[k] + Ts * U[k, :2],
            ]

        eq_term   = (
            cp.norm1(X_pos[-1] - x_ref[:2])
            + YAW_ERROR_WEIGHT * cp.abs(X_yaw[-1] - x_ref[2])
            + cp.norm1(X_vel[-1] - x_ref[3:5])
        )
        ineq_u    = cp.sum(cp.pos(cp.abs(U)          - U_LIMS[None, :]))
        ineq_v    = cp.sum(cp.pos(cp.abs(X_vel[1:])  - VEL_LIMS[None, :]))
        ineq_tr_u = cp.sum(cp.pos(cp.abs(U - U_now)  - l_u))
        ineq_tr_x = cp.sum(cp.pos(cp.abs(X_pos[1:] - X_now[1:, :2]) - l_pos))

        ineq_obs = 0.0
        if x_obs is not None:
            for obs_idx in range(x_obs.shape[0]):
                obs_xy     = x_obs[obs_idx]
                d_now      = X_now[1:, :2] - obs_xy[None, :]
                d_now_norm = np.linalg.norm(d_now, axis=1)
                lin_obs    = (
                    MIN_CLEARANCE * d_now_norm
                    - cp.sum(cp.multiply(d_now, X_pos[1:] - obs_xy[None, :]), axis=1)
                )
                ineq_obs += cp.sum(cp.pos(lin_obs))

        lane_term = (
            0.0 if x_ref_path is None or not np.size(x_ref_path)
            else LANE_CENTER_WEIGHT * Ts
                 * cp.sum_squares(X_pos[1:] - np.asarray(x_ref_path, dtype=float))
        )
        objective = cp.Minimize(
            Ts * cp.sum_squares(U[:, :2])
            + YAW_RATE_WEIGHT * Ts * cp.sum_squares(U[:, 2])
            + lane_term
            + SCP_LAMBDA * (eq_term + ineq_u + ineq_v + ineq_tr_u + ineq_tr_x + ineq_obs)
        )
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            try:
                problem.solve(solver=cp.SCS, warm_start=True, verbose=False)
            except Exception:
                self.last_solver_status = 'solver_error'
                return None

        self.last_solver_status = str(problem.status)
        if problem.status not in ('optimal', 'optimal_inaccurate') or U.value is None:
            return None

        U_sol = np.asarray(U.value, dtype=float)
        self._project_speed_limits(U_sol)
        return U_sol


class DepthToObstacles:
    """
    Converts a depth image (32FC1, metres) to 2D obstacle points in the
    world XY frame using the camera-to-body and body-to-world transforms.
    """

    def __init__(self,
                 hfov_deg: float = DEPTH_HFOV,
                 vfov_deg: float = DEPTH_VFOV,
                 n_sample: int = 500):
        self.hfov     = np.deg2rad(hfov_deg)
        self.vfov     = np.deg2rad(vfov_deg)
        self.n_sample = n_sample
        # Camera (z-forward, x-right, y-down) → body FRD (x-forward, y-right, z-down)
        self.rotation_body_camera = np.array([
            [0.0,  0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)
        # Camera position in body frame [m]
        self.translation_body_camera_m = np.array([0.13233, 0.0, 0.26078], dtype=float)

    def depth_to_body_frame(self, depth_img: np.ndarray) -> np.ndarray:
        """Back-project depth pixels to body-frame 3D points."""
        H, W = depth_img.shape
        fy = (H / 2.0) / np.tan(self.vfov / 2.0)
        fx = (W / 2.0) / np.tan(self.hfov / 2.0)
        cx, cy = W / 2.0, H / 2.0

        total = H * W
        idx   = np.random.choice(total, min(self.n_sample, total), replace=False)
        rows, cols = np.unravel_index(idx, (H, W))
        depths = depth_img[rows, cols]

        valid  = (depths > DEPTH_MIN) & (depths < DEPTH_MAX) & np.isfinite(depths)
        depths, rows, cols = depths[valid], rows[valid], cols[valid]
        if len(depths) == 0:
            return np.zeros((0, 3))

        Xc = (cols - cx) / fx * depths
        Yc = (rows - cy) / fy * depths
        Zc = depths
        pts_cam = np.column_stack([Xc, Yc, Zc])
        return (self.rotation_body_camera @ pts_cam.T).T + self.translation_body_camera_m

    def filter_body_points(self, pts_body: np.ndarray) -> np.ndarray:
        """Keep only points in a sensible forward arc; discard floor/ceiling hits."""
        if pts_body.shape[0] == 0:
            return np.zeros((0, 3))
        pts     = np.asarray(pts_body, dtype=float)
        forward = pts[:, 0]
        lateral = pts[:, 1]
        mask    = np.isfinite(pts).all(axis=1)
        mask   &= (forward >= DEPTH_FILTER_MIN_FORWARD_M) & (forward <= DEPTH_FILTER_MAX_FORWARD_M)
        mask   &= np.abs(lateral) <= DEPTH_FILTER_MAX_LATERAL_M
        return pts[mask] if pts[mask].shape[0] > 0 else np.zeros((0, 3))

    def body_to_world_xy(
        self,
        pts_body: np.ndarray,
        yaw: float,
        pos_xy: np.ndarray,
    ) -> np.ndarray:
        """
        Project body-frame 3D points to world XY [m].

        Only the forward (x) and lateral (y) body-frame components are used;
        the vertical component is discarded so the obstacle map is 2D.

        yaw    : vehicle heading [rad] in world frame
        pos_xy : (2,) vehicle XY position in world frame [m]
        Returns: (N, 2) obstacle XY positions in world frame
        """
        if pts_body.shape[0] == 0:
            return np.zeros((0, 2))
        c   = float(np.cos(yaw))
        s   = float(np.sin(yaw))
        R2  = np.array([[c, -s], [s, c]], dtype=float)  # body → world (XY only)
        xy_body = pts_body[:, :2]                        # forward, lateral
        return (R2 @ xy_body.T).T + np.asarray(pos_xy, dtype=float)[None, :]


# ─────────────────────────────────────────────────────────────────────────────
#  ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────

class MPCUAVNode(Node):
    """
    ROS2 node: GEM perception + 2D MPC trajectory planning.

    State  : [x, y, yaw, vx, vy]   (world frame, metres / m·s⁻¹)
    Control: [ax, ay, yaw_rate]     (world frame, m·s⁻² and rad/s)
    Output : AckermannDrive    (speed + steering angle)
    """

    def __init__(self):
        super().__init__('mpc_gem_2d_node')

        # ── parameters ──────────────────────────────────────────────────────
        self.declare_parameter('enable_depth_camera',         DEFAULT_ENABLE_DEPTH_CAMERA)
        self.declare_parameter('enable_lidar',                DEFAULT_ENABLE_LIDAR)
        self.declare_parameter('mpc_solver_backend',          DEFAULT_MPC_SOLVER_BACKEND)
        self.declare_parameter('depth_topic',                 DEFAULT_GEM_DEPTH_TOPIC)
        self.declare_parameter('lidar_topic',                 DEFAULT_GEM_LIDAR_TOPIC)
        self.declare_parameter('odom_topic',                  DEFAULT_GEM_ODOM_TOPIC)
        self.declare_parameter('ackermann_topic',             DEFAULT_GEM_ACKERMANN_TOPIC)
        self.declare_parameter('gem_max_speed_mps',           DEFAULT_GEM_MAX_SPEED_MPS)
        self.declare_parameter('gem_wheelbase_m',             DEFAULT_GEM_WHEELBASE_M)
        self.declare_parameter('gem_max_steering_rad',        DEFAULT_GEM_MAX_STEERING_RAD)
        self.declare_parameter('gem_steering_kp',             DEFAULT_GEM_STEERING_KP)
        self.declare_parameter('enable_lane_tracking',        DEFAULT_ENABLE_LANE_TRACKING)
        self.declare_parameter('rgb_topic',                   DEFAULT_GEM_RGB_TOPIC)
        self.declare_parameter('lane_model_path',             DEFAULT_LANE_MODEL_PATH)
        self.declare_parameter('lane_bev_config_path',        DEFAULT_BEV_CONFIG_PATH)
        self.declare_parameter('disable_obstacle_constraints', DEFAULT_OBSTACLE_CONSTRAINTS_DISABLED)
        self.declare_parameter('goal_reached_threshold_m',    GOAL_REACHED_THRESHOLD_M)
        self.declare_parameter('enable_delay_compensation',   ENABLE_DELAY_COMPENSATION)
        self.declare_parameter('delay_compensation_sec',      DELAY_COMPENSATION_SEC)
        self.declare_parameter('goal_x', 32.91)
        self.declare_parameter('goal_y',  0.0)

        self._enable_depth_camera  = bool(self.get_parameter('enable_depth_camera').value)
        self._enable_lidar         = bool(self.get_parameter('enable_lidar').value)
        self._enable_lane_tracking = bool(self.get_parameter('enable_lane_tracking').value)
        self._mpc_solver_backend   = str(self.get_parameter('mpc_solver_backend').value)
        self._gem_max_speed_mps    = max(0.0,  float(self.get_parameter('gem_max_speed_mps').value))
        self._gem_wheelbase_m      = max(0.1,  float(self.get_parameter('gem_wheelbase_m').value))
        self._gem_max_steering_rad = max(0.01, float(self.get_parameter('gem_max_steering_rad').value))
        self._gem_steering_kp      = max(0.0,  float(self.get_parameter('gem_steering_kp').value))

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value,
                                 self._odom_cb, sensor_qos)
        if self._enable_depth_camera:
            self.create_subscription(Image, self.get_parameter('depth_topic').value,
                                     self._depth_cb, sensor_qos)
        if self._enable_lidar:
            self.create_subscription(PointCloud2, self.get_parameter('lidar_topic').value,
                                     self._lidar_cb, sensor_qos)
        if self._enable_lane_tracking:
            self.create_subscription(Image, self.get_parameter('rgb_topic').value,
                                     self._rgb_cb, sensor_qos)

        # ── publishers ───────────────────────────────────────────────────────
        self._ackermann_pub = self.create_publisher(
            AckermannDrive, self.get_parameter('ackermann_topic').value, 10)
        self._mpc_label_path_pub = self.create_publisher(Path, '/mpc/trajectory_sequence', 10)
        self._mpc_local_pred_pub = self.create_publisher(Path, LOCAL_PREDICTION_SEQUENCE_TOPIC, 10)
        self._lane_ref_pub       = self.create_publisher(Path, LANE_REFERENCE_TOPIC, 10)
        self._lane_mask_pub      = self.create_publisher(Image, LANE_MASK_TOPIC, 10)
        self._mpc_traj_pub       = self.create_publisher(PoseStamped, '/mpc/trajectory', 10)
        self._fresh_wp_pub       = self.create_publisher(PoseStamped, FRESH_COMMITTED_WAYPOINT_TOPIC, 10)
        self._exec_wp_pub        = self.create_publisher(PoseStamped, EXECUTED_COMMITTED_WAYPOINT_TOPIC, 10)

        # ── MPC components ───────────────────────────────────────────────────
        self._mpc           = MPCPlanner(backend=self._mpc_solver_backend)
        self._obs_map       = LocalObstacleMap(FIFO_LEN)
        self._depth2obs     = DepthToObstacles()
        self._bridge        = CvBridge()
        self._lane_detector = LaneCenterDetector(self, self._bridge) if self._enable_lane_tracking else None
        self._debug_plotter = MPCDebugPlotter2D()

        # ── state ────────────────────────────────────────────────────────────
        self._solver_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='mpc_2d')
        self._solver_future: Future | None = None

        # [x, y, yaw, vx, vy] — 2D pose and velocity with heading in MPC state
        self._x_state = np.zeros(5, dtype=float)
        self._yaw: float = 0.0

        # warm-start control sequence [ax, ay, yaw_rate]
        self._U_warm = np.zeros((T_horizon, 3), dtype=float)

        # navigation goal [x, y] in world frame
        self._goal_xy = np.array([20.0, 0.0], dtype=float)
        self._goal_generation: int = 0
        self._goal_reached_announced: bool = False

        self._tick_count: int = 0
        self._warned_obstacle_constraints_disabled: bool = False
        self._warned_delay_compensation_enabled: bool = False

        self._last_solve_duration_sec: float | None = None
        self._last_slow_solver_warn_monotonic: float = 0.0
        self._last_setpoint_pub_monotonic: float | None = None
        self._setpoint_pub_dt: deque = deque(maxlen=RATE_WINDOW_LEN)

        # latest commands (re-published at fixed rate independent of solve time)
        self._latest_vel_sp = np.zeros(3, dtype=float)   # [vx_cmd, vy_cmd, yaw_rate_cmd]
        self._last_u_cmd    = np.zeros(3, dtype=float)   # used for delay compensation

        # lane tracking state
        self._lane_path_world: np.ndarray = np.zeros((0, 2), dtype=float)
        self._lane_confidence: float      = 0.0
        self._lane_stamp_sec: float | None = None
        self._lane_xte_m: float | None    = None
        self._lane_heading_error_rad: float | None = None

        # ── timers ───────────────────────────────────────────────────────────
        self.create_timer(Ts, self._mpc_step)
        self.create_timer(PUBLISH_PERIOD_SEC, self._republish_latest_setpoint)
        self.create_timer(1.0 / max(DEBUG_PLOT_RATE_HZ, 0.2), self._debug_plot_step)

        if not self._enable_depth_camera:
            self.get_logger().warn(
                'enable_depth_camera=False: depth subscription is disabled. '
                'Restart with --ros-args -p enable_depth_camera:=true to restore.'
            )
        self.get_logger().info(
            'MPC 2D node ready. '
            f'backend={self._mpc.backend_name} '
            f'odom_topic={self.get_parameter("odom_topic").value} '
            f'ackermann_topic={self.get_parameter("ackermann_topic").value} '
            f'rgb_topic={self.get_parameter("rgb_topic").value} '
            f'depth_topic={self.get_parameter("depth_topic").value} '
            f'lidar_topic={self.get_parameter("lidar_topic").value} '
            f'enable_lane_tracking={self._enable_lane_tracking} '
            f'enable_depth_camera={self._enable_depth_camera} '
            f'enable_lidar={self._enable_lidar}'
        )
        self.get_logger().info(
            f'lane_model_path={self.get_parameter("lane_model_path").value} '
            f'lane_bev_config_path={self.get_parameter("lane_bev_config_path").value}'
        )

    # ── odometry callback ────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        """Update 2D vehicle state from /odom."""
        q = msg.pose.pose.orientation
        self._yaw = self._quat_to_yaw(float(q.w), float(q.x), float(q.y), float(q.z))
        pos   = msg.pose.pose.position
        twist = msg.twist.twist.linear
        self._x_state = np.array(
            [float(pos.x), float(pos.y), self._yaw, float(twist.x), float(twist.y)],
            dtype=float,
        )

    # ── sensor callbacks ─────────────────────────────────────────────────────

    def _depth_cb(self, msg: Image) -> None:
        """Depth image → 2D world-frame obstacle points → obstacle map."""
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f'Depth decode error: {e}')
            return
        pts_body = self._depth2obs.depth_to_body_frame(depth)
        pts_body = self._depth2obs.filter_body_points(pts_body)
        pts_xy   = self._depth2obs.body_to_world_xy(pts_body, self._yaw, self._x_state[:2])
        self._obs_map.update(pts_xy)

    def _lidar_cb(self, msg: PointCloud2) -> None:
        """Lidar point cloud → 2D world-frame obstacle points → obstacle map."""
        if point_cloud2 is None:
            self.get_logger().warn('sensor_msgs_py unavailable; lidar disabled.')
            return
        pts_body = []
        try:
            for x, y, z in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                fwd   = float(x)
                right = -float(y)   # FLU → FRD
                if not np.isfinite([fwd, right]).all():
                    continue
                if fwd < DEPTH_FILTER_MIN_FORWARD_M or fwd > DEPTH_FILTER_MAX_FORWARD_M:
                    continue
                if abs(right) > DEPTH_FILTER_MAX_LATERAL_M:
                    continue
                pts_body.append((fwd, right, 0.0))   # z=0 placeholder
                if len(pts_body) >= self._depth2obs.n_sample:
                    break
        except Exception as exc:
            self.get_logger().warn(f'Lidar decode error: {exc}')
            return
        if not pts_body:
            return
        pts = np.asarray(pts_body, dtype=float)
        pts_xy = self._depth2obs.body_to_world_xy(pts, self._yaw, self._x_state[:2])
        self._obs_map.update(pts_xy)

    def _rgb_cb(self, msg: Image) -> None:
        """RGB image → lane segmentation → lane centre reference."""
        if self._lane_detector is None or not self._lane_detector.is_enabled():
            return
        result = self._lane_detector.process(msg, self._yaw, self._x_state[:2])
        if result is None:
            return
        self._lane_path_world        = np.asarray(result['path_world'], dtype=float)
        self._lane_confidence        = float(result['confidence'])
        self._lane_xte_m             = None if result['xte_m'] is None else float(result['xte_m'])
        self._lane_heading_error_rad = None if result['heading_error_rad'] is None else float(result['heading_error_rad'])
        self._lane_stamp_sec         = float(self.get_clock().now().nanoseconds) * 1e-9
        self._publish_lane_reference(self._lane_path_world, result['stamp'])
        try:
            mask_msg = self._bridge.cv2_to_imgmsg(
                np.asarray(result['mask'], dtype=np.uint8), encoding='mono8')
            mask_msg.header = msg.header
            self._lane_mask_pub.publish(mask_msg)
        except Exception:
            pass

    # ── MPC pipeline ─────────────────────────────────────────────────────────

    def _mpc_step(self) -> None:
        self._tick_count += 1
        if self._solver_future is not None and self._solver_future.done():
            self._consume_solver_result()
        if self._solver_future is not None:
            return
        req = self._build_solver_request()
        self._solver_future = self._solver_executor.submit(self._solve_mpc_request, req)

    def _build_solver_request(self) -> dict:
        """Snapshot current state for background MPC solve."""
        now = self.get_clock().now()
        x0  = self._x_state.copy()

        # Optional delay compensation: forward-predict state by τ seconds
        if bool(self.get_parameter('enable_delay_compensation').value):
            tau = max(0.0, float(self.get_parameter('delay_compensation_sec').value))
            if tau > 0.0:
                x0[:2] += tau * self._last_u_cmd[:2]      # position
                x0[2] = self._wrap_pi(x0[2] + tau * self._last_u_cmd[2])
                x0[3:5] = self._last_u_cmd[:2]            # velocity
            if not self._warned_delay_compensation_enabled:
                self.get_logger().warn('Delay compensation is ON.')
                self._warned_delay_compensation_enabled = True
        elif self._warned_delay_compensation_enabled:
            self._warned_delay_compensation_enabled = False

        goal_threshold_m = max(0.0, float(self.get_parameter('goal_reached_threshold_m').value))
        dist_goal = float(norm(x0[:2] - self._goal_xy))
        goal_reached = dist_goal <= goal_threshold_m

        if goal_reached and not self._goal_reached_announced:
            self.get_logger().info(f'GOAL REACHED: err={dist_goal:.2f} m <= {goal_threshold_m:.2f} m')
            self._goal_reached_announced = True
        elif not goal_reached:
            self._goal_reached_announced = False

        yaw_ref = self._goal_yaw_reference(lane_path_world=self._lane_path_world)

        # Terminal reference: goal position, desired heading, zero terminal velocity
        x_ref = np.array([self._goal_xy[0], self._goal_xy[1], yaw_ref, 0.0, 0.0], dtype=float)

        # Lane path (or None if stale/unavailable)
        lane_path = self._lane_path_for_solver(now)
        x_ref     = self._terminal_reference_for_solver(x_ref, lane_path)

        # Nearest obstacle(s)
        x_min_debug = self._obs_map.nearest_k(x0[:2], N_OBS)
        x_min       = None if x_min_debug.shape[0] == 0 else np.array(x_min_debug[0], dtype=float)
        x_obs       = None if x_min_debug.shape[0] == 0 else np.array(x_min_debug, dtype=float)

        if bool(self.get_parameter('disable_obstacle_constraints').value):
            x_min = x_obs = None
            x_min_debug = np.zeros((0, 2), dtype=float)
            if not self._warned_obstacle_constraints_disabled:
                self.get_logger().warn('Obstacle constraints disabled.')
                self._warned_obstacle_constraints_disabled = True

        return {
            'goal_generation':  self._goal_generation,
            'goal_reached':     goal_reached,
            'goal_threshold_m': goal_threshold_m,
            'dist_goal':        dist_goal,
            'obs_snapshot':     self._obs_map.snapshot(DEBUG_PLOT_OBS_MAX),
            'stamp_msg':        now.to_msg(),
            'u_warm':           self._U_warm.copy(),
            'x0':               x0,
            'x_min':            x_min,
            'x_min_debug':      x_min_debug,
            'x_obs':            x_obs,
            'lane_path':        lane_path,
            'x_ref':            x_ref,
            't_now':            float(now.nanoseconds) * 1e-9,
        }

    def _solve_mpc_request(self, req: dict) -> dict:
        """Run SCP solve in background thread."""
        t0 = time.monotonic()
        U_opt, X_opt = self._mpc.solve(
            req['x0'], req['x_ref'], req['x_obs'], req['u_warm'],
            x_ref_path=req['lane_path'],
        )
        solve_sec = time.monotonic() - t0

        U_warm_next = np.roll(U_opt, -1, axis=0)
        U_warm_next[-1] = U_opt[-1]

        # First-step velocity from the optimised trajectory
        vel_cmd = np.array(X_opt[1, 3:5], dtype=float) if X_opt.shape[0] > 1 else (
            req['x0'][3:5] + Ts * U_opt[0, :2])
        np.clip(vel_cmd, -V_MAX, V_MAX, out=vel_cmd)
        yaw_rate_cmd = float(U_opt[0, 2]) if U_opt.shape[0] > 0 else 0.0

        # Waypoints for label publishing
        label_wps = X_opt[1:1 + MPC_LABEL_WAYPOINT_COUNT, :2]

        return {
            'goal_generation':  req['goal_generation'],
            'goal_reached':     req['goal_reached'],
            'goal_threshold_m': req['goal_threshold_m'],
            'dist_goal':        req['dist_goal'],
            'X_opt':            X_opt,
            'U_warm_next':      U_warm_next,
            'vel_cmd':          np.array([vel_cmd[0], vel_cmd[1], yaw_rate_cmd], dtype=float),
            'label_wps':        label_wps,
            'obs_snapshot':     req['obs_snapshot'],
            'stamp_msg':        req['stamp_msg'],
            'solve_sec':        solve_sec,
            'solver_status':    self._mpc.last_solver_status,
            'x0':               req['x0'],
            'x_ref':            req['x_ref'],
            'x_min':            req['x_min'],
            'x_min_debug':      req['x_min_debug'],
            't_now':            req['t_now'],
        }

    def _consume_solver_result(self) -> None:
        """Apply background solve result on the ROS timer thread."""
        future = self._solver_future
        self._solver_future = None
        if future is None:
            return
        try:
            res = future.result()
        except Exception as exc:
            self.get_logger().error(f'MPC solve failed: {exc}')
            self._U_warm[:] = 0.0
            return

        if res['goal_generation'] != self._goal_generation:
            return

        self._last_solve_duration_sec = float(res['solve_sec'])
        if self._last_solve_duration_sec > SOLVE_WARN_SEC:
            now_m = time.monotonic()
            if now_m - self._last_slow_solver_warn_monotonic >= 2.0:
                self.get_logger().warn(
                    f'MPC solve {1000*self._last_solve_duration_sec:.0f} ms '
                    f'> {1000*SOLVE_WARN_SEC:.0f} ms threshold.'
                )
                self._last_slow_solver_warn_monotonic = now_m

        self._U_warm = np.array(res['U_warm_next'], dtype=float)
        if res['label_wps'].shape[0] == 0:
            self.get_logger().warn('MPC returned no future waypoints; keeping previous command.')
            return

        self._last_u_cmd   = np.array(res['vel_cmd'], dtype=float)
        self._latest_vel_sp = self._last_u_cmd.copy()

        stamp = res['stamp_msg']
        self._publish_label_trajectory(res['label_wps'], stamp)
        self._publish_traj_point(res['label_wps'][0], stamp)
        self._publish_waypoint(self._exec_wp_pub,  res['label_wps'][0], self.get_clock().now().to_msg())
        self._publish_waypoint(self._fresh_wp_pub, res['label_wps'][0], stamp)

        self._debug_plotter.submit(
            x0=res['x0'], x_ref=res['x_ref'], X_opt=res['X_opt'],
            x_min=res['x_min_debug'],
            obs_pts=res['obs_snapshot'],
            t_now=res['t_now'],
            yaw_ref=float(res['x_ref'][2]),
            goal_reached=bool(res['goal_reached']),
            dist_goal=float(res['dist_goal']),
        )

        if self._tick_count % SETPOINT_LOG_EVERY_N == 0:
            x0   = res['x0']
            x_ref = res['x_ref']
            x_min = res['x_min']
            d_obs = float(norm(x_min[:2] - x0[:2])) if x_min is not None else float('inf')
            sp_hz = self._timer_rate_hz(self._setpoint_pub_dt)
            self.get_logger().info(
                f'pos=[{x0[0]:.2f},{x0[1]:.2f}] '
                f'yaw={x0[2]:.2f} '
                f'vel=[{x0[3]:.2f},{x0[4]:.2f}] '
                f'goal=[{x_ref[0]:.2f},{x_ref[1]:.2f}] '
                f'dist_goal={res["dist_goal"]:.2f}m '
                f'dist_obs={d_obs:.2f}m '
                f'cmd=[{self._latest_vel_sp[0]:.2f},{self._latest_vel_sp[1]:.2f},{self._latest_vel_sp[2]:.2f}] '
                f'solver={res["solver_status"]} '
                f'solve_ms={1000*res["solve_sec"]:.0f} '
                f'sp_rate={"n/a" if sp_hz is None else f"{sp_hz:.1f}"}Hz'
            )

    def _debug_plot_step(self) -> None:
        self._debug_plotter.draw_latest()

    # ── lane helpers ─────────────────────────────────────────────────────────

    def _lane_path_for_solver(self, request_now) -> np.ndarray | None:
        """Return a T-step world-frame lane reference, or None if stale."""
        if not self._enable_lane_tracking or self._lane_path_world.shape[0] == 0:
            return None
        now_sec = float(request_now.nanoseconds) * 1e-9
        if self._lane_stamp_sec is None or (now_sec - self._lane_stamp_sec) > LANE_REFERENCE_TIMEOUT_SEC:
            return None
        path = np.asarray(self._lane_path_world, dtype=float)
        if path.shape[0] < T_horizon:
            pad  = np.repeat(path[-1:], T_horizon - path.shape[0], axis=0)
            path = np.vstack([path, pad])
        else:
            path = path[:T_horizon]
        # Blend toward goal as vehicle approaches target
        remaining = float(norm(self._goal_xy - self._x_state[:2]))
        blend     = float(np.clip(remaining / max(LANE_BLEND_FAR_METERS, 1e-3), 0.0, 1.0))
        if blend < 1.0:
            goal_line = np.linspace(self._x_state[:2], self._goal_xy, T_horizon + 1)[1:]
            path = blend * path + (1.0 - blend) * goal_line
        return path

    def _terminal_reference_for_solver(
        self,
        x_ref: np.ndarray,
        lane_path: np.ndarray | None,
    ) -> np.ndarray:
        """Bias terminal velocity toward lane tangent when tracking is active."""
        if lane_path is None or lane_path.shape[0] < 2:
            return x_ref
        tangent      = np.asarray(lane_path[-1] - lane_path[-2], dtype=float)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-6:
            return x_ref
        speed         = min(self._gem_max_speed_mps, LANE_HEADING_REF_SPEED_MPS)
        x_ref_out     = x_ref.copy()
        x_ref_out[2] = float(np.arctan2(tangent[1], tangent[0]))
        x_ref_out[3:5] = speed * tangent / tangent_norm
        return x_ref_out

    def _goal_yaw_reference(self, lane_path_world: np.ndarray | None) -> float:
        if lane_path_world is not None and lane_path_world.shape[0] >= 2:
            tangent = np.asarray(lane_path_world[-1] - lane_path_world[-2], dtype=float)
            if np.linalg.norm(tangent) > 1e-6:
                return float(np.arctan2(tangent[1], tangent[0]))
        delta_goal = np.asarray(self._goal_xy - self._x_state[:2], dtype=float)
        if np.linalg.norm(delta_goal) > 1e-6:
            return float(np.arctan2(delta_goal[1], delta_goal[0]))
        return float(self._yaw)

    def _publish_lane_reference(self, path_xy: np.ndarray, stamp_msg) -> None:
        if path_xy.shape[0] == 0:
            return
        msg = Path()
        msg.header.stamp    = stamp_msg
        msg.header.frame_id = 'map'
        for xy in path_xy:
            pose = PoseStamped()
            pose.header       = msg.header
            pose.pose.position.x = float(xy[0])
            pose.pose.position.y = float(xy[1])
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self._lane_ref_pub.publish(msg)

    # ── publishing helpers ───────────────────────────────────────────────────

    def _republish_latest_setpoint(self) -> None:
        now = time.monotonic()
        if self._last_setpoint_pub_monotonic is not None:
            self._setpoint_pub_dt.append(now - self._last_setpoint_pub_monotonic)
        self._last_setpoint_pub_monotonic = now
        self._publish_ackermann(self._latest_vel_sp)

    def _publish_ackermann(self, vel_world: np.ndarray) -> None:
        """Convert world-frame velocity command to AckermannDrive."""
        vx, vy = float(vel_world[0]), float(vel_world[1])
        yaw_rate_cmd = float(vel_world[2]) if vel_world.shape[0] > 2 else 0.0
        speed_mag = float(np.hypot(vx, vy))
        if speed_mag < 1e-3:
            target_speed = 0.0
            steering     = 0.0
        else:
            c        = float(np.cos(self._yaw))
            s        = float(np.sin(self._yaw))
            forward  = c * vx + s * vy
            target_speed = float(np.clip(forward, -self._gem_max_speed_mps, self._gem_max_speed_mps))
            desired_heading = float(np.arctan2(vy, vx))
            heading_error   = self._wrap_pi(desired_heading - self._yaw)
            yaw_rate_cmd    = yaw_rate_cmd + self._gem_steering_kp * heading_error
            steer_speed     = max(abs(target_speed), 0.25)
            steering = float(np.clip(
                np.arctan2(self._gem_wheelbase_m * yaw_rate_cmd, steer_speed),
                -self._gem_max_steering_rad, self._gem_max_steering_rad,
            ))
        msg = AckermannDrive()
        msg.speed                   = float(target_speed)
        msg.steering_angle          = float(steering)
        msg.steering_angle_velocity = 0.0
        msg.acceleration            = 0.0
        msg.jerk                    = 0.0
        self._ackermann_pub.publish(msg)

    def _publish_label_trajectory(self, wps_xy: np.ndarray, stamp_msg) -> None:
        path = Path()
        path.header.stamp    = stamp_msg
        path.header.frame_id = 'map'
        for p in wps_xy:
            pose = PoseStamped()
            pose.header       = path.header
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._mpc_label_path_pub.publish(path)
        self._mpc_local_pred_pub.publish(path)

    def _publish_traj_point(self, xy: np.ndarray, stamp_msg) -> None:
        msg = PoseStamped()
        msg.header.stamp    = stamp_msg
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(xy[0])
        msg.pose.position.y = float(xy[1])
        msg.pose.orientation.w = 1.0
        self._mpc_traj_pub.publish(msg)

    @staticmethod
    def _publish_waypoint(pub, xy: np.ndarray, stamp_msg) -> None:
        msg = PoseStamped()
        msg.header.stamp    = stamp_msg
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(xy[0])
        msg.pose.position.y = float(xy[1])
        msg.pose.orientation.w = 1.0
        pub.publish(msg)

    # ── goal API ─────────────────────────────────────────────────────────────

    def set_goal(self, x: float, y: float) -> None:
        """Set navigation goal in world XY [m]."""
        self._goal_xy = np.array([x, y], dtype=float)
        self._goal_generation += 1
        self._obs_map.clear()
        self._U_warm[:] = 0.0
        self._last_u_cmd[:] = 0.0
        self._latest_vel_sp[:] = 0.0
        self._goal_reached_announced = False
        self.get_logger().info(f'New goal: x={x:.2f} m, y={y:.2f} m')

    # ── static helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
        n = np.sqrt(w*w + x*x + y*y + z*z)
        if n < 1e-9:
            return 0.0
        w, x, y, z = w/n, x/n, y/n, z/n
        return float(np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)))

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _timer_rate_hz(samples: deque) -> float | None:
        if not samples:
            return None
        mean_dt = float(np.mean(samples))
        return None if mean_dt <= 1e-6 else 1.0 / mean_dt

    def destroy_node(self):
        if self._solver_future is not None:
            self._solver_future.cancel()
            self._solver_future = None
        try:
            self._solver_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._solver_executor.shutdown(wait=False)
        self._debug_plotter.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MPCUAVNode()
    goal_x = float(node.get_parameter('goal_x').value)
    goal_y = float(node.get_parameter('goal_y').value)
    node.set_goal(goal_x, goal_y)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
