#!/usr/bin/env python3
"""
MPC-based UAV Trajectory Planner for ROS2
==========================================

System:
  - Sensor  : /depth_camera  (sensor_msgs/Image, encoding=32FC1, metres)
  - Output  : /fmu/in/trajectory_setpoint  (px4_msgs/TrajectorySetpoint, NED frame)
  - State   : /fmu/out/vehicle_local_position (px4_msgs/VehicleLocalPosition)

MPC Formulation (discrete-time, horizon T steps)
-------------------------------------------------
Solver state  : [x, y, vx, vy] in the horizontal plane (NED)
Solver control: [ax, ay] planar accelerations
PX4 output    : [vx_ref, vy_ref, vz_ref] velocity setpoint in NED
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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from concurrent.futures import Future, ThreadPoolExecutor
import time
import numpy as np
from numpy.linalg import norm
import cv2
from cv_bridge import CvBridge
from collections import deque
import os
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
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import TrajectorySetpoint, VehicleOdometry, OffboardControlMode, VehicleStatus

# ─────────────────────────────────────────────────────────────────────────────
# MPC PARAMETERS  (tune these)
# ─────────────────────────────────────────────────────────────────────────────
Ts          = 0.1         # sampling time [s]
T_horizon   = 50         # MPC prediction horizon [steps]
V_MAX       = 15.0        # per-axis max commanded velocity [m/s]
A_MAX       = 8.0         # per-axis max commanded acceleration [m/s^2]
N_OBS       = 1          # nearest-point method: N=1 (Section III-A)
Q_diag      = np.array([20.0, 20.0, 20.0, 1, 1, 1])   # tracking weight Q
SCP_MAX_OUTER_ITERS = 4    # sequential convexification iterations per MPC cycle
SCP_INNER_ITERS = 6        # projected-gradient iterations per convex sub-problem
SCP_EPS      = 1e-3        # stop when trajectory update is small
SCP_LR       = 0.003       # projected-gradient base step size (stability-critical)
SCP_LR_BACKTRACK = 0.5     # multiplicative backtracking factor for failed inner steps
SCP_LS_MAX_STEPS = 6       # max line-search reductions per inner iteration
SCP_LAMBDA   = 100.0       # penalty multiplier (constraints), per MATLAB-style phi/phi_hat
SCP_ALPHA    = 0.1         # trust-region acceptance parameter
SCP_BETA_GROW = 1.1        # trust-region growth factor after accepted step
SCP_BETA_SHRINK = 0.5      # trust-region shrink factor after rejected step
TRUST_POS0   = 1.0         # initial trust region on position trajectory [m]
TRUST_U0     = 1.0         # initial tust region on velocity command [m/s]
MIN_CLEARANCE = 1.5    # desired clearance [m] from nearest obstacle point
# OakD-Lite StereoOV7251 depth sensor from PX4 Gazebo SDF:
#   x500_depth mount + OakD-Lite/model.sdf (horizontal_fov=1.274 rad, 640x480,
#   near=0.2 m, far=19.1 m). Vertical FOV is derived from aspect ratio.
DEPTH_HFOV  = 72.995       # depth camera horizontal FOV [deg] (OakD-Lite StereoOV7251)
DEPTH_VFOV  = 58.053       # depth camera vertical FOV [deg] (derived from 640x480 + HFOV)
DEPTH_MAX   = 19.1         # depth camera far clip [m] from OakD-Lite SDF
DEPTH_MIN   = 0.2          # depth camera near clip [m] from OakD-Lite SDF
BOX_THRESH  = 0.05         # bounding-box volume threshold [m³] to filter micro-debris
FIFO_LEN    = 200          # local obstacle map FIFO size (Section III-B)
SETPOINT_LOG_EVERY_N = 10  # print MPC/PX4 setpoint debug every N control ticks
DEBUG_PLOT_ENABLED = False  # non-blocking 2D MPC debug plots (optional)
DEBUG_ENABLE_DEPTH_CAMERA = True  # non-blocking 2D MPC debug plots (optional)
DEBUG_PLOT_RATE_HZ = 5.0   # plot refresh rate, decoupled from MPC loop
DEBUG_PLOT_OBS_MAX = 200   # max local obstacle points shown in debug plot
DEBUG_PATH_HISTORY_LEN = 300
DEPTH_FILTER_MIN_FORWARD_M = 0.35
DEPTH_FILTER_MAX_FORWARD_M = 12.0
DEPTH_FILTER_MAX_LATERAL_M = 4.0
DEPTH_FILTER_MIN_BELOW_BODY_M = -0.25
DEPTH_FILTER_MAX_BELOW_BODY_M = 2.5
DEPTH_FILTER_FLOOR_CLEARANCE_M = 0.20
MPC_LABEL_WAYPOINT_COUNT = 5  # publish next 5 waypoints from final solved X_opt for dataset labels
USE_VELOCITY_ONLY_SETPOINT = True  # publish velocity setpoints as primary NED command
LOCAL_PREDICTION_SEQUENCE_TOPIC = '/mpc/local_prediction_sequence'
FRESH_COMMITTED_WAYPOINT_TOPIC = '/mpc/committed_waypoint_fresh'
EXECUTED_COMMITTED_WAYPOINT_TOPIC = '/mpc/committed_waypoint_executed'
PUBLISH_PERIOD_SEC = 0.05
OFFBOARD_HEARTBEAT_PERIOD_SEC = 0.05
SOLVE_WARN_SEC = 0.10
RATE_WINDOW_LEN = 40
GOAL_REACHED_THRESHOLD_M = 0.5
GOAL_REACHED_YAW_THRESHOLD_RAD = 0.15
ENABLE_DELAY_COMPENSATION = False  # global default: compensate PX4 tracking lag in MPC initial state
DELAY_COMPENSATION_SEC = 0.15      # forward-prediction horizon for delay compensation [s]
DEFAULT_MPC_SOLVER_BACKEND = 'python'
DEFAULT_ENABLE_DEPTH_CAMERA = True
DEFAULT_OBSTACLE_CONSTRAINTS_DISABLED = False
# Target heading: Gazebo +X (ENU East). Converted to NED yaw reference:
# yaw_ned = pi/2 - yaw_enu, with yaw_enu(+X)=0 => yaw_ned=+pi/2.
YAW_TARGET_RAD = np.pi / 2.0       # terminal yaw target in NED [rad]
Z_HOLD_THRESHOLD_M = 0.25          # if |z_goal - z| <= threshold, command vz=0
Z_HOLD_KP = 1.5                    # proportional gain for z hold velocity command
VZ_HOLD_MAX = V_MAX                # clamp z-hold velocity command [m/s]
U_LIMS = np.array([A_MAX, A_MAX], dtype=float)
VEL_LIMS = np.array([V_MAX, V_MAX], dtype=float)
# ─────────────────────────────────────────────────────────────────────────────


class LocalObstacleMap:
    """
    FIFO-based local obstacle map (Section III-B of paper).

    Stores recent laser/depth scan points in body frame and returns the
    nearest obstacle point to a query position.
    """

    def __init__(self, maxlen: int = FIFO_LEN):
        self._buf: list[np.ndarray] = []   # list of (3,) spatial-frame points
        self._maxlen = maxlen

    def update(self, points_S: np.ndarray) -> None:
        """
        Add new spatial-frame obstacle points to map.
        points_S : (N, 3) array in NED metres
        """
        if points_S.shape[0] == 0:
            return

        # Salt-and-pepper bounding-box filter disabled on request.
        # if points_S.shape[0] >= 3:
        #     lo = points_S.min(axis=0)
        #     hi = points_S.max(axis=0)
        #     vol = float(np.prod(np.maximum(hi - lo, 1e-3)))
        #     if vol < BOX_THRESH:
        #         return  # tiny cluster → treat as debris, ignore

        for pt in points_S:
            self._buf.append(pt.copy())

        # FIFO eviction
        if len(self._buf) > self._maxlen:
            self._buf = self._buf[-self._maxlen:]

    def nearest(self, x_ref: np.ndarray) -> np.ndarray | None:
        """
        Return the point in the map closest to x_ref (3-vector, NED).
        Implements Eq.(11):  X_O^min = argmin_{X_O in S_obs} ||X_O - X_ref||_2
        Returns None if map is empty.
        """
        if not self._buf: ## therres a local map
            return None
        pts   = np.array(self._buf)          # (N, 3)
        dists = norm(pts - x_ref[None, :3], axis=1)
        return pts[np.argmin(dists)]

    def nearest_k(self, x_ref: np.ndarray, k: int) -> np.ndarray:
        """Return up to k nearest obstacle points to x_ref, sorted nearest-first."""
        if not self._buf or k <= 0:
            return np.zeros((0, 3), dtype=float)
        pts = np.array(self._buf, dtype=float)
        dists = norm(pts - np.asarray(x_ref, dtype=float)[None, :3], axis=1)
        order = np.argsort(dists)[:int(k)]
        return pts[order]

    def clear(self):
        self._buf.clear()

    def snapshot(self, max_points: int | None = None) -> np.ndarray:
        """Return a copy of cached obstacle points for debug/visualisation."""
        if not self._buf:
            return np.zeros((0, 3), dtype=float)
        pts = np.array(self._buf, dtype=float)
        if max_points is not None and pts.shape[0] > max_points:
            pts = pts[-max_points:]
        return pts


class MPCDebugPlotter2D:
    """
    Optional async matplotlib plotter for MPC debug views.

    Rendering is triggered by a lightweight ROS timer callback on the main
    thread so matplotlib GUI handling remains reliable.
    """

    def __init__(self,
                 enabled: bool = DEBUG_PLOT_ENABLED,
                 rate_hz: float = DEBUG_PLOT_RATE_HZ,
                 path_history_len: int = DEBUG_PATH_HISTORY_LEN):
        self._enabled = bool(enabled)
        self._rate_hz = max(rate_hz, 0.2)
        self._latest: dict | None = None
        self._path_hist = deque(maxlen=path_history_len)
        self._plt = None
        self._fig = None
        self._axes = None
        self._artists: dict[str, object] = {}

        if not self._enabled:
            return

        try:
            import matplotlib.pyplot as plt  # type: ignore
            self._plt = plt
            self._plt.ion()
            self._fig, self._axes = self._plt.subplots(1, 2, figsize=(11, 5))
            try:
                self._fig.canvas.manager.set_window_title('MPC Local Planner Debug')
            except Exception:
                pass
            self._init_figure()
            self._plt.show(block=False)
            display = os.environ.get('DISPLAY', '')
            print(f'[MPCDebugPlotter2D] Started (DISPLAY={display or "unset"}, rate={self._rate_hz:.1f} Hz)')
        except Exception as e:
            self._enabled = False
            print(f'[MPCDebugPlotter2D] Disabled (matplotlib init failed): {e}')

    def _init_figure(self) -> None:
        ax_xy, ax_xz = self._axes
        ax_xy.set_title('Top View (N-E)')
        ax_xy.set_xlabel('North [m]')
        ax_xy.set_ylabel('East [m]')
        ax_xy.grid(True, alpha=0.3)
        ax_xy.axis('equal')

        ax_xz.set_title('Vertical View (N-D)')
        ax_xz.set_xlabel('North [m]')
        ax_xz.set_ylabel('Down [m] (NED)')
        ax_xz.grid(True, alpha=0.3)

        self._artists['obs_xy'] = ax_xy.scatter([], [], s=10, c='0.7', alpha=0.6, label='local obs')
        self._artists['path_xy'], = ax_xy.plot([], [], 'k-', lw=1.5, alpha=0.8, label='actual path')
        self._artists['pred_xy'], = ax_xy.plot([], [], 'b.-', lw=1.5, ms=5, label='MPC pred')
        self._artists['cur_xy'], = ax_xy.plot([], [], 'go', ms=7, label='current')
        self._artists['goal_xy'], = ax_xy.plot([], [], 'r*', ms=12, label='goal')
        self._artists['nearest_xy'], = ax_xy.plot([], [], 'mx', ms=8, mew=2, label='nearest obs')
        self._artists['goal_msg_xy'] = ax_xy.text(
            0.03, 0.97, '', transform=ax_xy.transAxes, va='top', ha='left',
            fontsize=11, color='tab:green', fontweight='bold'
        )
        ax_xy.legend(loc='best', fontsize=8)

        self._artists['path_xz'], = ax_xz.plot([], [], 'k-', lw=1.5, alpha=0.8, label='actual path')
        self._artists['pred_xz'], = ax_xz.plot([], [], 'b.-', lw=1.5, ms=5, label='MPC pred')
        self._artists['cur_xz'], = ax_xz.plot([], [], 'go', ms=7, label='current')
        self._artists['goal_xz'], = ax_xz.plot([], [], 'r*', ms=12, label='goal')
        self._artists['nearest_xz'], = ax_xz.plot([], [], 'mx', ms=8, mew=2, label='nearest obs')
        self._artists['goal_msg_xz'] = ax_xz.text(
            0.03, 0.97, '', transform=ax_xz.transAxes, va='top', ha='left',
            fontsize=11, color='tab:green', fontweight='bold'
        )
        ax_xz.legend(loc='best', fontsize=8)

        self._fig.tight_layout()

    def submit(self,
               x0: np.ndarray,
               x_ref: np.ndarray,
               X_opt: np.ndarray,
               x_min: np.ndarray | None,
               obs_pts: np.ndarray | None = None,
               goal_reached: bool = False,
               dist_goal: float | None = None,
               yaw_err: float | None = None,
               goal_threshold_m: float = GOAL_REACHED_THRESHOLD_M,
               goal_yaw_threshold_rad: float = GOAL_REACHED_YAW_THRESHOLD_RAD) -> None:
        """Store latest MPC snapshot for async plotting."""
        if not self._enabled:
            return

        x0_pos = np.array(x0[:3], dtype=float, copy=True)
        x_ref_pos = np.array(x_ref[:3], dtype=float, copy=True)
        x_pred = np.array(X_opt[:, :3], dtype=float, copy=True)
        x_min_copy = None if x_min is None else np.array(x_min, dtype=float, copy=True)
        obs_copy = None
        if obs_pts is not None:
            obs_copy = np.array(obs_pts[:, :3], dtype=float, copy=True)

        self._path_hist.append(x0_pos)
        self._latest = {
            'x0': x0_pos,
            'x_ref': x_ref_pos,
            'x_pred': x_pred,
            'x_min': x_min_copy,
            'obs': obs_copy,
            'goal_reached': bool(goal_reached),
            'dist_goal': None if dist_goal is None else float(dist_goal),
            'yaw_err': None if yaw_err is None else float(yaw_err),
            'goal_threshold_m': float(goal_threshold_m),
            'goal_yaw_threshold_rad': float(goal_yaw_threshold_rad),
        }

    def draw_latest(self) -> None:
        """Render the latest snapshot on the main thread."""
        if not self._enabled or self._latest is None:
            return
        try:
            snap = self._latest.copy()
            path_hist = np.array(self._path_hist, dtype=float) if self._path_hist else None
            self._draw(snap, path_hist)
        except Exception as e:
            print(f'[MPCDebugPlotter2D] Plot loop stopped: {e}')
            self._enabled = False

    def _draw(self, snap: dict, path_hist: np.ndarray | None) -> None:
        x0 = snap['x0']
        x_ref = snap['x_ref']
        x_pred = snap['x_pred']
        x_min = snap['x_min']
        obs = snap['obs']
        goal_reached = bool(snap.get('goal_reached', False))
        dist_goal = snap.get('dist_goal', None)
        yaw_err = snap.get('yaw_err', None)
        goal_threshold_m = float(snap.get('goal_threshold_m', GOAL_REACHED_THRESHOLD_M))
        goal_yaw_threshold_rad = float(
            snap.get('goal_yaw_threshold_rad', GOAL_REACHED_YAW_THRESHOLD_RAD)
        )

        pred_n = x_pred[:, 0]
        pred_e = x_pred[:, 1]
        pred_d = x_pred[:, 2]

        path_n = np.array([])
        path_e = np.array([])
        path_d = np.array([])
        if path_hist is not None and path_hist.size:
            path_n = path_hist[:, 0]
            path_e = path_hist[:, 1]
            path_d = path_hist[:, 2]

        self._artists['pred_xy'].set_data(pred_n, pred_e)
        self._artists['cur_xy'].set_data([x0[0]], [x0[1]])
        self._artists['goal_xy'].set_data([x_ref[0]], [x_ref[1]])
        self._artists['path_xy'].set_data(path_n, path_e)
        if x_min is not None and np.size(x_min) > 0:
            x_min_2d = np.atleast_2d(x_min)
            self._artists['nearest_xy'].set_data(x_min_2d[:, 0], x_min_2d[:, 1])
        else:
            self._artists['nearest_xy'].set_data([], [])

        if obs is not None and obs.size:
            self._artists['obs_xy'].set_offsets(obs[:, :2])
        else:
            self._artists['obs_xy'].set_offsets(np.zeros((0, 2)))

        self._artists['pred_xz'].set_data(pred_n, pred_d)
        self._artists['cur_xz'].set_data([x0[0]], [x0[2]])
        self._artists['goal_xz'].set_data([x_ref[0]], [x_ref[2]])
        self._artists['path_xz'].set_data(path_n, path_d)
        if x_min is not None and np.size(x_min) > 0:
            x_min_2d = np.atleast_2d(x_min)
            self._artists['nearest_xz'].set_data(x_min_2d[:, 0], x_min_2d[:, 2])
        else:
            self._artists['nearest_xz'].set_data([], [])

        if goal_reached:
            if dist_goal is None:
                msg = 'GOAL REACHED'
            else:
                if yaw_err is None:
                    msg = f'GOAL REACHED (pos_err={dist_goal:.2f} m <= {goal_threshold_m:.2f} m)'
                else:
                    msg = (
                        f'GOAL REACHED (pos_err={dist_goal:.2f} m <= {goal_threshold_m:.2f} m, '
                        f'yaw_err={abs(float(yaw_err)):.2f} rad <= {goal_yaw_threshold_rad:.2f} rad)'
                    )
            self._artists['goal_msg_xy'].set_text(msg)
            self._artists['goal_msg_xz'].set_text(msg)
        else:
            self._artists['goal_msg_xy'].set_text('')
            self._artists['goal_msg_xz'].set_text('')

        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()

        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def close(self) -> None:
        if not self._enabled:
            return
        if self._plt is not None and self._fig is not None:
            try:
                self._plt.close(self._fig)
            except Exception:
                pass


class MPCPlanner:
    """
    MATLAB-style SCP planner:
      - penalty objective phi (real) and phi_hat (convexified),
      - linearized obstacle constraints around previous trajectory,
      - trust-region accept/reject update using delta and delta_hat.
    """
    def __init__(self, backend: str = DEFAULT_MPC_SOLVER_BACKEND):
        requested_backend = str(backend).strip().lower()
        if requested_backend == 'auto':
            if native_osqp_available():
                resolved_backend = 'cpp_osqp'
            elif native_backend_available():
                resolved_backend = 'cpp_subgradient'
            else:
                resolved_backend = 'python'
        elif requested_backend == 'cpp_osqp':
            if not native_osqp_available():
                raise RuntimeError(
                    'mpc_solver_backend=cpp_osqp requested, but llm_drone._mpc_native.solve_osqp is not available'
                )
            resolved_backend = 'cpp_osqp'
        elif requested_backend == 'cpp_subgradient':
            if not native_backend_available():
                raise RuntimeError(
                    'mpc_solver_backend=cpp_subgradient requested, but llm_drone._mpc_native is not available'
                )
            resolved_backend = 'cpp_subgradient'
        elif requested_backend == 'python':
            resolved_backend = 'python'
        else:
            raise ValueError(
                f'Unsupported mpc_solver_backend={backend!r}; expected one of '
                "'python', 'cpp_osqp', 'cpp_subgradient', or 'auto'"
            )

        self.backend_name = resolved_backend
        self.last_solver_status = 'not_run'

    @staticmethod
    def _native_solver_config() -> dict:
        return {
            'Ts': Ts,
            'T_horizon': T_horizon,
            'V_MAX': V_MAX,
            'A_MAX': A_MAX,
            'SCP_MAX_OUTER_ITERS': SCP_MAX_OUTER_ITERS,
            'SCP_INNER_ITERS': SCP_INNER_ITERS,
            'SCP_EPS': SCP_EPS,
            'SCP_LR': SCP_LR,
            'SCP_LAMBDA': SCP_LAMBDA,
            'SCP_ALPHA': SCP_ALPHA,
            'SCP_BETA_GROW': SCP_BETA_GROW,
            'SCP_BETA_SHRINK': SCP_BETA_SHRINK,
            'TRUST_POS0': TRUST_POS0,
            'TRUST_U0': TRUST_U0,
            'MIN_CLEARANCE': MIN_CLEARANCE,
        }

    @staticmethod
    def _rollout(x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Rollout with controls [ax, ay] for 2D state [x, y, vx, vy]."""
        X = np.zeros((T_horizon + 1, 4), dtype=float)
        X[0] = x0
        for k in range(T_horizon):
            X[k + 1, :2] = X[k, :2] + Ts * X[k, 2:4]
            X[k + 1, 2:4] = X[k, 2:4] + Ts * U[k, :2]
        return X

    @staticmethod
    def _project_speed_limits(U: np.ndarray) -> None:
        """Component-wise saturation for [ax, ay]."""
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

    def _phi_real(
        self,
        X: np.ndarray,
        U: np.ndarray,
        x_ref: np.ndarray,
        x_obs: np.ndarray | None,
    ) -> float:
        """MATLAB-like phi(X,U): objective + lambda*(eq violations + ineq violations)."""
        x_obs = self._normalize_x_obs(x_obs)
        eqns = 0.0
        eqns += self._sum_abs(X[-1, :2] - x_ref[:2])
        eqns += self._sum_abs(X[-1, 2:4] - x_ref[2:4])

        ineq_u = np.sum(self._hinge(np.abs(U) - U_LIMS[None, :]))
        ineq_v = np.sum(self._hinge(np.abs(X[1:, 2:4]) - VEL_LIMS[None, :]))

        ineq_obs = 0.0
        if x_obs is not None:
            d = X[1:, None, :2] - x_obs[None, :, :]
            dist = np.linalg.norm(d, axis=2)
            ineq_obs = float(np.sum(self._hinge(MIN_CLEARANCE - dist)))

        obj = float(Ts * np.sum(U * U))
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
    ) -> float:
        """MATLAB-like convexified phi_hat(X,U) around (X_now,U_now)."""
        x_obs = self._normalize_x_obs(x_obs)
        eqns = 0.0
        eqns += self._sum_abs(X[-1, :2] - x_ref[:2])
        eqns += self._sum_abs(X[-1, 2:4] - x_ref[2:4])

        ineq_u = np.sum(self._hinge(np.abs(U) - U_LIMS[None, :]))
        ineq_v = np.sum(self._hinge(np.abs(X[1:, 2:4]) - VEL_LIMS[None, :]))
        ineq_tr_u = np.sum(self._hinge(np.abs(U - U_now) - l_u))
        ineq_tr_x = np.sum(self._hinge(np.abs(X[1:, :2] - X_now[1:, :2]) - l_pos))

        ineq_obs = 0.0
        if x_obs is not None:
            d_now = X_now[1:, None, :2] - x_obs[None, :, :]
            d_now_norm = np.linalg.norm(d_now, axis=2)
            d = X[1:, None, :2] - x_obs[None, :, :]
            lin_obs = MIN_CLEARANCE * d_now_norm - np.sum(d_now * d, axis=2)
            ineq_obs = float(np.sum(self._hinge(lin_obs)))

        obj = float(Ts * np.sum(U * U))
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Subgradient of convexified objective wrt controls.
        Returns (grad, X(U)).
        """
        x_obs = self._normalize_x_obs(x_obs)
        X = self._rollout(x0, U)
        grad = 2.0 * Ts * U

        # terminal position L1 penalties via x(T) sensitivity wrt accelerations
        sign_pos = np.sign(X[-1, :2] - x_ref[:2])
        for j in range(T_horizon - 1):
            weight = Ts * Ts * float(T_horizon - 1 - j)
            grad[j, :2] += SCP_LAMBDA * weight * sign_pos

        # terminal velocity L1 penalties via v(T) sensitivity wrt accelerations
        sign_vel = np.sign(X[-1, 2:4] - x_ref[2:4])
        grad += SCP_LAMBDA * Ts * sign_vel[None, :]

        # control bounds |u| <= U_LIMS
        viol_u = np.abs(U) - U_LIMS[None, :]
        mask_u = viol_u > 0.0
        grad += SCP_LAMBDA * (mask_u * np.sign(U))

        # velocity bounds |v| <= VEL_LIMS
        for k in range(1, T_horizon + 1):
            v_k = X[k, 2:4]
            viol_v = np.abs(v_k) - VEL_LIMS
            mask_v = viol_v > 0.0
            if not np.any(mask_v):
                continue
            g_v = np.zeros(2, dtype=float)
            g_v[mask_v] = np.sign(v_k[mask_v])
            # v_k depends on accelerations a_0..a_{k-1}
            for j in range(k):
                grad[j, :2] += SCP_LAMBDA * Ts * g_v

        # control trust region |u-u_now| <= l_u
        viol_tr_u = np.abs(U - U_now) - l_u
        mask_tr_u = viol_tr_u > 0.0
        grad += SCP_LAMBDA * (mask_tr_u * np.sign(U - U_now))

        # state trust region |xy - xy_now| <= l_pos
        for k in range(1, T_horizon + 1):
            dx = X[k, :2] - X_now[k, :2]
            mask = np.abs(dx) - l_pos > 0.0
            if not np.any(mask):
                continue
            g_x = np.zeros(2, dtype=float)
            g_x[mask] = np.sign(dx[mask])
            for j in range(k - 1):
                weight = Ts * Ts * float(k - 1 - j)
                grad[j, :2] += SCP_LAMBDA * weight * g_x

        # convexified obstacle constraint
        if x_obs is not None:
            d_now = X_now[1:, None, :2] - x_obs[None, :, :]
            d_now_norm = np.linalg.norm(d_now, axis=2)
            d = X[1:, None, :2] - x_obs[None, :, :]
            lin_obs = MIN_CLEARANCE * d_now_norm - np.sum(d_now * d, axis=2)
            for k in range(T_horizon):
                for obs_idx in range(x_obs.shape[0]):
                    if lin_obs[k, obs_idx] <= 0.0:
                        continue
                    g_x = -d_now[k, obs_idx]
                    # x_{k+1} depends on accelerations a_0..a_{k-1}
                    for j in range(k):
                        weight = Ts * Ts * float(k - j)
                        grad[j, :2] += SCP_LAMBDA * weight * g_x

        return grad, X

    def solve(self,
              x0: np.ndarray,
              x_ref: np.ndarray,
              x_obs: np.ndarray | None,
              U_warm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run sequential convex optimisation with penalty constraints.

        Parameters
        ----------
        x0      : (4,) current planning state [x, y, vx, vy]
        x_ref   : (4,) desired terminal state [x, y, vx, vy]
        x_obs   : (K, 2) nearest obstacle positions in XY, or None
        U_warm  : (T, 2) warm-start control sequence [ax, ay]

        Returns
        -------
        U_opt   : (T, 2) optimised control sequence [ax, ay]
        X_opt   : (T+1, 4) optimised state trajectory
        """
        if self.backend_name == 'cpp_osqp':
            native_x_obs = None if x_obs is None else np.asarray(x_obs, dtype=float)
            if native_x_obs is not None and native_x_obs.ndim == 2:
                native_x_obs = native_x_obs[0]
            U_opt, X_opt, solver_status = solve_native_osqp(
                x0=x0,
                x_ref=x_ref,
                x_obs=native_x_obs,
                u_warm=U_warm,
                config=self._native_solver_config(),
            )
            self.last_solver_status = solver_status
            return U_opt, X_opt

        if self.backend_name == 'cpp_subgradient':
            native_x_obs = None if x_obs is None else np.asarray(x_obs, dtype=float)
            if native_x_obs is not None and native_x_obs.ndim == 2:
                native_x_obs = native_x_obs[0]
            U_opt, X_opt, solver_status = solve_native_subgradient(
                x0=x0,
                x_ref=x_ref,
                x_obs=native_x_obs,
                u_warm=U_warm,
                config=self._native_solver_config(),
            )
            self.last_solver_status = solver_status
            return U_opt, X_opt

        U_now = U_warm.copy()
        self._project_speed_limits(U_now)
        X_now = self._rollout(x0, U_now)

        l_pos = TRUST_POS0
        l_u = TRUST_U0

        for _ in range(SCP_MAX_OUTER_ITERS):
            U_candidate = self._solve_convex_subproblem(
                x0=x0,
                x_ref=x_ref,
                x_obs=x_obs,
                X_now=X_now,
                U_now=U_now,
                l_pos=l_pos,
                l_u=l_u,
            )
            if U_candidate is None:
                # Fallback to projected subgradient if cvxpy is unavailable/failed.
                self.last_solver_status = f'{self.last_solver_status}|fallback_subgradient'
                U_candidate = U_now.copy()
                for _ in range(SCP_INNER_ITERS):
                    grad, _ = self._phi_hat_gradient(
                        x0=x0,
                        U=U_candidate,
                        x_ref=x_ref,
                        x_obs=x_obs,
                        X_now=X_now,
                        U_now=U_now,
                        l_pos=l_pos,
                        l_u=l_u,
                    )
                    U_candidate -= SCP_LR * grad
                    self._project_speed_limits(U_candidate)

            X_candidate = self._rollout(x0, U_candidate)

            phi_real_now = self._phi_real(X_now, U_now, x_ref, x_obs)
            phi_hat_new = self._phi_hat(
                X=X_candidate,
                U=U_candidate,
                x_ref=x_ref,
                x_obs=x_obs,
                X_now=X_now,
                U_now=U_now,
                l_pos=l_pos,
                l_u=l_u,
            )
            phi_real_new = self._phi_real(X_candidate, U_candidate, x_ref, x_obs)

            delta_hat = phi_real_now - phi_hat_new
            delta = phi_real_now - phi_real_new

            #  trust-region update.
            if delta > SCP_ALPHA * delta_hat:
                l_pos *= SCP_BETA_GROW
                l_u *= SCP_BETA_GROW
                step_norm = float(np.max(np.abs(X_candidate - X_now)))
                X_now = X_candidate
                U_now = U_candidate
                if step_norm < SCP_EPS:
                    break
            else:
                l_pos *= SCP_BETA_SHRINK
                l_u *= SCP_BETA_SHRINK

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
    ) -> np.ndarray | None:
        """
        Solve one SCP convex subproblem with CVX
          minimize phi_hat(U)
        with no explicit user constraints ("no subject to"), matching the
        MATLAB formulation where bounds/trust/obstacle terms are soft penalties.
        Returns U or None on solver failure.
        """
        if cp is None:
            self.last_solver_status = 'no_cvxpy'
            return None
        x_obs = self._normalize_x_obs(x_obs)

        U = cp.Variable((T_horizon, 2))
        X_pos = cp.Variable((T_horizon + 1, 2))
        X_vel = cp.Variable((T_horizon + 1, 2))
        constraints = [
            X_pos[0, :] == x0[:2],
            X_vel[0, :] == x0[2:4],
        ]
        for k in range(T_horizon):
            constraints += [
                X_pos[k + 1, :] == X_pos[k, :] + Ts * X_vel[k, :],
                X_vel[k + 1, :] == X_vel[k, :] + Ts * U[k, :],
            ]

        eq_term = (
            cp.norm1(X_pos[-1, :] - x_ref[:2])
            + cp.norm1(X_vel[-1, :] - x_ref[2:4])
        )
        ineq_u = cp.sum(cp.pos(cp.abs(U) - U_LIMS[None, :]))
        ineq_v = cp.sum(cp.pos(cp.abs(X_vel[1:, :]) - VEL_LIMS[None, :]))
        ineq_tr_u = cp.sum(cp.pos(cp.abs(U - U_now) - l_u))
        ineq_tr_x = cp.sum(cp.pos(cp.abs(X_pos[1:, :] - X_now[1:, :2]) - l_pos))

        ineq_obs = 0.0
        if x_obs is not None:
            for obs_idx in range(x_obs.shape[0]):
                obs_xy = x_obs[obs_idx]
                d_now = X_now[1:, :2] - obs_xy[None, :]
                d_now_norm = np.linalg.norm(d_now, axis=1)
                lin_obs = (
                    MIN_CLEARANCE * d_now_norm
                    - cp.sum(cp.multiply(d_now, X_pos[1:, :] - obs_xy[None, :]), axis=1)
                )
                ineq_obs += cp.sum(cp.pos(lin_obs))

        objective = cp.Minimize(
            Ts * cp.sum_squares(U)
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
        if problem.status not in ("optimal", "optimal_inaccurate") or U.value is None:
            return None

        U_sol = np.asarray(U.value, dtype=float)
        self._project_speed_limits(U_sol)
        return U_sol


class DepthToObstacles:
    """
    Converts a depth image (32FC1, metres) to 3D obstacle points
    in the NED spatial frame using the coordinate transformation chain
    described in Section II-B of the paper (Eq.6, Eq.7):

      X_D^S = R_{S/B} * R_{B/L}(α) * X_{D/L}^L + R_{S/B} * X_{L/B}^B + X_B^S

    Simplified assumption: camera is rigidly mounted, facing forward (+x body),
    and the NED vehicle position/yaw are provided externally.
    """

    def __init__(self,
                 hfov_deg: float = DEPTH_HFOV,
                 vfov_deg: float = DEPTH_VFOV,
                 n_sample: int   = 500):
        self.hfov = np.deg2rad(hfov_deg)
        self.vfov = np.deg2rad(vfov_deg)
        self.n_sample = n_sample          # random pixel sub-sample for speed
        # Match mpc_vision_controller camera->body transform so both nodes
        # consume depth data with the same frame convention and camera offset.
        self.rotation_body_camera = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)
        # x500_depth mount pose (.12, .03, .242) + OakD-Lite StereoOV7251 pose
        # (0.01233, -0.03, 0.01878) from PX4 Gazebo SDFs.
        self.translation_body_camera_m = np.array([0.13233, 0.0, 0.26078], dtype=float)

    def depth_to_body_frame(self,
                             depth_img: np.ndarray) -> np.ndarray:
        """
        Convert depth image to 3D points in vehicle body frame (FRD).

        depth_img : (H, W) float32 array, metres
        Returns   : (N, 3) array of valid obstacle points in body FRD frame
        """
        H, W = depth_img.shape
        fy   = (H / 2.0) / np.tan(self.vfov / 2.0)
        fx   = (W / 2.0) / np.tan(self.hfov / 2.0)
        cx, cy = W / 2.0, H / 2.0

        # sub-sample pixels for real-time performance
        total     = H * W
        idx       = np.random.choice(total, min(self.n_sample, total), replace=False)
        rows, cols = np.unravel_index(idx, (H, W))
        depths    = depth_img[rows, cols]

        # mask valid depths
        valid   = (depths > DEPTH_MIN) & (depths < DEPTH_MAX) & np.isfinite(depths)
        depths  = depths[valid]
        rows    = rows[valid]
        cols    = cols[valid]

        if len(depths) == 0:
            return np.zeros((0, 3))

        # ── back-project to camera frame (Eq.5 generalised to 2D) ──
        # Camera: z-forward, x-right, y-down  (standard pinhole)
        Xc = (cols - cx) / fx * depths   # right
        Yc = (rows - cy) / fy * depths   # down
        Zc = depths                       # forward (into scene)

        # Build camera-frame points from the depth image back-projection, then
        # apply the same camera->body transform used by mpc_vision_controller.
        pts_cam = np.column_stack([Xc, Yc, Zc])
        return (self.rotation_body_camera @ pts_cam.T).T + self.translation_body_camera_m

    def filter_body_points(self, pts_body: np.ndarray) -> np.ndarray:
        """Remove near-body, floor-like, and far outlier points before NED projection."""
        if pts_body.shape[0] == 0:
            return np.zeros((0, 3))

        pts = np.asarray(pts_body, dtype=float)
        forward = pts[:, 0]
        lateral = pts[:, 1]
        down = pts[:, 2]

        mask = np.isfinite(pts).all(axis=1)
        mask &= forward >= DEPTH_FILTER_MIN_FORWARD_M
        mask &= forward <= DEPTH_FILTER_MAX_FORWARD_M
        mask &= np.abs(lateral) <= DEPTH_FILTER_MAX_LATERAL_M
        mask &= down >= DEPTH_FILTER_MIN_BELOW_BODY_M
        mask &= down <= DEPTH_FILTER_MAX_BELOW_BODY_M

        # Reject points too close to the ground plane when the camera is above ground.
        camera_height_above_ground_m = max(0.0, -float(self.translation_body_camera_m[2]))
        if camera_height_above_ground_m > DEPTH_FILTER_FLOOR_CLEARANCE_M:
            mask &= down <= camera_height_above_ground_m - DEPTH_FILTER_FLOOR_CLEARANCE_M

        filtered = pts[mask]
        if filtered.shape[0] == 0:
            return np.zeros((0, 3))
        return filtered

    def body_to_spatial(self,
                         pts_body: np.ndarray,
                         R_ned_body: np.ndarray,
                         pos_ned: np.ndarray) -> np.ndarray:
        """
        Transform body-frame points to NED spatial frame (Eq.7):
          X_D^S = R_{S/B} * X_D^B + X_B^S

        R_ned_body : (3,3) body-FRD to NED rotation from PX4 vehicle_odometry
        pos_ned : (3,) vehicle position in NED [m]
        Returns : (N, 3) obstacle points in NED frame
        """
        if pts_body.shape[0] == 0:
            return np.zeros((0, 3))

        pts_ned = (R_ned_body @ pts_body.T).T + pos_ned[None, :]
        return pts_ned


# ─────────────────────────────────────────────────────────────────────────────
#  ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────

class MPCUAVNode(Node):
    """
    ROS2 node integrating depth camera perception + MPC trajectory planning.

    Subscriptions:
      /depth_camera                   sensor_msgs/Image
      /fmu/out/vehicle_odometry       px4_msgs/VehicleOdometry

    Publications:
      /fmu/in/trajectory_setpoint     px4_msgs/TrajectorySetpoint
    """

    def __init__(self):
        super().__init__('mpc_optimal_uav_node')
        self.declare_parameter('enable_depth_camera', DEFAULT_ENABLE_DEPTH_CAMERA)
        self.declare_parameter('mpc_solver_backend', DEFAULT_MPC_SOLVER_BACKEND)
        self._enable_depth_camera = bool(self.get_parameter('enable_depth_camera').value)
        self._mpc_solver_backend = str(self.get_parameter('mpc_solver_backend').value)

        # ── QoS for PX4 topics ──────────────────────────────────────────────
        px4_qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability  = DurabilityPolicy.TRANSIENT_LOCAL,
            history     = HistoryPolicy.KEEP_LAST,
            depth       = 1
        )
        sensor_qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            history     = HistoryPolicy.KEEP_LAST,
            depth       = 1
        )

        # ── subscribers ─────────────────────────────────────────────────────
        if self._enable_depth_camera:
            self.create_subscription(Image,
                '/depth_camera',
                self._depth_cb,
                sensor_qos)

        self.create_subscription(VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self._state_cb,
            px4_qos)
        self.create_subscription(VehicleStatus,
            '/fmu/out/vehicle_status',
            self._status_cb,
            px4_qos)

        # ── publisher ───────────────────────────────────────────────────────
        self._sp_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            px4_qos)
        self._offboard_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            px4_qos)
        # Dataset-label publishers derived from the final solved MPC trajectory X_opt.
        self._mpc_label_path_pub = self.create_publisher(Path, '/mpc/trajectory_sequence', 10)
        self._mpc_local_prediction_path_pub = self.create_publisher(
            Path,
            LOCAL_PREDICTION_SEQUENCE_TOPIC,
            10,
        )
        self._mpc_traj_point_pub = self.create_publisher(PoseStamped, '/mpc/trajectory', 10)
        self._fresh_committed_waypoint_pub = self.create_publisher(PoseStamped, FRESH_COMMITTED_WAYPOINT_TOPIC, 10)
        self._executed_committed_waypoint_pub = self.create_publisher(
            PoseStamped,
            EXECUTED_COMMITTED_WAYPOINT_TOPIC,
            10,
        )

        # ── MPC components ──────────────────────────────────────────────────
        self._mpc       = MPCPlanner(backend=self._mpc_solver_backend)
        self._obs_map   = LocalObstacleMap(FIFO_LEN)
        self._depth2obs = DepthToObstacles()
        self._bridge    = CvBridge()
        self._debug_plotter = MPCDebugPlotter2D()

        # ── state ───────────────────────────────────────────────────────────
        self._solver_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mpc_optimal')
        self._solver_future: Future | None = None
        self._x_state   = np.zeros(8)          # [px, py, pz, vx, vy, vz, yaw, yaw_rate]
        self._yaw       = 0.0                  # vehicle yaw [rad]
        self._R_ned_body = np.eye(3)           # body FRD -> NED rotation
        self._U_warm    = np.zeros((T_horizon, 2))  # warm-start [ax, ay] for 2D SCP
        self._goal_ned  = np.array([20.0, 0.0, -5.0])  # target pos NED [m]
        self._goal_generation = 0
        self._tick_count = 0
        self._warned_pose_frame = False
        self._warned_velocity_frame = False
        self._warned_obstacle_constraints_disabled = False
        self._warned_delay_compensation_enabled = False
        self._goal_reached_announced = False
        self._nav_state = -1
        self._last_solve_duration_sec: float | None = None
        self._last_slow_solver_warn_monotonic = 0.0
        self._last_setpoint_pub_monotonic: float | None = None
        self._last_offboard_pub_monotonic: float | None = None
        self._setpoint_pub_dt = deque(maxlen=RATE_WINDOW_LEN)
        self._offboard_pub_dt = deque(maxlen=RATE_WINDOW_LEN)
        self._last_u_cmd_ned = np.zeros(4, dtype=float)
        self._latest_pos_sp_ned = np.array([0.0, 20.0, -3.0], dtype=float)
        self._latest_vel_sp_ned = np.zeros(3, dtype=float)
        self._latest_yaw_sp: float | None = YAW_TARGET_RAD
        self._latest_yaw_rate_sp: float | None = 0.0
        self.declare_parameter('disable_obstacle_constraints', DEFAULT_OBSTACLE_CONSTRAINTS_DISABLED)
        self.declare_parameter('goal_reached_threshold_m', GOAL_REACHED_THRESHOLD_M)
        self.declare_parameter('goal_reached_yaw_threshold_rad', GOAL_REACHED_YAW_THRESHOLD_RAD)
        self.declare_parameter('enable_delay_compensation', ENABLE_DELAY_COMPENSATION)
        self.declare_parameter('delay_compensation_sec', DELAY_COMPENSATION_SEC)

        # ── MPC timer (10 Hz matches paper) ─────────────────────────────────
        self.create_timer(Ts, self._mpc_step)
        self.create_timer(PUBLISH_PERIOD_SEC, self._republish_latest_setpoint)
        self.create_timer(OFFBOARD_HEARTBEAT_PERIOD_SEC, self._publish_offboard_heartbeat)
        self.create_timer(1.0 / max(DEBUG_PLOT_RATE_HZ, 0.2), self._debug_plot_step)

        if not self._enable_depth_camera:
            self.get_logger().warn(
                'enable_depth_camera=False: depth subscription is disabled, so the obstacle map '
                'will remain empty and MPC will run without depth-based obstacle avoidance. '
                'Restart with --ros-args -p enable_depth_camera:=true to restore depth input.'
            )

        self.get_logger().info(f'Using MPC solver backend: {self._mpc.backend_name}')
        self._log_status_info()

    # ── callbacks ────────────────────────────────────────────────────────────

    def _state_cb(self, msg: VehicleOdometry) -> None:
        """Update vehicle state from PX4 vehicle_odometry (position/velocity + attitude)."""
        # PX4 VehicleOdometry quaternion is [w, x, y, z].
        w, x, y, z = (float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3]))
        self._R_ned_body = self._quat_to_rotmat(w, x, y, z)
        self._yaw = self._quat_to_yaw(w, x, y, z)
        pos_ned = np.array([
            float(msg.position[0]), float(msg.position[1]), float(msg.position[2])
        ], dtype=float)
        vel_raw = np.array([
            float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])
        ], dtype=float)

        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED and not self._warned_pose_frame:
            self.get_logger().warn(
                f'Unexpected VehicleOdometry.pose_frame={msg.pose_frame} '
                f'(expected POSE_FRAME_NED={VehicleOdometry.POSE_FRAME_NED}). '
                'Planner assumes position is local NED.'
            )
            self._warned_pose_frame = True

        if msg.velocity_frame == VehicleOdometry.VELOCITY_FRAME_NED:
            vel_ned = vel_raw
        elif msg.velocity_frame in (
            VehicleOdometry.VELOCITY_FRAME_BODY_FRD,
            VehicleOdometry.VELOCITY_FRAME_FRD,
        ):
            # Convert body/world FRD-aligned velocity to NED using current attitude.
            vel_ned = self._R_ned_body @ vel_raw
            if not self._warned_velocity_frame:
                self._log_status_info()
                self._warned_velocity_frame = True
        else:
            vel_ned = vel_raw
            if not self._warned_velocity_frame:
                self.get_logger().warn(
                    f'Unexpected VehicleOdometry.velocity_frame={msg.velocity_frame}; '
                    'using raw velocity as NED.'
                )
                self._warned_velocity_frame = True

        yaw_rate = 0.0
        if hasattr(msg, 'angular_velocity') and len(msg.angular_velocity) >= 3:
            yaw_rate = float(msg.angular_velocity[2])
        self._x_state = np.array(
            [pos_ned[0], pos_ned[1], pos_ned[2], vel_ned[0], vel_ned[1], vel_ned[2], self._yaw, yaw_rate],
            dtype=float
        )

    def _depth_cb(self, msg: Image) -> None:
        """
        Process depth image → 3-D obstacle points → update local map.
        Implements Section II-B coordinate transform chain (Eq.6, Eq.7).
        """
        # ── decode depth image ──────────────────────────────────────────────
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f'Depth decode error: {e}')
            return

        # ── Step 1: pixel → body-frame 3D points  (Eq.5, Eq.6) ────────────
        pts_body = self._depth2obs.depth_to_body_frame(depth)
        pts_body = self._depth2obs.filter_body_points(pts_body)

        # ── Step 2: body frame → NED spatial frame  (Eq.7) ─────────────────
        pts_ned = self._depth2obs.body_to_spatial(
            pts_body,
            self._R_ned_body,
            self._x_state[:3]
        )

        # ── Step 3: cache in FIFO local map  (Section III-B) ────────────────
        self._obs_map.update(pts_ned)

    def _status_cb(self, msg: VehicleStatus) -> None:
        """Track PX4 nav state to detect offboard dropouts/failsafe transitions."""
        self._nav_state = int(msg.nav_state)

    # ── MPC step ─────────────────────────────────────────────────────────────

    def _mpc_step(self) -> None:
        """
        Drive the asynchronous MPC pipeline without blocking ROS timers.
        """
        self._tick_count += 1
        if self._solver_future is not None and self._solver_future.done():
            self._consume_solver_result()

        if self._solver_future is not None:
            return

        request = self._build_solver_request()
        self._solver_future = self._solver_executor.submit(self._solve_mpc_request, request)

    def _build_solver_request(self) -> dict:
        """Snapshot the current planner state for a background MPC solve."""
        request_now = self.get_clock().now()
        x0_meas = self._x_state.copy()
        x0 = x0_meas.copy()
        goal_threshold_m = max(0.0, float(self.get_parameter('goal_reached_threshold_m').value))
        x_ref_plan = np.array([self._goal_ned[0], self._goal_ned[1], 0.0, 0.0], dtype=float)
        x_ref_log = np.array([self._goal_ned[0], self._goal_ned[1], self._goal_ned[2], YAW_TARGET_RAD], dtype=float)
        dist_goal_xy = norm(x0_meas[:2] - self._goal_ned[:2])
        z_err = float(self._goal_ned[2] - x0_meas[2])
        goal_reached = bool(
            (dist_goal_xy <= goal_threshold_m) and (abs(z_err) <= Z_HOLD_THRESHOLD_M)
        )
        if goal_reached and not self._goal_reached_announced:
            self.get_logger().info(
                f'GOAL REACHED: xy_err={dist_goal_xy:.2f}m <= {goal_threshold_m:.2f}m, '
                f'z_err={abs(z_err):.2f}m <= {Z_HOLD_THRESHOLD_M:.2f}m'
            )
            self._goal_reached_announced = True
        elif (not goal_reached) and self._goal_reached_announced:
            self._goal_reached_announced = False

        if bool(self.get_parameter('enable_delay_compensation').value):
            tau = max(0.0, float(self.get_parameter('delay_compensation_sec').value))
            if tau > 0.0:
                x0[:3] = x0_meas[:3] + tau * self._last_u_cmd_ned[:3]
                x0[3:6] = self._last_u_cmd_ned[:3]
                x0[6] = x0_meas[6] + tau * self._last_u_cmd_ned[3]
                x0[7] = self._last_u_cmd_ned[3]
            if not self._warned_delay_compensation_enabled:
                self._log_status_info(x_cur=x0, x_goal=x_ref_log)
                self._warned_delay_compensation_enabled = True
        elif self._warned_delay_compensation_enabled:
            self._warned_delay_compensation_enabled = False

        x0_plan = np.array([x0[0], x0[1], x0[3], x0[4]], dtype=float)
        x_min_debug = self._obs_map.nearest_k(x0[:3], N_OBS)
        x_min = None if x_min_debug.shape[0] == 0 else np.array(x_min_debug[0], dtype=float, copy=True)
        x_obs_plan = None if x_min_debug.shape[0] == 0 else np.array(x_min_debug[:, :2], dtype=float, copy=True)
        if bool(self.get_parameter('disable_obstacle_constraints').value):
            x_min = None
            x_min_debug = np.zeros((0, 3), dtype=float)
            x_obs_plan = None
            if not self._warned_obstacle_constraints_disabled:
                self.get_logger().warn(
                    'Obstacle constraints disabled: MPC is running goal-tracking only.'
                )
                self._warned_obstacle_constraints_disabled = True

        return {
            'goal_generation': self._goal_generation,
            'goal_reached': goal_reached,
            'goal_threshold_m': goal_threshold_m,
            'dist_goal_xy': float(dist_goal_xy),
            'obs_snapshot': self._obs_map.snapshot(DEBUG_PLOT_OBS_MAX),
            'request_stamp_msg': request_now.to_msg(),
            'request_stamp_s': float(request_now.nanoseconds) * 1e-9,
            'u_warm': self._U_warm.copy(),
            'x0': x0,
            'x0_meas': x0_meas,
            'x_min_debug': x_min_debug,
            'x0_plan': x0_plan,
            'x_min': x_min,
            'x_obs_plan': x_obs_plan,
            'x_ref_log': x_ref_log,
            'x_ref_plan': x_ref_plan,
            'z_err': z_err,
        }

    def _solve_mpc_request(self, request: dict) -> dict:
        """Run the heavy MPC solve in a worker thread."""
        solve_start = time.monotonic()
        U_opt, X_opt_plan = self._mpc.solve(
            request['x0_plan'],
            request['x_ref_plan'],
            request['x_obs_plan'],
            request['u_warm'],
        )
        solve_duration_sec = time.monotonic() - solve_start

        U_warm_next = np.roll(U_opt, -1, axis=0)
        U_warm_next[-1] = U_opt[-1]

        X_opt_plot = np.zeros((X_opt_plan.shape[0], 3), dtype=float)
        X_opt_plot[:, 0:2] = X_opt_plan[:, 0:2]
        X_opt_plot[:, 2] = float(request['x0_meas'][2])

        label_waypoints_xy = X_opt_plan[1:1 + MPC_LABEL_WAYPOINT_COUNT, :2]
        if X_opt_plan.shape[0] > 1:
            vel_cmd_xy = np.array(X_opt_plan[1, 2:4], dtype=float, copy=True)
        else:
            vel_cmd_xy = np.array(
                request['x0_plan'][2:4] + Ts * U_opt[0, 0:2],
                dtype=float,
                copy=True,
            )
        np.clip(vel_cmd_xy, -V_MAX, V_MAX, out=vel_cmd_xy)

        u_cmd = np.zeros(4, dtype=float)
        u_cmd[0:2] = vel_cmd_xy
        if abs(request['z_err']) > Z_HOLD_THRESHOLD_M:
            u_cmd[2] = float(np.clip(Z_HOLD_KP * request['z_err'], -VZ_HOLD_MAX, VZ_HOLD_MAX))

        goal_z_ned = float(self._goal_ned[2])

        if label_waypoints_xy.shape[0] > 0:
            pos_sp_ned = np.array(
                [label_waypoints_xy[0, 0], label_waypoints_xy[0, 1], goal_z_ned],
                dtype=float,
            )
        else:
            pos_sp_ned = np.array(
                [float(request['x0_meas'][0]), float(request['x0_meas'][1]), goal_z_ned],
                dtype=float,
            )

        return {
            'X_opt_plot': X_opt_plot,
            'U_warm_next': U_warm_next,
            'goal_generation': request['goal_generation'],
            'goal_z_ned': goal_z_ned,
            'goal_reached': request['goal_reached'],
            'goal_threshold_m': request['goal_threshold_m'],
            'dist_goal_xy': request['dist_goal_xy'],
            'label_waypoints_xy': label_waypoints_xy,
            'obs_snapshot': request['obs_snapshot'],
            'pos_sp_ned': pos_sp_ned,
            'request_stamp_msg': request['request_stamp_msg'],
            'request_stamp_s': request['request_stamp_s'],
            'solve_duration_sec': solve_duration_sec,
            'solver_status': getattr(self._mpc, 'last_solver_status', 'unknown'),
            'u_cmd': u_cmd,
            'x0': request['x0'],
            'x_min': request['x_min'],
            'x_min_debug': request['x_min_debug'],
            'x_ref_log': request['x_ref_log'],
            'z_err': request['z_err'],
        }

    def _consume_solver_result(self) -> None:
        """Apply a completed background MPC solve on the ROS timer thread."""
        future = self._solver_future
        self._solver_future = None
        if future is None:
            return

        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f'MPC solve failed: {exc}')
            self._U_warm[:] = 0.0
            return

        if result['goal_generation'] != self._goal_generation:
            return

        self._last_solve_duration_sec = float(result['solve_duration_sec'])
        if self._last_solve_duration_sec > SOLVE_WARN_SEC:
            now_monotonic = time.monotonic()
            if now_monotonic - self._last_slow_solver_warn_monotonic >= 2.0:
                self.get_logger().warn(
                    f'MPC solve latency {1000.0 * self._last_solve_duration_sec:.0f} ms exceeds '
                    f'{1000.0 * SOLVE_WARN_SEC:.0f} ms; publish timers stay live, but MPC commands can go stale.'
                )
                self._last_slow_solver_warn_monotonic = now_monotonic

        self._U_warm = np.array(result['U_warm_next'], dtype=float, copy=True)
        label_waypoints_xy = np.array(result['label_waypoints_xy'], dtype=float, copy=False)
        if label_waypoints_xy.shape[0] == 0:
            self.get_logger().warn('MPC solve returned no future states in X_opt; keeping previous command')
            return

        stamp_msg = result['request_stamp_msg']
        self._publish_mpc_label_trajectory(label_waypoints_xy, stamp_msg, float(result['goal_z_ned']))

        self._last_u_cmd_ned = np.array(result['u_cmd'], dtype=float, copy=True)
        self._latest_pos_sp_ned = np.array(result['pos_sp_ned'], dtype=float, copy=True)
        self._latest_vel_sp_ned = np.array(result['u_cmd'][:3], dtype=float, copy=True)
        self._latest_yaw_sp = YAW_TARGET_RAD
        self._latest_yaw_rate_sp = 0.0
        self._publish_mpc_trajectory_point(self._latest_pos_sp_ned, stamp_msg)
        self._publish_waypoint_event(
            self._executed_committed_waypoint_pub,
            self._latest_pos_sp_ned,
            self.get_clock().now().to_msg(),
        )
        self._publish_waypoint_event(
            self._fresh_committed_waypoint_pub,
            self._latest_pos_sp_ned,
            stamp_msg,
        )
        self._log_setpoint_debug(
            result['x0'],
            result['x_ref_log'],
            result['x_min'],
            result['X_opt_plot'],
            self._latest_pos_sp_ned,
            self._latest_vel_sp_ned,
            solver_status=result['solver_status'],
        )
        self._debug_plotter.submit(
            x0=result['x0'],
            x_ref=np.array([self._goal_ned[0], self._goal_ned[1], self._goal_ned[2]], dtype=float),
            X_opt=result['X_opt_plot'],
            x_min=result['x_min_debug'],
            obs_pts=result['obs_snapshot'],
            goal_reached=bool(result['goal_reached']),
            dist_goal=float(result['dist_goal_xy']),
            goal_threshold_m=float(result['goal_threshold_m']),
            yaw_err=None,
            goal_yaw_threshold_rad=0.0,
        )

        dist_obs = norm(result['x_min'] - result['x0'][:3]) if result['x_min'] is not None else float('inf')
        self.get_logger().debug(
            f'dist_goal_xy={result["dist_goal_xy"]:.2f}m  z_err={result["z_err"]:.2f}m  dist_obs={dist_obs:.2f}m  '
            f'u=[{self._last_u_cmd_ned[0]:.2f},{self._last_u_cmd_ned[1]:.2f},{self._last_u_cmd_ned[2]:.2f}] m/s '
            f'yaw={result["x0"][6]:.2f} rad yaw_rate_cmd={self._last_u_cmd_ned[3]:.2f} rad/s')

    def _debug_plot_step(self) -> None:
        """Drive visible matplotlib updates from the main thread at a low rate."""
        self._debug_plotter.draw_latest()

    @staticmethod
    def _publish_waypoint_event(pub, pos_ned: np.ndarray, stamp_msg) -> None:
        """Publish a single NED waypoint event for downstream dataset consumers."""
        msg = PoseStamped()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = 'ned'
        msg.pose.position.x = float(pos_ned[0])
        msg.pose.position.y = float(pos_ned[1])
        msg.pose.position.z = float(pos_ned[2])
        msg.pose.orientation.w = 1.0
        pub.publish(msg)

    def _publish_setpoint(
        self,
        pos_ned: np.ndarray,
        vel_ned: np.ndarray,
        yaw_sp: float | None = None,
        yaw_rate_sp: float | None = None,
        stamp_us: int | None = None,
    ) -> None:
        """
        Publish TrajectorySetpoint (position + velocity feedforward) to PX4.
        PX4 TrajectorySetpoint uses NED frame with NaN for unused fields.
        """
        sp = TrajectorySetpoint()
        sp.timestamp = int(stamp_us) if stamp_us is not None else self.get_clock().now().nanoseconds // 1000   # µs

        # Position is optional. In velocity-only mode keep it NaN so PX4 tracks
        # the velocity feedforward command directly in NED.
        if USE_VELOCITY_ONLY_SETPOINT:
            sp.position[0] = float('nan')
            sp.position[1] = float('nan')
            sp.position[2] = float('nan')
        else:
            sp.position[0] = float(pos_ned[0])
            sp.position[1] = float(pos_ned[1])
            sp.position[2] = float(pos_ned[2])

        # velocity feedforward [m/s] NED
        sp.velocity[0] = float(vel_ned[0])
        sp.velocity[1] = float(vel_ned[1])
        sp.velocity[2] = float(vel_ned[2])

        # acceleration not controlled by MPC → NaN
        sp.acceleration[0] = float('nan')
        sp.acceleration[1] = float('nan')
        sp.acceleration[2] = float('nan')
        sp.yaw             = float('nan') if yaw_sp is None else float(yaw_sp)
        sp.yawspeed        = float('nan') if yaw_rate_sp is None else float(yaw_rate_sp)

        self._sp_pub.publish(sp)

    def _record_timer_interval(self, last_attr: str, samples: deque) -> None:
        """Track actual timer cadence for lightweight publish-rate diagnostics."""
        now = time.monotonic()
        last = getattr(self, last_attr)
        if last is not None:
            samples.append(now - last)
        setattr(self, last_attr, now)

    @staticmethod
    def _timer_rate_hz(samples: deque) -> float | None:
        """Return average callback rate in Hz from recent timer intervals."""
        if not samples:
            return None
        mean_dt = float(np.mean(samples))
        if mean_dt <= 1e-6:
            return None
        return 1.0 / mean_dt

    def _publish_offboard_heartbeat(self) -> None:
        """Keep PX4 in velocity offboard mode for NED velocity command tracking."""
        self._record_timer_interval('_last_offboard_pub_monotonic', self._offboard_pub_dt)
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self._offboard_mode_pub.publish(msg)

    def _republish_latest_setpoint(self) -> None:
        """Republish the latest setpoint at a stable rate independent of MPC solve time."""
        self._record_timer_interval('_last_setpoint_pub_monotonic', self._setpoint_pub_dt)
        self._publish_setpoint(
            self._latest_pos_sp_ned,
            self._latest_vel_sp_ned,
            yaw_sp=self._latest_yaw_sp,
            yaw_rate_sp=self._latest_yaw_rate_sp,
        )

    def _publish_mpc_label_trajectory(self, waypoints_xy: np.ndarray, stamp_msg, goal_z_ned: float) -> None:
        """
        Publish the open-loop 5-step MPC prediction from a single solve.
        These are not guaranteed to be the 5 waypoints eventually executed after future replans.
        """
        path = Path()
        path.header.stamp = stamp_msg
        path.header.frame_id = 'ned'

        if waypoints_xy.shape[0] < MPC_LABEL_WAYPOINT_COUNT:
            self.get_logger().warn(
                f'MPC label trajectory shorter than expected: {waypoints_xy.shape[0]} < {MPC_LABEL_WAYPOINT_COUNT}'
            )

        for idx, p in enumerate(waypoints_xy):
            pose = PoseStamped()
            pose.header.stamp = stamp_msg
            pose.header.frame_id = 'ned'
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            pose.pose.position.z = float(goal_z_ned)
            # Orientation unused for label trajectory; identity quaternion for completeness.
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self._mpc_label_path_pub.publish(path)
        self._mpc_local_prediction_path_pub.publish(path)

    def _publish_mpc_trajectory_point(self, pos_ned: np.ndarray, stamp_msg) -> None:
        """Publish the executed first waypoint for compatibility with /mpc/trajectory consumers."""
        msg = PoseStamped()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = 'ned'
        msg.pose.position.x = float(pos_ned[0])
        msg.pose.position.y = float(pos_ned[1])
        msg.pose.position.z = float(pos_ned[2])
        msg.pose.orientation.w = 1.0
        self._mpc_traj_point_pub.publish(msg)

    def _log_setpoint_debug(self,
                            x0: np.ndarray,
                            x_ref: np.ndarray,
                            x_min: np.ndarray | None,
                            X_opt: np.ndarray,
                            pos_sp_ned: np.ndarray,
                            vel_sp_ned: np.ndarray,
                            solver_status: str | None = None) -> None:
        """Throttle MPC/PX4 setpoint logs to debug frame/sign mismatches."""
        if self._tick_count % SETPOINT_LOG_EVERY_N != 0:
            return
        self._log_status_info(x_cur=x0, x_goal=x_ref, solver_status=solver_status)

    def set_goal(self, x: float, y: float, z: float) -> None:
        """Update navigation goal in NED [m]. z negative = up in NED."""
        self._goal_ned = np.array([x, y, z])
        self._goal_generation += 1
        self._obs_map.clear()
        self._U_warm[:] = 0.0
        self._last_u_cmd_ned[:] = 0.0
        self._latest_vel_sp_ned[:] = 0.0
        self._goal_reached_announced = False
        self._log_status_info()

    @staticmethod
    def _gazebo_enu_to_ned(x_enu: float, y_enu: float, z_enu: float) -> np.ndarray:
        """
        Convert Gazebo world ENU -> PX4 local NED.
        ENU: x=East, y=North, z=Up
        NED: x=North, y=East, z=Down
        """
        return np.array([y_enu, x_enu, -z_enu], dtype=float)

    @staticmethod
    def _ned_to_gazebo_enu(x_ned: float, y_ned: float, z_ned: float) -> np.ndarray:
        """
        Convert PX4 local NED -> Gazebo world ENU.
        NED: x=North, y=East, z=Down
        ENU: x=East, y=North, z=Up
        """
        return np.array([y_ned, x_ned, -z_ned], dtype=float)

    def set_goal_gazebo(self, x: float, y: float, z: float) -> None:
        """Update navigation goal from Gazebo world coordinates (ENU) [m]."""
        goal_ned = self._gazebo_enu_to_ned(x, y, z)
        self.set_goal(float(goal_ned[0]), float(goal_ned[1]), float(goal_ned[2]))
        self._log_status_info()

    def _log_status_info(
        self,
        x_cur: np.ndarray | None = None,
        x_goal: np.ndarray | None = None,
        solver_status: str | None = None,
    ) -> None:
        """Unified INFO log: current state, goal state, and error."""
        if x_cur is None:
            x_cur = self._x_state
        if x_goal is None:
            x_goal = np.array([self._goal_ned[0], self._goal_ned[1], self._goal_ned[2], YAW_TARGET_RAD], dtype=float)
        if solver_status is None:
            solver_status = getattr(self._mpc, 'last_solver_status', 'unknown')
        yaw_cur = float(x_cur[6]) if x_cur.shape[0] >= 7 else 0.0
        yaw_goal = float(x_goal[3]) if x_goal.shape[0] >= 4 else YAW_TARGET_RAD
        pos_err_vec = x_goal[:3] - x_cur[:3]
        pos_err = float(norm(pos_err_vec))
        yaw_err = float(yaw_goal - yaw_cur)
        vel_odom = x_cur[3:6] if x_cur.shape[0] >= 6 else np.zeros(3, dtype=float)
        vel_cmd = self._last_u_cmd_ned[:3]
        setpoint_rate_hz = self._timer_rate_hz(self._setpoint_pub_dt)
        offboard_rate_hz = self._timer_rate_hz(self._offboard_pub_dt)
        solve_ms = None if self._last_solve_duration_sec is None else (1000.0 * self._last_solve_duration_sec)
        setpoint_rate_txt = 'n/a' if setpoint_rate_hz is None else f'{setpoint_rate_hz:.1f}'
        offboard_rate_txt = 'n/a' if offboard_rate_hz is None else f'{offboard_rate_hz:.1f}'
        solve_ms_txt = 'n/a' if solve_ms is None else f'{solve_ms:.0f}'
        obs_snapshot = self._obs_map.snapshot(DEBUG_PLOT_OBS_MAX)
        obs_count = int(obs_snapshot.shape[0]) if obs_snapshot.ndim == 2 else 0
        nearest_obs = self._obs_map.nearest(x_cur[:3])
        nearest_obs_dist = float(norm(nearest_obs - x_cur[:3])) if nearest_obs is not None else float('inf')
        nearest_obs_txt = 'n/a' if not np.isfinite(nearest_obs_dist) else f'{nearest_obs_dist:.2f}'
        self.get_logger().info(
            f'current=[{x_cur[0]:.2f},{x_cur[1]:.2f},{x_cur[2]:.2f},{yaw_cur:.2f}] '
            f'vel_odom=[{vel_odom[0]:.2f},{vel_odom[1]:.2f},{vel_odom[2]:.2f}] '
            f'vel_cmd=[{vel_cmd[0]:.2f},{vel_cmd[1]:.2f},{vel_cmd[2]:.2f}] '
            f'goal=[{x_goal[0]:.2f},{x_goal[1]:.2f},{x_goal[2]:.2f},{yaw_goal:.2f}] '
            f'Error=[{pos_err_vec[0]:.2f},{pos_err_vec[1]:.2f},{pos_err_vec[2]:.2f},{yaw_err:.2f}] '
            f'ror_norm={pos_err:.2f} Solver result={solver_status} '
            f'sp_rate={setpoint_rate_txt}Hz offboard_rate={offboard_rate_txt}Hz solve_ms={solve_ms_txt} '
            f'obs_count={obs_count} nearest_obs={nearest_obs_txt}m'
        )

    @staticmethod
    def _quat_to_rotmat(w: float, x: float, y: float, z: float) -> np.ndarray:
        """Quaternion (w,x,y,z) -> body-to-NED rotation matrix."""
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
    def _quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
        """Extract yaw from quaternion (w,x,y,z) for debug/diagnostics."""
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-9:
            return 0.0
        w, x, y, z = w / n, x / n, y / n, z / n
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

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

    # Example Gazebo world goal (ENU): x=East, y=North, z=Up
    # This maps to PX4 local NED internally.
    node.set_goal_gazebo(32.91, 0.00, 2.5)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
