#!/usr/bin/env python3
"""
MPC-based UAV Trajectory Planner for ROS2
==========================================
Based on: Shim et al., "Autonomous Exploration in Unknown Urban Environments
          for Unmanned Aerial Vehicles", AIAA GN&C 2005

System:
  - Sensor  : /depth_camera  (sensor_msgs/Image, encoding=32FC1, metres)
  - Output  : /fmu/in/trajectory_setpoint  (px4_msgs/TrajectorySetpoint, NED frame)
  - State   : /fmu/out/vehicle_local_position (px4_msgs/VehicleLocalPosition)

MPC Formulation (discrete-time, horizon T steps)
-------------------------------------------------
State       : x(k) = [x, y, z, vx, vy, vz]^T  (NED, metres / m·s⁻¹)
Control     : u(k) = [vx_ref, vy_ref, vz_ref]^T (reference velocities)
Dynamics    : x(k+1) = x(k) + Ts * u(k)         (Eq.9 from paper, single integrator)

Cost function per step  (Eq.8 + Eq.10):
  q(x,u) = q_trk(x) + q_obs(x)

  q_trk(x) = 0.5 * (y_ref - x)^T Q (y_ref - x)       tracking penalty
  q_obs(x) = K_obs / (||x_S - x_min||² + ε)           obstacle repulsion (Eq.10)

Gradient search: gradient of total cost wrt U is computed analytically
  and Adam-style gradient descent is used for real-time optimisation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import numpy as np
from numpy.linalg import norm
import cv2
from cv_bridge import CvBridge
from collections import deque
import os

# ROS2 message types
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import TrajectorySetpoint, VehicleOdometry

# ─────────────────────────────────────────────────────────────────────────────
# MPC PARAMETERS  (tune these)
# ─────────────────────────────────────────────────────────────────────────────
Ts          = 0.1          # sampling time [s]
T_horizon   = 10           # MPC prediction horizon [steps]
V_MAX       = 3.0          # max velocity magnitude [m/s]
K_obs       = 100.0         # obstacle repulsion gain  (Eq.10)
EPS         = 0.5         # singularity guard [m²]     (ε in paper)
N_OBS       = 1            # nearest-point method: N=1 (Section III-A)
Q_diag      = np.array([2.0, 2.0, 2.0, 0.1, 0.1, 0.1])   # tracking weight Q
MAX_ITER    = 30           # gradient-descent iterations per MPC cycle
LR          = 0.05         # Adam learning rate
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
DEBUG_PLOT_ENABLED = True  # non-blocking 2D MPC debug plots (optional)
DEBUG_PLOT_RATE_HZ = 4.0   # plot refresh rate, decoupled from MPC loop
DEBUG_PLOT_OBS_MAX = 200   # max local obstacle points shown in debug plot
DEBUG_PATH_HISTORY_LEN = 300
MPC_LABEL_WAYPOINT_COUNT = 5  # publish next 5 waypoints from final solved X_opt for dataset labels
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

    Rendering is triggered by a separate ROS timer callback (main thread), so
    the MPC control timer only publishes lightweight snapshots.
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
        ax_xy.legend(loc='best', fontsize=8)

        self._artists['path_xz'], = ax_xz.plot([], [], 'k-', lw=1.5, alpha=0.8, label='actual path')
        self._artists['pred_xz'], = ax_xz.plot([], [], 'b.-', lw=1.5, ms=5, label='MPC pred')
        self._artists['cur_xz'], = ax_xz.plot([], [], 'go', ms=7, label='current')
        self._artists['goal_xz'], = ax_xz.plot([], [], 'r*', ms=12, label='goal')
        self._artists['nearest_xz'], = ax_xz.plot([], [], 'mx', ms=8, mew=2, label='nearest obs')
        ax_xz.legend(loc='best', fontsize=8)

        self._fig.tight_layout()

    def submit(self,
               x0: np.ndarray,
               x_ref: np.ndarray,
               X_opt: np.ndarray,
               x_min: np.ndarray | None,
               obs_pts: np.ndarray | None = None) -> None:
        """Store latest MPC snapshot for async plotting."""
        if not self._enabled:
            return

        x0_pos = np.array(x0[:3], dtype=float, copy=True)
        x_ref_pos = np.array(x_ref[:3], dtype=float, copy=True)
        x_pred = np.array(X_opt[:, :3], dtype=float, copy=True)
        x_min_copy = None if x_min is None else np.array(x_min[:3], dtype=float, copy=True)
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
        }

    def draw_latest(self) -> None:
        """Render the latest snapshot (call from main thread / ROS timer)."""
        if not self._enabled:
            return
        if self._latest is None:
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
        if x_min is not None:
            self._artists['nearest_xy'].set_data([x_min[0]], [x_min[1]])
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
        if x_min is not None:
            self._artists['nearest_xz'].set_data([x_min[0]], [x_min[2]])
        else:
            self._artists['nearest_xz'].set_data([], [])

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
    Real-time NMPC trajectory generator.

    Optimises a sequence of reference velocities U = [u(1),...,u(T)]
    over the horizon using gradient descent (Adam optimiser) to minimise
    the combined tracking + obstacle-repulsion cost (Eq.2, Eq.8, Eq.10).
    """

    def __init__(self):
        self.Q   = np.diag(Q_diag)              # 6×6 tracking weight
        self._reset_adam()

    def _reset_adam(self):
        """Reset Adam optimiser state."""
        self._m  = np.zeros((T_horizon, 3))     # 1st moment
        self._v2 = np.zeros((T_horizon, 3))     # 2nd moment
        self._t  = 0

    # ── cost functions ────────────────────────────────────────────────────────

    @staticmethod
    def _q_trk(x: np.ndarray, y_ref: np.ndarray) -> float:
        """
        Tracking cost  Eq.(8):  q_trk = 0.5*(y_ref - x)^T Q (y_ref - x)
        x, y_ref : (6,) state vectors [pos(3) vel(3)]
        """
        e = y_ref - x
        return 0.5 * float(e @ Q_diag * e)     # diagonal Q → element-wise

    @staticmethod
    def _q_obs(x_pos: np.ndarray, x_min: np.ndarray) -> float:
        """
        Obstacle repulsion cost  Eq.(10):
          q_obs = K_obs / (||x_S - x_min||² + ε)
        x_pos, x_min : (3,) position vectors [NED]
        """
        d2 = float(norm(x_pos - x_min) ** 2)
        return K_obs / (d2 + EPS)

    @staticmethod
    def _grad_q_obs(x_pos: np.ndarray, x_min: np.ndarray) -> np.ndarray:
        """
        Gradient of q_obs wrt x_pos:
          ∂q_obs/∂x_pos = -2*K_obs*(x_pos - x_min) / (||x_pos - x_min||² + ε)²
        Returns (3,) vector.
        """
        diff = x_pos - x_min
        d2   = float(norm(diff) ** 2)
        denom = (d2 + EPS) ** 2
        return -2.0 * K_obs * diff / denom

    # ── forward rollout ───────────────────────────────────────────────────────

    @staticmethod
    def _rollout(x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Propagate state forward T steps using simplified dynamics Eq.(9):
          x(k+1) = x(k) + Ts * u(k)
        x0 : (6,) initial state  [px, py, pz, vx, vy, vz]  NED
        U  : (T, 3) control sequence of reference velocities
        Returns X : (T+1, 6) state trajectory
        """
        X    = np.zeros((T_horizon + 1, 6))
        X[0] = x0
        for k in range(T_horizon):
            X[k+1, :3] = X[k, :3] + Ts * U[k]          # position update
            X[k+1, 3:] = U[k]                            # velocity = control
        return X

    # ── optimisation ─────────────────────────────────────────────────────────

    def solve(self,
              x0: np.ndarray,
              x_ref: np.ndarray,
              x_min: np.ndarray | None,
              U_warm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run gradient-descent MPC optimisation.

        Parameters
        ----------
        x0      : (6,) current vehicle state [NED]
        x_ref   : (6,) desired goal state   [NED]
        x_min   : (3,) nearest obstacle position, or None
        U_warm  : (T, 3) warm-start control sequence

        Returns
        -------
        U_opt   : (T, 3) optimised control sequence
        X_opt   : (T+1, 6) optimised state trajectory
        """
        U = U_warm.copy()
        β1, β2, η, δ = 0.9, 0.999, LR, 1e-8   # Adam hyper-parameters

        for _ in range(MAX_ITER):
            X   = self._rollout(x0, U)
            grad = np.zeros_like(U)             # (T, 3)

            for k in range(T_horizon):
                xk     = X[k+1]
                xk_pos = xk[:3]

                # ── gradient of tracking cost wrt u(k) ──────────────────
                # ∂q_trk/∂u(k):  q_trk = 0.5*(y_ref - xk)^T Q_pos (y_ref - xk)
                # ∂xk_pos/∂u(j) = Ts * I  for j <= k  (triangular chain rule)
                e_pos   = xk_pos - x_ref[:3]
                g_trk   = Q_diag[:3] * e_pos * Ts   # (3,)  ##tracking error

                # ── gradient of obstacle cost wrt u(k) ──────────────────
                g_obs = np.zeros(3)
                if x_min is not None:
                    g_obs = self._grad_q_obs(xk_pos, x_min) * Ts  # chain rule

                grad[k] = g_trk + g_obs

            # Adam update  ##own algo not tensorflow Adam
            self._t  += 1
            self._m   = β1 * self._m  + (1 - β1) * grad
            self._v2  = β2 * self._v2 + (1 - β2) * grad**2
            m_hat  = self._m  / (1 - β1**self._t)
            v2_hat = self._v2 / (1 - β2**self._t)
            U     -= η * m_hat / (np.sqrt(v2_hat) + δ)

            # clip to velocity limits
            for k in range(T_horizon):
                spd = norm(U[k])
                if spd > V_MAX:
                    U[k] *= V_MAX / spd

        X_opt = self._rollout(x0, U)
        self._reset_adam()          # fresh Adam state for next MPC call
        return U, X_opt


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
        super().__init__('mpc_uav_node')

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
        self.create_subscription(Image,
            '/depth_camera',
            self._depth_cb,
            sensor_qos)

        self.create_subscription(VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self._state_cb,
            px4_qos)

        # ── publisher ───────────────────────────────────────────────────────
        self._sp_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            px4_qos)
        # Dataset-label publishers derived from the final solved MPC trajectory X_opt.
        self._mpc_label_path_pub = self.create_publisher(Path, '/mpc/trajectory_sequence', 10)
        self._mpc_traj_point_pub = self.create_publisher(PoseStamped, '/mpc/trajectory', 10)

        # ── MPC components ──────────────────────────────────────────────────
        self._mpc       = MPCPlanner()
        self._obs_map   = LocalObstacleMap(FIFO_LEN)
        self._depth2obs = DepthToObstacles()
        self._bridge    = CvBridge()
        self._debug_plotter = MPCDebugPlotter2D()

        # ── state ───────────────────────────────────────────────────────────
        self._x_state   = np.zeros(6)          # [px, py, pz, vx, vy, vz] NED
        self._yaw       = 0.0                  # vehicle yaw [rad]
        self._R_ned_body = np.eye(3)           # body FRD -> NED rotation
        self._U_warm    = np.zeros((T_horizon, 3))  # warm-start velocities
        self._goal_ned  = np.array([20.0, 0.0, -5.0])  # target pos NED [m]
        self._tick_count = 0
        self._warned_pose_frame = False
        self._warned_velocity_frame = False

        # ── MPC timer (10 Hz matches paper) ─────────────────────────────────
        self.create_timer(Ts, self._mpc_step)
        self.create_timer(1.0 / max(DEBUG_PLOT_RATE_HZ, 0.2), self._debug_plot_step)

        self.get_logger().info(
            f'MPC UAV Node started | Ts={Ts}s | horizon={T_horizon} | '
            f'K_obs={K_obs} | FIFO={FIFO_LEN}')

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
                self.get_logger().info(
                    f'Converting VehicleOdometry velocity from frame={msg.velocity_frame} '
                    f'to NED using current attitude'
                )
                self._warned_velocity_frame = True
        else:
            vel_ned = vel_raw
            if not self._warned_velocity_frame:
                self.get_logger().warn(
                    f'Unexpected VehicleOdometry.velocity_frame={msg.velocity_frame}; '
                    'using raw velocity as NED.'
                )
                self._warned_velocity_frame = True

        self._x_state = np.concatenate([pos_ned, vel_ned])

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

        # ── Step 2: body frame → NED spatial frame  (Eq.7) ─────────────────
        pts_ned = self._depth2obs.body_to_spatial(
            pts_body,
            self._R_ned_body,
            self._x_state[:3]
        )

        # ── Step 3: cache in FIFO local map  (Section III-B) ────────────────
        self._obs_map.update(pts_ned)

    # ── MPC step ─────────────────────────────────────────────────────────────

    def _mpc_step(self) -> None:
        """
        Main MPC loop, called at Ts Hz.

        1.  Query local obstacle map for nearest obstacle (Eq.11)
        2.  Run gradient-search MPC optimisation (Eq.2, Eq.8, Eq.10)
        3.  Extract first control u*(1) and publish as TrajectorySetpoint
        """
        x0 = self._x_state.copy()
        self._tick_count += 1

        # build reference state: goal position, zero velocity
        x_ref = np.concatenate([self._goal_ned, np.zeros(3)])

        # ── Step 1: find nearest obstacle  (Eq.11) ──────────────────────────
        x_min = self._obs_map.nearest(x0[:3])

        # ── Step 2: MPC solve ────────────────────────────────────────────────
        U_opt, X_opt = self._mpc.solve(x0, x_ref, x_min, self._U_warm)

        # warm-start shift (receding horizon)  redduce convergence time by starting next MPC solve from previous solution
        self._U_warm      = np.roll(U_opt, -1, axis=0)
        self._U_warm[-1]  = U_opt[-1]

        # Label waypoints come from the *final* solved trajectory and are timestamped
        # in the same MPC tick as the executed first setpoint.
        label_waypoints_ned = X_opt[1:1 + MPC_LABEL_WAYPOINT_COUNT, :3]
        if label_waypoints_ned.shape[0] == 0:
            self.get_logger().warn('MPC solve returned no future states in X_opt; skipping publish')
            return

        now_ros = self.get_clock().now()
        stamp_msg = now_ros.to_msg()
        stamp_us = now_ros.nanoseconds // 1000
        self._publish_mpc_label_trajectory(label_waypoints_ned, stamp_msg)

        # ── Step 3: publish first optimal control u*(1) ─────────────────────
        u_cmd = U_opt[0]                    # [vx_ref, vy_ref, vz_ref] NED
        pos_sp_ned = label_waypoints_ned[0]
        self._publish_mpc_trajectory_point(pos_sp_ned, stamp_msg)
        self._publish_setpoint(pos_sp_ned, u_cmd, stamp_us=stamp_us)
        self._log_setpoint_debug(x0, x_ref, x_min, X_opt, pos_sp_ned, u_cmd)
        self._debug_plotter.submit(
            x0=x0,
            x_ref=x_ref,
            X_opt=X_opt,
            x_min=x_min,
            obs_pts=self._obs_map.snapshot(DEBUG_PLOT_OBS_MAX),
        )

        # diagnostics
        dist_goal = norm(x0[:3] - self._goal_ned)
        dist_obs  = norm(x_min - x0[:3]) if x_min is not None else float('inf')
        self.get_logger().debug(
            f'dist_goal={dist_goal:.2f}m  dist_obs={dist_obs:.2f}m  '
            f'u=[{u_cmd[0]:.2f},{u_cmd[1]:.2f},{u_cmd[2]:.2f}] m/s')

    def _debug_plot_step(self) -> None:
        """Separate timer loop for debug plotting (main thread-safe)."""
        self._debug_plotter.draw_latest()

    def _publish_setpoint(self, pos_ned: np.ndarray, vel_ned: np.ndarray, stamp_us: int | None = None) -> None:
        """
        Publish TrajectorySetpoint (position + velocity feedforward) to PX4.
        PX4 TrajectorySetpoint uses NED frame with NaN for unused fields.
        """
        sp = TrajectorySetpoint()
        sp.timestamp = int(stamp_us) if stamp_us is not None else self.get_clock().now().nanoseconds // 1000   # µs

        # position setpoint [m] NED
        sp.position[0] = float(pos_ned[0])
        sp.position[1] = float(pos_ned[1])
        sp.position[2] = float(pos_ned[2])

        # velocity feedforward [m/s] NED
        sp.velocity[0] = float(vel_ned[0])
        sp.velocity[1] = float(vel_ned[1])
        sp.velocity[2] = float(vel_ned[2])

        # acceleration / yaw not controlled by MPC → NaN
        sp.acceleration[0] = float('nan')
        sp.acceleration[1] = float('nan')
        sp.acceleration[2] = float('nan')
        sp.yaw             = float('nan')
        sp.yawspeed        = float('nan')

        self._sp_pub.publish(sp)

    def _publish_mpc_label_trajectory(self, waypoints_ned: np.ndarray, stamp_msg) -> None:
        """
        Publish timestamped MPC label trajectory (next 5 waypoints) for dataset generation.
        Waypoints are taken directly from the final solved X_opt[1:1+K, :3].
        """
        path = Path()
        path.header.stamp = stamp_msg
        path.header.frame_id = 'ned'

        if waypoints_ned.shape[0] < MPC_LABEL_WAYPOINT_COUNT:
            self.get_logger().warn(
                f'MPC label trajectory shorter than expected: {waypoints_ned.shape[0]} < {MPC_LABEL_WAYPOINT_COUNT}'
            )

        for idx, p in enumerate(waypoints_ned):
            pose = PoseStamped()
            pose.header.stamp = stamp_msg
            pose.header.frame_id = 'ned'
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            pose.pose.position.z = float(p[2])
            # Orientation unused for label trajectory; identity quaternion for completeness.
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self._mpc_label_path_pub.publish(path)

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
                            vel_sp_ned: np.ndarray) -> None:
        """Throttle MPC/PX4 setpoint logs to debug frame/sign mismatches."""
        if self._tick_count % SETPOINT_LOG_EVERY_N != 0:
            return

        pos_cur_enu = self._ned_to_gazebo_enu(*x0[:3])
        pos_sp_enu = self._ned_to_gazebo_enu(*pos_sp_ned)
        goal_enu = self._ned_to_gazebo_enu(*x_ref[:3])
        vel_sp_enu = self._ned_to_gazebo_enu(*vel_sp_ned)
        nearest_obs = 'None' if x_min is None else (
            f'NED[{x_min[0]:.2f},{x_min[1]:.2f},{x_min[2]:.2f}] '
            f'ENU[{self._ned_to_gazebo_enu(*x_min)[0]:.2f},'
            f'{self._ned_to_gazebo_enu(*x_min)[1]:.2f},'
            f'{self._ned_to_gazebo_enu(*x_min)[2]:.2f}]'
        )
        z_seq = ','.join(f'{p[2]:.2f}' for p in X_opt[1:min(4, len(X_opt))])

        self.get_logger().info(
            'MPC->PX4 TrajectorySetpoint '
            f'pose_frame=NED vel_frame_assumed=NED | '
            f'cur_NED=[{x0[0]:.2f},{x0[1]:.2f},{x0[2]:.2f}] '
            f'goal_NED=[{x_ref[0]:.2f},{x_ref[1]:.2f},{x_ref[2]:.2f}] '
            f'pub_pos_NED=[{pos_sp_ned[0]:.2f},{pos_sp_ned[1]:.2f},{pos_sp_ned[2]:.2f}] '
            f'pub_vel_NED=[{vel_sp_ned[0]:.2f},{vel_sp_ned[1]:.2f},{vel_sp_ned[2]:.2f}] | '
            f'cur_ENU=[{pos_cur_enu[0]:.2f},{pos_cur_enu[1]:.2f},{pos_cur_enu[2]:.2f}] '
            f'goal_ENU=[{goal_enu[0]:.2f},{goal_enu[1]:.2f},{goal_enu[2]:.2f}] '
            f'pub_pos_ENU=[{pos_sp_enu[0]:.2f},{pos_sp_enu[1]:.2f},{pos_sp_enu[2]:.2f}] '
            f'pub_vel_ENU=[{vel_sp_enu[0]:.2f},{vel_sp_enu[1]:.2f},{vel_sp_enu[2]:.2f}] | '
            f'pred_z_NED=[{z_seq}] nearest_obs={nearest_obs}'
        )

    def set_goal(self, x: float, y: float, z: float) -> None:
        """Update navigation goal in NED [m]. z negative = up in NED."""
        self._goal_ned = np.array([x, y, z])
        self._obs_map.clear()
        self._U_warm[:] = 0.0
        self.get_logger().info(f'New goal set: N={x:.1f} E={y:.1f} D={z:.1f}')

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
        self.get_logger().info(
            f'Gazebo goal (ENU): E={x:.1f} N={y:.1f} U={z:.1f} -> '
            f'NED: N={goal_ned[0]:.1f} E={goal_ned[1]:.1f} D={goal_ned[2]:.1f}'
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
        self._debug_plotter.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MPCUAVNode()

    # Example Gazebo world goal (ENU): x=East, y=North, z=Up
    # This maps to PX4 local NED internally.
    node.set_goal_gazebo(32.91, 0.00, 2.0)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
