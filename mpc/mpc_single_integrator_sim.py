#!/usr/bin/env python3
"""
Standalone MPC simulation for 3D single-integrator dynamics.

State: x = [px, py, pz]
Control: u = [vx, vy, vz]
Dynamics: x_{k+1} = x_k + dt * u_k

This script does not use ROS runtime data. It simulates closed-loop MPC
to reach a goal while avoiding spherical obstacles, then plots results.
"""

import argparse
import math

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
try:
    # Register "3d" projection for add_subplot(..., projection="3d")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL_3D = True
except Exception:
    HAS_MPL_3D = False


def parse_vec3(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got: '{text}'")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def parse_obstacles(obstacle_specs):
    """
    Parse obstacle list from strings in form: "x,y,z,r".
    """
    obstacles = []
    for spec in obstacle_specs:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid obstacle '{spec}'. Use x,y,z,r")
        obstacles.append(
            (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
        )
    return obstacles


class MPC3DSimulator:
    def __init__(
        self,
        start,
        goal,
        obstacles,
        dt=0.2,
        horizon=12,
        control_horizon=6,
        v_max=2.0,
        safety_distance=0.6,
        max_steps=250,
        goal_tolerance=0.25,
    ):
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.position = np.array(start, dtype=float)
        self.obstacles = list(obstacles)

        self.dt = float(dt)
        self.N = int(horizon)
        self.M = int(control_horizon)
        self.v_max = float(v_max)
        self.safety_distance = float(safety_distance)
        self.max_steps = int(max_steps)
        self.goal_tolerance = float(goal_tolerance)

        self.Q = np.diag([20.0, 20.0, 20.0])
        self.R = np.diag([0.2, 0.2, 0.2])

        self.available_solvers = cp.installed_solvers()
        if "ECOS" in self.available_solvers:
            self.conic_solver = cp.ECOS
        elif "SCS" in self.available_solvers:
            self.conic_solver = cp.SCS
        else:
            self.conic_solver = None

        self.time_hist = []
        self.pos_hist = [self.position.copy()]
        self.err_hist = []
        self.effort_hist = []

    def solve_mpc(self):
        """
        Receding-horizon MPC with linearized obstacle constraints.
        """
        x = cp.Variable((3, self.N + 1))
        u = cp.Variable((3, self.M))

        cost = 0
        constraints = [x[:, 0] == self.position]

        for k in range(self.N):
            cost += cp.quad_form(x[:, k] - self.goal, self.Q)
            if k < self.M:
                cost += cp.quad_form(u[:, k], self.R)
        cost += cp.quad_form(x[:, self.N] - self.goal, self.Q * 10.0)

        for k in range(self.M):
            constraints.append(x[:, k + 1] == x[:, k] + self.dt * u[:, k])
            constraints.append(cp.norm(u[:, k], 2) <= self.v_max)

        for k in range(self.M, self.N):
            constraints.append(x[:, k + 1] == x[:, k] + self.dt * u[:, self.M - 1])

        for ox, oy, oz, radius in self.obstacles:
            center = np.array([ox, oy, oz], dtype=float)
            safe_radius = float(radius) + self.safety_distance

            ref = self.position - center
            ref_norm = np.linalg.norm(ref)
            if ref_norm < 1e-6:
                ref = np.array([1.0, 0.0, 0.0], dtype=float)
                ref_norm = 1.0
            normal = ref / ref_norm

            for k in range(1, self.N + 1):
                # First-order conservative half-space from current linearization point.
                constraints.append(
                    normal @ (x[:, k] - self.position) + ref_norm >= safe_radius
                )

        problem = cp.Problem(cp.Minimize(cost), constraints)
        if self.conic_solver is None:
            return None

        try:
            problem.solve(solver=self.conic_solver, verbose=False, warm_start=True)
        except Exception:
            return None

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None
        if u.value is None:
            return None
        return np.array(u[:, 0].value, dtype=float).reshape(3)

    def fallback_control(self):
        """
        Safe fallback if solver fails: move to goal with speed cap.
        """
        direction = self.goal - self.position
        dist = np.linalg.norm(direction)
        if dist < 1e-9:
            return np.zeros(3, dtype=float)
        speed = min(self.v_max * 0.5, dist / max(self.dt, 1e-6))
        return direction / dist * speed

    def run(self):
        if self.conic_solver is None:
            raise RuntimeError(
                f"No conic solver found in CVXPY. Installed: {self.available_solvers}"
            )

        reached = False
        for step in range(self.max_steps):
            err = float(np.linalg.norm(self.goal - self.position))
            if err <= self.goal_tolerance:
                reached = True
                break

            u = self.solve_mpc()
            if u is None:
                u = self.fallback_control()

            self.position = self.position + self.dt * u

            self.time_hist.append(step * self.dt)
            self.pos_hist.append(self.position.copy())
            self.err_hist.append(err)
            self.effort_hist.append(float(np.linalg.norm(u)))

        return reached

    def plot_results(self, reached):
        pos = np.array(self.pos_hist)
        t = np.array(self.time_hist)
        err = np.array(self.err_hist)
        effort = np.array(self.effort_hist)

        fig = plt.figure(figsize=(12, 8))
        if HAS_MPL_3D:
            ax3d = fig.add_subplot(2, 2, 1, projection="3d")
        else:
            ax3d = fig.add_subplot(2, 2, 1)
        ax_err = fig.add_subplot(2, 2, 2)
        ax_effort = fig.add_subplot(2, 2, 4)

        if HAS_MPL_3D:
            ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", linewidth=2, label="Trajectory")
            ax3d.scatter(
                [self.start[0]], [self.start[1]], [self.start[2]], c="green", s=70, marker="o", label="Start"
            )
            ax3d.scatter(
                [self.goal[0]], [self.goal[1]], [self.goal[2]], c="red", s=90, marker="*", label="Goal"
            )

            u = np.linspace(0, 2 * np.pi, 24)
            v = np.linspace(0, np.pi, 12)
            for ox, oy, oz, radius in self.obstacles:
                rs = radius + self.safety_distance
                xs = ox + rs * np.outer(np.cos(u), np.sin(v))
                ys = oy + rs * np.outer(np.sin(u), np.sin(v))
                zs = oz + rs * np.outer(np.ones_like(u), np.cos(v))
                ax3d.plot_wireframe(
                    xs, ys, zs, rstride=2, cstride=2, linewidth=0.5, color="gray", alpha=0.5
                )
            ax3d.set_title("3D MPC Trajectory")
            ax3d.set_zlabel("Z (m)")
        else:
            ax3d.plot(pos[:, 0], pos[:, 1], "b-", linewidth=2, label="Trajectory (XY)")
            ax3d.scatter([self.start[0]], [self.start[1]], c="green", s=70, marker="o", label="Start")
            ax3d.scatter([self.goal[0]], [self.goal[1]], c="red", s=90, marker="*", label="Goal")
            for ox, oy, _oz, radius in self.obstacles:
                rs = radius + self.safety_distance
                circle = plt.Circle((ox, oy), rs, color="gray", fill=False, alpha=0.6)
                ax3d.add_patch(circle)
            ax3d.set_title("MPC Trajectory (XY fallback; 3D unavailable)")
            ax3d.axis("equal")

        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.legend(loc="best")

        ax_err.plot(t, err, color="tab:orange", linewidth=2)
        ax_err.set_title("MPC Error vs Time")
        ax_err.set_xlabel("Time (s)")
        ax_err.set_ylabel("||goal - position|| (m)")
        ax_err.grid(True, alpha=0.3)

        ax_effort.plot(t, effort, color="tab:green", linewidth=2)
        ax_effort.set_title("Control Effort vs Time")
        ax_effort.set_xlabel("Time (s)")
        ax_effort.set_ylabel("||u|| (m/s)")
        ax_effort.grid(True, alpha=0.3)

        status = "Reached goal" if reached else "Goal not reached"
        fig.suptitle(status, fontsize=12)
        fig.tight_layout()
        plt.show()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Standalone MPC simulation (3D single integrator + obstacle avoidance)"
    )
    parser.add_argument("--start", default="0,0,0", help="Start position x,y,z")
    parser.add_argument("--goal", default="12,8,-2", help="Goal position x,y,z")
    parser.add_argument(
        "--obstacle",
        action="append",
        default=[],
        help="Obstacle sphere x,y,z,r (repeatable)",
    )
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--control_horizon", type=int, default=6)
    parser.add_argument("--v_max", type=float, default=2.0)
    parser.add_argument("--safety_distance", type=float, default=0.6)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--goal_tolerance", type=float, default=0.25)
    return parser


def main(args=None):
    parser = build_arg_parser()
    ns = parser.parse_args(args=args)

    start = parse_vec3(ns.start)
    goal = parse_vec3(ns.goal)

    if ns.obstacle:
        obstacles = parse_obstacles(ns.obstacle)
    else:
        obstacles = [
            (4.0, 2.0, -1.0, 1.2),
            (7.0, 5.5, -1.5, 1.0),
            (9.5, 3.5, -2.0, 0.9),
        ]

    sim = MPC3DSimulator(
        start=start,
        goal=goal,
        obstacles=obstacles,
        dt=ns.dt,
        horizon=ns.horizon,
        control_horizon=ns.control_horizon,
        v_max=ns.v_max,
        safety_distance=ns.safety_distance,
        max_steps=ns.max_steps,
        goal_tolerance=ns.goal_tolerance,
    )

    reached = sim.run()
    sim.plot_results(reached)


if __name__ == "__main__":
    main()
