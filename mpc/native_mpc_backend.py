from __future__ import annotations

import numpy as np

try:
    from llm_drone import _mpc_native
except Exception:
    _mpc_native = None


def native_backend_available() -> bool:
    return _mpc_native is not None


def native_osqp_available() -> bool:
    return _mpc_native is not None and hasattr(_mpc_native, 'solve_osqp')


def solve_subgradient(
    x0: np.ndarray,
    x_ref: np.ndarray,
    x_obs: np.ndarray | None,
    u_warm: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, str]:
    if _mpc_native is None:
        raise RuntimeError('Native MPC backend is not available in this build')

    result = _mpc_native.solve_subgradient(
        np.asarray(x0, dtype=float),
        np.asarray(x_ref, dtype=float),
        None if x_obs is None else np.asarray(x_obs, dtype=float),
        np.asarray(u_warm, dtype=float),
        config,
    )
    return (
        np.asarray(result['U_opt'], dtype=float),
        np.asarray(result['X_opt'], dtype=float),
        str(result['solver_status']),
    )


def solve_osqp(
    x0: np.ndarray,
    x_ref: np.ndarray,
    x_obs: np.ndarray | None,
    u_warm: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, str]:
    if _mpc_native is None or not hasattr(_mpc_native, 'solve_osqp'):
        raise RuntimeError('Native OSQP backend is not available in this build')

    result = _mpc_native.solve_osqp(
        np.asarray(x0, dtype=float),
        np.asarray(x_ref, dtype=float),
        None if x_obs is None else np.asarray(x_obs, dtype=float),
        np.asarray(u_warm, dtype=float),
        config,
    )
    return (
        np.asarray(result['U_opt'], dtype=float),
        np.asarray(result['X_opt'], dtype=float),
        str(result['solver_status']),
    )
