from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_native_backend():
    so_candidates: list[Path] = []
    base_dir = Path(__file__).resolve().parent
    search_roots = [
        base_dir,
        base_dir / 'build',
        base_dir / 'install',
        base_dir / 'src',
    ]
    patterns = (
        '_mpc_native*.so',
        '**/_mpc_native*.so',
    )
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            so_candidates.extend(root.glob(pattern))

    so_candidates.sort(key=lambda p: (0 if p.parent == base_dir else 1, len(str(p))))

    seen: set[Path] = set()
    for so_path in so_candidates:
        so_path = so_path.resolve()
        if so_path in seen or not so_path.is_file():
            continue
        seen.add(so_path)
        try:
            spec = importlib.util.spec_from_file_location('_mpc_native', so_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception:
            continue

    return None


_mpc_native = _load_native_backend()


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
