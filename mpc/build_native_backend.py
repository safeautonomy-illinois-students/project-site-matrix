#!/usr/bin/env python3
from __future__ import annotations

import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    src = root / "src" / "mpc_native.cpp"
    if not src.is_file():
        print(f"error: source file not found: {src}", file=sys.stderr)
        return 1

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    include_py = sysconfig.get_config_var("INCLUDEPY")
    cflags = (sysconfig.get_config_var("CFLAGS") or "").split()
    ldflags = (sysconfig.get_config_var("LDFLAGS") or "").split()

    try:
        import pybind11  # type: ignore
    except Exception as exc:
        print(f"error: pybind11 is required to build the native backend: {exc}", file=sys.stderr)
        return 1

    if not ext_suffix or not include_py:
        print("error: could not determine Python build flags", file=sys.stderr)
        return 1

    out = root / f"_mpc_native{ext_suffix}"
    cmd = [
        "c++",
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        "-fPIC",
        str(src),
        "-o",
        str(out),
        f"-I{include_py}",
        f"-I{pybind11.get_include()}",
        "-ldl",
    ]
    cmd.extend(cflags)
    cmd.extend(ldflags)

    print("building native backend:")
    print(" ".join(shlex.quote(part) for part in cmd))
    completed = subprocess.run(cmd, cwd=root)
    if completed.returncode != 0:
        return completed.returncode

    print(f"built {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
