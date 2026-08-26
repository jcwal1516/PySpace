#!/usr/bin/env python3
"""Install a wheel into an isolated virtual environment and smoke-test it."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def smoke_test_wheel(wheel: Path) -> None:
    """Install the wheel with dependencies and exercise import plus CLI entry point."""
    with tempfile.TemporaryDirectory(prefix="pyspace-wheel-smoke-") as temporary:
        environment = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        scripts = environment / ("Scripts" if sys.platform == "win32" else "bin")
        python = scripts / ("python.exe" if sys.platform == "win32" else "python")
        command = scripts / ("pyspace.exe" if sys.platform == "win32" else "pyspace")
        _run([str(python), "-m", "pip", "install", str(wheel.resolve())])
        _run(
            [
                str(python),
                "-c",
                "import pyspace; assert pyspace.__version__ == '0.1.0'; assert pyspace.calc_vol([1,1,1],[3,3,3]) == 7",
            ]
        )
        _run([str(command), "--version"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args(argv)
    smoke_test_wheel(args.wheel)
    print(f"Clean-wheel smoke test passed: {args.wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
