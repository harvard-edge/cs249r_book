#!/usr/bin/env python3
"""Build a wheel from clean setuptools state and reject retired modules."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STALE_BUILD_PATHS = (Path("build"), Path("src/mlperf_edu.egg-info"))
FORBIDDEN_WHEEL_MEMBERS = frozenset(
    {
        "mlperf/cli.py",
        "mlperf/core.py",
        "mlperf/datasets.py",
        "mlperf/grader.py",
        "mlperf/hardware.py",
        "mlperf/plotting.py",
        "mlperf/provenance.py",
        "mlperf/report.py",
        "mlperf/reference/cloud/nanogpt_core.py",
        "mlperf/reference/cloud/nanogpt_infer.py",
        "mlperf/reference/mobile/mobilenet_infer.py",
        "mlperf_edu/core.py",
    }
)


def project_path(relative_path: Path) -> Path:
    """Resolve one cleanup target and refuse paths outside the project root."""
    target = (ROOT / relative_path).resolve()
    target.relative_to(ROOT.resolve())
    return target


def clean_stale_build_state() -> None:
    """Remove only known, ignored setuptools staging directories."""
    for relative_path in STALE_BUILD_PATHS:
        target = project_path(relative_path)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def verify_wheel(wheel_path: Path) -> None:
    """Fail if a wheel contains a retired module from stale build state."""
    with zipfile.ZipFile(wheel_path) as archive:
        members = set(archive.namelist())
    forbidden = sorted(FORBIDDEN_WHEEL_MEMBERS.intersection(members))
    if forbidden:
        raise RuntimeError(
            "wheel contains retired module(s), likely from stale build state: "
            + ", ".join(forbidden)
        )
    required = {"mlperf_edu/workloads.yaml", "mlperf_edu/slm_quality_prompts.json"}
    missing = sorted(required.difference(members))
    if missing:
        raise RuntimeError(
            "wheel is missing required packaged asset(s): " + ", ".join(missing)
        )


def build_wheel(out_dir: Path) -> Path:
    clean_stale_build_state()
    resolved_out = out_dir if out_dir.is_absolute() else ROOT / out_dir
    try:
        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--clear",
                "--no-create-gitignore",
                "--out-dir",
                str(resolved_out),
                ".",
            ],
            cwd=ROOT,
            check=True,
        )
        wheels = sorted(resolved_out.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                f"expected exactly one wheel in {resolved_out}, found {len(wheels)}"
            )
        verify_wheel(wheels[0])
        return wheels[0]
    finally:
        clean_stale_build_state()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheel_path = build_wheel(args.out_dir)
    print(f"verified clean wheel: {wheel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
