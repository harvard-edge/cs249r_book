#!/usr/bin/env python3
"""Run and verify the noninteractive path used by Tutorial 01."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_tutorial_benchmark(output_dir: Path) -> dict[str, Any]:
    """Run Tutorial 01 and verify its expected artifacts."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    pythonpath = str(PROJECT_ROOT / "src")
    if environment.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{environment['PYTHONPATH']}"
    environment["PYTHONPATH"] = pythonpath

    command = [
        sys.executable,
        "-m",
        "mlperf_edu",
        "run",
        "--workload",
        "time-series-forecasting",
        "--profile",
        "min",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"benchmark command failed ({completed.returncode}): {detail}"
        )

    report_path = output_dir / "time-series-forecasting_min_report.json"
    manifest_path = output_dir / "time-series-forecasting_min.provd.json"
    html_path = report_path.with_suffix(".html")
    csv_path = report_path.with_suffix(".csv")
    missing = [
        path
        for path in (report_path, manifest_path, html_path, csv_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "benchmark did not produce expected artifacts: "
            + ", ".join(str(path) for path in missing)
        )

    report = json.loads(report_path.read_text())
    if report.get("status") != "passed":
        raise RuntimeError(f"workload status is not passed: {report.get('status')}")
    if not isinstance(report.get("metrics"), dict) or not report["metrics"]:
        raise RuntimeError("workload report has no metrics")
    if not isinstance(report.get("run_fingerprint"), dict):
        raise RuntimeError("workload report has no run fingerprint")

    verification = subprocess.run(
        [sys.executable, "-m", "mlperf_edu", "verify", str(manifest_path)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if verification.returncode:
        detail = verification.stderr.strip() or verification.stdout.strip()
        raise RuntimeError(
            f"provenance verification failed ({verification.returncode}): {detail}"
        )

    aggregate_paths = sorted(output_dir.glob("mlperf_edu_min_*.json"))
    if not aggregate_paths:
        raise RuntimeError("benchmark did not produce an aggregate JSON report")
    return {
        "command": command,
        "stdout": completed.stdout,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "html_path": html_path,
        "csv_path": csv_path,
        "aggregate_path": aggregate_paths[-1],
        "report": report,
        "verification_stdout": verification.stdout,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Tutorial 01.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tutorials" / "_runs" / "01-smoke",
    )
    args = parser.parse_args(argv)
    try:
        artifacts = run_tutorial_benchmark(args.output_dir)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"TUTORIAL 01 SMOKE FAIL: {exc}", file=sys.stderr)
        return 1
    print("MLPerf EDU Tutorial 01")
    print(f"  report: {artifacts['report_path']}")
    print(f"  provenance: {artifacts['manifest_path']}")
    print("  provenance verification: passed")
    print("TUTORIAL 01 SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
