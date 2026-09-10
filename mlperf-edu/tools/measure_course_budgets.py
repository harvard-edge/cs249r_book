#!/usr/bin/env python3
"""Measure one-run classroom resource budgets for every functional path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlperf.fingerprint import detect_hardware  # noqa: E402
from mlperf.registry import load_registry  # noqa: E402


SCHEMA = "mlperf-edu-course-budget-measurement/0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def process_tree_rss(process: psutil.Process) -> int:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    rss = 0
    for item in processes:
        try:
            rss += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss


def run_with_peak_rss(command: list[str], *, cwd: Path) -> tuple[int, float, int, str]:
    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "MLPERF_EDU_NO_BROWSER": "1"},
    )
    monitored = psutil.Process(child.pid)
    peak_rss = 0
    while child.poll() is None:
        peak_rss = max(peak_rss, process_tree_rss(monitored))
        time.sleep(0.01)
    output, _ = child.communicate()
    peak_rss = max(peak_rss, process_tree_rss(monitored))
    return child.returncode, time.perf_counter() - started, peak_rss, output


def find_workload_report(output_dir: Path, workload: str) -> dict[str, Any]:
    candidates = sorted(output_dir.glob(f"{workload}*_min_report.json"))
    if len(candidates) != 1:
        return {}
    try:
        value = json.loads(candidates[0].read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def git_record() -> dict[str, Any]:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--", "mlperf-edu"],
        cwd=ROOT.parent,
        text=True,
    )
    return {"sha": sha, "clean": not bool(status.strip())}


def available_devices() -> list[str]:
    devices = ["cpu"]
    try:
        import torch

        if torch.backends.mps.is_available():
            devices.append("mps")
        elif torch.cuda.is_available():
            devices.append("cuda")
    except ImportError:
        pass
    return devices


def measure(
    *,
    workloads: list[str],
    devices: list[str],
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for device in devices:
        for workload in workloads:
            output_dir = output_root / device / workload
            if output_dir.exists():
                shutil.rmtree(output_dir)
            command = [
                sys.executable,
                "-m",
                "mlperf.edu_cli",
                "run",
                "--workload",
                workload,
                "--profile",
                "min",
                "--device",
                device,
                "--output-dir",
                str(output_dir),
                "--no-open-report",
            ]
            returncode, wall_seconds, peak_rss_bytes, output = run_with_peak_rss(
                command, cwd=ROOT
            )
            report = find_workload_report(output_dir, workload)
            rows.append(
                {
                    "workload": workload,
                    "profile": "min",
                    "requested_device": device,
                    "executed_device": report.get("device_executed"),
                    "status": report.get("status") or "missing_report",
                    "returncode": returncode,
                    "wall_seconds": wall_seconds,
                    "peak_rss_bytes": peak_rss_bytes,
                    "artifact_disk_bytes": directory_bytes(output_dir),
                    "required_download_bytes": 0,
                    "network_access_required": False,
                    "command": command[2:],
                    "stdout_tail": output[-2000:],
                }
            )
    hardware = detect_hardware()
    lockfile = ROOT / "uv.lock"
    return {
        "schema": SCHEMA,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "profile": "min",
            "acceptance_runs": 1,
            "stability_claim": False,
            "network_policy": "functional min paths use bundled deterministic inputs",
        },
        "source": git_record(),
        "course_image": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "uv_lock_sha256": sha256_file(lockfile),
        },
        "hardware": hardware,
        "runs": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", action="append", default=None)
    parser.add_argument("--device", action="append", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/private/tmp/mlperf-edu-course-budget-runs"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("conformance_results/course-budgets-latest.json"),
    )
    args = parser.parse_args(argv)
    registry = load_registry(ROOT / "registry")
    selected_workloads = args.workload or sorted(registry)
    unknown = sorted(set(selected_workloads) - set(registry))
    if unknown:
        parser.error("unknown workload(s): " + ", ".join(unknown))
    selected_devices = args.device or available_devices()
    invalid_devices = sorted(set(selected_devices) - {"cpu", "mps", "cuda"})
    if invalid_devices:
        parser.error("unsupported device(s): " + ", ".join(invalid_devices))

    result = measure(
        workloads=selected_workloads,
        devices=selected_devices,
        output_root=args.output_root.resolve(),
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    failed = [
        row
        for row in result["runs"]
        if row["returncode"] != 0 or row["status"] != "passed"
    ]
    print(f"wrote {len(result['runs'])} measurement(s) to {args.result}")
    if failed:
        print(f"{len(failed)} measurement(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
