#!/usr/bin/env python3
"""Build a wheel from clean setuptools state and reject retired modules."""

from __future__ import annotations

import argparse
import hashlib
import json
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
        "mlperf/runners/agent.py",
        "mlperf/runners/dlrm.py",
        "mlperf/runners/extended.py",
        "mlperf/runners/slm.py",
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

        evidence_index_path = ROOT / "reference_results" / "index.json"
        evidence_index_bytes = evidence_index_path.read_bytes()
        evidence_index = json.loads(evidence_index_bytes)
        exact_assets = {
            "mlperf_edu/workloads.yaml": ROOT / "src" / "mlperf_edu" / "workloads.yaml",
            "mlperf_edu/datasets.yaml": ROOT / "src" / "mlperf_edu" / "datasets.yaml",
            "mlperf_edu/reference_results/index.json": ROOT
            / "src"
            / "mlperf_edu"
            / "reference_results"
            / "index.json",
            "mlperf_edu/reference_results/source_lock.json": ROOT
            / "src"
            / "mlperf_edu"
            / "reference_results"
            / "source_lock.json",
        }
        expected_reference_members = {
            "mlperf_edu/reference_results/index.json",
            "mlperf_edu/reference_results/source_lock.json",
            *{
                f"mlperf_edu/{entry['path']}"
                for entry in evidence_index.get("summaries") or []
            },
        }
        required = {*exact_assets, *expected_reference_members}
        missing = sorted(required.difference(members))
        if missing:
            raise RuntimeError(
                "wheel is missing required packaged asset(s): " + ", ".join(missing)
            )
        actual_reference_members = {
            member
            for member in members
            if member.startswith("mlperf_edu/reference_results/")
            and member.endswith(".json")
        }
        extras = sorted(actual_reference_members.difference(expected_reference_members))
        if extras:
            raise RuntimeError(
                "wheel contains stale unindexed reference evidence: "
                + ", ".join(extras)
            )

        for member, source in exact_assets.items():
            if archive.read(member) != source.read_bytes():
                raise RuntimeError(
                    f"wheel asset {member} does not match the source mirror"
                )
        if (
            archive.read("mlperf_edu/datasets.yaml")
            != (ROOT / "datasets.yaml").read_bytes()
        ):
            raise RuntimeError("packaged dataset catalog does not match datasets.yaml")
        if (
            archive.read("mlperf_edu/reference_results/index.json")
            != evidence_index_bytes
        ):
            raise RuntimeError(
                "wheel reference index does not match the canonical index"
            )
        lock_data = archive.read("mlperf_edu/reference_results/source_lock.json")
        expected_lock_digest = (evidence_index.get("source_lock") or {}).get("sha256")
        actual_lock_digest = "sha256:" + hashlib.sha256(lock_data).hexdigest()
        if actual_lock_digest != expected_lock_digest:
            raise RuntimeError("wheel reference source-lock digest mismatch")
        for entry in evidence_index.get("summaries") or []:
            member = f"mlperf_edu/{entry['path']}"
            data = archive.read(member)
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry.get("evidence_sha256"):
                raise RuntimeError(f"wheel reference summary digest mismatch: {member}")
            source = ROOT / "src" / "mlperf_edu" / str(entry["path"])
            if data != source.read_bytes():
                raise RuntimeError(
                    f"wheel reference summary does not match source mirror: {member}"
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
