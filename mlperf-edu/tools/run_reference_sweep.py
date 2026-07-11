#!/usr/bin/env python3
"""Produce reviewable multi-seed reference evidence through the product path.

Each seed runs in a fresh process using the public ``MLPERF_EDU_MAX_SEED``
interface. The tool never patches framework RNG functions. A run is invalid when
the report or provenance manifest records a different seed, when its manifest
does not verify, when grading fails, or when a public candidate uses a data mode
outside the contract for its score-bearing or performance-bearing status.

Evidence is written to a new attempt directory and never overwritten. Every run
artifact is SHA-256 indexed, and the final evidence summary receives a separate
unauthenticated SHA-256 digest sidecar. These digests detect changes; they do not
authenticate who produced the evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOOL_NAME = "run_reference_sweep.py"
TOOL_VERSION = "2.3.0"
TOOL_ID = f"tools/{TOOL_NAME} v{TOOL_VERSION}"
SCORE_PUBLIC_DATA_MODES = frozenset({"real"})
PERFORMANCE_PUBLIC_DATA_MODES = frozenset({"real", "checkpoint-backed", "local-prompt"})
PUBLIC_STATUSES = frozenset({"score-bearing", "performance-bearing"})
DEFAULT_TIMEOUT_SECONDS = 7200.0
DEFAULT_OUTPUT_DIR = Path.home() / ".mlperf-edu" / "reference_runs"
MAX_LINEAGE_PACKAGE_MEMBERS = 256
MAX_LINEAGE_PACKAGE_BYTES = 2 * 1024**3
MAX_LINEAGE_PACKAGE_INDEX_BYTES = 1 << 20
NANOGPT_LINEAGE_ENV = {
    "checkpoint": "MLPERF_EDU_NANOGPT_CHECKPOINT",
    "report": "MLPERF_EDU_NANOGPT_TRAIN_REPORT",
    "manifest": "MLPERF_EDU_NANOGPT_TRAIN_MANIFEST",
}


_CHILD_BOOTSTRAP = r"""
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def resolve_workload(registry, workload_id, variant):
    if variant:
        for workload in registry.values():
            if (getattr(workload, "canonical_workload", None) == workload_id
                    and getattr(workload, "variant", None) == variant):
                return workload
        raise KeyError(f"no variant {variant!r} for workload {workload_id!r}")
    if workload_id in registry:
        return registry[workload_id]
    raise KeyError(f"workload {workload_id!r} not found")


def artifact_index(report, report_path, manifest_path, exports):
    candidates = {"report": report_path, "provenance": manifest_path}
    for role, path in exports.items():
        candidates[f"report_{role}"] = Path(path)
    for role, value in (report.get("artifacts") or {}).items():
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = report_path.parent / path
        candidates[str(role)] = path
    indexed = []
    seen = set()
    for role, path in sorted(candidates.items()):
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        indexed.append({
            "role": role,
            "path": str(path),
            "sha256": sha256_file(path),
            "n_bytes": path.stat().st_size,
        })
    return indexed


def main():
    args_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    args = json.loads(args_path.read_text())
    root = Path(args["root"])
    sys.path.insert(0, str(root / "src"))
    seed = int(args["seed"])
    os.environ.pop("MLPERF_EDU_SEED", None)
    os.environ.pop("MLPERF_EDU_SLM_SEED", None)
    os.environ["MLPERF_EDU_MAX_SEED"] = str(seed)
    if args.get("device"):
        os.environ["MLPERF_EDU_DEVICE"] = str(args["device"])

    result = {
        "requested_seed": seed,
        "workload_id": args["workload_id"],
        "profile": args["profile"],
        "execution_ok": False,
        "evidence_valid": False,
    }
    try:
        from mlperf.edu_cli import (
            attach_run_fingerprints,
            enrich_report_for_display,
            grade_manifest,
            metric_key_for_quality,
            run_workload,
            update_measurement_manifest,
            write_report_exports,
        )
        from mlperf.manifest import verify_provd
        from mlperf.registry import load_registry

        registry = load_registry()
        workload = resolve_workload(registry, args["workload_id"], args.get("variant"))
        output_dir = Path(args["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        report = run_workload(workload, args["profile"], output_dir)
        wall_seconds = time.perf_counter() - started

        artifacts = report.get("artifacts") or {}
        report_path = Path(str(artifacts.get("report", ""))).resolve()
        manifest_path = Path(str(artifacts.get("provenance", ""))).resolve()
        if not report_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("runner did not produce both report and provenance artifacts")

        # Match the normal CLI post-run path before verification and grading.
        report = json.loads(report_path.read_text())
        enrich_report_for_display(report, registry)
        attach_run_fingerprints(report)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        update_measurement_manifest(report, report_path, manifest_path)
        exports = write_report_exports(report, report_path, open_report=False)

        manifest = json.loads(manifest_path.read_text())
        verification = verify_provd(manifest_path, repo_root=root)
        grade = grade_manifest(manifest_path)
        quality = report.get("quality") or {}
        metrics = report.get("metrics") or {}
        functional_metric = quality.get("metric") or getattr(workload, "quality_metric", None)
        measurement_protocol = workload.raw.get("measurement_protocol") or {}
        performance_metric = (
            measurement_protocol.get("primary_metric")
            if workload.public_status == "performance-bearing"
            else None
        )
        declared_metric = performance_metric or functional_metric
        metric_key = metric_key_for_quality(declared_metric, metrics)
        metric_value = metrics.get(metric_key) if metric_key else None
        if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
            metric_value = None
        elif not math.isfinite(float(metric_value)):
            metric_value = None
        else:
            metric_value = float(metric_value)

        report_seed = report.get("seed")
        manifest_seed = ((manifest.get("leaves") or {}).get("rng") or {}).get("seed")
        data_mode = report.get("data_mode")
        execution_backend = str(report.get("backend") or "")
        requested_device = args.get("device")
        invalid_reasons = []
        if report.get("status") != "passed":
            invalid_reasons.append(f"report status is {report.get('status')!r}, not 'passed'")
        if not verification.all_ok:
            invalid_reasons.append("provenance verification failed")
        if not grade.get("passed"):
            invalid_reasons.append("submission grading failed")
        if report_seed != seed:
            invalid_reasons.append(f"report recorded seed {report_seed!r}, requested {seed}")
        if manifest_seed != seed:
            invalid_reasons.append(f"manifest recorded seed {manifest_seed!r}, requested {seed}")
        if metric_value is None:
            invalid_reasons.append(f"declared reference metric {declared_metric!r} has no finite numeric report value")
        if workload.public_status == "performance-bearing" and quality.get("target_met") is not True:
            invalid_reasons.append("performance-bearing functional check did not pass")
        if requested_device and requested_device.lower() not in execution_backend.lower():
            invalid_reasons.append(
                f"report execution backend {execution_backend!r} does not match requested device {requested_device!r}"
            )
        if args["evidence_tier"] == "public-candidate" and data_mode not in args["allowed_data_modes"]:
            invalid_reasons.append(
                f"public candidate data_mode {data_mode!r} is not allowed for {workload.public_status} evidence"
            )
        review_contract = report.get("review_contract") or {}
        if args["evidence_tier"] == "public-candidate" and review_contract.get("status") != "passed":
            invalid_reasons.append("report review contract did not pass")

        fingerprint = report.get("run_fingerprint") or {}
        hardware = fingerprint.get("hardware") or {}
        execution = fingerprint.get("execution") or {}
        result.update({
            "execution_ok": True,
            "evidence_valid": not invalid_reasons,
            "invalid_reasons": invalid_reasons,
            "status": report.get("status"),
            "report_recorded_seed": report_seed,
            "manifest_recorded_seed": manifest_seed,
            "quality_metric_declared": declared_metric,
            "functional_metric_declared": functional_metric,
            "reference_metric_role": "performance" if performance_metric else "quality",
            "quality_metric_key": metric_key,
            "quality_value": metric_value,
            "quality_target_met": quality.get("target_met"),
            "wall_seconds": wall_seconds,
            "backend": execution_backend or None,
            "hardware_backend": hardware.get("backend"),
            "chip": hardware.get("chip"),
            "fingerprint_backends": execution.get("backends"),
            "data_mode": data_mode,
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
            "manifest_verified": verification.all_ok,
            "verification_checks": [
                {"check": name, "ok": ok, "detail": detail}
                for name, ok, detail in verification.checks
            ],
            "grade": {
                "passed": grade.get("passed"),
                "status": grade.get("status"),
                "metric": grade.get("metric"),
                "value": grade.get("value"),
                "target": grade.get("target"),
                "target_met": grade.get("target_met"),
            },
            "artifacts": artifact_index(report, report_path, manifest_path, exports),
        })
    except Exception as exc:
        result.update({
            "execution_ok": False,
            "evidence_valid": False,
            "invalid_reasons": [f"{type(exc).__name__}: {exc}"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result.get("evidence_valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""


def parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(token.strip()) for token in raw.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--seeds must be comma-separated integers"
        ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must include at least one seed")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must not contain duplicates")
    return seeds


def aggregate(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "stdev": None,
        }
    return {
        "count": len(clean),
        "median": statistics.median(clean),
        "mean": statistics.fmean(clean),
        "min": min(clean),
        "max": max(clean),
        "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class LineagePackageError(ValueError):
    """Raised when a NanoGPT training package is unsafe or unverifiable."""


def _safe_archive_member_name(name: str) -> bool:
    """Return whether *name* is a strict, portable archive-relative path."""
    if not name or "\\" in name or "\x00" in name or name.endswith("/"):
        return False
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name:
        return False
    return ":" not in parts[0]


def _preflight_lineage_archive(package_path: Path) -> dict[str, Any]:
    """Reject unsafe ZIP structure before any package verifier extracts it."""
    try:
        with zipfile.ZipFile(package_path) as zf:
            infos = zf.infolist()
            names = [info.filename for info in infos]
            if len(infos) > MAX_LINEAGE_PACKAGE_MEMBERS:
                raise LineagePackageError(
                    f"lineage package contains {len(infos)} members; "
                    f"limit is {MAX_LINEAGE_PACKAGE_MEMBERS}"
                )
            if len(names) != len(set(names)):
                raise LineagePackageError(
                    "lineage package contains duplicate member names"
                )
            total_size = 0
            for info in infos:
                if not _safe_archive_member_name(info.filename):
                    raise LineagePackageError(
                        f"unsafe lineage package member path: {info.filename!r}"
                    )
                if info.flag_bits & 0x1:
                    raise LineagePackageError(
                        f"encrypted lineage package member is unsupported: {info.filename!r}"
                    )
                file_type = stat.S_IFMT((info.external_attr >> 16) & 0xFFFF)
                if info.is_dir() or file_type not in {0, stat.S_IFREG}:
                    raise LineagePackageError(
                        f"lineage package member is not a regular file: {info.filename!r}"
                    )
                total_size += info.file_size
            if total_size > MAX_LINEAGE_PACKAGE_BYTES:
                raise LineagePackageError(
                    f"lineage package expands to {total_size} bytes; "
                    f"limit is {MAX_LINEAGE_PACKAGE_BYTES}"
                )

            try:
                index_info = zf.getinfo("package_index.json")
            except KeyError as exc:
                raise LineagePackageError(
                    "lineage package is missing package_index.json"
                ) from exc
            if index_info.file_size > MAX_LINEAGE_PACKAGE_INDEX_BYTES:
                raise LineagePackageError("lineage package index is unexpectedly large")
            try:
                index = json.loads(zf.read(index_info))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LineagePackageError(
                    "lineage package index is not valid UTF-8 JSON"
                ) from exc
    except (OSError, zipfile.BadZipFile) as exc:
        raise LineagePackageError(f"cannot read lineage package: {exc}") from exc

    if not isinstance(index, dict) or index.get("schema") != "mlperf-edu-package/0.2":
        raise LineagePackageError(
            "lineage package must use schema mlperf-edu-package/0.2"
        )
    if index.get("workload") != "nanogpt-train":
        raise LineagePackageError("lineage package must contain a nanogpt-train result")
    included = index.get("included_files")
    if not isinstance(included, list) or not included:
        raise LineagePackageError("lineage package index has no included files")

    indexed_names: list[str] = []
    info_by_name = {info.filename: info for info in infos}
    for item in included:
        if not isinstance(item, dict):
            raise LineagePackageError("lineage package index entries must be objects")
        archive_name = item.get("path")
        if not isinstance(archive_name, str) or not _safe_archive_member_name(
            archive_name
        ):
            raise LineagePackageError(
                f"lineage package index has an unsafe path: {archive_name!r}"
            )
        indexed_names.append(archive_name)
        info = info_by_name.get(archive_name)
        if info is None:
            raise LineagePackageError(
                f"lineage package index references a missing member: {archive_name}"
            )
        if item.get("n_bytes") != info.file_size:
            raise LineagePackageError(
                f"lineage package index size does not match {archive_name}"
            )
    if len(indexed_names) != len(set(indexed_names)):
        raise LineagePackageError("lineage package index contains duplicate paths")
    expected_names = {"package_index.json", *indexed_names}
    if set(names) != expected_names:
        extras = sorted(set(names) - expected_names)
        missing = sorted(expected_names - set(names))
        raise LineagePackageError(
            f"lineage package/index coverage mismatch; extras={extras}, missing={missing}"
        )

    manifest_name = index.get("manifest") or index.get("source_manifest")
    if (
        not isinstance(manifest_name, str)
        or not _safe_archive_member_name(manifest_name)
        or manifest_name not in indexed_names
    ):
        raise LineagePackageError(
            "lineage package index does not identify an indexed manifest"
        )
    return index


def _verify_package_checks(package_path: Path) -> list[tuple[str, bool, str]]:
    from mlperf.edu_cli import verify_package_archive

    return verify_package_archive(package_path, repo_root=ROOT)


def _verify_provenance_checks(manifest_path: Path) -> list[tuple[str, bool, str]]:
    from mlperf.manifest import verify_provd

    return verify_provd(manifest_path, repo_root=ROOT).checks


def validate_nanogpt_lineage_package(package_path: Path) -> dict[str, Any]:
    """Verify a portable NanoGPT training package before creating an attempt."""
    package_path = package_path.expanduser().resolve()
    if not package_path.is_file():
        raise LineagePackageError(f"lineage package not found: {package_path}")
    package_sha256 = sha256_file(package_path)
    index = _preflight_lineage_archive(package_path)
    checks = _verify_package_checks(package_path)
    failed = [name for name, ok, _detail in checks if not ok]
    if failed:
        raise LineagePackageError(
            f"lineage package failed clean-extraction verification: {failed}"
        )
    if sha256_file(package_path) != package_sha256:
        raise LineagePackageError("lineage package changed during verification")
    return {
        "package_path": package_path,
        "package_sha256": package_sha256,
        "index": index,
        "verification_checks": checks,
    }


def _staged_artifact_path(
    owner_path: Path, raw_path: Any, stage_root: Path, role: str
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise LineagePackageError(f"lineage manifest does not declare {role}")
    artifact_path = Path(raw_path)
    if artifact_path.is_absolute():
        raise LineagePackageError(f"lineage manifest {role} path must be relative")
    resolved = (owner_path.parent / artifact_path).resolve()
    try:
        resolved.relative_to(stage_root.resolve())
    except ValueError as exc:
        raise LineagePackageError(
            f"lineage manifest {role} path escapes the staged package"
        ) from exc
    if not resolved.is_file():
        raise LineagePackageError(f"staged lineage {role} is missing: {raw_path}")
    return resolved


def _validate_staged_lineage(
    stage_root: Path, index: dict[str, Any]
) -> dict[str, Path]:
    """Locate and semantically validate the packaged max-training lineage."""
    manifest_name = str(index.get("manifest") or index.get("source_manifest"))
    manifest_path = (stage_root / manifest_name).resolve()
    try:
        manifest_path.relative_to(stage_root.resolve())
    except ValueError as exc:
        raise LineagePackageError(
            "packaged manifest escapes the staging directory"
        ) from exc
    if not manifest_path.is_file():
        raise LineagePackageError("packaged NanoGPT training manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LineagePackageError(
            "packaged NanoGPT training manifest is invalid"
        ) from exc
    if manifest.get("workload") != "nanogpt-train":
        raise LineagePackageError("packaged provenance is not for nanogpt-train")
    leaves = manifest.get("leaves") or {}
    report_path = _staged_artifact_path(
        manifest_path,
        (leaves.get("measurement") or {}).get("report_path"),
        stage_root,
        "training report",
    )
    checkpoint_path = _staged_artifact_path(
        manifest_path,
        (leaves.get("weights") or {}).get("path"),
        stage_root,
        "checkpoint",
    )
    try:
        report = json.loads(report_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LineagePackageError(
            "packaged NanoGPT training report is invalid"
        ) from exc
    quality = report.get("quality") or {}
    if (
        report.get("workload") != "nanogpt-train"
        or report.get("profile") != "max"
        or report.get("status") != "passed"
        or report.get("data_mode") != "real"
        or quality.get("quality_required") is not True
        or quality.get("target_met") is not True
    ):
        raise LineagePackageError(
            "lineage package must contain a passing real-data nanogpt-train max report"
        )
    provenance_checks = _verify_provenance_checks(manifest_path)
    failed = [name for name, ok, _detail in provenance_checks if not ok]
    if failed:
        raise LineagePackageError(
            f"staged NanoGPT training provenance failed verification: {failed}"
        )
    return {
        "manifest": manifest_path,
        "report": report_path,
        "checkpoint": checkpoint_path,
    }


def stage_nanogpt_lineage_package(
    validation: dict[str, Any], attempt_dir: Path
) -> dict[str, Any]:
    """Safely extract verified NanoGPT training lineage inside an attempt."""
    package_path = Path(validation["package_path"])
    expected_sha256 = str(validation["package_sha256"])
    if sha256_file(package_path) != expected_sha256:
        raise LineagePackageError("lineage package changed after verification")

    stage_root = attempt_dir / "inputs" / "nanogpt-training"
    stage_tmp = attempt_dir / "inputs" / ".nanogpt-training.tmp"
    stage_root.parent.mkdir(parents=True, exist_ok=True)
    stage_tmp.mkdir(exist_ok=False)
    try:
        with zipfile.ZipFile(package_path) as zf:
            # Re-run structural preflight immediately before extraction to close
            # the gap between verification and staging.
            current_index = _preflight_lineage_archive(package_path)
            if current_index != validation["index"]:
                raise LineagePackageError(
                    "lineage package index changed after verification"
                )
            for info in zf.infolist():
                target = stage_tmp / PurePosixPath(info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1 << 20)

        for item in current_index["included_files"]:
            staged = stage_tmp / PurePosixPath(str(item["path"]))
            if staged.stat().st_size != item["n_bytes"]:
                raise LineagePackageError(
                    f"staged lineage size does not match index: {item['path']}"
                )
            if sha256_file(staged) != item.get("sha256"):
                raise LineagePackageError(
                    f"staged lineage digest does not match index: {item['path']}"
                )
        located = _validate_staged_lineage(stage_tmp, current_index)
        if sha256_file(package_path) != expected_sha256:
            raise LineagePackageError("lineage package changed during staging")
        relative_locations = {
            role: path.relative_to(stage_tmp) for role, path in located.items()
        }
        stage_tmp.rename(stage_root)
    except Exception:
        shutil.rmtree(stage_tmp, ignore_errors=True)
        raise

    paths = {role: stage_root / path for role, path in relative_locations.items()}
    environment = {
        NANOGPT_LINEAGE_ENV[role]: str(path.resolve()) for role, path in paths.items()
    }
    return {
        "package_sha256": expected_sha256,
        "package_schema": current_index["schema"],
        "source_workload": current_index["workload"],
        "stage_root": stage_root,
        "paths": paths,
        "environment": environment,
        "verification_check_count": len(validation["verification_checks"]),
    }


def _tail(value: str | bytes | None, lines: int = 20) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return "\n".join((value or "").strip().splitlines()[-lines:])


def _relative_to_attempt(path_value: str | None, attempt_dir: Path) -> str | None:
    if not path_value:
        return None
    path = Path(path_value).resolve()
    try:
        return path.relative_to(attempt_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def run_one_seed(
    bootstrap_path: Path,
    *,
    workload_id: str,
    variant: str | None,
    profile: str,
    seed: int,
    device: str | None,
    attempt_dir: Path,
    timeout_seconds: float,
    evidence_tier: str,
    allowed_data_modes: frozenset[str],
    environment_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one seed in a fresh process and return its validation record."""
    seed_dir = attempt_dir / f"seed_{seed}"
    with tempfile.TemporaryDirectory(prefix=f"mlperf-edu-seed-{seed}-") as tmp:
        args_path = Path(tmp) / "args.json"
        result_path = Path(tmp) / "result.json"
        child_args = {
            "root": str(ROOT),
            "workload_id": workload_id,
            "variant": variant,
            "profile": profile,
            "seed": seed,
            "device": device,
            "output_dir": str(seed_dir),
            "evidence_tier": evidence_tier,
            "allowed_data_modes": sorted(allowed_data_modes),
        }
        args_path.write_text(json.dumps(child_args, indent=2, sort_keys=True) + "\n")
        env = sweep_environment(seed, device, environment_overrides)
        command = [
            sys.executable,
            str(bootstrap_path),
            str(args_path),
            str(result_path),
        ]
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            returncode = process.returncode
            stdout_tail = _tail(process.stdout)
            stderr_tail = _tail(process.stderr)
        except subprocess.TimeoutExpired as exc:
            return {
                "requested_seed": seed,
                "execution_ok": False,
                "evidence_valid": False,
                "timed_out": True,
                "timeout_seconds": timeout_seconds,
                "subprocess_wall_seconds": time.perf_counter() - started,
                "invalid_reasons": [f"run exceeded {timeout_seconds:g}-second timeout"],
                "stdout_tail": _tail(exc.stdout),
                "stderr_tail": _tail(exc.stderr),
                "reproduce": reproduce_record(
                    workload_id,
                    variant,
                    profile,
                    seed,
                    device,
                    bool(environment_overrides),
                ),
            }

        if result_path.is_file():
            result = json.loads(result_path.read_text())
        else:
            result = {
                "requested_seed": seed,
                "execution_ok": False,
                "evidence_valid": False,
                "invalid_reasons": ["child process produced no result record"],
            }
        result["returncode"] = returncode
        result["subprocess_wall_seconds"] = time.perf_counter() - started
        result["stdout_tail"] = stdout_tail
        result["stderr_tail"] = stderr_tail
        result["reproduce"] = reproduce_record(
            workload_id,
            variant,
            profile,
            seed,
            device,
            bool(environment_overrides),
        )
        result["report_path"] = _relative_to_attempt(
            result.get("report_path"), attempt_dir
        )
        result["manifest_path"] = _relative_to_attempt(
            result.get("manifest_path"), attempt_dir
        )
        for artifact in result.get("artifacts") or []:
            artifact["path"] = _relative_to_attempt(artifact.get("path"), attempt_dir)
        return result


def reproduce_record(
    workload_id: str,
    variant: str | None,
    profile: str,
    seed: int,
    device: str | None,
    uses_nanogpt_lineage_package: bool = False,
) -> dict[str, Any]:
    command = ["uv", "run", "mlperf", "run", "--workload", workload_id]
    if variant:
        command.extend(["--variant", variant])
    command.extend(["--profile", profile, "--output-dir", "<NEW_OUTPUT_DIR>"])
    env = {"MLPERF_EDU_MAX_SEED": str(seed)}
    if device:
        env["MLPERF_EDU_DEVICE"] = device
    if uses_nanogpt_lineage_package:
        env.update(
            {
                NANOGPT_LINEAGE_ENV["checkpoint"]: "<STAGED_TRAINING_CHECKPOINT>",
                NANOGPT_LINEAGE_ENV["report"]: "<STAGED_TRAINING_REPORT>",
                NANOGPT_LINEAGE_ENV["manifest"]: "<STAGED_TRAINING_MANIFEST>",
            }
        )
    return {"command": command, "env": env}


def sweep_environment(
    seed: int,
    device: str | None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return an isolated seed environment with no higher-priority override."""
    env = dict(os.environ)
    env.pop("MLPERF_EDU_SEED", None)
    env.pop("MLPERF_EDU_SLM_SEED", None)
    env["MLPERF_EDU_MAX_SEED"] = str(seed)
    if device:
        env["MLPERF_EDU_DEVICE"] = device
    else:
        env.pop("MLPERF_EDU_DEVICE", None)
    if overrides:
        env.update(overrides)
    return env


def build_row(result: dict[str, Any]) -> dict[str, Any]:
    requested = result.get("requested_seed")
    report_seed = result.get("report_recorded_seed")
    manifest_seed = result.get("manifest_recorded_seed")
    seed_match = report_seed == requested and manifest_seed == requested
    return {
        "requested_seed": requested,
        "report_recorded_seed": report_seed,
        "manifest_recorded_seed": manifest_seed,
        "seed_match": seed_match,
        "status": result.get("status"),
        "quality_metric_declared": result.get("quality_metric_declared"),
        "functional_metric_declared": result.get("functional_metric_declared"),
        "reference_metric_role": result.get("reference_metric_role"),
        "quality_metric_key": result.get("quality_metric_key"),
        "quality_value": result.get("quality_value"),
        "quality_target_met": result.get("quality_target_met"),
        "wall_seconds": result.get("wall_seconds"),
        "subprocess_wall_seconds": result.get("subprocess_wall_seconds"),
        "backend": result.get("backend"),
        "hardware_backend": result.get("hardware_backend"),
        "fingerprint_backends": result.get("fingerprint_backends"),
        "chip": result.get("chip"),
        "data_mode": result.get("data_mode"),
        "report_path": result.get("report_path"),
        "manifest_path": result.get("manifest_path"),
        "manifest_verified": result.get("manifest_verified", False),
        "grade": result.get("grade"),
        "artifacts": result.get("artifacts") or [],
        "execution_ok": bool(result.get("execution_ok")),
        "evidence_valid": bool(result.get("evidence_valid")) and seed_match,
        "timed_out": bool(result.get("timed_out")),
        "invalid_reasons": list(result.get("invalid_reasons") or []),
        "reproduce": result.get("reproduce"),
        "stderr_tail": result.get("stderr_tail"),
    }


def seed_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("evidence_valid")
        and isinstance(row.get("quality_value"), (int, float))
    ]
    distinct = {round(float(row["quality_value"]), 12) for row in valid}
    if len(valid) < 2:
        verdict = "inconclusive"
        note = "Fewer than two valid runs; a seed-sensitivity claim cannot be made."
    elif len(distinct) < 2:
        verdict = "identical"
        note = "Every valid seed produced the same quality value; this is not usable variance evidence."
    else:
        verdict = "sensitive"
        note = f"Observed {len(distinct)} distinct quality values across {len(valid)} valid runs."
    return {
        "verdict": verdict,
        "distinct_quality_values": len(distinct),
        "valid_runs": len(valid),
        "note": note,
    }


def aggregate_acceptance(
    aggregate_value: Any, target: Any, direction: str | None
) -> dict[str, Any]:
    if not isinstance(aggregate_value, (int, float)) or not isinstance(
        target, (int, float)
    ):
        return {"passed": False, "reason": "numeric aggregate and target are required"}
    if direction == "lower":
        passed = float(aggregate_value) <= float(target)
        operator = "<="
    elif direction == "higher":
        passed = float(aggregate_value) >= float(target)
        operator = ">="
    else:
        return {
            "passed": False,
            "reason": f"unsupported quality direction {direction!r}",
        }
    return {
        "passed": passed,
        "statistic": "median",
        "value": aggregate_value,
        "operator": operator,
        "target": target,
    }


def performance_acceptance(
    rows: list[dict[str, Any]], condition: str | None
) -> dict[str, Any]:
    """Require the functional serving gate on every reference run.

    Performance itself is summarized, not thresholded. This prevents a benchmark
    from defining a circular speed target from the same machine used to measure it.
    """
    passed_runs = [
        row
        for row in rows
        if row.get("evidence_valid") and row.get("quality_target_met") is True
    ]
    return {
        "passed": len(passed_runs) == len(rows) and bool(rows),
        "statistic": "all_runs",
        "value": len(passed_runs),
        "operator": "==",
        "target": len(rows),
        "condition": condition,
        "note": "Every measured run must pass the functional check; performance values are reported without a machine-derived pass threshold.",
    }


def public_data_modes_for(workload: Any) -> frozenset[str]:
    if workload.public_status == "score-bearing":
        return SCORE_PUBLIC_DATA_MODES
    if workload.public_status == "performance-bearing":
        return PERFORMANCE_PUBLIC_DATA_MODES
    return frozenset()


def source_snapshot() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    status = git("status", "--porcelain")
    try:
        patch = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        patch_sha256 = "sha256:" + hashlib.sha256(patch).hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError):
        patch_sha256 = None
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_status_sha256": (
            "sha256:" + hashlib.sha256(status.encode()).hexdigest()
            if status is not None
            else None
        ),
        "git_patch_sha256": patch_sha256,
        "tool_path": f"tools/{TOOL_NAME}",
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
    }


def build_basis(
    *,
    workload: Any,
    profile: str,
    rows: list[dict[str, Any]],
    quality: dict[str, Any],
    metric_name: str | None,
    dataset_mode: Any,
    eligible: bool,
) -> dict[str, Any]:
    performance = workload.public_status == "performance-bearing"
    functional = dict(workload.raw.get("functional_check") or {})
    functional_targets = {
        json.dumps((row.get("grade") or {}).get("target"), sort_keys=True)
        for row in rows
        if (row.get("grade") or {}).get("target") is not None
    }
    functional_target = None
    if len(functional_targets) == 1:
        functional_target = json.loads(next(iter(functional_targets)))
    if performance:
        functional["target"] = functional_target
    return {
        "eligible_for_public_baseline": eligible,
        "variance_summary": {
            "runs": len([row for row in rows if row.get("evidence_valid")]),
            "statistic": "median",
            "metric": metric_name,
            **quality,
        },
        "reference_protocol": {
            "profile": profile,
            "seeds": [row.get("requested_seed") for row in rows],
            "seed_interface": "MLPERF_EDU_MAX_SEED",
            "dataset_mode": dataset_mode,
            "observed_data_modes": sorted(
                {str(row.get("data_mode")) for row in rows if row.get("data_mode")}
            ),
            "rerun_policy": "If any run fails or times out, create a new full-sweep attempt; never replace one seed in an existing attempt.",
            "artifact_policy": "Preserve and SHA-256 index every report, provenance manifest, checkpoint, and runner-declared artifact.",
            "generated_by": TOOL_ID,
        },
        "quality_target": (
            None
            if performance
            else {
                "metric": metric_name,
                "target": getattr(workload, "quality_value", None),
                "direction": getattr(workload, "quality_direction", None),
            }
        ),
        "functional_check": functional,
    }


def _declared_protocol(workload: Any) -> dict[str, Any]:
    if workload.public_status == "performance-bearing":
        protocol = workload.raw.get("performance_reference_protocol")
        return protocol if isinstance(protocol, dict) else {}
    protocol = getattr(workload, "quality_reference_protocol", None)
    return protocol if isinstance(protocol, dict) else {}


def validate_sweep(
    *,
    workload: Any,
    seeds: list[int],
    rows: list[dict[str, Any]],
    sensitivity: dict[str, Any],
    acceptance: dict[str, Any],
    evidence_tier: str,
) -> list[str]:
    reasons: list[str] = []
    for row in rows:
        if not row.get("evidence_valid"):
            details = "; ".join(
                row.get("invalid_reasons") or ["unknown validation failure"]
            )
            reasons.append(f"seed {row.get('requested_seed')}: {details}")
    if sensitivity["verdict"] != "sensitive":
        reasons.append(f"seed sensitivity is {sensitivity['verdict']}, not sensitive")
    metric_keys = {
        row.get("quality_metric_key") for row in rows if row.get("quality_metric_key")
    }
    if len(metric_keys) != 1:
        reasons.append(
            f"runs did not resolve to exactly one metric key: {sorted(metric_keys)}"
        )
    if not acceptance.get("passed"):
        reasons.append(
            f"median quality acceptance failed: {acceptance.get('reason') or acceptance}"
        )

    if evidence_tier == "public-candidate":
        protocol = _declared_protocol(workload)
        declared_runs = (
            protocol.get("reference_runs")
            if workload.public_status == "performance-bearing"
            else getattr(workload, "quality_reference_runs", None)
        )
        if isinstance(declared_runs, int) and len(seeds) != declared_runs:
            reasons.append(
                f"registry requires {declared_runs} reference runs, received {len(seeds)}"
            )
        protocol_seeds = protocol.get("seeds")
        if isinstance(protocol_seeds, list) and seeds != protocol_seeds:
            reasons.append(
                f"requested seeds {seeds} do not match registry protocol seeds {protocol_seeds}"
            )
        allowed_modes = public_data_modes_for(workload)
        invalid_modes = sorted(
            {
                str(row.get("data_mode"))
                for row in rows
                if row.get("data_mode") not in allowed_modes
            }
        )
        if invalid_modes:
            reasons.append(
                f"public candidates allow {sorted(allowed_modes)} for {workload.public_status}, observed {invalid_modes}"
            )
    return reasons


def write_evidence_summary(
    attempt_dir: Path, artifact: dict[str, Any]
) -> tuple[Path, Path, str]:
    artifact_path = attempt_dir / "evidence_summary.json"
    payload = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with artifact_path.open("xb") as handle:
        handle.write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    digest_path = artifact_path.with_suffix(".json.sha256")
    with digest_path.open("x") as handle:
        handle.write(f"{digest}  {artifact_path.name}\n")
    return artifact_path, digest_path, digest


def print_summary(rows: list[dict[str, Any]], artifact_path: Path, valid: bool) -> None:
    print("seed  valid  metric                 value       data_mode")
    print("----  -----  ---------------------  ----------  -----------------")
    for row in rows:
        print(
            f"{str(row.get('requested_seed')):>4}  "
            f"{str(bool(row.get('evidence_valid'))):<5}  "
            f"{str(row.get('quality_metric_key') or '-'):21.21}  "
            f"{str(row.get('quality_value')):10.10}  "
            f"{str(row.get('data_mode') or '-')}"
        )
    print(f"evidence status: {'VALID' if valid else 'INVALID'}")
    print(f"evidence summary: {artifact_path}")


def _uses_nanogpt_training_lineage(workload: Any) -> bool:
    return str(workload.raw.get("shared_checkpoint") or "") == "nanogpt-train"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Run a verified multi-seed reference sweep for one MLPerf EDU workload.",
    )
    parser.add_argument("--workload", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--profile", default="max")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2,3,4"))
    parser.add_argument(
        "--device", choices=("cpu", "mps", "default"), default="default"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"create-once evidence root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"per-seed timeout (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument(
        "--evidence-tier",
        choices=("auto", "public-candidate", "development"),
        default="auto",
    )
    parser.add_argument(
        "--nanogpt-lineage-package",
        help=(
            "verified mlperf-edu-package/0.2 archive from a passing real-data "
            "nanogpt-train max run; required for public-candidate NanoGPT inference"
        ),
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be a finite positive number")

    try:
        from mlperf.edu_cli import load_runner
        from mlperf.registry import load_registry
    except Exception as exc:
        print(f"error: cannot import MLPerf EDU: {exc}", file=sys.stderr)
        return 2
    registry = load_registry()
    workload = None
    if args.variant:
        for candidate in registry.values():
            if (
                getattr(candidate, "canonical_workload", None) == args.workload
                and getattr(candidate, "variant", None) == args.variant
            ):
                workload = candidate
                break
    else:
        workload = registry.get(args.workload)
    if workload is None:
        print(
            f"error: workload/variant not found: {args.workload}/{args.variant}",
            file=sys.stderr,
        )
        return 2
    if load_runner(workload, args.profile) is None:
        print(f"error: no {args.profile!r} runner for {workload.id}", file=sys.stderr)
        return 2

    evidence_tier = args.evidence_tier
    if evidence_tier == "auto":
        evidence_tier = (
            "public-candidate"
            if workload.public_status in PUBLIC_STATUSES
            else "development"
        )
    uses_nanogpt_lineage = _uses_nanogpt_training_lineage(workload)
    lineage_required = evidence_tier == "public-candidate" and uses_nanogpt_lineage
    if args.nanogpt_lineage_package and not uses_nanogpt_lineage:
        print(
            "error: --nanogpt-lineage-package is only valid for a workload that "
            "declares shared_checkpoint: nanogpt-train",
            file=sys.stderr,
        )
        return 2
    if lineage_required and not args.nanogpt_lineage_package:
        print(
            "error: public-candidate NanoGPT inference requires "
            "--nanogpt-lineage-package from a passing real-data nanogpt-train max run",
            file=sys.stderr,
        )
        return 2

    lineage_validation: dict[str, Any] | None = None
    if args.nanogpt_lineage_package:
        try:
            lineage_validation = validate_nanogpt_lineage_package(
                Path(args.nanogpt_lineage_package)
            )
        except LineagePackageError as exc:
            print(f"error: invalid NanoGPT lineage package: {exc}", file=sys.stderr)
            return 2

    allowed_data_modes = public_data_modes_for(workload)
    device = None if args.device == "default" else args.device
    # Bind source state before creating evidence. An explicitly selected output
    # directory may live inside the checkout; its artifacts are not source edits
    # and must not make an otherwise clean run reject itself.
    source = source_snapshot()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    evidence_id = (
        f"{workload.id}_{args.profile}_{started.strftime('%Y%m%dT%H%M%S.%fZ')}"
    )
    attempt_dir = output_root / evidence_id
    attempt_dir.mkdir(exist_ok=False)

    lineage_stage: dict[str, Any] | None = None
    if lineage_validation is not None:
        try:
            lineage_stage = stage_nanogpt_lineage_package(
                lineage_validation, attempt_dir
            )
        except (LineagePackageError, OSError, zipfile.BadZipFile) as exc:
            shutil.rmtree(attempt_dir, ignore_errors=True)
            print(
                f"error: could not stage NanoGPT lineage package: {exc}",
                file=sys.stderr,
            )
            return 2

    print(
        f"{TOOL_ID}: workload={workload.id} profile={args.profile} seeds={args.seeds} "
        f"tier={evidence_tier} timeout={args.timeout_seconds:g}s"
    )
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mlperf-edu-sweep-bootstrap-") as tmp:
        bootstrap_path = Path(tmp) / "child.py"
        bootstrap_path.write_text(_CHILD_BOOTSTRAP)
        for seed in args.seeds:
            print(f"running seed {seed} ...", flush=True)
            result = run_one_seed(
                bootstrap_path,
                workload_id=args.workload,
                variant=args.variant,
                profile=args.profile,
                seed=seed,
                device=device,
                attempt_dir=attempt_dir,
                timeout_seconds=args.timeout_seconds,
                evidence_tier=evidence_tier,
                allowed_data_modes=allowed_data_modes,
                environment_overrides=(
                    lineage_stage["environment"] if lineage_stage else None
                ),
            )
            rows.append(build_row(result))

    metric_name = next(
        (
            row.get("quality_metric_declared")
            for row in rows
            if row.get("quality_metric_declared")
        ),
        getattr(workload, "quality_metric", None),
    )
    quality_values = [
        float(row["quality_value"])
        for row in rows
        if row.get("evidence_valid")
        and isinstance(row.get("quality_value"), (int, float))
    ]
    wall_values = [
        float(row["wall_seconds"])
        for row in rows
        if row.get("execution_ok") and isinstance(row.get("wall_seconds"), (int, float))
    ]
    quality_aggregate = aggregate(quality_values)
    wall_aggregate = aggregate(wall_values)
    sensitivity = seed_sensitivity(rows)
    if workload.public_status == "performance-bearing":
        functional = workload.raw.get("functional_check") or {}
        acceptance = performance_acceptance(rows, functional.get("condition"))
    else:
        acceptance = aggregate_acceptance(
            quality_aggregate.get("median"),
            getattr(workload, "quality_value", None),
            getattr(workload, "quality_direction", None),
        )
    protocol = _declared_protocol(workload)
    invalid_reasons = validate_sweep(
        workload=workload,
        seeds=args.seeds,
        rows=rows,
        sensitivity=sensitivity,
        acceptance=acceptance,
        evidence_tier=evidence_tier,
    )
    repeatability = None
    if workload.public_status == "performance-bearing":
        repeatability_limit = float(protocol.get("repeatability_limit"))
        mean = quality_aggregate.get("mean")
        stdev = quality_aggregate.get("stdev")
        coefficient_of_variation = (
            float(stdev) / float(mean)
            if isinstance(mean, (int, float))
            and float(mean) > 0
            and isinstance(stdev, (int, float))
            else None
        )
        repeatability = {
            "metric": protocol.get("repeatability_metric"),
            "coefficient_of_variation": coefficient_of_variation,
            "limit": repeatability_limit,
            "passed": coefficient_of_variation is not None
            and coefficient_of_variation <= repeatability_limit,
        }
        if repeatability["passed"] is not True:
            invalid_reasons.append(
                "performance reference repeatability exceeds the declared "
                f"coefficient-of-variation limit {repeatability_limit:g}"
            )
    if evidence_tier == "public-candidate" and source.get("git_dirty") is not False:
        invalid_reasons.append(
            "public reference evidence must be produced from a clean Git worktree"
        )
    eligible = not invalid_reasons
    dataset_mode = protocol.get("dataset_mode")
    finished = datetime.now(timezone.utc)
    lineage_summary = None
    if uses_nanogpt_lineage:
        lineage_summary = {
            "required": lineage_required,
            "status": "staged" if lineage_stage else "not-supplied",
        }
        if lineage_stage:
            lineage_summary.update(
                {
                    "package_schema": lineage_stage["package_schema"],
                    "package_sha256": lineage_stage["package_sha256"],
                    "source_workload": lineage_stage["source_workload"],
                    "verification_check_count": lineage_stage[
                        "verification_check_count"
                    ],
                    "staged_root": _relative_to_attempt(
                        str(lineage_stage["stage_root"]), attempt_dir
                    ),
                    "source_training_report": _relative_to_attempt(
                        str(lineage_stage["paths"]["report"]), attempt_dir
                    ),
                    "source_training_manifest": _relative_to_attempt(
                        str(lineage_stage["paths"]["manifest"]), attempt_dir
                    ),
                    "source_training_checkpoint": _relative_to_attempt(
                        str(lineage_stage["paths"]["checkpoint"]), attempt_dir
                    ),
                }
            )
    basis = build_basis(
        workload=workload,
        profile=args.profile,
        rows=rows,
        quality=quality_aggregate,
        metric_name=metric_name,
        dataset_mode=dataset_mode,
        eligible=eligible,
    )
    artifact = {
        "schema": "mlperf-edu-reference-evidence/0.3",
        "evidence_id": evidence_id,
        "status": "valid" if eligible else "invalid",
        "eligible_for_public_baseline": eligible
        and evidence_tier == "public-candidate",
        "invalid_reasons": invalid_reasons,
        "tool": {"id": TOOL_ID, "version": TOOL_VERSION},
        "generated_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "write_policy": "create-once attempt directory; this tool never overwrites or edits prior evidence",
        "digest_policy": "The adjacent SHA-256 sidecar is an unauthenticated integrity digest, not a signature.",
        "rerun_policy": {
            "mode": "full-sweep-only",
            "rule": "If any seed fails or times out, create a new attempt and rerun every declared seed. Never replace an individual run in an existing attempt.",
        },
        "workload": workload.id,
        "canonical_workload": getattr(workload, "canonical_workload", None)
        or workload.id,
        "variant": args.variant,
        "profile": args.profile,
        "public_status": workload.public_status,
        "evidence_tier": evidence_tier,
        "device_requested": args.device,
        "timeout_seconds_per_seed": args.timeout_seconds,
        "seeds_requested": args.seeds,
        "dataset_mode_declared": dataset_mode,
        "allowed_public_data_modes": sorted(allowed_data_modes),
        "reference_metric_role": (
            "performance"
            if workload.public_status == "performance-bearing"
            else "quality"
        ),
        "primary_metric": {
            "name": metric_name,
            "role": (
                "performance"
                if workload.public_status == "performance-bearing"
                else "quality"
            ),
        },
        "quality_metric": metric_name,
        "quality_target": (
            None
            if workload.public_status == "performance-bearing"
            else getattr(workload, "quality_value", None)
        ),
        "quality_direction": getattr(workload, "quality_direction", None),
        "functional_gate": (
            basis["functional_check"]
            if workload.public_status == "performance-bearing"
            else None
        ),
        "runs": rows,
        "aggregate": {
            "primary_metric": quality_aggregate,
            "quality": quality_aggregate,
            "wall_seconds": wall_aggregate,
        },
        "repeatability": repeatability,
        "seed_sensitivity": sensitivity,
        "acceptance": acceptance,
        "basis": basis,
        "source": source,
        "nanogpt_training_lineage": lineage_summary,
    }
    artifact_path, _, _ = write_evidence_summary(attempt_dir, artifact)
    print_summary(rows, artifact_path, eligible)
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
