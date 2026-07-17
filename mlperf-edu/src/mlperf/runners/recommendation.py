from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from mlperf.assets import (
    DLRM_CHECKPOINT_MD5,
    DLRM_CHECKPOINT_URL,
    DLRM_IMPLEMENTATION_COMMIT,
    DLRM_IMPLEMENTATION_FILES,
    DLRM_INFERENCE_COMMIT,
    DLRM_INFERENCE_FILES,
    DLRM_TRAINING_SUBMODULE_COMMIT,
    dlrm_reference_paths,
    ensure_dlrm_reference,
    sha256_file,
)
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import configured_seed


TARGET_ROC_AUC = 0.8025
MAX_IND_RANGE = 40_000_000
SAMPLES_PER_QUERY_OFFLINE = 204_800
MAX_BATCH_SIZE = 2_048
DAY_23_ROWS = 178_274_637
DATA_DIR_ENV = "MLPERF_EDU_DLRM_DATA_DIR"
CHECKPOINT_ENV = "MLPERF_EDU_DLRM_CHECKPOINT"
PYTHON_ENV = "MLPERF_EDU_DLRM_PYTHON"
DEVICE_ENV = "MLPERF_EDU_DLRM_DEVICE"
TERMS_ENV = "MLPERF_EDU_CRITEO_TERMS_ACCEPTED"


def required_preprocessed_files(data_dir: Path) -> tuple[Path, ...]:
    """Return the exact memory-mapped Criteo files consumed by the runner."""
    return (
        data_dir / "day_day_count.npz",
        data_dir / "day_fea_count.npz",
        *(data_dir / f"day_{day}_reordered.npz" for day in range(24)),
    )


def _checkpoint_digests(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    n_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            md5.update(chunk)
            sha256.update(chunk)
            n_bytes += len(chunk)
    return md5.hexdigest(), sha256.hexdigest(), n_bytes


def _resolve_python(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    executable = shutil.which(value)
    if executable:
        return executable
    raise FileNotFoundError(
        f"{PYTHON_ENV} does not resolve to an executable Python runtime: {value}"
    )


def preflight_environment() -> dict[str, Any]:
    """Validate the manually licensed assets and prepared legacy runtime."""
    if os.environ.get(TERMS_ENV) != "1":
        raise RuntimeError(
            "The authoritative DLRM run uses the Criteo Terabyte dataset. Review "
            f"and accept its upstream terms, then set {TERMS_ENV}=1. MLPerf EDU "
            "will not download or redistribute this dataset."
        )

    data_value = os.environ.get(DATA_DIR_ENV)
    checkpoint_value = os.environ.get(CHECKPOINT_ENV)
    missing_variables = [
        name
        for name, value in (
            (DATA_DIR_ENV, data_value),
            (CHECKPOINT_ENV, checkpoint_value),
        )
        if not value
    ]
    if missing_variables:
        raise RuntimeError(
            "The authoritative DLRM quality run is environment-gated. Set "
            f"{', '.join(missing_variables)} after preparing the official "
            "unshuffled Criteo Terabyte data and 40M checkpoint."
        )

    data_dir = Path(str(data_value)).expanduser().resolve()
    checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Criteo data directory does not exist: {data_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DLRM checkpoint does not exist: {checkpoint}")

    dataset_files = required_preprocessed_files(data_dir)
    missing_files = [str(path) for path in dataset_files if not path.is_file()]
    if missing_files:
        preview = "\n".join(f"- {path}" for path in missing_files[:5])
        suffix = (
            f"\n- ... and {len(missing_files) - 5} more"
            if len(missing_files) > 5
            else ""
        )
        raise FileNotFoundError(
            "The official memory-mapped DLRM accuracy set is incomplete. Missing:\n"
            f"{preview}{suffix}"
        )
    if any(path.stat().st_size == 0 for path in dataset_files):
        raise ValueError("DLRM preprocessed dataset files must be nonempty")

    checkpoint_md5, checkpoint_sha256, checkpoint_bytes = _checkpoint_digests(
        checkpoint
    )
    if checkpoint_md5 != DLRM_CHECKPOINT_MD5:
        raise ValueError(
            "DLRM checkpoint MD5 does not match the official tb00_40M.pt pin: "
            f"expected {DLRM_CHECKPOINT_MD5}, found {checkpoint_md5}"
        )

    python = _resolve_python(os.environ.get(PYTHON_ENV, sys.executable))
    runtime = subprocess.run(
        [
            python,
            "-c",
            (
                "import json, sklearn, torch, mlperf_loadgen; "
                "print(json.dumps({'python': __import__('sys').version, "
                "'torch': torch.__version__, 'sklearn': sklearn.__version__, "
                "'cuda_available': torch.cuda.is_available()}))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if runtime.returncode != 0:
        raise RuntimeError(
            f"{PYTHON_ENV} must provide torch, scikit-learn, and mlperf_loadgen "
            f"for the historical reference runner. Runtime check failed:\n"
            f"{runtime.stderr.strip()}"
        )
    try:
        runtime_versions = json.loads(runtime.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "DLRM runtime preflight returned invalid version data"
        ) from exc

    device = os.environ.get(DEVICE_ENV, "cpu").lower()
    if device not in {"cpu", "gpu"}:
        raise ValueError(f"{DEVICE_ENV} must be either cpu or gpu")
    if device == "gpu" and not bool(runtime_versions.get("cuda_available")):
        raise RuntimeError(
            f"{DEVICE_ENV}=gpu requires a CUDA-visible PyTorch runtime for MLPerf EDU"
        )

    return {
        "data_dir": data_dir,
        "dataset_files": dataset_files,
        "checkpoint": checkpoint,
        "checkpoint_md5": checkpoint_md5,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "python": python,
        "runtime_versions": runtime_versions,
        "device": device,
    }


def build_official_command(
    *,
    python: str,
    inference_root: Path,
    checkpoint: Path,
    data_dir: Path,
    official_output: Path,
    trace_path: Path,
    seed: int,
    device: str,
) -> list[str]:
    runner_root = inference_root / "recommendation" / "dlrm" / "pytorch"
    command = [
        python,
        str(runner_root / "python" / "main.py"),
        "--profile",
        "dlrm-terabyte-pytorch",
        "--model",
        "dlrm",
        "--model-path",
        str(checkpoint),
        "--dataset",
        "terabyte",
        "--dataset-path",
        str(data_dir),
        "--output",
        str(official_output),
        "--scenario",
        "Offline",
        "--max-ind-range",
        str(MAX_IND_RANGE),
        "--samples-to-aggregate-quantile-file",
        str(runner_root / "tools" / "dist_quantile.txt"),
        "--max-batchsize",
        str(MAX_BATCH_SIZE),
        "--samples-per-query-offline",
        str(SAMPLES_PER_QUERY_OFFLINE),
        "--samples-to-aggregate-trace-file",
        str(trace_path),
        "--numpy-rand-seed",
        str(seed),
        "--mlperf_conf",
        str(inference_root / "mlperf.conf"),
        "--user_conf",
        str(runner_root / "user.conf"),
        "--accuracy",
    ]
    if device == "gpu":
        command.append("--use-gpu")
    return command


def parse_official_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    candidates = [
        value
        for value in payload.values()
        if isinstance(value, dict) and "roc_auc" in value
    ]
    if len(candidates) != 1:
        raise ValueError("DLRM results.json must contain one ROC AUC result")
    result = candidates[0]
    official_auc_percent = float(result["roc_auc"])
    if not 0.0 <= official_auc_percent <= 100.0:
        raise ValueError("official DLRM ROC AUC percentage is outside [0, 100]")
    return {
        "roc_auc": official_auc_percent / 100.0,
        "official_roc_auc_percent": official_auc_percent,
        "accuracy_percent": float(result.get("accuracy", 0.0)),
        "duration_seconds": float(result["took"]),
        "queries_per_second": float(result["qps"]),
        "queries": int(result["count"]),
        "evaluated_pairs": int(result["total_items"]),
        "runtime": payload.get("runtime"),
        "runtime_version": payload.get("version"),
    }


def run_recommendation_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run the complete MLPerf Inference v1.0.1 DLRM accuracy contract."""
    contract = workload.raw.get("canonical_max_contract") or {}
    config = contract.get("config") or {}
    if (
        int(config.get("max_ind_range", 0)) != MAX_IND_RANGE
        or int(config.get("samples_per_query_offline", 0)) != SAMPLES_PER_QUERY_OFFLINE
        or not bool(config.get("accuracy_mode"))
    ):
        raise ValueError("registry DLRM quality contract does not match the runner")

    source_asset = ensure_dlrm_reference(download=True)
    paths = dlrm_reference_paths()
    environment = preflight_environment()
    output_dir.mkdir(parents=True, exist_ok=True)
    official_output = (output_dir / "recommendation_max_official").resolve()
    official_output.mkdir(parents=True, exist_ok=True)
    trace_path = (output_dir / "recommendation_max_aggregation_trace.txt").resolve()
    report_path = (output_dir / "recommendation_max_report.json").resolve()
    provenance_path = (output_dir / "recommendation_max.provd.json").resolve()
    command_path = (output_dir / "recommendation_max_command.json").resolve()
    command = build_official_command(
        python=environment["python"],
        inference_root=paths["inference_source"],
        checkpoint=environment["checkpoint"],
        data_dir=environment["data_dir"],
        official_output=official_output,
        trace_path=trace_path,
        seed=configured_seed(),
        device=environment["device"],
    )
    command_path.write_text(json.dumps(command, indent=2) + "\n")
    subprocess_environment = os.environ.copy()
    subprocess_environment.update(
        {
            "DLRM_DIR": str(paths["implementation_source"]),
            "PYTHONHASHSEED": "0",
        }
    )
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=paths["inference_source"] / "recommendation" / "dlrm" / "pytorch",
        env=subprocess_environment,
        check=False,
    )
    wall_seconds = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"official DLRM accuracy runner exited with {completed.returncode}; "
            f"partial artifacts remain at {official_output}"
        )

    official_results_path = official_output / "results.json"
    accuracy_log_path = official_output / "mlperf_log_accuracy.json"
    if not official_results_path.is_file() or not accuracy_log_path.is_file():
        raise FileNotFoundError(
            "official DLRM runner did not emit results.json and "
            "mlperf_log_accuracy.json"
        )
    metrics = parse_official_results(official_results_path)
    target = float(workload.quality_value or TARGET_ROC_AUC)
    tolerance = float(workload.quality_tolerance or 0.0)
    target_met = metrics["roc_auc"] >= target - tolerance
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": f"mlperf-inference-pytorch-{environment['device']}",
        "data_mode": "real-unshuffled-criteo-terabyte-day-23",
        "model": {
            "id": "mlperf-inference-v1.0.1-dlrm-40m",
            "revision": DLRM_IMPLEMENTATION_COMMIT,
        },
        "model_source": {
            "training_repository": "https://github.com/mlcommons/training",
            "training_revision": DLRM_TRAINING_SUBMODULE_COMMIT,
            "implementation_repository": "https://github.com/facebookresearch/dlrm",
            "implementation_revision": DLRM_IMPLEMENTATION_COMMIT,
            "training_entrypoint": str(
                paths["implementation_source"] / "dlrm_s_pytorch.py"
            ),
            "checkpoint": str(environment["checkpoint"]),
            "checkpoint_uri": DLRM_CHECKPOINT_URL,
            "checkpoint_md5": environment["checkpoint_md5"],
            "checkpoint_sha256": f"sha256:{environment['checkpoint_sha256']}",
        },
        "inference_source": {
            "repository": "https://github.com/mlcommons/inference",
            "revision": DLRM_INFERENCE_COMMIT,
            "entrypoint": str(
                paths["inference_source"]
                / "recommendation"
                / "dlrm"
                / "pytorch"
                / "python"
                / "main.py"
            ),
            "critical_files": {
                **{
                    name: f"sha256:{digest}"
                    for name, digest in DLRM_INFERENCE_FILES.items()
                },
                **{
                    f"implementation/{name}": f"sha256:{digest}"
                    for name, digest in DLRM_IMPLEMENTATION_FILES.items()
                },
            },
        },
        "dataset": {
            "name": "criteo-terabyte",
            "source": "Criteo Terabyte under upstream terms",
            "root": str(environment["data_dir"]),
            "split": "unshuffled-day-23-accuracy-set",
            "day_23_rows": DAY_23_ROWS,
            "terms_acceptance_asserted": True,
            "files": len(environment["dataset_files"]),
        },
        "evaluator": {
            "repository": "https://github.com/mlcommons/inference",
            "revision": DLRM_INFERENCE_COMMIT,
            "official_results_sha256": f"sha256:{sha256_file(official_results_path)}",
            "accuracy_log_sha256": f"sha256:{sha256_file(accuracy_log_path)}",
        },
        "seed": configured_seed(),
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "scenario": "Offline",
            "accuracy_mode": True,
            "max_ind_range": MAX_IND_RANGE,
            "max_batch_size": MAX_BATCH_SIZE,
            "samples_per_query_offline": SAMPLES_PER_QUERY_OFFLINE,
            "runtime": environment["runtime_versions"],
        },
        "metrics": {**metrics, "wall_seconds": wall_seconds},
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "roc_auc",
            "target": target,
            "tolerance": tolerance,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": False,
        },
        "functional_readiness": {
            "schema": "mlperf-edu-functional-readiness/0.1",
            "stage": "quality-conformance",
            "end_to_end_execution": True,
            "authoritative_quality_contract_executed": True,
            "repeatability_verified": False,
            "promotion_eligible": False,
            "next_stage": "stability" if target_met else "quality-target-review",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(provenance_path),
            "weights": str(environment["checkpoint"]),
            "official_output": str(official_output),
            "official_results": str(official_results_path),
            "accuracy_log": str(accuracy_log_path),
            "aggregation_trace": str(trace_path),
            "command": str(command_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "offline",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        weights_path=environment["checkpoint"],
        weights_dtype="float32",
        dataset_name="criteo-terabyte-day-23-and-reference-source",
        dataset_files=[
            *environment["dataset_files"],
            *source_asset.files,
            official_results_path,
            accuracy_log_path,
            trace_path,
            command_path,
        ],
        rng_seed=configured_seed(),
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=find_project_root(),
    )
    provenance_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report
