from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .fingerprint import PERFORMANCE_ENVIRONMENT_ALLOWLIST


EXPERIMENT_PLAN_SCHEMA = "mlperf-edu-experiment-plan/0.1"
DEVICES = {"auto", "cpu", "cuda", "mps"}
MODES = {"training", "inference"}
PHASES = {"full", "prefill", "decode"}
ENVIRONMENT_KEY = re.compile(r"^MLPERF_EDU_[A-Z0-9_]+$")
RESERVED_ENVIRONMENT_KEYS = {"MLPERF_EDU_DEVICE", "MLPERF_EDU_PRO_REPETITIONS"}
IMMUTABLE_CONTRACT_KEYS = {"MLPERF_EDU_MAX_QUALITY_TARGET"}
SENSITIVE_TOKENS = {"TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL"}
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
PLAN_ENVIRONMENT_KEYS = {
    key for key in PERFORMANCE_ENVIRONMENT_ALLOWLIST if key.startswith("MLPERF_EDU_")
} | {
    "MLPERF_EDU_ANOMALY_DETECTION_MAX_BATCH_SIZE",
    "MLPERF_EDU_ANOMALY_DETECTION_MAX_REPETITIONS",
    "MLPERF_EDU_ANOMALY_DETECTION_MAX_WARMUP_REPETITIONS",
    "MLPERF_EDU_CRITEO_TERMS_ACCEPTED",
    "MLPERF_EDU_DATA_DIR",
    "MLPERF_EDU_DLRM_CHECKPOINT",
    "MLPERF_EDU_DLRM_DATA_DIR",
    "MLPERF_EDU_DLRM_DEVICE",
    "MLPERF_EDU_DLRM_PYTHON",
    "MLPERF_EDU_EDM_BATCH_SIZE",
    "MLPERF_EDU_EVALPLUS_WORKERS",
    "MLPERF_EDU_HF_LOCAL_ONLY",
    "MLPERF_EDU_MINIGO_CONTAINER_RUNTIME",
    "MLPERF_EDU_MINIGO_IMAGE",
    "MLPERF_EDU_MINIGO_PRO_GAMES_REVIEWED",
    "MLPERF_EDU_NANOGPT_CHECKPOINT",
    "MLPERF_EDU_NANOGPT_TRAIN_MANIFEST",
    "MLPERF_EDU_NANOGPT_TRAIN_REPORT",
    "MLPERF_EDU_VISUAL_WAKE_WORDS_MAX_BATCH_SIZE",
    "MLPERF_EDU_VISUAL_WAKE_WORDS_MAX_REPETITIONS",
    "MLPERF_EDU_VISUAL_WAKE_WORDS_MAX_WARMUP_REPETITIONS",
}


def _known_keys(value: dict[str, Any], *, allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {', '.join(unknown)}")


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _identifier(value: Any, *, label: str) -> str:
    normalized = _nonempty_string(value, label=label)
    if not IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            f"{label} must use lowercase letters, numbers, dots, underscores, or hyphens"
        )
    return normalized


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _string_list(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return [
        _nonempty_string(item, label=f"{label} item")
        for item in value
    ]


def _environment(value: Any, *, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not ENVIRONMENT_KEY.fullmatch(key):
            raise ValueError(f"{label} key {key!r} is not an MLPERF_EDU_* setting")
        if key in RESERVED_ENVIRONMENT_KEYS:
            raise ValueError(f"{label} key {key!r} has a dedicated plan field")
        if key in IMMUTABLE_CONTRACT_KEYS:
            raise ValueError(f"{label} key {key!r} cannot override a quality target")
        if set(key.split("_")) & SENSITIVE_TOKENS:
            raise ValueError(f"{label} key {key!r} may contain sensitive data")
        if key not in PLAN_ENVIRONMENT_KEYS:
            raise ValueError(f"{label} key {key!r} is not a supported plan setting")
        if not isinstance(item, (str, int, float, bool)):
            raise ValueError(f"{label} value for {key!r} must be scalar")
        normalized[key] = str(item)
    return normalized


def load_experiment_plan(path: Path) -> dict[str, Any]:
    """Load a versioned, fail-closed pro experiment plan."""
    source_bytes = path.read_bytes()
    payload = yaml.safe_load(source_bytes)
    if not isinstance(payload, dict):
        raise ValueError("experiment plan must be a mapping")
    if payload.get("schema") != EXPERIMENT_PLAN_SCHEMA:
        raise ValueError(f"experiment plan schema must be {EXPERIMENT_PLAN_SCHEMA!r}")
    _known_keys(
        payload,
        allowed={
            "schema",
            "id",
            "title",
            "description",
            "study",
            "profile",
            "power",
            "keep_going",
            "defaults",
            "output",
            "runs",
        },
        label="experiment plan",
    )
    plan_id = _identifier(payload.get("id"), label="experiment plan id")
    profile = payload.get("profile", "pro")
    if profile != "pro":
        raise ValueError("experiment plans currently require profile: pro")
    study = payload.get("study") or {}
    if not isinstance(study, dict):
        raise ValueError("experiment plan study must be a mapping")
    _known_keys(
        study,
        allowed={
            "question",
            "hypothesis",
            "independent_variables",
            "controls",
            "analysis_metrics",
        },
        label="experiment plan study",
    )
    defaults = payload.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("experiment plan defaults must be a mapping")
    _known_keys(
        defaults,
        allowed={"device", "repetitions", "environment"},
        label="experiment plan defaults",
    )
    default_device = defaults.get("device", "auto")
    if default_device not in DEVICES:
        raise ValueError(f"experiment plan device must be one of {sorted(DEVICES)}")
    default_repetitions = _positive_integer(
        defaults.get("repetitions", 1), label="default repetitions"
    )
    default_environment = _environment(
        defaults.get("environment"), label="default environment"
    )
    output = payload.get("output") or {}
    if not isinstance(output, dict):
        raise ValueError("experiment plan output must be a mapping")
    _known_keys(
        output,
        allowed={"directory", "open_report"},
        label="experiment plan output",
    )
    output_dir = _nonempty_string(
        output.get("directory", f"submissions/research/{plan_id}"),
        label="experiment output directory",
    )
    open_report = _boolean(
        output.get("open_report", True), label="output.open_report"
    )
    power = _boolean(payload.get("power", False), label="experiment power")
    keep_going = _boolean(
        payload.get("keep_going", True), label="experiment keep_going"
    )
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("experiment plan runs must be a nonempty list")

    normalized_runs: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"experiment run {index} must be a mapping")
        _known_keys(
            run,
            allowed={
                "name",
                "workload",
                "variant",
                "mode",
                "phase",
                "device",
                "repetitions",
                "environment",
            },
            label=f"experiment run {index}",
        )
        workload = _nonempty_string(
            run.get("workload"), label=f"experiment run {index} workload"
        )
        variant = run.get("variant")
        if variant is not None:
            variant = _nonempty_string(
                variant, label=f"experiment run {index} variant"
            )
        mode = run.get("mode")
        if mode is not None and mode not in MODES:
            raise ValueError(
                f"experiment run {index} mode must be one of {sorted(MODES)}"
            )
        phase = run.get("phase")
        if phase is not None and phase not in PHASES:
            raise ValueError(
                f"experiment run {index} phase must be one of {sorted(PHASES)}"
            )
        if phase is not None and mode not in {None, "inference"}:
            raise ValueError(f"experiment run {index} phase requires inference mode")
        device = run.get("device", default_device)
        if device not in DEVICES:
            raise ValueError(
                f"experiment run {index} device must be one of {sorted(DEVICES)}"
            )
        repetitions = _positive_integer(
            run.get("repetitions", default_repetitions),
            label=f"experiment run {index} repetitions",
        )
        environment = {**default_environment}
        environment.update(
            _environment(
                run.get("environment"), label=f"experiment run {index} environment"
            )
        )
        name = _identifier(
            run.get("name")
            or "-".join(
                part
                for part in (workload, str(variant or ""), str(phase or mode or ""))
                if part
            ),
            label=f"experiment run {index} name",
        )
        if name in names:
            raise ValueError(f"duplicate experiment run name {name!r}")
        names.add(name)
        normalized_runs.append(
            {
                "name": name,
                "workload": workload,
                "variant": variant,
                "mode": mode,
                "phase": phase,
                "device": device,
                "repetitions": repetitions,
                "environment": environment,
            }
        )

    normalized = {
        "schema": EXPERIMENT_PLAN_SCHEMA,
        "id": plan_id,
        "title": str(payload.get("title") or plan_id),
        "description": str(payload.get("description") or ""),
        "study": {
            "question": str(study.get("question") or ""),
            "hypothesis": str(study.get("hypothesis") or ""),
            "independent_variables": _string_list(
                study.get("independent_variables"),
                label="study independent_variables",
            ),
            "controls": _string_list(study.get("controls"), label="study controls"),
            "analysis_metrics": _string_list(
                study.get("analysis_metrics"), label="study analysis_metrics"
            ),
        },
        "profile": "pro",
        "power": power,
        "keep_going": keep_going,
        "output": {"directory": output_dir, "open_report": open_report},
        "runs": normalized_runs,
        "source_sha256": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
    }
    json.dumps(normalized, allow_nan=False)
    return normalized
