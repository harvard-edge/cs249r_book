from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .fingerprint import PERFORMANCE_ENVIRONMENT_ALLOWLIST


EXPERIMENT_PLAN_SCHEMA = "mlperf-edu-experiment-plan/0.3"
LEGACY_EXPERIMENT_PLAN_SCHEMAS = {
    "mlperf-edu-experiment-plan/0.1",
    "mlperf-edu-experiment-plan/0.2",
}
SUPPORTED_EXPERIMENT_PLAN_SCHEMAS = {
    EXPERIMENT_PLAN_SCHEMA,
    *LEGACY_EXPERIMENT_PLAN_SCHEMAS,
}
EDIT_POLICY_SCHEMAS = {
    EXPERIMENT_PLAN_SCHEMA,
    "mlperf-edu-experiment-plan/0.2",
}
INSTRUCTOR_BINDING_SCHEMA = "mlperf-edu-instructor-plan-binding/0.1"
DEVICES = {"auto", "cpu", "cuda", "mps"}
MODES = {"training", "inference"}
PHASES = {"full", "prefill", "decode"}
ROLES = {"baseline", "candidate", "condition"}
ENVIRONMENT_KEY = re.compile(r"^MLPERF_EDU_[A-Z0-9_]+$")
RESERVED_ENVIRONMENT_KEYS = {"MLPERF_EDU_DEVICE", "MLPERF_EDU_PRO_REPETITIONS"}
IMMUTABLE_CONTRACT_KEYS = {"MLPERF_EDU_MAX_QUALITY_TARGET"}
SENSITIVE_TOKENS = {"TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL"}
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
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
    return [_nonempty_string(item, label=f"{label} item") for item in value]


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


def _edit_policy(
    value: Any,
    *,
    runs: list[dict[str, Any]],
    independent_variables: list[str],
) -> dict[str, dict[str, list[str]]]:
    if value is None:
        return {"allowed_candidate_environment": {}}
    if not isinstance(value, dict):
        raise ValueError("experiment plan edit_policy must be a mapping")
    _known_keys(
        value,
        allowed={"allowed_candidate_environment"},
        label="experiment plan edit_policy",
    )
    raw_allowed = value.get("allowed_candidate_environment") or {}
    if not isinstance(raw_allowed, dict):
        raise ValueError("edit_policy.allowed_candidate_environment must be a mapping")
    runs_by_name = {str(run["name"]): run for run in runs}
    normalized: dict[str, list[str]] = {}
    for run_name, raw_keys in raw_allowed.items():
        if not isinstance(run_name, str) or run_name not in runs_by_name:
            raise ValueError(
                "edit_policy.allowed_candidate_environment names an unknown run: "
                f"{run_name!r}"
            )
        run = runs_by_name[run_name]
        if run["role"] != "candidate":
            raise ValueError(
                "edit_policy may allow environment edits only for candidate runs: "
                f"{run_name!r}"
            )
        keys = _string_list(
            raw_keys,
            label=f"edit_policy.allowed_candidate_environment.{run_name}",
        )
        if not keys or len(keys) != len(set(keys)):
            raise ValueError(
                f"edit policy for candidate run {run_name!r} must contain unique settings"
            )
        for key in keys:
            if key not in PLAN_ENVIRONMENT_KEYS:
                raise ValueError(
                    f"edit policy setting {key!r} is not a supported plan setting"
                )
            if key not in run["environment"]:
                raise ValueError(
                    f"edit policy setting {key!r} is missing from candidate run "
                    f"{run_name!r}"
                )
            if key not in independent_variables:
                raise ValueError(
                    f"edit policy setting {key!r} is not a declared independent variable"
                )
        normalized[run_name] = sorted(keys)
    return {"allowed_candidate_environment": normalized}


def _baseline_import(value: Any, *, label: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    _known_keys(value, allowed={"manifest", "sha256"}, label=label)
    manifest = _nonempty_string(value.get("manifest"), label=f"{label}.manifest")
    manifest_path = Path(manifest)
    if (
        manifest_path.is_absolute()
        or "\\" in manifest
        or any(part in {"", ".", ".."} for part in manifest_path.parts)
    ):
        raise ValueError(
            f"{label}.manifest must be a normalized path within the plan directory"
        )
    digest = _nonempty_string(value.get("sha256"), label=f"{label}.sha256")
    if not SHA256_DIGEST.fullmatch(digest):
        raise ValueError(f"{label}.sha256 must be a complete SHA-256 digest")
    return {"manifest": manifest_path.as_posix(), "sha256": digest}


def _difference_paths(left: Any, right: Any, *, prefix: str = "plan") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}"
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(
                    _difference_paths(left[key], right[key], prefix=path)
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            differences.append(f"{prefix}.length")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _difference_paths(left_item, right_item, prefix=f"{prefix}[{index}]")
            )
        return differences
    return [] if left == right else [prefix]


def bind_instructor_reference(
    plan: dict[str, Any],
    reference: dict[str, Any],
    *,
    reference_source: str,
) -> dict[str, Any]:
    """Validate a student plan against instructor-owned allowed edits."""
    if plan["schema"] != reference["schema"]:
        raise ValueError("student and instructor plans must use the same schema")
    if reference["schema"] not in EDIT_POLICY_SCHEMAS:
        raise ValueError(
            "instructor plan binding requires an experiment-plan schema with "
            "edit_policy support"
        )

    expected = copy.deepcopy(reference)
    submitted = copy.deepcopy(plan)
    expected.pop("source_sha256", None)
    submitted.pop("source_sha256", None)
    expected_runs = {run["name"]: run for run in expected["runs"]}
    submitted_runs = {run["name"]: run for run in submitted["runs"]}
    allowed = (reference.get("edit_policy") or {}).get(
        "allowed_candidate_environment", {}
    )
    changes: list[dict[str, Any]] = []
    for run_name, keys in allowed.items():
        if run_name not in submitted_runs:
            continue
        reference_run = expected_runs[run_name]
        submitted_run = submitted_runs[run_name]
        for key in keys:
            reference_environment = reference_run["environment"]
            submitted_environment = submitted_run["environment"]
            if key not in submitted_environment:
                continue
            reference_value = reference_environment[key]
            submitted_value = submitted_environment[key]
            if submitted_value != reference_value:
                changes.append(
                    {
                        "run": run_name,
                        "setting": key,
                        "reference_value": reference_value,
                        "submitted_value": submitted_value,
                    }
                )
            submitted_environment[key] = reference_value

    differences = _difference_paths(submitted, expected)
    if differences:
        raise ValueError(
            "student plan differs from the instructor reference outside allowed "
            "candidate settings: " + ", ".join(differences[:8])
        )
    return {
        "schema": INSTRUCTOR_BINDING_SCHEMA,
        "status": "passed",
        "reference_source": reference_source,
        "reference_source_sha256": reference["source_sha256"],
        "submitted_source_sha256": plan["source_sha256"],
        "allowed_candidate_environment": copy.deepcopy(allowed),
        "accepted_changes": changes,
    }


def load_experiment_plan(path: Path) -> dict[str, Any]:
    """Load a versioned, fail-closed pro experiment plan."""
    source_bytes = path.read_bytes()
    payload = yaml.safe_load(source_bytes)
    if not isinstance(payload, dict):
        raise ValueError("experiment plan must be a mapping")
    schema = payload.get("schema")
    if schema not in SUPPORTED_EXPERIMENT_PLAN_SCHEMAS:
        raise ValueError(
            "experiment plan schema must be one of "
            f"{sorted(SUPPORTED_EXPERIMENT_PLAN_SCHEMAS)}"
        )
    allowed_top_level = {
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
    }
    if schema in EDIT_POLICY_SCHEMAS:
        allowed_top_level.add("edit_policy")
    _known_keys(
        payload,
        allowed=allowed_top_level,
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
    open_report = _boolean(output.get("open_report", False), label="output.open_report")
    power = _boolean(payload.get("power", False), label="experiment power")
    keep_going = _boolean(
        payload.get("keep_going", True), label="experiment keep_going"
    )
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("experiment plan runs must be a nonempty list")

    normalized_runs: list[dict[str, Any]] = []
    names: set[str] = set()
    allowed_run_fields = {
        "name",
        "role",
        "workload",
        "variant",
        "mode",
        "phase",
        "device",
        "repetitions",
        "environment",
    }
    if schema == EXPERIMENT_PLAN_SCHEMA:
        allowed_run_fields.add("baseline_import")
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"experiment run {index} must be a mapping")
        _known_keys(
            run,
            allowed=allowed_run_fields,
            label=f"experiment run {index}",
        )
        workload = _nonempty_string(
            run.get("workload"), label=f"experiment run {index} workload"
        )
        role = run.get("role", "condition")
        if role not in ROLES:
            raise ValueError(
                f"experiment run {index} role must be one of {sorted(ROLES)}"
            )
        variant = run.get("variant")
        if variant is not None:
            variant = _nonempty_string(variant, label=f"experiment run {index} variant")
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
        baseline_import = (
            _baseline_import(
                run.get("baseline_import"),
                label=f"experiment run {index} baseline_import",
            )
            if schema == EXPERIMENT_PLAN_SCHEMA
            else None
        )
        if baseline_import is not None and role != "baseline":
            raise ValueError(
                f"experiment run {index} baseline_import requires role: baseline"
            )
        if baseline_import is not None and repetitions != 1:
            raise ValueError(
                f"experiment run {index} baseline_import requires repetitions: 1"
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
        normalized_run = {
            "name": name,
            "role": role,
            "workload": workload,
            "variant": variant,
            "mode": mode,
            "phase": phase,
            "device": device,
            "repetitions": repetitions,
            "environment": environment,
        }
        if baseline_import is not None:
            normalized_run["baseline_import"] = baseline_import
        normalized_runs.append(normalized_run)

    comparison_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in normalized_runs:
        group = (
            run["workload"],
            run["variant"],
            run["mode"],
            run["phase"],
        )
        comparison_groups.setdefault(group, []).append(run)
    for group, grouped_runs in comparison_groups.items():
        baseline_count = sum(run["role"] == "baseline" for run in grouped_runs)
        candidate_count = sum(run["role"] == "candidate" for run in grouped_runs)
        group_label = "/".join(str(value or "default") for value in group)
        if baseline_count > 1:
            raise ValueError(
                f"experiment comparison group {group_label!r} can declare at most "
                "one baseline run"
            )
        if candidate_count and baseline_count != 1:
            raise ValueError(
                f"candidate runs in comparison group {group_label!r} require "
                "exactly one baseline run"
            )

    independent_variables = _string_list(
        study.get("independent_variables"),
        label="study independent_variables",
    )
    edit_policy = (
        _edit_policy(
            payload.get("edit_policy"),
            runs=normalized_runs,
            independent_variables=independent_variables,
        )
        if schema in EDIT_POLICY_SCHEMAS
        else {"allowed_candidate_environment": {}}
    )
    normalized = {
        "schema": schema,
        "id": plan_id,
        "title": str(payload.get("title") or plan_id),
        "description": str(payload.get("description") or ""),
        "study": {
            "question": str(study.get("question") or ""),
            "hypothesis": str(study.get("hypothesis") or ""),
            "independent_variables": independent_variables,
            "controls": _string_list(study.get("controls"), label="study controls"),
            "analysis_metrics": _string_list(
                study.get("analysis_metrics"), label="study analysis_metrics"
            ),
        },
        "profile": "pro",
        "power": power,
        "keep_going": keep_going,
        "output": {"directory": output_dir, "open_report": open_report},
        "edit_policy": edit_policy,
        "runs": normalized_runs,
        "source_sha256": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
    }
    json.dumps(normalized, allow_nan=False)
    return normalized
