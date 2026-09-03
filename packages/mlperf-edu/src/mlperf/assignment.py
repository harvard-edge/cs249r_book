from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


ASSIGNMENT_SCHEMA = "mlperf-edu-assignment/0.1"
SELECTOR_FIELDS = (
    "workload",
    "canonical_workload",
    "variant",
    "profile",
    "mode",
    "phase",
)


def load_assignment_contract(path: Path) -> dict[str, Any]:
    """Load and validate a versioned classroom assignment contract."""
    source_bytes = path.read_bytes()
    payload = yaml.safe_load(source_bytes)
    if not isinstance(payload, dict):
        raise ValueError("assignment contract must be a mapping")
    unknown_top_level = set(payload) - {
        "schema",
        "id",
        "title",
        "allow_extra_results",
        "requirements",
    }
    if unknown_top_level:
        raise ValueError(
            f"assignment contract has unknown fields: {sorted(unknown_top_level)}"
        )
    if payload.get("schema") != ASSIGNMENT_SCHEMA:
        raise ValueError(f"assignment schema must be {ASSIGNMENT_SCHEMA!r}")
    assignment_id = payload.get("id")
    if not isinstance(assignment_id, str) or not assignment_id.strip():
        raise ValueError("assignment id must be a nonempty string")
    allow_extra = payload.get("allow_extra_results", False)
    if not isinstance(allow_extra, bool):
        raise ValueError("allow_extra_results must be a boolean")
    requirements = payload.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("assignment requirements must be a nonempty list")

    normalized_requirements: list[dict[str, Any]] = []
    selector_keys: set[tuple[tuple[str, Any], ...]] = set()
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, dict):
            raise ValueError(f"assignment requirement {index + 1} must be a mapping")
        unknown_requirement = set(requirement) - {
            *SELECTOR_FIELDS,
            "count",
            "quality",
            "config",
        }
        if unknown_requirement:
            raise ValueError(
                f"assignment requirement {index + 1} has unknown fields: "
                f"{sorted(unknown_requirement)}"
            )
        workload = requirement.get("workload")
        if not isinstance(workload, str) or not workload.strip():
            raise ValueError(f"assignment requirement {index + 1} needs a workload")
        normalized = dict(requirement)
        for field in SELECTOR_FIELDS:
            if field not in normalized:
                continue
            value = normalized[field]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"assignment requirement {index + 1} field {field!r} "
                    "must be a nonempty string or null"
                )
        count = normalized.get("count", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(
                f"assignment requirement {index + 1} count must be a positive integer"
            )
        normalized["count"] = count
        quality = normalized.get("quality", {})
        if not isinstance(quality, dict):
            raise ValueError(
                f"assignment requirement {index + 1} quality must be a mapping"
            )
        unknown_quality = set(quality) - {"required", "target_met"}
        if unknown_quality:
            raise ValueError(
                f"assignment requirement {index + 1} has unknown quality fields: "
                f"{sorted(unknown_quality)}"
            )
        if any(not isinstance(value, bool) for value in quality.values()):
            raise ValueError(
                f"assignment requirement {index + 1} quality values must be booleans"
            )
        normalized["quality"] = quality
        config = normalized.get("config", {})
        if not isinstance(config, dict):
            raise ValueError(
                f"assignment requirement {index + 1} config must be a mapping"
            )
        normalized["config"] = config
        selector_key = tuple(
            (field, normalized[field])
            for field in SELECTOR_FIELDS
            if field in normalized
        )
        if selector_key in selector_keys:
            raise ValueError(
                f"assignment requirement {index + 1} duplicates an earlier selector"
            )
        selector_keys.add(selector_key)
        normalized_requirements.append(normalized)

    return {
        "schema": ASSIGNMENT_SCHEMA,
        "id": assignment_id,
        "source_sha256": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        "title": str(payload.get("title") or assignment_id),
        "allow_extra_results": allow_extra,
        "requirements": normalized_requirements,
    }


def _selector_matches(requirement: dict[str, Any], row: dict[str, Any]) -> bool:
    return all(
        field not in requirement or row.get(field) == requirement[field]
        for field in SELECTOR_FIELDS
    )


def _config_mismatches(
    expected: dict[str, Any], actual: Any, *, prefix: str = "config"
) -> list[str]:
    if not isinstance(actual, dict):
        return [f"{prefix} is missing"]
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        path = f"{prefix}.{key}"
        if key not in actual:
            mismatches.append(f"{path} is missing")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            mismatches.extend(
                _config_mismatches(expected_value, actual_value, prefix=path)
            )
        elif actual_value != expected_value:
            mismatches.append(
                f"{path} is {actual_value!r}, expected {expected_value!r}"
            )
    return mismatches


def evaluate_assignment_contract(
    contract: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate verified grade rows against workload and configuration rules."""
    requirement_results: list[dict[str, Any]] = []
    matched_indices: set[int] = set()
    for requirement_index, requirement in enumerate(contract["requirements"]):
        selected = [
            (row_index, row)
            for row_index, row in enumerate(rows)
            if _selector_matches(requirement, row)
        ]
        matched_indices.update(row_index for row_index, _ in selected)
        errors: list[str] = []
        expected_count = int(requirement["count"])
        if len(selected) != expected_count:
            errors.append(
                f"matched {len(selected)} result(s), expected {expected_count}"
            )
        quality = requirement.get("quality") or {}
        for _, row in selected:
            label = str(
                row.get("manifest") or row.get("package") or row.get("workload")
            )
            if not row.get("verified"):
                errors.append(f"{label} did not verify")
            if not row.get("passed"):
                errors.append(f"{label} did not pass its declared result contract")
            if (
                "required" in quality
                and row.get("quality_required") is not quality["required"]
            ):
                errors.append(
                    f"{label} quality.required is {row.get('quality_required')!r}, "
                    f"expected {quality['required']!r}"
                )
            if (
                "target_met" in quality
                and row.get("target_met") is not quality["target_met"]
            ):
                errors.append(
                    f"{label} quality.target_met is {row.get('target_met')!r}, "
                    f"expected {quality['target_met']!r}"
                )
            errors.extend(
                f"{label} {message}"
                for message in _config_mismatches(
                    requirement.get("config") or {}, row.get("config") or {}
                )
            )
        selector = {
            field: requirement[field]
            for field in SELECTOR_FIELDS
            if field in requirement
        }
        requirement_results.append(
            {
                "requirement": requirement_index + 1,
                "selector": selector,
                "expected_count": expected_count,
                "matched_count": len(selected),
                "passed": not errors,
                "errors": errors,
            }
        )

    extra_indices = [
        index for index in range(len(rows)) if index not in matched_indices
    ]
    extra_results = [
        str(rows[index].get("manifest") or rows[index].get("package") or "unknown")
        for index in extra_indices
    ]
    errors = [
        error for requirement in requirement_results for error in requirement["errors"]
    ]
    if extra_results and not contract["allow_extra_results"]:
        errors.append(f"unexpected result(s): {extra_results}")
    return {
        "schema": "mlperf-edu-assignment-grade/0.1",
        "assignment_schema": contract["schema"],
        "assignment_id": contract["id"],
        "assignment_source_sha256": contract["source_sha256"],
        "title": contract["title"],
        "allow_extra_results": contract["allow_extra_results"],
        "passed": not errors,
        "requirements": requirement_results,
        "extra_results": extra_results,
        "errors": errors,
    }
