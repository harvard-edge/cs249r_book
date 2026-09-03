#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "registry" / "selection-ledger.yaml"
STATUSES = {"admitted", "candidate", "deferred", "rejected"}
UPSTREAM_FIELDS = {
    "authority",
    "task",
    "model",
    "dataset",
    "split",
    "evaluator",
    "quality_target",
    "published_baseline",
    "provenance",
}
RATIONALE_FIELDS = {
    "task_significance",
    "benchmark_lineage",
    "classroom_value",
    "systems_behavior",
    "reason_for_model",
    "reason_for_dataset",
    "reason_for_metric",
    "alternatives_rejected",
}


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def construct_unique_mapping(
    loader: UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict:
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_unique_mapping
)


def validate(path: Path = LEDGER) -> list[str]:
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        return [f"selection ledger YAML is invalid: {exc}"]
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["selection ledger root must be a mapping"]
    if data.get("schema") != "mlperf-edu-workload-selection/0.1":
        errors.append("unexpected or missing selection-ledger schema")
    workloads = data.get("workloads")
    if not isinstance(workloads, dict) or not workloads:
        return [*errors, "workloads must be a nonempty mapping"]

    for name, entry in workloads.items():
        if not isinstance(entry, dict):
            errors.append(f"{name}: entry must be a mapping")
            continue
        status = entry.get("status")
        if status not in STATUSES:
            errors.append(f"{name}: invalid status {status!r}")
            continue
        if status == "rejected":
            if not entry.get("reason"):
                errors.append(f"{name}: rejected entries require a reason")
            continue

        upstream = entry.get("upstream") or {}
        missing_upstream = sorted(UPSTREAM_FIELDS - set(upstream))
        if missing_upstream:
            errors.append(
                f"{name}: missing upstream fields {', '.join(missing_upstream)}"
            )
        provenance = upstream.get("provenance")
        if not isinstance(provenance, list) or not provenance:
            errors.append(f"{name}: provenance must be a nonempty list")

        rationale = entry.get("rationale") or {}
        missing_rationale = sorted(RATIONALE_FIELDS - set(rationale))
        if missing_rationale:
            errors.append(
                f"{name}: missing rationale fields {', '.join(missing_rationale)}"
            )
        if not entry.get("laptop_evidence"):
            errors.append(f"{name}: laptop_evidence is required")
        if not entry.get("implementation_state"):
            errors.append(f"{name}: implementation_state is required")
        if status == "admitted" and entry.get("laptop_evidence") == "pending":
            errors.append(
                f"{name}: admitted workload cannot have pending laptop evidence"
            )

    return errors


def main() -> int:
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"selection ledger valid: {LEDGER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
