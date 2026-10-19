"""Deterministic data-selection experiments for Volume I, Chapter 9.

This module uses small, fixed teaching fixtures.  Candidate records have known
duplicate groups and cohort membership.  Quality values are supplied outcome
observations for those fixtures; they are not inferred from data volume, cost,
or hardware.  The scenarios are illustrative and are not production
measurements or claims about named datasets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from mlsysim.core.units import Q_, ureg
from mlsysim.physics.performance import dTime


TRACK_IDS = ("tinyml", "mobile", "edge", "cloud")
POLICY_IDS = ("uniform", "deduplicate", "coverage")
RETAINED_COUNTS = (6, 8, 10)
COHORTS = ("common", "rare", "edge_case")
MODEL_ID = "v1_09_experiments"


@dataclass(frozen=True)
class CandidateExample:
    """One representative candidate with auditable identity and membership."""

    example_id: str
    duplicate_group: str
    cohort: str
    size: Any


def _pool(prefix: str, size: Any) -> tuple[CandidateExample, ...]:
    rows = (
        ("01", "d01", "common"),
        ("02", "d01", "common"),
        ("03", "d03", "common"),
        ("04", "d04", "common"),
        ("05", "d05", "rare"),
        ("06", "d06", "rare"),
        ("07", "d06", "rare"),
        ("08", "d08", "edge_case"),
        ("09", "d09", "common"),
        ("10", "d10", "edge_case"),
        ("11", "d11", "common"),
        ("12", "d11", "common"),
    )
    return tuple(CandidateExample(f"{prefix}-{i}", group, cohort, size) for i, group, cohort in rows)


_RANKINGS = {
    "uniform": ("01", "03", "05", "08", "02", "06", "09", "11", "04", "07", "10", "12"),
    "deduplicate": ("01", "03", "04", "09", "11", "05", "08", "10", "06", "02", "07", "12"),
    "coverage": ("05", "08", "10", "06", "01", "03", "04", "09", "11", "02", "07", "12"),
}


def _outcomes(
    uniform: tuple[tuple[float, float, float], ...],
    deduplicate: tuple[tuple[float, float, float], ...],
    coverage: tuple[tuple[float, float, float], ...],
) -> dict[str, dict[int, dict[str, float]]]:
    return {
        policy: {
            retained: dict(zip(COHORTS, values, strict=True))
            for retained, values in zip(RETAINED_COUNTS, rows, strict=True)
        }
        for policy, rows in {
            "uniform": uniform,
            "deduplicate": deduplicate,
            "coverage": coverage,
        }.items()
    }


# Every numeric entry below is an explicitly supplied illustrative fixture.
# The code interpolates none of these observations and claims no general law.
TRACKS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "label": "TinyML",
        "workload": "wearable sensor windows",
        "pool": _pool("tw", Q_(48, "kilobyte")),
        "represented_examples": 120_000,
        "read_bandwidth": Q_(18, "megabyte / second"),
        "selection_time_per_example": Q_(1.8, "microsecond"),
        "scoring_time_per_example": Q_(8, "microsecond"),
        "scoring_time_min": Q_(0, "microsecond"),
        "scoring_time_max": Q_(100, "microsecond"),
        "scoring_time_step": Q_(1, "microsecond"),
        "training_flops_per_example": Q_(6.0e6, "flop"),
        "training_epochs": 12,
        "development_peak": Q_(1.8e12, "flop / second"),
        "efficiency": 0.32,
        "quality_floor_pct": 75.0,
        "cohort_floor_pct": 65.0,
        "population_pct": {"common": 76.0, "rare": 16.0, "edge_case": 8.0},
        "budget_min": Q_(0, "USD"),
        "budget_max": Q_(5000, "USD"),
        "budget_default": Q_(2000, "USD"),
        "budget_step": Q_(100, "USD"),
        "learning": {
            "redundant": ((2, 70.0), (4, 75.0), (6, 77.0), (8, 77.4), (10, 77.5), (12, 77.5)),
            "informative": ((2, 70.0), (4, 76.0), (6, 79.0), (8, 81.0), (10, 82.0), (12, 82.4)),
        },
        "outcomes": _outcomes(
            ((76, 55, 48), (78, 58, 52), (79, 61, 55)),
            ((77, 58, 54), (79, 62, 58), (80, 64, 60)),
            ((75, 64, 61), (78, 69, 66), (79, 71, 68)),
        ),
        "acquisition": {
            "labels": {"label": "Expert labels", "count": 900, "creation": Q_(2700, "USD"), "validation": Q_(450, "USD"), "turnaround": Q_(72, "hour"), "validation_count": 180, "outcomes": {"common": 80, "rare": 72, "edge_case": 68}},
            "generated": {"label": "Generated sensor windows", "count": 3600, "creation": Q_(720, "USD"), "validation": Q_(900, "USD"), "turnaround": Q_(30, "hour"), "validation_count": 900, "outcomes": {"common": 81, "rare": 65, "edge_case": 59}},
        },
    },
    "mobile": {
        "label": "Mobile",
        "workload": "on-device intent examples",
        "pool": _pool("mi", Q_(180, "kilobyte")),
        "represented_examples": 480_000,
        "read_bandwidth": Q_(75, "megabyte / second"),
        "selection_time_per_example": Q_(2.4, "microsecond"),
        "scoring_time_per_example": Q_(18, "microsecond"),
        "scoring_time_min": Q_(0, "microsecond"),
        "scoring_time_max": Q_(150, "microsecond"),
        "scoring_time_step": Q_(1, "microsecond"),
        "training_flops_per_example": Q_(2.2e8, "flop"),
        "training_epochs": 8,
        "development_peak": Q_(8.0e12, "flop / second"),
        "efficiency": 0.38,
        "quality_floor_pct": 81.0,
        "cohort_floor_pct": 72.0,
        "population_pct": {"common": 70.0, "rare": 20.0, "edge_case": 10.0},
        "budget_min": Q_(0, "USD"),
        "budget_max": Q_(10000, "USD"),
        "budget_default": Q_(4500, "USD"),
        "budget_step": Q_(250, "USD"),
        "learning": {
            "redundant": ((2, 76.0), (4, 81.0), (6, 82.4), (8, 82.8), (10, 82.9), (12, 82.9)),
            "informative": ((2, 76.0), (4, 82.0), (6, 85.0), (8, 87.0), (10, 88.0), (12, 88.5)),
        },
        "outcomes": _outcomes(
            ((82, 62, 57), (84, 66, 61), (85, 69, 64)),
            ((83, 65, 61), (85, 70, 66), (86, 72, 68)),
            ((81, 72, 68), (84, 77, 73), (85, 79, 75)),
        ),
        "acquisition": {
            "labels": {"label": "Reviewed user labels", "count": 2400, "creation": Q_(4800, "USD"), "validation": Q_(1200, "USD"), "turnaround": Q_(96, "hour"), "validation_count": 480, "outcomes": {"common": 86, "rare": 80, "edge_case": 75}},
            "generated": {"label": "Generated intent examples", "count": 9600, "creation": Q_(1900, "USD"), "validation": Q_(2400, "USD"), "turnaround": Q_(42, "hour"), "validation_count": 2400, "outcomes": {"common": 87, "rare": 73, "edge_case": 67}},
        },
    },
    "edge": {
        "label": "Edge",
        "workload": "industrial inspection frames",
        "pool": _pool("ef", Q_(1.6, "megabyte")),
        "represented_examples": 300_000,
        "read_bandwidth": Q_(240, "megabyte / second"),
        "selection_time_per_example": Q_(3.0, "microsecond"),
        "scoring_time_per_example": Q_(32, "microsecond"),
        "scoring_time_min": Q_(0, "microsecond"),
        "scoring_time_max": Q_(200, "microsecond"),
        "scoring_time_step": Q_(1, "microsecond"),
        "training_flops_per_example": Q_(7.5e8, "flop"),
        "training_epochs": 10,
        "development_peak": Q_(12e12, "flop / second"),
        "efficiency": 0.42,
        "quality_floor_pct": 86.0,
        "cohort_floor_pct": 78.0,
        "population_pct": {"common": 74.0, "rare": 18.0, "edge_case": 8.0},
        "budget_min": Q_(0, "USD"),
        "budget_max": Q_(25000, "USD"),
        "budget_default": Q_(12000, "USD"),
        "budget_step": Q_(500, "USD"),
        "learning": {
            "redundant": ((2, 80.0), (4, 85.0), (6, 86.5), (8, 86.9), (10, 87.0), (12, 87.0)),
            "informative": ((2, 80.0), (4, 86.0), (6, 89.0), (8, 91.0), (10, 92.0), (12, 92.6)),
        },
        "outcomes": _outcomes(
            ((86, 67, 61), (88, 71, 65), (89, 74, 69)),
            ((87, 70, 66), (89, 75, 71), (90, 77, 73)),
            ((85, 78, 74), (88, 83, 80), (89, 85, 82)),
        ),
        "acquisition": {
            "labels": {"label": "Engineer-reviewed labels", "count": 1200, "creation": Q_(14400, "USD"), "validation": Q_(3600, "USD"), "turnaround": Q_(120, "hour"), "validation_count": 240, "outcomes": {"common": 90, "rare": 87, "edge_case": 83}},
            "generated": {"label": "Generated defect frames", "count": 6000, "creation": Q_(4200, "USD"), "validation": Q_(7200, "USD"), "turnaround": Q_(60, "hour"), "validation_count": 1200, "outcomes": {"common": 91, "rare": 79, "edge_case": 72}},
        },
    },
    "cloud": {
        "label": "Cloud",
        "workload": "document passages for fine-tuning",
        "pool": _pool("cp", Q_(32, "kilobyte")),
        "represented_examples": 2_400_000,
        "read_bandwidth": Q_(420, "megabyte / second"),
        "selection_time_per_example": Q_(1.2, "microsecond"),
        "scoring_time_per_example": Q_(55, "microsecond"),
        "scoring_time_min": Q_(0, "microsecond"),
        "scoring_time_max": Q_(250, "microsecond"),
        "scoring_time_step": Q_(1, "microsecond"),
        "training_flops_per_example": Q_(2.8e9, "flop"),
        "training_epochs": 3,
        "development_peak": Q_(45e12, "flop / second"),
        "efficiency": 0.46,
        "quality_floor_pct": 79.0,
        "cohort_floor_pct": 70.0,
        "population_pct": {"common": 68.0, "rare": 22.0, "edge_case": 10.0},
        "budget_min": Q_(0, "USD"),
        "budget_max": Q_(50000, "USD"),
        "budget_default": Q_(25000, "USD"),
        "budget_step": Q_(1000, "USD"),
        "learning": {
            "redundant": ((2, 73.0), (4, 78.0), (6, 79.2), (8, 79.5), (10, 79.6), (12, 79.6)),
            "informative": ((2, 73.0), (4, 79.0), (6, 82.0), (8, 84.0), (10, 85.0), (12, 85.4)),
        },
        "outcomes": _outcomes(
            ((80, 60, 55), (82, 64, 59), (83, 67, 62)),
            ((81, 63, 59), (83, 68, 64), (84, 70, 66)),
            ((79, 71, 67), (82, 76, 72), (83, 78, 74)),
        ),
        "acquisition": {
            "labels": {"label": "Specialist passage labels", "count": 5000, "creation": Q_(25000, "USD"), "validation": Q_(7500, "USD"), "turnaround": Q_(144, "hour"), "validation_count": 1000, "outcomes": {"common": 84, "rare": 80, "edge_case": 76}},
            "generated": {"label": "Generated document passages", "count": 30000, "creation": Q_(9000, "USD"), "validation": Q_(15000, "USD"), "turnaround": Q_(54, "hour"), "validation_count": 6000, "outcomes": {"common": 85, "rare": 72, "edge_case": 65}},
        },
    },
}


def _scenario(track_id: str) -> dict[str, Any]:
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACK_IDS)}")
    return TRACKS[track_id]


def _policy(policy_id: str) -> str:
    if policy_id not in POLICY_IDS:
        raise ValueError(f"policy_id must be one of {', '.join(POLICY_IDS)}")
    return policy_id


def _retained_count(retained_count: int) -> int:
    if (
        isinstance(retained_count, bool)
        or not isinstance(retained_count, int)
        or retained_count not in RETAINED_COUNTS
    ):
        choices = ", ".join(str(value) for value in RETAINED_COUNTS)
        raise ValueError(f"retained_count must be one of {choices}")
    return retained_count


def _weighted_quality(outcomes: Mapping[str, float], population_pct: Mapping[str, float]) -> float:
    return sum(outcomes[cohort] * population_pct[cohort] / 100.0 for cohort in COHORTS)


def serialize_value(value: object):
    """Convert experiment data to JSON-safe values without discarding units."""
    if isinstance(value, ureg.Quantity):
        magnitude = value.magnitude
        if not isinstance(magnitude, (int, float)):
            raise TypeError("only scalar Pint quantities can be serialized")
        if not math.isfinite(float(magnitude)):
            raise ValueError("quantity magnitude must be finite")
        return {
            "__type__": "quantity",
            "magnitude": magnitude,
            "unit": f"{value.units:~}",
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: serialize_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize experiment value of type {type(value).__name__}")


def restore_quantities(value: object):
    """Restore Pint quantities inside data produced by :func:`serialize_value`."""
    if isinstance(value, Mapping):
        if value.get("__type__") == "quantity":
            if set(value) != {"__type__", "magnitude", "unit"}:
                raise ValueError("serialized quantity has unexpected fields")
            return Q_(value["magnitude"], value["unit"])
        return {str(key): restore_quantities(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_quantities(item) for item in value]
    return value


def _selected_examples(scenario: Mapping[str, Any], policy_id: str, retained_count: int) -> tuple[CandidateExample, ...]:
    by_suffix = {example.example_id.rsplit("-", 1)[1]: example for example in scenario["pool"]}
    return tuple(by_suffix[suffix] for suffix in _RANKINGS[policy_id][:retained_count])


def learning_curve(track_id: str, pool_kind: str) -> dict[str, Any]:
    """Return supplied observations and adjacent marginal changes."""
    scenario = _scenario(track_id)
    if pool_kind not in {"redundant", "informative"}:
        raise ValueError("pool_kind must be redundant or informative")
    observations = scenario["learning"][pool_kind]
    points = []
    previous_quality = None
    for retained, quality in observations:
        points.append(
            {
                "retained_count": retained,
                "quality_pct": quality,
                "marginal_quality_pp": None if previous_quality is None else round(quality - previous_quality, 4),
            }
        )
        previous_quality = quality
    return {
        "track_id": track_id,
        "model_id": MODEL_ID,
        "model_key": "learning_curve",
        "pool_kind": pool_kind,
        "inputs": {"track_id": track_id, "pool_kind": pool_kind},
        "points": tuple(points),
        "observation_label": "Supplied illustrative learning outcomes; not measured benchmark results.",
    }


def evaluate_policy(track_id: str, policy_id: str, retained_count: int = 8) -> dict[str, Any]:
    """Evaluate one selection using exact record counts and supplied outcomes."""
    scenario = _scenario(track_id)
    policy_id = _policy(policy_id)
    retained_count = _retained_count(retained_count)
    selected = _selected_examples(scenario, policy_id, retained_count)
    duplicate_counts: dict[str, int] = {}
    cohort_counts = {cohort: 0 for cohort in COHORTS}
    for example in selected:
        duplicate_counts[example.duplicate_group] = duplicate_counts.get(example.duplicate_group, 0) + 1
        cohort_counts[example.cohort] += 1
    repeated_records = sum(count - 1 for count in duplicate_counts.values() if count > 1)
    outcomes = scenario["outcomes"][policy_id][retained_count]
    weighted_quality = _weighted_quality(outcomes, scenario["population_pct"])
    uncovered = tuple(cohort for cohort, count in cohort_counts.items() if count == 0)
    below_floor = tuple(cohort for cohort, value in outcomes.items() if value < scenario["cohort_floor_pct"])
    failures = []
    if repeated_records:
        failures.append(f"{repeated_records} retained record(s) repeat a duplicate group")
    if uncovered:
        failures.append(f"no selected example for: {', '.join(uncovered)}")
    if weighted_quality < scenario["quality_floor_pct"]:
        failures.append(
            f"population-weighted quality {weighted_quality:.1f}% is below {scenario['quality_floor_pct']:.1f}%"
        )
    if below_floor:
        failures.append(f"cohort outcome below floor for: {', '.join(below_floor)}")
    return {
        "track_id": track_id,
        "model_id": MODEL_ID,
        "model_key": "evaluate_policy",
        "policy_id": policy_id,
        "retained_count": retained_count,
        "inputs": {
            "track_id": track_id,
            "policy_id": policy_id,
            "retained_count": retained_count,
        },
        "selected_ids": tuple(example.example_id for example in selected),
        "cohort_counts": cohort_counts,
        "duplicate_records": repeated_records,
        "retained_bytes": sum((example.size for example in selected), Q_(0, "byte")).to("megabyte"),
        "cohort_outcomes_pct": dict(outcomes),
        "population_pct": dict(scenario["population_pct"]),
        "weighted_quality_pct": round(weighted_quality, 4),
        "quality_floor_pct": scenario["quality_floor_pct"],
        "cohort_floor_pct": scenario["cohort_floor_pct"],
        "feasible": not failures,
        "failures": tuple(failures),
        "observation_label": "Supplied illustrative cohort outcomes; not measured benchmark results.",
    }


def compare_policies(track_id: str, retained_count: int = 8) -> dict[str, Any]:
    """Compare policies at exactly the same retained count."""
    retained_count = _retained_count(retained_count)
    return {
        "track_id": track_id,
        "model_id": MODEL_ID,
        "model_key": "compare_policies",
        "retained_count": retained_count,
        "inputs": {"track_id": track_id, "retained_count": retained_count},
        "policies": tuple(evaluate_policy(track_id, policy_id, retained_count) for policy_id in POLICY_IDS),
    }


def full_pool_baseline(track_id: str) -> dict[str, Any]:
    """Return the unchanged candidate pool when no selection policy is accepted."""
    scenario = _scenario(track_id)
    cohort_counts = {cohort: 0 for cohort in COHORTS}
    duplicate_counts: dict[str, int] = {}
    for example in scenario["pool"]:
        cohort_counts[example.cohort] += 1
        duplicate_counts[example.duplicate_group] = duplicate_counts.get(example.duplicate_group, 0) + 1
    duplicate_records = sum(count - 1 for count in duplicate_counts.values() if count > 1)
    return {
        "track_id": track_id,
        "model_id": MODEL_ID,
        "model_key": "full_pool_baseline",
        "inputs": {"track_id": track_id},
        "status": "selection_not_applied",
        "retained_count": len(scenario["pool"]),
        "selected_ids": tuple(example.example_id for example in scenario["pool"]),
        "cohort_counts": cohort_counts,
        "duplicate_records": duplicate_records,
        "retained_bytes": sum((example.size for example in scenario["pool"]), Q_(0, "byte")).to("megabyte"),
        "cohort_outcomes_pct": None,
        "weighted_quality_pct": None,
        "quality_status": "unavailable: no supplied full-pool cohort outcome fixture",
        "feasible": None,
        "failures": (),
    }


def selection_amortization(
    track_id: str,
    policy_id: str,
    retained_count: int = 8,
    scoring_time_per_example: Any | None = None,
    repeated_runs: int = 1,
    comparison_path: str = "selected_subset",
) -> dict[str, Any]:
    """Compare selection plus repeated subset training with full-pool training."""
    scenario = _scenario(track_id)
    policy_id = _policy(policy_id)
    retained_count = _retained_count(retained_count)
    if isinstance(repeated_runs, bool) or not isinstance(repeated_runs, int) or repeated_runs < 1:
        raise ValueError("repeated_runs must be an integer of at least 1")
    if comparison_path not in {"full_pool", "selected_subset"}:
        raise ValueError("comparison_path must be full_pool or selected_subset")
    scoring = scenario["scoring_time_per_example"] if scoring_time_per_example is None else scoring_time_per_example
    try:
        scoring = scoring.to("second")
    except Exception as exc:
        raise ValueError("scoring_time_per_example must be a time Quantity") from exc
    if scoring.magnitude < 0:
        raise ValueError("scoring_time_per_example cannot be negative")

    representative_examples = scenario["represented_examples"]
    full_records = len(scenario["pool"])
    retained_fraction = retained_count / full_records
    selected_examples = representative_examples * retained_fraction
    total_bytes = sum((example.size for example in scenario["pool"]), Q_(0, "byte"))
    scaled_bytes = total_bytes * (representative_examples / full_records)
    read_time = (scaled_bytes / scenario["read_bandwidth"]).to("second")
    scoring_time = scoring * representative_examples
    selection_time = scenario["selection_time_per_example"] * representative_examples

    full_ops = scenario["training_flops_per_example"] * representative_examples * scenario["training_epochs"]
    subset_ops = scenario["training_flops_per_example"] * selected_examples * scenario["training_epochs"]
    full_train = dTime(full_ops, 1, scenario["development_peak"], scenario["efficiency"])
    subset_train = dTime(subset_ops, 1, scenario["development_peak"], scenario["efficiency"])
    selection_overhead = (read_time + scoring_time + selection_time).to("second")
    baseline_total = (full_train * repeated_runs).to("second")
    selected_total = (selection_overhead + subset_train * repeated_runs).to("second")
    per_run_saving = (full_train - subset_train).to("second")
    if per_run_saving.magnitude > 0:
        overhead_to_saving = (selection_overhead / per_run_saving).to_base_units().magnitude
        first_profitable_runs = math.floor(overhead_to_saving) + 1
    else:
        first_profitable_runs = None
    policy_evaluation = evaluate_policy(track_id, policy_id, retained_count)
    time_saved = (baseline_total - selected_total).to("second")
    selection_pays = time_saved.magnitude > 1e-9
    return {
        "track_id": track_id,
        "model_id": MODEL_ID,
        "model_key": "selection_amortization",
        "policy_id": policy_id,
        "retained_count": retained_count,
        "inputs": {
            "track_id": track_id,
            "policy_id": policy_id,
            "retained_count": retained_count,
            "scoring_time_per_example": scoring,
            "repeated_runs": repeated_runs,
            "comparison_path": comparison_path,
        },
        "retained_fraction": retained_fraction,
        "repeated_runs": repeated_runs,
        "read_time": read_time,
        "scoring_time": scoring_time.to("second"),
        "selection_time": selection_time.to("second"),
        "selection_overhead": selection_overhead,
        "full_training_time_per_run": full_train.to("second"),
        "subset_training_time_per_run": subset_train.to("second"),
        "baseline_total": baseline_total,
        "selected_total": selected_total,
        "time_saved": time_saved,
        "first_profitable_runs": first_profitable_runs,
        "selection_pays": selection_pays,
        "policy_evaluation": policy_evaluation,
        "quality_and_coverage_accepted": policy_evaluation["feasible"],
        "decision_supported": selection_pays and policy_evaluation["feasible"],
        "reported_path": comparison_path,
        "reported_total": baseline_total if comparison_path == "full_pool" else selected_total,
        "training_assumption": "Fixed epochs and per-example work on an illustrative development host; this is not a time-to-common-quality comparison.",
        "io_boundary": "Read time is one extra selection scan. Full and subset training times exclude training I/O.",
        "comparison_limitation": "A time saving supports the selection only when the supplied quality and coverage evidence also passes.",
    }


def scoring_time_policy(track_id: str) -> dict[str, Any]:
    """Return scoring-time slider bounds, step, and default for the track."""
    scenario = _scenario(track_id)
    return {
        "track_id": track_id,
        "min_scoring_time": scenario["scoring_time_min"],
        "max_scoring_time": scenario["scoring_time_max"],
        "default_scoring_time": scenario["scoring_time_per_example"],
        "step_scoring_time": scenario["scoring_time_step"],
    }


def acquisition_budget_policy(track_id: str) -> dict[str, Any]:
    """Return budget slider range, step, and default scaled from package costs."""
    scenario = _scenario(track_id)
    return {
        "track_id": track_id,
        "min_budget": scenario["budget_min"],
        "max_budget": scenario["budget_max"],
        "default_budget": scenario["budget_default"],
        "step_budget": scenario["budget_step"],
    }


def acquisition_options(track_id: str, budget: Any) -> dict[str, Any]:
    """Compare one fixed label package with one fixed generated-data package."""
    scenario = _scenario(track_id)
    try:
        budget = budget.to("USD")
    except Exception as exc:
        raise ValueError("budget must be a currency Quantity") from exc
    if budget.magnitude < 0:
        raise ValueError("budget cannot be negative")
    options = []
    for option_id, fixture in scenario["acquisition"].items():
        total_cost = (fixture["creation"] + fixture["validation"]).to("USD")
        outcomes = fixture["outcomes"]
        weighted = _weighted_quality(outcomes, scenario["population_pct"])
        below_floor = tuple(
            cohort for cohort, value in outcomes.items() if value < scenario["cohort_floor_pct"]
        )
        options.append(
            {
                "option_id": option_id,
                "label": fixture["label"],
                "created_examples": fixture["count"],
                "validated_examples": fixture["validation_count"],
                "creation_cost": fixture["creation"],
                "validation_cost": fixture["validation"],
                "total_cost": total_cost,
                "turnaround": fixture["turnaround"],
                "cohort_outcomes_pct": dict(outcomes),
                "weighted_quality_pct": round(weighted, 4),
                "below_cohort_floor": below_floor,
                "affordable": total_cost <= budget,
                "observation_label": "Supplied illustrative package outcome; no extrapolation to other package sizes.",
            }
        )
    return {
        "track_id": track_id,
        "model_id": MODEL_ID,
        "model_key": "acquisition_options",
        "budget": budget,
        "inputs": {"track_id": track_id, "budget": budget},
        "options": tuple(options),
    }


def acquisition_option(track_id: str, option_id: str, budget: Any) -> dict[str, Any]:
    """Return one acquisition candidate with replayable option identity."""
    if option_id not in {"none", "labels", "generated"}:
        raise ValueError("option_id must be none, labels, or generated")
    comparison = acquisition_options(track_id, budget)
    if option_id == "none":
        return {
            "track_id": track_id,
            "model_id": MODEL_ID,
            "model_key": "acquisition_option",
            "inputs": {"track_id": track_id, "option_id": option_id, "budget": comparison["budget"]},
            "option_id": "none",
            "label": "No purchase",
            "status": "no_acquisition_selected",
            "created_examples": 0,
            "validated_examples": 0,
            "creation_cost": Q_(0, "USD"),
            "validation_cost": Q_(0, "USD"),
            "total_cost": Q_(0, "USD"),
            "turnaround": Q_(0, "hour"),
            "cohort_outcomes_pct": None,
            "weighted_quality_pct": None,
            "quality_status": "unavailable: no acquisition outcome",
            "below_cohort_floor": (),
            "affordable": True,
            "budget": comparison["budget"],
        }
    selected = next(option for option in comparison["options"] if option["option_id"] == option_id)
    return {
        **selected,
        "track_id": track_id,
        "model_id": MODEL_ID,
        "model_key": "acquisition_option",
        "budget": comparison["budget"],
        "inputs": {
            "track_id": track_id,
            "option_id": option_id,
            "budget": comparison["budget"],
        },
    }


def population_shift(track_id: str, policy_id: str, retained_count: int, rare_share_pct: float) -> dict[str, Any]:
    """Reweight fixed cohort outcomes under a changed population mix."""
    scenario = _scenario(track_id)
    evaluated = evaluate_policy(track_id, policy_id, retained_count)
    if isinstance(rare_share_pct, bool) or not isinstance(rare_share_pct, (int, float)):
        raise ValueError("rare_share_pct must be numeric")
    rare_share_pct = float(rare_share_pct)
    edge_share = scenario["population_pct"]["edge_case"]
    if rare_share_pct < 0 or rare_share_pct + edge_share > 100:
        raise ValueError("rare_share_pct leaves an invalid common-cohort share")
    changed_population = {
        "common": 100.0 - rare_share_pct - edge_share,
        "rare": rare_share_pct,
        "edge_case": edge_share,
    }
    outcomes = evaluated["cohort_outcomes_pct"]
    baseline_quality = _weighted_quality(outcomes, scenario["population_pct"])
    changed_quality = _weighted_quality(outcomes, changed_population)
    selected_shares = {
        cohort: 100.0 * evaluated["cohort_counts"][cohort] / retained_count for cohort in COHORTS
    }
    underrepresented = tuple(
        cohort
        for cohort in COHORTS
        if changed_population[cohort] > scenario["population_pct"][cohort]
        and selected_shares[cohort] < changed_population[cohort]
    )
    failures = []
    if changed_quality < scenario["quality_floor_pct"]:
        failures.append(
            f"changed-population quality {changed_quality:.1f}% is below {scenario['quality_floor_pct']:.1f}%"
        )
    if underrepresented:
        failures.append(f"selected share trails population share for: {', '.join(underrepresented)}")
    return {
        "track_id": track_id,
        "model_id": MODEL_ID,
        "model_key": "population_shift",
        "policy_id": policy_id,
        "retained_count": retained_count,
        "inputs": {
            "track_id": track_id,
            "policy_id": policy_id,
            "retained_count": retained_count,
            "rare_share_pct": rare_share_pct,
        },
        "baseline_population_pct": dict(scenario["population_pct"]),
        "changed_population_pct": changed_population,
        "selected_cohort_pct": selected_shares,
        "cohort_outcomes_pct": outcomes,
        "baseline_quality_pct": round(baseline_quality, 4),
        "changed_quality_pct": round(changed_quality, 4),
        "quality_change_pp": round(changed_quality - baseline_quality, 4),
        "underrepresented_cohorts": underrepresented,
        "still_supported": not failures,
        "failures": tuple(failures),
        "observation_label": "Fixed supplied cohort outcomes reweighted to an illustrative changed population.",
    }


def replay(model_key: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Replay a recognized evaluator from raw or JSON-restored input values."""
    evaluators = {
        "learning_curve": learning_curve,
        "evaluate_policy": evaluate_policy,
        "compare_policies": compare_policies,
        "full_pool_baseline": full_pool_baseline,
        "selection_amortization": selection_amortization,
        "acquisition_options": acquisition_options,
        "acquisition_option": acquisition_option,
        "population_shift": population_shift,
    }
    try:
        evaluator = evaluators[model_key]
    except KeyError as exc:
        raise ValueError(f"unknown model_key for {MODEL_ID}: {model_key}") from exc
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping")
    decoded = restore_quantities(dict(inputs))
    return evaluator(**decoded)


__all__ = [
    "COHORTS",
    "MODEL_ID",
    "POLICY_IDS",
    "RETAINED_COUNTS",
    "TRACK_IDS",
    "TRACKS",
    "CandidateExample",
    "acquisition_budget_policy",
    "acquisition_options",
    "acquisition_option",
    "compare_policies",
    "evaluate_policy",
    "full_pool_baseline",
    "learning_curve",
    "population_shift",
    "replay",
    "restore_quantities",
    "scoring_time_policy",
    "selection_amortization",
    "serialize_value",
]
