"""Repeatable benchmarking experiments for the Volume I benchmarking lab.

The scenarios in this module are bounded teaching fixtures.  They are not
measurements of branded hardware or production applications.  Their purpose is
to make benchmark scope, warmup, drift, repeated runs, quality parity, and
protocol comparability causal and inspectable.

Timing and quality observations are seeded and paired: a baseline and an
intervention see the same synthetic disturbance or example.  The module reports
sample statistics only.  It deliberately does not attach an IID confidence
interval to deterministic traces that contain warmup or drift.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from pint.errors import DimensionalityError

from mlsysim.core.units import Q_, ureg


SCENARIO_ASSUMPTION = (
    "Illustrative seeded teaching fixture; not an empirical measurement of a "
    "named device, model, or production service."
)


@dataclass(frozen=True)
class BenchmarkScenario:
    """One bounded workload and measurement environment."""

    display_name: str
    workload_id: str
    preprocess: Any
    kernel: Any
    transfer: Any
    postprocess: Any
    candidate_kernel_speedup: float
    candidate_extra_overhead: Any
    deadline: Any
    baseline_power: Any
    candidate_power: Any
    jitter_fraction: float
    run_jitter_fraction: float
    warmup_fraction: float
    warmup_decay_iterations: float
    baseline_drift_fraction: float
    candidate_drift_fraction: float
    candidate_thermal_onset: float
    quality_floor_pct: float
    seed: int


TRACKS: dict[str, BenchmarkScenario] = {
    "tinyml": BenchmarkScenario(
        "TinyML", "always-on acoustic windows", Q_(1.8, "ms"), Q_(8.0, "ms"),
        Q_(0.7, "ms"), Q_(0.9, "ms"), 2.0, Q_(0.5, "ms"), Q_(14, "ms"),
        Q_(0.055, "W"), Q_(0.072, "W"), 0.018, 0.012, 0.20, 4.0,
        0.025, 0.20, 0.55, 82.0, 1201,
    ),
    "mobile": BenchmarkScenario(
        "Mobile", "interactive camera requests", Q_(7.0, "ms"), Q_(15.0, "ms"),
        Q_(3.0, "ms"), Q_(4.0, "ms"), 2.4, Q_(2.0, "ms"), Q_(31, "ms"),
        Q_(2.8, "W"), Q_(4.2, "W"), 0.025, 0.018, 0.24, 5.0,
        0.035, 0.24, 0.48, 85.0, 1202,
    ),
    "edge": BenchmarkScenario(
        "Edge", "sensor inspection frames", Q_(4.0, "ms"), Q_(18.0, "ms"),
        Q_(4.0, "ms"), Q_(3.0, "ms"), 2.6, Q_(2.5, "ms"), Q_(34, "ms"),
        Q_(11.0, "W"), Q_(16.0, "W"), 0.032, 0.020, 0.18, 4.0,
        0.030, 0.28, 0.46, 85.0, 1203,
    ),
    "cloud": BenchmarkScenario(
        "Cloud", "single-node online requests", Q_(3.0, "ms"), Q_(12.0, "ms"),
        Q_(2.0, "ms"), Q_(3.0, "ms"), 3.0, Q_(3.0, "ms"), Q_(23, "ms"),
        Q_(210, "W"), Q_(295, "W"), 0.040, 0.025, 0.16, 3.5,
        0.035, 0.30, 0.42, 84.0, 1204,
    ),
}


def _scenario(track_id: str) -> BenchmarkScenario:
    try:
        return TRACKS[track_id]
    except KeyError as exc:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}") from exc


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return float(value)


def _ms(value: Any) -> float:
    return value.to(ureg.millisecond).magnitude


def to_jsonable(value: Any) -> Any:
    """Convert experiment evidence to JSON values without discarding units."""

    if hasattr(value, "magnitude") and hasattr(value, "units"):
        magnitude = value.magnitude
        return {
            "magnitude": float(magnitude) if isinstance(magnitude, (int, float, np.number)) else magnitude,
            "units": str(value.units),
        }
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _time_quantity(value: Any, name: str) -> Any:
    if isinstance(value, dict) and set(value) == {"magnitude", "units"}:
        value = Q_(value["magnitude"], value["units"])
    try:
        return value.to(ureg.second)
    except (AttributeError, DimensionalityError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive Pint time quantity or serialized quantity") from exc


def _sample_summary(values_ms: list[float]) -> dict[str, Any]:
    values = np.asarray(values_ms, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "count": len(values_ms),
        "mean": Q_(mean, "ms"),
        "median": Q_(float(np.median(values)), "ms"),
        "p95": Q_(float(np.quantile(values, 0.95)), "ms"),
        "p99": Q_(float(np.quantile(values, 0.99)), "ms"),
        "std": Q_(std, "ms"),
        "cv_pct": 100.0 * std / mean if mean else 0.0,
    }


def _paired_noise(count: int, seed: int, jitter_fraction: float) -> list[float]:
    rng = random.Random(seed)
    return [max(0.75, 1.0 + rng.gauss(0.0, jitter_fraction)) for _ in range(count)]


def compare_scopes(
    track_id: str,
    *,
    sample_count: int = 80,
    kernel_speedup: float | None = None,
    workload_scale: float = 1.0,
    seed_offset: int = 0,
    reported_scope: str = "both",
) -> dict[str, Any]:
    """Compare an isolated kernel claim with the same paired whole-path trace."""

    scenario = _scenario(track_id)
    sample_count = _positive_int(sample_count, "sample_count")
    workload_scale = _finite_positive(workload_scale, "workload_scale")
    speedup = scenario.candidate_kernel_speedup if kernel_speedup is None else _finite_positive(kernel_speedup, "kernel_speedup")
    if speedup < 1.0:
        raise ValueError("kernel_speedup must be at least 1")
    if reported_scope not in {"kernel", "whole_path", "both"}:
        raise ValueError("reported_scope must be kernel, whole_path, or both")

    kernel_ms = _ms(scenario.kernel) * workload_scale
    non_kernel_ms = sum(_ms(value) for value in (
        scenario.preprocess, scenario.transfer, scenario.postprocess
    )) * workload_scale
    candidate_extra_ms = _ms(scenario.candidate_extra_overhead) * workload_scale
    noise = _paired_noise(sample_count, scenario.seed + seed_offset, scenario.jitter_fraction)
    baseline_kernel = [kernel_ms * disturbance for disturbance in noise]
    candidate_kernel = [(kernel_ms / speedup) * disturbance for disturbance in noise]
    baseline_path = [(non_kernel_ms + kernel_ms) * disturbance for disturbance in noise]
    candidate_path = [(non_kernel_ms + kernel_ms / speedup + candidate_extra_ms) * disturbance for disturbance in noise]

    baseline_kernel_summary = _sample_summary(baseline_kernel)
    candidate_kernel_summary = _sample_summary(candidate_kernel)
    baseline_path_summary = _sample_summary(baseline_path)
    candidate_path_summary = _sample_summary(candidate_path)
    kernel_ratio = baseline_kernel_summary["median"] / candidate_kernel_summary["median"]
    whole_path_ratio = baseline_path_summary["median"] / candidate_path_summary["median"]
    reported_summary = {
        "kernel": candidate_kernel_summary,
        "whole_path": candidate_path_summary,
        "both": {"kernel": candidate_kernel_summary, "whole_path": candidate_path_summary},
    }[reported_scope]
    reported_speedup = {
        "kernel": kernel_ratio,
        "whole_path": whole_path_ratio,
        "both": {"kernel": kernel_ratio, "whole_path": whole_path_ratio},
    }[reported_scope]
    return {
        "track_id": track_id,
        "scope": "paired kernel and whole request path",
        "baseline_kernel": baseline_kernel_summary,
        "candidate_kernel": candidate_kernel_summary,
        "baseline_path": baseline_path_summary,
        "candidate_path": candidate_path_summary,
        "kernel_speedup": kernel_ratio,
        "whole_path_speedup": whole_path_ratio,
        "reported_scope": reported_scope,
        "reported_summary": reported_summary,
        "reported_speedup": reported_speedup,
        "paired_delta": tuple(Q_(before - after, "ms") for before, after in zip(baseline_path, candidate_path)),
        "assumption": SCENARIO_ASSUMPTION,
        "inputs": {
            "track_id": track_id, "sample_count": sample_count, "kernel_speedup": speedup,
            "workload_scale": workload_scale, "seed_offset": seed_offset,
            "reported_scope": reported_scope,
        },
    }


def analyze_repeats(
    track_id: str,
    *,
    warmup_iterations: int = 8,
    measured_iterations: int = 40,
    repeat_count: int = 5,
    drift_scale: float = 1.0,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Generate repeated runs and report sample summaries without an IID interval."""

    scenario = _scenario(track_id)
    if isinstance(warmup_iterations, bool) or not isinstance(warmup_iterations, int) or warmup_iterations < 0:
        raise ValueError("warmup_iterations must be a nonnegative integer")
    measured_iterations = _positive_int(measured_iterations, "measured_iterations")
    repeat_count = _positive_int(repeat_count, "repeat_count")
    if repeat_count < 2:
        raise ValueError("repeat_count must be at least 2 to estimate run-to-run variability")
    if isinstance(drift_scale, bool) or not isinstance(drift_scale, (int, float)) or not math.isfinite(drift_scale) or drift_scale < 0:
        raise ValueError("drift_scale must be a finite nonnegative number")

    steady_ms = sum(_ms(value) for value in (
        scenario.preprocess, scenario.kernel, scenario.transfer, scenario.postprocess
    ))
    total_iterations = warmup_iterations + measured_iterations
    run_summaries: list[dict[str, Any]] = []
    traces: list[tuple[Any, ...]] = []
    for repeat_index in range(repeat_count):
        rng = random.Random(scenario.seed + seed_offset + repeat_index * 101)
        run_offset = rng.gauss(0.0, scenario.run_jitter_fraction)
        trace_ms: list[float] = []
        for iteration in range(total_iterations):
            warmup = scenario.warmup_fraction * math.exp(-iteration / scenario.warmup_decay_iterations)
            progress = iteration / max(1, total_iterations - 1)
            drift = scenario.baseline_drift_fraction * drift_scale * progress
            jitter = rng.gauss(0.0, scenario.jitter_fraction)
            trace_ms.append(steady_ms * max(0.5, 1.0 + run_offset + warmup + drift + jitter))
        measured = trace_ms[warmup_iterations:]
        run_summaries.append(_sample_summary(measured))
        traces.append(tuple(Q_(value, "ms") for value in trace_ms))

    run_medians_ms = [summary["median"].to("ms").magnitude for summary in run_summaries]
    median_mean = float(np.mean(run_medians_ms))
    median_std = float(np.std(run_medians_ms, ddof=1))
    return {
        "track_id": track_id,
        "traces": tuple(traces),
        "runs": tuple(run_summaries),
        "median_of_run_medians": Q_(float(np.median(run_medians_ms)), "ms"),
        "run_to_run_std": Q_(median_std, "ms"),
        "run_to_run_cv_pct": 100.0 * median_std / median_mean if median_mean else 0.0,
        "interval": None,
        "interval_note": "No IID confidence interval: the disclosed trace contains warmup and within-run drift.",
        "assumption": SCENARIO_ASSUMPTION,
        "inputs": {
            "track_id": track_id, "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "repeat_count": repeat_count,
            "drift_scale": float(drift_scale),
            "seed_offset": seed_offset,
        },
    }


def _sustained_trace(
    scenario: BenchmarkScenario,
    *,
    candidate: bool,
    sample_count: int,
    duration_s: float,
    stress_scale: float,
    seed: int,
) -> list[float]:
    kernel_ms = _ms(scenario.kernel) / (scenario.candidate_kernel_speedup if candidate else 1.0)
    other_ms = sum(_ms(value) for value in (scenario.preprocess, scenario.transfer, scenario.postprocess))
    extra_ms = _ms(scenario.candidate_extra_overhead) if candidate else 0.0
    steady_ms = other_ms + kernel_ms + extra_ms
    noise = _paired_noise(sample_count, seed, scenario.jitter_fraction * stress_scale)
    values: list[float] = []
    for index, disturbance in enumerate(noise):
        elapsed_fraction = index / max(1, sample_count - 1)
        drift_fraction = scenario.baseline_drift_fraction * stress_scale * elapsed_fraction
        if candidate and elapsed_fraction >= scenario.candidate_thermal_onset:
            thermal_progress = (elapsed_fraction - scenario.candidate_thermal_onset) / (1.0 - scenario.candidate_thermal_onset)
            drift_fraction += scenario.candidate_drift_fraction * stress_scale * thermal_progress
        duration_exposure = min(1.0, duration_s / 600.0)
        values.append(steady_ms * disturbance * (1.0 + drift_fraction * duration_exposure))
    return values


def compare_sustained(
    track_id: str,
    *,
    duration: Any = Q_(600, "s"),
    sample_count: int = 240,
    stress_scale: float = 1.0,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Compare paired baseline/candidate traces under one sustained condition."""

    scenario = _scenario(track_id)
    sample_count = _positive_int(sample_count, "sample_count")
    stress_scale = _finite_positive(stress_scale, "stress_scale")
    duration_s = _time_quantity(duration, "duration").magnitude
    duration_s = _finite_positive(duration_s, "duration")
    seed = scenario.seed + seed_offset + 50_000
    baseline_ms = _sustained_trace(
        scenario, candidate=False, sample_count=sample_count, duration_s=duration_s,
        stress_scale=stress_scale, seed=seed,
    )
    candidate_ms = _sustained_trace(
        scenario, candidate=True, sample_count=sample_count, duration_s=duration_s,
        stress_scale=stress_scale, seed=seed,
    )

    deadline_ms = _ms(scenario.deadline)
    baseline_energy = [scenario.baseline_power * Q_(value, "ms") for value in baseline_ms]
    candidate_energy = [scenario.candidate_power * Q_(value, "ms") for value in candidate_ms]
    baseline_summary = _sample_summary(baseline_ms)
    candidate_summary = _sample_summary(candidate_ms)
    baseline_misses = sum(value > deadline_ms for value in baseline_ms)
    candidate_misses = sum(value > deadline_ms for value in candidate_ms)
    baseline_energy_median = Q_(float(np.median([value.to("J").magnitude for value in baseline_energy])), "J")
    candidate_energy_median = Q_(float(np.median([value.to("J").magnitude for value in candidate_energy])), "J")
    return {
        "track_id": track_id,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "deadline": scenario.deadline,
        "baseline_deadline_misses": baseline_misses,
        "candidate_deadline_misses": candidate_misses,
        "baseline_deadline_miss_pct": 100.0 * baseline_misses / sample_count,
        "candidate_deadline_miss_pct": 100.0 * candidate_misses / sample_count,
        "baseline_energy_per_request": baseline_energy_median,
        "candidate_energy_per_request": candidate_energy_median,
        "median_speedup": baseline_summary["median"] / candidate_summary["median"],
        "candidate_uses_more_energy": candidate_energy_median > baseline_energy_median,
        "paired_delta": tuple(Q_(before - after, "ms") for before, after in zip(baseline_ms, candidate_ms)),
        "assumption": SCENARIO_ASSUMPTION,
        "inputs": {
            "track_id": track_id,
            "duration": {"magnitude": duration_s, "units": "second"},
            "sample_count": sample_count,
            "stress_scale": stress_scale, "seed_offset": seed_offset,
        },
    }


def _quality_fixture(
    scenario: BenchmarkScenario,
    examples_per_slice: int,
    seed_offset: int,
    candidate_mode: str,
) -> dict[str, Any]:
    rng = random.Random(scenario.seed + seed_offset + 90_000)
    records: list[dict[str, Any]] = []
    for slice_name, difficulty in (("ordinary", 0.0), ("difficult", 0.18)):
        for example_id in range(examples_per_slice):
            label = example_id % 2
            signed_label = 1.0 if label else -1.0
            margin = 0.25 - difficulty + rng.uniform(-0.16, 0.16)
            baseline_score = min(0.99, max(0.01, 0.5 + signed_label * margin))
            if candidate_mode in {"baseline", "quality_preserving"}:
                candidate_shift = 0.0
            else:
                candidate_shift = (0.035 if slice_name == "ordinary" else -0.12) * signed_label
            candidate_score = min(0.99, max(0.01, baseline_score + candidate_shift))
            records.append({
                "slice": slice_name, "example_id": example_id, "label": label,
                "baseline_score": baseline_score, "candidate_score": candidate_score,
            })
    return {"records": records, "assumption": SCENARIO_ASSUMPTION}


def compare_quality_slices(
    track_id: str,
    *,
    threshold: float = 0.5,
    examples_per_slice: int = 40,
    quality_floor_pct: float | None = None,
    worst_slice_floor_pct: float | None = None,
    candidate_mode: str = "aggressive",
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Evaluate paired fixed labels/scores and report aggregate and slice gates."""

    scenario = _scenario(track_id)
    examples_per_slice = _positive_int(examples_per_slice, "examples_per_slice")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if candidate_mode not in {"baseline", "quality_preserving", "aggressive"}:
        raise ValueError("candidate_mode must be baseline, quality_preserving, or aggressive")
    floor = scenario.quality_floor_pct if quality_floor_pct is None else float(quality_floor_pct)
    if not math.isfinite(floor) or not 0.0 <= floor <= 100.0:
        raise ValueError("quality_floor_pct must be between 0 and 100")
    if worst_slice_floor_pct is not None:
        worst_slice_floor_pct = float(worst_slice_floor_pct)
        if not math.isfinite(worst_slice_floor_pct) or not 0.0 <= worst_slice_floor_pct <= 100.0:
            raise ValueError("worst_slice_floor_pct must be between 0 and 100")
    fixture = _quality_fixture(scenario, examples_per_slice, seed_offset, candidate_mode)

    slices: dict[str, dict[str, Any]] = {}
    for slice_name in ("ordinary", "difficult"):
        records = [row for row in fixture["records"] if row["slice"] == slice_name]
        baseline_correct = sum((row["baseline_score"] >= threshold) == bool(row["label"]) for row in records)
        candidate_correct = sum((row["candidate_score"] >= threshold) == bool(row["label"]) for row in records)
        slices[slice_name] = {
            "count": len(records),
            "baseline_accuracy_pct": 100.0 * baseline_correct / len(records),
            "candidate_accuracy_pct": 100.0 * candidate_correct / len(records),
        }
    total = 2 * examples_per_slice
    baseline_overall = sum(item["baseline_accuracy_pct"] * item["count"] for item in slices.values()) / total
    candidate_overall = sum(item["candidate_accuracy_pct"] * item["count"] for item in slices.values()) / total
    baseline_worst_slice = min(item["baseline_accuracy_pct"] for item in slices.values())
    candidate_worst_slice = min(item["candidate_accuracy_pct"] for item in slices.values())
    return {
        "track_id": track_id,
        "system_role": "reference baseline" if candidate_mode == "baseline" else "efficiency candidate",
        "slices": slices,
        "baseline_overall_accuracy_pct": baseline_overall,
        "candidate_overall_accuracy_pct": candidate_overall,
        "quality_floor_pct": floor,
        "baseline_aggregate_eligible": baseline_overall >= floor,
        "candidate_aggregate_eligible": candidate_overall >= floor,
        "observed_overall_noninferior": candidate_overall >= baseline_overall,
        "baseline_worst_slice_accuracy_pct": baseline_worst_slice,
        "candidate_worst_slice_accuracy_pct": candidate_worst_slice,
        "worst_slice_floor_pct": worst_slice_floor_pct,
        "baseline_worst_slice_eligible": (
            None if worst_slice_floor_pct is None else baseline_worst_slice >= worst_slice_floor_pct
        ),
        "candidate_worst_slice_eligible": (
            None if worst_slice_floor_pct is None else candidate_worst_slice >= worst_slice_floor_pct
        ),
        "records": tuple(fixture["records"]),
        "assumption": fixture["assumption"],
        "inputs": {
            "track_id": track_id, "threshold": float(threshold),
            "examples_per_slice": examples_per_slice, "candidate_mode": candidate_mode,
            "quality_floor_pct": floor, "worst_slice_floor_pct": worst_slice_floor_pct,
            "seed_offset": seed_offset,
        },
    }


@dataclass(frozen=True)
class ProtocolSpec:
    """Fields that define whether two benchmark results answer the same claim."""

    workload_id: str
    scope: str
    warmup_iterations: int
    measured_iterations: int
    repeat_count: int
    trace_id: str
    quality_fixture_id: str
    quality_floor_pct: float
    power_boundary: str


_PROTOCOL_FIELDS = (
    "workload_id", "scope", "warmup_iterations", "measured_iterations",
    "repeat_count", "trace_id", "quality_fixture_id", "quality_floor_pct",
    "power_boundary",
)


def default_protocol(track_id: str) -> ProtocolSpec:
    """Return a complete, disclosed protocol for one track fixture."""

    scenario = _scenario(track_id)
    return ProtocolSpec(
        workload_id=scenario.workload_id,
        scope="whole request path",
        warmup_iterations=8,
        measured_iterations=40,
        repeat_count=5,
        trace_id=f"{track_id}-paired-trace-v1",
        quality_fixture_id=f"{track_id}-quality-fixture-v1",
        quality_floor_pct=scenario.quality_floor_pct,
        power_boundary="whole system during request",
    )


def audit_protocols(reference: ProtocolSpec, candidate: ProtocolSpec) -> dict[str, Any]:
    """Identify concrete mismatches; never collapse protocol validity to a score."""

    if not isinstance(reference, ProtocolSpec) or not isinstance(candidate, ProtocolSpec):
        raise TypeError("reference and candidate must be ProtocolSpec instances")
    mismatches = tuple(
        {
            "field": field,
            "reference": getattr(reference, field),
            "candidate": getattr(candidate, field),
        }
        for field in _PROTOCOL_FIELDS
        if getattr(reference, field) != getattr(candidate, field)
    )
    comparable = not mismatches
    return {
        "comparable": comparable,
        "include_in_headline_comparison": comparable,
        "claim_status": "comparable" if comparable else "exclude until protocols are aligned",
        "mismatches": mismatches,
        "repair": replace(candidate, **{field: getattr(reference, field) for field in _PROTOCOL_FIELDS}),
        "inputs": {"reference": asdict(reference), "candidate": asdict(candidate)},
    }


def replay(model_key: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Replay one saved experiment arm from its JSON-native input mapping."""

    if not isinstance(inputs, dict):
        raise TypeError("inputs must be a dictionary")
    short_key = model_key.removeprefix("v1_12_experiments.")
    if short_key == "compare_scopes":
        return compare_scopes(**inputs)
    if short_key == "analyze_repeats":
        return analyze_repeats(**inputs)
    if short_key == "compare_sustained":
        return compare_sustained(**inputs)
    if short_key == "compare_quality_slices":
        return compare_quality_slices(**inputs)
    if short_key == "audit_protocols":
        try:
            reference = ProtocolSpec(**inputs["reference"])
            candidate = ProtocolSpec(**inputs["candidate"])
        except (KeyError, TypeError) as exc:
            raise ValueError("audit_protocols replay requires reference and candidate protocol inputs") from exc
        return audit_protocols(reference, candidate)
    raise ValueError(
        "model_key must name compare_scopes, analyze_repeats, compare_sustained, "
        "compare_quality_slices, or audit_protocols"
    )


__all__ = [
    "BenchmarkScenario", "ProtocolSpec", "SCENARIO_ASSUMPTION", "TRACKS",
    "analyze_repeats", "audit_protocols", "compare_quality_slices",
    "compare_scopes", "compare_sustained", "default_protocol",
    "replay", "to_jsonable",
]
