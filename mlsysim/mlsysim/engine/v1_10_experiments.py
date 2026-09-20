"""Chapter 10 model-compression experiments.

This module keeps the chapter lab's quantitative work in MLSysIM.  The
quality observations are small, explicitly illustrative fixtures.  They are
not inferred from bit width, sparsity, artifact size, or hardware speed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
from math import ceil, sqrt
from typing import Any, Mapping, Sequence

from ..core.types import Quantity
from ..core.units import Q_, ureg
from ..hardware.registry import Hardware
from ..hardware.types import HardwareNode
from ..models.registry import Models
from ..models.types import Workload
from .solvers.compression import CompressionModel


TRACK_IDS = ("iphone", "oura_ring", "robotaxi", "cloud_fleet")
CANONICAL_TRACK_IDS = ("mobile", "tinyml", "edge", "cloud")
MODEL_ID = "v1_10_experiments"
MODEL_KEYS = {
    "resource": "v1_10.resource.v1",
    "precision": "v1_10.precision.v1",
    "sparsity": "v1_10.sparsity.v1",
    "distillation": "v1_10.distillation.v1",
    "recipe": "v1_10.recipe.v1",
}
SPARSITY_CASES = {
    "dense_mask_50": ("dense_mask", 0.50),
    "indexed_25": ("indexed", 0.25),
    "indexed_50": ("indexed", 0.50),
    "structured_25": ("structured", 0.25),
    "structured_50": ("structured", 0.50),
    "n_m_50": ("n_m", 0.50),
}
RECIPE_SEQUENCES = {
    "distill_prune_quant": ("distill_small", "prune_structured_25", "quant_int8_ptq"),
    "prune_distill_quant": ("prune_structured_25", "distill_small", "quant_int8_ptq"),
    "prune_quant": ("prune_structured_25", "quant_int8_ptq"),
    "quant_prune": ("quant_int8_ptq", "prune_structured_25"),
    "quant_distill": ("quant_int8_ptq", "distill_small"),
}
_TRACK_ALIASES = {
    "mobile": "iphone",
    "tinyml": "oura_ring",
    "edge": "robotaxi",
    "cloud": "cloud_fleet",
}


@dataclass(frozen=True)
class PrecisionEvidence:
    storage_bits: int
    calibration: str
    task_quality: float
    preparation_time: Quantity
    evidence_label: str = "illustrative supplied task outcome"


@dataclass(frozen=True)
class SparseEvidence:
    representation: str
    sparsity: float
    task_quality: float
    evidence_label: str = "illustrative supplied task outcome"


@dataclass(frozen=True)
class DistillationEvidence:
    candidate_id: str
    parameter_fraction: float
    operation_fraction: float
    state_fraction: float
    task_quality: float
    training_time: Quantity
    evidence_label: str = "illustrative supplied task outcome"


@dataclass(frozen=True)
class CompressionScenario:
    track_id: str
    model: Workload
    hardware: HardwareNode
    quality_metric: str
    baseline_quality: float
    maximum_quality_drop: float
    package_budget: Quantity
    working_set_budget: Quantity
    latency_budget: Quantity
    effective_load_bandwidth: Quantity
    baseline_model_time: Quantity
    fixed_application_time: Quantity
    runtime_state_values: int
    minimum_runtime_state: Quantity
    application_memory: Quantity
    quant_group_size: int
    quant_scale_bits: int
    quant_zero_point_bits: int
    supported_compute_speedups: Mapping[int, float]
    sparse_execution_speedups: Mapping[str, float]
    precision_evidence: tuple[PrecisionEvidence, ...]
    sparse_evidence: tuple[SparseEvidence, ...]
    distillation_evidence: tuple[DistillationEvidence, ...]
    recipe_quality: Mapping[tuple[str, ...], float]
    lifetime_inferences: Mapping[str, int]
    default_inferences_key: str

    @property
    def default_inferences(self) -> int:
        return self.lifetime_inferences[self.default_inferences_key]

    @property
    def quality_floor(self) -> float:
        return self.baseline_quality - self.maximum_quality_drop

    @property
    def ledger_track_id(self) -> str:
        return {
            "iphone": "mobile",
            "oura_ring": "tinyml",
            "robotaxi": "edge",
            "cloud_fleet": "cloud",
        }[self.track_id]


@dataclass(frozen=True)
class CompressionResult:
    track_id: str
    artifact_size: Quantity
    runtime_state_size: Quantity
    working_set_size: Quantity
    weight_load_time: Quantity
    model_execution_time: Quantity
    end_to_end_latency: Quantity
    task_quality: float | None
    quality_evidence: str | None
    preparation_time: Quantity
    inputs: Mapping[str, object]
    execution_supported: bool
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class DistillationResult:
    deployment: CompressionResult
    training_time: Quantity
    deployment_inferences: int
    lifecycle_time: Quantity
    amortized_time_per_inference: Quantity
    latency_saved_per_inference: Quantity
    break_even_inferences: int | None
    training_amortized: bool


@dataclass(frozen=True)
class RecipeStage:
    transformation: str
    logical_parameters: int
    operation_fraction: float
    storage_bits: int
    sparsity: float
    representation: str
    artifact_size: Quantity


@dataclass(frozen=True)
class RecipeResult:
    deployment: CompressionResult
    stages: tuple[RecipeStage, ...]
    outcome_available: bool
    naive_speedup_product: float
    naive_end_to_end_latency: Quantity


@dataclass(frozen=True)
class ExperimentSnapshot:
    """Versioned JSON-only evidence arm with exact replay inputs and result."""

    model_key: str
    track_id: str
    inputs: dict[str, Any]
    result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        # A JSON round trip returns a detached dictionary and enforces the public contract.
        return json.loads(json.dumps({
            "model_key": self.model_key,
            "track_id": self.track_id,
            "inputs": self.inputs,
            "result": self.result,
        }, allow_nan=False))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentSnapshot":
        return restore_snapshot(value)


def _precision_rows(
    baseline: float,
    *,
    int8_ptq_drop: float,
    int8_qat_drop: float,
    int6_ptq_drop: float,
    int6_qat_drop: float,
    int4_ptq_drop: float,
    int4_qat_drop: float,
) -> tuple[PrecisionEvidence, ...]:
    return (
        PrecisionEvidence(16, "none", baseline, Q_(0, "minute")),
        PrecisionEvidence(8, "none", baseline - int8_ptq_drop * 2.2, Q_(0, "minute")),
        PrecisionEvidence(8, "ptq", baseline - int8_ptq_drop, Q_(12, "minute")),
        PrecisionEvidence(8, "qat", baseline - int8_qat_drop, Q_(3.5, "hour")),
        PrecisionEvidence(6, "ptq", baseline - int6_ptq_drop, Q_(18, "minute")),
        PrecisionEvidence(6, "qat", baseline - int6_qat_drop, Q_(5, "hour")),
        PrecisionEvidence(4, "ptq", baseline - int4_ptq_drop, Q_(25, "minute")),
        PrecisionEvidence(4, "qat", baseline - int4_qat_drop, Q_(8, "hour")),
    )


def _sparse_rows(baseline: float, strictness: float = 1.0) -> tuple[SparseEvidence, ...]:
    return (
        SparseEvidence("dense_mask", 0.50, baseline - 0.003 * strictness),
        SparseEvidence("indexed", 0.25, baseline - 0.002 * strictness),
        SparseEvidence("indexed", 0.50, baseline - 0.006 * strictness),
        SparseEvidence("structured", 0.25, baseline - 0.003 * strictness),
        SparseEvidence("structured", 0.50, baseline - 0.012 * strictness),
        SparseEvidence("n_m", 0.50, baseline - 0.005 * strictness),
    )


def _distill_rows(
    baseline: float,
    *,
    strictness: float = 1.0,
    small_training_time: Quantity = Q_(14, "hour"),
    tiny_training_time: Quantity = Q_(22, "hour"),
) -> tuple[DistillationEvidence, ...]:
    return (
        DistillationEvidence(
            "small_dense", 0.58, 0.62, 0.72,
            baseline - 0.004 * strictness, small_training_time,
        ),
        DistillationEvidence(
            "tiny_dense", 0.28, 0.34, 0.48,
            baseline - 0.024 * strictness, tiny_training_time,
        ),
    )


def _recipe_rows(baseline: float, strictness: float = 1.0) -> Mapping[tuple[str, ...], float]:
    # Exact-sequence fixtures only.  No interpolation or nearest-sequence lookup.
    return {
        ("distill_small", "prune_structured_25", "quant_int8_ptq"):
            baseline - 0.007 * strictness,
        ("prune_structured_25", "distill_small", "quant_int8_ptq"):
            baseline - 0.010 * strictness,
        ("prune_structured_25", "quant_int8_ptq"):
            baseline - 0.006 * strictness,
        ("quant_int8_ptq", "prune_structured_25"):
            baseline - 0.008 * strictness,
    }


def _scenario(
    track_id: str,
    model: Workload,
    hardware: HardwareNode,
    *,
    quality_metric: str,
    baseline_quality: float,
    maximum_quality_drop: float,
    package_budget: str,
    working_set_budget: str,
    latency_budget: str,
    effective_load_bandwidth: str,
    baseline_model_time: str,
    fixed_application_time: str,
    runtime_state_values: int,
    minimum_runtime_state: str,
    application_memory: str,
    supported_compute_speedups: Mapping[int, float],
    sparse_execution_speedups: Mapping[str, float],
    precision_drops: tuple[float, float, float, float, float, float],
    strictness: float = 1.0,
    small_training_time: str = "14 hour",
    tiny_training_time: str = "22 hour",
    lifetime_inferences: Mapping[str, int] = {
        "100 thousand": 100_000,
        "10 million": 10_000_000,
        "100 million": 100_000_000,
    },
    default_inferences_key: str = "10 million",
) -> CompressionScenario:
    return CompressionScenario(
        track_id=track_id,
        model=model,
        hardware=hardware,
        quality_metric=quality_metric,
        baseline_quality=baseline_quality,
        maximum_quality_drop=maximum_quality_drop,
        package_budget=Q_(package_budget),
        working_set_budget=Q_(working_set_budget),
        latency_budget=Q_(latency_budget),
        effective_load_bandwidth=Q_(effective_load_bandwidth),
        baseline_model_time=Q_(baseline_model_time),
        fixed_application_time=Q_(fixed_application_time),
        runtime_state_values=runtime_state_values,
        minimum_runtime_state=Q_(minimum_runtime_state),
        application_memory=Q_(application_memory),
        quant_group_size=32,
        quant_scale_bits=16,
        quant_zero_point_bits=8,
        supported_compute_speedups=dict(supported_compute_speedups),
        sparse_execution_speedups=dict(sparse_execution_speedups),
        precision_evidence=_precision_rows(
            baseline_quality,
            int8_ptq_drop=precision_drops[0],
            int8_qat_drop=precision_drops[1],
            int6_ptq_drop=precision_drops[2],
            int6_qat_drop=precision_drops[3],
            int4_ptq_drop=precision_drops[4],
            int4_qat_drop=precision_drops[5],
        ),
        sparse_evidence=_sparse_rows(baseline_quality, strictness),
        distillation_evidence=_distill_rows(
            baseline_quality,
            strictness=strictness,
            small_training_time=Q_(small_training_time),
            tiny_training_time=Q_(tiny_training_time),
        ),
        recipe_quality=_recipe_rows(baseline_quality, strictness),
        lifetime_inferences=dict(lifetime_inferences),
        default_inferences_key=default_inferences_key,
    )


_SCENARIOS = {
    "iphone": _scenario(
        "iphone", Models.Vision.MobileNetV2, Hardware.Mobile.iPhone15Pro,
        quality_metric="top-1 accuracy", baseline_quality=0.902,
        maximum_quality_drop=0.010, package_budget="8 MiB",
        working_set_budget="28 MiB", latency_budget="35 ms",
        effective_load_bandwidth="2.4 GB/s", baseline_model_time="24 ms",
        fixed_application_time="7 ms", runtime_state_values=5_500_000,
        minimum_runtime_state="8 MiB", application_memory="6 MiB",
        supported_compute_speedups={16: 1.0, 8: 1.65},
        sparse_execution_speedups={"structured": 1.35},
        precision_drops=(0.004, 0.001, 0.013, 0.004, 0.041, 0.010),
        small_training_time="4 hour", tiny_training_time="7 hour",
        lifetime_inferences={"100 thousand": 100_000, "1 million": 1_000_000, "10 million": 10_000_000},
        default_inferences_key="1 million",
    ),
    "oura_ring": _scenario(
        "oura_ring", Models.Tiny.DS_CNN, Hardware.Tiny.OuraRing,
        quality_metric="keyword accuracy", baseline_quality=0.925,
        maximum_quality_drop=0.020, package_budget="260 KiB",
        working_set_budget="500 KiB", latency_budget="120 ms",
        effective_load_bandwidth="12 MB/s", baseline_model_time="82 ms",
        fixed_application_time="18 ms", runtime_state_values=90_000,
        minimum_runtime_state="128 KiB", application_memory="96 KiB",
        supported_compute_speedups={16: 1.0, 8: 1.35},
        sparse_execution_speedups={"structured": 1.18},
        precision_drops=(0.006, 0.002, 0.017, 0.007, 0.034, 0.014),
        small_training_time="20 minute", tiny_training_time="35 minute",
        lifetime_inferences={"10 thousand": 10_000, "100 thousand": 100_000, "1 million": 1_000_000},
        default_inferences_key="100 thousand",
    ),
    "robotaxi": _scenario(
        "robotaxi", Models.Vision.YOLOv8_Nano, Hardware.Edge.RoboTaxi,
        quality_metric="rare-hazard recall", baseline_quality=0.947,
        maximum_quality_drop=0.005, package_budget="12 MiB",
        working_set_budget="62 MiB", latency_budget="28 ms",
        effective_load_bandwidth="5.0 GB/s", baseline_model_time="21 ms",
        fixed_application_time="5 ms", runtime_state_values=18_000_000,
        minimum_runtime_state="24 MiB", application_memory="12 MiB",
        supported_compute_speedups={16: 1.0, 8: 1.8},
        sparse_execution_speedups={"structured": 1.42, "n_m": 1.55},
        precision_drops=(0.008, 0.003, 0.016, 0.006, 0.032, 0.012),
        strictness=0.6,
        small_training_time="6 hour", tiny_training_time="10 hour",
        lifetime_inferences={"1 million": 1_000_000, "10 million": 10_000_000, "100 million": 100_000_000},
        default_inferences_key="10 million",
    ),
    "cloud_fleet": _scenario(
        "cloud_fleet", Models.Language.BERT_Base, Hardware.Cloud.H100,
        quality_metric="matched-task accuracy", baseline_quality=0.918,
        maximum_quality_drop=0.010, package_budget="240 MiB",
        working_set_budget="430 MiB", latency_budget="24 ms",
        effective_load_bandwidth="45 GB/s", baseline_model_time="17 ms",
        fixed_application_time="4 ms", runtime_state_values=42_000_000,
        minimum_runtime_state="96 MiB", application_memory="48 MiB",
        supported_compute_speedups={16: 1.0, 8: 1.9},
        sparse_execution_speedups={"structured": 1.45, "n_m": 1.75, "indexed": 1.15},
        precision_drops=(0.004, 0.001, 0.012, 0.004, 0.028, 0.009),
        small_training_time="14 hour", tiny_training_time="22 hour",
        lifetime_inferences={"10 million": 10_000_000, "100 million": 100_000_000, "1 billion": 1_000_000_000},
        default_inferences_key="100 million",
    ),
}


def get_scenario(track_id: str) -> CompressionScenario:
    """Return a canonical chapter scenario."""
    track_id = _TRACK_ALIASES.get(track_id, track_id)
    try:
        return _SCENARIOS[track_id]
    except KeyError as exc:
        raise KeyError(f"Unknown track {track_id!r}; expected one of {TRACK_IDS}") from exc


def _parameters(scenario: CompressionScenario) -> int:
    parameters = scenario.model.parameters
    if parameters is None:
        raise ValueError(f"Model {scenario.model.name!r} has no parameter count")
    return int(round(parameters.to("param").magnitude))


def _dense_artifact(parameters: int, storage_bits: int) -> Quantity:
    return Q_(parameters * storage_bits / 8, "byte")


def _quantized_artifact(
    scenario: CompressionScenario, parameters: int, storage_bits: int
) -> Quantity:
    groups = ceil(parameters / scenario.quant_group_size)
    metadata_bits = groups * (scenario.quant_scale_bits + scenario.quant_zero_point_bits)
    return Q_((parameters * storage_bits + metadata_bits) / 8, "byte")


def _runtime_state(
    scenario: CompressionScenario, state_bits: int, state_fraction: float = 1.0
) -> Quantity:
    if state_bits not in {8, 16, 32}:
        raise ValueError("runtime state precision must be 8, 16, or 32 bits")
    calculated = Q_(scenario.runtime_state_values * state_fraction * state_bits / 8, "byte")
    return max(calculated, scenario.minimum_runtime_state)


def _deployment_result(
    scenario: CompressionScenario,
    *,
    artifact_size: Quantity,
    state_bits: int,
    state_fraction: float,
    operation_fraction: float,
    compute_bits: int,
    execution_multiplier: float,
    task_quality: float | None,
    quality_evidence: str | None,
    preparation_time: Quantity = Q_(0, "second"),
    inputs: Mapping[str, object] | None = None,
    require_outcome: bool = True,
) -> CompressionResult:
    runtime_state = _runtime_state(scenario, state_bits, state_fraction)
    working_set = artifact_size + runtime_state + scenario.application_memory
    weight_load_time = (artifact_size / scenario.effective_load_bandwidth).to("ms")
    compute_speedup = scenario.supported_compute_speedups.get(compute_bits)
    execution_supported = compute_speedup is not None and execution_multiplier > 0
    applied_compute_speedup = compute_speedup if compute_speedup is not None else 1.0
    model_time = (
        scenario.baseline_model_time
        * operation_fraction
        / (applied_compute_speedup * max(execution_multiplier, 1.0))
    ).to("ms")
    latency = (scenario.fixed_application_time + weight_load_time + model_time).to("ms")

    violations: list[str] = []
    if artifact_size > scenario.package_budget:
        violations.append("package")
    if working_set > scenario.working_set_budget:
        violations.append("working_set")
    if latency > scenario.latency_budget:
        violations.append("latency")
    if not execution_supported:
        violations.append("execution_support")
    if require_outcome and task_quality is None:
        violations.append("quality_outcome_unavailable")
    elif task_quality is not None and task_quality < scenario.quality_floor:
        violations.append("quality")

    return CompressionResult(
        track_id=scenario.track_id,
        artifact_size=artifact_size.to("byte"),
        runtime_state_size=runtime_state.to("byte"),
        working_set_size=working_set.to("byte"),
        weight_load_time=weight_load_time,
        model_execution_time=model_time,
        end_to_end_latency=latency,
        task_quality=task_quality,
        quality_evidence=quality_evidence,
        preparation_time=preparation_time.to("second"),
        inputs=dict(inputs or {}),
        execution_supported=execution_supported,
        feasible=not violations,
        violations=tuple(violations),
    )


def evaluate_resource(
    track_id: str,
    *,
    storage_bits: int = 16,
    compute_bits: int = 16,
    runtime_state_bits: int = 16,
    calibration: str = "none",
) -> CompressionResult:
    """Evaluate package, working-set, and whole-application consequences."""
    return evaluate_precision(
        track_id,
        storage_bits=storage_bits,
        compute_bits=compute_bits,
        runtime_state_bits=runtime_state_bits,
        calibration=calibration,
    )


def evaluate_precision(
    track_id: str,
    *,
    storage_bits: int,
    compute_bits: int,
    runtime_state_bits: int = 16,
    calibration: str,
) -> CompressionResult:
    """Evaluate a precision choice using an exact supplied quality outcome."""
    scenario = get_scenario(track_id)
    if storage_bits < 1:
        raise ValueError("storage_bits must be positive")
    evidence = next(
        (
            row for row in scenario.precision_evidence
            if row.storage_bits == storage_bits and row.calibration == calibration
        ),
        None,
    )
    artifact = (
        _dense_artifact(_parameters(scenario), storage_bits)
        if storage_bits >= 16
        else _quantized_artifact(scenario, _parameters(scenario), storage_bits)
    )
    return _deployment_result(
        scenario,
        artifact_size=artifact,
        state_bits=runtime_state_bits,
        state_fraction=1.0,
        operation_fraction=1.0,
        compute_bits=compute_bits,
        execution_multiplier=1.0,
        task_quality=evidence.task_quality if evidence else None,
        quality_evidence=evidence.evidence_label if evidence else None,
        preparation_time=evidence.preparation_time if evidence else Q_(0, "second"),
        inputs={
            "track_id": scenario.ledger_track_id,
            "storage_bits": storage_bits,
            "compute_bits": compute_bits,
            "runtime_state_bits": runtime_state_bits,
            "calibration": calibration,
        },
    )


def _sparse_artifact(
    scenario: CompressionScenario,
    *,
    sparsity: float,
    representation: str,
    storage_bits: int,
    index_bits: int,
) -> Quantity:
    parameters = _parameters(scenario)
    remaining = ceil(parameters * (1.0 - sparsity))
    value_bits = remaining * storage_bits
    if representation == "dense_mask":
        return Q_((parameters * storage_bits + parameters) / 8, "byte")
    if representation == "indexed":
        rows = ceil(sqrt(parameters))
        metadata_bits = remaining * index_bits + (rows + 1) * index_bits
        return Q_((value_bits + metadata_bits) / 8, "byte")
    if representation in {"structured", "n_m"}:
        layout_bits = ceil(parameters / 32) * 16
        return Q_((value_bits + layout_bits) / 8, "byte")
    raise ValueError("representation must be dense_mask, indexed, structured, or n_m")


def evaluate_sparsity(
    track_id: str,
    *,
    sparsity: float,
    representation: str,
    storage_bits: int = 16,
    index_bits: int = 16,
    runtime_state_bits: int = 16,
) -> CompressionResult:
    """Evaluate sparse storage metadata and target execution support."""
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    if index_bits not in {8, 16, 32}:
        raise ValueError("index_bits must be 8, 16, or 32")
    scenario = get_scenario(track_id)
    evidence = next(
        (
            row for row in scenario.sparse_evidence
            if row.representation == representation and abs(row.sparsity - sparsity) < 1e-12
        ),
        None,
    )
    execution_multiplier = scenario.sparse_execution_speedups.get(representation, 0.0)
    if representation == "dense_mask":
        operation_fraction = 1.0
    elif execution_multiplier > 0:
        operation_fraction = 1.0 - sparsity
    else:
        # Unsupported sparse execution falls back to dense work, even if storage is smaller.
        operation_fraction = 1.0
    artifact = _sparse_artifact(
        scenario,
        sparsity=sparsity,
        representation=representation,
        storage_bits=storage_bits,
        index_bits=index_bits,
    )
    return _deployment_result(
        scenario,
        artifact_size=artifact,
        state_bits=runtime_state_bits,
        state_fraction=1.0,
        operation_fraction=operation_fraction,
        compute_bits=16,
        execution_multiplier=execution_multiplier,
        task_quality=evidence.task_quality if evidence else None,
        quality_evidence=evidence.evidence_label if evidence else None,
        inputs={
            "track_id": scenario.ledger_track_id,
            "sparsity": sparsity,
            "representation": representation,
            "storage_bits": storage_bits,
            "index_bits": index_bits,
            "runtime_state_bits": runtime_state_bits,
        },
    )


def evaluate_sparsity_case(
    track_id: str,
    case_id: str,
    *,
    storage_bits: int = 16,
    index_bits: int = 16,
    runtime_state_bits: int = 16,
) -> CompressionResult:
    """Evaluate one named sparse representation without notebook-side decoding."""
    try:
        representation, sparsity = SPARSITY_CASES[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown sparsity case {case_id!r}") from exc
    return evaluate_sparsity(
        track_id,
        sparsity=sparsity,
        representation=representation,
        storage_bits=storage_bits,
        index_bits=index_bits,
        runtime_state_bits=runtime_state_bits,
    )


def evaluate_distillation(
    track_id: str,
    *,
    candidate_id: str,
    deployment_inferences: int = 1_000_000,
    runtime_state_bits: int = 16,
) -> DistillationResult:
    """Evaluate one supplied dense-student outcome and its one-time training cost."""
    if deployment_inferences < 1:
        raise ValueError("deployment_inferences must be positive")
    scenario = get_scenario(track_id)
    evidence = next(
        (row for row in scenario.distillation_evidence if row.candidate_id == candidate_id),
        None,
    )
    if evidence is None:
        raise KeyError(f"No supplied distillation outcome for {candidate_id!r}")
    artifact = _dense_artifact(
        ceil(_parameters(scenario) * evidence.parameter_fraction), 16
    )
    deployment = _deployment_result(
        scenario,
        artifact_size=artifact,
        state_bits=runtime_state_bits,
        state_fraction=evidence.state_fraction,
        operation_fraction=evidence.operation_fraction,
        compute_bits=16,
        execution_multiplier=1.0,
        task_quality=evidence.task_quality,
        quality_evidence=evidence.evidence_label,
        preparation_time=evidence.training_time,
        inputs={
            "track_id": scenario.ledger_track_id,
            "candidate_id": candidate_id,
            "deployment_inferences": deployment_inferences,
            "runtime_state_bits": runtime_state_bits,
        },
    )
    baseline = evaluate_precision(
        track_id, storage_bits=16, compute_bits=16,
        runtime_state_bits=runtime_state_bits, calibration="none",
    )
    saved = (baseline.end_to_end_latency - deployment.end_to_end_latency).to("ms")
    break_even = None
    if saved.magnitude > 0:
        break_even = ceil(evidence.training_time.to("ms").magnitude / saved.magnitude)
    lifecycle_time = (
        evidence.training_time
        + deployment.end_to_end_latency * deployment_inferences
    ).to("hour")
    amortized = (lifecycle_time / deployment_inferences).to("ms")
    return DistillationResult(
        deployment=deployment,
        training_time=evidence.training_time,
        deployment_inferences=deployment_inferences,
        lifecycle_time=lifecycle_time,
        amortized_time_per_inference=amortized,
        latency_saved_per_inference=saved,
        break_even_inferences=break_even,
        training_amortized=(break_even is not None and deployment_inferences >= break_even),
    )


_TRANSFORMATIONS = {
    "distill_small",
    "prune_structured_25",
    "quant_int8_ptq",
}


def evaluate_recipe(
    track_id: str,
    transformations: Sequence[str],
    *,
    runtime_state_bits: int = 16,
) -> RecipeResult:
    """Compose transformations in order and use only exact-sequence quality evidence."""
    sequence = tuple(transformations)
    unknown = [name for name in sequence if name not in _TRANSFORMATIONS]
    if unknown:
        raise ValueError(f"Unknown transformations: {', '.join(unknown)}")
    scenario = get_scenario(track_id)
    parameters = _parameters(scenario)
    operations = 1.0
    state_fraction = 1.0
    storage_bits = 16
    sparsity = 0.0
    representation = "dense"
    stages: list[RecipeStage] = []

    for transformation in sequence:
        if transformation == "distill_small":
            parameters = ceil(parameters * 0.58)
            operations *= 0.62
            state_fraction = max(0.72 * state_fraction, 0.35)
            storage_bits = 16
            sparsity = 0.0
            representation = "dense"
        elif transformation == "prune_structured_25":
            parameters = ceil(parameters * 0.75)
            operations *= 0.75
            sparsity = 0.25
            representation = "structured"
        elif transformation == "quant_int8_ptq":
            storage_bits = 8

        if representation == "structured":
            # Parameters already represent the remaining structured tensor.
            layout_bits = ceil(parameters / 32) * 16
            artifact = Q_((parameters * storage_bits + layout_bits) / 8, "byte")
        elif storage_bits < 16:
            artifact = _quantized_artifact(scenario, parameters, storage_bits)
        else:
            artifact = _dense_artifact(parameters, storage_bits)
        stages.append(
            RecipeStage(
                transformation=transformation,
                logical_parameters=parameters,
                operation_fraction=operations,
                storage_bits=storage_bits,
                sparsity=sparsity,
                representation=representation,
                artifact_size=artifact.to("byte"),
            )
        )

    if not stages:
        artifact = _dense_artifact(parameters, storage_bits)
    else:
        artifact = stages[-1].artifact_size
    outcome = scenario.recipe_quality.get(sequence)
    compute_bits = 8 if storage_bits == 8 else 16
    sparse_multiplier = (
        scenario.sparse_execution_speedups.get("structured", 0.0)
        if representation == "structured" else 1.0
    )
    deployment = _deployment_result(
        scenario,
        artifact_size=artifact,
        state_bits=runtime_state_bits,
        state_fraction=state_fraction,
        operation_fraction=operations,
        compute_bits=compute_bits,
        execution_multiplier=sparse_multiplier,
        task_quality=outcome,
        quality_evidence=("illustrative supplied exact-sequence outcome" if outcome is not None else None),
        inputs={
            "track_id": scenario.ledger_track_id,
            "transformations": sequence,
            "runtime_state_bits": runtime_state_bits,
        },
        require_outcome=True,
    )

    baseline = evaluate_precision(
        track_id, storage_bits=16, compute_bits=16,
        runtime_state_bits=runtime_state_bits, calibration="none",
    )
    standalone_speedups: list[float] = []
    for transformation in sequence:
        if transformation == "distill_small":
            standalone = evaluate_distillation(track_id, candidate_id="small_dense").deployment
        elif transformation == "prune_structured_25":
            standalone = evaluate_sparsity(
                track_id, sparsity=0.25, representation="structured"
            )
        else:
            standalone = evaluate_precision(
                track_id, storage_bits=8, compute_bits=8, calibration="ptq"
            )
        standalone_speedups.append(
            baseline.end_to_end_latency.to("ms").magnitude
            / standalone.end_to_end_latency.to("ms").magnitude
        )
    naive_product = 1.0
    for speedup in standalone_speedups:
        naive_product *= speedup
    naive_latency = (baseline.end_to_end_latency / naive_product).to("ms")
    return RecipeResult(
        deployment=deployment,
        stages=tuple(stages),
        outcome_available=outcome is not None,
        naive_speedup_product=naive_product,
        naive_end_to_end_latency=naive_latency,
    )


def evaluate_recipe_id(
    track_id: str,
    recipe_id: str,
    *,
    runtime_state_bits: int = 16,
) -> RecipeResult:
    """Evaluate one named ordered recipe from the MLSysIM scenario catalog."""
    try:
        transformations = RECIPE_SEQUENCES[recipe_id]
    except KeyError as exc:
        raise KeyError(f"Unknown recipe {recipe_id!r}") from exc
    return evaluate_recipe(
        track_id, transformations, runtime_state_bits=runtime_state_bits
    )


def to_jsonable(value: Any) -> Any:
    """Convert experiment values to plain JSON data without repr fallbacks."""
    if isinstance(value, ureg.Quantity):
        magnitude = value.magnitude
        if hasattr(magnitude, "tolist"):
            magnitude = magnitude.tolist()
        return {"magnitude": magnitude, "unit": str(value.units)}
    if is_dataclass(value):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize experiment value of type {type(value).__name__}")


def _result_inputs(
    result: CompressionResult | DistillationResult | RecipeResult,
) -> Mapping[str, object]:
    if isinstance(result, CompressionResult):
        return result.inputs
    return result.deployment.inputs


def snapshot_result(
    model_key: str,
    result: CompressionResult | DistillationResult | RecipeResult,
) -> ExperimentSnapshot:
    """Create a JSON-safe evidence arm from one already evaluated result."""
    if model_key not in MODEL_KEYS.values():
        raise ValueError(f"unsupported model_key for {MODEL_ID}: {model_key}")
    plain_inputs = to_jsonable(_result_inputs(result))
    if not isinstance(plain_inputs, dict):
        raise TypeError("experiment inputs must serialize to a dictionary")
    track_id = str(plain_inputs.get("track_id", ""))
    if track_id not in CANONICAL_TRACK_IDS:
        raise ValueError("experiment inputs require a canonical track_id")
    plain_result = to_jsonable(result)
    if not isinstance(plain_result, dict):
        raise TypeError("experiment result must serialize to a dictionary")
    # Every evidence arm exposes its exact arguments at the result's top level,
    # including composite result types whose deployment is nested.
    plain_result["inputs"] = plain_inputs
    return ExperimentSnapshot(model_key, track_id, plain_inputs, plain_result)


def replay(
    model_key: str, inputs: Mapping[str, Any]
) -> CompressionResult | DistillationResult | RecipeResult:
    """Replay one method/version from exact JSON-restored evaluation inputs."""
    if model_key not in MODEL_KEYS.values():
        raise ValueError(f"unsupported model_key for {MODEL_ID}: {model_key}")
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping")
    plain_inputs = json.loads(json.dumps(dict(inputs), allow_nan=False))
    if "track_id" not in plain_inputs:
        raise ValueError("inputs require track_id")
    track_id = str(plain_inputs.pop("track_id"))
    scenario = get_scenario(track_id)
    canonical_track = scenario.ledger_track_id
    if track_id != canonical_track:
        raise ValueError(
            f"snapshot track_id must be canonical {canonical_track!r}, got {track_id!r}"
        )
    if "transformations" in plain_inputs:
        plain_inputs["transformations"] = tuple(plain_inputs["transformations"])

    evaluator = {
        MODEL_KEYS["resource"]: evaluate_resource,
        MODEL_KEYS["precision"]: evaluate_precision,
        MODEL_KEYS["sparsity"]: evaluate_sparsity,
        MODEL_KEYS["distillation"]: evaluate_distillation,
        MODEL_KEYS["recipe"]: evaluate_recipe,
    }[model_key]
    return evaluator(canonical_track, **plain_inputs)


def capture_experiment(model_key: str, **inputs: Any) -> ExperimentSnapshot:
    """Evaluate and freeze one method-specific evidence arm."""
    return snapshot_result(model_key, replay(model_key, inputs))


def restore_snapshot(value: Mapping[str, Any]) -> ExperimentSnapshot:
    """Validate and restore a JSON-decoded snapshot without evaluating it."""
    if not isinstance(value, Mapping):
        raise TypeError("snapshot must be a mapping")
    required = {"model_key", "track_id", "inputs", "result"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"snapshot missing: {', '.join(sorted(missing))}")
    model_key = str(value["model_key"])
    if model_key not in MODEL_KEYS.values():
        raise ValueError(f"unsupported model_key for {MODEL_ID}: {model_key}")
    track_id = str(value["track_id"])
    if track_id not in CANONICAL_TRACK_IDS:
        raise ValueError("snapshot track_id must be canonical")
    inputs = value["inputs"]
    result = value["result"]
    if not isinstance(inputs, Mapping) or not isinstance(result, Mapping):
        raise TypeError("snapshot inputs and result must be mappings")
    if inputs.get("track_id") != track_id:
        raise ValueError("snapshot track_id must match inputs.track_id")
    detached = json.loads(json.dumps({"inputs": inputs, "result": result}, allow_nan=False))
    return ExperimentSnapshot(
        model_key=model_key,
        track_id=track_id,
        inputs=detached["inputs"],
        result=detached["result"],
    )


def replay_experiment(
    snapshot: ExperimentSnapshot | Mapping[str, Any],
) -> ExperimentSnapshot:
    """Restore and directly replay an evidence arm from its plain inputs."""
    restored = snapshot if isinstance(snapshot, ExperimentSnapshot) else restore_snapshot(snapshot)
    replayed = replay(restored.model_key, restored.inputs)
    return snapshot_result(restored.model_key, replayed)


def compression_model_baseline_size(track_id: str) -> Quantity:
    """Return CompressionModel's FP32 baseline for a cross-check in tests/UI notes."""
    scenario = get_scenario(track_id)
    return CompressionModel().solve(
        scenario.model, scenario.hardware, method="quantization", target_bitwidth=32
    ).original_size_gb.to("byte")


__all__ = [
    "MODEL_ID",
    "MODEL_KEYS",
    "RECIPE_SEQUENCES",
    "SPARSITY_CASES",
    "CompressionResult",
    "CompressionScenario",
    "CANONICAL_TRACK_IDS",
    "DistillationResult",
    "RecipeResult",
    "RecipeStage",
    "ExperimentSnapshot",
    "TRACK_IDS",
    "capture_experiment",
    "compression_model_baseline_size",
    "evaluate_distillation",
    "evaluate_precision",
    "evaluate_recipe",
    "evaluate_recipe_id",
    "evaluate_resource",
    "evaluate_sparsity",
    "evaluate_sparsity_case",
    "get_scenario",
    "replay",
    "replay_experiment",
    "restore_snapshot",
    "snapshot_result",
    "to_jsonable",
]
