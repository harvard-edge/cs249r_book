"""Chapter 6 architecture experiments grounded in operation and state counts.

The fixtures in this module compare architecture families on a matched task within
each teaching track.  Physical results are analytical scenario calculations.
Quality observations are deliberately separate, explicitly illustrative fixtures;
they are never inferred from an architecture family or a hardware result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Literal, Mapping

from mlsysim.core.units import ureg


TrackId = Literal["mobile", "tinyml", "edge", "cloud"]
Family = Literal["dense", "conv1d", "conv2d", "recurrent", "attention"]
ScaleAxis = Literal["resolution", "window", "context"]
EvidenceKind = Literal["illustrative", "empirical"]
MODEL_KEY = "v1_06.evaluate_candidate.v1"


@dataclass(frozen=True)
class QualityObservation:
    """A supplied task-quality observation, independent of system physics."""

    training_examples: int
    quality_percent: float


@dataclass(frozen=True)
class QualityCurve:
    evidence_kind: EvidenceKind
    metric: str
    observations: tuple[QualityObservation, ...]
    source: str


@dataclass(frozen=True)
class ArchitectureCandidate:
    candidate_id: str
    label: str
    family: Family
    compatible_axis: ScaleAxis
    input_channels: int
    hidden_width: int
    output_width: int
    layers: int = 1
    kernel_size: int = 1
    stride: int = 1
    gates: int = 1
    heads: int = 1
    patch_size: int = 1
    causal_attention: bool = False
    materialize_attention_scores: bool = True
    quality_curve: QualityCurve | None = None


@dataclass(frozen=True)
class ArchitectureScenario:
    track_id: TrackId
    label: str
    task: str
    scale_axis: ScaleAxis
    baseline_scale: int
    sweep_values: tuple[int, ...]
    sample_rate_hz: int
    batch_size: int
    bytes_per_element: int
    compute_rate: object
    memory_bandwidth: object
    serial_step_latency: object
    memory_budget: object
    latency_budget: object
    quality_floor_percent: float
    training_examples: int
    candidates: tuple[ArchitectureCandidate, ...]


@dataclass(frozen=True)
class PhysicalSignature:
    candidate_id: str
    scale_axis: ScaleAxis
    scale_value: int
    logical_steps: int
    tokens_or_positions: int
    parameters: int
    macs: int
    flops: object
    weight_memory: object
    activation_memory: object
    resident_state_memory: object
    attention_score_memory: object
    total_runtime_memory: object
    traffic: object
    arithmetic_intensity: object
    parallel_time: object
    serial_floor_time: object
    latency: object
    bottleneck: Literal["compute", "memory", "serial dependency"]


@dataclass(frozen=True)
class EvaluationInputs:
    """Resolved arguments sufficient to replay one candidate evaluation."""

    track_id: TrackId
    candidate_id: str
    scale_value: int
    training_examples: int
    compute_multiplier: float
    bandwidth_multiplier: float
    memory_budget: object
    latency_budget: object
    quality_floor_percent: float


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    inputs: EvaluationInputs
    signature: PhysicalSignature
    quality_percent: float
    quality_evidence_kind: EvidenceKind
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class ScalingPoint:
    scale_value: int
    evaluation: CandidateEvaluation


@dataclass(frozen=True)
class ArchitectureDecision:
    selected_id: str | None
    feasible_ids: tuple[str, ...]
    rejected: tuple[tuple[str, tuple[str, ...]], ...]
    priority: Literal["quality", "latency", "memory"]


def _curve(*values: tuple[int, float]) -> QualityCurve:
    return QualityCurve(
        evidence_kind="illustrative",
        metric="matched-task validation quality",
        observations=tuple(QualityObservation(*value) for value in values),
        source="Chapter 6 matched-task scenario assumption; not a benchmark measurement",
    )


def _candidate(
    candidate_id: str,
    label: str,
    family: Family,
    axis: ScaleAxis,
    *,
    input_channels: int,
    hidden_width: int,
    output_width: int,
    layers: int = 1,
    kernel_size: int = 1,
    stride: int = 1,
    gates: int = 1,
    heads: int = 1,
    patch_size: int = 1,
    causal_attention: bool = False,
    materialize_attention_scores: bool = True,
    quality: tuple[tuple[int, float], ...],
) -> ArchitectureCandidate:
    return ArchitectureCandidate(
        candidate_id=candidate_id,
        label=label,
        family=family,
        compatible_axis=axis,
        input_channels=input_channels,
        hidden_width=hidden_width,
        output_width=output_width,
        layers=layers,
        kernel_size=kernel_size,
        stride=stride,
        gates=gates,
        heads=heads,
        patch_size=patch_size,
        causal_attention=causal_attention,
        materialize_attention_scores=materialize_attention_scores,
        quality_curve=_curve(*quality),
    )


_SCENARIOS: dict[TrackId, ArchitectureScenario] = {
    "mobile": ArchitectureScenario(
        track_id="mobile",
        label="Mobile",
        task="matched local image classification",
        scale_axis="resolution",
        baseline_scale=224,
        sweep_values=(112, 160, 224, 320, 448),
        sample_rate_hz=1,
        batch_size=1,
        bytes_per_element=2,
        compute_rate=1.2 * ureg.TFLOP / ureg.second,
        memory_bandwidth=48 * ureg.GB / ureg.second,
        serial_step_latency=2 * ureg.microsecond,
        memory_budget=220 * ureg.MB,
        latency_budget=25 * ureg.millisecond,
        quality_floor_percent=80.0,
        training_examples=20_000,
        candidates=(
            _candidate(
                "mobile_conv",
                "Local convolution",
                "conv2d",
                "resolution",
                input_channels=3,
                hidden_width=32,
                output_width=20,
                layers=4,
                kernel_size=3,
                stride=2,
                quality=((2_000, 68.0), (10_000, 79.0), (20_000, 84.0)),
            ),
            _candidate(
                "mobile_dense",
                "Dense image model",
                "dense",
                "resolution",
                input_channels=3,
                hidden_width=96,
                output_width=20,
                layers=2,
                quality=((2_000, 52.0), (10_000, 70.0), (20_000, 78.0)),
            ),
            _candidate(
                "mobile_attention",
                "Patch attention",
                "attention",
                "resolution",
                input_channels=3,
                hidden_width=128,
                output_width=20,
                layers=4,
                heads=4,
                patch_size=16,
                quality=((2_000, 55.0), (10_000, 77.0), (20_000, 86.0)),
            ),
        ),
    ),
    "tinyml": ArchitectureScenario(
        track_id="tinyml",
        label="TinyML",
        task="matched wearable signal classification",
        scale_axis="window",
        baseline_scale=8,
        sweep_values=(2, 4, 8, 12, 24),
        sample_rate_hz=50,
        batch_size=1,
        bytes_per_element=1,
        compute_rate=160 * ureg.MFLOP / ureg.second,
        memory_bandwidth=80 * ureg.MB / ureg.second,
        serial_step_latency=8 * ureg.microsecond,
        memory_budget=450 * ureg.KiB,
        latency_budget=120 * ureg.millisecond,
        quality_floor_percent=78.0,
        training_examples=8_000,
        candidates=(
            _candidate(
                "tiny_conv",
                "Streaming temporal convolution",
                "conv1d",
                "window",
                input_channels=6,
                hidden_width=24,
                output_width=6,
                layers=3,
                kernel_size=5,
                quality=((1_000, 70.0), (4_000, 78.0), (8_000, 82.0)),
            ),
            _candidate(
                "tiny_recurrent",
                "Recurrent sequence model",
                "recurrent",
                "window",
                input_channels=6,
                hidden_width=32,
                output_width=6,
                layers=1,
                gates=4,
                quality=((1_000, 68.0), (4_000, 80.0), (8_000, 84.0)),
            ),
            _candidate(
                "tiny_attention",
                "Window attention",
                "attention",
                "window",
                input_channels=6,
                hidden_width=48,
                output_width=6,
                layers=2,
                heads=4,
                patch_size=10,
                quality=((1_000, 61.0), (4_000, 79.0), (8_000, 86.0)),
            ),
        ),
    ),
    "edge": ArchitectureScenario(
        track_id="edge",
        label="Edge",
        task="matched vehicle-local image classification",
        scale_axis="resolution",
        baseline_scale=640,
        sweep_values=(320, 480, 640, 800, 960),
        sample_rate_hz=1,
        batch_size=1,
        bytes_per_element=2,
        compute_rate=12 * ureg.TFLOP / ureg.second,
        memory_bandwidth=180 * ureg.GB / ureg.second,
        serial_step_latency=1.5 * ureg.microsecond,
        memory_budget=1.5 * ureg.GB,
        latency_budget=45 * ureg.millisecond,
        quality_floor_percent=84.0,
        training_examples=50_000,
        candidates=(
            _candidate(
                "edge_conv",
                "Spatial convolution",
                "conv2d",
                "resolution",
                input_channels=3,
                hidden_width=96,
                output_width=40,
                layers=8,
                kernel_size=3,
                stride=2,
                quality=((5_000, 72.0), (20_000, 83.0), (50_000, 88.0)),
            ),
            _candidate(
                "edge_dense",
                "Dense image model",
                "dense",
                "resolution",
                input_channels=3,
                hidden_width=128,
                output_width=40,
                layers=3,
                quality=((5_000, 58.0), (20_000, 73.0), (50_000, 82.0)),
            ),
            _candidate(
                "edge_attention",
                "Patch attention",
                "attention",
                "resolution",
                input_channels=3,
                hidden_width=192,
                output_width=40,
                layers=6,
                heads=6,
                patch_size=16,
                quality=((5_000, 64.0), (20_000, 84.0), (50_000, 91.0)),
            ),
        ),
    ),
    "cloud": ArchitectureScenario(
        track_id="cloud",
        label="Cloud",
        task="matched document classification",
        scale_axis="context",
        baseline_scale=512,
        sweep_values=(128, 256, 512, 1024, 2048),
        sample_rate_hz=1,
        batch_size=8,
        bytes_per_element=2,
        compute_rate=120 * ureg.TFLOP / ureg.second,
        memory_bandwidth=2.4 * ureg.TB / ureg.second,
        serial_step_latency=1 * ureg.microsecond,
        memory_budget=20 * ureg.GB,
        latency_budget=80 * ureg.millisecond,
        quality_floor_percent=86.0,
        training_examples=100_000,
        candidates=(
            _candidate(
                "cloud_recurrent",
                "Recurrent encoder",
                "recurrent",
                "context",
                input_channels=256,
                hidden_width=512,
                output_width=12,
                layers=2,
                gates=4,
                quality=((10_000, 76.0), (50_000, 85.0), (100_000, 88.0)),
            ),
            _candidate(
                "cloud_attention",
                "Bidirectional attention encoder",
                "attention",
                "context",
                input_channels=256,
                hidden_width=768,
                output_width=12,
                layers=12,
                heads=12,
                quality=((10_000, 78.0), (50_000, 88.0), (100_000, 92.0)),
            ),
            _candidate(
                "cloud_dense",
                "Pooled dense encoder",
                "dense",
                "context",
                input_channels=256,
                hidden_width=512,
                output_width=12,
                layers=3,
                quality=((10_000, 70.0), (50_000, 81.0), (100_000, 85.0)),
            ),
        ),
    ),
}

_TRACK_ALIASES = {
    "iphone": "mobile",
    "oura_ring": "tinyml",
    "robotaxi": "edge",
    "cloud_fleet": "cloud",
}


def get_scenario(track_id: str) -> ArchitectureScenario:
    """Return one of the four matched-task teaching scenarios."""
    canonical = _TRACK_ALIASES.get(track_id, track_id)
    if canonical not in _SCENARIOS:
        raise ValueError(f"unknown track {track_id!r}; expected mobile, tinyml, edge, or cloud")
    return _SCENARIOS[canonical]  # type: ignore[index]


def get_candidate(scenario: ArchitectureScenario, candidate_id: str) -> ArchitectureCandidate:
    for candidate in scenario.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise ValueError(f"candidate {candidate_id!r} is not part of the {scenario.track_id} task")


def quality_at(candidate: ArchitectureCandidate, training_examples: int) -> float:
    """Interpolate only the candidate's supplied quality observations."""
    if training_examples <= 0:
        raise ValueError("training_examples must be positive")
    if candidate.quality_curve is None or not candidate.quality_curve.observations:
        raise ValueError(f"candidate {candidate.candidate_id!r} has no quality evidence")
    points = tuple(sorted(candidate.quality_curve.observations, key=lambda item: item.training_examples))
    if training_examples <= points[0].training_examples:
        return points[0].quality_percent
    if training_examples >= points[-1].training_examples:
        return points[-1].quality_percent
    for left, right in zip(points, points[1:]):
        if left.training_examples <= training_examples <= right.training_examples:
            fraction = (training_examples - left.training_examples) / (right.training_examples - left.training_examples)
            return left.quality_percent + fraction * (right.quality_percent - left.quality_percent)
    raise AssertionError("quality interpolation interval was not found")


def _positions(scenario: ArchitectureScenario, candidate: ArchitectureCandidate, scale: int) -> int:
    if scale <= 0:
        raise ValueError("scale_value must be positive")
    if scenario.scale_axis != candidate.compatible_axis:
        raise ValueError(f"{candidate.candidate_id} supports {candidate.compatible_axis}, not {scenario.scale_axis}")
    if scenario.scale_axis == "resolution":
        if candidate.family == "attention":
            return ceil(scale / candidate.patch_size) ** 2 + 1
        return scale * scale
    if scenario.scale_axis == "window":
        raw_steps = scale * scenario.sample_rate_hz
        if candidate.family == "attention":
            return ceil(raw_steps / candidate.patch_size)
        return raw_steps
    return scale


def _validate_candidate(candidate: ArchitectureCandidate) -> None:
    if candidate.family not in ("dense", "conv1d", "conv2d", "recurrent", "attention"):
        raise ValueError(f"unsupported architecture family {candidate.family!r}")
    positive_fields = {
        "input_channels": candidate.input_channels,
        "hidden_width": candidate.hidden_width,
        "output_width": candidate.output_width,
        "layers": candidate.layers,
        "kernel_size": candidate.kernel_size,
        "stride": candidate.stride,
        "gates": candidate.gates,
        "heads": candidate.heads,
        "patch_size": candidate.patch_size,
    }
    invalid = tuple(name for name, value in positive_fields.items() if value <= 0)
    if invalid:
        raise ValueError(f"{candidate.candidate_id} has nonpositive topology fields: {', '.join(invalid)}")


def physical_signature(
    scenario: ArchitectureScenario,
    candidate_id: str,
    *,
    scale_value: int | None = None,
    compute_multiplier: float = 1.0,
    bandwidth_multiplier: float = 1.0,
) -> PhysicalSignature:
    """Derive counts, memory, traffic, and execution bounds from topology.

    Execution uses a critical-path lower-bound model.  The parallel roofline
    bound is ``max(FLOPs / compute rate, traffic / bandwidth)``.  A dependency
    bound is ``logical steps * minimum step latency``.  Reported latency is the
    maximum of those bounds, never their sum, because recurrent work is already
    included in the operation count.  The per-step latency is a disclosed
    scenario assumption for the minimum schedule/control duration of a dependent
    step; this educational model is not a device benchmark.
    """
    if compute_multiplier <= 0 or bandwidth_multiplier <= 0:
        raise ValueError("compute and bandwidth multipliers must be positive")
    candidate = get_candidate(scenario, candidate_id)
    _validate_candidate(candidate)
    scale = scenario.baseline_scale if scale_value is None else int(scale_value)
    positions = _positions(scenario, candidate, scale)
    batch = scenario.batch_size
    width = candidate.hidden_width
    outputs = candidate.output_width
    input_channels = candidate.input_channels
    element_bytes = scenario.bytes_per_element
    state_elements = 0
    score_elements = 0

    if candidate.family == "dense":
        input_width = positions * input_channels
        parameters = input_width * width + (candidate.layers - 1) * width * width + width * outputs
        macs = batch * parameters
        activation_elements = batch * (input_width + candidate.layers * width + outputs)
        logical_steps = candidate.layers + 1
    elif candidate.family in ("conv1d", "conv2d"):
        spatial_positions = positions
        if candidate.family == "conv2d":
            spatial_positions = ceil(scale / candidate.stride) ** 2
        parameters = (
            candidate.kernel_size
            * (candidate.kernel_size if candidate.family == "conv2d" else 1)
            * input_channels
            * width
            + (candidate.layers - 1)
            * candidate.kernel_size
            * (candidate.kernel_size if candidate.family == "conv2d" else 1)
            * width
            * width
            + width * outputs
        )
        first_kernel_macs = (
            spatial_positions
            * candidate.kernel_size
            * (candidate.kernel_size if candidate.family == "conv2d" else 1)
            * input_channels
            * width
        )
        later_kernel_macs = (
            (candidate.layers - 1)
            * spatial_positions
            * candidate.kernel_size
            * (candidate.kernel_size if candidate.family == "conv2d" else 1)
            * width
            * width
        )
        macs = batch * (first_kernel_macs + later_kernel_macs + width * outputs)
        activation_elements = batch * (
            positions * input_channels + candidate.layers * spatial_positions * width + outputs
        )
        logical_steps = candidate.layers + 1
    elif candidate.family == "recurrent":
        first_layer_parameters = candidate.gates * (input_channels * width + width * width + width)
        later_layer_parameters = (candidate.layers - 1) * candidate.gates * (2 * width * width + width)
        parameters = first_layer_parameters + later_layer_parameters + width * outputs
        per_step_macs = candidate.gates * (input_channels * width + width * width)
        per_step_macs += (candidate.layers - 1) * candidate.gates * 2 * width * width
        macs = batch * (positions * per_step_macs + width * outputs)
        activation_elements = batch * (positions * input_channels + positions * width + outputs)
        state_elements = batch * candidate.layers * width * (2 if candidate.gates == 4 else 1)
        logical_steps = positions * candidate.layers
    elif candidate.family == "attention":
        tokens = positions
        parameters = input_channels * width + candidate.layers * 12 * width * width + width * outputs
        input_projection_macs = tokens * input_channels * width
        projection_and_ffn_macs = candidate.layers * 12 * tokens * width * width
        attention_macs = candidate.layers * 2 * tokens * tokens * width
        macs = batch * (input_projection_macs + projection_and_ffn_macs + attention_macs + width * outputs)
        activation_elements = batch * (tokens * input_channels + candidate.layers * 4 * tokens * width + outputs)
        if candidate.causal_attention:
            state_elements = batch * 2 * candidate.layers * tokens * width
        if candidate.materialize_attention_scores:
            score_elements = batch * candidate.layers * candidate.heads * tokens * tokens
        logical_steps = candidate.layers
    else:  # Kept explicit so a malformed/unsupported family cannot enter a comparison.
        raise ValueError(f"unsupported architecture family {candidate.family!r}")

    flops = (2 * macs) * ureg.flop
    weight_memory = (parameters * element_bytes) * ureg.byte
    activation_memory = (activation_elements * element_bytes) * ureg.byte
    resident_state_memory = (state_elements * element_bytes) * ureg.byte
    attention_score_memory = (score_elements * element_bytes) * ureg.byte
    total_runtime_memory = (weight_memory + activation_memory + resident_state_memory + attention_score_memory).to(
        ureg.byte
    )
    traffic = (weight_memory + 2 * activation_memory + resident_state_memory + attention_score_memory).to(ureg.byte)
    compute_rate = scenario.compute_rate * compute_multiplier
    bandwidth = scenario.memory_bandwidth * bandwidth_multiplier
    compute_time = (flops / compute_rate).to(ureg.second)
    memory_time = (traffic / bandwidth).to(ureg.second)
    parallel_time = max(compute_time, memory_time)
    serial_floor = (logical_steps * scenario.serial_step_latency).to(ureg.second)
    latency = max(parallel_time, serial_floor)
    if serial_floor >= parallel_time:
        bottleneck: Literal["compute", "memory", "serial dependency"] = "serial dependency"
    elif compute_time >= memory_time:
        bottleneck = "compute"
    else:
        bottleneck = "memory"
    arithmetic_intensity = (flops / traffic).to(ureg.flop / ureg.byte)
    return PhysicalSignature(
        candidate_id=candidate.candidate_id,
        scale_axis=scenario.scale_axis,
        scale_value=scale,
        logical_steps=logical_steps,
        tokens_or_positions=positions,
        parameters=parameters,
        macs=macs,
        flops=flops,
        weight_memory=weight_memory,
        activation_memory=activation_memory,
        resident_state_memory=resident_state_memory,
        attention_score_memory=attention_score_memory,
        total_runtime_memory=total_runtime_memory,
        traffic=traffic,
        arithmetic_intensity=arithmetic_intensity,
        parallel_time=parallel_time,
        serial_floor_time=serial_floor,
        latency=latency,
        bottleneck=bottleneck,
    )


def evaluate_candidate(
    scenario: ArchitectureScenario,
    candidate_id: str,
    *,
    scale_value: int | None = None,
    training_examples: int | None = None,
    compute_multiplier: float = 1.0,
    bandwidth_multiplier: float = 1.0,
    memory_budget=None,
    latency_budget=None,
    quality_floor_percent: float | None = None,
) -> CandidateEvaluation:
    candidate = get_candidate(scenario, candidate_id)
    signature = physical_signature(
        scenario,
        candidate_id,
        scale_value=scale_value,
        compute_multiplier=compute_multiplier,
        bandwidth_multiplier=bandwidth_multiplier,
    )
    examples = scenario.training_examples if training_examples is None else training_examples
    quality = quality_at(candidate, examples)
    memory_limit = scenario.memory_budget if memory_budget is None else memory_budget
    latency_limit = scenario.latency_budget if latency_budget is None else latency_budget
    quality_floor = scenario.quality_floor_percent if quality_floor_percent is None else quality_floor_percent
    resolved_inputs = EvaluationInputs(
        track_id=scenario.track_id,
        candidate_id=candidate_id,
        scale_value=signature.scale_value,
        training_examples=examples,
        compute_multiplier=compute_multiplier,
        bandwidth_multiplier=bandwidth_multiplier,
        memory_budget=memory_limit,
        latency_budget=latency_limit,
        quality_floor_percent=quality_floor,
    )
    violations: list[str] = []
    if signature.total_runtime_memory > memory_limit:
        violations.append("memory budget")
    if signature.latency > latency_limit:
        violations.append("latency budget")
    if quality < quality_floor:
        violations.append("quality floor")
    evidence_kind = candidate.quality_curve.evidence_kind if candidate.quality_curve else "illustrative"
    return CandidateEvaluation(
        candidate_id=candidate_id,
        inputs=resolved_inputs,
        signature=signature,
        quality_percent=quality,
        quality_evidence_kind=evidence_kind,
        feasible=not violations,
        violations=tuple(violations),
    )


def sweep_scale(
    scenario: ArchitectureScenario,
    candidate_id: str,
    *,
    axis: ScaleAxis | None = None,
    values: tuple[int, ...] | None = None,
) -> tuple[ScalingPoint, ...]:
    """Sweep only the scenario's architecture-compatible input commitment."""
    requested_axis = scenario.scale_axis if axis is None else axis
    if requested_axis != scenario.scale_axis:
        raise ValueError(f"{scenario.track_id} supports a {scenario.scale_axis} sweep, not {requested_axis}")
    candidate = get_candidate(scenario, candidate_id)
    if candidate.compatible_axis != requested_axis:
        raise ValueError(f"{candidate_id} is incompatible with a {requested_axis} sweep")
    sweep_values = scenario.sweep_values if values is None else values
    if not sweep_values:
        raise ValueError("sweep values cannot be empty")
    return tuple(
        ScalingPoint(value, evaluate_candidate(scenario, candidate_id, scale_value=value)) for value in sweep_values
    )


def _quantity_to_plain(value) -> dict[str, object]:
    if not hasattr(value, "magnitude") or not hasattr(value, "units"):
        raise TypeError("evaluation budget must be a Pint quantity")
    return {"value": float(value.magnitude), "unit": str(value.units)}


def _quantity_from_plain(value: object, field_name: str):
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a quantity mapping")
    if "value" not in value or "unit" not in value:
        raise ValueError(f"{field_name} requires value and unit")
    return float(value["value"]) * ureg(str(value["unit"]))


def evaluation_inputs_to_dict(inputs: EvaluationInputs) -> dict[str, object]:
    """Serialize replay inputs to the JSON-safe shared evidence format."""
    return {
        "track_id": inputs.track_id,
        "candidate_id": inputs.candidate_id,
        "scale_value": inputs.scale_value,
        "training_examples": inputs.training_examples,
        "compute_multiplier": inputs.compute_multiplier,
        "bandwidth_multiplier": inputs.bandwidth_multiplier,
        "memory_budget": _quantity_to_plain(inputs.memory_budget),
        "latency_budget": _quantity_to_plain(inputs.latency_budget),
        "quality_floor_percent": inputs.quality_floor_percent,
    }


def evaluation_inputs_from_dict(values: Mapping[str, object]) -> EvaluationInputs:
    """Restore and validate typed replay inputs from a JSON-decoded mapping."""
    required = {
        "track_id",
        "candidate_id",
        "scale_value",
        "training_examples",
        "compute_multiplier",
        "bandwidth_multiplier",
        "memory_budget",
        "latency_budget",
        "quality_floor_percent",
    }
    missing = required.difference(values)
    if missing:
        raise ValueError(f"evaluation inputs missing: {', '.join(sorted(missing))}")
    track_id = str(values["track_id"])
    scenario = get_scenario(track_id)
    candidate_id = str(values["candidate_id"])
    get_candidate(scenario, candidate_id)
    return EvaluationInputs(
        track_id=scenario.track_id,
        candidate_id=candidate_id,
        scale_value=int(values["scale_value"]),
        training_examples=int(values["training_examples"]),
        compute_multiplier=float(values["compute_multiplier"]),
        bandwidth_multiplier=float(values["bandwidth_multiplier"]),
        memory_budget=_quantity_from_plain(values["memory_budget"], "memory_budget"),
        latency_budget=_quantity_from_plain(values["latency_budget"], "latency_budget"),
        quality_floor_percent=float(values["quality_floor_percent"]),
    )


def evaluation_replay_snapshot(evaluation: CandidateEvaluation) -> dict[str, object]:
    """Return the versioned JSON-safe replay payload for an evidence arm."""
    return {
        "model_key": MODEL_KEY,
        "inputs": evaluation_inputs_to_dict(evaluation.inputs),
    }


def evaluation_to_dict(evaluation: CandidateEvaluation) -> dict[str, object]:
    """Serialize one complete evaluation for JSON-backed evidence capture."""
    signature = evaluation.signature
    scenario = get_scenario(evaluation.inputs.track_id)
    return {
        "inputs": evaluation_inputs_to_dict(evaluation.inputs),
        "candidate_id": evaluation.candidate_id,
        "interpretation": scaling_interpretation(scenario, evaluation.candidate_id),
        "quality_percent": evaluation.quality_percent,
        "quality_evidence_kind": evaluation.quality_evidence_kind,
        "feasible": evaluation.feasible,
        "violations": list(evaluation.violations),
        "scale_axis": signature.scale_axis,
        "scale_value": signature.scale_value,
        "logical_steps": signature.logical_steps,
        "tokens_or_positions": signature.tokens_or_positions,
        "parameters": signature.parameters,
        "macs": signature.macs,
        "flops": _quantity_to_plain(signature.flops),
        "weight_memory": _quantity_to_plain(signature.weight_memory),
        "activation_memory": _quantity_to_plain(signature.activation_memory),
        "resident_state_memory": _quantity_to_plain(signature.resident_state_memory),
        "attention_score_memory": _quantity_to_plain(signature.attention_score_memory),
        "total_runtime_memory": _quantity_to_plain(signature.total_runtime_memory),
        "traffic": _quantity_to_plain(signature.traffic),
        "arithmetic_intensity": _quantity_to_plain(signature.arithmetic_intensity),
        "parallel_time": _quantity_to_plain(signature.parallel_time),
        "serial_floor_time": _quantity_to_plain(signature.serial_floor_time),
        "latency": _quantity_to_plain(signature.latency),
        "latency_ms": signature.latency.m_as(ureg.millisecond),
        "parallel_time_ms": signature.parallel_time.m_as(ureg.millisecond),
        "serial_floor_time_ms": signature.serial_floor_time.m_as(ureg.millisecond),
        "weight_memory_mb": signature.weight_memory.m_as(ureg.MB),
        "activation_memory_mb": signature.activation_memory.m_as(ureg.MB),
        "resident_state_memory_mb": signature.resident_state_memory.m_as(ureg.MB),
        "attention_score_memory_mb": signature.attention_score_memory.m_as(ureg.MB),
        "total_runtime_memory_mb": signature.total_runtime_memory.m_as(ureg.MB),
        "bottleneck": signature.bottleneck,
    }


def replay_evaluation(
    inputs: EvaluationInputs | Mapping[str, object],
) -> CandidateEvaluation:
    """Replay an immutable evidence arm from typed or JSON-decoded inputs."""
    if isinstance(inputs, Mapping):
        inputs = evaluation_inputs_from_dict(inputs)
    return evaluate_candidate(
        get_scenario(inputs.track_id),
        inputs.candidate_id,
        scale_value=inputs.scale_value,
        training_examples=inputs.training_examples,
        compute_multiplier=inputs.compute_multiplier,
        bandwidth_multiplier=inputs.bandwidth_multiplier,
        memory_budget=inputs.memory_budget,
        latency_budget=inputs.latency_budget,
        quality_floor_percent=inputs.quality_floor_percent,
    )


def replay_snapshot(snapshot: Mapping[str, object]) -> CandidateEvaluation:
    """Replay a shared evidence snapshot after checking its model method/version."""
    if snapshot.get("model_key") != MODEL_KEY:
        raise ValueError(f"unsupported model_key {snapshot.get('model_key')!r}; expected {MODEL_KEY!r}")
    inputs = snapshot.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("snapshot inputs must be a mapping")
    return replay_evaluation(inputs)


def choose_architecture(
    scenario: ArchitectureScenario,
    *,
    priority: Literal["quality", "latency", "memory"] = "quality",
    scale_value: int | None = None,
    training_examples: int | None = None,
    memory_budget=None,
    latency_budget=None,
    quality_floor_percent: float | None = None,
) -> ArchitectureDecision:
    """Choose among feasible candidates by one observable metric, without a score."""
    evaluations = tuple(
        evaluate_candidate(
            scenario,
            candidate.candidate_id,
            scale_value=scale_value,
            training_examples=training_examples,
            memory_budget=memory_budget,
            latency_budget=latency_budget,
            quality_floor_percent=quality_floor_percent,
        )
        for candidate in scenario.candidates
    )
    feasible = tuple(item for item in evaluations if item.feasible)
    if not feasible:
        selected_id = None
    elif priority == "quality":
        selected_id = max(feasible, key=lambda item: item.quality_percent).candidate_id
    elif priority == "latency":
        selected_id = min(feasible, key=lambda item: item.signature.latency).candidate_id
    else:
        selected_id = min(feasible, key=lambda item: item.signature.total_runtime_memory).candidate_id
    return ArchitectureDecision(
        selected_id=selected_id,
        feasible_ids=tuple(item.candidate_id for item in feasible),
        rejected=tuple((item.candidate_id, item.violations) for item in evaluations if not item.feasible),
        priority=priority,
    )


def highest_quality_candidate(scenario: ArchitectureScenario, *, training_examples: int | None = None) -> str:
    """Return the candidate with the highest supplied matched-task quality."""
    examples = scenario.training_examples if training_examples is None else training_examples
    return max(
        scenario.candidates,
        key=lambda candidate: quality_at(candidate, examples),
    ).candidate_id


def default_scale_candidate(scenario: ArchitectureScenario) -> str:
    """Return the candidate intended for Part C scaling analysis."""
    for candidate in scenario.candidates:
        if candidate.family == "attention":
            return candidate.candidate_id
    return scenario.candidates[0].candidate_id


def scaling_interpretation(scenario: ArchitectureScenario, candidate_id: str) -> str:
    """Explain how the architecture's shape or kernel boundary scales along the track axis."""
    candidate = get_candidate(scenario, candidate_id)
    axis = scenario.scale_axis
    if candidate.family == "conv1d":
        return (
            f"1D temporal convolution maintains fixed kernel parameters across the {axis}; "
            "activation memory scales linearly with time steps along the temporal boundary."
        )
    if candidate.family == "conv2d":
        return (
            f"2D spatial convolution maintains fixed kernel parameters via spatial weight sharing; "
            f"activations scale quadratically with 2D {axis} across the strided kernel boundary."
        )
    if candidate.family == "recurrent":
        return (
            f"Recurrent hidden state stays bounded per sequence stream; "
            f"sequential dependency steps and operation counts grow linearly with {axis}."
        )
    if candidate.family == "dense":
        return (
            f"Dense projection lacks weight sharing across the {axis} boundary; "
            "parameter and activation commitments scale directly with input dimensions."
        )
    if candidate.family == "attention":
        return (
            f"Attention pairwise score memory scales quadratically with {axis} positions, "
            "while channel projections and activations scale linearly."
        )
    return f"{candidate.label} scales along the {axis} axis."


def with_execution_resources(
    scenario: ArchitectureScenario,
    *,
    compute_rate=None,
    memory_bandwidth=None,
) -> ArchitectureScenario:
    """Create a comparison arm while keeping task and candidates identical."""
    return replace(
        scenario,
        compute_rate=scenario.compute_rate if compute_rate is None else compute_rate,
        memory_bandwidth=(scenario.memory_bandwidth if memory_bandwidth is None else memory_bandwidth),
    )


__all__ = [
    "ArchitectureCandidate",
    "ArchitectureDecision",
    "ArchitectureScenario",
    "CandidateEvaluation",
    "EvaluationInputs",
    "MODEL_KEY",
    "PhysicalSignature",
    "QualityCurve",
    "QualityObservation",
    "ScalingPoint",
    "choose_architecture",
    "default_scale_candidate",
    "evaluate_candidate",
    "evaluation_inputs_from_dict",
    "evaluation_inputs_to_dict",
    "evaluation_replay_snapshot",
    "evaluation_to_dict",
    "get_candidate",
    "get_scenario",
    "highest_quality_candidate",
    "physical_signature",
    "quality_at",
    "replay_evaluation",
    "replay_snapshot",
    "scaling_interpretation",
    "sweep_scale",
    "with_execution_resources",
]


def attention_state_comparison(track_id: str, scale_value: int | None = None) -> dict[str, object]:
    """Compare full score storage with no full score materialization at fixed input.

    Uses the same native architecture fixture and physical_signature equation in
    both arms. The unmaterialized arm removes only the full score tensor;
    implementation-specific tile scratch is excluded, so it is a state lower
    bound, not a FlashAttention memory/latency measurement.
    """
    scenario = get_scenario(track_id)
    candidate = next(item for item in scenario.candidates if item.family == "attention")
    materialized = replace(candidate, materialize_attention_scores=True)
    unmaterialized = replace(candidate, materialize_attention_scores=False)

    def arm(item):
        changed = replace(scenario, candidates=tuple(item if c.candidate_id == item.candidate_id else c for c in scenario.candidates))
        sig = physical_signature(changed, item.candidate_id, scale_value=scale_value)
        return {"materialize_attention_scores": item.materialize_attention_scores,
                "tokens_or_positions": sig.tokens_or_positions, "parameters": sig.parameters,
                "macs": sig.macs, "flops": sig.flops,
                "attention_score_memory": sig.attention_score_memory,
                "activation_memory": sig.activation_memory,
                "resident_state_memory": sig.resident_state_memory,
                "total_runtime_memory": sig.total_runtime_memory,
                "fits_memory": sig.total_runtime_memory <= scenario.memory_budget}

    baseline, result = arm(materialized), arm(unmaterialized)
    return {"inputs": {"track_id": scenario.track_id, "scale_value": scenario.baseline_scale if scale_value is None else scale_value},
            "baseline": baseline, "result": result,
            "memory_budget": scenario.memory_budget,
            "dense_attention_arithmetic_unchanged": baseline["macs"] == result["macs"],
            "scope_note": "Fixed candidate and history; full dense score arithmetic is unchanged. Removing full score storage excludes tile scratch and is not a measured kernel speedup or quality result."}


def _reference_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _reference_element_bytes(precision: str):
    from mlsysim.core.units import resolve_precision
    _, byte_width = resolve_precision(precision)
    return byte_width.to(ureg.byte)


def language_phase_reference(*, history_tokens: int = 128, width: int = 64,
                             layers: int = 2, batch_size: int = 1,
                             precision: str = "fp32") -> dict[str, object]:
    """Analytical language-decoder reference; never the native track workload.

    A bias-free block has four D-by-D attention projections and two FFN
    matrices with expansion four: twelve D-squared weights. Dense square
    attention (including masked positions) is counted for prefill. One decode
    token attends to all existing tokens plus itself. Excludes embeddings,
    output head, normalization, softmax FLOPs, allocators, and kernel workspaces.
    No hardware rates are assumed and no bottleneck label is inferred.
    """
    n = _reference_positive_int(history_tokens, "history_tokens")
    d = _reference_positive_int(width, "width")
    l = _reference_positive_int(layers, "layers")
    b = _reference_positive_int(batch_size, "batch_size")
    element = _reference_element_bytes(precision)
    parameters = 12 * l * d * d
    weights = parameters * element
    kv_before = 2 * b * l * n * d * element
    kv_after = 2 * b * l * (n + 1) * d * element
    prefill_macs = b * l * (12 * n * d * d + 2 * n * n * d)
    decode_macs = b * l * (12 * d * d + 2 * (n + 1) * d)
    prefill = {"tokens_processed_per_request": n, "macs": prefill_macs,
               "flops": 2 * prefill_macs * ureg.flop,
               "weight_read_lower_bound": weights,
               "weight_bytes_per_processed_token": weights / (b * n),
               "kv_state": kv_before,
               "kv_write": kv_before}
    decode = {"tokens_processed_per_request": 1, "macs": decode_macs,
              "flops": 2 * decode_macs * ureg.flop,
              "weight_read_lower_bound": weights,
              "weight_bytes_per_processed_token": weights / b,
              "kv_state": kv_after, "kv_read": kv_before,
              "kv_write": 2 * b * l * d * element}
    return {"reference_name": "Language-decoder phase reference",
            "inputs": {"history_tokens": n, "width": d, "layers": l, "batch_size": b, "precision": precision},
            "parameters": parameters, "weight_memory": weights,
            "prefill": prefill, "decode": decode,
            "scope_note": "Analytical bias-free decoder blocks, separate from all native project tasks. Prefill processes the whole prompt; decode emits one next token per request. Weight reads assume one ideal reuse per batch; KV counts include keys and values. No quality, measured latency, or hardware bottleneck prediction."}


def embedding_lookup_reference(*, vocabulary_size: int = 1024, vector_width: int = 32,
                               requested_rows: int = 4, precision: str = "fp32") -> dict[str, object]:
    """Count a table and raw gather output at fixed valid lookup count.

    Requests may repeat an ID. Each requested row is counted as a logical read;
    caches, duplicate coalescing, indices, and downstream arithmetic are outside
    this reference. Vocabulary is capacity, not evidence for unseen categories.
    """
    v = _reference_positive_int(vocabulary_size, "vocabulary_size")
    d = _reference_positive_int(vector_width, "vector_width")
    r = _reference_positive_int(requested_rows, "requested_rows")
    element = _reference_element_bytes(precision)
    return {"reference_name": "Embedding lookup capacity reference",
            "inputs": {"vocabulary_size": v, "vector_width": d, "requested_rows": r, "precision": precision},
            "table_elements": v * d, "table_memory": v * d * element,
            "requested_elements": r * d, "request_read_bytes": r * d * element,
            "output_memory": r * d * element, "lookup_macs": 0,
            "scope_note": "Exact table/gather accounting for valid IDs. Vocabulary changes capacity while fixed requests read the same rows; vector width changes both. No cache, placement, latency, training evidence, or cold-start quality is inferred."}


__all__ += ["attention_state_comparison", "language_phase_reference", "embedding_lookup_reference"]


# Exact reference cases for the chapter's distinction between invariance and
# retaining position. They are constructed logical counterexamples, not samples
# from a training distribution or observations of trained architectures.
STRUCTURE_CASES = (
    ("earlier pulse", (0, 1, 0, 0)),
    ("later pulse", (0, 0, 1, 0)),
    ("no pulse", (0, 0, 0, 0)),
    ("two pulses", (0, 1, 1, 0)),
)
STRUCTURE_MODEL_KEY = "v1_06_experiments.structure_probe"


def structure_probe(*, task: str = "direction", transform: str = "mean") -> dict[str, object]:
    """Find exact representation collisions with incompatible task labels.

    ``direction`` asks whether any pulse occurs in the first half of the
    window. ``presence`` asks whether a pulse occurs anywhere. ``identity``
    retains order; ``mean`` replaces the window with its global mean;
    ``reversal_invariant`` maps a window and its reversal to the same canonical
    ordered tuple. The last operation encodes *only* reversal invariance, not
    arbitrary permutation invariance.

    A collision with different targets proves no deterministic downstream
    function of these representations can satisfy both cases. No collision on
    this finite fixture is only a witness check, not proof of general adequacy.
    """
    if task not in {"direction", "presence"}:
        raise ValueError("task must be direction or presence")
    if transform not in {"identity", "mean", "reversal_invariant"}:
        raise ValueError("transform must be identity, mean, or reversal_invariant")
    rows = []
    for name, values in STRUCTURE_CASES:
        if transform == "mean":
            representation = (sum(values) / len(values),)
        elif transform == "reversal_invariant":
            representation = min(values, values[::-1])
        else:
            representation = values
        label = int(any(values[:len(values) // 2])) if task == "direction" else int(any(values))
        rows.append({"case": name, "input": list(values),
                     "representation": list(representation), "required_label": label})
    collisions = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if left["representation"] == right["representation"]:
                collisions.append({"cases": [left["case"], right["case"]],
                                   "labels": [left["required_label"], right["required_label"]],
                                   "conflicting": left["required_label"] != right["required_label"]})
    conflicts = [pair for pair in collisions if pair["conflicting"]]
    return {"model_key": STRUCTURE_MODEL_KEY,
            "reference_name": "Position and invariance counterexample",
            "inputs": {"task": task, "transform": transform},
            "task_question": "Is there a pulse in the first half?" if task == "direction" else "Is there a pulse anywhere?",
            "rows": rows, "collisions": collisions, "conflicts": conflicts,
            "required_distinctions_preserved": not bool(conflicts),
            "representation_elements_per_case": len(rows[0]["representation"]),
            "scope_note": "Exact logical reference, shared across tracks; not a native project dataset, trained architecture, quality curve, or sample-efficiency measurement. Conflicting labels for identical representations prove an information loss; no conflicts only establishes compatibility with these supplied cases."}


__all__ += ["STRUCTURE_MODEL_KEY", "structure_probe"]
