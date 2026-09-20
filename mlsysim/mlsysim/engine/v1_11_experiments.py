"""Chapter 11 hardware-acceleration experiments.

The functions in this module compose registry hardware facts into five small,
deterministic experiments.  They are analytical teaching models, not measured
benchmark results.  Track defaults are explicitly illustrative workload and
execution-contract assumptions; hardware throughput, bandwidth, capacity,
dispatch, power, and purchase cost come from the hardware registry.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from math import ceil
from typing import Literal, Mapping, Sequence

from ..core.types import Quantity
from ..core.units import Q_, resolve_precision, ureg
from ..hardware import Hardware
from ..hardware.types import HardwareNode


MODEL_ID = "v1_11_experiments"
Track = Literal["tinyml", "mobile", "edge", "cloud"]
Objective = Literal["latency", "energy", "cost"]


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _nonnegative_quantity(value: Quantity, name: str) -> None:
    if value.to_base_units().magnitude < 0:
        raise ValueError(f"{name} must be nonnegative")


def _peak_rate(hardware: HardwareNode, precision: str) -> Quantity:
    key, _ = resolve_precision(precision)
    return hardware.compute.precision_flops.get(key, hardware.compute.peak_flops)


def _hardware_registry_ref(hardware: HardwareNode) -> str | None:
    for tier_name in ("Tiny", "Mobile", "Edge", "Cloud", "Workstation"):
        tier = getattr(Hardware, tier_name)
        for key in dir(tier):
            if key.startswith("_") or key == "list":
                continue
            if getattr(tier, key) is hardware:
                return f"Hardware.{tier_name}.{key}"
    return None


def to_jsonable(value):
    """Convert experiment inputs/results to stable JSON-compatible values.

    Quantities retain both magnitude and unit so a captured experiment can be
    reconstructed with ``Q_(payload["magnitude"], payload["unit"])``.
    """

    if isinstance(value, ureg.Quantity):
        magnitude = value.magnitude
        if hasattr(magnitude, "tolist"):
            magnitude = magnitude.tolist()
        return {"magnitude": magnitude, "unit": str(value.units)}
    if isinstance(value, HardwareNode):
        return {
            "registry_ref": _hardware_registry_ref(value),
            "hardware_name": value.name,
        }
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        items = sorted(value) if isinstance(value, (frozenset, set)) else value
        return [to_jsonable(item) for item in items]
    return value


@dataclass(frozen=True)
class RooflineResult:
    hardware_name: str
    hardware_ref: str | None
    precision: str
    operations: Quantity
    base_bytes_moved: Quantity
    bytes_moved: Quantity
    reuse: float
    compute_scale: float
    bandwidth_scale: float
    arithmetic_intensity: Quantity
    ridge_point: Quantity
    compute_time: Quantity
    memory_time: Quantity
    latency: Quantity
    bottleneck: Literal["compute", "memory"]
    attainable_rate: Quantity


def analyze_roofline(
    *,
    hardware: HardwareNode,
    operations: Quantity,
    base_bytes_moved: Quantity,
    precision: str,
    reuse: float = 1.0,
    compute_scale: float = 1.0,
    bandwidth_scale: float = 1.0,
) -> RooflineResult:
    """Apply the single-node Roofline with independently variable resources.

    ``reuse`` means each off-chip byte supplies that many uses, so effective
    traffic is ``base_bytes_moved / reuse``.  Resource scale factors represent
    counterfactual supply changes and do not alter the workload.
    """

    for value, name in (
        (reuse, "reuse"),
        (compute_scale, "compute_scale"),
        (bandwidth_scale, "bandwidth_scale"),
    ):
        _positive(value, name)
    _nonnegative_quantity(operations, "operations")
    _nonnegative_quantity(base_bytes_moved, "base_bytes_moved")

    peak = (_peak_rate(hardware, precision) * compute_scale).to("flop/s")
    bandwidth = (hardware.memory.bandwidth * bandwidth_scale).to("byte/s")
    moved = (base_bytes_moved / reuse).to("byte")
    compute_time = (operations / peak).to("ms")
    memory_time = (moved / bandwidth).to("ms")
    latency = max(compute_time, memory_time)
    bottleneck = "memory" if memory_time > compute_time else "compute"
    intensity = (operations / moved).to("flop/byte")
    return RooflineResult(
        hardware_name=hardware.name,
        hardware_ref=_hardware_registry_ref(hardware),
        precision=precision.lower(),
        operations=operations.to("flop"),
        base_bytes_moved=base_bytes_moved.to("byte"),
        bytes_moved=moved,
        reuse=reuse,
        compute_scale=compute_scale,
        bandwidth_scale=bandwidth_scale,
        arithmetic_intensity=intensity,
        ridge_point=(peak / bandwidth).to("flop/byte"),
        compute_time=compute_time,
        memory_time=memory_time,
        latency=latency,
        bottleneck=bottleneck,
        attainable_rate=(operations / latency.to("s")).to("flop/s"),
    )


@dataclass(frozen=True)
class TileResult:
    dimensions: tuple[int, int, int]
    padded_dimensions: tuple[int, int, int]
    tile: tuple[int, int, int]
    precision: str
    fused: bool
    scratchpad_required: Quantity
    scratchpad_capacity: Quantity
    fits: bool
    operations: Quantity
    dram_bytes: Quantity
    local_bytes: Quantity
    movement_energy: Quantity
    movement_time: Quantity | None


def analyze_tile_mapping(
    *,
    hardware: HardwareNode,
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    precision: str,
    fused: bool,
    scratchpad_capacity: Quantity | None = None,
) -> TileResult:
    """Count blocked-GEMM traffic and test a tile against local capacity.

    A and B tiles are loaded once per output tile; the C tile remains local
    across K tiles.  An unfused following elementwise operation writes and
    rereads one C-sized intermediate.  A non-fitting tile is returned as a
    failed mapping with no invented spill-time multiplier.
    """

    for value, name in (
        (m, "m"), (n, "n"), (k, "k"),
        (tile_m, "tile_m"), (tile_n, "tile_n"), (tile_k, "tile_k"),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _, element_bytes = resolve_precision(precision)
    capacity = scratchpad_capacity or hardware.memory.sram_capacity
    if capacity is None:
        capacity = hardware.memory.shared_memory_per_sm
    if capacity is None:
        raise ValueError(f"{hardware.name} has no registry-backed local-memory capacity")

    pm, pn, pk = (
        ceil(m / tile_m) * tile_m,
        ceil(n / tile_n) * tile_n,
        ceil(k / tile_k) * tile_k,
    )
    mt, nt, kt = pm // tile_m, pn // tile_n, pk // tile_k
    scratch_elements = tile_m * tile_k + tile_k * tile_n + tile_m * tile_n
    required = (scratch_elements * element_bytes).to("byte")
    fits = required <= capacity

    input_elements = mt * nt * kt * (tile_m * tile_k + tile_k * tile_n)
    output_elements = pm * pn
    intermediate_elements = 0 if fused else 2 * output_elements
    dram_bytes = ((input_elements + output_elements + intermediate_elements) * element_bytes).to("byte")
    # Two operand reads per multiply-accumulate plus the final output write.
    local_bytes = ((2 * pm * pn * pk + output_elements) * element_bytes).to("byte")
    movement_energy = (
        dram_bytes * Hardware.Tech.Memory.DRAM.energy_per_byte
        + local_bytes * Hardware.Tech.Movement.L1.energy_per_byte
    ).to("mJ")
    movement_time = (dram_bytes / hardware.memory.bandwidth).to("ms") if fits else None
    return TileResult(
        dimensions=(m, n, k),
        padded_dimensions=(pm, pn, pk),
        tile=(tile_m, tile_n, tile_k),
        precision=precision.lower(),
        fused=fused,
        scratchpad_required=required,
        scratchpad_capacity=capacity.to("byte"),
        fits=fits,
        operations=Q_(2 * pm * pn * pk, "flop"),
        dram_bytes=dram_bytes,
        local_bytes=local_bytes,
        movement_energy=movement_energy,
        movement_time=movement_time,
    )


@dataclass(frozen=True)
class ExecutionContract:
    """Illustrative kernel contract; it is not a hardware-registry claim."""

    label: str
    supported_operations: frozenset[str]
    supported_precisions: frozenset[str]
    shape_multiple: int
    padding_allowed: bool = True

    def __post_init__(self) -> None:
        if self.shape_multiple <= 0:
            raise ValueError("shape_multiple must be positive")


@dataclass(frozen=True)
class ExecutionPathResult:
    path: Literal["native", "padded", "fallback"]
    reason: str
    hardware_name: str
    hardware_ref: str | None
    fallback_hardware_name: str
    fallback_hardware_ref: str | None
    contract: ExecutionContract
    operation: str
    precision: str
    original_dimensions: tuple[int, int, int]
    executed_dimensions: tuple[int, int, int]
    operations: Quantity
    bytes_moved: Quantity
    extra_operations: Quantity
    latency: Quantity
    quality_observation: str | None


def analyze_execution_path(
    *,
    hardware: HardwareNode,
    fallback_hardware: HardwareNode,
    contract: ExecutionContract,
    operation: str,
    precision: str,
    m: int,
    n: int,
    k: int,
    quality_observation: str | None = None,
) -> ExecutionPathResult:
    """Choose a native, exactly padded, or explicit fallback execution path."""

    if min(m, n, k) <= 0:
        raise ValueError("m, n, and k must be positive")
    precision_key, element_bytes = resolve_precision(precision)
    supported = (
        operation in contract.supported_operations
        and precision_key in contract.supported_precisions
    )
    aligned = all(d % contract.shape_multiple == 0 for d in (m, n, k))
    if supported and aligned:
        path = "native"
        target = hardware
        dims = (m, n, k)
        reason = "operation, precision, and shape satisfy the scenario contract"
    elif supported and contract.padding_allowed:
        path = "padded"
        target = hardware
        dims = tuple(ceil(d / contract.shape_multiple) * contract.shape_multiple for d in (m, n, k))
        reason = f"shape padded to multiples of {contract.shape_multiple}"
    else:
        path = "fallback"
        target = fallback_hardware
        dims = (m, n, k)
        reason = "operation or precision is unsupported by the scenario contract"

    em, en, ek = dims
    operations = Q_(2 * em * en * ek, "flop")
    original_ops = Q_(2 * m * n * k, "flop")
    bytes_moved = ((em * ek + ek * en + em * en) * element_bytes).to("byte")
    roofline = analyze_roofline(
        hardware=target,
        operations=operations,
        base_bytes_moved=bytes_moved,
        precision=precision_key,
    )
    return ExecutionPathResult(
        path=path,
        reason=reason,
        hardware_name=target.name,
        hardware_ref=_hardware_registry_ref(target),
        fallback_hardware_name=fallback_hardware.name,
        fallback_hardware_ref=_hardware_registry_ref(fallback_hardware),
        contract=contract,
        operation=operation,
        precision=precision_key,
        original_dimensions=(m, n, k),
        executed_dimensions=dims,
        operations=operations,
        bytes_moved=bytes_moved,
        extra_operations=(operations - original_ops).to("flop"),
        latency=roofline.latency,
        quality_observation=quality_observation,
    )


@dataclass(frozen=True)
class ApplicationPathResult:
    host_time: Quantity
    transfer_time: Quantity
    launch_time: Quantity
    kernel_time: Quantity
    postprocess_time: Quantity
    transfer_bytes: Quantity
    transfer_fixed_latency: Quantity
    launches: int
    total_time: Quantity


def compose_application_path(
    *,
    hardware: HardwareNode,
    kernel_time: Quantity,
    host_time: Quantity,
    transfer_bytes: Quantity,
    transfer_fixed_latency: Quantity,
    launches: int,
    postprocess_time: Quantity,
) -> ApplicationPathResult:
    """Compose preserved host, transfer, launch, kernel, and postprocess stages."""

    if launches < 0:
        raise ValueError("launches must be nonnegative")
    for value, name in (
        (kernel_time, "kernel_time"),
        (host_time, "host_time"),
        (transfer_bytes, "transfer_bytes"),
        (transfer_fixed_latency, "transfer_fixed_latency"),
        (postprocess_time, "postprocess_time"),
    ):
        _nonnegative_quantity(value, name)
    if transfer_bytes.to("byte").magnitude == 0:
        # Some single-machine device tracks keep input resident in unified or
        # on-device memory.  Their host-transfer mechanism is inapplicable.
        transfer = transfer_fixed_latency.to("ms")
    else:
        if hardware.interconnect is None:
            raise ValueError(f"{hardware.name} has no registry-backed host interconnect")
        transfer = (
            transfer_bytes / hardware.interconnect.bandwidth_per_direction
            + transfer_fixed_latency
        ).to("ms")
    launch = (launches * hardware.dispatch_tax).to("ms")
    total = (host_time + transfer + launch + kernel_time + postprocess_time).to("ms")
    return ApplicationPathResult(
        host_time=host_time.to("ms"),
        transfer_time=transfer,
        launch_time=launch,
        kernel_time=kernel_time.to("ms"),
        postprocess_time=postprocess_time.to("ms"),
        transfer_bytes=transfer_bytes.to("byte"),
        transfer_fixed_latency=transfer_fixed_latency.to("ms"),
        launches=launches,
        total_time=total,
    )


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    hardware: HardwareNode
    contract: ExecutionContract
    hourly_operating_cost: Quantity


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    hardware_name: str
    execution_path: str
    application: ApplicationPathResult
    latency: Quantity
    accelerator_energy: Quantity
    operating_cost: Quantity
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class RankingResult:
    objective: Objective
    candidates: tuple[Candidate, ...]
    fallback_hardware_name: str
    fallback_hardware_ref: str | None
    operation: str
    precision: str
    dimensions: tuple[int, int, int]
    rows: tuple[CandidateResult, ...]
    ranked_feasible: tuple[CandidateResult, ...]
    recommendation: CandidateResult | None
    latency_budget: Quantity
    energy_budget: Quantity
    cost_budget: Quantity


def rank_accelerators(
    *,
    candidates: Sequence[Candidate],
    fallback_hardware: HardwareNode,
    operation: str,
    precision: str,
    m: int,
    n: int,
    k: int,
    host_time: Quantity,
    transfer_bytes: Quantity,
    transfer_fixed_latency: Quantity,
    launches: int,
    postprocess_time: Quantity,
    objective: Objective,
    latency_budget: Quantity,
    energy_budget: Quantity,
    cost_budget: Quantity,
) -> RankingResult:
    """Evaluate every candidate, filter constraints, then sort the objective."""

    if objective not in ("latency", "energy", "cost"):
        raise ValueError("objective must be latency, energy, or cost")
    rows: list[CandidateResult] = []
    for candidate in candidates:
        execution = analyze_execution_path(
            hardware=candidate.hardware,
            fallback_hardware=fallback_hardware,
            contract=candidate.contract,
            operation=operation,
            precision=precision,
            m=m,
            n=n,
            k=k,
        )
        application = compose_application_path(
            hardware=candidate.hardware,
            kernel_time=execution.latency,
            host_time=host_time,
            transfer_bytes=transfer_bytes,
            transfer_fixed_latency=transfer_fixed_latency,
            launches=launches,
            postprocess_time=postprocess_time,
        )
        power_hardware = fallback_hardware if execution.path == "fallback" else candidate.hardware
        power = power_hardware.tdp
        if power is None:
            raise ValueError(f"{power_hardware.name} has no registry-backed TDP")
        accelerator_active_time = (
            application.transfer_time + application.launch_time + application.kernel_time
        ).to("s")
        energy = (power * accelerator_active_time).to("mJ")
        cost = (candidate.hourly_operating_cost * accelerator_active_time).to("dollar")
        violations: list[str] = []
        if execution.path == "fallback":
            violations.append("native execution support")
        if application.total_time > latency_budget:
            violations.append("latency")
        if energy > energy_budget:
            violations.append("energy")
        if cost > cost_budget:
            violations.append("cost")
        rows.append(
            CandidateResult(
                candidate_id=candidate.candidate_id,
                hardware_name=candidate.hardware.name,
                execution_path=execution.path,
                application=application,
                latency=application.total_time,
                accelerator_energy=energy,
                operating_cost=cost,
                feasible=not violations,
                violations=tuple(violations),
            )
        )
    metric = {
        "latency": lambda row: row.latency.to("s").magnitude,
        "energy": lambda row: row.accelerator_energy.to("J").magnitude,
        "cost": lambda row: row.operating_cost.to("dollar").magnitude,
    }[objective]
    feasible = tuple(sorted((row for row in rows if row.feasible), key=lambda row: (metric(row), row.candidate_id)))
    return RankingResult(
        objective=objective,
        candidates=tuple(candidates),
        fallback_hardware_name=fallback_hardware.name,
        fallback_hardware_ref=_hardware_registry_ref(fallback_hardware),
        operation=operation,
        precision=precision.lower(),
        dimensions=(m, n, k),
        rows=tuple(rows),
        ranked_feasible=feasible,
        recommendation=feasible[0] if feasible else None,
        latency_budget=latency_budget,
        energy_budget=energy_budget,
        cost_budget=cost_budget,
    )


@dataclass(frozen=True)
class TrackScenario:
    """Illustrative workload defaults over registry-backed physical hardware."""

    track: Track
    hardware: HardwareNode
    alternative: HardwareNode
    fallback: HardwareNode
    precision: str
    dimensions: tuple[int, int, int]
    tile: tuple[int, int, int]
    contract: ExecutionContract
    local_capacity: Quantity
    host_time: Quantity
    transfer_bytes: Quantity
    transfer_fixed_latency: Quantity
    launches: int
    postprocess_time: Quantity
    primary_hourly_cost: Quantity
    alternative_hourly_cost: Quantity
    latency_budget: Quantity
    energy_budget: Quantity
    cost_budget: Quantity
    assumption_label: str = "illustrative workload and execution-contract assumptions"


@dataclass(frozen=True)
class GemmDemand:
    operations: Quantity
    bytes_moved: Quantity


@dataclass(frozen=True)
class ApplicationComparison:
    baseline: ApplicationPathResult
    accelerated: ApplicationPathResult
    local_speedup: float
    end_to_end_speedup: float


TRACK_SCENARIOS: Mapping[Track, TrackScenario] = {
    "tinyml": TrackScenario(
        "tinyml", Hardware.Tiny.nRF52840, Hardware.Tiny.ESP32_S3, Hardware.Tiny.ESP32_S3,
        "int8", (64, 64, 64), (16, 16, 16),
        ExecutionContract("illustrative MCU SIMD contract", frozenset({"gemm"}), frozenset({"int8"}), 8),
        Hardware.Tiny.nRF52840.memory.sram_capacity,
        Q_(3, "ms"), Q_(0, "byte"), Q_(0, "ms"), 1, Q_(1, "ms"),
        Q_(0.01, "dollar/hour"), Q_(0.01, "dollar/hour"),
        Q_(50, "ms"), Q_(1, "mJ"), Q_(0.001, "dollar"),
    ),
    "mobile": TrackScenario(
        "mobile", Hardware.Mobile.AppleM2, Hardware.Mobile.iPhone15Pro, Hardware.Mobile.iPhone15Pro,
        "fp16", (512, 512, 512), (32, 32, 16),
        ExecutionContract("illustrative mobile NPU contract", frozenset({"gemm"}), frozenset({"fp16", "int8"}), 16),
        Q_(1, "MiB"),  # illustrative local-buffer allocation; registry has unified memory only
        Q_(4, "ms"), Q_(0, "byte"), Q_(0, "ms"), 2, Q_(2, "ms"),
        Q_(0.10, "dollar/hour"), Q_(0.08, "dollar/hour"),
        Q_(25, "ms"), Q_(100, "mJ"), Q_(0.001, "dollar"),
    ),
    "edge": TrackScenario(
        "edge", Hardware.Edge.JetsonAGXOrin, Hardware.Edge.JetsonOrinNX, Hardware.Edge.JetsonOrinNX,
        "int8", (1024, 1024, 1024), (32, 32, 32),
        ExecutionContract("illustrative edge tensor-unit contract", frozenset({"gemm"}), frozenset({"fp16", "int8"}), 16),
        Q_(2, "MiB"),  # illustrative local-buffer allocation; registry has device memory only
        Q_(2, "ms"), Q_(0, "byte"), Q_(0, "ms"), 3, Q_(1, "ms"),
        Q_(1, "dollar/hour"), Q_(0.60, "dollar/hour"),
        Q_(20, "ms"), Q_(1, "kJ"), Q_(0.001, "dollar"),
    ),
    "cloud": TrackScenario(
        "cloud", Hardware.Cloud.H100, Hardware.Cloud.A100, Hardware.Cloud.A100,
        "fp16", (2048, 2048, 2048), (64, 64, 16),
        ExecutionContract("illustrative cloud tensor-core contract", frozenset({"gemm"}), frozenset({"fp16", "tf32", "fp8", "int8"}), 16),
        Hardware.Cloud.H100.memory.shared_memory_per_sm,
        Q_(3, "ms"), Q_(64, "MB"), Q_(0.05, "ms"), 6, Q_(1, "ms"),
        Q_(12, "dollar/hour"), Q_(5, "dollar/hour"),
        Q_(10, "ms"), Q_(10, "J"), Q_(0.01, "dollar"),
    ),
}


def get_track_scenario(track: Track) -> TrackScenario:
    try:
        return TRACK_SCENARIOS[track]
    except KeyError as exc:
        raise ValueError(f"unknown track {track!r}; choose tinyml, mobile, edge, or cloud") from exc


def scenario_gemm_demand(track: Track) -> GemmDemand:
    scenario = get_track_scenario(track)
    m, n, k = scenario.dimensions
    _, element_bytes = resolve_precision(scenario.precision)
    return GemmDemand(
        operations=Q_(2 * m * n * k, "flop"),
        bytes_moved=((m * k + k * n + m * n) * element_bytes).to("byte"),
    )


def scenario_tile_choices(track: Track) -> Mapping[str, tuple[int, int, int]]:
    scenario = get_track_scenario(track)
    tm, tn, tk = scenario.tile
    m, n, k = scenario.dimensions
    return {
        "compact": (max(1, tm // 2), max(1, tn // 2), max(1, tk // 2)),
        "baseline": scenario.tile,
        "oversized": (max(tm * 64, m * 2), max(tn * 64, n * 2), max(tk * 64, k * 2)),
    }


def scenario_execution_dimensions(track: Track, *, aligned: bool) -> tuple[int, int, int]:
    scenario = get_track_scenario(track)
    if aligned:
        return scenario.dimensions
    m, n, k = scenario.dimensions
    return (m + 1, n, k)


def scenario_candidates(track: Track) -> tuple[Candidate, Candidate]:
    scenario = get_track_scenario(track)
    return (
        Candidate("primary", scenario.hardware, scenario.contract, scenario.primary_hourly_cost),
        Candidate("alternative", scenario.alternative, scenario.contract, scenario.alternative_hourly_cost),
    )


def scenario_budgets(track: Track, *, scale: float = 1.0) -> tuple[Quantity, Quantity, Quantity]:
    _positive(scale, "scale")
    scenario = get_track_scenario(track)
    return (
        scenario.latency_budget * scale,
        scenario.energy_budget * scale,
        scenario.cost_budget * scale,
    )


def compare_application_speedup(
    *,
    hardware: HardwareNode,
    baseline_kernel_time: Quantity,
    local_speedup: float,
    host_time: Quantity,
    transfer_bytes: Quantity,
    transfer_fixed_latency: Quantity,
    launches: int,
    postprocess_time: Quantity,
) -> ApplicationComparison:
    _positive(local_speedup, "local_speedup")
    common = dict(
        hardware=hardware,
        host_time=host_time,
        transfer_bytes=transfer_bytes,
        transfer_fixed_latency=transfer_fixed_latency,
        launches=launches,
        postprocess_time=postprocess_time,
    )
    baseline = compose_application_path(kernel_time=baseline_kernel_time, **common)
    accelerated = compose_application_path(
        kernel_time=baseline_kernel_time / local_speedup,
        **common,
    )
    return ApplicationComparison(
        baseline=baseline,
        accelerated=accelerated,
        local_speedup=local_speedup,
        end_to_end_speedup=(baseline.total_time / accelerated.total_time).to_base_units().magnitude,
    )


_EXPERIMENTS = {
    "roofline": analyze_roofline,
    "tile_mapping": analyze_tile_mapping,
    "execution_path": analyze_execution_path,
    "application_path": compose_application_path,
    "application_comparison": compare_application_speedup,
    "accelerator_ranking": rank_accelerators,
}


def make_replay_packet(*, track: Track, experiment: str, inputs: Mapping[str, object]) -> dict:
    """Create a JSON-compatible, method-specific experiment replay packet."""

    get_track_scenario(track)
    if experiment not in _EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}")
    return {
        "model_key": f"{MODEL_ID}.{experiment}",
        "track_id": track,
        "inputs": to_jsonable(inputs),
    }


def _restore_hardware(payload: Mapping[str, object]) -> HardwareNode:
    ref = payload.get("registry_ref")
    if not isinstance(ref, str):
        raise ValueError("replay hardware requires a registry_ref")
    parts = ref.split(".")
    if len(parts) != 3 or parts[0] != "Hardware" or parts[1] not in {
        "Tiny", "Mobile", "Edge", "Cloud", "Workstation"
    }:
        raise ValueError(f"invalid hardware registry_ref {ref!r}")
    try:
        return getattr(getattr(Hardware, parts[1]), parts[2])
    except AttributeError as exc:
        raise ValueError(f"unknown hardware registry_ref {ref!r}") from exc


def _restore_value(value):
    if isinstance(value, dict):
        if set(value) == {"magnitude", "unit"}:
            return Q_(value["magnitude"], value["unit"])
        if "registry_ref" in value and "hardware_name" in value:
            return _restore_hardware(value)
        return {key: _restore_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_value(item) for item in value]
    return value


def _restore_contract(payload: Mapping[str, object]) -> ExecutionContract:
    return ExecutionContract(
        label=str(payload["label"]),
        supported_operations=frozenset(payload["supported_operations"]),
        supported_precisions=frozenset(payload["supported_precisions"]),
        shape_multiple=int(payload["shape_multiple"]),
        padding_allowed=bool(payload["padding_allowed"]),
    )


def replay_experiment(packet: Mapping[str, object]):
    """Restore a replay packet and run the named Chapter 11 experiment."""

    track = packet.get("track_id")
    if not isinstance(track, str):
        raise ValueError("replay packet requires track_id")
    get_track_scenario(track)
    model_key = packet.get("model_key")
    prefix = f"{MODEL_ID}."
    if not isinstance(model_key, str) or not model_key.startswith(prefix):
        raise ValueError(f"model_key must start with {prefix!r}")
    experiment = model_key.removeprefix(prefix)
    if experiment not in _EXPERIMENTS:
        raise ValueError(f"unknown experiment model_key {model_key!r}")
    encoded_inputs = packet.get("inputs")
    if not isinstance(encoded_inputs, Mapping):
        raise ValueError("replay packet requires an inputs mapping")
    inputs = _restore_value(encoded_inputs)
    if experiment == "execution_path":
        inputs["contract"] = _restore_contract(inputs["contract"])
    elif experiment == "accelerator_ranking":
        inputs["candidates"] = tuple(
            Candidate(
                candidate_id=item["candidate_id"],
                hardware=item["hardware"],
                contract=_restore_contract(item["contract"]),
                hourly_operating_cost=item["hourly_operating_cost"],
            )
            for item in inputs["candidates"]
        )
    return _EXPERIMENTS[experiment](**inputs)


__all__ = [
    "ApplicationComparison", "ApplicationPathResult", "Candidate", "CandidateResult", "ExecutionContract",
    "ExecutionPathResult", "RankingResult", "RooflineResult", "TRACK_SCENARIOS",
    "GemmDemand", "MODEL_ID", "TileResult", "TrackScenario", "analyze_execution_path", "analyze_roofline",
    "analyze_tile_mapping", "compare_application_speedup", "compose_application_path", "get_track_scenario",
    "make_replay_packet", "rank_accelerators", "replay_experiment", "to_jsonable",
    "scenario_budgets", "scenario_candidates", "scenario_execution_dimensions",
    "scenario_gemm_demand", "scenario_tile_choices",
]
