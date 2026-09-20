"""Chapter 7 framework execution-path experiments.

The module evaluates one explicit chain of tensor operations.  It is a bounded
teaching model, not a compiler or production performance estimator.  Hardware
limits and runtime dispatch costs come from the MLSysIM registries.  Graph
shape, operation mix, compilation work, and support policy are explicitly
illustrative scenario assumptions used consistently across the four tracks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mlsysim import Hardware, Ops
from mlsysim.core.units import BYTES_FP16, Q_, ureg


@dataclass(frozen=True)
class GraphOperation:
    """One operation in the explicit teaching graph."""

    name: str
    flops_per_element: int


@dataclass(frozen=True)
class TrackScenario:
    """Registry-backed execution targets plus bounded support assumptions."""

    display: str
    workload: str
    target: Any
    development_host: Any
    default_elements: int
    supported_operators: frozenset[str]
    supported_shapes: frozenset[str]
    fallback_available: bool


# Explicit small operation graph used by every investigation.  The per-element
# FLOP counts describe the scenario fixture; they are not empirical kernel data.
OPERATION_GRAPH = (
    GraphOperation("center", 1),
    GraphOperation("scale", 1),
    GraphOperation("bias", 1),
    GraphOperation("relu", 1),
    GraphOperation("square", 1),
    GraphOperation("clamp", 2),
)

OPERATOR_FLOPS_PER_ELEMENT = {
    "relu": 1,
    "layer_norm": 5,
    "dynamic_slice": 1,
    "custom_attention": 8,
    "host_callback": 1,
}

# Track differences are concrete: deployment target, development host, tensor
# scale, operator catalog, shape policy, and whether a fallback runtime exists.
# Tensor scales and support sets are illustrative assumptions, not measurements.
TRACKS = {
    "tinyml": TrackScenario(
        display="TinyML",
        workload="A fixed-shape sensor classifier",
        target=Hardware.Tiny.ESP32_S3,
        development_host=Hardware.Cloud.ReferenceCPU,
        default_elements=16_384,
        supported_operators=frozenset({"relu"}),
        supported_shapes=frozenset({"static"}),
        fallback_available=False,
    ),
    "mobile": TrackScenario(
        display="Mobile",
        workload="An on-device interactive model",
        target=Hardware.Mobile.Pixel8,
        development_host=Hardware.Workstation.MacBookM3Max,
        default_elements=524_288,
        supported_operators=frozenset({"relu", "layer_norm"}),
        supported_shapes=frozenset({"static", "bounded"}),
        fallback_available=True,
    ),
    "edge": TrackScenario(
        display="Edge",
        workload="A local inspection model",
        target=Hardware.Edge.JetsonOrinNano,
        development_host=Hardware.Edge.GenericServer,
        default_elements=2_097_152,
        supported_operators=frozenset({"relu", "layer_norm", "dynamic_slice"}),
        supported_shapes=frozenset({"static", "bounded", "dynamic"}),
        fallback_available=True,
    ),
    "cloud": TrackScenario(
        display="Cloud",
        workload="A compiled single-accelerator endpoint",
        target=Hardware.Cloud.T4,
        development_host=Hardware.Cloud.ReferenceCPU,
        default_elements=8_388_608,
        supported_operators=frozenset({"relu", "layer_norm", "dynamic_slice", "custom_attention"}),
        supported_shapes=frozenset({"static", "bounded", "dynamic"}),
        fallback_available=True,
    ),
}

SHAPE_MODES = frozenset({"static", "bounded", "dynamic"})
RECOMPUTE_POLICIES = frozenset({"retain_all", "alternate", "input_only"})
COMPILATION_ANALYSIS_PASSES = 25
MODEL_KEY = "v1_07_experiments"
SCENARIO_NOTE = (
    "Operation mix, tensor scales, compilation passes, and runtime support are "
    "illustrative scenario assumptions, not measured framework benchmarks."
)


def _track(track_id: str) -> TrackScenario:
    try:
        return TRACKS[track_id]
    except KeyError as exc:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}") from exc


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _elements(track: TrackScenario, tensor_elements: int | None) -> int:
    if tensor_elements is None:
        return track.default_elements
    return _positive_int(tensor_elements, "tensor_elements")


def _as_us(duration: Any) -> float:
    return float(duration.to(ureg.microsecond).magnitude)


def _as_ms(duration: Any) -> float:
    return float(duration.to(ureg.millisecond).magnitude)


def _as_mb(volume: Any) -> float:
    return float(volume.to(ureg.megabyte).magnitude)


def _graph_flops(tensor_elements: int) -> Any:
    return Q_(
        tensor_elements * sum(op.flops_per_element for op in OPERATION_GRAPH),
        "flop",
    )


def _tensor_bytes(tensor_elements: int) -> Any:
    return (tensor_elements * BYTES_FP16).to(ureg.byte)


def _roofline_time(flops: Any, traffic: Any, hardware: Any) -> Any:
    compute = (flops / hardware.compute.peak_flops).to(ureg.second)
    movement = (traffic / hardware.memory.bandwidth).to(ureg.second)
    return max(compute, movement)


def evaluate_dispatch(
    track_id: str,
    dispatches: int,
    *,
    tensor_elements: int | None = None,
) -> dict[str, Any]:
    """Vary launch granularity while keeping arithmetic and payload fixed."""

    track = _track(track_id)
    dispatches = _positive_int(dispatches, "dispatches")
    elements = _elements(track, tensor_elements)
    flops = _graph_flops(elements)
    payload = 2 * _tensor_bytes(elements)
    useful_time = _roofline_time(flops, payload, track.target)
    dispatch_time = dispatches * track.target.dispatch_tax
    total_time = useful_time + dispatch_time

    return {
        "model_key": MODEL_KEY,
        "track_id": track_id,
        "track": track.display,
        "workload": track.workload,
        "target": track.target.name,
        "dispatches": dispatches,
        "tensor_elements": elements,
        "flops": float(flops.to("flop").magnitude),
        "payload_mb": _as_mb(payload),
        "useful_time_us": _as_us(useful_time),
        "dispatch_time_us": _as_us(dispatch_time),
        "total_time_us": _as_us(total_time),
        "dispatch_fraction": float(dispatch_time / total_time),
        "inputs": {
            "track_id": track_id,
            "dispatches": dispatches,
            "tensor_elements": elements,
        },
        "scenario_note": SCENARIO_NOTE,
    }


def evaluate_compilation(
    track_id: str,
    executions: int,
    *,
    recompilations: int = 0,
    tensor_elements: int | None = None,
) -> dict[str, Any]:
    """Compare eager and compiled totals over repeated identical execution.

    Compilation setup time uses a shared illustrative compiler pass assumption
    (operations × analysis passes × constant framework dispatch tax) rather than
    measuring execution on a named development host. For TinyML targets, this
    represents ahead-of-time (AOT) host compilation, where recompilations
    reflect offline host rebuilds and redeployments rather than dynamic on-device
    JIT guard failures.
    """

    track = _track(track_id)
    executions = _positive_int(executions, "executions")
    recompilations = _positive_int(recompilations, "recompilations", minimum=0)
    if recompilations >= executions:
        raise ValueError("recompilations must be fewer than executions")
    elements = _elements(track, tensor_elements)
    eager_once = evaluate_dispatch(track_id, len(OPERATION_GRAPH), tensor_elements=elements)
    compiled_once = evaluate_dispatch(track_id, 1, tensor_elements=elements)

    compilation_count = 1 + recompilations
    framework_dispatch = Ops.RuntimeOverheads.PythonDispatch + Ops.RuntimeOverheads.KernelLaunch
    setup_time = len(OPERATION_GRAPH) * COMPILATION_ANALYSIS_PASSES * framework_dispatch
    eager_total = executions * Q_(eager_once["total_time_us"], "microsecond")
    compiled_run_total = executions * Q_(compiled_once["total_time_us"], "microsecond")
    compiled_total = compilation_count * setup_time + compiled_run_total
    saved_per_execution = Q_(
        eager_once["total_time_us"] - compiled_once["total_time_us"],
        "microsecond",
    )
    if saved_per_execution.magnitude > 0:
        break_even = math.ceil(float((compilation_count * setup_time / saved_per_execution).to("")))
    else:
        break_even = None

    return {
        "model_key": MODEL_KEY,
        "track_id": track_id,
        "executions": executions,
        "recompilations": recompilations,
        "compilation_count": compilation_count,
        "setup_time_ms": _as_ms(compilation_count * setup_time),
        "eager_total_ms": _as_ms(eager_total),
        "compiled_run_total_ms": _as_ms(compiled_run_total),
        "compiled_total_ms": _as_ms(compiled_total),
        "break_even_executions": break_even,
        "compilation_repaid": compiled_total <= eager_total,
        "inputs": {
            "track_id": track_id,
            "executions": executions,
            "recompilations": recompilations,
            "tensor_elements": elements,
        },
        "scenario_note": SCENARIO_NOTE,
    }


def evaluate_operator_support(
    track_id: str,
    operator: str,
    shape_mode: str,
    *,
    tensor_elements: int | None = None,
) -> dict[str, Any]:
    """Evaluate native execution, explicit fallback, or an unsupported failure."""

    track = _track(track_id)
    if operator not in OPERATOR_FLOPS_PER_ELEMENT:
        raise ValueError(f"operator must be one of {', '.join(OPERATOR_FLOPS_PER_ELEMENT)}")
    if shape_mode not in SHAPE_MODES:
        raise ValueError(f"shape_mode must be one of {', '.join(sorted(SHAPE_MODES))}")
    elements = _elements(track, tensor_elements)
    tensor = _tensor_bytes(elements)
    operator_flops = Q_(elements * OPERATOR_FLOPS_PER_ELEMENT[operator], "flop")
    operator_ok = operator in track.supported_operators
    shape_ok = shape_mode in track.supported_shapes
    native = operator_ok and shape_ok

    if native:
        execution = _roofline_time(operator_flops, 2 * tensor, track.target) + track.target.dispatch_tax
        status = "native"
        copy_bytes = 0 * ureg.byte
        fallback_time = 0 * ureg.second
        executable = True
    elif track.fallback_available:
        # The fallback materializes one input and one output through the target
        # memory path, then executes through a generic target CPU path.  The
        # reference CPU caps that illustrative path rather than discounting
        # target performance by an invented percentage.
        copy_bytes = 2 * tensor
        copy_time = (copy_bytes / track.target.memory.bandwidth).to(ureg.second)
        fallback_flops = min(
            track.target.compute.peak_flops,
            Hardware.Cloud.ReferenceCPU.compute.peak_flops,
        )
        fallback_compute = (operator_flops / fallback_flops).to(ureg.second)
        fallback_movement = (2 * tensor / track.target.memory.bandwidth).to(ureg.second)
        fallback_work = max(fallback_compute, fallback_movement)
        fallback_time = copy_time + fallback_work + track.target.dispatch_tax
        execution = fallback_time
        status = "fallback"
        executable = True
    else:
        copy_bytes = 0 * ureg.byte
        fallback_time = 0 * ureg.second
        execution = None
        status = "unsupported"
        executable = False

    return {
        "model_key": MODEL_KEY,
        "track_id": track_id,
        "operator": operator,
        "shape_mode": shape_mode,
        "operator_supported": operator_ok,
        "shape_supported": shape_ok,
        "status": status,
        "executable": executable,
        "execution_target": (
            track.target.name
            if native
            else f"Illustrative portable fallback on {track.target.name}"
            if executable
            else None
        ),
        "fallback_copy_mb": _as_mb(copy_bytes),
        "fallback_time_us": _as_us(fallback_time),
        "total_time_us": _as_us(execution) if execution is not None else None,
        "inputs": {
            "track_id": track_id,
            "operator": operator,
            "shape_mode": shape_mode,
            "tensor_elements": elements,
        },
        "scenario_note": SCENARIO_NOTE,
    }


def evaluate_fusion(
    track_id: str,
    *,
    graph_breaks: int = 0,
    unsupported_operator: str | None = None,
    shape_mode: str = "static",
    tensor_elements: int | None = None,
) -> dict[str, Any]:
    """Count graph launches, boundary traffic, and fallback copies."""

    track = _track(track_id)
    graph_breaks = _positive_int(graph_breaks, "graph_breaks", minimum=0)
    if graph_breaks >= len(OPERATION_GRAPH):
        raise ValueError("graph_breaks must be smaller than the operation count")
    elements = _elements(track, tensor_elements)
    tensor = _tensor_bytes(elements)
    compiled_regions = graph_breaks + 1
    fallback = None
    fallback_boundaries = 0
    fallback_launches = 0
    fallback_copies = 0 * ureg.byte
    fallback_time = 0 * ureg.second
    native_flops = _graph_flops(elements)
    operation_count = len(OPERATION_GRAPH)
    executable = True

    if unsupported_operator is not None:
        operation_count += 1
        fallback = evaluate_operator_support(
            track_id,
            unsupported_operator,
            shape_mode,
            tensor_elements=elements,
        )
        selected_operator_flops = Q_(elements * OPERATOR_FLOPS_PER_ELEMENT[unsupported_operator], "flop")
        if fallback["status"] == "native":
            native_flops += selected_operator_flops
        else:
            fallback_boundaries = 2
            fallback_launches = 1 if fallback["executable"] else 0
            fallback_copies = 2 * tensor if fallback["executable"] else 0 * ureg.byte
            fallback_time = Q_(fallback["fallback_time_us"], "microsecond")
            executable = fallback["executable"]
            # The inserted fallback sits inside the chain, so it separates the
            # native work into regions before and after the operator.
            if executable:
                compiled_regions += 1

    launches = compiled_regions + fallback_launches if executable else 0
    graph_traffic = 2 * compiled_regions * tensor
    total_traffic = graph_traffic + fallback_copies
    graph_time = _roofline_time(native_flops, graph_traffic, track.target)
    total_time = graph_time + compiled_regions * track.target.dispatch_tax + fallback_time if executable else None
    unfused_traffic = 2 * operation_count * tensor

    return {
        "model_key": MODEL_KEY,
        "track_id": track_id,
        "operation_count": operation_count,
        "graph_breaks": graph_breaks,
        "boundary_count": graph_breaks + fallback_boundaries,
        "launches": launches,
        "graph_traffic_mb": _as_mb(graph_traffic),
        "fallback_copy_mb": _as_mb(fallback_copies),
        "total_traffic_mb": _as_mb(total_traffic),
        "unfused_traffic_mb": _as_mb(unfused_traffic),
        "eliminated_traffic_mb": _as_mb(unfused_traffic - total_traffic),
        "executable": executable,
        "total_time_us": _as_us(total_time) if total_time is not None else None,
        "fallback": fallback,
        "inputs": {
            "track_id": track_id,
            "graph_breaks": graph_breaks,
            "unsupported_operator": unsupported_operator,
            "shape_mode": shape_mode,
            "tensor_elements": elements,
        },
        "scenario_note": SCENARIO_NOTE,
    }


def evaluate_recomputation(
    track_id: str,
    policy: str,
    *,
    tensor_elements: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Trade retained activation bytes for repeated forward operations."""

    track = _track(track_id)
    if policy not in RECOMPUTE_POLICIES:
        raise ValueError(f"policy must be one of {', '.join(sorted(RECOMPUTE_POLICIES))}")
    batch_size = _positive_int(batch_size, "batch_size")
    elements = _elements(track, tensor_elements) * batch_size
    tensor = _tensor_bytes(elements)
    forward_flops = _graph_flops(elements)

    if policy == "retain_all":
        retained_indices = tuple(range(len(OPERATION_GRAPH)))
        recomputed_indices: tuple[int, ...] = ()
    elif policy == "alternate":
        retained_indices = tuple(range(0, len(OPERATION_GRAPH), 2))
        recomputed_indices = tuple(range(1, len(OPERATION_GRAPH), 2))
    else:
        retained_indices = (0,)
        recomputed_indices = tuple(range(len(OPERATION_GRAPH)))

    repeated_flops = Q_(
        elements * sum(OPERATION_GRAPH[index].flops_per_element for index in recomputed_indices),
        "flop",
    )
    retained_bytes = len(retained_indices) * tensor
    # One forward plus an explicit two-forward-equivalent backward pass.  The
    # recomputed operations add only their repeated forward arithmetic/traffic.
    training_flops = 3 * forward_flops + repeated_flops
    base_traffic = 3 * 2 * len(OPERATION_GRAPH) * tensor
    repeated_traffic = 2 * len(recomputed_indices) * tensor
    total_traffic = base_traffic + repeated_traffic
    executed_operations = 3 * len(OPERATION_GRAPH) + len(recomputed_indices)
    host = track.development_host if track_id in {"tinyml", "mobile", "edge"} else track.target
    step_time = _roofline_time(training_flops, total_traffic, host) + executed_operations * host.dispatch_tax
    memory_capacity = host.memory.capacity

    return {
        "model_key": MODEL_KEY,
        "track_id": track_id,
        "policy": policy,
        "training_host": host.name,
        "batch_size": batch_size,
        "retained_activation_count": len(retained_indices),
        "recomputed_operation_count": len(recomputed_indices),
        "retained_activation_mb": _as_mb(retained_bytes),
        "repeated_flops": float(repeated_flops.to("flop").magnitude),
        "total_step_flops": float(training_flops.to("flop").magnitude),
        "total_traffic_mb": _as_mb(total_traffic),
        "step_time_ms": _as_ms(step_time),
        "memory_capacity_mb": _as_mb(memory_capacity),
        "memory_feasible": retained_bytes <= memory_capacity,
        "inputs": {
            "track_id": track_id,
            "policy": policy,
            "tensor_elements": elements // batch_size,
            "batch_size": batch_size,
        },
        "scenario_note": SCENARIO_NOTE,
    }


def compare_dispatch(
    track_id: str,
    baseline_dispatches: int,
    intervention_dispatches: int,
    *,
    tensor_elements: int | None = None,
) -> dict[str, Any]:
    """Return an identical-work baseline/intervention dispatch comparison."""

    baseline = evaluate_dispatch(track_id, baseline_dispatches, tensor_elements=tensor_elements)
    intervention = evaluate_dispatch(track_id, intervention_dispatches, tensor_elements=tensor_elements)
    return {
        "model_key": MODEL_KEY,
        "baseline": baseline,
        "intervention": intervention,
        "speedup": baseline["total_time_us"] / intervention["total_time_us"],
        "saved_time_us": baseline["total_time_us"] - intervention["total_time_us"],
        "inputs": {
            "track_id": track_id,
            "baseline_dispatches": baseline_dispatches,
            "intervention_dispatches": intervention_dispatches,
            "tensor_elements": baseline["tensor_elements"],
        },
    }


def replay(model_key: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one Chapter 7 result from its exact JSON-safe inputs.

    Chapter 7 evidence arms share one stable model identifier.  Dispatch is
    therefore based on the complete evaluator input schema, rather than on
    partial key or value inference.
    """

    if model_key != MODEL_KEY:
        raise ValueError(f"unknown Chapter 7 model_key {model_key!r}")
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a mapping")

    evaluators = {
        frozenset({"track_id", "dispatches", "tensor_elements"}): evaluate_dispatch,
        frozenset(
            {"track_id", "executions", "recompilations", "tensor_elements"}
        ): evaluate_compilation,
        frozenset(
            {
                "track_id",
                "graph_breaks",
                "unsupported_operator",
                "shape_mode",
                "tensor_elements",
            }
        ): evaluate_fusion,
        frozenset(
            {"track_id", "operator", "shape_mode", "tensor_elements"}
        ): evaluate_operator_support,
        frozenset(
            {"track_id", "policy", "tensor_elements", "batch_size"}
        ): evaluate_recomputation,
        frozenset(
            {
                "track_id",
                "baseline_dispatches",
                "intervention_dispatches",
                "tensor_elements",
            }
        ): compare_dispatch,
    }
    input_keys = frozenset(inputs)
    evaluator = evaluators.get(input_keys)
    if evaluator is None:
        raise ValueError(
            "inputs do not match a known Chapter 7 evaluator schema: "
            f"{sorted(input_keys)}"
        )
    return evaluator(**dict(inputs))
