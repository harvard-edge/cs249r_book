"""Exact dense-network accounting and small numerical experiments for V1-05.

The track profiles in :data:`TRACKS` are illustrative teaching scenarios. They
describe generic memory envelopes and small multilayer perceptrons; they are not
measurements of branded devices. The accounting is analytical and intentionally
does not predict latency, energy, throughput, or model quality.

The forward-work convention is explicit: one dense weight application is one
multiply-accumulate (MAC), converted at the fixed convention of two FLOPs per
MAC. Bias additions and nonlinearities are reported separately rather than
silently folded into that convention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mlsysim.core.units import Q_, resolve_precision, ureg


_PHASES = ("inference", "training")
_NUMERIC_FORMATS = ("fp32", "fp16", "int8")
_OPTIMIZER_SLOTS = {
    "none": 0,
    "sgd": 0,
    "momentum": 1,
    "adam": 2,
}
EVALUATE_MODEL_KEY = "v1_05.evaluate_track"
NUMERICAL_MODEL_KEY = "v1_05.run_numerical_case"

# Conventional dense-arithmetic accounting: one multiply and one add per MAC.
# Bias additions and nonlinearities are exposed separately below.
FLOPS_PER_MAC = 2


# Generic, deliberately small teaching envelopes. Each memory limit sits above
# the default FP32 inference state and below its FP32 Adam training state, which
# makes the inference/training contrast reachable in every track.
TRACKS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "display": "TinyML",
        "scenario": "A local classifier with a small working-memory envelope",
        "layer_dims": (64, 32, 8),
        "batch_size": 1,
        "memory_limit": Q_(16, "KiB"),
        "effective_compute_rate": Q_(50, "MFLOP / second"),
        "effective_bandwidth": Q_(100, "MB / second"),
        "launch_overhead": Q_(0.05, "millisecond"),
    },
    "mobile": {
        "display": "Mobile",
        "scenario": "An interactive on-device dense prediction stage",
        "layer_dims": (256, 128, 32),
        "batch_size": 4,
        "memory_limit": Q_(256, "KiB"),
        "effective_compute_rate": Q_(20, "GFLOP / second"),
        "effective_bandwidth": Q_(20, "GB / second"),
        "launch_overhead": Q_(0.10, "millisecond"),
    },
    "edge": {
        "display": "Edge",
        "scenario": "A sensor-side dense prediction stage over several samples",
        "layer_dims": (512, 256, 64),
        "batch_size": 8,
        "memory_limit": Q_(1, "MiB"),
        "effective_compute_rate": Q_(100, "GFLOP / second"),
        "effective_bandwidth": Q_(50, "GB / second"),
        "launch_overhead": Q_(0.05, "millisecond"),
    },
    "cloud": {
        "display": "Cloud",
        "scenario": "A shared dense prediction stage using a larger batch",
        "layer_dims": (1024, 512, 128),
        "batch_size": 32,
        "memory_limit": Q_(5, "MiB"),
        "effective_compute_rate": Q_(1, "TFLOP / second"),
        "effective_bandwidth": Q_(200, "GB / second"),
        "launch_overhead": Q_(0.02, "millisecond"),
    },
}

TRACK_PROVENANCE = (
    "Illustrative generic workload and memory envelope; not an empirical "
    "measurement or a branded-hardware specification."
)


# Small deterministic arrays used to expose representation and arithmetic
# behavior. They are numerical test cases, not learned-quality measurements.
NUMERICAL_CASES: dict[str, dict[str, Any]] = {
    "rounding": {
        "inputs": (0.1, 0.2),
        "weights": (0.3, 0.4),
        "bias": 0.05,
        "description": "Finite decimal values that low-precision floats round.",
    },
    "float_overflow": {
        "inputs": (400.0, 400.0),
        "weights": (200.0, 200.0),
        "bias": 0.0,
        "description": "Representable FP16 operands whose dot product exceeds FP16 range.",
    },
    "integer_clipping": {
        "inputs": (200.0, 200.0),
        "weights": (1.0, 1.0),
        "bias": 0.0,
        "description": "Integer operands outside the unit-scale INT8 representable range.",
    },
}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _layer_dimensions(layer_dims: Sequence[int]) -> tuple[int, ...]:
    if isinstance(layer_dims, (str, bytes)) or len(layer_dims) < 2:
        raise ValueError("layer_dims must contain an input and at least one output dimension")
    return tuple(_positive_int(value, f"layer_dims[{index}]") for index, value in enumerate(layer_dims))


def _memory_limit(value: Any):
    try:
        limit = value.to(ureg.byte)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("memory_limit must be a Pint byte quantity") from exc
    if not math.isfinite(float(limit.magnitude)) or limit.magnitude <= 0:
        raise ValueError("memory_limit must be positive and finite")
    return limit


def _positive_quantity(value: Any, unit: Any, name: str):
    try:
        quantity = value.to(unit)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a Pint quantity compatible with {unit}") from exc
    if not math.isfinite(float(quantity.magnitude)) or quantity.magnitude <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return quantity


def analyze_network(
    layer_dims: Sequence[int],
    *,
    batch_size: int = 1,
    precision: str = "fp32",
    phase: str = "inference",
    optimizer: str = "adam",
    optimizer_precision: str = "fp32",
) -> dict[str, Any]:
    """Count a dense network's work, state, and minimum forward tensor traffic.

    ``layer_dims=(d0, d1, ..., dn)`` describes dense affine layers from each
    adjacent pair. Every layer includes a bias vector. Inference activation
    memory is an exact two-buffer schedule: the largest adjacent input/output
    pair held at once. Training retains the batch input and every layer output
    for backward propagation. Parameter gradients have the parameter storage width.
    Optimizer memory is the named number of state slots per parameter, stored at
    ``optimizer_precision``. The model does not include allocator, framework,
    kernel-workspace, or nonlinear-activation implementation overheads.

    Minimum forward traffic counts, for each layer, one read of its input,
    weights, and biases plus one write of its output. It is a tensor-accounting
    lower bound, not a cache or hardware performance model.
    """
    dims = _layer_dimensions(layer_dims)
    batch = _positive_int(batch_size, "batch_size")
    if phase not in _PHASES:
        raise ValueError(f"phase must be one of {', '.join(_PHASES)}")
    if optimizer not in _OPTIMIZER_SLOTS:
        raise ValueError(f"optimizer must be one of {', '.join(_OPTIMIZER_SLOTS)}")
    precision_key, bytes_per_element = resolve_precision(precision)
    if precision_key not in _NUMERIC_FORMATS:
        raise ValueError("precision must be fp32, fp16, or int8 for this experiment")
    optimizer_precision_key, optimizer_bytes_per_element = resolve_precision(optimizer_precision)

    bpe = bytes_per_element.to(ureg.byte).magnitude
    optimizer_bpe = optimizer_bytes_per_element.to(ureg.byte).magnitude
    layers: list[dict[str, Any]] = []
    total_weights = 0
    total_biases = 0
    total_macs = 0
    total_bias_adds = 0
    total_output_elements = 0
    total_traffic_elements = 0

    for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
        weight_parameters = input_dim * output_dim
        bias_parameters = output_dim
        parameters = weight_parameters + bias_parameters
        macs = batch * weight_parameters
        bias_additions = batch * output_dim
        input_elements = batch * input_dim
        output_elements = batch * output_dim
        traffic_elements = input_elements + parameters + output_elements
        layer = {
            "index": index,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "weight_parameters": weight_parameters,
            "bias_parameters": bias_parameters,
            "parameters": parameters,
            "batch_macs": macs,
            "mac_convention_flops": macs * FLOPS_PER_MAC,
            "bias_additions": bias_additions,
            "input_elements": input_elements,
            "output_elements": output_elements,
            "minimum_forward_traffic": Q_(traffic_elements * bpe, "byte"),
        }
        layers.append(layer)
        total_weights += weight_parameters
        total_biases += bias_parameters
        total_macs += macs
        total_bias_adds += bias_additions
        total_output_elements += output_elements
        total_traffic_elements += traffic_elements

    parameters = total_weights + total_biases
    inference_live_activation_elements = batch * max(
        input_dim + output_dim for input_dim, output_dim in zip(dims[:-1], dims[1:])
    )
    retained_training_activation_elements = batch * sum(dims)
    activation_elements = (
        inference_live_activation_elements
        if phase == "inference"
        else retained_training_activation_elements
    )

    weight_memory = Q_(parameters * bpe, "byte")
    activation_memory = Q_(activation_elements * bpe, "byte")
    gradient_memory = Q_(parameters * bpe if phase == "training" else 0, "byte")
    optimizer_slots = _OPTIMIZER_SLOTS[optimizer] if phase == "training" else 0
    optimizer_memory = Q_(parameters * optimizer_slots * optimizer_bpe, "byte")
    total_state_memory = (
        weight_memory + activation_memory + gradient_memory + optimizer_memory
    ).to(ureg.byte)
    minimum_forward_traffic = Q_(total_traffic_elements * bpe, "byte")
    activation_forward_traffic = (minimum_forward_traffic - weight_memory).to(ureg.byte)

    return {
        "inputs": {
            "layer_dims": list(dims),
            "batch_size": batch,
            "precision": precision_key,
            "phase": phase,
            "optimizer": optimizer,
            "optimizer_precision": optimizer_precision_key,
        },
        "layer_dims": dims,
        "batch_size": batch,
        "precision": precision_key,
        "phase": phase,
        "optimizer": optimizer,
        "optimizer_precision": optimizer_precision_key,
        "bytes_per_element": int(bpe),
        "optimizer_bytes_per_element": int(optimizer_bpe),
        "flops_per_mac": FLOPS_PER_MAC,
        "layers": layers,
        "weight_parameters": total_weights,
        "bias_parameters": total_biases,
        "parameters": parameters,
        "batch_macs": total_macs,
        "mac_convention_flops": total_macs * FLOPS_PER_MAC,
        "bias_additions": total_bias_adds,
        "nonlinearity_evaluations": total_output_elements,
        "inference_live_activation_elements": inference_live_activation_elements,
        "retained_training_activation_elements": retained_training_activation_elements,
        "activation_elements": activation_elements,
        "weight_memory": weight_memory,
        "activation_memory": activation_memory,
        "gradient_memory": gradient_memory,
        "gradient_memory_kind": "parameter gradients",
        "optimizer_slots": optimizer_slots,
        "optimizer_memory": optimizer_memory,
        "total_state_memory": total_state_memory,
        "weight_forward_traffic": weight_memory,
        "activation_forward_traffic": activation_forward_traffic,
        "minimum_forward_traffic": minimum_forward_traffic,
        "minimum_forward_traffic_per_sample": (minimum_forward_traffic / batch).to(ureg.byte),
        "scope_note": (
            "Analytical dense-tensor counts; excludes nonlinear FLOPs, allocator overhead, "
            "framework state, transient activation-gradient scratch, kernel workspaces, "
            "and cache effects."
        ),
    }


def evaluate_track(
    track_id: str,
    *,
    layer_dims: Sequence[int] | None = None,
    batch_size: int | None = None,
    precision: str = "fp32",
    phase: str = "inference",
    optimizer: str = "adam",
    optimizer_precision: str = "fp32",
    memory_limit: Any | None = None,
    effective_compute_rate: Any | None = None,
    effective_bandwidth: Any | None = None,
    launch_overhead: Any | None = None,
) -> dict[str, Any]:
    """Evaluate exact network state and additive execution time for one track.

    Execution is an illustrative sequential model:
    ``batch time = FLOPs/rate + counted bytes/bandwidth + layers*launch overhead``.
    It excludes overlap, arrivals, queue delay, and latency percentiles.
    """
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    track = TRACKS[track_id]
    limit = _memory_limit(track["memory_limit"] if memory_limit is None else memory_limit)
    compute_rate = _positive_quantity(
        track["effective_compute_rate"] if effective_compute_rate is None else effective_compute_rate,
        ureg.flop / ureg.second,
        "effective_compute_rate",
    )
    bandwidth = _positive_quantity(
        track["effective_bandwidth"] if effective_bandwidth is None else effective_bandwidth,
        ureg.byte / ureg.second,
        "effective_bandwidth",
    )
    overhead = _positive_quantity(
        track["launch_overhead"] if launch_overhead is None else launch_overhead,
        ureg.second,
        "launch_overhead",
    )
    result = analyze_network(
        track["layer_dims"] if layer_dims is None else layer_dims,
        batch_size=track["batch_size"] if batch_size is None else batch_size,
        precision=precision,
        phase=phase,
        optimizer=optimizer,
        optimizer_precision=optimizer_precision,
    )
    required = result["total_state_memory"].to(ureg.byte)
    margin = (limit - required).to(ureg.byte)
    batch_work = Q_(result["mac_convention_flops"], "flop")
    compute_time = (batch_work / compute_rate).to(ureg.millisecond)
    movement_time = (result["minimum_forward_traffic"] / bandwidth).to(ureg.millisecond)
    launch_time = (len(result["layers"]) * overhead).to(ureg.millisecond)
    batch_latency = (compute_time + movement_time + launch_time).to(ureg.millisecond)
    throughput = Q_(result["batch_size"], "count") / batch_latency.to(ureg.second)
    inputs = {
        **result["inputs"],
        "track_id": track_id,
        "memory_limit": {
            "magnitude": float(limit.to(ureg.byte).magnitude),
            "unit": "byte",
        },
        "effective_compute_rate": {
            "magnitude": float(compute_rate.to(ureg.flop / ureg.second).magnitude),
            "unit": "flop / second",
        },
        "effective_bandwidth": {
            "magnitude": float(bandwidth.to(ureg.byte / ureg.second).magnitude),
            "unit": "byte / second",
        },
        "launch_overhead": {
            "magnitude": float(overhead.to(ureg.millisecond).magnitude),
            "unit": "millisecond",
        },
    }
    return {
        **result,
        "inputs": inputs,
        "track_id": track_id,
        "track_display": track["display"],
        "scenario": track["scenario"],
        "memory_limit": limit,
        "memory_margin": margin,
        "fits_memory": required <= limit,
        "violations": [] if required <= limit else ["memory"],
        "compute_time": compute_time,
        "movement_time": movement_time,
        "launch_time": launch_time,
        "effective_compute_rate": compute_rate,
        "effective_bandwidth": bandwidth,
        "launch_overhead": overhead,
        "batch_latency": batch_latency,
        "throughput": throughput.to(1 / ureg.second),
        "amortized_service_time": (batch_latency / result["batch_size"]).to(ureg.millisecond),
        "execution_assumption": (
            "Illustrative sequential additive time from exact batch FLOPs and minimum "
            "tensor traffic; excludes overlap, arrivals, queue delay, and percentiles."
        ),
        "provenance": TRACK_PROVENANCE,
    }


def compare_track(
    track_id: str,
    *,
    baseline: Mapping[str, Any] | None = None,
    baseline_options: Mapping[str, Any] | None = None,
    intervention_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare an intervention with an explicitly preserved baseline.

    When ``baseline`` is supplied it is copied, never recomputed. Otherwise the
    baseline is evaluated from ``baseline_options``. Intervention options start
    from the same option set, so changing one knob does not reset another.
    """
    base_options = dict(baseline_options or {})
    saved_baseline = dict(baseline) if baseline is not None else evaluate_track(track_id, **base_options)
    action_options = {**base_options, **dict(intervention_options or {})}
    action = evaluate_track(track_id, **action_options)
    delta = {
        "parameters": action["parameters"] - saved_baseline["parameters"],
        "batch_macs": action["batch_macs"] - saved_baseline["batch_macs"],
        "mac_convention_flops": (
            action["mac_convention_flops"] - saved_baseline["mac_convention_flops"]
        ),
        "activation_memory": (
            action["activation_memory"] - saved_baseline["activation_memory"]
        ).to(ureg.byte),
        "total_state_memory": (
            action["total_state_memory"] - saved_baseline["total_state_memory"]
        ).to(ureg.byte),
        "minimum_forward_traffic": (
            action["minimum_forward_traffic"] - saved_baseline["minimum_forward_traffic"]
        ).to(ureg.byte),
        "batch_latency": (action["batch_latency"] - saved_baseline["batch_latency"]).to(
            ureg.millisecond
        ),
        "throughput": (action["throughput"] - saved_baseline["throughput"]).to(1 / ureg.second),
        "amortized_service_time": (
            action["amortized_service_time"] - saved_baseline["amortized_service_time"]
        ).to(ureg.millisecond),
    }
    return {"baseline": saved_baseline, "result": action, "delta": delta}


def growth_comparison(track_id: str, target: str, factor: int = 2) -> dict[str, Any]:
    """Compare one input, hidden-width, or batch growth intervention."""
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    if target not in {"input", "hidden", "batch"}:
        raise ValueError("target must be input, hidden, or batch")
    factor = _positive_int(factor, "factor")
    if factor == 1:
        raise ValueError("factor must create a contrast")
    track = TRACKS[track_id]
    dims = tuple(track["layer_dims"])
    options: dict[str, Any] = {}
    if target == "input":
        options["layer_dims"] = (dims[0] * factor, *dims[1:])
    elif target == "hidden":
        options["layer_dims"] = (dims[0], *(value * factor for value in dims[1:-1]), dims[-1])
    else:
        options["batch_size"] = int(track["batch_size"]) * factor
    return compare_track(track_id, intervention_options=options)


def repair_comparison(track_id: str, action: str) -> dict[str, Any]:
    """Apply one resource intervention to the same failed training scenario."""
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    if action not in {"tensor", "batch", "format"}:
        raise ValueError("action must be tensor, batch, or format")
    track = TRACKS[track_id]
    dims = tuple(track["layer_dims"])
    baseline_options = {"batch_size": 8, "phase": "training", "optimizer": "adam"}
    interventions = {
        "tensor": {
            "layer_dims": (dims[0], *(max(1, value // 2) for value in dims[1:-1]), dims[-1])
        },
        "batch": {"batch_size": 1},
        "format": {"precision": "fp16"},
    }
    return compare_track(
        track_id,
        baseline_options=baseline_options,
        intervention_options=interventions[action],
    )


def default_batch(track_id: str) -> int:
    """Return the default intervention batch size contrasting with batch one."""
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    return max(2, int(TRACKS[track_id]["batch_size"]))


def _replay_quantity(value: Any, expected_unit: str, name: str):
    if not isinstance(value, Mapping) or set(value) != {"magnitude", "unit"}:
        raise ValueError(f"{name} must contain exactly magnitude and unit")
    if value["unit"] != expected_unit:
        raise ValueError(f"{name} unit must be {expected_unit}")
    magnitude = value["magnitude"]
    if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)):
        raise ValueError(f"{name} magnitude must be a finite number")
    if not math.isfinite(float(magnitude)):
        raise ValueError(f"{name} magnitude must be a finite number")
    return Q_(magnitude, expected_unit)


def replay(model_key: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one saved V1-05 result from its method-specific evidence inputs.

    The dispatcher recognizes only the two public experiment methods used by
    the notebook. It rejects missing or extra input fields so evidence cannot be
    silently reinterpreted by a different method or a future default change.
    """
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a mapping")
    if model_key == NUMERICAL_MODEL_KEY:
        expected = {"case_id", "numeric_format"}
        if set(inputs) != expected:
            raise ValueError(f"numerical replay inputs must contain exactly {sorted(expected)}")
        return run_numerical_case(str(inputs["case_id"]), str(inputs["numeric_format"]))
    if model_key != EVALUATE_MODEL_KEY:
        raise ValueError(
            f"model_key must be {EVALUATE_MODEL_KEY!r} or {NUMERICAL_MODEL_KEY!r}"
        )

    expected = {
        "track_id",
        "layer_dims",
        "batch_size",
        "precision",
        "phase",
        "optimizer",
        "optimizer_precision",
        "memory_limit",
        "effective_compute_rate",
        "effective_bandwidth",
        "launch_overhead",
    }
    if set(inputs) != expected:
        raise ValueError(f"evaluation replay inputs must contain exactly {sorted(expected)}")
    return evaluate_track(
        str(inputs["track_id"]),
        layer_dims=inputs["layer_dims"],
        batch_size=inputs["batch_size"],
        precision=str(inputs["precision"]),
        phase=str(inputs["phase"]),
        optimizer=str(inputs["optimizer"]),
        optimizer_precision=str(inputs["optimizer_precision"]),
        memory_limit=_replay_quantity(inputs["memory_limit"], "byte", "memory_limit"),
        effective_compute_rate=_replay_quantity(
            inputs["effective_compute_rate"], "flop / second", "effective_compute_rate"
        ),
        effective_bandwidth=_replay_quantity(
            inputs["effective_bandwidth"], "byte / second", "effective_bandwidth"
        ),
        launch_overhead=_replay_quantity(
            inputs["launch_overhead"], "millisecond", "launch_overhead"
        ),
    )


def snapshot(value: Any) -> Any:
    """Convert an experiment payload to independent JSON-compatible values."""
    if isinstance(value, ureg.Quantity):
        if value.units == ureg.byte:
            quantity, unit = value.to(ureg.byte), "byte"
        elif value.units == ureg.millisecond:
            quantity, unit = value.to(ureg.millisecond), "millisecond"
        elif value.units == ureg.flop / ureg.second:
            quantity, unit = value.to(ureg.flop / ureg.second), "flop / second"
        elif value.units == ureg.byte / ureg.second:
            quantity, unit = value.to(ureg.byte / ureg.second), "byte / second"
        elif value.units == 1 / ureg.second:
            quantity, unit = value.to(1 / ureg.second), "1 / second"
        else:
            quantity, unit = value, f"{value.units:~}"
        magnitude = quantity.magnitude
        if isinstance(magnitude, np.generic):
            magnitude = magnitude.item()
        return {"magnitude": magnitude, "unit": unit}
    if isinstance(value, Mapping):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_numerical_case(case_id: str, numeric_format: str) -> dict[str, Any]:
    """Execute a curated affine dot product in FP32, FP16, or INT8.

    INT8 uses a stated unit scale (one real unit per integer code), round-to-nearest
    conversion, saturation to the signed INT8 range, and an INT32 product and
    accumulator. The returned flags distinguish input clipping, representation
    rounding, arithmetic overflow, and nonfinite output.
    """
    if case_id not in NUMERICAL_CASES:
        raise ValueError(f"case_id must be one of {', '.join(NUMERICAL_CASES)}")
    if numeric_format not in _NUMERIC_FORMATS:
        raise ValueError(f"numeric_format must be one of {', '.join(_NUMERIC_FORMATS)}")
    _, numeric_bytes = resolve_precision(numeric_format)
    bytes_per_element = int(numeric_bytes.to(ureg.byte).magnitude)
    case = NUMERICAL_CASES[case_id]
    inputs64 = np.asarray(case["inputs"], dtype=np.float64)
    weights64 = np.asarray(case["weights"], dtype=np.float64)
    bias64 = float(case["bias"])
    reference_output = float(np.dot(inputs64, weights64) + bias64)

    clipped = False
    arithmetic_overflow = False
    if numeric_format == "int8":
        rounded_inputs = np.rint(inputs64)
        rounded_weights = np.rint(weights64)
        rounded_bias = float(np.rint(bias64))
        clipped = bool(
            np.any((rounded_inputs < -128) | (rounded_inputs > 127))
            or np.any((rounded_weights < -128) | (rounded_weights > 127))
            or rounded_bias < -128
            or rounded_bias > 127
        )
        represented_inputs = np.clip(rounded_inputs, -128, 127).astype(np.int8)
        represented_weights = np.clip(rounded_weights, -128, 127).astype(np.int8)
        represented_bias = np.int8(np.clip(rounded_bias, -128, 127))
        exact_products = represented_inputs.astype(np.int64) * represented_weights.astype(np.int64)
        exact_sum = int(exact_products.sum()) + int(represented_bias)
        arithmetic_overflow = bool(
            np.any((exact_products < np.iinfo(np.int32).min) | (exact_products > np.iinfo(np.int32).max))
            or exact_sum < np.iinfo(np.int32).min
            or exact_sum > np.iinfo(np.int32).max
        )
        products = np.multiply(
            represented_inputs.astype(np.int32), represented_weights.astype(np.int32), dtype=np.int32
        )
        output_value = int(np.add(products.sum(dtype=np.int32), np.int32(represented_bias), dtype=np.int32))
        represented_bias_value: int | float = int(represented_bias)
    else:
        dtype = np.float32 if numeric_format == "fp32" else np.float16
        with np.errstate(over="ignore", invalid="ignore"):
            represented_inputs = inputs64.astype(dtype)
            represented_weights = weights64.astype(dtype)
            represented_bias = dtype(bias64)
            products = np.multiply(represented_inputs, represented_weights, dtype=dtype)
            output = np.add(products.sum(dtype=dtype), represented_bias, dtype=dtype)
        output_value = float(output)
        arithmetic_overflow = bool(math.isfinite(reference_output) and not math.isfinite(output_value))
        represented_bias_value = float(represented_bias)

    represented_inputs_float = represented_inputs.astype(np.float64)
    represented_weights_float = represented_weights.astype(np.float64)
    representation_rounded = bool(
        np.any(represented_inputs_float != inputs64)
        or np.any(represented_weights_float != weights64)
        or represented_bias_value != bias64
    )
    output_nonfinite = not math.isfinite(float(output_value))
    output_changed = output_nonfinite or not math.isclose(
        float(output_value), reference_output, rel_tol=1e-7, abs_tol=1e-9
    )

    return {
        "inputs": {"case_id": case_id, "numeric_format": numeric_format},
        "case_id": case_id,
        "numeric_format": numeric_format,
        "description": case["description"],
        "bytes_per_element": bytes_per_element,
        "quantization_scale": 1.0 if numeric_format == "int8" else None,
        "accumulator_format": "int32" if numeric_format == "int8" else numeric_format,
        "represented_inputs": represented_inputs.tolist(),
        "weights": represented_weights.tolist(),
        "bias": represented_bias_value,
        "products": products.tolist(),
        "reference_output": reference_output,
        "output": output_value,
        "output_status": "nonfinite_overflow" if output_nonfinite else "available",
        "input_clipped": clipped,
        "representation_rounded": representation_rounded,
        "arithmetic_overflow": arithmetic_overflow,
        "output_nonfinite": output_nonfinite,
        "output_changed": output_changed,
        "assumption": (
            "Curated deterministic arithmetic case; it demonstrates numerical behavior, "
            "not task accuracy or learned-model quality."
        ),
    }
