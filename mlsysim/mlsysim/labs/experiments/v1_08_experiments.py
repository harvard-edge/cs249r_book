"""Chapter 8 training experiments built from MLSysIM memory accounting.

The convergence and numerical-outcome traces in this module are finite,
explicitly illustrative fixtures.  They let a learner compare time to the same
target without pretending that hardware throughput predicts convergence.
Physical memory is computed by :class:`TrainingMemoryModel`; execution time is
composed from supplied input/optimizer fixtures and analytical forward, backward,
and recomputation work at an explicitly assumed fraction of dense precision peak.

All four teaching tracks use GPT-2 as a common *development-host teaching
workload*.  This does not imply that GPT-2 is deployed on a TinyML, mobile, or
edge target.  Those tracks instead frame upstream teacher training or
adaptation for a downstream constrained-device task.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mlsysim import Hardware, Literature, Models
from mlsysim.core.units import Q_, resolve_precision
from mlsysim.engine.solvers.training import TrainingMemoryModel


_PRECISIONS = ("fp32", "bf16", "fp16", "fp8")
_OPTIMIZERS = ("sgd", "adam", "adamw")
_CHECKPOINTING = ("none", "selective", "full")

FIXTURE_NOTE = (
    "Illustrative scenario evidence for a matched teaching workload; "
    "not a benchmark measurement or a universal convergence law."
)

# These are scenario assumptions, not measurements of the named accelerators.
# Hardware capacity and workload structure come from their MLSysIM registries.
TRACKS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "label": "TinyML",
        "task": "Upstream teacher adaptation for a compact wake-word model",
        "hardware": Hardware.Cloud.A100,
        "seq_len": 128,
        "physical_batch": 4,
        "accumulation_steps": 8,
        "host_hour_usd": 2.40,
        "schedule_limit_hours": 0.45,
        "cost_limit_usd": 1.00,
        "checkpoint_probe_batch": 256,
    },
    "mobile": {
        "label": "Mobile",
        "task": "Upstream assistant adaptation for a mobile deployment",
        "hardware": Hardware.Cloud.A100,
        "seq_len": 256,
        "physical_batch": 8,
        "accumulation_steps": 4,
        "host_hour_usd": 2.40,
        "schedule_limit_hours": 0.40,
        "cost_limit_usd": 0.90,
        "checkpoint_probe_batch": 64,
    },
    "edge": {
        "label": "Edge",
        "task": "Upstream language adaptation for an edge inspection service",
        "hardware": Hardware.Cloud.A100,
        "seq_len": 512,
        "physical_batch": 8,
        "accumulation_steps": 4,
        "host_hour_usd": 2.40,
        "schedule_limit_hours": 0.55,
        "cost_limit_usd": 1.20,
        "checkpoint_probe_batch": 32,
    },
    "cloud": {
        "label": "Cloud",
        "task": "Continuous language-model training on a cloud development host",
        "hardware": Hardware.Cloud.H100,
        "seq_len": 1024,
        "physical_batch": 4,
        "accumulation_steps": 8,
        "host_hour_usd": 4.10,
        "schedule_limit_hours": 0.75,
        "cost_limit_usd": 2.50,
        "checkpoint_probe_batch": 32,
    },
}

# Each trace is a supplied sequence of (optimizer update, held-out quality).
# A result reaches the common target only at an actually listed observation.
CONVERGENCE_TRACES: dict[str, dict[Any, tuple[tuple[int, float], ...]]] = {
    "optimizer": {
        "sgd": ((0, 0.55), (400, 0.65), (800, 0.74), (1200, 0.805)),
        "adam": ((0, 0.55), (250, 0.68), (550, 0.77), (850, 0.815)),
        "adamw": ((0, 0.55), (250, 0.69), (500, 0.78), (750, 0.825)),
    },
    "effective_batch": {
        8: ((0, 0.55), (300, 0.67), (600, 0.75), (900, 0.81)),
        32: ((0, 0.55), (250, 0.69), (500, 0.78), (750, 0.825)),
        128: ((0, 0.55), (200, 0.68), (425, 0.77), (650, 0.81)),
        512: ((0, 0.55), (175, 0.67), (400, 0.75), (700, 0.79)),
    },
    "precision": {
        "fp32": ((0, 0.55), (250, 0.69), (500, 0.78), (750, 0.825)),
        "bf16": ((0, 0.55), (250, 0.69), (500, 0.78), (750, 0.824)),
        "fp16": ((0, 0.55), (275, 0.68), (550, 0.77), (800, 0.812)),
        "fp8": ((0, 0.55), (300, 0.66), (600, 0.74), (900, 0.785)),
    },
}

# Supplied numerical replay outcomes remain independent of execution speed.
NUMERICAL_FIXTURES: dict[str, dict[str, Any]] = {
    "fp32": {"finite_updates": 1000, "total_updates": 1000, "max_abs_delta": 0.0},
    "bf16": {"finite_updates": 1000, "total_updates": 1000, "max_abs_delta": 0.0031},
    "fp16": {"finite_updates": 998, "total_updates": 1000, "max_abs_delta": 0.0084},
    "fp8": {"finite_updates": 971, "total_updates": 1000, "max_abs_delta": 0.0310},
}

# Finite stage-time fixtures in milliseconds. Each value is a supplied
# illustrative observation, never extrapolated to an unlisted batch or format.
# Input entries are (raw preparation, overlapped portion). ``starved`` is an
# explicit adverse replay used to cross the schedule/cost boundary.
_INPUT_STAGE_FIXTURES = {
    "tinyml": {
        4: {"baseline": (11.0, 5.0), "prefetch": (5.5, 4.5), "starved": (30_000.0, 0.0)},
        8: {"baseline": (20.0, 8.0), "prefetch": (9.0, 7.0), "starved": (30_000.0, 0.0)},
        32: {"baseline": (58.0, 20.0), "prefetch": (27.0, 23.0), "starved": (30_000.0, 0.0)},
    },
    "mobile": {
        4: {"baseline": (7.5, 3.0), "prefetch": (4.0, 3.2), "starved": (30_000.0, 0.0)},
        8: {"baseline": (12.0, 5.0), "prefetch": (6.0, 5.0), "starved": (30_000.0, 0.0)},
        32: {"baseline": (32.0, 12.0), "prefetch": (15.0, 13.0), "starved": (30_000.0, 0.0)},
    },
    "edge": {
        4: {"baseline": (5.0, 2.0), "prefetch": (2.8, 2.2), "starved": (30_000.0, 0.0)},
        8: {"baseline": (8.0, 3.0), "prefetch": (4.0, 3.0), "starved": (30_000.0, 0.0)},
        32: {"baseline": (21.0, 8.0), "prefetch": (10.0, 8.5), "starved": (30_000.0, 0.0)},
    },
    "cloud": {
        4: {"baseline": (4.0, 1.5), "prefetch": (2.0, 1.6), "starved": (30_000.0, 0.0)},
        8: {"baseline": (6.0, 2.0), "prefetch": (3.0, 2.5), "starved": (30_000.0, 0.0)},
        32: {"baseline": (15.0, 5.0), "prefetch": (7.0, 6.0), "starved": (30_000.0, 0.0)},
    },
}

# Explicit scenario assumptions for arithmetic efficiency, not hardware specs
# or measured utilization. Only supported finite batches are exposed below.
_ARITHMETIC_EFFICIENCY = {"baseline": 0.50, "faster_arithmetic": 0.65}


def _dense_peak(profile: Mapping[str, Any], precision: str):
    # These hosts share BF16/FP16 dense Tensor Core throughput. No sparsity bonus.
    # Vendor tables: https://www.nvidia.com/en-gb/data-center/a100/
    # and https://www.nvidia.com/en-eu/data-center/h100/ (sparse rates excluded).
    key = {"bf16": "fp16_dense", "fp16": "fp16_dense"}.get(precision, precision)
    return key, profile["hardware"].compute.precision_flops.get(key)


# Optimizer state remains FP32 in this scenario, so these update-stage times do
# not change when activation/compute precision changes.
_OPTIMIZER_STAGE_FIXTURES = {
    "tinyml": {"sgd": 3.2, "adam": 5.8, "adamw": 6.0},
    "mobile": {"sgd": 3.2, "adam": 5.8, "adamw": 6.0},
    "edge": {"sgd": 3.2, "adam": 5.8, "adamw": 6.0},
    "cloud": {"sgd": 2.3, "adam": 4.2, "adamw": 4.3},
}
_RECOMPUTE_FORWARD_FRACTION = {"none": 0.0, "selective": 0.25, "full": 0.55}
_ACTIVATION_STRATEGY = {"none": "none", "selective": "selective", "full": "full"}
_TARGET_QUALITY = 0.80


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def track_profile(track_id: str) -> dict[str, Any]:
    """Return a copy of one track's illustrative development-host scenario."""
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    profile = dict(TRACKS[track_id])
    profile["track_id"] = track_id
    profile["model"] = Models.Language.GPT2
    profile["fixture_note"] = FIXTURE_NOTE
    return profile


def _validate_choices(precision: str, optimizer: str, checkpointing: str) -> None:
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {', '.join(_PRECISIONS)}")
    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {', '.join(_OPTIMIZERS)}")
    if checkpointing not in _CHECKPOINTING:
        raise ValueError(f"checkpointing must be one of {', '.join(_CHECKPOINTING)}")


def memory_outcome(
    track_id: str,
    *,
    physical_batch: int | None = None,
    accumulation_steps: int | None = None,
    precision: str = "bf16",
    optimizer: str = "adamw",
    checkpointing: str = "none",
) -> dict[str, Any]:
    """Compute the training memory stack through ``TrainingMemoryModel``."""
    profile = track_profile(track_id)
    _validate_choices(precision, optimizer, checkpointing)
    physical_batch = _positive_int(
        profile["physical_batch"] if physical_batch is None else physical_batch,
        "physical_batch",
    )
    accumulation_steps = _positive_int(
        profile["accumulation_steps"] if accumulation_steps is None else accumulation_steps,
        "accumulation_steps",
    )
    effective_batch = physical_batch * accumulation_steps
    memory = TrainingMemoryModel().solve(
        profile["model"],
        profile["hardware"],
        batch_size=effective_batch,
        seq_len=profile["seq_len"],
        precision=precision,
        optimizer=optimizer,
        activation_checkpointing=_ACTIVATION_STRATEGY[checkpointing],
        gradient_accumulation_steps=accumulation_steps,
    )
    _, precision_bytes = resolve_precision(precision)
    inference_weights = profile["model"].size_in_bytes(precision_bytes)
    return {
        "track_id": track_id,
        "task": profile["task"],
        "model_name": profile["model"].name,
        "hardware_name": profile["hardware"].name,
        "physical_batch": physical_batch,
        "accumulation_steps": accumulation_steps,
        "effective_batch": effective_batch,
        "seq_len": profile["seq_len"],
        "precision": precision,
        "optimizer": optimizer,
        "checkpointing": checkpointing,
        "inference_weights_gb": round(inference_weights.to("GB").magnitude, 6),
        "weights_gb": round(memory.weights.to("GB").magnitude, 6),
        "gradients_gb": round(memory.gradients.to("GB").magnitude, 6),
        "optimizer_state_gb": round(memory.optimizer_state.to("GB").magnitude, 6),
        "activations_gb": round(memory.activations.to("GB").magnitude, 6),
        "communication_buffers_gb": round(memory.communication_buffers.to("GB").magnitude, 6),
        "total_memory_gb": round(memory.total_memory.to("GB").magnitude, 6),
        "available_memory_gb": round(memory.available_memory.to("GB").magnitude, 6),
        "memory_utilization": round(memory.memory_utilization, 6),
        "feasible": memory.feasible,
        "constraint_trace": tuple(memory.constraint_trace),
        "fixture_note": FIXTURE_NOTE,
        "inputs": {
            "track_id": track_id,
            "physical_batch": physical_batch,
            "accumulation_steps": accumulation_steps,
            "precision": precision,
            "optimizer": optimizer,
            "checkpointing": checkpointing,
        },
    }


def adaptation_outcome(
    track_id: str,
    *,
    adaptation_scope: str = "full",
    physical_batch: int | None = None,
    accumulation_steps: int | None = None,
    precision: str = "bf16",
    optimizer: str = "adamw",
    checkpointing: str = "none",
) -> dict[str, Any]:
    """Account for full or restricted adaptation without inventing progress.

    Restricted adaptation is a bounded, hypothetical recipe: one tenth of the
    existing parameters are trainable, distributed through the model. The
    backward path still crosses the frozen layers, so the solver retains its
    full activation estimate. This is not a detached frozen-prefix recipe or
    an adapter architecture; no extra adapter weights are silently introduced.
    """
    fractions = {"full": 1.0, "restricted": 0.1}
    if adaptation_scope not in fractions:
        raise ValueError("adaptation_scope must be full or restricted")
    baseline = memory_outcome(
        track_id, physical_batch=physical_batch,
        accumulation_steps=accumulation_steps, precision=precision,
        optimizer=optimizer, checkpointing=checkpointing,
    )
    profile = track_profile(track_id)
    memory = TrainingMemoryModel().solve(
        profile["model"], profile["hardware"],
        batch_size=baseline["effective_batch"], seq_len=profile["seq_len"],
        precision=precision, optimizer=optimizer,
        activation_checkpointing=checkpointing,
        gradient_accumulation_steps=baseline["accumulation_steps"],
        trainable_fraction=fractions[adaptation_scope],
    )
    result = dict(baseline)
    for field in ("weights", "gradients", "optimizer_state", "activations", "communication_buffers"):
        result[f"{field}_gb"] = round(getattr(memory, field).to("GB").magnitude, 6)
    result.update({
        "model_key": "v1_08_experiments.adaptation_outcome",
        "adaptation_scope": adaptation_scope,
        "trainable_fraction": fractions[adaptation_scope],
        "total_memory_gb": round(memory.total_memory.to("GB").magnitude, 6),
        "memory_utilization": round(memory.memory_utilization, 6),
        "feasible": memory.feasible,
        "constraint_trace": tuple(memory.constraint_trace),
        "quality_evidence": None,
        "quality_status": "Unavailable: no supplied matched adaptation-scope progress observations.",
        "activation_assumption": "Backward dependencies cross frozen layers; forward activations remain retained. This is not a detached frozen prefix.",
        "state_boundary": "Optimizer state includes the solver's master-copy and optimizer allocations; no timing or quality reduction is inferred from freezing.",
        "inputs": {**baseline["inputs"], "adaptation_scope": adaptation_scope},
    })
    return result


def stage_timing(
    track_id: str,
    *,
    physical_batch: int | None = None,
    accumulation_steps: int | None = None,
    precision: str = "bf16",
    optimizer: str = "adamw",
    checkpointing: str = "none",
    input_policy: str = "baseline",
    compute_policy: str = "baseline",
) -> dict[str, Any]:
    """Compose an update from finite input fixtures and analytical arithmetic time.

    Input, forward, backward, and recomputation occur for every physical
    microbatch.  The optimizer stage occurs once after all accumulated
    gradients form the effective batch. Arithmetic efficiency is an explicit scenario assumption, not measured telemetry.
    Unlisted physical batches fail rather than being extrapolated.
    """
    profile = track_profile(track_id)
    _validate_choices(precision, optimizer, checkpointing)
    physical_batch = _positive_int(
        profile["physical_batch"] if physical_batch is None else physical_batch,
        "physical_batch",
    )
    accumulation_steps = _positive_int(
        profile["accumulation_steps"] if accumulation_steps is None else accumulation_steps,
        "accumulation_steps",
    )
    if physical_batch not in _INPUT_STAGE_FIXTURES[track_id]:
        choices = ", ".join(str(item) for item in _INPUT_STAGE_FIXTURES[track_id])
        raise ValueError(f"physical_batch for stage timing must be one of {choices}")
    if input_policy not in {"baseline", "prefetch", "starved"}:
        raise ValueError("input_policy must be baseline, prefetch, or starved")
    if compute_policy not in {"baseline", "faster_arithmetic"}:
        raise ValueError("compute_policy must be baseline or faster_arithmetic")
    if compute_policy == "faster_arithmetic" and precision != "bf16":
        raise ValueError("faster_arithmetic is supplied only for the BF16 bottleneck replay")

    raw_input_ms, overlapped_input_ms = _INPUT_STAGE_FIXTURES[track_id][physical_batch][input_policy]
    visible_input_ms = raw_input_ms - overlapped_input_ms
    peak_key, peak = _dense_peak(profile, precision)
    supported = peak is not None
    # 6ND counts a single microbatch's forward + backward work. Accumulation
    # repeats it below; the effective batch must not be multiplied here again.
    model_ops = Q_(profile["model"].parameters.to("param").magnitude
                   * Literature.Chinchilla.ComputeConstant
                   * physical_batch * profile["seq_len"], "flop")
    efficiency = _ARITHMETIC_EFFICIENCY[compute_policy]
    # No hardware execution time exists for an unsupported numerical format.
    # Temporary zeros allow composition; public fields become None below.
    arithmetic_ms = (model_ops / (peak * efficiency)).to("millisecond").magnitude if supported else 0.0
    forward_ms, backward_ms = arithmetic_ms / 3, arithmetic_ms * 2 / 3
    recompute_ms = forward_ms * _RECOMPUTE_FORWARD_FRACTION[checkpointing]
    optimizer_ms = _OPTIMIZER_STAGE_FIXTURES[track_id][optimizer]
    microstep_time = sum(
        (
            Q_(visible_input_ms, "millisecond"),
            Q_(forward_ms, "millisecond"),
            Q_(backward_ms, "millisecond"),
            Q_(recompute_ms, "millisecond"),
        ),
        Q_(0, "millisecond"),
    )
    update_time = accumulation_steps * microstep_time + Q_(optimizer_ms, "millisecond")
    microstep_ms = microstep_time.to("millisecond").magnitude
    update_time_ms = update_time.to("millisecond").magnitude
    effective_batch = physical_batch * accumulation_steps
    stage_totals_ms = {
        "input": visible_input_ms * accumulation_steps,
        "forward": forward_ms * accumulation_steps,
        "backward": backward_ms * accumulation_steps,
        "recompute": recompute_ms * accumulation_steps,
        "optimizer": optimizer_ms,
    }
    dominant_stage = max(stage_totals_ms, key=stage_totals_ms.get)
    return {
        "track_id": track_id,
        "physical_batch": physical_batch,
        "accumulation_steps": accumulation_steps,
        "effective_batch": effective_batch,
        "precision": precision,
        "optimizer": optimizer,
        "checkpointing": checkpointing,
        "arithmetic_efficiency_assumption": efficiency,
        "arithmetic_timing_basis": "Analytical 6ND / (dense precision peak × assumed arithmetic efficiency); not a measured stage profile.",
        "performance_supported": supported,
        "peak_key": peak_key,
        "raw_input_ms": round(raw_input_ms, 6),
        "overlapped_input_ms": round(overlapped_input_ms, 6),
        "visible_input_ms": round(visible_input_ms, 6),
        "forward_ms": round(forward_ms, 6) if supported else None,
        "backward_ms": round(backward_ms, 6) if supported else None,
        "recompute_ms": round(recompute_ms, 6) if supported else None,
        "optimizer_ms": round(optimizer_ms, 6),
        "microstep_ms": round(microstep_ms, 6) if supported else None,
        "update_time_ms": round(update_time_ms, 6) if supported else None,
        "samples_per_second": round(effective_batch / (update_time_ms / 1000.0), 6) if supported else None,
        "dominant_stage": dominant_stage if supported else "unavailable",
        "input_policy": input_policy,
        "compute_policy": compute_policy,
        "timing_assumption": FIXTURE_NOTE,
        "inputs": {
            "track_id": track_id,
            "physical_batch": physical_batch,
            "accumulation_steps": accumulation_steps,
            "precision": precision,
            "optimizer": optimizer,
            "checkpointing": checkpointing,
            "input_policy": input_policy,
            "compute_policy": compute_policy,
        },
    }


def model_flops_diagnostic(track_id: str, timing: Mapping[str, Any], *, fits: bool = True) -> dict[str, Any]:
    """Analytical dense 6ND MFU for a supplied full-adaptation timing replay.

    Dense BF16 and FP16 Tensor Core peaks share the registry's fp16_dense
    capability on these development hosts. Sparse peaks are never credited.
    Busy time is the replay's device-stage occupancy, not profiler telemetry.
    Recomputed forward work occupies device time but earns no model-FLOP credit.
    """
    profile = track_profile(track_id)
    if timing["track_id"] != track_id:
        raise ValueError("timing track must match the diagnostic track")
    tokens = timing["effective_batch"] * profile["seq_len"]
    useful = Q_(profile["model"].parameters.to("param").magnitude
                * Literature.Chinchilla.ComputeConstant * tokens, "flop")
    peak_key, peak = _dense_peak(profile, timing["precision"])
    duration = Q_(timing["update_time_ms"], "millisecond") if timing["update_time_ms"] is not None else None
    ratio = (useful / (peak * duration)).to("").magnitude if peak is not None and duration is not None else None
    busy_ms = timing["accumulation_steps"] * sum(timing[k] for k in ("forward_ms", "backward_ms", "recompute_ms")) + timing["optimizer_ms"] if duration is not None else None
    reason = ("No run: training state exceeds capacity." if not fits else
              "No registered dense peak for this precision on this host." if peak is None else
              "Supplied stage timing is inconsistent with this precision's peak; MFU is unavailable." if ratio > 1 else "")
    return {
        "model_name": profile["model"].name,
        "tokens_per_update": tokens,
        "useful_model_flops": useful.to("flop").magnitude,
        "peak_key": peak_key,
        "precision": timing["precision"],
        "dense_peak_tflops": peak.to("TFLOPs / second").magnitude if peak is not None else None,
        "update_time_ms": timing["update_time_ms"],
        "busy_fraction": busy_ms / timing["update_time_ms"] if fits and busy_ms is not None else None,
        "mfu": ratio if not reason else None,
        "status": "unavailable" if reason else "analytical reference",
        "reason": reason,
        "counting_convention": "6 × parameters × effective-batch tokens; forward + backward, excluding rematerialization, padding and optimizer work.",
        "boundary": "Supplied teaching replay, not a measured profile; MFU does not measure convergence or identify the bottleneck by itself.",
    }


def convergence_trace(family: str, key: Any) -> tuple[tuple[int, float], ...]:
    """Return one immutable supplied convergence trace."""
    if family not in CONVERGENCE_TRACES:
        raise ValueError(f"family must be one of {', '.join(CONVERGENCE_TRACES)}")
    if key not in CONVERGENCE_TRACES[family]:
        choices = ", ".join(str(item) for item in CONVERGENCE_TRACES[family])
        raise ValueError(f"key for {family} must be one of {choices}")
    return CONVERGENCE_TRACES[family][key]


def compare_time_to_target(
    track_id: str,
    *,
    evidence_family: str,
    evidence_key: Any,
    physical_batch: int | None = None,
    accumulation_steps: int | None = None,
    precision: str = "bf16",
    optimizer: str = "adamw",
    checkpointing: str = "none",
    input_policy: str = "baseline",
    compute_policy: str = "baseline",
) -> dict[str, Any]:
    """Calculate time and cost to a common target from a supplied trace."""
    _validate_choices(precision, optimizer, checkpointing)
    profile = track_profile(track_id)
    timing = stage_timing(
        track_id,
        physical_batch=physical_batch,
        accumulation_steps=accumulation_steps,
        precision=precision,
        optimizer=optimizer,
        checkpointing=checkpointing,
        input_policy=input_policy,
        compute_policy=compute_policy,
    )
    memory = memory_outcome(
        track_id,
        physical_batch=timing["physical_batch"],
        accumulation_steps=timing["accumulation_steps"],
        precision=precision,
        optimizer=optimizer,
        checkpointing=checkpointing,
    )
    trace = convergence_trace(evidence_family, evidence_key)
    target_observation = next(((step, quality) for step, quality in trace if quality >= _TARGET_QUALITY), None)
    reached_target = target_observation is not None
    updates = target_observation[0] if target_observation else None
    total_time = Q_(updates * timing["update_time_ms"], "millisecond") if updates is not None and timing["update_time_ms"] is not None else None
    total_time_ms = total_time.to("millisecond").magnitude if total_time is not None else None
    total_time_seconds = total_time.to("second").magnitude if total_time is not None else None
    total_time_hours = total_time.to("hour").magnitude if total_time is not None else None
    cost = (
        total_time.to("hour") * Q_(profile["host_hour_usd"], "USD / hour")
        if total_time is not None
        else None
    )
    cost_usd = cost.to("USD").magnitude if cost is not None else None
    samples_presented = updates * timing["effective_batch"] if updates is not None else None
    diagnostic = model_flops_diagnostic(track_id, timing, fits=memory["feasible"])
    violations: list[str] = []
    if memory["feasible"] and diagnostic["mfu"] is None:
        violations.append("unvalidated_precision_timing")
    if not memory["feasible"]:
        violations.append("memory")
    if not reached_target:
        violations.append("target_quality")
    if total_time_hours is not None and total_time_hours > profile["schedule_limit_hours"]:
        violations.append("schedule")
    if cost_usd is not None and cost_usd > profile["cost_limit_usd"]:
        violations.append("cost")
    return {
        "track_id": track_id,
        "evidence_family": evidence_family,
        "evidence_key": evidence_key,
        "target_quality": _TARGET_QUALITY,
        "trace": trace,
        "reached_target": reached_target,
        "updates_to_target": updates,
        "observed_quality": target_observation[1] if target_observation else trace[-1][1],
        "samples_presented": samples_presented,
        "total_time_ms": round(total_time_ms, 6) if total_time_ms is not None else None,
        "total_time_seconds": round(total_time_seconds, 6) if total_time_seconds is not None else None,
        "total_time_hours": round(total_time_hours, 9) if total_time_hours is not None else None,
        "cost_usd": round(cost_usd, 6) if cost_usd is not None else None,
        "host_hour_usd": profile["host_hour_usd"],
        "schedule_limit_hours": profile["schedule_limit_hours"],
        "cost_limit_usd": profile["cost_limit_usd"],
        "timing": timing,
        "model_flops_diagnostic": diagnostic,
        "memory": memory,
        "feasible": not violations,
        "violations": tuple(violations),
        "fixture_note": FIXTURE_NOTE,
        "inputs": {
            "track_id": track_id,
            "evidence_family": evidence_family,
            "evidence_key": evidence_key,
            "physical_batch": timing["physical_batch"],
            "accumulation_steps": timing["accumulation_steps"],
            "precision": precision,
            "optimizer": optimizer,
            "checkpointing": checkpointing,
            "input_policy": input_policy,
            "compute_policy": compute_policy,
        },
    }


def memory_experiment(track_id: str, *, physical_batch: int | None = None) -> dict[str, Any]:
    """Compare inference weights with SGD and AdamW training allocations."""
    rows = tuple(
        memory_outcome(track_id, physical_batch=physical_batch, optimizer=optimizer)
        for optimizer in ("sgd", "adamw")
    )
    return {"track_id": track_id, "rows": rows, "fixture_note": FIXTURE_NOTE}


def optimizer_experiment(track_id: str, *, optimizer: str) -> dict[str, Any]:
    """Evaluate optimizer state, update time, and supplied convergence evidence."""
    return compare_time_to_target(
        track_id,
        evidence_family="optimizer",
        evidence_key=optimizer,
        optimizer=optimizer,
    )


def batch_experiment(track_id: str, *, physical_batch: int, accumulation_steps: int) -> dict[str, Any]:
    """Evaluate one physical/effective batch choice with batch convergence evidence."""
    effective_batch = _positive_int(physical_batch, "physical_batch") * _positive_int(
        accumulation_steps, "accumulation_steps"
    )
    return compare_time_to_target(
        track_id,
        evidence_family="effective_batch",
        evidence_key=effective_batch,
        physical_batch=physical_batch,
        accumulation_steps=accumulation_steps,
    )


def batch_policy(track_id: str) -> dict[str, Any]:
    """Derive batch baseline, strategy options, and choices from the track profile."""
    profile = track_profile(track_id)
    p = profile["physical_batch"]
    acc_base = 8 // p
    acc_target = 32 // p
    acc_128 = 128 // p
    acc_512 = 512 // p

    baseline_label = f"Resident batch {p}" if acc_base == 1 else f"Physical {p} × accumulate {acc_base}"
    target_label = f"Physical {p} × accumulate {acc_target}"

    options = {
        baseline_label: f"{p}x{acc_base}",
        target_label: f"{p}x{acc_target}",
        "Resident batch 32": "32x1",
        f"Physical {p} × accumulate {acc_128}": f"{p}x{acc_128}",
        f"Physical {p} × accumulate {acc_512}": f"{p}x{acc_512}",
    }
    choices = {
        f"{p}x{acc_base}": (p, acc_base),
        f"{p}x{acc_target}": (p, acc_target),
        "32x1": (32, 1),
        f"{p}x{acc_128}": (p, acc_128),
        f"{p}x{acc_512}": (p, acc_512),
    }
    return {
        "track_id": track_id,
        "physical_batch": p,
        "baseline": (p, acc_base),
        "baseline_key": f"{p}x{acc_base}",
        "baseline_label": baseline_label,
        "default_label": target_label,
        "default_key": f"{p}x{acc_target}",
        "options": options,
        "choices": choices,
    }


def precision_experiment(track_id: str, *, precision: str) -> dict[str, Any]:
    """Compare precision execution with independent numerical and convergence evidence."""
    if precision not in NUMERICAL_FIXTURES:
        raise ValueError(f"precision must be one of {', '.join(NUMERICAL_FIXTURES)}")
    outcome = compare_time_to_target(
        track_id,
        evidence_family="precision",
        evidence_key=precision,
        precision=precision,
    )
    numerical = dict(NUMERICAL_FIXTURES[precision])
    numerical["finite_fraction"] = numerical["finite_updates"] / numerical["total_updates"]
    numerical["fixture_note"] = FIXTURE_NOTE
    outcome["numerical_replay"] = numerical
    outcome["exact_precision_rate_listed"] = precision in track_profile(track_id)["hardware"].compute.precision_flops
    return outcome


def checkpoint_experiment(track_id: str, *, checkpointing: str) -> dict[str, Any]:
    """Compare activation memory saved with explicitly added recomputation work."""
    baseline = compare_time_to_target(
        track_id,
        evidence_family="effective_batch",
        evidence_key=32,
        checkpointing="none",
    )
    result = compare_time_to_target(
        track_id,
        evidence_family="effective_batch",
        evidence_key=32,
        checkpointing=checkpointing,
    )
    return {
        "track_id": track_id,
        "baseline": baseline,
        "result": result,
        "activation_memory_saved_gb": round(
            baseline["memory"]["activations_gb"] - result["memory"]["activations_gb"], 6
        ),
        "added_update_time_ms": round(
            result["timing"]["update_time_ms"] - baseline["timing"]["update_time_ms"], 6
        ),
        "fixture_note": FIXTURE_NOTE,
    }


def bottleneck_experiment(track_id: str, *, intervention: str, input_stress: bool = False) -> dict[str, Any]:
    """Apply a cause-specific intervention to the same training baseline."""
    interventions: dict[str, dict[str, Any]] = {
        "none": {},
        "prefetch": {"input_policy": "prefetch"},
        "arithmetic": {"compute_policy": "faster_arithmetic"},
        "checkpoint": {"checkpointing": "full"},
    }
    if intervention not in interventions:
        raise ValueError(f"intervention must be one of {', '.join(interventions)}")
    if not isinstance(input_stress, bool):
        raise ValueError("input_stress must be boolean")
    common = {"evidence_family": "effective_batch", "evidence_key": 32, "input_policy": "starved" if input_stress else "baseline"}
    baseline = compare_time_to_target(track_id, **common)
    result = compare_time_to_target(track_id, **{**common, **interventions[intervention]})
    return {
        "track_id": track_id,
        "intervention": intervention,
        "baseline": baseline,
        "result": result,
        "delta_update_time_ms": round(
            result["timing"]["update_time_ms"] - baseline["timing"]["update_time_ms"], 6
        ),
        "delta_memory_gb": round(
            result["memory"]["total_memory_gb"] - baseline["memory"]["total_memory_gb"], 6
        ),
        "fixture_note": FIXTURE_NOTE,
    }


_REPLAY_KEYS = {
    "v1_08_experiments.adaptation_outcome",
    "v1_08_experiments.optimizer_experiment",
    "v1_08_experiments.batch_experiment",
    "v1_08_experiments.precision_experiment",
    "v1_08_experiments.memory_outcome",
    "v1_08_experiments.bottleneck_experiment",
}
_COMPARE_INPUT_KEYS = {
    "track_id",
    "evidence_family",
    "evidence_key",
    "physical_batch",
    "accumulation_steps",
    "precision",
    "optimizer",
    "checkpointing",
    "input_policy",
    "compute_policy",
}
_MEMORY_INPUT_KEYS = {
    "track_id",
    "physical_batch",
    "accumulation_steps",
    "precision",
    "optimizer",
    "checkpointing",
}


def _replay_mapping(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a mapping")
    data = dict(inputs)
    if not data:
        raise ValueError("inputs must not be empty")
    return data


def _require_exact_keys(data: Mapping[str, Any], expected: set[str]) -> None:
    supplied = set(data)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"replay inputs mismatch; missing={missing}, extra={extra}")


def replay(model_key: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one captured Chapter 8 outcome through a closed dispatcher.

    ``inputs`` must be the mapping stored on the captured baseline, result, or
    chosen result itself. No dynamic import, attribute lookup, or arbitrary
    callable dispatch is performed.
    """
    if model_key not in _REPLAY_KEYS:
        raise ValueError(f"unknown Chapter 8 replay model_key: {model_key}")
    data = _replay_mapping(inputs)

    if model_key == "v1_08_experiments.adaptation_outcome":
        _require_exact_keys(data, _MEMORY_INPUT_KEYS | {"adaptation_scope"})
        return adaptation_outcome(**data)

    if model_key == "v1_08_experiments.memory_outcome":
        _require_exact_keys(data, _MEMORY_INPUT_KEYS)
        return memory_outcome(**data)

    _require_exact_keys(data, _COMPARE_INPUT_KEYS)
    if model_key == "v1_08_experiments.precision_experiment":
        outcome = precision_experiment(data["track_id"], precision=data["precision"])
        if outcome["inputs"] != data:
            raise ValueError("inputs do not describe a precision_experiment outcome")
        return outcome

    expected_family = {
        "v1_08_experiments.optimizer_experiment": "optimizer",
        "v1_08_experiments.batch_experiment": "effective_batch",
    }.get(model_key)
    if expected_family is not None and data["evidence_family"] != expected_family:
        raise ValueError(f"{model_key} requires evidence_family={expected_family}")
    return compare_time_to_target(**data)


__all__ = [
    "adaptation_outcome",
    "CONVERGENCE_TRACES",
    "FIXTURE_NOTE",
    "NUMERICAL_FIXTURES",
    "TRACKS",
    "batch_experiment",
    "batch_policy",
    "bottleneck_experiment",
    "checkpoint_experiment",
    "compare_time_to_target",
    "convergence_trace",
    "model_flops_diagnostic",
    "memory_experiment",
    "memory_outcome",
    "optimizer_experiment",
    "precision_experiment",
    "replay",
    "stage_timing",
    "track_profile",
]
