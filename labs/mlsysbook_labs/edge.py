"""Shared edge-placement helpers for track-aware MLSysBook labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class EdgeDeviceProfile:
    track_id: str
    label: str
    category: str
    hardware_ref: str
    memory_capacity_mb: float
    available_memory_mb: float
    energy_budget_wh: float
    energy_budget_label: str
    cpu_power_w: float
    gpu_power_w: float
    accelerator_power_w: float
    accelerator_label: str
    accelerator_speedup: float
    accelerator_energy_gain: float
    default_model_params_m: float
    default_batch_size: int
    default_contexts: int
    iid_rounds: int
    update_payload_mb: float
    latency_budget_ms: float
    placement_options: tuple[str, ...]
    adaptation_options: tuple[str, ...]
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class TrainingMemoryBreakdown:
    weights_mb: float
    gradients_mb: float
    optimizer_mb: float
    activations_mb: float
    total_mb: float
    inference_mb: float
    amplification: float
    fits_memory: bool


@dataclass(frozen=True)
class AdaptationStorageResult:
    contexts: int
    base_model_mb: float
    full_total_mb: float
    lora_total_mb: float
    bias_total_mb: float
    lora_savings_ratio: float


@dataclass(frozen=True)
class EnergyDrainResult:
    target: str
    label: str
    power_w: float
    duration_s: float
    energy_wh: float
    budget_used_pct: float
    sessions_per_budget: float


@dataclass(frozen=True)
class FederatedCommunicationResult:
    iid_rounds: int
    noniid_rounds: float
    compressed_rounds: float
    round_multiplier: float
    bytes_per_round_mb: float
    compressed_bytes_per_round_mb: float
    total_communication_mb: float
    compression_label: str
    drift_penalty: float


_TRACK_EDGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "iphone": {
        "available_memory_mb": 300.0,
        "energy_budget_label": "phone battery",
        "cpu_power_w": 3.0,
        "gpu_power_w": 2.0,
        "accelerator_power_w": 0.5,
        "accelerator_label": "Neural Engine",
        "accelerator_speedup": 20.0,
        "accelerator_energy_gain": 50.0,
        "default_model_params_m": 10.0,
        "default_batch_size": 8,
        "default_contexts": 10,
        "iid_rounds": 50,
        "update_payload_mb": 40.0,
        "latency_budget_ms": 100.0,
    },
    "oura_ring": {
        "available_memory_mb": 0.5,
        "energy_budget_label": "ring battery",
        "cpu_power_w": 0.015,
        "gpu_power_w": 0.010,
        "accelerator_power_w": 0.004,
        "accelerator_label": "tiny int8 path",
        "accelerator_speedup": 5.0,
        "accelerator_energy_gain": 12.0,
        "default_model_params_m": 0.08,
        "default_batch_size": 1,
        "default_contexts": 4,
        "iid_rounds": 40,
        "update_payload_mb": 0.32,
        "latency_budget_ms": 250.0,
    },
    "robotaxi": {
        "available_memory_mb": 4096.0,
        "energy_budget_wh": 0.5,
        "energy_budget_label": "safety-cycle compute budget",
        "cpu_power_w": 60.0,
        "gpu_power_w": 45.0,
        "accelerator_power_w": 35.0,
        "accelerator_label": "vehicle accelerator",
        "accelerator_speedup": 12.0,
        "accelerator_energy_gain": 18.0,
        "default_model_params_m": 25.0,
        "default_batch_size": 4,
        "default_contexts": 6,
        "iid_rounds": 60,
        "update_payload_mb": 100.0,
        "latency_budget_ms": 50.0,
    },
    "cloud_fleet": {
        "available_memory_mb": 40960.0,
        "energy_budget_wh": 0.2,
        "energy_budget_label": "request energy budget",
        "cpu_power_w": 700.0,
        "gpu_power_w": 700.0,
        "accelerator_power_w": 700.0,
        "accelerator_label": "H100 GPU",
        "accelerator_speedup": 1.0,
        "accelerator_energy_gain": 1.0,
        "default_model_params_m": 110.0,
        "default_batch_size": 16,
        "default_contexts": 20,
        "iid_rounds": 50,
        "update_payload_mb": 440.0,
        "latency_budget_ms": 200.0,
    },
}


def _quantity_to_float(value: Any, unit: str, default: float) -> float:
    if value is None:
        return default
    if hasattr(value, "m_as"):
        try:
            return float(value.m_as(unit))
        except Exception:
            return default
    if hasattr(value, "to"):
        try:
            return float(value.to(unit).magnitude)
        except Exception:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _hardware_attr(hardware: Any, dotted: str) -> Any:
    current = hardware
    for part in dotted.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def edge_device_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
) -> EdgeDeviceProfile:
    """Build a track edge profile from shared track metadata and MLSysIM hardware."""
    defaults = _TRACK_EDGE_DEFAULTS[profile.track_id]
    memory_capacity_mb = _quantity_to_float(
        _hardware_attr(hardware, "memory.capacity"),
        "MB",
        defaults["available_memory_mb"],
    )
    battery_wh = _quantity_to_float(
        getattr(hardware, "battery_capacity", None),
        "Wh",
        defaults.get("energy_budget_wh", 1.0),
    )
    energy_budget_wh = float(defaults.get("energy_budget_wh", battery_wh))
    placement_options = tuple(variant.defaults.get("placements", ()))
    adaptation_options = tuple(variant.defaults.get("adaptation_options", ()))

    return EdgeDeviceProfile(
        track_id=profile.track_id,
        label=profile.label,
        category=profile.category,
        hardware_ref=profile.hardware_ref,
        memory_capacity_mb=memory_capacity_mb,
        available_memory_mb=float(defaults["available_memory_mb"]),
        energy_budget_wh=energy_budget_wh,
        energy_budget_label=str(defaults["energy_budget_label"]),
        cpu_power_w=float(defaults["cpu_power_w"]),
        gpu_power_w=float(defaults["gpu_power_w"]),
        accelerator_power_w=float(defaults["accelerator_power_w"]),
        accelerator_label=str(defaults["accelerator_label"]),
        accelerator_speedup=float(defaults["accelerator_speedup"]),
        accelerator_energy_gain=float(defaults["accelerator_energy_gain"]),
        default_model_params_m=float(defaults["default_model_params_m"]),
        default_batch_size=int(defaults["default_batch_size"]),
        default_contexts=int(defaults["default_contexts"]),
        iid_rounds=int(defaults["iid_rounds"]),
        update_payload_mb=float(defaults["update_payload_mb"]),
        latency_budget_ms=float(defaults["latency_budget_ms"]),
        placement_options=placement_options,
        adaptation_options=adaptation_options,
        source_refs=(profile.hardware_ref, variant.model_ref),
    )


def training_memory_breakdown(
    *,
    params_m: float,
    batch_size: int,
    strategy: str,
    available_memory_mb: float,
    bytes_fp16: int = 2,
    bytes_fp32: int = 4,
    adam_multiplier: int = 2,
    activation_ratio: float = 0.39,
    lora_fraction: float = 0.01,
    bias_fraction: float = 0.001,
) -> TrainingMemoryBreakdown:
    """Estimate training memory from weights, gradients, optimizer state, and activations."""
    params = params_m * 1e6
    train_frac = {"full": 1.0, "lora": lora_fraction, "bias": bias_fraction}[strategy]
    trainable = params * train_frac

    weights_mb = params * bytes_fp16 / (1024 * 1024)
    gradients_mb = trainable * bytes_fp32 / (1024 * 1024)
    optimizer_mb = trainable * bytes_fp32 * adam_multiplier / (1024 * 1024)
    activations_mb = params * batch_size * activation_ratio * bytes_fp32 / (1024 * 1024)
    if strategy != "full":
        activations_mb *= train_frac * 10

    total_mb = weights_mb + gradients_mb + optimizer_mb + activations_mb
    inference_mb = weights_mb
    amplification = total_mb / max(inference_mb, 0.01)
    return TrainingMemoryBreakdown(
        weights_mb=weights_mb,
        gradients_mb=gradients_mb,
        optimizer_mb=optimizer_mb,
        activations_mb=activations_mb,
        total_mb=total_mb,
        inference_mb=inference_mb,
        amplification=amplification,
        fits_memory=total_mb <= available_memory_mb,
    )


def adaptation_storage(
    *,
    contexts: int,
    model_mb: float,
    lora_fraction: float = 0.01,
    bias_fraction: float = 0.001,
) -> AdaptationStorageResult:
    """Estimate storage for full fine-tuning, LoRA, and bias-only personalization."""
    full_per_context = model_mb
    lora_per_context = model_mb * lora_fraction + 0.2
    bias_per_context = model_mb * bias_fraction + 0.1
    full_total = model_mb + contexts * full_per_context
    lora_total = model_mb + contexts * lora_per_context
    bias_total = model_mb + contexts * bias_per_context
    return AdaptationStorageResult(
        contexts=contexts,
        base_model_mb=model_mb,
        full_total_mb=full_total,
        lora_total_mb=lora_total,
        bias_total_mb=bias_total,
        lora_savings_ratio=full_total / max(lora_total, 0.01),
    )


def energy_drain(
    profile: EdgeDeviceProfile,
    *,
    target: str,
    base_duration_s: float = 30.0,
) -> EnergyDrainResult:
    """Estimate energy used by one local adaptation or inference session."""
    target_props = {
        "cpu": {
            "power_w": profile.cpu_power_w,
            "duration_s": base_duration_s * 2.5,
            "label": "CPU",
        },
        "gpu": {
            "power_w": profile.gpu_power_w,
            "duration_s": base_duration_s * 1.5,
            "label": "GPU",
        },
        "npu": {
            "power_w": profile.accelerator_power_w,
            "duration_s": base_duration_s / profile.accelerator_speedup,
            "label": profile.accelerator_label,
        },
    }
    props = target_props[target]
    energy_wh = props["power_w"] * props["duration_s"] / 3600
    budget_used_pct = energy_wh / max(profile.energy_budget_wh, 1e-9) * 100
    sessions_per_budget = 100.0 / budget_used_pct if budget_used_pct > 0 else float("inf")
    return EnergyDrainResult(
        target=target,
        label=props["label"],
        power_w=props["power_w"],
        duration_s=props["duration_s"],
        energy_wh=energy_wh,
        budget_used_pct=budget_used_pct,
        sessions_per_budget=sessions_per_budget,
    )


def federated_communication(
    profile: EdgeDeviceProfile,
    *,
    beta: float,
    local_epochs: int,
    compression: str,
    alpha: float = 3.0,
) -> FederatedCommunicationResult:
    """Estimate non-IID federated rounds and compressed communication volume."""
    noniid_multiplier = 1 + alpha / beta
    if local_epochs <= 3:
        drift_penalty = 1.0
    elif local_epochs <= 10:
        drift_penalty = 1 + 0.15 * (local_epochs - 3)
    else:
        drift_penalty = 1 + 0.15 * 7 + 0.3 * (local_epochs - 10)

    compression_props = {
        "none": {"bytes_mult": 1.0, "quality_penalty": 1.0, "label": "None"},
        "int8": {"bytes_mult": 0.25, "quality_penalty": 1.05, "label": "INT8"},
        "int4": {"bytes_mult": 0.125, "quality_penalty": 1.15, "label": "INT4"},
        "topk": {"bytes_mult": 0.1, "quality_penalty": 1.25, "label": "Top-K"},
    }
    props = compression_props[compression]
    noniid_rounds = profile.iid_rounds * noniid_multiplier * drift_penalty
    compressed_rounds = noniid_rounds * props["quality_penalty"]
    compressed_bytes = profile.update_payload_mb * props["bytes_mult"]
    total_mb = compressed_rounds * compressed_bytes
    return FederatedCommunicationResult(
        iid_rounds=profile.iid_rounds,
        noniid_rounds=noniid_rounds,
        compressed_rounds=compressed_rounds,
        round_multiplier=noniid_rounds / profile.iid_rounds,
        bytes_per_round_mb=profile.update_payload_mb,
        compressed_bytes_per_round_mb=compressed_bytes,
        total_communication_mb=total_mb,
        compression_label=props["label"],
        drift_penalty=drift_penalty,
    )
