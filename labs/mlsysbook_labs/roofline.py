"""Shared roofline helpers for track-aware hardware acceleration labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class RooflineHardwareProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    peak_tflops: float
    bandwidth_gbs: float
    memory_capacity_gb: float
    tdp_w: float
    ridge_flop_per_byte: float
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class RooflinePoint:
    arithmetic_intensity: float
    attainable_gflops: float
    peak_gflops: float
    mfu_pct: float
    regime: str
    ridge_flop_per_byte: float


@dataclass(frozen=True)
class GemmWorkload:
    dimension: int
    precision: str
    bytes_per_element: int
    flops: float
    bytes_moved: float
    arithmetic_intensity: float


@dataclass(frozen=True)
class FusionTrafficResult:
    eager_bytes: float
    fused_bytes: float
    eager_time_us: float
    fused_time_us: float
    speedup: float


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


def hardware_roofline_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    *,
    precision: str | None = None,
) -> RooflineHardwareProfile:
    """Build a source-traced roofline profile from MLSysIM hardware."""
    peak = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops) if precision else hardware.compute.peak_flops
    peak_tflops = _quantity_to_float(peak, "TFLOPs/s", 0.0)
    bandwidth_gbs = _quantity_to_float(hardware.memory.bandwidth, "GB/s", 0.0)
    memory_capacity_gb = _quantity_to_float(hardware.memory.capacity, "GB", 0.0)
    tdp_w = _quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0)
    ridge = peak_tflops * 1000 / bandwidth_gbs if bandwidth_gbs > 0 else 0.0

    return RooflineHardwareProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        peak_tflops=peak_tflops,
        bandwidth_gbs=bandwidth_gbs,
        memory_capacity_gb=memory_capacity_gb,
        tdp_w=tdp_w,
        ridge_flop_per_byte=ridge,
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=(variant.hardware_ref, variant.model_ref),
    )


def gemm_workload(*, dimension: int, precision: str) -> GemmWorkload:
    """Return the standard square GEMM workload used in the roofline lab."""
    bytes_per_element = {"fp32": 4, "fp16": 2, "int8": 1}[precision]
    flops = 2 * dimension**3
    bytes_moved = 3 * dimension**2 * bytes_per_element
    arithmetic_intensity = flops / bytes_moved
    return GemmWorkload(
        dimension=dimension,
        precision=precision,
        bytes_per_element=bytes_per_element,
        flops=float(flops),
        bytes_moved=float(bytes_moved),
        arithmetic_intensity=arithmetic_intensity,
    )


def roofline_point(profile: RooflineHardwareProfile, arithmetic_intensity: float) -> RooflinePoint:
    """Evaluate a workload point against a hardware roofline."""
    peak_gflops = profile.peak_tflops * 1000
    attainable = min(peak_gflops, profile.bandwidth_gbs * arithmetic_intensity)
    mfu_pct = attainable / peak_gflops * 100 if peak_gflops > 0 else 0.0
    regime = "Compute-bound" if arithmetic_intensity >= profile.ridge_flop_per_byte else "Memory-bound"
    return RooflinePoint(
        arithmetic_intensity=arithmetic_intensity,
        attainable_gflops=attainable,
        peak_gflops=peak_gflops,
        mfu_pct=mfu_pct,
        regime=regime,
        ridge_flop_per_byte=profile.ridge_flop_per_byte,
    )


def fusion_traffic(
    *,
    elements: int,
    bytes_per_element: int,
    bandwidth_gbs: float,
    eager_reads: int = 3,
    eager_writes: int = 3,
    fused_reads: int = 1,
    fused_writes: int = 1,
) -> FusionTrafficResult:
    """Estimate memory traffic saved by fusing elementwise kernels."""
    tensor_bytes = elements * bytes_per_element
    eager_bytes = (eager_reads + eager_writes) * tensor_bytes
    fused_bytes = (fused_reads + fused_writes) * tensor_bytes
    eager_time_us = eager_bytes / (bandwidth_gbs * 1e9) * 1e6
    fused_time_us = fused_bytes / (bandwidth_gbs * 1e9) * 1e6
    speedup = eager_time_us / fused_time_us if fused_time_us > 0 else 1.0
    return FusionTrafficResult(
        eager_bytes=float(eager_bytes),
        fused_bytes=float(fused_bytes),
        eager_time_us=eager_time_us,
        fused_time_us=fused_time_us,
        speedup=speedup,
    )
