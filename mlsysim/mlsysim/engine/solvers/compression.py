"""Model compression trade-off solvers.

These implementations live outside ``engine.solver`` so the public import
module can stay small while domain logic remains easier to review.
"""

# ruff: noqa: F401
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Type

from ..engine import Engine, PerformanceProfile
from ..results import (
    SolverResult,
    DistributedResult,
    ReliabilityResult,
    CheckpointResult,
    SustainabilityResult,
    ServingResult,
    TrainingMemoryResult,
    ServingCapacityResult,
    MoERoutingResult,
    ContinuousBatchingResult,
    WeightStreamingResult,
    TailLatencyResult,
    EconomicsResult,
    DataResult,
    TopologyResult,
    EfficiencyResult,
    TransformationResult,
    ScalingResult,
    CompressionResult,
    SynthesisResult,
    OrchestrationResult,
    InferenceScalingResult,
    SensitivityResult,
    ResponsibleEngineeringResult,
    ParallelismOptimizerResult,
    BatchingOptimizerResult,
    PlacementOptimizerResult,
)
from ...physics import (
    calc_ring_allreduce_time,
    calc_hierarchical_allreduce_time,
    calc_all_to_all_time,
    calc_bottleneck,
    calc_mtbf_cluster,
    calc_mtbf_node,
    calc_young_daly_interval,
    calc_failure_probability,
    calc_pipeline_bubble,
)
from ...core.constants import ureg, Q_, resolve_precision
from ...infrastructure.registry import Infrastructure
from ...literature.registry import Literature
from ...systems.reliability import Reliability
from .. import calibration as cal
from ...core.types import Quantity
from ...models.types import Workload, TransformerWorkload, SparseTransformerWorkload
from ...hardware.types import HardwareNode
from ...systems.types import Fleet, NetworkFabric, Node
from ...infrastructure.types import Datacenter
from .base import BaseOptimizer, BaseResolver, BaseSolver, ForwardModel
from .utils import _inter_node_latency, _intra_node_latency

class CompressionModel(ForwardModel):
    """
    Analyzes model compression trade-offs (Accuracy vs. Efficiency).

    This model simulates the 'Compression Tax' — the accuracy degradation
    that occurs when reducing model size via quantization or pruning,
    balanced against the gains in memory footprint and inference latency.

    Literature Source:
    1. Han et al. (2015), "Deep Compression: Compressing Deep Neural Networks
       with Pruning, Trained Quantization and Huffman Coding."
    2. Gholami et al. (2021), "A Survey of Quantization Methods for
       Efficient Neural Network Inference."
    3. Blalock et al. (2020), "What is the State of Neural Network Pruning?"
    """
    requires = ("workload", "hardware")
    produces = CompressionResult

    def solve(self, model: Workload, hardware: HardwareNode, method: str = "quantization",
              target_bitwidth: int = 8, sparsity: float = 0.0,
              sparsity_type: str = "unstructured") -> CompressionResult:
        """
        Solves for compression gains and estimated accuracy impact.

        Parameters
        ----------
        model : Workload
            The model to be compressed.
        hardware : HardwareNode
            The target execution hardware.
        method : str
            The compression method ('quantization', 'pruning', 'distillation').
        target_bitwidth : int
            Target numerical precision in bits (e.g., 8 for INT8/FP8, 4 for INT4).
            At 8-bit, accuracy delta uses the FP8 estimate (near-lossless) by default.
        sparsity : float
            Target sparsity ratio (0.0 to 1.0) for pruning.
        sparsity_type : str
            Type of sparsity pattern: 'unstructured', 'structured', or 'n_m' (2:4).
            - unstructured: storage savings only, no inference speedup
            - structured: both storage and compute savings
            - n_m: hardware 2:4 sparsity with 2x speedup at 50% sparsity (Ampere+)

        Returns
        -------
        CompressionResult
            Compression metrics including memory savings, inference speedup,
            and estimated accuracy delta.

        Notes
        -----
        Conventions and branch logic:

        - Sizes are measured against an **FP32 baseline** (4 bytes/param), so
          quantization's ``compression_ratio = 32 / target_bitwidth``.
        - ``estimated_accuracy_delta`` is a signed fraction (e.g. -0.005 =
          -0.5 percentage points top-1), taken from survey medians in
          ``engine/calibration.py`` (Gholami 2021 for quantization; Blalock
          2020 for pruning, where degradation goes exponential past the
          ~50% sparsity threshold).
        - ``inference_speedup`` depends on the Roofline regime: memory-bound
          workloads gain the full compression ratio (less data to move);
          compute-bound workloads gain only what the hardware's low-precision
          path provides (1.0 when unsupported). For pruning, only structured
          and 2:4 N:M sparsity translate into compute speedup; unstructured
          sparsity saves storage only.
        - Unknown ``method`` values fall through to a no-op result
          (ratio 1.0, delta 0.0).
        """
        from ...core._validation import validate_at_least, validate_range
        validate_at_least(target_bitwidth, 1, "target_bitwidth")
        validate_range(sparsity, 0.0, 1.0, "sparsity")
        original_size = model.size_in_bytes(Q_("4 byte")) # FP32 baseline
        inference_speedup = 1.0

        if method == "quantization":
            compression_ratio = 32 / target_bitwidth
            # Source: Gholami et al. (2021), "A Survey of Quantization Methods"
            # Conservative estimates: <1% for FP8, <1% for INT8, 2-5% for INT4
            if target_bitwidth >= 16:
                # FP16/BF16/FP32: no meaningful compression from FP32 baseline
                accuracy_delta = 0.0
                compression_ratio = 32 / target_bitwidth  # 2x for FP16, 1x for FP32
            elif target_bitwidth == 8:
                # FP8/INT8: use FP8 accuracy delta (near-lossless, -0.2%)
                accuracy_delta = cal.QUANT_ACCURACY_DELTA_FP8
            elif target_bitwidth >= 4:
                accuracy_delta = cal.QUANT_ACCURACY_DELTA_INT4
            else:
                accuracy_delta = cal.QUANT_ACCURACY_DELTA_SUB_INT4   # Sub-INT4: significant degradation

            # Inference speedup depends on compute vs memory boundedness
            # Memory-bound workloads: speedup ≈ compression_ratio (less data to move)
            # Compute-bound workloads: speedup depends on hardware low-precision support
            graph = model.lower(Q_("4 byte"))  # FP32 baseline graph
            roofline = calc_bottleneck(
                graph.total_ops, graph.weight_bytes,
                hardware.compute.peak_flops, hardware.memory.bandwidth
            )
            if roofline["bottleneck"] == "Memory":
                inference_speedup = compression_ratio
            else:
                # Compute-bound: check if hardware has accelerated low-precision paths
                prec_key = f"int{target_bitwidth}" if target_bitwidth <= 8 else f"fp{target_bitwidth}"
                if prec_key in hardware.compute.precision_flops:
                    hw_speedup = (hardware.compute.precision_flops[prec_key] / hardware.compute.peak_flops).magnitude
                    inference_speedup = min(hw_speedup, compression_ratio)
                else:
                    inference_speedup = 1.0  # No hardware support → no compute speedup

        elif method == "pruning":
            compression_ratio = 1.0 / (1.0 - sparsity) if sparsity < 1.0 else 100.0
            # Source: Blalock et al. (2020), "What is the State of Neural Network Pruning?"
            # Log-linear degradation accelerates after 50% sparsity
            if sparsity <= cal.PRUNING_ACCURACY_THRESHOLD:
                accuracy_delta = cal.PRUNING_MILD_DELTA
            else:
                accuracy_delta = -cal.PRUNING_STEEP_COEFFICIENT * math.exp(sparsity * cal.PRUNING_STEEP_EXPONENT)

            # Inference speedup depends on sparsity type
            if sparsity_type == "structured":
                # Structured pruning removes entire rows/columns → direct compute savings
                inference_speedup = compression_ratio
            elif sparsity_type == "n_m":
                # N:M sparsity (2:4): hardware-accelerated 2x speedup at exactly 50% sparsity
                # Source: NVIDIA Ampere Architecture Whitepaper (2020)
                if abs(sparsity - 0.5) < 0.05:
                    inference_speedup = 2.0
                else:
                    inference_speedup = 1.0  # N:M only works at 50%
            else:
                # Unstructured: irregular access patterns → storage savings only
                inference_speedup = 1.0
        else:
            compression_ratio = 1.0
            accuracy_delta = 0.0

        compressed_size = original_size / compression_ratio

        return CompressionResult(
            original_size_gb=original_size.to("GB"),
            compressed_size_gb=compressed_size.to("GB"),
            compression_ratio=compression_ratio,
            estimated_accuracy_delta=accuracy_delta,
            memory_savings_pct=(1.0 - 1.0/compression_ratio) * 100,
            inference_speedup=inference_speedup,
        )

__all__ = [
    "CompressionModel",
]
