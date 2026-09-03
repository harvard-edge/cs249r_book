"""Model compression trade-off solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..results import (
    CompressionCandidate,
    CompressionResult,
    CompressionSweepResult,
)
from ...physics import (
    calc_bottleneck,
)
from ...core.types import Quantity
from ...core.units import Q_
from .. import calibration as cal
from ...models.types import Workload
from ...hardware.types import HardwareNode
from .base import ForwardModel

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

    def candidate(
        self,
        model: Workload,
        hardware: HardwareNode,
        *,
        label: Optional[str] = None,
        method: str = "quantization",
        target_bitwidth: int = 8,
        sparsity: float = 0.0,
        sparsity_type: str = "unstructured",
        size_limit: Optional[Quantity] = None,
        max_accuracy_drop: Optional[float] = None,
        min_speedup: Optional[float] = None,
        require_hardware_support: bool = False,
    ) -> CompressionCandidate:
        """Evaluate one compression configuration with feasibility metadata."""
        normalized_method = method.lower()
        normalized_sparsity_type = self._normalize_sparsity_type(sparsity_type)
        result = self.solve(
            model,
            hardware,
            method=normalized_method,
            target_bitwidth=target_bitwidth,
            sparsity=sparsity,
            sparsity_type=normalized_sparsity_type,
        )
        hardware_supported = self._hardware_supported(
            hardware,
            method=normalized_method,
            target_bitwidth=target_bitwidth,
            sparsity_type=normalized_sparsity_type,
        )

        violations: List[str] = []
        if size_limit is not None and result.compressed_size_gb.to("GB").magnitude > size_limit.to("GB").magnitude:
            violations.append(
                f"model_size: {result.compressed_size_gb.to('MB').magnitude:.3g} MB exceeds "
                f"{size_limit.to('MB').magnitude:.3g} MB"
            )
        if max_accuracy_drop is not None and abs(result.estimated_accuracy_delta) > max_accuracy_drop:
            violations.append(
                f"quality: accuracy drop {abs(result.estimated_accuracy_delta):.3g} exceeds "
                f"{max_accuracy_drop:.3g}"
            )
        if min_speedup is not None and result.inference_speedup < min_speedup:
            violations.append(
                f"speedup: {result.inference_speedup:.3g}x is below required {min_speedup:.3g}x"
            )
        if require_hardware_support and not hardware_supported:
            violations.append("hardware_support: no explicit fast path in the hardware profile")

        source_trace = [
            "CompressionModel.solve",
            f"model={model.name}",
            f"hardware={hardware.name}",
            f"method={normalized_method}",
            f"target_bitwidth={target_bitwidth}",
            f"sparsity={sparsity}",
            f"sparsity_type={normalized_sparsity_type}",
        ]
        if size_limit is not None:
            source_trace.append(f"size_limit={size_limit}")
        if max_accuracy_drop is not None:
            source_trace.append(f"max_accuracy_drop={max_accuracy_drop}")
        if min_speedup is not None:
            source_trace.append(f"min_speedup={min_speedup}")

        return CompressionCandidate(
            label=label or self._candidate_label(normalized_method, target_bitwidth, sparsity, normalized_sparsity_type),
            method=normalized_method,
            target_bitwidth=target_bitwidth if normalized_method == "quantization" else None,
            sparsity=sparsity,
            sparsity_type=normalized_sparsity_type,
            original_size_gb=result.original_size_gb,
            compressed_size_gb=result.compressed_size_gb,
            compression_ratio=result.compression_ratio,
            estimated_accuracy_delta=result.estimated_accuracy_delta,
            memory_savings_pct=result.memory_savings_pct,
            inference_speedup=result.inference_speedup,
            hardware_supported=hardware_supported,
            feasible=not violations,
            binding_constraint=violations[0].split(":", 1)[0] if violations else "none",
            guardrail_violations=violations,
            pareto_status="unranked",
            source_trace=source_trace,
            constraint_trace=violations or ["feasible"],
        )

    def sweep(
        self,
        model: Workload,
        hardware: HardwareNode,
        candidate_configs: List[Dict[str, Any]],
        *,
        size_limit: Optional[Quantity] = None,
        max_accuracy_drop: Optional[float] = None,
        min_speedup: Optional[float] = None,
        require_hardware_support: bool = False,
        objective: str = "min_size_max_speed_preserve_quality",
    ) -> CompressionSweepResult:
        """Evaluate a compression design space and mark Pareto candidates."""
        candidates: List[CompressionCandidate] = []
        for config in candidate_configs:
            candidate_kwargs = dict(config)
            candidate = self.candidate(
                model,
                hardware,
                size_limit=size_limit,
                max_accuracy_drop=max_accuracy_drop,
                min_speedup=min_speedup,
                require_hardware_support=require_hardware_support,
                **candidate_kwargs,
            )
            candidates.append(candidate)

        self._mark_pareto(candidates)
        frontier_labels = [candidate.label for candidate in candidates if candidate.pareto_status == "frontier"]
        dominated_labels = [candidate.label for candidate in candidates if candidate.pareto_status == "dominated"]
        feasible_frontier = [
            candidate for candidate in candidates
            if candidate.feasible and candidate.pareto_status == "frontier"
        ]
        best = None
        if feasible_frontier:
            best = min(
                feasible_frontier,
                key=lambda candidate: (
                    abs(candidate.estimated_accuracy_delta),
                    candidate.compressed_size_gb.to("GB").magnitude / max(candidate.inference_speedup, 1e-9),
                ),
            )

        return CompressionSweepResult(
            candidates=candidates,
            frontier_labels=frontier_labels,
            dominated_labels=dominated_labels,
            best_candidate_label=best.label if best else None,
            objective=objective,
            constraint_trace=[
                f"{len(candidates)} candidates evaluated",
                f"{len(frontier_labels)} candidates on Pareto frontier",
            ],
        )

    @staticmethod
    def _normalize_sparsity_type(sparsity_type: str) -> str:
        key = sparsity_type.lower().replace(":", "_").replace("-", "_")
        if key in {"2_4", "n_m", "nm"}:
            return "n_m"
        return key

    @staticmethod
    def _hardware_supported(
        hardware: HardwareNode,
        *,
        method: str,
        target_bitwidth: int,
        sparsity_type: str,
    ) -> bool:
        precision_flops = hardware.compute.precision_flops
        if method == "quantization":
            if target_bitwidth >= 16:
                return True
            precision_key = f"int{target_bitwidth}" if target_bitwidth <= 8 else f"fp{target_bitwidth}"
            return precision_key in precision_flops
        if method == "pruning":
            if sparsity_type == "structured":
                return True
            if sparsity_type == "n_m":
                return bool({"int8", "fp8", "tf32"} & set(precision_flops))
            return False
        return True

    @staticmethod
    def _candidate_label(method: str, target_bitwidth: int, sparsity: float, sparsity_type: str) -> str:
        if method == "quantization":
            if target_bitwidth >= 16:
                return f"FP{target_bitwidth} quantization"
            return f"INT{target_bitwidth} quantization"
        if method == "pruning":
            return f"{sparsity_type} pruning {sparsity:.0%}"
        return method.replace("_", " ").title()

    @classmethod
    def _mark_pareto(cls, candidates: List[CompressionCandidate]) -> None:
        for candidate in candidates:
            candidate.pareto_status = "frontier"
        for candidate in candidates:
            if any(cls._dominates(other, candidate) for other in candidates if other is not candidate):
                candidate.pareto_status = "dominated"

    @staticmethod
    def _dominates(left: CompressionCandidate, right: CompressionCandidate) -> bool:
        left_size = left.compressed_size_gb.to("GB").magnitude
        right_size = right.compressed_size_gb.to("GB").magnitude
        left_quality = 1.0 + left.estimated_accuracy_delta
        right_quality = 1.0 + right.estimated_accuracy_delta
        no_worse = (
            left_size <= right_size
            and left.inference_speedup >= right.inference_speedup
            and left_quality >= right_quality
        )
        strictly_better = (
            left_size < right_size
            or left.inference_speedup > right.inference_speedup
            or left_quality > right_quality
        )
        return no_worse and strictly_better

__all__ = [
    "CompressionModel",
]
