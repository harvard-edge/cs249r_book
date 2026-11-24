"""Model compression trade-off solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

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
from ...core.units import BITS_PER_BYTE, resolve_precision
from ...models.types import Workload
from ...hardware.types import HardwareNode
from ..accuracy_curve import AccuracyCurve
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
              sparsity_type: str = "unstructured",
              baseline_precision: str = "fp32",
              group_size: Optional[int] = None,
              scale_bits: int = 16,
              accuracy_delta: float = 0.0,
              accuracy_curve: Optional[AccuracyCurve] = None) -> CompressionResult:
        """Estimate physical compression footprint, Roofline speedup, and supplied quality delta.

        Parameters
        ----------
        model : Workload
            The model to be compressed.
        hardware : HardwareNode
            The target execution hardware.
        method : str, default: 'quantization'
            Compression method (``'quantization'``, ``'pruning'``, or ``'distillation'``).
        target_bitwidth : int, default: 8
            Positive quantized bits per weight (e.g., 8 for INT8/FP8, 4 for INT4).
        sparsity : float, default: 0.0
            Removed-weight fraction in ``[0, 1]`` for pruning.
        sparsity_type : str, default: 'unstructured'
            Sparsity pattern: ``'unstructured'`` (storage savings only), ``'structured'``
            (storage and compute savings), or ``'n_m'`` (Ampere+ 2:4 hardware sparsity,
            2x speedup at 50% sparsity).
        baseline_precision : str, default: 'fp32'
            Uncompressed precision resolved through ``core.units.PRECISION_MAP``
            (e.g., ``'fp32'`` or ``'fp16'``).
        group_size : int, optional
            Positive quantization block size ``g`` sharing one scale metadata element.
            When ``None``, scale metadata overhead is omitted (per-tensor / channel limit).
        scale_bits : int, default: 16
            Positive bitwidth ``b_scale`` of each quantization group's scale metadata
            when ``group_size`` is set.
        accuracy_delta : float, default: 0.0
            Caller-supplied empirical task quality change (e.g., ``-0.01`` for a 1%
            drop). Task accuracy depends on model, calibration data, and evaluation
            benchmark rather than bitwidth or sparsity alone, so no synthetic accuracy
            curve is invented when omitted.
        accuracy_curve : AccuracyCurve, optional
            A measured accuracy curve for this model and method. When given,
            ``estimated_accuracy_delta`` is read from the curve at
            ``target_bitwidth`` (a bitwidth curve, with quantization) or at
            ``sparsity`` (a sparsity curve, with pruning), and the result's
            ``constraint_trace`` names the curve. Cannot be combined with a
            non-zero ``accuracy_delta``.

        Returns
        -------
        CompressionResult
            Original and compressed sizes in GB, compression ratio, memory savings
            percentage, Roofline inference speedup, and the supplied ``accuracy_delta``
            (or curve-derived delta).

        Raises
        ------
        TypeError
            If ``accuracy_curve`` is not an ``AccuracyCurve``.
        ValueError
            If ``accuracy_curve`` has the wrong axis for ``method``, if
            ``method`` is neither quantization nor pruning, if the requested
            level is outside the curve's measured range, or if both a non-zero
            ``accuracy_delta`` and ``accuracy_curve`` are passed.

        See Also
        --------
        mlsysim.physics.performance.calc_bottleneck : Weight-only Roofline regime.

        Notes
        -----
        Equation: for quantization, effective bits per weight are
        ``b_eff = target_bitwidth + (scale_bits / group_size if group_size else 0)``
        and ``compression_ratio = baseline_bits / b_eff``. For pruning,
        ``compression_ratio = 1 / (1 - sparsity)`` (capped at 100x at ``sparsity == 1``).
        ``compressed_bytes = original_bytes / compression_ratio``.

        Assumptions: baseline weight-only Roofline regime determines whether quantization
        speedup follows bandwidth savings (``compression_ratio``) or low-precision Tensor
        Core peak ratios; structured and 2:4 ``n_m`` pruning convert sparsity into compute
        speedup while unstructured pruning saves storage only. ``estimated_accuracy_delta``
        echoes ``accuracy_delta`` directly unless ``accuracy_curve`` is supplied, in which
        case the curve is taken to describe this model, method, and task (the solver cannot
        verify workload identity).

        Excludes: sparse index metadata (CSR/bitmask overhead), dequantization kernel
        overhead, activation quantization, and unmeasured task-accuracy prediction.

        References
        ----------
        Derivation: bit-packing storage divides element count by effective bits per
        weight including ``scale_bits / group_size`` block scales; Roofline speedup
        compares weight-bandwidth savings against precision FLOP/s ceilings.

        Examples
        --------
        >>> from mlsysim import Models, Hardware
        >>> result = CompressionModel().solve(Models.Language.Llama3_8B,
        ...     Hardware.Cloud.H100, target_bitwidth=8, baseline_precision="fp16")
        >>> result.compression_ratio == 2
        True
        """
        from ...core._validation import validate_at_least, validate_range
        validate_at_least(target_bitwidth, 1, "target_bitwidth")
        validate_range(sparsity, 0.0, 1.0, "sparsity")
        validate_at_least(scale_bits, 1, "scale_bits")
        if group_size is not None:
            validate_at_least(group_size, 1, "group_size")

        baseline_precision, baseline_bytes = resolve_precision(baseline_precision)
        baseline_bits = baseline_bytes.to("byte").magnitude * BITS_PER_BYTE
        original_size = model.size_in_bytes(baseline_bytes)
        inference_speedup = 1.0
        reported_delta = float(accuracy_delta)

        if method == "quantization":
            effective_bits = float(target_bitwidth) + (
                float(scale_bits) / float(group_size) if group_size is not None else 0.0
            )
            compression_ratio = baseline_bits / effective_bits

            graph = model.lower(baseline_bytes)
            roofline = calc_bottleneck(
                graph.total_ops, graph.weight_bytes,
                hardware.compute.peak_flops, hardware.memory.bandwidth
            )
            if roofline["bottleneck"] == "Memory":
                inference_speedup = compression_ratio
            else:
                prec_key = f"int{target_bitwidth}" if target_bitwidth <= 8 else f"fp{target_bitwidth}"
                if prec_key in hardware.compute.precision_flops:
                    hw_speedup = (
                        hardware.compute.precision_flops[prec_key] / hardware.compute.peak_flops
                    ).to_base_units().magnitude
                    inference_speedup = min(hw_speedup, compression_ratio)
                else:
                    inference_speedup = 1.0

        elif method == "pruning":
            compression_ratio = 1.0 / (1.0 - sparsity) if sparsity < 1.0 else 100.0
            if sparsity_type == "structured":
                inference_speedup = compression_ratio
            elif sparsity_type == "n_m":
                if abs(sparsity - 0.5) < 0.05:
                    inference_speedup = 2.0
                else:
                    inference_speedup = 1.0
            else:
                inference_speedup = 1.0
        else:
            compression_ratio = 1.0
            reported_delta = float(accuracy_delta)

        # The trace is set only with a curve, so a result without one is unchanged in every field.
        curve_fields = {}
        if accuracy_curve is not None:
            if accuracy_delta != 0.0:
                raise ValueError("pass either accuracy_delta or accuracy_curve, not both")
            reported_delta, accuracy_trace = self._curve_delta(accuracy_curve, method, target_bitwidth, sparsity)
            curve_fields["constraint_trace"] = accuracy_trace

        compressed_size = original_size / compression_ratio

        return CompressionResult(
            baseline_precision=baseline_precision,
            original_size_gb=original_size.to("GB"),
            compressed_size_gb=compressed_size.to("GB"),
            compression_ratio=compression_ratio,
            estimated_accuracy_delta=reported_delta,
            memory_savings_pct=(1.0 - 1.0 / compression_ratio) * 100,
            inference_speedup=inference_speedup,
            **curve_fields,
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
        baseline_precision: str = "fp32",
        group_size: Optional[int] = None,
        scale_bits: int = 16,
        accuracy_delta: float = 0.0,
        accuracy_curve: Optional[AccuracyCurve] = None,
    ) -> CompressionCandidate:
        """Apply declared size, quality, speed, and support constraints to a candidate.

        Parameters
        ----------
        model : Workload
            Workload specification supplying parameter count and a lowered graph.
        hardware : HardwareNode
            Device specification supplying memory bandwidth and precision compute rates.
        label : str, optional
            Display name; defaults to a name derived from the compression configuration.
        method : str, optional
            Compression branch, ``quantization`` or ``pruning``; other names are no-ops.
        target_bitwidth : int, optional
            Positive target bits per weight for quantization; default eight.
        sparsity : float, optional
            Removed-weight fraction in [0, 1] for pruning; default zero.
        sparsity_type : str, optional
            Pruning pattern: unstructured, structured, or N:M; common 2:4 spellings
            normalize to ``n_m``. Default unstructured.
        size_limit : Quantity, optional
            Maximum compressed checkpoint size, convertible to bytes.
        max_accuracy_drop : float, optional
            Maximum accuracy drop as a fraction, not percent. A gain never violates it.
        min_speedup : float, optional
            Minimum dimensionless modeled speed ratio relative to baseline precision.
        require_hardware_support : bool, optional
            Require the registry's precision/sparsity support heuristic; default False.
        baseline_precision : str, optional
            Uncompressed precision key; default fp32. All ratios use this baseline.
        group_size : int, optional
            Positive quantization block size sharing one scale metadata element.
        scale_bits : int, default: 16
            Bitwidth of each quantization group's scale metadata when ``group_size`` is set.
        accuracy_delta : float, default: 0.0
            Caller-supplied empirical task quality change passed to ``solve``.
        accuracy_curve : AccuracyCurve, optional
            Measured accuracy curve passed to ``solve``. The ``max_accuracy_drop``
            gate then tests the curve's value, and ``source_trace`` names the curve.

        Returns
        -------
        CompressionCandidate
            Sizes in GB, speed/storage ratios, supplied quality delta, support status,
            and constraint violations. Feasible means these requested gates pass;
            it does not certify runtime memory residency or deployment performance.

        See Also
        --------
        CompressionModel.solve : Compression estimates and their assumptions.

        Notes
        -----
        Equation: feasible = all(requested limits pass), where size <= size_limit,
        max(0, -quality_delta) <= max_accuracy_drop, speedup >= min_speedup, and
        optional hardware support. Missing limits impose no condition.

        Assumptions: the estimates returned by ``solve`` apply unchanged. Hardware
        support is inferred from registry keys and pattern names, not from a kernel
        execution. Constraint checks use the same baseline for every metric.

        Excludes: validation of task accuracy, sparse-kernel availability, workspace,
        and the quality of an actually compressed checkpoint.

        References
        ----------
        Derivation: candidate admission is the conjunction of user-supplied scalar
        bounds on the compression model outputs; the first failed check is reported
        as binding while all violations remain in the result.

        Examples
        --------
        >>> from mlsysim import Models, Hardware, Q_
        >>> result = CompressionModel().candidate(Models.Language.Llama3_8B,
        ...     Hardware.Cloud.H100, target_bitwidth=8, size_limit=Q_("1 byte"))
        >>> result.feasible
        False
        >>> result.binding_constraint
        'model_size'
        """
        normalized_method = method.lower()
        normalized_sparsity_type = self._normalize_sparsity_type(sparsity_type)
        result = self.solve(
            model,
            hardware,
            method=normalized_method,
            target_bitwidth=target_bitwidth,
            sparsity=sparsity,
            sparsity_type=normalized_sparsity_type,
            baseline_precision=baseline_precision,
            group_size=group_size,
            scale_bits=scale_bits,
            accuracy_delta=accuracy_delta,
            accuracy_curve=accuracy_curve,
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
        # A measured curve can sit above its baseline (a positive delta), and a gain is not a drop.
        accuracy_drop = max(0.0, -result.estimated_accuracy_delta)
        if max_accuracy_drop is not None and accuracy_drop > max_accuracy_drop:
            violations.append(
                f"quality: accuracy drop {accuracy_drop:.3g} exceeds "
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
            f"baseline_precision={result.baseline_precision}",
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
        if accuracy_curve is not None:
            source_trace.append(f"accuracy_curve={accuracy_curve.name}")

        return CompressionCandidate(
            label=label or self._candidate_label(normalized_method, target_bitwidth, sparsity, normalized_sparsity_type),
            method=normalized_method,
            target_bitwidth=target_bitwidth if normalized_method == "quantization" else None,
            sparsity=sparsity,
            sparsity_type=normalized_sparsity_type,
            baseline_precision=result.baseline_precision,
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
        baseline_precision: str = "fp32",
    ) -> CompressionSweepResult:
        """Evaluate compression configurations and identify nondominated candidates.

        Parameters
        ----------
        model : Workload
            Common workload evaluated for every candidate.
        hardware : HardwareNode
            Common device profile evaluated for every candidate.
        candidate_configs : list of dict
            Candidate keyword dictionaries accepted by ``candidate``, such as method,
            target_bitwidth, sparsity, sparsity_type, accuracy_curve, and label. Do
            not repeat the shared constraints or baseline passed separately here.
        size_limit : Quantity, optional
            Maximum checkpoint size for each candidate, convertible to bytes.
        max_accuracy_drop : float, optional
            Maximum accuracy drop as a fraction, not percent. A gain never violates it.
        min_speedup : float, optional
            Minimum modeled speed ratio relative to the common baseline.
        require_hardware_support : bool, optional
            Require the candidate support heuristic; default False.
        objective : str, optional
            Descriptive result label only; changing it does not change the selection
            algorithm. Default ``min_size_max_speed_preserve_quality``.
        baseline_precision : str, optional
            Common uncompressed precision key; default fp32.

        Returns
        -------
        CompressionSweepResult
            Candidate records, frontier/dominated labels, and a selected label among
            feasible frontier candidates, or None when none exist. Sizes are in GB;
            quality and speed comparisons are dimensionless modeled quantities.

        See Also
        --------
        CompressionModel.candidate : Constraint checks and support heuristics.

        Notes
        -----
        Equation: A dominates B if A has no larger size, no lower speed, and no lower
        quality (1 + accuracy_delta), with at least one strict improvement. Among
        feasible frontier entries, selection minimizes the tuple
        (max(0, -accuracy_delta), size/max(speedup, 1e-9)).

        Assumptions: each candidate uses the same workload, hardware, and baseline.
        Dominance is evaluated across all candidates, including infeasible ones;
        feasibility is applied afterward when choosing the returned best label.

        Excludes: global optimization outside the supplied candidate list, measured
        accuracy/performance, uncertainty intervals, and runtime memory feasibility.

        References
        ----------
        Derivation: Pareto dominance is the componentwise ordering of storage cost,
        speed benefit, and assumed quality. The lexicographic tie-break is an explicit
        implementation policy, not a learned or universally optimal preference.

        Examples
        --------
        >>> from mlsysim import Models, Hardware
        >>> result = CompressionModel().sweep(Models.Language.Llama3_8B,
        ...     Hardware.Cloud.H100, [{"target_bitwidth": 8, "label": "int8"}])
        >>> result.frontier_labels
        ['int8']
        """
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
                baseline_precision=baseline_precision,
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
                # Rank by accuracy drop, so a measured gain counts as no loss. Default deltas are never positive.
                key=lambda candidate: (
                    max(0.0, -candidate.estimated_accuracy_delta),
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
    def _curve_delta(curve: AccuracyCurve, method: str, target_bitwidth: int, sparsity: float):
        """Read the accuracy delta for one configuration from a measured curve."""
        if not isinstance(curve, AccuracyCurve):
            raise TypeError(
                f"accuracy_curve must be an AccuracyCurve, got {type(curve).__name__}. "
                "Load a shipped curve with mlsysim.engine.accuracy_curve.shipped_curve(name)."
            )
        axis = {"quantization": "bitwidth", "pruning": "sparsity"}.get(method)
        if axis is None:
            raise ValueError(f"accuracy_curve applies to quantization or pruning, got method={method!r}")
        if curve.axis != axis:
            raise ValueError(
                f"accuracy curve {curve.name!r} is a {curve.axis} curve, and {method} needs a {axis} curve"
            )
        level = float(target_bitwidth) if axis == "bitwidth" else float(sparsity)
        delta = curve.delta_at(level)
        how = "measured point" if curve.is_measured(level) else "interpolated between measured points"
        return delta, [f"accuracy from curve {curve.name}: {how} at {axis} {level:g}"]

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
