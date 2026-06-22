"""LLM serving, batching, tail-latency, and inference-scaling solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

import math
from typing import Optional

from ..results import (
    ServingResult,
    ServingCapacityResult,
    ContinuousBatchingResult,
    WeightStreamingResult,
    TailLatencyResult,
    InferenceScalingResult,
    BatchingOptimizerResult,
)
from ...core.units import ureg, Q_, resolve_precision
from .. import calibration as cal
from ...core.types import Quantity
from ...models.types import TransformerWorkload
from ...hardware.types import HardwareNode
from .base import BaseOptimizer, ForwardModel

class ServingModel(ForwardModel):
    """
    Analyzes the two-phase LLM serving lifecycle: Pre-fill vs. Decoding.

    LLM inference is not a single mathematical operation; it is a stateful
    process with two distinct physical regimes (Compute-bound Pre-fill and
    Memory-bound Decoding).

    Literature Source:
    1. Pope et al. (2023), "Efficiently Scaling Transformer Inference."
    2. Agrawal et al. (2024), "Sarathi-Serve" (chunked prefill scheduling).
    3. Patel et al. (2024), "Splitwise" and Zhong et al. (2024),
       "DistServe" (prefill/decode disaggregation).

    Formula contract:
    - TTFT is prefill compute over new prompt tokens plus dispatch overhead and,
      for disaggregated serving, KV-transfer time.
    - ITL is the decode lower bound: model weights plus KV cache streamed from
      decode memory bandwidth, plus the configured framework layer tax.
    - KV memory always covers the full sequence, even when prompt caching skips
      part of the prefill computation.
    """
    requires = ("workload", "hardware")
    produces = ServingResult

    def solve(self, model: TransformerWorkload, hardware: HardwareNode, seq_len: int, batch_size: int = 1, precision: str = "fp16", efficiency: float = 0.5,
              decode_hardware: Optional[HardwareNode] = None, network_bandwidth: Quantity = Q_("100 GB/s"),
              draft_model: Optional[TransformerWorkload] = None, draft_acceptance_rate: float = 0.7,
              cached_prefix_len: int = 0, prefill_chunk_tokens: Optional[int] = None) -> ServingResult:
        """
        Solves for LLM serving performance.

        Parameters
        ----------
        model : TransformerWorkload
            The primary model to be served.
        hardware : HardwareNode
            The hardware node for serving (or pre-fill node in disaggregated serving).
        seq_len : int
            Sequence length (context window).
        batch_size : int
            Batch size.
        precision : str
            Numerical precision.
        efficiency : float
            Compute efficiency.
        decode_hardware : HardwareNode, optional
            If provided, models Disaggregated Serving where 'hardware' does pre-fill
            and 'decode_hardware' does decoding. KV-cache is transferred over the network.
        network_bandwidth : Quantity
            Network bandwidth between pre-fill and decode nodes.
        draft_model : TransformerWorkload, optional
            If provided, models Speculative Decoding using this smaller draft model.
        draft_acceptance_rate : float
            Expected acceptance rate (0.0 to 1.0) of draft tokens per step.
        cached_prefix_len : int
            Number of tokens with pre-computed KV-cache (prompt caching / prefix caching).
            When > 0, the prefill phase only processes (seq_len - cached_prefix_len) new
            tokens, reducing TTFT proportionally. The full KV-cache (including cached prefix)
            still occupies memory. Must be < seq_len.
        prefill_chunk_tokens : int, optional
            If provided, split new prefill tokens into chunks of at most this size. This estimates
            a Sarathi-Serve-style chunked-prefill stall proxy: total TTFT keeps the same compute
            work plus one dispatch tax per chunk, while decode_stall_bound reports the slowest
            single chunk that can interfere with ongoing decode iterations. It is not a full
            scheduler simulation.

        Returns
        -------
        ServingResult
            Serving performance metrics.
        """
        precision, bpp = resolve_precision(precision)

        # 0. Input validation
        if seq_len < 1:
            raise ValueError(f"seq_len ({seq_len}) must be >= 1")
        if batch_size < 1:
            raise ValueError(f"batch_size ({batch_size}) must be >= 1")
        if efficiency <= 0.0 or efficiency > 1.0:
            raise ValueError(f"efficiency ({efficiency}) must be > 0 and <= 1")
        if cached_prefix_len < 0:
            raise ValueError(f"cached_prefix_len ({cached_prefix_len}) must be >= 0")
        if cached_prefix_len >= seq_len:
            raise ValueError(f"cached_prefix_len ({cached_prefix_len}) must be < seq_len ({seq_len})")
        if draft_acceptance_rate < 0.0 or draft_acceptance_rate > 1.0:
            raise ValueError(f"draft_acceptance_rate ({draft_acceptance_rate}) must be between 0 and 1")
        if prefill_chunk_tokens is not None and prefill_chunk_tokens <= 0:
            raise ValueError(f"prefill_chunk_tokens ({prefill_chunk_tokens}) must be positive")
        if network_bandwidth.to("B/s").magnitude <= 0:
            raise ValueError(f"network_bandwidth ({network_bandwidth:~P}) must be positive")

        # 1. Pre-fill Phase (with optional prompt caching)
        # Prefer the precision-specific FLOP rate (e.g. FP16 tensor cores); fall back
        # to the generic peak when the registry has no entry for this dtype.
        peak_flops_prefill = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)
        # Prompt caching: only new tokens need prefill computation. The max(1, ...)
        # floor keeps the model well-defined even when nearly the whole prompt is
        # cached — at least one token must pass through prefill to produce output.
        new_tokens = max(1, seq_len - cached_prefix_len)
        # Linear layer FLOPs: 2 * params * tokens * batch (standard 2P approximation)
        linear_flops = 2 * model.parameters.to(ureg.count).magnitude * new_tokens * batch_size
        # Attention FLOPs: 2 * n_layers * n_heads * head_dim * seq_len^2 * batch_size
        # This captures the O(S^2) self-attention cost, which dominates for long contexts.
        # getattr-with-fallback tolerates plain Workloads that lack transformer
        # geometry; the `or` guards convert an explicit None into the same default.
        n_layers = getattr(model, 'layers', 1) or 1
        n_heads = getattr(model, 'heads', 32) or 32
        head_dim = (getattr(model, 'hidden_dim', 4096) or 4096) // n_heads
        # Upper-bound approximation: each new token attends over the final context.
        # The leading 4 = 2 FLOPs/MAC x 2 attention matmuls (QK^T scores and
        # attention-weighted V). When cached_prefix_len=0, new_tokens == seq_len
        # so this simplifies to S^2.
        attention_flops = 4 * n_layers * n_heads * head_dim * new_tokens * seq_len * batch_size
        prefill_ops = (linear_flops + attention_flops) * ureg.flop
        t_prefill_compute = (prefill_ops / (peak_flops_prefill * efficiency)).to("ms")
        prefill_chunks = 1
        prefill_chunk_time: Optional[Quantity] = None
        decode_stall_bound: Optional[Quantity] = None
        if prefill_chunk_tokens is not None:
            # Chunked prefill (Sarathi-Serve): split the prompt into bounded chunks
            # so prefill work can interleave with ongoing decode iterations. Total
            # compute is unchanged; what we gain is a bound on how long any single
            # chunk can stall a decode step.
            prefill_chunks = max(1, math.ceil(new_tokens / prefill_chunk_tokens))
            chunk_times = []
            tokens_remaining = new_tokens
            while tokens_remaining > 0:
                chunk_tokens = min(prefill_chunk_tokens, tokens_remaining)
                chunk_linear = 2 * model.parameters.to(ureg.count).magnitude * chunk_tokens * batch_size
                # Conservative: every chunk's tokens attend over the full final
                # context (seq_len), matching the upper-bound convention above.
                chunk_attention = 4 * n_layers * n_heads * head_dim * chunk_tokens * seq_len * batch_size
                chunk_ops = (chunk_linear + chunk_attention) * ureg.flop
                chunk_time = (chunk_ops / (peak_flops_prefill * efficiency)).to("ms") + hardware.dispatch_tax
                chunk_times.append(chunk_time)
                tokens_remaining -= chunk_tokens
            # The slowest chunk is the worst-case decode interference window.
            prefill_chunk_time = max(chunk_times, key=lambda t: t.to("ms").magnitude)
            decode_stall_bound = prefill_chunk_time
        # One kernel-dispatch tax per chunk: chunking trades extra launch overhead
        # for bounded decode stalls (prefill_chunks == 1 when chunking is off).
        t_prefill = t_prefill_compute + (hardware.dispatch_tax * prefill_chunks)

        # KV-cache covers the full sequence (cached prefix + new tokens)
        kv_cache_bytes = model.get_kv_cache_size(seq_len=seq_len, batch_size=batch_size, precision=bpp)

        # 2. Disaggregated Serving (KV-Cache Transfer)
        # Splitwise/DistServe style: prefill and decode run on different nodes, so
        # the prompt's KV cache must cross the network before decoding can start.
        # That transfer lands on TTFT (the first token waits for it); decode then
        # reads from the decode node's own HBM.
        if decode_hardware:
            t_transfer = (kv_cache_bytes / network_bandwidth).to("ms")
            t_prefill += t_transfer
            decode_hw = decode_hardware
        else:
            decode_hw = hardware

        # 3. Decode Phase
        # Per decode step: load weights once (W) + load KV cache for all B requests.
        # kv_cache_bytes already includes batch_size from get_kv_cache_size().
        # Per-token ITL = step_time (each step produces 1 token per request).
        model_weights_bytes = model.size_in_bytes(bpp)
        t_decode_per_token = ((model_weights_bytes + kv_cache_bytes) / decode_hw.memory.bandwidth).to("ms")

        # 4. Framework Tax (Per-token decode also incurs launch overhead)
        from ..calibration import FRAMEWORK_LAYER_TAX_MS
        layer_tax = Q_(model.layers * FRAMEWORK_LAYER_TAX_MS, "ms")
        t_decode_per_token += layer_tax

        # 5. Speculative Decoding
        # A small draft model proposes K tokens cheaply; the target model verifies
        # them in one parallel pass. The win comes from amortizing the target's
        # memory-bound weight load over several accepted tokens per step.
        if draft_model:
            # Time to generate 1 token with draft model (uses draft model's own KV cache, not target's)
            draft_weights_bytes = draft_model.size_in_bytes(bpp)
            draft_kv_cache = draft_model.get_kv_cache_size(seq_len=seq_len, batch_size=batch_size, precision=bpp)
            t_draft_token = ((draft_weights_bytes + draft_kv_cache) / decode_hw.memory.bandwidth).to("ms")

            # Speculative decoding batches K draft tokens (e.g., K=4)
            K = cal.SPECULATIVE_GAMMA
            t_draft_phase = t_draft_token * K

            # Verification phase: Target model verifies K tokens in parallel (compute-bound or memory-bound)
            # Simplification: verifying K tokens is roughly 1 forward pass of target model + small compute overhead
            t_verify = t_decode_per_token # essentially memory bound by loading target weights once

            # Expected tokens per step via geometric series (Leviathan et al., 2023):
            # E[accepted] = alpha*(1 - alpha^K)/(1 - alpha), plus 1 bonus token
            alpha = draft_acceptance_rate
            if alpha < 1.0:
                expected_tokens = 1 + alpha * (1 - alpha**K) / (1 - alpha)
            else:
                expected_tokens = 1 + K  # perfect acceptance

            # Effective ITL
            t_decode_per_token = (t_draft_phase + t_verify) / expected_tokens

        # Memory-wall feasibility on the decode node: weights + full-sequence KV
        # (+ draft weights when speculating — the draft lives alongside the target).
        total_memory_required = model_weights_bytes + kv_cache_bytes
        if draft_model:
            total_memory_required += draft_model.size_in_bytes(bpp)
        feasible = total_memory_required <= decode_hw.memory.capacity

        constraint_trace = []
        if feasible:
            constraint_trace.append(f"Memory Wall: Passed. Required {total_memory_required.to('GB'):~P} (Weights: {model_weights_bytes.to('GB'):~P}, KV Cache: {kv_cache_bytes.to('GB'):~P}) <= Available {decode_hw.memory.capacity.to('GB'):~P} on {decode_hw.name}.")
        else:
            constraint_trace.append(f"Memory Wall: FAILED. Required {total_memory_required.to('GB'):~P} (Weights: {model_weights_bytes.to('GB'):~P}, KV Cache: {kv_cache_bytes.to('GB'):~P}) > Available {decode_hw.memory.capacity.to('GB'):~P} on {decode_hw.name}.")
        if prefill_chunk_tokens is not None and decode_stall_bound is not None:
            constraint_trace.append(
                f"Serving Wall: Chunked prefill uses {prefill_chunks} chunks of up to {prefill_chunk_tokens} tokens; "
                f"decode interference is bounded by {decode_stall_bound.to('ms'):~P} per prefill chunk."
            )

        cache_hit_ratio = cached_prefix_len / seq_len if seq_len > 0 else 0.0

        return ServingResult(
            feasible=feasible,
            constraint_trace=constraint_trace,
            ttft=t_prefill,
            itl=t_decode_per_token,
            kv_cache_size=kv_cache_bytes.to("GB"),
            model_weights_size=model_weights_bytes.to("GB"),
            total_memory_required=total_memory_required.to("GB"),
            memory_utilization=(total_memory_required / decode_hw.memory.capacity).to_base_units().magnitude,
            prompt_cache_hit_ratio=cache_hit_ratio,
            prefill_chunks=prefill_chunks,
            prefill_chunk_time=prefill_chunk_time,
            decode_stall_bound=decode_stall_bound,
        )

class ServingCapacityModel(ForwardModel):
    """
    Sizes an LLM serving deployment from a QPS and tail-latency target.

    The model deliberately composes existing first-order pieces: ``ServingModel``
    for TTFT/ITL, ``ContinuousBatchingModel`` for per-replica token capacity,
    and ``TailLatencyModel`` for queueing pressure. It is a capacity planner,
    not a request-level scheduler.
    """
    requires = ("workload", "hardware")
    produces = ServingCapacityResult

    def solve(
        self,
        model: TransformerWorkload,
        hardware: HardwareNode,
        qps: float,
        target_p99_latency_ms: float,
        seq_len: int = 2048,
        output_tokens: int = 128,
        max_batch_size: int = 32,
        precision: str = "fp16",
        efficiency: float = 0.5,
        max_replicas: int = 1024,
        service_time_cv: float = 1.0,
    ) -> ServingCapacityResult:
        """Return the minimum replica count that satisfies the target P99.

        Composes three first-order models and then scans replica counts
        upward (1, 2, ...) until the estimated request P99 meets the target:

        1. ``ContinuousBatchingModel`` sizes the per-replica active batch and
           token throughput under the memory wall;
        2. ``ServingModel`` (at that batch) gives TTFT/ITL, so the base
           request latency is ``TTFT + ITL * output_tokens``;
        3. ``TailLatencyModel`` (M/M/c) adds the P99 queue wait at each
           candidate replica count.

        Estimated P99 = base request latency + P99 queue wait. Per-replica
        QPS capacity = token throughput / output_tokens.

        Parameters
        ----------
        model : TransformerWorkload
            The model being served.
        hardware : HardwareNode
            The per-replica accelerator.
        qps : float
            Target aggregate arrival rate in queries per second (> 0).
        target_p99_latency_ms : float
            End-to-end P99 latency target in milliseconds.
        seq_len : int
            Input context length in tokens.
        output_tokens : int
            Generated tokens per request; converts token throughput to QPS.
        max_batch_size : int
            Cap on concurrent requests per replica.
        precision : str
            Numerical precision for weights and KV cache.
        efficiency : float
            Software efficiency factor in (0, 1].
        max_replicas : int
            Search cap; reported as the requirement when infeasible.
        service_time_cv : float
            Coefficient of variation of service time (1.0 = M/M/c).

        Returns
        -------
        ServingCapacityResult
            Replica count, capacity/utilization, latency decomposition, and
            the limiting ``bottleneck`` ('Memory', 'Capacity', 'Tail Latency',
            'Base Latency', or 'Feasible').
        """
        from ...core._validation import validate_at_least, validate_nonnegative, validate_positive, validate_range

        validate_positive(qps, "qps")
        validate_positive(target_p99_latency_ms, "target_p99_latency_ms")
        validate_at_least(seq_len, 1, "seq_len")
        validate_at_least(output_tokens, 1, "output_tokens")
        validate_at_least(max_batch_size, 1, "max_batch_size")
        validate_at_least(max_replicas, 1, "max_replicas")
        validate_range(efficiency, 1e-9, 1.0, "efficiency")
        validate_nonnegative(service_time_cv, "service_time_cv")

        batching = ContinuousBatchingModel().solve(
            model=model,
            hardware=hardware,
            seq_len=seq_len,
            max_batch_size=max_batch_size,
            precision=precision,
            efficiency=efficiency,
        )
        active_batch = max(1, batching.max_active_requests)
        serving = ServingModel().solve(
            model=model,
            hardware=hardware,
            seq_len=seq_len,
            batch_size=active_batch,
            precision=precision,
            efficiency=efficiency,
        )
        # A request "costs" output_tokens decode steps, so replica QPS capacity is
        # token throughput spread over the per-request token budget.
        per_replica_qps = batching.throughput_tokens_per_sec / output_tokens if output_tokens > 0 else 0.0
        # Queue-free request latency: one prefill plus output_tokens decode steps.
        # Queue wait is layered on top of this per candidate replica count below.
        base_request_latency = (serving.ttft + serving.itl * output_tokens).to("ms")

        if not serving.feasible or not batching.feasible or per_replica_qps <= 0:
            bottleneck = "Memory" if not serving.feasible or not batching.feasible else "Capacity"
            return ServingCapacityResult(
                feasible=False,
                constraint_trace=[
                    "Serving Capacity: infeasible before queueing because the model does not fit or has no token capacity."
                ],
                required_replicas=max_replicas,
                qps_target=qps,
                qps_capacity=0.0,
                per_replica_qps_capacity=0.0,
                utilization=float("inf"),
                target_p99_latency=Q_(target_p99_latency_ms, "ms"),
                estimated_p99_latency=Q_(float("inf"), "ms"),
                base_request_latency=base_request_latency,
                queue_wait_p99=Q_(float("inf"), "ms"),
                active_batch_size=active_batch,
                ttft=serving.ttft,
                itl=serving.itl,
                bottleneck=bottleneck,
            )

        # The queueing model's "service time" is the replica's capacity interval
        # (1/QPS-capacity in ms): how long a replica is occupied per request from
        # the scheduler's point of view, not the request's wall-clock latency.
        capacity_service_ms = 1000.0 / per_replica_qps
        best_tail: Optional[TailLatencyResult] = None
        best_queue_wait = Q_(float("inf"), "ms")
        best_p99 = Q_(float("inf"), "ms")
        required_replicas = max_replicas
        feasible = False

        # Scan replica counts upward; queue wait shrinks monotonically with more
        # servers, so the first count that meets the target is the minimum.
        for replicas in range(1, max_replicas + 1):
            tail = TailLatencyModel().solve(
                arrival_rate_qps=qps,
                service_latency_ms=capacity_service_ms,
                num_replicas=replicas,
                service_time_cv=service_time_cv,
            )
            # TailLatencyModel folds the service time into its P99; strip it back
            # out to isolate pure queue wait, which we then add to the *request's*
            # real latency (TTFT + decode), a different quantity from the
            # capacity interval used inside the queueing model.
            queue_wait = max(0.0, tail.p99_latency.to("ms").magnitude - capacity_service_ms) * ureg.ms
            estimated_p99 = (base_request_latency + queue_wait).to("ms")
            best_tail = tail
            best_queue_wait = queue_wait
            best_p99 = estimated_p99
            if tail.is_stable and estimated_p99.magnitude <= target_p99_latency_ms:
                required_replicas = replicas
                feasible = True
                break

        qps_capacity = required_replicas * per_replica_qps
        utilization = qps / qps_capacity if qps_capacity > 0 else float("inf")
        # Name what binds: an unstable queue at max_replicas means raw capacity is
        # short; a stable queue that still misses P99 means tail latency binds; a
        # base latency already over target means no replica count can ever help
        # (the fix is a faster model/hardware, not more replicas).
        if not feasible:
            bottleneck = "Tail Latency" if best_tail and best_tail.is_stable else "Capacity"
        elif base_request_latency.magnitude > target_p99_latency_ms:
            bottleneck = "Base Latency"
        else:
            bottleneck = "Feasible"

        return ServingCapacityResult(
            feasible=feasible,
            constraint_trace=[
                (
                    f"Serving Capacity: {required_replicas} replicas provide {qps_capacity:.2f} QPS "
                    f"for target {qps:.2f} QPS at utilization {utilization:.2%}."
                ),
                (
                    f"Tail Latency: base request latency {base_request_latency:~P}; "
                    f"estimated P99 {best_p99:~P} vs target {Q_(target_p99_latency_ms, 'ms'):~P}."
                ),
            ],
            required_replicas=required_replicas,
            qps_target=qps,
            qps_capacity=qps_capacity,
            per_replica_qps_capacity=per_replica_qps,
            utilization=utilization,
            target_p99_latency=Q_(target_p99_latency_ms, "ms"),
            estimated_p99_latency=best_p99,
            base_request_latency=base_request_latency,
            queue_wait_p99=best_queue_wait,
            active_batch_size=active_batch,
            ttft=serving.ttft,
            itl=serving.itl,
            bottleneck=bottleneck,
        )

class ContinuousBatchingModel(ForwardModel):
    """
    Analyzes production LLM serving with Continuous Batching and PagedAttention.

    Traditional static batching suffers from severe memory fragmentation and
    padding waste. This model simulates the throughput improvements achieved by
    iteration-level scheduling and non-contiguous KV cache allocation.

    Literature Source:
    1. Kwon et al. (2023), "Efficient Memory Management for Large Language Model
       Serving with PagedAttention."
    2. Yu et al. (2022), "ORCA: A Distributed Serving System for
       Transformer-Based Generative Models."
    """
    requires = ("workload", "hardware")
    produces = ContinuousBatchingResult

    def solve(self, model: TransformerWorkload, hardware: HardwareNode, seq_len: int, max_batch_size: int = 1, page_size: int = 16, precision: str = "fp16", efficiency: float = 0.5) -> ContinuousBatchingResult:
        """Calculate continuous-batching throughput and PagedAttention memory.

        Models a decode-dominated serving replica:

        - **Capacity**: KV memory per sequence comes from
          ``calc_paged_kv_cache_size`` (page-granular allocation, so internal
          fragmentation is bounded by one page per sequence); the number of
          concurrent requests is what fits in HBM after the weights.
        - **Latency**: TTFT is the prefill compute time (linear ``2*P*S`` plus
          attention ``4*L*H*d*S^2`` FLOPs) plus the dispatch tax; ITL is
          memory-bound — (weights + total KV) streamed once per token over
          HBM bandwidth.
        - **Speedup vs static batching**: static allocation is modeled as
          reaching only ~60% of the continuous batch size, reflecting the
          20-40% external fragmentation measured by Kwon et al. (2023).

        Parameters
        ----------
        model : TransformerWorkload
            The model being served (layers, heads, hidden_dim, kv_heads).
        hardware : HardwareNode
            The accelerator (HBM capacity/bandwidth, dispatch tax).
        seq_len : int
            Sequence length in tokens (context for KV sizing and prefill).
        max_batch_size : int
            Scheduler cap on concurrent requests.
        page_size : int
            PagedAttention page size in tokens (vLLM default 16).
        precision : str
            Numerical precision for weights and KV cache.
        efficiency : float
            Software efficiency factor in (0, 1] applied to prefill FLOPS.

        Returns
        -------
        ContinuousBatchingResult
            Token throughput (tokens/s), max active requests, fragmentation
            percentage, KV cache size, TTFT/ITL, and speedup vs static.
        """
        precision, bpp = resolve_precision(precision)
        peak_flops = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)

        # Base latency metrics (including attention O(S²) term)
        n_layers = getattr(model, 'layers', 1) or 1
        n_heads = getattr(model, 'heads', 32) or 32
        head_dim = (getattr(model, 'hidden_dim', 4096) or 4096) // n_heads
        linear_flops = 2 * model.parameters.to(ureg.count).magnitude * seq_len * max_batch_size
        attention_flops = 4 * n_layers * n_heads * head_dim * seq_len**2 * max_batch_size
        prefill_ops = (linear_flops + attention_flops) * ureg.flop
        t_prefill = (prefill_ops / (peak_flops * efficiency)).to("ms") + hardware.dispatch_tax

        # KV budget is whatever HBM remains after the weights are pinned resident;
        # the concurrent-request count is bounded by how many sequences fit in it.
        model_weights_bytes = model.size_in_bytes(bpp)
        max_memory_for_kv = hardware.memory.capacity - model_weights_bytes

        if max_memory_for_kv.magnitude <= 0:
            return ContinuousBatchingResult(
                feasible=False,
                constraint_trace=[f"Memory Wall: FAILED. Weights ({model_weights_bytes.to('GB'):~P}) exceed available {hardware.memory.capacity.to('GB'):~P} on {hardware.name}."],
                throughput_tokens_per_sec=0.0, max_active_requests=0,
                memory_fragmentation_pct=0.0, paged_kv_cache_size=Q_("0 GB"),
                ttft=t_prefill, itl=Q_("1000000 ms"), speedup_vs_static=1.0
            )

        # Calculate memory using PagedAttention formulas
        from ...physics import calc_paged_kv_cache_size
        n_heads = model.kv_heads or model.heads or 32
        h_dim = model.hidden_dim or 4096
        head_dim = h_dim // (model.heads or 32)

        bytes_per_seq, frag_pct = calc_paged_kv_cache_size(
            model.layers, n_heads,
            head_dim,
            seq_len, batch_size=1, page_size_tokens=page_size, bytes_per_elem=bpp
        )

        # int() truncates: a partially-fitting sequence cannot be admitted, so the
        # memory-derived limit rounds down; the scheduler cap then takes the min.
        max_possible_requests = int((max_memory_for_kv / bytes_per_seq).to_base_units().magnitude)
        active_requests = min(max_possible_requests, max_batch_size)

        constraint_trace = []
        if active_requests > 0:
            constraint_trace.append(f"Memory Wall: Passed. Can fit {active_requests} concurrent requests (Weights: {model_weights_bytes.to('GB'):~P}, Max KV available: {max_memory_for_kv.to('GB'):~P}) on {hardware.name}.")
        else:
            constraint_trace.append(f"Memory Wall: FAILED. Cannot fit even 1 request. Weights ({model_weights_bytes.to('GB'):~P}) exceed or leave no room for KV cache in available {hardware.memory.capacity.to('GB'):~P} on {hardware.name}.")

        total_kv_cache = bytes_per_seq * active_requests

        # Throughput: each decode step streams weights once (shared by the whole
        # batch) plus every active sequence's KV — the memory-bound step time.
        t_decode_per_token = ((model_weights_bytes + total_kv_cache) / hardware.memory.bandwidth).to("ms")

        if active_requests == 0 or t_decode_per_token.magnitude == 0:
            # Degenerate cases (nothing fits, or zero step time) — report zero
            # throughput rather than dividing by zero below.
            throughput = 0.0
            speedup = 1.0
        else:
            # One step emits one token per active request, so tokens/s is the
            # batch size over the step time.
            throughput = (active_requests / t_decode_per_token).to("1/s").magnitude
            # Static batching comparison:
            # With contiguous KV allocation, memory fragmentation limits the max batch.
            # Kwon et al. (2023) measured 20-40% external fragmentation in static allocation.
            # We model static batching as achieving ~60% of continuous batching's batch size
            # due to fragmentation waste, with no internal fragmentation savings.
            static_frag_factor = 0.6  # effective batch = 60% of continuous (Kwon et al. 2023)
            static_effective_batch = max(1, int(active_requests * static_frag_factor))
            static_t_decode = ((model_weights_bytes + (bytes_per_seq * static_effective_batch)) / hardware.memory.bandwidth).to("ms")
            static_throughput = (static_effective_batch / static_t_decode).to("1/s").magnitude
            speedup = throughput / static_throughput if static_throughput > 0 else 1.0

        return ContinuousBatchingResult(
            feasible=active_requests > 0,
            constraint_trace=constraint_trace,
            throughput_tokens_per_sec=throughput,
            max_active_requests=active_requests,
            memory_fragmentation_pct=frag_pct,
            paged_kv_cache_size=total_kv_cache.to("GB"),
            ttft=t_prefill,
            itl=t_decode_per_token,
            speedup_vs_static=speedup
        )

class WeightStreamingModel(ForwardModel):
    """
    Analyzes Wafer-Scale inference (e.g., Cerebras CS-3) using Weight Streaming.

    Instead of holding weights in HBM and streaming activations (the GPU Memory Wall),
    this architecture holds massive activation batches on-wafer (SRAM) and streams
    the model weights from external MemoryX nodes.

    The bottleneck shifts from Memory Bandwidth to Injection Interconnect Bandwidth.

    Literature Source:
    1. Lie et al. (2022), "Cerebras Architecture Deep Dive: First Look Inside
       the Hardware/Software Co-Design for Deep Learning."
    """
    requires = ("workload", "hardware")
    produces = WeightStreamingResult

    def solve(self, model: TransformerWorkload, hardware: HardwareNode, seq_len: int,
              batch_size: int = 1, precision: str = "fp16", efficiency: float = 0.5,
              phase: str = "decode") -> WeightStreamingResult:
        """Simulates Weight Streaming throughput and SRAM feasibility.

        Parameters
        ----------
        phase : str
            Inference phase: 'prefill' or 'decode' (default 'decode').
            - prefill: processes all S tokens in parallel (compute-heavy, O(S^2) attention)
            - decode: processes one token at a time per request (memory-bound)
        """
        precision, bpp = resolve_precision(precision)
        peak_flops = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)

        # 1. SRAM Capacity Constraint
        # Entire batch's KV Cache must fit in the 44GB on-wafer SRAM
        # Working set (activations for the current layer) must also fit, but KV cache dominates
        n_heads = model.kv_heads or model.heads or 32
        h_dim = model.hidden_dim or 4096
        head_dim = h_dim // (model.heads or 32)

        # KV dimensions per layer per sequence; the factor 2 counts the separate
        # K and V tensors stored per attention head.
        bytes_per_seq_per_layer = seq_len * n_heads * head_dim * 2 * bpp.magnitude
        total_kv_bytes = bytes_per_seq_per_layer * model.layers * batch_size * ureg.byte

        # 10% headroom for the current layer's activation working set — small next
        # to KV at these batch sizes, but not free.
        total_memory_required = (total_kv_bytes * 1.1).to("GB")

        feasible = total_memory_required <= hardware.memory.capacity
        utilization = (total_memory_required / hardware.memory.capacity).magnitude if hardware.memory.capacity.magnitude > 0 else 1.0

        constraint_trace = []
        if feasible:
            constraint_trace.append(f"SRAM Wall: Passed. Required {total_memory_required.to('GB'):~P} (KV + 10% Overhead) <= Available {hardware.memory.capacity.to('GB'):~P} on {hardware.name}.")
        else:
            constraint_trace.append(f"SRAM Wall: FAILED. Required {total_memory_required.to('GB'):~P} (KV + 10% Overhead) > Available {hardware.memory.capacity.to('GB'):~P} on {hardware.name}.")

        # 2. Injection Bottleneck vs Compute Bottleneck per Layer
        layer_params = model.parameters / model.layers
        layer_weight_bytes = layer_params.to(ureg.count).magnitude * bpp.magnitude * ureg.byte

        # Injection time (MemoryX -> WSE). Fall back to a calibrated default when
        # the registry entry is missing or zero — a zero injection bandwidth would
        # otherwise make the division below blow up.
        inj_bw = hardware.interconnect.bandwidth if hardware.interconnect else Q_(cal.FALLBACK_INTERCONNECT_BANDWIDTH_GB_S, "GB/s")
        if inj_bw.magnitude == 0:
            inj_bw = Q_(cal.FALLBACK_INTERCONNECT_BANDWIDTH_GB_S, "GB/s")

        layer_injection_time = (layer_weight_bytes / inj_bw).to("ms")

        # Compute time depends on phase
        if phase == "prefill":
            # Prefill: process all S tokens in parallel.
            # Linear FLOPs: 2 * params * S * batch_size
            # Attention FLOPs: 2 * n_heads * S^2 * head_dim * n_layers (per layer already factored below)
            layer_linear_flops = 2 * layer_params.to(ureg.count).magnitude * seq_len * batch_size
            # Attention for one layer: 2 * n_heads * seq_len^2 * head_dim
            layer_attn_flops = 2 * n_heads * seq_len * seq_len * head_dim
            layer_total_flops = (layer_linear_flops + layer_attn_flops * batch_size) * ureg.flop
            layer_decode_flops = layer_total_flops
        else:
            # Decode: one new token per request across the batch
            # 2 * parameters * batch_size
            layer_decode_flops = 2 * layer_params.to(ureg.count).magnitude * batch_size * ureg.flop
        layer_compute_time = (layer_decode_flops / (peak_flops * efficiency)).to("ms") + hardware.dispatch_tax

        # True layer time is bounded by the slowest process: weight streaming
        # overlaps injection with compute, so per-layer latency is max(), not sum.
        if layer_compute_time >= layer_injection_time:
            bottleneck = "Compute-Bound"
            layer_time = layer_compute_time
        else:
            bottleneck = "Interconnect-Bandwidth-Bound"
            layer_time = layer_injection_time

        total_token_time = layer_time * model.layers
        # In prefill, we process batch_size * seq_len tokens; in decode, batch_size tokens per step
        tokens_produced = batch_size * seq_len if phase == "prefill" else batch_size
        tps = (tokens_produced / total_token_time).to("1/s").magnitude if total_token_time.magnitude > 0 else 0.0

        # 3. Optimal Batch Size Analysis
        # What batch size perfectly overlaps injection and compute?
        # Solve: t_inject = (2 * layer_params * B) / (peak_flops * efficiency)
        # => B = t_inject * peak_flops * efficiency / (2 * layer_params)
        optimal_batch = (layer_injection_time * peak_flops * efficiency) / (2 * layer_params.to(ureg.count).magnitude * ureg.flop)
        optimal_batch_int = max(1, int(optimal_batch.to_base_units().magnitude))

        # Short-circuit: if infeasible (SRAM overflow), report zero throughput
        if not feasible:
            tps = 0.0

        return WeightStreamingResult(
            feasible=feasible,
            constraint_trace=constraint_trace,
            throughput_tokens_per_sec=tps,
            bottleneck=bottleneck,
            layer_compute_time=layer_compute_time,
            layer_injection_time=layer_injection_time,
            optimal_batch_size=optimal_batch_int,
            wafer_memory_utilization=min(utilization, 1.0) if feasible else utilization,
        )

class TailLatencyModel(ForwardModel):
    """
    Analyzes queueing delays and P99 tail latency for deployed inference models.

    Models inference servers as M/M/c queues to determine if the deployment
    can sustain the target arrival rate while meeting strict SLA latency bounds.

    Approximations (deliberate, pedagogical):
    - ``p50_latency``/``p99_latency`` add the WAIT-time quantile to the MEAN
      service time. The true quantile of sojourn time (wait + service) is
      generally larger, because service time has its own variance.
    - Kingman's ``(cv^2 + 1)/2`` factor is derived for the MEAN wait (M/G/1);
      applying it to the P50/P99 wait quantiles is a further heuristic.
    Quote these fields as model estimates, not exact M/G/c percentiles.

    Literature Source:
    1. Dean & Barroso (2013), "The Tail at Scale."
    """
    requires = ("hardware",)
    produces = TailLatencyResult

    def solve(self, arrival_rate_qps: float, service_latency_ms: float, num_replicas: int = 1, service_time_cv: float = 1.0) -> TailLatencyResult:
        """Solves for P50 and P99 tail latencies under variable load.

        Parameters
        ----------
        arrival_rate_qps : float
            Request arrival rate in queries per second.
        service_latency_ms : float
            Mean service time per request in milliseconds.
        num_replicas : int
            Number of server replicas (c in M/M/c).
        service_time_cv : float
            Coefficient of variation of service time (default 1.0 = exponential).
            When CV != 1, applies Kingman's M/G/1 correction factor
            (cv^2 + 1) / 2 to queue wait times, approximating M/G/c behavior.
        """
        from ...core._validation import validate_nonnegative
        validate_nonnegative(service_time_cv, "service_time_cv")
        from ...physics import calc_queue_latency_mmc

        # Queueing theory works in rates: mean service time (ms) -> per-server
        # service rate mu (requests/s).
        service_rate_hz = 1000.0 / service_latency_ms if service_latency_ms > 0 else 0.0

        rho, p50_w, p99_w = calc_queue_latency_mmc(arrival_rate_qps, service_rate_hz, num_replicas)

        # Kingman's formula correction for M/G/c approximation:
        # W_q(M/G/c) ≈ W_q(M/M/c) * (cv² + 1) / 2
        # When cv=1 (exponential), the factor is 1.0 (no correction).
        if service_time_cv != 1.0:
            kingman_factor = (service_time_cv ** 2 + 1) / 2
            p50_w = p50_w * kingman_factor
            p99_w = p99_w * kingman_factor

        is_stable = rho < 1.0

        # P99 wait time exceeding 5x service latency signifies severe SLA risk
        slo_threshold = service_latency_ms * 5

        p99_w_ms = p99_w.m_as(ureg.millisecond)
        p50_w_ms = p50_w.m_as(ureg.millisecond)

        # SLO headroom: ratio of P99 wait to SLO threshold.
        # Values > 1.0 indicate SLO violation. This is a ratio, not a probability.
        slo_headroom_ratio = 1.0 if not is_stable else (p99_w_ms / slo_threshold if slo_threshold > 0 else 0)
        slo_headroom_ratio = max(0.0, slo_headroom_ratio)  # No upper clamp: values > 1.0 indicate SLO violation

        return TailLatencyResult(
            p50_latency=Q_(p50_w_ms + service_latency_ms, "ms"),
            p99_latency=Q_(p99_w_ms + service_latency_ms, "ms"),
            queue_utilization=rho,
            is_stable=is_stable,
            slo_headroom_ratio=slo_headroom_ratio,
        )

class InferenceScalingModel(ForwardModel):
    """
    Models inference-time compute scaling (Wall 12: Reasoning/CoT Cost).

    This model quantifies the cost of 'System-2 thinking' — inference-time
    compute scaling via chain-of-thought (CoT) reasoning, where the model
    generates K intermediate reasoning steps before producing the final answer.
    Each step incurs the full cost of autoregressive decoding.

    Literature Source:
    1. Wei et al. (2022), "Chain-of-Thought Prompting Elicits Reasoning in
       Large Language Models."
    2. Snell et al. (2024), "Scaling LLM Test-Time Compute Optimally Can Be
       More Effective Than Scaling Model Parameters."
    3. OpenAI (2024), "Learning to Reason with LLMs." (o1 reasoning model.)
    """
    requires = ("workload", "hardware")
    produces = InferenceScalingResult

    def solve(self, model: TransformerWorkload, hardware: HardwareNode,
              reasoning_steps: int = 8, context_length: int = 2048,
              precision: str = "fp16", efficiency: float = 0.5) -> InferenceScalingResult:
        """
        Solves for inference-time reasoning cost.

        Parameters
        ----------
        model : TransformerWorkload
            The language model used for reasoning.
        hardware : HardwareNode
            The target hardware node.
        reasoning_steps : int
            Number of reasoning steps K (each generates tokens).
        context_length : int
            Input context length in tokens.
        precision : str
            Numerical precision.
        efficiency : float
            Compute efficiency factor (0.0 to 1.0).

        Returns
        -------
        InferenceScalingResult
            Total reasoning time, cost per query, and token counts.
        """
        # Use ServingModel internally to get per-step latency
        serving = ServingModel()
        serving_result = serving.solve(
            model=model, hardware=hardware, seq_len=context_length,
            batch_size=1, precision=precision, efficiency=efficiency
        )

        # Inter-token latency (ITL) is the cost per generated token
        itl = serving_result.itl

        # Average tokens per reasoning step (heuristic — see core/calibration.py)
        tokens_per_step = cal.TOKENS_PER_REASONING_STEP
        total_tokens = reasoning_steps * tokens_per_step

        # T_reason = K × tokens_per_step × ITL + TTFT (initial prefill)
        ttft = serving_result.ttft
        t_step = tokens_per_step * itl
        total_reasoning_time = ttft + reasoning_steps * t_step

        # Cost estimate (based on hardware TDP and time)
        if hardware.tdp is not None:
            energy_j = (hardware.tdp * total_reasoning_time.to("s")).to("J")
        else:
            energy_j = Q_("0 J")

        return InferenceScalingResult(
            total_reasoning_time=total_reasoning_time.to("ms"),
            ttft=ttft,
            itl=itl,
            tokens_generated=total_tokens,
            reasoning_steps=reasoning_steps,
            time_per_step=t_step.to("ms"),
            energy_per_query=energy_j,
            feasible=serving_result.feasible,
            serving_detail=serving_result,
        )

class BatchingOptimizer(BaseOptimizer):
    """
    Finds the maximum batch size that satisfies a P99 latency SLA.

    Searches the continuous batching design space using an M/M/c queueing
    model to find the optimal balance between throughput and tail latency.
    """
    requires = ("workload", "hardware", "sla_latency_ms")
    produces = BatchingOptimizerResult

    def solve(self, model: TransformerWorkload, hardware: HardwareNode,
              seq_len: int, sla_latency_ms: float, arrival_rate_qps: float,
              num_replicas: int = 1, precision: str = "fp16",
              efficiency: float = 0.5, max_search_batch: int = 256) -> BatchingOptimizerResult:
        """
        Determine the maximum batch size that satisfies a P99 latency SLA.

        Exhaustively scans batch sizes 1..``max_search_batch``. For each
        candidate ``b``, the per-batch service latency is
        ``TTFT + ITL * seq_len`` from ``ServingModel``, and the batch arrival
        rate is ``arrival_rate_qps / b`` (requests are grouped into batches).
        A candidate is feasible when the model fits in memory, the M/M/c
        queue is stable, and the queueing P99 meets the SLA. Among feasible
        candidates the largest batch wins, since throughput grows with batch
        size.

        Parameters
        ----------
        model : TransformerWorkload
            The model being served.
        hardware : HardwareNode
            The serving accelerator.
        seq_len : int
            Tokens generated per request (used as the decode count).
        sla_latency_ms : float
            P99 latency SLA in milliseconds.
        arrival_rate_qps : float
            Aggregate request arrival rate in queries per second.
        num_replicas : int
            Number of serving replicas (the 'c' in M/M/c).
        precision : str
            Numerical precision.
        efficiency : float
            Software efficiency factor in (0, 1].
        max_search_batch : int
            Upper bound of the batch-size search range.

        Returns
        -------
        BatchingOptimizerResult
            Best batch size, resulting throughput (tokens/s), P99 latency,
            and SLO headroom; ``is_feasible=False`` if no batch size works.
        """
        from ...core.optimization.registry import OptimizationRegistry
        serving_model = ServingModel()
        tail_model = TailLatencyModel()

        def objective(b_array):
            """Search objective: returns -b for feasible batch sizes (so the
            exhaustive minimizer picks the largest feasible batch) and a 1e12
            penalty for memory-infeasible, unstable, or SLA-violating ones."""
            b = int(b_array[0])
            res = serving_model.solve(model, hardware, seq_len=seq_len, batch_size=b, precision=precision, efficiency=efficiency)
            if not res.feasible:
                return 1e12 # Infeasible due to memory

            service_latency = res.ttft + (res.itl * seq_len)
            # Requests are grouped b-at-a-time, so the queue sees batches arriving
            # at qps/b — larger batches mean fewer, slower queue customers.
            tail_res = tail_model.solve(arrival_rate_qps / b, service_latency.m_as("ms"), num_replicas)

            if not tail_res.is_stable or tail_res.p99_latency.m_as("ms") > sla_latency_ms:
                return 1e12 # Infeasible due to queueing instability or SLA violation

            # We want to maximize batch size (which maximizes valid throughput), so minimize -b
            return -b

        backend = OptimizationRegistry.get_backend("exhaustive")
        backend.compile(objective_fn=objective, ranges=[(1, max_search_batch)], grid_size=max_search_batch)
        opt_res = backend.solve()

        if not opt_res.feasible:
            return BatchingOptimizerResult(
                objective_value=0.0,
                best_config={"max_batch_size": 0},
                best_batch_size=0,
                max_throughput=0.0,
                p99_latency=Q_("0 ms"),
                slo_headroom_ratio=0.0,
                is_feasible=False,
                total_searched=max_search_batch
            )

        best_b = int(opt_res.best_configuration["optimal_variable"])
        max_tps = arrival_rate_qps * seq_len

        # Resolve final exact metrics for the optimal batch size
        res = serving_model.solve(model, hardware, seq_len=seq_len, batch_size=best_b, precision=precision, efficiency=efficiency)
        service_latency = res.ttft + (res.itl * seq_len)
        tail_res = tail_model.solve(arrival_rate_qps / best_b, service_latency.m_as("ms"), num_replicas)

        return BatchingOptimizerResult(
            objective_value=max_tps,
            best_config={"max_batch_size": best_b},
            best_batch_size=best_b,
            max_throughput=max_tps,
            p99_latency=tail_res.p99_latency,
            slo_headroom_ratio=tail_res.slo_headroom_ratio,
            is_feasible=True,
            total_searched=max_search_batch
        )

__all__ = [
    "ServingModel",
    "ServingCapacityModel",
    "ContinuousBatchingModel",
    "WeightStreamingModel",
    "TailLatencyModel",
    "InferenceScalingModel",
    "BatchingOptimizer",
]
