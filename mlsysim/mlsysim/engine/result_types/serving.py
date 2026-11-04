"""Typed result models for serving calculations."""

from __future__ import annotations

from typing import Any, Dict, Optional
from ...core.types import Quantity
from .base import SolverResult

__all__ = [
    "ServingResult",
    "ServingCapacityResult",
    "ContinuousBatchingResult",
    "WeightStreamingResult",
    "TailLatencyResult",
    "InferenceScalingResult",
    "GoodputResult",
    "RuntimeGoodputResult",
    "KvAdmissionResult",
    "PrecisionDividendResult",
    "KvAttentionDesignResult",
    "PagedAllocationResult",
    "PrefixRebuildResult",
    "ChunkedPrefillResult",
    "PausedKvStateResult",
    "KvBlockSharingResult",
    "KvResidencyArchetypeResult",
]


class ServingResult(SolverResult):
    """Result from ServingModel: LLM two-phase inference performance."""

    __module__ = "mlsysim.engine.results"

    feasible: bool
    ttft: Quantity
    itl: Quantity
    kv_cache_size: Quantity
    model_weights_size: Quantity
    total_memory_required: Quantity
    memory_utilization: float
    prompt_cache_hit_ratio: float = 0.0
    prefill_chunks: int = 1
    prefill_chunk_time: Optional[Quantity] = None
    decode_stall_bound: Optional[Quantity] = None


class ServingCapacityResult(SolverResult):
    """Result from ServingCapacityModel: replica count for an inference SLA."""

    __module__ = "mlsysim.engine.results"

    feasible: bool
    required_replicas: int
    qps_target: float
    qps_capacity: float
    per_replica_qps_capacity: float
    utilization: float
    target_p99_latency: Quantity
    estimated_p99_latency: Quantity
    base_request_latency: Quantity
    queue_wait_p99: Quantity
    active_batch_size: int
    ttft: Quantity
    itl: Quantity
    bottleneck: str


class ContinuousBatchingResult(SolverResult):
    """Result from ContinuousBatchingModel: static max-length KV reservation vs PagedAttention.

    Unprefixed throughput, request, TTFT, and ITL fields describe the paged
    allocator; ``static_*`` fields describe max-length reservation on the same
    KV budget. Fragmentation fields are fractions in [0, 1] of allocated KV
    tokens left unused inside each request's allocation.
    """

    __module__ = "mlsysim.engine.results"

    feasible: bool
    throughput_tokens_per_sec: float
    max_active_requests: int
    static_throughput_tokens_per_sec: float
    static_max_active_requests: int
    mean_request_tokens: float
    paged_internal_fragmentation: float
    static_internal_fragmentation: float
    paged_kv_cache_size: Quantity
    static_kv_cache_size: Quantity
    ttft: Quantity
    itl: Quantity
    speedup_vs_static: float


class WeightStreamingResult(SolverResult):
    """Result from WeightStreamingModel: Wafer-scale SRAM batch processing."""

    __module__ = "mlsysim.engine.results"

    feasible: bool
    throughput_tokens_per_sec: float
    bottleneck: str
    layer_compute_time: Quantity
    layer_injection_time: Quantity
    optimal_batch_size: int
    wafer_memory_utilization: float


class TailLatencyResult(SolverResult):
    """Result from TailLatencyModel: queuing theory limits (M/M/c)."""

    __module__ = "mlsysim.engine.results"

    p50_latency: Quantity
    p99_latency: Quantity
    queue_utilization: float
    is_stable: bool
    slo_headroom_ratio: float


class InferenceScalingResult(SolverResult):
    """Result from InferenceScalingModel: CoT reasoning cost."""

    __module__ = "mlsysim.engine.results"

    total_reasoning_time: Quantity
    ttft: Quantity
    itl: Quantity
    tokens_generated: int
    reasoning_steps: int
    time_per_step: Quantity
    energy_per_query: Quantity
    feasible: bool
    serving_detail: Any  # ServingResult (avoids circular validation)


class GoodputResult(SolverResult):
    """Result from GoodputModel: ML productivity goodput S x R x P and where the rest went.

    Shares are of offered capacity-time; each lost interval is charged at the
    first boundary it fails (scheduling, then runtime, then program).
    """

    __module__ = "mlsysim.engine.results"

    mpg: float  # S x R x P
    job_product: float  # R x P, the job's own share of held time
    after_scheduling: float  # S
    after_runtime: float  # S x R
    scheduling_share: float  # 1 - S
    runtime_share: float  # S (1 - R)
    program_share: float  # S R (1 - P)
    point_value: Dict[str, float]  # one point of a factor = product of the other two
    ceiling_ratio: Dict[str, float]  # ceiling / current, per factor with a ceiling
    ceiling_gain: Dict[str, float]  # MPG gained at each factor's ceiling, the others held
    mpg_at_ceiling: Dict[str, float]
    full_scheduling_gain: float  # MPG gained by a perfect S
    cost_multiple: float  # 1 / MPG, price of a useful unit over the raw price
    job_cost_multiple: float  # 1 / (R x P), a chargeback on held hours only
    runtime_headroom: float  # 1 / R - 1, the largest relative gain left in R
    useful_rate: Optional[Quantity] = None  # device-hour rate / MPG


class RuntimeGoodputResult(SolverResult):
    """Result from RuntimeGoodputModel: held time a job keeps as retained steps, term by term.

    Every ``*_share`` is a share of held time; the five add to ``runtime_loss``
    (first-order: each term small, so overlaps are neglected).
    """

    __module__ = "mlsysim.engine.results"

    mtbf: Quantity  # system failure interval
    write_time: Quantity  # one checkpoint write
    interval: Quantity  # Young-Daly checkpoint interval
    write_share: float
    rework_share: float
    checkpoint_share: float  # write + rework at the interval
    recovery_time: Quantity  # detect + restart + reload + warmup
    recovery_share: float  # recovery_time / MTBF (rework counted in checkpoints)
    startup_time: Quantity  # restart + reload + warmup, no detection
    startup_share: float  # startup_time / held window
    input_wait_share: float
    preemption_cost: Quantity  # lost work + startup, per forced preemption
    preemption_share: float  # rate x cost
    failure_side: float  # checkpoints + recovery
    runtime_loss: float
    runtime_goodput: float
    interruption_interval: Quantity  # MTBF with forced preemptions counted as interruptions
    interruption_checkpoint_interval: Quantity


class KvAdmissionResult(SolverResult):
    """Result from KvAdmissionModel: requests of one context length that fit a replica's KV budget."""

    __module__ = "mlsysim.engine.results"

    head_dim: int
    weights: Quantity
    hbm_total: Quantity
    kv_budget: Quantity  # HBM - weights - reserve
    kv_per_token: Quantity
    kv_per_request: Quantity
    max_requests: int  # floor(kv_budget / kv_per_request)


class PrecisionDividendResult(SolverResult):
    """Result from PrecisionDividendModel: maximum batch before and after lower precision."""

    __module__ = "mlsysim.engine.results"

    weights_per_device: Quantity
    weights_per_device_low: Quantity
    kv_budget_per_device: Quantity
    kv_budget_per_device_low: Quantity
    kv_per_request: Quantity  # whole request at the base KV precision
    kv_per_request_low: Quantity
    kv_per_request_per_device: Quantity
    kv_per_request_per_device_low: Quantity
    max_batch: int
    max_batch_low: int
    dividend: float  # max_batch_low / max_batch
    budget_gain: float  # KV budget low / KV budget base


class KvAttentionDesignResult(SolverResult):
    """Result from KvAttentionDesignModel: bytes per token by attention design."""

    __module__ = "mlsysim.engine.results"

    per_token: Dict[str, Quantity]  # model name -> KV bytes per token
    design: Dict[str, str]  # model name -> MHA, GQA, or MLA
    mha_over_mla: float
    gqa_over_mla: float
    gqa_session: Quantity  # one session of session_tokens, GQA model
    mla_session: Quantity


class PagedAllocationResult(SolverResult):
    """Result from PagedAllocationModel: full-cap reservation against paged blocks."""

    __module__ = "mlsysim.engine.results"

    reservation: Quantity  # cap x bytes per token
    reservations: int  # full-cap reservations in the pool
    final_length_efficiency: float  # mean final length / cap
    small_block: Quantity  # bytes of one small block
    large_block: Quantity
    small_tail_tokens: int  # worst-case unused slots, small blocks
    large_tail_tokens: int
    small_tail: Quantity
    large_tail: Quantity
    large_tail_share: float  # worst large tail / cap
    small_allocated_tokens: float  # expected tokens held per request
    small_waste: float  # expected internal fragmentation
    large_allocated_tokens: float
    large_waste: float
    paged_fit: int  # final-length snapshots, small blocks
    paged_gain: float  # paged fit / reservations


class PrefixRebuildResult(SolverResult):
    """Result from PrefixRebuildModel: the prefill a prefix hit avoids, against one suffix pass."""

    __module__ = "mlsysim.engine.results"

    n_params: float
    state: Quantity  # KV bytes of the prefix
    linear_ops: Quantity  # 2 P S
    attention_ops: Quantity  # 2 L d S^2 (causal)
    sustained_rate: Quantity  # accelerators x peak x utilization
    rebuild_time: Quantity  # (linear + attention) / rate
    weight_read_time: Quantity  # weights / node bandwidth
    suffix_ops_time: Quantity  # 2 P S_suffix / rate
    rebuild_over_hit: float  # rebuild time / max(weight read, suffix ops)


class ChunkedPrefillResult(SolverResult):
    """Result from ChunkedPrefillModel: decode gap with a monolithic prefill against chunks."""

    __module__ = "mlsysim.engine.results"

    decode_step: Quantity  # decode-only iteration
    monolithic_prefill: Quantity  # the whole prompt in one iteration
    monolithic_gap: Quantity  # decode step + monolithic prefill
    chunks: int
    chunked_iteration: Quantity  # one chunk beside the decodes
    chunked_done: Quantity  # chunks x iteration
    gap_ratio: float  # monolithic gap / chunked iteration


class PausedKvStateResult(SolverResult):
    """Result from PausedKvStateModel: keep, move, or drop a paused request's KV state."""

    __module__ = "mlsysim.engine.results"

    blocks: int
    state: Quantity  # uniquely reclaimable KV bytes
    link: Quantity  # accelerators x host link
    copy_out: Quantity
    copy_in: Quantity
    round_trip: Quantity  # copy out + copy in
    recompute: Quantity  # 2 P S / rate, weight work only
    recompute_over_reload: float
    recompute_over_reload_derated: float  # at the derated link fraction
    held_occupancy: Quantity  # state x wait (bytes x time)
    released_on_demand: Quantity  # wait - copy out
    released_prefetch: Quantity  # wait - copy out - copy in
    burst: Quantity  # returns x state
    burst_time: Quantity  # burst / link
    burst_slowdown: float  # burst time / one reload


class KvBlockSharingResult(SolverResult):
    """Result from KvBlockSharingModel: block-rounded resident fit, shared prefix, and turn rate."""

    __module__ = "mlsysim.engine.results"

    block_bytes: Quantity
    usable_blocks: int  # floor((1 - headroom) pool / block)
    usable: Quantity
    unshared_blocks: int  # ceil((context + output) / block)
    common_blocks: int
    private_blocks: int
    unshared_bytes: Quantity
    common_bytes: Quantity
    private_bytes: Quantity
    fit_unshared: int
    fit_shared: int
    retained_time: Quantity  # generation + wait
    retained_rate: Quantity  # fit unshared / retained time
    released_rate: Quantity  # fit unshared / generation
    shared_retained_rate: Quantity  # fit shared / retained time
    shared_released_rate: Quantity  # fit shared / generation
    residency_factor: float  # 1 + wait / generation
    shared_fraction: float  # common blocks / unshared blocks
    share_gain: float  # fit shared / fit unshared
    share_bound: float  # 1 / (1 - f)


class KvResidencyArchetypeResult(SolverResult):
    """Result from KvResidencyArchetypeModel: sharing, residency tax, and resume cost for one workload."""

    __module__ = "mlsysim.engine.results"

    shared_fraction: float  # shared prefix / context
    fit_unshared: int
    fit_shared: int
    share_gain: float
    residency_factor: float
    turn_cycle: Quantity  # generation + wait
    recompute: Quantity
    reload: Quantity
    recompute_share: float  # recompute / turn cycle
    shared_retained_rate: Quantity  # fit shared / turn cycle
    shared_released_rate: Quantity  # fit shared / generation
