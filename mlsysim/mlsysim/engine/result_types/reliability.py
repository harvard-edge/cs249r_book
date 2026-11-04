"""Typed result models for reliability calculations."""

from __future__ import annotations

from typing import Dict, List, Tuple
from ...core.types import Quantity
from .base import SolverResult

__all__ = [
    "ReliabilityResult",
    "FailureIntervalResult",
    "SeriesFailureResult",
    "CheckpointIntervalResult",
    "RecoveryBudgetResult",
    "GroupCommitResult",
    "SnapshotVolumeResult",
    "CheckpointTraceResult",
    "RetentionCostResult",
    "RepairBreakEvenResult",
    "IrreversibleLossResult",
    "UndetectedLoopResult",
    "RetryAmplificationResult",
    "BulkheadAdmissionResult",
    "RetryJitterResult",
]


class ReliabilityResult(SolverResult):
    """Result from ReliabilityModel: failure modeling and checkpointing."""

    __module__ = "mlsysim.engine.results"

    fleet_mtbf: Quantity
    failure_probability: float
    optimal_checkpoint_interval: Quantity
    expected_failures: float
    goodput_ratio: float = 1.0


class FailureIntervalResult(SolverResult):
    """Result from FailureIntervalModel: series failures across identical components."""

    __module__ = "mlsysim.engine.results"

    interval: Quantity
    expected_failures: float
    failures_per_day: float


class SeriesFailureResult(SolverResult):
    """Result from SeriesFailureModel: a job that stops when any one component fails.

    Rates are per hour; ``component_rates`` is each component's share per device.
    """

    __module__ = "mlsysim.engine.results"

    component_rates: Dict[str, float]
    device_rate: float
    system_rate: float
    mtbf: Quantity
    failures_per_day: float


class CheckpointIntervalResult(SolverResult):
    """Result from CheckpointIntervalModel: the Young-Daly interval and its waste."""

    __module__ = "mlsysim.engine.results"

    write_time: Quantity
    mtbf: Quantity
    interval: Quantity
    write_share: float
    rework_share: float
    waste: float


class RecoveryBudgetResult(SolverResult):
    """Result from RecoveryBudgetModel: one cold restart, and its daily cost at a failure rate."""

    __module__ = "mlsysim.engine.results"

    detect: Quantity
    restart: Quantity
    reload: Quantity
    warmup: Quantity
    total: Quantity
    failure_interval: Quantity
    failures_per_day: float
    lost_per_day: Quantity
    day_share: float
    gpu_hours_per_day: float


class GroupCommitResult(SolverResult):
    """Result from GroupCommitModel: storage work per record against the wait each record sees."""

    __module__ = "mlsysim.engine.results"

    work_per_record: Quantity  # flush / batch size
    first_wait: Quantity  # batch timer + flush
    last_wait: Quantity  # flush alone


class SnapshotVolumeResult(SolverResult):
    """Result from SnapshotVolumeModel: bytes written by full snapshots against deltas."""

    __module__ = "mlsysim.engine.results"

    checkpoints: int
    full_bytes: Quantity
    delta_bytes: Quantity
    ratio: float  # full / delta = (n + 1) / 2


class CheckpointTraceResult(SolverResult):
    """Result from CheckpointTraceModel: checkpoint strides priced on one crash trace."""

    __module__ = "mlsysim.engine.results"

    checkpoints: Dict[int, int]  # per stride
    lost_turns: Dict[int, Tuple[int, ...]]  # per stride, turns since the last checkpoint at each crash
    capture_time: Dict[int, Quantity]  # checkpoints x pause
    recovery_time: Dict[int, Quantity]  # restores + lost turns rebuilt from the log
    total_time: Dict[int, Quantity]
    best_stride: int  # least total on this trace
    every_turn_time: Quantity  # checkpoint after every turn
    crash_rate: float  # crashes per turn
    stride_logged: float  # balance stride when lost turns are rebuilt from the log
    stride_reexecuted: float  # balance stride when lost turns are re-executed
    stride_gain: float  # logged / re-executed


class RetentionCostResult(SolverResult):
    """Result from RetentionCostModel: bytes kept for recovery and replay, and what keeping them costs."""

    __module__ = "mlsysim.engine.results"

    log_bytes: Quantity
    total_bytes: Quantity
    cost: Quantity  # USD over the retention period
    log_cost: Quantity
    attempt_share: float  # cost / attempt cost
    log_share: float
    workspace_share: float  # workspace snapshot share of the bytes


class RepairBreakEvenResult(SolverResult):
    """Result from RepairBreakEvenModel: one bounded repair against unwinding and rebuilding."""

    __module__ = "mlsysim.engine.results"

    fallback_time: Quantity  # compensation + rebuild to the same endpoint
    fallback_cost: Quantity  # USD, model tokens of the rebuild
    repair_time: Quantity  # startup + output at the decode rate + validation
    repair_cost: Quantity  # USD, the whole isolated request and its output
    break_even_time: float  # p* on the time axis
    break_even_cost: float  # p* on the dollar axis
    expected_time: Quantity  # repair first at the assumed success probability
    expected_cost: Quantity
    time_saved_fraction: float  # 1 - expected / fallback
    cost_saved_fraction: float
    one_in: int  # one success in this many attempts clears both axes


class IrreversibleLossResult(SolverResult):
    """Result from IrreversibleLossModel: expected loss with the check after or before the pivot."""

    __module__ = "mlsysim.engine.results"

    loss_without_check: Quantity  # P(defect) x consequence, nothing recovers
    loss_caught: Quantity  # caught before the pivot, the part a revert misses
    loss_missed: Quantity  # crossed the pivot undetected
    loss_with_check: Quantity
    reduction: float  # without / with
    missed_share: float  # missed / with
    loss_per_catch_step: Quantity  # loss removed by raising the catch rate one step
    raised_catch_rate: float
    loss_caught_raised: Quantity
    loss_missed_raised: Quantity
    loss_with_check_raised: Quantity
    attempt_cost: Quantity
    loss_vs_attempt: float  # residual loss / attempt cost
    step_vs_attempt: float  # one catch step / attempt cost


class UndetectedLoopResult(SolverResult):
    """Result from UndetectedLoopModel: an append-only loop run to its ceiling or stopped early."""

    __module__ = "mlsysim.engine.results"

    appended_per_turn: int  # output + observation added to the next input
    ceiling_input: int  # input tokens resent over the ceiling's turns
    stop_input: int
    ceiling_cost: Quantity
    stop_cost: Quantity
    ceiling_time: Quantity
    stop_time: Quantity
    cost_ratio: float
    avoided_cost: Quantity
    peak_request: int  # largest input plus its output
    fits_context: bool
    half_turns: int
    half_ratio: float  # input at the ceiling / input at half the ceiling
    input_by_turn: List[int]  # cumulative input after 0..ceiling turns


class RetryAmplificationResult(SolverResult):
    """Result from RetryAmplificationModel: attempts one failing operation sends through nested layers."""

    __module__ = "mlsysim.engine.results"

    by_layer: List[int]  # attempts reaching each depth, from the caller down
    layered: int  # attempts at the dependency
    coordinated: int  # one owner with the same allowance


class BulkheadAdmissionResult(SolverResult):
    """Result from BulkheadAdmissionModel: one burst of blocking calls, uncapped and capped."""

    __module__ = "mlsysim.engine.results"

    uncapped_occupied: int
    uncapped_free: int
    occupied: int
    queued: int
    rejected: int
    protected_workers: int


class RetryJitterResult(SolverResult):
    """Result from RetryJitterModel: a retry cohort released together, synchronized or jittered."""

    __module__ = "mlsysim.engine.results"

    synchronized_rate: Quantity  # all retries inside the burst window
    jittered_mean_rate: Quantity  # spread uniformly over the jitter span
    spread_ratio: float
    expected_in_window: float  # expected count in one burst-sized window
