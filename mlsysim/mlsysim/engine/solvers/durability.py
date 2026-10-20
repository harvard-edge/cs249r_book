"""Durable-execution solvers: what a runtime pays to survive a crash (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. Each model is arithmetic on
stated inputs: one group commit's work and waits, the bytes full snapshots
and deltas write, checkpoint strides priced on a fixed crash trace with
Young's balance in turns, and the cost of keeping a trajectory's recovery and
replay data. None models a particular storage engine or workflow runtime.
"""

from __future__ import annotations

from typing import Sequence

from ..results import (
    CheckpointTraceResult,
    GroupCommitResult,
    RetentionCostResult,
    SnapshotVolumeResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...physics.reliability import (
    calc_checkpoint_stride,
    calc_group_commit_waits,
    calc_snapshot_write_volume,
)


class GroupCommitModel(ForwardModel):
    """
    Storage work per record against the wait each record sees in one group commit.

    One flush acknowledges ``batch_size`` records; the work charged to each
    falls as ``flush / batch_size``, but every record still waits the whole
    flush, and the first arrival also waits the batch timer. Amortization
    raises throughput; it never shortens one record's wait.
    """
    requires = ()
    produces = GroupCommitResult

    def solve(self, batch_timer: Quantity, flush_time: Quantity, batch_size: int) -> GroupCommitResult:
        work, first, last = calc_group_commit_waits(batch_timer, flush_time, batch_size)
        return GroupCommitResult(work_per_record=work, first_wait=first, last_wait=last)


class SnapshotVolumeModel(ForwardModel):
    """
    Bytes written by full snapshots and by deltas over one monotonically growing trajectory.

    ``turns // stride`` checkpoints; full snapshots grow with the square of
    the checkpoint count, deltas linearly, so the ratio is ``(n + 1) / 2``.
    """
    requires = ()
    produces = SnapshotVolumeResult

    def solve(self, turns: int, stride: int, state_growth: Quantity) -> SnapshotVolumeResult:
        n = turns // stride
        full, delta = calc_snapshot_write_volume(state_growth, stride, n)
        return SnapshotVolumeResult(checkpoints=n, full_bytes=full, delta_bytes=delta,
                                    ratio=float((full / delta).to("").magnitude))


class CheckpointTraceModel(ForwardModel):
    """
    Checkpoint strides priced on one fixed crash trace, against Young's balance in turns.

    A policy checkpoints at every multiple of its stride (and at the final
    turn). A crash after turn ``c`` restores the last checkpoint and rebuilds
    the ``c mod stride`` turns since it from the log; the policy's total is
    its checkpoint pauses plus the restores plus the rebuilt turns. The
    balance stride ``sqrt(2 pause / (rate * cost))`` uses the trace's crash
    rate, once with the cost of rebuilding a logged turn and once with the
    cost of re-executing an unlogged one.
    """
    requires = ()
    produces = CheckpointTraceResult

    def solve(self, turns: int, crashes: Sequence[int], strides: Sequence[int], pause: Quantity,
              restore: Quantity, rebuild_per_turn: Quantity,
              reexecute_per_turn: Quantity) -> CheckpointTraceResult:
        checkpoints, lost, capture, recovery, total = {}, {}, {}, {}, {}
        for k in strides:
            marks = set(range(k, turns + 1, k)) | {turns}
            checkpoints[k] = len(marks)
            lost[k] = tuple(c % k for c in crashes)
            capture[k] = (len(marks) * pause).to("second")
            recovery[k] = (len(crashes) * restore + sum(lost[k]) * rebuild_per_turn).to("second")
            total[k] = (capture[k] + recovery[k]).to("second")
        best = min(strides, key=lambda k: total[k].m_as("second"))
        every = (turns * pause + len(crashes) * restore).to("second")
        rate = len(crashes) / turns
        logged = calc_checkpoint_stride(pause, rate, rebuild_per_turn)
        reexec = calc_checkpoint_stride(pause, rate, reexecute_per_turn)
        return CheckpointTraceResult(
            checkpoints=checkpoints, lost_turns=lost, capture_time=capture, recovery_time=recovery,
            total_time=total, best_stride=best, every_turn_time=every, crash_rate=rate,
            stride_logged=logged, stride_reexecuted=reexec, stride_gain=logged / reexec,
        )


class RetentionCostModel(ForwardModel):
    """
    What keeping one trajectory's recovery and replay data costs, against one attempt.

    The log is ``turns * (envelope + payload)``; one workspace snapshot is
    added; the total is priced at an object-storage rate for the retention
    period and compared with the cost of one attempt.
    """
    requires = ()
    produces = RetentionCostResult

    def solve(self, turns: int, envelope: Quantity, payload: Quantity, workspace: Quantity,
              rate: Quantity, period: Quantity, attempt_cost: Quantity) -> RetentionCostResult:
        log = (turns * (envelope + payload)).to("byte")
        total = (log + workspace).to("byte")
        cost = (total * rate * period).to("USD")
        log_cost = (log * rate * period).to("USD")
        return RetentionCostResult(
            log_bytes=log, total_bytes=total, cost=cost, log_cost=log_cost,
            attempt_share=float((cost / attempt_cost).to("").magnitude),
            log_share=float((log_cost / attempt_cost).to("").magnitude),
            workspace_share=float((workspace / total).to("").magnitude),
        )


__all__ = ["GroupCommitModel", "SnapshotVolumeModel", "CheckpointTraceModel", "RetentionCostModel"]
