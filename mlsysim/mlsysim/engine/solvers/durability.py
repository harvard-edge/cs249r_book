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
        """Separate amortized flush work from the first and last record waits.

        Parameters
        ----------
        batch_timer : Quantity
            Nonnegative collection interval before the flush begins.
        flush_time : Quantity
            Nonnegative duration of one synchronous flush.
        batch_size : int
            Positive record count acknowledged by the flush.

        Returns
        -------
        GroupCommitResult
            Amortized work per record and first/last record waits,
            all time quantities.

        See Also
        --------
        mlsysim.physics.reliability.calc_group_commit_waits :
            Compute the amortized work and per-arrival waits.

        Notes
        -----
        Equation: Work per record = flush_time/batch_size; first wait = timer + flush;
        last wait = flush_time.

        Assumptions: The first record arrives when collection starts and the last
        arrives when the flush begins; one flush acknowledges the whole batch.

        Excludes: Queueing between groups, variable flush duration, and partial
        acknowledgments.

        References
        ----------
        Derivation: Divide the shared flush work among the acknowledged records. The
        first arrival waits for collection and flushing; the last arrival
        waits only for flushing, independent of that amortized work.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import GroupCommitModel
        >>> r = GroupCommitModel().solve(Q_(1, "ms"), Q_(2, "ms"), 4)
        >>> assert r.first_wait >= r.last_wait >= r.work_per_record
        """
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
        """Compare full-snapshot and incremental-delta checkpoint write volume.

        Parameters
        ----------
        turns : int
            Nonnegative completed trajectory turns; must contain at least one stride.
        stride : int
            Positive checkpoint spacing in turns.
        state_growth : Quantity
            Positive added state bytes per turn.

        Returns
        -------
        SnapshotVolumeResult
            Checkpoint count, full and incremental byte totals, and
            their dimensionless ratio.

        See Also
        --------
        mlsysim.physics.reliability.calc_snapshot_write_volume :
            Sum full snapshots and incremental deltas.

        Notes
        -----
        Equation: N = floor(turns/stride); full = ``growth*stride*N*(N+1)/2``;
        delta = ``growth*stride*N``.

        Assumptions: State starts empty and grows by equal bytes every turn. Only
        complete strides are checkpointed; no final partial-stride save is added.

        Excludes: State deletion, compression, metadata, and checkpoint service
        time.

        References
        ----------
        Derivation: Full snapshots contain one, two, and then up to N strides of new
        state; summing this arithmetic progression gives ``N*(N+1)/2`` strides.
        Deltas contain one stride each, for N strides in total.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import SnapshotVolumeModel
        >>> r = SnapshotVolumeModel().solve(4, 2, Q_(1, "MB"))
        >>> assert r.full_bytes >= r.delta_bytes
        """
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
        """Price checkpoint strides on a supplied crash trace.

        Parameters
        ----------
        turns : int
            Positive trajectory length in turns.
        crashes : sequence of int
            Nonempty crash positions in [0, turns], interpreted after the indicated turn.
        strides : sequence of int
            Nonempty positive checkpoint spacings to compare.
        pause : Quantity
            Positive pause to capture one checkpoint.
        restore : Quantity
            Nonnegative fixed restore time per crash.
        rebuild_per_turn : Quantity
            Positive time to rebuild one lost turn from a log.
        reexecute_per_turn : Quantity
            Positive time to re-execute one lost turn without the log.

        Returns
        -------
        CheckpointTraceResult
            Per-stride pause/recovery/total seconds and lost-turn counts,
            cheapest candidate, and continuous balance strides in turns.

        See Also
        --------
        mlsysim.physics.reliability.calc_checkpoint_stride :
            Compute the continuous pause/rework balance.

        Notes
        -----
        Equation: Lost turns at crash c for stride k = c mod k. Total time equals
        ``checkpoint_count*pause`` + ``crash_count*restore`` + ``lost_turns*rebuild``.
        Continuous balance stride = ``sqrt(2*pause/(crash_rate*turn_cost))``.

        Assumptions: Every candidate sees the same trace; capture occurs at stride
        multiples and at the final turn. The balance uses crashes/turns as a rate.

        Excludes: Failures during checkpoint or recovery, checkpoint corruption,
        and a probabilistic guarantee about future traces.

        References
        ----------
        Young (1974), "A first order approximation to the optimum checkpoint interval."
        https://doi.org/10.1145/361147.361115 (bibliography key: young1974).
        The continuous balance minimizes pause/k + ``crash_rate*turn_cost*k/2``.

        Derivation: Count every scheduled pause, fixed restore, and rebuilt turn on the
        supplied trace. For the continuous comparison, minimizing pause/k +
        ``crash_rate*turn_cost*k/2`` gives the square-root balance; this is a
        separate expected-loss approximation, not the discrete trace optimum.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import CheckpointTraceModel
        >>> r = CheckpointTraceModel().solve(8, [3], [1, 2], Q_(1, "s"),
        ...     Q_(1, "s"), Q_(1, "s"), Q_(2, "s"))
        >>> assert r.best_stride in r.total_time
        """
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
        """Price recovery logs and one workspace snapshot over a retention period.

        Parameters
        ----------
        turns : int
            Nonnegative number of retained trajectory turns.
        envelope : Quantity
            Nonnegative metadata bytes per turn.
        payload : Quantity
            Nonnegative payload bytes per turn.
        workspace : Quantity
            Nonnegative bytes in one workspace snapshot; total retained bytes must be positive.
        rate : Quantity
            Nonnegative storage price in currency/(byte*time).
        period : Quantity
            Nonnegative retention duration.
        attempt_cost : Quantity
            Positive cost of one attempt, in currency.

        Returns
        -------
        RetentionCostResult
            Log and total bytes, USD cost quantities, and dimensionless
            cost and workspace shares.

        Notes
        -----
        Equation: Log bytes = ``turns*(envelope+payload)``; total = log + workspace;
        cost = ``total*rate*period``; attempt share = cost/attempt_cost.

        Assumptions: All retained bytes live for the full period at a uniform rate.
        One workspace snapshot accompanies the per-turn log.

        Excludes: Replication, retrieval charges, tier changes, and deletion
        overhead.

        References
        ----------
        Derivation: Add per-turn envelope and payload bytes across the trajectory, then
        add the single workspace snapshot. Byte-time times unit storage price
        gives the retention cost; divide by attempt cost to express its share.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import RetentionCostModel
        >>> r = RetentionCostModel().solve(2, Q_(1, "byte"), Q_(2, "byte"),
        ...     Q_(3, "byte"), Q_(1, "USD/GB/day"), Q_(1, "day"), Q_(1, "USD"))
        >>> assert r.total_bytes >= r.log_bytes
        """
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
