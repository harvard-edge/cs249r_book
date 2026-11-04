"""Fleet goodput solvers: ML productivity goodput and runtime goodput.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). Added 2026-10-20.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..results import GoodputResult, RuntimeGoodputResult
from ...core.types import Quantity
from ...core.units import Q_, ureg
from .base import ForwardModel
from .reliability import CheckpointIntervalModel


class GoodputModel(ForwardModel):
    """
    ML productivity goodput: the share of offered capacity-time that becomes
    ideal-rate work, ``MPG = S x R x P``.

    Scheduling goodput ``S`` is held over offered capacity-time, runtime
    goodput ``R`` retained over held time, and program goodput ``P`` ideal over
    retained time. Each factor divides by what the previous one passed, so the
    factors multiply and every lost interval is charged once, at the first
    boundary it fails: ``1 = (1 - S) + S(1 - R) + SR(1 - P) + SRP``. One point
    of a factor is worth the product of the other two, and a lever that lifts a
    factor to its ceiling multiplies MPG by the ceiling's ratio. A useful
    accelerator-hour costs the offered rate divided by MPG.

    Literature Source:
    1. Wongpanich et al. (2025), "Machine Learning Fleet Efficiency: Analyzing
       and Optimizing Large-Scale Google TPU Systems with ML Productivity Goodput."
    """
    requires = ("scheduling", "runtime", "program")
    produces = GoodputResult

    def solve(self, scheduling: float, runtime: float, program: float,
              ceilings: Optional[Dict[str, float]] = None,
              device_hour_rate: Optional[Quantity] = None) -> GoodputResult:
        """Partition offered capacity-time into retained work and stage losses.

        Parameters
        ----------
        scheduling : float
            Scheduling fraction S in (0, 1], measured over offered capacity-time.
        runtime : float
            Runtime fraction R in (0, 1], measured over scheduled capacity-time.
        program : float
            Program fraction P in (0, 1], measured over retained runtime.
        ceilings : dict of str to float, optional
            Proposed fractions in (0, 1] keyed by ``S``, ``R``, or ``P``; each change
            is evaluated alone with the other factors held fixed.
        device_hour_rate : Quantity, optional
            Nonnegative price per offered device-hour, convertible to USD/hour.

        Returns
        -------
        GoodputResult
            Dimensionless goodput, disjoint loss shares, factor sensitivities, and
            one-factor ceiling gains. ``useful_rate`` is USD/hour of useful work,
            or None when no price is supplied.

        Notes
        -----
        Equation: MPG = ``S*R*P``; losses are 1-S, ``S*(1-R)``, and ``S*R*(1-P)``.
        A factor's marginal point value is the product of the other two factors;
        useful_rate = device_hour_rate / MPG.

        Assumptions: factors have consecutive nested denominators. Positive factors
        are required because costs and relative gains divide by them.

        Excludes: correlated responses to simultaneous interventions, capital costs,
        and forecasts of attainable ceilings. Inputs are supplied fractions.

        References
        ----------
        Derivation: expand 1 = (1-S) + ``S*(1-R)`` + ``S*R*(1-P)`` + ``S*R*P``;
        the partition charges each lost interval only at its first failed boundary.

        Examples
        --------
        >>> result = GoodputModel().solve(0.5, 0.5, 1.0)
        >>> result.mpg == 0.25
        True
        >>> result.scheduling_share + result.runtime_share + result.program_share + result.mpg == 1
        True
        """
        f = {"S": scheduling, "R": runtime, "P": program}
        mpg = scheduling * runtime * program
        point = {k: mpg / v for k, v in f.items()}
        ceilings = ceilings or {}
        ratio = {k: c / f[k] for k, c in ceilings.items()}
        gain = {k: (c - f[k]) * point[k] for k, c in ceilings.items()}
        rate = (device_hour_rate / mpg).to("USD/hour") if device_hour_rate is not None else None
        return GoodputResult(
            mpg=mpg,
            job_product=runtime * program,
            after_scheduling=scheduling,
            after_runtime=scheduling * runtime,
            scheduling_share=1 - scheduling,
            runtime_share=scheduling * (1 - runtime),
            program_share=scheduling * runtime * (1 - program),
            point_value=point,
            ceiling_ratio=ratio,
            ceiling_gain=gain,
            mpg_at_ceiling={k: mpg + g for k, g in gain.items()},
            full_scheduling_gain=(1 - scheduling) * point["S"],
            cost_multiple=1 / mpg,
            job_cost_multiple=1 / (runtime * program),
            runtime_headroom=1 / runtime - 1,
            useful_rate=rate,
        )


class RuntimeGoodputModel(ForwardModel):
    """
    Runtime goodput of one job: the share of its held time spent on retained steps.

    The loss is the sum of five owners' terms on one denominator, held time
    (first order; each term is small):

    - startup: restart, checkpoint load, and warmup once per held window;
    - input waits: complete allocations stalled on the input pipeline;
    - checkpoint waste at the Young-Daly interval, write share plus rework;
    - recovery: the four-term cold-restart budget over the failure interval
      (rework is already counted in the checkpoint term);
    - forced preemption: rate times (work since the last save + startup).

    Reuses ``CheckpointIntervalModel`` for the checkpoint term and the
    recovery profile's ``cold_restart_time`` for recovery, startup, and the
    preemption relaunch (the same source as ``RecoveryBudgetModel`` and
    ``PreemptionTaxModel``).

    Literature Source:
    1. Young (1974); Daly (2006), optimal checkpoint interval.
    2. Wongpanich et al. (2025), runtime goodput.
    """
    requires = ("mtbf", "checkpoint_bytes", "recovery", "input_wait", "window")
    produces = RuntimeGoodputResult

    def solve(self, mtbf: Quantity, checkpoint_bytes: Quantity, recovery, input_wait: float,
              window: Quantity, lost_work: Quantity, preemption_rate: Quantity) -> RuntimeGoodputResult:
        """Account for first-order runtime losses on one held-time denominator.

        Parameters
        ----------
        mtbf : Quantity
            Positive mean time between job failures, convertible to hours.
        checkpoint_bytes : Quantity
            Nonnegative bytes written and reloaded per checkpoint.
        recovery : object
            Recovery profile with positive ``checkpoint_write_bw_gbs`` and a
            ``cold_restart_time`` method returning time, including optional detection.
        input_wait : float
            Fraction of held time lost to input stalls, between zero and one.
        window : Quantity
            Positive held-time window over which one startup is amortized.
        lost_work : Quantity
            Nonnegative work duration discarded by each forced preemption.
        preemption_rate : Quantity
            Nonnegative forced-preemption frequency, convertible to inverse hours.

        Returns
        -------
        RuntimeGoodputResult
            Checkpoint and recovery time budgets, dimensionless loss shares, and
            runtime_goodput = 1 - summed losses. The value is not clipped; a negative
            result signals that the small-loss approximation is outside its regime.

        See Also
        --------
        mlsysim.engine.solvers.reliability.CheckpointIntervalModel.solve : Checkpoint waste accounting.

        Notes
        -----
        Equation: loss = startup/window + input_wait + checkpoint_waste
        + recovery_time/mtbf + ``preemption_rate*(lost_work`` + startup).
        The effective interruption interval is 1/(1/mtbf + preemption_rate).

        Assumptions: terms are small, disjoint first-order charges. The checkpoint
        term includes failure rework; the recovery term excludes that rework.
        Startup excludes failure detection for a scheduler-initiated restart.

        Excludes: overlapping outages, correlated preemptions, recovery queueing,
        and dynamic checkpoint adaptation. The interruption checkpoint interval is
        reported separately rather than substituted into the primary loss budget.

        References
        ----------
        Derivation: amortize each event duration over its observation window or
        multiply it by its event frequency, then sum losses on held time.
        The checkpoint submodel supplies its separately documented interval rule.

        Examples
        --------
        >>> from mlsysim import Systems, Q_
        >>> result = RuntimeGoodputModel().solve(Q_("100 hour"), Q_("1 GB"),
        ...     Systems.Reliability.Recovery, 0.0, Q_("24 hour"), Q_("0 s"), Q_("0 / hour"))
        >>> result.preemption_share == 0
        True
        >>> result.runtime_goodput == 1 - result.runtime_loss
        True
        """
        bw = Q_(float(recovery.checkpoint_write_bw_gbs), "GB/s")
        write_time = (checkpoint_bytes / bw).to(ureg.second)
        ckpt = CheckpointIntervalModel().solve(write_time, mtbf.to(ureg.hour))
        recovery_time = recovery.cold_restart_time(checkpoint_bytes).to(ureg.minute)
        recovery_share = (recovery_time / mtbf).to("").magnitude
        startup = recovery.cold_restart_time(checkpoint_bytes, include_detection=False).to(ureg.minute)
        startup_share = (startup / window).to("").magnitude
        preempt = (lost_work + startup).to(ureg.minute)
        preempt_share = (preemption_rate * preempt).to("").magnitude
        loss = startup_share + input_wait + ckpt.waste + recovery_share + preempt_share
        interrupt = (1 / (1 / mtbf + preemption_rate)).to(ureg.hour)
        interrupt_ckpt = CheckpointIntervalModel().solve(write_time, interrupt)
        return RuntimeGoodputResult(
            mtbf=ckpt.mtbf,
            write_time=write_time,
            interval=ckpt.interval.to(ureg.minute),
            write_share=ckpt.write_share,
            rework_share=ckpt.rework_share,
            checkpoint_share=ckpt.waste,
            recovery_time=recovery_time,
            recovery_share=recovery_share,
            startup_time=startup,
            startup_share=startup_share,
            input_wait_share=input_wait,
            preemption_cost=preempt,
            preemption_share=preempt_share,
            failure_side=ckpt.waste + recovery_share,
            runtime_loss=loss,
            runtime_goodput=1 - loss,
            interruption_interval=interrupt,
            interruption_checkpoint_interval=interrupt_ckpt.interval.to(ureg.minute),
        )


__all__ = ["GoodputModel", "RuntimeGoodputModel"]
