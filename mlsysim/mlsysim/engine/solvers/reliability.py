"""Reliability and checkpoint-interval solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations


from ..results import (
    ReliabilityResult,
)
from ...physics import (
    calc_mtbf_cluster,
    calc_mtbf_node,
    calc_young_daly_interval,
    calc_failure_probability,
)
from ...core.units import ureg, Q_
from ...core.types import Quantity
from ...systems.reliability import Reliability
from ...systems.types import Fleet
from .base import ForwardModel

class ReliabilityModel(ForwardModel):
    """
    Calculates Mean Time Between Failures (MTBF) and optimal checkpointing intervals.

    This model handles the reliability modeling of massive clusters, helping
    determine the 'Goodput' of long-running training jobs. It identifies
    the probability of a job failure before completion and calculates the
    Young-Daly optimal interval to minimize wasted compute time.

    Literature Source:
    1. Young (1974), "A First-Order Approximation to the Optimum Checkpoint
       Interval."
    2. Daly (2006), "A Higher Order Estimate of the Optimum Checkpoint
       Interval for Restart-Dump Strategy."
    """
    requires = ("fleet",)
    produces = ReliabilityResult

    def solve(self, fleet: Fleet, job_duration_hours: float, checkpoint_time_s: float = 60.0,
              avg_recovery_time_s: float = 300.0) -> ReliabilityResult:
        """Estimate fleet failure exposure and a checkpoint interval.

        Parameters
        ----------
        fleet : Fleet
            Cluster configuration with positive node count and per-node device counts.
        job_duration_hours : float
            Nonnegative planned job duration in hours.
        checkpoint_time_s : float, optional
            Positive write pause per checkpoint in seconds; default 60.
        avg_recovery_time_s : float, optional
            Nonnegative recovery duration per failure in seconds; default 300, including reload,
            restart, and warmup.

        Returns
        -------
        ReliabilityResult
            Fleet MTBF and checkpoint interval with time units, failure probability, expected
            failure count, and dimensionless goodput estimate.

        See Also
        --------
        mlsysim.physics.reliability.calc_mtbf_node : Sum independent component failure rates.
        mlsysim.physics.reliability.calc_young_daly_interval : Balance checkpoint writing and lost work.

        Notes
        -----
        Equation: lambda_node = n_gpu / MTBF_gpu + n_nic / MTBF_nic + n_psu / MTBF_psu; M = 1 /
        (fleet.count * lambda_node); P(failure) = 1-exp(-job_duration/M); tau = sqrt(2
        checkpoint_time M). The reported goodput is max(0,
        1-checkpoint_time/tau-recovery_time/M).

        Assumptions: Independent exponential component failures and a synchronous series system.
        Nonredundant PSU counts come from the node; component lifetimes come from the
        Reliability registry.

        Excludes: Correlated failures, repair contention, and explicit re-execution loss in the
        reported goodput. The goodput expression charges checkpoint writes and recovery only,
        even though the checkpoint interval balances write and rework costs.

        References
        ----------
        Derivation: Series-system hazard rates add. The no-failure survival function is
        exponential. The checkpoint helper minimizes the first-order write-plus-rework
        approximation; the goodput subtraction is the implementation's separate overhead
        convention.

        Examples
        --------
        >>> from mlsysim import Systems
        >>> r = ReliabilityModel().solve(Systems.Clusters.Research_256, 24)
        >>> 0 <= r.failure_probability <= 1 and r.fleet_mtbf.m_as("hour") > 0
        True
        """
        # Series-system reliability: any single component failure stalls the
        # whole synchronous job, so failure rates ADD — first across a node's
        # GPUs, NICs, and non-redundant PSUs, then across all nodes. Fleet MTBF therefore shrinks
        # roughly as 1/N with cluster size.
        node_mtbf = calc_mtbf_node(
            gpu_mtbf_h=Reliability.Gpu.mttf_hours, n_gpus=fleet.node.accelerators_per_node,
            nic_mtbf_h=Reliability.Nic.mttf_hours, n_nics=fleet.node.nics_per_node,
            psu_mtbf_h=Reliability.Psu.mttf_hours, n_psus=fleet.node.series_psus,
        )
        fleet_mtbf = calc_mtbf_cluster(node_mtbf, fleet.count)

        job_dur_q = Q_(job_duration_hours, "hour")
        prob_fail = calc_failure_probability(fleet_mtbf, job_dur_q)

        ckpt_time_q = Q_(checkpoint_time_s, "second")
        optimal_interval = calc_young_daly_interval(ckpt_time_q, fleet_mtbf.to("second"))

        # Goodput ratio: fraction of rawput that produces useful training progress.
        # Lost compute = P(failure) * (avg_recovery_time / checkpoint_interval)
        # Steady-state goodput: fraction of time spent on useful training.
        # Overhead = checkpoint_write_time / interval + recovery_time / MTBF
        # Source: Daly (2006), Young (1974)
        interval_s = optimal_interval.m_as(ureg.second)
        cluster_mtbf_s = fleet_mtbf.m_as(ureg.second)
        if interval_s > 0 and cluster_mtbf_s > 0:
            checkpoint_overhead = ckpt_time_q.m_as(ureg.second) / interval_s
            recovery_overhead = avg_recovery_time_s / cluster_mtbf_s
            goodput_ratio = max(0.0, 1.0 - checkpoint_overhead - recovery_overhead)
        else:
            goodput_ratio = 0.0

        return ReliabilityResult(
            fleet_mtbf=fleet_mtbf,
            failure_probability=prob_fail,
            optimal_checkpoint_interval=optimal_interval,
            expected_failures=(job_dur_q / fleet_mtbf).magnitude,
            goodput_ratio=goodput_ratio,
        )

__all__ = [
    "ReliabilityModel",
]


class FailureIntervalModel(ForwardModel):
    """
    Interval between failures across identical, independent components.

    With N components that each fail on average once per MTBF, some component
    fails on average every MTBF / N (the minimum of N exponentials). Over a run
    of a given length the expected failure count is the run length divided by
    that interval.
    """
    requires = ("component_mtbf", "n_components")

    def solve(self, component_mtbf: Quantity, n_components: int, run_duration: Quantity):
        """Calculate the failure interval of independent identical components.

        Parameters
        ----------
        component_mtbf : Quantity
            Positive per-component MTBF in hours or compatible units.
        n_components : int
            Positive number of independent components.
        run_duration : Quantity
            Positive observation duration in time units.

        Returns
        -------
        FailureIntervalResult
            System interval in hours, dimensionless expected failure count over the run, and
            failures per day.

        See Also
        --------
        mlsysim.physics.reliability.calc_mtbf_cluster : Aggregate identical node failure rates.

        Notes
        -----
        Equation: interval = component_mtbf / n_components; expected_failures = run_duration /
        interval; failures_per_day = one_day / interval.

        Assumptions: Independent, identical exponential failures and an unchanged population
        over the run.

        Excludes: Common-cause faults, repair downtime, and redundancy masking component failures.

        References
        ----------
        Derivation: The minimum of independent exponential lifetimes has the sum of their rates;
        count expectation is rate times duration.

        Examples
        --------
        >>> r = FailureIntervalModel().solve(Q_(100, "hour"), 10, Q_(20, "hour"))
        >>> r.interval == Q_(10, "hour") and r.expected_failures == 2
        True
        """
        from ..results import FailureIntervalResult
        interval = calc_mtbf_cluster(component_mtbf, n_components)
        run = run_duration.to("hour")
        expected = (run / interval).to("").magnitude
        return FailureIntervalResult(
            interval=interval,
            expected_failures=expected,
            failures_per_day=expected / run.to("day").magnitude,
        )


class SeriesFailureModel(ForwardModel):
    """
    Failure interval of a synchronous job built from heterogeneous components.

    Any single component failure stops the job, so failure rates add (a series
    system with independent, exponentially distributed failures). Each
    component is stated per node, with its MTTF and its count per node; its
    share per device is ``count / devices_per_node / MTTF``. A redundant part
    that cannot fail the node singly (an N+M power supply) enters with count 0.
    Added 2026-10-20.
    """
    requires = ("n_devices", "devices_per_node", "component_mttf", "per_node")

    def solve(self, n_devices: int, devices_per_node: int, component_mttf: dict, per_node: dict):
        """Aggregate heterogeneous component hazards for a synchronous job.

        Parameters
        ----------
        n_devices : int
            Positive accelerator count in the job.
        devices_per_node : int
            Positive accelerator count per node.
        component_mttf : dict of str to Quantity
            Positive component MTTFs in hours or compatible units.
        per_node : dict of str to int
            Nonnegative component counts per node, with the same keys as component_mttf. Use
            zero for parts whose single failure cannot stop the node.

        Returns
        -------
        SeriesFailureResult
            Per-device component rates, total device and system rates in inverse hours, MTBF in
            hours, and failures per day.

        Notes
        -----
        Equation: component_rate[k] = per_node[k] / (devices_per_node * MTTF[k]); system_rate =
        n_devices * sum(component_rate); MTBF = 1/system_rate.

        Assumptions: Independent exponential component failures; any included component failure
        stops the entire job. At least one component rate must be positive.

        Excludes: Fault correlation, shared infrastructure, and explicit redundant failure
        combinations.

        References
        ----------
        Derivation: Summing independent exponential hazards gives the series-system rate.
        Per-node rates are divided by devices per node and multiplied by the job's device count.

        Examples
        --------
        >>> r = SeriesFailureModel().solve(8, 4, {"gpu": Q_(100, "hour")}, {"gpu": 4})
        >>> r.mtbf == Q_(12.5, "hour")
        True
        """
        from ..results import SeriesFailureResult
        rates = {k: per_node[k] / devices_per_node / mttf.m_as(ureg.hour)
                 for k, mttf in component_mttf.items()}
        device_rate = sum(rates.values())
        system_rate = n_devices * device_rate
        return SeriesFailureResult(
            component_rates=rates,
            device_rate=device_rate,
            system_rate=system_rate,
            mtbf=Q_(1 / system_rate, "hour"),
            failures_per_day=system_rate * Q_(1, "day").m_as(ureg.hour),
        )


class CheckpointIntervalModel(ForwardModel):
    """
    Young-Daly checkpoint interval and the share of wall time it wastes.

    ``tau* = sqrt(2 * write_time * MTBF)`` balances the write share
    ``write_time / tau`` against the rework share ``tau / (2 * MTBF)``; at the
    optimum the two are equal. Literature: Young (1974); Daly (2006).
    Added 2026-10-20.
    """
    requires = ("write_time", "mtbf")

    def solve(self, write_time: Quantity, mtbf: Quantity):
        """Balance checkpoint write time against expected rework.

        Parameters
        ----------
        write_time : Quantity
            Positive checkpoint write pause in time units.
        mtbf : Quantity
            Positive job mean time between failures in time units.

        Returns
        -------
        CheckpointIntervalResult
            Write duration in seconds, MTBF in hours, optimal interval, and dimensionless write,
            rework, and total waste shares.

        See Also
        --------
        mlsysim.physics.reliability.calc_young_daly_interval : Balance checkpoint writing and lost work.
        mlsysim.physics.reliability.calc_checkpoint_waste : Sum write and rework overhead fractions.

        Notes
        -----
        Equation: tau = sqrt(2 write_time MTBF); write_share = write_time/tau; rework_share =
        tau/(2 MTBF); waste = write_share + rework_share.

        Assumptions: First-order periodic checkpointing, failures uniformly placed within an
        interval, and checkpoint cost small enough for the approximation to be useful.

        Excludes: Checkpoint overlap, recovery time, correlated failures, and higher-order
        scheduling corrections.

        References
        ----------
        Derivation: Differentiating write_time/tau + tau/(2 MTBF) and setting its derivative to
        zero yields tau; the two contributions are equal at that point.

        Examples
        --------
        >>> r = CheckpointIntervalModel().solve(Q_(10, "s"), Q_(1000, "s"))
        >>> abs(r.write_share - r.rework_share) < 1e-12
        True
        """
        from ..results import CheckpointIntervalResult
        from ...physics.reliability import calc_checkpoint_waste
        tau = calc_young_daly_interval(write_time, mtbf)
        write_share = (write_time / tau).to("").magnitude
        rework_share = (tau / (2 * mtbf)).to("").magnitude
        return CheckpointIntervalResult(
            write_time=write_time.to(ureg.second), mtbf=mtbf.to(ureg.hour), interval=tau,
            write_share=write_share, rework_share=rework_share,
            waste=calc_checkpoint_waste(write_time, tau, mtbf),
        )


class RecoveryBudgetModel(ForwardModel):
    """
    Cold-restart budget for one failure, and its cost per day at a failure interval.

    The budget is detection + restart + checkpoint load + warmup
    (``RecoveryProfile.cold_restart_time``); a job that fails every
    ``failure_interval`` pays it ``1 day / failure_interval`` times a day, on
    every one of its ``n_devices``. Rework since the last save is charged to
    checkpoint waste, not here. Added 2026-10-20.
    """
    requires = ("checkpoint_bytes", "recovery", "failure_interval", "n_devices")

    def solve(self, checkpoint_bytes: Quantity, recovery, failure_interval: Quantity, n_devices: int):
        """Price serial cold-restart phases at a given failure frequency.

        Parameters
        ----------
        checkpoint_bytes : Quantity
            Nonnegative checkpoint size in bytes.
        recovery : RecoveryProfile
            Detection, restart, warmup, and positive checkpoint bandwidth profile.
        failure_interval : Quantity
            Positive interval between job failures in time units.
        n_devices : int
            Nonnegative number of reserved job devices.

        Returns
        -------
        RecoveryBudgetResult
            Phase and total restart times, failures/day, lost minutes/day, dimensionless day
            share, and device-hours lost per day.

        Notes
        -----
        Equation: reload = checkpoint_bytes / bandwidth; total = detect + restart + reload +
        warmup; failures_per_day = one_day / failure_interval; daily_loss = total *
        failures_per_day.

        Assumptions: All phases execute serially, their durations are fixed, and the whole job
        waits for recovery.

        Excludes: Rework since the checkpoint, concurrent recovery, retry failures, and
        overlapping incidents.

        References
        ----------
        Derivation: Adding serial phase durations gives one recovery budget; multiplying by
        event frequency and device count gives expected device-time loss.

        Examples
        --------
        >>> r = RecoveryBudgetModel().solve(Q_(1, "GB"), Reliability.Recovery, Q_(1, "day"), 4)
        >>> r.failures_per_day == 1 and r.lost_per_day == r.total
        True
        """
        from ..results import RecoveryBudgetResult
        bw = float(recovery.checkpoint_write_bw_gbs) * ureg.GB / ureg.second
        reload = (checkpoint_bytes / bw).to(ureg.second)
        total = recovery.cold_restart_time(checkpoint_bytes)
        per_day = (Q_(1, "day") / failure_interval).to("").magnitude
        loss = (per_day * total).to(ureg.minute)
        return RecoveryBudgetResult(
            detect=float(recovery.detection_time_s) * ureg.second,
            restart=float(recovery.restart_time_s) * ureg.second,
            reload=reload,
            warmup=float(recovery.warmup_time_s) * ureg.second,
            total=total,
            failure_interval=failure_interval.to(ureg.hour),
            failures_per_day=per_day,
            lost_per_day=loss,
            day_share=(loss / Q_(1, "day")).to("").magnitude,
            gpu_hours_per_day=loss.m_as(ureg.hour) * n_devices,
        )


__all__ += ["SeriesFailureModel", "CheckpointIntervalModel", "RecoveryBudgetModel"]
