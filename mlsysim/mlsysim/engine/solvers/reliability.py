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
        """
        Calculates reliability and checkpointing metrics for a fleet.

        Parameters
        ----------
        fleet : Fleet
            The hardware cluster configuration.
        job_duration_hours : float
            Total job duration in hours.
        checkpoint_time_s : float
            Time to write one checkpoint in seconds (default 60s).
        avg_recovery_time_s : float
            Average time to recover from a failure in seconds (default 300s).
            Includes checkpoint reload, process restart, and re-warmup.
        """
        # Series-system reliability: any single component failure stalls the
        # whole synchronous job, so failure rates ADD — first across a node's
        # GPUs/NICs/PSUs, then across all nodes. Fleet MTBF therefore shrinks
        # roughly as 1/N with cluster size.
        node_mtbf = calc_mtbf_node(
            gpu_mtbf_h=Reliability.Gpu.mttf_hours, n_gpus=fleet.node.accelerators_per_node,
            nic_mtbf_h=Reliability.Nic.mttf_hours, n_nics=fleet.node.nics_per_node,
            psu_mtbf_h=Reliability.Psu.mttf_hours, n_psus=fleet.node.psus_per_node,
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
