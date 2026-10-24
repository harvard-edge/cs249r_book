"""Cluster orchestration and queueing solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations


import math

from ..results import (
    OrchestrationResult,
    PreemptionTaxResult,
)
from ...core.units import ureg, Q_
from ...systems.types import Fleet
from .base import ForwardModel

class OrchestrationModel(ForwardModel):
    """
    Analyzes Cluster Orchestration and Queueing (Little's Law).

    **Caveat:** This model uses a pedagogical M/D/1 queue (single server, deterministic
    service) to establish macroscopic wait-time bounds for dedicated, monolithic cluster workloads.
    For detailed multi-tenant job packing and preemption, a discrete-event M/G/c
    scheduler simulation would be required.

    This model simulates the 'Wait Wall' in shared research clusters,
    calculating job completion times and researcher wait times based on
    cluster utilization and arrival rates.

    Literature Source:
    1. Little (1961), "A Proof for the Queuing Formula: L = λW."
    2. Barroso et al. (2018), "The Datacenter as a Computer" (Cluster Mgmt).
    3. Jeon et al. (2019), "Analysis of Large-Scale Multi-Tenant GPU Clusters."
    """
    requires = ("fleet",)
    produces = OrchestrationResult

    def solve(self, fleet: Fleet, arrival_rate_jobs_per_day: float, avg_job_duration_days: float) -> OrchestrationResult:
        """
        Solves for cluster wait times and utilization.

        Parameters
        ----------
        fleet : Fleet
            The hardware cluster configuration.
        arrival_rate_jobs_per_day : float
            λ: Rate at which new training jobs are submitted.
        avg_job_duration_days : float
            The average time a job takes to run if it has the whole cluster.

        Returns
        -------
        OrchestrationResult
            Wait time, queue length, utilization, and stability metrics.
        """
        # ρ = λ / μ  (Utilization)
        # μ = 1 / avg_duration

        lambda_rate = arrival_rate_jobs_per_day
        mu_rate = 1.0 / avg_job_duration_days

        utilization = lambda_rate / mu_rate

        # M/D/1 Queue approximation for wait time (Fixed duration jobs)
        # T_wait = ρ / (2μ(1-ρ))  — half the M/M/1 wait, because deterministic
        # service removes the service-time variance term.
        if utilization < 1.0:
            wait_time_days = utilization / (2 * mu_rate * (1 - utilization))
        else:
            # ρ >= 1: arrivals outpace service; the queue grows without bound.
            wait_time_days = float('inf')

        return OrchestrationResult(
            cluster_utilization=utilization,
            avg_wait_time_days=Q_(wait_time_days, ureg.day),
            avg_queue_length=utilization**2 / (2 * (1 - utilization)) if utilization < 1.0 else float('inf'),
            is_stable=utilization < 1.0,
        )



class PreemptionTaxModel(ForwardModel):
    """
    The cost of forced preemptions to the job that is stopped.

    One forced preemption discards the work since the last durable checkpoint
    and then pays the cold-restart steps of the recovery budget: restart,
    checkpoint reload, and warmup. There is no detection term, because the
    scheduler itself stopped the job. A preempted job and a failed job
    therefore pay the same relaunch.
    """
    requires = ("recovery", "checkpoint_bytes")
    produces = PreemptionTaxResult

    def solve(self, n_devices: int, checkpoint_bytes, lost_work, preemption_rate, recovery,
              device_hour_rate) -> PreemptionTaxResult:
        restart = Q_(float(recovery.restart_time_s), ureg.second)
        reload = (checkpoint_bytes / Q_(float(recovery.checkpoint_write_bw_gbs), "GB/s")).to(ureg.second)
        warmup = Q_(float(recovery.warmup_time_s), ureg.second)
        relaunch = recovery.cold_restart_time(checkpoint_bytes, include_detection=False)
        per_event = (lost_work + relaunch).to(ureg.minute)
        rate = device_hour_rate.to("USD/hour").magnitude
        cost = n_devices * per_event.to(ureg.hour).magnitude * rate
        events = (preemption_rate * Q_(1, ureg.day)).to("").magnitude
        lost_day = events * per_event
        return PreemptionTaxResult(lost_work=lost_work, restart=restart, reload=reload, warmup=warmup,
                                   relaunch=relaunch, per_event=per_event, cost_per_event=cost,
                                   events_per_day=events, lost_per_day=lost_day,
                                   gpu_hours_per_day=n_devices * lost_day.to(ureg.hour).magnitude)


__all__ = [
    "OrchestrationModel",
    "PreemptionTaxModel",
]
