"""Cluster orchestration and queueing solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

# ruff: noqa: F401
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Type

from ..engine import Engine, PerformanceProfile
from ..results import (
    SolverResult,
    DistributedResult,
    ReliabilityResult,
    CheckpointResult,
    SustainabilityResult,
    ServingResult,
    TrainingMemoryResult,
    ServingCapacityResult,
    MoERoutingResult,
    ContinuousBatchingResult,
    WeightStreamingResult,
    TailLatencyResult,
    EconomicsResult,
    DataResult,
    TopologyResult,
    EfficiencyResult,
    TransformationResult,
    ScalingResult,
    CompressionResult,
    SynthesisResult,
    OrchestrationResult,
    InferenceScalingResult,
    SensitivityResult,
    ResponsibleEngineeringResult,
    ParallelismOptimizerResult,
    BatchingOptimizerResult,
    PlacementOptimizerResult,
)
from ...physics import (
    calc_ring_allreduce_time,
    calc_hierarchical_allreduce_time,
    calc_all_to_all_time,
    calc_bottleneck,
    calc_mtbf_cluster,
    calc_mtbf_node,
    calc_young_daly_interval,
    calc_failure_probability,
    calc_pipeline_bubble,
)
from ...core.units import ureg, Q_, resolve_precision
from ...infrastructure.registry import Infrastructure
from ...literature.registry import Literature
from ...systems.reliability import Reliability
from .. import calibration as cal
from ...core.types import Quantity
from ...models.types import Workload, TransformerWorkload, SparseTransformerWorkload
from ...hardware.types import HardwareNode
from ...systems.types import Fleet, NetworkFabric, Node
from ...infrastructure.types import Datacenter
from .base import BaseOptimizer, BaseResolver, BaseSolver, ForwardModel
from .utils import _inter_node_latency, _intra_node_latency

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

__all__ = [
    "OrchestrationModel",
]
