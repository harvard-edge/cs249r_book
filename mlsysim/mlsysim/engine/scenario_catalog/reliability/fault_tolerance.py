"""Named reliability scenarios for fault tolerance.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....systems.registry import Clusters as _Clusters
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Nodes as _Nodes
from ....systems.reliability import Reliability as _Reliability
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st
from ..training.data_parallel import FourNinesAvailability10K


# --- Fault tolerance: failure clocks, checkpoints, recovery, serving (added 2026-10-20) ---
GpuFailureLadder = _st.FailureIntervalSweepScenario(
    name="GPU-only failure interval from 8 to 25,000 GPUs",
    description="Independent GPU failures at the per-GPU MTTF; the job's interval is MTTF / N.",
    component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
    sizes=[8, 64, 512, 1000, 4000, 10_000, 25_000],  # scenario assumption: illustrative job sizes
)


FullSystemFailures10K = _st.SeriesFailureScenario(
    name="Full-system failure interval of the 10,000-GPU cluster",
    description="GPU, PCIe link, and each node's network path (NIC and cables) in series; the "
    "DGX H100's 4+2 redundant power supplies add no single-failure term.",
    n_devices=_Clusters.Training_10K.total_accelerators,
    devices_per_node=_Nodes.DGX_H100.accelerators_per_node,
    component_mttf={
        "gpu": Q_(_Reliability.Gpu.mttf_hours, "hour"),
        "pcie": Q_(_Reliability.PcieSwitch.mttf_hours, "hour"),
        "network": Q_(_Reliability.Nic.mttf_hours, "hour"),
        "power": Q_(_Reliability.Psu.mttf_hours, "hour"),
    },
    per_node={
        "gpu": _Nodes.DGX_H100.accelerators_per_node,
        "pcie": _Nodes.DGX_H100.accelerators_per_node,  # one PCIe link to the host per GPU
        "network": 1,  # scenario assumption: the node's network path (NIC and cables) as one unit
        "power": _Nodes.DGX_H100.series_psus,  # 0: no single supply failure stops the node
    },
)


Gpt3CheckpointInterval10K = _st.CheckpointIntervalScenario(
    name="Young-Daly interval of the 175B job on the 10,000-GPU cluster",
    description="A 14-byte-per-parameter resumable checkpoint written at the per-job checkpoint "
    "bandwidth, against the cluster's full-system MTBF.",
    workload=_Models.Language.GPT3,
    bytes_per_param=_ReferenceStats.MixedPrecisionAdam.CheckpointBytesPerParameter,
    write_bandwidth=Q_(float(_Reliability.Recovery.checkpoint_write_bw_gbs), "GB/s"),
    # scenario assumption: the worked example carries the MTBF at the two decimals it displays (3.87 h)
    mtbf=Q_(round(FullSystemFailures10K.result.mtbf.m_as("hour"), 2), "hour"),
)


FiveMinuteCheckpoint1K = _st.CheckpointIntervalScenario(
    name="A five-minute checkpoint on 1,000 GPUs",
    description="A stated write time against the GPU-only MTBF of 1,000 GPUs.",
    write_time=Q_(5, "minute"),  # scenario assumption
    mtbf=GpuFailureLadder.intervals[1000],
)


FifteenMinuteCheckpoint1K = FiveMinuteCheckpoint1K.variant(
    name="A fifteen-minute checkpoint on 1,000 GPUs",
    write_time=Q_(15, "minute"),  # scenario assumption: slower storage or a larger model
)


Gpt3RecoveryBudget1K = _st.RecoveryBudgetScenario(
    name="Cold-restart budget of the 175B job on 1,024 GPUs",
    description="Detection, restart, checkpoint load, and warmup per failure, paid at the GPU-only failure rate.",
    workload=_Models.Language.GPT3,
    bytes_per_param=_ReferenceStats.MixedPrecisionAdam.CheckpointBytesPerParameter,
    recovery=_Reliability.Recovery,
    component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
    n_devices=_Clusters.Training_1K.total_accelerators,
)


Gpt3RecoveryBudget10K = Gpt3RecoveryBudget1K.variant(
    name="Cold-restart budget of the 175B job on 10,000 GPUs",
    n_devices=_Clusters.Training_10K.total_accelerators,
)


HotSpareNode10K = _st.HotSpareScenario(
    name="One idle DGX H100 node beside the 10,000-GPU job",
    description="A hot spare priced against one minute removed from every full-system recovery.",
    failures=FullSystemFailures10K,
    spare_devices=_Nodes.DGX_H100.accelerators_per_node,
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
)


HalfSpeedStraggler1K = _st.StragglerCostScenario(
    name="One of 1,024 GPUs thermally throttled to half speed",
    description="A compute-bound synchronous job with a 1 s step; the barrier waits for the slow rank.",
    n_devices=_Clusters.Training_1K.total_accelerators,
    step_time=Q_(1, "second"),  # scenario assumption
    slow_speed=0.5,  # scenario assumption: thermal throttling to half speed
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
)


ServingReplica99 = _st.AvailabilityScenario(
    name="One serving replica at 99 percent",
    description="Independent replicas; the service is up while any one is up.",
    availability=0.99,  # scenario assumption: per-replica availability
    n_components=1,
    replicas=1,
)


ServingReplicas99x2 = ServingReplica99.variant(name="Two serving replicas at 99 percent", replicas=2)


ServingReplicas99x3 = ServingReplica99.variant(name="Three serving replicas at 99 percent", replicas=3)


GpuAvailability10K = FourNinesAvailability10K.variant(
    name="10,000 GPUs at 99.99 percent",
    description="Illustrative per-GPU availability; a synchronous job needs every GPU up.",
    n_components=_Clusters.Training_10K.total_accelerators,
)


InferenceReadinessProbe = _st.HealthProbeScenario(
    name="A readiness probe on an inference replica",
    description="Probe every 5 s with a 2 s timeout; three misses remove the replica, two clean probes readmit it.",
    interval=Q_(5, "second"),  # scenario assumption: periodSeconds
    timeout=Q_(2, "second"),  # scenario assumption: timeoutSeconds
    failure_threshold=3,  # scenario assumption
    success_threshold=2,  # scenario assumption
)


SdcExposure10K = _st.SdcExposureScenario(
    name="Silent-corruption exposure of the 10,000-GPU cluster",
    description="The generic per-GPU-hour SDC rate across every GPU of a 2 s synchronous step, over a three-month run.",
    n_devices=_Clusters.Training_10K.total_accelerators,
    event_rate=_Reliability.SdcRatePerGpuHr,
    step_time=Q_(2, "second"),  # scenario assumption: illustrative step time
    run_duration=Q_(90, "day"),  # scenario assumption: a three-month run
)


SdcExposure100K = SdcExposure10K.variant(
    name="Silent-corruption exposure of a 100,000-GPU fleet",
    n_devices=10 * _Clusters.Training_10K.total_accelerators,  # scenario assumption: ten times the cluster
    run_duration=None,
)
