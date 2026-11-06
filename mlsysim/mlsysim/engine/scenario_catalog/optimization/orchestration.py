"""Named optimization scenarios for orchestration.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Clusters as _Clusters
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Fabrics as _Fabrics, Nodes as _Nodes
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_TRAINING_STATE_MIXED_PRECISION as _BYTES_TRAINING_STATE
from ....literature.registry import Literature as _Literature
from ....systems.reliability import Reliability as _Reliability
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st
from ....physics.memory import calc_checkpoint_size as _calc_checkpoint_size
from ..base.applications import DenseLLMPretrainingReference


# --- Fleet orchestration: queueing, packing, interruption, and sharing (added 2026-10-20) ---
Mg1MixedJobQueueAt80 = _st.QueueWaitScenario(
    name="A fleet at 80 percent load with widely mixed job durations (M/G/1)",
    description="One-server queue model; job durations spread three standard deviations per mean, "
    "as when minute-long debugging jobs share a fleet with week-long training runs.",
    utilization=0.8,  # scenario assumption: offered load
    service_cv=3,  # scenario assumption: representative mixed ML queue
)


Mg1ExponentialJobQueueAt80 = Mg1MixedJobQueueAt80.variant(
    name="A fleet at 80 percent load with exponential job durations (M/G/1)",
    description="The reference case of the one-server queue model: exponential service, C_s = 1.",
    service_cv=1,
)


DrfTwoTenantExample = _st.DominantShareScenario(
    name="Dominant-resource fairness for two tenants on a small cluster",
    description="Tenant A's one-GPU replica and tenant B's two-GPU worker, filled progressively "
    "on 8 GPUs, 32 cores, 256 GB, and 80 Gb/s.",
    # scenario assumption: a deliberately small teaching cluster and two bundles
    capacity={"accelerators": 8, "cpu_cores": 32, "memory_gb": 256, "network_gbps": 80},
    bundles={
        "A": {"accelerators": 1, "cpu_cores": 4, "memory_gb": 32, "network_gbps": 10},
        "B": {"accelerators": 2, "cpu_cores": 2, "memory_gb": 16, "network_gbps": 20},
    },
)


SixGpuJobsOnDgxH100Nodes = _st.NodePackingScenario(
    name="Six-GPU jobs on 64 eight-GPU nodes",
    description="Each node fits one six-GPU job and strands two GPUs.",
    nodes=64,  # scenario assumption
    gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
    job_gpus=6,  # scenario assumption
)


FragmentedEightNodeSnapshot = _st.NodePackingScenario(
    name="A fragmented eight-node snapshot with 19 free GPUs",
    description="Free GPUs scattered so that every node keeps a busy slot; an eight-GPU job cannot start.",
    nodes=8,  # scenario assumption: illustrative layout
    gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
    job_gpus=_Nodes.DGX_H100.accelerators_per_node,
    free_slots=19,  # scenario assumption: illustrative layout
)


ReservedGpusIdleThreeWeeks = _st.IdleCapacityCostScenario(
    name="500 reserved GPUs held idle for three weeks",
    description="A team's reservation waits on data; the guarantee is not lent.",
    gpus=500,  # scenario assumption
    duration=Q_(3, "week"),  # scenario assumption
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
)


PartialGangHold = _st.IdleCapacityCostScenario(
    name="A pretraining gang holding 900 of its GPUs for an hour",
    description="The reference pretraining gang waits for its last members while the GPUs it holds idle.",
    gpus=900,  # scenario assumption: GPUs already held while the gang assembles
    duration=Q_(1, "hour"),
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    gang_gpus=DenseLLMPretrainingReference.system.total_accelerators,
)


Training10KIdleHour = _st.IdleCapacityCostScenario(
    name="The 10,000-GPU training cluster idle for an hour",
    description="What an hour of a cluster-wide hold-and-wait deadlock costs.",
    gpus=_Clusters.Training_10K.total_accelerators,
    duration=Q_(1, "hour"),
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
)


MixedFleetPolicyFix1K = _st.UtilizationGainScenario(
    name="A three-pool 1,024-GPU cluster raised from 60 to 84 percent utilization",
    description="Capability matching, memory profiling, and backfill on a fleet of A100-80 GB, "
    "A100-40 GB, and V100-32 GB pools bought in three rounds.",
    fleet_gpus=_Clusters.Training_1K.total_accelerators,
    utilization_before=0.60,  # scenario assumption: observed pre-fix utilization
    utilization_after=0.84,  # scenario assumption: observed post-fix utilization
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    # scenario assumption: pool inventory and per-pool measurements before and after
    pools={"A100-80GB": 400, "A100-40GB": 424, "V100-32GB": 200},
    pool_metrics={
        "A100-80GB": {"allocated_before": 0.95, "active_before": 0.72, "allocated_after": 0.89, "active_after": 0.94},
        "A100-40GB": {"allocated_before": 0.64, "allocated_after": 0.88},
        "V100-32GB": {"allocated_before": 0.0, "allocated_after": 0.71},
    },
    pending_jobs=50,  # scenario assumption: queue depth before the fix
    job_memory_need=Q_(35, "GB"),  # scenario assumption: peak memory of the template-bound jobs
)


ForcedPreemption175BFineTune64 = _st.PreemptionTaxScenario(
    name="Forced preemptions of a 64-GPU fine-tuning run of the 175B model",
    description="A low-priority run with no preemption budget, stopped without a clean save.",
    n_devices=_Clusters.Lab_64_H100.total_accelerators,
    checkpoint_bytes=_calc_checkpoint_size(_Models.Language.GPT3.parameters),  # 14 B/param resumable state
    lost_work=_ReferenceStats.ForcedPreemptionProfile.LostWork,
    preemption_rate=_ReferenceStats.ForcedPreemptionProfile.UnbudgetedRate,
    recovery=_Reliability.Recovery,
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
)


SpotInterruptionEffectiveCost = _st.SpotEffectiveCostScenario(
    name="Spot capacity priced per hour of retained progress",
    description="A 65 percent nominal discount, with a graceful save to slow storage and a restart "
    "after every two hours of retained progress.",
    on_demand_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    spot_rate=_Infrastructure.Pricing.Fleet.SpotGpuHourRef.rate,
    checkpoint_overhead=_ReferenceStats.SpotInterruptionProfile.CheckpointOverhead,
    restart_overhead=_ReferenceStats.SpotInterruptionProfile.RestartOverhead,
    retained_interval=_ReferenceStats.SpotInterruptionProfile.RetainedInterval,
)


Llama2_70BSpotRun512 = _st.SpotRunScenario(
    name="A 14-day 70B training run on 512 H100s, on demand against spot",
    description="Spot adds checkpoint overhead and a restart per interruption.",
    fleet=_Clusters.Training_512_H100,
    workload=_Models.Language.Llama2_70B,
    run_duration=Q_(14, "day"),  # scenario assumption: 14-day run
    on_demand_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    spot_rate=_Infrastructure.Pricing.Fleet.SpotGpuHourRef.rate,
    checkpoint_overhead_fraction=0.05,  # scenario assumption
    interruptions_per_day=14,  # scenario assumption: fixed budget per day of useful training
    restart_per_interruption=Q_(0.5, "hour"),  # scenario assumption
)


ElasticResizeDecision = _st.ElasticResizeScenario(
    name="Start small, wait for 512 GPUs, or step up to 256 over 24 hours",
    description="One job's three schedules over a 24-hour window (ReferenceStats.ElasticResizeProfile).",
    window=_ReferenceStats.ElasticResizeProfile.Window,
    queue_wait=_ReferenceStats.ElasticResizeProfile.QueueWait,
    rescale_overhead=_ReferenceStats.ElasticResizeProfile.RescaleOverhead,
    small_gpus=int(_ReferenceStats.ElasticResizeProfile.SmallGpus),
    step_up_gpus=int(_ReferenceStats.ElasticResizeProfile.StepUpGpus),
    full_gpus=int(_ReferenceStats.ElasticResizeProfile.FullGpus),
    small_throughput=float(_ReferenceStats.ElasticResizeProfile.SmallThroughput),
    step_up_throughput=float(_ReferenceStats.ElasticResizeProfile.StepUpThroughput),
    full_throughput=float(_ReferenceStats.ElasticResizeProfile.FullThroughput),
)


ServingConsolidationA100 = _st.ServingConsolidationScenario(
    name="100 7B and 10 175B serving models on A100s, exclusive against shared pairs",
    description="Small models share a device in pairs when load tests allow; each large model "
    "takes an 8-GPU tensor-parallel group either way.",
    # branded decimal budget: the model footprints in this scenario are decimal GB
    device_memory=_Hardware.Cloud.A100.memory.capacity.to("GiB").magnitude * Q_(1, "GB"),
    partition_memory=next(p for p in _Hardware.Cloud.A100.mig_profiles if p.name == "3g.40gb")
    .gpu_memory.to("GiB")
    .magnitude
    * Q_(1, "GB"),
    small_model=_Models.Language.Llama2_7B,
    small_footprint=Q_(26, "GB"),  # scenario assumption: weights + KV cache + activations
    tenants=2,  # scenario assumption: models per shared GPU
    n_small=100,  # scenario assumption
    large_model=_Models.Language.GPT3,
    n_large=10,  # scenario assumption
    gpus_per_large=8,  # scenario assumption: one 8-GPU tensor-parallel group per replica
    bytes_per_param=_BYTES_FP16,
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
)


DiurnalServingLending1K = _st.DiurnalLendingScenario(
    name="A 1,024-GPU fleet lending its serving reservation to training by the day",
    description="Serving reserves its forecast peak plus a surge buffer; lending returns the rest to training.",
    fleet_gpus=_Clusters.Training_1K.total_accelerators,
    serving_reservation=400,  # scenario assumption: static serving reservation
    serving_average=150,  # scenario assumption: average serving demand
    surge_buffer=117,  # scenario assumption: ready headroom above the forecast
    training_base=618,  # scenario assumption: training pool after reservation and headroom
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
)


TensorParallelExchange1GB = _st.TransferScenario(
    name="A 1 GB tensor-parallel exchange inside a node and across nodes",
    description="One activation exchange over H100 NVLink (one direction) and one NDR InfiniBand port.",
    payload=Q_(1, "GB"),  # scenario assumption: activation size
    links={
        "NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        "InfiniBand NDR": _Fabrics.InfiniBand_NDR.bandwidth,
    },
)


TeamQuotaLending500 = _st.QuotaLendingScenario(
    name="A 500-GPU guarantee used at 60 percent, its idle 200 lent to a sibling team",
    description="Hierarchical quota: the owner keeps the guarantee; the borrower runs preemptible work on the rest.",
    guarantee=500,  # scenario assumption
    used_share=0.60,  # scenario assumption
    borrower_guarantee=300,  # scenario assumption
)


Gpt3SelfContainedReplicaDgxH100 = _st.SelfContainedReplicaScenario(
    name="The smallest self-contained 175B training replica on DGX H100 nodes",
    description="TP across one node, pipeline depth the smallest power of two that fits "
    "mixed-precision Adam state in HBM without sharding across replicas.",
    workload=_Models.Language.GPT3,
    fleet=DenseLLMPretrainingReference.system,
    bytes_per_param=_BYTES_TRAINING_STATE,
)


# Added 2026-11-03 (add-only). One definition of the 175B reference job's
# training step for every chapter: TP8 x PP8 x DP16 on the 1,024-GPU
# reference cluster, 16-bit mixed precision with unsharded mixed-precision
# Adam state, and one step = one optimizer update over the global batch.
DenseLLMPretrainingStep = _st.ReferenceTrainingStepScenario(
    name="One optimizer step of the 175B reference job on 1,024 H100 GPUs",
    description="TP across one node, PP8, DP16; 16-bit mixed precision; one optimizer update "
    "over the global batch, one sequence per microbatch. Step time is modeled.",
    replica=Gpt3SelfContainedReplicaDgxH100,
    fleet=_Clusters.Training_1K_NDR,  # the reference cluster as Volume II describes it (NDR)
    batch_tokens=float(_Literature.TrainingRuns.Gpt3BatchTokens),  # Brown et al. (2020), Table 2.1
    batch_divisor=64,  # scenario assumption: divisible by every DP degree from 1 to 64 (ch08 sweep)
    sequences_per_microbatch=1,  # scenario assumption
    precision="fp16",  # 16-bit mixed precision (BF16 or FP16 have the same width)
    local_efficiency=float(_ReferenceStats.TrainingScaleProfiles.LocalComputeEfficiency),
    overlap_comm=True,  # scenario assumption: solver default 85% DP overlap
)


# Added 2026-11-03 (add-only). The same step on the 8,192-GPU frontier tier
# (128 replicas, same global batch), to show the coordination cost growing with scale.
DenseLLMPretrainingStep8K = _st.ReferenceTrainingStepScenario(
    name="One optimizer step of the 175B reference job on 8,192 H100 GPUs",
    description="The reference step's layout, batch, and precision on 128 replicas instead of 16.",
    replica=Gpt3SelfContainedReplicaDgxH100,
    fleet=_Clusters.Frontier_8K,  # DGX H100 nodes on NDR InfiniBand
    batch_tokens=float(_Literature.TrainingRuns.Gpt3BatchTokens),  # Brown et al. (2020), Table 2.1
    batch_divisor=64,  # scenario assumption: same global batch as the reference step
    sequences_per_microbatch=1,  # scenario assumption
    precision="fp16",
    local_efficiency=float(_ReferenceStats.TrainingScaleProfiles.LocalComputeEfficiency),
    overlap_comm=True,  # scenario assumption: solver default 85% DP overlap
)


# Added 2026-11-03 (add-only). The same step on a cluster twice the reference
# size (32 replicas), for the storage chapter's running cluster.
DenseLLMPretrainingStep2K = _st.ReferenceTrainingStepScenario(
    name="One optimizer step of the 175B reference job on 2,048 H100 GPUs",
    description="The reference step's layout, batch, and precision on 32 replicas instead of 16.",
    replica=Gpt3SelfContainedReplicaDgxH100,
    fleet=_Clusters.Production_2K_NDR,
    batch_tokens=float(_Literature.TrainingRuns.Gpt3BatchTokens),  # Brown et al. (2020), Table 2.1
    batch_divisor=64,  # scenario assumption: same global batch as the reference step
    sequences_per_microbatch=1,  # scenario assumption
    precision="fp16",
    local_efficiency=float(_ReferenceStats.TrainingScaleProfiles.LocalComputeEfficiency),
    overlap_comm=True,  # scenario assumption: solver default 85% DP overlap
)
