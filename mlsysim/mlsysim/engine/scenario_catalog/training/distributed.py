"""Named training scenarios for distributed.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Clusters as _Clusters
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Fabrics as _Fabrics, Nodes as _Nodes
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....literature.registry import Literature as _Literature
from ....systems.reliability import Reliability as _Reliability
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st


# --- Distributed training (added 2026-10-20) ----------------------------
# The 175B reference model on the 1,024-GPU reference cluster, laid out
# four ways. All share one step: 2M tokens per update at 6 FLOPs per
# parameter per token, H100 FP8 peak at 50 percent, FP16 gradients over one
# NDR port per GPU, and up to 75 percent of each AllReduce hidden under
# compute. The compute window is carried at 0.1 s, the precision the napkin
# math prints, so every downstream ratio recomputes from the printed value.
Gpt3DataParallel1K = _st.ParallelLayoutScenario(
    name="175B pure data parallelism on 1,024 H100s",
    description="Every GPU holds the full model and AllReduces the full FP16 gradient over NDR.",
    workload=_Models.Language.GPT3,
    dp_size=_Clusters.Training_1K.total_accelerators,
    batch_tokens=2e6,  # scenario assumption: tokens per optimizer update
    flops_per_param_token=float(_Literature.Chinchilla.ComputeConstant),
    accelerator=_Hardware.Cloud.H100,
    precision="fp8",
    local_efficiency=0.5,  # scenario assumption: local share of peak before communication
    link_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    overlap_fraction=0.75,  # scenario assumption: optimistic overlap schedule
    window_resolution=Q_(0.1, "second"),
    # Budget against the published "80 GB" label (80 GiB, labeled GB by the vendor).
    device_memory=Q_(_Hardware.Cloud.H100.memory.capacity.to("GiB").magnitude, "GB"),
)


Gpt3TensorDataParallel1K = Gpt3DataParallel1K.variant(
    name="175B with TP 8 inside each node and DP 128 across nodes",
    description="Each rank AllReduces one eighth of the gradient; state is split eight ways.",
    tp_size=_Nodes.DGX_H100.accelerators_per_node,
    dp_size=_Clusters.Training_1K.total_accelerators // _Nodes.DGX_H100.accelerators_per_node,
)


Gpt3ThreeDParallel1K = Gpt3DataParallel1K.variant(
    name="175B with TP 8, PP 4, and DP 32 on 1,024 H100s",
    description="Adds four pipeline stages at 32 microbatches; the DP shard falls to 1/32.",
    tp_size=_Nodes.DGX_H100.accelerators_per_node,
    pp_size=4,  # scenario assumption: pipeline stages across nodes
    dp_size=_Clusters.Training_1K.total_accelerators // (4 * _Nodes.DGX_H100.accelerators_per_node),
    microbatches=32,  # scenario assumption: microbatches per step
)


Gpt3ThreeDParallel16K = Gpt3ThreeDParallel1K.variant(
    name="175B with TP 8, PP 16, and DP 128 on 16,384 H100s",
    description="A sixteen-stage layout at the same 32 microbatches; a deeper pipeline, a larger bubble.",
    pp_size=16,  # scenario assumption
    dp_size=128,  # scenario assumption
)


Gpt3ActivationsMicrobatch4 = _st.ActivationMemoryScenario(
    name="175B activations at four sequences per microbatch",
    description="Full-context GPT-3 activations per replica, with and without recomputation.",
    workload=_Models.Language.GPT3,
    micro_batch=4,  # scenario assumption: per-GPU microbatch
)


Gpt3ActivationsMicrobatch1 = Gpt3ActivationsMicrobatch4.variant(
    name="175B activations at one sequence per microbatch", micro_batch=1
)  # scenario assumption: single-sequence microbatch


Llama2_7B_ZeroStages64 = _st.ZeroShardingScenario(
    name="7B ZeRO stages on 64 GPUs",
    description="Mixed-precision Adam state of a 7B model replicated, then sharded stage by stage.",
    workload=_Models.Language.Llama2_7B,
    n_devices=64,  # scenario assumption: data-parallel group
)


Gpt3Zero3On64 = _st.ZeroShardingScenario(
    name="175B fully sharded over 64 GPUs",
    description="ZeRO-3 state plus fully checkpointed activations of one sequence; every layer "
    "regathered over one NDR port per GPU.",
    workload=_Models.Language.GPT3,
    n_devices=64,  # scenario assumption: one sixteenth of the reference cluster
    activations=Gpt3ActivationsMicrobatch1,
    activation_strategy="full",
    link_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
)


Gpt2DataParallelOneNode = _st.DataParallelScaleOutScenario(
    name="GPT-2 data parallelism on one DGX H100 node",
    description="Eight GPUs at the one-GPU profile's local batch, gradients AllReduced over NVLink.",
    workload=_Models.Language.GPT2,
    step_time=_ReferenceStats.DataParallelScalingProfile.SingleGpuStepTime,
    training_time=_ReferenceStats.DataParallelScalingProfile.SingleGpuTrainingTime,
    per_device_batch=int(_ReferenceStats.DataParallelScalingProfile.PerGpuBatch),
    devices_per_node=_Nodes.DGX_H100.accelerators_per_node,
    intra_bandwidth=_Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
    node_hour_rate=_Nodes.DGX_H100.accelerators_per_node * _Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
)


Gpt2DataParallelFourNodes10GbE = Gpt2DataParallelOneNode.variant(
    name="GPT-2 data parallelism on four nodes over 10 Gb Ethernet",
    description="The same local batch on 32 GPUs; the cross-node ring runs at 10 GbE.",
    nodes=4,  # scenario assumption: four-node scale-out
    inter_bandwidth=_Fabrics.Ethernet_10G.bandwidth,
)


Gpt2GradientAccumulationOneNode = Gpt2DataParallelOneNode.variant(
    name="GPT-2 on one node with four-step gradient accumulation",
    description="Reaches the four-node global batch without leaving NVLink.",
    accumulation_steps=4,  # scenario assumption
)


GPipeFourStageWalkthrough = _st.PipelineScheduleScenario(
    name="GPipe, four stages and four microbatches",
    description="The schedule drawn slot by slot: 14 slots per step, 6 idle per stage.",
    stages=4,
    microbatches=4,
)  # scenario assumption: the drawn schedule


Pipeline8Stage32Micro = _st.PipelineScheduleScenario(
    name="Eight pipeline stages at 32 microbatches",
    description="A cross-node pipeline of eight stages.",
    stages=8,
    microbatches=32,
)


Gpt3TensorParallelLayerTp8 = _st.TensorParallelLayerScenario(
    name="One 175B layer under 8-way tensor parallelism",
    description="One 2,048-token sequence at half the H100 dense FP16 peak; two activation "
    "AllReduces per layer over NVLink or one NDR port.",
    workload=_Models.Language.GPT3,
    tp_size=_Nodes.DGX_H100.accelerators_per_node,
    tokens=_Models.Language.GPT3.context_length,  # scenario assumption: one sequence per microbatch
    accelerator=_Hardware.Cloud.H100,
    local_efficiency=0.5,  # scenario assumption: same local efficiency as the data-parallel step
    links={
        "NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        "InfiniBand NDR": _Fabrics.InfiniBand_NDR.bandwidth,
    },
)


H100RingAttentionChunk = _st.RingAttentionScenario(
    name="Ring attention chunk size on H100",
    description="FP16 keys and values at half the dense FP16 peak, over NVLink or one NDR port.",
    accelerator=_Hardware.Cloud.H100,
    local_efficiency=0.5,  # scenario assumption: 50 percent of dense peak
    element_bytes=_BYTES_FP16,
    links={
        "NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        "InfiniBand NDR": _Fabrics.InfiniBand_NDR.bandwidth,
    },
)


# Scaling a transformer from 8 to 64 workers at 32 examples per worker.
# Iteration counts and communication shares are hypothetical inputs, not
# measured results; one 0.4 s local iteration is shared by every run.
TransformerBsp8Workers = _st.SyncStrategyScenario(
    name="Eight workers, bulk synchronous",
    description="12,800 iterations to the target at 15 percent communication overhead.",
    workers=8,
    per_worker_batch=32,
    iterations=12_800,
    communication_overhead=0.15,  # scenario assumption
    base_iterations=100_000,  # scenario assumption: one worker to the same target
    base_iteration_time=Q_(0.4, "second"),  # scenario assumption: local compute per iteration
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
)


TransformerBsp64Workers = TransformerBsp8Workers.variant(
    name="64 workers, bulk synchronous",
    description="1,720 iterations at 45 percent communication overhead.",
    workers=64,
    iterations=1_720,
    communication_overhead=0.45,
)  # scenario assumption


TransformerSsp64Workers = TransformerBsp8Workers.variant(
    name="64 workers, stale synchronous (s = 4)",
    description="Less waiting, 25 percent overhead, but stale updates need 2,100 iterations.",
    workers=64,
    iterations=2_100,
    communication_overhead=0.25,
    staleness=4,
)  # scenario assumption


Llama2_70B_PpoAlignment = _st.AlignmentMemoryScenario(
    name="PPO and DPO alignment of a 70B policy",
    description="70B policy and reference with 13B reward and value models; policy under "
    "TP 8 x PP 4 with 8 replicas; rollouts of 256 sequences of 1,024 tokens.",
    policy=_Models.Language.Llama2_70B,
    critic=_Models.Language.Llama2_13B,
    tp_size=8,
    pp_size=4,
    dp_size=8,  # scenario assumption: policy layout on 256 H100s
    rollout_sequences=256,
    rollout_tokens=1024,  # scenario assumption
    device_memory=Q_(_Hardware.Cloud.H100.memory.capacity.to("GiB").magnitude, "GB"),
)


GpuFailuresTraining1K = _st.FailureIntervalScenario(
    name="GPU failures across the 1,024-GPU reference cluster",
    description="Independent GPU failures at the per-GPU MTTF over a two-week run.",
    component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
    n_components=_Clusters.Training_1K.total_accelerators,
    run_duration=Q_(14, "day"),  # scenario assumption: two-week training run
)


StragglerFreeStep1K = _st.AvailabilityScenario(
    name="A synchronous step on 1,000 workers, each late 1 percent of the time",
    description="A step is on time only when every worker is; independent stragglers.",
    availability=0.99,  # scenario assumption: per-worker on-time probability per step
    n_components=1000,  # scenario assumption
)


ResidualOnePercent512 = _st.ResidualShareScenario(
    name="A 1 percent residual on 512 devices",
    description="A residual equal to 1 percent of the one-device step fills most of the distributed step.",
    residual=0.01,  # scenario assumption: residual fraction of the one-device step
    devices=512,
)  # scenario assumption
