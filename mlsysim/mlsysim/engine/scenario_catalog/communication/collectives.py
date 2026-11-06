"""Named communication scenarios for collectives.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Fabrics as _Fabrics, Nodes as _Nodes
from ....systems.registry import SwitchFabric as _SwitchFabric
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_FP32 as _BYTES_FP32, BYTES_INT8 as _BYTES_INT8
from ... import scenario_types as _st
from ..base.applications import DenseLLMPretrainingReference


# --- Collective communication (added 2026-10-20) ------------------------
CriticalSizeByInterconnect = _st.AlphaBetaScenario(
    name="Critical message size n* = alpha beta on five interconnects",
    description="Effective collective-launch alpha at the midpoint of each registered range (NDR at its "
    "canonical anchor) and each link's per-direction bandwidth.",
    alphas={
        "NVLink": (_SwitchFabric.EffectiveAlphaNvlinkLow + _SwitchFabric.EffectiveAlphaNvlinkHigh) / 2,
        "PCIe Gen5": (_SwitchFabric.EffectiveAlphaPcieGen5Low + _SwitchFabric.EffectiveAlphaPcieGen5High) / 2,
        "RoCE": (_SwitchFabric.EffectiveAlphaRoceLow + _SwitchFabric.EffectiveAlphaRoceHigh) / 2,
        "HDR": (_SwitchFabric.EffectiveAlphaHdrLow + _SwitchFabric.EffectiveAlphaHdrHigh) / 2,
        "NDR": _SwitchFabric.AlphaNdr,
    },
    bandwidths={
        "NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        "PCIe Gen5": _Hardware.Cloud.H100.interconnect.bandwidth,
        "RoCE": _Fabrics.Ethernet_100G.bandwidth,
        "HDR": _Fabrics.InfiniBand_HDR.bandwidth,
        "NDR": _Fabrics.InfiniBand_NDR.bandwidth,
    },
    messages={},
)


RoutingMessageFusionNdr = _st.MessageFusionScenario(
    name="One hundred 4 KB routing messages on NDR, separately and fused",
    description="Each 4 KB message sits far below n*, so alone it costs its startup; fused into one "
    "400 KB message the batch pays one startup and its bytes.",
    alpha=_SwitchFabric.AlphaNdr,
    bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    message=Q_(4, "KB"),  # scenario assumption: MoE routing message
    count=100,
)  # scenario assumption: messages fused into one


DdpBucketRingNdr = _st.AllReduceScheduleScenario(
    name="One 25 MB gradient bucket reduced on 64 and on 1,024 NDR ranks",
    description="DDP's common bucket size; the ring's per-round chunk M/N decides the regime.",
    message=Q_(25, "MB"),  # scenario assumption: DDP's common bucket reference size
    group_sizes=[
        64,  # scenario assumption: eight DGX H100 nodes
        DenseLLMPretrainingReference.system.total_accelerators,
    ],
    alpha=_SwitchFabric.AlphaNdr,
    bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
)


GradientBufferAllReduceNdr = _st.AllReduceScheduleScenario(
    name="The 1 GB reference gradient buffer reduced on NDR at 8, 64, and 1,024 ranks",
    description="Ring, simple tree, and parameter server for one 1 GB buffer over 50 GB/s ports.",
    message=_ReferenceStats.CollectiveCommunicationProfile.GradientBuffer,
    group_sizes=[
        8,  # scenario assumption: one node's group
        64,  # scenario assumption: eight DGX H100 nodes
        DenseLLMPretrainingReference.system.total_accelerators,
    ],
    alpha=_SwitchFabric.AlphaNdr,
    bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    run_steps=_ReferenceStats.CollectiveCommunicationProfile.TrainingRunSteps,
)


Llama2_70BGradientRing64Ndr = _st.AllReduceScheduleScenario(
    name="A 70B model's BF16 gradient ring-AllReduced on 64 NDR GPUs",
    description="Every rank reduces the full 140 GB gradient; the what-if halves the exposed AllReduce "
    "on every step of a 100,000-step run.",
    message=_Models.Language.Llama2_70B.size_in_bytes(_BYTES_FP16),
    group_sizes=[64],  # scenario assumption: eight DGX H100 nodes
    alpha=_SwitchFabric.AlphaNdr,
    bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    run_steps=_ReferenceStats.CollectiveCommunicationProfile.TrainingRunSteps,
    time_saved_fraction=0.5,
)  # scenario assumption: an optimization that halves the AllReduce


RingVsTreeSmallBuffer = _st.AllReduceScheduleScenario(
    name="Ring against tree for a 1 MB buffer on 64 ranks",
    description="Illustrative link: 10 microseconds of startup and 10 GB/s; the buffer sits near the crossover.",
    message=Q_(1, "MB"),  # scenario assumption
    group_sizes=[64],  # scenario assumption
    alpha=Q_(10, "microsecond"),  # scenario assumption
    bandwidth=Q_(10, "GB/s"),
)  # scenario assumption


HierarchicalAllReduce8x8DgxH100 = _st.HierarchicalAllReduceScenario(
    name="Hierarchical AllReduce of the 1 GB buffer on eight DGX H100 nodes",
    description="NVLink ReduceScatter, one NDR ring per GPU position on the 125 MB shard, NVLink AllGather.",
    message=_ReferenceStats.CollectiveCommunicationProfile.GradientBuffer,
    n_nodes=8,  # scenario assumption: a 64-GPU cluster
    gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
    intra_bandwidth=_Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
    inter_bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    inter_alpha=_SwitchFabric.AlphaNdr,
)


HierarchicalAllReduceBucket100MB = HierarchicalAllReduce8x8DgxH100.variant(
    name="Hierarchical AllReduce of one 100 MB bucket on eight DGX H100 nodes",
    description="The same three phases at the ideal rate, for one overlap bucket.",
    message=Q_(100, "MB"),
)  # scenario assumption: AllReduce bucket size


HierarchicalAllReduceReferenceCluster = HierarchicalAllReduce8x8DgxH100.variant(
    name="Hierarchical AllReduce of the 175B BF16 gradient on the 1,024-GPU reference cluster",
    description="128 DGX H100 nodes of eight; flat ring across all 1,024 GPUs for comparison.",
    message=DenseLLMPretrainingReference.workload.size_in_bytes(_BYTES_FP16),
    n_nodes=DenseLLMPretrainingReference.system.count,
    gpus_per_node=DenseLLMPretrainingReference.system.node.accelerators_per_node,
)


MoeDispatch8Gpu64Experts = _st.AllToAllDispatchScenario(
    name="MoE dispatch of 4,096 tokens to 64 experts on eight GPUs over NDR",
    description="Uniform routing: each GPU keeps 64 of its 512 tokens and sends 64 to each of seven peers; "
    "one GPU's experts draw an assumed 1.5 times the uniform remote load.",
    total_tokens=4096,  # scenario assumption
    n_devices=8,  # scenario assumption
    n_experts=64,  # scenario assumption
    hidden_dim=2048,  # scenario assumption
    bytes_per_element=_BYTES_FP16,
    alpha=_SwitchFabric.AlphaNdr,
    bandwidth=_Fabrics.InfiniBand_NDR.bandwidth,
    receive_skew=1.5,
)  # scenario assumption: realized remote-receive skew on one rank


RingTraceFourGpu = _st.RingTraceScenario(
    name="Ring AllReduce traced on four GPUs with four-element vectors",
    description="Small integers so every partial sum can be followed by hand.",
    vectors=[[1, 5, 3, 7], [2, 6, 4, 8], [3, 7, 5, 9], [4, 8, 6, 10]],
)  # scenario assumption


ErrorFeedbackFiveSteps = _st.ErrorFeedbackScenario(
    name="Error feedback on five gradients below half a unit",
    description="A compressor that rounds to whole units sends nothing without a residual.",
    gradients=[0.4, 0.3, 0.2, 0.4, 0.3],
)  # scenario assumption


CompressionPayback8x = _st.CompressionPaybackScenario(
    name="Compressing one 40 ms synchronization by an effective 8x",
    description="2 ms of encoding and decoding; on a fast NVLink domain the same buffer takes 1 ms.",
    communication_time=Q_(40, "ms"),  # scenario assumption: uncompressed synchronization
    compression_ratio=8,  # scenario assumption: effective gain after metadata and protocol overhead
    codec_time=Q_(2, "ms"),  # scenario assumption: quantize and dequantize
    fast_communication_time=Q_(1, "ms"),
)  # scenario assumption: the same buffer over NVLink


Int8CompressionRun = _st.CompressionRunScenario(
    name="INT8 compression with error feedback, time to target on a slow and a fast network",
    description="100,000 steps of 300 ms compute; INT8 against FP32 cuts communication fourfold but "
    "needs 10 percent more steps.",
    steps=100_000,  # scenario assumption: steps to target quality, uncompressed
    compute_time=Q_(300, "ms"),  # scenario assumption
    communication_times={
        "slow": Q_(200, "ms"),  # scenario assumption
        "fast": Q_(30, "ms"),
    },  # scenario assumption
    compression_ratio=(_BYTES_FP32 / _BYTES_INT8).to("").magnitude,
    step_penalty=0.10,
)  # scenario assumption: extra steps from compression noise


OverlapBudget7B64Gpu = _st.OverlapBudgetScenario(
    name="Overlap budget for a 7B transformer on 64 GPUs",
    description="32 equal layers of FP32 gradients in 100 MB buckets at an assumed 3 ms each, against "
    "15 ms of backward per layer.",
    workload=_Models.Language.Llama2_7B,
    bytes_per_value=_BYTES_FP32,
    backward_per_layer=Q_(15, "ms"),  # scenario assumption
    bucket_size=HierarchicalAllReduceBucket100MB.message,
    time_per_bucket=Q_(3, "ms"),  # scenario assumption: achieved time per bucket, final partial included
    n_devices=HierarchicalAllReduceBucket100MB.n_devices,
)


OverlapBudget7B64GpuIdealBuckets = OverlapBudget7B64Gpu.variant(
    name="The same overlap budget at the hierarchical schedule's ideal bucket time",
    description="Each 100 MB bucket at the three-phase model's time on eight DGX H100 nodes.",
    time_per_bucket=HierarchicalAllReduceBucket100MB.result.total_time,
)
