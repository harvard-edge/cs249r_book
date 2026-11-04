"""Named serving scenarios for goodput.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Nodes as _Nodes
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....literature.registry import Literature as _Literature
from ....systems.reliability import Reliability as _Reliability
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ....ops import Ops as _Ops
from ... import scenario_types as _st
from ..base.applications import DenseLLMPretrainingReference


# ── Fleet goodput: one 70B training job on a shared pool (added 2026-10-20) ──
SharedPoolSchedulingSnapshot = _st.SchedulingSnapshotScenario(
    name="Scheduling goodput of a shared pool in one utilization snapshot",
    description="Assigned capacity less the gangs still waiting for members; input waits inside "
    "complete allocations are left to runtime goodput.",
    assigned_share=float(_ReferenceStats.FleetUtilizationSnapshot.AssignedShare),
    incomplete_gang_share=float(_ReferenceStats.FleetUtilizationSnapshot.IncompleteGangShare),
    input_wait_share=float(_ReferenceStats.FleetUtilizationSnapshot.InputWaitShare),
)


Llama2_70BJobFailures128 = _st.SeriesFailureScenario(
    name="Failure interval of a 128-GPU Llama 2 70B training job",
    description="Sixteen DGX H100 nodes; GPU, PCIe link, and the node's NIC in series.",
    n_devices=16 * _Nodes.DGX_H100.accelerators_per_node,  # scenario assumption: 16 nodes of 8
    devices_per_node=_Nodes.DGX_H100.accelerators_per_node,
    component_mttf={
        "gpu": Q_(_Reliability.Gpu.mttf_hours, "hour"),
        "pcie": Q_(_Reliability.PcieSwitch.mttf_hours, "hour"),
        "network": Q_(_Reliability.Nic.mttf_hours, "hour"),
    },
    per_node={
        "gpu": _Nodes.DGX_H100.accelerators_per_node,
        "pcie": _Nodes.DGX_H100.accelerators_per_node,  # one PCIe link to the host per GPU
        "network": 1,  # scenario assumption: one NIC shared by the node's GPUs
    },
)


Llama2_70BRuntimeGoodput128 = _st.RuntimeGoodputScenario(
    name="Runtime goodput of a 128-GPU Llama 2 70B training job over one week",
    description="Startup, input waits, checkpoint waste at the Young-Daly interval, recovery, and one "
    "budgeted forced preemption a day, on one held-time denominator.",
    failures=Llama2_70BJobFailures128,
    workload=_Models.Language.Llama2_70B,
    bytes_per_param=_ReferenceStats.MixedPrecisionAdam.CheckpointBytesPerParameter,
    recovery=_Reliability.Recovery,
    input_wait=SharedPoolSchedulingSnapshot.input_wait_of_held,  # the pool's input waits, of held time
    window=Q_(1, "week"),  # scenario assumption: one launch per week of held time
    lost_work=_ReferenceStats.ForcedPreemptionProfile.LostWork,
    preemption_rate=Q_(1, "1/day"),  # scenario assumption: a per-job budget of one forced preemption a day
)


Llama2_70BFleetMfu128 = _st.FleetMfuScenario(
    name="Llama 2 70B training MFU on one DGX H100 node and on 128 H100s",
    description="The same work per GPU: 180 ms node steps at the node's batch, 245 ms steps on the "
    "shared fleet with exposed cross-node gradient reduction.",
    workload=_Models.Language.Llama2_70B,
    hardware=_Hardware.Cloud.H100,
    node_devices=_Nodes.DGX_H100.accelerators_per_node,
    fleet_devices=Llama2_70BJobFailures128.n_devices,
    node_tokens=2204,  # scenario assumption: node baseline tokens per step
    fleet_tokens=35446,  # scenario assumption: 128-GPU tokens per step
    node_step_time=Q_(180, "ms"),  # scenario assumption: node baseline, optimizer state offloaded to host
    fleet_step_time=Q_(245, "ms"),  # scenario assumption: 128-GPU step on the shared fleet
)


Llama2_70BSharedPoolGoodput128 = _st.GoodputScenario(
    name="ML productivity goodput of the 128-GPU Llama 2 70B job on the shared pool",
    description="The pool's scheduling goodput, the job's runtime goodput, and its fleet MFU, with "
    "each owner's ceiling: partial holds refilled, input waits removed, the node baseline.",
    scheduling=SharedPoolSchedulingSnapshot.scheduling_goodput,
    runtime=Llama2_70BRuntimeGoodput128.result.runtime_goodput,
    program=Llama2_70BFleetMfu128.fleet_mfu,
    ceilings={
        "S": SharedPoolSchedulingSnapshot.partial_hold_ceiling,
        "R": (Llama2_70BRuntimeGoodput128.result.runtime_goodput + Llama2_70BRuntimeGoodput128.result.input_wait_share),
        "P": Llama2_70BFleetMfu128.node_mfu,
    },
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
)


Llama3_405BGoodput16K = _st.GoodputScenario(
    name="Llama 3 405B on its dedicated 16,384-GPU cluster",
    description="Effective training time (a lower bound) as R and BF16 MFU at 16,384 GPUs as P; S = 1 "
    "on a dedicated cluster. The two come from different windows, so the product is an estimate.",
    scheduling=1.0,
    runtime=float(_Literature.TrainingRuns.Llama3_405B_EffectiveTrainingTimeFraction),
    program=float(_Literature.TrainingRuns.Llama3_405B_Mfu16K),
)


Llama3_405BOnSharedPool = Llama3_405BGoodput16K.variant(
    name="Llama 3 405B's runtime and program goodput on the shared pool",
    scheduling=SharedPoolSchedulingSnapshot.scheduling_goodput,
)


Llama2_70BDecodeFloorTp8 = _st.DecodeScenario(
    name="Batch-one decode floor of Llama 2 70B on one 8-way tensor-parallel H100 node",
    description="Each GPU streams its FP16 weight shard once per token.",
    workload=_Models.Language.Llama2_70B,
    hardware=_Hardware.Cloud.H100,
    precision="fp16",
    batch_size=1,
    tp_size=_Nodes.DGX_H100.accelerators_per_node,
)


LayerNormMeasuredH100 = _st.RooflinePointScenario(
    name="A profiled LayerNorm kernel placed on the H100 roofline",
    description="Illustrative counter readings: achieved FLOP rate and achieved HBM bandwidth.",
    hardware=_Hardware.Cloud.H100,
    achieved_rate=Q_(5, "TFLOPs/s"),  # scenario assumption: counter reading
    achieved_bandwidth=Q_(2.8, "TB/s"),  # scenario assumption: counter reading
)


Llama2_70BNodeTensorParallelStep = _st.TensorParallelScenario(
    name="Tensor-parallel AllReduces of one Llama 2 70B node-baseline training step",
    description="Two exchanges per layer forward and two backward over NVLink, bandwidth only, "
    "against the 180 ms node step.",
    workload=_Models.Language.Llama2_70B,
    tp_size=_Nodes.DGX_H100.accelerators_per_node,
    microbatch=1,
    seq_len=Llama2_70BFleetMfu128.node_tokens,  # the node step's tokens as one sequence
    collectives_per_layer=4,  # Megatron layout: 2 forward + 2 backward
    link_bandwidth=_Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
    compute_time=Llama2_70BFleetMfu128.node_step_time,
)


Llama2_70BDecodeAllReduceTp8 = _st.MessageFusionScenario(
    name="Batch-one tensor-parallel AllReduces of the Llama 2 70B serving replica",
    description="Each per-layer exchange moves tens of kilobytes over NVLink; the collective launch "
    "costs far more than the bytes.",
    alpha=_Ops.RuntimeOverheads.KernelLaunch,
    bandwidth=_Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
    # 2 x d_model activations of 2 bytes per AllReduce at batch one
    message=(2 * _Models.Language.Llama2_70B.hidden_dim * 1 * _BYTES_FP16).to("byte"),
    count=2 * _Models.Language.Llama2_70B.layers,  # two exchanges per layer per token
)


Decode13BCompilationPayback = _st.CompilationPaybackScenario(
    name="Compilation payback for a 13B decode step on one H100",
    description="A constructed decode budget: compilation keeps the matrix multiplies, removes most "
    "element-wise and launch time; default and autotuning builds; 128 one-GPU replicas.",
    token_rate=Q_(120, "1/s"),  # scenario assumption: eager tokens per second, 13B INT4 weights
    step_time=Q_(8.33, "ms"),  # scenario assumption: 1/120 s, carried at the two decimals shown
    shares={"matmul": 0.50, "elementwise": 0.35, "launch": 0.15},  # scenario assumption
    removed={"elementwise": 0.70, "launch": 0.80},  # scenario assumption
    builds={"default": Q_(1, "minute"), "autotune": Q_(15, "minute")},  # scenario assumption
    replicas=Llama2_70BJobFailures128.n_devices,  # replicas filling an allocation of the same size
    gpus_per_replica=1,
    replica_life=Q_(1, "hour"),  # scenario assumption: a replica added for one burst hour
)


NsightTraceVolume1K = _st.TraceVolumeScenario(
    name="A 5-second Nsight Systems trace of every GPU in a 1,024-GPU job",
    description="Trace volume grows with every traced device.",
    per_device=_ReferenceStats.ProfilerTraceProfile.NsightTracePerGpu,
    devices=DenseLLMPretrainingReference.system.total_accelerators,
    window=_ReferenceStats.ProfilerTraceProfile.TraceWindow,
)
