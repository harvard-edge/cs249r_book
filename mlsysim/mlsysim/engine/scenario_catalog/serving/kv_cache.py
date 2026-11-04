"""Named serving scenarios for kv cache.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Nodes as _Nodes
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_INT8 as _BYTES_INT8
from ....core.units import BYTES_INT4 as _BYTES_INT4
from ... import scenario_types as _st


# --- KV cache management on one DGX H100 node (added 2026-10-20) ----------
RepairRequestZones = _st.RequestZonesScenario(
    name="Repair-agent request at revision r18, laid out in zones",
    description="A 32,768-token request: system prompt, tool schemas, and workspace form the "
    "invariant shared prefix; turn history and the latest observation follow.",
    zones={  # scenario assumption: tokens per zone, in serialization order
        "System prompt": 2_048,
        "Tool schemas": 4_096,
        "Workspace": 8_192,
        "Turn history": 16_384,
        "Latest observation": 2_048,
    },
    shared_zones=3,  # system prompt, tool schemas, workspace
)


# Copied verbatim from the v2-10 inference branch; on merge keep one copy.
Llama3_70BLongContextAdmissionDgxH100 = _st.KvAdmissionScenario(
    name="128K-token requests on the 70B GQA replica (8 H100s)",
    description="FP16 weights and KV cache; HBM at the registered byte capacity, less a 20 GB runtime reserve.",
    workload=_Models.Language.Llama3_70B,
    accelerators=_Nodes.DGX_H100.accelerators_per_node,
    hbm_per_accelerator=_ReferenceStats.ServingProfiles.H100VendorMemoryBudget,
    weight_bytes_per_param=_BYTES_FP16,
    reserve=Q_(20, "GB"),  # scenario assumption: system reserve
    context_tokens=_Models.Language.Llama3_70B.context_length,
    kv_bytes_per_elem=_BYTES_FP16,
)


ChatbotSharedSystemPrompt = _st.PrefixSharingScenario(
    name="1,000 chat users sharing a 2,000-token system prompt",
    description="Full-width multi-head KV (64 heads of 128, FP16) with Llama-2 70B's 80 layers; "
    "500 unique response tokens per user.",
    workload=_Models.Language.Llama2_70B,
    prefix_tokens=2000,  # scenario assumption
    users=1000,  # scenario assumption
    unique_tokens_per_user=500,  # scenario assumption: mean response
    kv_width=_Models.Language.Llama2_70B.hidden_dim,  # full-width MHA: heads x head width
    bytes_per_elem=_BYTES_FP16,
)


Llama2_70BPrecisionDividendGqaTp8 = _st.PrecisionDividendScenario(
    name="Precision dividend of a 70B GQA model on 8 H100s",
    description="INT4 weights and an INT8 KV cache against FP16 for both, at a 4,096-token "
    "context, with a decimal 80 GB budget per GPU.",
    workload=_Models.Language.Llama2_70B,
    tp_size=int(_ReferenceStats.ServingProfiles.PrecisionDividendTensorParallelDegree),
    hbm_per_accelerator=Q_(
        _ReferenceStats.ServingProfiles.H100VendorMemoryBudget.to("GiB").magnitude, "GB"
    ),  # nameplate label read as decimal GB
    context_tokens=int(_ReferenceStats.ServingProfiles.PrecisionDividendContextLengthTokens),
    kv_heads=_Models.Language.Llama2_70B.kv_heads,
    weight_bytes=_BYTES_FP16,
    weight_bytes_low=_BYTES_INT4,
    kv_bytes=_BYTES_FP16,
    kv_bytes_low=_BYTES_INT8,
)


Llama2_70BPrecisionDividendMhaTp8 = Llama2_70BPrecisionDividendGqaTp8.variant(
    name="Precision dividend of the same 70B model with 64-head attention",
    kv_heads=_Models.Language.Llama2_70B.heads,
)


LongPromptChunkedPrefill = _st.ChunkedPrefillScenario(
    name="A 10,000-token prompt in 512-token chunks",
    description="Each iteration runs the active streams' decode tokens plus one chunk.",
    prompt_tokens=10_000,  # scenario assumption
    chunk_tokens=512,  # scenario assumption
)


Llama2_70BPrefillTensorParallel8 = _st.TensorParallelSpeedupScenario(
    name="Long-prompt prefill layer of a 70B model, 8-way tensor parallel",
    description="Stated 30 ms of serial compute per layer (12 attention, 18 feed-forward) and "
    "0.3 ms per AllReduce, two per layer.",
    serial_layer_times={"attention": Q_(12, "ms"), "feed-forward": Q_(18, "ms")},  # scenario assumption
    tp_size=_Nodes.DGX_H100.accelerators_per_node,
    allreduce_time=Q_(0.3, "ms"),  # scenario assumption
    layers=_Models.Language.Llama2_70B.layers,
)


HeterogeneousH100A100Routing = _st.WeightedRoutingScenario(
    name="15,000 QPS over 10 H100 and 20 A100 servers",
    description="Weights proportional to each server's capacity equalize utilization.",
    servers={
        "H100": int(_ReferenceStats.ServingProfiles.HeterogeneousRoutingH100Servers),
        "A100": int(_ReferenceStats.ServingProfiles.HeterogeneousRoutingA100Servers),
    },
    capacity={
        "H100": int(_ReferenceStats.ServingProfiles.HeterogeneousRoutingH100CapacityQps),
        "A100": int(_ReferenceStats.ServingProfiles.HeterogeneousRoutingA100CapacityQps),
    },
    offered=int(_ReferenceStats.ServingProfiles.HeterogeneousRoutingTargetQps),
)


ServingFanout100Leaves = _st.AvailabilityScenario(
    name="A request that waits for 100 independent leaves",
    description="Each leaf meets its own p99 (fast 99 percent of the time); 1 - all_up is the share of slow requests.",
    availability=0.99,  # scenario assumption: each leaf fast at its p99
    n_components=100,  # scenario assumption: leaves per request
)


RecommenderTailEvents = _st.TailEventRateScenario(
    name="A one-in-10,000 slow request at recommender platform scale",
    description="Archetype B request rate times a rare per-request event.",
    request_rate=_ReferenceStats.RecommenderAtScaleProfile.PlatformQps,
    event_probability=1e-4,  # scenario assumption: rare slow-request event
)


Llama70BReplicaColdStart = _st.ColdStartScenario(
    name="Cold start of a 70B FP16 serving replica",
    description="Provision, fetch the FP16 weights from object storage, load and warm up; a "
    "forecast that starts replicas early lowers the peak fleet.",
    workload=_Models.Language.Llama2_70B,
    weight_bytes_per_param=_BYTES_FP16,
    fetch_bandwidth=_ReferenceStats.ReplicaColdStartProfile.ArtifactFetchBandwidth,
    provisioning_time=_ReferenceStats.ReplicaColdStartProfile.ProvisioningTime,
    load_warmup_time=_ReferenceStats.ReplicaColdStartProfile.LoadWarmupTime,
    reactive_peak_replicas=45,  # scenario assumption: reactive-only peak fleet
    predictive_peak_replicas=30,  # scenario assumption: forecast peak fleet
    prewarm_lead=Q_(30, "minute"),  # scenario assumption: pre-warm lead before each ramp
)


SharedPoolNoisyNeighbor = _st.NoisyNeighborScenario(
    name="Ten tenants on shared H100s, one bursting fivefold",
    description="Three tenants share a 60 GB KV pool; one swells to 45 GB. Latencies are "
    "scenario values, not measurements.",
    kv_pool=Q_(60, "GB"),  # scenario assumption
    pool_tenants=3,  # scenario assumption
    swollen_share=Q_(45, "GB"),  # scenario assumption
    base_sequences=200,  # scenario assumption
    base_rate=100,  # scenario assumption: QPS per tenant
    burst_rate=500,  # scenario assumption
    slo=Q_(50, "ms"),  # scenario assumption: p99 SLO
    unprotected_p99_low=Q_(22, "ms"),  # scenario assumption
    unprotected_p99_high=Q_(45, "ms"),  # scenario assumption
    quota_rate=120,  # scenario assumption: quota-enforced QPS
    protected_p99=Q_(15, "ms"),  # scenario assumption
)


ReasoningTokenLatency = _st.ReasoningLatencyScenario(
    name="128 serial reasoning tokens at 100 ms per token",
    description="A round figure per token; the ratio does not depend on it.",
    per_token_time=Q_(100, "ms"),  # scenario assumption
    reasoning_tokens=128,  # scenario assumption
)


FourRequestDecodeBatch = _st.FixedBatchScheduleScenario(
    name="Four concurrent requests with answers of 50 to 200 tokens",
    description="One token per 20 ms decode iteration; fixed membership keeps every slot until "
    "the longest answer ends.",
    output_lengths=[50, 200, 100, 150],  # scenario assumption
    prompt_lengths=[100, 80, 120, 90],  # scenario assumption
    iteration_time=Q_(20, "ms"),  # scenario assumption
)
