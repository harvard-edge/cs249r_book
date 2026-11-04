"""Named serving scenarios for scale.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....core.units import DAYS_PER_YEAR as _DAYS_PER_YEAR
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ... import scenario_types as _st


# --- Serving at scale (added 2026-10-20) --------------------------------
# What counts as a served request, how many one replica admits, and what a
# fleet of replicas costs. The reference replica is the Llama-3 70B GQA
# model on one DGX H100 node; the measured traffic is the Azure LLM
# inference traces; the measured frontier is InferenceX.
LlmServingFirstYear = _st.ServingCostScenario(
    name="A 70B model trained once, then served for a year",
    description="One million daily users send 50 requests each at an assumed tenth of a cent "
    "per request; training cost $2M once.",
    training_cost=2_000_000,  # scenario assumption: one-time training spend, USD
    daily_users=1_000_000,  # scenario assumption
    requests_per_user_per_day=50,  # scenario assumption
    cost_per_request=0.001,  # scenario assumption: serving cost per request, USD
    efficiency_gain=0.10,  # scenario assumption: cut in cost per request
    days_per_year=_DAYS_PER_YEAR,
)


InteractiveRequestServingTax = _st.ServingTaxScenario(
    name="A 100 ms interactive request with 70 ms of model execution",
    description="Everything outside the model (routing, network, serialization, coordination, "
    "queueing) shares what is left; four sequential exchanges of 50 to 500 us.",
    latency_target=Q_(100, "ms"),  # scenario assumption
    model_time=Q_(70, "ms"),  # scenario assumption: idle-replica model execution
    exchanges=4,  # scenario assumption: sequential network exchanges
    exchange_low=Q_(50, "microsecond"),  # scenario assumption
    exchange_high=Q_(500, "microsecond"),  # scenario assumption
)


VisionBatchingKnee = _st.BatchingKneeScenario(
    name="Vision batch: 2 ms fixed, 1 ms per image",
    description="A compute-bound vision model whose fixed cost per batch is small.",
    fixed_time=Q_(2, "ms"),  # scenario assumption
    per_request_time=Q_(1, "ms"),  # scenario assumption
)


LlmDecodeBatchingKnee = VisionBatchingKnee.variant(
    name="LLM decode batch: 40 ms fixed, 0.5 ms per sequence",
    description="About 40 ms streams a 70B model's FP16 weights through one H100's HBM "
    "(Llama3_70B_DecodeH100); 0.5 ms reads one sequence's own KV state.",
    fixed_time=Q_(40, "ms"),  # scenario assumption: rounded weight-stream time
    per_request_time=Q_(0.5, "ms"),  # scenario assumption: per-sequence KV read
)


VisionServingLittlesLaw = _st.LittlesLawScenario(
    name="Vision service at 1,000 requests per second, 100 ms mean latency",
    description="Replicas run batches of 8 in 80 ms; the fleet targets 80 percent mean service utilization.",
    arrival_rate=Q_(1000, "1/second"),  # scenario assumption
    time_in_system=Q_(100, "ms"),  # scenario assumption: mean end-to-end latency
    batch_size=8,  # scenario assumption
    batch_service_time=Q_(80, "ms"),  # scenario assumption
    utilization_target=0.8,  # scenario assumption
)


Llama3_70B_ReplicaDecodeBudget = _st.DecodeBudgetScenario(
    name="Constructed batch-one decode budget of a 70B replica on 8 H100s",
    description="An 85 ms baseline step split over GEMM, attention, element-wise, AllReduce, and "
    "launch; INT4 weights cut GEMM, then fusion and graph replay cut the rest.",
    workload=_Models.Language.Llama2_70B,
    tp_size=int(_ReferenceStats.ServingProfiles.PrecisionDividendTensorParallelDegree),
    accelerator=_Hardware.Cloud.H100,
    weight_bytes_per_param=_BYTES_FP16,
    tpot_target=Q_(50, "ms"),  # scenario assumption: illustrative endpoint TPOT target
    baseline_step=Q_(85, "ms"),  # scenario assumption: constructed baseline step
    intervals=["GEMM", "attention", "element-wise", "AllReduce", "launch"],
    baseline_shares=[0.40, 0.24, 0.12, 0.12, 0.12],  # scenario assumption
    gemm_cut=0.45,  # scenario assumption: INT4 weights
    attention_cut=0.25,  # scenario assumption: fused decode-attention kernel
    fused_elementwise=Q_(3.4, "ms"),  # scenario assumption
    replay_overhead=Q_(2.0, "ms"),  # scenario assumption: graph replay
    collective_launch=Q_(5, "microsecond"),  # scenario assumption: collective launch latency
)


Llama3_70B_ReplicaSpeculation = _st.SpeculativeRoundScenario(
    name="Speculative decoding on the tuned 70B replica step",
    description="A draft proposes 5 tokens in 4 ms; the target verifies them in one tuned step; "
    "each candidate is accepted with probability 0.78.",
    acceptance=0.78,  # scenario assumption: draft/target pair
    draft_length=5,  # scenario assumption
    draft_time=Q_(4, "ms"),  # scenario assumption
    verify_time=Q_(Llama3_70B_ReplicaDecodeBudget.totals[2], "ms"),
)
