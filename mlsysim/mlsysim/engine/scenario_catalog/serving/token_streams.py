"""Named serving scenarios for token streams.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_INT4 as _BYTES_INT4
from ... import scenario_types as _st
from ..optimization.framework import Gpt2XlFp16DecodeOnH100


# --- Serving a stream of tokens on one machine (added 2026-10-20) ---------
# GPT-2 XL is the bandwidth-bound chat decoder, Llama 3 8B the case-study
# service, and a 70B-class model (Llama 3 70B geometry) the larger model.
Gpt2XlChatDecodeBatchesH100 = _st.KvDecodeBatchScenario(
    name="GPT-2 XL chat decode batches on one H100 with full 1,024-token contexts",
    description="FP16 weights read once per step; every sequence reads its own FP16 cache (no grouped-query "
    "sharing); a 512-token prompt prefilled at 40 percent of the dense FP16 peak.",
    workload=_Models.Language.GPT2,
    hardware=_Hardware.Cloud.H100,
    weight_bytes_per_param=_BYTES_FP16,
    kv_bytes_per_value=_BYTES_FP16,
    context_tokens=_Models.Language.GPT2.context_length,
    batch_sizes=[1, 8, 32, 64, 128],  # scenario assumption: batch sizes compared
    prompt_tokens=512,  # scenario assumption: chat prompt length
    prefill_efficiency=0.4,  # scenario assumption: share of the FP16 peak sustained in prefill
)


Gpt2XlSpeculativeGpt2SmallH100 = _st.SpeculativeRoundScenario(
    name="Speculative decoding of GPT-2 XL with a GPT-2 Small draft on one H100",
    description="Batch one, both models bandwidth bound: each draft step reads GPT-2 Small's FP16 weights, "
    "the verification pass GPT-2 XL's once.",
    acceptance=0.7,  # scenario assumption: per-token acceptance probability
    draft_length=4,  # scenario assumption: tokens proposed per round
    draft_step=(_Models.Language.GPT2_Small.size_in_bytes(_BYTES_FP16) / _Hardware.Cloud.H100.memory.bandwidth).to(
        "ms"
    ),
    verify_time=Gpt2XlFp16DecodeOnH100.result.memory_time,
)


Llama3_8BFp16ConversationsH100 = _st.KvAdmissionScenario(
    name="Llama 3 8B conversations of 8,192 tokens on one H100, 16-bit weights and cache",
    description="The H100's memory after the weights, over each conversation's cache; no runtime reserve.",
    workload=_Models.Language.Llama3_8B,
    accelerators=1,
    hbm_per_accelerator=_Hardware.Cloud.H100.memory.capacity,
    weight_bytes_per_param=_BYTES_FP16,
    reserve=Q_(0, "GB"),  # scenario assumption: before runtime workspace
    context_tokens=8192,  # scenario assumption: one long conversation
    kv_bytes_per_elem=_BYTES_FP16,
)


Llama3_70BInt4KvBudgetH100 = Llama3_8BFp16ConversationsH100.variant(
    name="A 70B-class model with 4-bit weights on one H100: the 16-bit cache budget",
    description="Llama 3 70B geometry (80 layers, 8 key-value heads of 128); 4-bit weights.",
    workload=_Models.Language.Llama3_70B,
    weight_bytes_per_param=_BYTES_INT4,
    requests=32,  # scenario assumption: concurrent conversations for the scale-up
)


Llama3_8BInt4ChatServiceH100 = _st.SingleAcceleratorLlmServiceScenario(
    name="An 8-billion-parameter Llama 3 chat service on one H100 with 4-bit weights",
    description="1,000-token prompts and 256-token answers; AWQ 4-bit weights and a 16-bit cache; a 72 GiB "
    "cache reserve; prefill admits 10,000 prompt tokens per second; $3 per H100-hour.",
    workload=_Models.Language.Llama3_8B,
    hardware=_Hardware.Cloud.H100,
    precision="int4",
    kv_bytes_per_value=_BYTES_FP16,
    prompt_tokens=1000,  # scenario assumption
    decode_tokens=256,  # scenario assumption
    prefill_efficiency=0.4,  # scenario assumption: share of peak in prefill
    kv_reserve=Q_(72, "GiB"),  # scenario assumption: memory set aside for the cache
    prefill_rate=Q_(10_000, "1/second"),  # scenario assumption: isolated prefill rate, prompt tokens
    price=Q_(3.00, "dollar/hour"),  # scenario assumption: H100 SXM5 rental
    ttft_target=Q_(200, "ms"),  # scenario assumption
    tpot_target=Q_(20, "ms"),  # scenario assumption
)


H100ChatEnergyPerToken = _st.TokenEnergyScenario(
    name="Energy per generated token of an H100 chat service, well used and poorly used",
    description="The H100 at its TDP plus a 300 W host share, 114 requests at 8 tokens per second each; "
    "at 10 percent of that rate the GPU still draws 300 W.",
    accelerator_power=_Hardware.Cloud.H100.tdp,
    host_power=Q_(300, "watt"),  # scenario assumption: host server share
    low_use_accelerator_power=Q_(300, "watt"),  # scenario assumption: GPU near idle
    concurrent_requests=114,  # scenario assumption: concurrent requests per H100
    per_request_rate=Q_(8, "1/second"),  # scenario assumption: decode tokens/s per request
    response_tokens=500,  # scenario assumption: typical response length
    low_use_fraction=0.1,  # scenario assumption: poorly batched
)
