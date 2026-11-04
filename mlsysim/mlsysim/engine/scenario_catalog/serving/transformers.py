"""Named serving scenarios for transformers.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....datasets.registry import Datasets as _Datasets
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ... import scenario_types as _st
from ..serving.foundations import Llama2_70B_DecodeH100


# --- Network architectures: attention, transformer blocks, decode, and recommenders (added 2026-10-20) ---
# GPT-2 XL is the running transformer; Llama 2 and Llama 3.1 shapes size
# decode and the KV cache; DLRM-style tables carry the recommender.
Gpt2SmallSixTokenQkvProjection = _st.PackedQkvProjectionScenario(
    name="Six tokens through GPT-2 Small's packed query-key-value projection",
    description="One 768 x 2,304 weight matrix produces all three projections in one kernel.",
    workload=_Models.Language.GPT2_Small,
    tokens=6,  # scenario assumption: a six-token sequence
)


Gpt2XlBlockFullWindow = _st.TransformerBlockScenario(
    name="GPT-2 XL decoder blocks at its 1,024-token window",
    description="The classic 12 d^2 block (feed-forward width 4d); dense against attention multiply-adds.",
    workload=_Models.Language.GPT2,
    context_tokens=_Models.Language.GPT2.context_length,
)


TwelveHeadScoresAt100kTokens = _st.MaterializedScoresScenario(
    name="Materialized attention scores of a 12-head, 32-layer model at 100,000 tokens",
    description="FP16 scores stored in full, against one 80 GB accelerator.",
    seq_len=100_000,  # scenario assumption: long context
    heads=12,  # scenario assumption: generic model
    layers=32,  # scenario assumption: generic model
    bytes_per_score=_BYTES_FP16,
    hardware=_Hardware.Cloud.H100,
)


BertWindowToFourK = _st.ContextGrowthScenario(
    name="BERT's 512 learned positions against a 4,096-token input",
    description="Score memory grows with the square of the length ratio.",
    base_tokens=_Models.Language.BERT_Base.context_length,
    tokens=4096,  # scenario assumption: a longer input
)


BertWindowToOneK = BertWindowToFourK.variant(name="BERT's window doubled", tokens=1024)  # scenario assumption


BertWindowToTwoK = BertWindowToFourK.variant(name="BERT's window quadrupled", tokens=2048)  # scenario assumption


Gpt2XlKvFullWindow = _st.KvFootprintScenario(
    name="GPT-2 XL's FP16 KV cache at its 1,024-token window",
    description="2 x 48 layers x 1,600 x 2 bytes per kept token.",
    workload=_Models.Language.GPT2,
    tokens=_Models.Language.GPT2.context_length,
    bytes_per_value=_BYTES_FP16,
)


Llama2_7BKvAt2048Tokens = Gpt2XlKvFullWindow.variant(
    name="Llama 2 7B's FP16 KV cache for one 2,048-token request",
    description="32 layers, 32 key-value heads of 128, FP16.",
    workload=_Models.Language.Llama2_7B,
    tokens=2048,  # scenario assumption: serving context length
)


Gpt2XlContextMemoryOnH100 = _st.ContextMemoryCrossingScenario(
    name="GPT-2 XL's weights, KV cache, and one layer's scores against one H100",
    description="FP16 throughout; where the linear cache and the quadratic scores cross fixed budgets.",
    workload=_Models.Language.GPT2,
    hardware=_Hardware.Cloud.H100,
    bytes_per_value=_BYTES_FP16,
)


Llama2_70B_DecodeTwoH100 = Llama2_70B_DecodeH100.variant(
    name="Llama 2 70B batch-1 decode split across two H100s",
    description="140 GB of FP16 weights do not fit one 80 GB H100; each device streams its half per token.",
    tp_size=2,  # scenario assumption: the smallest split that fits
)


Gpt2XlDecodeH100 = Llama2_70B_DecodeH100.variant(
    name="GPT-2 XL batch-1 decode on one H100",
    description="FP16 weights streamed once per token.",
    workload=_Models.Language.GPT2,
)


CriteoShapedRequestOnH100 = _st.EmbeddingGatherScenario(
    name="One Criteo-shaped recommender request on an H100",
    description="One 128-wide FP32 row per categorical feature; device HBM against the PCIe host link.",
    dataset=_Datasets.CriteoKaggle,
    width=128,  # scenario assumption: embedding width
    bytes_per_value=_BYTES_FP32,
    hardware=_Hardware.Cloud.H100,
)


HundredMillionItemTableOnA100 = _st.EmbeddingTableScenario(
    name="A 100-million-item embedding table on one 80 GB A100",
    description="128 FP32 values per row, one categorical feature.",
    rows=100_000_000,  # scenario assumption: catalog size
    width=CriteoShapedRequestOnH100.width,
    bytes_per_value=_BYTES_FP32,
    hardware=_Hardware.Cloud.A100,
)


BillionUserTable = HundredMillionItemTableOnA100.variant(
    name="A billion-user embedding table, 128 FP32 values per row",
    rows=1_000_000_000,  # scenario assumption: user count
)
