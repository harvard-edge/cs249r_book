"""Named training scenarios for working set.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ....core.units import BYTES_INT64 as _BYTES_INT64
from ... import scenario_types as _st
from ..data.selection import (
    Gpt2MixedPrecisionAdamOnH100,
    Gpt2XlBatch32Activations,
    Gpt2XlBatch32Block,
    Gpt2XlRunOnOneH100,
)


# ── Fitting GPT-2 XL's training step on one H100 (added 2026-10-20) ──────
# The batch-32 step from FP32 through mixed precision, FlashAttention,
# micro-batches, and segment checkpointing; the number formats and loss
# scaling behind mixed precision; and the attention tiles and online
# softmax behind FlashAttention.
Gpt2XlRunOnOneH100Fp32 = Gpt2XlRunOnOneH100.variant(
    name="GPT-2 XL's training run on one H100's FP32 cores",
    description="The same run at the H100's FP32 (non-tensor) peak, utilization held at 50 percent.",
    peak_precision="fp32",
)


Gpt2XlRunOnOneH100Tf32 = Gpt2XlRunOnOneH100.variant(
    name="GPT-2 XL's training run on one H100's TF32 tensor path",
    description="The same run at the H100's dense TF32 peak, utilization held at 50 percent.",
    peak_precision="tf32",
)


Gpt2XlMicroBatch16Activations = Gpt2XlBatch32Activations.variant(
    name="GPT-2 XL's saved activations for a micro-batch of 16 sequences of 1,024 tokens, 16-bit",
    description="Half the batch resident at a time, accumulated twice per update.",
    micro_batch=Gpt2XlBatch32Block.batch_size // 2,  # scenario assumption: two micro-batches per update
)


Gpt2XlBatch32SegmentCheckpointed = Gpt2XlBatch32Activations.variant(
    name="GPT-2 XL's batch-32 activations, checkpointed in 8 segments of 6 layers, scores not stored",
    description="Each segment's 16-bit input is kept; the backward pass rebuilds one segment at a time.",
    checkpoint_segments=8,  # scenario assumption: 48 layers in 8 equal segments
)


Gpt2MixedPrecisionFlashAttentionOnH100 = Gpt2MixedPrecisionAdamOnH100.variant(
    name="GPT-2 XL in mixed precision at batch 32 on one H100, attention scores not stored",
    description="16 B per parameter of state and the per-token activations only (FlashAttention).",
    activations=Gpt2XlBatch32Activations.per_token_activations,
)


Gpt2MixedPrecisionTwoMicroBatchesOnH100 = Gpt2MixedPrecisionFlashAttentionOnH100.variant(
    name="GPT-2 XL's batch of 32 as two accumulated micro-batches of 16 on one H100",
    description="Mixed precision, FlashAttention, and 16 sequences' activations resident at a time.",
    batch_size=Gpt2XlMicroBatch16Activations.micro_batch,
    activations=Gpt2XlMicroBatch16Activations.per_token_activations,
)


Gpt2MixedPrecisionSegmentCheckpointedOnH100 = Gpt2MixedPrecisionFlashAttentionOnH100.variant(
    name="GPT-2 XL's batch of 32 with segment checkpointing on one H100",
    description="Mixed precision, FlashAttention, and 8 kept segment inputs plus one rebuilt segment.",
    activations=Gpt2XlBatch32SegmentCheckpointed.checkpointed_activations,
)


Gpt2XlSingleCoreTokenizerFeed = _st.TokenizedBatchFeedScenario(
    name="GPT-2 XL's batch of 32 tokenized on one host core and copied to an H100, in sequence",
    description="Host tokenization and the PCIe copy of the token ids before every step, no prefetching.",
    compute_time=Gpt2XlRunOnOneH100.step_time,
    tokens_per_batch=Gpt2XlRunOnOneH100.tokens_per_step,
    token_rate=Q_(50_000, "1/s"),  # scenario assumption: tokens per second on one host core
    bytes_per_token=_BYTES_INT64,  # token ids as int64
    link_bandwidth=_Hardware.Cloud.H100.interconnect.bandwidth,
    workers=8,  # scenario assumption: loader workers in the parallel case
)


AttentionScores4096TokensFp32 = _st.AttentionTilingScenario(
    name="One attention head at 4,096 tokens in FP32, in 128 x 128 tiles",
    description="The score matrix, a tile, 16 heads, and what standard and FlashAttention store per head.",
    seq_len=4096,  # scenario assumption: long-context example
    head_dim=_Models.Language.GPT2.hidden_dim // _Models.Language.GPT2.heads,
    tile=128,  # scenario assumption: tile side
    heads=16,  # scenario assumption: heads in the example layer
    bytes_per_value=_BYTES_FP32,
)
