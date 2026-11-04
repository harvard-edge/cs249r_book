"""Named data scenarios for selection.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Nodes as _Nodes
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ....literature.registry import Literature as _Literature
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st
from ....physics.memory import calc_low_rank_update_params as _calc_low_rank_update_params
from ..data.delivery import ResNet50A100InputFeed
from ..serving.transformers import Gpt2XlBlockFullWindow
from ..training.neural import Gpt2Fp32AdamOnH100


# Gpt2XlRunOnOneH100 and Gpt2XlRunOnOneDgxH100 carry the same names and
# inputs as the training lecture's entries (feat/numbers-v1-08), which add
# optional step and price fields to ClusterTrainingTimeScenario.
Gpt2XlRunOnOneH100 = _st.ClusterTrainingTimeScenario(
    name="GPT-2 XL's training run on one H100",
    description="1.5e21 FLOPs at the H100's dense 16-bit peak and 50 percent utilization, in steps of "
    "32 sequences of 1,024 tokens, at the H100's TDP and a cloud rental price.",
    workload=_Models.Language.GPT2,
    hardware=_Hardware.Cloud.H100,
    n_devices=1,
    efficiency=float(_Literature.Training.MfuHigh),
    tokens_per_step=32 * _Models.Language.GPT2.context_length,  # scenario assumption: 32 sequences per step
    hourly_price=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
)


Gpt2XlBatch32Block = Gpt2XlBlockFullWindow.variant(
    name="GPT-2 XL decoder blocks, one training step of 32 sequences at its 1,024-token window",
    description="The step counted layer by layer, attention products included, against the 6P rule.",
    batch_size=Gpt2XlRunOnOneH100.tokens_per_step // _Models.Language.GPT2.context_length,
)


Gpt2XlBatch32Activations = _st.ActivationMemoryScenario(
    name="GPT-2 XL's saved activations at batch 32 and 1,024 tokens, 16-bit",
    description="Korthikanti et al. per-layer bytes with no recomputation, and the per-layer line items "
    "(projections, feed-forward expansion, attention scores, normalization).",
    workload=_Models.Language.GPT2,
    micro_batch=Gpt2XlBatch32Block.batch_size,
    seq_len=_Models.Language.GPT2.context_length,
    normalization_per_layer=Q_(10, "MB"),  # scenario assumption: normalization buffers per layer
)


Gpt2XlBatch32ActivationsFp32 = Gpt2XlBatch32Activations.variant(
    name="GPT-2 XL's saved activations at batch 32 and 1,024 tokens, FP32",
    description="The same tensors stored in 32 bits.",
    precision_bytes=_BYTES_FP32.to("byte").magnitude,
)


Gpt2Fp32AdamBatch32OnH100 = Gpt2Fp32AdamOnH100.variant(
    name="GPT-2 XL's FP32 Adam training step at batch 32 on one H100",
    description="FP32 weights, gradients, and Adam moments, plus FP32 activations at batch 32.",
    batch_size=Gpt2XlBatch32Block.batch_size,
    activations=Gpt2XlBatch32ActivationsFp32.activations("none"),
)


Gpt2MixedPrecisionAdamOnH100 = Gpt2Fp32AdamOnH100.variant(
    name="GPT-2 XL trained in mixed precision with Adam on one H100, batch 32",
    description="16-bit weights and gradients, an FP32 master copy, two FP32 Adam moments (16 B per "
    "parameter), and 16-bit activations at batch 32 and 1,024 tokens.",
    bytes_per_value=_BYTES_FP16,
    master_bytes_per_parameter=_BYTES_FP32,
    batch_size=Gpt2XlBatch32Block.batch_size,
    activations=Gpt2XlBatch32Activations.activations("none"),
)


Llama2_7BMixedPrecisionAdam = Gpt2MixedPrecisionAdamOnH100.variant(
    name="A 7B model trained in mixed precision with Adam, against one H100",
    description="16 B per parameter before activations, against the 2 B per parameter it serves from.",
    workload=_Models.Language.Llama2_7B,
    batch_size=None,
    activations=None,
)


Llama3_8BBatch4Activations = _st.ActivationMemoryScenario(
    name="Llama 3.1 8B's saved activations at batch 4 and 2,048 tokens, 16-bit",
    description="Per-token tensors only: fused attention keeps no score matrices.",
    workload=_Models.Language.Llama3_8B,
    micro_batch=4,  # scenario assumption: sequences per step
    seq_len=2048,  # scenario assumption: tokens per sequence
)


Llama3_8BFullFineTuneOnH100 = Llama2_7BMixedPrecisionAdam.variant(
    name="Full fine-tuning of Llama 3.1 8B on one H100",
    description="Mixed-precision Adam on every parameter plus activations at batch 4 and 2,048 tokens.",
    workload=_Models.Language.Llama3_8B,
    batch_size=Llama3_8BBatch4Activations.micro_batch,
    activations=Llama3_8BBatch4Activations.per_token_activations,
)


Llama3_8BLoraOnH100 = Llama3_8BFullFineTuneOnH100.variant(
    name="Rank-16 LoRA fine-tuning of Llama 3.1 8B on one H100",
    description="Frozen 16-bit base weights; rank-16 adapters on every query and value projection carry "
    "the mixed-precision Adam state; the same activations.",
    adapter_parameters=_Models.Language.Llama3_8B.layers
    * (
        _calc_low_rank_update_params(
            _Models.Language.Llama3_8B.hidden_dim, _Models.Language.Llama3_8B.hidden_dim, 16
        )  # scenario assumption: rank 16
        + _calc_low_rank_update_params(
            _Models.Language.Llama3_8B.hidden_dim,
            _Models.Language.Llama3_8B.kv_heads
            * (_Models.Language.Llama3_8B.hidden_dim // _Models.Language.Llama3_8B.heads),
            16,
        )
    ),
)


ResNet50H100InputFeed = ResNet50A100InputFeed.variant(
    name="ResNet-50 training on one H100 of a DGX H100, fed by the host's loader workers",
    description="The H100 sustains the same fraction of its arithmetic ceiling as the A100; the host gives "
    "each accelerator its share of the CPU cores as loader workers.",
    hardware=_Hardware.Cloud.H100,
    sustained_rate=(
        ResNet50A100InputFeed.sustained_fraction
        * ResNet50A100InputFeed.variant(hardware=_Hardware.Cloud.H100).ceiling_rate
    ),
    host_workers=_Nodes.DGX_H100.host_cpu_cores // _Nodes.DGX_H100.accelerators_per_node,
)


GilThreadedLoaderStall = _st.DataStallScenario(
    name="Python threads that hold the GIL preparing each batch",
    description="Host preparation of 250 ms and 12 ms of accelerator compute, run in sequence.",
    compute_time=Q_(12, "ms"),  # scenario assumption: accelerator time per batch
    io_time=Q_(250, "ms"),  # scenario assumption: serialized host preparation per batch
)


Gpt2XlRunOnOneDgxH100 = Gpt2XlRunOnOneH100.variant(
    name="GPT-2 XL's training run on the eight H100s of one DGX H100, scaling perfectly",
    description="The one-H100 run divided across one node's accelerators with no communication cost.",
    n_devices=_Nodes.DGX_H100.accelerators_per_node,
)
