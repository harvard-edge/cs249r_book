"""Shared first-order decode accounting for serving and batching.

The memory term assumes one weight read per step and reads the resident KV
cache. The compute term includes linear work and both attention matrix
products. The larger term bounds the step; the layer tax is additive.
This is a dense/active-parameter roofline, not a kernel or scheduler model.
"""

from dataclasses import dataclass

from ...core._validation import validate_positive
from ...core.units import Q_
from ...core.types import Quantity
from ...hardware.types import HardwareNode
from ...models.types import TransformerWorkload
from ...physics.transformer import calc_attention_matmul_flops
from ..calibration import FRAMEWORK_LAYER_TAX_MS


@dataclass(frozen=True)
class DecodeStep:
    compute: Quantity
    memory: Quantity
    overhead: Quantity

    @property
    def latency(self) -> Quantity:
        return max(self.compute, self.memory) + self.overhead

    @property
    def bottleneck(self) -> str:
        return "Compute" if self.compute >= self.memory else "Memory"


def decode_step(
    model: TransformerWorkload,
    hardware: HardwareNode,
    *,
    batch_size: int,
    context_tokens: float,
    precision: str,
    bytes_per_element: Quantity,
    kv_bytes: Quantity,
    efficiency: float,
    query_tokens: int = 1,
) -> DecodeStep:
    """One step, optionally verifying several speculative query tokens at once.

    KV bytes cover all requests; weights are charged once. MoE weight traffic
    conservatively reads all resident weights, while compute uses active work.
    Workspace, activation traffic, and inter-device collectives are excluded.

    Notes
    -----
    Equation: T = max(F / (eta * peak), (W + KV) / BW) + T_layer.
    F adds linear work and both attention products over the supplied context.
    The layer tax is an MLSysIM runtime assumption, not part of Roofline.

    References
    ----------
    Williams et al. (2009), Roofline (bibliography key: williams2009),
    https://doi.org/10.1145/1498765.1498785.
    Pope et al. (2023), Efficiently Scaling Transformer Inference
    (bibliography key: pope2023efficiently), https://arxiv.org/abs/2211.05102.
    See ServingModel.solve for the public contract and executable example.
    """
    peak = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)
    validate_positive(peak, "decode peak FLOP/s")
    validate_positive(hardware.memory.bandwidth, "decode memory bandwidth")
    linear = model.lower(bytes_per_element).total_ops * batch_size * query_tokens
    attention = calc_attention_matmul_flops(
        model.layers, model.heads * model.head_width, query_tokens, context_tokens, batch_size
    )
    compute = ((linear + attention) / (peak * efficiency)).to("ms")
    memory = ((model.size_in_bytes(bytes_per_element) + kv_bytes) / hardware.memory.bandwidth).to("ms")
    return DecodeStep(compute, memory, Q_(model.layers * FRAMEWORK_LAYER_TAX_MS, "ms"))
