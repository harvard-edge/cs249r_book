"""
MLPerf EDU iter-6 SUT: W8A16 dynamic quantization on the LLM target.

PEDAGOGICAL NEGATIVE RESULT.

Per Han's iter-6 proposal: PyTorch's MPS backend does NOT accelerate
INT8 matmul. `torch.quantization.quantize_dynamic` produces a quantized
model whose linear layers fall back to CPU per-op, which makes the
*end-to-end* wall-clock SLOWER than the fp16 baseline despite using
half the memory bandwidth on paper.

This is one of the most important lessons in efficient AI: quantization
without hardware support is worse than no quantization. We don't hide
this; we measure it and put the measurement in the textbook.

Run via scripts/smoke_iter6_serving.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .nanogpt_train import NanoGPTWhiteBox, GPT2_SMALL_CONFIG
from .nanogpt_decode import NanoGPTDecode


def build_int8_quantized_target(config: dict | None = None) -> nn.Module:
    """Construct the GPT-2-Small target and dynamically quantize its linears."""
    cfg = config or GPT2_SMALL_CONFIG
    model = NanoGPTWhiteBox(**cfg).eval()
    # Dynamic quantization: weights stored INT8, computed FP32 at runtime
    # via dequant. On MPS, the quantized linear ops fall back to CPU.
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized


def run_int8_decode(prefill_ctx: int = 2048, decode_steps: int = 32,
                     batch_size: int = 32) -> dict:
    """Drive NanoGPTDecode with the int8-quantized target.

    Returns the decode harness's measurement dict. Sidecar emission goes
    through the existing measure_roofline path inside NanoGPTDecode.run().
    """
    # NB: stays on CPU — quantize_dynamic doesn't ship MPS kernels.
    model = build_int8_quantized_target()
    # Override the workload name so the sidecar lands as "nanogpt-decode-int8".
    decoder = NanoGPTDecode(
        model, prefill_ctx=prefill_ctx, decode_steps=decode_steps,
        batch_size=batch_size,
    )
    # Monkey-patch the roofline name (the existing class hardcodes
    # "nanogpt-decode" inside its run()).
    import os
    os.environ["MLPERF_EDU_WORKLOAD_NAME_OVERRIDE"] = "nanogpt-decode-int8"
    try:
        result = decoder.run()
    finally:
        os.environ.pop("MLPERF_EDU_WORKLOAD_NAME_OVERRIDE", None)
    result["sut"] = "int8-dynamic-cpu-fallback"
    return result
