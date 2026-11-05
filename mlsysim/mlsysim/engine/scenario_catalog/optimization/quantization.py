"""Named optimization scenarios for quantization.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_FP32 as _BYTES_FP32, BYTES_INT8 as _BYTES_INT8
from ... import scenario_types as _st
from ...serving_profile import resnet50_h100_profile as _resnet50_h100_profile
from ..optimization.compression import Gpt2XlFp16WeightReadPhone, Gpt2XlInt4WeightReadPhone


# --- Quantization (added 2026-10-20) --------------------------------------
Uint8RangeMinusOneToThree = _st.QuantizationGridScenario(
    name="A 2 by 2 activation matrix with range [-1, 3] stored in UINT8",
    description="Affine scale and zero-point by hand; real zero must land on an integer.",
    values=[-1.0, 1.0, 0.0, 3.0],  # scenario assumption: the matrix, row-major; no entry sits on a rounding tie
    bits=8,
    alpha=-1.0,  # scenario assumption: clipping range
    beta=3.0,
)


Gpt2XlFp16DecodeFullContextPhone = Gpt2XlFp16WeightReadPhone.variant(
    name="GPT-2 XL decoding at a full context, FP16 weights and KV cache, on a flagship phone",
    description="Each batch-one token reads every weight and the whole KV cache.",
    context_tokens=_Models.Language.GPT2.context_length,
    kv_bytes_per_value=_BYTES_FP16,
)


Gpt2XlInt4DecodeFullContextPhone = Gpt2XlInt4WeightReadPhone.variant(
    name="GPT-2 XL decoding at a full context, INT4 weights and an FP16 KV cache, on a flagship phone",
    description="Unpacked to FP16 inside the kernel, or in a separate pass that writes and rereads them.",
    context_tokens=_Models.Language.GPT2.context_length,
    kv_bytes_per_value=_BYTES_FP16,
    unpacked_bytes_per_weight=_BYTES_FP16,
)


Int8VsFp32OperationEnergy45nm = _st.OperationEnergyComparisonScenario(
    name="INT8 against FP32 energy per add, multiply, and multiply-accumulate at 45 nm",
    description="The INT8 MAC accumulates in INT32; a 32-bit DRAM read against the INT8 MAC.",
    wide_add=_Hardware.Tech.Op.AddFp32.energy,
    wide_mul=_Hardware.Tech.Op.FlopFp32.energy,
    narrow_add=_Hardware.Tech.Op.AddInt8.energy,
    narrow_mul=_Hardware.Tech.Op.OpInt8.energy,
    accumulate_add=_Hardware.Tech.Op.AddInt32.energy,
    memory_read=_Hardware.Tech.Memory.DRAM.energy_per_access,
)


ResNet50Int8ProfileH100 = _st.QuantizedDeploymentProfileScenario(
    name="ResNet-50 in FP32 and INT8 on an H100, against accuracy and calibration limits",
    description="Accuracy and confidence calibration are stipulated; latency and throughput come from the "
    "shared H100 engine profile; energy per image assumes full board power in both.",
    workload=_Models.Vision.ResNet50,
    hardware=_Hardware.Cloud.H100,
    profile=_resnet50_h100_profile(),
    float_bytes=_BYTES_FP32,
    int_bytes=_BYTES_INT8,
    float_top1_pct=76.1,  # scenario assumption: FP32 top-1
    int_top1_pct=75.8,  # scenario assumption: INT8 top-1
    float_calibration_error_pct=2.1,  # scenario assumption: FP32 confidence-calibration error
    int_calibration_error_pct=3.4,  # scenario assumption: INT8 confidence-calibration error
    top1_limit_pct=75.5,  # scenario assumption: acceptance limit
    calibration_error_limit_pct=3.0,  # scenario assumption: acceptance limit
)
