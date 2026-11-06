"""Named optimization scenarios for mapping.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....hardware.registry import Hardware as _Hardware
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ... import scenario_types as _st
from ..optimization.acceleration import ResNet50Stage2ConvBatch32Int8


# --- Mapping a computation onto an accelerator (added 2026-10-20) ---
Matmul4096Fp16Tile128 = _st.MatmulTileScenario(
    name="A 4,096-square FP16 matrix multiply in 128 x 128 x 128 tiles, single buffered",
    description="FP16 operands, FP32 accumulator; 1,024 output tiles, each summing 32 tile products.",
    m=4096,
    n=4096,
    k=4096,  # scenario assumption: a transformer-width layer
    tile_m=_Hardware.Cloud.TPUv4.compute.systolic_array_dim,  # tiles the width of a 128-wide array
    tile_n=_Hardware.Cloud.TPUv4.compute.systolic_array_dim,
    tile_k=_Hardware.Cloud.TPUv4.compute.systolic_array_dim,
    operand_bytes=_BYTES_FP16,
    accumulator_bytes=_BYTES_FP32,
)


ResNet50Stage2MapInt8ReadOnce = _st.MemoryRoundTripScenario(
    name="One ResNet-50 stage-2 output map (INT8, one image) read once from DRAM at 160 pJ/byte",
    description="45 nm off-chip DRAM energy per byte (Horowitz).",
    tensor_bytes=ResNet50Stage2ConvBatch32Int8.activation_bytes,
    energy_per_byte=_Hardware.Tech.Memory.DRAM.energy_per_byte,
    trips=1,
)
