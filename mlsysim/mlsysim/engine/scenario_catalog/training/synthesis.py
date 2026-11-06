"""Named training scenarios for synthesis.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ..training.layers import ImageNet64ChannelConv


# --- Volume I synthesis: recalled calculations (added 2026-10-20) -------
# The conclusion of a course recalls earlier calculations; each entry
# below varies an existing scenario or reuses an existing type.
Conv3x3At256Channels = ImageNet64ChannelConv.variant(
    name="A 3x3 convolution from 256 to 256 channels, FP32",
    description="A representative later layer; the standard-over-separable ratio depends only on the "
    "kernel and the output channels.",
    in_channels=256,  # scenario assumption: representative channel count
    out_channels=256,  # scenario assumption
)
