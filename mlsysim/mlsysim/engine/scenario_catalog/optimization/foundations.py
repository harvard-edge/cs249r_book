"""Named optimization scenarios for foundations.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....hardware.registry import Hardware as _Hardware
from ... import scenario_types as _st


V100ToB200 = _st.AcceleratorComparison(
    name="V100 to B200",
    description="Dense FP16 peak, TDP, and peak per watt.",
    baseline=_Hardware.Cloud.V100,
    candidate=_Hardware.Cloud.B200,
)


V100ToH100 = _st.AcceleratorComparison(
    name="V100 to H100",
    description="Dense FP16 peak, TDP, and peak per watt; a 10,000 V100 GPU-hour job restated in H100 GPU-hours.",
    baseline=_Hardware.Cloud.V100,
    candidate=_Hardware.Cloud.H100,
    baseline_device_hours=10_000,
)  # scenario assumption: job size measured on V100s
