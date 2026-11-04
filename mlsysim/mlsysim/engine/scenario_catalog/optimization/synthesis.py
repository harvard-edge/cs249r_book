"""Named optimization scenarios for synthesis.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ... import scenario_types as _st


HostPreprocessingRequestPath = _st.AmdahlScenario(
    name="A request that is 90 percent host preprocessing, with inference made 10x faster",
    description="Shares of the original request time; only the accelerator inference speeds up.",
    stages={
        "preprocess": Q_(90, "percent"),  # scenario assumption: CPU-bound preprocessing share
        "inference": Q_(10, "percent"),
    },  # scenario assumption
    stage="inference",
    speedup=10.0,  # scenario assumption: rewritten inference kernel
)
