"""Named statistics scenarios for synthesis.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ... import scenario_types as _st


ChatEndpointTtftMeanVersusP99 = _st.MeanVersusTailScenario(
    name="An LLM chat endpoint: mean time to first token against its p99",
    description="Illustrative; the p99 is driven by long prompts and queueing behind long generations.",
    mean=Q_(50, "ms"),  # scenario assumption: mean time to first token
    tail_low=Q_(2000, "ms"),  # scenario assumption: p99 time to first token
    tail_high=Q_(2000, "ms"),  # scenario assumption: one stated value
)
