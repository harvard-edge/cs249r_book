"""Named agents scenarios for tools.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ... import scenario_types as _st
from ..agents.context import RepairContextBudget, RepairPrefixLayoutBill
from ..agents.deliberation import ProbeVsGeneratedExplanation


# --- Tool calling (added 2026-10-20) -------------------------------------
# One mediated call, its retry, its observation, and its catalog. Loss
# rates, sizes, counts, and prices are teaching assumptions; the window is
# the repair harness's, and the input price and cached share are the ones
# the context-layout bill uses, so the volume prices tokens one way.
BlindRetryTwoPercentLoss = _st.BlindRetryScenario(
    name="One blind retry per timeout at 2 percent request and reply loss, 200,000 calls a day",
    description="A non-idempotent call with no deduplication; lost messages never arrive later; "
    "a fleet of 50 workers making 4,000 mutating calls a day each.",
    request_loss=0.02,  # scenario assumption: hypothetical, not a measured rate
    reply_loss=0.02,  # scenario assumption: hypothetical, not a measured rate
    workers=50,  # scenario assumption: fleet size
    calls_per_worker=4000,  # scenario assumption: mutating calls per worker per day
)


TwentyFourMegabyteLogObservation = _st.ObservationCarryScenario(
    name="A 24 MB test log against a 4,000-token shaped view carried by 30 later calls",
    description="An assumed 3 bytes per token; the repair harness's window; uncached input at a "
    "hypothetical $3 per million tokens, cached input at 10 percent of that.",
    raw_bytes=Q_(24, "MB"),  # scenario assumption: raw log size
    bytes_per_token=Q_(3, "byte / token"),  # scenario assumption: sizing ratio, not a tokenizer law
    window=RepairContextBudget.window,
    shaped_tokens=4000,  # scenario assumption: staged observation
    carry_calls=30,  # scenario assumption: later calls that carry it
    input_price=RepairPrefixLayoutBill.input_price,
    cached_fraction=RepairPrefixLayoutBill.cached_fraction,
)


ModelPollingEveryFiveSeconds = _st.ModelPollingScenario(
    name="A model polling a 300 s test job every 5 s",
    description="32,768 input and 40 output tokens per status call, 95 percent of input cached; "
    "hypothetical $3 uncached, cached at 10 percent, and $15 output per million tokens.",
    job_duration=Q_(300, "second"),  # scenario assumption: fixed completion time
    poll_interval=Q_(5, "second"),  # scenario assumption: checks start one interval after dispatch
    context_tokens=32_768,  # scenario assumption: selected input per status call
    output_tokens=40,  # scenario assumption: generated status request
    cache_hit=0.95,  # scenario assumption: input share billed as cached
    input_price=RepairPrefixLayoutBill.input_price,
    cached_fraction=RepairPrefixLayoutBill.cached_fraction,
    output_price=ProbeVsGeneratedExplanation.output_price,
)


FiftyToolCatalogSixSelected = _st.ToolCatalogSelectionScenario(
    name="50 registered tools of 300 tokens, 6 staged with 400 tokens of selection overhead",
    description="Stipulated counts in final serialization; selection needs no extra model call.",
    registered=50,  # scenario assumption
    selected=6,  # scenario assumption
    descriptor_tokens=300,  # scenario assumption: per descriptor
    selection_overhead=400,  # scenario assumption: selection interface and metadata
)


ShellAgainstTwelveStructuredTools = _st.ToolkitGranularityScenario(
    name="One 500-token shell tool against 12 structured tools of 350 tokens",
    description="30 calls per trajectory at the volume's uncached input price; structured attempt "
    "$1.20 accepted at 0.80 with a 0.001 chance of an unconfined change; shell accepted "
    "at 0.78 with 0.004; each unconfined change costs $2,000 to detect and repair.",
    tools=12,  # scenario assumption
    tool_tokens=350,  # scenario assumption: per structured descriptor
    shell_tokens=500,  # scenario assumption
    calls=30,  # scenario assumption: model calls per trajectory
    input_price=RepairPrefixLayoutBill.input_price,
    structured_attempt=Q_(1.20, "dollar"),  # scenario assumption
    structured_acceptance=0.80,  # scenario assumption
    structured_change=0.001,  # scenario assumption
    shell_acceptance=0.78,  # scenario assumption
    shell_change=0.004,  # scenario assumption
    change_cost=Q_(2000, "dollar"),  # scenario assumption: detect and repair one change
)
