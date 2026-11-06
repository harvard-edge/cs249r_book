"""Named agents scenarios for observability.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ... import scenario_types as _st


# --- Agent operations: traces, retention, postmortems, staged release (added 2026-10-20) ---
# Types in scenario_types.py, solvers in engine/solvers/operations.py. Every
# input is a stated teaching assumption unless a source is named.
RepairTrialSpanTimeline = _st.SpanCriticalPathScenario(
    name="A repair-agent timing trial on one host clock, with the tests cut to 1 s",
    description="Retrieve supporting context, a model call proposing two checks, targeted tests and a "
    "static lint check in parallel, and a model call synthesizing the report after both; "
    "a 0.5 s gap before synthesis that no span covers. The intervention shortens the "
    "tests and keeps every other duration and wait.",
    intervals={
        "retrieve": (Q_(0.0, "s"), Q_(0.2, "s")),  # scenario assumption: observed interval
        "propose": (Q_(0.2, "s"), Q_(3.2, "s")),  # scenario assumption
        "tests": (Q_(3.2, "s"), Q_(9.2, "s")),  # scenario assumption
        "lint": (Q_(3.2, "s"), Q_(5.2, "s")),  # scenario assumption
        "synthesize": (Q_(9.7, "s"), Q_(11.7, "s")),
    },  # scenario assumption
    depends_on={"propose": ["retrieve"], "tests": ["propose"], "lint": ["propose"], "synthesize": ["tests", "lint"]},
    intervention={"tests": Q_(1.0, "s")},  # scenario assumption: a cache shortens the tests
)


RareFailureUnderHeadSampling = _st.HeadSamplingMissScenario(
    name="A failure mode seen five times under a 10 percent head sample",
    description="Each trial kept independently before its outcome is known.",
    retention_probability=0.10,  # scenario assumption: head-sampling probability
    occurrences=5,  # scenario assumption: occurrences of the failure mode
)


RecurringFailureUnderHeadSampling = RareFailureUnderHeadSampling.variant(
    name="The same failure mode seen fifty times under the 10 percent head sample",
    occurrences=50,  # scenario assumption
)


FailureUnderFivePercentHeadSample = _st.HeadSamplingMissScenario(
    name="A failure mode seen twenty times under a 5 percent head sample",
    description="Each trial kept independently before its outcome is known.",
    retention_probability=0.05,  # scenario assumption
    occurrences=20,  # scenario assumption
)


VerdictAwareTraceRetention = _st.TraceRetentionScenario(
    name="10,000 trials an hour with 1.4 MB trace bodies, kept for 24 hours",
    description="400 failed, 100 pending or unknown, and 9,500 ordinary accepted trials each hour; "
    "head sampling at 5 percent against verdict-aware selection with a 2 percent control "
    "sample; decimal megabytes, no capture loss.",
    failed=400,  # scenario assumption: failed a criterion per hour
    unresolved=100,  # scenario assumption: pending or unknown per hour
    ordinary=9_500,  # scenario assumption: ordinary accepted per hour
    cohort_period=Q_(1, "hour"),
    trace_size=Q_(1.4, "MB"),  # scenario assumption: complete permitted body, decimal MB
    window=Q_(24, "hour"),  # scenario assumption: retention window
    head_probability=0.05,  # scenario assumption
    control_probability=0.02,  # scenario assumption
)


InterfaceRepairBranches = _st.ControlledBranchScenario(
    name="Twenty continuations per branch from one saved prefix of a truncated-search repair",
    description="Original interface, a warning that results may be incomplete, and a paginated "
    "interface with typed partial-result metadata; same model, sampling, budget, checks, "
    "and admission gate. Hypothetical outcomes.",
    outcomes={
        "original": {"accepted": 1, "forbidden": 19},  # scenario assumption
        "warning": {"accepted": 8, "budget_stop": 12},  # scenario assumption
        "paginated": {"accepted": 20},
    },  # scenario assumption
    target_bound=0.05,  # scenario assumption: failure-rate bound to demonstrate
)


TestLintToolDrift = _st.ToolDriftScenario(
    name="A candidate that splits its checks evenly against a baseline at 70 percent tests",
    description="Two-tool call distributions from a shadow run; hypothetical shares.",
    baseline={"tests": 0.70, "lint": 0.30},  # scenario assumption
    candidate={"tests": 0.50, "lint": 0.50},  # scenario assumption
)


AcceptanceCanaryMonitor = _st.SequentialCanaryScenario(
    name="An SPRT canary: acceptance 0.90 against 0.75 on 5 percent of 60 tasks an hour",
    description="False-hold target 0.01 and missed-degradation target 0.05; stationary independent "
    "labels, regular admissions, each label 0.5 h after admission, no tripwire.",
    p_base=0.90,  # scenario assumption: baseline acceptance
    p_degraded=0.75,  # scenario assumption: specified degradation
    alpha=0.01,  # scenario assumption
    beta=0.05,  # scenario assumption
    arrival_rate=Q_(60, "1/hour"),  # scenario assumption: eligible tasks
    fraction=0.05,  # scenario assumption: canary allocation
    label_delay=Q_(0.5, "hour"),  # scenario assumption
)


AcceptanceCanaryMonitorDoubled = AcceptanceCanaryMonitor.variant(
    name="The same SPRT canary at twice the allocation",
    fraction=AcceptanceCanaryMonitor.fraction * 2,  # scenario assumption: doubled allocation
)
