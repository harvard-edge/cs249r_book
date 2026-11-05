"""Named statistics scenarios for evaluation.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ... import scenario_types as _st


# --- Agent evaluation (added 2026-10-20) ----------------------------------
# What graded trials support about the repair agent's candidate_v2 against
# baseline_v1, and the leaderboards and judges such claims rest on. Counts,
# rates, and prices are teaching assumptions unless a Literature series
# supplies them; the solvers live in engine/solvers/acceptance.py.
RecurringTaskConsistency = _st.HorizonCompoundingScenario(
    name="A recurring task accepted on 90 percent of single runs, over eight recurrences",
    description="pass^k is horizon compounding over recurrences: all eight runs pass with 0.9^8, "
    "and holding 90 percent over eight needs a single-run rate of 0.9^(1/8).",
    step_success=0.90,  # scenario assumption: hypothetical single-run acceptance
    short_horizon=1,
    long_horizon=8,  # scenario assumption: recurrences the claim covers
    target_success=0.90,  # scenario assumption: consistency the recurring work must hold
)


RecurringTaskConsistencyLowerRate = RecurringTaskConsistency.variant(
    name="The same recurring task at 80 percent per run",
    description="Eight recurrences at an 80 percent single-run rate.",
    step_success=0.80,  # scenario assumption
)


RecurringTaskConsistencyHalvedFailure = RecurringTaskConsistency.variant(
    name="The same recurring task with its single-run failure rate halved",
    description="Eight recurrences at a 95 percent single-run rate.",
    step_success=0.95,  # scenario assumption: failure rate halved from 10 to 5 percent
)


RefundAnswerFiveSessions = RecurringTaskConsistency.variant(
    name="One refund-policy answer asked in five sessions at 95 percent per run",
    description="Consistency over five sessions and the single-run rate that holds 90 percent.",
    step_success=0.95,  # scenario assumption: checkpoint single-run rate
    long_horizon=5,  # scenario assumption: sessions
)


IndependentJudgesNinetyPercent = _st.AnnotatorAgreementScenario(
    name="Two judges that each pass 90 percent of reports independently of the reports",
    description="Observed agreement equals chance agreement, so kappa is zero.",
    class_shares=[0.90, 0.10],  # scenario assumption: shared marginal pass rate
    independent_raters=True,
)


IndependentJudgesEightyPercent = IndependentJudgesNinetyPercent.variant(
    name="Two judges that each pass 80 percent of reports independently of the reports",
    description="The same independence at an 80 percent marginal pass rate.",
    class_shares=[0.80, 0.20],  # scenario assumption
)


FiftyTaskIntervals = _st.BinomialIntervalsScenario(
    name="Wald and Wilson intervals at 50 independent tasks",
    description="0, 2, and 25 accepted (or violating) tasks of 50 at 95 percent confidence.",
    trials=50,  # scenario assumption: the size of many agent safety suites
    successes=[0, 2, 25],
)


AcceptanceRatePlanning = _st.ProportionPlanningScenario(
    name="Tasks one acceptance rate and a two-arm comparison need",
    description="Half-width at p = 0.5 and 100 tasks; tasks for a 2-point half-width; tasks per "
    "independent arm to detect 4 and 2 points at 5 percent significance and 80 percent power.",
    rate=0.5,  # worst case for a proportion's variance
    planning_tasks=100,  # scenario assumption
    target_half_width=0.02,  # scenario assumption
    differences=[0.04, 0.02],  # scenario assumption: detectable differences
    significance=0.05,
    power=0.80,
    z_alpha=1.96,  # rounded two-sided 5 percent quantile, as planning tables print it
    z_beta=0.84,  # rounded 80 percent power quantile
)


RepairCandidatePairedComparison = _st.PairedComparisonScenario(
    name="candidate_v2 against baseline_v1 on 150 paired repair tasks",
    description="One trial of each on every task: both accepted 70, candidate only 18, baseline "
    "only 12, neither 50; a 2-point noninferiority tolerance.",
    both=70,  # scenario assumption: illustrative complete verdicts
    candidate_only=18,  # scenario assumption
    baseline_only=12,  # scenario assumption
    neither=50,  # scenario assumption
    tolerance=0.02,  # scenario assumption: acceptable loss chosen before outcomes
)


TwoTaskTypeConsistency = _st.TaskMixConsistencyScenario(
    name="Two equally likely task types passing on 50 and 90 percent of runs",
    description="One task repeated twice against two independently drawn tasks.",
    task_pass_rates=[0.5, 0.9],  # scenario assumption
    repetitions=2,
)


CappedCandidateSelection = _st.CappedSelectionCostScenario(
    name="Batch against sequential selection of up to three candidates",
    description="A fixed task passing on 40 percent of candidates, $0.03 to generate and $0.01 to "
    "check each, under an exact checker.",
    pass_rate=0.4,  # scenario assumption
    cap=3,  # scenario assumption
    generation_cost=Q_(0.03, "USD"),  # scenario assumption: illustrative price
    check_cost=Q_(0.01, "USD"),  # scenario assumption: illustrative price
)
