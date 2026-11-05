"""Named statistics scenarios for fairness.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ... import scenario_types as _st


# --- Fairness gates on a credit service and a ranking service (added 2026-10-20) -----
# A constructed credit service whose two confusion matrices make every
# fairness criterion computable, the price and stability of closing its
# TPR gap, the data a per-group gate needs, published COMPAS rates against
# Chouldechova's identity, and a ranking service's exploration slice.
CreditServiceTwoGroups = _st.GroupConfusionScenario(
    name="A credit service scored on 10,000 Group A and 2,000 Group B applicants",
    description="Constructed confusion counts; half of each group would repay. Group A: TP 4,500, "
    "FN 500, FP 1,000, TN 4,000; Group B: TP 600, FN 400, FP 200, TN 800.",
    groups={  # scenario assumption: per-group confusion counts
        "A": {"tp": 4500, "fp": 1000, "fn": 500, "tn": 4000},
        "B": {"tp": 600, "fp": 200, "fn": 400, "tn": 800},
    },
)


CreditServiceEqualOpportunityPrice = _st.EqualOpportunityPriceScenario(
    name="Raising Group B's TPR to Group A's by moving its threshold",
    description="A repaid loan earns $1,500 and a default loses $3,000; the move raises Group B's FPR by 20 points.",
    confusion=CreditServiceTwoGroups,
    group="B",
    reference_group="A",
    repay_margin=Q_(1_500, "dollar"),  # scenario assumption: lifetime margin on a repaid loan
    default_loss=Q_(3_000, "dollar"),  # scenario assumption: loss on a defaulted loan
    fpr_increase=0.20,  # scenario assumption: FPR rise that closes the TPR gap
)


CreditServiceBiasFeedback = _st.BiasFeedbackScenario(
    name="The credit service's mitigated TPR gap under quarterly retraining",
    description="A 3.0-point gap at launch and 3.3 points at the next audit against a 5-point gate; "
    "sampling marginal denials refits the factor to 0.90.",
    launch_gap=3.0,  # scenario assumption: residual gap after mitigation, points
    next_audit_gap=3.3,  # scenario assumption: gap at the next quarterly audit, points
    gate=5.0,  # scenario assumption: release gate on the TPR gap, points
    sampled_factor=0.90,  # scenario assumption: refit factor with outcome sampling
    unmitigated_gap=100 * CreditServiceTwoGroups.tpr("A") - 100 * CreditServiceTwoGroups.tpr("B"),
    horizon_cycles=4,  # one year of quarterly retraining
)


OnePercentGroupEvaluationBudget = _st.SubgroupEvaluationBudgetScenario(
    name="Test data for a group that is 1 percent of users, error rate within one point at 95 percent",
    description="Worst-case rate 0.5; the rounded target of 10,000 examples from the group; a gap "
    "between two equal groups pinned to 3.5 points.",
    rate=0.5,  # worst case for the binomial standard error
    half_width=0.01,  # scenario assumption: one percentage point
    confidence=0.95,  # scenario assumption
    group_share=0.01,  # scenario assumption: the group's share of users
    target_per_group=10_000,  # scenario assumption: per-group target, rounded
    gap_half_width=0.035,  # scenario assumption: an illustrative slice's interval
)


FairnessAccuracyFrontierSketch = _st.AccuracyFrontierScenario(
    name="A hypothetical fairness-accuracy frontier with three candidate models",
    description="accuracy = 0.85 + 0.10 (1 - exp(-20 x)); A unconstrained at disparity 0.18, B at "
    "0.05, C at strict equality.",
    floor=0.85,  # scenario assumption: accuracy at zero disparity
    span=0.10,  # scenario assumption
    rate=20.0,  # scenario assumption
    candidates={"A": 0.18, "B": 0.05, "C": 0.0},  # scenario assumption
)


RecommenderExplorationExposure = _st.ExplorationExposureScenario(
    name="A ranking service: 10 million daily users, 20 items each, a 1 million item catalog",
    description="1 percent of slots served by randomized exploration.",
    daily_users=10_000_000,  # scenario assumption
    slots_per_user=20,  # scenario assumption
    catalog_items=1_000_000,  # scenario assumption
    explore_share=0.01,  # scenario assumption
)
