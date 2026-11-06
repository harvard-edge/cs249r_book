"""Named statistics scenarios for responsibility.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Clusters as _Clusters
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ... import scenario_types as _st
from ....core.units import DAYS_PER_MONTH as _DAYS_PER_MONTH


# --- Responsible AI (added 2026-10-20) -------------------------------------
LoanScreeningTwoGroups = _st.GroupConfusionScenario(
    name="A loan-screening model scored on 100 applicants in each of two groups",
    description="Group A: 70 approved (40 repaid, 30 defaulted), 30 rejected (5 would repay); "
    "Group B: 40 approved (30 repaid, 10 defaulted), 60 rejected (20 would repay). "
    "Rejected applicants' repayment labels are hypothetical evaluation labels.",
    groups={  # scenario assumption: per-group confusion counts
        "A": {"tp": 40, "fp": 30, "fn": 5, "tn": 25},
        "B": {"tp": 30, "fp": 10, "fn": 20, "tn": 40},
    },
)


FraudFlagTwoGroups = _st.GroupConfusionScenario(
    name="A fraud model's flags on 1,000 records in each of two groups",
    description="Confusion counts for an audit of demographic parity and equal opportunity.",
    groups={  # scenario assumption: per-group confusion counts
        "A": {"tp": 450, "fp": 50, "fn": 30, "tn": 470},
        "B": {"tp": 180, "fp": 70, "fn": 120, "tn": 630},
    },
)


LoanApprovalGapTest = _st.ProportionGapTestScenario(
    name="The loan model's approval-rate gap, 0.70 against 0.40, tested at 100 and 10 per group",
    description="Pooled two-proportion z test and the gap's 95 percent interval.",
    rate_a=LoanScreeningTwoGroups.positive_rate("A"),
    rate_b=LoanScreeningTwoGroups.positive_rate("B"),
    n_per_group=round(LoanScreeningTwoGroups.size("A")),
    significance=0.05,  # scenario assumption: two-sided test level
    small_n_per_group=10,  # scenario assumption: a small-sample subgroup
)


CreditParityAccuracyCost = _st.ConstraintAccuracyCostScenario(
    name="Demographic parity imposed on a credit model: accuracy 85 to 81 percent",
    description="Holding Group A at 80 percent approval and raising Group B from 60 to 80 percent.",
    baseline_accuracy=85.0,  # scenario assumption
    constrained_accuracy=81.0,  # scenario assumption
    context={  # scenario assumption: the policy's stated rates, percent
        "group_a_default": 20,
        "group_b_default": 40,
        "default_threshold": 30,
        "group_a_approval": 80,
        "group_b_approval_unconstrained": 60,
        "default_threshold_parity": 50,
    },
)


FairnessMonitor10K = _st.SampledDecisionMonitorScenario(
    name="Fairness monitoring of a decision service at 10,000 decisions per second",
    description="1 percent of decisions logged at 200 bytes; 50 groups x 3 metrics at a 5 percent "
    "false-alarm rate per metric.",
    request_rate=Q_(10_000, "1/s"),  # scenario assumption
    sample_share=0.01,  # scenario assumption
    record_size=Q_(200, "byte"),  # scenario assumption
    groups=50,  # scenario assumption
    metrics_per_group=3,  # scenario assumption: FPR, TPR, calibration error
    false_alarm_rate=0.05,  # scenario assumption
)


InlineShapRetrofit = _st.InlineOverheadScenario(
    name="Sampled SHAP retrofitted inline on a 100 ms recommender",
    description="The approximate-SHAP inference overhead range added to every request.",
    base_latency=_ReferenceStats.RecommenderAtScaleProfile.P99LatencySlo,
    overhead_low=50.0 / 100,  # Responsible-AI technique overheads (accuracy, training, inference, ...
    overhead_high=200.0 / 100,  # Responsible-AI technique overheads (accuracy, training, inference, ...
    request_rate=Q_(10_000, "1/s"),  # scenario assumption
)


ImageExplanationLog = _st.ExplanationLogScenario(
    name="Logging an explanation with each of a million image predictions a day",
    description="1 to 10 MB per attribution map, or 4 to 8 bytes per feature for tabular SHAP.",
    predictions_per_day=1_000_000,  # scenario assumption
    days_per_month=_DAYS_PER_MONTH,
    map_size_low=Q_(1, "MB"),  # scenario assumption
    map_size_high=Q_(10, "MB"),  # scenario assumption
    attribution_bytes_low=Q_(4, "byte"),  # scenario assumption
    attribution_bytes_high=Q_(8, "byte"),  # scenario assumption
)


Gpt3ShardedUnlearning = _st.ShardedUnlearningScenario(
    name="Forgetting one person from a GPT-3-scale model: full retrain against 100 SISA shards",
    description="Full retrain on 1,024 A100s at about 32 percent MFU; 1,000 deletion requests a day.",
    workload=_Models.Language.GPT3,
    accelerator=_Hardware.Cloud.A100,
    n_accelerators=_Clusters.Training_1K_A100.total_accelerators,
    mfu=0.32,  # scenario assumption: MFU of the retrain
    full_cost=_ReferenceStats.TrainingCostAnchors.Gpt3CostEstimate,
    shards=100,  # scenario assumption
    requests_per_day=1000,  # scenario assumption
    accuracy_drop_low=0.03,  # scenario assumption
    accuracy_drop_high=0.07,  # scenario assumption
)


RewardModelLabelBudget = _st.PreferenceLabelBudgetScenario(
    name="Preference labels for a reward model: 50,000 to 500,000 comparisons",
    description="At 0.50 to 5 dollars per comparison.",
    comparisons_low=50_000,  # scenario assumption
    comparisons_high=500_000,  # scenario assumption
    price_low=Q_(0.50, "USD"),  # scenario assumption
    price_high=Q_(5.00, "USD"),  # scenario assumption
)


RadiologyAutomationBias = _st.AutomationBiasScenario(
    name="A radiologist reviewing an AI that misses 5 percent of tumors",
    description="Radiologist alone 92 percent, AI alone 95 percent; the radiologist accepts 70 "
    "percent (range 60 to 80) of the AI's misses.",
    human_sensitivity=0.92,  # scenario assumption
    ai_sensitivity=0.95,  # scenario assumption
    miss_acceptance=0.70,  # scenario assumption
    acceptance_low=0.60,  # scenario assumption
    acceptance_high=0.80,  # scenario assumption
)


Gpt3PrivacyPlan = _st.DpTrainingPlanScenario(
    name="Differential privacy planned for a GPT-3-scale training run at three budgets",
    description="Assumed accountant results epsilon 1, 8, and 10; the epsilon-1 run needs three times the training.",
    epsilon_strong=1.0,  # scenario assumption
    epsilon_moderate=8.0,  # scenario assumption
    epsilon_weak=10.0,  # scenario assumption
    base_cost=_ReferenceStats.TrainingCostAnchors.Gpt3CostEstimate,
    cost_multiplier=3.0,  # scenario assumption: training multiple at epsilon 1
    accuracy_drop_strong=6.0,  # scenario assumption
    accuracy_drop_moderate=1.0,  # scenario assumption
    clipped_share=40,  # scenario assumption
    overhead_moderate=30.0,  # scenario assumption
)
