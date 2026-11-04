"""Named economics scenarios for duties.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st
from ..statistics.fairness import CreditServiceTwoGroups, RecommenderExplorationExposure


# --- Responsible engineering: lifecycle cost, carbon accounting, and duties (added 2026-10-20) ---
RankingServiceLifecycleCost = _st.ServiceLifecycleCostScenario(
    name="Three years of the ranking service: twelve training cycles, unbatched inference, operations",
    description="The exploration-exposure ranking service's traffic at 10 ms of dedicated GPU time per "
    "request, reserved inference and on-demand training prices, and the fleet GPU-hour "
    "carbon factor (GPU only: no host, PUE, or embodied term).",
    daily_users=RecommenderExplorationExposure.daily_users,
    items_per_user=RecommenderExplorationExposure.slots_per_user,
    service_time=Q_(10, "ms"),  # scenario assumption: unbatched, non-overlapped GPU time per request
    inference_rate=_Infrastructure.Pricing.Cloud.GpuInferencePerHour.rate,
    training_rate=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    data_prep_time=Q_(100, "hour"),  # scenario assumption: data preparation GPU-hours per cycle
    search_experiments=50,  # scenario assumption: hyperparameter configurations per cycle
    search_cost_per_experiment=Q_(40, "USD"),  # scenario assumption
    final_run_time=Q_(200, "hour"),  # scenario assumption: final training GPU-hours per cycle
    cycles=12,  # scenario assumption: quarterly retraining, initial cycle included
    years=3,  # scenario assumption: operational period
    operations_annual={
        "monitoring": Q_(50_000, "USD"),  # scenario assumption
        "on-call": Q_(100_000, "USD"),  # scenario assumption: 0.5 FTE
        "incident response": Q_(20_000, "USD"),
    },  # scenario assumption
    carbon_per_gpu_hour=_Infrastructure.Pricing.Fleet.CarbonPerGpuHr.rate,
    serving_cut=0.20,  # scenario assumption: per-query accelerator demand reduction
    training_cut=0.50,  # scenario assumption: halving all training
)


Gpt3TrainingCarbon = _st.OperationalCarbonScenario(
    name="GPT-3 training energy on the US average grid",
    description="1,287 MWh, already at the facility boundary (PUE 1.1 included), against passenger-car years.",
    it_energy=_Models.Language.GPT3.training_energy_mwh,
    pue=1.0,  # the training energy already includes the datacenter's PUE
    intensity=Q_(_Infrastructure.Grids.US_Avg.carbon_intensity_g_kwh, "g/kWh"),
    comparison=Q_(4.6, "metric_ton"),  # scenario assumption: one passenger car's annual emissions
)


CreditDenialExplanationLoad = _st.ExplanationLoadScenario(
    name="Reasons for every denial of a credit service making 1 million decisions a day",
    description="Denial share from the two-group credit service; a 1,000-sample attributor against "
    "exact Shapley over 20 features.",
    decisions_per_day=1_000_000,  # scenario assumption
    denial_share=1 - CreditServiceTwoGroups.pooled_positive_rate,
    samples=1_000,  # scenario assumption: model evaluations per sampled explanation
    features=20,  # scenario assumption: input features of the credit service
)


CreditAppealReview = _st.ReviewStaffingScenario(
    name="Substantive review of appeals against the credit service, against Cigna's reported pace",
    description="0.1 percent of 1 million daily decisions appealed, 6 minutes each, 8-hour shifts; "
    "ProPublica's Cigna denials for comparison.",
    decisions_per_day=CreditDenialExplanationLoad.decisions_per_day,
    appeal_rate=0.001,  # scenario assumption: share of decisions appealed
    time_per_case=Q_(6, "minute"),  # scenario assumption
    shift=Q_(8, "hour"),  # scenario assumption
    compare_cases=300000.0,  # Rucker, Miller, and Armstrong, How Cigna Saves Millions by Having I...
    compare_time_per_case=Q_(1.2, "s"),  # Rucker, Miller, and Armstrong, How Cigna Saves Millions by Having I...
)


DpSgdPerExampleGradients = _st.PerExampleGradientScenario(
    name="Per-example FP32 gradients for the KWS DS-CNN and GPT-2 at batch 256",
    description="Naive DP-SGD clipping keeps every example's gradient before the batch sum.",
    models={"KWS": _Models.Tiny.DS_CNN, "GPT-2": _Models.Language.GPT2},
    batch_size=256,  # scenario assumption: DP-SGD batch size
    bytes_per_param=_BYTES_FP32,
)
