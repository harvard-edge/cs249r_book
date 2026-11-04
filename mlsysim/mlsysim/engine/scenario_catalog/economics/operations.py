"""Named economics scenarios for operations.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....core.units import DAYS_PER_YEAR as _DAYS_PER_YEAR
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Clusters as _Clusters
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Nodes as _Nodes
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st
from ....systems.registry import Systems as _Systems


# ── ML operations at scale (added 2026-10-20) ─────────────────────────────
OperationalLoadSketch = _st.OperationalLoadScenario(
    name="Operational load of a model portfolio: alerts N, deployments N ln N, conflicts N^2/2",
    description="Assumed growth shapes, not measurements, against one team's capacity; the total "
    "crosses it near 50 models.",
    alerts_per_model=20,  # scenario assumption: alert units each model adds
    conflict_coefficient=0.5,  # scenario assumption: N models form about N^2/2 pairs
    team_capacity=2500,  # scenario assumption: load one team can absorb
    max_models=500,  # scenario assumption: portfolio range plotted
)


RecommendationFeatureHistory = _st.FeatureHistoryScenario(
    name="Hourly feature snapshots for 100 million users and 1,000 features, kept one year",
    description="Full snapshots, no compression; each row an 8 B value, 8 B entity key, and event and "
    "availability timestamps of 8 B each, as a point-in-time join requires.",
    entities=100e6,  # scenario assumption: users
    features=1000,  # scenario assumption
    retention=Q_(_DAYS_PER_YEAR, "day"),  # scenario assumption: one year of history
    update_interval=Q_(1, "hour"),  # scenario assumption: hourly snapshots
    value_bytes=Q_(8, "byte"),  # scenario assumption
    key_bytes=Q_(8, "byte"),  # scenario assumption
    timestamp_bytes=Q_(8, "byte"),  # scenario assumption
)


DailyBillionRecommendationLookups = _st.FeatureLookupScenario(
    name="One billion recommendations a day, 100 features each",
    description="Average feature-value lookups before peak traffic, and the store's share of a 50 ms "
    "recommendation budget.",
    request_rate=Q_(1e9, "1/day"),  # scenario assumption: recommendations per day
    reads_per_request=100,  # scenario assumption: features per recommendation
    request_budget=Q_(50, "ms"),  # scenario assumption
    lookup_budget_low=Q_(5, "ms"),  # scenario assumption
    lookup_budget_high=Q_(10, "ms"),  # scenario assumption
)


AggregationWindowRevenueIncident = _st.RevenueRegressionScenario(
    name="A pipeline version changes a feature's aggregation window; CTR falls for five days",
    description="Revenue tracks CTR and traffic is uniform; an alert with containment would bound the "
    "exposure to four hours (constructed values).",
    weekly_revenue=Q_(15e6, "USD"),  # scenario assumption: revenue per week
    relative_drop=0.08,  # scenario assumption: CTR and revenue fall
    metric_before=0.042,  # scenario assumption: CTR
    metric_after=0.0387,  # scenario assumption: CTR
    undetected=Q_(5, "day"),  # scenario assumption
    alerted=Q_(4, "hour"),  # scenario assumption: alert and containment
    feature_mean_before=0.42,  # scenario assumption: user_engagement_score mean
    feature_mean_after=0.31,  # scenario assumption
    freshness_slo=Q_(1, "hour"),  # scenario assumption
    shift_threshold=0.10,  # scenario assumption: distribution-gate band
)


DelayedAlertClickLoss = _st.ClickLossScenario(
    name="A silent CTR drop from 5 to 4.5 percent at 5,000 requests/s, caught after one day",
    description="Lost clicks and their value before detection and rollback.",
    request_rate=Q_(5000, "1/s"),  # scenario assumption
    detection_time=Q_(24, "hour"),  # scenario assumption
    ctr_before=0.05,  # scenario assumption
    ctr_after=0.045,  # scenario assumption
    value_per_click=Q_(0.50, "USD"),  # scenario assumption
)


CanaryVersusCutoverExposure = _st.RolloutExposureScenario(
    name="All-at-once cutover against a 5 percent canary, same time to rollback",
    description="Exposure is proportional to the affected traffic share when request rate, harm, and "
    "rollback time are the same.",
    full_share=1.0,
    canary_share=0.05,  # scenario assumption
    time_to_rollback=Q_(2, "hour"),  # scenario assumption: time to detect and roll back
)


CanaryGuardrailSchedule = _st.CanaryScheduleScenario(
    name="Canary ladder for a model at 1 million requests/h, 10,000 guardrail samples per stage",
    description="Minimum stage time n / (r p) and the organization's configured ladder: 1 percent for "
    "2 hours, 5 and 25 percent for 30 minutes each, 50 percent for an hour, then all traffic.",
    samples_per_variant=10000,  # scenario assumption: guardrail sample per variant
    request_rate=Q_(1e6, "1/hour"),  # scenario assumption
    stages=[
        (0.01, Q_(2, "hour")),  # scenario assumption: configured ladder
        (0.05, Q_(30, "minute")),
        (0.25, Q_(30, "minute")),
        (0.50, Q_(1, "hour")),
    ],
)


CtrLiftExperiment = _st.TwoProportionTestScenario(
    name="Two-arm test for a 0.5-point CTR lift on a 5 percent baseline",
    description="95 percent confidence and 80 percent power on 1 million requests a day split evenly.",
    baseline_rate=0.05,  # scenario assumption
    lift=0.005,  # scenario assumption: 0.5 percentage points
    z_alpha=1.96,  # two-sided 5 percent significance
    z_beta=0.84,  # 80 percent power
    daily_requests=Q_(1e6, "1/day"),  # scenario assumption
)


ThreeSigmaModelFleetAlerts = _st.MonitorFleetAlertScenario(
    name="100 models x 10 metrics checked every 5 minutes at 3-sigma thresholds",
    description="Raw false threshold crossings per day on a stable Gaussian process, before grouping.",
    models=100,  # scenario assumption
    metrics_per_model=10,  # scenario assumption
    check_interval=Q_(5, "minute"),  # scenario assumption
    false_positive_rate=0.003,  # 3-sigma two-sided tail, rounded
)


TwoSigmaModelFleetAlerts = ThreeSigmaModelFleetAlerts.variant(
    name="The same fleet at 2-sigma thresholds, 99 percent of crossings deduplicated",
    false_positive_rate=0.05,  # 2-sigma two-sided tail, rounded
    dedup_share=0.99,  # scenario assumption: crossings merged or auto-resolved
)


OwnVsRentH100Cluster1K = _st.OwnVsRentScenario(
    name="Own or rent a 1,024-H100 cluster for three years at 80 percent utilization",
    description="128 DGX H100 nodes plus the reference fabric and facility, a fixed $1.5M a year to run, "
    "against on-demand GPU-hours.",
    gpus=_Clusters.Training_1K.total_accelerators,
    gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
    node_price=_Infrastructure.Pricing.Capital.DgxH100NodeUsd.rate,
    rent_rate=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    utilization=_Systems.Orchestration.target_cluster_utilization,
    years=_Infrastructure.Pricing.Capital.AcceleratorServiceLifeYears.rate,
    network_capex=_ReferenceStats.ClusterTcoProfile.Reference1KNetworkCapex,
    facility_capex=_ReferenceStats.ClusterTcoProfile.Reference1KFacilityCapex,
    opex_per_year=_ReferenceStats.ClusterTcoProfile.Reference1KAnnualOpex,
)


OwnVsRentDgxH100Node = _st.OwnVsRentScenario(
    name="Own or rent one DGX H100 node: purchase and energy only",
    description="The node price amortized over its service life plus facility electricity drawn while "
    "busy, against on-demand GPU-hours.",
    gpus=_Nodes.DGX_H100.accelerators_per_node,
    gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
    node_price=_Infrastructure.Pricing.Capital.DgxH100NodeUsd.rate,
    rent_rate=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    utilization=_Systems.Orchestration.target_cluster_utilization,
    years=_Infrastructure.Pricing.Capital.AcceleratorServiceLifeYears.rate,
    power=_Systems.Racks.DGX_H100_4Node.total_power / _Systems.Racks.DGX_H100_4Node.nodes_per_rack,
    pue=_Infrastructure.FacilityCooling.StateOfArt.pue,
    electricity_price=_Infrastructure.Pricing.OnPremises.ElectricityPerKwh.rate,
    energy_scales_with_use=True,
)


OwnVsRentH100Cluster10K = _st.OwnVsRentScenario(
    name="Own or rent 10,000 H100s for three years at 80 percent utilization",
    description="1,250 DGX H100 nodes, the reference fabric and liquid-cooled hall, GPU-only electricity "
    "at all hours, and staffing; no maintenance or financing.",
    gpus=_Clusters.Training_10K.total_accelerators,
    gpus_per_node=_Nodes.DGX_H100.accelerators_per_node,
    node_price=_Infrastructure.Pricing.Capital.DgxH100NodeUsd.rate,
    rent_rate=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    utilization=_Systems.Orchestration.target_cluster_utilization,
    years=_Infrastructure.Pricing.Capital.AcceleratorServiceLifeYears.rate,
    network_capex=_ReferenceStats.ClusterTcoProfile.Reference10KNetworkCapex,
    facility_capex=_ReferenceStats.ClusterTcoProfile.Reference10KFacilityCapex,
    staffing_per_year=_ReferenceStats.ClusterTcoProfile.Reference10KAnnualStaffing,
    power=_Clusters.Training_10K.total_accelerators * _Hardware.Cloud.H100.tdp,
    pue=_Infrastructure.FacilityCooling.StateOfArt.pue,
    electricity_price=_Infrastructure.Pricing.OnPremises.ElectricityPerKwh.rate,
)


BurstyTrainingOn1KCluster = _st.TrainingCampaignCostScenario(
    name="Six two-to-four-week training runs a year on the owned 1,024-H100 cluster",
    description="Owned lifetime cost per run against renting the same GPUs only while a run lasts.",
    ownership=OwnVsRentH100Cluster1K,
    runs_per_year=6,  # scenario assumption: bursty training cadence
    run_low=Q_(2, "week"),  # scenario assumption: run length
    run_high=Q_(4, "week"),  # scenario assumption: run length
)


StartupOneModelTco = _st.LifecycleTcoScenario(
    name="Monthly lifecycle cost of a one-model startup",
    description="Constructed budget: engineers iterating dominate.",
    monthly={
        "training": Q_(5000, "USD/month"),  # scenario assumption
        "inference": Q_(2000, "USD/month"),  # scenario assumption
        "data": Q_(500, "USD/month"),  # scenario assumption
        "iteration": Q_(40000, "USD/month"),
    },  # scenario assumption
)


FiftyModelCompanyTco = _st.LifecycleTcoScenario(
    name="Monthly lifecycle cost of a 50-model production company",
    description="Constructed budget: inference is the largest term.",
    monthly={
        "training": Q_(150000, "USD/month"),  # scenario assumption
        "inference": Q_(400000, "USD/month"),  # scenario assumption
        "data": Q_(80000, "USD/month"),  # scenario assumption
        "iteration": Q_(350000, "USD/month"),
    },  # scenario assumption
)


Int8InferencePayback = _st.OptimizationPaybackScenario(
    name="INT8 quantization that cuts a $400K monthly inference bill by 40 percent",
    description="One-time implementation and validation cost against the monthly saving.",
    implementation_cost=Q_(80000, "USD"),  # scenario assumption
    monthly_bill=FiftyModelCompanyTco.monthly["inference"],
    reduction=0.40,  # scenario assumption
)


TrainingOptimizationPayback = Int8InferencePayback.variant(
    name="A training optimization of the same cost that cuts a $150K monthly training bill by 30 percent",
    monthly_bill=FiftyModelCompanyTco.monthly["training"],
    reduction=0.30,  # scenario assumption
)


SharedPlatformBenefit50 = _st.PlatformBenefitScenario(
    name="A $120K-a-month ML platform serving 50 model teams, 20 hours saved per team per month",
    description="Engineering time saved at the loaded ML-engineer rate against the platform's cost.",
    models=50,  # scenario assumption
    hours_saved=Q_(20, "hour/month"),  # scenario assumption: per model team
    engineer_rate=_Infrastructure.Pricing.Labor.MlEngineerLoadedHourly.rate,
    platform_cost=Q_(120000, "USD/month"),  # scenario assumption: platform team and infrastructure
)


SharedPlatformBenefit20 = SharedPlatformBenefit50.variant(
    name="The same platform serving 20 model teams",
    models=20,  # scenario assumption
)
