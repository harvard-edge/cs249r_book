"""Named statistics scenarios for monitoring.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st
from ..reliability.release import DlrmReleaseFootprintH100, RankingServiceAccuracyDrift


# --- ML operations, part 2: watching and acting (added 2026-10-20) -------
# Monitoring, alerting, and retraining on the same ranking service.
LabeledOutcomeDriftConfirmation = _st.DriftConfirmationScenario(
    name="Confirming a 5-point drop from 95 percent accuracy with 1,000 labeled outcomes",
    description="The policy's evidence budget above the one-sample binomial floor, collected at one "
    "labeled outcome a second or 100 a day. Illustrative.",
    baseline_accuracy=0.95,  # scenario assumption: original accuracy
    drop=0.05,  # scenario assumption: a 5-percentage-point drop to confirm
    confidence=0.95,  # scenario assumption: two-sided confidence
    power=0.80,  # scenario assumption: chance of catching a real drop of that size
    evidence_budget=1000,  # scenario assumption: the monitoring policy's budget
    label_rates={
        "high": Q_(1, "1/s"),  # scenario assumption: one labeled outcome a second
        "low": Q_(100, "1/day"),
    },  # scenario assumption: 100 labeled outcomes a day
    compared_drops=[0.02, 0.01, 0.005],  # scenario assumption: smaller drops to size
)


RecommenderDetectionDelayLoss = _st.DetectionDelayLossScenario(
    name="A 5 percent conversion drop on a $50M-a-year recommender, found monthly or daily",
    description="Revenue lost per incident at a 28-day review and at a daily check, four incidents a year.",
    annual_value=Q_(50e6, "USD"),  # scenario assumption: a year's recommendation revenue
    quality_drop=0.05,  # scenario assumption: conversion-rate degradation
    slow_detection=Q_(28, "day"),  # scenario assumption: monthly review cycle (about four weeks)
    fast_detection=Q_(1, "day"),  # scenario assumption: daily automated monitoring
    incidents_per_year=4,  # scenario assumption: a high-drift domain
)


RankingRetrainingInterval = _st.RetrainingIntervalScenario(
    name="Retraining interval of the ranking service: 1 million queries a day at $5,000 a run",
    description="Square-root optimum of run cost against staleness on the service's fitted decay; the "
    "run's compute is four H100s for six hours at the cloud training rate. Illustrative.",
    drift=RankingServiceAccuracyDrift,
    query_rate=Q_(1e6, "1/day"),  # scenario assumption: ranking requests per day
    value_per_query=Q_(0.50, "USD"),  # scenario assumption: value per query of a unit accuracy change
    run_cost=Q_(5000, "USD"),  # scenario assumption: total cost of one retraining run
    footprint=DlrmReleaseFootprintH100,
    accelerator_price=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    compared_intervals=[Q_(1, "day"), Q_(3, "day"), Q_(7, "day"), Q_(30, "day")],  # scenario assumption
    scheduled_interval=Q_(90, "day"),  # scenario assumption: a quarterly schedule
)


RankingRetrainingIntervalFourfoldTraffic = RankingRetrainingInterval.variant(
    name="The same retraining interval at four times the traffic",
    query_rate=Q_(1e6, "1/day") * 4,  # scenario assumption: a fourfold change in query volume
)


RankingUserAgePsi = _st.BinShareShiftScenario(
    name="User-age PSI of the ranking service as its audience ages",
    description="Six age bins; four move by 3 points and two by 2. Illustrative.",
    bins=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
    baseline_shares=[0.15, 0.25, 0.20, 0.18, 0.12, 0.10],  # scenario assumption: training
    current_shares=[0.12, 0.22, 0.18, 0.20, 0.15, 0.13],  # scenario assumption: serving
)


OneModelDailyMetricTests = _st.MonitorFleetAlertScenario(
    name="One model in 3 deployment variants x 50 metrics, each tested once a day at 0.05",
    description="False alerts a day from a monitor with nothing to find; the variants are production, "
    "canary, and staging.",
    models=3,  # scenario assumption: deployment variants of one model
    metrics_per_model=50,  # scenario assumption: metrics per deployment
    check_interval=Q_(1, "day"),  # scenario assumption: one test per series a day
    false_positive_rate=0.05,  # scenario assumption: per-test false-positive rate
)


RecommendationAccuracyByDevice = _st.SegmentAccuracyScenario(
    name="Recommendation accuracy by device: tablets fail behind a healthy aggregate",
    description="Traffic share and accuracy of four device segments. Illustrative.",
    traffic_shares={"Desktop": 0.45, "iOS": 0.30, "Android": 0.20, "Tablet": 0.05},  # scenario assumption
    accuracies={"Desktop": 0.94, "iOS": 0.92, "Android": 0.88, "Tablet": 0.62},  # scenario assumption
)
