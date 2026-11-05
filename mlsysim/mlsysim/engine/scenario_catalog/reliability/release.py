"""Named reliability scenarios for release.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from statistics import NormalDist as _NormalDist
from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_INT8 as _BYTES_INT8
from ... import scenario_types as _st


# --- ML operations, part 1 (added 2026-10-20) ------------------------------
# The ranking service (DLRM tables on H100s) and the keyword spotter
# (DS-CNN on an ESP32-S3) as release, rebuild, and parity scenarios.
RankingServiceAccuracyDrift = _st.AccuracyDecayScenario(
    name="The ranking service's click-prediction accuracy, launch against four weeks later",
    description="94 percent at launch and 88 percent four weeks later after a competitor's promotion "
    "shifted clicks; a product manager noticed in week eight. Illustrative.",
    initial_accuracy=0.94,  # scenario assumption: accuracy at launch
    later_accuracy=0.88,  # scenario assumption: accuracy four weeks later
    elapsed=Q_(28, "day"),  # scenario assumption: four weeks
    noticed=Q_(8, "week"),  # scenario assumption: when a product manager noticed
)


DlrmReleaseFootprintH100 = _st.ReleaseFootprintScenario(
    name="Release controls for the DLRM ranking service on an eight-H100 node",
    description="Each control holds a second copy of the 100 GB tables; a retraining run takes four "
    "H100s for six hours; a 5 percent canary ramps through 25 and 50 percent to all traffic.",
    model=_Models.Recommendation.DLRM,
    device=_Hardware.Cloud.H100,
    node_accelerators=8,  # scenario assumption: one eight-accelerator node
    retrain_accelerators=4,  # scenario assumption: accelerators a retraining run occupies
    retrain_time=Q_(6, "hour"),  # scenario assumption: duration of a retraining run
    canary_share=0.05,  # scenario assumption: the canary's share of live queries
    ramp=[0.05, 0.25, 0.50, 1.0],  # scenario assumption: staged exposure after the canary
)


ResNet50Fp16ReleaseFootprintH100 = DlrmReleaseFootprintH100.variant(
    name="Release controls for an FP16 ResNet-50 service on an H100",
    description="The same controls; a copy is the FP16 weights.",
    model=_Models.Vision.ResNet50,
    bytes_per_param=_BYTES_FP16,
)


KeywordSpotterDualBankEsp32S3 = _st.DualBankUpdateScenario(
    name="Dual-bank over-the-air update of the INT8 keyword spotter on an ESP32-S3",
    description="The device runs from bank A while bank B receives the new model; two INT8 DS-CNN "
    "copies in the microcontroller's flash.",
    model=_Models.Tiny.DS_CNN,
    device=_Hardware.Tiny.ESP32_S3,
    bytes_per_param=_BYTES_INT8,
)


RetrainingPipelineAutomation = _st.AutomationBreakevenScenario(
    name="An 80-hour retraining pipeline against 4 hours a week of manual retraining",
    description="Simplified break-even with no pipeline upkeep.",
    build_time=Q_(80, "hour"),  # scenario assumption: one-time pipeline build
    manual_rate=Q_(4, "hour/week"),  # scenario assumption: manual retraining effort
)


RankingSkewOnePercent = _st.SkewCostScenario(
    name="One percent skew-induced errors on 1 million queries a day at $0.10 each",
    description="The yearly cost of training-serving skew that persists until consistency is restored.",
    skew_rate=0.01,  # scenario assumption: fraction of queries skew makes wrong
    query_rate=Q_(1e6, "1/day"),  # scenario assumption: daily query volume
    cost_per_error=Q_(0.10, "USD"),  # scenario assumption: business cost per wrong prediction
)


ConversionRelativeLiftExperiment = _st.TwoProportionTestScenario(
    name="Two-arm test for a 2 percent relative lift on a 5 percent conversion rate",
    description="Two-sided 95 percent confidence and 80 percent power at exact normal quantiles; "
    "25,000 users per variant are what the traffic allows.",
    baseline_rate=0.05,  # scenario assumption
    lift=0.05 * (1 + 0.02) - 0.05,  # scenario assumption: 2 percent relative, 5.0 to 5.1 percent
    z_alpha=_NormalDist().inv_cdf(1 - (1 - 0.95) / 2),
    z_beta=_NormalDist().inv_cdf(0.80),
    confidence=0.95,  # scenario assumption
    power=0.80,  # scenario assumption
    relative_lift=0.02,  # scenario assumption
    available_per_variant=25_000,  # scenario assumption: users per variant traffic allows
)
