"""Named statistics scenarios for measurement.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ... import scenario_types as _st


# --- Benchmarking: deciding from measurements (added 2026-10-20) --------
AccuracyWithinOnePointSampleSize = _st.ProportionSampleSizeScenario(
    name="Test images that pin a 95 percent accuracy to within one point",
    description="n = z^2 p (1 - p) / m^2 at 95 percent confidence.",
    rate=0.95,  # scenario assumption: accuracy being estimated
    half_width=0.01,  # scenario assumption: one point
    confidence=0.95,  # scenario assumption
)


NinetyFiveAgainstNinetyFourOnSameImages = _st.SameTestSetAccuracyComparisonScenario(
    name="A 95 percent baseline against a 94 percent compressed model on the same 1,000 images",
    description="Independent-set interval on the one-point drop, and McNemar's disagreement limit.",
    n_samples=1000,  # scenario assumption: test images
    baseline_accuracy=0.95,  # scenario assumption
    candidate_accuracy=0.94,  # scenario assumption
    confidence=0.95,  # scenario assumption
)


WiderBeamSearchTranslation = _st.ProxyMetricLatencyScenario(
    name="Beam search widened from 1 to 10 candidates for half a BLEU point",
    description="An offline metric gain that breaks a 100 ms serving deadline (illustrative).",
    original_metric=28.0,  # scenario assumption: BLEU
    optimized_metric=28.5,  # scenario assumption: BLEU
    original_width=1,  # scenario assumption: beam size
    optimized_width=10,  # scenario assumption: beam size
    original_latency=Q_(50, "ms"),  # scenario assumption
    optimized_latency=Q_(200, "ms"),  # scenario assumption
    latency_budget=Q_(100, "ms"),  # scenario assumption: serving deadline
)


MobileNetV2Int8CameraTrapGates = _st.QuantizedAcceptanceGatesScenario(
    name="MobileNetV2 FP32 against INT8 for a camera trap, against gates set before compression",
    description="Assumed top-1, top-5, ECE, and edge-case accuracy; edge cases are about 5 percent of inputs.",
    workload=_Models.Vision.MobileNetV2,
    float_top1=71.8,  # scenario assumption: FP32 top-1, percent
    int_top1=70.9,  # scenario assumption: INT8 top-1, percent
    float_top5=91.0,  # scenario assumption
    int_top5=90.4,  # scenario assumption
    float_ece=0.031,  # scenario assumption
    int_ece=0.089,  # scenario assumption
    float_edge=68.2,  # scenario assumption: edge-case accuracy, percent
    int_edge=61.4,  # scenario assumption
    accuracy_loss_limit=1.0,  # scenario assumption: gate, percentage points
    edge_share=0.05,  # scenario assumption: edge cases among real inputs
    edge_occlusion=0.50,  # scenario assumption: edge-case definition
    edge_illuminance=Q_(100, "lux"),  # scenario assumption: edge-case definition
    edge_rotation_deg=30,  # scenario assumption: edge-case definition
    confidence_threshold=0.85,  # scenario assumption: the trap acts above this confidence
    test_images=1000,  # scenario assumption: lab test set
    confidence=0.95,  # scenario assumption
)
