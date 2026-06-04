"""Reference statistics for real-world scenarios and case studies.

This registry is the home for reusable real-world reference figures:
illustrative scale anchors (Gmail volume, Waymo sensor rate) and case-study model
metrics (the TinyML anomaly detector). Every value carries sourced() provenance.

Note: *evaluatable* scenario bundles (workload + system + SLA, with .evaluate())
live in the Scenario model in core/scenarios.py. This registry is the
reference-statistics counterpart — sourced numbers the prose cites, not things to run.
"""
from ..core.provenance import sourced, sourced_qty
from ..core.registry import Registry
from ..core import provenance_catalog as pc
from ..core.units import ureg, GB, MB, MWh, TB, byte, count, day, minute, param, TRILLION

_hour = ureg.hour
_joule = ureg.joule


class Workloads(Registry):
    """Illustrative real-world workload scale anchors (order-of-magnitude intuition)."""

    GmailEmailsPerDay = sourced(
        121e9, pc.REFERENCE_WORKLOAD_SCALE,
        name="Gmail emails per day", description="Approximate daily Gmail volume.")
    GoogleSearchesPerDay = sourced(
        8.5e9, pc.REFERENCE_WORKLOAD_SCALE,
        name="Google searches per day", description="Approximate daily Google search volume.")
    WaymoDataPerHourLow = sourced_qty(
        1 * TB / _hour, pc.REFERENCE_WORKLOAD_SCALE,
        name="Waymo sensor data rate (low)", description="Lower-bound AV sensor data generation rate.")
    WaymoDataPerHourHigh = sourced_qty(
        19 * TB / _hour, pc.REFERENCE_WORKLOAD_SCALE,
        name="Waymo sensor data rate (high)", description="Upper-bound AV sensor data generation rate.")


class AnomalyModel(Registry):
    """TinyML anomaly-detection case study (benchmarking example)."""

    Latency = sourced_qty(
        10.4 * ureg.ms, pc.TINYML_ANOMALY_CASE,
        name="Anomaly model latency", description="Inference latency of the TinyML anomaly detector.")
    Auc = sourced(
        0.86, pc.TINYML_ANOMALY_CASE,
        name="Anomaly model AUC", description="Area under the ROC curve for the TinyML anomaly detector.")
    Energy = sourced_qty(
        516 * ureg.microjoule, pc.TINYML_ANOMALY_CASE,
        name="Anomaly model energy", description="Per-inference energy of the TinyML anomaly detector.")


class OuraSleepStudy(Registry):
    """Oura Ring sleep-stage case-study anchors."""

    Participants = sourced(
        106, pc.OURA_SLEEP_STAGE_STUDY,
        name="Oura sleep-study participants",
        description="Participants in the Oura Ring sleep-stage validation study.")
    RecordingNights = sourced(
        440, pc.OURA_SLEEP_STAGE_STUDY,
        name="Oura sleep-study nights",
        description="Recorded nights in the Oura Ring sleep-stage validation study.")
    RecordingHours = sourced_qty(
        3444 * _hour, pc.OURA_SLEEP_STAGE_STUDY,
        name="Oura sleep-study recording hours",
        description="Combined PSG and wearable-ring recording duration.")
    CrossValidationFolds = sourced(
        5, pc.OURA_SLEEP_STAGE_STUDY,
        name="Oura sleep-stage cross-validation folds",
        description="Cross-validation folds used for model evaluation.")
    AccelOnlyAccuracy = sourced(
        0.57, pc.OURA_SLEEP_STAGE_STUDY,
        name="Oura accelerometer-only sleep-stage accuracy",
        description="Four-stage sleep classification accuracy for the accelerometer-only baseline.")
    EnhancedAccuracy = sourced(
        0.79, pc.OURA_SLEEP_STAGE_STUDY,
        name="Oura enhanced sleep-stage accuracy",
        description="Four-stage sleep classification accuracy for the enhanced multi-sensor model.")
    PsgScorerAgreementLow = sourced(
        0.82, pc.OURA_SLEEP_STAGE_STUDY,
        name="PSG inter-scorer agreement low",
        description="Lower bound of the expert PSG inter-scorer agreement band used as a practical ceiling.")
    PsgScorerAgreementHigh = sourced(
        0.83, pc.OURA_SLEEP_STAGE_STUDY,
        name="PSG inter-scorer agreement high",
        description="Upper bound of the expert PSG inter-scorer agreement band used as a practical ceiling.")


class ClinicalImaging(Registry):
    """Clinical-imaging workflow anchors used by edge-deployment examples."""

    RetinalPhotoSize = sourced_qty(
        5.0 * MB, pc.CLINICAL_IMAGING_WORKFLOW_ANCHORS,
        name="Retinal screening image size",
        description="Reference size for one retinal screening photograph in the rural-clinic workflow.")


class EnergyAnchors(Registry):
    """Everyday energy-scale comparison anchors (order-of-magnitude intuition)."""

    SmartphoneCharge = sourced_qty(
        40_000 * _joule, pc.ENERGY_SCALE_ANCHORS,
        name="Smartphone full charge", description="Approximate energy to fully charge a smartphone battery.")
    BoilingWater = sourced_qty(
        100_000 * _joule, pc.ENERGY_SCALE_ANCHORS,
        name="Boiling 1 L of water", description="Approximate energy to bring one liter of water to a boil.")
    USHouseholdAnnualElectricity = sourced_qty(
        10.7 * MWh, pc.ENERGY_SCALE_ANCHORS,
        name="US household annual electricity",
        description="Rounded annual electricity use baseline for one average US household-year.")


class EmissionsAnchors(Registry):
    """Everyday emissions-scale comparison anchors (order-of-magnitude intuition)."""

    TransatlanticRoundTripCo2Kg = sourced(
        1000.0,
        pc.LIT_TRANSATLANTIC_ROUND_TRIP_CO2,
        name="Transatlantic round-trip passenger CO₂e",
        description="One economy passenger, New York to London and return (kg CO₂e).",
    )


class TrainingScaleProfiles(Registry):
    """Reusable scenario assumptions for distributed training scale efficiency."""

    Eff32Gpu = sourced(
        0.9,
        pc.SCALING_EFFICIENCY_TIERS,
        name="Scaling efficiency (32 GPUs)",
        description="Near-linear scaling regime for a reference training scenario.",
    )
    Eff256Gpu = sourced(
        0.7,
        pc.SCALING_EFFICIENCY_TIERS,
        name="Scaling efficiency (256 GPUs)",
        description="Reference training scenario where communication begins to reduce scaling efficiency.",
    )
    Eff1024Gpu = sourced(
        0.5,
        pc.SCALING_EFFICIENCY_TIERS,
        name="Scaling efficiency (1024 GPUs)",
        description="Reference training scenario with significant communication overhead at 1k GPUs.",
    )
    Eff8192Gpu = sourced(
        0.35,
        pc.MEGASCALE,
        name="Scaling efficiency (8192 GPUs)",
        description="Illustrative scaling efficiency at 8192 GPUs for LLM training.",
    )


class StorageTrainingCorpus(Registry):
    """Reusable storage-chapter running example for a 175B-model training corpus."""

    TrainingTokens = sourced_qty(
        1.5 * TRILLION * count,
        pc.STORAGE_TRAINING_CORPUS_REFERENCE,
        name="Storage running-example training tokens",
        description="Reference token count for the 175B-model storage running example.",
    )
    CompressedSource = sourced_qty(
        3 * TB,
        pc.STORAGE_TRAINING_CORPUS_REFERENCE,
        name="Storage running-example compressed source corpus",
        description="Compressed source corpus size for the storage running example.",
    )
    TokenIdBytes = sourced_qty(
        4 * byte,
        pc.STORAGE_TRAINING_CORPUS_REFERENCE,
        name="Storage running-example token ID width",
        description="Serialized token-ID width for the tokenized corpus.",
    )
    TokenizedText = sourced_qty(
        TrainingTokens * TokenIdBytes,
        pc.STORAGE_TRAINING_CORPUS_REFERENCE,
        name="Storage running-example tokenized corpus",
        description="Derived serialized token-ID corpus size for one epoch.",
    )
    TrainingWindow = sourced_qty(
        30 * day,
        pc.STORAGE_TRAINING_CORPUS_REFERENCE,
        name="Storage running-example training window",
        description="Reference training-window duration for checkpoint-count examples.",
    )
    CheckpointInterval = sourced_qty(
        10 * minute,
        pc.STORAGE_TRAINING_CORPUS_REFERENCE,
        name="Storage running-example checkpoint interval",
        description="Reference checkpoint interval for checkpoint-count examples.",
    )
    CheckpointBytesPerParameter = sourced_qty(
        10 * (byte / param),
        pc.STORAGE_TRAINING_CORPUS_REFERENCE,
        name="Storage running-example checkpoint bytes per parameter",
        description="Reference checkpoint footprint per model parameter.",
    )


class ModelLoading(Registry):
    """Reusable cold-start model-loading scenario anchors."""

    StableDiffusionV15CheckpointSize = sourced_qty(
        5 * GB,
        pc.MODEL_LOADING_SCENARIO_ASSUMPTIONS,
        name="Stable Diffusion v1.5 serialized checkpoint size",
        description="Representative checkpoint footprint for cold-start model-loading examples.",
    )
    StableDiffusionV15PickleLoadTime = sourced_qty(
        15 * ureg.second,
        pc.MODEL_LOADING_SCENARIO_ASSUMPTIONS,
        name="Stable Diffusion v1.5 Pickle load time",
        description="Reference cold-start load time for the object-reconstruction path.",
    )
    StableDiffusionV15SafetensorsLoadTime = sourced_qty(
        0.5 * ureg.second,
        pc.MODEL_LOADING_SCENARIO_ASSUMPTIONS,
        name="Stable Diffusion v1.5 Safetensors load time",
        description="Reference cold-start load time for the memory-mapped tensor path.",
    )


class MobilePower(Registry):
    """Mobile/edge device + workload power-envelope reference figures."""

    MobileNpuTypical = sourced_qty(3 * ureg.watt, pc.MOBILE_DEVICE_ANCHORS,
        name="Mobile NPU typical power", description="Typical sustained power for on-device mobile inference.")
    MobileNpuPeak = sourced_qty(4 * ureg.watt, pc.MOBILE_DEVICE_ANCHORS,
        name="Mobile NPU peak power", description="Higher-bound mobile inference power envelope.")
    ObjectDetector = sourced_qty(2 * ureg.watt, pc.MOBILE_DEVICE_ANCHORS,
        name="Object-detector power", description="Reference power draw of an always-on object-detection workload.")
    MobileMlSustainedLow = sourced_qty(2 * ureg.watt, pc.MOBILE_DEVICE_ANCHORS,
        name="Mobile ML sustained power (low)", description="Lower bound for sustained smartphone ML processing.")
    MobileMlSustainedHigh = sourced_qty(3 * ureg.watt, pc.MOBILE_DEVICE_ANCHORS,
        name="Mobile ML sustained power (high)", description="Upper bound for sustained smartphone ML processing.")
    MobileMlBurstLow = sourced_qty(5 * ureg.watt, pc.MOBILE_DEVICE_ANCHORS,
        name="Mobile ML burst power (low)", description="Lower bound for brief smartphone ML burst processing.")
    MobileMlBurstHigh = sourced_qty(10 * ureg.watt, pc.MOBILE_DEVICE_ANCHORS,
        name="Mobile ML burst power (high)", description="Upper bound for brief smartphone ML burst processing.")
    BackgroundAdaptationLow = sourced_qty(500 * ureg.milliwatt, pc.MOBILE_DEVICE_ANCHORS,
        name="Background adaptation power budget (low)", description="Lower bound for background on-device adaptation power.")
    BackgroundAdaptationHigh = sourced_qty(1000 * ureg.milliwatt, pc.MOBILE_DEVICE_ANCHORS,
        name="Background adaptation power budget (high)", description="Upper bound for background on-device adaptation power.")


class PhoneBattery(Registry):
    """Flagship smartphone battery reference figures.

    Note: EnergyWh (15 Wh, a flagship pack rating) and EnergyJ (capacity x voltage =
    3000 mAh x 3.7 V = 11.1 Wh) are two DISTINCT battery reference models used in
    different scenarios; both are preserved as-is rather than reconciled."""

    CapacityMah = sourced_qty(3000 * ureg.milliampere_hour, pc.MOBILE_DEVICE_ANCHORS,
        name="Phone battery capacity", description="Typical flagship smartphone battery charge capacity.")
    VoltageV = sourced_qty(3.7 * ureg.volt, pc.MOBILE_DEVICE_ANCHORS,
        name="Phone battery voltage", description="Nominal Li-ion cell voltage.")
    EnergyJ = sourced_qty((3000 * ureg.milliampere_hour * 3.7 * ureg.volt).to(ureg.joule), pc.MOBILE_DEVICE_ANCHORS,
        name="Phone battery energy (capacity x voltage)", description="Battery energy derived from capacity x voltage.")
    EnergyWh = sourced_qty(15 * ureg.watt * ureg.hour, pc.MOBILE_DEVICE_ANCHORS,
        name="Flagship phone battery energy", description="Modern flagship smartphone battery pack energy rating.")


class ReferenceStats(Registry):
    """Registry namespace for non-executable real-world scenario statistics."""

    Workloads = Workloads
    AnomalyModel = AnomalyModel
    OuraSleepStudy = OuraSleepStudy
    ClinicalImaging = ClinicalImaging
    EnergyAnchors = EnergyAnchors
    EmissionsAnchors = EmissionsAnchors
    TrainingScaleProfiles = TrainingScaleProfiles
    StorageTrainingCorpus = StorageTrainingCorpus
    ModelLoading = ModelLoading
    MobilePower = MobilePower
    PhoneBattery = PhoneBattery


# Backward-compatible alias while book cells migrate to ``ReferenceStats``.
Scenarios = ReferenceStats
