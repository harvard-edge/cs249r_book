"""Reference statistics for real-world scenarios and case studies.

This registry is the home for reusable real-world reference figures:
illustrative scale anchors (Gmail volume, Waymo sensor rate) and case-study model
metrics (the TinyML anomaly detector). Every value carries sourced() provenance.

Note: *evaluatable* scenario bundles (workload + system + SLA, with .evaluate())
live in the Scenario model in engine/scenarios.py. This registry is the
reference-statistics counterpart — sourced numbers the prose cites, not things to run.
"""
from ..core.provenance import sourced, sourced_qty
from ..core.registry import Registry
from ..core import provenance_catalog as pc
from ..core.units import (
    ureg,
    GB,
    GiB,
    KiB,
    MB,
    MiB,
    GFLOPs,
    MILLION,
    MWh,
    TB,
    TOPS,
    USD,
    byte,
    count,
    day,
    minute,
    param,
    second,
    TRILLION,
)

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
        # 4.186 J/g·K x 80 K x ~250 g ≈ 84 kJ; ~100 kJ with kettle losses. A full
        # LITER takes ~335 kJ — the anchor is cup-sized, and the name now says so
        # (audit fix 2026-06-06: value was right for a cup, name claimed 1 L).
        100_000 * _joule, pc.ENERGY_SCALE_ANCHORS,
        name="Boiling a cup (~250 mL) of water",
        description="Approximate energy to bring one cup (~250 mL) of water to a boil.")
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


class TrainingCostAnchors(Registry):
    """Cited training-cost estimates anchoring the training-vs-inference cost asymmetry."""

    Gpt2Cost2019 = sourced_qty(
        50_000 * USD,
        pc.GPT2_TRAINING_COST_EST,
        name="GPT-2 training cost (2019)",
        description="Widely cited estimate of GPT-2 (1.5B) cloud training cost in 2019.",
    )
    Gpt4CostEstimate = sourced_qty(
        100 * MILLION * USD,
        pc.GPT4_TRAINING_COST_EST,
        name="GPT-4 training cost (estimate)",
        description="Commonly cited nine-figure estimate for GPT-4-class training runs.",
    )


class FleetEvolution(Registry):
    """Reusable conclusion profile for multiplicative fleet-efficiency reasoning."""

    TargetGain = sourced(
        100,
        pc.FLEET_EVOLUTION_HEURISTIC,
        name="Fleet-evolution target gain",
        description="Reference total efficiency gain for conclusion synthesis.",
    )
    HardwareGain = sourced(
        4.0,
        pc.FLEET_EVOLUTION_HEURISTIC,
        name="Fleet-evolution hardware gain",
        description="Reference workload-dependent hardware contribution.",
    )
    AlgorithmGain = sourced(
        2.5,
        pc.FLEET_EVOLUTION_HEURISTIC,
        name="Fleet-evolution algorithmic gain",
        description="Reference workload-compatible compression or distillation contribution.",
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
        14 * (byte / param),
        pc.STORAGE_TRAINING_CORPUS_REFERENCE,
        name="Storage running-example checkpoint bytes per parameter",
        description="Resumable mixed-precision Adam checkpoint footprint: FP16 weights, FP32 master weights, and two FP32 moments; gradients excluded.",
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
        1.5 * ureg.second,
        pc.MODEL_LOADING_SCENARIO_ASSUMPTIONS,
        name="Stable Diffusion v1.5 Safetensors load time",
        description="Reference cold-start load time for the memory-mapped tensor path at Gen3 NVMe bandwidth.",
    )
    PcieSwapReferenceModelSize = sourced_qty(
        10 * GB,
        pc.MODEL_LOADING_SCENARIO_ASSUMPTIONS,
        name="PCIe model-swap reference model size",
        description="Reference model footprint for host-to-device model-swap latency examples.",
    )


class ServingProfiles(Registry):
    """Reusable serving-shape profiles for performance-engineering examples."""

    H100VendorMemoryBudget = sourced_qty(
        80 * GiB,
        pc.LLM_SERVING_PRECISION_DIVIDEND_PROFILE,
        name="H100 vendor-facing serving memory budget",
        description="Vendor-facing H100 memory budget used for serving capacity arithmetic.",
    )
    PrecisionDividendTensorParallelDegree = sourced(
        8,
        pc.LLM_SERVING_PRECISION_DIVIDEND_PROFILE,
        name="Precision-dividend tensor-parallel degree",
        description="H100 tensor-parallel degree for the 70B LLM KV-cache precision-dividend example.",
    )
    PrecisionDividendContextLengthTokens = sourced(
        4096,
        pc.LLM_SERVING_PRECISION_DIVIDEND_PROFILE,
        name="Precision-dividend context length",
        description="Reference context length in tokens for the 70B LLM KV-cache precision-dividend example.",
    )
    PrecisionDividendBaselinePolicyBatchLimit = sourced(
        4,
        pc.LLM_SERVING_PRECISION_DIVIDEND_PROFILE,
        name="Baseline policy batch limit",
        description="Baseline maximum admitted batch size in the 70B LLM serving case study.",
    )
    PrecisionDividendOptimizedPolicyBatchLimit = sourced(
        32,
        pc.LLM_SERVING_PRECISION_DIVIDEND_PROFILE,
        name="Optimized policy batch limit",
        description="Post-precision maximum admitted batch size in the 70B LLM serving case study.",
    )
    PrecisionDividendSpeculationBatchThreshold = sourced(
        16,
        pc.LLM_SERVING_PRECISION_DIVIDEND_PROFILE,
        name="Speculation policy batch threshold",
        description="Batch-size threshold below which the case-study policy enables speculative decoding.",
    )
    HeterogeneousRoutingH100Servers = sourced(
        10,
        pc.HETEROGENEOUS_ROUTING_SCENARIO,
        name="Heterogeneous routing H100 servers",
        description="Number of H100 serving servers in the weighted-routing example.",
    )
    HeterogeneousRoutingA100Servers = sourced(
        20,
        pc.HETEROGENEOUS_ROUTING_SCENARIO,
        name="Heterogeneous routing A100 servers",
        description="Number of A100 serving servers in the weighted-routing example.",
    )
    HeterogeneousRoutingH100CapacityQps = sourced(
        1000,
        pc.HETEROGENEOUS_ROUTING_SCENARIO,
        name="H100 per-server routing capacity",
        description="Workload-specific per-server QPS capacity for H100 routing examples.",
    )
    HeterogeneousRoutingA100CapacityQps = sourced(
        600,
        pc.HETEROGENEOUS_ROUTING_SCENARIO,
        name="A100 per-server routing capacity",
        description="Workload-specific per-server QPS capacity for A100 routing examples.",
    )
    HeterogeneousRoutingTargetQps = sourced(
        15000,
        pc.HETEROGENEOUS_ROUTING_SCENARIO,
        name="Heterogeneous routing target traffic",
        description="Total incoming QPS target for the H100/A100 weighted-routing example.",
    )
    CircuitBreakerErrorThreshold = sourced(
        0.50,
        pc.CIRCUIT_BREAKER_SERVING_PROFILE,
        name="Circuit-breaker error threshold",
        description="Reference error-rate threshold for opening a GPU inference circuit breaker.",
    )
    CircuitBreakerLatencyThresholdMultiple = sourced(
        2,
        pc.CIRCUIT_BREAKER_SERVING_PROFILE,
        name="Circuit-breaker latency threshold multiple",
        description="Reference latency multiple over baseline for opening a GPU inference circuit breaker.",
    )
    CircuitBreakerOpenDuration = sourced_qty(
        30 * ureg.second,
        pc.CIRCUIT_BREAKER_SERVING_PROFILE,
        name="Circuit-breaker open duration",
        description="Reference recovery interval before probing a half-open GPU inference circuit breaker.",
    )
    CircuitBreakerHalfOpenRequests = sourced(
        5,
        pc.CIRCUIT_BREAKER_SERVING_PROFILE,
        name="Circuit-breaker half-open probe requests",
        description="Reference number of probe requests allowed while half-open.",
    )


class CheckpointArchetypes(Registry):
    """Reusable checkpoint-size scenario anchors for fault-tolerance examples."""

    MixedPrecisionOptimizerBytesPerParameter = sourced_qty(
        12 * (byte / param),
        pc.CHECKPOINT_ARCHETYPE_SCENARIO_ASSUMPTIONS,
        name="Mixed-precision optimizer checkpoint bytes per parameter",
        description="Reference checkpoint footprint for FP32 master weights plus optimizer state.",
    )
    Dense20BTransformerCheckpointSize = sourced_qty(
        240 * GB,
        pc.CHECKPOINT_ARCHETYPE_SCENARIO_ASSUMPTIONS,
        name="20B dense transformer checkpoint size",
        description="Representative checkpoint footprint for a 20B dense transformer class model.",
    )
    EmbeddingHeavyRecommenderCheckpointSize = sourced_qty(
        4 * TB,
        pc.CHECKPOINT_ARCHETYPE_SCENARIO_ASSUMPTIONS,
        name="Embedding-heavy recommender checkpoint size",
        description="Representative checkpoint footprint for an embedding-heavy recommender.",
    )
    MediumVisionTransformerCheckpointSize = sourced_qty(
        1.2 * GB,
        pc.CHECKPOINT_ARCHETYPE_SCENARIO_ASSUMPTIONS,
        name="Medium vision-transformer checkpoint size",
        description="Representative checkpoint footprint for a medium vision transformer.",
    )


class EdgeDeviceSpectrum(Registry):
    """Reusable device-class range endpoints for edge-learning heterogeneity."""

    TinyRamLow = sourced_qty(
        32 * KiB,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Tiny sensor-class RAM endpoint",
        description="Lower memory endpoint used for edge device-spectrum examples.",
    )
    MicrocontrollerSram = sourced_qty(
        256 * KiB,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Microcontroller SRAM endpoint",
        description="Representative microcontroller SRAM endpoint used in memory-wall examples.",
    )
    MicrocontrollerSramHigh = sourced_qty(
        2 * MiB,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Microcontroller SRAM upper endpoint",
        description="Upper microcontroller SRAM endpoint used in memory hierarchy examples.",
    )
    ArduinoNano33BleFlash = sourced_qty(
        1 * MiB,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Arduino Nano 33 BLE Sense flash",
        description="Flash storage anchor used in the Arduino memory-wall example.",
    )
    FlagshipSmartphoneRamHigh = sourced_qty(
        16 * GiB,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Flagship smartphone RAM endpoint",
        description="Upper mobile memory endpoint used for edge device-spectrum examples.",
    )
    CortexMClock = sourced_qty(
        48 * ureg.megahertz,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Cortex-M-class clock endpoint",
        description="Representative Cortex-M-class clock endpoint used for device-spectrum prose.",
    )
    MobileClassClock = sourced_qty(
        3 * ureg.gigahertz,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Mobile-class CPU clock endpoint",
        description="Representative mobile-class CPU clock endpoint used for device-spectrum prose.",
    )
    TinyCpuThroughputMips = sourced(
        10,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Tiny CPU throughput endpoint",
        description="Representative MIPS endpoint for tiny Cortex-M-class processors.",
    )
    MobileCpuThroughputMips = sourced(
        100_000,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Mobile CPU throughput endpoint",
        description="Representative MIPS endpoint for high-end mobile-class processors.",
    )
    SensorPowerLow = sourced_qty(
        10 * ureg.microwatt,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Tiny sensor power endpoint",
        description="Lower power endpoint used for edge device-spectrum examples.",
    )
    MicrocontrollerBoardCost = sourced_qty(
        10 * USD,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Microcontroller board cost endpoint",
        description="Representative low-cost microcontroller board price used in hardware-spectrum examples.",
    )
    LowEndEdgeRam = sourced_qty(
        512 * MB,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Low-end edge RAM target",
        description="Representative RAM envelope for low-end edge deployment pitfall examples.",
    )
    LowEndEdgeCompute = sourced_qty(
        1 * (GFLOPs / second),
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Low-end edge compute target",
        description="Representative compute envelope for low-end edge deployment pitfall examples.",
    )
    FlagshipPhonePowerHigh = sourced_qty(
        5 * ureg.watt,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="Flagship phone power endpoint",
        description="Upper mobile power endpoint used for edge device-spectrum examples.",
    )
    IotMicrocontrollerComputeLow = sourced_qty(
        0.03 * TOPS,
        pc.EDGE_DEVICE_SPECTRUM_ANCHORS,
        name="IoT microcontroller compute endpoint",
        description="Representative low-end compute endpoint for federated heterogeneity examples.",
    )


class EdgeAdaptationTierProfile(Registry):
    """Reusable device-tier assumptions for on-device adaptation examples."""

    WearablePersonalizationMemory = sourced_qty(
        500 * MiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Wearable personalization memory",
        description="Reference wearable memory envelope for impossible-full-finetuning examples.",
    )
    TinyWearableRam = sourced_qty(
        1 * MiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Tiny wearable RAM",
        description="Severely constrained wearable/sensor RAM envelope for adaptation-strategy selection.",
    )
    BudgetPhoneRam = sourced_qty(
        4 * GiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Budget phone RAM",
        description="Reference budget smartphone total memory in edge adaptation examples.",
    )
    BudgetPhoneAvailableLow = sourced_qty(
        1 * GiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Budget phone ML-available memory low",
        description="Lower ML-available memory envelope after OS and background-process overhead.",
    )
    BudgetPhoneAvailableHigh = sourced_qty(
        2 * GiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Budget phone ML-available memory high",
        description="Upper ML-available memory envelope after OS and background-process overhead.",
    )
    BudgetDeviceMemoryLimit = sourced_qty(
        1 * GiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Budget device adaptation memory limit",
        description="Reference budget-device memory limit for lightweight personalization examples.",
    )
    BudgetKeyboardPhoneRam = sourced_qty(
        2 * GiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Budget keyboard phone RAM",
        description="Budget-device RAM anchor in the mobile-keyboard heterogeneity example.",
    )
    IotMemoryLow = sourced_qty(
        64 * MiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="IoT memory envelope low",
        description="Lower memory envelope for IoT embedded systems in edge adaptation examples.",
    )
    IotMemoryHigh = sourced_qty(
        1 * GiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="IoT memory envelope high",
        description="Upper memory envelope for IoT embedded systems in edge adaptation examples.",
    )
    KeyboardGradientMemoryLow = sourced_qty(
        50 * MB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Keyboard adaptation gradient memory low",
        description="Lower memory envelope for a compact keyboard-model gradient update.",
    )
    KeyboardGradientMemoryHigh = sourced_qty(
        100 * MB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Keyboard adaptation gradient memory high",
        description="Upper memory envelope for a compact keyboard-model gradient update.",
    )
    KeyboardBackgroundBudgetLow = sourced_qty(
        200 * MB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Keyboard background app budget low",
        description="Lower background-app memory budget for smartphone keyboard adaptation.",
    )
    KeyboardBackgroundBudgetHigh = sourced_qty(
        300 * MB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Keyboard background app budget high",
        description="Upper background-app memory budget for smartphone keyboard adaptation.",
    )
    KeyboardGradientExample = sourced_qty(
        75 * MB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Keyboard adaptation gradient example",
        description="Representative gradient-update memory used for the keyboard memory-share example.",
    )
    KeyboardBackgroundBudgetExample = sourced_qty(
        300 * MB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Keyboard background app budget example",
        description="Representative background-app memory budget used for the keyboard memory-share example.",
    )
    VoiceAssistantFleetDevices = sourced(
        50_000_000,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Voice assistant fleet devices",
        description="Reference fleet size for tiered on-device voice-assistant adaptation.",
    )
    FlagshipFleetShare = sourced(
        0.20,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Flagship device fleet share",
        description="Flagship-device share in the tiered adaptation profile.",
    )
    MidTierFleetShare = sourced(
        0.60,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Mid-tier device fleet share",
        description="Mid-tier device share in the tiered adaptation profile.",
    )
    BudgetFleetShare = sourced(
        0.20,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Budget device fleet share",
        description="Budget-device share in the tiered adaptation profile.",
    )
    FlagshipLoraRank = sourced(
        32,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Flagship LoRA adapter rank",
        description="Reference LoRA rank for flagship-device adaptation.",
    )
    MidTierLoraRank = sourced(
        16,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Mid-tier LoRA adapter rank",
        description="Reference LoRA rank for mid-tier-device adaptation.",
    )
    BudgetReplayBuffer = sourced_qty(
        10 * MiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Budget device replay buffer",
        description="Reference replay-buffer budget for constrained edge devices.",
    )
    FlagshipReplayBuffer = sourced_qty(
        100 * MiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Flagship device replay buffer",
        description="Reference replay-buffer budget for flagship edge devices.",
    )
    FewShotInteractionsLow = sourced(
        5,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Few-shot personalization interactions low",
        description="Lower interaction count for few-shot personalization examples.",
    )
    FewShotInteractionsHigh = sourced(
        10,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Few-shot personalization interactions high",
        description="Upper interaction count for few-shot personalization examples.",
    )
    FederatedLoraUpdate = sourced_qty(
        50 * MiB,
        pc.EDGE_ADAPTATION_TIER_PROFILE,
        name="Federated LoRA adapter update",
        description="Reference LoRA adapter update payload for mobile federated coordination.",
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


class Sensors(Registry):
    """Physical AI and embodied perception reference sensors (Volume IV)."""

    Camera_4K_30Hz_Raw12_FrameSize = sourced_qty(
        12.44 * MB, pc.PHYSICAL_AI_SENSORS,
        name="4K 30Hz 12-bit RAW camera frame size",
        description="Uncompressed frame footprint for 3840x2160 12-bit RAW sensor data.",
    )
    Camera_4K_30Hz_QuadStream_Ingress = sourced_qty(
        1.49 * GB / second, pc.PHYSICAL_AI_SENSORS,
        name="4-camera 4K 30Hz aggregate ingress",
        description="Sustained DRAM ingress traffic generated by four 4K 30Hz 12-bit cameras.",
    )
    LiDAR_128Beam_PointRate = sourced(
        2.4e6, pc.PHYSICAL_AI_SENSORS,
        name="128-beam automotive LiDAR point rate",
        description="Point cloud generation rate for a 128-beam automotive LiDAR (points/sec).",
    )
    LiDAR_128Beam_Ingress = sourced_qty(
        100 * MB / second, pc.PHYSICAL_AI_SENSORS,
        name="128-beam automotive LiDAR ingress rate",
        description="Packetized point cloud ingestion rate over automotive Ethernet.",
    )
    ManipulatorEyeInHand_BurstIngress = sourced_qty(
        600 * MB / second, pc.PHYSICAL_AI_SENSORS,
        name="Manipulator eye-in-hand burst ingress",
        description="Combined burst ingress for dual high-speed eye-in-hand cameras with tactile arrays.",
    )
    ProcessPyrometry_ContinuousIngress = sourced_qty(
        1.2 * GB / second, pc.PHYSICAL_AI_SENSORS,
        name="Coaxial NIR pyrometry continuous ingress",
        description="High-frequency 1000 Hz optical ingress for laser powder-bed additive manufacturing.",
    )


class PhysicalAIRates(Registry):
    """Execution frequency and period hierarchy across the five physical AI machine levels."""

    IntentLo = sourced_qty(1.0 * ureg.Hz, pc.PHYSICAL_AI_NUMBERS,
        name="Intent loop frequency low", description="Lower bound on deliberative intent planning rate.")
    IntentHi = sourced_qty(5.0 * ureg.Hz, pc.PHYSICAL_AI_NUMBERS,
        name="Intent loop frequency high", description="Upper bound on deliberative intent planning rate.")
    ChunkLo = sourced_qty(10.0 * ureg.Hz, pc.PHYSICAL_AI_NUMBERS,
        name="Chunk policy frequency low", description="Lower bound on action chunk trajectory proposal rate.")
    ChunkHi = sourced_qty(50.0 * ureg.Hz, pc.PHYSICAL_AI_NUMBERS,
        name="Chunk policy frequency high", description="Upper bound on action chunk trajectory proposal rate.")
    PermissionRate = sourced_qty(1000.0 * ureg.Hz, pc.PHYSICAL_AI_NUMBERS,
        name="Permission loop frequency", description="Canonical permission loop and runtime safety barrier rate.")
    CurrentLoopLo = sourced_qty(10.0 * ureg.kHz, pc.PHYSICAL_AI_NUMBERS,
        name="Current loop frequency low", description="Lower bound on motor field-oriented current control rate.")
    CurrentLoopHi = sourced_qty(25.0 * ureg.kHz, pc.PHYSICAL_AI_NUMBERS,
        name="Current loop frequency high", description="Upper bound on motor field-oriented current control rate.")
    GateSwitchingLo = sourced_qty(50.0 * ureg.kHz, pc.PHYSICAL_AI_NUMBERS,
        name="Inverter PWM switching frequency low", description="Lower bound on power MOSFET switching frequency.")
    GateSwitchingHi = sourced_qty(200.0 * ureg.kHz, pc.PHYSICAL_AI_NUMBERS,
        name="Inverter PWM switching frequency high", description="Upper bound on power MOSFET switching frequency.")


class PhysicalAIKinematics(Registry):
    """Kinetic translation rates and stopping clearance conversions."""

    BlindTravelRate1mps = sourced_qty(1.0 * (ureg.millimeter / ureg.ms), pc.PHYSICAL_AI_NUMBERS,
        name="Blind travel rate at 1 m/s", description="Kinetic blind travel conversion at 1 m/s: 1 mm per ms.")
    BlindTravelRate10mps = sourced_qty(10.0 * (ureg.millimeter / ureg.ms), pc.PHYSICAL_AI_NUMBERS,
        name="Blind travel rate at 10 m/s", description="Kinetic blind travel conversion at 10 m/s: 10 mm (1 cm) per ms.")
    BlindTravelRate30mps = sourced_qty(30.0 * (ureg.millimeter / ureg.ms), pc.PHYSICAL_AI_NUMBERS,
        name="Blind travel rate at 30 m/s", description="Kinetic blind travel conversion at 30 m/s (108 km/h): 30 mm (3 cm) per ms.")
    IsoApproachSpeed = sourced_qty(1.6 * (ureg.meter / ureg.second), pc.ISO_13855_APPROACH_SPEED,
        name="ISO 13855 walking approach speed", description="Standard human walking approach speed K.")


class PhysicalAIThermal(Registry):
    """Thermal time constants across compute silicon and electromechanical actuators."""

    TauSiliconJunction = sourced_qty(5.0 * ureg.ms, pc.THERMAL_CONSTANTS_ELECTROMECHANICAL,
        name="Silicon junction thermal time constant", description="Adiabatic heating time constant for semiconductor die.")
    TauSiliconPackage = sourced_qty(30.0 * ureg.second, pc.THERMAL_CONSTANTS_ELECTROMECHANICAL,
        name="Silicon heatsink thermal time constant", description="Convective dissipation time constant for SoC heatsinks.")
    TauMotorStator = sourced_qty(120.0 * ureg.second, pc.THERMAL_CONSTANTS_ELECTROMECHANICAL,
        name="Motor stator winding thermal time constant", description="Copper winding resistive Joule heating time constant.")
    TauMotorFrame = sourced_qty(1200.0 * ureg.second, pc.THERMAL_CONSTANTS_ELECTROMECHANICAL,
        name="Motor frame thermal time constant", description="Bulk stator casing and mechanical frame thermal dissipation time constant.")
    TauBatteryPack = sourced_qty(600.0 * ureg.second, pc.THERMAL_CONSTANTS_ELECTROMECHANICAL,
        name="Battery pack thermal time constant", description="Thermal mass time constant for lithium-ion battery cells.")


class PhysicalAISafety(Registry):
    """Statistical bounds, exposure walls, and functional safety failure rate targets."""

    RuleOfThreeMultiplier = sourced(3, pc.RULE_OF_THREE_SAFETY,
        name="Rule of Three Poisson factor", description="Factor 3 for upper 95% Poisson confidence bound with zero events.")
    TargetFailureRateSIL3 = sourced_qty(1e-7 / _hour, pc.IEC_61508_SIL_3,
        name="IEC 61508 SIL 3 / PL e target failure rate", description="Target probability of dangerous failure per hour for SIL 3.")
    TargetFailureRateASILD = sourced_qty(1e-8 / _hour, pc.ISO_26262_ASIL_D,
        name="ISO 26262 ASIL D target failure rate", description="Target probability of dangerous failure per hour (10 FIT) for automotive ASIL D.")
    HumanFatalCrashMiles = sourced(1e8, pc.PHYSICAL_AI_NUMBERS,
        name="Human fatal crash miles", description="Approximate vehicle miles traveled per fatal crash in human driving.")


class PhysicalAIBuses(Registry):
    """Fieldbus and network communication latencies, jitter, and determinism profiles."""

    EtherCatJitter = sourced_qty(1.0 * ureg.microsecond, pc.ETHERCAT_STANDARD,
        name="EtherCAT cycle jitter", description="Worst-case cycle jitter for hardware cut-through EtherCAT fieldbus.")
    EtherCatCycle = sourced_qty(1.0 * ureg.ms, pc.ETHERCAT_STANDARD,
        name="EtherCAT cycle period", description="Canonical EtherCAT real-time servo cycle period.")
    CanFdLatency = sourced_qty(250.0 * ureg.microsecond, pc.CAN_FD_SPECIFICATION,
        name="CAN-FD frame transit latency", description="Nominal frame transit latency on 5 Mbps CAN-FD bus.")
    CanFdJitter = sourced_qty(150.0 * ureg.microsecond, pc.CAN_FD_SPECIFICATION,
        name="CAN-FD arbitration jitter", description="Worst-case priority-inversion and non-preemptive queueing jitter.")
    WifiLatencyNominal = sourced_qty(15.0 * ureg.ms, pc.PHYSICAL_AI_NUMBERS,
        name="Wi-Fi round-trip latency nominal", description="Nominal unloaded Wi-Fi 6 round-trip ping latency.")
    WifiLatencyP99 = sourced_qty(150.0 * ureg.ms, pc.PHYSICAL_AI_NUMBERS,
        name="Wi-Fi round-trip latency P99", description="Tail latency on wireless links under contention or RF fade.")


class PhysicalAICompute(Registry):
    """Edge memory bandwidth and VLA action chunk amortization anchors."""

    EdgeMemoryBandwidthOrin = sourced_qty(204.8 * GB / second, pc.PHYSICAL_AI_NUMBERS,
        name="Jetson AGX Orin peak memory bandwidth", description="Theoretical 256-bit LPDDR5 memory bandwidth.")
    EdgeMemorySustainedOrin = sourced_qty(140.0 * GB / second, pc.PHYSICAL_AI_NUMBERS,
        name="Jetson AGX Orin sustained bandwidth", description="Achievable sustained DRAM streaming bandwidth (~70% efficiency).")
    VlaParameters7B = sourced(7e9, pc.PHYSICAL_AI_NUMBERS,
        name="7B VLA parameter count", description="Parameter count for OpenVLA / Octo class models.")
    VlaChunkHorizon = sourced(16, pc.PHYSICAL_AI_NUMBERS,
        name="Action chunk horizon", description="Typical action chunk sequence length emitted by chunk policies.")


class PhysicalAINumbers(Registry):
    """Authoritative physical AI reference numbers and operational anchors (Volume IV)."""

    Rates = PhysicalAIRates
    Kinematics = PhysicalAIKinematics
    Thermal = PhysicalAIThermal
    Safety = PhysicalAISafety
    Buses = PhysicalAIBuses
    Compute = PhysicalAICompute


class ReferenceStats(Registry):
    """Registry namespace for non-executable real-world scenario statistics."""

    Workloads = Workloads
    AnomalyModel = AnomalyModel
    OuraSleepStudy = OuraSleepStudy
    ClinicalImaging = ClinicalImaging
    EnergyAnchors = EnergyAnchors
    EmissionsAnchors = EmissionsAnchors
    TrainingScaleProfiles = TrainingScaleProfiles
    TrainingCostAnchors = TrainingCostAnchors
    FleetEvolution = FleetEvolution
    StorageTrainingCorpus = StorageTrainingCorpus
    ModelLoading = ModelLoading
    ServingProfiles = ServingProfiles
    CheckpointArchetypes = CheckpointArchetypes
    EdgeDeviceSpectrum = EdgeDeviceSpectrum
    EdgeAdaptationTierProfile = EdgeAdaptationTierProfile
    MobilePower = MobilePower
    PhoneBattery = PhoneBattery
    Sensors = Sensors
    PhysicalAIRates = PhysicalAIRates
    PhysicalAIKinematics = PhysicalAIKinematics
    PhysicalAIThermal = PhysicalAIThermal
    PhysicalAISafety = PhysicalAISafety
    PhysicalAIBuses = PhysicalAIBuses
    PhysicalAICompute = PhysicalAICompute
    PhysicalAI = PhysicalAINumbers

