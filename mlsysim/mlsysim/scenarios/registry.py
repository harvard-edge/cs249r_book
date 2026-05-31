"""Scenarios: real-world case studies and workload reference statistics.

This registry is the home for the book's recurring real-world reference figures —
illustrative scale anchors (Gmail volume, Waymo sensor rate) and case-study model
metrics (the TinyML anomaly detector). Every value carries sourced() provenance.

Note: *evaluatable* scenario bundles (workload + system + SLA, with .evaluate())
live in ``engine/scenarios.py``. This registry holds book-facing scenario
metadata: sourced scalar anchors plus archetype records that bind textbook
labels to canonical Models, Hardware, and Systems objects. Keep model specs in
``models/`` and machine specs in ``hardware/``/``systems/``; this file only
composes them into recurring textbook scenarios.
"""
from pydantic import BaseModel, ConfigDict

from ..core.provenance import sourced, sourced_qty
from ..core.registry import Registry
from ..core import provenance_catalog as pc
from ..core.units import ureg, TB
from ..hardware.registry import Hardware
from ..hardware.types import HardwareNode
from ..models.registry import Models
from ..models.types import Workload
from ..systems.registry import Systems
from ..systems.types import Fleet

_hour = ureg.hour
_joule = ureg.joule


class WorkloadArchetypeSpec(BaseModel):
    """Book-facing workload archetype metadata tied to an actual model object."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    label: str
    workload: Workload
    workload_label: str
    archetype: str
    deployment_paradigm: str
    bottleneck: str
    textbook_role: str
    dominant_drift_pattern: str
    monitoring_metric: str
    retraining_trigger: str


class FleetArchetypeSpec(BaseModel):
    """Volume II fleet archetype metadata tied to canonical workloads/systems."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    label: str
    scale_anchor: Workload
    open_reference: Workload | None = None
    system: HardwareNode | Fleet | None = None
    c3_constraint: str
    fleet_challenge: str
    primary_communication: str
    dominant_friction: str
    optimization_strategy: str
    partitioning_strategy: str
    partitioning_logic: str
    update_cadence: str
    deployment_pattern: str
    primary_risk: str
    rollback_window: str


class ApplicationMissionSpec(BaseModel):
    """Recurring end-to-end book mission with workload and machine anchors."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    mission: str
    system_archetype: str
    workload: Workload
    workload_label: str
    system: HardwareNode | Fleet
    critical_constraint: str


class Workloads(Registry):
    """Illustrative real-world workload scale anchors (order-of-magnitude intuition)."""

    GmailEmailsPerDay = sourced(
        121e9, pc.BOOK_WORKLOAD_SCALE,
        name="Gmail emails per day", description="Approximate daily Gmail volume.")
    GoogleSearchesPerDay = sourced(
        8.5e9, pc.BOOK_WORKLOAD_SCALE,
        name="Google searches per day", description="Approximate daily Google search volume.")
    WaymoDataPerHourLow = sourced_qty(
        1 * TB / _hour, pc.BOOK_WORKLOAD_SCALE,
        name="Waymo sensor data rate (low)", description="Lower-bound AV sensor data generation rate.")
    WaymoDataPerHourHigh = sourced_qty(
        19 * TB / _hour, pc.BOOK_WORKLOAD_SCALE,
        name="Waymo sensor data rate (high)", description="Upper-bound AV sensor data generation rate.")


class AnomalyModel(Registry):
    """TinyML anomaly-detection case study (benchmarking example)."""

    Latency = sourced_qty(
        10.4 * ureg.ms, pc.BOOK_ANOMALY_CASE,
        name="Anomaly model latency", description="Inference latency of the TinyML anomaly detector.")
    Auc = sourced(
        0.86, pc.BOOK_ANOMALY_CASE,
        name="Anomaly model AUC", description="Area under the ROC curve for the TinyML anomaly detector.")
    Energy = sourced_qty(
        516 * ureg.microjoule, pc.BOOK_ANOMALY_CASE,
        name="Anomaly model energy", description="Per-inference energy of the TinyML anomaly detector.")


class EnergyAnchors(Registry):
    """Everyday energy-scale comparison anchors (order-of-magnitude intuition)."""

    SmartphoneCharge = sourced_qty(
        40_000 * _joule, pc.BOOK_ENERGY_ANCHORS,
        name="Smartphone full charge", description="Approximate energy to fully charge a smartphone battery.")
    BoilingWater = sourced_qty(
        100_000 * _joule, pc.BOOK_ENERGY_ANCHORS,
        name="Boiling 1 L of water", description="Approximate energy to bring one liter of water to a boil.")


class MobilePower(Registry):
    """Mobile/edge device + workload power-envelope reference figures."""

    MobileNpuTypical = sourced_qty(3 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile NPU typical power", description="Typical sustained power for on-device mobile inference.")
    MobileNpuPeak = sourced_qty(4 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile NPU peak power", description="Higher-bound mobile inference power envelope.")
    ObjectDetector = sourced_qty(2 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Object-detector power", description="Reference power draw of an always-on object-detection workload.")
    MobileMlSustainedLow = sourced_qty(2 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile ML sustained power (low)", description="Lower bound for sustained smartphone ML processing.")
    MobileMlSustainedHigh = sourced_qty(3 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile ML sustained power (high)", description="Upper bound for sustained smartphone ML processing.")
    MobileMlBurstLow = sourced_qty(5 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile ML burst power (low)", description="Lower bound for brief smartphone ML burst processing.")
    MobileMlBurstHigh = sourced_qty(10 * ureg.watt, pc.BOOK_DEVICE_ANCHORS,
        name="Mobile ML burst power (high)", description="Upper bound for brief smartphone ML burst processing.")
    BackgroundAdaptationLow = sourced_qty(500 * ureg.milliwatt, pc.BOOK_DEVICE_ANCHORS,
        name="Background adaptation power budget (low)", description="Lower bound for background on-device adaptation power.")
    BackgroundAdaptationHigh = sourced_qty(1000 * ureg.milliwatt, pc.BOOK_DEVICE_ANCHORS,
        name="Background adaptation power budget (high)", description="Upper bound for background on-device adaptation power.")


class PhoneBattery(Registry):
    """Flagship smartphone battery reference figures.

    Note: EnergyWh (15 Wh, a flagship pack rating) and EnergyJ (capacity x voltage =
    3000 mAh x 3.7 V = 11.1 Wh) are two DISTINCT battery models the book uses in
    different examples; both are preserved as-is rather than reconciled."""

    CapacityMah = sourced_qty(3000 * ureg.milliampere_hour, pc.BOOK_DEVICE_ANCHORS,
        name="Phone battery capacity", description="Typical flagship smartphone battery charge capacity.")
    VoltageV = sourced_qty(3.7 * ureg.volt, pc.BOOK_DEVICE_ANCHORS,
        name="Phone battery voltage", description="Nominal Li-ion cell voltage.")
    EnergyJ = sourced_qty((3000 * ureg.milliampere_hour * 3.7 * ureg.volt).to(ureg.joule), pc.BOOK_DEVICE_ANCHORS,
        name="Phone battery energy (capacity x voltage)", description="Battery energy derived from capacity x voltage.")
    EnergyWh = sourced_qty(15 * ureg.watt * ureg.hour, pc.BOOK_DEVICE_ANCHORS,
        name="Flagship phone battery energy", description="Modern flagship smartphone battery pack energy rating.")


class WorkloadArchetypes(Registry):
    """Canonical Volume I lighthouse workloads and operational archetypes."""

    ResNet50 = WorkloadArchetypeSpec(
        label="ResNet-50",
        workload=Models.Vision.ResNet50,
        workload_label="ResNet-50",
        archetype="Compute Beast",
        deployment_paradigm="Cloud training, edge inference",
        bottleneck="Compute",
        textbook_role="Parallelism, quantization, and batching",
        dominant_drift_pattern="Visual distribution shift (lighting, camera, new object classes)",
        monitoring_metric="Accuracy on holdout set (ground truth available)",
        retraining_trigger="Accuracy drops > 2 percent from baseline (~monthly for stable domains)",
    )
    GPT2 = WorkloadArchetypeSpec(
        label="GPT-2 XL",
        workload=Models.Language.GPT2,
        workload_label="GPT-2",
        archetype="Bandwidth Hog",
        deployment_paradigm="Cloud inference",
        bottleneck="Mem. Bandwidth",
        textbook_role="Autoregressive generation and KV caching",
        dominant_drift_pattern="Vocabulary drift, topic shift, emerging entities",
        monitoring_metric="Perplexity on live traffic (no ground truth needed)",
        retraining_trigger="Perplexity increases > 10 percent; new vocabulary detected (~weekly for news domains)",
    )
    DLRM = WorkloadArchetypeSpec(
        label="DLRM",
        workload=Models.Recommendation.DLRM,
        workload_label="DLRM",
        archetype="Sparse Scatter",
        deployment_paradigm="Cloud only (distributed)",
        bottleneck="Mem. Capacity",
        textbook_role="Embedding tables and scale-out systems",
        dominant_drift_pattern="User behavior shift, item catalog churn, cold-start items",
        monitoring_metric="CTR/CVR delta vs. historical cohorts",
        retraining_trigger="Engagement drops > 5 percent; catalog refresh (~daily for e-commerce)",
    )
    MobileNetV2 = WorkloadArchetypeSpec(
        label="MobileNetV2",
        workload=Models.Vision.MobileNetV2,
        workload_label="MobileNet",
        archetype="Compute Beast (efficient)",
        deployment_paradigm="Mobile, edge",
        bottleneck="Latency",
        textbook_role="Depthwise convolutions and efficiency",
        dominant_drift_pattern="Visual distribution shift on-device",
        monitoring_metric="Latency, thermal headroom, and accuracy on sampled holdout data",
        retraining_trigger="Accuracy or latency regression after app/model update",
    )
    KWS = WorkloadArchetypeSpec(
        label="Keyword Spotting (KWS)",
        workload=Models.Tiny.DS_CNN,
        workload_label="DS-CNN",
        archetype="Tiny Constraint",
        deployment_paradigm="TinyML, always-on",
        bottleneck="Power",
        textbook_role="Extreme quantization and always-on ops",
        dominant_drift_pattern="Acoustic environment change (noise floor shift)",
        monitoring_metric="Duty cycle (wakeups/hour) + false positive rate",
        retraining_trigger="False wake rate > 1 percent; battery drain exceeds spec (~quarterly OTA update)",
    )


class FleetArchetypes(Registry):
    """Canonical Volume II scale archetypes used across fleet chapters."""

    ArchetypeA = FleetArchetypeSpec(
        label="Archetype A (GPT-4/Llama-3)",
        scale_anchor=Models.Language.GPT4,
        open_reference=Models.Language.Llama3_70B,
        system=Systems.Clusters.Frontier_8K,
        c3_constraint="Communication (fleet-wide)",
        fleet_challenge=(
            "Partition hundreds of billions of parameters, and potentially larger "
            "proprietary models, across thousands to tens of thousands of GPUs using "
            "3D Parallelism without the network becoming the bottleneck."
        ),
        primary_communication="AllReduce",
        dominant_friction=r"Bandwidth ($\beta$)",
        optimization_strategy="Hierarchical AllReduce; Rail-optimization",
        partitioning_strategy="Hybrid 3D Parallelism",
        partitioning_logic=(
            "Combine Tensor (width), Pipeline (depth), and Data (throughput) "
            "to fit frontier-scale dense weights."
        ),
        update_cadence="Monthly to quarterly",
        deployment_pattern="Staged, careful",
        primary_risk="Quality regression, safety",
        rollback_window="Hours to days",
    )
    ArchetypeB = FleetArchetypeSpec(
        label="Archetype B (DLRM at Scale)",
        scale_anchor=Models.Recommendation.DLRM,
        system=Systems.Clusters.Production_2K,
        c3_constraint="Coordination (routing and placement)",
        fleet_challenge=(
            "Shard 10 TB+ embedding tables across hundreds of nodes; process "
            "millions of QPS with sub-100 ms tail latency while managing sparse "
            "feature routing, shard placement, and O(N^2) all-to-all contention."
        ),
        primary_communication="AllToAll",
        dominant_friction=r"Latency ($\alpha$) & Contention",
        optimization_strategy="Topology-aware routing; token load-balancing",
        partitioning_strategy="Embedding Sharding",
        partitioning_logic=(
            "Partition massive embedding tables across a Parameter Server fleet; "
            "use sparse AllToAll updates."
        ),
        update_cadence="Daily to weekly",
        deployment_pattern="Shadow, interleaving",
        primary_risk="Engagement drop",
        rollback_window="Minutes",
    )
    ArchetypeC = FleetArchetypeSpec(
        label="Archetype C (Federated MobileNet)",
        scale_anchor=Models.Vision.MobileNetV2,
        system=Hardware.Mobile.iPhone15Pro,
        c3_constraint="Compute (per-device envelope)",
        fleet_challenge=(
            "Coordinate learning across millions of compute-constrained, unreliable "
            "edge devices using Federated updates; raw data cannot leave the device."
        ),
        primary_communication="P2P/Async",
        dominant_friction="Connectivity & Latency",
        optimization_strategy="Aggressive quantization; Error Feedback",
        partitioning_strategy="Federated Learning",
        partitioning_logic=(
            "Keep raw data on edge devices while coordinating quantized local "
            "updates across unreliable clients."
        ),
        update_cadence="Weekly to monthly",
        deployment_pattern="Staged OTA",
        primary_risk="Convergence and privacy regression",
        rollback_window="Staged OTA",
    )


class ApplicationMissions(Registry):
    """Recurring engineering missions from the Volume I systems hierarchy."""

    FrontierTraining = ApplicationMissionSpec(
        mission="Frontier Training",
        system_archetype="Cloud Cluster",
        workload=Models.Language.GPT4,
        workload_label="GPT-4",
        system=Systems.Clusters.Frontier_8K,
        critical_constraint="Target: 500 ms/step",
    )
    AutonomousPerception = ApplicationMissionSpec(
        mission="Autonomous Perception",
        system_archetype="Edge Robotics",
        workload=Models.Vision.YOLOv8_Nano,
        workload_label="YOLOv8-nano",
        system=Hardware.Edge.JetsonOrinNX,
        critical_constraint="SLA: 10 ms latency",
    )
    MobileAssistant = ApplicationMissionSpec(
        mission="Mobile Assistant",
        system_archetype="Smartphone",
        workload=Models.Language.Llama3_8B,
        workload_label="Quantized small LLM",
        system=Hardware.Mobile.iPhone15Pro,
        critical_constraint="Thermal Throttling/RAM",
    )
    SmartDoorbell = ApplicationMissionSpec(
        mission="Smart Doorbell",
        system_archetype="TinyML (MCU)",
        workload=Models.Tiny.WakeVision,
        workload_label="Wake Vision",
        system=Hardware.Tiny.ESP32_S3,
        critical_constraint="Power: 100 mW",
    )


class Archetypes(Registry):
    """Book-facing archetype taxonomy and recurring missions."""

    Workload = WorkloadArchetypes
    Fleet = FleetArchetypes
    Missions = ApplicationMissions


class Scenarios(Registry):
    """Registry namespace for real-world case studies and workload statistics."""

    Workloads = Workloads
    AnomalyModel = AnomalyModel
    EnergyAnchors = EnergyAnchors
    MobilePower = MobilePower
    PhoneBattery = PhoneBattery
    Archetypes = Archetypes
