"""Named base scenarios for applications.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Clusters as _Clusters
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ...scenario_definitions.base import Scenario


# --- TINYML WORLD ---
SmartDoorbell = Scenario(
    name="Smart Doorbell",
    description="Identifying humans at the door using a sub-watt microcontroller.",
    application="Smart Doorbell",
    workload=_Models.Tiny.WakeVision,
    system=_Hardware.Tiny.ESP32_S3,
    sla_latency=Q_("200 ms"),
    power_budget=Q_("100 mW"),
)


TinySensor = Scenario(
    name="Anomaly Sensor",
    description="Low-power vibration monitoring for industrial predictive maintenance.",
    application="Industrial Anomaly Detection",
    workload=_Models.Tiny.AnomalyDetector,
    system=_Hardware.Tiny.ESP32_S3,
    sla_latency=Q_("10 ms"),
    power_budget=Q_("50 mW"),
)


# --- EDGE WORLD ---
AutonomousVehicle = Scenario(
    name="Autonomous Vehicle",
    description="Real-time object detection for safe urban navigation.",
    application="Autonomous Vehicle",
    workload=_Models.Vision.YOLOv8_Nano,
    system=_Hardware.Edge.JetsonOrinNX,
    sla_latency=Q_("10 ms"),
)


AutonomousVehicle_Waymo = Scenario(
    name="Waymo AV Data Pipeline",
    description="High-throughput data ingestion for autonomous fleet training.",
    application="Autonomous Vehicle",
    workload=_Models.Vision.YOLOv8_Nano.model_copy(
        update={
            "name": "Waymo (High)",
            "data_rate": _ReferenceStats.Workloads.WaymoDataPerHourHigh,
        }
    ),
    system=_Hardware.Edge.JetsonOrinNX,
    sla_latency=Q_("10 ms"),
)


# --- MOBILE WORLD ---
MobileHealth = Scenario(
    name="Mobile Health",
    description="On-device medical image analysis for remote diagnostics.",
    application="Mobile Health",
    workload=_Models.Vision.MobileNetV2,
    system=_Hardware.Mobile.iPhone15Pro,
    sla_latency=Q_("30 ms"),
)


# Added 2026-09-28. Federated on-device vision (the book's "Archetype C"):
# MobileNet trained/served locally on smartphone-class devices. The
# sustained power budget is the registry's 3 W upper sustained anchor; the
# burst ceiling stays at ReferenceStats.MobilePower.MobileMlBurstHigh.
FederatedMobileVision = Scenario(
    name="Federated Mobile Vision",
    description="On-device federated training and inference of a MobileNet on smartphone-class hardware.",
    application="Federated mobile learning",
    workload=_Models.Vision.MobileNetV2,
    system=_Hardware.Mobile.iPhone15Pro,
    power_budget=_ReferenceStats.MobilePower.MobileMlSustainedHigh,
)


MobileAssistant = Scenario(
    name="Mobile Assistant",
    description="On-device assistant with a quantized small LLM on a smartphone.",
    application="Mobile Assistant",
    workload=_Models.Language.Llama3_8B,
    system=_Hardware.Mobile.iPhone15Pro,
    sla_latency=Q_("100 ms"),
    power_budget=Q_("5 W"),
)


# --- WORKSTATION WORLD ---
LocalTraining = Scenario(
    name="Local LLM Fine-tuning",
    description="Fine-tuning a Llama-3 model on a high-end student workstation.",
    application="Local Fine-tuning",
    workload=_Models.Language.Llama3_8B,
    system=_Hardware.Workstation.MacBookM3Max,
    sla_latency=Q_("100 ms"),
)


# --- CLOUD WORLD ---
# Teaching mission, H100 era. Frontier_8K (8,192 H100, ~26K A100-equivalents
# of dense BF16 peak) is the H100 restatement of the GPT-4-class run;
# SemiAnalysis (2023) estimates the same pre-training at ~8,192 H100 for
# ~55 days. The historical A100 fleet lives on
# Models.Language.GPT4.training_*_ref.
FrontierTraining = Scenario(
    name="Frontier LLM Training",
    description="Pre-training a GPT-4-class frontier model on an 8,192-GPU H100 fleet.",
    application="Frontier Training",
    workload=_Models.Language.GPT4,
    system=_Clusters.Frontier_8K,
    sla_latency=Q_("500 ms"),  # Per-step target
)


# Added 2026-09-28. The dense-LLM pretraining reference (the book's
# "Archetype A"): GPT-3-class 175B pretrained with mixed-precision Adam on
# the 1,024-GPU reference cluster (128 DGX H100 nodes). No SLA: it is a
# composition anchor. Byte policy: ReferenceStats.MixedPrecisionAdam.
# NOTE: Training_1K.fabric is InfiniBand_HDR (vol1 reads it; unchanged),
# while the book describes the reference cluster on NDR.
DenseLLMPretrainingReference = Scenario(
    name="Dense LLM Pretraining (175B, 1,024 H100)",
    description="GPT-3-class dense transformer pretrained with mixed-precision Adam on the 1,024-GPU reference cluster.",
    application="Dense LLM pretraining",
    workload=_Models.Language.GPT3,
    system=_Clusters.Training_1K,
)


# --- SERVING WORLD ---
ChatbotServing = Scenario(
    name="Chatbot Serving",
    description="Serving a Llama-3 8B chatbot on a single H100 with latency SLA.",
    application="Chatbot Serving",
    workload=_Models.Language.Llama3_8B,
    system=_Hardware.Cloud.H100,
    sla_latency=Q_("500 ms"),  # TTFT target
)


# --- TINYML: KEYWORD SPOTTING ---
KeywordSpotting = Scenario(
    name="Keyword Spotting",
    description="Always-on wake-word detection on a microcontroller (MLPerf Tiny benchmark).",
    application="Keyword Spotting",
    workload=_Models.Tiny.DS_CNN,
    system=_Hardware.Tiny.ESP32_S3,
    sla_latency=Q_("30 ms"),
    power_budget=Q_("1 mW"),
)
