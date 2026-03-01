# scenarios.py
# Application and Fleet Scenarios for MLSys Textbook
# Ties Models + Systems/Clusters into concrete named missions.
#
# Two scenario types mirror the two-volume scope:
#
#   ApplicationScenario  — single-machine deployment (Vol1)
#     system: SystemArchetype  (one node, 1–8 GPUs)
#     Exposes: .hardware, .tier, .latency_slo, .accuracy_target
#
#   ClusterScenario      — multi-machine distributed workload (Vol2)
#     cluster: ClusterSpec  (N nodes over a fabric)
#     Exposes: .hardware (lead accelerator), .cluster, .latency_slo
#
# Both share the same .name / .mission_goal / .critical_constraint
# interface so LEGO blocks work identically across volumes.

from dataclasses import dataclass
from typing import Optional
from .models import ModelSpec, Models
from .systems import SystemArchetype, Systems, Archetypes
from .clusters import ClusterSpec, Clusters
from .constants import ureg, Q_


# ─────────────────────────────────────────────────────────────────────────────
# ApplicationScenario — Vol1: single-machine deployment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ApplicationScenario:
    """
    A single-machine ML deployment scenario (Vol1 scope).
    Binds a SystemArchetype to a ModelSpec with a mission description.
    """
    name: str
    system: SystemArchetype
    model: ModelSpec
    mission_goal: str
    critical_constraint: str
    latency_slo: Optional[Q_] = None
    accuracy_target: Optional[float] = None

    @property
    def hardware(self):
        """The underlying accelerator spec (for direct hardware access)."""
        return self.system.hardware

    @property
    def tier(self):
        """The deployment tier (Cloud / Edge / Mobile / Tiny)."""
        return self.system.tier

    def __repr__(self):
        return f"Scenario({self.name})"


# ─────────────────────────────────────────────────────────────────────────────
# ClusterScenario — Vol2: multi-machine distributed workload
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClusterScenario:
    """
    A distributed ML workload scenario (Vol2 scope).
    Binds a ClusterSpec to a ModelSpec with a mission description.

    .hardware  — lead accelerator (same interface as ApplicationScenario)
    .cluster   — full ClusterSpec (nodes, fabric, efficiency)
    """
    name: str
    cluster: ClusterSpec
    model: ModelSpec
    mission_goal: str
    critical_constraint: str
    latency_slo: Optional[Q_] = None
    accuracy_target: Optional[float] = None

    @property
    def hardware(self):
        """Lead accelerator spec (consistent interface with ApplicationScenario)."""
        return self.cluster.node.accelerator

    @property
    def total_gpus(self) -> int:
        return self.cluster.total_gpus

    def __repr__(self):
        return f"ClusterScenario({self.name}, {self.total_gpus} GPUs)"


# ─────────────────────────────────────────────────────────────────────────────
# Vol1 Scenarios — four single-machine "Lighthouse" missions
# ─────────────────────────────────────────────────────────────────────────────

class Scenarios:
    """
    Named single-machine application scenarios (Vol1).

    The four Lighthouse missions span the full deployment spectrum:
      Cloud   → FrontierTraining  (H100, GPT-4, TCO/convergence)
      Edge    → AutonomousVehicle (Jetson Orin, YOLOv8, <10ms latency)
      Mobile  → OnDeviceAssistant (Smartphone, Llama-2-70B compressed)
      Tiny    → SmartDoorbell     (ESP32-CAM, WakeVision, battery life)
      Tiny    → KeywordSpotting   (Cortex-M7, DS-CNN, always-on μW budget)
    """

    # --- CLOUD: Frontier Training ---
    # Single-node proxy; use FleetScenarios.LargeScaleTraining for cluster scope
    FrontierTraining = ApplicationScenario(
        name="Frontier Model Training (Single Node)",
        system=Systems.Cloud,               # H100 SXM
        model=Models.GPT4,
        mission_goal="Push the boundary of general intelligence.",
        critical_constraint="Total Cost of Ownership (TCO) and Convergence Stability.",
        accuracy_target=0.99,
    )

    # --- EDGE: Autonomous Vehicle Perception ---
    AutonomousVehicle = ApplicationScenario(
        name="Autonomous Vehicle Perception",
        system=Systems.Edge,                # Jetson Orin NX
        model=Models.Vision.YOLOv8_Nano,
        mission_goal="Enable safe, real-time navigation in urban environments.",
        critical_constraint="End-to-end Latency (< 10 ms) and Safety Certification.",
        latency_slo=10 * ureg.ms,
        accuracy_target=0.95,
    )

    # --- MOBILE: On-Device Language Assistant ---
    OnDeviceAssistant = ApplicationScenario(
        name="On-Device Language Assistant",
        system=Systems.Mobile,              # Flagship smartphone
        model=Models.Language.Llama2_70B,  # Highly compressed at inference
        mission_goal="Provide private, offline conversational AI.",
        critical_constraint="Thermal Throttling and Memory Fragmentation.",
        latency_slo=50 * ureg.ms,
        accuracy_target=0.90,
    )

    # --- TINYML: Smart Doorbell (Vision) ---
    # Primary TinyML Lighthouse used across Vol1 labs and data chapters.
    SmartDoorbell = ApplicationScenario(
        name="Smart Doorbell (Wake Vision)",
        system=Systems.Tiny,                # ESP32-CAM
        model=Models.Tiny.WakeVision,
        mission_goal="Identify humans at the door to trigger high-power alerts.",
        critical_constraint="Battery Life (> 1 year) and KB-scale SRAM limits.",
        latency_slo=200 * ureg.ms,
        accuracy_target=0.85,
    )

    # --- TINYML: Keyword Spotting (Audio) ---
    # Always-on microphone wake-word detection; complementary Tiny Lighthouse.
    KeywordSpotting = ApplicationScenario(
        name="Keyword Spotting (Always-On Wake Word)",
        system=Archetypes.TinyML_M7,        # Cortex-M7 MCU
        model=Models.Tiny.DS_CNN,
        mission_goal="Detect wake words continuously on a μW power budget.",
        critical_constraint="Always-on Power (< 1 mW) and sub-100ms response.",
        latency_slo=100 * ureg.ms,
        accuracy_target=0.92,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vol2 Fleet Scenarios — distributed multi-machine workloads
# ─────────────────────────────────────────────────────────────────────────────

class FleetScenarios:
    """
    Named distributed workload scenarios (Vol2).

    Each binds a ClusterSpec to a model and a mission.

    Research     → ResearchTraining      (256 GPUs, GPT-3-scale fine-tuning)
    Production   → LargeScaleTraining    (8 192 GPUs, Llama-2-70B pre-training)
    Mega         → FrontierTraining      (100 000 GPUs, GPT-4-scale pre-training)
    Distributed  → DistributedInference  (2 048 GPUs, LLM serving fleet)
    """

    # --- RESEARCH: Fine-tuning / mid-scale pre-training ---
    ResearchTraining = ClusterScenario(
        name="Research Cluster Training (256 GPUs)",
        cluster=Clusters.Research_256,
        model=Models.GPT3,
        mission_goal="Fine-tune or pre-train a GPT-3-class model for research.",
        critical_constraint="Job Turnaround Time and Cluster Utilization.",
        accuracy_target=0.95,
    )

    # --- PRODUCTION: Large-scale pre-training ---
    # The canonical Vol2 running example: Llama-2-70B on 8K H100s.
    LargeScaleTraining = ClusterScenario(
        name="Large-Scale Pre-Training (8 192 GPUs)",
        cluster=Clusters.Frontier_8K,
        model=Models.Language.Llama2_70B,
        mission_goal="Pre-train a 70B parameter foundation model end-to-end.",
        critical_constraint="Fault Tolerance, Communication Overhead, and MFU.",
        accuracy_target=0.95,
    )

    # --- MEGA: Frontier model training ---
    # GPT-4-scale; used in reliability and fleet orchestration chapters.
    FrontierTraining = ClusterScenario(
        name="Frontier Model Training (100 000 GPUs)",
        cluster=Clusters.Mega_100K,
        model=Models.GPT4,
        mission_goal="Train a frontier general-intelligence model.",
        critical_constraint="Continuous Failure Recovery and TCO at Mega-Scale.",
        accuracy_target=0.99,
    )

    # --- DISTRIBUTED INFERENCE: LLM serving fleet ---
    # Used in inference chapter; 2K GPUs serving concurrent user requests.
    DistributedInference = ClusterScenario(
        name="Distributed LLM Inference Fleet (2 048 GPUs)",
        cluster=Clusters.Production_2K,
        model=Models.Language.Llama2_70B,
        mission_goal="Serve a 70B LLM to thousands of concurrent users globally.",
        critical_constraint="P99 Latency SLO (< 200 ms TTFT) and Cost per Token.",
        latency_slo=200 * ureg.ms,
        accuracy_target=0.90,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience aliases — what chapters actually import
# ─────────────────────────────────────────────────────────────────────────────

class Applications:
    """Short aliases for Vol1 single-machine scenarios."""
    Frontier  = Scenarios.FrontierTraining
    AutoDrive = Scenarios.AutonomousVehicle
    Assistant = Scenarios.OnDeviceAssistant
    Doorbell  = Scenarios.SmartDoorbell
    KWS       = Scenarios.KeywordSpotting


class Fleet:
    """Short aliases for Vol2 distributed scenarios."""
    Research  = FleetScenarios.ResearchTraining
    Training  = FleetScenarios.LargeScaleTraining
    Frontier  = FleetScenarios.FrontierTraining
    Inference = FleetScenarios.DistributedInference
