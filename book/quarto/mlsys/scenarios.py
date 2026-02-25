# scenarios.py
# High-level Application Scenarios for MLSys Textbook
# Ties Models and System Archetypes into concrete "User Missions".

from dataclasses import dataclass
from typing import Optional
from .models import ModelSpec, Models
from .systems import SystemArchetype, Systems, Archetypes
from .constants import ureg, Q_

@dataclass(frozen=True)
class ApplicationScenario:
    name: str
    system: SystemArchetype
    model: ModelSpec
    mission_goal: str
    critical_constraint: str
    latency_slo: Optional[Q_] = None
    accuracy_target: Optional[float] = None
    
    @property
    def hardware(self):
        return self.system.hardware

class Scenarios:
    # --- CLOUD MISSION ---
    FrontierTraining = ApplicationScenario(
        name="Frontier Model Training",
        system=Systems.Cloud, # H100 Cluster
        model=Models.GPT4,
        mission_goal="Push the boundary of general intelligence.",
        critical_constraint="Total Cost of Ownership (TCO) and Convergence Stability.",
        accuracy_target=0.99
    )

    # --- EDGE MISSION ---
    AutonomousVehicle = ApplicationScenario(
        name="Autonomous Vehicle Perception",
        system=Systems.Edge, # Jetson Orin
        model=Models.Vision.YOLOv8_Nano,
        mission_goal="Enable safe, real-time navigation in urban environments.",
        critical_constraint="End-to-end Latency (< 10 ms) and Safety Certification.",
        latency_slo=10 * ureg.ms,
        accuracy_target=0.95
    )

    # --- MOBILE MISSION ---
    OnDeviceAssistant = ApplicationScenario(
        name="On-Device Language Assistant",
        system=Systems.Mobile, # Smartphone
        model=Models.Language.Llama2_70B, # Note: highly compressed version
        mission_goal="Provide private, offline conversational AI.",
        critical_constraint="Thermal Throttling and Memory Fragmentation.",
        latency_slo=50 * ureg.ms,
        accuracy_target=0.90
    )

    # --- TINYML MISSION (The "Lighthouse" for Labs) ---
    SmartDoorbell = ApplicationScenario(
        name="Smart Doorbell (Wake Vision)",
        system=Systems.Tiny, # ESP32-CAM
        model=Models.Tiny.WakeVision,
        mission_goal="Identify humans at the door to trigger high-power alerts.",
        critical_constraint="Battery Life (> 1 year) and KB-scale SRAM limits.",
        latency_slo=200 * ureg.ms,
        accuracy_target=0.85
    )

class Applications:
    Frontier = Scenarios.FrontierTraining
    AutoDrive = Scenarios.AutonomousVehicle
    Assistant = Scenarios.OnDeviceAssistant
    Doorbell = Scenarios.SmartDoorbell
