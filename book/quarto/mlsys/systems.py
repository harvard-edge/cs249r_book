# systems.py
# System Archetypes for MLSys Textbook
# Ties Hardware, Tier, and Environment into a single "Environment Context".

from dataclasses import dataclass
from .hardware import HardwareSpec, Hardware
from .deployment import DeploymentTier, Tiers
from .constants import ureg, Q_

@dataclass(frozen=True)
class SystemArchetype:
    name: str
    hardware: HardwareSpec
    tier: DeploymentTier
    network_bw: Q_
    power_budget: Q_
    
    @property
    def ram(self):
        return self.hardware.memory_capacity
    
    @property
    def peak_flops(self):
        return self.hardware.peak_flops
    
    @property
    def memory_bw(self):
        return self.hardware.memory_bw

class Archetypes:
    # --- CLOUD LAYER ---
    Cloud_H100 = SystemArchetype(
        name="Cloud (H100 Node)",
        hardware=Hardware.H100,
        tier=Tiers.Cloud,
        network_bw=400 * ureg.Gbps, # NDR InfiniBand
        power_budget=700 * ureg.watt
    )
    
    Cloud_A100 = SystemArchetype(
        name="Cloud (A100 Node)",
        hardware=Hardware.A100,
        tier=Tiers.Cloud,
        network_bw=200 * ureg.Gbps, # HDR InfiniBand
        power_budget=400 * ureg.watt
    )

    Cloud_V100 = SystemArchetype(
        name="Cloud (V100 Node)",
        hardware=Hardware.V100,
        tier=Tiers.Cloud,
        network_bw=100 * ureg.Gbps, # EDR InfiniBand
        power_budget=300 * ureg.watt
    )

    # --- EDGE LAYER ---
    Edge_Server = SystemArchetype(
        name="Edge Server",
        hardware=Hardware.Edge.GenericServer,
        tier=Tiers.Edge,
        network_bw=10 * ureg.Gbps,
        power_budget=300 * ureg.watt
    )
    
    Edge_Robotics = SystemArchetype(
        name="Edge (Jetson Orin)",
        hardware=Hardware.Edge.JetsonOrinNX,
        tier=Tiers.Edge,
        network_bw=1 * ureg.Gbps,
        power_budget=25 * ureg.watt
    )

    # --- MOBILE LAYER ---
    Mobile_Phone = SystemArchetype(
        name="Mobile (Smartphone)",
        hardware=Hardware.Edge.Generic_Phone,
        tier=Tiers.Mobile,
        network_bw=100 * ureg.Mbps,
        power_budget=5 * ureg.watt
    )

    # --- TINYML LAYER ---
    TinyML_MCU = SystemArchetype(
        name="TinyML (ESP32)",
        hardware=Hardware.Tiny.ESP32,
        tier=Tiers.Tiny,
        network_bw=1 * ureg.Mbps,
        power_budget=0.5 * ureg.watt
    )
    
    TinyML_M7 = SystemArchetype(
        name="TinyML (Cortex-M7)",
        hardware=Hardware.Tiny.Generic_MCU,
        tier=Tiers.Tiny,
        network_bw=1 * ureg.Mbps,
        power_budget=0.1 * ureg.watt
    )

class Systems:
    Cloud = Archetypes.Cloud_H100
    Edge = Archetypes.Edge_Robotics
    Mobile = Archetypes.Mobile_Phone
    Tiny = Archetypes.TinyML_MCU
