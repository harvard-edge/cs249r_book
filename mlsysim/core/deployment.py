# book/quarto/mlsysim/deployment.py
# Hierarchical Deployment Tier Definitions for MLSys Textbook

from dataclasses import dataclass
from ..core.constants import (
    ureg, Q_,
    SMARTPHONE_RAM_GB, MCU_RAM_KIB, CLOUD_MEM_GIB,
    TINY_MEM_KIB
)

@dataclass(frozen=True)
class DeploymentTier:
    name: str
    ram: Q_
    storage: Q_
    typical_latency_budget: Q_

class Tiers:
    Cloud = DeploymentTier(
        name="Cloud",
        ram=512 * ureg.GB,
        storage=10 * ureg.TB,
        typical_latency_budget=200 * ureg.ms
    )
    Edge = DeploymentTier(
        name="Edge",
        ram=32 * ureg.GB,
        storage=1 * ureg.TB,
        typical_latency_budget=50 * ureg.ms
    )
    Mobile = DeploymentTier(
        name="Mobile",
        ram=SMARTPHONE_RAM_GB,
        storage=256 * ureg.GB,
        typical_latency_budget=30 * ureg.ms
    )
    Tiny = DeploymentTier(
        name="TinyML",
        ram=MCU_RAM_KIB,
        storage=4 * ureg.MB,
        typical_latency_budget=100 * ureg.ms
    )
