# ledger.py
"""
MLSys Scorecard Module
======================
The multi-dimensional 'Scorecard' for MLSys simulations.
It tracks metrics across four primary engineering axes: 
Performance, Sustainability, Economics, and Reliability.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
from ..core.constants import ureg, Q_

@dataclass(frozen=True)
class PerformanceMetrics:
    """🚀 Performance: Speed and Utilization metrics."""
    latency: Q_
    throughput: Q_
    mfu: float
    hfu: float
    bottleneck: str

@dataclass(frozen=True)
class SustainabilityMetrics:
    """🌍 Sustainability: Environmental impact and resource efficiency."""
    energy: Q_
    carbon_kg: float
    pue: float
    water_liters: float

@dataclass(frozen=True)
class EconomicMetrics:
    """💰 Economics: Total Cost of Ownership (TCO) and unit economics."""
    capex: float
    opex: float
    tco: float
    cost_per_million: float

@dataclass(frozen=True)
class ReliabilityMetrics:
    """🛡️ Reliability: Resilience, uptime, and recovery metrics."""
    mttf: Q_
    goodput: float
    recovery_time: Q_

@dataclass(frozen=True)
class SystemLedger:
    """
    The Universal Scorecard for all MLSys simulation results.
    Binds the four dimensions into a single immutable result object.
    """
    performance: PerformanceMetrics
    sustainability: SustainabilityMetrics
    economics: EconomicMetrics
    reliability: ReliabilityMetrics
    
    mission_name: str
    track_name: str
    choice_summary: str
    
    def validate(self) -> None:
        """Ensures physical invariants are maintained."""
        assert 0 <= self.performance.mfu <= 1.0, f"MFU {self.performance.mfu} must be between 0 and 1"
        assert self.performance.latency.m >= 0, "Latency cannot be negative"
        assert self.sustainability.carbon_kg >= 0, "Carbon footprint cannot be negative"
        
    def to_dict(self) -> Dict[str, Any]:
        """Flattens the ledger into a simple dictionary for JSON/UI consumption."""
        return {
            "mission": self.mission_name,
            "track": self.track_name,
            "choice": self.choice_summary,
            "latency_ms": self.performance.latency.m_as("ms"),
            "throughput_sps": self.performance.throughput.m_as("1/second"),
            "mfu_pct": self.performance.mfu * 100,
            "carbon_kg": self.sustainability.carbon_kg,
            "tco_usd": self.economics.tco,
            "goodput_pct": self.reliability.goodput * 100
        }

    def to_df(self) -> pd.DataFrame:
        """Converts the metrics to a single-row Pandas DataFrame for easy plotting."""
        return pd.DataFrame([self.to_dict()])
