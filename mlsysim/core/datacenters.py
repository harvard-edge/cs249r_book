# datacenters.py
# Regional Grid and Datacenter Profiles for MLSys Textbook (Volume II)
# Lifts the sustainability constants from constants.py into named, typed objects.
#
# The reusable entity is the *regional grid profile* — carbon intensity, PUE,
# and WUE — because Vol2 chapters compare Quebec vs. Poland vs. US, not
# specific buildings.
#
# Usage in QMD cells:
#   from mlsysim import Datacenters
#   ci = Datacenters.Quebec.carbon_intensity_kg_kwh   # 0.02 kg/kWh
#   pue = Datacenters.HyperscaleAir.pue               # 1.12

from dataclasses import dataclass
from typing import Optional
from .constants import (
    ureg, Q_,
    PUE_LIQUID_COOLED, PUE_BEST_AIR, PUE_TYPICAL, PUE_LEGACY,
    WUE_AIR_COOLED, WUE_EVAPORATIVE, WUE_LIQUID,
    CARBON_US_AVG_GCO2_KWH, CARBON_EU_AVG_GCO2_KWH,
    CARBON_QUEBEC_GCO2_KWH, CARBON_FRANCE_GCO2_KWH,
    CARBON_POLAND_GCO2_KWH, CARBON_NORWAY_GCO2_KWH,
    RACK_POWER_TRADITIONAL_KW, RACK_POWER_AI_TYPICAL_KW, RACK_POWER_AI_HIGH_KW,
    AIR_COOLING_LIMIT_KW,
)


# ─────────────────────────────────────────────────────────────────────────────
# GridProfile: one regional electricity grid
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GridProfile:
    """
    A regional electricity grid characterized by carbon intensity and
    the cooling infrastructure typical for that location.

    carbon_intensity_g_kwh  — operational gCO₂ per kWh (IEA 2023)
    pue                     — Power Usage Effectiveness (facility / IT power)
    wue                     — Water Usage Effectiveness (liters per kWh)
    primary_source          — dominant generation type (informational)
    """
    name: str
    carbon_intensity_g_kwh: float   # gCO₂/kWh
    pue: float
    wue: float
    primary_source: str             # e.g. "hydro", "nuclear", "coal", "mixed"

    # ── Derived convenience properties ────────────────────────────────────

    @property
    def carbon_intensity_kg_kwh(self) -> float:
        """Carbon intensity in kg CO₂/kWh (the unit most prose uses)."""
        return self.carbon_intensity_g_kwh / 1000.0

    def carbon_kg(self, energy_kwh: float) -> float:
        """
        Operational CO₂ emissions in kg for a given energy draw.

        Args:
            energy_kwh: Energy consumed by IT equipment (kWh)

        Returns:
            kg CO₂ (includes PUE overhead automatically)
        """
        facility_kwh = energy_kwh * self.pue
        return facility_kwh * self.carbon_intensity_kg_kwh

    def carbon_tonnes(self, energy_kwh: float) -> float:
        """Same as carbon_kg() but in metric tonnes (÷ 1000)."""
        return self.carbon_kg(energy_kwh) / 1000.0

    def intensity_ratio_vs(self, other: "GridProfile") -> float:
        """
        How many times more carbon-intensive this grid is vs. another.
        Useful for 'Poland is Nx worse than Quebec' prose comparisons.
        """
        return self.carbon_intensity_g_kwh / other.carbon_intensity_g_kwh

    def __repr__(self):
        return f"Grid({self.name}, {self.carbon_intensity_g_kwh} gCO₂/kWh, PUE {self.pue})"


# ─────────────────────────────────────────────────────────────────────────────
# RackProfile: physical rack power and cooling class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RackProfile:
    """
    A rack power class, used when the chapter argument is about power
    density rather than regional carbon.

    power_kw         — Rack IT power draw (kW)
    cooling_type     — "air", "immersion", "liquid"
    air_cooled       — whether air cooling is feasible at this density
    """
    name: str
    power_kw: float
    cooling_type: str

    @property
    def air_cooled(self) -> bool:
        return self.power_kw <= AIR_COOLING_LIMIT_KW

    def __repr__(self):
        return f"Rack({self.name}, {self.power_kw} kW, {self.cooling_type})"


# ─────────────────────────────────────────────────────────────────────────────
# Named Grid Profiles — one per region referenced across Vol2 chapters
# ─────────────────────────────────────────────────────────────────────────────

class Grids:
    """
    Named regional grid profiles for carbon-aware scheduling examples.

    These are the grids actually referenced in Vol2 prose:
    Quebec, Norway   → clean hydro/hydro (best case)
    France           → nuclear (low carbon)
    EU_Avg, US_Avg   → mixed (reference baselines)
    Poland           → coal-heavy (worst case in book)
    """

    # Best-in-class clean grids (hydro/nuclear)
    Quebec = GridProfile(
        name="Quebec (Hydro)",
        carbon_intensity_g_kwh=CARBON_QUEBEC_GCO2_KWH,
        pue=PUE_LIQUID_COOLED,       # Modern liquid-cooled hyperscale
        wue=WUE_LIQUID,
        primary_source="hydro",
    )

    Norway = GridProfile(
        name="Norway (Hydro)",
        carbon_intensity_g_kwh=CARBON_NORWAY_GCO2_KWH,
        pue=PUE_LIQUID_COOLED,
        wue=WUE_LIQUID,
        primary_source="hydro",
    )

    France = GridProfile(
        name="France (Nuclear)",
        carbon_intensity_g_kwh=CARBON_FRANCE_GCO2_KWH,
        pue=PUE_BEST_AIR,
        wue=WUE_AIR_COOLED,
        primary_source="nuclear",
    )

    # Reference baselines
    EU_Avg = GridProfile(
        name="EU Average",
        carbon_intensity_g_kwh=CARBON_EU_AVG_GCO2_KWH,
        pue=PUE_TYPICAL,
        wue=WUE_EVAPORATIVE,
        primary_source="mixed",
    )

    US_Avg = GridProfile(
        name="US Average",
        carbon_intensity_g_kwh=CARBON_US_AVG_GCO2_KWH,
        pue=PUE_BEST_AIR,            # Modern hyperscale; use PUE_TYPICAL for enterprise
        wue=WUE_EVAPORATIVE,
        primary_source="mixed",
    )

    # Worst-case carbon grid (used in Vol2 comparisons)
    Poland = GridProfile(
        name="Poland (Coal)",
        carbon_intensity_g_kwh=CARBON_POLAND_GCO2_KWH,
        pue=PUE_LEGACY,             # Older datacenter infrastructure typical here
        wue=WUE_EVAPORATIVE,
        primary_source="coal",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Named Rack Profiles — canonical power density tiers
# ─────────────────────────────────────────────────────────────────────────────

class Racks:
    """Reference rack power classes for power-density arguments."""

    Traditional = RackProfile(
        name="Traditional Enterprise",
        power_kw=RACK_POWER_TRADITIONAL_KW,
        cooling_type="air",
    )

    AI_Standard = RackProfile(
        name="AI Cluster (Standard)",
        power_kw=RACK_POWER_AI_TYPICAL_KW,
        cooling_type="liquid",
    )

    AI_HighDensity = RackProfile(
        name="AI Cluster (High Density)",
        power_kw=RACK_POWER_AI_HIGH_KW,
        cooling_type="immersion",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Datacenters — top-level namespace (the public API)
# ─────────────────────────────────────────────────────────────────────────────

class Datacenters:
    """
    Top-level namespace for datacenter sustainability profiles.

    Two sub-namespaces:
      Datacenters.Grids.*   — regional grid profiles (carbon intensity + PUE/WUE)
      Datacenters.Racks.*   — rack power density profiles

    Common aliases at the top level for the most-cited grids:
      Datacenters.Quebec    → Grids.Quebec
      Datacenters.Poland    → Grids.Poland
      Datacenters.US_Avg    → Grids.US_Avg
    """
    Grids = Grids
    Racks = Racks

    # Top-level aliases for the grids most referenced in prose
    Quebec  = Grids.Quebec
    Norway  = Grids.Norway
    France  = Grids.France
    EU_Avg  = Grids.EU_Avg
    US_Avg  = Grids.US_Avg
    Poland  = Grids.Poland
