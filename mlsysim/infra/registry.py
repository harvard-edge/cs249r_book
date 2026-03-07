from .types import GridProfile, RackProfile
from ..core.constants import (
    PUE_LIQUID_COOLED, PUE_BEST_AIR, PUE_TYPICAL, PUE_LEGACY,
    WUE_AIR_COOLED, WUE_EVAPORATIVE, WUE_LIQUID,
    CARBON_US_AVG_GCO2_KWH, CARBON_EU_AVG_GCO2_KWH,
    CARBON_QUEBEC_GCO2_KWH, CARBON_FRANCE_GCO2_KWH,
    CARBON_POLAND_GCO2_KWH, CARBON_NORWAY_GCO2_KWH,
    RACK_POWER_TRADITIONAL_KW, RACK_POWER_AI_TYPICAL_KW, RACK_POWER_AI_HIGH_KW
)

class Grids:
    Quebec = GridProfile(
        name="Quebec (Hydro)",
        carbon_intensity_g_kwh=CARBON_QUEBEC_GCO2_KWH,
        pue=PUE_LIQUID_COOLED,
        wue=WUE_LIQUID,
        primary_source="hydro",
        metadata={"source_url": "https://www.hydroquebec.com/about/our-energy.html", "last_verified": "2025-03-06"}
    )
    Norway = GridProfile(
        name="Norway (Hydro)",
        carbon_intensity_g_kwh=CARBON_NORWAY_GCO2_KWH,
        pue=PUE_LIQUID_COOLED,
        wue=WUE_LIQUID,
        primary_source="hydro"
    )
    US_Avg = GridProfile(
        name="US Average",
        carbon_intensity_g_kwh=CARBON_US_AVG_GCO2_KWH,
        pue=PUE_BEST_AIR,
        wue=WUE_EVAPORATIVE,
        primary_source="mixed"
    )
    Poland = GridProfile(
        name="Poland (Coal)",
        carbon_intensity_g_kwh=CARBON_POLAND_GCO2_KWH,
        pue=PUE_LEGACY,
        wue=WUE_EVAPORATIVE,
        primary_source="coal"
    )

class Racks:
    Traditional = RackProfile(
        name="Traditional Enterprise",
        power_kw=RACK_POWER_TRADITIONAL_KW,
        cooling_type="air"
    )
    AI_Standard = RackProfile(
        name="AI Cluster (Standard)",
        power_kw=RACK_POWER_AI_TYPICAL_KW,
        cooling_type="liquid"
    )

class Infra:
    Grids = Grids
    Racks = Racks
    
    Quebec = Grids.Quebec
    US_Avg = Grids.US_Avg
    Poland = Grids.Poland
