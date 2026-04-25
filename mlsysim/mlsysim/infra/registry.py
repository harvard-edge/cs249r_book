from .types import GridProfile, RackProfile
from ..core.constants import (
    PUE_LIQUID_COOLED, PUE_BEST_AIR, PUE_LEGACY,
    WUE_EVAPORATIVE, WUE_LIQUID,
    CARBON_US_AVG_GCO2_KWH, CARBON_QUEBEC_GCO2_KWH, CARBON_IOWA_GCO2_KWH,
    CARBON_POLAND_GCO2_KWH, CARBON_NORWAY_GCO2_KWH,
    RACK_POWER_TRADITIONAL_KW, RACK_POWER_AI_TYPICAL_KW
)

class Grids:
    Quebec = GridProfile(
        name="Quebec (Hydro)",
        carbon_intensity_g_kwh=CARBON_QUEBEC_GCO2_KWH,
        pue=PUE_LIQUID_COOLED,
        wue=WUE_LIQUID,
        primary_source="hydro",
        lat=52.9399,
        lon=-73.5491,
        renewable_pct=99.0,
        metadata={"source_url": "https://www.hydroquebec.com/about/our-energy.html", "last_verified": "2025-03-06"}
    )
    Norway = GridProfile(
        name="Norway (Hydro)",
        carbon_intensity_g_kwh=CARBON_NORWAY_GCO2_KWH,
        pue=PUE_LIQUID_COOLED,
        wue=WUE_LIQUID,
        primary_source="hydro",
        lat=60.472,
        lon=8.4689,
        renewable_pct=98.0
    )
    US_Avg = GridProfile(
        name="US Average",
        carbon_intensity_g_kwh=CARBON_US_AVG_GCO2_KWH,
        pue=PUE_BEST_AIR,
        wue=WUE_EVAPORATIVE,
        primary_source="mixed",
        lat=39.8283,
        lon=-98.5795,
        renewable_pct=21.0
    )
    Iowa = GridProfile(
        name="Iowa (Coal/Gas Reference)",
        carbon_intensity_g_kwh=CARBON_IOWA_GCO2_KWH,
        pue=PUE_BEST_AIR,
        wue=WUE_EVAPORATIVE,
        primary_source="coal_gas",
        lat=42.0329,
        lon=-93.5815,
        renewable_pct=64.0,
        metadata={
            "description": "Reference high-carbon grid profile used in MLSys·im tutorials for regional contrast.",
            "last_verified": "2026-04-25",
        },
    )
    Poland = GridProfile(
        name="Poland (Coal)",
        carbon_intensity_g_kwh=CARBON_POLAND_GCO2_KWH,
        pue=PUE_LEGACY,
        wue=WUE_EVAPORATIVE,
        primary_source="coal",
        lat=51.9194,
        lon=19.1451,
        renewable_pct=17.0
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
    Norway = Grids.Norway
    US_Avg = Grids.US_Avg
    Iowa = Grids.Iowa
    Poland = Grids.Poland
