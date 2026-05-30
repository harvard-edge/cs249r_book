from pathlib import Path

from .types import GridProfile, RackProfile, Datacenter, CoolingProfile
from ..core.provenance import sourced
from ..core.registry import Registry
from ..core.loader import load_collection
from ..core.types import Metadata
from ..core import provenance_catalog as pc

# --- Facility cooling tiers (PUE / WUE anchors for appendices) ---
PUE_LIQUID_COOLED = sourced(1.06, pc.UPTIME_PUE_2022, name="PUE (Liquid-Cooled)", description="Best-in-class liquid-cooled AI datacenter PUE.")
PUE_BEST_AIR = sourced(1.12, pc.UPTIME_PUE_2022, name="PUE (Best Air-Cooled)", description="Best-in-class air-cooled hyperscale datacenter PUE.")
PUE_TYPICAL = sourced(1.40, pc.UPTIME_PUE_2022, name="PUE (Industry Average)", description="Industry average traditional datacenter PUE.")
PUE_LEGACY = sourced(1.58, pc.UPTIME_PUE_2022, name="PUE (Legacy Air-Cooled)", description="Older enterprise datacenter PUE tier.")
PUE_STATE_OF_ART = sourced(1.10, pc.UPTIME_PUE_2022, name="PUE (state of art)", description="Highly optimized modern datacenter PUE benchmark.")
WUE_AIR_COOLED = sourced(0.5, pc.BOOK_WUE_ANCHORS, name="WUE (air-cooled)", description="Water usage effectiveness for air-cooled facilities.")
WUE_EVAPORATIVE = sourced(1.8, pc.BOOK_WUE_ANCHORS, name="WUE (evaporative)", description="Water usage effectiveness for evaporative cooling.")
WUE_LIQUID = sourced(0.0, pc.BOOK_WUE_ANCHORS, name="WUE (liquid-cooled)", description="Closed-loop liquid cooling (near-zero WUE).")

RACK_POWER_TRADITIONAL_KW = sourced(12, pc.BOOK_RACK_POWER, name="Rack power (traditional)", description="Traditional enterprise rack power (kW).")
RACK_POWER_AI_TYPICAL_KW = sourced(70, pc.BOOK_RACK_POWER, name="Rack power (AI typical)", description="Typical AI cluster rack power (kW).")
RACK_POWER_AI_HIGH_KW = sourced(100, pc.BOOK_RACK_POWER, name="Rack power (AI high)", description="High-density AI rack power (kW).")
AIR_COOLING_LIMIT_KW = sourced(30, pc.BOOK_RACK_POWER, name="Air cooling limit (kW)", description="Approximate rack power where air cooling becomes impractical.")


class FacilityCooling(Registry):
    """Datacenter cooling efficiency tiers (not regional grids)."""
    LiquidCooled = CoolingProfile(name="Liquid-Cooled", pue=PUE_LIQUID_COOLED, wue=WUE_LIQUID, metadata=Metadata(provenance=pc.UPTIME_PUE_2022))
    BestAir = CoolingProfile(name="Best Air-Cooled", pue=PUE_BEST_AIR, wue=WUE_EVAPORATIVE, metadata=Metadata(provenance=pc.UPTIME_PUE_2022))
    Typical = CoolingProfile(name="Industry Average", pue=PUE_TYPICAL, wue=WUE_EVAPORATIVE, metadata=Metadata(provenance=pc.UPTIME_PUE_2022))
    Legacy = CoolingProfile(name="Legacy Air-Cooled", pue=PUE_LEGACY, wue=WUE_EVAPORATIVE, metadata=Metadata(provenance=pc.UPTIME_PUE_2022))
    StateOfArt = CoolingProfile(name="State of Art", pue=PUE_STATE_OF_ART, wue=WUE_EVAPORATIVE, metadata=Metadata(provenance=pc.UPTIME_PUE_2022))


# Regional grid carbon intensity (gCO2/kWh) with typical facility PUE/WUE — leaf
# reference data, loaded from YAML (composition below — Datacenters — stays Python).
Grids = load_collection(
    Path(__file__).parent / "data" / "grids.yaml", GridProfile, name="Grids",
    doc="Regional grid carbon intensity (gCO2/kWh) with typical facility PUE/WUE.",
)


class Datacenters(Registry):
    """Registry namespace for Datacenters."""
    Quebec_Hydro = Datacenter(name="Quebec Hydro Campus", grid=Grids.Quebec)
    Norway_Hydro = Datacenter(name="Norway Hydro Campus", grid=Grids.Norway)
    US_Hyperscale = Datacenter(name="US Hyperscale Region", grid=Grids.US_Avg)
    EU_Average = Datacenter(name="EU Average Region", grid=Grids.EU_Avg)
    France_Nuclear = Datacenter(name="France Nuclear Region", grid=Grids.France)
    Iowa_Reference = Datacenter(name="Iowa Reference Site", grid=Grids.Iowa)
    Poland_Coal = Datacenter(name="Poland Coal Region", grid=Grids.Poland)


class Racks(Registry):
    """Registry namespace for Racks."""
    Traditional = RackProfile(
        name="Traditional Enterprise",
        power_kw=RACK_POWER_TRADITIONAL_KW,
        cooling_type="air",
        metadata=Metadata(provenance=pc.BOOK_RACK_POWER),
    )
    AI_Standard = RackProfile(
        name="AI Cluster (Standard)",
        power_kw=RACK_POWER_AI_TYPICAL_KW,
        cooling_type="liquid",
        metadata=Metadata(provenance=pc.BOOK_RACK_POWER),
    )
    AI_High = RackProfile(
        name="AI Cluster (High Density)",
        power_kw=RACK_POWER_AI_HIGH_KW,
        cooling_type="liquid",
        metadata=Metadata(provenance=pc.BOOK_RACK_POWER),
    )


from .pricing import Pricing
from .capacity import Capacity


class Infrastructure(Registry):
    """Registry namespace for Infrastructure."""
    Grids = Grids
    Datacenters = Datacenters
    Racks = Racks
    FacilityCooling = FacilityCooling
    Pricing = Pricing
    Capacity = Capacity
    AIR_COOLING_LIMIT_KW = AIR_COOLING_LIMIT_KW
