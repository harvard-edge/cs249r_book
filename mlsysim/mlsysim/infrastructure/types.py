from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import Metadata

class GridProfile(BaseModel):
    """
    Layer C (Infrastructure Context): Represents a regional power grid.
    
    A GridProfile defines the environmental constraints of a physical location,
    specifically its carbon intensity and energy mix. This is essential for 
    the SustainabilityModel to calculate the exact kg CO2e footprint of a job.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    carbon_intensity_g_kwh: float
    pue: float
    wue: float
    primary_source: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    renewable_pct: Optional[float] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @property
    def carbon_intensity_kg_kwh(self) -> float:
        return self.carbon_intensity_g_kwh / 1000.0

    def carbon_kg(self, facility_energy_kwh: float) -> float:
        """Carbon footprint from total facility energy (PUE already applied).

        Args:
            facility_energy_kwh: Total facility energy in kWh (IT energy × PUE).
        """
        return facility_energy_kwh * self.carbon_intensity_kg_kwh

class CoolingProfile(BaseModel):
    """
    Facility cooling tier (PUE / WUE), isolated from a geographic grid.
    
    Used to model different cooling strategies (Air, Direct-to-Chip Liquid, 
    Evaporative) independent of location.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    pue: float
    wue: float  # L/kWh
    metadata: Metadata = Field(default_factory=Metadata)

class RackProfile(BaseModel):
    """
    Represents the physical constraints of a datacenter rack.
    
    Crucial for analyzing power density (kW/rack) limits, where high-density 
    AI deployments force transitions from air to liquid cooling.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    power_kw: float
    cooling_type: str
    metadata: Metadata = Field(default_factory=Metadata)

class PricePoint(BaseModel):
    """Named rate (pint quantity) with provenance for economics tables."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    rate: Any
    metadata: Metadata = Field(default_factory=Metadata)

class Datacenter(BaseModel):
    """
    Represents a specific physical facility combining a Grid and Cooling tech.
    
    Allows overriding the regional grid's average PUE with the facility's 
    specific highly-optimized PUE.
    """
    name: str
    grid: GridProfile
    pue_override: Optional[float] = None
    
    @property
    def pue(self) -> float:
        return self.pue_override or self.grid.pue
