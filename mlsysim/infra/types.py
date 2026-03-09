from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Any, Annotated
from ..core.constants import Q_, ureg
from ..core.types import Quantity, Metadata

class GridProfile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    carbon_intensity_g_kwh: float
    pue: float
    wue: float
    primary_source: str
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

class RackProfile(BaseModel):
    name: str
    power_kw: float
    cooling_type: str

class Datacenter(BaseModel):
    name: str
    grid: GridProfile
    pue_override: Optional[float] = None
    
    @property
    def pue(self) -> float:
        return self.pue_override or self.grid.pue
