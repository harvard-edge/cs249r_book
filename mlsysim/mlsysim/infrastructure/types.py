from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import Metadata

class GridProfile(BaseModel):
    """
    Layer C (Infrastructure Context): Represents a regional power grid.
    
    A GridProfile defines the environmental constraints of a physical location,
    specifically its carbon intensity and energy mix. This is essential for 
    the SustainabilityModel to estimate a job's operational CO2e footprint
    under a declared constant-intensity assumption.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
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
        """Convert stored grid carbon intensity from grams to kilograms.

        Returns
        -------
        float
            Kilograms CO2-equivalent per kilowatt-hour.

        Notes
        -----
        Equation: I_kg = carbon_intensity_g_kwh / 1000.

        Assumptions: the stored grams-per-kWh figure represents the intended location and time boundary.

        Excludes: PUE multiplication, time-varying grid mix, and embodied hardware emissions.

        References
        ----------
        Derivation: one kilogram contains one thousand grams; divide the mass coefficient while retaining the kWh denominator.

        Examples
        --------
        >>> from mlsysim import Infrastructure
        >>> grid = Infrastructure.Datacenters.Quebec_Hydro.grid
        >>> grid.carbon_intensity_kg_kwh == grid.carbon_intensity.m_as('kg/kWh')
        True
        """
        return self.carbon_intensity_g_kwh / 1000.0

    @property
    def carbon_intensity(self):
        """Attach units to the stored grid-emissions coefficient.

        Returns
        -------
        Quantity
            Carbon intensity in grams CO2-equivalent per kWh.

        Notes
        -----
        Equation: I = carbon_intensity_g_kwh * gram / kWh.

        Assumptions: the scalar already contains the correct grid boundary and source estimate.

        Excludes: facility overhead, temporal variation, and recomputation of grid emissions.

        References
        ----------
        Derivation: attach the declared units to the registry scalar so energy-times-intensity yields a mass.

        Examples
        --------
        >>> from mlsysim import Infrastructure
        >>> grid = Infrastructure.Datacenters.Quebec_Hydro.grid
        >>> grid.carbon_intensity.m_as('g/kWh') == grid.carbon_intensity_g_kwh
        True
        """
        from ..core.units import Q_
        return Q_(self.carbon_intensity_g_kwh, "gram / kWh")

    @property
    def water_intensity(self):
        """Attach volume-per-energy units to regional water intensity.

        Returns
        -------
        Quantity
            Stored WUE in liters per kWh.

        Notes
        -----
        Equation: WUE_quantity = wue * liter / kWh.

        Assumptions: wue is supplied for the accounting boundary used by the registry source; the property does not distinguish direct and upstream water use.

        Excludes: PUE adjustment, cooling-system simulation, and regional water-scarcity weighting.

        References
        ----------
        Derivation: attach the declared volume/energy units; the source scalar remains the underlying assumption.

        Examples
        --------
        >>> from mlsysim import Infrastructure
        >>> grid = Infrastructure.Datacenters.Quebec_Hydro.grid
        >>> grid.water_intensity.m_as('liter/kWh') == grid.wue
        True
        """
        from ..core.units import Q_
        return Q_(self.wue, "liter / kWh")

    def carbon_kg(self, facility_energy_kwh: float) -> float:
        """Multiply facility energy by the regional carbon coefficient.

        Parameters
        ----------
        facility_energy_kwh : float
            Nonnegative total facility energy in kWh. Apply PUE to IT energy
            before calling; this method does not apply it again.

        Returns
        -------
        float
            Operational CO2-equivalent mass in kilograms.

        Notes
        -----
        Equation: C_kg = facility_energy_kwh * carbon_intensity_kg_kwh.

        Assumptions: constant regional intensity over the energy-consumption
        period and matching energy/emissions accounting boundaries.

        Excludes: manufacturing emissions, hourly marginal grid intensity,
        transmission changes, and additional PUE multiplication.

        References
        ----------
        Derivation: multiply energy by emissions mass per unit energy; kWh
        cancels, leaving kilograms. The registry source supplies the coefficient.

        Examples
        --------
        >>> from mlsysim import Infrastructure
        >>> grid = Infrastructure.Datacenters.Quebec_Hydro.grid
        >>> grid.carbon_kg(2) == 2 * grid.carbon_kg(1)
        True
        """
        return facility_energy_kwh * self.carbon_intensity_kg_kwh

class CoolingProfile(BaseModel):
    """
    Facility cooling tier (PUE / WUE), isolated from a geographic grid.
    
    Used to model different cooling strategies (Air, Direct-to-Chip Liquid, 
    Evaporative) independent of location.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    pue: float
    wue: float  # L/kWh
    metadata: Metadata = Field(default_factory=Metadata)

    @property
    def water_intensity(self):
        """Attach volume-per-energy units to the cooling WUE.

        Returns
        -------
        Quantity
            Cooling water coefficient in liters per kWh.

        Notes
        -----
        Equation: WUE_quantity = wue * liter / kWh.

        Assumptions: the scalar describes the cooling profile's stated energy boundary.

        Excludes: regional upstream water use, weather-dependent evaporation, and a cooling thermodynamics model.

        References
        ----------
        Derivation: attach units to the declared cooling coefficient; multiplication by matching-boundary energy gives the allocated water volume.

        Examples
        --------
        >>> from mlsysim import Infrastructure
        >>> cooling = Infrastructure.FacilityCooling.BestAir
        >>> cooling.water_intensity.m_as('liter/kWh') == cooling.wue
        True
        """
        from ..core.units import Q_
        return Q_(self.wue, "liter / kWh")

class RackProfile(BaseModel):
    """
    Represents the physical constraints of a datacenter rack.
    
    Crucial for analyzing power density (kW/rack) limits, where high-density 
    AI deployments force transitions from air to liquid cooling.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    power_kw: float
    cooling_type: str
    metadata: Metadata = Field(default_factory=Metadata)

    @property
    def power(self):
        """Attach power units to a rack capacity envelope.

        Returns
        -------
        Quantity
            Rated or assumed rack power envelope in kilowatts.

        Notes
        -----
        Equation: P_envelope = power_kw * kilowatt.

        Assumptions: power_kw is the registry's rack design limit rather than observed demand.

        Excludes: device inventory, utilization, electrical derating, and measured cooling capacity.

        References
        ----------
        Derivation: attach the declared kilowatt unit to a stored design constraint; actual rack load must be calculated separately.

        Examples
        --------
        >>> from mlsysim import Infrastructure
        >>> rack = Infrastructure.Racks.AI_Standard
        >>> rack.power.m_as('kW') == rack.power_kw
        True
        """
        from ..core.units import Q_
        return Q_(self.power_kw, "kilowatt")

class PricePoint(BaseModel):
    """Named rate (pint quantity) with provenance for economics tables."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    rate: Any
    metadata: Metadata = Field(default_factory=Metadata)

class Datacenter(BaseModel):
    """
    Represents a specific physical facility combining a Grid and Cooling tech.
    
    Allows overriding the regional grid's average PUE with the facility's 
    specific highly-optimized PUE.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    name: str
    grid: GridProfile
    pue_override: Optional[float] = None
    metadata: Metadata = Field(default_factory=Metadata)
    
    @property
    def pue(self) -> float:
        """Select a facility PUE override or the grid-profile default.

        Returns
        -------
        float
            Dimensionless facility-to-IT energy ratio.

        Notes
        -----
        Equation: PUE = pue_override or grid.pue.

        Assumptions: a nonzero override takes precedence. None and zero fall back to the grid value under the implementation's truth-value selection.

        Excludes: estimating PUE from workload utilization, metered power, or weather; no lower-bound validation is performed here.

        References
        ----------
        Derivation: this property chooses a contextual coefficient. PUE itself is the ratio of facility energy to IT energy, not an additional energy amount.

        Examples
        --------
        >>> from mlsysim import Infrastructure
        >>> facility = Infrastructure.Datacenters.Quebec_Hydro
        >>> facility.pue == (facility.pue_override or facility.grid.pue)
        True
        """
        return self.pue_override or self.grid.pue
