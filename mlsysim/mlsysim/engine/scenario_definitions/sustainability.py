"""Power, heat, energy, carbon, and material accounting.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from ...core.types import Quantity
from ...hardware.types import HardwareNode
from ...models.types import Workload
from ...systems.types import Fleet
from .base import BaseScenario
from .optimization import SingleNodeInferenceScenario


class PowerTransientScenario(BaseScenario):
    """A barrier-aligned load step on ``hardware`` and across ``n_devices``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.PowerTransientModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.PowerTransientModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.PowerTransientModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.PowerTransientModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.H100BarrierLoadStep
    >>> type(case.result).__name__
    'PowerTransientResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    hardware: HardwareNode
    low_power: Quantity  # communication-phase power per device
    rail_voltage: Quantity
    loop_inductance: Quantity
    rise_times: Dict[str, Quantity]
    n_devices: int
    ramp_window: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.PowerTransientModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import PowerTransientModel

        return PowerTransientModel().solve(
            self.hardware.tdp,
            self.low_power,
            self.rail_voltage,
            self.loop_inductance,
            self.rise_times,
            self.n_devices,
            self.ramp_window,
        )


class CoolingScenario(BaseScenario):
    """Coolant flow that carries ``heat_load`` at ``temperature_rise``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.CoolantFlowModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.CoolantFlowModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.CoolantFlowModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.CoolantFlowModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.AiRackAirCooling
    >>> type(case.result).__name__
    'CoolantFlowResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    heat_load: Quantity
    temperature_rise: Quantity
    coolant: str = "air"

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.CoolantFlowModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import CoolantFlowModel

        return CoolantFlowModel().solve(self.heat_load, self.temperature_rise, self.coolant)


class OutageScenario(BaseScenario):
    """A failure domain of ``failed_devices`` down for ``repair_time`` during a job on
    ``fleet``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.OutageCostModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.OutageCostModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.OutageCostModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.OutageCostModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.CduOutageTraining10K
    >>> type(case.result).__name__
    'OutageCostResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    failed_devices: int
    fleet: Fleet
    repair_time: Quantity
    device_hour_rate: Quantity
    rollback_times: Dict[str, Quantity] = {}

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.OutageCostModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import OutageCostModel

        return OutageCostModel().solve(
            self.failed_devices,
            self.fleet.total_accelerators,
            self.repair_time,
            self.device_hour_rate,
            self.rollback_times,
        )


class BatteryBudgetScenario(BaseScenario):
    """A continuous ``power_draw`` against a battery and a required runtime.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Power draw is constant and all stored battery energy is usable.

    Excludes: Conversion losses, reserve margins, battery aging, and discharge-rate effects
    are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.AlwaysOnObjectDetectorPhone
    >>> str(case.runtime)
    '7.5 h'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    battery_energy: Quantity
    power_draw: Quantity
    required_runtime: Quantity

    @property
    def runtime(self) -> Quantity:
        """Ideal runtime from stored energy at constant power draw. Equation: ``battery_energy /
        power_draw``. Units: hours; conversion losses and battery derating are excluded.
        """
        return (self.battery_energy / self.power_draw).to("hour")

    @property
    def required_energy(self) -> Quantity:
        """Stored energy required for the requested constant-power runtime. Equation: ``power_draw
        * required_runtime``. Units: Wh; no reserve margin is added.
        """
        return (self.power_draw * self.required_runtime).to("Wh")

    @property
    def budget_ratio(self) -> float:
        """Required energy relative to available battery energy. Equation: ``required_energy /
        battery_energy``. Units: dimensionless; values above one exceed the ideal energy budget.
        """
        return (self.required_energy / self.battery_energy).to("").magnitude


class ThermalCeilingScenario(BaseScenario):
    """A model's sustained power against a passively cooled device's ceiling, before and after
    a reduction.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: A supplied reduction factor divides power uniformly; the sustained thermal
    ceiling is fixed.

    Excludes: Transient thermal storage, ambient changes, and throttling dynamics are
    absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.UnoptimizedVisionModelPhone
    >>> str(case.reduced_power)
    '3.0 W'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    power: Quantity
    reduction_factor: float
    sustained_limit: Quantity
    heating_rate: Quantity  # junction temperature rise at ``power``
    trip_time: Quantity
    trip_temperature: Quantity
    burst_rate: float  # frames per second before throttling
    throttled_rate: float  # frames per second after

    @property
    def reduced_power(self) -> Quantity:
        """Power after the stated reduction factor. Equation: ``power / reduction_factor``. Units:
        watts; the factor is an assumed multiplicative improvement.
        """
        return (self.power / self.reduction_factor).to("watt")

    @property
    def headroom(self) -> Quantity:
        """Remaining sustained-power allowance after reduction. Equation: ``sustained_limit -
        reduced_power``. Units: watts; negative headroom exceeds the thermal ceiling.
        """
        return (self.sustained_limit - self.reduced_power).to("watt")


class EnergyPerInferenceScenario(BaseScenario):
    """Energy per inference across tiers: solver-derived device energies plus stated anchors.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Named modeled runs and external energy anchors retain their distinct
    provenance; a modeled run replaces a duplicate anchor name.

    Excludes: Ratios do not establish equal task quality or identical system boundaries
    across entries.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.EnergyPerInferenceTiers
    >>> print(f"{next(iter(case.energies.values())).m_as('microjoule'):.1f} microjoule")
    10.0 microjoule
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    anchors: Dict[str, Quantity]
    runs: Dict[str, SingleNodeInferenceScenario]

    @property
    def energies(self) -> Dict[str, Quantity]:
        """Named per-inference energies combining supplied anchors and modeled runs. Equation: copy
        ``anchors`` then replace matching keys with ``run.result.energy``. Units: joules for
        modeled runs; anchors retain their supplied units and provenance.
        """
        out = dict(self.anchors)  # stated units kept, so ratios stay exact
        out.update({k: s.result.energy.to("joule") for k, s in self.runs.items()})
        return out

    def span(self, low: str, high: str) -> float:
        """Energy ratio between two named entries. Equation: ``energies[high] / energies[low]``.
        Units: dimensionless; comparisons require equivalent task definitions.
        """
        e = self.energies
        return (e[high] / e[low]).to("").magnitude


class OffloadEnergyScenario(BaseScenario):
    """Energy to send ``payload`` by radio against running the model locally.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: A constant radio energy-per-byte coefficient prices the payload;
    accelerator-only and whole-device local energy remain separate comparisons.

    Excludes: Radio setup, codec energy, and model-quality differences are absent unless
    already included in inputs.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RadioUploadVsMobileNetV2
    >>> str(case.transmit_energy)
    '100.0 mJ'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    payload: Quantity
    transmit_energy_per_byte: Quantity
    local_run: SingleNodeInferenceScenario

    @property
    def transmit_energy(self) -> Quantity:
        """Radio energy allocated to transmitting the payload. Equation: ``payload *
        transmit_energy_per_byte``. Units: mJ; radio setup and idle energy are excluded unless
        built into the coefficient.
        """
        return (self.payload * self.transmit_energy_per_byte).to("mJ")

    @property
    def accelerator_energy(self) -> Quantity:
        """Workload's registered accelerator-only inference energy. Equation: read
        ``local_run.workload.inference_energy``. Units: energy; this anchor need not equal
        modeled whole-device energy.
        """
        return self.local_run.workload.inference_energy

    @property
    def device_energy(self) -> Quantity:
        """Energy from the local inference solver profile. Equation: ``local_run.result.energy``.
        Units: joules; it inherits the solver's power and duration assumptions.
        """
        return self.local_run.result.energy.to("joule")

    @property
    def ratio_accelerator(self) -> float:
        """Transmission energy relative to the accelerator-only anchor. Equation: ``transmit_energy
        / accelerator_energy``. Units: dimensionless; system boundaries differ from the whole-
        device comparison.
        """
        return (self.transmit_energy / self.accelerator_energy).to("").magnitude

    @property
    def ratio_device(self) -> float:
        """Transmission energy relative to modeled whole-device inference energy. Equation:
        ``transmit_energy / device_energy``. Units: dimensionless; codec energy is not added.
        """
        return (self.transmit_energy / self.device_energy).to("").magnitude


def _rack_it_ratio(rack) -> float:
    """IT load over accelerator power for a rack profile (hosts, switches, conversion inside IT)."""
    return (rack.it_power / rack.accelerator_power).to("").magnitude


class FacilityPowerScenario(BaseScenario):
    """``devices`` accelerators at ``device_power`` in racks shaped like ``rack``, at ``pue``.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.FacilityPowerModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.FacilityPowerModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.FacilityPowerModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Mega100KFacilityPower
    >>> round(float(scenario.it_ratio), 4)
    1.375"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    devices: float
    device_power: Quantity
    rack: Any  # a Systems.Racks profile: supplies the IT-to-accelerator ratio
    pue: float

    @property
    def it_ratio(self) -> float:
        """Dimensionless total-IT-to-accelerator power ratio from the rack's power accounting.

        Equation: ``_rack_it_ratio(rack)``."""
        return _rack_it_ratio(self.rack)

    @property
    def result(self):
        """Return the modeled FacilityPower result from
        ``mlsysim.solvers.FacilityPowerModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import FacilityPowerModel

        return FacilityPowerModel().solve(self.devices, self.device_power, self.it_ratio, self.pue)


class RunFootprintScenario(BaseScenario):
    """A training run of ``device_hours`` on ``devices`` of ``device``: energy, carbon,
    embodied carbon, water.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.RunFootprintModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.RunFootprintModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes ``mlsysim.solvers.RunFootprintModel.solve``
    with the stored fields and any derived properties described below. Individual property
    contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Lab64A100TrainingFootprint
    >>> round(float(scenario.it_ratio), 4)
    1.375"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    device_hours: Quantity
    devices: float
    device: HardwareNode  # rated power (TDP) and embodied carbon per device
    rack: Any  # a Systems.Racks profile (IT-to-accelerator ratio)
    pue: float
    grids: Dict[str, Any]  # named Infrastructure.Grids profiles
    service_life: Quantity
    wue: Dict[str, Quantity] = {}

    @property
    def it_ratio(self) -> float:
        """Dimensionless total-IT-to-accelerator power ratio supplied by the rack profile.

        Equation: ``_rack_it_ratio(rack)``."""
        return _rack_it_ratio(self.rack)

    @property
    def result(self):
        """Return the modeled RunFootprint result from ``mlsysim.solvers.RunFootprintModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import RunFootprintModel

        return RunFootprintModel().solve(
            self.device_hours,
            self.devices,
            self.device.tdp,
            self.it_ratio,
            self.pue,
            {k: g.carbon_intensity for k, g in self.grids.items()},
            self.device.embodied_carbon,
            self.service_life,
            self.wue,
        )


class WaterUseScenario(BaseScenario):
    """An IT load held for ``period`` at each named WUE.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.WaterUseModel.solve`` with
    the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.WaterUseModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes ``mlsysim.solvers.WaterUseModel.solve``
    with the stored fields and any derived properties described below. Individual property
    contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ReferenceItLoadAnnualWater
    >>> round(float(scenario.it_energy.magnitude), 4)
    87600000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    it_power: Quantity
    period: Quantity
    wue: Dict[str, Quantity]

    @property
    def it_energy(self) -> Quantity:
        """IT electricity consumption in kWh for constant load over the stated period.

        Equation: ``(it_power * period).to('kWh')``."""
        return (self.it_power * self.period).to("kWh")

    @property
    def result(self):
        """Return the modeled WaterUse result from ``mlsysim.solvers.WaterUseModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import WaterUseModel

        return WaterUseModel().solve(self.it_energy, self.wue)


class PueRetrofitScenario(BaseScenario):
    """A fixed IT load moved from ``pue_before`` to ``pue_after`` for ``period`` at an
    electricity price.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.PueRetrofitModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.PueRetrofitModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes ``mlsysim.solvers.PueRetrofitModel.solve``
    with the stored fields and any derived properties described below. Individual property
    contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.PueRetrofit2MW
    >>> round(float(scenario.result.facility_power_before.magnitude), 4)
    3.16"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    it_power: Quantity
    pue_before: float
    pue_after: float
    period: Quantity
    electricity_price: Quantity

    @property
    def result(self):
        """Return the modeled PueRetrofit result from ``mlsysim.solvers.PueRetrofitModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import PueRetrofitModel

        return PueRetrofitModel().solve(
            self.it_power, self.pue_before, self.pue_after, self.period, self.electricity_price
        )


class GridPlacementScenario(BaseScenario):
    """One facility ``energy`` placed on each named grid, with an optional fractional energy
    cut.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.GridPlacementModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.GridPlacementModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.GridPlacementModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.QuebecVersusPolandPlacement
    >>> round(float(scenario.intensities['Quebec'].magnitude), 4)
    20.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    energy: Quantity
    grids: Dict[str, Any]
    energy_cut: float = 0.0

    @property
    def intensities(self) -> Dict[str, Quantity]:
        """Named grid carbon intensities copied from the supplied grid profiles, retaining their
        quantity units.

        Equation: ``{k: g.carbon_intensity for k, g in grids.items()}``."""
        return {k: g.carbon_intensity for k, g in self.grids.items()}

    @property
    def result(self):
        """Return the modeled GridPlacement result from
        ``mlsysim.solvers.GridPlacementModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import GridPlacementModel

        return GridPlacementModel().solve(self.energy, self.intensities, self.energy_cut)


class EmbodiedPaybackScenario(BaseScenario):
    """A replacement for ``device`` that does its work with ``saved_fraction`` less energy, on
    named grids.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.EmbodiedPaybackModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.EmbodiedPaybackModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.EmbodiedPaybackModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.H100ReplacementPayback
    >>> round(float(scenario.result.saved_power.magnitude), 4)
    0.35"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    device: HardwareNode
    saved_fraction: float
    grids: Dict[str, Any]

    @property
    def result(self):
        """Return the modeled EmbodiedPayback result from
        ``mlsysim.solvers.EmbodiedPaybackModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import EmbodiedPaybackModel

        return EmbodiedPaybackModel().solve(
            self.device.embodied_carbon,
            self.device.tdp,
            self.saved_fraction,
            {k: g.carbon_intensity for k, g in self.grids.items()},
        )


class ServiceLifeExtensionScenario(BaseScenario):
    """A hardware service life extended from ``short_life`` to ``long_life``.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.ServiceLifeExtensionModel.solve`` with the stored fields and any
    derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.ServiceLifeExtensionModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.ServiceLifeExtensionModel.solve`` with the stored fields and any
    derived properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.AcceleratorLifeExtension
    >>> round(float(scenario.result.annual_charge_reduction), 4)
    0.4"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    short_life: Quantity
    long_life: Quantity

    @property
    def result(self):
        """Return the modeled ServiceLifeExtension result from
        ``mlsysim.solvers.ServiceLifeExtensionModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import ServiceLifeExtensionModel

        return ServiceLifeExtensionModel().solve(self.short_life, self.long_life)


class RepeatedWorkEnergyScenario(BaseScenario):
    """``fleet``'s accelerators at rated power, rolled back by each named interval.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.RepeatedWorkEnergyModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.RepeatedWorkEnergyModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.RepeatedWorkEnergyModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Training10KCoolingRollback
    >>> round(float(scenario.result.energy['low'].magnitude), 4)
    3.5"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fleet: Fleet
    rollbacks: Dict[str, Quantity]

    @property
    def result(self):
        """Return the modeled RepeatedWorkEnergy result from
        ``mlsysim.solvers.RepeatedWorkEnergyModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import RepeatedWorkEnergyModel

        return RepeatedWorkEnergyModel().solve(
            self.fleet.total_accelerators, self.fleet.node.accelerator.tdp, self.rollbacks
        )


class ServingCrossoverScenario(BaseScenario):
    """Serving ``requests_per_day`` at ``energy_per_request`` against a one-time energy.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.ServingCrossoverModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.ServingCrossoverModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.ServingCrossoverModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.TenMillionQueriesAgainstLab64Training
    >>> round(float(scenario.result.daily_energy.magnitude), 4)
    10.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    one_time_energy: Quantity
    requests_per_day: float
    energy_per_request: Quantity

    @property
    def result(self):
        """Return the modeled ServingCrossover result from
        ``mlsysim.solvers.ServingCrossoverModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import ServingCrossoverModel

        return ServingCrossoverModel().solve(self.one_time_energy, self.requests_per_day, self.energy_per_request)


class CascadeEnergyScenario(BaseScenario):
    """An edge model on every input, escalating a fraction to a cloud model over a radio link.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.CascadeEnergyModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.CascadeEnergyModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.CascadeEnergyModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.EdgeFirstImageCascade
    >>> round(float(scenario.result.escalated_energy.magnitude), 4)
    6.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    edge_energy: Quantity
    transmit_energy: Quantity
    cloud_energy: Quantity
    escalation: float

    @property
    def result(self):
        """Return the modeled CascadeEnergy result from
        ``mlsysim.solvers.CascadeEnergyModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import CascadeEnergyModel

        return CascadeEnergyModel().solve(self.edge_energy, self.transmit_energy, self.cloud_energy, self.escalation)


class DutyCycledPowerScenario(BaseScenario):
    """An always-on detector, a stage active a share of the time, and a full model run per
    event.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.DutyCycledPowerModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.DutyCycledPowerModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.DutyCycledPowerModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WakeWordHierarchyPower
    >>> round(float(scenario.full_duty), 4)
    0.0001"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    always_on_power: Quantity
    stage_power: Quantity
    stage_activity: float
    full_power: Quantity
    event_rate: Quantity
    full_duration: Quantity

    @property
    def full_duty(self) -> float:
        """Dimensionless full-model activity fraction, computed as event rate times per-event
        duration.

        Equation: ``(event_rate * full_duration).to('').magnitude``."""
        return (self.event_rate * self.full_duration).to("").magnitude

    @property
    def result(self):
        """Return the modeled DutyCycledPower result from
        ``mlsysim.solvers.DutyCycledPowerModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import DutyCycledPowerModel

        return DutyCycledPowerModel().solve(
            {"always_on": self.always_on_power, "stage": self.stage_power, "full": self.full_power},
            {"always_on": 1.0, "stage": self.stage_activity, "full": self.full_duty},
            self.full_power,
        )


class DataMovementEnergyScenario(BaseScenario):
    """One weight of ``bytes_per_param`` used for ``flops_per_param`` FLOPs at stated per-op
    energies.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.DataMovementEnergyModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.DataMovementEnergyModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.DataMovementEnergyModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.BatchOneDecodeWeightEnergy
    >>> round(float(scenario.result.compute_energy.magnitude), 4)
    20.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    energy_per_flop: Quantity
    energy_per_byte: Quantity
    flops_per_param: float
    bytes_per_param: Quantity
    operand_bytes: Quantity

    @property
    def result(self):
        """Return the modeled DataMovementEnergy result from
        ``mlsysim.solvers.DataMovementEnergyModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import DataMovementEnergyModel

        return DataMovementEnergyModel().solve(
            self.energy_per_flop, self.energy_per_byte, self.flops_per_param, self.bytes_per_param, self.operand_bytes
        )


class DemandReboundScenario(BaseScenario):
    """Per-unit energy scaled by a factor (or by a unit-price ratio) while volume scales by
    another.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.DemandReboundModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.DemandReboundModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.DemandReboundModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.TokenPriceRebound
    >>> round(float(scenario.unit_factor), 4)
    0.0333"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    volume_factor: float
    per_unit_factor: Optional[float] = None
    unit_price_before: Optional[float] = None  # used when per_unit_factor is None: energy tracks price
    unit_price_after: Optional[float] = None

    @property
    def unit_factor(self) -> float:
        """Dimensionless per-unit energy factor, using the explicit factor or the assumed
        after/before price proxy.

        Equation: Use per_unit_factor when supplied; otherwise
        unit_price_after/unit_price_before."""
        if self.per_unit_factor is not None:
            return self.per_unit_factor
        return self.unit_price_after / self.unit_price_before

    @property
    def result(self):
        """Return the modeled DemandRebound result from
        ``mlsysim.solvers.DemandReboundModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import DemandReboundModel

        return DemandReboundModel().solve(self.unit_factor, self.volume_factor)


class DevelopmentSweepScenario(BaseScenario):
    """A sweep of ``candidates`` runs, each costing ``relative_cost`` of the final run.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.DevelopmentSweepModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.DevelopmentSweepModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.DevelopmentSweepModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CandidateSweep50
    >>> round(float(scenario.result.final_run_equivalents), 4)
    5.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    candidates: float
    relative_cost: float

    @property
    def result(self):
        """Return the modeled DevelopmentSweep result from
        ``mlsysim.solvers.DevelopmentSweepModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import DevelopmentSweepModel

        return DevelopmentSweepModel().solve(self.candidates, self.relative_cost)


class CarbonPricedPlacementScenario(BaseScenario):
    """A run of ``it_energy`` on ``fleet`` placed among ``regions`` with and without a carbon price.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Regional grids and the fleet describe the same supplied IT-energy workload.

    Excludes: Scheduling uncertainty, future grid changes, and geographic service constraints.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``CarbonPricedPlacementModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.CarbonPricedRegionPlacement
    >>> value = case.result
    >>> value.best_region
    'Quebec'"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fleet: Fleet
    it_energy: Quantity
    regions: List[str]
    carbon_price_per_ton: float  # dollars per tonne CO2
    mfu: float
    reference_grid: Any

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.CarbonPricedPlacementModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = CarbonPricedPlacementModel().solve(...)`` using the fields shown below."""
        from ..solvers import CarbonPricedPlacementModel

        return CarbonPricedPlacementModel().solve(
            self.fleet, self.it_energy, self.regions, self.carbon_price_per_ton, self.mfu, self.reference_grid
        )


class PeriodicInferencePowerScenario(BaseScenario):
    """``workload`` run once per ``period`` on ``hardware`` at its peak ``precision`` rate, idling between runs.

    Active power is the board's ``tdp`` and idle power its ``tdp_min``; average
    power weights each by its share of the period (``DutyCycledPowerModel``).
    ``energy_cut`` (optional) divides the energy of each inference, to show how
    little a cheaper inference moves a duty-cycled average.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.DutyCycledPowerModel.solve``.

    Assumptions: One inference occurs per period; peak FLOPs bound execution time and registry TDP values proxy active/idle power.

    Excludes: Measured board energy, wake-up cost, memory-bound latency, and a period shorter than inference time.

    References
    ----------
    Derivation: This scenario binds named inputs to DutyCycledPowerModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.KeywordSpottingOnEsp32S3
    >>> value = case.average_power
    >>> print(f"{value:.3g~}")
    52.9 mW"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    period: Quantity
    precision: str = "int8"
    energy_cut: Optional[float] = None

    @property
    def inference_time(self) -> Quantity:
        """Peak-compute execution bound in ms, workload FLOPs divided by precision-specific peak rate.

        Equation: ``(workload.inference_flops / hardware.compute.precision_flops[precision]).to('ms')``."""
        return (self.workload.inference_flops / self.hardware.compute.precision_flops[self.precision]).to("ms")

    @property
    def duty(self) -> float:
        """Active fraction of the period, inference time divided by period; no clamping.

        Equation: ``(inference_time / period).to('').magnitude``."""
        return (self.inference_time / self.period).to("").magnitude

    @property
    def active_power(self) -> Quantity:
        """Registry TDP used as the active-power assumption, a power quantity.

        Equation: ``hardware.tdp``."""
        return self.hardware.tdp

    @property
    def idle_power(self) -> Quantity:
        """Registry minimum TDP used as the idle-power assumption, a power quantity.

        Equation: ``hardware.tdp_min``."""
        return self.hardware.tdp_min

    @property
    def inference_energy(self) -> Quantity:
        """Modeled active energy in mJ, active power times compute-bound execution time.

        Equation: ``(active_power * inference_time).to('mJ')``."""
        return (self.active_power * self.inference_time).to("mJ")

    @property
    def result(self):
        """Evaluate the stated periodic inference power case through ``DutyCycledPowerModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.DutyCycledPowerModel.solve``."""
        from ..solvers import DutyCycledPowerModel

        return DutyCycledPowerModel().solve(
            {"active": self.active_power, "idle": self.idle_power},
            {"active": self.duty, "idle": 1 - self.duty},
            self.active_power,
        )

    @property
    def active_contribution(self) -> Quantity:
        """Active state's duty-weighted average-power contribution in mW.

        Equation: ``result.contributions['active'].to('mW')``."""
        return self.result.contributions["active"].to("mW")

    @property
    def idle_contribution(self) -> Quantity:
        """Idle state's duty-weighted average-power contribution in mW.

        Equation: ``result.contributions['idle'].to('mW')``."""
        return self.result.contributions["idle"].to("mW")

    @property
    def average_power(self) -> Quantity:
        """Duty-weighted average power in mW from ``DutyCycledPowerModel.solve``.

        Equation: ``result.average_power.to('mW')``."""
        return self.result.average_power.to("mW")

    @property
    def idle_share(self) -> float:
        """Idle contribution divided by average power, dimensionless.

        Equation: ``(idle_contribution / average_power).to('').magnitude``."""
        return (self.idle_contribution / self.average_power).to("").magnitude

    @property
    def crossover_duty(self) -> float:
        """Active fraction where active and idle energy contributions are equal.

        Duty cycle at which active energy equals idle energy.

        Equation: ``(idle_power / (active_power + idle_power)).to('').magnitude``."""
        return (self.idle_power / (self.active_power + self.idle_power)).to("").magnitude

    @property
    def cut_saving(self) -> Quantity:
        """Average mW saved by reducing active inference energy by ``energy_cut``; idle power unchanged.

        Average power saved when each inference costs ``1 / energy_cut`` of its energy.

        Equation: ``(active_contribution * (1 - 1 / energy_cut)).to('mW')``."""
        return (self.active_contribution * (1 - 1 / self.energy_cut)).to("mW")

    @property
    def cut_gain(self) -> float:
        """Dimensionless fraction of average power removed by the active-energy reduction.

        Equation: ``(cut_saving / average_power).to('').magnitude``."""
        return (self.cut_saving / self.average_power).to("").magnitude


class TokenEnergyScenario(BaseScenario):
    """Joules per generated token: node power over aggregate token rate, well used and poorly used.

    The node draws the accelerator's power plus a host share whether a step
    carries one sequence or many. ``concurrent_requests`` at
    ``per_request_rate`` set the well-used token rate; at
    ``low_use_fraction`` of that rate the accelerator still draws
    ``low_use_accelerator_power``.

    Notes
    -----
    Equation: Per-token energy is attributed node watts divided by aggregate tokens per
    second, then multiplied by response token count for response joules.

    Assumptions: Power and throughput are steady within each declared utilization regime;
    host power is attributed equally to both regimes.

    Excludes: Idle transitions, a power trace, other tenants, and prediction of low-use
    power from throughput.

    References
    ----------
    Derivation: Per-token energy is attributed node watts divided by aggregate tokens per
    second, then multiplied by response token count for response joules. This is declared
    scenario accounting rather than a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.H100ChatEnergyPerToken
    >>> value = scenario.node_power
    >>> str(value.units)
    'W'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    accelerator_power: Quantity
    host_power: Quantity
    low_use_accelerator_power: Quantity
    concurrent_requests: int
    per_request_rate: Quantity  # tokens per second per request
    response_tokens: int
    low_use_fraction: float

    @property
    def node_power(self) -> Quantity:
        """Accelerator plus attributed host power in watts.

        Equation: ``(accelerator_power + host_power).to('watt')``.
        """
        return (self.accelerator_power + self.host_power).to("watt")

    @property
    def low_use_power(self) -> Quantity:
        """Low-use accelerator plus the same attributed host power, in watts.

        Equation: ``(low_use_accelerator_power + host_power).to('watt')``.
        """
        return (self.low_use_accelerator_power + self.host_power).to("watt")

    @property
    def token_rate(self) -> Quantity:
        """Aggregate token rate from concurrency times per-request token rate, in tokens per
        second.

        Equation: ``(concurrent_requests * per_request_rate).to('1/second')``.
        """
        return (self.concurrent_requests * self.per_request_rate).to("1/second")

    def energy_per_token(self, token_rate: Quantity, low_use: bool = False) -> Quantity:
        """Joules per token from selected node power divided by positive aggregate ``token_rate``.

        Equation: ``(power / token_rate).to('joule')``.
        """
        power = self.low_use_power if low_use else self.node_power
        return (power / token_rate).to("joule")

    @property
    def joules_per_token(self) -> Quantity:
        """Joules per token at the declared well-used aggregate rate.

        Equation: ``energy_per_token(token_rate)``.
        """
        return self.energy_per_token(self.token_rate)

    @property
    def response_energy(self) -> Quantity:
        """Joules allocated to one response at the well-used per-token energy.

        Equation: ``(joules_per_token * response_tokens).to('joule')``.
        """
        return (self.joules_per_token * self.response_tokens).to("joule")

    @property
    def low_use_token_rate(self) -> Quantity:
        """Aggregate tokens per second after applying the low-use throughput fraction.

        Equation: ``(token_rate * low_use_fraction).to('1/second')``.
        """
        return (self.token_rate * self.low_use_fraction).to("1/second")

    @property
    def low_use_joules_per_token(self) -> Quantity:
        """Joules per token under the separate low-use power and throughput assumptions.

        Equation: ``energy_per_token(low_use_token_rate, low_use=True)``.
        """
        return self.energy_per_token(self.low_use_token_rate, low_use=True)

    @property
    def low_use_ratio(self) -> float:
        """Low-use per-token energy divided by well-used per-token energy, dimensionless.

        Equation: ``(low_use_joules_per_token / joules_per_token).to('').magnitude``.
        """
        return (self.low_use_joules_per_token / self.joules_per_token).to("").magnitude


class OperationalCarbonScenario(BaseScenario):
    """Operational carbon = energy x PUE x grid intensity.

    Give the IT energy directly (``it_energy``) or as ``devices`` drawing ``device_power`` for
    ``duration``. ``pue`` is 1 when the energy is already counted at the facility boundary.
    ``comparison`` is an optional emissions anchor (mass per unit) to express the result in.

    Notes
    -----
    Equation: Facility kWh equal IT kWh times PUE; operational carbon is facility energy
    times supplied grid intensity.

    Assumptions: Direct IT energy takes precedence over ``count*power*duration``, and PUE is one
    for an already facility-level energy input.

    Excludes: Embodied emissions, time-varying grid intensity, and verification that the
    input energy boundary is correct.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.physics.quantities.carbon_from_energy``.
    Their contracts supply the underlying equations; property equations show the remaining
    unit conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Gpt3TrainingCarbon
    >>> value = scenario.it_energy_total
    >>> str(value.units)
    'kWh'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    intensity: Quantity
    pue: float = 1.0
    it_energy: Optional[Quantity] = None
    devices: Optional[float] = None
    device_power: Optional[Quantity] = None
    duration: Optional[Quantity] = None
    comparison: Optional[Quantity] = None

    @property
    def it_energy_total(self) -> Quantity:
        """IT energy in kWh, using the direct input or device count times power times duration.

        Equation: ``(devices * device_power * duration).to('kWh'); otherwise
        it_energy.to('kWh')``.
        """
        if self.it_energy is not None:
            return self.it_energy.to("kWh")
        return (self.devices * self.device_power * self.duration).to("kWh")

    @property
    def energy(self) -> Quantity:
        """Facility energy in kWh after multiplying IT energy by PUE.

        Equation: ``(it_energy_total * pue).to('kWh')``.
        """
        return (self.it_energy_total * self.pue).to("kWh")

    @property
    def carbon(self) -> Quantity:
        """Operational emissions in metric tonnes from facility energy times grid intensity.

        Equation: ``carbon_from_energy(energy, intensity).to('metric_ton')``.
        """
        from ...physics.quantities import carbon_from_energy

        return carbon_from_energy(self.energy, self.intensity).to("metric_ton")

    @property
    def comparisons(self) -> float:
        """Dimensionless equivalent count of the optional emissions comparison anchor.

        Equation: ``(carbon / comparison).to('').magnitude``.
        """
        return (self.carbon / self.comparison).to("").magnitude
