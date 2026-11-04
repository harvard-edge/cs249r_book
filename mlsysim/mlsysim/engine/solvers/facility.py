"""Facility solvers: power transients, coolant flow, and outage cost.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

from typing import Dict

from ..results import CoolantFlowResult, OutageCostResult, PowerTransientResult
from ...core.types import Quantity
from ...physics.constants import (
    AIR_DENSITY_SEA_LEVEL,
    AIR_SPECIFIC_HEAT,
    WATER_DENSITY,
    WATER_SPECIFIC_HEAT,
)
from ...physics.power import calc_inductive_droop, calc_rail_current_step
from ...physics.thermal import calc_coolant_mass_flow
from .base import ForwardModel

_COOLANTS = {
    "air": (AIR_SPECIFIC_HEAT, AIR_DENSITY_SEA_LEVEL),
    "water": (WATER_SPECIFIC_HEAT, WATER_DENSITY),
}


class PowerTransientModel(ForwardModel):
    """
    One accelerator's load step and the ramp a synchronized cluster produces.

    When a collective barrier releases every device from communication into
    dense arithmetic, each one steps from its communication-phase power toward
    its design power at once. On one device's core rail the step is a current
    change ``dP / V``, and the board loop's inductance turns a fast edge into a
    voltage droop ``L dI/dt``. Across the cluster the same step, aligned by the
    barrier, is a power ramp the facility supplies must follow.
    """
    requires = ("hardware",)
    produces = PowerTransientResult

    def solve(self, high_power: Quantity, low_power: Quantity, rail_voltage: Quantity,
              loop_inductance: Quantity, rise_times: Dict[str, Quantity],
              n_devices: int, ramp_window: Quantity) -> PowerTransientResult:
        """Calculate device voltage droop and a synchronized fleet power ramp.

        Parameters
        ----------
        high_power : Quantity
            Power after the step in W; at least low_power.
        low_power : Quantity
            Power before the step in W; nonnegative.
        rail_voltage : Quantity
            Positive core supply voltage in V.
        loop_inductance : Quantity
            Nonnegative supply-loop inductance in H.
        rise_times : dict of str to Quantity
            Named positive current-rise times in seconds.
        n_devices : int
            Nonnegative number of synchronized devices.
        ramp_window : Quantity
            Positive facility ramp duration in seconds.

        Returns
        -------
        PowerTransientResult
            Device step in W and A, droop in mV and voltage fractions, fleet step in kW, and
            ramp in GW/s.

        See Also
        --------
        mlsysim.physics.power.calc_rail_current_step : Compute current change from a power step.
        mlsysim.physics.power.calc_inductive_droop : Compute ideal inductive voltage droop.

        Notes
        -----
        Equation: dP = high_power - low_power; dI = dP / V; droop = L dI / dt; fleet ramp =
        n_devices dP / ramp_window.

        Assumptions: All devices share one power step and change state together. Each rise time
        is a separate scenario.

        Excludes: Regulator feedback, capacitance, parasitic resistance, and facility supply
        dynamics. A droop estimate is not a board safety assessment.

        References
        ----------
        Derivation: Power P = VI gives the current change at fixed rail voltage; the ideal
        inductor relation V = L dI/dt gives droop magnitude. Summing identical load steps gives
        the fleet ramp.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> r = PowerTransientModel().solve(Q_(200, "W"), Q_(100, "W"),
        ...     Q_(1, "V"), Q_(1, "nH"), {"edge": Q_(1, "us")}, 2, Q_(1, "s"))
        >>> r.current_step == Q_(100, "A")
        True
        """
        step = (high_power - low_power).to("watt")
        di = calc_rail_current_step(step, rail_voltage)
        droop = {k: calc_inductive_droop(loop_inductance, di, t).to("millivolt") for k, t in rise_times.items()}
        share = {k: (d / rail_voltage).to("").magnitude for k, d in droop.items()}
        cluster_step = (n_devices * step).to("kilowatt")
        return PowerTransientResult(
            power_step=step, current_step=di, droop=droop, droop_share=share,
            cluster_power_step=cluster_step,
            ramp_rate=(cluster_step / ramp_window).to("GW/second"),
        )


class CoolantFlowModel(ForwardModel):
    """
    Coolant flow that carries a heat load at a stated temperature rise.

    ``m_dot = Q / (c_p dT)``, then volume flow from the coolant's density.
    Water's specific heat is about four times air's and its density about
    830 times, so it carries the same heat in thousands of times less volume.
    """
    requires = ("heat_load",)
    produces = CoolantFlowResult

    def solve(self, heat_load: Quantity, temperature_rise: Quantity, coolant: str = "air") -> CoolantFlowResult:
        """Calculate coolant mass and volume flow for a steady heat load.

        Parameters
        ----------
        heat_load : Quantity
            Nonnegative heat delivered to the coolant in W.
        temperature_rise : Quantity
            Positive inlet-to-outlet temperature difference in K.
        coolant : str, optional
            Either "air" or "water"; default "air" selects its registered density and specific heat.

        Returns
        -------
        CoolantFlowResult
            Required mass flow in kg/s and volume flow in L/s.

        See Also
        --------
        mlsysim.physics.thermal.calc_coolant_mass_flow : Apply the sensible-heat energy balance.

        Notes
        -----
        Equation: mass_flow = heat_load / (specific_heat * temperature_rise); volume_flow =
        mass_flow / density.

        Assumptions: Steady sensible heat transport and constant coolant properties from
        physics.constants.

        Excludes: Pump or fan power, latent heat, pressure losses, and heat-exchanger effectiveness.

        References
        ----------
        Derivation: The steady energy balance Q_dot = m_dot c_p delta_T sets mass flow; dividing
        by density converts mass flow to volume flow.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> a = CoolantFlowModel().solve(Q_(1, "kW"), Q_(10, "K"), "water")
        >>> b = CoolantFlowModel().solve(Q_(2, "kW"), Q_(10, "K"), "water")
        >>> b.mass_flow == 2 * a.mass_flow
        True
        """
        cp, rho = _COOLANTS[coolant]
        m = calc_coolant_mass_flow(heat_load, cp, temperature_rise)
        return CoolantFlowResult(mass_flow=m, volume_flow=(m / rho).to("liter/second"))


class OutageCostModel(ForwardModel):
    """
    Idle cost of a failure domain going down during a synchronous job.

    The direct cost counts only the devices in the failed domain. A
    synchronous job that cannot continue idles every device, and rolling back
    to the last checkpoint adds the lost work to the repair window.
    """
    requires = ("fleet",)
    produces = OutageCostResult

    def solve(self, failed_devices: int, job_devices: int, repair_time: Quantity,
              device_hour_rate: Quantity, rollback_times: Dict[str, Quantity] = None) -> OutageCostResult:
        """Price repair downtime for a failed domain and its synchronous job.

        Parameters
        ----------
        failed_devices : int
            Nonnegative number of failed devices, no greater than job_devices.
        job_devices : int
            Nonnegative number of devices reserved by the job.
        repair_time : Quantity
            Nonnegative repair duration in hours or compatible time units.
        device_hour_rate : Quantity
            Nonnegative rental or opportunity-cost rate in USD/hour per device.
        rollback_times : dict of str to Quantity, optional
            Named nonnegative lost-work durations added to repair time; None omits rollback
            scenarios.

        Returns
        -------
        OutageCostResult
            Direct failed-device cost, whole-job idle cost, and each rollback-adjusted job cost,
            as USD scalars.

        Notes
        -----
        Equation: direct_cost = failed_devices * rate * repair_hours; job_cost = job_devices *
        rate * repair_hours; rollback_cost[k] = job_devices * rate * (repair_hours +
        rollback_hours[k]).

        Assumptions: The complete synchronous job is idle during repair, and every reserved
        device has the same hourly cost.

        Excludes: Redundant operation, partial progress, revenue loss, and replacement equipment
        cost.

        References
        ----------
        Derivation: Device-hours equal the number of reserved devices times unavailable hours.
        Multiplication by the hourly rate gives each cost.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> r = OutageCostModel().solve(1, 4, Q_(2, "hour"), Q_(3, "USD/hour"))
        >>> r.job_cost == 4 * r.direct_cost
        True
        """
        rate = device_hour_rate.to("dollar/hour").magnitude
        hours = repair_time.to("hour").magnitude
        rollback = {k: job_devices * rate * (hours + t.to("hour").magnitude)
                    for k, t in (rollback_times or {}).items()}
        return OutageCostResult(
            direct_cost=failed_devices * rate * hours,
            job_cost=job_devices * rate * hours,
            job_cost_with_rollback=rollback,
        )
