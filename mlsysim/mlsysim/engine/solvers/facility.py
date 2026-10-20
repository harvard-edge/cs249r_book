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
        rate = device_hour_rate.to("dollar/hour").magnitude
        hours = repair_time.to("hour").magnitude
        rollback = {k: job_devices * rate * (hours + t.to("hour").magnitude)
                    for k, t in (rollback_times or {}).items()}
        return OutageCostResult(
            direct_cost=failed_devices * rate * hours,
            job_cost=job_devices * rate * hours,
            job_cost_with_rollback=rollback,
        )
