"""Physical-plant solvers: the stopping, actuation, heat, and power budgets of an
embodied machine (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). Each model takes stated inputs and
returns every intermediate a budget quotes, so a chapter, a slide deck, and a
lab read the same arithmetic. The equations themselves live in ``physics/``
(``robotics/stopping.py``, ``drivetrain.py``, ``motors.py``); these models only
compose them into the budgets of one machine.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

from ..results import (
    BrakingCeilingsResult,
    ContinuousDeratingResult,
    DriveBusResult,
    FreshnessProfileResult,
    GearedRotorResult,
    GearRatioCostResult,
    HubDriveElectricalResult,
    SpeedStepResult,
    StallDeratingResult,
    ThermalBurstResult,
    ThrottledStageResult,
    ZeroFailureResult,
)
from ...core.types import Quantity
from ...core.units import ureg
from ...physics.constants import COPPER_TEMPERATURE_COEFFICIENT, COPPER_COEFFICIENT_REFERENCE_TEMPERATURE
from ...physics.drivetrain import (
    calc_braking_limits,
    calc_drive_torque_deceleration_limit,
    calc_drive_voltage_headroom,
    calc_required_friction,
    calc_traction_deceleration_limit,
)
from ...physics.motors import (
    calc_copper_resistance_factor,
    calc_first_order_time_to_temperature,
    calc_inertia_matched_ratio,
    calc_peak_normalized_response,
    calc_periodic_burst_peak_rise,
)
from ...physics.robotics.actuation import calc_kinetic_energy, calc_reflected_inertia, calc_voltage_droop
from ...physics.robotics.assurance import calc_zero_failure_rate_upper_bound
from .base import ForwardModel

_MM = ureg.millimeter
_MS = ureg.millisecond
_K = ureg.kelvin


def _ratio(a, b) -> float:
    return float((a / b).to("").magnitude)


class SpeedStepModel(ForwardModel):
    """
    Braking travel added by raising the speed setpoint, and the delay that costs the same distance.

    Braking travel grows as v^2; the added travel divided by the base speed is
    the pre-brake delay that would cost the same distance at the base speed.
    """
    requires = ()
    produces = SpeedStepResult

    def solve(self, base_speed: Quantity, raised_speed: Quantity, deceleration: Quantity) -> SpeedStepResult:
        base = (base_speed ** 2 / (2 * deceleration)).to(_MM)
        raised = (raised_speed ** 2 / (2 * deceleration)).to(_MM)
        added = (raised - base).to(_MM)
        return SpeedStepResult(braking_growth=_ratio(raised, base) - 1, braking_added=added,
                               equivalent_delay=(added / base_speed).to(_MS))


class FreshnessProfileModel(ForwardModel):
    """
    A moving target's freshness deadline against a capture-to-onset age profile.

    The deadline is the tolerance left after the initial error, divided by the
    target speed: Delta t_fresh = (r_tol - e_0) / v. Stage percentiles are
    summed per percentile; the target's travel at the median chain and at the
    system tail age is compared with the allowed error.
    """
    requires = ()
    produces = FreshnessProfileResult

    def solve(self, target_speed: Quantity, tolerance: Quantity, initial_error: Quantity,
              stages: Sequence[Tuple[str, Quantity, Quantity, Quantity]], focus_stage: str,
              tail_age: Quantity) -> FreshnessProfileResult:
        allowed = (tolerance - initial_error).to(_MM)
        p50 = sum((s[1] for s in stages), 0 * _MS).to(_MS)
        p99 = sum((s[2] for s in stages), 0 * _MS).to(_MS)
        pmax = sum((s[3] for s in stages), 0 * _MS).to(_MS)
        focus = next(s[1] for s in stages if s[0] == focus_stage).to(_MS)
        t50 = (target_speed * p50).to(_MM)
        ttail = (target_speed * tail_age).to(_MM)
        return FreshnessProfileResult(
            allowed_error=allowed, deadline=(allowed / target_speed).to(_MS),
            stage_sum_p50=p50, stage_sum_p99=p99, stage_sum_max=pmax,
            focus_stage=focus, rest_of_chain=(p50 - focus).to(_MS), focus_share=_ratio(focus, p50),
            focus_rate=(1 / focus).to(ureg.hertz),
            travel_p50=t50, travel_tail=ttail, slack_p50=(allowed - t50).to(_MM), overrun_tail=(ttail - allowed).to(_MM),
        )


class BrakingCeilingsModel(ForwardModel):
    """
    The three ceilings on a credible deceleration (traction, tip-over, drive torque) and back-EMF headroom.

    Composes ``calc_braking_limits`` for a wheeled base driven by direct-drive
    hub motors at a configured current limit, adds the rated-current torque
    ceiling and the inspected-floor traction ceiling (all weight on the driven
    wheels), and checks the terminal voltage braking and driving need at speed.
    """
    requires = ()
    produces = BrakingCeilingsResult

    def solve(self, mass: Quantity, deceleration: Quantity, speed: Quantity, hub, n_wheels: int,
              current_limit: Quantity, drive_load_share: float, mu_dry: float, mu_inspected: float,
              bus_low: Quantity, cg_height: Optional[Quantity] = None,
              tip_arm: Optional[Quantity] = None) -> BrakingCeilingsResult:
        r = hub.wheel_radius
        tau_wheel = (hub.torque_constant * current_limit).to(ureg.newton * ureg.meter)
        m_rot = (n_wheels * hub.rotating_mass_bound).to(ureg.kg)
        lim = calc_braking_limits(mass, deceleration, wheel_torque_max=tau_wheel, wheel_radius=r,
                                  n_driven_wheels=n_wheels, friction_coefficient=mu_dry,
                                  drive_load_share=drive_load_share, rotating_equivalent_mass=m_rot,
                                  cg_height=cg_height, tip_arm=tip_arm)
        a_rated = calc_drive_torque_deceleration_limit(hub.rated_torque, r, n_wheels, mass, m_rot)["a_max"]
        volts = calc_drive_voltage_headroom(hub.back_emf_constant, speed, r, bus_low, current_limit,
                                            hub.line_resistance)
        return BrakingCeilingsResult(
            wheel_torque=tau_wheel, force=lim["force"], rotating_mass=m_rot, rotating_share=_ratio(m_rot, mass),
            m_eff=lim["m_eff"], a_torque=lim["a_torque"], torque_margin=_ratio(lim["a_torque"], deceleration) - 1,
            a_torque_rated=a_rated, mu_required=calc_required_friction(deceleration),
            drive_load_share=drive_load_share, mu_required_rated=lim["mu_required"],
            min_load_share_dry=_ratio(deceleration, mu_dry * ureg.standard_gravity),
            a_traction_dry=lim["a_traction"], a_traction_inspected=calc_traction_deceleration_limit(mu_inspected),
            cg_ratio_max=lim["cg_ratio_max"], a_tip=lim["a_tip"], passes=lim["passes"],
            wheel_speed=volts["omega"], back_emf=volts["e_bemf"], v_brake=volts["v_brake"],
            v_drive=volts["v_drive"], headroom=volts["headroom"],
        )


class HubDriveElectricalModel(ForwardModel):
    """
    Electrical time constant, voltage-limited current ramp, and phase-short braking peak of a hub drive.

    tau_e = L / R (line quantities, the same ratio as per phase for a star
    winding); at stall the back-EMF is zero, so the full bus voltage ramps the
    current at V / L: t = L I / V. Shorted windings brake hardest where the
    winding reactance equals its resistance: omega_e = R / L, which is
    omega_e / p at the wheel for p pole pairs.
    """
    requires = ()
    produces = HubDriveElectricalResult

    def solve(self, hub, current_limit: Quantity, bus_voltage: Quantity, brake_onset: Quantity) -> HubDriveElectricalResult:
        l_, r_ = hub.line_inductance, hub.line_resistance
        ramp = (l_ * current_limit / bus_voltage).to(_MS)
        full = (bus_voltage / r_).to(ureg.ampere)
        w_e = (r_ / l_).to(ureg.radian / ureg.second)
        return HubDriveElectricalResult(
            time_constant=(l_ / r_).to(_MS), ramp_time=ramp, ramp_share=_ratio(ramp, brake_onset),
            full_step_current=full, step_over_peak=_ratio(full, hub.peak_current),
            short_peak_electrical=w_e, short_peak_wheel_speed=(w_e / hub.pole_pairs).to(ureg.revolution / ureg.minute),
        )


class GearedRotorModel(ForwardModel):
    """
    Reflected rotor inertia N^2 J_rotor, sized as a solid flywheel, and the inertia match.

    A solid disk of radius r has J = m r^2 / 2, so m = 2 J / r^2. Joint
    acceleration for a fixed motor torque peaks at N* = sqrt(J_load / J_rotor);
    at N it is 2x / (1 + x^2) of that peak, x = N / N*.
    """
    requires = ()
    produces = GearedRotorResult

    def solve(self, gear_ratio: float, rotor_mass: Quantity, rotor_inertia: Quantity,
              flywheel_radius: Quantity, load_inertia: Quantity) -> GearedRotorResult:
        j_ref = calc_reflected_inertia(gear_ratio, rotor_inertia).to(ureg.kg * ureg.meter ** 2)
        n_star = calc_inertia_matched_ratio(load_inertia, rotor_inertia)
        return GearedRotorResult(
            reflected_inertia=j_ref, flywheel_mass=(2 * j_ref / flywheel_radius ** 2).to(ureg.kg),
            radius_of_gyration=((rotor_inertia / rotor_mass) ** 0.5).to(_MM), matched_ratio=n_star,
            acceleration_share=calc_peak_normalized_response(gear_ratio / n_star),
        )


class GearRatioCostModel(ForwardModel):
    """
    Holding heat and reflected inertia of one motor holding one joint torque, per gear ratio.

    Motor torque tau_m = tau_out / (eta N); copper loss (tau_m / K_m)^2 with
    K_m = K_t / sqrt(R_loss); reflected inertia N^2 J_in. Their product is
    tau_out^2 J_in / (eta^2 K_m^2), independent of N.
    """
    requires = ()
    produces = GearRatioCostResult

    def solve(self, motor, loss_resistance: Quantity, input_inertia: Quantity, output_torque: Quantity,
              gear_efficiency: float, ratios: Iterable[int]) -> GearRatioCostResult:
        k_m = (motor.torque_constant / loss_resistance ** 0.5).to(ureg.newton * ureg.meter / ureg.watt ** 0.5)
        tm, loss, jr, prod, over = {}, {}, {}, {}, {}
        for n in ratios:
            tm[n] = (output_torque / (gear_efficiency * n)).to(ureg.newton * ureg.meter)
            loss[n] = ((tm[n] / k_m) ** 2).to(ureg.watt)
            jr[n] = (n ** 2 * input_inertia).to(ureg.kg * ureg.meter ** 2)
            prod[n] = (loss[n] * jr[n]).to(ureg.watt * ureg.kg * ureg.meter ** 2)
            over[n] = _ratio(tm[n], motor.peak_torque)
        return GearRatioCostResult(motor_constant=k_m, motor_torque=tm, copper_loss=loss, reflected_inertia=jr,
                                   loss_inertia_product=prod, over_peak=over,
                                   peak_over_rated=_ratio(motor.peak_torque, motor.rated_torque))


class ThermalBurstModel(ForwardModel):
    """
    Burst horizons from a cold and a heat-soaked start, the average-power duty screen, and the periodic peak.

    First-order lumped winding at fixed resistance: tau = theta C; the burst
    drives the winding toward T_amb + P theta; the duty screen holds the average
    loss to (T_ceiling - T_amb) / theta; the off-time is rounded up to 0.1 s and
    the resulting schedule's periodic peak is checked against the ceiling.
    """
    requires = ()
    produces = ThermalBurstResult

    def solve(self, rated_current: Quantity, loss_resistance: Quantity, thermal_capacitance: Quantity,
              thermal_resistance: Quantity, ambient: Quantity, limit: Quantity, ceiling: Quantity,
              burst_factor: float, burst_on: Quantity) -> ThermalBurstResult:
        tau = (thermal_resistance * thermal_capacitance).to(ureg.second)
        amb = ambient.to(_K)
        p_rated = (rated_current ** 2 * loss_resistance).to(ureg.watt)
        t_eq = (amb + p_rated * thermal_resistance).to(_K)
        i_burst = (burst_factor * rated_current).to(ureg.ampere)
        p_burst = (i_burst ** 2 * loss_resistance).to(ureg.watt)
        t_inf = (amb + p_burst * thermal_resistance).to(_K)
        cold = calc_first_order_time_to_temperature(amb, limit, t_inf, tau)
        hot = calc_first_order_time_to_temperature(t_eq, limit, t_inf, tau)
        p_avg = ((ceiling.to(_K) - amb) / thermal_resistance).to(ureg.watt)
        duty = _ratio(p_avg, p_burst)
        off = (burst_on * (1 - duty) / duty).to(ureg.second)
        off_sched = math.ceil(off.magnitude * 10) / 10 * ureg.second   # screening value rounded up to 0.1 s
        peak = (amb + calc_periodic_burst_peak_rise(p_burst, thermal_resistance, burst_on, burst_on + off_sched, tau)).to(_K)
        return ThermalBurstResult(
            time_constant=tau, rated_loss=p_rated, rated_equilibrium=t_eq, burst_current=i_burst, burst_loss=p_burst,
            asymptote=t_inf, horizon_cold=cold, horizon_hot=hot, hot_cut=1 - _ratio(hot, cold),
            average_loss_max=p_avg, duty=duty, off_time=off, off_time_scheduled=off_sched, periodic_peak=peak,
        )


class StallDeratingModel(ForwardModel):
    """
    Steady stall temperature with copper's resistance feedback, and the hold the winding can sustain.

    Steady state: dT = I^2 R_0 [1 + alpha (T_amb - T_0 + dT)] theta, linear in
    dT, so dT = c / (1 - g) with loop gain g = I^2 R_0 theta alpha (runaway if
    g >= 1). The sustainable loss at the limit is (T_limit - T_amb) / theta at
    the hot resistance R(T_limit).
    """
    requires = ()
    produces = StallDeratingResult

    def solve(self, hold_torque: Quantity, gear_ratio: float, gear_efficiency: float, torque_constant: Quantity,
              loss_resistance: Quantity, thermal_resistance: Quantity, ambient: Quantity,
              limit: Quantity) -> StallDeratingResult:
        alpha, t0 = COPPER_TEMPERATURE_COEFFICIENT, COPPER_COEFFICIENT_REFERENCE_TEMPERATURE
        tau_m = (hold_torque / (gear_efficiency * gear_ratio)).to(ureg.newton * ureg.meter)
        i = (tau_m / torque_constant).to(ureg.ampere)
        p_cold = (i ** 2 * loss_resistance).to(ureg.watt)
        base = (i ** 2 * loss_resistance * thermal_resistance).to(_K)
        const = (base * calc_copper_resistance_factor(ambient, alpha, t0)).to(_K)
        gain = float((base * alpha).to("").magnitude)
        rise = (const / (1 - gain)).to(_K)
        p_max = ((limit.to(_K) - ambient.to(_K)) / thermal_resistance).to(ureg.watt)
        r_hot = (loss_resistance * calc_copper_resistance_factor(limit, alpha, t0)).to(ureg.ohm)
        i_max = ((p_max / r_hot) ** 0.5).to(ureg.ampere)
        hold = (gear_efficiency * gear_ratio * torque_constant * i_max).to(ureg.newton * ureg.meter)
        return StallDeratingResult(
            motor_torque=tau_m, stall_current=i, cold_loss=p_cold, feedback_constant=const, feedback_gain=gain,
            steady_rise=rise, winding_temperature=(ambient.to(_K) + rise).to(_K), sustainable_loss=p_max,
            hot_resistance=r_hot, sustainable_current=i_max, sustainable_hold=hold, derating=1 - _ratio(hold, hold_torque),
        )


class ContinuousDeratingModel(ForwardModel):
    """
    Installed continuous torque relative to the catalog rating.

    Continuous torque scales as sqrt((T_max - T_amb) / theta): a hotter
    ambient and a larger installed thermal resistance each cut it.
    """
    requires = ()
    produces = ContinuousDeratingResult

    def solve(self, ceiling: Quantity, catalog_ambient: Quantity, installed_ambient: Quantity,
              thermal_resistance_factor: float) -> ContinuousDeratingResult:
        amb = _ratio(ceiling.to(_K) - installed_ambient.to(_K), ceiling.to(_K) - catalog_ambient.to(_K)) ** 0.5
        return ContinuousDeratingResult(ambient_share=amb, combined_share=amb / thermal_resistance_factor ** 0.5)


class DriveBusModel(ForwardModel):
    """
    A shared drive bus: sustained sag under a coincident load step, and the regenerative surge of a stop.

    Sag Delta V = Delta I R_bus against the converter lockout; the step current
    that would spend the depleted pack's headroom is (V_low - V_uvlo) / R_bus.
    A stop from speed v returns a share of (1/2) m v^2; into the bus capacitance
    alone, V_final = sqrt(V_full^2 + 2 E / C). Speed falls linearly at constant
    deceleration, so regenerated power falls linearly from 2 E / t_stop, and a
    chopper resistor at the clamp must satisfy R <= V_clamp^2 / P_peak.
    """
    requires = ()
    produces = DriveBusResult

    def solve(self, mass: Quantity, speed: Quantity, deceleration: Quantity, step_current: Quantity,
              bus_resistance: Quantity, full: Quantity, nominal: Quantity, depleted: Quantity, lockout: Quantity,
              compute_power: Quantity, converter_efficiency: float, capacitance: Quantity, clamp: Quantity,
              regen_fraction: float) -> DriveBusResult:
        i_compute = (compute_power / (converter_efficiency * nominal)).to(ureg.ampere)
        sag = calc_voltage_droop(step_current, bus_resistance).to(ureg.volt)
        e_k = calc_kinetic_energy(mass, speed).to(ureg.joule)
        e_regen = (regen_fraction * e_k).to(ureg.joule)
        t_stop = (speed / deceleration).to(_MS)
        p_peak = (2 * e_regen / t_stop).to(ureg.watt)
        return DriveBusResult(
            compute_current=i_compute, drive_current=(step_current - i_compute).to(ureg.ampere), sag=sag,
            sustained_full=(full - sag).to(ureg.volt), sustained_depleted=(depleted - sag).to(ureg.volt),
            margin_depleted=(depleted - sag - lockout).to(ureg.volt), headroom_depleted=(depleted - lockout).to(ureg.volt),
            crossing_current=((depleted - lockout) / bus_resistance).to(ureg.ampere),
            kinetic_energy=e_k, regenerated=e_regen,
            capacitor_only_endpoint=((full ** 2 + 2 * e_regen / capacitance) ** 0.5).to(ureg.volt),
            capacitor_to_clamp=(0.5 * capacitance * (clamp ** 2 - full ** 2)).to(ureg.joule),
            stop_time=t_stop, peak_power=p_peak, resistor_max=(clamp ** 2 / p_peak).to(ureg.ohm),
        )


class ZeroFailureModel(ForwardModel):
    """Upper bound on the per-trial failure rate after a failure-free campaign: -ln(1 - C) / n."""
    requires = ()
    produces = ZeroFailureResult

    def solve(self, trials: int, confidence: float) -> ZeroFailureResult:
        p = calc_zero_failure_rate_upper_bound(trials, confidence)
        return ZeroFailureResult(coefficient=-math.log(1 - confidence), rate_upper=p, one_in=1 / p)


class ThrottledStageModel(ForwardModel):
    """A compute-bound stage at a fraction of full clock: its time scales by the inverse clock ratio,
    and the added age is paid as travel at the stated speed."""
    requires = ()
    produces = ThrottledStageResult

    def solve(self, stage_time: Quantity, clock_ratio: float, speed: Quantity) -> ThrottledStageResult:
        slow = (stage_time / clock_ratio).to(_MS)
        added = (slow - stage_time).to(_MS)
        return ThrottledStageResult(throttled=slow, added_age=added, added_travel=(speed * added).to(_MM))
