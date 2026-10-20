#!/usr/bin/env python3
"""Every number on the Lecture 2 slides and notes (The Physical Body), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* machine and its site record, the
Actuators.* parts, Literature.*) and the calculations (Scenarios.*, each backed
by a solver in engine/solvers/plant.py); the chapter reads the same scenarios.
This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex. make_figs.py imports
it for the scenario objects.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Embodied, Actuators, Scenarios, Literature = (mlsysim.Embodied, mlsysim.Actuators,
                                              mlsysim.Scenarios, mlsysim.Literature)
ureg = mlsysim.core.units.ureg

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
HUB = RM.drive_unit
MOTOR = Actuators.FramelessMotor.TQ_ILM70x18
JOINT = Actuators.PandaClassJoint.CSG_25_50_ILM70x18
ESA = Literature.SpaceflightAnomalies

GRASP = Scenarios.WristCameraGraspAgeProfile
BODY = Scenarios.WarehouseBaseStopBodyTerms
FULL = Scenarios.WarehouseBaseStopFullDelay
FAST = Scenarios.WarehouseBaseStopFastSetpoint
INSPECTED = Scenarios.WarehouseBaseStopInspectedFloor
STEP = Scenarios.WarehouseBaseSpeedStep
CEIL = Scenarios.WarehouseBaseBrakingCeilings
ELEC = Scenarios.WarehouseBaseHubDriveElectrical
ROTOR = Scenarios.GearedRotorFlywheel
GEARS = Scenarios.PandaClassJointGearRatios
BURST = Scenarios.PandaClassJointBurst
STALL = Scenarios.PandaClassElbowStall
DERATE = Scenarios.SealedJointCatalogDerating
BUS = Scenarios.WarehouseDriveBusStop
CAMPAIGN = Scenarios.CleanLoadedStopCampaign
THROTTLE = Scenarios.ThrottledNavigationBackbone


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def degc(q):
    """A temperature in degrees Celsius."""
    return q.to(ureg.degC).magnitude


print("== Hook: Schiaparelli (ESA inquiry, 2017)")
v("esa_sat_ms", ESA.SchiaparelliSaturationAcceptanceMs, "{:.0f}", "saturation accepted, ms")
v("esa_drift_deg", ESA.SchiaparelliAttitudeDriftDeg, "{:.0f}", "attitude drift, degrees")
v("esa_alt_km", ESA.SchiaparelliReleaseAltitudeKm, "{:.1f}", "release altitude, km")
v("esa_burn_s", ESA.SchiaparelliThrusterBurnS, "{:.0f}", "thruster burn, s")
v("esa_plan_s", ESA.SchiaparelliPlannedBurnS, "{:.0f}", "planned burn, s")
v("esa_impact_kmh", ESA.SchiaparelliImpactKmh, "{:.0f}", "impact, km/h")
v("esa_impact_ms", m(ESA.SchiaparelliImpactKmh * ureg("km/hour"), "m/s"), "{:.0f}", "impact, m/s (derived)")

print("== The running machine")
v("mass", m(RM.loaded_mass, "kg"), "{:.0f}", "loaded mass, kg (upper bound)")
v("v_drive", m(RM.base.max_velocity, "m/s"), "{:.1f}", "drive limit, m/s")
v("ke", m(BUS.result.kinetic_energy, "J"), "{:.0f}", "kinetic energy at the drive limit, J")
v("a_brake", m(AISLE.a_brake, "m/s**2"), "{:.0f}", "credible deceleration, m/s^2")
v("a_brake_1", m(AISLE.a_brake, "m/s**2"), "{:.1f}", "the same, one decimal")
v("d_clear_mm", m(AISLE.d_clear, "mm"), "{:,.0f}", "clear distance, mm")
v("d_clear_m", m(AISLE.d_clear, "m"), "{:.2f}", "clear distance, m")
v("d_clear_m1", m(AISLE.d_clear, "m"), "{:.1f}", "the same, one decimal (figure label)")
v("d_loc_mm", m(AISLE.delta_loc, "mm"), "{:.0f}", "localization bound, mm")
v("d_margin_mm", m(AISLE.delta_margin, "mm"), "{:.0f}", "protective clearance, mm")
v("t_onset_ms", m(AISLE.t_brake_onset, "ms"), "{:.0f}", "brake onset, ms")
v("tau_delay_ms", m(AISLE.tau_delay, "ms"), "{:.1f}", "full pre-brake delay, ms")
v("tau_delay_s", m(AISLE.tau_delay, "s"), "{:.4f}", "the same, s")

print("== Recall: the elementary stop (full delay, no allowances)")
full = FULL.result
v("blind_full", m(full.blind_travel, "mm"), "{:.1f}", "blind travel, mm")
v("brake_mm", m(full.braking_travel, "mm"), "{:.1f}", "braking travel at the drive limit, mm")
v("rest_full", m(full.travel_to_rest, "mm"), "{:.1f}", "travel to rest, mm")
v("two_a", m(FULL.speed ** 2 / full.braking_travel, "m/s**2"), "{:.0f}", "2 a_brake, m/s^2")
v("crossover", m(full.crossover_speed, "m/s"), "{:.2f}", "crossover speed 2 a tau, m/s")

print("== Freshness: the grasp age profile (illustrative)")
g = GRASP.result
stages = {name: (p50, p99, mx) for name, p50, p99, mx in GRASP.stages}
v("st_exposure", m(stages["Exposure and readout"][0], "ms"), "{:.1f}", "exposure and readout P50, ms")
v("st_transport", m(stages["Transport and DMA"][0], "ms"), "{:.1f}", "transport and DMA P50, ms")
v("st_backbone", m(g.focus_stage, "ms"), "{:.1f}", "backbone and policy P50, ms")
v("st_sched", m(stages["Scheduling and IPC"][0], "ms"), "{:.1f}", "scheduling and IPC P50, ms")
v("st_fieldbus", m(stages["Fieldbus"][0], "ms"), "{:.0f}", "fieldbus P50, ms")
v("st_current", m(stages["Current rise"][0], "ms"), "{:.1f}", "current rise P50, ms")
v("p50_sum", m(g.stage_sum_p50, "ms"), "{:.0f}", "median chain, ms")
v("p99_sum", m(g.stage_sum_p99, "ms"), "{:.0f}", "stage P99 sum, ms")
v("p99_sys", m(GRASP.system_p99, "ms"), "{:.1f}", "system P99, ms")
v("p999_sys", m(GRASP.system_p999, "ms"), "{:.1f}", "system P99.9, ms")
v("backbone_hz", m(g.focus_rate, "Hz"), "{:.1f}", "loop rate the benchmark suggests, Hz")
v("backbone_pct", 100 * g.focus_share, "{:.0f}", "backbone share of the median, %")
v("belt", m(AISLE.v_conveyor, "m/s"), "{:.2f}", "conveyor speed, m/s")
v("tol_mm", m(AISLE.grasp_tolerance, "mm"), "{:.0f}", "grasp tolerance, mm")
v("err0_mm", m(AISLE.grasp_initial_error, "mm"), "{:.0f}", "initial error, mm")
v("emax_mm", m(g.allowed_error, "mm"), "{:.0f}", "allowed error, mm")
v("deadline", m(g.deadline, "ms"), "{:.0f}", "freshness deadline, ms")
v("mug_p50", m(g.travel_p50, "mm"), "{:.0f}", "mug travel at the median, mm")
v("mug_slack", m(g.slack_p50, "mm"), "{:.0f}", "median slack, mm")
v("mug_tail", m(g.travel_tail, "mm"), "{:.1f}", "mug travel at P99.9, mm")
v("mug_over", m(g.overrun_tail, "mm"), "{:.1f}", "tail overrun, mm")

print("== Stopping (equation 2, Napkin Math 2.2)")
body = BODY.result
fast = FAST.result
v("overhead", m(body.overhead, "mm"), "{:.0f}", "allowances, mm")
v("onset_mm", m(body.blind_travel, "mm"), "{:.0f}", "travel during the brake onset, mm")
v("stop_body", m(body.stopping_distance, "mm"), "{:.1f}", "body-owned terms, mm")
v("left_body", m(body.clearance_left, "mm"), "{:.1f}", "left for other delays, mm")
v("stop_full", m(full.stopping_distance, "mm"), "{:.1f}", "full-delay stop, mm")
v("left_full", m(full.clearance_left, "mm"), "{:.1f}", "full-delay margin, mm")
v("v_fast", m(FAST.speed, "m/s"), "{:.0f}", "faster setpoint, m/s")
v("stop_fast", m(fast.stopping_distance, "mm"), "{:,.1f}", "stop at the faster setpoint, mm")
v("breach_fast", m(-fast.clearance_left, "mm"), "{:.1f}", "breach, mm")
v("blind_fast", m(fast.blind_travel, "mm"), "{:.1f}", "blind travel at the faster setpoint, mm")
v("brake_fast", m(fast.braking_travel, "mm"), "{:,.0f}", "braking at the faster setpoint, mm")
v("brake_fast_m", m(fast.braking_travel, "m"), "{:.0f}", "the same, m")
v("brake_growth", 100 * STEP.result.braking_growth, "{:.1f}", "braking growth, %")
v("brake_added", m(STEP.result.braking_added, "mm"), "{:.1f}", "braking added, mm")
v("dt_equiv", m(STEP.result.equivalent_delay, "ms"), "{:.1f}", "equivalent delay at the drive limit, ms")
v("dt_equiv_s", m(STEP.result.equivalent_delay, "s"), "{:.1f}", "the same, s")
v("speed_up_pct", 100 * (ratio(FAST.speed, BODY.speed) - 1), "{:.0f}", "faster setpoint over the drive limit, %")
v("dt_step", m(BODY.added_delay, "ms"), "{:.0f}", "delay step, ms")
v("dt_step_mm", m(body.added_delay_travel, "mm"), "{:.0f}", "travel per delay step, mm")
v("v_mm_per_ms", m(BODY.speed, "mm/ms"), "{:.1f}", "drive limit, mm per ms")

print("== Velocity ceiling (equation 3)")
v("vmax_body", m(body.speed_ceiling, "m/s"), "{:.2f}", "body-only ceiling, m/s")
v("vmax_full", m(full.speed_ceiling, "m/s"), "{:.2f}", "full-delay ceiling, m/s")
v("dneed_full", m(full.stopping_distance, "m"), "{:.3f}", "clear distance the drive limit needs, m")
v("mu_insp", AISLE.mu_inspected_floor, "{:.2f}", "inspected floor friction")
v("a_insp", m(INSPECTED.a_brake, "m/s**2"), "{:.2f}", "inspected floor ceiling, m/s^2")
v("vmax_insp", m(INSPECTED.result.speed_ceiling, "m/s"), "{:.2f}", "inspected-floor ceiling, m/s")

print("== Credible deceleration (Napkin Math 2.3)")
c = CEIL.result
v("hub_tau_peak", m(c.wheel_torque, "N*m"), "{:.0f}", "torque per wheel at the current limit, N m")
v("hub_i_peak", m(RM.drive_current_limit, "A"), "{:.0f}", "current limit, A")
v("hub_tau_rated", m(HUB.rated_torque, "N*m"), "{:.0f}", "rated torque, N m")
v("hub_i_rated", m(HUB.rated_current, "A"), "{:.1f}", "rated current, A")
v("hub_kt", m(HUB.torque_constant, "N*m/A"), "{:.1f}", "torque constant, N m/A")
v("load_pair", m(HUB.load_rating_per_pair, "kg"), "{:.0f}", "load rating per pair, kg")
v("phi_max", c.drive_load_share, "{:.3f}", "driven-wheel load share bound")
v("a_dry", m(c.a_traction_dry, "m/s**2"), "{:.2f}", "dry floor ceiling, m/s^2")
v("force", m(c.force, "N"), "{:.1f}", "floor force, N")
v("m_eff", m(c.m_eff, "kg"), "{:.0f}", "effective mass bound, kg")
v("a_torque", m(c.a_torque, "m/s**2"), "{:.2f}", "peak-current ceiling, m/s^2")
v("torque_margin", 100 * c.torque_margin, "{:.1f}", "margin over a_brake, %")
v("a_rated", m(c.a_torque_rated, "m/s**2"), "{:.2f}", "rated-current ceiling, m/s^2")
v("mu_min", c.mu_required, "{:.3f}", "friction premise, all weight driven")

print("== Actuators (2.4.2, Napkin Math 2.3 step 4)")
e = ELEC.result
v("tau_e", m(e.time_constant, "ms"), "{:.1f}", "electrical time constant, ms")
v("t_ramp", m(e.ramp_time, "ms"), "{:.1f}", "current ramp to the limit, ms")
v("ramp_pct", 100 * e.ramp_share, "{:.1f}", "ramp share of the brake onset, %")
v("v_low", m(RM.drive_bus_battery_low, "V"), "{:.0f}", "depleted pack, V")
v("l_line", m(HUB.line_inductance, "mH"), "{:.3f}", "line inductance, mH")
v("r_line", m(HUB.line_resistance, "ohm"), "{:.1f}", "line resistance, ohm")
v("ke_hub", m(HUB.back_emf_constant, "V/rpm"), "{:.3f}", "back-EMF constant, V/rpm")
v("rpm_drive", m(c.wheel_speed, "rpm"), "{:.1f}", "wheel speed at the drive limit, rpm")
v("bemf", m(c.back_emf, "V"), "{:.1f}", "back-EMF at the drive limit, V")
v("v_brake", m(c.v_brake, "V"), "{:.1f}", "braking terminal voltage, V")
v("v_drive_need", m(c.v_drive, "V"), "{:.1f}", "driving terminal voltage, V")
v("i_full_step", m(e.full_step_current, "A"), "{:.0f}", "full-bus step current, A")

print("== Gearing (2.4.3)")
rr = ROTOR.result
v("rotor_g", m(ROTOR.rotor_mass, "g"), "{:.0f}", "rotor mass, g")
v("n_gear", ROTOR.gear_ratio, "{:.0f}", "joint reduction")
v("j_ref", m(rr.reflected_inertia, "kg*m**2"), "{:.3f}", "reflected inertia, kg m^2")
v("fly_kg", m(rr.flywheel_mass, "kg"), "{:.1f}", "equivalent flywheel, kg")
v("fly_cm", m(ROTOR.flywheel_radius, "cm"), "{:.0f}", "flywheel radius, cm")
v("n_star", rr.matched_ratio, "{:.0f}", "matched ratio N*")
v("accel_pct", 100 * rr.acceleration_share, "{:.1f}", "acceleration kept at the stated ratio, %")
v("j_rotor_mant", m(ROTOR.rotor_inertia, "kg*m**2") * 10 ** 6, "{:.1f}", "rotor inertia, x 10^-6 kg m^2")
v("j_load_mant", m(ROTOR.load_inertia, "kg*m**2") * 10 ** 3, "{:.1f}", "link inertia, x 10^-3 kg m^2")

print("== Heat against inertia (Napkin Math 2.4)")
gr = GEARS.result
n_joint, n_high, n_qdd = GEARS.ratios
v("tau_out", m(GEARS.output_torque, "N*m"), "{:.0f}", "held joint torque, N m")
v("motor_peak", m(MOTOR.peak_torque, "N*m"), "{:.2f}", "motor peak torque, N m")
v("motor_rated", m(MOTOR.rated_torque, "N*m"), "{:.2f}", "motor rated torque, N m")
v("eta", GEARS.gear_efficiency, "{:.2f}", "gear efficiency")
v("eta_pct", 100 * GEARS.gear_efficiency, "{:.0f}", "the same, %")
v("km", m(gr.motor_constant, "N*m/W**0.5"), "{:.3f}", "motor constant, N m/sqrt(W)")
v("n_joint", n_joint, "{:.0f}", "joint ratio")
v("n_high", n_high, "{:.0f}", "catalog ratio")
v("n_qdd", n_qdd, "{:.0f}", "QDD-class ratio")
v("tm_high", m(gr.motor_torque[n_high], "N*m"), "{:.2f}", "motor torque at the catalog ratio, N m")
v("tm_joint", m(gr.motor_torque[n_joint], "N*m"), "{:.2f}", "motor torque at the joint ratio, N m")
v("tm_qdd", m(gr.motor_torque[n_qdd], "N*m"), "{:.1f}", "motor torque at the QDD ratio, N m")
v("loss_high", m(gr.copper_loss[n_high], "W"), "{:.1f}", "copper loss at the catalog ratio, W")
v("loss_joint", m(gr.copper_loss[n_joint], "W"), "{:.1f}", "copper loss at the joint ratio, W")
v("jr_high", m(gr.reflected_inertia[n_high], "kg*m**2"), "{:.3f}", "reflected inertia at the catalog ratio")
v("jr_joint", m(gr.reflected_inertia[n_joint], "kg*m**2"), "{:.3f}", "reflected inertia at the joint ratio")
v("qdd_over", gr.over_peak[n_qdd], "{:.1f}", "QDD motor torque over the motor peak")
v("loss_ratio_words", ratio(gr.copper_loss[n_joint], gr.copper_loss[n_high]), "words", "loss ratio, joint to catalog")
v("product", m(gr.loss_inertia_product[n_joint], "W*kg*m**2"), "{:.2f}", "loss x reflected inertia")
v("peak_over_rated", gr.peak_over_rated, "{:.1f}", "peak over continuous torque")

print("== Torque-speed envelope (figure 6a) and derating pitfall")
v("n0_rpm", m(MOTOR.no_load_speed, "rpm"), "{:,.0f}", "no-load speed, rpm")
n_pk = mlsysim.calc_voltage_limited_speed(MOTOR.peak_torque, MOTOR.torque_constant, MOTOR.terminal_resistance,
                                          MOTOR.rated_voltage, MOTOR.no_load_speed)
v("n_pk_rpm", round(m(n_pk, "rpm"), -1), "{:,.0f}", "voltage-limited speed at peak torque, rpm (to tens)")
v("t_catalog", degc(DERATE.catalog_ambient), "{:.0f}", "catalog ambient, C")
v("derate_ambient", 100 * (1 - DERATE.result.ambient_share), "{:.1f}", "loss to ambient alone, %")
v("derate_both", 100 * (1 - DERATE.result.combined_share), "{:.1f}", "loss with theta doubled, %")

print("== Thermal burst (Napkin Math 2.5)")
b = BURST.result
v("burst_words", BURST.burst_factor, "words", "burst factor")
v("burst_x", BURST.burst_factor, "{:.0f}", "burst factor")
v("heat_words", ratio(b.burst_loss, b.rated_loss), "words", "heat factor")
v("p_burst", m(b.burst_loss, "W"), "{:.0f}", "burst loss, W")
v("tau_th", m(b.time_constant, "s"), "{:.0f}", "thermal time constant, s")
v("c_th", m(BURST.thermal_capacitance, "J/K"), "{:.0f}", "thermal capacitance, J/K")
v("theta_burst", m(BURST.thermal_resistance, "K/W"), "{:.1f}", "thermal resistance, K/W")
v("t_amb", degc(BURST.ambient), "{:.0f}", "ambient, C")
v("t_limit", degc(BURST.limit), "{:.0f}", "Class F limit, C")
v("t_inf", degc(b.asymptote), "{:.0f}", "burst asymptote, C")
v("t_cold", m(b.horizon_cold, "s"), "{:.1f}", "cold-start horizon, s")
v("t_cold_0", m(b.horizon_cold, "s"), "{:.0f}", "the same, whole seconds")
v("t_hot", m(b.horizon_hot, "s"), "{:.1f}", "heat-soaked horizon, s")
v("t_hot_0", m(b.horizon_hot, "s"), "{:.0f}", "the same, whole seconds")
v("t_eq", degc(b.rated_equilibrium), "{:.1f}", "rated-current equilibrium, C")
v("hot_cut", 100 * b.hot_cut, "{:.1f}", "heat-soaked shortfall, %")
v("duty_pct", 100 * b.duty, "{:.0f}", "duty-cycle screen, %")
v("burst_on", m(BURST.burst_on, "s"), "{:.0f}", "burst length, s")
v("burst_period", m(BURST.burst_on + b.off_time_scheduled, "s"), "{:.0f}", "screened period, s")
v("t_peak", degc(b.periodic_peak), "{:.1f}", "periodic peak, C")
v("t_ceiling", degc(BURST.ceiling), "{:.0f}", "working ceiling, C")

print("== Stall derating (Napkin Math 2.6)")
s = STALL.result
v("kt_motor", m(MOTOR.torque_constant, "N*m/A"), "{:.3f}", "motor torque constant, N m/A")
v("r_loss", m(JOINT.internal_resistance, "ohm"), "{:.3f}", "loss resistance at 20 C, ohm")
v("t_ref", degc(mlsysim.COPPER_COEFFICIENT_REFERENCE_TEMPERATURE), "{:.0f}", "resistance reference, C")
v("theta_stall", m(STALL.thermal_resistance, "K/W"), "{:.2f}", "limb thermal resistance, K/W")
v("alpha_pct", m(mlsysim.COPPER_TEMPERATURE_COEFFICIENT, "1/K") * 100, "{:.3f}", "copper coefficient, % per K")
v("i_stall", m(s.stall_current, "A"), "{:.1f}", "stall current, A")
v("gain", s.feedback_gain, "{:.4f}", "feedback factor")
v("rise_ss", m(s.steady_rise, "K"), "{:.1f}", "steady rise, K")
v("t_wind", degc(s.winding_temperature), "{:.1f}", "stall equilibrium, C")
v("t_wind_0", degc(s.winding_temperature), "{:.0f}", "the same, whole degrees")
v("r_hot", m(s.hot_resistance, "ohm"), "{:.3f}", "hot resistance, ohm")
v("p_sus", m(s.sustainable_loss, "W"), "{:.2f}", "sustainable loss, W")
v("i_sus", m(s.sustainable_current, "A"), "{:.2f}", "sustainable current, A")
v("tau_sus", m(s.sustainable_hold, "N*m"), "{:.1f}", "sustainable hold, N m")
v("derating", 100 * s.derating, "{:.1f}", "derating, %")

print("== Drive bus (Napkin Math 2.7)")
u = BUS.result
v("v_full", m(RM.drive_bus_full_charge, "V"), "{:.0f}", "full pack, V")
v("i_step", m(AISLE.i_rail_transient, "A"), "{:.0f}", "coincident step, A")
v("r_bus_mohm", m(RM.drive_bus_resistance, "milliohm"), "{:.0f}", "source resistance, mOhm")
v("r_bus_ohm", m(RM.drive_bus_resistance, "ohm"), "{:.3f}", "the same, ohm")
v("sag", m(u.sag, "V"), "{:.1f}", "sustained sag, V")
v("v_sag_full", m(u.sustained_full, "V"), "{:.1f}", "full pack under the step, V")
v("v_sag_low", m(u.sustained_depleted, "V"), "{:.1f}", "depleted pack under the step, V")
v("v_margin", m(u.margin_depleted, "V"), "{:.1f}", "margin above lockout, V")
v("v_headroom", m(u.headroom_depleted, "V"), "{:.0f}", "depleted pack to lockout, V")
v("i_cross", m(u.crossing_current, "A"), "{:.0f}", "step that spends it, A")
v("v_uvlo", m(RM.drive_bus_uvlo, "V"), "{:.0f}", "converter lockout, V")
v("i_drive", m(u.drive_current, "A"), "{:.1f}", "drive share of the step, A")
v("i_compute", m(u.compute_current, "A"), "{:.1f}", "compute share of the step, A")
v("p_soc", m(RM.brain_soc.tdp, "W"), "{:.0f}", "application processor, W")
v("regen_pct", 100 * BUS.regen_fraction, "{:.0f}", "share returned, %")
v("regen_frac", BUS.regen_fraction, "{:.2f}", "the same, fraction")
v("c_bus", m(BUS.link_capacitance, "microfarad"), "{:,.0f}", "bus capacitance, uF")
v("t_stop", m(u.stop_time, "ms"), "{:.0f}", "stop duration, ms")
v("e_regen", m(u.regenerated, "J"), "{:.1f}", "regenerated energy, J")
v("v_final", m(u.capacitor_only_endpoint, "V"), "{:.1f}", "capacitor-only endpoint, V")
v("v_conv_max", m(RM.drive_bus_converter_max, "V"), "{:.1f}", "converter input maximum, V")
v("v_switch", m(RM.drive_bus_switch_rating, "V"), "{:.0f}", "switch rating, V")
v("v_clamp", m(RM.drive_bus_clamp, "V"), "{:.0f}", "chopper clamp, V")
v("e_clamp", m(u.capacitor_to_clamp, "J"), "{:.1f}", "capacitor energy to the clamp, J")
v("p_peak", m(u.peak_power, "W"), "{:.0f}", "peak regenerated power, W")
v("r_max", m(u.resistor_max, "ohm"), "{:.2f}", "largest chopper resistor, ohm")
v("r_cand", m(BUS.candidate_resistor, "ohm"), "{:.1f}", "chosen resistor, ohm")
v("p_cand", m(BUS.candidate_power, "kW"), "{:.0f}", "its pulse rating, kW")

print("== Unpowered states and the permission rail (2.6.2, 2.8.2)")
v("brake_min", m(RM.spring_brake_engage_min, "ms"), "{:.0f}", "spring brake, fast end, ms")
v("brake_max", m(RM.spring_brake_engage_max, "ms"), "{:.0f}", "spring brake, slow end, ms")
v("rpm_short", m(e.short_peak_wheel_speed, "rpm"), "{:.1f}", "phase-short peak, wheel rpm")
v("pole_pairs", HUB.pole_pairs, "{:.0f}", "hub pole pairs")
v("c_hold", m(RM.permission_rail_holdup_capacitance, "F"), "{:.3f}", "hold-up store, F")
v("v_hold_hi", m(RM.control_rail_battery_low, "V"), "{:.1f}", "store charged to, V")
v("v_hold_lo", m(RM.control_rail_dropout, "V"), "{:.0f}", "regulator dropout, V")
v("e_hold", m(RM.permission_rail_holdup_energy, "J"), "{:.0f}", "usable store energy, J")
v("t_hold", m(RM.permission_rail_holdup, "s"), "{:.1f}", "hold-up time, s")
v("p_perm", m(RM.permission_rail_load, "W"), "{:.0f}", "permission load, W")
v("t_backbone", m(AISLE.t_backbone, "ms"), "{:.0f}", "navigation backbone, ms")
v("t_throttled", m(THROTTLE.result.throttled, "ms"), "{:.0f}", "throttled backbone, ms")
v("d_throttle", m(THROTTLE.result.added_travel, "mm"), "{:.1f}", "added blind travel, mm")

print("== Limits (2.7)")
v("n_stops", CAMPAIGN.trials, "{:,.0f}", "clean loaded stops")
v("one_in", round(CAMPAIGN.result.one_in, -2), "{:,.0f}", "rate bound, one stop in")
v("conf_pct", 100 * CAMPAIGN.confidence, "{:.0f}", "confidence, %")


if __name__ == "__main__":
    write(__file__)
