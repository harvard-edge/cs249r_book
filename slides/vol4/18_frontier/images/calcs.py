#!/usr/bin/env python3
"""Every number on the Lecture 18 slides and notes (Engineering Physical AI), read from MLSysIM.

The closing lecture recalls the course's running machine, so its numbers are
the same MLSysIM objects the earlier decks read: the site record
Embodied.Scenario.WarehouseAisle and the machine
Embodied.MobileManipulator.WarehouseMobileManipulator; the finished stopping
budgets (Scenarios.WarehouseFinishedStop*, StoppingBudgetModel) dry, on the
inspected floor, on a clear oil film, with the dispatch-to-admission allocation,
and in the restricted mode; the film detector's range and closure test
(WarehouseFilmDetector*); the restricted transit (WarehouseRestrictedTransit);
the hypothetical packaging cell (Scenarios.PackagingCell*); the coverage bill
of a shielded hazard claim (ShieldCoverageBill); the series spring at the latch
(WarehouseLatchSeriesSpring); the SICK deTec4 light curtain
(Sensors.SafetyLightCurtain); and the cited records in Literature (TRL PPR073
friction, Waymo's Safety Impact data, Open X-Embodiment, Agility's tote count).
The chapter reads the same scenarios. This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports it for the figure inputs.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

from mlsysim.physics.drivetrain import calc_required_friction  # noqa: E402

Embodied, Scenarios, Literature, Sensors = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature, mlsysim.Sensors
Q_ = mlsysim.core.units.Q_

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
STOP = Scenarios.WarehouseFinishedStopAtAisleSpeed
INSPECTED = Scenarios.WarehouseFinishedStopInspectedFloor
FILM = Scenarios.WarehouseFinishedStopOilFilm
DISPATCH = Scenarios.WarehouseFinishedStopWithDispatch
RESTRICTED = Scenarios.WarehouseFinishedStopRestricted
DETECT = Scenarios.WarehouseFilmDetectorRange
CLOSURE = Scenarios.WarehouseFilmDetectorClosure
TRANSIT = Scenarios.WarehouseRestrictedTransit
CELL = Scenarios.PackagingCellTransit
MODE = Scenarios.PackagingCellRestrictedMode
FASTPROC = Scenarios.PackagingCellHalfProcessing
HALFBRAKE = Scenarios.PackagingCellHalfBrake
QUALIFIED = Scenarios.PackagingCellQualifiedBrake
CURTAIN_PATH = Scenarios.PackagingCellCurtainPath
CURTAIN = Sensors.SafetyLightCurtain.SICK_deTec4_Set_30_450
BILL = Scenarios.ShieldCoverageBill
SPRING = Scenarios.WarehouseLatchSeriesSpring
TRL = Literature.ContaminatedSurfaceFriction
OXE = Scenarios.OpenXEmbodimentPoolComposition


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def sci(key, value, fmt, label):
    """Record a positive value as mantissa (key_mant), power of ten (key_exp), and |power| in words."""
    exp = math.floor(math.log10(value))
    mant = value / 10 ** exp
    if float(fmt.format(mant)) >= 10:  # the mantissa rounds up to the next decade
        exp, mant = exp + 1, mant / 10
    v(f"{key}_mant", mant, fmt, f"{label}, mantissa")
    v(f"{key}_exp", exp, "{:.0f}", f"{label}, power of ten")
    v(f"{key}_exp_words", abs(exp), "words", f"{label}, power of ten spoken")


dry, insp, film, disp = STOP.result, INSPECTED.result, FILM.result, DISPATCH.result
cell, mode, qual = CELL.result, MODE.result, QUALIFIED.result
bill = BILL.result
G = m(Q_(1, "standard_gravity"), "m/s**2")

# Inputs for make_figs.py, at full precision.
A_BRAKE = m(AISLE.a_brake, "m/s**2")
MU_DRY, MU_INSP, MU_OIL = AISLE.mu_dry, AISLE.mu_inspected_floor, AISLE.mu_oil_film
MU_MIN = calc_required_friction(AISLE.a_brake)
D_CLEAR_MM = m(AISLE.d_clear, "mm")
TERMS = {  # finished-budget rows, mm: (delay, braking, jerk ramp, overheads)
    name: (m(r.delay_travel, "mm"), m(r.braking_travel, "mm"), m(r.jerk_ramp_travel, "mm"),
           m(r.overhead + r.tracking_inset, "mm"))
    for name, r in (("dry", dry), ("film", film))
}
CELL_ROWS = [(m(s.speed, "m/s"), m(s.result.delay_travel, "mm"), m(s.result.braking_travel, "mm"))
             for s in (CELL, MODE)]
CELL_CLEAR_MM, CELL_RESERVE_MM = m(CELL.clearance, "mm"), m(CELL.reserve, "mm")
CEILINGS = [(s, m(s.result.speed_ceiling, "m/s")) for s in (CURTAIN_PATH, FASTPROC, CELL, HALFBRAKE)]
V_MODE = m(MODE.speed, "m/s")
T_STOP_MS, T_EQ_MS = m(dry.stop_time, "ms"), m(dry.scurve_travel / dry.speed, "ms")
SPARE_MS = m(dry.delay_spare, "ms")
TRL_ROWS = [  # (surface, contaminant, tire, coefficient), in the slide's order
    ("Concrete, dry, skid cars", TRL.ConcreteDrySkidCars),
    ("Concrete, wet, smooth tyre", TRL.ConcreteWetPft),
    ("Concrete, diesel, skid cars", TRL.ConcreteDieselSkidCars),
    ("Concrete, diesel, smooth tyre", TRL.ConcreteDieselPft),
    ("Asphalt, engine oil, skid cars", TRL.AsphaltEngineOilSkidCars),
    ("Asphalt, engine oil, smooth tyre", TRL.AsphaltEngineOilPft),
]
LATCH_BARS = [(m(AISLE.v_latch_approach, "m/s"), m(SPRING.coasting_rigid_guarded, "N")),
              (m(AISLE.v_latch_high, "m/s"), m(SPRING.coasting_rigid_fast, "N")),
              (m(AISLE.v_latch_high, "m/s"), m(SPRING.coasting_spring_fast, "N"))]
F_FLOOR, F_TRIP = m(AISLE.f_latch_limit, "N"), m(AISLE.f_latch_tripwire, "N")
A_AVAIL = {"dry": m(dry.deceleration, "m/s**2"), "insp": m(insp.deceleration, "m/s**2"),
           "film": m(film.deceleration, "m/s**2")}


print("== Hook and the finished aisle budget (Napkin Math 18.2; Scenarios.WarehouseFinishedStop*)")
v("v_aisle", m(AISLE.v_aisle, "m/s"), "{:.1f}", "aisle speed, m/s")
v("d_clear_m", m(AISLE.d_clear, "m"), "{:.2f}", "clear distance, m")
v("d_clear_mm", D_CLEAR_MM, "{:,.0f}", "clear distance, mm")
v("budget", m(dry.required_distance, "mm"), "{:.1f}", "dry budget, known components, mm")
v("film_m", m(film.required_distance, "m"), "{:.2f}", "film budget, m")
v("film_m1", m(film.required_distance, "m"), "{:.1f}", "film budget, m, one place (vote choice)")
v("film_brake", m(film.braking_travel, "mm"), "{:,.1f}", "film braking, mm")
v("film_brake_words", round(m(film.braking_travel / dry.braking_travel, "")), "words", "film braking over dry, nearest whole times")
v("overrun_m", m(film.required_distance - AISLE.d_clear, "m"), "{:.0f}", "film overrun of the clear distance, m")
v("reaction", m(dry.delay_travel, "mm"), "{:.1f}", "delay travel, mm")
v("braking", m(dry.braking_travel, "mm"), "{:.1f}", "dry braking, mm")
v("ramp", m(dry.jerk_ramp_travel, "mm"), "{:.1f}", "jerk ramp, mm")
v("allowances", m(dry.overhead + dry.tracking_inset, "mm"), "{:.0f}", "fixed overheads, mm")
v("t_da", m(AISLE.t_dispatch_to_admission, "ms"), "{:.1f}", "dispatch-to-admission allocation, ms")
v("travel_da", m(dry.travel_over(AISLE.t_dispatch_to_admission), "mm"), "{:.1f}", "its travel, mm")
v("reserve", m(disp.spare, "mm"), "{:.1f}", "spare after the allocation, mm")

print("== The closed loop at three rates (registry machine)")
v("intent_hz", m(RM.intent_rate, "Hz"), "{:.0f}", "intent rate, Hz")
v("renewal_ms", m(1 / RM.chunk_rate, "ms"), "{:.0f}", "chunk renewal, ms")
v("permission_khz", m(RM.permission_rate, "kHz"), "{:.0f}", "permission tick, kHz")
v("current_khz", m(RM.current_loop_rate, "kHz"), "{:.0f}", "drive current loop, kHz")
v("loop_orders_words", math.log10(m(RM.current_loop_rate / RM.intent_rate, "")), "words",
  "orders of magnitude between the fastest and slowest loop")

print("== History: Open X-Embodiment (Lecture 5; Scenarios.OpenXEmbodimentPoolComposition)")
v("oxe_total_m", OXE.total / 10 ** 6, "{:.2f}", "episodes, millions")
v("oxe_robots", len({r["robot"] for r in OXE.rows}), "{:.0f}", "distinct robots in the pool")

print("== Vote and restriction cost (Scenarios.WarehouseRestrictedTransit)")
v("v_restricted", m(TRANSIT.restricted_speed, "m/s"), "{:.0f}", "restricted speed, m/s")
v("leg_m", m(AISLE.aisle_length, "m"), "{:.0f}", "aisle leg, m")
v("t_leg_nom", m(TRANSIT.leg_time_nominal, "s"), "{:.1f}", "leg at the aisle speed, s")
v("t_leg_restr", m(TRANSIT.leg_time_restricted, "s"), "{:.0f}", "leg at the restricted speed, s")
v("time_increase_pct", 100 * TRANSIT.time_increase, "{:.0f}", "leg time increase, %")
v("speed_cut_pct", 100 * TRANSIT.speed_cut, "{:.1f}", "speed cut, %")
v("speed_cut_pct0", 100 * TRANSIT.speed_cut, "{:.0f}", "speed cut, %, whole")
v("totes", Literature.RobotDeployments.AgilityDigitTotesMoved, "{:,.0f}", "totes moved, Agility Digit")

print("== Napkin Math 18.1: the delay wall (Scenarios.PackagingCell*)")
v("cell_v", m(CELL.speed, "m/s"), "{:.2f}", "carriage nominal speed, m/s")
v("cell_clear", m(CELL.clearance, "mm"), "{:.0f}", "clearance to the stop line, mm")
v("cell_reserve", m(CELL.reserve, "mm"), "{:.0f}", "reserve, mm")
v("cell_a", m(CELL.deceleration, "m/s**2"), "{:.0f}", "braking bound, m/s^2")
v("cell_two_a", m(CELL.speed ** 2 / cell.braking_travel, "m/s**2"), "{:.0f}", "2a: speed squared over braking travel, m/s^2")
v("cell_tau", m(CELL.delay, "ms"), "{:.0f}", "delay chain, ms")
v("cell_tau_s", m(CELL.delay, "s"), "{:.3f}", "delay chain, s")
v("cell_exposure", m(CELL.exposure, "ms"), "{:.0f}", "event to usable image, ms")
v("cell_processing", m(CELL.processing, "ms"), "{:.0f}", "transfer through admission, ms")
v("cell_bus", m(CELL.bus, "ms"), "{:.0f}", "bus, ms")
v("cell_onset", m(CELL.onset, "ms"), "{:.0f}", "brake onset, ms")
v("cell_delay_nom", m(cell.delay_travel, "mm"), "{:.0f}", "delay travel at nominal speed, mm")
v("cell_brake_nom", m(cell.braking_travel, "mm"), "{:.0f}", "braking at nominal speed, mm")
v("cell_stop_nom", m(cell.stop_distance, "mm"), "{:.0f}", "stop at nominal speed, mm")
v("cell_deficit", m(CELL.deficit, "mm"), "{:.0f}", "deficit against the stop line, mm")
v("cell_budget", m(CELL.stop_budget, "mm"), "{:.0f}", "stop budget, mm")
a_tau, two_a_d, root = CELL.root_terms
v("cell_a_tau", m(a_tau, "m/s"), "{:.2f}", "a tau, m/s")
v("cell_two_a_d", m(two_a_d, "m**2/s**2"), "{:.2f}", "2ad, m^2/s^2")
v("cell_root", m(root, "m/s"), "{:.2f}", "root, m/s")
v("cell_ceiling", m(cell.speed_ceiling, "m/s"), "{:.2f}", "speed ceiling, m/s")
v("cell_v_mode", m(MODE.speed, "m/s"), "{:.2f}", "candidate mode speed, m/s")
v("cell_delay_mode", m(mode.delay_travel, "mm"), "{:.0f}", "delay travel in the mode, mm")
v("cell_brake_mode", m(mode.braking_travel, "mm"), "{:.1f}", "braking in the mode, mm")
v("cell_stop_mode", m(mode.stop_distance, "mm"), "{:.1f}", "stop in the mode, mm")
v("cell_left_mode", m(MODE.remaining, "mm"), "{:.1f}", "left before the stop line, mm")
v("cell_mass", m(CELL.mass, "kg"), "{:.0f}", "moving mass, kg")
v("cell_ke_nom", m(CELL.kinetic_energy, "J"), "{:.1f}", "kinetic energy at nominal speed, J")
v("cell_ke_mode", m(MODE.kinetic_energy, "J"), "{:.1f}", "kinetic energy in the mode, J")
v("cell_ke_cut_pct", 100 * m(1 - MODE.kinetic_energy / CELL.kinetic_energy, ""), "{:.1f}", "energy reduction, %")
v("cell_delay_ratio", m(cell.delay_travel / mode.delay_travel, ""), "{:.0f}", "delay travel falls, times")
v("cell_brake_ratio", m(cell.braking_travel / mode.braking_travel, ""), "{:.0f}", "braking falls, times")
v("cell_delay_ratio_words", m(cell.delay_travel / mode.delay_travel, ""), "words", "delay travel falls, spoken")
v("cell_brake_ratio_words", m(cell.braking_travel / mode.braking_travel, ""), "words", "braking falls, spoken")

print("== Sensitivity and the certified curtain (deTec4 data sheet)")
v("cell_processing_half", m(FASTPROC.processing, "ms"), "{:.0f}", "processing halved, ms")
v("cell_tau_fastproc", m(FASTPROC.delay, "ms"), "{:.0f}", "chain with processing halved, ms")
v("cell_ceiling_fastproc", m(FASTPROC.result.speed_ceiling, "m/s"), "{:.2f}", "ceiling, processing halved, m/s")
v("cell_a_half", m(HALFBRAKE.deceleration, "m/s**2"), "{:.1f}", "braking bound halved, m/s^2")
v("cell_ceiling_halfbrake", m(HALFBRAKE.result.speed_ceiling, "m/s"), "{:.2f}", "ceiling, brake halved, m/s")
v("curtain_ms", m(CURTAIN.response_time, "ms"), "{:.0f}", "deTec4 response, uncoded, ms")
v("curtain_pfh_e9", m(CURTAIN.pfh_d, "1/hour") * 10 ** 9, "{:.1f}", "deTec4 PFHD, single device, per 10^9 hours")
v("cell_tau_curtain", m(CURTAIN_PATH.delay, "ms"), "{:.0f}", "curtain chain, ms")
v("cell_ceiling_curtain", m(CURTAIN_PATH.result.speed_ceiling, "m/s"), "{:.2f}", "ceiling, curtain path, m/s")
v("cell_check_khz", m(CELL.check_rate, "kHz"), "{:.0f}", "kinematic-check rate, kHz")
v("cell_force_cap", m(CELL.force_limit, "N"), "{:.0f}", "force limiter, N")
v("cell_temp_trip", m(CELL.temperature_trip, "degC"), "{:.0f}", "thermistor trip, degrees C")

print("== Checkpoint 18.1: the brake qualified lower (Scenarios.PackagingCellQualifiedBrake)")
q_a_tau, q_two_a_d, _ = QUALIFIED.root_terms
v("q_a", m(QUALIFIED.deceleration, "m/s**2"), "{:.0f}", "qualified brake, m/s^2")
v("q_a_tau", m(q_a_tau, "m/s"), "{:.2f}", "a tau, m/s")
v("q_two_a_d", m(q_two_a_d, "m**2/s**2"), "{:.2f}", "2ad, m^2/s^2")
v("q_ceiling", m(qual.speed_ceiling, "m/s"), "{:.2f}", "ceiling, m/s")
v("q_brake_mode", m(qual.braking_travel, "mm"), "{:.1f}", "braking in the mode, mm")
v("q_stop_mode", m(qual.stop_distance, "mm"), "{:.1f}", "stop in the mode, mm")
v("q_left", m(QUALIFIED.stop_budget - qual.stop_distance, "mm"), "{:.1f}", "budget left, mm")

print("== Friction floor (equation 1; site record)")
v("a_brake", A_BRAKE, "{:.0f}", "credible deceleration, m/s^2")
v("g", G, "{:.2f}", "standard gravity, m/s^2")
v("mu_min", MU_MIN, "{:.3f}", "friction floor")
v("mu_dry", MU_DRY, "{:.2f}", "dry concrete")
v("mu_insp", MU_INSP, "{:.2f}", "inspected floor")
v("mu_oil", MU_OIL, "{:.2f}", "clear oil film")
v("a_insp", m(insp.deceleration, "m/s**2"), "{:.2f}", "inspected-floor deceleration, m/s^2")
v("a_oil", m(film.deceleration, "m/s**2"), "{:.2f}", "film deceleration, m/s^2")
v("ceiling_insp", m(insp.speed_ceiling, "m/s"), "{:.2f}", "inspected-floor ceiling, m/s")

print("== Measured friction (Literature.ContaminatedSurfaceFriction, TRL PPR073 table 5.1)")
for key, (_, mu) in zip(("trl_dry", "trl_wet_smooth", "trl_diesel", "trl_diesel_smooth", "trl_oil", "trl_oil_smooth"),
                        TRL_ROWS):
    v(key, mu, "{:.2f}", "locked-wheel coefficient")

print("== Time form against distance form (sec. 18.3.4 pitfall)")
v("t_stop", T_STOP_MS, "{:.1f}", "S-curve stop duration, ms")
v("t_eq", T_EQ_MS, "{:.1f}", "stop distance over speed, ms")
v("t_over", T_STOP_MS - T_EQ_MS, "{:.1f}", "over-reserve, ms")
v("d_stop", m(dry.scurve_travel, "mm"), "{:.1f}", "S-curve stop distance, mm")
v("t_spare", SPARE_MS, "{:.1f}", "spare as delay, ms")
v("over_words", math.floor((T_STOP_MS - T_EQ_MS) / SPARE_MS), "words", "over-reserve, whole multiples of the spare")

print("== Latch compliance (Scenarios.WarehouseLatchSeriesSpring)")
v("f_floor", F_FLOOR, "{:.0f}", "declared floor, N")
v("f_trip", F_TRIP, "{:.0f}", "tripwire, N")
v("m_eff", m(RM.arm_effective_contact_mass, "kg"), "{:.0f}", "arm effective mass, kg")
sci("k_plate", m(AISLE.k_latch, "N/m"), "{:.1f}", "strike-plate stiffness, N/m")
v("v_guard", m(AISLE.v_latch_approach, "m/s"), "{:.2f}", "guarded approach, m/s")
v("v_fast", m(AISLE.v_latch_high, "m/s"), "{:.2f}", "fast approach, m/s")
v("f_rigid_guard", m(SPRING.coasting_rigid_guarded, "N"), "{:.1f}", "rigid plate, guarded, N")
v("f_rigid_fast", m(SPRING.coasting_rigid_fast, "N"), "{:.1f}", "rigid plate, fast, N")
v("f_spring_fast", m(SPRING.coasting_spring_fast, "N"), "{:.1f}", "series spring, fast, N")
v("k_spring", m(SPRING.spring_stiffness_max, "kN/m"), "{:.1f}", "spring stiffness bound, kN/m")
v("spring_stroke", m(SPRING.spring_stroke, "mm"), "{:.1f}", "spring stroke at the floor, mm")

print("== Register: the film entry's closure test (Scenarios.WarehouseFilmDetectorClosure)")
v("n_closure", CLOSURE.result.trials_for_target, "{:,.0f}", "seeded-film trials, no miss")
v("p_miss_pct", 100 * CLOSURE.target_rate, "{:.1f}", "miss bound, %")
v("conf_pct", 100 * CLOSURE.confidence, "{:.0f}", "confidence, %")

print("== Napkin Math 18.3: the coverage bill (Scenarios.ShieldCoverageBill)")
v("fleet", BILL.fleet_units, "{:.0f}", "machines")
v("cycle_days", m(BILL.cycle_days, "day"), "{:.0f}", "release cycle, days")
v("hours_per_day", m(BILL.hours_per_day, "hour/day"), "{:.0f}", "hours per day")
v("coverage", BILL.coverage, "{:.4f}", "shield coverage")
v("lambda_cc", m(BILL.common_cause_rate, "1/hour"), "{:.0f}", "common-cause rate, per hour")
sci("target", m(BILL.target_rate, "1/hour"), "{:.0f}", "target, per hour")
v("target_exp_abs", -math.log10(m(BILL.target_rate, "1/hour")), "{:.0f}", "target, power of ten, magnitude")
v("conf_alpha", 1 - BILL.confidence, "{:.2f}", "alpha")
v("ln_inv_alpha", bill.ln_inv_alpha, "{:.3f}", "-ln alpha")
v("exposure_h", m(bill.exposure, "hour"), "{:,.0f}", "proposal-hours")
sci("exposure_h", m(bill.exposure, "hour"), "{:.2f}", "proposal-hours")
sci("p_upper", m(bill.proposal_rate_upper, "1/hour"), "{:.3f}", "proposal bound, per hour")
sci("residual", m(bill.residual_rate, "1/hour"), "{:.3f}", "residual, per hour")
sci("p_limit", m(bill.proposal_rate_limit, "1/hour"), "{:.0f}", "proposal bound the target needs, per hour")
v("n_coverage", bill.coverage_trials, "{:,.0f}", "seeded hazards for coverage")
v("days_needed", m(bill.days_needed, "day"), "{:.1f}", "days of log to meet the bound")
v("days_beyond", m(bill.days_beyond_cycle, "day"), "{:.1f}", "beyond the cycle, days")

print("== Waymo: rider-only injury crashes (vol4/16_release/data/waymo_safety_impact_headline.csv, as in Lecture 16)")
rows = load_series("vol4/16_release/data/waymo_safety_impact_headline.csv").entries
last = max(r["data_cut"] for r in rows)
inj = next(r for r in rows if r["data_cut"] == last and r["outcome"] == "Any-injury-reported")
v("waymo_n", inj["waymo_count"], "{:.0f}", "injury crashes")
v("waymo_miles", inj["ro_miles_millions"], "{:.1f}", "rider-only miles, millions")
v("waymo_rate", inj["waymo_ipmm"], "{:.2f}", "per million miles")
v("waymo_bench", inj["benchmark_ipmm"], "{:.2f}", "benchmark per million miles")

if __name__ == "__main__":
    write(__file__)
