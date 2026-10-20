#!/usr/bin/env python3
"""Every number on the Lecture 12 slides and notes (Safety Enforcement), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* running machine and its aisle record,
the SICK nanoScan3 in Sensors.SafetyScanner, Literature.* for the Therac-25,
Cruise, and Kirschner et al. records) and the calculations: the finished
stopping budgets (Scenarios.WarehouseFinishedStop*, StoppingBudgetModel with
the S-curve jerk bound, tracking inset, and walking person), the loaded arm
joint and its kinetic barrier (Scenarios.LoadedArmJointGravityStop), the
two-joint projection, the permission check's stage budget, the hold drift,
and the physics.robotics latch and contact-force equations. The chapter reads
the same scenarios. This file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports it for the scenario objects and plot inputs.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

from mlsysim.physics.robotics import (  # noqa: E402
    calc_cbf_qp_orthogonal_projection,
    calc_latch_contact_stopping_profile,
    calc_reduced_mass,
    calc_scurve_stop_distance,
    calc_scurve_stop_time,
    calc_transient_contact_speed_limit,
    calc_transient_impact_force,
)

Embodied, Scenarios, Literature = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature
Sensors = mlsysim.Sensors
Q_ = mlsysim.core.units.Q_

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
REPORTS = Literature.IncidentReports
FORCES = Literature.ContactForceMeasurements
SCANNER = Sensors.SafetyScanner.SICK_nanoScan3

DRIVE_STOP = Scenarios.WarehouseFinishedStopAtDriveLimit
AISLE_STOP = Scenarios.WarehouseFinishedStopAtAisleSpeed
DISPATCH = Scenarios.WarehouseFinishedStopWithDispatch
INSPECTED = Scenarios.WarehouseFinishedStopInspectedFloor
INSPECTED_DISPATCH = Scenarios.WarehouseFinishedStopInspectedFloorWithDispatch
WALKER = Scenarios.WarehouseFinishedStopWalkingPerson
WALKER_DISPATCH = Scenarios.WarehouseFinishedStopWalkingPersonWithDispatch
SLOWER = Scenarios.WarehouseFinishedStopTenMsSlower
STALL = Scenarios.WarehouseHostLeaseStall
UNWIND = Scenarios.WarehouseIntegratorUnwind
JOINT = Scenarios.LoadedArmJointGravityStop
PROJ = Scenarios.TwoJointTorqueProjection
CYCLE = Scenarios.WarehouseEnforcerCycleBudget
HOLD = Scenarios.WarehouseBaseHoldDrift


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


drive, aisle, dispatch = DRIVE_STOP.result, AISLE_STOP.result, DISPATCH.result
walker = WALKER.result

# Plot inputs for make_figs.py, SI magnitudes at full precision.
V_DRIVE = m(DRIVE_STOP.speed, "m/s")
V_AISLE = m(AISLE_STOP.speed, "m/s")
A_BRAKE = m(AISLE.a_brake, "m/s**2")
A_INSP = m(INSPECTED.a_brake, "m/s**2")
J = m(AISLE.j_brake, "m/s**3")
D_CLEAR = m(AISLE.d_clear, "m")
FIXED = m(AISLE.fixed_overhead, "m")
T_DELAY = m(AISLE.tau_delay, "s")
V_WALK = m(AISLE.v_human_approach, "m/s")
V_CEIL = m(aisle.speed_ceiling, "m/s")
V_CEIL_INSP = m(INSPECTED.result.speed_ceiling, "m/s")
V_WALKER = m(walker.speed_ceiling, "m/s")
T_SCURVE_DRIVE = m(drive.stop_time, "s")
T_CONST_DRIVE = m(drive.braking_time, "s")


def stop_dist(speed, decel=A_BRAKE):
    """Resident S-curve stop distance in m (calc_scurve_stop_distance)."""
    return m(calc_scurve_stop_distance(Q_(speed, "m/s"), Q_(decel, "m/s**2"), AISLE.j_brake), "m")


def stop_time(speed, decel=A_BRAKE):
    """Resident S-curve stop duration in s (calc_scurve_stop_time)."""
    return m(calc_scurve_stop_time(Q_(speed, "m/s"), Q_(decel, "m/s**2"), AISLE.j_brake), "s")


def ceiling_at(clear):
    """Pre-dispatch speed ceiling in m/s at a clear distance in m (the aisle budget's root)."""
    if clear <= FIXED:
        return 0.0
    return m(AISLE_STOP.variant(clear_distance=Q_(clear, "m")).result.speed_ceiling, "m/s")


print("== Hook: Therac-25 (SE War Story 12.1; Leveson and Turner 1993)")
v("therac_mev", REPORTS.TheracBeamEnergyMeV, "{:.0f}", "beam energy, MeV")
v("therac_patients_words", REPORTS.TheracOverdosedPatients, "words", "patients overdosed")
v("therac_deaths_words", REPORTS.TheracDeaths, "words", "patients who died")
v("therac_dose_low", REPORTS.TheracDoseLowRads, "{:,.0f}", "dose estimate, low, rad")
v("therac_dose_high", REPORTS.TheracDoseHighRads, "{:,.0f}", "dose estimate, high, rad")
v("therac_over", REPORTS.TheracDoseOverTherapeutic, "{:.0f}", "times the therapeutic dose")

print("== Recall and the permission tick (Lecture 11; SE 12.1, 12.10)")
v("v_drive", V_DRIVE, "{:.1f}", "drive limit, m/s")
v("a_brake", A_BRAKE, "{:.0f}", "credible deceleration, m/s^2")
v("j_brake", J, "{:.2f}", "jerk bound, m/s^3")
v("t_const_drive", m(drive.braking_time, "ms"), "{:.0f}", "constant-deceleration stop from the drive limit, ms")
v("t_scurve_drive", m(drive.stop_time, "ms"), "{:,.2f}", "S-curve stop from the drive limit, ms")
v("t_tick", m(AISLE.t_tick, "ms"), "{:.0f}", "permission tick, ms")
v("tick_rate", m(1 / AISLE.t_tick, "Hz"), "{:,.0f}", "ticks per second")
v("chunk_period", m(1 / RM.chunk_rate, "ms"), "{:.0f}", "chunk renewal period, ms")

print("== The finished budget at the drive limit and the aisle speed (SE table 1)")
lease_path = AISLE.t_lease + AISLE.t_tick + AISLE.t_bus
rows_drive = [drive.braking_travel, drive.overhead, drive.travel_over(AISLE.t_brake_onset),
              drive.travel_over(lease_path), drive.travel_over(AISLE.perception_age),
              drive.jerk_ramp_travel, drive.tracking_inset]
run = [sum(rows_drive[:i + 1], 0 * rows_drive[0]) for i in range(len(rows_drive))]
v("d_clear_m", m(AISLE.d_clear, "m"), "{:.2f}", "clear distance, m")
v("d_clear_mm", m(AISLE.d_clear, "mm"), "{:,.0f}", "clear distance, mm")
v("brake_drive", m(drive.braking_travel, "mm"), "{:.1f}", "row 1 at the drive limit, mm")
v("overhead", m(drive.overhead, "mm"), "{:.0f}", "row 2, localization plus protective, mm")
v("run5", m(run[4], "mm"), "{:.1f}", "rows 1-5 at the drive limit, mm")
v("shape_drive", m(drive.jerk_ramp_travel, "mm"), "{:.2f}", "row 6, jerk ramps at the drive limit, mm")
v("run6", m(run[5], "mm"), "{:,.1f}", "rows 1-6 at the drive limit, mm")
v("over6", m(run[5] - AISLE.d_clear, "mm"), "{:.1f}", "rows 1-6 overrun, mm")
v("run7", m(drive.required_distance, "mm"), "{:,.1f}", "rows 1-7 at the drive limit, mm")
v("over7", m(-drive.spare, "mm"), "{:.1f}", "rows 1-7 overrun, mm")
v("eps_track", m(AISLE.eps_track, "mm"), "{:.0f}", "tracking bound, mm")
v("eps_doubled", m(AISLE.eps_track + AISLE.eps_track, "mm"), "{:.0f}", "tracking error doubled by wear, mm")
v("v_aisle", V_AISLE, "{:.1f}", "candidate aisle speed, m/s")
v("total_aisle", m(aisle.required_distance, "mm"), "{:.1f}", "rows 1-7 at the aisle speed, mm")
v("total_aisle_2", m(aisle.required_distance, "mm"), "{:.2f}", "the same, two decimals")
v("spare", m(aisle.spare, "mm"), "{:.1f}", "spare at the aisle speed, mm")
travel_da = AISLE_STOP.result.travel_over(AISLE.t_dispatch_to_admission)
v("travel_da", m(travel_da, "mm"), "{:.1f}", "dispatch-to-admission travel, mm")
v("delta_da", m(AISLE.t_dispatch_to_admission, "ms"), "{:.1f}", "dispatch-to-admission allocation, ms")

print("== Worked example: the ceiling (SE Napkin Math 12.1, equation 2)")
tau_eff = AISLE.tau_delay + drive.ramp_delay
d_avail = AISLE.d_clear - AISLE.fixed_overhead
a_tau = AISLE.a_brake * tau_eff
root = aisle.speed_ceiling + a_tau  # equation 2: the ceiling is the root less a tau
v("delta_loc", m(AISLE.delta_loc, "mm"), "{:.0f}", "localization bound, mm")
v("delta_margin", m(AISLE.delta_margin, "mm"), "{:.0f}", "protective clearance, mm")
v("fixed_all", m(AISLE.fixed_overhead, "mm"), "{:.0f}", "all three allowances, mm")
v("d_avail", m(d_avail, "mm"), "{:.0f}", "room to brake, mm")
v("tau_delay", m(AISLE.tau_delay, "ms"), "{:.1f}", "pre-brake delay, ms")
v("ramp_delay", m(drive.ramp_delay, "ms"), "{:.1f}", "jerk ramps as delay, a/2J, ms")
v("tau_eff", m(tau_eff, "ms"), "{:.1f}", "effective delay, ms")
v("a_tau", m(a_tau, "m/s"), "{:.3f}", "a times the effective delay, m/s")
v("root", m(root, "m/s"), "{:.3f}", "square-root term, m/s")
v("ceiling", V_CEIL, "{:.2f}", "pre-dispatch ceiling, m/s")
v("v_plateau", m(drive.jerk_plateau_speed, "m/s"), "{:.2f}", "a^2/J, m/s")
v("t_latest", m(aisle.delay_spare, "ms"), "{:.1f}", "spare as delay at the aisle speed, ms")
v("reserve_d", m(dispatch.spare, "mm"), "{:.1f}", "reserve after dispatch travel, mm")
v("reserve_t", m(dispatch.delay_spare, "ms"), "{:.1f}", "reserve as time, ms")
v("mm_per_ms", m(AISLE_STOP.result.travel_over(AISLE.t_tick), "mm"), "{:.1f}", "travel per ms at the aisle speed, mm")
v("slower_ms", m(SLOWER.delay - AISLE.tau_delay, "ms"), "{:.0f}", "added pre-brake delay of the sensitivity case, ms")
v("ceiling_drop_10ms", m(aisle.speed_ceiling - SLOWER.result.speed_ceiling, "mm/s"), "{:.0f}",
  "ceiling drop for 10 ms more delay, mm/s")

print("== Certified scanner path (SICK nanoScan3 data sheet)")
v("scanner_range", m(SCANNER.protective_field_range, "m"), "{:.0f}", "protective field range, m")
v("scanner_resp", m(SCANNER.response_time, "ms"), "{:.0f}", "response time, ms")
v("scanner_travel", m(AISLE_STOP.result.travel_over(SCANNER.response_time), "mm"), "{:.1f}",
  "travel during the response at the aisle speed, mm")

print("== Estimate: host-enforced lease stall (SE 12.2.1, 12.3)")
stall = STALL.added_delay + AISLE.t_lease
v("t_lease", m(AISLE.t_lease, "ms"), "{:.0f}", "chunk lease, ms")
v("t_stall", m(stall, "ms"), "{:.0f}", "host stall, ms")
v("stall_extra", m(STALL.added_delay, "ms"), "{:.0f}", "stall beyond the lease, ms")
v("stall_extra_s", m(STALL.added_delay, "s"), "{:.3f}", "the same, s")
v("stall_added", m(STALL.result.added_delay_travel, "mm"), "{:.0f}", "extra unbraked travel, mm")
v("stall_breach", m(STALL.result.added_delay_travel - aisle.spare, "mm"), "{:.2f}", "overrun past the spare, mm")
v("t_unwind", m(UNWIND.added_delay, "ms"), "{:.0f}", "integrator unwind, ms")
v("unwind_added", m(UNWIND.result.added_delay_travel, "mm"), "{:.0f}", "unwind travel, mm")
v("unwind_excess", m(UNWIND.result.added_delay_travel - aisle.spare, "mm"), "{:.2f}", "unwind past the spare, mm")

print("== Three ceilings (SE 12.2.1, figure 3)")
v("mu_inspected", AISLE.mu_inspected_floor, "{:.2f}", "inspected-floor friction")
v("a_inspected", A_INSP, "{:.2f}", "inspected-floor deceleration, m/s^2")
v("ceiling_inspected", V_CEIL_INSP, "{:.2f}", "inspected-floor ceiling, m/s")
v("v_human", V_WALK, "{:.1f}", "approach speed, ISO 13855, m/s")
v("v_walk", V_WALKER, "{:.2f}", "walking-person ceiling, m/s")
t_stop_walk = calc_scurve_stop_time(walker.speed_ceiling, AISLE.a_brake, AISLE.j_brake)
v("t_stop_walk", m(t_stop_walk, "ms"), "{:.0f}", "S-curve stop from the walking-person ceiling, ms")
v("closure_walk", m(AISLE.v_human_approach * t_stop_walk, "mm"), "{:.0f}", "person travel during that stop, mm")
v("ceiling_complete", m(dispatch.speed_ceiling, "m/s"), "{:.2f}", "dry ceiling, dispatch charged, m/s")
v("ceiling_inspected_complete", m(INSPECTED_DISPATCH.result.speed_ceiling, "m/s"), "{:.2f}",
  "inspected ceiling, dispatch charged, m/s")
v("v_walk_complete", m(WALKER_DISPATCH.result.speed_ceiling, "m/s"), "{:.2f}", "walking ceiling, dispatch charged, m/s")

print("== Gravity braking tax (SE Napkin Math 12.2)")
v("bias_pct", 100 * JOINT.bias_fraction, "{:.0f}", "share of peak torque gravity holds, %")
v("joint_speed", m(JOINT.joint_speed, "1/s"), "{:.1f}", "joint speed, rad/s")
v("lever", m(JOINT.lever, "m"), "{:.2f}", "lever, m")
v("bias", m(JOINT.bias, "N*m"), "{:.2f}", "bias torque, N m")
v("tau_max", m(JOINT.torque_limit, "N*m"), "{:.0f}", "peak joint torque, N m")
v("inertia", m(JOINT.inertia, "kg*m**2"), "{:.0f}", "joint inertia, kg m^2")
v("torque_down", m(JOINT.torque_descending, "N*m"), "{:.2f}", "braking torque left, descending, N m")
v("alpha_free", m(JOINT.alpha_free, "1/s**2"), "{:.0f}", "deceleration, no bias, rad/s^2")
v("alpha_down", m(JOINT.alpha_descending, "1/s**2"), "{:.2f}", "deceleration, descending, rad/s^2")
v("stop_free", m(JOINT.tool_stop_free, "mm"), "{:.2f}", "tool stop, no bias, mm")
v("stop_down", m(JOINT.tool_stop_descending, "mm"), "{:.2f}", "tool stop, descending, mm")
v("stop_up", m(JOINT.tool_stop_ascending, "mm"), "{:.2f}", "tool stop, ascending, mm")
v("stop_ratio_words", (JOINT.tool_stop_descending / JOINT.tool_stop_free).to("").magnitude, "words",
  "descending over unbiased stop")
v("overrun", m(JOINT.overrun, "mm"), "{:.2f}", "unbudgeted overrun, mm")
v("arm_mass", m(RM.arm.mass, "kg"), "{:.0f}", "arm mass, kg")

print("== Kinetic barrier torque demand (SE 12.4.1, equation 6)")
clear_far, clear_near = JOINT.barrier_clearances
far_demand, near_demand = JOINT.barrier_demands
v("gamma", m(JOINT.barrier_gain, "1/s"), "{:.0f}", "barrier gain, 1/s")
v("clear_far", m(clear_far, "mm"), "{:.0f}", "far clearance, mm")
v("clear_near", m(clear_near, "mm"), "{:.0f}", "near clearance, mm")
v("brake_far", m(far_demand, "N*m"), "{:.1f}", "demand at the far clearance, N m")
v("brake_near", m(near_demand, "N*m"), "{:.1f}", "demand at the near clearance, N m")

print("== Two-axis projection (SE 12.6.1)")
proj = PROJ.projection
v("proj_a1", PROJ.normal[0], "{:.0f}", "normal, joint 1 coefficient")
v("proj_b", PROJ.bound, "{:.0f}", "barrier bound, N m")
v("proj_u1", PROJ.proposal[0], "{:.1f}", "proposal, joint 1, N m")
v("proj_u2", PROJ.proposal[1], "{:.0f}", "proposal, joint 2, N m")
v("proj_value", PROJ.proposal_value, "{:.0f}", "a . u_nom")
v("proj_breach", PROJ.breach, "{:.0f}", "breach")
v("proj_clamp1", PROJ.clamp[0], "{:.0f}", "clamp, joint 1, N m")
v("proj_clamp2", PROJ.clamp[1], "{:.0f}", "clamp, joint 2, N m")
v("proj_clamp_value", PROJ.clamp_value, "{:.0f}", "a . clamp")
v("proj_clamp_pen", PROJ.clamp_penetration, "{:.0f}", "clamp still outside by")
v("proj_lambda", proj["lambda_star"], "{:.0f}", "lambda*")
v("proj_us1", proj["u_projected"][0], "{:.1f}", "projection, joint 1, N m")
v("proj_us2", proj["u_projected"][1], "{:.0f}", "projection, joint 2, N m")
v("proj_norm", proj["correction_norm"], "{:.3f}", "intervention norm, N m")
v("proj_angle_nom", PROJ.angle_deg(PROJ.proposal), "{:.1f}", "proposal direction, deg")
v("proj_angle_clamp", PROJ.angle_deg(PROJ.clamp), "{:.1f}", "clamp direction, deg")
v("qp_us", m(CYCLE.stage("QP solve"), "us"), "{:.0f}", "QP solve allocation, us")
latch_proj = calc_cbf_qp_orthogonal_projection([m(AISLE.v_latch_high, "m/s")], [1.0], m(AISLE.v_latch_approach, "m/s"))
v("latch_lambda", latch_proj["lambda_star"], "{:.2f}", "latch speed projection lambda, m/s")

print("== Latch (SE 12.6.2, equations 11 and 12)")
lo = calc_latch_contact_stopping_profile(AISLE.v_latch_approach, scenario=AISLE)
hi = calc_latch_contact_stopping_profile(AISLE.v_latch_high, scenario=AISLE)
k_exp = math.floor(math.log10(m(AISLE.k_latch, "N/m")))
v("k_latch_mant", m(AISLE.k_latch, "N/m") / 10 ** k_exp, "{:.1f}", "plate stiffness mantissa")
v("k_latch_exp", k_exp, "{:.0f}", "plate stiffness exponent")
v("v_latch", m(AISLE.v_latch_approach, "m/s"), "{:.2f}", "guarded approach, m/s")
v("v_latch_high", m(AISLE.v_latch_high, "m/s"), "{:.2f}", "proposed approach, m/s")
v("df_dt", m(lo["df_dt"], "N/ms"), "{:.0f}", "force rise at the guarded approach, N/ms")
v("f_trip", m(AISLE.f_latch_tripwire, "N"), "{:.0f}", "tripwire, N")
v("t_contact", m(AISLE.t_contact_response, "ms"), "{:.0f}", "contact loop, ms")
v("f_cmd", m(lo["f_at_brake_onset"], "N"), "{:.0f}", "force at brake onset, guarded, N")
v("x_rem", m(lo["x_remaining"], "mm"), "{:.2f}", "plate travel left, mm")
v("f_latch", m(AISLE.f_latch_limit, "N"), "{:.0f}", "latch limit, N")
v("a_arm", m(lo["a_arm_needed"], "m/s**2"), "{:.0f}", "stop deceleration, guarded, m/s^2")
v("t_arm_stop", m(lo["t_arm_stop"], "ms"), "{:.0f}", "stop time, guarded, ms")
v("f_cmd_hi", m(hi["f_at_brake_onset"], "N"), "{:.0f}", "force at brake onset, fast, N")
v("x_rem_hi_um", m(hi["x_remaining"], "um"), "{:.1f}", "plate travel left, fast, um")
v("a_arm_hi", m(hi["a_arm_needed"], "m/s**2"), "{:.0f}", "stop deceleration, fast, m/s^2")

print("== Clamped-hand penalty (SE Napkin Math 12.3)")
m_eff, m_hand, k_hand = RM.arm_effective_contact_mass, AISLE.m_contact_human, AISLE.k_contact_human
m_red = calc_reduced_mass(m_eff, m_hand)
s_free = (k_hand * m_red) ** 0.5
s_clamp = (k_hand * m_eff) ** 0.5
f_free = calc_transient_impact_force(AISLE.v_handover, m_eff, k_hand, human_mass=m_hand)
f_clamp = calc_transient_impact_force(AISLE.v_handover, m_eff, k_hand)
v("m_eff", m(m_eff, "kg"), "{:.0f}", "arm-side effective mass, kg")
v("m_hand", m(m_hand, "kg"), "{:.1f}", "hand effective mass, kg")
v("k_hand_n_mm", m(k_hand, "N/mm"), "{:.0f}", "hand spring, N/mm")
v("k_hand_n_m", m(k_hand, "N/m"), "{:,.0f}", "hand spring, N/m")
v("m_red", m(m_red, "kg"), "{:.3f}", "reduced mass, kg")
v("v_handover", m(AISLE.v_handover, "m/s"), "{:.2f}", "handover speed, m/s")
v("s_free", m(s_free, "N*s/m"), "{:.0f}", "free stiffness term, N per m/s")
v("s_clamp", m(s_clamp, "N*s/m"), "{:.0f}", "clamped stiffness term, N per m/s")
v("f_free", m(f_free, "N"), "{:.1f}", "free-hand force, N")
v("f_clamp", m(f_clamp, "N"), "{:.1f}", "clamped-hand force, N")
v("f_crit", m(AISLE.f_contact_criterion, "N"), "{:.0f}", "force criterion, N")
v("v_free_ceiling", m(calc_transient_contact_speed_limit(AISLE.f_contact_criterion, m_eff, k_hand, human_mass=m_hand), "m/s"),
  "{:.2f}", "free-hand speed ceiling, m/s")
v("v_clamp_ceiling", m(calc_transient_contact_speed_limit(AISLE.f_contact_criterion, m_eff, k_hand), "m/s"),
  "{:.3f}", "clamped-hand speed ceiling, m/s")

print("== Kirschner et al. 2022, Table II, hand, B1")
KIRSCHNER = {  # robot: (allowed speed m/s, force at C N, force at N N)
    "UR10e": (FORCES.KirschnerUr10eSpeedMps, FORCES.KirschnerUr10eForceCentralN, FORCES.KirschnerUr10eForceOuterN),
    "Franka (FE)": (FORCES.KirschnerFrankaSpeedMps, FORCES.KirschnerFrankaForceCentralN,
                    FORCES.KirschnerFrankaForceOuterN),
    "KUKA LBR iiwa 14": (FORCES.KirschnerIiwaSpeedMps, FORCES.KirschnerIiwaForceCentralN,
                         FORCES.KirschnerIiwaForceOuterN),
    "Techman TM5": (FORCES.KirschnerTm5SpeedMps, FORCES.KirschnerTm5ForceCentralN, FORCES.KirschnerTm5ForceOuterN),
}
KIRSCHNER = {k: tuple(float(x) for x in row) for k, row in KIRSCHNER.items()}
F_TS = float(FORCES.KirschnerHandForceThresholdN)
forces = [f for _, c, n in KIRSCHNER.values() for f in (c, n)]
speeds = [s for s, _, _ in KIRSCHNER.values()]
v("f_ts", F_TS, "{:.0f}", "force the speeds were set for, N")
v("kirschner_min", min(forces), "{:.0f}", "smallest measured force, N")
v("kirschner_max", max(forces), "{:.0f}", "largest measured force, N")
v("kirschner_excess", max(forces) - F_TS, "{:.0f}", "largest over the threshold, N")
v("kirschner_ratio", max(forces) / F_TS, "{:.2f}", "largest over the threshold, ratio")
v("kirschner_v_min", min(speeds), "{:.2f}", "slowest allowed speed, m/s")
v("kirschner_v_max", max(speeds), "{:.2f}", "fastest allowed speed, m/s")

print("== Speed and separation monitoring (SE 12.8.3)")
v("t_stop_down", m(JOINT.stop_time_descending, "ms"), "{:.0f}", "descending joint stop, ms")
v("v_tcp", m(JOINT.tcp_speed, "m/s"), "{:.2f}", "descending tool speed, m/s")
v("ssm_person", m(JOINT.separation_person, "mm"), "{:.0f}", "person travel, mm")
v("ssm_reaction", m(JOINT.separation_reaction, "mm"), "{:.0f}", "robot reaction travel, mm")
v("ssm_stop", m(JOINT.tool_stop_descending, "mm"), "{:.0f}", "robot stop travel, mm")
v("ssm_total", m(JOINT.separation_total, "mm"), "{:.0f}", "sum, mm")
v("ssm_share", 100 * JOINT.separation_person_share, "{:.0f}", "person share, %")

print("== Cycle budget and the hold (SE 12.6.3, 12.6.4)")
v("cyc_state", m(CYCLE.stage("state acquisition"), "us"), "{:.0f}", "state in, us")
v("cyc_lease", m(CYCLE.stage("proposal and lease check"), "us"), "{:.0f}", "proposal and lease, us")
v("cyc_tracking", m(CYCLE.stage("tracking check"), "us"), "{:.0f}", "tracking, us")
v("cyc_output", m(CYCLE.stage("output write"), "us"), "{:.0f}", "output, us")
v("cyc_reserve", m(CYCLE.stage("reserve"), "us"), "{:.0f}", "reserve, us")
v("wcet", m(CYCLE.total, "us"), "{:.0f}", "declared WCET, us")
v("deadline", m(AISLE.enforcer_deadline, "us"), "{:.0f}", "deadline, us")
v("unloaded", m(AISLE.enforcer_unloaded, "us"), "{:.0f}", "unloaded run, us")
v("hold_accel", m(HOLD.acceleration, "m/s**2"), "{:.0f}", "approach-acceleration bound, m/s^2")
v("hold_drift", m(HOLD.drift, "mm"), "{:.3f}", "drift over one held tick, mm")

print("== STOP against STO (SE 12.8.1, 12.8.2)")
v("t_stop_aisle", m(aisle.stop_time, "ms"), "{:,.2f}", "S-curve stop from the aisle speed, ms")
v("d_stop_aisle", m(aisle.braking_travel + aisle.jerk_ramp_travel, "mm"), "{:.2f}", "its distance, mm")
v("heartbeat_timeout", m(AISLE.heartbeat_timeout, "ms"), "{:.0f}", "heartbeat trips within, ms")
v("brake_set_min", m(RM.spring_brake_engage_min, "ms"), "{:.0f}", "spring brakes set, earliest, ms")
v("brake_set_max", m(RM.spring_brake_engage_max, "ms"), "{:.0f}", "spring brakes set, latest, ms")

print("== Cruise, October 2, 2023 (SE War Story 12.2)")
v("cruise_drag_ft", REPORTS.CruisePulloverDistanceFt, "{:.0f}", "pullover distance, ft")
v("cruise_drag_m", m(Q_(float(REPORTS.CruisePulloverDistanceFt), "ft"), "m"), "{:.1f}", "the same, m")
v("cruise_mph", REPORTS.CruisePulloverMaxSpeedMph, "{:.1f}", "pullover speed, mph")
v("cruise_ms", m(Q_(float(REPORTS.CruisePulloverMaxSpeedMph), "mph"), "m/s"), "{:.1f}", "the same, m/s")
v("cruise_recall", REPORTS.CruiseRecallVehicles, "{:.0f}", "vehicles recalled")

# Plot inputs for make_figs.py (SI magnitudes and recorded rows).
STOP_UP, STOP_FREE, STOP_DOWN = (m(q, "mm") for q in (JOINT.tool_stop_ascending, JOINT.tool_stop_free,
                                                       JOINT.tool_stop_descending))
TAU_MAX, TAU_BIAS = m(JOINT.torque_limit, "N*m"), m(JOINT.bias, "N*m")
CLEAR_FAR, CLEAR_NEAR = m(clear_far, "m"), m(clear_near, "m")
v("brake_zero", m(JOINT.barrier_torque_demand(JOINT.tool_stop_descending), "N*m"), "{:.1f}",
  "demand at the barrier's zero set, N m")


def barrier_demand(clear):
    """Braking torque in N m the kinetic barrier demands at a tool clearance in m."""
    return m(JOINT.barrier_torque_demand(Q_(clear, "m")), "N*m")


V_LATCH, V_LATCH_HI = m(AISLE.v_latch_approach, "m/s"), m(AISLE.v_latch_high, "m/s")
K_PLATE, TRIP = m(AISLE.k_latch, "N/m"), m(AISLE.f_latch_tripwire, "N")
T_LOOP, F_LATCH = m(AISLE.t_contact_response, "s"), m(AISLE.f_latch_limit, "N")
F_FREE, F_CLAMP, F_CRIT = m(f_free, "N"), m(f_clamp, "N"), m(AISLE.f_contact_criterion, "N")
S_H, S_R, S_STOP = (m(q, "mm") for q in (JOINT.separation_person, JOINT.separation_reaction,
                                         JOINT.tool_stop_descending))
STAGES = [(name, m(q, "us")) for name, q in CYCLE.stages]
WCET, DEADLINE, UNLOADED = m(CYCLE.total, "us"), m(AISLE.enforcer_deadline, "us"), m(AISLE.enforcer_unloaded, "us")
BRAKE_SET = (m(RM.spring_brake_engage_min, "ms"), m(RM.spring_brake_engage_max, "ms"))

# The two-axis example for the figure.
u, a, b = PROJ.proposal, PROJ.normal, PROJ.bound
clamp, ustar = PROJ.clamp, proj["u_projected"]
LATCH_LO, LATCH_HI = lo, hi

if __name__ == "__main__":
    write(__file__)
