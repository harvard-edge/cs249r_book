#!/usr/bin/env python3
"""Every number on the Lecture 11 slides and notes (Trajectory Planning), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* running machine and its aisle
record, the Panda-class joint in Actuators.*, Literature.* for Diffusion
Policy, pi0, and Ruckig) and the calculations (Scenarios.*, backed by the
solvers in engine/solvers/trajectory.py); the chapter reads the same
scenarios. This file only selects values, converts units, and formats them;
it types no numbers. Each value is recorded with v() under the key the deck
prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports it for the plotted quantities (module-level names below)
and the formatted labels (val()).

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Embodied, Scenarios, Literature, Actuators = (mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature,
                                              mlsysim.Actuators)
from mlsysim.physics.robotics import calc_dls_joint_rate, calc_two_link_singular_values  # noqa: E402

Q_ = mlsysim.core.units.Q_

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
JOINT = Actuators.PandaClassJoint.CSG_25_50_ILM70x18
VLA = Literature.VlaSystems
RUCKIG = Literature.TrajectoryGeneration

MISSION = Scenarios.GenericReplannerMissionExposure
GOOD, BAD = Scenarios.WarehouseArmRightAngleElbow, Scenarios.WarehouseArmNearStraightElbow
REPLACE = Scenarios.WarehouseArmChunkReplacement
ANCHOR = Scenarios.WarehouseArmStaleAnchor
SWING_S = Scenarios.WarehouseArmShoulderSwing
SEAM_S = Scenarios.PandaClassJointSeamBridge
SHORT_S = Scenarios.PandaClassJointSeamBridgeCompressed
HALF_S = Scenarios.PandaClassJointSeamBridgeHalved
ARM_S = Scenarios.WarehouseArmStallStop
MUG_S = Scenarios.WarehouseArmOpenMugStop
SCHED_S = Scenarios.WarehouseArmCommitmentSchedule
BASE_S = Scenarios.WarehouseBaseJerkLimitedStop
WINDOW_S = Scenarios.WarehouseBaseDynamicWindow


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


mission, good, bad, repl, anchor = MISSION.result, GOOD.result, BAD.result, REPLACE.result, ANCHOR.result
swing, seam, short, half = SWING_S.result, SEAM_S.result, SHORT_S.result, HALF_S.result
arm, mug, sched, base, window = ARM_S.result, MUG_S.result, SCHED_S.result, BASE_S.result, WINDOW_S.result

# =============================================================================
# Module-level quantities make_figs.py plots (units in the comments)
# =============================================================================
L1, L2 = m(RM.arm.upper_arm_length, "m"), m(RM.arm.forearm_length, "m")   # m
V_TCP = m(RM.tcp_speed_limit, "m/s")
QDOT_MAX = m(RM.arm.joint_velocity_limits[1], "rad/s")
QDDOT_MAX = m(RM.arm.joint_acceleration_limits[1], "rad/s**2")
TAU_MAX = m(RM.arm.max_torque, "N*m")
LAMBDA = m(GOOD.damping, "m")
Q2_GOOD, Q2_BAD = round(m(GOOD.elbow_angle, "degree")), round(m(BAD.elbow_angle, "degree"))
JAC = {
    Q2_GOOD: dict(exact=m(good.joint_rate_norm, "rad/s"), damped=m(good.damped_joint_rate, "rad/s")),
    Q2_BAD: dict(exact=m(bad.joint_rate_norm, "rad/s"), damped=m(bad.damped_joint_rate, "rad/s")),
}


def sing_vals(q2_deg):
    """(sigma_max, sigma_min) of the arm's planar Jacobian at elbow angle q2_deg, m per rad."""
    s_max, s_min = calc_two_link_singular_values(RM.arm.upper_arm_length, RM.arm.forearm_length,
                                                 Q_(float(q2_deg), "degree"))
    return m(s_max, "m"), m(s_min, "m")


def dls_rate(sigma):
    """Damped joint rate (rad/s) for the TCP limit along a singular value sigma (m per rad)."""
    return m(calc_dls_joint_rate(RM.tcp_speed_limit, Q_(sigma, "m"), GOOD.damping)[0], "rad/s")


P_LATE, MISSION_BUDGET, P_TARGET = MISSION.late_probability, MISSION.mission_budget, mission.per_handoff_budget
T_AGE, T_DA = m(AISLE.perception_age, "ms"), m(AISLE.t_dispatch_to_admission, "ms")
T_INFER, T_VALIDATE = m(RM.chunk_inference_latency, "ms"), m(AISLE.t_plan_validate, "ms")
T_REPL, T_RENEW, T_LEASE = m(repl.latency, "ms"), m(REPLACE.renewal_period, "ms"), m(AISLE.t_lease, "ms")
T_TICK = m(AISLE.t_tick, "ms")
T_SETPOINT, N_SETPOINTS = m(RM.chunk_step, "ms"), RM.chunk_horizon
T_NOMINAL = m(sched.nominal_end, "ms")
CHUNKS = [  # (name, predicted steps, executed steps)
    ("Diffusion Policy, Push-T", round(VLA.DiffusionPolicyPredictionHorizon),
     round(VLA.DiffusionPolicyRealPushTExecutionHorizon)),
    (f"pi0, {VLA.Pi0ControlRateHz:.0f} Hz robots", round(VLA.Pi0ChunkHorizon), round(VLA.Pi0ExecutedActions50Hz)),
    ("Running machine", N_SETPOINTS, round(sched.executed_setpoints)),
]
W_JOINT, T_DELAY, GAP = m(ANCHOR.joint_speed, "rad/s"), m(ANCHOR.delay, "s"), m(anchor.gap, "rad")
SWING = m(SWING_S.swing, "rad")
TAU_GRAV, J_EFF = m(swing.holding_torque, "N*m"), m(swing.effective_inertia, "kg*m**2")
A_UP, A_DN = m(swing.accel_up, "rad/s**2"), m(swing.accel_down, "rad/s**2")
V_PEAK_SW, JERK_SW = m(swing.peak_speed, "rad/s"), m(swing.jerk_bound, "rad/s**3")
T_BANG, T_S = m(swing.bang_bang_time, "s"), m(swing.jerk_limited_time, "s")
DW, T_BRIDGE = m(SEAM_S.velocity_gap, "rad/s"), m(SEAM_S.bridge_duration, "ms")
J_REFL = m(seam.reflected_inertia, "kg*m**2")
TAU_STEP, TAU_BRIDGE = m(seam.step_torque, "N*m"), m(seam.peak_torque, "N*m")
TAU_RATED, TAU_PEAK = m(JOINT.rated_torque, "N*m"), m(JOINT.peak_torque, "N*m")
A_ARM, J_ARM = m(ARM_S.deceleration, "m/s**2"), m(arm.jerk, "m/s**3")
T_PATH, D_PATH = m(arm.delay, "ms"), m(arm.delay_travel, "mm")
D_STOP_ARM, T_STOP_ARM = m(arm.scurve_travel, "mm"), m(arm.stop_time, "s")
D_CLEAR_ARM, TOTAL_ARM = m(ARM_S.clear_distance, "mm"), m(arm.required_distance, "mm")
A_MUG, D_STOP_MUG, TOTAL_MUG = (m(mug.spill_deceleration, "m/s**2"), m(mug.stop.scurve_travel, "mm"),
                                m(mug.stop.required_distance, "mm"))
T_COMMIT, T_PREF, T_CUTOFF = m(SCHED_S.commit, "ms"), m(sched.preferred_admission, "ms"), m(sched.admission_cutoff, "ms")
T_END, TOTAL_ALT = m(sched.rest, "ms"), m(sched.used, "mm")
V_BASE, A_BASE, J_BASE = m(BASE_S.speed, "m/s"), m(BASE_S.deceleration, "m/s**2"), m(base.jerk, "m/s**3")
D_CONST, D_SC, D_POLY = m(base.braking_travel, "mm"), m(base.scurve_travel, "mm"), m(base.polynomial_distance, "mm")
T_CONST, T_SC, T_POLY = m(base.braking_time, "ms"), m(base.stop_time, "ms"), m(base.polynomial_time, "ms")
RUCKIG_MEAN = m(Q_(float(RUCKIG.RuckigMeanUs), "us"), "ms")
RUCKIG_WORST = m(Q_(float(RUCKIG.RuckigWorstUs), "us"), "ms")

# =============================================================================
# Values the slides and notes print
# =============================================================================
print("== Hook: handoff deadlines across a mission (Napkin Math 11.1)")
v("n_handoff", mission.handoffs, "{:,.0f}", "handoffs per mission")
v("mission_min_words", m(MISSION.mission_time, "min"), "words", "mission, minutes")
v("mission_s", m(MISSION.mission_time, "s"), "{:.0f}", "mission, s")
v("f_replan", m(MISSION.replan_rate, "Hz"), "{:.0f}", "replanning rate, Hz")
v("p99_on_time", 100 * (1 - P_LATE), "{:.0f}", "plans on time per hundred at p99")
v("p99_of", mission.handoffs_per_expected_late, "{:.0f}", "one late plan in this many")
v("p_late", P_LATE, "{:.2f}", "late probability per handoff")
v("p_on_time", 1 - P_LATE, "{:.2f}", "on-time probability per handoff")
v("mission_one_in", 1 / MISSION_BUDGET, "{:,.0f}", "mission target, one failure in")
p_all_exp = -math.floor(math.log10(mission.all_on_time))
v("p_all_mant", mission.all_on_time * 10 ** p_all_exp, "{:.0f}", "P(all on time), mantissa")
v("p_all_exp", p_all_exp, "{:.0f}", "P(all on time), negative exponent")
v("n_late", mission.expected_late, "{:.0f}", "expected late handoffs")
v("one_in_millions", 1 / P_TARGET / 10 ** 6, "{:.0f}", "per-handoff budget, one late plan in this many million")
v("q_design", 100 * mission.design_quantile, "{:.5f}", "design quantile, %")
v("orders_deeper", mission.orders_deeper, "{:.1f}", "orders of magnitude deeper than p99")
v("p_target_one_in", 1 / P_TARGET, "{:,.0f}", "per-handoff budget, one in")

print("== Recall and the opening (Sec. 11.1)")
v("r_tol", m(AISLE.grasp_tolerance, "mm"), "{:.0f}", "intent-lease grasp tolerance, mm")
v("t_infer", T_INFER, "{:.0f}", "chunk pass p99, ms")
v("t_tail", m(REPLACE.interference_tail, "ms"), "{:.0f}", "interference tail, ms")
v("v_tcp", V_TCP, "{:.0f}", "TCP speed limit, m/s")
v("n_setpoints", N_SETPOINTS, "{:.0f}", "setpoints per chunk")
v("t_setpoint", T_SETPOINT, "{:.0f}", "setpoint period, ms")
v("t_nominal", T_NOMINAL, "{:.0f}", "nominal end, ms")
v("dw", DW, "{:.1f}", "seam velocity gap, rad/s")
v("t_lease", T_LEASE, "{:.0f}", "chunk lease, ms")
v("t_path", T_PATH, "{:.0f}", "lease path, ms")
v("t_rest_run", m(arm.rest_time, "ms"), "{:.0f}", "running lease: at rest, ms")

print("== Jacobian and damped least squares (Sec. 11.2.1)")
v("qd_max", QDOT_MAX, "{:.3f}", "joint velocity limit, rad/s")
v("q2_good", Q2_GOOD, "{:.0f}", "right-angle elbow, degrees")
v("q2_bad", Q2_BAD, "{:.0f}", "nearly straight elbow, degrees")
v("qd_norm_good", JAC[Q2_GOOD]["exact"], "{:.1f}", "joint-rate norm at the right angle, rad/s")
v("qd_one_good", m(good.joint_rate_one_joint, "rad/s"), "{:.1f}", "at least one joint, rad/s")
v("qd_norm_bad", JAC[Q2_BAD]["exact"], "{:.1f}", "joint-rate norm near straight, rad/s")
v("lam_mm", m(GOOD.damping, "mm"), "{:.0f}", "damping, mm")
v("qd_dls_good", JAC[Q2_GOOD]["damped"], "{:.1f}", "damped joint rate at the right angle, rad/s")
v("qd_dls_bad", JAC[Q2_BAD]["damped"], "{:.1f}", "damped joint rate near straight, rad/s")
v("frac_good", 100 * good.realized_fraction, "{:.1f}", "realized tool motion at the right angle, %")
v("frac_bad", 100 * bad.realized_fraction, "{:.1f}", "realized tool motion near straight, %")
v("qd_bound", m(good.damped_ceiling, "rad/s"), "{:.0f}", "damped ceiling, rad/s per (m/s)")

print("== Trajectory record (Sec. 11.2.2, 11.8)")
v("d_stop_arm", D_STOP_ARM, "{:.1f}", "arm S-curve stop, mm")
v("t_stop_arm", m(arm.stop_time, "ms"), "{:.0f}", "arm S-curve stop, ms")
v("t_commit", T_COMMIT, "{:.0f}", "commitment instant, ms")
v("t_rest", T_END, "{:.0f}", "at rest, ms")
v("m_payload", m(RM.arm.payload_capacity, "kg"), "{:.0f}", "payload bound, kg")

print("== Replacement latency (equation 3)")
v("t_age", T_AGE, "{:.1f}", "capture to dispatch, ms")
v("t_da", T_DA, "{:.1f}", "dispatch to admission, ms")
v("t_validate", T_VALIDATE, "{:.0f}", "whole-plan validation, ms")
v("t_other", m(repl.other_stages, "ms"), "{:.1f}", "link, decoding, staging, tick wait, ms")
v("t_repl", T_REPL, "{:.1f}", "replacement latency, ms")
v("t_renew", T_RENEW, "{:.0f}", "renewal period, ms")
v("repl_over_renew", repl.latency_over_renewal, "{:.1f}", "latency over renewal")
v("n_inflight", repl.requests_in_flight, "{:.0f}", "requests in flight, at least")
v("n_inflight_words", repl.requests_in_flight, "words", "the same, in words")

print("== Chunk configurations (Sec. 11.3.2)")
v("dp_predict", CHUNKS[0][1], "{:.0f}", "Diffusion Policy predicted steps")
v("dp_execute", CHUNKS[0][2], "{:.0f}", "Diffusion Policy executed steps (real Push-T)")
v("pi0_h", CHUNKS[1][1], "{:.0f}", "pi0 chunk, actions")
v("pi0_hz", VLA.Pi0ControlRateHz, "{:.0f}", "pi0 robots, Hz")
v("pi0_exec", CHUNKS[1][2], "{:.0f}", "pi0 actions executed before re-inference")
v("pi0_reinfer", VLA.Pi0ExecutedActions50Hz / VLA.Pi0ControlRateHz, "{:.1f}", "pi0 re-inference interval, s")
v("rm_exec_words", sched.executed_setpoints, "words", "running machine setpoints per renewal")

print("== Stale anchor (Sec. 11.3.3)")
v("w_joint", W_JOINT, "{:.2f}", "joint speed, rad/s")
v("t_delay", m(ANCHOR.delay, "ms"), "{:.0f}", "replacement delay, ms")
v("gap", GAP, "{:.3f}", "joint gap, rad")
v("gap_deg", m(anchor.gap_degrees, "degree"), "{:.2f}", "joint gap, degrees")
v("reach", m(SWING_S.reach, "m"), "{:.2f}", "planar reach, m")
v("d_tool", m(anchor.tool_displacement, "mm"), "{:.0f}", "tool displacement, mm")

print("== Joint limits and the loaded shoulder (Sec. 11.4.1)")
v("qdd_max", QDDOT_MAX, "{:.1f}", "datasheet acceleration, rad/s^2")
v("tau_max", TAU_MAX, "{:.0f}", "joint torque limit, N m")
v("tau_grav", TAU_GRAV, "{:.1f}", "holding torque, N m")
v("grav_pct", 100 * swing.holding_share, "{:.1f}", "share of the torque limit, %")

print("== Seam spike and bridge (Napkin Math 11.2, Checkpoint 11.2)")
v("t_tick", T_TICK, "{:.0f}", "servo tick, ms")
v("f_arm_khz", m(RM.arm.control_frequency, "kHz"), "{:.0f}", "arm joint loop, kHz")
v("j_refl", J_REFL, "{:.4f}", "reflected rotor inertia, kg m^2")
v("tau_step", TAU_STEP, "{:.1f}", "one-tick torque, N m")
v("tau_rated", TAU_RATED, "{:.0f}", "rated torque, N m")
v("tau_peak", TAU_PEAK, "{:.0f}", "repeated peak torque, N m")
v("rate_step", m(seam.step_torque_rate, "N*m/s"), "{:,.0f}", "one-tick torque rate, N m/s")
v("step_over", seam.step_over_rate_limit, "{:.1f}", "multiple of the torque-rate limit")
v("taud_max", m(RM.arm.joint_torque_rate_limit, "N*m/s"), "{:,.0f}", "torque-rate limit, N m/s")
v("t_bridge", T_BRIDGE, "{:.0f}", "bridge duration, ms")
v("tau_bridge", TAU_BRIDGE, "{:.1f}", "bridge peak torque, N m")
v("rate_bridge", m(seam.torque_rate, "N*m/s"), "{:.1f}", "bridge torque rate, N m/s")
v("bridge_pct", 100 * seam.torque_rate_share, "{:.1f}", "bridge share of the limit, %")
v("peak_reduction", seam.peak_reduction, "{:.1f}", "step torque over bridge peak")
v("gear_ratio", JOINT.gear_ratio, "{:.0f}", "gear ratio")
v("gear_ratio_sq", JOINT.gear_ratio ** 2, "{:,.0f}", "gear ratio squared")
v("j_eff", J_EFF, "{:.2f}", "inertia with the payload, kg m^2")
v("tau_step_loaded", m(seam.loaded_step_torque, "N*m"), "{:.0f}", "one-tick step with the payload, N m")
v("t_short", m(SHORT_S.bridge_duration, "ms"), "{:.0f}", "compressed bridge, ms")
v("t_short_s", m(SHORT_S.bridge_duration, "s"), "{:.2f}", "compressed bridge, s")
v("t_short_sq", m(SHORT_S.bridge_duration, "s") ** 2, "{:.4f}", "compressed bridge squared, s^2")
v("a_short", m(short.peak_acceleration, "rad/s**2"), "{:.0f}", "compressed peak acceleration, rad/s^2")
v("tau_short", m(short.peak_torque, "N*m"), "{:.1f}", "compressed peak torque, N m")
v("jerk_short", m(short.peak_jerk, "rad/s**3"), "{:,.0f}", "compressed peak jerk, rad/s^3")
v("rate_short", m(short.torque_rate, "N*m/s"), "{:,.0f}", "compressed torque rate, N m/s")
v("short_over", short.torque_rate_share, "{:.1f}", "compressed, multiple of the limit")
v("t_bridge_min", m(seam.shortest_bridge, "ms"), "{:.1f}", "shortest bridge inside the limit, ms")
v("halving_rate_words", (half.torque_rate / seam.torque_rate).to("").magnitude, "words", "torque-rate factor when the bridge halves")

print("== Time-optimal swing (Napkin Math 11.3)")
v("swing", SWING, "{:.1f}", "swing, rad")
v("a_up", m(swing.accel_up, "rad/s**2"), "{:.1f}", "torque-limited lift, rad/s^2")
v("a_dn", m(swing.accel_down, "rad/s**2"), "{:.1f}", "torque-limited braking, rad/s^2")
v("t_bang", m(swing.bang_bang_time, "ms"), "{:.1f}", "bang-bang swing, ms")
v("v_peak_sw", V_PEAK_SW, "{:.2f}", "peak speed, rad/s")
v("tau_switch", m(swing.switch_torque_step, "N*m"), "{:.1f}", "switch-point torque step, N m")
v("jerk_sw", JERK_SW, "{:.1f}", "jerk bound from the torque rate, rad/s^3")
v("t_scurve_sw", m(swing.jerk_limited_time, "ms"), "{:.1f}", "jerk-limited swing, ms")
v("t_penalty", m(swing.penalty, "ms"), "{:.1f}", "penalty, ms")
v("penalty_pct", 100 * swing.penalty_share, "{:.1f}", "penalty, %")

print("== Ruckig's arm stop and the stall runway (Napkin Math 11.4, Sec. 11.5.2)")
v("a_arm", A_ARM, "{:.0f}", "arm deceleration bound, m/s^2")
v("j_arm", J_ARM, "{:.1f}", "arm jerk bound, m/s^3")
v("d_brake_arm", m(arm.braking_travel, "mm"), "{:.0f}", "constant-deceleration part, mm")
v("t_brake_arm", m(arm.braking_time, "ms"), "{:.0f}", "constant-deceleration stop, ms")
v("d_ramp_arm", m(arm.jerk_ramp_travel, "mm"), "{:.1f}", "jerk-ramp part, mm")
v("t_response", m(SCHED_S.response, "ms"), "{:.0f}", "tick, bus, and onset, ms")
v("t_onset", m(AISLE.t_brake_onset, "ms"), "{:.0f}", "brake onset charged to the arm, ms")
v("d_path", D_PATH, "{:.0f}", "lease-path travel, mm")
v("total_arm", TOTAL_ARM, "{:.1f}", "lease path + stop, mm")
v("d_clear_arm", D_CLEAR_ARM, "{:.0f}", "clearance, mm")
v("margin_arm", m(arm.spare, "mm"), "{:.1f}", "margin, mm")
v("r_mug", m(MUG_S.inner_radius, "mm"), "{:.0f}", "mug inner radius, mm")
v("h_free", m(MUG_S.freeboard, "mm"), "{:.0f}", "freeboard, mm")
v("rise_free", m(mug.rise_at_free_deceleration, "mm"), "{:.1f}", "liquid rise at the free-space stop, mm")
v("a_mug", A_MUG, "{:.2f}", "spill deceleration bound, m/s^2")
v("j_mug", m(mug.stop.jerk, "m/s**3"), "{:.0f}", "jerk at that peak, m/s^3 (slide)")
v("j_mug_1", m(mug.stop.jerk, "m/s**3"), "{:.1f}", "jerk at that peak, m/s^3 (notes)")
v("d_stop_mug", D_STOP_MUG, "{:.1f}", "open-mug stop, mm")
v("total_mug", TOTAL_MUG, "{:.1f}", "open-mug lease path + stop, mm")
v("d_mug_reused", m(mug.reused_jerk_stop_distance, "mm"), "{:.1f}", "open-mug stop with the 66.7 jerk, mm")
v("v_carry", m(mug.carry_speed, "m/s"), "{:.2f}", "carry speed that fits, m/s")

print("== The record's own deadlines (Sec. 11.5.1 to 11.5.3)")
v("t_request", m(SCHED_S.request_delay, "ms"), "{:.0f}", "request, ms")
v("q_latency", m(SCHED_S.latency_target, "ms"), "{:.0f}", "tail latency budget, ms")
v("t_sched_slack", m(SCHED_S.schedule_slack, "ms"), "{:.0f}", "scheduling slack, ms")
v("t_pref", T_PREF, "{:.0f}", "preferred admission, ms")
v("t_cutoff", T_CUTOFF, "{:.0f}", "admission cutoff, ms")
v("total_alt", TOTAL_ALT, "{:.1f}", "charged cruise + stop, mm")
v("t_wait", m(SCHED_S.late_wait, "ms"), "{:.0f}", "late wait at commitment, ms")
v("late_cruise", m(sched.late_cruise, "mm"), "{:.0f}", "late cruise, mm")
v("late_breach", m(sched.late_breach, "mm"), "{:.1f}", "breach, mm")

print("== Base stop at the drive limit (Sec. 11.5.4)")
v("v_base", V_BASE, "{:.1f}", "drive limit, m/s")
v("a_base", A_BASE, "{:.0f}", "credible deceleration, m/s^2")
v("j_base", J_BASE, "{:.2f}", "jerk bound, m/s^3")
v("d_clear_base", m(AISLE.d_clear, "m"), "{:.2f}", "rack-end clear distance, m")
v("t_const", T_CONST, "{:.0f}", "constant-deceleration stop, ms")
v("d_const", D_CONST, "{:.1f}", "constant-deceleration braking, mm")
v("t_sc_round", T_SC, "{:,.0f}", "S-curve stop, ms (title)")
v("t_sc", T_SC, "{:,.2f}", "S-curve stop, ms")
v("d_sc", D_SC, "{:.2f}", "S-curve stop, mm")
v("t_poly", T_POLY, "{:,.0f}", "polynomial stop, ms")
v("d_poly", D_POLY, "{:.2f}", "polynomial stop, mm")
v("d_jerk_inc", m(base.jerk_ramp_travel, "mm"), "{:.2f}", "S-curve adds, mm")
v("t_equiv", m(base.ramp_delay, "ms"), "{:.1f}", "equivalent cruise (jerk ramps as delay), ms")
v("total_const", m(base.constant_required, "mm"), "{:.1f}", "constant stopping total, mm")
v("total_sc", m(base.required_distance, "mm"), "{:,.2f}", "S-curve stopping total, mm")
v("total_poly", m(base.polynomial_required, "mm"), "{:,.2f}", "polynomial stopping total, mm")
v("overrun_base", m(-base.spare, "mm"), "{:.2f}", "overrun past the clearance, mm")
v("delay_base", m(base.delay_travel, "mm"), "{:.1f}", "delay travel at the drive limit, mm")
v("tau_delay", m(AISLE.tau_delay, "ms"), "{:.1f}", "pre-brake delay, ms")
v("overhead_base", m(base.overhead, "mm"), "{:.0f}", "localization and protective allowances, mm")

print("== Generators and the base's window (Sec. 11.6, table 4)")
v("ruckig_mean", RUCKIG.RuckigMeanUs, "{:.1f}", "Ruckig mean, us")
v("ruckig_worst", RUCKIG.RuckigWorstUs, "{:.0f}", "Ruckig worst case, us")
v("ruckig_axes_words", RUCKIG.RuckigBenchmarkAxes, "words", "axes in the benchmark")
v("dv_window", m(window.half_width, "m/s"), "{:.1f}", "dynamic window half-width, m/s")
v("v_admissible", m(window.admissible_speed, "m/s"), "{:.2f}", "textbook admissible speed, m/s")


if __name__ == "__main__":
    write(__file__)
