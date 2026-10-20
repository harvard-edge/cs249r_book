#!/usr/bin/env python3
"""Every number on the Lecture 4 slides and notes (The Nervous System), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* machine and its site record, the
Actuators.* safety relay, Literature.*) and the calculations (Scenarios.*,
each backed by a solver in engine/solvers/realtime.py, embodied.py, or
plant.py); the chapter reads the same scenarios. This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex. make_figs.py imports it for the scenario
objects and the measured samples.

Measured data in data/ (run by the lecture authors, 19 October 2026, on an
Apple M5 Max laptop, macOS 26.4, C compiled with cc -O2), summarized by
MLSysIM's LatencyTailModel:
  jitter_light.npy / jitter_heavy.npy: lateness in microseconds of 60,000
    releases of a 1 ms periodic task (tick.c), first with the machine's
    ordinary background work, then with 18 memory-streaming load processes
    (load.c); run_jitter.sh.
  mailbox_<variant>.npy: 20,000 reads of a 1 kHz real-time reader against a
    20 Hz writer that stalls 3 ms mid-write on every 4th write (mailbox.c);
    columns: read time in ns, outcome (0 fresh, 1 kept prior, 2 rejected torn).

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Embodied, Actuators, Scenarios, Literature = (mlsysim.Embodied, mlsysim.Actuators,
                                              mlsysim.Scenarios, mlsysim.Literature)
Q_ = mlsysim.core.units.Q_
TAIL = mlsysim.solvers.LatencyTailModel()

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
RELAY = Actuators.SafetyRelay.Pilz_PNOZ_s4
VLA = Literature.VlaSystems

L1_STOP = Scenarios.WarehouseAisleStopAtDriveLimit
BODY = Scenarios.WarehouseBaseStopBodyTerms
ELEC = Scenarios.WarehouseBaseHubDriveElectrical
LOOP = Scenarios.WarehouseJointLoopThroughTick
LEASE = Scenarios.WarehouseChunkLease
LEASE_LONG = Scenarios.WarehouseChunkLeaseLengthened
RTA = Scenarios.WarehousePermissionTickResponse
RTA_LONG = Scenarios.WarehousePermissionTickLongBlocking
CAN_ONE = Scenarios.WarehouseServoCanFdOneBus
CAN_ARM = Scenarios.WarehouseServoCanFdArmBus
CAN_BASE = Scenarios.WarehouseServoCanFdBaseBus
ECAT = Scenarios.WarehouseServoEtherCat

DATA = Path(__file__).resolve().parent / "data"
PERIOD = Q_(1, "ms")   # the measured task's period, and the tick a reader must not exceed


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def travel(t):
    """Travel at the drive limit during ``t`` (MLSysIM's stopping result prices it)."""
    return BODY.result.travel_over(t)


lease, lease_long = LEASE.result, LEASE_LONG.result
loop, rta, rta_long = LOOP.result, RTA.result, RTA_LONG.result
one, arm, base, ecat = CAN_ONE.result, CAN_ARM.result, CAN_BASE.result, ECAT.result

print("== The running machine and its rates")
v("v_drive", m(RM.base.max_velocity, "m/s"), "{:.1f}", "drive limit, m/s")
v("v_mm_per_ms", m(travel(Q_(1, "ms")), "mm"), "{:.1f}", "travel per millisecond, mm")
v("t_intent", m(RM.intent_inference_latency, "ms"), "{:.0f}", "intent model per evaluation, ms (computed)")
v("f_intent", m(RM.intent_rate, "Hz"), "{:.0f}", "intent model rate, Hz")
v("t_intent_period", m(1 / RM.intent_rate, "ms"), "{:.0f}", "intent model period, ms")
v("d_intent", m(travel(RM.intent_inference_latency), "mm"), "{:.0f}", "travel in one intent evaluation, mm")
v("t_chunk", m(RM.chunk_inference_latency, "ms"), "{:.0f}", "chunk inference p99, ms")
v("d_chunk", m(travel(RM.chunk_inference_latency), "mm"), "{:.0f}", "travel in one chunk inference, mm")
v("t_renew", m(RM.chunk_period, "ms"), "{:.0f}", "chunk renewal period, ms")
v("f_chunk", m(RM.chunk_rate, "Hz"), "{:.0f}", "chunk policy rate, Hz")
v("chunk_h", RM.chunk_horizon, "{:.0f}", "setpoints per chunk")
v("t_step", m(RM.chunk_step, "ms"), "{:.0f}", "setpoint spacing, ms")
v("t_tick", m(AISLE.t_tick, "ms"), "{:.0f}", "permission tick, ms")
v("f_tick_khz", m(RM.permission_rate, "kHz"), "{:.0f}", "permission tick rate, kHz")
v("d_tick", m(travel(AISLE.t_tick), "mm"), "{:.1f}", "travel per tick, mm")
v("t_current_us", m(1 / RM.current_loop_rate, "us"), "{:.0f}", "current-loop period, us")
v("f_current_khz", m(RM.current_loop_rate, "kHz"), "{:.0f}", "current-loop rate, kHz")
v("tier_span", ratio(1 / RM.intent_rate, 1 / RM.current_loop_rate), "{:,.0f}", "intent period over current period")
v("tau_e", m(ELEC.result.time_constant, "ms"), "{:.1f}", "hub winding L/R, ms")
v("f_pole", m(ELEC.result.pole_frequency, "Hz"), "{:.0f}", "winding pole, Hz")
v("d_l1_stop", m(L1_STOP.result.stop_distance, "mm"), "{:.1f}", "elementary stop at the drive limit, mm")
v("d_clear_m", m(AISLE.d_clear, "m"), "{:.2f}", "clear distance, m")
v("d_clear_mm", m(AISLE.d_clear, "mm"), "{:,.0f}", "clear distance, mm")
v("d_clear_mm_nc", m(AISLE.d_clear, "mm"), "{:.0f}", "the same, no comma (inside an equation)")

print("== Phase lag at crossover (equation 1)")
v("f_c", m(LOOP.crossover, "Hz"), "{:.0f}", "joint-loop crossover, Hz")
v("phi_allow", m(LOOP.phase_allowance, "degree"), "{:.0f}", "phase erosion allowance, degrees")
v("t_bus", m(AISLE.t_bus, "ms"), "{:.0f}", "bus cycle, ms")
v("t_wcet", m(AISLE.enforcer_wcet, "us"), "{:.0f}", "declared WCET, us")
v("phi_per_ms", m(loop.phase_per_delay, "degree/ms"), "{:.1f}", "phase per millisecond of delay, degrees")
v("phi_hold", m(loop.phase_hold, "degree"), "{:.1f}", "sample-and-hold phase, degrees")
v("phi_comp", m(loop.phase_compute, "degree"), "{:.1f}", "compute phase, degrees")
v("phi_bus", m(loop.phase_transport, "degree"), "{:.1f}", "bus phase, degrees")
v("phi_total", m(loop.phase_total, "degree"), "{:.1f}", "total phase, degrees")
v("tau_total", m(loop.total_delay, "ms"), "{:.2f}", "total delay, ms")
v("phi_over", loop.over_allowance, "{:.1f}", "total over the allowance")
v("tau_allowed", m(loop.allowed_delay, "ms"), "{:.2f}", "delay the allowance tolerates, ms")
v("f_c_max", m(loop.crossover_ceiling, "Hz"), "{:.1f}", "highest crossover the delay supports, Hz")

print("== Measured: 1 ms periodic task on a laptop OS (data/jitter_*.npy)")
JIT = {k: np.load(DATA / f"jitter_{k}.npy").astype(float) for k in ("light", "heavy")}
JSTAT = {k: TAIL.solve(a, "microsecond", threshold=PERIOD) for k, a in JIT.items()}
light, heavy = JSTAT["light"], JSTAT["heavy"]
v("jit_n", heavy.count, "{:,.0f}", "releases per run")
v("jit_p99_light", m(light.p99, "us"), "{:.0f}", "p99 lateness, ordinary load, us")
v("jit_p99_heavy", m(heavy.p99, "us"), "{:.0f}", "p99 lateness, loaded, us")
v("jit_max_light_ms", m(light.maximum, "ms"), "{:.1f}", "worst lateness, ordinary load, ms")
v("jit_max_heavy_ms", m(heavy.maximum, "ms"), "{:.1f}", "worst lateness, loaded, ms")
v("jit_over_heavy", heavy.over_threshold, "{:.0f}", "loaded releases more than one period late")
v("d_jit_heavy", m(travel(heavy.maximum), "mm"), "{:.1f}", "travel during the worst lateness, mm")
RUN = {n: TAIL.solve(JIT["heavy"][:n], "microsecond") for n in (10 ** 3,)}
v("jit_max_1k_ms", m(RUN[10 ** 3].maximum, "ms"), "{:.1f}", "worst lateness in the first 1,000 releases, ms")
v("jit_n_1k", 10 ** 3, "{:,.0f}", "the same test length")

print("== Safety relay (Pilz PNOZ s4 datasheet, Times)")
v("relay_typ", m(RELAY.estop_release_typical, "ms"), "{:.0f}", "E-STOP release, typical, ms")
v("relay_max", m(RELAY.estop_release_max, "ms"), "{:.0f}", "E-STOP release, maximum, ms")
v("d_relay_typ", m(travel(RELAY.estop_release_typical), "mm"), "{:.0f}", "travel during the typical delay, mm")
v("d_relay_max", m(travel(RELAY.estop_release_max), "mm"), "{:.0f}", "travel during the maximum, mm")

print("== Measured: 1 kHz reader against a stalling 20 Hz writer (data/mailbox_*.npy)")
MBX = {}
for k in ("mutex", "trylock", "seqlock"):
    a = np.load(DATA / f"mailbox_{k}.npy").astype(float)
    t_us = a[:, 0] / 10 ** 3
    MBX[k] = (TAIL.solve(t_us, "microsecond", threshold=PERIOD), int((a[:, 1] > 1).sum()))
v("mbx_n", MBX["mutex"][0].count, "{:,.0f}", "reads per exchange")
v("mbx_mutex_max", m(MBX["mutex"][0].maximum, "us"), "{:,.0f}", "mutex reader's longest wait, us")
v("mbx_mutex_max_ms", m(MBX["mutex"][0].maximum, "ms"), "{:.1f}", "the same, ms")
v("mbx_mutex_over", MBX["mutex"][0].over_threshold, "{:.0f}", "mutex reads longer than one tick")
v("mbx_trylock_max", m(MBX["trylock"][0].maximum, "us"), "{:.1f}", "try-lock reader's longest wait, us")
v("mbx_seqlock_max", m(MBX["seqlock"][0].maximum, "us"), "{:.1f}", "seqlock reader's longest wait, us")
v("mbx_torn", MBX["seqlock"][1], "{:.0f}", "torn reads the seqlock rejected")

print("== The chunk lease (Napkin Math 4.1, equation 2)")
body = BODY.result
v("a_brake", m(AISLE.a_brake, "m/s**2"), "{:.0f}", "credible deceleration, m/s^2")
v("t_onset", m(AISLE.t_brake_onset, "ms"), "{:.0f}", "brake onset, ms")
v("t_lease", m(AISLE.t_lease, "ms"), "{:.0f}", "chunk lease, ms")
v("d_brake", m(body.braking_travel, "mm"), "{:.1f}", "braking travel v^2/2a, mm")
v("d_overhead", m(body.overhead, "mm"), "{:.0f}", "localization bound plus protective clearance, mm")
v("d_loc", m(AISLE.delta_loc, "mm"), "{:.0f}", "localization bound, mm")
v("d_margin", m(AISLE.delta_margin, "mm"), "{:.0f}", "protective clearance, mm")
v("d_onset", m(body.delay_travel, "mm"), "{:.0f}", "travel during the brake onset, mm")
v("d_body_terms", m(lease.body_terms_distance, "mm"), "{:.1f}", "terms already charged, mm")
v("t_ceiling", m(lease.delay_ceiling, "ms"), "{:.1f}", "pre-brake ceiling, ms")
v("t_lease_path", m(lease.lease_path, "ms"), "{:.0f}", "lease path to the stop command, ms")
v("d_lease_path", m(lease.lease_path_travel, "mm"), "{:.0f}", "travel along the lease path, mm")
v("t_adm_onset", m(lease.admission_to_brake_onset, "ms"), "{:.0f}", "last admission to brake onset, ms")
v("d_adm_onset", m(lease.admission_to_brake_onset_travel, "mm"), "{:.0f}", "the same as travel, mm")
v("d_running", m(lease.running_total, "mm"), "{:.1f}", "running total of the stopping budget, mm")
v("d_spare", m(lease.spare, "mm"), "{:.1f}", "clear distance left, mm")
v("t_ceiling_left", m(lease.delay_spare, "ms"), "{:.1f}", "ceiling left after brake onset, ms")
v("tau_delay", m(AISLE.tau_delay, "ms"), "{:.1f}", "full pre-brake delay, ms")
v("t_last_adm", m(lease.stall_last_admission, "ms"), "{:.0f}", "last admission on the stall timeline, ms")
v("t_first_adm", m(RM.chunk_inference_latency, "ms"), "{:.0f}", "first admission on the stall timeline, ms")
v("t_stall", m(lease.stall_last_admission - RM.chunk_inference_latency + RM.chunk_period, "ms"), "{:.0f}",
  "the policy stalls (start of the third renewal), ms")
v("t_expiry", m(lease.stall_lease_expiry, "ms"), "{:.0f}", "lease expiry on the stall timeline, ms")
v("t_brake_at", m(lease.stall_brake_onset, "ms"), "{:.0f}", "brake onset on the stall timeline, ms")
v("t_tick_bus_onset", m(lease.stall_brake_onset - lease.stall_lease_expiry, "ms"), "{:.0f}",
  "tick + bus + onset, ms")
v("t_tick_bus_onset_1", m(lease.stall_brake_onset - lease.stall_lease_expiry, "ms"), "{:.1f}",
  "the same, one decimal (lease-rule figure)")

print("== Measured OpenVLA-7B on the Jetson AGX Orin (Jetson AI Lab tutorial, 2024)")
VLA_MS = [("FP16", Q_(VLA.OpenVlaOrinFp16LatencyMs, "ms")), ("FP8", Q_(VLA.OpenVlaOrin8BitLatencyMs, "ms")),
          ("INT4", RM.intent_reference_latency)]
v("vla_fp16", m(VLA_MS[0][1], "ms"), "{:.0f}", "FP16 per frame, ms")
v("vla_fp8", m(VLA_MS[1][1], "ms"), "{:.0f}", "8-bit weights per frame, ms")
v("vla_int4", m(VLA_MS[2][1], "ms"), "{:.0f}", "INT4 per frame, ms")
v("vla_int4_logged", VLA.OpenVlaOrinInt4LoggedFrameMs, "{:.1f}", "INT4 logged frame, ms")
v("vla_over_ceiling", ratio(VLA_MS[2][1], lease.delay_ceiling), "{:.2f}", "INT4 over the pre-brake ceiling")
v("vla_over_lease", ratio(VLA_MS[2][1], AISLE.t_lease), "{:.1f}", "INT4 over the lease")

print("== Estimate: a 100 ms lease (Checkpoint 4.2)")
v("t_lease_long", m(LEASE_LONG.lease, "ms"), "{:.0f}", "lengthened lease, ms")
v("t_lease_added", m(LEASE_LONG.lease - AISLE.t_lease, "ms"), "{:.0f}", "added lease, ms")
v("d_lease_added", m(lease_long.running_total - lease.running_total, "mm"), "{:.0f}", "added distance, mm")
v("t_lease_path_long", m(lease_long.lease_path, "ms"), "{:.0f}", "lengthened lease path, ms")
v("d_running_long", m(lease_long.running_total, "mm"), "{:.1f}", "running total, mm")
v("t_adm_onset_long", m(lease_long.admission_to_brake_onset, "ms"), "{:.0f}", "last admission to brake onset, ms")

print("== The lease rule (equation 4, table 1)")
v("t_lease_ceiling", m(lease.lease_ceiling, "ms"), "{:.1f}", "ceiling less tick, bus, onset, ms")
v("t_obs_age", m(AISLE.perception_age, "ms"), "{:.1f}", "observation age at dispatch, ms")
v("t_disp_adm", m(AISLE.t_dispatch_to_admission, "ms"), "{:.1f}", "dispatch to admission, ms")
v("t_age_adm", m(AISLE.t_capture_to_admission, "ms"), "{:.1f}", "evidence age at admission, ms")
v("t_lease_upper", m(lease.lease_ceiling_at_admission, "ms"), "{:.1f}", "upper limit on the lease, ms")
v("t_jitter", m(LEASE.renewal_jitter, "ms"), "{:.0f}", "renewal jitter allowance, ms")
v("t_lease_floor", m(lease.lease_floor, "ms"), "{:.0f}", "lease floor, ms")
v("t_capture_onset", m(AISLE.t_capture_to_brake_onset, "ms"), "{:.1f}", "capture to brake onset, ms")

print("== Authority-expiry timers (4.4.1)")
v("f_heartbeat", m(AISLE.heartbeat_rate, "Hz"), "{:.0f}", "heartbeat rate, Hz")
v("t_heartbeat", m(AISLE.heartbeat_timeout, "ms"), "{:.0f}", "heartbeat timeout, ms")
v("missed_beats_words", ratio(AISLE.heartbeat_timeout, 1 / AISLE.heartbeat_rate), "words", "missed beats")
v("t_watchdog", m(AISLE.mcu_self_watchdog, "ms"), "{:.0f}", "MCU self-watchdog, ms")

print("== Response time R = C + B + I (Napkin Math 4.2, Checkpoint 4.1)")
v("c_unloaded", m(AISLE.enforcer_unloaded, "us"), "{:.0f}", "one unloaded run, us")
v("c_claim", m(rta.wcet_claim, "us"), "{:.0f}", "declared WCET above the unloaded run, us")
v("rta_b", m(rta.blocking, "us"), "{:.0f}", "blocking, us")
v("rta_i", m(rta.interference, "us"), "{:.0f}", "interference, us")
v("rta_r", m(rta.response, "us"), "{:.0f}", "response-time bound, us")
v("rta_d", m(rta.deadline, "us"), "{:.0f}", "deadline, us")
v("rta_margin", m(rta.margin, "us"), "{:.0f}", "margin, us")
v("rta_b_long", m(rta_long.blocking, "us"), "{:.0f}", "longer blocking section, us")
v("rta_r_long", m(rta_long.response, "us"), "{:.0f}", "its response-time bound, us")
v("rta_miss", m(-rta_long.margin, "us"), "{:.0f}", "deadline miss, us")
v("rta_b_limit", m(rta.blocking_limit, "us"), "{:.0f}", "largest blocking that still fits, us")

print("== Fieldbus occupancy (Napkin Math 4.3)")
v("n_axes_words", CAN_ONE.num_axes, "words", "servo axes")
v("n_arm_words", CAN_ARM.num_axes, "words", "arm axes")
v("n_base_words", CAN_BASE.num_axes, "words", "drive-wheel axes")
v("n_axes", CAN_ONE.num_axes, "{:.0f}", "servo axes")
v("pdo_bytes", CAN_ONE.payload_bytes, "{:.0f}", "bytes per record")
v("can_frames", one.frames, "{:.0f}", "CAN FD frames per cycle, one bus")
v("can_frames_words", one.frames, "words", "the same, in words")
v("can_bits_nom", one.bits_nominal, "{:.0f}", "bits at the nominal rate")
v("can_bits_dat", one.bits_data, "{:.0f}", "bits at the data rate")
v("can_mbps_nom", m(CAN_ONE.baud_arbitration, "Mbit/s"), "{:.0f}", "nominal rate, Mbps")
v("can_mbps_dat", m(CAN_ONE.baud_data, "Mbit/s"), "{:.0f}", "data rate, Mbps")
v("can_frame_us", m(one.frame_time, "us"), "{:.1f}", "one CAN FD frame, us")
v("can_one_us", m(one.occupancy, "us"), "{:,.1f}", "one bus, all axes, us")
v("can_one_pct", 100 * one.utilization, "{:.1f}", "the same, % of the cycle")
v("can_arm_pct", 100 * arm.utilization, "{:.1f}", "arm bus, % of the cycle")
v("ecat_us", m(ecat.occupancy, "us"), "{:.1f}", "EtherCAT occupancy, us")
v("ecat_pct", 100 * ecat.utilization, "{:.1f}", "the same, % of the cycle")
v("can_axes_cap", one.axes_capacity, "{:.0f}", "axes one CAN FD bus can serve")


if __name__ == "__main__":
    write(__file__)
