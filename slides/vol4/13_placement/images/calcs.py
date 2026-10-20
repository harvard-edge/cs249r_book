#!/usr/bin/env python3
"""Every number on the Lecture 13 slides and notes (Workload Placement), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* running machine and its aisle record,
the LPDDR5-6400 timing record, Literature.* for the Valsan et al., Bechtel and
Yun, RT-2, and FogROS2 results and the Bao and Jetson AGX Orin configuration
values) and the calculations: the unloaded tick's sub-costs
(Scenarios.WarehouseEnforcerUnloadedTick), its response budget
(WarehousePermissionTickResponse and the loaded-cycle variants), the shared
DRAM read behind proposer backlogs (WarehouseSharedDieLoadedRead*), the
break-even miss-stall shares (WarehouseTick*StallShare), the package rail
droop (PackageRailDroop), the thermal throttle and its hysteresis
(WarehouseSharedDieThermalThrottle), the finished stopping budgets with and
without dispatch to admission (WarehouseFinishedStop*), the chunk lease
(WarehouseChunkLease), and the illustrative stress trace
(WarehouseRejectedPlacementTrace). The chapter reads the same scenarios. This
file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex. make_figs.py imports
it for the scenario objects and plot inputs.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Embodied, Scenarios, Literature, Hardware = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature, mlsysim.Hardware
Q_ = mlsysim.core.units.Q_

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
LPDDR5 = Hardware.Tech.Memory.LPDDR5_6400
INTERF = Literature.SharedHardwareInterference
PLATFORM = Literature.PlatformConfigurations
VLA = Literature.VlaSystems
OFFLOAD = Literature.RobotOffload

TICK = Scenarios.WarehouseEnforcerUnloadedTick
RESPONSE = Scenarios.WarehousePermissionTickResponse
READ = Scenarios.WarehouseSharedDieLoadedRead
READ_1CH = Scenarios.WarehouseSharedDieLoadedReadOneChannel
MSHR = Scenarios.WarehouseTickMshrStallShare
MSHR_DEAD = Scenarios.WarehouseTickMshrStallShareAtDeadline
WRITEBACK = Scenarios.WarehouseTickWritebackStallShare
TENFOLD = Scenarios.WarehouseTickTenfoldStallShare
DROOP = Scenarios.PackageRailDroop
THROTTLE = Scenarios.WarehouseSharedDieThermalThrottle
AISLE_STOP = Scenarios.WarehouseFinishedStopAtAisleSpeed
DISPATCH = Scenarios.WarehouseFinishedStopWithDispatch
LEASE = Scenarios.WarehouseChunkLease
CHECKPOINT = Scenarios.WarehouseLoadedCycleCheckpoint
EXAMPLE = Scenarios.WarehouseLoadedTickExample
TRACE = Scenarios.WarehouseRejectedPlacementTrace


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


resp, read, read1 = RESPONSE.result, READ.result, READ_1CH.result
thr, aisle, dispatch = THROTTLE.result, AISLE_STOP.result, DISPATCH.result

print("== The permission tick and its two paths (SE 13.2)")
v("tick_khz", m(1 / AISLE.t_tick, "kHz"), "{:.0f}", "permission tick rate, kHz")
v("t_tick_ms", m(AISLE.t_tick, "ms"), "{:.0f}", "permission tick, ms")
v("mcu_mhz", m(RM.permission_mcu.compute.clock_rate, "MHz"), "{:.0f}", "safety MCU clock, MHz")
v("chunk_infer_ms", m(RM.chunk_inference_latency, "ms"), "{:.0f}", "chunk inference, p99, ms")
v("chunk_period_ms", m(RM.chunk_period, "ms"), "{:.0f}", "chunk period, ms")
v("sense_us", m(TICK.stage("sense"), "us"), "{:.0f}", "sense, us")
v("check_us", m(TICK.stage("constraint check"), "us"), "{:.0f}", "constraint check, us")
v("stage_us", m(TICK.stage("drive-frame staging"), "us"), "{:.0f}", "drive-frame staging, us")
v("unloaded_us", m(TICK.total, "us"), "{:.0f}", "unloaded tick, us")
v("unloaded_cycles", m(TICK.total * RM.permission_mcu.compute.clock_rate, ""), "{:,.0f}", "unloaded cycles at the MCU clock")
v("wcet", m(AISLE.enforcer_wcet, "us"), "{:.0f}", "declared WCET, us")
v("deadline", m(AISLE.enforcer_deadline, "us"), "{:.0f}", "deadline, us")
v("blocking", m(resp.blocking, "us"), "{:.0f}", "blocking, us")
v("interference", m(resp.interference, "us"), "{:.0f}", "interference, us")
v("b_plus_i", m(resp.blocking + resp.interference, "us"), "{:.0f}", "blocking plus interference, us")
v("exec_ceiling", m(resp.execution_ceiling, "us"), "{:.0f}", "execution ceiling, us")
v("allowance_us", m(resp.margin, "us"), "{:.0f}", "shared-hardware allowance, us")
v("deadline_share", m(AISLE.enforcer_deadline / AISLE.t_tick, ""), "{:.1f}", "deadline as share of the period")
v("rail_holdup_s", m(RM.permission_rail_holdup, "s"), "{:.0f}", "isolated rail hold-up, s")

print("== Memory: row conflict and backlog budget (SE Napkin Math 13.1)")
v("backlog_bytes", read.backlog_bytes, "{:,.0f}", "bytes in one backlog")
v("backlog_tx", read.queued_transactions, "{:.0f}", "transactions in one backlog")
v("tile_tx", read.tile_transactions, "{:.0f}", "transactions of the weight tile")
v("tile_kib", m(Q_(READ.tile_bytes, "B"), "KiB"), "{:.0f}", "weight tile, KiB")
v("snapshot_kib", m(Q_(READ.snapshot_bytes, "B"), "KiB"), "{:.0f}", "snapshot read, KiB")
v("bw_gbs", m(READ.bandwidth, "GB/s"), "{:.0f}", "application SoC peak DRAM bandwidth, GB/s")
v("bank_switches", READ.bank_switches, "{:.0f}", "row conflicts per backlog")
v("conflict_ns", m(READ.bank_switch_time, "ns"), "{:.1f}", "row conflict, t_RP + t_RCD, ns")
v("t_rcd_min_ns", m(LPDDR5.t_rcd_min, "ns"), "{:.0f}", "JESD209-5 minimum t_RCD, ns")
v("t_ck_ns", m(LPDDR5.t_ck, "ns"), "{:.2f}", "command clock, ns")
v("rcd_cycles", m(LPDDR5.t_rcd / LPDDR5.t_ck, ""), "{:.0f}", "t_RCD in command-clock cycles")
v("data_rate", m(RM.brain_soc.memory.pin_data_rate, "Mbit/s"), "{:,.0f}", "LPDDR5 data rate, MT/s")
v("backlog_us", m(read.backlog_wait, "us"), "{:.3f}", "one backlog wait, us")
v("ladder_backlog_us", m(read.backlog_wait, "us"), "{:.2f}", "the same, two decimals")
v("compute_after_us", m(READ.compute_after_sense, "us"), "{:.0f}", "check and staging after the read, us")
v("n_sense", read.backlogs_sense, "{:.0f}", "backlogs within the sense allowance")
v("read_room_wcet", m(read.read_ceiling_wcet, "us"), "{:.0f}", "read room under the WCET, us")
v("n_wcet", read.backlogs_wcet, "{:.0f}", "backlogs within the declared WCET")
v("read_room_dead", m(read.read_ceiling_deadline, "us"), "{:.0f}", "read room under the execution ceiling, us")
v("n_dead", read.backlogs_deadline, "{:.0f}", "backlogs within the execution ceiling")
v("end_dead", m(read.end_deadline, "us"), "{:.0f}", "tick end at that count, us")
v("channels", READ_1CH.channels, "{:.0f}", "LPDDR5 channels on the interface")
v("backlog1_us", m(read1.backlog_wait, "us"), "{:.1f}", "one backlog on one channel, us")
v("n1_wcet", read1.backlogs_wcet, "{:.0f}", "one channel: backlogs within the WCET")

print("== Measured shared-cache interference (Valsan et al. 2016; Bechtel and Yun 2019)")
v("valsan_a9", INTERF.ValsanCortexA9ReadSlowdown, "{:.1f}", "Cortex-A9, read co-runners, x")
v("valsan_neh", INTERF.ValsanNehalemReadSlowdown, "{:.1f}", "Nehalem, read co-runners, x")
v("valsan_a15", INTERF.ValsanCortexA15ReadSlowdown, "{:.1f}", "Cortex-A15, read co-runners, x")
v("valsan_a15_w", INTERF.ValsanCortexA15WriteSlowdown, "{:.1f}", "Cortex-A15, write co-runners, x")
v("bechtel_max", INTERF.BechtelWritebackSlowdownMax, "{:.0f}", "Cortex-A53, writeback attackers, x")

print("== Tolerable miss-stall share (SE eq 2, Checkpoint 13.1)")
v("stretch_wcet", MSHR.allowed_stretch, "{:.2f}", "stretch to the WCET")
v("stretch_wcet_less1", MSHR.allowed_stretch - 1, "{:.2f}", "the same less 1")
v("stretch_dead", MSHR_DEAD.allowed_stretch, "{:.2f}", "stretch to the execution ceiling")
v("s_mshr", MSHR.slowdown, "{:.0f}", "MSHR slowdown, x")
v("phi21", 100 * MSHR.breakeven_share, "{:.1f}", "break-even share at 21x, WCET, %")
v("phi21_dead", 100 * MSHR_DEAD.breakeven_share, "{:.1f}", "break-even share at 21x, deadline, %")
v("s_wb", WRITEBACK.slowdown, "{:.0f}", "writeback slowdown, x")
v("phi300", 100 * WRITEBACK.breakeven_share, "{:.2f}", "break-even share at 300x, %")
v("s_ten", TENFOLD.slowdown, "{:.0f}", "checkpoint slowdown, x")
v("s_ten_less1", TENFOLD.slowdown - 1, "{:.0f}", "the same less 1")
v("phi10", 100 * TENFOLD.breakeven_share, "{:.1f}", "break-even share at 10x, %")
v("phi10_us", m(TENFOLD.breakeven_stall, "us"), "{:.1f}", "the same as time of the unloaded tick, us")

print("== Thermal hangover (SE Napkin Math 13.2) and the Orin thermal zone")
v("orin_poll_ms", PLATFORM.JetsonAgxOrinTjPollingDelayMs, "{:,.0f}", "tj zone polling delay, ms")
v("t_amb", m(THROTTLE.ambient, "degC"), "{:.0f}", "bay ambient, degC")
v("theta", m(THROTTLE.theta, "K/W"), "{:.0f}", "junction to ambient, K/W")
v("tau_s", m(THROTTLE.time_constant, "s"), "{:.0f}", "thermal time constant, s")
v("p_quiet", m(THROTTLE.quiet_power, "W"), "{:.0f}", "quiet power, W")
v("p_load", m(THROTTLE.loaded_power, "W"), "{:.0f}", "loaded power, W")
v("t_quiet", m(thr.junction_quiet, "degC"), "{:.0f}", "quiet junction, degC")
v("t_loaded", m(thr.junction_loaded, "degC"), "{:.0f}", "loaded junction, degC")
v("t_trip_c", m(THROTTLE.trip, "degC"), "{:.0f}", "trip, degC")
v("t_clear_c", m(THROTTLE.clear, "degC"), "{:.0f}", "clear, degC")
v("t_trip_s", m(thr.time_to_trip, "s"), "{:.1f}", "time to trip under load, s")
v("trip_head", m(THROTTLE.trip.to("K") - thr.junction_quiet.to("K"), "K"), "{:.0f}", "trip above the quiet junction, K")
v("clear_head", m(THROTTLE.clear.to("K") - thr.junction_quiet.to("K"), "K"), "{:.0f}", "clear above the quiet junction, K")
v("recover_s", m(thr.recovery, "s"), "{:.2f}", "recovery after the load stops, s")
v("recover_ticks", thr.ticks_exposed, "{:,.0f}", "ticks inside the recovery")
v("recover_loaded_s", m(thr.recovery_from_loaded, "s"), "{:.2f}", "recovery from the loaded steady state, s")
v("f_nom_ghz", m(THROTTLE.nominal_clock, "GHz"), "{:.1f}", "nominal cluster clock, GHz")
v("f_thr_mhz", m(THROTTLE.throttled_clock, "MHz"), "{:.0f}", "throttled clock, MHz")
v("tick_cycles", thr.cycles, "{:,.0f}", "tick cycles at the nominal clock")
v("thr_us", m(thr.execution_throttled, "us"), "{:.0f}", "tick at the throttled clock, us")
v("resp_nom", m(thr.response, "us"), "{:.0f}", "nominal response, us")
v("resp_thr", m(thr.response_throttled, "us"), "{:.0f}", "throttled response, us")
v("past_us", m(thr.overrun, "us"), "{:.0f}", "past the deadline, us")
v("overrun_pct", 100 * thr.overrun_ratio, "{:.1f}", "overrun of the deadline, %")
v("theta_max", m(thr.theta_never_trips, "K/W"), "{:.2f}", "theta that never trips, K/W")

print("== Rail droop (SE 13.3.3, eq 3)")
v("di_a", m(DROOP.delta_i, "A"), "{:.0f}", "current step, A")
v("rise_ns", m(DROOP.rise_time, "ns"), "{:.0f}", "rise time, ns")
v("l_ph", m(DROOP.inductance, "pH"), "{:.0f}", "package inductance, pH")
v("r_mohm", m(DROOP.resistance, "milliohm"), "{:.0f}", "package resistance, milliohm")
v("v_nom", m(DROOP.rail_voltage, "V"), "{:.2f}", "core rail, V")
v("droop_l_mv", m(DROOP.inductive_droop, "mV"), "{:.0f}", "inductive droop, mV")
v("droop_r_mv", m(DROOP.resistive_droop, "mV"), "{:.0f}", "resistive droop, mV")
v("droop_mv", m(DROOP.droop, "mV"), "{:.0f}", "total droop, mV")
v("delay_pct", DROOP.delay_expansion_percent, "{:.0f}", "gate delay increase, %")

print("== Remedies (SE 13.6, Bechtel and Yun 2019 Sec. VI, Bao demo)")
v("lease_ms", m(AISLE.t_lease, "ms"), "{:.0f}", "chunk lease, ms")
v("min_rate_hz", m(1 / AISLE.t_lease, "Hz"), "{:.1f}", "admissions needed above, Hz")
v("bechtel_unreg", INTERF.BechtelUnregulatedSlowdown, "{:.0f}", "unregulated, x")
v("bechtel_w100", INTERF.BechtelRegulated100MBpsSlowdown, "{:.2f}", "writes capped at 100 MB/s, x")
v("bechtel_w100_1", INTERF.BechtelRegulated100MBpsSlowdown, "{:.1f}", "the same, one decimal")
v("bechtel_w50", INTERF.BechtelRegulated50MBpsSlowdown, "{:.1f}", "writes capped at 50 MB/s, x")
v("bao_linux_cpus", PLATFORM.BaoZcu104LinuxVmCpus, "{:.0f}", "Linux VM cores")
v("bao_linux_gib", m(Q_(float(PLATFORM.BaoZcu104LinuxVmMemoryBytes), "B"), "GiB"), "{:.0f}", "Linux VM memory, GiB")
v("bao_rtos_cpus", PLATFORM.BaoZcu104FreeRtosVmCpus, "{:.0f}", "FreeRTOS VM cores")
v("bao_rtos_mib", m(Q_(float(PLATFORM.BaoZcu104FreeRtosVmMemoryBytes), "B"), "MiB"), "{:.0f}", "FreeRTOS VM memory, MiB")
v("bao_shmem_kib", m(Q_(float(PLATFORM.BaoZcu104SharedRegionBytes), "B"), "KiB"), "{:.0f}", "shared IPC region, KiB")

print("== Board link and dispatch to admission (SE Napkin Math 13.3, 13.6.3)")
d2a = AISLE.t_dispatch_to_admission
v("chunk_bytes", m(AISLE.chunk_wire_bytes, "B"), "{:,.0f}", "serialized chunk, B")
v("spi_mhz", m(AISLE.board_link_clock, "MHz"), "{:.0f}", "board link clock, MHz")
v("link_ms", m(AISLE.t_board_link, "ms"), "{:.2f}", "serialization, ms")
v("deser_ms", m(AISLE.t_chunk_deserialize, "ms"), "{:.0f}", "deserialization, ms")
v("valid_ms", m(AISLE.t_plan_validate, "ms"), "{:.0f}", "validation, ms")
v("activate_ms", m(AISLE.t_plan_activate, "ms"), "{:.0f}", "activation, ms")
v("d2a_ms", m(d2a, "ms"), "{:.1f}", "dispatch-to-admission allocation, ms")
v("d2a_mm", m(aisle.travel_over(d2a), "mm"), "{:.1f}", "travel during the allocation, mm")
v("v_aisle", m(AISLE.v_aisle, "m/s"), "{:.1f}", "aisle speed, m/s")
v("spare_mm", m(aisle.spare, "mm"), "{:.1f}", "spare at the aisle speed, mm")
v("ceil_ms", m(aisle.delay_spare, "ms"), "{:.1f}", "spare as time, ms")
v("reserve_ms", m(dispatch.delay_spare, "ms"), "{:.1f}", "reserve after dispatch to admission, ms")
v("reserve_mm", m(dispatch.spare, "mm"), "{:.1f}", "reserve, mm")
v("link_share", 100 * m(AISLE.t_board_link / d2a, ""), "{:.0f}", "link share of the allocation, %")
v("plus_infer_ms", m(d2a + RM.chunk_inference_latency, "ms"), "{:.1f}", "with one more inference, ms")
v("small_ms", m(AISLE.t_chunk_deserialize + AISLE.t_plan_activate + AISLE.t_tick, "ms"), "{:.0f}",
  "deserialize + activate + tick wait, ms")
renew_window = AISLE.t_lease - LEASE.renewal_jitter
rt2_5b_step = 1 / Q_(float(VLA.Rt2SmallRateHz), "Hz")
v("jitter_ms", m(LEASE.renewal_jitter, "ms"), "{:.0f}", "renewal jitter allowance, ms")
v("stretch_pct", 100 * m(LEASE.renewal_jitter / RM.chunk_inference_latency, ""), "{:.1f}",
  "inference stretch the jitter allows, %")
v("renew_window_ms", m(renew_window, "ms"), "{:.0f}", "renewal window, ms")
v("rt2_5b_hz", VLA.Rt2SmallRateHz, "{:.0f}", "RT-2 5B rate, Hz")
v("rt2_5b_ms", m(rt2_5b_step, "ms"), "{:.0f}", "RT-2 5B result interval, ms")
v("rt2_ratio", m(rt2_5b_step / renew_window, ""), "{:.1f}", "as a multiple of the renewal window")
v("rt2_55b_low", VLA.Rt2LargeRateLowHz, "{:.0f}", "RT-2 55B rate, low, Hz")
v("rt2_55b_high", VLA.Rt2LargeRateHighHz, "{:.0f}", "RT-2 55B rate, high, Hz")
v("fogros_local_s", OFFLOAD.FogRos2GraspPlanningLocalS, "{:.0f}", "FogROS2 grasp planning on the robot, s")
v("fogros_off_s", OFFLOAD.FogRos2GraspPlanningOffloadedS, "{:.1f}", "offloaded, s")

print("== Watchdog (SE 13.5.2)")
v("wd_ms", m(AISLE.mcu_self_watchdog, "ms"), "{:.0f}", "late trip, ms")
v("wd_mm", m(AISLE.v_aisle * AISLE.mcu_self_watchdog, "mm"), "{:.1f}", "travel before the trip, mm")

print("== Falsification threshold, checkpoint, trace, record (SE 13.7, 13.8)")
v("ex_tick", m(EXAMPLE.wcet, "us"), "{:.0f}", "loaded tick example, us")
v("ex_before_deadline", m(AISLE.enforcer_deadline - EXAMPLE.wcet, "us"), "{:.0f}", "before the deadline, us")
v("tick_mm", m(AISLE.v_aisle * AISLE.t_tick, "mm"), "{:.1f}", "one tick at the aisle speed, mm")
cp = CHECKPOINT.result
v("cp_us", m(CHECKPOINT.wcet, "us"), "{:.0f}", "checkpoint loaded cycle, us")
v("cp_over", m(CHECKPOINT.wcet - AISLE.enforcer_wcet, "us"), "{:.0f}", "past the WCET, us")
v("cp_resp", m(cp.response, "us"), "{:.0f}", "response with blocking and interference, us")
v("cp_left", m(cp.margin, "us"), "{:.0f}", "deadline left, us")
h, mins, secs = TRACE.hours_minutes_seconds
v("trace_exp", math.log10(TRACE.ticks), "{:.0f}", "trace length, power of ten")
v("trace_millions", TRACE.ticks / 10 ** 6, "{:.0f}", "trace length, millions of ticks")
v("trace_h", h, "{:.0f}", "trace hours")
v("trace_min", mins, "{:.0f}", "trace minutes")
v("trace_sec", secs, "{:.0f}", "trace seconds")
for label in ("p50", "p99", "max"):
    v(f"cold_{label}", m(TRACE.cold[label], "us"), "{:.0f}", f"cold {label}, us")
for label, key in (("p50", "p50"), ("p99", "p99"), ("p99.9", "p999"), ("max", "max")):
    v(f"hot_{key}", m(TRACE.warm[label], "us"), "{:.0f}", f"warm {label}, us")
v("tick_ns", m(AISLE.t_tick, "ns"), "{:,.0f}", "nominal_period_ns")
v("deadline_ns", m(AISLE.enforcer_deadline, "ns"), "{:,.0f}", "hard_deadline_ns")
v("renewal_onset_ms", m(LEASE.result.admission_to_brake_onset, "ms"), "{:.0f}", "renewal to brake onset, ms")

# Plot inputs for make_figs.py, magnitudes at full precision.
SENSE, CHECK, STAGE = (m(TICK.stage(n), "us") for n in ("sense", "constraint check", "drive-frame staging"))
WCET, DEADLINE, PERIOD = m(AISLE.enforcer_wcet, "us"), m(AISLE.enforcer_deadline, "us"), m(AISLE.t_tick, "us")
CEILING, B_PLUS_I = m(resp.execution_ceiling, "us"), m(resp.blocking + resp.interference, "us")
COMPUTE_AFTER = m(READ.compute_after_sense, "us")
BACKLOG_ROWS = [  # (label count, queued wait, snapshot read) per row, us
    (0, 0.0, SENSE),
    (read.backlogs_sense, read.backlogs_sense * m(read.backlog_wait, "us"), m(read.snapshot_read, "us")),
    (read.backlogs_wcet, read.backlogs_wcet * m(read.backlog_wait, "us"), m(read.snapshot_read, "us")),
    (read.backlogs_deadline, read.backlogs_deadline * m(read.backlog_wait, "us"), m(read.snapshot_read, "us")),
    (read1.backlogs_wcet, read1.backlogs_wcet * m(read1.backlog_wait, "us"), m(read1.snapshot_read, "us")),
]
MEASURED = [
    ("Cortex-A9, read co-runners", float(INTERF.ValsanCortexA9ReadSlowdown), "valsan_a9"),
    ("Nehalem, read co-runners", float(INTERF.ValsanNehalemReadSlowdown), "valsan_neh"),
    ("Cortex-A15, read co-runners", float(INTERF.ValsanCortexA15ReadSlowdown), "valsan_a15"),
    ("Cortex-A15, write co-runners", float(INTERF.ValsanCortexA15WriteSlowdown), "valsan_a15_w"),
    ("Cortex-A53, write attackers", float(INTERF.BechtelWritebackSlowdownMax), "bechtel_max"),
]
REGULATION = [
    ("no regulation", float(INTERF.BechtelUnregulatedSlowdown), "bechtel_unreg"),
    ("writes capped 100 MB/s", float(INTERF.BechtelRegulated100MBpsSlowdown), "bechtel_w100"),
    ("writes capped 50 MB/s", float(INTERF.BechtelRegulated50MBpsSlowdown), "bechtel_w50"),
]
UNLOADED = m(TICK.total, "us")
S_MSHR, S_WB = MSHR.slowdown, WRITEBACK.slowdown
PHI_MSHR, PHI_WB = 100 * MSHR.breakeven_share, 100 * WRITEBACK.breakeven_share
TAU = m(THROTTLE.time_constant, "s")
T_QUIET, T_LOADED = m(thr.junction_quiet, "degC"), m(thr.junction_loaded, "degC")
T_TRIP, T_CLEAR = m(THROTTLE.trip, "degC"), m(THROTTLE.clear, "degC")
T_TO_TRIP, RECOVERY = m(thr.time_to_trip, "s"), m(thr.recovery, "s")
EXEC_NOM, EXEC_THR = m(thr.execution, "us"), m(thr.execution_throttled, "us")
LADDER = [  # (label, seconds)
    ("thermal recovery", m(thr.recovery, "s")),
    ("permission tick", m(AISLE.t_tick, "s")),
    ("DRAM backlog", m(read.backlog_wait, "s")),
    ("bank conflict", m(READ.bank_switch_time, "s")),
    ("droop rise", m(DROOP.rise_time, "s")),
]
INFER, LINK = m(RM.chunk_inference_latency, "ms"), m(AISLE.t_board_link, "ms")
VALID = m(AISLE.t_plan_validate, "ms")
SMALL = m(AISLE.t_chunk_deserialize + AISLE.t_plan_activate + AISLE.t_tick, "ms")
CEIL_MS = m(aisle.delay_spare, "ms")
COLD = {k: m(q, "us") for k, q in TRACE.cold.items()}
HOT = {k: m(q, "us") for k, q in TRACE.warm.items()}

if __name__ == "__main__":
    write(__file__)
