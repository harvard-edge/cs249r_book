#!/usr/bin/env python3
"""Every derived number in Lecture 13 (Workload Placement), Volume IV.

One block per slide, in deck order. Inputs carry their source as a comment:
"ch" is the rendered chapter, Workload Placement (dev text, 2026-10-18), with
its section, Napkin Math, or table; outside sources name their paper and the
table, figure, or section the value was read from. Each printed line is a value
the slide or its note shows, under the label used there. make_figs.py imports
VALUES from this file, so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""
import math

VALUES = {}


def show(key, label, value):
    VALUES[key] = value
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
# Permission tick on the safety MCU (ch 13.2, illustrative sub-costs)
SENSE_US = 40.0            # sample encoders, IMU over SPI, check evidence epoch
CHECK_US = 80.0            # kinematic and dynamic envelope constraints
STAGE_US = 15.0            # stage setpoints into the outbound drive frame
MCU_HZ = 400e6             # safety MCU clock (ch 13.2)
WCET_US = 250.0            # declared WCET (chosen; ch 13.2, from the enforcement record)
DEADLINE_US = 400.0        # chosen enforcer deadline (ch 13.2)
PERIOD_US = 1000.0         # 1 kHz permission tick
BLOCK_US = 50.0            # blocking by lower-priority MCU tasks (ch 13.2, eq 4.3)
INTERF_US = 40.0           # interference by higher-priority MCU tasks (ch 13.2)
# Proposal path (ch 13.2, illustrative)
INFER_MS = 40.0            # chunk policy p99 inference incl. vision
CHUNK_MS = 50.0            # chunk period (20 Hz)

# Memory (ch Napkin Math 13.1, eq 1)
BW_PEAK = 204e9            # B/s, datasheet
BACKLOG_BYTES = 360_448    # one backlog: tile (64 tx) + camera + telemetry, 88 tx
ROW_CONFLICTS = 3
T_RP_NS = 18.0             # LPDDR5 minimum precharge (ch 13.3.1)
T_RCD_NS = 18.0            # LPDDR5 minimum activation
CMD_CLK_NS = 1.25          # command clock at 6,400 MT/s
SNAPSHOT_BYTES = 4096
LATER_WORK_US = 95.0       # constraint evaluation and drive-frame write
CHANNELS = 16              # LPDDR5 interface channels (ch Systems insight 13.1)

# Published interference (outside sources, read from the papers)
VALSAN_A9 = 2.1      # Valsan et al. RTAS 2016, Exp. 2 text (BwRead(LLC) vs BwRead(DRAM))
VALSAN_A15 = 10.6    # same, Fig. 2(b) label
VALSAN_NEH = 7.9     # same, text
VALSAN_A15_W = 21.4  # Fig. 2(e) label, write co-runners (the 21x of the abstract)
BECHTEL_RPI3 = 346.0          # Bechtel and Yun RTAS 2019, Sec. IV-C text ("up to 346X")
BECHTEL_FIG13_UNREG = 345.0   # Sec. VI text: "reduced from 345X"
BECHTEL_W100 = 1.34           # write threshold 100 MB/s
BECHTEL_W50 = 1.2             # write threshold 50 MB/s

# Rail droop (ch 13.3.3, illustrative package parameters)
L_PKG = 30e-12
DI_A = 6.0
DT_S = 2e-9
R_PKG = 8e-3
V_NOM = 0.850
V_TH = 0.420
ALPHA = 1.3

# Thermal hangover (ch Napkin Math 13.2, illustrative)
T_AMB = 45.0
R_JA = 2.0
TAU_S = 8.0
P_QUIET = 4.0
P_LOAD = 26.0
T_TRIP = 95.0
T_CLEAR = 85.0
F_NOM = 1.2e9
F_THR = 600e6
# Jetson AGX Orin upstream thermal zone (Linux v6.12 tegra234-p3701-0000.dtsi)
ORIN_POLL_MS = 1000

# Watchdog (ch 13.5.2)
WD_TIMEOUT_MS = 5.0
V_AISLE = 1.3              # m/s, chosen aisle speed (ch 13.6.2, from Lecture 12's budget)

# Lease (ch 13.6.1, from the Nervous System chapter)
LEASE_MS = 60.0
RENEW_WINDOW_MS = 55.0     # lease less renewal jitter
JITTER_MS = 5.0
STRETCH = 0.125            # illustrative 12.5 percent inference stretch

# Offload (ch 13.6.3, Brohan et al. 2023 as quoted)
RT2_5B_HZ = 5.0
RT2_55B_HZ_MAX = 3.0
RT2_55B_HZ_MIN = 1.0
FOGROS_LOCAL_S = 14.0      # Ichnowski et al. 2023, as quoted
FOGROS_CLOUD_S = 1.2

# Board link (ch Napkin Math 13.3)
CHUNK_BYTES = 4096
SPI_HZ = 25e6
DESER_MS = 2.0
VALID_MS = 18.0
ACT_MS = 1.0
TICK_WAIT_MS = 1.0
SPARE_MM = 131.0           # derived in Lecture 12 (ch Napkin Math 12.1), quoted in ch

# Falsification (ch 13.7)
LOADED_EXAMPLE_US = 260.0
CHECKPOINT_US = 300.0      # ch Checkpoint 13.3
TRACE_TICKS = 1e7
COLD = {"p50": 95, "p99": 128, "max": 152}          # illustrative trace (ch 13.7.3)
HOT = {"p50": 210, "p99": 330, "p99.9": 395, "max": 468}
ONSET_MS = 82.0            # renewal-to-onset budget (ch 13.7.3)


def main():
    head("Two paths: budget bar (slide 'The permission tick has 60 us to spare')")
    unloaded = SENSE_US + CHECK_US + STAGE_US
    show("unloaded_us", "unloaded tick (us)", f"{unloaded:.0f}")
    show("cycles", "cycles at 400 MHz", f"{unloaded * 1e-6 * MCU_HZ:,.0f}")
    show("slack_deadline", "slack to deadline (us)", f"{DEADLINE_US - unloaded:.0f}")
    show("headroom_wcet", "WCET headroom over deadline (us)", f"{DEADLINE_US - WCET_US:.0f}")
    overhead = BLOCK_US + INTERF_US
    show("overhead_us", "blocking + interference (us)", f"{overhead:.0f}")
    ceiling = DEADLINE_US - overhead
    show("ceiling_us", "execution ceiling (us)", f"{ceiling:.0f}")
    show("allowance_us", "shared-hardware allowance (us)", f"{ceiling - WCET_US:.0f}")
    show("proposer_margin_ms", "proposer margin in 50 ms period (ms)", f"{CHUNK_MS - INFER_MS:.0f}")
    show("tick_share", "deadline as share of period", f"{DEADLINE_US / PERIOD_US:.1f}")

    head("Memory: row conflict and backlog budget (Napkin Math 13.1)")
    step_cycles = math.ceil(T_RP_NS / CMD_CLK_NS)
    show("step_cycles", "cycles per precharge or activation", step_cycles)
    conflict_ns = step_cycles * CMD_CLK_NS + math.ceil(T_RCD_NS / CMD_CLK_NS) * CMD_CLK_NS
    show("conflict_ns", "row conflict (ns)", f"{conflict_ns:.1f}")
    wait_us = BACKLOG_BYTES / BW_PEAK * 1e6 + ROW_CONFLICTS * conflict_ns * 1e-3
    show("backlog_us", "one backlog wait (us)", f"{wait_us:.3f}")
    snap_us = SNAPSHOT_BYTES / BW_PEAK * 1e6
    show("snap_us", "snapshot read (us)", f"{snap_us:.3f}")
    n_sense = math.floor((SENSE_US - snap_us) / wait_us)
    n_wcet = math.floor((WCET_US - LATER_WORK_US - snap_us) / wait_us)
    n_dead = math.floor((ceiling - LATER_WORK_US - snap_us) / wait_us)
    show("n_sense", "backlogs within sense allowance", n_sense)
    show("read_room_wcet", "read room under WCET (us)", f"{WCET_US - LATER_WORK_US:.0f}")
    show("n_wcet", "backlogs within declared WCET", n_wcet)
    show("read_room_dead", "read room under deadline less 90 (us)", f"{ceiling - LATER_WORK_US:.0f}")
    show("n_dead", "backlogs within deadline", n_dead)
    bw_ch = BW_PEAK / CHANNELS
    wait1_us = BACKLOG_BYTES / bw_ch * 1e6 + ROW_CONFLICTS * conflict_ns * 1e-3
    snap1_us = SNAPSHOT_BYTES / bw_ch * 1e6
    show("backlog1_us", "one backlog, one channel (us)", f"{wait1_us:.1f}")
    show("n1_sense", "one channel: backlogs within sense", math.floor((SENSE_US - snap1_us) / wait1_us))
    show("n1_wcet", "one channel: backlogs within WCET", math.floor((WCET_US - LATER_WORK_US - snap1_us) / wait1_us))
    for k, n in (("end_sense", n_sense), ("end_wcet", n_wcet), ("end_dead", n_dead)):
        show(k, f"tick end with {n} backlogs (us)", f"{snap_us + n * wait_us + LATER_WORK_US:.0f}")
    n1 = VALUES["n1_wcet"]
    show("end_one", f"tick end with {n1} backlogs, one channel (us)",
         f"{snap1_us + n1 * wait1_us + LATER_WORK_US:.0f}")
    show("bypass_us", "read with priority bypass (us)", f"{2 * snap_us:.2f}")

    head("Memory: stretch factors and tolerable miss-stall share (eq 2)")
    show("stretch_wcet", "stretch to WCET", f"{WCET_US / unloaded:.2f}")
    show("stretch_dead", "stretch to deadline", f"{ceiling / unloaded:.2f}")

    def phi(slow, bound):
        return (bound / unloaded - 1) / (slow - 1) * 100
    show("phi21", "phi at 21x, WCET (%)", f"{phi(21, WCET_US):.1f}")
    show("phi21_dead", "phi at 21x, deadline (%)", f"{phi(21, ceiling):.1f}")
    show("phi300", "phi at 300x, WCET (%)", f"{phi(300, WCET_US):.2f}")
    show("phi346", "phi at 346x, WCET (%)", f"{phi(BECHTEL_RPI3, WCET_US):.2f}")
    show("phi10", "phi at 10x, WCET (%) [Estimate]", f"{phi(10, WCET_US):.1f}")
    show("phi10_us", "  as microseconds of the 135 us tick", f"{phi(10, WCET_US) / 100 * unloaded:.1f}")

    head("Rail droop (eq 3, alpha-power)")
    ind_mv = L_PKG * DI_A / DT_S * 1e3
    res_mv = DI_A * R_PKG * 1e3
    show("droop_l_mv", "inductive droop (mV)", f"{ind_mv:.0f}")
    show("droop_r_mv", "resistive droop (mV)", f"{res_mv:.0f}")
    show("droop_mv", "total droop (mV)", f"{ind_mv + res_mv:.0f}")
    vd = V_NOM - (ind_mv + res_mv) / 1e3
    show("v_drooped", "drooped rail (V)", f"{vd:.3f}")
    ratio = (vd / V_NOM) * ((V_NOM - V_TH) / (vd - V_TH)) ** ALPHA
    show("delay_pct", "gate delay increase (%)", f"{(ratio - 1) * 100:.0f}")

    head("Thermal hangover (Napkin Math 13.2)")
    tq = T_AMB + P_QUIET * R_JA
    tl = T_AMB + P_LOAD * R_JA
    show("t_quiet", "quiescent junction (C)", f"{tq:.0f}")
    show("t_loaded", "loaded junction (C)", f"{tl:.0f}")
    t_trip = TAU_S * math.log((tl - tq) / (tl - T_TRIP))
    show("t_trip_s", "time to trip (s)", f"{t_trip:.1f}")
    cycles = WCET_US * 1e-6 * F_NOM
    show("tick_cycles", "tick cycles at 1.2 GHz", f"{cycles:,.0f}")
    thr_us = cycles / F_THR * 1e6
    show("thr_us", "tick at 600 MHz (us)", f"{thr_us:.0f}")
    show("resp_nom", "nominal response (us)", f"{WCET_US + overhead:.0f}")
    show("resp_thr", "throttled response (us)", f"{thr_us + overhead:.0f}")
    past = thr_us + overhead - DEADLINE_US
    show("past_us", "past deadline (us)", f"{past:.0f}")
    show("overrun_pct", "overrun of deadline (%)", f"{past / DEADLINE_US * 100:.1f}")
    show("nominal_slack", "nominal slack to deadline (us)", f"{DEADLINE_US - WCET_US - overhead:.0f}")
    rec = TAU_S * math.log((T_TRIP - tq) / (T_CLEAR - tq))
    show("recover_s", "recovery time (s)", f"{rec:.2f}")
    show("recover_ticks", "recovery in 1 ms ticks", f"{rec * 1000:,.0f}")
    show("trip_margin", "trip needs R_ja below (K/W)", f"{(T_TRIP - T_AMB) / P_LOAD:.2f}")
    show("orin_poll_ticks", "Orin thermal poll period in 1 ms ticks", f"{ORIN_POLL_MS / 1:,.0f}")
    show("cap_jk", "lumped thermal capacitance (J/K)", f"{TAU_S / R_JA:.0f}")

    head("Coupling-timescale ladder (ch 13.4)")
    show("ladder_backlog_us", "DRAM backlog (us)", f"{wait_us:.2f}")
    show("ladder_span", "droop rise to thermal recovery (orders of magnitude)",
         f"{math.log10(rec / DT_S):.1f}")

    head("Watchdog (ch 13.5.2)")
    show("wd_mm", "5 ms timeout as base travel at 1.3 m/s (mm)", f"{WD_TIMEOUT_MS * V_AISLE:.1f}")
    show("wd_ticks", "5 ms timeout in ticks", f"{WD_TIMEOUT_MS / (PERIOD_US / 1000):.0f}")

    head("Bao config excerpt (bao-demos linux+freertos zcu104.c)")
    show("bao_linux_gib", "Linux VM memory 0x40000000 (GiB)", f"{0x40000000 / 2**30:.0f}")
    show("bao_rtos_mib", "FreeRTOS VM memory 0x8000000 (MiB)", f"{0x8000000 / 2**20:.0f}")
    show("bao_shmem_kib", "shared IPC region 0x00010000 (KiB)", f"{0x00010000 / 2**10:.0f}")

    head("Thermal: hangover if the proposer keeps running to the loaded steady state")
    show("recover_from_loaded_s", "recovery from 97 C (s)",
         f"{TAU_S * math.log((T_AMB + P_LOAD * R_JA - (T_AMB + P_QUIET * R_JA)) / (T_CLEAR - (T_AMB + P_QUIET * R_JA))):.2f}")

    head("Remedies: workload reduction and regulation")
    show("min_rate_hz", "admissions needed above (Hz)", f"{1000 / LEASE_MS:.1f}")
    show("stretch_ms", "12.5 percent inference stretch (ms)", f"{INFER_MS * STRETCH:.1f}")
    show("bechtel_gain_100", "regulation gain at 100 MB/s (x)", f"{BECHTEL_FIG13_UNREG / BECHTEL_W100:.0f}")

    head("Offload screen (ch 13.6.3)")
    p5 = 1000 / RT2_5B_HZ
    show("rt2_5b_ms", "RT-2 5B plan interval (ms)", f"{p5:.0f}")
    show("rt2_5b_ratio", "  as multiple of the 55 ms window", f"{p5 / RENEW_WINDOW_MS:.1f}")
    show("rt2_55b_ms", "RT-2 55B best interval (ms)", f"{1000 / RT2_55B_HZ_MAX:.1f}")
    show("rt2_55b_worst_ms", "RT-2 55B worst interval (ms)", f"{1000 / RT2_55B_HZ_MIN:.0f}")
    show("fogros_x", "FogROS2 grasp-planning speedup (x)", f"{FOGROS_LOCAL_S / FOGROS_CLOUD_S:.1f}")

    head("Board-link toll (Napkin Math 13.3)")
    link_ms = 8 * CHUNK_BYTES / SPI_HZ * 1e3
    show("link_ms", "serialization (ms)", f"{link_ms:.2f}")
    total = INFER_MS + link_ms + DESER_MS + VALID_MS + ACT_MS + TICK_WAIT_MS
    show("d2a_ms", "dispatch-to-admission allocation (ms)", f"{total:.1f}")
    show("d2a_mm", "travel during allocation (mm)", f"{total * V_AISLE:.1f}")
    ceil_ms = SPARE_MM / V_AISLE
    show("ceil_ms", "ceiling from 131.0 mm spare (ms)", f"{ceil_ms:.1f}")
    show("reserve_ms", "reserve (ms)", f"{ceil_ms - total:.1f}")
    show("reserve_mm", "reserve (mm)", f"{(ceil_ms - total) * V_AISLE:.1f}")
    show("link_share", "link share of allocation (%)", f"{link_ms / total * 100:.1f}")
    show("small_ms", "deserialize + activate + tick wait (ms)", f"{DESER_MS + ACT_MS + TICK_WAIT_MS:.0f}")
    show("plus_infer_ms", "with one more inference (ms)", f"{total + INFER_MS:.1f}")

    head("Falsification threshold (ch 13.7)")
    show("ex_over", "260 us example exceeds WCET by (us)", f"{LOADED_EXAMPLE_US - WCET_US:.0f}")
    show("tick_mm", "one tick at 1.3 m/s (mm)", f"{V_AISLE * PERIOD_US / 1000:.1f}")
    show("cp_over", "300 us cycle exceeds WCET by (us)", f"{CHECKPOINT_US - WCET_US:.0f}")
    show("cp_resp", "300 us cycle response with 90 us (us)", f"{CHECKPOINT_US + overhead:.0f}")
    show("cp_left", "  deadline left (us)", f"{DEADLINE_US - CHECKPOINT_US - overhead:.0f}")
    secs = TRACE_TICKS * PERIOD_US * 1e-6
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    show("trace_s", "1e7 ticks (s)", f"{secs:,.0f}")
    show("trace_hms", "  as h min s", f"{h} h {m} min {s} s")
    show("hot_p99_resp", "hot p99 + 90 us (us)", f"{HOT['p99'] + overhead:.0f}")
    show("hot_max_over", "hot max past WCET (us)", f"{HOT['max'] - WCET_US:.0f}")


if __name__ == "__main__":
    main()
