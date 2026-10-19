#!/usr/bin/env python3
"""Every derived number in Lecture 14 (Supervisory Intervention), one block per slide.

Inputs carry their source. "SI" is the rendered chapter, Supervisory
Intervention (Volume IV, Chapter 14); "SE" is Safety Enforcement (Chapter 12),
whose finished stopping budget this lecture reuses. Values that the MLSysIM
registry holds (AISLE scenario) are typed here with that registry name.
Every printed value is what the slide or its note shows, with the label used
there. make_figs.py imports from this file.

Run from this folder: python3 calcs.py
"""

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))


def head(title):
    print(f"\n--- {title}")


def show(label, value):
    print(f"  {label}: {value}")


# =============================================================================
# Shared inputs: the running machine at the rack end (SI Napkin Math 14.1;
# SE Napkin Math 12.1; MLSysIM Embodied.Scenario AISLE)
# =============================================================================
V_AISLE = 1.3          # m/s, chosen aisle speed (AISLE.v_aisle)
D_CLEAR = 1.10         # m, rack-end clear distance (illustrative, AISLE.d_clear)
LOC = 0.050            # m, localization bound (illustrative)
PROT = 0.100           # m, protective clearance margin (chosen)
TRACK = 0.040          # m, tracking bound (illustrative)
ALLOW = LOC + PROT + TRACK          # 190 mm fixed allowances (SI Napkin Math 14.1)
A_BRAKE = 2.0          # m/s^2, credible braking deceleration (illustrative)
# Jerk bound: SE 12.10 gives the S-curve stop from the 1.5 m/s drive limit as
# 1,031.25 ms against 750 ms at constant deceleration, so a/J = 0.28125 s.
J = A_BRAKE / (1.03125 - 1.5 / A_BRAKE)   # prints as 7.11 m/s^3 (SI: 7.11, derived)
T_ACT = 0.020          # s, brake onset (SE table 1)
T_LEASE_PATH = 0.062   # s, 60 ms chunk lease + one tick + one bus cycle (SE table 1)
T_AGE = 0.0516         # s, observation age (SE table 1)
TAU = T_ACT + T_LEASE_PATH + T_AGE        # 133.6 ms pre-brake delay (SI: derived)
T_DA = 0.0633          # s, dispatch-to-admission allocation (Lecture 13; SI 14.2.1)
T_CHUNK = 0.060        # s, chunk lease (Lecture 4)
T_MANUAL = 0.100       # s, manual lease, chosen (SI 14.3.4)
TICK = 0.001           # s, permission tick, 1 kHz (SI 14.1)

# Human and console terms (SI 14.2.2; AISLE registry, sourced)
T_PERC, T_COG, T_MOT = 0.200, 0.170, 0.100   # Model Human Processor slow bounds (Card et al. 1983)
T_PERC_N, T_COG_N, T_MOT_N = 0.100, 0.070, 0.070  # nominal cycles (same source)
T_VIDEO = 0.104        # s, glass-to-glass, wired teleoperation link (Schimpe et al. 2022)
HS = {"Request": 0.010, "Acknowledge": 0.005, "Commit": 0.001, "Confirm": 0.015}  # chosen (SI table 5)
T_BRAKE_EXP, T_BRAKE_UNEXP, T_BRAKE_SURP = 0.75, 1.25, 1.5   # Green 2000
TOT = {"p50": 2.37, "mean": 2.72, "p90": 4.33, "p99": 7.0}   # Zhang et al. 2019 condition means


def stop_dist(v, a=A_BRAKE, j=J):
    """Jerk-limited resident stop from zero acceleration (SI eq. 11.17 form), m."""
    if v >= a * a / j:
        return v * v / (2 * a) + v * a / (2 * j)
    return v * math.sqrt(v / j)       # triangular deceleration: t = 2 sqrt(v/j), d = v t / 2


def stop_total(v, wait=0.0):
    """SI equation 2: v (wait + tau) + resident stop + allowances, m."""
    return v * (wait + TAU) + stop_dist(v) + ALLOW


def ceiling(wait):
    """Largest v with stop_total(v, wait) <= D_CLEAR (bisection), m/s."""
    lo, hi = 0.0, 3.0
    for _ in range(200):
        m = (lo + hi) / 2
        if stop_total(m, wait) <= D_CLEAR:
            lo = m
        else:
            hi = m
    return lo


def deadline(v):
    """SI equation 4: latest fallback deadline at speed v, s."""
    return (D_CLEAR - ALLOW - stop_dist(v)) / v - TAU


# =============================================================================
# Hook: California DMV filings, Waymo (SI 14.1 figure 1; MLSysIM registry)
# =============================================================================
def dmv_rows():
    from mlsysim import Literature
    return [r for r in Literature.CaliforniaDisengagements.entries
            if r["company"] == "Waymo" and r["filer_type"] in ("annual", "first_report")]


def dmv_get(rows, year, permit):
    return next(r for r in rows if r["report_year"] == year and r["permit_type"] == permit)


DMV = dmv_rows()
DRV, DL = "drivered_AVT", "driverless_AVDT"
DMV_FIRST = dmv_get(DMV, 2015, DRV)
DMV_LAST = dmv_get(DMV, 2025, DRV)
DMV_DL = dmv_get(DMV, 2025, DL)
DMV_DROP = dmv_get(DMV, 2020, DRV)["miles_per_disengagement"] / dmv_get(DMV, 2021, DRV)["miles_per_disengagement"]
DMV_GROWTH = dmv_get(DMV, 2021, DRV)["autonomous_miles"] / dmv_get(DMV, 2020, DRV)["autonomous_miles"]

head("Hook: Waymo DMV filings (SI 14.1)")
show("2015 miles per disengagement", f"{DMV_FIRST['miles_per_disengagement']:,.1f} (about {round(DMV_FIRST['miles_per_disengagement'], -2):,.0f})")
show("2025 miles per disengagement, safety driver", f"{DMV_LAST['miles_per_disengagement']:,.1f} (about {round(DMV_LAST['miles_per_disengagement'], -2):,.0f})")
show("2025 takeovers, safety driver", f"{DMV_LAST['disengagements']:.0f}")
show("2025 driverless miles per disengagement", f"{DMV_DL['miles_per_disengagement']:,.1f} (about {round(DMV_DL['miles_per_disengagement'], -2):,.0f}), {DMV_DL['disengagements']:.0f} events")
show("2020 to 2021 rate fell", f"{DMV_DROP:.2f}x (3.8x)")
show("2020 to 2021 miles grew", f"{DMV_GROWTH:.2f}x (3.7x)")
show("2025 safety-driver miles (millions)", f"{DMV_LAST['autonomous_miles'] / 1e6:.2f}")

# DMV 2024 CSV excerpt counts (raw rows; SOURCES.yml). Counted from the file by hand-run
# script in the build folder; typed here as source values, not derived.
DMV24_WAYMO_ROWS, DMV24_BY_DRIVER, DMV24_BY_AV = 244, 230, 14
head("Hook artifact: DMV 2024 CSV, Waymo rows (source counts)")
show("rows, by test driver, by AV system", f"{DMV24_WAYMO_ROWS}, {DMV24_BY_DRIVER}, {DMV24_BY_AV}")
show("share initiated by the test driver", f"{100 * DMV24_BY_DRIVER / DMV24_WAYMO_ROWS:.1f} percent")
show("registry 2024 drivered disengagements (cross-check)", f"{dmv_get(DMV, 2024, DRV)['disengagements']:.0f}")

# =============================================================================
# Recall: the finished budget at 1.3 m/s (SE Napkin Math 12.1)
# =============================================================================
D0 = stop_total(V_AISLE)
SPARE = D_CLEAR - D0
head("Recall: finished budget at the aisle speed")
show("pre-brake delay tau", f"{TAU * 1e3:.1f} ms")
show("jerk bound J", f"{J:.2f} m/s^3")
show("delay travel v*tau", f"{V_AISLE * TAU * 1e3:.2f} mm")
show("resident stop at 1.3 m/s", f"{stop_dist(V_AISLE) * 1e3:.2f} mm")
show("machine-alone stop", f"{D0 * 1e3:.2f} mm")
show("spare", f"{SPARE * 1e3:.1f} mm")
show("spare as delay", f"{SPARE / V_AISLE * 1e3:.1f} ms")
show("dispatch-to-admission travel", f"{V_AISLE * T_DA * 1e3:.1f} mm")
show("delay left after dispatch-to-admission", f"{(SPARE / V_AISLE - T_DA) * 1e3:.1f} ms")
show("mm left after dispatch-to-admission", f"{(SPARE - V_AISLE * T_DA) * 1e3:.1f} mm")

# =============================================================================
# Segment 1, timing strip: the in-loop takeover (SI 14.2.2, figure 2)
# =============================================================================
T_HS = sum(HS.values())
T_CONSOLE = T_VIDEO + T_HS
T_IN = T_PERC + T_COG + T_MOT + T_CONSOLE
T_OOL = TOT["p90"] + T_CONSOLE
head("Takeover timing strip (SI eq. 1)")
show("four-phase handshake", f"{T_HS * 1e3:.0f} ms")
show("console term (video + handshake)", f"{T_CONSOLE * 1e3:.0f} ms")
show("operator end (video + perceive + decide + act)", f"{(T_VIDEO + T_PERC + T_COG + T_MOT) * 1e3:.0f} ms")
show("in-loop takeover T1", f"{T_IN * 1e3:.0f} ms")
show("travel during T1 at 1.3 m/s", f"{V_AISLE * T_IN * 1e3:.1f} mm")
show("spare runs out at", f"{SPARE / V_AISLE * 1e3:.1f} ms (inside the 104 ms video stage)")
show("travel / spare", f"{V_AISLE * T_IN / SPARE:.1f}x")
show("out-of-loop takeover T2", f"{T_OOL:.3f} s (4.5 s)")

# =============================================================================
# Segment 1, predict: the takeover tax (SI Napkin Math 14.1)
# =============================================================================
D1 = stop_total(V_AISLE, T_IN)
D2 = stop_total(V_AISLE, T_OOL)
head("Takeover tax (Napkin Math 14.1)")
show("d_stop,0 machine alone", f"{D0 * 1e3:.2f} mm")
show("d_stop,1 in-loop", f"{D1 * 1e3:.2f} mm, overrun {(D1 - D_CLEAR) * 1e3:.1f} mm")
show("d_stop,2 out-of-loop", f"{D2 * 1e3:.2f} mm, overrun {(D2 - D_CLEAR):.2f} m")
show("travel during T2", f"{V_AISLE * T_OOL * 1e3:.1f} mm ({V_AISLE * T_OOL:.2f} m)")
show("d2 / d1", f"{D2 / D1:.2f}x (3.9x)")
show("T2 travel / clear distance", f"{V_AISLE * T_OOL / D_CLEAR:.1f}x")

# =============================================================================
# Segment 1, data: Zhang et al. 2019 condition means and the table 2 ceilings
# =============================================================================
MTOT = [float(r["mean_takeover_time_s"]) for r in csv.DictReader(open(HERE / "zhang2019_mean_takeover_times.csv"))]


def pct(vals, p):
    s = sorted(vals)
    return s[max(math.ceil(p / 100 * len(s)) - 1, 0)]


head("Zhang et al. 2019, 520 condition means (data file)")
show("n", len(MTOT))
show("mean, SD", f"{sum(MTOT) / len(MTOT):.2f} s, {math.sqrt(sum((x - sum(MTOT) / len(MTOT)) ** 2 for x in MTOT) / (len(MTOT) - 1)):.2f} s")
show("p50, p90, p99", f"{pct(MTOT, 50):.2f}, {pct(MTOT, 90):.2f}, {pct(MTOT, 99):.2f} s")
show("max", f"{max(MTOT):.2f} s")
show("conditions above 10 s", sum(1 for x in MTOT if x > 10))
show("conditions above 4.33 s", sum(1 for x in MTOT if x > 4.33))

TABLE2 = [
    ("In-loop, nominal pace", T_PERC_N + T_COG_N + T_MOT_N),
    ("In-loop, slow bound (design)", T_PERC + T_COG + T_MOT),
    ("Expected brake signal", T_BRAKE_EXP),
    ("Unexpected but common signal", T_BRAKE_UNEXP),
    ("Surprise event", T_BRAKE_SURP),
    ("Out-of-loop, median condition", TOT["p50"]),
    ("Out-of-loop, average condition", TOT["mean"]),
    ("Out-of-loop, 90th percentile (design)", TOT["p90"]),
    ("Out-of-loop, 99th percentile", TOT["p99"]),
]
CEIL = {}
head("Table 2: speed ceilings (human + 135 ms console, partial budget)")
for name, th in TABLE2:
    w = th + T_CONSOLE
    CEIL[name] = ceiling(w)
    show(name, f"human {th * 1e3:,.0f} ms, with console {w * 1e3:,.0f} ms, travel {V_AISLE * w:.2f} m, ceiling {CEIL[name]:.2f} m/s")

# =============================================================================
# Segment 1, worked example: the speed a human buys (SI Napkin Math 14.2)
# =============================================================================
T_FULL = TAU + T_IN + A_BRAKE / (2 * J)
ROOM = D_CLEAR - ALLOW
V_IN = A_BRAKE * (math.sqrt(T_FULL ** 2 + 2 * ROOM / A_BRAKE) - T_FULL)
V_PERFECT = ROOM / (TAU + T_IN)
T_FULL_DA = T_FULL + T_DA
V_IN_DA = A_BRAKE * (math.sqrt(T_FULL_DA ** 2 + 2 * ROOM / A_BRAKE) - T_FULL_DA)
head("Speed a human buys (Napkin Math 14.2, eq. 3)")
show("room to brake D - allowances", f"{ROOM:.2f} m")
show("half jerk ramp a/(2J)", f"{A_BRAKE / (2 * J):.4f} s")
show("T_full = tau + T1 + a/2J", f"{T_FULL:.4f} s")
show("T_full^2 + 2 room / a", f"{T_FULL ** 2 + 2 * ROOM / A_BRAKE:.4f} s^2, root {math.sqrt(T_FULL ** 2 + 2 * ROOM / A_BRAKE):.4f} s")
show("in-loop ceiling", f"{V_IN:.3f} m/s")
show("transition speed a^2/J", f"{A_BRAKE ** 2 / J:.3f} m/s")
show("perfect brake ceiling", f"{V_PERFECT:.2f} m/s")
show("with dispatch-to-admission", f"{V_IN_DA:.3f} m/s")
show("solver cross-check, in-loop", f"{ceiling(T_IN):.3f} m/s; with T_DA {ceiling(T_IN + T_DA):.3f} m/s")
show("solver, out-of-loop", f"{ceiling(T_OOL):.3f} m/s; with T_DA {ceiling(T_OOL + T_DA):.3f} m/s")

# =============================================================================
# Segment 1, deadline (SI Napkin Math 14.3, eq. 4)
# =============================================================================
V_SLOW = 0.75   # m/s, the complete-budget in-loop ceiling 0.797 rounded down to 0.05 (SI 14.2.4)
V_CRAWL = 0.15  # m/s, chosen crawl before an out-of-loop request (SI 14.2.4)
DL_AISLE = deadline(V_AISLE)
DL_SLOW = deadline(V_SLOW)
DL_CRAWL = deadline(V_CRAWL)
head("Fallback deadline (Napkin Math 14.3)")
show("resident stop at 0.75 m/s", f"{stop_dist(V_SLOW):.4f} m")
show("deadline at 1.3 m/s", f"{DL_AISLE * 1e3:.1f} ms; whole ticks {math.floor(DL_AISLE / TICK)} ms")
show("deadline at 1.3 m/s after T_DA", f"{(DL_AISLE - T_DA) * 1e3:.1f} ms")
show("deadline at 0.75 m/s", f"{DL_SLOW * 1e3:.1f} ms")
show("left after the 605 ms takeover", f"{(DL_SLOW - T_IN) * 1e3:.1f} ms; after T_DA {(DL_SLOW - T_IN - T_DA) * 1e3:.1f} ms")
show("ratio deadline 0.75 / 1.3", f"{DL_SLOW / DL_AISLE:.1f}x")
show("spare at 0.75 m/s", f"{(D_CLEAR - stop_total(V_SLOW)) * 1e3:.1f} mm")
show("deadline at 0.8 m/s, minus T1, minus T_DA", f"{(deadline(0.8) - T_IN) * 1e3:.1f} ms, {(deadline(0.8) - T_IN - T_DA) * 1e3:.1f} ms")
show("deadline at 0.15 m/s", f"{DL_CRAWL:.2f} s; left after T2 {DL_CRAWL - T_OOL:.2f} s")
show("UN R157 10 s over T2", f"{10 / T_OOL:.1f}x; travel at 1.3 m/s {10 * V_AISLE:.0f} m")
show("takeover timeout: ceiling 500 ms vs deadline after T_DA", f"tighter by {(0.5 - (DL_AISLE - T_DA)) * 1e3:.1f} ms; whole ticks {math.floor(round((DL_AISLE - T_DA) / TICK, 6))}")

# =============================================================================
# Segment 1, estimate: an operator waiting for an expected brake signal
# =============================================================================
W_EXP = T_BRAKE_EXP + T_CONSOLE
head("Estimate: expected brake signal (Checkpoint 14.1)")
show("wait", f"{W_EXP * 1e3:.0f} ms")
T_FULL_EXP = TAU + W_EXP + A_BRAKE / (2 * J)
show("T_full", f"{T_FULL_EXP:.4f} s")
show("ceiling by eq. 3", f"{A_BRAKE * (math.sqrt(T_FULL_EXP ** 2 + 2 * ROOM / A_BRAKE) - T_FULL_EXP):.3f} m/s")
show("travel during the wait at 1.3 m/s", f"{V_AISLE * W_EXP:.2f} m")

# =============================================================================
# Segment 2: leases and the handshake (SI 14.3)
# =============================================================================
head("Leases and handshake")
show("lease extension travel at 0.75 m/s v(T_man - T_chunk)", f"{V_SLOW * (T_MANUAL - T_CHUNK) * 1e3:.1f} mm")
show("travel in one 100 ms manual lease at 1.3 m/s", f"{V_AISLE * T_MANUAL * 1e3:.0f} mm")
show("handshake phases", ", ".join(f"{k} {v * 1e3:.0f} ms" for k, v in HS.items()) + f"; total {T_HS * 1e3:.0f} ms")

# =============================================================================
# Segment 3: bumpless transfer (SI 14.4, eq. 6 and 7)
# =============================================================================
PEAK = 30 / 16                     # peak slope of 10s^3 - 15s^4 + 6s^5
DA_BASE = 2.0                      # m/s^2, base step to braking (SI 14.4.2)
DTAU_ARM, TAU_RATE = 15.0, 750.0   # N m and N m/s (illustrative, SI 14.4.2)
W_BASE = PEAK * DA_BASE / J
W_ARM = PEAK * DTAU_ARM / TAU_RATE


def blend_stop(v0, n=200000):
    """Smoothstep acceleration ramp 0 -> -a over W_BASE, then full braking (SI 14.4.2)."""
    dt = W_BASE / n
    v, x = v0, 0.0
    for i in range(n):
        s = (i + 0.5) / n
        acc = -DA_BASE * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
        x += v * dt + 0.5 * acc * dt * dt
        v += acc * dt
    return x, v, v * v / (2 * DA_BASE)


BX, BV, BREST = blend_stop(V_SLOW)
head("Bumpless transfer")
show("smoothstep peak slope", f"{PEAK:.3f}")
show("base window 1.875*2/J", f"{W_BASE * 1e3:.2f} ms -> {math.ceil(W_BASE / TICK)} ticks")
show("arm window 1.875*15/750", f"{W_ARM * 1e3:.1f} ms -> {math.ceil(round(W_ARM / TICK, 6))} ticks")
show("step in one 1 ms tick: jerk", f"{DA_BASE / TICK:,.0f} m/s^3")
show("resident ramp a/J", f"{A_BRAKE / J * 1e3:.2f} ms")
show("blend from 0.75: exit speed", f"{BV:.2f} m/s; travel {BX * 1e3:.1f} mm; then {BREST * 1e3:.1f} mm; total {(BX + BREST) * 1e3:.1f} mm")
show("ideal step stop from 0.75", f"{V_SLOW ** 2 / (2 * A_BRAKE) * 1e3:.1f} mm")
show("resident stop from 0.75", f"{stop_dist(V_SLOW) * 1e3:.1f} mm")
show("blend minus step", f"{(BX + BREST - V_SLOW ** 2 / (2 * A_BRAKE)) * 1e3:.1f} mm; minus resident {(BX + BREST - stop_dist(V_SLOW)) * 1e3:.1f} mm")

# =============================================================================
# Segment 3: the remote-assistant session at 0.75 m/s (SI table 7)
# =============================================================================
S0 = D_CLEAR - stop_total(V_SLOW)
S1 = S0 - V_SLOW * T_IN
S2 = S1 - V_SLOW * T_DA
S3 = S2 - V_SLOW * (T_MANUAL - T_CHUNK)
BLEND_RESERVE = 0.1570  # m, "up to 157.0 mm" the braking blend would use (SI table 7, source value)
head("Session admission waterfall (SI table 7)")
show("finished budget's spare", f"{S0 * 1e3:.1f} mm")
show("after takeover (travel)", f"{V_SLOW * T_IN * 1e3:.1f} mm -> {S1 * 1e3:.1f} mm")
show("after dispatch-to-admission", f"{V_SLOW * T_DA * 1e3:.1f} mm -> {S2 * 1e3:.1f} mm")
show("after manual lease extension", f"{V_SLOW * (T_MANUAL - T_CHUNK) * 1e3:.1f} mm -> {S3 * 1e3:.1f} mm")
show("braking blend would use (SI)", f"up to {BLEND_RESERVE * 1e3:.1f} mm; short by {(BLEND_RESERVE - S3) * 1e3:.1f} mm")
show("crawl remainder after T2", f"{(D_CLEAR - stop_total(V_CRAWL) - V_CRAWL * T_OOL) * 1e3:.1f} mm")

# =============================================================================
# Segment 3: authentication cost and verifier flooding (SI 14.5, 14.9)
# =============================================================================
T_ED = 0.0018     # s per Ed25519 verification, illustrative estimate (SI 14.5; Owens et al. 2024)
head("Authentication and flooding")
show("Ed25519 travel at 1.3 m/s", f"{V_AISLE * T_ED * 1e3:.2f} mm; spare left {(SPARE - V_AISLE * T_ED) * 1e3:.1f} mm")
show("100 ms freshness window travel", f"{V_AISLE * 0.100 * 1e3:.0f} mm; spare left {(SPARE - V_AISLE * 0.100) * 1e3:.1f} mm")


def forged_to_miss(budget):
    n = 0
    while (n + 1) * T_ED <= budget:
        n += 1
    return n   # with n forged queued ahead, the veto finishes at (n + 1) * T_ED > budget


N_MISS = forged_to_miss(DL_AISLE)
N_MISS_DA = forged_to_miss(DL_AISLE - T_DA)
show("deadline (unrounded)", f"{DL_AISLE * 1e3:.2f} ms")
show("forged packets to miss the partial deadline", f"{N_MISS} -> veto done at {(N_MISS + 1) * T_ED * 1e3:.2f} ms")
show("forged packets to miss after T_DA", f"{N_MISS_DA} -> veto done at {(N_MISS_DA + 1) * T_ED * 1e3:.2f} ms")
show("injection rate", f"{N_MISS / ((N_MISS + 1) * T_ED):.0f} packets per second")

# =============================================================================
# Segment 4: mode confusion and the record (SI 14.6 to 14.8)
# =============================================================================
head("Mode confusion and the record")
show("base travel while the operator aligns the arm (605 ms at 1.3 m/s)", f"{V_AISLE * T_IN * 1e3:.1f} mm = {V_AISLE * T_IN / SPARE:.1f}x the spare")
show("EDR today: samples in 5 s at 2 Hz", f"{5 * 2}")
show("EDR from 2027: samples in 20 s at 10 Hz", f"{20 * 10}; one per {1000 / 10:.0f} ms = {0.1 / TICK:.0f} ticks")
show("UN R157 timestamp accuracy 1 s in ticks", f"{1 / TICK:.0f}")
show("authority record: tuples in the 5 s pre-trigger window", f"{5 / TICK:,.0f}")
show("samples in the last 5 s: EDR 2 Hz, EDR 10 Hz, record 1 kHz", f"{5 * 2}, {5 * 10}, {5 / TICK:,.0f}")
DRIFT = T_OOL
show("interval 1 ticks for a 4.5 s takeover", f"{round(4.5 / TICK):,}")
show("interval 2 ticks (arm blend)", f"{math.ceil(round(W_ARM / TICK, 6))}")
show("Sirius window 2 s vs 4.5 s", f"{4.5 / 2.0:.1f}x; drift ticks outside the window {round((4.5 - 2.0) / TICK):,}")

# =============================================================================
# Pitfall: one fixed 500 ms window at two speeds (SI 14.9)
# =============================================================================
head("Fixed 500 ms window")
show("at 1.3 m/s: travel, total, overrun", f"{V_AISLE * 0.5:.2f} m, {stop_total(V_AISLE, 0.5):.2f} m, {stop_total(V_AISLE, 0.5) - D_CLEAR:.2f} m")
show("at 0.15 m/s: total", f"{stop_total(V_CRAWL, 0.5):.2f} m")

# =============================================================================
# Segment 2: a person beside the machine (SI 14.3.1 footnote 5)
# =============================================================================
V_MANUAL_REDUCED = 0.250   # m/s, ISO 10218-1:2025 manual reduced speed limit (SI footnote 5)
head("Pendant at manual reduced speed")
show("travel during a 470 ms in-loop human response at 250 mm/s", f"{V_MANUAL_REDUCED * (T_PERC + T_COG + T_MOT) * 1e3:.1f} mm")
show("travel during the 605 ms remote in-loop takeover at 1.3 m/s", f"{V_AISLE * T_IN * 1e3:.1f} mm")
show("ratio", f"{V_AISLE * T_IN / (V_MANUAL_REDUCED * (T_PERC + T_COG + T_MOT)):.1f}x")
