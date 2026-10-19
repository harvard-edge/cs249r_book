#!/usr/bin/env python3
"""Every derived number in Lecture 5 (Physical Data), Volume IV.

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered chapter, Physical Data, rendered 2026-10-18); each
printed line is a value the slide or its note shows, under the label used
there. make_figs.py imports from this file so figures and slides share one
computation. Published series (LeRobot Hub, Open X-Embodiment) are read from
the MLSysIM literature registry, the same source the chapter prints from.

Run from this folder: python3 calcs.py
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "mlsysim"))
from mlsysim import Literature  # noqa: E402


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
DROID_RECAL_EPISODES = 36_000      # ch Sec. 5.10.1 (Khazatsky et al. 2024)
DROID_TRAJ, DROID_HOURS, DROID_SCENES = 76_000, 350, 564   # ch table 5

# ESS (ch Sec. 5.4, equation 2)
LOG_HZ, RHO = 1000, 0.999

# Compounding error (ch Sec. 5.8 comparison)
S_STEPS, EPS, U_REC = 500, 0.001, 1
CONTROL_HZ = 100                   # ch: 500 steps = 5 s at 100 Hz

# Pairing lag (ch Napkin Math 5.1)
LAG_MS = 35
CLEAR_MM = 2.0
V_TRANSIT, V_GUARD, V_FAST = 0.50, 0.03, 0.10   # m/s, illustrative
CLEAR_TIGHT_MM = 1.0               # ch Checkpoint 5.1
SKEW_MS = (15, 33)                 # ch Sec. 5.5.2, rolling-shutter full frame

# Teleoperation loop (ch Sec. 5.6.1, illustrative reference setup)
T_DISP, T_HUM, T_CMD, T_MECH = 50, 200, 10, 80   # ms

# Yield and wear (ch Napkin Math 5.3, illustrative)
SESSION_MIN, ACTIVE_MIN = 60, 35
ATTEMPT_S, RESET_S = 25, 15
FAIL, QA = 0.20, 0.15
TARGET = 5000
GEAR_REV, CABLE_FLEX = 14, 8
GEAR_LIFE, CABLE_LIFE = 5_000_000, 200_000

# Bytes (ch Napkin Math 5.2)
NAV = (4056, 3040, 3, 60)
WRIST = (1440, 1080, 3, 60)
EXT_N, EXT_W, EXT_H, EXT_BPP, EXT_HZ = 2, 1920, 1080, 3 + 2, 30
AXES, VALS, BYTES, TS = 9, 6, 4, 8
SATA, NVME = 550, 7000             # MB/s, ch
COMPRESS = 8

# Contact-angle grid (ch Sec. 5.7, illustrative)
ANG_BINS, FORCE_BINS = 30, 20      # +/-15 deg at 1 deg; 0 to 20 N at 1 N
CAUTIOUS = (4, 2)                  # +/-2 deg x 4 to 6 N
EXPLORE_CELLS = 510
RMS = (0.001, 0.012)               # rad, ch table 4

# Curse of dimensionality (ch Sec. 5.7.1)
DIMS, BINS_COARSE, BINS_FINE = 28, 10, 100
ROBOTS, UNIVERSE_YR = 1000, 13.8e9
SEC_PER_YR = 365.25 * 86400

# Scenario table (ch Sec. 5.7)
OFFSETS, SPEEDS = 3, 2

# Interventions (ch Sec. 5.8)
TAKEOVER_S = 4.47                  # ch: out-of-loop takeover budget
TICK_HZ = 1000
# 49 CFR 563.7 Table I (eCFR, retrieved 2026-10-19): speed, throttle, brake
EDR_NEW = (20, 10)                 # -20 to 0 s at 10 samples/s (phase-in from 2028)
EDR_OLD = (5, 2)                   # -5.0 to 0 s at 2 samples/s (before phase-in)
# Hoque et al. 2022 (ThriftyDAgger), Table 3: physical cable routing,
# autonomous success out of 15 trials
THRIFTY = {"Behavior cloning": (0, 15), "HG-DAgger": (10, 15), "ThriftyDAgger": (12, 15)}

# Embodiment (ch Sec. 5.9.1)
NORM_CMD = 0.80
STROKE_A, STROKE_B, PART = 100, 60, 75     # mm
FORCE_A, FORCE_B = 30, 15                  # N

# MCO (ch War Story 5.1)
LBF_S = 4.45

# LeRobot example artifact: lerobot/aloha_static_coffee meta/info.json
# (images/data/lerobot_aloha_info.json, retrieved 2026-10-19)
ALOHA_EP, ALOHA_FRAMES, ALOHA_FPS = 50, 55_000, 50


def ess():
    head("ESS: an hour at 1 kHz")
    n = LOG_HZ * 3600
    e = n * (1 - RHO) / (1 + RHO)
    show("records per hour", f"{n:,.0f}")
    show("effective samples (rho 0.999)", f"{e:,.0f}")
    return n, e


def bounds():
    head("Compounding error bounds")
    bc = S_STEPS ** 2 * EPS
    ol = S_STEPS * EPS
    dag = U_REC * S_STEPS * EPS
    show("S", f"{S_STEPS} steps = {S_STEPS / CONTROL_HZ:g} s at {CONTROL_HZ} Hz")
    show("open loop S*eps", f"{ol:g}")
    show("cloning S^2*eps", f"{bc:g}")
    show("aggregation u*S*eps (u=1)", f"{dag:g}")
    show("vacuous above eps = 1/S", f"{1 / S_STEPS:g}")
    show("cloning bound / episode max", f"{bc / S_STEPS:g}")
    show("checkpoint: 10 s episode, threshold 1/S", f"{1 / (2 * S_STEPS):g}")
    return bc, ol, dag


def pairing():
    head("Pairing lag (Napkin Math 5.1)")
    for name, v in (("transit", V_TRANSIT), ("guarded", V_GUARD), ("fast", V_FAST)):
        d = v * LAG_MS
        show(f"{name} {v} m/s x {LAG_MS} ms", f"{d:.2f} mm ({d / CLEAR_MM:.2f}x clearance)")
    show("ceiling guarded", f"{CLEAR_MM / V_GUARD:.1f} ms")
    show("ceiling fast", f"{CLEAR_MM / V_FAST:.1f} ms")
    show("tight plate 1 mm at 0.10 m/s", f"{CLEAR_TIGHT_MM / V_FAST:.0f} ms")
    show("rolling shutter 33 ms at 0.10 m/s", f"{V_FAST * SKEW_MS[1]:.1f} mm")


def teleop():
    head("Teleoperation loop")
    tot = T_DISP + T_HUM + T_CMD + T_MECH
    show("loop", f"{tot} ms")
    show("human share", f"{100 * T_HUM / tot:.1f} %")
    show("operator command at", f"t + {T_DISP + T_HUM} ms")
    return tot


def yield_wear():
    head("Yield and wear (Napkin Math 5.3)")
    cycle = ATTEMPT_S + RESET_S
    attempts = math.floor(ACTIVE_MIN * 60 / cycle)
    idle = ACTIVE_MIN * 60 - attempts * cycle
    keep = (1 - FAIL) * (1 - QA)
    eps_kept = attempts * keep
    kept_min = eps_kept * ATTEMPT_S / 60
    share = 100 * kept_min / SESSION_MIN
    camp = math.ceil(TARGET / keep)
    hours = camp / attempts * SESSION_MIN / 60
    motion_h = TARGET * ATTEMPT_S / 3600
    gear = camp * GEAR_REV
    cable = camp * CABLE_FLEX
    show("cycle", f"{cycle} s; attempts {attempts}; idle {idle} s")
    show("retention", f"{keep:.2f}")
    show("episodes kept per session", f"{eps_kept:.2f}")
    show("kept minutes", f"{kept_min:.2f} min = {share:.1f} % of hour")
    show("campaign attempts", f"{camp:,}")
    show("scheduled hours", f"{hours:.1f} h")
    show("kept motion hours", f"{motion_h:.1f} h; ratio {hours / motion_h:.2f}x")
    show("gear cycles", f"{gear:,} = {100 * gear / GEAR_LIFE:.1f} % of life")
    show("cable bends", f"{cable:,} = {100 * cable / CABLE_LIFE:.1f} % of life")
    show("three campaigns, cable", f"{300 * cable / CABLE_LIFE:.0f} %")
    stages = {"startup": 8, "calibration": 7, "rest": 10,
              "resets": attempts * RESET_S / 60, "attempts": attempts * ATTEMPT_S / 60}
    show("session stages (min)", ", ".join(f"{k} {v:.2f}" for k, v in stages.items()))
    show("failed attempts (min)", f"{attempts * FAIL * ATTEMPT_S / 60:.2f}")
    show("QA discards (min)", f"{attempts * (1 - FAIL) * QA * ATTEMPT_S / 60:.2f}")
    return dict(attempts=attempts, keep=keep, kept_min=kept_min, share=share, camp=camp,
                hours=hours, gear_pct=100 * gear / GEAR_LIFE, cable_pct=100 * cable / CABLE_LIFE,
                stages=stages, eps_kept=eps_kept)


def bytes_rate():
    head("Bytes (Napkin Math 5.2)")
    kin = (AXES * VALS * BYTES + TS) * 1000 / 1e6
    nav = NAV[0] * NAV[1] * NAV[2] * NAV[3] / 1e6
    wri = WRIST[0] * WRIST[1] * WRIST[2] * WRIST[3] / 1e6
    ext = EXT_N * EXT_W * EXT_H * EXT_BPP * EXT_HZ / 1e6
    tot = kin + nav + wri + ext
    show("kinematic", f"{kin:.3f} MB/s")
    show("nav", f"{nav:.2f} MB/s; / SATA = {nav / SATA:.1f}x")
    show("wrist", f"{wri:.2f}; external {ext:.2f}")
    show("total", f"{tot:.2f} MB/s; per hour {tot * 3600 / 1e6:.2f} TB; 8 h {tot * 8 * 3600 / 1e6:.1f} TB")
    show("kinematic share", f"{100 * kin / tot:.3f} %")
    show("compressed 8x", f"{tot / COMPRESS:.1f} MB/s")
    return dict(kin=kin, nav=nav, tot=tot, comp=tot / COMPRESS)


def grid():
    head("Contact-angle grid")
    cells = ANG_BINS * FORCE_BINS
    c = CAUTIOUS[0] * CAUTIOUS[1]
    show("cells", cells)
    show("cautious", f"{c} = {100 * c / cells:.1f} %; empty {100 * (cells - c) / cells:.1f} %")
    show("exploratory", f"{EXPLORE_CELLS} = {100 * EXPLORE_CELLS / cells:.1f} %; empty "
         f"{100 * (cells - EXPLORE_CELLS) / cells:.1f} %")
    recs = ROBOTS * LOG_HZ * UNIVERSE_YR * SEC_PER_YR
    show("records since universe began", f"{recs:.3g}")
    show("share of 10^28 cells", f"{100 * recs / BINS_COARSE ** DIMS:.3f} %")
    return cells, c


def scenario():
    head("Scenario table")
    n = OFFSETS * SPEEDS
    show("cells", f"{n}; supported 1; absences {n - 1}")
    show("after latch-recovery-02", f"supported 2; absences {n - 2}")
    n3 = OFFSETS * 3
    show("third speed", f"{n3} cells; absences for original {n3 - 1}")


def interventions():
    head("Interventions")
    win = math.ceil(TAKEOVER_S)
    show("lookback", f"{win} s = {win * TICK_HZ:,} ticks at 1 kHz")
    show("EDR new", f"{EDR_NEW[0] * EDR_NEW[1]} samples")
    show("EDR old", f"{EDR_OLD[0] * EDR_OLD[1]} samples")
    for k, (s, n) in THRIFTY.items():
        show(f"ThriftyDAgger table 3, {k}", f"{s}/{n} = {100 * s / n:.0f} %")


def embodiment():
    head("Normalization fallacy")
    a = NORM_CMD * STROKE_A
    b = NORM_CMD * STROKE_B
    show("donor opening", f"{a:.0f} mm; target opening {b:.0f} mm against {PART} mm part")
    show("stroke excess", f"{a - STROKE_B:.0f} mm; force ratio {FORCE_A / FORCE_B:.0f}x")
    show("ALOHA example episode", f"{ALOHA_FRAMES / ALOHA_EP:,.0f} frames = "
         f"{ALOHA_FRAMES / ALOHA_EP / ALOHA_FPS:.0f} s at {ALOHA_FPS} fps")


def lerobot():
    head("LeRobot Hub (MLSysIM Literature.LeRobotHub*)")
    months = Literature.LeRobotHubMonthly.entries
    at = {m["month"]: m["cumulative_datasets"] for m in months}
    bins = Literature.LeRobotEpisodeHistogram.entries
    n = sum(b["datasets"] for b in bins)
    small = sum(b["datasets"] for b in bins if b["bin_hi"] <= 16)
    large = sum(b["datasets"] for b in bins if b["bin_lo"] >= 512)
    pooled = {}
    for b in bins:
        pooled[b["bin_lo"]] = pooled.get(b["bin_lo"], 0) + b["datasets"]
    run, med = 0, None
    for lo in sorted(pooled):
        run += pooled[lo]
        if run >= n / 2:
            med = lo
            break
    show("datasets end 2024", f"{at['2024-12']:,.0f}")
    show("datasets end 2025", f"{at['2025-12']:,.0f}")
    show("datasets now", f"{months[-1]['cumulative_datasets']:,.0f} ({months[-1]['month']})")
    show("growth since end 2024", f"{months[-1]['cumulative_datasets'] / at['2024-12']:.0f}x")
    show("datasets with a count", f"{n:,.0f}")
    show("under 16 episodes", f"{100 * small / n:.0f} %")
    show("512 or more", f"{100 * large / n:.0f} %")
    show("median bin", f"{med:g} to {2 * med - 1:g}")
    return months, pooled, n


def oxe():
    head("Open X-Embodiment (MLSysIM Literature.OpenXEmbodimentComposition)")
    rows = [r for r in Literature.OpenXEmbodimentComposition.entries if r["episodes"]]
    tot = sum(r["episodes"] for r in rows)
    morph, rate = {}, {}
    for r in rows:
        k = r["robot_morphology"]
        morph[k] = morph.get(k, 0) + r["episodes"]
        hz = r["control_frequency_hz"] if isinstance(r["control_frequency_hz"], float) else None
        rate[hz] = rate.get(hz, 0) + r["episodes"]
    single = morph.get("Single Arm", 0)
    mobile = morph.get("Mobile Manipulator", 0)
    table = sum(r["episodes"] for r in rows if "table" in str(r["scene_type"]).lower())
    mob = sorted((r["episodes"] for r in rows if r["robot_morphology"] == "Mobile Manipulator"))
    ge20 = sum(v for k, v in rate.items() if k is not None and k >= 20)
    show("datasets", len(rows))
    show("episodes", f"{tot:,.0f} ({tot / 1e6:.1f} million)")
    show("single arm", f"{100 * single / tot:.0f} %; table-top {100 * table / tot:.0f} %")
    show("mobile manipulator", f"{mobile:,.0f} = {100 * mobile / tot:.0f} %; top two "
         f"{100 * sum(mob[-2:]) / mobile:.0f} %")
    show("at 10 Hz", f"{100 * rate.get(10.0, 0) / tot:.0f} %; no rate {100 * rate[None] / tot:.0f} %")
    show("at or above 20 Hz", f"{100 * ge20 / tot:.0f} %")
    for k in sorted(morph, key=morph.get, reverse=True):
        show(f"  morph {k}", f"{morph[k]:,.0f}")
    return rows, morph, rate, tot


if __name__ == "__main__":
    head("Hook: DROID")
    show("episodes recalibrated", f"{DROID_RECAL_EPISODES:,}")
    show("share of trajectories", f"{100 * DROID_RECAL_EPISODES / DROID_TRAJ:.0f} %")
    ess()
    bounds()
    pairing()
    teleop()
    yield_wear()
    bytes_rate()
    grid()
    scenario()
    lerobot()
    oxe()
    interventions()
    embodiment()
