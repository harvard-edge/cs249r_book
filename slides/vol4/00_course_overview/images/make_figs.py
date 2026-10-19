"""Figures for Volume IV, Lecture 0 (Course overview).

Run:  python3 make_figs.py   (writes SVGs here; `make deck V=4 D=00_course_overview`
converts them to PDF). Every derived value comes from calcs.py or is computed
here from typed inputs; data plots read the MLSysIM registry.
"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))   # slides/assets/kit
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon  # noqa: E402
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402

# Volume IV roles (rules/volumes/vol4/palette.md)
NAVY = "#1F407A"      # cognitive deliberation (learned model)
PROPOSE = "#2563EB"   # unprivileged proposal
NERVE = "#059669"     # deterministic nervous system, permission path
RED = "#DC2626"       # consequence: clearance, veto, hazard
PLANT = "#334155"     # physical plant
AMBER = "#D97706"     # telemetry, latency lag
GRAY = "#94A3B8"
LIGHT = "#E2E8F0"
TEXT = "#1A1A2E"


def out(name):
    return str(HERE / name)


# ---------------------------------------------------------------------------
# Hook: the aisle in top view (stepped). Positions in mm along the aisle.
# ---------------------------------------------------------------------------
def aisle(step):
    d1, d2 = calcs.stop_mm(calcs.V_DRIVE)
    age_mm = calcs.V_DRIVE * calcs.obs_age()
    clear = calcs.D_CLEAR * 1000
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    ax.add_patch(Rectangle((-700, -400), 2400, 780, color="#F8FAFC", zorder=0))
    ax.add_patch(Rectangle((-700, 380), 1780, 110, color=GRAY, zorder=1))
    ax.add_patch(Rectangle((-700, -510), 1780, 110, color=GRAY, zorder=1))
    ax.text(-680, 435, "rack", color="white", fontsize=11, va="center", fontweight="bold")
    ax.add_patch(FancyBboxPatch((-600, -170), 560, 340, boxstyle="round,pad=0,rounding_size=40",
                                fc=PLANT, ec=PLANT, zorder=3))
    ax.text(-320, 0, "base\n1.5 m/s", color="white", fontsize=12, ha="center", va="center",
            fontweight="bold", zorder=4)
    ax.add_patch(Circle((clear + 90, 250), 60, color=RED, zorder=4))
    ax.text(clear + 170, 250, "person", color=RED, fontsize=12, va="center", fontweight="bold")
    ax.plot([clear, clear], [-400, 380], color=RED, lw=2, ls="--", zorder=2)
    ax.annotate("", xy=(clear, -250), xytext=(0, -250),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(clear / 2, -280, "clear distance 1,100 mm", color=RED, fontsize=12, ha="center",
            va="top", fontweight="bold")
    if step >= 2:
        ax.add_patch(Rectangle((0, 200), age_mm, 100, color=AMBER, zorder=3))
        ax.text(age_mm + 40, 250, f"one frame: {age_mm:.1f} mm", color=AMBER, fontsize=12,
                va="center", fontweight="bold")
    if step >= 3:
        ax.add_patch(Rectangle((0, -60), d1, 120, color=AMBER, zorder=3))
        ax.add_patch(Rectangle((d1, -60), d2, 120, color=PLANT, alpha=0.6, zorder=3))
        ax.text(d1 / 2, 0, f"{d1:.1f}", color="white", fontsize=11, ha="center",
                va="center", fontweight="bold", zorder=4)
        ax.text(d1 + d2 / 2, 0, f"{d2:.1f}", color="white", fontsize=11,
                ha="center", va="center", fontweight="bold", zorder=4)
        ax.text(d1 + d2 + 25, 0, f"stop\n{d1 + d2:.1f}", color=PLANT, fontsize=11,
                va="center", fontweight="bold")
    ax.set_xlim(-700, 1700)
    ax.set_ylim(-510, 490)
    ax.set_aspect("equal")
    ax.axis("off")
    sp.save(fig, out(f"aisle_step{step}.svg"))


# ---------------------------------------------------------------------------
# Observation age of one frame, grouped stages (Napkin Math 8.2)
# ---------------------------------------------------------------------------
def frame_age():
    s = dict(calcs.AGE_STAGES)
    groups = [
        ("Exposure + readout", s["Half exposure"] + s["Readout"], AMBER),
        ("Transfer + image processing", s["Link"] + s["DMA queue"] + s["ISP"], GRAY),
        ("Learned backbone", s["Backbone"], sp.C["compute"]),
        ("", s["Dispatch"], GRAY),
    ]
    total = sum(g[1] for g in groups)
    fig, ax = plt.subplots(figsize=(5.8, 1.8))
    x = 0
    for i, (name, v, c) in enumerate(groups):
        ax.barh(0, v, left=x, color=c, height=0.55, edgecolor="white", lw=1.5)
        if v > 3:
            ax.text(x + v / 2, 0, f"{v:.1f}", color="white", fontsize=13 if v > 10 else 11,
                    ha="center", va="center", fontweight="bold")
            above = i != 1
            ax.text(x + v / 2, 0.36 if above else -0.36, name, color=c if c != GRAY else PLANT,
                    fontsize=11, ha="center", va="bottom" if above else "top", fontweight="bold")
        x += v
    ax.text(total + 0.8, 0, f"{total:.1f} ms\n= {calcs.V_DRIVE * total:.1f} mm", fontsize=13,
            va="center", color=TEXT, fontweight="bold")
    ax.set_xlim(0, total * 1.28)
    ax.set_ylim(-0.75, 0.8)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms after the exposure midpoint")
    sp.save(fig, out("frame_age.svg"))


# ---------------------------------------------------------------------------
# Where the learned output lands (sec. 1.2, figure 1, band beneath the axis)
# ---------------------------------------------------------------------------
def learned_band(step):
    eras = [("nothing\nlearned", "to mid-1980s", LIGHT, PLANT),
            ("learned\nperception", "1986 to 2015", "#CBD5E1", PLANT),
            ("learned motor\ncommands", "2016 to 2023", "#93C5FD", NAVY),
            ("pretrained\nproposers", "2023 onward", NAVY, "white")]
    fig, ax = plt.subplots(figsize=(5.8, 1.2))
    for i, (lab, yrs, fc, tc) in enumerate(eras):
        a = 1 if i < step else 0
        ax.add_patch(Rectangle((i, 0), 1, 1, fc=fc, ec="white", lw=3, alpha=a))
        ax.text(i + 0.5, 0.64, lab, ha="center", va="center", fontsize=11, color=tc,
                fontweight="bold", alpha=a, linespacing=1.0)
        ax.text(i + 0.5, 0.17, yrs, ha="center", va="center", fontsize=10, color=tc, alpha=a)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    sp.save(fig, out(f"learned_band_step{step}.svg"))


# ---------------------------------------------------------------------------
# Epoch AI training compute by domain (sec. 1.2, figure 2)
# ---------------------------------------------------------------------------
def epoch(step):
    from datetime import date
    from mlsysim import Literature
    rows = Literature.ModelsByDomain.entries

    def yr(d):
        d = str(d)
        y, m = int(d[:4]), int(d[5:7]) if len(d) >= 7 else 6
        return y + (m - 0.5) / 12

    lang = [(yr(r["publication_date"]), r["training_flop"]) for r in rows
            if r["domain_group"] == "Language" and r["training_flop"] and yr(r["publication_date"]) >= 2018]
    rob = [(yr(r["publication_date"]), r["training_flop"], r["model"]) for r in rows
           if r["domain_group"] == "Robotics" and r["training_flop"] and yr(r["publication_date"]) >= 2018]
    fig, ax = plt.subplots(figsize=(5.8, 2.75))
    ax.scatter([p[0] for p in lang], [p[1] for p in lang], s=10, color="#B8B8B8", lw=0)
    ax.scatter([p[0] for p in rob], [p[1] for p in rob], s=34, color=sp.C["data"], lw=0, zorder=3)
    ax.text(2018.2, 3e25, "language models", color="#7A7A7A", fontsize=12, fontweight="bold")
    ax.text(2018.2, 2e17, "robotics models", bbox=dict(fc="white", ec="none", pad=1, alpha=0.85), color=sp.C["data"], fontsize=12, fontweight="bold")
    if step >= 2:
        for name, lab, side in (("OpenVLA", "OpenVLA", "left"), ("GR00T N1 2B", "GR00T N1", "right")):
            r = next(x for x in rob if x[2] == name)
            ymax = max(p[1] for p in lang if int(p[0]) == int(r[0]))
            gap = ymax / r[1]
            ax.plot([r[0], r[0]], [r[1], ymax], color=RED, lw=1.8, ls="--", zorder=2)
            ax.scatter([r[0]], [r[1]], s=70, color=sp.C["data"], ec=TEXT, lw=1.2, zorder=4)
            dx = -0.15 if side == "left" else 0.15
            ha = "right" if side == "left" else "left"
            ax.text(r[0] + dx, r[1] / 25, lab, color=TEXT, fontsize=11, ha=ha, fontweight="bold",
                    bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
            ax.text(r[0] + dx, math.sqrt(r[1] * ymax), f"{round(gap, -1 if gap < 1000 else -2):,.0f}×",
                    color=RED, fontsize=12, fontweight="bold", va="center", ha=ha,
                    bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
    ax.set_yscale("log")
    ax.set_ylim(1e16, 1e27)
    ax.set_xlim(2018, 2026.6)
    ax.set_yticks([1e16, 1e19, 1e22, 1e25])
    ax.set_yticklabels(["10¹⁶", "10¹⁹", "10²²", "10²⁵"])
    ax.set_xticks([2018, 2020, 2022, 2024, 2026])
    ax.set_ylabel("Training compute (FLOP)")
    ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
    sp.save(fig, out(f"epoch_compute_step{step}.svg"))


# ---------------------------------------------------------------------------
# Scope test Venn (sec. 1.6, figure 6)
# ---------------------------------------------------------------------------
def venn():
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    R = 1.0
    cs = [((0, 0.62), "Learned model", NAVY, (0, 1.78)),
          ((-0.62, -0.42), "Physical\nfeedback", PLANT, (-1.95, -0.95)),
          ((0.62, -0.42), "Delegated\nauthority", NERVE, (1.95, -0.95))]
    for (cx, cy), lab, col, (tx, ty) in cs:
        ax.add_patch(Circle((cx, cy), R, fc=col, alpha=0.12, ec=col, lw=2))
        ax.text(tx, ty, lab, color=col, fontsize=13, ha="center", va="center", fontweight="bold")
    ax.text(-0.62, 0.30, "advisory", fontsize=11, ha="center", color=TEXT)
    ax.text(0.62, 0.30, "digital\nautonomy", fontsize=11, ha="center", color=TEXT)
    ax.text(0, -0.92, "classical\ncontrol", fontsize=11, ha="center", color=TEXT)
    ax.text(0, -0.08, "Physical\nAI", fontsize=13, ha="center", va="center", color=RED,
            fontweight="bold")
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-1.55, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    sp.save(fig, out("scope_venn.svg"))


# ---------------------------------------------------------------------------
# Command-to-motion chain (sec. 1.4.1), four steps
# ---------------------------------------------------------------------------
def chain(step):
    boxes = [("Setpoint", "register write", NERVE),
             ("Current", "in a winding", AMBER),
             ("Torque", "on the shaft", PLANT),
             ("Mass", "accelerates", RED)]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    W, H, gap = 1.0, 0.62, 0.22
    for i, (a, b, c) in enumerate(boxes):
        y = 3 - i * (H + gap)
        shown = i < step
        ax.add_patch(FancyBboxPatch((0, y), 2.4, H, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="white", ec=c, lw=2.2, alpha=1 if shown else 0))
        ax.text(0.15, y + H / 2, a, color=c, fontsize=15, va="center", fontweight="bold",
                alpha=1 if shown else 0)
        ax.text(1.25, y + H / 2, b, color=TEXT, fontsize=13, va="center", alpha=1 if shown else 0)
        if 0 < i < step:
            ax.annotate("", xy=(1.2, y + H + 0.01), xytext=(1.2, y + H + gap - 0.01),
                        arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.8))
    if step >= 4:
        yb = 3 - (H + gap) + H + gap / 2
        ax.plot([-0.15, 2.55], [yb, yb], color=RED, lw=2, ls="--")
        ax.text(2.65, yb, "causal boundary", color=RED, fontsize=13, va="center",
                fontweight="bold")
        ax.text(2.65, 3.31, "can be rewritten", color=NERVE, fontsize=12, va="center")
        ax.text(2.65, 1.2, "cannot be recalled", color=RED, fontsize=12, va="center")
    ax.set_xlim(-0.2, 4.45)
    ax.set_ylim(0.4, 3.7)
    ax.axis("off")
    sp.save(fig, out(f"chain_step{step}.svg"))


# ---------------------------------------------------------------------------
# The stopping-budget bar, two segments (Napkin Math 1.1)
# ---------------------------------------------------------------------------
def stop_bar(step):
    d1, d2 = calcs.stop_mm(calcs.V_DRIVE)
    clear = calcs.D_CLEAR * 1000
    fig, ax = plt.subplots(figsize=(5.8, 1.7))
    ax.barh(0, d1, color=AMBER, height=0.5)
    ax.text(d1 / 2, 0, f"{d1:.1f}", color="white", fontsize=13, ha="center", va="center",
            fontweight="bold")
    ax.text(d1 / 2, 0.33, "delay  v·τ", color=AMBER, fontsize=12, ha="center", va="bottom",
            fontweight="bold")
    a2 = 1 if step >= 2 else 0
    ax.barh(0, d2, left=d1, color=PLANT, height=0.5, alpha=a2)
    ax.text(d1 + d2 / 2, 0, f"{d2:.1f}", color="white", fontsize=13, ha="center", va="center",
            fontweight="bold", alpha=a2)
    ax.text(d1 + d2 / 2, 0.33, "braking  v²/2a", color=PLANT, fontsize=12, ha="center",
            va="bottom", fontweight="bold", alpha=a2)
    a3 = 1 if step >= 3 else 0
    ax.axvline(clear, color=RED, lw=2, ls="--", alpha=a3)
    ax.text(clear - 12, 0.33, "clear 1,100", color=RED, fontsize=12, ha="right", va="bottom",
            fontweight="bold", alpha=a3)
    ax.text((d1 + d2 + clear) / 2, 0, f"spare\n{clear - d1 - d2:.1f}", color=RED, fontsize=11,
            ha="center", va="center", fontweight="bold", alpha=a3)
    ax.set_xlim(0, 1250)
    ax.set_ylim(-0.4, 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("mm traveled from detection, 1.5 m/s")
    sp.save(fig, out(f"stop_bar_step{step}.svg"))


# ---------------------------------------------------------------------------
# Delay travel against braking travel over speed (sec. 1.4.3, figure 4)
# ---------------------------------------------------------------------------
def delay_vs_speed():
    import numpy as np
    tau = calcs.tau_delay_ms() / 1000
    v = np.linspace(0, 2, 200)
    dly, brk = v * tau * 1000, v ** 2 / (2 * calcs.A_BRAKE) * 1000
    vstar = 2 * calcs.A_BRAKE * tau
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    ax.plot(v, dly, color=AMBER, lw=2.5)
    ax.plot(v, brk, color=PLANT, lw=2.5)
    ax.plot(v, dly + brk, color=TEXT, lw=3)
    ax.axhline(calcs.D_CLEAR * 1000, color=RED, lw=2, ls="--")
    ax.text(0.05, calcs.D_CLEAR * 1000 + 25, "clear distance 1,100 mm", color=RED, fontsize=12,
            fontweight="bold")
    ax.text(1.98, 2 * tau * 1000 + 20, "delay", color=AMBER, fontsize=12, ha="right",
            fontweight="bold", va="bottom")
    ax.text(1.6, 1.6 ** 2 / 4 * 1000 - 120, "braking", color=PLANT, fontsize=12,
            fontweight="bold")
    ax.text(1.08, 600, "stop", color=TEXT, fontsize=12, fontweight="bold", ha="right")
    ys = vstar * tau * 1000
    ax.plot([vstar], [ys], "o", color=RED, ms=8)
    ax.annotate(f"crossover {vstar:.2f} m/s", (vstar, ys), xytext=(vstar - 0.05, 420),
                fontsize=12, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.2), ha="center")
    d1, d2 = calcs.stop_mm(calcs.V_DRIVE)
    ax.plot([calcs.V_DRIVE], [d1 + d2], "o", color=TEXT, ms=9)
    ax.text(calcs.V_DRIVE - 0.06, d1 + d2 + 110, f"{d1 + d2:.1f} mm at 1.5 m/s", fontsize=12,
            color=TEXT, fontweight="bold", ha="right")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1400)
    ax.set_xlabel("Speed at detection (m/s)")
    ax.set_ylabel("Travel (mm)")
    ax.grid(color=sp.C["grid"], lw=0.6)
    sp.save(fig, out("delay_vs_speed.svg"))


# ---------------------------------------------------------------------------
# Pre-brake delay composition (sec. 1.4.3, margin bar)
# ---------------------------------------------------------------------------
def prebrake_bar():
    s = dict(calcs.AGE_STAGES)
    age = calcs.obs_age()
    segs = [("", s["Backbone"], sp.C["compute"]),
            ("rest of frame age", age - s["Backbone"], GRAY),
            ("chunk lease", calcs.LEASE, NERVE),
            ("", calcs.TICK + calcs.BUS, PLANT),
            ("brake\nonset", calcs.BRAKE_ONSET, PLANT)]
    tau = calcs.tau_delay_ms()
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    x = 0
    for name, v, c in segs:
        ax.barh(0, v, left=x, color=c, height=0.5, edgecolor="white", lw=1.5)
        if v >= 10:
            ax.text(x + v / 2, 0, f"{v:g}", color="white", fontsize=13, ha="center",
                    va="center", fontweight="bold")
            ax.text(x + v / 2, 0.32, name, color=c if c != GRAY else PLANT, fontsize=11,
                    ha="center", va="bottom", fontweight="bold")
        x += v
    ax.text(s["Backbone"] / 2, -0.33, f"backbone\n{calcs.V_DRIVE * s['Backbone']:.0f} mm", fontsize=12,
            ha="center", va="top", color=sp.C["compute"], fontweight="bold")
    lease_mid = age + calcs.LEASE / 2
    ax.text(lease_mid, -0.33, f"lease: {calcs.V_DRIVE * calcs.LEASE:.0f} mm", fontsize=12,
            ha="center", va="top", color=NERVE, fontweight="bold")
    ax.text(tau + 2, 0, f"{tau:.1f} ms", fontsize=13, va="center", fontweight="bold", color=TEXT)
    ax.set_xlim(0, tau * 1.17)
    ax.set_ylim(-1.0, 0.95)
    ax.set_yticks([])
    ax.set_xticks([])
    for sname in ("left", "bottom"):
        ax.spines[sname].set_visible(False)
    sp.save(fig, out("prebrake_bar.svg"))


# ---------------------------------------------------------------------------
# California DMV filings, Waymo (sec. 14.1, figure 1, panel b)
# ---------------------------------------------------------------------------
def dmv():
    from mlsysim import Literature
    rows = [r for r in Literature.CaliforniaDisengagements.entries
            if r["company"] == "Waymo" and r["filer_type"] in ("annual", "first_report")]
    sd = sorted((r["report_year"], r["miles_per_disengagement"]) for r in rows
                if r["permit_type"] == "drivered_AVT" and r["miles_per_disengagement"])
    dl = sorted((r["report_year"], r["miles_per_disengagement"]) for r in rows
                if r["permit_type"] == "driverless_AVDT" and r["miles_per_disengagement"])
    last = next(r for r in rows if r["report_year"] == 2025 and r["permit_type"] == "drivered_AVT")
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot([p[0] for p in sd], [p[1] for p in sd], "o-", color=sp.C["memory"], lw=2.5, ms=6)
    ax.plot([p[0] for p in dl], [p[1] for p in dl], "s-", color=sp.C["data"], lw=2.5, ms=7)
    ax.text(sd[0][0] + 0.2, sd[0][1], f"{sd[0][1]:,.0f}", color=sp.C["memory"], fontsize=12,
            va="center", fontweight="bold")
    ax.text(2025.2, sd[-1][1], f"{sd[-1][1]:,.0f}\n{last['disengagements']:.0f} takeovers",
            color=sp.C["memory"], fontsize=12, va="center", fontweight="bold")
    ax.text(2025.2, dl[-1][1], f"{dl[-1][1]:,.0f}", color=sp.C["data"], fontsize=12,
            va="center", fontweight="bold")
    ax.text(2017.3, 2400, "with a safety driver", color=sp.C["memory"], fontsize=12,
            fontweight="bold")
    ax.text(2020.6, 85000, "driverless testing", color=sp.C["data"], fontsize=12,
            fontweight="bold", ha="right")
    ax.set_yscale("log")
    ax.set_ylim(700, 2.5e5)
    ax.set_yticks([1e3, 1e4, 1e5])
    ax.set_yticklabels(["1,000", "10,000", "100,000"])
    ax.minorticks_off()
    ax.set_xlim(2014.6, 2027.6)
    ax.set_xticks([2015, 2017, 2019, 2021, 2023, 2025])
    ax.set_ylabel("Miles per disengagement")
    ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
    sp.save(fig, out("dmv_waymo.svg"))


# ---------------------------------------------------------------------------
# Clean runs needed at 95 percent confidence (sec. 7.6 figure)
# ---------------------------------------------------------------------------
def clean_runs():
    alpha = 0.05
    lo20 = alpha ** (1 / 20)
    rows = [(f"{100 * lo20:.1f}%", 20, "neutral", "20 runs")]
    for p, ck in ((0.90, "memory"), (0.99, "memory"), (0.999, "danger")):
        n = math.ceil(math.log(alpha) / math.log(p))
        rows.append((f"{100 * p:g}%", n, ck, f"{n:,} runs"))
    sp.hbars_steps(out("clean_runs"), rows, "runs", log=True,
                   xlabel="Consecutive clean runs (log scale)", size=(5.8, 2.3))


# ---------------------------------------------------------------------------
# RAND failure-free miles against Waymo rider-only miles (sec. 7.9.1, figure 5)
# ---------------------------------------------------------------------------
def rand_miles():
    rows = [("Fatalities", 275e6, sp.C["danger"], "275 million"),
            ("Injuries", 3.9e6, sp.C["neutral"], "3.9 million"),
            ("Crashes", 1.6e6, sp.C["neutral"], "1.6 million")]
    waymo = 56.7e6
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    for i, (lab, v, c, txt) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v, color=c, height=0.6)
        ax.text(v * 1.15, y, txt, color=c, fontsize=13, va="center", fontweight="bold")
        ax.text(5e5, y, lab + "  ", color=TEXT, fontsize=13, va="center", ha="right",
                fontweight="bold")
    ax.axvline(waymo, color=sp.C["data"], lw=2, ls="--")
    ax.text(waymo / 1.4, 2.62, "Waymo rider-only miles: 56.7 million", color=sp.C["data"], ha="right",
            fontsize=12, fontweight="bold", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(5e5, 3e9)
    ax.set_ylim(-0.5, 3.0)
    ax.set_yticks([])
    ax.set_xticks([1e6, 1e7, 1e8, 1e9])
    ax.set_xticklabels(["1 million", "10 million", "100 million", "1 billion"], fontsize=12)
    ax.minorticks_off()
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Failure-free miles to match human rates, 95% confidence", fontsize=12)
    ax.grid(axis="x", color=sp.C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out("rand_miles.svg"))


# ---------------------------------------------------------------------------
# The record chain (sec. 1.11, figure 9): one record per lecture
# ---------------------------------------------------------------------------
def record_chain():
    rows = [
        ("Part I", "irreversibility", PLANT, [(2, "limit"), (4, "handoff")]),
        ("Part II", "endogenous data", sp.C["data"], [(5, "provenance"), (6, "policy manifest"),
                                                      (7, "evaluation")]),
        ("Part III", "proposal is not", NERVE, [(8, "observation contract"), (9, "belief"),
                                                (10, "intent lease")]),
        ("", "permission", NERVE, [(11, "trajectory"), (12, "enforcement"),
                                    (13, "placement")]),
        ("Part IV", "permission must", NAVY, [(14, "authority"), (15, "fault"),
                                              (16, "release manifest")]),
        ("", "be earned", NAVY, [(17, "approval")]),
        ("Close", "", RED, [(18, "residual-claims register")]),
    ]
    fig = plt.figure(figsize=(5.8, 2.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 5.8)
    ax.set_ylim(-2.55, 0.2)
    ax.axis("off")
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def width(t):
        bb = t.get_window_extent(renderer=rend)
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x1 - x0

    y = 0
    for part, law, c, recs in rows:
        ax.text(0.05, y, part, color=c, fontsize=11, fontweight="bold", va="center")
        ax.text(0.78, y, law, color=c, fontsize=10, va="center")
        x = 2.0
        for lec, name in recs:
            t1 = ax.text(x + 0.06, y, f"L{lec}", color=c, fontsize=10, va="center", fontweight="bold")
            w1 = width(t1)
            t2 = ax.text(x + 0.06 + w1 + 0.06, y, name, color=TEXT, fontsize=10, va="center")
            w = w1 + width(t2) + 0.18
            ax.add_patch(FancyBboxPatch((x, y - 0.14), w, 0.28, boxstyle="round,pad=0,rounding_size=0.04",
                                        fc="white", ec=c, lw=1.5, clip_on=False))
            x += w + 0.07
        y -= 0.38
    sp.save(fig, out("record_chain.svg"))


# ---------------------------------------------------------------------------
# A text-free stack thumbnail (the kit stack, drawn here as shapes only)
# ---------------------------------------------------------------------------
def thumb_stack():
    fig, ax = plt.subplots(figsize=(2.0, 1.4))
    ys = [0, 1, 2, 3, 4.5]
    for y in ys:
        ax.add_patch(FancyBboxPatch((0.3, y), 4.4, 0.8, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc="#D7E3DE", ec="none"))
    ax.plot([0.3, 4.7], [4.15, 4.15], color=GRAY, lw=1.5, ls="--")
    ax.add_patch(FancyBboxPatch((0.05, -0.25), 4.9, 5.9, boxstyle="round,pad=0,rounding_size=0.2",
                                fc="none", ec="#1A4D3E", lw=1.5, ls="--"))
    ax.set_xlim(-0.1, 5.1)
    ax.set_ylim(-0.4, 5.8)
    ax.axis("off")
    sp.save(fig, out("thumb_stack.svg"))


def main():
    for k in (1, 2, 3):
        aisle(k)
    frame_age()
    for k in (1, 2, 3, 4):
        learned_band(k)
    for k in (1, 2):
        epoch(k)
    venn()
    for k in (1, 2, 3, 4):
        chain(k)
    for k in (1, 2, 3):
        stop_bar(k)
    delay_vs_speed()
    prebrake_bar()
    dmv()
    clean_runs()
    rand_miles()
    record_chain()
    thumb_stack()


if __name__ == "__main__":
    main()
