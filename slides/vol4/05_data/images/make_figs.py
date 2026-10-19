#!/usr/bin/env python3
"""Figures for Lecture 5 (Physical Data), Volume IV.

Data plots use slideplot (book semantic colors); schematics are hand-written
SVG in the Volume IV causal-boundary palette (proposal blue, nervous-system
green, plant slate, hazard red, sensing petrol). Every derived value comes
from calcs.py. Sources:
  - LeRobot Hub counts and episode histogram, Open X-Embodiment composition:
    MLSysIM Literature registry (Hugging Face Hub API, 2026-10-12; Open
    X-Embodiment overview sheet), as in the chapter's figures.
  - Pairing lag, teleoperation loop, yield, wear, bytes, contact grid,
    scenario table, takeover timeline, normalization: the chapter (Napkin
    Math 5.1 to 5.3, Sec. 5.6 to 5.9), illustrative values labeled so.
  - Autonomous success with intervention data: Hoque et al. 2022
    (ThriftyDAgger), Table 3, physical cable routing.
Run from this folder: python3 make_figs.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

import calcs  # noqa: E402

plt = sp.plt
C = sp.C
OUT = Path(__file__).resolve().parent
FONT = "Helvetica, Arial, sans-serif"

# Volume IV causal-boundary palette (rules/volumes/vol4/palette.md)
NAVY, BLUE, GREEN, RED, PETROL, SLATE, AMBER = (
    "#1F407A", "#2563EB", "#059669", "#DC2626", "#0F766E", "#334155", "#D97706")
GREY = "#94A3B8"


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------
class Svg:
    def __init__(self, w=900, h=500):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill="white", stroke=SLATE, sw=2, rx=6, dash=None, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d} opacity="{op}"/>')

    def text(self, x, y, s, size=22, fill=SLATE, anchor="middle", weight="normal", op=1.0, italic=False):
        st = ' font-style="italic"' if italic else ""
        self.items.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
                          f'text-anchor="{anchor}" font-weight="{weight}" opacity="{op}"{st}>{s}</text>')

    def line(self, x1, y1, x2, y2, stroke=SLATE, sw=2.5, dash=None, arrow=False, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{stroke[1:]})"' if arrow else ""
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
                          f'stroke-width="{sw}"{d}{m} opacity="{op}"/>')

    def path(self, d, stroke=SLATE, sw=2.5, fill="none", dash=None, arrow=False, op=1.0):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{stroke[1:]})"' if arrow else ""
        self.items.append(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{da}{m} '
                          f'opacity="{op}"/>')

    def circle(self, x, y, r, fill=SLATE, stroke="none", op=1.0):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" opacity="{op}"/>')

    def save(self, name):
        cols = {NAVY, BLUE, GREEN, RED, PETROL, SLATE, AMBER, GREY}
        marks = "".join(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in cols)
        body = "\n".join(self.items)
        (OUT / name).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}">\n<defs>{marks}</defs>\n'
            f'<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')


def alpha(shown):
    return 1.0 if shown else 0.0


# ---------------------------------------------------------------------------
# 1. Open loop against closed loop (Recovery)
# ---------------------------------------------------------------------------
def loops():
    for k in (1, 2):
        s = Svg(900, 360)
        s.text(200, 32, "Image classifier", 28, NAVY, weight="bold")
        for i, y in enumerate((60, 125, 190)):
            s.rect(15, y, 120, 50, "#F8FAFC", SLATE, rx=4)
            s.text(75, y + 33, f"image {i + 1}", 23)
            s.line(135, y + 25, 190, y + 25, NAVY, arrow=True)
            s.rect(192, y, 105, 50, "#EFF6FF", NAVY)
            s.text(244, y + 33, "model", 23, NAVY)
            s.line(297, y + 25, 340, y + 25, NAVY, arrow=True)
            s.text(345, y + 33, "label", 23, NAVY, anchor="start")
        s.text(200, 290, "A wrong label does not", 25)
        s.text(200, 322, "change the next image.", 25)
        o = alpha(k >= 2)
        s.line(440, 10, 440, 345, GREY, 2, dash="6,5")
        s.text(680, 32, "Robot policy", 28, GREEN, weight="bold", op=o)
        s.rect(475, 70, 140, 64, "#EFF6FF", BLUE, dash="6,4", op=o)
        s.text(545, 110, "policy", 25, BLUE, op=o)
        s.rect(725, 70, 160, 64, "#F8FAFC", SLATE, rx=0, op=o)
        s.text(805, 110, "body + world", 23, SLATE, op=o)
        s.line(615, 102, 721, 102, BLUE, arrow=True, op=o)
        s.text(668, 90, "action", 22, BLUE, op=o)
        s.path("M805,134 C805,225 545,225 545,140", PETROL, arrow=True, op=o)
        s.text(675, 240, "next state and image", 23, PETROL, op=o)
        s.text(680, 290, "An action chooses the", 25, op=o)
        s.text(680, 322, "states the robot sees next.", 25, op=o)
        s.save(f"loops_step{k}.svg")


# ---------------------------------------------------------------------------
# 2. The latch corridor, top view (Recovery)
# ---------------------------------------------------------------------------
def corridor():
    for k in (1, 2):
        s = Svg(900, 370)
        s.rect(40, 20, 680, 70, "#E2E8F0", "none", 0, rx=0)
        s.text(380, 64, "fixture", 26, SLATE)
        s.rect(40, 175, 680, 90, "#DCFCE7", GREEN, 2, rx=0)
        for x in (80, 210, 340, 470, 600):
            s.line(x, 220, x + 90, 220, GREEN, 3, arrow=True)
        s.rect(740, 135, 40, 170, "#F8FAFC", SLATE, 3, rx=0)
        s.text(830, 205, "strike", 25, anchor="middle")
        s.text(830, 235, "plate", 25, anchor="middle")
        s.text(380, 305, "every retained demonstration goes forward", 26, GREEN)
        s.text(380, 338, "along the corridor", 26, GREEN)
        o = alpha(k >= 2)
        s.rect(330, 108, 40, 40, "#EFF6FF", BLUE, 3, rx=2, op=o)
        s.text(318, 138, "gripper, 3 mm off", 25, BLUE, anchor="end", op=o)
        s.line(350, 150, 350, 185, NAVY, 3, dash="6,4", arrow=True, op=o)
        s.text(385, 160, "return action: no data", 25, NAVY, anchor="start", weight="bold", op=o)
        s.line(372, 112, 430, 96, RED, 3, dash="6,4", arrow=True, op=o)
        s.text(445, 112, "drift into contact", 25, RED, anchor="start", weight="bold", op=o)
        s.save(f"corridor_step{k}.svg")


# ---------------------------------------------------------------------------
# 3. Bounds against horizon (Recovery; step 3 in Interventions)
# ---------------------------------------------------------------------------
def bounds():
    S = np.logspace(1, np.log10(3000), 200)
    e = calcs.EPS
    s0 = calcs.S_STEPS
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.75))
        ax.plot(S, S, color=C["neutral"], lw=1.5, ls="--")
        ax.text(1500, 900, "episode\nmaximum", color=C["neutral"], fontsize=11, ha="center", va="top")
        ax.plot(S, S * e, color=C["memory"], lw=3)
        ax.text(2200, 2200 * e * 0.45, "open loop\n$S\\varepsilon$", color=C["memory"], fontsize=12,
                fontweight="bold", ha="center", va="top")
        a2 = alpha(k >= 2)
        ax.plot(S, np.minimum(S, S ** 2 * e), color=C["danger"], lw=3, alpha=a2)
        ax.text(60, 0.9, "cloning $S^2\\varepsilon$", color=C["danger"], fontsize=12.5, fontweight="bold",
                ha="left", va="top", alpha=a2)
        ax.plot([s0], [s0 ** 2 * e], "o", ms=9, color=C["danger"], alpha=a2)
        ax.text(s0 * 1.12, s0 ** 2 * e * 0.62, f"{s0 ** 2 * e:g}", color=C["danger"], fontsize=13,
                fontweight="bold", alpha=a2)
        ax.plot([s0], [s0 * e], "o", ms=9, color=C["memory"])
        ax.text(s0 * 1.12, s0 * e * 0.4, f"{s0 * e:g}", color=C["memory"], fontsize=13, fontweight="bold")
        a3 = alpha(k >= 3)
        ax.plot(S, calcs.U_REC * S * e * 1.0, color=C["data"], lw=3, ls=(0, (6, 3)), alpha=a3)
        ax.text(12, 0.0042, "aggregation, $u=1$: $uS\\varepsilon$", color=C["data"],
                fontsize=12, fontweight="bold", alpha=a3)
        ax.axvline(s0, color=C["grid"], lw=1, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10, 3000)
        ax.set_ylim(2e-3, 5e3)
        ax.minorticks_off()
        ax.set_xticks([10, 100, 1000])
        ax.set_xticklabels(["10", "100", "1,000"])
        ax.set_yticks([0.01, 1, 100])
        ax.set_yticklabels(["0.01", "1", "100"])
        ax.set_xlabel("Horizon $S$ (control steps)")
        ax.set_ylabel("Worst-case excess cost")
        ax.text(0.02, 0.97, r"$\varepsilon = 0.001$ per step", transform=ax.transAxes, ha="left", va="top",
                fontsize=11, color=C["text"])
        sp.save(fig, OUT / f"bound_step{k}.svg")


# ---------------------------------------------------------------------------
# 4. LeRobot Hub (Recovery: wide, not deep)
# ---------------------------------------------------------------------------
def lerobot():
    months, pooled, n = calcs.lerobot()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw=dict(width_ratios=[1, 1.2]))
    x = np.arange(len(months))
    y = [m["cumulative_datasets"] for m in months]
    a1.plot(x, y, color=C["data"], lw=3)
    a1.set_yscale("log")
    a1.set_ylim(3, 3e5)
    a2.set_ylim(0, 12.5)
    ticks = [i for i, m in enumerate(months) if m["month"].endswith("-01")]
    a1.set_xticks(ticks)
    a1.set_xticklabels([months[i]["month"][:4] for i in ticks])
    a1.set_yticks([10, 1e3, 1e5])
    a1.set_yticklabels(["10", "1,000", "100,000"])
    a1.set_ylabel("Datasets on the Hub")
    for key, lab in (("2024-12", "1,085"), ("2026-10", "80,132")):
        i = [m["month"] for m in months].index(key)
        a1.plot([i], [y[i]], "o", color=C["data"], ms=7)
        a1.text(i - 0.5, y[i] * 2.2, lab, color=C["data"], fontsize=12, fontweight="bold",
                ha="right" if key == "2026-10" else "center")
    los = sorted(pooled)
    vals = [pooled[lo] / 1000 for lo in los]
    cols = [C["danger"] if lo < 16 else C["memory"] for lo in los]
    a2.bar(range(len(los)), vals, color=cols, width=0.8)
    lab = ["0" if lo == 0 else f"{int(lo):,}" for lo in los]
    show = {0, 4, 9, 13, 17}
    a2.set_xticks([i for i in range(len(los)) if i in show])
    a2.set_xticklabels([lab[i] for i in range(len(los)) if i in show])
    a2.set_xlabel("Episodes per dataset (bin start)")
    a2.set_ylabel("Datasets (thousands)")
    a2.text(0, 12.2, "58% hold fewer than 16", color=C["danger"], fontsize=12,
            fontweight="bold", ha="left", va="top")
    a2.text(9.5, 4.0, "2% hold\n512 or more", color=C["memory"], fontsize=12,
            fontweight="bold", ha="left")
    fig.tight_layout(w_pad=1.5)
    sp.save(fig, OUT / "lerobot_hub.svg")


# ---------------------------------------------------------------------------
# 5. Four action taps (Labels)
# ---------------------------------------------------------------------------
def taps():
    for k in range(1, 5):
        s = Svg(900, 390)
        s.rect(20, 10, 410, 200, "#F8FAFC", GREY, 1.5, dash="6,4")
        s.text(225, 42, "application processor (proposes)", 21, NAVY, weight="bold")
        s.rect(470, 10, 410, 200, "#F0FDF4", GREY, 1.5, dash="6,4")
        s.text(675, 42, "safety MCU, 1 kHz (may command)", 21, GREEN, weight="bold")
        s.line(450, 0, 450, 360, SLATE, 2.5, dash="8,5")
        s.text(450, 384, "proposal boundary", 20, SLATE)
        s.rect(35, 80, 155, 74, "#EFF6FF", BLUE, dash="6,4")
        s.text(112, 112, "operator", 21, BLUE)
        s.text(112, 138, "or policy", 21, BLUE)
        s.rect(250, 80, 155, 74, "#EFF6FF", NAVY)
        s.text(327, 112, "kinematic", 21, NAVY)
        s.text(327, 138, "mapping", 21, NAVY)
        s.line(190, 117, 246, 117, BLUE, arrow=True)
        s.line(405, 117, 496, 117, NAVY, arrow=True)
        s.rect(500, 80, 155, 74, "#DCFCE7", GREEN)
        s.text(577, 112, "permission", 21, GREEN)
        s.text(577, 138, "check, clamp", 21, GREEN)
        s.rect(712, 80, 155, 74, "#FFFFFF", SLATE, rx=0)
        s.text(789, 112, "drives and", 21, SLATE)
        s.text(789, 138, "plant", 21, SLATE)
        s.line(655, 117, 708, 117, GREEN, arrow=True)
        taps = [(218, "1 Asked", "request", BLUE), (430, "2 Mapped", "setpoint", NAVY),
                (680, "3 Allowed", "enforced", GREEN), (830, "4 Done", "measured", PETROL)]
        for i, (x, a, b, col) in enumerate(taps):
            o = alpha(k >= i + 1)
            yy = 275
            if i == 3:
                s.line(789, 154, 789, 185, col, 2.5, op=o)
                s.line(789, 185, x, 185, col, 2.5, op=o)
                s.line(x, 185, x, yy - 34, col, 2.5, dash="4,4", arrow=True, op=o)
            else:
                s.circle(x, 117, 7, col, op=o)
                s.line(x, 124, x, yy - 34, col, 2.5, dash="4,4", arrow=True, op=o)
            bx = min(max(x - 65, 20), 762)
            s.rect(bx, yy - 30, 130, 68, "white", col, 2.5, op=o)
            s.text(bx + 65, yy - 3, a, 22, col, weight="bold", op=o)
            s.text(bx + 65, yy + 25, b, 20, col, op=o)
        s.text(880, 345, "one log, one hardware clock (PTP)", 20, PETROL, anchor="end", op=alpha(k >= 4))
        s.save(f"taps_step{k}.svg")


# ---------------------------------------------------------------------------
# 6. Pairing displacement against lag (Labels)
# ---------------------------------------------------------------------------
def pairing():
    lag = np.logspace(0, 2, 100)
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.75))
        a2 = alpha(k >= 2)
        ax.axvspan(calcs.SKEW_MS[0], calcs.SKEW_MS[1], color=C["neutral_l"], alpha=0.7 * a2, lw=0)
        ax.text(22, 0.045, "rolling-shutter\nskew", color=C["neutral"], fontsize=10.5, ha="center",
                alpha=a2)
        for v, name, col in ((calcs.V_TRANSIT, "0.50 m/s\ntransit", C["neutral"]),
                             (calcs.V_FAST, "0.10 m/s\nfast", C["danger"]),
                             (calcs.V_GUARD, "0.03 m/s\nguarded", C["data"])):
            ax.plot(lag, v * lag, color=col, lw=3)
            ax.plot([calcs.LAG_MS], [v * calcs.LAG_MS], "o", color=col, ms=8)
            ax.text(107, v * 100, name, color=col, fontsize=12, fontweight="bold", ha="left", va="center", clip_on=False)
        ax.axhline(calcs.CLEAR_MM, color=C["danger"], lw=1.6, ls="--")
        ax.text(1.1, calcs.CLEAR_MM * 1.25, "2 mm clearance", color=C["danger"], fontsize=11, ha="left",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.axvline(calcs.LAG_MS, color=C["text"], lw=1.2, ls=":")
        ax.text(calcs.LAG_MS * 1.06, 0.03, "35 ms", fontsize=11.5, color=C["text"])
        for v in (calcs.V_FAST, calcs.V_GUARD):
            c = calcs.CLEAR_MM / v
            ax.plot([c], [calcs.CLEAR_MM], "s", color=C["danger"], ms=8, alpha=a2)
            ax.text(c, 4.6 if c < 30 else 0.75, f"≤ {c:.3g} ms", color=C["danger"], fontsize=11.5,
                    fontweight="bold", ha="center", alpha=a2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 100)
        ax.set_ylim(0.02, 60)
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_yticks([0.1, 1, 10])
        ax.set_yticklabels(["0.1", "1", "10"])
        ax.set_xlabel("Observation-to-action lag (ms)")
        ax.set_ylabel("Pairing error (mm)")
        sp.save(fig, OUT / f"pairing_step{k}.svg")


# ---------------------------------------------------------------------------
# 7. Teleoperation loop strip (Labels)
# ---------------------------------------------------------------------------
def teleop():
    tot = calcs.teleop()
    fig, ax = plt.subplots(figsize=(5.8, 1.45))
    left = 0
    for lab, v, col in (("display", calcs.T_DISP, C["memory"]), ("human", calcs.T_HUM, C["compute"]),
                        ("cmd", calcs.T_CMD, C["network"]), ("mechanics", calcs.T_MECH, C["neutral"])):
        ax.barh(0, v, left=left, color=col, height=0.6, edgecolor="white")
        if v >= 40:
            ax.text(left + v / 2, 0, f"{lab}\n{v} ms", ha="center", va="center", color="white",
                    fontsize=12, fontweight="bold")
        else:
            ax.text(left + v / 2, -0.55, f"{lab} {v} ms", ha="center", va="top", color=col, fontsize=11)
        left += v
    ax.text(tot, 0.48, f"{tot} ms loop", ha="right", va="bottom", fontsize=13, fontweight="bold",
            color=C["text"])
    ax.set_xlim(0, tot)
    ax.set_ylim(-1.05, 0.95)
    ax.axis("off")
    sp.save(fig, OUT / "teleop_strip.svg")


# ---------------------------------------------------------------------------
# 8. Open X-Embodiment: command rate and morphology
# ---------------------------------------------------------------------------
def oxe():
    rows, morph, rate, tot = calcs.oxe()
    stated = sorted(k for k in rate if k is not None)
    labels = [f"{k:g}" for k in stated] + ["none"]
    h = [rate[k] / 1e6 for k in stated] + [rate[None] / 1e6]
    cols = [C["data"] if k == 20.0 else C["memory"] for k in stated] + [C["neutral"]]
    fig, ax = plt.subplots(figsize=(6.2, 2.55))
    ax.bar(range(len(h)), h, color=cols, width=0.75)
    ax.set_xticks(range(len(h)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Declared command rate (Hz)")
    ax.set_ylabel("Episodes (millions)")
    i10 = stated.index(10.0)
    ax.text(i10, h[i10] + 0.03, "57%", ha="center", va="bottom", color=C["memory"], fontsize=12.5,
            fontweight="bold")
    ax.text(len(h) - 1, h[-1] + 0.03, "28%", ha="center", va="bottom", color=C["neutral"], fontsize=12.5,
            fontweight="bold")
    i20 = stated.index(20.0)
    ax.annotate("20 Hz: the running\nmachine's chunk rate", (i20, h[i20] + 0.02), (i20 - 2.6, 0.75),
                color=C["data"], fontsize=11.5, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-", color=C["data"], lw=1.5))
    ax.set_ylim(0, 1.65)
    sp.save(fig, OUT / "oxe_rate.svg")

    names = {"Single Arm": "Single arm", "Mobile Manipulator": "Mobile manipulator",
             "Wheeled Robot": "Wheeled robot", "Human": "Human", "Bi-Manual": "Bimanual",
             "Quadrupedal Robot": "Quadruped"}
    order = [k for k in sorted(morph, key=morph.get, reverse=True)]
    rws = []
    for k in order:
        name = names.get(k, "Mixed human, robot")
        ck = "data" if k == "Mobile Manipulator" else "memory"
        rws.append((name, morph[k], ck, f"{morph[k]:,.0f}"))
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    n = len(rws)
    for i, (name, v, ck, txt) in enumerate(rws):
        y = n - 1 - i
        ax.barh(y, v, color=C[ck], height=0.62)
        ax.text(v * 1.2, y, txt, va="center", fontsize=12.5, fontweight="bold", color=C[ck])
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rws][::-1], fontsize=12.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(8, 4e7)
    ax.minorticks_off()
    ax.set_xticks([10, 1000, 1e5, 1e7])
    ax.set_xticklabels(["10", "1,000", "100,000", "10 million"])
    ax.set_xlabel("Episodes (log scale)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    sp.save(fig, OUT / "oxe_morph.svg")


# ---------------------------------------------------------------------------
# 9. Yield of one scheduled hour (Coverage)
# ---------------------------------------------------------------------------
def yield_fig():
    y = calcs.yield_wear()
    st = y["stages"]
    fail = calcs.FAIL * st["attempts"]
    qa = (1 - calcs.FAIL) * calcs.QA * st["attempts"]
    kept = y["kept_min"]
    rows1 = [("startup", st["startup"], C["neutral_l"], C["text"]),
             ("calib.", st["calibration"], C["neutral_l"], C["text"]),
             ("rest", st["rest"], C["neutral_l"], C["text"]),
             ("resets", st["resets"], C["neutral"], "white"),
             ("attempts", st["attempts"], C["compute"], "white")]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.1))
        left = 0
        for lab, v, col, tc in rows1:
            ax.barh(1, v, left=left, color=col, height=0.55, edgecolor="white", lw=1.5)
            ax.text(left + v / 2, 1, f"{lab}\n{v:.0f}" if v >= 9 else f"{lab}\n{v:.0f}", ha="center",
                    va="center", fontsize=10.5, color=tc, fontweight="bold")
            left += v
        ax.text(0, 1.42, "One scheduled hour (minutes)", fontsize=12, fontweight="bold", color=C["text"])
        a = alpha(k >= 2)
        x0 = 60 - st["attempts"]
        for lab, v, col in (("failed", fail, C["danger"]), ("QA", qa, C["neutral"]),
                            ("kept 14.73", kept, C["data"])):
            ax.barh(0, v, left=x0, color=col, height=0.55, edgecolor="white", lw=1.5, alpha=a)
            if v > 8:
                ax.text(x0 + v / 2, 0, lab, ha="center", va="center", fontsize=10.5, color="white",
                        fontweight="bold", alpha=a)
            else:
                up = lab == "QA"
                ax.text(x0 + v / 2, 0.33 if up else -0.33, lab, ha="center", va="bottom" if up else "top",
                        fontsize=10.5, color=col, fontweight="bold", alpha=a)
            x0 += v
        ax.text(60 - st["attempts"] - 0.8, 0, "of the attempts:", ha="right", va="center", fontsize=12,
                color=C["text"], alpha=a)
        ax.set_xlim(0, 60)
        ax.set_ylim(-0.75, 1.6)
        ax.axis("off")
        sp.save(fig, OUT / f"yield_step{k}.svg")


def wear():
    y = calcs.yield_wear()
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    vals = [y["cable_pct"], y["gear_pct"]]
    ax.bar([0, 1], vals, color=[C["danger"], C["memory"]], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=15, fontweight="bold",
                color=[C["danger"], C["memory"]][i])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["wrist\nflex cable", "strain-wave\ngear"], fontsize=13)
    ax.set_ylabel("Rated life used (%)", fontsize=13)
    ax.set_ylim(0, 36)
    ax.tick_params(axis="y", labelsize=12)
    sp.save(fig, OUT / "wear.svg")


def bytes_fig():
    b = calcs.bytes_rate()
    rows = [("All streams, raw", b["tot"], C["danger"], f"{b['tot']:,.0f}"),
            ("Navigation camera", b["nav"], C["danger"], f"{b['nav']:,.0f}"),
            ("Compressed 8×", b["comp"], C["data"], f"{b['comp']:,.0f}"),
            ("Four action taps", b["kin"], C["memory"], f"{b['kin']:.2f}")]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    for i, (lab, v, col, txt) in enumerate(rows):
        y = 3 - i
        ax.barh(y, v, color=col, height=0.6)
        ax.text(0.06, y, lab + "  ", ha="right", va="center", fontsize=12.5, fontweight="bold",
                color=C["text"])
        ax.text(v * 1.15 if v < 100 else v * 0.85, y, txt, ha="left" if v < 100 else "right", va="center",
                fontsize=12, fontweight="bold", color=col if v < 100 else "white")
    for x, lab, col in ((calcs.SATA, "SATA SSD\n550", C["danger"]), (calcs.NVME, "NVMe\n7,000", C["neutral"])):
        ax.axvline(x, color=col, lw=1.6, ls="--")
        ax.text(x * 1.12, 3.6, lab, color=col, fontsize=10.5, fontweight="bold", va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.06, 6e4)
    ax.minorticks_off()
    ax.set_xticks([0.1, 10, 1000])
    ax.set_xticklabels(["0.1", "10", "1,000"])
    ax.set_ylim(-0.5, 3.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Sustained write rate (MB/s)")
    sp.save(fig, OUT / "bytes.svg")


# ---------------------------------------------------------------------------
# 10. Contact-angle grids (Coverage)
# ---------------------------------------------------------------------------
def grids():
    cells, c = calcs.grid()
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.65))
    for ax, name, rms, mask in (
            (axes[0], "Cautious operator", calcs.RMS[0], "c"),
            (axes[1], "Exploratory policy", calcs.RMS[1], "e")):
        g = np.zeros((calcs.FORCE_BINS, calcs.ANG_BINS))
        if mask == "c":
            g[4:6, 13:17] = 1
            n = int(g.sum())
            tag = f"{n} of {cells} cells"
        else:
            g[1:18, :] = 1
            n = int(g.sum())
            tag = f"{n} of {cells} cells"
        ax.imshow(g, origin="lower", cmap=plt.matplotlib.colors.ListedColormap([C["neutral_l"], C["data"]]),
                  extent=(-15, 15, 0, 20), aspect="auto")
        ax.set_title(name, fontsize=12.5, fontweight="bold", color=C["text"], pad=20)
        ax.set_xlabel("Contact angle (deg)", fontsize=11)
        ax.set_xticks([-15, 0, 15])
        ax.set_yticks([0, 10, 20])
        ax.text(0, 21, f"{tag}; held-out error {rms:g} rad", ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color=C["data"] if mask == "e" else C["danger"])
    axes[0].set_ylabel("Normal force (N)", fontsize=11)
    fig.tight_layout(w_pad=2)
    sp.save(fig, OUT / "grids.svg")


# ---------------------------------------------------------------------------
# 11. Scenario-table join (Coverage)
# ---------------------------------------------------------------------------
def scenario():
    for k in (1, 2, 3):
        s = Svg(900, 370)
        x0, y0, cw, ch = 300, 50, 175, 100
        s.text(x0 - 130, y0 + ch + 6, "approach", 23, anchor="middle")
        s.text(x0 - 130, y0 + ch + 34, "speed", 23, anchor="middle")
        for j, sp_ in enumerate(("0.10 m/s", "0.03 m/s")):
            s.text(x0 - 14, y0 + j * ch + ch / 2 + 8, sp_, 23, anchor="end")
        for i, off in enumerate(("−3 mm", "0 mm", "+3 mm")):
            s.text(x0 + i * cw + cw / 2, y0 + 2 * ch + 32, off, 23)
        s.text(x0 + 1.5 * cw, y0 + 2 * ch + 64, "strike-plate offset", 23)
        for i in range(3):
            for j in range(2):
                x, y = x0 + i * cw, y0 + j * ch
                slow = j == 1
                nominal = i == 1
                if k == 1 and not (slow and nominal):
                    s.rect(x + 4, y + 4, cw - 8, ch - 8, "white", GREY, 1.5, dash="3,5")
                    s.text(x + cw / 2, y + ch / 2 + 8, "not in table", 21, GREY)
                    continue
                pop = (slow and nominal) or (k == 3 and slow and i == 2)
                if pop:
                    s.rect(x + 4, y + 4, cw - 8, ch - 8, "#DCFCE7", GREEN, 3)
                    s.text(x + cw / 2, y + ch / 2 + 8, "demonstrated", 22, GREEN, weight="bold")
                else:
                    s.rect(x + 4, y + 4, cw - 8, ch - 8, "#FEF2F2", RED, 2.5)
                    s.text(x + cw / 2, y + ch / 2 + 10, "0", 30, RED, weight="bold")
        msg = {1: "Uncritical table: 1 of 1 cell. The archive looks complete.",
               2: "The table the door needs: 1 of 6 cells; 5 known absences.",
               3: "After latch-recovery-02: 2 of 6 cells; 4 known absences."}[k]
        s.text(450, 28, msg, 25, SLATE, weight="bold")
        s.save(f"scenario_step{k}.svg")


# ---------------------------------------------------------------------------
# 12. One takeover, curated (Interventions)
# ---------------------------------------------------------------------------
def takeover():
    t = np.linspace(0, 10, 500)
    td, tt, tb, ts, tr = 2.0, 5.0, 5.6, 7.0, 9.4
    err = np.where(t < td, 0.05, 0)
    err = np.where((t >= td) & (t < tt), 0.05 + 0.95 * np.clip((t - td) / (tt - td), 0, 1) ** 1.6, err)
    err = np.where((t >= tt) & (t < tb), 1.0 + 0.03 * (t - tt), err)
    err = np.where((t >= tb) & (t < ts), 1.02 + 0.12 * np.sin((t - tb) * 9), err)
    err = np.where(t >= ts, 1.02 * np.clip(1 - (t - ts) / (tr - ts), 0, 1) ** 1.3, err)
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.9))
        ax.plot(t[t < tt], err[t < tt], color=C["danger"], lw=3)
        ax.plot(t[(t >= tt) & (t < tb)], err[(t >= tt) & (t < tb)], color=C["neutral"], lw=3, ls="--")
        ax.plot(t[(t >= tb) & (t < ts)], err[(t >= tb) & (t < ts)], color=C["neutral"], lw=2)
        ax.plot(t[t >= ts], err[t >= ts], color=C["data"], lw=3)
        for x, lab in ((td, "$t_{\\mathrm{div}}$"), (tt, "$t_{\\mathrm{take}}$"),
                       (ts, "$t_{\\mathrm{stab}}$"), (tr, "$t_{\\mathrm{rec}}$")):
            ax.axvline(x, color=C["grid"], lw=1, zorder=0)
            ax.text(x, 1.5, lab, ha="center", fontsize=12.5, color=C["text"])
        ax.annotate("", (tt - 5.0 + 0.05, 1.22), (tt - 0.05, 1.22),
                    arrowprops=dict(arrowstyle="<->", color=C["memory"], lw=1.8))
        ax.text(tt - 2.5, 1.28, "5 s lookback, frozen at takeover", ha="center", fontsize=10.5,
                color=C["memory"], fontweight="bold")
        a2, a3 = alpha(k >= 2), alpha(k >= 3)
        for x, lab, col in (((td + tt) / 2, "drift", C["danger"]), ((tt + tb) / 2 - 0.15, "blend", C["neutral"]),
                            ((tb + ts) / 2 + 0.35, "re-orient", C["neutral"]), ((ts + tr) / 2 + 0.35, "stabilized", C["data"])):
            ax.text(x, -0.28, lab, ha="center", fontsize=11, fontweight="bold", color=col, alpha=a2)
        for x, lab, col in (((td + tt) / 2, "states: relabel\ncommands: mask", C["danger"]),
                            ((tb + ts) / 2 - 0.25, "excluded", C["neutral"]),
                            ((ts + tr) / 2, "expert labels", C["data"])):
            ax.text(x, -0.55, lab, ha="center", va="top", fontsize=10.5, color=col, alpha=a3)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.05, 1.72)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["bottom"].set_position(("data", -0.12))
        ax.set_ylabel("Tracking error")
        ax.text(10, -0.3, "time", ha="right", va="top", fontsize=11, color=C["text"])
        sp.save(fig, OUT / f"takeover_step{k}.svg")


def thrifty():
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    names = list(calcs.THRIFTY)
    v = [calcs.THRIFTY[n][0] for n in names]
    cols = [C["danger"], C["memory"], C["data"]]
    ax.bar(range(3), v, color=cols, width=0.62)
    for i, n in enumerate(names):
        s_, m = calcs.THRIFTY[n]
        ax.text(i, s_ + 0.4, f"{s_} of {m}", ha="center", fontsize=14, fontweight="bold", color=cols[i])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["cloning\nonly", "human-gated\nDAgger", "Thrifty-\nDAgger"], fontsize=12.5)
    ax.set_ylabel("Autonomous successes", fontsize=12.5)
    ax.set_ylim(0, 15)
    ax.set_yticks([0, 5, 10, 15])
    ax.tick_params(axis="y", labelsize=12)
    sp.save(fig, OUT / "thrifty.svg")


# ---------------------------------------------------------------------------
# 13. Normalization fallacy (Provenance)
# ---------------------------------------------------------------------------
def normalize():
    for k in (1, 2):
        s = Svg(900, 400)
        sc = 3.4  # px per mm
        for row, (name, stroke, y) in enumerate((("Machine A", calcs.STROKE_A, 50),
                                                  ("Machine B", calcs.STROKE_B, 235))):
            o = 1.0 if row == 0 else alpha(k >= 2)
            cx = 610
            open_mm = calcs.NORM_CMD * stroke
            s.text(20, y + 12, f"{name}: {stroke} mm stroke", 26, SLATE, anchor="start", weight="bold", op=o)
            s.text(20, y + 50, "command 0.80", 26, BLUE, anchor="start", op=o)
            s.text(20, y + 86, f"opening {open_mm:.0f} mm", 26, BLUE, anchor="start", weight="bold", op=o)
            part = calcs.PART * sc
            s.rect(cx - part / 2, y - 5, part, 90, "#E2E8F0", SLATE, 2, rx=2, op=o)
            s.text(cx, y + 48, "75 mm part", 25, SLATE, op=o)
            half = open_mm * sc / 2
            col = GREEN if open_mm > calcs.PART else RED
            for sx in (-1, 1):
                fx = cx + sx * half
                s.rect(fx - (16 if sx < 0 else 0), y - 25, 16, 130, col, col, 1, rx=2, op=o)
            msg = "clears the part" if open_mm > calcs.PART else "fingers land inside the part"
            s.text(cx, y + 140, msg, 26, col, weight="bold", op=o)
        s.save(f"normalize_step{k}.svg")


if __name__ == "__main__":
    loops()
    corridor()
    bounds()
    lerobot()
    taps()
    pairing()
    teleop()
    oxe()
    yield_fig()
    wear()
    bytes_fig()
    grids()
    scenario()
    takeover()
    thrifty()
    normalize()
    print("figures written")
