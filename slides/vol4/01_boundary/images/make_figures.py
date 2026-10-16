"""Slide figures for vol4 ch01 (The Causal Boundary): data plots.

Values are the rendered chapter's LEGO outputs (MLSysIM registry): the running
machine's stopping budget (1.5 m/s, 2 m/s^2, 1.10 m, 133.6 ms = 51.6 + 60 + 1
+ 1 + 20 ms, of which 22 ms is the perception backbone) and the Epoch AI
training-compute record (Literature.ModelsByDomain), read from mlsysim. Each
figure is sized for its slot on a 16:9 Beamer slide so that text lands at
8 pt or more. Writes SVG with text kept as text.

Usage: python3 make_figures.py <outdir>
"""
import datetime as dt
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "mlsysim"))

C = dict(blue="#006395", blueL="#D1E6F3", orange="#E67817", orangeL="#FCE4CC",
         green="#008F45", greenL="#D4EFDF", violet="#7E317B", violetL="#E6D4E5",
         red="#CB202D", redL="#F5D2D5", ink="#333333", gray="#666666",
         grid="#CCCCCC", brown="#78492A", brownL="#E3D3C8", lgray="#B8B8B8")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 16, "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 14, "axes.edgecolor": C["gray"], "axes.labelcolor": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"], "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
COL = (5.4, 3.6)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# Running machine (registry values printed by the chapter) -------------------
V, A, D_CLEAR = 1.5, 2.0, 1100.0           # m/s, m/s^2, mm
AGE, BACKBONE, LEASE, TICK, BUS, ONSET = 51.6, 22.0, 60.0, 1.0, 1.0, 20.0  # ms
TAU = (AGE + LEASE + TICK + BUS + ONSET) / 1000.0                          # 0.1336 s

# 1. Pre-brake delay composition (wide strip) --------------------------------
fig, ax = plt.subplots(figsize=(9.5, 2.5))
segs = [(BACKBONE, C["orange"], "backbone\n22"), (AGE - BACKBONE, C["lgray"], "rest of\nobs. age 29.6"),
        (LEASE, C["blue"], "chunk lease 60"), (TICK + BUS, C["green"], ""),
        (ONSET, C["brown"], "brake\nonset 20")]
x = 0
for w, col, lab in segs:
    ax.barh(0, w, left=x, color=col, edgecolor="white", height=0.62)
    if lab:
        tc = C["ink"] if col == C["lgray"] else "white"
        ax.text(x + w / 2, 0, lab, ha="center", va="center", color=tc, fontsize=15,
                fontweight="bold", linespacing=1.0)
    x += w
ax.annotate("tick 1 + bus 1", xy=(AGE + LEASE + 1, -0.33), xytext=(AGE + LEASE - 2, -0.62),
            ha="center", fontsize=15, color=C["green"], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C["green"], lw=1.5))
ax.text(TAU * 1000, 0.48, f"{TAU*1000:.1f} ms = {V*TAU*1000:.1f} mm of travel at 1.5 m/s",
        ha="right", va="bottom", fontsize=15, fontweight="bold", color=C["ink"])
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 136)
ax.set_ylim(-0.75, 0.8)
ax.set_xlabel("milliseconds from the worker's first exposure to retarding force")
save(fig, "delay-waterfall")

# 2. Delay travel vs braking travel against speed ----------------------------
v = np.linspace(0, 2.0, 401)
delay = v * TAU * 1000
brake = v ** 2 / (2 * A) * 1000
v_star = 2 * A * TAU
d_op = V * TAU * 1000 + V ** 2 / (2 * A) * 1000
fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.plot(v, delay, color=C["brown"], lw=2.6)
ax.plot(v, brake, color=C["blue"], lw=2.6)
ax.plot(v, delay + brake, color=C["ink"], lw=3.0)
ax.axhline(D_CLEAR, color=C["red"], lw=2.2, ls="--")
ax.text(0.03, D_CLEAR + 25, "clear distance 1.10 m", color=C["red"], fontsize=15,
        fontweight="bold", va="bottom")
for y, lab, col in [(2.0 * TAU * 1000, "delay", C["brown"]), (1000, "braking", C["blue"]),
                    (2.0 * TAU * 1000 + 1000, "sum", C["ink"])]:
    ax.text(2.04, y, lab, color=col, fontsize=15, fontweight="bold", va="center", clip_on=False)
ax.plot([v_star], [v_star * TAU * 1000], "o", ms=9, mfc="white", mec=C["ink"], mew=2, zorder=5)
ax.annotate(f"v* = {v_star:.2f} m/s", xy=(v_star, v_star * TAU * 1000),
            xytext=(0.04, 430), fontsize=15, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C["ink"], lw=1.2))
ax.plot([V], [d_op], "o", ms=10, color=C["ink"], zorder=5)
ax.text(V - 0.06, d_op + 40, f"{d_op:.1f} mm", fontsize=15, fontweight="bold", ha="right", va="bottom")
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 1600)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax.set_xlabel("speed at detection (m/s)")
ax.set_ylabel("distance (mm)")
save(fig, "stopping-vs-speed")

# 3. Training compute by domain (Epoch AI via MLSysIM) ------------------------
from mlsysim import Literature  # noqa: E402

rows = [r for r in Literature.ModelsByDomain.entries if str(r["publication_date"]) >= "2018"]
date = lambda r: dt.datetime.fromisoformat(str(r["publication_date"])[:10])
lang = [r for r in rows if r["domain_group"] == "Language"]
robo = [r for r in rows if r["domain_group"] == "Robotics"]
world = [r for r in robo if "Cosmos" in r["model"]]
policy = [r for r in robo if "Cosmos" not in r["model"]]
fig, ax = plt.subplots(figsize=COL)
ax.scatter([date(r) for r in lang], [r["training_flop"] for r in lang], s=14,
           color=C["gray"], alpha=0.35, lw=0, label="language models")
ax.scatter([date(r) for r in policy], [r["training_flop"] for r in policy], s=70,
           color=C["green"], zorder=4, label="robotics models")
ax.scatter([date(r) for r in world], [r["training_flop"] for r in world], s=70, marker="s",
           facecolors="white", edgecolors=C["green"], lw=2, zorder=4, label="robot video world models")
for name, (dx, dy) in {"OpenVLA": (-40, 28), "GR00T N1 2B": (18, -52)}.items():
    r = next(r for r in policy if r["model"] == name)
    ax.annotate(name.replace(" 2B", ""), (date(r), r["training_flop"]), xytext=(dx, dy),
                textcoords="offset points", ha="left" if dx > 0 else "right",
                arrowprops=dict(arrowstyle="-", color=C["green"], lw=1.5),
                fontsize=15, fontweight="bold", color=C["green"],
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
ax.set_yscale("log")
ax.set_ylim(1e12, 1e28)
ax.set_yticks([1e13, 1e16, 1e19, 1e22, 1e25, 1e28])
ax.set_ylabel("training compute (FLOP)")
ax.set_xlim(dt.datetime(2018, 1, 1), dt.datetime(2026, 9, 1))
ax.set_xticks([dt.datetime(y, 1, 1) for y in (2018, 2020, 2022, 2024, 2026)],
              ["2018", "2020", "2022", "2024", "2026"])
ax.legend(frameon=False, loc="lower right", fontsize=14, handletextpad=0.2, borderaxespad=0.1)
save(fig, "compute-by-domain")
print("ok")
