#!/usr/bin/env python3
"""Generate fig15_qualification_ladder as a matplotlib figure."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Book style ──────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["mathtext.default"] = "regular"

# ── Colors ──────────────────────────────────────────────────────────────
ethbronze      = "#B87333"
ethpetrol      = "#007A87"
emeraldpine    = "#1A4D3E"
harvardcrimson = "#A51C30"
ethdarkblue    = "#1F407A"
ink            = "#1A202C"
paleink        = "#A0AEC0"

# ── Data ────────────────────────────────────────────────────────────────
# (name, hw_fidelity_pct, throughput_runs_per_day, color, subtitle)
stages = [
    ("SIL",     5,   1e6, ethbronze,      "Software-in-the-Loop\nCloud Cluster"),
    ("PIL",    30,   1e4, ethpetrol,      "Processor-in-the-Loop\nTarget Silicon"),
    ("HIL",    70,   1e2, emeraldpine,    "Hardware-in-the-Loop\nDyno Testbed"),
    ("In-Situ", 95,  1e0, harvardcrimson, "Physical Fault Rig\nField Plant"),
]

# Transition annotations
transitions = [
    ("+ Timing &\nRegister Access", 0, 1),
    ("+ Real Physics",              1, 2),
    ("+ Environmental\nInteraction", 2, 3),
]

fig, ax = plt.subplots(figsize=(7, 6))

# Log scale for y
ax.set_yscale("log")
ax.set_xlim(-5, 105)
ax.set_ylim(0.3, 5e6)

# Plot each stage as a labeled box
box_w, box_h_log = 18, 0.6  # box half-widths in data coords

for name, fid, tput, color, subtitle in stages:
    # Box
    box = FancyBboxPatch(
        (fid - 9, tput * 0.45), 18, tput * 1.5,
        boxstyle="round,pad=0.3", facecolor=color, alpha=0.12,
        edgecolor=color, linewidth=1.8, zorder=3
    )
    ax.add_patch(box)

    # Main label
    ax.text(fid, tput * 1.6, name, fontsize=13, fontweight="bold",
            color=color, ha="center", va="bottom", zorder=5)

    # Subtitle
    ax.text(fid, tput * 0.65, subtitle, fontsize=7.5, color=color,
            ha="center", va="top", zorder=5, linespacing=1.3)

    # Marker
    ax.plot(fid, tput, "o", color=color, markersize=8, zorder=6,
            markeredgecolor="white", markeredgewidth=1.5)

    # Dashed grid lines
    ax.plot([fid, fid], [0.3, tput], color=paleink, linewidth=0.7,
            linestyle="--", zorder=1)
    ax.plot([-5, fid], [tput, tput], color=paleink, linewidth=0.7,
            linestyle="--", zorder=1)

# Transition arrows
for label, i_from, i_to in transitions:
    _, f1, t1, c1, _ = stages[i_from]
    _, f2, t2, c2, _ = stages[i_to]
    mid_f = (f1 + f2) / 2
    mid_t = np.sqrt(t1 * t2)  # geometric mean for log scale
    ax.annotate("",
                xy=(f2, t2), xytext=(f1, t1),
                arrowprops=dict(arrowstyle="-|>", color=ethdarkblue,
                               lw=1.8, connectionstyle="arc3,rad=-0.15"),
                zorder=4)
    ax.text(mid_f + 8, mid_t, label, fontsize=7.5, fontweight="bold",
            color=ethdarkblue, ha="left", va="center")

# Axes labels
ax.set_xlabel("Hardware Fidelity (%)", fontsize=11, fontweight="bold",
              color=ink, labelpad=8)
ax.set_ylabel("Throughput (runs / day)", fontsize=11, fontweight="bold",
              color=ink, labelpad=8)

# X ticks
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=ink)

# Y ticks
ax.set_yticks([1e0, 1e2, 1e4, 1e6])
ax.set_yticklabels([r"$10^0$", r"$10^2$", r"$10^4$", r"$10^6$"],
                   fontsize=9, color=ink)
ax.yaxis.set_minor_locator(plt.NullLocator())

# Grid
ax.grid(False)

# Spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.tight_layout()

# Output paths relative to this script
_here = __import__("pathlib").Path(__file__).resolve().parent
_root = _here.parents[1]  # books/vol4/
(_root / "15_verification" / "images" / "png").mkdir(parents=True, exist_ok=True)
(_root / "15_verification" / "images" / "svg").mkdir(parents=True, exist_ok=True)
plt.savefig(_root / "15_verification" / "images" / "png" / "fig15_qualification_ladder.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(_root / "15_verification" / "images" / "svg" / "fig15_qualification_ladder.svg",
            bbox_inches="tight", facecolor="white")
print("OK: fig15_qualification_ladder")
