#!/usr/bin/env python3
"""Generate fig15_qualification_ladder as a matplotlib figure."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
stages = [
    ("SIL",      5,   1e6, ethbronze,      "Software-in-the-Loop",  "Cloud Cluster"),
    ("PIL",     30,   1e4, ethpetrol,      "Processor-in-the-Loop", "Target Silicon"),
    ("HIL",     70,   1e2, emeraldpine,    "Hardware-in-the-Loop",  "Dyno Testbed"),
    ("In-Situ", 95,   1e0, harvardcrimson, "Physical Fault Rig",    "Field Plant"),
]

transitions = [
    ("+ Timing &\nRegister Access", 0, 1),
    ("+ Real Physics",              1, 2),
    ("+ Environmental\nInteraction", 2, 3),
]

fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.set_yscale("log")
ax.set_xlim(-5, 114)
ax.set_ylim(0.08, 8e6)

# Dashed cross-hairs
for name, fid, tput, color, subtitle, infra in stages:
    ax.plot([fid, fid], [0.08, tput], color=paleink, linewidth=0.8,
            linestyle="--", zorder=1)
    ax.plot([-5, fid], [tput, tput], color=paleink, linewidth=0.8,
            linestyle="--", zorder=1)

# Markers and stage labels
for name, fid, tput, color, subtitle, infra in stages:
    ax.plot(fid, tput, "o", color=color, markersize=10, zorder=6,
            markeredgecolor="white", markeredgewidth=2)

    # Name label above marker (offset for In-Situ to prevent arrow strike)
    if name == "In-Situ":
        ax.text(fid + 2.5, tput * 2.5, name, fontsize=14, fontweight="bold",
                color=color, ha="left", va="bottom", zorder=5)
    else:
        ax.text(fid, tput * 2.5, name, fontsize=14, fontweight="bold",
                color=color, ha="center", va="bottom", zorder=5)

    # Subtitle and infra below marker — tailored offsets to avoid curve clipping
    if name == "SIL":
        sub_y = tput * 0.18
        inf_y = tput * 0.06
    elif name == "HIL":
        sub_y = tput * 0.28
        inf_y = tput * 0.10
    else:
        sub_y = tput * 0.35
        inf_y = tput * 0.14

    ax.text(fid, sub_y, subtitle, fontsize=8, color=color,
            ha="center", va="top", zorder=5, style="italic")
    ax.text(fid, inf_y, infra, fontsize=7.5, color=color,
            ha="center", va="top", zorder=5)

# Transition arrows and carefully positioned labels
t_data = [
    ("+ Timing &\nRegister Access", 0, 1, 19.0, 3.0e5, "left", "bottom"),
    ("+ Real Physics",              1, 2, 53.0, 4.0e3, "left", "bottom"),
    ("+ Environmental\nInteraction", 2, 3, 79.0, 9.0e1, "left", "bottom"),
]

for label, i_from, i_to, lx, ly, ha, va in t_data:
    _, f1, t1, _, _, _ = stages[i_from]
    _, f2, t2, _, _, _ = stages[i_to]
    ax.annotate("",
                xy=(f2, t2), xytext=(f1, t1),
                arrowprops=dict(arrowstyle="-|>", color=ethdarkblue,
                               lw=2.0, connectionstyle="arc3,rad=-0.15",
                               shrinkA=9, shrinkB=9),
                zorder=4)
    ax.text(lx, ly, label, fontsize=8, fontweight="bold",
            color=ethdarkblue, ha=ha, va=va, zorder=5)

# Axes
ax.set_xlabel("Hardware Fidelity (%)", fontsize=11, fontweight="bold",
              color=ink, labelpad=8)
ax.set_ylabel("Throughput (runs / day)", fontsize=11, fontweight="bold",
              color=ink, labelpad=8)

ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=ink)

ax.set_yticks([1e0, 1e2, 1e4, 1e6])
ax.set_yticklabels([r"$10^0$", r"$10^2$", r"$10^4$", r"$10^6$"],
                   fontsize=9, color=ink)
ax.yaxis.set_minor_locator(plt.NullLocator())

ax.grid(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.tight_layout()

_here = __import__("pathlib").Path(__file__).resolve().parent
_root = _here.parents[1]
(_root / "15_verification" / "images" / "png").mkdir(parents=True, exist_ok=True)
(_root / "15_verification" / "images" / "svg").mkdir(parents=True, exist_ok=True)
plt.savefig(_root / "15_verification" / "images" / "png" / "fig15_qualification_ladder.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(_root / "15_verification" / "images" / "svg" / "fig15_qualification_ladder.svg",
            bbox_inches="tight", facecolor="white")
print("OK: fig15_qualification_ladder")
