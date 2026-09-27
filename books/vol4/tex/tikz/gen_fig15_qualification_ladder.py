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
ax.set_xlim(-5, 112)
ax.set_ylim(0.08, 5e6)

for name, fid, tput, color, subtitle, infra in stages:
    # Marker dot
    ax.plot(fid, tput, "o", color=color, markersize=10, zorder=6,
            markeredgecolor="white", markeredgewidth=2)

    # Name label above marker
    ax.text(fid, tput * 2.5, name, fontsize=14, fontweight="bold",
            color=color, ha="center", va="bottom", zorder=5)

    # Subtitle and infra below marker — tighter spacing on log scale
    ax.text(fid, tput * 0.40, subtitle, fontsize=7.5, color=color,
            ha="center", va="top", zorder=5, style="italic")
    ax.text(fid, tput * 0.18, infra, fontsize=7, color=color,
            ha="center", va="top", zorder=5)

    # Dashed cross-hairs
    ax.plot([fid, fid], [0.08, tput], color=paleink, linewidth=0.6,
            linestyle="--", zorder=1)
    ax.plot([-5, fid], [tput, tput], color=paleink, linewidth=0.6,
            linestyle="--", zorder=1)

# Transition arrows
for label, i_from, i_to in transitions:
    _, f1, t1, _, _, _ = stages[i_from]
    _, f2, t2, _, _, _ = stages[i_to]
    mid_f = (f1 + f2) / 2 + 6
    mid_t = np.sqrt(t1 * t2)
    ax.annotate("",
                xy=(f2, t2), xytext=(f1, t1),
                arrowprops=dict(arrowstyle="-|>", color=ethdarkblue,
                               lw=1.8, connectionstyle="arc3,rad=-0.15"),
                zorder=4)
    ax.text(mid_f, mid_t, label, fontsize=7.5, fontweight="bold",
            color=ethdarkblue, ha="left", va="center")

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
