#!/usr/bin/env python3
"""Generate fig02_canonical_latency_waterfall as a matplotlib figure."""
import matplotlib.pyplot as plt
import numpy as np

# ── Book style ──────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["mathtext.default"] = "regular"

# ── Color palette (from theme-colors.tex) ───────────────────────────────
ethpetrol      = "#007A87"
ethdarkblue    = "#1F407A"
emeraldpine    = "#1A4D3E"
harvardcrimson = "#A51C30"
ethbronze      = "#B87333"
ink            = "#1A202C"
paleink        = "#A0AEC0"

# ── Stage data ──────────────────────────────────────────────────────────
stages = [
    ("Photon to Sensor ADC\n(exposure + readout)",   0,  5, ethpetrol),
    ("Sensor ADC to CSI-2 /\nGMSL transport",        5,  2, ethpetrol),
    ("DMA transfer to\nSoC memory",                  7,  1, ethbronze),
    ("Preprocessing\n(ISP, undistort)",               8,  4, ethdarkblue),
    ("Neural model\ninference",                      12, 15, ethdarkblue),
    ("Postprocessing /\nsafety enforcer",            27,  2, emeraldpine),
    ("CAN / EtherCAT\nbus transmission",             29,  2, ethbronze),
    ("Gate driver to\nMotor flux rise",              31,  4, harvardcrimson),
]

n = len(stages)

# ── Silicon boundary groups ─────────────────────────────────────────────
groups = [
    ("Image\nSensor",    0, 1, ethpetrol),
    ("Compute\nSoC",     2, 5, ethdarkblue),
    ("Drive\nActuator",  6, 7, harvardcrimson),
]

# ── Figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))

bar_height = 0.55
y_positions = np.arange(n)[::-1]

for i, (label, start, duration, color) in enumerate(stages):
    y = y_positions[i]
    ax.barh(y, duration, left=start, height=bar_height, color=color,
            edgecolor="white", linewidth=0.5, zorder=3)

    cx = start + duration / 2
    cumulative = start + duration
    if duration >= 4:
        ax.text(cx, y, f"{duration} ms", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)
        ax.text(cumulative + 0.5, y, f"{cumulative} ms",
                ha="left", va="center", fontsize=7, color=paleink, zorder=4)
    else:
        ax.text(cumulative + 0.5, y, f"{cumulative} ms",
                ha="left", va="center", fontsize=7, color=paleink, zorder=4)

# Stage labels
ax.set_yticks(y_positions)
ax.set_yticklabels([s[0] for s in stages], fontsize=7.5, color=ink)

# X-axis
ax.set_xlabel("Time (ms)", fontsize=10, fontweight="bold", color=ink, labelpad=6)
ax.set_xlim(-0.5, 42)
xticks = np.arange(0, 40, 5)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{t}" for t in xticks], fontsize=8, color=ink)

# Grid
ax.set_axisbelow(True)
ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.6, zorder=0)
ax.yaxis.grid(False)

# Group brackets on the right — inside the xlim so they don't clip
x_bracket = 37.0
for grp_label, i_start, i_end, grp_color in groups:
    y_top = y_positions[i_start] + bar_height / 2 + 0.1
    y_bot = y_positions[i_end] - bar_height / 2 - 0.1
    y_mid = (y_top + y_bot) / 2

    ax.plot([x_bracket, x_bracket + 0.4], [y_top, y_top],
            color=grp_color, linewidth=1.2, zorder=5)
    ax.plot([x_bracket + 0.4, x_bracket + 0.4], [y_top, y_bot],
            color=grp_color, linewidth=1.2, zorder=5)
    ax.plot([x_bracket, x_bracket + 0.4], [y_bot, y_bot],
            color=grp_color, linewidth=1.2, zorder=5)
    ax.text(x_bracket + 1.0, y_mid, grp_label, ha="left", va="center",
            fontsize=8, fontweight="bold", color=grp_color)

# Title
ax.set_title("Sense-to-Actuation Latency Waterfall", fontsize=12,
             fontweight="bold", color=ink, pad=10)

# Spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis="y", length=0, pad=6)
ax.tick_params(axis="x", colors=paleink, length=3)

plt.tight_layout()

# Output paths relative to this script
_here = __import__("pathlib").Path(__file__).resolve().parent
_root = _here.parents[1]
(_root / "02_body" / "images" / "png").mkdir(parents=True, exist_ok=True)
(_root / "02_body" / "images" / "svg").mkdir(parents=True, exist_ok=True)
plt.savefig(_root / "02_body" / "images" / "png" / "fig02_canonical_latency_waterfall.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(_root / "02_body" / "images" / "svg" / "fig02_canonical_latency_waterfall.svg",
            bbox_inches="tight", facecolor="white")
print("OK: fig02_canonical_latency_waterfall")
