#!/usr/bin/env python3
"""Generate fig02_canonical_latency_waterfall as a matplotlib figure."""
import pathlib
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
softink        = "#4A5568"
paleink        = "#94A3B8"
gridcolor      = "#E2E8F0"
deadline_bg    = "#FFF1F2"

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
fig, ax = plt.subplots(figsize=(10.5, 5.0))

bar_height = 0.55
y_positions = np.arange(n)[::-1]

# Freshness deadline vertical line at 20 ms
ax.axvline(20, color=harvardcrimson, linestyle="--", linewidth=1.2, zorder=2, alpha=0.85)

# Waterfalls bars and connectors
for i, (label, start, duration, color) in enumerate(stages):
    y = y_positions[i]
    cumulative = start + duration
    
    # Draw horizontal bar
    ax.barh(y, duration, left=start, height=bar_height, color=color,
            edgecolor="white", linewidth=0.8, zorder=3)

    # Connector line to next stage
    if i < n - 1:
        y_next = y_positions[i + 1]
        ax.plot([cumulative, cumulative], [y - bar_height / 2, y_next + bar_height / 2],
                color=paleink, linestyle=":", linewidth=1.0, zorder=2)

    # Label inside or above bar
    cx = start + duration / 2
    if duration >= 4:
        # Avoid deadline line at x=20 for stage 4
        if i == 4:
            text_x = 16.0
            ax.text(text_x, y, f"+{duration} ms", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white", zorder=4)
        else:
            ax.text(cx, y, f"+{duration} ms", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white", zorder=4)
    else:
        # Small duration text above narrow bar with matching color
        ax.text(cx, y + bar_height / 2 + 0.08, f"+{duration} ms", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=color, zorder=4)

    # Elapsed timestamp label at bar end
    if i == n - 1:
        ax.text(cumulative + 0.6, y, f"t = {cumulative} ms (Total)",
                ha="left", va="center", fontsize=8.0, fontweight="bold", color=harvardcrimson, zorder=4)
    else:
        ax.text(cumulative + 0.6, y, f"t = {cumulative} ms",
                ha="left", va="center", fontsize=7.5, fontweight="bold", color=softink, zorder=4)

# Freshness deadline callout badge at top
ax.text(20, n - 0.25, "Freshness Deadline: 20 ms (24 mm tolerance @ 1.2 m/s)",
        ha="center", va="bottom", fontsize=8, fontweight="bold", color=harvardcrimson,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=deadline_bg, edgecolor=harvardcrimson, linewidth=0.9),
        zorder=5)

# Stage labels on y-axis
ax.set_yticks(y_positions)
ax.set_yticklabels([s[0] for s in stages], fontsize=8.0, color=ink)

# X-axis configuration
ax.set_xlim(-0.5, 46.0)
ax.set_ylim(-0.8, n + 0.8)

xticks = np.arange(0, 40, 5)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{t}" for t in xticks], fontsize=8.5, color=ink)

# Set spines bounds so axis line stops cleanly at 35 ms
ax.spines["bottom"].set_bounds(0, 35)
ax.spines["left"].set_bounds(y_positions[-1], y_positions[0])

# Centered X-axis label under data range 0-35
ax.set_xlabel("Elapsed Time from Exposure Midpoint (ms)", fontsize=9.5, fontweight="bold", color=ink)
ax.xaxis.set_label_coords(17.5, -0.09, transform=ax.get_xaxis_transform())

# Grid (only up to 35 ms)
ax.set_axisbelow(True)
for xt in xticks:
    ax.axvline(xt, color=gridcolor, linewidth=0.7, zorder=0, ymin=0.08, ymax=0.92)

# Group brackets on the right
x_bracket = 41.0
for grp_label, i_start, i_end, grp_color in groups:
    y_top = y_positions[i_start] + bar_height / 2 + 0.12
    y_bot = y_positions[i_end] - bar_height / 2 - 0.12
    y_mid = (y_top + y_bot) / 2

    ax.plot([x_bracket, x_bracket + 0.5], [y_top, y_top],
            color=grp_color, linewidth=1.4, zorder=5)
    ax.plot([x_bracket + 0.5, x_bracket + 0.5], [y_top, y_bot],
            color=grp_color, linewidth=1.4, zorder=5)
    ax.plot([x_bracket, x_bracket + 0.5], [y_bot, y_bot],
            color=grp_color, linewidth=1.4, zorder=5)
    ax.text(x_bracket + 1.1, y_mid, grp_label, ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=grp_color)

# Title
ax.set_title("Sense-to-Actuation Latency Waterfall", fontsize=12.5,
             fontweight="bold", color=ink, pad=18)

# Spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis="y", length=0, pad=8)
ax.tick_params(axis="x", colors=softink, length=4)

plt.tight_layout()

# Output paths relative to this script
_here = pathlib.Path(__file__).resolve().parent
_root = _here.parents[1]
(_root / "02_body" / "images" / "png").mkdir(parents=True, exist_ok=True)
(_root / "02_body" / "images" / "svg").mkdir(parents=True, exist_ok=True)
plt.savefig(_root / "02_body" / "images" / "png" / "fig02_canonical_latency_waterfall.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(_root / "02_body" / "images" / "svg" / "fig02_canonical_latency_waterfall.svg",
            bbox_inches="tight", facecolor="white")
print("OK: fig02_canonical_latency_waterfall")
