#!/usr/bin/env python3
"""
Generate publication-grade matplotlib plot for Chapter 09: Agent Security.
Figure: asrr_waterfall.svg (Attack Surface Reduction Ratio & Defense Trade-offs)
Uses MLSysBook official plotting style via binder.tools.figures.style.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import matplotlib.pyplot as plt
import numpy as np
from binder.tools.figures import style as book_style

# Apply book style
book_style.set_book_style()

# Setup canvas with generous vertical headroom and footroom
fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=300)

tiers = [
    "Tier 0\n(Baseline)",
    "Tier 1\n(Hardened Prompt)",
    "Tier 2\n(Guardrail Model)",
    "Tier 3\n(Dual-LLM Reader)",
    "Tier 4\n(Full Invariant)",
]

subtitles = [
    "Naive Single Prompt",
    "Negative Instructions",
    "Front-Door Filter",
    "Grammar Extraction",
    "Quarantine + Taint + JIT",
]

asrs = [74.2, 48.6, 28.4, 2.1, 0.08]
asrrs = ["Ref", "34.5%", "61.7%", "97.2%", "99.89%"]
overheads = ["0 ms • 0.0%", "+45 ms • 1.2%", "+180 ms • 4.8%", "+420 ms • 0.3%", "+460 ms • 0.4%"]

# Palette aligned with MLSysBook book_style
colors = ["#E11D48", "#D97706", "#CA8A04", "#0D9488", "#059669"]
edge_colors = ["#9F1239", "#B45309", "#A16207", "#0F766E", "#047857"]

x = np.arange(len(tiers))
width = 0.50

# Grid & Spines
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#CBD5E1", alpha=0.8)
ax.xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#64748B")
ax.spines["bottom"].set_color("#64748B")

# Plot bars
bars = ax.bar(x, asrs, width=width, color=colors, edgecolor=edge_colors, linewidth=1.2, zorder=3)

# Y axis limits and formatting
ax.set_ylim(0, 100)
ax.set_ylabel("Attack Success Rate (ASR) %", fontsize=10, fontweight="bold", color="#1E293B")
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.5, color="#475569")

# Set x-ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(tiers, fontsize=8.2, fontweight="bold", color="#1E293B")
ax.tick_params(axis="x", pad=6)

# Annotations above bars
for i, (bar, asr) in enumerate(zip(bars, asrs)):
    h = bar.get_height()
    if asr >= 1.0:
        label_text = f"{asr:.1f}%"
    else:
        label_text = f"{asr:.2f}%"
    
    y_pos = h + 2.2
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y_pos,
        label_text,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=edge_colors[i],
    )

# Connecting step arrows showing ASRR drops
for i in range(len(asrs) - 1):
    x_start = x[i] + width / 2 + 0.03
    x_end = x[i + 1] - width / 2 - 0.03
    y_start = asrs[i]
    y_end = asrs[i + 1]
    
    ax.annotate(
        "",
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        arrowprops=dict(
            arrowstyle="->",
            color="#94A3B8",
            linestyle=":",
            linewidth=1.2,
            shrinkA=4,
            shrinkB=4,
        ),
        zorder=2,
    )

# Lower metadata labels using axis transform (fraction of axis height below 0)
for i in range(len(tiers)):
    # Mechanism subtitle
    ax.text(
        x[i],
        -0.16,
        subtitles[i],
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.2,
        color="#64748B",
        style="italic",
        clip_on=False,
    )
    # ASRR pill / badge
    badge_bg = "#F1F5F9" if i == 0 else "#F0FDF4"
    badge_border = "#CBD5E1" if i == 0 else "#86EFAC"
    badge_text_color = "#475569" if i == 0 else "#166534"
    asrr_label = f"ASRR: {asrrs[i]}"
    ax.text(
        x[i],
        -0.24,
        asrr_label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=badge_text_color,
        bbox=dict(boxstyle="round,pad=0.25", fc=badge_bg, ec=badge_border, lw=0.8),
        clip_on=False,
    )
    # Latency / ODR
    ax.text(
        x[i],
        -0.34,
        f"ΔT: {overheads[i].split(' • ')[0]}\nODR: {overheads[i].split(' • ')[1]}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.8,
        color="#475569",
        clip_on=False,
    )

# Top banner / summary badge
ax.text(
    0.5,
    1.08,
    "ATTACK SURFACE REDUCTION RATIO (ASRR) & DEFENSE TRADE-OFFS (INJECAGENT & AGENTDOJO)",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=9.2,
    fontweight="bold",
    color="#0F172A",
)

# Bottom trade-off summary box
fig.text(
    0.5,
    0.02,
    "System Trade-Off Summary: Moving from Tier 0 to Tier 4 drives ASRR to 99.89% (>900× safety boost) • ODR capped at 0.4% • Latency overhead: +460 ms",
    ha="center",
    va="bottom",
    fontsize=7.6,
    color="#334155",
    bbox=dict(boxstyle="round,pad=0.4", fc="#F8FAFC", ec="#CBD5E1", lw=0.9),
)

plt.subplots_adjust(top=0.90, bottom=0.32, left=0.10, right=0.96)

out_svg = root / "books/vol3/09_security/images/svg/asrr_waterfall.svg"
out_png = Path("/tmp/ch09_asrr_waterfall.png")

fig.savefig(out_svg, format="svg", bbox_inches="tight")
fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Generated {out_svg}")
print(f"Generated {out_png}")
