#!/usr/bin/env python3
"""Pipeline-parallelism bubble timeline (1F1B variant, 4 stages × 4 micro-batches).

Visualizes the warm-up + steady-state + cool-down regions of a 1F1B
schedule on 4 GPUs. Bubble fraction = (P-1) / (P-1+M) where P is the
number of pipeline stages and M is the number of micro-batches.

Renders to $VISUAL_OUT_PATH.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.environ.get("VISUAL_OUT_PATH", "cloud-2848.svg")

P = 4   # stages (GPUs)
M = 4   # micro-batches

# Each cell = 1 unit of time. Forward and backward each take 1 unit per stage.
# 1F1B: warm-up sends P-1 forwards, then alternates F/B in steady state,
# cool-down has P-1 backwards.

# We'll lay out a Gantt: y-axis = stage (top = stage 0), x-axis = time.
fig, ax = plt.subplots(figsize=(7.5, 3.5))

# Colors
COL_FW = "#cfe2f3"   # forward — compute blue
COL_BW = "#d4edda"   # backward — data-flow green
COL_BUBBLE = "#f9d6d5"  # bubble — error red

# Schedule generator (1F1B-naive, just for visualization)
# For visualization clarity, we use a simplified GPipe-like schedule:
# all forwards first, then all backwards. Bubble = (P-1) at start + (P-1) at end.
# This makes the bubble visually obvious; the question prose will then
# show how 1F1B reduces this.

# Forward pass: stage s starts at time s, processes M microbatches consecutively
for s in range(P):
    for m in range(M):
        t_start = s + m
        ax.add_patch(mpatches.Rectangle(
            (t_start, P - 1 - s), 1, 0.8,
            facecolor=COL_FW, edgecolor="#4a90c4", lw=0.8,
        ))
        ax.text(t_start + 0.5, P - 1 - s + 0.4, f"F{m}",
                ha="center", va="center", fontsize=8)

# Backward pass: stage s starts at time (P-1) + (M-1) + (P-1-s) + 1 = ...
# Simpler: backward from stage P-1 down to stage 0
fw_end = (P - 1) + M  # time when last forward finishes on stage P-1
for s in reversed(range(P)):
    rev_offset = (P - 1) - s
    for m in range(M):
        t_start = fw_end + rev_offset + m
        ax.add_patch(mpatches.Rectangle(
            (t_start, P - 1 - s), 1, 0.8,
            facecolor=COL_BW, edgecolor="#3d9e5a", lw=0.8,
        ))
        ax.text(t_start + 0.5, P - 1 - s + 0.4, f"B{m}",
                ha="center", va="center", fontsize=8)

# Bubble shading: idle time on each stage at start (warm-up)
for s in range(P):
    if s > 0:
        ax.add_patch(mpatches.Rectangle(
            (0, P - 1 - s), s, 0.8,
            facecolor=COL_BUBBLE, alpha=0.4, edgecolor="none",
        ))

# Bubble shading: idle time on each stage at end (cool-down)
total_time = fw_end + (P - 1) + M
for s in range(P):
    cool_start = fw_end + ((P - 1) - s) + M  # when this stage finishes backward
    cool_dur = total_time - cool_start
    if cool_dur > 0:
        ax.add_patch(mpatches.Rectangle(
            (cool_start, P - 1 - s), cool_dur, 0.8,
            facecolor=COL_BUBBLE, alpha=0.4, edgecolor="none",
        ))

# Annotate bubble fraction
bubble_units = P * (P - 1)  # warm-up + cool-down per stage
total_units = P * total_time
bubble_frac = bubble_units / total_units
ax.text(
    total_time / 2, P + 0.4,
    f"Bubble fraction = (P-1)/(P-1+M) = {(P-1)/(P-1+M):.2f}  "
    f"(P={P} stages, M={M} micro-batches)",
    ha="center", fontsize=9, color="#555",
)

ax.set_xlim(0, total_time + 0.5)
ax.set_ylim(-0.4, P + 1.0)
ax.set_yticks([P - 1 - s + 0.4 for s in range(P)])
ax.set_yticklabels([f"GPU {s}\n(stage {s})" for s in range(P)], fontsize=9)
ax.set_xlabel("Time (units)", fontsize=10)
ax.set_title("Pipeline parallelism: bubble structure (GPipe-style schedule)",
             fontsize=11, loc="left")

# Legend
legend_handles = [
    mpatches.Patch(facecolor=COL_FW, edgecolor="#4a90c4", label="Forward"),
    mpatches.Patch(facecolor=COL_BW, edgecolor="#3d9e5a", label="Backward"),
    mpatches.Patch(facecolor=COL_BUBBLE, alpha=0.4, label="Bubble (idle)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

# Clean spines
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.tight_layout()
fig.savefig(OUT, format="svg", bbox_inches="tight")
