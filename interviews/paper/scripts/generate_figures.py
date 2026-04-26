#!/usr/bin/env python3
"""Generate publication-quality data figures for the StaffML paper.

Pipeline: corpus.json → analyze_corpus.py → corpus_stats.json → THIS → PDFs

Run: python3 generate_figures.py
  (or: make figures)

Reads: corpus_stats.json (structured stats from analyze_corpus.py)
Writes: fig-corpus-distribution.pdf, fig-format-balance.pdf,
        fig-zone-distribution.pdf, fig-zone-level-heatmap.pdf
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# ── Config ──────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
PAPER_DIR = SCRIPTS_DIR.parent
FIGURES_DIR = PAPER_DIR / "figures"
STATS_PATH = PAPER_DIR / "corpus_stats.json"

TRACKS = ["cloud", "edge", "mobile", "tinyml", "global"]
LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6+"]
BLOOM_LABELS = {
    "L1": "Remember", "L2": "Understand", "L3": "Apply",
    "L4": "Analyze", "L5": "Evaluate", "L6+": "Create",
}

# Harvard/MIT color palette
CRIMSON = "#A31F34"
BLUE = "#4A90C4"
GREEN = "#3D9E5A"
ORANGE = "#C87B2A"
RED = "#C44444"
GRAY = "#888888"

TRACK_COLORS = {
    "cloud": "#4A90C4",
    "edge": "#3D9E5A",
    "mobile": "#C87B2A",
    "tinyml": "#A31F34",
    "global": "#888888",
}

FORMAT_COLORS = {
    "calculation": "#cfe2f3",  # blue   — compute / processing
    "design":      "#d4edda",  # green  — architecture / data flow
    "conceptual":  "#fdebd0",  # orange — routing / scheduling
    "optimization":"#e7d8ed",  # purple — improvement (distinct hue from green so the
                               #          stacked-bar reading is unambiguous)
    "diagnosis":   "#f9d6d5",  # red    — failure / cost
    "tradeoff":    "#f7f7f7",  # gray   — neutral
}
FORMAT_EDGES = {
    "calculation": "#4a90c4",
    "design":      "#3d9e5a",
    "conceptual":  "#c87b2a",
    "optimization":"#7d4f96",
    "diagnosis":   "#c44",
    "tradeoff":    "#bbb",
}

# Matplotlib defaults for paper — Helvetica to match SVG figures
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_stats():
    """Load pre-computed stats from analyze_corpus.py."""
    if not STATS_PATH.exists():
        print("Error: corpus_stats.json not found. Run: python3 analyze_corpus.py")
        sys.exit(1)
    return json.loads(STATS_PATH.read_text())


def classify_format(scenario: str) -> list[str]:
    s = scenario.lower()
    fmts = []
    if any(w in s for w in ["calculate", "compute", "estimate", "how many", "how much"]):
        fmts.append("calculation")
    if any(w in s for w in ["design", "architect", "propose", "how would you build"]):
        fmts.append("design")
    if any(w in s for w in ["explain", "what is", "define", "describe"]):
        fmts.append("conceptual")
    if any(w in s for w in ["optimize", "improve", "reduce", "speed up"]):
        fmts.append("optimization")
    if any(w in s for w in ["diagnose", "debug", "why is", "root cause", "fails"]):
        fmts.append("diagnosis")
    if any(w in s for w in ["compare", "trade-off", "tradeoff", "versus", " vs "]):
        fmts.append("tradeoff")
    return fmts if fmts else ["conceptual"]


# ── Figure 1: Track × Level Heatmap + Competency Bars ───────────
def fig_corpus_distribution(stats):
    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), width_ratios=[1.2, 1],
        gridspec_kw={"wspace": 0.35},
    )

    # Heatmap from stats
    tlm = stats["track_level_matrix"]
    matrix = np.zeros((len(TRACKS), len(LEVELS)), dtype=int)
    for i, t in enumerate(TRACKS):
        for j, l in enumerate(LEVELS):
            matrix[i, j] = tlm["data"][t][l]

    sns.heatmap(
        matrix, ax=ax_heat, annot=True, fmt="d",
        xticklabels=LEVELS, yticklabels=[t.capitalize() for t in TRACKS],
        cmap="Blues", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Questions", "shrink": 0.8},
    )
    # title in LaTeX caption

    # Totals on right
    for i, t in enumerate(TRACKS):
        total = matrix[i].sum()
        ax_heat.text(len(LEVELS) + 0.15, i + 0.5, f"= {total}",
                     va="center", fontsize=8, color="#555")

    # Competency bar chart from stats
    sorted_areas = list(stats["competency_areas"].items())
    labels = [a for a, _ in sorted_areas]
    counts = [c for _, c in sorted_areas]

    # Color by semantic category
    area_colors = []
    for a in labels:
        if a in ("compute", "memory", "architecture", "parallelism"):
            area_colors.append(BLUE)
        elif a in ("deployment", "data", "networking"):
            area_colors.append(GREEN)
        elif a in ("latency", "precision", "optimization"):
            area_colors.append(ORANGE)
        elif a in ("power", "reliability"):
            area_colors.append(RED)
        else:
            area_colors.append(GRAY)

    bars = ax_bar.barh(range(len(labels)), counts, color=area_colors, alpha=0.8, height=0.7)
    ax_bar.set_yticks(range(len(labels)))
    ax_bar.set_yticklabels(labels, fontsize=7.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Questions")
    # title in LaTeX caption

    # Count labels on bars
    for bar, count in zip(bars, counts):
        ax_bar.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=7, color="#555")

    # title in LaTeX caption (removed suptitle)

    fig.savefig(FIGURES_DIR / "fig-corpus-distribution.pdf")
    print("  Saved figures/fig-corpus-distribution.pdf")
    plt.close(fig)


# ── Figure 2: Question Format by Level (Stacked Bar) ────────────
def fig_format_balance(stats):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    formats = ["calculation", "design", "conceptual", "optimization", "diagnosis", "tradeoff"]
    data = {fmt: [] for fmt in formats}

    fbl = stats["format_by_level"]
    for level in LEVELS:
        for fmt in formats:
            data[fmt].append(fbl[level]["format_pct"].get(fmt, 0))

    x = np.arange(len(LEVELS))
    width = 0.65
    bottom = np.zeros(len(LEVELS))

    for fmt in formats:
        values = data[fmt]
        bars = ax.bar(
            x, values, width, bottom=bottom,
            label=fmt.capitalize(),
            color=FORMAT_COLORS[fmt],
            edgecolor=FORMAT_EDGES[fmt],
            linewidth=0.5,
        )
        # Label percentages > 10%
        for i, v in enumerate(values):
            if v > 10:
                ax.text(x[i], bottom[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=6.5, color="#333")
        bottom += values

    ax.set_xticks(x)
    xlabels = [f"{l}\n({BLOOM_LABELS[l]})" for l in LEVELS]
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_ylabel("Percentage of Questions")
    # title in LaTeX caption
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7)
    ax.set_ylim(0, 105)

    # Annotations removed — caption describes the pattern

    fig.savefig(FIGURES_DIR / "fig-format-balance.pdf")
    print("  Saved figures/fig-format-balance.pdf")
    plt.close(fig)


# ── Figure 3: Zone Distribution (Bar Chart) ──────────────────
def fig_zone_distribution(stats):
    zd = stats.get("zone_distribution", {})
    if not zd:
        print("  ⚠️ No zone_distribution in stats, skipping")
        return

    # Order by count descending
    sorted_zones = sorted(zd.items(), key=lambda x: -x[1])
    labels = [z for z, _ in sorted_zones]
    counts = [c for _, c in sorted_zones]

    # Color by zone type
    PURE = {"recall", "analyze", "design", "implement"}
    COMPOUND = {"diagnosis", "specification", "fluency", "evaluation", "realization", "optimization"}

    zone_colors = []
    for z in labels:
        if z == "mastery":
            zone_colors.append(CRIMSON)
        elif z in PURE:
            zone_colors.append(BLUE)
        else:
            zone_colors.append(GREEN)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    bars = ax.barh(range(len(labels)), counts, color=zone_colors, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([z.capitalize() for z in labels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Questions")
    # title in LaTeX caption

    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va="center", fontsize=7, color="#555")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLUE, alpha=0.85, label="Pure (single skill)"),
        Patch(facecolor=GREEN, alpha=0.85, label="Compound (two skills)"),
        Patch(facecolor=CRIMSON, alpha=0.85, label="Mastery (all four)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    fig.savefig(FIGURES_DIR / "fig-zone-distribution.pdf")
    print("  Saved figures/fig-zone-distribution.pdf")
    plt.close(fig)


# ── Figure 4: Zone × Level Heatmap ───────────────────────────
def fig_zone_level_heatmap(stats):
    zlm = stats.get("zone_level_matrix", {})
    if not zlm:
        print("  ⚠️ No zone_level_matrix in stats, skipping")
        return

    ZONES_ORDERED = [
        "recall", "implement", "fluency",
        "analyze", "diagnosis",
        "design", "specification", "optimization",
        "evaluation", "realization",
        "mastery",
    ]

    matrix = np.zeros((len(ZONES_ORDERED), len(LEVELS)), dtype=int)
    for i, z in enumerate(ZONES_ORDERED):
        for j, l in enumerate(LEVELS):
            matrix[i, j] = zlm.get(z, {}).get(l, 0)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    sns.heatmap(
        matrix, ax=ax, annot=True, fmt="d",
        xticklabels=LEVELS,
        yticklabels=[z.capitalize() for z in ZONES_ORDERED],
        cmap="YlOrRd", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Questions", "shrink": 0.8},
    )
    # title in LaTeX caption
    ax.set_xlabel("Mastery Level")
    ax.set_ylabel("Cognitive Zone")

    fig.savefig(FIGURES_DIR / "fig-zone-level-heatmap.pdf")
    print("  Saved figures/fig-zone-level-heatmap.pdf")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────
def main():
    print("Generating paper figures from corpus_stats.json...\n")
    stats = load_stats()
    print(f"  Published: {stats['summary']['published']}")
    print(f"  Chains: {stats['summary']['chains_total']}\n")

    fig_corpus_distribution(stats)
    fig_format_balance(stats)

    fig_zone_distribution(stats)
    fig_zone_level_heatmap(stats)

    print(f"\nDone. All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
