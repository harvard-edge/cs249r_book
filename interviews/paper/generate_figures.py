#!/usr/bin/env python3
"""Generate publication-quality data figures from corpus.json.

Produces PDF figures for the StaffML methodology paper.
Run from the paper/ directory: python3 generate_figures.py

Reads: ../corpus.json, ../chains.json
Writes: fig-corpus-distribution.pdf, fig-format-balance.pdf, fig-depth-chain.pdf
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
PAPER_DIR = Path(__file__).parent
CORPUS_PATH = PAPER_DIR.parent / "corpus.json"
CHAINS_PATH = PAPER_DIR.parent / "chains.json"

TRACKS = ["cloud", "edge", "mobile", "tinyml"]
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
}

FORMAT_COLORS = {
    "calculation": "#cfe2f3",
    "design": "#d4edda",
    "conceptual": "#fdebd0",
    "optimization": "#e8f5e9",
    "diagnosis": "#f9d6d5",
    "tradeoff": "#f7f7f7",
}
FORMAT_EDGES = {
    "calculation": "#4a90c4",
    "design": "#3d9e5a",
    "conceptual": "#c87b2a",
    "optimization": "#2d7a2d",
    "diagnosis": "#c44",
    "tradeoff": "#bbb",
}

# Matplotlib defaults for paper
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Times New Roman", "DejaVu Serif"],
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


def load_data():
    corpus = json.loads(CORPUS_PATH.read_text())
    pub = [q for q in corpus if q.get("status", "published") == "published"]
    chains = json.loads(CHAINS_PATH.read_text()) if CHAINS_PATH.exists() else []
    return pub, chains


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
def fig_corpus_distribution(pub):
    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), width_ratios=[1.2, 1],
        gridspec_kw={"wspace": 0.35},
    )

    # Heatmap
    matrix = np.zeros((len(TRACKS), len(LEVELS)), dtype=int)
    for q in pub:
        if q["track"] in TRACKS:
            i = TRACKS.index(q["track"])
            j = LEVELS.index(q["level"])
            matrix[i, j] += 1

    sns.heatmap(
        matrix, ax=ax_heat, annot=True, fmt="d",
        xticklabels=LEVELS, yticklabels=[t.capitalize() for t in TRACKS],
        cmap="Blues", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Questions", "shrink": 0.8},
    )
    ax_heat.set_title("Track × Level Distribution", fontweight="bold")

    # Totals on right
    for i, t in enumerate(TRACKS):
        total = matrix[i].sum()
        ax_heat.text(len(LEVELS) + 0.15, i + 0.5, f"= {total}",
                     va="center", fontsize=8, color="#555")

    # Competency bar chart
    areas = Counter(q["competency_area"] for q in pub)
    sorted_areas = areas.most_common()
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
    ax_bar.set_title("Competency Area Distribution", fontweight="bold")

    # Count labels on bars
    for bar, count in zip(bars, counts):
        ax_bar.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=7, color="#555")

    fig.suptitle(
        f"Corpus Distribution ({sum(counts):,} Published Questions)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    fig.savefig(PAPER_DIR / "fig-corpus-distribution.pdf")
    print("  Saved fig-corpus-distribution.pdf")
    plt.close(fig)


# ── Figure 2: Question Format by Level (Stacked Bar) ────────────
def fig_format_balance(pub):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    formats = ["calculation", "design", "conceptual", "optimization", "diagnosis", "tradeoff"]
    data = {fmt: [] for fmt in formats}

    for level in LEVELS:
        lqs = [q for q in pub if q["level"] == level]
        fmt_counts = Counter()
        for q in lqs:
            for fmt in classify_format(q["scenario"]):
                fmt_counts[fmt] += 1
        total = sum(fmt_counts.values()) or 1
        for fmt in formats:
            data[fmt].append(100 * fmt_counts.get(fmt, 0) / total)

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
    ax.set_title("Question Format Distribution by Mastery Level", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7)
    ax.set_ylim(0, 105)

    # Annotation
    ax.annotate(
        "L1–L2: recall + calculation",
        xy=(0.5, 85), fontsize=6.5, color="#555", ha="center",
    )
    ax.annotate(
        "L4–L6+: diagnosis, design, synthesis",
        xy=(4, 85), fontsize=6.5, color="#555", ha="center",
    )

    fig.savefig(PAPER_DIR / "fig-format-balance.pdf")
    print("  Saved fig-format-balance.pdf")
    plt.close(fig)


# ── Figure 3: Depth Chain Example (KV-Cache) ────────────────────
def fig_depth_chain(pub, chains):
    # Find KV-cache chain
    kv_chain = next((ch for ch in chains if "kv-cache" in ch.get("topic", "")), None)
    if not kv_chain:
        print("  ⚠️ No kv-cache chain found, skipping fig-depth-chain")
        return

    by_id = {q["id"]: q for q in pub}

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(kv_chain["questions"]) - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    level_colors = {
        "L1": "#d4edda", "L2": "#d4edda", "L3": "#cfe2f3",
        "L4": "#fdebd0", "L5": "#fdebd0", "L6+": "#f9d6d5",
    }
    level_edge = {
        "L1": "#3d9e5a", "L2": "#3d9e5a", "L3": "#4a90c4",
        "L4": "#c87b2a", "L5": "#c87b2a", "L6+": "#c44",
    }

    for i, cq in enumerate(kv_chain["questions"]):
        level = cq["level"]
        title = cq["title"]
        bloom = BLOOM_LABELS.get(level, "")
        q = by_id.get(cq["id"], {})
        scenario = q.get("scenario", "")[:80] + "..." if q.get("scenario") else ""

        # Level badge
        badge = plt.Rectangle((0.2, i - 0.35), 1.0, 0.7, linewidth=1,
                               edgecolor="white", facecolor=CRIMSON, zorder=3)
        ax.add_patch(badge)
        ax.text(0.7, i - 0.08, level, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4)
        ax.text(0.7, i + 0.18, bloom, ha="center", va="center",
                fontsize=5.5, color="white", zorder=4)

        # Question box
        box = plt.Rectangle((1.4, i - 0.35), 8.2, 0.7, linewidth=1,
                             edgecolor=level_edge[level],
                             facecolor=level_colors[level], zorder=2)
        ax.add_patch(box)
        ax.text(1.6, i - 0.08, title, fontsize=8, fontweight="bold",
                color="#333", va="center", zorder=3)
        ax.text(1.6, i + 0.18, scenario, fontsize=5.5,
                color="#555", va="center", zorder=3)

        # Arrow to next
        if i < len(kv_chain["questions"]) - 1:
            ax.annotate("", xy=(0.7, i + 0.4), xytext=(0.7, i + 0.6),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1))

    ax.set_title(
        f"Depth Chain: {kv_chain['topic']} ({kv_chain.get('competency_area', 'memory')})",
        fontsize=10, fontweight="bold", pad=10,
    )

    fig.savefig(PAPER_DIR / "fig-depth-chain.pdf")
    print("  Saved fig-depth-chain.pdf")
    plt.close(fig)


# ── Figure 4: Dedup / Quality Pipeline Summary ──────────────────
def fig_quality_summary(pub):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))

    # Panel 1: Field coverage
    ax = axes[0]
    fields = {
        "competency_area": sum(1 for q in pub if q.get("competency_area", "").strip()),
        "napkin_math": sum(1 for q in pub if q.get("details", {}).get("napkin_math", "").strip()),
        "common_mistake": sum(1 for q in pub if q.get("details", {}).get("common_mistake", "").strip()),
        "deep_dive_url": sum(1 for q in pub if q.get("details", {}).get("deep_dive_url", "").strip()),
        "bloom_level": sum(1 for q in pub if q.get("bloom_level", "").strip()),
    }
    total = len(pub)
    labels = list(fields.keys())
    pcts = [100 * v / total for v in fields.values()]
    colors = [GREEN if p >= 99 else ORANGE if p >= 80 else RED for p in pcts]

    bars = ax.barh(range(len(labels)), pcts, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.replace("_", "\n") for l in labels], fontsize=6.5)
    ax.set_xlim(95, 101)
    ax.set_xlabel("Coverage %")
    ax.set_title("Field Coverage", fontweight="bold", fontsize=9)
    ax.invert_yaxis()
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="right", fontsize=6.5, color="white", fontweight="bold")

    # Panel 2: Error rates across rounds
    ax = axes[1]
    rounds = ["Before\nR1", "After\nR1", "After\nR2"]
    rates = [4.3, 1.5, 0.22]
    bar_colors = [RED, ORANGE, GREEN]
    bars = ax.bar(rounds, rates, color=bar_colors, alpha=0.85, width=0.5)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Math Error Rate", fontweight="bold", fontsize=9)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{rate}%", ha="center", fontsize=7, fontweight="bold")
    ax.set_ylim(0, 5.5)

    # Panel 3: Dedup stages
    ax = axes[2]
    stages = ["Exact", "Fuzzy\n(>0.90)", "Semantic\n(>0.85)"]
    found = [0, 0, 502]
    archived = [0, 0, 87]
    x = np.arange(len(stages))
    w = 0.35
    ax.bar(x - w / 2, found, w, label="Flagged", color=ORANGE, alpha=0.85)
    ax.bar(x + w / 2, archived, w, label="Archived", color=RED, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=7)
    ax.set_ylabel("Question Pairs")
    ax.set_title("Deduplication", fontweight="bold", fontsize=9)
    ax.legend(fontsize=6.5)

    fig.suptitle("Quality Assurance Summary", fontsize=11, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "fig-quality-summary.pdf")
    print("  Saved fig-quality-summary.pdf")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────
def main():
    print("Generating paper figures from corpus.json...\n")
    pub, chains = load_data()
    print(f"  Corpus: {len(pub)} published questions")
    print(f"  Chains: {len(chains)} depth chains\n")

    fig_corpus_distribution(pub)
    fig_format_balance(pub)
    fig_depth_chain(pub, chains)
    fig_quality_summary(pub)

    print(f"\nDone. All figures saved to {PAPER_DIR}/")


if __name__ == "__main__":
    main()
