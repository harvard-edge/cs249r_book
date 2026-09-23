#!/usr/bin/env python3
"""
Chapter 5: Cross-Embodiment Data Scaling Law:
Out-of-Distribution Policy Success vs. Embodiment Diversity.

Demonstrates:
(a) Demonstration scale vs. generalization: Single-embodiment saturation vs.
    multi-scene / cross-embodiment power-law scaling.
(b) Relative transfer multiplier (S_multi / S_single) vs. number of unique
    physical embodiments in training mixture (Open X-Embodiment phase transition).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# Add script directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_style import PALETTE, ERA_COLORS


def add_badge(ax, text, xy, xytext, color=PALETTE["charcoal"], fontsize=8.0, arrow=True, arrow_color=None):
    """Clean rounded annotation badge."""
    arrowprops = None
    if arrow:
        ac = color if arrow_color is None else arrow_color
        arrowprops = dict(
            arrowstyle="->",
            color=ac,
            linewidth=1.0,
            shrinkA=4,
            shrinkB=4,
        )
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.32",
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            linewidth=0.8,
            alpha=0.96,
        ),
        arrowprops=arrowprops,
        zorder=10,
    )


def generate_ch05_plots():
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 0.9

    fig = plt.figure(figsize=(15.2, 6.8), dpi=300, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.0], wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # =========================================================================
    # PANEL (a): Demonstration Hours vs. ID & OOD Success
    # =========================================================================
    ax1.set_xscale("log")
    ax1.set_xlim(1, 25000)
    ax1.set_ylim(0, 108)

    # Background regimes
    ax1.axvspan(1, 50, color=ERA_COLORS[0], alpha=0.75, zorder=0)
    ax1.axvspan(50, 1000, color=ERA_COLORS[1], alpha=0.75, zorder=0)
    ax1.axvspan(1000, 25000, color=ERA_COLORS[2], alpha=0.75, zorder=0)

    # Regime banner pills
    ax1.text(7.0, 101, "Phase I: Memorization\nSingle Cell (<50h)",
             fontsize=7.8, fontweight="bold", color="#475569", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.95))
    ax1.text(220, 101, "Phase II: Multi-Scene\nIn-the-Wild (50–1kh)",
             fontsize=7.8, fontweight="bold", color="#0284C7", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#BAE6FD", alpha=0.95))
    ax1.text(5000, 101, "Phase III: Foundation Generalist\nCross-Embodiment (>1kh)",
             fontsize=7.8, fontweight="bold", color="#166534", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#A7F3D0", alpha=0.95))

    # Continuous scaling curves
    hours = np.logspace(0, 4.35, 300)
    log_h = np.log10(hours)

    id_curve = 68.0 + 7.5 * log_h - 0.5 * (log_h ** 2)
    id_curve = np.clip(id_curve, 60, 92)

    # Single-embodiment OOD plateau (plateaus around 35%)
    ood_single = 12.0 + 10.0 * (1.0 - np.exp(-hours / 40.0)) + 12.0 * (1.0 - np.exp(-hours / 250.0))
    ood_single = np.clip(ood_single, 10, 36)

    # Cross-embodiment multi-scene OOD (power-law continuation)
    ood_multi = 12.0 + 17.2 * log_h - 0.38 * (log_h ** 1.8)
    ood_multi = np.clip(ood_multi, 10, 82)

    # Plot curves
    ax1.plot(hours, id_curve, color=PALETTE["dark_blue"], linestyle="--", linewidth=2.0, zorder=3,
             label="In-Distribution (ID) Success")
    ax1.plot(hours, ood_multi, color=PALETTE["crimson"], linestyle="-", linewidth=2.4, zorder=3,
             label="Cross-Embodiment OOD Generalization")
    ax1.plot(hours[hours >= 30], ood_single[hours >= 30], color="#94A3B8", linestyle=":", linewidth=2.0, zorder=2,
             label="Single-Embodiment OOD Plateau")

    # Shaded brittleness envelope
    ax1.fill_between(hours, ood_multi, id_curve, color=PALETTE["cyan"], alpha=0.08, zorder=1)

    # Empirical data points
    datasets = [
        {"hours": 15, "id": 88.0, "ood": 22.0,
         "text": "ALOHA (15h)\nID: 88% | OOD: 22%\nBrittle off-distribution",
         "target": (15, 22.0), "pos": (3.2, 18), "c": PALETTE["charcoal"]},
        {"hours": 200, "id": 63.0, "ood": 41.0,
         "text": "BridgeData v2 (200h)\nID: 63% | OOD: 41%\n24 kitchens, 13 skills",
         "target": (200, 63.0), "pos": (70, 64), "c": PALETTE["charcoal"]},
        {"hours": 350, "id": 76.0, "ood": 53.0,
         "text": "DROID (350h)\nID: 76% | OOD: 53%\n564 diverse scenes",
         "target": (350, 53.0), "pos": (500, 42), "c": PALETTE["charcoal"]},
        {"hours": 500, "id": 70.0, "ood": 34.0,
         "text": "RT-1 (500h, Single-Robot)\nID: 70% | OOD: 34%\nSingle workcell plateau",
         "target": (500, 34.0), "pos": (700, 20), "c": PALETTE["slate"]},
        {"hours": 2400, "id": 86.0, "ood": 75.8,
         "text": "RT-2-X (2,400h, 22 Robots)\nID: 86% | OOD: 75.8%\n3× cross-robot transfer",
         "target": (2400, 75.8), "pos": (950, 76), "c": PALETTE["crimson"]},
        {"hours": 10000, "id": 91.0, "ood": 78.0,
         "text": "Frontier VLA (>10kh)\nID: 91% | OOD: 78%\nZero-shot manipulation",
         "target": (10000, 78.0), "pos": (12000, 52), "c": PALETTE["forest_green"]},
    ]

    for pt in datasets:
        ax1.scatter(pt["hours"], pt["id"], s=65, color=PALETTE["dark_blue"], edgecolors="#FFFFFF", linewidth=1.2, zorder=5)
        ax1.scatter(pt["hours"], pt["ood"], s=75, color=PALETTE["crimson"], edgecolors="#FFFFFF", linewidth=1.4, zorder=5)
        ax1.plot([pt["hours"], pt["hours"]], [pt["ood"], pt["id"]], color="#CBD5E1", linestyle=":", linewidth=1.0, zorder=2)
        add_badge(ax1, pt["text"], pt.get("target", (pt["hours"], pt["ood"])), pt["pos"], color=pt["c"], fontsize=7.2)

    # Brittleness annotation
    ax1.annotate("Brittleness Envelope\nΔ = 66% gap at 15h",
                 xy=(15, 55), xytext=(4.2, 68),
                 fontsize=7.8, fontweight="bold", color=PALETTE["amber"],
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFBEB", edgecolor="#FCD34D", alpha=0.95),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["amber"], linewidth=1.0),
                 zorder=10)

    ax1.set_xlabel("Robot Demonstration Hours (log scale)", fontsize=10.5, fontweight="bold", color="#1E293B", labelpad=8)
    ax1.set_ylabel("Physical Task Success Rate (%)", fontsize=10.5, fontweight="bold", color="#1E293B", labelpad=8)
    ax1.set_xticks([1, 10, 50, 100, 350, 1000, 2400, 10000])
    ax1.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, loc: f"{int(x):,}h" if x >= 1 else str(x)))
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, loc: f"{int(y)}%"))
    ax1.set_title("(a) Demonstration scale: single-cell saturation vs. cross-embodiment scaling",
                  fontsize=11.2, fontweight="bold", color="#1E293B", pad=12)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color=PALETTE["grid"])
    ax1.legend(loc="lower right", fontsize=8.0, framealpha=0.95, edgecolor="#CBD5E1")

    # =========================================================================
    # PANEL (b): Cross-Embodiment Transfer Multiplier vs. Unique Embodiments K
    # =========================================================================
    k_vals = np.linspace(1, 24, 200)

    # Transfer multiplier curve:
    # K=1: baseline 1.0x
    # K=2..4: slight negative transfer dip (0.88 - 0.95x) due to kinematic conflict & dilution
    # K=6..8: phase transition crosses 1.0x
    # K=22: Open X achieves ~3.0x on novel/emergent skills, ~1.25x on in-domain skills
    novel_mult = 1.0 - 0.16 * np.exp(-((k_vals - 2.5)**2) / 2.2) + 2.15 / (1.0 + np.exp(-(k_vals - 10.0) / 3.2))
    indomain_mult = 1.0 - 0.08 * np.exp(-((k_vals - 2.5)**2) / 2.0) + 0.32 / (1.0 + np.exp(-(k_vals - 8.0) / 3.0))

    # Shaded Zones
    ax2.axhspan(0.7, 1.0, color="#FEF2F2", alpha=0.6, zorder=0)
    ax2.axhspan(1.0, 3.4, color="#F0FDF4", alpha=0.6, zorder=0)

    # 1.0x Neutral baseline
    ax2.axhline(1.0, color="#64748B", linestyle="--", linewidth=1.4, zorder=2)
    ax2.text(23.5, 1.03, "Neutral Transfer Baseline ($1.0\\times$)", color="#64748B",
             fontsize=8.0, fontweight="bold", ha="right", va="bottom")

    # Curves
    ax2.plot(k_vals, novel_mult, color=PALETTE["crimson"], linewidth=2.5, linestyle="-", zorder=4,
             label=r"Out-of-Distribution / Novel Skills ($S_{\mathrm{multi}} / S_{\mathrm{single}}$)")
    ax2.plot(k_vals, indomain_mult, color=PALETTE["dark_blue"], linewidth=2.0, linestyle="-.", zorder=4,
             label=r"In-Distribution Familiar Skills ($S_{\mathrm{multi}} / S_{\mathrm{single}}$)")

    # Data markers from literature
    # K=1: baseline
    ax2.scatter([1], [1.0], color=PALETTE["charcoal"], s=75, zorder=6)
    # K=3: Bridge + Franka + UR5 small co-training
    ax2.scatter([3], [0.92], color="#DC2626", s=75, zorder=6)
    # K=22: Open X-Embodiment (RT-2-X)
    ax2.scatter([22], [3.0], color=PALETTE["crimson"], s=95, marker="*", zorder=6)
    ax2.scatter([22], [1.25], color=PALETTE["dark_blue"], s=85, marker="o", zorder=6)

    # Phase transition threshold vertical line
    k_thresh = 6.8
    ax2.axvline(k_thresh, color=PALETTE["amber"], linestyle=":", linewidth=1.5, zorder=3)
    ax2.text(k_thresh + 0.3, 0.76, f"Phase Transition Threshold\n($K^* \\approx 7$ Embodiments)",
             fontsize=8.2, fontweight="bold", color=PALETTE["amber"])

    # Badges & Annotations
    add_badge(ax2, "Negative Transfer Pitfall (K < 6):\nKinematic conflict & dilution\ndegrades task success",
              xy=(3, 0.92), xytext=(4.5, 1.45),
              color="#B91C1C", fontsize=7.8, arrow=True, arrow_color="#DC2626")

    add_badge(ax2, "Open X-Embodiment (K = 22):\n3.0× Emergent Skill Generalization\n(RT-2-X vs. Single-Robot RT-2)",
              xy=(22, 3.0), xytext=(16.0, 2.75),
              color=PALETTE["crimson"], fontsize=8.0, arrow=True, arrow_color=PALETTE["crimson"])

    add_badge(ax2, "In-Domain Skill Lift:\n+25% positive transfer\nfrom cross-embodiment data",
              xy=(22, 1.25), xytext=(17.0, 1.7),
              color=PALETTE["dark_blue"], fontsize=7.8, arrow=True, arrow_color=PALETTE["dark_blue"])

    ax2.set_xlim(0.5, 24.5)
    ax2.set_ylim(0.7, 3.35)
    ax2.set_xticks(range(2, 25, 2))
    ax2.set_xlabel("Number of Unique Physical Embodiments in Training Mixture ($K$)",
                   fontsize=10.5, fontweight="bold", color="#1E293B", labelpad=8)
    ax2.set_ylabel(r"Relative Transfer Multiplier ($S_{\mathrm{multi}} / S_{\mathrm{single}}$)",
                   fontsize=10.5, fontweight="bold", color="#1E293B", labelpad=8)
    ax2.set_title(r"(b) Positive transfer phase transition vs. embodiment diversity ($K$)",
                  fontsize=11.2, fontweight="bold", color="#1E293B", pad=12)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color=PALETTE["grid"])
    ax2.legend(loc="upper left", fontsize=8.0, framealpha=0.95, edgecolor="#CBD5E1")

    # -------------------------------------------------------------------------
    # Save & Export
    # -------------------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "ch05_cross_embodiment_data_scaling.png")
    plt.savefig(out_path, dpi=300, facecolor="#FFFFFF", bbox_inches="tight")

    repo_root = os.path.abspath(os.path.join(out_dir, ".."))
    book_png_dir = os.path.join(repo_root, "books/vol4/05_data/images/png")
    os.makedirs(book_png_dir, exist_ok=True)
    book_path = os.path.join(book_png_dir, "fig05_cross_embodiment_data_scaling.png")
    plt.savefig(book_path, dpi=300, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close()

    print(f"Generated Chapter 5 plot:")
    print(f"  Candidate: {out_path}")
    print(f"  Book PNG:  {book_path}")
    return out_path


if __name__ == "__main__":
    generate_ch05_plots()
