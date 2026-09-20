#!/usr/bin/env python3
"""
scripts/build_all_vol3_margin_svgs.py

Generates and collects all 72 canonical micro-SVGs (4 per chapter across 18 chapters)
for Volume III: Agentic Machine Learning Systems into their respective chapter
image directories:
    books/vol3/<chapter_dir>/images/svg/vol3_<chapter>_margin_00[1-4].svg
"""

import os
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

BASE_DIR = Path("/Users/VJ/GitHub/MLSysBook-vol3")

# Map of chapter number to directory and prefix
CHAPTER_MAP = {
    1: ("01_introduction", "vol3_introduction"),
    2: ("02_processor", "vol3_processor"),
    3: ("03_deliberation", "vol3_deliberation"),
    4: ("04_working_sets", "vol3_working_sets"),
    5: ("05_virtual_memory", "vol3_virtual_memory"),
    6: ("06_episodic_memory", "vol3_episodic_memory"),
    7: ("07_actuation", "vol3_actuation"),
    8: ("08_virtualization", "vol3_virtualization"),
    9: ("09_checkpointing", "vol3_checkpointing"),
    10: ("10_interrupts", "vol3_interrupts"),
    11: ("11_scheduling", "vol3_scheduling"),
    12: ("12_data_flywheel", "vol3_data_flywheel"),
    13: ("13_sft", "vol3_sft"),
    14: ("14_rlvr", "vol3_rlvr"),
    15: ("15_multi_agent", "vol3_multi_agent"),
    16: ("16_observability", "vol3_observability"),
    17: ("17_tokenomics", "vol3_tokenomics"),
    18: ("18_conclusion", "vol3_conclusion"),
}

# -----------------------------------------------------------------------------
# Copy already generated SVGs from subagents / tmp
# -----------------------------------------------------------------------------
COPY_SOURCES = {
    3: Path("/Users/VJ/.gemini/antigravity-cli/brain/df756c1c-9265-49aa-b927-d1cb8809022a/scratch"),
    5: Path("/Users/VJ/.gemini/antigravity-cli/brain/e05aeabd-b156-4da5-9773-d0b3d996dade/scratch"),
    6: Path("/Users/VJ/.gemini/antigravity-cli/brain/e88923a7-5a2f-4652-8297-897e63621717/scratch"),
    7: Path("/Users/VJ/.gemini/antigravity-cli/brain/1aa0f16f-1bfe-4571-9c40-f103dcc02292/scratch"),
    9: Path("/tmp"),
    10: Path("/tmp"),
    11: Path("/tmp"),
    12: Path("/tmp"),
    14: Path("/Users/VJ/.gemini/antigravity-cli/brain/c6b19075-b3e8-4263-bae2-37a2f10b4aa8/scratch"),
    16: Path("/Users/VJ/.gemini/antigravity-cli/brain/dc2c1d92-421c-4985-92a8-c47130a41c25/scratch"),
    17: Path("/Users/VJ/.gemini/antigravity-cli/brain/0ebfbaf7-dc4e-495e-98d3-bbbbbadde161/scratch"),
    18: Path("/Users/VJ/.gemini/antigravity-cli/brain/20b6030e-4c6f-4b53-b703-9fae6efa0a91/scratch"),
}

def copy_pregenerated():
    for ch_num, src_dir in COPY_SOURCES.items():
        ch_dir, prefix = CHAPTER_MAP[ch_num]
        dst_dir = BASE_DIR / "books" / "vol3" / ch_dir / "images" / "svg"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fig_idx in range(1, 5):
            fname = f"{prefix}_margin_00{fig_idx}.svg"
            src_file = src_dir / fname
            if src_file.exists():
                dst_file = dst_dir / fname
                shutil.copy2(src_file, dst_file)
                print(f"[COPY] Ch {ch_num:02d}: {fname} -> {dst_file}")

# -----------------------------------------------------------------------------
# Generators for missing chapters (01, 02, 04, 08, 13, 15)
# -----------------------------------------------------------------------------
CRIMSON = "#A51C30"
RED = "#CB202D"
SLATE = "#5E6B73"
EMERALD = "#008F45"
AMBER = "#D97706"
BLUE = "#006395"
INK = "#333333"
GRID = "#CCCCCC"
REDFILL = "#F2D7D5"

def gen_ch01(dst_dir: Path):
    # 001: 13 Orders of Magnitude
    plt.rcParams.update({
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.family": "sans-serif", "font.size": 5.5, "axes.grid": False, "svg.fonttype": "path",
    })
    fig, ax = plt.subplots(figsize=(1.25, 1.625), dpi=300)
    tiers = [
        ("Trajectory", 1e4, CRIMSON, "10⁴ s"),
        ("Sandbox Tool", 1e1, SLATE, "10¹ s"),
        ("Model Pass", 1e-1, SLATE, "10⁻¹ s"),
        ("Network RPC", 1e-3, SLATE, "10⁻³ s"),
        ("OS Context", 1e-6, SLATE, "10⁻⁶ s"),
        ("Hardware FMA", 1e-9, SLATE, "10⁻⁹ s"),
    ]
    n = len(tiers)
    xmin, xmax = 1e-10, 1e6
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, n - 0.3)
    for i, (label, val, col, exp_str) in enumerate(tiers):
        y = n - 1 - i
        ax.barh(y, val - xmin, left=xmin, height=0.58, color=col, alpha=0.92)
        if i == 0:
            ax.text(val * 0.45, y, f"{label} ({exp_str})", color="white", fontsize=4.4,
                    va="center", ha="right", fontweight="bold")
        else:
            ax.text(val * 2.5, y, f"{label} ({exp_str})", color=INK, fontsize=4.3,
                    va="center", ha="left")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(xmax * 0.9, -0.55, "13 orders (s)", color=SLATE, fontsize=3.9, ha="right", style="italic")
    fig.savefig(dst_dir / "vol3_introduction_margin_001.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # 002: Trajectory Goodput vs Badput
    fig, ax = plt.subplots(figsize=(1.25, 1.25), dpi=300)
    goodput, badput = 21.2, 78.8
    ax.barh(0, goodput, left=0, height=0.42, color=EMERALD, alpha=0.95)
    ax.barh(0, badput, left=goodput, height=0.42, color=CRIMSON, alpha=0.95)
    ax.text(goodput / 2, 0, "21.2%", color="white", fontsize=4.8, fontweight="bold", ha="center", va="center")
    ax.text(goodput + badput / 2, 0, "78.8%", color="white", fontsize=5.2, fontweight="bold", ha="center", va="center")
    ax.text(goodput / 2, 0.28, "Goodput\n(35 tasks)", color=EMERALD, fontsize=4.3, fontweight="bold", ha="center", va="bottom")
    ax.text(goodput + badput / 2, 0.28, "Badput (Waste)\n(65 fail loops)", color=CRIMSON, fontsize=4.3, fontweight="bold", ha="center", va="bottom")
    ax.text(50, -0.30, "Cluster Compute (4,125 Steps)", color=INK, fontsize=4.3, ha="center", va="top")
    ax.text(50, -0.46, "Macro-Efficiency (Goodput G = 21.2%)", color=SLATE, fontsize=3.8, ha="center", va="top", style="italic")
    ax.set_xlim(0, 100); ax.set_ylim(-0.62, 0.65)
    for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(dst_dir / "vol3_introduction_margin_002.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # 003: Context Poisoning Feedback Loop
    fig, ax = plt.subplots(figsize=(1.25, 1.25), dpi=300)
    pts = {"error": (0.50, 0.82), "cache": (0.80, 0.26), "action": (0.20, 0.26)}
    box_w, box_h = 0.37, 0.19
    items = [("error", AMBER, "Fail-Plausible\nError (a_t)"),
             ("cache", SLATE, "KV Cache\nAttn Sink"),
             ("action", CRIMSON, "Degenerate\nAction (a_t)")]
    for key, col, txt in items:
        px, py = pts[key]
        box = patches.FancyBboxPatch((px - box_w/2, py - box_h/2), box_w, box_h,
                                     boxstyle="round,pad=0.01,rounding_size=0.03",
                                     facecolor=col, edgecolor="none", alpha=0.95)
        ax.add_patch(box)
        ax.text(px, py, txt, color="white", fontsize=3.7, fontweight="bold", ha="center", va="center")
    arrow_kw = dict(arrowstyle="-|>", color=INK, lw=0.75, mutation_scale=6)
    ax.annotate("", xy=(pts["cache"][0] - 0.05, pts["cache"][1] + box_h/2 + 0.01),
                xytext=(pts["error"][0] + 0.10, pts["error"][1] - box_h/2 - 0.01), arrowprops=arrow_kw)
    ax.annotate("", xy=(pts["action"][0] + box_w/2 + 0.01, pts["action"][1]),
                xytext=(pts["cache"][0] - box_w/2 - 0.01, pts["cache"][1]), arrowprops=arrow_kw)
    ax.annotate("", xy=(pts["error"][0] - 0.10, pts["error"][1] - box_h/2 - 0.01),
                xytext=(pts["action"][0] + 0.05, pts["action"][1] + box_h/2 + 0.01), arrowprops=arrow_kw)
    ax.text(0.50, 0.44, "Context\nPoisoning\nLoop", color=CRIMSON, fontsize=4.1,
            fontweight="bold", ha="center", va="center", style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0.08, 0.98)
    for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(dst_dir / "vol3_introduction_margin_003.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # 004: Amdahl Trajectory Latency Wall
    fig, ax = plt.subplots(figsize=(1.25, 1.25), dpi=300)
    s_model = np.linspace(1.0, 10.0, 200)
    f_model = 0.2214
    f_non_model = 1.0 - f_model
    s_traj = 1.0 / (f_non_model + (f_model / s_model))
    s_max = 1.0 / f_non_model
    ax.axhspan(s_max - 0.03, 1.35, color=REDFILL, alpha=0.45)
    ax.axhline(s_max, color=CRIMSON, lw=0.8, ls="--")
    ax.text(9.7, s_max + 0.015, "1.28x Amdahl Wall", color=CRIMSON, fontsize=4.2,
            fontweight="bold", ha="right", va="bottom")
    ax.plot(s_model, s_traj, color=INK, lw=1.2)
    s_pt = 3.5
    traj_pt = 1.0 / (f_non_model + (f_model / s_pt))
    ax.plot(s_pt, traj_pt, "o", color=CRIMSON, ms=3.2)
    ax.annotate("3.5x to 1.19x", xy=(s_pt, traj_pt), xytext=(s_pt + 0.5, traj_pt - 0.08),
                fontsize=4.2, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.5))
    ax.set_xlim(1.0, 10.0); ax.set_ylim(0.95, 1.36)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color(GRID); ax.spines["left"].set_linewidth(0.5)
    ax.set_xlabel("Inference Speedup (S_model)", fontsize=4.4, color=INK, labelpad=3)
    ax.set_ylabel("Trajectory Speedup", fontsize=4.4, color=INK, labelpad=3)
    ax.tick_params(axis="both", which="both", labelsize=4.0, length=2, color=GRID)
    fig.savefig(dst_dir / "vol3_introduction_margin_004.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("[GEN] Ch 01 complete.")

def gen_ch02(dst_dir: Path):
    # 001: Vocabulary Simplex Truncation
    fig, ax = plt.subplots(figsize=(1.01, 1.39), dpi=300)
    fig.subplots_adjust(left=0.12, right=0.92, top=0.82, bottom=0.14)
    ranks = np.array([1, 2, 3, 4, 5])
    probs = np.array([0.784, 0.175, 0.039, 0.002, 0.0001])
    colors = [EMERALD, EMERALD, CRIMSON, CRIMSON, GRID]
    ax.bar(ranks, probs, width=0.74, color=colors, alpha=0.92, edgecolor='none')
    ax.text(1, 0.40, "v1", ha='center', va='center', fontsize=5.0, color='white', fontweight='bold')
    ax.text(1, probs[0] + 0.02, "78%", ha='center', va='bottom', fontsize=4.4, color=INK, fontweight='bold')
    ax.text(2, 0.08, "v2", ha='center', va='center', fontsize=4.4, color='white', fontweight='bold')
    ax.text(2, probs[1] + 0.02, "18%", ha='center', va='bottom', fontsize=4.2, color=INK, fontweight='bold')
    ax.text(3, probs[2] + 0.02, "4%", ha='center', va='bottom', fontsize=3.9, color=CRIMSON)
    ax.text(4, probs[3] + 0.02, "tail", ha='center', va='bottom', fontsize=3.9, color=CRIMSON)
    y_brk = 0.96
    ax.plot([0.63, 2.37], [y_brk, y_brk], color=EMERALD, lw=0.9)
    ax.plot([0.63, 0.63], [y_brk - 0.03, y_brk], color=EMERALD, lw=0.9)
    ax.plot([2.37, 2.37], [y_brk - 0.03, y_brk], color=EMERALD, lw=0.9)
    ax.text(1.5, y_brk + 0.03, "top-p (90%)\n2 tokens", color=EMERALD, fontsize=4.6,
            fontweight='bold', ha='center', va='bottom')
    ax.axvline(4.5, color=CRIMSON, ls="--", lw=0.65)
    ax.text(4.4, 0.52, "top-k (k=4)\nadmits tail", color=CRIMSON, fontsize=4.0,
            ha='right', va='center', rotation=90, fontweight='bold')
    ax.set_xlim(0.3, 5.5); ax.set_ylim(0, 1.16)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(0.55)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.3, -0.09, "Token rank (1 to 5)", color=INK, fontsize=4.4, ha='left', va='top')
    fig.savefig(dst_dir / "vol3_processor_margin_001.svg", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    # 002: Roofline Boundary
    fig, ax = plt.subplots(figsize=(1.01, 1.39), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.86, bottom=0.14)
    ridge = 295.2
    xmin, xmax = 0.2, 12000.0
    ymin, ymax = 0.4, 1800.0
    x = np.logspace(np.log10(xmin), np.log10(xmax), 240)
    y = np.minimum(989.0, 3.35 * x)
    m = x < ridge
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(x[m], y[m], color=BLUE, lw=1.35)
    ax.plot(x[~m], y[~m], color="#E67817", lw=1.35)
    ax.axvline(ridge, color=GRID, ls="--", lw=0.55)
    ax.text(ridge * 0.70, 1.0, "I*=295", color="#666666", fontsize=4.2, ha='right', va='bottom')
    ax.plot(1.0, 3.35, "o", color=BLUE, ms=3.5, zorder=5)
    ax.text(3.5, 1.6, "Decode (B=1)\n1 FLOP/B (0.7%)", fontsize=4.3, color=BLUE,
            fontweight='bold', ha='left', va='bottom')
    ax.plot(4096.0, 989.0, "o", color="#E67817", ms=3.5, zorder=5)
    ax.text(4200.0, 380.0, "Prefill\n4k FLOP/B", fontsize=4.3, color="#E67817",
            fontweight='bold', ha='center', va='top')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.55)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(xmin, ymax * 0.95, "Perf (TFLOP/s)", color=INK, fontsize=4.3, ha='left', va='bottom')
    ax.text(xmax, ymin * 1.15, "FLOPs / byte", color=INK, fontsize=4.3, ha='right', va='bottom')
    fig.savefig(dst_dir / "vol3_processor_margin_002.svg", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    # 003: 70B Shuttle Latency Floor
    fig, ax = plt.subplots(figsize=(1.01, 1.39), dpi=300)
    fig.subplots_adjust(left=0.22, right=0.92, top=0.86, bottom=0.14)
    tiers = [("A100", 34.3, "34.3 ms"), ("H100", 20.9, "20.9 ms"),
             ("H200", 14.6, "14.6 ms"), ("B200", 8.8, "8.8 ms")]
    n = len(tiers); xmax = 42.0; h = 0.54
    ax.set_ylim(-0.5, n - 0.2); ax.set_xlim(0, xmax)
    ax.plot([0, xmax], [n - 0.35, n - 0.35], color=RED, lw=0.8)
    ax.text(xmax, n - 0.25, "70B weight shuttle", color=RED, fontsize=4.3, ha='right', va='bottom', fontweight='bold')
    for i, (name, val, val_str) in enumerate(tiers):
        y = n - 1 - i
        ax.barh(y, val, height=h, color=SLATE, alpha=0.90, edgecolor='none')
        ax.text(-1.5, y, name, color=INK, fontsize=4.7, fontweight='bold', ha='right', va='center')
        ax.text(val + 1.2, y, val_str, color=INK, fontsize=4.5, fontweight='bold', ha='left', va='center')
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(0.55)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0, -0.44, "Floor latency / token (ms)", color=INK, fontsize=4.3, ha='left', va='top')
    fig.savefig(dst_dir / "vol3_processor_margin_003.svg", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    # 004: SLA Latency Budget Envelope
    fig, ax = plt.subplots(figsize=(1.01, 1.39), dpi=300)
    fig.subplots_adjust(left=0.28, right=0.92, top=0.84, bottom=0.14)
    limit, xmax = 15.0, 44.0
    ax.axvspan(limit, xmax, color=REDFILL, alpha=0.45, lw=0)
    ax.axvline(limit, color=RED, lw=0.8, ls="--")
    ax.text(limit + 1.2, 1.85, "SLA Limit: 15s", color=RED, fontsize=4.4, fontweight='bold', ha='left', va='bottom')
    y1, h = 1.25, 0.44
    ax.barh(y1, 1.21, height=h, color="#E67817", edgecolor='none')
    ax.barh(y1, 38.67, left=1.21, height=h, color=RED, edgecolor='none')
    ax.text(-1.5, y1, "JSON\n(1850 t)", color=INK, fontsize=4.5, fontweight='bold', ha='right', va='center')
    ax.text(38.0, y1, "39.9s FAIL", color="white", fontsize=4.5, fontweight='bold', ha='right', va='center')
    y2 = 0.40
    ax.barh(y2, 1.18, height=h, color="#E67817", edgecolor='none')
    ax.barh(y2, 1.67, left=1.18, height=h, color=BLUE, edgecolor='none')
    ax.text(-1.5, y2, "Diff\n(80 t)", color=INK, fontsize=4.5, fontweight='bold', ha='right', va='center')
    ax.text(3.4, y2, "2.85s\nPASS", color=BLUE, fontsize=4.4, fontweight='bold', ha='left', va='center')
    ax.set_ylim(-0.2, 2.0); ax.set_xlim(0, xmax)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(0.55)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0, -0.32, "Total latency T (seconds)", color=INK, fontsize=4.3, ha='left', va='top')
    fig.savefig(dst_dir / "vol3_processor_margin_004.svg", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("[GEN] Ch 02 complete.")

def gen_ch04(dst_dir: Path):
    # 001: Attention Crossover Knee
    fig, ax = plt.subplots(figsize=(1.15, 1.15), dpi=300)
    x = np.linspace(0, 160, 200)
    y_lin = x / 90.112
    y_quad = (x / 90.112) ** 2
    ax.plot(x, y_lin, color=SLATE, lw=1.1, ls="--")
    ax.plot(x, y_quad, color=CRIMSON, lw=1.35)
    m = x >= 90.112
    ax.fill_between(x[m], y_lin[m], y_quad[m], color=REDFILL, alpha=0.45, edgecolor="none")
    ax.plot(90.112, 1.0, "o", color=CRIMSON, ms=3.5)
    ax.axvline(90.112, color=GRID, ls=":", lw=0.6)
    ax.text(60, 2.5, "Attention\nO(N²)", fontsize=4.8, color=CRIMSON, ha="center", va="center", fontweight="bold")
    ax.text(82, 1.15, "90k crossover", fontsize=4.6, color=INK, ha="right", va="bottom", fontweight="bold")
    ax.text(130, 0.45, "Linear FFN\nO(N)", fontsize=4.4, color=SLATE, ha="center", va="center")
    ax.set_xlim(0, 160); ax.set_ylim(0, 3.2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.55)
    ax.set_xticks([0, 90, 160])
    ax.set_xticklabels(["0", "90k", "160k"], fontsize=4.2, color=SLATE)
    ax.tick_params(axis="x", which="both", length=2, color=GRID, pad=1)
    ax.set_yticks([])
    fig.savefig(dst_dir / "vol3_working_sets_margin_001.svg", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)

    # 002: Softmax Denominator Dilution Sparkline
    fig, ax = plt.subplots(figsize=(1.15, 1.15), dpi=300)
    log_N = np.linspace(2, 5, 200)
    N = 10 ** log_N
    e_target, e_bg = 403.4288, 0.135335
    alpha = e_target / (e_target + N * e_bg) * 100
    ax.axhline(100, color=GRID, ls=":", lw=0.6)
    ax.plot(log_N, alpha, color=CRIMSON, lw=1.4)
    ax.fill_between(log_N, 0, alpha, color=REDFILL, alpha=0.35, edgecolor="none")
    ax.plot(2.0, alpha[0], "o", color=EMERALD, ms=3.8)
    ax.plot(5.0, alpha[-1], "o", color=CRIMSON, ms=3.8)
    ax.text(2.15, 86, "97%", fontsize=5.2, color=EMERALD, fontweight="bold", ha="left", va="center")
    ax.text(3.9, 82, "33× attention\ncollapse", fontsize=4.6, color=CRIMSON, fontweight="bold", ha="center", va="center")
    ax.text(4.9, 14, "3%", fontsize=5.2, color=CRIMSON, fontweight="bold", ha="right", va="bottom")
    ax.set_xlim(1.85, 5.25); ax.set_ylim(0, 115)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.55)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xticklabels(["100", "1k", "10k", "100k"], fontsize=4.0, color=SLATE)
    ax.tick_params(axis="x", which="both", length=2, color=GRID, pad=1)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0%", "50%", "100%"], fontsize=4.0, color=SLATE)
    ax.tick_params(axis="y", which="both", length=2, color=GRID, pad=1)
    fig.savefig(dst_dir / "vol3_working_sets_margin_002.svg", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)

    # 003: Deliberative Reasoning 2PC Budget
    fig, ax = plt.subplots(figsize=(1.3, 1.05), dpi=300)
    limit, unpruned, committed, xmax = 10.0, 57.6, 0.96, 65.0
    ax.axvspan(limit, xmax, color=REDFILL, alpha=0.45, lw=0)
    ax.axvline(limit, color=RED, lw=0.75, ls="--", zorder=1)
    ax.text(limit + 1.5, 0.92, "10 GB GPU limit", color=RED, fontsize=4.4, fontweight="bold",
            ha="left", va="center", bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=3)
    y1, h = 0.58, 0.14
    ax.text(0, 0.73, "Unpruned CoT (30 turns)", color=INK, fontsize=4.4, fontweight="bold",
            ha="left", va="bottom", bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"), zorder=3)
    ax.add_patch(patches.Rectangle((0, y1 - h/2), unpruned, h, facecolor=CRIMSON, edgecolor="white", lw=0.4, zorder=2))
    ax.text(unpruned / 2, y1, "57.6 GB", color="white", fontsize=4.8, fontweight="bold", ha="center", va="center", zorder=3)
    y2 = 0.18
    ax.text(0, 0.33, "Two-Phase Commit", color=INK, fontsize=4.4, fontweight="bold",
            ha="left", va="bottom", bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"), zorder=3)
    ax.add_patch(patches.Rectangle((0, y2 - h/2), max(committed, 0.9), h, facecolor=EMERALD, edgecolor="white", lw=0.4, zorder=2))
    ax.text(committed + 1.8, y2, "0.96 GB", color=EMERALD, fontsize=4.8, fontweight="bold", ha="left", va="center", zorder=3)
    ax.text(xmax - 1, 0.18, "60× memory\nsaved", color=EMERALD, fontsize=4.6, fontweight="bold", ha="right", va="center", zorder=3)
    ax.set_xlim(-1, xmax); ax.set_ylim(0, 1.05)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(0.55)
    ax.set_xticks([0, 10, 30, 60])
    ax.set_xticklabels(["0", "10", "30", "60 GB"], fontsize=4.0, color=SLATE)
    ax.tick_params(axis="x", which="both", length=2, color=GRID, pad=1)
    ax.set_yticks([])
    fig.savefig(dst_dir / "vol3_working_sets_margin_003.svg", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)

    # 004: Attention KV Footprint Ladder
    fig, ax = plt.subplots(figsize=(1.25, 1.65), dpi=300)
    tiers = [("MHA", 2560.0, BLUE, "2,560 KB"), ("GQA", 320.0, BLUE, "320 KB"),
             ("MLA", 67.5, CRIMSON, "67.5 KB"), ("MQA", 40.0, BLUE, "40 KB")]
    n = len(tiers)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_xscale("log")
    xmin, xmax = 20.0, 7800.0
    ax.set_xlim(xmin, xmax)
    h = 0.58
    for i, (name, val, col, val_str) in enumerate(tiers):
        y = n - 1 - i
        ax.barh(y, val - xmin, left=xmin, height=h, color=col, alpha=0.92, edgecolor="none")
        if val >= 250:
            ax.text(xmin * 1.15, y, name, color="white", fontsize=5.0, fontweight="bold", ha="left", va="center")
            ax.text(val * 0.88, y, val_str, color="white", fontsize=4.8, fontweight="bold", ha="right", va="center")
        else:
            ax.text(val * 1.25, y, f"{name} {val_str}", color=col, fontsize=4.8, fontweight="bold", ha="left", va="center")
    bx = 3200.0
    ax.plot([bx, bx * 1.2, bx * 1.2, bx], [3.0, 3.0, 1.0, 1.0], color=CRIMSON, lw=0.7)
    ax.text(bx * 1.3, 2.0, "38×", color=CRIMSON, fontsize=5.4, fontweight="bold", ha="left", va="center")
    ax.text(xmax * 0.95, -0.52, "log scale", ha="right", va="bottom", color="#777777", fontsize=4.0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(0.55)
    ax.set_xticks([100, 1000])
    ax.set_xticklabels(["100 KB", "1 MB"], fontsize=4.0, color=SLATE)
    ax.tick_params(axis="x", which="both", length=2, color=GRID, pad=1)
    ax.set_yticks([])
    fig.savefig(dst_dir / "vol3_working_sets_margin_004.svg", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print("[GEN] Ch 04 complete.")

def gen_ch08(dst_dir: Path):
    # 001: Host Syscalls Ladder
    fig = plt.figure(figsize=(1.0, 1.389), dpi=300)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    for s in ["top", "right", "left", "bottom"]: ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.0, 0.96, "HOST SYSCALLS", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(0.0, 0.89, "Ring 0 attack surface", fontsize=4.2, color=SLATE)
    tiers = [("Container (runC)", 450, CRIMSON, "450"),
             ("gVisor (Sentry)", 50, AMBER, "50"),
             ("Firecracker", 35, EMERALD, "35"),
             ("Wasm (WASI)", 0, SLATE, "0")]
    ys = [0.70, 0.49, 0.28, 0.07]
    bar_h = 0.075
    for (name, val, col, val_str), y in zip(tiers, ys):
        ax.text(0.0, y + 0.065, name, fontsize=4.8, fontweight="bold", color=INK)
        ax.text(1.0, y + 0.065, val_str, fontsize=4.8, fontweight="bold", color=col, ha="right")
        ax.barh(y, 1.0, height=bar_h, color="#F1F5F9", left=0.0)
        frac = (val / 450.0) if val > 0 else 0.015
        ax.barh(y, frac, height=bar_h, color=col, left=0.0)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.savefig(dst_dir / "vol3_virtualization_margin_001.svg", format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # 002: Copy-Up Latency Cliff
    fig = plt.figure(figsize=(1.0, 1.389), dpi=300)
    ax = fig.add_axes([0.19, 0.17, 0.77, 0.65])
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.6)
    fig.text(0.05, 0.96, "COPY-UP STALL CLIFF", fontsize=5.8, fontweight="bold", color=INK)
    fig.text(0.05, 0.89, "Single-byte write penalty", fontsize=4.2, color=SLATE)
    x = np.logspace(-2, 3.2, 100)
    y = np.where(x < 50, (x / 3.5), (x / 3.5) * (1 + 31 * ((x - 50) / 1150)))
    ax.axvspan(100, 2000, color=REDFILL, alpha=0.55, lw=0)
    m1 = x <= 10; m2 = (x >= 10) & (x <= 100); m3 = x >= 100
    ax.plot(x[m1], y[m1], color=EMERALD, lw=1.3)
    ax.plot(x[m2], y[m2], color=AMBER, lw=1.3)
    ax.plot(x[m3], y[m3], color=CRIMSON, lw=1.6)
    ax.plot(1200, 10970, "o", color=CRIMSON, ms=3.2)
    ax.text(0.03, 4000, "1.2 GB:\n10.97 s stall\n(32x contention)", fontsize=3.9, fontweight="bold", color=CRIMSON, va="top", ha="left")
    ax.annotate("", xy=(1000, 10000), xytext=(0.6, 2500), arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=0.6))
    ax.plot(1, 0.28, "o", color=EMERALD, ms=2.6)
    ax.text(1.5, 0.08, "1 MB: 0.3 ms", fontsize=3.9, fontweight="bold", color=EMERALD, ha="left", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.01, 2000); ax.set_ylim(0.001, 35000)
    ax.set_xticks([0.1, 10, 1000]); ax.set_xticklabels(["100K", "10M", "1G"], fontsize=3.9, color=SLATE)
    ax.set_yticks([0.01, 1, 100, 10000]); ax.set_yticklabels(["10µs", "1ms", "0.1s", "10s"], fontsize=3.9, color=SLATE)
    ax.tick_params(axis="both", which="both", length=2, width=0.5, color=GRID, pad=1)
    fig.savefig(dst_dir / "vol3_virtualization_margin_002.svg", format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # 003: DNS Covert Channel
    fig = plt.figure(figsize=(1.0, 1.389), dpi=300)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    for s in ["top", "right", "left", "bottom"]: ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.0, 0.96, "DNS COVERT CHANNEL", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(0.0, 0.89, "Exfiltrating 64 KB key", fontsize=4.2, color=SLATE)
    ax.text(0.0, 0.79, "Unrestricted UDP", fontsize=4.8, fontweight="bold", color=CRIMSON)
    ax.text(1.0, 0.79, "234 KB/s", fontsize=4.6, fontweight="bold", color=CRIMSON, ha="right")
    ax.barh(0.65, 0.98, height=0.08, color=CRIMSON, left=0.0)
    ax.text(0.94, 0.65, "0.28 s (sub-second)", fontsize=4.2, fontweight="bold", color="white", ha="right", va="center")
    ax.add_patch(patches.Rectangle((0.02, 0.44), 0.96, 0.11, facecolor="#FEF3C7", edgecolor="#FDE68A", lw=0.6))
    ax.text(0.50, 0.495, "7,800x Bandwidth Collapse", fontsize=4.5, fontweight="bold", color=AMBER, ha="center", va="center")
    ax.text(0.0, 0.33, "Clamped (2 qps)", fontsize=4.8, fontweight="bold", color=EMERALD)
    ax.text(1.0, 0.33, "30 B/s", fontsize=4.6, fontweight="bold", color=EMERALD, ha="right")
    ax.barh(0.19, 0.03, height=0.08, color=EMERALD, left=0.0)
    ax.text(0.08, 0.19, "36.4 min (defensible)", fontsize=4.4, fontweight="bold", color=EMERALD, ha="left", va="center")
    ax.text(0.0, 0.03, "Clamped label (24 char) + 2 qps limit", fontsize=3.7, color=SLATE, style="italic")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.savefig(dst_dir / "vol3_virtualization_margin_003.svg", format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # 004: Sandbox Provisioning
    fig = plt.figure(figsize=(1.0, 1.389), dpi=300)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    for s in ["top", "right", "left", "bottom"]: ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.0, 0.97, "SANDBOX PROVISIONING", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(0.0, 0.90, "P50 acquisition latency", fontsize=4.2, color=SLATE)
    items = [("Cold MicroVM", 450.0, CRIMSON, "450 ms"),
             ("Snapshot Restore", 35.0, AMBER, "35 ms"),
             ("Pre-Warmed (UFFD)", 3.0, EMERALD, "3 ms"),
             ("Container Reuse", 2.0, SLATE, "2 ms*")]
    ys = [0.73, 0.52, 0.31, 0.10]; bar_h = 0.07
    for (name, val, col, val_str), y in zip(items, ys):
        ax.text(0.0, y + 0.062, name, fontsize=4.8, fontweight="bold", color=INK)
        ax.text(1.0, y + 0.062, val_str, fontsize=4.8, fontweight="bold", color=col, ha="right")
        log_pos = 0.06 + 0.92 * (np.log10(val) / np.log10(500))
        ax.barh(y, 1.0, height=bar_h, color="#F1F5F9", left=0.0)
        ax.barh(y, log_pos, height=bar_h, color=col, left=0.0)
    ax.text(0.0, 0.015, "*Compromised: state leak (A > 0)", fontsize=3.6, color=CRIMSON, style="italic")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.savefig(dst_dir / "vol3_virtualization_margin_004.svg", format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("[GEN] Ch 08 complete.")

def gen_ch13(dst_dir: Path):
    # 001: Hierarchy Ladder
    fig, ax = plt.subplots(figsize=(1.25, 1.70), dpi=300)
    tiers = [("16k: In-Context", 16384), ("2k: Schema", 2048),
             ("512: Minified", 512), ("128: SFT Policy", 128)]
    n = len(tiers)
    xmin, xmax = 40.0, 38000.0
    ax.set_xscale("log"); ax.set_xlim(xmin, xmax); ax.set_ylim(-0.65, n - 0.35)
    h = 0.62
    for i, (lab, v) in enumerate(tiers):
        yy = n - 1 - i
        ax.barh(yy, v - xmin, left=xmin, height=h, color=BLUE, alpha=0.92)
        if i == 0:
            ax.text(v * 0.85, yy, lab, fontsize=4.9, va="center", ha="right", color="white", fontweight="bold")
        else:
            ax.text(v * 1.35, yy, lab, fontsize=4.9, va="center", ha="left", color=INK, fontweight="bold")
    ax.plot([xmin, xmax], [n - 0.42, n - 0.42], color=RED, lw=0.75)
    ax.text(xmin * 1.05, n - 0.28, "16k prompt ceiling", color=RED, fontsize=4.4, fontweight="bold")
    ax.text(xmax * 0.95, -0.52, "log tokens", ha="right", va="bottom", color="#777777", fontsize=4.0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(0.55)
    ax.set_yticks([]); ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    fig.savefig(dst_dir / "vol3_sft_margin_001.svg", format="svg", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)

    # 002: Packed Varlen Efficiency Bar
    fig, ax = plt.subplots(figsize=(1.35, 1.05), dpi=300)
    y_packed, y_naive, h = 1.0, 0.0, 0.42
    ax.barh(y_packed, 98.0, height=h, color=EMERALD, alpha=0.95, edgecolor="white", lw=0.3)
    ax.barh(y_packed, 2.0, left=98.0, height=h, color=GRID, alpha=0.7, edgecolor="white", lw=0.3)
    ax.text(49.0, y_packed, "98% active", ha="center", va="center", color="white", fontsize=5.0, fontweight="bold")
    ax.text(-4, y_packed, "Packed", ha="right", va="center", color=INK, fontsize=5.0, fontweight="bold")
    ax.barh(y_naive, 19.5, height=h, color=EMERALD, alpha=0.95, edgecolor="white", lw=0.3)
    ax.barh(y_naive, 80.5, left=19.5, height=h, color=REDFILL, alpha=0.65, edgecolor="white", lw=0.3)
    ax.text(9.7, y_naive, "20%", ha="center", va="center", color="white", fontsize=4.6, fontweight="bold")
    ax.text(60.0, y_naive, "80% pad waste", ha="center", va="center", color=RED, fontsize=4.6, fontweight="bold")
    ax.text(-4, y_naive, "Naive", ha="right", va="center", color=INK, fontsize=5.0, fontweight="bold")
    ax.annotate("", xy=(104, y_packed), xytext=(104, y_naive), arrowprops=dict(arrowstyle="<->", color=CRIMSON, lw=0.75))
    ax.text(108, 0.5, "5×\nFLOPs", ha="left", va="center", color=CRIMSON, fontsize=4.8, fontweight="bold")
    ax.set_xlim(-26, 128); ax.set_ylim(-0.55, 1.55)
    for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(dst_dir / "vol3_sft_margin_002.svg", format="svg", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)

    # 003: LoRA Rank Knee
    fig, ax = plt.subplots(figsize=(1.30, 1.05), dpi=300)
    import matplotlib.ticker as ticker
    r_smooth = np.geomspace(2, 64, 150)
    fail_smooth = 2.5 + (62.0 - 2.5) / (1.0 + (r_smooth / 9.5)**3.2)
    ax.axvspan(2, 16, color=REDFILL, alpha=0.35, lw=0)
    m_danger = r_smooth <= 16
    ax.plot(r_smooth[m_danger], fail_smooth[m_danger], color=RED, lw=1.35)
    ax.plot(r_smooth[~m_danger], fail_smooth[~m_danger], color=EMERALD, lw=1.35)
    ax.plot(16, 4.2, "o", color=CRIMSON, ms=3.5)
    ax.axvline(16, color=GRID, ls="--", lw=0.55)
    ax.text(2.6, 22.0, "Syntax\nCollapse", ha="left", va="center", color=RED, fontsize=4.4, fontweight="bold")
    ax.text(20.0, 16.0, "r*=16\nKnee", ha="left", va="bottom", color=CRIMSON, fontsize=4.7, fontweight="bold")
    ax.text(45.0, 38.0, "Robust\nSchema", ha="center", va="center", color=EMERALD, fontsize=4.4, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(2, 64); ax.set_ylim(0, 70)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.55)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks([4, 16, 64]); ax.set_xticklabels(["r=4", "16", "64"], fontsize=4.4, color=INK)
    ax.set_yticks([0, 35, 70]); ax.set_yticklabels(["0%", "35%", "70%"], fontsize=4.4, color=INK)
    ax.tick_params(axis="both", which="both", length=2, color=GRID, pad=1)
    fig.savefig(dst_dir / "vol3_sft_margin_003.svg", format="svg", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)

    # 004: Quadratic Compounding Drift Sparkline
    fig, ax = plt.subplots(figsize=(1.30, 0.95), dpi=300)
    H = np.linspace(0, 30, 100); eps = 0.02
    err_bc = eps * H * (H + 1) / 2
    err_dagger = eps * H
    ax.plot(H, err_bc, color=RED, lw=1.35)
    ax.plot(H, err_dagger, color=EMERALD, lw=1.35)
    ax.fill_between(H, err_dagger, err_bc, color=REDFILL, alpha=0.35)
    ax.plot(30, err_bc[-1], "o", color=RED, ms=3.2)
    ax.plot(30, err_dagger[-1], "o", color=EMERALD, ms=3.2)
    ax.text(28, 10.2, "O(eH²)\n9.3 err", color=RED, fontsize=4.5, fontweight="bold", ha="right", va="bottom")
    ax.text(28, 1.2, "O(eH)\n0.6 err", color=EMERALD, fontsize=4.5, fontweight="bold", ha="right", va="bottom")
    ax.text(6.0, 7.2, "15× Drift\nGap", color=RED, fontsize=4.4, fontweight="bold", ha="left", va="center")
    ax.set_xlim(0, 32); ax.set_ylim(-0.4, 12.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.55)
    ax.set_xticks([0, 15, 30]); ax.set_xticklabels(["0", "15", "H=30"], fontsize=4.4, color=INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", length=2, color=GRID, pad=1)
    fig.savefig(dst_dir / "vol3_sft_margin_004.svg", format="svg", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print("[GEN] Ch 13 complete.")

def gen_ch15(dst_dir: Path):
    # 001: The Delegation Trade-Off
    fig, ax = plt.subplots(figsize=(1.05, 1.42), dpi=300)
    y_pos = np.array([1.30, 0.40]); h = 0.26
    ax.axvline(1.0, color=SLATE, ls="--", lw=0.7, zorder=1)
    ax.axvspan(0, 1.0, color="#D5F5E3", alpha=0.35, zorder=0)
    ax.axvspan(1.0, 2.45, color=REDFILL, alpha=0.30, zorder=0)
    ax.text(1.0, 2.08, "1-Agent Baseline", color=SLATE, fontsize=4.2, ha="center",
            fontweight="bold", bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none"))
    ax.barh(y_pos[0], 0.482, height=h, color=EMERALD, edgecolor="none", zorder=2)
    ax.text(0.482 / 2, y_pos[0], "0.48x\n(311s)", color="white", fontsize=4.0,
            fontweight="bold", va="center", ha="center", zorder=3)
    ax.text(0.24, y_pos[0] + 0.22, "2.07x speedup", color=EMERALD, fontsize=4.1,
            fontweight="bold", va="center", ha="center")
    ax.barh(y_pos[1], 2.197, height=h, color=CRIMSON, edgecolor="none", zorder=2)
    ax.text(1.0 + (2.197 - 1.0) / 2, y_pos[1], "2.20x\n(118k tok)", color="white", fontsize=4.0,
            fontweight="bold", va="center", ha="center", zorder=3)
    ax.text(1.70, y_pos[1] + 0.22, "+120% Token Tax", color=CRIMSON, fontsize=4.1,
            fontweight="bold", va="center", ha="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["Makespan\n(Wall-clock)", "Token Cost\n(Compute)"],
                       fontsize=4.6, fontweight="bold", color=INK)
    ax.set_xlim(0, 2.45); ax.set_ylim(-0.15, 2.30)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); ax.set_xticklabels(["0", "0.5x", "1.0x", "1.5x", "2.0x"], fontsize=4.3)
    ax.set_xlabel("Relative to Single Agent", fontsize=4.6, labelpad=2, color=INK)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(left=False, colors=SLATE, length=2, width=0.5, pad=1)
    fig.savefig(dst_dir / "vol3_multi_agent_margin_001.svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)

    # 002: OCC Write Collision Inevitability
    fig, ax = plt.subplots(figsize=(1.05, 1.42), dpi=300)
    N = np.linspace(1, 10, 100)
    p_uniform = 1.0 - np.exp(-N * (N - 1) * 0.009)
    p_zipf = 1.0 - np.exp(-N * (N - 1) * 0.055)
    ax.fill_between(N, 0.5, 1.0, color=REDFILL, alpha=0.28, zorder=1)
    ax.axhline(0.5, color=CRIMSON, ls=":", lw=0.6, alpha=0.7)
    ax.plot(N, p_uniform, color=SLATE, lw=1.2, ls="--")
    ax.plot(N, p_zipf, color=CRIMSON, lw=1.5)
    ax.text(1.5, 0.86, "Zipf Hotspot", color=CRIMSON, fontsize=4.6, fontweight="bold")
    ax.text(1.3, 0.53, "50% limit", color=CRIMSON, fontsize=4.0, fontweight="bold", ha="left")
    ax.text(8.0, 0.15, "Uniform", color=SLATE, fontsize=4.3, ha="center")
    p_4 = 1.0 - np.exp(-4 * 3 * 0.055)
    p_8 = 1.0 - np.exp(-8 * 7 * 0.055)
    ax.plot(4, p_4, "o", color=AMBER, ms=2.8, zorder=4)
    ax.plot(8, p_8, "o", color=CRIMSON, ms=2.8, zorder=4)
    ax.text(4.4, 0.38, "N=4: 48%", color=AMBER, fontsize=4.2, fontweight="bold")
    ax.text(8.0, 1.01, "N=8: 95%", color=CRIMSON, fontsize=4.2, fontweight="bold", ha="center")
    ax.set_xlim(1, 10); ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 4, 7, 10]); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_xlabel("Worker Fleet Size (N)", fontsize=4.8, labelpad=2, color=INK)
    ax.set_ylabel("P(Write Collision)", fontsize=4.8, labelpad=2, color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=SLATE, length=2, width=0.5, pad=1)
    fig.savefig(dst_dir / "vol3_multi_agent_margin_002.svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)

    # 003: Monotonic Capability Attenuation Ladder
    fig, ax = plt.subplots(figsize=(1.05, 1.42), dpi=300)
    tiers = [("d=0 Root", 100, "100k · Root /", EMERALD),
             ("d=1 Plan", 40, "40k · /src", SLATE),
             ("d=2 Code", 20, "20k · /auth", AMBER),
             ("d=3 Test", 5, "5k · pytest", CRIMSON)]
    y_pos = np.array([3, 2, 1, 0])
    widths = [t[1] for t in tiers]; colors = [t[3] for t in tiers]
    ax.barh(y_pos, [100]*4, height=0.55, color="#E2E8F0", alpha=0.4, edgecolor="none", zorder=1)
    ax.barh(y_pos, widths, height=0.55, color=colors, edgecolor="none", zorder=2)
    for i, (label, val, desc, col) in enumerate(tiers):
        y = y_pos[i]
        if val >= 40:
            ax.text(val - 3, y, desc, color="white", fontsize=4.3, fontweight="bold", va="center", ha="right", zorder=3)
        else:
            ax.text(val + 3, y, desc, color=col, fontsize=4.3, fontweight="bold", va="center", ha="left", zorder=3)
    ax.text(50, 3.82, r"$C_0 \supseteq C_1 \supseteq C_2 \supseteq C_3$", color=INK, fontsize=5.0, fontweight="bold", ha="center")
    ax.text(50, -0.72, "HMAC cost: 1.95 µs · 256 B", color=SLATE, fontsize=3.9, ha="center")
    ax.set_yticks(y_pos); ax.set_yticklabels([t[0] for t in tiers], fontsize=4.6, fontweight="bold", color=INK)
    ax.set_xlim(0, 105); ax.set_ylim(-1.0, 4.25)
    ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0", "50k", "100k"], fontsize=4.4)
    ax.set_xlabel("Token Budget Ceiling", fontsize=4.6, labelpad=1, color=INK)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(left=False, colors=SLATE, length=2, width=0.5, pad=1)
    fig.savefig(dst_dir / "vol3_multi_agent_margin_003.svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)

    # 004: Gunther USL Retrograde Speedup
    fig, ax = plt.subplots(figsize=(1.05, 1.42), dpi=300)
    M = np.linspace(1, 10, 100); f, sigma, kappa = 0.70, 0.04, 0.02
    denom = (1.0 - f) + (f / M) + sigma * (M - 1) + kappa * M * (M - 1)
    s_usl = 1.0 / denom
    idx_peak = np.argmax(s_usl); m_peak = M[idx_peak]; s_peak = s_usl[idx_peak]
    ax.axhline(1.0, color=SLATE, ls="--", lw=0.6, zorder=1)
    ax.text(9.8, 1.05, "1.0x Parity", color=SLATE, fontsize=3.9, ha="right")
    ax.fill_between(M, 0, 1.0, where=(M >= 5.0), color=REDFILL, alpha=0.30, zorder=1)
    mask_pre = M <= m_peak; mask_post = M >= m_peak
    ax.plot(M[mask_pre], s_usl[mask_pre], color=AMBER, lw=1.5)
    ax.plot(M[mask_post], s_usl[mask_post], color=CRIMSON, lw=1.5)
    ax.plot(m_peak, s_peak, "o", color=AMBER, ms=2.8, zorder=4)
    ax.text(m_peak + 0.6, s_peak + 0.09, f"Peak {s_peak:.2f}x\n(M=2.5)", color=AMBER,
            fontsize=4.0, fontweight="bold", ha="center")
    s_8 = 1.0 / ((1 - f) + f / 8.0 + sigma * 7 + kappa * 8 * 7)
    ax.plot(8, s_8, "o", color=CRIMSON, ms=2.8, zorder=4)
    ax.text(8.0, 0.22, "M=8: 0.56x\n(2x slower)", color=CRIMSON, fontsize=4.0, fontweight="bold", ha="center")
    ax.text(3.0, 0.40, "Retrograde\nTax Zone", color=CRIMSON, fontsize=4.2, ha="center")
    ax.set_xlim(1, 10); ax.set_ylim(0, 1.75)
    ax.set_xticks([1, 4, 6, 8, 10]); ax.set_yticks([0, 0.5, 1.0, 1.5]); ax.set_yticklabels(["0x", "0.5x", "1.0x", "1.5x"])
    ax.set_xlabel("Worker Fleet Size (M)", fontsize=4.8, labelpad=2, color=INK)
    ax.set_ylabel("Parallel Speedup S(M)", fontsize=4.8, labelpad=2, color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=SLATE, length=2, width=0.5, pad=1)
    fig.savefig(dst_dir / "vol3_multi_agent_margin_004.svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print("[GEN] Ch 15 complete.")

def main():
    print("=== Step 1: Copying existing verified SVGs ===")
    copy_pregenerated()

    print("\n=== Step 2: Generating missing chapter SVGs ===")
    ch_generators = {
        1: gen_ch01,
        2: gen_ch02,
        4: gen_ch04,
        8: gen_ch08,
        13: gen_ch13,
        15: gen_ch15,
    }

    for ch_num, gen_func in ch_generators.items():
        ch_dir, _ = CHAPTER_MAP[ch_num]
        dst_dir = BASE_DIR / "books" / "vol3" / ch_dir / "images" / "svg"
        dst_dir.mkdir(parents=True, exist_ok=True)
        gen_func(dst_dir)

    print("\n=== Step 3: Auditing all 72 micro-SVGs ===")
    missing = []
    total_count = 0
    for ch_num in range(1, 19):
        ch_dir, prefix = CHAPTER_MAP[ch_num]
        dst_dir = BASE_DIR / "books" / "vol3" / ch_dir / "images" / "svg"
        for idx in range(1, 5):
            target = dst_dir / f"{prefix}_margin_00{idx}.svg"
            if target.exists():
                total_count += 1
            else:
                missing.append(str(target))

    print(f"Total micro-SVGs present: {total_count} / 72")
    if missing:
        print(f"WARNING: Missing {len(missing)} files:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("ALL 72 MICRO-SVGS ARE IN PLACE!")

if __name__ == "__main__":
    main()
