"""
Generate Chapter 3 Plot: Longitudinal VLA Frontier
Parameter count vs onboard physical action frequency (2020-2026).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add current dir to path to import plot_style
sys.path.append(os.path.dirname(__file__))
from plot_style import PALETTE, ERA_COLORS, setup_canvas, add_badge

def generate_plot():
    fig, ax = setup_canvas(figsize=(14, 8.5), dpi=300)
    
    # ---------------------------------------------------------
    # 1. Background Shaded Regimes & Memory Walls
    # ---------------------------------------------------------
    # Autoregressive Deliberation Trap (Lower region, 0.35 Hz to 8 Hz)
    ax.axhspan(0.35, 8.0, color="#FEF2F2", alpha=0.55, zorder=0)
    # Action Chunking & Flow Matching Frontier (Top region, 25 Hz to 200 Hz)
    ax.axhspan(25.0, 200.0, color="#ECFDF5", alpha=0.5, zorder=0)
    
    # Regime Annotations - positioned where there is zero collision
    ax.text(2.2e7, 180.0, "CONTINUOUS ACTION CHUNKING & FLOW MATCHING FRONTIER (2023–2026)\nDecoupled proposal generation: multi-step chunks ($H = 16\\text{--}64$) + fast generative decoders\nMulti-billion parameter foundation models achieve real-time 50–100 Hz closed-loop control",
            fontsize=8.5, fontweight="bold", color="#047857", alpha=0.9, va="top")
    
    ax.text(2.2e8, 0.52, "AUTOREGRESSIVE DELIBERATION WALL (2020–2023)\nSingle-token weight streaming bound by DRAM bandwidth ($f \\leq B_{\\mathrm{mem}} / M_{\\mathrm{weights}}$)\nEach action step requires a full model pass -> catastrophic latency scaling",
            fontsize=8.5, fontweight="bold", color="#B91C1C", alpha=0.85, va="center")

    # ---------------------------------------------------------
    # 2. Key Physical Frequency Deadlines
    # ---------------------------------------------------------
    deadlines = [
        (50.0, "50 Hz: Real-Time Dynamic Manipulation & Tactile Contact Deadline", PALETTE["forest_green"], "--"),
        (10.0, "10 Hz: Coarse Visual Servoing & Quasistatic Pick-and-Place Limit", PALETTE["amber"], ":"),
        (2.0,  "2 Hz: Cloud VLM Autoregressive Latency Floor (0.5–2 s)", PALETTE["crimson"], "-."),
    ]
    for freq, lbl, col, ls in deadlines:
        ax.axhline(freq, color=col, linestyle=ls, linewidth=1.2, alpha=0.7, zorder=1)
        ax.text(9.5e10, freq * 1.08, lbl, fontsize=7.5, fontweight="bold", color=col, ha="right", va="bottom")

    # ---------------------------------------------------------
    # 3. Frontier Trajectory Curves
    # ---------------------------------------------------------
    # Autoregressive memory wall trend curve (scaled between RT-1 and RT-2-PaLI-X)
    ar_params = np.logspace(np.log10(3.5e7), np.log10(5.5e10), 100)
    ar_freq = 3.0 * (3.5e7 / ar_params)**0.15
    ax.plot(ar_params, ar_freq, color=PALETTE["crimson"], lw=2.2, ls="--", zorder=4, alpha=0.8)
    ax.text(1.2e9, 1.8, "Autoregressive Memory Wall: $f \\propto 1 / M_{\\mathrm{weights}}$",
            fontsize=8.5, fontweight="bold", color=PALETTE["crimson"], rotation=-6, ha="center")

    # Chunking / Flow-matching frontier line
    fx = np.array([8e7, 5e8, 2e9, 3.3e9, 5e9, 1.5e10])
    fy = np.array([50.0, 50.0, 50.0, 52.0, 80.0, 100.0])
    ax.plot(fx, fy, color=PALETTE["forest_green"], lw=2.5, ls="-", zorder=4, alpha=0.85)
    ax.text(1.8e8, 56.0, "Action Chunking Frontier ($H \\geq 16$, Flow Matching)",
            fontsize=8.5, fontweight="bold", color=PALETTE["forest_green"], ha="left", va="bottom")

    # ---------------------------------------------------------
    # 4. Empirical VLA Model Data Points
    # ---------------------------------------------------------
    models = [
        # Autoregressive / Early Models
        {
            "name": "BC-Z (2022)",
            "params": 1e8, "freq": 10.0, "year": 2022,
            "color": PALETTE["slate"], "marker": "s",
            "badge_xytext": (2.2e7, 14.0),
            "text": "2022 · Google BC-Z\n100M params · 10 Hz\nResNet behavioral cloning",
        },
        {
            "name": "RT-1 (2022)",
            "params": 3.5e7, "freq": 3.0, "year": 2022,
            "color": PALETTE["dark_blue"], "marker": "o",
            "badge_xytext": (1.8e7, 4.8),
            "text": "2022 · Google RT-1\n35M params · 3 Hz (330 ms)\nEfficientNet + TokenLearner",
        },
        {
            "name": "RT-2-PaLM-E (2023)",
            "params": 1.2e10, "freq": 3.0, "year": 2023,
            "color": PALETTE["gold"], "marker": "D",
            "badge_xytext": (4.0e9, 4.5),
            "text": "2023 · RT-2-PaLM-E (12B)\n12B params · ~3 Hz\nDiscrete action tokenization",
        },
        {
            "name": "RT-2-PaLI-X (2023)",
            "params": 5.5e10, "freq": 1.0, "year": 2023,
            "color": PALETTE["crimson"], "marker": "D",
            "badge_xytext": (2.2e10, 1.8),
            "text": "2023 · RT-2-PaLI-X (55B)\n55B params · ~1 Hz (1,000 ms)\nCloud TPU v4 multi-token sweep",
        },
        {
            "name": "OpenVLA (Base, 2024)",
            "params": 7e9, "freq": 6.0, "year": 2024,
            "color": PALETTE["coral"], "marker": "o",
            "badge_xytext": (1.5e9, 7.5),
            "text": "2024 · OpenVLA (Base)\n7B params · 6 Hz (RTX 4090)\nLlama-2 + SigLIP discrete bins",
        },

        # Action Chunking / Diffusion / Flow Matching Frontier
        {
            "name": "ACT (Zhao et al., 2023)",
            "params": 8e7, "freq": 50.0, "year": 2023,
            "color": PALETTE["teal"], "marker": "^",
            "badge_xytext": (2.0e7, 72.0),
            "text": "2023 · Stanford ACT\n80M params · 50 Hz actions\nAction Chunking Transformer ($H=100$)",
        },
        {
            "name": "Diffusion Policy (2023)",
            "params": 1e8, "freq": 30.0, "year": 2023,
            "color": PALETTE["purple"], "marker": "^",
            "badge_xytext": (1.8e8, 23.0),
            "text": "2023 · Diffusion Policy\n100M params · 30 Hz execution\nReceding-horizon DDPM ($H=16$)",
        },
        {
            "name": "Octo-Base (2023)",
            "params": 9.3e7, "freq": 10.0, "year": 2023,
            "color": PALETTE["cyan"], "marker": "s",
            "badge_xytext": (2.2e8, 14.0),
            "text": "2023 · Octo-Base (UC Berkeley)\n93M params · 10 Hz actions\nTransformer diffusion policy",
        },
        {
            "name": "OpenVLA-OFT+ (2024)",
            "params": 7e9, "freq": 50.0, "year": 2024,
            "color": PALETTE["amber"], "marker": "*",
            "badge_xytext": (4.5e9, 25.0),
            "text": "2024 · OpenVLA-OFT+\n7B params · 50 Hz actions\nChunked parallel decoding",
        },
        {
            "name": "Physical Intelligence pi0 (2024)",
            "params": 3.3e9, "freq": 50.0, "year": 2024,
            "color": PALETTE["forest_green"], "marker": "*",
            "badge_xytext": (1.4e9, 82.0),
            "text": "2024 · Physical Intelligence π₀\n3.3B params · 50 Hz actions\nFlow Matching Action Expert",
        },
        {
            "name": "NVIDIA Project GR00T (2024–25)",
            "params": 5e9, "freq": 80.0, "year": 2025,
            "color": PALETTE["dark_blue"], "marker": "*",
            "badge_xytext": (3.5e9, 140.0),
            "text": "2024–25 · NVIDIA Project GR00T\n~5B params · 50–100 Hz closed-loop\nDual-System slow VLM + fast DiT policy",
        },
        {
            "name": "2026 Frontier (Dual-Brain SoCs)",
            "params": 1.5e10, "freq": 100.0, "year": 2026,
            "color": PALETTE["crimson"], "marker": "*",
            "badge_xytext": (2.6e10, 140.0),
            "text": "2026 Frontier · Embodied Dual-Brain\n15B params · 100 Hz actuation\nHardware-isolated proposal-permission SoC",
        },
    ]

    # Plot points
    for m in models:
        s_size = 200 if m["marker"] == "*" else (110 if m["marker"] == "D" else 100)
        ax.scatter(m["params"], m["freq"], color=m["color"], s=s_size,
                   marker=m["marker"], edgecolor="#FFFFFF", linewidth=1.5, zorder=10)
        add_badge(ax, m["text"], xy=(m["params"], m["freq"]), xytext=m["badge_xytext"],
                  color=PALETTE["charcoal"], fontsize=7.8, arrow=True, arrow_color=m["color"])

    # ---------------------------------------------------------
    # 5. Axes, Scales, Titles, and Legends
    # ---------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlim(1.5e7, 1.2e11)
    ax.set_ylim(0.35, 210)
    
    # Custom Ticks
    ax.set_xticks([2e7, 1e8, 1e9, 1e10, 1e11])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{int(x/1e6)}M" if x < 1e9 else f"{int(x/1e9)}B"
    ))
    
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(
        lambda y, p: f"{y:g} Hz"
    ))
    
    ax.set_xlabel("Foundation Policy Parameter Count $M_{\\mathrm{weights}}$ (log scale)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Onboard Physical Action Frequency (Hz, log scale)", fontsize=11, fontweight="bold", labelpad=8)
    
    # Title block
    fig.suptitle("The Embodied Brain Frontier: Parameter Scaling vs Physical Action Frequency (2020–2026)",
                 fontsize=14, fontweight="bold", x=0.08, y=0.98, ha="left", color=PALETTE["charcoal"])
    ax.set_title("Overcoming the deliberation wall: how continuous action chunking and flow matching break the memory wall ($1\\,\\mathrm{Hz} \\rightarrow 100\\,\\mathrm{Hz}$)",
                 fontsize=9.5, color="#475569", loc="left", pad=14)
    
    # Legend - placed at bottom left
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['crimson'], markersize=8, label='Discrete Autoregressive Tokenization (Single-Step)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=PALETTE['teal'], markersize=8, label='Diffusion Action Chunking (ACT / DDPM)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=PALETTE['forest_green'], markersize=12, label='Continuous Flow Matching / Dual-System VLA (π₀, GR00T)'),
        Line2D([0], [0], color=PALETTE['crimson'], lw=2, ls='--', label='Autoregressive Memory Wall Trend ($f \\propto 1/M$)'),
        Line2D([0], [0], color=PALETTE['forest_green'], lw=2, ls='-', label='Action Chunking Scaling Frontier'),
    ]
    legend = ax.legend(handles=legend_elements, loc="lower left", frameon=True, framealpha=0.96,
                       edgecolor="#CBD5E1", fontsize=7.8, title="Embodied Policy Architectures",
                       title_fontsize=8.2)
    legend.get_title().set_fontweight("bold")
    legend.set_zorder(15)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.94)
    
    out_path = os.path.join(os.path.dirname(__file__), "ch03_vla_scaling_timeline.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_plot()
