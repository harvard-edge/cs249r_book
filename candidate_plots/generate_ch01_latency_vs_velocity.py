"""
Generate Chapter 1 Plot: The Causal Boundary
Sense-to-act loop latency vs safe vehicle operational speed envelope (1970-2026).
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
    fig, ax = setup_canvas(figsize=(14, 8), dpi=300)
    
    # ---------------------------------------------------------
    # 1. Background Era Shading (Vertical Latency Regimes)
    # ---------------------------------------------------------
    # Latencies from 10 ms to 60,000 ms
    # Shakey era (10 s - 60 s)
    ax.axvspan(5000, 60000, color=ERA_COLORS[0], alpha=0.6, zorder=0)
    # Early neural / mobile era (500 ms - 5000 ms)
    ax.axvspan(400, 5000, color=ERA_COLORS[1], alpha=0.5, zorder=0)
    # DARPA & early autonomy era (70 ms - 400 ms)
    ax.axvspan(50, 400, color=ERA_COLORS[2], alpha=0.4, zorder=0)
    # Modern real-time autonomy & racing era (10 ms - 50 ms)
    ax.axvspan(10, 50, color=ERA_COLORS[3], alpha=0.5, zorder=0)
    
    # Era Labels at top (in the dedicated top margin)
    era_y = 450
    ax.text(25000, era_y, "Deliberative Era\n(1970–1980)", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="#64748B", style="italic")
    ax.text(2000, era_y, "Early Vision & Systolic\n(1985–1995)", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="#64748B", style="italic")
    ax.text(180, era_y, "DARPA Grand Challenges\n(2004–2010)", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="#64748B", style="italic")
    ax.text(23, era_y, "High-Speed Real-Time\n(2020–2026)", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="#64748B", style="italic")

    # ---------------------------------------------------------
    # 2. Kinematic Envelope Curves: d_stop = v*tau + v^2/(2*a) <= R
    # ---------------------------------------------------------
    # Let a = 6.86 m/s^2 (0.7 g, dry asphalt emergency stop)
    a = 6.86
    tau_ms = np.logspace(np.log10(10), np.log10(60000), 500)
    tau_s = tau_ms / 1000.0
    
    horizons = [
        (200, "Sensor Horizon R = 200 m (Highway LiDAR/Radar)", PALETTE["crimson"], "-"),
        (100, "R = 100 m (Suburban / Rural)", PALETTE["amber"], "--"),
        (30,  "R = 30 m (Urban Occlusion / Crosswalk)", PALETTE["teal"], "-."),
        (5,   "R = 5 m (Indoor Warehouse / Hallway)", PALETTE["purple"], ":")
    ]
    
    for R, label, col, ls in horizons:
        # v_max = sqrt((a*tau)^2 + 2*a*R) - a*tau in m/s
        v_max_ms = np.sqrt((a * tau_s)**2 + 2 * a * R) - a * tau_s
        v_max_kmh = v_max_ms * 3.6
        ax.plot(tau_ms, v_max_kmh, linestyle=ls, color=col, linewidth=1.8, alpha=0.85, label=label, zorder=2)
        
    # Shading unsafe zone above R = 200 m
    v_max_200 = (np.sqrt((a * tau_s)**2 + 2 * a * 200) - a * tau_s) * 3.6
    ax.fill_between(tau_ms, v_max_200, 520, color="#FEE2E2", alpha=0.35, zorder=1)
    ax.text(12000, 180, "PHYSICALLY UNRECOVERABLE REGIME\n(Stopping Distance > 200 m Sensor Horizon)",
            fontsize=8.8, fontweight="bold", color=PALETTE["crimson"], alpha=0.75, rotation=-14, ha="center")

    # Blind Travel Iso-Lines: d_blind = v * tau
    # v_kmh = (d_blind / tau_s) * 3.6
    for d_blind, txt_pos, rot in [(20, 1600, -28)]:
        v_blind_kmh = (d_blind / tau_s) * 3.6
        mask = (v_blind_kmh >= 0.1) & (v_blind_kmh <= 500)
        ax.plot(tau_ms[mask], v_blind_kmh[mask], color="#94A3B8", linestyle=":", linewidth=1.0, alpha=0.6, zorder=2)
        ax.text(txt_pos, (d_blind / (txt_pos / 1000.0)) * 3.6 * 1.15, f"Blind Travel d = {d_blind}m",
                fontsize=7.5, color="#64748B", rotation=rot, alpha=0.85)

    # ---------------------------------------------------------
    # 3. Empirical Milestones (Data Points & Badges)
    # ---------------------------------------------------------
    milestones = [
        {
            "name": "SRI Shakey (1970)",
            "tau": 30000,
            "v": 0.18,
            "badge_xytext": (14000, 0.45),
            "text": "1970 · SRI Shakey\nτ ≈ 30 s · v = 0.18 km/h\nSTRIPS stop-and-plan",
            "color": PALETTE["purple"],
            "marker": "o",
        },
        {
            "name": "Stanford Cart (1979)",
            "tau": 15000,
            "v": 0.10,
            "badge_xytext": (4500, 0.20),
            "text": "1979 · Stanford Cart (Moravec)\nτ ≈ 15 s · v = 0.10 km/h\nStereo obstacle mapping",
            "color": PALETTE["slate"],
            "marker": "s",
        },
        {
            "name": "CMU Navlab 1 (1986)",
            "tau": 1500,
            "v": 15.0,
            "badge_xytext": (2800, 6.5),
            "text": "1986 · CMU Navlab 1\nτ ≈ 1.5 s · v = 15 km/h\nALVINN neural net on WARP",
            "color": PALETTE["amber"],
            "marker": "o",
        },
        {
            "name": "UniBwM VaMoRs (1987)",
            "tau": 40,
            "v": 96.0,
            "badge_xytext": (16, 50),
            "text": "1987 · Dickmanns VaMoRs\nτ = 40 ms (25 Hz) · v = 96 km/h\nAutobahn 4D dynamic vision",
            "color": PALETTE["teal"],
            "marker": "^",
        },
        {
            "name": "CMU Navlab 5 (1995)",
            "tau": 120,
            "v": 95.0,
            "badge_xytext": (230, 85),
            "text": "1995 · CMU Navlab 5\nτ ≈ 120 ms · v = 95 km/h\nNo Hands Across America",
            "color": PALETTE["gold"],
            "marker": "s",
        },
        {
            "name": "Stanford Stanley (2005)",
            "tau": 100,
            "v": 60.0,
            "badge_xytext": (40, 20),
            "text": "2005 · Stanford Stanley\nτ ≈ 100 ms (10 Hz) · v = 60 km/h\nDARPA Grand Challenge winner",
            "color": PALETTE["dark_blue"],
            "marker": "D",
        },
        {
            "name": "CMU Tartan Boss (2007)",
            "tau": 85,
            "v": 48.0,
            "badge_xytext": (150, 18),
            "text": "2007 · CMU Boss (Urban Challenge)\nτ ≈ 85 ms · v = 48 km/h\nUrban intersection autonomy",
            "color": PALETTE["forest_green"],
            "marker": "D",
        },
        {
            "name": "Waymo Driver (2024)",
            "tau": 80,
            "v": 110.0,
            "badge_xytext": (280, 220),
            "text": "2024 · Waymo Driver (Gen 5/6)\nτ ≈ 80 ms · v = 110 km/h (68 mph)\nCommercial highway robotaxi fleet",
            "color": PALETTE["cyan"],
            "marker": "o",
        },
        {
            "name": "Indy Autonomous (2024)",
            "tau": 28,
            "v": 309.3,
            "badge_xytext": (13, 190),
            "text": "2024 · Indy Autonomous (PoliMOVE)\nτ ≈ 28 ms (35+ Hz) · v = 309 km/h (192 mph)\nAV-24 oval racing world record",
            "color": PALETTE["crimson"],
            "marker": "*",
        },
    ]

    for m in milestones:
        # Plot point
        ax.scatter(m["tau"], m["v"], color=m["color"], s=110 if m["marker"] != "*" else 220,
                   marker=m["marker"], edgecolor="#FFFFFF", linewidth=1.5, zorder=12)
        # Add badge annotation
        add_badge(ax, m["text"], xy=(m["tau"], m["v"]), xytext=m["badge_xytext"],
                  color=PALETTE["charcoal"], fontsize=8.2, arrow=True, arrow_color=m["color"])

    # ---------------------------------------------------------
    # 4. Axes, Scales, Titles, and Legends
    # ---------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlim(10, 65000)
    ax.set_ylim(0.06, 520)
    
    # Custom Ticks
    ax.set_xticks([10, 30, 100, 300, 1000, 3000, 10000, 30000, 60000])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{int(x)} ms" if x < 1000 else f"{int(x/1000)} s"
    ))
    
    ax.set_yticks([0.1, 0.5, 1, 5, 15, 50, 100, 200, 350])
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(
        lambda y, p: f"{y:g} km/h"
    ))
    
    ax.set_xlabel("Sense-to-Act Loop Latency $\\tau_{\\mathrm{delay}}$ (log scale)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Vehicle Operational Speed Envelope $v$ (km/h, log scale)", fontsize=11, fontweight="bold", labelpad=8)
    
    # Title block
    fig.suptitle("The Causal Boundary: Sense-to-Act Latency vs Safe Operating Speed (1970–2026)",
                 fontsize=14, fontweight="bold", x=0.08, y=0.98, ha="left", color=PALETTE["charcoal"])
    ax.set_title(r"Physical AI kinematic roofline: stopping envelope $d_{\mathrm{stop}} = v \tau + \frac{v^2}{2 a_{\max}} \leq R_{\mathrm{sensor}}$ under emergency braking ($a = 0.7\,\mathrm{g}$)",
                 fontsize=9.5, color="#475569", loc="left", pad=14)
    
    # Legend
    legend = ax.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#CBD5E1",
                       fontsize=8.5, title="Kinematic Safety Ceilings ($a_{\\max} = 6.86\\,\\mathrm{m/s}^2$)",
                       title_fontsize=9.0)
    legend.get_title().set_fontweight("bold")
    legend.set_zorder(15)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.96)
    
    out_path = os.path.join(os.path.dirname(__file__), "ch01_latency_vs_velocity.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_plot()
