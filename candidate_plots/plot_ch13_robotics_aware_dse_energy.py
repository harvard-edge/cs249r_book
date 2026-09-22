#!/usr/bin/env python3
"""
Chapter 13: Placement / Hardware Co-Design Plot
Robotics-Aware Design Space Exploration (DSE):
Edge NPU TOPS and Memory Bandwidth vs. Closed-Loop Mission Energy & System Efficiency.
Reproducing the inverted-U Pareto frontier with optimal operating point matching user reference.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# Import style helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import setup_canvas, PALETTE

def add_clean_badge(ax, text, xy, xytext, color=PALETTE["charcoal"], fontsize=8.5, arrow_color="#94A3B8", ha="center", va="center", connectionstyle=None):
    """Adds a clean rounded badge with an arrow."""
    bbox = dict(
        boxstyle="round,pad=0.45",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        alpha=0.96,
        linewidth=0.8,
    )
    ap = dict(
        arrowstyle="->",
        color=arrow_color,
        linewidth=1.1,
        shrinkA=3,
        shrinkB=4,
    )
    if connectionstyle:
        ap["connectionstyle"] = connectionstyle
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        fontweight="medium",
        color=color,
        bbox=bbox,
        arrowprops=ap,
        ha=ha,
        va=va,
        zorder=10,
    )


def generate_ch13_plot():
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 1.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.0, 9.8), sharex=True, dpi=300, facecolor="#FFFFFF")
    ax1.set_facecolor="#FFFFFF"
    ax2.set_facecolor="#FFFFFF"
    
    # ----------------------------------------------------
    # Hardware Platforms Data
    # ----------------------------------------------------
    platforms = [
        {
            "name": "Coral Edge TPU",
            "spec": "4 TOPS, 2W",
            "x": 1,
            "tops": 4,
            "bw": 1.6,
            "tdp": 4.0,
            "t_mission": 215.0,
            "e_static": 23.65,
            "e_comp": 0.86,
            "e_total": 24.51,
            "eff": 40.8,
            "cat": "under",
        },
        {
            "name": "RPi 5 + Hailo-8",
            "spec": "26 TOPS, 5W",
            "x": 2,
            "tops": 26,
            "bw": 8.0,
            "tdp": 9.0,
            "t_mission": 98.0,
            "e_static": 10.78,
            "e_comp": 0.88,
            "e_total": 11.66,
            "eff": 85.8,
            "cat": "under",
        },
        {
            "name": "Jetson Orin Nano",
            "spec": "40 TOPS, 15W",
            "x": 3,
            "tops": 40,
            "bw": 68.0,
            "tdp": 15.0,
            "t_mission": 48.0,
            "e_static": 5.28,
            "e_comp": 0.72,
            "e_total": 6.00,
            "eff": 166.7,
            "cat": "optimal",
        },
        {
            "name": "Jetson Orin NX",
            "spec": "100 TOPS, 25W",
            "x": 4,
            "tops": 100,
            "bw": 102.0,
            "tdp": 25.0,
            "t_mission": 33.0,
            "e_static": 3.63,
            "e_comp": 0.83,
            "e_total": 4.46,
            "eff": 224.2,
            "cat": "optimal",
        },
        {
            "name": "Jetson AGX Orin",
            "spec": "275 TOPS, 50W",
            "x": 5,
            "tops": 275,
            "bw": 204.0,
            "tdp": 50.0,
            "t_mission": 27.5,
            "e_static": 3.03,
            "e_comp": 1.38,
            "e_total": 4.40,
            "eff": 227.3,
            "cat": "optimal_peak",
        },
        {
            "name": "Dual AGX Orin",
            "spec": "550 TOPS, 130W",
            "x": 6,
            "tops": 550,
            "bw": 408.0,
            "tdp": 130.0,
            "t_mission": 25.2,
            "e_static": 2.77,
            "e_comp": 3.28,
            "e_total": 6.05,
            "eff": 165.3,
            "cat": "over",
        },
        {
            "name": "RTX 4090 Mobile",
            "spec": "1300 TOPS, 300W",
            "x": 7,
            "tops": 1300,
            "bw": 576.0,
            "tdp": 300.0,
            "t_mission": 24.0,
            "e_static": 2.64,
            "e_comp": 7.20,
            "e_total": 9.84,
            "eff": 101.6,
            "cat": "over",
        },
    ]
    
    xs = np.array([p["x"] for p in platforms])
    effs = np.array([p["eff"] for p in platforms])
    e_totals = np.array([p["e_total"] for p in platforms])
    e_statics = np.array([p["e_static"] for p in platforms])
    e_comps = np.array([p["e_comp"] for p in platforms])
    
    # PCHIP spline for monotonic, overshoot-free inverted-U curve passing through all points
    pchip_eff = PchipInterpolator(xs, effs)
    x_fine = np.linspace(1.0, 7.0, 300)
    eff_fine = pchip_eff(x_fine)
    
    # ----------------------------------------------------
    # Shaded Zones (applied to both ax1 and ax2)
    # ----------------------------------------------------
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color=PALETTE["grid"])
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        
        # Red zone: Under-provisioned (stalls & chassis energy drain)
        ax.axvspan(0.5, 2.5, color="#FEF2F2", alpha=0.55, zorder=1)
        
        # Green zone: Optimal balanced co-design
        ax.axvspan(2.5, 5.5, color="#ECFDF5", alpha=0.60, zorder=1)
        
        # Orange zone: Over-provisioned (power explosion & velocity saturation)
        ax.axvspan(5.5, 7.5, color="#FFFBEB", alpha=0.55, zorder=1)
    
    # ----------------------------------------------------
    # Top Panel: System Performance / Efficiency Inverted-U
    # ----------------------------------------------------
    # Plot smooth inverted-U curve
    ax1.plot(x_fine, eff_fine, color="#059669", linewidth=2.8, alpha=0.88, zorder=4,
             label="Robotics-Aware System Efficiency Frontier")
    
    # Scatter points for platforms
    for p in platforms:
        if p["cat"] == "optimal_peak":
            # Bold green 'X' matching user reference diagram
            ax1.scatter(p["x"], p["eff"], color="#047857", marker="x", s=280, linewidths=3.8, zorder=7,
                        label="Pareto Optimal Operating Point (AGX Orin)")
            # Surrounding circle
            ax1.scatter(p["x"], p["eff"], facecolors="none", edgecolors="#047857", s=320, linewidths=1.8, linestyle="--", zorder=6)
        else:
            ax1.scatter(p["x"], p["eff"], color=PALETTE["dark_blue"], marker="o", s=70, edgecolors="#FFFFFF", linewidth=1.2, zorder=5)
            # Offset labels
            y_offset = 10 if p["x"] != 4 else -18
            va_align = "bottom" if p["x"] != 4 else "top"
            ax1.annotate(
                f"{p['eff']:.1f}",
                (p["x"], p["eff"]),
                xytext=(0, y_offset),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color=PALETTE["charcoal"],
                ha="center",
                va=va_align,
                zorder=6
            )
            
    ax1.set_ylim(0, 315)
    ax1.set_ylabel("System Energy Efficiency\n(Closed-Loop Tasks Solved / MJ)", fontsize=10.5, fontweight="bold", color=PALETTE["charcoal"], labelpad=8)
    ax1.set_title("Robotics-Aware Design Space Exploration (DSE): Chip Configuration vs. System Efficiency",
                  fontsize=13.0, fontweight="bold", color=PALETTE["charcoal"], pad=14)
    
    # Zone Labels in top panel
    ax1.text(1.5, 296, "Under-Provisioned Zone\n(Sluggish VLA Execution -> Static Drain)", ha="center", va="top", fontsize=8.2, fontweight="bold", color="#B91C1C")
    ax1.text(4.0, 296, "Optimal Co-Design Window\n(Balanced TOPS & Memory Bandwidth)", ha="center", va="top", fontsize=8.2, fontweight="bold", color="#047857")
    ax1.text(6.5, 296, "Over-Provisioned Zone\n(Mechanical Kinematic Limit Saturation)", ha="center", va="top", fontsize=8.2, fontweight="bold", color="#B45309")
    
    # Key Badges in Top Panel
    # 1. Optimal 'X' badge positioned in optimal zone at x=3.6, pointing to 'X' at x=5.0
    add_clean_badge(ax1, "Optimal System Perf. (Matching DSE Reference):\nJetson AGX Orin (275 TOPS, 204 GB/s, 50W)\nGlobal Peak: 227.3 Tasks / Megajoule",
                    xy=(4.95, 230.0), xytext=(3.60, 250),
                    color="#047857", fontsize=8.3, arrow_color="#047857", ha="center")
    
    # 2. Compute Starvation badge positioned BELOW curve in bottom-left whitespace
    add_clean_badge(ax1, "Compute & Memory Starvation:\nReasoning (93.1% FLOPs) & DiT (85.6% mem) stalled;\n215 s mission burns 23.6 kJ chassis standby energy.",
                    xy=(1.0, 40.8), xytext=(2.0, 20),
                    color="#DC2626", fontsize=8.1, arrow_color="#DC2626", ha="center")
    
    # 3. Mechanical Saturation badge positioned BELOW curve in bottom-right whitespace
    add_clean_badge(ax1, "Kinematic Velocity Ceiling:\nInference latency drops to 55 ms, but robot hits\nmotor velocity limit (v_max = 0.8 m/s); 300W TDP drains battery.",
                    xy=(7.0, 101.6), xytext=(5.9, 45),
                    color="#B45309", fontsize=8.1, arrow_color="#B45309", ha="center")
    
    ax1.legend(loc="lower left", bbox_to_anchor=(0.02, 0.52), frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.8)
    
    # ----------------------------------------------------
    # Bottom Panel: Energy Breakdown (kJ)
    # ----------------------------------------------------
    bar_width = 0.44
    
    # Stacked bars for Energy Breakdown
    b1 = ax2.bar(xs, e_statics, width=bar_width, color=PALETTE["slate"], alpha=0.85, label="Static Chassis & Motor Standby Energy (P_static × T_mission)", zorder=3)
    b2 = ax2.bar(xs, e_comps, bottom=e_statics, width=bar_width, color=PALETTE["crimson"], alpha=0.88, label="Dynamic Accelerator Energy (P_chip × T_mission)", zorder=3)
    
    # Overlay U-shaped total energy line connecting bar tops
    ax2.plot(xs, e_totals, linestyle="--", color=PALETTE["dark_blue"], linewidth=2.0, marker="D", markersize=6,
             label="Closed-Loop Total Mission Energy (kJ)", zorder=4)
    
    # Total energy numbers above bars
    for i, p in enumerate(platforms):
        tot = p["e_total"]
        weight_txt = "bold"
        color_txt = "#047857" if p["cat"] == "optimal_peak" else PALETTE["charcoal"]
        label_str = f"{tot:.1f} kJ\n(Min)" if p["cat"] == "optimal_peak" else f"{tot:.1f} kJ"
        ax2.text(p["x"], tot + 0.6, label_str, ha="center", va="bottom", fontsize=8.3, fontweight=weight_txt, color=color_txt, zorder=5)
        
    ax2.set_ylabel("Closed-Loop Energy per Task (kJ)", fontsize=10.5, fontweight="bold", color=PALETTE["charcoal"], labelpad=8)
    ax2.set_ylim(0, 32)
    
    ax2.set_title("Energy Breakdown: Static Chassis Standby Burden vs. Dynamic Accelerator Dissipation",
                  fontsize=12.5, fontweight="bold", color=PALETTE["charcoal"], pad=14)
    
    # Badges in Bottom Panel (placed in large whitespace above bars)
    add_clean_badge(ax2, "Minimum Energy: 4.40 kJ / Task\nCo-design equilibrium:\nAmortizes inference to 155 ms\nwithout wasteful GPU TDP.",
                    xy=(5.0, 4.40), xytext=(3.7, 16.0),
                    color="#047857", fontsize=8.2, arrow_color="#047857", ha="center")
    
    add_clean_badge(ax2, "Kinematic Saturation Limit:\nBelow 27.5 s, physical robot velocity limits\nprevent faster task execution. RTX 4090 saves\nonly 3.5 s while burning 5.8 kJ more energy.",
                    xy=(6.78, 9.84), xytext=(5.8, 18.0),
                    color="#B45309", fontsize=8.1, arrow_color="#B45309", ha="center")
    
    ax2.legend(loc="upper right", bbox_to_anchor=(0.985, 0.98), frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.6)
    
    # X-axis platform labels with hardware specs and mission time
    x_tick_labels = [
        f"{p['name']}\n{p['tops']} TOPS | {p['tdp']:.0f}W\n(T = {p['t_mission']:.1f}s)" for p in platforms
    ]
    ax2.set_xticks(xs)
    ax2.set_xticklabels(x_tick_labels, fontsize=8.8, fontweight="bold", color=PALETTE["charcoal"])
    ax2.set_xlabel("Edge Accelerator Platform & Closed-Loop Mission Duration (T_mission)",
                   fontsize=11.0, fontweight="bold", color=PALETTE["charcoal"], labelpad=10)
    
    ax1.set_xlim(0.4, 7.6)
    ax2.set_xlim(0.4, 7.6)
    
    plt.tight_layout()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ch13_placement_robotics_aware_dse.png")
    plt.savefig(out_path, dpi=300, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_ch13_plot()
