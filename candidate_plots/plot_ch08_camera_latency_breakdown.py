#!/usr/bin/env python3
"""
Chapter 8: Perception Plot
Photon-to-actuator sensor pipeline latency breakdown across camera interfaces:
USB 3.0, Automotive GMSL2, Embedded MIPI CSI-2, and Neuromorphic Event Cameras.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Import style helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import setup_canvas, add_badge, PALETTE

def generate_ch08_plot():
    # Setup canvas with ample room
    fig, ax = setup_canvas(figsize=(14.0, 8.2), dpi=300)
    
    interfaces = [
        "Neuromorphic Event Camera\n(Prophesee Metavision DVS)",
        "Embedded MIPI CSI-2\n(Direct SoC D-PHY DMA)",
        "Automotive GMSL2\n(15 m Coaxial SerDes)",
        "Industrial USB 3.0\n(Standard UVC Driver)",
    ]
    
    # Stages and their measured latencies (in milliseconds)
    # [Event Camera, MIPI CSI-2, GMSL2, USB 3.0]
    stages = [
        ("Optical Transduction / Exposure", [0.05, 8.0, 8.0, 10.0], "#F59E0B"),       # Amber
        ("Sensor Readout & ADC Digitization", [0.10, 8.3, 8.3, 14.2], "#F97316"),   # Orange
        ("Interface Transport & Link SerDes", [0.20, 0.15, 0.05, 4.8], "#EF4444"),  # Red
        ("Host Ingest, DMA & ISP Pipeline", [0.80, 1.4, 1.6, 8.5], "#8B5CF6"),     # Purple
        ("Neural Perception Inference", [2.8, 8.5, 8.5, 11.2], "#0284C7"),         # Cyan/Blue
        ("Planning & Control Dispatch", [1.5, 1.8, 1.8, 2.2], "#0D9488"),          # Teal
        ("Actuator Bus & Motor Torque Rise", [6.5, 6.5, 6.5, 7.5], "#16A34A"),     # Green
    ]
    
    y_pos = np.arange(len(interfaces))
    bar_height = 0.50
    
    lefts = np.zeros(len(interfaces))
    
    for stage_name, values, color in stages:
        vals = np.array(values)
        ax.barh(y_pos, vals, left=lefts, height=bar_height, label=stage_name,
               color=color, edgecolor="#FFFFFF", linewidth=1.2, alpha=0.95, zorder=3)
        
        # Add values inside bars if wide enough
        for i, val in enumerate(vals):
            if val >= 4.0:
                ax.text(lefts[i] + val / 2.0, y_pos[i], f"{val:.1f}",
                        ha="center", va="center", color="#FFFFFF", fontsize=8.5, fontweight="bold", zorder=4)
        lefts += vals
        
    # Total latency labels at the end of each bar
    for i, total in enumerate(lefts):
        ax.text(total + 1.2, y_pos[i], f"{total:.1f} ms",
                ha="left", va="center", color=PALETTE["charcoal"], fontsize=11.0, fontweight="bold", zorder=4)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(interfaces, fontsize=10.5, fontweight="bold", color=PALETTE["charcoal"])
    ax.invert_yaxis()  # Event camera at top, USB 3.0 at bottom
    
    ax.set_xlabel("Photon-to-Actuator Pipeline Latency (milliseconds)", fontsize=11.5, fontweight="bold", color=PALETTE["charcoal"], labelpad=10)
    ax.set_title("Photon-to-Actuator Latency Waterfall Across Camera Interfaces",
                 fontsize=14.5, fontweight="bold", color=PALETTE["charcoal"], pad=24)
    
    ax.set_xlim(0, 72)
    ax.set_ylim(3.7, -0.7)  # Generous vertical margins
    
    # Legend placed cleanly at top right
    legend = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True,
                       facecolor="#FFFFFF", edgecolor="#CBD5E1", framealpha=0.95,
                       fontsize=8.5, ncol=2)
    legend.set_zorder(10)
    
    # Clean Badges without collisions
    # 1. Event camera badge - placed right next to the event camera bar
    add_badge(ax, "Neuromorphic DVS: Bypasses exposure & rolling readout\n11.9 ms total photon-to-actuator reaction",
              xy=(13.5, y_pos[0]), xytext=(24.5, y_pos[0] + 0.12),
              color=PALETTE["forest_green"], fontsize=8.8, arrow=True, arrow_color="#16A34A")
    
    # 2. GMSL2 badge - placed above GMSL2 bar pointing down to SerDes
    add_badge(ax, "GMSL2 SerDes: 15 m automotive reach with only ~50 µs link delay\nHardware zero-copy ISP DMA enables deterministic 34.8 ms loop",
              xy=(16.35, y_pos[2] - 0.26), xytext=(28.0, y_pos[2] - 0.42),
              color=PALETTE["dark_blue"], fontsize=8.8, arrow=True, arrow_color="#0284C7")
    
    # 3. USB 3.0 badge - placed cleanly below USB 3.0 bar
    add_badge(ax, "USB 3.0 UVC Tax: Host kernel queueing & driver copy\nadds ~13.3 ms protocol overhead + non-deterministic jitter",
              xy=(26.5, y_pos[3] + 0.26), xytext=(36.0, y_pos[3] + 0.42),
              color=PALETTE["crimson"], fontsize=8.8, arrow=True, arrow_color="#DC2626")
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch08_perception_camera_latency_breakdown.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_ch08_plot()
