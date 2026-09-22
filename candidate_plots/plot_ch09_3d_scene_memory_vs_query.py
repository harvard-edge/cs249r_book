#!/usr/bin/env python3
"""
Chapter 9: Memory Plot
3D Scene Representation Trade-off:
Volumetric Memory Footprint (MB/m^3) vs. Spatial Point Query Latency (microseconds)
Across OctoMap, Voxblox TSDF/ESDF, 3D Gaussian Splatting, and NeRF.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Import style helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import setup_canvas, PALETTE

def add_clean_badge(ax, text, xy, xytext, color=PALETTE["charcoal"], fontsize=8.5, arrow_color="#94A3B8"):
    """Adds a clean rounded badge with a distinct arrowhead."""
    bbox = dict(
        boxstyle="round,pad=0.4",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        alpha=0.96,
        linewidth=0.8,
    )
    arrowprops = dict(
        arrowstyle="->",
        color=arrow_color,
        linewidth=1.0,
        shrinkA=3,
        shrinkB=4,
    )
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        fontweight="medium",
        color=color,
        bbox=bbox,
        arrowprops=arrowprops,
        va="center",
        zorder=10,
    )

def generate_ch09_plot():
    fig, ax = setup_canvas(figsize=(14.0, 8.2), dpi=300)
    
    # 1. Shaded Region for Real-Time Control Clearance
    # In a 1 kHz control cycle (1,000 µs), collision checking must complete in <10 µs for multi-body checking
    ax.axhspan(0.2, 10.0, color="#ECFDF5", alpha=0.55, zorder=1)
    ax.text(0.046, 6.2, "Real-Time 1 kHz Control Budget\n(Sub-10 µs Collision Query Limit)",
            fontsize=8.5, fontweight="bold", color="#059669", zorder=2)
    
    # High-latency warning zone
    ax.axhspan(100.0, 1500.0, color="#FEF2F2", alpha=0.45, zorder=1)
    ax.text(0.046, 210.0, "Deliberative / Offline Zone\n(High Latency: >100 µs / Query)",
            fontsize=8.5, fontweight="bold", color="#DC2626", zorder=2)
    
    # Data points: (Name, Memory_MB_per_m3, Query_Latency_us, Category, ha, va, (dx, dy))
    data = [
        ("iMAP / NeRF (Implicit MLP)", 0.08, 650.0, "neural", "center", "bottom", (0, 10)),
        ("Instant-NGP (Hash Grid)", 0.42, 140.0, "neural", "left", "center", (10, 0)),
        ("OctoMap (5 cm Octree)", 0.18, 24.0, "octree", "right", "bottom", (-10, 8)),
        ("OctoMap (2 cm Octree)", 1.25, 38.0, "octree", "left", "bottom", (10, 6)),
        ("3DGS (Compressed Scaffold)", 1.60, 8.2, "3dgs", "right", "top", (-10, -6)),
        ("3DGS (SplaTAM Dense)", 5.20, 5.8, "3dgs", "left", "top", (10, -8)),
        ("Voxblox TSDF (5 cm Hashed)", 4.10, 1.8, "voxblox", "center", "top", (0, -10)),
        ("Voxblox ESDF (2 cm Hashed + Grad)", 28.5, 0.85, "voxblox", "left", "center", (10, 0)),
        ("Dense Occupancy Array (2 cm Grid)", 125.0, 0.45, "dense", "right", "top", (-10, -10)),
    ]
    
    cat_styles = {
        "neural": {"label": "Implicit Neural Field (NeRF / NGP)", "color": "#7C3AED", "marker": "D", "size": 115},
        "octree": {"label": "Hierarchical Octree (OctoMap)", "color": PALETTE["dark_blue"], "marker": "s", "size": 110},
        "3dgs": {"label": "3D Gaussian Splatting (3DGS)", "color": PALETTE["crimson"], "marker": "*", "size": 190},
        "voxblox": {"label": "Hashed Voxel TSDF / ESDF (Voxblox)", "color": PALETTE["forest_green"], "marker": "^", "size": 125},
        "dense": {"label": "Dense Uniform Array", "color": PALETTE["slate"], "marker": "o", "size": 95},
    }
    
    # Plot points
    for name, mem, lat, cat, ha, va, (dx, dy) in data:
        style = cat_styles[cat]
        ax.scatter(mem, lat, color=style["color"], marker=style["marker"], s=style["size"],
                   edgecolors="#FFFFFF", linewidth=1.2, zorder=5)
        
        ax.annotate(
            name,
            (mem, lat),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.8,
            fontweight="bold",
            color=PALETTE["charcoal"],
            ha=ha,
            va=va,
            zorder=6
        )
    
    # Pareto boundary line connecting optimal frontier points
    frontier_x = np.array([0.08, 0.18, 1.60, 4.10, 28.5, 125.0])
    frontier_y = np.array([650.0, 24.0, 8.2, 1.8, 0.85, 0.45])
    
    ax.plot(frontier_x, frontier_y, linestyle="--", color=PALETTE["line_fit"], linewidth=1.8, alpha=0.85, zorder=4,
            label="Empirical Pareto Frontier")
    
    # Log scale
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlim(0.04, 250.0)
    ax.set_ylim(0.25, 1500.0)
    
    ax.set_xlabel("Volumetric Memory Footprint (MB / m³ at operational resolution)", fontsize=11.5, fontweight="bold", color=PALETTE["charcoal"], labelpad=10)
    ax.set_ylabel("Spatial Point Distance / Collision Query Latency (microseconds, µs)", fontsize=11.5, fontweight="bold", color=PALETTE["charcoal"], labelpad=10)
    ax.set_title("3D Scene Representation Frontier: Memory Density vs. Spatial Query Latency",
                 fontsize=14.5, fontweight="bold", color=PALETTE["charcoal"], pad=24)
    
    # Badges for key trade-offs in open whitespace
    # 1. NeRF badge - top center
    add_clean_badge(ax, "NeRF (Implicit MLP):\nMinimal memory (~8 MB/room),\nbut requires forward pass per query point",
                    xy=(0.08, 650.0), xytext=(0.32, 650.0),
                    color="#7C3AED", fontsize=8.5, arrow_color="#7C3AED")
    
    # 2. OctoMap badge
    add_clean_badge(ax, "OctoMap (Hierarchical Octree):\nPrunes empty space via tree compression;\npointer indirection causes CPU cache misses",
                    xy=(0.18, 24.0), xytext=(0.20, 95.0),
                    color=PALETTE["dark_blue"], fontsize=8.5, arrow_color=PALETTE["dark_blue"])
    
    # 3. 3DGS badge
    add_clean_badge(ax, "3D Gaussian Splatting (3DGS):\nExplicit ellipsoids enable photo-realism\n+ GPU BVH analytic distance evaluation",
                    xy=(5.20, 5.8), xytext=(16.0, 22.0),
                    color=PALETTE["crimson"], fontsize=8.5, arrow_color=PALETTE["crimson"])
    
    # 4. Voxblox ESDF badge
    add_clean_badge(ax, "Voxblox ESDF (Hashed Voxels):\nHigher memory footprint (28.5 MB/m³),\nbut enables sub-µs O(1) continuous gradients",
                    xy=(28.5, 0.85), xytext=(42.0, 2.5),
                    color=PALETTE["forest_green"], fontsize=8.5, arrow_color=PALETTE["forest_green"])
    
    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], linestyle="--", color=PALETTE["line_fit"], lw=1.8, label="Empirical Pareto Frontier"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#7C3AED", markersize=9, label="Implicit Neural Field (NeRF / NGP)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["dark_blue"], markersize=9, label="Hierarchical Octree (OctoMap)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=PALETTE["crimson"], markersize=12, label="3D Gaussian Splatting (3DGS)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=PALETTE["forest_green"], markersize=9, label="Hashed Voxel TSDF / ESDF (Voxblox)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["slate"], markersize=8, label="Dense Uniform Array"),
    ]
    
    legend = ax.legend(handles=legend_elements, loc="upper right", frameon=True,
                       facecolor="#FFFFFF", edgecolor="#CBD5E1", framealpha=0.96,
                       fontsize=8.5, ncol=1)
    legend.set_zorder(10)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch09_memory_3d_representation_pareto.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_ch09_plot()
