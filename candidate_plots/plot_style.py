"""
Shared publication styling for Volume IV empirical and historical plots.
Matches MIT Press / Harvard textbook standards and high-end visual aesthetics:
- Clean white background
- Subtle dashed grid
- High-contrast sans-serif typography
- Rounded white annotation badges
- Distinct accessible color palette
- Shaded era backgrounds for longitudinal plots
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Palette
PALETTE = {
    "crimson": "#A51C30",
    "dark_blue": "#004B87",
    "teal": "#008080",
    "forest_green": "#1E793C",
    "amber": "#D97706",
    "purple": "#6B21A8",
    "coral": "#E15759",
    "slate": "#475569",
    "gold": "#B45309",
    "cyan": "#0284C7",
    "charcoal": "#1E293B",
    "grid": "#E2E8F0",
    "line_fit": "#0284C7",
}

# Pastel shades for historical eras
ERA_COLORS = [
    "#F1F5F9",  # light slate
    "#EFF6FF",  # light sky
    "#ECFDF5",  # light emerald
    "#FAF5FF",  # light purple
    "#FFFBEB",  # light amber
]

def setup_canvas(figsize=(13, 7.5), dpi=200):
    """Initializes a publication-grade figure and axis."""
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 1.0
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    
    # Hide top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Subtle dashed grid
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color=PALETTE["grid"])
    ax.set_axisbelow(True)
    
    return fig, ax

def add_badge(ax, text, xy, xytext, color=PALETTE["charcoal"], fontsize=9.5, arrow=False, arrow_color="#94A3B8"):
    """Adds a clean rounded white badge annotation."""
    arrowprops = dict(
        arrowstyle="-",
        color=arrow_color,
        linewidth=0.8,
        shrinkA=3,
        shrinkB=3,
    ) if arrow else None
    
    bbox = dict(
        boxstyle="round,pad=0.35",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        alpha=0.95,
        linewidth=0.8,
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
