# viz.py
# Centralized Visualization Style for MLSys Book
# Ensures all generated figures across Vol 1 & 2 share a consistent,
# MIT Press-ready aesthetic.

import matplotlib.pyplot as plt

# --- Brand & Book Palette ---
COLORS = {
    "crimson":    "#A51C30", # Harvard Crimson
    "primary":    "#333333", # Dark Gray (Text)
    "grid":       "#CCCCCC", # Light Gray
    
    # TikZ / Technical Palette (The "Golden" set)
    "BlueLine":   "#006395", "BlueL":   "#D1E6F3", "BlueFill": "#D6EAF8",
    "RedLine":    "#CB202D", "RedL":    "#F5D2D5", "RedFill": "#F2D7D5",
    "GreenLine":  "#008F45", "GreenL":  "#D4EFDF", "GreenFill": "#D5F5E3",
    "OrangeLine": "#E67817", "OrangeL": "#FCE4CC",
    "VioletLine": "#7E317B", "VioletL": "#E6D4E5",
    "BrownLine":  "#78492A", "BrownL":  "#E3D3C8",
    "YellowFill": "#FEF9E0"
}

def set_book_style():
    """Applies the global matplotlib style configuration."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'Inter', 'DejaVu Sans'],
        'font.size': 10,
        'text.color': COLORS['primary'],
        'axes.labelsize': 11,
        'axes.labelcolor': COLORS['primary'],
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.edgecolor': COLORS['primary'],
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': COLORS['primary'],
        'ytick.color': COLORS['primary'],
        'axes.grid': True,
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'legend.title_fontsize': 10,
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.figsize': (8, 5),
        'figure.autolayout': True
    })

# --- Lightweight helpers ---

def setup_plot(figsize=None):
    """
    One-line plot setup for QMD blocks.
    Returns (fig, ax, COLORS, plt) after applying book style.
    The plt is returned so code blocks don't need separate matplotlib import.
    """
    set_book_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, COLORS, plt
