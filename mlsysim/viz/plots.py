# viz.py
# Centralized Visualization Style for MLSys Book
# Ensures all generated figures across Vol 1 & 2 share a consistent,
# MIT Press-ready aesthetic.

try:
    import matplotlib.pyplot as plt
    import numpy as np
    _viz_available = True
except ImportError:
    plt = None
    np = None
    _viz_available = False

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
    if not _viz_available:
        raise ImportError(
            "matplotlib and numpy are required for plot generation. "
            "Install them with: pip install matplotlib numpy"
        )
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [
            'Helvetica',         # macOS native
            'Helvetica Neue',    # macOS modern variant
            'Nimbus Sans L',     # Free Helvetica clone (TeX/Linux)
            'TeX Gyre Heros',    # Free Helvetica clone (TeX)
            'Arial',             # Windows fallback
            'DejaVu Sans',       # Universal last resort
        ],
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
    """One-line plot setup for QMD blocks."""
    set_book_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, COLORS, plt

def bar_compare(labels, values, title, ylabel, goal_line=None, colors=None):
    """Creates a standard comparison bar chart with value labels."""
    fig, ax, COLORS, plt = setup_plot()
    if colors is None:
        colors = [COLORS['BlueLine'], COLORS['GreenLine'], COLORS['OrangeLine'], COLORS['VioletLine']]
    
    bars = ax.bar(labels, values, color=colors[:len(labels)], alpha=0.8, edgecolor='white', linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max(values)*0.02),
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    if goal_line:
        ax.axhline(y=goal_line, color=COLORS['RedLine'], linestyle='--', linewidth=1.5, label='Constraint')
        ax.legend()
        
    return fig

def plot_latency_breakdown(profile, title="Latency Breakdown"):
    """Plots a stacked bar showing Compute vs Memory vs Overhead."""
    fig, ax, COLORS, plt = setup_plot(figsize=(6, 5))
    
    comp = profile.latency_compute.m_as('ms')
    mem = profile.latency_memory.m_as('ms')
    ovh = profile.latency_overhead.m_as('ms')
    
    labels = ['Latency']
    ax.bar(labels, [comp], label='Compute', color=COLORS['BlueLine'], alpha=0.8)
    ax.bar(labels, [mem], bottom=[comp], label='Memory', color=COLORS['OrangeLine'], alpha=0.8)
    ax.bar(labels, [ovh], bottom=[comp+mem], label='Overhead', color=COLORS['RedLine'], alpha=0.8)
    
    ax.set_title(title)
    ax.set_ylabel("Time (ms)")
    ax.legend(loc='upper right')
    
    total = comp + mem + ovh
    ax.text(0, total/2, f"Total: {total:.2f} ms", ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    
    return fig

def plot_roofline(profile, title="System Roofline Analysis"):
    """Plots the hardware roofline and the current workload point."""
    fig, ax, COLORS, plt = setup_plot()
    
    max_perf = profile.peak_flops_actual.m_as('GFLOPs/s')
    max_bw = profile.peak_bw_actual.m_as('GB/s')
    ridge_point = max_perf / max_bw
    
    ai = profile.arithmetic_intensity.m_as('flop/byte')
    achieved_perf = min(ai * max_bw, max_perf)
    
    # X axis: Arithmetic Intensity
    x = np.logspace(np.log10(ridge_point/100), np.log10(ridge_point*100), 100)
    y = np.minimum(x * max_bw, max_perf)
    
    ax.plot(x, y, color=COLORS['primary'], linewidth=2, label='Roofline')
    ax.scatter([ai], [achieved_perf], color=COLORS['RedLine'], s=100, zorder=5, label=f'Workload (AI={ai:.1f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_ylabel('Performance (GFLOPs/s)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()
    
    return fig
