# viz/plots.py
# Centralized Visualization Style for MLSys Book
# Ensures all generated figures across Vol 1 & 2 share a consistent,
# MIT Press-ready aesthetic.

try:
    import matplotlib.pyplot as plt
    import numpy as np
    _matplotlib_available = True
except ImportError:
    plt = None
    np = None
    _matplotlib_available = False

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
    if not _matplotlib_available:
        raise ImportError("matplotlib is required for plot generation.")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'text.color': COLORS['primary'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })

def setup_plot(figsize=(8, 5)):
    """One-line plot setup for QMD blocks."""
    set_book_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, COLORS, plt

def plot_roofline(hardware_node, workloads=None):
    """
    Plots a standard Roofline Model for a given HardwareNode.
    Follows the LEGO-style visualization pattern.
    """
    # 1. PARAMETERS
    peak_flops = hardware_node.compute.peak_flops.to('TFLOPs/s').magnitude
    peak_bw = hardware_node.memory.bandwidth.to('GB/s').magnitude
    
    # 2. INVARIANTS
    x_intensities = np.logspace(-1, 4, 100)
    
    # 3. CALCULATION
    y_memory_bound = peak_bw * x_intensities / 1000 # TFLOPs equivalent
    y_compute_bound = np.full_like(x_intensities, peak_flops)
    y_roofline = np.minimum(y_memory_bound, y_compute_bound)
    
    # 4. OUTPUT (Visualization)
    fig, ax, colors, plt = setup_plot()
    ax.loglog(x_intensities, y_roofline, color=colors['BlueLine'], linewidth=2.5, label=f'{hardware_node.name} Roofline')
    ax.fill_between(x_intensities, 0, y_roofline, color=colors['BlueFill'], alpha=0.3)
    
    if workloads:
        from ..core.engine import Engine
        for model in workloads:
            profile = Engine.solve(model, hardware_node, efficiency=1.0)
            intensity = profile.arithmetic_intensity.magnitude
            theoretical_perf = min(peak_bw * intensity / 1000, peak_flops)
            ax.plot(intensity, theoretical_perf, 'o', color=colors['crimson'], markersize=8)
            ax.text(intensity * 1.2, theoretical_perf, model.name, color=colors['crimson'], fontsize=9, fontweight='bold')

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_ylabel('Performance (TFLOPs/s)')
    ax.set_title(f'Roofline: {hardware_node.name}')
    return fig, ax

def plot_evaluation_scorecard(evaluation):
    """
    Visualizes the supply-vs-demand scorecard for a SystemEvaluation.
    Follows the LEGO-style visualization pattern.
    """
    # 1. PARAMETERS
    from ..core.constants import Q_
    l1_metrics = evaluation.feasibility.metrics
    l2_metrics = evaluation.performance.metrics
    
    # 2. CALCULATION
    l1_ratio = (l1_metrics['weight_size'] / l1_metrics['capacity']).to_base_units().magnitude
    l2_ratio = (l2_metrics['latency'] / l2_metrics.get('sla_latency', Q_("1000 ms"))).to_base_units().magnitude
    
    levels = ['Memory (RAM)', 'Latency (SLA)']
    ratios = [l1_ratio, l2_ratio]
    
    # 3. OUTPUT (Visualization)
    fig, ax, colors, plt = setup_plot(figsize=(8, 4))
    bar_colors = [colors['RedLine'] if r > 1.0 else colors['GreenLine'] for r in ratios]
    bars = ax.barh(levels, ratios, color=bar_colors, alpha=0.7, edgecolor='black')
    
    ax.axvline(1.0, color=colors['primary'], linestyle='--', linewidth=2, label='Physical Limit / SLA')
    
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, f"{ratio:.1%}", 
                va='center', fontweight='bold', color=bar_colors[i])

    ax.set_xlim(0, max(max(ratios) + 0.5, 1.5))
    ax.set_xlabel('Resource Utilization (Demand / Supply)')
    ax.set_title(f'System Evaluation: {evaluation.scenario_name}')
    return fig, ax
