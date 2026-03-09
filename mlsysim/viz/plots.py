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
    Plots a publication-quality Roofline Model for a given HardwareNode.

    Features:
    - Ridge point annotated with numeric value
    - Memory-bound and compute-bound regions shaded and labeled
    - Memory bandwidth ceiling (diagonal) and compute ceiling (flat)
    - Workloads plotted with bottleneck classification
    """
    # 1. PARAMETERS
    peak_flops = hardware_node.compute.peak_flops.to("TFLOPs/s").magnitude
    peak_bw = hardware_node.memory.bandwidth.to("GB/s").magnitude
    ridge_point = peak_flops / (peak_bw / 1000)  # FLOP/Byte

    # 2. AXIS RANGE
    x_min, x_max = 0.1, 10000
    x = np.logspace(np.log10(x_min), np.log10(x_max), 500)

    # 3. ROOFLINE CURVES
    y_mem = peak_bw * x / 1000  # BW * AI, converted to TFLOP/s
    y_compute = np.full_like(x, peak_flops)
    y_roof = np.minimum(y_mem, y_compute)

    # 4. PLOT
    fig, ax, colors, _ = setup_plot(figsize=(9, 5.5))

    # Shaded regions
    mem_mask = x <= ridge_point
    comp_mask = x >= ridge_point
    ax.fill_between(
        x[mem_mask],
        y_roof[mem_mask] * 0.001,
        y_roof[mem_mask],
        color=colors["OrangeL"],
        alpha=0.5,
        label="Memory-bound region",
    )
    ax.fill_between(
        x[comp_mask],
        y_roof[comp_mask] * 0.001,
        y_roof[comp_mask],
        color=colors["BlueFill"],
        alpha=0.5,
        label="Compute-bound region",
    )

    # Roofline line
    ax.loglog(
        x,
        y_roof,
        color=colors["BlueLine"],
        linewidth=2.5,
        zorder=5,
    )

    # Memory bandwidth ceiling label (on the slope)
    slope_x = ridge_point * 0.08
    slope_y = peak_bw * slope_x / 1000
    ax.text(
        slope_x,
        slope_y * 1.6,
        f"BW ceiling: {peak_bw:.0f} GB/s",
        color=colors["OrangeLine"],
        fontsize=8.5,
        fontweight="bold",
        rotation=38,
        ha="center",
        va="bottom",
    )

    # Compute ceiling label (on the flat)
    ax.text(
        ridge_point * 8,
        peak_flops * 1.12,
        f"Compute ceiling: {peak_flops:.0f} TFLOP/s",
        color=colors["BlueLine"],
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # Ridge point
    ax.plot(
        ridge_point,
        peak_flops,
        "D",
        color=colors["crimson"],
        markersize=9,
        zorder=10,
    )
    ax.annotate(
        f"Ridge Point\n{ridge_point:.1f} FLOP/Byte",
        xy=(ridge_point, peak_flops),
        xytext=(ridge_point * 3, peak_flops * 0.35),
        fontsize=8.5,
        fontweight="bold",
        color=colors["crimson"],
        ha="center",
        arrowprops=dict(
            arrowstyle="->",
            color=colors["crimson"],
            lw=1.2,
        ),
    )

    # Vertical dashed line at ridge point
    ax.axvline(
        ridge_point,
        color=colors["crimson"],
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )

    # Region labels
    ax.text(
        x_min * 1.5,
        peak_flops * 0.6,
        "MEMORY\nBOUND",
        color=colors["OrangeLine"],
        fontsize=11,
        fontweight="bold",
        alpha=0.25,
        ha="left",
        va="center",
    )
    ax.text(
        x_max * 0.4,
        peak_flops * 0.6,
        "COMPUTE\nBOUND",
        color=colors["BlueLine"],
        fontsize=11,
        fontweight="bold",
        alpha=0.25,
        ha="right",
        va="center",
    )

    # Plot workloads
    if workloads:
        from ..core.solver import SingleNodeSolver

        workload_colors = [
            colors["crimson"],
            colors["GreenLine"],
            colors["VioletLine"],
            colors["BrownLine"],
        ]
        for i, model in enumerate(workloads):
            profile = SingleNodeSolver().solve(model, hardware_node, efficiency=1.0)
            ai = profile.arithmetic_intensity.magnitude
            perf = min(peak_bw * ai / 1000, peak_flops)
            c = workload_colors[i % len(workload_colors)]
            bound = "memory" if ai < ridge_point else "compute"
            ax.plot(ai, perf, "o", color=c, markersize=9, zorder=10)
            ax.annotate(
                f"{model.name}\n({bound}-bound)",
                xy=(ai, perf),
                xytext=(ai * 0.3, perf * 0.4),
                fontsize=8,
                fontweight="bold",
                color=c,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=c, lw=1),
            )

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title(f"Roofline: {hardware_node.name}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(peak_flops * 0.001, peak_flops * 2)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
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
