# viz/plots.py
# Simulator-aware plotters for generated MLSysIM figures. Every plotter draws
# through the shared style in viz/style.py.

try:
    import numpy as np
except ImportError:
    np = None

from .style import (
    BODY_FIGSIZE,
    COLORS,
    MIN_FONT_PT,
    WEB_FIG_DPI,
    bar_compare,
    finalize_web_figure as _finalize_web_figure,
    plt,
    set_book_style,
    setup_plot,
)

__all__ = [
    "BODY_FIGSIZE",
    "COLORS",
    "MIN_FONT_PT",
    "WEB_FIG_DPI",
    "bar_compare",
    "plt",
    "set_book_style",
    "setup_plot",
    "plot_roofline",
    "plot_distributed_roofline",
    "plot_evaluation_scorecard",
]


def _slope_screen_angle(ax, x0, y0, x1, y1):
    """Return the display-space angle (in degrees) of the line from (x0, y0) to (x1, y1)."""
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    return float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))


def _pick_slope_label_x(x_slope_start, ridge_point, workload_ais):
    """Pick an x position along the visible memory slope clear of axes, ridge, and workloads."""
    u_lo = np.log10(max(x_slope_start * 8.0, 1.2))
    u_hi = np.log10(max(ridge_point * 0.15, u_lo + 0.1))
    if u_hi <= u_lo + 0.12:
        return float(10.0 ** ((np.log10(x_slope_start) + np.log10(ridge_point)) * 0.44))
    center_u = 0.5 * (u_lo + u_hi)
    if not workload_ais:
        return float(10.0**center_u)
    candidates = np.linspace(u_lo, u_hi, 9)
    w_logs = [np.log10(max(ai, 1e-3)) for ai in workload_ais]
    best_u = max(
        candidates,
        key=lambda u: min(abs(u - w) for w in w_logs) - 0.15 * abs(u - center_u),
    )
    return float(10.0**best_u)


def plot_roofline(hardware_node, workloads=None, title=None):
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
    x_min = 0.1
    x_max = max(10000.0, 10.0 ** np.ceil(np.log10(ridge_point * 30.0)))
    y_min, y_max = peak_flops * 0.0005, peak_flops * 3.0
    x = np.logspace(np.log10(x_min), np.log10(x_max), 500)

    # 3. ROOFLINE CURVES
    y_mem = peak_bw * x / 1000  # BW * AI, converted to TFLOP/s
    y_compute = np.full_like(x, peak_flops)
    y_roof = np.minimum(y_mem, y_compute)

    # 4. PLOT
    fig, ax, colors, _ = setup_plot(figsize=(BODY_FIGSIZE[0], 4.6))

    # Shaded regions
    mem_mask = x <= ridge_point
    comp_mask = x >= ridge_point
    ax.fill_between(
        x[mem_mask],
        y_min,
        y_roof[mem_mask],
        color=colors["BlueL"],
        alpha=0.5,
        label="Memory-bound region",
    )
    ax.fill_between(
        x[comp_mask],
        y_min,
        y_roof[comp_mask],
        color=colors["OrangeL"],
        alpha=0.5,
        label="Compute-bound region",
    )

    # Roofline line
    ax.loglog(
        x[mem_mask],
        y_roof[mem_mask],
        color=colors["BlueLine"],
        linewidth=2.5,
        zorder=5,
    )
    ax.loglog(
        x[comp_mask],
        y_roof[comp_mask],
        color=colors["OrangeLine"],
        linewidth=2.5,
        zorder=5,
    )

    # Set limits before computing display angles and label placements
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Resolve workload points upfront so labels never collide with them
    resolved_workloads = []
    if workloads:
        from ..engine.solvers import SingleNodeModel

        for model in workloads:
            profile = SingleNodeModel().solve(model, hardware_node, efficiency=1.0)
            ai = profile.arithmetic_intensity.magnitude
            perf = min(peak_bw * ai / 1000, peak_flops)
            resolved_workloads.append((model, ai, perf))

    # Memory bandwidth ceiling label (placed on the open middle of the slope)
    x_slope_start = max(x_min, y_min * 1000.0 / peak_bw)
    avoid_xs = []
    for _, ai, _ in resolved_workloads:
        if ai < ridge_point:
            avoid_xs.append(ai if ai < 2.0 else ai * 0.25)
    slope_x = _pick_slope_label_x(x_slope_start, ridge_point, avoid_xs)
    slope_y = peak_bw * slope_x / 1000
    slope_angle = _slope_screen_angle(
        ax, x_slope_start, peak_bw * x_slope_start / 1000, ridge_point, peak_flops
    )
    ax.text(
        slope_x,
        slope_y * 1.38,
        f"BW ceiling: {peak_bw:.0f} GB/s",
        color=colors["BlueLine"],
        fontsize=MIN_FONT_PT,
        fontweight="bold",
        rotation=slope_angle,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
    )

    # Compute ceiling label (above the flat roof, clear of ridge diamond)
    ax.text(
        x_max * 0.94,
        peak_flops * 1.24,
        f"Compute ceiling: {peak_flops:.0f} TFLOP/s",
        color=colors["OrangeLine"],
        fontsize=MIN_FONT_PT,
        fontweight="bold",
        ha="right",
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
    ridge_label_x = (ridge_point * x_max) ** 0.45
    ax.annotate(
        f"Ridge Point\n{ridge_point:.1f} FLOP/Byte",
        xy=(ridge_point, peak_flops),
        xytext=(ridge_label_x, peak_flops * 0.24),
        fontsize=MIN_FONT_PT,
        fontweight="bold",
        color=colors["crimson"],
        ha="center",
        va="top",
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
        (x_slope_start * ridge_point) ** 0.64,
        (y_min * peak_flops) ** 0.30,
        "MEMORY\nBOUND",
        color=colors["BlueLine"],
        fontsize=11,
        fontweight="bold",
        alpha=0.25,
        ha="center",
        va="center",
    )
    ax.text(
        (ridge_point * x_max) ** 0.5,
        (y_min * peak_flops) ** 0.30,
        "COMPUTE\nBOUND",
        color=colors["OrangeLine"],
        fontsize=11,
        fontweight="bold",
        alpha=0.25,
        ha="center",
        va="center",
    )

    # Plot workloads with collision-free label placement
    if resolved_workloads:
        workload_colors = [
            colors["primary"],
            colors["GreenLine"],
            colors["VioletLine"],
            colors["BrownLine"],
        ]
        for i, (model, ai, perf) in enumerate(resolved_workloads):
            c = workload_colors[i % len(workload_colors)]
            bound = "memory" if ai < ridge_point else "compute"
            ax.plot(ai, perf, "o", color=c, markersize=8.5, zorder=11)

            if ai < ridge_point:
                if ai < 2.0:
                    # Low-AI workload near bottom-left: place label to the right inside the shaded region
                    text_xy = (ai * 3.8, perf * (0.55 * (0.55**i)))
                    ha_align, va_align = "left", "center"
                else:
                    # Mid/high-AI memory-bound workload: place label in the empty upper-left white region
                    text_xy = (ai * 0.12, perf * (0.72 * (0.55**i)))
                    ha_align, va_align = "center", "center"
            else:
                # Compute-bound workload: place label below the flat roof, offset from ridge label
                text_xy = (max(ai * 0.85, ridge_point * 2.2), peak_flops * (0.06 * (0.55**i)))
                ha_align, va_align = "center", "top"

            ax.annotate(
                f"{model.name}\n({bound}-bound)",
                xy=(ai, perf),
                xytext=text_xy,
                fontsize=MIN_FONT_PT,
                fontweight="bold",
                color=c,
                ha=ha_align,
                va=va_align,
                arrowprops=dict(arrowstyle="->", color=c, lw=1.1, shrinkB=5),
                zorder=12,
            )

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    # The document caption owns the figure title; pass `title` only for
    # standalone use outside the book.
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=MIN_FONT_PT, framealpha=0.9)
    return _finalize_web_figure(fig), ax


def plot_distributed_roofline(fleet_system, workloads=None):
    """
    Plots a Distributed (Network) Roofline Model for a given fleet.
    Maps Performance (TFLOP/s) against Communication Intensity (FLOP/Byte).

    Timeless Principle: The Network Wall is hit when gradient/activation
    synchronization exceeds the fleet's bisection bandwidth.
    """
    # 1. PARAMETERS (Fleet Specs)
    peak_flops = (
        fleet_system.node.accelerator.compute.peak_flops.to("TFLOPs/s").magnitude
        * fleet_system.count
    )
    # Bisection Bandwidth (total network capacity for All-Reduce)
    peak_network_bw = fleet_system.fabric.bandwidth.to("GB/s").magnitude
    ridge_point = peak_flops / (peak_network_bw / 1000)  # FLOP/Byte

    # 2. AXIS RANGE
    x_min = 0.1
    x_max = max(10000.0, 10.0 ** np.ceil(np.log10(ridge_point * 50.0)))
    y_min, y_max = peak_flops * 0.0005, peak_flops * 3.0
    x = np.logspace(np.log10(x_min), np.log10(x_max), 500)

    # 3. ROOFLINE CURVES
    y_net = peak_network_bw * x / 1000  # Network BW * CI, converted to TFLOP/s
    y_compute = np.full_like(x, peak_flops)
    y_roof = np.minimum(y_net, y_compute)

    # 4. PLOT
    fig, ax, colors, _ = setup_plot(figsize=(BODY_FIGSIZE[0], 4.6))

    # Shaded regions
    net_mask = x <= ridge_point
    comp_mask = x >= ridge_point
    ax.fill_between(
        x[net_mask],
        y_min,
        y_roof[net_mask],
        color=colors["RedL"],
        alpha=0.5,
        label="Network-bound region",
    )
    ax.fill_between(
        x[comp_mask],
        y_min,
        y_roof[comp_mask],
        color=colors["BlueFill"],
        alpha=0.5,
        label="Compute-bound region",
    )

    # Roofline line
    ax.loglog(
        x,
        y_roof,
        color=colors["VioletLine"],
        linewidth=2.5,
        zorder=5,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Network bandwidth ceiling label
    x_slope_start = max(x_min, y_min * 1000.0 / peak_network_bw)
    slope_x = _pick_slope_label_x(x_slope_start, ridge_point, [])
    slope_y = peak_network_bw * slope_x / 1000
    slope_angle = _slope_screen_angle(
        ax, x_slope_start, peak_network_bw * x_slope_start / 1000, ridge_point, peak_flops
    )
    ax.text(
        slope_x,
        slope_y * 1.38,
        f"Bisection BW: {peak_network_bw:.0f} GB/s",
        color=colors["RedLine"],
        fontsize=MIN_FONT_PT,
        fontweight="bold",
        rotation=slope_angle,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
    )

    # Fleet Compute ceiling label
    ax.text(
        x_max * 0.94,
        peak_flops * 1.24,
        f"Fleet Compute: {peak_flops:.0f} TFLOP/s",
        color=colors["BlueLine"],
        fontsize=MIN_FONT_PT,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    # Ridge point
    ax.plot(
        ridge_point,
        peak_flops,
        "X",
        color=colors["crimson"],
        markersize=10,
        zorder=10,
    )
    ridge_label_x = (ridge_point * x_max) ** 0.45
    ridge_str = f"{ridge_point:,.0f}" if ridge_point >= 1000 else f"{ridge_point:.1f}"
    ax.annotate(
        f"Network Ridge\n{ridge_str} FLOP/Byte",
        xy=(ridge_point, peak_flops),
        xytext=(ridge_label_x, peak_flops * 0.24),
        fontsize=MIN_FONT_PT,
        fontweight="bold",
        color=colors["crimson"],
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="->", color=colors["crimson"], lw=1.2),
    )

    # Region labels
    ax.text(
        (x_slope_start * ridge_point) ** 0.52,
        (y_min * peak_flops) ** 0.36,
        "NETWORK\nBOUND",
        color=colors["RedLine"],
        fontsize=11,
        fontweight="bold",
        alpha=0.25,
        ha="center",
        va="center",
    )
    ax.text(
        (ridge_point * x_max) ** 0.5,
        (y_min * peak_flops) ** 0.36,
        "COMPUTE\nBOUND",
        color=colors["BlueLine"],
        fontsize=11,
        fontweight="bold",
        alpha=0.25,
        ha="center",
        va="center",
    )

    ax.set_xlabel("Communication Intensity (FLOP/Byte)")
    ax.set_ylabel("Fleet Performance (TFLOP/s)")
    ax.set_title(
        f"Distributed Roofline: {fleet_system.count}x {fleet_system.node.accelerator.name}"
    )
    ax.legend(loc="lower right", fontsize=MIN_FONT_PT, framealpha=0.9)
    return _finalize_web_figure(fig), ax

def _metric_as_float(value, unit=None):
    """Return a numeric metric, converting Pint quantities when needed."""
    if hasattr(value, "m_as") and unit is not None:
        return float(value.m_as(unit))
    if hasattr(value, "to") and unit is not None:
        return float(value.to(unit).magnitude)
    if hasattr(value, "magnitude"):
        return float(value.magnitude)
    return float(value)

def _metric_ratio(numerator, denominator):
    """Return numerator / denominator for scalar or Pint quantity metrics."""
    if hasattr(numerator, "to") and hasattr(denominator, "units"):
        return float((numerator / denominator).to_base_units().magnitude)
    return float(numerator) / float(denominator)

def _memory_utilization(metrics):
    """Read or derive memory utilization from an evaluation metrics payload."""
    if "memory_utilization" in metrics:
        return float(metrics["memory_utilization"])
    if "weight_size" in metrics and "capacity" in metrics:
        return _metric_ratio(metrics["weight_size"], metrics["capacity"])
    if "memory_used_gb" in metrics and "memory_capacity_gb" in metrics:
        return _metric_ratio(metrics["memory_used_gb"], metrics["memory_capacity_gb"])
    return 0.0

def plot_evaluation_scorecard(evaluation):
    """
    Visualizes the supply-vs-demand scorecard for a SystemEvaluation.
    Follows the compact, labeled visualization pattern used by generated examples.
    """
    l1_metrics = evaluation.feasibility.metrics
    l2_metrics = evaluation.performance.metrics
    
    # 2. CALCULATION
    l1_ratio = _memory_utilization(l1_metrics)
    latency_ms = _metric_as_float(l2_metrics.get("latency", 0.0), "ms")
    sla_ms = _metric_as_float(
        l2_metrics.get("sla_latency_ms", l2_metrics.get("sla_latency", 1000.0)),
        "ms",
    )
    l2_ratio = latency_ms / sla_ms if sla_ms > 0 else 0.0
    
    levels = ['Memory (RAM)', 'Latency (SLA)']
    ratios = [l1_ratio, l2_ratio]
    
    # 3. OUTPUT (Visualization)
    fig, ax, colors, _plt = setup_plot(figsize=(BODY_FIGSIZE[0], 3.6))
    bar_colors = [colors['RedLine'] if r > 1.0 else colors['GreenLine'] for r in ratios]
    bars = ax.barh(levels, ratios, color=bar_colors, alpha=0.7, edgecolor='black')
    
    ax.axvline(1.0, color=colors['primary'], linestyle='--', linewidth=2, label='Physical Limit / SLA')
    
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, f"{ratio:.1%}", 
                va='center', fontweight='bold', color=bar_colors[i])

    ax.set_xlim(0, max(max(ratios) + 0.5, 1.5))
    ax.set_xlabel('Resource Utilization (Demand / Supply)')
    ax.set_title(f'System Evaluation: {evaluation.scenario_name}')
    return _finalize_web_figure(fig), ax
