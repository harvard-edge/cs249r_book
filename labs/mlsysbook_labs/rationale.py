"""
Rationale-first pedagogical feedback engine for MLSysBook Co-Labs.

Bridges student hypotheses with mlsysim.Engine.solve() physics and provides
automated senior-architect level differential critique, regime-shift detection,
and literature-grounded feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import marimo as mo
import plotly.graph_objects as go
import numpy as np

from mlsysim.labs.style import COLORS, apply_plotly_theme


@dataclass(frozen=True)
class RationaleChallenge:
    """A pedagogical challenge requiring both a prediction and a physical mechanism."""

    question: str
    metric_label: str
    options: Dict[str, str]  # id -> display label for prediction (e.g. {"2x": "2.0x faster", ...})
    mechanisms: Dict[str, str]  # id -> physical mechanism description
    correct_option: str
    correct_mechanism: str
    concept_title: str
    chapter_reference: str
    literature_source: str
    fallacy_explanation: str


@dataclass(frozen=True)
class RationaleEvaluation:
    """Result of evaluating a student's prediction and rationale against mlsysim."""

    prediction_correct: bool
    mechanism_correct: bool
    actual_metric_value: float
    baseline_metric_value: float
    delta_ratio: float
    regime_shifted: bool
    baseline_regime: str
    proposal_regime: str
    binding_constraint: str
    critique_markdown: str


def evaluate_rationale(
    challenge: RationaleChallenge,
    student_prediction: str,
    student_mechanism: str,
    baseline_profile: Any,
    proposal_profile: Any,
    metric_extractor: Optional[Any] = None,
) -> RationaleEvaluation:
    """Compare student hypothesis and rationale against simulated ground truth."""

    b_prof = baseline_profile
    p_prof = proposal_profile

    # Determine regimes
    b_neck = getattr(b_prof, "bottleneck", "Unknown")
    p_neck = getattr(p_prof, "bottleneck", "Unknown")
    b_is_mem = "memory" in str(b_neck).lower()
    p_is_mem = "memory" in str(p_neck).lower()
    regime_shifted = b_is_mem != p_is_mem

    # Compute speedup / throughput ratio
    if metric_extractor:
        b_val, p_val = metric_extractor(b_prof, p_prof)
    else:
        b_lat = float(getattr(b_prof, "latency").m_as("ms"))
        p_lat = float(getattr(p_prof, "latency").m_as("ms"))
        b_val = b_lat
        p_val = p_lat

    delta_ratio = (b_val / p_val) if p_val > 0 else 0.0

    prediction_correct = student_prediction == challenge.correct_option
    mechanism_correct = student_mechanism == challenge.correct_mechanism

    # Determine binding constraint
    binding_constraint = p_neck
    if hasattr(p_prof, "constraint_trace") and p_prof.constraint_trace:
        binding_constraint = p_prof.constraint_trace[-1]

    # Generate Senior Architect Critique Markdown
    lines = []
    if prediction_correct and mechanism_correct:
        status_icon = "🟢"
        header = f"### {status_icon} Architect Audit: Verified First-Principles Prediction"
        lines.append(header)
        lines.append(
            f"**Outstanding reasoning.** Your hypothesis ({challenge.options.get(student_prediction, student_prediction)}) "
            f"and physical mechanism were confirmed by the simulator."
        )
    elif mechanism_correct and not prediction_correct:
        status_icon = "🟡"
        header = f"### {status_icon} Architect Audit: Correct Physical Mechanism, Magnitude Calibration Needed"
        lines.append(header)
        lines.append(
            f"**Good physical intuition, but magnitude differed.** You correctly identified the binding mechanism "
            f"(`{challenge.mechanisms.get(student_mechanism, student_mechanism)}`), but the quantitative impact was "
            f"`{delta_ratio:.2f}x` rather than your prediction."
        )
    else:
        status_icon = "🔴"
        header = f"### {status_icon} Architect Audit: Intuition Trap / Regime Shift Detected"
        lines.append(header)
        lines.append(
            f"**Warning: Systems Intuition Trap.** You selected: *\"{challenge.mechanisms.get(student_mechanism, student_mechanism)}\"*. "
            f"However, the physical simulation shows the active constraint is **{p_neck}**."
        )

    lines.append("")
    lines.append("#### 🔬 Physics & Regime Breakdown")
    lines.append(f"* **Baseline Regime:** `{b_neck}` ({b_val:.2f} ms)")
    lines.append(f"* **Proposal Regime:** `{p_neck}` ({p_val:.2f} ms)")
    lines.append(f"* **Attained Speedup:** `{delta_ratio:.2f}x`")

    if regime_shifted:
        lines.append(
            f"\n> ⚠️ **Regime Shift:** The workload transitioned from **{b_neck}** to **{p_neck}**. "
            "At small scale, memory bandwidth bounded execution. Once operational intensity crossed the machine ridge point, "
            "compute throughput became the binding wall."
        )

    lines.append(f"\n**Fallacy Debunked:** {challenge.fallacy_explanation}")
    lines.append(f"\n*Source Reference:* {challenge.literature_source} · *Book Chapter:* {challenge.chapter_reference}")

    critique_md = "\n".join(lines)

    return RationaleEvaluation(
        prediction_correct=prediction_correct,
        mechanism_correct=mechanism_correct,
        actual_metric_value=p_val,
        baseline_metric_value=b_val,
        delta_ratio=delta_ratio,
        regime_shifted=regime_shifted,
        baseline_regime=str(b_neck),
        proposal_regime=str(p_neck),
        binding_constraint=str(binding_constraint),
        critique_markdown=critique_md,
    )


def render_interactive_roofline(
    hardware: Any,
    points: List[Tuple[str, float, float, str]],  # [(label, ai, gflops, color)]
    title: str = "Hardware Roofline & Operating Regimes",
) -> go.Figure:
    """Render an interactive Plotly roofline model showing memory vs compute regimes."""

    peak_flops = getattr(hardware, "peak_flops", None) or hardware.compute.peak_flops
    memory_bw = getattr(hardware, "memory_bw", None) or hardware.memory.bandwidth

    peak_gflops = peak_flops.to("GFLOPs/s").magnitude
    bw_gbs = memory_bw.to("GB/s").magnitude
    ridge_ai = peak_gflops / bw_gbs if bw_gbs > 0 else 1.0

    x_range = np.logspace(-1, 4, 200)
    y_roof = np.minimum(x_range * bw_gbs, peak_gflops)

    fig = go.Figure()

    # Hardware Roofline Ceiling
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_roof,
            mode="lines",
            name="Hardware Limit (Roofline)",
            line=dict(color=COLORS["Grey"], width=3),
            fill="tozeroy",
            fillcolor="rgba(226, 232, 240, 0.25)",
        )
    )

    # Ridge Line Annotation
    fig.add_vline(
        x=ridge_ai,
        line_dash="dash",
        line_color=COLORS["OrangeLine"],
        annotation_text=f"Machine Ridge: {ridge_ai:.1f} FLOPs/B",
        annotation_position="top left",
    )

    # Shaded Zones: Memory-Bound (Left) vs Compute-Bound (Right)
    fig.add_vrect(
        x0=0.1,
        x1=ridge_ai,
        fillcolor="rgba(203, 32, 45, 0.04)",
        layer="below",
        line_width=0,
        annotation_text="Memory-Bound Regime",
        annotation_position="bottom left",
    )
    fig.add_vrect(
        x0=ridge_ai,
        x1=10000,
        fillcolor="rgba(0, 99, 149, 0.04)",
        layer="below",
        line_width=0,
        annotation_text="Compute-Bound Regime",
        annotation_position="bottom right",
    )

    # Plot operating points
    for label, ai, gflops, color in points:
        fig.add_trace(
            go.Scatter(
                x=[ai],
                y=[gflops],
                mode="markers+text",
                name=label,
                text=[f"  {label} ({ai:.1f}, {gflops:,.0f})"],
                textposition="top right",
                marker=dict(size=14, color=color, symbol="diamond", line=dict(color="white", width=2)),
            )
        )

    fig.update_layout(
        title=dict(text=title, y=0.96, font=dict(size=14, color=COLORS["Text"])),
        xaxis=dict(
            title="Arithmetic Intensity (FLOPs / Byte)",
            type="log",
            range=[-1, 4],
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(
            title="Attained Throughput (GFLOPs / s)",
            type="log",
            range=[1, 6.5],
            gridcolor="#f1f5f9",
        ),
        height=380,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", y=-0.25, x=0),
    )

    return apply_plotly_theme(fig)


def render_latency_breakdown(
    baseline_profile: Any,
    proposal_profile: Any,
    labels: Tuple[str, str] = ("Baseline", "Proposal"),
) -> go.Figure:
    """Render a clean waterfall/bar breakdown of compute vs memory vs overhead latency."""

    b_comp = baseline_profile.latency_compute.to("ms").magnitude
    b_mem = baseline_profile.latency_memory.to("ms").magnitude
    b_ovh = baseline_profile.latency_overhead.to("ms").magnitude

    p_comp = proposal_profile.latency_compute.to("ms").magnitude
    p_mem = proposal_profile.latency_memory.to("ms").magnitude
    p_ovh = proposal_profile.latency_overhead.to("ms").magnitude

    categories = ["Compute Time", "Memory Transfer Time", "Framework Overhead"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name=labels[0],
            x=categories,
            y=[b_comp, b_mem, b_ovh],
            marker_color="rgba(148, 163, 184, 0.7)",
        )
    )

    fig.add_trace(
        go.Bar(
            name=labels[1],
            x=categories,
            y=[p_comp, p_mem, p_ovh],
            marker_color=COLORS["BlueLine"],
        )
    )

    fig.update_layout(
        title="Latency Component Decomposition",
        yaxis=dict(title="Execution Time (ms)", gridcolor="#f1f5f9"),
        barmode="group",
        height=300,
        margin=dict(l=60, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.15, x=0),
    )

    return apply_plotly_theme(fig)


def render_distributed_memory_breakdown(
    weights_gb: float,
    grads_gb: float,
    opt_gb: float,
    act_gb: float,
    vram_capacity_gb: float = 80.0,
    title: str = "Per-GPU VRAM Allocation vs Capacity Wall",
) -> go.Figure:
    """Render a stacked bar chart of per-GPU memory components against VRAM capacity."""
    total_gb = weights_gb + grads_gb + opt_gb + act_gb
    is_oom = total_gb > vram_capacity_gb

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Model Weights",
            y=["VRAM"],
            x=[weights_gb],
            orientation="h",
            marker=dict(color="#006395"),
            text=[f"{weights_gb:.1f} GB"] if weights_gb > 2 else [],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Gradients",
            y=["VRAM"],
            x=[grads_gb],
            orientation="h",
            marker=dict(color="#4A777A"),
            text=[f"{grads_gb:.1f} GB"] if grads_gb > 2 else [],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Adam Optimizer States",
            y=["VRAM"],
            x=[opt_gb],
            orientation="h",
            marker=dict(color="#A51C30" if is_oom else "#E06D53"),
            text=[f"{opt_gb:.1f} GB"] if opt_gb > 2 else [],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Activations",
            y=["VRAM"],
            x=[act_gb],
            orientation="h",
            marker=dict(color="#D97706"),
            text=[f"{act_gb:.1f} GB"] if act_gb > 2 else [],
            textposition="inside",
        )
    )

    fig.add_vline(
        x=vram_capacity_gb,
        line_dash="dash",
        line_color="#A51C30",
        line_width=2.5,
        annotation_text=f"VRAM Capacity: {vram_capacity_gb:.0f} GB ({'OOM 💥' if is_oom else 'SAFE ✅'})",
        annotation_position="top right",
    )

    max_x = max(vram_capacity_gb * 1.2, total_gb * 1.05)

    fig.update_layout(
        title=dict(text=f"{title} (Total: {total_gb:.1f} GB / {vram_capacity_gb:.0f} GB)", font=dict(size=14, color=COLORS["Text"])),
        barmode="stack",
        xaxis=dict(title="Allocated Memory (GB)", range=[0, max_x], gridcolor="#F1F5F9"),
        yaxis=dict(showticklabels=False),
        height=190,
        margin=dict(l=30, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.28, x=0),
    )
    return apply_plotly_theme(fig)


def render_distributed_step_breakdown(
    compute_ms: float,
    tp_comm_ms: float,
    dp_comm_ms: float,
    bubble_ms: float,
    tp_spans_nodes: bool = False,
    title: str = "Training Step Latency Decomposition",
) -> go.Figure:
    """Render a stacked bar breakdown of compute vs TP comm vs DP comm vs PP bubble."""
    total_ms = compute_ms + tp_comm_ms + dp_comm_ms + bubble_ms
    fig = go.Figure()

    tp_color = "#A51C30" if tp_spans_nodes else "#006395"
    tp_name = "TP Comm (Cross-Node InfiniBand ⚠️)" if tp_spans_nodes else "TP Comm (Intra-Node NVLink ✅)"

    fig.add_trace(
        go.Bar(
            name="Forward + Backward Compute",
            y=["Step Latency"],
            x=[compute_ms],
            orientation="h",
            marker=dict(color="#4A777A"),
            text=[f"{compute_ms:.0f} ms ({compute_ms/total_ms*100:.1f}%)"] if total_ms > 0 and compute_ms / total_ms > 0.08 else [],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            name=tp_name,
            y=["Step Latency"],
            x=[tp_comm_ms],
            orientation="h",
            marker=dict(color=tp_color),
            text=[f"{tp_comm_ms:.0f} ms ({tp_comm_ms/total_ms*100:.1f}%)"] if total_ms > 0 and tp_comm_ms / total_ms > 0.08 else [],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="DP AllReduce Sync",
            y=["Step Latency"],
            x=[dp_comm_ms],
            orientation="h",
            marker=dict(color="#D97706"),
            text=[f"{dp_comm_ms:.0f} ms ({dp_comm_ms/total_ms*100:.1f}%)"] if total_ms > 0 and dp_comm_ms / total_ms > 0.08 else [],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="1F1B Pipeline Bubble Idle",
            y=["Step Latency"],
            x=[bubble_ms],
            orientation="h",
            marker=dict(color="#94A3B8"),
            text=[f"{bubble_ms:.0f} ms ({bubble_ms/total_ms*100:.1f}%)"] if total_ms > 0 and bubble_ms / total_ms > 0.08 else [],
            textposition="inside",
        )
    )

    fig.update_layout(
        title=dict(text=f"{title} (Total: {total_ms:,.0f} ms)", font=dict(size=14, color=COLORS["Text"])),
        barmode="stack",
        xaxis=dict(title="Execution Time (ms)", gridcolor="#F1F5F9"),
        yaxis=dict(showticklabels=False),
        height=190,
        margin=dict(l=30, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.28, x=0),
    )
    return apply_plotly_theme(fig)
