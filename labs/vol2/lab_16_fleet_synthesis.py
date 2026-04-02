import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-16: THE FLEET SYNTHESIS (CAPSTONE)
#
# Volume II, Chapter 16 — Conclusion
#
# THE FINAL LAB OF THE ENTIRE TWO-VOLUME CURRICULUM.
#
# Core Invariant: No single architectural decision satisfies all constraints
#   independently. The binding constraint shifts with scale. The art of
#   distributed ML engineering is not maximizing any single dimension but
#   finding the best compromise across all of them.
#
# 4 Parts (~52 minutes):
#   Part A — The Sensitivity Wall (12 min)
#   Part B — The Failure Budget (10 min)
#   Part C — The Principle Interaction Map (15 min)
#   Part D — The Fleet Architecture Blueprint (15 min)
#
# Design Ledger: reads from V2-14 (carbon cap) and V2-15 (fairness overhead).
#   Provides sensible defaults for students who haven't completed prior labs.
#
# Design Ledger: saves chapter="v2_16"
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 0: SETUP ────────────────────────────────────────────────────────
@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import numpy as np

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install(
            "../../../wheels/mlsysim-0.1.0-py3-none-any.whl", keep_going=False
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog
    from mlsysim.hardware.registry import Hardware
    from mlsysim.models.registry import Models

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()

    # ── Load Design Ledger from prior labs with defaults ────────────────────
    _ch14 = ledger.get_design(14) or {}
    _ch15 = ledger.get_design(15) or {}

    # V2-14 defaults (carbon)
    CARBON_CAP = _ch14.get("carbon_cap", 0.8)       # fraction of baseline
    CARBON_STRATEGY = _ch14.get("geographic_strategy", "quebec")

    # V2-15 defaults (fairness)
    FAIRNESS_METRIC = _ch15.get("fairness_metric", "eqop")         # Equal Opportunity
    FAIRNESS_OVERHEAD_MS = _ch15.get("fairness_overhead_ms", 15)    # ms per request
    FAIRNESS_THRESHOLD = _ch15.get("fairness_disparity_threshold", 0.05)

    # ── Hardware from registry ──────────────────────────────────────────────
    _cloud = Hardware.Cloud.H100
    _edge  = Hardware.Edge.JetsonOrinNX

    # ── Fleet constants ──────────────────────────────────────────────────────
    H100_TFLOPS_FP16    = _cloud.compute.peak_flops.m_as("TFLOPs/s")  # 989
    H100_RAM_GB         = _cloud.memory.capacity.m_as("GiB")          # 80 GiB
    H100_TDP_W          = _cloud.tdp.m_as("W")                        # 700 W
    NVLINK_BW_GBS       = 900     # GB/s NVLink4 bidirectional per GPU
    IB_BW_GBPS          = 400     # Gb/s InfiniBand NDR per port
    MTBF_GPU_HOURS      = 2000    # Mean time between failures per GPU (hours)
    CHECKPOINT_COST_S   = 120     # Seconds per checkpoint

    # Edge tier — for fleet heterogeneity comparison
    EDGE_TFLOPS_FP16    = _edge.compute.peak_flops.m_as("TFLOPs/s")  # 25
    EDGE_RAM_GB         = _edge.memory.capacity.m_as("GB")            # 16
    EDGE_TDP_W          = _edge.tdp.m_as("W")                         # 25 W

    # Communication constants
    ALLREDUCE_RING_EFF  = 0.85    # AllReduce ring efficiency
    GRADIENT_COMPRESS   = 0.1     # 10x compression ratio (Top-K)

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        CARBON_CAP, CARBON_STRATEGY, FAIRNESS_METRIC,
        FAIRNESS_OVERHEAD_MS, FAIRNESS_THRESHOLD,
        H100_TFLOPS_FP16, H100_RAM_GB, H100_TDP_W,
        EDGE_TFLOPS_FP16, EDGE_RAM_GB, EDGE_TDP_W,
        NVLINK_BW_GBS, IB_BW_GBPS, MTBF_GPU_HOURS, CHECKPOINT_COST_S,
        ALLREDUCE_RING_EFF, GRADIENT_COMPRESS,
        DecisionLog,
    )


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS, CARBON_CAP, FAIRNESS_METRIC, FAIRNESS_OVERHEAD_MS):
    _metric_display = {"dp": "Demographic Parity", "eo": "Equalized Odds", "eqop": "Equal Opportunity"}.get(FAIRNESS_METRIC, "Equal Opportunity")
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 40%, #16213e 70%, #0f3460 100%);
                    padding: 40px 48px; border-radius: 16px; color: white;
                    box-shadow: 0 12px 48px rgba(0,0,0,0.45);
                    border: 1px solid rgba(99,102,241,0.2);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 16 &middot; Capstone
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.6rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Fleet Synthesis
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #a5b4fc; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Everything connects. Every optimization creates a new bottleneck.
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 700px; line-height: 1.65;">
                You have spent 15 labs learning individual principles: communication, fault
                tolerance, inference, performance, edge intelligence, operations, security,
                robustness, sustainability, and fairness. This capstone reveals that these
                principles interact as a coupled system. The binding constraint shifts with scale.
                The art is not maximizing any single dimension but finding the best compromise.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.25); color: #c7d2fe;
                             padding: 6px 16px; border-radius: 20px; font-size: 0.85rem;
                             font-weight: 700; border: 1px solid rgba(99,102,241,0.4);">
                    CAPSTONE &middot; 4 Parts &middot; ~52 min
                </span>
                <span style="background: rgba(0,143,69,0.15); color: #6ee7b7;
                             padding: 6px 16px; border-radius: 20px; font-size: 0.85rem;
                             font-weight: 600; border: 1px solid rgba(0,143,69,0.25);">
                    Final Lab of the Curriculum
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px;">
                <span class="badge badge-info">Sensitivity Analysis</span>
                <span class="badge badge-warn">Failure Budget</span>
                <span class="badge badge-ok">Principle Interactions</span>
                <span class="badge badge-fail">Fleet Blueprint</span>
            </div>
            <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 8px; padding: 12px 16px; margin-top: 8px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: #94a3b8;
                            text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
                    Design Ledger Inputs (from prior labs)
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                    Carbon cap: <strong>{CARBON_CAP:.0%}</strong> of baseline &middot;
                    Fairness metric: <strong>{_metric_display}</strong> &middot;
                    Fairness overhead: <strong>{FAIRNESS_OVERHEAD_MS}ms</strong>
                </div>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ──────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid #6366f1;
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Identify the binding constraint</strong>: demonstrate that communication (not compute) is the most sensitive system dimension at 1,000+ GPUs, and watch the sensitivity ordering flip as fleet size changes.</div>
                <div style="margin-bottom: 3px;">2. <strong>Calculate fleet failure rates</strong>: apply MTBF/N to show that a 10,000-GPU cluster has MTBF of 1 hour, and find the Young-Daly optimal checkpoint interval that maximizes goodput.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a fleet architecture</strong> that achieves &ge;50&times; effective system gain over single-GPU baseline while keeping all six principle axes (compute, communication, fault tolerance, scheduling, sustainability, fairness) within acceptable zones.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Prerequisites</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    All Volume II chapters &middot; Labs V2-13 (Robustness), V2-14 (Carbon), V2-15 (Fairness)
                    recommended but not required (defaults provided)
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Duration</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~52 min</strong><br/>A: 12 &middot; B: 10 &middot; C: 15 &middot; D: 15
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: #6366f1;
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Core Question</div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;If every optimization creates a new bottleneck, and the binding constraint
                shifts with every decision you make, how do you design a fleet that balances
                computation, communication, and coordination under sustainability and fairness
                constraints?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: READING ───────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- This is the capstone lab. Ideally, complete all Volume II
    labs first. At minimum, read the Conclusion chapter and review the C-Cube framework
    (Computation, Communication, Coordination) with sustainability and fairness as
    cross-cutting constraints.
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET CELLS
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 4: Part A prediction + Part A fleet toggle ──────────────────────
@app.cell(hide_code=True)
def _(mo):
    partA_pred = mo.ui.radio(
        options={
            "A) 10% more FLOPS per GPU": "compute",
            "B) 10% more network bandwidth": "communication",
            "C) 10% better fault tolerance (fewer restarts)": "fault",
            "D) 10% better scheduling (less idle time)": "scheduling",
        },
        label="A 1,000-GPU training cluster. Which 10% improvement yields the largest throughput gain?",
    )
    partA_fleet_toggle = mo.ui.radio(
        options={"8 GPUs": 8, "64 GPUs": 64, "1,000 GPUs": 1000, "10,000 GPUs": 10000},
        value="1,000 GPUs", label="Fleet size:", inline=True,
    )
    return (partA_pred, partA_fleet_toggle,)


# ─── CELL 5: Part B prediction + Part B sliders ──────────────────────────
@app.cell(hide_code=True)
def _(mo, partA_pred):
    mo.stop(partA_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partB_pred = mo.ui.number(
        start=0.01, stop=1000, value=100, step=1,
        label="10,000 GPUs, each with 10,000-hour MTBF. What is the cluster MTBF (hours)?",
    )
    partB_fleet_slider = mo.ui.slider(start=10, stop=100000, value=10000, step=100, label="Fleet size (GPUs)")
    partB_mtbf_slider = mo.ui.slider(start=1000, stop=100000, value=10000, step=1000, label="Per-GPU MTBF (hours)")
    partB_ckpt_slider = mo.ui.slider(start=1, stop=30, value=10, step=1, label="Checkpoint interval (minutes)")
    return (partB_pred, partB_fleet_slider, partB_mtbf_slider, partB_ckpt_slider,)


# ─── CELL 6: Part C prediction + Part C sliders ──────────────────────────
@app.cell(hide_code=True)
def _(mo, partB_pred):
    mo.stop(partB_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partC_pred = mo.ui.radio(
        options={
            "A) No effect -- they are independent": "independent",
            "B) Slight degradation -- compressed checkpoints are less reliable": "slight",
            "C) Significant degradation -- compressed gradients increase sensitivity to bit errors": "significant",
            "D) Improvement -- faster communication means faster checkpointing": "improvement",
        },
        label="You push communication efficiency to 99% (aggressive gradient compression). What happens to fault tolerance?",
    )
    partC_compute = mo.ui.slider(start=30, stop=100, value=70, step=5, label="Compute (%)")
    partC_comm = mo.ui.slider(start=30, stop=100, value=70, step=5, label="Communication (%)")
    partC_fault = mo.ui.slider(start=30, stop=100, value=70, step=5, label="Fault Tolerance (%)")
    partC_sched = mo.ui.slider(start=30, stop=100, value=70, step=5, label="Scheduling (%)")
    partC_sustain = mo.ui.slider(start=30, stop=100, value=70, step=5, label="Sustainability (%)")
    partC_fair = mo.ui.slider(start=30, stop=100, value=70, step=5, label="Fairness (%)")
    return (partC_pred, partC_compute, partC_comm, partC_fault, partC_sched, partC_sustain, partC_fair,)


# ─── CELL 7: Part D prediction + Part D controls ─────────────────────────
@app.cell(hide_code=True)
def _(mo, partC_pred):
    mo.stop(partC_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partD_refl = mo.ui.radio(
        options={
            "A) Yes -- fairness overhead for communication efficiency": "yes_trade",
            "B) No -- fairness is non-negotiable": "no_trade",
            "C) It depends on the deployment context": "depends",
        },
        label="Would you trade fairness overhead for communication efficiency?",
    )
    partD_gpus = mo.ui.slider(start=8, stop=10000, value=1000, step=8, label="GPU count")
    partD_comm_strategy = mo.ui.dropdown(
        options={"AllReduce Ring": "ring", "AllReduce Tree": "tree", "Gradient Compression (10x)": "compress"},
        value="AllReduce Ring", label="Communication strategy:",
    )
    partD_ckpt_freq = mo.ui.slider(start=1, stop=30, value=10, step=1, label="Checkpoint freq (min)")
    partD_scheduling = mo.ui.dropdown(
        options={"Static": "static", "Elastic": "elastic"},
        value="Static", label="Scheduling mode:",
    )
    return (partD_refl, partD_gpus, partD_comm_strategy, partD_ckpt_freq, partD_scheduling,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: SINGLE TABS CELL
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(
    mo, go, np, math, apply_plotly_theme, COLORS,
    CARBON_CAP, CARBON_STRATEGY, FAIRNESS_METRIC,
    FAIRNESS_OVERHEAD_MS, FAIRNESS_THRESHOLD,
    H100_TFLOPS_FP16, H100_RAM_GB, H100_TDP_W,
    NVLINK_BW_GBS, IB_BW_GBPS, MTBF_GPU_HOURS, CHECKPOINT_COST_S,
    ALLREDUCE_RING_EFF, GRADIENT_COMPRESS,
    partA_pred, partA_fleet_toggle,
    partB_pred, partB_fleet_slider, partB_mtbf_slider, partB_ckpt_slider,
    partC_pred, partC_compute, partC_comm, partC_fault, partC_sched, partC_sustain, partC_fair,
    partD_refl, partD_gpus, partD_comm_strategy, partD_ckpt_freq, partD_scheduling,
    ledger,
):
    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER -- The Sensitivity Wall
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
        <div id="part-a" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: #6366f1; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">A</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part A &middot; 12 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px;">The Sensitivity Wall</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                At fleet scale (1,000+ GPUs), communication &mdash; not computation &mdash; is the
                most sensitive system dimension. A 10% network bandwidth improvement yields a larger
                throughput gain than 10% more FLOPS. But this flips at small scale. Watch the
                sensitivity ordering change as you scale.
            </div>
        </div>
        """))

        items.append(mo.md("### Your Prediction"))
        items.append(partA_pred)

        if partA_pred.value is None:
            items.append(mo.callout(mo.md("**Select your prediction to unlock.**"), kind="warn"))
            return mo.vstack(items)

        # Fleet toggle control
        items.append(partA_fleet_toggle)

        _N = partA_fleet_toggle.value

        # Sensitivity analysis: throughput impact of 10% improvement in each dimension
        _comm_fraction = min(0.7, 0.05 + 0.65 * (1 - 8 / _N))
        _compute_fraction = max(0.15, 0.8 - 0.65 * (1 - 8 / _N))
        _fault_fraction = min(0.15, 0.01 + 0.14 * (_N / 10000))
        _schedule_fraction = 0.05
        _sustain_fraction = min(0.05, 0.01 + 0.04 * (_N / 10000))
        _fairness_fraction = 0.02

        _total = _comm_fraction + _compute_fraction + _fault_fraction + _schedule_fraction + _sustain_fraction + _fairness_fraction
        _dims = ["Compute", "Communication", "Fault\nTolerance", "Scheduling", "Sustainability\nOverhead", "Fairness\nOverhead"]
        _sensitivities = [
            _compute_fraction / _total * 10,
            _comm_fraction / _total * 10,
            _fault_fraction / _total * 10,
            _schedule_fraction / _total * 10,
            _sustain_fraction / _total * 10,
            _fairness_fraction / _total * 10,
        ]

        _sorted_pairs = sorted(zip(_dims, _sensitivities), key=lambda x: x[1], reverse=True)
        _sorted_dims = [p[0] for p in _sorted_pairs]
        _sorted_vals = [p[1] for p in _sorted_pairs]

        _bar_colors = []
        for d in _sorted_dims:
            if "Compute" in d:
                _bar_colors.append(COLORS["BlueLine"])
            elif "Comm" in d:
                _bar_colors.append(COLORS["OrangeLine"])
            elif "Fault" in d:
                _bar_colors.append(COLORS["RedLine"])
            elif "Sched" in d:
                _bar_colors.append(COLORS["GreenLine"])
            else:
                _bar_colors.append(COLORS["TextMuted"])

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Bar(
            y=_sorted_dims, x=_sorted_vals, orientation="h",
            marker_color=_bar_colors,
            text=[f"{v:.1f}%" for v in _sorted_vals],
            textposition="outside",
        ))
        fig_sens.update_layout(
            height=350,
            xaxis=dict(title="Throughput Impact of 10% Improvement (%)"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=120),
        )
        apply_plotly_theme(fig_sens)

        _top_dim = _sorted_dims[0].replace("\n", " ")
        _top_val = _sorted_vals[0]

        items.append(mo.md(f"### Sensitivity Tornado (N = {_N:,} GPUs)"))
        items.append(mo.as_html(fig_sens))
        items.append(mo.callout(mo.md(
            f"At **{_N:,} GPUs**, the most sensitive dimension is **{_top_dim}** ({_top_val:.1f}% "
            f"throughput gain per 10% improvement). "
            + ("At this scale, communication dominates because AllReduce synchronization barriers "
               "amplify bandwidth bottlenecks nonlinearly." if "Comm" in _top_dim else
               "At this scale, compute still dominates. Try increasing fleet size to see the crossover.")
        ), kind="info"))

        if _N >= 1000:
            items.append(mo.callout(mo.md(
                "**The thesis of this book:** Scale creates qualitative change. "
                "At 8 GPUs, compute dominates. At 1,000+, communication dominates decisively. "
                "The same system behaves differently at different scales."
            ), kind="warn"))

        # Prediction reveal
        _correct = partA_pred.value == "communication"
        _msg = ("Correct. At 1,000 GPUs, network bandwidth (communication) is 2-3x more sensitive "
                "than compute. Students default to 'compute is king' even after 15 labs of learning otherwise."
                if _correct else
                "The answer is (B): network bandwidth. At 1,000 GPUs, the AllReduce synchronization "
                "cost dominates. The 2(N-1)/N * G/BW term approaches 2G/BW, making bandwidth the "
                "irreducible floor.")
        items.append(mo.callout(mo.md(f"**{_msg}**"), kind="success" if _correct else "warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER -- The Failure Budget
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
        <div id="part-b" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['OrangeLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">B</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part B &middot; 10 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px;">The Failure Budget</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                Communication dominates at scale. But the fleet also fails at scale &mdash; with
                mathematical certainty. Meta's Llama 3 training on 16,384 GPUs experienced 419
                failures in 54 days (one every 3 hours). Checkpointing overhead and recovery
                strategy dominate system design.
            </div>
        </div>
        """))

        items.append(mo.md("### Your Prediction"))
        items.append(partB_pred)

        if partB_pred.value is None:
            items.append(mo.callout(mo.md("**Enter your prediction to unlock.**"), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.hstack([partB_fleet_slider, partB_mtbf_slider, partB_ckpt_slider], justify="start", gap=1))

        _N = partB_fleet_slider.value
        _mtbf_gpu = partB_mtbf_slider.value
        _ckpt_min = partB_ckpt_slider.value

        # Cluster MTBF = per-GPU MTBF / N
        _mtbf_cluster = _mtbf_gpu / _N
        _mtbf_cluster_min = _mtbf_cluster * 60  # minutes

        # P(all up for 1 hour) = (1 - 1/MTBF_gpu)^N
        _p_all_up_1h = (1 - 1 / _mtbf_gpu) ** _N

        # Goodput: Young-Daly model
        _T_ckpt_min = CHECKPOINT_COST_S / 60  # minutes
        _T_optimal = math.sqrt(2 * _T_ckpt_min * _mtbf_cluster_min) if _mtbf_cluster_min > 0 else 10

        # Goodput at chosen interval
        _ckpt_overhead = _T_ckpt_min / _ckpt_min if _ckpt_min > 0 else 1
        _failure_waste = _ckpt_min / (2 * _mtbf_cluster_min) if _mtbf_cluster_min > 0 else 1
        _goodput = max(0, 1 - _ckpt_overhead - _failure_waste)

        # Goodput at optimal interval
        _ckpt_overhead_opt = _T_ckpt_min / _T_optimal if _T_optimal > 0 else 1
        _failure_waste_opt = _T_optimal / (2 * _mtbf_cluster_min) if _mtbf_cluster_min > 0 else 1
        _goodput_optimal = max(0, 1 - _ckpt_overhead_opt - _failure_waste_opt)

        # Goodput curve across intervals
        _intervals = np.linspace(1, 30, 60)
        _goodputs = []
        for _t in _intervals:
            _co = _T_ckpt_min / _t
            _fw = _t / (2 * _mtbf_cluster_min) if _mtbf_cluster_min > 0 else 1
            _goodputs.append(max(0, 1 - _co - _fw))

        fig_goodput = go.Figure()
        fig_goodput.add_trace(go.Scatter(
            x=_intervals, y=_goodputs, name="Goodput",
            line=dict(color=COLORS["BlueLine"], width=3),
            fill="tozeroy", fillcolor="rgba(0,99,149,0.08)",
        ))

        # Current operating point
        fig_goodput.add_trace(go.Scatter(
            x=[_ckpt_min], y=[_goodput], mode="markers",
            marker=dict(size=14, color=COLORS["OrangeLine"], line=dict(color="white", width=2)),
            name=f"Current ({_ckpt_min}min)",
        ))

        # Optimal point
        fig_goodput.add_trace(go.Scatter(
            x=[_T_optimal], y=[_goodput_optimal], mode="markers",
            marker=dict(size=14, color=COLORS["GreenLine"], symbol="star",
                        line=dict(color="white", width=2)),
            name=f"Young-Daly Optimal ({_T_optimal:.1f}min)",
        ))

        fig_goodput.update_layout(
            height=380,
            xaxis=dict(title="Checkpoint Interval (minutes)"),
            yaxis=dict(title="Goodput (fraction of useful work)", range=[0, 1.05]),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        )
        apply_plotly_theme(fig_goodput)

        _mtbf_color = COLORS["GreenLine"] if _mtbf_cluster > 10 else COLORS["OrangeLine"] if _mtbf_cluster > 1 else COLORS["RedLine"]
        _goodput_color = COLORS["GreenLine"] if _goodput > 0.8 else COLORS["OrangeLine"] if _goodput > 0.5 else COLORS["RedLine"]

        items.append(mo.md(f"### Fleet Reliability (N = {_N:,}, per-GPU MTBF = {_mtbf_gpu:,}h)"))
        items.append(mo.as_html(fig_goodput))
        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; justify-content: center; margin-top: 12px; flex-wrap: wrap;">
            <div style="padding: 14px 20px; border: 2px solid {_mtbf_color}; border-radius: 10px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Cluster MTBF</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {_mtbf_color};">
                    {_mtbf_cluster:.2f}h</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">{_mtbf_cluster*60:.0f} minutes</div>
            </div>
            <div style="padding: 14px 20px; border: 2px solid {_goodput_color}; border-radius: 10px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Current Goodput</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {_goodput_color};">
                    {_goodput:.1%}</div>
            </div>
            <div style="padding: 14px 20px; border: 2px solid {COLORS['GreenLine']}; border-radius: 10px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Optimal Interval</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['GreenLine']};">
                    {_T_optimal:.1f}min</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">Young-Daly</div>
            </div>
        </div>
        """))

        # Prediction reveal
        _actual = 1.0  # 10000 / 10000
        _predicted = partB_pred.value
        _diff = abs(_predicted - _actual)
        _msg = (f"You predicted {_predicted:.1f} hours. Actual: {_actual:.1f} hour. "
                + ("Excellent." if _diff < 2 else
                   "Most students predict days or weeks. MTBF_cluster = MTBF_gpu / N. "
                   "Even 99.99% per-GPU uptime yields only 37% probability that all 10,000 GPUs "
                   "are simultaneously operational."))
        items.append(mo.callout(mo.md(f"**{_msg}**"), kind="success" if _diff < 2 else "warn"))

        items.append(mo.accordion({
            "Math Peek: Fleet Failure Rate": mo.md("""
```
MTBF_cluster = MTBF_gpu / N = 10,000h / 10,000 = 1 hour

P(all GPUs up for 1 hour) = (1 - 1/MTBF)^N
                           = (1 - 0.0001)^10,000
                           = 0.37  (37%)

Young-Daly Optimal Checkpoint Interval:
T_opt = sqrt(2 * T_checkpoint * MTBF_cluster)
```

Failure is not an exception at scale -- it is the baseline operating condition.
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER -- The Principle Interaction Map
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
        <div id="part-c" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['GreenLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">C</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part C &middot; 15 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px;">The Principle Interaction Map</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                You have seen how individual principles behave. Now: how do they interact?
                Push communication efficiency to 99% (aggressive gradient compression) and
                watch fault tolerance degrade. Maximize sustainability and watch available
                compute shrink. No configuration achieves maximum on all six axes.
            </div>
        </div>
        """))

        items.append(mo.md("### Your Prediction"))
        items.append(partC_pred)

        if partC_pred.value is None:
            items.append(mo.callout(mo.md("**Select your prediction to unlock.**"), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.md("### Adjust Each Principle Axis"))
        items.append(mo.hstack([partC_compute, partC_comm, partC_fault], justify="start", gap=1))
        items.append(mo.hstack([partC_sched, partC_sustain, partC_fair], justify="start", gap=1))

        _vals = {
            "Compute": partC_compute.value,
            "Communication": partC_comm.value,
            "Fault Tolerance": partC_fault.value,
            "Scheduling": partC_sched.value,
            "Sustainability": partC_sustain.value,
            "Fairness": partC_fair.value,
        }

        # Coupling interactions: pushing one axis degrades others
        _effective = dict(_vals)

        # Communication vs Fault Tolerance: aggressive compression reduces error margins
        if _vals["Communication"] > 85:
            _penalty = (_vals["Communication"] - 85) * 1.5
            _effective["Fault Tolerance"] = max(20, _vals["Fault Tolerance"] - _penalty)

        # Sustainability vs Compute: carbon caps limit GPU-hours
        if _vals["Sustainability"] > 80:
            _penalty = (_vals["Sustainability"] - 80) * 0.8
            _effective["Compute"] = max(20, _vals["Compute"] - _penalty)

        # Fairness vs Scheduling: monitoring overhead consumes latency budget
        if _vals["Fairness"] > 80:
            _penalty = (_vals["Fairness"] - 80) * 0.5
            _effective["Scheduling"] = max(20, _vals["Scheduling"] - _penalty)

        # Compute vs Communication: more compute per GPU means larger gradients
        if _vals["Compute"] > 85:
            _penalty = (_vals["Compute"] - 85) * 0.6
            _effective["Communication"] = max(20, _vals["Communication"] - _penalty)

        # Effective system gain: multiplicative product of all efficiencies
        _eff_product = 1.0
        for v in _effective.values():
            _eff_product *= (v / 100)
        _effective_gain = _eff_product * 1000  # scale factor vs single GPU

        # Radar chart
        _categories = list(_effective.keys())
        _eff_values = [_effective[c] for c in _categories]
        _raw_values = [_vals[c] for c in _categories]

        fig_radar = go.Figure()

        # Target zone (minimum acceptable)
        fig_radar.add_trace(go.Scatterpolar(
            r=[50] * 7, theta=_categories + [_categories[0]],
            fill="toself", fillcolor="rgba(203,32,45,0.08)",
            line=dict(color=COLORS["RedLine"], width=1, dash="dash"),
            name="Minimum Acceptable",
        ))

        # Raw settings (before coupling)
        fig_radar.add_trace(go.Scatterpolar(
            r=_raw_values + [_raw_values[0]], theta=_categories + [_categories[0]],
            line=dict(color=COLORS["TextMuted"], width=1, dash="dot"),
            name="Your Settings (raw)",
        ))

        # Effective (after coupling)
        fig_radar.add_trace(go.Scatterpolar(
            r=_eff_values + [_eff_values[0]], theta=_categories + [_categories[0]],
            fill="toself", fillcolor="rgba(0,99,149,0.12)",
            line=dict(color=COLORS["BlueLine"], width=2.5),
            name="Effective (after coupling)",
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=420, showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        apply_plotly_theme(fig_radar)

        # Check for any axis in red zone
        _red_axes = [c for c, v in _effective.items() if v < 50]
        _all_green = len(_red_axes) == 0
        _gain_color = COLORS["GreenLine"] if _effective_gain >= 50 else COLORS["OrangeLine"] if _effective_gain >= 20 else COLORS["RedLine"]

        items.append(mo.md("### Principle Interaction Map"))
        items.append(mo.as_html(fig_radar))
        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; justify-content: center; margin-top: 12px; flex-wrap: wrap;">
            <div style="padding: 16px 24px; border: 2px solid {_gain_color}; border-radius: 12px;
                        text-align: center; min-width: 200px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Effective System Gain</div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {_gain_color};">
                    {_effective_gain:.0f}x</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">target: &ge;50x</div>
            </div>
            <div style="padding: 16px 24px; border: 2px solid {'#008F45' if _all_green else COLORS['RedLine']};
                        border-radius: 12px; text-align: center; min-width: 200px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Constraint Status</div>
                <div style="font-size: 1.2rem; font-weight: 900; color: {'#008F45' if _all_green else COLORS['RedLine']};">
                    {'ALL AXES GREEN' if _all_green else f'{len(_red_axes)} AXIS IN RED'}</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">
                    {', '.join(_red_axes) if _red_axes else 'No violations'}</div>
            </div>
        </div>
        """))
        items.append(mo.callout(mo.md(
            "**Coupling effects are visible:** The dotted line shows your raw settings. The solid "
            "line shows the effective values after principle interactions. Notice how pushing one "
            "axis to maximum pulls others down. Maximum on any single axis never produces maximum "
            "total gain."
        ), kind="info"))

        # Prediction reveal
        _correct = partC_pred.value == "significant"
        _msg = ("Correct. Aggressive gradient compression (99% communication efficiency) reduces "
                "error detection capability and increases sensitivity to bit errors, significantly "
                "degrading fault tolerance."
                if _correct else
                "The answer is (C): significant degradation. Students treat principles as independent "
                "knobs. In reality, compressed gradients have less redundancy for error detection, "
                "and lossy compression amplifies the impact of any bit flip.")
        items.append(mo.callout(mo.md(f"**{_msg}**"), kind="success" if _correct else "warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER -- The Fleet Architecture Blueprint
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        _metric_display = {"dp": "Demographic Parity", "eo": "Equalized Odds", "eqop": "Equal Opportunity"}.get(FAIRNESS_METRIC, "Equal Opportunity")

        items.append(mo.Html(f"""
        <div id="part-d" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['RedLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">D</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part D &middot; 15 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px;">The Fleet Architecture Blueprint</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                The terminal synthesis of the entire two-volume curriculum. Design a fleet that
                achieves &ge;50&times; effective gain while keeping all six axes green. Your carbon
                cap from Lab 14 ({CARBON_CAP:.0%} of baseline) and fairness overhead from Lab 15
                ({_metric_display}, {FAIRNESS_OVERHEAD_MS}ms) feed in as starting constraints.
            </div>
        </div>
        """))

        # Fleet controls
        items.append(mo.md("### Configure Your Fleet"))
        items.append(mo.hstack([partD_gpus, partD_comm_strategy], justify="start", gap=1))
        items.append(mo.hstack([partD_ckpt_freq, partD_scheduling], justify="start", gap=1))

        _N = partD_gpus.value
        _comm = partD_comm_strategy.value
        _ckpt = partD_ckpt_freq.value
        _sched = partD_scheduling.value

        # ── Compute axis ────────────────────────────────────────────────────
        _compute_base = 80
        _carbon_penalty = max(0, (1 - CARBON_CAP) * 30)
        _compute = max(30, _compute_base - _carbon_penalty)

        # ── Communication axis ──────────────────────────────────────────────
        _comm_eff = {"ring": 75, "tree": 80, "compress": 92}[_comm]
        _scale_penalty = min(15, _N / 1000 * 5)
        _communication = max(30, _comm_eff - _scale_penalty)

        # ── Fault Tolerance axis ────────────────────────────────────────────
        _mtbf_cluster = MTBF_GPU_HOURS / _N
        _T_ckpt_min = CHECKPOINT_COST_S / 60
        _T_optimal = math.sqrt(2 * _T_ckpt_min * _mtbf_cluster * 60) if _mtbf_cluster > 0 else 10
        _ckpt_overhead = _T_ckpt_min / _ckpt if _ckpt > 0 else 1
        _failure_waste = _ckpt / (2 * _mtbf_cluster * 60) if _mtbf_cluster > 0 else 1
        _goodput = max(0, 1 - _ckpt_overhead - _failure_waste)
        _fault_tol = min(95, _goodput * 100)
        if _comm == "compress":
            _fault_tol = max(30, _fault_tol - 12)

        # ── Scheduling axis ────────────────────────────────────────────────
        _sched_base = 65 if _sched == "static" else 82
        _fairness_latency_penalty = FAIRNESS_OVERHEAD_MS / 100 * 10
        _scheduling = max(30, _sched_base - _fairness_latency_penalty)

        # ── Sustainability axis ─────────────────────────────────────────────
        _sustainability = min(95, CARBON_CAP * 100 + 10)

        # ── Fairness axis ───────────────────────────────────────────────────
        _fairness = min(90, 60 + (1 - FAIRNESS_THRESHOLD) * 40)

        # ── Effective gain ──────────────────────────────────────────────────
        _axes = {
            "Compute": _compute,
            "Communication": _communication,
            "Fault Tolerance": _fault_tol,
            "Scheduling": _scheduling,
            "Sustainability": _sustainability,
            "Fairness": _fairness,
        }
        _eff_product = 1.0
        for _v in _axes.values():
            _eff_product *= (_v / 100)
        _effective_gain = _eff_product * _N

        _red_axes = [c for c, _v in _axes.items() if _v < 50]
        _all_green = len(_red_axes) == 0
        _target_met = _effective_gain >= 50 and _all_green

        # Radar chart
        _categories = list(_axes.keys())
        _values = [_axes[c] for c in _categories]

        fig_fleet = go.Figure()
        fig_fleet.add_trace(go.Scatterpolar(
            r=[50] * 7, theta=_categories + [_categories[0]],
            fill="toself", fillcolor="rgba(203,32,45,0.08)",
            line=dict(color=COLORS["RedLine"], width=1, dash="dash"),
            name="Minimum (50%)",
        ))
        fig_fleet.add_trace(go.Scatterpolar(
            r=_values + [_values[0]], theta=_categories + [_categories[0]],
            fill="toself",
            fillcolor="rgba(0,143,69,0.12)" if _all_green else "rgba(203,32,45,0.12)",
            line=dict(color=COLORS["GreenLine"] if _all_green else COLORS["RedLine"], width=2.5),
            name="Your Fleet",
        ))

        fig_fleet.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=450, showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        apply_plotly_theme(fig_fleet)

        _gain_color = COLORS["GreenLine"] if _target_met else COLORS["OrangeLine"] if _effective_gain >= 20 else COLORS["RedLine"]

        _banner = ""
        if _target_met:
            _banner = f"""<div style="background: linear-gradient(135deg, {COLORS['GreenLL']}, #d1fae5);
                         border: 2px solid {COLORS['GreenLine']}; border-radius: 12px; padding: 20px;
                         text-align: center; margin-bottom: 16px;">
                         <div style="font-size: 1.4rem; font-weight: 900; color: {COLORS['GreenLine']};">
                         FLEET DEPLOYED SUCCESSFULLY</div>
                         <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; margin-top: 4px;">
                         {_effective_gain:.0f}x effective gain &middot; All axes green &middot;
                         Constraints satisfied</div></div>"""
        elif _red_axes:
            _banner = f"""<div style="background: {COLORS['RedLL']}; border: 2px solid {COLORS['RedLine']};
                         border-radius: 12px; padding: 16px; text-align: center; margin-bottom: 16px;">
                         <div style="font-weight: 700; color: {COLORS['RedLine']}; font-size: 1.1rem;">
                         CONSTRAINT VIOLATION: {', '.join(_red_axes)} below threshold</div></div>"""

        if _banner:
            items.append(mo.Html(_banner))

        items.append(mo.md("### Fleet Architecture Blueprint"))
        items.append(mo.as_html(fig_fleet))
        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; justify-content: center; margin-top: 16px; flex-wrap: wrap;">
            <div style="padding: 16px 24px; border: 2px solid {_gain_color}; border-radius: 12px;
                        text-align: center; min-width: 200px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Effective System Gain</div>
                <div style="font-size: 2.4rem; font-weight: 900; color: {_gain_color};">
                    {_effective_gain:.0f}x</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">target: &ge;50x</div>
            </div>
            <div style="padding: 16px 24px; border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Fleet Size</div>
                <div style="font-size: 2rem; font-weight: 900; color: {COLORS['BlueLine']};">
                    {_N:,}</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">GPUs</div>
            </div>
            <div style="padding: 16px 24px; border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Goodput</div>
                <div style="font-size: 2rem; font-weight: 900; color: {COLORS['OrangeLine']};">
                    {_goodput:.0%}</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">useful work</div>
            </div>
        </div>
        """))

        # Final reflection
        items.append(mo.md("### Final Reflection"))
        items.append(partD_refl)

        if partD_refl.value == "depends":
            items.append(mo.callout(mo.md(
                "**The mature answer.** There is no universally correct trade-off. In healthcare, "
                "fairness is likely non-negotiable. In content recommendation, communication efficiency "
                "might take priority. The binding constraint depends on the stakeholders, the deployment "
                "context, and the consequences of violation. This is engineering judgment, not optimization."
            ), kind="success"))
        elif partD_refl.value is not None:
            items.append(mo.callout(mo.md(
                "**Consider the context.** Both absolute positions have merit, but the strongest "
                "engineering answer is 'it depends' -- because the right trade-off changes with "
                "the deployment context, the stakeholders, and the consequences of each violation."
            ), kind="info"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        items = []

        items.append(mo.md("---"))

        # ── GRADUATION BANNER ───────────────────────────────────────────────
        items.append(mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 50%, #0f3460 100%);
                    border-radius: 16px; padding: 32px 40px; margin: 24px 0;
                    border: 1px solid rgba(99,102,241,0.3);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="text-align: center;">
                <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.2em;
                            color: #94a3b8; text-transform: uppercase; margin-bottom: 12px;">
                    Curriculum Complete
                </div>
                <div style="font-size: 1.8rem; font-weight: 900; color: #f8fafc;
                            line-height: 1.2; margin-bottom: 16px;">
                    The Physics of AI Engineering
                </div>
                <div style="font-size: 1.0rem; color: #94a3b8; max-width: 600px;
                            margin: 0 auto; line-height: 1.7;">
                    You have completed 16 labs across two volumes. You have learned that
                    constraints drive architecture, that scale creates qualitative change,
                    and that the binding constraint shifts with every decision you make.
                    <br/><br/>
                    <strong style="color: #a5b4fc;">The invariants do not change.
                    The bottleneck moves.</strong>
                </div>
            </div>
        </div>
        """))

        # ── KEY TAKEAWAYS ───────────────────────────────────────────────────
        items.append(mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. The binding constraint shifts with scale.</strong>
                    At 8 GPUs, compute dominates. At 1,000+, communication dominates.
                    At fleet scale, failure is the baseline operating condition (MTBF = 1 hour
                    for 10,000 GPUs). The same system behaves qualitatively differently at
                    different scales.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Principles interact as a coupled system.</strong>
                    Optimizing communication degrades fault tolerance. Sustainability constraints
                    limit compute. Fairness monitoring consumes the latency budget. No configuration
                    achieves maximum on all six axes. The effective system gain is the product of
                    all principle efficiencies &mdash; pushing one to maximum while ignoring others
                    reduces the total.
                </div>
                <div>
                    <strong>3. The art is not maximizing any single metric.</strong>
                    Distributed ML engineering is the art of balancing computation, communication,
                    and coordination under sustainability and fairness constraints. Every optimization
                    creates a new bottleneck. The skilled architect does not escape the invariants;
                    they navigate them. That is the physics of AI engineering.
                </div>
            </div>
        </div>
        """))

        # ── CONNECTIONS ─────────────────────────────────────────────────────
        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: #6366f1;
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    The Journey
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Volume I</strong> taught you to understand ML systems on a single machine.
                    <strong>Volume II</strong> taught you to build ML systems at scale.
                    This capstone showed that the principles you learned individually form
                    a coupled system where every decision has consequences across all dimensions.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What Endures
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    Frameworks will change. Hardware will evolve. But the physics does not change:
                    constraints drive architecture, communication bottlenecks intensify with scale,
                    failure is certain at fleet size, and efficiency without caps increases
                    consumption. These invariants will serve you for the next decade.
                </div>
            </div>
        </div>
        """))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # TAB COMPOSITION
    # ─────────────────────────────────────────────────────────────────────

    # ── Compute fleet metrics for LEDGER_HUD (same logic as build_part_d) ──
    _N = partD_gpus.value
    _comm = partD_comm_strategy.value
    _ckpt = partD_ckpt_freq.value
    _sched = partD_scheduling.value

    _compute_base = 80
    _carbon_penalty = max(0, (1 - CARBON_CAP) * 30)
    _compute = max(30, _compute_base - _carbon_penalty)

    _comm_eff = {"ring": 75, "tree": 80, "compress": 92}[_comm]
    _scale_penalty = min(15, _N / 1000 * 5)
    _communication = max(30, _comm_eff - _scale_penalty)

    _mtbf_cluster = MTBF_GPU_HOURS / _N
    _T_ckpt_min = CHECKPOINT_COST_S / 60
    _ckpt_overhead = _T_ckpt_min / _ckpt if _ckpt > 0 else 1
    _failure_waste = _ckpt / (2 * _mtbf_cluster * 60) if _mtbf_cluster > 0 else 1
    _goodput = max(0, 1 - _ckpt_overhead - _failure_waste)
    _fault_tol = min(95, _goodput * 100)
    if _comm == "compress":
        _fault_tol = max(30, _fault_tol - 12)

    _sched_base = 65 if _sched == "static" else 82
    _fairness_latency_penalty = FAIRNESS_OVERHEAD_MS / 100 * 10
    _scheduling = max(30, _sched_base - _fairness_latency_penalty)

    _sustainability = min(95, CARBON_CAP * 100 + 10)
    _fairness = min(90, 60 + (1 - FAIRNESS_THRESHOLD) * 40)

    fleet_axes = {
        "Compute": _compute,
        "Communication": _communication,
        "Fault Tolerance": _fault_tol,
        "Scheduling": _scheduling,
        "Sustainability": _sustainability,
        "Fairness": _fairness,
    }
    _eff_product = 1.0
    for _v in fleet_axes.values():
        _eff_product *= (_v / 100)
    fleet_effective_gain = _eff_product * _N
    _all_green = all(v >= 50 for v in fleet_axes.values())
    fleet_target_met = fleet_effective_gain >= 50 and _all_green

    tabs = mo.ui.tabs({
        "Part A -- The Sensitivity Wall": build_part_a(),
        "Part B -- The Failure Budget": build_part_b(),
        "Part C -- The Principle Interaction Map": build_part_c(),
        "Part D -- The Fleet Architecture Blueprint": build_part_d(),
        "Synthesis -- Graduation": build_synthesis(),
    })
    tabs
    return (tabs, fleet_target_met, fleet_effective_gain, fleet_axes,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: LEDGER HUD
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(mo, ledger, COLORS, fleet_target_met, fleet_effective_gain, fleet_axes):
    if fleet_axes:
        ledger.save(chapter=16, design={
            "chapter": "v2_16",
            "fleet_deployed": fleet_target_met,
            "effective_gain": float(f"{fleet_effective_gain:.1f}"),
            "binding_constraint": min(fleet_axes, key=fleet_axes.get),
            "principle_scores": {k: float(f"{v:.1f}") for k, v in fleet_axes.items()},
        })

    mo.Html(f"""
    <div class="lab-hud" style="border: 1px solid rgba(99,102,241,0.3);">
        <span class="hud-label">LAB</span>
        <span class="hud-value">V2-16: The Fleet Synthesis (Capstone)</span>
        <span class="hud-label">LEDGER</span>
        <span class="hud-active">Saved (ch16)</span>
        <span class="hud-label">STATUS</span>
        <span class="{'hud-active' if fleet_target_met else 'hud-none'}">
            {'FLEET DEPLOYED' if fleet_target_met else 'FLEET NOT YET VIABLE'}</span>
        <span class="hud-label">CURRICULUM</span>
        <span class="hud-active">COMPLETE</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
