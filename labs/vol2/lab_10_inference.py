import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-10: THE INFERENCE ECONOMY
#
# Chapter: Inference at Scale (@sec-inference-at-scale)
# Core Invariant: Recurring inference cost eclipses one-time setup cost. Live
#                 state/cache memory often binds concurrency before peak compute.
#                 Continuous batching or duty-cycle scheduling transforms a
#                 stop-and-go assembly line into a flowing pipeline. The design
#                 challenge requires jointly optimizing precision, scheduling,
#                 and serving units under the selected track's guardrail.
#
# Tabbed Structure (35-40 minutes):
#   Part A — The Serving Cost Inversion (12-15 min)
#             Recurring cost crosses one-time budget. Scheduling trades latency
#             for throughput along a hockey stick.
#
#   Part B — The KV Cache Wall + Fleet Design Challenge (20-25 min)
#             State/cache memory sets concurrency. Precision + continuous
#             scheduling + right serving-unit count reduces recurring cost.
#
# Track source-of-truth:
#   Hardware, model, and scenario defaults resolve through MLSysIM and
#   mlsysbook_labs track variants. Notebook code composes and renders them.
# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP ─────────────────────────────────────────────────────────────
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
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        batching_result,
        cost_crossover,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        inference_economy_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        serving_plan,
        source_trace,
        state_capacity,
        track_context,
        track_arc_context,
        track_selector,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, DecisionLog, LAB_CSS, apply_plotly_theme,
        batching_result, build_lab_report, cost_crossover, get_lab_metadata,
        get_lab_track_variant, get_track_profile, go, inference_economy_profile,
        ledger, math, mo, np, report_export_panel, resolve_mlsysim_ref,
        serving_plan, source_trace, state_capacity, track_context,
        track_arc_context, track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v2_10_metadata = get_lab_metadata("vol2/lab_10_inference.py")
    return (v2_10_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_10_track_picker = track_selector(default=_default_track)
    v2_10_track_picker
    return (v2_10_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    inference_economy_profile,
    resolve_mlsysim_ref,
    v2_10_track_picker,
):
    v2_10_track_id = v2_10_track_picker.value
    v2_10_profile = get_track_profile(v2_10_track_id)
    v2_10_variant = get_lab_track_variant("v2_10_inference_economy", v2_10_profile.track_id)
    v2_10_hardware = resolve_mlsysim_ref(v2_10_variant.hardware_ref)
    v2_10_model = resolve_mlsysim_ref(v2_10_variant.model_ref)
    v2_10_inference = inference_economy_profile(
        v2_10_profile,
        v2_10_variant,
        v2_10_hardware,
        v2_10_model,
    )
    return (
        v2_10_hardware,
        v2_10_inference,
        v2_10_model,
        v2_10_profile,
        v2_10_track_id,
        v2_10_variant,
    )

# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    COLORS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
        track_arc_context,
    v2_10_inference,
    v2_10_metadata,
    v2_10_profile,
    v2_10_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, {COLORS['Surface0']} 0%, {COLORS['Surface1']} 100%);
                    border-radius: 16px; padding: 32px 40px; margin-bottom: 8px;
                    border: 1px solid #2d3748;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                <div>
                    <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                                text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;">
                        Vol 2 &middot; Lab 10 &middot; Inference at Scale
                    </div>
                    <div style="font-size: 2.0rem; font-weight: 800; color: #f1f5f9; line-height: 1.15; margin-bottom: 10px;">
                        The Inference Economy
                    </div>
                    <div style="font-size: 0.95rem; color: #94a3b8; max-width: 640px; line-height: 1.6;">
                        {v2_10_variant.workload_summary} Inference cost is the recurring
                        constraint: {v2_10_inference.cost_label}, state/cache memory,
                        batching, and replicas or local schedules determine whether the
                        selected track can meet {v2_10_variant.guardrail_metric}.
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;">
                    <span class="badge badge-info">{v2_10_profile.label}</span>
                    <span class="badge badge-info">{v2_10_inference.hardware_ref}</span>
                    <span class="badge badge-info">{v2_10_inference.model_name}</span>
                    <span class="badge badge-warn">45&ndash;55 minutes &middot; 4 Parts + Synthesis</span>
                </div>
            </div>
        </div>
        """),
        track_context(v2_10_profile),
        track_arc_context(v2_10_profile, v2_10_metadata.lab_id),
    ])
    return

# ─── CELL 2: BRIEFING ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, v2_10_inference, v2_10_variant):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Quantify the cost inversion:</strong> calculate when cumulative {v2_10_inference.cost_label} exceeds the one-time track budget.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose the state/cache wall:</strong> compute how many concurrent requests fit in {v2_10_inference.hardware_name} memory.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design an inference plan:</strong> jointly optimize precision, batching, and replicas or local scheduling under {v2_10_variant.guardrail_metric}.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    State/cache formula from the Inference at Scale chapter &middot;
                    Queuing theory (Kingman's formula) from the Fleet Orchestration chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
                    Part A: ~12 min &middot; Part B: ~25 min
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                "For {v2_10_inference.label}, when does recurring {v2_10_inference.cost_label}
                exceed the one-time budget, and which state/cache or batching constraint
                determines whether the system can serve locally or at fleet scale?"
            </div>
        </div>
    </div>
    """)
    return

# ─── CELL 3: RECOMMENDED READING ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete before this lab:

    - **The Inference at Scale chapter** -- Serving economics, state/cache scaling, continuous batching
    - The state/cache section -- `KV = 2 * L * H * S * B * P` for transformer KV cache
    - The Continuous Batching section -- Static vs iteration-level scheduling
    - The Queuing Theory section from the Fleet Orchestration chapter -- Kingman's formula
    """), kind="info")
    return

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 4: Part A widgets ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, v2_10_inference):
    # -- Part A prediction --
    partA_prediction = mo.ui.radio(
        options={
            "A) 6 months -- training dominates for a long time": "A",
            "B) 3 months -- serving catches up gradually": "B",
            "C) Weeks or days -- recurring inference cost grows fast": "C",
            "D) Never -- training is always more expensive": "D",
        },
        label=(
            f"{v2_10_inference.label}: at {v2_10_inference.demand_qps:g} events/s and "
            f"{v2_10_inference.cost_per_event:g} {v2_10_inference.cost_unit}/event, "
            f"when does cumulative {v2_10_inference.cost_label} exceed the one-time budget?"
        ),
    )
    return (partA_prediction,)

# ─── CELL 5: Part A controls + Part A reflection + Part B prediction ─────────
@app.cell(hide_code=True)
def _(mo, v2_10_inference):
    _qps = v2_10_inference.demand_qps
    _unit_cost = v2_10_inference.cost_per_event
    a1_qps = mo.ui.number(
        start=0.0,
        stop=max(_qps * 10, 1.0),
        value=_qps,
        step=max(_qps / 20, 0.01),
        label="Demand rate (events/s)",
    )
    a1_cost_query = mo.ui.number(
        start=0.0,
        stop=max(_unit_cost * 10, 0.001),
        value=_unit_cost,
        step=max(_unit_cost / 20, 0.000001),
        label=f"Cost per event ({v2_10_inference.cost_unit})",
    )
    a1_weeks = mo.ui.slider(start=1, stop=52, value=v2_10_inference.horizon_weeks, step=1, label="Deployment duration (weeks)")
    a1_optimization = mo.ui.slider(start=0, stop=50, value=0, step=5, label="Inference optimization (%)")

    # -- Part A reflection --
    partA_reflection = mo.ui.radio(
        options={
            "A) Reduce QPS by throttling users": "A",
            "B) Optimize inference efficiency (quantization, batching, caching)": "B",
            "C) Train a smaller model": "C",
            "D) Increase the price per query": "D",
        },
        label="What is the highest-ROI lever for controlling total cost of ownership?",
    )

    # -- Part B prediction --
    partB_prediction = mo.ui.radio(
        options={
            "A) Many requests -- memory is not the constraint": "A",
            "B) A modest batch -- state/cache overhead is visible": "B",
            "C) Only a few requests -- memory is tighter than expected": "C",
            "D) About one request -- live state/cache dominates memory": "D",
        },
        label=(
            f"{v2_10_inference.model_name} on {v2_10_inference.hardware_name}: "
            f"with {v2_10_inference.state_kind}, how many concurrent requests fit?"
        ),
    )
    return (a1_cost_query, a1_optimization, a1_qps, a1_weeks, partA_reflection, partB_prediction)

# ─── CELL 6: Part B controls + Part B reflection ────────────────────────────
@app.cell(hide_code=True)
def _(mo, v2_10_inference, v2_10_variant):
    _precision_default = float(v2_10_variant.defaults.get("precision_bytes", 2.0))
    _precision_label = {
        2.0: "FP16 (2 bytes)",
        1.0: "INT8 (1 byte)",
        0.5: "INT4 (0.5 bytes)",
    }.get(_precision_default, "FP16 (2 bytes)")
    a2_precision = mo.ui.dropdown(
        options={"FP16 (2 bytes)": 2, "INT8 (1 byte)": 1, "INT4 (0.5 bytes)": 0.5},
        value=_precision_label,
        label="Weight precision",
    )
    _context = v2_10_inference.context_tokens
    _context_step = max(128, min(2048, _context // 8))
    a2_context_len = mo.ui.slider(
        start=max(128, _context_step),
        stop=max(_context * 2, _context_step * 2),
        value=_context,
        step=_context_step,
        label="State/cache context window",
    )
    a2_n_gpus = mo.ui.slider(
        start=1,
        stop=max(8, v2_10_inference.default_devices_per_replica),
        value=v2_10_inference.default_devices_per_replica,
        step=1,
        label="Devices per serving unit",
    )

    # -- Part B reflection --
    partB_reflection = mo.ui.radio(
        options={
            "A) Reduce precision to free memory for live state/cache slots": "A",
            "B) Add peak compute without changing memory capacity": "B",
            "C) Replace the model with a much smaller one": "C",
            "D) Spill live state/cache to slower off-device memory": "D",
        },
        label=f"What is the most effective way to increase concurrent capacity for {v2_10_inference.state_kind}?",
    )
    return (a2_context_len, a2_n_gpus, a2_precision, partB_reflection)

# ─── CELL 6b: Part C prediction + controls ─────────────────────────────────
@app.cell(hide_code=True)
def _(mo, v2_10_inference):
    partC_prediction = mo.ui.radio(
        options={
            "A) 1.5x -- modest improvement over static batching": "A",
            "B) 2-4x -- continuous batching fills freed slots immediately": "B",
            "C) 10x -- batching is the dominant optimization": "C",
            "D) No improvement -- batching does not affect memory-bound workloads": "D",
        },
        label=f"You switch {v2_10_inference.label} from static scheduling to continuous scheduling. What throughput improvement do you expect for mixed request lengths?",
    )
    c1_avg_len = mo.ui.slider(start=256, stop=65536, value=4096, step=256, label="Average output length (tokens)")
    c1_max_len = mo.ui.slider(start=2048, stop=131072, value=32768, step=2048, label="Max context length (tokens)")
    c1_batch_size = mo.ui.slider(start=1, stop=32, value=8, step=1, label="Static batch size")
    partC_reflection = mo.ui.radio(
        options={
            "A) Static batching is fine -- just increase batch size": "A",
            "B) Continuous batching is strictly better because it eliminates padding waste and fills freed slots with new requests": "B",
            "C) Continuous batching only helps with short requests": "C",
            "D) The choice between static and continuous batching depends on model size": "D",
        },
        label="Why is continuous batching the standard for production LLM serving?",
    )
    return (c1_avg_len, c1_batch_size, c1_max_len, partC_prediction, partC_reflection)

# ─── CELL 6c: Part D prediction + controls ─────────────────────────────────
@app.cell(hide_code=True)
def _(mo, v2_10_inference, v2_10_variant):
    partD_prediction = mo.ui.radio(
        options={
            "A) 200 replicas of FP16 with static batching -- brute force": "A",
            "B) 50 replicas of INT4 with continuous batching -- optimized per-replica throughput": "B",
            "C) 100 replicas of FP16 with continuous batching -- balanced approach": "C",
            "D) 25 replicas of INT4 with static batching -- minimize replica count": "D",
        },
        label=(
            f"Design: meet {v2_10_inference.demand_qps:g} events/s under "
            f"{v2_10_inference.slo_ms:g} ms for {v2_10_inference.label}. "
            "Which configuration is the best economic shape?"
        ),
    )
    _target = v2_10_inference.demand_qps
    d1_target_qps = mo.ui.number(
        start=0.01,
        stop=max(_target * 10, 1.0),
        value=_target,
        step=max(_target / 20, 0.01),
        label="Target events/s",
    )
    _precision_default = float(v2_10_variant.defaults.get("precision_bytes", 2.0))
    _precision_label = {
        2.0: "FP16 (2 bytes)",
        1.0: "INT8 (1 byte)",
        0.5: "INT4 (0.5 bytes)",
    }.get(_precision_default, "FP16 (2 bytes)")
    d1_quant = mo.ui.dropdown(
        options={"FP16 (2 bytes)": 2.0, "INT8 (1 byte)": 1.0, "INT4 (0.5 bytes)": 0.5},
        value=_precision_label,
        label="Weight quantization",
    )
    d1_batching = mo.ui.dropdown(
        options={"Static": 1.0, "Continuous": 3.0},
        value="Continuous",
        label="Batching strategy",
    )
    d1_gpus_per_replica = mo.ui.slider(
        start=1,
        stop=max(8, v2_10_inference.default_devices_per_replica),
        value=v2_10_inference.default_devices_per_replica,
        step=1,
        label="Devices per serving unit",
    )
    partD_reflection = mo.ui.radio(
        options={
            "A) Minimize replica count to reduce management overhead": "A",
            "B) Minimize recurring cost subject to demand, latency, and guardrails": "B",
            "C) Maximize batch size for best utilization": "C",
            "D) Use the largest serving unit to maximize memory": "D",
        },
        label="What is the correct objective function for fleet design?",
    )
    return (d1_batching, d1_gpus_per_replica, d1_quant, d1_target_qps, partD_prediction, partD_reflection)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: SINGLE TABS CELL
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(
    COLORS, apply_plotly_theme, batching_result, cost_crossover,
    go, math, mo, np, serving_plan, state_capacity,
    a1_cost_query, a1_optimization, a1_qps,
    a1_weeks, a2_context_len, a2_n_gpus,
    a2_precision, c1_avg_len, c1_batch_size, c1_max_len,
    d1_batching, d1_gpus_per_replica, d1_quant, d1_target_qps,
    partA_prediction, partA_reflection, partB_prediction, partB_reflection,
    partC_prediction, partC_reflection, partD_prediction, partD_reflection,
    v2_10_inference, v2_10_model, v2_10_profile, v2_10_variant,
):

    # ═════════════════════════════════════════════════════════════════════════
    # PART A: THE SERVING COST INVERSION
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_a():
        items = []

        # ── Stakeholder message ────────────────────────────────────────────
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['BlueLine']}; background: {COLORS['BlueLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; VP of Engineering
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "{v2_10_variant.stakeholder}: the one-time budget is visible, but
                recurring {v2_10_inference.cost_label} keeps accumulating. When does
                the operating side dominate for {v2_10_inference.label}?"
            </div>
        </div>
        """))

        # ── Concept introduction ───────────────────────────────────────────
        _default_cost = cost_crossover(
            setup_cost=v2_10_inference.setup_cost,
            demand_qps=v2_10_inference.demand_qps,
            cost_per_event=v2_10_inference.cost_per_event,
        )
        items.append(mo.md(f"""
    Inference economy is a recurring-cost problem. For the **{v2_10_profile.label}** track,
    the cost unit is **{v2_10_inference.cost_unit}**, and the recurring metric is
    **{v2_10_inference.cost_label}**.

    At the default demand:

    ```
    Daily recurring cost = {v2_10_inference.demand_qps:g} events/s x 86,400 s/day
                         x {v2_10_inference.cost_per_event:g} {v2_10_inference.cost_unit}/event
                         = {_default_cost.daily_cost:,.2f} {v2_10_inference.cost_unit}/day
    Crossover = {v2_10_inference.setup_cost:,.2f} / {_default_cost.weekly_cost:,.2f}
              = {_default_cost.crossover_weeks:.1f} weeks
    ```

    Higher demand or higher per-event cost moves the crossover earlier.
        """))

        # ── Prediction lock ────────────────────────────────────────────────
        items.append(partA_prediction)
        if partA_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part A instruments."), kind="warn"))
            return mo.vstack(items)

        # ── Controls ───────────────────────────────────────────────────────
        items.append(mo.md("### Serving Cost Calculator"))
        items.append(mo.hstack([
            mo.vstack([a1_qps, a1_cost_query]),
            mo.vstack([a1_weeks, a1_optimization]),
        ], justify="center", gap=2))

        # ── Instruments ────────────────────────────────────────────────────
        _qps = a1_qps.value
        _cpq = a1_cost_query.value
        _weeks = a1_weeks.value
        _opt_pct = a1_optimization.value / 100

        _cost = cost_crossover(
            setup_cost=v2_10_inference.setup_cost,
            demand_qps=_qps,
            cost_per_event=_cpq,
            optimization_pct=a1_optimization.value,
        )
        _daily_cost = _cost.daily_cost
        _weekly_cost = _cost.weekly_cost
        _crossover_weeks = _cost.crossover_weeks
        _crossover_days = _cost.crossover_days
        _annual_savings = _cost.annual_savings

        # ── Cost curves ───────────────────────────────────────────────
        _week_range = np.arange(0, _weeks + 1)
        _training_line = [v2_10_inference.setup_cost] * len(_week_range)
        _serving_cumulative = [w * _weekly_cost for w in _week_range]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_week_range, y=_training_line, mode="lines",
            name=f"One-time budget ({v2_10_inference.cost_unit})", line=dict(color=COLORS["BlueLine"], width=2.5, dash="dash"),
            hovertemplate="Week %{x}: $%{y:,.0f}<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=_week_range, y=_serving_cumulative, mode="lines",
            name="Cumulative serving cost", line=dict(color=COLORS["RedLine"], width=2.5),
            fill="tonexty", fillcolor="rgba(203,32,45,0.1)",
            hovertemplate="Week %{x}: $%{y:,.0f}<extra></extra>",
        ))
        if _crossover_weeks <= _weeks:
            _fig.add_vline(x=_crossover_weeks, line=dict(color=COLORS["OrangeLine"], width=2, dash="dot"),
                           annotation_text=f"Crossover: week {_crossover_weeks:.1f}",
                           annotation_position="top left")

        _fig.update_layout(
            height=340,
            xaxis=dict(title="Weeks Since Deployment"),
            yaxis=dict(title=f"Cumulative Cost ({v2_10_inference.cost_unit})"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=70, r=20),
        )
        apply_plotly_theme(_fig)

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Serving Cost Model
            </div>
            <div>Daily recurring cost = {_qps:g} events/s &times; 86,400 s/day &times; {_cpq:g} {v2_10_inference.cost_unit}/event &times; (1 - {_opt_pct:.0%}) = <strong>{_daily_cost:,.2f} {v2_10_inference.cost_unit}/day</strong></div>
            <div>Crossover at week <strong>{_crossover_weeks:.1f}</strong> ({_crossover_days:.0f} days)</div>
            <div>Annual recurring cost: <strong>{_daily_cost * 365:,.2f} {v2_10_inference.cost_unit}</strong></div>
            {'<div>Annual savings from ' + str(a1_optimization.value) + '% optimization: <strong style=color:' + COLORS["GreenLine"] + ';>' + f"{_annual_savings:,.2f} " + v2_10_inference.cost_unit + '</strong></div>' if _opt_pct > 0 else ''}
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Crossover</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['OrangeLine']}; font-family:monospace;">wk {_crossover_weeks:.0f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">serving &gt; training</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Daily Cost</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['RedLine']}; font-family:monospace;">{_daily_cost:.1f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">{v2_10_inference.cost_unit}/day</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Annual Savings</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['GreenLine']}; font-family:monospace;">{_annual_savings:.1f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">{a1_optimization.value}% optimization</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # ── Reveal ─────────────────────────────────────────────────────────
        if partA_prediction.value == "C":
            items.append(mo.callout(mo.md(
                f"**Correct.** The default crossover for this track is {_default_cost.crossover_weeks:.1f} weeks. "
                "The same recurring-cost math applies to the selected track: once a local "
                "or fleet inference loop runs continuously, small per-event costs compound quickly."
            ), kind="success"))
        elif partA_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Far too conservative.** Students anchor on how expensive training *felt* "
                "but underestimate the relentless compounding of per-query cost at scale. "
                f"For this track, the default crossover is {_default_cost.crossover_weeks:.1f} weeks."
            ), kind="warn"))
        elif partA_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**In the right direction but too slow.** 3 months is possible at very low "
                "demand, but recurring inference cost often crosses the one-time budget quickly."
            ), kind="warn"))
        elif partA_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Categorically wrong.** Training is a one-time cost; serving is a continuous "
                f"operating expense. At this track's default demand, recurring {v2_10_inference.cost_label} "
                "eventually dominates. This is why inference optimization is high leverage."
            ), kind="warn"))

        # ── MathPeek ───────────────────────────────────────────────────────
        items.append(mo.accordion({
            "Governing equations -- serving cost model": mo.md("""
        **Recurring Inference Cost**

        ```
        C_operation(t) = demand_rate * seconds_per_day * cost_per_event * t_days
        ```

        **Crossover Condition**

        ```
        C_setup = C_operation(t_crossover)
        t_crossover = C_setup / (demand_rate * 86400 * cost_per_event)
        ```

        **ROI of Inference Optimization**

        ```
        Annual_savings = demand_rate * 86400 * 365 * cost_per_event * optimization_fraction
        ```

        Use the selected track's cost unit: cloud uses dollars, device tracks use energy units.
            """)
        }))

        # ── Reflection ─────────────────────────────────────────────────────
        items.append(partA_reflection)
        if partA_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer."), kind="warn"))
        elif partA_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** Inference optimization directly reduces cost_per_query. "
                f"At the selected demand, a 10% improvement saves {_cost.annual_savings:.2f} "
                f"{v2_10_inference.cost_unit}/year. Precision, scheduling, and state/cache "
                "management are the primary levers."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Inference optimization is the highest-ROI lever** because it reduces "
                "cost_per_query without reducing service quality or user access. "
                "Quantization + continuous batching can achieve 2-4x cost reduction."
            ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B: THE KV CACHE WALL AND FLEET DESIGN
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_b():
        items = []

        # ── Stakeholder message ────────────────────────────────────────────
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['Cloud']}; background: {COLORS['BlueLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['Cloud']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; Inference Platform Lead
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "{v2_10_variant.stakeholder}: the model fits, but the live state/cache
                grows with requests. How many concurrent requests fit before
                {v2_10_inference.hardware_name} memory becomes the wall?"
            </div>
        </div>
        """))

        # ── Concept introduction ───────────────────────────────────────────
        _default_state = state_capacity(
            v2_10_inference,
            v2_10_model,
            context_tokens=v2_10_inference.context_tokens,
            precision_bytes=float(v2_10_variant.defaults.get("precision_bytes", 2.0)),
            devices_per_replica=v2_10_inference.default_devices_per_replica,
        )
        items.append(mo.md(f"""
    Stateful serving has a memory wall. For transformers this is the KV cache;
    for the device tracks it is the live activation, sensor-window, or runtime
    buffer state that must fit beside model weights.

    For **{v2_10_profile.label}**:

    - Hardware memory: **{_default_state.total_memory_gb:.3g} GB**
    - Model weights at default precision: **{_default_state.weight_gb:.3g} GB**
    - Per-request {v2_10_inference.state_kind}: **{_default_state.state_per_request_gb:.3g} GB**
    - Max concurrent requests: **{_default_state.max_concurrent}**
        """))

        # ── Prediction lock ────────────────────────────────────────────────
        items.append(partB_prediction)
        if partB_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part B instruments."), kind="warn"))
            return mo.vstack(items)

        # ── Controls ───────────────────────────────────────────────────────
        items.append(mo.md("### State/Cache Memory Wall Explorer"))
        items.append(mo.hstack([
            mo.vstack([a2_precision]),
            mo.vstack([a2_context_len, a2_n_gpus]),
        ], justify="center", gap=2))

        # ── Instruments ────────────────────────────────────────────────────
        _bytes_per_elem = a2_precision.value
        _seq_len = a2_context_len.value
        _devices = a2_n_gpus.value

        _state = state_capacity(
            v2_10_inference,
            v2_10_model,
            context_tokens=_seq_len,
            precision_bytes=_bytes_per_elem,
            devices_per_replica=_devices,
        )
        _total_hbm_gb = _state.total_memory_gb
        _weight_gb = _state.weight_gb
        _available_gb = _state.available_gb
        _kv_per_req_gb = _state.state_per_request_gb
        _max_concurrent = _state.max_concurrent
        _oom = _state.oom

        # ── Stacked memory chart ──────────────────────────────────────────────
        _n_requests = list(range(0, min(_max_concurrent + 3, 20)))
        _weight_vals = [_weight_gb] * len(_n_requests)
        _kv_vals = [n * _kv_per_req_gb for n in _n_requests]
        _total_vals = [w + k for w, k in zip(_weight_vals, _kv_vals)]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=_n_requests, y=_weight_vals, name="Model weights",
                               marker_color=COLORS["BlueLine"],
                               hovertemplate="Requests %{x}: %{y:.1f} GB<extra></extra>"))
        _kv_colors = [COLORS["GreenLine"] if t <= _total_hbm_gb else COLORS["RedLine"] for t in _total_vals]
        _fig.add_trace(go.Bar(x=_n_requests, y=_kv_vals, name="State/cache",
                               marker_color=_kv_colors,
                               hovertemplate="Requests %{x}: %{y:.1f} GB<extra></extra>"))
        _fig.add_hline(y=_total_hbm_gb, line=dict(color=COLORS["RedLine"], width=2, dash="dash"),
                       annotation_text=f"Total memory: {_total_hbm_gb:.3g} GB", annotation_position="top right")
        _fig.update_layout(
            height=300, barmode="stack",
            xaxis=dict(title="Concurrent Requests"),
            yaxis=dict(title="Memory (GB)", range=[0, max(max(_total_vals) * 1.1, _total_hbm_gb * 1.1)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=50, r=20),
        )
        apply_plotly_theme(_fig)

        # ── OOM banner ────────────────────────────────────────────────────────
        _oom_banner = ""
        if _oom:
            _oom_banner = f"""
            <div style="background:{COLORS['RedLL']}; border:2px solid {COLORS['RedLine']};
                        border-radius:10px; padding:14px 18px; margin:10px 0;">
                <div style="font-size:0.88rem; font-weight:800; color:{COLORS['RedLine']}; margin-bottom:4px;">
                    OOM &mdash; Cannot Serve Even 1 Request
                </div>
                <div style="font-size:0.85rem; color:#7f1d1d; line-height:1.6;">
                    State/cache per request ({_kv_per_req_gb:.3g} GB) exceeds available memory ({_available_gb:.3g} GB).<br>
                    Reduce context length, add devices, use quantization, or change the serving policy.
                </div>
            </div>
            """

        _conc_color = COLORS["RedLine"] if _max_concurrent <= 1 else (COLORS["OrangeLine"] if _max_concurrent <= 4 else COLORS["GreenLine"])

        items.append(mo.Html(f"""
        {_oom_banner}
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; State/Cache Memory Wall
            </div>
            <div>{v2_10_inference.state_kind} per request = <strong>{_kv_per_req_gb:.3g} GB</strong></div>
            <div>Weights = {v2_10_inference.model_params_b:.3g}B params &times; {_bytes_per_elem:g} bytes = <strong>{_weight_gb:.3g} GB</strong></div>
            <div>Available memory = {_total_hbm_gb:.3g} - {_weight_gb:.3g} = <strong>{_available_gb:.3g} GB</strong></div>
            <div>Max concurrent = floor({_available_gb:.3g} / {_kv_per_req_gb:.3g}) = <strong style="color:{_conc_color};">{_max_concurrent}</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Max Concurrent</div>
                <div style="font-size:2rem; font-weight:800; color:{_conc_color}; font-family:monospace;">{_max_concurrent}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">requests</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">State/Request</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_kv_per_req_gb:.2g}GB</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Weights</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['OrangeLine']}; font-family:monospace;">{_weight_gb:.2g}GB</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Available</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['GreenLine']}; font-family:monospace;">{_available_gb:.2g}GB</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # ── Reveal ─────────────────────────────────────────────────────────
        if partB_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Correct.** In the selected track, the active bottleneck is the state/cache wall: "
                f"{v2_10_inference.state_kind} leaves room for {_max_concurrent} concurrent requests."
            ), kind="success"))
        elif partB_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Too optimistic.** It is easy to focus on peak compute and forget live state/cache. "
                f"For this track, max concurrency is {_max_concurrent} before memory becomes the wall."
            ), kind="warn"))
        elif partB_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Sometimes right.** A modest batch fits when state/cache per request is small enough. "
                f"With the selected settings, the computed capacity is {_max_concurrent}."
            ), kind="warn"))
        elif partB_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Close for some configurations.** The important move is to compute the memory frontier, "
                f"not guess from hardware class. Here the frontier is {_max_concurrent} requests."
            ), kind="warn"))

        # ── MathPeek ───────────────────────────────────────────────────────
        items.append(mo.accordion({
            "Governing equations -- state/cache and inference-plan design": mo.md("""
        **State/Cache Formula**

        ```
        transformer_KV_bytes = 2 * num_layers * hidden_dim * seq_len * batch_size * bytes_per_elem
        ```

        - Factor 2: one K tensor + one V tensor per layer
        - KV cache is often stored at higher precision than quantized weights
        - Grows linearly with seq_len and batch_size
        - Device tracks may substitute measured activation, sensor-window, or runtime state per request

        **Memory Constraint**

        ```
        W + B * state_per_request <= total_memory
        Max_batch = floor((total_memory - W) / state_per_request)
        ```

        **Continuous Batching Throughput**

        ```
        Throughput_continuous = Throughput_static * (avg_len / max_len) * fill_factor
        ```

        - Static batching wastes (1 - avg/max) fraction of accelerator cycles
        - Continuous batching fills freed slots immediately: fill_factor = 2-4x
            """)
        }))

        # ── Reflection ─────────────────────────────────────────────────────
        items.append(partB_reflection)
        if partB_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer."), kind="warn"))
        elif partB_reflection.value == "A":
            items.append(mo.callout(mo.md(
                "**Correct.** Lower precision reduces model weight memory, which frees capacity "
                f"for {v2_10_inference.state_kind}. If memory is the active wall, those freed bytes "
                "translate into more live slots or more safety headroom."
            ), kind="success"))
        elif partB_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Does not address the binding constraint.** The bottleneck is memory, not "
                "compute. More peak compute does not help if state/cache fills available memory."
            ), kind="warn"))
        elif partB_reflection.value == "C":
            items.append(mo.callout(mo.md(
                "**Effective but changes the product.** A smaller model can fit better, but it also "
                "changes quality, safety, or product behavior. Treat that as a model-selection decision."
            ), kind="warn"))
        elif partB_reflection.value == "D":
            items.append(mo.callout(mo.md(
                "**Usually a last resort.** Spilling live state/cache to slower memory can preserve "
                "capacity but often violates latency, energy, or reliability guardrails."
            ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C: CONTINUOUS BATCHING
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_c():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['OrangeLine']}; background: {COLORS['OrangeLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['OrangeLine']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; ML Serving Engineer
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "{v2_10_variant.stakeholder}: static scheduling waits for the slowest
                or longest request. Most {v2_10_inference.label} work is shorter than
                the worst case. How much throughput do we recover if freed slots are
                filled immediately?"
            </div>
        </div>
        """))

        # Concept framing
        items.append(mo.md("""
    **Static batching** pads all requests to `max_len` and processes them together.
    When a short request finishes, the accelerator slot sits idle until all requests
    in the batch complete. Waste = `1 - avg_len / max_len`.

    **Continuous batching** (iteration-level scheduling) processes one token per iteration.
    When a request finishes, its slot is immediately filled by a new request.

    The throughput advantage:
    - Static: effective throughput = batch_size / max_len_time (includes padding waste)
    - Continuous: effective throughput = batch_size / avg_len_time * fill_factor

    The fill_factor (2-4x) comes from:
    1. No padding waste: compute only on real tokens
    2. Immediate slot filling: no idle accelerator cycles between requests
    3. Higher effective batch occupancy over time
        """))

        # Prediction lock
        items.append(partC_prediction)
        if partC_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part C instruments."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.md("### Continuous vs Static Batching Simulator"))
        items.append(mo.hstack([c1_avg_len, c1_max_len, c1_batch_size], justify="center", gap=2))

        _avg = c1_avg_len.value
        _max = c1_max_len.value
        _batch = c1_batch_size.value

        _batching = batching_result(
            avg_len=_avg,
            max_len=_max,
            batch_size=_batch,
            fill_factor=v2_10_inference.batching_fill_factor,
        )
        _padding_waste = _batching.padding_waste_pct / 100
        _static_throughput = _batching.static_throughput
        _continuous_throughput = _batching.continuous_throughput
        _speedup = _batching.speedup

        # Chart: throughput vs avg_len ratio
        _ratios = np.linspace(0.05, 1.0, 50)
        _static_tp = [_batch for _ in _ratios]
        _continuous_tp = [_batch * (1 / r) * v2_10_inference.batching_fill_factor for r in _ratios]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_ratios * 100, y=_static_tp, mode="lines",
            name="Static batching", line=dict(color=COLORS["RedLine"], width=2.5),
            hovertemplate="%{x:.0f}%%: %{y:.1f} req/cycle<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=_ratios * 100, y=_continuous_tp, mode="lines",
            name="Continuous batching", line=dict(color=COLORS["GreenLine"], width=2.5),
            hovertemplate="%{x:.0f}%%: %{y:.1f} req/cycle<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=[_avg / _max * 100], y=[_continuous_throughput],
            mode="markers", marker=dict(size=14, color=COLORS["OrangeLine"], symbol="diamond"),
            name="Current config",
            hovertemplate="%{x:.0f}%%: %{y:.1f} req/cycle<extra></extra>",
        ))
        _fig.update_layout(
            height=300,
            xaxis=dict(title="Avg Length / Max Length (%)"),
            yaxis=dict(title="Effective Throughput (requests/cycle)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=50, r=20),
        )
        apply_plotly_theme(_fig)

        _waste_color = COLORS["RedLine"] if _padding_waste > 0.5 else (COLORS["OrangeLine"] if _padding_waste > 0.2 else COLORS["GreenLine"])
        _speedup_color = COLORS["GreenLine"] if _speedup > 2 else (COLORS["OrangeLine"] if _speedup > 1.3 else COLORS["RedLine"])

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Batching Strategy Comparison
            </div>
            <div>Avg length: {_avg:,} tokens &mdash; Max length: {_max:,} tokens</div>
            <div>Padding waste (static): <strong style="color:{_waste_color};">{_padding_waste*100:.1f}%</strong></div>
            <div>Static throughput: <strong>{_static_throughput:.0f}</strong> req/cycle &mdash;
                 Continuous: <strong style="color:{COLORS['GreenLine']};">{_continuous_throughput:.1f}</strong> req/cycle</div>
            <div>Speedup: <strong style="color:{_speedup_color};">{_speedup:.1f}x</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Speedup</div>
                <div style="font-size:2rem; font-weight:800; color:{_speedup_color}; font-family:monospace;">{_speedup:.1f}x</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Padding Waste</div>
                <div style="font-size:2rem; font-weight:800; color:{_waste_color}; font-family:monospace;">{_padding_waste*100:.0f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">static only</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Continuous TP</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['GreenLine']}; font-family:monospace;">{_continuous_throughput:.0f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">req/cycle</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # Prediction reveal
        if partC_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** Continuous batching achieves 2-4x throughput improvement over static "
                "batching when avg_len << max_len. At avg=4K, max=32K, padding waste is 87.5%. "
                "Continuous batching eliminates this waste and immediately fills freed slots, "
                "achieving ~3x higher effective throughput in this scenario."
            ), kind="success"))
        elif partC_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Too conservative.** 1.5x would be the case if avg_len is close to max_len "
                "(e.g., 24K/32K). When avg_len is 4K vs max=32K, the 87.5% padding waste "
                "means continuous batching achieves 3-4x improvement."
            ), kind="warn"))
        elif partC_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Too optimistic.** 10x would require avg_len to be <3% of max_len AND perfect "
                "slot filling. In practice, continuous batching achieves 2-4x because slot "
                "filling is not instantaneous and prefill compute is non-trivial."
            ), kind="warn"))
        elif partC_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Wrong.** Memory-bound workloads benefit greatly from batching optimizations. "
                "Continuous batching increases the effective batch occupancy, which improves "
                "memory bandwidth utilization (more requests share the same weight reads)."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- continuous batching": mo.md("""
        **Static Batching Throughput**

        ```
        TP_static = batch_size / T_max_request
        ```

        All requests padded to max_len. Waste = 1 - avg_len/max_len.

        **Continuous Batching Throughput**

        ```
        TP_continuous = batch_size * (max_len / avg_len) * fill_factor
        ```

        - fill_factor = 0.7-0.9 (accounts for prefill overhead and scheduling gaps)
        - Speedup = (max_len / avg_len) * fill_factor
        - At avg=4K, max=32K: speedup = 8 * 0.85 = 6.8x theoretical, ~3x practical

        **Why Practical < Theoretical**

        - Prefill phase for new requests is compute-intensive
        - Not all slots fill instantly (scheduling latency)
        - State/cache management overhead
            """)
        }))

        # Reflection
        items.append(partC_reflection)
        if partC_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer."), kind="warn"))
        elif partC_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** Continuous batching is strictly better because: (1) no padding waste, "
                "(2) freed slots are filled immediately with new requests, (3) the accelerator processes "
                "real work instead of padding or idle time. This is why production serving systems use "
                "continuous or iteration-level scheduling."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Not the full picture.** Continuous batching is the standard because it "
                "eliminates padding waste AND fills slots immediately. It works for all model "
                "sizes and request length distributions."
            ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D: FLEET DESIGN CHALLENGE
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_d():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['RedLine']}; background: {COLORS['RedLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['RedLine']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; VP of AI Infrastructure
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "{v2_10_variant.stakeholder}: we need to meet
                {v2_10_inference.demand_qps:g} events/s under {v2_10_inference.slo_ms:g} ms.
                Which precision, scheduling, and serving-unit count minimizes recurring cost
                without violating {v2_10_variant.guardrail_metric}?"
            </div>
        </div>
        """))

        # Concept framing
        items.append(mo.md(f"""
    Inference-plan design jointly optimizes three levers:
    1. **Precision**: smaller weights free memory for more live state/cache slots.
    2. **Scheduling**: continuous batching or duty-cycle scheduling improves effective throughput.
    3. **Serving units**: replicas, devices per replica, or local schedules set recurring cost.

    For **{v2_10_profile.label}**, the objective is:

    ```
    minimize recurring {v2_10_inference.cost_label}
    subject to demand >= {v2_10_inference.demand_qps:g} events/s
               guardrail = {v2_10_variant.guardrail_metric}
    ```
        """))

        # Prediction lock
        items.append(partD_prediction)
        if partD_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part D instruments."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.md("### Fleet Design Optimizer"))
        items.append(mo.hstack([
            mo.vstack([d1_target_qps, d1_quant]),
            mo.vstack([d1_batching, d1_gpus_per_replica]),
        ], justify="center", gap=2))

        _target = d1_target_qps.value
        _bytes_per_elem = d1_quant.value
        _batch_mult = d1_batching.value
        _devices = d1_gpus_per_replica.value

        _plan = serving_plan(
            v2_10_inference,
            v2_10_model,
            target_qps=_target,
            precision_bytes=_bytes_per_elem,
            batching_multiplier=_batch_mult,
            devices_per_replica=_devices,
            context_tokens=v2_10_inference.context_tokens,
        )
        _state_for_plan = state_capacity(
            v2_10_inference,
            v2_10_model,
            context_tokens=v2_10_inference.context_tokens,
            precision_bytes=_bytes_per_elem,
            devices_per_replica=_devices,
        )
        _total_hbm = _state_for_plan.total_memory_gb
        _weight_gb = _state_for_plan.weight_gb
        _available_gb = _state_for_plan.available_gb
        _kv_per_req_gb = _state_for_plan.state_per_request_gb
        _max_batch = _plan.max_batch
        _per_replica_qps = _plan.per_replica_qps
        _replicas_needed = _plan.replicas_needed
        _total_gpus = _plan.total_devices
        _daily_cost = _plan.daily_cost
        _naive_daily = _plan.baseline_daily_cost
        _savings_pct = _plan.savings_pct
        _oom = _plan.oom

        # Chart: cost comparison across configurations
        _configs = ["Baseline\nStatic", "Continuous\nSame Precision", "Low Precision\nContinuous", "Your Config"]
        _same_precision = serving_plan(
            v2_10_inference,
            v2_10_model,
            target_qps=_target,
            precision_bytes=float(v2_10_variant.defaults.get("precision_bytes", 2.0)),
            batching_multiplier=_batch_mult,
            devices_per_replica=_devices,
            context_tokens=v2_10_inference.context_tokens,
        )
        _low_precision = serving_plan(
            v2_10_inference,
            v2_10_model,
            target_qps=_target,
            precision_bytes=0.5,
            batching_multiplier=max(1.0, _batch_mult),
            devices_per_replica=_devices,
            context_tokens=v2_10_inference.context_tokens,
        )
        _costs = [
            _naive_daily,
            _same_precision.daily_cost,
            _low_precision.daily_cost,
            _daily_cost,
        ]
        _bar_colors_d = [COLORS["RedLine"], COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["BlueLine"]]

        _fig = go.Figure()
        for _i, (_name, _cost) in enumerate(zip(_configs, _costs)):
            _fig.add_trace(go.Bar(
                x=[_name], y=[_cost / 1000],
                marker_color=_bar_colors_d[_i],
                text=[f"${_cost/1000:.0f}K"],
                textposition="auto",
                showlegend=False,
                hovertemplate=f"%{{x}}: %{{y:.1f}}K {v2_10_inference.cost_unit}/day<extra></extra>",
            ))
        _fig.update_layout(
            height=300,
            xaxis=dict(title="Configuration"),
            yaxis=dict(title=f"Daily Cost (K {v2_10_inference.cost_unit})"),
            margin=dict(t=30, b=70, l=70, r=20),
        )
        apply_plotly_theme(_fig)

        _cost_color = COLORS["GreenLine"] if _savings_pct > 50 else (COLORS["OrangeLine"] if _savings_pct > 20 else COLORS["RedLine"])

        if _oom:
            items.append(mo.Html(f"""
            <div style="background:{COLORS['RedLL']}; border:2px solid {COLORS['RedLine']};
                        border-radius:10px; padding:14px 18px; margin:10px 0;">
                <div style="font-size:0.88rem; font-weight:800; color:{COLORS['RedLine']}; margin-bottom:4px;">
                    OOM &mdash; Cannot Fit Any Request
                </div>
                <div style="font-size:0.85rem; color:#7f1d1d;">
                    Weights ({_weight_gb:.3g} GB) + one {_state_for_plan.state_kind}
                    ({_kv_per_req_gb:.3g} GB) exceed {_total_hbm:.3g} GB memory.
                    Increase devices, use stronger quantization, or change the serving policy.
                </div>
            </div>
            """))

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Fleet Design
            </div>
            <div>Weights: {_weight_gb:.3g} GB &mdash; Available memory: {_available_gb:.3g} GB &mdash; Max live slots: {_max_batch}</div>
            <div>Per-replica QPS: {_per_replica_qps:.0f} &mdash; Replicas needed: {_replicas_needed}</div>
            <div>Total devices: <strong>{_total_gpus}</strong> &mdash; Daily cost: <strong>{_daily_cost:,.2f} {v2_10_inference.cost_unit}</strong></div>
            <div>Baseline daily cost: {_naive_daily:,.2f} {v2_10_inference.cost_unit}/day</div>
            <div>Savings: <strong style="color:{_cost_color};">{_savings_pct:.0f}%</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Daily Cost</div>
                <div style="font-size:2rem; font-weight:800; color:{_cost_color}; font-family:monospace;">{_daily_cost/1000:.1f}K</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Serving Units</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_replicas_needed}</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Savings</div>
                <div style="font-size:2rem; font-weight:800; color:{_cost_color}; font-family:monospace;">{_savings_pct:.0f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">vs naive</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # Prediction reveal
        if partD_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** INT4 + continuous batching achieves ~3x throughput per replica "
                "in the cloud reference. In the selected track, the same principle holds: precision "
                "frees state/cache memory and scheduling fills live slots more efficiently."
            ), kind="success"))
        elif partD_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**The most expensive option.** Brute force adds serving units without attacking the active bottleneck. "
                "If memory and scheduling are binding, static scheduling wastes accelerator cycles on padding or idle slots."
            ), kind="warn"))
        elif partD_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Better than naive but still possibly suboptimal.** Higher precision leaves less room "
                "for live state/cache. If memory is active, lower precision can increase serving-slot capacity."
            ), kind="warn"))
        elif partD_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Static batching loses most of the INT4 benefit.** INT4 frees memory for "
                "larger batches, but static batching wastes those larger batches on padding. "
                "INT4 + continuous batching is the winning combination."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- inference fleet design": mo.md("""
        **Fleet Cost Objective**

        ```
        minimize: serving_units * devices_per_unit * cost_per_device_or_event
        subject to: replicas * QPS_per_replica >= target_QPS
                    P99_latency <= SLO
        ```

        **Per-Replica Throughput**

        ```
        QPS_per_replica = max_batch * base_qps * batching_multiplier
        max_batch = floor(available_memory / state_per_request)
        available_memory = devices * memory_per_device - weight_memory
        ```

        **Quantization Impact**

        INT4 vs FP16 weights: 4x memory reduction -> more live state/cache slots
        -> 4x larger batch -> ~4x higher QPS per replica (if memory-bound)

        Combined with 3x from continuous batching: ~12x total improvement.
            """)
        }))

        # Reflection
        items.append(partD_reflection)
        if partD_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer."), kind="warn"))
        elif partD_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** The objective is minimizing total cost subject to QPS and latency "
                "constraints. This is a constrained optimization: for each combination of "
                "(precision, scheduling, devices_per_unit), compute the minimum serving units needed "
                "to meet the QPS target, then pick the cheapest configuration that also meets "
                "the latency SLO."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Not the right objective.** The correct objective is minimizing total fleet cost "
                "(serving units x devices x recurring cost) subject to meeting both demand and the "
                "latency SLO. This requires jointly optimizing quantization, batching, and "
                "replica count."
            ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════════

    def build_synthesis():
        return mo.vstack([
            mo.Html(f"""
            <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                        border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                    Key Takeaways
                </div>
                <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                    <div style="margin-bottom: 10px;">
                        <strong>1. Recurring inference cost can exceed the one-time budget quickly.</strong>
                        For {v2_10_profile.label}, the unit is {v2_10_inference.cost_unit} and the
                        recurring metric is {v2_10_inference.cost_label}. Small per-event costs compound.
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>2. Live state/cache often binds concurrency before compute.</strong>
                        The selected track's {v2_10_inference.state_kind} must fit beside model
                        weights in {v2_10_inference.hardware_name} memory.
                    </div>
                    <div>
                        <strong>3. Precision + continuous scheduling transforms the economics.</strong>
                        Lower precision frees memory for live slots. Continuous scheduling fills freed
                        slots immediately. Combined, they reduce recurring cost when they attack the active bottleneck.
                    </div>
                </div>
            </div>
            """),
            mo.Html(f"""
            <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        What's Next
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        <strong>Lab V2-11: Edge Intelligence</strong> &mdash; The next deployment lab asks
                        what should remain local, what can be offloaded, and how privacy, battery,
                        and feedback loops change the serving architecture.
                    </div>
                </div>
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        Textbook &amp; TinyTorch
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        <strong>Read:</strong> the Inference at Scale chapter for the full state/cache derivation,
                        continuous batching mechanics, and fleet design principles.<br/>
                        <strong>Build:</strong> TinyTorch inference module &mdash; implement state/cache
                        management and continuous batching in <code>tinytorch/src/inference/</code>.
                    </div>
                </div>
            </div>
            """),
            mo.accordion({
                "Self-Assessment": mo.md("""
1. For the selected track, when does recurring inference cost exceed the one-time budget?
2. How many concurrent requests fit before state/cache memory becomes the wall?
3. Why does INT4 weight quantization increase serving throughput, even though it does not speed up compute?
4. What is the throughput advantage of continuous batching over static batching, and why?

*If you cannot answer all four from memory, revisit Parts A and B.*
""")
            }),
        ])

    # ═════════════════════════════════════════════════════════════════════════
    # COMPOSE TABS
    # ═════════════════════════════════════════════════════════════════════════

    tabs = mo.ui.tabs({
        "Part A -- The Serving Cost Inversion": build_part_a(),
        "Part B -- The KV Cache Wall": build_part_b(),
        "Part C -- Continuous Batching": build_part_c(),
        "Part D -- Fleet Design Challenge": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: LEDGER_HUD
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, DecisionLog):
    decision_input, decision_ui = DecisionLog()
    return (decision_input, decision_ui)

@app.cell(hide_code=True)
def _(COLORS, partA_prediction, partB_prediction, partC_prediction, partD_prediction,
      partA_reflection, partB_reflection, partC_reflection, partD_reflection,
      ledger, mo, decision_input, decision_ui, v2_10_inference, v2_10_profile, v2_10_variant):
    ledger.save(
        chapter=10,
        design={
            "lab": "inference_economy",
            "track_id": v2_10_profile.track_id,
            "scenario_id": v2_10_variant.scenario_id,
            "hardware_ref": v2_10_inference.hardware_ref,
            "model_ref": v2_10_inference.model_ref,
            "partA_prediction": partA_prediction.value or "no_selection",
            "partA_correct": partA_prediction.value == "C",
            "partA_reflection": partA_reflection.value or "no_selection",
            "partB_prediction": partB_prediction.value or "no_selection",
            "partB_correct": partB_prediction.value == "D",
            "partB_reflection": partB_reflection.value or "no_selection",
            "partC_prediction": partC_prediction.value or "no_selection",
            "partC_correct": partC_prediction.value == "B",
            "partC_reflection": partC_reflection.value or "no_selection",
            "partD_prediction": partD_prediction.value or "no_selection",
            "partD_correct": partD_prediction.value == "B",
            "partD_reflection": partD_reflection.value or "no_selection",
            "student_justification": str(decision_input.value),
        },
    )

    _a1_ok = partA_prediction.value == "C"
    _a2_ok = partB_prediction.value == "D"
    _tier = "Optimal" if (_a1_ok and _a2_ok) else ("Partial" if (_a1_ok or _a2_ok) else "Developing")
    _tier_color = COLORS["GreenLine"] if _tier == "Optimal" else (COLORS["OrangeLine"] if _tier == "Partial" else COLORS["TextMuted"])

    decision_ui
    mo.Html(f"""
    <div class="lab-hud">
        <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab 10</span></div>
        <div><span class="hud-label">CHAPTER</span> <span class="hud-value">v2_10 &middot; Inference at Scale</span></div>
        <div><span class="hud-label">TRACK</span> <span class="hud-value">{v2_10_profile.label}</span></div>
        <div><span class="hud-label">PART A</span> <span class="{'hud-active' if _a1_ok else 'hud-none'}">{"CORRECT" if _a1_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">PART B</span> <span class="{'hud-active' if _a2_ok else 'hud-none'}">{"CORRECT" if _a2_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">TIER</span> <span style="color:{_tier_color}; font-family:var(--font-mono);">{_tier.upper()}</span></div>
    </div>
    """)
    return


# ─── DOWNLOADABLE TRACK REPORT ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    a1_cost_query,
    a1_optimization,
    a1_qps,
    a2_context_len,
    a2_n_gpus,
    a2_precision,
    batching_result,
    build_lab_report,
    c1_avg_len,
    c1_batch_size,
    c1_max_len,
    cost_crossover,
    d1_batching,
    d1_gpus_per_replica,
    d1_quant,
    d1_target_qps,
    mo,
    partA_prediction,
    partA_reflection,
    partB_prediction,
    partB_reflection,
    partC_prediction,
    partC_reflection,
    partD_prediction,
    partD_reflection,
    report_export_panel,
    serving_plan,
    state_capacity,
    v2_10_inference,
    v2_10_metadata,
    v2_10_model,
    v2_10_profile,
    v2_10_variant,
):
    _cost = cost_crossover(
        setup_cost=v2_10_inference.setup_cost,
        demand_qps=a1_qps.value,
        cost_per_event=a1_cost_query.value,
        optimization_pct=a1_optimization.value,
    )
    _state = state_capacity(
        v2_10_inference,
        v2_10_model,
        context_tokens=a2_context_len.value,
        precision_bytes=a2_precision.value,
        devices_per_replica=a2_n_gpus.value,
    )
    _batching = batching_result(
        avg_len=c1_avg_len.value,
        max_len=c1_max_len.value,
        batch_size=c1_batch_size.value,
        fill_factor=v2_10_inference.batching_fill_factor,
    )
    _plan = serving_plan(
        v2_10_inference,
        v2_10_model,
        target_qps=d1_target_qps.value,
        precision_bytes=d1_quant.value,
        batching_multiplier=d1_batching.value,
        devices_per_replica=d1_gpus_per_replica.value,
        context_tokens=v2_10_inference.context_tokens,
    )

    _incomplete = []
    if partA_prediction.value is None:
        _incomplete.append("Part A cost inversion prediction")
    if partA_reflection.value is None:
        _incomplete.append("Part A cost-control reflection")
    if partB_prediction.value is None:
        _incomplete.append("Part B state/cache prediction")
    if partB_reflection.value is None:
        _incomplete.append("Part B capacity reflection")
    if partC_prediction.value is None:
        _incomplete.append("Part C batching prediction")
    if partC_reflection.value is None:
        _incomplete.append("Part C batching reflection")
    if partD_prediction.value is None:
        _incomplete.append("Part D serving-plan prediction")
    if partD_reflection.value is None:
        _incomplete.append("Part D objective reflection")

    _report = build_lab_report(
        v2_10_metadata,
        track=v2_10_profile.label,
        scenario=v2_10_variant.workload_summary,
        learning_objectives=(
            "Quantify when recurring inference cost exceeds the one-time setup or training budget.",
            "Compute the selected track's state/cache memory wall and explain the binding constraint.",
            "Choose a serving or local-inference plan using precision, batching, and serving-unit tradeoffs.",
        ),
        predictions={
            "cost_inversion": partA_prediction.value,
            "cost_control": partA_reflection.value,
            "state_cache_wall": partB_prediction.value,
            "capacity_lever": partB_reflection.value,
            "batching_speedup": partC_prediction.value,
            "batching_reason": partC_reflection.value,
            "serving_plan": partD_prediction.value,
            "fleet_objective": partD_reflection.value,
        },
        knob_settings={
            "demand_qps": a1_qps.value,
            "cost_per_event": a1_cost_query.value,
            "optimization_pct": a1_optimization.value,
            "context_tokens": a2_context_len.value,
            "devices_per_serving_unit": a2_n_gpus.value,
            "precision_bytes": a2_precision.value,
            "avg_length": c1_avg_len.value,
            "max_length": c1_max_len.value,
            "batch_size": c1_batch_size.value,
            "target_qps": d1_target_qps.value,
            "plan_precision_bytes": d1_quant.value,
            "plan_batching_multiplier": d1_batching.value,
            "plan_devices_per_unit": d1_gpus_per_replica.value,
        },
        evidence_summary={
            "hardware_ref": v2_10_inference.hardware_ref,
            "model_ref": v2_10_inference.model_ref,
            "cost_unit": v2_10_inference.cost_unit,
            "daily_cost": round(_cost.daily_cost, 6),
            "crossover_weeks": round(_cost.crossover_weeks, 3),
            "state_kind": _state.state_kind,
            "state_per_request_gb": round(_state.state_per_request_gb, 6),
            "max_concurrent": _state.max_concurrent,
            "batching_speedup": round(_batching.speedup, 3),
            "plan_replicas_or_units": _plan.replicas_needed,
            "plan_total_devices": _plan.total_devices,
            "plan_daily_cost": round(_plan.daily_cost, 6),
            "plan_savings_pct": round(_plan.savings_pct, 3),
        },
        final_decision=(
            f"Use {v2_10_variant.assumptions.get('serving_policy', v2_10_profile.label)} "
            f"only if it meets {v2_10_variant.guardrail_metric} at the computed recurring cost."
        ),
        big_takeaways=(
            "Inference economy is recurring cost, not only a training bill.",
            "Live state/cache memory determines concurrency before compute often does.",
            "Precision and continuous scheduling improve economics only when they attack the active bottleneck.",
        ),
        reflections={
            "cost_inversion": (
                f"The selected settings cross the one-time budget after {_cost.crossover_weeks:.1f} weeks."
            ),
            "state_wall": (
                f"{v2_10_inference.state_kind} allows {_state.max_concurrent} live requests per serving unit."
            ),
            "serving_plan": (
                f"The plan uses {_plan.total_devices} devices and costs {_plan.daily_cost:.2f} "
                f"{v2_10_inference.cost_unit}/day."
            ),
        },
        residual_risk=(
            "These are source-traced teaching estimates. Validate with profiler traces, workload distributions, "
            "thermal behavior, p99 latency, quality regression, and real pricing or battery measurements."
        ),
        source_trace={
            "track_id": v2_10_profile.track_id,
            "scenario_id": v2_10_variant.scenario_id,
            "hardware_ref": v2_10_variant.hardware_ref,
            "model_ref": v2_10_variant.model_ref,
            "shared_helper": "mlsysbook_labs.inference",
            "source_policy": v2_10_profile.source_policy,
        },
        result_snapshot={
            "inference_profile": v2_10_inference,
            "cost_crossover": _cost,
            "state_capacity": _state,
            "batching": _batching,
            "serving_plan": _plan,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V2-10 report is generated locally from the selected track, MLSysIM hardware/model refs, "
                "and shared `mlsysbook_labs.inference` calculations."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return

if __name__ == "__main__":
    app.run()
