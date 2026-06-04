import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-11: THE EDGE THERMODYNAMICS LAB
#
# Volume II, Chapter: Edge Intelligence (edge_intelligence.qmd)
#
# Four Parts (~55 minutes):
#   Part A — The Memory Amplification Tax (10 min)
#             On-device training requires 4-12x more memory than inference.
#             Prediction: how much memory does full fine-tuning require?
#
#   Part B — The Adaptation Strategy Selector (10 min)
#             LoRA reduces storage for multi-context personalization by 200x.
#             Prediction: total LoRA storage for 10 user contexts?
#
#   Part C — The Battery Drain Reality (10 min)
#             Accelerator choice changes energy budget use.
#             Prediction: energy budget used per local session.
#
#   Part D — The Federation Paradox (15 min)
#             Non-IID data causes 4-8x communication rounds explosion.
#             Merges original Parts D+E: federation + communication-compression.
#
# Hardware: selected canonical track hardware from MLSysIM
# Design Ledger: chapter="v2_11"
# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: SETUP + OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP ────────────────────────────────────────────────────────────
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
        adaptation_storage,
        build_lab_report,
        edge_device_profile,
        energy_drain,
        federated_communication,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        training_memory_breakdown,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()

    return (
        ACADEMIC_LAB_CSS,
        COLORS,
        DecisionLog,
        LAB_CSS,
        adaptation_storage,
        apply_plotly_theme,
        build_lab_report,
        edge_device_profile,
        energy_drain,
        federated_communication,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        math,
        mo,
        np,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        training_memory_breakdown,
    )


@app.cell
def _(get_lab_metadata):
    v2_11_metadata = get_lab_metadata("vol2/lab_11_edge_intelligence.py")
    return (v2_11_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v2_11_track_picker = track_selector(default=_default_track)
    v2_11_track_picker
    return (v2_11_track_picker,)


@app.cell
def _(
    edge_device_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v2_11_track_picker,
):
    v2_11_track_id = v2_11_track_picker.value
    v2_11_profile = get_track_profile(v2_11_track_id)
    v2_11_variant = get_lab_track_variant("v2_11_edge_thermodynamics", v2_11_track_id)
    v2_11_hardware = resolve_mlsysim_ref(v2_11_variant.hardware_ref)
    v2_11_device = edge_device_profile(v2_11_profile, v2_11_variant, v2_11_hardware)
    return (
        v2_11_device,
        v2_11_hardware,
        v2_11_profile,
        v2_11_track_id,
        v2_11_variant,
    )

# ─── CELL 1: HEADER ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
        track_arc_context,
    v2_11_device,
    v2_11_metadata,
    v2_11_profile,
    v2_11_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 11
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The Edge Thermodynamics Lab
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.1rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Memory &middot; Adaptation &middot; Battery &middot; Federation
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 700px; line-height: 1.65;">
                {v2_11_variant.workload_summary} The thermodynamic question is whether
                memory, energy, latency, privacy, and communication leave enough margin
                for the selected edge architecture.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    4 Parts &middot; ~55 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v2_11_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v2_11_device.hardware_ref}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px;">
                <span class="badge badge-fail">{v2_11_device.available_memory_mb:g} MB active memory budget</span>
                <span class="badge badge-warn">{v2_11_device.energy_budget_wh:g} Wh {v2_11_device.energy_budget_label}</span>
                <span class="badge badge-info">{v2_11_device.accelerator_label}: {v2_11_device.accelerator_energy_gain:g}x energy gain</span>
            </div>
        </div>
        """),
        track_context(v2_11_profile),
        track_arc_context(v2_11_profile, v2_11_metadata.lab_id),
    ])
    return

# ─── CELL 2: BRIEFING ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo, v2_11_device, v2_11_profile):
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the training memory amplification tax</strong> &mdash;
                    calculate that full fine-tuning requires 4-12x more memory than inference due to gradients,
                    optimizer state, and activations.</div>
                <div style="margin-bottom: 3px;">2. <strong>Compare adaptation strategies</strong> &mdash; discover that
                    LoRA reduces multi-context storage by 200x while preserving 95% of fine-tuning quality.</div>
                <div style="margin-bottom: 3px;">3. <strong>Predict federated communication cost</strong> &mdash; determine
                    that non-IID data causes 4-8x more communication rounds than IID, and that gradient
                    compression is the natural engineering response.</div>
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
                    Training memory breakdown from the Edge Intelligence chapter &middot;
                    LoRA rank decomposition &middot; FedAvg algorithm
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~55 min</strong><br/>
                    A: 10 &middot; B: 10 &middot; C: 10 &middot; D: 15 min
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
                &ldquo;On-device training is 'just inference with a backward pass.' Why does it
                require 4-12x more memory, consume the {v2_11_device.energy_budget_label}
                on CPU, and need 4-8x more communication rounds when data is non-IID
                for {v2_11_profile.label}?&rdquo;
            </div>
        </div>
    </div>
    """)
    return

# ─── CELL 3: RECOMMENDED READING ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** &mdash; Complete before this lab:

    - **Training Memory Amplification** &mdash; The 4-12x memory multiplier from activations,
      gradients, and optimizer state (the Edge Intelligence chapter).
    - **Adaptation Strategies** &mdash; LoRA, bias-only, and full fine-tuning trade-offs.
    - **On-Device Energy** &mdash; CPU vs accelerator power and latency for adaptation.
    - **Federated Learning** &mdash; FedAvg, non-IID data impact, gradient compression.
    """), kind="info")
    return

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 4: PART A WIDGETS ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, v2_11_device):
    pA_pred = mo.ui.radio(
        options={
            "A: Gradients add only a small amount": "A",
            "B: Training is roughly 2x inference": "B",
            "C: Training is often 5-9x inference": "C",
            "D: Training is always impossible at the edge": "D",
        },
        label=(
            f"A {v2_11_device.default_model_params_m:g}M-parameter model is being adapted "
            f"for {v2_11_device.label}. How much memory does full fine-tuning with Adam "
            "typically require relative to inference?"
        ),
    )
    return (pA_pred,)

# ─── CELL 5: PART A CONTROLS + PART B WIDGETS ────────────────────────────────
@app.cell(hide_code=True)
def _(mo, v2_11_device):
    pA_params = mo.ui.slider(
        start=0.01, stop=500, value=v2_11_device.default_model_params_m, step=0.01,
        label="Model parameters (millions)",
    )
    pA_batch = mo.ui.slider(
        start=1, stop=32, value=v2_11_device.default_batch_size, step=1,
        label="Batch size",
    )
    pA_strategy = mo.ui.dropdown(
        options={"Full Fine-Tuning": "full", "LoRA (rank-16)": "lora", "Bias-Only": "bias"},
        value="Full Fine-Tuning",
        label="Adaptation strategy",
    )

    pB_pred = mo.ui.radio(
        options={
            "A: ~200 MB -- half the full model cost": "A",
            "B: ~100 MB -- 4x savings": "B",
            "C: ~42 MB -- nearly 10x savings": "C",
            "D: ~4 MB -- adapters are negligible": "D",
        },
        label=(
            f"You need to store personalized models for {v2_11_device.default_contexts} "
            "contexts. Full fine-tuning stores a complete model per context. "
            "LoRA stores only adapter weights. What shape should the storage curve have?"
        ),
    )
    return (pA_batch, pA_params, pA_strategy, pB_pred)

# ─── CELL 6: PART B CONTROLS + PART C WIDGETS ────────────────────────────────
@app.cell(hide_code=True)
def _(mo, v2_11_device):
    pB_contexts = mo.ui.slider(
        start=1, stop=20, value=v2_11_device.default_contexts, step=1,
        label="Number of user contexts",
    )

    pC_pred = mo.ui.number(
        start=0.1, stop=50.0, value=None, step=0.1,
        label=(
            f"A local adaptation session runs against the {v2_11_device.energy_budget_label}. "
            "What percentage of that budget does one CPU session consume? "
            "(Account for throttling or scheduling overhead extending duration 2-3x.)"
        ),
    )
    return (pB_contexts, pC_pred)

# ─── CELL 7: PART C CONTROLS + PART D WIDGETS ────────────────────────────────
@app.cell(hide_code=True)
def _(mo, v2_11_device):
    pC_target = mo.ui.radio(
        options={"CPU": "cpu", "GPU": "gpu", v2_11_device.accelerator_label: "npu"},
        value="CPU",
        label="Execution target",
        inline=True,
    )

    pD_pred = mo.ui.radio(
        options={
            "A: 60-80 rounds -- modest increase": "A",
            "B: 100-150 rounds -- 2-3x more": "B",
            "C: 200-400 rounds -- 4-8x more": "C",
            "D: 1000+ rounds -- effectively never converges": "D",
        },
        label=(
            f"Non-IID edge data (beta=0.5). IID convergence takes {v2_11_device.iid_rounds} rounds. "
            "How many rounds does non-IID require to reach the same accuracy?"
        ),
    )
    return (pC_target, pD_pred)

# ─── CELL 8: PART D CONTROLS ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    pD_beta = mo.ui.slider(
        start=0.1, stop=2.0, value=0.5, step=0.1,
        label="Data heterogeneity (beta) -- lower = more non-IID",
    )
    pD_epochs = mo.ui.slider(
        start=1, stop=20, value=3, step=1,
        label="Local epochs (E)",
    )
    pD_compress = mo.ui.dropdown(
        options={
            "No compression": "none",
            "INT8 quantized (4x reduction)": "int8",
            "INT4 quantized (8x reduction)": "int4",
            "Top-K sparse (10x reduction)": "topk",
        },
        value="No compression",
        label="Gradient compression",
    )
    return (pD_beta, pD_compress, pD_epochs)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ALL PARTS AS TABS (SINGLE CELL)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 9: TABS CELL ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, apply_plotly_theme, go, math,
    mo, np, ledger, adaptation_storage, energy_drain,
    federated_communication, training_memory_breakdown,
    pA_batch, pA_params, pA_pred,
    pA_strategy, pB_contexts, pB_pred, pC_pred,
    pC_target, pD_beta, pD_compress, pD_epochs,
    pD_pred, v2_11_device, v2_11_profile, v2_11_variant,
):
    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER -- The Memory Amplification Tax
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
<div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; {v2_11_variant.stakeholder}</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;{v2_11_variant.objective} The active memory budget is
        {v2_11_device.available_memory_mb:g} MB. Can the training graph fit, or do
        optimizer states and activations make local adaptation impossible?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; {v2_11_variant.stakeholder} &middot; {v2_11_profile.label}</div>
</div>
"""))

        # Banner
        items.append(mo.Html(f"""
        <div id="part-a" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['BlueLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">A</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Part A &middot; 10 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                The Memory Amplification Tax
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                A model that comfortably runs inference on one deployment target may not learn
                on that same target. Full fine-tuning requires 4-12x more memory due to gradients,
                optimizer state, and activation caching.
            </div>
        </div>
        """))

        # Prediction
        items.append(mo.md("### Your Prediction"))
        items.append(pA_pred)

        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the memory breakdown."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.hstack([pA_params, pA_batch, pA_strategy], gap="1.5rem"))

        # Computation
        _params_m = pA_params.value
        _batch = pA_batch.value
        _strategy = pA_strategy.value
        _memory = training_memory_breakdown(
            params_m=_params_m,
            batch_size=_batch,
            strategy=_strategy,
            available_memory_mb=v2_11_device.available_memory_mb,
        )
        _weights_mb = _memory.weights_mb
        _grads_mb = _memory.gradients_mb
        _optim_mb = _memory.optimizer_mb
        _activ_mb = _memory.activations_mb
        _total_mb = _memory.total_mb
        _infer_mb = _memory.inference_mb
        _amplification = _memory.amplification
        _oom = not _memory.fits_memory

        # Stacked bar chart
        _fig = go.Figure()
        _segments = [
            ("Weights", _weights_mb, COLORS["BlueLine"]),
            ("Gradients", _grads_mb, COLORS["OrangeLine"]),
            ("Optimizer State", _optim_mb, "#7c3aed"),
            ("Activations", _activ_mb, COLORS["GreenLine"]),
        ]
        for _name, _val, _color in _segments:
            _fig.add_trace(go.Bar(
                name=_name, x=["Training Memory"], y=[_val],
                marker_color=_color, opacity=0.85,
                text=[f"{_val:.0f} MB"], textposition="inside",
            ))
        # RAM ceiling
        _fig.add_hline(y=v2_11_device.available_memory_mb, line_dash="dash", line_color=COLORS["RedLine"],
                       annotation_text=f"{v2_11_profile.label} active budget: {v2_11_device.available_memory_mb:g} MB",
                       annotation_position="top right")
        # Inference reference
        _fig.add_trace(go.Bar(
            name="Inference Only", x=["Inference"], y=[_infer_mb],
            marker_color=COLORS["BlueLine"], opacity=0.5,
            text=[f"{_infer_mb:.0f} MB"], textposition="inside",
        ))
        _fig.update_layout(
            barmode="stack", height=380,
            yaxis=dict(title="Memory (MB)"),
            margin=dict(l=50, r=20, t=30, b=40),
            legend=dict(orientation="h", y=-0.15, x=0),
        )
        apply_plotly_theme(_fig)

        items.append(mo.as_html(_fig))

        # OOM banner
        if _oom:
            items.append(mo.callout(mo.md(
                f"**OOM -- Training infeasible on this device.** "
                f"Required: {_total_mb:.2f} MB | Available: {v2_11_device.available_memory_mb:g} MB. "
                f"Switch to LoRA or reduce model size."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"Training fits within the {v2_11_profile.label} active memory budget "
                f"({_total_mb:.2f} MB used, {v2_11_device.available_memory_mb - _total_mb:.2f} MB headroom)."
            ), kind="success"))

        # Cards
        _amp_color = COLORS["RedLine"] if _amplification > 5 else COLORS["OrangeLine"] if _amplification > 2 else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Inference Memory</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_infer_mb:.0f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_amp_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Training Memory</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_amp_color};">{_total_mb:.0f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_amp_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Amplification</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_amp_color};">{_amplification:.1f}x</div>
            </div>
        </div>"""))

        items.append(mo.md(f"""
**Training Memory Breakdown** ({_params_m}M params, batch={_batch}, {_strategy})

```
Weights       = {_weights_mb:.2f} MB
Gradients     = {_grads_mb:.2f} MB
Optimizer     = {_optim_mb:.2f} MB
Activations   = {_activ_mb:.2f} MB
Total         = {_total_mb:.2f} MB ({_amplification:.1f}x inference)
```
*Source: `mlsysbook_labs.training_memory_breakdown`, using `{v2_11_device.hardware_ref}`.*
"""))

        # Reveal
        if pA_pred.value == "C":
            _msg = (
                "**Correct.** Full fine-tuning with Adam requires 5-9x more memory than inference. "
                "Gradients equal the model size, Adam adds 2x more for momentum and variance, and "
                "activations scale with batch size and depth. The model that *runs* on the device "
                "cannot *learn* on the device without LoRA or similar adaptation."
            )
            _kind = "success"
        elif pA_pred.value == "B":
            _msg = (
                "**You forgot optimizer state and activations.** Training is not 'inference + gradients.' "
                "Adam stores two additional copies of every trainable parameter (momentum and variance). "
                "Activations cached for the backward pass scale with batch size. Total: 5-9x, not 2x."
            )
            _kind = "warn"
        else:
            _msg = (
                "**Full fine-tuning with Adam requires ~200-360 MB for a 10M-param model.** "
                "Weights (40 MB) + Gradients (40 MB) + Optimizer State (80 MB) + Activations (~40-200 MB). "
                "That is 5-9x the inference footprint, depending on batch size."
            )
            _kind = "warn"

        items.append(mo.callout(mo.md(_msg), kind=_kind))

        items.append(mo.accordion({
            "Math Peek: Training Memory Amplification": mo.md("""
**Formula:**
$$
M_{\\text{train}} = M_{\\text{weights}} + M_{\\text{grad}} + M_{\\text{opt}} + M_{\\text{act}}
$$

**Where:**
- **$M_{\\text{weights}}$**: Model parameters in FP16 = $P \\times 2$ bytes
- **$M_{\\text{grad}}$**: Gradient storage (same size as weights) = $P \\times 2$ bytes
- **$M_{\\text{opt}}$**: Adam stores momentum + variance in FP32 = $P \\times 4 \\times 2$ bytes
- **$M_{\\text{act}}$**: Activations cached for backward pass, scales with batch size $B$

**Amplification factor** over inference ($M_{\\text{inf}} = P \\times 2$):

$$
\\text{Amplification} = \\frac{M_{\\text{train}}}{M_{\\text{inf}}} = 1 + 1 + 4 + \\frac{M_{\\text{act}}}{P \\times 2} \\approx 5\\text{-}9\\times
$$
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER -- The Adaptation Strategy Selector
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
<div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; {v2_11_variant.stakeholder}</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;I keep hearing that LoRA solves the on-device fine-tuning problem, but nobody tells me
        the actual memory numbers. For a 350M model, how much storage does LoRA rank-16 really save
        compared to full fine-tuning? And does bias-only tuning even move the needle on accuracy?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; {v2_11_variant.stakeholder} &middot; {v2_11_profile.label}</div>
</div>
"""))

        # Banner
        items.append(mo.Html(f"""
        <div id="part-b" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['GreenLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">B</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Part B &middot; 10 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                The Adaptation Strategy Selector
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                LoRA makes fine-tuning fit in memory. But the storage advantage is even more
                dramatic for multi-context personalization: 10 user profiles require 400 MB
                with full fine-tuning but only ~42 MB with LoRA adapters.
            </div>
        </div>
        """))

        # Prediction
        items.append(mo.md("### Your Prediction"))
        items.append(pB_pred)

        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the storage comparison."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(pB_contexts)

        _n_ctx = pB_contexts.value
        _model_mb = v2_11_device.default_model_params_m * 1e6 * 4 / (1024 * 1024)
        _storage = adaptation_storage(contexts=_n_ctx, model_mb=_model_mb)
        _full_total = _storage.full_total_mb
        _lora_total = _storage.lora_total_mb
        _bias_total = _storage.bias_total_mb

        _ctx_range = np.arange(1, 21)
        _storage_curve = [adaptation_storage(contexts=int(_ctx), model_mb=_model_mb) for _ctx in _ctx_range]
        _full_curve = [_result.full_total_mb for _result in _storage_curve]
        _lora_curve = [_result.lora_total_mb for _result in _storage_curve]
        _bias_curve = [_result.bias_total_mb for _result in _storage_curve]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_ctx_range, y=_full_curve, mode="lines+markers",
            name="Full Fine-Tuning", line=dict(color=COLORS["RedLine"], width=3),
        ))
        _fig.add_trace(go.Scatter(
            x=_ctx_range, y=_lora_curve, mode="lines+markers",
            name="LoRA (rank-16)", line=dict(color=COLORS["GreenLine"], width=3),
        ))
        _fig.add_trace(go.Scatter(
            x=_ctx_range, y=_bias_curve, mode="lines+markers",
            name="Bias-Only", line=dict(color=COLORS["BlueLine"], width=3),
        ))
        _fig.update_layout(
            height=340,
            xaxis=dict(title="Number of User Contexts"),
            yaxis=dict(title="Total Storage (MB)"),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=50, r=20, t=30, b=80),
        )
        apply_plotly_theme(_fig)

        _savings = _storage.lora_savings_ratio

        items.append(mo.as_html(_fig))
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Full Storage ({_n_ctx} ctx)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_full_total:.0f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">LoRA Storage ({_n_ctx} ctx)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_lora_total:.0f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Savings Ratio</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_savings:.0f}x</div>
            </div>
        </div>"""))

        # Reveal
        if pB_pred.value == "C":
            _msg = (
                f"**Correct shape.** LoRA adapters stay close to 1% of model size. "
                f"For {_n_ctx} contexts on {v2_11_profile.label}, full fine-tuning uses "
                f"{_full_total:.2f} MB while LoRA uses {_lora_total:.2f} MB."
            )
            _kind = "success"
        else:
            _msg = (
                f"**LoRA adapters are small because they avoid copying the full model per context.** "
                f"Here the base model is {_model_mb:.2f} MB. Full fine-tuning for {_n_ctx} "
                f"contexts reaches {_full_total:.2f} MB, while LoRA reaches {_lora_total:.2f} MB."
            )
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))

        items.append(mo.accordion({
            "Math Peek: LoRA Storage Savings": mo.md("""
**Formula:**
$$
M_{\\text{LoRA}} = M_{\\text{base}} + K \\times r \\times (d_{\\text{in}} + d_{\\text{out}}) \\times 2
$$

**Where:**
- **$M_{\\text{base}}$**: Base model weights (shared, stored once)
- **$K$**: Number of user contexts (adapters)
- **$r$**: LoRA rank (typically 4-16)
- **$d_{\\text{in}}, d_{\\text{out}}$**: Dimensions of the adapted weight matrices
- Factor of 2: bytes per FP16 parameter

**Savings ratio** vs. full fine-tuning ($K$ separate models):

$$
\\text{Ratio} = \\frac{K \\times M_{\\text{base}}}{M_{\\text{base}} + K \\times r/d \\times M_{\\text{base}}} \\approx \\frac{K}{1 + K \\times 0.01} \\approx 10\\times \\text{ at } K{=}10
$$
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER -- The Battery Drain Reality
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
<div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; {v2_11_variant.stakeholder}</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;Product wants local adaptation for {v2_11_profile.label}, but the available
        {v2_11_device.energy_budget_label} is only {v2_11_device.energy_budget_wh:g} Wh.
        How much does one local session consume, and when should we move work to
        {v2_11_device.accelerator_label} or off-device?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; {v2_11_variant.stakeholder} &middot; {v2_11_profile.label}</div>
</div>
"""))

        # Banner
        items.append(mo.Html(f"""
        <div id="part-c" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['OrangeLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">C</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Part C &middot; 10 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                The Battery Drain Reality
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                LoRA makes fine-tuning fit in memory. But does it make it practical? A fine-tuning
                session that visibly drains the energy budget is a product-killing feature,
                not a product feature. The execution target changes the equation entirely.
            </div>
        </div>
        """))

        # Prediction
        items.append(mo.md("### Your Prediction"))
        items.append(mo.md(f"*Enter budget-use percentage for one CPU session on {v2_11_profile.label}:*"))
        items.append(pC_pred)

        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Enter your prediction to unlock the energy budget simulator."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(pC_target)

        _target = pC_target.value
        _energy = energy_drain(v2_11_device, target=_target)
        _power = _energy.power_w
        _duration = _energy.duration_s
        _energy_wh = _energy.energy_wh
        _drain_pct = _energy.budget_used_pct
        _sessions_per_charge = _energy.sessions_per_budget

        # Comparison bars
        _target_keys = ["cpu", "gpu", "npu"]
        _results = [energy_drain(v2_11_device, target=_key) for _key in _target_keys]
        _targets = [_result.label for _result in _results]
        _drains = []
        for _t in ["cpu", "gpu", "npu"]:
            _drains.append(energy_drain(v2_11_device, target=_t).budget_used_pct)

        _fig = go.Figure()
        _bar_colors = [COLORS["RedLine"] if d > 5 else COLORS["OrangeLine"] if d > 1 else COLORS["GreenLine"]
                       for d in _drains]
        _fig.add_trace(go.Bar(
            x=_targets, y=_drains, marker_color=_bar_colors,
            text=[f"{d:.1f}%" for d in _drains], textposition="outside",
        ))
        _fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["GreenLine"],
                       annotation_text="Target: <1% per session")
        _fig.update_layout(
            height=340,
            yaxis=dict(title=f"{v2_11_device.energy_budget_label.title()} Used per Session (%)"),
            margin=dict(l=50, r=20, t=30, b=40),
        )
        apply_plotly_theme(_fig)

        items.append(mo.as_html(_fig))

        # Failure state
        _drain_color = COLORS["RedLine"] if _drain_pct > 5 else COLORS["OrangeLine"] if _drain_pct > 1 else COLORS["GreenLine"]
        if _drain_pct > 5:
            items.append(mo.callout(mo.md(
                f"**Product-killing energy budget use.** {_drain_pct:.1f}% per session means only "
                f"{_sessions_per_charge:.0f} sessions per {v2_11_device.energy_budget_label}. "
                "Users or operators will disable this feature."
            ), kind="danger"))
        elif _drain_pct > 1:
            items.append(mo.callout(mo.md(
                f"**Marginal.** {_drain_pct:.1f}% per session is noticeable. "
                f"{_sessions_per_charge:.0f} sessions per {v2_11_device.energy_budget_label}. "
                f"Consider {v2_11_device.accelerator_label} for production deployment."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Viable.** {_drain_pct:.2f}% per session. {_sessions_per_charge:.0f} sessions per "
                f"{v2_11_device.energy_budget_label}. This is a product feature, not a budget drain."
            ), kind="success"))

        # Cards
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_drain_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Budget Use ({_energy.label})</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_drain_color};">{_drain_pct:.2f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Duration</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_duration:.1f}s</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Sessions/Charge</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">
                    {_sessions_per_charge:.0f}</div>
            </div>
        </div>"""))

        # Prediction comparison
        _predicted = pC_pred.value if pC_pred.value else 0
        _cpu_drain = _drains[0]

        items.append(mo.md(f"""
**Energy Budget Formula**

```
Energy      = Power x Duration = {_power:.1f}W x {_duration:.1f}s = {_energy_wh:.4f} Wh
Budget (%)  = Energy / Budget x 100 = {_energy_wh:.4f} / {v2_11_device.energy_budget_wh:g} x 100 = {_drain_pct:.2f}%
Sessions    = 100% / {_drain_pct:.2f}% = {_sessions_per_charge:.0f}
```

You predicted: {_predicted:.1f}%. Actual CPU budget use: {_cpu_drain:.1f}%.

*Source: `mlsysbook_labs.energy_drain`, using `{v2_11_device.hardware_ref}`.*
"""))

        items.append(mo.accordion({
            "Math Peek: Energy per Inference and Battery Drain": mo.md("""
**Dynamic power consumption:**
$$
P = C \\cdot V^2 \\cdot f
$$

**Where:**
- **$C$**: Switching capacitance (depends on circuit activity)
- **$V$**: Supply voltage
- **$f$**: Clock frequency

**Energy per session and budget use:**
$$
E_{\\text{session}} = P \\times t_{\\text{duration}} \\quad \\text{(Wh)}
$$
$$
\\text{BudgetUse}(\\%) = \\frac{E_{\\text{session}}}{E_{\\text{budget}}} \\times 100
$$

**Accelerator advantage:** A specialized local accelerator achieves the same computation
at lower effective switching cost, yielding large energy-efficiency gains over CPU for ML workloads.
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER -- The Federation Paradox
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
<div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; {v2_11_variant.stakeholder}</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;Legal says we cannot send raw user data to the cloud, so we are betting on federated learning.
        But my engineers warn that with only 50 heterogeneous devices per round, the model may never converge.
        How many federation rounds does it actually take, and when does communication cost exceed centralized training?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; {v2_11_variant.stakeholder} &middot; {v2_11_profile.label}</div>
</div>
"""))

        # Banner
        items.append(mo.Html(f"""
        <div id="part-d" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: #7c3aed; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">D</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Part D &middot; 15 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                The Federation Paradox
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                Federated learning keeps data on-device. But non-IID data (each user types
                differently) causes 4-8x more communication rounds than IID. Gradient compression
                is the natural engineering response &mdash; but aggressive compression can add
                rounds, creating a U-shaped optimum in total communication cost.
            </div>
        </div>
        """))

        # Prediction
        items.append(mo.md("### Your Prediction"))
        items.append(pD_pred)

        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the federation simulator."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.hstack([pD_beta, pD_epochs, pD_compress], gap="1.5rem"))

        _beta = pD_beta.value
        _E = pD_epochs.value
        _compress = pD_compress.value
        _comm = federated_communication(
            v2_11_device,
            beta=_beta,
            local_epochs=_E,
            compression=_compress,
        )
        _iid_rounds = _comm.iid_rounds
        _noniid_rounds = _comm.noniid_rounds
        _compressed_rounds = _comm.compressed_rounds
        _compressed_bytes = _comm.compressed_bytes_per_round_mb
        _total_comm_mb = _comm.total_communication_mb
        _drift_penalty = _comm.drift_penalty
        _noniid_multiplier = _comm.round_multiplier / max(_drift_penalty, 1e-9)

        # Build convergence curves
        _round_range = np.arange(1, int(max(_noniid_rounds * 1.5, 200)))
        # IID accuracy curve: 1 - exp(-r / R_iid) * (1 - target)
        _iid_acc = 0.90 * (1 - np.exp(-_round_range / _iid_rounds * 3))
        _iid_acc = np.clip(_iid_acc, 0, 0.95)
        # Non-IID curve: slower convergence
        _noniid_rate = _iid_rounds / _noniid_rounds
        _noniid_acc = 0.90 * (1 - np.exp(-_round_range / (_iid_rounds / _noniid_rate) * 3))
        _noniid_acc = np.clip(_noniid_acc, 0, 0.92)
        # Compressed curve: slightly worse convergence rate
        _comp_rate = _iid_rounds / _compressed_rounds
        _comp_acc = 0.90 * (1 - np.exp(-_round_range / (_iid_rounds / _comp_rate) * 3))
        _comp_acc = np.clip(_comp_acc, 0, 0.91)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_round_range, y=_iid_acc, mode="lines",
            name="IID baseline", line=dict(color=COLORS["GreenLine"], width=3),
        ))
        _fig.add_trace(go.Scatter(
            x=_round_range, y=_noniid_acc, mode="lines",
            name=f"Non-IID (beta={_beta})", line=dict(color=COLORS["RedLine"], width=3),
        ))
        _fig.add_trace(go.Scatter(
            x=_round_range, y=_comp_acc, mode="lines",
            name=f"Non-IID + {_comm.compression_label} compression",
            line=dict(color=COLORS["BlueLine"], width=2, dash="dash"),
        ))
        _fig.add_hline(y=0.90, line_dash="dot", line_color="#94a3b8",
                       annotation_text="Target accuracy: 90%")
        _fig.update_layout(
            height=380,
            xaxis=dict(title="Communication Rounds"),
            yaxis=dict(title="Accuracy", range=[0, 1]),
            legend=dict(orientation="h", y=-0.2, font_size=11),
            margin=dict(l=50, r=20, t=30, b=80),
        )
        apply_plotly_theme(_fig)

        _round_ratio = _comm.round_multiplier
        _r_color = COLORS["RedLine"] if _round_ratio > 5 else COLORS["OrangeLine"] if _round_ratio > 2 else COLORS["GreenLine"]

        items.append(mo.as_html(_fig))
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">IID Rounds</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_iid_rounds}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_r_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Non-IID Rounds</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_r_color};">{_noniid_rounds:.0f}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total Comm</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_total_comm_mb/1024:.1f} GB</div>
            </div>
        </div>"""))

        items.append(mo.md(f"""
**Federation Physics** (beta={_beta}, E={_E}, compression={_comm.compression_label})

```
Non-IID multiplier  = {_noniid_multiplier:.1f}x
Drift penalty (E={_E}) = {_drift_penalty:.2f}x
Non-IID rounds      = {_iid_rounds} x {_noniid_multiplier:.1f} x {_drift_penalty:.2f} = {_noniid_rounds:.0f}
Bytes/round          = {_compressed_bytes:.1f} MB ({_comm.compression_label})
Total communication  = {_compressed_rounds:.0f} x {_compressed_bytes:.1f} MB = {_total_comm_mb/1024:.1f} GB
```
*Source: `mlsysbook_labs.federated_communication`, using `{v2_11_device.hardware_ref}`.*
"""))

        # Reveal
        if pD_pred.value == "C":
            _msg = (
                "**Correct.** Non-IID data at beta=0.5 requires 4-8x more communication rounds. "
                "The heterogeneity penalty is not linear -- it grows inversely with beta. "
                "Gradient compression (INT8) reduces per-round bytes by 4x but can add rounds "
                "due to information loss, creating a U-shaped optimum in total communication cost."
            )
            _kind = "success"
        else:
            _msg = (
                "**Non-IID data requires 4-8x more rounds.** At beta=0.5, the heterogeneity penalty "
                "multiplies the baseline 50 rounds by ~7x to ~350 rounds. Students underestimate "
                "this because they think 'more rounds just means a bit slower.' But each round "
                "requires a full model upload from every participating client."
            )
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))

        items.append(mo.accordion({
            "Math Peek: FedAvg Convergence under Non-IID Data": mo.md("""
**Convergence rounds scaling:**
$$
R_{\\text{non-IID}} = R_{\\text{IID}} \\times \\left(1 + \\frac{\\sigma^2_{\\text{het}}}{\\beta^2}\\right)
$$

**Where:**
- **$R_{\\text{IID}}$**: Baseline rounds under IID data distribution
- **$\\beta$**: Dirichlet concentration parameter (lower = more heterogeneous)
- **$\\sigma^2_{\\text{het}}$**: Variance of local data distributions across clients

**Communication cost per round:**
$$
C_{\\text{round}} = N_{\\text{clients}} \\times |\\theta| \\times b_{\\text{precision}}
$$

- **$N_{\\text{clients}}$**: Number of participating devices
- **$|\\theta|$**: Model parameter count
- **$b_{\\text{precision}}$**: Bytes per parameter (2 for FP16, reduced by compression)

**Gradient compression** reduces $b_{\\text{precision}}$ by 4-10x (INT8, Top-K sparsification).
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        items = []

        items.append(mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. Training memory is 4-12x inference memory.</strong>
                    Gradients, Adam optimizer state (2x params), and activation caching create a
                    memory amplification tax that makes the model that runs on a device unable to
                    learn on it without adaptation strategies like LoRA.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. The hardware execution target determines viability.</strong>
                    CPU adaptation can consume measurable energy budget per session.
                    Specialized acceleration can be dramatically more energy-efficient. Same
                    algorithm, different viability.
                </div>
                <div>
                    <strong>3. Non-IID data is the federation wall.</strong>
                    Heterogeneous client data causes 4-8x more communication rounds. Gradient
                    compression helps but introduces a U-shaped trade-off: too aggressive and
                    convergence degrades, requiring even more rounds.
                </div>
            </div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-takeaway-panel">
          <h2>Synthesis Checkpoint</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field">
              <strong>Prediction thread</strong>
              Compare the memory, storage, energy, and federation predictions against
              the computed result before writing the report.
            </div>
            <div class="mlsysbook-field">
              <strong>Evidence summary</strong>
              Use the active memory budget, adaptation storage curve, energy budget
              use, and non-IID round count as the evidence set.
            </div>
            <div class="mlsysbook-field">
              <strong>Decision</strong>
              Choose the adaptation strategy, execution target, and communication
              policy that you would defend for {v2_11_profile.label}.
            </div>
            <div class="mlsysbook-field">
              <strong>Residual risk</strong>
              State which measurement or rollout validation would still be needed
              before treating the edge architecture as deployable.
            </div>
          </div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab V2-12: The Silent Fleet</strong> &mdash; You learned to adapt
                    on a single device. Now manage 200 models in production where silent failures
                    cost $1M/day and operational complexity grows quadratically with model count.
                    Carry forward the prediction, budget controls, evidence summary, decision,
                    report reflection, and residual risk from this edge architecture review.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> the Edge Intelligence chapter for full derivations.<br/>
                    <strong>Build:</strong> TinyTorch federated averaging module &mdash;
                    implement FedAvg with non-IID data simulation.
                </div>
            </div>
        </div>
        """))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A -- The Memory Amplification Tax":   build_part_a(),
        "Part B -- The Adaptation Strategy Selector": build_part_b(),
        "Part C -- The Battery Drain Reality":       build_part_c(),
        "Part D -- The Federation Paradox":          build_part_d(),
        "Synthesis":                                  build_synthesis(),
    })
    tabs
    return

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: LEDGER HUD
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 10: LEDGER HUD ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    ledger,
    mo,
    pA_pred,
    pA_strategy,
    pB_pred,
    pC_pred,
    pC_target,
    pD_pred,
    pD_compress,
    v2_11_device,
    v2_11_profile,
    v2_11_variant,
):
    _mem_pred = pA_pred.value if hasattr(pA_pred, 'value') else None
    _adapt = pA_strategy.value if hasattr(pA_strategy, 'value') else "full"
    _lora_pred = pB_pred.value if hasattr(pB_pred, 'value') else None
    _drain_pred = pC_pred.value if hasattr(pC_pred, 'value') else None
    _exec_target = pC_target.value if hasattr(pC_target, 'value') else "cpu"
    _fed_pred = pD_pred.value if hasattr(pD_pred, 'value') else None
    _compress = pD_compress.value if hasattr(pD_compress, 'value') else "none"
    ledger.save(chapter=11, design={
        "chapter": "v2_11",
        "track_id": v2_11_profile.track_id,
        "scenario_id": v2_11_variant.scenario_id,
        "hardware_ref": v2_11_device.hardware_ref,
        "partA_memory_prediction": _mem_pred,
        "partA_adaptation_strategy": _adapt,
        "partB_lora_prediction": _lora_pred,
        "partC_drain_prediction_pct": _drain_pred,
        "partC_execution_target": _exec_target,
        "partD_federation_prediction": _fed_pred,
        "partD_compression_choice": _compress,
    })

    mo.Html(f"""
    <div style="background: #0f172a; border-radius: 10px; padding: 18px 24px;
                margin-top: 32px; font-family: 'SF Mono', 'Fira Code', monospace;">
        <div style="color: #475569; font-size: 0.7rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 10px;">
            Design Ledger &middot; Lab V2-11 Saved
        </div>
        <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.8;">
            <span style="color: #64748b;">track:</span>
            <span style="color: {COLORS['BlueLine']};">{v2_11_profile.label}</span><br/>
            <span style="color: #64748b;">memory_amplification:</span>
            <span style="color: {COLORS['RedLine']};">4-12x</span><br/>
            <span style="color: #64748b;">best_strategy:</span>
            <span style="color: {COLORS['GreenLine']};">LoRA + {v2_11_device.accelerator_label}</span><br/>
            <span style="color: #64748b;">federation_penalty:</span>
            <span style="color: {COLORS['OrangeLine']};">4-8x rounds (non-IID)</span>
        </div>
    </div>
    """)
    return


# ─── DOWNLOADABLE TRACK REPORT ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    adaptation_storage,
    build_lab_report,
    energy_drain,
    federated_communication,
    mo,
    pA_batch,
    pA_params,
    pA_pred,
    pA_strategy,
    pB_contexts,
    pB_pred,
    pC_pred,
    pC_target,
    pD_beta,
    pD_compress,
    pD_epochs,
    pD_pred,
    report_export_panel,
    training_memory_breakdown,
    v2_11_device,
    v2_11_metadata,
    v2_11_profile,
    v2_11_variant,
):
    _memory = training_memory_breakdown(
        params_m=pA_params.value,
        batch_size=pA_batch.value,
        strategy=pA_strategy.value,
        available_memory_mb=v2_11_device.available_memory_mb,
    )
    _model_mb = v2_11_device.default_model_params_m * 1e6 * 4 / (1024 * 1024)
    _storage = adaptation_storage(contexts=pB_contexts.value, model_mb=_model_mb)
    _energy = energy_drain(v2_11_device, target=pC_target.value)
    _comm = federated_communication(
        v2_11_device,
        beta=pD_beta.value,
        local_epochs=pD_epochs.value,
        compression=pD_compress.value,
    )

    _incomplete = []
    if pA_pred.value is None:
        _incomplete.append("Part A prediction")
    if pB_pred.value is None:
        _incomplete.append("Part B prediction")
    if pC_pred.value is None:
        _incomplete.append("Part C energy prediction")
    if pD_pred.value is None:
        _incomplete.append("Part D federation prediction")

    _report = build_lab_report(
        v2_11_metadata,
        track=v2_11_profile.label,
        scenario=v2_11_variant.workload_summary,
        learning_objectives=(
            "Quantify edge training memory amplification for the selected track.",
            "Compare full fine-tuning, LoRA, and bias-only adaptation storage.",
            f"Estimate local energy budget use on {v2_11_profile.label}.",
            "Explain why non-IID edge data increases communication rounds.",
        ),
        predictions={
            "memory_amplification": pA_pred.value,
            "lora_storage_shape": pB_pred.value,
            "energy_budget_prediction_pct": pC_pred.value,
            "federation_round_prediction": pD_pred.value,
        },
        knob_settings={
            "model_params_m": pA_params.value,
            "batch_size": pA_batch.value,
            "adaptation_strategy": pA_strategy.value,
            "contexts": pB_contexts.value,
            "execution_target": pC_target.value,
            "heterogeneity_beta": pD_beta.value,
            "local_epochs": pD_epochs.value,
            "compression": pD_compress.value,
        },
        evidence_summary={
            "hardware_ref": v2_11_device.hardware_ref,
            "model_ref": v2_11_variant.model_ref,
            "active_memory_budget_mb": v2_11_device.available_memory_mb,
            "training_memory_mb": round(_memory.total_mb, 3),
            "training_fits": _memory.fits_memory,
            "lora_storage_mb": round(_storage.lora_total_mb, 3),
            "energy_budget_used_pct": round(_energy.budget_used_pct, 3),
            "noniid_rounds": round(_comm.noniid_rounds, 1),
            "total_communication_mb": round(_comm.total_communication_mb, 3),
        },
        final_decision=(
            f"Use {pA_strategy.value} adaptation with {pC_target.value} execution for "
            f"{v2_11_profile.label}, while treating non-IID communication as the residual scaling wall."
        ),
        big_takeaways=(
            "Edge feasibility is a joint memory, energy, latency, privacy, and communication decision.",
            "The same equations lead to different conclusions for iPhone, Oura Ring, RoboTaxi, and Cloud Fleet.",
            "Federation protects data locality but does not make communication free.",
        ),
        reflections={
            "diagnosis": (
                f"{v2_11_profile.label} is constrained first by "
                f"{', '.join(v2_11_profile.dominant_constraints)}."
            ),
            "tradeoff": (
                f"The selected path optimizes {v2_11_variant.primary_metric} while guarding "
                f"{v2_11_variant.guardrail_metric}."
            ),
            "residual_risk": (
                "The helper uses first-order teaching estimates; production deployment still needs measured "
                "device traces and rollout validation."
            ),
        },
        residual_risk=(
            "The report records source-traced estimates, not measured hardware traces. "
            "Validate on representative devices before deployment."
        ),
        source_trace={
            "track_id": v2_11_profile.track_id,
            "scenario_id": v2_11_variant.scenario_id,
            "hardware_ref": v2_11_variant.hardware_ref,
            "model_ref": v2_11_variant.model_ref,
            "shared_helper": "mlsysbook_labs.edge",
            "source_policy": v2_11_profile.source_policy,
        },
        result_snapshot={
            "device": v2_11_device,
            "memory": _memory,
            "storage": _storage,
            "energy": _energy,
            "communication": _comm,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V2-11 report is generated locally from the selected track, current controls, "
                "computed evidence, final decision, and residual risk."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return

if __name__ == "__main__":
    app.run()
