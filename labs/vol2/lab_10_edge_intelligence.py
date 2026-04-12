import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-10: THE EDGE THERMODYNAMICS LAB
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
#             NPU achieves 50x energy gain over CPU for fine-tuning.
#             Prediction: battery drain per fine-tuning session on CPU?
#
#   Part D — The Federation Paradox (15 min)
#             Non-IID data causes 4-8x communication rounds explosion.
#             Merges original Parts D+E: federation + communication-compression.
#
# Hardware: Smartphone — 8 GB RAM, 15 Wh battery, 35 TOPS NPU
# Design Ledger: chapter="v2_10"
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
        await micropip.install(
            "../../wheels/mlsysim-0.1.0-py3-none-any.whl", keep_going=False
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

    # ── Hardware from registry ──────────────────────────────────────────────
    _phone = Hardware.Edge.Generic_Phone   # iPhone 15 Pro (A17 Pro)
    _cloud = Hardware.Cloud.H100           # Cloud tier for comparison

    # ── Mobile hardware constants ────────────────────────────────────────────
    # Source: @sec-edge-intelligence, smartphone compute constraints
    MOBILE_RAM_GB     = _phone.memory.capacity.m_as("GB")  # 8 GB
    MOBILE_RAM_AVAIL_MB = 300.0   # available for ML after OS/apps
    MOBILE_BATTERY_WH = _phone.battery_capacity.m_as("Wh")  # 15 Wh
    MOBILE_CPU_POWER_W = 3.0      # sustained CPU power draw
    MOBILE_NPU_POWER_W = 0.5     # NPU power draw for equivalent workload
    NPU_SPEEDUP       = 20.0     # NPU latency speedup over CPU
    NPU_ENERGY_GAIN   = 50.0     # NPU energy efficiency gain over CPU

    # Cloud tier specs — for edge-vs-cloud contrast
    CLOUD_RAM_GB      = _cloud.memory.capacity.m_as("GiB")  # 80 GiB
    CLOUD_TDP_W       = _cloud.tdp.m_as("W")                # 700 W

    # ── Training memory model constants ──────────────────────────────────────
    # Source: @tbl-training-amplification, @fig-training-memory-amplifier
    BYTES_FP16 = 2
    BYTES_FP32 = 4
    # Adam optimizer stores 2 extra copies of parameters (m, v) in FP32
    ADAM_MULTIPLIER = 2  # 2x parameter size for optimizer state (FP32)
    # Activation memory ratio calibrated to chapter worked example
    ACTIVATION_RATIO = 0.39  # FP32 activation bytes per param per batch item

    # ── LoRA constants ───────────────────────────────────────────────────────
    # Source: @sec-edge-intelligence, LoRA rank decomposition
    LORA_RANK = 16
    LORA_FRACTION = 0.01  # LoRA trainable parameters ~1% of full model
    BIAS_FRACTION = 0.001  # Bias-only ~0.1% of parameters

    # ── Federated learning constants ─────────────────────────────────────────
    # Source: @sec-edge-intelligence, FedAvg convergence analysis
    IID_ROUNDS = 50         # baseline IID convergence rounds
    FL_TARGET_ACC = 0.90    # target accuracy for convergence

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math, DecisionLog,
        MOBILE_RAM_GB, MOBILE_RAM_AVAIL_MB, MOBILE_BATTERY_WH,
        MOBILE_CPU_POWER_W, MOBILE_NPU_POWER_W,
        CLOUD_RAM_GB, CLOUD_TDP_W,
        NPU_SPEEDUP, NPU_ENERGY_GAIN,
        BYTES_FP16, BYTES_FP32, ADAM_MULTIPLIER, ACTIVATION_RATIO,
        LORA_RANK, LORA_FRACTION, BIAS_FRACTION,
        IID_ROUNDS, FL_TARGET_ACC,
    )


# ─── CELL 1: HEADER ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(LAB_CSS, mo):
    mo.vstack([
        LAB_CSS,
        mo.Html("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 10
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Edge Thermodynamics Lab
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.1rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Memory &middot; Adaptation &middot; Battery &middot; Federation
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 700px; line-height: 1.65;">
                A product manager wants on-device fine-tuning: "It is just inference with
                a backward pass, right?" Wrong. Training memory is 4-12x inference memory,
                battery drain makes naive CPU training a product-killing feature, and
                federated learning's communication cost explodes under non-IID data.
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
                    Chapter: Edge Intelligence
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px;">
                <span class="badge badge-fail">4-12x memory amplification</span>
                <span class="badge badge-warn">significant battery drain on CPU</span>
                <span class="badge badge-info">NPU: 50x energy savings</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
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
                require 4-12x more memory, significantly drain the battery on CPU, and
                need 4-8x more communication rounds when data is non-IID?&rdquo;
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
    - **On-Device Energy** &mdash; CPU vs NPU power and latency for fine-tuning.
    - **Federated Learning** &mdash; FedAvg, non-IID data impact, gradient compression.
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 4: PART A WIDGETS ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    pA_pred = mo.ui.radio(
        options={
            "A: ~60 MB -- gradients add a bit": "A",
            "B: ~120 MB -- double the inference footprint": "B",
            "C: ~200-360 MB -- 5-9x amplification": "C",
            "D: ~1 GB -- training is always an order of magnitude more": "D",
        },
        label=(
            "A 10M-parameter model runs inference comfortably on a smartphone (40 MB). "
            "How much memory does full fine-tuning (Adam optimizer, batch size 8) require?"
        ),
    )
    return (pA_pred,)


# ─── CELL 5: PART A CONTROLS + PART B WIDGETS ────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pA_pred):
    mo.stop(pA_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pA_params = mo.ui.slider(
        start=1, stop=100, value=10, step=1,
        label="Model parameters (millions)",
    )
    pA_batch = mo.ui.slider(
        start=1, stop=32, value=8, step=1,
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
            "You need to store personalized models for 10 user contexts. Full fine-tuning "
            "stores a complete model per context (40 MB each = 400 MB). LoRA stores only "
            "the adapter weights. What is the total LoRA storage?"
        ),
    )
    return (pA_params, pA_batch, pA_strategy, pB_pred)


# ─── CELL 6: PART B CONTROLS + PART C WIDGETS ────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pB_pred):
    mo.stop(pB_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pB_contexts = mo.ui.slider(
        start=1, stop=20, value=10, step=1,
        label="Number of user contexts",
    )

    pC_pred = mo.ui.number(
        start=0.1, stop=50.0, value=None, step=0.1,
        label=(
            "A LoRA fine-tuning session takes 30 seconds on CPU at 3W. The phone has a 15 Wh "
            "battery. What percentage of battery does this single session consume? "
            "(Account for thermal throttling extending duration 2-3x.)"
        ),
    )
    return (pB_contexts, pC_pred)


# ─── CELL 7: PART C CONTROLS + PART D WIDGETS ────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pC_pred):
    mo.stop(pC_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pC_target = mo.ui.radio(
        options={"CPU": "cpu", "GPU (mobile)": "gpu", "NPU": "npu"},
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
            "100 clients, non-IID data (beta=0.5). IID convergence takes 50 rounds. "
            "How many rounds does non-IID require to reach the same accuracy?"
        ),
    )
    return (pC_target, pD_pred)


# ─── CELL 8: PART D CONTROLS ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pD_pred):
    mo.stop(pD_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

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
    return (pD_beta, pD_epochs, pD_compress)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ALL PARTS AS TABS (SINGLE CELL)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 9: TABS CELL ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, apply_plotly_theme, go, math, mo, np, ledger,
    ACTIVATION_RATIO, ADAM_MULTIPLIER, BYTES_FP16, BYTES_FP32,
    BIAS_FRACTION, LORA_FRACTION,
    MOBILE_RAM_AVAIL_MB, MOBILE_BATTERY_WH,
    MOBILE_CPU_POWER_W, MOBILE_NPU_POWER_W,
    NPU_SPEEDUP, NPU_ENERGY_GAIN,
    IID_ROUNDS, FL_TARGET_ACC,
    pA_pred, pA_params, pA_batch, pA_strategy,
    pB_pred, pB_contexts,
    pC_pred, pC_target,
    pD_pred, pD_beta, pD_epochs, pD_compress,
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
        Incoming Message &middot; Mobile ML Product Manager, Paxos Devices</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;We want to fine-tune our 350M-parameter language model directly on-device. The phone has 6 GB of RAM,
        but the OS and apps already consume 4 GB. Can we even fit the training graph in the remaining 2 GB,
        or does the memory amplification from optimizer states and activations make this impossible?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; Priya Ramanathan, Mobile ML Product Manager &middot; Paxos Devices</div>
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
                A model that comfortably runs inference on a smartphone cannot learn on that
                same device. Full fine-tuning requires 4-12x more memory due to gradients,
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
        _params = _params_m * 1e6
        _batch = pA_batch.value
        _strategy = pA_strategy.value

        # Trainable fraction
        _train_frac = {"full": 1.0, "lora": LORA_FRACTION, "bias": BIAS_FRACTION}[_strategy]
        _trainable = _params * _train_frac

        # Memory breakdown (all in MB)
        _weights_mb = _params * BYTES_FP16 / (1024 * 1024)
        _grads_mb = _trainable * BYTES_FP32 / (1024 * 1024)
        _optim_mb = _trainable * BYTES_FP32 * ADAM_MULTIPLIER / (1024 * 1024)
        _activ_mb = _params * _batch * ACTIVATION_RATIO * BYTES_FP32 / (1024 * 1024)
        if _strategy != "full":
            _activ_mb *= _train_frac * 10  # LoRA/bias need fewer activations but some
        _total_mb = _weights_mb + _grads_mb + _optim_mb + _activ_mb
        _infer_mb = _weights_mb

        _amplification = _total_mb / max(_infer_mb, 0.01)
        _oom = _total_mb > MOBILE_RAM_AVAIL_MB

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
        _fig.add_hline(y=MOBILE_RAM_AVAIL_MB, line_dash="dash", line_color=COLORS["RedLine"],
                       annotation_text=f"Smartphone RAM: {MOBILE_RAM_AVAIL_MB:.0f} MB",
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
                f"Required: {_total_mb:.0f} MB | Available: {MOBILE_RAM_AVAIL_MB:.0f} MB. "
                f"Switch to LoRA or reduce model size."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"Training fits within {MOBILE_RAM_AVAIL_MB:.0f} MB mobile RAM "
                f"({_total_mb:.0f} MB used, {MOBILE_RAM_AVAIL_MB - _total_mb:.0f} MB headroom)."
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
Weights       = {_params_m}M x {BYTES_FP16} bytes  = {_weights_mb:.0f} MB
Gradients     = {_trainable/1e6:.2f}M x {BYTES_FP32} bytes  = {_grads_mb:.0f} MB
Optimizer (Adam) = {_trainable/1e6:.2f}M x {BYTES_FP32} x {ADAM_MULTIPLIER}  = {_optim_mb:.0f} MB
Activations   = f(params, batch)         = {_activ_mb:.0f} MB
Total         = {_total_mb:.0f} MB ({_amplification:.1f}x inference)
```
*Source: @sec-edge-intelligence, training amplification table*
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
        Incoming Message &middot; Embedded Systems Engineer, Paxos Devices</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;I keep hearing that LoRA solves the on-device fine-tuning problem, but nobody tells me
        the actual memory numbers. For a 350M model, how much storage does LoRA rank-16 really save
        compared to full fine-tuning? And does bias-only tuning even move the needle on accuracy?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; Marcus Chen, Embedded Systems Engineer &middot; Paxos Devices</div>
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
        _model_mb = 40.0  # 10M params x 4 bytes (FP32 stored)

        # Storage per context
        _full_per_ctx = _model_mb  # store full model
        _lora_per_ctx = _model_mb * LORA_FRACTION + 0.2  # adapters + metadata
        _bias_per_ctx = _model_mb * BIAS_FRACTION + 0.1

        # Total = base model + N context-specific weights
        _full_total = _model_mb + _n_ctx * _full_per_ctx
        _lora_total = _model_mb + _n_ctx * _lora_per_ctx  # shared base + N adapters
        _bias_total = _model_mb + _n_ctx * _bias_per_ctx

        _ctx_range = np.arange(1, 21)
        _full_curve = _model_mb + _ctx_range * _full_per_ctx
        _lora_curve = _model_mb + _ctx_range * _lora_per_ctx
        _bias_curve = _model_mb + _ctx_range * _bias_per_ctx

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

        _savings = _full_total / max(_lora_total, 0.01)

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
            _msg = "**Correct.** LoRA adapters are ~1% of model size. 10 contexts: base model (40 MB) + 10 adapters (~0.6 MB each) = ~46 MB total, roughly 10x savings over full fine-tuning."
            _kind = "success"
        else:
            _msg = "**LoRA adapters are ~1% of model size.** A 10M-param model produces ~0.4 MB adapters per context. 10 contexts = base model (40 MB) + 10 x 0.6 MB = ~46 MB. Full fine-tuning would require 440 MB for the same 10 contexts."
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
        Incoming Message &middot; Battery Team Lead, Paxos Devices</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;Product wants continuous on-device inference, but our 5,000 mAh battery already struggles
        to last a full day. How many joules does a single forward pass actually cost at different
        batch sizes, and at what query rate do we start draining faster than the user can charge?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; Tomoko Ishida, Battery Systems Lead &middot; Paxos Devices</div>
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
                session that visibly drains the battery is a product-killing feature, not a
                product feature. The NPU changes the equation entirely.
            </div>
        </div>
        """))

        # Prediction
        items.append(mo.md("### Your Prediction"))
        items.append(mo.md("*Enter battery drain percentage for one CPU fine-tuning session:*"))
        items.append(pC_pred)

        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Enter your prediction to unlock the battery drain simulator."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(pC_target)

        _target = pC_target.value
        _base_duration_s = 30.0  # CPU baseline

        # Execution target properties
        _target_props = {
            "cpu": {"power_w": MOBILE_CPU_POWER_W, "duration_s": _base_duration_s * 2.5, "label": "CPU"},
            "gpu": {"power_w": 2.0, "duration_s": _base_duration_s * 1.5, "label": "Mobile GPU"},
            "npu": {"power_w": MOBILE_NPU_POWER_W, "duration_s": _base_duration_s / NPU_SPEEDUP, "label": "NPU"},
        }
        _props = _target_props[_target]
        _power = _props["power_w"]
        _duration = _props["duration_s"]

        # Battery drain calculation
        # drain_pct = (power_W * duration_s) / (battery_Wh * 3600) * 100
        _energy_wh = _power * _duration / 3600
        _drain_pct = (_energy_wh / MOBILE_BATTERY_WH) * 100
        _sessions_per_charge = 100.0 / _drain_pct if _drain_pct > 0 else float('inf')

        # Comparison bars
        _targets = ["CPU", "Mobile GPU", "NPU"]
        _drains = []
        _durations = []
        for _t in ["cpu", "gpu", "npu"]:
            _p = _target_props[_t]
            _e = _p["power_w"] * _p["duration_s"] / 3600
            _d = (_e / MOBILE_BATTERY_WH) * 100
            _drains.append(_d)
            _durations.append(_p["duration_s"])

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
            yaxis=dict(title="Battery Drain per Session (%)"),
            margin=dict(l=50, r=20, t=30, b=40),
        )
        apply_plotly_theme(_fig)

        items.append(mo.as_html(_fig))

        # Failure state
        _drain_color = COLORS["RedLine"] if _drain_pct > 5 else COLORS["OrangeLine"] if _drain_pct > 1 else COLORS["GreenLine"]
        if _drain_pct > 5:
            items.append(mo.callout(mo.md(
                f"**Product-killing battery drain.** {_drain_pct:.1f}% per session means only "
                f"{_sessions_per_charge:.0f} sessions per full charge. Users will disable this feature."
            ), kind="danger"))
        elif _drain_pct > 1:
            items.append(mo.callout(mo.md(
                f"**Marginal.** {_drain_pct:.1f}% per session is noticeable. {_sessions_per_charge:.0f} "
                f"sessions per charge. Consider NPU for production deployment."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Viable.** {_drain_pct:.2f}% per session. {_sessions_per_charge:.0f} sessions per "
                f"full charge. This is a product feature, not a battery drain."
            ), kind="success"))

        # Cards
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_drain_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Drain ({_props['label']})</div>
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
**Battery Drain Formula**

```
Energy      = Power x Duration = {_power:.1f}W x {_duration:.1f}s = {_energy_wh:.4f} Wh
Drain (%)   = Energy / Battery x 100 = {_energy_wh:.4f} / {MOBILE_BATTERY_WH} x 100 = {_drain_pct:.2f}%
Sessions    = 100% / {_drain_pct:.2f}% = {_sessions_per_charge:.0f}
```

You predicted: {_predicted:.1f}%. Actual CPU drain: {_cpu_drain:.1f}%.

*Source: @sec-edge-intelligence, on-device energy model*
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

**Energy per session and battery drain:**
$$
E_{\\text{session}} = P \\times t_{\\text{duration}} \\quad \\text{(Wh)}
$$
$$
\\text{Drain}(\\%) = \\frac{E_{\\text{session}}}{E_{\\text{battery}}} \\times 100
$$

**NPU advantage:** NPU achieves the same computation at lower $V$ and optimized $C$,
yielding $\\sim$50x energy efficiency over CPU for ML workloads.
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
        Incoming Message &middot; Privacy Officer, Paxos Devices</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;Legal says we cannot send raw user data to the cloud, so we are betting on federated learning.
        But my engineers warn that with only 50 heterogeneous devices per round, the model may never converge.
        How many federation rounds does it actually take, and when does communication cost exceed centralized training?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; Elena Vasquez, Chief Privacy Officer &middot; Paxos Devices</div>
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

        # Convergence model: non-IID penalty from heterogeneity
        # Source: @sec-edge-intelligence, FedAvg convergence analysis
        # Rounds scale as: R_noniid = R_iid * (1 + alpha / beta)
        # where alpha captures the heterogeneity penalty
        _alpha = 3.0  # calibrated so beta=0.5 gives ~4-8x rounds
        _noniid_multiplier = 1 + _alpha / _beta

        # Client drift from excess local epochs
        # Beyond E=5, drift causes convergence to slow or diverge
        _drift_penalty = 1.0 if _E <= 3 else 1 + 0.15 * (_E - 3)
        if _E > 10:
            _drift_penalty = 1 + 0.15 * 7 + 0.3 * (_E - 10)  # accelerating penalty

        _noniid_rounds = IID_ROUNDS * _noniid_multiplier * _drift_penalty

        # Compression effects
        _compress_props = {
            "none": {"bytes_mult": 1.0, "quality_penalty": 1.0, "label": "None"},
            "int8": {"bytes_mult": 0.25, "quality_penalty": 1.05, "label": "INT8"},
            "int4": {"bytes_mult": 0.125, "quality_penalty": 1.15, "label": "INT4"},
            "topk": {"bytes_mult": 0.1, "quality_penalty": 1.25, "label": "Top-K"},
        }
        _cp = _compress_props[_compress]
        _compressed_rounds = _noniid_rounds * _cp["quality_penalty"]
        _bytes_per_round = 40.0  # MB baseline (10M params x 4 bytes)
        _compressed_bytes = _bytes_per_round * _cp["bytes_mult"]
        _total_comm_mb = _compressed_rounds * _compressed_bytes

        # Build convergence curves
        _round_range = np.arange(1, int(max(_noniid_rounds * 1.5, 200)))
        # IID accuracy curve: 1 - exp(-r / R_iid) * (1 - target)
        _iid_acc = 0.90 * (1 - np.exp(-_round_range / IID_ROUNDS * 3))
        _iid_acc = np.clip(_iid_acc, 0, 0.95)
        # Non-IID curve: slower convergence
        _noniid_rate = IID_ROUNDS / _noniid_rounds
        _noniid_acc = 0.90 * (1 - np.exp(-_round_range / (IID_ROUNDS / _noniid_rate) * 3))
        _noniid_acc = np.clip(_noniid_acc, 0, 0.92)
        # Compressed curve: slightly worse convergence rate
        _comp_rate = IID_ROUNDS / _compressed_rounds
        _comp_acc = 0.90 * (1 - np.exp(-_round_range / (IID_ROUNDS / _comp_rate) * 3))
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
            name=f"Non-IID + {_cp['label']} compression",
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

        _round_ratio = _noniid_rounds / IID_ROUNDS
        _r_color = COLORS["RedLine"] if _round_ratio > 5 else COLORS["OrangeLine"] if _round_ratio > 2 else COLORS["GreenLine"]

        items.append(mo.as_html(_fig))
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">IID Rounds</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{IID_ROUNDS}</div>
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
**Federation Physics** (beta={_beta}, E={_E}, compression={_cp['label']})

```
Non-IID multiplier  = 1 + alpha/beta = 1 + {_alpha}/{_beta} = {_noniid_multiplier:.1f}x
Drift penalty (E={_E}) = {_drift_penalty:.2f}x
Non-IID rounds      = {IID_ROUNDS} x {_noniid_multiplier:.1f} x {_drift_penalty:.2f} = {_noniid_rounds:.0f}
Bytes/round          = {_compressed_bytes:.1f} MB ({_cp['label']})
Total communication  = {_compressed_rounds:.0f} x {_compressed_bytes:.1f} MB = {_total_comm_mb/1024:.1f} GB
```
*Source: @sec-edge-intelligence, FedAvg convergence*
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
                    CPU fine-tuning drains measurable battery per session due to thermal throttling.
                    NPU fine-tuning is 50-100x more energy-efficient. Same algorithm,
                    50x energy difference. The NPU makes on-device training a product feature.
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
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab V2-11: The Silent Fleet</strong> &mdash; You learned to train
                    on a single device. Now manage 200 models in production where silent failures
                    cost $1M/day and operational complexity grows quadratically with model count.
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
def _(COLORS, ledger, mo, pA_pred, pA_strategy, pB_pred, pC_pred, pC_target, pD_pred, pD_compress):
    _mem_pred = pA_pred.value if hasattr(pA_pred, 'value') else None
    _adapt = pA_strategy.value if hasattr(pA_strategy, 'value') else "full"
    _lora_pred = pB_pred.value if hasattr(pB_pred, 'value') else None
    _drain_pred = pC_pred.value if hasattr(pC_pred, 'value') else None
    _exec_target = pC_target.value if hasattr(pC_target, 'value') else "cpu"
    _fed_pred = pD_pred.value if hasattr(pD_pred, 'value') else None
    _compress = pD_compress.value if hasattr(pD_compress, 'value') else "none"
    ledger.save(chapter=10, design={
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
            Design Ledger &middot; Lab V2-10 Saved
        </div>
        <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.8;">
            <span style="color: #64748b;">memory_amplification:</span>
            <span style="color: {COLORS['RedLine']};">4-12x</span><br/>
            <span style="color: #64748b;">best_strategy:</span>
            <span style="color: {COLORS['GreenLine']};">LoRA + NPU</span><br/>
            <span style="color: #64748b;">federation_penalty:</span>
            <span style="color: {COLORS['OrangeLine']};">4-8x rounds (non-IID)</span>
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
