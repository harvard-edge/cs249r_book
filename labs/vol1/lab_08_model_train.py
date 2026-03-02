import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 08: THE TRAINING MEMORY BUDGET
#
# Chapter: Model Training (@sec-model-training)
# Core Invariant:
#   Training memory = weights + gradients + optimizer state + activations.
#   For Adam in FP32: 16× model size (4× base × 4 bytes). Activations scale
#   with batch size. OOM is the most common training failure.
#
# 2 Contexts: Cloud (H100, 80 GB) vs Mobile (fine-tuning device, 8 GB)
#
# Act I  — The Memory Budget Shock (12–15 min)
#   Prediction: 7B FP32 Adam minimum memory (correct: 112 GB)
#   Instrument: Memory Ledger stacked bar
#   Reflection: Why bf16 cuts memory by 2×
#
# Act II — Gradient Accumulation vs Batch Size (20–25 min)
#   Prediction: Gradient accumulation throughput vs direct batch
#   Instrument: Activation + accumulation explorer
#   Failure state: OOM banner when total > device RAM
#   Reflection: Gradient checkpointing trade-off
#
# Design Ledger: chapter=8, context, model_size_params, precision_chosen,
#                oom_triggered, grad_accum_steps, optimizer_chosen
# ─────────────────────────────────────────────────────────────────────────────


# ── CELL 0: SETUP ────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    # ── Hardware constants (from LABS_SPEC.md, all plain floats, source: NVIDIA specs) ──
    H100_RAM_GB      = 80     # H100 SXM5 HBM3e memory capacity
    H100_TDP_W       = 700    # H100 TDP
    MOBILE_RAM_GB    = 8      # Smartphone / fine-tuning device budget
    MOBILE_TDP_W     = 5      # Watts sustained

    # ── Precision bytes per value ─────────────────────────────────────────────
    # Sources: IEEE 754 standards; BYTES_FP32=4, BYTES_BF16=2, INT8=1
    DTYPE_BYTES = {
        "fp32": 4,
        "bf16": 2,
        "fp16": 2,
        "int8": 1,
    }

    # ── Optimizer memory multipliers relative to weight bytes ─────────────────
    # SGD: weights only → 1×; Momentum: +1 vector → 2×;
    # Adam: +m_t + v_t (FP32 always, even in mixed precision) → +2× FP32
    # Source: training.qmd §Training Systems Fundamentals:
    #   "4× the inference memory cost per parameter when using Adam:
    #    14 GB (weights FP16) + 14 GB (grads) + 28 GB (Adam m+v FP32) = 56 GB"
    #   (for 7B model). For FP32: weights+grads+2×optimizer = 4×weight_size.
    OPTIMIZER_EXTRA_MULTIPLIER = {
        "sgd":       0,   # no extra state
        "momentum":  1,   # one moment vector (same dtype as weights)
        "adam":      2,   # two FP32 moment vectors (m_t, v_t)
        "adafactor": 0.5, # factored moments, ~0.5× weight size extra
    }

    ledger = DesignLedger()
    return (
        mo, go, np, math,
        ledger, COLORS, LAB_CSS, apply_plotly_theme,
        H100_RAM_GB, MOBILE_RAM_GB, H100_TDP_W, MOBILE_TDP_W,
        DTYPE_BYTES, OPTIMIZER_EXTRA_MULTIPLIER,
    )


# ── CELL 1: HEADER ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 08
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Training Memory Budget
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 680px; line-height: 1.65;">
                Training a 7B-parameter model requires not 7 GB of memory — it requires
                <strong style="color:#f8fafc;">112 GB</strong>.
                OOM is the most common training failure, and it is entirely predictable
                from first principles.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: Memory Budget Shock &middot; Act II: Gradient Accumulation
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35&ndash;40 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    OOM failure state active
                </span>
            </div>
        </div>
        """),
    ])
    return


# ── CELL 2: RECOMMENDED READING ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-model-training-training-systems-fundamentals-05d2** — Training Systems Fundamentals:
      the 4× memory multiplier, Adam optimizer state, the most common OOM failure mode.
    - **@sec-model-training-iron-law-training-performance-a53f** — Iron Law of Training Performance:
      the three levers (operations, peak throughput, utilization).
    - The **Mixed-Precision Training** and **Gradient Checkpointing** footnotes in the chapter.
    """), kind="info")
    return


# ── CELL 3: CONTEXT TOGGLE ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Cloud (H100, 80 GB HBM)": "cloud",
            "Mobile / Fine-tuning (8 GB)": "mobile",
        },
        value="Cloud (H100, 80 GB HBM)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md("**Select your deployment context.** This determines the memory budget for both acts."),
        context_toggle,
    ])
    return (context_toggle,)


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — THE MEMORY BUDGET SHOCK
# ─────────────────────────────────────────────────────────────────────────────

# ── CELL 4: ACT I SCENARIO ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["RedLine"]
    mo.vstack([
        mo.md("---"),
        mo.md("## Act I — The Memory Budget Shock"),
        mo.Html(f"""
        <div style="border-left: 4px solid {_color}; background: #fef2f2;
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; Junior Engineer
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "Our new 7B model training job OOM'd on the H100 after 2 steps. I set
                precision to fp32 and used Adam — that should be fine, right? H100 has
                80 GB and the model is only 7 billion parameters."
            </div>
        </div>
        """),
        mo.md("""
        The junior engineer is wrong, but not carelessly — the mistake is a specific,
        systematic underestimate of what training actually needs. Before we debug the job,
        predict the actual memory requirement from first principles.
        """),
    ])
    return


# ── CELL 5: ACT I PREDICTION LOCK ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) 7 GB — one byte per parameter": "7gb",
            "B) 28 GB — FP32 weights only": "28gb",
            "C) 56 GB — weights + gradients in FP32": "56gb",
            "D) 112 GB — weights + gradients + Adam m_t + v_t, all FP32": "112gb",
        },
        label=(
            "Training a 7B parameter model in FP32 with Adam optimizer. "
            "What is the minimum memory required for weights + gradients + "
            "optimizer state (before any activations)?"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background: #1e293b; border-radius: 12px; padding: 20px;
                    border-left: 4px solid #6366f1; margin: 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #a5b4fc;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                Prediction Lock — Act I
            </div>
            <div style="color: #e2e8f0; font-size: 0.88rem; margin-bottom: 12px;">
                Commit to a prediction before touching any controls.
                The instruments will unlock once you select an answer.
            </div>
        </div>
        """),
        act1_prediction,
    ])
    return (act1_prediction,)


# ── CELL 6: ACT I GATE ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Memory Ledger."),
            kind="warn",
        ),
    )
    return


# ── CELL 7: ACT I CONTROLS ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    # Model size slider: 100M to 70B parameters
    # Range source: training.qmd — GPT-2 (1.5B), GPT-3 (175B), 7B class models
    act1_model_billions = mo.ui.slider(
        start=0.1, stop=70, value=7, step=0.1,
        label="Model size (billions of parameters)",
    )
    # Precision selector
    act1_precision = mo.ui.dropdown(
        options={
            "FP32 (4 bytes/value)": "fp32",
            "BF16 (2 bytes/value)": "bf16",
            "FP16 (2 bytes/value)": "fp16",
            "INT8 (1 byte/value)":  "int8",
        },
        value="FP32 (4 bytes/value)",
        label="Training precision",
    )
    # Optimizer selector
    act1_optimizer = mo.ui.dropdown(
        options={
            "Adam (m_t + v_t in FP32)": "adam",
            "SGD (no optimizer state)": "sgd",
            "Momentum (one extra vector)": "momentum",
            "Adafactor (~0.5× extra)": "adafactor",
        },
        value="Adam (m_t + v_t in FP32)",
        label="Optimizer",
    )
    # Batch size slider: activations scale with batch
    act1_batch_size = mo.ui.slider(
        start=1, stop=128, value=8, step=1,
        label="Batch size (for activation estimate)",
    )
    mo.vstack([
        mo.md("### Memory Ledger Controls"),
        mo.hstack([act1_model_billions, act1_precision], justify="start", gap="2rem"),
        mo.hstack([act1_optimizer, act1_batch_size], justify="start", gap="2rem"),
    ])
    return (act1_model_billions, act1_precision, act1_optimizer, act1_batch_size)


# ── CELL 8: ACT I PHYSICS ENGINE + CHART ────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, np,
    act1_model_billions, act1_precision, act1_optimizer, act1_batch_size,
    context_toggle, COLORS, apply_plotly_theme,
    H100_RAM_GB, MOBILE_RAM_GB, DTYPE_BYTES, OPTIMIZER_EXTRA_MULTIPLIER,
):
    # ── Physics ───────────────────────────────────────────────────────────────
    # Source: training.qmd §Training Systems Fundamentals callout-definition:
    #   "a 7B-parameter model requires 14 GB (FP16 weights) + 14 GB (gradients)
    #    + 28 GB (Adam first and second moments in FP32) = 56 GB minimum"
    # For FP32 weights (7B): 7B×4B = 28 GB weights, 28 GB grads, 56 GB optimizer = 112 GB

    _params_b = act1_model_billions.value * 1e9   # total parameter count
    _dtype    = act1_precision.value
    _opt      = act1_optimizer.value
    _batch    = act1_batch_size.value
    _ctx      = context_toggle.value

    _bpv = DTYPE_BYTES[_dtype]   # bytes per value for weights/grads

    # Weights: params × bytes_per_value
    _weights_gb = (_params_b * _bpv) / 1e9

    # Gradients: same dtype and size as weights (one gradient per parameter)
    _grads_gb = _weights_gb

    # Optimizer state: stored in FP32 regardless of training precision
    # Adam: 2 FP32 vectors (m_t, v_t) → 2 × params × 4 bytes
    # SGD: 0 extra; Momentum: 1 FP32 vector; Adafactor: ~0.5× FP32
    _opt_extra_fp32_vectors = OPTIMIZER_EXTRA_MULTIPLIER[_opt]
    _optimizer_gb = (_params_b * 4 * _opt_extra_fp32_vectors) / 1e9

    # Activations: linear in batch size and model depth
    # Simplified estimate: batch × 1024 (seq_len) × 4096 (hidden) × 32 (layers) × bpv
    # Calibrated to GPT-2 XL (1.5B, 48 layers, h=1600): ~2 GB at batch=8
    # Source: training.qmd footnote [fn-checkpointing-training]: activation storage
    #   grows linearly with model depth and batch size.
    _hidden_dim  = max(512, int(act1_model_billions.value ** 0.5 * 1024))
    _n_layers    = max(12,  int(act1_model_billions.value ** 0.33 * 12))
    _seq_len     = 1024   # standard transformer sequence length
    _act_bytes   = _batch * _seq_len * _hidden_dim * _n_layers * _bpv
    _activations_gb = _act_bytes / 1e9

    _total_gb = _weights_gb + _grads_gb + _optimizer_gb + _activations_gb
    _static_gb = _weights_gb + _grads_gb + _optimizer_gb  # before activations

    # Device memory budget
    _device_ram = H100_RAM_GB if _ctx == "cloud" else MOBILE_RAM_GB
    _device_name = "H100 (80 GB)" if _ctx == "cloud" else "Mobile / Fine-tuning (8 GB)"
    _oom = _total_gb > _device_ram
    _oom_static = _static_gb > _device_ram  # OOM even before first activation

    # ── Colors ───────────────────────────────────────────────────────────────
    if _oom:
        _bar_colors = [
            COLORS["RedLine"], COLORS["RedLine"], COLORS["RedLine"], COLORS["RedLine"]
        ]
    else:
        _bar_colors = [
            COLORS["BlueLine"],   # weights
            COLORS["GreenLine"],  # gradients
            COLORS["OrangeLine"], # optimizer state
            "#6366f1",            # activations (indigo)
        ]

    # ── Stacked bar chart ─────────────────────────────────────────────────────
    _components = ["Weights", "Gradients", "Optimizer State", "Activations"]
    _values_gb  = [_weights_gb, _grads_gb, _optimizer_gb, _activations_gb]

    _fig = go.Figure()
    for _i, (_name, _val, _color) in enumerate(
        zip(_components, _values_gb, _bar_colors)
    ):
        _fig.add_trace(go.Bar(
            name=_name,
            x=["Training Memory"],
            y=[_val],
            marker_color=_color,
            text=f"{_val:.1f} GB",
            textposition="inside",
            textfont=dict(color="white", size=12, family="SF Mono, monospace"),
        ))

    # Device RAM threshold line
    _fig.add_hline(
        y=_device_ram,
        line_color=COLORS["RedLine"],
        line_width=2.5,
        line_dash="dash",
        annotation_text=f"{_device_name} RAM limit",
        annotation_position="right",
        annotation_font_color=COLORS["RedLine"],
    )

    _fig.update_layout(
        barmode="stack",
        height=420,
        yaxis_title="Memory (GB)",
        yaxis=dict(range=[0, max(_total_gb * 1.15, _device_ram * 1.15)]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=160, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    _fig = apply_plotly_theme(_fig)

    # ── Formula display ───────────────────────────────────────────────────────
    _bpv_opt = 4  # optimizer moments always FP32
    _opt_label = {
        "adam":      f"2 × {act1_model_billions.value:.1f}B × 4 B = {_optimizer_gb:.1f} GB",
        "sgd":       "0 GB (no optimizer state)",
        "momentum":  f"1 × {act1_model_billions.value:.1f}B × 4 B = {_optimizer_gb:.1f} GB",
        "adafactor": f"0.5 × {act1_model_billions.value:.1f}B × 4 B = {_optimizer_gb:.1f} GB",
    }[_opt]

    _formula_md = f"""
### Memory Ledger — Physics

```
Weights        = {act1_model_billions.value:.1f}B params × {_bpv} bytes  = {_weights_gb:.1f} GB
Gradients      = {act1_model_billions.value:.1f}B params × {_bpv} bytes  = {_grads_gb:.1f} GB
Optimizer state= {_opt_label}
Activations    ≈ batch={_batch} × seq={_seq_len} × h={_hidden_dim} × L={_n_layers} × {_bpv}B
               = {_activations_gb:.1f} GB
─────────────────────────────────────────────────
Static floor   = {_static_gb:.1f} GB  (weights + grads + optimizer, before any batch)
Total          = {_total_gb:.1f} GB
Device budget  = {_device_ram:.0f} GB  ({_device_name})
```
"""

    # ── Summary metric cards ──────────────────────────────────────────────────
    _status_color = COLORS["RedLine"] if _oom else COLORS["GreenLine"]
    _status_label = "OOM" if _oom else "Fits"

    _cards_html = f"""
<div style="display: flex; gap: 16px; justify-content: flex-start;
            flex-wrap: wrap; margin: 16px 0;">
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                min-width: 150px; text-align: center; background: white;">
        <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.06em;">Static Floor</div>
        <div style="font-size: 2rem; font-weight: 800; color: {COLORS['BlueLine']};
                    font-family: SF Mono, monospace;">
            {_static_gb:.1f} GB
        </div>
        <div style="color: #94a3b8; font-size: 0.78rem;">before activations</div>
    </div>
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                min-width: 150px; text-align: center; background: white;">
        <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.06em;">Total Required</div>
        <div style="font-size: 2rem; font-weight: 800; color: {COLORS['OrangeLine']};
                    font-family: SF Mono, monospace;">
            {_total_gb:.1f} GB
        </div>
        <div style="color: #94a3b8; font-size: 0.78rem;">incl. activations</div>
    </div>
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                min-width: 150px; text-align: center; background: white;">
        <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.06em;">Device Budget</div>
        <div style="font-size: 2rem; font-weight: 800; color: #475569;
                    font-family: SF Mono, monospace;">
            {_device_ram:.0f} GB
        </div>
        <div style="color: #94a3b8; font-size: 0.78rem;">{_device_name}</div>
    </div>
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                min-width: 150px; text-align: center; background: white;">
        <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.06em;">Status</div>
        <div style="font-size: 2rem; font-weight: 800; color: {_status_color};
                    font-family: SF Mono, monospace;">
            {_status_label}
        </div>
        <div style="color: #94a3b8; font-size: 0.78rem;">
            {"exceeds budget" if _oom else "within budget"}
        </div>
    </div>
</div>
"""

    # ── Assemble ──────────────────────────────────────────────────────────────
    _items = [
        mo.md(_formula_md),
        mo.Html(_cards_html),
        mo.ui.plotly(_fig),
    ]

    if _oom_static:
        _items.append(mo.callout(
            mo.md(
                f"**OOM — Static memory floor exceeds device budget.** "
                f"The {_opt.upper()} optimizer requires {_static_gb:.1f} GB for weights, "
                f"gradients, and optimizer state alone — before a single activation is "
                f"computed. {_device_name} has {_device_ram:.0f} GB. "
                f"**No batch size change can fix this.** "
                f"Try switching to bf16 precision or a smaller optimizer."
            ),
            kind="danger",
        ))
    elif _oom:
        _items.append(mo.callout(
            mo.md(
                f"**OOM — Activation memory pushes over budget.** "
                f"Static memory ({_static_gb:.1f} GB) fits, but activations at "
                f"batch={_batch} add {_activations_gb:.1f} GB, totalling "
                f"{_total_gb:.1f} GB. {_device_name} has {_device_ram:.0f} GB. "
                f"**Reduce batch size, enable gradient checkpointing, or switch precision.**"
            ),
            kind="danger",
        ))
    else:
        _items.append(mo.callout(
            mo.md(
                f"**Configuration fits.** Total {_total_gb:.1f} GB within "
                f"{_device_ram:.0f} GB budget — {_device_ram - _total_gb:.1f} GB headroom."
            ),
            kind="success",
        ))

    mo.vstack(_items)
    return (
        _weights_gb, _grads_gb, _optimizer_gb, _activations_gb,
        _static_gb, _total_gb,
    )


# ── CELL 9: ACT I PREDICTION REVEAL ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction):
    # Source: training.qmd §Training Systems Fundamentals callout-definition:
    #   "a 7B-parameter model requires 14 GB (FP16 weights) + 14 GB (gradients)
    #    + 28 GB (Adam first and second moments in FP32) = 56 GB minimum"
    # For FP32 training: 28 GB weights + 28 GB grads + 56 GB Adam state = 112 GB.

    _actual_gb = 112  # 7B × 4B × 4 components (weights, grads, m_t, v_t)

    _predicted_label = {
        "7gb":   "A) 7 GB",
        "28gb":  "B) 28 GB",
        "56gb":  "C) 56 GB",
        "112gb": "D) 112 GB",
    }.get(act1_prediction.value, "—")

    _predicted_val = {
        "7gb":   7,
        "28gb":  28,
        "56gb":  56,
        "112gb": 112,
    }.get(act1_prediction.value, 0)

    _ratio = _actual_gb / _predicted_val if _predicted_val > 0 else float("inf")
    _correct = act1_prediction.value == "112gb"

    if _correct:
        mo.callout(mo.md(
            f"**Correct.** You predicted {_predicted_label} = {_predicted_val} GB. "
            f"The actual minimum is **{_actual_gb} GB** — exactly matching your prediction. "
            f"The breakdown: 7B × 4 bytes = 28 GB weights + 28 GB gradients "
            f"+ 28 GB Adam m_t (FP32) + 28 GB Adam v_t (FP32) = 112 GB. "
            f"This exceeds the H100's 80 GB capacity before a single activation is stored. "
            f"The junior engineer's 7B model requires 112 GB, not 7 GB — a 16× underestimate."
        ), kind="success")
    else:
        mo.callout(mo.md(
            f"**You predicted {_predicted_label} = {_predicted_val} GB. "
            f"The actual minimum is {_actual_gb} GB — you were off by {_ratio:.1f}×.** "
            f"The 7B model in FP32 with Adam needs: "
            f"28 GB (weights) + 28 GB (gradients) + 28 GB (Adam m_t) + 28 GB (Adam v_t) = **112 GB**. "
            f"This is the 4× training multiplier: each of the four components is exactly "
            f"one copy of the model. The H100's 80 GB is not enough — training OOM'd "
            f"because the static memory floor alone exceeds the device budget."
        ), kind="warn")
    return


# ── CELL 10: ACT I REFLECTION ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) bf16 is less accurate so it stores less data per value": "wrong_accuracy",
            "B) bf16 uses 2 bytes per value instead of 4 bytes, halving every buffer": "correct",
            "C) bf16 disables gradient computation entirely": "wrong_no_grad",
            "D) bf16 fuses optimizer kernels, reducing memory overhead": "wrong_fusion",
        },
        label=(
            "Switching from fp32 to bf16 training precision cuts the memory for "
            "weights and gradients by 2×. Why?"
        ),
    )
    mo.vstack([
        mo.md("### Reflection — Act I"),
        act1_reflection,
    ])
    return (act1_reflection,)


# ── CELL 11: ACT I REFLECTION FEEDBACK ───────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"),
    )

    _feedback = {
        "correct": (
            "**Correct.** bf16 is a 16-bit floating-point format: 1 sign bit, "
            "8 exponent bits (same as fp32), 7 mantissa bits (vs 23 in fp32). "
            "Fewer bits per value = fewer bytes: 2 bytes instead of 4. "
            "Every weight and gradient buffer halves. The accuracy difference is small "
            "because bf16 preserves fp32's exponent range — making it suitable for "
            "training without the overflow problems of fp16. "
            "**However,** Adam's m_t and v_t moment vectors are still stored in fp32 "
            "for numerical stability, so the full optimizer-state savings are 1.5× "
            "rather than 2× on total memory."
        ),
        "wrong_accuracy": (
            "**Not quite.** The bit-width change is not about accuracy tolerance — "
            "it is about the physical representation size. bf16 uses 16 bits (2 bytes) "
            "vs fp32's 32 bits (4 bytes). The precision difference (7 vs 23 mantissa bits) "
            "is a consequence of the smaller size, not its cause."
        ),
        "wrong_no_grad": (
            "**Not quite.** bf16 does not disable gradients. The backward pass "
            "runs in bf16 just as the forward pass does. What changes is the number "
            "of bytes per gradient value: 2 bytes in bf16 vs 4 in fp32."
        ),
        "wrong_fusion": (
            "**Not quite.** Kernel fusion (covered in Lab 07) reduces kernel launch "
            "overhead and improves compute efficiency, but it does not change the "
            "number of bytes stored for weights or gradients. The memory savings "
            "from bf16 come from the reduced bit-width of the format itself."
        ),
    }[act1_reflection.value]

    _kind = "success" if act1_reflection.value == "correct" else "warn"
    mo.callout(mo.md(_feedback), kind=_kind)
    return


# ── CELL 12: ACT I MATHPEEK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — Training Memory": mo.md(r"""
**Memory Budget Formula** (from @sec-model-training-training-systems-fundamentals-05d2):

$$M_{train} = \underbrace{P \cdot b_w}_{\text{weights}} + \underbrace{P \cdot b_w}_{\text{gradients}} + \underbrace{2 \cdot P \cdot b_{opt}}_{\text{Adam } m_t, v_t} + \underbrace{B \cdot L \cdot s \cdot h \cdot b_w}_{\text{activations}}$$

Where:
- $P$ = number of parameters (e.g., $7 \times 10^9$)
- $b_w$ = bytes per value for weights/gradients: 4 (FP32), 2 (BF16/FP16), 1 (INT8)
- $b_{opt}$ = bytes per optimizer moment value: **always 4 (FP32)** for numerical stability
- $B$ = batch size, $L$ = number of layers, $s$ = sequence length, $h$ = hidden dimension

**For Adam in FP32** ($b_w = b_{opt} = 4$):

$$M_{static} = P \cdot 4 + P \cdot 4 + 2 \cdot P \cdot 4 = 16 \cdot P \text{ bytes}$$

At 7B parameters: $16 \times 7 \times 10^9 = 112 \text{ GB}$

**For Adam in BF16** (weights/grads in BF16, moments still FP32):

$$M_{static} = P \cdot 2 + P \cdot 2 + 2 \cdot P \cdot 4 = 12 \cdot P \text{ bytes}$$

At 7B parameters: $12 \times 7 \times 10^9 = 84 \text{ GB}$ — still exceeds H100's 80 GB for bare static state.

**The 4× training multiplier** cited in the chapter refers to Adam in FP32: training needs 4× the inference memory per parameter (weights only).
        """),
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — GRADIENT ACCUMULATION VS BATCH SIZE
# ─────────────────────────────────────────────────────────────────────────────

# ── CELL 13: ACT II INTRO ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["OrangeLine"]
    mo.vstack([
        mo.md("---"),
        mo.md("## Act II — Gradient Accumulation vs Batch Size"),
        mo.Html(f"""
        <div style="border-left: 4px solid {_color}; background: #fff7ed;
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; ML Engineer
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "We need an effective batch size of 256 for stable training, but the
                device only fits batch=8 in activation memory. Someone suggested
                gradient accumulation — apparently it gives us the same effective batch
                without the memory. Is that true? What's the throughput cost?"
            </div>
        </div>
        """),
        mo.md("""
        Gradient accumulation runs multiple forward-backward passes with a small
        micro-batch before performing a single optimizer step. The gradients from each
        micro-batch are summed (accumulated). After all accumulation steps complete,
        the optimizer updates the weights exactly once — as if it had seen the full
        effective batch all at once.

        Before we measure the throughput cost, predict what happens.
        """),
    ])
    return


# ── CELL 14: ACT II PREDICTION LOCK ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_prediction = mo.ui.radio(
        options={
            "A) Same throughput — accumulation is mathematically equivalent": "same",
            "B) 32× slower — each accumulation step serializes the compute": "32x_slower",
            "C) Slightly slower — small overhead per step from extra gradient bookkeeping": "slightly_slower",
            "D) Faster — smaller per-step working set fits better in cache": "faster",
        },
        label=(
            "Effective batch size = 256, micro-batch = 8, gradient accumulation steps = 32. "
            "Compared to training with a native batch of 256 directly, throughput using "
            "gradient accumulation is:"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background: #1e293b; border-radius: 12px; padding: 20px;
                    border-left: 4px solid #f97316; margin: 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #fdba74;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                Prediction Lock — Act II
            </div>
            <div style="color: #e2e8f0; font-size: 0.88rem;">
                Commit before adjusting the sliders below.
            </div>
        </div>
        """),
        act2_prediction,
    ])
    return (act2_prediction,)


# ── CELL 15: ACT II GATE ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction):
    mo.stop(
        act2_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Act II instruments."),
            kind="warn",
        ),
    )
    return


# ── CELL 16: ACT II CONTROLS ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    # Model size for act II (independent slider for act II exploration)
    act2_model_billions = mo.ui.slider(
        start=0.1, stop=70, value=7, step=0.1,
        label="Model size (billions of parameters)",
    )
    # Batch size (micro-batch): what actually loads into GPU memory
    act2_batch_size = mo.ui.slider(
        start=1, stop=512, value=8, step=1,
        label="Micro-batch size (per accumulation step)",
    )
    # Gradient accumulation steps
    act2_accum_steps = mo.ui.slider(
        start=1, stop=64, value=32, step=1,
        label="Gradient accumulation steps",
    )
    # Precision
    act2_precision = mo.ui.dropdown(
        options={
            "FP32 (4 bytes/value)": "fp32",
            "BF16 (2 bytes/value)": "bf16",
            "FP16 (2 bytes/value)": "fp16",
        },
        value="BF16 (2 bytes/value)",
        label="Training precision",
    )
    # Gradient checkpointing toggle
    act2_checkpointing = mo.ui.checkbox(
        value=False,
        label="Enable gradient checkpointing (recompute activations during backward pass)",
    )

    mo.vstack([
        mo.md("### Act II Simulator Controls"),
        mo.hstack([act2_model_billions, act2_precision], justify="start", gap="2rem"),
        mo.hstack([act2_batch_size, act2_accum_steps], justify="start", gap="2rem"),
        act2_checkpointing,
    ])
    return (
        act2_model_billions, act2_batch_size, act2_accum_steps,
        act2_precision, act2_checkpointing,
    )


# ── CELL 17: ACT II PHYSICS ENGINE ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go,
    act2_model_billions, act2_batch_size, act2_accum_steps,
    act2_precision, act2_checkpointing,
    context_toggle, COLORS, apply_plotly_theme,
    H100_RAM_GB, MOBILE_RAM_GB, DTYPE_BYTES, OPTIMIZER_EXTRA_MULTIPLIER,
):
    # ── Memory physics ────────────────────────────────────────────────────────
    _params_b = act2_model_billions.value * 1e9
    _dtype    = act2_precision.value
    _bpv      = DTYPE_BYTES[_dtype]
    _micro    = act2_batch_size.value
    _accum    = act2_accum_steps.value
    _ctx      = context_toggle.value
    _ckpt     = act2_checkpointing.value

    _effective_batch = _micro * _accum

    # Static memory (always Adam for this act, bf16 weights + fp32 moments)
    _weights_gb   = (_params_b * _bpv) / 1e9
    _grads_gb     = _weights_gb  # same dtype
    _optimizer_gb = (_params_b * 4 * 2) / 1e9  # Adam: 2 FP32 vectors

    # Activation memory with optional gradient checkpointing
    # Source: training.qmd footnote [fn-checkpointing-training]:
    #   "saving activations at only sqrt(L) strategic layers and recomputing the rest"
    #   "reducing activations by 4×" (line 4604 of plan's traceability table)
    _hidden_dim = max(512, int(act2_model_billions.value ** 0.5 * 1024))
    _n_layers   = max(12, int(act2_model_billions.value ** 0.33 * 12))
    _seq_len    = 1024

    if _ckpt:
        # Gradient checkpointing: O(sqrt(L)) layers stored instead of O(L)
        # This reduces activation memory by ~4× in practice (plan line 4604)
        _ckpt_layers = max(1, int(_n_layers ** 0.5))
        _act_bytes   = _micro * _seq_len * _hidden_dim * _ckpt_layers * _bpv
        _act_factor  = _ckpt_layers / _n_layers  # fraction of layers stored
    else:
        _act_bytes   = _micro * _seq_len * _hidden_dim * _n_layers * _bpv
        _act_factor  = 1.0

    _activations_gb = _act_bytes / 1e9
    _total_gb       = _weights_gb + _grads_gb + _optimizer_gb + _activations_gb
    _static_gb      = _weights_gb + _grads_gb + _optimizer_gb

    # Device budget
    _device_ram  = H100_RAM_GB if _ctx == "cloud" else MOBILE_RAM_GB
    _device_name = "H100 (80 GB)" if _ctx == "cloud" else "Mobile / Fine-tuning (8 GB)"
    _oom         = _total_gb > _device_ram

    # ── Throughput model ──────────────────────────────────────────────────────
    # Source: training.qmd §Iron Law of Training Performance
    #   Gradient accumulation overhead: each accumulation step adds a small
    #   gradient summation cost (~2–5% extra compute per step).
    #   Source: plan traceability — "Slightly slower due to optimizer overhead"
    _accum_overhead_per_step = 0.03  # 3% overhead per accumulation step
    _total_overhead          = 1.0 + _accum_overhead_per_step * _accum

    # Throughput: samples per second (normalized to native batch = 1.0)
    # Native large batch baseline: 1 optimizer step per _effective_batch samples
    # Gradient accumulation: same samples processed, but _total_overhead multiplier
    _throughput_native_norm = 1.0   # baseline = native batch size
    _throughput_accum_norm  = 1.0 / _total_overhead  # slightly lower

    # Gradient checkpointing compute overhead: +33% of compute
    # Source: training.qmd footnote [fn-checkpointing-training]:
    #   "trading roughly 33% additional compute"
    if _ckpt:
        _ckpt_compute_overhead = 1.33
    else:
        _ckpt_compute_overhead = 1.0

    _effective_throughput = _throughput_accum_norm / _ckpt_compute_overhead

    # ── Stacked memory bar chart ──────────────────────────────────────────────
    _bar_color = COLORS["RedLine"] if _oom else COLORS["BlueLine"]
    _bar_colors_act2 = [
        _bar_color,           # weights
        COLORS["GreenLine"],  # gradients
        COLORS["OrangeLine"], # optimizer
        "#6366f1",            # activations
    ]
    if _oom:
        _bar_colors_act2 = [COLORS["RedLine"]] * 4

    _fig2 = go.Figure()
    _comp_names = ["Weights", "Gradients", "Optimizer State", "Activations"]
    _comp_vals  = [_weights_gb, _grads_gb, _optimizer_gb, _activations_gb]

    for _name, _val, _col in zip(_comp_names, _comp_vals, _bar_colors_act2):
        _fig2.add_trace(go.Bar(
            name=_name,
            x=["Memory Breakdown"],
            y=[_val],
            marker_color=_col,
            text=f"{_val:.1f} GB",
            textposition="inside",
            textfont=dict(color="white", size=11, family="SF Mono, monospace"),
        ))

    _fig2.add_hline(
        y=_device_ram,
        line_color=COLORS["RedLine"],
        line_width=2.5,
        line_dash="dash",
        annotation_text=f"{_device_name} RAM",
        annotation_position="right",
        annotation_font_color=COLORS["RedLine"],
    )

    _fig2.update_layout(
        barmode="stack",
        height=360,
        yaxis_title="Memory (GB)",
        yaxis=dict(range=[0, max(_total_gb * 1.2, _device_ram * 1.15)]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=160, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    _fig2 = apply_plotly_theme(_fig2)

    # ── Metric cards ─────────────────────────────────────────────────────────
    _status_color = COLORS["RedLine"] if _oom else COLORS["GreenLine"]
    _ckpt_label   = f"ON ({int((1 - _act_factor) * 100):.0f}% activations freed)" if _ckpt else "OFF"

    _cards2 = f"""
<div style="display: flex; gap: 14px; flex-wrap: wrap; margin: 16px 0;">
    <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                min-width: 155px; text-align: center; background: white;">
        <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600;
                    text-transform: uppercase;">Effective Batch</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {COLORS['BlueLine']};
                    font-family: SF Mono, monospace;">{_effective_batch}</div>
        <div style="color: #94a3b8; font-size: 0.75rem;">micro={_micro} × steps={_accum}</div>
    </div>
    <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                min-width: 155px; text-align: center; background: white;">
        <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600;
                    text-transform: uppercase;">Activation Memory</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {COLORS['OrangeLine']};
                    font-family: SF Mono, monospace;">{_activations_gb:.1f} GB</div>
        <div style="color: #94a3b8; font-size: 0.75rem;">checkpointing: {_ckpt_label}</div>
    </div>
    <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                min-width: 155px; text-align: center; background: white;">
        <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600;
                    text-transform: uppercase;">Relative Throughput</div>
        <div style="font-size: 1.9rem; font-weight: 800;
                    color: {"#CC5500" if _effective_throughput < 0.9 else COLORS['GreenLine']};
                    font-family: SF Mono, monospace;">{_effective_throughput:.2f}×</div>
        <div style="color: #94a3b8; font-size: 0.75rem;">vs native large batch</div>
    </div>
    <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                min-width: 155px; text-align: center; background: white;">
        <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600;
                    text-transform: uppercase;">Status</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {_status_color};
                    font-family: SF Mono, monospace;">{"OOM" if _oom else "OK"}</div>
        <div style="color: #94a3b8; font-size: 0.75rem;">{_total_gb:.1f} / {_device_ram:.0f} GB</div>
    </div>
</div>
"""

    # ── Physics formula ───────────────────────────────────────────────────────
    _ckpt_note = (
        f"Gradient checkpointing: stores {_ckpt_layers} of {_n_layers} layers "
        f"({100 * _ckpt_layers // _n_layers}% of activation memory), "
        f"+33% compute overhead."
        if _ckpt else
        "Gradient checkpointing: OFF (all layer activations stored)."
    )

    _formula2_md = f"""
### Act II Physics

```
Effective batch size  = micro-batch ({_micro}) × accum steps ({_accum}) = {_effective_batch}
Activation memory     = {_micro} × {_seq_len} × {_hidden_dim} × {_n_layers if not _ckpt else _ckpt_layers} layers × {_bpv}B
                      = {_activations_gb:.2f} GB
{_ckpt_note}
Static memory floor   = {_static_gb:.1f} GB  (weights + grads + Adam)
Total required        = {_total_gb:.1f} GB
Accumulation overhead = {(_total_overhead - 1) * 100:.0f}%  ({_accum} steps × 3% per step)
Relative throughput   = 1 / ({_total_overhead:.2f} overhead × {_ckpt_compute_overhead:.2f} ckpt) = {_effective_throughput:.3f}×
```
"""

    # ── Failure state ─────────────────────────────────────────────────────────
    _items2 = [
        mo.md(_formula2_md),
        mo.Html(_cards2),
        mo.ui.plotly(_fig2),
    ]

    if _oom:
        _items2.append(mo.callout(
            mo.md(
                f"**OOM — Training requires {_total_gb:.1f} GB. "
                f"{_device_name} has {_device_ram:.0f} GB.** "
                f"Try: reduce micro-batch size, enable gradient checkpointing, "
                f"or switch to bf16 precision."
            ),
            kind="danger",
        ))
    else:
        _headroom = _device_ram - _total_gb
        _items2.append(mo.callout(
            mo.md(
                f"**Configuration fits.** {_total_gb:.1f} GB required, "
                f"{_device_ram:.0f} GB available — {_headroom:.1f} GB headroom. "
                f"Relative throughput: {_effective_throughput:.2f}× vs native large batch."
            ),
            kind="success",
        ))

    mo.vstack(_items2)
    return (
        _effective_batch, _total_gb, _effective_throughput,
        _activations_gb, _ckpt_compute_overhead, _oom,
    )


# ── CELL 18: ACT II PREDICTION REVEAL ────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction):
    # Source: training.qmd @tbl-iron-law-mapping:
    #   "Gradient Accumulation | Utilization ↑ | Maintains high batch parallelism efficiency"
    # The prediction-vs-reality: accumulation is slightly slower due to gradient
    # bookkeeping overhead per step, not 32× slower (compute is still serial-batch)
    # and not the same (there is real, small overhead).

    _labels = {
        "same":         "A) Same throughput",
        "32x_slower":   "B) 32× slower",
        "slightly_slower": "C) Slightly slower",
        "faster":       "D) Faster",
    }
    _predicted_label = _labels.get(act2_prediction.value, "—")
    _correct = act2_prediction.value == "slightly_slower"

    if _correct:
        mo.callout(mo.md(
            f"**Correct.** You predicted: {_predicted_label}. "
            f"Gradient accumulation runs the same total compute as a native batch — "
            f"every sample sees a forward and backward pass exactly once. "
            f"The cost difference comes only from the gradient summation bookkeeping "
            f"at each accumulation step: roughly 2–5% overhead per step. "
            f"With 32 steps this is 32 × 3% ≈ 96% overhead on the accumulation logic, "
            f"but that logic is tiny compared to the forward-backward pass itself, "
            f"making the total throughput hit roughly 3–8% below native batch."
        ), kind="success")
    elif act2_prediction.value == "same":
        mo.callout(mo.md(
            f"**Close, but not quite.** You predicted: {_predicted_label}. "
            f"Gradient accumulation is nearly equivalent, but each micro-batch step "
            f"incurs a small gradient summation and bookkeeping overhead. "
            f"Over 32 accumulation steps this accumulates to a real (small) throughput "
            f"penalty — typically 3–8% total. The chapter maps accumulation to "
            f"the Utilization ($\\eta$) term of the Iron Law: it keeps the GPU busy "
            f"but adds minor overhead per step."
        ), kind="warn")
    elif act2_prediction.value == "32x_slower":
        mo.callout(mo.md(
            f"**Not quite.** You predicted: {_predicted_label}. "
            f"Gradient accumulation does NOT serialize compute 32× — the GPU runs "
            f"a full forward-backward pass for each micro-batch just as efficiently "
            f"as it would for any small batch. The 32 steps run sequentially but each "
            f"is not blocked by the others beyond gradient summation. "
            f"Total throughput is nearly the same as the native large batch, "
            f"with only minor overhead from gradient bookkeeping."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**Not quite.** You predicted: {_predicted_label}. "
            f"Gradient accumulation does not improve cache behavior enough to be faster. "
            f"The smaller micro-batch does fit better in activation memory, "
            f"but the per-step gradient bookkeeping overhead is a real cost. "
            f"The net effect is slightly slower than native large batch."
        ), kind="warn")
    return


# ── CELL 19: ACT II REFLECTION ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) Speed for accuracy — checkpointing slows training to improve model quality": "wrong_accuracy",
            "B) Memory for compute — recomputes activations on backward pass instead of storing them": "correct",
            "C) Precision for speed — checkpointing uses lower precision to free memory": "wrong_precision",
            "D) Parameters for activations — smaller model needed when checkpointing is on": "wrong_params",
        },
        label="Gradient checkpointing trades what for what?",
    )
    mo.vstack([
        mo.md("### Reflection — Act II"),
        act2_reflection,
    ])
    return (act2_reflection,)


# ── CELL 20: ACT II REFLECTION FEEDBACK ──────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_reflection):
    mo.stop(
        act2_reflection.value is None,
        mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"),
    )

    _feedback2 = {
        "correct": (
            "**Correct.** Gradient checkpointing (activation checkpointing) discards "
            "the intermediate activations from the forward pass — activations that would "
            "normally be kept in GPU memory for use during the backward pass. "
            "Instead, it saves only activations at $\\sqrt{L}$ strategic checkpoint "
            "boundaries and recomputes the rest during the backward pass. "
            "The memory reduction is roughly $\\sqrt{L}$ — for GPT-2's 48 layers, "
            "storing 7 checkpoints instead of 48 full activation tensors. "
            "The compute cost is +33% (one extra forward pass worth of computation "
            "during backward). This is the classic memory-compute trade-off: "
            "you can always trade one for the other, but never get both for free."
        ),
        "wrong_accuracy": (
            "**Not quite.** Gradient checkpointing does not affect model quality "
            "or convergence — the gradients computed are mathematically identical "
            "to those computed with full activation storage. Only the memory "
            "footprint changes (smaller), at the cost of more compute (recomputation)."
        ),
        "wrong_precision": (
            "**Not quite.** Gradient checkpointing does not change the numerical "
            "precision of any stored tensor. It changes *which* tensors are stored "
            "at all — keeping fewer activations in memory and recomputing the rest "
            "on demand during the backward pass."
        ),
        "wrong_params": (
            "**Not quite.** Gradient checkpointing has no effect on the number of "
            "parameters in the model. The model size (weights + gradients + optimizer "
            "state) is unchanged. Only activation memory changes."
        ),
    }[act2_reflection.value]

    _kind2 = "success" if act2_reflection.value == "correct" else "warn"
    mo.callout(mo.md(_feedback2), kind=_kind2)
    return


# ── CELL 21: ACT II MATHPEEK ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — Gradient Accumulation and Checkpointing": mo.md(r"""
**Gradient Accumulation** (from @tbl-iron-law-mapping, `training.qmd`):

The effective batch size equals the micro-batch times accumulation steps:

$$B_{eff} = B_{micro} \times K_{accum}$$

Memory required scales only with $B_{micro}$ (the micro-batch), not $B_{eff}$:

$$M_{act} = B_{micro} \times s \times h \times L \times b_w$$

Throughput overhead (per-step gradient bookkeeping, $\epsilon \approx 0.03$):

$$\text{Throughput} \approx \frac{1}{1 + K_{accum} \cdot \epsilon} \times \text{Native throughput}$$

For $K_{accum} = 32$, $\epsilon = 0.03$: throughput $\approx \frac{1}{1.96} \approx 0.96 \times$ native — roughly 4% slower.

---

**Gradient Checkpointing** (from footnote `fn-checkpointing-training`, `training.qmd`):

Without checkpointing: all $L$ layers' activations are stored:

$$M_{act,full} = B \cdot s \cdot h \cdot L \cdot b_w \qquad [\text{linear in } L]$$

With checkpointing: only $\sqrt{L}$ checkpoint layers stored, rest recomputed:

$$M_{act,ckpt} = B \cdot s \cdot h \cdot \sqrt{L} \cdot b_w \qquad [\text{sublinear in } L]$$

Compute overhead: $+33\%$ of forward-pass FLOPs for recomputation during backward pass.

For GPT-2's 48 layers: $\sqrt{48} \approx 7$ — store 7 checkpoints instead of 48 activation tensors.
        """),
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTIONS + TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────

# ── CELL 22: CONNECTIONS ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("---"),
        mo.md("## Connections"),
        mo.callout(mo.md("""
        **Textbook:** This lab explores the memory accounting framework from
        @sec-model-training-training-systems-fundamentals-05d2 and the Iron Law of
        Training Performance from @sec-model-training-iron-law-training-performance-a53f.
        The gradient accumulation throughput model traces to the Iron Law's utilization
        term ($\\eta$): accumulation keeps the GPU busy but adds minor overhead per step.

        **TinyTorch:** In Module 08, you implement the optimizer state allocation and
        backward-pass gradient accumulation loop from scratch. When you see the Adam
        state tensors initialized at `model.parameters() × 2` you will recognize the
        memory cost computed in this lab.

        **Next Lab:** Lab 09 (Data Selection) examines how the *content* of the training
        data — not just the volume — affects how many optimizer steps are needed to
        converge, which multiplies the total training cost explored here.
        """), kind="info"),
    ])
    return


# ── CELL 23: KEY TAKEAWAYS ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## Key Takeaways"),
        mo.callout(mo.md("""
        **1. The 16× training multiplier.** A 7B model in FP32 with Adam requires
        16 bytes per parameter — not 4. Weights (4B) + gradients (4B) + Adam m_t (4B)
        + Adam v_t (4B) = 16 bytes × 7B params = 112 GB. OOM is not a surprise;
        it is a predictable consequence of this arithmetic. The H100's 80 GB is not
        a ceiling you hit by accident — you can calculate exactly when you will hit it
        before submitting the job.
        """), kind="warn"),
        mo.callout(mo.md("""
        **2. Gradient accumulation buys memory, not speed.** Accumulation gives you an
        effective large batch at the cost of micro-batch-sized activation memory.
        Throughput stays within 5–10% of native large-batch training. The technique
        belongs in the Iron Law's $\\eta$ column: it eliminates the activation OOM that
        would otherwise kill utilization entirely, at a small overhead cost per step.
        Gradient checkpointing adds another degree of freedom: trade +33% compute for
        $\\sqrt{L}$ activation memory — the right choice when activation memory is the
        binding constraint.
        """), kind="info"),
    ])
    return


# ── CELL 24: DESIGN LEDGER SAVE + HUD ────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle,
    act1_model_billions, act1_precision, act1_optimizer,
    act1_prediction, act1_reflection,
    act2_accum_steps, act2_checkpointing,
    act2_prediction, act2_reflection,
):
    # Save chapter results to Design Ledger
    # Fields consumed by: lab_10 (precision baseline), lab_11 (MFU starting point)
    _ctx = context_toggle.value

    _act1_correct = (act1_prediction.value == "112gb")
    _act1_refl_correct = (act1_reflection.value == "correct") if act1_reflection.value else False
    _act2_correct = (act2_prediction.value == "slightly_slower")
    _act2_refl_correct = (act2_reflection.value == "correct") if act2_reflection.value else False

    ledger.save(
        chapter=8,
        design={
            "context":            _ctx,
            "model_size_params":  act1_model_billions.value * 1e9,
            "precision_chosen":   act1_precision.value,
            "optimizer_chosen":   act1_optimizer.value,
            "oom_triggered":      act1_prediction.value != "112gb",   # proxy for OOM surprise
            "grad_accum_steps":   act2_accum_steps.value,
            "checkpointing_used": act2_checkpointing.value,
            "act1_prediction":    act1_prediction.value,
            "act1_correct":       _act1_correct,
            "act2_prediction":    act2_prediction.value,
            "act2_correct":       _act2_correct,
            "constraint_hit":     act1_prediction.value != "112gb",
        },
    )

    # HUD footer
    _p1_status  = "correct" if _act1_correct else ("pending" if act1_prediction.value is None else "incorrect")
    _p2_status  = "correct" if _act2_correct else ("pending" if act2_prediction.value is None else "incorrect")
    _r1_status  = "correct" if _act1_refl_correct else ("pending" if not act1_reflection.value else "incorrect")
    _r2_status  = "correct" if _act2_refl_correct else ("pending" if not act2_reflection.value else "incorrect")

    def _status_color(s):
        return {"correct": "#4ade80", "incorrect": "#f87171", "pending": "#94a3b8"}[s]

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB 08</span>
        <span style="color:#475569; margin: 0 4px;">|</span>
        <span class="hud-label">CTX:</span>
        <span class="hud-value">{_ctx}</span>
        <span style="color:#475569; margin: 0 4px;">|</span>
        <span class="hud-label">ACT I PRED:</span>
        <span style="color: {_status_color(_p1_status)}; font-family: SF Mono, monospace;
                     font-size: 0.8rem;">{_p1_status}</span>
        <span style="color:#475569; margin: 0 4px;">|</span>
        <span class="hud-label">ACT I REFL:</span>
        <span style="color: {_status_color(_r1_status)}; font-family: SF Mono, monospace;
                     font-size: 0.8rem;">{_r1_status}</span>
        <span style="color:#475569; margin: 0 4px;">|</span>
        <span class="hud-label">ACT II PRED:</span>
        <span style="color: {_status_color(_p2_status)}; font-family: SF Mono, monospace;
                     font-size: 0.8rem;">{_p2_status}</span>
        <span style="color:#475569; margin: 0 4px;">|</span>
        <span class="hud-label">ACT II REFL:</span>
        <span style="color: {_status_color(_r2_status)}; font-family: SF Mono, monospace;
                     font-size: 0.8rem;">{_r2_status}</span>
        <span style="color:#475569; margin: 0 4px;">|</span>
        <span class="hud-label">LEDGER:</span>
        <span class="hud-active">ch08 saved</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
