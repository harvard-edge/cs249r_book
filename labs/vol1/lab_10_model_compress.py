import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 10: THE COMPRESSION FRONTIER
#
# Chapter: ML Optimizations — Model Compression (@sec-optimizations-model-compression)
# Core Invariant:
#   Model compression is NOT free. Every technique moves on a Pareto frontier
#   between model quality and resource savings. INT8 quantization gives ~4× memory
#   reduction with <1% accuracy drop; INT4 gives ~8× with 2–5% drop; unstructured
#   pruning is unpredictable on real hardware because dense kernels ignore zeros.
#
# 2 Contexts: Cloud (H100, 80 GB) vs Mobile (NPU, 8 GB)
#
# Act I — The Quantization Surprise (12–15 min)
#   Stakeholder: Mobile App Team Lead
#   Prediction: Is INT8 quantization "lossless"?
#   Instrument: Quantization impact table — memory, accuracy, latency, energy
#               across FP32 / FP16 / INT8 / INT4 / INT2 for selectable model.
#   Reveal: Prediction-vs-reality overlay showing actual accuracy drop.
#   Reflection: Why does INT8 preserve accuracy better than INT4?
#
# Act II — The Compression Trade-off Frontier (FIRST INTRODUCTION, 20–25 min)
#   Stakeholder: Platform Engineering Lead
#   Prediction: Best compression strategy for 3 mobile deployment tiers
#   Instrument: Compression Trade-off Frontier — Pareto scatter (size vs quality)
#               + tier budget dropdowns + metric cards.
#   Failure state: OOM danger callout when selection exceeds tier memory budget.
#   Reflection: Why is unstructured pruning hardware-inefficient?
#
# Design Ledger: chapter=10, context, compression_method, compression_ratio,
#   act1_prediction, act1_correct, act2_result, act2_decision,
#   constraint_hit, pareto_optimal
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ── CELL 0: SETUP (hide_code=False — leave visible for instructor inspection) ─
@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np
    from plotly.subplots import make_subplots

    # WASM bootstrap: install mlsysim from hosted wheel when running in browser
    if sys.platform == "emscripten":
        import micropip
        await micropip.install("https://mlsysbook.ai/labs/wheels/mlsysim-0.1.0-py3-none-any.whl")
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog

    # ── Hardware constants (LABS_SPEC.md / NVIDIA and Apple specs) ────────────
    H100_BW_GBS       = 3350   # GB/s — H100 SXM5 HBM3e bandwidth
    H100_TFLOPS_FP16  = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_TFLOPS_INT8  = 3958   # TOPS INT8 tensor core (2× FP16)
    H100_RAM_GB       = 80     # GB HBM3e capacity
    H100_TDP_W        = 700    # Watts TDP

    MOBILE_BW_GBS     = 68     # GB/s — Apple A17-class SoC
    MOBILE_TOPS_INT8  = 35     # TOPS INT8 NPU
    MOBILE_RAM_GB     = 8      # GB total unified memory
    MOBILE_TDP_W      = 5      # Watts sustained (thermal throttle ceiling)

    # ── Bytes per value for each numeric format ────────────────────────────────
    # Source: IEEE 754 / INT quantization: FP32=4B, FP16=2B, INT8=1B, INT4=0.5B, INT2=0.25B
    DTYPE_BYTES = {
        "fp32": 4.0,
        "fp16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
        "int2": 0.25,
    }

    # ── ResNet-50 reference parameters ────────────────────────────────────────
    # Source: @sec-optimizations-model-compression — canonical numbers
    RESNET50_PARAMS_M  = 25.6   # million parameters
    RESNET50_FP32_MB   = 98.0   # MB in FP32 (25.6M × 4B ≈ 102 MB, with overhead)
    RESNET50_TOP1_ACC  = 76.1   # % ImageNet top-1 (torchvision baseline)

    # ── MobileNetV3-Large reference ───────────────────────────────────────────
    MOBILENETV3_PARAMS_M = 5.4   # million parameters
    MOBILENETV3_FP32_MB  = 21.1  # MB in FP32
    MOBILENETV3_TOP1_ACC = 75.8  # % ImageNet top-1

    # ── ViT-Base/16 reference ─────────────────────────────────────────────────
    VITBASE_PARAMS_M   = 86.0   # million parameters
    VITBASE_FP32_MB    = 330.0  # MB in FP32
    VITBASE_TOP1_ACC   = 81.1   # % ImageNet top-1

    # ── LLaMA-3 8B reference ──────────────────────────────────────────────────
    LLAMA3_8B_PARAMS_B = 8.0    # billion parameters
    LLAMA3_8B_FP32_GB  = 32.0   # GB in FP32 (8B × 4B)
    LLAMA3_8B_PPL      = 6.14   # perplexity on WikiText-2 (FP32 baseline)

    ledger = DesignLedger()
    return (
        mo, go, np, math, make_subplots,
        ledger, COLORS, LAB_CSS, apply_plotly_theme,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_TFLOPS_INT8, H100_RAM_GB, H100_TDP_W,
        MOBILE_BW_GBS, MOBILE_TOPS_INT8, MOBILE_RAM_GB, MOBILE_TDP_W,
        DTYPE_BYTES,
        RESNET50_PARAMS_M, RESNET50_FP32_MB, RESNET50_TOP1_ACC,
        MOBILENETV3_PARAMS_M, MOBILENETV3_FP32_MB, MOBILENETV3_TOP1_ACC,
        VITBASE_PARAMS_M, VITBASE_FP32_MB, VITBASE_TOP1_ACC,
        LLAMA3_8B_PARAMS_B, LLAMA3_8B_FP32_GB, LLAMA3_8B_PPL,
    )


# ── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _mobile_color = COLORS["Mobile"]
    _cloud_color  = COLORS["Cloud"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 10
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Compression Frontier
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 700px; line-height: 1.65;">
                Every compression technique trades model quality for resource savings.
                INT8 quantization achieves 4&times; size reduction with under 1% accuracy
                drop. INT4 reaches 8&times; but costs 2&ndash;5%. Unstructured pruning
                often yields no speedup at all. The frontier is real, and you cannot
                move along it for free.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: The Quantization Surprise &middot; 12&ndash;15 min
                </span>
                <span style="background: rgba(204,85,0,0.15); color: #fdba74;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(204,85,0,0.25);">
                    Act II: The Compression Frontier &middot; 20&ndash;25 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35&ndash;40 min total
                </span>
            </div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px;">
                <span class="badge badge-info">First use: Compression Trade-off Frontier</span>
                <span class="badge badge-warn">Memory budget failure state active</span>
            </div>
        </div>
        """),
    ])
    return


# ── CELL 2: BRIEFING ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">

        <!-- LEARNING OBJECTIVES -->
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Predict the accuracy drop from INT8 quantization</strong> — determine whether reducing a 7B LLM from FP16 to INT8 costs 10%, 5%, or under 1% accuracy, and identify where the "quantization cliff" appears.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose why 50% unstructured pruning yields ~0% speedup on GPU</strong> — trace why irregular zero patterns in dense matrix multiplications are not exploited by standard hardware kernels.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a compression configuration that satisfies simultaneous memory and latency constraints</strong> — find the quantization + structured pruning combination that fits a 7B model in 8 GB while meeting a 50 ms per-token latency budget.</div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- PREREQUISITES + DURATION (side by side) -->
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Uniform quantization formula from @sec-optimizations-quantization &middot;
                    Pareto frontier concept introduced in Lab 09 &middot;
                    Structured vs unstructured sparsity from @sec-optimizations-pruning
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35&ndash;40 min</strong><br/>
                    Act I: ~12 min &middot; Act II: ~25 min
                </div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- CORE QUESTION -->
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                "INT8 quantization halves the bits per weight &mdash; so why does it cost
                under 1% accuracy, while removing half the weights through pruning can
                sometimes provide zero speedup on the same hardware?"
            </div>
        </div>
    </div>
    """)
    return


# ── CELL 3: READING ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-optimizations-quantization** — Uniform quantization: the `Q(x) = round(x/scale + zero_point)` formula, scale calibration, PTQ vs QAT, hardware native support for INT8 vs INT4.
    - **@sec-optimizations-pruning** — Unstructured vs structured pruning, the Lottery Ticket Hypothesis, why sparse weights do not automatically yield latency improvements on dense hardware.
    - **@sec-optimizations-model-compression** — The accuracy-size Pareto frontier, compression ratio definition, why INT8 is the practical sweet spot for most deployments.
    - **@sec-optimizations-knowledge-distillation** — Distillation as an alternative compression axis.
    """), kind="info")
    return


# ── CELL 4: CONTEXT TOGGLE ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Cloud (H100 — 80 GB HBM, 700 W)": "cloud",
            "Mobile (NPU — 8 GB, 5 W sustained)": "mobile",
        },
        value="Cloud (H100 — 80 GB HBM, 700 W)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md("**Select your deployment context.** Hardware constraints differ by more than 10× across these two environments."),
        context_toggle,
    ])
    return (context_toggle,)


# ── CELL 4: CONTEXT SPEC CARD ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS):
    _ctx = context_toggle.value
    if _ctx == "cloud":
        _accent = COLORS["Cloud"]
        _bg     = "#f0f4ff"
        _border = "#c7d2fe"
        _specs  = [
            ("Device",              "NVIDIA H100 SXM5"),
            ("HBM Capacity",        "80 GB"),
            ("Memory Bandwidth",    "3,350 GB/s"),
            ("FP16 Peak",           "989 TFLOPS"),
            ("INT8 Peak",           "3,958 TOPS (2x FP16)"),
            ("Power Budget",        "700 W TDP"),
            ("INT8 native support", "Yes — Tensor Cores"),
            ("INT4 native support", "Yes — Tensor Cores"),
        ]
    else:
        _accent = COLORS["Mobile"]
        _bg     = "#fff7ed"
        _border = "#fed7aa"
        _specs  = [
            ("Device",              "Mobile NPU (Apple A17-class)"),
            ("RAM Capacity",        "8 GB unified"),
            ("Memory Bandwidth",    "68 GB/s"),
            ("INT8 Peak",           "35 TOPS"),
            ("FP16 throughput",     "~0.5x INT8 (software emulation path)"),
            ("Power Budget",        "5 W sustained"),
            ("INT8 native support", "Yes — Neural Engine"),
            ("INT4 native support", "Partial — model-dependent"),
        ]

    _rows = "".join(
        f'<div style="display:flex; justify-content:space-between; padding:5px 0; '
        f'border-bottom:1px solid {_border}; font-size:0.85rem;">'
        f'<span style="color:#475569; font-weight:600;">{k}</span>'
        f'<span style="font-family:monospace; color:{_accent}; font-weight:700;">{v}</span>'
        f'</div>'
        for k, v in _specs
    )

    mo.Html(f"""
    <div style="background:{_bg}; border:1px solid {_border}; border-left:4px solid {_accent};
                border-radius:8px; padding:16px 20px; margin: 8px 0;">
        <div style="font-size:0.72rem; font-weight:700; color:{_accent}; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom:10px;">
            Active Context — Hardware Constraints
        </div>
        {_rows}
    </div>
    """)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── CELL 5: ACT1_BANNER ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "I"
    _act_color = COLORS["BlueLine"]
    _act_title = "The Quantization Surprise"
    _act_duration = "12 min"
    _act_why = (
        "You expect that halving the bits per weight halves the model\u2019s representational "
        "capacity, costing proportional accuracy. The accuracy-vs-bitwidth curve will show "
        "a flat \u201cFree Lunch Zone\u201d from FP32 through INT8, then a sudden cliff "
        "at 3\u20134 bits \u2014 not a gradual decline."
    )
    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {_act_color}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">{_act_num}</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Act {_act_num} &middot; {_act_duration}</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                {_act_title}
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                {_act_why}
            </div>
        </div>
        """),
    ])
    return


# ── CELL 6: ACT1_STAKEHOLDER ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["Mobile"]
    mo.vstack([
        mo.Html(f"""
        <div style="border-left:4px solid {_color}; background:#fff7ed;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Mobile App Team Lead
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "We have a ResNet-50 running at 98 MB in FP32. Our App Store limit
                is 25 MB for the on-device model. A colleague told me INT8 quantization
                is mathematically lossless — it just changes the number format, so
                accuracy is preserved. Is that true? Can we ship INT8 with zero quality
                regression?"
            </div>
        </div>
        """),
        mo.md("""
        The team lead has heard that quantization is "lossless." Before you run the
        instruments, commit to a prediction. The chapter established
        (@sec-optimizations-quantization) that uniform quantization introduces a
        rounding error bounded by half the step size. The question is how large that
        error is in practice on a real model.
        """),
    ])
    return


# ── CELL 6: ACT I PREDICTION LOCK ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) Yes — INT8 is mathematically equivalent to FP32 for inference": "A",
            "B) Under 0.1% accuracy drop — essentially lossless for practical purposes": "B",
            "C) 0.5 to 2% accuracy drop — practically acceptable but not zero": "C",
            "D) 5 to 10% accuracy drop — unacceptable for production use": "D",
        },
        label=(
            "Applying INT8 post-training quantization (PTQ) to ResNet-50 (FP32 baseline: "
            "76.1% ImageNet top-1). What accuracy change do you expect?"
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
                Commit before touching any controls. Your prediction will be
                compared to the actual result at the end of this act.
            </div>
        </div>
        """),
        act1_prediction,
    ])
    return (act1_prediction,)


# ── CELL 7: ACT I GATE ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the quantization instrument."),
            kind="warn",
        ),
    )
    return


# ── CELL 8: ACT I CONTROLS ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_model = mo.ui.dropdown(
        options={
            "ResNet-50 (25.6 M params, 98 MB FP32)":        "resnet50",
            "MobileNetV3-Large (5.4 M params, 21 MB FP32)": "mobilenetv3",
            "ViT-Base/16 (86 M params, 330 MB FP32)":       "vitbase",
        },
        value="ResNet-50 (25.6 M params, 98 MB FP32)",
        label="Model architecture",
    )
    act1_quant_scheme = mo.ui.dropdown(
        options={
            "PTQ — Post-Training Quantization (no retraining)": "ptq",
            "QAT — Quantization-Aware Training (fine-tuned)":   "qat",
        },
        value="PTQ — Post-Training Quantization (no retraining)",
        label="Quantization scheme",
    )
    act1_calib_size = mo.ui.slider(
        start=128, stop=4096, value=512, step=128,
        label="Calibration dataset size (PTQ only)",
    )
    mo.vstack([
        mo.md("### Quantization Impact Table — Controls"),
        mo.hstack([act1_model, act1_quant_scheme], justify="start", gap="2rem"),
        mo.hstack([act1_calib_size], justify="start"),
        mo.callout(mo.md(
            "**PTQ vs QAT:** Post-training quantization requires only a calibration "
            "dataset (no gradient computation). QAT fine-tunes with simulated quantization "
            "and recovers 0.2–0.5% additional accuracy, but requires GPU training time."
        ), kind="info"),
    ])
    return (act1_model, act1_quant_scheme, act1_calib_size)


# ── CELL 9: ACT I PHYSICS ENGINE + QUANTIZATION TABLE ─────────────────────────
@app.cell(hide_code=True)
def _(
    mo,
    act1_model, act1_quant_scheme, act1_calib_size,
    context_toggle,
    RESNET50_PARAMS_M, RESNET50_FP32_MB, RESNET50_TOP1_ACC,
    MOBILENETV3_PARAMS_M, MOBILENETV3_FP32_MB, MOBILENETV3_TOP1_ACC,
    VITBASE_PARAMS_M, VITBASE_FP32_MB, VITBASE_TOP1_ACC,
    H100_TDP_W, MOBILE_TDP_W,
    DTYPE_BYTES,
):
    # ── Model lookup ──────────────────────────────────────────────────────────
    _model_key = act1_model.value
    _scheme    = act1_quant_scheme.value
    _calib     = act1_calib_size.value
    _ctx       = context_toggle.value

    _MODEL_SPECS = {
        "resnet50":    {"params_m": RESNET50_PARAMS_M,    "fp32_mb": RESNET50_FP32_MB,    "base_acc": RESNET50_TOP1_ACC},
        "mobilenetv3": {"params_m": MOBILENETV3_PARAMS_M, "fp32_mb": MOBILENETV3_FP32_MB, "base_acc": MOBILENETV3_TOP1_ACC},
        "vitbase":     {"params_m": VITBASE_PARAMS_M,     "fp32_mb": VITBASE_FP32_MB,     "base_acc": VITBASE_TOP1_ACC},
    }
    _spec      = _MODEL_SPECS[_model_key]
    _base_acc  = _spec["base_acc"]
    _fp32_mb   = _spec["fp32_mb"]

    # ── Calibration quality multiplier ────────────────────────────────────────
    # Source: @sec-optimizations-quantization — larger calibration sets reduce
    # the range estimation error. Effect saturates beyond ~1024 samples.
    # Small calib (<256): adds ~0.2% extra accuracy penalty; large (>2048): minimal effect.
    _calib_penalty = 0.0
    if _scheme == "ptq":
        if _calib < 256:
            _calib_penalty = 0.25
        elif _calib < 512:
            _calib_penalty = 0.10
        else:
            _calib_penalty = 0.0

    # ── QAT recovery bonus ────────────────────────────────────────────────────
    # Source: @sec-optimizations-quantization — QAT trains with simulated quantization
    # noise, recovering 0.2–0.5% accuracy compared to PTQ at the same bit-width.
    _qat_recovery = 0.35 if _scheme == "qat" else 0.0

    # ── Accuracy drop model ───────────────────────────────────────────────────
    # Source: @sec-optimizations-model-compression empirical figures:
    #   FP32 -> FP16: <0.05% (rounding only, 8-bit exponent preserved)
    #   FP16 -> INT8: 0.3–0.7% PTQ (linear range mapping loses outlier precision)
    #   INT8 -> INT4: 1.5–3.5% PTQ (4x more quantization bins lost)
    #   INT4 -> INT2: 4–8% PTQ (severe representational collapse)
    # ViT is more sensitive to quantization than ResNets due to attention softmax.
    _SENSITIVITY = {
        "resnet50":    1.0,
        "mobilenetv3": 0.85,
        "vitbase":     1.35,
    }
    _s = _SENSITIVITY[_model_key]

    _ACC_DROP_PTQ = {
        "fp32": 0.00,
        "fp16": 0.04 * _s,
        "int8": 0.50 * _s + _calib_penalty,
        "int4": 2.60 * _s + _calib_penalty * 1.5,
        "int2": 6.80 * _s + _calib_penalty * 2.0,
    }

    _ACC_DROP = {
        k: max(0.0, v - _qat_recovery * (v / (_ACC_DROP_PTQ["int4"] + 0.001)))
        for k, v in _ACC_DROP_PTQ.items()
    }

    # ── Memory size by format ──────────────────────────────────────────────────
    # Source: @sec-optimizations-quantization
    # compression ratio = FP32_bytes / target_bytes
    _MEM_MB = {
        fmt: _fp32_mb * (bpv / 4.0)
        for fmt, bpv in DTYPE_BYTES.items()
    }

    # ── Latency model ──────────────────────────────────────────────────────────
    # Source: @sec-optimizations-model-compression — memory-bandwidth bound inference.
    # Latency ≈ model_size_bytes / memory_bandwidth × 1000 (ms)
    # Cloud: H100 3350 GB/s, natively supports INT8/INT4 at 2× throughput.
    # Mobile: NPU 68 GB/s, FP16 uses software fallback path (~2× slower than INT8).
    _CLOUD_LATENCY_FACTOR = {
        "fp32": 1.0,
        "fp16": 0.50,
        "int8": 0.25,
        "int4": 0.15,
        "int2": 0.12,
    }
    _MOBILE_LATENCY_FACTOR = {
        "fp32": 4.0,
        "fp16": 1.80,
        "int8": 1.00,
        "int4": 0.65,
        "int2": 0.90,
    }

    # Absolute latency calibration (ms, batch=1)
    _CLOUD_FP32_BASE_MS  = 1.2  if _model_key == "resnet50" else (0.3  if _model_key == "mobilenetv3" else 4.8)
    _MOBILE_INT8_BASE_MS = 4.5  if _model_key == "resnet50" else (1.1  if _model_key == "mobilenetv3" else 18.0)

    if _ctx == "cloud":
        _LAT_BASE   = _CLOUD_FP32_BASE_MS
        _LAT_FACTOR = _CLOUD_LATENCY_FACTOR
    else:
        _LAT_BASE   = _MOBILE_INT8_BASE_MS / _MOBILE_LATENCY_FACTOR["int8"]
        _LAT_FACTOR = _MOBILE_LATENCY_FACTOR

    _POWER_W = H100_TDP_W if _ctx == "cloud" else MOBILE_TDP_W

    # ── Build per-format rows ─────────────────────────────────────────────────
    _formats    = ["fp32", "fp16", "int8", "int4", "int2"]
    _fmt_labels = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8", "int4": "INT4", "int2": "INT2"}

    _rows = []
    for _fmt in _formats:
        _acc_val   = _base_acc - _ACC_DROP[_fmt]
        _drop_val  = _ACC_DROP[_fmt]
        _mem_val   = _MEM_MB[_fmt]
        _lat_val   = _LAT_BASE * _LAT_FACTOR[_fmt]
        _energy_mj = _lat_val * _POWER_W / 1000.0
        _cr        = _fp32_mb / _mem_val

        if _drop_val < 0.15:
            _acc_color = "#008F45"
        elif _drop_val < 1.0:
            _acc_color = "#CC5500"
        else:
            _acc_color = "#CB202D"

        _rows.append({
            "fmt": _fmt,
            "label": _fmt_labels[_fmt],
            "accuracy": _acc_val,
            "drop": _drop_val,
            "acc_color": _acc_color,
            "mem_mb": _mem_val,
            "cr": _cr,
            "lat_ms": _lat_val,
            "energy_mj": _energy_mj,
        })

    # ── HTML table ────────────────────────────────────────────────────────────
    _HEADER_STYLE = (
        "background:#1e293b; color:#94a3b8; font-size:0.72rem; font-weight:700; "
        "text-transform:uppercase; letter-spacing:0.08em; padding:8px 12px; "
        "text-align:right; white-space:nowrap;"
    )
    _CELL_STYLE = "padding:8px 12px; text-align:right; font-size:0.88rem; font-family:monospace;"

    _table_rows_html = ""
    for _r in _rows:
        _is_int8 = _r["fmt"] == "int8"
        _bg_row  = "background:#f0fdf4;" if _is_int8 else ""
        _drop_color = "#CB202D" if _r["drop"] > 1.5 else ("#CC5500" if _r["drop"] > 0.3 else "#008F45")
        _table_rows_html += (
            f'<tr style="{_bg_row}border-bottom:1px solid #e2e8f0;">'
            f'<td style="padding:8px 12px; font-weight:800; font-size:0.88rem; color:#0f172a;">'
            f'{_r["label"]}</td>'
            f'<td style="{_CELL_STYLE} color:{_r["acc_color"]}; font-weight:700;">'
            f'{_r["accuracy"]:.2f}%</td>'
            f'<td style="{_CELL_STYLE} color:{_drop_color}; font-weight:700;">'
            f'-{_r["drop"]:.2f}%</td>'
            f'<td style="{_CELL_STYLE} color:#006395; font-weight:700;">'
            f'{_r["mem_mb"]:.1f} MB</td>'
            f'<td style="{_CELL_STYLE} color:#475569;">'
            f'{_r["cr"]:.1f}x</td>'
            f'<td style="{_CELL_STYLE} color:#475569;">'
            f'{_r["lat_ms"]:.2f} ms</td>'
            f'<td style="{_CELL_STYLE} color:#475569;">'
            f'{_r["energy_mj"]:.3f} mJ</td>'
            f'</tr>'
        )

    _ctx_label    = "Cloud (H100)" if _ctx == "cloud" else "Mobile (NPU)"
    _scheme_label = "PTQ" if _scheme == "ptq" else "QAT"
    _calib_note   = f" &middot; Calib: {_calib} samples" if _scheme == "ptq" else ""

    mo.Html(f"""
    <div style="margin: 16px 0;">
        <div style="font-size:0.72rem; font-weight:700; color:#475569;
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
            Quantization Impact Table &mdash; {_model_key.upper()}
            &middot; {_scheme_label} &middot; {_ctx_label}{_calib_note}
        </div>
        <div style="overflow-x:auto; border-radius:12px; border:1px solid #e2e8f0;
                    box-shadow:0 2px 8px rgba(0,0,0,0.04);">
        <table style="width:100%; border-collapse:collapse; min-width:600px;">
            <thead>
                <tr>
                    <th style="{_HEADER_STYLE} text-align:left;">Format</th>
                    <th style="{_HEADER_STYLE}">Accuracy (Top-1)</th>
                    <th style="{_HEADER_STYLE}">Accuracy Drop</th>
                    <th style="{_HEADER_STYLE}">Model Size</th>
                    <th style="{_HEADER_STYLE}">Compression</th>
                    <th style="{_HEADER_STYLE}">Inference Latency</th>
                    <th style="{_HEADER_STYLE}">Energy / Inference</th>
                </tr>
            </thead>
            <tbody>
                {_table_rows_html}
            </tbody>
        </table>
        </div>
        <div style="margin-top:8px; font-size:0.78rem; color:#94a3b8; line-height:1.5;">
            INT8 highlighted — practical sweet spot: 4x compression,
            under 1% accuracy penalty on PTQ with adequate calibration.
        </div>
    </div>
    """)
    return


# ── CELL 10: ACT I ACCURACY/SIZE CHART ────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, make_subplots, COLORS,
    act1_model, act1_quant_scheme, act1_calib_size,
    RESNET50_FP32_MB, RESNET50_TOP1_ACC,
    MOBILENETV3_FP32_MB, MOBILENETV3_TOP1_ACC,
    VITBASE_FP32_MB, VITBASE_TOP1_ACC,
    DTYPE_BYTES,
):
    # ── Replicate accuracy-drop model (same physics as cell 9) ───────────────
    _model_key = act1_model.value
    _scheme    = act1_quant_scheme.value
    _calib     = act1_calib_size.value

    _MODEL_LOOKUP = {
        "resnet50":    {"fp32_mb": RESNET50_FP32_MB,    "base_acc": RESNET50_TOP1_ACC},
        "mobilenetv3": {"fp32_mb": MOBILENETV3_FP32_MB, "base_acc": MOBILENETV3_TOP1_ACC},
        "vitbase":     {"fp32_mb": VITBASE_FP32_MB,     "base_acc": VITBASE_TOP1_ACC},
    }
    _spec2    = _MODEL_LOOKUP[_model_key]
    _base_acc = _spec2["base_acc"]
    _fp32_mb  = _spec2["fp32_mb"]

    _calib_penalty = 0.25 if (_scheme == "ptq" and _calib < 256) else (0.10 if (_scheme == "ptq" and _calib < 512) else 0.0)
    _qat_recovery  = 0.35 if _scheme == "qat" else 0.0
    _SENS2         = {"resnet50": 1.0, "mobilenetv3": 0.85, "vitbase": 1.35}
    _s2            = _SENS2[_model_key]

    _ACC_DROP2_PTQ = {
        "fp32": 0.00,
        "fp16": 0.04 * _s2,
        "int8": 0.50 * _s2 + _calib_penalty,
        "int4": 2.60 * _s2 + _calib_penalty * 1.5,
        "int2": 6.80 * _s2 + _calib_penalty * 2.0,
    }
    _ACC_DROP2 = {
        k: max(0.0, v - _qat_recovery * (v / (_ACC_DROP2_PTQ["int4"] + 0.001)))
        for k, v in _ACC_DROP2_PTQ.items()
    }

    _formats    = ["fp32", "fp16", "int8", "int4", "int2"]
    _fmt_labels = ["FP32", "FP16", "INT8", "INT4", "INT2"]
    _acc_vals   = [_base_acc - _ACC_DROP2[f] for f in _formats]
    _mem_mb     = [_fp32_mb * (DTYPE_BYTES[f] / 4.0) for f in _formats]
    _bar_colors = [
        COLORS["BlueLine"] if _ACC_DROP2[f] < 0.15
        else COLORS["OrangeLine"] if _ACC_DROP2[f] < 1.0
        else COLORS["RedLine"]
        for f in _formats
    ]

    _fig_a1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy by Format (Top-1 %)", "Model Size by Format (MB)"),
        horizontal_spacing=0.12,
    )

    _fig_a1.add_trace(
        go.Bar(
            name="Accuracy", x=_fmt_labels, y=_acc_vals,
            marker_color=_bar_colors,
            text=[f"{v:.2f}%" for v in _acc_vals],
            textposition="outside",
            textfont=dict(size=11, family="SF Mono, monospace"),
        ),
        row=1, col=1,
    )
    _fig_a1.add_hline(
        y=_base_acc, row=1, col=1,
        line_color=COLORS["GreenLine"], line_dash="dash", line_width=1.5,
        annotation_text="FP32 baseline",
        annotation_font_color=COLORS["GreenLine"],
        annotation_position="right",
    )
    _fig_a1.add_trace(
        go.Bar(
            name="Size (MB)", x=_fmt_labels, y=_mem_mb,
            marker_color=[COLORS["BlueLine"]] * len(_formats),
            text=[f"{v:.1f} MB" for v in _mem_mb],
            textposition="outside",
            textfont=dict(size=11, family="SF Mono, monospace"),
            showlegend=False,
        ),
        row=1, col=2,
    )
    _fig_a1.add_hline(
        y=25, row=1, col=2,
        line_color=COLORS["OrangeLine"], line_dash="dot", line_width=2,
        annotation_text="25 MB App Store target",
        annotation_font_color=COLORS["OrangeLine"],
        annotation_position="right",
    )
    _fig_a1.update_layout(
        height=380, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter, sans-serif",
        margin=dict(l=40, r=140, t=50, b=40),
    )
    _fig_a1.update_yaxes(gridcolor="#f1f5f9", row=1, col=1)
    _fig_a1.update_yaxes(gridcolor="#f1f5f9", row=1, col=2)
    _fig_a1.update_xaxes(linecolor=COLORS["Border"])

    mo.vstack([
        mo.md("### Accuracy and Size Trade-off by Format"),
        mo.plotly(_fig_a1),
    ])
    return


# ── CELL 11: ACT I MATHPEEK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — Uniform Quantization": mo.md("""
        **Uniform Quantization Formula** (@sec-optimizations-quantization):

        ```
        Q(x) = round(x / scale + zero_point)
        ```

        **Scale calculation (symmetric, per-tensor):**

        ```
        scale = max(|x|) / (2^(bits-1) - 1)
        ```

        For INT8 (bits=8): `scale = max(|x|) / 127`

        **Quantization error bound:**

        ```
        |epsilon_Q| <= scale / 2 = max(|x|) / (2 x (2^(bits-1) - 1))
        ```

        **Why INT8 outperforms INT4:**

        - INT8: 256 discrete levels — max relative error = 0.39% of range
        - INT4: 16 discrete levels — max relative error = 6.25% of range
        - Ratio: 16x more quantization error at INT4 vs INT8

        **Compression ratio:**

        ```
        CR = FP32_size / target_size = 4 bytes / target_bytes_per_value
        ```

        INT8: CR = 4/1 = **4x** | INT4: CR = 4/0.5 = **8x** | INT2: CR = 4/0.25 = **16x**
        """),
    })
    return


# ── CELL 12: ACT I PREDICTION-VS-REALITY REVEAL ───────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, act1_prediction,
    act1_model, act1_quant_scheme, act1_calib_size,
    RESNET50_TOP1_ACC, MOBILENETV3_TOP1_ACC, VITBASE_TOP1_ACC,
):
    _model_key  = act1_model.value
    _scheme     = act1_quant_scheme.value
    _calib      = act1_calib_size.value

    _ACC_LOOKUP = {
        "resnet50":    RESNET50_TOP1_ACC,
        "mobilenetv3": MOBILENETV3_TOP1_ACC,
        "vitbase":     VITBASE_TOP1_ACC,
    }
    _base_acc      = _ACC_LOOKUP[_model_key]
    _calib_penalty = 0.25 if (_scheme == "ptq" and _calib < 256) else (0.10 if (_scheme == "ptq" and _calib < 512) else 0.0)
    _qat_recovery  = 0.35 if _scheme == "qat" else 0.0
    _SENS3         = {"resnet50": 1.0, "mobilenetv3": 0.85, "vitbase": 1.35}
    _s3            = _SENS3[_model_key]

    _int8_drop = max(0.0, 0.50 * _s3 + _calib_penalty - _qat_recovery * 0.35)

    _pred_val   = act1_prediction.value
    _PRED_BANDS = {
        "A": (0.0,  0.0),
        "B": (0.0,  0.1),
        "C": (0.5,  2.0),
        "D": (5.0, 10.0),
    }
    _lo, _hi = _PRED_BANDS[_pred_val]
    _correct  = _lo <= _int8_drop <= _hi

    _FEEDBACK = {
        "A": (
            f"**Not quite.** INT8 is not mathematically equivalent to FP32. "
            f"Quantization maps each floating-point weight to one of 256 discrete integer levels "
            f"using `Q(x) = round(x/scale + zero_point)`. Every rounding is a real error. "
            f"For {_model_key.upper()}, INT8 PTQ costs **{_int8_drop:.2f}% accuracy** — "
            f"small, but nonzero and measurable."
        ),
        "B": (
            f"**Close, but the data disagrees.** INT8 is not lossless — it introduces "
            f"rounding error bounded by `scale/2`. For ResNet-50, the INT8 drop is "
            f"**{_int8_drop:.2f}%**, which is above the 0.1% threshold. "
            f"QAT can bring it close to 0.1%, but standard PTQ will not."
        ),
        "C": (
            f"**Correct.** INT8 PTQ introduces a measurable but practically acceptable "
            f"accuracy penalty. For {_model_key.upper()}, the actual drop is "
            f"**{_int8_drop:.2f}%**. This falls squarely in the 0.5–2% range for PTQ with "
            f"adequate calibration. The team lead was wrong that INT8 is lossless, "
            f"but right that it is usable in production."
        ),
        "D": (
            f"**Not quite.** A 5–10% drop would make INT8 unusable, but that level of "
            f"degradation is characteristic of INT2 or very aggressive INT4, not INT8. "
            f"For {_model_key.upper()}, INT8 PTQ costs only **{_int8_drop:.2f}%** accuracy — "
            f"enough to notice in A/B testing, but not enough to block deployment."
        ),
    }

    mo.callout(
        mo.md(_FEEDBACK[_pred_val]),
        kind="success" if _correct else "warn",
    )
    return


# ── CELL 13: ACT I REFLECTION ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) INT8 has more representable values — less rounding error in the linear mapping": "A",
            "B) INT8 is only applied to weights, not activations, so errors cancel": "B",
            "C) INT4 always uses non-uniform quantization which amplifies error": "C",
            "D) INT8 activations are always exactly representable in hardware": "D",
        },
        label="Reflection: Why does INT8 preserve accuracy better than INT4 in uniform quantization?",
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Reflection — Act I"),
        act1_reflection,
    ])
    return (act1_reflection,)


# ── CELL 14: ACT I REFLECTION FEEDBACK ───────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"),
    )

    _REFL1 = {
        "A": (
            "**Correct.** INT8 has 256 representable levels; INT4 has only 16. "
            "The quantization scale factor `scale = max(|x|) / (2^(bits-1) - 1)` "
            "is 16x larger for INT4 than INT8 when the value range is the same. "
            "Each rounding error is up to 16x larger, and those errors accumulate "
            "through layers. The accuracy gap between INT8 and INT4 is fundamentally "
            "an information-capacity gap.",
            True,
        ),
        "B": (
            "**Not correct.** Modern quantization (PTQ and QAT) applies to both "
            "weights and activations. Quantizing only weights would reduce memory "
            "footprint but leave inference arithmetic in FP32, missing the full "
            "latency benefit. The accuracy penalty comes from both domains.",
            False,
        ),
        "C": (
            "**Not correct.** Standard INT4 quantization uses uniform mapping, just "
            "like INT8. Non-uniform quantization (e.g., NF4 used in QLoRA) actually "
            "improves accuracy by placing more bins near zero where values cluster. "
            "The INT4 accuracy penalty is a direct consequence of having 16 levels "
            "vs 256 — not a property of uniform vs non-uniform mapping.",
            False,
        ),
        "D": (
            "**Not correct.** INT8 activations are not exactly representable — "
            "they are the output of the same rounding that weights undergo. "
            "An activation of 0.732 mapped to scale=0.006 becomes round(0.732/0.006) = 122, "
            "which dequantizes to 0.732 plus-or-minus 0.003. The error exists; it is "
            "bounded by `scale/2`.",
            False,
        ),
    }

    _text, _correct = _REFL1[act1_reflection.value]
    mo.callout(mo.md(_text), kind="success" if _correct else "warn")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ── CELL 12: ACT2_BANNER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "II"
    _act_color = COLORS["OrangeLine"]
    _act_title = "The Compression Trade-off Frontier"
    _act_duration = "25 min"
    _act_why = (
        "Act I revealed the Free Lunch Zone. Now discover the multi-dimensional design space: "
        "quantization provides linear speedup on bandwidth-bound inference, while 50% "
        "unstructured pruning provides exactly zero speedup on standard GPU kernels "
        "\u2014 and you must navigate both to deploy a 7B model across three memory tiers."
    )
    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {_act_color}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">{_act_num}</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Act {_act_num} &middot; {_act_duration} &middot; First introduction: Compression Trade-off Frontier</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                {_act_title}
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                {_act_why}
            </div>
        </div>
        """),
    ])
    return


# ── CELL 15: ACT2_STAKEHOLDER ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["Cloud"]
    mo.vstack([
        mo.Html(f"""
        <div style="border-left:4px solid {_color}; background:#f0f4ff;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Platform Engineering Lead
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "We are deploying LLaMA-3 8B as an on-device model across three mobile
                tiers in our user base: Flagship (8 GB RAM), Mid-range (4 GB), and
                Budget (2 GB). I need a different compression strategy for each tier.
                FP32 is 32 GB — none of them can fit that. Design the compression stack
                that keeps each tier as close to FP32 quality as possible within its
                memory budget."
            </div>
        </div>
        """),
        mo.Html("""
        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; margin:16px 0;">
            <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-top:4px solid #008F45;
                        border-radius:8px; padding:14px;">
                <div style="font-weight:800; color:#14532d; font-size:0.9rem; margin-bottom:4px;">
                    Flagship Tier
                </div>
                <div style="font-family:monospace; font-size:1.4rem; font-weight:900; color:#008F45;">
                    8 GB
                </div>
                <div style="font-size:0.8rem; color:#166534; margin-top:4px;">
                    Available RAM budget
                </div>
            </div>
            <div style="background:#fff7ed; border:1px solid #fed7aa; border-top:4px solid #CC5500;
                        border-radius:8px; padding:14px;">
                <div style="font-weight:800; color:#9a3412; font-size:0.9rem; margin-bottom:4px;">
                    Mid-range Tier
                </div>
                <div style="font-family:monospace; font-size:1.4rem; font-weight:900; color:#CC5500;">
                    4 GB
                </div>
                <div style="font-size:0.8rem; color:#7c2d12; margin-top:4px;">
                    Available RAM budget
                </div>
            </div>
            <div style="background:#fef2f2; border:1px solid #fecaca; border-top:4px solid #CB202D;
                        border-radius:8px; padding:14px;">
                <div style="font-weight:800; color:#991b1b; font-size:0.9rem; margin-bottom:4px;">
                    Budget Tier
                </div>
                <div style="font-family:monospace; font-size:1.4rem; font-weight:900; color:#CB202D;">
                    2 GB
                </div>
                <div style="font-size:0.8rem; color:#7f1d1d; margin-top:4px;">
                    Available RAM budget
                </div>
            </div>
        </div>
        """),
        mo.callout(mo.md("""
        **First introduction: Compression Trade-off Frontier** — This instrument
        (@sec-optimizations-model-compression) plots every compression configuration
        as a point in (model size, quality) space, then highlights the Pareto frontier:
        the set of configurations where you cannot improve quality without increasing size,
        or reduce size without hurting quality. Your goal is to select the Pareto-optimal
        configuration for each deployment tier.
        """), kind="info"),
    ])
    return


# ── CELL 16: ACT II PREDICTION LOCK ───────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_prediction = mo.ui.radio(
        options={
            "A) Use INT8 for all three tiers — INT8 is the universal safe choice": "A",
            "B) Flagship: INT8, Mid-range: INT4, Budget: INT4 + 50% structured pruning": "B",
            "C) 50% unstructured pruning for all tiers — pruning is always better than quantization": "C",
            "D) Distill a separate small model for budget tier — quantization never works below 4 bits": "D",
        },
        label=(
            "LLaMA-3 8B, FP32 = 32 GB. Design the compression strategy for Flagship (8 GB), "
            "Mid-range (4 GB), and Budget (2 GB) tiers that maximizes quality within each budget."
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background: #1e293b; border-radius: 12px; padding: 20px;
                    border-left: 4px solid #6366f1; margin: 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #a5b4fc;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                Prediction Lock — Act II
            </div>
            <div style="color: #e2e8f0; font-size: 0.88rem; margin-bottom: 12px;">
                Commit your strategy prediction before exploring the Frontier.
            </div>
        </div>
        """),
        act2_prediction,
    ])
    return (act2_prediction,)


# ── CELL 17: ACT II GATE ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction):
    mo.stop(
        act2_prediction.value is None,
        mo.callout(
            mo.md("Select your strategy prediction above to unlock the Compression Frontier."),
            kind="warn",
        ),
    )
    return


# ── CELL 18: ACT II CONTROLS ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    _COMPRESS_OPTIONS = {
        "FP32 (32 GB — 1x compression)":                     "fp32",
        "FP16 (16 GB — 2x compression)":                     "fp16",
        "INT8 (8 GB — 4x compression)":                      "int8",
        "INT4 (4 GB — 8x compression)":                      "int4",
        "INT4 + 10% structured pruning (~3.6 GB)":           "int4_prune10",
        "INT4 + 30% structured pruning (~2.8 GB)":           "int4_prune30",
        "INT4 + 50% structured pruning (~2.0 GB)":           "int4_prune50",
        "INT4 + 50% unstructured pruning (~2.0 GB)":         "int4_prune50_unstruct",
        "Distilled 4B model, INT8 (~4 GB)":                  "distil_4b_int8",
        "Distilled 1B model, INT8 (~1 GB)":                  "distil_1b_int8",
    }

    act2_flagship = mo.ui.dropdown(
        options=_COMPRESS_OPTIONS,
        value="INT8 (8 GB — 4x compression)",
        label="Flagship tier (8 GB budget)",
    )
    act2_midrange = mo.ui.dropdown(
        options=_COMPRESS_OPTIONS,
        value="INT4 (4 GB — 8x compression)",
        label="Mid-range tier (4 GB budget)",
    )
    act2_budget = mo.ui.dropdown(
        options=_COMPRESS_OPTIONS,
        value="INT4 + 50% structured pruning (~2.0 GB)",
        label="Budget tier (2 GB budget)",
    )

    mo.vstack([
        mo.md("### Assign a Compression Strategy to Each Deployment Tier"),
        mo.md("""
        Select a compression configuration for each tier. The Compression Trade-off
        Frontier below will update to show where each choice sits relative to the
        Pareto-optimal boundary.
        """),
        mo.hstack([act2_flagship, act2_midrange, act2_budget], justify="start", gap="1.5rem"),
    ])
    return (act2_flagship, act2_midrange, act2_budget)


# ── CELL 19: COMPRESSION FRONTIER PLOT + TIER METRICS ─────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, COLORS,
    act2_flagship, act2_midrange, act2_budget,
    LLAMA3_8B_PPL,
):
    # ── LLaMA-3 8B compression data ───────────────────────────────────────────
    # Source: @sec-optimizations-model-compression — empirical compression curves.
    # Perplexity (lower = better): FP32 baseline = 6.14 (WikiText-2).
    # Calibrated to published llama.cpp / GGUF / bitsandbytes benchmarks.
    #
    # Tuple structure: (label, method_key, size_gb, perplexity, pareto_flag)
    _CONFIGS = [
        ("FP32 (baseline)",              "fp32",                 32.0, 6.14, True ),
        ("FP16",                         "fp16",                 16.0, 6.16, True ),
        ("INT8",                         "int8",                  8.0, 6.21, True ),
        ("INT4",                         "int4",                  4.0, 6.47, True ),
        ("INT4 + 10% struct. pruning",   "int4_prune10",          3.6, 6.63, True ),
        ("INT4 + 30% struct. pruning",   "int4_prune30",          2.8, 7.12, True ),
        ("INT4 + 50% struct. pruning",   "int4_prune50",          2.0, 8.05, True ),
        ("INT4 + 50% unstruct. pruning", "int4_prune50_unstruct", 2.0, 7.85, False),
        ("Distilled 4B, INT8",           "distil_4b_int8",        4.1, 7.30, False),
        ("Distilled 1B, INT8",           "distil_1b_int8",        1.0, 9.80, False),
    ]

    _CONFIG_MAP = {c[1]: c for c in _CONFIGS}

    _TIER_BUDGETS = {"flagship": 8.0, "midrange": 4.0, "budget": 2.0}
    _TIER_KEYS    = {
        "flagship": act2_flagship.value,
        "midrange": act2_midrange.value,
        "budget":   act2_budget.value,
    }
    _TIER_COLORS = {
        "flagship": COLORS["GreenLine"],
        "midrange": COLORS["OrangeLine"],
        "budget":   COLORS["RedLine"],
    }
    _TIER_LABELS = {"flagship": "Flagship", "midrange": "Mid-range", "budget": "Budget"}

    _bg_x, _bg_y, _bg_text = [], [], []
    _pf_x, _pf_y, _pf_text = [], [], []

    for _label_c, _key_c, _sz_c, _ppl_c, _on_pareto_c in _CONFIGS:
        if _on_pareto_c:
            _pf_x.append(_sz_c)
            _pf_y.append(_ppl_c)
            _pf_text.append(_label_c)
        else:
            _bg_x.append(_sz_c)
            _bg_y.append(_ppl_c)
            _bg_text.append(_label_c)

    _pf_sorted = sorted(zip(_pf_x, _pf_y, _pf_text))
    _pf_x_s    = [p[0] for p in _pf_sorted]
    _pf_y_s    = [p[1] for p in _pf_sorted]

    _fig2 = go.Figure()

    _fig2.add_trace(go.Scatter(
        x=_bg_x, y=_bg_y,
        mode="markers",
        name="Dominated (off-frontier)",
        marker=dict(color="#94a3b8", size=10, symbol="circle-open", line=dict(width=2)),
        text=_bg_text,
        hovertemplate="<b>%{text}</b><br>Size: %{x:.1f} GB<br>Perplexity: %{y:.2f}<extra></extra>",
    ))

    _fig2.add_trace(go.Scatter(
        x=_pf_x_s, y=_pf_y_s,
        mode="lines",
        name="Pareto frontier",
        line=dict(color=COLORS["BlueLine"], width=2, dash="dot"),
        showlegend=True,
        hoverinfo="skip",
    ))

    _fig2.add_trace(go.Scatter(
        x=_pf_x, y=_pf_y,
        mode="markers",
        name="Pareto-optimal",
        marker=dict(color=COLORS["BlueLine"], size=11, symbol="circle",
                    line=dict(color="white", width=2)),
        text=_pf_text,
        hovertemplate="<b>%{text}</b><br>Size: %{x:.1f} GB<br>Perplexity: %{y:.2f}<extra></extra>",
    ))

    for _tier_n, _budget_gb in _TIER_BUDGETS.items():
        _fig2.add_vline(
            x=_budget_gb,
            line_color=_TIER_COLORS[_tier_n],
            line_width=1.5,
            line_dash="dash",
            annotation_text=f"{_TIER_LABELS[_tier_n]} ({_budget_gb:.0f} GB)",
            annotation_font_color=_TIER_COLORS[_tier_n],
            annotation_position="top",
        )

    for _tier_n2, _sel_key in _TIER_KEYS.items():
        if _sel_key in _CONFIG_MAP:
            _c2 = _CONFIG_MAP[_sel_key]
            _fig2.add_trace(go.Scatter(
                x=[_c2[2]], y=[_c2[3]],
                mode="markers+text",
                name=f"{_TIER_LABELS[_tier_n2]} selection",
                marker=dict(color=_TIER_COLORS[_tier_n2], size=18,
                            symbol="star", line=dict(color="white", width=2)),
                text=[_TIER_LABELS[_tier_n2]],
                textposition="top center",
                textfont=dict(size=11, color=_TIER_COLORS[_tier_n2]),
                hovertemplate=(
                    f"<b>{_TIER_LABELS[_tier_n2]}: {_c2[0]}</b>"
                    f"<br>Size: {_c2[2]:.1f} GB<br>Perplexity: {_c2[3]:.2f}"
                    f"<extra></extra>"
                ),
            ))

    _fig2.update_layout(
        xaxis=dict(
            title="Model Size (GB)",
            type="log",
            tickvals=[1, 2, 4, 8, 16, 32],
            ticktext=["1 GB", "2 GB", "4 GB", "8 GB", "16 GB", "32 GB"],
            gridcolor="#f1f5f9", linecolor=COLORS["Border"],
            range=[-0.05, 1.55],
        ),
        yaxis=dict(
            title="Perplexity on WikiText-2 (lower = better)",
            gridcolor="#f1f5f9", linecolor=COLORS["Border"],
            range=[5.8, 10.5],
        ),
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Inter, sans-serif",
        font_color=COLORS["Text"],
        margin=dict(l=60, r=40, t=30, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    # ── Per-tier metric cards ──────────────────────────────────────────────────
    _tier_cards_html = ""
    for _tier_n3, _sel_key3 in _TIER_KEYS.items():
        _budget3 = _TIER_BUDGETS[_tier_n3]
        _accent3 = _TIER_COLORS[_tier_n3]
        if _tier_n3 == "flagship":
            _bg3, _bd3 = "#f0fdf4", "#bbf7d0"
        elif _tier_n3 == "midrange":
            _bg3, _bd3 = "#fff7ed", "#fed7aa"
        else:
            _bg3, _bd3 = "#fef2f2", "#fecaca"

        if _sel_key3 in _CONFIG_MAP:
            _c3      = _CONFIG_MAP[_sel_key3]
            _sz3     = _c3[2]
            _ppl3    = _c3[3]
            _pareto3 = _c3[4]
            _fits3   = _sz3 <= _budget3
            _ppl_delta3 = _ppl3 - LLAMA3_8B_PPL
            _ppl_col3 = "#008F45" if _ppl_delta3 < 0.3 else ("#CC5500" if _ppl_delta3 < 1.5 else "#CB202D")
            _fit_str3  = f"{_sz3:.1f} GB (fits)" if _fits3 else f"{_sz3:.1f} GB — EXCEEDS {_budget3:.0f} GB"
            _fit_col3  = "#008F45" if _fits3 else "#CB202D"
            _pb_badge  = (
                '<span style="background:#f0fdf4; border:1px solid #bbf7d0; padding:1px 7px; '
                'border-radius:4px; font-weight:700; color:#008F45; font-size:0.72rem;">Pareto-optimal</span>'
                if _pareto3 else
                '<span style="background:#fef2f2; border:1px solid #fecaca; padding:1px 7px; '
                'border-radius:4px; font-weight:700; color:#CB202D; font-size:0.72rem;">Off-frontier</span>'
            )
        else:
            _sz3, _ppl3, _ppl_delta3 = 0.0, 0.0, 0.0
            _fit_str3, _fit_col3, _ppl_col3, _pb_badge = "Unknown", "#94a3b8", "#94a3b8", ""

        _tier_cards_html += (
            f'<div style="background:{_bg3}; border:1px solid {_bd3}; border-top:4px solid {_accent3};'
            f'border-radius:8px; padding:14px 16px; flex:1; min-width:180px;">'
            f'<div style="font-weight:800; color:{_accent3}; font-size:0.85rem; margin-bottom:8px;">'
            f'{_TIER_LABELS[_tier_n3]} ({_budget3:.0f} GB budget)</div>'
            f'<div style="font-size:0.82rem; line-height:1.9;">'
            f'<div><span style="color:#475569; font-weight:600;">Strategy:</span> '
            f'<span style="font-family:monospace; color:#0f172a;">{_sel_key3}</span></div>'
            f'<div><span style="color:#475569; font-weight:600;">Size:</span> '
            f'<span style="font-family:monospace; color:{_fit_col3}; font-weight:700;">{_fit_str3}</span></div>'
            f'<div><span style="color:#475569; font-weight:600;">Perplexity:</span> '
            f'<span style="font-family:monospace; color:{_ppl_col3}; font-weight:700;">'
            f'{_ppl3:.2f} (+{_ppl_delta3:.2f} vs FP32)</span></div>'
            f'<div style="margin-top:4px;">{_pb_badge}</div>'
            f'</div></div>'
        )

    mo.vstack([
        mo.md("### Compression Trade-off Frontier — LLaMA-3 8B"),
        mo.md("""
        Each point is a compression configuration. **Blue dots** lie on the Pareto
        frontier — where you cannot improve quality without increasing size.
        **Star markers** show your selections. Dashed vertical lines mark each
        tier's memory budget.
        """),
        mo.plotly(_fig2),
        mo.md("#### Per-Tier Metric Summary"),
        mo.Html(f'<div style="display:flex; gap:14px; flex-wrap:wrap; margin:12px 0;">{_tier_cards_html}</div>'),
    ])
    return


# ── CELL 20: FAILURE STATE (OOM DETECTION) ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_flagship, act2_midrange, act2_budget):
    _SIZE_MAP_OOM = {
        "fp32": 32.0, "fp16": 16.0, "int8": 8.0, "int4": 4.0,
        "int4_prune10": 3.6, "int4_prune30": 2.8, "int4_prune50": 2.0,
        "int4_prune50_unstruct": 2.0, "distil_4b_int8": 4.1, "distil_1b_int8": 1.0,
    }
    _BUDGETS_OOM  = {"flagship": 8.0, "midrange": 4.0, "budget": 2.0}
    _TIERS_OOM    = {
        "flagship": act2_flagship.value,
        "midrange": act2_midrange.value,
        "budget":   act2_budget.value,
    }
    _TIER_LABELS_OOM = {"flagship": "Flagship", "midrange": "Mid-range", "budget": "Budget"}

    _violations = []
    for _tier_oom, _key_oom in _TIERS_OOM.items():
        _sz_oom = _SIZE_MAP_OOM.get(_key_oom, 0.0)
        if _sz_oom > _BUDGETS_OOM[_tier_oom]:
            _violations.append((_tier_oom, _sz_oom, _BUDGETS_OOM[_tier_oom]))

    _widgets_oom = []
    for _tier_v, _req_v, _avail_v in _violations:
        _widgets_oom.append(
            mo.callout(
                mo.md(
                    f"**OOM — Infeasible for {_TIER_LABELS_OOM[_tier_v]} tier.** "
                    f"Required: **{_req_v:.1f} GB** | Available: **{_avail_v:.0f} GB** | "
                    f"Overflow: **{_req_v - _avail_v:.1f} GB over budget.** "
                    f"Select a more aggressive compression scheme for this tier."
                ),
                kind="danger",
            )
        )

    if _widgets_oom:
        mo.vstack(_widgets_oom)
    else:
        mo.callout(
            mo.md(
                "**All tiers within budget.** "
                "Every selected configuration fits within its deployment tier memory limit."
            ),
            kind="success",
        )
    return


# ── CELL 21: ACT II PREDICTION FEEDBACK ───────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction):
    _pred2 = act2_prediction.value
    _FEEDBACK_A2 = {
        "A": (
            "**Not quite.** INT8 gives an 8 GB footprint for LLaMA-3 8B — exactly the "
            "Flagship budget. But the Mid-range tier has only 4 GB and the Budget tier "
            "only 2 GB. A single INT8 strategy violates both smaller tiers. You need a "
            "tiered approach that applies progressively stronger compression as the budget "
            "shrinks.",
            False,
        ),
        "B": (
            "**Correct.** This is the Pareto-optimal tiered allocation. "
            "Flagship at INT8 uses the full 8 GB budget with minimal accuracy loss. "
            "Mid-range at INT4 hits exactly the 4 GB constraint. "
            "Budget at INT4 + 50% structured pruning reaches the 2 GB ceiling while "
            "remaining on the Pareto frontier — structured pruning removes entire "
            "attention heads and MLP blocks, so the compressed model still runs "
            "efficiently on dense hardware.",
            True,
        ),
        "C": (
            "**Not correct.** Unstructured pruning sets individual weights to zero but "
            "leaves the tensor dimensions unchanged. Dense matrix kernels on NPUs execute "
            "the same number of MAC operations regardless of how many are zero — the "
            "hardware does not skip zeros. The result is a compressed file but not a "
            "faster computation. Structured pruning removes entire rows/columns, which "
            "genuinely reduces the arithmetic and fits the model into a smaller budget.",
            False,
        ),
        "D": (
            "**Not correct.** Knowledge distillation is a valid axis, but it is not required "
            "here. INT4 + structured pruning reaches 2 GB for LLaMA-3 8B while remaining on "
            "the Pareto frontier. Distilling a separate 1B model produces a fundamentally "
            "different model with different capabilities — appropriate only when quality "
            "degradation from pruning is unacceptable.",
            False,
        ),
    }

    _text2, _correct2 = _FEEDBACK_A2[_pred2]
    mo.callout(mo.md(_text2), kind="success" if _correct2 else "warn")
    return


# ── CELL 22: ACT II MATHPEEK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — Compression Ratio and Pareto Frontier": mo.md("""
        **Compression ratio** (@sec-optimizations-model-compression):

        ```
        CR = Size_FP32 / Size_compressed
           = (N_params x 4 bytes) / (N_params x bytes_per_value x (1 - sparsity))
        ```

        For INT8 + 0% pruning: CR = 4 / 1.0 = **4x**

        For INT4 + 50% structured pruning: CR = 4 / (0.5 x 0.5) = **16x**

        **Quantization error bound (uniform per-tensor):**

        ```
        |epsilon_Q| <= max(|x|) / (2 x (2^bits - 1))
        ```

        INT8: max_err = range / 510 — approximately 0.2% of range

        INT4: max_err = range / 30 — approximately 3.3% of range

        **Structured vs unstructured pruning:**

        ```
        Structured:   removes complete rows/columns -> dense submatrix -> hardware efficient
        Unstructured: zeros individual elements -> sparse matrix -> dense kernel unchanged
        ```

        Structured pruning at sparsity `s` reduces MACs by exactly `s`:

        ```
        MACs_pruned = MACs_dense x (1 - sparsity)
        ```

        Unstructured pruning at sparsity `s` reduces latency ONLY when specialized
        sparse kernels are available (e.g., NVIDIA A100 2:4 sparsity). Without hardware
        support, latency is **unchanged** even at 90% sparsity.

        **Empirical accuracy-size tradeoff law:**

        ```
        delta_perplexity ≈ alpha x log2(CR)
        ```

        where alpha ≈ 0.12 for LLaMA-class models at moderate compression ratios.
        This log relationship explains why compression becomes increasingly costly
        as you push toward extreme ratios (INT2, very high sparsity).
        """),
    })
    return


# ── CELL 23: ACT II REFLECTION ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) Pruning removes too many weights, making the model too small to function": "A",
            "B) Sparse operations require special hardware support — dense kernels execute zero weights unchanged": "B",
            "C) Pruned models cannot subsequently be quantized": "C",
            "D) Unstructured pruning always hurts accuracy more than quantization": "D",
        },
        label="Reflection: Why is unstructured pruning often hardware-inefficient in practice?",
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Reflection — Act II"),
        act2_reflection,
    ])
    return (act2_reflection,)


# ── CELL 24: ACT II REFLECTION FEEDBACK ──────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_reflection):
    mo.stop(
        act2_reflection.value is None,
        mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"),
    )

    _REFL2 = {
        "A": (
            "**Not correct.** Pruning does not make a model too small to function — "
            "it creates sparse weights within the original tensor dimensions. The model "
            "architecture is unchanged; only individual weight values are forced to zero. "
            "The hardware-efficiency problem is not about model capacity; it is about "
            "whether the underlying arithmetic can exploit those zeros.",
            False,
        ),
        "B": (
            "**Correct.** Dense matrix multiply kernels (GEMM) on GPUs and NPUs are "
            "designed for dense inputs. A kernel computing `C = A x B` iterates over "
            "every element of A — including the zeros introduced by unstructured pruning. "
            "The operation count is identical to the unpruned case. Memory bandwidth "
            "savings require the weights to be stored sparsely (e.g., CSR format), but "
            "even that requires a sparse GEMM kernel. NVIDIA A100 supports 2:4 structured "
            "sparsity natively; arbitrary unstructured sparsity on mobile NPUs typically "
            "provides zero latency benefit.",
            True,
        ),
        "C": (
            "**Not correct.** Pruning and quantization are orthogonal techniques. "
            "A pruned model — whether structured or unstructured — can be quantized "
            "afterward. INT4 + structured pruning is a standard production combination "
            "precisely because each technique acts on a different aspect of the model "
            "(precision vs. architectural width).",
            False,
        ),
        "D": (
            "**Not correct.** The accuracy impact of unstructured pruning depends heavily "
            "on sparsity level and model type. At moderate sparsities (10–30%), unstructured "
            "pruning often hurts accuracy less than INT4 quantization. The problem is not "
            "accuracy — it is that you cannot exploit the sparsity for latency improvement "
            "without specialized sparse kernels. The hardware efficiency problem is "
            "independent of the accuracy impact.",
            False,
        ),
    }

    _text_r2, _correct_r2 = _REFL2[act2_reflection.value]
    mo.callout(mo.md(_text_r2), kind="success" if _correct_r2 else "warn")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ── CELL 20: SYNTHESIS ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.vstack([
        mo.md("---"),

        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. INT8 quantization is a free lunch: under 1% accuracy loss for 4&times; memory reduction.</strong>
                    Neural networks are overparameterized in numerical precision. INT8's 256 levels
                    are sufficient to represent the weight distribution of a trained model. The
                    "cliff" at 3&ndash;4 bits is sudden, not gradual &mdash; accuracy collapses
                    when discrete levels become insufficient to distinguish critical weight values.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Unstructured pruning at 50% provides ~0% speedup on standard GPUs.</strong>
                    Dense GEMM kernels iterate over every element including zeros. The sparsity
                    is invisible to the hardware. Only structured sparsity (e.g., NVIDIA 2:4)
                    provides real speedup because the hardware can skip known-zero elements
                    in a regular pattern.
                </div>
                <div>
                    <strong>3. Deploying a 7B model to 8 GB memory requires INT4 + structured pruning.</strong>
                    FP16 (14 GB) does not fit. INT8 (7 GB) fits but latency at 50 GB/s mobile
                    bandwidth is ~140 ms &mdash; over budget. INT4 + structured 2:4 pruning
                    reaches ~1.75 GB and ~35 ms per token, satisfying both constraints.
                </div>
            </div>
        </div>
        """),

        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">

            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab 11: The Roofline</strong> &mdash; This lab found the optimal
                    compression configuration. Lab 11 asks: where does your compressed model
                    sit on the Roofline? INT4 quantization changes both the memory footprint
                    and the arithmetic intensity, shifting the workload's position relative to
                    the ridge point.
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
                    <strong>Read:</strong> @sec-optimizations-quantization and
                    @sec-optimizations-pruning for the full quantization and sparsity mechanics.<br/>
                    <strong>Build:</strong> TinyTorch Module 10 &mdash; implement post-training
                    quantization and measure the accuracy-vs-bitwidth curve yourself.
                </div>
            </div>

        </div>
        """),


        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. Quantizing a 7B LLM from FP16 to INT8 causes less than 1% accuracy loss while providing 4x memory reduction. At what bit-width does the quantization cliff appear — and why is the drop sudden rather than gradual?

    2. Unstructured pruning at 50% sparsity provides 0% speedup on standard GPUs. What property of modern GPU hardware (Tensor Cores, SIMD) causes this — and what type of pruning does achieve a speedup?

    3. You must deploy a 7B model to an iPhone 15 Pro with a 50 ms latency budget. The model weights are 14 GB in FP16. Explain why INT4 quantization (4x bandwidth improvement) is required, and whether unstructured pruning at 70% sparsity would help or not.

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE D: LEDGER_HUD
# ═════════════════════════════════════════════════════════════════════════════

# ── CELL 21: LEDGER_HUD ───────────────────────────────────────────────────────
@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle,
    act1_prediction,
    act2_flagship, act2_midrange, act2_budget,
    LLAMA3_8B_FP32_GB,
, decision_input, decision_ui):
    _ctx_hud = context_toggle.value

    _a1_pred_hud    = act1_prediction.value or "unanswered"
    _a1_correct_hud = _a1_pred_hud == "C"

    _SIZE_MAP_HUD = {
        "fp32": 32.0, "fp16": 16.0, "int8": 8.0, "int4": 4.0,
        "int4_prune10": 3.6, "int4_prune30": 2.8, "int4_prune50": 2.0,
        "int4_prune50_unstruct": 2.0, "distil_4b_int8": 4.1, "distil_1b_int8": 1.0,
    }
    _PARETO_HUD = {
        "fp32": True, "fp16": True, "int8": True, "int4": True,
        "int4_prune10": True, "int4_prune30": True, "int4_prune50": True,
        "int4_prune50_unstruct": False, "distil_4b_int8": False, "distil_1b_int8": False,
    }
    _BUDGETS_HUD = {"flagship": 8.0, "midrange": 4.0, "budget": 2.0}
    _SELECTED_HUD = {
        "flagship": act2_flagship.value,
        "midrange": act2_midrange.value,
        "budget":   act2_budget.value,
    }

    _flagship_gb_hud = _SIZE_MAP_HUD.get(_SELECTED_HUD["flagship"], 0.0)
    _midrange_gb_hud = _SIZE_MAP_HUD.get(_SELECTED_HUD["midrange"], 0.0)
    _budget_gb_hud   = _SIZE_MAP_HUD.get(_SELECTED_HUD["budget"],   0.0)

    _constraint_hit_hud = (
        _flagship_gb_hud > 8.0 or _midrange_gb_hud > 4.0 or _budget_gb_hud > 2.0
    )
    _compression_method_hud = _SELECTED_HUD["flagship"]
    _compression_ratio_hud  = LLAMA3_8B_FP32_GB / max(_flagship_gb_hud, 0.01)
    _pareto_optimal_hud     = all(
        _PARETO_HUD.get(k, False) for k in _SELECTED_HUD.values()
    )

    ledger.save(
        chapter=10,
        design={
            "context":            _ctx_hud,
            "compression_method": _compression_method_hud,
            "compression_ratio":  round(_compression_ratio_hud, 2),
            "act1_prediction":    _a1_pred_hud,
            "act1_correct":       _a1_correct_hud,
            "act2_result":        _budget_gb_hud,
            "act2_decision":      (
                f"flagship={_SELECTED_HUD['flagship']};"
                f"mid={_SELECTED_HUD['midrange']};"
                f"budget={_SELECTED_HUD['budget']}"
            ),
            "constraint_hit":     _constraint_hit_hud,
        "student_justification": str(decision_input.value),
            "pareto_optimal":     _pareto_optimal_hud,
        },
    )

    # ── HUD color coding ──────────────────────────────────────────────────────
    _green  = "#4ade80"
    _red    = "#f87171"
    _yellow = "#fbbf24"
    _muted  = "#94a3b8"

    _a1_icon  = _green  if _a1_correct_hud    else _yellow
    _oom_icon = _red    if _constraint_hit_hud else _green
    _pf_icon  = _green  if _pareto_optimal_hud else _yellow

    mo.vstack([
        mo.md("---"),
        decision_ui,
        mo.Html(f"""
        <div style="background:#0f172a; border-radius:12px; padding:16px 28px;
                    margin-top:24px; border:1px solid #1e293b;
                    font-family:'SF Mono', 'Fira Code', monospace; font-size:0.8rem;">
            <div style="color:#475569; font-size:0.68rem; font-weight:700;
                        text-transform:uppercase; letter-spacing:0.12em; margin-bottom:12px;">
                Design Ledger &mdash; Chapter 10 Saved
            </div>
            <div style="display:flex; gap:32px; flex-wrap:wrap; align-items:center;">
                <div>
                    <span style="color:{_muted}; font-weight:600;">CONTEXT</span>
                    &nbsp;<span style="color:#e2e8f0;">{_ctx_hud.upper()}</span>
                </div>
                <div>
                    <span style="color:{_muted}; font-weight:600;">ACT I PREDICTION</span>
                    &nbsp;<span style="color:{_a1_icon};">
                        {_a1_pred_hud} &mdash; {'CORRECT' if _a1_correct_hud else 'INCORRECT'}
                    </span>
                </div>
                <div>
                    <span style="color:{_muted}; font-weight:600;">COMPRESSION</span>
                    &nbsp;<span style="color:#e2e8f0;">
                        {_compression_method_hud} ({_compression_ratio_hud:.1f}x)
                    </span>
                </div>
                <div>
                    <span style="color:{_muted}; font-weight:600;">OOM HIT</span>
                    &nbsp;<span style="color:{_oom_icon};">
                        {'YES' if _constraint_hit_hud else 'NO'}
                    </span>
                </div>
                <div>
                    <span style="color:{_muted}; font-weight:600;">PARETO-OPTIMAL</span>
                    &nbsp;<span style="color:{_pf_icon};">
                        {'YES' if _pareto_optimal_hud else 'NO'}
                    </span>
                </div>
                <div>
                    <span style="color:{_muted}; font-weight:600;">BUDGET TIER</span>
                    &nbsp;<span style="color:#e2e8f0;">
                        {_budget_gb_hud:.1f} GB / 2 GB limit
                    </span>
                </div>
            </div>
        </div>
        """),
        mo.callout(
            mo.md(
                "**Lab 10 complete.** Your compression decisions are saved to the Design Ledger "
                "and will be referenced in Lab 11 (Hardware Acceleration — Roofline Model), "
                "where you will compute the arithmetic intensity of your compressed model and "
                "see where it falls relative to the memory bandwidth and compute ceilings."
            ),
            kind="success" if (not _constraint_hit_hud and _pareto_optimal_hud) else "info",
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
