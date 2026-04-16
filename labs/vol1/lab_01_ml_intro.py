import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


# ═══════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════


# ─── CELL 0: SETUP ────────────────────────────────────────────────────────
@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import numpy as np

    # WASM bootstrap: install mlsysim from hosted wheel when running in browser
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
    import mlsysim
    from mlsysim import Engine, Models, Hardware

    # ── Hardware constants from registry ──────────────────────────────────
    H100 = Hardware.Cloud.H100
    H100_TFLOPS_FP16 = H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_BW_GBS      = H100.memory.bandwidth.m_as("GB/s")
    H100_RAM_GB      = H100.memory.capacity.m_as("GB")
    H100_TDP_W       = H100.tdp.m_as("W")

    A100 = Hardware.Cloud.A100
    A100_TFLOPS_FP16 = A100.compute.peak_flops.m_as("TFLOPs/s")
    A100_BW_GBS      = A100.memory.bandwidth.m_as("GB/s")
    A100_RAM_GB      = A100.memory.capacity.m_as("GB")

    JETSON = Hardware.Edge.JetsonOrinNX
    JETSON_TFLOPS = JETSON.compute.peak_flops.m_as("TFLOPs/s")
    JETSON_BW_GBS = JETSON.memory.bandwidth.m_as("GB/s")
    JETSON_RAM_GB = JETSON.memory.capacity.m_as("GB")
    JETSON_TDP_W  = JETSON.tdp.m_as("W")

    IPHONE = Hardware.Mobile.iPhone15Pro
    IPHONE_TFLOPS = IPHONE.compute.peak_flops.m_as("TFLOPs/s")
    IPHONE_BW_GBS = IPHONE.memory.bandwidth.m_as("GB/s")
    IPHONE_RAM_GB = IPHONE.memory.capacity.m_as("GB")
    IPHONE_TDP_W  = IPHONE.tdp.m_as("W")

    ESP32 = Hardware.Tiny.ESP32_S3
    ESP32_TFLOPS  = ESP32.compute.peak_flops.m_as("TFLOPs/s")
    ESP32_BW_GBS  = ESP32.memory.bandwidth.m_as("GB/s")
    ESP32_RAM_KB  = ESP32.memory.capacity.m_as("KiB")
    ESP32_RAM_GB  = ESP32_RAM_KB / (1024 * 1024)
    ESP32_TDP_W   = ESP32.tdp.m_as("W")

    HIMAX = Hardware.Tiny.HimaxWE1
    HIMAX_TFLOPS  = HIMAX.compute.peak_flops.m_as("TFLOPs/s")
    HIMAX_BW_GBS  = HIMAX.memory.bandwidth.m_as("GB/s")
    HIMAX_RAM_GB  = HIMAX.memory.capacity.m_as("GB")
    HIMAX_TDP_W   = HIMAX.tdp.m_as("W")

    # ── Model constants ────────────────────────────────────────────────────
    RESNET50_PARAMS  = Models.ResNet50.parameters.m_as("count")
    RESNET50_FLOPS   = Models.ResNet50.inference_flops.m_as("flop")
    RESNET50_SIZE_MB = RESNET50_PARAMS * 2 / (1024 * 1024)  # FP16

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return (
        COLORS, LAB_CSS, apply_plotly_theme, DecisionLog,
        go, mo, np, math,
        Engine, Models, Hardware,
        H100_TFLOPS_FP16, H100_BW_GBS, H100_RAM_GB, H100_TDP_W,
        A100_TFLOPS_FP16, A100_BW_GBS, A100_RAM_GB,
        JETSON_TFLOPS, JETSON_BW_GBS, JETSON_RAM_GB, JETSON_TDP_W,
        IPHONE_TFLOPS, IPHONE_BW_GBS, IPHONE_RAM_GB, IPHONE_TDP_W,
        ESP32_TFLOPS, ESP32_BW_GBS, ESP32_RAM_KB, ESP32_RAM_GB, ESP32_TDP_W,
        HIMAX_TFLOPS, HIMAX_BW_GBS, HIMAX_RAM_GB, HIMAX_TDP_W,
        RESNET50_PARAMS, RESNET50_FLOPS, RESNET50_SIZE_MB,
        ledger,
    )


# ─── CELL 1: HEADER ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(LAB_CSS, mo):
    mo.vstack([
        LAB_CSS,
        mo.Html("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 01
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The AI Triad
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Data &middot; Algorithm &middot; Machine
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                A single model deployed across three hardware targets fails for
                three different physical reasons. The D-A-M triad is the diagnostic
                framework you will use for the entire course.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~51 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter 1: Introduction to ML Systems
                </span>
                <span style="background: rgba(0,143,69,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(0,143,69,0.25);">
                    First Lab of the Curriculum
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">D-A-M Triad Diagnosis</span>
                <span class="badge badge-warn">Iron Law T = D/BW + O/R + L</span>
                <span class="badge badge-fail">200x Out of Memory (OOM) on ESP32</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ─────────────────────────────────────────────────────
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
                <div style="margin-bottom: 3px;">1. <strong>Diagnose which D-A-M axis is binding</strong> &mdash;
                    given a system with poor performance, identify whether Data, Algorithm,
                    or Machine is the bottleneck before investing resources.</div>
                <div style="margin-bottom: 3px;">2. <strong>Apply the Iron Law T&nbsp;=&nbsp;D/BW&nbsp;+&nbsp;O/R&nbsp;+&nbsp;L</strong>
                    to ResNet-50 inference and discover that the H100 is memory-bound
                    at batch=1, not compute-bound.</div>
                <div style="margin-bottom: 3px;">3. <strong>Quantify the deployment spectrum</strong> &mdash; measure
                    the ~1,000,000x compute gap between H100 and ESP32 and explain why
                    the same model is infeasible on a microcontroller by 200x.</div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    D-A-M framework from the Introduction chapter &middot;
                    Iron Law equation from the Iron Law section (Ch. 1) &middot;
                    Deployment spectrum from the Deployment Spectrum section (Ch. 1)
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~51 min</strong><br/>
                    Part A: ~12 min &middot; Part B: ~12 min<br/>
                    Part C: ~12 min &middot; Part D: ~9 min
                </div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 12px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;If the same model deployed to an H100 (NVIDIA's flagship datacenter GPU), a Jetson (NVIDIA's edge AI module), and an ESP32 (Espressif's low-power microcontroller, 512 KB SRAM) fails
                for three different physical reasons &mdash; how do you diagnose which axis
                to fix before spending the budget?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: READING ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **The Introduction chapter** -- The D-A-M framework (Data, Algorithm, Machine) and
      the three worked examples (recommendation system, vision model, language model).
    - **The Iron Law section (Ch. 1)** -- The Iron Law `T = D/BW + O/R + L` with
      variable definitions and the ResNet-50 worked example.
    - **The Deployment Spectrum section (Ch. 1)** -- The deployment spectrum table
      showing Cloud, Edge, Mobile, and TinyML hardware tiers.
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════
# ZONE B-D: ALL PARTS AS TABS
# ═══════════════════════════════════════════════════════════════════════════


# ─── CELL 4: TABS CELL ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    H100_TFLOPS_FP16, H100_BW_GBS, H100_RAM_GB, H100_TDP_W,
    A100_TFLOPS_FP16, A100_BW_GBS, A100_RAM_GB,
    JETSON_TFLOPS, JETSON_BW_GBS, JETSON_RAM_GB, JETSON_TDP_W,
    IPHONE_TFLOPS, IPHONE_BW_GBS, IPHONE_RAM_GB, IPHONE_TDP_W,
    ESP32_TFLOPS, ESP32_BW_GBS, ESP32_RAM_KB, ESP32_RAM_GB, ESP32_TDP_W,
    HIMAX_TFLOPS, HIMAX_BW_GBS, HIMAX_RAM_GB, HIMAX_TDP_W,
    RESNET50_PARAMS, RESNET50_FLOPS, RESNET50_SIZE_MB,
    Engine, Models, Hardware,
    apply_plotly_theme, go, math, mo, np,
):
    # ─────────────────────────────────────────────────────────────────────
    # SHARED WIDGET STATE
    # ─────────────────────────────────────────────────────────────────────

    # Part A widgets
    partA_prediction = mo.ui.radio(
        options={
            "A) Improves proportionally (~4x better)":                  "proportional",
            "B) Improves modestly (~1.3x better)":                       "modest",
            "C) No change -- the bottleneck is elsewhere":               "no_change",
            "D) Gets worse due to overfitting":                          "worse",
        },
        label="A recommendation system is showing poor accuracy. The team proposes "
              "buying 4x more GPUs. What happens to accuracy?",
    )
    return (partA_prediction,)

@app.cell(hide_code=True)
def _(mo, partA_prediction):
    mo.stop(partA_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    # Part B widgets
    partB_prediction = mo.ui.radio(
        options={
            "A) Data loading (D_vol/BW) -- weight loading dominates at batch=1": "data",
            "B) Compute (O/R_peak) -- it is a GPU, compute must dominate":       "compute",
            "C) Framework overhead -- kernel dispatch is the bottleneck":         "overhead",
            "D) All three terms are roughly equal":                                "balanced",
        },
        label="For ResNet-50 inference at batch=1 on an H100 GPU, which Iron Law "
              "term dominates total inference latency?",
    )
    return (partB_prediction,)

@app.cell(hide_code=True)
def _(mo, partB_prediction):
    mo.stop(partB_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    # Part C widgets
    partC_prediction = mo.ui.radio(
        options={
            "A) ~2x over budget (need a small trim)":                   "2x",
            "B) ~10x over budget (need significant compression)":        "10x",
            "C) ~200x over budget (fundamentally infeasible)":           "200x",
            "D) It fits with INT8 quantization":                         "fits",
        },
        label="ResNet-50 requires ~49 MB in FP16. The ESP32-S3 has 512 KB of SRAM. "
              "What is the ratio of model size to available memory?",
    )
    return (partC_prediction,)

@app.cell(hide_code=True)
def _(mo, partC_prediction):
    mo.stop(partC_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    # Part D widgets
    partD_prediction = mo.ui.radio(
        options={
            "A) ~100x":            "100x",
            "B) ~10,000x":         "10000x",
            "C) ~1,000,000x":      "1000000x",
            "D) ~1,000,000,000x":  "1000000000x",
        },
        label="What is the compute ratio between an H100 GPU and an ESP32 "
              "microcontroller?",
    )
    return (partD_prediction,)

@app.cell(hide_code=True)
def _(DecisionLog, mo, partD_prediction):
    mo.stop(partD_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partA_scenario = mo.ui.dropdown(
        options={
            "Rec System: stale training data": "stale_data",
            "Vision Model: insufficient compute": "low_compute",
            "LLM: exceeds device memory": "oom_model",
        },
        value="Rec System: stale training data",
        label="Select system scenario:",
    )
    partA_fix = mo.ui.dropdown(
        options={
            "Add more GPUs (Machine)": "machine",
            "Refresh training data (Data)": "data",
            "Use smaller model (Algorithm)": "algorithm",
        },
        value="Add more GPUs (Machine)",
        label="Prescribe fix:",
    )
    partB_batch = mo.ui.slider(
        start=1, stop=256, value=1, step=1, label="Batch size",
    )
    partC_target = mo.ui.radio(
        options={
            "H100 (Cloud)": "h100",
            "Jetson Orin NX (Edge)": "jetson",
            "ESP32-S3 (TinyML)": "esp32",
        },
        value="H100 (Cloud)",
        label="Deployment target:",
        inline=True,
    )
    partD_scale = mo.ui.radio(
        options={"Linear scale": "linear", "Log scale": "log"},
        value="Linear scale",
        label="Chart scale:",
        inline=True,
    )

    synth_decision_input, synth_decision_ui = DecisionLog(
        placeholder="Based on what I learned in this lab, the most important diagnostic "
                    "question before investing in hardware is..."
    )
    return (partA_scenario, partA_fix, partB_batch, partC_target, partD_scale, synth_decision_input, synth_decision_ui)

@app.cell(hide_code=True)
def _(
    COLORS,
    H100_TFLOPS_FP16, H100_BW_GBS, H100_RAM_GB, H100_TDP_W,
    A100_TFLOPS_FP16, A100_BW_GBS, A100_RAM_GB,
    JETSON_TFLOPS, JETSON_BW_GBS, JETSON_RAM_GB, JETSON_TDP_W,
    IPHONE_TFLOPS, IPHONE_BW_GBS, IPHONE_RAM_GB, IPHONE_TDP_W,
    ESP32_TFLOPS, ESP32_BW_GBS, ESP32_RAM_KB, ESP32_RAM_GB, ESP32_TDP_W,
    HIMAX_TFLOPS, HIMAX_BW_GBS, HIMAX_RAM_GB, HIMAX_TDP_W,
    RESNET50_PARAMS, RESNET50_FLOPS, RESNET50_SIZE_MB,
    Engine, Models, Hardware,
    apply_plotly_theme, go, math, mo, np,
    partA_prediction, partA_scenario, partA_fix,
    partB_prediction, partB_batch,
    partC_prediction, partC_target,
    partD_prediction, partD_scale,
    synth_decision_input, synth_decision_ui,
    ledger,
):
    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER -- Three Axes, One System
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; CTO, MedVision Health
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our DR screening model is showing poor accuracy on the latest patient
                cohort. The infrastructure team says we should buy 4x more GPUs. The data
                team says the training data is 18 months old. The ML team says we need a
                bigger model. Who is right?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Dr. Priya Mehta, CTO &middot; MedVision Health
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The AI Triad: Data, Algorithm, Machine

        Every ML system has three axes -- **Data** (quality, freshness, volume),
        **Algorithm** (architecture, parameters, training), and **Machine** (compute,
        memory, bandwidth). A system performing poorly cannot be fixed by throwing
        resources at the wrong axis. You must diagnose *which* axis is binding
        before you can improve anything.

        This is the foundational diagnostic skill of the entire course: identifying
        which constraint is active before investing.
        """))

        # Prediction
        items.append(partA_prediction)

        if partA_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the diagnostic simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        # Reveal prediction
        _pred = partA_prediction.value
        if _pred == "no_change":
            _reveal = ("**Correct.** The system has stale training data (Data axis). "
                       "Adding GPUs (Machine axis) changes nothing because the bottleneck "
                       "is not compute. Diagnosis must precede investment.")
            _kind = "success"
        elif _pred == "proportional":
            _reveal = ("**No. More GPUs do not fix stale data.** The training data is "
                       "18 months old. The distribution has shifted. The Machine axis is "
                       "not the bottleneck -- the Data axis is. 4x more GPUs at $120K each "
                       "yields 0% accuracy improvement.")
            _kind = "warn"
        elif _pred == "modest":
            _reveal = ("**The improvement is actually zero, not modest.** The bottleneck is "
                       "Data (stale training distribution), not Machine. Adding compute to "
                       "a data-starved system is like buying a faster engine for a car with "
                       "no fuel.")
            _kind = "warn"
        else:
            _reveal = ("**Overfitting is a real concern, but the primary issue is simpler.** "
                       "The training data is 18 months stale. The model is not overfitting "
                       "to current data -- it was trained on a distribution that no longer "
                       "exists. The Data axis is the binding constraint.")
            _kind = "warn"

        items.append(mo.callout(mo.md(_reveal), kind=_kind))

        # Interactive diagnosis
        items.append(mo.md("""
        ### Diagnose the Patient

        Three systems are presented as patient charts. For each, identify the
        binding D-A-M axis and prescribe the correct fix. Try applying the wrong
        fix first to see zero improvement.
        """))

        items.append(mo.hstack([partA_scenario, partA_fix], justify="start", gap="2rem"))

        _scenario = partA_scenario.value
        _fix = partA_fix.value

        # Scenario definitions
        _scenarios = {
            "stale_data": {
                "name": "Recommendation System",
                "problem": "Accuracy dropped 8% over 6 months",
                "binding": "data",
                "detail": "Training data is 18 months old. User preferences have shifted. "
                          "The model is confidently recommending products from last year's trends.",
                "metrics": {"Accuracy": "72%", "GPU Util": "85%", "Latency": "12 ms",
                            "Data Age": "18 months"},
            },
            "low_compute": {
                "name": "Medical Vision Model",
                "problem": "Inference takes 340 ms, SLA requires 100 ms",
                "binding": "machine",
                "detail": "The model architecture is correct and the data is fresh, but the "
                          "inference hardware (T4 GPU) cannot process the high-resolution images "
                          "fast enough. The compute term dominates the Iron Law.",
                "metrics": {"Accuracy": "94%", "GPU Util": "99%", "Latency": "340 ms",
                            "Data Age": "2 weeks"},
            },
            "oom_model": {
                "name": "On-Device Language Model",
                "problem": "Model does not fit in device memory (8 GB phone, 14 GB model)",
                "binding": "algorithm",
                "detail": "The model is too large for the target device. More data or faster "
                          "compute will not help -- the Algorithm axis (model architecture and "
                          "size) must change. Quantization or a smaller architecture is needed.",
                "metrics": {"Accuracy": "N/A (OOM)", "GPU Util": "N/A", "Latency": "N/A",
                            "Model Size": "14 GB"},
            },
        }

        _sc = _scenarios[_scenario]
        _correct_fix = _sc["binding"]
        _fixed = _fix == _correct_fix

        # Diagnostic card
        _status_color = COLORS["GreenLine"] if _fixed else COLORS["RedLine"]
        _status_bg = COLORS["GreenLL"] if _fixed else COLORS["RedLL"]
        _status_text = "FIX APPLIED -- SYSTEM RECOVERING" if _fixed else "WRONG AXIS -- NO IMPROVEMENT"

        _metrics_html = ""
        for _k, _v in _sc["metrics"].items():
            _metrics_html += f"""
            <div style="display:flex; justify-content:space-between; padding:6px 0;
                        border-bottom:1px solid {COLORS['Border']};">
                <span style="color:{COLORS['TextSec']}; font-size:0.85rem;">{_k}</span>
                <span style="font-family:monospace; font-weight:700;
                             color:{COLORS['Text']}; font-size:0.85rem;">{_v}</span>
            </div>"""

        items.append(mo.Html(f"""
        <div style="display:flex; gap:20px; flex-wrap:wrap; margin:16px 0;">
            <div style="flex:1; min-width:280px; background:white;
                        border:1px solid {COLORS['Border']}; border-radius:12px; padding:20px;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px;">
                    Patient Chart: {_sc['name']}</div>
                <div style="font-size:0.95rem; font-weight:700; color:{COLORS['Text']};
                            margin-bottom:8px;">{_sc['problem']}</div>
                <div style="font-size:0.85rem; color:{COLORS['TextSec']}; line-height:1.6;
                            margin-bottom:14px;">{_sc['detail']}</div>
                {_metrics_html}
                <div style="margin-top:14px; padding:8px 12px; background:{COLORS['OrangeLL']};
                            border-radius:8px; font-size:0.82rem; font-weight:700;
                            color:{COLORS['OrangeLine']}; text-align:center;">
                    Binding Axis: {_correct_fix.upper()}</div>
            </div>
            <div style="flex:1; min-width:280px; background:white;
                        border:2px solid {_status_color}; border-radius:12px; padding:20px;">
                <div style="font-size:0.72rem; font-weight:700; color:{_status_color};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px;">
                    Prescribed Fix: {_fix.replace('_', ' ').title()}</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_status_color};
                            text-align:center; margin:20px 0;">
                    {'Improvement: RECOVERED' if _fixed else 'Improvement: 0%'}</div>
                <div style="font-size:0.85rem; color:{COLORS['TextSec']}; line-height:1.6;
                            text-align:center;">
                    {'The fix addresses the binding constraint.' if _fixed
                     else f'This fix targets the {_fix.title()} axis, but the binding constraint is on the {_correct_fix.title()} axis. No improvement results.'}</div>
                <div style="margin-top:14px; padding:10px; background:{_status_bg};
                            border-radius:8px; text-align:center; font-size:0.82rem;
                            font-weight:700; color:{_status_color};">
                    {_status_text}</div>
            </div>
        </div>
        """))

        if not _fixed:
            items.append(mo.callout(mo.md(
                f"**Wrong axis.** You prescribed a fix for the **{_fix.title()}** axis, "
                f"but the binding constraint is **{_correct_fix.title()}**. "
                "Change the fix dropdown to see the system recover. "
                "This is the lesson: diagnosis must precede investment."
            ), kind="danger"))

        # MathPeek
        items.append(mo.accordion({
            "Math Peek: Why Diagnosis Precedes Investment": mo.md("""
The D-A-M framework formalizes a simple principle: system performance is bounded
by the **worst** axis, not the best.

$$
\\text{Performance} = f(\\min(D_{\\text{quality}},\\; A_{\\text{capacity}},\\; M_{\\text{throughput}}))
$$

Improving an axis that is not the minimum has zero marginal return. The Iron Law
(Part B) makes this quantitative for the Machine axis. Data quality metrics and
Algorithm capacity metrics complete the picture.
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER -- The Iron Law Surprise
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; VP Engineering (escalated from Part A)
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;You diagnosed the bottleneck by intuition. But how do you <em>quantify</em>
                which axis is binding? Our H100 inference is slower than expected. Engineering
                proposes doubling compute throughput. Finance wants a number. Is this worth it?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; David Park, VP Engineering &middot; MedVision Health
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Iron Law Decomposes Inference Latency

        ```
        T  =  D_vol/BW  +  O/(R_peak * eta)  +  L_lat
              --------     ----------------     ------
              Data term    Compute term         Overhead
        ```

        Each term is independent. Doubling compute throughput (R_peak) reduces **only**
        the Compute term. If the Data term is larger -- which it is at low batch sizes --
        the speedup is negligible.

        **Arithmetic Intensity (AI)** = O / D_vol (FLOPs per byte). Note: AI here refers to arithmetic intensity, not artificial intelligence.
        Below the Ridge Point (R_peak / BW), the workload is **memory-bound**.
        """))

        items.append(partB_prediction)

        if partB_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the Latency Waterfall."),
                kind="warn",
            ))
            return mo.vstack(items)

        # Controls
        items.append(partB_batch)

        _batch = partB_batch.value
        _eta = 0.5

        # Use the Engine API — the same Roofline solver students will use for all 14 labs
        _profile = Engine.solve(
            Models.ResNet50, Hardware.Cloud.H100,
            batch_size=_batch, precision="fp16", efficiency=_eta,
        )
        _t_data_ms = _profile.latency_memory.m_as("ms")
        _t_comp_ms = _profile.latency_compute.m_as("ms")
        _t_ovh_ms = _profile.latency_overhead.m_as("ms")
        _t_total = _profile.latency.m_as("ms")
        _bottleneck = _profile.bottleneck
        _mfu = _profile.mfu * 100

        _terms = {"Data": _t_data_ms, "Compute": _t_comp_ms, "Overhead": _t_ovh_ms}

        # Waterfall chart
        _labels = ["Data Term (D/BW)", "Compute Term (O/R*eta)", "Overhead (L)"]
        _vals = [_t_data_ms, _t_comp_ms, _t_ovh_ms]
        _bar_colors = [COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["Grey"]]
        _tkeys = list(_terms.keys())

        _bottleneck_map = {"Memory": "Data", "Compute": "Compute"}
        _fig = go.Figure()
        for _i, (_lbl, _v, _bc) in enumerate(zip(_labels, _vals, _bar_colors)):
            _bw = 3 if _tkeys[_i] == _bottleneck_map.get(_bottleneck, "") else 1
            _fig.add_trace(go.Bar(
                name=_lbl, x=[_lbl], y=[_v],
                marker_color=_bc, marker_line_color="white",
                marker_line_width=_bw, opacity=0.88,
                hovertemplate="%{x}: %{y:.4f} ms<extra></extra>",
            ))
        _fig.update_layout(
            barmode="group", height=320,
            yaxis=dict(title="Latency (ms)", gridcolor="#f1f5f9"),
            xaxis=dict(gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=50, r=20, t=60, b=30),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md(f"### Latency Waterfall -- ResNet-50 on H100 (batch = {_batch})"))
        items.append(mo.as_html(_fig))

        # Metric cards
        def _mcard(label, val_ms, is_bot, sub):
            _col = COLORS["RedLine"] if is_bot else COLORS["BlueLine"]
            _brd = f"border: 2px solid {COLORS['RedLine']};" if is_bot else "border: 1px solid #e2e8f0;"
            return f"""
            <div style="padding:16px; {_brd} border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.75rem; font-weight:600; margin-bottom:4px;">
                    {label}{"  (bottleneck)" if is_bot else ""}</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_col};">{val_ms:.4f} ms</div>
                <div style="font-size:0.7rem; color:#94a3b8; margin-top:4px;">{sub}</div>
            </div>"""

        _mem_gb = _profile.memory_footprint.m_as("GB")
        _ai = _profile.arithmetic_intensity.magnitude
        _ridge_point = H100_TFLOPS_FP16 * 1000 / H100_BW_GBS  # FLOPs/Byte

        _cards = f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            {_mcard("Data Term", _t_data_ms, _bottleneck == "Memory",
                    f"{_mem_gb:.4f} GB / {H100_BW_GBS:,.0f} GB/s")}
            {_mcard("Compute Term", _t_comp_ms, _bottleneck == "Compute",
                    f"{RESNET50_FLOPS*_batch/1e9:.1f} GF / ({H100_TFLOPS_FP16:.0f}T * {_eta})")}
            {_mcard("Overhead", _t_ovh_ms, False, "dispatch tax")}
            {_mcard("Total", _t_total, False, f"Bottleneck: {_bottleneck}")}
        </div>
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:0 0 16px 0;">
            <div style="padding:12px 20px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        background:white; flex:1; text-align:center;">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Model FLOPs Utilization (MFU)</div>
                <div style="font-size:1.3rem; font-weight:800;
                            color:{'#008F45' if _mfu > 50 else '#CC5500' if _mfu > 20 else '#CB202D'};">
                    {_mfu:.1f}%</div>
            </div>
            <div style="padding:12px 20px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        background:white; flex:1; text-align:center;">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Arithmetic Intensity</div>
                <div style="font-size:1.3rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_ai:.1f} F/B</div>
            </div>
            <div style="padding:12px 20px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        background:white; flex:1; text-align:center;">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Ridge Point (H100)</div>
                <div style="font-size:1.3rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    ~{_ridge_point:.0f} F/B</div>
            </div>
        </div>"""
        items.append(mo.Html(_cards))

        # Formula
        items.append(mo.md(f"""
**Iron Law -- Live Calculation** (`batch = {_batch}`)

```
T  =  D_vol/BW              +  O/(R * eta)                       +  L
   =  {_mem_gb:.4f} GB / {H100_BW_GBS:,} GB/s  +  {RESNET50_FLOPS*_batch:.2e} / ({H100_TFLOPS_FP16:.0f}T * {_eta})  +  {_t_ovh_ms:.3f} ms
   =  {_t_data_ms:.4f} ms           +  {_t_comp_ms:.4f} ms                     +  {_t_ovh_ms:.4f} ms
   =  {_t_total:.4f} ms total  (Bottleneck: {_bottleneck})

AI = {_ai:.1f} FLOPs/Byte  {'<<' if _ai < _ridge_point else '>>'} Ridge Point ~{_ridge_point:.0f} FLOPs/Byte
     => {'MEMORY-BOUND' if _ai < _ridge_point else 'COMPUTE-BOUND'}
```
"""))

        # Batch sweep chart showing crossover — powered by Engine.solve()
        _batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        _data_times = []
        _comp_times = []
        for _b in _batches:
            _p = Engine.solve(Models.ResNet50, Hardware.Cloud.H100, batch_size=_b, precision="fp16", efficiency=_eta)
            _data_times.append(_p.latency_memory.m_as("ms"))
            _comp_times.append(_p.latency_compute.m_as("ms"))

        _fig2 = go.Figure()
        _fig2.add_trace(go.Scatter(
            x=_batches, y=_data_times, mode="lines+markers",
            name="Data Term (D/BW)", line=dict(color=COLORS["BlueLine"], width=2.5),
            hovertemplate="Batch %{x}: %{y:.4f} ms<extra></extra>",
        ))
        _fig2.add_trace(go.Scatter(
            x=_batches, y=_comp_times, mode="lines+markers",
            name="Compute Term (O/R*eta)", line=dict(color=COLORS["OrangeLine"], width=2.5),
            hovertemplate="Batch %{x}: %{y:.4f} ms<extra></extra>",
        ))
        _fig2.update_layout(
            height=280,
            xaxis=dict(title="Batch Size", type="log", gridcolor="#f1f5f9"),
            yaxis=dict(title="Latency (ms)", type="log", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig2)
        items.append(mo.md("### Bottleneck Crossover: Data vs Compute by Batch Size"))
        items.append(mo.as_html(_fig2))

        # Prediction reveal — use Engine at batch=1 for reference values
        _pred = partB_prediction.value
        _ref = Engine.solve(Models.ResNet50, Hardware.Cloud.H100, batch_size=1, precision="fp16", efficiency=_eta)
        _t_data_ref = _ref.latency_memory.m_as("ms")
        _t_comp_ref = _ref.latency_compute.m_as("ms")

        if _pred == "data":
            _rev_msg = (
                f"**Correct.** At batch=1 on the H100, the Data Term is {_t_data_ref:.4f} ms -- "
                f"approximately {_t_data_ref/_t_comp_ref:.0f}x the Compute Term ({_t_comp_ref:.4f} ms). "
                "The H100 is so fast at arithmetic that it spends most time waiting for data from High Bandwidth Memory (HBM). "
                "Buying a 2x faster GPU would yield less than 10% latency improvement."
            )
            _rev_kind = "success"
        elif _pred == "compute":
            _rev_msg = (
                f"**The compute term is actually the smallest.** "
                f"Data Term ({_t_data_ref:.4f} ms) is ~{_t_data_ref/_t_comp_ref:.0f}x the "
                f"Compute Term ({_t_comp_ref:.4f} ms) at batch=1. "
                "A GPU is not always compute-bound. "
                "Increase the batch slider to watch the crossover happen."
            )
            _rev_kind = "warn"
        else:
            _rev_msg = (
                f"**The dominant term is Data at batch=1.** "
                f"Data: {_t_data_ref:.4f} ms, Compute: {_t_comp_ref:.4f} ms, "
                f"Overhead: {_ref.latency_overhead.m_as('ms'):.4f} ms. "
                "The workload is memory-bound. "
                "Slide batch size up to ~32-64 to watch it become compute-bound."
            )
            _rev_kind = "warn"

        items.append(mo.callout(mo.md(_rev_msg), kind=_rev_kind))

        # MathPeek
        items.append(mo.accordion({
            "Math Peek: The Iron Law": mo.md(f"""
$$
T = \\frac{{D_{{\\text{{vol}}}}}}{{BW}} + \\frac{{O}}{{R_{{\\text{{peak}}}} \\cdot \\eta}} + L_{{\\text{{lat}}}}
$$

**Ridge Point** = $R_{{\\text{{peak}}}} / BW$ = {H100_TFLOPS_FP16:.0f} TFLOPS / {H100_BW_GBS:,.0f} GB/s = **~{_ridge_point:.0f} FLOPs/Byte**

When Arithmetic Intensity < Ridge Point, the workload is **memory-bound**.
At batch=1: AI = {RESNET50_FLOPS/1e9:.1f} GFLOPs / {RESNET50_SIZE_MB/1024:.4f} GB = ~{RESNET50_FLOPS/(RESNET50_SIZE_MB/1024*1e9):.0f} FLOPs/Byte

Since {RESNET50_FLOPS/(RESNET50_SIZE_MB/1024*1e9):.0f} << {_ridge_point:.0f}, the workload is **deeply memory-bound**.
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER -- The Triad Across Targets
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; VP Product (escalated)
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;The Iron Law told you the H100 is memory-bound at batch=1. But what
                happens when you take that <em>same</em> model and deploy it on an edge device?
                Or a microcontroller? We need to know before we commit to hardware.&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; MedVision Health, Deployment Planning
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Same Model, Three Targets, Three Different Failures

        ResNet-50 deployed on three hardware targets is diagnosed by the Engine as
        having *different* binding constraints at each tier:
        - **H100**: Memory-bound (Machine axis is fine but underutilized)
        - **Jetson Orin NX**: Fits, but severely bandwidth-limited
        - **ESP32**: Flatly infeasible -- 49 MB model vs 512 KB SRAM

        "The model works" is meaningless without specifying *where*.
        """))

        items.append(partC_prediction)

        if partC_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the cross-target comparison."),
                kind="warn",
            ))
            return mo.vstack(items)

        # Target selector
        items.append(partC_target)

        # Run Engine.solve for all three targets
        _hw_specs = {
            "h100": {
                "name": "H100 (Cloud)", "ram_gb": H100_RAM_GB,
                "tflops": H100_TFLOPS_FP16, "bw_gbs": H100_BW_GBS,
                "tdp_w": H100_TDP_W, "color": COLORS["BlueLine"],
            },
            "jetson": {
                "name": "Jetson Orin NX (Edge)", "ram_gb": JETSON_RAM_GB,
                "tflops": JETSON_TFLOPS, "bw_gbs": JETSON_BW_GBS,
                "tdp_w": JETSON_TDP_W, "color": COLORS["OrangeLine"],
            },
            "esp32": {
                "name": "ESP32-S3 (TinyML)", "ram_gb": ESP32_RAM_GB,
                "tflops": ESP32_TFLOPS, "bw_gbs": ESP32_BW_GBS,
                "tdp_w": ESP32_TDP_W, "color": COLORS["RedLine"],
            },
        }

        # Build comparison table
        _table_rows = ""
        for _key, _hw in _hw_specs.items():
            _model_gb = RESNET50_SIZE_MB / 1024
            _feasible = _model_gb < _hw["ram_gb"]
            _mem_ratio = _model_gb / _hw["ram_gb"]

            if not _feasible:
                _bottleneck_str = "INFEASIBLE (OOM)"
                _latency_str = "N/A"
                _status_badge = f'<span class="badge badge-fail">INFEASIBLE ({_mem_ratio:.0f}x over)</span>'
                _dam_axis = "Machine (capacity)"
            else:
                _t_data = (_model_gb / _hw["bw_gbs"]) * 1000
                _t_comp = (RESNET50_FLOPS / (_hw["tflops"] * 1e12 * 0.5)) * 1000
                _t_total_hw = _t_data + _t_comp + 0.05
                if _t_data > _t_comp:
                    _bottleneck_str = "Memory-bound"
                    _dam_axis = "Machine (bandwidth)"
                else:
                    _bottleneck_str = "Compute-bound"
                    _dam_axis = "Machine (compute)"
                _latency_str = f"{_t_total_hw:.2f} ms"
                _status_badge = f'<span class="badge badge-ok">Feasible</span>'

            _highlight = "background:#f0f8ff;" if _key == partC_target.value else ""
            _table_rows += f"""
            <tr style="{_highlight}">
                <td style="padding:10px; font-weight:700; color:{_hw['color']};">{_hw['name']}</td>
                <td style="padding:10px; font-family:monospace;">{_hw['ram_gb']:.2f} GB</td>
                <td style="padding:10px; font-family:monospace;">{_hw['tflops']:.3f} TFLOPS</td>
                <td style="padding:10px;">{_status_badge}</td>
                <td style="padding:10px; font-family:monospace;">{_latency_str}</td>
                <td style="padding:10px; font-weight:600;">{_bottleneck_str}</td>
                <td style="padding:10px; color:{COLORS['OrangeLine']}; font-weight:700;">{_dam_axis}</td>
            </tr>"""

        items.append(mo.Html(f"""
        <div style="overflow-x:auto; margin:16px 0;">
            <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
                <thead>
                    <tr style="background:{COLORS['Surface2']}; border-bottom:2px solid {COLORS['Border']};">
                        <th style="padding:10px; text-align:left;">Target</th>
                        <th style="padding:10px; text-align:left;">RAM</th>
                        <th style="padding:10px; text-align:left;">Compute</th>
                        <th style="padding:10px; text-align:left;">Feasibility</th>
                        <th style="padding:10px; text-align:left;">Latency</th>
                        <th style="padding:10px; text-align:left;">Bottleneck</th>
                        <th style="padding:10px; text-align:left;">Binding D-A-M Axis</th>
                    </tr>
                </thead>
                <tbody>{_table_rows}</tbody>
            </table>
        </div>
        """))

        # Detailed view for selected target
        _sel = partC_target.value
        _hw = _hw_specs[_sel]
        _model_gb = RESNET50_SIZE_MB / 1024
        _feasible = _model_gb < _hw["ram_gb"]

        if not _feasible:
            _mem_ratio = _model_gb / _hw["ram_gb"]
            items.append(mo.callout(mo.md(
                f"**INFEASIBLE: {_mem_ratio:.0f}x over memory budget.** "
                f"ResNet-50 requires {RESNET50_SIZE_MB:.0f} MB in FP16. "
                f"The {_hw['name']} has {ESP32_RAM_KB:.0f} KB of SRAM. "
                f"This is a **{_mem_ratio:.0f}x memory gap**. "
                "No amount of compression will make ResNet-50 run on this device. "
                "You need a *different algorithm entirely* (e.g., DS-CNN at 200K params)."
            ), kind="danger"))

            # OOM visualization
            items.append(mo.Html(f"""
            <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
                <div style="padding:16px; border:2px solid {COLORS['RedLine']}; border-radius:10px;
                            min-width:150px; text-align:center; background:{COLORS['RedLL']}; flex:1;">
                    <div style="color:{COLORS['RedLine']}; font-size:0.78rem; font-weight:700;">Model Size</div>
                    <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">
                        {RESNET50_SIZE_MB:.0f} MB</div>
                </div>
                <div style="padding:16px; border:2px solid {COLORS['RedLine']}; border-radius:10px;
                            min-width:150px; text-align:center; background:{COLORS['RedLL']}; flex:1;">
                    <div style="color:{COLORS['RedLine']}; font-size:0.78rem; font-weight:700;">Available SRAM</div>
                    <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">
                        {ESP32_RAM_KB:.0f} KB</div>
                </div>
                <div style="padding:16px; border:2px solid {COLORS['RedLine']}; border-radius:10px;
                            min-width:150px; text-align:center; background:{COLORS['RedLL']}; flex:1;">
                    <div style="color:{COLORS['RedLine']}; font-size:0.78rem; font-weight:700;">Overflow</div>
                    <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">
                        {_mem_ratio:.0f}x</div>
                </div>
            </div>
            """))
        else:
            _t_data = (_model_gb / _hw["bw_gbs"]) * 1000
            _t_comp = (RESNET50_FLOPS / (_hw["tflops"] * 1e12 * 0.5)) * 1000
            _t_total_hw = _t_data + _t_comp + 0.05
            _pct_data = _t_data / _t_total_hw * 100
            _pct_comp = _t_comp / _t_total_hw * 100
            items.append(mo.callout(mo.md(
                f"**Feasible on {_hw['name']}.** "
                f"Latency = {_t_total_hw:.2f} ms "
                f"(Data: {_t_data:.2f} ms [{_pct_data:.0f}%], "
                f"Compute: {_t_comp:.2f} ms [{_pct_comp:.0f}%]). "
                f"{'Memory-bound: the bandwidth term dominates.' if _t_data > _t_comp else 'Compute-bound: the operations term dominates.'}"
            ), kind="info"))

        # Prediction reveal
        _pred = partC_prediction.value
        _actual_ratio = RESNET50_SIZE_MB * 1024 / ESP32_RAM_KB

        if _pred == "200x":
            _rev = (f"**Correct.** The ratio is ~{_actual_ratio:.0f}x. "
                    "This is not a compression problem -- it is a feasibility violation. "
                    "You need a fundamentally different model architecture for TinyML.")
            _rkind = "success"
        elif _pred == "2x":
            _rev = (f"**The gap is ~{_actual_ratio:.0f}x, not 2x.** "
                    "Most students underestimate MCU-scale memory because they have never "
                    "worked with 512 KB. Select the ESP32 target above to see the OOM.")
            _rkind = "warn"
        elif _pred == "10x":
            _rev = (f"**The gap is ~{_actual_ratio:.0f}x, not 10x.** "
                    "Even aggressive 8x compression (INT4 quantization) would leave a "
                    "~25x memory gap. The model architecture must change entirely.")
            _rkind = "warn"
        else:
            _rev = (f"**INT8 quantization only halves the memory to ~25 MB.** "
                    f"The ESP32 has {ESP32_RAM_KB:.0f} KB. The gap is ~{_actual_ratio:.0f}x. "
                    "Quantization helps but cannot bridge a 200x gap.")
            _rkind = "warn"

        items.append(mo.callout(mo.md(
            f"**You predicted:** {_pred}  |  **Actual ratio:** ~{_actual_ratio:.0f}x"
        ), kind="info"))
        items.append(mo.callout(mo.md(_rev), kind=_rkind))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER -- The Deployment Spectrum
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Strategic Review &middot; CEO, MedVision Health
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We want to deploy our DR screening model everywhere: cloud for hospitals,
                edge for mobile clinics, and on-device for remote villages with no connectivity.
                How different are these targets, really? Can we just run the same model at
                different speeds?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; James Chen, CEO &middot; MedVision Health
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Full Deployment Spectrum: 9 Orders of Magnitude

        The deployment spectrum spans 9 orders of magnitude in compute and 5 orders
        of magnitude in memory. This gap is so vast that a universal ML software stack
        is physically impossible -- each deployment tier requires fundamentally different
        choices on every D-A-M axis.
        """))

        items.append(partD_prediction)

        if partD_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the deployment spectrum chart."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(partD_scale)

        _scale = partD_scale.value

        # Hardware data from registry
        _hw_data = [
            ("H100",     H100_TFLOPS_FP16,  H100_RAM_GB,   H100_TDP_W,  COLORS["BlueLine"]),
            ("A100",     A100_TFLOPS_FP16,   A100_RAM_GB,   700,         COLORS["BlueLine"]),
            ("Jetson",   JETSON_TFLOPS,      JETSON_RAM_GB, JETSON_TDP_W, COLORS["OrangeLine"]),
            ("iPhone",   IPHONE_TFLOPS,      IPHONE_RAM_GB, IPHONE_TDP_W, COLORS["OrangeLine"]),
            ("ESP32",    ESP32_TFLOPS,       ESP32_RAM_GB,  ESP32_TDP_W,  COLORS["RedLine"]),
            ("Himax",    HIMAX_TFLOPS,       HIMAX_RAM_GB,  HIMAX_TDP_W,  COLORS["RedLine"]),
        ]

        _names = [d[0] for d in _hw_data]
        _compute = [d[1] for d in _hw_data]
        _memory = [d[2] for d in _hw_data]
        _colors = [d[4] for d in _hw_data]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            name="Compute (TFLOPS)", x=_names, y=_compute,
            marker_color=[COLORS["BlueLine"]] * len(_names),
            opacity=0.85,
            hovertemplate="%{x}: %{y:,.0f} TFLOPS<extra></extra>",
        ))
        _fig.add_trace(go.Bar(
            name="Memory (GB)", x=_names, y=_memory,
            marker_color=[COLORS["GreenLine"]] * len(_names),
            opacity=0.85,
            hovertemplate="%{x}: %{y:,.0f} GB<extra></extra>",
        ))
        _fig.update_layout(
            barmode="group", height=380,
            yaxis=dict(title="Magnitude (units vary)", type=_scale, gridcolor="#f1f5f9"),
            xaxis=dict(gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        if _scale == "linear":
            items.append(mo.callout(mo.md(
                "**On linear scale, the ESP32 and Himax bars are invisible.** "
                "The H100's compute is over 1,000,000x larger. "
                "Switch to log scale to see all devices. "
                "This invisibility IS the point."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                "**On log scale, each gridline is 10x, not +1.** "
                "The visual gap between H100 and ESP32 represents six orders of magnitude "
                "in compute. No compiler can bridge this gap."
            ), kind="info"))

        # Ratio cards
        _compute_ratio = H100_TFLOPS_FP16 / ESP32_TFLOPS
        _memory_ratio = H100_RAM_GB / ESP32_RAM_GB

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Compute Gap</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">
                    ~{_compute_ratio:,.0f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">H100 vs ESP32</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Memory Gap</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    ~{_memory_ratio:,.0f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">H100 vs ESP32</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Orders of Magnitude</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">
                    ~{math.log10(_compute_ratio):.0f}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">in compute TFLOPS</div>
            </div>
        </div>
        """))

        # What this means per D-A-M axis
        items.append(mo.Html(f"""
        <div style="overflow-x:auto; margin:16px 0;">
            <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
                <thead>
                    <tr style="background:{COLORS['Surface2']}; border-bottom:2px solid {COLORS['Border']};">
                        <th style="padding:10px; text-align:left;">Tier</th>
                        <th style="padding:10px; text-align:left;">Data Strategy</th>
                        <th style="padding:10px; text-align:left;">Algorithm Strategy</th>
                        <th style="padding:10px; text-align:left;">Machine Constraint</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding:10px; font-weight:700; color:{COLORS['BlueLine']};">Cloud</td>
                        <td style="padding:10px;">Full datasets, TB-scale</td>
                        <td style="padding:10px;">Full models (ResNet-50, GPT-3)</td>
                        <td style="padding:10px;">Throughput, cost</td>
                    </tr>
                    <tr style="background:{COLORS['Surface2']};">
                        <td style="padding:10px; font-weight:700; color:{COLORS['OrangeLine']};">Edge</td>
                        <td style="padding:10px;">Filtered, preprocessed</td>
                        <td style="padding:10px;">Compressed models (MobileNet, INT8)</td>
                        <td style="padding:10px;">Bandwidth, power</td>
                    </tr>
                    <tr>
                        <td style="padding:10px; font-weight:700; color:{COLORS['RedLine']};">TinyML</td>
                        <td style="padding:10px;">Preprocessed features only</td>
                        <td style="padding:10px;">Purpose-built models (DS-CNN, 200K params)</td>
                        <td style="padding:10px;">Capacity (KB-scale SRAM)</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """))

        # Prediction reveal
        _pred = partD_prediction.value
        _actual = H100_TFLOPS_FP16 / ESP32_TFLOPS
        _pred_labels = {
            "100x": "~100x", "10000x": "~10,000x",
            "1000000x": "~1,000,000x", "1000000000x": "~1,000,000,000x",
        }

        if _pred == "1000000x":
            _rev = (f"**Correct.** The compute gap is ~{_actual:,.0f}x -- "
                    "six orders of magnitude.")
            _rkind = "success"
        elif _pred == "100x":
            _rev = (f"**The gap is ~{_actual/100:.0f}x larger than your prediction.** "
                    f"Actual: ~{_actual:,.0f}x. Each tier drops ~100x, "
                    "compounding across three steps.")
            _rkind = "warn"
        elif _pred == "10000x":
            _rev = (f"**Close but ~100x too low.** Actual: ~{_actual:,.0f}x.")
            _rkind = "warn"
        else:
            _rev = (f"**Overshot.** Actual: ~{_actual:,.0f}x (~10^6, not 10^9).")
            _rkind = "warn"

        items.append(mo.callout(mo.md(
            f"**You predicted:** {_pred_labels.get(_pred, _pred)}  |  "
            f"**Actual compute gap:** ~{_actual:,.0f}x"
        ), kind="info"))
        items.append(mo.callout(mo.md(_rev), kind=_rkind))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        _compute_ratio = H100_TFLOPS_FP16 / ESP32_TFLOPS
        _mem_ratio = RESNET50_SIZE_MB * 1024 / ESP32_RAM_KB
        _ridge_point = H100_TFLOPS_FP16 * 1000 / H100_BW_GBS

        # Persist to Design Ledger when student writes a decision
        if synth_decision_input.value:
            ledger.save(chapter=1, design={
                "insight": synth_decision_input.value,
                "bottleneck_at_batch1": "Memory",
                "compute_ratio_h100_esp32": f"{_compute_ratio:,.0f}x",
                "memory_ratio_resnet50_esp32": f"{_mem_ratio:.0f}x",
            })

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
                        <strong>1. Diagnosis precedes investment.</strong>
                        The D-A-M triad is a diagnostic framework: a system performing poorly
                        cannot be fixed by throwing resources at the wrong axis. A recommendation
                        system with stale data gains 0% from 4x more GPUs.
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>2. The Iron Law reveals the binding constraint.</strong>
                        At batch=1, ResNet-50 on H100 is memory-bound
                        (AI ~40 FLOPs/Byte << Ridge Point ~{_ridge_point:.0f} FLOPs/Byte).
                        Doubling compute yields less than 10% latency improvement.
                    </div>
                    <div>
                        <strong>3. The deployment spectrum spans ~{_compute_ratio:,.0f}x in compute.</strong>
                        ResNet-50 is infeasible on ESP32 by ~{_mem_ratio:.0f}x in memory.
                        Each deployment tier requires architecturally different models --
                        the tiers exist because physics creates discrete constraints.
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
                        <strong>Lab 02: The Physics of Deployment</strong> -- deepens the Iron Law.
                        Part A shows why a $2M H100 upgrade yielded only 8% latency improvement.
                        Part B shows where the speed of light makes cloud inference physically
                        impossible.
                    </div>
                </div>
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        Textbook Connection
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        <strong>Read:</strong> the Introduction chapter for the D-A-M framework,
                        the Iron Law section (Ch. 1) for the full Iron Law derivation,
                        the Deployment Spectrum section (Ch. 1) for the hardware tier table.
                        <br/><strong>Build:</strong> TinyTorch Module 01 -- implement the D-A-M diagnostic framework and Iron Law calculator from scratch.
                    </div>
                </div>
            </div>
            """),

            mo.accordion({
                "Self-Assessment: Can you answer these?": mo.md("""
1. A vision system has 94% accuracy but 340 ms latency (SLA: 100 ms). Which D-A-M axis is binding?
2. Why does doubling H100 compute yield <10% latency improvement for ResNet-50 at batch=1?
3. ResNet-50 requires ~49 MB in FP16. The ESP32 has 512 KB. Can INT8 quantization fix this?

*If you cannot answer all three from memory, revisit Parts A, B, and C.*
""")
            }),

            mo.md("---"),
            mo.md("### Decision Log"),
            mo.md("Record the single most important insight from this lab. "
                   "This entry carries forward to Lab 02 and beyond via the Design Ledger."),
            synth_decision_ui,
        ])

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A -- Three Axes, One System":    build_part_a(),
        "Part B -- The Iron Law Surprise":     build_part_b(),
        "Part C -- The Triad Across Targets":  build_part_c(),
        "Part D -- The Deployment Spectrum":    build_part_d(),
        "Synthesis":                            build_synthesis(),
    })
    tabs
    return


# ═══════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════


# ─── CELL 5: LEDGER HUD ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, ledger, mo):
    _chapter = ledger._state.history.get(1, {})
    _track = ledger._state.track or "not set"

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">01 &middot; The AI Triad</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">CHAPTER&nbsp;1</span>
        <span class="hud-value">Introduction to ML Systems</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
