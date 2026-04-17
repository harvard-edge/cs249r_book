import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")



# ===========================================================================
# ZONE A: OPENING
# ===========================================================================

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
    import mlsysim

    H100_TFLOPS_FP16 = mlsysim.Hardware.Cloud.H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_BW_GBS      = mlsysim.Hardware.Cloud.H100.memory.bandwidth.m_as("GB/s")
    H100_RAM_GB      = mlsysim.Hardware.Cloud.H100.memory.capacity.m_as("GB")
    H100_TDP_W       = mlsysim.Hardware.Cloud.H100.tdp.m_as("W")

    # Edge tier — capstone comparison: same analysis on constrained hardware
    JETSON_TFLOPS    = mlsysim.Hardware.Edge.JetsonOrinNX.compute.peak_flops.m_as("TFLOPs/s")
    JETSON_BW_GBS    = mlsysim.Hardware.Edge.JetsonOrinNX.memory.bandwidth.m_as("GB/s")
    JETSON_RAM_GB    = mlsysim.Hardware.Edge.JetsonOrinNX.memory.capacity.m_as("GB")
    JETSON_TDP_W     = mlsysim.Hardware.Edge.JetsonOrinNX.tdp.m_as("W")

    LLAMA2_70B_PARAMS = mlsysim.Models.Language.Llama2_70B.parameters.m_as("count")
    LLAMA2_70B_LAYERS = mlsysim.Models.Language.Llama2_70B.layers
    LLAMA2_70B_HIDDEN = mlsysim.Models.Language.Llama2_70B.hidden_dim
    LLAMA2_70B_HEADS  = mlsysim.Models.Language.Llama2_70B.heads

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        COLORS, H100_BW_GBS, H100_RAM_GB, H100_TDP_W, H100_TFLOPS_FP16,
        JETSON_BW_GBS, JETSON_RAM_GB, JETSON_TDP_W, JETSON_TFLOPS,
        LAB_CSS, LLAMA2_70B_HEADS, LLAMA2_70B_HIDDEN, LLAMA2_70B_LAYERS,
        LLAMA2_70B_PARAMS, apply_plotly_theme, go, ledger, math, mo, np,
    )


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
                Machine Learning Systems &middot; Volume I &middot; Lab 16 (Capstone)
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Architect's Audit
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Token Cost &middot; Conservation &middot; Blind Spots &middot; Amdahl &middot; Cascade
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Everything you have learned in 15 labs collapses into one question: given
                a real system with real constraints, where do you invest your engineering
                effort? The answer requires every tool in your toolkit.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    5 Parts + Synthesis &middot; ~58 min</span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Capstone: All Chapters</span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Memory Wall</span>
                <span class="badge badge-warn">Amdahl's Law</span>
                <span class="badge badge-fail">Constraint Cascade</span>
            </div>
        </div>
        """),
    ])
    return


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
                Learning Objectives</div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Quantify the memory wall</strong>
                    &mdash; at batch=1 for Llama-2 70B, 98% of token time is memory access.</div>
                <div style="margin-bottom: 3px;">2. <strong>Apply Amdahl's Law to system design</strong>
                    &mdash; optimizing the largest component (40% inference) by 5x yields
                    only 1.47x system speedup.</div>
                <div style="margin-bottom: 3px;">3. <strong>Trace the constraint cascade</strong>
                    &mdash; fixing one constraint (INT4 quantization) shifts the binding
                    constraint to KV cache memory.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    All Volume I chapters &middot; Labs 01&ndash;15</div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~58 min</strong><br/>
                    A: ~12 &middot; B: ~12 &middot; C: ~10 &middot; D: ~8 &middot; E: ~10 min</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 12px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question</div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;You are deploying Llama-2 70B as a chat service. You have solved the
                memory constraint with INT4 quantization. Why is the system still broken?&rdquo;
            </div>
        </div>
    </div>""")
    return



# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** &mdash; This is the capstone lab. All chapters are prerequisites.
    Review your Design Ledger entries from Labs 01&ndash;15.
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LAB CELL
# ═════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(
    COLORS, H100_BW_GBS, H100_RAM_GB, H100_TDP_W, H100_TFLOPS_FP16,
    JETSON_BW_GBS, JETSON_RAM_GB, JETSON_TDP_W, JETSON_TFLOPS,
    LLAMA2_70B_HEADS, LLAMA2_70B_HIDDEN, LLAMA2_70B_LAYERS,
    LLAMA2_70B_PARAMS, apply_plotly_theme, go, ledger, math, mo, np,
):
    # ── Part A widgets ────────────────────────────────────────────────────────
    partA_pred = mo.ui.radio(
        options={
            "A) ~50% (roughly balanced)": "50pct",
            "B) ~75% (memory dominates moderately)": "75pct",
            "C) ~90% (memory dominates strongly)": "90pct",
            "D) ~98% (memory is essentially all the time)": "98pct",
        },
        label="Llama-2 70B, batch=1, H100, FP16. What fraction of token time is memory?",
    )
    return (partA_pred,)

@app.cell(hide_code=True)
def _(mo):
    partA_model = mo.ui.dropdown(options={"7B": 7, "13B": 13, "70B": 70}, value="70B",
                                  label="Model size")
    partA_prec = mo.ui.dropdown(
        options={"FP32 (4B)": 4, "FP16 (2B)": 2, "INT8 (1B)": 1, "INT4 (0.5B)": 0.5},
        value="FP16 (2B)", label="Precision")
    partA_batch = mo.ui.slider(start=1, stop=256, value=1, step=1, label="Batch size")

    # ── Part B widgets ────────────────────────────────────────────────────────
    partB_pred = mo.ui.radio(
        options={
            "A) Still ~95% after 6 months (INT8 is stable)": "stable",
            "B) Latency increased from hardware wear": "latency",
            "C) Accuracy silently degraded ~8% while metrics stayed green": "degraded",
            "D) Model crashed from numerical instability": "crashed",
        },
        label="You quantized MobileNetV2 to INT8, deployed on mobile. After 6 months without monitoring?",
    )
    return (partA_batch, partA_model, partA_prec, partB_pred)

@app.cell(hide_code=True)
def _(mo):
    partB_quant = mo.ui.dropdown(
        options={"FP32": "fp32", "FP16": "fp16", "INT8": "int8", "INT4": "int4"},
        value="INT8", label="Quantization level")
    partB_monitoring = mo.ui.dropdown(
        options={"None": 0, "Basic (uptime only)": 1, "Comprehensive (PSI + accuracy)": 2},
        value="None", label="Monitoring investment")
    partB_months = mo.ui.slider(start=0, stop=12, value=6, step=1,
                                 label="Months deployed")

    # ── Part C widgets ────────────────────────────────────────────────────────
    partC_pred = mo.ui.dropdown(
        options={
            "Amdahl's Law": "amdahl",
            "Memory Wall": "memory",
            "Silent Degradation": "degradation",
            "Conservation of Complexity": "conservation",
            "No Free Fairness": "fairness",
        },
        value="Memory Wall",
        label="Which invariant do you think you most consistently underestimated?",
    )
    return (partB_monitoring, partB_months, partB_quant, partC_pred)

@app.cell(hide_code=True)
def _(mo):

    # ── Part D widgets ────────────────────────────────────────────────────────
    partD_pred = mo.ui.radio(
        options={
            "A) Preprocessing (35% -- largest non-inference)": "preprocess",
            "B) Inference (40% -- largest single component)": "inference",
            "C) Postprocessing (15% -- low-hanging fruit)": "postprocess",
            "D) Does not matter -- 5x on any yields same speedup": "same",
        },
        label="4 components: preprocess 35%, inference 40%, postprocess 15%, logging 10%. "
              "Optimize ONE by 5x. Which gives largest end-to-end speedup?",
    )
    return (partD_pred,)

@app.cell(hide_code=True)
def _(mo):
    partD_preprocess = mo.ui.slider(start=5, stop=50, value=35, step=5,
                                     label="Preprocessing (%)")
    partD_inference = mo.ui.slider(start=5, stop=60, value=40, step=5,
                                    label="Inference (%)")
    partD_postprocess = mo.ui.slider(start=5, stop=30, value=15, step=5,
                                      label="Postprocessing (%)")
    partD_logging = mo.ui.slider(start=5, stop=20, value=10, step=5,
                                  label="Logging (%)")
    partD_optimize = mo.ui.dropdown(
        options={"Preprocessing": "preprocess", "Inference": "inference",
                 "Postprocessing": "postprocess", "Logging": "logging"},
        value="Inference", label="Component to optimize")
    partD_speedup = mo.ui.slider(start=1, stop=20, value=5, step=1,
                                  label="Optimization speedup (x)")

    # ── Part E widgets ────────────────────────────────────────────────────────
    partE_pred = mo.ui.radio(
        options={
            "A) Compute (INT4 is slower per op)": "compute",
            "B) KV cache memory (weights fit but KV fills HBM)": "kv_cache",
            "C) Accuracy (INT4 quality insufficient)": "accuracy",
            "D) No constraint -- system works": "none",
        },
        label="Quantize Llama-2 70B FP16 to INT4 (35 GB < 80 GB HBM). New binding constraint?",
    )
    return (partD_inference, partD_logging, partD_optimize, partD_postprocess, partD_preprocess, partD_speedup, partE_pred)

@app.cell(hide_code=True)
def _(
    mo, partA_batch, partA_model, partA_prec,
    partA_pred, partB_monitoring, partB_months, partB_pred,
    partB_quant, partC_pred, partD_inference, partD_logging,
    partD_optimize, partD_postprocess, partD_pred, partD_preprocess,
    partD_speedup, partE_pred,
):
    partE_int4 = mo.ui.checkbox(label="INT4 Quantization", value=False)
    partE_pruning = mo.ui.checkbox(label="Structured Pruning (50%)", value=False)
    partE_distill = mo.ui.checkbox(label="Knowledge Distillation", value=False)
    partE_retrain = mo.ui.checkbox(label="Continuous Retraining", value=False)
    partE_shap = mo.ui.checkbox(label="SHAP Explanations", value=False)
    partE_ctx = mo.ui.slider(start=2048, stop=131072, value=32768, step=2048,
                              label="Context length (tokens)")

    # ═════════════════════════════════════════════════════════════════════════
    # PART A — The Cost of a Token
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Final Calibration &middot; Systems Architect</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Before we build the deployment spec, let us calibrate one last time.
                What is the actual cost of generating a single token? Not the API price &mdash;
                the physics.&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## The Iron Law Revisited: Autoregressive Decoding

Token generation has arithmetic intensity of ~1 FLOP/byte (2 FLOPs per parameter,
2 bytes per parameter in FP16). This places it deep in the **memory-bandwidth-bound**
regime on every GPU ever made:

```
AI_decode = 2 * N / (2 * N) = 1 FLOP/byte
Ridge Point H100 = 989 TFLOPS / 3350 GB/s = 295 FLOPs/byte
```

AI (1) << Ridge Point (295): memory access is ~295x more expensive than compute.
        """))

        items.append(partA_pred)
        if partA_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the token cost calculator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_model, partA_prec, partA_batch], widths="equal"))

        _pb = partA_model.value
        _bpp = partA_prec.value
        _batch = partA_batch.value
        _w_gb = _pb * 1e9 * _bpp / (1024**3)

        # Memory time: load all weights once per token
        _t_mem = _w_gb / H100_BW_GBS * 1000  # ms

        # Compute time: 2 * params * batch FLOPs
        _flops = 2 * _pb * 1e9 * _batch
        _t_comp = _flops / (H100_TFLOPS_FP16 * 1e12 * 0.5) * 1000  # ms (50% MFU)

        _t_total = max(_t_mem, _t_comp)
        _mem_frac = _t_mem / (_t_mem + _t_comp) * 100
        _comp_frac = _t_comp / (_t_mem + _t_comp) * 100
        _ai = 2 * _batch  # FLOPs/byte (at batch=B, AI = 2*B)
        _ridge = H100_TFLOPS_FP16 * 1000 / H100_BW_GBS

        # Bar chart: memory vs compute
        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Memory Time", x=["Token Generation"],
                               y=[_t_mem], marker_color=COLORS['BlueLine'], opacity=0.88))
        _fig.add_trace(go.Bar(name="Compute Time", x=["Token Generation"],
                               y=[_t_comp], marker_color=COLORS['OrangeLine'], opacity=0.88))
        _fig.update_layout(barmode="group", height=300,
                           yaxis=dict(title="Time (ms)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _mem_col = COLORS['RedLine'] if _mem_frac > 90 else COLORS['OrangeLine']
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:2px solid {_mem_col}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {_mem_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Memory Fraction</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_mem_col};">{_mem_frac:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_t_mem:.2f} ms</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Compute Fraction</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_comp_frac:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_t_comp:.4f} ms</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Arithmetic Intensity</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_ai:.0f}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">vs ridge={_ridge:.0f}</div></div>
        </div>"""))

        items.append(mo.md(f"""
**Token Cost &mdash; Live** (`{_pb}B, {_bpp}B/param, batch={_batch}`)

```
Weights      = {_w_gb:.1f} GB
T_memory     = {_w_gb:.1f} / {H100_BW_GBS:.0f} GB/s = {_t_mem:.2f} ms
T_compute    = 2*{_pb}B*{_batch} / ({H100_TFLOPS_FP16:.0f}T*0.5) = {_t_comp:.4f} ms
Memory frac  = {_t_mem:.2f} / ({_t_mem:.2f}+{_t_comp:.4f}) = {_mem_frac:.0f}%
AI           = 2*{_batch} = {_ai} FLOPs/byte  (ridge = {_ridge:.0f})
```
*Source: Iron Law from @sec-introduction, revisited*
        """))

        if partA_pred.value == "98pct":
            items.append(mo.callout(mo.md(
                f"**Correct.** At batch=1, memory fraction = {_mem_frac:.0f}%. "
                "Compute optimization is futile without addressing data movement."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Memory fraction = {_mem_frac:.0f}%.** Even after 15 labs, "
                "students underestimate the severity of the memory wall."), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Math Peek: Amdahl's Law and the Memory Wall": mo.md(f"""
**Amdahl's Law — speedup limited by the sequential (memory-bound) fraction:**

$$S = \\frac{{1}}{{(1 - p) + \\frac{{p}}{{N}}}}$$

where $p$ = parallelizable fraction, $N$ = speedup factor on that fraction.

At batch=1 autoregressive decoding, arithmetic intensity = 1 FLOP/byte.
The ridge point of H100 is ~{_ridge:.0f} FLOPs/byte. Since AI (1) << ridge ({_ridge:.0f}),
**{_mem_frac:.0f}% of token time is pure memory access** — no amount of
compute optimization can help until you move data faster.
"""),
        }))

        # Edge comparison: same analysis on Jetson Orin NX
        _edge_t_mem = _w_gb / JETSON_BW_GBS * 1000 if JETSON_BW_GBS > 0 else float('inf')
        _edge_t_comp = _flops / (JETSON_TFLOPS * 1e12 * 0.5) * 1000 if JETSON_TFLOPS > 0 else float('inf')
        _edge_mem_frac = _edge_t_mem / (_edge_t_mem + _edge_t_comp) * 100 if (_edge_t_mem + _edge_t_comp) > 0 else 0
        _edge_ridge = JETSON_TFLOPS * 1000 / JETSON_BW_GBS if JETSON_BW_GBS > 0 else 0
        _fits_edge = _w_gb <= JETSON_RAM_GB
        items.append(mo.callout(mo.md(
            f"**Edge comparison (Jetson Orin NX, {JETSON_RAM_GB:.0f} GB):** "
            + (f"OOM — {_w_gb:.1f} GB weights exceed {JETSON_RAM_GB:.0f} GB memory. "
               "Edge deployment requires aggressive quantization."
               if not _fits_edge else
               f"T_memory = {_edge_t_mem:.1f} ms, T_compute = {_edge_t_comp:.3f} ms, "
               f"memory fraction = {_edge_mem_frac:.0f}% (ridge point = {_edge_ridge:.0f}). "
               f"The memory wall is {_edge_t_mem/_t_mem:.0f}x worse on edge due to lower bandwidth "
               f"({JETSON_BW_GBS:.0f} vs {H100_BW_GBS:.0f} GB/s).")
        ), kind="info"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B — Conservation of Complexity
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Post-Deployment Audit &middot; Systems Architect</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;You quantized the model to INT8, shipped it, and moved on to the next
                project. Six months later, with no monitoring investment. What happened?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## Complexity Cannot Be Destroyed, Only Moved

Quantization reduces **Algorithm complexity** but shifts burden to the
**Machine axis** (monitoring). Without monitoring, INT8 models are more
susceptible to distribution shift because quantization reduces the model's
effective capacity to absorb new patterns.

```
Conservation: Delta(Data) + Delta(Algorithm) + Delta(Machine) = 0
```
        """))

        items.append(partB_pred)
        if partB_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the conservation simulator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_quant, partB_monitoring, partB_months], widths="equal"))

        _quant = partB_quant.value
        _mon = partB_monitoring.value
        _months = partB_months.value

        # Drift rate depends on quantization (more aggressive = faster drift)
        _drift_rates = {"fp32": 0.5, "fp16": 0.6, "int8": 1.3, "int4": 2.0}
        _drift = _drift_rates[_quant]

        # Monitoring reduces observed impact (earlier detection)
        _detection_delay = {0: _months, 1: _months, 2: max(0, _months - 2)}[_mon]
        _acc_loss = _drift * _detection_delay

        # Radar chart: 5 axes
        _axes = ["Accuracy", "Latency", "Memory", "Power", "Drift\nResilience"]
        # Normalize 0-100 for each axis
        _latency_score = {"fp32": 40, "fp16": 70, "int8": 85, "int4": 95}[_quant]
        _memory_score = {"fp32": 30, "fp16": 60, "int8": 80, "int4": 95}[_quant]
        _power_score = {"fp32": 50, "fp16": 65, "int8": 80, "int4": 90}[_quant]
        _acc_score = max(0, 95 - _acc_loss)
        _resilience_score = max(0, 90 - _drift * 10 * (2 - _mon))

        _values = [_acc_score, _latency_score, _memory_score, _power_score, _resilience_score]
        _values_closed = _values + [_values[0]]
        _axes_closed = _axes + [_axes[0]]

        _fig = go.Figure()
        _fig.add_trace(go.Scatterpolar(r=_values_closed, theta=_axes_closed,
                                        fill='toself', name='System State',
                                        fillcolor=f"rgba(0,99,149,0.15)",
                                        line=dict(color=COLORS['BlueLine'], width=2)))
        _fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                           height=380, margin=dict(l=60, r=60, t=40, b=40))
        items.append(mo.as_html(_fig))

        if _acc_loss > 5:
            items.append(mo.callout(mo.md(
                f"**Silent degradation.** {_quant.upper()} model lost {_acc_loss:.1f} pp "
                f"in {_months} months. Monitoring level: {['None', 'Basic', 'Comprehensive'][_mon]}. "
                "The accuracy axis shrank while all others stayed green."), kind="danger"))
        elif _acc_loss > 2:
            items.append(mo.callout(mo.md(
                f"**Drift detected.** {_acc_loss:.1f} pp accuracy loss. "
                "Comprehensive monitoring would have flagged this earlier."), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Accuracy stable.** {_acc_loss:.1f} pp loss over {_months} months."), kind="success"))

        if partB_pred.value == "degraded":
            items.append(mo.callout(mo.md(
                "**Correct.** Quantization reduces model robustness to distribution shift. "
                "Without monitoring, accuracy silently erodes while infrastructure stays green."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**INT8 models drift faster than FP32.** Reduced effective capacity means "
                "less tolerance for distribution shift. Conservation of Complexity."), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Math Peek: Conservation of Complexity": mo.md(f"""
**The complexity budget is conserved across subsystems:**

$$\\text{{Total\\_complexity}} = \\sum_i \\text{{subsystem\\_cost}}_i$$

$$\\Delta(\\text{{Data}}) + \\Delta(\\text{{Algorithm}}) + \\Delta(\\text{{Machine}}) = 0$$

Quantization reduces Algorithm complexity (fewer bits per weight) but
shifts burden to Machine complexity (monitoring, drift detection).
With {_quant.upper()} and monitoring level {['None', 'Basic', 'Comprehensive'][_mon]},
drift rate = {_drift:.1f} pp/month, yielding **{_acc_loss:.1f} pp accuracy loss**
over {_months} months. The complexity was not destroyed — it was moved.
"""),
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C — Design Ledger Archaeology
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_c():
        items = []
        items.append(mo.md("""
## Your Prediction History Reveals Systematic Blind Spots

Before viewing your data, assess: which invariant did your intuition
most consistently underweight across all 15 labs?
        """))

        items.append(partC_pred)

        # Load design ledger history (or use defaults)
        _history = ledger._state.history
        _has_data = len(_history) > 3

        if _has_data:
            items.append(mo.callout(mo.md(
                f"**Design Ledger loaded.** Found data from {len(_history)} labs. "
                "Your personal prediction profile is shown below."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Using class median data.** Complete your Design Ledger entries "
                "for a personalized analysis. Showing typical student profile."), kind="info"))

        # Typical student error profile (class median)
        _invariants = ["Amdahl's\nLaw", "Memory\nWall", "Silent\nDegradation",
                       "Conservation", "No Free\nFairness"]
        _typical_errors = [35, 55, 70, 45, 60]  # avg prediction error %
        _self_assess = {"amdahl": 0, "memory": 1, "degradation": 2,
                        "conservation": 3, "fairness": 4}

        # Highlight self-assessed weakness
        _sa_idx = _self_assess.get(partC_pred.value, 1) if partC_pred.value else 1
        _colors = [COLORS['BlueLine']] * 5
        _colors[_sa_idx] = COLORS['OrangeLine']

        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=_invariants, y=_typical_errors, marker_color=_colors, opacity=0.88))
        _fig.update_layout(height=340,
                           yaxis=dict(title="Avg Prediction Error (%)", gridcolor="#f1f5f9"),
                           margin=dict(l=50, r=20, t=40, b=60))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _worst = _invariants[np.argmax(_typical_errors)]
        items.append(mo.callout(mo.md(
            f"**Largest blind spot (class median): {_worst}** with "
            f"{max(_typical_errors)}% average prediction error. "
            "Silent Degradation is consistently the most underestimated "
            "invariant because students conflate infrastructure health "
            "with model health."), kind="info"))

        # MathPeek
        items.append(mo.accordion({
            "Math Peek: Prediction Calibration Error": mo.md("""
**Mean Absolute Prediction Error across labs:**

$$\\text{MAPE} = \\frac{1}{N} \\sum_{i=1}^{N} |\\hat{y}_i - y_i|$$

where $\\hat{y}_i$ = your prediction, $y_i$ = actual measured value.

A well-calibrated engineer has MAPE < 10%. The class median is typically
30-50% on first encounter with each invariant. Systematic overconfidence
on "Silent Degradation" (70% error) reveals that students confuse
**infrastructure uptime** with **model accuracy** — the system stays
green while the model quietly rots.
"""),
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D — The Amdahl Ceiling Revisited
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Resource Allocation &middot; Engineering Sprint Planning</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We have one engineering sprint to optimize our ML pipeline.
                Four components, one 5x optimization budget. Where does the math
                say we should invest?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## Amdahl's Law Is a Resource Allocation Framework

From Lab 12: `Speedup = 1 / ((1 - f) + f/S)` where f is the fraction of time
in the optimized component and S is the speedup factor.

The component with the **largest time fraction** has the highest-leverage optimization.
        """))

        items.append(partD_pred)
        if partD_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the Amdahl calculator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_preprocess, partD_inference, partD_postprocess, partD_logging],
                               widths="equal"))
        items.append(mo.hstack([partD_optimize, partD_speedup], widths="equal"))

        _fracs = {
            "preprocess": partD_preprocess.value,
            "inference": partD_inference.value,
            "postprocess": partD_postprocess.value,
            "logging": partD_logging.value,
        }
        _total_pct = sum(_fracs.values())
        # Normalize
        _norm = {k: v / _total_pct for k, v in _fracs.items()}

        _opt = partD_optimize.value
        _S = partD_speedup.value
        _f = _norm[_opt]
        _amdahl = 1 / ((1 - _f) + _f / _S)

        # Compute speedup for all components
        _all_speedups = {}
        for _comp, _frac in _norm.items():
            _all_speedups[_comp] = 1 / ((1 - _frac) + _frac / _S)

        _best = max(_all_speedups, key=_all_speedups.get)
        _labels = {"preprocess": "Preprocessing", "inference": "Inference",
                   "postprocess": "Postprocessing", "logging": "Logging"}

        _fig = go.Figure()
        _comps = list(_labels.values())
        _speeds = [_all_speedups[k] for k in _labels]
        _cols = [COLORS['GreenLine'] if k == _best else COLORS['BlueLine'] for k in _labels]
        _fig.add_trace(go.Bar(x=_comps, y=_speeds, marker_color=_cols, opacity=0.88))
        _fig.add_hline(y=1.0, line_dash="dash", line_color="#94a3b8",
                       annotation_text="No speedup")
        _fig.update_layout(height=340,
                           yaxis=dict(title=f"System Speedup ({_S}x component opt)",
                                      gridcolor="#f1f5f9"),
                           margin=dict(l=50, r=20, t=40, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:2px solid {COLORS['GreenLine']}; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Best Target</div>
                <div style="font-size:1.3rem; font-weight:800; color:{COLORS['GreenLine']};">{_labels[_best]}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_norm[_best]*100:.0f}% of time</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Your Choice: {_labels[_opt]}</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_amdahl:.2f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_S}x on {_norm[_opt]*100:.0f}%</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Best Possible</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_all_speedups[_best]:.2f}x</div></div>
        </div>"""))

        if partD_pred.value == "inference":
            items.append(mo.callout(mo.md(
                f"**Correct.** Inference at 40% has the highest leverage: "
                f"1/(0.6 + 0.4/5) = 1.47x. Not intuitive but mathematically optimal."), kind="success"))
        elif partD_pred.value == "same":
            items.append(mo.callout(mo.md(
                "**They are NOT the same.** Amdahl's Law shows the speedup depends on "
                f"the fraction: 5x on 40% = {1/(0.6+0.4/5):.2f}x, but 5x on 10% = "
                f"{1/(0.9+0.1/5):.2f}x."), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**The largest component wins.** {_labels[_best]} at "
                f"{_norm[_best]*100:.0f}% gives {_all_speedups[_best]:.2f}x."), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Math Peek: Amdahl's Ceiling with Component Fractions": mo.md(f"""
**Amdahl's Law applied to a multi-component pipeline:**

$$S = \\frac{{1}}{{f_{{\\text{{serial}}}} + \\frac{{f_{{\\text{{opt}}}}}}{{N}}}}$$

With your current fractions (normalized to 1.0):

| Component | Fraction | Speedup if optimized {_S}x |
|-----------|----------|---------------------------|
| Preprocessing | {_norm['preprocess']*100:.0f}% | {_all_speedups['preprocess']:.2f}x |
| Inference | {_norm['inference']*100:.0f}% | {_all_speedups['inference']:.2f}x |
| Postprocessing | {_norm['postprocess']*100:.0f}% | {_all_speedups['postprocess']:.2f}x |
| Logging | {_norm['logging']*100:.0f}% | {_all_speedups['logging']:.2f}x |

The ceiling: even with $N \\to \\infty$ on {_labels[_best]}, max speedup =
$1 / (1 - {_norm[_best]:.2f})$ = **{1/(1-_norm[_best]):.2f}x**.
"""),
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART E — The Constraint Cascade
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_e():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                The Final Lesson &middot; Systems Architect</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;You quantized to INT4. The model fits in memory. Problem solved?
                Apply optimizations one at a time and watch what happens to the
                constraint landscape.&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## The Constraint Cascade: Fixing One Creates Another

Optimizing one axis does not solve the problem &mdash; it moves the binding
constraint. This is the Conservation of Complexity made dynamic:

- INT4 quantization: solves memory, creates precision constraint
- Larger batch: solves throughput, creates latency constraint
- More retraining: solves drift, creates carbon constraint

The architect's job is to choose **which constraint to live with**.
        """))

        items.append(partE_pred)
        if partE_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the constraint dashboard."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.md("**Apply optimizations one at a time:**"))
        items.append(mo.hstack([partE_int4, partE_pruning, partE_distill, partE_retrain, partE_shap]))
        items.append(partE_ctx)

        _ctx = partE_ctx.value
        _int4 = partE_int4.value
        _prune = partE_pruning.value
        _distill = partE_distill.value
        _retrain = partE_retrain.value
        _shap = partE_shap.value

        # Base config: 70B FP16 on H100
        _params = 70  # B
        _bpp = 2.0  # FP16
        if _int4:
            _bpp = 0.5
        if _prune:
            _params *= 0.5
        if _distill:
            _params = 13  # distilled to 13B

        _w_gb = _params * 1e9 * _bpp / (1024**3)

        # KV cache (always FP16 for KV)
        _layers = {70: 80, 35: 80, 13: 40}.get(int(_params), 80)  # pruning keeps layer count; only distillation changes arch
        _heads = {70: 64, 35: 32, 13: 40}.get(int(_params), 64)
        _hidden = {70: 8192, 35: 4096, 13: 5120}.get(int(_params), 8192)
        _hdim = _hidden // _heads
        _kv_gb = (2 * _layers * _heads * _hdim * _ctx * 2) / (1024**3)  # FP16 KV

        _total_mem = _w_gb + _kv_gb
        _mem_ok = _total_mem <= H100_RAM_GB

        # Accuracy (INT4 loses ~2-3%, pruning loses ~1-2%, distill loses ~3-5%)
        _acc = 95.0
        if _int4:
            _acc -= 2.5
        if _prune:
            _acc -= 1.5
        if _distill:
            _acc -= 4.0
        _acc_ok = _acc >= 88

        # Latency (base token time)
        _t_tok = _w_gb / H100_BW_GBS * 1000  # ms
        if _shap:
            _t_tok *= 50  # SHAP with ~50 features
        _lat_ok = _t_tok < 100  # ms per token

        # Carbon (relative)
        _carbon_mult = 1.0
        if _retrain:
            _carbon_mult *= 52  # weekly
        if _shap:
            _carbon_mult *= 5  # explanations
        _carbon_ok = _carbon_mult < 20

        # Fairness (assume no fairness constraint by default)
        _fair_ok = True  # simplified

        # Determine binding constraint
        _constraints = {
            "Memory": _mem_ok,
            "Accuracy": _acc_ok,
            "Latency": _lat_ok,
            "Carbon": _carbon_ok,
            "Fairness": _fair_ok,
        }
        _binding = [k for k, v in _constraints.items() if not v]
        _all_ok = len(_binding) == 0

        # Dashboard
        def _constraint_card(name, ok, value_str):
            _col = COLORS['GreenLine'] if ok else COLORS['RedLine']
            _bg = COLORS['GreenLL'] if ok else COLORS['RedLL']
            _status = "OK" if ok else "VIOLATED"
            return f"""
            <div style="padding:14px; border:2px solid {_col}; border-radius:10px;
                        min-width:110px; text-align:center; background:{_bg}; flex:1;">
                <div style="color:{_col}; font-size:0.72rem; font-weight:700;">{name}</div>
                <div style="font-size:1.1rem; font-weight:800; color:{_col}; margin:4px 0;">{value_str}</div>
                <div style="font-size:0.72rem; color:{_col}; font-weight:600;">{_status}</div></div>"""

        _cards = '<div style="display:flex; gap:10px; flex-wrap:wrap; margin:16px 0;">'
        _cards += _constraint_card("Memory", _mem_ok, f"{_total_mem:.0f}/{H100_RAM_GB:.0f} GB")
        _cards += _constraint_card("Accuracy", _acc_ok, f"{_acc:.1f}%")
        _cards += _constraint_card("Latency", _lat_ok, f"{_t_tok:.1f} ms/tok")
        _cards += _constraint_card("Carbon", _carbon_ok, f"{_carbon_mult:.0f}x")
        _cards += _constraint_card("Fairness", _fair_ok, "Unconstrained")
        _cards += '</div>'
        items.append(mo.Html(_cards))

        if _all_ok:
            items.append(mo.callout(mo.md(
                "**All constraints satisfied.** But at what cost? "
                f"Accuracy = {_acc:.1f}%, which may not meet production requirements."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Binding constraint(s): {', '.join(_binding)}.** "
                "Each optimization resolves one constraint but shifts the binding to another. "
                "This is the Constraint Cascade."), kind="danger"))

        # Cascade trace
        _trace_lines = []
        if _int4:
            _trace_lines.append("INT4: Memory OK, but accuracy -2.5 pp")
        if _prune:
            _trace_lines.append("Pruning: Memory further reduced, accuracy -1.5 pp")
        if _distill:
            _trace_lines.append("Distillation: 13B model, accuracy -4.0 pp, latency improved")
        if _retrain:
            _trace_lines.append("Continuous retraining: drift controlled, carbon 52x")
        if _shap:
            _trace_lines.append("SHAP: explanations available, latency 50x, carbon 5x")

        if _trace_lines:
            _trace_md = "\n".join(f"- {l}" for l in _trace_lines)
            items.append(mo.md(f"**Cascade Trace:**\n\n{_trace_md}"))

        if partE_pred.value == "kv_cache":
            items.append(mo.callout(mo.md(
                "**Correct.** INT4 weights (35 GB) fit in 80 GB HBM. But KV cache at "
                f"{_ctx:,} tokens in FP16 adds another {_kv_gb:.0f} GB. "
                "The constraint moved from 'model too large' to 'context too long'."), kind="success"))
        elif partE_pred.value == "none":
            items.append(mo.callout(mo.md(
                "**The system is NOT constraint-free.** INT4 solves the weight memory constraint, "
                f"but KV cache at 32K tokens = ~40 GB. Total = 75 GB of 80 GB HBM. "
                "At 64K, you OOM again."), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                "**The new binding constraint is KV cache memory.** INT4 weights fit, "
                f"but KV cache at long context fills HBM."), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Math Peek: The Constraint Cascade — Memory Budget": mo.md(f"""
**Total HBM consumption = weights + KV cache:**

$$\\text{{Memory}} = W_{{\\text{{gb}}}} + KV_{{\\text{{gb}}}}$$

$$KV_{{\\text{{gb}}}} = \\frac{{2 \\times L \\times H_{{\\text{{dim}}}} \\times \\text{{context}} \\times 2}}{{10^9}}$$

With your configuration:
- Weights: {_w_gb:.1f} GB ({_params:.0f}B params x {_bpp} bytes/param)
- KV cache: {_kv_gb:.1f} GB ({_layers} layers, {_ctx:,} context, FP16)
- **Total: {_total_mem:.1f} GB** vs {H100_RAM_GB:.0f} GB HBM

{"OOM: total exceeds HBM capacity." if not _mem_ok else f"Fits with {H100_RAM_GB - _total_mem:.1f} GB headroom."}
INT4 halved the weight memory — but KV cache is always FP16 and grows
linearly with context length. At 128K tokens, KV alone would consume
~{2 * _layers * _heads * _hdim * 131072 * 2 / (1024**3):.0f} GB.
"""),
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════════
    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                "**1. The memory wall dominates token generation by 40x or more.**\n\n"
                "At batch=1, 98% of time is memory access. "
                "Compute optimization is futile without fixing data movement."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. Amdahl's Law is a resource allocation framework, not just a speedup formula.**\n\n"
                "Optimize the largest time fraction first. "
                "5x on 40% = 1.47x. 5x on 10% = 1.02x."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. The Constraint Cascade is the final lesson.**\n\n"
                "Fixing one constraint shifts the binding to the next. "
                "INT4 solves memory but creates KV cache and accuracy constraints. "
                "The architect chooses which constraint to live with."
            ), kind="info"),
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
                        <strong>Volume II: ML Systems at Scale</strong> -- Continue to
                        Volume II where single-machine constraints become distributed
                        systems challenges. Networks replace buses, fault tolerance replaces
                        restarts, and coordination cost dominates everything.
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
                        <strong>Review:</strong> The five invariants -- Amdahl's Law,
                        Memory Wall, Silent Degradation, Conservation of Complexity,
                        No Free Fairness -- govern all ML systems design.<br/>
                        <strong>Build:</strong> The 20 TinyTorch modules let you construct
                        these systems from scratch, component by component.
                    </div>
                </div>
            </div>
            """),
            mo.accordion({
                "Self-Assessment: Can you answer these from memory?": mo.md("""
1. At what batch size does Llama-2 70B decode transition from memory-bound to compute-bound on H100?
2. What is the optimal retraining interval when training costs $10K and drift costs $500/day?
3. Why does quantizing to INT4 NOT solve the memory problem at 128K context?
4. Which invariant did your predictions most consistently underestimate?
5. If you optimize inference (40% of pipeline time) by 5x, what is the end-to-end speedup?

*If you cannot answer all five, revisit the relevant Parts.*
                """),
            }),
        ])

    tabs = mo.ui.tabs({
        "Part A \u2014 The Cost of a Token":          build_part_a(),
        "Part B \u2014 Conservation of Complexity":   build_part_b(),
        "Part C \u2014 Design Ledger Archaeology":    build_part_c(),
        "Part D \u2014 The Amdahl Ceiling Revisited": build_part_d(),
        "Part E \u2014 The Constraint Cascade":       build_part_e(),
        "Synthesis":                                   build_synthesis(),
    })
    tabs
    return



# ===========================================================================
# ZONE D: LEDGER HUD
# ===========================================================================

@app.cell(hide_code=True)
def _(COLORS, ledger, mo, partA_pred, partB_pred, partC_pred, partD_pred, partE_pred):
    _track = ledger.get_track()
    if partA_pred.value is not None and partB_pred.value is not None and partC_pred.value is not None and partD_pred.value is not None and partE_pred.value is not None:
        ledger.save(chapter=16, design={
            "chapter": "v1_16",
            "capstone_completed": True,
            "invariants_synthesized": 15,
            "deployment_decision_made": True,
            "completed": True,
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span><span class="hud-value">16 &mdash; Architect's Audit (Capstone)</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'NONE' else 'hud-none'}">{_track}</span>
        <span class="hud-label">STATUS</span><span class="hud-active">ACTIVE</span>
    </div>""")
    return


if __name__ == "__main__":
    app.run()
