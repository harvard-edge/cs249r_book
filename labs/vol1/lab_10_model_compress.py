import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 0: SETUP
# ═════════════════════════════════════════════════════════════════════════════

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

    H100_TFLOPS = mlsysim.Hardware.Cloud.H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_BW     = mlsysim.Hardware.Cloud.H100.memory.bandwidth.m_as("GB/s")
    H100_RAM    = mlsysim.Hardware.Cloud.H100.memory.capacity.m_as("GB")
    H100_TDP    = mlsysim.Hardware.Cloud.H100.tdp.m_as("W")

    IPHONE_TFLOPS = mlsysim.Hardware.Mobile.iPhone15Pro.compute.peak_flops.m_as("TFLOPs/s")
    IPHONE_BW     = mlsysim.Hardware.Mobile.iPhone15Pro.memory.bandwidth.m_as("GB/s")
    IPHONE_RAM    = mlsysim.Hardware.Mobile.iPhone15Pro.memory.capacity.m_as("GB")
    IPHONE_TDP    = mlsysim.Hardware.Mobile.iPhone15Pro.tdp.m_as("W")

    JETSON_TFLOPS = mlsysim.Hardware.Edge.JetsonOrinNX.compute.peak_flops.m_as("TFLOPs/s")
    JETSON_BW     = mlsysim.Hardware.Edge.JetsonOrinNX.memory.bandwidth.m_as("GB/s")
    JETSON_RAM    = mlsysim.Hardware.Edge.JetsonOrinNX.memory.capacity.m_as("GB")

    RESNET50_PARAMS = mlsysim.Models.ResNet50.parameters.m_as("count")
    RESNET50_FLOPS  = mlsysim.Models.ResNet50.inference_flops.m_as("flop")
    MOBILENET_PARAMS = mlsysim.Models.MobileNetV2.parameters.m_as("count")
    MOBILENET_FLOPS  = mlsysim.Models.MobileNetV2.inference_flops.m_as("flop")
    LLAMA3_8B_PARAMS = mlsysim.Models.Llama3_8B.parameters.m_as("count")

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        COLORS, H100_BW, H100_RAM, H100_TDP, H100_TFLOPS,
        IPHONE_BW, IPHONE_RAM, IPHONE_TDP, IPHONE_TFLOPS,
        JETSON_BW, JETSON_RAM, JETSON_TFLOPS,
        LAB_CSS, LLAMA3_8B_PARAMS,
        MOBILENET_FLOPS, MOBILENET_PARAMS,
        RESNET50_FLOPS, RESNET50_PARAMS,
        apply_plotly_theme, go, ledger, math, mo, np,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CELL 1: HEADER
# ═════════════════════════════════════════════════════════════════════════════
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
                Machine Learning Systems &middot; Volume I &middot; Lab 10
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Compression Paradox
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Quantization &middot; Pruning &middot; Pareto &middot; Energy &middot; Distillation
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Compression looks like free performance &mdash; until the hardware ignores
                your zeros, the accuracy cliff is vertical, and the only reliable shortcut
                is stealing knowledge from a model too large to deploy.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    5 Parts &middot; ~56 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter 10: Model Compression
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Quantization Free Lunch</span>
                <span class="badge badge-fail">Pruning Hardware Trap</span>
                <span class="badge badge-warn">Dark Knowledge</span>
            </div>
        </div>
        """),
    ])
    return


# ═════════════════════════════════════════════════════════════════════════════
# CELL 2: BRIEFING
# ═════════════════════════════════════════════════════════════════════════════
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the quantization free lunch</strong> &mdash;
                    FP32 to INT8 gives 4x compression with &lt;1% accuracy loss, then accuracy
                    collapses catastrophically below 4 bits.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose the pruning hardware trap</strong> &mdash;
                    90% unstructured sparsity yields zero speedup on dense GPU kernels because
                    the hardware cannot skip sparse multiplications.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a compression strategy</strong> &mdash; compose
                    quantization, pruning, and distillation along the Pareto frontier to fit
                    models across 8 GB, 4 GB, and 2 GB memory tiers.</div>
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
                    Neural network fundamentals from the Neural Computation chapter &middot;
                    Training and optimization from the Training chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~56 min</strong><br/>
                    A: ~12 &middot; B: ~12 &middot; C: ~12 &middot; D: ~8 &middot; E: ~12 min
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
                &ldquo;If removing 90% of a neural network's weights sounds like a 10x speedup,
                why does the GPU not notice?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ═════════════════════════════════════════════════════════════════════════════
# CELL 3: READING
# ═════════════════════════════════════════════════════════════════════════════

# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **Chapter 10: Model Compression** -- quantization levels, pruning types
      (structured vs. unstructured), knowledge distillation, and the Pareto frontier.
    - **Chapter 5: Neural Computation** -- weight representation and memory hierarchy.
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# CELL 4: TABS (Parts A-E + Synthesis)
# ═════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(
    COLORS, H100_BW, H100_RAM, H100_TDP, H100_TFLOPS,
    IPHONE_BW, IPHONE_RAM, IPHONE_TDP, IPHONE_TFLOPS,
    JETSON_BW, JETSON_RAM, JETSON_TFLOPS,
    LLAMA3_8B_PARAMS,
    MOBILENET_FLOPS, MOBILENET_PARAMS,
    RESNET50_FLOPS, RESNET50_PARAMS,
    apply_plotly_theme, go, math, mo, np,
):
    # ── Widgets ───────────────────────────────────────────────────────────
    pA_pred = mo.ui.radio(
        options={
            "A) ~5% (noticeable but tolerable)": "5",
            "B) ~2% (moderate degradation)": "2",
            "C) < 1% (essentially free)": "lt1",
            "D) ~0.1% (unmeasurable)": "01",
        },
        label="You quantize ResNet-50 from FP32 to INT8 (4x compression). "
              "How much accuracy do you lose on ImageNet?",
    )
    return (pA_pred,)

@app.cell(hide_code=True)
def _(mo, pA_pred):
    mo.stop(pA_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pA_precision = mo.ui.dropdown(
        options={"FP32": "fp32", "FP16": "fp16", "INT8": "int8", "INT4": "int4", "INT2": "int2"},
        value="FP32",
        label="Precision",
    )
    pA_model_sel = mo.ui.dropdown(
        options={"ResNet-50": "resnet", "MobileNetV2": "mobilenet"},
        value="ResNet-50",
        label="Model",
    )

    pB_pred = mo.ui.radio(
        options={
            "A) ~10x (removed 90% of work)": "10x",
            "B) ~5x (some overhead)": "5x",
            "C) ~2x (significant overhead)": "2x",
            "D) ~1.0x (no speedup at all)": "1x",
        },
        label="You prune ResNet-50 to 90% sparsity (unstructured). "
              "What inference speedup on an H100?",
    )
    return (pB_pred,)

@app.cell(hide_code=True)
def _(mo, pB_pred):
    mo.stop(pB_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pB_sparsity = mo.ui.slider(start=0, stop=95, value=0, step=5, label="Sparsity (%)")
    pB_type = mo.ui.dropdown(
        options={"Unstructured": "unstructured", "Structured": "structured", "2:4 Structured": "2_4"},
        value="Unstructured",
        label="Pruning type",
    )

    pC_pred = mo.ui.radio(
        options={
            "A) FP16 (16 GB) -- does not fit": "fp16",
            "B) INT8 (8 GB) -- does not fit": "int8",
            "C) INT4 (4 GB) -- fits with accuracy loss": "int4",
            "D) INT4 + 50% pruning (2 GB) -- fits but risky": "int4_prune",
        },
        label="Deploy Llama-3 8B on 4 GB RAM. Which strategy fits?",
    )
    return (pC_pred,)

@app.cell(hide_code=True)
def _(mo, pC_pred):
    mo.stop(pC_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pD_pred = mo.ui.radio(
        options={
            "A) 50/50 -- balanced": "50_50",
            "B) 70/30 -- memory dominates somewhat": "70_30",
            "C) 90/10 -- memory dominates strongly": "90_10",
            "D) 99/1 -- memory is essentially all the energy": "99_1",
        },
        label="MobileNetV2 inference on mobile: what fraction of energy is memory access vs. compute?",
    )
    return (pD_pred,)

@app.cell(hide_code=True)
def _(mo, pD_pred):
    mo.stop(pD_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pD_precision_e = mo.ui.dropdown(
        options={"FP32": "fp32", "FP16": "fp16", "INT8": "int8", "INT4": "int4"},
        value="FP32",
        label="Precision",
    )
    pD_hw = mo.ui.radio(
        options={"Cloud (H100)": "cloud", "Mobile (iPhone)": "mobile"},
        value="Mobile (iPhone)",
        label="Target:", inline=True,
    )

    pE_pred = mo.ui.radio(
        options={
            "A) ~65% (significant loss)": "65",
            "B) ~70% (moderate transfer)": "70",
            "C) ~73% (retains 95% of teacher)": "73",
            "D) ~76% (perfect transfer)": "76",
        },
        label="ResNet-50 teacher: 76.1% ImageNet. Distill to MobileNetV2 student. Student accuracy?",
    )
    return (pE_pred,)

@app.cell(hide_code=True)
def _(mo, pE_pred):
    mo.stop(pE_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pE_temp = mo.ui.slider(start=1.0, stop=20.0, value=4.0, step=0.5, label="Temperature")

    # ─────────────────────────────────────────────────────────────────────
    # PART A: Quantization Free Lunch
    # ─────────────────────────────────────────────────────────────────────
    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Edge Deployment Lead
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our ResNet-50 model is 100 MB in FP32. We need to fit it on a device with
                25 MB budget. The team says quantization to INT8 will destroy accuracy. Is that true?&rdquo;
            </div>
        </div>
        """))
        items.append(mo.md("""
        ## The Quantization Free Lunch

        Reducing precision from FP32 to INT8 costs **under 1% accuracy**, then accuracy
        collapses catastrophically at 3-4 bits. The curve is **flat-then-vertical**, not gradual.

        ```
        Model Size = Parameters x Bytes_per_Parameter
        FP32: 4 bytes   FP16: 2 bytes   INT8: 1 byte   INT4: 0.5 bytes
        ```

        INT8 has 256 discrete levels -- far more than needed for typical weight distributions
        that cluster tightly around zero.
        """))
        items.append(pA_pred)
        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the quantization explorer."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pA_precision, pA_model_sel], justify="start"))

        _prec = pA_precision.value
        _model = pA_model_sel.value
        _params = RESNET50_PARAMS if _model == "resnet" else MOBILENET_PARAMS
        _base_flops = RESNET50_FLOPS if _model == "resnet" else MOBILENET_FLOPS
        _model_label = "ResNet-50" if _model == "resnet" else "MobileNetV2"

        _bpp = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5, "int2": 0.25}
        _acc_table = {
            "resnet":    {"fp32": 76.1, "fp16": 76.0, "int8": 75.5, "int4": 71.0, "int2": 45.0},
            "mobilenet": {"fp32": 71.8, "fp16": 71.7, "int8": 71.2, "int4": 66.0, "int2": 38.0},
        }
        _cur_bpp = _bpp[_prec]
        _size_mb = _params * _cur_bpp / (1024 * 1024)
        _base_size = _params * 4 / (1024 * 1024)
        _compression = _base_size / _size_mb if _size_mb > 0 else 1
        _acc = _acc_table[_model][_prec]
        _acc_loss = _acc_table[_model]["fp32"] - _acc

        # Latency via roofline
        _eta = 0.5
        _t_compute = (_base_flops / (H100_TFLOPS * 1e12 * _eta)) * 1000
        _t_memory = (_size_mb / 1024 / H100_BW) * 1000
        _latency = max(_t_compute, _t_memory)

        # Bar chart: all precisions
        _precs = ["FP32", "FP16", "INT8", "INT4", "INT2"]
        _prec_keys = ["fp32", "fp16", "int8", "int4", "int2"]
        _sizes = [_params * _bpp[p] / (1024*1024) for p in _prec_keys]
        _accs = [_acc_table[_model][p] for p in _prec_keys]
        _colors = [COLORS["BlueLine"], COLORS["BlueLine"], COLORS["GreenLine"],
                   COLORS["OrangeLine"], COLORS["RedLine"]]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=_precs, y=_accs, marker_color=_colors, opacity=0.88,
            text=[f"{a:.1f}%" for a in _accs], textposition="outside",
        ))
        _fig.add_hline(y=_acc_table[_model]["fp32"] * 0.99, line_dash="dash",
                       line_color=COLORS["GreenLine"],
                       annotation_text="<1% loss threshold")
        _fig.update_layout(
            height=340, yaxis=dict(title="ImageNet Accuracy (%)", range=[30, 82]),
            showlegend=False,
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _cliff = _acc_loss > 5
        _acc_color = COLORS["RedLine"] if _cliff else (COLORS["GreenLine"] if _acc_loss < 1 else COLORS["OrangeLine"])
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Model Size</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_size_mb:.0f} MB</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_compression:.0f}x compression</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_acc_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Accuracy Loss</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_acc_color};">{_acc_loss:.1f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_acc:.1f}% vs {_acc_table[_model]['fp32']:.1f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Latency (H100)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_latency:.3f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_prec.upper()} precision</div>
            </div>
        </div>
        """))

        if _cliff:
            items.append(mo.callout(mo.md(
                f"**QUANTIZATION CLIFF.** At {_prec.upper()}, accuracy drops {_acc_loss:.1f}% -- "
                "the model has collapsed. Below 4 bits, the 256-level quantization grid becomes "
                "too coarse to represent the weight distribution faithfully."
            ), kind="danger"))

        items.append(mo.md(f"""
**Quantization -- Live Calculation** (`{_model_label}, {_prec.upper()}`)

```
Model size = {_params/1e6:.1f}M params x {_cur_bpp} bytes = {_size_mb:.0f} MB
Compression = {_base_size:.0f} MB / {_size_mb:.0f} MB = {_compression:.1f}x
Accuracy:  {_acc:.1f}% (loss: {_acc_loss:.1f}%)
```
*Source: Chapter 10, quantization accuracy table*
        """))

        if pA_pred.value == "lt1":
            items.append(mo.callout(mo.md("**Correct.** FP32 to INT8 loses under 1% accuracy. "
                "The representational capacity of 256 levels is sufficient for most weight distributions."), kind="success"))
        else:
            items.append(mo.callout(mo.md("**INT8 quantization is essentially free.** "
                "The accuracy loss is under 1%. Try selecting INT4 and INT2 to see the cliff."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Quantization Compression Ratio": mo.md("""
**Formula:**
$$
\\text{Compression Ratio} = \\frac{b_{\\text{original}}}{b_{\\text{quantized}}}
$$

Quantization error bound (per-tensor, symmetric):
$$
\\epsilon_{\\text{quant}} \\leq \\frac{\\Delta}{2} = \\frac{w_{\\max} - w_{\\min}}{2^{b+1}}
$$

**Variables:**
- **$b_{\\text{original}}$**: original bit-width (e.g., 32 for FP32)
- **$b_{\\text{quantized}}$**: target bit-width (e.g., 8 for INT8)
- **$\\Delta$**: quantization step size
- **$w_{\\max}, w_{\\min}$**: weight range extremes

FP32 to INT8 = 4x compression. Below 4 bits, $\\Delta$ grows large enough to collapse decision boundaries.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B: Pruning Hardware Trap
    # ─────────────────────────────────────────────────────────────────────
    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Performance Engineer
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We pruned 90% of the weights. The model file is 10x smaller. But inference
                is the same speed. I have checked the code three times. What am I missing?&rdquo;
            </div>
        </div>
        """))
        items.append(mo.md("""
        ## The Pruning Hardware Trap

        **Unstructured pruning** sets 90% of weights to zero but yields **zero speedup**
        on standard GPU kernels. Dense GEMM loads every element -- including zeros.

        - **Unstructured**: zeros scattered randomly. GPU still iterates all elements.
        - **Structured**: entire channels/filters removed. Smaller matrix = real speedup.
        - **2:4 Structured**: NVIDIA sparse tensor cores. 2x speedup, hardware-supported.

        The hardware cannot skip sparse multiplications without specialized support.
        """))
        items.append(pB_pred)
        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the pruning simulator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pB_sparsity, pB_type], justify="start"))

        _sparsity = pB_sparsity.value / 100.0
        _ptype = pB_type.value

        # Speedup model
        if _ptype == "unstructured":
            _speedup = 1.0  # no speedup on dense hardware
            _acc_loss = 0.3 * _sparsity  # gentle accuracy loss
        elif _ptype == "structured":
            _speedup = 1.0 / (1.0 - _sparsity) if _sparsity < 0.95 else 20.0
            _speedup = min(_speedup, 20.0)
            _acc_loss = 0.5 * _sparsity + 2.0 * max(0, _sparsity - 0.7)  # steeper at high sparsity
        else:  # 2:4
            _speedup = 2.0 if _sparsity >= 50 else 1.0 + _sparsity / 50.0
            _acc_loss = 0.2 * min(_sparsity, 0.5)

        _base_acc = 76.1
        _pruned_acc = max(0, _base_acc - _acc_loss * 100)
        _acc_collapse = _pruned_acc < 60

        # Chart: speedup vs sparsity for all three types
        _sparsities = np.linspace(0, 0.95, 40)
        _unstr_speed = [1.0] * len(_sparsities)
        _str_speed = [min(1.0/(1.0-s), 15) if s < 0.95 else 15 for s in _sparsities]
        _two4_speed = [2.0 if s >= 0.5 else 1.0 + s for s in _sparsities]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_sparsities*100, y=_unstr_speed, mode="lines",
            line=dict(color=COLORS["RedLine"], width=2), name="Unstructured"))
        _fig.add_trace(go.Scatter(x=_sparsities*100, y=_str_speed, mode="lines",
            line=dict(color=COLORS["GreenLine"], width=2), name="Structured"))
        _fig.add_trace(go.Scatter(x=(_sparsities*100).tolist(), y=_two4_speed, mode="lines",
            line=dict(color=COLORS["BlueLine"], width=2), name="2:4 Structured"))
        _fig.add_trace(go.Scatter(x=[_sparsity*100], y=[_speedup], mode="markers",
            marker=dict(size=14, color=COLORS["OrangeLine"], symbol="diamond"),
            name="Current"))
        _fig.update_layout(
            height=340, xaxis=dict(title="Sparsity (%)"),
            yaxis=dict(title="Inference Speedup (x)"),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _speed_color = COLORS["RedLine"] if _speedup < 1.5 else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_speed_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Speedup</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_speed_color};">{_speedup:.1f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_ptype} at {_sparsity*100:.0f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Accuracy</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_pruned_acc:.1f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">from {_base_acc}% baseline</div>
            </div>
        </div>
        """))

        if _ptype == "unstructured" and _sparsity > 0.5:
            items.append(mo.callout(mo.md(
                f"**HARDWARE TRAP.** {_sparsity*100:.0f}% of weights are zero, but speedup is {_speedup:.1f}x. "
                "Dense GEMM kernels iterate all elements including zeros. The zeros save no memory "
                "bandwidth and no compute. Without hardware sparse support, unstructured pruning is invisible."
            ), kind="danger"))

        if _acc_collapse:
            items.append(mo.callout(mo.md(
                f"**ACCURACY COLLAPSE.** Structured pruning at {_sparsity*100:.0f}% has destroyed "
                f"accuracy ({_pruned_acc:.1f}%). The model is no longer usable."
            ), kind="danger"))

        items.append(mo.md(f"""
**Pruning -- Live Calculation** (`{_ptype}, {_sparsity*100:.0f}% sparsity`)

```
Pruning type: {_ptype}
Sparsity:     {_sparsity*100:.0f}%
Speedup:      {_speedup:.1f}x  (unstructured = 1.0x always on dense kernels)
Accuracy:     {_pruned_acc:.1f}% (loss: {_base_acc - _pruned_acc:.1f}%)
```
*Source: Chapter 10, pruning types and hardware support*
        """))

        if pB_pred.value == "1x":
            items.append(mo.callout(mo.md("**Correct.** This is the most counterintuitive result in the lab. "
                "90% fewer weights = 0% speedup on dense GPU kernels. The hardware does not skip zeros."), kind="success"))
        else:
            items.append(mo.callout(mo.md("**The answer is 1.0x -- no speedup at all.** "
                "Dense GEMM loads and multiplies every element. Try switching to Structured or 2:4 "
                "to see real speedup."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Pruning Speedup on Dense vs. Sparse Hardware": mo.md("""
**Formula (theoretical vs. actual):**
$$
\\text{Speedup}_{\\text{theoretical}} = \\frac{1}{1 - s} \\qquad \\text{Speedup}_{\\text{dense GPU}} = 1.0
$$

For structured N:M sparsity (e.g., 2:4):
$$
\\text{Speedup}_{\\text{N:M}} = \\frac{M}{M - N} = \\frac{4}{2} = 2\\times
$$

**Variables:**
- **$s$**: sparsity fraction (e.g., 0.9 for 90% zeros)
- **$N$**: zeros per group (e.g., 2 of every 4 elements)
- **$M$**: group size

Dense GPU kernels cannot skip zero multiplications. Only structured sparsity with hardware support (e.g., Ampere 2:4) delivers real speedup.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C: Compression Pareto Frontier
    # ─────────────────────────────────────────────────────────────────────
    def build_part_c():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Mobile Deployment Team
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We need Llama-3 8B running on devices with 8 GB, 4 GB, and 2 GB RAM.
                Which compression strategies fit each tier? INT8? INT4? Or do we need to combine techniques?&rdquo;
            </div>
        </div>
        """))
        items.append(mo.md("""
        ## The Compression Pareto Frontier

        Deploying an 8B-parameter LLM across memory tiers requires composing
        techniques along a **Pareto frontier**. No single technique spans all tiers.

        ```
        Memory = Parameters x Bytes_per_param + KV_cache + Runtime_overhead
        FP16:  8B x 2 = 16 GB   (needs 8xGPU)
        INT8:  8B x 1 = 8 GB    (barely fits 8 GB device)
        INT4:  8B x 0.5 = 4 GB  (fits 4 GB with ~2% accuracy loss)
        ```

        Memory accounting must include KV cache, activations, and runtime overhead.
        """))
        items.append(pC_pred)
        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the Pareto explorer."), kind="warn"))
            return mo.vstack(items)

        _configs = [
            ("FP16", 16.0, 0.0, COLORS["BlueLine"]),
            ("INT8", 8.0, 0.8, COLORS["BlueLine"]),
            ("INT8 + 30% prune", 5.6, 2.5, COLORS["OrangeLine"]),
            ("INT4", 4.0, 2.2, COLORS["GreenLine"]),
            ("INT4 + 30% prune", 2.8, 4.5, COLORS["OrangeLine"]),
            ("INT4 + 50% prune", 2.0, 8.0, COLORS["RedLine"]),
            ("INT2", 1.0, 25.0, COLORS["RedLine"]),
        ]

        _fig = go.Figure()
        for _name, _size, _loss, _col in _configs:
            _fig.add_trace(go.Scatter(
                x=[_size], y=[_loss], mode="markers+text",
                marker=dict(size=12, color=_col),
                text=[_name], textposition="top center",
                name=_name, showlegend=False,
            ))
        # Memory tier lines
        for _tier, _gb in [("8 GB", 8), ("4 GB", 4), ("2 GB", 2)]:
            _fig.add_vline(x=_gb, line_dash="dash", line_color=COLORS["Grey"],
                           annotation_text=_tier)
        _fig.update_layout(
            height=380,
            xaxis=dict(title="Model Size (GB)", range=[0, 18]),
            yaxis=dict(title="Accuracy Loss (%)"),
            margin=dict(l=50, r=20, t=40, b=50),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        # Per-tier recommendations
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">8 GB Tier</div>
                <div style="font-size:1.1rem; font-weight:800; color:{COLORS['GreenLine']};">INT8</div>
                <div style="font-size:0.72rem; color:#94a3b8;">8 GB, ~0.8% loss</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">4 GB Tier</div>
                <div style="font-size:1.1rem; font-weight:800; color:{COLORS['OrangeLine']};">INT4</div>
                <div style="font-size:0.72rem; color:#94a3b8;">4 GB, ~2.2% loss</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">2 GB Tier</div>
                <div style="font-size:1.1rem; font-weight:800; color:{COLORS['RedLine']};">INT4 + Prune</div>
                <div style="font-size:0.72rem; color:#94a3b8;">2 GB, ~8% loss -- risky</div>
            </div>
        </div>
        """))

        if pC_pred.value == "int4":
            items.append(mo.callout(mo.md("**Correct.** INT4 at 4 GB fits the 4 GB tier with ~2% accuracy loss. "
                "INT8 at 8 GB does not fit when you account for KV cache overhead."), kind="success"))
        else:
            items.append(mo.callout(mo.md("**INT4 is the answer for 4 GB.** INT8 weights alone consume 8 GB, "
                "leaving zero room for KV cache and runtime. INT4 halves that to 4 GB."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Model Memory Budget": mo.md("""
**Formula:**
$$
M_{\\text{weights}} = P \\times \\frac{b}{8} \\quad \\text{(bytes)}
$$

Total deployment memory must satisfy:
$$
M_{\\text{weights}} + M_{\\text{KV cache}} + M_{\\text{runtime}} \\leq M_{\\text{device}}
$$

**Variables:**
- **$P$**: number of parameters (e.g., 8B for Llama-3 8B)
- **$b$**: bits per weight (32 for FP32, 8 for INT8, 4 for INT4)
- **$M_{\\text{KV cache}}$**: key-value cache memory (grows with context length)
- **$M_{\\text{runtime}}$**: activations, optimizer state, framework overhead

At INT4, Llama-3 8B weighs 4 GB -- just barely fitting a 4 GB device with minimal KV cache headroom.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D: Energy Dividend
    # ─────────────────────────────────────────────────────────────────────
    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid #6366f1; background:#f0f4ff;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Battery Life Engineer
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our mobile app runs inference continuously. Battery life is 2 hours with
                FP32. Marketing wants 8 hours. Is quantization enough?&rdquo;
            </div>
        </div>
        """))
        items.append(mo.md("""
        ## The Energy Dividend: Bits are Joules

        Moving data costs **40,000x more energy** than computing on it:
        - DRAM read: 640 pJ
        - SRAM read: 5 pJ
        - FP32 multiply: 3.7 pJ
        - INT8 multiply: 0.2 pJ
        - Integer add: 0.015 pJ

        INT8 inference uses up to **20x less energy** than FP32 because the
        dominant cost is data movement, not arithmetic.
        """))
        items.append(pD_pred)
        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the energy analyzer."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pD_precision_e, pD_hw], justify="start"))

        _prec = pD_precision_e.value
        _hw = pD_hw.value

        # Energy per operation (pJ)
        _dram_pj = 640.0
        _compute_pj = {"fp32": 3.7, "fp16": 1.8, "int8": 0.2, "int4": 0.1}
        _bpp = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}

        _n_params = MOBILENET_PARAMS
        _n_ops = MOBILENET_FLOPS

        # Memory access energy: each parameter loaded from DRAM once
        _mem_energy_pj = _n_params * _bpp[_prec] * _dram_pj / _bpp["fp32"]
        # Compute energy
        _comp_energy_pj = _n_ops * _compute_pj[_prec]

        _total_energy_pj = _mem_energy_pj + _comp_energy_pj
        _mem_frac = _mem_energy_pj / _total_energy_pj * 100
        _comp_frac = _comp_energy_pj / _total_energy_pj * 100

        # Battery life estimate (iPhone 15 Pro: 15 Wh)
        _energy_j = _total_energy_pj * 1e-12
        if _energy_j > 0:
            _inferences_per_wh = (3600 / _energy_j) * 1e-3  # thousands
        else:
            _inferences_per_wh = 0

        _base_energy = _n_params * 4 * _dram_pj / 4 + _n_ops * _compute_pj["fp32"]
        _energy_reduction = _base_energy / _total_energy_pj if _total_energy_pj > 0 else 1

        _fig = go.Figure()
        _prec_labels = ["FP32", "FP16", "INT8", "INT4"]
        _prec_keys = ["fp32", "fp16", "int8", "int4"]
        _mem_energies = [_n_params * _bpp[p] * _dram_pj / _bpp["fp32"] for p in _prec_keys]
        _comp_energies = [_n_ops * _compute_pj[p] for p in _prec_keys]

        _fig.add_trace(go.Bar(name="Memory Access", x=_prec_labels,
            y=[m/1e12 for m in _mem_energies], marker_color=COLORS["RedLine"]))
        _fig.add_trace(go.Bar(name="Compute", x=_prec_labels,
            y=[c/1e12 for c in _comp_energies], marker_color=COLORS["BlueLine"]))
        _fig.update_layout(
            barmode="stack", height=340,
            yaxis=dict(title="Energy per Inference (J)"),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Memory Energy</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_mem_frac:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">of total inference energy</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Compute Energy</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_comp_frac:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">of total inference energy</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Energy Reduction</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_energy_reduction:.1f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">vs FP32 baseline</div>
            </div>
        </div>
        """))

        if pD_pred.value == "99_1":
            items.append(mo.callout(mo.md("**Correct.** Memory access dominates energy by 100x+. "
                "DRAM read (640 pJ) dwarfs INT8 multiply (0.2 pJ) by 3,200x per operation."), kind="success"))
        else:
            items.append(mo.callout(mo.md("**Memory access is essentially all of the energy.** "
                f"At {_prec.upper()}, memory is {_mem_frac:.0f}% of total energy. "
                "Every bit you do not move saves Joules."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Energy Cost of Memory Access vs. Compute": mo.md("""
**Formula:**
$$
E_{\\text{total}} = P \\cdot (E_{\\text{mem}} + E_{\\text{comp}}) = P \\cdot \\left(\\frac{b}{8} \\cdot E_{\\text{DRAM}} + E_{\\text{MAC}}\\right)
$$

**Variables:**
- **$P$**: number of MAC operations
- **$E_{\\text{DRAM}}$**: energy per byte of DRAM access (~640 pJ for DDR4)
- **$E_{\\text{MAC}}$**: energy per multiply-accumulate (~0.2 pJ for INT8, ~3.7 pJ for FP32)
- **$b$**: bits per weight

At FP32: $E_{\\text{mem}} / E_{\\text{comp}} = (4 \\times 640) / 3.7 \\approx 690\\times$. Memory access dominates energy by 100-1000x.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART E: Dark Knowledge Transfer
    # ─────────────────────────────────────────────────────────────────────
    def build_part_e():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; TinyML Deployment Lead
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our teacher model (ResNet-50) cannot run on the ESP32 sensor nodes.
                We need a MobileNetV2 that matches teacher quality. Distillation? What accuracy
                should we expect from the student?&rdquo;
            </div>
        </div>
        """))
        items.append(mo.md("""
        ## Dark Knowledge Transfer

        Knowledge distillation trains a small **student** model to mimic the **soft
        probability outputs** of a large teacher. The student learns not just the
        correct class but the teacher's uncertainty structure -- which wrong classes
        are similar, how confident to be. This is **dark knowledge**.

        ```
        L_distill = alpha * KL(softmax(z_t/T), softmax(z_s/T)) + (1-alpha) * CE(y, z_s)
        ```

        Temperature T controls how much of the dark knowledge transfers. Higher T
        softens the distribution, revealing more inter-class relationships.
        """))
        items.append(pE_pred)
        if pE_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the distillation dashboard."), kind="warn"))
            return mo.vstack(items)

        items.append(pE_temp)

        _T = pE_temp.value
        _teacher_acc = 76.1
        _student_hard = 71.8  # MobileNetV2 trained on hard labels
        _gap = _teacher_acc - _student_hard

        # Distillation recovery: diminishing returns with temperature
        # Optimal around T=4-6, saturates at higher T
        _recovery = _gap * 0.75 * (1 - math.exp(-0.5 * _T))
        _student_distill = min(_student_hard + _recovery, _teacher_acc - 0.5)

        # Teacher: ResNet-50 specs, Student: MobileNetV2 specs
        _teacher_size = RESNET50_PARAMS * 4 / (1024 * 1024)  # FP32 MB
        _student_size = MOBILENET_PARAMS * 1 / (1024 * 1024)  # INT8 MB
        _speedup = RESNET50_FLOPS / MOBILENET_FLOPS

        _fig = go.Figure()
        _categories = ["Student (hard labels)", f"Student (distilled, T={_T:.1f})", "Teacher"]
        _accs = [_student_hard, _student_distill, _teacher_acc]
        _colors = [COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["BlueLine"]]
        _fig.add_trace(go.Bar(
            x=_categories, y=_accs, marker_color=_colors, opacity=0.88,
            text=[f"{a:.1f}%" for a in _accs], textposition="outside",
        ))
        _fig.update_layout(
            height=320, yaxis=dict(title="ImageNet Accuracy (%)", range=[65, 80]),
            showlegend=False,
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _pct_teacher = (_student_distill / _teacher_acc) * 100
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Distilled Accuracy</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_student_distill:.1f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_pct_teacher:.0f}% of teacher</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Size Reduction</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_teacher_size/_student_size:.0f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_teacher_size:.0f} MB vs {_student_size:.0f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Speedup</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_speedup:.1f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">inference FLOPS reduction</div>
            </div>
        </div>
        """))

        items.append(mo.md(f"""
**Distillation -- Live Calculation** (`T = {_T:.1f}`)

```
Teacher:         ResNet-50, {_teacher_acc}% accuracy, {_teacher_size:.0f} MB (FP32)
Student (hard):  MobileNetV2, {_student_hard}% accuracy
Student (dist):  MobileNetV2, {_student_distill:.1f}% accuracy (T={_T:.1f})
Gap closed:      {_student_distill - _student_hard:.1f}% of {_gap:.1f}% gap
Size ratio:      {_teacher_size/_student_size:.0f}x smaller (INT8 student vs FP32 teacher)
```
*Source: Chapter 10, knowledge distillation and dark knowledge*
        """))

        if pE_pred.value == "73":
            items.append(mo.callout(mo.md("**Correct.** The distilled student retains ~95% of teacher accuracy. "
                "Soft labels carry far more information than hard labels -- the dark knowledge "
                "of which classes are confusable closes most of the capacity gap."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"**The distilled student achieves ~{_student_distill:.0f}%.** "
                "Soft labels from the teacher carry rich inter-class information that hard labels discard. "
                "Adjust the temperature slider to see how dark knowledge transfer changes."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Knowledge Distillation Loss": mo.md("""
**Formula:**
$$
\\mathcal{L}_{\\text{KD}} = (1-\\alpha)\\,\\mathcal{L}_{\\text{CE}}(y, \\sigma(z_s)) \\;+\\; \\alpha\\,T^2\\,\\text{KL}\\!\\left(\\sigma\\!\\left(\\frac{z_t}{T}\\right) \\| \\sigma\\!\\left(\\frac{z_s}{T}\\right)\\right)
$$

**Variables:**
- **$z_t, z_s$**: teacher and student logits
- **$T$**: temperature (higher = softer probability distribution)
- **$\\alpha$**: interpolation weight between hard and soft targets
- **$\\sigma$**: softmax function
- **$\\text{KL}$**: Kullback-Leibler divergence

The $T^2$ factor compensates for gradient magnitude reduction at high temperature. Optimal $T$ is typically 3-5.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS
    # ─────────────────────────────────────────────────────────────────────
    def build_synthesis():
        return mo.vstack([
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
                        <strong>1. INT8 is a free lunch; INT4 is a cliff.</strong>
                        FP32 to INT8 loses &lt;1% accuracy. Below 4 bits, accuracy collapses
                        catastrophically. The curve is flat-then-vertical, not gradual.
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>2. Unstructured pruning is invisible to GPUs.</strong>
                        90% sparsity yields exactly 0% speedup on dense GEMM kernels.
                        Structured pruning or hardware sparse support is required for real gains.
                    </div>
                    <div>
                        <strong>3. Distillation unlocks deployments compression cannot reach.</strong>
                        A distilled student retains 95% of teacher accuracy at 10x smaller size,
                        running on hardware where the teacher physically cannot fit.
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
                        <strong>Lab 11:</strong> The Hardware Roofline -- compression changed the
                        model. Now discover why the same kernel is memory-bound on one chip and
                        compute-bound on another.
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
                        <strong>Read:</strong> the Model Compression chapter for quantization theory
                        and pruning hardware analysis.<br/>
                        <strong>Build:</strong> TinyTorch Module 10 -- INT8 quantizer and
                        magnitude pruning.
                    </div>
                </div>
            </div>
            """),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Quantization": build_part_a(),
        "Part B: Pruning Trap": build_part_b(),
        "Part C: Pareto Frontier": build_part_c(),
        "Part D: Energy Dividend": build_part_d(),
        "Part E: Distillation": build_part_e(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


# ═════════════════════════════════════════════════════════════════════════════
# CELL 5: LEDGER HUD
# ═════════════════════════════════════════════════════════════════════════════

# ===========================================================================
# ZONE D: LEDGER HUD
# ===========================================================================

@app.cell(hide_code=True)
def _(COLORS, ledger, mo, pA_pred, pB_pred, pC_pred, pD_pred, pE_pred):
    if pA_pred.value is not None and pB_pred.value is not None and pC_pred.value is not None and pD_pred.value is not None and pE_pred.value is not None:
        ledger.save(chapter=10, design={
            "lab": "model_compress",
            "completed": True,
            "quantization_accuracy_loss": pA_pred.value,
            "pruning_speedup_prediction": pB_pred.value,
            "deployment_strategy": pC_pred.value,
            "energy_memory_vs_compute": pD_pred.value,
            "distillation_accuracy": pE_pred.value,
        })
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">10 &middot; Model Compression</span>
        <span style="flex:1;"></span>
        <span class="hud-label">CH</span>
        <span class="hud-value">10</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">COMPLETE</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
