import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


# ═════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP ──────────────────────────────────────────────────────────
@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly"], keep_going=False)
        await micropip.install(
            "../../wheels/mlsysim-0.1.0-py3-none-any.whl", keep_going=False
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    import mlsysim
    from mlsysim.core.engine import Engine

    # ── Hardware constants ─────────────────────────────────────────────────
    H100 = mlsysim.Hardware.Cloud.H100
    JETSON = mlsysim.Hardware.Edge.JetsonOrinNX
    IPHONE = mlsysim.Hardware.Mobile.iPhone15Pro

    H100_RAM_GB   = H100.memory.capacity.m_as("GB")
    JETSON_RAM_GB = JETSON.memory.capacity.m_as("GB")
    IPHONE_RAM_GB = IPHONE.memory.capacity.m_as("GB")

    H100_TFLOPS   = H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_BW_GBS   = H100.memory.bandwidth.m_as("GB/s")
    H100_DISPATCH = H100.dispatch_tax.m_as("ms")

    JETSON_DISPATCH = JETSON.dispatch_tax.m_as("ms")

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return (
        COLORS, Engine, H100, JETSON, IPHONE,
        H100_RAM_GB, JETSON_RAM_GB, IPHONE_RAM_GB,
        H100_TFLOPS, H100_BW_GBS, H100_DISPATCH, JETSON_DISPATCH,
        LAB_CSS, apply_plotly_theme, go, math, mo, np, ledger, mlsysim,
    )


# ─── CELL 1: HEADER ─────────────────────────────────────────────────────────
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
                Machine Learning Systems &middot; Volume I &middot; Lab 06
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Architecture Tax
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Inductive Bias &middot; Quadratic Wall &middot; Depth vs. Width &middot; Workload Signatures
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Architecture is not just an accuracy choice -- it is a systems choice
                that determines parameter count, memory access patterns, parallelism,
                and hardware utilization. Each architecture family occupies a distinct
                point in the compute-memory trade-off space.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~52 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter 6: Network Architectures
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">MLP: 22.7B params / layer</span>
                <span class="badge badge-warn">Attention: O(N^2) memory</span>
                <span class="badge badge-fail">Depth: dispatch tax</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the parameter explosion</strong>
                    from removing inductive bias: an MLP first layer on 224x224 images requires
                    22.7 billion parameters vs. 1,728 for a 3x3 CNN -- a 13.1 million-fold reduction.</div>
                <div style="margin-bottom: 3px;">2. <strong>Calculate the quadratic attention wall</strong>:
                    doubling context length from 4K to 8K tokens quadruples attention memory,
                    creating hard OOM ceilings on context window size.</div>
                <div style="margin-bottom: 3px;">3. <strong>Diagnose the depth-vs-width trade-off</strong>:
                    two networks with identical FLOPs can have 10x different latencies due to
                    sequential dispatch overhead.</div>
                <div style="margin-bottom: 3px;">4. <strong>Compare workload signatures</strong>:
                    predict which architecture achieves highest GPU utilization based on
                    arithmetic intensity.</div>
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
                    Dense layer FLOPs from @sec-nn-computation-flop-counting &middot;
                    Memory hierarchy cliffs from @sec-nn-computation-memory-hierarchy &middot;
                    Iron Law from @sec-introduction-iron-law
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~52 min</strong><br/>
                    Part A: ~12 min &middot; Part B: ~12 min<br/>
                    Part C: ~10 min &middot; Part D: ~8 min
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
                &ldquo;Two networks have identical FLOPs and parameter counts. Why is one
                10x faster than the other -- and why does the &lsquo;modern&rsquo; Transformer
                achieve lower GPU utilization than the &lsquo;old&rsquo; CNN?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: READING ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **Chapter 6: Inductive Bias** -- why CNNs have 13M fewer parameters than MLPs
      for the same image task. Weight sharing and locality as physical constraints.
    - **Chapter 6: Attention Complexity** -- the N*N score matrix, quadratic memory
      scaling, and OOM ceilings on context length.
    - **Chapter 6: Depth vs. Width** -- sequential dispatch overhead, parallelism,
      and why FLOPs are an insufficient proxy for latency.
    - **Chapter 6: Workload Signatures** -- arithmetic intensity by architecture
      family and the roofline model connection.
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE B-D: ALL PARTS AS TABS
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 4: TABS CELL ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, Engine, H100, JETSON, IPHONE,
    H100_RAM_GB, JETSON_RAM_GB, IPHONE_RAM_GB,
    H100_TFLOPS, H100_BW_GBS, H100_DISPATCH, JETSON_DISPATCH,
    apply_plotly_theme, go, math, mo, np, ledger, mlsysim,
):
    # ─────────────────────────────────────────────────────────────────────
    # SHARED WIDGET STATE
    # ─────────────────────────────────────────────────────────────────────

    # Part A widgets
    partA_prediction = mo.ui.radio(
        options={
            "A) ~150K (about the input size)": "150k",
            "B) ~23M (like ResNet-50 total)": "23m",
            "C) ~1B (a billion)": "1b",
            "D) ~22.7B (twenty-two billion)": "22b",
        },
        label="An MLP takes a 224x224 RGB image as a flattened input (150,528 dims). "
              "The first hidden layer also has 150,528 neurons. How many parameters in this single layer?",
    )
    return (partA_prediction,)

@app.cell(hide_code=True)
def _(mo, partA_prediction):
    partA_arch = mo.ui.radio(
        options={"MLP (no structure)": "mlp", "CNN 3x3": "cnn3", "CNN 5x5": "cnn5"},
        value="MLP (no structure)", label="Architecture:", inline=True,
    )
    partA_resolution = mo.ui.slider(
        start=28, stop=512, value=224, step=28, label="Image resolution (px)",
    )

    # Part B widgets
    partB_prediction = mo.ui.radio(
        options={
            "A) 2x (linear with tokens)": "2x",
            "B) 4x (quadratic)": "4x",
            "C) 8x": "8x",
            "D) 16x": "16x",
        },
        label="Doubling a Transformer's context from 4,096 to 8,192 tokens -- "
              "how much more memory does attention require?",
    )
    return (partB_prediction,)

@app.cell(hide_code=True)
def _(mo, partB_prediction):
    partB_seq_len = mo.ui.slider(
        start=512, stop=131072, value=4096, step=512, label="Sequence length (tokens)",
    )
    partB_heads = mo.ui.slider(
        start=1, stop=64, value=32, step=1, label="Attention heads",
    )

    # Part C widgets
    partC_prediction = mo.ui.radio(
        options={
            "A) Same speed -- same FLOPs means same time": "same",
            "B) Deep network -- more specialized": "deep",
            "C) Shallow-wide -- by ~2x": "2x",
            "D) Shallow-wide -- by ~10x": "10x",
        },
        label="Two networks: identical FLOPs/params. One is 128 layers deep (width 32). "
              "The other is 2 layers deep (width 512). Which is faster at inference?",
    )
    return (partC_prediction,)

@app.cell(hide_code=True)
def _(mo, partC_prediction):
    partC_depth = mo.ui.slider(
        start=2, stop=128, value=64, step=2, label="Depth (layers)",
    )
    partC_context_c = mo.ui.radio(
        options={"Cloud (H100)": "cloud", "Edge (Jetson)": "edge"},
        value="Cloud (H100)", label="Deployment:", inline=True,
    )

    # Part D widgets
    partD_prediction = mo.ui.radio(
        options={
            "A) Transformer (modern and optimized)": "transformer",
            "B) CNN (high arithmetic intensity)": "cnn",
            "C) MLP (simplest architecture)": "mlp",
            "D) All roughly equal": "equal",
        },
        label="Which architecture family achieves the highest GPU utilization (MFU) at batch=1?",
    )
    return (partD_prediction,)

@app.cell(hide_code=True)
def _(mo, partD_prediction):
    partD_batch_d = mo.ui.slider(
        start=1, stop=256, value=1, step=1, label="Batch size",
    )

    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER: The Cost of No Structure
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; ML Architect, WildlifeVision
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We want to classify camera trap images using a simple fully-connected
                network. No fancy convolutions, no attention -- just pure MLPs. Our intern
                says this should work fine for 224x224 images. How many parameters do we need?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Dr. Sarah Kimani, ML Architect &middot; WildlifeVision
            </div>
        </div>
        """))

        items.append(mo.md("""
## Inductive Bias Is a Physical Memory Constraint

An MLP processing a 224x224 RGB image flattens the input to 150,528 dimensions.
With a matching hidden layer, the first-layer weight matrix has:

```
Parameters = 150,528 * 150,528 = 22.66 billion
Memory     = 22.66B * 4 bytes  = 90.6 GB (FP32)
```

A CNN with 3x3 filters requires only **1,728 parameters** for the same first layer
(3 * 3 * 3 channels_in * 64 channels_out). That is a **13.1 million-fold reduction**.

Inductive bias (locality, weight sharing) is not an abstract concept -- it is the
mechanism that makes computer vision physically feasible.
        """))

        items.append(partA_prediction)

        if partA_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the architecture comparison."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partA_arch, partA_resolution], justify="start"))

        _res = partA_resolution.value
        _arch = partA_arch.value
        _channels_in = 3
        _channels_out = 64
        _input_dim = _res * _res * _channels_in

        if _arch == "mlp":
            _params = _input_dim * _input_dim
            _arch_label = f"MLP ({_input_dim:,} x {_input_dim:,})"
        elif _arch == "cnn3":
            _params = 3 * 3 * _channels_in * _channels_out
            _arch_label = f"CNN 3x3 ({3}x{3}x{_channels_in}x{_channels_out})"
        else:
            _params = 5 * 5 * _channels_in * _channels_out
            _arch_label = f"CNN 5x5 ({5}x{5}x{_channels_in}x{_channels_out})"

        _mem_gb = _params * 4 / (1024**3)
        _cnn3_params = 3 * 3 * _channels_in * _channels_out
        _fold_reduction = _params / _cnn3_params if _cnn3_params > 0 else 1

        # Bar chart on log scale
        _archs = ["MLP", "CNN 3x3", "CNN 5x5"]
        _p_mlp = _input_dim * _input_dim
        _p_cnn3 = 3 * 3 * _channels_in * _channels_out
        _p_cnn5 = 5 * 5 * _channels_in * _channels_out
        _all_params = [_p_mlp, _p_cnn3, _p_cnn5]
        _bar_colors = [COLORS["RedLine"], COLORS["BlueLine"], COLORS["GreenLine"]]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=_archs, y=_all_params,
            marker_color=_bar_colors, opacity=0.88,
            text=[f"{p:,.0f}" for p in _all_params],
            textposition="outside",
        ))
        # Device threshold lines
        for _dev, _ram, _col in [("H100 80GB", 80, COLORS["BlueLine"]),
                                  ("Jetson 16GB", 16, COLORS["OrangeLine"]),
                                  ("iPhone 8GB", 8, COLORS["RedLine"])]:
            _max_params = _ram * (1024**3) / 4
            _fig.add_hline(y=_max_params, line_dash="dash", line_color=_col,
                           annotation_text=_dev, annotation_position="right")

        _fig.update_layout(
            height=400, yaxis_title="First-Layer Parameters", yaxis_type="log",
            title=f"First-Layer Parameters at {_res}x{_res} Resolution (log scale)",
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _oom_color = COLORS["RedLine"] if _mem_gb > 80 else COLORS["OrangeLine"] if _mem_gb > 8 else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_oom_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Parameters ({_arch.upper()})</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_oom_color};">{_params:,.0f}</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_oom_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Memory (FP32)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_oom_color};">{_mem_gb:,.1f} GB</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">MLP/CNN Fold Reduction</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_fold_reduction:,.0f}x</div>
            </div>
        </div>
        """))

        if _mem_gb > H100_RAM_GB and _arch == "mlp":
            items.append(mo.callout(mo.md(
                f"**OOM -- Even the H100 cannot hold this single layer.** "
                f"The MLP first layer requires {_mem_gb:,.1f} GB in FP32, exceeding "
                f"the H100's {H100_RAM_GB:.0f} GB. This is not a performance problem -- "
                f"it is a **feasibility violation**. CNNs exist because MLPs cannot."
            ), kind="danger"))

        _pred = partA_prediction.value
        if _pred == "22b":
            items.append(mo.callout(mo.md(
                "**Correct.** 150,528^2 = 22.66 billion parameters in one layer. "
                "This exceeds even an H100 in FP32. CNNs reduce this by 13.1 million-fold "
                "through locality and weight sharing -- inductive bias is a physical necessity."
            ), kind="success"))
        elif _pred == "150k":
            items.append(mo.callout(mo.md(
                "**You confused input size with parameter count.** A dense layer connecting "
                "N inputs to N outputs has N*N parameters. At 150,528 inputs, that is "
                "150,528^2 = 22.66 billion -- not 150,528."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Not quite.** The first dense layer has input_dim * hidden_dim = "
                f"150,528 * 150,528 = 22.66 billion parameters. The O(d^2) scaling of "
                f"dense layers makes MLPs physically infeasible for image-sized inputs."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Parameter Count": mo.md(f"""
**MLP:** params = input_dim * hidden_dim = {_input_dim:,} * {_input_dim:,} = {_input_dim*_input_dim:,}
**CNN 3x3:** params = 3 * 3 * C_in * C_out = 9 * {_channels_in} * {_channels_out} = {_p_cnn3:,}
**Fold reduction:** {_input_dim*_input_dim / _p_cnn3:,.0f}x

Source: @sec-architectures-inductive-bias
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER: The Quadratic Wall
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Product Manager, WildlifeVision
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We want to use a Transformer for analyzing long audio recordings from
                wildlife sensors. The research paper uses 4K token context. We need 128K for
                full recordings. Just double the context a few times -- how much more memory?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Alex Torres, Product Manager &middot; WildlifeVision
            </div>
        </div>
        """))

        items.append(mo.md("""
## The Quadratic Wall: Attention Memory Scales as N^2

Transformer self-attention creates an N x N score matrix. Doubling context
from N to 2N:

```
Attention memory = 2 * N^2 * heads * bytes_per_element
At 2N:           = 2 * (2N)^2 * heads * bytes = 4 * original
```

This is a **4x increase**, not 2x. At 128K tokens with 32 heads in FP16,
the attention matrix alone requires ~64 GB -- exceeding even an H100 for
a single head's computation.
        """))

        items.append(partB_prediction)

        if partB_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the attention memory simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partB_seq_len, partB_heads], justify="start"))

        _seq = partB_seq_len.value
        _heads = partB_heads.value
        _bytes = 2  # FP16
        _attn_mem_bytes = 2 * _seq * _seq * _heads * _bytes
        _attn_mem_gb = _attn_mem_bytes / (1024**3)

        # Memory curve
        _seq_range = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
        _mem_curve = 2 * _seq_range.astype(float)**2 * _heads * _bytes / (1024**3)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_seq_range.tolist(), y=_mem_curve.tolist(),
            mode="lines+markers", name="Attention Memory (GB)",
            line=dict(color=COLORS["RedLine"], width=2.5),
        ))
        _fig.add_trace(go.Scatter(
            x=[_seq], y=[_attn_mem_gb],
            mode="markers", name="Current Setting",
            marker=dict(size=14, color=COLORS["BlueLine"], symbol="diamond",
                        line=dict(width=2, color="white")),
        ))
        _fig.add_hline(y=H100_RAM_GB, line_dash="dash", line_color=COLORS["BlueLine"],
                       annotation_text=f"H100 ({H100_RAM_GB:.0f} GB)")
        _fig.add_hline(y=JETSON_RAM_GB, line_dash="dash", line_color=COLORS["OrangeLine"],
                       annotation_text=f"Jetson ({JETSON_RAM_GB:.0f} GB)")
        _fig.update_layout(
            height=380, xaxis_title="Sequence Length (tokens)",
            yaxis_title="Attention Memory (GB)", xaxis_type="log", yaxis_type="log",
            title=f"Attention Memory vs. Sequence Length ({_heads} heads, FP16)",
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _oom_h100 = _attn_mem_gb > H100_RAM_GB
        _oom_jetson = _attn_mem_gb > JETSON_RAM_GB
        _color = COLORS["RedLine"] if _oom_h100 else COLORS["OrangeLine"] if _oom_jetson else COLORS["GreenLine"]

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Attention Memory</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_color};">{_attn_mem_gb:,.2f} GB</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Sequence Length</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_seq:,} tokens</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Score Matrix Size</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_seq:,} x {_seq:,}</div>
            </div>
        </div>
        """))

        if _oom_h100:
            items.append(mo.callout(mo.md(
                f"**OOM on H100.** Attention alone requires {_attn_mem_gb:,.1f} GB, exceeding "
                f"the H100's {H100_RAM_GB:.0f} GB. Techniques like FlashAttention or sparse "
                f"attention are required to make this sequence length feasible."
            ), kind="danger"))
        elif _oom_jetson:
            items.append(mo.callout(mo.md(
                f"**OOM on Jetson.** Attention requires {_attn_mem_gb:,.1f} GB, exceeding "
                f"the Jetson's {JETSON_RAM_GB:.0f} GB. This context length requires cloud hardware."
            ), kind="danger"))

        _pred = partB_prediction.value
        if _pred == "4x":
            items.append(mo.callout(mo.md(
                "**Correct.** The N*N attention matrix means doubling context gives 4x memory. "
                "This quadratic wall is why context length extensions (4K->128K) require "
                "fundamental algorithmic innovations like FlashAttention, not just more RAM."
            ), kind="success"))
        elif _pred == "2x":
            items.append(mo.callout(mo.md(
                "**Linear intuition fails for attention.** The score matrix is N*N. "
                "Doubling N gives (2N)^2 = 4N^2 -- a 4x increase, not 2x."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                "**Not quite.** Attention memory scales as N^2. Doubling N yields exactly "
                "4x memory. The key insight: each token must attend to every other token."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Attention Memory": mo.md(f"""
```
Attention_memory = 2 * seq_len^2 * heads * bytes_per_element
                 = 2 * {_seq:,}^2 * {_heads} * {_bytes}
                 = {_attn_mem_bytes:,.0f} bytes = {_attn_mem_gb:,.2f} GB
```
Doubling seq_len: 2 * (2N)^2 = 4 * 2N^2 = **4x memory**

Source: @sec-architectures-attention-complexity
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER: Depth vs. Width
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Systems Engineer, WildlifeVision
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We have two model candidates with identical FLOP counts and identical
                accuracy. One is deep-narrow (128 layers, width 32), the other shallow-wide
                (2 layers, width 512). Our benchmarks show a 10x latency difference. Must be
                a measurement error -- same FLOPs should mean same speed, right?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Ravi Patel, Systems Engineer &middot; WildlifeVision
            </div>
        </div>
        """))

        items.append(mo.md("""
## FLOPs Are Not Latency: Depth Imposes Sequential Overhead

Each layer dispatch incurs a fixed overhead (kernel launch tax). A 128-layer
network pays this tax 128 times; a 2-layer network pays it twice.

Additionally, narrow layers (width 32) have fewer parallel operations per
layer, reducing hardware utilization. The combination of high dispatch tax
and low per-layer parallelism makes deep-narrow networks dramatically
slower than shallow-wide ones -- even at identical total FLOPs.
        """))

        items.append(partC_prediction)

        if partC_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the depth-vs-width simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partC_depth, partC_context_c], justify="start"))

        _depth = partC_depth.value
        _ctx = partC_context_c.value
        _dispatch_ms = H100_DISPATCH if _ctx == "cloud" else JETSON_DISPATCH

        # Fixed total param budget: ~32,768 params
        _total_param_budget = 32768
        _width = max(4, int(math.sqrt(_total_param_budget / _depth)))
        _actual_params = _depth * _width * _width
        _flops_per_layer = 2 * _width * _width
        _total_flops = _depth * _flops_per_layer

        _peak_tflops = H100_TFLOPS if _ctx == "cloud" else 25.0
        _compute_per_layer_ms = (_flops_per_layer / (_peak_tflops * 1e12)) * 1000
        _total_compute_ms = _depth * _compute_per_layer_ms
        _total_dispatch_ms = _depth * _dispatch_ms
        _total_latency_ms = _total_compute_ms + _total_dispatch_ms
        _dispatch_fraction = _total_dispatch_ms / _total_latency_ms * 100 if _total_latency_ms > 0 else 0

        # Reference: shallow-wide (2 layers)
        _ref_depth = 2
        _ref_width = max(4, int(math.sqrt(_total_param_budget / _ref_depth)))
        _ref_compute = _ref_depth * (2 * _ref_width * _ref_width) / (_peak_tflops * 1e12) * 1000
        _ref_dispatch = _ref_depth * _dispatch_ms
        _ref_total = _ref_compute + _ref_dispatch
        _speedup = _total_latency_ms / _ref_total if _ref_total > 0 else 1

        # Depth curve
        _depths = np.arange(2, 129, 2)
        _widths_c = np.array([max(4, int(math.sqrt(_total_param_budget / d))) for d in _depths])
        _compute_c = _depths * (2 * _widths_c**2) / (_peak_tflops * 1e12) * 1000
        _dispatch_c = _depths * _dispatch_ms
        _total_c = _compute_c + _dispatch_c

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_depths.tolist(), y=_total_c.tolist(),
            mode="lines", name="Total Latency",
            line=dict(color=COLORS["RedLine"], width=2.5),
        ))
        _fig.add_trace(go.Scatter(
            x=_depths.tolist(), y=_dispatch_c.tolist(),
            mode="lines", name="Dispatch Overhead",
            line=dict(color=COLORS["OrangeLine"], width=2, dash="dash"),
            fill="tozeroy", fillcolor="rgba(204,85,0,0.1)",
        ))
        _fig.add_trace(go.Scatter(
            x=[_depth], y=[_total_latency_ms],
            mode="markers", name="Current",
            marker=dict(size=12, color=COLORS["BlueLine"], symbol="diamond",
                        line=dict(width=2, color="white")),
        ))
        _fig.update_layout(
            height=360, xaxis_title="Depth (layers)", yaxis_title="Latency (ms)",
            title=f"Latency vs. Depth (fixed ~{_total_param_budget:,} params) -- {'Cloud' if _ctx=='cloud' else 'Edge'}",
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Depth</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_depth} layers</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Width</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_width}</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine'] if _dispatch_fraction > 30 else COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Dispatch Overhead</div>
                <div style="font-size:1.5rem; font-weight:800;
                     color:{COLORS['OrangeLine'] if _dispatch_fraction > 30 else COLORS['GreenLine']};">{_dispatch_fraction:.0f}%</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['RedLine'] if _speedup > 5 else COLORS['OrangeLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">vs. 2-Layer (same FLOPs)</div>
                <div style="font-size:1.5rem; font-weight:800;
                     color:{COLORS['RedLine'] if _speedup > 5 else COLORS['OrangeLine']};">{_speedup:.1f}x slower</div>
            </div>
        </div>
        """))

        _pred = partC_prediction.value
        if _pred == "10x":
            items.append(mo.callout(mo.md(
                f"**Correct.** At {_depth} layers, the deep-narrow network is {_speedup:.1f}x "
                f"slower than the 2-layer alternative. Dispatch overhead ({_dispatch_fraction:.0f}% "
                f"of total) and reduced per-layer parallelism make FLOPs a necessary but "
                f"insufficient proxy for latency."
            ), kind="success"))
        elif _pred == "same":
            items.append(mo.callout(mo.md(
                f"**FLOPs are not latency.** Same FLOPs, but the 128-layer network pays "
                f"dispatch overhead 128 times. At current settings, the deep network is "
                f"{_speedup:.1f}x slower."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**The gap is larger than expected.** The deep-narrow network is {_speedup:.1f}x "
                f"slower due to accumulated dispatch overhead and reduced parallelism."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Dispatch Overhead": mo.md(f"""
```
Per-layer dispatch: {_dispatch_ms} ms
Total dispatch:     {_depth} * {_dispatch_ms} = {_total_dispatch_ms:.3f} ms
Total compute:      {_total_compute_ms:.4f} ms
Dispatch fraction:  {_dispatch_fraction:.1f}%
```
The shallow-wide (2 layers, width {_ref_width}) completes in {_ref_total:.4f} ms.

Source: @sec-architectures-depth-vs-width
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER: Workload Signatures
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; CTO, WildlifeVision
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We are choosing between a CNN and a Transformer for our edge deployment.
                The Transformer is newer and gets better benchmark scores. But our edge GPU
                utilization numbers look backwards -- the CNN achieves higher MFU. Is our
                profiling tool broken?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Dr. Maya Rodriguez, CTO &middot; WildlifeVision
            </div>
        </div>
        """))

        items.append(mo.md("""
## Workload Signatures: Arithmetic Intensity Determines Utilization

Each architecture family has a characteristic arithmetic intensity (AI = FLOPs/byte):

- **CNNs**: High AI (>20 FLOPs/byte) -- weight reuse across spatial positions
  makes them compute-bound and GPU-efficient.
- **Transformers at batch=1**: Low AI (<5 FLOPs/byte) for attention --
  memory-bound, lower GPU utilization.
- **MLPs at batch=1**: Very low AI (~0.5 FLOPs/byte) -- permanently bandwidth-bound.

The hardware ridge point (peak_FLOPS / peak_BW) determines the crossover.
        """))

        items.append(partD_prediction)

        if partD_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the workload signature comparison."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(partD_batch_d)

        _bs = partD_batch_d.value

        # Use Engine.solve for representative models
        _models = [
            ("ResNet-50 (CNN)", mlsysim.Models.ResNet50),
            ("GPT-2 (Transformer)", mlsysim.Models.GPT2),
            ("MobileNetV2 (CNN)", mlsysim.Models.MobileNetV2),
        ]
        _results = []
        for _name, _model in _models:
            try:
                _profile = Engine.solve(_model, H100, batch_size=_bs, precision="fp16", efficiency=0.5)
                _ai = _profile.arithmetic_intensity.magnitude
                _mfu = _profile.mfu
                _bn = _profile.bottleneck
                _results.append((_name, _ai, _mfu, _bn))
            except Exception:
                _results.append((_name, 0.0, 0.0, "Error"))

        # Ridge point for H100
        _ridge = H100_TFLOPS * 1000 / H100_BW_GBS  # FLOPS/byte

        _names = [r[0] for r in _results]
        _ais = [r[1] for r in _results]
        _mfus = [r[2] for r in _results]
        _bottlenecks = [r[3] for r in _results]

        _ai_colors = [COLORS["GreenLine"] if ai > _ridge else COLORS["OrangeLine"] if ai > _ridge/2 else COLORS["RedLine"]
                      for ai in _ais]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=_names, y=_ais,
            marker_color=_ai_colors, opacity=0.88,
            text=[f"{ai:.1f}" for ai in _ais], textposition="outside",
        ))
        _fig.add_hline(y=_ridge, line_dash="dash", line_color=COLORS["BlueLine"],
                       annotation_text=f"H100 Ridge Point ({_ridge:.0f} FLOPs/byte)")
        _fig.update_layout(
            height=360, yaxis_title="Arithmetic Intensity (FLOPs/byte)", yaxis_type="log",
            title=f"Arithmetic Intensity by Architecture (batch={_bs})",
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _cards = '<div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">'
        for _name, _ai, _mfu, _bn in _results:
            _mfu_col = COLORS["GreenLine"] if _mfu > 0.5 else COLORS["OrangeLine"] if _mfu > 0.2 else COLORS["RedLine"]
            _cards += f"""
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:180px; text-align:center; background:white; border-top:3px solid {_mfu_col}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.72rem; font-weight:700; margin-bottom:6px;">{_name}</div>
                <div style="font-size:1.3rem; font-weight:800; color:{_mfu_col};">MFU: {_mfu:.1%}</div>
                <div style="font-size:0.78rem; color:{COLORS['TextSec']}; margin-top:4px;">
                    AI: {_ai:.1f} | {_bn}
                </div>
            </div>"""
        _cards += '</div>'
        items.append(mo.Html(_cards))

        _pred = partD_prediction.value
        if _pred == "cnn":
            items.append(mo.callout(mo.md(
                "**Correct.** CNNs achieve the highest GPU utilization because weight reuse "
                "across spatial positions gives them high arithmetic intensity. They operate "
                "in the compute-bound regime. Transformers at batch=1 are memory-bound due "
                "to the attention mechanism's low arithmetic intensity."
            ), kind="success"))
        elif _pred == "transformer":
            items.append(mo.callout(mo.md(
                "**Common misconception.** Transformers are newer but are memory-bound at "
                "small batch sizes due to low arithmetic intensity in attention. CNNs achieve "
                "higher MFU through spatial weight reuse. Try increasing batch size to see "
                "Transformers improve."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                "**Not quite.** CNNs win on utilization because spatial weight reuse gives "
                "them the highest arithmetic intensity. Architecture determines hardware "
                "efficiency, not recency."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Arithmetic Intensity": mo.md(f"""
**Ridge Point (H100):**
```
Ridge = peak_FLOPS / peak_BW = {H100_TFLOPS:.0f} TFLOPS / {H100_BW_GBS:.0f} GB/s
      = {_ridge:.0f} FLOPs/byte
```
Workloads with AI < Ridge are **memory-bound** (low MFU).
Workloads with AI > Ridge are **compute-bound** (high MFU).

Source: @sec-architectures-workload-signatures
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        items = []
        items.append(mo.md("""
## Synthesis: Design for the Edge

A wildlife conservation project needs to classify animals from camera trap
images on a 16 GB Jetson Orin NX with a 50 ms latency SLA.
        """))

        items.append(mo.callout(mo.md("""
Using the four analyses from this lab, justify:

1. **Why not an MLP?** (cite the parameter explosion from Part A)
2. **Why not a large Transformer?** (cite the quadratic wall from Part B)
3. **Deep-narrow or shallow-wide CNN?** (cite dispatch overhead from Part C)
4. **What MFU should you expect?** (cite arithmetic intensity from Part D)
        """), kind="info"))

        items.append(mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. Inductive bias is a physical necessity.</strong>
                    An MLP on 224x224 images needs 22.7B parameters in its first layer --
                    exceeding even an H100. CNNs reduce this by 13.1 million-fold through
                    locality and weight sharing.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Attention scales quadratically.</strong>
                    Doubling context length quadruples attention memory (N^2).
                    At 128K tokens, attention alone can exceed 80 GB.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>3. FLOPs are not latency.</strong>
                    A 128-layer network with identical FLOPs to a 2-layer network can be
                    10x slower due to accumulated dispatch overhead and reduced parallelism.
                </div>
                <div>
                    <strong>4. Architecture determines utilization.</strong>
                    CNNs achieve higher GPU utilization than Transformers at small batch sizes
                    because spatial weight reuse gives them higher arithmetic intensity.
                </div>
            </div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab 07: ML Frameworks</strong> -- Architecture determines the workload.
                    But the framework determines how that workload executes. Lab 07 shows how
                    eager vs. compiled execution and kernel fusion can change latency by 17x
                    without touching a single weight.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-architectures for inductive bias,
                    attention complexity, depth-width trade-offs.<br/>
                    <strong>Build:</strong> TinyTorch Module 06 -- implement CNN
                    convolution and self-attention from scratch.
                </div>
            </div>
        </div>
        """))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A \u2014 The Cost of No Structure":    build_part_a(),
        "Part B \u2014 The Quadratic Wall":          build_part_b(),
        "Part C \u2014 Depth vs. Width":             build_part_c(),
        "Part D \u2014 Workload Signatures":         build_part_d(),
        "Synthesis":                                 build_synthesis(),
    })
    tabs
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: LEDGER HUD ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, ledger, mo):
    _track = ledger._state.track or "not set"
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">06 &middot; The Architecture Tax</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">CHAPTER&nbsp;6</span>
        <span class="hud-value">Network Architectures</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
