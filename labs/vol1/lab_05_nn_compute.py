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

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    import mlsysim

    # ── Hardware constants ─────────────────────────────────────────────────
    H100_TFLOPS   = mlsysim.Hardware.Cloud.H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_BW_GBS   = mlsysim.Hardware.Cloud.H100.memory.bandwidth.m_as("GB/s")
    H100_RAM_GB   = mlsysim.Hardware.Cloud.H100.memory.capacity.m_as("GB")

    MOBILE_TFLOPS = mlsysim.Hardware.Mobile.iPhone15Pro.compute.peak_flops.m_as("TFLOPs/s")
    MOBILE_BW_GBS = mlsysim.Hardware.Mobile.iPhone15Pro.memory.bandwidth.m_as("GB/s")
    MOBILE_RAM_GB = mlsysim.Hardware.Mobile.iPhone15Pro.memory.capacity.m_as("GB")

    # ── Activation function transistor costs ───────────────────────────────
    # Source: @sec-nn-computation-activation-functions (textbook Table 5.x)
    TRANSISTOR_COSTS = {
        "ReLU":    50,      # Single comparison: max(0, x)
        "GELU":    1200,    # Approximate erf() or tanh polynomial
        "Sigmoid": 2500,    # exp(-x), division, addition
        "Swish":   2550,    # Sigmoid(x) * x
    }

    # ── Memory hierarchy tier latencies (ns per access) ────────────────────
    # Source: @sec-nn-computation-memory-hierarchy
    TIER_LATENCY_NS = {"L1": 1.0, "L2": 5.0, "HBM": 100.0, "DRAM": 200.0}
    CLOUD_TIERS_KB  = {"L1": 256, "L2": 50_000, "HBM": 80_000_000}
    MOBILE_TIERS_KB = {"L1": 128, "L2": 32_000, "HBM": 8_000_000}

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return (
        COLORS, H100_TFLOPS, H100_BW_GBS, H100_RAM_GB,
        MOBILE_TFLOPS, MOBILE_BW_GBS, MOBILE_RAM_GB,
        TRANSISTOR_COSTS, TIER_LATENCY_NS, CLOUD_TIERS_KB, MOBILE_TIERS_KB,
        LAB_CSS, apply_plotly_theme, go, math, mo, np, ledger,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 05
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Transistor Tax
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Silicon Cost &middot; Memory Cliffs &middot; Width-Squared Scaling &middot; Backprop Memory
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Four quantitative realities about neural computation that shape every
                architecture decision: activation functions have wildly different silicon
                costs, memory hierarchies create cliffs not slopes, width scales
                quadratically, and training demands storing everything inference can discard.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~50 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter 5: Neural Computation
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">ReLU: 50 transistors</span>
                <span class="badge badge-warn">Sigmoid: 2,500 transistors</span>
                <span class="badge badge-fail">Memory Cliffs: 10-100x</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the transistor cost ratio</strong>
                    between ReLU (~50 transistors) and Sigmoid (~2,500 transistors) and predict
                    when this 50x gap becomes a dominant fraction of inference time.</div>
                <div style="margin-bottom: 3px;">2. <strong>Predict which memory hierarchy tier</strong>
                    a layer's activations land in given batch size and width, and identify the
                    batch size threshold where a 10x latency cliff appears.</div>
                <div style="margin-bottom: 3px;">3. <strong>Calculate the FLOPs scaling law</strong>
                    for dense layers: doubling width yields ~4x FLOPs, not 2x.</div>
                <div style="margin-bottom: 3px;">4. <strong>Compare forward vs. backward memory</strong>:
                    training stores all layer activations simultaneously, creating a 4-10x
                    memory multiplier over inference.</div>
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
                    Activation function definitions from @sec-neural-computation-artificial-neuron
                    &middot; Memory hierarchy tiers from @sec-neural-computation-transistor-tax
                    &middot; Iron Law equation from @sec-introduction-iron-law
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~50 min</strong><br/>
                    Part A: ~10 min &middot; Part B: ~10 min<br/>
                    Part C: ~10 min &middot; Part D: ~10 min
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
                &ldquo;ReLU and Sigmoid produce similar accuracy &mdash; so why does the
                choice of activation function determine whether your model fits in cache
                or spills to memory 100x slower?&rdquo;
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

    - **Chapter 5: The Artificial Neuron** -- activation function definitions, computational
      graph of a single neuron, transistor-level implementation of ReLU vs. Sigmoid.
    - **Chapter 5: The Transistor Tax** -- silicon cost table for common activation functions,
      percentage of inference time consumed by activations on Cloud vs. Mobile.
    - **Chapter 5: Memory Hierarchy** -- cache tiers (L1/L2/HBM/DRAM), tier latencies,
      and the concept of memory cliffs vs. gradual degradation.
    - **Chapter 5: Backpropagation Memory** -- why training must store all intermediate
      activations, and the forward-vs-backward memory multiplier.
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE B-D: ALL PARTS AS TABS
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 4: TABS CELL ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, CLOUD_TIERS_KB, H100_BW_GBS, H100_RAM_GB, H100_TFLOPS,
    MOBILE_BW_GBS, MOBILE_RAM_GB, MOBILE_TFLOPS, MOBILE_TIERS_KB,
    TIER_LATENCY_NS, TRANSISTOR_COSTS, apply_plotly_theme,
    go, math, mo, np, ledger,
):
    # ─────────────────────────────────────────────────────────────────────
    # SHARED WIDGET STATE
    # ─────────────────────────────────────────────────────────────────────

    # Part A widgets
    partA_prediction = mo.ui.radio(
        options={
            "A) <1% (negligible, like on GPUs)": "lt1",
            "B) ~5% (noticeable but small)": "5pct",
            "C) ~23% (significant cost)": "23pct",
            "D) ~50% (dominant cost)": "50pct",
        },
        label="On a mobile NPU, what fraction of inference time comes from activation "
              "functions if you use Sigmoid instead of ReLU in every layer?",
    )
    return (partA_prediction,)

@app.cell(hide_code=True)
def _(mo, partA_prediction):
    partA_context = mo.ui.radio(
        options={"Cloud GPU (H100)": "cloud", "Mobile NPU (iPhone)": "mobile"},
        value="Cloud GPU (H100)",
        label="Deployment context:",
        inline=True,
    )
    partA_act_l1 = mo.ui.dropdown(
        options={"ReLU": "ReLU", "GELU": "GELU", "Sigmoid": "Sigmoid", "Swish": "Swish"},
        value="Sigmoid", label="Layer 1",
    )
    partA_act_l2 = mo.ui.dropdown(
        options={"ReLU": "ReLU", "GELU": "GELU", "Sigmoid": "Sigmoid", "Swish": "Swish"},
        value="Sigmoid", label="Layer 2",
    )
    partA_act_l3 = mo.ui.dropdown(
        options={"ReLU": "ReLU", "GELU": "GELU", "Sigmoid": "Sigmoid", "Swish": "Swish"},
        value="Sigmoid", label="Layer 3",
    )
    partA_act_l4 = mo.ui.dropdown(
        options={"ReLU": "ReLU", "GELU": "GELU", "Sigmoid": "Sigmoid", "Swish": "Swish"},
        value="Sigmoid", label="Layer 4",
    )

    # Part B widgets
    partB_prediction = mo.ui.radio(
        options={
            "A) 2x (linear with data size)": "2x",
            "B) 1.5x (some overhead)": "1.5x",
            "C) 10x (cache tier boundary crossed)": "10x",
            "D) No change (hardware handles it)": "none",
        },
        label="A layer produces a 16 KB activation tensor in L2 cache. You double the "
              "batch size (32 KB now). How does latency change on a mobile NPU?",
    )
    return (partB_prediction,)

@app.cell(hide_code=True)
def _(mo, partB_prediction):
    partB_context = mo.ui.radio(
        options={"Cloud GPU (H100)": "cloud", "Mobile NPU (iPhone)": "mobile"},
        value="Mobile NPU (iPhone)",
        label="Deployment context:",
        inline=True,
    )
    partB_batch = mo.ui.slider(
        start=1, stop=512, value=1, step=1, label="Batch size",
    )
    partB_width = mo.ui.slider(
        start=64, stop=4096, value=256, step=64, label="Layer width",
    )

    # Part C widgets
    partC_prediction = mo.ui.radio(
        options={
            "A) 2x (linear with width)": "2x",
            "B) 3x": "3x",
            "C) ~4x (quadratic)": "4x",
            "D) 8x (cubic)": "8x",
        },
        label="A 3-layer MLP has hidden layers of width 128. You double the hidden "
              "width to 256. By how much do total FLOPs increase?",
    )
    return (partC_prediction,)

@app.cell(hide_code=True)
def _(mo, partC_prediction):
    partC_width = mo.ui.slider(
        start=32, stop=2048, value=128, step=32, label="Hidden layer width",
    )

    # Part D widgets
    partD_prediction = mo.ui.radio(
        options={
            "A) ~50 MB (same as inference)": "50mb",
            "B) ~100 MB (2x for gradients)": "100mb",
            "C) ~200 MB (4x)": "200mb",
            "D) ~500 MB+ (10x+)": "500mb",
        },
        label="A 20-layer model uses 50 MB for inference. How much memory does training "
              "require (weights + gradients + activations, ignoring optimizer state)?",
    )
    return (partD_prediction,)

@app.cell(hide_code=True)
def _(mo, partD_prediction):
    partD_depth = mo.ui.slider(
        start=3, stop=50, value=20, step=1, label="Network depth (layers)",
    )
    partD_batch = mo.ui.slider(
        start=1, stop=128, value=32, step=1, label="Batch size",
    )
    partD_phase = mo.ui.radio(
        options={"Inference": "inference", "Training": "training"},
        value="Inference",
        label="Phase:",
        inline=True,
    )
    partD_width_d = mo.ui.slider(
        start=64, stop=2048, value=512, step=64, label="Layer width",
    )

    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER: The Transistor Tax
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; ML Compiler Engineer, NeuralEdge Inc.
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our mobile vision model uses Sigmoid activations in every layer. The
                cloud version runs perfectly, but on the phone it is 30% slower than our
                latency budget. Engineering says the activations are irrelevant &mdash;
                they are just element-wise ops. Can you check the silicon cost?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Priya Nair, ML Compiler Engineer &middot; NeuralEdge Inc.
            </div>
        </div>
        """))

        items.append(mo.md("""
## The Transistor Tax: Not All Activations Are Created Equal

Every activation function compiles down to transistor-level logic. The cost
varies enormously:

| Function | Transistors | Operation |
|----------|-------------|-----------|
| **ReLU** | ~50 | Single comparison: `max(0, x)` |
| **GELU** | ~1,200 | Approximate `erf()` or tanh polynomial |
| **Sigmoid** | ~2,500 | Exponentiation + division: `1/(1+exp(-x))` |
| **Swish** | ~2,550 | Sigmoid(x) * x |

On a cloud GPU, activation compute is <1% of total inference time because
matrix multiplies dominate. On a mobile NPU, the 50x transistor gap between
ReLU and Sigmoid becomes a **significant fraction** of total inference time.
        """))

        items.append(partA_prediction)

        if partA_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the activation cost simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partA_context], justify="start"))
        items.append(mo.hstack([partA_act_l1, partA_act_l2, partA_act_l3, partA_act_l4], justify="start"))

        # Simulation
        _ctx = partA_context.value
        _is_mobile = _ctx == "mobile"
        _hw_clock_ghz = 1.5 if _is_mobile else 2.1
        _hw_label = "Mobile NPU (iPhone)" if _is_mobile else "Cloud GPU (H100)"
        _channels = 64
        _spatial = 56 * 56
        _activations_per_layer = _channels * _spatial

        _matmul_flops_per_layer = 2 * _channels * _channels * _spatial
        _total_mm_flops = 4 * _matmul_flops_per_layer
        _peak_tflops = MOBILE_TFLOPS if _is_mobile else H100_TFLOPS
        _matmul_time_ms = (_total_mm_flops / (_peak_tflops * 1e12)) * 1000

        _pipeline_width = 256 if _is_mobile else 16384
        _act_names = [partA_act_l1.value, partA_act_l2.value, partA_act_l3.value, partA_act_l4.value]
        _layer_times_act = []
        for _act in _act_names:
            _transistors = TRANSISTOR_COSTS.get(_act, 50)
            _cycles_per_act = _transistors / 50
            _act_time = (_activations_per_layer * _cycles_per_act) / (_hw_clock_ghz * 1e9 * _pipeline_width) * 1000
            _layer_times_act.append(_act_time)

        _total_act_time = sum(_layer_times_act)
        _total_mm_time = _matmul_time_ms
        _norm_time = _total_mm_time * 0.05
        _other_time = _total_mm_time * 0.02
        _total_time = _total_mm_time + _total_act_time + _norm_time + _other_time
        _act_pct = (_total_act_time / _total_time) * 100 if _total_time > 0 else 0

        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Matrix Multiply", x=["Breakdown"], y=[_total_mm_time],
                              marker_color=COLORS["BlueLine"], opacity=0.88))
        _fig.add_trace(go.Bar(name="Activations", x=["Breakdown"], y=[_total_act_time],
                              marker_color=COLORS["RedLine"] if _act_pct > 10 else COLORS["OrangeLine"],
                              opacity=0.88))
        _fig.add_trace(go.Bar(name="Normalization", x=["Breakdown"], y=[_norm_time],
                              marker_color=COLORS["GreenLine"], opacity=0.88))
        _fig.add_trace(go.Bar(name="Other", x=["Breakdown"], y=[_other_time],
                              marker_color=COLORS["Grey"], opacity=0.88))
        _fig.update_layout(barmode="stack", height=320, yaxis_title="Time (ms)",
                           title=f"Inference Time Decomposition -- {_hw_label}",
                           legend=dict(orientation="h", y=1.12, x=0))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _total_transistors = sum(TRANSISTOR_COSTS[a] * _activations_per_layer for a in _act_names)
        _relu_baseline = 50 * _activations_per_layer * 4
        _color = COLORS["RedLine"] if _act_pct > 15 else COLORS["OrangeLine"] if _act_pct > 5 else COLORS["GreenLine"]

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Activation % of Total</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_color};">{_act_pct:.1f}%</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Total Transistors (Act.)</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_total_transistors:,.0f}</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">vs. All-ReLU Baseline</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_total_transistors/_relu_baseline:.1f}x</div>
            </div>
        </div>
        """))

        # Reveal
        _pred = partA_prediction.value
        if _pred == "23pct":
            items.append(mo.callout(mo.md(
                "**Correct.** On a mobile NPU, switching all layers to Sigmoid pushes "
                "activation compute to ~23% of total inference time. On cloud hardware "
                "with 16,000+ ALUs, the same switch is barely measurable (<1%). "
                "The deployment context determines whether this design choice has a real cost."
            ), kind="success"))
        elif _pred == "lt1":
            items.append(mo.callout(mo.md(
                "**That is the cloud GPU answer, not mobile.** On cloud hardware, activations "
                "are indeed <1% thanks to massive parallelism. But on mobile with ~256 ALUs, "
                "the 50x transistor gap translates to ~23% of inference time. "
                "Try switching the context toggle to Mobile to see the difference."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Close but not quite.** The actual fraction on mobile is ~23%. "
                f"Sigmoid requires ~2,500 transistors per activation vs. ReLU's ~50 -- "
                f"a 50x gap that becomes significant when hardware parallelism is limited."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Activation Cost Formula": mo.md(f"""
**Activation time per layer:**
```
T_act = (N_activations * C_transistors) / (f_clock * N_ALUs)
      = ({_activations_per_layer:,} * C) / ({_hw_clock_ghz} GHz * {_pipeline_width})
```
Where C varies: ReLU=50, GELU=1200, Sigmoid=2500, Swish=2550.

Source: @sec-nn-computation-activation-functions, @sec-nn-computation-transistor-tax
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER: The Memory Hierarchy Cliff
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Performance Engineer, NeuralEdge Inc.
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We doubled our batch size and expected 2x throughput. Instead, latency
                jumped 10x on the mobile target. On the cloud GPU, everything was fine. The
                model is identical. What happened?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Marcus Chen, Performance Engineer &middot; NeuralEdge Inc.
            </div>
        </div>
        """))

        items.append(mo.md("""
## Memory Hierarchies Create Cliffs, Not Slopes

Memory is not flat. Hardware organizes it into tiers with dramatically different latencies:

| Tier | Typical Capacity | Latency | Relative Speed |
|------|-----------------|---------|---------------|
| **L1 Cache** | ~128-256 KB | ~1 ns | 1x (baseline) |
| **L2 Cache** | 32-50 MB | ~5 ns | 5x slower |
| **HBM / DRAM** | 8-80 GB | ~100-200 ns | 100-200x slower |

When a tensor exceeds a tier capacity, latency does not degrade gradually -- it
**falls off a cliff** to the next tier. Doubling batch size can push activations
from L2 (5 ns) to HBM (100 ns): a **20x latency jump**, not 2x.
        """))

        items.append(partB_prediction)

        if partB_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the memory tier simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partB_context, partB_batch, partB_width], justify="start"))

        _ctx = partB_context.value
        _batch = partB_batch.value
        _width = partB_width.value
        _is_mobile = _ctx == "mobile"
        _tiers = MOBILE_TIERS_KB if _is_mobile else CLOUD_TIERS_KB

        _tensor_bytes = _batch * _width * 4
        _tensor_kb = _tensor_bytes / 1024

        def _get_tier(sz_kb, tiers):
            if sz_kb <= tiers["L1"]:
                return "L1"
            elif sz_kb <= tiers["L2"]:
                return "L2"
            elif sz_kb <= tiers["HBM"]:
                return "HBM"
            return "DRAM"

        _tier = _get_tier(_tensor_kb, _tiers)
        _tier_colors = {"L1": COLORS["GreenLine"], "L2": COLORS["BlueLine"],
                        "HBM": COLORS["OrangeLine"], "DRAM": COLORS["RedLine"]}
        _tier_color = _tier_colors[_tier]
        _access_ns = TIER_LATENCY_NS[_tier]
        _latency_ratio = _access_ns / TIER_LATENCY_NS["L2"]

        _batch_range = np.arange(1, 513)
        _sizes_kb = _batch_range * _width * 4 / 1024
        _latencies = np.array([TIER_LATENCY_NS[_get_tier(s, _tiers)] for s in _sizes_kb])

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_batch_range.tolist(), y=_latencies.tolist(),
            mode="lines", name="Access Latency",
            line=dict(color=COLORS["BlueLine"], width=2.5),
            fill="tozeroy", fillcolor="rgba(0,99,149,0.1)",
        ))
        _fig.add_trace(go.Scatter(
            x=[_batch], y=[_access_ns],
            mode="markers", name="Current Setting",
            marker=dict(size=14, color=_tier_color, symbol="diamond",
                        line=dict(width=2, color="white")),
        ))
        for _tier_name, _tier_cap in _tiers.items():
            _boundary_batch = max(1, int(_tier_cap / (_width * 4 / 1024)))
            if 1 < _boundary_batch < 512:
                _fig.add_vline(x=_boundary_batch, line_dash="dash",
                               line_color=COLORS["OrangeLine"], opacity=0.6,
                               annotation_text=f"{_tier_name} limit", annotation_position="top")
        _fig.update_layout(
            height=360, xaxis_title="Batch Size", yaxis_title="Access Latency (ns)",
            yaxis_type="log",
            title=f"Memory Tier Latency vs. Batch Size -- {'Mobile' if _is_mobile else 'Cloud'} (width={_width})",
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _oom = _tensor_kb > _tiers["HBM"]
        if _oom:
            items.append(mo.callout(mo.md(
                f"**OOM -- Activation tensor ({_tensor_kb:,.0f} KB) exceeds device memory "
                f"({_tiers['HBM']:,} KB).** Reduce batch size or layer width."
            ), kind="danger"))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_tier_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Tensor Size</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_tier_color};">{_tensor_kb:,.1f} KB</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_tier_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Memory Tier</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_tier_color};">{_tier}</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_tier_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Access Latency</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_tier_color};">{_access_ns:.0f} ns</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['RedLine'] if _latency_ratio > 5 else COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">vs. L2 Baseline</div>
                <div style="font-size:1.5rem; font-weight:800;
                     color:{COLORS['RedLine'] if _latency_ratio > 5 else COLORS['GreenLine']};">{_latency_ratio:.0f}x</div>
            </div>
        </div>
        """))

        _pred = partB_prediction.value
        if _pred == "10x":
            items.append(mo.callout(mo.md(
                "**Correct.** On mobile, doubling from 16 KB to 32 KB can cross the L2 boundary "
                "into HBM, triggering a ~20x latency jump. Memory hierarchies create step "
                "functions, not gradual slopes. On cloud with 50 MB L2, the same doubling "
                "stays comfortably in cache."
            ), kind="success"))
        elif _pred == "2x":
            items.append(mo.callout(mo.md(
                "**That assumes flat memory -- but memory is tiered.** A 2x increase in tensor "
                "size that crosses a tier boundary triggers a 10-20x latency jump. Switch the "
                "context to Mobile and watch at the L2 boundary."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                "**The answer depends on context.** On mobile, crossing L2->HBM gives ~20x. "
                "On cloud, 32 KB fits comfortably in 50 MB L2, so ~2x is roughly correct."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Memory Tier Placement": mo.md(f"""
**Tensor size:** `batch * width * 4 bytes = {_batch} * {_width} * 4 = {_tensor_bytes:,} bytes = {_tensor_kb:,.1f} KB`

**Cliff at boundary:** L2 (5 ns) to HBM (100 ns) = **20x jump**

Source: @sec-nn-computation-memory-hierarchy
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER: The Width-Squared Surprise
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Research Lead, NeuralEdge Inc.
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We doubled the hidden layer width and expected the model to be twice
                as expensive. But our profiler shows a 3.8x increase in FLOPs. There must
                be a bug in the profiler. Can you verify?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Dr. Liang Wei, Research Lead &middot; NeuralEdge Inc.
            </div>
        </div>
        """))

        items.append(mo.md("""
## Dense Layer FLOPs Scale as O(width^2)

For a dense (fully connected) layer: `FLOPs = 2 * width_in * width_out`

For hidden-to-hidden layers, both dimensions are the hidden width W,
so FLOPs = 2W^2. Doubling W yields 2*(2W)^2 = 8W^2 -- a **4x increase**.

For a 3-layer MLP (784 -> W -> W -> 10):
- **Input-to-hidden**: 2 * 784 * W (linear in W)
- **Hidden-to-hidden**: 2 * W * W (quadratic in W)
- **Hidden-to-output**: 2 * W * 10 (linear in W)

The quadratic term dominates as W grows.
        """))

        items.append(partC_prediction)

        if partC_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the FLOP scaling simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(partC_width)

        _w = partC_width.value
        _input_dim = 784
        _output_dim = 10
        _flops_l1 = 2 * _input_dim * _w
        _flops_l2 = 2 * _w * _w
        _flops_l3 = 2 * _w * _output_dim
        _total_flops = _flops_l1 + _flops_l2 + _flops_l3

        _ref_w = 128
        _ref_flops = 2 * _input_dim * _ref_w + 2 * _ref_w * _ref_w + 2 * _ref_w * _output_dim
        _flops_ratio = _total_flops / _ref_flops if _ref_flops > 0 else 1
        _width_ratio = _w / _ref_w

        _fig = go.Figure()
        for _name, _val, _col in zip(
            ["Input->Hidden", "Hidden->Hidden", "Hidden->Output"],
            [_flops_l1, _flops_l2, _flops_l3],
            [COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"]],
        ):
            _fig.add_trace(go.Bar(name=_name, x=["Current Width"], y=[_val],
                                  marker_color=_col, opacity=0.88))
        _fig.update_layout(barmode="stack", height=300, yaxis_title="FLOPs",
                           title=f"Per-Layer FLOPs -- MLP (784->{_w}->{_w}->10)",
                           legend=dict(orientation="h", y=1.12, x=0))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _widths = np.arange(32, 2049, 32)
        _total_curve = (2 * _input_dim * _widths + 2 * _widths**2 + 2 * _widths * _output_dim).astype(float)
        _linear_ref = _total_curve[0] * (_widths / _widths[0])

        _fig2 = go.Figure()
        _fig2.add_trace(go.Scatter(x=_widths.tolist(), y=_total_curve.tolist(),
                                   mode="lines", name="Actual FLOPs (quadratic)",
                                   line=dict(color=COLORS["RedLine"], width=2.5)))
        _fig2.add_trace(go.Scatter(x=_widths.tolist(), y=_linear_ref.tolist(),
                                   mode="lines", name="If linear (2x width = 2x FLOPs)",
                                   line=dict(color=COLORS["Grey"], width=2, dash="dash")))
        _fig2.add_trace(go.Scatter(x=[_w], y=[float(_total_flops)],
                                   mode="markers", name="Current Width",
                                   marker=dict(size=12, color=COLORS["BlueLine"], symbol="diamond",
                                               line=dict(width=2, color="white"))))
        _fig2.update_layout(height=360, xaxis_title="Hidden Width", yaxis_title="Total FLOPs",
                            title="FLOPs Scaling: Actual (Quadratic) vs. Linear Assumption",
                            legend=dict(orientation="h", y=1.12, x=0))
        apply_plotly_theme(_fig2)
        items.append(mo.as_html(_fig2))

        _qcolor = COLORS["RedLine"] if _flops_ratio > 3 else COLORS["OrangeLine"] if _flops_ratio > 1.5 else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Width Ratio (vs 128)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_width_ratio:.1f}x</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_qcolor}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">FLOPs Ratio (vs 128)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_qcolor};">{_flops_ratio:.1f}x</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Total FLOPs</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_total_flops:,.0f}</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Hidden->Hidden Share</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_flops_l2/_total_flops*100:.0f}%</div>
            </div>
        </div>
        """))

        _actual_256 = (2*784*256 + 2*256*256 + 2*256*10) / (2*784*128 + 2*128*128 + 2*128*10)
        _pred = partC_prediction.value
        if _pred == "4x":
            items.append(mo.callout(mo.md(
                f"**Correct.** Doubling width from 128 to 256 increases total FLOPs by "
                f"~{_actual_256:.1f}x. The hidden-to-hidden layer scales as W^2. "
                f"Architecture decisions -- not just hardware -- dominate the Operations term."
            ), kind="success"))
        elif _pred == "2x":
            items.append(mo.callout(mo.md(
                f"**Linear intuition fails here.** Doubling width actually yields ~{_actual_256:.1f}x "
                f"FLOPs. The hidden-to-hidden layer has FLOPs=2*W*W. Double W: 4x that layer."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Not quite.** The actual increase is ~{_actual_256:.1f}x. The quadratic layer "
                f"dominates, but input/output layers scale linearly, pulling total below 4x."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: FLOPs Scaling": mo.md(f"""
**Per-layer FLOPs for MLP (784->{_w}->{_w}->10):**
```
Layer 1: 2 * 784 * {_w} = {_flops_l1:,}
Layer 2: 2 * {_w} * {_w} = {_flops_l2:,}
Layer 3: 2 * {_w} * 10  = {_flops_l3:,}
Total = {_total_flops:,}
```
Doubling 128->256: ratio = {_actual_256:.2f}x (not 2x!)

Source: @sec-nn-computation-flop-counting
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER: Forward vs. Backward Memory
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Training Infra Lead, NeuralEdge Inc.
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Inference works perfectly on our edge device. We want to add on-device
                fine-tuning, but training immediately runs OOM. The model is the same. Why
                does training need so much more memory than inference?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; James Okafor, Training Infra Lead &middot; NeuralEdge Inc.
            </div>
        </div>
        """))

        items.append(mo.md("""
## Inference Discards; Training Stores Everything

During **inference**, each layer's activations can be discarded after the next
layer consumes them. Memory usage is approximately constant.

During **training**, backpropagation needs every intermediate activation to
compute gradients. All layer activations must be stored **simultaneously**.

```
Inference memory = weights + max_single_layer_activation
Training memory  = weights + gradients + ALL_layer_activations
```

The training-to-inference ratio grows with depth and batch size.
        """))

        items.append(partD_prediction)

        if partD_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the memory comparison."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partD_phase, partD_depth, partD_batch, partD_width_d], justify="start"))

        _phase = partD_phase.value
        _depth = partD_depth.value
        _batch = partD_batch.value
        _w = partD_width_d.value
        _is_training = _phase == "training"
        _bpp = 4
        _input_dim = 784
        _output_dim = 10

        _params_l1 = _input_dim * _w
        _params_hidden = max(0, _depth - 2) * _w * _w
        _params_out = _w * _output_dim
        _total_params = _params_l1 + _params_hidden + _params_out
        _weight_mb = _total_params * _bpp / (1024 * 1024)

        _act_per_layer_mb = _batch * _w * _bpp / (1024 * 1024)

        if _is_training:
            _act_total_mb = _act_per_layer_mb * _depth
            _grad_mb = _weight_mb
            _total_mb = _weight_mb + _grad_mb + _act_total_mb
        else:
            _act_total_mb = _act_per_layer_mb
            _grad_mb = 0.0
            _total_mb = _weight_mb + _act_total_mb

        _inference_mb = _weight_mb + _act_per_layer_mb
        _ratio = _total_mb / _inference_mb if _inference_mb > 0 else 1

        _fig = go.Figure()
        _colors_bar = [COLORS["BlueLine"], COLORS["GreenLine"],
                       COLORS["RedLine"] if _is_training else COLORS["OrangeLine"]]
        for _name, _val, _col in zip(["Weights", "Gradients", "Activations"],
                                      [_weight_mb, _grad_mb, _act_total_mb], _colors_bar):
            _fig.add_trace(go.Bar(name=_name, x=[_phase.capitalize()], y=[_val],
                                  marker_color=_col, opacity=0.88))
        _fig.add_hline(y=80_000, line_dash="dash", line_color=COLORS["BlueLine"],
                       annotation_text="H100 (80 GB)", annotation_position="right")
        _fig.add_hline(y=8_000, line_dash="dash", line_color=COLORS["OrangeLine"],
                       annotation_text="iPhone (8 GB)", annotation_position="right")
        _fig.update_layout(barmode="stack", height=380, yaxis_title="Memory (MB)",
                           title=f"Memory Breakdown -- {_phase.capitalize()} (depth={_depth}, batch={_batch}, width={_w})",
                           legend=dict(orientation="h", y=1.12, x=0))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _depths = np.arange(3, 51)
        _inf_mem = np.array([
            (_input_dim * _w + max(0, d-2) * _w * _w + _w * _output_dim) * _bpp / (1024*1024)
            + _act_per_layer_mb for d in _depths])
        _train_mem = np.array([
            ((_input_dim * _w + max(0, d-2) * _w * _w + _w * _output_dim) * _bpp * 2
            + d * _batch * _w * _bpp) / (1024*1024) for d in _depths])

        _fig2 = go.Figure()
        _fig2.add_trace(go.Scatter(x=_depths.tolist(), y=_inf_mem.tolist(),
                                   mode="lines", name="Inference",
                                   line=dict(color=COLORS["GreenLine"], width=2.5)))
        _fig2.add_trace(go.Scatter(x=_depths.tolist(), y=_train_mem.tolist(),
                                   mode="lines", name="Training",
                                   line=dict(color=COLORS["RedLine"], width=2.5),
                                   fill="tonexty", fillcolor="rgba(203,32,45,0.1)"))
        _fig2.add_trace(go.Scatter(x=[_depth], y=[_total_mb], mode="markers", name="Current",
                                   marker=dict(size=12, color=COLORS["BlueLine"], symbol="diamond",
                                               line=dict(width=2, color="white"))))
        _fig2.update_layout(height=360, xaxis_title="Network Depth (layers)", yaxis_title="Memory (MB)",
                            title=f"Memory vs. Depth (batch={_batch}, width={_w})",
                            legend=dict(orientation="h", y=1.12, x=0))
        apply_plotly_theme(_fig2)
        items.append(mo.as_html(_fig2))

        _ratio_color = COLORS["RedLine"] if _ratio > 5 else COLORS["OrangeLine"] if _ratio > 2 else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Weights</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_weight_mb:,.1f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Gradients</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_grad_mb:,.1f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_colors_bar[2]}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Activations</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_colors_bar[2]};">{_act_total_mb:,.1f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_ratio_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Train/Inference Ratio</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_ratio_color};">{_ratio:.1f}x</div>
            </div>
        </div>
        """))

        if _total_mb > 8_000 and _is_training:
            items.append(mo.callout(mo.md(
                f"**OOM on mobile (8 GB).** Training requires {_total_mb:,.0f} MB. "
                f"On-device fine-tuning is infeasible. Reduce batch size, depth, or width."
            ), kind="danger"))

        _pred = partD_prediction.value
        if _pred in ("200mb", "500mb"):
            items.append(mo.callout(mo.md(
                f"**Correct range.** Training requires {_ratio:.1f}x the memory of inference "
                f"at current settings. Stored activations dominate: training keeps all {_depth} "
                f"layers simultaneously for backpropagation."
            ), kind="success"))
        elif _pred == "100mb":
            items.append(mo.callout(mo.md(
                "**You accounted for gradients but forgot activations.** Gradients double "
                "weight memory (2x), but stored activations add depth * batch * width per "
                "layer -- the dominant term at deep networks."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Training is much more expensive than inference.** The actual ratio is "
                f"{_ratio:.1f}x. Backpropagation requires storing all intermediate activations."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Training vs. Inference Memory": mo.md(f"""
**Inference:** {_weight_mb:,.1f} + {_act_per_layer_mb:,.1f} = {_inference_mb:,.1f} MB
**Training:** {_weight_mb:,.1f} + {_grad_mb:,.1f} + {_act_total_mb:,.1f} = {_total_mb:,.1f} MB
**Ratio:** {_ratio:.1f}x

Note: Optimizer state (Lab 08) adds 2 more weight-sized buffers for Adam.
Source: @sec-nn-computation-backprop-memory
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        items = []
        items.append(mo.md("""
## Synthesis: Design a Mobile Inference Pipeline

You are deploying a 10-layer vision model on a mobile NPU (iPhone, 8 GB RAM,
5W power budget) for both real-time inference (30 FPS) and on-device fine-tuning.
        """))

        items.append(mo.callout(mo.md("""
1. **Activation function choice**: Which activation and why? What is the impact on mobile?
2. **Maximum batch size**: Before activations cross the L2 boundary on mobile
   (32 MB L2, width=256), what is the maximum batch size?
   *Hint: batch * 256 * 4 bytes <= 32 MB*
3. **Training memory**: At depth=10, batch=32, width=256, what is the total
   (weights + gradients + activations, ignoring optimizer state)?
4. **Feasibility**: Can on-device fine-tuning fit in 8 GB?
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
                    <strong>1. The Transistor Tax is deployment-dependent.</strong>
                    Sigmoid costs 50x more transistors than ReLU. On cloud: invisible (<1%).
                    On mobile: ~23% of inference time. Architecture choices free on one target
                    are expensive on another.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Memory creates cliffs, not slopes.</strong>
                    Crossing L2 to HBM triggers a 10-20x latency jump. Memory-aware design
                    means keeping tensors within tier boundaries.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>3. Width scales quadratically.</strong>
                    Doubling hidden width increases dense layer FLOPs by ~4x. The Operations
                    term in the Iron Law is dominated by architecture decisions.
                </div>
                <div>
                    <strong>4. Training stores everything inference discards.</strong>
                    Backpropagation requires all intermediate activations simultaneously,
                    creating a 4-10x memory multiplier over inference.
                </div>
            </div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab 06: Network Architectures</strong> -- This lab showed how
                    individual layer costs scale. Lab 06 asks: when you compose layers into
                    architectures (MLP, CNN, Transformer), which designs are physically feasible?
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-nn-computation for transistor costs,
                    memory hierarchies, and FLOP scaling.<br/>
                    <strong>Build:</strong> TinyTorch Module 05 -- implement forward and
                    backward passes for dense layers and activation functions.
                </div>
            </div>
        </div>
        """))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A \u2014 The Transistor Tax":       build_part_a(),
        "Part B \u2014 The Memory Cliff":         build_part_b(),
        "Part C \u2014 Width-Squared Surprise":   build_part_c(),
        "Part D \u2014 Forward vs. Backward":     build_part_d(),
        "Synthesis":                              build_synthesis(),
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
        <span class="hud-value">05 &middot; The Transistor Tax</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">CHAPTER&nbsp;5</span>
        <span class="hud-value">Neural Computation</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
