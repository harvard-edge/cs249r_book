import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 05: THE ACTIVATION TAX
#
# Chapter: Neural Computation (@sec-neural-computation)
# Core Invariant: Activation functions have wildly different hardware costs.
#   ReLU ≈ free (50 transistors). Sigmoid/Tanh require exponential computation
#   (2,500 transistors) — a 50× silicon penalty. The memory hierarchy adds a
#   second multiplier: L1 → L2 → HBM → DRAM each 10× slower and larger.
#
# Two deployment contexts: Cloud (H100) vs Mobile (NPU).
# 2-Act structure: ~35-40 minutes total.
#
# Act I — The Activation Cost Blindspot (12–15 min)
#   Prediction: How much more expensive is Sigmoid than ReLU in transistors?
#   Instrument: Activation cost bar chart, layer-by-layer selector.
#   Reveal: Swapping all Sigmoid→ReLU saves 47× activation time.
#   Reflection: Why does GELU cost more than ReLU despite similar accuracy?
#
# Act II — The Memory Hierarchy (20–25 min)
#   Introduces the Memory Ledger instrument for the first time.
#   Prediction: What fraction of a 3×3 conv's activations fit in L2 on mobile?
#   Instrument: Layer size + batch → Memory Ledger with tier coloring.
#   Failure state: OOM when total_activation_memory > device RAM.
#   Reflection: Why does batch size affect memory-boundedness?
#
# Design Ledger save: chapter=5, context, activation_choice, oom_triggered,
#   cache_miss_rate.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CELL 0: SETUP (hide_code=False — leave visible for instructor inspection)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    # ── Hardware constants from chapter (@sec-neural-computation-transistor-tax)
    # ReLU: comparator + mux (~50 transistors)
    # Sigmoid/Tanh: exponential approximation unit (~2,500 transistors)
    # Ratio: 2500 / 50 = 50× — "The Transistor Tax"
    RELU_TRANSISTORS    = 50        # single comparator + mux
    SIGMOID_TRANSISTORS = 2500      # exponential Taylor/lookup unit
    ACTIVATION_TAX_RATIO = SIGMOID_TRANSISTORS / RELU_TRANSISTORS  # 50

    # Memory hierarchy latency multipliers (from footnote fn-memory-wall-nn)
    # L1 cache: ~1 ns; main memory: ~100 ns → 100× gap
    MEM_L1_MULT   = 1       # L1 SRAM (fastest, smallest)
    MEM_L2_MULT   = 4       # L2 cache (~4× L1 latency)
    MEM_HBM_MULT  = 10      # HBM / GDDR (~10× L1)
    MEM_DRAM_MULT = 100     # Main DRAM (~100× L1)

    # Cloud context: H100 SXM5
    H100_RAM_GB      = 80       # GB HBM3e
    H100_L2_CACHE_MB = 50       # MB L2 (H100 SXM5 spec)
    H100_L1_CACHE_KB = 6 * 1024 # KB per SM × 108 SMs (shared L1 budget approx)
    # For per-layer L1 we use a practical per-SM allocation
    H100_L1_PER_SM_KB = 256     # KB per SM L1/shared mem

    # Mobile NPU context
    MOBILE_RAM_GB     = 8        # GB
    MOBILE_L2_CACHE_KB = 512     # KB L2
    MOBILE_L1_CACHE_KB = 64      # KB L1

    ledger = DesignLedger()
    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme, go, np,
        RELU_TRANSISTORS, SIGMOID_TRANSISTORS, ACTIVATION_TAX_RATIO,
        MEM_L1_MULT, MEM_L2_MULT, MEM_HBM_MULT, MEM_DRAM_MULT,
        H100_RAM_GB, H100_L2_CACHE_MB, H100_L1_PER_SM_KB,
        MOBILE_RAM_GB, MOBILE_L2_CACHE_KB, MOBILE_L1_CACHE_KB,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: HEADER
# ─────────────────────────────────────────────────────────────────────────────

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
                Machine Learning Systems · Volume I · Lab 05
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Activation Tax
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 700px; line-height: 1.65;">
                Not all activation functions cost the same. ReLU is a comparator.
                Sigmoid requires an exponential. The memory hierarchy multiplies
                every cost by 10× per tier. Where you put your data is as important
                as what you compute with it.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: Activation Cost Blindspot · 12–15 min
                </span>
                <span style="background: rgba(204,85,0,0.15); color: #fdba74;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(204,85,0,0.25);">
                    Act II: Memory Hierarchy · 20–25 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min total
                </span>
            </div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px;">
                <span class="badge badge-info">First use: Memory Ledger instrument</span>
                <span class="badge badge-info">First use: Activation Comparator</span>
            </div>
        </div>
        """),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 2: BRIEFING ─────────────────────────────────────────────────────────
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the transistor cost ratio</strong> between ReLU (~50 transistors) and Sigmoid (~2,500 transistors) and verify the 50&times; silicon penalty using the Activation Comparator instrument.</div>
                <div style="margin-bottom: 3px;">2. <strong>Predict which memory hierarchy tier</strong> a given layer&rsquo;s activations will land in for a specified batch size, channel count, and spatial dimension, given L1/L2/HBM (High Bandwidth Memory)/DRAM capacity boundaries.</div>
                <div style="margin-bottom: 3px;">3. <strong>Identify the batch size threshold</strong> where activation memory exceeds the mobile L2 cache (512 KB) and triggers the 10&times; latency penalty of HBM access for a 3&times;3 convolutional layer.</div>
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
                    Activation function definitions from @sec-neural-computation-artificial-neuron-computing-primitive-45b4 &middot;
                    Memory hierarchy tiers from @sec-neural-computation-transistor-tax
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
                &ldquo;ReLU and Sigmoid produce similar accuracy on the same network &mdash;
                so why does the choice of activation function determine whether your model
                fits in cache or spills to memory 100&times; slower, and whether gradient
                signals survive 20 layers of backpropagation?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: READING ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete these sections before this lab:

    - **@sec-neural-computation-artificial-neuron-computing-primitive-45b4** — The artificial neuron: inputs, weights, activation functions, MAC operations
    - **@sec-neural-computation-transistor-tax** — The Transistor Tax: ReLU vs Sigmoid silicon cost (50×)
    - **@sec-neural-computation-computational-implementation-details-1ecc** — Training memory decomposition: weights, gradients, optimizer state, activations
    - **Footnote fn-memory-wall-nn** — L1 cache ~1 ns vs main memory ~100 ns: the 100× latency gap
    """), kind="info")
    return


# ─────────────────────────────────────────────────────────────────────────────
# ─── CELL 4: CONTEXT_TOGGLE ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={"Cloud (H100 — 80 GB HBM)": "cloud", "Mobile (NPU — 8 GB)": "mobile"},
        value="Cloud (H100 — 80 GB HBM)",
        label="Deployment context:",
        inline=True,
    )
    context_toggle
    return (context_toggle,)


# ─────────────────────────────────────────────────────────────────────────────
# ─── CELL 4b: CONTEXT SPECS DISPLAY ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS):
    _ctx = context_toggle.value
    if _ctx == "cloud":
        _accent = COLORS["Cloud"]
        _bg = "#f0f4ff"
        _border = "#c7d2fe"
        _specs = [
            ("Device", "NVIDIA H100 SXM5"),
            ("HBM Capacity", "80 GB"),
            ("L2 Cache", "40 MB"),
            ("L1 / SM", "256 KB"),
            ("Memory BW", "3,350 GB/s"),
            ("Power Budget", "700 W TDP"),
        ]
    else:
        _accent = COLORS["Mobile"]
        _bg = "#fff7ed"
        _border = "#fed7aa"
        _specs = [
            ("Device", "Mobile NPU"),
            ("RAM Capacity", "8 GB"),
            ("L2 Cache", "512 KB"),
            ("L1 Cache", "64 KB"),
            ("Memory BW", "68 GB/s"),
            ("Power Budget", "5 W sustained"),
        ]

    _rows = "".join(
        f'<div style="display:flex; justify-content:space-between; padding:4px 0; '
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
# ZONE B: ACT I -- CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["BlueLine"]
    _act_title    = "The Activation Cost Blindspot"
    _act_duration = "12&ndash;15 min"
    _act_why      = ("You expect activation functions to be &ldquo;free&rdquo; &mdash; just a "
                     "nonlinearity tacked onto a matrix multiply. The Transistor Tax shows "
                     "that Sigmoid costs 50&times; more silicon than ReLU. For a mobile AR "
                     "model running 4 sigmoid layers at 12 FPS, swapping to ReLU is not an "
                     "accuracy trade-off; it is an architectural decision with a measurable "
                     "thermal and throughput consequence.")

    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="margin: 8px 0 12px 0;">
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
        mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeLL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Mobile AR Team Lead
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "Our mobile AR model runs at 12 FPS &mdash; half our 24 FPS target. The team
                wants to add a larger backbone. Before you approve that, look at what
                activations are already costing us. We have four layers that could each
                use different activation functions. Right now they all use Sigmoid."
            </div>
        </div>
        """),
        mo.md("""
        The chapter established the **Transistor Tax** (@sec-neural-computation-transistor-tax):
        choosing an activation function is a hardware design decision, not just a mathematical
        one. ReLU requires a single comparator. Sigmoid requires an exponential unit built from
        lookup tables or Taylor series. The silicon cost difference is not marginal — it is the
        difference between 50 transistors and 2,500.

        Before you explore the instruments, commit to a prediction.
        """),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: ACT I PREDICTION LOCK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) About the same cost — both are just nonlinear functions": "1x",
            "B) ~5× more — Sigmoid requires a division operation": "5x",
            "C) ~50× more — Sigmoid needs an exponential unit": "50x",
            "D) ~500× more — Sigmoid requires iterative convergence": "500x",
        },
        label="Compared to ReLU, a Sigmoid activation requires approximately how much more silicon (transistor count)?",
    )
    mo.vstack([
        mo.Html("""
        <div style="background:#1e293b; border-radius:8px; padding:14px 20px;
                    border-left:4px solid #6366f1; margin:8px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:#a5b4fc;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
                Prediction Lock — Act I
            </div>
            <div style="font-size:0.88rem; color:#94a3b8; line-height:1.5;">
                Select your prediction before the instruments unlock. Your answer is
                recorded and compared to the actual result at the end of this act.
            </div>
        </div>
        """),
        act1_prediction,
    ])
    return (act1_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Act I instruments."),
            kind="warn",
        ),
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: ACT I INSTRUMENT — ACTIVATION COST COMPARATOR
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The Activation Cost Comparator

    The chart below shows the silicon cost of each activation function in transistors,
    relative to ReLU. Move through the four layer selectors to assign an activation
    function to each layer in the AR model. The total inference cost updates live.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Layer activation selectors — 4 layers in the mobile AR model
    _act_options = {
        "ReLU   (max(0,x) — comparator)": "relu",
        "Leaky ReLU   (max(αx,x))": "leaky_relu",
        "Sigmoid   (1/(1+e^−x))": "sigmoid",
        "Tanh   ((e^x−e^−x)/(e^x+e^−x))": "tanh",
        "GELU   (x·Φ(x) — erf approx)": "gelu",
        "Softmax   (exp / sum(exp))": "softmax",
    }
    act1_layer1 = mo.ui.dropdown(
        options=_act_options, value="Sigmoid   (1/(1+e^−x))", label="Layer 1 activation"
    )
    act1_layer2 = mo.ui.dropdown(
        options=_act_options, value="Sigmoid   (1/(1+e^−x))", label="Layer 2 activation"
    )
    act1_layer3 = mo.ui.dropdown(
        options=_act_options, value="Sigmoid   (1/(1+e^−x))", label="Layer 3 activation"
    )
    act1_layer4 = mo.ui.dropdown(
        options=_act_options, value="Sigmoid   (1/(1+e^−x))", label="Layer 4 activation"
    )
    mo.vstack([
        mo.md("**Assign an activation function to each of the four AR model layers:**"),
        mo.hstack([act1_layer1, act1_layer2], justify="start", gap="2rem"),
        mo.hstack([act1_layer3, act1_layer4], justify="start", gap="2rem"),
    ])
    return (act1_layer1, act1_layer2, act1_layer3, act1_layer4)


@app.cell(hide_code=True)
def _(
    mo, go, np, apply_plotly_theme, COLORS,
    act1_layer1, act1_layer2, act1_layer3, act1_layer4,
    RELU_TRANSISTORS, SIGMOID_TRANSISTORS, ACTIVATION_TAX_RATIO,
    context_toggle,
):
    # ── Activation transistor cost model ─────────────────────────────────────
    # Source: @sec-neural-computation-transistor-tax
    # ReLU: ~50 transistors (comparator + mux)
    # Sigmoid/Tanh: ~2,500 transistors (exponential unit)
    # GELU: ~1,500 transistors (erf polynomial approximation, cheaper than full exp)
    # Leaky ReLU: ~60 transistors (comparator + mux + scale)
    # Softmax: ~3,000 transistors (exp + accumulate + divide, worst case)
    _TRANSISTOR_COST = {
        "relu":       RELU_TRANSISTORS,        # 50    — comparator + mux
        "leaky_relu": 60,                      # 60    — comparator + scale + mux
        "sigmoid":    SIGMOID_TRANSISTORS,     # 2,500 — exponential unit
        "tanh":       SIGMOID_TRANSISTORS,     # 2,500 — same exponential complexity
        "gelu":       1500,                    # 1,500 — polynomial erf approximation
        "softmax":    3000,                    # 3,000 — exp + sum + divide
    }
    _CYCLE_COST = {
        "relu":       1,    # 1 cycle
        "leaky_relu": 2,    # 2 cycles
        "sigmoid":    25,   # 20–30 cycles (exponential pipeline)
        "tanh":       25,   # 20–30 cycles
        "gelu":       8,    # 6–10 cycles (polynomial)
        "softmax":    30,   # exp + reduction
    }
    _ACT_NAMES = {
        "relu":       "ReLU",
        "leaky_relu": "Leaky ReLU",
        "sigmoid":    "Sigmoid",
        "tanh":       "Tanh",
        "gelu":       "GELU",
        "softmax":    "Softmax",
    }

    _selected = [
        act1_layer1.value,
        act1_layer2.value,
        act1_layer3.value,
        act1_layer4.value,
    ]
    _layer_names = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]

    # ── Reference bar chart: transistor cost comparison ───────────────────────
    _all_acts  = list(_TRANSISTOR_COST.keys())
    _all_costs = [_TRANSISTOR_COST[a] for a in _all_acts]
    _all_names = [_ACT_NAMES[a] for a in _all_acts]
    _bar_colors = [
        COLORS["RedLine"] if c >= 1000 else
        COLORS["OrangeLine"] if c >= 200 else
        COLORS["GreenLine"]
        for c in _all_costs
    ]

    _fig_ref = go.Figure()
    _fig_ref.add_trace(go.Bar(
        x=_all_names, y=_all_costs,
        marker_color=_bar_colors,
        text=[f"{c:,}" for c in _all_costs],
        textposition="outside",
        name="Transistor cost",
    ))
    _fig_ref.add_hline(
        y=RELU_TRANSISTORS, line_dash="dot",
        line_color=COLORS["GreenLine"], line_width=2,
        annotation_text="ReLU baseline",
        annotation_position="right",
    )
    _fig_ref.update_layout(
        title_text="Silicon Cost by Activation Function (transistors)",
        xaxis_title="Activation Function",
        yaxis_title="Transistor Count (log scale)",
        yaxis_type="log",
        showlegend=False,
        height=320,
    )
    apply_plotly_theme(_fig_ref)

    # ── Live: selected layers total cost ─────────────────────────────────────
    _selected_costs   = [_TRANSISTOR_COST[a] for a in _selected]
    _selected_cycles  = [_CYCLE_COST[a] for a in _selected]
    _selected_colors  = [
        COLORS["RedLine"] if c >= 1000 else
        COLORS["OrangeLine"] if c >= 200 else
        COLORS["GreenLine"]
        for c in _selected_costs
    ]

    _fig_layers = go.Figure()
    _fig_layers.add_trace(go.Bar(
        x=_layer_names, y=_selected_costs,
        marker_color=_selected_colors,
        text=[f"{_ACT_NAMES[a]}<br>{c:,} transistors" for a, c in zip(_selected, _selected_costs)],
        textposition="outside",
        name="Selected layers",
    ))
    _fig_layers.update_layout(
        title_text="Your Layer Assignments — Silicon Cost",
        xaxis_title="Layer",
        yaxis_title="Transistors (log scale)",
        yaxis_type="log",
        showlegend=False,
        height=320,
    )
    apply_plotly_theme(_fig_layers)

    # ── Metrics ───────────────────────────────────────────────────────────────
    _total_transistors    = sum(_selected_costs)
    _total_cycles         = sum(_selected_cycles)
    _relu_total           = RELU_TRANSISTORS * 4
    _ratio_vs_all_relu    = _total_transistors / _relu_total
    _cycle_ratio          = _total_cycles / 4.0  # vs 1 cycle/layer ReLU baseline

    # Context-dependent inference latency impact
    # Mobile NPU: activation compute is tighter because total throughput ~35 TOPS
    # A 50× heavier activation function occupies ~3–5% of inference budget vs <0.1%
    _ctx = context_toggle.value
    if _ctx == "mobile":
        _act_time_baseline_us  = 0.8   # µs total activation time with all-ReLU
        _act_time_current_us   = _act_time_baseline_us * (_total_cycles / 4.0)
        _inference_total_us    = 42.0  # µs total inference for this layer config
        _act_pct               = 100.0 * _act_time_current_us / _inference_total_us
        _speedup_if_all_relu   = _act_time_current_us / _act_time_baseline_us
        _ctx_label             = "Mobile NPU"
    else:
        _act_time_baseline_us  = 0.05  # µs on H100 (activation near-free in compute terms)
        _act_time_current_us   = _act_time_baseline_us * (_total_cycles / 4.0)
        _inference_total_us    = 2.1
        _act_pct               = 100.0 * _act_time_current_us / _inference_total_us
        _speedup_if_all_relu   = _act_time_current_us / _act_time_baseline_us
        _ctx_label             = "Cloud H100"

    # Color coding
    _ratio_color = (
        COLORS["RedLine"] if _ratio_vs_all_relu > 10
        else COLORS["OrangeLine"] if _ratio_vs_all_relu > 3
        else COLORS["GreenLine"]
    )
    _pct_color = (
        COLORS["RedLine"] if _act_pct > 5
        else COLORS["OrangeLine"] if _act_pct > 1
        else COLORS["GreenLine"]
    )

    mo.vstack([
        mo.hstack([_fig_ref, _fig_layers], justify="center"),
        mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin-top:16px; flex-wrap:wrap;">
            <div style="padding:16px 20px; border:1px solid #e2e8f0; border-radius:10px;
                        width:200px; text-align:center; background:white; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                <div style="color:#475569; font-size:0.82rem; font-weight:600; margin-bottom:4px;">
                    Total Silicon vs All-ReLU
                </div>
                <div style="font-size:2rem; font-weight:800; color:{_ratio_color};">
                    {_ratio_vs_all_relu:.1f}×
                </div>
                <div style="color:#94a3b8; font-size:0.75rem;">{_total_transistors:,} transistors</div>
            </div>
            <div style="padding:16px 20px; border:1px solid #e2e8f0; border-radius:10px;
                        width:200px; text-align:center; background:white; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                <div style="color:#475569; font-size:0.82rem; font-weight:600; margin-bottom:4px;">
                    Activation Time ({_ctx_label})
                </div>
                <div style="font-size:2rem; font-weight:800; color:{_pct_color};">
                    {_act_pct:.1f}%
                </div>
                <div style="color:#94a3b8; font-size:0.75rem;">of total inference</div>
            </div>
            <div style="padding:16px 20px; border:1px solid #e2e8f0; border-radius:10px;
                        width:200px; text-align:center; background:white; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                <div style="color:#475569; font-size:0.82rem; font-weight:600; margin-bottom:4px;">
                    Speedup if All-ReLU
                </div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_speedup_if_all_relu:.1f}×
                </div>
                <div style="color:#94a3b8; font-size:0.75rem;">activation compute only</div>
            </div>
        </div>
        """),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7: ACT I PHYSICS PEEK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, RELU_TRANSISTORS, SIGMOID_TRANSISTORS, ACTIVATION_TAX_RATIO):
    mo.accordion({
        "The governing equation — Transistor Tax": mo.md(f"""
        **The Transistor Tax** (@sec-neural-computation-transistor-tax):

        ```
        ReLU  cost = {RELU_TRANSISTORS} transistors    (comparator + mux)
        Sigmoid cost = {SIGMOID_TRANSISTORS:,} transistors (exponential unit)

        Tax ratio = Sigmoid / ReLU = {int(ACTIVATION_TAX_RATIO)}×
        ```

        **Why exponentials are expensive:**
        Hardware cannot compute `exp(x)` in one clock cycle. It must approximate
        using a piecewise lookup table or a Taylor series expansion:

        ```
        exp(x) ≈ 1 + x + x²/2! + x³/3! + ...  (Taylor, ~8 terms for FP32 precision)
        ```

        Each term requires a multiply-accumulate. A dedicated floating-point exponential
        unit pipelines this over 20–30 cycles. A ReLU does the same job in 1 cycle with
        a single comparison.

        **GELU** uses a polynomial approximation to the Gaussian CDF Φ(x):

        ```
        GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
        ```

        This polynomial is cheaper than a full sigmoid (fewer terms) but still 6–10×
        the cost of ReLU. GELU is accurate-better and hardware-worse than ReLU — a direct
        accuracy-vs-silicon trade-off.

        **Activation cost fraction:**

        ```
        activation_time = layer_neurons × cycles_per_activation / TOPS
        fraction = activation_time / total_inference_time
        ```

        On a mobile NPU with 35 TOPS (INT8): even a 4-layer sigmoid network
        spends ~23% of inference time on activation alone — compute that could
        be reclaimed instantly by switching to ReLU.
        """)
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8: ACT I REVEAL — PREDICTION vs REALITY
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_prediction, ACTIVATION_TAX_RATIO):
    _predicted_map = {"1x": 1, "5x": 5, "50x": 50, "500x": 500}
    _predicted = _predicted_map[act1_prediction.value]
    _actual = int(ACTIVATION_TAX_RATIO)  # 50
    _ratio = _actual / _predicted if _predicted > 0 else float("inf")

    if abs(_ratio - 1.0) < 0.15:
        _kind = "success"
        _verdict = "Correct."
    elif _predicted < _actual:
        _kind = "warn"
        _verdict = "Underestimate."
    else:
        _kind = "warn"
        _verdict = "Overestimate."

    mo.callout(
        mo.md(
            f"**{_verdict}** You predicted **{_predicted}×**. "
            f"The actual silicon cost ratio is **{_actual}×** "
            f"(ReLU: 50 transistors vs Sigmoid: 2,500 transistors). "
            + (
                "" if abs(_ratio - 1.0) < 0.15 else
                f"You were off by **{_ratio:.1f}×**. "
            )
            + """
            This is the **Transistor Tax** from @sec-neural-computation-transistor-tax.
            Selecting Sigmoid over ReLU multiplies the silicon budget of each activation unit
            by 50× — not 2×, not 5×, but fifty times. On a mobile NPU that constraint
            translates directly to heat, battery drain, and missed frame rate targets.
            Swapping all four Sigmoid layers to ReLU drops activation compute by **47×**
            and reclaims roughly 23% of total inference time on mobile.
            """
        ),
        kind=_kind,
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9: ACT I STRUCTURED REFLECTION
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) GELU has more trainable parameters than ReLU": "wrong_params",
            "B) GELU uses a polynomial approximation to erf() requiring multi-step computation": "correct",
            "C) GELU requires backpropagation while ReLU does not": "wrong_backprop",
            "D) GELU is only defined for transformer architectures": "wrong_arch",
        },
        label="Why does GELU outperform ReLU in accuracy but cost more in silicon?",
    )
    mo.vstack([
        mo.md("**Reflection:** Answer before reading the explanation below."),
        act1_reflection,
    ])
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(act1_reflection.value is None, mo.md(""))

    _feedback = {
        "wrong_params": mo.callout(mo.md(
            "**Not quite.** GELU has *zero* additional parameters — it is a fixed "
            "mathematical function, not a learned one. The cost difference is purely "
            "computational: the function itself requires more arithmetic operations per call."
        ), kind="warn"),
        "correct": mo.callout(mo.md(
            "**Correct.** GELU(x) = x · Φ(x) where Φ is the Gaussian CDF. "
            "Hardware cannot compute Φ(x) exactly, so it uses a polynomial approximation: "
            "`GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. "
            "That tanh itself requires an exponential unit. The accuracy benefit comes "
            "from the smooth, probabilistic gating behavior — neurons are not hard-zeroed "
            "as in ReLU, which helps gradient flow. The silicon cost is real and unavoidable: "
            "GELU is 6–10× more expensive than ReLU per evaluation. Transformers "
            "(BERT, GPT) use GELU because they run on cloud hardware where that tax is "
            "affordable; mobile architectures use ReLU or Leaky ReLU where it is not."
        ), kind="success"),
        "wrong_backprop": mo.callout(mo.md(
            "**Not quite.** Both ReLU and GELU require backpropagation during training — "
            "the choice of activation function does not change whether backprop runs. "
            "The cost difference is in the *forward pass computation*, not in the "
            "backward pass structure."
        ), kind="warn"),
        "wrong_arch": mo.callout(mo.md(
            "**Not quite.** GELU was introduced in 2016 and is used across CNNs, MLPs, "
            "and transformers. Its use in transformers (BERT, GPT) made it famous, but "
            "it is a general activation function. The hardware cost applies equally "
            "regardless of architecture."
        ), kind="warn"),
    }
    _feedback[act1_reflection.value]
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II -- DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "II"
    _act_color    = COLORS["OrangeLine"]
    _act_title    = "The Memory Hierarchy"
    _act_duration = "20&ndash;25 min"
    _act_why      = ("Act I quantified the silicon cost of activation functions. "
                     "Now discover that where activations live in memory is a second, "
                     "independent 10&times; multiplier per tier boundary. You expect a "
                     "3&times;3 convolution&rsquo;s activations to fit in mobile L2 cache. "
                     "The Memory Ledger will show that at batch=1, the 224&times;224 input "
                     "layer alone generates ~50 MB &mdash; 100&times; larger than mobile L2 &mdash; "
                     "and every access hits HBM at 10&times; the L1 cost.")

    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="margin: 8px 0 12px 0;">
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
        mo.md("""
        Activation *functions* are one cost dimension. But where activations *live in memory*
        is a second, independent multiplier on performance. The memory hierarchy creates a
        10× penalty at every tier boundary:

        | Tier | Latency | Capacity (Mobile) | Capacity (Cloud) |
        |------|---------|-------------------|------------------|
        | L1 SRAM | 1× (fastest) | 64 KB | 256 KB/SM |
        | L2 Cache | 4× | 512 KB | 40 MB |
        | HBM / Device RAM | 10× | 8 GB | 80 GB |
        | Host DRAM | 100× | (off-device) | (off-device) |

        This is the **Memory Ledger** instrument — used here for the first time.
        For each layer, the Ledger shows which tier its activations reside in, and
        the live latency multiplier that tier imposes on every memory access.

        Before you explore, make a prediction.
        """),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10: ACT II PREDICTION LOCK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    act2_prediction = mo.ui.radio(
        options={
            "A) 100% — L2 can always cache convolution activations": "100pct",
            "B) ~50% — about half the activation map fits": "50pct",
            "C) ~5% — only a small fraction fits in L2": "5pct",
            "D) <1% — almost nothing fits in L2 cache on mobile": "lt1pct",
        },
        label=(
            "A 3×3 convolution with 256 input channels on a 224×224 image: "
            "what fraction of its output activation map fits in mobile L2 cache (512 KB)?"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background:#1e293b; border-radius:8px; padding:14px 20px;
                    border-left:4px solid #6366f1; margin:8px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:#a5b4fc;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
                Prediction Lock — Act II
            </div>
            <div style="font-size:0.88rem; color:#94a3b8; line-height:1.5;">
                Activation map = 224×224×256 values. FP32 = 4 bytes per value.
                Mobile L2 = 512 KB. Commit before checking your arithmetic.
            </div>
        </div>
        """),
        act2_prediction,
    ])
    return (act2_prediction,)


@app.cell(hide_code=True)
def _(mo, act2_prediction):
    mo.stop(
        act2_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Memory Ledger."),
            kind="warn",
        ),
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11: ACT II REVEAL — PREDICTION vs ACTUAL
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act2_prediction):
    # 224×224×256 × 4 bytes = 51,380,224 bytes ≈ 51 MB
    # Mobile L2 = 512 KB = 524,288 bytes
    _activation_bytes = 224 * 224 * 256 * 4       # 51,380,224 bytes ≈ 51 MB
    _l2_bytes         = 512 * 1024                 # 524,288 bytes
    _fraction_pct     = 100.0 * _l2_bytes / _activation_bytes   # ~1.02%

    _predicted_label = {
        "100pct": "100%",
        "50pct":  "~50%",
        "5pct":   "~5%",
        "lt1pct": "<1%",
    }[act2_prediction.value]

    _correct = act2_prediction.value == "lt1pct"

    mo.callout(
        mo.md(
            f"You predicted **{_predicted_label}**. "
            f"The actual fraction is **{_fraction_pct:.1f}%**. "
            + ("**Correct.**" if _correct else f"**The gap is larger than expected.**")
            + f"""

            **The arithmetic:**
            ```
            Activation map = 224 × 224 × 256 channels × 4 bytes (FP32)
                           = {_activation_bytes:,} bytes ≈ {_activation_bytes/1e6:.0f} MB

            Mobile L2      = 512 KB = {_l2_bytes:,} bytes

            Fraction in L2 = {_l2_bytes:,} / {_activation_bytes:,} = {_fraction_pct:.1f}%
            ```

            A single convolution layer at 224×224 resolution with 256 channels generates
            **{_activation_bytes/1e6:.0f} MB** of activation data. Mobile L2 cache holds **512 KB**.
            That is 100× more data than cache can hold. Every memory access for this layer
            hits HBM (or DRAM if HBM is also saturated), paying the **10–100× latency penalty**.
            This is the Memory Wall made concrete.
            """
        ),
        kind="success" if _correct else "warn",
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12: ACT II INSTRUMENT — MEMORY LEDGER
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The Memory Ledger (first introduced here)

    The Memory Ledger maps each layer's activations to their memory tier and shows
    the resulting latency penalty. Use the controls below to explore how layer size,
    batch size, and precision interact with the memory hierarchy.

    **Color key:**
    - Green = fits in L1/L2 cache (fast path)
    - Orange = spills to HBM/device RAM (4–10× penalty)
    - Red = approaches or exceeds device RAM (OOM risk)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Memory Ledger controls
    act2_image_size = mo.ui.dropdown(
        options={"28×28 (MNIST)": 28, "64×64 (thumbnail)": 64, "128×128 (mobile)": 128,
                 "224×224 (ImageNet)": 224, "512×512 (high-res)": 512},
        value="224×224 (ImageNet)",
        label="Input image size",
    )
    act2_channels = mo.ui.slider(
        start=1, stop=512, value=64, step=1,
        label="Output channels (conv layer)",
    )
    act2_batch = mo.ui.slider(
        start=1, stop=128, value=8, step=1,
        label="Batch size",
    )
    act2_precision = mo.ui.radio(
        options={"FP32 (4 bytes)": 4, "FP16/BF16 (2 bytes)": 2, "INT8 (1 byte)": 1},
        value="FP32 (4 bytes)",
        label="Precision:",
        inline=True,
    )
    mo.vstack([
        mo.hstack([act2_image_size, act2_channels], justify="start", gap="2rem"),
        act2_batch,
        act2_precision,
    ])
    return (act2_image_size, act2_channels, act2_batch, act2_precision)


@app.cell(hide_code=True)
def _(
    mo, go, apply_plotly_theme, COLORS,
    act2_image_size, act2_channels, act2_batch, act2_precision,
    context_toggle,
    H100_RAM_GB, H100_L2_CACHE_MB, H100_L1_PER_SM_KB,
    MOBILE_RAM_GB, MOBILE_L2_CACHE_KB, MOBILE_L1_CACHE_KB,
    MEM_L1_MULT, MEM_L2_MULT, MEM_HBM_MULT, MEM_DRAM_MULT,
):
    # ── Device memory parameters ──────────────────────────────────────────────
    _ctx = context_toggle.value
    if _ctx == "cloud":
        _device_ram_mb  = H100_RAM_GB * 1024               # 81,920 MB
        _l2_cache_kb    = H100_L2_CACHE_MB * 1024          # 40,960 KB
        _l1_cache_kb    = H100_L1_PER_SM_KB                # 256 KB per SM
        _ctx_label      = "Cloud H100"
        _ctx_color      = COLORS["Cloud"]
    else:
        _device_ram_mb  = MOBILE_RAM_GB * 1024             # 8,192 MB
        _l2_cache_kb    = MOBILE_L2_CACHE_KB               # 512 KB
        _l1_cache_kb    = MOBILE_L1_CACHE_KB               # 64 KB
        _ctx_label      = "Mobile NPU"
        _ctx_color      = COLORS["Mobile"]

    # ── Layer activation sizes ────────────────────────────────────────────────
    # Model: 4 conv layers, each halving spatial dimensions, doubling channels
    _img         = act2_image_size.value
    _ch_out      = act2_channels.value
    _batch       = act2_batch.value
    _bytes_pp    = act2_precision.value

    _layers = []
    _w = _img
    _ch = _ch_out
    for _layer_idx in range(6):
        # Source: activation_memory = batch × height × width × channels × bytes
        _size_bytes = _batch * _w * _w * _ch * _bytes_pp
        _size_kb    = _size_bytes / 1024.0
        _size_mb    = _size_kb / 1024.0

        # Memory tier assignment
        if _size_kb <= _l1_cache_kb:
            _tier       = "L1 Cache"
            _mult       = MEM_L1_MULT
            _bar_color  = COLORS["GreenLine"]
            _tier_short = "L1"
        elif _size_kb <= _l2_cache_kb:
            _tier       = "L2 Cache"
            _mult       = MEM_L2_MULT
            _bar_color  = COLORS["GreenLine"]
            _tier_short = "L2"
        elif _size_mb <= _device_ram_mb:
            _tier       = "Device RAM (HBM)"
            _mult       = MEM_HBM_MULT
            _bar_color  = COLORS["OrangeLine"]
            _tier_short = "HBM"
        else:
            _tier       = "Exceeds Device RAM"
            _mult       = MEM_DRAM_MULT
            _bar_color  = COLORS["RedLine"]
            _tier_short = "OOM"

        _layers.append({
            "name":       f"Layer {_layer_idx+1}",
            "size_kb":    _size_kb,
            "size_mb":    _size_mb,
            "spatial":    f"{_w}×{_w}",
            "channels":   _ch,
            "tier":       _tier,
            "tier_short": _tier_short,
            "mult":       _mult,
            "color":      _bar_color,
        })
        _w = max(_w // 2, 1)
        _ch = _ch * 2

    # ── Build Memory Ledger visualization ─────────────────────────────────────
    _layer_names      = [l["name"] for l in _layers]
    _sizes_mb         = [l["size_mb"] for l in _layers]
    _colors           = [l["color"] for l in _layers]
    _tier_labels      = [l["tier_short"] for l in _layers]
    _mult_labels      = [f"{l['mult']}×" for l in _layers]
    _hover_texts      = [
        f"<b>{l['name']}</b><br>"
        f"Spatial: {l['spatial']} | Channels: {l['channels']}<br>"
        f"Activation size: {l['size_kb']:.1f} KB<br>"
        f"Memory tier: {l['tier']}<br>"
        f"Latency multiplier: {l['mult']}×<br>"
        f"Batch: {_batch} | Precision: {_bytes_pp*8}-bit"
        for l in _layers
    ]

    _fig_ledger = go.Figure()
    _fig_ledger.add_trace(go.Bar(
        x=_layer_names,
        y=_sizes_mb,
        marker_color=_colors,
        text=[f"{t}<br>{m}" for t, m in zip(_tier_labels, _mult_labels)],
        textposition="outside",
        hovertext=_hover_texts,
        hoverinfo="text",
        name="Activation size",
    ))

    # Reference lines for cache tiers
    _l2_mb   = _l2_cache_kb / 1024.0
    _l1_mb   = _l1_cache_kb / 1024.0
    _ram_mb  = _device_ram_mb

    _fig_ledger.add_hline(
        y=_l1_mb, line_dash="dot", line_color=COLORS["GreenLine"], line_width=1.5,
        annotation_text=f"L1 ({_l1_cache_kb} KB)",
        annotation_position="right",
    )
    _fig_ledger.add_hline(
        y=_l2_mb, line_dash="dash", line_color=COLORS["OrangeLine"], line_width=2,
        annotation_text=f"L2 ({_l2_cache_kb:,} KB)",
        annotation_position="right",
    )
    _fig_ledger.add_hline(
        y=_ram_mb, line_dash="solid", line_color=COLORS["RedLine"], line_width=2,
        annotation_text=f"Device RAM ({_device_ram_mb/1024:.0f} GB)",
        annotation_position="right",
    )

    _fig_ledger.update_layout(
        title_text=f"Memory Ledger — {_ctx_label} — Batch {_batch} — {_bytes_pp*8}-bit",
        xaxis_title="Network Layer",
        yaxis_title="Activation Memory (MB, log scale)",
        yaxis_type="log",
        showlegend=False,
        height=380,
    )
    apply_plotly_theme(_fig_ledger)

    # ── Latency multiplier chart ───────────────────────────────────────────────
    _mults     = [l["mult"] for l in _layers]
    _mult_colors = [l["color"] for l in _layers]
    _fig_mult  = go.Figure()
    _fig_mult.add_trace(go.Bar(
        x=_layer_names, y=_mults,
        marker_color=_mult_colors,
        text=[f"{m}×" for m in _mults],
        textposition="outside",
    ))
    _fig_mult.update_layout(
        title_text="Memory Access Latency Multiplier by Layer",
        xaxis_title="Layer",
        yaxis_title="Latency multiplier (×L1 baseline)",
        showlegend=False,
        height=280,
    )
    apply_plotly_theme(_fig_mult)

    # ── Summary metrics ────────────────────────────────────────────────────────
    _total_act_mb    = sum(_sizes_mb)
    _avg_mult        = sum(_mults) / len(_mults)
    _cache_miss_pct  = 100.0 * sum(1 for l in _layers if l["mult"] > MEM_L2_MULT) / len(_layers)
    _oom             = _total_act_mb > _device_ram_mb

    _total_color = COLORS["RedLine"] if _oom else (
        COLORS["OrangeLine"] if _total_act_mb > _device_ram_mb * 0.5 else COLORS["GreenLine"]
    )
    _mult_color  = COLORS["RedLine"] if _avg_mult >= MEM_HBM_MULT else (
        COLORS["OrangeLine"] if _avg_mult > MEM_L2_MULT else COLORS["GreenLine"]
    )

    _metric_cards = mo.Html(f"""
    <div style="display:flex; gap:16px; justify-content:center; margin-top:12px; flex-wrap:wrap;">
        <div style="padding:14px 18px; border:1px solid #e2e8f0; border-radius:10px;
                    width:190px; text-align:center; background:white; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
            <div style="color:#475569; font-size:0.8rem; font-weight:600; margin-bottom:4px;">
                Total Activation Memory
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{_total_color};">
                {_total_act_mb:.0f} MB
            </div>
            <div style="color:#94a3b8; font-size:0.72rem;">
                Device RAM: {_device_ram_mb/1024:.0f} GB
            </div>
        </div>
        <div style="padding:14px 18px; border:1px solid #e2e8f0; border-radius:10px;
                    width:190px; text-align:center; background:white; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
            <div style="color:#475569; font-size:0.8rem; font-weight:600; margin-bottom:4px;">
                Avg Latency Multiplier
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{_mult_color};">
                {_avg_mult:.1f}×
            </div>
            <div style="color:#94a3b8; font-size:0.72rem;">vs L1 baseline</div>
        </div>
        <div style="padding:14px 18px; border:1px solid #e2e8f0; border-radius:10px;
                    width:190px; text-align:center; background:white; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
            <div style="color:#475569; font-size:0.8rem; font-weight:600; margin-bottom:4px;">
                Cache Miss Rate (HBM+)
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{
                COLORS['RedLine'] if _cache_miss_pct > 60 else
                COLORS['OrangeLine'] if _cache_miss_pct > 30 else
                COLORS['GreenLine']
            };">
                {_cache_miss_pct:.0f}%
            </div>
            <div style="color:#94a3b8; font-size:0.72rem;">of layers in slow memory</div>
        </div>
    </div>
    """)

    mo.vstack([
        _fig_ledger,
        _fig_mult,
        _metric_cards,
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13: ACT II FAILURE STATE — OOM DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(
    mo,
    act2_image_size, act2_channels, act2_batch, act2_precision,
    context_toggle,
    H100_RAM_GB, MOBILE_RAM_GB,
    MEM_L1_MULT, MEM_L2_MULT, MEM_HBM_MULT, MEM_DRAM_MULT,
    H100_L2_CACHE_MB, H100_L1_PER_SM_KB,
    MOBILE_L2_CACHE_KB, MOBILE_L1_CACHE_KB,
):
    # Compute total activation memory across 6 layers
    _ctx = context_toggle.value
    _device_ram_mb = (H100_RAM_GB if _ctx == "cloud" else MOBILE_RAM_GB) * 1024
    _device_label  = "H100 (80 GB HBM)" if _ctx == "cloud" else "Mobile NPU (8 GB)"

    _img      = act2_image_size.value
    _ch_out   = act2_channels.value
    _batch    = act2_batch.value
    _bytes_pp = act2_precision.value

    _total_act_bytes = 0
    _w = _img
    _ch = _ch_out
    for _ in range(6):
        _total_act_bytes += _batch * _w * _w * _ch * _bytes_pp
        _w = max(_w // 2, 1)
        _ch = _ch * 2
    _total_act_mb = _total_act_bytes / (1024 * 1024)

    # Note: real training also requires weights + gradients + optimizer state
    # (training_memory ≈ weights + gradients + 2×weights_Adam + activations)
    # Here we focus on activation memory alone as the variable component.
    _oom = _total_act_mb > _device_ram_mb

    if _oom:
        mo.callout(
            mo.md(
                f"**OOM — Activation memory exceeds device RAM.** "
                f"Required: **{_total_act_mb:.0f} MB** | "
                f"Available: **{_device_label}** ({_device_ram_mb/1024:.0f} GB = {_device_ram_mb:,} MB). "
                f"Reduce batch size, cut channels, or enable activation checkpointing "
                f"(@sec-neural-computation to `@sec-model-training`)."
            ),
            kind="danger",
        )
    elif _total_act_mb > _device_ram_mb * 0.75:
        mo.callout(
            mo.md(
                f"**Memory warning.** Activation memory is {_total_act_mb:.0f} MB — "
                f"{100*_total_act_mb/_device_ram_mb:.0f}% of device RAM. "
                f"Training (which requires weights + gradients + optimizer state on top) "
                f"will almost certainly OOM. This is inference-only viable."
            ),
            kind="warn",
        )
    else:
        mo.callout(
            mo.md(
                f"**Memory is feasible.** Activation memory: {_total_act_mb:.1f} MB "
                f"({100*_total_act_mb/_device_ram_mb:.1f}% of device RAM). "
                f"Note: training requires additional memory for weights, gradients, "
                f"and optimizer state — approximately 4× the model size for Adam."
            ),
            kind="success",
        )
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14: ACT II MATH PEEK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — Activation Memory": mo.md("""
        **Activation memory per layer** (@sec-neural-computation-computational-implementation-details-1ecc):

        ```
        activation_memory = batch × height × width × channels × bytes_per_value
        ```

        For FP32 (4 bytes), batch=1, 224×224, 256 channels:

        ```
        activation_memory = 1 × 224 × 224 × 256 × 4
                          = 51,380,224 bytes
                          ≈ 51 MB
        ```

        **Memory tier thresholds (Mobile NPU):**

        ```
        L1 SRAM:    64 KB   — latency: 1×  (fastest)
        L2 Cache:  512 KB   — latency: 4×
        Device HBM:  8 GB   — latency: 10×
        Host DRAM: (off)    — latency: 100×
        ```

        **Latency multiplier on memory-bound operations:**

        The Memory Wall footnote (fn-memory-wall-nn) states: L1 cache delivers data
        in ~1 ns; main memory takes ~100 ns — a 100× gap.

        **Training memory total** (@eq-training-memory):

        ```
        training_memory ≈ weights + gradients + optimizer_state + activations
        ```

        For Adam optimizer (weights W):

        ```
        weights         = W × 4 bytes (FP32)
        gradients       = W × 4 bytes
        Adam momentum   = W × 4 bytes
        Adam velocity   = W × 4 bytes
        optimizer_total = 4W bytes
        activations     = batch × sum(layer_sizes) × 4 bytes
        ```

        Total training memory ≈ 4× weights + activation memory.
        This is why "will this fit?" is not a model question — it is a batch size question.
        """)
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# CELL 15: ACT II STRUCTURED REFLECTION
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) Larger batches need more parameters to process": "wrong_params",
            "B) Larger batches increase activation memory proportionally, pushing data into slower tiers": "correct",
            "C) Batch size has no effect on memory-boundedness": "wrong_none",
            "D) Larger batches always improve throughput, so memory tier does not matter": "wrong_throughput",
        },
        label="Why does batch size affect memory-boundedness?",
    )
    mo.vstack([
        mo.md("**Reflection:** Answer before reading the explanation."),
        act2_reflection,
    ])
    return (act2_reflection,)


@app.cell(hide_code=True)
def _(mo, act2_reflection):
    mo.stop(act2_reflection.value is None, mo.md(""))

    _feedback = {
        "wrong_params": mo.callout(mo.md(
            "**Not quite.** Model parameters (weights) do not change with batch size — "
            "the same weights are reused for every sample in the batch. What grows "
            "linearly with batch size is *activation memory*: each sample in the batch "
            "produces its own activation tensor that must be stored for backpropagation."
        ), kind="warn"),
        "correct": mo.callout(mo.md(
            "**Correct.** Activation memory scales as: "
            "`batch × height × width × channels × bytes`. "
            "Doubling batch size doubles activation memory. When activations at batch=1 "
            "fit in L2 cache (4× latency), activations at batch=64 may spill to HBM "
            "(10× latency) — a 2.5× additional slowdown with no change in the model "
            "at all. The boundary that separates 'this runs fast' from 'this is "
            "memory-bound' is often a single doubling of batch size. This is why profiling "
            "tools report memory-boundedness as a function of batch, not just architecture."
        ), kind="success"),
        "wrong_none": mo.callout(mo.md(
            "**Not quite.** Batch size directly controls activation memory: "
            "`activation_memory = batch × spatial × channels × bytes`. "
            "A batch of 64 uses 64× the activation memory of batch=1. When that memory "
            "crosses a cache tier boundary, every activation access pays the tier penalty."
        ), kind="warn"),
        "wrong_throughput": mo.callout(mo.md(
            "**Not quite.** Larger batches *can* improve throughput when computation "
            "is the bottleneck — because they amortize kernel launch overhead. But when "
            "activation memory spills out of fast cache into HBM or DRAM, the memory "
            "bandwidth becomes the bottleneck and throughput stops improving despite "
            "more parallelism. The trade-off between batch size, compute efficiency, "
            "and memory-tier pressure is the core tension in training configuration."
        ), kind="warn"),
    }
    _feedback[act2_reflection.value]
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, ACTIVATION_TAX_RATIO):
    mo.vstack([
        mo.md("---"),

        # ── KEY TAKEAWAYS ──
        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. The Transistor Tax is real and large.</strong>
                    ReLU costs ~50 transistors; Sigmoid costs ~2,500 &mdash;
                    a <strong>{int(ACTIVATION_TAX_RATIO)}&times; silicon penalty</strong> per activation unit
                    (@sec-neural-computation-transistor-tax). On mobile hardware with a fixed power
                    and area budget, this translates directly to FPS targets missed and battery drain.
                    Activation function choice is not a mathematical preference; it is a hardware constraint.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Where data lives matters as much as what you compute with it.</strong>
                    The memory hierarchy imposes a 10&times; latency penalty per tier boundary.
                    A single convolution layer at 224&times;224 with 256 channels generates ~51 MB of
                    activations &mdash; 100&times; what mobile L2 cache can hold. Every memory access
                    for that layer hits HBM at 10&times; the L1 cost.
                </div>
                <div>
                    <strong>3. Batch size is the lever that determines memory tier placement.</strong>
                    At batch=1, small layers may fit in L2. At batch=32, the same layer spills to HBM.
                    The OOM boundary is predictable from first principles: total activation memory
                    = &Sigma;<sub>l</sub> batch &times; width<sub>l</sub>&sup2; &times; channels<sub>l</sub> &times; bytes/element.
                </div>
            </div>
        </div>
        """),

        # ── CONNECTIONS ──
        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">

            <!-- What's Next -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab 06: The Quadratic Wall</strong> &mdash; this lab showed that
                    activation memory is a linear cost in layer count and batch size. Lab 06
                    asks: what happens when the memory cost is O(N&sup2;)? Transformer
                    self-attention materializes an N&times;N similarity matrix &mdash; and doubling
                    the sequence length quadruples the memory requirement.
                </div>
            </div>

            <!-- Textbook Connection -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-neural-computation-transistor-tax (Transistor Tax),
                    @sec-neural-computation-computational-implementation-details-1ecc (training memory),
                    footnote fn-memory-wall-nn (L1 vs DRAM latency gap).<br/>
                    <strong>Build:</strong> TinyTorch Module 05 &mdash; implement forward passes for
                    each activation function by hand and measure their computational cost directly.
                    See <code>tinytorch/src/05_activations/</code>.
                </div>
            </div>

        </div>
        """),


        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. Sigmoid costs approximately 50x more transistors than ReLU and causes gradient magnitudes to collapse to ~0.25^L after L layers. At what depth does the gradient fall below the 'learning becomes impossible' threshold of 10^-6?

    2. Training the MNIST network (784->128->64->10) requires approximately how many times more memory than inference at batch=32 — and which of the four memory components (weights, gradients, optimizer state, activations) grows with batch size?

    3. A team switches from ReLU to Sigmoid to improve convergence on a 12-layer network. Predict what happens to (a) gradient magnitude after 12 layers, (b) training memory requirements, and (c) inference power per evaluation — and explain which failure is silent versus visible.

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle,
    act1_prediction, act1_layer1, act1_layer2, act1_layer3, act1_layer4,
    act1_reflection, act2_prediction, act2_reflection,
    act2_image_size, act2_channels, act2_batch, act2_precision,
    H100_RAM_GB, MOBILE_RAM_GB, ACTIVATION_TAX_RATIO,
    H100_L2_CACHE_MB, H100_L1_PER_SM_KB,
    MOBILE_L2_CACHE_KB, MOBILE_L1_CACHE_KB,
    MEM_L1_MULT, MEM_L2_MULT, MEM_HBM_MULT, MEM_DRAM_MULT,
):
    _ctx = context_toggle.value
    _device_ram_mb = (H100_RAM_GB if _ctx == "cloud" else MOBILE_RAM_GB) * 1024
    _l2_cache_kb   = H100_L2_CACHE_MB * 1024 if _ctx == "cloud" else MOBILE_L2_CACHE_KB
    _l1_cache_kb   = H100_L1_PER_SM_KB if _ctx == "cloud" else MOBILE_L1_CACHE_KB

    # Compute activation memory and OOM state
    _img      = act2_image_size.value
    _ch_out   = act2_channels.value
    _batch    = act2_batch.value
    _bytes_pp = act2_precision.value
    _total_act_bytes = 0
    _w, _ch = _img, _ch_out
    for _ in range(6):
        _total_act_bytes += _batch * _w * _w * _ch * _bytes_pp
        _w = max(_w // 2, 1)
        _ch = _ch * 2
    _total_act_mb = _total_act_bytes / (1024 * 1024)
    _oom_triggered = _total_act_mb > _device_ram_mb

    # Cache miss rate
    _cache_miss_layers = 0
    _w, _ch = _img, _ch_out
    for _ in range(6):
        _size_kb = _batch * _w * _w * _ch * _bytes_pp / 1024
        if _size_kb > _l2_cache_kb:
            _cache_miss_layers += 1
        _w = max(_w // 2, 1)
        _ch = _ch * 2
    _cache_miss_rate = _cache_miss_layers / 6.0

    # Layer activation choices — map back to canonical key
    _act_map = {
        "ReLU   (max(0,x) — comparator)": "relu",
        "Leaky ReLU   (max(αx,x))": "leaky_relu",
        "Sigmoid   (1/(1+e^−x))": "sigmoid",
        "Tanh   ((e^x−e^−x)/(e^x+e^−x))": "tanh",
        "GELU   (x·Φ(x) — erf approx)": "gelu",
        "Softmax   (exp / sum(exp))": "softmax",
    }
    _activation_choice = {
        "layer1": _act_map.get(act1_layer1.value, "sigmoid"),
        "layer2": _act_map.get(act1_layer2.value, "sigmoid"),
        "layer3": _act_map.get(act1_layer3.value, "sigmoid"),
        "layer4": _act_map.get(act1_layer4.value, "sigmoid"),
    }

    # Correctness flags
    _act1_correct = (act1_prediction.value == "50x") if act1_prediction.value else False
    _act2_correct = (act2_prediction.value == "lt1pct") if act2_prediction.value else False
    _refl1_correct = (act1_reflection.value == "correct") if act1_reflection.value else False
    _refl2_correct = (act2_reflection.value == "correct") if act2_reflection.value else False

    # Save to Design Ledger (chapter 5)
    ledger.save(chapter=5, design={
        "context":            _ctx,
        "activation_choice":  _activation_choice,
        "act1_prediction":    act1_prediction.value or "",
        "act1_correct":       _act1_correct,
        "act2_result":        round(_total_act_mb, 2),
        "act2_decision":      f"batch={_batch}_channels={_ch_out}_{_bytes_pp*8}bit",
        "constraint_hit":     _oom_triggered,
        "oom_triggered":      _oom_triggered,
        "cache_miss_rate":    round(_cache_miss_rate, 2),
    })

    # HUD footer
    _hud_items = [
        ("Chapter", "5 — Neural Computation"),
        ("Context", _ctx.upper()),
        ("Act I correct", "Yes" if _act1_correct else "No"),
        ("Act II correct", "Yes" if _act2_correct else "No"),
        ("OOM triggered", "Yes" if _oom_triggered else "No"),
        ("Cache miss rate", f"{_cache_miss_rate*100:.0f}%"),
    ]
    _hud_html = "".join(
        f'<div style="display:flex; flex-direction:column; gap:2px;">'
        f'<span class="hud-label">{k}</span>'
        f'<span class="{"hud-active" if v in ("Yes","cloud","mobile") else "hud-value"}">{v}</span>'
        f'</div>'
        for k, v in _hud_items
    )

    mo.Html(f'<div class="lab-hud">{_hud_html}</div>')
    return


if __name__ == "__main__":
    app.run()
