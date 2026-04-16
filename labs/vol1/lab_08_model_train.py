import marimo

__generated_with = "0.23.1"
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
    from mlsysim.core.engine import Engine

    H100 = mlsysim.Hardware.Cloud.H100
    V100 = mlsysim.Hardware.Cloud.V100
    A100 = mlsysim.Hardware.Cloud.A100
    JETSON = mlsysim.Hardware.Edge.JetsonOrinNX

    H100_RAM_GB = H100.memory.capacity.m_as("GB")
    H100_BW_GBS = H100.memory.bandwidth.m_as("GB/s")
    V100_RAM_GB = V100.memory.capacity.m_as("GB")
    A100_RAM_GB = A100.memory.capacity.m_as("GB")

    # Optimizer bytes per parameter
    OPTIMIZER_BPP = {
        "SGD":          8,   # weights (4B) + gradients (4B)
        "SGD+Momentum": 12,  # + momentum (4B)
        "Adam":         16,  # + momentum (4B) + variance (4B)
        "Adafactor":    10,  # approximate factored state
    }

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        COLORS, Engine, H100, V100, A100, JETSON,
        H100_RAM_GB, H100_BW_GBS, V100_RAM_GB, A100_RAM_GB,
        OPTIMIZER_BPP,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 08
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Training Gauntlet
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Memory Budget &middot; Pipeline Bubbles &middot; Mixed Precision &middot; Communication Tax
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Training is not "forward pass but bigger" -- it is a four-stage pipeline
                where memory budgets, pipeline bottlenecks, precision traps, and
                communication overhead each create surprising walls, and optimizing the
                wrong stage wastes resources while the true bottleneck goes untouched.
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
                    Chapter 8: Model Training
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Adam: 16 bytes/param</span>
                <span class="badge badge-warn">7B model: 112 GB static</span>
                <span class="badge badge-fail">Mixed precision: 1.5x, not 2x</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Calculate the training memory budget</strong>:
                    Adam on a 7B model requires 112 GB of parameter state alone (16 bytes/param),
                    exceeding an H100 before storing any activations.</div>
                <div style="margin-bottom: 3px;">2. <strong>Identify the pipeline bottleneck</strong>:
                    diagnose whether data loading, PCIe transfer, compute, or gradient sync
                    limits training throughput.</div>
                <div style="margin-bottom: 3px;">3. <strong>Quantify the mixed-precision trap</strong>:
                    FP32 master weights + Adam state persist at full precision, yielding only
                    1.5-1.7x memory savings, not the expected 2x.</div>
                <div style="margin-bottom: 3px;">4. <strong>Model the communication tax</strong>:
                    predict multi-GPU scaling efficiency using Speedup = N / (1 + (N-1)*r).</div>
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
                    Forward vs. backward memory from the Neural Computation chapter &middot;
                    Iron Law from the Iron Law section (Ch. 1) &middot;
                    Dispatch overhead from the ML Frameworks chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~52 min</strong><br/>
                    Part A: ~10 min &middot; Part B: ~12 min<br/>
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
                &ldquo;Your 7B model runs inference perfectly on an H100. You switch to
                training and immediately hit OOM -- before processing a single batch.
                Where did 112 GB of memory come from, and why does &lsquo;half precision&rsquo;
                only save 35%?&rdquo;
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

    - **Chapter 8: Training Memory** -- the 16-byte-per-parameter breakdown
      for Adam (weights + gradients + momentum + variance).
    - **Chapter 8: The Training Pipeline** -- four-stage pipeline (data loading,
      transfer, compute, gradient sync) and pipeline bubble analysis.
    - **Chapter 8: Mixed Precision** -- FP16/BF16 forward + FP32 master weights,
      and why savings are 1.5-1.7x not 2x.
    - **Chapter 8: Communication Tax** -- AllReduce overhead and the scaling
      formula Speedup = N / (1 + (N-1)*r).
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE B-D: ALL PARTS AS TABS
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 4: TABS CELL ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, Engine, H100, V100, A100, JETSON,
    H100_RAM_GB, H100_BW_GBS, V100_RAM_GB, A100_RAM_GB,
    OPTIMIZER_BPP, apply_plotly_theme, go, math, mo, np, ledger, mlsysim,
):
    # ─────────────────────────────────────────────────────────────────────
    # SHARED WIDGET STATE
    # ─────────────────────────────────────────────────────────────────────

    # Part A widgets
    partA_prediction = mo.ui.radio(
        options={
            "A) 28 GB (7B * 4 bytes)": "28gb",
            "B) 56 GB (weights + gradients)": "56gb",
            "C) 84 GB (+ one momentum buffer)": "84gb",
            "D) 112 GB (+ two momentum buffers)": "112gb",
        },
        label="A 7B-parameter model trained with Adam in FP32. Minimum memory for "
              "parameter state (weights + gradients + optimizer), before any activations?",
    )
    return (partA_prediction,)

@app.cell(hide_code=True)
def _(mo):
    partA_model_size = mo.ui.slider(
        start=0.1, stop=70, value=7, step=0.1, label="Model size (billions of params)",
    )
    partA_optimizer = mo.ui.dropdown(
        options={"SGD (8 B/param)": "SGD", "SGD+Momentum (12 B/param)": "SGD+Momentum",
                 "Adam (16 B/param)": "Adam", "Adafactor (10 B/param)": "Adafactor"},
        value="Adam (16 B/param)", label="Optimizer",
    )
    partA_precision_a = mo.ui.radio(
        options={"FP32": "fp32", "Mixed BF16 (BF16 compute, FP32 optimizer)": "bf16"},
        value="FP32", label="Precision:", inline=True,
    )

    # Part B widgets
    partB_prediction = mo.ui.radio(
        options={
            "A) Data loading (disk I/O)": "data",
            "B) Host-to-device transfer (PCIe)": "pcie",
            "C) Forward + backward pass (compute)": "compute",
            "D) Gradient synchronization (inter-GPU)": "sync",
        },
        label="GPT-2 training on V100 with SSD storage and 4 GPUs over PCIe. "
              "Which stage is the bottleneck?",
    )
    return (partB_prediction,)

@app.cell(hide_code=True)
def _(mo):
    partB_data_ms = mo.ui.slider(
        start=1, stop=100, value=50, step=1, label="Data loading (ms)",
    )
    partB_pcie_ms = mo.ui.slider(
        start=0.1, stop=10, value=2, step=0.1, label="PCIe transfer (ms)",
    )
    partB_compute_ms = mo.ui.slider(
        start=5, stop=200, value=30, step=1, label="Forward+Backward (ms)",
    )
    partB_sync_ms = mo.ui.slider(
        start=0, stop=50, value=15, step=1, label="Gradient sync (ms)",
    )

    # Part C widgets
    partC_prediction = mo.ui.radio(
        options={
            "A) ~38 GB (exactly half)": "38gb",
            "B) ~45 GB (~1.7x savings)": "45gb",
            "C) ~55 GB (~1.4x savings)": "55gb",
            "D) ~70 GB (almost no savings)": "70gb",
        },
        label="GPT-2 (1.5B) requires ~77 GB in full FP32 (with activations). "
              "Mixed precision (FP16 forward + FP32 master + FP32 Adam) requires how much?",
    )
    return (partC_prediction,)

@app.cell(hide_code=True)
def _(mo):
    partC_model_c = mo.ui.slider(
        start=0.1, stop=13, value=1.5, step=0.1, label="Model size (billions)",
    )
    partC_precision_c = mo.ui.radio(
        options={"Full FP32": "fp32", "Mixed BF16+FP32 master": "mixed", "Pure BF16": "bf16"},
        value="Full FP32", label="Precision mode:", inline=True,
    )

    # Part D widgets
    partD_prediction = mo.ui.radio(
        options={
            "A) ~8x (linear scaling)": "8x",
            "B) ~6.5x (slight overhead)": "6.5x",
            "C) ~3.9x (significant overhead)": "3.9x",
            "D) ~2x (communication dominates)": "2x",
        },
        label="Training on 8 GPUs with r=0.15 (gradient sync = 15% of step time). "
              "What speedup over 1 GPU?",
    )
    return (partD_prediction,)

@app.cell(hide_code=True)
def _(mo, partD_prediction):
    mo.stop(partD_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partD_gpus = mo.ui.slider(
        start=1, stop=256, value=8, step=1, label="Number of GPUs",
    )
    partD_r = mo.ui.slider(
        start=0.01, stop=0.50, value=0.15, step=0.01, label="Communication fraction (r)",
    )

    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER: The Memory Budget Shock
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Training Engineer, FoundationAI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We just got an H100 with 80 GB of memory. Our 7B model runs inference
                perfectly. But the moment we switch to training with Adam, we get OOM --
                before processing a single batch. The model is only 28 GB in FP32. Where
                did the other 84 GB come from?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Jordan Kim, Training Engineer &middot; FoundationAI
            </div>
        </div>
        """))

        items.append(mo.md("""
## Training Memory = Weights + Gradients + Optimizer State + Activations

For Adam in FP32, each parameter requires **16 bytes** of static state:

| Component | Size per Parameter | Purpose |
|-----------|-------------------|---------|
| Weights | 4 bytes (FP32) | Current parameter values |
| Gradients | 4 bytes (FP32) | Computed during backward pass |
| Momentum (m) | 4 bytes (FP32) | First moment estimate |
| Variance (v) | 4 bytes (FP32) | Second moment estimate |
| **Total** | **16 bytes** | **Before any activations** |

For a 7B model: 7B * 16 = **112 GB** -- exceeding an H100's 80 GB.
        """))

        items.append(partA_prediction)

        if partA_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the memory budget calculator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partA_model_size, partA_optimizer, partA_precision_a], justify="start"))

        _params_b = partA_model_size.value
        _opt = partA_optimizer.value
        _prec = partA_precision_a.value
        _bpp = OPTIMIZER_BPP[_opt]

        if _prec == "bf16":
            # BF16 training: weights in BF16, gradients in BF16, but Adam states still FP32
            _weight_bytes = _params_b * 1e9 * 2  # BF16
            _grad_bytes = _params_b * 1e9 * 2   # BF16
            if _opt == "Adam":
                _opt_bytes = _params_b * 1e9 * 8  # momentum + variance in FP32
            elif _opt == "SGD+Momentum":
                _opt_bytes = _params_b * 1e9 * 4
            else:
                _opt_bytes = 0
            _total_bytes = _weight_bytes + _grad_bytes + _opt_bytes
        else:
            _total_bytes = _params_b * 1e9 * _bpp

        _total_gb = _total_bytes / (1024**3)

        # Component breakdown
        if _prec == "fp32":
            _w_gb = _params_b * 1e9 * 4 / (1024**3)
            _g_gb = _params_b * 1e9 * 4 / (1024**3)
            _o_gb = _total_gb - _w_gb - _g_gb
        else:
            _w_gb = _weight_bytes / (1024**3)
            _g_gb = _grad_bytes / (1024**3)
            _o_gb = _opt_bytes / (1024**3)

        _fig = go.Figure()
        _components = ["Weights", "Gradients", "Optimizer State"]
        _vals = [_w_gb, _g_gb, _o_gb]
        _cols = [COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"]]
        for _name, _val, _col in zip(_components, _vals, _cols):
            _fig.add_trace(go.Bar(name=_name, x=["Training Memory"], y=[_val],
                                  marker_color=_col, opacity=0.88))
        _fig.add_hline(y=H100_RAM_GB, line_dash="dash", line_color=COLORS["BlueLine"],
                       annotation_text=f"H100 ({H100_RAM_GB:.0f} GB)")
        _fig.add_hline(y=A100_RAM_GB, line_dash="dash", line_color=COLORS["OrangeLine"],
                       annotation_text=f"A100 ({A100_RAM_GB:.0f} GB)")
        _fig.update_layout(barmode="stack", height=380, yaxis_title="Memory (GB)",
                           title=f"Training Memory: {_params_b:.1f}B params, {_opt}, {_prec.upper()}",
                           legend=dict(orientation="h", y=1.12, x=0))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _oom_h100 = _total_gb > H100_RAM_GB
        _color = COLORS["RedLine"] if _oom_h100 else COLORS["GreenLine"]

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Total Static Memory</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_color};">{_total_gb:,.1f} GB</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Bytes per Parameter</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_bpp} B</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Optimizer Overhead</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_o_gb:,.1f} GB</div>
            </div>
        </div>
        """))

        if _oom_h100:
            items.append(mo.callout(mo.md(
                f"**OOM on H100 -- static state alone exceeds device memory.** "
                f"Training requires {_total_gb:,.1f} GB for parameter state, but the H100 "
                f"has only {H100_RAM_GB:.0f} GB. This is before storing a single activation. "
                f"Model parallelism, ZeRO, or a larger device is required."
            ), kind="danger"))

        _pred = partA_prediction.value
        if _pred == "112gb":
            items.append(mo.callout(mo.md(
                "**Correct.** 7B * 16 bytes/param = 112 GB. Adam's two additional state "
                "tensors (momentum and variance) each add 4 bytes/param, tripling the "
                "memory beyond what naive 'weights + gradients' would suggest."
            ), kind="success"))
        elif _pred == "56gb":
            items.append(mo.callout(mo.md(
                "**You forgot the optimizer state.** Weights (28 GB) + Gradients (28 GB) = 56 GB. "
                "But Adam adds momentum (28 GB) + variance (28 GB) = 112 GB total."
            ), kind="warn"))
        elif _pred == "28gb":
            items.append(mo.callout(mo.md(
                "**That is only the weights.** Training requires gradients (same size) plus "
                "optimizer state. Adam adds two more buffers: 7B * 16 = 112 GB total."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                "**Close -- but Adam has two state tensors, not one.** Both momentum and "
                "variance require 4 bytes/param each. Total: 7B * 16 = 112 GB."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Training Memory Budget": mo.md(f"""
```
Adam FP32: {_bpp} bytes/param
  = 4 (weights) + 4 (gradients) + 4 (momentum) + 4 (variance)

For {_params_b:.1f}B parameters:
  = {_params_b:.1f}B * {_bpp} = {_total_gb:,.1f} GB
```
Note: This excludes activations (Lab 05) which add batch*depth*width per layer.

Source: @sec-training-memory-budget
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER: The Training Pipeline
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Training Lead, FoundationAI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our GPU utilization during training is only 40%. We assumed the GPU
                would be busy 90%+ of the time since training is compute-heavy. Where is the
                other 60% going?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Sarah Zhang, Training Lead &middot; FoundationAI
            </div>
        </div>
        """))

        items.append(mo.md("""
## Training Is a Four-Stage Pipeline

```
Data Loading -> PCIe Transfer -> Forward+Backward -> Gradient Sync
    (disk)       (host->GPU)      (GPU compute)     (GPU<->GPU)
```

Total throughput is limited by the **slowest stage**. The GPU sits idle
("accelerator bubble") whenever it waits for data or communication.
Most training runs are **not** compute-bound.
        """))

        items.append(partB_prediction)

        if partB_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the pipeline simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partB_data_ms, partB_pcie_ms, partB_compute_ms, partB_sync_ms], justify="start"))

        _data = partB_data_ms.value
        _pcie = partB_pcie_ms.value
        _compute = partB_compute_ms.value
        _sync = partB_sync_ms.value
        _stages = {"Data Loading": _data, "PCIe Transfer": _pcie,
                   "Forward+Backward": _compute, "Gradient Sync": _sync}
        _total_sequential = sum(_stages.values())
        _bottleneck_stage = max(_stages, key=_stages.get)
        _bottleneck_time = _stages[_bottleneck_stage]

        # With prefetching, total = max(stages) + small overlap overhead
        _total_overlapped = max(_stages.values()) * 1.1  # 10% overlap overhead
        _gpu_util = (_compute / _total_sequential) * 100
        _bubble_pct = 100 - _gpu_util

        _stage_colors = {
            "Data Loading": COLORS["OrangeLine"],
            "PCIe Transfer": COLORS["BlueLine"],
            "Forward+Backward": COLORS["GreenLine"],
            "Gradient Sync": COLORS["RedLine"],
        }

        # Gantt-style waterfall
        _fig = go.Figure()
        _x_pos = 0
        for _stage, _time in _stages.items():
            _is_bottleneck = _stage == _bottleneck_stage
            _fig.add_trace(go.Bar(
                name=_stage, x=[_time], y=["Sequential"],
                orientation="h", marker_color=_stage_colors[_stage],
                opacity=1.0 if _is_bottleneck else 0.6,
                base=_x_pos,
            ))
            _x_pos += _time

        _fig.update_layout(
            barmode="stack", height=240,
            xaxis_title="Time (ms)", showlegend=True,
            title=f"Training Step Pipeline (bottleneck: {_bottleneck_stage})",
            legend=dict(orientation="h", y=1.2, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _util_color = COLORS["GreenLine"] if _gpu_util > 70 else COLORS["OrangeLine"] if _gpu_util > 40 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Bottleneck</div>
                <div style="font-size:1.2rem; font-weight:800; color:{COLORS['RedLine']};">{_bottleneck_stage}</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {_util_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">GPU Utilization</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_util_color};">{_gpu_util:.0f}%</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Bubble (idle)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_bubble_pct:.0f}%</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Step Time</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_total_sequential:.0f} ms</div>
            </div>
        </div>
        """))

        _pred = partB_prediction.value
        if _pred == "data" and _bottleneck_stage == "Data Loading":
            items.append(mo.callout(mo.md(
                f"**Correct for these settings.** Data loading ({_data} ms) dominates the "
                f"pipeline. The GPU is idle {_bubble_pct:.0f}% of the time waiting for data. "
                f"Faster storage or more DataLoader workers would help; more GPUs would not."
            ), kind="success"))
        elif _pred == "compute" and _bottleneck_stage == "Forward+Backward":
            items.append(mo.callout(mo.md(
                f"**Correct for these settings.** Compute ({_compute} ms) is the bottleneck. "
                f"This is the rare case where a GPU upgrade would actually help."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**At current settings, the bottleneck is {_bottleneck_stage} ({_bottleneck_time} ms).** "
                f"Try the preset configurations: with SSD and 1 GPU, data loading often dominates. "
                f"With 4+ GPUs on PCIe, gradient sync can dominate. The compute stage is "
                f"rarely the actual bottleneck."
            ), kind="warn"))

        # Presets
        items.append(mo.callout(mo.md(
            "**Try these presets** (adjust sliders manually):\n"
            "- **V100 + SSD + 1 GPU**: Data=50ms, PCIe=2ms, Compute=30ms, Sync=0ms (data-bound)\n"
            "- **V100 + NVMe + 4 GPU PCIe**: Data=10ms, PCIe=2ms, Compute=30ms, Sync=25ms (sync-bound)\n"
            "- **H100 + NVMe + 1 GPU**: Data=5ms, PCIe=1ms, Compute=50ms, Sync=0ms (compute-bound)"
        ), kind="info"))

        items.append(mo.accordion({
            "MathPeek: Pipeline Bottleneck": mo.md(f"""
```
Sequential: T_total = T_data + T_pcie + T_compute + T_sync
          = {_data} + {_pcie} + {_compute} + {_sync} = {_total_sequential:.0f} ms

GPU utilization = T_compute / T_total = {_compute}/{_total_sequential:.0f} = {_gpu_util:.0f}%

Bottleneck: {_bottleneck_stage} ({_bottleneck_time} ms)
```
Source: @sec-training-pipeline
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER: Mixed Precision Trap
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; ML Researcher, FoundationAI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We switched to mixed precision to halve our memory usage. But nvidia-smi
                shows only a 35% reduction. Is the driver reporting wrong? Half precision
                should mean half memory, right?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Dr. Elena Volkov, ML Researcher &middot; FoundationAI
            </div>
        </div>
        """))

        items.append(mo.md("""
## The FP32 Master Copy Trap

Mixed precision uses FP16/BF16 for forward and backward passes but **retains
FP32 master copies** of weights and Adam state for numerical stability.

| Component | Full FP32 | Mixed (BF16 + FP32 master) |
|-----------|-----------|---------------------------|
| Weights | 4B (FP32) | 2B (BF16) + 4B (FP32 master) = 6B |
| Gradients | 4B | 2B (BF16) |
| Adam momentum | 4B | 4B (FP32) |
| Adam variance | 4B | 4B (FP32) |
| **Total** | **16B** | **16B** (no saving on state!) |

Savings come only from **activations** being stored in BF16 (half size).
The FP32 "tail" is identical in both modes.
        """))

        items.append(partC_prediction)

        if partC_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the precision comparison."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partC_model_c, partC_precision_c], justify="start"))

        _params_b = partC_model_c.value
        _prec = partC_precision_c.value

        # Memory model with activations (estimated as fraction of base)
        # Simplified: activation memory ~ 2x weight memory for typical batch
        _act_multiplier = 2.0

        if _prec == "fp32":
            _w_gb = _params_b * 4 / 1.074  # GB
            _g_gb = _params_b * 4 / 1.074
            _m_gb = _params_b * 4 / 1.074
            _v_gb = _params_b * 4 / 1.074
            _act_gb = _w_gb * _act_multiplier
        elif _prec == "mixed":
            _w_gb = _params_b * 6 / 1.074  # BF16 (2B) + FP32 master (4B)
            _g_gb = _params_b * 2 / 1.074  # BF16
            _m_gb = _params_b * 4 / 1.074  # FP32
            _v_gb = _params_b * 4 / 1.074  # FP32
            _act_gb = (_params_b * 4 / 1.074) * _act_multiplier * 0.5  # BF16 activations
        else:  # pure bf16
            _w_gb = _params_b * 2 / 1.074
            _g_gb = _params_b * 2 / 1.074
            _m_gb = _params_b * 2 / 1.074
            _v_gb = _params_b * 2 / 1.074
            _act_gb = _w_gb * _act_multiplier

        _total_gb = _w_gb + _g_gb + _m_gb + _v_gb + _act_gb
        _fp32_total = _params_b * 16 / 1.074 + (_params_b * 4 / 1.074) * _act_multiplier
        _savings = _fp32_total / _total_gb if _total_gb > 0 else 1

        _fig = go.Figure()
        _components = ["Weights", "Gradients", "Momentum", "Variance", "Activations"]
        _vals = [_w_gb, _g_gb, _m_gb, _v_gb, _act_gb]
        _cols = [COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"],
                 COLORS["OrangeLine"], COLORS["RedLine"]]

        # Show FP32 baseline for comparison
        _fp32_vals = [_params_b * 4 / 1.074] * 4 + [(_params_b * 4 / 1.074) * _act_multiplier]

        for i, (_name, _val, _col) in enumerate(zip(_components, _vals, _cols)):
            _fig.add_trace(go.Bar(name=_name, x=[_prec.upper()], y=[_val],
                                  marker_color=_col, opacity=0.88))

        _fig.add_hline(y=H100_RAM_GB, line_dash="dash", line_color=COLORS["BlueLine"],
                       annotation_text=f"H100 ({H100_RAM_GB:.0f} GB)")
        _fig.update_layout(barmode="stack", height=380, yaxis_title="Memory (GB)",
                           title=f"Training Memory: {_params_b:.1f}B params, {_prec.upper()}",
                           legend=dict(orientation="h", y=1.15, x=0))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Total Memory</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_total_gb:,.1f} GB</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">vs FP32 Baseline</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_savings:.2f}x savings</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">FP32 State Persists</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_m_gb + _v_gb:,.1f} GB</div>
            </div>
        </div>
        """))

        _pred = partC_prediction.value
        if _pred == "45gb":
            items.append(mo.callout(mo.md(
                "**Correct.** Mixed precision yields ~1.5-1.7x savings, not 2x. "
                "The FP32 master weights and Adam state persist unchanged. "
                "Only activations and gradient buffers shrink to BF16."
            ), kind="success"))
        elif _pred == "38gb":
            items.append(mo.callout(mo.md(
                "**Half precision does not mean half memory.** The FP32 master weights "
                "and Adam momentum/variance persist at full precision in mixed mode. "
                "Actual savings are ~1.5-1.7x, not 2x."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**The actual savings are ~{_savings:.1f}x.** FP32 master weights and "
                f"Adam state ({_m_gb + _v_gb:,.1f} GB) persist unchanged. Only activations "
                f"and gradient accumulation buffers benefit from BF16."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Mixed Precision Memory": mo.md(f"""
```
FP32 total:  {_fp32_total:,.1f} GB (W=4B + G=4B + m=4B + v=4B + Act)
Mixed total: {_total_gb:,.1f} GB  (W=2B+4B_master + G=2B + m=4B + v=4B + Act_BF16)
Savings:     {_savings:.2f}x  (not 2x!)
```
What shrank: activations (BF16) and gradient buffers (BF16).
What did NOT shrink: FP32 master weights, Adam momentum, Adam variance.

Source: @sec-training-mixed-precision
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER: The Communication Tax
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Infrastructure Director, FoundationAI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We went from 4 GPUs to 8 GPUs expecting 2x speedup. We got 1.5x.
                We went to 16 GPUs expecting another 2x. We got 1.3x. Management is asking
                why we are buying GPUs that do not deliver proportional speedup.&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; David Kim, Infrastructure Director &middot; FoundationAI
            </div>
        </div>
        """))

        items.append(mo.md("""
## The Communication Tax: Diminishing Returns from Multi-GPU

Multi-GPU data-parallel training follows:

```
Speedup(N, r) = N / (1 + (N-1) * r)
```

Where r = fraction of step time spent on gradient synchronization (AllReduce).

| Scenario | r | 8-GPU Speedup |
|----------|---|---------------|
| ResNet + NVLink | 0.05 | 5.7x |
| LLM + NVLink | 0.10 | 4.7x |
| LLM + PCIe | 0.25 | 2.9x |
| Slow network | 0.40 | 2.1x |

The "communication tax" is the gap between ideal linear scaling and actual throughput.
        """))

        items.append(partD_prediction)

        if partD_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the scaling simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partD_gpus, partD_r], justify="start"))

        _n = partD_gpus.value
        _r = partD_r.value

        def _scaling(n, r):
            return n / (1 + (n - 1) * r) if n > 0 else 0

        _speedup = _scaling(_n, _r)
        _ideal = float(_n)
        _efficiency = _speedup / _ideal * 100 if _ideal > 0 else 0
        _wasted_gpus = _ideal - _speedup

        # Scaling curve
        _gpu_range = np.arange(1, 257)
        _actual_curve = np.array([_scaling(n, _r) for n in _gpu_range])
        _ideal_curve = _gpu_range.astype(float)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_gpu_range.tolist(), y=_ideal_curve.tolist(),
            mode="lines", name="Ideal (linear)",
            line=dict(color=COLORS["Grey"], width=2, dash="dash"),
        ))
        _fig.add_trace(go.Scatter(
            x=_gpu_range.tolist(), y=_actual_curve.tolist(),
            mode="lines", name=f"Actual (r={_r:.2f})",
            line=dict(color=COLORS["BlueLine"], width=2.5),
            fill="tonexty", fillcolor="rgba(203,32,45,0.08)",
        ))
        _fig.add_trace(go.Scatter(
            x=[_n], y=[_speedup],
            mode="markers", name="Current",
            marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond",
                        line=dict(width=2, color="white")),
        ))
        _fig.update_layout(
            height=380, xaxis_title="Number of GPUs", yaxis_title="Effective Speedup",
            title=f"Multi-GPU Scaling (r={_r:.2f}) -- shaded = communication tax",
            xaxis_type="log", yaxis_type="log",
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _eff_color = COLORS["GreenLine"] if _efficiency > 70 else COLORS["OrangeLine"] if _efficiency > 40 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Speedup ({_n} GPUs)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_speedup:.1f}x</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_eff_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Scaling Efficiency</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_eff_color};">{_efficiency:.0f}%</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Wasted GPU-equivalents</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_wasted_gpus:.1f}</div>
            </div>
        </div>
        """))

        _pred = partD_prediction.value
        _actual_8_015 = _scaling(8, 0.15)
        if _pred == "3.9x":
            items.append(mo.callout(mo.md(
                f"**Correct.** 8 / (1 + 7 * 0.15) = 8 / 2.05 = {_actual_8_015:.1f}x. "
                f"At r=0.15, gradient sync consumes enough of each step to cut efficiency "
                f"to {_actual_8_015/8*100:.0f}%. Scaling further to 16 GPUs yields only "
                f"{_scaling(16, 0.15):.1f}x, not 16x."
            ), kind="success"))
        elif _pred == "8x":
            items.append(mo.callout(mo.md(
                f"**Linear scaling assumes zero communication cost.** At r=0.15, "
                f"actual speedup is {_actual_8_015:.1f}x, not 8x. The formula: "
                f"Speedup = N / (1 + (N-1)*r) = 8 / (1 + 7*0.15) = {_actual_8_015:.1f}x."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**At r=0.15 with 8 GPUs, speedup is {_actual_8_015:.1f}x.** "
                f"The communication fraction r determines how quickly returns diminish. "
                f"Try different presets: NVLink (r=0.05) vs PCIe (r=0.25)."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Communication Tax": mo.md(f"""
```
Speedup(N, r) = N / (1 + (N-1) * r)
             = {_n} / (1 + {_n-1} * {_r:.2f})
             = {_n} / {1 + (_n-1)*_r:.2f}
             = {_speedup:.2f}x

Efficiency = {_speedup:.1f} / {_n} = {_efficiency:.0f}%
Wasted GPU-equiv = {_n} - {_speedup:.1f} = {_wasted_gpus:.1f}
```
Source: @sec-training-communication-tax
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        items = []
        items.append(mo.md("""
## Synthesis: Train a 7B Model

You must train a 7B parameter model on H100 GPUs. Using numbers from this lab:
        """))

        items.append(mo.callout(mo.md("""
1. **Precision mode**: Full FP32 or mixed BF16? Justify whether the model's
   parameter state fits in 80 GB with your chosen optimizer.

2. **Pipeline bottleneck**: Which stage would you optimize first? With NVMe
   storage and NVLink interconnect, which stage dominates?

3. **GPU count**: How many GPUs do you need, and what scaling efficiency
   do you expect? Show the formula with your r estimate.

4. **Communication fraction**: For your setup, estimate r and compute the
   expected speedup. Is it worth going from 8 to 16 GPUs?
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
                    <strong>1. Adam requires 16 bytes per parameter.</strong>
                    A 7B model needs 112 GB of static state before any activations.
                    Optimizer choice is a first-order memory constraint.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Training is rarely compute-bound.</strong>
                    Data loading (slow storage) and gradient sync (slow interconnect)
                    are the typical bottlenecks. Faster GPUs help only when compute
                    is actually the binding constraint.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>3. Mixed precision saves 1.5x, not 2x.</strong>
                    FP32 master weights and Adam state persist unchanged. Only
                    activations and gradient buffers benefit from half precision.
                </div>
                <div>
                    <strong>4. Multi-GPU scaling follows diminishing returns.</strong>
                    At r=0.15, 8 GPUs yield only 3.9x speedup. The communication tax
                    is the gap between ideal and actual -- and it grows with GPU count.
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
                    <strong>Lab 09: Data Selection</strong> -- Training is expensive.
                    Lab 09 shows how selecting the right training data can achieve the
                    same accuracy with 10x less compute -- making the training budget
                    from this lab go much further.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> the Training chapter for the full training memory
                    model, pipeline analysis, and scaling formulas.<br/>
                    <strong>Build:</strong> TinyTorch Module 08 -- implement a training
                    loop with gradient accumulation and mixed precision.
                </div>
            </div>
        </div>
        """))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A \u2014 The Memory Budget Shock":       build_part_a(),
        "Part B \u2014 The Pipeline Bottleneck":       build_part_b(),
        "Part C \u2014 Mixed Precision Trap":          build_part_c(),
        "Part D \u2014 The Communication Tax":         build_part_d(),
        "Synthesis":                                   build_synthesis(),
    })
    tabs
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: LEDGER HUD ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, ledger, mo, partA_prediction, partD_prediction):
    _track = ledger._state.track or "not set"
    if partA_prediction.value is not None and partD_prediction.value is not None:
        ledger.save(chapter=8, design={
            "chapter": "v1_08",
            "memory_budget_surprise": True,
            "optimizer_state_dominant": True,
            "mixed_precision_savings": "2x_memory",
            "gradient_accumulation_used": True,
            "completed": True,
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">08 &middot; The Training Gauntlet</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">CHAPTER&nbsp;8</span>
        <span class="hud-value">Model Training</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
