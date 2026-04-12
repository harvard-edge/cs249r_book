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
    ESP32 = mlsysim.Hardware.Tiny.ESP32_S3

    H100_BW_GBS = H100.memory.bandwidth.m_as("GB/s")
    H100_DISPATCH = H100.dispatch_tax.m_as("ms")
    H100_RAM_GB = H100.memory.capacity.m_as("GB")
    ESP32_RAM_KB = ESP32.memory.capacity.m_as("KiB")

    # Framework runtime overheads (MB) -- from textbook Table 7.x
    FRAMEWORK_RUNTIMES_MB = {
        "PyTorch":       1800,
        "TensorFlow":    1200,
        "ONNX Runtime":  300,
        "TF Lite":       5,
        "TensorRT":      800,
        "TF Lite Micro": 0.05,
    }

    # Framework relative latency multipliers (vs TensorRT baseline)
    FRAMEWORK_LATENCY_MULT = {
        "PyTorch":       17.3,
        "TensorFlow":    12.0,
        "ONNX Runtime":  4.5,
        "TF Lite":       8.0,
        "TensorRT":      1.0,
        "TF Lite Micro": 25.0,
    }

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return (
        COLORS, Engine, H100, ESP32,
        H100_BW_GBS, H100_DISPATCH, H100_RAM_GB, ESP32_RAM_KB,
        FRAMEWORK_RUNTIMES_MB, FRAMEWORK_LATENCY_MULT,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 07
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Framework Tax
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Dispatch Tax &middot; Kernel Fusion &middot; Compilation Break-Even &middot; Deployment Spectrum
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Frameworks are not just programming conveniences -- they are execution
                engines whose architectural decisions determine whether the same model
                runs 17x faster or fails entirely, without changing a single weight.
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
                    Chapter 7: ML Frameworks
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Dispatch: ~10 us/op</span>
                <span class="badge badge-warn">Fusion: 3x speedup</span>
                <span class="badge badge-fail">Runtime: 3,600x device memory</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the dispatch tax</strong>:
                    a model with 1,000 tiny kernels achieves <1% GPU utilization because
                    dispatch overhead exceeds compute time for every kernel.</div>
                <div style="margin-bottom: 3px;">2. <strong>Calculate the fusion dividend</strong>:
                    fusing 3 element-wise operations eliminates intermediate HBM writes,
                    yielding a ~3x speedup for memory-bound operations.</div>
                <div style="margin-bottom: 3px;">3. <strong>Find the compilation break-even</strong>:
                    torch.compile takes 30s upfront but saves ~2 ms per inference -- requiring
                    ~15,000 inferences before net positive.</div>
                <div style="margin-bottom: 3px;">4. <strong>Diagnose framework feasibility</strong>:
                    the PyTorch runtime (1,800 MB) exceeds ESP32 memory (512 KB) by 3,600x,
                    making framework choice a feasibility constraint, not just a speed choice.</div>
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
                    Dispatch overhead from the ML Frameworks chapter &middot;
                    Kernel fusion from the ML Frameworks chapter &middot;
                    Memory hierarchy from the Neural Computation chapter
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
                    Part C: ~12 min &middot; Part D: ~8 min
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
                &ldquo;The model weights are identical. The hardware is identical. So why
                does the same model run 17x faster in one framework than another -- and
                why does the fastest framework not even fit on the target device?&rdquo;
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

    - **Chapter 7: Dispatch Overhead** -- Python dispatch tax per operation,
      kernel launch overhead, and the utilization equation.
    - **Chapter 7: Kernel Fusion** -- fusing element-wise operations, HBM traffic
      reduction, and the memory-bound speedup model.
    - **Chapter 7: Compilation** -- torch.compile, JIT compilation, amortization
      and break-even analysis.
    - **Chapter 7: Framework Deployment Spectrum** -- runtime memory footprints
      from PyTorch (1,800 MB) to TF Lite Micro (50 KB).
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE B-D: ALL PARTS AS TABS
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 4: TABS CELL ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, Engine, H100, ESP32, H100_BW_GBS, H100_DISPATCH, H100_RAM_GB,
    ESP32_RAM_KB, FRAMEWORK_RUNTIMES_MB, FRAMEWORK_LATENCY_MULT,
    apply_plotly_theme, go, math, mo, np, ledger, mlsysim,
):
    # ─────────────────────────────────────────────────────────────────────
    # SHARED WIDGET STATE
    # ─────────────────────────────────────────────────────────────────────

    # Part A widgets
    partA_prediction = mo.ui.radio(
        options={
            "A) Model A (1,000 ops) -- more operations means busier GPU": "a",
            "B) Model B (20 ops) -- fewer but larger operations": "b",
            "C) About the same -- same GPU": "same",
            "D) Depends on batch size": "depends",
        },
        label="Two models on the same GPU. Model A (KWS) has 1,000 operations of 5 us each. "
              "Model B (GPT-2-like) has 20 operations of 500 us each. Which achieves higher GPU utilization?",
    )
    return (partA_prediction,)

@app.cell(hide_code=True)
def _(mo, partA_prediction):
    mo.stop(partA_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partA_kernels = mo.ui.slider(
        start=10, stop=2000, value=1000, step=10, label="Number of kernels",
    )
    partA_compute_us = mo.ui.slider(
        start=1, stop=10000, value=5, step=1, label="Compute per kernel (us)",
    )

    # Part B widgets
    partB_prediction = mo.ui.radio(
        options={
            "A) ~1.2x (minor improvement)": "1.2x",
            "B) ~1.5x (noticeable)": "1.5x",
            "C) ~3x (significant)": "3x",
            "D) ~10x": "10x",
        },
        label="Fusing 3 element-wise operations into one kernel eliminates intermediate "
              "HBM writes. What speedup do you expect?",
    )
    return (partB_prediction,)

@app.cell(hide_code=True)
def _(mo, partB_prediction):
    mo.stop(partB_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partB_num_ops = mo.ui.slider(
        start=2, stop=8, value=3, step=1, label="Operations to fuse",
    )
    partB_tensor_mb = mo.ui.slider(
        start=1, stop=1000, value=100, step=10, label="Tensor size (MB)",
    )

    # Part C widgets
    partC_prediction = mo.ui.radio(
        options={
            "A) ~100 (almost immediately)": "100",
            "B) ~1,000": "1000",
            "C) ~10,000": "10000",
            "D) ~15,000+": "15000",
        },
        label="torch.compile gives 48% speedup on ResNet-50 but takes 30s to compile. "
              "How many inferences before compilation pays off?",
    )
    return (partC_prediction,)

@app.cell(hide_code=True)
def _(mo, partC_prediction):
    mo.stop(partC_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partC_compile_time = mo.ui.slider(
        start=5, stop=300, value=30, step=5, label="Compile time (seconds)",
    )
    partC_volume = mo.ui.slider(
        start=10, stop=1000000, value=10000, step=100, label="Deployment volume (req/hour)",
    )

    # Part D widgets
    partD_prediction = mo.ui.radio(
        options={
            "A) ~10x": "10x",
            "B) ~100x": "100x",
            "C) ~3,500x": "3500x",
            "D) It fits with optimization": "fits",
        },
        label="PyTorch runtime alone requires ~1,800 MB. The ESP32 has 512 KB. "
              "By what factor does the runtime exceed device memory?",
    )
    return (partD_prediction,)

@app.cell(hide_code=True)
def _(mo, partD_prediction):
    mo.stop(partD_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partD_framework = mo.ui.dropdown(
        options={"PyTorch": "PyTorch", "TensorFlow": "TensorFlow",
                 "ONNX Runtime": "ONNX Runtime", "TF Lite": "TF Lite",
                 "TensorRT": "TensorRT", "TF Lite Micro": "TF Lite Micro"},
        value="PyTorch", label="Framework",
    )
    partD_target = mo.ui.radio(
        options={"Cloud (H100)": "cloud", "Edge (Jetson 16GB)": "edge", "MCU (ESP32 512KB)": "mcu"},
        value="MCU (ESP32 512KB)", label="Deployment target:", inline=True,
    )

    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER: The Dispatch Tax
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; ML Platform Engineer, SoundSense AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our keyword spotting model has 1,000 tiny operations and achieves less
                than 1% GPU utilization. Our language model has only 20 operations but hits
                90%. Both run on the same H100. The intern says more operations should mean
                higher utilization. What is going on?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Kevin Park, ML Platform Engineer &middot; SoundSense AI
            </div>
        </div>
        """))

        items.append(mo.md("""
## The Dispatch Tax: Every Kernel Launch Costs ~10 us

Python dispatch overhead is approximately **10 us per operation**, regardless of
how much compute the operation performs. GPU utilization depends on the ratio:

```
Utilization = total_compute / (total_compute + total_dispatch)
            = (N * T_compute) / (N * T_compute + N * T_dispatch)
            = T_compute / (T_compute + T_dispatch)
```

A kernel with 5 us of compute and 10 us of dispatch achieves only 33% utilization.
A kernel with 500 us of compute and 10 us of dispatch achieves 98% utilization.
        """))

        items.append(partA_prediction)

        if partA_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the dispatch tax simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partA_kernels, partA_compute_us], justify="start"))

        _n = partA_kernels.value
        _compute_us = partA_compute_us.value
        _dispatch_us = 10.0  # Fixed dispatch overhead

        _total_compute = _n * _compute_us
        _total_dispatch = _n * _dispatch_us
        _total_time = _total_compute + _total_dispatch
        _utilization = (_total_compute / _total_time) * 100 if _total_time > 0 else 0

        # Presets for comparison
        _kws = {"name": "KWS (1000 kernels, 5 us)", "n": 1000, "c": 5}
        _gpt = {"name": "GPT-2-like (20 kernels, 500 us)", "n": 20, "c": 500}
        _kws_util = (_kws["c"] / (_kws["c"] + _dispatch_us)) * 100
        _gpt_util = (_gpt["c"] / (_gpt["c"] + _dispatch_us)) * 100

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            name="Compute", x=["KWS-like", "GPT-2-like", "Your Setting"],
            y=[_kws["n"] * _kws["c"] / 1000, _gpt["n"] * _gpt["c"] / 1000, _total_compute / 1000],
            marker_color=COLORS["GreenLine"], opacity=0.88,
        ))
        _fig.add_trace(go.Bar(
            name="Dispatch Overhead", x=["KWS-like", "GPT-2-like", "Your Setting"],
            y=[_kws["n"] * _dispatch_us / 1000, _gpt["n"] * _dispatch_us / 1000, _total_dispatch / 1000],
            marker_color=COLORS["OrangeLine"], opacity=0.88,
        ))
        _fig.update_layout(
            barmode="stack", height=340, yaxis_title="Time (ms)",
            title="Compute vs. Dispatch Overhead",
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _util_color = COLORS["GreenLine"] if _utilization > 70 else COLORS["OrangeLine"] if _utilization > 30 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_util_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">GPU Utilization</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_util_color};">{_utilization:.1f}%</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">KWS Utilization</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">{_kws_util:.1f}%</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">GPT-2 Utilization</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['GreenLine']};">{_gpt_util:.1f}%</div>
            </div>
        </div>
        """))

        _pred = partA_prediction.value
        if _pred == "b":
            items.append(mo.callout(mo.md(
                f"**Correct.** Model B's 20 large kernels (500 us each) amortize the 10 us "
                f"dispatch overhead, achieving {_gpt_util:.0f}% utilization. Model A's 1,000 "
                f"tiny kernels (5 us each) spend more time on dispatch than compute: "
                f"only {_kws_util:.0f}% utilization."
            ), kind="success"))
        elif _pred == "a":
            items.append(mo.callout(mo.md(
                f"**More operations does not mean higher utilization.** Each of Model A's "
                f"1,000 kernels takes only 5 us to compute but 10 us to launch -- the GPU "
                f"is idle 67% of the time. Model B's 20 large kernels achieve {_gpt_util:.0f}%."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**The key factor is compute-per-kernel, not total operations.** "
                f"KWS: {_kws_util:.0f}% utilization. GPT-2: {_gpt_util:.0f}%. "
                f"The dispatch tax is fixed (~10 us) regardless of kernel compute time."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Dispatch Tax": mo.md(f"""
```
Utilization = T_compute / (T_compute + T_dispatch)
KWS:  5 / (5 + 10) = {_kws_util:.1f}%
GPT2: 500 / (500 + 10) = {_gpt_util:.1f}%
```
Source: @sec-frameworks-dispatch-overhead
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER: The Fusion Dividend
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Compiler Engineer, SoundSense AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;I fused LayerNorm + Dropout + ReLU into a single kernel. The compute
                is identical. But inference is 3x faster. How can removing zero compute
                operations give a 3x speedup?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Dr. Yuki Tanaka, Compiler Engineer &middot; SoundSense AI
            </div>
        </div>
        """))

        items.append(mo.md("""
## Kernel Fusion: The Speedup Comes from Memory, Not Compute

Element-wise operations (ReLU, Dropout, LayerNorm) have arithmetic intensity
< 1 FLOP/byte -- they are **permanently memory-bound**. Each unfused operation:
1. Reads the entire tensor from HBM
2. Performs a trivial computation
3. Writes the entire tensor back to HBM

Fusing N operations into one kernel: **1 read + 1 write** instead of N reads + N writes.
Memory traffic drops by ~Nx. Since these ops are memory-bound, latency drops proportionally.
        """))

        items.append(partB_prediction)

        if partB_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the fusion simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partB_num_ops, partB_tensor_mb], justify="start"))

        _n_ops = partB_num_ops.value
        _tensor_mb = partB_tensor_mb.value
        _bw = H100_BW_GBS  # GB/s

        # Unfused: each op does 1 read + 1 write = 2 * tensor_size
        _unfused_traffic_mb = 2 * _tensor_mb * _n_ops
        _unfused_time_ms = _unfused_traffic_mb / (_bw * 1000) * 1000  # MB / (GB/s * 1000 MB/GB) * 1000 ms

        # Fused: 1 read + 1 write = 2 * tensor_size
        _fused_traffic_mb = 2 * _tensor_mb
        _fused_time_ms = _fused_traffic_mb / (_bw * 1000) * 1000

        _speedup = _unfused_time_ms / _fused_time_ms if _fused_time_ms > 0 else 1
        _traffic_reduction = _unfused_traffic_mb / _fused_traffic_mb if _fused_traffic_mb > 0 else 1

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            name="HBM Traffic (MB)", x=["Eager (unfused)", "Fused"],
            y=[_unfused_traffic_mb, _fused_traffic_mb],
            marker_color=[COLORS["RedLine"], COLORS["GreenLine"]], opacity=0.88,
            text=[f"{_unfused_traffic_mb:,.0f} MB", f"{_fused_traffic_mb:,.0f} MB"],
            textposition="outside",
        ))
        _fig.update_layout(
            height=320, yaxis_title="HBM Traffic (MB)",
            title=f"Memory Traffic: Eager vs. Fused ({_n_ops} ops, {_tensor_mb} MB tensor)",
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Speedup</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['GreenLine']};">{_speedup:.1f}x</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Traffic Reduction</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">{_traffic_reduction:.1f}x</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Unfused AI</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">~0.5</div>
            </div>
        </div>
        """))

        _pred = partB_prediction.value
        if _pred == "3x":
            items.append(mo.callout(mo.md(
                f"**Correct.** Fusing {_n_ops} ops reduces HBM traffic by {_traffic_reduction:.0f}x. "
                f"Since element-wise ops are entirely memory-bound (AI < 1), latency drops "
                f"proportionally: {_speedup:.1f}x speedup with zero compute reduction."
            ), kind="success"))
        elif _pred == "1.2x":
            items.append(mo.callout(mo.md(
                f"**You are thinking about compute savings -- but there are none.** "
                f"The speedup comes entirely from memory traffic reduction. "
                f"Fusing {_n_ops} ops: {_unfused_traffic_mb:,.0f} MB -> {_fused_traffic_mb:,.0f} MB "
                f"= {_speedup:.1f}x speedup."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Close.** Fusing {_n_ops} ops gives exactly {_speedup:.1f}x speedup because "
                f"memory traffic drops by {_traffic_reduction:.0f}x and these ops are 100% memory-bound."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Fusion Traffic Model": mo.md(f"""
```
Unfused: 2 * tensor * N_ops = 2 * {_tensor_mb} * {_n_ops} = {_unfused_traffic_mb:,} MB
Fused:   2 * tensor         = 2 * {_tensor_mb}             = {_fused_traffic_mb:,} MB
Speedup: {_unfused_traffic_mb} / {_fused_traffic_mb} = {_speedup:.1f}x
```
Source: @sec-frameworks-kernel-fusion
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER: The Compilation Break-Even
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; DevOps Lead, SoundSense AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;torch.compile gives us a 48% speedup. Our research team wants to use
                it for every experiment. But each compilation takes 30 seconds. For a quick
                10-run experiment, that is 30 seconds of compilation for 0.04 seconds of
                inference. Is this actually worth it?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Chen Li, DevOps Lead &middot; SoundSense AI
            </div>
        </div>
        """))

        items.append(mo.md("""
## Compilation Has a Fixed Upfront Cost That Must Be Amortized

torch.compile (and similar JIT compilers) optimize the computation graph,
fuse kernels, and generate specialized code. This takes **30 seconds** for
ResNet-50 (minutes for larger models).

The savings per inference are small: from ~4.2 ms (eager) to ~2.2 ms (compiled)
= **2.0 ms saved per inference**.

```
Break-even = compile_time / savings_per_inference
           = 30,000 ms / 2.0 ms = 15,000 inferences
```

For a research notebook with 10 runs: compilation makes things **slower**.
For a production endpoint with 1M requests/day: it pays off in seconds.
        """))

        items.append(partC_prediction)

        if partC_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the break-even simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partC_compile_time, partC_volume], justify="start"))

        _compile_s = partC_compile_time.value
        _compile_ms = _compile_s * 1000
        _volume = partC_volume.value

        _eager_ms = 4.2
        _compiled_ms = 2.2
        _saving_ms = _eager_ms - _compiled_ms
        _breakeven = int(math.ceil(_compile_ms / _saving_ms)) if _saving_ms > 0 else float("inf")
        _breakeven_hours = _breakeven / _volume if _volume > 0 else float("inf")

        # Cumulative time curves
        _inferences = np.arange(0, min(_breakeven * 3, 100001), max(1, min(_breakeven * 3, 100001) // 500))
        _eager_cum = _inferences * _eager_ms
        _compiled_cum = _compile_ms + _inferences * _compiled_ms

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_inferences.tolist(), y=(_eager_cum / 1000).tolist(),
            mode="lines", name="Eager",
            line=dict(color=COLORS["OrangeLine"], width=2.5),
        ))
        _fig.add_trace(go.Scatter(
            x=_inferences.tolist(), y=(_compiled_cum / 1000).tolist(),
            mode="lines", name="Compiled",
            line=dict(color=COLORS["GreenLine"], width=2.5),
        ))
        if _breakeven < _inferences[-1]:
            _fig.add_vline(x=_breakeven, line_dash="dash", line_color=COLORS["RedLine"],
                           annotation_text=f"Break-even: {_breakeven:,} inferences")
        _fig.update_layout(
            height=380, xaxis_title="Inferences", yaxis_title="Cumulative Time (seconds)",
            title=f"Compilation Break-Even Analysis (compile={_compile_s}s, saving={_saving_ms:.1f}ms/inf)",
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _roi_positive = _volume > 0 and _breakeven_hours < 1
        _roi_color = COLORS["GreenLine"] if _roi_positive else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Break-Even Point</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_breakeven:,} inferences</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_roi_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Time to Break-Even</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_roi_color};">
                    {_breakeven_hours:.1f} hours</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Saving per Inference</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_saving_ms:.1f} ms</div>
            </div>
        </div>
        """))

        _pred = partC_prediction.value
        if _pred == "15000":
            items.append(mo.callout(mo.md(
                f"**Correct.** At {_saving_ms:.1f} ms savings per inference, recovering "
                f"{_compile_s} seconds requires {_breakeven:,} inferences. For research (10 runs): "
                f"compilation is a net loss. For production (1M/day): break-even in seconds."
            ), kind="success"))
        elif _pred == "100":
            items.append(mo.callout(mo.md(
                f"**The saving per inference is smaller than you think.** Each inference saves "
                f"only {_saving_ms:.1f} ms. To recover {_compile_s} seconds, you need "
                f"{_breakeven:,} inferences, not 100."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Close.** The actual break-even is {_breakeven:,} inferences. "
                f"The key insight: compilation is an investment that only pays off at scale."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Break-Even Analysis": mo.md(f"""
```
Eager latency:    {_eager_ms} ms
Compiled latency: {_compiled_ms} ms
Saving:           {_saving_ms:.1f} ms/inference
Compile cost:     {_compile_ms:,.0f} ms ({_compile_s}s)
Break-even:       {_compile_ms:,.0f} / {_saving_ms:.1f} = {_breakeven:,} inferences
```
At {_volume:,} req/hour: break-even in {_breakeven_hours:.1f} hours.

Source: @sec-frameworks-compilation
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER: The Deployment Spectrum
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Embedded Engineer, SoundSense AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Management wants us to run our PyTorch model on ESP32 microcontrollers.
                I tried explaining the memory gap but they think I am being difficult. Can you
                give me the exact number for how much the runtime exceeds device memory?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Ana Santos, Embedded Engineer &middot; SoundSense AI
            </div>
        </div>
        """))

        items.append(mo.md("""
## Framework Selection Determines Feasibility, Not Just Speed

The same ResNet-50 -- identical weights -- spans **17x latency** and **56x memory**
across frameworks. On a 512 KB microcontroller, the question is not "which framework
is fastest" but "which framework fits at all."

| Framework | Runtime Memory | Relative |
|-----------|---------------|----------|
| PyTorch | ~1,800 MB | 3,600x ESP32 |
| TensorFlow | ~1,200 MB | 2,400x ESP32 |
| ONNX Runtime | ~300 MB | 600x ESP32 |
| TF Lite | ~5 MB | 10x ESP32 |
| TensorRT | ~800 MB | 1,600x ESP32 |
| TF Lite Micro | ~0.05 MB | Fits! |
        """))

        items.append(partD_prediction)

        if partD_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the framework comparison."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partD_framework, partD_target], justify="start"))

        _fw = partD_framework.value
        _target = partD_target.value
        _target_ram_mb = {"cloud": H100_RAM_GB * 1024, "edge": 16 * 1024, "mcu": ESP32_RAM_KB / 1024}[_target]
        _target_name = {"cloud": f"H100 ({H100_RAM_GB:.0f} GB)", "edge": "Jetson (16 GB)", "mcu": f"ESP32 ({ESP32_RAM_KB:.0f} KB)"}[_target]

        _runtime_mb = FRAMEWORK_RUNTIMES_MB[_fw]
        _fits = _runtime_mb <= _target_ram_mb
        _ratio = _runtime_mb / _target_ram_mb if _target_ram_mb > 0 else float("inf")

        # All frameworks comparison
        _fw_names = list(FRAMEWORK_RUNTIMES_MB.keys())
        _fw_runtimes = [FRAMEWORK_RUNTIMES_MB[f] for f in _fw_names]
        _fw_fits = [r <= _target_ram_mb for r in _fw_runtimes]
        _fw_colors = [COLORS["GreenLine"] if f else COLORS["RedLine"] for f in _fw_fits]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=_fw_names, y=_fw_runtimes,
            marker_color=_fw_colors, opacity=0.88,
            text=[f"{r:,.1f} MB" if r >= 1 else f"{r*1024:.0f} KB" for r in _fw_runtimes],
            textposition="outside",
        ))
        _fig.add_hline(y=_target_ram_mb, line_dash="dash", line_color=COLORS["BlueLine"],
                       annotation_text=f"{_target_name}")
        _fig.update_layout(
            height=380, yaxis_title="Runtime Memory (MB)", yaxis_type="log",
            title=f"Framework Runtime Memory vs. {_target_name}",
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _status_color = COLORS["GreenLine"] if _fits else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_status_color}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Status</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_status_color};">
                    {"FITS" if _fits else "OOM"}</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">{_fw} Runtime</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_runtime_mb:,.1f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['RedLine'] if _ratio > 10 else COLORS['GreenLine']}; flex:1;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.78rem; font-weight:600;">Runtime/Device Ratio</div>
                <div style="font-size:1.5rem; font-weight:800;
                     color:{COLORS['RedLine'] if _ratio > 10 else COLORS['GreenLine']};">
                    {_ratio:,.0f}x</div>
            </div>
        </div>
        """))

        if not _fits:
            items.append(mo.callout(mo.md(
                f"**OOM -- Framework runtime alone exceeds device memory by {_ratio:,.0f}x.** "
                f"{_fw} requires {_runtime_mb:,.1f} MB. {_target_name} has {_target_ram_mb:,.1f} MB. "
                f"This is not a model size problem -- the **framework itself** does not fit."
            ), kind="danger"))

        _pred = partD_prediction.value
        if _pred == "3500x":
            items.append(mo.callout(mo.md(
                "**Correct.** PyTorch runtime (1,800 MB) / ESP32 memory (0.5 MB) = 3,600x. "
                "Framework selection is a feasibility constraint on microcontrollers. "
                "Only TF Lite Micro (50 KB runtime) fits on an ESP32."
            ), kind="success"))
        elif _pred == "fits":
            items.append(mo.callout(mo.md(
                "**No amount of optimization makes PyTorch fit on 512 KB.** "
                "The runtime alone -- before loading any model -- requires 1,800 MB. "
                "That is 3,600x the ESP32's total memory."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**The gap is larger than expected.** 1,800 MB / 0.5 MB = 3,600x. "
                f"Most engineers never think about framework runtime footprint because "
                f"on cloud hardware, 1,800 MB is negligible."
            ), kind="warn"))

        items.append(mo.accordion({
            "MathPeek: Framework Feasibility": mo.md(f"""
```
PyTorch runtime:  1,800 MB
ESP32 memory:     {ESP32_RAM_KB:.0f} KB = {ESP32_RAM_KB/1024:.2f} MB
Ratio:            1,800 / {ESP32_RAM_KB/1024:.2f} = {1800/(ESP32_RAM_KB/1024):,.0f}x
```
Only TF Lite Micro (0.05 MB) fits on the ESP32.

Source: @sec-frameworks-deployment-spectrum
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        items = []
        items.append(mo.md("""
## Synthesis: Two Deployments, Two Frameworks

You manage two deployments:
1. A KWS model (1,000 small kernels, 5 us each) on an ESP32
2. ResNet-50 on a cloud endpoint serving 500K requests/day
        """))

        items.append(mo.callout(mo.md("""
For each deployment, specify:
- **Framework choice** and why
- **Whether to compile** (will compilation pay off?)
- **Whether kernel fusion helps** (are the kernels memory-bound?)
- **Expected GPU/NPU utilization**

Justify each choice with specific numbers from the lab.
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
                    <strong>1. The dispatch tax punishes small kernels.</strong>
                    At 10 us dispatch overhead, a 5 us kernel achieves only 33% utilization.
                    Models with many tiny operations (KWS) suffer far more than models with
                    few large operations (LLMs).
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Fusion speeds up memory-bound ops proportionally.</strong>
                    Fusing 3 element-wise operations yields ~3x speedup by eliminating
                    intermediate HBM traffic. The compute is identical -- all savings come
                    from memory.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>3. Compilation is an investment with a break-even point.</strong>
                    30s compile time / 2ms savings = 15,000 inferences to break even.
                    Production endpoints amortize instantly. Research notebooks lose time.
                </div>
                <div>
                    <strong>4. Framework selection is a feasibility constraint.</strong>
                    PyTorch runtime (1,800 MB) exceeds ESP32 memory (512 KB) by 3,600x.
                    On microcontrollers, framework choice determines whether deployment
                    is physically possible.
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
                    <strong>Lab 08: Model Training</strong> -- Frameworks execute the model.
                    But training adds four new costs: optimizer state memory, pipeline bubbles,
                    mixed-precision traps, and communication overhead. Lab 08 reveals why
                    a 7B model needs 112 GB before storing a single activation.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> the ML Frameworks chapter for dispatch overhead,
                    kernel fusion, and compilation trade-offs.<br/>
                    <strong>Build:</strong> TinyTorch Module 07 -- implement a simple
                    computation graph executor with eager and compiled modes.
                </div>
            </div>
        </div>
        """))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A \u2014 The Dispatch Tax":            build_part_a(),
        "Part B \u2014 The Fusion Dividend":         build_part_b(),
        "Part C \u2014 Compilation Break-Even":      build_part_c(),
        "Part D \u2014 The Deployment Spectrum":     build_part_d(),
        "Synthesis":                                 build_synthesis(),
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
        ledger.save(chapter=7, design={
            "chapter": "v1_07",
            "dispatch_overhead_discovered": True,
            "fusion_speedup_ratio": "17x",
            "compilation_breakeven_batches": 50,
            "eager_vs_compiled_tradeoff": "compile_for_production",
            "completed": True,
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">07 &middot; The Framework Tax</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">CHAPTER&nbsp;7</span>
        <span class="hud-value">ML Frameworks</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
