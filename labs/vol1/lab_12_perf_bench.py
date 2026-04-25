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
            "../../wheels/mlsysim-0.1.1-py3-none-any.whl", keep_going=False
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
    H100_TDP    = mlsysim.Hardware.Cloud.H100.tdp.m_as("W")

    A100_TFLOPS = mlsysim.Hardware.Cloud.A100.compute.peak_flops.m_as("TFLOPs/s")
    A100_BW     = mlsysim.Hardware.Cloud.A100.memory.bandwidth.m_as("GB/s")

    JETSON_TFLOPS = mlsysim.Hardware.Edge.JetsonOrinNX.compute.peak_flops.m_as("TFLOPs/s")
    JETSON_TDP    = mlsysim.Hardware.Edge.JetsonOrinNX.tdp.m_as("W")

    RESNET50_FLOPS = mlsysim.Models.ResNet50.inference_flops.m_as("flop")
    RESNET50_PARAMS = mlsysim.Models.ResNet50.parameters.m_as("count")

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        A100_BW, A100_TFLOPS,
        COLORS, H100_BW, H100_TDP, H100_TFLOPS,
        JETSON_TDP, JETSON_TFLOPS,
        LAB_CSS, RESNET50_FLOPS, RESNET50_PARAMS,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 12
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Benchmarking Trap
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Amdahl &middot; Thermal Cliff &middot; Multi-Metric &middot; Tail Latency
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Vendor benchmarks are designed to make hardware look good. Your job is to
                make hardware look honest &mdash; and the gap between the two is where
                millions of dollars disappear.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts &middot; ~52 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter 12: Performance Benchmarking
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Amdahl's Law</span>
                <span class="badge badge-warn">Thermal Throttle</span>
                <span class="badge badge-fail">Tail Latency</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Apply Amdahl's Law to system speedup</strong> &mdash;
                    a 10x inference speedup with 45% non-inference overhead yields only 2.0x
                    end-to-end improvement.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose thermal throttling</strong> &mdash;
                    vendor burst benchmarks at 30 FPS degrade to 15 FPS sustained in
                    fanless enclosures after thermal steady-state.</div>
                <div style="margin-bottom: 3px;">3. <strong>Distinguish average from tail latency</strong> &mdash;
                    50 ms average latency can hide 500 ms p99 that violates a 200 ms SLO
                    for 1% of requests.</div>
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
                    Iron Law from the ML Systems chapter &middot;
                    Roofline model from the Hardware Acceleration chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~52 min</strong><br/>
                    A: ~12 &middot; B: ~12 &middot; C: ~10 &middot; D: ~12 min
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
                &ldquo;A vendor claims 10x faster inference, 30 FPS sustained, 50 ms average
                latency. Which of these numbers can you trust in production?&rdquo;
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

    - **Chapter 12: Performance Benchmarking** -- Amdahl's Law for systems, thermal
      throttling model, multi-metric SLO gates, and latency distributions.
    - **Chapter 11: Hardware Acceleration** -- Roofline model and hardware constraints.
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# CELL 4: TABS (Parts A-D + Synthesis)
# ═════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(
    COLORS, H100_BW, H100_TDP, H100_TFLOPS,
    JETSON_TDP, JETSON_TFLOPS,
    RESNET50_FLOPS, RESNET50_PARAMS,
    apply_plotly_theme, go, math, mo, np,
):
    # ── Widgets ───────────────────────────────────────────────────────────
    pA_pred = mo.ui.radio(
        options={
            "A) ~10x (the new GPU is 10x faster)": "10x",
            "B) ~5x (about half speeds up)": "5x",
            "C) ~2.0x (Amdahl's Law caps the gain)": "2x",
            "D) ~1.5x (even worse than expected)": "1_5x",
        },
        label="Replace inference GPU with 10x faster. Pipeline: 45% preprocessing + 55% inference. "
              "End-to-end speedup?",
    )
    return (pA_pred,)

@app.cell(hide_code=True)
def _(mo):
    pA_speedup = mo.ui.slider(start=1, stop=100, value=10, step=1, label="Inference speedup (x)")
    pA_serial = mo.ui.slider(start=5, stop=80, value=45, step=5, label="Non-inference fraction (%)")

    pB_pred = mo.ui.radio(
        options={
            "A) ~28 FPS (minor degradation)": "28",
            "B) ~24 FPS (some throttling)": "24",
            "C) ~15 FPS (thermal throttle halves performance)": "15",
            "D) ~8 FPS (severe throttling)": "8",
        },
        label="Edge device benchmarks at 30 FPS (1-min vendor test). "
              "Sustained FPS after 10 min in fanless enclosure at 35C?",
    )
    return (pA_serial, pA_speedup, pB_pred)

@app.cell(hide_code=True)
def _(mo):
    pB_time = mo.ui.slider(start=0, stop=600, value=0, step=10, label="Time (seconds)")
    pB_ambient = mo.ui.slider(start=20, stop=45, value=35, step=1, label="Ambient temp (C)")
    pB_cooling = mo.ui.dropdown(
        options={"Active fan": "active", "Passive heatsink": "passive", "Fanless": "fanless"},
        value="Fanless",
        label="Cooling type",
    )

    pC_pred = mo.ui.radio(
        options={
            "A) Yes -- highest accuracy is always best": "yes",
            "B) Probably -- other metrics secondary": "probably",
            "C) No -- violates latency SLO (p99 > 100 ms)": "no_latency",
            "D) No -- violates power budget (> 5 W)": "no_power",
        },
        label="Config A has highest accuracy (94%). Is it deployable?",
    )
    return (pB_ambient, pB_cooling, pB_time, pC_pred)

@app.cell(hide_code=True)
def _(mo):
    pC_batch = mo.ui.slider(start=1, stop=64, value=1, step=1, label="Batch size")
    pC_precision = mo.ui.dropdown(
        options={"FP32": "fp32", "FP16": "fp16", "INT8": "int8"},
        value="FP32",
        label="Precision",
    )

    pD_pred = mo.ui.radio(
        options={
            "A) Yes -- 200 ms is 4x the average": "yes",
            "B) Probably -- p99 ~2x the average": "probably",
            "C) No -- p99 ~500 ms due to heavy tail": "no",
            "D) Cannot determine": "unknown",
        },
        label="Inference service: 50 ms average latency. SLO: 200 ms p99. Is the SLO satisfied?",
    )
    return (pC_batch, pC_precision, pD_pred)

# ─── widget cell: extracted from tabs cell body (#1332 polish) ────
@app.cell(hide_code=True)
def _(mo):
    pD_sigma = mo.ui.slider(start=0.1, stop=1.5, value=0.8, step=0.05, label="Tail heaviness (sigma)")
    pD_slo = mo.ui.slider(start=50, stop=500, value=200, step=10, label="SLO threshold (ms)")
    return (pD_sigma, pD_slo)


@app.cell(hide_code=True)
def _(
    mo, pA_pred, pA_serial, pA_speedup,
    pB_ambient, pB_cooling, pB_pred, pB_time,
    pC_batch, pC_precision, pC_pred, pD_pred,
    pD_sigma, pD_slo,
):

    # ─────────────────────────────────────────────────────────────────────
    # PART A: The Amdahl Ceiling
    # ─────────────────────────────────────────────────────────────────────
    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; CFO, VisionStack AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Engineering wants a $30K GPU upgrade that is 10x faster for inference.
                They claim it will make the whole pipeline 10x faster. Finance wants to know
                the actual end-to-end speedup before approving the budget.&rdquo;
            </div>
        </div>
        """))
        items.append(mo.md("""
        ## Amdahl's Law: Component Speedup is Not System Speedup

        ```
        System Speedup = 1 / (f_serial + f_parallel / S)
        ```

        Where f_serial is the fraction of time NOT affected by the speedup.
        A 10x inference speedup with 45% preprocessing overhead yields:

        ```
        Speedup = 1 / (0.45 + 0.55/10) = 1 / 0.505 = 1.98x
        ```

        The other 8x of hardware investment is wasted on a bottleneck that has already moved.
        """))
        items.append(pA_pred)
        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the Amdahl simulator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pA_speedup, pA_serial], justify="start"))

        _S = pA_speedup.value
        _f = pA_serial.value / 100.0
        _system_speedup = 1.0 / (_f + (1 - _f) / _S)
        _asymptote = 1.0 / _f if _f > 0 else float('inf')
        _wasted = (_S - _system_speedup) / _S * 100  # % of component speedup wasted

        # Amdahl curve
        _speeds = np.arange(1, 101)
        _sys_speeds = [1.0 / (_f + (1 - _f) / s) for s in _speeds]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_speeds.tolist(), y=_sys_speeds, mode="lines",
            line=dict(color=COLORS["BlueLine"], width=3),
            name="System Speedup",
        ))
        _fig.add_hline(y=_asymptote, line_dash="dash", line_color=COLORS["RedLine"],
                       annotation_text=f"Asymptote: {_asymptote:.1f}x")
        _fig.add_trace(go.Scatter(
            x=[_S], y=[_system_speedup], mode="markers",
            marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond"),
            name=f"Your GPU: {_S}x",
        ))
        _fig.update_layout(
            height=340,
            xaxis=dict(title="Component Speedup (x)"),
            yaxis=dict(title="System Speedup (x)", range=[0, max(_asymptote * 1.2, 5)]),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        # Before/after waterfall
        _t_pre = _f
        _t_inf_before = 1 - _f
        _t_inf_after = (1 - _f) / _S
        _total_before = 1.0
        _total_after = _t_pre + _t_inf_after

        _fig2 = go.Figure()
        _fig2.add_trace(go.Bar(name="Preprocessing", x=["Before", "After"],
            y=[_t_pre * 100, _t_pre * 100], marker_color=COLORS["OrangeLine"]))
        _fig2.add_trace(go.Bar(name="Inference", x=["Before", "After"],
            y=[_t_inf_before * 100, _t_inf_after * 100], marker_color=COLORS["BlueLine"]))
        _fig2.update_layout(
            barmode="stack", height=280,
            yaxis=dict(title="Time (% of original)"),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig2)
        items.append(mo.as_html(_fig2))

        _new_pre_frac = _t_pre / _total_after * 100
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">System Speedup</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_system_speedup:.2f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">not {_S}x</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Wasted Speedup</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">{_wasted:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">of hardware investment</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">New Bottleneck</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_new_pre_frac:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">preprocessing (was {_f*100:.0f}%)</div>
            </div>
        </div>
        """))

        items.append(mo.md(f"""
**Amdahl's Law -- Live Calculation**

```
f_serial = {_f:.2f} (non-inference fraction)
S        = {_S}x (inference speedup)
System   = 1 / ({_f:.2f} + {1-_f:.2f}/{_S}) = 1 / {_f + (1-_f)/_S:.3f} = {_system_speedup:.2f}x
Asymptote = 1 / {_f:.2f} = {_asymptote:.1f}x (cannot exceed this regardless of S)
```
*Source: Chapter 12, Amdahl's Law applied to ML pipelines*
        """))

        if pA_pred.value == "2x":
            items.append(mo.callout(mo.md(f"**Correct.** A {_S}x component speedup yields only "
                f"{_system_speedup:.1f}x system speedup. The {_f*100:.0f}% preprocessing fraction "
                f"is now {_new_pre_frac:.0f}% of total time. The bottleneck has moved."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"**The system speedup is only {_system_speedup:.1f}x, not {_S}x.** "
                f"Amdahl's Law: the {_f*100:.0f}% serial fraction caps the gain at {_asymptote:.1f}x. "
                "The $30K GPU upgrade delivered 2x, not 10x."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Amdahl's Law": mo.md("""
**Formula:**
$$
\\text{Speedup}_{\\text{system}} = \\frac{1}{(1 - f) + \\frac{f}{S}}
$$

Asymptotic limit as $S \\to \\infty$:
$$
\\text{Speedup}_{\\max} = \\frac{1}{1 - f}
$$

**Variables:**
- **$f$**: fraction of execution time that is parallelizable/accelerated
- **$S$**: speedup of the accelerated portion
- **$1 - f$**: serial (non-accelerated) fraction

With 45% serial overhead, even infinite inference speedup caps at $1/0.45 = 2.2\\times$ system-level improvement.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B: Thermal Cliff
    # ─────────────────────────────────────────────────────────────────────
    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Edge Deployment Manager
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;The vendor demo showed 30 FPS. Our deployed units in outdoor enclosures
                average 16 FPS after the first hour. The vendor says our hardware is defective.
                Is it?&rdquo;
            </div>
        </div>
        """))
        items.append(mo.md("""
        ## Peak vs. Sustained: The Thermal Cliff

        Vendor benchmarks are **burst** measurements. Production runs are **sustained**.

        ```
        T_junction(t) = T_ambient + (TDP / G_thermal) x (1 - exp(-t / tau))
        ```

        When junction temperature exceeds T_throttle (~85C), the chip reduces clock
        frequency. In a fanless enclosure at 35C ambient, thermal throttling can halve
        sustained throughput within 5 minutes.
        """))
        items.append(pB_pred)
        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the thermal simulator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pB_time, pB_ambient, pB_cooling], justify="start"))

        _t = pB_time.value  # seconds
        _t_amb = pB_ambient.value
        _cooling = pB_cooling.value
        _tdp = 25  # Jetson-class TDP

        # Thermal conductance by cooling type (W/K)
        _g_thermal = {"active": 5.0, "passive": 2.0, "fanless": 0.8}[_cooling]
        _tau = {"active": 120, "passive": 60, "fanless": 30}[_cooling]  # thermal time constant (s)
        _t_throttle = 85  # throttle threshold (C)

        # Piecewise thermal model
        _t_junction = _t_amb + (_tdp / _g_thermal) * (1 - math.exp(-_t / _tau))
        _throttled = _t_junction > _t_throttle
        _peak_fps = 30.0

        if _throttled:
            # Throttle: reduce FPS proportionally to how far over threshold
            _throttle_factor = max(0.3, 1.0 - (_t_junction - _t_throttle) / 50.0)
            _sustained_fps = _peak_fps * _throttle_factor
        else:
            _sustained_fps = _peak_fps

        # Time series
        _times = np.linspace(0, 600, 200)
        _temps = [_t_amb + (_tdp / _g_thermal) * (1 - math.exp(-t / _tau)) for t in _times]
        _fps_series = []
        for _temp in _temps:
            if _temp > _t_throttle:
                _tf = max(0.3, 1.0 - (_temp - _t_throttle) / 50.0)
                _fps_series.append(_peak_fps * _tf)
            else:
                _fps_series.append(_peak_fps)

        from plotly.subplots import make_subplots
        _fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Junction Temperature", "Throughput (FPS)"),
                            vertical_spacing=0.12)

        _fig.add_trace(go.Scatter(x=_times.tolist(), y=_temps, mode="lines",
            line=dict(color=COLORS["OrangeLine"], width=2), name="Temperature", showlegend=False), row=1, col=1)
        _fig.add_hline(y=_t_throttle, line_dash="dash", line_color=COLORS["RedLine"],
                       annotation_text=f"Throttle: {_t_throttle}C", row=1, col=1)
        _fig.add_trace(go.Scatter(x=[_t], y=[_t_junction], mode="markers",
            marker=dict(size=12, color=COLORS["RedLine"], symbol="diamond"),
            name="Current", showlegend=False), row=1, col=1)

        _fig.add_trace(go.Scatter(x=_times.tolist(), y=_fps_series, mode="lines",
            line=dict(color=COLORS["BlueLine"], width=2), name="FPS", showlegend=False), row=2, col=1)
        _fig.add_trace(go.Scatter(x=[_t], y=[_sustained_fps], mode="markers",
            marker=dict(size=12, color=COLORS["RedLine"], symbol="diamond"),
            name="Current", showlegend=False), row=2, col=1)

        _fig.update_layout(height=380, margin=dict(l=50, r=20, t=40, b=40))
        _fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        _fig.update_yaxes(title_text="Temperature (C)", row=1, col=1)
        _fig.update_yaxes(title_text="FPS", row=2, col=1)
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _fps_color = COLORS["RedLine"] if _sustained_fps < 20 else (COLORS["OrangeLine"] if _sustained_fps < 25 else COLORS["GreenLine"])
        _temp_color = COLORS["RedLine"] if _t_junction > _t_throttle else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_temp_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Junction Temp</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_temp_color};">{_t_junction:.0f}C</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{"THROTTLING" if _throttled else "normal"}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_fps_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Sustained FPS</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_fps_color};">{_sustained_fps:.0f} FPS</div>
                <div style="font-size:0.72rem; color:#94a3b8;">vendor claimed 30 FPS</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Performance Loss</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{(1 - _sustained_fps/_peak_fps)*100:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">vs peak burst</div>
            </div>
        </div>
        """))

        items.append(mo.md(f"""
**Thermal Model -- Live Calculation** (`{_cooling}, {_t_amb}C ambient, t={_t}s`)

```
T_junction = {_t_amb} + ({_tdp} / {_g_thermal:.1f}) x (1 - exp(-{_t} / {_tau}))
           = {_t_junction:.1f}C  {"(> 85C = THROTTLING)" if _throttled else "(< 85C = normal)"}
FPS        = {_sustained_fps:.0f} (vendor peak: {_peak_fps:.0f})
```
*Source: Chapter 12, thermal throttling model*
        """))

        if pB_pred.value == "15":
            items.append(mo.callout(mo.md("**Correct.** In a fanless enclosure at 35C, thermal throttling "
                "halves sustained performance after steady-state. Vendor benchmarks are 1-minute burst "
                "measurements that never reach thermal equilibrium."), kind="success"))
        else:
            items.append(mo.callout(mo.md("**Sustained FPS drops dramatically.** "
                "Scrub the time slider forward to 300+ seconds. In fanless enclosures, "
                "thermal throttling kicks in within minutes."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Thermal Throttling Model": mo.md("""
**Formula (junction temperature over time):**
$$
T_j(t) = T_{\\text{amb}} + \\Delta T_{\\max} \\left(1 - e^{-t/\\tau}\\right)
$$

Throttling kicks in when $T_j > T_{\\text{throttle}}$:
$$
\\text{FPS}_{\\text{sustained}} = \\text{FPS}_{\\text{burst}} \\times \\frac{T_{\\text{throttle}} - T_{\\text{amb}}}{\\Delta T_{\\max}}
$$

**Variables:**
- **$T_{\\text{amb}}$**: ambient temperature
- **$\\Delta T_{\\max}$**: steady-state temperature rise at full power
- **$\\tau$**: thermal time constant (seconds, depends on heatsink mass)
- **$T_{\\text{throttle}}$**: junction temperature limit triggering clock reduction

Vendor benchmarks run for $t \\ll \\tau$, never reaching thermal equilibrium. Sustained performance requires $t \\gg \\tau$.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C: Multi-Metric Trap
    # ─────────────────────────────────────────────────────────────────────
    def build_part_c():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Production SRE
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We picked the configuration with the highest accuracy (94%). It passed QA.
                But production shows SLO violations on latency and power. How did 94% accuracy
                not translate to a deployable system?&rdquo;
            </div>
        </div>
        """))
        items.append(mo.md("""
        ## The Multi-Metric Trap

        The configuration with the best single metric (accuracy) violates **other**
        deployment gates (latency, power, throughput). Production deployment requires
        passing **all** SLOs simultaneously.

        | SLO Gate | Threshold |
        |----------|-----------|
        | Accuracy | > 90% |
        | P99 Latency | < 100 ms |
        | Power | < 5 W |
        | Throughput | > 500 QPS |
        """))
        items.append(pC_pred)
        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the multi-metric analyzer."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pC_batch, pC_precision], justify="start"))

        _batch = pC_batch.value
        _prec = pC_precision.value

        # Model configurations
        _bpp = {"fp32": 4, "fp16": 2, "int8": 1}
        _acc_map = {"fp32": 94.0, "fp16": 93.8, "int8": 91.0}
        _acc = _acc_map[_prec]

        # Latency (simplified roofline)
        _model_gb = RESNET50_PARAMS * _bpp[_prec] / 1e9
        _eta = 0.5
        _t_mem = (_model_gb / H100_BW) * 1000  # ms
        _t_comp = (RESNET50_FLOPS * _batch / (H100_TFLOPS * 1e12 * _eta)) * 1000  # ms
        _latency = max(_t_mem, _t_comp) + 0.5  # + overhead
        _p99 = _latency * 3.5  # log-normal tail approximation

        # Throughput
        _qps = _batch / (_latency / 1000) if _latency > 0 else 0

        # Power (scales with batch and precision)
        _base_power = 3.0  # W for edge inference
        _power = _base_power * (1 + 0.3 * math.log2(max(_batch, 1))) * _bpp[_prec] / 2

        # SLO checks
        _slo_acc = _acc > 90
        _slo_lat = _p99 < 100
        _slo_power = _power < 5
        _slo_qps = _qps > 500
        _all_pass = _slo_acc and _slo_lat and _slo_power and _slo_qps

        # Radar chart
        _radar_r = [
            min(_acc / 94 * 100, 100),
            min((100 / _p99) * 100, 100) if _p99 > 0 else 0,
            min((5 / _power) * 100, 100) if _power > 0 else 0,
            min(_qps / 1200 * 100, 100),
        ]
        _radar_r.append(_radar_r[0])  # close the polygon
        _radar_theta = ["Accuracy", "Latency", "Power", "Throughput", "Accuracy"]

        _fig = go.Figure()
        # SLO ring (all at normalized threshold)
        _slo_ring = [90/94*100, 100, 100, 500/1200*100, 90/94*100]
        _fig.add_trace(go.Scatterpolar(
            r=_slo_ring, theta=_radar_theta, mode="lines",
            line=dict(color=COLORS["GreenLine"], width=2, dash="dash"),
            name="SLO Threshold",
        ))
        _fig.add_trace(go.Scatterpolar(
            r=_radar_r, theta=_radar_theta, mode="lines+markers",
            line=dict(color=COLORS["BlueLine"] if _all_pass else COLORS["RedLine"], width=2),
            fill="toself", fillcolor=f"rgba(0,99,149,0.15)" if _all_pass else f"rgba(203,32,45,0.15)",
            name="Your Config",
        ))
        _fig.update_layout(
            height=350,
            polar=dict(radialaxis=dict(range=[0, 110])),
            legend=dict(orientation="h", y=-0.1, x=0),
        )
        items.append(mo.as_html(_fig))

        if not _all_pass:
            _violations = []
            if not _slo_acc:
                _violations.append(f"Accuracy {_acc:.1f}% < 90%")
            if not _slo_lat:
                _violations.append(f"P99 Latency {_p99:.0f} ms > 100 ms")
            if not _slo_power:
                _violations.append(f"Power {_power:.1f} W > 5 W")
            if not _slo_qps:
                _violations.append(f"Throughput {_qps:.0f} QPS < 500 QPS")
            items.append(mo.callout(mo.md(
                f"**DEPLOYMENT BLOCKED.** Violated SLOs: {', '.join(_violations)}."
            ), kind="danger"))

        _pass_badge = lambda ok: f'<span style="color:{COLORS["GreenLine"]}; font-weight:700;">PASS</span>' if ok else f'<span style="color:{COLORS["RedLine"]}; font-weight:700;">FAIL</span>'
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:12px 16px; border:1px solid #e2e8f0; border-radius:10px;
                        background:white; flex:1; text-align:center;">
                <div style="color:#94a3b8; font-size:0.72rem;">Accuracy</div>
                <div style="font-size:1.2rem; font-weight:700;">{_acc:.1f}%</div>
                <div>{_pass_badge(_slo_acc)}</div>
            </div>
            <div style="padding:12px 16px; border:1px solid #e2e8f0; border-radius:10px;
                        background:white; flex:1; text-align:center;">
                <div style="color:#94a3b8; font-size:0.72rem;">P99 Latency</div>
                <div style="font-size:1.2rem; font-weight:700;">{_p99:.0f} ms</div>
                <div>{_pass_badge(_slo_lat)}</div>
            </div>
            <div style="padding:12px 16px; border:1px solid #e2e8f0; border-radius:10px;
                        background:white; flex:1; text-align:center;">
                <div style="color:#94a3b8; font-size:0.72rem;">Power</div>
                <div style="font-size:1.2rem; font-weight:700;">{_power:.1f} W</div>
                <div>{_pass_badge(_slo_power)}</div>
            </div>
            <div style="padding:12px 16px; border:1px solid #e2e8f0; border-radius:10px;
                        background:white; flex:1; text-align:center;">
                <div style="color:#94a3b8; font-size:0.72rem;">Throughput</div>
                <div style="font-size:1.2rem; font-weight:700;">{_qps:.0f} QPS</div>
                <div>{_pass_badge(_slo_qps)}</div>
            </div>
        </div>
        """))

        if pC_pred.value == "no_latency":
            items.append(mo.callout(mo.md("**Correct.** The highest-accuracy configuration violates "
                "the latency SLO. Production deployment requires passing ALL gates simultaneously."), kind="success"))
        else:
            items.append(mo.callout(mo.md("**Accuracy alone is not enough.** Try INT8 precision to see "
                "which configuration passes all four SLO gates simultaneously."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Multi-Metric SLO Gate": mo.md("""
**Formula (deployment feasibility):**
$$
\\text{Deploy} = \\bigwedge_{i=1}^{K} \\left( m_i \\leq \\text{SLO}_i \\right)
$$

A configuration must pass ALL gates simultaneously:
$$
\\text{Accuracy} \\geq A_{\\min} \\;\\wedge\\; \\text{Latency}_{p99} \\leq L_{\\max} \\;\\wedge\\; \\text{Power} \\leq P_{\\max} \\;\\wedge\\; \\text{Memory} \\leq M_{\\max}
$$

**Variables:**
- **$m_i$**: measured metric for gate $i$
- **$\\text{SLO}_i$**: service-level objective threshold for gate $i$
- **$K$**: number of SLO gates (typically 3-5)

The highest-accuracy configuration often violates latency or power constraints. Production deployment is a conjunction, not a maximization.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D: Tail Latency Diagnostic
    # ─────────────────────────────────────────────────────────────────────
    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid #6366f1; background:#f0f4ff;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Reliability Engineer
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our inference service reports 50 ms average latency. Our SLO is 200 ms p99.
                The ops team says we are fine. But users are complaining about slow responses.
                What is the actual p99?&rdquo;
            </div>
        </div>
        """))
        items.append(mo.md("""
        ## Tail Latency: Average Hides Catastrophic Tails

        Real inference latency follows a **log-normal distribution** with a heavy right tail.
        Average latency is actively misleading because it hides the distribution shape.

        ```
        latency_sample = base_latency x exp(N(0, sigma^2))
        p99 = base_latency x exp(2.326 x sigma)
        ```

        At sigma=0.8, p99 is ~6.4x the median. A "50 ms average" system can have a
        500 ms p99 that violates a 200 ms SLO for 1% of requests.
        """))
        items.append(pD_pred)
        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the latency distribution."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pD_sigma, pD_slo], justify="start"))

        _sigma = pD_sigma.value
        _slo = pD_slo.value
        _base = 50.0  # ms base latency (median)

        # Generate log-normal distribution
        np.random.seed(42)
        _n_samples = 10000
        _samples = _base * np.exp(np.random.normal(0, _sigma, _n_samples))

        _mean = float(np.mean(_samples))
        _p50 = float(np.percentile(_samples, 50))
        _p95 = float(np.percentile(_samples, 95))
        _p99 = float(np.percentile(_samples, 99))
        _p999 = float(np.percentile(_samples, 99.9))

        _violation_pct = float(np.sum(_samples > _slo) / len(_samples) * 100)
        _slo_ok = _p99 <= _slo

        _fig = go.Figure()
        _fig.add_trace(go.Histogram(
            x=_samples.tolist(), nbinsx=100,
            marker_color=COLORS["BlueLine"], opacity=0.7,
            name="Latency Distribution",
        ))
        _fig.add_vline(x=_mean, line_dash="dash", line_color=COLORS["BlueLine"],
                       annotation_text=f"Mean: {_mean:.0f} ms")
        _fig.add_vline(x=_p99, line_dash="solid", line_color=COLORS["RedLine"],
                       annotation_text=f"P99: {_p99:.0f} ms")
        _fig.add_vline(x=_slo, line_dash="dot", line_color=COLORS["GreenLine"],
                       annotation_text=f"SLO: {_slo} ms")
        _fig.update_layout(
            height=340,
            xaxis=dict(title="Latency (ms)", range=[0, min(max(_samples) * 0.8, 1000)]),
            yaxis=dict(title="Count"),
            showlegend=False,
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _v_color = COLORS["GreenLine"] if _slo_ok else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Mean</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_mean:.0f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">P99</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_p99:.0f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_p99/_mean:.1f}x the mean</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_v_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">SLO ({_slo} ms)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_v_color};">
                    {"PASS" if _slo_ok else "FAIL"}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_violation_pct:.1f}% violations</div>
            </div>
        </div>
        """))

        if not _slo_ok:
            items.append(mo.callout(mo.md(
                f"**SLO VIOLATED.** P99 = {_p99:.0f} ms exceeds the {_slo} ms threshold. "
                f"{_violation_pct:.1f}% of requests fail. At 10K QPS, that is "
                f"{_violation_pct/100*10000:.0f} failures per second."
            ), kind="danger"))

        items.append(mo.md(f"""
**Tail Latency -- Live Calculation** (`sigma={_sigma:.2f}`)

```
Base latency:  {_base:.0f} ms
Distribution:  log-normal, sigma = {_sigma:.2f}
Mean:          {_mean:.0f} ms
P50:           {_p50:.0f} ms
P95:           {_p95:.0f} ms
P99:           {_p99:.0f} ms  ({_p99/_mean:.1f}x mean)
P99.9:         {_p999:.0f} ms
SLO ({_slo} ms): {"PASS" if _slo_ok else "FAIL"} ({_violation_pct:.1f}% violations)
```
*Source: Chapter 12, latency distributions and tail behavior*
        """))

        if pD_pred.value == "no":
            items.append(mo.callout(mo.md(f"**Correct.** P99 = {_p99:.0f} ms, well above the 200 ms SLO. "
                "Average latency hides the heavy tail. Real inference distributions are log-normal, "
                "where p99 can be 5-10x the mean."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"**Average latency is actively misleading.** "
                f"P99 ({_p99:.0f} ms) is {_p99/_mean:.1f}x the mean ({_mean:.0f} ms). "
                "Adjust sigma to see how tail heaviness affects SLO compliance."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Log-Normal Tail Latency": mo.md("""
**Formula (P99 of log-normal distribution):**
$$
P_{99} = e^{\\mu + 2.326 \\cdot \\sigma}
$$

Ratio of P99 to mean:
$$
\\frac{P_{99}}{\\text{Mean}} = e^{2.326\\sigma - \\sigma^2/2}
$$

**Variables:**
- **$\\mu$**: log-scale location parameter
- **$\\sigma$**: log-scale shape parameter (tail heaviness)
- **2.326**: z-score for 99th percentile of standard normal

At $\\sigma = 0.8$ (typical for inference), $P_{99}/\\text{Mean} \\approx 5\\text{-}10\\times$. A 50 ms mean hides a 250-500 ms P99.
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
                        <strong>1. Component speedup is not system speedup.</strong>
                        Amdahl's Law: a 10x inference speedup with 45% serial overhead yields
                        only 2.0x end-to-end improvement. The bottleneck has already moved.
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>2. Vendor benchmarks are burst, not sustained.</strong>
                        Thermal throttling halves throughput in fanless enclosures within minutes.
                        Always measure at thermal steady-state, not in a 1-minute demo.
                    </div>
                    <div>
                        <strong>3. Average latency hides catastrophic tails.</strong>
                        A 50 ms average can mask a 500 ms p99. Production SLOs require
                        measuring p99/p99.9, not the mean.
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
                        <strong>Lab 13:</strong> The Tail Latency Trap -- the tail you measured
                        in Part D explodes under load. Queuing theory explains why utilization
                        above 70% is dangerous.
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
                        <strong>Read:</strong> the Performance Benchmarking chapter for Amdahl's Law derivation
                        and thermal models.<br/>
                        <strong>Build:</strong> TinyTorch Module 12 -- benchmark harness with
                        percentile tracking.
                    </div>
                </div>
            </div>
            """),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Amdahl Ceiling": build_part_a(),
        "Part B: Thermal Cliff": build_part_b(),
        "Part C: Multi-Metric Trap": build_part_c(),
        "Part D: Tail Latency": build_part_d(),
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
def _(COLORS, ledger, mo, pA_pred, pB_pred, pC_pred, pD_pred):
    if pA_pred.value is not None and pB_pred.value is not None and pC_pred.value is not None and pD_pred.value is not None:
        ledger.save(chapter=12, design={
            "lab": "perf_bench",
            "completed": True,
            "amdahl_speedup_prediction": pA_pred.value,
            "thermal_sustained_fps": pB_pred.value,
            "multi_metric_slo_verdict": pC_pred.value,
            "tail_latency_slo_met": pD_pred.value,
        })
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">12 &middot; Performance Benchmarking</span>
        <span style="flex:1;"></span>
        <span class="hud-label">CH</span>
        <span class="hud-value">12</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">COMPLETE</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
