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
    import mlsysim
    from mlsysim import Engine, Models, Hardware

    # ── Hardware constants ────────────────────────────────────────────────
    H100_TFLOPS = Hardware.Cloud.H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_BW     = Hardware.Cloud.H100.memory.bandwidth.m_as("GB/s")
    H100_RAM    = Hardware.Cloud.H100.memory.capacity.m_as("GB")
    H100_TDP    = Hardware.Cloud.H100.tdp.m_as("W")

    A100_TFLOPS = Hardware.Cloud.A100.compute.peak_flops.m_as("TFLOPs/s")
    A100_BW     = Hardware.Cloud.A100.memory.bandwidth.m_as("GB/s")
    A100_RAM    = Hardware.Cloud.A100.memory.capacity.m_as("GB")

    JETSON_TFLOPS = Hardware.Edge.JetsonOrinNX.compute.peak_flops.m_as("TFLOPs/s")
    JETSON_BW     = Hardware.Edge.JetsonOrinNX.memory.bandwidth.m_as("GB/s")
    JETSON_RAM    = Hardware.Edge.JetsonOrinNX.memory.capacity.m_as("GB")
    JETSON_TDP    = Hardware.Edge.JetsonOrinNX.tdp.m_as("W")

    IPHONE_TFLOPS = Hardware.Mobile.iPhone15Pro.compute.peak_flops.m_as("TFLOPs/s")
    IPHONE_TDP    = Hardware.Mobile.iPhone15Pro.tdp.m_as("W")

    ESP32_TFLOPS  = Hardware.Tiny.ESP32_S3.compute.peak_flops.m_as("TFLOPs/s")
    ESP32_TDP     = Hardware.Tiny.ESP32_S3.tdp.m_as("W")
    ESP32_RAM_KB  = Hardware.Tiny.ESP32_S3.memory.capacity.m_as("KiB")

    # ── Model constants ────────────────────────────────────────────────────
    RESNET50_PARAMS = Models.ResNet50.parameters.m_as("count")
    RESNET50_FLOPS  = Models.ResNet50.inference_flops.m_as("flop")
    RESNET50_SIZE_MB = RESNET50_PARAMS * 2 / (1024 * 1024)  # FP16

    MOBILENET_FLOPS = Models.MobileNetV2.inference_flops.m_as("flop")
    DSCNN_FLOPS = Models.Tiny.DS_CNN.inference_flops.m_as("flop")

    # ── Physical constants ────────────────────────────────────────────────
    SPEED_OF_LIGHT_KM_S = 299_792  # km/s in vacuum
    FIBER_FACTOR = 0.67  # refractive index factor for fiber optic

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return (
        COLORS, LAB_CSS, apply_plotly_theme,
        go, mo, np, math,
        Engine, Models, Hardware,
        H100_TFLOPS, H100_BW, H100_RAM, H100_TDP,
        A100_TFLOPS, A100_BW, A100_RAM,
        JETSON_TFLOPS, JETSON_BW, JETSON_RAM, JETSON_TDP,
        IPHONE_TFLOPS, IPHONE_TDP,
        ESP32_TFLOPS, ESP32_TDP, ESP32_RAM_KB,
        RESNET50_PARAMS, RESNET50_FLOPS, RESNET50_SIZE_MB,
        MOBILENET_FLOPS, DSCNN_FLOPS,
        SPEED_OF_LIGHT_KM_S, FIBER_FACTOR,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 02
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Physics of Deployment
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Memory Wall &middot; Light Barrier &middot; Power Wall &middot; Energy Wall
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Deployment decisions are not engineering preferences -- they are
                physical constraints. The speed of light, the power wall, and the
                energy of data transmission each impose hard limits that no software
                optimization can overcome.
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
                    Chapter 2: ML Systems
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Arithmetic Intensity &amp; Ridge Point</span>
                <span class="badge badge-warn">Speed of Light Floor</span>
                <span class="badge badge-fail">Thermal Throttle at 5W</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, ledger, mo):
    # Load Lab 01 context from Design Ledger
    _lab01 = ledger.get_design(1)
    _lab01_banner = ""
    if _lab01:
        _lab01_banner = f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 10px; padding: 14px 20px; margin: 0 0 12px 0;
                    font-size: 0.82rem; color: {COLORS['TextSec']}; line-height: 1.6;">
            <span style="font-size: 0.68rem; font-weight: 700; color: {COLORS['GreenLine']};
                         text-transform: uppercase; letter-spacing: 0.1em;">
                From Lab 01 &middot; Design Ledger
            </span><br/>
            You established that ResNet-50 at batch=1 on H100 is <strong>{_lab01.get('bottleneck_at_batch1', 'Memory')}-bound</strong>,
            and the H100-to-ESP32 compute gap is <strong>{_lab01.get('compute_ratio_h100_esp32', '~1,000,000x')}</strong>.
            This lab deepens that analysis.
        </div>
        """

    mo.vstack([
        mo.Html(_lab01_banner) if _lab01_banner else mo.md(""),
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
                <div style="margin-bottom: 3px;">1. <strong>Diagnose the Memory Wall</strong> &mdash;
                    show that a 6x GPU upgrade (A100 to H100) yields only ~8% latency improvement
                    for a memory-bound workload (AI = 5 FLOPs/Byte).</div>
                <div style="margin-bottom: 3px;">2. <strong>Calculate the Light Barrier</strong> &mdash;
                    compute the propagation delay floor for a given datacenter distance and determine
                    when cloud inference is physically impossible.</div>
                <div style="margin-bottom: 3px;">3. <strong>Quantify the Power and Energy Walls</strong> &mdash;
                    predict thermal throttling on mobile devices and show that wireless transmission
                    costs ~1,000x more energy than local inference.</div>
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
                    Iron Law from @sec-introduction-iron-law &middot;
                    Arithmetic Intensity from @sec-ml-systems &middot;
                    Deployment tiers from Lab 01
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
                &ldquo;If you double the compute power of your inference server, why doesn't
                latency halve &mdash; and when does the speed of light make cloud inference
                physically impossible?&rdquo;
            </div>
        </div>
    </div>
    """),
    ])
    return


# ─── CELL 3: READING ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **@sec-ml-systems** -- Arithmetic Intensity, ridge point,
      and the memory wall. Pay attention to the A100 vs H100 bandwidth comparison.
    - **@sec-ml-systems-light-barrier** -- The speed of light constraint on cloud
      inference and the autonomous vehicle SLA example.
    - **@sec-ml-systems-deployment-paradigms** -- Power budgets, thermal throttling,
      and why Edge/TinyML exist as deployment paradigms.
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════
# ZONE B-D: ALL PARTS AS TABS
# ═══════════════════════════════════════════════════════════════════════════


# ─── CELL 4: TABS CELL ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    H100_TFLOPS, H100_BW, H100_RAM, H100_TDP,
    A100_TFLOPS, A100_BW, A100_RAM,
    JETSON_TFLOPS, JETSON_BW, JETSON_RAM, JETSON_TDP,
    IPHONE_TFLOPS, IPHONE_TDP,
    ESP32_TFLOPS, ESP32_TDP, ESP32_RAM_KB,
    RESNET50_FLOPS, RESNET50_SIZE_MB,
    MOBILENET_FLOPS, DSCNN_FLOPS,
    SPEED_OF_LIGHT_KM_S, FIBER_FACTOR,
    Engine, Models, Hardware,
    apply_plotly_theme, go, math, mo, np,
):
    # ─────────────────────────────────────────────────────────────────────
    # SHARED WIDGETS
    # ─────────────────────────────────────────────────────────────────────

    # Part A
    partA_prediction = mo.ui.radio(
        options={
            "A) ~6x (proportional to compute increase)":  "6x",
            "B) ~3x (half the compute gain)":              "3x",
            "C) ~1.5x (modest improvement)":               "1.5x",
            "D) <1.1x (almost no improvement)":            "1.1x",
        },
        label="Upgrading from A100 ($15K) to H100 ($30K) -- a 6x compute increase. "
              "For a workload with AI = 5 FLOPs/Byte, what latency improvement?",
    )
    return (partA_prediction,)

@app.cell(hide_code=True)
def _(mo, partA_prediction):
    mo.stop(partA_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partA_ai = mo.ui.slider(
        start=1, stop=400, value=5, step=1,
        label="Arithmetic Intensity (FLOPs/Byte)",
    )

    # Part B
    partB_prediction = mo.ui.radio(
        options={
            "A) Yes -- 1 ms compute leaves 9 ms for network":              "yes_easy",
            "B) Yes -- but barely, with ~1 ms margin":                      "yes_barely",
            "C) No -- propagation delay alone exceeds the SLA":             "no_physics",
            "D) Depends on network congestion":                             "depends",
        },
        label="An AV requires 10 ms end-to-end latency. "
              "Nearest datacenter: 1,500 km. Model runs in 1 ms on cloud GPU. "
              "Is cloud inference feasible?",
    )
    return (partB_prediction,)

@app.cell(hide_code=True)
def _(mo, partB_prediction):
    mo.stop(partB_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partB_distance = mo.ui.slider(
        start=0, stop=5000, value=1500, step=50,
        label="Datacenter distance (km)",
    )
    partB_sla = mo.ui.dropdown(
        options={"10 ms (AV safety)": 10, "50 ms (real-time)": 50, "200 ms (interactive)": 200},
        value="10 ms (AV safety)",
        label="SLA budget:",
    )

    # Part C
    partC_prediction = mo.ui.radio(
        options={
            "A) Still 60 FPS -- hardware is designed for this":  "60fps",
            "B) ~45 FPS -- slight thermal degradation":           "45fps",
            "C) ~15 FPS -- severe thermal throttling":            "15fps",
            "D) 0 FPS -- the phone shuts down":                   "0fps",
        },
        label="ResNet-50 achieves 60 FPS on a mobile NPU. "
              "After 90 seconds of continuous use, what frame rate?",
    )
    return (partC_prediction,)

@app.cell(hide_code=True)
def _(mo, partC_prediction):
    mo.stop(partC_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partC_target = mo.ui.dropdown(
        options={
            "Cloud (300W TDP)": "cloud",
            "Edge (25W TDP)": "edge",
            "Mobile (5W TDP)": "mobile",
            "TinyML (1.2W TDP)": "tiny",
        },
        value="Mobile (5W TDP)",
        label="Deployment target:",
    )
    partC_model = mo.ui.dropdown(
        options={
            "ResNet-50 (4.1 GFLOPs)": "resnet",
            "MobileNetV2 (0.3 GFLOPs)": "mobilenet",
            "DS-CNN (20 MFLOPs)": "dscnn",
        },
        value="ResNet-50 (4.1 GFLOPs)",
        label="Model:",
    )

    # Part D
    partD_prediction = mo.ui.radio(
        options={
            "A) ~2x (cloud is slightly more expensive)":  "2x",
            "B) ~10x":                                     "10x",
            "C) ~100x":                                    "100x",
            "D) ~1,000x":                                  "1000x",
        },
        label="A sensor captures 16 KB of audio. "
              "Energy ratio of cloud transmission vs. local inference?",
    )
    return (partD_prediction,)

@app.cell(hide_code=True)
def _(mo, partD_prediction):
    mo.stop(partD_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partD_data_size = mo.ui.slider(
        start=1, stop=1024, value=16, step=1,
        label="Data size (KB)",
    )
    partD_wireless = mo.ui.dropdown(
        options={"BLE (1 Mbps)": 1, "WiFi (50 Mbps)": 50, "LTE (10 Mbps)": 10},
        value="BLE (1 Mbps)",
        label="Wireless technology:",
    )

    # ─────────────────────────────────────────────────────────────────────
    # PART A -- The Memory Wall Revelation
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; CFO, CloudScale AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We are considering upgrading our inference fleet from A100s ($15K each)
                to H100s ($30K each). The H100 has 6x the compute throughput. Engineering
                says we should see proportional latency reduction. Finance wants your analysis
                before approving $2M in hardware spend.&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Rachel Kim, CFO &middot; CloudScale AI
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Memory Wall: When Faster Compute Changes Nothing

        **Arithmetic Intensity** (AI) = FLOPs / Bytes moved. This ratio determines
        whether a workload is compute-bound or memory-bound.

        ```
        Ridge Point = R_peak / BW
        ```

        Below the Ridge Point, the memory term dominates and compute upgrades are wasted.
        Above it, compute upgrades yield proportional improvements.
        """))

        items.append(partA_prediction)

        if partA_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the Memory Wall simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(partA_ai)

        _ai = partA_ai.value
        _eta = 0.5

        # Ridge points
        _ridge_a100 = A100_TFLOPS * 1000 / A100_BW
        _ridge_h100 = H100_TFLOPS * 1000 / H100_BW

        # Latency for generic workload: 1 GB data, FLOPs = AI * 1e9
        _data_bytes = 1e9
        _flops = _ai * _data_bytes

        _t_mem_a100 = (_data_bytes / (A100_BW * 1e9)) * 1000
        _t_comp_a100 = (_flops / (A100_TFLOPS * 1e12 * _eta)) * 1000
        _t_total_a100 = max(_t_mem_a100, _t_comp_a100) + 0.015

        _t_mem_h100 = (_data_bytes / (H100_BW * 1e9)) * 1000
        _t_comp_h100 = (_flops / (H100_TFLOPS * 1e12 * _eta)) * 1000
        _t_total_h100 = max(_t_mem_h100, _t_comp_h100) + 0.01

        _speedup = _t_total_a100 / _t_total_h100 if _t_total_h100 > 0 else 1
        _improvement_pct = (_speedup - 1) * 100

        _bound_a100 = "Memory-bound" if _ai < _ridge_a100 else "Compute-bound"
        _bound_h100 = "Memory-bound" if _ai < _ridge_h100 else "Compute-bound"

        # Speedup curve
        _ai_range = np.arange(1, 401)
        _speedups = []
        for _a in _ai_range:
            _f = _a * _data_bytes
            _ta = max(_data_bytes / (A100_BW * 1e9), _f / (A100_TFLOPS * 1e12 * _eta)) * 1000 + 0.015
            _th = max(_data_bytes / (H100_BW * 1e9), _f / (H100_TFLOPS * 1e12 * _eta)) * 1000 + 0.01
            _speedups.append(_ta / _th if _th > 0 else 1)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_ai_range.tolist(), y=_speedups, mode="lines",
            name="A100 to H100 Speedup",
            line=dict(color=COLORS["BlueLine"], width=2.5),
        ))
        _fig.add_hline(y=6, line_dash="dash", line_color=COLORS["GreenLine"], line_width=1,
                       annotation_text="Ideal 6x", annotation_position="top right",
                       annotation_font_color=COLORS["GreenLine"])
        _fig.add_hline(y=1, line_dash="dot", line_color=COLORS["TextMuted"], line_width=1)
        _fig.add_vline(x=_ridge_a100, line_dash="dash", line_color=COLORS["OrangeLine"], line_width=1.5,
                       annotation_text=f"A100 Ridge ({_ridge_a100:.0f})",
                       annotation_position="top left",
                       annotation_font_color=COLORS["OrangeLine"])
        _fig.add_vline(x=_ridge_h100, line_dash="dash", line_color=COLORS["RedLine"], line_width=1.5,
                       annotation_text=f"H100 Ridge ({_ridge_h100:.0f})",
                       annotation_position="top right",
                       annotation_font_color=COLORS["RedLine"])
        _fig.add_trace(go.Scatter(
            x=[_ai], y=[_speedup], mode="markers",
            name=f"AI={_ai}: {_speedup:.2f}x",
            marker=dict(color=COLORS["RedLine"], size=12, symbol="circle",
                        line=dict(color="white", width=2)),
        ))
        _fig.update_layout(
            height=340,
            xaxis=dict(title="Arithmetic Intensity (FLOPs/Byte)", gridcolor="#f1f5f9"),
            yaxis=dict(title="Speedup (A100 to H100)", range=[0.8, 7], gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md("### GPU Upgrade Speedup vs Arithmetic Intensity"))
        items.append(mo.as_html(_fig))

        _sp_color = COLORS["GreenLine"] if _speedup > 3 else COLORS["OrangeLine"] if _speedup > 1.5 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {_sp_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.75rem; font-weight:600;">Speedup</div>
                <div style="font-size:1.8rem; font-weight:800; color:{_sp_color};">
                    {_speedup:.2f}x</div>
                <div style="font-size:0.7rem; color:#94a3b8;">A100 -> H100</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.75rem; font-weight:600;">AI (current)</div>
                <div style="font-size:1.8rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_ai} F/B</div>
                <div style="font-size:0.7rem; color:#94a3b8;">{_bound_h100}</div>
            </div>
            <div style="padding:16px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.75rem; font-weight:600;">Budget Wasted</div>
                <div style="font-size:1.8rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    ${2_000_000 * max(0, 1 - _improvement_pct/500):,.0f}</div>
                <div style="font-size:0.7rem; color:#94a3b8;">of $2M spend</div>
            </div>
        </div>
        """))

        items.append(mo.md(f"""
**Memory Wall -- Live Calculation** (`AI = {_ai} FLOPs/Byte`)

```
A100:  max(mem={_t_mem_a100:.4f}ms, comp={_t_comp_a100:.4f}ms) = {_t_total_a100:.4f} ms  [{_bound_a100}]
H100:  max(mem={_t_mem_h100:.4f}ms, comp={_t_comp_h100:.4f}ms) = {_t_total_h100:.4f} ms  [{_bound_h100}]
Speedup = {_speedup:.2f}x   Improvement = {_improvement_pct:.1f}%
```

At AI=5: BW ratio is {H100_BW/A100_BW:.2f}x ({H100_BW:,.0f} vs {A100_BW:,.0f} GB/s).
The 6x compute upgrade is almost entirely wasted.
"""))

        _pred = partA_prediction.value
        if _pred == "1.1x":
            _rev = ("**Correct.** At AI=5, both GPUs are deeply memory-bound. "
                    f"The speedup is only {_speedup:.2f}x because bandwidth, not compute, "
                    "is the binding constraint.")
            _rkind = "success"
        else:
            _rev = (f"**The actual speedup at AI=5 is only ~{_speedup:.2f}x.** "
                    "The workload is memory-bound. Slide AI above the ridge point "
                    "to see where the 6x upgrade actually pays off.")
            _rkind = "warn"

        items.append(mo.callout(mo.md(_rev), kind=_rkind))

        items.append(mo.accordion({
            "Math Peek: The Roofline Model": mo.md(f"""
$$
\\text{{Latency}} = \\max\\left(\\frac{{D}}{{BW}},\\; \\frac{{O}}{{R_{{\\text{{peak}}}} \\cdot \\eta}}\\right) + L
$$

Ridge Point: A100 = {_ridge_a100:.0f} F/B, H100 = {_ridge_h100:.0f} F/B.
Below the ridge point, latency $\\approx D/BW$ and compute upgrades have near-zero effect.
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B -- The Light Barrier
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; VP Autonomy, DriveAI (safety-critical)
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our AV obstacle detection requires 10 ms end-to-end latency. Our cloud
                server runs the model in 1 ms. The nearest datacenter is 1,500 km away.
                Can we use cloud inference? The on-vehicle module costs $800 per unit
                across 50,000 vehicles.&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; James Chen, VP Autonomy &middot; DriveAI
            </div>
        </div>
        """))

        _speed_fiber = SPEED_OF_LIGHT_KM_S * FIBER_FACTOR
        items.append(mo.md(f"""
        ## The Speed of Light Sets an Irreducible Latency Floor

        Light in fiber: ~{_speed_fiber:,.0f} km/s. At 1,500 km round-trip:

        ```
        t_prop = 2 * 1500 / {_speed_fiber:,.0f} = {2*1500/_speed_fiber*1000:.1f} ms
        ```

        This is physics, not engineering. No optimization can make photons faster.
        """))

        items.append(partB_prediction)

        if partB_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the Light Barrier simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partB_distance, partB_sla], justify="start", gap="2rem"))

        _dist = partB_distance.value
        _sla = partB_sla.value
        _compute_ms = 1.0
        _serial_ms = 0.5
        _queue_ms = 1.0

        _prop_ms = (2 * _dist / _speed_fiber) * 1000 if _dist > 0 else 0
        _total_ms = _prop_ms + _compute_ms + _serial_ms + _queue_ms
        _sla_violated = _total_ms > _sla
        _max_dist = max(0, (_sla - _compute_ms - _serial_ms - _queue_ms) / 2 * _speed_fiber / 1000)

        # Stacked bar
        _components = ["Propagation", "Compute", "Serialization", "Queueing"]
        _values = [_prop_ms, _compute_ms, _serial_ms, _queue_ms]
        _comp_colors = [COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"], COLORS["Grey"]]

        _fig = go.Figure()
        _cumul = 0
        for _nm, _vl, _cc in zip(_components, _values, _comp_colors):
            _fig.add_trace(go.Bar(
                name=_nm, x=[_vl], y=["Latency"], orientation="h",
                base=_cumul, marker_color=_cc, opacity=0.85,
                text=[f"{_vl:.1f} ms"], textposition="inside",
                textfont=dict(color="white", size=11),
            ))
            _cumul += _vl

        _fig.add_vline(x=_sla, line_dash="dash", line_color=COLORS["RedLine"], line_width=2,
                       annotation_text=f"SLA: {_sla} ms",
                       annotation_font_color=COLORS["RedLine"])
        _fig.update_layout(
            barmode="stack", height=140,
            xaxis=dict(title="End-to-End Latency (ms)",
                       range=[0, max(_total_ms * 1.3, _sla * 1.3)],
                       gridcolor="#f1f5f9"),
            yaxis=dict(visible=False),
            legend=dict(orientation="h", y=1.3, x=0),
            margin=dict(l=20, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        if _sla_violated:
            items.append(mo.callout(mo.md(
                f"**SLA VIOLATED -- the speed of light cannot be optimized.** "
                f"Total: {_total_ms:.1f} ms > {_sla} ms. "
                f"Propagation alone: {_prop_ms:.1f} ms. "
                f"Max feasible distance: ~{_max_dist:.0f} km."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"**SLA met with {_sla - _total_ms:.1f} ms margin.** "
                f"Total: {_total_ms:.1f} ms."
            ), kind="success"))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['BlueLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Propagation</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_prop_ms:.1f} ms</div>
                <div style="font-size:0.68rem; color:#94a3b8;">irreducible</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {'#CB202D' if _sla_violated else '#008F45'};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Total</div>
                <div style="font-size:1.4rem; font-weight:800;
                            color:{'#CB202D' if _sla_violated else '#008F45'};">
                    {_total_ms:.1f} ms</div>
                <div style="font-size:0.68rem; color:#94a3b8;">
                    {'EXCEEDS SLA' if _sla_violated else 'within SLA'}</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['OrangeLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Max Distance</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {_max_dist:,.0f} km</div>
                <div style="font-size:0.68rem; color:#94a3b8;">for {_sla} ms SLA</div>
            </div>
        </div>
        """))

        # Distance sweep
        _dists = np.linspace(0, 5000, 200)
        _lats = [(2 * d / _speed_fiber) * 1000 + _compute_ms + _serial_ms + _queue_ms for d in _dists]

        _fig2 = go.Figure()
        _fig2.add_trace(go.Scatter(x=_dists.tolist(), y=_lats, mode="lines",
                                   name="Total Latency",
                                   line=dict(color=COLORS["BlueLine"], width=2.5)))
        for _s, _sc, _sl in [(10, COLORS["RedLine"], "10 ms AV"),
                              (50, COLORS["OrangeLine"], "50 ms real-time"),
                              (200, COLORS["GreenLine"], "200 ms interactive")]:
            _fig2.add_hline(y=_s, line_dash="dash", line_color=_sc, line_width=1,
                            annotation_text=_sl, annotation_font_color=_sc)
        _fig2.update_layout(
            height=300,
            xaxis=dict(title="Datacenter Distance (km)", gridcolor="#f1f5f9"),
            yaxis=dict(title="End-to-End Latency (ms)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=50, b=40),
        )
        apply_plotly_theme(_fig2)
        items.append(mo.md("### Latency vs Distance"))
        items.append(mo.as_html(_fig2))

        _pred = partB_prediction.value
        _ref_prop = (2 * 1500 / _speed_fiber) * 1000
        if _pred == "no_physics":
            _rev = (f"**Correct.** At 1,500 km, propagation alone is {_ref_prop:.1f} ms "
                    "-- exceeding the 10 ms SLA. This is why Edge ML exists: physics.")
            _rkind = "success"
        else:
            _rev = (f"**Propagation at 1,500 km is {_ref_prop:.1f} ms -- already "
                    "over the 10 ms SLA.** The speed of light is the constraint.")
            _rkind = "warn"
        items.append(mo.callout(mo.md(_rev), kind=_rkind))

        items.append(mo.accordion({
            "Math Peek: Propagation Delay": mo.md(f"""
$$t_{{\\text{{prop}}}} = \\frac{{2d}}{{c \\cdot n}} = \\frac{{2 \\times 1500}}{{{SPEED_OF_LIGHT_KM_S:,} \\times {FIBER_FACTOR}}} = {_ref_prop:.1f}\\text{{ ms}}$$
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C -- The Power Wall
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Head of Mobile ML, PhoneVision
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our camera filter runs ResNet-50 at 60 FPS on the A17 Pro.
                Marketing wants 'always-on 60 FPS.' QA says the phone gets hot after
                30 seconds. Can we ship this?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Lisa Wang &middot; PhoneVision
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Power Wall: Thermal Throttling

        Mobile devices at 5W hit thermal limits within seconds under sustained load.

        ```
        if power_sustained <= TDP:
            performance = peak
        else:
            performance = peak * (TDP / power_sustained)
            performance = max(performance, 0.25 * peak)
        ```
        """))

        items.append(partC_prediction)

        if partC_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the Power Wall simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partC_target, partC_model], justify="start", gap="2rem"))

        _target_specs = {
            "cloud":  {"name": "Cloud GPU",  "tdp": H100_TDP, "tflops": H100_TFLOPS, "peak_fps": 250},
            "edge":   {"name": "Jetson Edge", "tdp": JETSON_TDP, "tflops": JETSON_TFLOPS, "peak_fps": 120},
            "mobile": {"name": "Mobile NPU", "tdp": IPHONE_TDP, "tflops": IPHONE_TFLOPS, "peak_fps": 60},
            "tiny":   {"name": "ESP32 MCU",  "tdp": ESP32_TDP, "tflops": ESP32_TFLOPS, "peak_fps": 0.1},
        }
        _model_flops = {"resnet": RESNET50_FLOPS, "mobilenet": MOBILENET_FLOPS, "dscnn": DSCNN_FLOPS}

        _ts = _target_specs[partC_target.value]
        _mf = _model_flops[partC_model.value]

        _inference_ms = (_mf / (_ts["tflops"] * 1e12 * 0.5)) * 1000
        _peak_fps = min(1000 / _inference_ms, _ts["peak_fps"]) if _inference_ms > 0 else 0

        # Thermal curve
        _times = np.linspace(0, 120, 240)
        _fps_curve = []
        _thermal_onset = 30

        for _t in _times:
            if _t < _thermal_onset:
                _fps_curve.append(_peak_fps)
            else:
                _factor = max(0.25, 1.0 - (_t - _thermal_onset) / 120)
                _fps_curve.append(_peak_fps * _factor)

        _fps_90 = _fps_curve[min(179, len(_fps_curve) - 1)]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_times.tolist(), y=_fps_curve, mode="lines", name="FPS",
            line=dict(color=COLORS["BlueLine"], width=2.5),
            fill="tozeroy", fillcolor="rgba(0,99,149,0.1)",
        ))
        _fig.add_vline(x=30, line_dash="dash", line_color=COLORS["OrangeLine"], line_width=1.5,
                       annotation_text="Throttle onset",
                       annotation_font_color=COLORS["OrangeLine"])
        _fig.add_hline(y=_peak_fps * 0.25, line_dash="dot", line_color=COLORS["RedLine"], line_width=1,
                       annotation_text=f"Floor: {_peak_fps*0.25:.0f} FPS",
                       annotation_font_color=COLORS["RedLine"])
        _fig.update_layout(
            height=300,
            xaxis=dict(title="Time (seconds)", gridcolor="#f1f5f9"),
            yaxis=dict(title="Frame Rate (FPS)", range=[0, _peak_fps * 1.15], gridcolor="#f1f5f9"),
            margin=dict(l=50, r=20, t=30, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md(f"### Sustained Performance: {_ts['name']}"))
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['BlueLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Peak FPS</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_peak_fps:.0f}</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['OrangeLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">FPS at 90s</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {_fps_90:.0f}</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['RedLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">TDP</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['RedLine']};">
                    {_ts['tdp']:.0f} W</div>
            </div>
        </div>
        """))

        _pred = partC_prediction.value
        if _pred == "15fps":
            items.append(mo.callout(mo.md(
                "**Correct.** Thermal throttling drops to ~25% of peak after ~30 seconds."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Mobile hardware cannot sustain peak performance.** "
                "After 30 seconds, throttling drops FPS to ~25% of peak."
            ), kind="warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D -- The Energy of Transmission
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Lead Engineer, WildWatch
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our wildlife sensors run on coin cell batteries (250 mAh). Each sensor
                captures 16 KB of audio every 10 seconds. Should we transmit to the cloud
                or run KWS locally on the ESP32?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Dr. Sara Obi &middot; WildWatch Conservation
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Energy Wall: Transmission Costs ~1,000x More Than Local Inference

        Radio power amplifiers are energy-hungry. Even if cloud inference were free
        and instantaneous, the energy cost of wireless transmission makes cloud
        offloading impossible for always-on battery-powered sensing.
        """))

        items.append(partD_prediction)

        if partD_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the Energy comparison."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partD_data_size, partD_wireless], justify="start", gap="2rem"))

        _data_kb = partD_data_size.value
        _wireless_mbps = partD_wireless.value

        _radio_power_mw = 100
        _data_bits = _data_kb * 1024 * 8
        _tx_time_ms = (_data_bits / (_wireless_mbps * 1e6)) * 1000
        _tx_energy_mj = _radio_power_mw * _tx_time_ms / 1000

        _local_energy_mj = 0.01  # DS-CNN on ESP32

        _energy_ratio = _tx_energy_mj / _local_energy_mj if _local_energy_mj > 0 else float('inf')

        _battery_mj = 250 * 3.0 * 3600  # 250mAh * 3V -> mWh -> mJ
        _inferences_per_day = 24 * 3600 / 10
        _cloud_days = _battery_mj / (_tx_energy_mj * _inferences_per_day) if _tx_energy_mj > 0 else 9999
        _local_days = _battery_mj / (_local_energy_mj * _inferences_per_day)

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=["Cloud (Transmit)", "Local (Inference)"],
            y=[_tx_energy_mj, _local_energy_mj],
            marker_color=[COLORS["RedLine"], COLORS["GreenLine"]],
            text=[f"{_tx_energy_mj:.2f} mJ", f"{_local_energy_mj:.4f} mJ"],
            textposition="outside",
        ))
        _fig.update_layout(
            height=300,
            yaxis=dict(title="Energy per Classification (mJ)", type="log", gridcolor="#f1f5f9"),
            margin=dict(l=50, r=20, t=30, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md("### Energy: Cloud vs Local"))
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:14px; border:2px solid {COLORS['RedLine']}; border-radius:10px;
                        text-align:center; background:{COLORS['RedLL']}; flex:1;">
                <div style="color:{COLORS['RedLine']}; font-size:0.72rem; font-weight:700;">
                    Cloud Energy</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['RedLine']};">
                    {_tx_energy_mj:.2f} mJ</div>
            </div>
            <div style="padding:14px; border:2px solid {COLORS['GreenLine']}; border-radius:10px;
                        text-align:center; background:{COLORS['GreenLL']}; flex:1;">
                <div style="color:{COLORS['GreenLine']}; font-size:0.72rem; font-weight:700;">
                    Local Energy</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['GreenLine']};">
                    {_local_energy_mj:.4f} mJ</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['OrangeLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Ratio</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    ~{_energy_ratio:,.0f}x</div>
            </div>
        </div>
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:0 0 16px 0;">
            <div style="padding:14px; border:2px solid {COLORS['RedLine']}; border-radius:10px;
                        text-align:center; background:{COLORS['RedLL']}; flex:1;">
                <div style="color:{COLORS['RedLine']}; font-size:0.72rem; font-weight:700;">
                    Cloud Battery Life</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['RedLine']};">
                    {_cloud_days:.1f} days</div>
            </div>
            <div style="padding:14px; border:2px solid {COLORS['GreenLine']}; border-radius:10px;
                        text-align:center; background:{COLORS['GreenLL']}; flex:1;">
                <div style="color:{COLORS['GreenLine']}; font-size:0.72rem; font-weight:700;">
                    Local Battery Life</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['GreenLine']};">
                    {min(_local_days, 9999):.0f} days</div>
            </div>
        </div>
        """))

        _pred = partD_prediction.value
        if _pred == "1000x":
            items.append(mo.callout(mo.md(
                f"**Correct.** Energy ratio is ~{_energy_ratio:,.0f}x. "
                "This is why TinyML exists."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**The ratio is ~{_energy_ratio:,.0f}x.** "
                f"Cloud battery: {_cloud_days:.1f} days. Local: {min(_local_days,9999):.0f} days."
            ), kind="warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        _ref_prop = (2 * 1500 / (SPEED_OF_LIGHT_KM_S * FIBER_FACTOR)) * 1000
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
                        <strong>1. The Memory Wall makes compute upgrades worthless for
                        memory-bound workloads.</strong> At AI=5, a 6x GPU upgrade yields
                        only ~8% improvement.
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>2. The speed of light is irreducible.</strong>
                        At 1,500 km, propagation takes {_ref_prop:.1f} ms -- exceeding a
                        10 ms AV SLA. This is why Edge ML exists.
                    </div>
                    <div>
                        <strong>3. Energy physics determines where computation must happen.</strong>
                        Wireless transmission costs ~1,000x more than local inference.
                    </div>
                </div>
            </div>
            """),

            mo.Html(f"""
            <div style="display: flex; gap: 16px; margin: 8px 0; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        What's Next
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        <strong>Lab 03: The Constraint Tax</strong> -- Shows what happens when
                        teams discover these physical walls late in the ML lifecycle.
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
                        <strong>Read:</strong> @sec-ml-systems for the Roofline model,
                        light barrier, and power wall derivations.
                    </div>
                </div>
            </div>
            """),
        ])

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A -- The Memory Wall":          build_part_a(),
        "Part B -- The Light Barrier":        build_part_b(),
        "Part C -- The Power Wall":           build_part_c(),
        "Part D -- The Energy Wall":          build_part_d(),
        "Synthesis":                           build_synthesis(),
    })
    tabs
    return


# ═══════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(COLORS, ledger, mo):
    _track = ledger._state.track or "not set"
    ledger.save(chapter=2, design={
        "chapter": "v1_02",
        "completed": True,
    })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">02 &middot; The Physics of Deployment</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">CHAPTER&nbsp;2</span>
        <span class="hud-value">ML Systems</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
