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
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        fusion_traffic,
        gemm_workload,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        hardware_roofline_profile,
        part_workflow,
        report_export_panel,
        resolve_mlsysim_ref,
        roofline_point,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, apply_plotly_theme,
        build_lab_report, fusion_traffic, gemm_workload,
        get_lab_metadata, get_lab_track_variant, get_track_profile, go,
        hardware_roofline_profile, ledger, math, mo, np, part_workflow, report_export_panel,
        resolve_mlsysim_ref, roofline_point, source_trace, track_context,
        track_arc_context, track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_11_metadata = get_lab_metadata("vol1/lab_11_hw_accel.py")
    return (v1_11_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_11_track_picker = track_selector(default=_default_track)
    v1_11_track_picker
    return (v1_11_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    hardware_roofline_profile,
    resolve_mlsysim_ref,
    v1_11_track_picker,
):
    v1_11_track_id = v1_11_track_picker.value
    v1_11_profile = get_track_profile(v1_11_track_id)
    v1_11_variant = get_lab_track_variant("v1_11_hardware_roofline", v1_11_track_id)
    v1_11_hardware = resolve_mlsysim_ref(v1_11_variant.hardware_ref)
    v1_11_roofline = hardware_roofline_profile(v1_11_profile, v1_11_variant, v1_11_hardware)

    _compare_ids = tuple(v1_11_variant.defaults.get("compare_tracks", (v1_11_track_id,)))
    _profiles = []
    for _track_id in dict.fromkeys((v1_11_track_id, *_compare_ids)):
        _profile = get_track_profile(_track_id)
        _variant = get_lab_track_variant("v1_11_hardware_roofline", _profile.track_id)
        _hardware = resolve_mlsysim_ref(_variant.hardware_ref)
        _profiles.append(hardware_roofline_profile(_profile, _variant, _hardware))
    v1_11_comparison_profiles = tuple(_profiles)
    return (
        v1_11_comparison_profiles,
        v1_11_hardware,
        v1_11_profile,
        v1_11_roofline,
        v1_11_track_id,
        v1_11_variant,
    )

# ═════════════════════════════════════════════════════════════════════════════
# CELL 1: HEADER
# ═════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
        track_arc_context,
    v1_11_metadata,
    v1_11_profile,
    v1_11_roofline,
    v1_11_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 11
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The Hardware Roofline
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Roofline &middot; Kernel Fusion &middot; Balance Shift &middot; Energy &middot; Tiling
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                {v1_11_variant.workload_summary} Your code is not broken &mdash; the selected
                hardware has a ceiling, and that ceiling changes shape depending on arithmetic
                intensity, memory bandwidth, precision, and fusion.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    5 Parts &middot; ~52 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_11_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_11_roofline.hardware_ref}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Roofline Model</span>
                <span class="badge badge-warn">{v1_11_roofline.ridge_flop_per_byte:.1f} FLOP/B ridge</span>
                <span class="badge badge-fail">{v1_11_roofline.bandwidth_gbs:g} GB/s bandwidth</span>
            </div>
        </div>
        """),
        track_context(v1_11_profile),
        track_arc_context(v1_11_profile, v1_11_metadata.lab_id),
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
                <div style="margin-bottom: 3px;">1. <strong>Diagnose the Roofline regime</strong> &mdash;
                    determine whether a General Matrix Multiply (GEMM) kernel is memory-bound or compute-bound by comparing
                    arithmetic intensity to the hardware ridge point.</div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify kernel fusion speedup</strong> &mdash;
                    fusing LayerNorm + Dropout + ReLU eliminates 2 HBM round-trips, yielding
                    3-5x speedup for memory-bound elementwise sequences.</div>
                <div style="margin-bottom: 3px;">3. <strong>Predict the balance shift</strong> &mdash;
                    the same operation changes regime when moving from edge (low ridge) to cloud
                    (high ridge) because compute grows faster than bandwidth.</div>
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
                    Iron Law latency decomposition from the ML Systems chapter &middot;
                    Memory hierarchy from the Neural Computation chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~52 min</strong><br/>
                    A: ~12 &middot; B: ~12 &middot; C: ~10 &middot; D: ~10 &middot; E: ~8 min
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
                &ldquo;A GEMM kernel achieves only 31% of peak TFLOPS. Is the code broken,
                or is the hardware ceiling lower than you think?&rdquo;
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
def _(mo, v1_11_comparison_profiles):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **Chapter 11: Hardware Acceleration** -- Roofline model, arithmetic intensity,
      ridge point, kernel fusion, and tiling strategies.
    - **Chapter 2: ML Systems** -- Iron Law latency decomposition.
    """), kind="info")
    return

# ═════════════════════════════════════════════════════════════════════════════
# CELL 4: TABS (Parts A-E + Synthesis)
# ═════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(mo, v1_11_roofline, v1_11_variant):
    # ── Widgets ───────────────────────────────────────────────────────────
    pA_pred = mo.ui.radio(
        options={
            "A) Bug -- should be 90%+": "bug",
            "B) Thermal throttling": "thermal",
            "C) Memory-bandwidth-bound -- hitting the Roofline correctly": "roofline",
            "D) FP16 reduces peak throughput": "fp16",
        },
        label=(
            f"GEMM on {v1_11_roofline.label} is far below peak for "
            f"{v1_11_variant.primary_metric}. What is the most likely explanation?"
        ),
    )
    return (pA_pred,)

@app.cell(hide_code=True)
def _(mo, v1_11_variant):
    _default_dim = int(v1_11_variant.defaults.get("matrix_dim", 512))
    _default_precision = str(v1_11_variant.defaults.get("precision", "fp16")).upper()
    pA_dim = mo.ui.slider(start=128, stop=8192, value=_default_dim, step=128, label="Matrix dimension N")
    pA_prec = mo.ui.radio(
        options={"FP32": "fp32", "FP16": "fp16", "INT8": "int8"},
        value=_default_precision, label="Precision:", inline=True,
    )

    pB_pred = mo.ui.radio(
        options={
            "A) ~1.3x (modest)": "1_3",
            "B) ~2x (half the work)": "2x",
            "C) ~3-5x (eliminated memory traffic)": "3_5",
            "D) ~10x (nearly all overhead)": "10x",
        },
        label="Fuse LayerNorm + Dropout + ReLU into one kernel, eliminating 2 HBM writes. Speedup?",
    )
    return (pA_dim, pA_prec, pB_pred)

@app.cell(hide_code=True)
def _(mo):
    pB_mode = mo.ui.radio(
        options={"Eager (separate kernels)": "eager", "Fused (single kernel)": "fused"},
        value="Eager (separate kernels)", label="Execution mode:", inline=True,
    )
    pB_batch = mo.ui.slider(start=1, stop=128, value=1, step=1, label="Batch size")

    _labels = " and ".join(profile.label for profile in v1_11_comparison_profiles)
    pC_pred = mo.ui.radio(
        options={
            "A) Yes -- compute-bound on weak hw means compute-bound on strong": "yes",
            "B) No -- ridge point can shift the regime": "no",
            "C) Depends on precision": "depends",
            "D) Memory-bound on both": "both",
        },
        label=f"GEMM at N=1024 is evaluated on {_labels}. Must it stay in the same regime?",
    )
    return (pB_batch, pB_mode, pC_pred)

@app.cell(hide_code=True)
def _(mo, v1_11_comparison_profiles, v1_11_roofline):
    _options = {profile.label: profile.track_id for profile in v1_11_comparison_profiles}
    pC_hw = mo.ui.radio(options=_options, value=v1_11_roofline.label, label="Hardware:", inline=True)

    pD_pred = mo.ui.radio(
        options={
            "A) Memory-bound uses ~10x more energy": "10x",
            "B) Same energy (same FLOPs = same work)": "same",
            "C) Compute-bound uses more (higher utilization)": "compute_more",
            "D) Depends on clock frequency": "clock",
        },
        label="Memory-bound (AI=10) and compute-bound (AI=500) do the same FLOPs. Which uses more energy?",
    )
    return (pC_hw, pD_pred)

@app.cell(hide_code=True)
def _(mo):
    pE_pred = mo.ui.radio(
        options={
            "A) ~1.2x (minor)": "1_2",
            "B) ~1.5x (moderate)": "1_5",
            "C) ~2-4x (significant)": "2_4",
            "D) ~10x (transformative)": "10x",
        },
        label="FlashAttention tiles attention to fit in SRAM. Speedup at seq_len=4096?",
    )
    return (pE_pred,)

# ─── widget cell: extracted from tabs cell body (#1332 polish) ────
@app.cell(hide_code=True)
def _(mo):
    pE_tile = mo.ui.slider(start=32, stop=2048, value=256, step=32, label="Tile size (elements)")
    pE_seq = mo.ui.slider(start=512, stop=16384, value=4096, step=512, label="Sequence length")
    return (pE_seq, pE_tile)

@app.cell(hide_code=True)
def _(
    fusion_traffic, gemm_workload, mo, part_workflow, pA_dim, pA_prec, pA_pred,
    pB_batch, pB_mode, pB_pred, pC_hw,
    pC_pred, pD_pred, pE_pred, pE_seq,
    pE_tile, roofline_point, v1_11_comparison_profiles,
    v1_11_profile, v1_11_roofline, v1_11_variant,
):

    # ── Helper: draw roofline ─────────────────────────────────────────────
    def _draw_roofline(fig, peak_tflops, bw_gbs, ridge, color, name):
        _ais = np.logspace(-1, 4, 200)
        _bw_ceil = [bw_gbs * ai for ai in _ais]  # GFLOP/s = GB/s * FLOP/byte
        _comp_ceil = [peak_tflops * 1000] * len(_ais)  # GFLOP/s
        _roof = [min(b, c) for b, c in zip(_bw_ceil, _comp_ceil)]
        fig.add_trace(go.Scatter(
            x=_ais.tolist(), y=_roof, mode="lines",
            line=dict(color=color, width=2.5), name=name,
            hovertemplate="AI %{x:.1f} FLOP/B: %{y:,.0f} GFLOP/s<extra></extra>",
        ))
        fig.add_vline(x=ridge, line_dash="dot", line_color=color,
                      annotation_text=f"Ridge: {ridge:.0f}")

    def _part_loop(part, concept, prediction, controls, evidence, decision):
        return part_workflow(
            f"{part} Workflow",
            (
                {
                    "part": part,
                    "concept": concept,
                    "prediction": prediction,
                    "controls": controls,
                    "evidence": evidence,
                    "decision": decision,
                },
            ),
            scenario=(
                f"For {v1_11_profile.label}, keep the hardware ceiling, "
                "memory movement, and selected accelerator path in view."
            ),
            reflection="Before moving on, connect the computed evidence to the hardware decision you would defend.",
        )

    # ─────────────────────────────────────────────────────────────────────
    # PART A: The Memory Wall (Roofline Diagnosis)
    # ─────────────────────────────────────────────────────────────────────
    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Kernel Optimization Team
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our GEMM-like kernel is far below peak on {v1_11_roofline.hardware_name}.
                We optimized the code for two weeks. The profiler shows no bugs. Are we
                compute-bound, memory-bound, or using the wrong accelerator path?&rdquo;
            </div>
        </div>
        """))
        items.append(_part_loop(
            "Part A",
            "Roofline Diagnosis",
            "Predict whether low MFU means a bug or a memory-bandwidth ceiling.",
            "Adjust matrix dimension and precision.",
            "Inspect arithmetic intensity, ridge point, attainable GFLOP/s, and MFU.",
            "Decide whether the kernel is memory-bound, compute-bound, or on the wrong path.",
        ))
        items.append(mo.md(f"""
        ## The Roofline Model: Two Ceilings

        Every operation is bounded by **two** ceilings:
        - **Compute ceiling**: peak TFLOPS of the hardware
        - **Bandwidth ceiling**: memory bandwidth x arithmetic intensity

        ```
        Attainable GFLOP/s = min(Peak_TFLOPS, BW x AI)
        AI = FLOPs / Bytes_accessed
        Ridge Point = Peak_TFLOPS / BW = {v1_11_roofline.ridge_flop_per_byte:.1f} FLOP/byte ({v1_11_roofline.label})
        ```

        Below the ridge point: **memory-bound**. Above: **compute-bound**.

        **Intuition:** Think of a factory where workers (compute) process raw materials
        (data from memory). If raw materials arrive slowly, workers sit idle no matter
        how fast they are — that is memory-bound. If workers are slow, materials pile
        up — that is compute-bound. The ridge point is where the two bottlenecks balance.
        """))
        items.append(pA_pred)
        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the Roofline explorer."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pA_dim, pA_prec], justify="start"))

        _N = pA_dim.value
        _workload = gemm_workload(dimension=_N, precision=pA_prec.value)
        _point = roofline_point(v1_11_roofline, _workload.arithmetic_intensity)
        _bpp = _workload.bytes_per_element
        _flops = _workload.flops
        _bytes = _workload.bytes_moved
        _ai = _workload.arithmetic_intensity

        _peak_gflops = _point.peak_gflops
        _attainable = _point.attainable_gflops
        _mfu = _point.mfu_pct
        _regime = _point.regime

        _fig = go.Figure()
        _draw_roofline(
            _fig,
            v1_11_roofline.peak_tflops,
            v1_11_roofline.bandwidth_gbs,
            v1_11_roofline.ridge_flop_per_byte,
            COLORS["BlueLine"],
            f"{v1_11_roofline.label} Roofline",
        )
        _fig.add_trace(go.Scatter(
            x=[_ai], y=[_attainable], mode="markers+text",
            marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond"),
            text=[f"N={_N}"], textposition="top right",
            name=f"GEMM N={_N}",
            hovertemplate="AI %{x:.1f} FLOP/B: %{y:,.0f} GFLOP/s<extra></extra>",
        ))
        _fig.update_layout(
            height=380,
            xaxis=dict(title="Arithmetic Intensity (FLOP/byte)", type="log", range=[-1, 4]),
            yaxis=dict(title="Attainable GFLOP/s", type="log"),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _regime_color = COLORS["GreenLine"] if _regime == "Compute-bound" else COLORS["OrangeLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Arithmetic Intensity</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_ai:.0f} FLOP/B</div>
                <div style="font-size:0.72rem; color:#94a3b8;">Ridge: {v1_11_roofline.ridge_flop_per_byte:.1f} FLOP/B</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_regime_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Model FLOPs Utilization (MFU)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_regime_color};">{_mfu:.1f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_regime}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Attainable</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_attainable:.0f} GF/s</div>
                <div style="font-size:0.72rem; color:#94a3b8;">of {_peak_gflops:.0f} peak</div>
            </div>
        </div>
        """))

        items.append(mo.md(f"""
**Roofline -- Live Calculation** (`N={_N}, {pA_prec.value.upper()}`)

```
FLOPs  = 2 x N^3 = 2 x {_N}^3 = {_flops:.2e}
Bytes  = 3 x N^2 x {_bpp} = {_bytes:.2e}
AI     = {_flops:.2e} / {_bytes:.2e} = {_ai:.0f} FLOP/byte
Ridge  = {v1_11_roofline.peak_tflops:.3g} TFLOP/s / {v1_11_roofline.bandwidth_gbs:.3g} GB/s = {v1_11_roofline.ridge_flop_per_byte:.1f} FLOP/byte
Regime = {"AI < Ridge => MEMORY-BOUND" if _ai < v1_11_roofline.ridge_flop_per_byte else "AI >= Ridge => COMPUTE-BOUND"}
MFU    = {_mfu:.1f}%
```
*Source: `mlsysbook_labs.roofline_point`, using `{v1_11_roofline.hardware_ref}`.*
        """))

        if pA_pred.value == "roofline":
            items.append(mo.callout(mo.md("**Correct.** The kernel is hitting the bandwidth ceiling correctly. "
                f"AI = {_ai:.0f} vs ridge = {v1_11_roofline.ridge_flop_per_byte:.1f} means it is {_regime.lower()}. "
                "Increase N with the slider to cross the ridge point and reach compute-bound territory."), kind="success"))
        else:
            items.append(mo.callout(mo.md("**Low utilization does not mean broken code.** "
                f"At N={_N}, AI = {_ai:.0f} and the {v1_11_profile.label} ridge point is "
                f"{v1_11_roofline.ridge_flop_per_byte:.1f}. The selected workload is {_regime.lower()}."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: The Roofline Model": mo.md("""
**Formula:**
$$
\\text{Attainable Performance} = \\min\\!\\left(\\text{AI} \\times \\text{BW},\\; \\text{Peak FLOPS}\\right)
$$

Ridge point (transition from memory-bound to compute-bound):
$$
\\text{Ridge} = \\frac{\\text{Peak FLOPS}}{\\text{BW}} \\quad \\text{(FLOP/byte)}
$$

**Variables:**
- **AI**: arithmetic intensity = FLOPs / bytes transferred (FLOP/byte)
- **BW**: memory bandwidth (bytes/s)
- **Peak FLOPS**: theoretical maximum compute throughput

When $\\text{AI} < \\text{Ridge}$, the kernel is memory-bound and performance scales with bandwidth. When $\\text{AI} > \\text{Ridge}$, it is compute-bound and hits the FLOPS ceiling.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B: Kernel Fusion
    # ─────────────────────────────────────────────────────────────────────
    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; ML Compiler Team
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;LLM inference mixes GEMM with LayerNorm and Softmax. Each elementwise op
                requires a full round-trip to HBM. Can we fuse them into one kernel?&rdquo;
            </div>
        </div>
        """))
        items.append(_part_loop(
            "Part B",
            "Kernel Fusion",
            "Predict the speedup from fusing memory-bound elementwise kernels.",
            "Switch eager versus fused mode and vary batch size.",
            "Compare HBM traffic, latency, and eliminated intermediate writes.",
            "Decide whether fusion is worth implementing for the selected hardware.",
        ))
        items.append(mo.md("""
        ## Kernel Fusion: The Elementwise Trap

        LLM inference mixes kernels spanning **3 orders of magnitude** in arithmetic intensity:
        - **GEMM**: AI ~ 100-1000 FLOP/byte (can be compute-bound)
        - **LayerNorm**: AI ~ 0.83 FLOP/byte (permanently memory-bound)
        - **Softmax**: AI ~ 1.5 FLOP/byte (permanently memory-bound)

        Each separate kernel requires a full HBM read + write. Fusion eliminates
        intermediate writes, collapsing 3 memory-bound kernels into 1.
        """))
        items.append(pB_pred)
        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the fusion analyzer."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pB_mode, pB_batch], justify="start"))

        _mode = pB_mode.value
        _batch = pB_batch.value
        _hidden = 4096  # typical LLM hidden dim

        # Per-operation specs
        _bytes_per_elem = 2  # FP16
        _tensor_bytes = _batch * _hidden * _bytes_per_elem

        # Eager: each op does a full memory read + write
        _ln_flops = _batch * _hidden * 5  # mean, var, normalize, scale, shift
        _dropout_flops = _batch * _hidden * 1
        _relu_flops = _batch * _hidden * 1

        _eager_reads = 3
        _eager_writes = 3
        _eager_flops = _ln_flops + _dropout_flops + _relu_flops

        # Fused: single read + single write
        _fused_reads = 1
        _fused_writes = 1
        _fused_flops = _eager_flops

        _fusion = fusion_traffic(
            elements=_batch * _hidden,
            bytes_per_element=_bytes_per_elem,
            bandwidth_gbs=v1_11_roofline.bandwidth_gbs,
            eager_reads=_eager_reads,
            eager_writes=_eager_writes,
            fused_reads=_fused_reads,
            fused_writes=_fused_writes,
        )
        _eager_bytes = _fusion.eager_bytes
        _fused_bytes = _fusion.fused_bytes
        _eager_time = _fusion.eager_time_us
        _fused_time = _fusion.fused_time_us
        _speedup = _fusion.speedup

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=["Eager (3 kernels)", "Fused (1 kernel)"],
            y=[_eager_time, _fused_time],
            marker_color=[COLORS["RedLine"], COLORS["GreenLine"]],
            text=[f"{_eager_time:.1f} us", f"{_fused_time:.1f} us"],
            textposition="outside", opacity=0.88,
            hovertemplate="%{x}: %{y:.1f} us<extra></extra>",
        ))
        _fig.update_layout(
            height=300, yaxis=dict(title="Latency (microseconds)"),
            showlegend=False,
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Fusion Speedup</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['GreenLine']};">{_speedup:.1f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">3 kernels -> 1 kernel</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">HBM Traffic Saved</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">{(_eager_bytes - _fused_bytes)/1024:.0f} KB</div>
                <div style="font-size:0.72rem; color:#94a3b8;">eliminated intermediate writes</div>
            </div>
        </div>
        """))

        items.append(mo.md(f"""
**Fusion -- Live Calculation** (`batch={_batch}, hidden={_hidden}`)

```
Tensor size:   {_batch} x {_hidden} x {_bytes_per_elem} = {_tensor_bytes:,} bytes
Eager traffic: ({_eager_reads} reads + {_eager_writes} writes) x {_tensor_bytes:,} = {_eager_bytes:,} bytes
Fused traffic: ({_fused_reads} read + {_fused_writes} write) x {_tensor_bytes:,} = {_fused_bytes:,} bytes
Speedup:       {_eager_bytes:,} / {_fused_bytes:,} = {_speedup:.1f}x
```
*Source: `mlsysbook_labs.fusion_traffic`, using {v1_11_roofline.bandwidth_gbs:g} GB/s from `{v1_11_roofline.hardware_ref}`.*
        """))

        if pB_pred.value == "3_5":
            items.append(mo.callout(mo.md(f"**Correct.** Fusion yields {_speedup:.1f}x speedup by eliminating "
                "intermediate HBM round-trips. For memory-bound ops, the dominant cost is data movement, "
                "not computation."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"**Fusion gives {_speedup:.1f}x speedup.** "
                "The speedup comes from eliminating memory traffic, not compute. "
                "Each eliminated HBM write saves bandwidth-limited time."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Kernel Fusion Memory Traffic Reduction": mo.md("""
**Formula (unfused):**
$$
T_{\\text{unfused}} = \\sum_{i=1}^{K} \\frac{2 \\cdot B_{\\text{tensor}}}{\\text{BW}} = K \\cdot \\frac{2 \\cdot B_{\\text{tensor}}}{\\text{BW}}
$$

**Formula (fused):**
$$
T_{\\text{fused}} = \\frac{2 \\cdot B_{\\text{tensor}}}{\\text{BW}} \\qquad \\text{Speedup} = K
$$

**Variables:**
- **$K$**: number of elementwise kernels (e.g., 3 for LayerNorm + Dropout + ReLU)
- **$B_{\\text{tensor}}$**: size of the intermediate tensor in bytes
- **$\\text{BW}$**: HBM bandwidth
- Factor of 2: each unfused kernel reads from and writes to HBM

Fusion eliminates $K-1$ round-trips to HBM, giving up to $K\\times$ speedup for memory-bound ops.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C: Hardware Balance Shift
    # ─────────────────────────────────────────────────────────────────────
    def build_part_c():
        items = []
        _comparison_names = " and ".join(profile.label for profile in v1_11_comparison_profiles)
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Cross-Platform Deployment
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our GEMM kernel changes regime between {_comparison_names}. The code path
                is the same, so why does the bottleneck move when the hardware changes?&rdquo;
            </div>
        </div>
        """))
        items.append(_part_loop(
            "Part C",
            "Balance Shift",
            "Predict whether the same kernel keeps the same regime across hardware.",
            "Select the comparison hardware profile.",
            "Inspect each ridge point, arithmetic intensity, attainable performance, and MFU.",
            "Decide which target is easier to saturate and why.",
        ))
        _ridge_list = "\n".join(
            f"- {profile.label}: {profile.ridge_flop_per_byte:.1f} FLOP/byte"
            for profile in v1_11_comparison_profiles
        )
        items.append(mo.md(f"""
        ## The Hardware Balance Shift

        The **ridge point** differs across hardware because compute grows faster than bandwidth:
        {_ridge_list}

        A kernel at the same arithmetic intensity can change regime across these targets.
        **More powerful accelerators can be harder to saturate** when peak compute grows
        faster than memory bandwidth.
        """))
        items.append(pC_pred)
        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the hardware comparison."), kind="warn"))
            return mo.vstack(items)

        items.append(pC_hw)

        _color_cycle = (COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"], "#7c3aed")
        _hw_map = {profile.track_id: profile for profile in v1_11_comparison_profiles}
        _selected = _hw_map[pC_hw.value]
        _name = _selected.label
        _ridge = _selected.ridge_flop_per_byte
        _color = COLORS["RedLine"]

        _N = 1024
        _workload = gemm_workload(dimension=_N, precision="fp16")
        _ai = _workload.arithmetic_intensity
        _point = roofline_point(_selected, _ai)
        _attainable = _point.attainable_gflops
        _mfu = _point.mfu_pct
        _regime = _point.regime

        _fig = go.Figure()
        # Draw selected comparison rooflines
        for _idx, _profile in enumerate(v1_11_comparison_profiles):
            _draw_roofline(
                _fig,
                _profile.peak_tflops,
                _profile.bandwidth_gbs,
                _profile.ridge_flop_per_byte,
                _color_cycle[_idx % len(_color_cycle)],
                _profile.label,
            )

        _fig.add_trace(go.Scatter(
            x=[_ai], y=[_attainable], mode="markers",
            marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond"),
            name=f"GEMM N=1024 on {_name}",
            hovertemplate="AI %{x:.1f} FLOP/B: %{y:,.0f} GFLOP/s<extra></extra>",
        ))
        _fig.update_layout(
            height=400,
            xaxis=dict(title="Arithmetic Intensity (FLOP/byte)", type="log", range=[-1, 4]),
            yaxis=dict(title="Attainable GFLOP/s", type="log"),
            legend=dict(orientation="h", y=1.14, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _rc = COLORS["GreenLine"] if _regime == "Compute-bound" else COLORS["OrangeLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Hardware</div>
                <div style="font-size:1.3rem; font-weight:800; color:{_color};">{_name}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">Ridge: {_ridge:.1f} FLOP/B</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_rc}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Regime (N=1024)</div>
                <div style="font-size:1.3rem; font-weight:800; color:{_rc};">{_regime}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">AI = {_ai:.0f}, MFU = {_mfu:.1f}%</div>
            </div>
        </div>
        """))

        _ridge_values = ", ".join(
            f"{profile.label}: {profile.ridge_flop_per_byte:.1f} FLOP/B"
            for profile in v1_11_comparison_profiles
        )
        if pC_pred.value in {"no", "depends"}:
            items.append(mo.callout(mo.md(
                f"**Correct idea.** The same AI={_ai:.0f} workload can change regime when ridge points differ. "
                f"Here: {_ridge_values}. Toggle hardware to see the shift."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**More powerful does not mean easier to saturate.** Ridge point is peak compute divided by bandwidth. "
                f"The selected comparison set has these ridges: {_ridge_values}."
            ), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Arithmetic Intensity of GEMM": mo.md("""
**Formula (square GEMM, N x N):**
$$
\\text{AI}_{\\text{GEMM}} = \\frac{2N^3}{3N^2 \\cdot b} = \\frac{2N}{3b} \\quad \\text{(FLOP/byte)}
$$

**Variables:**
- **$N$**: matrix dimension
- **$b$**: bytes per element (4 for FP32, 2 for FP16)
- **$2N^3$**: FLOPs for matrix multiply (multiply + accumulate)
- **$3N^2 \\cdot b$**: bytes transferred (read A, read B, write C)

AI grows linearly with $N$. Small matrices are memory-bound; large matrices are compute-bound. The crossover $N$ differs by hardware because ridge points differ.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D: Energy Roofline
    # ─────────────────────────────────────────────────────────────────────
    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid #6366f1; background:#f0f4ff;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Sustainability Engineer
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We need to minimize energy per inference. Does the Roofline tell us anything
                about where energy goes? Are memory-bound operations wasting Joules?&rdquo;
            </div>
        </div>
        """))
        items.append(_part_loop(
            "Part D",
            "Energy Roofline",
            "Predict whether memory-bound or compute-bound execution uses more energy.",
            "Compare low and high arithmetic-intensity operating points.",
            "Inspect energy per FLOP and the memory-traffic energy ratio.",
            "Decide which optimization reduces Joules for the selected track.",
        ))
        items.append(mo.md(f"""
        ## The Energy Roofline

        Energy efficiency has its own Roofline. For the **{v1_11_profile.label}** track,
        the important question is whether bytes are crossing the expensive memory hierarchy
        or staying close to the selected accelerator path. In the memory-bound regime, most
        Joules go to data movement. In the compute-bound regime, most Joules go to useful arithmetic.
        The exact numbers vary with process node, but the ratio between data movement
        and compute energy remains the enduring insight.

        ```
        Energy/FLOP (memory-bound) = E_memory / AI  (decreases with AI)
        Energy/FLOP (compute-bound) = E_compute      (constant floor)
        ```

        The energy-optimal operating point is deep in the compute-bound regime.
        """))
        items.append(pD_pred)
        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the energy Roofline."), kind="warn"))
            return mo.vstack(items)

        _e_dram = 640  # pJ per off-chip memory byte, chosen as a teaching-scale reference
        _e_compute = 3.7  # pJ per FP32 FLOP

        _ais = np.logspace(0, 3, 100)
        _energy_per_flop = [max(_e_dram / ai, _e_compute) for ai in _ais]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_ais.tolist(), y=_energy_per_flop, mode="lines",
            line=dict(color=COLORS["RedLine"], width=3), name="Energy per FLOP",
            hovertemplate="AI %{x:.1f} FLOP/B: %{y:.1f} pJ/FLOP<extra></extra>",
        ))
        _fig.add_hline(y=_e_compute, line_dash="dash", line_color=COLORS["GreenLine"],
                       annotation_text=f"Compute floor: {_e_compute} pJ")

        # Mark two operations
        _fig.add_trace(go.Scatter(x=[10], y=[max(_e_dram/10, _e_compute)],
            mode="markers+text", marker=dict(size=12, color=COLORS["OrangeLine"]),
            text=["Memory-bound (AI=10)"], textposition="top right", name="AI=10",
            hovertemplate="AI %{x}: %{y:.1f} pJ/FLOP<extra></extra>"))
        _fig.add_trace(go.Scatter(x=[500], y=[max(_e_dram/500, _e_compute)],
            mode="markers+text", marker=dict(size=12, color=COLORS["GreenLine"]),
            text=["Compute-bound (AI=500)"], textposition="top left", name="AI=500",
            hovertemplate="AI %{x}: %{y:.1f} pJ/FLOP<extra></extra>"))

        _fig.update_layout(
            height=360,
            xaxis=dict(title="Arithmetic Intensity (FLOP/byte)", type="log"),
            yaxis=dict(title="Energy per FLOP (pJ)", type="log"),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _e_mem_bound = _e_dram / 10
        _e_comp_bound = _e_compute
        _ratio = _e_mem_bound / _e_comp_bound

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Memory-bound (AI=10)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_e_mem_bound:.0f} pJ/FLOP</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Compute-bound (AI=500)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_e_comp_bound:.1f} pJ/FLOP</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Energy Ratio</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_ratio:.0f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">more energy in mem-bound</div>
            </div>
        </div>
        """))

        if pD_pred.value == "10x":
            items.append(mo.callout(mo.md(f"**Correct.** The memory-bound operation uses {_ratio:.0f}x more "
                "energy per FLOP. At AI=10, each FLOP costs 64 pJ in memory traffic energy alone. "
                "Same FLOPs does NOT mean same energy."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"**Same FLOPs does not mean same energy.** "
                f"The memory-bound operation wastes {_ratio:.0f}x more energy on data movement. "
                "This is why hardware architects build larger caches."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: Energy per FLOP in Different Roofline Regimes": mo.md("""
**Formula:**
$$
E_{\\text{per FLOP}} = E_{\\text{compute}} + \\frac{E_{\\text{DRAM per byte}}}{\\text{AI}}
$$

**Variables:**
- **$E_{\\text{compute}}$**: energy per arithmetic operation (~0.2-3.7 pJ)
- **$E_{\\text{DRAM per byte}}$**: energy per off-chip memory byte (~20-640 pJ teaching range)
- **$\\text{AI}$**: arithmetic intensity (FLOP/byte)

At low AI (memory-bound), the $E_{\\text{DRAM}}/\\text{AI}$ term dominates. At AI=10, each FLOP costs ~64 pJ in memory traffic energy alone. At AI=500, the memory traffic cost is negligible.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART E: Tiling Dividend
    # ─────────────────────────────────────────────────────────────────────
    def build_part_e():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Transformer Optimization
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Standard attention materializes the full N x N attention matrix in HBM.
                FlashAttention tiles the computation to fit in SRAM. How much does tiling help?&rdquo;
            </div>
        </div>
        """))
        items.append(_part_loop(
            "Part E",
            "Tiling Dividend",
            "Predict how much tiling improves attention-style memory movement.",
            "Adjust tile size and sequence length.",
            "Inspect speedup and memory traffic saved.",
            "Decide whether tiled attention is a required implementation tactic.",
        ))
        _memory_name = "HBM" if v1_11_roofline.bandwidth_gbs >= 1000 else "device memory"
        _scratchpad_name = "SRAM/cache" if v1_11_profile.track_id != "cloud_fleet" else "SRAM/L2"
        items.append(mo.md(f"""
        ## The Tiling Dividend

        Standard attention computes Q*K^T as a full NxN matrix, writes to {_memory_name},
        then reads back for softmax. **FlashAttention-style tiling** tiles the computation into
        {_scratchpad_name}-sized blocks, eliminating redundant memory traffic.

        ```
        Standard:  memory reads = O(N^2 * d)       (full attention matrix)
        Tiled:     memory reads = O(N^2 * d / M)   (M = scratchpad/cache size)
        Speedup    ~ M / (d * tile_ratio)
        ```

        Tiling keeps hot data near the accelerator before it escapes to slower memory.
        """))
        items.append(pE_pred)
        if pE_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the tiling analyzer."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pE_tile, pE_seq], justify="start"))

        _tile = pE_tile.value
        _seq = pE_seq.value
        _d = 128  # head dimension
        _sram_kb = 192 if v1_11_profile.track_id == "cloud_fleet" else 64

        # Standard attention: read/write full NxN attention matrix
        _std_bytes = 2 * _seq * _seq * 2  # Q*K^T + softmax output, FP16
        # Tiled attention: only tile_size chunks at a time
        _n_tiles = math.ceil(_seq / _tile)
        _tiled_bytes = 2 * _n_tiles * _tile * _d * 2 + _seq * _d * 2  # much less HBM traffic
        _tiled_bytes = max(_tiled_bytes, _seq * _d * 2)  # minimum is reading Q,K,V

        _std_time_us = _std_bytes / (v1_11_roofline.bandwidth_gbs * 1e9) * 1e6
        _tiled_time_us = _tiled_bytes / (v1_11_roofline.bandwidth_gbs * 1e9) * 1e6
        _speedup = _std_time_us / _tiled_time_us if _tiled_time_us > 0 else 1
        _speedup = min(_speedup, 10)  # cap at realistic max

        _fig = go.Figure()
        _tiles_range = [2**i for i in range(5, 12) if 2**i <= _seq]
        _speedups = []
        for _t in _tiles_range:
            _nt = math.ceil(_seq / _t)
            _tb = 2 * _nt * _t * _d * 2 + _seq * _d * 2
            _tb = max(_tb, _seq * _d * 2)
            _s = _std_bytes / _tb if _tb > 0 else 1
            _speedups.append(min(_s, 10))

        _fig.add_trace(go.Scatter(
            x=_tiles_range, y=_speedups, mode="lines+markers",
            line=dict(color=COLORS["GreenLine"], width=2),
            marker=dict(size=8), name="Speedup vs Tile Size",
            hovertemplate="Tile %{x}: %{y:.2f}x speedup<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=[_tile], y=[_speedup], mode="markers",
            marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond"),
            name=f"Current: {_tile}",
            hovertemplate="Tile %{x}: %{y:.2f}x speedup<extra></extra>",
        ))
        _fig.update_layout(
            height=320, xaxis=dict(title="Tile Size (elements)", type="log"),
            yaxis=dict(title="Speedup vs Standard Attention"),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Tiling Speedup</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['GreenLine']};">{_speedup:.1f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">tile={_tile}, seq={_seq}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">HBM Traffic Saved</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">
                    {(_std_bytes - _tiled_bytes) / (1024*1024):.1f} MB</div>
                <div style="font-size:0.72rem; color:#94a3b8;">per layer on {v1_11_roofline.label}</div>
            </div>
        </div>
        """))

        if pE_pred.value == "10x":
            items.append(mo.callout(mo.md(f"**Correct.** FlashAttention achieves {_speedup:.1f}x by tiling "
                "the attention computation to fit in SRAM, eliminating redundant HBM round-trips. "
                "Increase sequence length to see even larger gains."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"**Tiling gives {_speedup:.1f}x speedup.** "
                "Standard attention materializes the full NxN matrix in memory. Tiling keeps "
                "hot data near the accelerator, dramatically reducing memory traffic."), kind="warn"))
        items.append(mo.accordion({
            "Math Peek: FlashAttention Memory Complexity": mo.md("""
**Standard attention memory access:**
$$
\\text{Memory}_{\\text{standard}} = \\Theta(N^2 \\cdot d) \\quad \\text{(materializes full } N \\times N \\text{ matrix)}
$$

**Tiled (FlashAttention-style) memory access:**
$$
\\text{Memory}_{\\text{tiled}} = \\Theta\\!\\left(\\frac{N^2 \\cdot d^2}{M}\\right)
$$

**Variables:**
- **$N$**: sequence length
- **$d$**: head dimension
- **$M$**: available scratchpad/cache tile capacity

Speedup $\\approx N / B$ (where $B$ is tile size), reaching ~10x at seq_len=4096, tile=256. Tiling keeps Q, K, V blocks close to the accelerator, avoiding the $N^2$ materialization in slower memory.
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
                        <strong>1. Low MFU is not broken code.</strong>
                        At AI below the ridge point, the kernel is correctly hitting the memory
                        bandwidth ceiling. The Roofline is a diagnostic tool -- the ceiling
                        changes with matrix size and hardware.
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>2. Fusion eliminates memory traffic, not compute.</strong>
                        Fusing 3 memory-bound kernels into 1 yields 3x speedup by eliminating
                        intermediate HBM writes. The dominant cost is data movement.
                    </div>
                    <div>
                        <strong>3. More powerful hardware is harder to saturate.</strong>
                        The selected comparison set has different ridge points. The same kernel
                        can be compute-bound on one track and memory-bound on another.
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
                        <strong>Lab 12:</strong> The Benchmarking Trap -- vendor benchmarks measure
                        burst performance. Production runs hit Amdahl's ceiling, thermal throttling,
                        and tail latency.
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
                        <strong>Read:</strong> the Hardware Acceleration chapter for Roofline model derivation.<br/>
                        <strong>Build:</strong> TinyTorch Module 11 -- GEMM tiling and FlashAttention.
                    </div>
                </div>
            </div>
            """),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Roofline Diagnosis": build_part_a(),
        "Part B: Kernel Fusion": build_part_b(),
        "Part C: Balance Shift": build_part_c(),
        "Part D: Energy Roofline": build_part_d(),
        "Part E: Tiling Dividend": build_part_e(),
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
def _(COLORS, ledger, mo, pA_pred, pB_pred, pC_pred, pD_pred, pE_pred, v1_11_profile, v1_11_roofline, v1_11_variant):
    if pA_pred.value is not None and pB_pred.value is not None and pC_pred.value is not None and pD_pred.value is not None and pE_pred.value is not None:
        ledger.save(chapter=11, design={
            "lab": "hw_accel",
            "track_id": v1_11_profile.track_id,
            "scenario_id": v1_11_variant.scenario_id,
            "hardware_ref": v1_11_roofline.hardware_ref,
            "completed": True,
            "roofline_diagnosis": pA_pred.value,
            "fusion_speedup_prediction": pB_pred.value,
            "balance_shift_prediction": pC_pred.value,
            "energy_regime_prediction": pD_pred.value,
            "tiling_speedup_prediction": pE_pred.value,
        })
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">11 &middot; Hardware Roofline</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_11_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">CH</span>
        <span class="hud-value">11</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">COMPLETE</span>
    </div>
    """)
    return


# ─── DOWNLOADABLE TRACK REPORT ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    build_lab_report,
    fusion_traffic,
    gemm_workload,
    mo,
    pA_dim,
    pA_prec,
    pA_pred,
    pB_batch,
    pB_mode,
    pB_pred,
    pC_hw,
    pC_pred,
    pD_pred,
    pE_pred,
    pE_seq,
    pE_tile,
    report_export_panel,
    roofline_point,
    v1_11_comparison_profiles,
    v1_11_metadata,
    v1_11_profile,
    v1_11_roofline,
    v1_11_variant,
):
    _workload = gemm_workload(dimension=pA_dim.value, precision=pA_prec.value)
    _primary_point = roofline_point(v1_11_roofline, _workload.arithmetic_intensity)
    _comparison_map = {profile.track_id: profile for profile in v1_11_comparison_profiles}
    _comparison = _comparison_map[pC_hw.value]
    _comparison_point = roofline_point(_comparison, gemm_workload(dimension=1024, precision="fp16").arithmetic_intensity)
    _fusion = fusion_traffic(
        elements=pB_batch.value * 4096,
        bytes_per_element=2,
        bandwidth_gbs=v1_11_roofline.bandwidth_gbs,
    )

    _incomplete = []
    if pA_pred.value is None:
        _incomplete.append("Part A roofline prediction")
    if pB_pred.value is None:
        _incomplete.append("Part B fusion prediction")
    if pC_pred.value is None:
        _incomplete.append("Part C balance-shift prediction")
    if pD_pred.value is None:
        _incomplete.append("Part D energy prediction")
    if pE_pred.value is None:
        _incomplete.append("Part E tiling prediction")

    _report = build_lab_report(
        v1_11_metadata,
        track=v1_11_profile.label,
        scenario=v1_11_variant.workload_summary,
        learning_objectives=(
            "Diagnose whether a workload is memory-bound or compute-bound on the selected hardware.",
            "Explain how fusion, precision, batching, or tiling moves the roofline point.",
            "Choose an accelerator path while naming the remaining hardware limitation.",
        ),
        predictions={
            "roofline_diagnosis": pA_pred.value,
            "fusion_speedup": pB_pred.value,
            "balance_shift": pC_pred.value,
            "energy_regime": pD_pred.value,
            "tiling_speedup": pE_pred.value,
        },
        knob_settings={
            "matrix_dim": pA_dim.value,
            "precision": pA_prec.value,
            "fusion_mode": pB_mode.value,
            "fusion_batch": pB_batch.value,
            "comparison_track": pC_hw.value,
            "tile_size": pE_tile.value,
            "sequence_length": pE_seq.value,
        },
        evidence_summary={
            "hardware_ref": v1_11_roofline.hardware_ref,
            "model_ref": v1_11_variant.model_ref,
            "peak_tflops": round(v1_11_roofline.peak_tflops, 6),
            "bandwidth_gbs": round(v1_11_roofline.bandwidth_gbs, 6),
            "ridge_flop_per_byte": round(v1_11_roofline.ridge_flop_per_byte, 3),
            "workload_ai": round(_workload.arithmetic_intensity, 3),
            "primary_regime": _primary_point.regime,
            "primary_mfu_pct": round(_primary_point.mfu_pct, 3),
            "fusion_speedup": round(_fusion.speedup, 3),
            "comparison_track": _comparison.label,
            "comparison_regime": _comparison_point.regime,
        },
        final_decision=(
            f"Use the {v1_11_variant.assumptions.get('accelerator_path', v1_11_profile.label)} path "
            f"only if the workload point has enough headroom against {v1_11_variant.guardrail_metric}."
        ),
        big_takeaways=(
            "Peak TOPS is not performance; ridge point decides which ceiling is active.",
            "Fusion and tiling improve roofline position by reducing memory traffic.",
            "The same workload can change regime across iPhone, Oura Ring, RoboTaxi, and Cloud Fleet.",
        ),
        reflections={
            "diagnosis": (
                f"{v1_11_profile.label} currently reports {_primary_point.regime.lower()} behavior "
                f"for the selected GEMM point."
            ),
            "tradeoff": (
                f"The track optimizes {v1_11_variant.primary_metric} while guarding "
                f"{v1_11_variant.guardrail_metric}."
            ),
            "residual_risk": (
                "Roofline estimates expose ceilings; production still needs profiler traces and supported-op checks."
            ),
        },
        residual_risk=(
            "The report uses source-traced roofline estimates. Validate with hardware counters, "
            "operator support, thermal behavior, and p99 latency before deployment."
        ),
        source_trace={
            "track_id": v1_11_profile.track_id,
            "scenario_id": v1_11_variant.scenario_id,
            "hardware_ref": v1_11_variant.hardware_ref,
            "model_ref": v1_11_variant.model_ref,
            "shared_helper": "mlsysbook_labs.roofline",
            "source_policy": v1_11_profile.source_policy,
        },
        result_snapshot={
            "roofline_profile": v1_11_roofline,
            "workload": _workload,
            "primary_point": _primary_point,
            "comparison_profile": _comparison,
            "comparison_point": _comparison_point,
            "fusion": _fusion,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-11 report is generated locally from the selected track, MLSysIM hardware refs, "
                "and shared `mlsysbook_labs.roofline` calculations."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return

if __name__ == "__main__":
    app.run()
