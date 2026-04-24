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
    from mlsysim import Engine, Models, Hardware

    A100_TFLOPS = Hardware.Cloud.A100.compute.peak_flops.m_as("TFLOPs/s")
    A100_BW     = Hardware.Cloud.A100.memory.bandwidth.m_as("GB/s")

    RESNET50_FLOPS = Models.ResNet50.inference_flops.m_as("flop")
    DSCNN_FLOPS = Models.Tiny.DS_CNN.inference_flops.m_as("flop")

    ESP32_TFLOPS = Hardware.Tiny.ESP32_S3.compute.peak_flops.m_as("TFLOPs/s")

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        COLORS, LAB_CSS, apply_plotly_theme,
        go, mo, np, math,
        Engine, Models, Hardware,
        A100_TFLOPS, A100_BW,
        RESNET50_FLOPS, DSCNN_FLOPS,
        ESP32_TFLOPS,
        ledger,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 04
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Data Gravity Trap
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Feeding Tax &middot; Data Gravity &middot; Cascades &middot; False Positives
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Data is the heaviest object in ML systems. Your GPU starves when storage
                is slow, moving data costs more than compute, small errors amplify through
                the pipeline, and 99% accuracy means nothing at deployment scale.
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
                    Chapter 4: Data Engineering
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">GPU Utilization &lt; 5%</span>
                <span class="badge badge-warn">$4,000 Egress for 50 TB</span>
                <span class="badge badge-fail">2% Error -> 15% Accuracy Drop</span>
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
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Diagnose the feeding tax</strong> &mdash;
                    show that standard cloud storage leaves an A100 idle &gt;95% of the time.</div>
                <div style="margin-bottom: 3px;">2. <strong>Calculate data gravity economics</strong> &mdash;
                    find the crossover where moving compute is cheaper than moving data.</div>
                <div style="margin-bottom: 3px;">3. <strong>Trace data cascade amplification</strong> &mdash;
                    show how a 2% ingestion error amplifies to ~15% accuracy degradation.</div>
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
                    Iron Law D_vol/BW term from Lab 01-02 &middot;
                    ML lifecycle from Lab 03 &middot;
                    the Data Engineering chapter
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
                &ldquo;Your A100 costs $15K but sits idle 95% of the time. Moving 50 TB
                costs $4,000 in egress fees. A 2% data error erases 15% accuracy.
                Is data just input to your model, or the heaviest constraint in the system?&rdquo;
            </div>
        </div>
    </div>
    """)
    return



# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **The Data Engineering chapter** -- Data pipelines, I/O bottlenecks, and the feeding tax.
    - **The Data Gravity section (Ch. 4)** -- Egress costs and data gravity economics.
    - **The Data Cascades section (Ch. 4)** -- Error amplification and data contracts.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    A100_TFLOPS, A100_BW,
    RESNET50_FLOPS, DSCNN_FLOPS, ESP32_TFLOPS,
    Engine, Models, Hardware,
    apply_plotly_theme, go, math, mo, np,
):
    # ── Part A widgets ───────────────────────────────────────────────────
    partA_prediction = mo.ui.radio(
        options={
            "A) 80-90% (GPUs are expensive, they should be busy)": "80",
            "B) 50-60% (some I/O overhead expected)":              "50",
            "C) 20-30% (significant I/O bottleneck)":              "20",
            "D) <5% (GPU almost entirely idle)":                   "5",
        },
        label="Training ResNet-50 on an A100 with a standard cloud SSD (250 MB/s). "
              "What percentage of wall-clock time is the GPU computing?",
    )
    return (partA_prediction,)

@app.cell(hide_code=True)
def _(mo):
    partA_storage = mo.ui.dropdown(
        options={
            "HDD (100 MB/s)": 100, "SSD (250 MB/s)": 250,
            "NVMe (3,000 MB/s)": 3000, "RAM disk (25,000 MB/s)": 25000,
        },
        value="SSD (250 MB/s)",
        label="Storage type:",
    )
    partA_workers = mo.ui.slider(
        start=1, stop=16, value=1, step=1,
        label="DataLoader workers",
    )

    # ── Part B widgets ───────────────────────────────────────────────────
    partB_prediction = mo.ui.radio(
        options={
            "A) ~30 minutes":  "30min",
            "B) ~5 hours":     "5hr",
            "C) ~11 hours":    "11hr",
            "D) ~3 days":      "3days",
        },
        label="50 TB dataset, 10 Gbps link. How long to transfer?",
    )
    return (partA_storage, partA_workers, partB_prediction)

@app.cell(hide_code=True)
def _(mo):
    partB_dataset = mo.ui.slider(
        start=1, stop=500, value=50, step=1,
        label="Dataset size (TB)",
    )
    partB_bandwidth = mo.ui.dropdown(
        options={"1 Gbps": 1, "10 Gbps": 10, "25 Gbps": 25, "100 Gbps": 100},
        value="10 Gbps",
        label="Network bandwidth:",
    )

    # ── Part C widgets ───────────────────────────────────────────────────
    partC_prediction = mo.ui.radio(
        options={
            "A) ~2% (error passes linearly)":        "2",
            "B) ~5% (slight amplification)":          "5",
            "C) ~15% (significant amplification)":    "15",
            "D) ~50% (catastrophic)":                 "50",
        },
        label="A pipeline has 2% error at ingestion. What accuracy degradation at output?",
    )
    return (partB_bandwidth, partB_dataset, partC_prediction)

@app.cell(hide_code=True)
def _(mo):
    partC_error_rate = mo.ui.slider(
        start=0.5, stop=10.0, value=2.0, step=0.5,
        label="Ingestion error rate (%)",
    )
    partC_stages = mo.ui.slider(
        start=2, stop=8, value=5, step=1,
        label="Pipeline stages",
    )

    # ── Part D widgets ───────────────────────────────────────────────────
    partD_prediction = mo.ui.radio(
        options={
            "A) 99% (one nine)":              "99",
            "B) 99.9% (three nines)":          "999",
            "C) 99.999% (five nines)":         "99999",
            "D) 99.99996% (six nines)":        "999999",
        },
        label="Always-on speaker, 1 false wake per month max. "
              "What rejection rate is required?",
    )
    return (partC_error_rate, partC_stages, partD_prediction)

# ─── widget cell: extracted from tabs cell body (#1332 polish) ────
@app.cell(hide_code=True)
def _(mo):
    partD_tolerance = mo.ui.slider(
        start=1, stop=50, value=1, step=1,
        label="False wake-ups per month (tolerance)",
    )
    partD_duty = mo.ui.slider(
        start=1, stop=24, value=24, step=1,
        label="Duty cycle (hours per day)",
    )
    return (partD_duty, partD_tolerance)


@app.cell(hide_code=True)
def _(
    mo, partA_prediction, partA_storage, partA_workers,
    partB_bandwidth, partB_dataset, partB_prediction, partC_error_rate,
    partC_prediction, partC_stages, partD_prediction, partD_duty,
    partD_tolerance,
):

    # ═════════════════════════════════════════════════════════════════════
    # PART A -- The Feeding Tax
    # ═════════════════════════════════════════════════════════════════════

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; ML Infrastructure Lead
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We bought 8 A100s for training ResNet-50. The GPUs cost $120K total.
                Training is slower than expected. Should we buy more GPUs?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Infrastructure Team
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Feeding Tax: When Your GPU Starves

        A standard cloud SSD (250 MB/s) cannot feed an A100 fast enough.
        The GPU completes its compute in microseconds, then waits milliseconds
        for the next batch. The bottleneck is I/O, not compute.

        ```
        GPU utilization = compute_time / (compute_time + io_time)
        ```
        """))

        items.append(partA_prediction)

        if partA_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the GPU utilization simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partA_storage, partA_workers], justify="start", gap="2rem"))

        _storage_mbps = partA_storage.value
        _workers = partA_workers.value
        _batch_size = 32

        # Batch data size: 32 images * 224*224*3 * 4 bytes (FP32) = ~19 MB
        _batch_data_mb = _batch_size * 224 * 224 * 3 * 4 / (1024 * 1024)

        # I/O time per batch
        _effective_bw = _storage_mbps * min(_workers, 8)  # diminishing returns after 8
        _io_time_ms = (_batch_data_mb / _effective_bw) * 1000

        # Compute time: forward + backward (3x inference FLOPs) on A100
        _training_flops = RESNET50_FLOPS * 3 * _batch_size
        _compute_ms = (_training_flops / (A100_TFLOPS * 1e12 * 0.5)) * 1000

        _total_ms = _compute_ms + _io_time_ms
        _gpu_util = (_compute_ms / _total_ms * 100) if _total_ms > 0 else 0

        # Timeline bar
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            name="GPU Compute", x=[_compute_ms], y=["Step"], orientation="h",
            marker_color=COLORS["GreenLine"], opacity=0.85,
            text=[f"{_compute_ms:.1f} ms"], textposition="inside",
            textfont=dict(color="white"),
        ))
        _fig.add_trace(go.Bar(
            name="I/O Wait", x=[_io_time_ms], y=["Step"], orientation="h",
            base=_compute_ms,
            marker_color=COLORS["RedLine"], opacity=0.85,
            text=[f"{_io_time_ms:.1f} ms"], textposition="inside",
            textfont=dict(color="white"),
        ))
        _fig.update_layout(
            barmode="stack", height=120,
            xaxis=dict(title="Time per Training Step (ms)", gridcolor="#f1f5f9"),
            yaxis=dict(visible=False),
            legend=dict(orientation="h", y=1.4, x=0),
            margin=dict(l=20, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md("### Training Step Timeline"))
        items.append(mo.as_html(_fig))

        _util_color = COLORS["GreenLine"] if _gpu_util > 80 else COLORS["OrangeLine"] if _gpu_util > 30 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:14px; border:2px solid {_util_color}; border-radius:10px;
                        text-align:center; background:white; flex:1;">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">GPU Utilization</div>
                <div style="font-size:2rem; font-weight:800; color:{_util_color};">
                    {_gpu_util:.1f}%</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['GreenLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Compute Time</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['GreenLine']};">
                    {_compute_ms:.1f} ms</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['RedLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">I/O Time</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['RedLine']};">
                    {_io_time_ms:.1f} ms</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['BlueLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Effective I/O BW</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_effective_bw:,.0f} MB/s</div>
            </div>
        </div>
        """))

        # Failure state: GPU effectively idle
        if _gpu_util < 10:
            items.append(mo.callout(mo.md(
                f"**GPU STARVING — {_gpu_util:.1f}% utilization.** "
                f"The A100 is idle {100 - _gpu_util:.0f}% of the time, waiting for data. "
                f"At ${32:.0f}/hr, you are paying ${32 * (1 - _gpu_util/100):.0f}/hr for nothing. "
                "Increase storage bandwidth or add workers to recover."
            ), kind="danger"))

        _pred = partA_prediction.value
        if _pred == "5":
            items.append(mo.callout(mo.md(
                f"**Correct.** At {_storage_mbps} MB/s with {_workers} worker(s), "
                f"GPU utilization is only {_gpu_util:.1f}%. "
                "The fix is faster storage + parallel loading, not more GPUs."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**GPU utilization is only {_gpu_util:.1f}%.** "
                "The A100 finishes compute in milliseconds but waits for data. "
                "Try NVMe + 8 workers to see utilization improve."
            ), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: GPU Utilization (The Feeding Tax)": mo.md("""
**Formula:**
$$
\\eta_{\\text{GPU}} = \\frac{t_{\\text{compute}}}{t_{\\text{compute}} + t_{\\text{IO}}}
\\quad\\text{where}\\quad
t_{\\text{IO}} = \\frac{D_{\\text{batch}}}{\\text{BW}_{\\text{storage}} \\cdot W}
$$

**Variables:**
- **$\\eta_{\\text{GPU}}$**: GPU utilization (fraction of wall-clock time spent computing)
- **$t_{\\text{compute}}$**: forward + backward pass time = $3 \\cdot \\text{FLOPs} \\cdot B / (R_{\\text{peak}} \\cdot \\eta)$
- **$t_{\\text{IO}}$**: I/O time per batch
- **$D_{\\text{batch}}$**: batch data size in bytes (e.g., 32 images x 224x224x3x4 = ~19 MB)
- **$\\text{BW}_{\\text{storage}}$**: storage bandwidth (SSD ~250 MB/s, NVMe ~3,000 MB/s)
- **$W$**: number of DataLoader workers (diminishing returns above ~8)

When $t_{\\text{IO}} \\gg t_{\\text{compute}}$, the GPU starves regardless of its TFLOPS rating.
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════
    # PART B -- Data Gravity
    # ═════════════════════════════════════════════════════════════════════

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Decision Point &middot; Multi-Cloud Training
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our 50 TB training dataset is in us-east-1 but our GPU cluster is
                in eu-west-1 (cheaper spot pricing). The network link is 10 Gbps. Should
                we move the data or provision GPUs near the data?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Cloud Infrastructure Team
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Data Gravity: Move the Compute, Not the Data

        Moving data across regions is slow and expensive. Use the formulas below
        to estimate the cost before making your prediction:

        ```
        Transfer time  = Dataset / Bandwidth
        Egress cost    = Data size (GB) * rate per GB
        ```
        """))

        items.append(partB_prediction)

        if partB_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the data gravity calculator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partB_dataset, partB_bandwidth], justify="start", gap="2rem"))

        _dataset_tb = partB_dataset.value
        _bw_gbps = partB_bandwidth.value

        _dataset_gb = _dataset_tb * 1000
        _dataset_bits = _dataset_gb * 8  # Gbits
        _transfer_hours = _dataset_bits / (_bw_gbps * 3600) if _bw_gbps > 0 else 0
        _egress_cost = _dataset_gb * 0.08  # $0.08/GB
        _compute_cost = 200  # baseline compute cost for the job

        _move_data_cost = _egress_cost + _compute_cost
        _move_compute_cost = _compute_cost * 1.3  # 30% premium for local provisioning

        # Crossover chart
        _sizes = np.logspace(-1, 3, 200)  # 0.1 GB to 1000 TB
        _move_data_costs = [s * 1000 * 0.08 + _compute_cost for s in _sizes]
        _move_compute_costs = [_compute_cost * 1.3] * len(_sizes)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_sizes.tolist(), y=_move_data_costs, mode="lines",
            name="Move Data (egress + compute)",
            line=dict(color=COLORS["RedLine"], width=2.5),
        ))
        _fig.add_trace(go.Scatter(
            x=_sizes.tolist(), y=_move_compute_costs, mode="lines",
            name="Move Compute (local provisioning)",
            line=dict(color=COLORS["GreenLine"], width=2.5),
        ))
        _fig.add_trace(go.Scatter(
            x=[_dataset_tb], y=[_move_data_cost], mode="markers",
            name=f"Your dataset: {_dataset_tb} TB",
            marker=dict(color=COLORS["BlueLine"], size=12, symbol="circle",
                        line=dict(color="white", width=2)),
        ))
        _fig.update_layout(
            height=320,
            xaxis=dict(title="Dataset Size (TB)", type="log", gridcolor="#f1f5f9"),
            yaxis=dict(title="Total Cost ($)", type="log", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md("### Cost Crossover: Move Data vs Move Compute"))
        items.append(mo.as_html(_fig))

        _cheaper = "Move Compute" if _move_compute_cost < _move_data_cost else "Move Data"
        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:14px; border:2px solid {COLORS['RedLine']}; border-radius:10px;
                        text-align:center; background:{COLORS['RedLL']}; flex:1;">
                <div style="color:{COLORS['RedLine']}; font-size:0.72rem; font-weight:700;">
                    Move Data Cost</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['RedLine']};">
                    ${_move_data_cost:,.0f}</div>
                <div style="font-size:0.68rem; color:#94a3b8;">
                    egress: ${_egress_cost:,.0f}</div>
            </div>
            <div style="padding:14px; border:2px solid {COLORS['GreenLine']}; border-radius:10px;
                        text-align:center; background:{COLORS['GreenLL']}; flex:1;">
                <div style="color:{COLORS['GreenLine']}; font-size:0.72rem; font-weight:700;">
                    Move Compute Cost</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['GreenLine']};">
                    ${_move_compute_cost:,.0f}</div>
                <div style="font-size:0.68rem; color:#94a3b8;">30% provisioning premium</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['BlueLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Transfer Time</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_transfer_hours:.1f} hrs</div>
                <div style="font-size:0.68rem; color:#94a3b8;">at {_bw_gbps} Gbps</div>
            </div>
        </div>
        """))

        _pred = partB_prediction.value
        _ref_hours = 50 * 1000 * 8 / (10 * 3600)
        if _pred == "11hr":
            items.append(mo.callout(mo.md(
                f"**Correct.** 50 TB / 10 Gbps = {_ref_hours:.1f} hours. "
                "And that is just transfer time -- the $4,000 egress cost is the real shock."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**50 TB / 10 Gbps = {_ref_hours:.1f} hours.** "
                "Most students dramatically underestimate because they think in small-file terms."
            ), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Data Transfer Time & Egress Cost": mo.md("""
**Formulas:**
$$
t_{\\text{transfer}} = \\frac{D_{\\text{dataset}}}{\\text{BW}_{\\text{net}}}
\\qquad
C_{\\text{egress}} = D_{\\text{dataset}} \\times p_{\\text{egress}}
$$

**Variables:**
- **$D_{\\text{dataset}}$**: dataset size (bytes); convert TB to bits: $\\times 8 \\times 10^{12}$
- **$\\text{BW}_{\\text{net}}$**: network bandwidth (bits/s); 10 Gbps = $10 \\times 10^9$ bps
- **$t_{\\text{transfer}}$**: wall-clock transfer time (seconds)
- **$p_{\\text{egress}}$**: cloud egress price (~\\$0.08/GB for major providers)
- **$C_{\\text{egress}}$**: total egress cost

For 50 TB at 10 Gbps: $t = 50 \\times 10^{12} \\times 8 / (10 \\times 10^9) \\approx 11.1$ hours.
Egress cost: $50,000 \\text{ GB} \\times \\$0.08 = \\$4,000$. Above ~5 TB, moving compute to
data is cheaper than moving data to compute.
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════
    # PART C -- Data Cascades
    # ═════════════════════════════════════════════════════════════════════

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incident Report &middot; Data Pipeline Failure
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;A schema change in our upstream data provider dropped leading zeros
                from zip codes. We did not notice for 4 weeks. Model accuracy has dropped
                from 92% to 78%. How did a 2% data error cause a 15% accuracy drop?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; ML Ops post-mortem, Week 4
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Data Cascades: The 2% Error That Ate 15% Accuracy

        Errors at early pipeline stages amplify through downstream stages.
        Use the formula to estimate the output error before making your prediction:

        ```
        error_stage_n = error_0 * amplification_factor^n
        ```

        Plus the 4-week detection delay: silent degradation on every request.
        """))

        items.append(partC_prediction)

        if partC_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the cascade simulator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partC_error_rate, partC_stages], justify="start", gap="2rem"))

        _error_pct = partC_error_rate.value
        _stages = partC_stages.value
        _amp_factor = 1.4  # from chapter

        _errors_by_stage = []
        for _s in range(_stages + 1):
            _err = _error_pct * (_amp_factor ** _s)
            _errors_by_stage.append(min(_err, 100))

        _output_error = _errors_by_stage[-1]
        _amplification = _output_error / _error_pct if _error_pct > 0 else 1

        # Cascade chart
        _stage_labels = [f"Stage {i}" for i in range(_stages + 1)]
        _stage_labels[0] = "Ingestion"
        _stage_labels[-1] = "Output"

        _bar_colors = [COLORS["GreenLine"] if e < 5 else
                       COLORS["OrangeLine"] if e < 15 else
                       COLORS["RedLine"] for e in _errors_by_stage]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=_stage_labels, y=_errors_by_stage,
            marker_color=_bar_colors, opacity=0.85,
            text=[f"{e:.1f}%" for e in _errors_by_stage],
            textposition="outside",
        ))
        _fig.update_layout(
            height=320,
            yaxis=dict(title="Error Rate (%)", gridcolor="#f1f5f9"),
            margin=dict(l=50, r=20, t=30, b=80),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md("### Error Amplification Through Pipeline Stages"))
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['GreenLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Input Error</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['GreenLine']};">
                    {_error_pct:.1f}%</div>
            </div>
            <div style="padding:14px; border:2px solid {COLORS['RedLine']}; border-radius:10px;
                        text-align:center; background:{COLORS['RedLL']}; flex:1;">
                <div style="color:{COLORS['RedLine']}; font-size:0.72rem; font-weight:700;">
                    Output Error</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['RedLine']};">
                    {_output_error:.1f}%</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['OrangeLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Amplification</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {_amplification:.1f}x</div>
            </div>
        </div>
        """))

        # Detection timeline
        items.append(mo.Html(f"""
        <div style="background:{COLORS['RedLL']}; border:1px solid {COLORS['RedLine']};
                    border-radius:10px; padding:16px 20px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; margin-bottom:8px;">
                Silent Degradation Window</div>
            <div style="font-size:0.88rem; color:{COLORS['Text']}; line-height:1.6;">
                <strong>Detection latency:</strong> median 4 weeks (Sambasivan et al. 2021).<br/>
                During this window, the model silently makes degraded predictions on
                <strong>every single request</strong>. At 1,000 requests/day, that is
                28,000 degraded predictions before anyone notices.
            </div>
        </div>
        """))

        _pred = partC_prediction.value
        if _pred == "15":
            items.append(mo.callout(mo.md(
                f"**Correct.** A 2% error amplifies to ~{_output_error:.0f}% over "
                f"{_stages} stages (amplification factor {_amp_factor}). "
                "Plus the 4-week detection delay."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Errors compound, they do not pass through linearly.** "
                f"2% input -> {_output_error:.1f}% output ({_amplification:.1f}x amplification). "
                "Data contracts and schema validation prevent this."
            ), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Error Cascade Amplification": mo.md("""
**Formula:**
$$
\\epsilon_{\\text{out}} = 1 - (1 - \\epsilon_{\\text{in}})^{k}
$$

**Variables:**
- **$\\epsilon_{\\text{in}}$**: ingestion error rate (fraction, e.g., 0.02 for 2%)
- **$k$**: number of pipeline stages (each stage compounds the error)
- **$\\epsilon_{\\text{out}}$**: output error rate after all stages

**Amplification factor:**
$$
A = \\frac{\\epsilon_{\\text{out}}}{\\epsilon_{\\text{in}}} = \\frac{1 - (1 - \\epsilon_{\\text{in}})^{k}}{\\epsilon_{\\text{in}}}
$$

For $\\epsilon_{\\text{in}} = 0.02$ and $k = 5$: $\\epsilon_{\\text{out}} = 1 - 0.98^5 \\approx 0.096$ (9.6%),
with an amplification factor of ~4.8x. Combined with downstream model sensitivity,
a 2% ingestion error can degrade accuracy by ~15%.
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════
    # PART D -- The False Positive Trap
    # ═════════════════════════════════════════════════════════════════════

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Product Spec &middot; Smart Speaker KWS
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our always-on smart speaker evaluates 1-second audio windows continuously.
                The product requirement is: at most 1 false wake-up per month.
                Our model has 99% accuracy. Ship it?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Product Team, VoiceAI
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## When 99% Accuracy Is Not Enough

        An always-on device evaluates ~2.6 million windows per month
        (30 days * 24 hours * 3600 seconds). At that scale, "99% accuracy"
        means 26,000 false wakes per month.

        ```
        required_rejection = 1 - (tolerance / windows_per_month)
        ```
        """))

        items.append(partD_prediction)

        if partD_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the false positive calculator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partD_tolerance, partD_duty], justify="start", gap="2rem"))

        _tolerance = partD_tolerance.value
        _duty_hours = partD_duty.value

        _windows_per_month = _duty_hours * 3600 * 30
        _required_rejection = 1 - (_tolerance / _windows_per_month) if _windows_per_month > 0 else 0

        # Count nines
        _nines = -math.log10(1 - _required_rejection) if _required_rejection < 1 else 0

        # False wakes at various accuracy levels
        _acc_levels = [0.99, 0.999, 0.9999, 0.99999, 0.999999]
        _false_wakes = [_windows_per_month * (1 - a) for a in _acc_levels]

        _fig = go.Figure()
        _acc_labels = ["99%", "99.9%", "99.99%", "99.999%", "99.9999%"]
        _colors_bar = [COLORS["RedLine"] if fw > _tolerance else COLORS["GreenLine"]
                       for fw in _false_wakes]
        _fig.add_trace(go.Bar(
            x=_acc_labels, y=_false_wakes,
            marker_color=_colors_bar, opacity=0.85,
            text=[f"{fw:,.0f}" for fw in _false_wakes],
            textposition="outside",
        ))
        _fig.add_hline(y=_tolerance, line_dash="dash", line_color=COLORS["OrangeLine"],
                       line_width=1.5,
                       annotation_text=f"Tolerance: {_tolerance}/month",
                       annotation_font_color=COLORS["OrangeLine"])
        _fig.update_layout(
            height=320,
            yaxis=dict(title="False Wakes per Month", type="log", gridcolor="#f1f5f9"),
            xaxis=dict(title="Model Accuracy (rejection rate)"),
            margin=dict(l=50, r=20, t=30, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md("### False Wakes at Different Accuracy Levels"))
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['BlueLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Windows/Month</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_windows_per_month:,.0f}</div>
            </div>
            <div style="padding:14px; border:2px solid {COLORS['OrangeLine']}; border-radius:10px;
                        text-align:center; background:white; flex:1;">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Required Nines</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {_nines:.1f}</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['RedLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">
                    False Wakes at 99%</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['RedLine']};">
                    {_windows_per_month * 0.01:,.0f}</div>
            </div>
        </div>
        """))

        _pred = partD_prediction.value
        if _pred == "999999":
            items.append(mo.callout(mo.md(
                f"**Correct.** {_windows_per_month:,.0f} windows/month requires "
                f"~{_nines:.1f} nines of rejection to achieve {_tolerance} false wake(s)."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**{_windows_per_month:,.0f} windows per month means even 99.999% "
                f"produces {_windows_per_month * 0.00001:,.0f} false wakes.** "
                f"You need ~{_nines:.1f} nines for {_tolerance} false wake(s)/month."
            ), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Required Rejection Rate (Nines)": mo.md("""
**Formula:**
$$
\\text{Nines} = -\\log_{10}\\!\\left(\\frac{F_{\\text{tol}}}{N_{\\text{windows}}}\\right)
$$

**Variables:**
- **$N_{\\text{windows}}$**: total classification windows per month = $\\frac{3600}{t_{\\text{window}}} \\times H_{\\text{duty}} \\times 30$
- **$F_{\\text{tol}}$**: acceptable false activations per month (e.g., 1)
- **$t_{\\text{window}}$**: audio window duration (typically 1 second for keyword spotting)
- **$H_{\\text{duty}}$**: duty cycle (hours per day)

For a 24/7 device with 1-second windows: $N = 3600 \\times 24 \\times 30 = 2,592,000$ windows/month.
To achieve $\\leq 1$ false wake: Nines $= -\\log_{10}(1/2{,}592{,}000) \\approx 6.4$.
That is **six nines** of rejection -- far beyond typical "99% accuracy" claims.
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════

    def build_synthesis():
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
                        <strong>1. The feeding tax leaves GPUs idle &gt;95% of the time</strong>
                        on standard storage. The fix is faster I/O (NVMe + parallel workers),
                        not more GPUs.
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>2. Data gravity is real: $0.08/GB egress makes moving 50 TB cost
                        $4,000.</strong> Above ~5 TB, moving compute to the data is cheaper
                        than moving data to the compute.
                    </div>
                    <div>
                        <strong>3. Data errors amplify exponentially through pipelines.</strong>
                        A 2% ingestion error becomes ~15% accuracy degradation over 5 stages.
                        Always-on devices evaluate millions of windows per month, requiring
                        6+ nines of rejection rate.
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
                        <strong>Lab 05: Neural Computation</strong> -- Now that you understand
                        data constraints, Lab 05 dives into the compute side: activation functions
                        cost 50x more transistors than you think, and memory hierarchy cliffs
                        are step functions, not slopes.
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
                        <strong>Read:</strong> the Data Engineering chapter for the full treatment
                        of data pipelines, I/O bottlenecks, egress costs, and data cascades.
                        <br/><strong>Build:</strong> TinyTorch Module 04 -- implement a data loader with prefetching and pipeline profiling.
                    </div>
                </div>
            </div>
            """),
        ])

    # ── COMPOSE TABS ─────────────────────────────────────────────────────
    tabs = mo.ui.tabs({
        "Part A -- The Feeding Tax":            build_part_a(),
        "Part B -- Data Gravity":               build_part_b(),
        "Part C -- Data Cascades":              build_part_c(),
        "Part D -- The False Positive Trap":    build_part_d(),
        "Synthesis":                             build_synthesis(),
    })
    tabs
    return



# ===========================================================================
# ZONE D: LEDGER HUD
# ===========================================================================

@app.cell(hide_code=True)
def _(COLORS, ledger, mo, partA_prediction, partB_prediction, partC_prediction, partD_prediction):
    _track = ledger._state.track or "not set"
    if partA_prediction.value is not None and partD_prediction.value is not None:
        ledger.save(chapter=4, design={
            "chapter": "v1_04",
            "feeding_tax_prediction": partA_prediction.value,
            "data_gravity_prediction": partB_prediction.value,
            "data_cascade_prediction": partC_prediction.value,
            "false_positive_prediction": partD_prediction.value,
            "feeding_tax_correct": partA_prediction.value == "5",
            "data_gravity_correct": partB_prediction.value == "11hr",
            "data_cascade_correct": partC_prediction.value == "15",
            "false_positive_correct": partD_prediction.value == "999999",
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">04 &middot; The Data Gravity Trap</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">CHAPTER&nbsp;4</span>
        <span class="hud-value">Data Engineering</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
