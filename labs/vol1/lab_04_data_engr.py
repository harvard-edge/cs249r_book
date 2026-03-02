import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 04: THE DATA GRAVITY TRAP
#
# Chapter: data_engineering.qmd (@sec-data-engineering)
# Core Invariant: Data gravity — large datasets attract compute to their
#   location. Moving data costs more than moving compute when data volume
#   exceeds network bandwidth × time budget.
#
# 2-Act structure (35-40 min total):
#   Act I:  The Pipeline Bottleneck (12-15 min)
#     The GPU is idle 77% of the time — but the team wants more GPUs.
#     Prediction lock → timeline instrument → reveal → reflection → MathPeek
#   Act II: The Data Gravity Calculation (20-25 min)
#     50 TB in us-east-1, training GPUs in us-west-2. Transfer or co-locate?
#     Prediction lock → gravity instruments → failure state → reflection → MathPeek
#
# Deployment contexts:
#   Cloud: Multi-region (100 Gbps inter-DC link, AWS egress $0.08/GB)
#   Edge:  Local processing (1 Gbps LAN, zero egress cost)
#
# Traceability:
#   GPU utilization formula   — @sec-data-engineering-feeding-problem
#   Feeding tax               — FeedingProblem class in data_engineering.qmd
#   Data gravity T = D/BW     — DataGravity class in data_engineering.qmd
#   AWS egress $0.08/GB       — DataGravity.egress_cost_per_gb_str
#   100 Gbps = 12.5 GB/s      — DataGravity.network_gbs_str
#   HDD 0.15 GB/s             — Storage tier physics, @sec-data-engineering
#   SSD 0.55 GB/s             — Storage tier physics, @sec-data-engineering
#   NVMe 3.5 GB/s             — Storage tier physics, @sec-data-engineering
#   RAM 50 GB/s               — DRAM bandwidth, @sec-data-engineering
#
# Design Ledger save:
#   chapter=4, context, storage_type_chosen, gpu_util_at_start,
#   data_gravity_triggered, act1_correct, act2_correct
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible) ─────────────────────────
@app.cell
def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    ledger = DesignLedger()
    return mo, ledger, COLORS, LAB_CSS, apply_plotly_theme, go, np, math


# ─── CELL 1: HEADER ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _c = COLORS["BlueLine"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume I · Lab 04
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Data Gravity Trap
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.02rem; color: #94a3b8;
                      max-width: 660px; line-height: 1.65;">
                Your GPU is idle 77% of the time. The team wants to buy more hardware.
                Before spending another dollar, you need to diagnose whether the bottleneck
                is compute — or data movement.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: Pipeline Bottleneck · 12–15 min
                </span>
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act II: Data Gravity · 20–25 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min total
                </span>
                <span class="badge badge-info">Chapter 4: Data Engineering</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: RECOMMENDED READING ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-data-engineering-physics-data-cdcb** (Physics of Data) — Data gravity and
      the energy-movement invariant: why moving a bit costs 100–1,000x more than computing on it.
    - **@sec-data-engineering-feeding-problem** (The Feeding Problem) — The Feeding Tax,
      GPU utilization formula, and why storage bandwidth determines training throughput.
    - **@sec-data-engineering-data-gravity-adcb** (Data Gravity) — T = D/BW, the rule of
      thumb for when to move compute vs. data, and the lakehouse architectural response.
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Cloud (multi-region, 100 Gbps inter-DC)": "cloud",
            "Edge (local processing, 1 Gbps LAN)": "edge",
        },
        value="Cloud (multi-region, 100 Gbps inter-DC)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Select Your Deployment Context"),
        mo.md(
            "This choice persists across both acts. It changes the network bandwidth "
            "and egress cost assumptions in the data gravity calculations."
        ),
        context_toggle,
    ])
    return (context_toggle,)


@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS):
    _ctx = context_toggle.value
    if _ctx == "cloud":
        _bw_desc = "100 Gbps inter-region link = 12.5 GB/s"
        _cost_desc = "AWS egress: $0.08/GB"
        _color = COLORS["Cloud"]
        _label = "Cloud — Multi-Region"
    else:
        _bw_desc = "1 Gbps LAN = 0.125 GB/s"
        _cost_desc = "Local network: $0.00/GB"
        _color = COLORS["Edge"]
        _label = "Edge — Local Processing"

    mo.callout(mo.md(
        f"**Context: {_label}** — Network bandwidth: {_bw_desc}. Transfer cost: {_cost_desc}. "
        f"The data gravity calculation in Act II will use these parameters."
    ), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# ACT I: THE PIPELINE BOTTLENECK
# ═════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _c = COLORS["BlueLine"]
    mo.Html(f"""
    <div style="margin: 32px 0 8px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
                    text-transform: uppercase; color: {_c}; margin-bottom: 6px;">
            Act I · 12–15 minutes
        </div>
        <div style="font-size: 1.6rem; font-weight: 800; color: #0f172a; line-height: 1.2;">
            The Pipeline Bottleneck
        </div>
        <div style="font-size: 0.95rem; color: #475569; margin-top: 6px; max-width: 680px;">
            A team of ML engineers is training a computer vision model. Their GPU utilization
            has been sitting at 23% for three days. Before you touch a single slider,
            you need to commit to a diagnosis.
        </div>
    </div>
    """)
    return


# ─── ACT I: STAKEHOLDER MESSAGE ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _c = COLORS["OrangeLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_c}; background: {COLORS['OrangeLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_c};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · ML Engineering Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We've been training ResNet-50 on a 10 TB image dataset for 3 days.
            GPU utilization is stuck at 23%. The model is clearly too complex for
            our hardware — we need to request 4× more GPUs before Friday's deadline.
            Can you sign off on the procurement request?"
        </div>
    </div>
    """)
    return


# ─── ACT I: PREDICTION LOCK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Your Prediction

    *Before interacting with the simulator, commit to your diagnosis.*
    GPU utilization is 23% during training. The team believes this means the GPU
    cannot keep up with the model's compute demands.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) Insufficient GPU FLOPS — the model requires more compute than the GPU provides": "A",
            "B) Data loading is slower than GPU computation — the GPU is starving for input": "B",
            "C) The model is too small — it does not utilize the GPU's parallel units": "C",
            "D) The learning rate is too high — training is unstable and wasting cycles": "D",
        },
        label="GPU utilization is 23% during training. The most likely bottleneck is:",
    )
    act1_prediction
    return (act1_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction to unlock the pipeline simulator."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(
            f"**Prediction locked: {act1_prediction.value[:2]}** "
            "Now explore the simulator to test your hypothesis."
        ),
        kind="info",
    )
    return


# ─── ACT I: INSTRUMENTS ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### The Pipeline Simulator

    Adjust the parameters below to see how storage type and data loading
    configuration affect GPU utilization. The timeline shows where time goes
    within a single training batch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_dataset_size_tb = mo.ui.slider(
        start=1, stop=100, value=10, step=1,
        label="Dataset size (TB)",
        show_value=True,
    )
    act1_storage_type = mo.ui.dropdown(
        options={
            "HDD (0.15 GB/s)": "hdd",
            "SSD (0.55 GB/s)": "ssd",
            "NVMe (3.5 GB/s)": "nvme",
            "RAM disk (50 GB/s)": "ram",
        },
        value="HDD (0.15 GB/s)",
        label="Storage type",
    )
    act1_num_workers = mo.ui.slider(
        start=1, stop=32, value=4, step=1,
        label="DataLoader workers",
        show_value=True,
    )
    mo.hstack([
        act1_dataset_size_tb,
        act1_storage_type,
        act1_num_workers,
    ], gap=2, justify="start")
    return (act1_dataset_size_tb, act1_storage_type, act1_num_workers)


@app.cell(hide_code=True)
def _(mo, act1_storage_type, act1_num_workers, act1_dataset_size_tb, go, apply_plotly_theme, COLORS):
    # ── Physics engine ────────────────────────────────────────────────────────
    # Source: @sec-data-engineering-feeding-problem
    # GPU compute time per batch (ResNet-50, batch=64, A100)
    # ResNet-50: 4.1 GFLOPs per image, A100: 312 TFLOPS FP32
    # t_compute = (64 * 4.1e9) / (312e12) ≈ 0.00084 s ≈ 0.84 ms
    _BATCH_SIZE = 64
    _RESNET50_GFLOPS_PER_IMG = 4.1  # GFLOPs
    _A100_TFLOPS_FP32 = 312.0       # TFLOPS (A100 FP32 tensor, NVIDIA spec)
    _IMG_SIZE_MB = (224 * 224 * 3 * 4) / (1024 * 1024)  # 224x224 RGB FP32

    _t_compute_s = (_BATCH_SIZE * _RESNET50_GFLOPS_PER_IMG * 1e9) / (_A100_TFLOPS_FP32 * 1e12)

    # Storage bandwidth (GB/s) — source: @sec-data-engineering storage tier data
    _storage_bw = {
        "hdd":  0.15,   # HDD sequential read, GB/s
        "ssd":  0.55,   # SATA SSD, GB/s
        "nvme": 3.5,    # NVMe PCIe 4.0, GB/s
        "ram":  50.0,   # DRAM, GB/s
    }
    _bw = _storage_bw[act1_storage_type.value]

    # Effective bandwidth scales with num_workers (diminishing returns after ~8)
    _worker_factor = min(act1_num_workers.value, 8) / 4.0
    _effective_bw = _bw * min(_worker_factor, 2.0)

    # I/O time per batch: bytes to load / effective_bandwidth
    _batch_bytes_gb = _BATCH_SIZE * _IMG_SIZE_MB / 1024
    _t_io_s = _batch_bytes_gb / _effective_bw

    # Preprocessing: fixed 0.2× of IO time (decode, augment)
    _t_preprocess_s = _t_io_s * 0.2

    # Total batch time and GPU utilization
    # GPU util = t_compute / (t_compute + max(t_io - t_compute, 0) + t_preprocess)
    # When IO > compute, GPU waits. When compute > IO, pipeline overlaps.
    _t_wait = max(_t_io_s - _t_compute_s, 0.0)
    _t_total = _t_compute_s + _t_wait + _t_preprocess_s
    _gpu_util = min(_t_compute_s / _t_total, 1.0) * 100.0

    # Color coding for utilization
    if _gpu_util >= 80:
        _util_color = COLORS["GreenLine"]
        _util_label = "Healthy"
    elif _gpu_util >= 50:
        _util_color = COLORS["OrangeLine"]
        _util_label = "Degraded"
    else:
        _util_color = COLORS["RedLine"]
        _util_label = "Starved"

    # Scale to ms for display
    _t_compute_ms = _t_compute_s * 1000
    _t_io_ms = _t_io_s * 1000
    _t_preprocess_ms = _t_preprocess_s * 1000
    _t_total_ms = _t_total * 1000

    # ── Timeline bar chart ────────────────────────────────────────────────────
    _fig = go.Figure()

    _fig.add_trace(go.Bar(
        name="GPU Compute",
        x=[_t_compute_ms],
        y=["Batch Timeline"],
        orientation="h",
        marker_color=COLORS["GreenLine"],
        text=[f"GPU Compute<br>{_t_compute_ms:.2f} ms"],
        textposition="inside",
        insidetextanchor="middle",
    ))
    _fig.add_trace(go.Bar(
        name="Data Loading (I/O)",
        x=[max(_t_io_ms - _t_compute_ms, 0)],
        y=["Batch Timeline"],
        orientation="h",
        marker_color=COLORS["RedLine"],
        text=[f"I/O Wait<br>{max(_t_io_ms - _t_compute_ms, 0):.2f} ms"],
        textposition="inside",
        insidetextanchor="middle",
    ))
    _fig.add_trace(go.Bar(
        name="Preprocessing",
        x=[_t_preprocess_ms],
        y=["Batch Timeline"],
        orientation="h",
        marker_color=COLORS["OrangeLine"],
        text=[f"Preprocess<br>{_t_preprocess_ms:.2f} ms"],
        textposition="inside",
        insidetextanchor="middle",
    ))

    _fig.update_layout(
        barmode="stack",
        height=160,
        xaxis_title="Time per batch (ms)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=40),
        title=dict(text=f"Batch Timeline — Total: {_t_total_ms:.2f} ms", font=dict(size=13)),
    )
    apply_plotly_theme(_fig)

    # ── Metric cards ──────────────────────────────────────────────────────────
    _cards_html = f"""
    <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0;">
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 160px; text-align: center; background: white;
                    border-top: 4px solid {_util_color};">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                GPU Utilization
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {_util_color}; line-height: 1;">
                {_gpu_util:.0f}%
            </div>
            <div style="color: {_util_color}; font-size: 0.78rem; font-weight: 700; margin-top: 4px;">
                {_util_label}
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 160px; text-align: center; background: white;">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                GPU Compute
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {COLORS['GreenLine']}; line-height: 1;">
                {_t_compute_ms:.2f} ms
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">per batch</div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 160px; text-align: center; background: white;">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                I/O Wait
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {_util_color}; line-height: 1;">
                {_t_io_ms:.2f} ms
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">per batch</div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 160px; text-align: center; background: white;">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                Storage BW
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {COLORS['BlueLine']}; line-height: 1;">
                {_effective_bw:.2f} GB/s
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
                {act1_num_workers.value} workers
            </div>
        </div>
    </div>
    """

    # ── Physics formula display ───────────────────────────────────────────────
    _formula_md = f"""
    **The physics (from @sec-data-engineering-feeding-problem):**

    ```
    GPU Utilization = t_compute / (t_compute + t_io_wait + t_preprocess)
                    = {_t_compute_ms:.2f} ms / ({_t_compute_ms:.2f} + {max(_t_io_ms - _t_compute_ms, 0):.2f} + {_t_preprocess_ms:.2f}) ms
                    = {_gpu_util:.1f}%

    Effective BW    = Storage_BW × worker_factor
                    = {_bw:.2f} GB/s × {min(act1_num_workers.value, 8) / 4.0:.2f}
                    = {_effective_bw:.2f} GB/s

    I/O Time        = batch_bytes / effective_BW
                    = {_batch_bytes_gb * 1024:.1f} MB / {_effective_bw * 1024:.0f} MB/s
                    = {_t_io_ms:.2f} ms
    ```
    """

    mo.vstack([
        mo.Html(_cards_html),
        mo.as_html(_fig),
        mo.md(_formula_md),
    ])
    return (
        _gpu_util,
        _t_compute_ms,
        _t_io_ms,
        _effective_bw,
        _bw,
    )


# ─── ACT I: REVEAL ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction, _gpu_util):
    # Prediction-vs-reality overlay
    _pred_text = {
        "A": "GPU FLOPS insufficient",
        "B": "Data loading bottleneck (I/O bound)",
        "C": "Model too small",
        "D": "Learning rate too high",
    }[act1_prediction.value]

    _is_correct = act1_prediction.value == "B"
    _actual_gpu_util = _gpu_util

    if _is_correct:
        mo.callout(mo.md(
            f"**Correct. You predicted: {_pred_text}.**\n\n"
            f"The GPU is sitting at **{_actual_gpu_util:.0f}% utilization** not because "
            "it lacks FLOPS, but because it finishes each batch computation in "
            "~0.84 ms while the HDD requires ~3.6 ms to load the next batch. "
            "The GPU is I/O-bound: it spends 77% of its wall-clock time waiting "
            "for the data pipeline to deliver the next 64 images. "
            "Adding more GPUs would make the problem worse — each additional GPU "
            "would compete for the same storage bandwidth."
        ), kind="success")
    elif act1_prediction.value == "A":
        mo.callout(mo.md(
            f"**Not quite. You predicted: {_pred_text}.**\n\n"
            f"The GPU is at **{_actual_gpu_util:.0f}% utilization** but this is not "
            "because it lacks FLOPS. An A100 can process a ResNet-50 batch in ~0.84 ms. "
            "The bottleneck is that an HDD delivers only 0.15 GB/s — loading the same "
            "batch takes ~3.6 ms. The GPU completes its work, then waits. "
            "The 'fix' of adding GPUs would only increase I/O contention. "
            "**Correct answer: B — the pipeline is I/O-bound.**"
        ), kind="warn")
    elif act1_prediction.value == "C":
        mo.callout(mo.md(
            f"**Not quite. You predicted: {_pred_text}.**\n\n"
            f"The GPU is at **{_actual_gpu_util:.0f}% utilization** because it is waiting "
            "for data, not because the model is too simple. ResNet-50 requires 4.1 GFLOPs "
            "per image — this is not a trivial model. Even a 50-layer ResNet finishes its "
            "batch in 0.84 ms on an A100, which is then idle for 2.76 ms waiting for the "
            "HDD. Model complexity is irrelevant when the bottleneck is I/O. "
            "**Correct answer: B — the pipeline is I/O-bound.**"
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**Not quite. You predicted: {_pred_text}.**\n\n"
            f"Learning rate affects convergence quality, not hardware utilization. "
            f"The GPU is at **{_actual_gpu_util:.0f}% utilization** because it finishes "
            "computing in 0.84 ms and then idles for 2.76 ms waiting for the storage "
            "system to load the next batch. This is a physical bottleneck in the "
            "data pipeline — it has nothing to do with the optimization algorithm. "
            "**Correct answer: B — the pipeline is I/O-bound.**"
        ), kind="warn")
    return


# ─── ACT I: REFLECTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Reflection

    You have seen that HDD storage produces 23% GPU utilization while NVMe reaches 78%.
    Now commit to the correct engineering response.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) Add more GPUs to the training cluster": "A",
            "B) Increase batch size from 64 to 512 to reduce I/O frequency": "B",
            "C) Switch to faster storage or increase num_workers to saturate the pipeline": "C",
            "D) Reduce model size to lower compute time per batch": "D",
        },
        label="What is the correct fix for 23% GPU utilization caused by data loading?",
    )
    act1_reflection
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Select your answer to continue."), kind="warn"),
    )

    if act1_reflection.value == "C":
        mo.callout(mo.md(
            "**Correct.** Faster storage (NVMe: 3.5 GB/s vs HDD: 0.15 GB/s) eliminates "
            "I/O wait directly. More DataLoader workers parallelize reads across multiple "
            "CPU cores, increasing effective bandwidth. Both approaches attack the same "
            "root cause: insufficient I/O throughput relative to GPU compute speed. "
            "Adding GPUs (A) worsens the I/O competition. Larger batches (B) reduce "
            "the *number* of I/O operations but each operation loads more data, so total "
            "I/O time increases proportionally. Smaller models (D) reduce compute time, "
            "which actually *lowers* utilization further by making the GPU finish even faster."
        ), kind="success")
    elif act1_reflection.value == "A":
        mo.callout(mo.md(
            "**Incorrect.** Adding GPUs distributes the compute load — but the storage "
            "bottleneck is shared. Each new GPU would compete for the same HDD bandwidth "
            "(0.15 GB/s), reducing the effective bandwidth per GPU. You would spend more "
            "money and achieve lower per-GPU utilization. The correct fix is C: faster "
            "storage or more DataLoader workers."
        ), kind="warn")
    elif act1_reflection.value == "B":
        mo.callout(mo.md(
            "**Partially helpful, but not the root fix.** Larger batches reduce the "
            "*number* of I/O calls per epoch, but each call loads 8× more data (512 vs 64 "
            "images). Total I/O bytes per epoch is unchanged, so the Feeding Tax "
            "remains proportionally similar. The root cause — storage bandwidth below "
            "what the GPU needs — requires faster storage (C). Larger batches also "
            "affect gradient statistics and may require learning rate adjustments."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Incorrect.** A smaller model finishes each batch computation even faster, "
            "making the GPU idle for even longer while waiting for I/O. This worsens "
            "the utilization metric. The problem is that storage is too slow, not that "
            "the model is too slow. The correct fix is C: faster storage or more workers."
        ), kind="warn")
    return


# ─── ACT I: MATHPEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — GPU Utilization and the Feeding Tax": mo.md("""
        **GPU Utilization** (from @sec-data-engineering-feeding-problem):

        $$\\eta_{GPU} = \\frac{t_{compute}}{t_{compute} + t_{IO\\ wait} + t_{preprocess}}$$

        Where:
        - **t_compute** — time the GPU is executing the forward/backward pass
        - **t_IO wait** — time the GPU idles waiting for the next batch from storage
        - **t_preprocess** — time for CPU-side decode and augmentation

        **The Feeding Tax** is the complement: `Feeding Tax = (1 - η_GPU) × 100%`

        When storage bandwidth (BW_storage) is less than the GPU's required bandwidth
        (BW_required = batch_bytes × GPU_throughput), the pipeline stalls:

        $$t_{IO\\ wait} = \\frac{batch\\ bytes}{BW_{storage}} - t_{compute}$$
        $$\\text{(positive when storage is the bottleneck, zero when GPU is the bottleneck)}$$

        **Numerical example (10 TB dataset, HDD, 4 workers):**
        ```
        t_compute   = (64 × 4.1 GFLOPs) / 312 TFLOPS ≈ 0.84 ms
        t_IO        = (64 × 600 KB) / 150 MB/s ≈ 2.56 ms
        t_IO_wait   = 2.56 - 0.84 = 1.72 ms  (GPU is idle)
        t_preprocess = 2.56 × 0.2 = 0.51 ms
        η_GPU       = 0.84 / (0.84 + 1.72 + 0.51) ≈ 27%  ← near the 23% observation
        ```

        **Pipeline overlap** occurs when t_IO < t_compute: prefetching can hide I/O
        latency and η_GPU → 100%. NVMe at 3.5 GB/s achieves this for ResNet-50.
        """),
    })
    return


# ═════════════════════════════════════════════════════════════════════════════
# ACT II: THE DATA GRAVITY CALCULATION
# ═════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _c = COLORS["Cloud"]
    mo.Html(f"""
    <div style="margin: 40px 0 8px 0; border-top: 2px solid #e2e8f0; padding-top: 32px;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
                    text-transform: uppercase; color: {_c}; margin-bottom: 6px;">
            Act II · 20–25 minutes
        </div>
        <div style="font-size: 1.6rem; font-weight: 800; color: #0f172a; line-height: 1.2;">
            The Data Gravity Calculation
        </div>
        <div style="font-size: 0.95rem; color: #475569; margin-top: 6px; max-width: 700px;">
            You have fixed the training pipeline. Now a new problem: the team's 50 TB
            training dataset lives in AWS us-east-1. The GPUs they need are available
            in us-west-2. Should they move the data, or move the compute?
        </div>
    </div>
    """)
    return


# ─── ACT II: STAKEHOLDER MESSAGE ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _c = COLORS["BlueLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_c}; background: {COLORS['BlueLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_c};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · Cloud Infrastructure Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We have 50 TB of training data in us-east-1 and a cluster of 8 A100s
            available in us-west-2. We have a 100 Gbps inter-region link.
            The training run takes about 6 hours. Should we transfer the data
            to us-west-2 first, or spin up compute in us-east-1?"
        </div>
    </div>
    """)
    return


# ─── ACT II: PREDICTION LOCK ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Your Prediction

    *Commit to an estimate before using the calculator.*
    You have a 50 TB dataset in AWS us-east-1. Training GPUs are in us-west-2.
    The inter-region link runs at 100 Gbps. How long does the data transfer take?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_prediction = mo.ui.radio(
        options={
            "A) About 1 minute — 100 Gbps is very fast": "A",
            "B) About 67 minutes — 100 Gbps = 12.5 GB/s, 50 TB ÷ 12.5 GB/s ≈ 67 min": "B",
            "C) About 11 hours — the practical throughput is much lower than the rated speed": "C",
            "D) About 4.6 days — transfer overhead and routing make 100 Gbps unusable": "D",
        },
        label="Transfer time for 50 TB over a 100 Gbps inter-region link:",
    )
    act2_prediction
    return (act2_prediction,)


@app.cell(hide_code=True)
def _(mo, act2_prediction):
    mo.stop(
        act2_prediction.value is None,
        mo.callout(mo.md("Select your transfer time estimate to unlock the data gravity calculator."), kind="warn"),
    )
    mo.callout(
        mo.md(
            f"**Prediction locked: {act2_prediction.value[:2]}** "
            "Now use the calculator to determine whether your estimate was correct — "
            "and more importantly, whether transferring is the right decision at all."
        ),
        kind="info",
    )
    return


# ─── ACT II: INSTRUMENTS ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### The Data Gravity Calculator

    Adjust the dataset size and network parameters to see when data transfer
    exceeds your training time budget — and when it becomes cheaper to move
    the compute instead of the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_dataset_gb = mo.ui.slider(
        start=1, stop=6,
        value=4,  # log10(50 TB in GB) = log10(51200) ≈ 4.7, use 4 as default (10 TB)
        step=1,
        label="Dataset size (log10 GB) — 10¹ = 10 GB, 10⁶ = 1 PB",
        show_value=True,
    )
    act2_network_bw = mo.ui.dropdown(
        options={
            "1 Gbps Ethernet (0.125 GB/s)": "1g",
            "10 Gbps Ethernet (1.25 GB/s)": "10g",
            "100 Gbps Ethernet (12.5 GB/s)": "100g",
        },
        value="100 Gbps Ethernet (12.5 GB/s)",
        label="Network bandwidth",
    )
    act2_training_hours = mo.ui.slider(
        start=1, stop=72, value=6, step=1,
        label="Training time budget (hours)",
        show_value=True,
    )
    mo.hstack([
        act2_dataset_gb,
        act2_network_bw,
        act2_training_hours,
    ], gap=2, justify="start")
    return (act2_dataset_gb, act2_network_bw, act2_training_hours)


@app.cell(hide_code=True)
def _(mo, act2_dataset_gb, act2_network_bw, act2_training_hours, context_toggle, go, apply_plotly_theme, COLORS, math):
    # ── Physics engine ────────────────────────────────────────────────────────
    # Source: DataGravity class in data_engineering.qmd
    # T_transfer = D_vol / BW  (from @sec-data-engineering-data-gravity-adcb)
    # AWS egress: $0.08/GB     (DataGravity.egress_cost_per_gb_str)

    _dataset_gb_val = 10 ** act2_dataset_gb.value

    # Network bandwidth (GB/s) by tier
    _net_bw_map = {
        "1g":   0.125,   # 1 Gbps = 0.125 GB/s
        "10g":  1.25,    # 10 Gbps = 1.25 GB/s
        "100g": 12.5,    # 100 Gbps = 12.5 GB/s
    }
    _net_bw_gbs = _net_bw_map[act2_network_bw.value]

    # Context-dependent cost: cloud has egress, edge has none
    _ctx = context_toggle.value
    if _ctx == "cloud":
        _egress_cost_per_gb = 0.08   # AWS egress, DataGravity.egress_cost_per_gb_str
    else:
        _egress_cost_per_gb = 0.00   # local LAN, no egress fee

    # Transfer calculations
    _transfer_seconds = _dataset_gb_val / _net_bw_gbs
    _transfer_hours = _transfer_seconds / 3600
    _transfer_cost = _dataset_gb_val * _egress_cost_per_gb

    # Training budget
    _training_hours = act2_training_hours.value

    # Failure state: transfer > training budget
    _transfer_exceeds_training = _transfer_hours > _training_hours

    # Decision metric: compare transfer cost to compute-in-place cost
    # Approximate: spinning up equivalent compute in source region
    # A100 spot price ~$2.50/GPU-hr, 8 GPUs
    _compute_spot_cost_per_hour = 20.0  # $20/hr for 8× A100 spot
    _compute_cost_to_stay = _compute_spot_cost_per_hour * _training_hours
    _total_transfer_cost = _transfer_cost  # (ignoring compute differential for clarity)

    # Format dataset size for display
    if _dataset_gb_val >= 1e6:
        _ds_label = f"{_dataset_gb_val/1e6:.1f} PB"
    elif _dataset_gb_val >= 1e3:
        _ds_label = f"{_dataset_gb_val/1e3:.1f} TB"
    else:
        _ds_label = f"{_dataset_gb_val:.0f} GB"

    # Format transfer time for display
    if _transfer_hours >= 24:
        _time_label = f"{_transfer_hours/24:.1f} days"
    elif _transfer_hours >= 1:
        _time_label = f"{_transfer_hours:.1f} hours"
    else:
        _time_label = f"{_transfer_hours * 60:.0f} minutes"

    # ── Bar chart: transfer time vs training budget ───────────────────────────
    _transfer_color = COLORS["RedLine"] if _transfer_exceeds_training else COLORS["GreenLine"]
    _train_color = COLORS["BlueLine"]

    _fig2 = go.Figure()
    _fig2.add_trace(go.Bar(
        name="Data Transfer",
        x=["Time Comparison (hours)"],
        y=[_transfer_hours],
        marker_color=_transfer_color,
        text=[f"{_transfer_hours:.1f}h"],
        textposition="outside",
        width=0.3,
    ))
    _fig2.add_trace(go.Bar(
        name="Training Budget",
        x=["Time Comparison (hours)"],
        y=[_training_hours],
        marker_color=_train_color,
        text=[f"{_training_hours}h budget"],
        textposition="outside",
        width=0.3,
    ))
    _fig2.update_layout(
        barmode="group",
        height=280,
        yaxis_title="Hours",
        showlegend=True,
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text=f"Transfer vs Training: {_ds_label} over {act2_network_bw.value[:6]}", font=dict(size=13)),
    )
    apply_plotly_theme(_fig2)

    # ── Metric cards ──────────────────────────────────────────────────────────
    _cost_color = COLORS["RedLine"] if _transfer_cost > _compute_cost_to_stay else COLORS["GreenLine"]

    _cards2_html = f"""
    <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0;">
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 155px; text-align: center; background: white;
                    border-top: 4px solid {_transfer_color};">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                Transfer Time
            </div>
            <div style="font-size: 1.9rem; font-weight: 800; color: {_transfer_color}; line-height: 1;">
                {_time_label}
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
                {_ds_label} at {_net_bw_gbs} GB/s
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 155px; text-align: center; background: white;
                    border-top: 4px solid {_cost_color};">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                Egress Cost
            </div>
            <div style="font-size: 1.9rem; font-weight: 800; color: {_cost_color}; line-height: 1;">
                ${_transfer_cost:,.0f}
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
                @ ${_egress_cost_per_gb:.2f}/GB
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 155px; text-align: center; background: white;
                    border-top: 4px solid {COLORS['BlueLine']};">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                Compute-in-Place
            </div>
            <div style="font-size: 1.9rem; font-weight: 800; color: {COLORS['BlueLine']}; line-height: 1;">
                ${_compute_cost_to_stay:,.0f}
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
                8× A100 spot × {_training_hours}h
            </div>
        </div>
    </div>
    """

    # ── Physics formula display ───────────────────────────────────────────────
    _formula2_md = f"""
    **The physics (from @sec-data-engineering-data-gravity-adcb):**

    ```
    T_transfer  = D_vol / BW
                = {_dataset_gb_val:,.0f} GB / {_net_bw_gbs} GB/s
                = {_transfer_seconds:,.0f} s
                = {_time_label}

    Egress cost = D_vol × $0.08/GB
                = {_dataset_gb_val:,.0f} × $0.08
                = ${_transfer_cost:,.0f}
    ```
    """

    mo.vstack([
        mo.Html(_cards2_html),
        mo.as_html(_fig2),
        mo.md(_formula2_md),
    ])
    return (
        _transfer_exceeds_training,
        _transfer_hours,
        _training_hours,
        _transfer_cost,
        _compute_cost_to_stay,
        _time_label,
        _ds_label,
        _transfer_color,
    )


# ─── ACT II: FAILURE STATE ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, _transfer_exceeds_training, _transfer_hours, _training_hours, _time_label, _ds_label):
    if _transfer_exceeds_training:
        mo.callout(mo.md(
            f"**Data transfer ({_time_label}) exceeds training budget ({_training_hours} hours). "
            f"Moving {_ds_label} over the network costs more time than the training run itself.**\n\n"
            "This is the **data gravity trap**: at this scale, compute must move to the data. "
            "Options: (1) spin up training in the same region as the data, "
            "(2) use a Data Lakehouse — run the training job directly on the storage node, "
            "or (3) upgrade to a faster network link. "
            "Pull the dataset size or training budget slider to find the breakeven point."
        ), kind="danger")
    else:
        _ratio = _transfer_hours / _training_hours
        if _ratio > 0.5:
            mo.callout(mo.md(
                f"**Transfer feasible but costly: {_time_label} is {_ratio*100:.0f}% of your "
                f"{_training_hours}-hour training budget.**\n\n"
                "Data is approaching the gravity threshold. A small increase in dataset size "
                "or reduction in training time will trigger the trap. Consider co-locating "
                "compute with data as a proactive architectural decision."
            ), kind="warn")
        else:
            mo.callout(mo.md(
                f"**Transfer is viable: {_time_label} is well within the {_training_hours}-hour budget.**\n\n"
                "At this scale, data transfer is not the bottleneck. Data gravity has not yet "
                "trapped this workload. Increase the dataset size to find where the physics "
                "forces the architectural switch from 'move data' to 'move compute.'"
            ), kind="success")
    return


# ─── ACT II: PREDICTION REVEAL ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction):
    # 50 TB at 100 Gbps: 50,000 GB / 12.5 GB/s = 4,000 s = 66.7 min ≈ 67 min
    _actual_minutes = 67
    _actual_cost = 50_000 * 0.08  # 50 TB × $0.08/GB = $4,000

    _predicted = {
        "A": 1,
        "B": 67,
        "C": 660,
        "D": 6624,
    }[act2_prediction.value]

    _ratio = _actual_minutes / _predicted if _predicted > 0 else float("inf")
    _is_correct = act2_prediction.value == "B"

    if _is_correct:
        mo.callout(mo.md(
            f"**Correct. You predicted ~{_predicted} minutes. The actual transfer time is ~{_actual_minutes} minutes.**\n\n"
            f"50 TB ÷ 12.5 GB/s = 4,000 s = **{_actual_minutes} minutes**. "
            f"Plus the egress cost: 50,000 GB × $0.08 = **$4,000**. "
            "The transfer is feasible for a 6-hour training run — but $4,000 in egress "
            "may exceed the cost of spinning up equivalent compute in us-east-1."
        ), kind="success")
    elif act2_prediction.value == "A":
        mo.callout(mo.md(
            f"**You were off by {_ratio:.0f}×. You predicted ~{_predicted} minute. "
            f"The actual transfer time is ~{_actual_minutes} minutes.**\n\n"
            "100 Gbps sounds fast, but it equals only 12.5 GB/s. Dividing 50 TB "
            "(= 50,000 GB) by 12.5 GB/s gives 4,000 seconds = **67 minutes**. "
            "Sustained 100 Gbps is rare in practice; real transfers are slower."
        ), kind="warn")
    elif act2_prediction.value == "C":
        mo.callout(mo.md(
            f"**You were off by {1/_ratio:.1f}×. You predicted ~{_predicted} minutes. "
            f"The actual transfer time is ~{_actual_minutes} minutes.**\n\n"
            "The calculation: 50 TB ÷ 12.5 GB/s = 4,000 s = **67 minutes**. "
            "A sustained 100 Gbps connection is fast enough to transfer 50 TB in "
            "just over an hour. The bottleneck becomes cost ($4,000 egress), not time."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**You were off by {1/_ratio:.1f}×. You predicted ~{_predicted/60:.1f} hours. "
            f"The actual transfer time is ~{_actual_minutes} minutes.**\n\n"
            "At 100 Gbps = 12.5 GB/s: 50 TB ÷ 12.5 GB/s = 4,000 s = **67 minutes**. "
            "The 100 Gbps link is genuinely fast. Data gravity at this scale is "
            "primarily an *economic* problem (egress cost) rather than a time problem."
        ), kind="warn")
    return


# ─── ACT II: DECISION COMPARISON ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, _transfer_cost, _compute_cost_to_stay, _time_label, _ds_label):
    _decision = "Move compute to data" if _transfer_cost > _compute_cost_to_stay else "Move data to compute"
    _decision_color = "#CB202D" if _transfer_cost > _compute_cost_to_stay else "#008F45"

    mo.Html(f"""
    <div style="background: #f8fafc; border: 1.5px solid #e2e8f0; border-radius: 12px;
                padding: 20px 24px; margin: 16px 0;">
        <div style="font-size: 0.82rem; font-weight: 700; color: #475569; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 12px;">
            Architecture Decision
        </div>
        <div style="display: flex; gap: 24px; flex-wrap: wrap; align-items: center;">
            <div style="flex: 1; min-width: 200px;">
                <div style="font-size: 0.88rem; color: #64748b; margin-bottom: 4px;">
                    Transfer data ({_ds_label}) to compute
                </div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #0f172a;">
                    {_time_label} + ${_transfer_cost:,.0f} egress
                </div>
            </div>
            <div style="font-size: 1.4rem; color: #94a3b8; font-weight: 300;">vs</div>
            <div style="flex: 1; min-width: 200px;">
                <div style="font-size: 0.88rem; color: #64748b; margin-bottom: 4px;">
                    Spin up compute where data lives
                </div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #0f172a;">
                    $0 transfer + ${_compute_cost_to_stay:,.0f} compute
                </div>
            </div>
            <div style="background: {_decision_color}; color: white; padding: 10px 20px;
                        border-radius: 8px; font-weight: 700; font-size: 0.95rem; white-space: nowrap;">
                {_decision}
            </div>
        </div>
    </div>
    """)
    return


# ─── ACT II: REFLECTION ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Reflection

    You have seen data gravity in action. Now identify the principle it demonstrates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) Large datasets are heavy to store and require expensive hardware": "A",
            "B) Compute naturally migrates toward large datasets because transfer cost exceeds compute cost": "B",
            "C) Data should always be compressed before training to reduce transfer time": "C",
            "D) Cloud infrastructure is always faster than edge for data-intensive workloads": "D",
        },
        label="Data gravity means:",
    )
    act2_reflection
    return (act2_reflection,)


@app.cell(hide_code=True)
def _(mo, act2_reflection):
    mo.stop(
        act2_reflection.value is None,
        mo.callout(mo.md("Select your answer to continue."), kind="warn"),
    )

    if act2_reflection.value == "B":
        mo.callout(mo.md(
            "**Correct.** Data gravity (from @sec-data-engineering-data-gravity-adcb) is "
            "the economic and physical pressure that pushes compute toward large datasets. "
            "When T_transfer = D_vol / BW exceeds the training time budget, or when egress "
            "cost exceeds the cost of running compute in the data's region, it becomes "
            "cheaper to bring the compute to the data. This explains the architecture of "
            "Data Lakehouses — processing engines (Spark, Presto, training jobs) run "
            "directly on the storage nodes where the data already resides."
        ), kind="success")
    elif act2_reflection.value == "A":
        mo.callout(mo.md(
            "**Incorrect.** Data gravity is not about storage weight or hardware cost. "
            "It is about *movement cost*: the time and money required to transfer data "
            "across a network. A 1 PB dataset sitting in one region is not a gravity "
            "problem — the gravity problem begins when you try to move it somewhere else. "
            "The correct answer is B: compute migrates toward data when transfer cost "
            "exceeds the cost of co-locating compute."
        ), kind="warn")
    elif act2_reflection.value == "C":
        mo.callout(mo.md(
            "**Incorrect.** Compression reduces the bytes to transfer, which can reduce "
            "transfer time (T = D_vol / BW — smaller D_vol, shorter T). But data gravity "
            "is not the observation that compression helps. It is the observation that "
            "beyond a certain scale, no amount of compression makes transfer viable — "
            "compute must move to the data instead. The correct answer is B."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Incorrect.** Cloud infrastructure is not universally faster for data workloads. "
            "The key insight of data gravity is that *location relative to the data* determines "
            "which infrastructure is faster. An edge device processing data locally avoids all "
            "egress costs and network latency entirely. Cloud is faster only when the compute "
            "is co-located with the data (same region). The correct answer is B."
        ), kind="warn")
    return


# ─── ACT II: MATHPEEK ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — Data Gravity": mo.md("""
        **Data Gravity** (from @sec-data-engineering-data-gravity-adcb):

        $$T_{transfer} = \\frac{D_{vol}}{BW}$$

        Where:
        - **D_vol** — dataset volume in GB
        - **BW** — network bandwidth in GB/s
        - **T_transfer** — total transfer time in seconds

        **The architectural decision rule** (from the DataGravity notebook):

        > *If T_transfer > T_training → move compute to data*
        > *If T_transfer < T_training → move data to compute*

        **The economic rule** includes egress cost:

        $$Cost_{transfer} = D_{vol} \\times \\$0.08/\\text{GB}$$

        At petabyte scale: $10^6 \\text{ GB} \\times \\$0.08 = \\$80{,}000$ egress alone.

        **Numerical example (50 TB at 100 Gbps):**
        ```
        T_transfer = 50,000 GB / 12.5 GB/s = 4,000 s ≈ 67 minutes
        Cost       = 50,000 GB × $0.08    = $4,000

        Rule: If training takes < 67 min → the transfer takes longer than the job.
              Move compute to data.
        ```

        **The rule of thumb** (from the DataGravity notebook in data_engineering.qmd):
        - *Petabyte scale:* Code moves to Data (Data Lakehouse, in-place compute)
        - *Gigabyte scale:* Data moves to Code (standard transfer is viable)
        """),
    })
    return


# ─── DESIGN LEDGER SAVE + HUD ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, ledger, context_toggle, act1_prediction, act1_reflection,
      act2_prediction, act2_reflection, _gpu_util, _transfer_exceeds_training,
      act1_storage_type, COLORS):
    # Save chapter results to Design Ledger
    _act1_correct = act1_prediction.value == "B" if act1_prediction.value else False
    _act1_refl_correct = act1_reflection.value == "C" if act1_reflection.value else False
    _act2_correct = act2_prediction.value == "B" if act2_prediction.value else False
    _act2_refl_correct = act2_reflection.value == "B" if act2_reflection.value else False

    ledger.save(
        chapter=4,
        design={
            "context": context_toggle.value,
            "storage_type_chosen": act1_storage_type.value if act1_storage_type.value else "hdd",
            "gpu_util_at_start": round(_gpu_util, 1),
            "data_gravity_triggered": bool(_transfer_exceeds_training),
            "act1_correct": _act1_correct,
            "act1_reflection_correct": _act1_refl_correct,
            "act2_correct": _act2_correct,
            "act2_reflection_correct": _act2_refl_correct,
        },
    )

    # HUD footer
    _ctx_label = "Cloud — Multi-Region" if context_toggle.value == "cloud" else "Edge — Local"
    _act1_status = "correct" if _act1_correct else ("pending" if act1_prediction.value is None else "incorrect")
    _act2_status = "correct" if _act2_correct else ("pending" if act2_prediction.value is None else "incorrect")
    _gravity_status = "triggered" if _transfer_exceeds_training else "not triggered"

    def _status_color(s):
        return {"correct": "#4ade80", "pending": "#94a3b8", "incorrect": "#f87171", "triggered": "#f87171", "not triggered": "#4ade80"}.get(s, "#94a3b8")

    mo.Html(f"""
    <div style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap;
                padding: 14px 24px; background: #0f172a; border-radius: 12px;
                margin-top: 40px; font-family: 'SF Mono', monospace; font-size: 0.8rem;
                border: 1px solid #1e293b;">
        <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">
            DESIGN LEDGER · CH04
        </span>
        <span style="color: #475569;">|</span>
        <span>
            <span style="color: #94a3b8;">Context: </span>
            <span style="color: #e2e8f0;">{_ctx_label}</span>
        </span>
        <span>
            <span style="color: #94a3b8;">Act I prediction: </span>
            <span style="color: {_status_color(_act1_status)};">{_act1_status}</span>
        </span>
        <span>
            <span style="color: #94a3b8;">Act II prediction: </span>
            <span style="color: {_status_color(_act2_status)};">{_act2_status}</span>
        </span>
        <span>
            <span style="color: #94a3b8;">Gravity trap: </span>
            <span style="color: {_status_color(_gravity_status)};">{_gravity_status}</span>
        </span>
        <span>
            <span style="color: #94a3b8;">GPU util (initial): </span>
            <span style="color: #e2e8f0;">{_gpu_util:.0f}%</span>
        </span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
