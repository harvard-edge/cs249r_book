import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 04: THE DATA GRAVITY TRAP — DISTRIBUTED SCALE
#
# Chapter: data_storage.qmd (@sec-data-storage)
# Core Invariant: Data gravity — once data is stored somewhere, computation
#   tends to move TO the data rather than data moving to compute. At
#   distributed training scale, the I/O system becomes the bottleneck when
#   reading training data. Distributed file systems (Lustre, GPFS) provide
#   aggregate bandwidth but introduce coordination overhead.
#
# 2-Act structure (35-40 min total):
#   Act I:  The I/O Bottleneck (12-15 min)
#     128 H100s at 12% GPU utilization despite a 672 GB/s Lustre cluster.
#     Prediction lock → I/O analyzer → reveal → reflection → MathPeek
#   Act II: Distributed Storage Architecture (20-25 min)
#     Design a 4096-GPU cluster's storage layer. Metadata bottleneck lurks.
#     Prediction lock → storage designer → failure state → reflection → MathPeek
#
# Deployment contexts:
#   NVMe:            Single-node local NVMe (7 GB/s per drive, 4 drives = 28 GB/s)
#   Distributed FS:  Lustre/GPFS cluster (OST nodes × 28 GB/s each, MDS ceiling)
#
# Traceability:
#   H100 HBM bandwidth      — @sec-data-storage-fuel-line (H100_MEM_BW = 3.35 TB/s)
#   NVMe sequential BW      — @sec-data-storage-fuel-line (NVME_SEQUENTIAL_BW = 7 GB/s)
#   Lustre OST BW           — @sec-data-storage (4 × NVMe per OST = 28 GB/s)
#   GPU utilization formula — @sec-data-storage (t_compute / t_total)
#   Prefetch depth model    — @sec-data-storage (queue depth × batch_read_time)
#   Metadata ops rate       — @sec-data-storage (gpus × workers × ops_per_open)
#   IB HDR200 bandwidth     — @sec-network-fabrics (200 Gbps = 25 GB/s per port)
#
# Design Ledger save:
#   chapter="v2_04", context, storage_type, gpu_count, data_workers_per_gpu,
#   io_throughput_gbs, act1_prediction, act1_correct, act2_result,
#   act2_decision, constraint_hit, mds_saturated
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
    _c = COLORS["Cloud"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 04
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The I/O Bottleneck
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.02rem; color: #94a3b8;
                      max-width: 680px; line-height: 1.65;">
                128 H100s sit at 12% GPU utilization despite a 672 GB/s Lustre cluster.
                The storage system looks fine on paper. The accelerators are starving.
                Before buying more hardware, you need to find the actual bottleneck.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: I/O Bottleneck · 12–15 min
                </span>
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act II: Storage Architecture · 20–25 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min total
                </span>
                <span class="badge badge-info">Chapter 4: Data Storage</span>
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

    - **@sec-data-storage-fuel-line** (The Fuel Line) — The 479× gap between H100 HBM
      bandwidth (3.35 TB/s) and a single NVMe drive (7 GB/s), and why this gap cannot be
      closed by any single technology. The hierarchical storage model is the only response.
    - **@sec-data-storage-workload-inversion** (How ML Workloads Invert Storage Assumptions)
      — Why ML training reads every byte sequentially exactly once per epoch, why there is
      no hot-data subset, and how this inverts every assumption from database storage design.
    - **@sec-data-storage** (Data Pipeline Architecture) — Prefetching, pipelining, and the
      GPU Direct Storage pathway that eliminates the CPU from the data path entirely.
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "NVMe (single node, 4 drives at 7 GB/s each)": "nvme",
            "Distributed FS (Lustre/GPFS cluster, OST nodes)": "distributed_fs",
        },
        value="Distributed FS (Lustre/GPFS cluster, OST nodes)",
        label="Storage context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Select Your Storage Context"),
        mo.md(
            "This choice sets the storage architecture for both acts. "
            "NVMe reflects a single training node reading local drives. "
            "Distributed FS reflects a production multi-node Lustre/GPFS deployment."
        ),
        context_toggle,
    ])
    return (context_toggle,)


@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS):
    _ctx = context_toggle.value
    if _ctx == "nvme":
        _bw_desc = "4 × NVMe = 28 GB/s local storage bandwidth per node"
        _arch_desc = "No metadata server overhead — direct kernel I/O"
        _color = COLORS["GreenLine"]
        _label = "NVMe — Single Node"
    else:
        _bw_desc = "24 OST nodes × 28 GB/s = 672 GB/s aggregate Lustre bandwidth"
        _arch_desc = "Metadata server (MDS) coordinates all namespace operations"
        _color = COLORS["Cloud"]
        _label = "Distributed FS — Lustre/GPFS"

    mo.callout(mo.md(
        f"**Context: {_label}** — Storage bandwidth: {_bw_desc}. "
        f"Architecture note: {_arch_desc}. "
        "The I/O bottleneck analysis in Act I will use these parameters."
    ), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# ACT I: THE I/O BOTTLENECK
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
            The I/O Bottleneck
        </div>
        <div style="font-size: 0.95rem; color: #475569; margin-top: 6px; max-width: 700px;">
            Your training platform has 128 H100 GPUs and a 24-node Lustre cluster providing
            672 GB/s of aggregate storage bandwidth. GPU utilization is at 12%.
            Before touching the simulator, commit to a diagnosis.
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
            Incoming Message · Training Platform Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We have 128 H100s, each capable of consuming training data at 3.35 TB/s from HBM.
            We store our training data on a 24-node Lustre cluster — each OST node has 4 NVMe
            SSDs at 7 GB/s each, giving 28 GB/s per node and 672 GB/s in aggregate.
            Our GPUs are sitting at 12% utilization during the data loading phase.
            The infrastructure team says the Lustre cluster is healthy and nowhere near saturated.
            We're about to request budget for another 256 GPUs. Should we?"
        </div>
    </div>
    """)
    return


# ─── ACT I: THE NUMBERS ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _c = COLORS["BlueLine"]
    mo.Html(f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
                padding: 20px 24px; margin: 12px 0;">
        <div style="font-size: 0.82rem; font-weight: 700; color: #475569; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 14px;">
            The Arithmetic Before You Touch the Simulator
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px;">
                    H100 HBM bandwidth (NVIDIA spec)
                </div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {_c};">
                    3,350 GB/s per GPU
                </div>
            </div>
            <div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px;">
                    Lustre aggregate bandwidth (24 OSTs × 28 GB/s)
                </div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {_c};">
                    672 GB/s total
                </div>
            </div>
            <div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px;">
                    Actual data ingestion needed per GPU (4 MB batch / 50 ms step)
                </div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {COLORS['GreenLine']};">
                    ~80 MB/s per GPU
                </div>
            </div>
            <div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px;">
                    Total I/O demand for 128 GPUs (128 × 80 MB/s)
                </div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {COLORS['GreenLine']};">
                    ~10.24 GB/s needed
                </div>
            </div>
        </div>
        <div style="margin-top: 14px; font-size: 0.88rem; color: #475569; line-height: 1.65;
                    border-top: 1px solid #e2e8f0; padding-top: 12px;">
            <strong>The gap is striking:</strong> 672 GB/s of Lustre capacity against only 10.24 GB/s
            of actual demand. The cluster is at ~1.5% utilization. The Lustre hardware is
            not the problem. Something else is consuming the time between batches.
        </div>
    </div>
    """)
    return


# ─── ACT I: PREDICTION LOCK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Your Prediction

    *Before interacting with the I/O analyzer, commit to your diagnosis.*
    The storage cluster is at ~1.5% utilization. The GPUs are at 12%. The gap is real.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) The Lustre cluster is saturated — 672 GB/s is not enough for 128 GPUs": "A",
            "B) Small random reads — the access pattern does not match Lustre's sequential throughput profile": "B",
            "C) Network bandwidth between the Lustre nodes and the GPU compute nodes is saturated": "C",
            "D) 12% utilization is normal — data loading always accounts for 88% of training time": "D",
        },
        label="GPU utilization is 12% despite a 672 GB/s Lustre cluster. The most likely root cause is:",
    )
    act1_prediction
    return (act1_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction to unlock the I/O bottleneck analyzer."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(
            f"**Prediction locked: {act1_prediction.value[:2]}** "
            "Now use the I/O analyzer to test your hypothesis. "
            "Adjust the data loader configuration to see what moves the needle."
        ),
        kind="info",
    )
    return


# ─── ACT I: INSTRUMENTS ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### The I/O Bottleneck Analyzer

    Adjust the data loading configuration below to see how GPU utilization responds.
    The physics engine models the relationship between prefetch depth, worker count,
    read pattern, and effective I/O throughput.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_num_workers = mo.ui.slider(
        start=1, stop=16, value=1, step=1,
        label="Data loader workers per GPU",
        show_value=True,
    )
    act1_batch_size = mo.ui.slider(
        start=32, stop=512, value=64, step=32,
        label="Batch size (samples)",
        show_value=True,
    )
    act1_read_pattern = mo.ui.dropdown(
        options={
            "Sequential (large chunks, Lustre-optimal)": "sequential",
            "Random (small 4K reads, worst case)": "random",
            "Prefetch pipeline (async, overlapped with compute)": "prefetch",
        },
        value="Random (small 4K reads, worst case)",
        label="Read pattern",
    )
    act1_ost_count = mo.ui.slider(
        start=4, stop=48, value=24, step=4,
        label="Active Lustre OST nodes",
        show_value=True,
    )
    mo.hstack([
        act1_num_workers,
        act1_batch_size,
        act1_read_pattern,
        act1_ost_count,
    ], gap=2, justify="start")
    return (act1_num_workers, act1_batch_size, act1_read_pattern, act1_ost_count)


@app.cell(hide_code=True)
def _(
    mo, act1_num_workers, act1_batch_size, act1_read_pattern, act1_ost_count,
    context_toggle, go, apply_plotly_theme, COLORS, math,
):
    # ── Hardware constants ─────────────────────────────────────────────────────
    # Source: @sec-data-storage-fuel-line
    _H100_BW_GBS       = 3350.0    # H100 SXM5 HBM3e bandwidth, NVIDIA spec sheet
    _NVME_SEQ_BW_GBS   = 7.0       # PCIe Gen4 NVMe sequential read bandwidth
    _NVME_RAND_BW_GBS  = 0.5       # NVMe random 4K read effective throughput
    _LUSTRE_OST_BW_GBS = 28.0      # Per OST node (4 × NVMe at 7 GB/s each)
    _IB_HDR200_BW_GBS  = 25.0      # InfiniBand HDR-200 per port (200 Gbps = 25 GB/s)

    # ── Workload parameters ────────────────────────────────────────────────────
    _GPU_COUNT         = 128        # fixed for Act I scenario
    _SAMPLE_SIZE_MB    = 4.0        # 4 MB per training sample (typical image/text)
    _COMPUTE_TIME_MS   = 50.0       # GPU compute time per batch (H100, realistic)

    # ── Context-dependent base bandwidth ──────────────────────────────────────
    _ctx = context_toggle.value
    if _ctx == "nvme":
        # Single node: 4 × NVMe = 28 GB/s, no network overhead
        _base_storage_bw = 28.0     # GB/s (4 × 7 GB/s NVMe)
        _network_overhead = 1.0     # no network factor for local NVMe
    else:
        # Distributed Lustre: aggregate bandwidth from active OSTs
        _base_storage_bw = act1_ost_count.value * _LUSTRE_OST_BW_GBS
        # Network overhead: IB connects compute to storage, capped at IB bandwidth
        # 128 GPUs × 1 IB port per node = need aggregate fabric bandwidth
        _network_overhead = min(1.0, (_IB_HDR200_BW_GBS * 16) / _base_storage_bw)

    # ── Read pattern factor ────────────────────────────────────────────────────
    # Sequential reads can sustain near-peak bandwidth on Lustre
    # Random 4K reads hit IOPS wall: OST NVMe at ~800K IOPS × 4KB = 3.2 GB/s per OST
    # Prefetch: async pipeline overlaps I/O with compute, hides much of the latency
    _pattern_factor = {
        "sequential": 0.85,   # near-peak sequential read efficiency
        "random":     0.07,   # 4K random read: ~7% of sequential bandwidth
        "prefetch":   0.78,   # async prefetch hides latency, near-sequential efficiency
    }[act1_read_pattern.value]

    # ── Worker scaling ─────────────────────────────────────────────────────────
    # Workers per GPU parallelize reads across CPU cores
    # Diminishing returns: Amdahl-style. Beyond 8 workers, OS scheduling overhead grows.
    # Source: empirical characterization from @sec-data-storage
    _workers = act1_num_workers.value
    if _workers == 1:
        _worker_factor = 0.25   # 1 worker: sequential, blocking reads — severe bottleneck
    elif _workers <= 4:
        _worker_factor = 0.25 + (_workers - 1) * 0.20   # near-linear scaling 1–4
    elif _workers <= 8:
        _worker_factor = 0.85 + (_workers - 4) * 0.03   # sub-linear scaling 4–8
    else:
        _worker_factor = min(0.97 + (_workers - 8) * 0.002, 1.0)   # minimal gain > 8

    # ── Prefetch queue depth ───────────────────────────────────────────────────
    # Prefetch depth: how many batches ahead we read asynchronously
    # depth = workers × overlap_factor (pattern-dependent)
    _overlap_factor = {"sequential": 2.0, "random": 0.5, "prefetch": 4.0}[act1_read_pattern.value]
    _prefetch_depth = max(1, int(_workers * _overlap_factor))

    # ── Effective I/O bandwidth per GPU ───────────────────────────────────────
    _effective_storage_bw = _base_storage_bw * _pattern_factor * _worker_factor * _network_overhead
    # Per-GPU share of the storage bandwidth
    _per_gpu_bw = _effective_storage_bw / _GPU_COUNT   # GB/s per GPU

    # ── Batch load time and GPU utilization ───────────────────────────────────
    _batch_gb = (act1_batch_size.value * _SAMPLE_SIZE_MB) / 1024.0
    _t_io_ms = (_batch_gb / _per_gpu_bw) * 1000.0 if _per_gpu_bw > 0 else 9999.0

    # Prefetch pipeline: if prefetch, I/O and compute overlap
    # Effective wait = max(0, t_io - compute_time × prefetch_depth)
    if act1_read_pattern.value == "prefetch":
        _t_wait_ms = max(0.0, _t_io_ms - _COMPUTE_TIME_MS * min(_prefetch_depth, 4))
    else:
        _t_wait_ms = max(0.0, _t_io_ms - _COMPUTE_TIME_MS)

    _t_total_ms = _COMPUTE_TIME_MS + _t_wait_ms
    _gpu_util = min((_COMPUTE_TIME_MS / _t_total_ms) * 100.0, 100.0) if _t_total_ms > 0 else 0.0

    # ── Color coding ──────────────────────────────────────────────────────────
    if _gpu_util >= 80:
        _util_color = COLORS["GreenLine"]
        _util_label = "Healthy"
    elif _gpu_util >= 50:
        _util_color = COLORS["OrangeLine"]
        _util_label = "Degraded"
    else:
        _util_color = COLORS["RedLine"]
        _util_label = "Starved"

    # ── GPU utilization vs num_workers curve ──────────────────────────────────
    _worker_range = list(range(1, 17))
    _util_curve = []
    for _w in _worker_range:
        if _w == 1:
            _wf = 0.25
        elif _w <= 4:
            _wf = 0.25 + (_w - 1) * 0.20
        elif _w <= 8:
            _wf = 0.85 + (_w - 4) * 0.03
        else:
            _wf = min(0.97 + (_w - 8) * 0.002, 1.0)
        _eff_bw_w = _base_storage_bw * _pattern_factor * _wf * _network_overhead
        _per_gpu_bw_w = _eff_bw_w / _GPU_COUNT
        _t_io_w = (_batch_gb / _per_gpu_bw_w) * 1000.0 if _per_gpu_bw_w > 0 else 9999.0
        _ov_f = {"sequential": 2.0, "random": 0.5, "prefetch": 4.0}[act1_read_pattern.value]
        _pd_w = max(1, int(_w * _ov_f))
        if act1_read_pattern.value == "prefetch":
            _tw = max(0.0, _t_io_w - _COMPUTE_TIME_MS * min(_pd_w, 4))
        else:
            _tw = max(0.0, _t_io_w - _COMPUTE_TIME_MS)
        _tt = _COMPUTE_TIME_MS + _tw
        _u = min((_COMPUTE_TIME_MS / _tt) * 100.0, 100.0) if _tt > 0 else 0.0
        _util_curve.append(_u)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_worker_range,
        y=_util_curve,
        mode="lines+markers",
        line=dict(color=COLORS["BlueLine"], width=2.5),
        marker=dict(size=7, color=COLORS["BlueLine"]),
        name="GPU Utilization (%)",
    ))
    # Highlight current workers setting
    _fig.add_trace(go.Scatter(
        x=[_workers],
        y=[_gpu_util],
        mode="markers",
        marker=dict(size=14, color=_util_color, symbol="circle", line=dict(width=2, color="white")),
        name=f"Current: {_workers} workers → {_gpu_util:.0f}%",
    ))
    # Target line at 80%
    _fig.add_hline(
        y=80,
        line_dash="dash",
        line_color=COLORS["GreenLine"],
        annotation_text="Target: 80% GPU utilization",
        annotation_position="top right",
    )
    _fig.update_layout(
        height=300,
        xaxis_title="Data loader workers per GPU",
        yaxis_title="GPU Utilization (%)",
        yaxis=dict(range=[0, 105]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=50, b=50),
        title=dict(
            text=f"GPU Utilization vs. Workers ({act1_read_pattern.value}, {act1_ost_count.value} OSTs)",
            font=dict(size=13),
        ),
    )
    apply_plotly_theme(_fig)

    # ── Metric cards ──────────────────────────────────────────────────────────
    _lustre_util_pct = (_effective_storage_bw / (_base_storage_bw + 0.001)) * 100.0
    _lustre_color = COLORS["RedLine"] if _lustre_util_pct < 20 else COLORS["OrangeLine"] if _lustre_util_pct < 60 else COLORS["GreenLine"]

    _cards_html = f"""
    <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0;">
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 165px; text-align: center; background: white;
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
                    min-width: 165px; text-align: center; background: white;">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                Effective I/O BW
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {COLORS['BlueLine']}; line-height: 1;">
                {_effective_storage_bw:.1f}
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">GB/s total</div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 165px; text-align: center; background: white;">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                I/O Wait
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {_util_color}; line-height: 1;">
                {_t_wait_ms:.1f}
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">ms per batch</div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 165px; text-align: center; background: white;">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                Prefetch Queue
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {COLORS['BlueLine']}; line-height: 1;">
                {_prefetch_depth}
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">batches ahead</div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 165px; text-align: center; background: white;">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                Storage Utilization
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {_lustre_color}; line-height: 1;">
                {_lustre_util_pct:.0f}%
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">of capacity</div>
        </div>
    </div>
    """

    # ── Physics formula display ───────────────────────────────────────────────
    _formula_md = f"""
    **The physics (from @sec-data-storage):**

    ```
    Effective BW  = Storage_BW × pattern_factor × worker_factor × network_overhead
                  = {_base_storage_bw:.0f} GB/s × {_pattern_factor:.2f} × {_worker_factor:.2f} × {_network_overhead:.2f}
                  = {_effective_storage_bw:.1f} GB/s

    Per-GPU BW    = Effective_BW / GPU_count
                  = {_effective_storage_bw:.1f} GB/s / {_GPU_COUNT}
                  = {_per_gpu_bw:.4f} GB/s = {_per_gpu_bw * 1024:.1f} MB/s

    I/O time      = batch_bytes / per_gpu_bw
                  = {_batch_gb * 1024:.1f} MB / {_per_gpu_bw * 1024:.1f} MB/s
                  = {_t_io_ms:.1f} ms

    GPU Util      = t_compute / (t_compute + t_io_wait)
                  = {_COMPUTE_TIME_MS:.1f} ms / ({_COMPUTE_TIME_MS:.1f} + {_t_wait_ms:.1f}) ms
                  = {_gpu_util:.1f}%
    ```
    """

    mo.vstack([
        mo.Html(_cards_html),
        mo.as_html(_fig),
        mo.md(_formula_md),
    ])
    return (
        _gpu_util,
        _t_wait_ms,
        _effective_storage_bw,
        _per_gpu_bw,
        _base_storage_bw,
        _pattern_factor,
        _worker_factor,
        _prefetch_depth,
    )


# ─── ACT I: PREDICTION-VS-REALITY OVERLAY ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction, _gpu_util):
    _pred_text = {
        "A": "Lustre cluster saturated (too little storage bandwidth)",
        "B": "Small random reads — access pattern mismatch",
        "C": "Network bandwidth between Lustre and GPU nodes saturated",
        "D": "12% GPU utilization is normal for data loading",
    }[act1_prediction.value]

    _is_correct = act1_prediction.value == "B"

    if _is_correct:
        mo.callout(mo.md(
            f"**Correct. You predicted: {_pred_text}.**\n\n"
            f"The simulation confirms it. With 1 worker and random reads, GPU utilization is "
            f"**{_gpu_util:.0f}%** — matching the reported 12%. "
            "The Lustre cluster is at roughly 1.5% of its capacity. The bottleneck is not "
            "aggregate bandwidth — it is the *access pattern*. Each training sample requires "
            "an independent file open, seek, and read. At 128 GPUs × 1 worker each, this "
            "generates 128 independent random I/O streams hitting the Lustre namespace. "
            "Each random 4K read delivers ~7% of the sequential bandwidth the hardware can provide. "
            "The fix is **not more hardware** — it is changing the access pattern to sequential "
            "reads (e.g., WebDataset tar shards) and increasing workers to 4–8 per GPU."
        ), kind="success")
    elif act1_prediction.value == "A":
        mo.callout(mo.md(
            f"**Not quite. You predicted: {_pred_text}.**\n\n"
            f"The numbers refute this immediately: 128 GPUs need only ~10.24 GB/s total, "
            "while Lustre provides 672 GB/s — a 65× margin. Saturation would require the "
            "actual demand to approach the cluster capacity. The cluster is at ~1.5% utilization. "
            "The root cause is the *read pattern*: random 4K reads from individual files deliver "
            "only ~7% of the sequential bandwidth. Increase num_workers and switch to a "
            "sequential prefetch pattern to see utilization climb. "
            "**Correct answer: B — access pattern mismatch.**"
        ), kind="warn")
    elif act1_prediction.value == "C":
        mo.callout(mo.md(
            f"**Not quite. You predicted: {_pred_text}.**\n\n"
            "InfiniBand HDR-200 provides 25 GB/s per port, and a 24-node Lustre cluster with "
            "one IB port per node has 600 GB/s of aggregate network fabric — more than enough "
            "for 10.24 GB/s of actual demand. The OST-to-compute network is not saturated. "
            "The bottleneck is within the storage access pattern itself: random small reads "
            "from many independent file handles serialize I/O that Lustre's hardware could "
            "service as fast sequential streams. "
            "**Correct answer: B — access pattern mismatch.**"
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**Not quite. You predicted: {_pred_text}.**\n\n"
            "12% GPU utilization is definitively not normal. A properly configured data "
            "pipeline should achieve 75–90% GPU utilization during training. The simulation "
            "shows that switching to 4 workers with a prefetch pipeline raises utilization "
            "above 75% with the same storage hardware. This is a configuration failure, "
            "not a hardware limitation. "
            "**Correct answer: B — access pattern mismatch.**"
        ), kind="warn")
    return


# ─── ACT I: REFLECTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Reflection

    You have seen how access pattern dominates throughput on distributed filesystems.
    Now identify the correct architectural response.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) Purchase faster NVMe SSDs for each Lustre OST node": "A",
            "B) Pre-shuffle and pack data into large sequential shards (WebDataset / TFDS format)": "B",
            "C) Add more DataLoader processes to increase RAM cache utilization": "C",
            "D) Replace the filesystem with a database to enable faster indexed lookups": "D",
        },
        label="What is the primary fix for small random read performance on a distributed filesystem?",
    )
    act1_reflection
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Select your answer to continue to Act II."), kind="warn"),
    )

    if act1_reflection.value == "B":
        mo.callout(mo.md(
            "**Correct.** Packing data into large sequential shards (WebDataset `.tar` archives, "
            "TensorFlow Record files, Parquet) converts random access into sequential streaming. "
            "Lustre's aggregate bandwidth — 672 GB/s — is genuinely available to sequential reads. "
            "The hardware was never the problem; the access pattern was. "
            "Sequential shards allow a single worker per GPU to sustain near-peak throughput "
            "by reading one large file contiguously rather than opening thousands of small files. "
            "This is the standard production fix for I/O-bound distributed training pipelines, "
            "documented in @sec-data-storage."
        ), kind="success")
    elif act1_reflection.value == "A":
        mo.callout(mo.md(
            "**Incorrect.** The NVMe SSDs are not the bottleneck. Each OST node already provides "
            "28 GB/s (4 × 7 GB/s NVMe) — far more than the ~80 MB/s per GPU the workload demands. "
            "Faster SSDs would give you faster hardware with the same utilization ceiling. "
            "The constraint is software: the access pattern generates random I/O that cannot "
            "benefit from higher sequential bandwidth. The correct fix is B: sequential shards."
        ), kind="warn")
    elif act1_reflection.value == "C":
        mo.callout(mo.md(
            "**Incorrect.** Adding more DataLoader processes increases the *number* of concurrent "
            "I/O requests, but each request is still a small random read from a different file. "
            "More processes multiply the IOPS demand, which can worsen metadata server load on "
            "Lustre without improving actual throughput. The root cause — fragmented random access "
            "pattern — is not addressed by adding more workers. The correct fix is B: pack data "
            "into sequential shards so each worker reads large contiguous chunks."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Incorrect.** A database adds indexing overhead, transactions, and write-ahead "
            "logging — all of which increase latency and reduce throughput for the purely "
            "sequential read pattern that training requires. Databases are optimized for "
            "transactional workloads with random read/write patterns. ML training requires "
            "high-throughput sequential streaming, which a well-configured filesystem (with "
            "sequential shards) handles better than any database engine. The correct fix is B."
        ), kind="warn")
    return


# ─── ACT I: MATHPEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — I/O Throughput and Prefetch Depth": mo.md("""
        **I/O Throughput Model** (from @sec-data-storage):

        $$\\text{Effective BW} = BW_{storage} \\times f_{pattern} \\times f_{workers} \\times f_{network}$$

        Where:
        - **BW_storage** — peak sequential storage bandwidth (OSTs × 28 GB/s for Lustre)
        - **f_pattern** — read pattern efficiency: sequential ≈ 0.85, random ≈ 0.07, prefetch ≈ 0.78
        - **f_workers** — worker scaling factor (near-linear 1–4, diminishing returns beyond 8)
        - **f_network** — IB fabric utilization fraction (usually close to 1.0 if not saturated)

        **GPU Utilization:**

        $$\\eta_{GPU} = \\frac{t_{compute}}{t_{compute} + t_{IO\\ wait}}$$

        With prefetch pipelining:

        $$t_{IO\\ wait} = \\max\\left(0,\\ t_{IO} - t_{compute} \\times d_{prefetch}\\right)$$

        Where $d_{prefetch}$ is the number of batches pre-loaded asynchronously.

        **Random vs. Sequential Bandwidth Gap:**
        ```
        Sequential read (1 worker, Lustre):  672 GB/s × 0.85 = 571 GB/s aggregate
        Random 4K read  (1 worker, Lustre):  672 GB/s × 0.07 = 47 GB/s aggregate
        Per-GPU share   (128 GPUs):          571 / 128 = 4.46 GB/s  vs  47 / 128 = 0.37 GB/s

        Batch load time (64 samples × 4 MB = 256 MB):
          Sequential:  256 MB / 4,460 MB/s ≈ 0.057 ms  →  GPU util ≈ 99%
          Random:      256 MB /   370 MB/s ≈ 0.69 ms  →  GPU util ≈ 7%  (matches ~12% observed)
        ```

        **WebDataset Sequential Read Benchmark:**
        - Individual JPEG files, random access: ~70 MB/s per node
        - WebDataset `.tar` shards, sequential: ~2.8 GB/s per node (40× improvement)
        - Source: NVIDIA DGX documentation, @sec-data-storage
        """),
    })
    return


# ═════════════════════════════════════════════════════════════════════════════
# ACT II: DISTRIBUTED STORAGE ARCHITECTURE
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
            Distributed Storage Architecture
        </div>
        <div style="font-size: 0.95rem; color: #475569; margin-top: 6px; max-width: 700px;">
            You have fixed the training pipeline. Now a harder problem: design the storage
            architecture for a 4,096-GPU cluster. The bandwidth math looks fine. But at this
            scale, a different bottleneck emerges — one that hardware purchases cannot solve.
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
            Incoming Message · Infrastructure CTO
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We're building a new 4,096-GPU training cluster. Current design: 48 Lustre OST
            nodes with 24 TB NVMe each, an InfiniBand fabric, compute-storage separated.
            Aggregate bandwidth: 48 × 28 GB/s = 1,344 GB/s. At 80 MB/s per GPU, we need
            only 328 GB/s — a 4× safety margin. The storage team says the design is sound.
            We need to provision enough I/O for the next 3 years of growth. What are we missing?"
        </div>
    </div>
    """)
    return


# ─── ACT II: PREDICTION LOCK ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Your Prediction

    *Commit to your hypothesis before using the storage architecture designer.*
    The bandwidth math is correct. The safety margin exists. Something else will fail first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_prediction = mo.ui.radio(
        options={
            "A) The Lustre cluster will hit bandwidth saturation immediately at 4096 GPUs": "A",
            "B) At 4096 GPUs with 8 workers each, the metadata server (MDS) becomes the bottleneck": "B",
            "C) Storage will always be sufficient — just add OST nodes as needed to meet demand": "C",
            "D) Compute-storage network separation causes unacceptable latency (>1 ms per read)": "D",
        },
        label="A 4096-GPU cluster with 1344 GB/s aggregate Lustre bandwidth. What fails first?",
    )
    act2_prediction
    return (act2_prediction,)


@app.cell(hide_code=True)
def _(mo, act2_prediction):
    mo.stop(
        act2_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction to unlock the distributed storage designer."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(
            f"**Prediction locked: {act2_prediction.value[:2]}** "
            "Now use the architecture designer to find the failure mode. "
            "Adjust GPU count, worker count, and MDS configuration to see where the ceiling hits."
        ),
        kind="info",
    )
    return


# ─── ACT II: INSTRUMENTS ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### The Distributed Storage Architecture Designer

    Adjust the cluster parameters below to find the failure boundary.
    The designer models bandwidth demand, metadata operation rate, and the MDS throughput ceiling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_gpu_count = mo.ui.slider(
        start=128, stop=4096, value=512, step=128,
        label="GPU count",
        show_value=True,
    )
    act2_workers_per_gpu = mo.ui.slider(
        start=1, stop=16, value=4, step=1,
        label="Data workers per GPU",
        show_value=True,
    )
    act2_dataset_files = mo.ui.dropdown(
        options={
            "Sequential shards (1 file per 10K samples, ~1K total files)": "shards",
            "Individual files (1 file per sample, ~100M total files)": "individual",
            "Object store (no POSIX, content-addressed)": "object_store",
        },
        value="Individual files (1 file per sample, ~100M total files)",
        label="Dataset organization",
    )
    act2_mds_count = mo.ui.slider(
        start=1, stop=8, value=1, step=1,
        label="Metadata server (MDS) count",
        show_value=True,
    )
    mo.hstack([
        act2_gpu_count,
        act2_workers_per_gpu,
        act2_dataset_files,
        act2_mds_count,
    ], gap=2, justify="start")
    return (act2_gpu_count, act2_workers_per_gpu, act2_dataset_files, act2_mds_count)


@app.cell(hide_code=True)
def _(
    mo, act2_gpu_count, act2_workers_per_gpu, act2_dataset_files, act2_mds_count,
    context_toggle, go, apply_plotly_theme, COLORS,
):
    # ── Hardware constants ─────────────────────────────────────────────────────
    # Source: @sec-data-storage, @sec-network-fabrics
    _H100_BW_GBS_II     = 3350.0   # H100 HBM bandwidth, NVIDIA spec
    _NVME_SEQ_BW_II     = 7.0      # NVMe Gen4 sequential read bandwidth
    _LUSTRE_OST_BW_II   = 28.0     # Per OST node: 4 × 7 GB/s NVMe
    _OST_COUNT          = 48       # Fixed per the design spec
    _TOTAL_LUSTRE_BW    = _OST_COUNT * _LUSTRE_OST_BW_II   # 1344 GB/s aggregate

    # ── Metadata server capacity model ────────────────────────────────────────
    # Lustre MDS handles: open, stat, close, readdir, rename, create, unlink
    # Each file access typically generates 2–4 metadata ops (open + read + close + stat)
    # Source: Lustre filesystem documentation, @sec-data-storage
    _MDS_OPS_PER_NODE   = 50_000    # ops/sec per MDS node (Lustre typical, enterprise NVMe MDS)
    _METADATA_OPS_PER_FILE_ACCESS = 3.0    # open + close + stat per file read

    # ── Dataset organization factor ────────────────────────────────────────────
    # Sequential shards: one large tar file opened per worker per epoch → very few metadata ops
    # Individual files: one file open per sample → maximum metadata pressure
    # Object store: no POSIX metadata server — content-addressed, no namespace overhead
    _dataset_type = act2_dataset_files.value
    if _dataset_type == "shards":
        # Each worker opens ~1 shard file at a time, rotates through epoch
        # Metadata ops = workers × files_per_epoch ÷ samples_per_file
        _files_open_per_sec_per_worker = 0.2   # ~5 seconds per shard
        _metadata_pressure_factor = 0.02       # very low — large sequential files
    elif _dataset_type == "individual":
        # Each worker opens one file per sample at ~20ms training steps
        # Metadata ops = workers × (1 / step_time) × metadata_ops_per_file
        _files_open_per_sec_per_worker = 50.0  # 50 files/sec at ~20ms/sample
        _metadata_pressure_factor = 1.0        # maximum pressure
    else:
        # Object store: no POSIX MDS. Operations go directly to object server.
        _files_open_per_sec_per_worker = 50.0  # same access rate
        _metadata_pressure_factor = 0.0        # object store has no MDS bottleneck

    # ── Total metadata operations per second ──────────────────────────────────
    _total_workers = act2_gpu_count.value * act2_workers_per_gpu.value
    _ops_per_sec_demand = (
        _total_workers
        * _files_open_per_sec_per_worker
        * _METADATA_OPS_PER_FILE_ACCESS
        * _metadata_pressure_factor
    )

    # ── MDS capacity and saturation ────────────────────────────────────────────
    _mds_capacity = act2_mds_count.value * _MDS_OPS_PER_NODE
    _mds_saturated = (_dataset_type != "object_store") and (_ops_per_sec_demand > _mds_capacity)
    _ops_k = _ops_per_sec_demand / 1000
    _mds_capacity_k = _mds_capacity / 1000
    _mds_util_pct = min((_ops_per_sec_demand / _mds_capacity) * 100.0, 200.0) if _mds_capacity > 0 else 0.0

    # ── Bandwidth demand vs supply ─────────────────────────────────────────────
    # Actual bandwidth demand: 80 MB/s per GPU (realistic per @sec-data-storage)
    _bw_demand_gbs = act2_gpu_count.value * 0.08   # 80 MB/s = 0.08 GB/s per GPU

    # Context factor: NVMe context has different supply
    _ctx_ii = context_toggle.value
    if _ctx_ii == "nvme":
        _bw_supply_gbs = 28.0    # single node local NVMe
    else:
        _bw_supply_gbs = _TOTAL_LUSTRE_BW   # 1344 GB/s aggregate Lustre

    _bw_headroom_pct = ((_bw_supply_gbs - _bw_demand_gbs) / _bw_supply_gbs) * 100.0
    _bw_saturated = _bw_demand_gbs > _bw_supply_gbs

    # ── Needed MDS nodes to handle demand ─────────────────────────────────────
    _needed_mds = max(1, int(math.ceil(_ops_per_sec_demand / _MDS_OPS_PER_NODE))) if _dataset_type != "object_store" else 1

    # ── Architecture recommendation ────────────────────────────────────────────
    if _dataset_type == "object_store":
        _recommendation = "Object store — no MDS. Scales to any GPU count."
        _rec_color = COLORS["GreenLine"]
    elif _mds_saturated:
        _recommendation = f"Add {_needed_mds - act2_mds_count.value} more MDS nodes or migrate to object store."
        _rec_color = COLORS["RedLine"]
    elif _mds_util_pct > 60:
        _recommendation = "MDS approaching capacity. Switch to sequential shards or pre-provision additional MDS."
        _rec_color = COLORS["OrangeLine"]
    else:
        _recommendation = "Architecture is sound at current scale. Monitor MDS utilization as GPU count grows."
        _rec_color = COLORS["GreenLine"]

    # ── Chart: demand vs supply (bandwidth and metadata) ─────────────────────
    _gpu_range = list(range(128, 4097, 128))
    _bw_demand_curve = [g * 0.08 for g in _gpu_range]
    _ops_demand_curve = [
        (g * act2_workers_per_gpu.value * _files_open_per_sec_per_worker
         * _METADATA_OPS_PER_FILE_ACCESS * _metadata_pressure_factor) / 1000
        for g in _gpu_range
    ]

    _fig2 = go.Figure()

    # Bandwidth demand vs supply
    _fig2.add_trace(go.Scatter(
        x=_gpu_range, y=_bw_demand_curve,
        mode="lines", name="Bandwidth Demand (GB/s)",
        line=dict(color=COLORS["BlueLine"], width=2),
        yaxis="y1",
    ))
    _fig2.add_hline(
        y=_bw_supply_gbs, line_dash="dash", line_color=COLORS["GreenLine"],
        annotation_text=f"Storage BW Supply: {_bw_supply_gbs:.0f} GB/s",
        annotation_position="top left", yref="y1",
    )

    # Metadata ops demand vs MDS capacity (secondary y axis)
    _fig2.add_trace(go.Scatter(
        x=_gpu_range, y=_ops_demand_curve,
        mode="lines", name="Metadata Ops Demand (K ops/sec)",
        line=dict(color=COLORS["OrangeLine"], width=2, dash="dot"),
        yaxis="y2",
    ))
    _fig2.add_hline(
        y=_mds_capacity_k, line_dash="dash", line_color=COLORS["RedLine"],
        annotation_text=f"MDS Capacity: {_mds_capacity_k:.0f}K ops/sec ({act2_mds_count.value} MDS)",
        annotation_position="top right", yref="y2",
    )

    # Mark current GPU count
    _fig2.add_vline(
        x=act2_gpu_count.value, line_dash="solid", line_color="#6366f1",
        annotation_text=f"{act2_gpu_count.value} GPUs",
        annotation_position="top",
    )

    _fig2.update_layout(
        height=340,
        xaxis_title="GPU Count",
        yaxis=dict(title="Bandwidth (GB/s)", side="left", gridcolor="#f1f5f9"),
        yaxis2=dict(title="Metadata Ops (K ops/sec)", side="right", overlaying="y", gridcolor="#f1f5f9"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=60, b=50),
        title=dict(
            text=f"Bandwidth and Metadata Demand vs. Cluster Scale ({act2_dataset_files.value[:20]}...)",
            font=dict(size=13),
        ),
    )
    apply_plotly_theme(_fig2)

    # ── Metric cards ──────────────────────────────────────────────────────────
    _mds_color = COLORS["RedLine"] if _mds_saturated else COLORS["OrangeLine"] if _mds_util_pct > 60 else COLORS["GreenLine"]
    _bw_color = COLORS["RedLine"] if _bw_saturated else COLORS["GreenLine"]

    _cards2_html = f"""
    <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0;">
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 170px; text-align: center; background: white;
                    border-top: 4px solid {_bw_color};">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                Bandwidth Demand
            </div>
            <div style="font-size: 1.9rem; font-weight: 800; color: {_bw_color}; line-height: 1;">
                {_bw_demand_gbs:.1f}
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
                GB/s of {_bw_supply_gbs:.0f} GB/s supply
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 170px; text-align: center; background: white;
                    border-top: 4px solid {_mds_color};">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                Metadata Demand
            </div>
            <div style="font-size: 1.9rem; font-weight: 800; color: {_mds_color}; line-height: 1;">
                {_ops_k:.0f}K
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
                ops/sec of {_mds_capacity_k:.0f}K capacity
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 170px; text-align: center; background: white;
                    border-top: 4px solid {_mds_color};">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                MDS Utilization
            </div>
            <div style="font-size: 1.9rem; font-weight: 800; color: {_mds_color}; line-height: 1;">
                {min(_mds_util_pct, 199):.0f}%
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
                {act2_mds_count.value} MDS node(s)
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 12px;
                    min-width: 170px; text-align: center; background: white;
                    border-top: 4px solid {_rec_color};">
            <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px;">
                Concurrent Connections
            </div>
            <div style="font-size: 1.9rem; font-weight: 800; color: {COLORS['BlueLine']}; line-height: 1;">
                {_total_workers:,}
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
                {act2_gpu_count.value} GPUs × {act2_workers_per_gpu.value} workers
            </div>
        </div>
    </div>
    """

    # ── Physics formula ────────────────────────────────────────────────────────
    _formula2_md = f"""
    **The physics (from @sec-data-storage):**

    ```
    Metadata ops/sec  = GPU_count × workers_per_gpu × files_per_sec_per_worker
                        × metadata_ops_per_file × pressure_factor
                      = {act2_gpu_count.value} × {act2_workers_per_gpu.value} × {_files_open_per_sec_per_worker:.1f}
                        × {_METADATA_OPS_PER_FILE_ACCESS:.0f} × {_metadata_pressure_factor:.2f}
                      = {_ops_per_sec_demand:,.0f} ops/sec = {_ops_k:.0f}K ops/sec

    MDS capacity      = MDS_nodes × ops_per_MDS_node
                      = {act2_mds_count.value} × {_MDS_OPS_PER_NODE:,}
                      = {_mds_capacity:,} ops/sec = {_mds_capacity_k:.0f}K ops/sec

    MDS utilization   = demand / capacity = {_ops_k:.0f}K / {_mds_capacity_k:.0f}K = {_mds_util_pct:.0f}%

    Bandwidth demand  = GPU_count × 80 MB/s = {act2_gpu_count.value} × 0.08 GB/s = {_bw_demand_gbs:.1f} GB/s
    Bandwidth headroom= {_bw_headroom_pct:.0f}% of {_bw_supply_gbs:.0f} GB/s supply
    ```
    """

    # ── Architecture recommendation panel ────────────────────────────────────
    _rec_html = f"""
    <div style="background: #f8fafc; border: 1.5px solid {_rec_color}; border-radius: 12px;
                padding: 18px 22px; margin: 12px 0;">
        <div style="font-size: 0.82rem; font-weight: 700; color: #475569; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 8px;">
            Architecture Recommendation
        </div>
        <div style="font-size: 1.0rem; font-weight: 600; color: {_rec_color}; line-height: 1.6;">
            {_recommendation}
        </div>
    </div>
    """

    mo.vstack([
        mo.Html(_cards2_html),
        mo.as_html(_fig2),
        mo.md(_formula2_md),
        mo.Html(_rec_html),
    ])
    return (
        _mds_saturated,
        _ops_k,
        _mds_capacity_k,
        _needed_mds,
        _mds_util_pct,
        _bw_demand_gbs,
        _bw_supply_gbs,
        _bw_headroom_pct,
        _dataset_type,
        _total_workers,
    )


# ─── ACT II: FAILURE STATE ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, _mds_saturated, _ops_k, _mds_capacity_k, _needed_mds,
    act2_mds_count, _dataset_type, _total_workers, act2_gpu_count,
):
    if _dataset_type == "object_store":
        mo.callout(mo.md(
            "**Object store bypasses the MDS bottleneck entirely.** "
            "Content-addressed object stores (S3, GCS, Azure Blob) have no POSIX namespace "
            "and no metadata server. Each object is accessed by a globally unique key — "
            "no directory traversal, no file locking, no inode table. This is the primary "
            "architectural reason object stores dominate production ML training pipelines "
            "at scale. The bandwidth remains the same, but the scaling ceiling disappears. "
            "Switch back to 'individual files' to see the bottleneck re-emerge."
        ), kind="success")
    elif _mds_saturated:
        mo.callout(mo.md(
            f"**Metadata server saturation. This cluster cannot function at this configuration.**\n\n"
            f"Demand: **{_ops_k:.0f}K ops/sec** | MDS capacity: **{_mds_capacity_k:.0f}K ops/sec** "
            f"({act2_mds_count.value} MDS node{'s' if act2_mds_count.value > 1 else ''}).\n\n"
            f"**{_total_workers:,} concurrent connections** ({act2_gpu_count.value} GPUs × "
            f"{_total_workers // act2_gpu_count.value} workers) are all generating file-open "
            "requests to the same metadata server. Lustre's MDS is a single coordination "
            "point for all namespace operations — even though the bandwidth is fine, "
            "every file access requires an MDS round-trip for the inode lookup. "
            f"To fix: add {_needed_mds - act2_mds_count.value} more MDS nodes, switch to "
            "sequential shards (reduces opens by 50×+), or migrate to object storage entirely."
        ), kind="danger")
    elif _mds_util_pct > 60:
        mo.callout(mo.md(
            f"**MDS approaching saturation: {_mds_util_pct:.0f}% utilization.** "
            f"At {act2_gpu_count.value} GPUs with individual files, metadata operations are consuming "
            "more than half of MDS capacity. Doubling the GPU count will push this into "
            "saturation before bandwidth demand becomes significant. "
            "The architectural decision: switch to sequential shards now, or provision "
            f"{_needed_mds} MDS nodes proactively. Bandwidth headroom is not the constraint to watch."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**Architecture is currently feasible.** "
            f"MDS utilization: {_mds_util_pct:.0f}%. Bandwidth demand: {_ops_k * 0 + 0:.0f} — "
            "well within supply. Push the GPU count slider to 4,096 with individual files "
            "and 8 workers to see the metadata bottleneck emerge. The bandwidth will still "
            "be fine; the MDS will not."
        ), kind="success")
    return


# ─── ACT II: PREDICTION REVEAL ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction, _ops_k, _mds_capacity_k):
    _is_correct = act2_prediction.value == "B"
    # At 4096 GPUs × 8 workers × 50 opens/sec × 3 ops/file = 4,915,200 ops/sec = 4,915K ops/sec
    # vs MDS capacity of 50K ops/sec (1 MDS node)
    _actual_ops_demand_k = (4096 * 8 * 50.0 * 3.0) / 1000
    _actual_mds_cap_k = 50.0

    if _is_correct:
        mo.callout(mo.md(
            f"**Correct. You predicted: metadata server saturation.**\n\n"
            f"At 4,096 GPUs × 8 workers = 32,768 concurrent connections, all generating "
            "file-open requests against the Lustre namespace: "
            f"**{_actual_ops_demand_k:,.0f}K ops/sec demand** against "
            f"**{_actual_mds_cap_k:.0f}K ops/sec MDS capacity** — a "
            f"{_actual_ops_demand_k/_actual_mds_cap_k:.0f}× overload. "
            "Meanwhile, bandwidth demand is 4,096 × 80 MB/s = 328 GB/s — well within "
            "the 1,344 GB/s supply. The bandwidth safety margin the team computed is real; "
            "the metadata capacity they never measured is the actual constraint. "
            "Every add-OST investment solves the wrong problem."
        ), kind="success")
    elif act2_prediction.value == "A":
        mo.callout(mo.md(
            f"**Not quite. You predicted: bandwidth saturation.**\n\n"
            "4,096 GPUs × 80 MB/s = 328 GB/s demand against 1,344 GB/s supply — a 4× safety margin. "
            "The bandwidth math the CTO computed is correct. But the metadata server running "
            "at 50K ops/sec capacity against 4,915K ops/sec demand (a 98× overload) fails "
            "first and catastrophically. Adding more OST nodes improves bandwidth capacity "
            "but does nothing for the MDS bottleneck. "
            "**Correct answer: B — metadata server becomes the bottleneck.**"
        ), kind="warn")
    elif act2_prediction.value == "C":
        mo.callout(mo.md(
            f"**Not quite. You predicted: storage scales linearly with OST nodes.**\n\n"
            "For *bandwidth*, this is correct — adding OSTs increases aggregate throughput. "
            "But the Lustre Metadata Server (MDS) is a separate component that manages the "
            "filesystem namespace. It does not scale with OST node count. A single MDS "
            "serves all namespace operations for the entire cluster, and its throughput "
            "ceiling is a hard constraint no OST addition can move. "
            "**Correct answer: B — metadata server becomes the bottleneck.**"
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**Not quite. You predicted: network latency is the bottleneck.**\n\n"
            "InfiniBand HDR-200 introduces ~1–5 microseconds of fabric latency — "
            "far below the millisecond-scale batch processing window. At 128 GB/s per "
            "compute node IB port, the network fabric is not the constraint. "
            "The Lustre metadata server, handling every file-open and close operation "
            "for 32,768 concurrent workers, saturates at 50K ops/sec against a demand "
            "of 4,915K ops/sec. Network latency is measurable but negligible; "
            "metadata throughput is the hard ceiling. "
            "**Correct answer: B — metadata server becomes the bottleneck.**"
        ), kind="warn")
    return


# ─── ACT II: REFLECTION ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Reflection

    You have seen metadata saturation emerge at scale where bandwidth was never the issue.
    Now identify the architectural principle this demonstrates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) Object stores are faster for sequential reads than POSIX filesystems": "A",
            "B) Object stores eliminate the metadata bottleneck — objects are addressed by content hash, removing directory traversal": "B",
            "C) POSIX filesystems cannot store files larger than 4 GB": "C",
            "D) Object stores support more concurrent connections per storage node than POSIX": "D",
        },
        label="Why is object storage (S3/GCS) often preferred over POSIX filesystems for training data at scale?",
    )
    act2_reflection
    return (act2_reflection,)


@app.cell(hide_code=True)
def _(mo, act2_reflection):
    mo.stop(
        act2_reflection.value is None,
        mo.callout(mo.md("Select your answer to complete Act II."), kind="warn"),
    )

    if act2_reflection.value == "B":
        mo.callout(mo.md(
            "**Correct.** Object stores (S3, GCS, Azure Blob Storage) address objects by "
            "globally unique keys — no POSIX namespace, no inode table, no directory hierarchy. "
            "There is no metadata server to saturate. A GET request goes directly to the "
            "object's physical location, resolved through a distributed key-value index that "
            "scales horizontally without a coordination bottleneck. "
            "This explains why production training pipelines at Google, Meta, and Microsoft "
            "all use object storage as the primary data tier — not because sequential bandwidth "
            "is higher (it is similar), but because the namespace scaling barrier does not exist. "
            "The data gravity consequence: if training data lives in S3, compute must also run "
            "in the same cloud region. Moving 100 TB to a local Lustre cluster costs the same "
            "as running the training job where the data already lives."
        ), kind="success")
    elif act2_reflection.value == "A":
        mo.callout(mo.md(
            "**Incorrect.** Sequential read bandwidth for S3 and a well-configured Lustre "
            "cluster are similar for large files — both can sustain hundreds of GB/s in aggregate. "
            "The advantage of object stores is not in raw sequential throughput; it is in the "
            "absence of a namespace coordination bottleneck. A POSIX filesystem routes all "
            "namespace operations through one or a few MDS nodes. An object store does not "
            "have a metadata server at all. **Correct answer: B.**"
        ), kind="warn")
    elif act2_reflection.value == "C":
        mo.callout(mo.md(
            "**Incorrect.** Modern POSIX filesystems (Lustre, GPFS, ext4, XFS) support files "
            "measured in petabytes — there is no practical size limit for training workloads. "
            "The limitation of POSIX at ML scale is metadata coordination overhead, not file "
            "size. **Correct answer: B — the metadata bottleneck is the architectural constraint.**"
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Incorrect.** POSIX filesystems with appropriate tuning (Lustre with many OST "
            "nodes, GPFS with distributed metadata) can handle tens of thousands of concurrent "
            "connections. The issue is not raw concurrency capacity — it is that every connection "
            "must serialize through the MDS for namespace operations. Object stores route each "
            "request directly to the storage backend without a centralized coordinator. "
            "**Correct answer: B — content-addressed objects eliminate namespace bottlenecks.**"
        ), kind="warn")
    return


# ─── ACT II: MATHPEEK ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — Metadata Rate and Data Gravity at Scale": mo.md("""
        **Metadata Operation Rate** (from @sec-data-storage):

        $$\\text{MDS demand} = N_{GPU} \\times W_{GPU} \\times f_{open} \\times m_{ops}$$

        Where:
        - **N_GPU** — number of GPU accelerators
        - **W_GPU** — data loader workers per GPU
        - **f_open** — file opens per second per worker (varies by dataset organization)
        - **m_ops** — metadata operations per file access (open + stat + close ≈ 3)

        **Example: 4096 GPUs, 8 workers, individual files:**
        ```
        MDS demand = 4,096 × 8 × 50 ops/sec × 3
                   = 4,915,200 ops/sec ≈ 4,915K ops/sec
        MDS supply = 1 node × 50K ops/sec = 50K ops/sec
        Saturation = 4,915 / 50 = 98× overload → training stalls
        ```

        **Sequential shards fix (same GPUs):**
        ```
        f_open     ≈ 0.2 opens/sec per worker (5s per shard)
        MDS demand = 4,096 × 8 × 0.2 × 3 = 19,661 ops/sec ≈ 20K ops/sec
        MDS supply = 1 node × 50K ops/sec = 50K ops/sec
        Utilization = 40%  ← within bounds
        ```

        **Object Store vs POSIX Comparison:**
        | Property | Lustre (POSIX) | S3 / GCS (Object) |
        |----------|:-------------:|:-----------------:|
        | Metadata server | Required (MDS) | None |
        | Namespace scalability | MDS-bound | Distributed hash |
        | Sequential BW | 672 GB/s (48 OSTs) | ~same per region |
        | Random access IOPS | NVMe-bound | Tiered by class |
        | Training suitability | Good (with shards) | Excellent (any pattern) |

        **Data Gravity Implication:**
        Once training data is in S3 (us-east-1), moving 100 TB to a local Lustre cluster:
        ```
        T_transfer = 100,000 GB / 12.5 GB/s (100 Gbps) = 8,000 s ≈ 2.2 hours
        Cost       = 100,000 GB × $0.09/GB (S3 egress) = $9,000

        Compare to running the training job in us-east-1 for 24 hours:
        Cost       = 512 × H100 spot × $3.50/GPU-hr × 24h = $43,008

        Data gravity decision: move compute to data if transfer cost + latency
        exceeds the cost of provisioning compute near the data.
        ```
        """),
    })
    return


# ─── DESIGN LEDGER SAVE + HUD ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, ledger, context_toggle,
    act1_prediction, act1_reflection,
    act2_prediction, act2_reflection,
    act2_gpu_count, act2_workers_per_gpu,
    act1_num_workers,
    _gpu_util, _effective_storage_bw,
    _mds_saturated, _ops_k, _dataset_type, _total_workers,
    COLORS,
):
    _act1_correct = (act1_prediction.value == "B") if act1_prediction.value else False
    _act1_refl_correct = (act1_reflection.value == "B") if act1_reflection.value else False
    _act2_correct = (act2_prediction.value == "B") if act2_prediction.value else False
    _act2_refl_correct = (act2_reflection.value == "B") if act2_reflection.value else False

    # Map dataset type to a clean label for the ledger
    _storage_type_map = {
        "shards": "sequential_shards",
        "individual": "individual_files",
        "object_store": "object_store",
    }
    _storage_type = _storage_type_map.get(_dataset_type, "individual_files")

    ledger.save(
        chapter="v2_04",
        design={
            "context": context_toggle.value,
            "storage_type": _storage_type,
            "gpu_count": act2_gpu_count.value,
            "data_workers_per_gpu": act2_workers_per_gpu.value,
            "io_throughput_gbs": round(float(_effective_storage_bw), 2),
            "act1_prediction": act1_prediction.value if act1_prediction.value else "none",
            "act1_correct": _act1_correct,
            "act2_result": round(float(_ops_k), 1),   # metadata ops/sec demand in K
            "act2_decision": _storage_type,
            "constraint_hit": bool(_mds_saturated),
            "mds_saturated": bool(_mds_saturated),
        },
    )

    # HUD footer
    _ctx_label = "NVMe — Single Node" if context_toggle.value == "nvme" else "Distributed FS — Lustre"
    _act1_status = "correct" if _act1_correct else ("pending" if not act1_prediction.value else "incorrect")
    _act2_status = "correct" if _act2_correct else ("pending" if not act2_prediction.value else "incorrect")
    _mds_status = "saturated" if _mds_saturated else "healthy"

    def _sc(s):
        return {
            "correct": "#4ade80", "pending": "#94a3b8", "incorrect": "#f87171",
            "saturated": "#f87171", "healthy": "#4ade80",
        }.get(s, "#94a3b8")

    mo.Html(f"""
    <div style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap;
                padding: 14px 24px; background: #0f172a; border-radius: 12px;
                margin-top: 40px; font-family: 'SF Mono', monospace; font-size: 0.8rem;
                border: 1px solid #1e293b;">
        <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">
            DESIGN LEDGER · V2-CH04
        </span>
        <span style="color: #475569;">|</span>
        <span>
            <span style="color: #94a3b8;">Context: </span>
            <span style="color: #e2e8f0;">{_ctx_label}</span>
        </span>
        <span>
            <span style="color: #94a3b8;">Act I prediction: </span>
            <span style="color: {_sc(_act1_status)};">{_act1_status}</span>
        </span>
        <span>
            <span style="color: #94a3b8;">Act II prediction: </span>
            <span style="color: {_sc(_act2_status)};">{_act2_status}</span>
        </span>
        <span>
            <span style="color: #94a3b8;">GPU util (Act I): </span>
            <span style="color: #e2e8f0;">{_gpu_util:.0f}%</span>
        </span>
        <span>
            <span style="color: #94a3b8;">Effective I/O BW: </span>
            <span style="color: #e2e8f0;">{_effective_storage_bw:.1f} GB/s</span>
        </span>
        <span>
            <span style="color: #94a3b8;">MDS status: </span>
            <span style="color: {_sc(_mds_status)};">{_mds_status} ({_ops_k:.0f}K ops/sec)</span>
        </span>
        <span>
            <span style="color: #94a3b8;">Workers: </span>
            <span style="color: #e2e8f0;">{_total_workers:,} ({act2_gpu_count.value}×{act2_workers_per_gpu.value})</span>
        </span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
