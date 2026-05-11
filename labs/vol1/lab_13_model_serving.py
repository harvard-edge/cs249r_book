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

    H100_TFLOPS_FP16 = mlsysim.Hardware.Cloud.H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_BW_GBS      = mlsysim.Hardware.Cloud.H100.memory.bandwidth.m_as("GB/s")
    H100_RAM_GB      = mlsysim.Hardware.Cloud.H100.memory.capacity.m_as("GB")
    H100_TDP_W       = mlsysim.Hardware.Cloud.H100.tdp.m_as("W")

    # Edge tier — contrast serving constraints: limited memory, lower bandwidth
    JETSON_TFLOPS    = mlsysim.Hardware.Edge.JetsonOrinNX.compute.peak_flops.m_as("TFLOPs/s")
    JETSON_BW_GBS    = mlsysim.Hardware.Edge.JetsonOrinNX.memory.bandwidth.m_as("GB/s")
    JETSON_RAM_GB    = mlsysim.Hardware.Edge.JetsonOrinNX.memory.capacity.m_as("GB")
    JETSON_TDP_W     = mlsysim.Hardware.Edge.JetsonOrinNX.tdp.m_as("W")

    PCIE_GEN5_GBS = 64.0
    PCIE_GEN4_GBS = 32.0
    NVME_SEQ_GBS  = 7.0
    NET_FS_GBS    = 1.25

    RESNET50_PARAMS = mlsysim.Models.ResNet50.parameters.m_as("count")
    RESNET50_FLOPS  = mlsysim.Models.ResNet50.inference_flops.m_as("flop")

    LLAMA2_70B_PARAMS = mlsysim.Models.Language.Llama2_70B.parameters.m_as("count")
    LLAMA2_70B_LAYERS = mlsysim.Models.Language.Llama2_70B.layers
    LLAMA2_70B_HIDDEN = mlsysim.Models.Language.Llama2_70B.hidden_dim
    LLAMA2_70B_HEADS  = mlsysim.Models.Language.Llama2_70B.heads

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        COLORS, H100_BW_GBS, H100_RAM_GB, H100_TDP_W, H100_TFLOPS_FP16,
        JETSON_BW_GBS, JETSON_RAM_GB, JETSON_TDP_W, JETSON_TFLOPS,
        LAB_CSS, LLAMA2_70B_HEADS, LLAMA2_70B_HIDDEN, LLAMA2_70B_LAYERS,
        LLAMA2_70B_PARAMS, NET_FS_GBS, NVME_SEQ_GBS, PCIE_GEN4_GBS,
        PCIE_GEN5_GBS, RESNET50_FLOPS, RESNET50_PARAMS,
        apply_plotly_theme, go, ledger, math, mo, np,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 13
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Tail Latency Trap
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Queuing &middot; Batching &middot; KV Cache &middot; Cold Start
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Your serving system looks healthy at 50% utilization and is on fire
                at 80% &mdash; and the fire is invisible to every metric except the
                one you are not measuring.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~50 min</span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter 13: Model Serving</span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">M/M/1 Queuing</span>
                <span class="badge badge-warn">Batching Tax</span>
                <span class="badge badge-fail">KV Cache OOM</span>
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
                Learning Objectives</div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Quantify the P99/mean latency divergence</strong>
                    &mdash; at 80% utilization, P99 is 23x the service time while mean is only 5x.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose the batching tax</strong> &mdash;
                    batch formation delay consumes 62% of a 50&nbsp;ms SLO budget before inference starts.</div>
                <div style="margin-bottom: 3px;">3. <strong>Calculate KV cache memory</strong> &mdash;
                    at 128K context, a single Llama-2 70B request fills ~320&nbsp;GB of HBM.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Iron Law equation from the Introduction chapter &middot;
                    Memory wall from the Hardware Acceleration chapter &middot;
                    Transformer architecture from the Network Architectures chapter</div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~50 min</strong><br/>
                    A: ~12 &middot; B: ~12 &middot; C: ~12 &middot; D: ~8 min</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 12px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question</div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;Your inference server reports 25&nbsp;ms average latency. Why are 1 in 100
                users experiencing 115&nbsp;ms &mdash; and why does adding batching make it worse?&rdquo;
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
    **Recommended Reading** &mdash; Complete before this lab:

    - **The Model Serving chapter** &mdash; Queuing theory for inference, M/M/1 model,
      batching strategies, KV cache management, and cold start latency.
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LAB CELL
# ═════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(
    COLORS, H100_BW_GBS, H100_RAM_GB, JETSON_BW_GBS, JETSON_RAM_GB,
    LLAMA2_70B_HEADS, LLAMA2_70B_HIDDEN,
    LLAMA2_70B_LAYERS, NET_FS_GBS, NVME_SEQ_GBS, PCIE_GEN4_GBS,
    PCIE_GEN5_GBS, apply_plotly_theme, go, math, mo, np,
):
    # ── Part A widgets ────────────────────────────────────────────────────────
    partA_pred = mo.ui.radio(
        options={
            "A) ~10 ms (2x service time)": "10ms",
            "B) ~25 ms (5x, matches the mean)": "25ms",
            "C) ~50 ms (10x service time)": "50ms",
            "D) ~115 ms (23x service time)": "115ms",
        },
        label="ResNet-50 server, 5 ms service time, 80% utilization. P99 latency?",
    )
    return (partA_pred,)

@app.cell(hide_code=True)
def _(mo):
    partA_rho = mo.ui.slider(start=0.10, stop=0.95, value=0.80, step=0.05,
                              label="Server utilization (rho)")
    partA_svc = mo.ui.slider(start=1.0, stop=20.0, value=5.0, step=1.0,
                              label="Service time (ms)")
    partA_slo = mo.ui.slider(start=10.0, stop=200.0, value=50.0, step=10.0,
                              label="SLO budget (ms)")

    # ── Part B widgets ────────────────────────────────────────────────────────
    partB_pred = mo.ui.radio(
        options={
            "A) Throughput improves 6.4x, latency within SLO": "ok",
            "B) Throughput improves but latency slightly exceeds SLO": "slight",
            "C) Formation delay alone (31 ms) consumes 62% of SLO": "delay",
            "D) System crashes from memory overflow": "crash",
        },
        label="Batch size 1 to 32, arrival 500 QPS, SLO 50 ms. What happens?",
    )
    return (partA_rho, partA_slo, partA_svc, partB_pred)

@app.cell(hide_code=True)
def _(mo):
    partB_batch = mo.ui.slider(start=1, stop=64, value=1, step=1, label="Batch size")
    partB_arr = mo.ui.slider(start=100, stop=2000, value=500, step=50, label="Arrival rate (QPS)")
    partB_slo = mo.ui.slider(start=10, stop=100, value=50, step=5, label="SLO budget (ms)")

    # ── Part C widgets ────────────────────────────────────────────────────────
    partC_pred = mo.ui.radio(
        options={
            "A) 8 (one per GPU)": "8",
            "B) 4 (split model and cache)": "4",
            "C) 1 (KV cache fills all memory)": "1",
            "D) 16 (tensor parallelism)": "16",
        },
        label="Llama-2 70B, 128K context, 8xH100 (640 GB). Max concurrent batch?",
    )
    return (partB_arr, partB_batch, partB_slo, partC_pred)

@app.cell(hide_code=True)
def _(mo):
    partC_model = mo.ui.dropdown(options={"7B": 7, "13B": 13, "70B": 70}, value="70B",
                                  label="Model size")
    partC_prec = mo.ui.dropdown(
        options={"FP16 (2B)": 2, "INT8 (1B)": 1, "INT4 (0.5B)": 0.5},
        value="FP16 (2B)", label="Weight precision (bytes)")
    partC_ctx = mo.ui.slider(start=2048, stop=131072, value=131072, step=2048,
                              label="Context length (tokens)")
    partC_gpus = mo.ui.dropdown(
        options={"1 GPU (80 GB)": 1, "4 GPUs (320 GB)": 4, "8 GPUs (640 GB)": 8},
        value="8 GPUs (640 GB)", label="GPU count")

    # ── Part D widgets ────────────────────────────────────────────────────────
    partD_pred = mo.ui.radio(
        options={
            "A) ~200 ms (normal inference latency)": "200ms",
            "B) ~2 seconds (some loading overhead)": "2s",
            "C) ~26 seconds (NVMe read bottleneck + deserialization)": "26s",
            "D) ~60 seconds (loading from network storage)": "60s",
        },
        label="Auto-scaling Llama-2 70B during traffic spike. First-user wait?",
    )
    return (partC_ctx, partC_gpus, partC_model, partC_prec, partD_pred)

# ─── widget cell: extracted from tabs cell body (#1332 polish) ────
@app.cell(hide_code=True)
def _(mo):
    partD_model = mo.ui.dropdown(options={"7B": 7, "13B": 13, "70B": 70}, value="70B",
                                  label="Model size")
    partD_stor = mo.ui.dropdown(
        options={"NVMe SSD": "nvme", "Network FS": "nfs", "Cached in RAM": "ram"},
        value="NVMe SSD", label="Storage type")
    partD_pcie = mo.ui.dropdown(options={"PCIe Gen4": "gen4", "PCIe Gen5": "gen5"},
                                 value="PCIe Gen5", label="Interconnect")
    return (partD_model, partD_pcie, partD_stor)


@app.cell(hide_code=True)
def _(
    mo, partA_pred, partA_rho, partA_slo,
    partA_svc, partB_arr, partB_batch, partB_pred,
    partB_slo, partC_ctx, partC_gpus, partC_model,
    partC_prec, partC_pred, partD_pred, partD_model,
    partD_pcie, partD_stor,
):

    # ═════════════════════════════════════════════════════════════════════════
    # PART A — The Tail Latency Explosion
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; SRE Lead, InferenceCloud</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our ResNet-50 cluster reports 25 ms average latency at 80% utilization.
                Product says the system is healthy. But top-tier customers are complaining
                about timeouts. Can you investigate?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## The M/M/1 Queuing Model Predicts Tail Latency Divergence

```
Mean wait  = service_time / (1 - rho)
P99 wait   = ln(100) * service_time / (1 - rho) = 4.6 * service_time / (1 - rho)
```

The P99/mean ratio is always **4.6x** for M/M/1, but absolute values explode
as rho approaches 1.0. At rho=0.80, the 1/(1-rho) amplifier is 5x, making
P99 = 4.6 * 5 * service_time = **23x service time**.
        """))

        items.append(mo.callout(mo.md(
            "**Note:** M/M/1 assumes Poisson arrivals and exponential service times. Production "
            "serving systems have bursty arrivals, bimodal latencies (cache hits vs misses), and "
            "multiple servers. The M/M/1 model is a pedagogical lower bound — real P99 latencies "
            "are typically 2-5x worse due to autocorrelation and load balancer overhead."
        ), kind="info"))

        items.append(partA_pred)
        if partA_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the P99 latency instruments."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_rho, partA_svc, partA_slo], widths="equal"))

        _rho = partA_rho.value
        _svc = partA_svc.value
        _slo = partA_slo.value
        _mean = _svc / (1 - _rho) if _rho < 1.0 else 9999.0
        _p99 = 4.6 * _svc / (1 - _rho) if _rho < 1.0 else 9999.0
        _p99_ratio = _p99 / _svc if _svc > 0 else 0

        # Divergence chart
        _rhos = np.linspace(0.05, 0.95, 50)
        _means = [_svc / (1 - r) for r in _rhos]
        _p99s = [4.6 * _svc / (1 - r) for r in _rhos]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_rhos, y=_means, mode='lines', name='Mean Latency',
                                   line=dict(color=COLORS['BlueLine'], width=2),
                                   hovertemplate="rho=%{x:.2f}: %{y:.1f} ms<extra></extra>"))
        _fig.add_trace(go.Scatter(x=_rhos, y=_p99s, mode='lines', name='P99 Latency',
                                   line=dict(color=COLORS['RedLine'], width=3),
                                   hovertemplate="rho=%{x:.2f}: %{y:.1f} ms<extra></extra>"))
        _fig.add_hline(y=_slo, line_dash="dash", line_color=COLORS['GreenLine'],
                       annotation_text=f"SLO = {_slo:.0f} ms")
        if _rho < 1.0:
            _fig.add_trace(go.Scatter(x=[_rho], y=[_p99], mode='markers',
                                       name=f'P99 @ rho={_rho:.2f}',
                                       marker=dict(color=COLORS['RedLine'], size=14, symbol='diamond'),
                                       hovertemplate="rho=%{x:.2f}: %{y:.1f} ms<extra></extra>"))
            _fig.add_trace(go.Scatter(x=[_rho], y=[_mean], mode='markers',
                                       name=f'Mean @ rho={_rho:.2f}',
                                       marker=dict(color=COLORS['BlueLine'], size=10),
                                       hovertemplate="rho=%{x:.2f}: %{y:.1f} ms<extra></extra>"))
        _fig.update_layout(height=380, xaxis=dict(title="Utilization (rho)", range=[0, 1]),
                           yaxis=dict(title="Latency (ms)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _slo_viol = _p99 > _slo
        _p99_col = COLORS['RedLine'] if _slo_viol else COLORS['GreenLine']
        _mean_col = COLORS['GreenLine'] if _mean < _slo else COLORS['OrangeLine']

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {_mean_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Mean Latency</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_mean_col};">{_mean:.1f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_mean/_svc:.1f}x service time</div></div>
            <div style="padding:16px; border:2px solid {_p99_col}; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {_p99_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">P99 Latency</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_p99_col};">{_p99:.1f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_p99_ratio:.1f}x service time</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">P99/Mean Ratio</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">4.6x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">ln(100) for M/M/1</div></div>
        </div>"""))

        if _slo_viol:
            items.append(mo.callout(mo.md(
                f"**SLO VIOLATED.** P99 = {_p99:.1f} ms > {_slo:.0f} ms SLO. "
                f"Mean looks healthy ({_mean:.1f} ms) but 1 in 100 users sees "
                f"{_p99/_mean:.1f}x worse."), kind="danger"))

        items.append(mo.md(f"""
**M/M/1 &mdash; Live** (`rho={_rho:.2f}, service={_svc:.1f} ms`)

```
Mean = {_svc:.1f} / (1 - {_rho:.2f}) = {_svc:.1f} / {1-_rho:.2f} = {_mean:.1f} ms
P99  = 4.6 * {_svc:.1f} / {1-_rho:.2f} = {_p99:.1f} ms  ({_p99_ratio:.1f}x service)
```
*Source: @eq-mm1-wait and @eq-p99-latency*
        """))

        _actual_p99 = 4.6 * 5.0 / 0.2  # 115 ms
        if partA_pred.value == "115ms":
            items.append(mo.callout(mo.md(
                f"**Correct.** P99 = 4.6 * 5 / 0.2 = {_actual_p99:.0f} ms (23x service time). "
                "Most engineers only monitor mean latency."), kind="success"))
        elif partA_pred.value == "25ms":
            items.append(mo.callout(mo.md(
                f"**You confused mean with P99.** Mean is 25 ms. P99 = 4.6 * mean = "
                f"{_actual_p99:.0f} ms. The ln(100) multiplier is the gap."), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Actual P99 is {_actual_p99:.0f} ms** &mdash; 23x service time. "
                "The 1/(1-rho) amplifier at 0.80 is 5x, times ln(100)=4.6."), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: M/M/1 Queue Latency": mo.md("""
**Formula (mean sojourn time):**
$$
W = \\frac{1}{\\mu - \\lambda} = \\frac{1}{\\mu} \\cdot \\frac{1}{1 - \\rho}
$$

**P99 approximation (exponential tail):**
$$
P_{99} \\approx \\frac{\\ln(100)}{\\mu(1 - \\rho)} = \\frac{4.6}{\\mu(1 - \\rho)}
$$

**Variables:**
- **$\\lambda$**: arrival rate (requests/s)
- **$\\mu$**: service rate ($1/T_{\\text{service}}$)
- **$\\rho = \\lambda / \\mu$**: server utilization
- **$W$**: mean time in system (wait + service)

At $\\rho = 0.80$: mean wait = $5\\times$ service time, P99 = $23\\times$ service time. The nonlinear $1/(1-\\rho)$ amplifier is why 80% utilization feels like a crisis.
""")
        }))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B — The Batching Tax
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Escalation &middot; Performance Engineer, InferenceCloud</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We know tail latency is the problem. Throughput team says batching is the
                fix &mdash; batch-32 gives 6.4x throughput. Will this fix SLO violations?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## Batching Imposes a Formation Delay Tax

```
Formation delay = (B - 1) / (2 * lambda)     [seconds]
Total latency   = Formation delay + Inference(B) + Queuing delay
```

Each request waits for the batch to fill. This delay is consumed **before**
the GPU fires a single kernel.
        """))

        items.append(partB_pred)
        if partB_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the batching instruments."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_batch, partB_arr, partB_slo], widths="equal"))

        _B = partB_batch.value
        _lam = partB_arr.value
        _slo = partB_slo.value
        _form = (_B - 1) / (2 * _lam) * 1000  # ms
        _infer = 0.5 + 0.1 * (_B - 1)  # ms (ResNet-50 on H100)
        _eff_svc = _form + _infer
        _eff_arr = _lam / max(_B, 1)
        _eff_rho = _eff_arr * (_eff_svc / 1000) if _eff_svc > 0 else 0
        _q_del = _eff_svc * _eff_rho / (1 - _eff_rho) if 0 < _eff_rho < 1 else 999.0
        _total = _form + _infer + min(_q_del, 999)
        _slo_viol = _total > _slo
        _form_pct = _form / _slo * 100 if _slo > 0 else 0
        _tput = _B / (_eff_svc / 1000) if _eff_svc > 0 else 0
        _tput_ratio = _tput / (1 / (0.5 / 1000))

        _fig = go.Figure()
        for _name, _val, _col in [
            ("Formation Delay", _form, COLORS['OrangeLine']),
            ("Inference", _infer, COLORS['BlueLine']),
            ("Queuing", min(_q_del, 500), COLORS['RedLine']),
        ]:
            _fig.add_trace(go.Bar(name=_name, x=[_name], y=[_val],
                                   marker_color=_col, opacity=0.88,
                                   hovertemplate="%{x}: %{y:.1f} ms<extra></extra>"))
        _fig.add_hline(y=_slo, line_dash="dash", line_color=COLORS['GreenLine'],
                       annotation_text=f"SLO = {_slo} ms")
        _fig.update_layout(height=340, yaxis=dict(title="Latency (ms)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _form_col = COLORS['RedLine'] if _form_pct > 50 else COLORS['OrangeLine']
        _tot_col = COLORS['RedLine'] if _slo_viol else COLORS['GreenLine']
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {_form_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Formation Delay</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_form_col};">{_form:.1f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_form_pct:.0f}% of SLO</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Throughput</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_tput:.0f} QPS</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_tput_ratio:.1f}x over batch=1</div></div>
            <div style="padding:16px; border:2px solid {_tot_col}; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {_tot_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total Latency</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_tot_col};">{min(_total,999):.1f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{"SLO VIOLATED" if _slo_viol else "Within SLO"}</div></div>
        </div>"""))

        if _slo_viol:
            items.append(mo.callout(mo.md(
                f"**SLO VIOLATED.** {_form:.1f} + {_infer:.1f} + {min(_q_del,999):.1f} = "
                f"{min(_total,999):.1f} ms > {_slo} ms. Formation delay is "
                f"{_form_pct:.0f}% of the SLO."), kind="danger"))

        items.append(mo.md(f"""
**Batching Tax &mdash; Live** (`B={_B}, lambda={_lam} QPS, SLO={_slo} ms`)

```
Formation delay = ({_B}-1) / (2*{_lam}) * 1000 = {_form:.1f} ms  ({_form_pct:.0f}% of SLO)
Inference       = {_infer:.1f} ms  (batch={_B} on H100)
Queuing delay   = {min(_q_del,999):.1f} ms  (rho_eff={_eff_rho:.2f})
Total           = {min(_total,999):.1f} ms
```
*Source: formation delay from @sec-model-serving-batching*
        """))

        _ref_form = (32 - 1) / (2 * 500) * 1000
        if partB_pred.value == "delay":
            items.append(mo.callout(mo.md(
                f"**Correct.** Formation delay for B=32 at 500 QPS = {_ref_form:.0f} ms = "
                f"62% of a 50 ms SLO."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Formation delay kills the SLO.** B=32 at 500 QPS: {_ref_form:.0f} ms "
                "of waiting before the GPU fires."), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Batch Formation Delay": mo.md("""
**Formula (expected wait for batch to fill):**
$$
T_{\\text{formation}} = \\frac{B - 1}{2\\lambda}
$$

**Throughput gain vs. latency cost:**
$$
\\text{Throughput} = \\frac{B \\cdot \\mu}{1 + B \\cdot \\mu \\cdot T_{\\text{formation}}}
$$

**Variables:**
- **$B$**: batch size
- **$\\lambda$**: arrival rate (requests/s)
- **$T_{\\text{formation}}$**: expected time for the last request in a batch to arrive

At $B=32$ and $\\lambda=500$ QPS, the average request waits $(32-1)/(2 \\times 500) = 31$ ms just for the batch to fill -- 62% of a 50 ms SLO budget consumed before inference starts.
""")
        }))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C — The LLM Memory Wall
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_c():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Head of ML Infrastructure</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We need to serve Llama-2 70B with 128K context on 8xH100. Compute team
                says we can batch 8 requests &mdash; one per GPU. Can you verify?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## KV Cache Capacity Determines LLM Concurrency

```
KV cache (bytes) = 2 * layers * heads * head_dim * seq_len * batch * bytes_per_elem
Total memory     = model_weights + KV_cache
Max batch        = floor((HBM - weights) / KV_per_request)
```

At long context, KV cache exceeds model weights. Compute parallelism is irrelevant
when memory capacity is the binding constraint.
        """))

        items.append(partC_pred)
        if partC_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the KV cache calculator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_model, partC_prec, partC_ctx, partC_gpus], widths="equal"))

        _pb = partC_model.value
        _bpp = partC_prec.value
        _ctx = partC_ctx.value
        _ng = partC_gpus.value

        _cfgs = {7: (32, 32, 4096), 13: (40, 40, 5120), 70: (80, 64, 8192)}
        _layers, _heads, _hidden = _cfgs[_pb]
        _hdim = _hidden // _heads
        _w_gb = _pb * 1e9 * _bpp / (1024**3)
        _kv_gb = (2 * _layers * _heads * _hdim * _ctx * _bpp) / (1024**3)
        _hbm = _ng * H100_RAM_GB
        _avail = _hbm - _w_gb
        _max_b = max(0, int(_avail / _kv_gb)) if _kv_gb > 0 else 0
        _bw = _ng * H100_BW_GBS
        _tok_ms = (_w_gb + _kv_gb) / _bw * 1000 if _bw > 0 else 0

        _br = list(range(1, min(_max_b + 3, 20)))
        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Weights", x=[str(b) for b in _br],
                               y=[_w_gb]*len(_br), marker_color=COLORS['BlueLine'], opacity=0.88,
                               hovertemplate="Batch %{x}: %{y:.1f} GB<extra></extra>"))
        _fig.add_trace(go.Bar(name="KV Cache", x=[str(b) for b in _br],
                               y=[_kv_gb*b for b in _br], marker_color=COLORS['OrangeLine'], opacity=0.88,
                               hovertemplate="Batch %{x}: %{y:.1f} GB<extra></extra>"))
        _fig.add_hline(y=_hbm, line_dash="dash", line_color=COLORS['RedLine'],
                       annotation_text=f"HBM = {_hbm:.0f} GB")
        _fig.update_layout(barmode="stack", height=380,
                           xaxis=dict(title="Concurrent Batch Size"),
                           yaxis=dict(title="Memory (GB)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _oom = _max_b == 0
        _mem_col = COLORS['RedLine'] if _oom else COLORS['GreenLine']
        _kv_col = COLORS['RedLine'] if _kv_gb > _avail else COLORS['OrangeLine']
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Weights</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_w_gb:.0f} GB</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_pb}B x {_bpp} B/param</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {_kv_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">KV / Request</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_kv_col};">{_kv_gb:.1f} GB</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_ctx:,} tokens</div></div>
            <div style="padding:16px; border:2px solid {_mem_col}; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {_mem_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Max Batch</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_mem_col};">{_max_b}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_hbm:.0f} GB HBM</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Token Time</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_tok_ms:.1f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">mem-BW bound</div></div>
        </div>"""))

        if _oom:
            items.append(mo.callout(mo.md(
                f"**OOM.** Weights ({_w_gb:.0f} GB) + KV ({_kv_gb:.1f} GB) = "
                f"{_w_gb+_kv_gb:.0f} GB > {_hbm:.0f} GB. Reduce context or quantize."), kind="danger"))
        elif _max_b <= 2:
            items.append(mo.callout(mo.md(
                f"**Severe concurrency limit.** Max {_max_b} concurrent request(s). "
                "KV cache dominates memory."), kind="warn"))

        items.append(mo.md(f"""
**KV Cache &mdash; Live** (`{_pb}B, {_bpp}B/param, {_ctx:,} tokens, {_ng} GPUs`)

```
Weights      = {_pb}B * {_bpp} = {_w_gb:.0f} GB
KV/request   = 2*{_layers}*{_heads}*{_hdim}*{_ctx:,}*{_bpp} = {_kv_gb:.1f} GB
Available    = {_hbm:.0f} - {_w_gb:.0f} = {_avail:.0f} GB
Max batch    = floor({_avail:.0f} / {_kv_gb:.1f}) = {_max_b}
T_token      = ({_w_gb:.0f}+{_kv_gb:.1f}) / {_bw:.0f} GB/s = {_tok_ms:.1f} ms
```
*Source: @sec-model-serving-kv-cache*
        """))

        # Edge comparison: same model on Jetson Orin NX
        _edge_hbm = JETSON_RAM_GB
        _edge_fits = _w_gb <= _edge_hbm
        _edge_max_b = max(0, int((_edge_hbm - _w_gb) / _kv_gb)) if _edge_fits and _kv_gb > 0 else 0
        _edge_tok_ms = _w_gb / JETSON_BW_GBS * 1000 if JETSON_BW_GBS > 0 else float('inf')
        items.append(mo.callout(mo.md(
            f"**Edge comparison (Jetson Orin NX, {_edge_hbm:.0f} GB):** "
            + (f"OOM — weights alone ({_w_gb:.0f} GB) exceed {_edge_hbm:.0f} GB memory."
               if not _edge_fits else
               f"Max batch = {_edge_max_b}, token time = {_edge_tok_ms:.0f} ms "
               f"({_edge_tok_ms/_tok_ms:.0f}x slower). "
               "Edge serving requires aggressive quantization and small context windows.")
        ), kind="warn"))

        if partC_pred.value == "1":
            items.append(mo.callout(mo.md(
                "**Correct.** 70B FP16 at 128K: KV = ~320 GB per request. "
                "Weights = 140 GB. Total = 460 GB. Only 180 GB remains of 640 GB."), kind="success"))
        elif partC_pred.value == "8":
            items.append(mo.callout(mo.md(
                "**Memory, not compute, sets concurrency.** 8 GPUs give 8-way parallelism, "
                "but a single 128K request needs ~320 GB of KV cache."), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                "**KV cache at 128K is ~320 GB per request.** Only 1 fits "
                "after model weights in 640 GB HBM."), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: KV Cache Memory per Request": mo.md("""
**Formula:**
$$
M_{\\text{KV}} = 2 \\times L \\times H \\times d_h \\times S \\times b_{\\text{kv}}
$$

Total HBM budget:
$$
M_{\\text{weights}} + n \\times M_{\\text{KV}} \\leq M_{\\text{HBM}}
$$

**Variables:**
- **$L$**: number of transformer layers
- **$H$**: number of attention heads
- **$d_h$**: head dimension ($d_{\\text{model}} / H$)
- **$S$**: sequence length (context window)
- **$b_{\\text{kv}}$**: bytes per KV element (2 for FP16)
- **$n$**: concurrent batch size (requests in flight)
- Factor of 2: one K tensor + one V tensor per layer

At 128K context, Llama-2 70B KV cache = $2 \\times 80 \\times 64 \\times 128 \\times 131072 \\times 2 \\approx 320$ GB per request.
""")
        }))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D — The Cold Start Tax
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; DevOps Lead, ServerlessLLM</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We auto-scale Llama-2 70B on Kubernetes. During traffic spikes, new pods
                take forever. Users complain about first-request timeouts. What is the
                theoretical minimum cold start time?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## Cold Start = Data Movement + Initialization

```
T_cold = model_size / min(storage_BW, interconnect_BW)
       + deserialization + CUDA_init + warmup
```

For large models this is seconds to minutes, experienced by real users.
        """))

        items.append(partD_pred)
        if partD_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the cold start calculator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_model, partD_stor, partD_pcie], widths="equal"))

        _pb = partD_model.value
        _stor = partD_stor.value
        _ic = partD_pcie.value
        _m_gb = _pb * 1e9 * 2 / (1024**3)
        _s_bw = {"nvme": NVME_SEQ_GBS, "nfs": NET_FS_GBS, "ram": 50.0}[_stor]
        _s_lbl = {"nvme": "NVMe SSD", "nfs": "Network FS", "ram": "Host RAM"}[_stor]
        _p_bw = PCIE_GEN5_GBS if _ic == "gen5" else PCIE_GEN4_GBS
        _p_lbl = "PCIe Gen5" if _ic == "gen5" else "PCIe Gen4"
        _eff = min(_s_bw, _p_bw)
        _t_read = _m_gb / _eff
        _t_deser = _m_gb / 20.0
        _t_cuda = 0.8
        _t_warm = 0.5
        _t_total = _t_read + _t_deser + _t_cuda + _t_warm
        _patience = 3.0

        _fig = go.Figure()
        _cum = 0
        for _nm, _dur, _cl in [
            ("Disk Read", _t_read, COLORS['BlueLine']),
            ("Deserialization", _t_deser, COLORS['OrangeLine']),
            ("CUDA Init", _t_cuda, "#64748b"),
            ("Warmup", _t_warm, COLORS['GreenLine']),
        ]:
            _fig.add_trace(go.Bar(name=_nm, x=[_dur], y=["Cold Start"], orientation='h',
                                   marker_color=_cl, opacity=0.88, base=_cum,
                                   hovertemplate="%{fullData.name}: %{x:.2f} s<extra></extra>"))
            _cum += _dur
        _fig.add_vline(x=_patience, line_dash="dash", line_color=COLORS['RedLine'],
                       annotation_text="User patience (3s)")
        _fig.update_layout(height=240, barmode="stack",
                           xaxis=dict(title="Time (seconds)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.3, x=0),
                           margin=dict(l=80, r=20, t=50, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _t_col = COLORS['RedLine'] if _t_total > _patience else COLORS['GreenLine']
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Model (FP16)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_m_gb:.0f} GB</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Effective BW</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_eff:.1f} GB/s</div>
                <div style="font-size:0.72rem; color:#94a3b8;">min({_s_lbl}, {_p_lbl})</div></div>
            <div style="padding:16px; border:2px solid {_t_col}; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {_t_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Cold Start</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_t_col};">{_t_total:.1f} s</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{"EXCEEDS PATIENCE" if _t_total > _patience else "OK"}</div></div>
        </div>"""))

        if _t_total > _patience:
            items.append(mo.callout(mo.md(
                f"**Cold start exceeds patience.** {_t_total:.1f}s > 3s. "
                "Pre-warm instances or cache models."), kind="danger"))

        items.append(mo.md(f"""
**Cold Start &mdash; Live** (`{_pb}B FP16, {_s_lbl}, {_p_lbl}`)

```
Disk read    = {_m_gb:.0f} / {_eff:.1f} = {_t_read:.1f} s
Deserialize  = {_m_gb:.0f} / 20 = {_t_deser:.1f} s
CUDA init    = {_t_cuda:.1f} s
Warmup       = {_t_warm:.1f} s
Total        = {_t_total:.1f} s
```
*Source: @sec-model-serving-cold-start*
        """))

        _ref = 130/7.0 + 130/20.0 + 0.8 + 0.5
        if partD_pred.value == "26s":
            items.append(mo.callout(mo.md(
                f"**Correct.** NVMe bottleneck (7 GB/s): ~{130/7:.0f}s read + "
                f"~{130/20:.0f}s deserialize + 1.3s overhead = ~{_ref:.0f}s total."), kind="success"))
        elif partD_pred.value == "200ms":
            items.append(mo.callout(mo.md(
                "**200 ms is inference latency, not cold start.** 140 GB must "
                f"load first: ~{_ref:.0f}s."), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**Cold start dominated by data movement.** 140 GB at 7 GB/s "
                f"= ~{130/7:.0f}s transfer alone. Total: ~{_ref:.0f}s."), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Cold Start Latency Decomposition": mo.md("""
**Formula:**
$$
T_{\\text{cold}} = \\frac{M_{\\text{weights}}}{\\text{BW}_{\\text{storage}}} + \\frac{M_{\\text{weights}}}{\\text{BW}_{\\text{deser}}} + T_{\\text{CUDA}} + T_{\\text{warmup}}
$$

**Variables:**
- **$M_{\\text{weights}}$**: model size in bytes (e.g., 140 GB for 70B FP16)
- **$\\text{BW}_{\\text{storage}}$**: storage read bandwidth (NVMe ~7 GB/s, network FS ~1.25 GB/s)
- **$\\text{BW}_{\\text{deser}}$**: deserialization throughput (~20 GB/s)
- **$T_{\\text{CUDA}}$**: CUDA context initialization (~0.8s)
- **$T_{\\text{warmup}}$**: first-inference JIT warmup (~0.5s)

Cold start is dominated by data movement: 140 GB / 7 GB/s = 20s from NVMe alone. Pre-warming and model caching are essential for auto-scaling.
""")
        }))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════════
    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                "**1. P99 diverges from mean by 4.6x at every utilization level.**\n\n"
                "At 80% utilization, P99 is 23x service time while mean is 5x. "
                "Monitor P99, not mean."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. Batching taxes latency before inference starts.**\n\n"
                "Formation delay = (B-1)/(2*lambda). At B=32, 500 QPS: 31 ms = "
                "62% of a 50 ms SLO. Find the Pareto knee."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. KV cache, not compute, limits LLM concurrency.**\n\n"
                "At 128K context, 70B needs ~320 GB KV per request. On 8xH100, "
                "max concurrency = 1."
            ), kind="info"),
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
                        <strong>Lab 14: ML Operations</strong> -- Your model shipped. Lab 14
                        reveals why it silently loses 3 accuracy points by Friday and 7 by
                        month six -- while your dashboard stays green the entire time.
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
                        <strong>Read:</strong> the Model Serving chapter for queuing theory,
                        KV cache management, and cold start mitigation.<br/>
                        <strong>Build:</strong> TinyTorch Module 13 -- implement a basic
                        model server with batching and latency tracking.
                    </div>
                </div>
            </div>
            """),
        ])

    tabs = mo.ui.tabs({
        "Part A \u2014 Tail Latency Explosion": build_part_a(),
        "Part B \u2014 The Batching Tax":        build_part_b(),
        "Part C \u2014 The LLM Memory Wall":     build_part_c(),
        "Part D \u2014 The Cold Start Tax":       build_part_d(),
        "Synthesis":                              build_synthesis(),
    })
    tabs
    return



# ===========================================================================
# ZONE D: LEDGER HUD
# ===========================================================================

@app.cell(hide_code=True)
def _(COLORS, ledger, mo, partA_pred, partB_pred, partC_pred, partD_pred):
    _ch = ledger._state.history.get(13, {})
    _track = ledger.get_track()
    if partA_pred.value is not None and partB_pred.value is not None and partC_pred.value is not None and partD_pred.value is not None:
        ledger.save(chapter=13, design={
            "chapter": "v1_13",
            "completed": True,
            "p99_latency_prediction": partA_pred.value,
            "batching_tax_prediction": partB_pred.value,
            "kv_cache_max_batch": partC_pred.value,
            "cold_start_prediction": partD_pred.value,
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span><span class="hud-value">13 &mdash; Model Serving</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'NONE' else 'hud-none'}">{_track}</span>
        <span class="hud-label">STATUS</span><span class="hud-active">ACTIVE</span>
    </div>""")
    return


if __name__ == "__main__":
    app.run()
