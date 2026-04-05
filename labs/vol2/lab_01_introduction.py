import marimo

__generated_with = "0.19.6"
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

    # WASM bootstrap: install mlsysim from hosted wheel when running in browser
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
    from mlsysim.core.defaults import (
        GPU_MTTF_HOURS,
        INFINIBAND_NDR_BW_GBS,
        IB_NDR_LATENCY_US,
        CHINCHILLA_TOKENS_PER_PARAM,
    )
    from mlsysim.core.formulas import (
        calc_ring_allreduce_time,
        calc_mtbf_cluster,
    )
    from mlsysim.core.constants import ureg

    # ── Hardware registry ─────────────────────────────────────────────────────
    H100 = mlsysim.Hardware.Cloud.H100
    EDGE = mlsysim.Hardware.Edge.JetsonOrinNX

    H100_TFLOPS_FP16 = H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_BW_GBS = H100.memory.bandwidth.m_as("GB/s")
    H100_RAM_GB = H100.memory.capacity.m_as("GB")
    H100_TDP_W = H100.tdp.m_as("W")

    EDGE_TFLOPS = EDGE.compute.peak_flops.m_as("TFLOPs/s")
    EDGE_RAM_GB = EDGE.memory.capacity.m_as("GB")
    EDGE_TDP_W = EDGE.tdp.m_as("W")

    # ── Model registry ────────────────────────────────────────────────────────
    GPT2 = mlsysim.Models.GPT2
    GPT2_PARAMS = GPT2.parameters.m_as("dimensionless")

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return (
        COLORS,
        EDGE_TFLOPS,
        EDGE_RAM_GB,
        EDGE_TDP_W,
        H100_BW_GBS,
        H100_RAM_GB,
        H100_TDP_W,
        H100_TFLOPS_FP16,
        GPT2_PARAMS,
        LAB_CSS,
        apply_plotly_theme,
        calc_mtbf_cluster,
        calc_ring_allreduce_time,
        go,
        GPU_MTTF_HOURS,
        IB_NDR_LATENCY_US,
        INFINIBAND_NDR_BW_GBS,
        ledger,
        math,
        mo,
        np,
        CHINCHILLA_TOKENS_PER_PARAM,
        ureg,
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
                Machine Learning Systems &middot; Volume II &middot; Lab 01
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Scale Illusion
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Reliability &middot; Communication &middot; Scaling Laws &middot; Amdahl at Scale
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Everything you learned about single-machine ML breaks in surprising ways
                at fleet scale. 1,000 GPUs do not deliver 1,000x speedup. A cluster with
                99.9% per-node reliability is healthy barely a third of the time. Scale
                creates qualitative change, not just quantitative increase.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    5 Parts &middot; ~55 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Vol II Ch 1: Introduction to Scale
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">P_fleet = P_node^N</span>
                <span class="badge badge-warn">Fleet Law: T = Compute + Comm + Coord</span>
                <span class="badge badge-fail">Amdahl at Scale</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify fleet reliability collapse</strong> &mdash; calculate
                    that a 1,000-GPU cluster with 99.9% per-node uptime is healthy only 36.8%
                    of the time, and predict failure frequency at GPT-4 scale (25K GPUs).</div>
                <div style="margin-bottom: 3px;">2. <strong>Decompose distributed step time</strong> &mdash; identify
                    that communication overhead consumes 35&ndash;45% of training step time at
                    256 GPUs on InfiniBand NDR for a 175B model.</div>
                <div style="margin-bottom: 3px;">3. <strong>Apply the extended Amdahl&rsquo;s Law</strong> &mdash; find the
                    GPU count where scaling efficiency drops below 50%, and classify workloads
                    by dominant bottleneck (Computation, Communication, Coordination).</div>
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
                    Volume I complete &middot; Iron Law equation &middot;
                    Roofline model basics &middot; Memory hierarchy
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~55 min</strong><br/>
                    Parts A&ndash;E: ~10&ndash;12 min each
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
                &ldquo;If 1,000 GPUs each have 99.9% uptime and you connect them with the
                fastest network available, why don&rsquo;t you get 1,000x speedup &mdash;
                and what actually limits distributed training?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **Vol II Ch 1: The Scale Illusion** -- fleet reliability (exponential decay),
      the Fleet Law (T_step = T_compute + T_comm + T_coord), and the Conservation of Overhead.
    - **Vol II Ch 1: Scaling Laws** -- Chinchilla compute-optimal allocation (D = 20P).
    - **Vol II Ch 1: Amdahl's Law at Scale** -- communication fraction and scaling efficiency.
    """), kind="info")
    return



# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================

@app.cell(hide_code=True)
def _(
    COLORS,
    H100_TFLOPS_FP16,
    apply_plotly_theme,
    calc_ring_allreduce_time,
    go,
    GPU_MTTF_HOURS,
    IB_NDR_LATENCY_US,
    INFINIBAND_NDR_BW_GBS,
    math,
    mo,
    np,
    ureg,
    CHINCHILLA_TOKENS_PER_PARAM,
):
    # ═════════════════════════════════════════════════════════════════════════
    # WIDGET DEFINITIONS
    # ═════════════════════════════════════════════════════════════════════════

    # -- Part A widgets --
    partA_prediction = mo.ui.radio(
        options={
            "A) ~99% -- nearly always healthy": "99",
            "B) ~90% -- healthy most of the time": "90",
            "C) ~60% -- healthy more often than not": "60",
            "D) ~37% -- healthy barely a third of the time": "37",
        },
        label="Your cluster has 1,000 GPUs, each with 99.9% individual uptime. "
              "What fraction of the time is the entire cluster healthy?",
    )
    return (partA_prediction,)

@app.cell(hide_code=True)
def _(mo, partA_prediction):
    mo.stop(partA_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partA_fleet_size = mo.ui.slider(
        start=100, stop=25000, value=1000, step=100,
        label="Fleet size (GPUs)",
    )
    partA_node_rel = mo.ui.slider(
        start=0.990, stop=0.9999, value=0.999, step=0.0001,
        label="Per-node reliability",
    )

    # -- Part B widgets --
    partB_prediction = mo.ui.radio(
        options={
            "A) ~95% -- InfiniBand is fast enough": "95",
            "B) ~80% -- some communication overhead": "80",
            "C) ~55-65% -- communication is substantial": "60",
            "D) ~30% -- communication dominates": "30",
        },
        label="You scale a 175B model from 1 GPU to 256 GPUs on InfiniBand NDR. "
              "What fleet efficiency do you expect?",
    )
    return (partB_prediction,)

@app.cell(hide_code=True)
def _(mo, partB_prediction):
    mo.stop(partB_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partB_gpu_count = mo.ui.slider(
        start=1, stop=1024, value=256, step=1, label="Number of GPUs",
    )
    partB_model_size = mo.ui.dropdown(
        options={"1B": 1e9, "7B": 7e9, "70B": 70e9, "175B": 175e9},
        value="175B", label="Model parameters",
    )
    partB_network = mo.ui.dropdown(
        options={
            "IB NDR (50 GB/s)": 50.0,
            "IB HDR (25 GB/s)": 25.0,
            "100GbE (12.5 GB/s)": 12.5,
        },
        value="IB NDR (50 GB/s)", label="Interconnect",
    )

    # -- Part C widgets --
    partC_prediction = mo.ui.radio(
        options={
            "A) 10B on 200B tokens -- bigger models are always better": "10B",
            "B) 3B on 600B tokens -- balanced allocation wins": "3B",
            "C) Both achieve the same loss -- total FLOPs is what matters": "same",
            "D) Neither -- you need at least 70B parameters": "70B",
        },
        label="Fixed budget of 10^23 FLOPs. Which achieves lower loss: "
              "a 10B model on 200B tokens, or a 3B model on 600B tokens?",
    )
    return (partC_prediction,)

@app.cell(hide_code=True)
def _(mo, partC_prediction):
    mo.stop(partC_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partC_params = mo.ui.slider(start=1, stop=100, value=10, step=1, label="Model params (B)")
    partC_tokens = mo.ui.slider(start=10, stop=10000, value=200, step=10, label="Training tokens (B)")

    # -- Part D widgets --
    partD_prediction = mo.ui.radio(
        options={
            "A) ~512 GPUs -- efficiency holds a long time": "512",
            "B) ~128 GPUs -- moderate scale": "128",
            "C) ~32-64 GPUs -- surprisingly few": "48",
            "D) ~8 GPUs -- almost immediately": "8",
        },
        label="For a workload with 20% communication overhead (r = 0.20), "
              "how many GPUs before scaling efficiency drops below 50%?",
    )
    return (partD_prediction,)

@app.cell(hide_code=True)
def _(mo, partD_prediction):
    mo.stop(partD_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partD_comm_frac = mo.ui.slider(start=0.01, stop=0.50, value=0.20, step=0.01, label="Communication fraction (r)")
    partD_n_gpus = mo.ui.slider(start=1, stop=512, value=64, step=1, label="Number of GPUs")
    partD_overlap = mo.ui.slider(start=0, stop=80, value=0, step=10, label="Overlap (%)")

    # -- Part E widgets --
    partE_llm = mo.ui.radio(
        options={"Computation": "comp", "Communication": "comm", "Coordination": "coord"},
        label="GPT-4 LLM training (175B, 25K GPUs) -- dominant bottleneck?",
    )
    partE_dlrm = mo.ui.radio(
        options={"Computation": "comp", "Communication": "comm", "Coordination": "coord"},
        label="DLRM recommendation (embedding-heavy) -- dominant bottleneck?",
    )
    partE_fed = mo.ui.radio(
        options={"Computation": "comp", "Communication": "comm", "Coordination": "coord"},
        label="Federated MobileNet (edge devices) -- dominant bottleneck?",
    )
    return (partE_llm, partE_dlrm, partE_fed)

@app.cell(hide_code=True)
def _(mo, partE_llm, partE_dlrm, partE_fed):

    # ═════════════════════════════════════════════════════════════════════════
    # PART A: THE RELIABILITY COLLAPSE
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Director of Infrastructure, MegaScale AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We are deploying a 1,000-GPU cluster for frontier training.
                Each node has 99.9% reliability. We expect near-continuous operation.
                Should we budget for redundancy?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; James Chen, Director of Infrastructure &middot; MegaScale AI
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Fleet Reliability Decays Exponentially with Fleet Size

        A single GPU with 99.9% uptime sounds reliable. But fleet-wide availability
        requires **all** nodes healthy simultaneously:

        ```
        P_fleet = P_node ^ N
        ```

        At N = 1,000 and P_node = 0.999: P_fleet = 0.999^1000 = 0.368 = **36.8%**.
        At GPT-4 scale (25,000 GPUs), a hardware failure occurs roughly every **2 hours**.
        Failure is the common case, not the exception.
        """))

        items.append(partA_prediction)
        if partA_prediction.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction above to unlock the reliability simulator."
            ), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_fleet_size, partA_node_rel], justify="start", gap="2rem"))

        _N = partA_fleet_size.value
        _P = partA_node_rel.value
        _fleet_avail = _P ** _N
        _mtbf_hours = GPU_MTTF_HOURS / _N
        _failures_per_day = 24.0 / _mtbf_hours if _mtbf_hours > 0 else 0

        # Chart: fleet availability vs size
        _sizes = np.array([1, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 25000])
        _avail_curve = _P ** _sizes
        _avail_ref = 0.9999 ** _sizes

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_sizes, y=_avail_curve * 100, mode="lines+markers",
            name=f"P_node = {_P:.4f}",
            line=dict(color=COLORS["BlueLine"], width=2.5), marker=dict(size=6),
        ))
        _fig.add_trace(go.Scatter(
            x=_sizes, y=_avail_ref * 100, mode="lines",
            name="P_node = 0.9999 (reference)",
            line=dict(color=COLORS["GreenLine"], width=1.5, dash="dash"),
        ))
        _fig.add_hline(y=50, line_dash="dot", line_color=COLORS["OrangeLine"],
                       annotation_text="50% availability", annotation_position="top right",
                       annotation_font_size=10)
        _fig.add_vline(x=_N, line_dash="dash", line_color=COLORS["TextMuted"])
        _fig.add_trace(go.Scatter(
            x=[_N], y=[_fleet_avail * 100], mode="markers",
            name=f"Your cluster: {_fleet_avail*100:.1f}%",
            marker=dict(color=COLORS["RedLine"], size=14, symbol="star",
                        line=dict(color="white", width=2)),
        ))
        _fig.update_layout(
            height=360,
            xaxis=dict(title="Fleet Size (GPUs)", type="log", gridcolor="#f1f5f9"),
            yaxis=dict(title="Fleet Availability (%)", range=[0, 105], gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        # Metric cards
        _ac = COLORS["GreenLine"] if _fleet_avail > 0.8 else COLORS["OrangeLine"] if _fleet_avail > 0.5 else COLORS["RedLine"]
        _mc = COLORS["GreenLine"] if _mtbf_hours > 100 else COLORS["OrangeLine"] if _mtbf_hours > 10 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white; border-top:3px solid {_ac}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Fleet Availability</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_ac};">{_fleet_avail*100:.1f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">{_P:.4f}^{_N:,}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white; border-top:3px solid {_mc}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Cluster MTBF</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_mc};">{_mtbf_hours:.1f} hrs</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">{GPU_MTTF_HOURS:,} / {_N:,}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Failures / Day</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_failures_per_day:.1f}</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">24h / {_mtbf_hours:.1f}h</div>
            </div>
        </div>"""))

        items.append(mo.md(f"""
**Reliability -- Live Calculation** (`N = {_N:,}, P_node = {_P:.4f}`)

```
P_fleet      = P_node^N = {_P:.4f}^{_N:,} = {_fleet_avail:.6f} ({_fleet_avail*100:.1f}%)
Cluster MTBF = GPU_MTTF / N = {GPU_MTTF_HOURS:,} / {_N:,} = {_mtbf_hours:.1f} hours
Failures/day = 24 / MTBF = 24 / {_mtbf_hours:.1f} = {_failures_per_day:.1f}
```
*Source: Vol II Ch 1 -- Fleet Reliability*
        """))

        _pred = partA_prediction.value
        _ref = 0.999 ** 1000
        if _pred == "37":
            _msg = (f"**Correct.** (0.999)^1000 = {_ref:.3f} = {_ref*100:.1f}%. "
                    "The exponential makes even tiny per-node failure rates catastrophic at scale.")
            _kind = "success"
        elif _pred == "99":
            _msg = (f"**Off by 62 pp.** Actual fleet availability is {_ref*100:.1f}%, not ~99%. "
                    "Students anchor on per-node 99.9% and assume the fleet is similar.")
            _kind = "warn"
        elif _pred == "90":
            _msg = (f"**Off by 53 pp.** (0.999)^1000 = {_ref*100:.1f}%. "
                    "Each bad nine compounds catastrophically at 1,000 nodes.")
            _kind = "warn"
        else:
            _msg = (f"**Close but lower.** (0.999)^1000 = {_ref*100:.1f}%, not 60%. "
                    "The compounding is more aggressive than linear extrapolation suggests.")
            _kind = "warn"

        items.append(mo.vstack([
            mo.md(f"**You predicted:** ~{_pred}%  |  **Actual:** {_ref*100:.1f}%"),
            mo.callout(mo.md(_msg), kind=_kind),
        ]))

        items.append(mo.accordion({
            "Math Peek: Fleet Reliability Decay": mo.md("""
**Formula:**
$$
P_{\\text{fleet}} = P_{\\text{node}}^{N}
$$

**Variables:**
- **$P_{\\text{fleet}}$**: probability that the entire cluster is healthy (all nodes up simultaneously)
- **$P_{\\text{node}}$**: per-node reliability (e.g., 0.999 for 99.9% uptime)
- **$N$**: number of nodes in the fleet

**Key insight:** At $P_{\\text{node}} = 0.999$ and $N = 1{,}000$: $P_{\\text{fleet}} = 0.999^{1000} \\approx 0.368$ (36.8%). This is $1/e$ -- the exponential decay threshold.
""")
        }))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B: THE COORDINATION TAX
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; VP Training, MegaScale AI (escalated)
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We scaled from 8 to 256 GPUs expecting 32x faster training.
                Actual throughput improved only 15x. Where is the missing 17x?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Sarah Kim, VP Training &middot; MegaScale AI
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Fleet Law: T_step = T_compute + T_comm + T_coordination

        **Ring AllReduce** transfers 2(N-1)/N x M bytes of gradients each step.
        For 175B FP16 parameters that is ~700 GB per step. Even at 50 GB/s (IB NDR),
        this takes ~14 seconds -- a large fraction of compute time.

        Fleet efficiency = T_compute / T_step. When communication grows with N,
        efficiency drops far below the linear ideal.
        """))

        items.append(partB_prediction)
        if partB_prediction.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the fleet efficiency simulator."
            ), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_gpu_count, partB_model_size], justify="start", gap="2rem"))
        items.append(partB_network)

        _N = max(1, partB_gpu_count.value)
        _params = partB_model_size.value
        _bw_gbs = partB_network.value
        _grad_bytes = _params * 2  # FP16
        _grad_gb = _grad_bytes / 1e9

        # Compute time: approximate one training step
        _batch_tokens = 2048
        _flops_step = 6 * _params * _batch_tokens
        _t_compute_ms = (_flops_step / (H100_TFLOPS_FP16 * 1e12 * 0.5) / _N) * 1000

        # Communication: Ring AllReduce
        if _N > 1:
            _ar = calc_ring_allreduce_time(_grad_bytes, _N, _bw_gbs * 1e9, IB_NDR_LATENCY_US * 1e-6)
            _t_comm_ms = _ar.m_as(ureg.millisecond)
        else:
            _t_comm_ms = 0.0

        _t_coord_ms = 5.0 * math.log2(max(2, _N))
        _t_step_ms = _t_compute_ms + _t_comm_ms + _t_coord_ms
        _efficiency = (_t_compute_ms / _t_step_ms * 100) if _t_step_ms > 0 else 100.0

        # Stacked bar
        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Compute", x=["Step Time"], y=[_t_compute_ms],
                              marker_color=COLORS["BlueLine"], width=0.4))
        _fig.add_trace(go.Bar(name="Communication", x=["Step Time"], y=[_t_comm_ms],
                              marker_color=COLORS["RedLine"], width=0.4))
        _fig.add_trace(go.Bar(name="Coordination", x=["Step Time"], y=[_t_coord_ms],
                              marker_color=COLORS["OrangeLine"], width=0.4))
        _fig.update_layout(barmode="stack", height=300,
                           yaxis=dict(title="Time (ms)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _ec = COLORS["GreenLine"] if _efficiency > 80 else COLORS["OrangeLine"] if _efficiency > 50 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_ec}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Fleet Efficiency</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_ec};">{_efficiency:.1f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Compute</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_t_compute_ms:.0f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Communication</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">{_t_comm_ms:.0f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Gradient Size</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_grad_gb:.0f} GB</div>
            </div>
        </div>"""))

        items.append(mo.md(f"""
**Fleet Law -- Live Calculation** (`N = {_N}, model = {_params/1e9:.0f}B, BW = {_bw_gbs} GB/s`)

```
T_compute = {_t_compute_ms:.0f} ms    T_comm = {_t_comm_ms:.0f} ms    T_coord = {_t_coord_ms:.0f} ms
T_step    = {_t_step_ms:.0f} ms       Efficiency = {_t_compute_ms:.0f}/{_t_step_ms:.0f} = {_efficiency:.1f}%
```
*Source: Vol II Ch 1 -- The Fleet Law*
        """))

        _pred = partB_prediction.value
        if _pred == "60":
            _msg = ("**Correct.** At 256 GPUs with 175B on IB NDR, communication takes "
                    "a substantial fraction, yielding ~55-65% efficiency.")
            _kind = "success"
        else:
            _msg = ("**Actual efficiency is ~55-65%.** 350 GB of FP16 gradients across 256 GPUs "
                    "on IB NDR takes ~14 seconds per step.")
            _kind = "warn"
        items.append(mo.vstack([
            mo.md(f"**You predicted:** ~{_pred}%  |  **Simulated:** {_efficiency:.1f}%"),
            mo.callout(mo.md(_msg), kind=_kind),
        ]))

        items.append(mo.accordion({
            "Math Peek: The Fleet Law": mo.md("""
**Formula:**
$$
T_{\\text{step}} = T_{\\text{compute}} + T_{\\text{comm}} + T_{\\text{coord}}
$$

$$
\\eta_{\\text{fleet}} = \\frac{T_{\\text{compute}}}{T_{\\text{step}}} = \\frac{T_{\\text{compute}}}{T_{\\text{compute}} + T_{\\text{comm}} + T_{\\text{coord}}}
$$

**Variables:**
- **$T_{\\text{compute}}$**: time for forward + backward pass (scales as $\\text{FLOPs} / (N \\times \\text{peak\\_TFLOPS})$)
- **$T_{\\text{comm}}$**: gradient synchronization time (Ring AllReduce: $2(N-1)/N \\times M / BW$)
- **$T_{\\text{coord}}$**: barrier synchronization, straggler waits, scheduling overhead
- **$\\eta_{\\text{fleet}}$**: fleet efficiency (fraction of ideal linear speedup actually achieved)
""")
        }))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C: THE SCALING LAW BUDGET PLANNER
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_c():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Chief Scientist, MegaScale AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We have 10^23 FLOPs of compute budget. The team wants the biggest
                model possible -- 10B parameters on 200B tokens. I think we should train
                smaller but longer. Who is right?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Dr. Lena Petrova, Chief Scientist &middot; MegaScale AI
            </div>
        </div>
        """))

        items.append(mo.md(f"""
        ## Chinchilla Scaling: Compute-Optimal Allocation

        ```
        C = 6 * P * D       (total training FLOPs)
        D = {int(CHINCHILLA_TOKENS_PER_PARAM)} * P         (Chinchilla optimal ratio)
        ```

        Over-allocating to model size under-trains the model. Over-allocating to data
        under-parameterizes it. There is a unique optimum for each compute budget.
        """))

        items.append(partC_prediction)
        if partC_prediction.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the scaling law planner."
            ), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_params, partC_tokens], justify="start", gap="2rem"))

        _P = partC_params.value * 1e9
        _D = partC_tokens.value * 1e9
        _C_actual = 6 * _P * _D
        _C_budget = 1e23
        _C_ratio = _C_actual / _C_budget

        # Chinchilla optimal: C = 120 P^2 => P = sqrt(C/120)
        _P_opt = math.sqrt(_C_budget / 120)
        _D_opt = 20 * _P_opt
        _P_opt_B = _P_opt / 1e9
        _D_opt_B = _D_opt / 1e9

        # Simplified loss model: L(P,D) ~ A/P^alpha + B/D^beta + E
        _A, _alpha, _B, _beta, _E = 406.4, 0.34, 410.7, 0.28, 1.69
        _loss = _A / (partC_params.value) ** _alpha + _B / max(partC_tokens.value, 0.1) ** _beta + _E
        _loss_opt = _A / _P_opt_B ** _alpha + _B / _D_opt_B ** _beta + _E

        # IsoFLOP curves
        _p_range = np.logspace(np.log10(0.5), np.log10(200), 100)
        _fig = go.Figure()
        for _c_exp, _c_label, _cc in [
            (22, "10^22", COLORS["GreenLine"]),
            (23, "10^23 (budget)", COLORS["BlueLine"]),
            (24, "10^24", COLORS["OrangeLine"]),
        ]:
            _cval = 10.0 ** _c_exp
            _d_r = _cval / (6 * _p_range * 1e9) / 1e9
            _ls = _A / _p_range ** _alpha + _B / np.maximum(_d_r, 0.1) ** _beta + _E
            _fig.add_trace(go.Scatter(x=_p_range, y=_ls, mode="lines",
                                      name=f"C = {_c_label}", line=dict(color=_cc, width=2)))

        _fig.add_trace(go.Scatter(
            x=[partC_params.value], y=[_loss], mode="markers",
            name=f"Your choice: {partC_params.value}B",
            marker=dict(color=COLORS["RedLine"], size=14, symbol="star",
                        line=dict(color="white", width=2)),
        ))
        _fig.add_trace(go.Scatter(
            x=[_P_opt_B], y=[_loss_opt], mode="markers",
            name=f"Chinchilla optimal: {_P_opt_B:.1f}B",
            marker=dict(color=COLORS["GreenLine"], size=14, symbol="diamond",
                        line=dict(color="white", width=2)),
        ))
        _fig.update_layout(
            height=380,
            xaxis=dict(title="Model Parameters (B)", type="log", gridcolor="#f1f5f9"),
            yaxis=dict(title="Loss (lower is better)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.15, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _bc = COLORS["GreenLine"] if 0.8 <= _C_ratio <= 1.2 else COLORS["OrangeLine"] if _C_ratio <= 2.0 else COLORS["RedLine"]
        _ou = "OVER BUDGET" if _C_ratio > 1.2 else "UNDER BUDGET" if _C_ratio < 0.8 else "ON BUDGET"
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_bc}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Compute Used</div>
                <div style="font-size:1.3rem; font-weight:800; color:{_bc};">{_C_actual:.1e}</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">{_ou}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Your Loss</div>
                <div style="font-size:1.3rem; font-weight:800; color:{COLORS['BlueLine']};">{_loss:.3f}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Optimal Loss</div>
                <div style="font-size:1.3rem; font-weight:800; color:{COLORS['GreenLine']};">{_loss_opt:.3f}</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">P={_P_opt_B:.1f}B, D={_D_opt_B:.0f}B</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">D/P Ratio</div>
                <div style="font-size:1.3rem; font-weight:800; color:{COLORS['OrangeLine']};">{_D/_P:.0f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">Optimal: {int(CHINCHILLA_TOKENS_PER_PARAM)}x</div>
            </div>
        </div>"""))

        _pred = partC_prediction.value
        if _pred == "3B":
            _msg = ("**Correct.** Balanced allocation (more tokens, fewer parameters) achieves "
                    "lower loss. Chinchilla showed most pre-2022 LLMs were under-trained.")
            _kind = "success"
        else:
            _msg = ("**Balanced allocation wins.** Model size alone does not determine capability. "
                    "The compute-optimal D/P ratio of 20 means smaller models trained on proportionally "
                    "more data achieve lower loss for the same compute budget.")
            _kind = "warn"
        items.append(mo.vstack([
            mo.md(f"**You predicted:** {_pred}"),
            mo.callout(mo.md(_msg), kind=_kind),
        ]))

        items.append(mo.accordion({
            "Math Peek: Chinchilla Scaling Law": mo.md("""
**Formula:**
$$
L(N, D) = \\frac{A}{N^{\\alpha}} + \\frac{B}{D^{\\beta}} + L_0
$$

**Compute-optimal allocation (Chinchilla):**
$$
D^{*} \\approx 20 \\times N
$$

**Variables:**
- **$L$**: training loss
- **$N$**: number of model parameters
- **$D$**: number of training tokens
- **$A, B, L_0$**: fitted constants ($\\alpha \\approx 0.34$, $\\beta \\approx 0.28$)
- **$D^{*}$**: compute-optimal token count for a given $N$

**Key insight:** For a fixed compute budget $C \\approx 6ND$, allocating tokens as $D = 20N$ minimizes loss. Most pre-2022 LLMs were severely under-trained.
""")
        }))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D: THE IRON LAW OF SCALE
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; CFO, MegaScale AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Engineering wants 512 GPUs. At what point does each additional GPU
                cost more than the speedup it delivers?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Michael Torres, CFO &middot; MegaScale AI
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Extended Amdahl's Law with Communication Overhead

        ```
        T_step(N) = T_compute/N + r * T_compute * (1 - overlap) + T_coord
        Efficiency(N) = Speedup(N) / N
        ```

        Beyond a critical GPU count, adding hardware **reduces** cost-efficiency.
        The communication fraction r and overlap percentage determine the ceiling.
        """))

        items.append(partD_prediction)
        if partD_prediction.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the scaling efficiency simulator."
            ), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_comm_frac, partD_n_gpus], justify="start", gap="2rem"))
        items.append(partD_overlap)

        _r = partD_comm_frac.value
        _N = max(1, partD_n_gpus.value)
        _ov = partD_overlap.value / 100.0
        _t_1 = 1.0 + _r

        _n_range = np.arange(1, 513)
        _speedups = np.zeros(len(_n_range))
        _effs = np.zeros(len(_n_range))
        _costs = np.zeros(len(_n_range))
        for _i, _n in enumerate(_n_range):
            _tc = 1.0 / _n
            _tm = _r * (1.0 - _ov) if _n > 1 else 0.0
            _td = 0.01 * math.log2(max(2, _n)) if _n > 1 else 0.0
            _tn = _tc + _tm + _td
            _speedups[_i] = _t_1 / _tn
            _effs[_i] = _speedups[_i] / _n
            _costs[_i] = _n * _tn

        _eff_50_idx = np.argmax(_effs < 0.5)
        _eff_50_n = int(_n_range[_eff_50_idx]) if _eff_50_idx > 0 else 512
        _cost_opt_idx = int(np.argmin(_costs))
        _cost_opt_n = int(_n_range[_cost_opt_idx])

        # Speedup chart
        _fig1 = go.Figure()
        _fig1.add_trace(go.Scatter(x=_n_range, y=_n_range.astype(float), mode="lines",
                                   name="Ideal linear", line=dict(color=COLORS["GreenLine"], width=1.5, dash="dash")))
        _fig1.add_trace(go.Scatter(x=_n_range, y=_speedups, mode="lines",
                                   name=f"Actual (r={_r:.2f}, overlap={_ov*100:.0f}%)",
                                   line=dict(color=COLORS["BlueLine"], width=2.5)))
        _fig1.add_vline(x=_N, line_dash="dash", line_color=COLORS["TextMuted"])
        _fig1.add_trace(go.Scatter(x=[_N], y=[_speedups[_N-1]], mode="markers",
                                   name=f"N={_N}: {_speedups[_N-1]:.1f}x",
                                   marker=dict(color=COLORS["RedLine"], size=12, symbol="star",
                                               line=dict(color="white", width=2))))
        _fig1.update_layout(height=320,
                            xaxis=dict(title="GPUs", type="log", gridcolor="#f1f5f9"),
                            yaxis=dict(title="Speedup", type="log", gridcolor="#f1f5f9"),
                            legend=dict(orientation="h", y=1.12, x=0),
                            margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig1)

        # Cost chart
        _fig2 = go.Figure()
        _fig2.add_trace(go.Scatter(x=_n_range, y=_costs / _costs[0] * 100, mode="lines",
                                   name="Relative cost per sample",
                                   line=dict(color=COLORS["OrangeLine"], width=2.5)))
        _fig2.add_trace(go.Scatter(x=[_cost_opt_n], y=[_costs[_cost_opt_idx] / _costs[0] * 100],
                                   mode="markers", name=f"Cost optimum: N={_cost_opt_n}",
                                   marker=dict(color=COLORS["GreenLine"], size=14, symbol="diamond",
                                               line=dict(color="white", width=2))))
        _fig2.add_hline(y=100, line_dash="dot", line_color=COLORS["TextMuted"],
                        annotation_text="Single-GPU baseline")
        _fig2.update_layout(height=280,
                            xaxis=dict(title="GPUs", type="log", gridcolor="#f1f5f9"),
                            yaxis=dict(title="Relative Cost (%)", gridcolor="#f1f5f9"),
                            legend=dict(orientation="h", y=1.12, x=0),
                            margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig2)
        items.append(mo.as_html(_fig1))
        items.append(mo.as_html(_fig2))

        _eff_cur = _effs[_N - 1] * 100
        _ecc = COLORS["GreenLine"] if _eff_cur > 80 else COLORS["OrangeLine"] if _eff_cur > 50 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {_ecc}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Efficiency (N={_N})</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_ecc};">{_eff_cur:.1f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Speedup (N={_N})</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_speedups[_N-1]:.1f}x</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">50% Eff. at</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">N={_eff_50_n}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Cost Optimum</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['GreenLine']};">N={_cost_opt_n}</div>
            </div>
        </div>"""))

        _pred = partD_prediction.value
        if _pred == "48":
            _msg = (f"**Correct.** At r = 0.20, efficiency drops below 50% at ~N = {_eff_50_n} GPUs.")
            _kind = "success"
        else:
            _msg = (f"**The 50% threshold is at N = {_eff_50_n}.** At r = 0.20, "
                    "communication overhead dominates sooner than most expect.")
            _kind = "warn"
        items.append(mo.vstack([
            mo.md(f"**You predicted:** ~{_pred} GPUs  |  **Actual:** N = {_eff_50_n}"),
            mo.callout(mo.md(_msg), kind=_kind),
        ]))

        items.append(mo.accordion({
            "Math Peek: Extended Amdahl's Law for Distributed Training": mo.md("""
**Formula:**
$$
\\eta(N) = \\frac{1}{1 + r \\cdot (N - 1)}
$$

**50% efficiency threshold:**
$$
N_{50\\%} = \\frac{1}{r} + 1 - \\frac{1}{r \\cdot N} \\approx \\frac{1}{r}
$$

**With overlap:**
$$
\\eta(N) = \\frac{1}{1 + r \\cdot (1 - o) \\cdot (N - 1)}
$$

**Variables:**
- **$\\eta(N)$**: scaling efficiency at $N$ GPUs (1.0 = perfect linear scaling)
- **$r$**: communication fraction (ratio of communication time to compute time per step)
- **$N$**: number of GPUs
- **$o$**: overlap fraction (0 = no overlap, 1 = perfect overlap of backward + AllReduce)

**Key insight:** At $r = 0.20$, efficiency drops below 50% at just $N \\approx 1/0.20 = 5$ equivalent serial overheads, which in practice is ~32-64 GPUs depending on model size.
""")
        }))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART E: THE C-CUBED DIAGNOSTIC
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_e():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid #6366f1; background:#f0f4ff;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; CTO, MegaScale AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We run three workloads. Each one is slow for different reasons.
                What is the dominant bottleneck for each?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Raj Patel, CTO &middot; MegaScale AI
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## C-Cubed: Computation, Communication, Coordination

        Every distributed bottleneck falls into one of three categories.
        The **Conservation of Overhead** means reducing one C causes another to dominate.
        Classify each workload below:
        """))

        items.append(mo.vstack([partE_llm, partE_dlrm, partE_fed]))

        _answered = (partE_llm.value is not None and partE_dlrm.value is not None
                     and partE_fed.value is not None)
        if not _answered:
            items.append(mo.callout(mo.md("Classify all three to see the diagnostic."), kind="warn"))
            return mo.vstack(items)

        _wkloads = [
            {"name": "GPT-4 LLM Training (175B, 25K GPUs)", "comp": 35, "comm": 50, "coord": 15,
             "dom": "comm", "why": "Gradient sync of 175B params dominates at 25K GPUs."},
            {"name": "DLRM Recommendation (embedding)", "comp": 20, "comm": 30, "coord": 50,
             "dom": "coord", "why": "All-to-All with irregular access patterns; coordination dominates."},
            {"name": "Federated MobileNet (edge)", "comp": 15, "comm": 25, "coord": 60,
             "dom": "coord", "why": "Straggler handling and privacy overhead dominate."},
        ]

        _fig = go.Figure()
        _names = [w["name"] for w in _wkloads]
        _fig.add_trace(go.Bar(name="Computation", y=_names, x=[w["comp"] for w in _wkloads],
                              marker_color=COLORS["BlueLine"], orientation="h"))
        _fig.add_trace(go.Bar(name="Communication", y=_names, x=[w["comm"] for w in _wkloads],
                              marker_color=COLORS["RedLine"], orientation="h"))
        _fig.add_trace(go.Bar(name="Coordination", y=_names, x=[w["coord"] for w in _wkloads],
                              marker_color=COLORS["OrangeLine"], orientation="h"))
        _fig.update_layout(barmode="stack", height=250,
                           xaxis=dict(title="% of Step Time", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.15, x=0),
                           margin=dict(l=250, r=20, t=40, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _answers = [partE_llm.value, partE_dlrm.value, partE_fed.value]
        _correct = ["comm", "coord", "coord"]
        _label = {"comp": "Computation", "comm": "Communication", "coord": "Coordination"}
        _score = 0
        for _w, _a, _c in zip(_wkloads, _answers, _correct):
            _ok = _a == _c
            if _ok:
                _score += 1
            items.append(mo.callout(mo.md(
                f"**{_w['name']}:** {'Correct' if _ok else 'You selected ' + _label.get(_a, '?') + ', but it is'} "
                f"**{_label[_c]}**. {_w['why']}"
            ), kind="success" if _ok else "warn"))

        items.append(mo.callout(mo.md(
            f"**Score: {_score}/3.** The Conservation of Overhead means every workload has "
            "a dominant bottleneck -- often not the one you expect."
        ), kind="info"))

        items.append(mo.accordion({
            "Math Peek: Conservation of Overhead": mo.md("""
**Formula (The Fleet Law as a conservation equation):**
$$
T_{\\text{step}} = T_{\\text{compute}} + T_{\\text{comm}} + T_{\\text{coord}} = \\text{const}
$$

**Bottleneck classification:**
$$
\\text{Dominant} = \\arg\\max(T_{\\text{compute}},\\; T_{\\text{comm}},\\; T_{\\text{coord}})
$$

**Variables:**
- **$T_{\\text{compute}}$**: forward + backward pass time (proportional to FLOPs per GPU)
- **$T_{\\text{comm}}$**: gradient synchronization, AllReduce, All-to-All
- **$T_{\\text{coord}}$**: barrier syncs, straggler waits, scheduling, privacy overhead

**Key insight:** Reducing one term shifts the bottleneck to another. LLM training is communication-bound; DLRM is coordination-bound (irregular All-to-All); federated learning is coordination-bound (straggler + privacy).
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
                "**1. Fleet reliability decays exponentially, not linearly.** "
                "A 1,000-GPU cluster with 99.9% per-node reliability is healthy only 36.8% of the time. "
                "At 25,000 GPUs, a failure occurs every ~2 hours. "
                "Fault tolerance is a first-order design requirement."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. Communication overhead caps distributed speedup far below the linear ideal.** "
                "Ring AllReduce of 175B FP16 parameters across 256 GPUs on IB NDR takes ~14 seconds. "
                "Fleet efficiency drops to 55-65%. The Fleet Law (T = Compute + Comm + Coord) "
                "diagnoses any distributed bottleneck."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. Every optimization trades one bottleneck for another.** "
                "Chinchilla scaling requires balanced model/data allocation. "
                "Extended Amdahl's Law reveals diminishing GPU returns. "
                "The C-Cubed diagnostic classifies bottlenecks. "
                "Scale creates qualitative change, not just quantitative increase."
            ), kind="info"),
            mo.md("""
## Connections

**Textbook:** Vol II Ch 1 -- Fleet Law, fleet reliability, Chinchilla scaling, extended Amdahl's Law.

**Next Lab:** V2-02 deepens the compute infrastructure story: the memory wall, the roofline
at fleet scale, the bandwidth staircase, and the hidden cost of ownership.
            """),
        ])

    # ═════════════════════════════════════════════════════════════════════════
    # COMPOSE TABS
    # ═════════════════════════════════════════════════════════════════════════

    tabs = mo.ui.tabs({
        "Part A -- The Reliability Collapse": build_part_a(),
        "Part B -- The Coordination Tax": build_part_b(),
        "Part C -- The Scaling Law Budget": build_part_c(),
        "Part D -- The Iron Law of Scale": build_part_d(),
        "Part E -- The C-Cubed Diagnostic": build_part_e(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return



# ===========================================================================
# ZONE D: LEDGER HUD
# ===========================================================================

@app.cell(hide_code=True)
def _(COLORS, ledger, mo, partA_prediction, partB_prediction, partC_prediction, partD_prediction, partE_llm, partE_dlrm, partE_fed):
    _track = ledger._state.track or "not set"
    if partA_prediction.value is not None:
        ledger.save(chapter=1, design={
            "chapter": "v2_01",
            "completed": True,
            "fleet_reliability_prediction": partA_prediction.value,
            "fleet_efficiency_prediction": partB_prediction.value,
            "scaling_law_allocation": partC_prediction.value,
            "amdahl_gpu_threshold": partD_prediction.value,
            "bottleneck_llm": partE_llm.value,
            "bottleneck_dlrm": partE_dlrm.value,
            "bottleneck_federated": partE_fed.value,
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">V2-01 &middot; The Scale Illusion</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">VOL&nbsp;II&nbsp;CH&nbsp;1</span>
        <span class="hud-value">Introduction to Scale</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
