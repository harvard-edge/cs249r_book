import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly"], keep_going=False)
        await micropip.install(
            "../../wheels/mlsysim-0.1.0-py3-none-any.whl", keep_going=False
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    import mlsysim
    from mlsysim.core.defaults import GPU_MTTF_HOURS
    from mlsysim.core.formulas import calc_young_daly_interval, calc_mtbf_cluster
    from mlsysim.core.constants import (
        ureg,
        H100_FLOPS_FP16_TENSOR,
        A100_FLOPS_FP16_TENSOR,
        B200_FLOPS_FP16_TENSOR,
        V100_FLOPS_FP16_TENSOR,
        NVME_SEQUENTIAL_BW,
    )

    # Scalar extraction
    H100_TFLOPS = H100_FLOPS_FP16_TENSOR.m_as("TFLOPs/s")
    A100_TFLOPS = A100_FLOPS_FP16_TENSOR.m_as("TFLOPs/s")
    B200_TFLOPS = B200_FLOPS_FP16_TENSOR.m_as("TFLOPs/s")
    V100_TFLOPS = V100_FLOPS_FP16_TENSOR.m_as("TFLOPs/s")
    NVME_GBS = NVME_SEQUENTIAL_BW.m_as("GB/s")

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return (
        COLORS, LAB_CSS, apply_plotly_theme, go, math, mo, np, ledger, ureg,
        H100_TFLOPS, A100_TFLOPS, B200_TFLOPS, V100_TFLOPS, NVME_GBS,
        GPU_MTTF_HOURS,
        calc_young_daly_interval, calc_mtbf_cluster,
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
                Machine Learning Systems &middot; Volume II &middot; Lab 04
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Data Pipeline Wall
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Storage Chasm &middot; Pipeline Equation &middot; Shard Contention &middot; Checkpoints
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Storage -- the least glamorous infrastructure component -- can silently
                determine whether your cluster is productive or an expensive space heater.
                The compute-storage gap has widened 60x in seven years and is getting worse.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    5 Parts &middot; ~58 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Vol II Ch 4: Data and Storage at Scale
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">60x widening gap</span>
                <span class="badge badge-warn">Birthday problem in shards</span>
                <span class="badge badge-fail">Young-Daly checkpoint interval</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the storage-compute chasm</strong> &mdash; show that
                    accelerator throughput has grown 236x (V100 to B200) while NVMe bandwidth
                    grew only 4x, and explain why faster GPUs make the storage problem worse.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose data stalls</strong> &mdash; calculate that prefetching
                    cannot eliminate stalls when I/O time exceeds compute time, and apply
                    the pipelining equation T_step = max(T_compute, T_IO).</div>
                <div style="margin-bottom: 3px;">3. <strong>Optimize checkpoint frequency</strong> &mdash; apply the Young-Daly
                    formula to find the optimal checkpoint interval that minimizes total waste
                    (checkpoint overhead + expected rework).</div>
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
                    V2-01 (reliability) &middot; V2-02 (memory hierarchy)
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~58 min</strong><br/>
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
                &ldquo;Your cluster is storage-bottlenecked at 30% GPU utilization. You upgrade
                to GPUs with 2x the TFLOPS. Does utilization improve, stay the same, or get
                worse &mdash; and how often should you save checkpoints?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **Vol II Ch 4: The Storage-Compute Chasm** -- 60x widening gap, pipeline equation.
    - **Vol II Ch 4: Data Stalls and Prefetching** -- when pipelining cannot help.
    - **Vol II Ch 4: Checkpoint Economics** -- Young-Daly optimal interval.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(
    COLORS, apply_plotly_theme, go, math, mo, np, ureg,
    H100_TFLOPS, A100_TFLOPS, B200_TFLOPS, V100_TFLOPS, NVME_GBS,
    GPU_MTTF_HOURS,
    calc_young_daly_interval, calc_mtbf_cluster,
):
    # ═════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ═════════════════════════════════════════════════════════════════════════
    pA_pred = mo.ui.radio(
        options={
            "A) ~60% -- faster GPUs process data faster": "60",
            "B) ~30% -- no change, storage is the bottleneck": "30",
            "C) ~15% -- utilization drops because GPUs are faster but storage is not": "15",
            "D) ~5% -- catastrophic collapse": "5",
        },
        label="Storage-bottlenecked at 30% utilization. Upgrade GPUs to 2x TFLOPS. What happens?",
    )
    return (pA_pred,)

@app.cell(hide_code=True)
def _(mo, pA_pred):
    pA_gen = mo.ui.dropdown(
        options={"V100 (2017)": "v100", "A100 (2020)": "a100", "H100 (2022)": "h100", "B200 (2024)": "b200"},
        value="H100 (2022)", label="GPU generation",
    )

    pB_pred = mo.ui.radio(
        options={
            "A) ~80% -- utilization stays": "80",
            "B) ~60% -- moderate drop": "60",
            "C) ~40% -- storage BW split across twice as many GPUs": "40",
            "D) ~20% -- catastrophic starving": "20",
        },
        label="128 GPUs at 80% utilization. Double to 256 without upgrading storage. What happens?",
    )
    return (pB_pred,)

@app.cell(hide_code=True)
def _(mo, pB_pred):
    pB_gpus = mo.ui.slider(start=8, stop=1024, value=128, step=8, label="GPU count")
    pB_target = mo.ui.slider(start=50, stop=95, value=80, step=5, label="Target utilization (%)")

    pC_pred = mo.ui.radio(
        options={
            "A) ~10% -- 1,000 shards is plenty for 256 workers": "10",
            "B) ~50% -- borderline": "50",
            "C) ~100% -- collisions are essentially guaranteed": "100",
            "D) ~75% -- high but not certain": "75",
        },
        label="256 GPUs, 1,000 shards, random selection. Collision probability?",
    )
    return (pC_pred,)

@app.cell(hide_code=True)
def _(mo, pC_pred):
    pC_workers = mo.ui.slider(start=8, stop=256, value=256, step=8, label="GPU workers")
    pC_shards = mo.ui.slider(start=100, stop=10000, value=1000, step=100, label="Dataset shards")

    pD_pred = mo.ui.radio(
        options={
            "A) Yes -- 4 batches of prefetch hide the 300 ms I/O": "yes",
            "B) No -- stall drops from 60% to 33% but never reaches zero": "no_partial",
            "C) Partially -- stall drops to ~10%": "partial",
            "D) No effect -- prefetching only helps random access": "no_effect",
        },
        label="200 ms compute, 300 ms I/O. Add prefetch depth 4. Does the stall disappear?",
    )
    return (pD_pred,)

@app.cell(hide_code=True)
def _(mo, pD_pred):
    pD_compute = mo.ui.slider(start=100, stop=500, value=200, step=10, label="Compute time (ms)")
    pD_io = mo.ui.slider(start=50, stop=1000, value=300, step=10, label="I/O time (ms)")
    pD_prefetch = mo.ui.slider(start=0, stop=8, value=0, step=1, label="Prefetch depth")

    pE_pred = mo.ui.radio(
        options={
            "A) Every 5 minutes -- minimize lost work": "5",
            "B) Every ~27 minutes -- Young-Daly sweet spot": "27",
            "C) Every hour -- minimize I/O overhead": "60",
            "D) Every 2 hours -- checkpoints are expensive": "120",
        },
        label="1,000-GPU cluster, MTBF = 5 hours, checkpoint write = 2 min. Optimal interval?",
    )
    return (pE_pred,)

@app.cell(hide_code=True)
def _(mo, pE_pred):
    pE_mtbf = mo.ui.slider(start=1, stop=24, value=5, step=1, label="Cluster MTBF (hours)")
    pE_write = mo.ui.slider(start=30, stop=300, value=120, step=10, label="Checkpoint write time (s)")
    pE_interval = mo.ui.slider(start=1, stop=120, value=30, step=1, label="Checkpoint interval (min)")

    # ═════════════════════════════════════════════════════════════════════════
    # PART A: THE STORAGE-COMPUTE CHASM
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; VP Infrastructure, DataScale Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We upgraded from A100s to H100s. Our storage-bottlenecked cluster was
                at 30% GPU utilization. Surely the faster GPUs will help?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md(f"""
        ## The Storage-Compute Chasm Widens Every Generation

        | Generation | Compute (TFLOPS) | Storage BW (GB/s) | Ratio |
        |---|---|---|---|
        | V100 (2017) | {V100_TFLOPS:.0f} | ~{NVME_GBS*0.5:.1f} | {V100_TFLOPS/(NVME_GBS*0.5):.0f}x |
        | A100 (2020) | {A100_TFLOPS:.0f} | ~{NVME_GBS*0.7:.1f} | {A100_TFLOPS/(NVME_GBS*0.7):.0f}x |
        | H100 (2022) | {H100_TFLOPS:.0f} | ~{NVME_GBS:.1f} | {H100_TFLOPS/NVME_GBS:.0f}x |
        | B200 (2024) | {B200_TFLOPS:.0f} | ~{NVME_GBS*1.4:.1f} | {B200_TFLOPS/(NVME_GBS*1.4):.0f}x |

        Faster GPUs make the storage problem **worse**, not better.
        """))

        items.append(pA_pred)
        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(pA_gen)

        _hw = {
            "v100": ("V100", V100_TFLOPS, NVME_GBS * 0.5),
            "a100": ("A100", A100_TFLOPS, NVME_GBS * 0.7),
            "h100": ("H100", H100_TFLOPS, NVME_GBS),
            "b200": ("B200", B200_TFLOPS, NVME_GBS * 1.4),
        }
        _name, _tflops, _storage_bw = _hw[pA_gen.value]

        # If storage-bottlenecked, utilization = storage_delivery / gpu_demand
        # Doubling GPU speed halves utilization when storage stays fixed
        _base_util = 30.0
        _ratio_vs_a100 = _tflops / A100_TFLOPS
        _new_util = _base_util / _ratio_vs_a100  # storage-limited

        # Generational chart
        _gens = ["V100", "A100", "H100", "B200"]
        _gen_tflops = [V100_TFLOPS, A100_TFLOPS, H100_TFLOPS, B200_TFLOPS]
        _gen_storage = [NVME_GBS * 0.5, NVME_GBS * 0.7, NVME_GBS, NVME_GBS * 1.4]
        _gen_ratios = [t / s for t, s in zip(_gen_tflops, _gen_storage)]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Compute (TFLOPS)", x=_gens, y=_gen_tflops,
                              marker_color=COLORS["BlueLine"], yaxis="y"))
        _fig.add_trace(go.Scatter(name="Compute/Storage Ratio", x=_gens, y=_gen_ratios,
                                  mode="lines+markers", line=dict(color=COLORS["RedLine"], width=2.5),
                                  marker=dict(size=10), yaxis="y2"))
        _fig.update_layout(
            height=340,
            yaxis=dict(title="TFLOPS", gridcolor="#f1f5f9"),
            yaxis2=dict(title="Compute/Storage Ratio", overlaying="y", side="right", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=60, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">New Utilization</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['RedLine']};">{_new_util:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">Was: 30% (A100)</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Compute/Storage</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_tflops/_storage_bw:.0f}x</div>
            </div>
        </div>"""))

        _pred = pA_pred.value
        if _pred == "15":
            _msg = ("**Correct.** When storage is the bottleneck, faster GPUs reduce utilization. "
                    f"The {_name} processes data 2x faster but storage delivers at the same rate.")
            _kind = "success"
        else:
            _msg = (f"**Utilization drops to ~{_new_util:.0f}%.** Faster GPUs wait longer for data. "
                    "The storage-compute chasm widens every hardware generation.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B: THE DATA PIPELINE EQUATION
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Training Lead, DataScale Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We doubled our GPU count from 128 to 256 without upgrading storage.
                Why did utilization drop?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Pipeline Equation

        ```
        BW_required = N_GPUs x U_target x S_batch / T_iteration
        ```

        Storage bandwidth is a **shared resource**. Doubling GPUs without upgrading
        storage halves the per-GPU bandwidth, proportionally dropping utilization.
        """))

        items.append(pB_pred)
        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pB_gpus, pB_target], justify="start", gap="2rem"))

        _n = pB_gpus.value
        _target = pB_target.value / 100
        _base_n = 128
        _base_bw = NVME_GBS * 20  # 20 NVMe drives
        _bw_per_gpu = _base_bw / _n
        _base_bw_per_gpu = _base_bw / _base_n
        _util = min(_target, _bw_per_gpu / _base_bw_per_gpu * _target) * 100

        # S-curve: utilization vs GPU count
        _n_range = np.arange(8, 1025, 8)
        _util_curve = np.minimum(_target * 100, _base_bw / _n_range / _base_bw_per_gpu * _target * 100)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_n_range, y=_util_curve, mode="lines",
                                  name="GPU Utilization", line=dict(color=COLORS["BlueLine"], width=2.5)))
        _fig.add_hline(y=50, line_dash="dash", line_color=COLORS["OrangeLine"],
                       annotation_text="50% stall threshold")
        _fig.add_vline(x=_n, line_dash="dash", line_color=COLORS["TextMuted"])
        _fig.add_trace(go.Scatter(x=[_n], y=[_util], mode="markers",
                                  name=f"N={_n}: {_util:.0f}%",
                                  marker=dict(color=COLORS["RedLine"], size=12, symbol="star",
                                              line=dict(color="white", width=2))))
        _fig.update_layout(height=320,
                           xaxis=dict(title="GPU Count", gridcolor="#f1f5f9"),
                           yaxis=dict(title="GPU Utilization (%)", range=[0, 100], gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _uc = COLORS["GreenLine"] if _util > 70 else COLORS["OrangeLine"] if _util > 40 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_uc}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Utilization</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_uc};">{_util:.0f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">BW per GPU</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_bw_per_gpu:.1f} GB/s</div>
            </div>
        </div>"""))

        if _util < 50:
            items.append(mo.callout(mo.md(
                f"**Storage-starved.** At {_n} GPUs, utilization is {_util:.0f}%. "
                "More than half the compute budget is wasted waiting for data."
            ), kind="danger"))

        _pred = pB_pred.value
        if _pred == "40":
            _msg = "**Correct.** Storage BW is split across 2x GPUs, halving per-GPU BW and utilization."
            _kind = "success"
        else:
            _msg = "**Utilization drops proportionally.** Storage bandwidth is shared; doubling GPUs halves per-GPU BW."
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C: THE SHARD CONTENTION BIRTHDAY PROBLEM
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_c():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Data Engineer, DataScale Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We have 1,000 dataset shards and 256 workers. Each worker randomly
                selects a shard. Should collisions be rare?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Birthday Problem at Datacenter Scale

        Even with many shards, random access by many workers creates surprisingly
        high collision probability. The birthday problem strikes at n = sqrt(N):

        ```
        P(collision) = 1 - e^(-n^2 / 2N)
        ```

        With 256 workers and 1,000 shards: exponent = -256^2/2000 = -32.8.
        P = 1 - e^(-32.8) = **near certainty**.
        """))

        items.append(pC_pred)
        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pC_workers, pC_shards], justify="start", gap="2rem"))

        _n = pC_workers.value
        _N = pC_shards.value
        _exponent = -(_n ** 2) / (2 * _N)
        _p_collision = 1 - math.exp(max(_exponent, -500))
        _birthday_threshold = math.sqrt(_N)

        # Collision probability vs workers
        _w_range = np.arange(1, 257)
        _p_curve = 1 - np.exp(np.maximum(-_w_range ** 2 / (2 * _N), -500))

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_w_range, y=_p_curve * 100, mode="lines",
                                  name=f"Shards = {_N:,}",
                                  line=dict(color=COLORS["BlueLine"], width=2.5)))
        _fig.add_hline(y=50, line_dash="dash", line_color=COLORS["OrangeLine"],
                       annotation_text="50% collision probability")
        _fig.add_vline(x=_birthday_threshold, line_dash="dot", line_color=COLORS["GreenLine"],
                       annotation_text=f"sqrt({_N}) = {_birthday_threshold:.0f}",
                       annotation_font_size=10)
        _fig.add_trace(go.Scatter(x=[_n], y=[_p_collision * 100], mode="markers",
                                  name=f"Your config: {_p_collision*100:.1f}%",
                                  marker=dict(color=COLORS["RedLine"], size=14, symbol="star",
                                              line=dict(color="white", width=2))))
        _fig.update_layout(height=340,
                           xaxis=dict(title="Workers", gridcolor="#f1f5f9"),
                           yaxis=dict(title="Collision Probability (%)", range=[0, 105], gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _pc = COLORS["GreenLine"] if _p_collision < 0.5 else COLORS["OrangeLine"] if _p_collision < 0.9 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_pc}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Collision Prob.</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_pc};">{_p_collision*100:.1f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Birthday Threshold</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_birthday_threshold:.0f} workers</div>
            </div>
        </div>"""))

        _pred = pC_pred.value
        if _pred == "100":
            _msg = f"**Correct.** With {_n} workers and {_N:,} shards, collisions are near-certain ({_p_collision*100:.1f}%)."
            _kind = "success"
        else:
            _msg = (f"**Collisions are {_p_collision*100:.1f}% probable.** The birthday problem "
                    f"strikes at sqrt({_N}) = {_birthday_threshold:.0f} workers, far below {_n}.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D: THE DATA STALL DIAGNOSTIC
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Performance Engineer, DataScale Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We added prefetch depth 4. The PM says the stall should be eliminated.
                But GPU utilization barely improved. Why?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Pipelining Cannot Fix I/O > Compute

        ```
        Without pipeline: T_step = T_IO + T_compute
        With pipeline:    T_step = max(T_compute, T_IO)
        Stall %         = (T_step - T_compute) / T_step
        ```

        When I/O exceeds compute, **no amount of prefetching eliminates the stall**.
        The only fix is faster storage or smaller data.
        """))

        items.append(pD_pred)
        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pD_compute, pD_io], justify="start", gap="2rem"))
        items.append(pD_prefetch)

        _tc = pD_compute.value
        _tio = pD_io.value
        _depth = pD_prefetch.value

        _sequential_ms = _tc + _tio
        _sequential_stall = _tio / _sequential_ms * 100

        _pipelined_ms = max(_tc, _tio)
        _pipelined_stall = max(0, (_pipelined_ms - _tc)) / _pipelined_ms * 100 if _pipelined_ms > 0 else 0

        # With partial pipeline (depth effect: transitions from sequential to pipelined)
        _alpha = min(1.0, _depth / 4.0)  # 0 = sequential, 1 = fully pipelined
        _actual_ms = (1 - _alpha) * _sequential_ms + _alpha * _pipelined_ms
        _actual_stall = max(0, (_actual_ms - _tc)) / _actual_ms * 100 if _actual_ms > 0 else 0

        # Timeline bars
        _fig = go.Figure()
        _configs = [
            ("No prefetch", _sequential_ms, _sequential_stall),
            (f"Prefetch={_depth}", _actual_ms, _actual_stall),
            ("Perfect pipeline", _pipelined_ms, _pipelined_stall),
        ]
        for _name, _time, _stall in _configs:
            _col = COLORS["GreenLine"] if _stall < 10 else COLORS["OrangeLine"] if _stall < 30 else COLORS["RedLine"]
            _fig.add_trace(go.Bar(name=f"{_name} ({_stall:.0f}% stall)",
                                  x=[_name], y=[_time],
                                  marker_color=_col, width=0.4))

        _fig.add_hline(y=_tc, line_dash="dash", line_color=COLORS["BlueLine"],
                       annotation_text=f"Compute = {_tc} ms")
        _fig.update_layout(height=300,
                           yaxis=dict(title="Step Time (ms)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.15, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _sc = COLORS["GreenLine"] if _actual_stall < 10 else COLORS["OrangeLine"] if _actual_stall < 30 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {_sc}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Current Stall</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_sc};">{_actual_stall:.0f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Best Possible</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_pipelined_stall:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">Perfect pipeline</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">I/O Bound?</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{'Yes' if _tio > _tc else 'No'}</div>
            </div>
        </div>"""))

        _pred = pD_pred.value
        if _pred == "no_partial":
            _msg = (f"**Correct.** With I/O ({_tio} ms) > compute ({_tc} ms), "
                    f"perfect pipelining gives T_step = max({_tc}, {_tio}) = {_pipelined_ms} ms. "
                    f"Stall = ({_pipelined_ms} - {_tc})/{_pipelined_ms} = {_pipelined_stall:.0f}%. "
                    "The only fix is faster storage or smaller data.")
            _kind = "success"
        else:
            _msg = (f"**Stall persists at {_pipelined_stall:.0f}% even with perfect pipelining.** "
                    f"When I/O > compute, T_step = max(compute, I/O) = I/O. "
                    "Prefetching converts sequential to pipelined but cannot shrink max().")
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART E: CHECKPOINT ECONOMICS
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_e():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid #6366f1; background:#f0f4ff;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Training Manager, DataScale Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;How often should we checkpoint? Too frequent wastes I/O bandwidth.
                Too infrequent wastes compute on rework after failures. What is optimal?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Young-Daly Optimal Checkpoint Interval

        ```
        tau_opt = sqrt(2 * delta * MTBF)
        ```

        The U-shaped waste curve has three terms:
        - Checkpoint overhead (decreases with interval)
        - Expected rework (increases with interval)
        - Total waste = overhead + rework (U-curve with minimum at tau_opt)
        """))

        items.append(pE_pred)
        if pE_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pE_mtbf, pE_write], justify="start", gap="2rem"))
        items.append(pE_interval)

        _mtbf_s = pE_mtbf.value * 3600
        _delta_s = pE_write.value
        _interval_s = pE_interval.value * 60

        # Young-Daly optimal
        _yd = calc_young_daly_interval(_delta_s, _mtbf_s)
        _tau_opt_s = _yd.m_as(ureg.second)
        _tau_opt_min = _tau_opt_s / 60

        # Waste components at current interval
        _overhead_pct = (_delta_s / _interval_s) * 100 if _interval_s > 0 else 100
        _rework_pct = (_interval_s / (2 * _mtbf_s)) * 100
        _total_waste = _overhead_pct + _rework_pct

        # Waste at optimal
        _opt_overhead = (_delta_s / _tau_opt_s) * 100
        _opt_rework = (_tau_opt_s / (2 * _mtbf_s)) * 100
        _opt_waste = _opt_overhead + _opt_rework

        # U-curve
        _intervals = np.linspace(60, 7200, 200)
        _oh_curve = (_delta_s / _intervals) * 100
        _rw_curve = (_intervals / (2 * _mtbf_s)) * 100
        _total_curve = _oh_curve + _rw_curve

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_intervals / 60, y=_oh_curve, mode="lines",
                                  name="Checkpoint overhead",
                                  line=dict(color=COLORS["OrangeLine"], width=1.5, dash="dash")))
        _fig.add_trace(go.Scatter(x=_intervals / 60, y=_rw_curve, mode="lines",
                                  name="Expected rework",
                                  line=dict(color=COLORS["RedLine"], width=1.5, dash="dash")))
        _fig.add_trace(go.Scatter(x=_intervals / 60, y=_total_curve, mode="lines",
                                  name="Total waste", line=dict(color=COLORS["BlueLine"], width=2.5)))
        _fig.add_vline(x=_tau_opt_min, line_dash="dot", line_color=COLORS["GreenLine"],
                       annotation_text=f"Optimal: {_tau_opt_min:.0f} min",
                       annotation_font_size=10)
        _fig.add_trace(go.Scatter(x=[pE_interval.value], y=[_total_waste], mode="markers",
                                  name=f"Your interval: {pE_interval.value} min",
                                  marker=dict(color=COLORS["RedLine"], size=14, symbol="star",
                                              line=dict(color="white", width=2))))
        _fig.update_layout(height=360,
                           xaxis=dict(title="Checkpoint Interval (min)", gridcolor="#f1f5f9"),
                           yaxis=dict(title="Waste (%)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.15, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        # Failure: checkpoint write exceeds interval
        _write_exceeds = _delta_s >= _interval_s
        if _write_exceeds:
            items.append(mo.callout(mo.md(
                f"**Checkpoint write ({_delta_s}s) exceeds interval ({_interval_s}s).** "
                "The system spends all its time writing checkpoints and never trains."
            ), kind="danger"))

        _wc = COLORS["GreenLine"] if abs(pE_interval.value - _tau_opt_min) < 5 else COLORS["OrangeLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Optimal Interval</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['GreenLine']};">{_tau_opt_min:.0f} min</div>
                <div style="font-size:0.72rem; color:#94a3b8;">Young-Daly</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {_wc}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Your Waste</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_wc};">{_total_waste:.1f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Optimal Waste</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_opt_waste:.1f}%</div>
            </div>
        </div>"""))

        _pred = pE_pred.value
        if _pred == "27":
            _msg = f"**Correct.** Young-Daly optimal = sqrt(2 x {_delta_s} x {_mtbf_s}) = {_tau_opt_min:.0f} min."
            _kind = "success"
        else:
            _msg = (f"**Optimal interval is {_tau_opt_min:.0f} minutes.** "
                    "Too frequent wastes I/O on checkpoint writes. "
                    "Too infrequent wastes compute on rework after failures.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════════

    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                "**1. Faster GPUs make storage bottlenecks worse.** "
                "Accelerator throughput grew 236x while NVMe bandwidth grew only 4x. "
                "Upgrading GPUs on a storage-starved cluster reduces utilization, not improves it."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. Prefetching cannot fix I/O > compute.** "
                "With perfect pipelining, T_step = max(T_compute, T_IO). "
                "When I/O exceeds compute, the stall is irreducible without faster storage. "
                "The birthday problem makes shard contention near-certain at scale."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. Checkpoint frequency is a U-shaped optimization.** "
                "Young-Daly gives tau_opt = sqrt(2 * delta * MTBF). "
                "Too-frequent checkpointing saturates storage bandwidth. "
                "Too-infrequent checkpointing wastes millions in rework after failures."
            ), kind="info"),
            mo.md("""
## Connections

**Textbook:** Vol II Ch 4 -- storage-compute chasm, data pipeline equation,
birthday problem in shard contention, Young-Daly checkpointing.

**Next Lab:** V2-05 explores the parallelism puzzle: data parallelism hits a communication
wall, ZeRO trades communication for memory, pipeline parallelism creates bubbles,
and 3D parallelism maps strategies to the bandwidth hierarchy.
            """),
        ])

    # ═════════════════════════════════════════════════════════════════════════
    # COMPOSE TABS
    # ═════════════════════════════════════════════════════════════════════════

    tabs = mo.ui.tabs({
        "Part A -- Storage-Compute Chasm": build_part_a(),
        "Part B -- Pipeline Equation": build_part_b(),
        "Part C -- Shard Contention": build_part_c(),
        "Part D -- Data Stall Diagnostic": build_part_d(),
        "Part E -- Checkpoint Economics": build_part_e(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(COLORS, ledger, mo):
    _track = ledger._state.track or "not set"
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">V2-04 &middot; The Data Pipeline Wall</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">VOL&nbsp;II&nbsp;CH&nbsp;4</span>
        <span class="hud-value">Data and Storage at Scale</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
