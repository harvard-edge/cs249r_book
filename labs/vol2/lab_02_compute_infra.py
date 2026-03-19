import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
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

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    import mlsysim
    from mlsysim.core.defaults import (
        GPU_MTTF_HOURS,
        INFINIBAND_NDR_BW_GBS,
        PUE_BEST_AIR,
        DEFAULT_KWH_PRICE,
        ANNUAL_MAINTENANCE_RATIO,
        MFU_INFERENCE_BATCH1,
    )
    from mlsysim.core.constants import (
        ureg,
        H100_FLOPS_FP16_TENSOR,
        H100_MEM_BW,
        H100_MEM_CAPACITY,
        H100_TDP,
        A100_FLOPS_FP16_TENSOR,
        A100_MEM_BW,
        A100_MEM_CAPACITY,
        B200_FLOPS_FP16_TENSOR,
        B200_MEM_BW,
        B200_MEM_CAPACITY,
        V100_FLOPS_FP16_TENSOR,
        V100_MEM_BW,
        NVLINK_H100_BW,
        PCIE_GEN5_BW,
        NVME_SEQUENTIAL_BW,
    )

    # Extract scalar values for chart use
    H100_TFLOPS = H100_FLOPS_FP16_TENSOR.m_as("TFLOPs/s")
    H100_BW_GBS = H100_MEM_BW.m_as("GB/s")
    H100_RAM_GB = H100_MEM_CAPACITY.m_as("GB")
    H100_TDP_W = H100_TDP.m_as("W")
    H100_RIDGE = H100_TFLOPS * 1000 / H100_BW_GBS  # FLOPs/Byte

    A100_TFLOPS = A100_FLOPS_FP16_TENSOR.m_as("TFLOPs/s")
    A100_BW_GBS = A100_MEM_BW.m_as("GB/s")
    B200_TFLOPS = B200_FLOPS_FP16_TENSOR.m_as("TFLOPs/s")
    B200_BW_GBS = B200_MEM_BW.m_as("GB/s")
    V100_TFLOPS = V100_FLOPS_FP16_TENSOR.m_as("TFLOPs/s")
    V100_BW_GBS = V100_MEM_BW.m_as("GB/s")

    NVLINK_GBS = NVLINK_H100_BW.m_as("GB/s")
    PCIE_GBS = PCIE_GEN5_BW.m_as("GB/s")
    IB_NDR_GBS = INFINIBAND_NDR_BW_GBS
    NVME_GBS = NVME_SEQUENTIAL_BW.m_as("GB/s")

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return (
        COLORS, LAB_CSS, apply_plotly_theme, go, math, mo, np, ledger, ureg,
        H100_TFLOPS, H100_BW_GBS, H100_RAM_GB, H100_TDP_W, H100_RIDGE,
        A100_TFLOPS, A100_BW_GBS, B200_TFLOPS, B200_BW_GBS,
        V100_TFLOPS, V100_BW_GBS,
        NVLINK_GBS, PCIE_GBS, IB_NDR_GBS, NVME_GBS,
        PUE_BEST_AIR, DEFAULT_KWH_PRICE, ANNUAL_MAINTENANCE_RATIO,
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
                Machine Learning Systems &middot; Volume II &middot; Lab 02
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Compute Infrastructure Wall
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Memory Wall &middot; Roofline &middot; Bandwidth Staircase &middot; TCO
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Even the fastest accelerator spends most of its time waiting for data.
                The roofline reveals why, the bandwidth staircase dictates parallelism
                strategy placement, and TCO shows the real cost of scale.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    5 Parts &middot; ~57 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Vol II Ch 2: Compute Infrastructure
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">MFU &lt; 1% at batch=1</span>
                <span class="badge badge-warn">NVLink/IB = 18x cliff</span>
                <span class="badge badge-fail">TCO: GPU price is only the start</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Diagnose the memory wall</strong> &mdash; show that at batch=1
                    an H100 is ~99% idle during LLM token generation because HBM bandwidth,
                    not compute, is the binding constraint.</div>
                <div style="margin-bottom: 3px;">2. <strong>Map the bandwidth staircase</strong> &mdash; quantify the 18x
                    NVLink-to-InfiniBand cliff and explain why tensor parallelism cannot cross
                    the node boundary.</div>
                <div style="margin-bottom: 3px;">3. <strong>Calculate Total Cost of Ownership</strong> &mdash; show that
                    electricity reaches ~30% of 3-year TCO for a 1,000-GPU cluster and that
                    utilization rate matters more than hardware generation.</div>
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
                    V2-01 (Scale Illusion) &middot; Iron Law equation &middot;
                    Roofline model from Vol I
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~57 min</strong><br/>
                    Parts A&ndash;E: ~9&ndash;12 min each
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
                &ldquo;If the H100 delivers 989 TFLOPS, why does single-token LLM generation
                use less than 1% of that compute &mdash; and what does it actually cost to
                run a 1,000-GPU cluster for three years?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **Vol II Ch 2: The Memory Wall** -- why token generation is bandwidth-bound.
    - **Vol II Ch 2: The Roofline at Fleet Scale** -- ridge point and arithmetic intensity.
    - **Vol II Ch 2: The Bandwidth Hierarchy** -- NVLink, PCIe, InfiniBand staircase.
    - **Vol II Ch 2: Total Cost of Ownership** -- CapEx vs OpEx breakdown.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(
    COLORS, apply_plotly_theme, go, math, mo, np,
    H100_TFLOPS, H100_BW_GBS, H100_RAM_GB, H100_TDP_W, H100_RIDGE,
    A100_TFLOPS, A100_BW_GBS, B200_TFLOPS, B200_BW_GBS,
    V100_TFLOPS, V100_BW_GBS,
    NVLINK_GBS, PCIE_GBS, IB_NDR_GBS, NVME_GBS,
    PUE_BEST_AIR, DEFAULT_KWH_PRICE, ANNUAL_MAINTENANCE_RATIO,
):
    # ═════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ═════════════════════════════════════════════════════════════════════════
    pA_pred = mo.ui.radio(
        options={
            "A) ~10% idle -- GPUs are mostly computing": "10",
            "B) ~50% idle -- memory and compute are balanced": "50",
            "C) ~80% idle -- memory is a drag": "80",
            "D) ~99% idle -- almost entirely waiting for data": "99",
        },
        label="During single-token generation of a 70B model on H100, "
              "what fraction of the time are the arithmetic units idle?",
    )
    return (pA_pred,)

@app.cell(hide_code=True)
def _(mo, pA_pred):
    pA_model = mo.ui.dropdown(
        options={"7B": 7, "70B": 70, "175B": 175},
        value="70B", label="Model size (B params)",
    )
    pA_batch = mo.ui.slider(start=1, stop=128, value=1, step=1, label="Batch size")

    pB_pred = mo.ui.radio(
        options={
            "A) All are compute-bound -- H100 is always compute-bound for large models": "all",
            "B) LLM training and prefill -- they have large batch sizes": "train_prefill",
            "C) Only prefill and training at large batch -- decode is always below": "prefill_only",
            "D) None -- all LLM workloads are memory-bound": "none",
        },
        label="Which fleet-scale workloads fall above the H100 ridge point?",
    )
    return (pB_pred,)

@app.cell(hide_code=True)
def _(mo, pB_pred):
    pB_hw = mo.ui.dropdown(
        options={"V100": "v100", "A100": "a100", "H100": "h100", "B200": "b200"},
        value="H100", label="Accelerator",
    )

    pC_pred = mo.ui.radio(
        options={
            "A) ~2x slower -- InfiniBand is fast": "2",
            "B) ~5x slower -- significant but manageable": "5",
            "C) ~18x slower -- an order of magnitude gap": "18",
            "D) ~100x slower -- completely different regime": "100",
        },
        label="How much slower is a 10 GB AllReduce over IB NDR vs NVLink 4.0?",
    )
    return (pC_pred,)

@app.cell(hide_code=True)
def _(mo, pC_pred):
    pC_size = mo.ui.slider(start=1, stop=10000, value=10000, step=100, label="Transfer size (MB)")

    pD_pred = mo.ui.radio(
        options={
            "A) Yes -- 640 GB is plenty for 175B": "yes",
            "B) Barely -- fits with ~50 GB headroom": "barely",
            "C) No -- static memory alone exceeds 640 GB": "no_static",
            "D) No, but ZeRO-1 fixes it": "zero1",
        },
        label="Can a single 8-GPU DGX H100 node (640 GB HBM) train a 175B "
              "model with Adam in FP16 without ZeRO?",
    )
    return (pD_pred,)

@app.cell(hide_code=True)
def _(mo, pD_pred):
    pD_model_b = mo.ui.slider(start=1, stop=175, value=175, step=1, label="Model params (B)")
    pD_gpus = mo.ui.slider(start=1, stop=8, value=8, step=1, label="GPUs per node")
    pD_zero = mo.ui.dropdown(
        options={"No ZeRO (Stage 0)": 0, "ZeRO-1": 1, "ZeRO-2": 2, "ZeRO-3": 3},
        value="No ZeRO (Stage 0)", label="ZeRO Stage",
    )

    pE_pred = mo.ui.radio(
        options={
            "A) GPUs by far -- $30M vs ~$3M electricity": "gpus",
            "B) GPUs cost more, but electricity is ~30% of TCO": "close",
            "C) Electricity is more expensive": "elec",
            "D) Roughly equal": "equal",
        },
        label="For a 1,000-GPU H100 cluster over 3 years: GPUs or electricity costs more?",
    )
    return (pE_pred,)

@app.cell(hide_code=True)
def _(mo, pE_pred):
    pE_n_gpus = mo.ui.slider(start=100, stop=5000, value=1000, step=100, label="GPU count")
    pE_util = mo.ui.slider(start=30, stop=90, value=70, step=5, label="Utilization (%)")
    pE_pue = mo.ui.slider(start=1.06, stop=1.60, value=1.12, step=0.02, label="PUE")

    # ═════════════════════════════════════════════════════════════════════════
    # PART A: THE MEMORY WALL
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; VP Inference, CloudScale AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We bought H100s for LLM serving but our token generation is slow.
                The H100 has 989 TFLOPS. Surely we are compute-limited?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md(f"""
        ## The Memory Wall: Token Generation is Bandwidth-Bound

        At batch=1, every model weight must be loaded from HBM for each token.
        The H100 delivers {H100_BW_GBS:,.0f} GB/s HBM bandwidth vs {H100_TFLOPS:.0f} TFLOPS.
        The **ridge point** = {H100_RIDGE:.0f} FLOPs/Byte. LLM decode at batch=1 has
        arithmetic intensity ~1 FLOP/Byte -- far below the ridge point.

        The GPU is not compute-limited. It is waiting for data.
        """))

        items.append(pA_pred)
        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pA_model, pA_batch], justify="start", gap="2rem"))

        _params_b = pA_model.value
        _batch = pA_batch.value
        _params = _params_b * 1e9
        _weight_bytes = _params * 2  # FP16
        _weight_gb = _weight_bytes / 1e9

        # Memory time: load all weights once per token (decode)
        _t_mem_ms = (_weight_gb / H100_BW_GBS) * 1000
        # Compute time: 2 FLOPs per param per token, times batch
        _flops = 2 * _params * _batch
        _t_comp_ms = (_flops / (H100_TFLOPS * 1e12)) * 1000
        _t_total = _t_mem_ms + _t_comp_ms
        _idle_pct = (_t_mem_ms / _t_total) * 100 if _t_total > 0 else 0
        _mfu = (_t_comp_ms / _t_total) * 100 if _t_total > 0 else 0
        _ai = (2 * _batch)  # FLOPs per byte = 2*batch / 2 bytes = batch

        # Waterfall chart
        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Memory (HBM load)", x=["Latency"], y=[_t_mem_ms],
                              marker_color=COLORS["RedLine"], width=0.4))
        _fig.add_trace(go.Bar(name="Compute (arithmetic)", x=["Latency"], y=[_t_comp_ms],
                              marker_color=COLORS["BlueLine"], width=0.4))
        _fig.update_layout(barmode="stack", height=280,
                           yaxis=dict(title="Time (ms)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _mc = COLORS["GreenLine"] if _mfu > 50 else COLORS["OrangeLine"] if _mfu > 10 else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Memory Time</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_t_mem_ms:.2f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Compute Time</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_t_comp_ms:.4f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {_mc}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">MFU</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_mc};">{_mfu:.1f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Arith. Intensity</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_ai:.0f} FLOPs/B</div>
                <div style="font-size:0.72rem; color:#94a3b8;">Ridge: {H100_RIDGE:.0f}</div>
            </div>
        </div>"""))

        items.append(mo.md(f"""
**Memory Wall -- Live Calculation** (`{_params_b}B, batch={_batch}`)

```
Weight load  = {_weight_gb:.0f} GB / {H100_BW_GBS:,.0f} GB/s = {_t_mem_ms:.2f} ms
Compute      = 2 * {_params:.0e} * {_batch} / {H100_TFLOPS:.0f} TFLOPS = {_t_comp_ms:.4f} ms
GPU idle     = {_t_mem_ms:.2f} / {_t_total:.2f} = {_idle_pct:.1f}%
AI           = {_ai:.0f} FLOPs/Byte  (ridge = {H100_RIDGE:.0f})
```
*Source: Vol II Ch 2 -- The Memory Wall*
        """))

        _pred = pA_pred.value
        if _pred == "99":
            _msg = (f"**Correct.** At batch=1 the GPU is {_idle_pct:.1f}% idle. "
                    f"The {_params_b}B model loads {_weight_gb:.0f} GB from HBM "
                    f"in {_t_mem_ms:.2f} ms; compute takes just {_t_comp_ms:.4f} ms.")
            _kind = "success"
        else:
            _msg = (f"**The GPU is {_idle_pct:.1f}% idle at batch=1.** "
                    "LLM decode has arithmetic intensity ~1, far below the "
                    f"ridge point of {H100_RIDGE:.0f}. The H100 is a $30,000 space heater "
                    "at batch=1. Increase batch size to improve MFU.")
            _kind = "warn"
        items.append(mo.vstack([
            mo.md(f"**You predicted:** ~{_pred}% idle  |  **Actual:** {_idle_pct:.1f}% idle"),
            mo.callout(mo.md(_msg), kind=_kind),
        ]))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B: THE ROOFLINE DIAGNOSTIC
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Performance Engineer, CloudScale AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We have four LLM workloads. The team says they are all compute-bound
                because the models are huge. Can you verify with the roofline?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md(f"""
        ## The Roofline Model at Fleet Scale

        The ridge point separates memory-bound (left) from compute-bound (right):

        ```
        Ridge Point = Peak FLOPS / Memory Bandwidth
        H100:  {H100_TFLOPS:.0f} TFLOPS / {H100_BW_GBS:,.0f} GB/s = {H100_RIDGE:.0f} FLOPs/Byte
        ```

        LLM decode at batch=1 has AI ~ 1 FLOP/Byte. LLM training at large batch
        can reach AI > 100. The model size is not what determines the regime --
        the arithmetic intensity is.
        """))

        items.append(pB_pred)
        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(pB_hw)

        _hw_map = {
            "v100": ("V100", V100_TFLOPS, V100_BW_GBS),
            "a100": ("A100", A100_TFLOPS, A100_BW_GBS),
            "h100": ("H100", H100_TFLOPS, H100_BW_GBS),
            "b200": ("B200", B200_TFLOPS, B200_BW_GBS),
        }
        _hw_name, _hw_tflops, _hw_bw = _hw_map[pB_hw.value]
        _ridge = _hw_tflops * 1000 / _hw_bw

        # Roofline curve
        _ai_range = np.logspace(-1, 4, 200)
        _perf = np.minimum(_ai_range * _hw_bw / 1000, _hw_tflops)

        # Workloads: (name, AI FLOPs/Byte, color)
        _workloads = [
            ("LLM decode B=1", 1, COLORS["RedLine"]),
            ("LLM decode B=32", 32, COLORS["OrangeLine"]),
            ("175B training", 150, COLORS["BlueLine"]),
            ("175B prefill", 200, COLORS["GreenLine"]),
            ("DLRM embedding", 0.5, "#6366f1"),
        ]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_ai_range, y=_perf, mode="lines",
                                  name=f"{_hw_name} Roofline",
                                  line=dict(color=COLORS["BlueLine"], width=2.5)))
        _fig.add_vline(x=_ridge, line_dash="dash", line_color=COLORS["OrangeLine"],
                       annotation_text=f"Ridge: {_ridge:.0f}", annotation_position="top right",
                       annotation_font_size=10)

        for _wname, _wai, _wcolor in _workloads:
            _wperf = min(_wai * _hw_bw / 1000, _hw_tflops)
            _fig.add_trace(go.Scatter(
                x=[_wai], y=[_wperf], mode="markers+text",
                name=_wname, text=[_wname], textposition="top center",
                marker=dict(color=_wcolor, size=12, line=dict(color="white", width=2)),
                textfont=dict(size=9),
            ))

        _fig.update_layout(height=400,
                           xaxis=dict(title="Arithmetic Intensity (FLOPs/Byte)", type="log", gridcolor="#f1f5f9"),
                           yaxis=dict(title="Throughput (TFLOPS)", type="log", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.15, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _pred = pB_pred.value
        if _pred == "prefill_only":
            _msg = ("**Correct.** Only prefill and training at large batch sizes exceed "
                    "the ridge point. LLM decode is always deeply memory-bound regardless "
                    "of model size -- AI depends on batch, not parameters.")
            _kind = "success"
        else:
            _msg = (f"**Only prefill and large-batch training exceed the ridge point ({_ridge:.0f}).** "
                    "LLM decode at batch=1 has AI ~ 1, far below the ridge. "
                    "'Large model' does not mean 'compute-bound.'")
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C: THE BANDWIDTH STAIRCASE
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_c():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Network Architect, CloudScale AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;The team wants to run tensor parallelism across nodes.
                NVLink inside a node is fast. How much slower is InfiniBand between nodes?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md(f"""
        ## The Bandwidth Staircase

        Data transfer speed drops by orders of magnitude at each boundary:

        | Interconnect | Bandwidth |
        |---|---|
        | HBM3 | {H100_BW_GBS:,.0f} GB/s |
        | NVLink 4.0 | {NVLINK_GBS:.0f} GB/s |
        | PCIe Gen5 | {PCIE_GBS:.0f} GB/s |
        | IB NDR | {IB_NDR_GBS:.0f} GB/s |

        NVLink-to-IB = {NVLINK_GBS/IB_NDR_GBS:.0f}x cliff. This is why tensor parallelism
        stays within a node and data parallelism crosses nodes.
        """))

        items.append(pC_pred)
        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(pC_size)

        _size_mb = pC_size.value
        _size_gb = _size_mb / 1000
        _tiers = [
            ("HBM3", H100_BW_GBS, COLORS["BlueLine"]),
            ("NVLink 4.0", NVLINK_GBS, COLORS["GreenLine"]),
            ("PCIe Gen5", PCIE_GBS, COLORS["OrangeLine"]),
            ("IB NDR", IB_NDR_GBS, COLORS["RedLine"]),
        ]
        _times_ms = [(_size_gb / bw) * 1000 for _, bw, _ in _tiers]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=[n for n, _, _ in _tiers], y=_times_ms,
            marker_color=[c for _, _, c in _tiers], width=0.5,
        ))
        _fig.update_layout(height=320, yaxis=dict(title="Transfer Time (ms)", type="log", gridcolor="#f1f5f9"),
                           margin=dict(l=50, r=20, t=30, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _nvlink_ms = (_size_gb / NVLINK_GBS) * 1000
        _ib_ms = (_size_gb / IB_NDR_GBS) * 1000
        _ratio = _ib_ms / _nvlink_ms if _nvlink_ms > 0 else 0

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">NVLink Time</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_nvlink_ms:.1f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">IB NDR Time</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_ib_ms:.1f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">IB/NVLink Ratio</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_ratio:.0f}x</div>
            </div>
        </div>"""))

        _pred = pC_pred.value
        if _pred == "18":
            _msg = f"**Correct.** NVLink = {NVLINK_GBS:.0f} GB/s, IB NDR = {IB_NDR_GBS:.0f} GB/s. Ratio = {NVLINK_GBS/IB_NDR_GBS:.0f}x."
            _kind = "success"
        else:
            _msg = f"**The ratio is {NVLINK_GBS/IB_NDR_GBS:.0f}x.** This cliff dictates that TP stays within a node."
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D: THE NODE MEMORY BUDGET
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Training Lead, CloudScale AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our DGX H100 has 640 GB total HBM. Can it train a 175B model?
                175B x 2 bytes = 350 GB for weights -- should fit, right?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Training Memory Budget

        Training memory is far more than just weights:

        ```
        Total = Weights + Gradients + Optimizer States + Activations
        ```

        Adam optimizer stores FP32 master weights + momentum + variance = 12 bytes/param.
        With FP16 weights (2B/param) + FP16 grads (2B/param): **16 bytes per parameter**.
        For 175B params: 175B x 16 = **2,800 GB static memory** (no ZeRO).
        """))

        items.append(pD_pred)
        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pD_model_b, pD_gpus], justify="start", gap="2rem"))
        items.append(pD_zero)

        _P = pD_model_b.value * 1e9
        _N = pD_gpus.value
        _zero = pD_zero.value
        _hbm_per_gpu = H100_RAM_GB

        # Memory components (bytes per parameter)
        _w_bytes = 2  # FP16 weights
        _g_bytes = 2  # FP16 gradients
        _o_bytes = 12  # Adam: 4 (FP32 master) + 4 (momentum) + 4 (variance)

        # ZeRO sharding
        if _zero == 0:
            _w_gb = _P * _w_bytes / 1e9
            _g_gb = _P * _g_bytes / 1e9
            _o_gb = _P * _o_bytes / 1e9
        elif _zero == 1:
            _w_gb = _P * _w_bytes / 1e9
            _g_gb = _P * _g_bytes / 1e9
            _o_gb = _P * _o_bytes / 1e9 / _N
        elif _zero == 2:
            _w_gb = _P * _w_bytes / 1e9
            _g_gb = _P * _g_bytes / 1e9 / _N
            _o_gb = _P * _o_bytes / 1e9 / _N
        else:  # ZeRO-3
            _w_gb = _P * _w_bytes / 1e9 / _N
            _g_gb = _P * _g_bytes / 1e9 / _N
            _o_gb = _P * _o_bytes / 1e9 / _N

        # Activations (not sharded by ZeRO): approximate
        _act_gb = min(50, pD_model_b.value * 0.3)  # rough estimate
        _total_gb = _w_gb + _g_gb + _o_gb + _act_gb
        _oom = _total_gb > _hbm_per_gpu

        # Stacked bar
        _fig = go.Figure()
        _components = [
            ("Weights", _w_gb, COLORS["BlueLine"]),
            ("Gradients", _g_gb, COLORS["OrangeLine"]),
            ("Optimizer", _o_gb, COLORS["RedLine"]),
            ("Activations", _act_gb, "#6366f1"),
        ]
        for _name, _val, _col in _components:
            _fig.add_trace(go.Bar(name=_name, x=["Per-GPU Memory"], y=[_val],
                                  marker_color=_col, width=0.4))
        _fig.add_hline(y=_hbm_per_gpu, line_dash="dash", line_color=COLORS["RedLine"],
                       annotation_text=f"HBM Capacity: {_hbm_per_gpu:.0f} GB",
                       annotation_position="top right")
        _fig.update_layout(barmode="stack", height=300,
                           yaxis=dict(title="Memory (GB)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.15, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        if _oom:
            items.append(mo.callout(mo.md(
                f"**OOM -- Training infeasible.** Required: {_total_gb:.0f} GB | "
                f"Available: {_hbm_per_gpu:.0f} GB per GPU. "
                f"Overflow: {_total_gb - _hbm_per_gpu:.0f} GB."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"**Fits.** Required: {_total_gb:.0f} GB | Available: {_hbm_per_gpu:.0f} GB. "
                f"Headroom: {_hbm_per_gpu - _total_gb:.0f} GB."
            ), kind="success"))

        _pred = pD_pred.value
        if _pred == "no_static":
            _msg = ("**Correct.** Static memory (weights + gradients + optimizer) = "
                    "175B x 16 bytes = 2,800 GB. That is 4.4x the total 640 GB HBM "
                    "of an 8-GPU DGX H100 node. ZeRO-3 is required at minimum.")
            _kind = "success"
        else:
            _msg = ("**Static memory alone is 2,800 GB -- 4.4x more than 640 GB.** "
                    "Students compute only weight memory (350 GB) and forget gradients "
                    "(350 GB) and Adam states (2,100 GB at FP32).")
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART E: TCO -- THE HIDDEN COST OF SCALE
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_e():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid #6366f1; background:#f0f4ff;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; CFO, CloudScale AI
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;The GPUs cost $30M. How much will electricity add over three years?
                I need TCO for the board.&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Total Cost of Ownership Goes Far Beyond GPUs

        TCO = CapEx (GPUs + networking + storage) + OpEx (power + cooling + staff).
        A 1,000-GPU H100 cluster at 700W each = 700 kW. With PUE 1.12, facility
        power = 784 kW. At $0.12/kWh, annual electricity alone is ~$824K.
        """))

        items.append(pE_pred)
        if pE_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pE_n_gpus, pE_util], justify="start", gap="2rem"))
        items.append(pE_pue)

        _n = pE_n_gpus.value
        _util = pE_util.value / 100
        _pue = pE_pue.value
        _years = 3
        _gpu_cost = 30_000
        _tdp = H100_TDP_W

        # CapEx
        _capex_gpus = _n * _gpu_cost
        _capex_network = _n * 2000  # ~$2K networking per GPU
        _capex_storage = _n * 500
        _capex_total = _capex_gpus + _capex_network + _capex_storage

        # OpEx (annual)
        _power_kw = _n * _tdp / 1000 * _util * _pue
        _annual_elec = _power_kw * 8760 * DEFAULT_KWH_PRICE
        _annual_maint = _capex_total * ANNUAL_MAINTENANCE_RATIO
        _annual_staff = _n * 200  # rough staff allocation per GPU
        _opex_total = (_annual_elec + _annual_maint + _annual_staff) * _years

        _tco = _capex_total + _opex_total
        _elec_pct = (_annual_elec * _years) / _tco * 100 if _tco > 0 else 0

        # TCO breakdown chart
        _fig = go.Figure()
        _labels = ["GPUs", "Networking", "Storage", "Electricity\n(3yr)", "Maintenance\n(3yr)", "Staff\n(3yr)"]
        _vals = [_capex_gpus, _capex_network, _capex_storage,
                 _annual_elec * _years, _annual_maint * _years, _annual_staff * _years]
        _cols = [COLORS["BlueLine"], COLORS["BlueLine"], COLORS["BlueLine"],
                 COLORS["OrangeLine"], COLORS["OrangeLine"], COLORS["OrangeLine"]]
        _fig.add_trace(go.Bar(x=_labels, y=[v / 1e6 for v in _vals],
                              marker_color=_cols, width=0.5))
        _fig.update_layout(height=320, yaxis=dict(title="Cost ($M)", gridcolor="#f1f5f9"),
                           margin=dict(l=50, r=20, t=30, b=60))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">CapEx</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">${_capex_total/1e6:.1f}M</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">OpEx (3yr)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">${_opex_total/1e6:.1f}M</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total TCO</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">${_tco/1e6:.1f}M</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Electricity %</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_elec_pct:.0f}%</div>
            </div>
        </div>"""))

        _pred = pE_pred.value
        if _pred == "close":
            _msg = (f"**Correct.** Over 3 years, electricity is {_elec_pct:.0f}% of TCO. "
                    "GPUs cost more in absolute terms, but operational costs are a much "
                    "larger fraction than most engineers expect.")
            _kind = "success"
        else:
            _msg = (f"**Electricity is {_elec_pct:.0f}% of 3-year TCO.** "
                    "GPUs dominate CapEx but operational costs compound over years. "
                    "Utilization rate matters enormously -- a cluster at 30% utilization "
                    "wastes 70% of its power budget on idle GPUs.")
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
                f"**1. The Memory Wall dominates LLM inference.** At batch=1, an H100 is ~99% idle. "
                f"AI ~ 1 FLOPs/Byte vs ridge point {H100_RIDGE:.0f}. The GPU is waiting for HBM, not computing."
            ), kind="info"),
            mo.callout(mo.md(
                f"**2. The Bandwidth Staircase dictates parallelism placement.** "
                f"NVLink ({NVLINK_GBS:.0f} GB/s) to IB NDR ({IB_NDR_GBS:.0f} GB/s) = "
                f"{NVLINK_GBS/IB_NDR_GBS:.0f}x cliff. TP stays within a node; DP crosses nodes."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. TCO goes far beyond GPU purchase price.** Electricity reaches ~25-35% "
                "of 3-year TCO. Utilization rate matters more than hardware generation -- "
                "a well-utilized A100 cluster outperforms an idle H100 cluster on cost-per-FLOP."
            ), kind="info"),
            mo.md("""
## Connections

**Textbook:** Vol II Ch 2 -- memory wall, roofline at fleet scale, bandwidth hierarchy, TCO.

**Next Lab:** V2-03 explores communication at scale: the alpha-beta model, Ring vs Tree
AllReduce, hierarchical communication, and gradient compression.
            """),
        ])

    # ═════════════════════════════════════════════════════════════════════════
    # COMPOSE TABS
    # ═════════════════════════════════════════════════════════════════════════

    tabs = mo.ui.tabs({
        "Part A -- The Memory Wall": build_part_a(),
        "Part B -- The Roofline Diagnostic": build_part_b(),
        "Part C -- The Bandwidth Staircase": build_part_c(),
        "Part D -- Node Memory Budget": build_part_d(),
        "Part E -- TCO: Hidden Cost of Scale": build_part_e(),
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
        <span class="hud-value">V2-02 &middot; The Compute Infrastructure Wall</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">VOL&nbsp;II&nbsp;CH&nbsp;2</span>
        <span class="hud-value">Compute Infrastructure</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
