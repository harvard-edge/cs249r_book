import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-08: THE INFERENCE ECONOMY
#
# Chapter: Inference at Scale (@sec-inference-at-scale)
# Core Invariant: Serving cost eclipses training cost within weeks. The KV
#                 cache memory wall (not compute) is the binding constraint on
#                 concurrent serving. Continuous batching transforms a stop-and-
#                 go assembly line into a flowing pipeline. The inference fleet
#                 design challenge requires jointly optimizing quantization,
#                 batching, and replica count under a latency SLO.
#
# Tabbed Structure (35-40 minutes):
#   Part A — The Serving Cost Inversion (12-15 min)
#             Serving cost > training cost within ~6 weeks at 100 QPS.
#             Batching trades latency for throughput along a hockey stick.
#
#   Part B — The KV Cache Wall + Fleet Design Challenge (20-25 min)
#             At 128K context, even 8xH100 can serve only 1 request for 70B.
#             Design a fleet: INT4 + continuous batching + right replica count
#             achieves the SLO at 40% lower cost than naive FP16 + static.
#
# Hardware Constants:
#   H100_RAM_GB        = 80     GB HBM3e per GPU
#   H100_TFLOPS_FP16   = 989    TFLOPS dense tensor core
#   H100_COST_HR       = 3.0    $/GPU-hour cloud pricing
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP ─────────────────────────────────────────────────────────────
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
    from mlsysim.labs.components import DecisionLog
    from mlsysim import Hardware

    # ── Hardware registry ─────────────────────────────────────────────────
    H100 = Hardware.Cloud.H100
    T4 = Hardware.Cloud.T4
    EDGE = Hardware.Edge.JetsonOrinNX
    H100_RAM_GB = H100.memory.capacity.m_as("GB")
    H100_COST_HR = 3.0
    T4_COST_HR = 0.35
    EDGE_RAM_GB = EDGE.memory.capacity.m_as("GB")
    TRAINING_COST_2M = 2_000_000  # $2M training cost for 70B model

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return COLORS, LAB_CSS, apply_plotly_theme, go, ledger, math, mo, np, H100_RAM_GB, H100_COST_HR, T4_COST_HR, TRAINING_COST_2M, DecisionLog, Hardware, H100, T4, EDGE, EDGE_RAM_GB


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, LAB_CSS, mo):
    mo.Html(f"""
    {LAB_CSS}
    <div style="background: linear-gradient(135deg, {COLORS['Surface0']} 0%, {COLORS['Surface1']} 100%);
                border-radius: 16px; padding: 32px 40px; margin-bottom: 8px;
                border: 1px solid #2d3748;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                            text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;">
                    Vol 2 &middot; Lab 08 &middot; Inference at Scale
                </div>
                <div style="font-size: 2.0rem; font-weight: 800; color: #f1f5f9; line-height: 1.15; margin-bottom: 10px;">
                    The Inference Economy
                </div>
                <div style="font-size: 0.95rem; color: #94a3b8; max-width: 600px; line-height: 1.6;">
                    You trained a 70B model for $2M. Congratulations -- that was the cheap
                    part. Serving cost eclipses training cost within weeks. The KV cache
                    memory wall, not compute, caps your concurrency. Continuous batching
                    transforms the economics. Design a fleet that serves 10K QPS at
                    200ms P99 for minimum cost.
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;">
                <span class="badge badge-info">Serving Cost Inversion</span>
                <span class="badge badge-info">KV Cache Memory Wall</span>
                <span class="badge badge-info">Continuous Batching</span>
                <span class="badge badge-info">Fleet Design Challenge</span>
                <span class="badge badge-warn">45&ndash;55 minutes &middot; 4 Parts + Synthesis</span>
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 2: BRIEFING ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the serving cost inversion:</strong> calculate when cumulative serving cost exceeds the $2M training cost for a 70B LLM at 100 QPS.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose the KV cache memory wall:</strong> compute max concurrent requests at 128K context on 8xH100 for a 70B model and explain why memory, not compute, is the binding constraint.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design an inference fleet:</strong> jointly optimize quantization, batch size, and replica count to meet a 10K QPS / 200ms P99 SLO at minimum cost.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    KV cache formula from the Inference at Scale chapter &middot;
                    Queuing theory (Kingman's formula) from the Fleet Orchestration chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
                    Part A: ~12 min &middot; Part B: ~25 min
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                "You spent $2M training a 70B model. After how many weeks does serving cost
                exceed training cost, and why does the KV cache &mdash; not compute &mdash;
                determine whether you can afford to serve it at all?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete before this lab:

    - **The Inference at Scale chapter** -- Serving economics, KV cache scaling, continuous batching
    - The KV Cache section -- `KV = 2 * L * H * S * B * P` bytes formula
    - The Continuous Batching section -- Static vs iteration-level scheduling
    - The Queuing Theory section from the Fleet Orchestration chapter -- Kingman's formula
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 4: Part A widgets ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    # -- Part A prediction --
    partA_prediction = mo.ui.radio(
        options={
            "A) 6 months -- training dominates for a long time": "A",
            "B) 3 months -- serving catches up gradually": "B",
            "C) ~6 weeks -- serving cost grows fast": "C",
            "D) Never -- training is always more expensive": "D",
        },
        label="You spent $2M training a 70B LLM. At 100 QPS and $0.01/query, when does cumulative serving cost exceed training cost?",
    )
    return (partA_prediction,)


# ─── CELL 5: Part A controls + Part A reflection + Part B prediction ─────────
@app.cell(hide_code=True)
def _(mo, partA_prediction):
    mo.stop(partA_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    a1_qps = mo.ui.slider(start=10, stop=1000, value=100, step=10, label="Queries per second (QPS)")
    a1_cost_query = mo.ui.slider(start=0.001, stop=0.05, value=0.01, step=0.001, label="Cost per query ($)")
    a1_weeks = mo.ui.slider(start=1, stop=52, value=26, step=1, label="Deployment duration (weeks)")
    a1_optimization = mo.ui.slider(start=0, stop=50, value=0, step=5, label="Inference optimization (%)")

    # -- Part A reflection --
    partA_reflection = mo.ui.radio(
        options={
            "A) Reduce QPS by throttling users": "A",
            "B) Optimize inference efficiency (quantization, batching, caching)": "B",
            "C) Train a smaller model": "C",
            "D) Increase the price per query": "D",
        },
        label="What is the highest-ROI lever for controlling total cost of ownership?",
    )

    # -- Part B prediction --
    partB_prediction = mo.ui.radio(
        options={
            "A) 16-32 concurrent requests -- GPUs have plenty of memory": "A",
            "B) 4-8 concurrent requests -- some memory overhead": "B",
            "C) 2-3 concurrent requests -- memory is tighter than expected": "C",
            "D) Just 1 -- the KV cache consumes all available memory": "D",
        },
        label="70B model (FP16) on 8xH100 (640 GB). Weights = 140 GB. At 128K context, how many concurrent requests?",
    )
    return (a1_qps, a1_cost_query, a1_weeks, a1_optimization, partA_reflection, partB_prediction)


# ─── CELL 6: Part B controls + Part B reflection ────────────────────────────
@app.cell(hide_code=True)
def _(mo, partB_prediction):
    mo.stop(partB_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    a2_model_size = mo.ui.dropdown(
        options={"7B (32L, 4096H)": (7, 32, 4096), "70B (80L, 8192H)": (70, 80, 8192), "175B (96L, 12288H)": (175, 96, 12288)},
        value="70B (80L, 8192H)",
        label="Model",
    )
    a2_precision = mo.ui.dropdown(
        options={"FP16 (2 bytes)": 2, "INT8 (1 byte)": 1, "INT4 (0.5 bytes)": 0.5},
        value="FP16 (2 bytes)",
        label="Weight precision",
    )
    a2_context_len = mo.ui.slider(start=2048, stop=131072, value=131072, step=2048, label="Context length (tokens)")
    a2_n_gpus = mo.ui.slider(start=1, stop=8, value=8, step=1, label="GPUs per replica")

    # -- Part B reflection --
    partB_reflection = mo.ui.radio(
        options={
            "A) INT4 weight quantization frees HBM for more KV cache, enabling larger batches": "A",
            "B) Add more compute (faster GPUs) to process more requests per second": "B",
            "C) Reduce model size from 70B to 7B": "C",
            "D) Use CPU offloading to store KV cache in system RAM": "D",
        },
        label="What is the most effective way to increase concurrent serving capacity at 128K context?",
    )
    return (a2_model_size, a2_precision, a2_context_len, a2_n_gpus, partB_reflection)


# ─── CELL 6b: Part C prediction + controls ─────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partB_reflection):
    mo.stop(partB_reflection.value is None, mo.md("**Complete Part B reflection to unlock Part C.**"))

    partC_prediction = mo.ui.radio(
        options={
            "A) 1.5x -- modest improvement over static batching": "A",
            "B) 2-4x -- continuous batching fills freed slots immediately": "B",
            "C) 10x -- batching is the dominant optimization": "C",
            "D) No improvement -- batching does not affect memory-bound workloads": "D",
        },
        label="You switch from static batching (pad all requests to max_len) to continuous batching (iteration-level scheduling). What throughput improvement do you expect for a 70B model at mixed context lengths?",
    )
    c1_avg_len = mo.ui.slider(start=256, stop=65536, value=4096, step=256, label="Average output length (tokens)")
    c1_max_len = mo.ui.slider(start=2048, stop=131072, value=32768, step=2048, label="Max context length (tokens)")
    c1_batch_size = mo.ui.slider(start=1, stop=32, value=8, step=1, label="Static batch size")
    partC_reflection = mo.ui.radio(
        options={
            "A) Static batching is fine -- just increase batch size": "A",
            "B) Continuous batching is strictly better because it eliminates padding waste and fills freed slots with new requests": "B",
            "C) Continuous batching only helps with short requests": "C",
            "D) The choice between static and continuous batching depends on model size": "D",
        },
        label="Why is continuous batching the standard for production LLM serving?",
    )
    return (partC_prediction, c1_avg_len, c1_max_len, c1_batch_size, partC_reflection)


# ─── CELL 6c: Part D prediction + controls ─────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partC_reflection):
    mo.stop(partC_reflection.value is None, mo.md("**Complete Part C reflection to unlock Part D.**"))

    partD_prediction = mo.ui.radio(
        options={
            "A) 200 replicas of FP16 with static batching -- brute force": "A",
            "B) 50 replicas of INT4 with continuous batching -- optimized per-replica throughput": "B",
            "C) 100 replicas of FP16 with continuous batching -- balanced approach": "C",
            "D) 25 replicas of INT4 with static batching -- minimize replica count": "D",
        },
        label="Design: serve 10K QPS at 200ms P99 for a 70B model at 32K context. Which fleet configuration achieves this at lowest cost?",
    )
    d1_target_qps = mo.ui.slider(start=1000, stop=20000, value=10000, step=1000, label="Target QPS")
    d1_quant = mo.ui.dropdown(
        options={"FP16 (2 bytes)": 2.0, "INT8 (1 byte)": 1.0, "INT4 (0.5 bytes)": 0.5},
        value="INT4 (0.5 bytes)",
        label="Weight quantization",
    )
    d1_batching = mo.ui.dropdown(
        options={"Static": 1.0, "Continuous": 3.0},
        value="Continuous",
        label="Batching strategy",
    )
    d1_gpus_per_replica = mo.ui.slider(start=1, stop=8, value=4, step=1, label="GPUs per replica")
    partD_reflection = mo.ui.radio(
        options={
            "A) Minimize replica count to reduce management overhead": "A",
            "B) Minimize cost = replicas * GPUs_per_replica * cost_per_GPU_hr, subject to QPS and latency constraints": "B",
            "C) Maximize batch size per replica for best GPU utilization": "C",
            "D) Use the largest GPU count per replica to maximize memory": "D",
        },
        label="What is the correct objective function for fleet design?",
    )
    return (partD_prediction, d1_target_qps, d1_quant, d1_batching, d1_gpus_per_replica, partD_reflection)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: SINGLE TABS CELL
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(
    COLORS,
    H100_RAM_GB,
    H100_COST_HR,
    TRAINING_COST_2M,
    a1_cost_query,
    a1_optimization,
    a1_qps,
    a1_weeks,
    a2_context_len,
    a2_model_size,
    a2_n_gpus,
    a2_precision,
    apply_plotly_theme,
    go,
    math,
    mo,
    np,
    partA_prediction,
    partA_reflection,
    partB_prediction,
    partB_reflection,
    partC_prediction,
    c1_avg_len,
    c1_max_len,
    c1_batch_size,
    partC_reflection,
    partD_prediction,
    d1_target_qps,
    d1_quant,
    d1_batching,
    d1_gpus_per_replica,
    partD_reflection,
):

    # ═════════════════════════════════════════════════════════════════════════
    # PART A: THE SERVING COST INVERSION
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_a():
        items = []

        # ── Stakeholder message ────────────────────────────────────────────
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['BlueLine']}; background: {COLORS['BlueLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; VP of Engineering
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "We spent $2M training our 70B LLM. Finance wants to know the total cost of
                ownership. I told them the training cost is the big expense. Our serving team
                says I am wrong. Who is right?"
            </div>
        </div>
        """))

        # ── Concept introduction ───────────────────────────────────────────
        items.append(mo.md("""
    At 100 QPS with a cost of $0.01 per query (GPU-hours amortized per request),
    daily serving cost = 100 QPS x 86,400 s/day x $0.01 = **$86,400/day**.

    The $2M training cost is a one-time capital expenditure. Serving cost accrues daily.
    At $86,400/day, the crossover occurs at 2,000,000 / 86,400 = **~23 days = ~3.3 weeks**.

    At higher QPS or cost-per-query, the crossover is even sooner.
        """))

        # ── Prediction lock ────────────────────────────────────────────────
        items.append(partA_prediction)
        if partA_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part A instruments."), kind="warn"))
            return mo.vstack(items)

        # ── Controls ───────────────────────────────────────────────────────
        items.append(mo.md("### Serving Cost Calculator"))
        items.append(mo.hstack([
            mo.vstack([a1_qps, a1_cost_query]),
            mo.vstack([a1_weeks, a1_optimization]),
        ], justify="center", gap=2))

        # ── Instruments ────────────────────────────────────────────────────
        _qps = a1_qps.value
        _cpq = a1_cost_query.value
        _weeks = a1_weeks.value
        _opt_pct = a1_optimization.value / 100

        # Daily serving cost
        _daily_cost = _qps * 86400 * _cpq * (1 - _opt_pct)
        _weekly_cost = _daily_cost * 7

        # Crossover week
        _crossover_weeks = TRAINING_COST_2M / _weekly_cost if _weekly_cost > 0 else 999
        _crossover_days = _crossover_weeks * 7

        # Annual savings from optimization
        _annual_savings = _qps * 86400 * 365 * _cpq * _opt_pct

        # ── Cost curves ───────────────────────────────────────────────
        _week_range = np.arange(0, _weeks + 1)
        _training_line = [TRAINING_COST_2M] * len(_week_range)
        _serving_cumulative = [w * _weekly_cost for w in _week_range]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_week_range, y=_training_line, mode="lines",
            name="Training cost ($2M)", line=dict(color=COLORS["BlueLine"], width=2.5, dash="dash"),
            hovertemplate="Week %{x}: $%{y:,.0f}<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=_week_range, y=_serving_cumulative, mode="lines",
            name="Cumulative serving cost", line=dict(color=COLORS["RedLine"], width=2.5),
            fill="tonexty", fillcolor="rgba(203,32,45,0.1)",
            hovertemplate="Week %{x}: $%{y:,.0f}<extra></extra>",
        ))
        if _crossover_weeks <= _weeks:
            _fig.add_vline(x=_crossover_weeks, line=dict(color=COLORS["OrangeLine"], width=2, dash="dot"),
                           annotation_text=f"Crossover: week {_crossover_weeks:.1f}",
                           annotation_position="top left")

        _fig.update_layout(
            height=340,
            xaxis=dict(title="Weeks Since Deployment"),
            yaxis=dict(title="Cumulative Cost ($)", tickformat="$,.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=70, r=20),
        )
        apply_plotly_theme(_fig)

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Serving Cost Model
            </div>
            <div>Daily serving cost = {_qps} QPS &times; 86,400 s/day &times; ${_cpq:.3f}/query &times; (1 - {_opt_pct:.0%}) = <strong>${_daily_cost:,.0f}/day</strong></div>
            <div>Crossover at week <strong>{_crossover_weeks:.1f}</strong> ({_crossover_days:.0f} days)</div>
            <div>Annual serving cost: <strong>${_daily_cost * 365:,.0f}</strong></div>
            {'<div>Annual savings from ' + str(a1_optimization.value) + '% optimization: <strong style=color:' + COLORS["GreenLine"] + ';>$' + f"{_annual_savings:,.0f}" + '</strong></div>' if _opt_pct > 0 else ''}
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Crossover</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['OrangeLine']}; font-family:monospace;">wk {_crossover_weeks:.0f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">serving &gt; training</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Daily Cost</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['RedLine']}; font-family:monospace;">${_daily_cost/1000:.0f}K</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">serving</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Annual Savings</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['GreenLine']}; font-family:monospace;">${_annual_savings/1e6:.1f}M</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">{a1_optimization.value}% optimization</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # ── Reveal ─────────────────────────────────────────────────────────
        if partA_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Correct.** At 100 QPS and $0.01/query, daily serving cost is $86,400. "
                "The $2M training cost is exceeded in ~23 days = ~3.3 weeks. At 500 QPS "
                "it crosses within a week. A 10% inference optimization at 100 QPS saves "
                "$3.15M/year -- more than the training cost itself."
            ), kind="success"))
        elif partA_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Far too conservative.** Students anchor on how expensive training *felt* "
                "but underestimate the relentless compounding of per-query cost at scale. "
                "At $86,400/day, the crossover is in weeks, not months."
            ), kind="warn"))
        elif partA_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**In the right direction but too slow.** 3 months is possible at very low "
                "QPS (~15 QPS), but 100 QPS crosses in under 4 weeks. Production LLM "
                "services typically serve 100-10,000 QPS."
            ), kind="warn"))
        elif partA_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Categorically wrong.** Training is a one-time cost; serving is a continuous "
                "operating expense. At any non-trivial QPS, serving dominates within weeks. "
                "This is why inference optimization is the highest-ROI activity for deployed models."
            ), kind="warn"))

        # ── MathPeek ───────────────────────────────────────────────────────
        items.append(mo.accordion({
            "Governing equations -- serving cost model": mo.md("""
        **Serving Cost**

        ```
        C_serving(t) = QPS * seconds_per_day * cost_per_query * t_days
        ```

        **Crossover Condition**

        ```
        C_training = C_serving(t_crossover)
        t_crossover = C_training / (QPS * 86400 * cost_per_query)
        ```

        **ROI of Inference Optimization**

        ```
        Annual_savings = QPS * 86400 * 365 * cost_per_query * optimization_fraction
        ```

        At 100 QPS and $0.01/query, a 10% optimization saves $3.15M/year.
        The training cost was $2M. The optimization pays for itself in 8 months.
            """)
        }))

        # ── Reflection ─────────────────────────────────────────────────────
        items.append(partA_reflection)
        if partA_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer."), kind="warn"))
        elif partA_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** Inference optimization directly reduces cost_per_query. "
                "A 10% improvement at 100 QPS saves $3.15M/year -- more than the training cost. "
                "Quantization (INT4 frees KV cache memory for larger batches), continuous batching "
                "(2-4x throughput), and KV cache optimization are the primary levers."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Inference optimization is the highest-ROI lever** because it reduces "
                "cost_per_query without reducing service quality or user access. "
                "Quantization + continuous batching can achieve 2-4x cost reduction."
            ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B: THE KV CACHE WALL AND FLEET DESIGN
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_b():
        items = []

        # ── Stakeholder message ────────────────────────────────────────────
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['Cloud']}; background: {COLORS['BlueLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['Cloud']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; Inference Platform Lead
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "We are deploying a 70B FP16 model on 8xH100 (640 GB total HBM). Weights take
                ~140 GB. We planned to serve 16-32 concurrent users at 128K context. Our load
                test failed -- we can only serve 1 user at a time. What went wrong?"
            </div>
        </div>
        """))

        # ── Concept introduction ───────────────────────────────────────────
        items.append(mo.md("""
    The KV cache formula for a transformer:

    **KV_cache = 2 x num_layers x hidden_dim x seq_len x batch_size x bytes_per_element**

    For a 70B model (80 layers, 8192 hidden_dim) at 128K context in FP16:
    - Per-request KV cache = 2 x 80 x 8192 x 131,072 x 2 bytes = ~343 GB
    - Available HBM after weights = 640 - 140 = 500 GB
    - Max concurrent requests = floor(500 / 343) = **1 request**

    The KV cache alone exceeds what remains after loading the model weights.
        """))

        # ── Prediction lock ────────────────────────────────────────────────
        items.append(partB_prediction)
        if partB_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part B instruments."), kind="warn"))
            return mo.vstack(items)

        # ── Controls ───────────────────────────────────────────────────────
        items.append(mo.md("### KV Cache Memory Wall Explorer"))
        items.append(mo.hstack([
            mo.vstack([a2_model_size, a2_precision]),
            mo.vstack([a2_context_len, a2_n_gpus]),
        ], justify="center", gap=2))

        # ── Instruments ────────────────────────────────────────────────────
        _params_b, _layers, _hidden = a2_model_size.value
        _bytes_per_elem = a2_precision.value
        _seq_len = a2_context_len.value
        _gpus = a2_n_gpus.value

        _total_hbm_gb = _gpus * H100_RAM_GB
        _weight_gb = _params_b * 1e9 * _bytes_per_elem / 1e9
        _available_gb = max(0, _total_hbm_gb - _weight_gb)

        # KV cache per request (bytes)
        # KV = 2 * layers * hidden * seq_len * 2 bytes (FP16 for KV regardless of weight precision)
        _kv_per_req_bytes = 2 * _layers * _hidden * _seq_len * 2  # KV always FP16
        _kv_per_req_gb = _kv_per_req_bytes / 1e9

        _max_concurrent = math.floor(_available_gb / _kv_per_req_gb) if _kv_per_req_gb > 0 else 0
        _oom = _max_concurrent < 1

        # ── Stacked memory chart ──────────────────────────────────────────────
        _n_requests = list(range(0, min(_max_concurrent + 3, 20)))
        _weight_vals = [_weight_gb] * len(_n_requests)
        _kv_vals = [n * _kv_per_req_gb for n in _n_requests]
        _total_vals = [w + k for w, k in zip(_weight_vals, _kv_vals)]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=_n_requests, y=_weight_vals, name="Model weights",
                               marker_color=COLORS["BlueLine"],
                               hovertemplate="Requests %{x}: %{y:.1f} GB<extra></extra>"))
        _kv_colors = [COLORS["GreenLine"] if t <= _total_hbm_gb else COLORS["RedLine"] for t in _total_vals]
        _fig.add_trace(go.Bar(x=_n_requests, y=_kv_vals, name="KV cache",
                               marker_color=_kv_colors,
                               hovertemplate="Requests %{x}: %{y:.1f} GB<extra></extra>"))
        _fig.add_hline(y=_total_hbm_gb, line=dict(color=COLORS["RedLine"], width=2, dash="dash"),
                       annotation_text=f"Total HBM: {_total_hbm_gb:.0f} GB", annotation_position="top right")
        _fig.update_layout(
            height=300, barmode="stack",
            xaxis=dict(title="Concurrent Requests"),
            yaxis=dict(title="Memory (GB)", range=[0, max(max(_total_vals) * 1.1, _total_hbm_gb * 1.1)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=50, r=20),
        )
        apply_plotly_theme(_fig)

        # ── OOM banner ────────────────────────────────────────────────────────
        _oom_banner = ""
        if _oom:
            _oom_banner = f"""
            <div style="background:{COLORS['RedLL']}; border:2px solid {COLORS['RedLine']};
                        border-radius:10px; padding:14px 18px; margin:10px 0;">
                <div style="font-size:0.88rem; font-weight:800; color:{COLORS['RedLine']}; margin-bottom:4px;">
                    OOM &mdash; Cannot Serve Even 1 Request
                </div>
                <div style="font-size:0.85rem; color:#7f1d1d; line-height:1.6;">
                    KV cache per request ({_kv_per_req_gb:.1f} GB) exceeds available memory ({_available_gb:.1f} GB).<br>
                    Reduce context length, add GPUs, or use weight quantization to free HBM.
                </div>
            </div>
            """

        _conc_color = COLORS["RedLine"] if _max_concurrent <= 1 else (COLORS["OrangeLine"] if _max_concurrent <= 4 else COLORS["GreenLine"])

        items.append(mo.Html(f"""
        {_oom_banner}
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; KV Cache Memory Wall
            </div>
            <div>KV per request = 2 &times; {_layers}L &times; {_hidden}H &times; {_seq_len:,} seq &times; 2 bytes = <strong>{_kv_per_req_gb:.1f} GB</strong></div>
            <div>Weights = {_params_b}B &times; {_bytes_per_elem} bytes = <strong>{_weight_gb:.1f} GB</strong></div>
            <div>Available HBM = {_total_hbm_gb:.0f} - {_weight_gb:.1f} = <strong>{_available_gb:.1f} GB</strong></div>
            <div>Max concurrent = floor({_available_gb:.1f} / {_kv_per_req_gb:.1f}) = <strong style="color:{_conc_color};">{_max_concurrent}</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Max Concurrent</div>
                <div style="font-size:2rem; font-weight:800; color:{_conc_color}; font-family:monospace;">{_max_concurrent}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">requests</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">KV/Request</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_kv_per_req_gb:.0f}GB</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Weights</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['OrangeLine']}; font-family:monospace;">{_weight_gb:.0f}GB</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Available</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['GreenLine']}; font-family:monospace;">{_available_gb:.0f}GB</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # ── Reveal ─────────────────────────────────────────────────────────
        if partB_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Correct.** At 128K context, the KV cache per request is ~343 GB for a 70B "
                "model. After loading 140 GB of FP16 weights on 8xH100 (640 GB total), only "
                "~500 GB remains. 500 / 343 = 1.46, so max concurrent = 1. The KV cache, "
                "not compute, is the binding constraint. INT4 weights (35 GB) free 605 GB, "
                "allowing 1 concurrent request. Shorter context (32K) drops KV to ~86 GB, "
                "allowing 5+ concurrent requests."
            ), kind="success"))
        elif partB_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Off by 16x.** Students think of GPUs as 'compute machines' and forget the "
                "KV cache. At 128K context, each request's KV cache is ~343 GB -- larger than "
                "the model weights themselves. Memory, not compute, is the wall."
            ), kind="warn"))
        elif partB_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Possible at shorter contexts.** At 32K context, KV cache drops to ~86 GB, "
                "allowing floor(500/86) = 5 concurrent requests. But at 128K, KV is 343 GB "
                "per request -- only 1 fits."
            ), kind="warn"))
        elif partB_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Close for some configurations.** At 64K context, KV is ~172 GB, allowing "
                "floor(500/172) = 2 concurrent requests. But at 128K, it is 343 GB -- only 1."
            ), kind="warn"))

        # ── MathPeek ───────────────────────────────────────────────────────
        items.append(mo.accordion({
            "Governing equations -- KV cache and inference fleet design": mo.md("""
        **KV Cache Formula**

        ```
        KV_bytes = 2 * num_layers * hidden_dim * seq_len * batch_size * bytes_per_elem
        ```

        - Factor 2: one K tensor + one V tensor per layer
        - KV cache is always FP16 (even with INT4 weights)
        - Grows linearly with seq_len and batch_size
        - 70B at 128K: 2 * 80 * 8192 * 131072 * 2 = ~343 GB per request

        **Memory Constraint**

        ```
        W + B * KV_per_request <= total_HBM
        Max_batch = floor((total_HBM - W) / KV_per_request)
        ```

        **Continuous Batching Throughput**

        ```
        Throughput_continuous = Throughput_static * (avg_len / max_len) * fill_factor
        ```

        - Static batching wastes (1 - avg/max) fraction of GPU cycles
        - Continuous batching fills freed slots immediately: fill_factor = 2-4x
            """)
        }))

        # ── Reflection ─────────────────────────────────────────────────────
        items.append(partB_reflection)
        if partB_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer."), kind="warn"))
        elif partB_reflection.value == "A":
            items.append(mo.callout(mo.md(
                "**Correct.** INT4 quantization reduces weight memory from 140 GB to 35 GB, "
                "freeing 105 GB of HBM for KV cache. This increases max concurrent requests "
                "from 1 to floor(605/343) = 1 at 128K context (still memory-bound), but at "
                "32K context: floor(605/86) = 7 vs 5 with FP16. The freed memory directly "
                "translates to higher throughput through larger batch sizes."
            ), kind="success"))
        elif partB_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Does not address the binding constraint.** The bottleneck is memory, not "
                "compute. Faster GPUs do not increase HBM capacity. The KV cache fills all "
                "available memory regardless of compute speed."
            ), kind="warn"))
        elif partB_reflection.value == "C":
            items.append(mo.callout(mo.md(
                "**Effective but changes the product.** A 7B model has lower quality than 70B. "
                "The goal is to serve the 70B model efficiently, not to serve a different model."
            ), kind="warn"))
        elif partB_reflection.value == "D":
            items.append(mo.callout(mo.md(
                "**Technically possible but too slow.** CPU RAM has ~10x lower bandwidth than "
                "HBM. Moving KV cache to CPU adds 10x latency to every attention computation, "
                "violating the 200ms P99 SLO. PagedAttention uses CPU offloading as a last "
                "resort, not as the primary strategy."
            ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C: CONTINUOUS BATCHING
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_c():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['OrangeLine']}; background: {COLORS['OrangeLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['OrangeLine']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; ML Serving Engineer
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "Our static batching system pads all requests to 32K tokens and waits for a
                full batch of 8. Most requests are only 4K tokens -- 87% of GPU cycles are
                wasted on padding. Someone mentioned 'continuous batching.' How much does it help?"
            </div>
        </div>
        """))

        # Concept framing
        items.append(mo.md("""
    **Static batching** pads all requests to `max_len` and processes them together.
    When a short request (4K tokens) finishes, the GPU sits idle until all 8 requests
    in the batch complete. Waste = `1 - avg_len / max_len`.

    **Continuous batching** (iteration-level scheduling) processes one token per iteration.
    When a request finishes, its slot is immediately filled by a new request.

    The throughput advantage:
    - Static: effective throughput = batch_size / max_len_time (includes padding waste)
    - Continuous: effective throughput = batch_size / avg_len_time * fill_factor

    The fill_factor (2-4x) comes from:
    1. No padding waste: compute only on real tokens
    2. Immediate slot filling: no idle GPU cycles between requests
    3. Higher effective batch occupancy over time
        """))

        # Prediction lock
        items.append(partC_prediction)
        if partC_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part C instruments."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.md("### Continuous vs Static Batching Simulator"))
        items.append(mo.hstack([c1_avg_len, c1_max_len, c1_batch_size], justify="center", gap=2))

        # Physics
        _avg = c1_avg_len.value
        _max = c1_max_len.value
        _batch = c1_batch_size.value

        _padding_waste = 1 - _avg / _max if _max > 0 else 0
        _static_throughput = _batch  # requests per batch cycle
        _continuous_throughput = _batch * (_max / _avg) * 0.85  # fill factor ~85% of theoretical
        _speedup = _continuous_throughput / _static_throughput if _static_throughput > 0 else 1

        # Chart: throughput vs avg_len ratio
        _ratios = np.linspace(0.05, 1.0, 50)
        _static_tp = [_batch for _ in _ratios]
        _continuous_tp = [_batch * (1 / r) * 0.85 for r in _ratios]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_ratios * 100, y=_static_tp, mode="lines",
            name="Static batching", line=dict(color=COLORS["RedLine"], width=2.5),
            hovertemplate="%{x:.0f}%%: %{y:.1f} req/cycle<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=_ratios * 100, y=_continuous_tp, mode="lines",
            name="Continuous batching", line=dict(color=COLORS["GreenLine"], width=2.5),
            hovertemplate="%{x:.0f}%%: %{y:.1f} req/cycle<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=[_avg / _max * 100], y=[_continuous_throughput],
            mode="markers", marker=dict(size=14, color=COLORS["OrangeLine"], symbol="diamond"),
            name="Current config",
            hovertemplate="%{x:.0f}%%: %{y:.1f} req/cycle<extra></extra>",
        ))
        _fig.update_layout(
            height=300,
            xaxis=dict(title="Avg Length / Max Length (%)"),
            yaxis=dict(title="Effective Throughput (requests/cycle)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=50, r=20),
        )
        apply_plotly_theme(_fig)

        _waste_color = COLORS["RedLine"] if _padding_waste > 0.5 else (COLORS["OrangeLine"] if _padding_waste > 0.2 else COLORS["GreenLine"])
        _speedup_color = COLORS["GreenLine"] if _speedup > 2 else (COLORS["OrangeLine"] if _speedup > 1.3 else COLORS["RedLine"])

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Batching Strategy Comparison
            </div>
            <div>Avg length: {_avg:,} tokens &mdash; Max length: {_max:,} tokens</div>
            <div>Padding waste (static): <strong style="color:{_waste_color};">{_padding_waste*100:.1f}%</strong></div>
            <div>Static throughput: <strong>{_static_throughput:.0f}</strong> req/cycle &mdash;
                 Continuous: <strong style="color:{COLORS['GreenLine']};">{_continuous_throughput:.1f}</strong> req/cycle</div>
            <div>Speedup: <strong style="color:{_speedup_color};">{_speedup:.1f}x</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Speedup</div>
                <div style="font-size:2rem; font-weight:800; color:{_speedup_color}; font-family:monospace;">{_speedup:.1f}x</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Padding Waste</div>
                <div style="font-size:2rem; font-weight:800; color:{_waste_color}; font-family:monospace;">{_padding_waste*100:.0f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">static only</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Continuous TP</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['GreenLine']}; font-family:monospace;">{_continuous_throughput:.0f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">req/cycle</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # Prediction reveal
        if partC_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** Continuous batching achieves 2-4x throughput improvement over static "
                "batching when avg_len << max_len. At avg=4K, max=32K, padding waste is 87.5%. "
                "Continuous batching eliminates this waste and immediately fills freed slots, "
                "achieving ~3x higher effective throughput in this scenario."
            ), kind="success"))
        elif partC_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Too conservative.** 1.5x would be the case if avg_len is close to max_len "
                "(e.g., 24K/32K). When avg_len is 4K vs max=32K, the 87.5% padding waste "
                "means continuous batching achieves 3-4x improvement."
            ), kind="warn"))
        elif partC_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Too optimistic.** 10x would require avg_len to be <3% of max_len AND perfect "
                "slot filling. In practice, continuous batching achieves 2-4x because slot "
                "filling is not instantaneous and prefill compute is non-trivial."
            ), kind="warn"))
        elif partC_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Wrong.** Memory-bound workloads benefit greatly from batching optimizations. "
                "Continuous batching increases the effective batch occupancy, which improves "
                "memory bandwidth utilization (more requests share the same weight reads)."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- continuous batching": mo.md("""
        **Static Batching Throughput**

        ```
        TP_static = batch_size / T_max_request
        ```

        All requests padded to max_len. Waste = 1 - avg_len/max_len.

        **Continuous Batching Throughput**

        ```
        TP_continuous = batch_size * (max_len / avg_len) * fill_factor
        ```

        - fill_factor = 0.7-0.9 (accounts for prefill overhead and scheduling gaps)
        - Speedup = (max_len / avg_len) * fill_factor
        - At avg=4K, max=32K: speedup = 8 * 0.85 = 6.8x theoretical, ~3x practical

        **Why Practical < Theoretical**

        - Prefill phase for new requests is compute-intensive
        - Not all slots fill instantly (scheduling latency)
        - KV cache management overhead
            """)
        }))

        # Reflection
        items.append(partC_reflection)
        if partC_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer."), kind="warn"))
        elif partC_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** Continuous batching is strictly better because: (1) no padding waste, "
                "(2) freed slots are filled immediately with new requests, (3) the GPU processes "
                "real tokens instead of padding tokens. This is why every production LLM serving "
                "system (vLLM, TensorRT-LLM, TGI) uses continuous batching."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Not the full picture.** Continuous batching is the standard because it "
                "eliminates padding waste AND fills slots immediately. It works for all model "
                "sizes and request length distributions."
            ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D: FLEET DESIGN CHALLENGE
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_d():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['RedLine']}; background: {COLORS['RedLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['RedLine']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; VP of AI Infrastructure
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "We need to serve 10,000 QPS at 200ms P99 for our 70B model at 32K context.
                The naive fleet (FP16, static batching) requires 200 replicas on 8xH100 each
                = 1,600 H100s at $115K/day. Can we do better?"
            </div>
        </div>
        """))

        # Concept framing
        items.append(mo.md("""
    Fleet design jointly optimizes three levers:
    1. **Quantization**: INT4 reduces weight memory, freeing HBM for larger KV cache batches
    2. **Continuous batching**: 2-4x throughput per replica
    3. **Replica count and GPU count per replica**: cost = replicas x GPUs x cost/hr

    The naive fleet (FP16, static, 8 GPUs/replica):
    - Per-replica throughput: ~50 QPS
    - Replicas needed: 10,000 / 50 = 200
    - Cost: 200 x 8 x $3/hr = $4,800/hr = $115,200/day

    The optimized fleet (INT4, continuous batching, 4 GPUs/replica):
    - INT4 weights: 35 GB (vs 140 GB FP16), freeing 105 GB for KV cache
    - Continuous batching: ~3x throughput = 150 QPS/replica
    - Replicas needed: ceil(10,000 / 150) = 67
    - Cost: 67 x 4 x $3/hr = $804/hr = $19,296/day

    **Savings: ~83%** -- same SLO, dramatically lower cost.
        """))

        # Prediction lock
        items.append(partD_prediction)
        if partD_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part D instruments."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.md("### Fleet Design Optimizer"))
        items.append(mo.hstack([
            mo.vstack([d1_target_qps, d1_quant]),
            mo.vstack([d1_batching, d1_gpus_per_replica]),
        ], justify="center", gap=2))

        # Physics
        _target = d1_target_qps.value
        _bytes_per_elem = d1_quant.value
        _batch_mult = d1_batching.value
        _gpus = d1_gpus_per_replica.value

        # Memory model
        _total_hbm = _gpus * H100_RAM_GB
        _weight_gb = 70 * 1e9 * _bytes_per_elem / 1e9
        _available_gb = max(0, _total_hbm - _weight_gb)

        # KV cache per request at 32K context (70B: 80 layers, 8192 hidden)
        _kv_per_req_gb = 2 * 80 * 8192 * 32768 * 2 / 1e9
        _max_batch = math.floor(_available_gb / _kv_per_req_gb) if _kv_per_req_gb > 0 else 0

        # Per-replica throughput
        _base_qps_per_req = 1.5  # tokens/sec/request baseline
        _effective_batch = max(1, _max_batch)
        _per_replica_qps = _effective_batch * _base_qps_per_req * _batch_mult

        # Fleet sizing
        _replicas_needed = math.ceil(_target / _per_replica_qps) if _per_replica_qps > 0 else 9999
        _total_gpus = _replicas_needed * _gpus
        _hourly_cost = _total_gpus * H100_COST_HR
        _daily_cost = _hourly_cost * 24

        # Naive baseline
        _naive_replicas = math.ceil(_target / 50)  # 50 QPS/replica naive
        _naive_gpus = _naive_replicas * 8
        _naive_daily = _naive_gpus * H100_COST_HR * 24
        _savings_pct = (1 - _daily_cost / _naive_daily) * 100 if _naive_daily > 0 else 0

        _oom = _max_batch < 1

        # Chart: cost comparison across configurations
        _configs = ["Naive FP16\nStatic 8GPU", "FP16 Cont.\n8GPU", "INT4 Cont.\n4GPU", "Your Config"]
        _costs = [
            _naive_daily,
            math.ceil(_target / (8 * 1.5 * 3.0)) * 8 * H100_COST_HR * 24,  # FP16 continuous
            math.ceil(_target / (max(1, math.floor((4 * 80 - 35) / _kv_per_req_gb)) * 1.5 * 3.0)) * 4 * H100_COST_HR * 24 if _kv_per_req_gb > 0 else 0,  # INT4 cont 4GPU
            _daily_cost,
        ]
        _bar_colors_d = [COLORS["RedLine"], COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["BlueLine"]]

        _fig = go.Figure()
        for _i, (_name, _cost) in enumerate(zip(_configs, _costs)):
            _fig.add_trace(go.Bar(
                x=[_name], y=[_cost / 1000],
                marker_color=_bar_colors_d[_i],
                text=[f"${_cost/1000:.0f}K"],
                textposition="auto",
                showlegend=False,
                hovertemplate="%{x}: $%{y:.1f}K/day<extra></extra>",
            ))
        _fig.update_layout(
            height=300,
            xaxis=dict(title="Configuration"),
            yaxis=dict(title="Daily Cost ($K)", tickformat="$,.0f"),
            margin=dict(t=30, b=70, l=70, r=20),
        )
        apply_plotly_theme(_fig)

        _cost_color = COLORS["GreenLine"] if _savings_pct > 50 else (COLORS["OrangeLine"] if _savings_pct > 20 else COLORS["RedLine"])

        if _oom:
            items.append(mo.Html(f"""
            <div style="background:{COLORS['RedLL']}; border:2px solid {COLORS['RedLine']};
                        border-radius:10px; padding:14px 18px; margin:10px 0;">
                <div style="font-size:0.88rem; font-weight:800; color:{COLORS['RedLine']}; margin-bottom:4px;">
                    OOM &mdash; Cannot Fit Any Request
                </div>
                <div style="font-size:0.85rem; color:#7f1d1d;">
                    Weights ({_weight_gb:.0f} GB) + 1 KV cache ({_kv_per_req_gb:.1f} GB) exceed
                    {_total_hbm:.0f} GB HBM. Increase GPUs per replica or use stronger quantization.
                </div>
            </div>
            """))

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Fleet Design
            </div>
            <div>Weights: {_weight_gb:.0f} GB &mdash; Available HBM: {_available_gb:.0f} GB &mdash; Max batch: {_max_batch}</div>
            <div>Per-replica QPS: {_per_replica_qps:.0f} &mdash; Replicas needed: {_replicas_needed}</div>
            <div>Total GPUs: <strong>{_total_gpus}</strong> &mdash; Daily cost: <strong>${_daily_cost:,.0f}</strong></div>
            <div>Naive baseline: {_naive_replicas} replicas &times; 8 GPUs = {_naive_gpus} GPUs = ${_naive_daily:,.0f}/day</div>
            <div>Savings: <strong style="color:{_cost_color};">{_savings_pct:.0f}%</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Daily Cost</div>
                <div style="font-size:2rem; font-weight:800; color:{_cost_color}; font-family:monospace;">${_daily_cost/1000:.0f}K</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Replicas</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_replicas_needed}</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Savings</div>
                <div style="font-size:2rem; font-weight:800; color:{_cost_color}; font-family:monospace;">{_savings_pct:.0f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">vs naive</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # Prediction reveal
        if partD_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** INT4 + continuous batching achieves ~3x throughput per replica "
                "while using only 4 GPUs instead of 8. The combination reduces fleet cost by "
                "~80% compared to the naive FP16 static approach. The key insight: quantization "
                "frees memory for larger batches, and continuous batching maximizes throughput "
                "per batch slot."
            ), kind="success"))
        elif partD_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**The most expensive option.** 200 replicas x 8 GPUs = 1,600 H100s at $115K/day. "
                "FP16 wastes HBM on weight precision that INT4 can deliver. Static batching "
                "wastes GPU cycles on padding."
            ), kind="warn"))
        elif partD_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Better than naive but still suboptimal.** FP16 weights (140 GB) leave less "
                "room for KV cache than INT4 (35 GB). With continuous batching, INT4 achieves "
                "higher per-replica throughput because it can batch more concurrent requests."
            ), kind="warn"))
        elif partD_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Static batching loses most of the INT4 benefit.** INT4 frees memory for "
                "larger batches, but static batching wastes those larger batches on padding. "
                "INT4 + continuous batching is the winning combination."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- inference fleet design": mo.md("""
        **Fleet Cost Objective**

        ```
        minimize: replicas * GPUs_per_replica * cost_per_GPU_hour
        subject to: replicas * QPS_per_replica >= target_QPS
                    P99_latency <= SLO
        ```

        **Per-Replica Throughput**

        ```
        QPS_per_replica = max_batch * base_qps * batching_multiplier
        max_batch = floor(available_HBM / KV_per_request)
        available_HBM = GPUs * RAM_per_GPU - weight_memory
        ```

        **Quantization Impact**

        INT4 vs FP16 weights: 4x memory reduction -> 4x more KV cache slots
        -> 4x larger batch -> ~4x higher QPS per replica (if memory-bound)

        Combined with 3x from continuous batching: ~12x total improvement.
            """)
        }))

        # Reflection
        items.append(partD_reflection)
        if partD_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer."), kind="warn"))
        elif partD_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** The objective is minimizing total cost subject to QPS and latency "
                "constraints. This is a constrained optimization: for each combination of "
                "(quantization, batching, GPUs_per_replica), compute the minimum replicas needed "
                "to meet the QPS target, then pick the cheapest configuration that also meets "
                "the latency SLO."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Not the right objective.** The correct objective is minimizing total fleet cost "
                "(replicas x GPUs x cost/hr) subject to meeting both the QPS target and the "
                "latency SLO. This requires jointly optimizing quantization, batching, and "
                "replica count."
            ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════════

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
                        <strong>1. Serving cost exceeds training cost within weeks, not months.</strong>
                        At 100 QPS and $0.01/query, the $2M training cost is crossed in ~3 weeks.
                        A 10% inference optimization saves $3.15M/year -- more than the training cost.
                        Inference efficiency is the highest-ROI investment for deployed models.
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>2. The KV cache, not compute, is the binding constraint on concurrency.</strong>
                        At 128K context, a 70B model's KV cache is ~343 GB per request. On 8xH100
                        (640 GB total), only 1 concurrent request fits after loading weights.
                        Memory determines how many users you can serve, not how fast you can compute.
                    </div>
                    <div>
                        <strong>3. Quantization + continuous batching transforms the economics.</strong>
                        INT4 weights free HBM for larger KV cache batches. Continuous batching
                        fills freed slots immediately, achieving 2-4x throughput over static batching.
                        Combined, they enable serving at 40% lower cost per query.
                    </div>
                </div>
            </div>
            """),
            mo.Html(f"""
            <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        What's Next
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        <strong>Lab V2-09: The Optimization Trap</strong> &mdash; You discovered that
                        inference is memory-bound. The next lab asks: if you apply the wrong
                        optimization (e.g., more compute for a memory-bound workload), what happens?
                        The roofline model diagnoses what optimization to apply and when.
                    </div>
                </div>
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        Textbook &amp; TinyTorch
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        <strong>Read:</strong> the Inference at Scale chapter for the full KV cache derivation,
                        continuous batching mechanics, and fleet design principles.<br/>
                        <strong>Build:</strong> TinyTorch inference module &mdash; implement KV cache
                        management and continuous batching in <code>tinytorch/src/inference/</code>.
                    </div>
                </div>
            </div>
            """),
            mo.accordion({
                "Self-Assessment": mo.md("""
1. At 100 QPS and $0.01/query, after how many weeks does serving cost exceed a $2M training cost?
2. For a 70B model at 128K context on 8xH100, how many concurrent requests can you serve?
3. Why does INT4 weight quantization increase serving throughput, even though it does not speed up compute?
4. What is the throughput advantage of continuous batching over static batching, and why?

*If you cannot answer all four from memory, revisit Parts A and B.*
""")
            }),
        ])

    # ═════════════════════════════════════════════════════════════════════════
    # COMPOSE TABS
    # ═════════════════════════════════════════════════════════════════════════

    tabs = mo.ui.tabs({
        "Part A -- The Serving Cost Inversion": build_part_a(),
        "Part B -- The KV Cache Wall": build_part_b(),
        "Part C -- Continuous Batching": build_part_c(),
        "Part D -- Fleet Design Challenge": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: LEDGER_HUD
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, DecisionLog):
    decision_input, decision_ui = DecisionLog()
    return (decision_input, decision_ui)


@app.cell(hide_code=True)
def _(COLORS, partA_prediction, partB_prediction, partC_prediction, partD_prediction,
      partA_reflection, partB_reflection, partC_reflection, partD_reflection,
      ledger, mo, decision_input, decision_ui):
    ledger.save(
        chapter="v2_08",
        design={
            "partA_prediction": partA_prediction.value or "no_selection",
            "partA_correct": partA_prediction.value == "C",
            "partA_reflection": partA_reflection.value or "no_selection",
            "partB_prediction": partB_prediction.value or "no_selection",
            "partB_correct": partB_prediction.value == "D",
            "partB_reflection": partB_reflection.value or "no_selection",
            "partC_prediction": partC_prediction.value or "no_selection",
            "partC_correct": partC_prediction.value == "B",
            "partC_reflection": partC_reflection.value or "no_selection",
            "partD_prediction": partD_prediction.value or "no_selection",
            "partD_correct": partD_prediction.value == "B",
            "partD_reflection": partD_reflection.value or "no_selection",
            "student_justification": str(decision_input.value),
        },
    )

    _a1_ok = partA_prediction.value == "C"
    _a2_ok = partB_prediction.value == "D"
    _tier = "Optimal" if (_a1_ok and _a2_ok) else ("Partial" if (_a1_ok or _a2_ok) else "Developing")
    _tier_color = COLORS["GreenLine"] if _tier == "Optimal" else (COLORS["OrangeLine"] if _tier == "Partial" else COLORS["TextMuted"])

    decision_ui
    mo.Html(f"""
    <div class="lab-hud">
        <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab 08</span></div>
        <div><span class="hud-label">CHAPTER</span> <span class="hud-value">v2_08 &middot; Inference at Scale</span></div>
        <div><span class="hud-label">PART A</span> <span class="{'hud-active' if _a1_ok else 'hud-none'}">{"CORRECT" if _a1_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">PART B</span> <span class="{'hud-active' if _a2_ok else 'hud-none'}">{"CORRECT" if _a2_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">TIER</span> <span style="color:{_tier_color}; font-family:var(--font-mono);">{_tier.upper()}</span></div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
