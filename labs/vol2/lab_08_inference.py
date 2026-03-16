import marimo

__generated_with = "0.19.6"
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
# 2-Act Structure (35-40 minutes):
#   Act I  — The Serving Cost Inversion + Queuing Hockey Stick (12-15 min)
#             Serving cost > training cost within ~6 weeks at 100 QPS.
#             Batching trades latency for throughput along a hockey stick.
#
#   Act II — The KV Cache Wall + Fleet Design Challenge (20-25 min)
#             At 128K context, even 8xH100 can serve only 1 request for 70B.
#             Design a fleet: INT4 + continuous batching + right replica count
#             achieves the SLO at 40% lower cost than naive FP16 + static.
#
# Hardware Constants:
#   H100_RAM_GB        = 80     GB HBM3e per GPU
#   H100_TFLOPS_FP16   = 989    TFLOPS dense tensor core
#   H100_COST_HR       = 3.0    $/GPU-hour cloud pricing
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP ─────────────────────────────────────────────────────────────
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
        await micropip.install("https://mlsysbook.ai/labs/wheels/mlsysim-0.1.0-py3-none-any.whl")
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog

    H100_RAM_GB = 80.0
    H100_COST_HR = 3.0
    TRAINING_COST_2M = 2_000_000  # $2M training cost for 70B model

    ledger = DesignLedger()
    return COLORS, LAB_CSS, apply_plotly_theme, go, ledger, math, mo, np, H100_RAM_GB, H100_COST_HR, TRAINING_COST_2M, DecisionLog


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

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
                <span class="badge badge-warn">35&ndash;40 minutes &middot; 2 Acts</span>
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
                    KV cache formula from @sec-inference-at-scale &middot;
                    Queuing theory (Kingman's formula) from @sec-fleet-orchestration
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
                    Act I: ~12 min &middot; Act II: ~25 min
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

    - **@sec-inference-at-scale** -- Serving economics, KV cache scaling, continuous batching
    - The KV Cache section -- `KV = 2 * L * H * S * B * P` bytes formula
    - The Continuous Batching section -- Static vs iteration-level scheduling
    - The Queuing Theory section from @sec-fleet-orchestration -- Kingman's formula
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I -- SERVING COST INVERSION + KV CACHE WALL
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['BlueLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">I</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act I &middot; 12&ndash;15 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The Serving Cost Inversion
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            You think training was the expensive part. The data will show that at 100 QPS
            with $0.01 per query, serving cost crosses the $2M training cost in just 6 weeks.
            A 10% inference optimization saves more money per month than the entire training run.
        </div>
    </div>
    """)
    return


# ─── ACT1: STAKEHOLDER ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
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
    """)
    return


# ─── ACT1: PREDICTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    At 100 QPS with a cost of $0.01 per query (GPU-hours amortized per request),
    daily serving cost = 100 QPS x 86,400 s/day x $0.01 = **$86,400/day**.

    The $2M training cost is a one-time capital expenditure. Serving cost accrues daily.
    At $86,400/day, the crossover occurs at 2,000,000 / 86,400 = **~23 days = ~3.3 weeks**.

    At higher QPS or cost-per-query, the crossover is even sooner.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    partA_prediction = mo.ui.radio(
        options={
            "A) 6 months -- training dominates for a long time": "A",
            "B) 3 months -- serving catches up gradually": "B",
            "C) ~6 weeks -- serving cost grows fast": "C",
            "D) Never -- training is always more expensive": "D",
        },
        label="You spent $2M training a 70B LLM. At 100 QPS and $0.01/query, when does cumulative serving cost exceed training cost?",
    )
    partA_prediction
    return (partA_prediction,)


@app.cell(hide_code=True)
def _(partA_prediction, mo):
    mo.stop(
        partA_prediction.value is None,
        mo.callout(mo.md("Select your prediction above to unlock the Act I instruments."), kind="warn"),
    )
    mo.md("")
    return


# ─── ACT1: INSTRUMENTS ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Serving Cost Calculator")
    return


@app.cell(hide_code=True)
def _(mo):
    a1_qps = mo.ui.slider(start=10, stop=1000, value=100, step=10, label="Queries per second (QPS)")
    a1_cost_query = mo.ui.slider(start=0.001, stop=0.05, value=0.01, step=0.001, label="Cost per query ($)")
    a1_weeks = mo.ui.slider(start=1, stop=52, value=26, step=1, label="Deployment duration (weeks)")
    a1_optimization = mo.ui.slider(start=0, stop=50, value=0, step=5, label="Inference optimization (%)")
    mo.hstack([
        mo.vstack([a1_qps, a1_cost_query]),
        mo.vstack([a1_weeks, a1_optimization]),
    ], justify="center", gap=2)
    return (a1_qps, a1_cost_query, a1_weeks, a1_optimization)


@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, a1_qps, a1_cost_query, a1_weeks, a1_optimization, go, mo, np, TRAINING_COST_2M):
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

    # ── Cost curves ───────────────────────────────────────────────────────
    _week_range = np.arange(0, _weeks + 1)
    _training_line = [TRAINING_COST_2M] * len(_week_range)
    _serving_cumulative = [w * _weekly_cost for w in _week_range]

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_week_range, y=_training_line, mode="lines",
        name="Training cost ($2M)", line=dict(color=COLORS["BlueLine"], width=2.5, dash="dash"),
    ))
    _fig.add_trace(go.Scatter(
        x=_week_range, y=_serving_cumulative, mode="lines",
        name="Cumulative serving cost", line=dict(color=COLORS["RedLine"], width=2.5),
        fill="tonexty", fillcolor="rgba(203,32,45,0.1)",
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

    mo.vstack([
        mo.Html(f"""
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
        """),
        mo.Html(f"""
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
        """),
        mo.ui.plotly(_fig),
    ])
    return


# ─── ACT1: REVEAL ─────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(partA_prediction, mo):
    if partA_prediction.value == "C":
        mo.callout(mo.md(
            "**Correct.** At 100 QPS and $0.01/query, daily serving cost is $86,400. "
            "The $2M training cost is exceeded in ~23 days = ~3.3 weeks. At 500 QPS "
            "it crosses within a week. A 10% inference optimization at 100 QPS saves "
            "$3.15M/year -- more than the training cost itself."
        ), kind="success")
    elif partA_prediction.value == "A":
        mo.callout(mo.md(
            "**Far too conservative.** Students anchor on how expensive training *felt* "
            "but underestimate the relentless compounding of per-query cost at scale. "
            "At $86,400/day, the crossover is in weeks, not months."
        ), kind="warn")
    elif partA_prediction.value == "B":
        mo.callout(mo.md(
            "**In the right direction but too slow.** 3 months is possible at very low "
            "QPS (~15 QPS), but 100 QPS crosses in under 4 weeks. Production LLM "
            "services typically serve 100-10,000 QPS."
        ), kind="warn")
    elif partA_prediction.value == "D":
        mo.callout(mo.md(
            "**Categorically wrong.** Training is a one-time cost; serving is a continuous "
            "operating expense. At any non-trivial QPS, serving dominates within weeks. "
            "This is why inference optimization is the highest-ROI activity for deployed models."
        ), kind="warn")
    return


# ─── ACT1: MATHPEEK + REFLECTION ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
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
    })
    return


@app.cell(hide_code=True)
def _(mo):
    partA_reflection = mo.ui.radio(
        options={
            "A) Reduce QPS by throttling users": "A",
            "B) Optimize inference efficiency (quantization, batching, caching)": "B",
            "C) Train a smaller model": "C",
            "D) Increase the price per query": "D",
        },
        label="What is the highest-ROI lever for controlling total cost of ownership?",
    )
    partA_reflection
    return (partA_reflection,)


@app.cell(hide_code=True)
def _(partA_reflection, mo):
    mo.stop(partA_reflection.value is None, mo.callout(mo.md("Select an answer."), kind="warn"))
    if partA_reflection.value == "B":
        mo.callout(mo.md(
            "**Correct.** Inference optimization directly reduces cost_per_query. "
            "A 10% improvement at 100 QPS saves $3.15M/year -- more than the training cost. "
            "Quantization (INT4 frees KV cache memory for larger batches), continuous batching "
            "(2-4x throughput), and KV cache optimization are the primary levers."
        ), kind="success")
    else:
        mo.callout(mo.md(
            "**Inference optimization is the highest-ROI lever** because it reduces "
            "cost_per_query without reducing service quality or user access. "
            "Quantization + continuous batching can achieve 2-4x cost reduction."
        ), kind="warn")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II -- KV CACHE WALL + FLEET DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="margin: 32px 0 12px 0; border-top: 2px solid {COLORS['Border']}; padding-top: 32px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['OrangeLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">II</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act II &middot; 20&ndash;25 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The KV Cache Wall and Fleet Design
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            The KV cache grows linearly with both sequence length and batch size. At 128K
            context, even 8xH100 (640 GB HBM) can serve only 1 concurrent request for a
            70B model. Memory, not compute, is the binding constraint. Your challenge:
            design a fleet that serves 10K QPS at 200ms P99 for minimum cost.
        </div>
    </div>
    """)
    return


# ─── ACT2: STAKEHOLDER ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
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
    """)
    return


# ─── ACT2: PREDICTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The KV cache formula for a transformer:

    **KV_cache = 2 x num_layers x hidden_dim x seq_len x batch_size x bytes_per_element**

    For a 70B model (80 layers, 8192 hidden_dim) at 128K context in FP16:
    - Per-request KV cache = 2 x 80 x 8192 x 131,072 x 2 bytes = ~343 GB
    - Available HBM after weights = 640 - 140 = 500 GB
    - Max concurrent requests = floor(500 / 343) = **1 request**

    The KV cache alone exceeds what remains after loading the model weights.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    partB_prediction = mo.ui.radio(
        options={
            "A) 16-32 concurrent requests -- GPUs have plenty of memory": "A",
            "B) 4-8 concurrent requests -- some memory overhead": "B",
            "C) 2-3 concurrent requests -- memory is tighter than expected": "C",
            "D) Just 1 -- the KV cache consumes all available memory": "D",
        },
        label="70B model (FP16) on 8xH100 (640 GB). Weights = 140 GB. At 128K context, how many concurrent requests?",
    )
    partB_prediction
    return (partB_prediction,)


@app.cell(hide_code=True)
def _(partB_prediction, mo):
    mo.stop(
        partB_prediction.value is None,
        mo.callout(mo.md("Select your prediction above to unlock the Act II instruments."), kind="warn"),
    )
    mo.md("")
    return


# ─── ACT2: INSTRUMENTS ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### KV Cache Memory Wall Explorer")
    return


@app.cell(hide_code=True)
def _(mo):
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
    mo.hstack([
        mo.vstack([a2_model_size, a2_precision]),
        mo.vstack([a2_context_len, a2_n_gpus]),
    ], justify="center", gap=2)
    return (a2_model_size, a2_precision, a2_context_len, a2_n_gpus)


@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, a2_model_size, a2_precision, a2_context_len, a2_n_gpus, go, math, mo, np, H100_RAM_GB):
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
                           marker_color=COLORS["BlueLine"]))
    _kv_colors = [COLORS["GreenLine"] if t <= _total_hbm_gb else COLORS["RedLine"] for t in _total_vals]
    _fig.add_trace(go.Bar(x=_n_requests, y=_kv_vals, name="KV cache",
                           marker_color=_kv_colors))
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

    mo.vstack([
        mo.Html(f"""
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
        """),
        mo.Html(f"""
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
        """),
        mo.ui.plotly(_fig),
    ])
    return (_max_concurrent, _oom)


# ─── ACT2: REVEAL ─────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(partB_prediction, mo):
    if partB_prediction.value == "D":
        mo.callout(mo.md(
            "**Correct.** At 128K context, the KV cache per request is ~343 GB for a 70B "
            "model. After loading 140 GB of FP16 weights on 8xH100 (640 GB total), only "
            "~500 GB remains. 500 / 343 = 1.46, so max concurrent = 1. The KV cache, "
            "not compute, is the binding constraint. INT4 weights (35 GB) free 605 GB, "
            "allowing 1 concurrent request. Shorter context (32K) drops KV to ~86 GB, "
            "allowing 5+ concurrent requests."
        ), kind="success")
    elif partB_prediction.value == "A":
        mo.callout(mo.md(
            "**Off by 16x.** Students think of GPUs as 'compute machines' and forget the "
            "KV cache. At 128K context, each request's KV cache is ~343 GB -- larger than "
            "the model weights themselves. Memory, not compute, is the wall."
        ), kind="warn")
    elif partB_prediction.value == "B":
        mo.callout(mo.md(
            "**Possible at shorter contexts.** At 32K context, KV cache drops to ~86 GB, "
            "allowing floor(500/86) = 5 concurrent requests. But at 128K, KV is 343 GB "
            "per request -- only 1 fits."
        ), kind="warn")
    elif partB_prediction.value == "C":
        mo.callout(mo.md(
            "**Close for some configurations.** At 64K context, KV is ~172 GB, allowing "
            "floor(500/172) = 2 concurrent requests. But at 128K, it is 343 GB -- only 1."
        ), kind="warn")
    return


# ─── ACT2: MATHPEEK ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
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
    })
    return


# ─── ACT2: REFLECTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    partB_reflection = mo.ui.radio(
        options={
            "A) INT4 weight quantization frees HBM for more KV cache, enabling larger batches": "A",
            "B) Add more compute (faster GPUs) to process more requests per second": "B",
            "C) Reduce model size from 70B to 7B": "C",
            "D) Use CPU offloading to store KV cache in system RAM": "D",
        },
        label="What is the most effective way to increase concurrent serving capacity at 128K context?",
    )
    partB_reflection
    return (partB_reflection,)


@app.cell(hide_code=True)
def _(partB_reflection, mo):
    mo.stop(partB_reflection.value is None, mo.callout(mo.md("Select an answer."), kind="warn"))
    if partB_reflection.value == "A":
        mo.callout(mo.md(
            "**Correct.** INT4 quantization reduces weight memory from 140 GB to 35 GB, "
            "freeing 105 GB of HBM for KV cache. This increases max concurrent requests "
            "from 1 to floor(605/343) = 1 at 128K context (still memory-bound), but at "
            "32K context: floor(605/86) = 7 vs 5 with FP16. The freed memory directly "
            "translates to higher throughput through larger batch sizes."
        ), kind="success")
    elif partB_reflection.value == "B":
        mo.callout(mo.md(
            "**Does not address the binding constraint.** The bottleneck is memory, not "
            "compute. Faster GPUs do not increase HBM capacity. The KV cache fills all "
            "available memory regardless of compute speed."
        ), kind="warn")
    elif partB_reflection.value == "C":
        mo.callout(mo.md(
            "**Effective but changes the product.** A 7B model has lower quality than 70B. "
            "The goal is to serve the 70B model efficiently, not to serve a different model."
        ), kind="warn")
    elif partB_reflection.value == "D":
        mo.callout(mo.md(
            "**Technically possible but too slow.** CPU RAM has ~10x lower bandwidth than "
            "HBM. Moving KV cache to CPU adds 10x latency to every attention computation, "
            "violating the 200ms P99 SLO. PagedAttention uses CPU offloading as a last "
            "resort, not as the primary strategy."
        ), kind="warn")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.vstack([
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
                    <strong>Read:</strong> @sec-inference-at-scale for the full KV cache derivation,
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

*If you cannot answer all four from memory, revisit Acts I and II.*
""")
        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return (decision_input, decision_ui)


@app.cell(hide_code=True)
def _(COLORS, partA_prediction, partB_prediction, partA_reflection, partB_reflection,
      ledger, mo, decision_input, decision_ui):
    _max_concurrent = 0
    _oom = False
    ledger.save(
        chapter="v2_08",
        design={
            "partA_prediction": partA_prediction.value or "no_selection",
            "partA_correct": partA_prediction.value == "C",
            "partA_reflection": partA_reflection.value or "no_selection",
            "partB_prediction": partB_prediction.value or "no_selection",
            "partB_correct": partB_prediction.value == "D",
            "partB_reflection": partB_reflection.value or "no_selection",
            "student_justification": str(decision_input.value),
            "max_concurrent_requests": _max_concurrent,
            "oom_hit": _oom,
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
        <div><span class="hud-label">ACT I</span> <span class="{'hud-active' if _a1_ok else 'hud-none'}">{"CORRECT" if _a1_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">ACT II</span> <span class="{'hud-active' if _a2_ok else 'hud-none'}">{"CORRECT" if _a2_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">TIER</span> <span style="color:{_tier_color}; font-family:var(--font-mono);">{_tier.upper()}</span></div>
        <div><span class="hud-label">MAX CONC</span> <span class="hud-value">{_max_concurrent}</span></div>
        <div><span class="hud-label">OOM</span> <span class="{'hud-none' if _oom else 'hud-active'}">{"YES" if _oom else "NO"}</span></div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
