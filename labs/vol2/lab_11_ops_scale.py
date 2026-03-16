import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-11: THE SILENT FLEET
#
# Volume II, Chapter: ML Operations at Scale (ops_scale.qmd)
#
# Five Parts (~56 minutes):
#   Part A — The Complexity Explosion (10 min)
#             Operational load crosses team capacity at ~50 models.
#
#   Part B — The Silent Failure Tax (14 min, anchor)
#             A 0.5% CTR drop at 5000 QPS costs $1.08M in 24 hours.
#
#   Part C — The Platform ROI Calculator (10 min)
#             A $2M/year platform breaks even at ~20 models.
#
#   Part D — The Canary Duration Designer (12 min)
#             Staged rollout observation windows depend on traffic and canary %.
#
#   Part E — The Alert Fatigue Wall (10 min)
#             3-sigma alerting on 1000 metrics produces 864 false alerts/day.
#
# Design Ledger: chapter="v2_11"
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: SETUP + OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP ────────────────────────────────────────────────────────────
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

    ledger = DesignLedger()

    # ── Operational constants ────────────────────────────────────────────────
    # Part A — Complexity Model
    # Source: @tbl-ops-scale-complexity
    TEAM_CAPACITY_HOURS = 4000   # person-hours/year operational capacity
    ALERT_COST_PER_MODEL = 20   # hours/year per-model alert handling
    COORD_COST_LOG = 1.0         # coordination scale factor (N log N)
    DEP_COST_QUAD = 0.5          # inter-model dependency scale (N^2)

    # Part B — Silent Failure
    # Source: @sec-ml-operations-scale-staged-rollout-strategies-2d1f
    DEFAULT_QPS = 5000
    DEFAULT_CTR_DROP = 0.005     # 0.5% CTR drop
    DEFAULT_REV_PER_CLICK = 0.50  # $0.50 per click

    # Part C — Platform ROI
    # Source: @eq-platform-roi
    PLATFORM_COST_YEAR = 2_000_000   # $2M/year
    PER_MODEL_SAVINGS = 100_000      # $100K/year per model operational savings

    # Part D — Canary Duration
    # Source: @eq-canary-duration
    REQUIRED_SAMPLES = 10_000  # statistical significance threshold

    # Part E — False Alarm
    # Source: @eq-false-alert-rate
    SIGMA_3_FPR = 0.0027       # 3-sigma false positive rate
    SIGMA_4_FPR = 0.00006      # 4-sigma
    HUMAN_CAPACITY = 50         # sustainable alerts/day

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math, DecisionLog,
        TEAM_CAPACITY_HOURS, ALERT_COST_PER_MODEL,
        COORD_COST_LOG, DEP_COST_QUAD,
        DEFAULT_QPS, DEFAULT_CTR_DROP, DEFAULT_REV_PER_CLICK,
        PLATFORM_COST_YEAR, PER_MODEL_SAVINGS,
        REQUIRED_SAMPLES,
        SIGMA_3_FPR, SIGMA_4_FPR, HUMAN_CAPACITY,
    )


# ─── CELL 1: HEADER ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(LAB_CSS, mo):
    mo.vstack([
        LAB_CSS,
        mo.Html("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 11
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Silent Fleet
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.1rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Complexity &middot; Silent Failure &middot; Platform ROI &middot; Canary &middot; Alerts
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 700px; line-height: 1.65;">
                Your company has 200 models in production. Everything looks green on
                the dashboard. You are losing $1M per day. Silent model regressions,
                quadratic operational complexity, and alert fatigue create a fleet
                management crisis that cannot be solved by hiring more engineers.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    5 Parts &middot; ~56 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter: ML Operations at Scale
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px;">
                <span class="badge badge-fail">$1.08M/day silent failure</span>
                <span class="badge badge-warn">Capacity crossed at 50 models</span>
                <span class="badge badge-info">864 false alerts/day at 3-sigma</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ───────────────────────────────────────────────────────
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
                <div style="margin-bottom: 3px;">1. <strong>Predict the model count where operational load exceeds team capacity</strong>
                    &mdash; discover that O(N^2) dependency scaling causes the crossover at ~50 models, not 200.</div>
                <div style="margin-bottom: 3px;">2. <strong>Calculate the dollar cost of silent model failure</strong>
                    &mdash; a 0.5% CTR drop at 5,000 QPS costs $1.08M in 24 undetected hours.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a canary rollout</strong> that balances detection
                    sensitivity against deployment velocity, and identify the false alarm wall at fleet scale.</div>
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
                    ML deployment pipelines from @sec-ml-operations-scale &middot;
                    Statistical significance and hypothesis testing basics
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~56 min</strong><br/>
                    A: 10 &middot; B: 14 &middot; C: 10 &middot; D: 12 &middot; E: 10 min
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
                &ldquo;Your dashboard is all green. Your models are all serving. How are you
                losing $1M per day, and why does hiring more engineers make it worse?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** &mdash; Complete before this lab:

    - **Operational Complexity Scaling** &mdash; How alerts grow O(N), coordination O(N log N),
      and dependencies O(N^2) with model count (@sec-ml-operations-scale).
    - **Silent Model Regression** &mdash; Why model failures produce no errors or crashes.
    - **Staged Rollout Strategies** &mdash; Canary deployments and observation window math.
    - **Fleet-Scale Monitoring** &mdash; False alarm rates and hierarchical aggregation.
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: PART A — THE COMPLEXITY EXPLOSION
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div id="part-a" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['BlueLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;">A</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">Part A &middot; 10 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">The Complexity Explosion</div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            You assume operational cost scales linearly: 20 hours per model, 200 models,
            hire more people. The O(N^2) dependency curve crosses team capacity at ~50 models
            &mdash; far sooner than anyone expects.
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pA_pred = mo.ui.radio(
        options={
            "A: ~200 models -- linear extrapolation": "A",
            "B: ~100 models -- some overhead": "B",
            "C: ~50 models -- superlinear scaling": "C",
            "D: ~20 models -- immediate overload": "D",
        },
        label=(
            "Your ML platform team has capacity for ~4,000 person-hours/year. "
            "At what model count does operational load exceed capacity?"
        ),
    )
    mo.vstack([mo.md("### Your Prediction"), pA_pred])
    return (pA_pred,)


@app.cell(hide_code=True)
def _(mo, pA_pred):
    mo.stop(pA_pred.value is None,
            mo.callout(mo.md("Select your prediction to unlock the complexity simulator."), kind="warn"))
    return


@app.cell(hide_code=True)
def _(mo):
    pA_models = mo.ui.slider(start=1, stop=500, value=50, step=1, label="Model count")
    pA_platform = mo.ui.radio(
        options={"Platform OFF": "off", "Platform ON": "on"},
        value="Platform OFF", label="Shared ML Platform", inline=True,
    )
    mo.hstack([pA_models, pA_platform], gap="1.5rem")
    return (pA_models, pA_platform)


@app.cell(hide_code=True)
def _(ALERT_COST_PER_MODEL, COLORS, COORD_COST_LOG, DEP_COST_QUAD,
      TEAM_CAPACITY_HOURS, apply_plotly_theme, go, math, mo, np,
      pA_models, pA_platform):
    _N = pA_models.value
    _platform = pA_platform.value == "on"

    _n_range = np.arange(1, 501)
    _alerts = ALERT_COST_PER_MODEL * _n_range
    _coord = COORD_COST_LOG * _n_range * np.log2(np.maximum(_n_range, 1))
    _deps = DEP_COST_QUAD * _n_range**2
    if _platform:
        _deps = COORD_COST_LOG * 2 * _n_range * np.log2(np.maximum(_n_range, 1))  # O(N log N) with platform
    _total = _alerts + _coord + _deps

    # Current values
    _cur_alerts = ALERT_COST_PER_MODEL * _N
    _cur_coord = COORD_COST_LOG * _N * math.log2(max(_N, 1))
    _cur_deps = DEP_COST_QUAD * _N**2 if not _platform else COORD_COST_LOG * 2 * _N * math.log2(max(_N, 1))
    _cur_total = _cur_alerts + _cur_coord + _cur_deps

    # Find crossover point
    _crossover = None
    for _i, _t in enumerate(_total):
        if _t > TEAM_CAPACITY_HOURS:
            _crossover = _i + 1
            break

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_n_range, y=_alerts, mode="lines", name="Alerts O(N)",
                              line=dict(color=COLORS["BlueLine"], width=2)))
    _fig.add_trace(go.Scatter(x=_n_range, y=_coord, mode="lines", name="Coordination O(N log N)",
                              line=dict(color=COLORS["OrangeLine"], width=2)))
    _fig.add_trace(go.Scatter(x=_n_range, y=_deps, mode="lines",
                              name=f"Dependencies {'O(N log N)' if _platform else 'O(N^2)'}",
                              line=dict(color=COLORS["RedLine"] if not _platform else COLORS["GreenLine"], width=2)))
    _fig.add_trace(go.Scatter(x=_n_range, y=_total, mode="lines", name="Total Load",
                              line=dict(color="#334155", width=3)))
    _fig.add_hline(y=TEAM_CAPACITY_HOURS, line_dash="dash", line_color=COLORS["RedLine"],
                   annotation_text=f"Team Capacity: {TEAM_CAPACITY_HOURS:,} hrs/yr")
    if _crossover:
        _fig.add_vline(x=_crossover, line_dash="dot", line_color="#94a3b8",
                       annotation_text=f"Overload at {_crossover} models")
    _fig.update_layout(
        height=380,
        xaxis=dict(title="Number of Models in Production"),
        yaxis=dict(title="Operational Load (person-hours/year)"),
        legend=dict(orientation="h", y=-0.2, font_size=11),
        margin=dict(l=60, r=20, t=30, b=80),
    )
    apply_plotly_theme(_fig)

    _overload = _cur_total > TEAM_CAPACITY_HOURS
    _status_color = COLORS["RedLine"] if _overload else COLORS["GreenLine"]

    if _overload:
        _banner = mo.callout(mo.md(
            f"**OPERATIONAL OVERLOAD** -- Team cannot maintain {_N} models. "
            f"Load: {_cur_total:,.0f} hrs/yr exceeds capacity of {TEAM_CAPACITY_HOURS:,} hrs/yr."
        ), kind="danger")
    else:
        _banner = mo.callout(mo.md(
            f"Within capacity. {_cur_total:,.0f} / {TEAM_CAPACITY_HOURS:,} hrs/yr "
            f"({_cur_total/TEAM_CAPACITY_HOURS*100:.0f}% utilized)."
        ), kind="success")

    mo.vstack([
        mo.as_html(_fig),
        _banner,
        mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_status_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total Load</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_status_color};">{_cur_total:,.0f} hrs</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Crossover Point</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_crossover if _crossover else '>500'} models</div>
            </div>
        </div>"""),
        mo.md(f"""
**Complexity Model** (N = {_N}, platform = {'ON' if _platform else 'OFF'})

```
Alerts       = {ALERT_COST_PER_MODEL} x {_N} = {_cur_alerts:,.0f} hrs/yr
Coordination = {_N} x log2({_N}) = {_cur_coord:,.0f} hrs/yr
Dependencies = {'N log N' if _platform else '0.5 x N^2'} = {_cur_deps:,.0f} hrs/yr
Total        = {_cur_total:,.0f} hrs/yr  (capacity: {TEAM_CAPACITY_HOURS:,})
```
*Source: @tbl-ops-scale-complexity*
"""),
    ])
    return


@app.cell(hide_code=True)
def _(mo, pA_pred):
    if pA_pred.value == "C":
        _msg = "**Correct.** The O(N^2) dependency term dominates above ~30 models, pushing total load past capacity around N=50."
        _kind = "success"
    else:
        _msg = "**The crossover is at ~50 models.** The O(N^2) dependency term grows much faster than linear alert handling. At 50 models, dependencies alone consume ~1,250 hours. At 200 models, dependencies consume ~20,000 hours -- 5x team capacity."
        _kind = "warn"
    mo.callout(mo.md(_msg), kind=_kind)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: PART B — THE SILENT FAILURE TAX (ANCHOR)
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div id="part-b" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['RedLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;">B</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">Part B &middot; 14 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">The Silent Failure Tax</div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            Model regressions are silent: no crashes, no error logs, no alerts. The cost
            multiplier is detection latency, not regression magnitude. A 0.5% CTR drop
            detected in 1 hour costs $45K. The same regression detected in 24 hours costs $1.08M.
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pB_pred = mo.ui.number(
        start=1000, stop=10_000_000, value=None, step=1000,
        label=(
            "A recommendation model silently drops 0.5% in CTR. Traffic: 5,000 QPS. "
            "Revenue per click: $0.50. Undetected for 24 hours. Total revenue loss ($)?"
        ),
    )
    mo.vstack([mo.md("### Your Prediction"), mo.md("*Enter dollar amount:*"), pB_pred])
    return (pB_pred,)


@app.cell(hide_code=True)
def _(mo, pB_pred):
    mo.stop(pB_pred.value is None,
            mo.callout(mo.md("Enter your prediction to unlock the silent failure simulator."), kind="warn"))
    return


@app.cell(hide_code=True)
def _(mo):
    pB_qps = mo.ui.slider(start=100, stop=10000, value=5000, step=100, label="QPS")
    pB_ctr_drop = mo.ui.slider(start=0.1, stop=2.0, value=0.5, step=0.1, label="CTR drop (%)")
    pB_rev = mo.ui.slider(start=0.10, stop=2.00, value=0.50, step=0.10, label="Revenue per click ($)")
    pB_detect = mo.ui.slider(start=1, stop=48, value=24, step=1, label="Detection time (hours)")
    mo.hstack([pB_qps, pB_ctr_drop, pB_rev, pB_detect], gap="1rem")
    return (pB_ctr_drop, pB_detect, pB_qps, pB_rev)


@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, go, mo, pB_ctr_drop, pB_detect, pB_pred, pB_qps, pB_rev):
    _qps = pB_qps.value
    _ctr_drop = pB_ctr_drop.value / 100
    _rev = pB_rev.value
    _detect_hrs = pB_detect.value

    # Loss formula: QPS * 3600 * T_detection * delta_CTR * revenue_per_click
    _loss_per_hour = _qps * 3600 * _ctr_drop * _rev
    _total_loss = _loss_per_hour * _detect_hrs

    # Comparison at different detection latencies
    _detect_points = [1, 6, 12, 24, 48]
    _losses = [_loss_per_hour * h for h in _detect_points]

    _fig = go.Figure()
    _bar_colors = [COLORS["GreenLine"] if l < 100_000 else COLORS["OrangeLine"] if l < 500_000 else COLORS["RedLine"]
                   for l in _losses]
    _fig.add_trace(go.Bar(
        x=[f"{h}h" for h in _detect_points], y=_losses,
        marker_color=_bar_colors,
        text=[f"${l:,.0f}" for l in _losses], textposition="outside",
    ))
    _fig.update_layout(
        height=340,
        yaxis=dict(title="Revenue Loss ($)"),
        xaxis=dict(title="Detection Latency"),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    apply_plotly_theme(_fig)

    _loss_color = COLORS["RedLine"] if _total_loss > 500_000 else COLORS["OrangeLine"] if _total_loss > 100_000 else COLORS["GreenLine"]
    _monitoring_roi = _total_loss / 50_000  # $50K monitoring investment

    # Prediction comparison
    _predicted = pB_pred.value if pB_pred.value else 0
    _gap = _total_loss / max(_predicted, 1)

    if 0.5 < _gap < 2.0:
        _pred_msg = f"**Good estimate.** You predicted ${_predicted:,.0f}. Actual: ${_total_loss:,.0f}."
        _pred_kind = "success"
    else:
        _pred_msg = (
            f"**You predicted ${_predicted:,.0f}. Actual: ${_total_loss:,.0f}. "
            f"Off by {_gap:.1f}x.** Most students predict $10K-$50K, anchoring on '0.5% sounds small.' "
            f"But 0.5% of 5,000 QPS x $0.50 x 24 hours = $1.08M."
        )
        _pred_kind = "warn"

    mo.vstack([
        mo.as_html(_fig),
        mo.callout(mo.md(_pred_msg), kind=_pred_kind),
        mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_loss_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total Loss ({_detect_hrs}h)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_loss_color};">${_total_loss:,.0f}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Loss/Hour</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">${_loss_per_hour:,.0f}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Monitoring ROI</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_monitoring_roi:.0f}x</div>
            </div>
        </div>"""),
        mo.md(f"""
**Silent Failure Formula**

```
Loss = QPS x 3600 x T_detection x delta_CTR x rev_per_click
     = {_qps:,} x 3600 x {_detect_hrs} x {_ctr_drop:.3f} x ${_rev:.2f}
     = ${_total_loss:,.0f}

Detection latency is a {_detect_hrs}x cost multiplier:
  1-hour detection:  ${_loss_per_hour:,.0f}
  {_detect_hrs}-hour detection: ${_total_loss:,.0f}
```
*Source: @sec-ml-operations-scale-staged-rollout-strategies-2d1f*
"""),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: PART C — THE PLATFORM ROI CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div id="part-c" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['GreenLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;">C</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">Part C &middot; 10 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">The Platform ROI Calculator</div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            A $2M/year ML platform sounds expensive. It breaks even at ~20 models because
            per-model operational savings compound against a fixed platform cost.
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pC_pred = mo.ui.radio(
        options={
            "A: 100+ models": "A",
            "B: 50 models": "B",
            "C: 20 models": "C",
            "D: 5 models": "D",
        },
        label=(
            "A shared ML platform costs $2M/year. It saves ~$100K/year per model in reduced "
            "operational overhead. How many models before the platform breaks even?"
        ),
    )
    mo.vstack([mo.md("### Your Prediction"), pC_pred])
    return (pC_pred,)


@app.cell(hide_code=True)
def _(mo, pC_pred):
    mo.stop(pC_pred.value is None,
            mo.callout(mo.md("Select your prediction to unlock the ROI calculator."), kind="warn"))
    return


@app.cell(hide_code=True)
def _(mo):
    pC_models = mo.ui.slider(start=1, stop=200, value=20, step=1, label="Model count")
    pC_platform_cost = mo.ui.dropdown(
        options={"$1M/year": 1_000_000, "$2M/year": 2_000_000, "$5M/year": 5_000_000},
        value="$2M/year", label="Platform cost tier",
    )
    mo.hstack([pC_models, pC_platform_cost], gap="1.5rem")
    return (pC_models, pC_platform_cost)


@app.cell(hide_code=True)
def _(COLORS, PER_MODEL_SAVINGS, apply_plotly_theme, go, math, mo, np,
      pC_models, pC_platform_cost):
    _N = pC_models.value
    _plat_cost = pC_platform_cost.value

    _savings = _N * PER_MODEL_SAVINGS
    _roi = _savings / _plat_cost
    _breakeven = math.ceil(_plat_cost / PER_MODEL_SAVINGS)
    _net = _savings - _plat_cost

    _n_range = np.arange(1, 201)
    _savings_curve = _n_range * PER_MODEL_SAVINGS
    _cost_line = np.full_like(_n_range, _plat_cost, dtype=float)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_n_range, y=_savings_curve, mode="lines",
                              name="Cumulative Savings", line=dict(color=COLORS["GreenLine"], width=3)))
    _fig.add_trace(go.Scatter(x=_n_range, y=_cost_line, mode="lines",
                              name=f"Platform Cost (${_plat_cost/1e6:.0f}M/yr)",
                              line=dict(color=COLORS["RedLine"], width=2, dash="dash")))
    _fig.add_vline(x=_breakeven, line_dash="dot", line_color=COLORS["BlueLine"],
                   annotation_text=f"Break-even: {_breakeven} models")
    _fig.update_layout(
        height=340,
        xaxis=dict(title="Model Count"),
        yaxis=dict(title="Annual Value ($)", tickformat="$,.0f"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=70, r=20, t=30, b=80),
    )
    apply_plotly_theme(_fig)

    _roi_color = COLORS["GreenLine"] if _roi > 1 else COLORS["RedLine"]

    mo.vstack([
        mo.as_html(_fig),
        mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_roi_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">ROI</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_roi_color};">{_roi:.1f}x</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Break-Even</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_breakeven} models</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Net Annual Value</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">${_net:,.0f}</div>
            </div>
        </div>"""),
        mo.md(f"""
```
ROI = (N x Savings_per_model) / Platform_cost
    = ({_N} x ${PER_MODEL_SAVINGS:,}) / ${_plat_cost:,}
    = {_roi:.1f}x
Break-even: ${_plat_cost:,} / ${PER_MODEL_SAVINGS:,} = {_breakeven} models
```
*Source: @eq-platform-roi*
"""),
    ])
    return


@app.cell(hide_code=True)
def _(mo, pC_pred):
    if pC_pred.value == "C":
        _msg = "**Correct.** $2M / $100K per model = 20 models to break even. At 200 models, savings are $18M/year."
        _kind = "success"
    else:
        _msg = "**Break-even is at 20 models.** Students anchor on the $2M price tag and overestimate the threshold. At $100K savings per model, break-even is simple division: $2M / $100K = 20 models."
        _kind = "warn"
    mo.callout(mo.md(_msg), kind=_kind)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE E: PART D — THE CANARY DURATION DESIGNER
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div id="part-d" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['OrangeLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;">D</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">Part D &middot; 12 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">The Canary Duration Designer</div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            Staged rollouts need a minimum observation window determined by traffic volume
            and canary percentage. Too short: miss the regression. Too long: deployment stalls.
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pD_pred = mo.ui.radio(
        options={
            "A: ~6 minutes": "A",
            "B: ~1 hour": "B",
            "C: ~6 hours": "C",
            "D: ~24 hours": "D",
        },
        label=(
            "1% canary, 1M requests/hour, need 10,000 samples for significance. "
            "How long is the minimum observation window?"
        ),
    )
    mo.vstack([mo.md("### Your Prediction"), pD_pred])
    return (pD_pred,)


@app.cell(hide_code=True)
def _(mo, pD_pred):
    mo.stop(pD_pred.value is None,
            mo.callout(mo.md("Select your prediction to unlock the canary designer."), kind="warn"))
    return


@app.cell(hide_code=True)
def _(mo):
    pD_req_rate = mo.ui.slider(start=100_000, stop=10_000_000, value=1_000_000, step=100_000,
                               label="Total request rate (req/hour)")
    pD_canary_pct = mo.ui.slider(start=1, stop=50, value=1, step=1, label="Canary percentage (%)")
    mo.hstack([pD_req_rate, pD_canary_pct], gap="1.5rem")
    return (pD_canary_pct, pD_req_rate)


@app.cell(hide_code=True)
def _(COLORS, REQUIRED_SAMPLES, apply_plotly_theme, go, mo,
      pD_canary_pct, pD_req_rate):
    _rate = pD_req_rate.value
    _canary = pD_canary_pct.value / 100

    # t_stage = n_samples / (r_requests * p_stage)
    _canary_rate = _rate * _canary
    _t_hours = REQUIRED_SAMPLES / _canary_rate
    _t_minutes = _t_hours * 60

    # Multi-stage rollout: 1% -> 5% -> 25% -> 50% -> 100%
    _stages = [0.01, 0.05, 0.25, 0.50, 1.0]
    _stage_durations = [REQUIRED_SAMPLES / (_rate * s) for s in _stages]
    _total_rollout = sum(_stage_durations)

    _fig = go.Figure()
    _stage_labels = [f"{int(s*100)}%" for s in _stages]
    _dur_minutes = [d * 60 for d in _stage_durations]
    _fig.add_trace(go.Bar(
        x=_stage_labels, y=_dur_minutes,
        marker_color=[COLORS["RedLine"] if d > 60 else COLORS["OrangeLine"] if d > 10 else COLORS["GreenLine"]
                      for d in _dur_minutes],
        text=[f"{d:.0f} min" for d in _dur_minutes], textposition="outside",
    ))
    _fig.update_layout(
        height=340,
        xaxis=dict(title="Canary Stage"),
        yaxis=dict(title="Observation Window (minutes)"),
        margin=dict(l=50, r=20, t=30, b=40),
    )
    apply_plotly_theme(_fig)

    # Failure states
    _too_short = _t_minutes < 5
    _too_long = _total_rollout > 48

    if _too_short:
        _banner = mo.callout(mo.md(
            f"**WARNING: Observation window is only {_t_minutes:.1f} minutes.** "
            "This may not provide enough statistical power to detect subtle regressions."
        ), kind="warn")
    elif _too_long:
        _banner = mo.callout(mo.md(
            f"**DEPLOYMENT STALL** -- Total rollout time is {_total_rollout:.1f} hours (>{48}h limit). "
            "Increase canary percentage or reduce required samples."
        ), kind="danger")
    else:
        _banner = mo.callout(mo.md(
            f"Rollout feasible. Current stage observation: {_t_minutes:.1f} min. "
            f"Total 5-stage rollout: {_total_rollout:.1f} hours."
        ), kind="success")

    mo.vstack([
        mo.as_html(_fig),
        _banner,
        mo.md(f"""
**Canary Duration Formula**

```
t_stage = n_samples / (r_requests x p_stage)
        = {REQUIRED_SAMPLES:,} / ({_rate:,} x {_canary:.2f})
        = {_t_hours:.2f} hours ({_t_minutes:.0f} minutes)

Total 5-stage rollout: {_total_rollout:.1f} hours
```
*Source: @eq-canary-duration*
"""),
    ])
    return


@app.cell(hide_code=True)
def _(mo, pD_pred):
    if pD_pred.value == "B":
        _msg = "**Correct.** 1% of 1M = 10K req/hour to canary. Need 10K samples -> 1 hour. Students forget that only 1% of traffic reaches the canary."
        _kind = "success"
    else:
        _msg = "**The answer is ~1 hour.** At 1% canary, only 10,000 requests/hour reach the canary. To accumulate 10,000 samples takes exactly 1 hour. Students who predict 6 minutes forget to account for the canary fraction."
        _kind = "warn"
    mo.callout(mo.md(_msg), kind=_kind)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE F: PART E — THE ALERT FATIGUE WALL
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div id="part-e" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: #7c3aed; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;">E</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">Part E &middot; 10 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">The Alert Fatigue Wall</div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            With 3-sigma alerting on 1,000 metrics checked every 5 minutes, the fleet
            produces ~864 false alerts per day. Per-metric raw alerting is mathematically
            useless at fleet scale. Hierarchical monitoring is the only viable solution.
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pE_models = mo.ui.slider(start=10, stop=500, value=100, step=10, label="Model count")
    pE_metrics = mo.ui.slider(start=5, stop=20, value=10, step=1, label="Metrics per model")
    pE_sigma = mo.ui.dropdown(
        options={"2-sigma (4.55% FPR)": 0.0455, "3-sigma (0.27% FPR)": 0.0027, "4-sigma (0.006% FPR)": 0.00006},
        value="3-sigma (0.27% FPR)", label="Alert threshold",
    )
    pE_hier = mo.ui.radio(
        options={"Raw Alerting": "raw", "Hierarchical Aggregation": "hier"},
        value="Raw Alerting", label="Monitoring strategy", inline=True,
    )
    mo.hstack([pE_models, pE_metrics, pE_sigma, pE_hier], gap="1rem")
    return (pE_hier, pE_metrics, pE_models, pE_sigma)


@app.cell(hide_code=True)
def _(COLORS, HUMAN_CAPACITY, apply_plotly_theme, go, mo,
      pE_hier, pE_metrics, pE_models, pE_sigma):
    _N = pE_models.value
    _M = pE_metrics.value
    _fpr = pE_sigma.value
    _hier = pE_hier.value == "hier"

    _total_monitors = _N * _M
    if _hier:
        _total_monitors = _N * _M / 10  # hierarchical reduces effective N by 10x

    _checks_per_day = 24 * 60 / 5  # every 5 minutes
    _daily_false = _total_monitors * _fpr * _checks_per_day

    _n_range = np.arange(10, 501, 10)
    _monitors = _n_range * _M
    if _hier:
        _monitors = _monitors / 10
    _false_curve = _monitors * _fpr * _checks_per_day

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_n_range, y=_false_curve, mode="lines",
        name=f"{'Hierarchical' if _hier else 'Raw'} False Alerts/Day",
        line=dict(color=COLORS["RedLine"] if not _hier else COLORS["GreenLine"], width=3),
    ))
    _fig.add_hline(y=HUMAN_CAPACITY, line_dash="dash", line_color=COLORS["OrangeLine"],
                   annotation_text=f"Human capacity: {HUMAN_CAPACITY}/day")
    _fig.update_layout(
        height=340,
        xaxis=dict(title="Model Count"),
        yaxis=dict(title="False Alerts per Day"),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    apply_plotly_theme(_fig)

    _alert_color = COLORS["RedLine"] if _daily_false > HUMAN_CAPACITY else COLORS["GreenLine"]
    _actionable = "MEANINGLESS" if _daily_false > 200 else "DEGRADED" if _daily_false > HUMAN_CAPACITY else "ACTIONABLE"

    mo.vstack([
        mo.as_html(_fig),
        mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_alert_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">False Alerts/Day</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_alert_color};">{_daily_false:.0f}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_alert_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Alert Quality</div>
                <div style="font-size:1.3rem; font-weight:800; color:{_alert_color};">{_actionable}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total Monitors</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_total_monitors:.0f}</div>
            </div>
        </div>"""),
        mo.md(f"""
```
Total monitors       = {_N} models x {_M} metrics{' / 10 (hierarchical)' if _hier else ''} = {_total_monitors:.0f}
Checks/day           = 24h x 60min / 5min = {_checks_per_day:.0f}
False alerts/day     = {_total_monitors:.0f} x {_fpr} x {_checks_per_day:.0f} = {_daily_false:.0f}
```
*Source: @eq-false-alert-rate*
"""),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE G: SYNTHESIS + LEDGER
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(COLORS, mo):
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
                    <strong>1. Operational complexity is O(N^2), not O(N).</strong>
                    Inter-model dependencies dominate above ~30 models, crossing team capacity
                    at ~50 models. A shared platform reduces dependencies to O(N log N).
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Detection latency is the cost multiplier.</strong>
                    A 0.5% CTR drop costs $45K in 1 hour but $1.08M in 24 hours. The regression
                    magnitude is the same. The detection time is a 24x cost multiplier.
                </div>
                <div>
                    <strong>3. Raw alerting is mathematically useless at fleet scale.</strong>
                    3-sigma alerting on 1,000 metrics produces 864 false alerts/day. Hierarchical
                    monitoring reduces this to ~86/day. The solution mirrors hierarchical AllReduce
                    from collective communication: aggregate before alerting.
                </div>
            </div>
        </div>
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">What's Next</div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab V2-12: The Price of Privacy</strong> &mdash; You learned to detect
                    silent failures. Now discover that privacy and security defenses extract
                    measurable throughput, and privacy budgets deplete over time.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">Textbook &amp; TinyTorch</div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-ml-operations-scale for full derivations.<br/>
                    <strong>Build:</strong> TinyTorch monitoring module &mdash; implement canary
                    deployment with statistical significance testing.
                </div>
            </div>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, ledger, mo):
    ledger.save(chapter=11, design={
        "complexity_crossover": 50,
        "silent_failure_24h_cost": 1_080_000,
        "platform_breakeven": 20,
        "alert_fatigue_threshold": 864,
    })

    mo.Html(f"""
    <div style="background: #0f172a; border-radius: 10px; padding: 18px 24px;
                margin-top: 32px; font-family: 'SF Mono', 'Fira Code', monospace;">
        <div style="color: #475569; font-size: 0.7rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 10px;">
            Design Ledger &middot; Lab V2-11 Saved
        </div>
        <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.8;">
            <span style="color: #64748b;">complexity_crossover:</span>
            <span style="color: {COLORS['OrangeLine']};">~50 models</span><br/>
            <span style="color: #64748b;">silent_failure_24h:</span>
            <span style="color: {COLORS['RedLine']};">$1,080,000</span><br/>
            <span style="color: #64748b;">platform_breakeven:</span>
            <span style="color: {COLORS['GreenLine']};">20 models</span><br/>
            <span style="color: #64748b;">alert_fatigue_3sigma:</span>
            <span style="color: {COLORS['OrangeLine']};">864 false/day</span>
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
