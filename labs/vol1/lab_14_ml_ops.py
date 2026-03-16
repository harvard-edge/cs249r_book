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
        await micropip.install(
            "https://mlsysbook.ai/labs/wheels/mlsysim-0.1.0-py3-none-any.whl"
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    import mlsysim

    H100_TFLOPS_FP16 = mlsysim.Hardware.Cloud.H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_TDP_W       = mlsysim.Hardware.Cloud.H100.tdp.m_as("W")

    ledger = DesignLedger()
    return (
        COLORS, H100_TDP_W, H100_TFLOPS_FP16, LAB_CSS,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 14
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Silent Degradation Problem
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                PSI Drift &middot; Retraining Cadence &middot; Cost Asymmetry &middot; Debt Cascade
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Your model shipped on Monday. By Friday it has silently lost 3 accuracy
                points. By month six, 7 points. Your dashboard has been green the entire time.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~54 min</span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter 14: ML Operations</span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">PSI Monitoring</span>
                <span class="badge badge-warn">Retraining Cadence</span>
                <span class="badge badge-fail">Debt Cascade</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Identify silent degradation</strong>
                    &mdash; a model loses 7 pp accuracy in 6 months while infrastructure metrics stay green.</div>
                <div style="margin-bottom: 3px;">2. <strong>Calculate optimal retraining cadence</strong>
                    &mdash; T* = sqrt(2C / C_drift) produces a 6-day interval when drift costs $500/day.</div>
                <div style="margin-bottom: 3px;">3. <strong>Quantify technical debt compounding</strong>
                    &mdash; 3 deferred retraining cycles produce 5&ndash;6x accumulated loss, not 3x.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Degradation equation from @sec-introduction &middot;
                    Model serving from @sec-model-serving &middot;
                    Training pipeline from @sec-model-training</div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~54 min</strong><br/>
                    A: ~12 &middot; B: ~12 &middot; C: ~12 &middot; D: ~12 min</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 12px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question</div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;Your dashboard says 100% uptime, sub-50ms latency, and 0.01% error rate.
                Has anything gone wrong &mdash; and if so, how long has it been happening?&rdquo;
            </div>
        </div>
    </div>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** &mdash; Complete before this lab:

    - **@sec-ml-operations** &mdash; Distribution drift, PSI monitoring, retraining cadence
      optimization, deployment cost asymmetry, and technical debt in ML systems.
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LAB CELL
# ═════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, go, math, mo, np):

    # ── Part A widgets ────────────────────────────────────────────────────────
    partA_pred = mo.ui.radio(
        options={
            "A) Still ~95% -- system is healthy": "healthy",
            "B) Down to ~91% -- slight degradation": "slight",
            "C) Down to ~88% -- significant degradation": "significant",
            "D) Model has crashed": "crashed",
        },
        label="Fraud model deployed 6 months. Dashboard: 100% uptime, <50 ms, 0.01% error. Accuracy?",
    )
    return (partA_pred,)

@app.cell(hide_code=True)
def _(mo, partA_pred):
    partA_weeks = mo.ui.slider(start=0, stop=26, value=0, step=1,
                                label="Weeks since deployment")
    partA_drift_rate = mo.ui.slider(start=0.005, stop=0.05, value=0.02, step=0.005,
                                     label="PSI drift rate (per week)")

    # ── Part B widgets ────────────────────────────────────────────────────────
    partB_pred = mo.ui.radio(
        options={
            "A) 7 days (retrain weekly)": "7d",
            "B) ~6 days (T* = sqrt(2*10000/500))": "6d",
            "C) 14 days (biweekly)": "14d",
            "D) 30 days (current cadence is fine)": "30d",
        },
        label="Retraining costs $10K. Drift costs $500/day. Current cadence: 30 days. Optimal?",
    )
    return (partB_pred,)

@app.cell(hide_code=True)
def _(mo, partB_pred):
    partB_retrain_cost = mo.ui.slider(start=1000, stop=50000, value=10000, step=1000,
                                       label="Retraining cost ($)")
    partB_drift_cost = mo.ui.slider(start=100, stop=5000, value=500, step=100,
                                     label="Drift cost ($/day)")

    # ── Part C widgets ────────────────────────────────────────────────────────
    partC_pred = mo.ui.radio(
        options={
            "A) 700 days (100x cost = 100x interval)": "700d",
            "B) 140 days (linear proportion)": "140d",
            "C) ~70 days (sqrt(100) = 10x interval)": "70d",
            "D) 14 days (edge needs more frequent updates)": "14d",
        },
        label="Cloud retraining: $1K, T*=7 days. Edge: $100K. What is edge T*?",
    )
    return (partC_pred,)

@app.cell(hide_code=True)
def _(mo, partC_pred):
    partC_cloud_cost = mo.ui.slider(start=500, stop=10000, value=1000, step=500,
                                     label="Cloud retrain cost ($)")
    partC_edge_cost = mo.ui.slider(start=10000, stop=500000, value=100000, step=10000,
                                    label="Edge retrain cost ($)")
    partC_drift_cost = mo.ui.slider(start=100, stop=5000, value=500, step=100,
                                     label="Drift cost ($/day)")

    # ── Part D widgets ────────────────────────────────────────────────────────
    partD_pred = mo.ui.radio(
        options={
            "A) 3x (linear accumulation)": "3x",
            "B) 4x (slightly superlinear)": "4x",
            "C) ~5-6x (debt compounds)": "5x",
            "D) 9x (quadratic)": "9x",
        },
        label="You defer retraining for 3 consecutive T* cycles. Total accuracy loss vs single miss?",
    )
    return (partD_pred,)

@app.cell(hide_code=True)
def _(mo, partD_pred):
    partD_missed = mo.ui.slider(start=1, stop=6, value=3, step=1,
                                 label="Missed retraining cycles")
    partD_downstream = mo.ui.slider(start=0, stop=5, value=2, step=1,
                                     label="Dependent downstream models")
    partD_base_loss = mo.ui.slider(start=1.0, stop=5.0, value=2.0, step=0.5,
                                    label="Accuracy loss per missed cycle (pp)")

    # ═════════════════════════════════════════════════════════════════════════
    # PART A — PSI Drift Detection
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; VP of Engineering, PayGuard</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our fraud detection model has been deployed for 6 months. Dashboard shows
                100% uptime, sub-50ms latency, 0.01% error rate. The system is rock solid.
                Can you confirm everything is fine?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## Infrastructure Health and Model Health Are Decoupled

Infrastructure metrics (uptime, latency, error rate) track the **Machine axis**.
They have no sensors on the **Data axis**. Distribution drift causes:

```
PSI(t) = sum_i (p_t(i) - p_0(i)) * ln(p_t(i) / p_0(i))
Accuracy(t) = Accuracy_0 - lambda * cumulative_drift(t)
```

PSI > 0.1: notable drift. PSI > 0.2: significant drift requiring investigation.
        """))

        items.append(partA_pred)
        if partA_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the drift dashboard."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_weeks, partA_drift_rate], widths="equal"))

        _w = partA_weeks.value
        _dr = partA_drift_rate.value
        _acc0 = 95.0
        _psi_threshold = 0.2

        # PSI trajectories for three features (different drift rates)
        _weeks = np.arange(0, 27)
        _psi_txn = _dr * _weeks * 1.2  # transaction_amount drifts fastest
        _psi_cat = _dr * _weeks * 0.8  # merchant_category
        _psi_time = _dr * _weeks * 0.5  # time_of_day slowest

        # Accuracy decay proportional to max PSI
        _cum_drift = np.cumsum(_psi_txn) * 0.3  # accuracy loss from cumulative drift
        _acc_t = np.clip(_acc0 - _cum_drift, 70, 100)

        _curr_psi = _psi_txn[min(_w, 26)]
        _curr_acc = _acc_t[min(_w, 26)]
        _drop = _acc0 - _curr_acc

        # PSI alert week (when transaction_amount crosses 0.2)
        _alert_week = next((i for i, p in enumerate(_psi_txn) if p > _psi_threshold), 26)
        # Accuracy drops below 90% week
        _acc_drop_week = next((i for i, a in enumerate(_acc_t) if a < 90), 26)
        _detection_gap = max(0, _acc_drop_week - _alert_week)

        # Charts: side-by-side infrastructure vs model health
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_weeks, y=_psi_txn, name="PSI: transaction_amount",
                                   line=dict(color=COLORS['RedLine'], width=2)))
        _fig.add_trace(go.Scatter(x=_weeks, y=_psi_cat, name="PSI: merchant_category",
                                   line=dict(color=COLORS['OrangeLine'], width=2)))
        _fig.add_trace(go.Scatter(x=_weeks, y=_psi_time, name="PSI: time_of_day",
                                   line=dict(color=COLORS['BlueLine'], width=2)))
        _fig.add_hline(y=_psi_threshold, line_dash="dash", line_color=COLORS['GreenLine'],
                       annotation_text="PSI alert threshold (0.2)")
        if _w > 0:
            _fig.add_vline(x=_w, line_dash="dot", line_color="#94a3b8",
                           annotation_text=f"Week {_w}")
        _fig.update_layout(height=320, xaxis=dict(title="Weeks", range=[0, 26]),
                           yaxis=dict(title="PSI Score", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.15, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        # Dual dashboard
        _drop_col = (COLORS['RedLine'] if _drop > 5 else
                     COLORS['OrangeLine'] if _drop > 2 else COLORS['GreenLine'])
        _status = ("RETRAINING REQUIRED" if _drop > 5 else
                   "SIGNIFICANT DRIFT" if _drop > 3 else
                   "MONITOR CLOSELY" if _drop > 1 else "NOMINAL")
        _status_bg = (COLORS['RedLL'] if _drop > 5 else
                      COLORS['OrangeLL'] if _drop > 2 else COLORS['GreenLL'])
        _status_col = (COLORS['RedLine'] if _drop > 5 else
                       COLORS['OrangeLine'] if _drop > 2 else COLORS['GreenLine'])

        items.append(mo.Html(f"""
        <div style="display:flex; gap:20px; flex-wrap:wrap; margin:16px 0;">
            <div style="flex:1; min-width:260px; background:white;
                        border:1px solid #e2e8f0; border-radius:12px; padding:20px;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:12px;">
                    Infrastructure Dashboard (Week {_w})</div>
                <div style="display:flex; flex-direction:column; gap:10px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']}; font-size:0.88rem;">Uptime</span>
                        <span class="badge badge-ok">100.0%</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']}; font-size:0.88rem;">Latency</span>
                        <span class="badge badge-ok">23 ms</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']}; font-size:0.88rem;">Error Rate</span>
                        <span class="badge badge-ok">0.01%</span></div>
                </div>
                <div style="margin-top:14px; padding:10px; background:{COLORS['GreenLL']};
                            border-radius:8px; text-align:center; font-size:0.82rem;
                            font-weight:700; color:{COLORS['GreenLine']};">
                    ALL SYSTEMS OPERATIONAL</div>
            </div>
            <div style="flex:1; min-width:260px; background:white;
                        border:1px solid #e2e8f0; border-radius:12px; padding:20px;">
                <div style="font-size:0.72rem; font-weight:700; color:{_drop_col};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:12px;">
                    Model Health (Week {_w})</div>
                <div style="display:flex; flex-direction:column; gap:10px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']}; font-size:0.88rem;">Initial Accuracy</span>
                        <span style="font-family:monospace; font-weight:700; color:{COLORS['BlueLine']};">{_acc0:.1f}%</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']}; font-size:0.88rem;">Current Accuracy</span>
                        <span style="font-family:monospace; font-weight:700; color:{_drop_col};">{_curr_acc:.1f}%</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']}; font-size:0.88rem;">PSI (max feature)</span>
                        <span style="font-family:monospace; font-weight:700; color:{_drop_col};">{_curr_psi:.3f}</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']}; font-size:0.88rem;">Detection Gap</span>
                        <span style="font-family:monospace; font-weight:700; color:{COLORS['OrangeLine']};">{_detection_gap} weeks</span></div>
                </div>
                <div style="margin-top:14px; padding:10px; background:{_status_bg};
                            border-radius:8px; text-align:center; font-size:0.82rem;
                            font-weight:700; color:{_status_col};">
                    {_status}</div>
            </div>
        </div>"""))

        # Live formula
        items.append(mo.md(f"""
**PSI Drift &mdash; Live** (`week={_w}, drift_rate={_dr:.3f}/week`)

```
PSI_txn(t)     = drift_rate * t * 1.2 = {_dr:.3f} * {_w} * 1.2 = {_curr_psi:.3f}
PSI threshold  = 0.2  (significant drift)
Alert fires at = week {_alert_week}

Accuracy(t)    = {_acc0:.1f}% - cumulative_drift * 0.3
               = {_acc0:.1f}% - {_drop:.1f}
               = {_curr_acc:.1f}%

Detection gap  = acc_drop_week - PSI_alert_week
               = {_acc_drop_week} - {_alert_week} = {_detection_gap} weeks
```
*Source: PSI monitoring from @sec-ml-operations-drift-detection*
        """))

        # Reveal
        if partA_pred.value == "healthy":
            items.append(mo.callout(mo.md(
                f"**The system has degraded.** At 26 weeks with drift rate {_dr}: "
                f"accuracy = {_acc_t[26]:.1f}% (down {_acc0 - _acc_t[26]:.1f} pp). "
                "Green dashboards tell you the server is healthy, not the predictions."), kind="warn"))
        elif partA_pred.value == "significant":
            items.append(mo.callout(mo.md(
                "**Correct.** Infrastructure health and model health are decoupled. "
                f"PSI alerts fire at week {_alert_week}, but accuracy does not visibly "
                f"degrade until week {_acc_drop_week} &mdash; a {_detection_gap}-week gap."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**The key insight: dashboards stay green while accuracy erodes.** "
                f"At 26 weeks: accuracy = {_acc_t[26]:.1f}% while infrastructure is 100%."), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B — Optimal Retraining Cadence
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Escalation &middot; ML Platform Lead, PayGuard</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;The model is degrading. How often should we retrain? Weekly retraining
                costs $10K per run. Too often wastes compute. Too rarely wastes accuracy.
                Is there a formula?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## The Staleness Cost Model Produces an Optimal Interval

```
T* = sqrt(2 * C_retrain / C_drift)
```

The total annual cost is a **U-shaped curve**: too-frequent retraining wastes money
on compute, too-rare retraining wastes money on accuracy. T* is the minimum.

Key insight: 4x more expensive retraining only **doubles** the interval (square root).
        """))

        items.append(partB_pred)
        if partB_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the retraining optimizer."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_retrain_cost, partB_drift_cost], widths="equal"))

        _C = partB_retrain_cost.value
        _Cd = partB_drift_cost.value
        _Tstar = math.sqrt(2 * _C / _Cd) if _Cd > 0 else 999

        # U-curve
        _days = np.linspace(1, 90, 200)
        _retrain_annual = [365 / d * _C for d in _days]  # retrain cost per year
        _stale_annual = [_Cd * d / 2 * (365 / d) for d in _days]  # avg staleness * retrains/yr
        _total_annual = [r + s for r, s in zip(_retrain_annual, _stale_annual)]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_days, y=_retrain_annual, name="Retraining Cost",
                                   line=dict(color=COLORS['BlueLine'], width=2, dash='dot')))
        _fig.add_trace(go.Scatter(x=_days, y=_stale_annual, name="Staleness Cost",
                                   line=dict(color=COLORS['OrangeLine'], width=2, dash='dot')))
        _fig.add_trace(go.Scatter(x=_days, y=_total_annual, name="Total Annual Cost",
                                   line=dict(color=COLORS['RedLine'], width=3)))
        if _Tstar < 90:
            _min_cost = 365 / _Tstar * _C + _Cd * _Tstar / 2 * (365 / _Tstar)
            _fig.add_trace(go.Scatter(x=[_Tstar], y=[_min_cost], mode='markers',
                                       name=f'T* = {_Tstar:.1f} days',
                                       marker=dict(color=COLORS['GreenLine'], size=14, symbol='star')))
        _fig.update_layout(height=380, xaxis=dict(title="Retraining Interval (days)"),
                           yaxis=dict(title="Annual Cost ($)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=60, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:2px solid {COLORS['GreenLine']}; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Optimal Interval T*</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['GreenLine']};">{_Tstar:.1f} days</div>
                <div style="font-size:0.72rem; color:#94a3b8;">sqrt(2*{_C:,}/{_Cd:,})</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Retrains / Year</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{365/_Tstar:.0f}</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">vs 30-day Cadence</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{30/_Tstar:.1f}x too slow</div></div>
        </div>"""))

        items.append(mo.md(f"""
**Retraining Cadence &mdash; Live** (`C_retrain=${_C:,}, C_drift=${_Cd:,}/day`)

```
T* = sqrt(2 * C_retrain / C_drift)
   = sqrt(2 * {_C:,} / {_Cd:,})
   = {_Tstar:.1f} days

Current cadence (30 days) is {30/_Tstar:.1f}x the optimal interval.
```
*Source: staleness cost model from @sec-ml-operations-retraining*
        """))

        if partB_pred.value == "6d":
            items.append(mo.callout(mo.md(
                "**Correct.** T* = sqrt(2*10000/500) = 6.3 days. "
                "Your 30-day cadence was 4.8x too slow."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**T* = {math.sqrt(2*10000/500):.1f} days.** "
                "The square-root law makes T* more sensitive to drift than students expect."), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C — Deployment Cost Asymmetry
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_c():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Escalation &middot; VP Operations, PayGuard</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We deploy the same model on Cloud and Edge. Cloud retraining costs $1K.
                Edge retraining (including OTA update logistics) costs $100K. If Cloud T* is
                7 days, what should Edge T* be?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## The Square Root Surprise

```
T*_edge / T*_cloud = sqrt(C_edge / C_cloud)
```

A **100x cost difference** produces only a **10x cadence difference**.
This is deeply counterintuitive: students assume linear scaling.
        """))

        items.append(partC_pred)
        if partC_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the deployment comparison."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_cloud_cost, partC_edge_cost, partC_drift_cost], widths="equal"))

        _Cc = partC_cloud_cost.value
        _Ce = partC_edge_cost.value
        _Cd = partC_drift_cost.value
        _Tc = math.sqrt(2 * _Cc / _Cd) if _Cd > 0 else 1
        _Te = math.sqrt(2 * _Ce / _Cd) if _Cd > 0 else 1
        _cost_ratio = _Ce / _Cc if _Cc > 0 else 1
        _cadence_ratio = _Te / _Tc if _Tc > 0 else 1
        _sqrt_ratio = math.sqrt(_cost_ratio)

        items.append(mo.Html(f"""
        <div style="display:flex; gap:20px; flex-wrap:wrap; margin:16px 0;">
            <div style="flex:1; min-width:200px; background:white; border:1px solid #c7d2fe;
                        border-radius:12px; padding:20px; border-top:4px solid #6366f1;">
                <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                            text-transform:uppercase; margin-bottom:12px;">Cloud</div>
                <div style="display:flex; flex-direction:column; gap:8px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']};">Retrain Cost</span>
                        <span style="font-weight:700;">${_Cc:,}</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']};">T*</span>
                        <span style="font-weight:700; color:{COLORS['GreenLine']};">{_Tc:.1f} days</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']};">Retrains/yr</span>
                        <span style="font-weight:700;">{365/_Tc:.0f}</span></div>
                </div>
            </div>
            <div style="flex:1; min-width:200px; background:white; border:1px solid {COLORS['RedL']};
                        border-radius:12px; padding:20px; border-top:4px solid {COLORS['RedLine']};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; margin-bottom:12px;">Edge</div>
                <div style="display:flex; flex-direction:column; gap:8px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']};">Retrain Cost</span>
                        <span style="font-weight:700;">${_Ce:,}</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']};">T*</span>
                        <span style="font-weight:700; color:{COLORS['OrangeLine']};">{_Te:.1f} days</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']};">Retrains/yr</span>
                        <span style="font-weight:700;">{365/_Te:.0f}</span></div>
                </div>
            </div>
            <div style="flex:1; min-width:200px; background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                        border-radius:12px; padding:20px; border-top:4px solid {COLORS['OrangeLine']};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; margin-bottom:12px;">Square Root Surprise</div>
                <div style="display:flex; flex-direction:column; gap:8px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']};">Cost Ratio</span>
                        <span style="font-weight:700;">{_cost_ratio:.0f}x</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']};">Cadence Ratio</span>
                        <span style="font-weight:700; color:{COLORS['OrangeLine']};">{_cadence_ratio:.1f}x</span></div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{COLORS['TextSec']};">sqrt(cost ratio)</span>
                        <span style="font-weight:700;">{_sqrt_ratio:.1f}x</span></div>
                </div>
            </div>
        </div>"""))

        items.append(mo.md(f"""
**Deployment Asymmetry &mdash; Live** (`Cloud=${_Cc:,}, Edge=${_Ce:,}, drift=${_Cd:,}/day`)

```
T*_cloud = sqrt(2*{_Cc:,}/{_Cd:,}) = {_Tc:.1f} days
T*_edge  = sqrt(2*{_Ce:,}/{_Cd:,}) = {_Te:.1f} days

Cost ratio    = {_Ce:,}/{_Cc:,} = {_cost_ratio:.0f}x
Cadence ratio = {_Te:.1f}/{_Tc:.1f} = {_cadence_ratio:.1f}x = sqrt({_cost_ratio:.0f}) = {_sqrt_ratio:.1f}x
```
*Source: @sec-ml-operations-deployment-cost*
        """))

        if partC_pred.value == "70d":
            items.append(mo.callout(mo.md(
                "**Correct.** sqrt(100) = 10x. A 100x cost difference only 10x-es the interval."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**The square root law is counterintuitive.** {_cost_ratio:.0f}x cost ratio "
                f"produces only {_sqrt_ratio:.1f}x cadence ratio."), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D — The Debt Cascade
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incident Report &middot; ML Risk Assessment, PayGuard</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We have deferred retraining for 3 consecutive cycles due to resource
                constraints. How much technical debt have we accumulated? Is it 3x worse
                than missing one cycle, or is it compounding?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## Technical Debt Compounds Through Cascading Effects

Each deferred cycle makes the next one worse (larger distribution shift to bridge).
Downstream models that depend on the stale model also degrade:

```
Debt(N) = sum_{k=1}^{N} [base_loss * k^alpha + cascade_factor * N_downstream]
```

With alpha > 1, the debt is superlinear. With downstream dependencies, it multiplies.
        """))

        items.append(partD_pred)
        if partD_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the debt cascade simulator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_missed, partD_downstream, partD_base_loss], widths="equal"))

        _N = partD_missed.value
        _nd = partD_downstream.value
        _bl = partD_base_loss.value
        _alpha = 1.3  # superlinear compounding

        # Compute debt for 1 through N missed cycles
        _single_loss = _bl  # loss from 1 missed cycle
        _total_loss = sum(_bl * (k ** _alpha) for k in range(1, _N + 1))
        _cascade_cost = _N * _nd * _bl * 0.3  # cascade adds 30% per downstream per miss
        _total_with_cascade = _total_loss + _cascade_cost
        _debt_mult = _total_with_cascade / _single_loss if _single_loss > 0 else 1

        # Timeline chart
        _cycles = list(range(1, _N + 1))
        _cum_linear = [_bl * k for k in _cycles]
        _cum_compound = [sum(_bl * (j ** _alpha) for j in range(1, k + 1)) for k in _cycles]
        _cum_cascade = [c + k * _nd * _bl * 0.3 for k, c in zip(_cycles, _cum_compound)]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Linear (no compounding)", x=[str(c) for c in _cycles],
                               y=_cum_linear, marker_color=COLORS['BlueLine'], opacity=0.5))
        _fig.add_trace(go.Bar(name="Compound (superlinear)", x=[str(c) for c in _cycles],
                               y=_cum_compound, marker_color=COLORS['OrangeLine'], opacity=0.7))
        _fig.add_trace(go.Bar(name="+ Cascade (downstream)", x=[str(c) for c in _cycles],
                               y=_cum_cascade, marker_color=COLORS['RedLine'], opacity=0.88))
        _fig.update_layout(barmode="group", height=360,
                           xaxis=dict(title="Missed Cycles"),
                           yaxis=dict(title="Accumulated Accuracy Loss (pp)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Linear (expected)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_N*_bl:.1f} pp</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_N}x single cycle</div></div>
            <div style="padding:16px; border:2px solid {COLORS['RedLine']}; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Actual (with cascade)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_total_with_cascade:.1f} pp</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_debt_mult:.1f}x single cycle</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Debt Multiplier</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_debt_mult:.1f}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">vs {_N}x if linear</div></div>
        </div>"""))

        items.append(mo.md(f"""
**Debt Cascade &mdash; Live** (`N={_N} missed, {_nd} downstream, {_bl} pp/cycle, alpha=1.3`)

```
Linear (expected):    {_N} * {_bl} = {_N*_bl:.1f} pp
Compound (k^alpha):   sum(k=1..{_N}) {_bl}*k^1.3 = {_total_loss:.1f} pp
Cascade (downstream): + {_N}*{_nd}*{_bl}*0.3 = {_cascade_cost:.1f} pp
Total actual:         {_total_loss:.1f} + {_cascade_cost:.1f} = {_total_with_cascade:.1f} pp
Debt multiplier:      {_total_with_cascade:.1f} / {_single_loss:.1f} = {_debt_mult:.1f}x  (vs {_N}x if linear)
```
*Source: technical debt model from @sec-ml-operations-debt-cascade*
        """))

        if partD_pred.value == "5x":
            items.append(mo.callout(mo.md(
                f"**Correct.** Debt compounds: {_N} missed cycles at {_bl} pp/cycle "
                f"with {_nd} downstream models = {_debt_mult:.1f}x, not {_N}x."), kind="success"))
        elif partD_pred.value == "3x":
            items.append(mo.callout(mo.md(
                f"**Debt does not accumulate linearly.** Each missed cycle increases "
                f"drift further. With {_nd} downstream models, total = {_debt_mult:.1f}x."), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**The actual multiplier is {_debt_mult:.1f}x.** Compounding through "
                "distribution shift + cascade dependencies."), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════════
    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                "**1. Infrastructure health and model health are decoupled.**\n\n"
                "PSI monitoring detects distribution drift weeks before accuracy degradation "
                "becomes visible. Green dashboards create false confidence."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. T* = sqrt(2C/C_drift) is the optimal retraining interval.**\n\n"
                "4x more expensive retraining only doubles the interval. "
                "100x cost difference between Cloud and Edge produces only 10x cadence difference."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. Technical debt compounds, not accumulates.**\n\n"
                "3 missed retraining cycles produce 5-6x the loss of 1 missed cycle. "
                "Downstream model dependencies multiply the damage."
            ), kind="info"),
            mo.md("""
## Connections

**Textbook:** @sec-ml-operations &mdash; PSI monitoring, retraining cadence,
deployment cost asymmetry, technical debt in ML systems.

**TinyTorch:** Module 14 &mdash; implement a drift detector with PSI monitoring.

**Next Lab:** Lab 15 explores the costs of responsible engineering &mdash;
fairness constraints cost accuracy, explanations cost latency, and all of it costs carbon.
            """),
        ])

    tabs = mo.ui.tabs({
        "Part A \u2014 The Silent Drift":            build_part_a(),
        "Part B \u2014 Optimal Retraining Cadence":  build_part_b(),
        "Part C \u2014 Deployment Cost Asymmetry":   build_part_c(),
        "Part D \u2014 The Debt Cascade":            build_part_d(),
        "Synthesis":                                  build_synthesis(),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(COLORS, ledger, mo):
    _track = ledger.get_track()
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span><span class="hud-value">14 &mdash; ML Operations</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'NONE' else 'hud-none'}">{_track}</span>
        <span class="hud-label">STATUS</span><span class="hud-active">ACTIVE</span>
    </div>""")
    return


if __name__ == "__main__":
    app.run()
