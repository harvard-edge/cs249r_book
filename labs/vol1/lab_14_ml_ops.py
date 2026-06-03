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
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    import mlsysim
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        debt_cascade,
        drift_visibility,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        ops_policy,
        ops_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        retraining_cadence,
        source_trace,
        track_context,
        track_selector,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        COLORS,
        LAB_CSS,
        apply_plotly_theme,
        build_lab_report,
        debt_cascade,
        drift_visibility,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        math,
        mlsysim,
        mo,
        np,
        ops_policy,
        ops_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        retraining_cadence,
        source_trace,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_14_metadata = get_lab_metadata("vol1/lab_14_ml_ops.py")
    return (v1_14_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v1_14_track_picker = track_selector(default=_default_track)
    v1_14_track_picker
    return (v1_14_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    ops_track_profile,
    resolve_mlsysim_ref,
    v1_14_track_picker,
):
    v1_14_track_id = v1_14_track_picker.value
    v1_14_profile = get_track_profile(v1_14_track_id)
    v1_14_variant = get_lab_track_variant("v1_14_silent_degradation", v1_14_profile.track_id)
    v1_14_hardware = resolve_mlsysim_ref(v1_14_variant.hardware_ref)
    v1_14_model = resolve_mlsysim_ref(v1_14_variant.model_ref)
    v1_14_ops = ops_track_profile(
        v1_14_profile,
        v1_14_variant,
        v1_14_hardware,
        v1_14_model,
    )
    return (
        v1_14_hardware,
        v1_14_model,
        v1_14_ops,
        v1_14_profile,
        v1_14_track_id,
        v1_14_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    v1_14_metadata,
    v1_14_ops,
    v1_14_profile,
    v1_14_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #94a3b8; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 14
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The Silent Degradation Problem
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Drift Visibility &middot; Retraining Cadence &middot; Rollback &middot; Debt Cascade
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 760px; line-height: 1.65;">
                {v1_14_variant.workload_summary} Infrastructure can stay green while
                quality silently crosses the guardrail.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~54 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_14_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_14_ops.hardware_ref}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Drift Monitor</span>
                <span class="badge badge-warn">Retrain T*</span>
                <span class="badge badge-fail">Rollback Policy</span>
            </div>
        </div>
        """),
        track_context(v1_14_profile),
        source_trace(
            {
                "lab_id": v1_14_metadata.lab_id,
                "track_id": v1_14_profile.track_id,
                "hardware_ref": v1_14_variant.hardware_ref,
                "model_ref": v1_14_variant.model_ref,
                "shared_helper": "mlsysbook_labs.ops",
                "drift_source": v1_14_ops.drift_source,
                "monitoring_signal": v1_14_ops.monitoring_signal,
                "source_policy": v1_14_profile.source_policy,
            },
            summary="V1-14 resolves operations scenarios through MLSysIM refs and mlsysbook_labs.ops calculations.",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_14_ops, v1_14_variant):
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
                <div style="margin-bottom: 3px;">1. <strong>Separate infrastructure health from model health:</strong>
                    detect when {v1_14_ops.drift_source} degrades quality before dashboards fail.</div>
                <div style="margin-bottom: 3px;">2. <strong>Choose retraining cadence:</strong>
                    use T* = sqrt(2C/C_drift) to balance retraining cost against stale-model risk.</div>
                <div style="margin-bottom: 3px;">3. <strong>Write an operations policy:</strong>
                    combine monitoring, canary rollout, rollback, escalation, and residual blind spot.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Monitoring Signal
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    {v1_14_ops.monitoring_signal}
                </div>
            </div>
            <div style="flex: 0 0 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Label Delay
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    {v1_14_ops.label_delay_days} days before the signal is fully visible
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
                "What operations policy prevents silent degradation while protecting
                {v1_14_variant.guardrail_metric}?"
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** - Complete before this lab:

    - **The ML Operations chapter** - drift, delayed labels, retraining cadence,
      rollback, escalation, and ML technical debt.
    """), kind="info")
    return


# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_14_ops):
    partA_pred = mo.ui.radio(
        options={
            "A) Quality is still healthy because uptime is green": "healthy",
            "B) True quality has fallen before labels fully show it": "silent",
            "C) Labels always arrive before quality degrades": "labels",
            "D) Drift only matters after a crash": "crash",
        },
        label=f"{v1_14_ops.label}: what happens when {v1_14_ops.drift_source} accumulates?",
    )
    return (partA_pred,)


@app.cell(hide_code=True)
def _(mo, v1_14_ops):
    partA_days = mo.ui.slider(
        start=0,
        stop=180,
        value=min(180, max(30, v1_14_ops.current_cadence_days * 2)),
        step=1,
        label="Days since deployment",
    )
    partA_rate = mo.ui.slider(
        start=0.001,
        stop=0.030,
        value=v1_14_ops.drift_rate_psi_per_day,
        step=0.001,
        label="Drift rate (PSI/day)",
    )
    partA_threshold = mo.ui.slider(
        start=0.05,
        stop=0.50,
        value=v1_14_ops.alert_threshold_psi,
        step=0.01,
        label="Alert threshold (PSI)",
    )

    partB_pred = mo.ui.radio(
        options={
            "A) Current cadence is fine": "current",
            "B) T* is shorter than current cadence": "shorter",
            "C) T* is longer because monitoring is expensive": "longer",
            "D) Retraining cadence cannot be computed": "unknown",
        },
        label=(
            f"Retrain cost ${v1_14_ops.retrain_cost:,.0f}; drift cost "
            f"${v1_14_ops.drift_cost_per_day:,.0f}/day. What cadence should ops use?"
        ),
    )
    return (partA_days, partA_rate, partA_threshold, partB_pred)


@app.cell(hide_code=True)
def _(mo, v1_14_ops):
    partB_retrain_cost = mo.ui.slider(
        start=1_000,
        stop=max(500_000, int(v1_14_ops.retrain_cost * 3)),
        value=int(v1_14_ops.retrain_cost),
        step=1_000,
        label="Retraining cost ($)",
    )
    partB_drift_cost = mo.ui.slider(
        start=100,
        stop=max(20_000, int(v1_14_ops.drift_cost_per_day * 3)),
        value=int(v1_14_ops.drift_cost_per_day),
        step=100,
        label="Drift cost ($/day)",
    )
    partB_current = mo.ui.slider(
        start=1,
        stop=120,
        value=v1_14_ops.current_cadence_days,
        step=1,
        label="Current cadence (days)",
    )

    partC_pred = mo.ui.radio(
        options={
            "A) Loose alerts and rare retraining minimize cost": "loose",
            "B) Balanced threshold, cadence, canary, and rollback": "balanced",
            "C) Canary is unnecessary if metrics are good": "no_canary",
            "D) Rollback can wait until the next release train": "slow_rollback",
        },
        label=f"Which policy protects {v1_14_ops.guardrail_metric}?",
    )
    return (partB_current, partB_drift_cost, partB_retrain_cost, partC_pred)


@app.cell(hide_code=True)
def _(mo, v1_14_ops):
    partC_threshold = mo.ui.slider(
        start=0.05,
        stop=0.50,
        value=v1_14_ops.alert_threshold_psi,
        step=0.01,
        label="Policy threshold (PSI)",
    )
    partC_cadence = mo.ui.slider(
        start=1,
        stop=120,
        value=v1_14_ops.current_cadence_days,
        step=1,
        label="Policy cadence (days)",
    )
    partC_canary = mo.ui.slider(start=0, stop=50, value=10, step=5, label="Canary traffic (%)")
    partC_rollback = mo.ui.slider(start=1, stop=72, value=8, step=1, label="Rollback exposure (hours)")

    partD_pred = mo.ui.radio(
        options={
            "A) Debt is linear in missed cycles": "linear",
            "B) Debt compounds and cascades downstream": "compound",
            "C) Debt disappears after rollback": "rollback",
            "D) Downstream models are unaffected": "isolated",
        },
        label=f"What happens if {v1_14_ops.label} misses several retraining cycles?",
    )
    return (partC_cadence, partC_canary, partC_rollback, partC_threshold, partD_pred)


@app.cell(hide_code=True)
def _(mo, v1_14_ops):
    partD_missed = mo.ui.slider(start=1, stop=6, value=3, step=1, label="Missed retraining cycles")
    partD_downstream = mo.ui.slider(
        start=0,
        stop=8,
        value=v1_14_ops.downstream_models,
        step=1,
        label="Dependent downstream models",
    )
    partD_base_loss = mo.ui.slider(
        start=0.5,
        stop=5.0,
        value=v1_14_ops.base_loss_pp,
        step=0.5,
        label="Base loss per missed cycle (pp)",
    )
    return (partD_base_loss, partD_downstream, partD_missed)


# ===========================================================================
# ZONE C: MAIN LAB
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    debt_cascade,
    drift_visibility,
    go,
    math,
    mo,
    np,
    ops_policy,
    partA_days,
    partA_pred,
    partA_rate,
    partA_threshold,
    partB_current,
    partB_drift_cost,
    partB_pred,
    partB_retrain_cost,
    partC_cadence,
    partC_canary,
    partC_pred,
    partC_rollback,
    partC_threshold,
    partD_base_loss,
    partD_downstream,
    partD_missed,
    partD_pred,
    retraining_cadence,
    v1_14_ops,
    v1_14_profile,
    v1_14_variant,
):
    def _metric_card(label, value, detail, color, border=False):
        border_style = f"2px solid {color}" if border else "1px solid #e2e8f0"
        return f"""
        <div style="padding:16px; border:{border_style}; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {color}; flex:1;">
            <div style="color:#64748b; font-size:0.78rem; font-weight:700;">{label}</div>
            <div style="font-size:1.55rem; font-weight:800; color:{color};">{value}</div>
            <div style="font-size:0.72rem; color:#64748b;">{detail}</div>
        </div>
        """

    def build_part_a():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Incoming Message &middot; {v1_14_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Infrastructure is green. The drift source is {v1_14_ops.drift_source}.
                    Are we still safe?"
                </div>
            </div>
            """),
            mo.md("""
## Model Health Can Fail While Infrastructure Health Stays Green

Drift monitoring needs both a true quality estimate and a delayed observed signal.
The key operations risk is the gap between quality crossing a guardrail and the
monitoring signal becoming actionable.
            """),
            partA_pred,
        ]
        if partA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the drift timeline."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_days, partA_rate, partA_threshold], widths="equal"))
        _result = drift_visibility(
            v1_14_ops,
            days_since_deploy=partA_days.value,
            drift_rate_psi_per_day=partA_rate.value,
            alert_threshold_psi=partA_threshold.value,
        )

        _days = np.arange(0, 181)
        _true_quality = []
        _observed_quality = []
        _true_psi = []
        _observed_psi = []
        for _day in _days:
            _r = drift_visibility(
                v1_14_ops,
                days_since_deploy=int(_day),
                drift_rate_psi_per_day=partA_rate.value,
                alert_threshold_psi=partA_threshold.value,
            )
            _true_quality.append(_r.true_quality_pct)
            _observed_quality.append(_r.observed_quality_pct)
            _true_psi.append(_r.true_psi)
            _observed_psi.append(_r.observed_psi)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_days, y=_true_quality, name="True quality", line=dict(color=COLORS["RedLine"], width=3)))
        _fig.add_trace(go.Scatter(x=_days, y=_observed_quality, name="Observed quality", line=dict(color=COLORS["BlueLine"], width=2, dash="dot")))
        _fig.add_hline(y=v1_14_ops.quality_floor_pct, line_dash="dash", line_color=COLORS["OrangeLine"], annotation_text="quality floor")
        _fig.add_vline(x=_result.alert_day, line_dash="dash", line_color=COLORS["GreenLine"], annotation_text="alert")
        _fig.add_vline(x=partA_days.value, line_dash="dot", line_color="#64748b", annotation_text=f"day {partA_days.value}")
        _fig.update_layout(
            height=360,
            xaxis=dict(title="Days since deployment"),
            yaxis=dict(title="Quality (%)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _quality_color = COLORS["RedLine"] if _result.quality_breached else COLORS["GreenLine"]
        _alert_color = COLORS["GreenLine"] if _result.alert_triggered else COLORS["OrangeLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("True PSI", f"{_result.true_psi:.3f}", "actual drift", COLORS["RedLine"])}
            {_metric_card("Observed PSI", f"{_result.observed_psi:.3f}", f"{v1_14_ops.label_delay_days} day delay", COLORS["BlueLine"])}
            {_metric_card("True Quality", f"{_result.true_quality_pct:.1f}%", f"floor {v1_14_ops.quality_floor_pct:.1f}%", _quality_color, True)}
            {_metric_card("Alert Day", f"{_result.alert_day}", "monitor visibility", _alert_color)}
        </div>
        """))

        if _result.quality_breached and not _result.alert_triggered:
            items.append(mo.callout(mo.md(
                f"**Silent degradation window.** Quality has crossed the floor, but the alert has not fired. "
                f"Detection delay is {_result.detection_delay_days} day(s)."
            ), kind="danger"))

        items.append(mo.md(f"""
**Drift Visibility - Live Calculation**

```
drift source      = {v1_14_ops.drift_source}
monitoring signal = {v1_14_ops.monitoring_signal}
true PSI          = {_result.true_psi:.3f}
observed PSI      = {_result.observed_psi:.3f}
true quality      = {_result.true_quality_pct:.1f}%
observed quality  = {_result.observed_quality_pct:.1f}%
damage cost       = ${_result.accumulated_damage_cost:,.0f}
```
*Source: `mlsysbook_labs.drift_visibility`, track `{v1_14_profile.track_id}`.*
        """))

        if partA_pred.value == "silent":
            items.append(mo.callout(mo.md("**Correct.** Delayed labels and proxy metrics mean true quality can fall first."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**The server can be healthy while the model is stale.** Model health needs drift and delayed-label monitoring."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Cadence Review &middot; ML Platform Lead
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Retrain too often and we waste resources. Retrain too slowly and stale-model
                    risk accumulates. What is the operating cadence?"
                </div>
            </div>
            """),
            mo.md("""
## Retraining Cadence Has a Square-Root Optimum

```
T* = sqrt(2 * retrain_cost / drift_cost_per_day)
```

The total annual cost curve is U-shaped: retraining cost falls with longer
intervals, while stale-model risk rises with longer intervals.
            """),
            partB_pred,
        ]
        if partB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the cadence optimizer."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_retrain_cost, partB_drift_cost, partB_current], widths="equal"))
        _cadence = retraining_cadence(
            retrain_cost=partB_retrain_cost.value,
            drift_cost_per_day=partB_drift_cost.value,
            current_days=partB_current.value,
        )
        _days = np.linspace(1, 120, 240)
        _costs = []
        for _d in _days:
            _retrain = 365 / _d * partB_retrain_cost.value
            _stale = partB_drift_cost.value * 365 * _d / 2
            _costs.append(_retrain + _stale)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_days, y=_costs, name="Total annual cost", line=dict(color=COLORS["RedLine"], width=3)))
        _fig.add_vline(x=_cadence.optimal_days, line_dash="dash", line_color=COLORS["GreenLine"], annotation_text=f"T* = {_cadence.optimal_days:.1f} d")
        _fig.add_vline(x=partB_current.value, line_dash="dot", line_color=COLORS["OrangeLine"], annotation_text="current")
        _fig.update_layout(
            height=360,
            xaxis=dict(title="Retraining interval (days)"),
            yaxis=dict(title="Annual cost ($)", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=50, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _factor_color = COLORS["RedLine"] if _cadence.current_too_slow_factor > 2 else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Optimal T*", f"{_cadence.optimal_days:.1f} d", "sqrt(2C/Cd)", COLORS["GreenLine"], True)}
            {_metric_card("Retrains / Year", f"{_cadence.retrains_per_year:.1f}", "at T*", COLORS["BlueLine"])}
            {_metric_card("Current Factor", f"{_cadence.current_too_slow_factor:.1f}x", f"current {partB_current.value} d", _factor_color)}
            {_metric_card("Annual Savings", f"${_cadence.savings_vs_current:,.0f}", "vs current cadence", COLORS["OrangeLine"])}
        </div>
        """))

        items.append(mo.md(f"""
**Retraining Cadence - Live Calculation**

```
retrain cost = ${partB_retrain_cost.value:,.0f}
drift cost   = ${partB_drift_cost.value:,.0f}/day
T*           = sqrt(2 * {partB_retrain_cost.value:,.0f} / {partB_drift_cost.value:,.0f})
             = {_cadence.optimal_days:.1f} days
current      = {partB_current.value} days ({_cadence.current_too_slow_factor:.1f}x T*)
```
*Source: `mlsysbook_labs.retraining_cadence`.*
        """))

        if partB_pred.value == "shorter" and _cadence.current_too_slow_factor > 1:
            items.append(mo.callout(mo.md("**Correct.** The current cadence is slower than the cost/risk optimum."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Cadence is computable.** Use the square-root law, then validate with track-specific rollout and guardrail costs."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Policy Review &middot; {v1_14_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Choose the monitor threshold, retraining cadence, canary, and rollback window.
                    What policy is defensible?"
                </div>
            </div>
            """),
            mo.md(f"""
## Operations Policy Combines Monitoring, Rollout, Rollback, And Escalation

For **{v1_14_ops.label}**, rollback is:

```
{v1_14_ops.rollback_policy}
```

The policy must make cost visible without hiding residual risk.
            """),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the policy scorer."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_threshold, partC_cadence, partC_canary, partC_rollback], widths="equal"))
        _policy = ops_policy(
            v1_14_ops,
            threshold_psi=partC_threshold.value,
            cadence_days=partC_cadence.value,
            canary_pct=partC_canary.value,
            rollback_hours=partC_rollback.value,
        )

        _fig = go.Figure()
        for _name, _value, _color in [
            ("Monitoring", _policy.annual_monitoring_cost, COLORS["BlueLine"]),
            ("Retraining", _policy.annual_retrain_cost, COLORS["OrangeLine"]),
            ("Residual risk", _policy.annual_risk_cost, COLORS["RedLine"]),
        ]:
            _fig.add_trace(go.Bar(x=[_name], y=[_value], name=_name, marker_color=_color))
        _fig.update_layout(
            height=330,
            yaxis=dict(title="Annual policy cost ($)", gridcolor="#f1f5f9"),
            showlegend=False,
            margin=dict(l=60, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _policy_color = COLORS["GreenLine"] if _policy.feasible else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Detection Day", f"{_policy.expected_detection_day:.1f}", "threshold/rate + delay", COLORS["BlueLine"])}
            {_metric_card("Stale Days", f"{_policy.stale_days:.1f}", "cadence beyond T*", COLORS["OrangeLine"])}
            {_metric_card("Total Annual Cost", f"${_policy.total_annual_cost:,.0f}", "monitor + retrain + risk", COLORS["RedLine"])}
            {_metric_card("Policy Status", "PASS" if _policy.feasible else "FAIL", ", ".join(_policy.violations) or "no violations", _policy_color, True)}
        </div>
        """))

        if not _policy.feasible:
            items.append(mo.callout(mo.md(
                "**Policy violations:** " + ", ".join(_policy.violations) + ". Tighten the threshold, cadence, canary, or rollback window."
            ), kind="danger"))

        items.append(mo.md(f"""
**Ops Policy - Live Calculation**

```
threshold      = {partC_threshold.value:.2f} PSI
cadence        = {partC_cadence.value} days
canary         = {partC_canary.value}%
rollback       = {partC_rollback.value} hours
escalation     = {v1_14_ops.escalation_policy}
annual cost    = ${_policy.total_annual_cost:,.0f}
```
*Source: `mlsysbook_labs.ops_policy`.*
        """))

        if partC_pred.value == "balanced" and _policy.feasible:
            items.append(mo.callout(mo.md("**Correct.** A defensible policy couples monitoring, cadence, canary, rollback, and owner escalation."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Ops policy is a system design.** A green metric without canary, rollback, and owner escalation is not enough."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_d():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Incident Review &middot; ML Risk Assessment
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "We missed several retraining cycles. Is this just linear debt,
                    or does it compound through downstream systems?"
                </div>
            </div>
            """),
            mo.md("""
## ML Technical Debt Compounds

Each missed cycle makes the next update harder because the distribution shift is
larger. Downstream models that consume stale predictions add cascade loss.
            """),
            partD_pred,
        ]
        if partD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the debt cascade simulator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_missed, partD_downstream, partD_base_loss], widths="equal"))
        _debt = debt_cascade(
            missed_cycles=partD_missed.value,
            downstream_models=partD_downstream.value,
            base_loss_pp=partD_base_loss.value,
        )

        _cycles = list(range(1, partD_missed.value + 1))
        _linear = [partD_base_loss.value * _c for _c in _cycles]
        _compound = [
            debt_cascade(missed_cycles=_c, downstream_models=0, base_loss_pp=partD_base_loss.value).compound_loss_pp
            for _c in _cycles
        ]
        _cascade = [
            debt_cascade(missed_cycles=_c, downstream_models=partD_downstream.value, base_loss_pp=partD_base_loss.value).total_loss_pp
            for _c in _cycles
        ]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Linear", x=[str(_c) for _c in _cycles], y=_linear, marker_color=COLORS["BlueLine"], opacity=0.55))
        _fig.add_trace(go.Bar(name="Compound", x=[str(_c) for _c in _cycles], y=_compound, marker_color=COLORS["OrangeLine"], opacity=0.75))
        _fig.add_trace(go.Bar(name="+ Cascade", x=[str(_c) for _c in _cycles], y=_cascade, marker_color=COLORS["RedLine"], opacity=0.9))
        _fig.update_layout(
            barmode="group",
            height=360,
            xaxis=dict(title="Missed cycles"),
            yaxis=dict(title="Accumulated quality loss (pp)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Linear Loss", f"{_debt.linear_loss_pp:.1f} pp", "missed * base", COLORS["BlueLine"])}
            {_metric_card("Compound Loss", f"{_debt.compound_loss_pp:.1f} pp", "superlinear drift", COLORS["OrangeLine"])}
            {_metric_card("Cascade Loss", f"{_debt.cascade_loss_pp:.1f} pp", f"{_debt.downstream_models} downstream", COLORS["RedLine"])}
            {_metric_card("Debt Multiplier", f"{_debt.debt_multiplier:.1f}x", "vs one missed cycle", COLORS["RedLine"], True)}
        </div>
        """))

        items.append(mo.md(f"""
**Debt Cascade - Live Calculation**

```
missed cycles = {_debt.missed_cycles}
downstream    = {_debt.downstream_models}
base loss     = {_debt.base_loss_pp:.1f} pp
linear loss   = {_debt.linear_loss_pp:.1f} pp
actual loss   = {_debt.total_loss_pp:.1f} pp
multiplier    = {_debt.debt_multiplier:.1f}x
```
*Source: `mlsysbook_labs.debt_cascade`.*
        """))

        if partD_pred.value == "compound":
            items.append(mo.callout(mo.md("**Correct.** Missed cycles compound and downstream dependencies cascade the loss."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Debt is not linear.** Missed retraining increases drift and propagates stale predictions downstream."
            ), kind="warn"))
        return mo.vstack(items)

    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                f"**1. Model health needs track-specific signals.** For {v1_14_ops.label}, the signal is {v1_14_ops.monitoring_signal}."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. Retraining cadence is an economics and risk calculation.** T* balances fixed retraining cost against stale-model risk."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. Operations policy is the student artifact.** It must name threshold, cadence, canary, rollback, escalation, and residual blind spot."
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
                        <strong>Lab 15: Responsible Engineering</strong> - after operations
                        policy, the next question is whose outcomes and constraints are protected.
                    </div>
                </div>
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        Report Focus
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        Submit a compact incident-prevention policy for {v1_14_ops.label}
                        with evidence from each part.
                    </div>
                </div>
            </div>
            """),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Drift Visibility": build_part_a(),
        "Part B: Retraining Cadence": build_part_b(),
        "Part C: Ops Policy": build_part_c(),
        "Part D: Debt Cascade": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


# ===========================================================================
# ZONE D: LEDGER HUD
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    partA_pred,
    partB_pred,
    partC_pred,
    partD_pred,
    v1_14_ops,
    v1_14_profile,
    v1_14_variant,
):
    if partA_pred.value is not None and partB_pred.value is not None and partC_pred.value is not None and partD_pred.value is not None:
        ledger.save(chapter=14, design={
            "chapter": "v1_14",
            "track_id": v1_14_profile.track_id,
            "scenario_id": v1_14_variant.scenario_id,
            "hardware_ref": v1_14_ops.hardware_ref,
            "model_ref": v1_14_ops.model_ref,
            "completed": True,
            "drift_visibility_prediction": partA_pred.value,
            "retraining_cadence_prediction": partB_pred.value,
            "ops_policy_prediction": partC_pred.value,
            "debt_cascade_prediction": partD_pred.value,
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">14 &middot; ML Operations</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_14_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">MONITOR</span>
        <span class="hud-value">{v1_14_ops.alert_threshold_psi:.2f} PSI</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    debt_cascade,
    drift_visibility,
    mo,
    ops_policy,
    partA_days,
    partA_pred,
    partA_rate,
    partA_threshold,
    partB_current,
    partB_drift_cost,
    partB_pred,
    partB_retrain_cost,
    partC_cadence,
    partC_canary,
    partC_pred,
    partC_rollback,
    partC_threshold,
    partD_base_loss,
    partD_downstream,
    partD_missed,
    partD_pred,
    report_export_panel,
    retraining_cadence,
    v1_14_metadata,
    v1_14_ops,
    v1_14_profile,
    v1_14_variant,
):
    _drift = drift_visibility(
        v1_14_ops,
        days_since_deploy=partA_days.value,
        drift_rate_psi_per_day=partA_rate.value,
        alert_threshold_psi=partA_threshold.value,
    )
    _cadence = retraining_cadence(
        retrain_cost=partB_retrain_cost.value,
        drift_cost_per_day=partB_drift_cost.value,
        current_days=partB_current.value,
    )
    _policy = ops_policy(
        v1_14_ops,
        threshold_psi=partC_threshold.value,
        cadence_days=partC_cadence.value,
        canary_pct=partC_canary.value,
        rollback_hours=partC_rollback.value,
    )
    _debt = debt_cascade(
        missed_cycles=partD_missed.value,
        downstream_models=partD_downstream.value,
        base_loss_pp=partD_base_loss.value,
    )

    _incomplete = []
    if partA_pred.value is None:
        _incomplete.append("Part A drift visibility prediction")
    if partB_pred.value is None:
        _incomplete.append("Part B retraining cadence prediction")
    if partC_pred.value is None:
        _incomplete.append("Part C ops policy prediction")
    if partD_pred.value is None:
        _incomplete.append("Part D debt cascade prediction")

    _report = build_lab_report(
        v1_14_metadata,
        track=v1_14_profile.label,
        scenario=v1_14_variant.workload_summary,
        learning_objectives=(
            "Explain why model quality can degrade while infrastructure metrics stay green.",
            "Compute a retraining cadence from retraining cost and stale-model risk.",
            "Write an operations policy with monitoring, canary, rollback, escalation, and residual risk.",
        ),
        predictions={
            "drift_visibility": partA_pred.value,
            "retraining_cadence": partB_pred.value,
            "ops_policy": partC_pred.value,
            "debt_cascade": partD_pred.value,
        },
        knob_settings={
            "days_since_deploy": partA_days.value,
            "drift_rate_psi_per_day": partA_rate.value,
            "alert_threshold_psi": partA_threshold.value,
            "retrain_cost": partB_retrain_cost.value,
            "drift_cost_per_day": partB_drift_cost.value,
            "current_cadence_days": partB_current.value,
            "policy_threshold_psi": partC_threshold.value,
            "policy_cadence_days": partC_cadence.value,
            "policy_canary_pct": partC_canary.value,
            "policy_rollback_hours": partC_rollback.value,
            "missed_cycles": partD_missed.value,
            "downstream_models": partD_downstream.value,
            "base_loss_pp": partD_base_loss.value,
        },
        evidence_summary={
            "hardware_ref": v1_14_ops.hardware_ref,
            "model_ref": v1_14_ops.model_ref,
            "drift_source": v1_14_ops.drift_source,
            "monitoring_signal": v1_14_ops.monitoring_signal,
            "true_psi": round(_drift.true_psi, 4),
            "observed_psi": round(_drift.observed_psi, 4),
            "true_quality_pct": round(_drift.true_quality_pct, 3),
            "alert_day": _drift.alert_day,
            "optimal_cadence_days": round(_cadence.optimal_days, 3),
            "policy_feasible": _policy.feasible,
            "policy_violations": _policy.violations,
            "debt_multiplier": round(_debt.debt_multiplier, 3),
        },
        final_decision=(
            f"Adopt {v1_14_ops.monitoring_signal}; retrain near T*; use "
            f"{v1_14_ops.rollback_policy}; escalate through {v1_14_ops.escalation_policy}."
        ),
        big_takeaways=(
            "Infrastructure health and model health are separate axes.",
            "Delayed labels create a silent degradation window.",
            "Operations policy must include monitoring, cadence, canary, rollback, escalation, and residual blind spot.",
        ),
        reflections={
            "report_artifact": v1_14_ops.report_artifact,
            "validation_tests": v1_14_ops.validation_tests,
            "residual_blind_spot": "Proxy monitors and delayed labels can still miss abrupt regime changes.",
        },
        residual_risk=(
            "Teaching estimates must be validated with real production traces, label-delay audits, "
            "cohort canaries, rollback drills, and post-deployment quality reviews."
        ),
        source_trace={
            "track_id": v1_14_profile.track_id,
            "scenario_id": v1_14_variant.scenario_id,
            "hardware_ref": v1_14_variant.hardware_ref,
            "model_ref": v1_14_variant.model_ref,
            "shared_helper": "mlsysbook_labs.ops",
            "source_policy": v1_14_profile.source_policy,
        },
        result_snapshot={
            "ops_profile": v1_14_ops,
            "drift_visibility": _drift,
            "retraining_cadence": _cadence,
            "ops_policy": _policy,
            "debt_cascade": _debt,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-14 report is generated locally from the selected track, MLSysIM hardware/model refs, "
                "and shared `mlsysbook_labs.ops` calculations."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
