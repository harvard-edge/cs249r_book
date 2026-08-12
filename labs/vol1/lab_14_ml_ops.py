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
        track_arc_context,
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
        track_arc_context,
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
    track_arc_context,
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
                Deployed Behavior &middot; Thresholds &middot; Rollback &middot; Error Budget
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
        track_arc_context(v1_14_profile, v1_14_metadata.lab_id),
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
            "A) Uptime and latency are enough because the service is green": "infra",
            "B) Offline validation is enough until the next model release": "offline",
            "C) Deployed behavior signals must be monitored before labels arrive": "deployed",
            "D) Delayed labels are the only signal that matters": "labels",
        },
        label=f"{v1_14_ops.label}: which signal should own first detection when {v1_14_ops.drift_source} accumulates?",
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
            "A) Set the tightest possible threshold; earlier alerts are always safer": "tight",
            "B) Calibrate the threshold against false alarms and missed damage": "calibrate",
            "C) Set a loose threshold to protect the on-call rotation": "loose",
            "D) Wait for a user-facing incident before tuning thresholds": "reactive",
        },
        label=(
            f"{v1_14_ops.label}: how should ops choose the PSI alert threshold?"
        ),
    )
    return (partA_days, partA_rate, partA_threshold, partB_pred)


@app.cell(hide_code=True)
def _(mo, v1_14_ops):
    partB_threshold = mo.ui.slider(
        start=0.05,
        stop=0.50,
        value=v1_14_ops.alert_threshold_psi,
        step=0.01,
        label="Candidate threshold (PSI)",
    )
    partB_review_cost = mo.ui.slider(
        start=100,
        stop=5000,
        value=700,
        step=100,
        label="Alert review cost ($)",
    )
    partB_false_alarm_rate = mo.ui.slider(
        start=0.5,
        stop=20.0,
        value=4.0,
        step=0.5,
        label="False alarms at default (/year)",
    )

    partC_pred = mo.ui.radio(
        options={
            "A) Full rollout is acceptable because validation already passed": "full_rollout",
            "B) Small canary plus tested rollback and fallback limits blast radius": "staged",
            "C) Large canary is best because it gathers evidence fastest": "large_canary",
            "D) Rollback can wait for the next scheduled release window": "slow_rollback",
        },
        label=f"What release policy protects {v1_14_ops.guardrail_metric}?",
    )
    return (partB_false_alarm_rate, partB_review_cost, partB_threshold, partC_pred)


@app.cell(hide_code=True)
def _(mo, v1_14_ops):
    partC_canary = mo.ui.slider(start=0, stop=50, value=10, step=5, label="Canary traffic (%)")
    partC_rollback = mo.ui.slider(start=0.25, stop=72, value=8, step=0.25, label="Rollback exposure (hours)")
    partC_fallback = mo.ui.slider(start=0, stop=100, value=80, step=5, label="Fallback coverage (%)")

    partD_pred = mo.ui.radio(
        options={
            "A) Minimize monitoring cost; the cheapest policy wins": "cheap",
            "B) Spend error budget across detection, staleness, rollback, and ownership": "budget",
            "C) Automate retraining and remove human ownership": "auto",
            "D) Reuse one policy across every deployment track": "same",
        },
        label=f"What makes a defensible {v1_14_ops.label} operations policy?",
    )
    return (partC_canary, partC_fallback, partC_rollback, partD_pred)


@app.cell(hide_code=True)
def _(mo, v1_14_ops):
    partD_threshold = mo.ui.slider(
        start=0.05,
        stop=0.50,
        value=v1_14_ops.alert_threshold_psi,
        step=0.01,
        label="Runbook threshold (PSI)",
    )
    partD_cadence = mo.ui.slider(
        start=1,
        stop=120,
        value=v1_14_ops.current_cadence_days,
        step=1,
        label="Retraining cadence (days)",
    )
    partD_canary = mo.ui.slider(start=0, stop=50, value=10, step=5, label="Runbook canary (%)")
    partD_rollback = mo.ui.slider(start=0.25, stop=72, value=8, step=0.25, label="Runbook rollback (hours)")
    return (partD_cadence, partD_canary, partD_rollback, partD_threshold)


@app.cell
def _():
    def v1_14_track_amounts(v1_14_profile, v1_14_ops):
        defaults = {
            "iphone": {
                "unit_label": "million app sessions exposed",
                "daily_units": 24.0,
                "allowed_blast_radius": 1.5,
                "rollback_limit_hours": 12.0,
                "default_fallback_pct": 85.0,
                "fallback_label": "remote kill switch coverage",
                "blind_spot": "privacy sampling can miss cohort-specific thermal or battery regressions",
                "carry_forward_risk": "mobile owner must revalidate privacy-safe telemetry before Lab 15 responsibility review",
                "error_budget_days": 18.0,
                "attention_budget_hours": 24.0,
                "review_hours": 1.5,
                "impact_cost_per_unit": 1600.0,
            },
            "oura_ring": {
                "unit_label": "thousand device-nights exposed",
                "daily_units": 280.0,
                "allowed_blast_radius": 4.0,
                "rollback_limit_hours": 24.0,
                "default_fallback_pct": 80.0,
                "fallback_label": "OTA holdout and firmware fallback coverage",
                "blind_spot": "delayed health labels can miss physiology shifts until the next labeled study",
                "carry_forward_risk": "firmware owner must carry duty-cycle and false-alert risk into the next review",
                "error_budget_days": 30.0,
                "attention_budget_hours": 18.0,
                "review_hours": 2.0,
                "impact_cost_per_unit": 900.0,
            },
            "robotaxi": {
                "unit_label": "thousand autonomous miles exposed",
                "daily_units": 40.0,
                "allowed_blast_radius": 0.25,
                "rollback_limit_hours": 1.0,
                "default_fallback_pct": 95.0,
                "fallback_label": "geofenced safety fallback coverage",
                "blind_spot": "rare-event drift can hide inside aggregate replay pass rates",
                "carry_forward_risk": "safety board must own unresolved rare-event recall risk",
                "error_budget_days": 6.0,
                "attention_budget_hours": 40.0,
                "review_hours": 3.0,
                "impact_cost_per_unit": 60000.0,
            },
            "cloud_fleet": {
                "unit_label": "million requests exposed",
                "daily_units": 120.0,
                "allowed_blast_radius": 3.0,
                "rollback_limit_hours": 4.0,
                "default_fallback_pct": 80.0,
                "fallback_label": "registry pin and traffic failback coverage",
                "blind_spot": "aggregate SLO dashboards can hide one tenant or cohort regression",
                "carry_forward_risk": "SRE and ML owner must carry cost/request and tenant-quality risk forward",
                "error_budget_days": 10.0,
                "attention_budget_hours": 32.0,
                "review_hours": 1.0,
                "impact_cost_per_unit": 5000.0,
            },
        }
        amounts = dict(defaults.get(v1_14_profile.track_id, defaults["cloud_fleet"]))
        amounts["track_label"] = v1_14_ops.label
        amounts["guardrail_metric"] = v1_14_ops.guardrail_metric
        return amounts

    def v1_14_threshold_economics(
        v1_14_ops,
        amounts,
        *,
        threshold_psi,
        alert_review_cost,
        false_alarm_rate,
    ):
        threshold = max(0.0001, float(threshold_psi))
        default_threshold = max(0.0001, float(v1_14_ops.alert_threshold_psi))
        rate = max(0.00001, float(v1_14_ops.drift_rate_psi_per_day))
        breach_psi = max(
            0.0,
            (v1_14_ops.baseline_quality_pct - v1_14_ops.quality_floor_pct)
            / max(0.0001, v1_14_ops.quality_loss_per_psi),
        )
        quality_breach_day = breach_psi / rate if breach_psi > 0 else 0.0
        detection_day = threshold / rate + v1_14_ops.label_delay_days
        missed_days = max(0.0, detection_day - quality_breach_day)
        false_alarms = max(0.0, float(false_alarm_rate)) * (default_threshold / threshold) ** 1.35
        false_alarm_cost = false_alarms * max(0.0, float(alert_review_cost))
        attention_hours = false_alarms * amounts["review_hours"]
        missed_damage_cost = missed_days * v1_14_ops.drift_cost_per_day * 12
        monitoring_cost = v1_14_ops.monitoring_cost_per_day * 365 * (default_threshold / threshold) ** 0.25
        total_cost = false_alarm_cost + missed_damage_cost + monitoring_cost
        too_tight = attention_hours > amounts["attention_budget_hours"]
        too_loose = missed_days > max(v1_14_ops.label_delay_days, amounts["error_budget_days"] / 2)
        if too_loose:
            failure_mode = "missed degradation dominates"
        elif too_tight:
            failure_mode = "alert fatigue dominates"
        else:
            failure_mode = "balanced threshold"
        return {
            "threshold_psi": threshold,
            "detection_day": detection_day,
            "quality_breach_day": quality_breach_day,
            "missed_days": missed_days,
            "false_alarms_per_year": false_alarms,
            "attention_hours": attention_hours,
            "false_alarm_cost": false_alarm_cost,
            "missed_damage_cost": missed_damage_cost,
            "monitoring_cost": monitoring_cost,
            "total_cost": total_cost,
            "failure_mode": failure_mode,
            "feasible": not (too_tight or too_loose),
        }

    def v1_14_rollout_risk(
        amounts,
        *,
        canary_pct,
        rollback_hours,
        fallback_pct,
    ):
        canary = max(0.0, min(100.0, float(canary_pct)))
        rollback = max(0.0, float(rollback_hours))
        fallback = max(0.0, min(100.0, float(fallback_pct)))
        exposed_units = (
            amounts["daily_units"]
            * (canary / 100)
            * (rollback / 24)
            * (1 - fallback / 100)
        )
        blast_radius_cost = exposed_units * amounts["impact_cost_per_unit"]
        if rollback <= 1 / 60:
            rollback_tier = "immediate"
        elif rollback <= 0.25:
            rollback_tier = "rapid"
        elif rollback <= 4:
            rollback_tier = "delayed"
        else:
            rollback_tier = "extended"
        violations = []
        if exposed_units > amounts["allowed_blast_radius"]:
            violations.append("blast radius above track budget")
        if rollback > amounts["rollback_limit_hours"]:
            violations.append("rollback window above track limit")
        if fallback < amounts["default_fallback_pct"] * 0.75:
            violations.append("fallback coverage too small")
        return {
            "canary_pct": canary,
            "rollback_hours": rollback,
            "fallback_pct": fallback,
            "rollback_tier": rollback_tier,
            "exposed_units": exposed_units,
            "blast_radius_cost": blast_radius_cost,
            "feasible": not violations,
            "violations": tuple(violations),
        }

    def v1_14_error_budget(amounts, threshold_result, policy_result, rollout_result):
        rows = [
            ("Detection delay", max(0.0, threshold_result["missed_days"])),
            ("Stale model", max(0.0, policy_result.stale_days)),
            ("Rollback exposure", max(0.0, rollout_result["rollback_hours"] / 24)),
            ("Residual blind spot", max(0.0, amounts["error_budget_days"] * 0.15)),
        ]
        total_days = sum(value for _, value in rows)
        binding_risk = max(rows, key=lambda item: item[1])[0]
        feasible = (
            total_days <= amounts["error_budget_days"]
            and policy_result.feasible
            and rollout_result["feasible"]
        )
        return {
            "rows": rows,
            "total_days": total_days,
            "budget_days": amounts["error_budget_days"],
            "binding_risk": binding_risk,
            "feasible": feasible,
        }

    return (
        v1_14_error_budget,
        v1_14_rollout_risk,
        v1_14_threshold_economics,
        v1_14_track_amounts,
    )


# ===========================================================================
# ZONE C: MAIN LAB
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    drift_visibility,
    go,
    mo,
    np,
    ops_policy,
    partA_days,
    partA_pred,
    partA_rate,
    partA_threshold,
    partB_false_alarm_rate,
    partB_pred,
    partB_review_cost,
    partB_threshold,
    partC_canary,
    partC_fallback,
    partC_pred,
    partC_rollback,
    partD_cadence,
    partD_canary,
    partD_pred,
    partD_rollback,
    partD_threshold,
    retraining_cadence,
    v1_14_error_budget,
    v1_14_ops,
    v1_14_profile,
    v1_14_rollout_risk,
    v1_14_threshold_economics,
    v1_14_track_amounts,
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

    _amounts = v1_14_track_amounts(v1_14_profile, v1_14_ops)

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
## Concept Module A - Deployed Behavior Is The Monitor

Uptime, latency, and offline validation can stay green while deployed behavior
drifts. The monitor has to include production telemetry, delayed labels, and the
track guardrail that users actually experience.
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

        _signal_rows = [
            ("Infrastructure health", "uptime/latency green", "Necessary but cannot observe statistical drift."),
            ("Offline model metric", f"last validated at {v1_14_ops.baseline_quality_pct:.1f}%", "Stale once the deployed population moves."),
            ("Deployed behavior", v1_14_ops.monitoring_signal, f"Track guardrail: {v1_14_ops.guardrail_metric}."),
            ("Delayed labels", f"{v1_14_ops.label_delay_days} day delay", "Confirms drift after the proxy has already carried risk."),
        ]
        _rows_html = "".join(
            f"""
            <tr>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; font-weight:700;">{_name}</td>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0;">{_value}</td>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0;">{_meaning}</td>
            </tr>
            """
            for _name, _value, _meaning in _signal_rows
        )
        items.append(mo.Html(f"""
        <div style="background:white; border:1px solid {COLORS['Border']}; border-radius:10px;
                    padding:14px 16px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:800; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
                Evidence Table &middot; Signal Stack
            </div>
            <table style="width:100%; border-collapse:collapse; font-size:0.86rem; color:#334155;">
                <thead>
                    <tr style="background:#f8fafc;">
                        <th style="text-align:left; padding:8px 10px;">Signal</th>
                        <th style="text-align:left; padding:8px 10px;">Amount</th>
                        <th style="text-align:left; padding:8px 10px;">Operational meaning</th>
                    </tr>
                </thead>
                <tbody>{_rows_html}</tbody>
            </table>
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
**Math Peek / Source Model**

```
quality(t) ~= A0 - lambda * PSI(t)
detection delay = alert_day - quality_breach_day
```

*Source: `mlsysbook_labs.drift_visibility`; chapter sections on observable
degradation and model/infrastructure monitoring.*
        """))

        items.append(mo.callout(mo.md(
            f"**Checkpoint.** Carry `{v1_14_ops.monitoring_signal}` into the runbook as the deployed-behavior signal, "
            f"with `{_amounts['blind_spot']}` named as the residual blind spot."
        ), kind="info"))

        if partA_pred.value == "deployed":
            items.append(mo.callout(mo.md("**Correct.** Deployed behavior proxies are the early signal; delayed labels confirm the diagnosis later."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**The server can be healthy while the model is stale.** Model health needs deployed-behavior telemetry and delayed-label confirmation."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Alert Review &middot; {v1_14_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "The default PSI threshold is creating pages. If we loosen it, what
                    degradation will we miss?"
                </div>
            </div>
            """),
            mo.md(f"""
## Concept Module B - Thresholds Spend Attention

The chapter gives PSI > 0.2 as a useful starting point for feature-distribution
drift, but a threshold is an operating policy. Tight thresholds spend on-call
attention; loose thresholds spend quality, safety, battery, or SLO budget.

Track amount system: **{_amounts['unit_label']}**, attention budget
**{_amounts['attention_budget_hours']:.0f} review hours/year**.
            """),
            partB_pred,
        ]
        if partB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the threshold sweep."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_threshold, partB_review_cost, partB_false_alarm_rate], widths="equal"))
        _threshold = v1_14_threshold_economics(
            v1_14_ops,
            _amounts,
            threshold_psi=partB_threshold.value,
            alert_review_cost=partB_review_cost.value,
            false_alarm_rate=partB_false_alarm_rate.value,
        )
        _thresholds = np.linspace(0.05, 0.50, 120)
        _total_costs = []
        _false_costs = []
        _missed_costs = []
        for _candidate in _thresholds:
            _r = v1_14_threshold_economics(
                v1_14_ops,
                _amounts,
                threshold_psi=float(_candidate),
                alert_review_cost=partB_review_cost.value,
                false_alarm_rate=partB_false_alarm_rate.value,
            )
            _total_costs.append(_r["total_cost"])
            _false_costs.append(_r["false_alarm_cost"])
            _missed_costs.append(_r["missed_damage_cost"])

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_thresholds, y=_total_costs, name="Total threshold cost", line=dict(color=COLORS["RedLine"], width=3)))
        _fig.add_trace(go.Scatter(x=_thresholds, y=_false_costs, name="False-alarm cost", line=dict(color=COLORS["BlueLine"], width=2, dash="dot")))
        _fig.add_trace(go.Scatter(x=_thresholds, y=_missed_costs, name="Missed-damage cost", line=dict(color=COLORS["OrangeLine"], width=2, dash="dash")))
        _fig.add_vline(x=partB_threshold.value, line_dash="dot", line_color="#64748b", annotation_text=f"{partB_threshold.value:.2f} PSI")
        _fig.add_vline(x=v1_14_ops.alert_threshold_psi, line_dash="dash", line_color=COLORS["GreenLine"], annotation_text="track default")
        _fig.update_layout(
            height=360,
            xaxis=dict(title="PSI alert threshold"),
            yaxis=dict(title="Annualized cost ($)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=60, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _status_color = COLORS["GreenLine"] if _threshold["feasible"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Detection Day", f"{_threshold['detection_day']:.1f}", "threshold/rate + delay", COLORS["BlueLine"])}
            {_metric_card("False Alarms", f"{_threshold['false_alarms_per_year']:.1f}/yr", f"{_threshold['attention_hours']:.1f} review hr", COLORS["OrangeLine"])}
            {_metric_card("Missed Damage", f"${_threshold['missed_damage_cost']:,.0f}", f"{_threshold['missed_days']:.1f} missed days", COLORS["RedLine"])}
            {_metric_card("Threshold", _threshold["failure_mode"], f"{partB_threshold.value:.2f} PSI", _status_color, True)}
        </div>
        """))

        _scenario_values = [
            ("Tight", max(0.05, v1_14_ops.alert_threshold_psi * 0.5)),
            ("Track default", v1_14_ops.alert_threshold_psi),
            ("Loose", min(0.50, v1_14_ops.alert_threshold_psi * 1.75)),
        ]
        _scenario_rows = []
        for _name, _value in _scenario_values:
            _r = v1_14_threshold_economics(
                v1_14_ops,
                _amounts,
                threshold_psi=_value,
                alert_review_cost=partB_review_cost.value,
                false_alarm_rate=partB_false_alarm_rate.value,
            )
            _scenario_rows.append(f"""
            <tr>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; font-weight:700;">{_name}</td>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0;">{_value:.2f}</td>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0;">{_r['false_alarms_per_year']:.1f}/yr</td>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0;">{_r['missed_days']:.1f} d</td>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0;">{_r['failure_mode']}</td>
            </tr>
            """)
        items.append(mo.Html(f"""
        <div style="background:white; border:1px solid {COLORS['Border']}; border-radius:10px;
                    padding:14px 16px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:800; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
                Evidence Table &middot; Threshold Choices
            </div>
            <table style="width:100%; border-collapse:collapse; font-size:0.86rem; color:#334155;">
                <thead>
                    <tr style="background:#f8fafc;">
                        <th style="text-align:left; padding:8px 10px;">Policy</th>
                        <th style="text-align:left; padding:8px 10px;">PSI</th>
                        <th style="text-align:left; padding:8px 10px;">False alarms</th>
                        <th style="text-align:left; padding:8px 10px;">Missed days</th>
                        <th style="text-align:left; padding:8px 10px;">Failure mode</th>
                    </tr>
                </thead>
                <tbody>{''.join(_scenario_rows)}</tbody>
            </table>
        </div>
        """))

        if not _threshold["feasible"]:
            items.append(mo.callout(mo.md(
                f"**Reversible failure state.** This threshold is not defensible because `{_threshold['failure_mode']}`. "
                f"Move the PSI slider until both review attention and missed degradation fit the {v1_14_ops.label} budget."
            ), kind="danger"))

        items.append(mo.md(f"""
**Math Peek / Source Model**

```
detection_day = threshold / drift_rate + label_delay
false alarms  ~= base false alarms * (default_threshold / threshold)^1.35
total cost    = false alarm cost + missed damage cost + monitoring cost
```

*Source: chapter feature-distribution thresholds and monitoring cost model;
notebook-local `v1_14_threshold_economics`.*
        """))

        items.append(mo.callout(mo.md(
            f"**Checkpoint.** Carry `{partB_threshold.value:.2f} PSI` into the runbook only if it avoids both alert fatigue and missed degradation."
        ), kind="info"))

        if partB_pred.value == "calibrate":
            items.append(mo.callout(mo.md("**Correct.** A threshold is a policy that trades false alarms against missed degradation."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Thresholds are not universal constants.** Tune the PSI boundary against the track's attention budget and degradation cost."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Release Review &middot; {v1_14_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "The candidate model passed validation. How much production exposure
                    should it get before rollback has to fire?"
                </div>
            </div>
            """),
            mo.md(f"""
## Concept Module C - Rollback Limits Blast Radius

For **{v1_14_ops.label}**, rollback is:

```
{v1_14_ops.rollback_policy}
```

Canary traffic, rollback time, and fallback coverage define the amount of
production behavior at risk before recovery. The same concept appears as a
kill switch on iPhone, OTA rollback for Oura, geofenced fallback for RoboTaxi,
and registry-pinned traffic rollback for Cloud Fleet.
            """),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the rollout risk instrument."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_canary, partC_rollback, partC_fallback], widths="equal"))
        _rollout = v1_14_rollout_risk(
            _amounts,
            canary_pct=partC_canary.value,
            rollback_hours=partC_rollback.value,
            fallback_pct=partC_fallback.value,
        )
        _hours = np.linspace(0.25, 72, 120)
        _exposed = [
            v1_14_rollout_risk(
                _amounts,
                canary_pct=partC_canary.value,
                rollback_hours=float(_hour),
                fallback_pct=partC_fallback.value,
            )["exposed_units"]
            for _hour in _hours
        ]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_hours, y=_exposed, name="Blast radius", line=dict(color=COLORS["RedLine"], width=3)))
        _fig.add_hline(y=_amounts["allowed_blast_radius"], line_dash="dash", line_color=COLORS["GreenLine"], annotation_text="track budget")
        _fig.add_vline(x=partC_rollback.value, line_dash="dot", line_color="#64748b", annotation_text=f"{partC_rollback.value:g} h")
        _fig.update_layout(
            height=350,
            xaxis=dict(title="Rollback exposure window (hours)"),
            yaxis=dict(title=_amounts["unit_label"], gridcolor="#f1f5f9"),
            margin=dict(l=70, r=20, t=50, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _rollout_color = COLORS["GreenLine"] if _rollout["feasible"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Canary", f"{_rollout['canary_pct']:.0f}%", "traffic under test", COLORS["BlueLine"])}
            {_metric_card("Rollback Tier", _rollout["rollback_tier"], f"{_rollout['rollback_hours']:g} hours", COLORS["OrangeLine"])}
            {_metric_card("Blast Radius", f"{_rollout['exposed_units']:.2f}", _amounts["unit_label"], COLORS["RedLine"])}
            {_metric_card("Release Status", "PASS" if _rollout["feasible"] else "FAIL", ", ".join(_rollout["violations"]) or "inside budget", _rollout_color, True)}
        </div>
        """))

        _tier_rows = [
            ("Immediate", "&lt; 1 minute", "hot standby / instant traffic switch"),
            ("Rapid", "&lt; 15 minutes", "registry redeploy, cache clear, session restart"),
            ("Delayed", "&lt; 4 hours", "business metric rollback with state handling"),
            ("Extended", "&gt; 4 hours", "exposure grows faster than the control loop can recover"),
        ]
        _tier_rows_html = "".join(
            f"""
            <tr>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; font-weight:700;">{_tier}</td>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0;">{_target}</td>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0;">{_meaning}</td>
            </tr>
            """
            for _tier, _target, _meaning in _tier_rows
        )
        items.append(mo.Html(f"""
        <div style="background:white; border:1px solid {COLORS['Border']}; border-radius:10px;
                    padding:14px 16px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:800; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
                Evidence Table &middot; Rollback Tiers
            </div>
            <table style="width:100%; border-collapse:collapse; font-size:0.86rem; color:#334155;">
                <thead>
                    <tr style="background:#f8fafc;">
                        <th style="text-align:left; padding:8px 10px;">Tier</th>
                        <th style="text-align:left; padding:8px 10px;">Recovery target</th>
                        <th style="text-align:left; padding:8px 10px;">State handling</th>
                    </tr>
                </thead>
                <tbody>{_tier_rows_html}</tbody>
            </table>
        </div>
        """))

        if not _rollout["feasible"]:
            items.append(mo.callout(mo.md(
                "**Reversible failure state.** " + ", ".join(_rollout["violations"]) +
                ". Reduce canary traffic, shorten rollback exposure, or increase fallback coverage."
            ), kind="danger"))

        items.append(mo.md(f"""
**Math Peek / Source Model**

```
blast radius = daily units * canary share * rollback hours / 24 * unprotected share
             = {_amounts['daily_units']:.1f} * {partC_canary.value / 100:.2f} * {partC_rollback.value / 24:.3f} * {(1 - partC_fallback.value / 100):.2f}
             = {_rollout['exposed_units']:.2f} {_amounts['unit_label']}
```

*Source: chapter rollback strategy table and staged deployment discussion;
notebook-local `v1_14_rollout_risk`.*
        """))

        items.append(mo.callout(mo.md(
            f"**Checkpoint.** Runbook rollback rule: `{v1_14_ops.rollback_policy}` with "
            f"{partC_canary.value}% canary, {partC_rollback.value:g} hour rollback exposure, and "
            f"{partC_fallback.value}% {_amounts['fallback_label']}."
        ), kind="info"))

        if partC_pred.value == "staged" and _rollout["feasible"]:
            items.append(mo.callout(mo.md("**Correct.** Rollout size only becomes safe when rollback and fallback bound the exposed behavior."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**A green aggregate canary is not the policy.** Blast radius and recovery time decide whether the rollout is operationally safe."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_d():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Runbook Review &middot; {v1_14_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "You have one policy memo. Which detection, retraining, canary,
                    rollback, and ownership choices fit the error budget?"
                </div>
            </div>
            """),
            mo.md(f"""
## Concept Module D - Error Budget Is A Policy Choice

The final policy spends an explicit error budget. Detection delay spends days
before the alert fires. Slow retraining spends stale-model exposure. Rollback
spends blast-radius exposure. Residual blind spots spend owner attention.

Track budget: **{_amounts['error_budget_days']:.1f} equivalent days** for
{v1_14_ops.guardrail_metric}.
            """),
            partD_pred,
        ]
        if partD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the policy ledger."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_threshold, partD_cadence, partD_canary, partD_rollback], widths="equal"))
        _threshold = v1_14_threshold_economics(
            v1_14_ops,
            _amounts,
            threshold_psi=partD_threshold.value,
            alert_review_cost=partB_review_cost.value,
            false_alarm_rate=partB_false_alarm_rate.value,
        )
        _policy = ops_policy(
            v1_14_ops,
            threshold_psi=partD_threshold.value,
            cadence_days=partD_cadence.value,
            canary_pct=partD_canary.value,
            rollback_hours=partD_rollback.value,
        )
        _rollout = v1_14_rollout_risk(
            _amounts,
            canary_pct=partD_canary.value,
            rollback_hours=partD_rollback.value,
            fallback_pct=_amounts["default_fallback_pct"],
        )
        _budget = v1_14_error_budget(_amounts, _threshold, _policy, _rollout)
        _cadence = retraining_cadence(
            retrain_cost=v1_14_ops.retrain_cost,
            drift_cost_per_day=v1_14_ops.drift_cost_per_day,
            current_days=partD_cadence.value,
        )

        _fig = go.Figure()
        _colors = [COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["RedLine"], COLORS["GreenLine"]]
        for (_name, _value), _color in zip(_budget["rows"], _colors):
            _fig.add_trace(go.Bar(x=["Error budget"], y=[_value], name=_name, marker_color=_color))
        _fig.add_trace(go.Scatter(
            x=["Error budget"],
            y=[_budget["budget_days"]],
            mode="markers+text",
            name="Budget limit",
            text=[f"budget {_budget['budget_days']:.1f} d"],
            textposition="top center",
            marker=dict(color="#0f172a", size=11, symbol="line-ew"),
        ))
        _fig.update_layout(
            barmode="stack",
            height=350,
            yaxis=dict(title="Equivalent error-budget days", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=70, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _budget_color = COLORS["GreenLine"] if _budget["feasible"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("T*", f"{_cadence.optimal_days:.1f} d", "economic cadence", COLORS["GreenLine"])}
            {_metric_card("Budget Used", f"{_budget['total_days']:.1f} d", f"limit {_budget['budget_days']:.1f} d", COLORS["RedLine"])}
            {_metric_card("Binding Risk", _budget["binding_risk"], "largest budget term", COLORS["OrangeLine"])}
            {_metric_card("Policy Status", "PASS" if _budget["feasible"] else "FAIL", ", ".join(_policy.violations + _rollout["violations"]) or "inside budget", _budget_color, True)}
        </div>
        """))

        _budget_rows_html = "".join(
            f"""
            <tr>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; font-weight:700;">{_name}</td>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0;">{_value:.2f} d</td>
                <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0;">{_value / max(0.0001, _budget['budget_days']) * 100:.0f}%</td>
            </tr>
            """
            for _name, _value in _budget["rows"]
        )
        items.append(mo.Html(f"""
        <div style="background:white; border:1px solid {COLORS['Border']}; border-radius:10px;
                    padding:14px 16px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:800; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
                Evidence Table &middot; Error-Budget Ledger
            </div>
            <table style="width:100%; border-collapse:collapse; font-size:0.86rem; color:#334155;">
                <thead>
                    <tr style="background:#f8fafc;">
                        <th style="text-align:left; padding:8px 10px;">Budget term</th>
                        <th style="text-align:left; padding:8px 10px;">Equivalent days</th>
                        <th style="text-align:left; padding:8px 10px;">Budget share</th>
                    </tr>
                </thead>
                <tbody>{_budget_rows_html}</tbody>
            </table>
        </div>
        """))

        if not _budget["feasible"]:
            _violations = ", ".join(_policy.violations + _rollout["violations"]) or "error budget overspent"
            items.append(mo.callout(mo.md(
                f"**Reversible failure state.** This runbook overspends the {_amounts['error_budget_days']:.1f}-day track budget: {_violations}."
            ), kind="danger"))

        items.append(mo.md(f"""
**Math Peek / Source Model**

```
T* = sqrt(2 * retrain_cost / drift_cost_per_day)
   = sqrt(2 * {v1_14_ops.retrain_cost:,.0f} / {v1_14_ops.drift_cost_per_day:,.0f})
   = {_cadence.optimal_days:.1f} days

monitoring cost = C_ingest + C_storage + C_compute + C_alert
policy cost     = ${_policy.total_annual_cost:,.0f}/year
```

*Source: chapter cost-aware automation, monitoring cost model, and on-call
practice sections; shared `mlsysbook_labs.ops_policy` and notebook-local
`v1_14_error_budget`.*
        """))

        items.append(mo.callout(mo.md(
            f"**Checkpoint.** Final runbook: alert at `{partD_threshold.value:.2f} PSI`, retrain every "
            f"`{partD_cadence.value} days`, canary `{partD_canary.value}%`, rollback within "
            f"`{partD_rollback.value:g} hours`, and carry `{_amounts['carry_forward_risk']}`."
        ), kind="info"))

        if partD_pred.value == "budget" and _budget["feasible"]:
            items.append(mo.callout(mo.md("**Correct.** A defensible policy spends the error budget deliberately and names who owns the residual risk."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Operations policy is not just cost minimization.** It must allocate detection, stale-model, rollout, rollback, and ownership risk under the track budget."
            ), kind="warn"))
        return mo.vstack(items)

    def build_synthesis():
        _threshold = v1_14_threshold_economics(
            v1_14_ops,
            _amounts,
            threshold_psi=partD_threshold.value,
            alert_review_cost=partB_review_cost.value,
            false_alarm_rate=partB_false_alarm_rate.value,
        )
        _policy = ops_policy(
            v1_14_ops,
            threshold_psi=partD_threshold.value,
            cadence_days=partD_cadence.value,
            canary_pct=partD_canary.value,
            rollback_hours=partD_rollback.value,
        )
        _rollout = v1_14_rollout_risk(
            _amounts,
            canary_pct=partD_canary.value,
            rollback_hours=partD_rollback.value,
            fallback_pct=_amounts["default_fallback_pct"],
        )
        _budget = v1_14_error_budget(_amounts, _threshold, _policy, _rollout)
        _status = "approved" if _budget["feasible"] else "not yet approved"
        return mo.vstack([
            mo.md("## Synthesis - Operations Runbook Memo"),
            mo.callout(mo.md(
                f"**Chapter invariant.** Production ML is a control loop: monitor deployed behavior, "
                f"calibrate the alert threshold, limit rollout blast radius, and spend error budget deliberately."
            ), kind="info"),
            mo.callout(mo.md(
                f"**Runbook status: {_status}.** Alert at `{partD_threshold.value:.2f} PSI`; retrain every "
                f"`{partD_cadence.value} days`; canary `{partD_canary.value}%`; rollback within "
                f"`{partD_rollback.value:g} hours`; expected budget use `{_budget['total_days']:.1f}` of "
                f"`{_budget['budget_days']:.1f}` equivalent days."
            ), kind="info"),
            mo.callout(mo.md(
                f"**Residual blind spot.** {_amounts['blind_spot']} "
                f"Carry-forward responsibility risk: {_amounts['carry_forward_risk']}."
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
                        policy, the next question is whose outcomes and constraints are protected
                        by the remaining blind spot.
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
                        Submit an operations runbook memo for {v1_14_ops.label} with alert
                        threshold, rollback rule, residual blind spot, and carry-forward owner risk.
                    </div>
                </div>
            </div>
            """),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Deployed Behavior": build_part_a(),
        "Part B: Threshold Trade-off": build_part_b(),
        "Part C: Rollout & Rollback": build_part_c(),
        "Part D: Error Budget Policy": build_part_d(),
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
    partB_threshold,
    partC_canary,
    partC_fallback,
    partC_rollback,
    partD_cadence,
    partD_canary,
    partA_pred,
    partB_pred,
    partC_pred,
    partD_pred,
    partD_rollback,
    partD_threshold,
    v1_14_ops,
    v1_14_profile,
    v1_14_track_amounts,
    v1_14_variant,
):
    _amounts = v1_14_track_amounts(v1_14_profile, v1_14_ops)
    if partA_pred.value is not None and partB_pred.value is not None and partC_pred.value is not None and partD_pred.value is not None:
        ledger.save(chapter=14, design={
            "chapter": "v1_14",
            "track_id": v1_14_profile.track_id,
            "scenario_id": v1_14_variant.scenario_id,
            "hardware_ref": v1_14_ops.hardware_ref,
            "model_ref": v1_14_ops.model_ref,
            "completed": True,
            "deployed_behavior_prediction": partA_pred.value,
            "threshold_tradeoff_prediction": partB_pred.value,
            "rollout_rollback_prediction": partC_pred.value,
            "error_budget_prediction": partD_pred.value,
            "part_b_threshold_psi": partB_threshold.value,
            "part_c_canary_pct": partC_canary.value,
            "part_c_rollback_hours": partC_rollback.value,
            "part_c_fallback_pct": partC_fallback.value,
            "runbook_threshold_psi": partD_threshold.value,
            "runbook_retraining_cadence_days": partD_cadence.value,
            "runbook_canary_pct": partD_canary.value,
            "runbook_rollback_hours": partD_rollback.value,
            "rollback_rule": v1_14_ops.rollback_policy,
            "residual_blind_spot": _amounts["blind_spot"],
            "carry_forward_responsibility_risk": _amounts["carry_forward_risk"],
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">14 &middot; ML Operations</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_14_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">MONITOR</span>
        <span class="hud-value">{partD_threshold.value:.2f} PSI</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    drift_visibility,
    mo,
    ops_policy,
    partA_days,
    partA_pred,
    partA_rate,
    partA_threshold,
    partB_false_alarm_rate,
    partB_pred,
    partB_review_cost,
    partB_threshold,
    partC_canary,
    partC_fallback,
    partC_pred,
    partC_rollback,
    partD_cadence,
    partD_canary,
    partD_pred,
    partD_rollback,
    partD_threshold,
    report_export_panel,
    retraining_cadence,
    v1_14_error_budget,
    v1_14_metadata,
    v1_14_ops,
    v1_14_profile,
    v1_14_rollout_risk,
    v1_14_threshold_economics,
    v1_14_track_amounts,
    v1_14_variant,
):
    _amounts = v1_14_track_amounts(v1_14_profile, v1_14_ops)
    _drift = drift_visibility(
        v1_14_ops,
        days_since_deploy=partA_days.value,
        drift_rate_psi_per_day=partA_rate.value,
        alert_threshold_psi=partA_threshold.value,
    )
    _threshold = v1_14_threshold_economics(
        v1_14_ops,
        _amounts,
        threshold_psi=partB_threshold.value,
        alert_review_cost=partB_review_cost.value,
        false_alarm_rate=partB_false_alarm_rate.value,
    )
    _runbook_threshold = v1_14_threshold_economics(
        v1_14_ops,
        _amounts,
        threshold_psi=partD_threshold.value,
        alert_review_cost=partB_review_cost.value,
        false_alarm_rate=partB_false_alarm_rate.value,
    )
    _rollout = v1_14_rollout_risk(
        _amounts,
        canary_pct=partC_canary.value,
        rollback_hours=partC_rollback.value,
        fallback_pct=partC_fallback.value,
    )
    _runbook_rollout = v1_14_rollout_risk(
        _amounts,
        canary_pct=partD_canary.value,
        rollback_hours=partD_rollback.value,
        fallback_pct=_amounts["default_fallback_pct"],
    )
    _cadence = retraining_cadence(
        retrain_cost=v1_14_ops.retrain_cost,
        drift_cost_per_day=v1_14_ops.drift_cost_per_day,
        current_days=partD_cadence.value,
    )
    _policy = ops_policy(
        v1_14_ops,
        threshold_psi=partD_threshold.value,
        cadence_days=partD_cadence.value,
        canary_pct=partD_canary.value,
        rollback_hours=partD_rollback.value,
    )
    _budget = v1_14_error_budget(_amounts, _runbook_threshold, _policy, _runbook_rollout)

    _incomplete = []
    if partA_pred.value is None:
        _incomplete.append("Part A deployed-behavior prediction")
    if partB_pred.value is None:
        _incomplete.append("Part B threshold trade-off prediction")
    if partC_pred.value is None:
        _incomplete.append("Part C rollout/rollback prediction")
    if partD_pred.value is None:
        _incomplete.append("Part D error-budget prediction")

    _report = build_lab_report(
        v1_14_metadata,
        track=v1_14_profile.label,
        scenario=v1_14_variant.workload_summary,
        learning_objectives=(
            "Explain why monitoring must measure deployed behavior, not only model or infrastructure metrics.",
            "Calibrate a drift threshold by trading false alarms against missed degradation.",
            "Limit release blast radius with canary, rollback, and fallback policy.",
            "Spend error budget deliberately in an operations runbook memo.",
        ),
        predictions={
            "deployed_behavior": partA_pred.value,
            "threshold_tradeoff": partB_pred.value,
            "rollout_rollback": partC_pred.value,
            "error_budget_policy": partD_pred.value,
        },
        knob_settings={
            "days_since_deploy": partA_days.value,
            "drift_rate_psi_per_day": partA_rate.value,
            "part_a_alert_threshold_psi": partA_threshold.value,
            "part_b_threshold_psi": partB_threshold.value,
            "alert_review_cost": partB_review_cost.value,
            "false_alarm_rate_per_year": partB_false_alarm_rate.value,
            "part_c_canary_pct": partC_canary.value,
            "part_c_rollback_hours": partC_rollback.value,
            "part_c_fallback_pct": partC_fallback.value,
            "runbook_threshold_psi": partD_threshold.value,
            "runbook_cadence_days": partD_cadence.value,
            "runbook_canary_pct": partD_canary.value,
            "runbook_rollback_hours": partD_rollback.value,
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
            "part_a_detection_delay_days": _drift.detection_delay_days,
            "part_b_threshold_psi": round(_threshold["threshold_psi"], 3),
            "part_b_false_alarm_cost": round(_threshold["false_alarm_cost"], 2),
            "part_b_missed_damage_cost": round(_threshold["missed_damage_cost"], 2),
            "part_b_failure_mode": _threshold["failure_mode"],
            "part_c_blast_radius_units": round(_rollout["exposed_units"], 4),
            "part_c_unit_label": _amounts["unit_label"],
            "part_c_rollback_tier": _rollout["rollback_tier"],
            "optimal_cadence_days": round(_cadence.optimal_days, 3),
            "runbook_policy_feasible": _budget["feasible"],
            "error_budget_days": round(_budget["total_days"], 3),
            "binding_risk": _budget["binding_risk"],
            "policy_violations": _policy.violations,
            "rollout_violations": _runbook_rollout["violations"],
            "residual_blind_spot": _amounts["blind_spot"],
            "carry_forward_responsibility_risk": _amounts["carry_forward_risk"],
        },
        final_decision=(
            f"Runbook memo: alert at {partD_threshold.value:.2f} PSI; retrain every "
            f"{partD_cadence.value} days; canary {partD_canary.value}% with rollback within "
            f"{partD_rollback.value:g} hours using {v1_14_ops.rollback_policy}; escalate through "
            f"{v1_14_ops.escalation_policy}."
        ),
        big_takeaways=(
            "Monitoring must measure deployed behavior, not only model or infrastructure metrics.",
            "Drift thresholds allocate operational attention and missed-degradation risk.",
            "Rollout and rollback policy control blast radius and recovery time.",
            "A runbook spends error budget deliberately and names residual ownership risk.",
        ),
        reflections={
            "report_artifact": v1_14_ops.report_artifact,
            "validation_tests": v1_14_ops.validation_tests,
            "rollback_rule": v1_14_ops.rollback_policy,
            "residual_blind_spot": _amounts["blind_spot"],
            "carry_forward_responsibility_risk": _amounts["carry_forward_risk"],
        },
        residual_risk=(
            "Teaching estimates must be validated with real production traces, label-delay audits, "
            "cohort canaries, rollback drills, post-deployment quality reviews, and owner handoff checks."
        ),
        source_trace={
            "track_id": v1_14_profile.track_id,
            "scenario_id": v1_14_variant.scenario_id,
            "hardware_ref": v1_14_variant.hardware_ref,
            "model_ref": v1_14_variant.model_ref,
            "shared_helper": "mlsysbook_labs.ops",
            "notebook_local_helpers": (
                "v1_14_track_amounts",
                "v1_14_threshold_economics",
                "v1_14_rollout_risk",
                "v1_14_error_budget",
            ),
            "source_policy": v1_14_profile.source_policy,
        },
        result_snapshot={
            "ops_profile": v1_14_ops,
            "drift_visibility": _drift,
            "threshold_economics": _threshold,
            "rollout_risk": _rollout,
            "retraining_cadence": _cadence,
            "ops_policy": _policy,
            "error_budget": _budget,
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
