import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# -----------------------------------------------------------------------------
# LAB V2-12: OPERATIONS AT SCALE AS CONTROL LOOPS
#
# Chapter invariant: operations at scale are control loops. SLOs, canaries,
# rollouts, incidents, and blast radius spend error budget over time.
#
# Packet modules:
#   Part A - SLO / error budget as an amount system
#   Part B - Canary learning speed versus blast radius
#   Part C - Incident recovery time and lost-work budgeting
#   Part D - Operations policy guardrail conjunction
#   Synthesis - Operations-at-scale memo and V2-13 security implication
# -----------------------------------------------------------------------------


@app.cell
async def _():
    import html as html_lib
    import math
    import sys
    from pathlib import Path

    import marimo as mo

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

    import pandas as pd
    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        source_trace,
        track_arc_context,
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
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        html_lib,
        ledger,
        math,
        mo,
        pd,
        report_export_panel,
        source_trace,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v2_12_lab_path = "vol2/lab_12_ops_scale.py"
    v2_12_chapter = 12
    v2_12_metadata = get_lab_metadata(v2_12_lab_path)
    return v2_12_chapter, v2_12_lab_path, v2_12_metadata


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_12_track_picker = track_selector(default=_default_track)
    v2_12_track_picker
    return (v2_12_track_picker,)


@app.cell
def _(get_lab_track_variant, get_track_profile, v2_12_metadata, v2_12_track_picker):
    v2_12_track_id = v2_12_track_picker.value
    v2_12_profile = get_track_profile(v2_12_track_id)
    v2_12_variant = get_lab_track_variant(v2_12_metadata.lab_id, v2_12_track_id)
    return v2_12_profile, v2_12_track_id, v2_12_variant


@app.cell
def _(html_lib, math):
    def v2_12_escape(value):
        return html_lib.escape(str(value))

    def v2_12_track_packet(profile, variant):
        base = {
            "track_id": profile.track_id,
            "label": profile.label,
            "stakeholder": variant.stakeholder,
            "scenario": variant.workload_summary,
            "mission": profile.narrative,
            "hardware_ref": variant.hardware_ref,
            "model_ref": variant.model_ref,
            "period_days": 30,
            "release_window_h": 4.0,
            "slo_default_pct": 99.9,
            "availability_floor_pct": 99.0,
            "baseline_quality_pct": 94.0,
            "quality_floor_pct": 91.0,
            "quality_loss_per_incident_pp": 0.6,
            "quality_drift_pp_per_h": 0.02,
            "incident_budget_count": 3,
            "incident_default_count": 3,
            "detection_default_min": 25,
            "impact_default_min": 40,
            "traffic_per_h": 1_000_000.0,
            "traffic_unit": "requests",
            "sample_default": 10_000,
            "canary_default_pct": 5,
            "canary_default_h": 2.0,
            "blast_budget_units": 400_000.0,
            "affected_units_per_min": 8_000.0,
            "incident_impact_fraction": 0.08,
            "response_budget_units": 110_000.0,
            "mttd_default_min": 20,
            "diagnosis_default_min": 25,
            "mitigation_default_pct": 60,
            "recovery_default_min": 50,
            "runbook_default_label": "Typed diagnostic runbook",
            "ops_unit": "service traffic",
            "release_unit": "traffic slice",
            "primary_signal": "online metrics, logs, canary quality, delayed labels, and SLO/cost alerts",
            "quality_signal": "quality and SLA regression",
            "failure_mode": "a fast rollout spends too much error budget before rollback completes",
            "report_frame": "Platform operations policy memo",
            "slo_guardrail_limit_pct": 100.0,
            "max_blast_pct": 30.0,
            "cost_limit_index": 125.0,
            "governance_min_score": 2.5,
            "v2_13_options": {
                "telemetry_minimization": "Telemetry minimization can hide attacks unless detection signals are explicitly preserved.",
                "exploit_blast_radius": "Rollout blast-radius limits also bound exploit exposure during a security event.",
                "audit_trail": "Governance review creates the audit trail V2-13 needs for access, model, and data changes.",
                "incident_escalation": "Incident evidence must route security and privacy anomalies to the right escalation path.",
            },
        }
        overrides = {
            "iphone": {
                "slo_default_pct": 99.5,
                "baseline_quality_pct": 92.0,
                "quality_floor_pct": 90.0,
                "quality_loss_per_incident_pp": 0.45,
                "quality_drift_pp_per_h": 0.015,
                "incident_budget_count": 4,
                "incident_default_count": 4,
                "detection_default_min": 45,
                "impact_default_min": 35,
                "traffic_per_h": 200_000.0,
                "traffic_unit": "opt-in sessions",
                "sample_default": 15_000,
                "canary_default_pct": 5,
                "canary_default_h": 2.0,
                "blast_budget_units": 25_000.0,
                "affected_units_per_min": 1_600.0,
                "incident_impact_fraction": 0.06,
                "response_budget_units": 12_000.0,
                "mttd_default_min": 35,
                "diagnosis_default_min": 30,
                "mitigation_default_pct": 55,
                "recovery_default_min": 80,
                "ops_unit": "app/model cohort",
                "release_unit": "device and OS cohort",
                "primary_signal": "privacy-safe opt-in telemetry, crash-free sessions, and on-device proxy quality",
                "quality_signal": "privacy-safe quality regression",
                "failure_mode": "a quality regression reaches broad app rollout before opt-in telemetry has enough evidence",
                "report_frame": "Mobile rollout policy memo",
                "max_blast_pct": 20.0,
                "cost_limit_index": 115.0,
                "governance_min_score": 2.0,
            },
            "oura_ring": {
                "slo_default_pct": 99.0,
                "baseline_quality_pct": 90.0,
                "quality_floor_pct": 88.0,
                "quality_loss_per_incident_pp": 0.35,
                "quality_drift_pp_per_h": 0.010,
                "incident_budget_count": 2,
                "incident_default_count": 2,
                "detection_default_min": 90,
                "impact_default_min": 60,
                "traffic_per_h": 30_000.0,
                "traffic_unit": "device syncs",
                "sample_default": 2_500,
                "canary_default_pct": 5,
                "canary_default_h": 2.0,
                "blast_budget_units": 2_500.0,
                "affected_units_per_min": 350.0,
                "incident_impact_fraction": 0.10,
                "response_budget_units": 6_000.0,
                "mttd_default_min": 80,
                "diagnosis_default_min": 45,
                "mitigation_default_pct": 45,
                "recovery_default_min": 120,
                "runbook_default_label": "Typed diagnostic runbook",
                "ops_unit": "firmware/model OTA cohort",
                "release_unit": "wearable OTA cohort",
                "primary_signal": "sensor-quality indicators, battery anomalies, and delayed health-adjacent labels",
                "quality_signal": "sensing and false-alert regression",
                "failure_mode": "firmware rollout damages sensing or battery while labels arrive too late",
                "report_frame": "Wearable OTA operations memo",
                "max_blast_pct": 10.0,
                "cost_limit_index": 110.0,
                "governance_min_score": 2.0,
            },
            "robotaxi": {
                "slo_default_pct": 99.95,
                "baseline_quality_pct": 97.0,
                "quality_floor_pct": 96.3,
                "quality_loss_per_incident_pp": 0.20,
                "quality_drift_pp_per_h": 0.008,
                "incident_budget_count": 1,
                "incident_default_count": 1,
                "detection_default_min": 6,
                "impact_default_min": 12,
                "traffic_per_h": 1_200.0,
                "traffic_unit": "live miles",
                "sample_default": 3_000,
                "canary_default_pct": 5,
                "canary_default_h": 2.0,
                "blast_budget_units": 120.0,
                "affected_units_per_min": 40.0,
                "incident_impact_fraction": 0.30,
                "response_budget_units": 1_000.0,
                "mttd_default_min": 5,
                "diagnosis_default_min": 12,
                "mitigation_default_pct": 80,
                "recovery_default_min": 35,
                "runbook_default_label": "Drilled automated playbook",
                "ops_unit": "geofenced vehicle cohort",
                "release_unit": "vehicle and geofence slice",
                "primary_signal": "near-miss telemetry, disengagements, simulation replay, and sensor-health monitors",
                "quality_signal": "rare-event recall and safety margin regression",
                "failure_mode": "a geofence expansion exposes too many live miles before safety evidence is sufficient",
                "report_frame": "Safety rollout control memo",
                "max_blast_pct": 5.0,
                "cost_limit_index": 135.0,
                "governance_min_score": 4.0,
            },
            "cloud_fleet": {},
        }
        packet = dict(base)
        packet.update(overrides.get(profile.track_id, {}))
        packet["source_policy"] = profile.source_policy
        return packet

    def v2_12_period_minutes(days):
        return days * 24 * 60

    def v2_12_fmt_number(value, digits=1):
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        value = float(value)
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.{digits}f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.{digits}f}K"
        if abs(value) >= 100:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def v2_12_fmt_minutes(value):
        value = float(value)
        if value >= 24 * 60:
            return f"{value / (24 * 60):.1f} days"
        if value >= 60:
            return f"{value / 60:.1f} h"
        return f"{value:.0f} min"

    def v2_12_fmt_hours(value):
        value = float(value)
        if value >= 24:
            return f"{value / 24:.1f} days"
        if value >= 1:
            return f"{value:.2f} h"
        return f"{value * 60:.0f} min"

    def v2_12_fmt_pct(value, digits=1):
        return f"{float(value):.{digits}f}%"

    def v2_12_guardrail_badge(ok):
        return "PASS" if ok else "FAIL"

    def v2_12_prediction_feedback(predicted, actual, labels):
        if predicted is None:
            return ("warn", "Commit to the structured prediction before treating the instrument as evidence.")
        if predicted == actual:
            return ("success", f"Prediction check: correct. The measured result is `{labels.get(actual, actual)}`.")
        return ("warn", f"Prediction check: the instrument found `{labels.get(actual, actual)}`, not `{labels.get(predicted, predicted)}`.")

    return (
        v2_12_escape,
        v2_12_fmt_hours,
        v2_12_fmt_minutes,
        v2_12_fmt_number,
        v2_12_fmt_pct,
        v2_12_guardrail_badge,
        v2_12_period_minutes,
        v2_12_prediction_feedback,
        v2_12_track_packet,
    )


@app.cell
def _(v2_12_profile, v2_12_track_packet, v2_12_variant):
    v2_12_packet = v2_12_track_packet(v2_12_profile, v2_12_variant)
    return (v2_12_packet,)


@app.cell
def _(mo, v2_12_packet):
    v2_12_partA_pred = mo.ui.radio(
        options={
            "Availability error-budget minutes will bind first.": "availability",
            "Quality/drift points will bind first.": "quality",
            "Incident count will bind first.": "incident_count",
        },
        label="Part A prediction",
    )
    v2_12_slo_pct = mo.ui.slider(
        start=float(v2_12_packet["availability_floor_pct"]),
        stop=99.99,
        step=0.01,
        value=float(v2_12_packet["slo_default_pct"]),
        label="Availability SLO (%)",
    )
    v2_12_quality_floor = mo.ui.slider(
        start=max(80.0, float(v2_12_packet["baseline_quality_pct"]) - 8.0),
        stop=float(v2_12_packet["baseline_quality_pct"]) - 0.5,
        step=0.5,
        value=float(v2_12_packet["quality_floor_pct"]),
        label="Quality floor (%)",
    )
    v2_12_incident_count = mo.ui.slider(
        start=0,
        stop=10,
        step=1,
        value=int(v2_12_packet["incident_default_count"]),
        label="Monthly incidents",
    )
    v2_12_detection_min = mo.ui.slider(
        start=0,
        stop=180,
        step=5,
        value=int(v2_12_packet["detection_default_min"]),
        label="Detection delay per incident (min)",
    )
    v2_12_impact_min = mo.ui.slider(
        start=5,
        stop=240,
        step=5,
        value=int(v2_12_packet["impact_default_min"]),
        label="Impact duration after detection (min)",
    )
    v2_12_partA_checkpoint = mo.ui.radio(
        options={
            "Tighten the rollout or monitoring loop before promotion.": "tighten_loop",
            "Spend the budget now and promise a later cleanup.": "spend_now",
            "Ignore quality budget if uptime remains green.": "uptime_only",
        },
        label="Part A checkpoint",
    )

    v2_12_partB_pred = mo.ui.radio(
        options={
            "A tiny canary is safest because exposure is smallest.": "tiny_blind",
            "A moderate canary balances evidence and exposure.": "balanced",
            "An aggressive canary is safest because it learns fastest.": "aggressive_exposed",
        },
        label="Part B prediction",
    )
    v2_12_canary_pct = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        value=int(v2_12_packet["canary_default_pct"]),
        label=f"Canary traffic (% of {v2_12_packet['release_unit']})",
    )
    v2_12_stage_hours = mo.ui.slider(
        start=0.25,
        stop=12.0,
        step=0.25,
        value=float(v2_12_packet["canary_default_h"]),
        label="Stage duration (hours)",
    )
    v2_12_sample_needed = mo.ui.slider(
        start=500,
        stop=50_000,
        step=500,
        value=int(v2_12_packet["sample_default"]),
        label="Samples needed for decision",
    )
    v2_12_traffic_multiplier = mo.ui.slider(
        start=0.25,
        stop=2.0,
        step=0.25,
        value=1.0,
        label="Traffic / evidence rate multiplier",
    )
    v2_12_partB_checkpoint = mo.ui.radio(
        options={
            "Hold or expand only after statistical evidence is sufficient.": "evidence_gate",
            "Promote once no hard failures appear.": "hard_failures_only",
            "Keep the canary tiny even if the release window expires.": "tiny_forever",
        },
        label="Part B checkpoint",
    )

    v2_12_partC_pred = mo.ui.radio(
        options={
            "Restart serving infrastructure first.": "restart_first",
            "Inspect data/model-quality signals before infrastructure fixes.": "inspect_semantic",
            "Roll back all traffic before classifying the failure.": "rollback_first",
            "Wait for more labels before action.": "wait_for_labels",
        },
        label="Part C prediction",
    )
    v2_12_mttd_min = mo.ui.slider(
        start=0,
        stop=180,
        step=5,
        value=int(v2_12_packet["mttd_default_min"]),
        label="Mean time to detect (min)",
    )
    v2_12_diagnosis_min = mo.ui.slider(
        start=5,
        stop=180,
        step=5,
        value=int(v2_12_packet["diagnosis_default_min"]),
        label="Diagnosis / attribution time (min)",
    )
    v2_12_mitigation_pct = mo.ui.slider(
        start=0,
        stop=95,
        step=5,
        value=int(v2_12_packet["mitigation_default_pct"]),
        label="Blast-radius reduction after mitigation (%)",
    )
    v2_12_recovery_min = mo.ui.slider(
        start=5,
        stop=240,
        step=5,
        value=int(v2_12_packet["recovery_default_min"]),
        label="Recovery time after mitigation (min)",
    )
    v2_12_runbook_level = mo.ui.dropdown(
        options={
            "Ad hoc notes": "ad_hoc",
            "Typed diagnostic runbook": "typed_runbook",
            "Drilled automated playbook": "drilled_automation",
        },
        value=str(v2_12_packet["runbook_default_label"]),
        label="Runbook maturity",
    )
    v2_12_partC_checkpoint = mo.ui.radio(
        options={
            "Add the missing signal/control and drill the runbook.": "add_control",
            "Accept this as normal operational noise.": "accept_noise",
            "Only increase page volume for the same signals.": "more_pages",
        },
        label="Part C checkpoint",
    )

    v2_12_partD_pred = mo.ui.radio(
        options={
            "SLO/error-budget guardrail rejects the naive fast policy.": "slo",
            "Blast-radius guardrail rejects the naive fast policy.": "blast",
            "Cost guardrail rejects the naive fast policy.": "cost",
            "Governance guardrail rejects the naive fast policy.": "governance",
        },
        label="Part D prediction",
    )
    v2_12_rollout_aggression = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=35,
        label="Proposed rollout aggressiveness",
    )
    v2_12_automation_level = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=65,
        label="Automation / rollback readiness",
    )
    v2_12_telemetry_depth = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=70,
        label="Telemetry and evidence depth",
    )
    v2_12_governance_level = mo.ui.dropdown(
        options={
            "Self-approved": "self_approved",
            "Peer reviewed": "peer_reviewed",
            "Governed release window": "governed_window",
            "Safety/security board": "safety_board",
        },
        value="Governed release window",
        label="Governance boundary",
    )
    v2_12_partD_policy_choice = mo.ui.dropdown(
        options={
            "Use proposed policy": "custom",
            "Fast rollout": "fast",
            "Balanced control loop": "balanced",
            "Conservative gated rollout": "conservative",
        },
        value="Balanced control loop",
        label="Selected operations policy",
    )
    v2_12_rejected_policy = mo.ui.dropdown(
        options={
            "Fast rollout": "fast",
            "Balanced control loop": "balanced",
            "Conservative gated rollout": "conservative",
        },
        value="Fast rollout",
        label="Rejected alternative",
    )
    v2_12_partD_checkpoint = mo.ui.radio(
        options={
            "Approve only if all guardrails pass together.": "all_guardrails",
            "Approve the lowest-cost policy even with a failed guardrail.": "cost_only",
            "Approve the fastest rollout if rollback is automated.": "speed_only",
            "Approve after governance review even if SLO fails.": "governance_only",
        },
        label="Part D checkpoint",
    )

    v2_12_student_id = mo.ui.text(label="Student identifier", placeholder="Optional")
    v2_12_security_implication = mo.ui.radio(
        options={
            "Telemetry minimization affects detection.": "telemetry_minimization",
            "Blast-radius policy bounds exploit exposure.": "exploit_blast_radius",
            "Governance review creates an audit trail.": "audit_trail",
            "Incident evidence defines security escalation.": "incident_escalation",
        },
        label="V2-13 security/privacy implication",
    )
    v2_12_memo_note = mo.ui.text_area(
        label="Optional memo note",
        placeholder="One sentence of local context or residual risk.",
    )

    return (
        v2_12_automation_level,
        v2_12_canary_pct,
        v2_12_detection_min,
        v2_12_diagnosis_min,
        v2_12_governance_level,
        v2_12_impact_min,
        v2_12_incident_count,
        v2_12_memo_note,
        v2_12_mitigation_pct,
        v2_12_mttd_min,
        v2_12_partA_checkpoint,
        v2_12_partA_pred,
        v2_12_partB_checkpoint,
        v2_12_partB_pred,
        v2_12_partC_checkpoint,
        v2_12_partC_pred,
        v2_12_partD_checkpoint,
        v2_12_partD_policy_choice,
        v2_12_partD_pred,
        v2_12_quality_floor,
        v2_12_recovery_min,
        v2_12_rejected_policy,
        v2_12_rollout_aggression,
        v2_12_runbook_level,
        v2_12_sample_needed,
        v2_12_security_implication,
        v2_12_slo_pct,
        v2_12_stage_hours,
        v2_12_student_id,
        v2_12_telemetry_depth,
        v2_12_traffic_multiplier,
    )


@app.cell
def _(math, v2_12_period_minutes):
    def v2_12_error_budget(
        packet,
        *,
        slo_pct,
        quality_floor_pct,
        incident_count,
        detection_min,
        impact_min,
    ):
        period_minutes = v2_12_period_minutes(packet["period_days"])
        error_budget_minutes = period_minutes * (1 - slo_pct / 100)
        impact_minutes = incident_count * (detection_min + impact_min)
        quality_budget_pp = max(0.01, packet["baseline_quality_pct"] - quality_floor_pct)
        quality_spend_pp = (
            incident_count * packet["quality_loss_per_incident_pp"]
            + incident_count * detection_min / 60 * packet["quality_drift_pp_per_h"]
        )
        incident_budget = max(1, packet["incident_budget_count"])
        rows = [
            {
                "amount": "Availability error budget",
                "budget": error_budget_minutes,
                "spend": impact_minutes,
                "unit": "min",
                "ratio": impact_minutes / max(0.001, error_budget_minutes),
            },
            {
                "amount": "Quality / drift budget",
                "budget": quality_budget_pp,
                "spend": quality_spend_pp,
                "unit": "pp",
                "ratio": quality_spend_pp / quality_budget_pp,
            },
            {
                "amount": "Incident count budget",
                "budget": incident_budget,
                "spend": incident_count,
                "unit": "incidents",
                "ratio": incident_count / incident_budget,
            },
        ]
        binding_row = max(rows, key=lambda row: row["ratio"])
        actual = {
            "Availability error budget": "availability",
            "Quality / drift budget": "quality",
            "Incident count budget": "incident_count",
        }[binding_row["amount"]]
        return {
            "period_minutes": period_minutes,
            "error_budget_minutes": error_budget_minutes,
            "impact_minutes": impact_minutes,
            "quality_budget_pp": quality_budget_pp,
            "quality_spend_pp": quality_spend_pp,
            "incident_budget": incident_budget,
            "incident_count": incident_count,
            "rows": rows,
            "binding": binding_row["amount"],
            "binding_key": actual,
            "binding_ratio": binding_row["ratio"],
            "ok": all(row["ratio"] <= 1 for row in rows),
        }

    def v2_12_canary_budget(
        packet,
        *,
        canary_pct,
        stage_hours,
        sample_needed,
        traffic_multiplier,
    ):
        traffic_rate = packet["traffic_per_h"] * traffic_multiplier
        traffic_fraction = max(0.0001, canary_pct / 100)
        required_hours = sample_needed / max(0.001, traffic_rate * traffic_fraction)
        samples_collected = traffic_rate * traffic_fraction * stage_hours
        blast_units = samples_collected
        evidence_ok = stage_hours >= required_hours
        blast_ok = blast_units <= packet["blast_budget_units"]
        if not evidence_ok and blast_ok:
            actual = "tiny_blind"
        elif evidence_ok and not blast_ok:
            actual = "aggressive_exposed"
        elif evidence_ok and blast_ok:
            actual = "balanced"
        else:
            actual = "aggressive_exposed" if blast_units / packet["blast_budget_units"] > required_hours / stage_hours else "tiny_blind"
        return {
            "traffic_rate": traffic_rate,
            "traffic_fraction": traffic_fraction,
            "required_hours": required_hours,
            "samples_collected": samples_collected,
            "blast_units": blast_units,
            "evidence_ok": evidence_ok,
            "blast_ok": blast_ok,
            "actual": actual,
            "blind_ratio": required_hours / max(0.001, stage_hours),
            "blast_ratio": blast_units / max(0.001, packet["blast_budget_units"]),
        }

    def v2_12_runbook_factor(level):
        table = {
            "ad_hoc": 1.25,
            "Ad hoc notes": 1.25,
            "typed_runbook": 1.0,
            "Typed diagnostic runbook": 1.0,
            "drilled_automation": 0.70,
            "Drilled automated playbook": 0.70,
        }
        return table.get(level, 1.0)

    def v2_12_incident_budget(
        packet,
        *,
        mttd_min,
        diagnosis_min,
        mitigation_pct,
        recovery_min,
        runbook_level,
    ):
        runbook_factor = v2_12_runbook_factor(runbook_level)
        effective_diagnosis = diagnosis_min * runbook_factor
        impact_fraction = packet["incident_impact_fraction"]
        affected = packet["affected_units_per_min"]
        pre_mitigation_minutes = mttd_min + effective_diagnosis
        residual_fraction = impact_fraction * (1 - mitigation_pct / 100)
        pre_loss = affected * impact_fraction * pre_mitigation_minutes
        recovery_loss = affected * residual_fraction * recovery_min
        lost_work = pre_loss + recovery_loss
        budget = packet["response_budget_units"]
        return {
            "runbook_factor": runbook_factor,
            "effective_diagnosis_min": effective_diagnosis,
            "pre_mitigation_minutes": pre_mitigation_minutes,
            "residual_fraction": residual_fraction,
            "pre_loss": pre_loss,
            "recovery_loss": recovery_loss,
            "lost_work": lost_work,
            "budget": budget,
            "ok": lost_work <= budget,
            "ratio": lost_work / max(0.001, budget),
        }

    def v2_12_governance_score(level):
        table = {
            "self_approved": 0.5,
            "Self-approved": 0.5,
            "peer_reviewed": 1.5,
            "Peer reviewed": 1.5,
            "governed_window": 2.5,
            "Governed release window": 2.5,
            "safety_board": 4.0,
            "Safety/security board": 4.0,
        }
        return table.get(level, 2.5)

    def v2_12_policy_eval(packet, *, name, rollout, automation, telemetry, governance):
        gov_score = v2_12_governance_score(governance)
        slo_spend_pct = max(5.0, 88 + rollout * 0.62 - automation * 0.34 - telemetry * 0.30)
        blast_pct = max(0.5, rollout * (1 - automation / 260))
        cost_index = 55 + telemetry * 0.38 + automation * 0.28 + gov_score * 8 - rollout * 0.08
        governance_effective = gov_score + telemetry / 120
        slo_ok = slo_spend_pct <= packet["slo_guardrail_limit_pct"]
        blast_ok = blast_pct <= packet["max_blast_pct"]
        cost_ok = cost_index <= packet["cost_limit_index"]
        governance_ok = governance_effective >= packet["governance_min_score"]
        guardrails = {
            "slo": slo_ok,
            "blast": blast_ok,
            "cost": cost_ok,
            "governance": governance_ok,
        }
        binding = max(
            (
                ("slo", slo_spend_pct / packet["slo_guardrail_limit_pct"]),
                ("blast", blast_pct / packet["max_blast_pct"]),
                ("cost", cost_index / packet["cost_limit_index"]),
                ("governance", packet["governance_min_score"] / max(0.01, governance_effective)),
            ),
            key=lambda item: item[1],
        )[0]
        return {
            "name": name,
            "rollout": rollout,
            "automation": automation,
            "telemetry": telemetry,
            "governance": governance,
            "governance_score": governance_effective,
            "slo_spend_pct": slo_spend_pct,
            "blast_pct": blast_pct,
            "cost_index": cost_index,
            "slo_ok": slo_ok,
            "blast_ok": blast_ok,
            "cost_ok": cost_ok,
            "governance_ok": governance_ok,
            "guardrails": guardrails,
            "binding": binding,
            "feasible": all(guardrails.values()),
        }

    def v2_12_policy_label(key):
        return {
            "custom": "Use proposed policy",
            "fast": "Fast rollout",
            "balanced": "Balanced control loop",
            "conservative": "Conservative gated rollout",
            "Use proposed policy": "Use proposed policy",
            "Fast rollout": "Fast rollout",
            "Balanced control loop": "Balanced control loop",
            "Conservative gated rollout": "Conservative gated rollout",
        }.get(key, str(key))

    def v2_12_policy_key(value):
        return {
            "Use proposed policy": "custom",
            "Fast rollout": "fast",
            "Balanced control loop": "balanced",
            "Conservative gated rollout": "conservative",
            "custom": "custom",
            "fast": "fast",
            "balanced": "balanced",
            "conservative": "conservative",
        }.get(value, "balanced")

    def v2_12_guardrail_label(key):
        return {
            "slo": "SLO / error budget",
            "blast": "Blast radius",
            "cost": "Cost",
            "governance": "Governance",
        }.get(key, str(key))

    return (
        v2_12_canary_budget,
        v2_12_error_budget,
        v2_12_guardrail_label,
        v2_12_incident_budget,
        v2_12_policy_eval,
        v2_12_policy_key,
        v2_12_policy_label,
        v2_12_runbook_factor,
    )


@app.cell
def _(
    v2_12_automation_level,
    v2_12_canary_budget,
    v2_12_canary_pct,
    v2_12_detection_min,
    v2_12_diagnosis_min,
    v2_12_error_budget,
    v2_12_governance_level,
    v2_12_impact_min,
    v2_12_incident_budget,
    v2_12_incident_count,
    v2_12_mitigation_pct,
    v2_12_mttd_min,
    v2_12_packet,
    v2_12_partD_policy_choice,
    v2_12_policy_eval,
    v2_12_policy_key,
    v2_12_quality_floor,
    v2_12_recovery_min,
    v2_12_rejected_policy,
    v2_12_rollout_aggression,
    v2_12_runbook_level,
    v2_12_sample_needed,
    v2_12_slo_pct,
    v2_12_stage_hours,
    v2_12_telemetry_depth,
    v2_12_traffic_multiplier,
):
    v2_12_a = v2_12_error_budget(
        v2_12_packet,
        slo_pct=v2_12_slo_pct.value,
        quality_floor_pct=v2_12_quality_floor.value,
        incident_count=v2_12_incident_count.value,
        detection_min=v2_12_detection_min.value,
        impact_min=v2_12_impact_min.value,
    )
    v2_12_b = v2_12_canary_budget(
        v2_12_packet,
        canary_pct=v2_12_canary_pct.value,
        stage_hours=v2_12_stage_hours.value,
        sample_needed=v2_12_sample_needed.value,
        traffic_multiplier=v2_12_traffic_multiplier.value,
    )
    v2_12_c = v2_12_incident_budget(
        v2_12_packet,
        mttd_min=v2_12_mttd_min.value,
        diagnosis_min=v2_12_diagnosis_min.value,
        mitigation_pct=v2_12_mitigation_pct.value,
        recovery_min=v2_12_recovery_min.value,
        runbook_level=v2_12_runbook_level.value,
    )
    v2_12_d_policies = {
        "custom": v2_12_policy_eval(
            v2_12_packet,
            name="Use proposed policy",
            rollout=v2_12_rollout_aggression.value,
            automation=v2_12_automation_level.value,
            telemetry=v2_12_telemetry_depth.value,
            governance=v2_12_governance_level.value,
        ),
        "fast": v2_12_policy_eval(
            v2_12_packet,
            name="Fast rollout",
            rollout=70,
            automation=45,
            telemetry=35,
            governance="self_approved",
        ),
        "balanced": v2_12_policy_eval(
            v2_12_packet,
            name="Balanced control loop",
            rollout=22,
            automation=75,
            telemetry=75,
            governance="governed_window",
        ),
        "conservative": v2_12_policy_eval(
            v2_12_packet,
            name="Conservative gated rollout",
            rollout=8,
            automation=88,
            telemetry=90,
            governance="safety_board",
        ),
    }
    v2_12_selected_policy_key = v2_12_policy_key(v2_12_partD_policy_choice.value)
    v2_12_selected_policy = v2_12_d_policies[v2_12_selected_policy_key]
    v2_12_rejected_policy_key = v2_12_policy_key(v2_12_rejected_policy.value)
    v2_12_rejected_policy_result = v2_12_d_policies[v2_12_rejected_policy_key]
    v2_12_fast_policy = v2_12_d_policies["fast"]
    return (
        v2_12_a,
        v2_12_b,
        v2_12_c,
        v2_12_d_policies,
        v2_12_fast_policy,
        v2_12_rejected_policy_key,
        v2_12_rejected_policy_result,
        v2_12_selected_policy,
        v2_12_selected_policy_key,
    )


@app.cell
def _(COLORS, apply_plotly_theme, go, pd, v2_12_fmt_hours, v2_12_fmt_number, v2_12_fmt_pct, v2_12_guardrail_badge):
    def v2_12_color(key, fallback):
        return COLORS.get(key, fallback)

    def v2_12_budget_table(a):
        return pd.DataFrame(
            [
                {
                    "Amount": row["amount"],
                    "Budget": f"{row['budget']:.2f} {row['unit']}",
                    "Spend": f"{row['spend']:.2f} {row['unit']}",
                    "Spend / budget": f"{row['ratio']:.2f}x",
                    "Status": "PASS" if row["ratio"] <= 1 else "FAIL",
                }
                for row in a["rows"]
            ]
        )

    def v2_12_budget_fig(a):
        fig = go.Figure()
        names = [row["amount"] for row in a["rows"]]
        spends = [row["ratio"] for row in a["rows"]]
        fig.add_bar(name="Spend / budget", x=names, y=spends, marker_color=v2_12_color("OrangeLine", "#f97316"))
        fig.add_hline(y=1.0, line_width=2, line_dash="dash", line_color=v2_12_color("BlueLine", "#2563eb"))
        fig.update_layout(
            title="Part A amount system: spend ratio by budget",
            yaxis_title="Spend / budget",
            height=390,
            legend_orientation="h",
        )
        apply_plotly_theme(fig)
        return fig

    def v2_12_canary_table(packet, b):
        return pd.DataFrame(
            [
                {"Metric": "Traffic rate", "Value": f"{v2_12_fmt_number(b['traffic_rate'])} {packet['traffic_unit']}/hour"},
                {"Metric": "Samples collected", "Value": f"{v2_12_fmt_number(b['samples_collected'])} samples"},
                {"Metric": "Required stage duration", "Value": v2_12_fmt_hours(b["required_hours"])},
                {"Metric": "Configured stage duration", "Value": v2_12_fmt_hours(b["samples_collected"] / max(1e-9, b["traffic_rate"] * b["traffic_fraction"]))},
                {"Metric": "Blast-radius spend", "Value": f"{v2_12_fmt_number(b['blast_units'])} {packet['traffic_unit']}"},
                {"Metric": "Blast-radius budget", "Value": f"{v2_12_fmt_number(packet['blast_budget_units'])} {packet['traffic_unit']}"},
                {"Metric": "Evidence status", "Value": v2_12_guardrail_badge(b["evidence_ok"])},
                {"Metric": "Blast status", "Value": v2_12_guardrail_badge(b["blast_ok"])},
            ]
        )

    def v2_12_canary_fig(packet, canary_pct, sample_needed, traffic_rate, stage_hours):
        pct_values = list(range(1, 51))
        required = [sample_needed / max(0.001, traffic_rate * (pct / 100)) for pct in pct_values]
        blast = [traffic_rate * (pct / 100) * stage_hours for pct in pct_values]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pct_values,
                y=required,
                mode="lines",
                name="Required hours",
                line=dict(color=v2_12_color("BlueLine", "#2563eb"), width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pct_values,
                y=blast,
                mode="lines",
                name=f"Blast-radius {packet['traffic_unit']}",
                yaxis="y2",
                line=dict(color=v2_12_color("OrangeLine", "#f97316"), width=3),
            )
        )
        fig.add_vline(x=canary_pct, line_width=2, line_dash="dash", line_color=v2_12_color("GreenLine", "#16a34a"))
        fig.add_hline(y=stage_hours, line_width=1, line_dash="dot", line_color=v2_12_color("TextMuted", "#64748b"))
        fig.update_layout(
            title="Part B canary trade-off: evidence time versus exposed amount",
            xaxis_title="Canary traffic (%)",
            yaxis_title="Hours needed for evidence",
            yaxis2=dict(title=f"Exposed {packet['traffic_unit']}", overlaying="y", side="right"),
            height=430,
            legend_orientation="h",
        )
        apply_plotly_theme(fig)
        return fig

    def v2_12_incident_table(packet, c):
        return pd.DataFrame(
            [
                {"Segment": "Detect + diagnose", "Lost work": f"{v2_12_fmt_number(c['pre_loss'])} {packet['traffic_unit']}", "Notes": "Full impact before mitigation."},
                {"Segment": "Recover after mitigation", "Lost work": f"{v2_12_fmt_number(c['recovery_loss'])} {packet['traffic_unit']}", "Notes": "Residual impact after blast radius reduction."},
                {"Segment": "Total", "Lost work": f"{v2_12_fmt_number(c['lost_work'])} {packet['traffic_unit']}", "Notes": f"Budget: {v2_12_fmt_number(c['budget'])} {packet['traffic_unit']}."},
            ]
        )

    def v2_12_incident_fig(packet, mttd_min, c, recovery_min):
        fig = go.Figure()
        fig.add_bar(
            x=["Incident timeline"],
            y=[mttd_min],
            name="Detect",
            marker_color=v2_12_color("BlueLine", "#2563eb"),
        )
        fig.add_bar(
            x=["Incident timeline"],
            y=[max(0.0, c["effective_diagnosis_min"])],
            name="Diagnose",
            marker_color=v2_12_color("OrangeLine", "#f97316"),
        )
        fig.add_bar(
            x=["Incident timeline"],
            y=[recovery_min],
            name="Recover",
            marker_color=v2_12_color("GreenLine", "#16a34a"),
        )
        fig.update_layout(
            title=f"Part C recovery time budget for {packet['ops_unit']}",
            barmode="stack",
            yaxis_title="Minutes",
            height=360,
            legend_orientation="h",
        )
        apply_plotly_theme(fig)
        return fig

    def v2_12_policy_table(d_policies, guardrail_label):
        rows = []
        for key in ("fast", "balanced", "conservative", "custom"):
            policy = d_policies[key]
            rows.append(
                {
                    "Policy": policy["name"],
                    "SLO spend": v2_12_fmt_pct(policy["slo_spend_pct"]),
                    "Blast radius": v2_12_fmt_pct(policy["blast_pct"]),
                    "Cost index": f"{policy['cost_index']:.1f}",
                    "Governance score": f"{policy['governance_score']:.2f}",
                    "Binding": guardrail_label(policy["binding"]),
                    "Feasible": v2_12_guardrail_badge(policy["feasible"]),
                }
            )
        return pd.DataFrame(rows)

    def v2_12_policy_fig(d_policies):
        keys = ("fast", "balanced", "conservative", "custom")
        labels = [d_policies[key]["name"] for key in keys]
        fig = go.Figure()
        fig.add_bar(name="SLO spend", x=labels, y=[d_policies[key]["slo_spend_pct"] for key in keys])
        fig.add_bar(name="Blast radius", x=labels, y=[d_policies[key]["blast_pct"] for key in keys])
        fig.add_bar(name="Cost index", x=labels, y=[d_policies[key]["cost_index"] for key in keys])
        fig.update_layout(
            title="Part D policy amounts by guardrail family",
            barmode="group",
            yaxis_title="Index / percent",
            height=410,
            legend_orientation="h",
        )
        apply_plotly_theme(fig)
        return fig

    return (
        v2_12_budget_fig,
        v2_12_budget_table,
        v2_12_canary_fig,
        v2_12_canary_table,
        v2_12_color,
        v2_12_incident_fig,
        v2_12_incident_table,
        v2_12_policy_fig,
        v2_12_policy_table,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    COLORS,
    LAB_CSS,
    mo,
    source_trace,
    track_arc_context,
    track_context,
    v2_12_escape,
    v2_12_metadata,
    v2_12_packet,
    v2_12_profile,
    v2_12_variant,
):
    mo.vstack(
        [
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            mo.Html(
                f"""
                <div style="background:linear-gradient(135deg, {COLORS['Surface0']} 0%, {COLORS['Surface1']} 100%);
                            border-radius:16px; padding:32px 40px; margin-bottom:8px;
                            border:1px solid #2d3748;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
                        <div>
                            <div style="font-size:0.72rem; font-weight:700; color:#94a3b8;
                                        text-transform:uppercase; letter-spacing:0.14em; margin-bottom:8px;">
                                Vol 2 &middot; Lab 12 &middot; ML Operations at Scale
                            </div>
                            <div style="font-size:2rem; font-weight:800; color:#f1f5f9; line-height:1.15; margin-bottom:10px;">
                                Operations at Scale as Control Loops
                            </div>
                            <div style="font-size:0.95rem; color:#94a3b8; max-width:760px; line-height:1.6;">
                                {v2_12_escape(v2_12_variant.workload_summary)} The shared concept sequence treats
                                SLOs, canaries, incidents, and policy guardrails as amounts that spend budget over time.
                            </div>
                        </div>
                        <div style="display:flex; flex-direction:column; gap:8px; flex-shrink:0;">
                            <span class="badge badge-info">{v2_12_escape(v2_12_profile.label)}</span>
                            <span class="badge badge-info">{v2_12_escape(v2_12_packet['ops_unit'])}</span>
                            <span class="badge badge-info">{v2_12_escape(v2_12_packet['hardware_ref'])}</span>
                            <span class="badge badge-warn">45-55 minutes &middot; 4 Parts + Synthesis</span>
                        </div>
                    </div>
                </div>
                """
            ),
            track_context(v2_12_profile),
            track_arc_context(v2_12_profile, v2_12_metadata.lab_id),
            source_trace(
                {
                    "chapter": "Volume II, Chapter 12: ML Operations at Scale",
                    "anchors": (
                        "SLOs and freshness SLOs",
                        "Canary duration equation",
                        "Runbook diagnostic flow",
                        "Self-service deployment invariants",
                    ),
                    "track_source": v2_12_packet["source_policy"],
                    "implementation": "Notebook-local v2_12_ amount-system formulas; shared track metadata and report helpers.",
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v2_12_escape, v2_12_packet):
    mo.Html(
        f"""
        <div style="border-left:4px solid {COLORS['BlueLine']};
                    background:white; border-radius:0 8px 8px 0;
                    padding:20px 28px; margin:8px 0 16px 0;
                    box-shadow:0 1px 4px rgba(0,0,0,0.06);">
            <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                Shared concept sequence
            </div>
            <div style="font-size:0.9rem; color:{COLORS['TextSec']}; line-height:1.7;">
                <div>1. <strong>SLO/error budget:</strong> reliability becomes an amount system.</div>
                <div>2. <strong>Canary/rollout:</strong> learning speed trades against blast radius.</div>
                <div>3. <strong>Incident response:</strong> recovery time becomes lost-work budget.</div>
                <div>4. <strong>Operations policy:</strong> SLO, blast radius, cost, and governance guardrails must all pass.</div>
            </div>
            <div style="border-top:1px solid {COLORS['Border']}; margin:16px -28px 0 -28px; padding:16px 28px 0 28px;
                        font-size:0.86rem; color:{COLORS['TextSec']}; line-height:1.65;">
                <strong>Track lens:</strong> {v2_12_escape(v2_12_packet['stakeholder'])} manages
                <strong>{v2_12_escape(v2_12_packet['ops_unit'])}</strong> using
                <strong>{v2_12_escape(v2_12_packet['primary_signal'])}</strong>.
                Natural failure: {v2_12_escape(v2_12_packet['failure_mode'])}.
            </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    ledger,
    mo,
    report_export_panel,
    v2_12_a,
    v2_12_automation_level,
    v2_12_b,
    v2_12_budget_fig,
    v2_12_budget_table,
    v2_12_c,
    v2_12_canary_fig,
    v2_12_canary_pct,
    v2_12_canary_table,
    v2_12_chapter,
    v2_12_color,
    v2_12_d_policies,
    v2_12_detection_min,
    v2_12_diagnosis_min,
    v2_12_escape,
    v2_12_fast_policy,
    v2_12_fmt_hours,
    v2_12_fmt_minutes,
    v2_12_fmt_number,
    v2_12_fmt_pct,
    v2_12_governance_level,
    v2_12_guardrail_label,
    v2_12_impact_min,
    v2_12_incident_count,
    v2_12_incident_fig,
    v2_12_incident_table,
    v2_12_memo_note,
    v2_12_metadata,
    v2_12_mitigation_pct,
    v2_12_mttd_min,
    v2_12_packet,
    v2_12_partA_checkpoint,
    v2_12_partA_pred,
    v2_12_partB_checkpoint,
    v2_12_partB_pred,
    v2_12_partC_checkpoint,
    v2_12_partC_pred,
    v2_12_partD_checkpoint,
    v2_12_partD_policy_choice,
    v2_12_partD_pred,
    v2_12_policy_fig,
    v2_12_policy_label,
    v2_12_policy_table,
    v2_12_prediction_feedback,
    v2_12_profile,
    v2_12_quality_floor,
    v2_12_recovery_min,
    v2_12_rejected_policy,
    v2_12_rejected_policy_result,
    v2_12_rollout_aggression,
    v2_12_runbook_level,
    v2_12_sample_needed,
    v2_12_security_implication,
    v2_12_selected_policy,
    v2_12_slo_pct,
    v2_12_stage_hours,
    v2_12_student_id,
    v2_12_telemetry_depth,
    v2_12_traffic_multiplier,
):
    def v2_12_gate(pred_widget, items):
        if pred_widget.value is None:
            items.append(
                mo.callout(
                    mo.md("Commit to the structured prediction first; the instrument is hidden until the prior is explicit."),
                    kind="warn",
                )
            )
            return True
        return False

    def v2_12_feedback(predicted, actual, labels):
        kind, message = v2_12_prediction_feedback(predicted, actual, labels)
        return mo.callout(mo.md(message), kind=kind)

    def v2_12_status_callout(ok, success, failure):
        return mo.callout(mo.md(success if ok else failure), kind="success" if ok else "danger")

    def v2_12_build_part_a():
        labels = {
            "availability": "availability error-budget minutes",
            "quality": "quality/drift points",
            "incident_count": "incident count",
        }
        items = [
            mo.md(
                f"""
                ### Scenario
                You are the {v2_12_packet['stakeholder']}. A release is ready for the
                {v2_12_packet['ops_unit']}, but the chapter's SLO idea requires you to
                translate reliability into budgets before approving more exposure.
                """
            ),
            v2_12_partA_pred,
        ]
        if v2_12_gate(v2_12_partA_pred, items):
            return mo.vstack(items)
        items.extend(
            [
                v2_12_feedback(v2_12_partA_pred.value, v2_12_a["binding_key"], labels),
                mo.hstack(
                    [
                        v2_12_slo_pct,
                        v2_12_quality_floor,
                        v2_12_incident_count,
                    ],
                    widths="equal",
                ),
                mo.hstack([v2_12_detection_min, v2_12_impact_min], widths="equal"),
                v2_12_budget_fig(v2_12_a),
                v2_12_budget_table(v2_12_a),
                v2_12_status_callout(
                    v2_12_a["ok"],
                    f"Recovered envelope. The current release stays within all three tracked operation amounts; binding amount: `{v2_12_a['binding']}` at {v2_12_a['binding_ratio']:.2f}x budget.",
                    f"Budget violation. `{v2_12_a['binding']}` is overspent at {v2_12_a['binding_ratio']:.2f}x budget. Reduce incidents, shorten detection, widen the SLO budget, or tighten rollout exposure before promotion.",
                ),
                mo.accordion(
                    {
                        "Math Peek / source model": mo.md(
                            f"""
                            Error-budget minutes = period minutes x (1 - SLO).

                            For this 30-day window: `{v2_12_fmt_minutes(v2_12_a['error_budget_minutes'])}`
                            are allowed by the availability SLO. Incident spend is
                            `{v2_12_incident_count.value}` incidents x
                            (`{v2_12_detection_min.value}` detection min + `{v2_12_impact_min.value}` impact min)
                            = `{v2_12_fmt_minutes(v2_12_a['impact_minutes'])}`.

                            Quality error budget = baseline quality - quality floor =
                            `{v2_12_packet['baseline_quality_pct']:.1f}% - {v2_12_quality_floor.value:.1f}%`
                            = `{v2_12_a['quality_budget_pp']:.2f}` percentage points.
                            """
                        )
                    }
                ),
                v2_12_partA_checkpoint,
            ]
        )
        return mo.vstack(items)

    def v2_12_build_part_b():
        labels = {
            "tiny_blind": "too small to learn inside the release window",
            "balanced": "balanced evidence and exposure",
            "aggressive_exposed": "too much blast-radius exposure",
        }
        items = [
            mo.md(
                f"""
                ### Scenario
                The same control loop now moves from target setting to live rollout.
                The candidate gets production evidence only from the canary slice, but
                every exposed {v2_12_packet['traffic_unit']} spends blast-radius budget.
                """
            ),
            v2_12_partB_pred,
        ]
        if v2_12_gate(v2_12_partB_pred, items):
            return mo.vstack(items)
        items.extend(
            [
                v2_12_feedback(v2_12_partB_pred.value, v2_12_b["actual"], labels),
                mo.hstack([v2_12_canary_pct, v2_12_stage_hours], widths="equal"),
                mo.hstack([v2_12_sample_needed, v2_12_traffic_multiplier], widths="equal"),
                v2_12_canary_fig(
                    v2_12_packet,
                    v2_12_canary_pct.value,
                    v2_12_sample_needed.value,
                    v2_12_b["traffic_rate"],
                    v2_12_stage_hours.value,
                ),
                v2_12_canary_table(v2_12_packet, v2_12_b),
                v2_12_status_callout(
                    v2_12_b["evidence_ok"] and v2_12_b["blast_ok"],
                    f"Rollout boundary is healthy. This stage gathers enough evidence in `{v2_12_fmt_hours(v2_12_stage_hours.value)}` while spending `{v2_12_fmt_number(v2_12_b['blast_units'])}` {v2_12_packet['traffic_unit']} of blast radius.",
                    f"Rollout boundary fails. Evidence ratio is `{v2_12_b['blind_ratio']:.2f}x` and blast ratio is `{v2_12_b['blast_ratio']:.2f}x`; adjust percentage, duration, sample requirement, or release window.",
                ),
                mo.accordion(
                    {
                        "Math Peek / source model": mo.md(
                            f"""
                            Chapter formula: `T_stage = n_samples_needed / (request_rate * p_stage)`.

                            Here, `T_stage = {v2_12_sample_needed.value:,} / `
                            `({v2_12_fmt_number(v2_12_b['traffic_rate'])} * {v2_12_canary_pct.value / 100:.2f})`
                            = `{v2_12_fmt_hours(v2_12_b['required_hours'])}`.

                            Blast-radius spend = traffic rate x canary fraction x stage hours =
                            `{v2_12_fmt_number(v2_12_b['blast_units'])}` {v2_12_packet['traffic_unit']}.
                            """
                        )
                    }
                ),
                v2_12_partB_checkpoint,
            ]
        )
        return mo.vstack(items)

    def v2_12_build_part_c():
        labels = {
            "restart_first": "restart serving first",
            "inspect_semantic": "inspect data/model-quality signals first",
            "rollback_first": "rollback before classifying",
            "wait_for_labels": "wait for more labels",
        }
        items = [
            mo.md(
                f"""
                ### Scenario
                A production incident is visible in `{v2_12_packet['quality_signal']}` while
                basic health checks can still look normal. The runbook must preserve diagnostic
                order and limit lost work while the control loop detects, attributes, mitigates,
                and recovers.
                """
            ),
            v2_12_partC_pred,
        ]
        if v2_12_gate(v2_12_partC_pred, items):
            return mo.vstack(items)
        items.extend(
            [
                v2_12_feedback(v2_12_partC_pred.value, "inspect_semantic", labels),
                mo.hstack([v2_12_mttd_min, v2_12_diagnosis_min, v2_12_runbook_level], widths="equal"),
                mo.hstack([v2_12_mitigation_pct, v2_12_recovery_min], widths="equal"),
                v2_12_incident_fig(v2_12_packet, v2_12_mttd_min.value, v2_12_c, v2_12_recovery_min.value),
                v2_12_incident_table(v2_12_packet, v2_12_c),
                v2_12_status_callout(
                    v2_12_c["ok"] and v2_12_partC_pred.value == "inspect_semantic",
                    f"Incident response stays inside the lost-work budget and starts with ML semantic evidence. Lost work: `{v2_12_fmt_number(v2_12_c['lost_work'])}` {v2_12_packet['traffic_unit']}.",
                    f"Response boundary is unsafe. Lost work is `{v2_12_fmt_number(v2_12_c['lost_work'])}` {v2_12_packet['traffic_unit']} against a budget of `{v2_12_fmt_number(v2_12_c['budget'])}`, or the first action violates diagnostic order.",
                ),
                mo.accordion(
                    {
                        "Math Peek / source model": mo.md(
                            f"""
                            Lost work = affected amount per minute x impact fraction x minutes.

                            Before mitigation: `{v2_12_fmt_number(v2_12_packet['affected_units_per_min'])}`
                            {v2_12_packet['traffic_unit']}/min x `{v2_12_fmt_pct(v2_12_packet['incident_impact_fraction'] * 100)}`
                            x `{v2_12_fmt_minutes(v2_12_c['pre_mitigation_minutes'])}`.

                            After mitigation, residual impact is
                            `{v2_12_fmt_pct(v2_12_c['residual_fraction'] * 100)}` for
                            `{v2_12_fmt_minutes(v2_12_recovery_min.value)}`.
                            The chapter runbook flow is detect user impact -> localize dependency ->
                            bound blast radius -> escalate by evidence -> learn from the gap.
                            """
                        )
                    }
                ),
                v2_12_partC_checkpoint,
            ]
        )
        return mo.vstack(items)

    def v2_12_build_part_d():
        labels = {
            "slo": "SLO / error budget",
            "blast": "Blast radius",
            "cost": "Cost",
            "governance": "Governance",
        }
        items = [
            mo.md(
                f"""
                ### Scenario
                You now choose the operating policy for the next rollout and incident path.
                The chapter's self-service deployment invariants make this a guardrail
                conjunction: SLO, blast radius, cost, and governance must all pass.
                """
            ),
            v2_12_partD_pred,
        ]
        if v2_12_gate(v2_12_partD_pred, items):
            return mo.vstack(items)
        items.extend(
            [
                v2_12_feedback(v2_12_partD_pred.value, v2_12_fast_policy["binding"], labels),
                mo.hstack([v2_12_rollout_aggression, v2_12_automation_level], widths="equal"),
                mo.hstack([v2_12_telemetry_depth, v2_12_governance_level], widths="equal"),
                v2_12_policy_fig(v2_12_d_policies),
                v2_12_policy_table(v2_12_d_policies, v2_12_guardrail_label),
                mo.hstack([v2_12_partD_policy_choice, v2_12_rejected_policy], widths="equal"),
                v2_12_status_callout(
                    v2_12_selected_policy["feasible"],
                    f"Selected policy passes all guardrails. Binding guardrail is `{v2_12_guardrail_label(v2_12_selected_policy['binding'])}`.",
                    f"Selected policy is not launchable. Binding guardrail is `{v2_12_guardrail_label(v2_12_selected_policy['binding'])}`; revise rollout, automation, telemetry, or governance.",
                ),
                mo.accordion(
                    {
                        "Math Peek / source model": mo.md(
                            f"""
                            Policy feasibility is not a weighted average. The release is valid only when:

                            `slo_ok and blast_ok and cost_ok and governance_ok`.

                            Current selected policy:
                            SLO spend `{v2_12_fmt_pct(v2_12_selected_policy['slo_spend_pct'])}`,
                            blast radius `{v2_12_fmt_pct(v2_12_selected_policy['blast_pct'])}`,
                            cost index `{v2_12_selected_policy['cost_index']:.1f}`,
                            governance score `{v2_12_selected_policy['governance_score']:.2f}`.
                            """
                        )
                    }
                ),
                v2_12_partD_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_synthesis():
        complete_widgets = (
            ("Part A prediction", v2_12_partA_pred),
            ("Part A checkpoint", v2_12_partA_checkpoint),
            ("Part B prediction", v2_12_partB_pred),
            ("Part B checkpoint", v2_12_partB_checkpoint),
            ("Part C prediction", v2_12_partC_pred),
            ("Part C checkpoint", v2_12_partC_checkpoint),
            ("Part D prediction", v2_12_partD_pred),
            ("Part D checkpoint", v2_12_partD_checkpoint),
            ("Selected operations policy", v2_12_partD_policy_choice),
            ("Rejected alternative", v2_12_rejected_policy),
            ("V2-13 security/privacy implication", v2_12_security_implication),
        )
        incomplete = [label for label, widget in complete_widgets if widget.value is None]
        security_text = v2_12_packet["v2_13_options"].get(
            v2_12_security_implication.value,
            "Select a V2-13 implication to complete the memo.",
        )
        binding_ops_amount = v2_12_guardrail_label(v2_12_selected_policy["binding"])
        snapshot = {
            "track_id": v2_12_profile.track_id,
            "scenario_id": v2_12_variant.scenario_id,
            "selected_policy": v2_12_selected_policy["name"],
            "rejected_policy": v2_12_rejected_policy_result["name"],
            "binding_ops_amount": binding_ops_amount,
            "partA": {
                "binding": v2_12_a["binding"],
                "error_budget_minutes": round(v2_12_a["error_budget_minutes"], 4),
                "spend_minutes": round(v2_12_a["impact_minutes"], 4),
                "quality_budget_pp": round(v2_12_a["quality_budget_pp"], 4),
                "quality_spend_pp": round(v2_12_a["quality_spend_pp"], 4),
                "ok": v2_12_a["ok"],
            },
            "partB": {
                "canary_pct": v2_12_canary_pct.value,
                "required_hours": round(v2_12_b["required_hours"], 4),
                "stage_hours": v2_12_stage_hours.value,
                "blast_radius_units": round(v2_12_b["blast_units"], 4),
                "evidence_ok": v2_12_b["evidence_ok"],
                "blast_ok": v2_12_b["blast_ok"],
            },
            "partC": {
                "mttd_min": v2_12_mttd_min.value,
                "effective_diagnosis_min": round(v2_12_c["effective_diagnosis_min"], 4),
                "recovery_min": v2_12_recovery_min.value,
                "lost_work_units": round(v2_12_c["lost_work"], 4),
                "response_budget_units": round(v2_12_c["budget"], 4),
                "ok": v2_12_c["ok"],
            },
            "partD": {
                "rollout_aggression": v2_12_rollout_aggression.value,
                "automation_level": v2_12_automation_level.value,
                "telemetry_depth": v2_12_telemetry_depth.value,
                "governance_level": v2_12_governance_level.value,
                "selected_policy_key": v2_12_partD_policy_choice.value,
                "rejected_policy_key": v2_12_rejected_policy.value,
                "binding_guardrail": v2_12_selected_policy["binding"],
                "policy_feasible": v2_12_selected_policy["feasible"],
            },
            "v2_13_security_implication": security_text,
            "memo_note": v2_12_memo_note.value,
        }
        report = build_lab_report(
            v2_12_metadata,
            student_id=v2_12_student_id.value or "",
            track=v2_12_profile.label,
            scenario=v2_12_variant.workload_summary,
            learning_objectives=(
                "Explain how SLO and error budget turn reliability into an amount system.",
                "Quantify the canary trade-off between learning speed and blast radius.",
                "Budget incident response by recovery time and lost work.",
                "Choose an operations policy that satisfies SLO, blast radius, cost, and governance guardrails.",
            ),
            predictions={
                "part_a_binding_amount": v2_12_partA_pred.value,
                "part_b_canary_tradeoff": v2_12_partB_pred.value,
                "part_c_first_response": v2_12_partC_pred.value,
                "part_d_naive_policy_binding": v2_12_partD_pred.value,
            },
            knob_settings={
                "availability_slo_pct": v2_12_slo_pct.value,
                "quality_floor_pct": v2_12_quality_floor.value,
                "incident_count": v2_12_incident_count.value,
                "canary_pct": v2_12_canary_pct.value,
                "stage_hours": v2_12_stage_hours.value,
                "sample_needed": v2_12_sample_needed.value,
                "mttd_min": v2_12_mttd_min.value,
                "diagnosis_min": v2_12_diagnosis_min.value,
                "mitigation_pct": v2_12_mitigation_pct.value,
                "recovery_min": v2_12_recovery_min.value,
                "rollout_aggression": v2_12_rollout_aggression.value,
                "automation_level": v2_12_automation_level.value,
                "telemetry_depth": v2_12_telemetry_depth.value,
                "governance_level": v2_12_governance_level.value,
            },
            binding_constraints={
                "part_a_binding": v2_12_a["binding"],
                "part_b_evidence_ok": v2_12_b["evidence_ok"],
                "part_b_blast_ok": v2_12_b["blast_ok"],
                "part_c_lost_work_ratio": round(v2_12_c["ratio"], 4),
                "part_d_binding_guardrail": binding_ops_amount,
            },
            decisions={
                "part_a_checkpoint": v2_12_partA_checkpoint.value,
                "part_b_checkpoint": v2_12_partB_checkpoint.value,
                "part_c_checkpoint": v2_12_partC_checkpoint.value,
                "part_d_checkpoint": v2_12_partD_checkpoint.value,
                "selected_rollout_incident_policy": v2_12_selected_policy["name"],
                "rejected_alternative": v2_12_rejected_policy_result["name"],
                "v2_13_security_implication": security_text,
            },
            reflections={"memo_note": v2_12_memo_note.value or "Not recorded."},
            residual_risk=(
                "Teaching estimates should be calibrated against production traces, incident records, "
                "governance policy, current threat model, and live traffic distributions before use."
            ),
            evidence_summary={
                "binding_ops_amount": binding_ops_amount,
                "part_a_budget": f"{v2_12_a['binding']} at {v2_12_a['binding_ratio']:.2f}x budget",
                "part_b_rollout": f"{v2_12_canary_pct.value}% canary, required {v2_12_fmt_hours(v2_12_b['required_hours'])}",
                "part_c_lost_work": f"{v2_12_fmt_number(v2_12_c['lost_work'])} {v2_12_packet['traffic_unit']}",
                "part_d_feasible": v2_12_selected_policy["feasible"],
            },
            final_decision={
                "selected_policy": v2_12_selected_policy["name"],
                "binding_ops_amount": binding_ops_amount,
                "rejected_alternative": v2_12_rejected_policy_result["name"],
                "v2_13_security_implication": security_text,
            },
            big_takeaways=(
                "Operations at scale spends reliability budget over time.",
                "Canaries buy evidence by spending bounded blast radius.",
                "Incident response is a lost-work budget, not just a page.",
                "A launch policy is valid only when every guardrail passes.",
            ),
            source_trace={
                "book_anchor": v2_12_metadata.book_anchor,
                "formulas": (
                    "error_budget_minutes = period_minutes * (1 - SLO)",
                    "T_stage = n_samples_needed / (request_rate * p_stage)",
                    "lost_work = affected_units_per_min * impact_fraction * minutes",
                    "policy_feasible = slo_ok and blast_ok and cost_ok and governance_ok",
                ),
                "track_source": v2_12_packet["source_policy"],
            },
            result_snapshot=snapshot,
            incomplete_fields=tuple(incomplete),
        )
        if not incomplete:
            ledger.save(
                chapter=v2_12_chapter,
                design={
                    "lab_id": v2_12_metadata.lab_id,
                    "track_id": v2_12_profile.track_id,
                    "scenario_id": v2_12_variant.scenario_id,
                    "selected_rollout_incident_policy": v2_12_selected_policy["name"],
                    "binding_ops_amount": binding_ops_amount,
                    "rejected_alternative": v2_12_rejected_policy_result["name"],
                    "v2_13_security_implication": security_text,
                    "policy_feasible": v2_12_selected_policy["feasible"],
                    "result_snapshot": snapshot,
                },
            )
        status = "SAVED" if not incomplete else "INCOMPLETE"
        status_kind = "success" if not incomplete else "warn"
        return mo.vstack(
            [
                mo.md(
                    f"""
                    ### Operations-at-Scale Memo

                    **Report frame:** {v2_12_packet['report_frame']}

                    **Selected rollout/incident policy:** `{v2_12_selected_policy['name']}`

                    **Binding ops amount:** `{binding_ops_amount}`

                    **Rejected alternative:** `{v2_12_rejected_policy_result['name']}`

                    **V2-13 implication:** {security_text}
                    """
                ),
                mo.hstack([v2_12_student_id, v2_12_security_implication], widths="equal"),
                v2_12_memo_note,
                mo.callout(
                    mo.md(
                        f"**Status:** {status}. "
                        + ("Complete all predictions, checkpoints, policy choices, and the V2-13 implication before final save." if incomplete else "Ledger snapshot saved for downstream labs.")
                    ),
                    kind=status_kind,
                ),
                report_export_panel(report),
            ]
        )

    v2_12_tabs = mo.ui.tabs(
        {
            "Part A - SLO Budget": v2_12_build_part_a(),
            "Part B - Canary Radius": v2_12_build_part_b(),
            "Part C - Incident Budget": v2_12_build_part_c(),
            "Part D - Ops Policy": v2_12_build_part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    v2_12_tabs
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    mo,
    v2_12_a,
    v2_12_b,
    v2_12_c,
    v2_12_fmt_hours,
    v2_12_fmt_number,
    v2_12_guardrail_label,
    v2_12_packet,
    v2_12_profile,
    v2_12_selected_policy,
):
    _complete = v2_12_selected_policy["feasible"] and v2_12_a["ok"] and v2_12_b["evidence_ok"] and v2_12_b["blast_ok"] and v2_12_c["ok"]
    _status = "POLICY PASS" if _complete else "BOUNDARY ACTIVE"
    _status_color = COLORS["GreenLine"] if _complete else COLORS["OrangeLine"]
    mo.Html(
        f"""
        <div class="lab-hud">
            <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab 12</span></div>
            <div><span class="hud-label">TRACK</span> <span class="hud-value">{v2_12_profile.label}</span></div>
            <div><span class="hud-label">PART A</span> <span class="hud-value">{v2_12_a['binding']}</span></div>
            <div><span class="hud-label">CANARY</span> <span class="hud-value">{v2_12_fmt_hours(v2_12_b['required_hours'])}</span></div>
            <div><span class="hud-label">LOST WORK</span> <span class="hud-value">{v2_12_fmt_number(v2_12_c['lost_work'])} {v2_12_packet['traffic_unit']}</span></div>
            <div><span class="hud-label">POLICY</span> <span class="hud-value">{v2_12_guardrail_label(v2_12_selected_policy['binding'])}</span></div>
            <div><span class="hud-label">STATUS</span> <span style="color:{_status_color}; font-family:var(--font-mono);">{_status}</span></div>
        </div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
