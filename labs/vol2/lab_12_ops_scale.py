import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 12: The Silent Fleet · MLSysBook")


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
        MathPeek,
        big_takeaways,
        build_lab_report,
        gated_hypothesis_card,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        instrumentation_console,
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
        MathPeek,
        apply_plotly_theme,
        big_takeaways,
        build_lab_report,
        gated_hypothesis_card,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        html_lib,
        instrumentation_console,
        ledger,
        math,
        mo,
        pd,
        report_export_panel,
    )


@app.cell
def _(get_lab_metadata):
    v2_12_lab_path = "vol2/lab_12_ops_scale.py"
    v2_12_chapter = 12
    v2_12_metadata = get_lab_metadata(v2_12_lab_path)
    return v2_12_chapter, v2_12_metadata


@app.cell(hide_code=True)
def _(mo):
    v2_12_track_picker = mo.ui.dropdown(
        options={
            "⚡ TinyML Track (ARM Cortex-M55 / ESP32-S3 & Remote Firmware OTA Blast Radius)": "oura_ring",
            "📱 Mobile Track (Apple Silicon / Snapdragon & Staged App Store Rollouts vs Instant Drift)": "iphone",
            "🤖 Edge & Embodied Track (NVIDIA Jetson AGX Orin & Geofenced Fleet Safety Control Loops)": "robotaxi",
            "☁️ Cloud Supercomputing Track (H100/B200 Clusters & Canary Traffic vs Error Budget)": "cloud_fleet",
        },
        value="☁️ Cloud Supercomputing Track (H100/B200 Clusters & Canary Traffic vs Error Budget)",
        label="Select Course / Industry Track",
    )
    v2_12_track_picker
    return (v2_12_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    v2_12_metadata,
    v2_12_track_picker,
):
    v2_12_track_id = v2_12_track_picker.value
    v2_12_profile = get_track_profile(v2_12_track_id)
    v2_12_variant = get_lab_track_variant(v2_12_metadata.lab_id, v2_12_track_id)
    return v2_12_profile, v2_12_variant


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


@app.cell(hide_code=True)
def _(ACADEMIC_LAB_CSS, mo, v2_12_packet, v2_12_profile, v2_12_variant):
    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell">
      <div class="mlsysbook-lab-header" style="--mlsysbook-accent: #A51C30;">
        <div class="mlsysbook-meta">
          ML SYSTEMS TEXTBOOK &middot; VOLUME II &middot; CHAPTER 12 &middot; LAB 12
        </div>
        <h1 style="margin: 8px 0 4px 0; color: #0F172A; font-weight: 800; font-size: 1.85rem; letter-spacing: -0.02em;">
          Operations at Scale: Error Budgets, Canary Rollouts &amp; Incident Control Loops
        </h1>
        <p style="margin: 0 0 14px 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">
          Transform production operations from reactive firefighting into quantifiable control loops: spend error budgets across availability and quality dimensions,
          balance canary statistical learning against blast-radius exposure, structure incident response to bound lost work, and enforce conjunctive multi-guardrail release gates.
        </p>
        <div class="mlsysbook-chip-row" style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;">
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Track:</strong> {v2_12_profile.label}
          </span>
          <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155;">
            <strong>Stakeholder:</strong> {v2_12_variant.stakeholder}
          </span>
          <span class="mlsysbook-chip" style="background: #F8FAFC; color: #475569;">
            <strong>Hardware:</strong> {v2_12_packet['hardware_ref']}
          </span>
          <span class="mlsysbook-chip" style="background: #F8FAFC; color: #475569;">
            <strong>Ops Unit:</strong> {v2_12_packet['ops_unit']}
          </span>
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Primary Metric:</strong> {v2_12_variant.primary_metric}
          </span>
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Guardrail:</strong> {v2_12_variant.guardrail_metric}
          </span>
        </div>
      </div>

      <div class="mlsysbook-panel" style="margin-bottom: 20px;">
        <h3 style="margin: 0 0 8px 0; color: #0F172A; font-size: 1.15rem;">
          System Scenario: {v2_12_profile.label} Production Operations Envelope
        </h3>
        <p style="margin: 0 0 12px 0; font-size: 0.92rem; color: #334155; line-height: 1.55;">
          {v2_12_variant.workload_summary} Operating machine learning systems at scale requires treating reliability not as an aspirational binary, but as an amount system where availability, semantic accuracy, and incident recovery draw from finite budgets. Fast rollouts gather statistical evidence quickly but risk catastrophic blast radius before automated rollbacks engage; slow rollouts protect users but stall deployment velocity and mask latent distribution drift.
        </p>
        <div style="background: #F8FAFC; border-left: 4px solid #006395; padding: 12px 16px; border-radius: 4px; font-size: 0.9rem; color: #1E293B;">
          <strong>The Architectural Invariants of Operations at Scale:</strong>
          <ul class="mlsysbook-list" style="margin: 8px 0 4px 0;">
            <li><strong>The Error-Budget Amount System:</strong> Reliability is a consumable resource spent over time (<i>B</i><sub>error</sub> = <i>T</i><sub>period</sub> &middot; (1 &minus; SLO)). Deployments and semantic drift spend availability minutes and quality points against hard operational ceilings.</li>
            <li><strong>Canary Learning vs. Blast Radius:</strong> Canary duration trades statistical power against user exposure: <i>T</i><sub>stage</sub> = <i>N</i><sub>samples</sub> / (<i>R</i><sub>traffic</sub> &middot; <i>p</i><sub>canary</sub>). A tiny canary fails to detect regressions before promotion; an oversized canary maximizes incident blast radius.</li>
            <li><strong>The Lost-Work Incident Flow Invariant:</strong> Total incident damage is governed by detection latency, diagnostic attribution, and mitigation efficacy: <i>W</i><sub>lost</sub> = <i>R</i><sub>affected</sub> &middot; <i>f</i><sub>impact</sub> &middot; (MTTD + <i>T</i><sub>diag</sub>) + <i>R</i><sub>affected</sub> &middot; <i>f</i><sub>residual</sub> &middot; <i>T</i><sub>recovery</sub>.</li>
            <li><strong>Conjunctive Operations Guardrail Bundle:</strong> Self-service deployment policies cannot trade off safety dimensions via weighted averages: Launchable = SLO<sub>pass</sub> &and; Blast<sub>pass</sub> &and; Cost<sub>pass</sub> &and; Governance<sub>pass</sub>.</li>
          </ul>
        </div>
      </div>
    </div>
    """)
    mo.vstack([ACADEMIC_LAB_CSS, header_html])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v2_12_packet, v2_12_profile):
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
            <ul class="mlsysbook-list" style="margin: 0; font-size: 0.9rem; color: {COLORS['TextSec']};">
                <li><strong>Manage error budgets as amount systems:</strong> calculate availability minutes, quality drift points, and incident allowances for {v2_12_profile.label}.</li>
                <li><strong>Balance canary learning against blast radius:</strong> derive the stage duration required for statistical significance without exceeding blast limits.</li>
                <li><strong>Execute typed incident runbooks:</strong> structure detection, attribution, mitigation, and recovery to minimize lost work during silent failures.</li>
                <li><strong>Enforce conjunctive release guardrails:</strong> authorize production policies that satisfy SLO, blast radius, cost, and governance boundaries simultaneously.</li>
            </ul>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    SLOs &amp; SLAs &middot; Canary analysis &middot; Incident lifecycle &middot; Conjunctive guardrails
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~50 min</strong><br/>
                    A: 10 &middot; B: 10 &middot; C: 15 &middot; D: 15 min
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
                &ldquo;When operating {v2_12_packet['ops_unit']} at scale, which operational constraint
                binds first: availability error budget, statistical canary duration, incident lost work,
                or multi-guardrail release compliance?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(f"""
    <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-bottom: 20px;">
        <h4 style="margin: 0 0 8px 0; color: #0F172A; font-size: 1.05rem;">
            Recommended Reading &mdash; Complete before this lab:
        </h4>
        <ul class="mlsysbook-list" style="margin: 0; font-size: 0.9rem; color: #334155;">
            <li><strong>Reliability Engineering &amp; SLOs:</strong> error-budget amount systems, burn rates, availability vs. semantic quality trade-offs.</li>
            <li><strong>Continuous Delivery &amp; Progressive Rollouts:</strong> statistical canary sizing, metric evaluation windows, automated rollback triggers.</li>
            <li><strong>Incident Response &amp; Runbooks:</strong> MTTD, MTTR, telemetry localization, attribution, and blast-radius containment.</li>
            <li><strong>Self-Service Deployment Policies:</strong> guardrail conjunctions, governance levels, and operational compliance audit trails.</li>
        </ul>
    </div>
    """)
    return


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
def _(v2_12_period_minutes):
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
        v2_12_rejected_policy_result,
        v2_12_selected_policy,
    )


@app.cell
def _(
    COLORS,
    apply_plotly_theme,
    go,
    pd,
    v2_12_fmt_hours,
    v2_12_fmt_number,
    v2_12_fmt_pct,
    v2_12_guardrail_badge,
):
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
        v2_12_incident_fig,
        v2_12_incident_table,
        v2_12_policy_fig,
        v2_12_policy_table,
    )


@app.cell(hide_code=True)
def _(
    COLORS,
    MathPeek,
    big_takeaways,
    build_lab_report,
    gated_hypothesis_card,
    instrumentation_console,
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
    v2_12_d_policies,
    v2_12_detection_min,
    v2_12_diagnosis_min,
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
    v2_12_variant,
):
    def v2_12_feedback(predicted, actual, labels):
        kind, message = v2_12_prediction_feedback(predicted, actual, labels)
        return mo.callout(mo.md(message), kind=kind)

    def v2_12_status_callout(ok, success, failure):
        return mo.callout(mo.md(success if ok else failure), kind="success" if ok else "danger")

    def build_part_a():
        labels = {
            "availability": "availability error-budget minutes",
            "quality": "quality/drift points",
            "incident_count": "incident count",
        }
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Mission Scenario &middot; {v2_12_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;A release is ready for {v2_12_packet['ops_unit']}, but our reliability contract
                    requires translating availability targets and semantic quality into consumable error budgets
                    before exposing production traffic.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; {v2_12_variant.stakeholder} &middot; {v2_12_profile.label}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_12_partA_pred,
                title="1. Formulate SLO & Error-Budget Hypothesis",
                subtitle=(
                    f"Scenario: You are the {v2_12_variant.stakeholder}. Predict which operations amount "
                    "(availability error-budget minutes, quality/drift points, or incident count) "
                    "will bind first under production operating conditions."
                ),
            ),
        ]
        if v2_12_partA_pred.value is None:
            return mo.vstack(items)

        items.extend(
            [
                v2_12_feedback(v2_12_partA_pred.value, v2_12_a["binding_key"], labels),
                instrumentation_console(
                    mo.vstack([
                        mo.hstack([v2_12_slo_pct, v2_12_quality_floor, v2_12_incident_count], widths="equal"),
                        mo.hstack([v2_12_detection_min, v2_12_impact_min], widths="equal"),
                    ]),
                    title="SLO & Error-Budget Parameters",
                    subtitle=f"Tune availability target, quality floor, and incident parameters for {v2_12_profile.label}",
                ),
                v2_12_budget_fig(v2_12_a),
                v2_12_budget_table(v2_12_a),
                v2_12_status_callout(
                    v2_12_a["ok"],
                    f"Recovered envelope. The current release stays within all three tracked operation amounts; binding amount: `{v2_12_a['binding']}` at {v2_12_a['binding_ratio']:.2f}x budget.",
                    f"Budget violation. `{v2_12_a['binding']}` is overspent at {v2_12_a['binding_ratio']:.2f}x budget. Reduce incidents, shorten detection, widen the SLO budget, or tighten rollout exposure before promotion.",
                ),
                mo.Html(f"""
                <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                    <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part A Error-Budget Commitment Decision</h4>
                    <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                        Status: binding amount is <code>{v2_12_a['binding']}</code> consuming <code>{v2_12_a['binding_ratio']:.2f}x</code> of budget.
                        Commit your checkpoint decision before advancing to live canary exposure:
                    </p>
                    {v2_12_partA_checkpoint}
                </div>
                """),
                MathPeek(
                    r"B_{\text{error}} = T_{\text{period}} \cdot (1 - \text{SLO}), \quad S_{\text{impact}} = N_{\text{inc}} \cdot (T_{\text{detect}} + T_{\text{impact}})",
                    {
                        "period duration": f"{v2_12_packet['period_days']} days ({v2_12_fmt_minutes(v2_12_a['period_minutes'])})",
                        "availability SLO": f"{v2_12_slo_pct.value:.2f}%",
                        "allowed error budget": v2_12_fmt_minutes(v2_12_a['error_budget_minutes']),
                        "incident count": f"{v2_12_incident_count.value} incidents",
                        "total impact time": v2_12_fmt_minutes(v2_12_a['impact_minutes']),
                        "quality budget": f"{v2_12_a['quality_budget_pp']:.2f} pp",
                        "quality spend": f"{v2_12_a['quality_spend_pp']:.2f} pp",
                        "chapter source": "Volume II, Chapter 12: Reliability Engineering & Error Budgets",
                    },
                ),
            ]
        )
        return mo.vstack(items)

    def build_part_b():
        labels = {
            "tiny_blind": "too small to learn inside the release window",
            "balanced": "balanced evidence and exposure",
            "aggressive_exposed": "too much blast-radius exposure",
        }
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Mission Scenario &middot; {v2_12_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;Moving from error-budget planning to live canary rollout on {v2_12_packet['release_unit']}.
                    We need statistically significant evidence that quality hasn't regressed before our release window closes,
                    without spending excess blast radius.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; {v2_12_variant.stakeholder} &middot; {v2_12_profile.label}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_12_partB_pred,
                title="2. Formulate Canary Learning vs. Blast-Radius Hypothesis",
                subtitle=(
                    f"Scenario: You are configuring canary exposure for {v2_12_packet['release_unit']}. "
                    "Predict how canary percentage trades statistical learning speed against user blast-radius exposure."
                ),
            ),
        ]
        if v2_12_partB_pred.value is None:
            return mo.vstack(items)

        items.extend(
            [
                v2_12_feedback(v2_12_partB_pred.value, v2_12_b["actual"], labels),
                instrumentation_console(
                    mo.vstack([
                        mo.hstack([v2_12_canary_pct, v2_12_stage_hours], widths="equal"),
                        mo.hstack([v2_12_sample_needed, v2_12_traffic_multiplier], widths="equal"),
                    ]),
                    title="Canary Traffic & Sample Size Controls",
                    subtitle=f"Tune canary traffic slice, evaluation window duration, and sample requirement for {v2_12_profile.label}",
                ),
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
                mo.Html(f"""
                <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                    <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part B Canary Promotion Gate Decision</h4>
                    <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                        Required duration: <code>{v2_12_fmt_hours(v2_12_b['required_hours'])}</code>;
                        blast radius: <code>{v2_12_fmt_number(v2_12_b['blast_units'])} {v2_12_packet['traffic_unit']}</code>.
                        Select your canary promotion policy:
                    </p>
                    {v2_12_partB_checkpoint}
                </div>
                """),
                MathPeek(
                    r"T_{\text{stage}} = \frac{N_{\text{needed}}}{R_{\text{traffic}} \cdot p_{\text{canary}}}, \quad U_{\text{blast}} = R_{\text{traffic}} \cdot p_{\text{canary}} \cdot T_{\text{actual}}",
                    {
                        "samples needed": f"{v2_12_sample_needed.value:,}",
                        "effective traffic rate": f"{v2_12_fmt_number(v2_12_b['traffic_rate'])} {v2_12_packet['traffic_unit']}/h",
                        "canary fraction": f"{v2_12_canary_pct.value / 100:.2f} ({v2_12_canary_pct.value}%)",
                        "required duration": v2_12_fmt_hours(v2_12_b['required_hours']),
                        "allocated duration": f"{v2_12_stage_hours.value:.2f} h",
                        "blast units spent": f"{v2_12_fmt_number(v2_12_b['blast_units'])} {v2_12_packet['traffic_unit']}",
                        "blast budget limit": f"{v2_12_fmt_number(v2_12_packet['blast_budget_units'])} {v2_12_packet['traffic_unit']}",
                        "chapter source": "Volume II, Chapter 12: Continuous Delivery & Progressive Rollouts",
                    },
                ),
            ]
        )
        return mo.vstack(items)

    def build_part_c():
        labels = {
            "restart_first": "restart serving first",
            "inspect_semantic": "inspect data/model-quality signals first",
            "rollback_first": "rollback before classifying",
            "wait_for_labels": "wait for more labels",
        }
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Incident Briefing &middot; {v2_12_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;An anomaly is surfacing in {v2_12_packet['quality_signal']} while infrastructure health
                    checks report green. We must enforce disciplined diagnostic ordering and contain lost-work budgets
                    before escalating or taking blunt mitigation steps.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; {v2_12_variant.stakeholder} &middot; {v2_12_profile.label}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_12_partC_pred,
                title="3. Formulate Incident Response & Lost-Work Hypothesis",
                subtitle=(
                    f"Scenario: Silent regression in {v2_12_packet['quality_signal']}. "
                    "Predict the initial response action that adheres to diagnostic order and minimizes lost work."
                ),
            ),
        ]
        if v2_12_partC_pred.value is None:
            return mo.vstack(items)

        items.extend(
            [
                v2_12_feedback(v2_12_partC_pred.value, "inspect_semantic", labels),
                instrumentation_console(
                    mo.vstack([
                        mo.hstack([v2_12_mttd_min, v2_12_diagnosis_min, v2_12_runbook_level], widths="equal"),
                        mo.hstack([v2_12_mitigation_pct, v2_12_recovery_min], widths="equal"),
                    ]),
                    title="Incident Lifecycle & Runbook Maturity Controls",
                    subtitle=f"Tune mean time to detect (MTTD), diagnostic latency, mitigation effectiveness, and runbook automation for {v2_12_profile.label}",
                ),
                v2_12_incident_fig(v2_12_packet, v2_12_mttd_min.value, v2_12_c, v2_12_recovery_min.value),
                v2_12_incident_table(v2_12_packet, v2_12_c),
                v2_12_status_callout(
                    v2_12_c["ok"] and v2_12_partC_pred.value == "inspect_semantic",
                    f"Incident response stays inside the lost-work budget and starts with ML semantic evidence. Lost work: `{v2_12_fmt_number(v2_12_c['lost_work'])}` {v2_12_packet['traffic_unit']}.",
                    f"Response boundary is unsafe. Lost work is `{v2_12_fmt_number(v2_12_c['lost_work'])}` {v2_12_packet['traffic_unit']} against a budget of `{v2_12_fmt_number(v2_12_c['budget'])}`, or the first action violates diagnostic order.",
                ),
                mo.Html(f"""
                <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                    <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part C Runbook Remediation Decision</h4>
                    <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                        Lost work: <code>{v2_12_fmt_number(v2_12_c['lost_work'])} {v2_12_packet['traffic_unit']}</code>
                        (budget: <code>{v2_12_fmt_number(v2_12_c['budget'])}</code>).
                        Select your post-incident runbook refinement action:
                    </p>
                    {v2_12_partC_checkpoint}
                </div>
                """),
                MathPeek(
                    r"W_{\text{lost}} = R \cdot f_{\text{impact}} \cdot (MTTD + T_{\text{diag}}) + R \cdot f_{\text{impact}} \cdot (1 - \eta_{\text{mit}}) \cdot T_{\text{rec}}",
                    {
                        "affected units/min (R)": f"{v2_12_fmt_number(v2_12_packet['affected_units_per_min'])} {v2_12_packet['traffic_unit']}/min",
                        "pre-mitigation impact": f"{v2_12_fmt_pct(v2_12_packet['incident_impact_fraction'] * 100)}",
                        "pre-mitigation duration": v2_12_fmt_minutes(v2_12_c['pre_mitigation_minutes']),
                        "mitigation efficiency": f"{v2_12_mitigation_pct.value}%",
                        "residual impact fraction": f"{v2_12_fmt_pct(v2_12_c['residual_fraction'] * 100)}",
                        "recovery duration": v2_12_fmt_minutes(v2_12_recovery_min.value),
                        "total lost work": f"{v2_12_fmt_number(v2_12_c['lost_work'])} {v2_12_packet['traffic_unit']}",
                        "chapter source": "Volume II, Chapter 12: Incident Lifecycle & Runbook Workflows",
                    },
                ),
            ]
        )
        return mo.vstack(items)

    def build_part_d():
        labels = {
            "slo": "SLO / error budget",
            "blast": "Blast radius",
            "cost": "Cost",
            "governance": "Governance",
        }
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Policy Authorization Briefing &middot; {v2_12_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;Deployments at scale must pass a conjunction of independent guardrails:
                    SLO budget, canary blast radius, serving cost, and governance review. If any single
                    boundary fails, the rollout is rejected.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; {v2_12_variant.stakeholder} &middot; {v2_12_profile.label}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_12_partD_pred,
                title="4. Formulate Conjunctive Operations Policy Hypothesis",
                subtitle=(
                    "Scenario: Compare candidate release policies against the production guardrail bundle. "
                    "Predict which constraint causes the naive fast-rollout policy to fail."
                ),
            ),
        ]
        if v2_12_partD_pred.value is None:
            return mo.vstack(items)

        items.extend(
            [
                v2_12_feedback(v2_12_partD_pred.value, v2_12_fast_policy["binding"], labels),
                instrumentation_console(
                    mo.vstack([
                        mo.hstack([v2_12_rollout_aggression, v2_12_automation_level], widths="equal"),
                        mo.hstack([v2_12_telemetry_depth, v2_12_governance_level], widths="equal"),
                    ]),
                    title="Operations Policy & Guardrail Knobs",
                    subtitle=f"Configure rollout aggressiveness, automated rollback readiness, telemetry depth, and governance tiers for {v2_12_profile.label}",
                ),
                v2_12_policy_fig(v2_12_d_policies),
                v2_12_policy_table(v2_12_d_policies, v2_12_guardrail_label),
                mo.hstack([v2_12_partD_policy_choice, v2_12_rejected_policy], widths="equal"),
                v2_12_status_callout(
                    v2_12_selected_policy["feasible"],
                    f"Selected policy passes all guardrails. Binding guardrail is `{v2_12_guardrail_label(v2_12_selected_policy['binding'])}`.",
                    f"Selected policy is not launchable. Binding guardrail is `{v2_12_guardrail_label(v2_12_selected_policy['binding'])}`; revise rollout, automation, telemetry, or governance.",
                ),
                mo.Html(f"""
                <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                    <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part D Guardrail Authorization Decision</h4>
                    <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                        Selected: <code>{v2_12_selected_policy['name']}</code> (Feasible: {v2_12_selected_policy['feasible']};
                        Binding: <code>{v2_12_guardrail_label(v2_12_selected_policy['binding'])}</code>).
                        Confirm your deployment authorization policy:
                    </p>
                    {v2_12_partD_checkpoint}
                </div>
                """),
                MathPeek(
                    r"\text{Launchable} = (\text{SLO}_{\text{spend}} \le 100\%) \land (\text{Blast} \le \text{Blast}_{\max}) \land (\text{Cost} \le \text{Cost}_{\max}) \land (\text{Gov} \ge \text{Gov}_{\min})",
                    {
                        "SLO spend check": f"{v2_12_fmt_pct(v2_12_selected_policy['slo_spend_pct'])} <= {v2_12_fmt_pct(v2_12_packet['slo_guardrail_limit_pct'])}",
                        "blast radius check": f"{v2_12_fmt_pct(v2_12_selected_policy['blast_pct'])} <= {v2_12_fmt_pct(v2_12_packet['max_blast_pct'])}",
                        "cost index check": f"{v2_12_selected_policy['cost_index']:.1f} <= {v2_12_packet['cost_limit_index']:.1f}",
                        "governance check": f"{v2_12_selected_policy['governance_score']:.2f} >= {v2_12_packet['governance_min_score']:.2f}",
                        "binding guardrail": v2_12_guardrail_label(v2_12_selected_policy['binding']),
                        "chapter source": "Volume II, Chapter 12: Self-Service Operations & Conjunction Policies",
                    },
                ),
            ]
        )
        return mo.vstack(items)

    def build_synthesis():
        items = []
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
                "selected_policy": v2_12_selected_policy["name"],
                "binding": binding_ops_amount,
                "feasible": v2_12_selected_policy["feasible"],
                "slo_spend_pct": round(v2_12_selected_policy["slo_spend_pct"], 4),
                "blast_pct": round(v2_12_selected_policy["blast_pct"], 4),
                "cost_index": round(v2_12_selected_policy["cost_index"], 4),
                "governance_score": round(v2_12_selected_policy["governance_score"], 4),
            },
            "v2_13_security_implication": security_text,
        }
        report = build_lab_report(
            v2_12_metadata,
            student_id=v2_12_student_id.value or "",
            track=v2_12_profile.label,
            scenario=v2_12_variant.workload_summary,
            learning_objectives=(
                "Manage error budgets as amount systems for the selected track.",
                "Balance canary statistical learning against blast-radius exposure.",
                "Structure incident response to minimize lost work during silent failures.",
                "Select and authorize an operations policy that satisfies conjunctive guardrails.",
            ),
            predictions={
                "part_a_error_budget_binding": v2_12_partA_pred.value,
                "part_b_canary_tradeoff": v2_12_partB_pred.value,
                "part_c_incident_first_action": v2_12_partC_pred.value,
                "part_d_guardrail_binding": v2_12_partD_pred.value,
            },
            knob_settings={
                "slo_pct": v2_12_slo_pct.value,
                "quality_floor": v2_12_quality_floor.value,
                "incident_count": v2_12_incident_count.value,
                "detection_min": v2_12_detection_min.value,
                "impact_min": v2_12_impact_min.value,
                "canary_pct": v2_12_canary_pct.value,
                "stage_hours": v2_12_stage_hours.value,
                "sample_needed": v2_12_sample_needed.value,
                "traffic_multiplier": v2_12_traffic_multiplier.value,
                "mttd_min": v2_12_mttd_min.value,
                "diagnosis_min": v2_12_diagnosis_min.value,
                "runbook_level": v2_12_runbook_level.value,
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
                "Canaries buy statistical evidence by spending bounded blast radius.",
                "Incident response is a lost-work budget, not just a reactive alert.",
                "A launch policy is valid only when every guardrail passes simultaneously.",
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

        items.append(mo.md("## Synthesis &mdash; ML Operations at Scale Policy Memo"))
        items.append(mo.Html(f"""
        <div class="mlsysbook-panel" style="border-left: 4px solid #1F407A; margin-top: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">STUDENT MEMO &amp; REFLECTIONS</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Operations-at-Scale Control Memo</h4>
            {v2_12_student_id}
            <div style="margin-top: 12px;">{v2_12_security_implication}</div>
            <div style="margin-top: 12px;">{v2_12_memo_note}</div>
        </div>
        """))

        items.append(mo.callout(
            mo.md(
                f"**Memo summary:** Selected `{v2_12_selected_policy['name']}` (binding constraint: `{binding_ops_amount}`); "
                f"rejected `{v2_12_rejected_policy_result['name']}`.  \n\n"
                f"**V2-13 implication:** {security_text}"
            ),
            kind=status_kind,
        ))
        items.append(mo.callout(
            mo.md(
                f"**Status:** {status}. "
                + ("Complete all predictions, checkpoints, policy choices, and the V2-13 implication before final save." if incomplete else "Ledger snapshot saved for downstream labs.")
            ),
            kind=status_kind,
        ))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="flex:1; min-width:220px; background:white; border:1px solid {COLORS['Border']};
                        border-radius:10px; padding:16px; border-top:3px solid {COLORS['GreenLine']};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']}; text-transform:uppercase;">
                    Selected rollout policy</div>
                <div style="font-size:1.05rem; font-weight:800; color:{COLORS['Text']}; margin-top:5px;">
                    {v2_12_selected_policy['name']}</div>
            </div>
            <div style="flex:1; min-width:220px; background:white; border:1px solid {COLORS['Border']};
                        border-radius:10px; padding:16px; border-top:3px solid {COLORS['OrangeLine']};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']}; text-transform:uppercase;">
                    Binding ops amount</div>
                <div style="font-size:1.05rem; font-weight:800; color:{COLORS['Text']}; margin-top:5px;">
                    {binding_ops_amount}</div>
            </div>
            <div style="flex:1; min-width:220px; background:white; border:1px solid {COLORS['Border']};
                        border-radius:10px; padding:16px; border-top:3px solid {COLORS['RedLine']};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']}; text-transform:uppercase;">
                    Rejected alternative</div>
                <div style="font-size:1.05rem; font-weight:800; color:{COLORS['Text']}; margin-top:5px;">
                    {v2_12_rejected_policy_result['name']}</div>
            </div>
        </div>
        """))

        items.append(big_takeaways([
            "Operations at scale spends reliability budget over time.",
            "Canaries buy statistical evidence by spending bounded blast radius.",
            "Incident response is a lost-work budget, not just a reactive alert.",
            "A launch policy is valid only when every guardrail passes simultaneously.",
        ]))

        items.append(mo.Html(f"""
        <div class="mlsysbook-panel" style="border-left: 4px solid #A51C30; margin-top: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">FINAL VERIFICATION &amp; SIGN-OFF</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Lead Production MLOps Architect Authorization</h4>
            <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #475569;">
                Confirm your production rollout policy, verify all four conjunctive guardrails (SLO, blast radius, cost, governance), and export the signed operations audit record.
            </p>
        </div>
        """))

        items.append(report_export_panel(report))

        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab V2-13: The Price of Privacy &amp; Security</strong> &mdash; Carry forward the
                    selected deployment policy, binding ops amount, and security implication. The next challenge is
                    securing these production pipelines against adversarial poisoning, model theft, and data leakage.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> Chapter 12 on ML Operations at Scale for control loop formalisms.<br/>
                    <strong>Build:</strong> TinyTorch canary analyzer &mdash; implement statistical difference-in-differences testing with automated rollback triggers.
                </div>
            </div>
        </div>
        """))

        return mo.vstack(items)

    v2_12_tabs = mo.ui.tabs(
        {
            "Part A - SLO Budget": build_part_a(),
            "Part B - Canary Radius": build_part_b(),
            "Part C - Incident Budget": build_part_c(),
            "Part D - Ops Policy": build_part_d(),
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
