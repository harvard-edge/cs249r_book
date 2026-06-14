import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


# ===========================================================================
# ZONE A: OPENING
# ===========================================================================


@app.cell
async def _():
    import html
    import math
    import marimo as mo
    import sys
    from pathlib import Path

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
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
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
        DesignLedger,
        apply_plotly_theme,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        html,
        ledger,
        math,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v2_16_chapter = 16
    v2_16_lab_path = "vol2/lab_16_responsible_ai.py"
    v2_16_metadata = get_lab_metadata(v2_16_lab_path)
    return v2_16_chapter, v2_16_lab_path, v2_16_metadata


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v2_16_track_picker = track_selector(default=_default_track)
    v2_16_track_picker
    return (v2_16_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v2_16_metadata,
    v2_16_track_picker,
):
    v2_16_track_id = v2_16_track_picker.value
    v2_16_profile = get_track_profile(v2_16_track_id)
    v2_16_variant = get_lab_track_variant(v2_16_metadata.lab_id, v2_16_profile.track_id)
    v2_16_hardware = resolve_mlsysim_ref(v2_16_variant.hardware_ref)
    v2_16_model = resolve_mlsysim_ref(v2_16_variant.model_ref)
    return (
        v2_16_hardware,
        v2_16_model,
        v2_16_profile,
        v2_16_track_id,
        v2_16_variant,
    )


# ===========================================================================
# ZONE B: NOTEBOOK-LOCAL V2-16 TEACHING MODELS
# ===========================================================================


@app.cell
def _(COLORS, apply_plotly_theme, go, html, math):
    def v2_16_first_label(options):
        return next(iter(options.keys()))

    def v2_16_option_label(options, value):
        for label, option_value in options.items():
            if option_value == value:
                return label
        return str(value)

    def v2_16_fmt(value, suffix="", digits=1):
        if isinstance(value, int):
            return f"{value:,}{suffix}"
        if abs(float(value)) >= 1000:
            return f"{float(value):,.0f}{suffix}"
        return f"{float(value):.{digits}f}{suffix}"

    def v2_16_metric_card(title, value, subtitle, color, strong=False):
        border = f"border-left:4px solid {color};" if strong else f"border-top:3px solid {color};"
        return f"""
        <div style="{border} background:white; border-radius:8px; padding:12px 14px;
                    min-width:185px; flex:1; box-shadow:0 1px 4px rgba(15,23,42,0.08);">
            <div style="font-size:0.68rem; font-weight:800; color:#64748b;
                        text-transform:uppercase; letter-spacing:0.08em;">{html.escape(str(title))}</div>
            <div style="font-size:1.25rem; font-weight:800; color:#0f172a; margin:4px 0;">
                {html.escape(str(value))}
            </div>
            <div style="font-size:0.78rem; color:#475467; line-height:1.35;">
                {html.escape(str(subtitle))}
            </div>
        </div>
        """

    def v2_16_html_table(headers, rows):
        head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
        body = ""
        for row in rows:
            body += "<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>"
        return f"""
        <div style="overflow-x:auto; margin:12px 0;">
          <table style="border-collapse:collapse; width:100%; font-size:0.88rem;">
            <thead><tr style="background:#f8fafc; color:#334155;">{head}</tr></thead>
            <tbody>{body}</tbody>
          </table>
        </div>
        """

    def v2_16_guardrail_callout(mo, violations, success_text, fail_prefix):
        if violations:
            return mo.callout(mo.md(f"**{fail_prefix}:** " + "; ".join(violations)), kind="danger")
        return mo.callout(mo.md(f"**Pass.** {success_text}"), kind="success")

    def v2_16_part_banner(mo, color, label, quote):
        return mo.Html(f"""
        <div style="border-left:4px solid {color}; background:white;
                    border-radius:0 8px 8px 0; padding:16px 22px; margin:12px 0;
                    box-shadow:0 1px 4px rgba(15,23,42,0.08);">
            <div style="font-size:0.72rem; font-weight:800; color:{color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                {html.escape(label)}
            </div>
            <div style="font-style:italic; color:#1e293b; line-height:1.6;">
                "{html.escape(quote)}"
            </div>
        </div>
        """)

    def v2_16_bar_figure(title, values, limit=None, ytitle="Amount"):
        labels = list(values.keys())
        ys = [float(value) for value in values.values()]
        palette = [COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["RedLine"], "#6D5BD0"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=ys, marker_color=palette[: len(labels)], opacity=0.9))
        if limit is not None:
            fig.add_hline(y=float(limit), line_dash="dash", line_color=COLORS["RedLine"], annotation_text="guardrail")
        fig.update_layout(
            title=title,
            height=340,
            yaxis=dict(title=ytitle, gridcolor="#eef2f7"),
            margin=dict(l=60, r=20, t=55, b=50),
        )
        apply_plotly_theme(fig)
        return fig

    def v2_16_track_packet(profile, variant, hardware, model):
        packets = {
            "iphone": {
                "mission": "privacy-safe mobile responsible release",
                "stakeholders": {
                    "accessibility": {"label": "Accessibility-context users", "exposure_pct": 18.0},
                    "low_light": {"label": "Low-light or motion users", "exposure_pct": 24.0},
                    "opt_out": {"label": "Users who opt out of cloud telemetry", "exposure_pct": 35.0},
                },
                "harms": {
                    "false_negative": {"label": "missed helpful intervention", "severity": 1.0, "gap_delta": 1.0},
                    "false_positive": {"label": "unwanted intervention or lockout", "severity": 1.2, "gap_delta": 2.0},
                    "no_recourse": {"label": "no local explanation or appeal", "severity": 1.5, "gap_delta": 3.5},
                },
                "events_per_day": 25_000,
                "baseline_gap_pp": 12.0,
                "target_gap_pp": 5.0,
                "residual_cap_units": 220.0,
                "base_latency_ms": 30.0,
                "latency_slo_ms": 100.0,
                "cost_factor_limit": 2.0,
                "energy_factor_limit": 1.8,
                "release_delay_limit_days": 10.0,
                "min_audit_confidence_pct": 72.0,
                "escalation_limit_hours": 48.0,
                "audit_sample_default": 4.0,
                "label_default": 55.0,
                "monitoring_default": 55,
                "explanation_default": 25,
                "privacy_default": 70,
                "human_review_default": 20,
                "evidence_emphasis": "privacy-safe counters and accessibility regression",
                "failure_mode": "small cohorts disappear when telemetry is minimized",
                "report_frame": "mobile responsible release memo",
                "residual_owner_options": {
                    "No named owner": "none",
                    "Mobile RAI lead": "rai_lead",
                    "Product plus accessibility reviewer": "cross_functional",
                },
            },
            "oura_ring": {
                "mission": "health-adjacent wearable governance",
                "stakeholders": {
                    "sensor_contact": {"label": "Variable sensor-contact wearers", "exposure_pct": 22.0},
                    "physiology": {"label": "Rare physiology or activity cohorts", "exposure_pct": 14.0},
                    "privacy_sensitive": {"label": "Privacy-sensitive health users", "exposure_pct": 30.0},
                },
                "harms": {
                    "false_alarm": {"label": "false health-adjacent alarm", "severity": 1.3, "gap_delta": 2.0},
                    "missed_signal": {"label": "missed health-adjacent signal", "severity": 1.6, "gap_delta": 3.0},
                    "unclear_message": {"label": "unclear risk communication", "severity": 1.1, "gap_delta": 1.0},
                },
                "events_per_day": 1_440,
                "baseline_gap_pp": 14.0,
                "target_gap_pp": 6.0,
                "residual_cap_units": 18.0,
                "base_latency_ms": 80.0,
                "latency_slo_ms": 300.0,
                "cost_factor_limit": 1.7,
                "energy_factor_limit": 1.45,
                "release_delay_limit_days": 21.0,
                "min_audit_confidence_pct": 80.0,
                "escalation_limit_hours": 72.0,
                "audit_sample_default": 8.0,
                "label_default": 45.0,
                "monitoring_default": 45,
                "explanation_default": 10,
                "privacy_default": 80,
                "human_review_default": 25,
                "evidence_emphasis": "sensor-contact cohorts, false alerts, and battery-safe review",
                "failure_mode": "battery-safe monitoring under-samples rare physiology",
                "report_frame": "wearable health-adjacent governance memo",
                "residual_owner_options": {
                    "No named owner": "none",
                    "Wearable firmware RAI lead": "rai_lead",
                    "Clinical reviewer plus firmware owner": "cross_functional",
                },
            },
            "robotaxi": {
                "mission": "autonomy safety accountability",
                "stakeholders": {
                    "vru": {"label": "Vulnerable road users", "exposure_pct": 8.0},
                    "construction": {"label": "Construction and weather edge cases", "exposure_pct": 11.0},
                    "rider": {"label": "Riders exposed to unsafe fallback", "exposure_pct": 6.0},
                },
                "harms": {
                    "missed_hazard": {"label": "missed rare hazard", "severity": 2.0, "gap_delta": 2.0},
                    "unsafe_fallback": {"label": "unsafe or opaque fallback", "severity": 1.7, "gap_delta": 1.5},
                    "no_trace": {"label": "no reconstructable incident trace", "severity": 1.5, "gap_delta": 1.0},
                },
                "events_per_day": 2_000_000,
                "baseline_gap_pp": 8.0,
                "target_gap_pp": 2.0,
                "residual_cap_units": 420.0,
                "base_latency_ms": 20.0,
                "latency_slo_ms": 50.0,
                "cost_factor_limit": 3.0,
                "energy_factor_limit": 2.4,
                "release_delay_limit_days": 5.0,
                "min_audit_confidence_pct": 92.0,
                "escalation_limit_hours": 2.0,
                "audit_sample_default": 18.0,
                "label_default": 85.0,
                "monitoring_default": 70,
                "explanation_default": 40,
                "privacy_default": 35,
                "human_review_default": 65,
                "evidence_emphasis": "rare-event replay, trace review, fallback drills",
                "failure_mode": "long-tail replay coverage leaves safety-case blind spots",
                "report_frame": "safety-case governance memo",
                "residual_owner_options": {
                    "No named owner": "none",
                    "Autonomy safety owner": "rai_lead",
                    "Safety board plus fleet ops owner": "cross_functional",
                },
            },
            "cloud_fleet": {
                "mission": "population-scale responsible AI platform",
                "stakeholders": {
                    "tenant": {"label": "Small or underserved tenants", "exposure_pct": 16.0},
                    "language": {"label": "Underserved languages or regions", "exposure_pct": 20.0},
                    "appeal": {"label": "Users needing appeal or recourse", "exposure_pct": 12.0},
                },
                "harms": {
                    "aggregate_masking": {"label": "harm hidden by aggregate metrics", "severity": 1.4, "gap_delta": 2.0},
                    "delayed_label": {"label": "delayed labels hide drift", "severity": 1.2, "gap_delta": 1.0},
                    "appeal_backlog": {"label": "appeal or review backlog", "severity": 1.5, "gap_delta": 3.0},
                },
                "events_per_day": 10_000_000,
                "baseline_gap_pp": 16.0,
                "target_gap_pp": 5.0,
                "residual_cap_units": 8_000.0,
                "base_latency_ms": 50.0,
                "latency_slo_ms": 200.0,
                "cost_factor_limit": 3.0,
                "energy_factor_limit": 3.0,
                "release_delay_limit_days": 14.0,
                "min_audit_confidence_pct": 78.0,
                "escalation_limit_hours": 24.0,
                "audit_sample_default": 2.0,
                "label_default": 60.0,
                "monitoring_default": 60,
                "explanation_default": 20,
                "privacy_default": 50,
                "human_review_default": 35,
                "evidence_emphasis": "tenant/language cohorts, appeal workflow, delayed-label governance",
                "failure_mode": "aggregate dashboards pass while intersectional cohorts are uncovered",
                "report_frame": "population-scale responsible AI memo",
                "residual_owner_options": {
                    "No named owner": "none",
                    "Responsible AI platform owner": "rai_lead",
                    "RAI, SRE, and policy review owner": "cross_functional",
                },
            },
        }
        packet = dict(packets[profile.track_id])
        packet.update(
            {
                "track_id": profile.track_id,
                "track_label": profile.label,
                "stakeholder": variant.stakeholder,
                "scenario_id": variant.scenario_id,
                "scenario": variant.workload_summary,
                "hardware_ref": variant.hardware_ref,
                "hardware_name": getattr(hardware, "name", variant.hardware_ref),
                "model_ref": variant.model_ref,
                "model_name": getattr(model, "name", variant.model_ref),
                "source_policy": profile.source_policy,
            }
        )
        packet["stakeholder_options"] = {
            value["label"]: key for key, value in packet["stakeholders"].items()
        }
        packet["harm_options"] = {
            value["label"]: key for key, value in packet["harms"].items()
        }
        return packet

    def v2_16_obligation_result(packet, stakeholder_key, harm_key, threshold_pp):
        stakeholder = packet["stakeholders"][stakeholder_key]
        harm = packet["harms"][harm_key]
        exposure_units = packet["events_per_day"] * stakeholder["exposure_pct"] / 100.0
        observed_gap = packet["baseline_gap_pp"] + harm["gap_delta"]
        residual_gap = max(0.0, observed_gap - float(threshold_pp))
        affected_units = exposure_units * residual_gap / 100.0 * harm["severity"]
        threshold_ok = float(threshold_pp) <= packet["target_gap_pp"]
        residual_ok = affected_units <= packet["residual_cap_units"]
        violations = []
        if not threshold_ok:
            violations.append(
                f"allowed gap {threshold_pp:.1f} pp exceeds track target {packet['target_gap_pp']:.1f} pp"
            )
        if not residual_ok:
            violations.append(
                f"residual affected units {affected_units:,.0f}/day exceeds cap {packet['residual_cap_units']:,.0f}/day"
            )
        return {
            "stakeholder_key": stakeholder_key,
            "stakeholder_label": stakeholder["label"],
            "harm_key": harm_key,
            "harm_label": harm["label"],
            "exposure_pct": stakeholder["exposure_pct"],
            "exposure_units": exposure_units,
            "observed_gap_pp": observed_gap,
            "threshold_pp": float(threshold_pp),
            "residual_gap_pp": residual_gap,
            "affected_units_per_day": affected_units,
            "binding_amount": "allowed subgroup gap" if not threshold_ok else "residual affected units",
            "obligation_ok": not violations,
            "violations": tuple(violations),
        }

    def v2_16_overhead_result(packet, obligation, monitoring, explanation, privacy, human_review):
        monitoring = float(monitoring)
        explanation = float(explanation)
        privacy = float(privacy)
        human_review = float(human_review)
        monitor_added_ms = (10.0 + 10.0 * monitoring / 100.0) * monitoring / 100.0
        explanation_added_ms = packet["base_latency_ms"] * (0.6 + packet["baseline_gap_pp"] / 20.0) * explanation / 100.0
        privacy_added_ms = packet["base_latency_ms"] * privacy / 350.0
        review_added_ms = packet["base_latency_ms"] * human_review / 500.0
        latency_ms = packet["base_latency_ms"] + monitor_added_ms + explanation_added_ms + privacy_added_ms + review_added_ms
        cost_factor = 1.0 + monitoring / 260.0 + explanation / 210.0 + privacy / 300.0 + human_review / 180.0
        energy_factor = 1.0 + monitoring / 420.0 + explanation / 260.0 + privacy / 340.0 + human_review / 500.0
        release_delay_days = (
            1.0
            + packet["release_delay_limit_days"] * 0.22
            + monitoring / 55.0
            + explanation / 80.0
            + privacy / 95.0
            + human_review / 45.0
        )
        risk_reduction_pct = min(
            88.0,
            0.32 * monitoring + 0.20 * explanation + 0.18 * privacy + 0.30 * human_review,
        )
        residual_gap_pp = max(
            0.0,
            obligation["observed_gap_pp"]
            - 0.075 * monitoring
            - 0.030 * explanation
            - 0.020 * privacy
            - 0.055 * human_review,
        )
        quality_delta_pp = -0.014 * privacy - 0.006 * explanation + 0.012 * monitoring + 0.010 * human_review
        violations = []
        if latency_ms > packet["latency_slo_ms"]:
            violations.append(f"latency {latency_ms:.1f} ms exceeds SLO {packet['latency_slo_ms']:.1f} ms")
        if cost_factor > packet["cost_factor_limit"]:
            violations.append(f"cost factor {cost_factor:.2f}x exceeds limit {packet['cost_factor_limit']:.2f}x")
        if energy_factor > packet["energy_factor_limit"]:
            violations.append(f"energy factor {energy_factor:.2f}x exceeds limit {packet['energy_factor_limit']:.2f}x")
        if release_delay_days > packet["release_delay_limit_days"]:
            violations.append(
                f"release delay {release_delay_days:.1f} days exceeds limit {packet['release_delay_limit_days']:.1f} days"
            )
        if residual_gap_pp > packet["target_gap_pp"]:
            violations.append(
                f"residual gap {residual_gap_pp:.1f} pp remains above target {packet['target_gap_pp']:.1f} pp"
            )
        return {
            "monitoring_intensity_pct": monitoring,
            "explanation_coverage_pct": explanation,
            "privacy_strictness_pct": privacy,
            "human_review_share_pct": human_review,
            "latency_ms": latency_ms,
            "cost_factor": cost_factor,
            "energy_factor": energy_factor,
            "release_delay_days": release_delay_days,
            "risk_reduction_pct": risk_reduction_pct,
            "residual_gap_pp": residual_gap_pp,
            "quality_delta_pp": quality_delta_pp,
            "feasible": not violations,
            "violations": tuple(violations),
        }

    def v2_16_audit_result(packet, obligation, overhead, sample_rate, label_availability, slice_depth, escalation_hours):
        slice_profiles = {
            "single_axis": {"label": "Single-axis groups", "factor": 1.0, "blind_spot": "intersectional harm"},
            "intersectional": {"label": "Intersectional slices", "factor": 2.4, "blind_spot": "rare-event tail"},
            "long_tail": {"label": "Long-tail and stress slices", "factor": 4.2, "blind_spot": "delayed labels and sparse cohorts"},
        }
        slice_profile = slice_profiles[slice_depth]
        sample_rate = float(sample_rate)
        label_availability = float(label_availability)
        escalation_hours = float(escalation_hours)
        observable_share = min(
            0.98,
            (sample_rate / 100.0) * (label_availability / 100.0) * (3.0 / slice_profile["factor"]),
        )
        covered_units = obligation["exposure_units"] * observable_share
        latent_harm_units = obligation["exposure_units"] * max(overhead["residual_gap_pp"], 0.0) / 100.0
        blind_units = max(0.0, latent_harm_units * (1.0 - observable_share))
        escalation_score = min(100.0, 100.0 * packet["escalation_limit_hours"] / max(escalation_hours, 0.1))
        confidence = max(
            0.0,
            min(
                100.0,
                25.0
                + sample_rate * 1.4
                + label_availability * 0.38
                - (slice_profile["factor"] - 1.0) * 8.0
                + min(18.0, escalation_score / 5.0),
            ),
        )
        violations = []
        if confidence < packet["min_audit_confidence_pct"]:
            violations.append(
                f"audit confidence {confidence:.1f}% below floor {packet['min_audit_confidence_pct']:.1f}%"
            )
        if blind_units > packet["residual_cap_units"]:
            violations.append(
                f"blind residual harm {blind_units:,.0f}/day exceeds cap {packet['residual_cap_units']:,.0f}/day"
            )
        if escalation_hours > packet["escalation_limit_hours"]:
            violations.append(
                f"escalation {escalation_hours:.1f} h exceeds limit {packet['escalation_limit_hours']:.1f} h"
            )
        return {
            "sample_rate_pct": sample_rate,
            "label_availability_pct": label_availability,
            "slice_depth": slice_depth,
            "slice_label": slice_profile["label"],
            "blind_spot": slice_profile["blind_spot"],
            "observable_share_pct": observable_share * 100.0,
            "covered_units_per_day": covered_units,
            "blind_units_per_day": blind_units,
            "audit_confidence_pct": confidence,
            "escalation_hours": escalation_hours,
            "escalation_score_pct": escalation_score,
            "audit_ok": not violations,
            "violations": tuple(violations),
        }

    def v2_16_policy_result(packet, obligation, overhead, audit, selected_policy, rejected_policy, residual_owner):
        policies = {
            "monitor_only": {
                "label": "Monitor-and-patch",
                "utility_pct": 92.0,
                "harm_reduction_bonus": 4.0,
                "cost_multiplier": 0.88,
                "audit_bonus": -8.0,
                "governance": "weak escalation and delayed remediation",
            },
            "guardrail_first": {
                "label": "Guardrail-first responsible release",
                "utility_pct": 88.0,
                "harm_reduction_bonus": 13.0,
                "cost_multiplier": 1.00,
                "audit_bonus": 8.0,
                "governance": "release gate, owner, escalation, and residual obligation",
            },
            "max_mitigation": {
                "label": "Maximum mitigation before launch",
                "utility_pct": 82.0,
                "harm_reduction_bonus": 20.0,
                "cost_multiplier": 1.18,
                "audit_bonus": 14.0,
                "governance": "strong evidence but high overhead and launch delay",
            },
            "hold_redesign": {
                "label": "Hold for redesign",
                "utility_pct": 76.0,
                "harm_reduction_bonus": 24.0,
                "cost_multiplier": 1.05,
                "audit_bonus": 20.0,
                "governance": "safest when current controls cannot bind residual harm",
            },
        }
        selected = policies[selected_policy]
        rejected = policies[rejected_policy]
        residual_gap = max(0.0, overhead["residual_gap_pp"] - selected["harm_reduction_bonus"] / 4.0)
        technical_cost = overhead["cost_factor"] * selected["cost_multiplier"]
        technical_latency = overhead["latency_ms"] * (0.96 if selected_policy == "monitor_only" else 1.0)
        audit_confidence = min(100.0, audit["audit_confidence_pct"] + selected["audit_bonus"])
        owner_ok = residual_owner not in {None, "none"}
        violations = []
        if selected_policy == rejected_policy:
            violations.append("selected policy and rejected alternative are the same")
        if technical_latency > packet["latency_slo_ms"]:
            violations.append("technical latency guardrail fails")
        if technical_cost > packet["cost_factor_limit"]:
            violations.append("technical cost guardrail fails")
        if residual_gap > packet["target_gap_pp"]:
            violations.append("residual harm remains above target")
        if audit_confidence < packet["min_audit_confidence_pct"]:
            violations.append("audit coverage below governance floor")
        if not audit["audit_ok"]:
            violations.append("audit blind spot still violates Part C")
        if not owner_ok:
            violations.append("no residual obligation owner assigned")
        return {
            "selected_policy": selected_policy,
            "selected_policy_label": selected["label"],
            "rejected_policy": rejected_policy,
            "rejected_policy_label": rejected["label"],
            "utility_pct": selected["utility_pct"],
            "residual_gap_pp": residual_gap,
            "technical_latency_ms": technical_latency,
            "technical_cost_factor": technical_cost,
            "audit_confidence_pct": audit_confidence,
            "governance_summary": selected["governance"],
            "residual_owner": residual_owner,
            "policy_pass": not violations,
            "violations": tuple(violations),
            "policy_table": [
                [policy["label"], f"{policy['utility_pct']:.0f}%", f"+{policy['harm_reduction_bonus']:.0f}", f"{policy['cost_multiplier']:.2f}x", f"{policy['audit_bonus']:+.0f}", policy["governance"]]
                for policy in policies.values()
            ],
        }

    return (
        v2_16_audit_result,
        v2_16_bar_figure,
        v2_16_first_label,
        v2_16_fmt,
        v2_16_guardrail_callout,
        v2_16_html_table,
        v2_16_metric_card,
        v2_16_obligation_result,
        v2_16_option_label,
        v2_16_overhead_result,
        v2_16_part_banner,
        v2_16_policy_result,
        v2_16_track_packet,
    )


@app.cell
def _(v2_16_hardware, v2_16_model, v2_16_profile, v2_16_track_packet, v2_16_variant):
    v2_16_packet = v2_16_track_packet(v2_16_profile, v2_16_variant, v2_16_hardware, v2_16_model)
    return (v2_16_packet,)


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    COLORS,
    LAB_CSS,
    mo,
    source_trace,
    track_arc_context,
    track_context,
    v2_16_metadata,
    v2_16_packet,
    v2_16_profile,
    v2_16_variant,
):
    mo.vstack(
        [
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            mo.Html(f"""
            <div style="background: linear-gradient(135deg, #111827 0%, #263238 62%, #12211e 100%);
                        padding: 36px 44px; border-radius: 16px; color: white;
                        box-shadow: 0 8px 32px rgba(0,0,0,0.32);">
                <div style="font-size: 0.72rem; font-weight: 800; letter-spacing: 0.18em;
                            color: #A7F3D0; text-transform: uppercase; margin-bottom: 10px;">
                    Machine Learning Systems &middot; Volume II &middot; Lab 16
                </div>
                <h1 style="margin:0 0 10px 0; font-size:2.35rem; font-weight:900;
                           color:#f8fafc; line-height:1.1;">
                    Responsible Fleet Governance
                </h1>
                <p style="margin:0 0 7px 0; font-size:1.05rem; font-weight:700;
                          color:#cbd5e1; font-family:'SF Mono', monospace;">
                    Stakeholder Harm &middot; Evidence Overhead &middot; Audit Coverage &middot; Residual Obligation
                </p>
                <p style="margin:0 0 20px 0; max-width:820px; color:#e2e8f0; line-height:1.65;">
                    {v2_16_variant.workload_summary} The lab treats responsible AI as fleet
                    infrastructure: every harm needs a measurable obligation, evidence budget,
                    audit path, owner, and residual obligation.
                </p>
                <div style="display:flex; gap:12px; flex-wrap:wrap;">
                    <span style="background:rgba(16,185,129,0.14); color:#A7F3D0; border:1px solid rgba(16,185,129,0.25);
                                 border-radius:20px; padding:5px 14px; font-size:0.8rem; font-weight:700;">
                        4 Parts + Synthesis &middot; ~55 min
                    </span>
                    <span style="background:rgba(96,165,250,0.14); color:#BFDBFE; border:1px solid rgba(96,165,250,0.25);
                                 border-radius:20px; padding:5px 14px; font-size:0.8rem; font-weight:700;">
                        {v2_16_profile.label}
                    </span>
                    <span style="background:rgba(251,191,36,0.14); color:#FDE68A; border:1px solid rgba(251,191,36,0.25);
                                 border-radius:20px; padding:5px 14px; font-size:0.8rem; font-weight:700;">
                        {v2_16_packet["report_frame"]}
                    </span>
                </div>
            </div>
            """),
            track_context(v2_16_profile),
            track_arc_context(v2_16_profile, v2_16_metadata.lab_id),
            source_trace(
                {
                    "Chapter anchor": "Volume II Chapter 16, Responsible AI",
                    "Part framing": "Responsible Fleet Principles: fairness and safety guarantees are coordination constraints before compute runs",
                    "Track scenario": v2_16_variant.scenario_id,
                    "Hardware ref": v2_16_variant.hardware_ref,
                    "Model ref": v2_16_variant.model_ref,
                    "Notebook-local helpers": "v2_16_track_packet, v2_16_obligation_result, v2_16_overhead_result, v2_16_audit_result, v2_16_policy_result",
                    "Source policy": v2_16_profile.source_policy,
                },
                collapsed=True,
                summary="Chapter claims plus track/variant metadata feed the responsible fleet memo.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v2_16_packet):
    mo.Html(f"""
    <div style="border-left:4px solid {COLORS['GreenLine']}; background:white;
                border-radius:0 10px 10px 0; padding:20px 26px; margin:8px 0 16px 0;
                box-shadow:0 1px 4px rgba(15,23,42,0.08);">
      <div style="font-size:0.72rem; font-weight:800; color:#64748b;
                  text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px;">
        Learning Objectives
      </div>
      <div style="color:#334155; line-height:1.7; font-size:0.92rem;">
        <div>1. Convert stakeholder harm into a measurable track-specific obligation.</div>
        <div>2. Budget fairness, accountability, explanation, monitoring, privacy, and review overhead.</div>
        <div>3. Analyze audit coverage, blind spots, escalation, and residual harm at fleet scale.</div>
        <div>4. Choose and defend a responsible AI policy with a rejected alternative and V2-17 implication.</div>
      </div>
      <div style="margin-top:14px; padding-top:14px; border-top:1px solid #E2E8F0; color:#475467; line-height:1.55;">
        <strong>Track obligation:</strong> {v2_16_packet["evidence_emphasis"]}.<br>
        <strong>Natural failure:</strong> {v2_16_packet["failure_mode"]}.
      </div>
    </div>
    """)
    return


# ===========================================================================
# ZONE C: CONTROLS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v2_16_first_label, v2_16_packet):
    v2_16_partA_pred = mo.ui.radio(
        options={
            "A fairness metric is enough if it is documented.": "metric_only",
            "A named stakeholder harm must become a measurable obligation.": "harm_to_obligation",
            "The owner can be assigned after the first audit.": "owner_later",
            "A high aggregate score makes stakeholder harm unlikely.": "aggregate_quality",
        },
        label=f"Part A prediction: what blocks {v2_16_packet['track_label']} deployment first?",
    )
    v2_16_stakeholder = mo.ui.dropdown(
        options=v2_16_packet["stakeholder_options"],
        value=v2_16_first_label(v2_16_packet["stakeholder_options"]),
        label="Affected stakeholder",
    )
    v2_16_harm = mo.ui.dropdown(
        options=v2_16_packet["harm_options"],
        value=v2_16_first_label(v2_16_packet["harm_options"]),
        label="Primary harm mode",
    )
    v2_16_obligation_threshold = mo.ui.slider(
        start=max(0.5, v2_16_packet["target_gap_pp"] / 2.0),
        stop=v2_16_packet["target_gap_pp"] * 4.0,
        value=v2_16_packet["target_gap_pp"] * 1.4,
        step=0.5,
        label="Allowed obligation gap (percentage points)",
    )
    v2_16_partA_checkpoint = mo.ui.radio(
        options={
            "Bind release to the named stakeholder obligation.": "bind_obligation",
            "Ship if aggregate quality remains high.": "aggregate_quality",
            "Defer stakeholder mapping to postlaunch monitoring.": "defer_mapping",
        },
        label="Part A checkpoint",
    )
    return (
        v2_16_harm,
        v2_16_obligation_threshold,
        v2_16_partA_checkpoint,
        v2_16_partA_pred,
        v2_16_stakeholder,
    )


@app.cell(hide_code=True)
def _(mo, v2_16_packet):
    v2_16_partB_pred = mo.ui.radio(
        options={
            "The strongest governance controls always dominate.": "strongest_wins",
            "Evidence reduces harm but can violate latency, energy, cost, or delay budgets.": "overhead_binds",
            "Explanations are free if they run asynchronously.": "explanations_free",
            "Monitoring creates accountability without review capacity.": "monitoring_only",
        },
        label="Part B prediction: which statement is true about responsible evidence?",
    )
    v2_16_monitoring = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v2_16_packet["monitoring_default"]),
        step=5,
        label="Fairness/accountability monitoring intensity (%)",
    )
    v2_16_explanation = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v2_16_packet["explanation_default"]),
        step=5,
        label="Explanation coverage (%)",
    )
    v2_16_privacy = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v2_16_packet["privacy_default"]),
        step=5,
        label="Privacy/data-minimization strictness (%)",
    )
    v2_16_human_review = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v2_16_packet["human_review_default"]),
        step=5,
        label="Human review or escalation share (%)",
    )
    v2_16_partB_checkpoint = mo.ui.radio(
        options={
            "Use the strongest feasible evidence package.": "strongest_feasible",
            "Reduce evidence until all overhead disappears.": "remove_overhead",
            "Ignore overhead and let governance decide later.": "ignore_overhead",
        },
        label="Part B checkpoint",
    )
    return (
        v2_16_explanation,
        v2_16_human_review,
        v2_16_monitoring,
        v2_16_partB_checkpoint,
        v2_16_partB_pred,
        v2_16_privacy,
    )


@app.cell(hide_code=True)
def _(mo, v2_16_packet):
    v2_16_partC_pred = mo.ui.radio(
        options={
            "A dashboard is accountable once it shows a fairness metric.": "dashboard_enough",
            "Coverage, labels, slice depth, and escalation decide what harm is visible.": "coverage_path",
            "Intersectional audits are always better regardless of sample size.": "always_intersectional",
            "Escalation speed matters only after an incident happens.": "escalation_later",
        },
        label="Part C prediction: where does the audit blind spot come from?",
    )
    v2_16_audit_sample = mo.ui.slider(
        start=0.5,
        stop=80.0,
        value=float(v2_16_packet["audit_sample_default"]),
        step=0.5,
        label="Audit sample rate (% of exposed decisions)",
    )
    v2_16_label_availability = mo.ui.slider(
        start=0,
        stop=100,
        value=float(v2_16_packet["label_default"]),
        step=5,
        label="Ground-truth or appeal label availability (%)",
    )
    v2_16_slice_depth = mo.ui.dropdown(
        options={
            "Single-axis groups": "single_axis",
            "Intersectional slices": "intersectional",
            "Long-tail and stress slices": "long_tail",
        },
        value="Intersectional slices",
        label="Audit slice depth",
    )
    v2_16_escalation_hours = mo.ui.slider(
        start=0.5,
        stop=max(6.0, v2_16_packet["escalation_limit_hours"] * 4.0),
        value=float(v2_16_packet["escalation_limit_hours"] * 1.5),
        step=0.5,
        label="Escalation path time (hours)",
    )
    v2_16_partC_checkpoint = mo.ui.radio(
        options={
            "Escalate the named blind spot with a residual owner.": "escalate_blind_spot",
            "Accept the dashboard because aggregate coverage is high.": "accept_dashboard",
            "Wait for the next retraining cycle before acting.": "wait_retrain",
        },
        label="Part C checkpoint",
    )
    return (
        v2_16_audit_sample,
        v2_16_escalation_hours,
        v2_16_label_availability,
        v2_16_partC_checkpoint,
        v2_16_partC_pred,
        v2_16_slice_depth,
    )


@app.cell(hide_code=True)
def _(mo, v2_16_first_label, v2_16_packet):
    v2_16_policy_options = {
        "Monitor-and-patch": "monitor_only",
        "Guardrail-first responsible release": "guardrail_first",
        "Maximum mitigation before launch": "max_mitigation",
        "Hold for redesign": "hold_redesign",
    }
    v2_16_partD_pred = mo.ui.radio(
        options={
            "The policy with the most mitigation is always responsible.": "max_is_best",
            "The policy must satisfy technical, audit, governance, and residual-owner gates.": "conjunction",
            "A model card and dashboard are sufficient for accountability.": "docs_only",
            "The release board can accept a policy without a rejected alternative.": "no_rejection",
        },
        label="Part D prediction: which policy can ship?",
    )
    v2_16_selected_policy = mo.ui.dropdown(
        options=v2_16_policy_options,
        value="Guardrail-first responsible release",
        label="Selected responsible AI policy",
    )
    v2_16_rejected_policy = mo.ui.dropdown(
        options=v2_16_policy_options,
        value="Monitor-and-patch",
        label="Rejected alternative",
    )
    v2_16_residual_owner = mo.ui.radio(
        options=v2_16_packet["residual_owner_options"],
        label="Residual obligation owner",
    )
    v2_16_v2_17_implication = mo.ui.dropdown(
        options={
            "Treat this policy as a hard guardrail in the fleet synthesis.": "hard_guardrail",
            "Carry only the metric target into V2-17.": "metric_only",
            "Reopen deployment architecture if audit blind spots remain.": "reopen_architecture",
        },
        value="Treat this policy as a hard guardrail in the fleet synthesis.",
        label="V2-17 synthesis implication",
    )
    v2_16_student_id = mo.ui.text(label="Student identifier", placeholder="Optional")
    return (
        v2_16_partD_pred,
        v2_16_policy_options,
        v2_16_rejected_policy,
        v2_16_residual_owner,
        v2_16_selected_policy,
        v2_16_student_id,
        v2_16_v2_17_implication,
    )


@app.cell
def _(
    v2_16_audit_result,
    v2_16_audit_sample,
    v2_16_escalation_hours,
    v2_16_explanation,
    v2_16_harm,
    v2_16_human_review,
    v2_16_label_availability,
    v2_16_monitoring,
    v2_16_obligation_result,
    v2_16_obligation_threshold,
    v2_16_overhead_result,
    v2_16_packet,
    v2_16_policy_result,
    v2_16_privacy,
    v2_16_rejected_policy,
    v2_16_residual_owner,
    v2_16_selected_policy,
    v2_16_slice_depth,
    v2_16_stakeholder,
):
    v2_16_obligation = v2_16_obligation_result(
        v2_16_packet,
        v2_16_stakeholder.value,
        v2_16_harm.value,
        v2_16_obligation_threshold.value,
    )
    v2_16_overhead = v2_16_overhead_result(
        v2_16_packet,
        v2_16_obligation,
        v2_16_monitoring.value,
        v2_16_explanation.value,
        v2_16_privacy.value,
        v2_16_human_review.value,
    )
    v2_16_audit = v2_16_audit_result(
        v2_16_packet,
        v2_16_obligation,
        v2_16_overhead,
        v2_16_audit_sample.value,
        v2_16_label_availability.value,
        v2_16_slice_depth.value,
        v2_16_escalation_hours.value,
    )
    v2_16_policy = v2_16_policy_result(
        v2_16_packet,
        v2_16_obligation,
        v2_16_overhead,
        v2_16_audit,
        v2_16_selected_policy.value,
        v2_16_rejected_policy.value,
        v2_16_residual_owner.value,
    )
    return v2_16_audit, v2_16_obligation, v2_16_overhead, v2_16_policy


# ===========================================================================
# ZONE D: CONCEPT MODULES
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    mo,
    v2_16_bar_figure,
    v2_16_fmt,
    v2_16_guardrail_callout,
    v2_16_harm,
    v2_16_html_table,
    v2_16_metric_card,
    v2_16_obligation,
    v2_16_obligation_threshold,
    v2_16_packet,
    v2_16_partA_checkpoint,
    v2_16_partA_pred,
    v2_16_part_banner,
    v2_16_stakeholder,
):
    _items = [
        v2_16_part_banner(
            mo,
            COLORS["BlueLine"],
            f"Part A - Harm Becomes an Obligation - {v2_16_packet['stakeholder']}",
            f"Name the stakeholder harm before choosing a metric for {v2_16_packet['mission']}.",
        ),
        mo.md(f"""
## Part A: Harm Becomes a Measurable Obligation

**Scenario.** You are reviewing `{v2_16_packet["track_label"]}` deployment for
**{v2_16_packet["stakeholder"]}**. The release cannot rely on aggregate quality;
it must name who can be harmed, what amount is unacceptable, and what evidence
will prove the obligation is being met.

**Concept.** Responsible AI starts by converting stakeholder harm into a
measurable deployment gate. A fairness metric with no stakeholder, threshold, or
owner is only a label.
        """),
        v2_16_partA_pred,
    ]
    if v2_16_partA_pred.value is None:
        _items.append(mo.callout(mo.md("Select a prediction to unlock the stakeholder obligation model."), kind="warn"))
        mo.stop(True, mo.vstack(_items))

    _items.append(mo.hstack([v2_16_stakeholder, v2_16_harm, v2_16_obligation_threshold], widths="equal"))
    _fig = v2_16_bar_figure(
        "Stakeholder Harm Amounts",
        {
            "Exposed decisions/day": v2_16_obligation["exposure_units"],
            "Residual affected/day": v2_16_obligation["affected_units_per_day"],
        },
        limit=v2_16_packet["residual_cap_units"],
        ytitle="Decisions or affected units per day",
    )
    _items.append(mo.as_html(_fig))
    _items.append(
        mo.Html(
            f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
          {v2_16_metric_card("Stakeholder", v2_16_obligation["stakeholder_label"], f"{v2_16_obligation['exposure_pct']:.0f}% exposed", COLORS["BlueLine"])}
          {v2_16_metric_card("Harm Mode", v2_16_obligation["harm_label"], "track-specific failure consequence", COLORS["OrangeLine"])}
          {v2_16_metric_card("Allowed Gap", f"{v2_16_obligation['threshold_pp']:.1f} pp", f"target {v2_16_packet['target_gap_pp']:.1f} pp", COLORS["GreenLine"])}
          {v2_16_metric_card("Binding Amount", v2_16_obligation["binding_amount"], f"{v2_16_obligation['affected_units_per_day']:,.0f}/day residual", COLORS["RedLine"], True)}
        </div>
        """
        )
    )
    _table = v2_16_html_table(
        ["Quantity", "Value"],
        [
            ["Affected stakeholder", v2_16_obligation["stakeholder_label"]],
            ["Primary harm", v2_16_obligation["harm_label"]],
            ["Fleet events/day", f"{v2_16_packet['events_per_day']:,}"],
            ["Exposure share", f"{v2_16_obligation['exposure_pct']:.1f}%"],
            ["Observed gap", f"{v2_16_obligation['observed_gap_pp']:.1f} pp"],
            ["Allowed obligation gap", f"{v2_16_obligation['threshold_pp']:.1f} pp"],
            ["Residual affected units/day", f"{v2_16_obligation['affected_units_per_day']:,.1f}"],
        ],
    )
    _items.append(mo.Html(_table))
    _items.append(
        v2_16_guardrail_callout(
            mo,
            v2_16_obligation["violations"],
            "The stakeholder harm has a measurable obligation and can flow into the evidence budget.",
            "Obligation boundary hit",
        )
    )
    _items.append(
        mo.accordion(
            {
                "Math Peek / Source Model - stakeholder harm amount": mo.md(f"""
The lab converts harm into an amount:

`affected_units = events_per_day * exposure_share * max(0, observed_gap - allowed_gap) * severity`.

For this track, `events_per_day = {v2_16_packet['events_per_day']:,}` and the
target gap is `{v2_16_packet['target_gap_pp']:.1f} pp`. This follows the
chapter's claim that fairness and accountability must become verifiable system
properties, not generic principles.
                """)
            }
        )
    )
    if v2_16_partA_pred.value == "harm_to_obligation":
        _items.append(mo.callout(mo.md("**Correct.** A metric becomes responsible only after a stakeholder harm and threshold make it actionable."), kind="success"))
    else:
        _items.append(mo.callout(mo.md("**Revise the prior.** Aggregate quality and metric names do not say who is harmed or how much harm is allowed."), kind="warn"))
    _items.append(v2_16_partA_checkpoint)
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    mo,
    v2_16_bar_figure,
    v2_16_explanation,
    v2_16_guardrail_callout,
    v2_16_html_table,
    v2_16_human_review,
    v2_16_metric_card,
    v2_16_monitoring,
    v2_16_overhead,
    v2_16_packet,
    v2_16_partB_checkpoint,
    v2_16_partB_pred,
    v2_16_part_banner,
    v2_16_privacy,
):
    _items = [
        v2_16_part_banner(
            mo,
            COLORS["OrangeLine"],
            f"Part B - Evidence Consumes Capacity - {v2_16_packet['track_label']}",
            "The responsible controls that create evidence also consume deployment budgets.",
        ),
        mo.md("""
## Part B: Responsible Evidence Consumes Capacity

**Scenario.** The Part A obligation now needs monitoring, explanations, privacy
controls, and review capacity. The release board wants the strongest evidence
package that still fits the track's technical and governance guardrails.

**Concept.** Responsible AI overhead is an amount system: fairness monitoring,
explanation, privacy, and human review reduce risk while adding latency, cost,
energy, and release delay.
        """),
        v2_16_partB_pred,
    ]
    if v2_16_partB_pred.value is None:
        _items.append(mo.callout(mo.md("Select a prediction to unlock the overhead frontier."), kind="warn"))
        mo.stop(True, mo.vstack(_items))

    _items.append(mo.hstack([v2_16_monitoring, v2_16_explanation, v2_16_privacy, v2_16_human_review], widths="equal"))
    _fig = v2_16_bar_figure(
        "Responsible AI Overhead",
        {
            "Latency ms": v2_16_overhead["latency_ms"],
            "Risk reduction %": v2_16_overhead["risk_reduction_pct"],
            "Release delay days": v2_16_overhead["release_delay_days"],
        },
        limit=v2_16_packet["latency_slo_ms"],
        ytitle="Mixed units shown in table below",
    )
    _items.append(mo.as_html(_fig))
    _items.append(
        mo.Html(
            f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
          {v2_16_metric_card("Latency", f"{v2_16_overhead['latency_ms']:.1f} ms", f"SLO {v2_16_packet['latency_slo_ms']:.1f} ms", COLORS["BlueLine"], v2_16_overhead["latency_ms"] > v2_16_packet["latency_slo_ms"])}
          {v2_16_metric_card("Cost Factor", f"{v2_16_overhead['cost_factor']:.2f}x", f"limit {v2_16_packet['cost_factor_limit']:.2f}x", COLORS["OrangeLine"], v2_16_overhead["cost_factor"] > v2_16_packet["cost_factor_limit"])}
          {v2_16_metric_card("Energy Factor", f"{v2_16_overhead['energy_factor']:.2f}x", f"limit {v2_16_packet['energy_factor_limit']:.2f}x", COLORS["GreenLine"], v2_16_overhead["energy_factor"] > v2_16_packet["energy_factor_limit"])}
          {v2_16_metric_card("Residual Gap", f"{v2_16_overhead['residual_gap_pp']:.1f} pp", f"target {v2_16_packet['target_gap_pp']:.1f} pp", COLORS["RedLine"], v2_16_overhead["residual_gap_pp"] > v2_16_packet["target_gap_pp"])}
        </div>
        """
        )
    )
    _items.append(
        mo.Html(
            v2_16_html_table(
                ["Control or result", "Value"],
                [
                    ["Monitoring intensity", f"{v2_16_overhead['monitoring_intensity_pct']:.0f}%"],
                    ["Explanation coverage", f"{v2_16_overhead['explanation_coverage_pct']:.0f}%"],
                    ["Privacy strictness", f"{v2_16_overhead['privacy_strictness_pct']:.0f}%"],
                    ["Human review share", f"{v2_16_overhead['human_review_share_pct']:.0f}%"],
                    ["Risk reduction", f"{v2_16_overhead['risk_reduction_pct']:.1f}%"],
                    ["Quality delta", f"{v2_16_overhead['quality_delta_pp']:+.2f} pp"],
                    ["Release delay", f"{v2_16_overhead['release_delay_days']:.1f} days"],
                    ["Feasible", "yes" if v2_16_overhead["feasible"] else "no"],
                ],
            )
        )
    )
    _items.append(
        v2_16_guardrail_callout(
            mo,
            v2_16_overhead["violations"],
            "The evidence package fits the track's latency, cost, energy, delay, and residual-gap guardrails.",
            "Evidence overhead boundary hit",
        )
    )
    _items.append(
        mo.accordion(
            {
                "Math Peek / Source Model - responsible evidence overhead": mo.md(f"""
The chapter reports that real-time fairness monitoring adds about 10-20 ms,
approximate explanations add tens to hundreds of percent, privacy controls add
training and evidence cost, and human review adds routing capacity.

The notebook-local model combines:

`latency = base_latency + monitoring_ms + explanation_ms + privacy_ms + review_ms`

`cost_factor = 1 + monitoring + explanation + privacy + review terms`

`residual_gap = observed_gap - monitoring_relief - review_relief - privacy_relief`

The result is track-specific because `{v2_16_packet['track_label']}` has a
`{v2_16_packet['latency_slo_ms']:.1f} ms` latency SLO and a
`{v2_16_packet['release_delay_limit_days']:.1f} day` release-delay limit.
                """)
            }
        )
    )
    if v2_16_partB_pred.value == "overhead_binds":
        _items.append(mo.callout(mo.md("**Correct.** Responsible evidence can be necessary and still violate another budget."), kind="success"))
    else:
        _items.append(mo.callout(mo.md("**Revise the prior.** Stronger controls are not automatically deployable when they break the track envelope."), kind="warn"))
    _items.append(v2_16_partB_checkpoint)
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    mo,
    v2_16_audit,
    v2_16_audit_sample,
    v2_16_bar_figure,
    v2_16_escalation_hours,
    v2_16_guardrail_callout,
    v2_16_html_table,
    v2_16_label_availability,
    v2_16_metric_card,
    v2_16_packet,
    v2_16_partC_checkpoint,
    v2_16_partC_pred,
    v2_16_part_banner,
    v2_16_slice_depth,
):
    _items = [
        v2_16_part_banner(
            mo,
            COLORS["GreenLine"],
            f"Part C - Audit Coverage Determines Blind Spots - {v2_16_packet['track_label']}",
            "A dashboard is not accountability unless it covers the harm and triggers an owned response.",
        ),
        mo.md("""
## Part C: Audit Coverage Determines Blind Spots

**Scenario.** The governance team asks whether the audit can actually see the
harm named in Part A. You choose how much of the fleet is sampled, how many
labels arrive, how deep the cohort slicing goes, and how quickly escalation
acts.

**Concept.** Audit coverage is a measurable resource. Sparse labels,
intersectional slices, and slow escalation turn residual percentages into
unseen affected units.
        """),
        v2_16_partC_pred,
    ]
    if v2_16_partC_pred.value is None:
        _items.append(mo.callout(mo.md("Select a prediction to unlock the audit coverage model."), kind="warn"))
        mo.stop(True, mo.vstack(_items))

    _items.append(mo.hstack([v2_16_audit_sample, v2_16_label_availability, v2_16_slice_depth, v2_16_escalation_hours], widths="equal"))
    _fig = v2_16_bar_figure(
        "Audit Coverage And Blind Units",
        {
            "Covered units/day": v2_16_audit["covered_units_per_day"],
            "Blind residual/day": v2_16_audit["blind_units_per_day"],
        },
        limit=v2_16_packet["residual_cap_units"],
        ytitle="Units per day",
    )
    _items.append(mo.as_html(_fig))
    _items.append(
        mo.Html(
            f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
          {v2_16_metric_card("Observable Share", f"{v2_16_audit['observable_share_pct']:.2f}%", "sample x labels x slice burden", COLORS["BlueLine"])}
          {v2_16_metric_card("Audit Confidence", f"{v2_16_audit['audit_confidence_pct']:.1f}%", f"floor {v2_16_packet['min_audit_confidence_pct']:.1f}%", COLORS["GreenLine"], v2_16_audit["audit_confidence_pct"] < v2_16_packet["min_audit_confidence_pct"])}
          {v2_16_metric_card("Blind Spot", v2_16_audit["blind_spot"], v2_16_audit["slice_label"], COLORS["OrangeLine"])}
          {v2_16_metric_card("Escalation", f"{v2_16_audit['escalation_hours']:.1f} h", f"limit {v2_16_packet['escalation_limit_hours']:.1f} h", COLORS["RedLine"], v2_16_audit["escalation_hours"] > v2_16_packet["escalation_limit_hours"])}
        </div>
        """
        )
    )
    _items.append(
        mo.Html(
            v2_16_html_table(
                ["Quantity", "Value"],
                [
                    ["Audit sample rate", f"{v2_16_audit['sample_rate_pct']:.1f}%"],
                    ["Label availability", f"{v2_16_audit['label_availability_pct']:.1f}%"],
                    ["Slice depth", v2_16_audit["slice_label"]],
                    ["Observable share", f"{v2_16_audit['observable_share_pct']:.3f}%"],
                    ["Covered units/day", f"{v2_16_audit['covered_units_per_day']:,.1f}"],
                    ["Blind residual harm/day", f"{v2_16_audit['blind_units_per_day']:,.1f}"],
                    ["Escalation score", f"{v2_16_audit['escalation_score_pct']:.1f}%"],
                    ["Audit status", "pass" if v2_16_audit["audit_ok"] else "fail"],
                ],
            )
        )
    )
    _items.append(
        v2_16_guardrail_callout(
            mo,
            v2_16_audit["violations"],
            "Audit coverage, blind units, and escalation time are within the track's governance envelope.",
            "Audit coverage boundary hit",
        )
    )
    _items.append(
        mo.accordion(
            {
                "Math Peek / Source Model - coverage and blind spots": mo.md(f"""
Coverage is a product of sampling, label availability, and slice burden:

`observable_share = sample_rate * label_availability * (3 / slice_factor)`.

Blind harm scales with fleet volume:

`blind_units = exposed_units * residual_gap * (1 - observable_share)`.

This implements the chapter claim that dashboards do not create accountability
unless the relevant cohorts are observable and an escalation path can act.
                """)
            }
        )
    )
    if v2_16_partC_pred.value == "coverage_path":
        _items.append(mo.callout(mo.md("**Correct.** Accountability depends on coverage, labels, slice depth, and escalation ownership."), kind="success"))
    else:
        _items.append(mo.callout(mo.md("**Revise the prior.** A dashboard without coverage and response authority is evidence without action."), kind="warn"))
    _items.append(v2_16_partC_checkpoint)
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    mo,
    v2_16_bar_figure,
    v2_16_guardrail_callout,
    v2_16_html_table,
    v2_16_metric_card,
    v2_16_packet,
    v2_16_partD_pred,
    v2_16_part_banner,
    v2_16_policy,
    v2_16_policy_options,
    v2_16_rejected_policy,
    v2_16_residual_owner,
    v2_16_selected_policy,
):
    _items = [
        v2_16_part_banner(
            mo,
            COLORS["RedLine"],
            f"Part D - Policy Is a Guardrailed Design Decision - {v2_16_packet['track_label']}",
            "The release board needs one policy, one rejected alternative, and a residual owner.",
        ),
        mo.md("""
## Part D: Choose a Responsible AI Policy

**Scenario.** A release board asks for a policy that satisfies technical
guardrails and governance guardrails at the same time. The memo must also state
which alternative you rejected and who owns residual obligation.

**Concept.** Responsible policy is a conjunction, not a single score. It must
pass technical fit, audit coverage, governance readiness, and residual-owner
assignment.
        """),
        v2_16_partD_pred,
    ]
    if v2_16_partD_pred.value is None:
        _items.append(mo.callout(mo.md("Select a prediction to unlock the policy scorecard."), kind="warn"))
        mo.stop(True, mo.vstack(_items))

    _items.append(mo.hstack([v2_16_selected_policy, v2_16_rejected_policy, v2_16_residual_owner], widths="equal"))
    _fig = v2_16_bar_figure(
        "Selected Policy Guardrails",
        {
            "Utility %": v2_16_policy["utility_pct"],
            "Audit confidence %": v2_16_policy["audit_confidence_pct"],
            "Residual gap pp": v2_16_policy["residual_gap_pp"],
            "Cost factor x10": v2_16_policy["technical_cost_factor"] * 10.0,
        },
        limit=v2_16_packet["min_audit_confidence_pct"],
        ytitle="Score or scaled amount",
    )
    _items.append(mo.as_html(_fig))
    _items.append(
        mo.Html(
            f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
          {v2_16_metric_card("Selected", v2_16_policy["selected_policy_label"], v2_16_policy["governance_summary"], COLORS["BlueLine"])}
          {v2_16_metric_card("Rejected", v2_16_policy["rejected_policy_label"], "must be different and defensible", COLORS["OrangeLine"], v2_16_policy["selected_policy"] == v2_16_policy["rejected_policy"])}
          {v2_16_metric_card("Residual Gap", f"{v2_16_policy['residual_gap_pp']:.1f} pp", f"target {v2_16_packet['target_gap_pp']:.1f} pp", COLORS["GreenLine"], v2_16_policy["residual_gap_pp"] > v2_16_packet["target_gap_pp"])}
          {v2_16_metric_card("Policy Gate", "PASS" if v2_16_policy["policy_pass"] else "FAIL", "; ".join(v2_16_policy["violations"]) or "all guardrails pass", COLORS["RedLine"], True)}
        </div>
        """
        )
    )
    _items.append(
        mo.Html(
            v2_16_html_table(
                ["Policy", "Utility", "Harm relief", "Cost multiplier", "Audit bonus", "Governance profile"],
                v2_16_policy["policy_table"],
            )
        )
    )
    _items.append(
        v2_16_guardrail_callout(
            mo,
            v2_16_policy["violations"],
            "The selected policy passes the technical, audit, governance, and residual-owner gates.",
            "Policy gate failed",
        )
    )
    _items.append(
        mo.accordion(
            {
                "Math Peek / Source Model - policy conjunction": mo.md(f"""
The release condition is a conjunction:

`policy_pass = technical_ok and audit_ok and governance_ok and residual_owner_assigned`.

Technical gates check latency and cost. Governance gates check audit confidence,
blind spots, and residual ownership. This follows the chapter's fallacy warning:
one fairness metric or one dashboard does not make a fleet accountable.
                """)
            }
        )
    )
    if v2_16_partD_pred.value == "conjunction":
        _items.append(mo.callout(mo.md("**Correct.** A responsible policy must pass simultaneous technical and governance gates."), kind="success"))
    else:
        _items.append(mo.callout(mo.md("**Revise the prior.** The most aggressive mitigation, best document, or highest utility score can still fail another guardrail."), kind="warn"))
    mo.vstack(_items)
    return


# ===========================================================================
# ZONE E: SYNTHESIS, LEDGER, REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    ledger,
    mo,
    v2_16_audit,
    v2_16_chapter,
    v2_16_obligation,
    v2_16_overhead,
    v2_16_packet,
    v2_16_partA_checkpoint,
    v2_16_partA_pred,
    v2_16_partB_checkpoint,
    v2_16_partB_pred,
    v2_16_partC_checkpoint,
    v2_16_partC_pred,
    v2_16_partD_pred,
    v2_16_policy,
    v2_16_profile,
    v2_16_v2_17_implication,
    v2_16_variant,
):
    _required_values = (
        v2_16_partA_pred.value,
        v2_16_partA_checkpoint.value,
        v2_16_partB_pred.value,
        v2_16_partB_checkpoint.value,
        v2_16_partC_pred.value,
        v2_16_partC_checkpoint.value,
        v2_16_partD_pred.value,
        v2_16_policy["residual_owner"],
    )
    _complete = all(value is not None for value in _required_values) and v2_16_policy["residual_owner"] != "none"
    if v2_16_partA_pred.value is not None and v2_16_partD_pred.value is not None:
        ledger.save(
            track=v2_16_profile.track_id,
            chapter=v2_16_chapter,
            design={
                "chapter": "v2_16",
                "track_id": v2_16_profile.track_id,
                "scenario_id": v2_16_variant.scenario_id,
                "hardware_ref": v2_16_variant.hardware_ref,
                "model_ref": v2_16_variant.model_ref,
                "completed": _complete,
                "stakeholder_group": v2_16_obligation["stakeholder_label"],
                "harm_mode": v2_16_obligation["harm_label"],
                "obligation_metric": "maximum subgroup gap plus residual affected units",
                "obligation_threshold_pp": v2_16_obligation["threshold_pp"],
                "binding_amount": v2_16_obligation["binding_amount"],
                "monitoring_intensity_pct": v2_16_overhead["monitoring_intensity_pct"],
                "explanation_coverage_pct": v2_16_overhead["explanation_coverage_pct"],
                "privacy_strictness_pct": v2_16_overhead["privacy_strictness_pct"],
                "human_review_share_pct": v2_16_overhead["human_review_share_pct"],
                "overhead_binding_constraint": "; ".join(v2_16_overhead["violations"]) or "none",
                "audit_sample_rate_pct": v2_16_audit["sample_rate_pct"],
                "label_availability_pct": v2_16_audit["label_availability_pct"],
                "slice_depth": v2_16_audit["slice_depth"],
                "audit_blind_spot": v2_16_audit["blind_spot"],
                "residual_harm_units_per_day": v2_16_audit["blind_units_per_day"],
                "selected_policy": v2_16_policy["selected_policy_label"],
                "rejected_alternative": v2_16_policy["rejected_policy_label"],
                "policy_pass": v2_16_policy["policy_pass"],
                "residual_owner": v2_16_policy["residual_owner"],
                "v2_17_synthesis_implication": v2_16_v2_17_implication.value,
            },
        )

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">16 &middot; Responsible AI</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v2_16_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">POLICY</span>
        <span class="hud-value">{v2_16_policy["selected_policy_label"]}</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active" style="background:{COLORS['GreenLine'] if _complete else COLORS['OrangeLine']};">
            {"COMPLETE" if _complete else "ACTIVE"}
        </span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    report_export_panel,
    source_trace,
    v2_16_audit,
    v2_16_html_table,
    v2_16_metadata,
    v2_16_obligation,
    v2_16_overhead,
    v2_16_packet,
    v2_16_partA_checkpoint,
    v2_16_partA_pred,
    v2_16_partB_checkpoint,
    v2_16_partB_pred,
    v2_16_partC_checkpoint,
    v2_16_partC_pred,
    v2_16_partD_pred,
    v2_16_policy,
    v2_16_profile,
    v2_16_student_id,
    v2_16_v2_17_implication,
    v2_16_variant,
):
    _incomplete = []
    for label, value in (
        ("Part A prediction", v2_16_partA_pred.value),
        ("Part A checkpoint", v2_16_partA_checkpoint.value),
        ("Part B prediction", v2_16_partB_pred.value),
        ("Part B checkpoint", v2_16_partB_checkpoint.value),
        ("Part C prediction", v2_16_partC_pred.value),
        ("Part C checkpoint", v2_16_partC_checkpoint.value),
        ("Part D prediction", v2_16_partD_pred.value),
    ):
        if value is None:
            _incomplete.append(label)
    if v2_16_policy["residual_owner"] in {None, "none"}:
        _incomplete.append("Named residual obligation owner")
    if v2_16_policy["selected_policy"] == v2_16_policy["rejected_policy"]:
        _incomplete.append("Rejected alternative different from selected policy")

    _final_decision = (
        f"{'Approve' if v2_16_policy['policy_pass'] else 'Hold or revise'} "
        f"{v2_16_policy['selected_policy_label']} for {v2_16_profile.label}. "
        f"Binding amount: {v2_16_obligation['binding_amount']} "
        f"({v2_16_obligation['threshold_pp']:.1f} pp threshold, "
        f"{v2_16_audit['blind_units_per_day']:,.0f} blind residual units/day). "
        f"Rejected alternative: {v2_16_policy['rejected_policy_label']}."
    )
    _report = build_lab_report(
        v2_16_metadata,
        student_id=v2_16_student_id.value or "",
        track=v2_16_profile.label,
        scenario=v2_16_variant.workload_summary,
        learning_objectives=(
            "Convert stakeholder harm into a measurable track-specific obligation.",
            "Budget responsible evidence overhead across latency, cost, energy, and release delay.",
            "Analyze audit coverage, blind spots, escalation path, and residual harm.",
            "Choose a responsible fleet policy with a rejected alternative and V2-17 implication.",
        ),
        predictions={
            "part_a_harm_to_obligation": v2_16_partA_pred.value,
            "part_b_evidence_overhead": v2_16_partB_pred.value,
            "part_c_audit_coverage": v2_16_partC_pred.value,
            "part_d_policy_gate": v2_16_partD_pred.value,
        },
        knob_settings={
            "stakeholder_group": v2_16_obligation["stakeholder_label"],
            "harm_mode": v2_16_obligation["harm_label"],
            "obligation_threshold_pp": v2_16_obligation["threshold_pp"],
            "monitoring_intensity_pct": v2_16_overhead["monitoring_intensity_pct"],
            "explanation_coverage_pct": v2_16_overhead["explanation_coverage_pct"],
            "privacy_strictness_pct": v2_16_overhead["privacy_strictness_pct"],
            "human_review_share_pct": v2_16_overhead["human_review_share_pct"],
            "audit_sample_rate_pct": v2_16_audit["sample_rate_pct"],
            "label_availability_pct": v2_16_audit["label_availability_pct"],
            "slice_depth": v2_16_audit["slice_depth"],
            "escalation_hours": v2_16_audit["escalation_hours"],
            "selected_policy": v2_16_policy["selected_policy_label"],
            "rejected_alternative": v2_16_policy["rejected_policy_label"],
            "v2_17_implication": v2_16_v2_17_implication.value,
        },
        binding_constraints={
            "part_a": "; ".join(v2_16_obligation["violations"]) or "stakeholder obligation stated",
            "part_b": "; ".join(v2_16_overhead["violations"]) or "evidence overhead feasible",
            "part_c": "; ".join(v2_16_audit["violations"]) or "audit coverage feasible",
            "part_d": "; ".join(v2_16_policy["violations"]) or "policy gate passes",
        },
        evidence_summary={
            "binding_amount": v2_16_obligation["binding_amount"],
            "observed_gap_pp": round(v2_16_obligation["observed_gap_pp"], 3),
            "residual_gap_pp_after_controls": round(v2_16_overhead["residual_gap_pp"], 3),
            "latency_ms": round(v2_16_overhead["latency_ms"], 3),
            "cost_factor": round(v2_16_overhead["cost_factor"], 3),
            "audit_confidence_pct": round(v2_16_audit["audit_confidence_pct"], 3),
            "blind_units_per_day": round(v2_16_audit["blind_units_per_day"], 3),
            "selected_policy": v2_16_policy["selected_policy_label"],
            "policy_pass": v2_16_policy["policy_pass"],
        },
        final_decision=_final_decision,
        big_takeaways=(
            "Stakeholder harm has to become a measurable obligation before a fairness metric is actionable.",
            "Responsible evidence reduces risk while consuming latency, cost, energy, storage, and release-review capacity.",
            "Audit coverage and escalation determine whether residual harm is seen and owned.",
            "V2-17 synthesis must treat the selected responsible-AI policy as a hard fleet guardrail.",
        ),
        reflections={
            "report_artifact": v2_16_packet["report_frame"],
            "residual_obligation": (
                f"{v2_16_audit['blind_units_per_day']:,.0f} blind residual units/day, "
                f"owned by {v2_16_policy['residual_owner'] if v2_16_policy['residual_owner'] not in {None, 'none'} else 'unassigned'}"
            ),
            "part_a_checkpoint": v2_16_partA_checkpoint.value,
            "part_b_checkpoint": v2_16_partB_checkpoint.value,
            "part_c_checkpoint": v2_16_partC_checkpoint.value,
            "v2_17_implication": v2_16_v2_17_implication.value,
        },
        residual_risk=(
            f"{v2_16_packet['failure_mode']}. Teaching estimates need representative "
            "track audits, privacy review, escalation drills, and postdeployment monitoring before production use."
        ),
        source_trace={
            "track_id": v2_16_profile.track_id,
            "scenario_id": v2_16_variant.scenario_id,
            "hardware_ref": v2_16_variant.hardware_ref,
            "model_ref": v2_16_variant.model_ref,
            "catalog_variant": "mlsysbook_labs.get_lab_track_variant",
            "track_profile": "mlsysbook_labs.get_track_profile",
            "notebook_local_helpers": (
                "v2_16_track_packet",
                "v2_16_obligation_result",
                "v2_16_overhead_result",
                "v2_16_audit_result",
                "v2_16_policy_result",
            ),
            "chapter_sources": (
                "responsible_ai.qmd: governance imperative, fairness measurement, overhead table, monitoring architecture, sociotechnical feedback, institutional responsibility",
                "responsible_fleet_principles.qmd: fairness impossibility and sociotechnical feedback invariants",
            ),
        },
        result_snapshot={
            "packet": v2_16_packet,
            "obligation": v2_16_obligation,
            "overhead": v2_16_overhead,
            "audit": v2_16_audit,
            "policy": v2_16_policy,
        },
        incomplete_fields=tuple(_incomplete),
    )

    _summary_table = v2_16_html_table(
        ["Memo field", "Value"],
        [
            ["Selected policy", v2_16_policy["selected_policy_label"]],
            ["Rejected alternative", v2_16_policy["rejected_policy_label"]],
            ["Binding amount", v2_16_obligation["binding_amount"]],
            ["Residual obligation", f"{v2_16_audit['blind_units_per_day']:,.0f} blind units/day"],
            ["Policy status", "pass" if v2_16_policy["policy_pass"] else "fail"],
            ["V2-17 implication", v2_16_v2_17_implication.value],
        ],
    )
    def build_synthesis():
        return mo.vstack(
            [
                mo.md("## Synthesis: Responsible Fleet Memo"),
                mo.callout(
                    mo.md(
                        "The memo carries one selected policy, a rejected alternative, the binding amount, "
                        "the residual obligation, and the implication for V2-17 fleet synthesis."
                    ),
                    kind="info",
                ),
                mo.Html(_summary_table),
                source_trace(
                    {
                        "Report builder": "mlsysbook_labs.build_lab_report",
                        "Report export": "mlsysbook_labs.report_export_panel",
                        "Ledger": "DesignLedger.save(chapter=16)",
                        "Local teaching models": "v2_16_* helpers in this notebook",
                    },
                    collapsed=True,
                    summary="The report and ledger snapshot are generated from local controls and source-traced helpers.",
                ),
                mo.md("## Download Report"),
                report_export_panel(_report),
            ]
        )

    build_synthesis()
    return


if __name__ == "__main__":
    app.run()
