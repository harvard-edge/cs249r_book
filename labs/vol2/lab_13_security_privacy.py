import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# -----------------------------------------------------------------------------
# LAB V2-13: THE PRICE OF PRIVACY
#
# Chapter invariant: security and privacy are amount systems. Threat surface,
# privacy budget, control overhead, access/deletion lineage, audit evidence, and
# residual risk must all fit inside the selected track's operating envelope.
#
# Packet modules:
#   Part A - threat surface and privacy budget as binding amounts
#   Part B - control strength versus latency, utility, and governance overhead
#   Part C - access, retention, deletion lineage, and residual exposure
#   Part D - deployment policy as multi-guardrail conjunction
#   Synthesis - security/privacy memo and V2-14 robustness implication
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
    v2_13_lab_path = "vol2/lab_13_security_privacy.py"
    v2_13_chapter = 13
    v2_13_metadata = get_lab_metadata(v2_13_lab_path)
    return v2_13_chapter, v2_13_lab_path, v2_13_metadata


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_13_track_picker = track_selector(default=_default_track)
    v2_13_track_picker
    return (v2_13_track_picker,)


@app.cell
def _(get_lab_track_variant, get_track_profile, v2_13_metadata, v2_13_track_picker):
    v2_13_track_id = v2_13_track_picker.value
    v2_13_profile = get_track_profile(v2_13_track_id)
    v2_13_variant = get_lab_track_variant(v2_13_metadata.lab_id, v2_13_track_id)
    return v2_13_profile, v2_13_track_id, v2_13_variant


@app.cell
def _(html_lib, math):
    def v2_13_escape(value):
        return html_lib.escape(str(value))

    def v2_13_track_packet(profile, variant):
        base = {
            "track_id": profile.track_id,
            "label": profile.label,
            "stakeholder": variant.stakeholder,
            "scenario": variant.workload_summary,
            "mission": profile.narrative,
            "hardware_ref": variant.hardware_ref,
            "model_ref": variant.model_ref,
            "protected_asset": "tenant prompts, training logs, model registry entries, and access traces",
            "trust_boundary": "tenant workload -> shared accelerator -> model registry -> audit log",
            "adversary": "credentialed tenant, compromised service principal, or systematic API extractor",
            "sensitive_flow": "prompt/log/fine-tune records through serving, monitoring, and retraining",
            "ops_unit": "tenant requests",
            "privacy_unit": "epsilon over fine-tuning and analytics accesses",
            "natural_failure": "broad logs improve debugging while leaking tenant and model behavior information",
            "report_frame": "Multi-tenant privacy/security memo",
            "surface_default_nodes": 16,
            "sensitive_paths_default": 6,
            "privacy_access_default": 6,
            "surface_limit": 210.0,
            "privacy_budget_limit": 10.0,
            "evidence_floor": 78.0,
            "path_surface_weight": 7.0,
            "channel_surface_weight": 0.90,
            "logging_surface_weight": 9.0,
            "epsilon_per_access": 0.85,
            "base_latency_ms": 180.0,
            "latency_limit_ms": 260.0,
            "tee_latency_ms": 18.0,
            "fhe_multiplier": 1000.0,
            "strength_latency_ms": 0.24,
            "strength_utility_pp": 0.090,
            "utility_loss_limit_pp": 8.0,
            "governance_base_items": 22.0,
            "governance_limit_items": 70.0,
            "protection_floor": 70.0,
            "role_limit": 8.0,
            "retention_default_days": 90,
            "retention_limit_days": 120.0,
            "deletion_default_days": 30,
            "deletion_sla_days": 30.0,
            "lineage_default_pct": 72,
            "audit_default_pct": 70,
            "residual_risk_limit": 92.0,
            "v2_14_options": {
                "telemetry_minimization": "Telemetry minimization reduces privacy exposure but leaves V2-14 with less drift and incident evidence.",
                "attack_monitoring": "Adversarial query monitoring becomes a robustness stress signal, not just a security alert.",
                "retention_replay": "Retention and deletion choices determine which failures can be replayed during robustness validation.",
                "fallback_boundary": "A privacy-preserving fallback must still preserve the robustness envelope under distribution shift.",
            },
        }
        overrides = {
            "iphone": {
                "protected_asset": "on-device personalization data, opt-in telemetry, permissions, and model update metadata",
                "trust_boundary": "sensor/app data -> local model -> opt-in telemetry -> app release channel",
                "adversary": "malicious app, device thief, credentialed support tool, or overbroad analytics workflow",
                "sensitive_flow": "local usage traces through personalization, crash reporting, and model updates",
                "ops_unit": "opt-in sessions",
                "privacy_unit": "epsilon over local analytics and personalization events",
                "natural_failure": "verbose telemetry improves debugging but violates consent and deletion expectations",
                "report_frame": "Mobile privacy release memo",
                "surface_default_nodes": 7,
                "sensitive_paths_default": 4,
                "privacy_access_default": 4,
                "surface_limit": 88.0,
                "privacy_budget_limit": 6.0,
                "evidence_floor": 74.0,
                "path_surface_weight": 6.0,
                "channel_surface_weight": 0.55,
                "logging_surface_weight": 8.0,
                "epsilon_per_access": 0.70,
                "base_latency_ms": 38.0,
                "latency_limit_ms": 80.0,
                "tee_latency_ms": 6.0,
                "fhe_multiplier": 1200.0,
                "strength_latency_ms": 0.16,
                "strength_utility_pp": 0.070,
                "utility_loss_limit_pp": 6.0,
                "governance_base_items": 16.0,
                "governance_limit_items": 52.0,
                "protection_floor": 66.0,
                "role_limit": 6.0,
                "retention_default_days": 30,
                "retention_limit_days": 45.0,
                "deletion_default_days": 14,
                "deletion_sla_days": 14.0,
                "lineage_default_pct": 76,
                "audit_default_pct": 64,
                "residual_risk_limit": 82.0,
            },
            "oura_ring": {
                "protected_asset": "biosignal windows, sleep summaries, BLE sync payloads, and firmware OTA evidence",
                "trust_boundary": "sensor ring -> phone handoff -> cloud sync -> firmware/model update",
                "adversary": "device thief, BLE observer, cloud support workflow, or untrusted aggregation participant",
                "sensitive_flow": "health-adjacent signals through local inference, sync, aggregation, and support logs",
                "ops_unit": "device syncs",
                "privacy_unit": "epsilon over biosignal aggregation and firmware validation cohorts",
                "natural_failure": "strong DP on a small cohort collapses utility, while broad sync retains too much biosignal data",
                "report_frame": "Wearable health-data privacy memo",
                "surface_default_nodes": 10,
                "sensitive_paths_default": 5,
                "privacy_access_default": 5,
                "surface_limit": 115.0,
                "privacy_budget_limit": 6.5,
                "evidence_floor": 80.0,
                "path_surface_weight": 6.5,
                "channel_surface_weight": 0.65,
                "logging_surface_weight": 7.5,
                "epsilon_per_access": 0.75,
                "base_latency_ms": 24.0,
                "latency_limit_ms": 60.0,
                "tee_latency_ms": 8.0,
                "fhe_multiplier": 1500.0,
                "strength_latency_ms": 0.12,
                "strength_utility_pp": 0.085,
                "utility_loss_limit_pp": 5.0,
                "governance_base_items": 18.0,
                "governance_limit_items": 56.0,
                "protection_floor": 72.0,
                "role_limit": 5.0,
                "retention_default_days": 45,
                "retention_limit_days": 60.0,
                "deletion_default_days": 21,
                "deletion_sla_days": 21.0,
                "lineage_default_pct": 82,
                "audit_default_pct": 72,
                "residual_risk_limit": 78.0,
            },
            "robotaxi": {
                "protected_asset": "sensor logs, location traces, safety incidents, replay datasets, and signed model artifacts",
                "trust_boundary": "vehicle sensors -> edge perception -> fleet upload -> safety replay/model registry",
                "adversary": "physical attacker, malicious rider, compromised upload path, or insider accessing safety logs",
                "sensitive_flow": "location/sensor evidence through incident replay, model validation, and geofence expansion",
                "ops_unit": "live miles",
                "privacy_unit": "epsilon over safety datasets and incident analysis cohorts",
                "natural_failure": "deleting evidence too soon protects privacy but weakens incident replay and robustness validation",
                "report_frame": "Safety data security memo",
                "surface_default_nodes": 14,
                "sensitive_paths_default": 7,
                "privacy_access_default": 4,
                "surface_limit": 170.0,
                "privacy_budget_limit": 8.0,
                "evidence_floor": 90.0,
                "path_surface_weight": 7.5,
                "channel_surface_weight": 0.85,
                "logging_surface_weight": 9.0,
                "epsilon_per_access": 0.95,
                "base_latency_ms": 86.0,
                "latency_limit_ms": 120.0,
                "tee_latency_ms": 12.0,
                "fhe_multiplier": 1200.0,
                "strength_latency_ms": 0.18,
                "strength_utility_pp": 0.055,
                "utility_loss_limit_pp": 2.5,
                "governance_base_items": 28.0,
                "governance_limit_items": 88.0,
                "protection_floor": 84.0,
                "role_limit": 5.0,
                "retention_default_days": 180,
                "retention_limit_days": 240.0,
                "deletion_default_days": 45,
                "deletion_sla_days": 45.0,
                "lineage_default_pct": 88,
                "audit_default_pct": 84,
                "residual_risk_limit": 72.0,
            },
            "cloud_fleet": {},
        }
        packet = dict(base)
        packet.update(overrides.get(profile.track_id, {}))
        packet["source_policy"] = profile.source_policy
        return packet

    def v2_13_fmt_number(value, digits=1):
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

    def v2_13_fmt_ms(value):
        value = float(value)
        if value >= 1000:
            return f"{value / 1000:.1f} s"
        return f"{value:.1f} ms"

    def v2_13_fmt_pct(value, digits=1):
        return f"{float(value):.{digits}f}%"

    def v2_13_guardrail_badge(ok):
        return "PASS" if ok else "FAIL"

    def v2_13_prediction_feedback(predicted, actual, labels):
        if predicted is None:
            return ("warn", "Commit to the structured prediction before treating the instrument as evidence.")
        if predicted == actual:
            return ("success", f"Prediction check: correct. The measured result is `{labels.get(actual, actual)}`.")
        return ("warn", f"Prediction check: the instrument found `{labels.get(actual, actual)}`, not `{labels.get(predicted, predicted)}`.")

    def v2_13_binding_from_ratios(ratios):
        return max(ratios, key=lambda key: ratios[key])

    return (
        v2_13_binding_from_ratios,
        v2_13_escape,
        v2_13_fmt_ms,
        v2_13_fmt_number,
        v2_13_fmt_pct,
        v2_13_guardrail_badge,
        v2_13_prediction_feedback,
        v2_13_track_packet,
    )


@app.cell
def _(v2_13_profile, v2_13_track_packet, v2_13_variant):
    v2_13_packet = v2_13_track_packet(v2_13_profile, v2_13_variant)
    return (v2_13_packet,)


@app.cell
def _(mo, v2_13_packet):
    v2_13_partA_pred = mo.ui.radio(
        options={
            "Threat surface grows past the track envelope first.": "threat_surface",
            "Privacy budget is consumed first.": "privacy_budget",
            "Audit/evidence lineage is the first gap.": "evidence_lineage",
        },
        label="Part A prediction",
    )
    v2_13_surface_nodes = mo.ui.slider(
        start=2,
        stop=32,
        step=1,
        value=int(v2_13_packet["surface_default_nodes"]),
        label="Distributed nodes / trust endpoints",
    )
    v2_13_sensitive_paths = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=int(v2_13_packet["sensitive_paths_default"]),
        label="Sensitive lifecycle paths",
    )
    v2_13_privacy_accesses = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=int(v2_13_packet["privacy_access_default"]),
        label="Privacy-consuming accesses",
    )
    v2_13_logging_scope = mo.ui.dropdown(
        options={
            "Minimal telemetry": "minimal",
            "Scoped security logging": "scoped",
            "Verbose debug logging": "verbose",
        },
        value="Scoped security logging",
        label="Logging and evidence scope",
    )
    v2_13_partA_checkpoint = mo.ui.radio(
        options={
            "Reduce exposed paths or segment the boundary.": "reduce_surface",
            "Tighten privacy accounting before another access.": "tighten_privacy",
            "Add lineage evidence before claiming compliance.": "add_lineage",
        },
        label="Part A checkpoint",
    )

    v2_13_partB_pred = mo.ui.radio(
        options={
            "Latency overhead will bind first.": "latency",
            "Utility loss will bind first.": "utility",
            "Governance overhead will bind first.": "governance",
            "Remaining protection gap will bind first.": "protection",
        },
        label="Part B prediction",
    )
    v2_13_control_strength = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=55,
        label="Privacy/security control strength",
    )
    v2_13_compute_boundary = mo.ui.dropdown(
        options={
            "Encrypted transport and storage": "aes",
            "Trusted execution boundary": "tee",
            "Fully encrypted compute": "fhe",
        },
        value="Trusted execution boundary",
        label="Compute protection boundary",
    )
    v2_13_output_policy = mo.ui.dropdown(
        options={
            "Full confidence outputs": "full",
            "Top-k rounded outputs": "topk",
            "Label-only outputs": "label",
        },
        value="Top-k rounded outputs",
        label="Output exposure policy",
    )
    v2_13_aggregation_policy = mo.ui.dropdown(
        options={
            "Central raw access": "central",
            "Secure aggregation": "secure_agg",
            "Secure aggregation plus DP": "secure_agg_dp",
        },
        value="Secure aggregation",
        label="Training/analytics privacy mode",
    )
    v2_13_partB_checkpoint = mo.ui.radio(
        options={
            "Carry the balanced control stack forward.": "balanced_stack",
            "Prefer stronger isolation even if an overhead fails.": "strongest_stack",
            "Prefer broad access to preserve utility.": "utility_first",
            "Delay deployment until governance evidence exists.": "governance_first",
        },
        label="Part B checkpoint",
    )

    v2_13_partC_pred = mo.ui.radio(
        options={
            "Access roles dominate residual exposure.": "access_roles",
            "Retention record-days dominate residual exposure.": "retention",
            "Deletion window dominates residual exposure.": "deletion",
            "Audit evidence gap dominates residual exposure.": "audit_gap",
        },
        label="Part C prediction",
    )
    v2_13_access_model = mo.ui.dropdown(
        options={
            "Broad project access": "broad",
            "Least privilege roles": "least_privilege",
            "Break-glass only": "break_glass",
        },
        value="Least privilege roles",
        label="Access model",
    )
    v2_13_retention_days = mo.ui.slider(
        start=7,
        stop=365,
        step=7,
        value=int(v2_13_packet["retention_default_days"]),
        label="Retention window (days)",
    )
    v2_13_deletion_window_days = mo.ui.slider(
        start=1,
        stop=90,
        step=1,
        value=int(v2_13_packet["deletion_default_days"]),
        label="Deletion completion window (days)",
    )
    v2_13_lineage_coverage_pct = mo.ui.slider(
        start=40,
        stop=100,
        step=5,
        value=int(v2_13_packet["lineage_default_pct"]),
        label="Lineage coverage (%)",
    )
    v2_13_audit_sampling_pct = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=int(v2_13_packet["audit_default_pct"]),
        label="Audit sampling (%)",
    )
    v2_13_partC_checkpoint = mo.ui.radio(
        options={
            "Shorten retention and prove deletion lineage.": "deletion_lineage",
            "Expand audit evidence before releasing.": "audit_first",
            "Keep broad access for operational speed.": "broad_access",
            "Accept residual exposure as documented risk.": "accept_risk",
        },
        label="Part C checkpoint",
    )

    v2_13_partD_pred = mo.ui.radio(
        options={
            "Residual-risk guardrail rejects broad access.": "residual",
            "Privacy-budget guardrail rejects broad access.": "privacy",
            "Evidence guardrail rejects broad access.": "evidence",
            "Deletion guardrail rejects broad access.": "deletion",
        },
        label="Part D prediction",
    )
    v2_13_partD_policy_choice = mo.ui.dropdown(
        options={
            "Broad data access": "broad",
            "Privacy-preserving control": "privacy",
            "Strict isolation": "strict",
            "Custom from Parts A-C": "custom",
        },
        value="Privacy-preserving control",
        label="Selected security/privacy policy",
    )
    v2_13_rejected_policy = mo.ui.dropdown(
        options={
            "Broad data access": "broad",
            "Privacy-preserving control": "privacy",
            "Strict isolation": "strict",
            "Custom from Parts A-C": "custom",
        },
        value="Broad data access",
        label="Rejected alternative",
    )
    v2_13_partD_checkpoint = mo.ui.radio(
        options={
            "Approve only if every guardrail passes.": "all_guardrails",
            "Approve strongest privacy even with latency failure.": "privacy_only",
            "Approve lowest latency even with residual risk.": "latency_only",
            "Approve broad access if audit logs are verbose.": "logging_only",
        },
        label="Part D checkpoint",
    )

    v2_13_student_id = mo.ui.text(label="Student identifier", placeholder="Optional")
    v2_13_robustness_implication = mo.ui.radio(
        options={
            "Telemetry minimization reduces robustness evidence.": "telemetry_minimization",
            "Attack monitoring becomes a robustness stress signal.": "attack_monitoring",
            "Retention/deletion choices affect incident replay.": "retention_replay",
            "Fallback policy must preserve robustness envelope.": "fallback_boundary",
        },
        label="V2-14 robustness implication",
    )
    v2_13_memo_note = mo.ui.text_area(
        label="Optional memo note",
        placeholder="One sentence of local residual risk or deployment context.",
    )

    return (
        v2_13_access_model,
        v2_13_aggregation_policy,
        v2_13_audit_sampling_pct,
        v2_13_compute_boundary,
        v2_13_control_strength,
        v2_13_deletion_window_days,
        v2_13_lineage_coverage_pct,
        v2_13_logging_scope,
        v2_13_memo_note,
        v2_13_output_policy,
        v2_13_partA_checkpoint,
        v2_13_partA_pred,
        v2_13_partB_checkpoint,
        v2_13_partB_pred,
        v2_13_partC_checkpoint,
        v2_13_partC_pred,
        v2_13_partD_checkpoint,
        v2_13_partD_policy_choice,
        v2_13_partD_pred,
        v2_13_privacy_accesses,
        v2_13_rejected_policy,
        v2_13_retention_days,
        v2_13_robustness_implication,
        v2_13_sensitive_paths,
        v2_13_student_id,
        v2_13_surface_nodes,
    )


@app.cell
def _(v2_13_binding_from_ratios):
    def v2_13_part_a_amounts(packet, *, nodes, sensitive_paths, privacy_accesses, logging_scope):
        logging_configs = {
            "minimal": {"label": "Minimal telemetry", "evidence": 48.0, "epsilon_per_path": 0.05, "surface": 0.8},
            "scoped": {"label": "Scoped security logging", "evidence": 76.0, "epsilon_per_path": 0.18, "surface": 1.7},
            "verbose": {"label": "Verbose debug logging", "evidence": 94.0, "epsilon_per_path": 0.42, "surface": 3.4},
        }
        cfg = logging_configs[logging_scope]
        pair_channels = nodes * (nodes - 1) / 2
        surface_index = (
            sensitive_paths * packet["path_surface_weight"]
            + pair_channels * packet["channel_surface_weight"]
            + cfg["surface"] * packet["logging_surface_weight"]
        )
        privacy_epsilon = privacy_accesses * packet["epsilon_per_access"] + sensitive_paths * cfg["epsilon_per_path"]
        evidence_score = min(100.0, cfg["evidence"] + 0.7 * sensitive_paths + 0.35 * privacy_accesses)
        ratios = {
            "threat_surface": surface_index / packet["surface_limit"],
            "privacy_budget": privacy_epsilon / packet["privacy_budget_limit"],
            "evidence_lineage": packet["evidence_floor"] / max(evidence_score, 1.0),
        }
        binding_key = v2_13_binding_from_ratios(ratios)
        binding_labels = {
            "threat_surface": "Threat surface",
            "privacy_budget": "Privacy budget",
            "evidence_lineage": "Evidence lineage",
        }
        return {
            "logging_label": cfg["label"],
            "nodes": nodes,
            "sensitive_paths": sensitive_paths,
            "privacy_accesses": privacy_accesses,
            "pair_channels": pair_channels,
            "surface_index": surface_index,
            "privacy_epsilon": privacy_epsilon,
            "evidence_score": evidence_score,
            "ratios": ratios,
            "binding_key": binding_key,
            "binding": binding_labels[binding_key],
            "binding_ratio": ratios[binding_key],
            "ok": all(value <= 1.0 for value in ratios.values()),
        }

    def v2_13_part_b_overhead(packet, *, strength, compute_boundary, output_policy, aggregation_policy):
        compute_configs = {
            "aes": {"label": "Encrypted transport/storage", "latency_ms": 0.5, "utility_pp": 0.0, "governance": 4.0, "protection": 12.0},
            "tee": {"label": "Trusted execution boundary", "latency_ms": packet["tee_latency_ms"], "utility_pp": 0.3, "governance": 13.0, "protection": 26.0},
            "fhe": {
                "label": "Fully encrypted compute",
                "latency_ms": packet["base_latency_ms"] * (packet["fhe_multiplier"] - 1.0),
                "utility_pp": 1.0,
                "governance": 28.0,
                "protection": 42.0,
            },
        }
        output_configs = {
            "full": {"label": "Full confidence outputs", "latency_ms": 0.0, "utility_pp": 0.0, "governance": 2.0, "protection": 0.0},
            "topk": {"label": "Top-k rounded outputs", "latency_ms": 0.8, "utility_pp": 0.8, "governance": 6.0, "protection": 14.0},
            "label": {"label": "Label-only outputs", "latency_ms": 0.4, "utility_pp": 2.1, "governance": 8.0, "protection": 22.0},
        }
        aggregation_configs = {
            "central": {"label": "Central raw access", "latency_ms": 0.0, "utility_pp": 0.0, "governance": 3.0, "protection": 0.0},
            "secure_agg": {"label": "Secure aggregation", "latency_ms": 4.0, "utility_pp": 0.6, "governance": 8.0, "protection": 15.0},
            "secure_agg_dp": {"label": "Secure aggregation plus DP", "latency_ms": 7.0, "utility_pp": 2.0, "governance": 13.0, "protection": 25.0},
        }
        compute = compute_configs[compute_boundary]
        output = output_configs[output_policy]
        aggregation = aggregation_configs[aggregation_policy]
        latency_ms = (
            packet["base_latency_ms"]
            + compute["latency_ms"]
            + output["latency_ms"]
            + aggregation["latency_ms"]
            + strength * packet["strength_latency_ms"]
        )
        utility_loss_pp = (
            compute["utility_pp"]
            + output["utility_pp"]
            + aggregation["utility_pp"]
            + strength * packet["strength_utility_pp"]
        )
        governance_items = (
            packet["governance_base_items"]
            + compute["governance"]
            + output["governance"]
            + aggregation["governance"]
            + strength * 0.22
        )
        protection_score = min(
            100.0,
            18.0
            + compute["protection"]
            + output["protection"]
            + aggregation["protection"]
            + strength * 0.42,
        )
        effective_epsilon = max(0.35, packet["privacy_budget_limit"] * (1.18 - protection_score / 125.0))
        ratios = {
            "latency": latency_ms / packet["latency_limit_ms"],
            "utility": utility_loss_pp / packet["utility_loss_limit_pp"],
            "governance": governance_items / packet["governance_limit_items"],
            "protection": packet["protection_floor"] / max(protection_score, 1.0),
        }
        binding_key = v2_13_binding_from_ratios(ratios)
        binding_labels = {
            "latency": "Latency overhead",
            "utility": "Utility loss",
            "governance": "Governance overhead",
            "protection": "Remaining protection gap",
        }
        return {
            "compute_label": compute["label"],
            "output_label": output["label"],
            "aggregation_label": aggregation["label"],
            "control_stack": f"{compute['label']} + {output['label']} + {aggregation['label']}",
            "strength": strength,
            "latency_ms": latency_ms,
            "utility_loss_pp": utility_loss_pp,
            "governance_items": governance_items,
            "protection_score": protection_score,
            "effective_epsilon": effective_epsilon,
            "ratios": ratios,
            "binding_key": binding_key,
            "binding": binding_labels[binding_key],
            "binding_ratio": ratios[binding_key],
            "ok": all(value <= 1.0 for value in ratios.values()),
        }

    def v2_13_part_c_lineage(
        packet,
        *,
        access_model,
        retention_days,
        deletion_window_days,
        lineage_coverage_pct,
        audit_sampling_pct,
    ):
        access_configs = {
            "broad": {"label": "Broad project access", "roles": 12.0, "evidence": 36.0, "exposure": 1.25},
            "least_privilege": {"label": "Least privilege roles", "roles": 5.0, "evidence": 72.0, "exposure": 0.75},
            "break_glass": {"label": "Break-glass only", "roles": 3.0, "evidence": 90.0, "exposure": 0.55},
        }
        cfg = access_configs[access_model]
        access_ratio = (cfg["roles"] / packet["role_limit"]) * cfg["exposure"]
        retention_ratio = retention_days / packet["retention_limit_days"]
        deletion_ratio = deletion_window_days / packet["deletion_sla_days"]
        audit_score = min(
            100.0,
            0.42 * lineage_coverage_pct + 0.33 * audit_sampling_pct + 0.25 * cfg["evidence"],
        )
        audit_ratio = packet["evidence_floor"] / max(audit_score, 1.0)
        residual_exposure = 100.0 * (
            0.30 * min(access_ratio, 2.0)
            + 0.24 * min(retention_ratio, 2.0)
            + 0.24 * min(deletion_ratio, 2.0)
            + 0.22 * max(0.0, 1.0 - audit_score / 100.0)
        )
        ratios = {
            "access_roles": access_ratio,
            "retention": retention_ratio,
            "deletion": deletion_ratio,
            "audit_gap": audit_ratio,
        }
        binding_key = v2_13_binding_from_ratios(ratios)
        binding_labels = {
            "access_roles": "Access roles",
            "retention": "Retention record-days",
            "deletion": "Deletion window",
            "audit_gap": "Audit evidence gap",
        }
        return {
            "access_label": cfg["label"],
            "access_roles": cfg["roles"],
            "retention_days": retention_days,
            "deletion_window_days": deletion_window_days,
            "lineage_coverage_pct": lineage_coverage_pct,
            "audit_sampling_pct": audit_sampling_pct,
            "audit_score": audit_score,
            "residual_exposure": residual_exposure,
            "ratios": ratios,
            "binding_key": binding_key,
            "binding": binding_labels[binding_key],
            "binding_ratio": ratios[binding_key],
            "ok": all(value <= 1.0 for value in ratios.values()) and residual_exposure <= packet["residual_risk_limit"],
        }

    def v2_13_guardrail_label(key):
        labels = {
            "privacy": "Privacy budget",
            "latency": "Latency",
            "utility": "Utility",
            "evidence": "Audit evidence",
            "deletion": "Deletion lineage",
            "residual": "Residual risk",
        }
        return labels.get(key, str(key))

    def v2_13_assess_policy(packet, policy):
        ratios = {
            "privacy": policy["privacy_epsilon"] / packet["privacy_budget_limit"],
            "latency": policy["latency_ms"] / packet["latency_limit_ms"],
            "utility": policy["utility_loss_pp"] / packet["utility_loss_limit_pp"],
            "evidence": packet["evidence_floor"] / max(policy["evidence_score"], 1.0),
            "deletion": policy["deletion_days"] / packet["deletion_sla_days"],
            "residual": policy["residual_risk"] / packet["residual_risk_limit"],
        }
        checks = {key: value <= 1.0 for key, value in ratios.items()}
        binding_key = v2_13_binding_from_ratios(ratios)
        assessed = dict(policy)
        assessed.update(
            {
                "ratios": ratios,
                "checks": checks,
                "binding": binding_key,
                "binding_ratio": ratios[binding_key],
                "feasible": all(checks.values()),
                "violations": tuple(v2_13_guardrail_label(key) for key, ok in checks.items() if not ok),
            }
        )
        return assessed

    return (
        v2_13_assess_policy,
        v2_13_guardrail_label,
        v2_13_part_a_amounts,
        v2_13_part_b_overhead,
        v2_13_part_c_lineage,
    )


@app.cell
def _(
    v2_13_logging_scope,
    v2_13_packet,
    v2_13_part_a_amounts,
    v2_13_privacy_accesses,
    v2_13_sensitive_paths,
    v2_13_surface_nodes,
):
    v2_13_a = v2_13_part_a_amounts(
        v2_13_packet,
        nodes=v2_13_surface_nodes.value,
        sensitive_paths=v2_13_sensitive_paths.value,
        privacy_accesses=v2_13_privacy_accesses.value,
        logging_scope=v2_13_logging_scope.value,
    )
    return (v2_13_a,)


@app.cell
def _(
    v2_13_aggregation_policy,
    v2_13_compute_boundary,
    v2_13_control_strength,
    v2_13_output_policy,
    v2_13_packet,
    v2_13_part_b_overhead,
):
    v2_13_b = v2_13_part_b_overhead(
        v2_13_packet,
        strength=v2_13_control_strength.value,
        compute_boundary=v2_13_compute_boundary.value,
        output_policy=v2_13_output_policy.value,
        aggregation_policy=v2_13_aggregation_policy.value,
    )
    return (v2_13_b,)


@app.cell
def _(
    v2_13_access_model,
    v2_13_audit_sampling_pct,
    v2_13_deletion_window_days,
    v2_13_lineage_coverage_pct,
    v2_13_packet,
    v2_13_part_c_lineage,
    v2_13_retention_days,
):
    v2_13_c = v2_13_part_c_lineage(
        v2_13_packet,
        access_model=v2_13_access_model.value,
        retention_days=v2_13_retention_days.value,
        deletion_window_days=v2_13_deletion_window_days.value,
        lineage_coverage_pct=v2_13_lineage_coverage_pct.value,
        audit_sampling_pct=v2_13_audit_sampling_pct.value,
    )
    return (v2_13_c,)


@app.cell
def _(v2_13_a, v2_13_assess_policy, v2_13_b, v2_13_c, v2_13_packet, v2_13_partD_policy_choice, v2_13_rejected_policy):
    def v2_13_policy_candidates(packet, a, b, c):
        broad = {
            "name": "Broad data access",
            "privacy_epsilon": packet["privacy_budget_limit"] * 1.25,
            "latency_ms": packet["base_latency_ms"] + 2.0,
            "utility_loss_pp": 0.5,
            "evidence_score": packet["evidence_floor"] * 0.62,
            "deletion_days": packet["deletion_sla_days"] * 1.65,
            "residual_risk": packet["residual_risk_limit"] * 1.38,
            "rationale": "Fast and high-utility, but exposes too much data and lacks deletion/audit proof.",
        }
        privacy = {
            "name": "Privacy-preserving control",
            "privacy_epsilon": min(packet["privacy_budget_limit"] * 0.82, max(a["privacy_epsilon"] * 0.82, packet["privacy_budget_limit"] * 0.45)),
            "latency_ms": min(packet["latency_limit_ms"] * 0.92, max(packet["base_latency_ms"] + packet["tee_latency_ms"] + 8.0, b["latency_ms"] * 0.92)),
            "utility_loss_pp": min(packet["utility_loss_limit_pp"] * 0.82, max(1.5, b["utility_loss_pp"] * 0.85)),
            "evidence_score": max(packet["evidence_floor"] + 6.0, c["audit_score"]),
            "deletion_days": min(packet["deletion_sla_days"] * 0.80, c["deletion_window_days"]),
            "residual_risk": min(packet["residual_risk_limit"] * 0.78, max(35.0, c["residual_exposure"] * 0.72)),
            "rationale": "Balances output limiting, scoped evidence, least privilege, and privacy accounting.",
        }
        strict = {
            "name": "Strict isolation",
            "privacy_epsilon": packet["privacy_budget_limit"] * 0.36,
            "latency_ms": packet["latency_limit_ms"] * 1.18,
            "utility_loss_pp": packet["utility_loss_limit_pp"] * 0.92,
            "evidence_score": min(100.0, packet["evidence_floor"] + 14.0),
            "deletion_days": packet["deletion_sla_days"] * 0.60,
            "residual_risk": packet["residual_risk_limit"] * 0.42,
            "rationale": "Minimizes exposure, but may exceed latency or utility budgets for real-time tracks.",
        }
        protection_gap = max(0.0, packet["protection_floor"] - b["protection_score"])
        custom_risk = min(
            packet["residual_risk_limit"] * 1.55,
            c["residual_exposure"] * (1.0 - min(b["protection_score"], 95.0) / 240.0)
            + 14.0 * a["ratios"]["threat_surface"]
            + 0.45 * protection_gap,
        )
        custom = {
            "name": "Custom from Parts A-C",
            "privacy_epsilon": a["privacy_epsilon"],
            "latency_ms": b["latency_ms"],
            "utility_loss_pp": b["utility_loss_pp"],
            "evidence_score": min(100.0, (a["evidence_score"] + c["audit_score"]) / 2.0),
            "deletion_days": c["deletion_window_days"],
            "residual_risk": custom_risk,
            "rationale": "Uses the student's current threat, control, and lineage settings.",
        }
        return {
            "broad": v2_13_assess_policy(packet, broad),
            "privacy": v2_13_assess_policy(packet, privacy),
            "strict": v2_13_assess_policy(packet, strict),
            "custom": v2_13_assess_policy(packet, custom),
        }

    v2_13_d_policies = v2_13_policy_candidates(v2_13_packet, v2_13_a, v2_13_b, v2_13_c)
    v2_13_selected_policy = v2_13_d_policies[v2_13_partD_policy_choice.value]
    v2_13_rejected_policy_result = v2_13_d_policies[v2_13_rejected_policy.value]
    v2_13_broad_policy = v2_13_d_policies["broad"]
    return (
        v2_13_broad_policy,
        v2_13_d_policies,
        v2_13_policy_candidates,
        v2_13_rejected_policy_result,
        v2_13_selected_policy,
    )


@app.cell
def _(apply_plotly_theme, go, pd, v2_13_fmt_ms, v2_13_fmt_number, v2_13_fmt_pct, v2_13_guardrail_badge):
    def v2_13_color(colors, key, fallback):
        try:
            return colors[key]
        except Exception:
            return fallback

    def v2_13_ratio_fig(colors, title, ratios, labels):
        keys = list(labels)
        values = [ratios[key] for key in keys]
        bar_colors = [
            v2_13_color(colors, "RedLine", "#dc2626") if value > 1.0 else v2_13_color(colors, "BlueLine", "#2563eb")
            for value in values
        ]
        fig = go.Figure()
        fig.add_bar(
            x=[labels[key] for key in keys],
            y=values,
            marker_color=bar_colors,
            text=[f"{value:.2f}x" for value in values],
            textposition="outside",
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color=v2_13_color(colors, "RedLine", "#dc2626"))
        fig.update_layout(
            title=title,
            yaxis_title="Guardrail ratio (1.0 is the limit)",
            height=380,
            showlegend=False,
        )
        apply_plotly_theme(fig)
        return fig

    def v2_13_part_a_table(packet, a):
        return pd.DataFrame(
            [
                {
                    "Amount": "Threat surface",
                    "Value": f"{a['surface_index']:.1f} index",
                    "Limit": f"{packet['surface_limit']:.1f} index",
                    "Ratio": f"{a['ratios']['threat_surface']:.2f}x",
                    "Status": v2_13_guardrail_badge(a["ratios"]["threat_surface"] <= 1.0),
                },
                {
                    "Amount": "Privacy budget",
                    "Value": f"{a['privacy_epsilon']:.2f} epsilon",
                    "Limit": f"{packet['privacy_budget_limit']:.2f} epsilon",
                    "Ratio": f"{a['ratios']['privacy_budget']:.2f}x",
                    "Status": v2_13_guardrail_badge(a["ratios"]["privacy_budget"] <= 1.0),
                },
                {
                    "Amount": "Evidence lineage",
                    "Value": v2_13_fmt_pct(a["evidence_score"]),
                    "Limit": f">= {v2_13_fmt_pct(packet['evidence_floor'])}",
                    "Ratio": f"{a['ratios']['evidence_lineage']:.2f}x",
                    "Status": v2_13_guardrail_badge(a["ratios"]["evidence_lineage"] <= 1.0),
                },
            ]
        )

    def v2_13_part_b_table(packet, b):
        return pd.DataFrame(
            [
                {
                    "Amount": "Latency",
                    "Value": v2_13_fmt_ms(b["latency_ms"]),
                    "Limit": v2_13_fmt_ms(packet["latency_limit_ms"]),
                    "Ratio": f"{b['ratios']['latency']:.2f}x",
                    "Status": v2_13_guardrail_badge(b["ratios"]["latency"] <= 1.0),
                },
                {
                    "Amount": "Utility loss",
                    "Value": f"{b['utility_loss_pp']:.2f} pp",
                    "Limit": f"{packet['utility_loss_limit_pp']:.2f} pp",
                    "Ratio": f"{b['ratios']['utility']:.2f}x",
                    "Status": v2_13_guardrail_badge(b["ratios"]["utility"] <= 1.0),
                },
                {
                    "Amount": "Governance overhead",
                    "Value": f"{b['governance_items']:.1f} evidence items",
                    "Limit": f"{packet['governance_limit_items']:.1f} items",
                    "Ratio": f"{b['ratios']['governance']:.2f}x",
                    "Status": v2_13_guardrail_badge(b["ratios"]["governance"] <= 1.0),
                },
                {
                    "Amount": "Protection score",
                    "Value": v2_13_fmt_pct(b["protection_score"]),
                    "Limit": f">= {v2_13_fmt_pct(packet['protection_floor'])}",
                    "Ratio": f"{b['ratios']['protection']:.2f}x",
                    "Status": v2_13_guardrail_badge(b["ratios"]["protection"] <= 1.0),
                },
            ]
        )

    def v2_13_part_c_table(packet, c):
        return pd.DataFrame(
            [
                {
                    "Lineage amount": "Access roles",
                    "Value": f"{c['access_roles']:.0f} roles",
                    "Limit": f"{packet['role_limit']:.0f} roles",
                    "Ratio": f"{c['ratios']['access_roles']:.2f}x",
                    "Status": v2_13_guardrail_badge(c["ratios"]["access_roles"] <= 1.0),
                },
                {
                    "Lineage amount": "Retention",
                    "Value": f"{c['retention_days']} days",
                    "Limit": f"{packet['retention_limit_days']:.0f} days",
                    "Ratio": f"{c['ratios']['retention']:.2f}x",
                    "Status": v2_13_guardrail_badge(c["ratios"]["retention"] <= 1.0),
                },
                {
                    "Lineage amount": "Deletion",
                    "Value": f"{c['deletion_window_days']} days",
                    "Limit": f"{packet['deletion_sla_days']:.0f} days",
                    "Ratio": f"{c['ratios']['deletion']:.2f}x",
                    "Status": v2_13_guardrail_badge(c["ratios"]["deletion"] <= 1.0),
                },
                {
                    "Lineage amount": "Audit evidence",
                    "Value": v2_13_fmt_pct(c["audit_score"]),
                    "Limit": f">= {v2_13_fmt_pct(packet['evidence_floor'])}",
                    "Ratio": f"{c['ratios']['audit_gap']:.2f}x",
                    "Status": v2_13_guardrail_badge(c["ratios"]["audit_gap"] <= 1.0),
                },
                {
                    "Lineage amount": "Residual exposure",
                    "Value": f"{c['residual_exposure']:.1f} index",
                    "Limit": f"{packet['residual_risk_limit']:.1f} index",
                    "Ratio": f"{c['residual_exposure'] / packet['residual_risk_limit']:.2f}x",
                    "Status": v2_13_guardrail_badge(c["residual_exposure"] <= packet["residual_risk_limit"]),
                },
            ]
        )

    def v2_13_policy_table(d_policies, guardrail_label):
        rows = []
        for key in ("broad", "privacy", "strict", "custom"):
            policy = d_policies[key]
            rows.append(
                {
                    "Policy": policy["name"],
                    "Privacy epsilon": f"{policy['privacy_epsilon']:.2f}",
                    "Latency": v2_13_fmt_ms(policy["latency_ms"]),
                    "Utility loss": f"{policy['utility_loss_pp']:.2f} pp",
                    "Evidence": v2_13_fmt_pct(policy["evidence_score"]),
                    "Deletion": f"{policy['deletion_days']:.0f} days",
                    "Residual risk": f"{policy['residual_risk']:.1f}",
                    "Binding": guardrail_label(policy["binding"]),
                    "Feasible": v2_13_guardrail_badge(policy["feasible"]),
                }
            )
        return pd.DataFrame(rows)

    def v2_13_policy_fig(colors, d_policies):
        keys = ("broad", "privacy", "strict", "custom")
        labels = [d_policies[key]["name"] for key in keys]
        fig = go.Figure()
        for guardrail_key, guardrail_label in (
            ("privacy", "Privacy"),
            ("latency", "Latency"),
            ("utility", "Utility"),
            ("evidence", "Evidence gap"),
            ("deletion", "Deletion"),
            ("residual", "Residual risk"),
        ):
            fig.add_bar(
                name=guardrail_label,
                x=labels,
                y=[d_policies[key]["ratios"][guardrail_key] for key in keys],
            )
        fig.add_hline(y=1.0, line_dash="dash", line_color=v2_13_color(colors, "RedLine", "#dc2626"))
        fig.update_layout(
            title="Part D policy guardrail ratios",
            barmode="group",
            yaxis_title="Guardrail ratio (1.0 is the limit)",
            height=430,
            legend_orientation="h",
        )
        apply_plotly_theme(fig)
        return fig

    return (
        v2_13_color,
        v2_13_part_a_table,
        v2_13_part_b_table,
        v2_13_part_c_table,
        v2_13_policy_fig,
        v2_13_policy_table,
        v2_13_ratio_fig,
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
    v2_13_escape,
    v2_13_metadata,
    v2_13_packet,
    v2_13_profile,
    v2_13_variant,
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
                                Vol 2 &middot; Lab 13 &middot; Security and Privacy
                            </div>
                            <div style="font-size:2rem; font-weight:800; color:#f1f5f9; line-height:1.15; margin-bottom:10px;">
                                The Price of Privacy
                            </div>
                            <div style="font-size:0.95rem; color:#94a3b8; max-width:760px; line-height:1.6;">
                                {v2_13_escape(v2_13_variant.workload_summary)} The shared concept sequence treats
                                threat surface, privacy budget, control overhead, access/deletion lineage, and residual risk
                                as amounts that must fit the track envelope.
                            </div>
                        </div>
                        <div style="display:flex; flex-direction:column; gap:8px; flex-shrink:0;">
                            <span class="badge badge-info">{v2_13_escape(v2_13_profile.label)}</span>
                            <span class="badge badge-info">{v2_13_escape(v2_13_packet['ops_unit'])}</span>
                            <span class="badge badge-info">{v2_13_escape(v2_13_packet['hardware_ref'])}</span>
                            <span class="badge badge-warn">45-55 minutes &middot; 4 Parts + Synthesis</span>
                        </div>
                    </div>
                </div>
                """
            ),
            track_context(v2_13_profile),
            track_arc_context(v2_13_profile, v2_13_metadata.lab_id),
            source_trace(
                {
                    "chapter": "Volume II, Chapter 13: Security and Privacy",
                    "anchors": (
                        "Expanded attack surface",
                        "Threat model fields: asset, boundary, adversary, control",
                        "Model extraction defenses and output leakage",
                        "Privacy budget composition",
                        "Security and privacy maturity model",
                    ),
                    "track_source": v2_13_packet["source_policy"],
                    "implementation": "Notebook-local v2_13_ amount-system formulas; shared track metadata, source trace, ledger, and report helpers.",
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v2_13_escape, v2_13_packet):
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
                <div>1. <strong>Threat/privacy surface:</strong> measure the amount that binds first.</div>
                <div>2. <strong>Control overhead:</strong> trade protection against latency, utility, and governance cost.</div>
                <div>3. <strong>Lineage:</strong> access, retention, deletion, and audit evidence determine residual exposure.</div>
                <div>4. <strong>Policy:</strong> privacy, latency, utility, evidence, deletion, and risk guardrails must all pass.</div>
            </div>
            <div style="border-top:1px solid {COLORS['Border']}; margin:16px -28px 0 -28px; padding:16px 28px 0 28px;
                        font-size:0.86rem; color:{COLORS['TextSec']}; line-height:1.65;">
                <strong>Track lens:</strong> {v2_13_escape(v2_13_packet['stakeholder'])} protects
                <strong>{v2_13_escape(v2_13_packet['protected_asset'])}</strong>.
                Trust boundary: {v2_13_escape(v2_13_packet['trust_boundary'])}.
                Natural failure: {v2_13_escape(v2_13_packet['natural_failure'])}.
            </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    build_lab_report,
    ledger,
    mo,
    report_export_panel,
    v2_13_a,
    v2_13_access_model,
    v2_13_aggregation_policy,
    v2_13_audit_sampling_pct,
    v2_13_b,
    v2_13_broad_policy,
    v2_13_c,
    v2_13_chapter,
    v2_13_color,
    v2_13_compute_boundary,
    v2_13_control_strength,
    v2_13_d_policies,
    v2_13_deletion_window_days,
    v2_13_escape,
    v2_13_fmt_ms,
    v2_13_fmt_number,
    v2_13_fmt_pct,
    v2_13_guardrail_label,
    v2_13_lineage_coverage_pct,
    v2_13_logging_scope,
    v2_13_memo_note,
    v2_13_metadata,
    v2_13_output_policy,
    v2_13_packet,
    v2_13_partA_checkpoint,
    v2_13_partA_pred,
    v2_13_partB_checkpoint,
    v2_13_partB_pred,
    v2_13_partC_checkpoint,
    v2_13_partC_pred,
    v2_13_partD_checkpoint,
    v2_13_partD_policy_choice,
    v2_13_partD_pred,
    v2_13_part_a_table,
    v2_13_part_b_table,
    v2_13_part_c_table,
    v2_13_policy_fig,
    v2_13_policy_table,
    v2_13_prediction_feedback,
    v2_13_privacy_accesses,
    v2_13_profile,
    v2_13_ratio_fig,
    v2_13_rejected_policy,
    v2_13_rejected_policy_result,
    v2_13_retention_days,
    v2_13_robustness_implication,
    v2_13_selected_policy,
    v2_13_sensitive_paths,
    v2_13_student_id,
    v2_13_surface_nodes,
    v2_13_variant,
):
    def v2_13_gate(pred_widget, items):
        if pred_widget.value is None:
            items.append(
                mo.callout(
                    mo.md("Commit to the structured prediction first; the instrument is hidden until the prior is explicit."),
                    kind="warn",
                )
            )
            return True
        return False

    def v2_13_feedback(predicted, actual, labels):
        kind, message = v2_13_prediction_feedback(predicted, actual, labels)
        return mo.callout(mo.md(message), kind=kind)

    def v2_13_status_callout(ok, success, failure):
        return mo.callout(mo.md(success if ok else failure), kind="success" if ok else "danger")

    def v2_13_build_part_a():
        labels = {
            "threat_surface": "threat surface",
            "privacy_budget": "privacy budget",
            "evidence_lineage": "audit/evidence lineage",
        }
        items = [
            mo.md(
                f"""
                ### Scenario
                You are the {v2_13_packet['stakeholder']}. Before selecting a control, you must
                model the asset, trust boundary, adversary, and control path for
                `{v2_13_packet['protected_asset']}`. The first task is to identify which amount
                is binding: exposed surface, privacy budget, or evidence lineage.
                """
            ),
            v2_13_partA_pred,
        ]
        if v2_13_gate(v2_13_partA_pred, items):
            return mo.vstack(items)
        items.extend(
            [
                v2_13_feedback(v2_13_partA_pred.value, v2_13_a["binding_key"], labels),
                mo.hstack([v2_13_surface_nodes, v2_13_sensitive_paths], widths="equal"),
                mo.hstack([v2_13_privacy_accesses, v2_13_logging_scope], widths="equal"),
                v2_13_ratio_fig(
                    COLORS,
                    "Part A threat surface, privacy spend, and evidence ratios",
                    v2_13_a["ratios"],
                    {
                        "threat_surface": "Threat surface",
                        "privacy_budget": "Privacy budget",
                        "evidence_lineage": "Evidence lineage",
                    },
                ),
                v2_13_part_a_table(v2_13_packet, v2_13_a),
                v2_13_status_callout(
                    v2_13_a["ok"],
                    f"Threat/privacy envelope holds. Binding amount: `{v2_13_a['binding']}` at {v2_13_a['binding_ratio']:.2f}x its limit.",
                    f"Boundary violation. `{v2_13_a['binding']}` is at {v2_13_a['binding_ratio']:.2f}x its limit. Reduce sensitive paths, segment nodes, spend fewer privacy-consuming accesses, or add scoped evidence.",
                ),
                mo.accordion(
                    {
                        "Math Peek / source model": mo.md(
                            f"""
                            Threat surface index = sensitive paths x path weight + communication channels x coupling + logging exposure.

                            Current pair channels: `{v2_13_surface_nodes.value} * ({v2_13_surface_nodes.value} - 1) / 2`
                            = `{v2_13_fmt_number(v2_13_a['pair_channels'])}`.

                            Privacy spend composes as summed epsilon:
                            `{v2_13_privacy_accesses.value}` accesses x `{v2_13_packet['epsilon_per_access']:.2f}`
                            plus logging-path spend = `{v2_13_a['privacy_epsilon']:.2f}` epsilon.

                            Evidence lineage compares the current logging/evidence score
                            `{v2_13_fmt_pct(v2_13_a['evidence_score'])}` with the track floor
                            `{v2_13_fmt_pct(v2_13_packet['evidence_floor'])}`.
                            """
                        )
                    }
                ),
                v2_13_partA_checkpoint,
            ]
        )
        return mo.vstack(items)

    def v2_13_build_part_b():
        labels = {
            "latency": "latency overhead",
            "utility": "utility loss",
            "governance": "governance overhead",
            "protection": "remaining protection gap",
        }
        items = [
            mo.md(
                f"""
                ### Scenario
                The threat model now has to become a control stack. The selected stack must
                reduce leakage and attacker economics without breaking `{v2_13_packet['ops_unit']}`
                latency, utility, or governance budgets.
                """
            ),
            v2_13_partB_pred,
        ]
        if v2_13_gate(v2_13_partB_pred, items):
            return mo.vstack(items)
        items.extend(
            [
                v2_13_feedback(v2_13_partB_pred.value, v2_13_b["binding_key"], labels),
                mo.hstack([v2_13_control_strength, v2_13_compute_boundary], widths="equal"),
                mo.hstack([v2_13_output_policy, v2_13_aggregation_policy], widths="equal"),
                v2_13_ratio_fig(
                    COLORS,
                    "Part B control stack overhead and protection ratios",
                    v2_13_b["ratios"],
                    {
                        "latency": "Latency",
                        "utility": "Utility",
                        "governance": "Governance",
                        "protection": "Protection gap",
                    },
                ),
                v2_13_part_b_table(v2_13_packet, v2_13_b),
                v2_13_status_callout(
                    v2_13_b["ok"],
                    f"Control stack is feasible. `{v2_13_b['control_stack']}` reaches a protection score of `{v2_13_fmt_pct(v2_13_b['protection_score'])}` with binding overhead `{v2_13_b['binding']}`.",
                    f"Control stack boundary fails. `{v2_13_b['binding']}` is at {v2_13_b['binding_ratio']:.2f}x its guardrail. Adjust isolation, output exposure, aggregation mode, or control strength.",
                ),
                mo.accordion(
                    {
                        "Math Peek / source model": mo.md(
                            f"""
                            Latency = base inference + compute boundary + output policy + aggregation policy + strength tax.

                            Current latency: `{v2_13_fmt_ms(v2_13_b['latency_ms'])}` against
                            `{v2_13_fmt_ms(v2_13_packet['latency_limit_ms'])}`.

                            Utility loss grows with privacy noise/output restriction:
                            `{v2_13_b['utility_loss_pp']:.2f}` percentage points against a
                            `{v2_13_packet['utility_loss_limit_pp']:.2f}` pp guardrail.

                            Governance overhead counts the evidence items needed to prove the
                            control remains active after deployment. Protection score is compared
                            with the track floor because weak controls can pass latency while still
                            leaving the threat model underdefended.
                            """
                        )
                    }
                ),
                v2_13_partB_checkpoint,
            ]
        )
        return mo.vstack(items)

    def v2_13_build_part_c():
        labels = {
            "access_roles": "access roles",
            "retention": "retention record-days",
            "deletion": "deletion window",
            "audit_gap": "audit evidence gap",
        }
        items = [
            mo.md(
                f"""
                ### Scenario
                An auditor or data subject asks which records, logs, checkpoints, and outputs
                still carry their influence. The answer depends on access breadth, retention,
                deletion timing, lineage coverage, and audit sampling, not on access control alone.
                """
            ),
            v2_13_partC_pred,
        ]
        if v2_13_gate(v2_13_partC_pred, items):
            return mo.vstack(items)
        items.extend(
            [
                v2_13_feedback(v2_13_partC_pred.value, v2_13_c["binding_key"], labels),
                mo.hstack([v2_13_access_model, v2_13_retention_days, v2_13_deletion_window_days], widths="equal"),
                mo.hstack([v2_13_lineage_coverage_pct, v2_13_audit_sampling_pct], widths="equal"),
                v2_13_ratio_fig(
                    COLORS,
                    "Part C access, retention, deletion, and evidence ratios",
                    v2_13_c["ratios"],
                    {
                        "access_roles": "Access roles",
                        "retention": "Retention",
                        "deletion": "Deletion",
                        "audit_gap": "Audit gap",
                    },
                ),
                v2_13_part_c_table(v2_13_packet, v2_13_c),
                v2_13_status_callout(
                    v2_13_c["ok"],
                    f"Lineage envelope holds. Residual exposure is `{v2_13_c['residual_exposure']:.1f}` against a limit of `{v2_13_packet['residual_risk_limit']:.1f}`.",
                    f"Lineage boundary fails. `{v2_13_c['binding']}` is at {v2_13_c['binding_ratio']:.2f}x its guardrail or residual exposure exceeds the track limit. Tighten access, shorten retention/deletion, or add auditable lineage.",
                ),
                mo.accordion(
                    {
                        "Math Peek / source model": mo.md(
                            f"""
                            Residual exposure is a weighted amount:

                            `0.30 * access_ratio + 0.24 * retention_ratio + 0.24 * deletion_ratio + 0.22 * audit_gap`.

                            Current access model `{v2_13_c['access_label']}` exposes
                            `{v2_13_c['access_roles']:.0f}` roles. Retention is
                            `{v2_13_c['retention_days']}` days, deletion window is
                            `{v2_13_c['deletion_window_days']}` days, and audit score is
                            `{v2_13_fmt_pct(v2_13_c['audit_score'])}`.

                            The chapter maturity model treats retention policy and reproducible
                            control checks as governance evidence: without lineage, deletion and
                            recovery are claims rather than verifiable controls.
                            """
                        )
                    }
                ),
                v2_13_partC_checkpoint,
            ]
        )
        return mo.vstack(items)

    def v2_13_build_part_d():
        labels = {
            "residual": "residual risk",
            "privacy": "privacy budget",
            "evidence": "audit evidence",
            "deletion": "deletion lineage",
        }
        selected_violations = ", ".join(v2_13_selected_policy["violations"]) or "none"
        rejected_violations = ", ".join(v2_13_rejected_policy_result["violations"]) or "none"
        items = [
            mo.md(
                f"""
                ### Scenario
                The release review needs one security/privacy policy. A policy is valid only
                when privacy budget, latency, utility, evidence, deletion, and residual-risk
                guardrails pass together.
                """
            ),
            v2_13_partD_pred,
        ]
        if v2_13_gate(v2_13_partD_pred, items):
            return mo.vstack(items)
        items.extend(
            [
                v2_13_feedback(v2_13_partD_pred.value, v2_13_broad_policy["binding"], labels),
                v2_13_policy_fig(COLORS, v2_13_d_policies),
                v2_13_policy_table(v2_13_d_policies, v2_13_guardrail_label),
                mo.hstack([v2_13_partD_policy_choice, v2_13_rejected_policy], widths="equal"),
                v2_13_status_callout(
                    v2_13_selected_policy["feasible"],
                    f"Selected policy passes all guardrails. Binding guardrail is `{v2_13_guardrail_label(v2_13_selected_policy['binding'])}`.",
                    f"Selected policy is not deployable. Violations: `{selected_violations}`. Rejected alternative violations: `{rejected_violations}`.",
                ),
                mo.accordion(
                    {
                        "Math Peek / source model": mo.md(
                            f"""
                            Policy feasibility is a conjunction:

                            `privacy_ok and latency_ok and utility_ok and evidence_ok and deletion_ok and residual_ok`.

                            Current selected policy `{v2_13_selected_policy['name']}` has:
                            epsilon `{v2_13_selected_policy['privacy_epsilon']:.2f}`,
                            latency `{v2_13_fmt_ms(v2_13_selected_policy['latency_ms'])}`,
                            utility loss `{v2_13_selected_policy['utility_loss_pp']:.2f}` pp,
                            evidence `{v2_13_fmt_pct(v2_13_selected_policy['evidence_score'])}`,
                            deletion `{v2_13_selected_policy['deletion_days']:.0f}` days,
                            residual risk `{v2_13_selected_policy['residual_risk']:.1f}`.

                            This is why a local mechanism cannot be treated as a system guarantee:
                            one failed predicate invalidates the deployment.
                            """
                        )
                    }
                ),
                v2_13_partD_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_synthesis():
        complete_widgets = (
            ("Part A prediction", v2_13_partA_pred),
            ("Part A checkpoint", v2_13_partA_checkpoint),
            ("Part B prediction", v2_13_partB_pred),
            ("Part B checkpoint", v2_13_partB_checkpoint),
            ("Part C prediction", v2_13_partC_pred),
            ("Part C checkpoint", v2_13_partC_checkpoint),
            ("Part D prediction", v2_13_partD_pred),
            ("Part D checkpoint", v2_13_partD_checkpoint),
            ("Selected security/privacy policy", v2_13_partD_policy_choice),
            ("Rejected alternative", v2_13_rejected_policy),
            ("V2-14 robustness implication", v2_13_robustness_implication),
        )
        incomplete = [label for label, widget in complete_widgets if widget.value is None]
        robustness_text = v2_13_packet["v2_14_options"].get(
            v2_13_robustness_implication.value,
            "Select a V2-14 robustness implication to complete the memo.",
        )
        binding_guardrail = v2_13_guardrail_label(v2_13_selected_policy["binding"])
        residual_risk_text = (
            f"{v2_13_selected_policy['residual_risk']:.1f} residual-risk index "
            f"against {v2_13_packet['residual_risk_limit']:.1f} limit"
        )
        snapshot = {
            "track_id": v2_13_profile.track_id,
            "scenario_id": v2_13_variant.scenario_id,
            "selected_policy": v2_13_selected_policy["name"],
            "rejected_policy": v2_13_rejected_policy_result["name"],
            "binding_guardrail": binding_guardrail,
            "residual_risk": residual_risk_text,
            "partA": {
                "binding": v2_13_a["binding"],
                "surface_index": round(v2_13_a["surface_index"], 4),
                "privacy_epsilon_spend": round(v2_13_a["privacy_epsilon"], 4),
                "evidence_score": round(v2_13_a["evidence_score"], 4),
                "ok": v2_13_a["ok"],
            },
            "partB": {
                "control_stack": v2_13_b["control_stack"],
                "latency_ms": round(v2_13_b["latency_ms"], 4),
                "utility_loss_pp": round(v2_13_b["utility_loss_pp"], 4),
                "governance_items": round(v2_13_b["governance_items"], 4),
                "protection_score": round(v2_13_b["protection_score"], 4),
                "binding_overhead": v2_13_b["binding"],
                "ok": v2_13_b["ok"],
            },
            "partC": {
                "access_model": v2_13_c["access_label"],
                "retention_days": v2_13_c["retention_days"],
                "deletion_window_days": v2_13_c["deletion_window_days"],
                "audit_score": round(v2_13_c["audit_score"], 4),
                "residual_exposure": round(v2_13_c["residual_exposure"], 4),
                "binding_lineage": v2_13_c["binding"],
                "ok": v2_13_c["ok"],
            },
            "partD": {
                "selected_policy_key": v2_13_partD_policy_choice.value,
                "rejected_policy_key": v2_13_rejected_policy.value,
                "binding_guardrail": v2_13_selected_policy["binding"],
                "policy_feasible": v2_13_selected_policy["feasible"],
                "violations": v2_13_selected_policy["violations"],
            },
            "v2_14_robustness_implication": robustness_text,
            "memo_note": v2_13_memo_note.value,
        }
        report = build_lab_report(
            v2_13_metadata,
            student_id=v2_13_student_id.value or "",
            track=v2_13_profile.label,
            scenario=v2_13_variant.workload_summary,
            learning_objectives=(
                "Model threat surface and privacy budget as binding deployment amounts.",
                "Quantify control strength against latency, utility, and governance overhead.",
                "Reason through access, retention, deletion lineage, audit evidence, and residual exposure.",
                "Choose a security/privacy policy that satisfies all guardrails and rejects an invalid alternative.",
            ),
            predictions={
                "part_a_binding_amount": v2_13_partA_pred.value,
                "part_b_binding_overhead": v2_13_partB_pred.value,
                "part_c_residual_exposure_driver": v2_13_partC_pred.value,
                "part_d_broad_policy_binding": v2_13_partD_pred.value,
            },
            knob_settings={
                "surface_nodes": v2_13_surface_nodes.value,
                "sensitive_paths": v2_13_sensitive_paths.value,
                "privacy_accesses": v2_13_privacy_accesses.value,
                "logging_scope": v2_13_logging_scope.value,
                "control_strength": v2_13_control_strength.value,
                "compute_boundary": v2_13_compute_boundary.value,
                "output_policy": v2_13_output_policy.value,
                "aggregation_policy": v2_13_aggregation_policy.value,
                "access_model": v2_13_access_model.value,
                "retention_days": v2_13_retention_days.value,
                "deletion_window_days": v2_13_deletion_window_days.value,
                "lineage_coverage_pct": v2_13_lineage_coverage_pct.value,
                "audit_sampling_pct": v2_13_audit_sampling_pct.value,
            },
            binding_constraints={
                "part_a_binding": v2_13_a["binding"],
                "part_b_binding": v2_13_b["binding"],
                "part_c_binding": v2_13_c["binding"],
                "part_d_binding_guardrail": binding_guardrail,
                "selected_policy_feasible": v2_13_selected_policy["feasible"],
            },
            decisions={
                "part_a_checkpoint": v2_13_partA_checkpoint.value,
                "part_b_checkpoint": v2_13_partB_checkpoint.value,
                "part_c_checkpoint": v2_13_partC_checkpoint.value,
                "part_d_checkpoint": v2_13_partD_checkpoint.value,
                "selected_security_privacy_policy": v2_13_selected_policy["name"],
                "rejected_alternative": v2_13_rejected_policy_result["name"],
                "v2_14_robustness_implication": robustness_text,
            },
            reflections={"memo_note": v2_13_memo_note.value or "Not recorded."},
            residual_risk=(
                "Teaching estimates must be recalibrated against current threat intelligence, production traces, "
                "legal obligations, security architecture, incident records, and live access logs before deployment."
            ),
            evidence_summary={
                "selected_policy": v2_13_selected_policy["name"],
                "binding_guardrail": binding_guardrail,
                "part_a_binding": f"{v2_13_a['binding']} at {v2_13_a['binding_ratio']:.2f}x",
                "part_b_control_stack": v2_13_b["control_stack"],
                "part_c_residual_exposure": residual_risk_text,
                "rejected_alternative": v2_13_rejected_policy_result["name"],
            },
            final_decision={
                "selected_policy": v2_13_selected_policy["name"],
                "binding_amount": binding_guardrail,
                "residual_risk": residual_risk_text,
                "rejected_alternative": v2_13_rejected_policy_result["name"],
                "v2_14_robustness_implication": robustness_text,
            },
            big_takeaways=(
                "Threat modeling turns asset, boundary, adversary, and control into measurable amounts.",
                "Privacy/security controls spend latency, utility, and governance budget.",
                "Access, retention, deletion, and audit lineage determine residual exposure.",
                "A deployable security/privacy policy is a conjunction of guardrails.",
            ),
            source_trace={
                "book_anchor": v2_13_metadata.book_anchor,
                "formulas": (
                    "surface_index = paths * path_weight + pair_channels * channel_weight + logging_surface",
                    "privacy_epsilon_spend = sum(epsilon_i)",
                    "latency = base + compute_boundary + output_policy + aggregation + strength_tax",
                    "residual_exposure = weighted(access, retention, deletion, audit_gap)",
                    "policy_feasible = privacy_ok and latency_ok and utility_ok and evidence_ok and deletion_ok and residual_ok",
                ),
                "track_source": v2_13_packet["source_policy"],
            },
            result_snapshot=snapshot,
            incomplete_fields=tuple(incomplete),
        )
        if not incomplete:
            ledger.save(
                chapter=v2_13_chapter,
                design={
                    "lab_id": v2_13_metadata.lab_id,
                    "track_id": v2_13_profile.track_id,
                    "scenario_id": v2_13_variant.scenario_id,
                    "selected_security_privacy_policy": v2_13_selected_policy["name"],
                    "binding_security_privacy_amount": binding_guardrail,
                    "residual_risk": residual_risk_text,
                    "rejected_alternative": v2_13_rejected_policy_result["name"],
                    "v2_14_robustness_implication": robustness_text,
                    "policy_feasible": v2_13_selected_policy["feasible"],
                    "result_snapshot": snapshot,
                },
            )
        status = "SAVED" if not incomplete else "INCOMPLETE"
        status_kind = "success" if not incomplete else "warn"
        return mo.vstack(
            [
                mo.md(
                    f"""
                    ### Security/Privacy Memo

                    **Report frame:** {v2_13_packet['report_frame']}

                    **Selected policy:** `{v2_13_selected_policy['name']}`

                    **Binding amount:** `{binding_guardrail}`

                    **Residual risk:** {residual_risk_text}

                    **Rejected alternative:** `{v2_13_rejected_policy_result['name']}`

                    **V2-14 implication:** {robustness_text}
                    """
                ),
                mo.hstack([v2_13_student_id, v2_13_robustness_implication], widths="equal"),
                v2_13_memo_note,
                mo.callout(
                    mo.md(
                        f"**Status:** {status}. "
                        + (
                            "Complete all predictions, checkpoints, policy choices, and the V2-14 implication before final save."
                            if incomplete
                            else "Ledger snapshot saved for downstream labs."
                        )
                    ),
                    kind=status_kind,
                ),
                report_export_panel(report),
            ]
        )

    v2_13_tabs = mo.ui.tabs(
        {
            "Part A - Surface Budget": v2_13_build_part_a(),
            "Part B - Control Overhead": v2_13_build_part_b(),
            "Part C - Lineage": v2_13_build_part_c(),
            "Part D - Policy": v2_13_build_part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    v2_13_tabs
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    mo,
    v2_13_a,
    v2_13_b,
    v2_13_c,
    v2_13_fmt_ms,
    v2_13_guardrail_label,
    v2_13_profile,
    v2_13_selected_policy,
):
    _complete = v2_13_selected_policy["feasible"] and v2_13_a["ok"] and v2_13_b["ok"] and v2_13_c["ok"]
    _status = "POLICY PASS" if _complete else "BOUNDARY ACTIVE"
    _status_color = COLORS["GreenLine"] if _complete else COLORS["OrangeLine"]
    mo.Html(
        f"""
        <div class="lab-hud">
            <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab 13</span></div>
            <div><span class="hud-label">TRACK</span> <span class="hud-value">{v2_13_profile.label}</span></div>
            <div><span class="hud-label">PART A</span> <span class="hud-value">{v2_13_a['binding']}</span></div>
            <div><span class="hud-label">CONTROL</span> <span class="hud-value">{v2_13_b['binding']}</span></div>
            <div><span class="hud-label">LINEAGE</span> <span class="hud-value">{v2_13_c['binding']}</span></div>
            <div><span class="hud-label">LATENCY</span> <span class="hud-value">{v2_13_fmt_ms(v2_13_selected_policy['latency_ms'])}</span></div>
            <div><span class="hud-label">POLICY</span> <span class="hud-value">{v2_13_guardrail_label(v2_13_selected_policy['binding'])}</span></div>
            <div><span class="hud-label">STATUS</span> <span style="color:{_status_color}; font-family:var(--font-mono);">{_status}</span></div>
        </div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
