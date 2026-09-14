import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


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
    v2_13_lab_path = "vol2/lab_13_security_privacy.py"
    v2_13_chapter = 13
    v2_13_metadata = get_lab_metadata(v2_13_lab_path)
    return v2_13_chapter, v2_13_metadata


@app.cell(hide_code=True)
def _(mo):
    v2_13_track_picker = mo.ui.dropdown(
        options={
            "⚡ TinyML Track (ARM Cortex-M55 / ESP32-S3 & On-Device Telemetry Scrubbing)": "oura_ring",
            "📱 Mobile Track (Apple Silicon / Snapdragon & Differential Privacy vs Utility)": "iphone",
            "🤖 Edge & Embodied Track (NVIDIA Jetson AGX Orin & Geofenced Video Sanitization)": "robotaxi",
            "☁️ Cloud Supercomputing Track (H100/B200 Multi-Tenant Enclaves & Model Extraction Defenses)": "cloud_fleet",
        },
        value="☁️ Cloud Supercomputing Track (H100/B200 Multi-Tenant Enclaves & Model Extraction Defenses)",
        label="Select Course / Industry Track",
    )
    v2_13_track_picker
    return (v2_13_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    v2_13_metadata,
    v2_13_track_picker,
):
    v2_13_track_id = v2_13_track_picker.value
    v2_13_profile = get_track_profile(v2_13_track_id)
    v2_13_variant = get_lab_track_variant(v2_13_metadata.lab_id, v2_13_track_id)
    return v2_13_profile, v2_13_variant


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


@app.cell(hide_code=True)
def _(ACADEMIC_LAB_CSS, mo, v2_13_packet, v2_13_profile, v2_13_variant):
    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell">
      <div class="mlsysbook-lab-header" style="--mlsysbook-accent: #A51C30;">
        <div class="mlsysbook-meta">
          ML SYSTEMS TEXTBOOK &middot; VOLUME II &middot; CHAPTER 13 &middot; LAB 13
        </div>
        <h1 style="margin: 8px 0 4px 0; color: #0F172A; font-weight: 800; font-size: 1.85rem; letter-spacing: -0.02em;">
          Security and Privacy: Threat Surfaces, Differential Privacy &amp; Lineage Control
        </h1>
        <p style="margin: 0 0 14px 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">
          Treat security and privacy as quantifiable amount systems across the lifecycle: bound attack surfaces and cumulative privacy budgets (&epsilon;),
          balance cryptographic and sanitization overhead against latency and utility, enforce immutable access and deletion lineage, and verify multi-guardrail release policies.
        </p>
        <div class="mlsysbook-chip-row" style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;">
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Track:</strong> {v2_13_profile.label}
          </span>
          <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155;">
            <strong>Stakeholder:</strong> {v2_13_variant.stakeholder}
          </span>
          <span class="mlsysbook-chip" style="background: #F8FAFC; color: #475569;">
            <strong>Hardware:</strong> {v2_13_packet['hardware_ref']}
          </span>
          <span class="mlsysbook-chip" style="background: #F8FAFC; color: #475569;">
            <strong>Protected Asset:</strong> {v2_13_packet['protected_asset']}
          </span>
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Primary Metric:</strong> {v2_13_variant.primary_metric}
          </span>
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Guardrail:</strong> {v2_13_variant.guardrail_metric}
          </span>
        </div>
      </div>

      <div class="mlsysbook-panel" style="margin-bottom: 20px;">
        <h3 style="margin: 0 0 8px 0; color: #0F172A; font-size: 1.15rem;">
          System Scenario: {v2_13_profile.label} Security &amp; Privacy Boundary
        </h3>
        <p style="margin: 0 0 12px 0; font-size: 0.92rem; color: #334155; line-height: 1.55;">
          {v2_13_variant.workload_summary} Security and privacy are amount systems rather than binary compliance checklists. Deploying machine learning models exposes sprawling attack surfaces spanning data ingestion, fine-tuning APIs, model weights, and inference outputs. Every query consumes differential privacy epsilon, every protective cryptographic enclave imposes latency and throughput penalties, and incomplete deletion lineage creates persistent legal and adversarial vulnerability.
        </p>
        <div style="background: #F8FAFC; border-left: 4px solid #006395; padding: 12px 16px; border-radius: 4px; font-size: 0.9rem; color: #1E293B;">
          <strong>The Architectural Invariants of ML Security &amp; Privacy:</strong>
          <ul class="mlsysbook-list" style="margin: 8px 0 4px 0;">
            <li><strong>The Threat Surface &amp; Privacy Budget Invariant:</strong> Attack surface and privacy leakage are consumable amounts (<i>S</i><sub>threat</sub> = &sum; <i>w</i><sub>i</sub> <i>N</i><sub>i</sub>, &epsilon;<sub>total</sub> = &sum; &epsilon;<sub>j</sub>). Unaudited telemetry and fine-tuning accesses steadily deplete safety reserves.</li>
            <li><strong>The Protection vs. System Overhead Law:</strong> Confidential computing and differential privacy exact measurable performance taxes: <i>T</i><sub>latency</sub> = <i>T</i><sub>base</sub> + &Delta;<i>T</i><sub>crypto</sub>, Utility = Utility<sub>base</sub> &minus; &Delta;<i>U</i>(&sigma;<sub>noise</sub>). Misconfigured controls induce massive latency blowups (TEE/FHE) or catastrophic utility degradation.</li>
            <li><strong>Lineage &amp; Deletion Invariant:</strong> Regulatory and security compliance requires provable data deletion within binding SLAs (<i>T</i><sub>del</sub> &le; SLA<sub>del</sub>) backed by verified lineage traces (<i>E</i><sub>audit</sub> &ge; <i>E</i><sub>floor</sub>).</li>
            <li><strong>Conjunctive Security Guardrail Bundle:</strong> A policy is deployable only when all independent security guardrails pass simultaneously: Launchable = Surface<sub>pass</sub> &and; Privacy<sub>pass</sub> &and; Latency<sub>pass</sub> &and; Lineage<sub>pass</sub> &and; Risk<sub>pass</sub>.</li>
          </ul>
        </div>
      </div>
    </div>
    """)
    mo.vstack([ACADEMIC_LAB_CSS, header_html])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v2_13_packet, v2_13_profile):
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
                <li><strong>Quantify threat surface &amp; privacy budget:</strong> calculate attack surfaces and cumulative differential privacy spend for {v2_13_profile.label}.</li>
                <li><strong>Evaluate defense overhead:</strong> balance cryptographic protection (TEE/HE) and DP noise against serving latency and model utility.</li>
                <li><strong>Track access, retention &amp; deletion lineage:</strong> measure audit readiness and verify data deletion SLAs to bound residual exposure.</li>
                <li><strong>Authorize conjunctive security policies:</strong> enforce multi-guardrail release gates ensuring all safety dimensions pass together.</li>
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
                    Threat modeling &middot; Differential privacy &middot; Confidential computing &middot; Data lineage
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~50 min</strong><br/>
                    A: 10 &middot; B: 15 &middot; C: 10 &middot; D: 15 min
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
                &ldquo;When securing {v2_13_packet['ops_unit']}, which constraint binds first:
                threat surface expansion, differential privacy budget depletion, serving latency overhead,
                or deletion lineage verification?&rdquo;
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
            <li><strong>Threat Surfaces in ML Systems:</strong> data poisoning, prompt injection, model inversion, and membership inference attacks.</li>
            <li><strong>Differential Privacy &amp; Privacy Budgets:</strong> Renyi DP, composition theorems, noise mechanisms, and utility trade-offs.</li>
            <li><strong>Confidential Computing &amp; Cryptographic Controls:</strong> Trusted Execution Environments (TEEs), secure multi-party computation, and homomorphic encryption.</li>
            <li><strong>Data Provenance &amp; Compliance:</strong> right-to-be-forgotten, machine unlearning, and immutable audit logs.</li>
        </ul>
    </div>
    """)
    return


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
def _(
    v2_13_a,
    v2_13_assess_policy,
    v2_13_b,
    v2_13_c,
    v2_13_packet,
    v2_13_partD_policy_choice,
    v2_13_rejected_policy,
):
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
        v2_13_rejected_policy_result,
        v2_13_selected_policy,
    )


@app.cell
def _(
    apply_plotly_theme,
    go,
    pd,
    v2_13_fmt_ms,
    v2_13_fmt_pct,
    v2_13_guardrail_badge,
):
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
        v2_13_part_a_table,
        v2_13_part_b_table,
        v2_13_part_c_table,
        v2_13_policy_fig,
        v2_13_policy_table,
        v2_13_ratio_fig,
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
    v2_13_a,
    v2_13_access_model,
    v2_13_aggregation_policy,
    v2_13_audit_sampling_pct,
    v2_13_b,
    v2_13_broad_policy,
    v2_13_c,
    v2_13_chapter,
    v2_13_compute_boundary,
    v2_13_control_strength,
    v2_13_d_policies,
    v2_13_deletion_window_days,
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
    def v2_13_feedback(predicted, actual, labels):
        kind, message = v2_13_prediction_feedback(predicted, actual, labels)
        return mo.callout(mo.md(message), kind=kind)

    def v2_13_status_callout(ok, success, failure):
        return mo.callout(mo.md(success if ok else failure), kind="success" if ok else "danger")

    def build_part_a():
        labels = {
            "threat_surface": "threat surface",
            "privacy_budget": "privacy budget",
            "evidence_lineage": "audit/evidence lineage",
        }
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Threat Surface &amp; Privacy Briefing &middot; {v2_13_packet['stakeholder']}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;Deploying {v2_13_profile.label} exposes sensitive pathways to model extraction, data reconstruction,
                    and unauthorized telemetry leakage. Before configuring protections, we must quantify which amount is binding:
                    exposed communication nodes, cumulative differential privacy epsilon, or audit lineage.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; {v2_13_packet['stakeholder']} &middot; {v2_13_profile.label}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_13_partA_pred,
                title="1. Formulate Threat Surface & Privacy Budget Hypothesis",
                subtitle=(
                    f"Scenario: Threat modeling for {v2_13_packet['protected_asset']}. "
                    "Predict which amount becomes the primary binding constraint under baseline deployment."
                ),
            ),
        ]
        if v2_13_partA_pred.value is None:
            return mo.vstack(items)

        items.extend(
            [
                v2_13_feedback(v2_13_partA_pred.value, v2_13_a["binding_key"], labels),
                instrumentation_console(
                    mo.vstack([
                        mo.hstack([v2_13_surface_nodes, v2_13_sensitive_paths], widths="equal"),
                        mo.hstack([v2_13_privacy_accesses, v2_13_logging_scope], widths="equal"),
                    ]),
                    title="Threat Surface & Privacy Budget Instrumentation",
                    subtitle=f"Adjust exposed topology nodes, sensitive ingress paths, privacy-consuming query access counts, and logging scope for {v2_13_profile.label}",
                ),
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
                mo.Html(f"""
                <div class="mlsysbook-panel" style="border-left: 4px solid #A51C30; margin-top: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                    <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part A Attack Surface &amp; Budget Commitment</h4>
                    <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                        Binding constraint: <code>{v2_13_a['binding']}</code> ({v2_13_a['binding_ratio']:.2f}x limit).
                        Select your architecture mitigation decision:
                    </p>
                    {v2_13_partA_checkpoint}
                </div>
                """),
                MathPeek(
                    r"S_{\text{threat}} = N_{\text{paths}} \cdot w_{\text{path}} + \frac{N(N-1)}{2} \cdot w_{\text{channel}} + L_{\text{logging}}, \quad \varepsilon_{\text{total}} = \sum_{j=1}^m \varepsilon_j",
                    {
                        "surface nodes (N)": f"{v2_13_surface_nodes.value}",
                        "pair communication channels": f"{v2_13_fmt_number(v2_13_a['pair_channels'])}",
                        "sensitive ingress paths": f"{v2_13_sensitive_paths.value}",
                        "surface index": f"{v2_13_a['surface_index']:.2f} (limit: {v2_13_packet['surface_index_limit']:.2f})",
                        "privacy-consuming queries": f"{v2_13_privacy_accesses.value}",
                        "epsilon per query": f"{v2_13_packet['epsilon_per_access']:.2f}",
                        "total privacy spend": f"{v2_13_a['privacy_epsilon']:.2f} \u03b5 (budget: {v2_13_packet['privacy_budget_epsilon']:.2f} \u03b5)",
                        "evidence score": f"{v2_13_fmt_pct(v2_13_a['evidence_score'])} (floor: {v2_13_fmt_pct(v2_13_packet['evidence_floor'])})",
                        "chapter source": "Volume II, Chapter 13: Expanded Attack Surface & Privacy Budget Accounting",
                    },
                ),
            ]
        )
        return mo.vstack(items)

    def build_part_b():
        labels = {
            "latency": "latency overhead",
            "utility": "utility loss",
            "governance": "governance overhead",
            "protection": "remaining protection gap",
        }
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Control Stack Briefing &middot; {v2_13_packet['stakeholder']}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;Security mechanisms are not costless abstractions. Confidential computing enclaves add memory bus overhead,
                    differential privacy noise degrades prediction quality, and output filters introduce token latency. We must find the
                    Pareto boundary between protection strength and production service feasibility.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; {v2_13_packet['stakeholder']} &middot; {v2_13_profile.label}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_13_partB_pred,
                title="2. Formulate Control Overhead & Utility Tradeoff Hypothesis",
                subtitle="Predict which operational dimension breaks first when ramping up defense isolation and differential privacy.",
            ),
        ]
        if v2_13_partB_pred.value is None:
            return mo.vstack(items)

        items.extend(
            [
                v2_13_feedback(v2_13_partB_pred.value, v2_13_b["binding_key"], labels),
                instrumentation_console(
                    mo.vstack([
                        mo.hstack([v2_13_control_strength, v2_13_compute_boundary], widths="equal"),
                        mo.hstack([v2_13_output_policy, v2_13_aggregation_policy], widths="equal"),
                    ]),
                    title="Defense Mechanisms & Operational Overhead Controls",
                    subtitle=f"Select cryptographic boundary, output token sanitation, aggregation isolation, and overall defense strength for {v2_13_profile.label}",
                ),
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
                mo.Html(f"""
                <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                    <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part B Control Stack Selection Decision</h4>
                    <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                        Current protection score: <code>{v2_13_fmt_pct(v2_13_b['protection_score'])}</code>
                        (latency: <code>{v2_13_fmt_ms(v2_13_b['latency_ms'])}</code>, utility loss: <code>{v2_13_b['utility_loss_pp']:.2f} pp</code>).
                        Commit your defense configuration:
                    </p>
                    {v2_13_partB_checkpoint}
                </div>
                """),
                MathPeek(
                    r"T_{\text{latency}} = T_{\text{base}} + \Delta T_{\text{boundary}} + \Delta T_{\text{output}} + \Delta T_{\text{agg}} + \Delta T_{\text{tax}}, \quad \Delta U \propto \sigma_{\text{noise}}",
                    {
                        "base inference latency": f"{v2_13_fmt_ms(v2_13_packet['base_latency_ms'])}",
                        "total defense latency": f"{v2_13_fmt_ms(v2_13_b['latency_ms'])} (limit: {v2_13_fmt_ms(v2_13_packet['latency_limit_ms'])})",
                        "utility penalty": f"{v2_13_b['utility_loss_pp']:.2f} pp (limit: {v2_13_packet['utility_loss_limit_pp']:.2f} pp)",
                        "governance verification items": f"{v2_13_b['governance_items']:.0f} (limit: {v2_13_packet['governance_limit']:.0f})",
                        "protection score achieved": f"{v2_13_fmt_pct(v2_13_b['protection_score'])} (floor: {v2_13_fmt_pct(v2_13_packet['protection_floor'])})",
                        "chapter source": "Volume II, Chapter 13: Model Extraction Defenses, Enclaves & Differential Privacy",
                    },
                ),
            ]
        )
        return mo.vstack(items)

    def build_part_c():
        labels = {
            "access_roles": "access roles",
            "retention": "retention record-days",
            "deletion": "deletion window",
            "audit_gap": "audit evidence gap",
        }
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Lineage &amp; Compliance Briefing &middot; {v2_13_packet['stakeholder']}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;When a user or regulator invokes the 'right to be forgotten' or an audit discovery request arrives,
                    a lack of immutable data lineage turns compliance into a crisis. We must trace training checkpoints, fine-tuning
                    shards, and telemetry caches, proving verifiable deletion within rigid SLA windows.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; {v2_13_packet['stakeholder']} &middot; {v2_13_profile.label}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_13_partC_pred,
                title="3. Formulate Data Lineage, Retention & Deletion Hypothesis",
                subtitle="Predict which lifecycle parameter drives residual exposure risk when managing sensitive records.",
            ),
        ]
        if v2_13_partC_pred.value is None:
            return mo.vstack(items)

        items.extend(
            [
                v2_13_feedback(v2_13_partC_pred.value, v2_13_c["binding_key"], labels),
                instrumentation_console(
                    mo.vstack([
                        mo.hstack([v2_13_access_model, v2_13_retention_days, v2_13_deletion_window_days], widths="equal"),
                        mo.hstack([v2_13_lineage_coverage_pct, v2_13_audit_sampling_pct], widths="equal"),
                    ]),
                    title="Access Breadth, Retention Windows & Lineage Verification",
                    subtitle=f"Configure role-based access granularity, data retention horizons, deletion SLAs, and automated audit coverage for {v2_13_profile.label}",
                ),
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
                mo.Html(f"""
                <div class="mlsysbook-panel" style="border-left: 4px solid #1F407A; margin-top: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                    <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part C Lineage &amp; Deletion Governance Choice</h4>
                    <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                        Residual exposure index: <code>{v2_13_c['residual_exposure']:.1f}</code>
                        (limit: <code>{v2_13_packet['residual_risk_limit']:.1f}</code>, audit score: <code>{v2_13_fmt_pct(v2_13_c['audit_score'])}</code>).
                        Confirm your compliance architecture:
                    </p>
                    {v2_13_partC_checkpoint}
                </div>
                """),
                MathPeek(
                    r"\text{Residual Exposure} = 0.30 R_{\text{access}} + 0.24 R_{\text{retention}} + 0.24 R_{\text{deletion}} + 0.22 \Delta_{\text{audit}}",
                    {
                        "access model": f"{v2_13_c['access_label']} ({v2_13_c['access_roles']:.0f} roles)",
                        "retention horizon": f"{v2_13_c['retention_days']} days (limit: {v2_13_packet['retention_limit_days']} d)",
                        "deletion SLA": f"{v2_13_c['deletion_window_days']} days (limit: {v2_13_packet['deletion_sla_days']} d)",
                        "audit completeness score": f"{v2_13_fmt_pct(v2_13_c['audit_score'])} (floor: {v2_13_fmt_pct(v2_13_packet['audit_floor'])})",
                        "residual exposure score": f"{v2_13_c['residual_exposure']:.1f} (limit: {v2_13_packet['residual_risk_limit']:.1f})",
                        "chapter source": "Volume II, Chapter 13: Data Lineage, Retention Policies & Verifiable Unlearning",
                    },
                ),
            ]
        )
        return mo.vstack(items)

    def build_part_d():
        labels = {
            "residual": "residual risk",
            "privacy": "privacy budget",
            "evidence": "audit evidence",
            "deletion": "deletion lineage",
        }
        selected_violations = ", ".join(v2_13_selected_policy["violations"]) or "none"
        rejected_violations = ", ".join(v2_13_rejected_policy_result["violations"]) or "none"
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Policy Authorization Briefing &middot; {v2_13_packet['stakeholder']}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;Security and privacy authorization requires a conjunctive release gate: every guardrail—privacy epsilon,
                    serving latency, utility floor, audit lineage, deletion SLA, and residual risk—must pass simultaneously.
                    A failure on any single dimension invalidates deployment.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; {v2_13_packet['stakeholder']} &middot; {v2_13_profile.label}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_13_partD_pred,
                title="4. Formulate Conjunctive Security & Privacy Policy Hypothesis",
                subtitle="Predict which guardrail causes naive permissive or broad-access deployment policies to fail.",
            ),
        ]
        if v2_13_partD_pred.value is None:
            return mo.vstack(items)

        items.extend(
            [
                v2_13_feedback(v2_13_partD_pred.value, v2_13_broad_policy["binding"], labels),
                v2_13_policy_fig(COLORS, v2_13_d_policies),
                v2_13_policy_table(v2_13_d_policies, v2_13_guardrail_label),
                instrumentation_console(
                    mo.hstack([v2_13_partD_policy_choice, v2_13_rejected_policy], widths="equal"),
                    title="Release Candidate Policy Selection",
                    subtitle="Select the candidate policy to authorize for production deployment and the explicit rejected alternative",
                ),
                v2_13_status_callout(
                    v2_13_selected_policy["feasible"],
                    f"Selected policy passes all guardrails. Binding guardrail is `{v2_13_guardrail_label(v2_13_selected_policy['binding'])}`.",
                    f"Selected policy is not deployable. Violations: `{selected_violations}`. Rejected alternative violations: `{rejected_violations}`.",
                ),
                mo.Html(f"""
                <div class="mlsysbook-panel" style="border-left: 4px solid #16A34A; margin-top: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                    <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part D Production Policy Authorization</h4>
                    <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                        Candidate policy: <strong>{v2_13_selected_policy['name']}</strong> &mdash; Status: <code>{'FEASIBLE' if v2_13_selected_policy['feasible'] else 'UNSAFE'}</code>.
                        Confirm release authorization:
                    </p>
                    {v2_13_partD_checkpoint}
                </div>
                """),
                MathPeek(
                    r"\text{Deployable} = \varepsilon \le \varepsilon_{\text{lim}} \land T_{\text{lat}} \le T_{\text{lim}} \land \Delta U \le \Delta U_{\text{lim}} \land E_{\text{audit}} \ge E_{\text{floor}} \land T_{\text{del}} \le \text{SLA} \land \text{Risk} \le \text{Risk}_{\text{lim}}",
                    {
                        "selected policy": v2_13_selected_policy["name"],
                        "privacy epsilon": f"{v2_13_selected_policy['privacy_epsilon']:.2f} \u03b5",
                        "latency": f"{v2_13_fmt_ms(v2_13_selected_policy['latency_ms'])}",
                        "utility loss": f"{v2_13_selected_policy['utility_loss_pp']:.2f} pp",
                        "audit evidence": f"{v2_13_fmt_pct(v2_13_selected_policy['evidence_score'])}",
                        "deletion timeline": f"{v2_13_selected_policy['deletion_days']:.0f} days",
                        "residual risk": f"{v2_13_selected_policy['residual_risk']:.1f}",
                        "chapter source": "Volume II, Chapter 13: Conjunctive Guardrail Architectures & Release Verification",
                    },
                ),
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

        items = [
            mo.md("## Synthesis &mdash; Security and Privacy Policy Memo"),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #1F407A; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">STUDENT MEMO &amp; REFLECTIONS</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Security &amp; Privacy Architecture Memo</h4>
                {v2_13_student_id}
                <div style="margin-top: 12px;">{v2_13_robustness_implication}</div>
                <div style="margin-top: 12px;">{v2_13_memo_note}</div>
            </div>
            """),
            mo.callout(
                mo.md(
                    f"**Memo summary:** Selected `{v2_13_selected_policy['name']}` (binding guardrail: `{binding_guardrail}`); "
                    f"residual risk: `{residual_risk_text}`; rejected `{v2_13_rejected_policy_result['name']}`.  \n\n"
                    f"**V2-14 robustness implication:** {robustness_text}"
                ),
                kind=status_kind,
            ),
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
            mo.Html(f"""
            <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
                <div style="flex:1; min-width:220px; background:white; border:1px solid {COLORS['Border']};
                            border-radius:10px; padding:16px; border-top:3px solid {COLORS['GreenLine']};">
                    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']}; text-transform:uppercase;">
                        Selected release policy</div>
                    <div style="font-size:1.05rem; font-weight:800; color:{COLORS['Text']}; margin-top:5px;">
                        {v2_13_selected_policy['name']}</div>
                </div>
                <div style="flex:1; min-width:220px; background:white; border:1px solid {COLORS['Border']};
                            border-radius:10px; padding:16px; border-top:3px solid {COLORS['OrangeLine']};">
                    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']}; text-transform:uppercase;">
                        Binding guardrail</div>
                    <div style="font-size:1.05rem; font-weight:800; color:{COLORS['Text']}; margin-top:5px;">
                        {binding_guardrail}</div>
                </div>
                <div style="flex:1; min-width:220px; background:white; border:1px solid {COLORS['Border']};
                            border-radius:10px; padding:16px; border-top:3px solid {COLORS['RedLine']};">
                    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']}; text-transform:uppercase;">
                        Rejected alternative</div>
                    <div style="font-size:1.05rem; font-weight:800; color:{COLORS['Text']}; margin-top:5px;">
                        {v2_13_rejected_policy_result['name']}</div>
                </div>
            </div>
            """),
            big_takeaways([
                "Threat modeling turns asset, boundary, adversary, and control into measurable amounts.",
                "Privacy/security controls spend latency, utility, and governance budget.",
                "Access, retention, deletion, and audit lineage determine residual exposure.",
                "A deployable security/privacy policy is a conjunction of guardrails.",
            ]),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #A51C30; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">FINAL VERIFICATION &amp; SIGN-OFF</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Lead Security &amp; Privacy Architect Authorization</h4>
                <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #475569;">
                    Confirm your security architecture, verify that all 6 independent guardrails pass, and export the signed privacy assurance report.
                </p>
            </div>
            """),
            report_export_panel(report),
            mo.Html(f"""
            <div style="display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        What's Next
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        <strong>Lab V2-14: Robust AI: Distribution Shift, Outliers &amp; Adversarial Perturbations</strong> &mdash;
                        Carry forward the selected security policy and residual risk ceiling. The next challenge is verifying robustness
                        against distributional drift, adversarial feature attacks, and sensor corruptions.
                    </div>
                </div>
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        Upstream Precedent
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        <strong>Lab V2-12: ML Operations at Scale</strong> provided the error budget and canary release harness.
                        Security and privacy controls now establish the cryptographic trust boundary and lineage guarantees.
                    </div>
                </div>
            </div>
            """),
        ]
        return mo.vstack(items)

    v2_13_tabs = mo.ui.tabs(
        {
            "Part A - Surface Budget": build_part_a(),
            "Part B - Control Overhead": build_part_b(),
            "Part C - Lineage": build_part_c(),
            "Part D - Policy": build_part_d(),
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
