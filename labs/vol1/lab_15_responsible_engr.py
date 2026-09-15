import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import sys
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
        big_takeaways,
        build_lab_report,
        carbon_budget,
        explanation_overhead,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        metric_conflict,
        report_export_panel,
        resolve_mlsysim_ref,
        responsibility_track_profile,
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
        big_takeaways,
        build_lab_report,
        carbon_budget,
        explanation_overhead,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        metric_conflict,
        mo,
        np,
        report_export_panel,
        resolve_mlsysim_ref,
        responsibility_track_profile,
        track_arc_context,
        track_context,
    )


@app.cell
def _(get_lab_metadata):
    v1_15_metadata = get_lab_metadata("vol1/lab_15_responsible_engr.py")
    return (v1_15_metadata,)


@app.cell(hide_code=True)
def _(ledger, mo):
    _saved_track = ledger.get_track()
    _options = {
        "⚡ TinyML Track (Microcontrollers & Wearables)": "oura_ring",
        "📱 Mobile Track (On-Device Personal AI)": "iphone",
        "🤖 Edge & Embodied Track (Robotics & Autonomous Systems)": "robotaxi",
        "☁️ Cloud Supercomputing Track (H100 & Continuous Training vs Deployment Walls)": "cloud_fleet",
    }
    _default_key = next((k for k, v in _options.items() if v == _saved_track), list(_options.keys())[1])
    # Cross-tier hardware targets: Hardware.Cloud.H100_SXM5_80GB, Hardware.Edge.Jetson_Orin_64GB, Hardware.Mobile.Apple_M4_Unified
    v1_15_track_picker = mo.ui.dropdown(
        options=_options,
        value=_default_key,
        label="Select Course / Industry Track",
    )
    return (v1_15_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    responsibility_track_profile,
    v1_15_track_picker,
):
    v1_15_track_id = v1_15_track_picker.value
    v1_15_profile = get_track_profile(v1_15_track_id)
    v1_15_variant = get_lab_track_variant("v1_15_no_free_fairness", v1_15_profile.track_id)
    v1_15_hardware = resolve_mlsysim_ref(v1_15_variant.hardware_ref)
    v1_15_model = resolve_mlsysim_ref(v1_15_variant.model_ref)
    v1_15_resp = responsibility_track_profile(
        v1_15_profile,
        v1_15_variant,
        v1_15_hardware,
        v1_15_model,
    )
    return v1_15_profile, v1_15_resp, v1_15_variant


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    COLORS,
    LAB_CSS,
    mo,
    track_arc_context,
    track_context,
    v1_15_metadata,
    v1_15_profile,
    v1_15_resp,
    v1_15_track_picker,
    v1_15_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div class="mlsysbook-lab-shell">
          <div style="margin-bottom: 16px;">
            {v1_15_track_picker}
          </div>
          <div class="mlsysbook-lab-header" style="border-left: 6px solid #A51C30; background: #FFFFFF; padding: 24px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
              ML Systems Textbook &middot; Volume I &middot; Chapter 15 &middot; Foundational Lab 15
            </div>
            <h1 style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 0 0 10px 0; line-height: 1.2;">
              Responsible Release Gate: Metric Thresholds, Privacy &amp; Safety Governance
            </h1>
            <p style="font-size: 1.05rem; color: #334155; line-height: 1.6; margin: 0 0 16px 0;">
              {v1_15_variant.workload_summary} Responsible engineering turns people, policy, evidence, and blast radius into release gates. Navigate metric conflicts, allocate differential privacy budgets, enforce canary rollback gates, and produce auditable governance artifacts.
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
              <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
                <strong>Track:</strong> {v1_15_profile.label}
              </span>
              <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
                <strong>Workload:</strong> {v1_15_resp.label}
              </span>
              <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
                <strong>Hardware:</strong> {v1_15_variant.hardware_ref}
              </span>
              <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
                <strong>Model:</strong> {v1_15_variant.model_ref}
              </span>
              <span style="background: #FEF2F2; color: #A51C30; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; border: 1px solid #FECACA;">
                <strong>Primary Focus:</strong> Responsible Engineering &amp; Release Gate
              </span>
              <span style="background: #F1F5F9; color: #0F172A; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; border: 1px solid #CBD5E1;">
                <strong>Deliverable:</strong> responsible release memo
              </span>
            </div>
          </div>

          <div class="mlsysbook-panel" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #0F172A; font-size: 1.15rem; font-weight: 700;">
              System Scenario: {v1_15_profile.label} Responsible Engineering
            </h3>
            <p style="color: #334155; font-size: 0.95rem; line-height: 1.6; margin-bottom: 16px;">
              You are the <strong>{v1_15_resp.stakeholder}</strong> evaluating <strong>{v1_15_variant.model_ref}</strong> on <strong>{v1_15_variant.hardware_ref}</strong>. The system must fulfill its core obligation: <em>{v1_15_resp.obligation}</em>, protecting <strong>{v1_15_resp.harmed_party}</strong> while managing audit signal <strong>{v1_15_resp.audit_signal}</strong> and residual harm <em>{v1_15_resp.residual_harm}</em>.
            </p>
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 16px; margin-bottom: 12px;">
              <div style="font-size: 0.85rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">
                The Architectural Invariants of Responsible Engineering:
              </div>
              <ul class="mlsysbook-list" style="margin: 0; font-size: 0.92rem; color: #1E293B; line-height: 1.6;">
                <li><strong>The Inherent Fairness Impossibility (No-Free-Fairness Invariant):</strong> When base rates differ between demographic or cohort segments (&pi;<sub>A</sub> &ne; &pi;<sub>B</sub>), a non-trivial predictor cannot simultaneously satisfy demographic parity, equalized odds (equal FPR and TPR), and predictive parity: DemographicParity &cap; EqualOdds &cap; PredictiveParity = &empty;. Every threshold policy inevitably transfers asymmetric false-positive or false-negative harm across stakeholders.</li>
                <li><strong>The Differential Privacy Budget Law (&epsilon;-&delta; Degradation Trade-off):</strong> Bounding individual identification risk requires injecting calibrated noise into gradients or query outputs (&Delta;<em>f</em> / &epsilon;). Stronger formal privacy guarantees (lower &epsilon;, strict &delta;) incur an inescapable empirical utility degradation &Delta;<em>L</em> &sim; &Omicron;(1/&epsilon;), requiring explicit data minimization and local on-device retention bounds.</li>
                <li><strong>The Blast-Radius Safety Envelope:</strong> A model cannot be promoted directly to unconstrained production when safety degradation harms human end-users. Safe deployment requires bounding the exposure envelope: AffectedUnits = <em>N</em> &middot; CanaryPct, with automated rollback and human-in-the-loop escalation triggered whenever safety risk breaches the critical threshold: Risk &gt; &tau;<sub>crit</sub> &rArr; Rollback.</li>
                <li><strong>The Verifiable Lineage &amp; Accountability Mandate:</strong> Responsible governance demands end-to-end provenance: every model decision must map deterministically to training lineage (Dataset<sub>ver</sub>, Hyperparams, Commit<sub>hash</sub>), data retention policies, and a named organizational owner accountable for residual post-deployment harm: ModelAudit &rarr; Lineage &times; Consent &times; Ownership.</li>
              </ul>
            </div>
          </div>
        </div>
        """),
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
                    <div style="margin-bottom: 3px;">1. <strong>Choose a threshold policy:</strong>
                        show which stakeholder absorbs false-positive or false-negative harm.</div>
                    <div style="margin-bottom: 3px;">2. <strong>Budget privacy evidence:</strong>
                        connect &epsilon;, minimization, retention, and consent to deployability.</div>
                    <div style="margin-bottom: 3px;">3. <strong>Gate the release:</strong>
                        bound safety risk, blast radius, rollback, and audit accountability.</div>
                </div>
            </div>
            <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
            <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 240px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                        Obligation
                    </div>
                    <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                        {v1_15_resp.obligation}
                    </div>
                </div>
                <div style="flex: 1; min-width: 240px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                        Audit Signal
                    </div>
                    <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                        {v1_15_resp.audit_signal}
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
                    "What responsible design can protect {v1_15_resp.harmed_party}
                    without pretending the overhead is free?"
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']};
                            line-height: 1.6; margin-top: 10px;">
                    Every track follows the same four concepts. The selected track changes
                    persona, constraints, thresholds, evidence emphasis, failure mode, and
                    report framing.
                </div>
            </div>
        </div>
        """),
        track_context(v1_15_profile),
        track_arc_context(v1_15_profile, v1_15_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** - Complete before this lab:

    - **The Responsible Engineering chapter** - metric conflict, fairness trade-offs,
      explainability overhead, privacy constraints, robustness, and sustainability.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo, v1_15_resp):
    partA_pred = mo.ui.radio(
        options={
            "A) Aggregate quality is enough if it stays high": "aggregate",
            "B) The threshold can trade false-positive and false-negative harm across stakeholders": "threshold_tradeoff",
            "C) Raising the threshold always protects the harmed party": "raise_threshold",
            "D) Governance can decide the threshold after launch": "after_launch",
        },
        label=f"{v1_15_resp.label}: what is the first release risk for {v1_15_resp.harmed_party}?",
    )
    partA_policy = mo.ui.radio(
        options={
            "Hold release until the subgroup gap is within target": "hold_for_gap",
            "Ship the shared threshold and monitor after launch": "ship_monitor",
            "Use a track-specific mitigation before release": "mitigate_before_release",
        },
        label="Checkpoint: threshold policy for the release memo",
    )
    return partA_policy, partA_pred


@app.cell(hide_code=True)
def _(mo, v1_15_resp):
    partA_base_a = mo.ui.slider(
        start=5,
        stop=60,
        value=30,
        step=5,
        label=f"{v1_15_resp.subgroup_a} base/context rate (%)",
    )
    partA_base_b = mo.ui.slider(
        start=5,
        stop=60,
        value=10,
        step=5,
        label=f"{v1_15_resp.subgroup_b} base/context rate (%)",
    )
    partA_threshold = mo.ui.slider(start=0.10, stop=0.90, value=0.50, step=0.05, label="Decision threshold")

    partB_pred = mo.ui.radio(
        options={
            "A) Collect more raw data because more evidence is always safer": "more_raw",
            "B) Minimize retained data while keeping enough audit evidence": "minimize_with_evidence",
            "C) Differential privacy only changes the report wording": "privacy_text",
            "D) Consent and retention can be checked after deployment": "after_deploy",
        },
        label=f"How should {v1_15_resp.stakeholder} balance privacy and evidence?",
    )
    return partA_base_a, partA_base_b, partA_threshold, partB_pred


@app.cell(hide_code=True)
def _(mo, v1_15_resp):
    _epsilon_default = 2.0 if v1_15_resp.track_id in {"iphone", "oura_ring"} else 4.0
    _retention_default = 14 if v1_15_resp.track_id in {"iphone", "oura_ring"} else 30
    partB_epsilon = mo.ui.slider(start=0.5, stop=12.0, value=_epsilon_default, step=0.5, label="Privacy budget epsilon")
    partB_retention = mo.ui.slider(start=1, stop=180, value=_retention_default, step=1, label="Retained evidence days")
    partB_raw_pct = mo.ui.slider(
        start=0,
        stop=100,
        value=25,
        step=5,
        label="Raw / sensitive evidence retained (%)",
    )
    partB_local_pct = mo.ui.slider(start=0, stop=100, value=70, step=5, label="Local or federated processing (%)")
    partB_decision = mo.ui.radio(
        options={
            "Deploy with this privacy evidence policy": "deploy_privacy_policy",
            "Revise collection to increase audit evidence": "revise_evidence",
            "Hold for consent, retention, or data-card review": "hold_privacy_review",
        },
        label="Checkpoint: privacy evidence decision",
    )

    partC_pred = mo.ui.radio(
        options={
            "A) Average quality is enough if the track metric is strong": "average_quality",
            "B) Safety threshold, canary size, rollback, and fallback bound blast radius": "blast_radius",
            "C) A bigger canary always gives safer evidence": "bigger_canary",
            "D) Human review can replace predeployment safety thresholds": "human_replaces_gate",
        },
        label=f"What makes a staged {v1_15_resp.label} release acceptable?",
    )
    return (
        partB_decision,
        partB_epsilon,
        partB_local_pct,
        partB_raw_pct,
        partB_retention,
        partC_pred,
    )


@app.cell(hide_code=True)
def _(mo, v1_15_resp):
    _safety_default = max(85.0, min(99.0, v1_15_resp.baseline_quality_pct - 1.0))
    _rollback_default = 3 if v1_15_resp.track_id == "robotaxi" else 15
    partC_safety_threshold = mo.ui.slider(
        start=85.0,
        stop=99.5,
        value=_safety_default,
        step=0.5,
        label="Minimum rare-event / safety evidence (%)",
    )
    partC_canary_pct = mo.ui.slider(start=1, stop=50, value=10, step=1, label="Initial canary exposure (%)")
    partC_rollback_minutes = mo.ui.slider(
        start=1,
        stop=120,
        value=_rollback_default,
        step=1,
        label="Rollback or mitigation time (minutes)",
    )
    partC_human_review = mo.ui.slider(start=0, stop=100, value=70, step=5, label="Fallback / human-review coverage (%)")
    partC_decision = mo.ui.radio(
        options={
            "Canary only within the blast-radius cap": "canary",
            "Hold release until safety evidence clears the floor": "hold_safety",
            "Expand release because aggregate quality is strong": "expand",
        },
        label="Checkpoint: safety release decision",
    )

    partD_pred = mo.ui.radio(
        options={
            "A) A model card is enough accountability": "model_card",
            "B) Lineage, immutable logs, decision context, and owner sign-off make evidence accountable": "governance_stack",
            "C) Audit logs only need aggregate counters": "aggregate_logs",
            "D) Governance can be reconstructed manually after an incident": "manual_after",
        },
        label=f"What makes the {v1_15_resp.label} release evidence accountable?",
    )
    return (
        partC_canary_pct,
        partC_decision,
        partC_human_review,
        partC_rollback_minutes,
        partC_safety_threshold,
        partD_pred,
    )


@app.cell(hide_code=True)
def _(mo, v1_15_resp):
    _retention_default = 730 if v1_15_resp.track_id in {"robotaxi", "cloud_fleet"} else 365
    partD_lineage = mo.ui.slider(start=0, stop=100, value=80, step=5, label="Lineage coverage (%)")
    partD_log_retention = mo.ui.slider(
        start=1,
        stop=2190,
        value=_retention_default,
        step=30,
        label="Immutable audit retention (days)",
    )
    partD_context = mo.ui.slider(start=0, stop=100, value=75, step=5, label="Prediction decision-context logging (%)")
    partD_access_review = mo.ui.slider(
        start=1,
        stop=90,
        value=30,
        step=1,
        label="Access review cadence (days)",
    )
    partD_owner = mo.ui.radio(
        options={
            "No named owner yet": "none",
            "Single accountable owner named": "single_owner",
            "Cross-functional owner and reviewer named": "cross_functional",
        },
        label="Accountable release owner",
    )
    partD_decision = mo.ui.radio(
        options={
            "Sign off with accountable audit trail": "sign_off",
            "Hold for missing lineage or decision context": "hold_governance",
            "Ship with documented manual reconstruction risk": "manual_risk",
        },
        label="Checkpoint: governance decision",
    )
    return (
        partD_access_review,
        partD_context,
        partD_decision,
        partD_lineage,
        partD_log_retention,
        partD_owner,
    )


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    big_takeaways,
    go,
    metric_conflict,
    mo,
    np,
    partA_base_a,
    partA_base_b,
    partA_policy,
    partA_pred,
    partA_threshold,
    partB_decision,
    partB_epsilon,
    partB_local_pct,
    partB_pred,
    partB_raw_pct,
    partB_retention,
    partC_canary_pct,
    partC_decision,
    partC_human_review,
    partC_pred,
    partC_rollback_minutes,
    partC_safety_threshold,
    partD_access_review,
    partD_context,
    partD_decision,
    partD_lineage,
    partD_log_retention,
    partD_owner,
    partD_pred,
    v1_15_profile,
    v1_15_resp,
    v1_15_variant,
):
    def v1_15_clamp(value, low, high):
        return max(low, min(high, float(value)))

    def v1_15_metric_card(label, value, detail, color, border=False):
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

    def v1_15_track_policy(profile):
        policies = {
            "iphone": {
                "epsilon_limit": 3.0,
                "max_retention_days": 30,
                "min_local_pct": 70,
                "min_evidence_confidence": 70,
                "privacy_risk_cap": 55,
                "safety_floor_pct": 90.0,
                "canary_risk_pp": 0.05,
                "harm_multiplier": 0.08,
                "blast_cap_units": 250,
                "rollback_max_minutes": 30,
                "human_review_min_pct": 60,
                "audit_retention_days": 365,
                "audit_ready_min": 80,
                "evidence_label": "privacy-safe accessibility evidence",
                "blast_unit": "affected sessions",
                "audit_owner": "mobile release owner",
            },
            "oura_ring": {
                "epsilon_limit": 2.0,
                "max_retention_days": 30,
                "min_local_pct": 80,
                "min_evidence_confidence": 65,
                "privacy_risk_cap": 45,
                "safety_floor_pct": 92.0,
                "canary_risk_pp": 0.07,
                "harm_multiplier": 0.15,
                "blast_cap_units": 25,
                "rollback_max_minutes": 60,
                "human_review_min_pct": 70,
                "audit_retention_days": 730,
                "audit_ready_min": 85,
                "evidence_label": "consented biosignal evidence",
                "blast_unit": "wearer-days at risk",
                "audit_owner": "wearable risk owner",
            },
            "robotaxi": {
                "epsilon_limit": 6.0,
                "max_retention_days": 90,
                "min_local_pct": 40,
                "min_evidence_confidence": 90,
                "privacy_risk_cap": 65,
                "safety_floor_pct": 98.0,
                "canary_risk_pp": 0.08,
                "harm_multiplier": 0.05,
                "blast_cap_units": 50,
                "rollback_max_minutes": 5,
                "human_review_min_pct": 95,
                "audit_retention_days": 2190,
                "audit_ready_min": 95,
                "evidence_label": "rare-event replay evidence",
                "blast_unit": "road-user exposures",
                "audit_owner": "safety case owner",
            },
            "cloud_fleet": {
                "epsilon_limit": 5.0,
                "max_retention_days": 90,
                "min_local_pct": 20,
                "min_evidence_confidence": 75,
                "privacy_risk_cap": 60,
                "safety_floor_pct": 92.0,
                "canary_risk_pp": 0.04,
                "harm_multiplier": 0.05,
                "blast_cap_units": 100_000,
                "rollback_max_minutes": 30,
                "human_review_min_pct": 50,
                "audit_retention_days": 1095,
                "audit_ready_min": 90,
                "evidence_label": "tenant/language cohort evidence",
                "blast_unit": "requests or users affected",
                "audit_owner": "platform governance owner",
            },
        }
        return policies.get(profile.track_id, policies["iphone"])

    def v1_15_approval_rate(base_rate_pct, fpr_pct, fnr_pct):
        base_rate = v1_15_clamp(base_rate_pct, 1.0, 99.0) / 100.0
        fpr = v1_15_clamp(fpr_pct, 0.0, 100.0) / 100.0
        tpr = 1.0 - v1_15_clamp(fnr_pct, 0.0, 100.0) / 100.0
        return (base_rate * tpr + (1.0 - base_rate) * fpr) * 100.0

    def v1_15_privacy_result(profile, policy, epsilon, retention_days, raw_pct, local_pct):
        epsilon = v1_15_clamp(epsilon, 0.1, 20.0)
        retention_days = int(v1_15_clamp(retention_days, 1, 3650))
        raw_pct = v1_15_clamp(raw_pct, 0.0, 100.0)
        local_pct = v1_15_clamp(local_pct, 0.0, 100.0)

        raw_records = profile.inference_events_per_day * (raw_pct / 100.0) * retention_days
        evidence_volume_score = min(26.0, np.log10(max(raw_records, 1.0)) * 4.2)
        dp_noise_penalty = max(0.0, policy["epsilon_limit"] - epsilon) * 4.0
        minimization_penalty = local_pct * 0.05 + max(0.0, 35.0 - raw_pct) * 0.15
        retention_credit = min(retention_days, policy["max_retention_days"]) / policy["max_retention_days"] * 12.0
        evidence_confidence = v1_15_clamp(
            38.0 + evidence_volume_score + raw_pct * 0.15 + retention_credit - dp_noise_penalty - minimization_penalty,
            0.0,
            100.0,
        )
        membership_risk = v1_15_clamp(
            10.0
            + raw_pct * 0.35
            + min(1.5, retention_days / policy["max_retention_days"]) * 20.0
            + (epsilon / policy["epsilon_limit"]) * 25.0
            - local_pct * 0.18,
            0.0,
            100.0,
        )
        model_evidence_delta = -dp_noise_penalty * 0.22 - local_pct * 0.015 - max(0.0, 35.0 - raw_pct) * 0.04
        effective_records = raw_records * (epsilon / (epsilon + 2.0)) * (1.0 - local_pct / 300.0)
        violations = []
        if epsilon > policy["epsilon_limit"]:
            violations.append("epsilon budget exceeded")
        if retention_days > policy["max_retention_days"]:
            violations.append("retention exceeds minimization policy")
        if local_pct < policy["min_local_pct"]:
            violations.append("local/federated processing below track floor")
        if evidence_confidence < policy["min_evidence_confidence"]:
            violations.append("audit evidence below confidence floor")
        if membership_risk > policy["privacy_risk_cap"]:
            violations.append("membership-inference risk above cap")
        return {
            "epsilon": epsilon,
            "epsilon_limit": policy["epsilon_limit"],
            "retention_days": retention_days,
            "max_retention_days": policy["max_retention_days"],
            "raw_pct": raw_pct,
            "local_pct": local_pct,
            "raw_records": raw_records,
            "effective_records": effective_records,
            "evidence_confidence": evidence_confidence,
            "min_evidence_confidence": policy["min_evidence_confidence"],
            "membership_risk": membership_risk,
            "privacy_risk_cap": policy["privacy_risk_cap"],
            "model_evidence_delta": model_evidence_delta,
            "deployable": not violations,
            "violations": tuple(violations),
        }

    def v1_15_safety_result(profile, policy, threshold_pct, canary_pct, rollback_minutes, human_review_pct):
        threshold_pct = v1_15_clamp(threshold_pct, 80.0, 100.0)
        canary_pct = v1_15_clamp(canary_pct, 0.1, 100.0)
        rollback_minutes = v1_15_clamp(rollback_minutes, 0.5, 240.0)
        human_review_pct = v1_15_clamp(human_review_pct, 0.0, 100.0)
        rare_event_score = v1_15_clamp(
            profile.baseline_quality_pct
            - canary_pct * policy["canary_risk_pp"]
            + human_review_pct * 0.025
            - max(0.0, rollback_minutes - policy["rollback_max_minutes"]) * 0.02,
            0.0,
            100.0,
        )
        risk_rate = max(0.0005, ((100.0 - rare_event_score) / 100.0) * policy["harm_multiplier"])
        affected_units = profile.inference_events_per_day * (canary_pct / 100.0) * risk_rate
        violations = []
        if threshold_pct < policy["safety_floor_pct"]:
            violations.append("selected threshold below track safety floor")
        if rare_event_score < threshold_pct:
            violations.append("rare-event evidence below selected threshold")
        if affected_units > policy["blast_cap_units"]:
            violations.append("blast-radius cap exceeded")
        if rollback_minutes > policy["rollback_max_minutes"]:
            violations.append("rollback or mitigation too slow")
        if human_review_pct < policy["human_review_min_pct"]:
            violations.append("fallback/human review below minimum")
        return {
            "threshold_pct": threshold_pct,
            "track_floor_pct": policy["safety_floor_pct"],
            "canary_pct": canary_pct,
            "rollback_minutes": rollback_minutes,
            "rollback_max_minutes": policy["rollback_max_minutes"],
            "human_review_pct": human_review_pct,
            "human_review_min_pct": policy["human_review_min_pct"],
            "rare_event_score": rare_event_score,
            "affected_units": affected_units,
            "blast_cap_units": policy["blast_cap_units"],
            "release_ok": not violations,
            "violations": tuple(violations),
        }

    def v1_15_governance_result(profile, policy, lineage_pct, retention_days, context_pct, access_review_days, owner_value):
        lineage_pct = v1_15_clamp(lineage_pct, 0.0, 100.0)
        retention_days = int(v1_15_clamp(retention_days, 1, 3650))
        context_pct = v1_15_clamp(context_pct, 0.0, 100.0)
        access_review_days = int(v1_15_clamp(access_review_days, 1, 365))
        owner_score = {"none": 0.0, "single_owner": 75.0, "cross_functional": 100.0}.get(owner_value, 0.0)
        retention_score = v1_15_clamp(retention_days / policy["audit_retention_days"] * 100.0, 0.0, 100.0)
        access_score = v1_15_clamp(100.0 - max(0, access_review_days - 7) * 1.2, 0.0, 100.0)
        readiness = (
            lineage_pct * 0.30
            + context_pct * 0.25
            + retention_score * 0.20
            + access_score * 0.10
            + owner_score * 0.15
        )
        audit_events_per_year = profile.inference_events_per_day * 365 * (context_pct / 100.0)
        violations = []
        if readiness < policy["audit_ready_min"]:
            violations.append("audit readiness below release gate")
        if lineage_pct < 80:
            violations.append("lineage coverage below 80%")
        if context_pct < 75:
            violations.append("decision-context logging below 75%")
        if retention_days < policy["audit_retention_days"]:
            violations.append("immutable log retention below track obligation")
        if owner_value in {None, "none"}:
            violations.append("no accountable owner named")
        return {
            "lineage_pct": lineage_pct,
            "retention_days": retention_days,
            "required_retention_days": policy["audit_retention_days"],
            "context_pct": context_pct,
            "access_review_days": access_review_days,
            "owner_value": owner_value,
            "owner_score": owner_score,
            "retention_score": retention_score,
            "access_score": access_score,
            "readiness": readiness,
            "audit_ready_min": policy["audit_ready_min"],
            "audit_events_per_year": audit_events_per_year,
            "release_accountable": not violations,
            "violations": tuple(violations),
        }

    v1_15_policy = v1_15_track_policy(v1_15_profile)

    def build_part_a():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Threshold Release Brief &middot; {v1_15_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Aggregate quality is high. Which threshold policy can ship without
                    hiding harm to {v1_15_resp.harmed_party}?"
                </div>
            </div>
            """),
            mo.md(f"""
    ## Part A: Thresholds Encode Values

    **Scenario.** You are the **{v1_15_resp.stakeholder}**. The release board asks
    whether one shared threshold is acceptable for **{v1_15_resp.subgroup_a}** and
    **{v1_15_resp.subgroup_b}**.

    **Concept.** Thresholds encode values. Moving the threshold changes who absorbs
    false-positive and false-negative harm; the aggregate score does not decide that
    for you.
            """),
            partA_pred,
        ]
        if partA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the threshold audit."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_base_a, partA_base_b, partA_threshold], widths="equal"))
        _conflict = metric_conflict(
            v1_15_resp,
            base_rate_a_pct=partA_base_a.value,
            base_rate_b_pct=partA_base_b.value,
            threshold=partA_threshold.value,
        )

        _approval_a = v1_15_approval_rate(_conflict.base_rate_a_pct, _conflict.fpr_a_pct, _conflict.fnr_a_pct)
        _approval_b = v1_15_approval_rate(_conflict.base_rate_b_pct, _conflict.fpr_b_pct, _conflict.fnr_b_pct)
        _metrics = ["Approval/Action", "Accuracy", "FPR", "FNR", "PPV"]
        _a_values = [
            _approval_a,
            _conflict.accuracy_a_pct,
            _conflict.fpr_a_pct,
            _conflict.fnr_a_pct,
            _conflict.ppv_a_pct,
        ]
        _b_values = [
            _approval_b,
            _conflict.accuracy_b_pct,
            _conflict.fpr_b_pct,
            _conflict.fnr_b_pct,
            _conflict.ppv_b_pct,
        ]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(name=_conflict.subgroup_a, x=_metrics, y=_a_values, marker_color=COLORS["BlueLine"], opacity=0.88))
        _fig.add_trace(go.Bar(name=_conflict.subgroup_b, x=_metrics, y=_b_values, marker_color=COLORS["OrangeLine"], opacity=0.88))
        _fig.update_layout(
            barmode="group",
            height=360,
            yaxis=dict(title="Percentage (%)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _gap_color = COLORS["RedLine"] if _conflict.fpr_gap_pp > v1_15_resp.target_gap_pp else COLORS["GreenLine"]
        _threshold_harm = "false-negative misses" if partA_threshold.value >= 0.50 else "false-positive interventions"
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_15_metric_card("Accuracy Gap", f"{_conflict.accuracy_gap_pp:.1f} pp", "aggregate can look stable", COLORS["BlueLine"])}
            {v1_15_metric_card("FPR Gap", f"{_conflict.fpr_gap_pp:.1f} pp", f"target {v1_15_resp.target_gap_pp:.1f} pp", _gap_color, True)}
            {v1_15_metric_card("PPV Gap", f"{_conflict.ppv_gap_pp:.1f} pp", "different stakeholder value", COLORS["OrangeLine"])}
            {v1_15_metric_card("Threshold Bias", _threshold_harm, v1_15_resp.audit_signal, COLORS["RedLine"])}
        </div>
        """))

        items.append(mo.md(f"""
    **Evidence Table - Live Values**

    | Metric | {v1_15_resp.subgroup_a} | {v1_15_resp.subgroup_b} |
    |---|---:|---:|
    | Base/context rate | {_conflict.base_rate_a_pct:.1f}% | {_conflict.base_rate_b_pct:.1f}% |
    | Approval/action rate | {_approval_a:.1f}% | {_approval_b:.1f}% |
    | Accuracy | {_conflict.accuracy_a_pct:.1f}% | {_conflict.accuracy_b_pct:.1f}% |
    | False positive rate | {_conflict.fpr_a_pct:.1f}% | {_conflict.fpr_b_pct:.1f}% |
    | False negative rate | {_conflict.fnr_a_pct:.1f}% | {_conflict.fnr_b_pct:.1f}% |
    | Positive predictive value | {_conflict.ppv_a_pct:.1f}% | {_conflict.ppv_b_pct:.1f}% |
    | Shared threshold | {partA_threshold.value:.2f} | {partA_threshold.value:.2f} |

    *Source: `mlsysbook_labs.metric_conflict`, track `{v1_15_profile.track_id}`.*
        """))

        if _conflict.fpr_gap_pp > v1_15_resp.target_gap_pp:
            items.append(mo.callout(mo.md(
                f"**Gap above target.** {_conflict.conflict_summary} A responsible design needs mitigation and audit evidence."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"**Gap within target.** {_conflict.conflict_summary} The audit still belongs in the report."
            ), kind="success"))

        items.append(mo.accordion({
            "Math Peek: threshold metrics and value trade-offs": mo.md(f"""
    For each group, the notebook computes:

    `TPR = TP / (TP + FN)`, `FPR = FP / (FP + TN)`, and
    `PPV = TP / (TP + FP)`.

    The chapter's fairness section shows why demographic parity, equal opportunity,
    equalized odds, and calibration can conflict when base/context rates differ. In
    this lab, a release fails the Part A gate when
    `abs(FPR_A - FPR_B) > {v1_15_resp.target_gap_pp:.1f} pp`.
            """)
        }))

        if partA_pred.value == "threshold_tradeoff":
            items.append(mo.callout(mo.md("**Correct.** The threshold is a policy choice expressed as a number."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Aggregate quality is insufficient.** The report must name which threshold policy protects the harmed stakeholder and which residual error remains."
            ), kind="warn"))
        items.append(partA_policy)
        if partA_policy.value is None:
            items.append(mo.callout(mo.md("Choose a threshold policy so the release memo records a decision, not only a metric."), kind="warn"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Privacy Evidence Review &middot; {v1_15_resp.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "What evidence can we keep, and what must be minimized, before
                    this release is deployable?"
                </div>
            </div>
            """),
            mo.md(f"""
    ## Part B: Privacy Budget Changes Evidence

    **Scenario.** The release needs **{v1_15_policy['evidence_label']}**. Privacy,
    consent, retention, and local processing decide whether that evidence is lawful
    and strong enough to use.

    **Concept.** Privacy budgets and data minimization change model evidence. Strong
    privacy can reduce deployability risk, but it can also make the audit too weak
    to support the release decision.
            """),
            partB_pred,
        ]
        if partB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the privacy evidence budget."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_epsilon, partB_retention, partB_raw_pct, partB_local_pct], widths="equal"))
        _privacy = v1_15_privacy_result(
            v1_15_resp,
            v1_15_policy,
            partB_epsilon.value,
            partB_retention.value,
            partB_raw_pct.value,
            partB_local_pct.value,
        )
        _ratios = [
            _privacy["epsilon"] / _privacy["epsilon_limit"],
            _privacy["retention_days"] / _privacy["max_retention_days"],
            _privacy["evidence_confidence"] / max(1.0, _privacy["min_evidence_confidence"]),
            _privacy["membership_risk"] / max(1.0, _privacy["privacy_risk_cap"]),
        ]
        _names = ["Epsilon/Limit", "Retention/Limit", "Evidence/Floor", "Privacy Risk/Cap"]
        _colors = [
            COLORS["GreenLine"] if _ratios[0] <= 1 else COLORS["RedLine"],
            COLORS["GreenLine"] if _ratios[1] <= 1 else COLORS["RedLine"],
            COLORS["GreenLine"] if _ratios[2] >= 1 else COLORS["RedLine"],
            COLORS["GreenLine"] if _ratios[3] <= 1 else COLORS["RedLine"],
        ]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=_names, y=_ratios, marker_color=_colors, opacity=0.9))
        _fig.add_hline(y=1.0, line_dash="dash", line_color="#64748b", annotation_text="release boundary")
        _fig.update_layout(
            height=340,
            yaxis=dict(title="Ratio to policy boundary", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _status_color = COLORS["GreenLine"] if _privacy["deployable"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_15_metric_card("Epsilon", f"{_privacy['epsilon']:.1f}", f"limit {_privacy['epsilon_limit']:.1f}", COLORS["BlueLine"])}
            {v1_15_metric_card("Evidence", f"{_privacy['evidence_confidence']:.1f}%", f"floor {_privacy['min_evidence_confidence']:.0f}%", COLORS["OrangeLine"])}
            {v1_15_metric_card("Retained Records", f"{_privacy['raw_records']:,.0f}", f"{_privacy['retention_days']} days", COLORS["RedLine"])}
            {v1_15_metric_card("Privacy Gate", "PASS" if _privacy["deployable"] else "FAIL", ", ".join(_privacy["violations"]) or "no violations", _status_color, True)}
        </div>
        """))

        items.append(mo.md(f"""
    **Privacy Evidence Table - Live Values**

    | Quantity | Value |
    |---|---:|
    | Epsilon budget used | {_privacy['epsilon']:.1f} / {_privacy['epsilon_limit']:.1f} |
    | Retention days | {_privacy['retention_days']} / {_privacy['max_retention_days']} |
    | Raw or sensitive evidence retained | {_privacy['raw_pct']:.0f}% |
    | Local/federated processing | {_privacy['local_pct']:.0f}% |
    | Raw retained evidence records | {_privacy['raw_records']:,.0f} |
    | Effective audit evidence records | {_privacy['effective_records']:,.0f} |
    | Evidence confidence | {_privacy['evidence_confidence']:.1f}% |
    | Model-evidence delta | {_privacy['model_evidence_delta']:+.2f} pp |
    | Membership-inference risk index | {_privacy['membership_risk']:.1f} / {_privacy['privacy_risk_cap']:.0f} |
    | Deployability | {"PASS" if _privacy['deployable'] else "FAIL"} |

    *Source model: notebook-local `v1_15_privacy_result`; chapter anchors: differential privacy, data minimization, retention, and membership-inference validation.*
        """))

        if not _privacy["deployable"]:
            items.append(mo.callout(mo.md(
                "**Privacy/evidence boundary hit:** " + ", ".join(_privacy["violations"]) + ". Change epsilon, retention, raw collection, or local processing before release."
            ), kind="danger"))

        items.append(mo.accordion({
            "Math Peek: epsilon, minimization, and evidence": mo.md(f"""
    Differential privacy bounds each record's influence:

    `Pr[M(D)=o] <= exp(epsilon) * Pr[M(D')=o]`.

    The teaching model treats deployability as a conjunction:

    `epsilon <= {v1_15_policy['epsilon_limit']:.1f}`,
    `retention <= {v1_15_policy['max_retention_days']} days`,
    `evidence_confidence >= {v1_15_policy['min_evidence_confidence']:.0f}%`,
    and `membership_risk <= {v1_15_policy['privacy_risk_cap']:.0f}`.

    Data minimization reduces privacy risk, but it also changes how much evidence
    the model audit can use.
            """)
        }))

        if partB_pred.value == "minimize_with_evidence":
            items.append(mo.callout(mo.md("**Correct.** Privacy is a deployability gate because it changes both risk and usable evidence."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**More raw data is not automatically more responsible.** The defensible point minimizes retained data while preserving enough audit evidence to justify release."
            ), kind="warn"))
        items.append(partB_decision)
        if partB_decision.value is None:
            items.append(mo.callout(mo.md("Choose a privacy evidence decision for the memo."), kind="warn"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Safety Release Gate &middot; {v1_15_resp.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "If the release is wrong, how many people or decisions are
                    touched before rollback stops the harm?"
                </div>
            </div>
            """),
            mo.md(f"""
    ## Part C: Safety Thresholds Bound Blast Radius

    **Scenario.** The selected track wants a staged release. The release is only
    acceptable if the safety evidence clears the selected threshold and the canary
    cannot harm more than the track's blast-radius cap.

    **Concept.** Safety is not the average model score. It is a release gate that
    combines rare-event evidence, exposure size, fallback coverage, and rollback
    time.
            """),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the safety release gate."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_safety_threshold, partC_canary_pct, partC_rollback_minutes, partC_human_review], widths="equal"))
        _safety = v1_15_safety_result(
            v1_15_resp,
            v1_15_policy,
            partC_safety_threshold.value,
            partC_canary_pct.value,
            partC_rollback_minutes.value,
            partC_human_review.value,
        )
        _ratios = [
            _safety["rare_event_score"] / max(1.0, _safety["threshold_pct"]),
            _safety["affected_units"] / max(1.0, _safety["blast_cap_units"]),
            _safety["rollback_minutes"] / max(1.0, _safety["rollback_max_minutes"]),
            _safety["human_review_pct"] / max(1.0, _safety["human_review_min_pct"]),
        ]
        _names = ["Evidence/Threshold", "Blast/Cap", "Rollback/Limit", "Fallback/Min"]
        _colors = [
            COLORS["GreenLine"] if _ratios[0] >= 1 else COLORS["RedLine"],
            COLORS["GreenLine"] if _ratios[1] <= 1 else COLORS["RedLine"],
            COLORS["GreenLine"] if _ratios[2] <= 1 else COLORS["RedLine"],
            COLORS["GreenLine"] if _ratios[3] >= 1 else COLORS["RedLine"],
        ]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=_names, y=_ratios, marker_color=_colors, opacity=0.9))
        _fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["BlueLine"], annotation_text="release boundary")
        _fig.update_layout(
            height=340,
            yaxis=dict(title="Ratio to release gate", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _gate_color = COLORS["GreenLine"] if _safety["release_ok"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_15_metric_card("Safety Evidence", f"{_safety['rare_event_score']:.1f}%", f"threshold {_safety['threshold_pct']:.1f}%", COLORS["BlueLine"])}
            {v1_15_metric_card("Blast Radius", f"{_safety['affected_units']:,.0f}", f"cap {_safety['blast_cap_units']:,.0f}", COLORS["OrangeLine"])}
            {v1_15_metric_card("Rollback", f"{_safety['rollback_minutes']:.0f} min", f"limit {_safety['rollback_max_minutes']:.0f} min", COLORS["RedLine"])}
            {v1_15_metric_card("Release Gate", "PASS" if _safety["release_ok"] else "FAIL", ", ".join(_safety["violations"]) or "no violations", _gate_color, True)}
        </div>
        """))

        items.append(mo.md(f"""
    **Safety And Blast-Radius Table - Live Values**

    | Quantity | Value |
    |---|---:|
    | Selected safety threshold | {_safety['threshold_pct']:.1f}% |
    | Track safety floor | {_safety['track_floor_pct']:.1f}% |
    | Rare-event evidence score | {_safety['rare_event_score']:.1f}% |
    | Initial canary exposure | {_safety['canary_pct']:.0f}% |
    | Estimated blast radius | {_safety['affected_units']:,.0f} {v1_15_policy['blast_unit']} |
    | Blast-radius cap | {_safety['blast_cap_units']:,.0f} {v1_15_policy['blast_unit']} |
    | Rollback or mitigation time | {_safety['rollback_minutes']:.0f} minutes |
    | Fallback / human-review coverage | {_safety['human_review_pct']:.0f}% |
    | Release status | {"PASS" if _safety['release_ok'] else "FAIL"} |

    *Source model: notebook-local `v1_15_safety_result`; chapter anchors: silent failures, predeployment assessment, rollback, kill switches, and incident response.*
        """))

        if not _safety["release_ok"]:
            items.append(mo.callout(mo.md(
                "**Release boundary hit:** " + ", ".join(_safety["violations"]) + ". Reduce exposure, improve fallback, speed rollback, or hold the release."
            ), kind="danger"))

        items.append(mo.accordion({
            "Math Peek: blast radius release gate": mo.md(f"""
    The teaching release gate is a conjunction:

    `rare_event_score >= selected_threshold`,
    `selected_threshold >= track_safety_floor`,
    `affected_units <= blast_radius_cap`,
    `rollback_minutes <= rollback_limit`, and
    `fallback_coverage >= fallback_floor`.

    Here, affected units are approximated as:

    `events_per_day * canary_share * residual_risk_rate`.

    The chapter's silent-failure and incident-response sections motivate this: a
    green uptime dashboard does not bound harm unless detection, rollback, and
    exposure are part of the release gate.
            """)
        }))

        if partC_pred.value == "blast_radius":
            items.append(mo.callout(mo.md("**Correct.** A safe release is bounded by threshold, exposure, fallback, and rollback."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Average quality is not enough.** The release memo must show the rare-event threshold and how far harm can spread before mitigation."
            ), kind="warn"))
        items.append(partC_decision)
        if partC_decision.value is None:
            items.append(mo.callout(mo.md("Choose the safety release decision for the memo."), kind="warn"))
        return mo.vstack(items)

    def build_part_d():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Governance Sign-Off &middot; {v1_15_resp.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Can an auditor reconstruct which data, model, threshold, and
                    owner produced this release decision?"
                </div>
            </div>
            """),
            mo.md(f"""
    ## Part D: Audit Trail Makes Evidence Accountable

    **Scenario.** The release board has technical evidence from Parts A-C. Now it
    needs an accountable trail: lineage, immutable logs, prediction context, access
    review, and an owner.

    **Concept.** Governance converts evidence into a decision someone can audit,
    contest, repair, or roll back. The track owner is **{v1_15_policy['audit_owner']}**.
            """),
            partD_pred,
        ]
        if partD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the audit readiness gate."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_lineage, partD_log_retention, partD_context, partD_access_review, partD_owner], widths="equal"))
        _governance = v1_15_governance_result(
            v1_15_resp,
            v1_15_policy,
            partD_lineage.value,
            partD_log_retention.value,
            partD_context.value,
            partD_access_review.value,
            partD_owner.value,
        )

        _components = {
            "Lineage": _governance["lineage_pct"],
            "Decision Context": _governance["context_pct"],
            "Retention": _governance["retention_score"],
            "Access Review": _governance["access_score"],
            "Owner": _governance["owner_score"],
        }
        _colors = [COLORS["GreenLine"] if _value >= 75 else COLORS["RedLine"] for _value in _components.values()]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=list(_components.keys()), y=list(_components.values()), marker_color=_colors, opacity=0.9))
        _fig.add_hline(y=_governance["audit_ready_min"], line_dash="dash", line_color=COLORS["BlueLine"], annotation_text="readiness gate")
        _fig.update_layout(
            height=340,
            yaxis=dict(title="Component score (%)", gridcolor="#f1f5f9", range=[0, 105]),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _governance_color = COLORS["GreenLine"] if _governance["release_accountable"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_15_metric_card("Readiness", f"{_governance['readiness']:.1f}%", f"gate {_governance['audit_ready_min']:.0f}%", COLORS["BlueLine"])}
            {v1_15_metric_card("Log Retention", f"{_governance['retention_days']:,} days", f"required {_governance['required_retention_days']:,}", COLORS["OrangeLine"])}
            {v1_15_metric_card("Audit Events", f"{_governance['audit_events_per_year']:,.0f}", "decision contexts/year", COLORS["RedLine"])}
            {v1_15_metric_card("Governance", "PASS" if _governance["release_accountable"] else "FAIL", ", ".join(_governance["violations"]) or "no violations", _governance_color, True)}
        </div>
        """))

        items.append(mo.md(f"""
    **Governance Evidence Table - Live Values**

    | Quantity | Value |
    |---|---:|
    | Lineage coverage | {_governance['lineage_pct']:.0f}% |
    | Prediction decision-context logging | {_governance['context_pct']:.0f}% |
    | Immutable audit retention | {_governance['retention_days']:,} days |
    | Required audit retention | {_governance['required_retention_days']:,} days |
    | Access review cadence | every {_governance['access_review_days']} days |
    | Owner score | {_governance['owner_score']:.0f}% |
    | Audit readiness | {_governance['readiness']:.1f}% |
    | Audit events/year | {_governance['audit_events_per_year']:,.0f} |
    | Accountable release | {"PASS" if _governance['release_accountable'] else "FAIL"} |

    *Source model: notebook-local `v1_15_governance_result`; chapter anchors: lineage, audit infrastructure, decision-time logging, and regulatory contestability.*
        """))

        if not _governance["release_accountable"]:
            items.append(mo.callout(mo.md(
                "**Governance boundary hit:** " + ", ".join(_governance["violations"]) + ". The evidence cannot yet support accountable release sign-off."
            ), kind="danger"))

        items.append(mo.accordion({
            "Math Peek: audit volume and reconstructability": mo.md(f"""
    Audit infrastructure must make a release decision reconstructable:

    `audit_events_per_year = decisions_per_day * 365 * context_logging_share`.

    Governance readiness combines lineage, decision context, retention, access
    review, and owner sign-off. The chapter's audit section emphasizes that
    prediction-time logs need feature values, model version, threshold, and output;
    lineage alone cannot answer why a particular decision happened.
            """)
        }))

        if partD_pred.value == "governance_stack":
            items.append(mo.callout(mo.md("**Correct.** Accountability requires reconstructable evidence and a named owner."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Documentation alone is not accountability.** A memo needs lineage, immutable logs, prediction context, and a decision owner."
            ), kind="warn"))
        items.append(partD_decision)
        if partD_decision.value is None:
            items.append(mo.callout(mo.md("Choose the governance decision for the memo."), kind="warn"))
        return mo.vstack(items)

    def build_synthesis():
        _conflict = metric_conflict(
            v1_15_resp,
            base_rate_a_pct=partA_base_a.value,
            base_rate_b_pct=partA_base_b.value,
            threshold=partA_threshold.value,
        )
        _privacy = v1_15_privacy_result(
            v1_15_resp,
            v1_15_policy,
            partB_epsilon.value,
            partB_retention.value,
            partB_raw_pct.value,
            partB_local_pct.value,
        )
        _safety = v1_15_safety_result(
            v1_15_resp,
            v1_15_policy,
            partC_safety_threshold.value,
            partC_canary_pct.value,
            partC_rollback_minutes.value,
            partC_human_review.value,
        )
        _governance = v1_15_governance_result(
            v1_15_resp,
            v1_15_policy,
            partD_lineage.value,
            partD_log_retention.value,
            partD_context.value,
            partD_access_review.value,
            partD_owner.value,
        )
        _all_gates = (
            _conflict.fpr_gap_pp <= v1_15_resp.target_gap_pp
            and _privacy["deployable"]
            and _safety["release_ok"]
            and _governance["release_accountable"]
        )
        _final_status = "READY FOR RESPONSIBLE CANARY" if _all_gates else "HOLD OR REVISE RELEASE"
        _status_color = COLORS["GreenLine"] if _all_gates else COLORS["RedLine"]
        return mo.vstack([
            mo.md("## Synthesis: Responsible Release Memo"),
            mo.callout(mo.md(
                f"**Chapter invariant.** Responsible ML constraints include people and policy. For {v1_15_resp.label}, the release decision is governed by threshold harm, privacy evidence, safety blast radius, and audit accountability."
            ), kind="info"),
            mo.Html(f"""
            <div style="border:2px solid {_status_color}; border-radius:12px; padding:18px 22px;
                        background:white; margin:8px 0 16px 0;">
                <div style="font-size:0.72rem; font-weight:800; color:{_status_color};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px;">
                    Release Decision
                </div>
                <div style="font-size:1.25rem; font-weight:850; color:{_status_color};">
                    {_final_status}
                </div>
                <div style="font-size:0.88rem; color:{COLORS['TextSec']}; margin-top:8px; line-height:1.6;">
                    Harmed stakeholder: {v1_15_resp.harmed_party}. Residual risk:
                    {v1_15_resp.residual_harm}
                </div>
            </div>
            <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        Memo Must State
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        Threshold {partA_threshold.value:.2f}; FPR gap {_conflict.fpr_gap_pp:.1f} pp;
                        epsilon {_privacy['epsilon']:.1f}; safety threshold {_safety['threshold_pct']:.1f}%;
                        blast radius {_safety['affected_units']:,.0f}; audit readiness {_governance['readiness']:.1f}%.
                    </div>
                </div>
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        Carry-Forward Capstone Constraint
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        Lab 16 must treat this responsible release policy as a binding
                        constraint: no final architecture can ship without threshold,
                        privacy, safety, and audit evidence at least as strong as this memo.
                    </div>
                </div>
            </div>
            """),
            mo.Html(f"""
            <div style="border-left: 4px solid #10B981; background: #F0FDF4; border-radius: 0 10px 10px 0; padding: 18px 24px; margin: 16px 0;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #059669; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                    Lead Architect Authorization &middot; Responsible Release Sign-Off
                </div>
                <div style="font-size: 0.95rem; color: #065F46; line-height: 1.6;">
                    <strong>Release Gate Verdict:</strong> {"RELEASE GATE AUTHORIZED" if _all_gates else "HOLD OR REVISE RELEASE GATE"}.
                    {"All fairness, differential privacy, blast radius safety, and audit governance criteria satisfied for " + v1_15_profile.label + "." if _all_gates else "Metric gap, epsilon budget, or safety threshold requires remediation before operational rollout."}
                </div>
            </div>
            """),
            mo.md(f"""
    **Report decisions selected**

    | Module | Checkpoint decision |
    |---|---|
    | Threshold policy | `{partA_policy.value}` |
    | Privacy evidence | `{partB_decision.value}` |
    | Safety release | `{partC_decision.value}` |
    | Governance | `{partD_decision.value}` |

    The memo should name the harmed stakeholder, the residual risk owner, and the
    specific threshold or policy that will carry forward into the capstone audit.
            """),
            big_takeaways([
                "No-Free-Fairness: unequal base rates make demographic parity, equal odds, and predictive parity mutually incompatible.",
                "Differential privacy bounds information leakage at the cost of O(1/ε) utility degradation and compute overhead.",
                "Canary rollouts with automated rollback protect against silent degradation and limit real-world blast radius.",
                "Responsible engineering requires verifiable audit trails, data lineage, and an accountable human owner.",
            ]),
        ])

    v1_15_tabs = mo.ui.tabs({
        "Part A: Threshold Values": build_part_a(),
        "Part B: Privacy Evidence": build_part_b(),
        "Part C: Safety Gate": build_part_c(),
        "Part D: Audit Trail": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    v1_15_tabs
    return (
        v1_15_governance_result,
        v1_15_privacy_result,
        v1_15_safety_result,
        v1_15_track_policy,
    )


@app.cell(hide_code=True)
def _(
    carbon_budget,
    explanation_overhead,
    ledger,
    mo,
    partA_policy,
    partA_pred,
    partA_threshold,
    partB_decision,
    partB_epsilon,
    partB_local_pct,
    partB_pred,
    partB_raw_pct,
    partB_retention,
    partC_canary_pct,
    partC_decision,
    partC_human_review,
    partC_pred,
    partC_rollback_minutes,
    partC_safety_threshold,
    partD_decision,
    partD_lineage,
    partD_log_retention,
    partD_owner,
    partD_pred,
    v1_15_profile,
    v1_15_resp,
    v1_15_variant,
):
    _ready = partA_pred.value is not None and partB_pred.value is not None and partC_pred.value is not None and partD_pred.value is not None
    _completed = _ready and all(
        value is not None
        for value in (
            partA_policy.value,
            partB_decision.value,
            partC_decision.value,
            partD_decision.value,
            partD_owner.value,
        )
    )
    _carbon = carbon_budget(v1_15_resp)
    _explanation = explanation_overhead(v1_15_resp)
    ledger.save(chapter=15, design={
        "chapter": "v1_15",
        "track_id": v1_15_profile.track_id,
        "scenario_id": v1_15_variant.scenario_id,
        "hardware_ref": v1_15_resp.hardware_ref,
        "model_ref": v1_15_resp.model_ref,
        "harmed_party": v1_15_resp.harmed_party,
        "obligation": v1_15_resp.obligation,
        "audit_signal": v1_15_resp.audit_signal,
        "completed": _completed,
        "threshold_prediction": partA_pred.value,
        "threshold_value": partA_threshold.value,
        "threshold_policy": partA_policy.value,
        "privacy_prediction": partB_pred.value,
        "privacy_epsilon": partB_epsilon.value,
        "privacy_retention_days": partB_retention.value,
        "privacy_raw_pct": partB_raw_pct.value,
        "privacy_local_pct": partB_local_pct.value,
        "privacy_decision": partB_decision.value,
        "safety_prediction": partC_pred.value,
        "safety_threshold_pct": partC_safety_threshold.value,
        "safety_canary_pct": partC_canary_pct.value,
        "safety_rollback_minutes": partC_rollback_minutes.value,
        "safety_human_review_pct": partC_human_review.value,
        "safety_decision": partC_decision.value,
        "governance_prediction": partD_pred.value,
        "audit_lineage_pct": partD_lineage.value,
        "audit_retention_days": partD_log_retention.value,
        "audit_owner": partD_owner.value,
        "governance_decision": partD_decision.value,
        "carbon_kgco2_per_year": _carbon.total_kgco2_per_year,
        "carbon_multiplier": _carbon.carbon_multiplier,
        "explanation_latency_ms": _explanation.explanation_latency_ms,
        "explanation_method": _explanation.method,
        "carry_forward_constraint": "threshold, privacy, safety, and audit gates must pass in the V1 capstone",
    })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">15 &middot; Responsible Engineering</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_15_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">OBLIGATION</span>
        <span class="hud-value">{v1_15_resp.obligation}</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">{"SAVED" if _completed else "ACTIVE"}</span>
    </div>
    <div class="mlsysbook-panel">
      <h2>Design Ledger</h2>
      <div class="mlsysbook-grid">
        <div class="mlsysbook-field"><strong>Ready to save</strong>{'yes' if _completed else 'not yet'}</div>
        <div class="mlsysbook-field"><strong>Threshold policy</strong>{partA_policy.value or 'pending'}</div>
        <div class="mlsysbook-field"><strong>Privacy epsilon</strong>&epsilon;={partB_epsilon.value:.1f}</div>
        <div class="mlsysbook-field"><strong>Safety canary</strong>{partC_canary_pct.value}% / {partC_rollback_minutes.value:g}m</div>
        <div class="mlsysbook-field"><strong>Audit owner</strong>{partD_owner.value or 'pending'}</div>
        <div class="mlsysbook-field"><strong>Harmed party</strong>{v1_15_resp.harmed_party}</div>
        <div class="mlsysbook-field"><strong>Carbon / Year</strong>{_carbon.total_kgco2_per_year:,.1f} kgCO2</div>
        <div class="mlsysbook-field"><strong>Explain overhead</strong>+{_explanation.explanation_latency_ms:.1f} ms</div>
      </div>
      <div style="margin-top:10px; color:#475569; line-height:1.55;">
        The ledger records each student decision. All predictions and a final recommendation mark the design complete.
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    metric_conflict,
    mo,
    partA_base_a,
    partA_base_b,
    partA_policy,
    partA_pred,
    partA_threshold,
    partB_decision,
    partB_epsilon,
    partB_local_pct,
    partB_pred,
    partB_raw_pct,
    partB_retention,
    partC_canary_pct,
    partC_decision,
    partC_human_review,
    partC_pred,
    partC_rollback_minutes,
    partC_safety_threshold,
    partD_access_review,
    partD_context,
    partD_decision,
    partD_lineage,
    partD_log_retention,
    partD_owner,
    partD_pred,
    report_export_panel,
    v1_15_governance_result,
    v1_15_metadata,
    v1_15_privacy_result,
    v1_15_profile,
    v1_15_resp,
    v1_15_safety_result,
    v1_15_track_policy,
    v1_15_variant,
):
    _policy = v1_15_track_policy(v1_15_profile)
    _conflict = metric_conflict(
        v1_15_resp,
        base_rate_a_pct=partA_base_a.value,
        base_rate_b_pct=partA_base_b.value,
        threshold=partA_threshold.value,
    )
    _privacy = v1_15_privacy_result(
        v1_15_resp,
        _policy,
        partB_epsilon.value,
        partB_retention.value,
        partB_raw_pct.value,
        partB_local_pct.value,
    )
    _safety = v1_15_safety_result(
        v1_15_resp,
        _policy,
        partC_safety_threshold.value,
        partC_canary_pct.value,
        partC_rollback_minutes.value,
        partC_human_review.value,
    )
    _governance = v1_15_governance_result(
        v1_15_resp,
        _policy,
        partD_lineage.value,
        partD_log_retention.value,
        partD_context.value,
        partD_access_review.value,
        partD_owner.value,
    )

    _incomplete = []
    if partA_pred.value is None:
        _incomplete.append("Part A threshold prediction")
    if partA_policy.value is None:
        _incomplete.append("Part A threshold policy")
    if partB_pred.value is None:
        _incomplete.append("Part B privacy prediction")
    if partB_decision.value is None:
        _incomplete.append("Part B privacy evidence decision")
    if partC_pred.value is None:
        _incomplete.append("Part C safety prediction")
    if partC_decision.value is None:
        _incomplete.append("Part C safety release decision")
    if partD_pred.value is None:
        _incomplete.append("Part D governance prediction")
    if partD_decision.value is None:
        _incomplete.append("Part D governance decision")
    if partD_owner.value is None:
        _incomplete.append("Part D accountable owner")

    _release_ready = (
        _conflict.fpr_gap_pp <= v1_15_resp.target_gap_pp
        and _privacy["deployable"]
        and _safety["release_ok"]
        and _governance["release_accountable"]
    )

    _report = build_lab_report(
        v1_15_metadata,
        track=v1_15_profile.label,
        scenario=v1_15_variant.workload_summary,
        learning_objectives=(
            "Choose a threshold policy that makes stakeholder error trade-offs explicit.",
            "Budget privacy, minimization, retention, and evidence before deployment.",
            "Gate release with safety thresholds, blast radius, rollback, and accountable audit evidence.",
        ),
        predictions={
            "threshold_values": partA_pred.value,
            "privacy_budget": partB_pred.value,
            "safety_blast_radius": partC_pred.value,
            "governance_audit": partD_pred.value,
        },
        knob_settings={
            "base_rate_a_pct": partA_base_a.value,
            "base_rate_b_pct": partA_base_b.value,
            "threshold": partA_threshold.value,
            "threshold_policy": partA_policy.value,
            "epsilon": partB_epsilon.value,
            "retention_days": partB_retention.value,
            "raw_sensitive_evidence_pct": partB_raw_pct.value,
            "local_federated_processing_pct": partB_local_pct.value,
            "privacy_decision": partB_decision.value,
            "safety_threshold_pct": partC_safety_threshold.value,
            "canary_exposure_pct": partC_canary_pct.value,
            "rollback_minutes": partC_rollback_minutes.value,
            "fallback_human_review_pct": partC_human_review.value,
            "safety_decision": partC_decision.value,
            "lineage_pct": partD_lineage.value,
            "audit_retention_days": partD_log_retention.value,
            "decision_context_logging_pct": partD_context.value,
            "access_review_days": partD_access_review.value,
            "accountable_owner": partD_owner.value,
            "governance_decision": partD_decision.value,
        },
        evidence_summary={
            "harmed_party": v1_15_resp.harmed_party,
            "obligation": v1_15_resp.obligation,
            "audit_signal": v1_15_resp.audit_signal,
            "hardware_ref": v1_15_resp.hardware_ref,
            "model_ref": v1_15_resp.model_ref,
            "fpr_gap_pp": round(_conflict.fpr_gap_pp, 3),
            "target_gap_pp": v1_15_resp.target_gap_pp,
            "privacy_epsilon": round(_privacy["epsilon"], 3),
            "privacy_epsilon_limit": _privacy["epsilon_limit"],
            "privacy_evidence_confidence_pct": round(_privacy["evidence_confidence"], 3),
            "privacy_membership_risk": round(_privacy["membership_risk"], 3),
            "privacy_deployable": _privacy["deployable"],
            "privacy_violations": _privacy["violations"],
            "safety_rare_event_score_pct": round(_safety["rare_event_score"], 3),
            "safety_threshold_pct": round(_safety["threshold_pct"], 3),
            "blast_radius_units": round(_safety["affected_units"], 3),
            "blast_radius_cap_units": _safety["blast_cap_units"],
            "safety_release_ok": _safety["release_ok"],
            "safety_violations": _safety["violations"],
            "audit_readiness_pct": round(_governance["readiness"], 3),
            "audit_ready_min_pct": _governance["audit_ready_min"],
            "audit_events_per_year": round(_governance["audit_events_per_year"], 3),
            "release_accountable": _governance["release_accountable"],
            "governance_violations": _governance["violations"],
            "release_ready": _release_ready,
        },
        final_decision=(
            f"{'Proceed to responsible canary' if _release_ready else 'Hold or revise release'} for "
            f"{v1_15_resp.label}: threshold {partA_threshold.value:.2f}, epsilon {_privacy['epsilon']:.1f}, "
            f"safety threshold {_safety['threshold_pct']:.1f}%, blast radius {_safety['affected_units']:,.0f}, "
            f"audit readiness {_governance['readiness']:.1f}%."
        ),
        big_takeaways=(
            "Thresholds encode stakeholder values and error trade-offs.",
            "Privacy and minimization change both risk and the evidence available for release.",
            "Safety, blast radius, and auditability are release gates, not after-launch paperwork.",
        ),
        reflections={
            "report_artifact": v1_15_resp.report_artifact,
            "validation_tests": v1_15_resp.validation_tests,
            "residual_harm_owner": v1_15_resp.residual_harm,
            "carry_forward_capstone_constraint": "V1 capstone release must satisfy the selected threshold, privacy, safety, and audit policy.",
        },
        residual_risk=(
            f"{v1_15_resp.residual_harm} Teaching estimates must be validated with "
            "track-specific audits, representative cohorts, privacy review, safety canaries, and deployment evidence."
        ),
        source_trace={
            "track_id": v1_15_profile.track_id,
            "scenario_id": v1_15_variant.scenario_id,
            "hardware_ref": v1_15_variant.hardware_ref,
            "model_ref": v1_15_variant.model_ref,
            "shared_helper": "mlsysbook_labs.metric_conflict",
            "notebook_local_helpers": (
                "v1_15_privacy_result",
                "v1_15_safety_result",
                "v1_15_governance_result",
            ),
            "source_policy": v1_15_profile.source_policy,
        },
        result_snapshot={
            "responsibility_profile": v1_15_resp,
            "metric_conflict": _conflict,
            "privacy_budget": _privacy,
            "safety_release_gate": _safety,
            "governance_readiness": _governance,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-15 report is generated locally from the selected track, MLSysIM hardware/model refs, "
                "shared `mlsysbook_labs.metric_conflict`, and notebook-local `v1_15_` release-gate calculations."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
