import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ===========================================================================
# ZONE A: SETUP
# ===========================================================================


@app.cell
async def _():
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
        coverage_profile,
        data_policy_decision,
        data_selection_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        selection_frontier,
        selection_utility,
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
        coverage_profile,
        data_policy_decision,
        data_selection_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        selection_frontier,
        selection_utility,
        source_trace,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_09_metadata = get_lab_metadata("vol1/lab_09_data_selection.py")
    return (v1_09_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_09_track_picker = track_selector(default=_default_track)
    v1_09_track_picker
    return (v1_09_track_picker,)


@app.cell
def _(
    data_selection_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_09_track_picker,
):
    v1_09_track_id = v1_09_track_picker.value
    v1_09_profile = get_track_profile(v1_09_track_id)
    v1_09_variant = get_lab_track_variant("v1_09_selection_paradox", v1_09_profile.track_id)
    v1_09_hardware = resolve_mlsysim_ref(v1_09_variant.hardware_ref)
    v1_09_model = resolve_mlsysim_ref(v1_09_variant.model_ref)
    v1_09_selection = data_selection_profile(
        v1_09_profile,
        v1_09_variant,
        v1_09_hardware,
        v1_09_model,
    )
    return (
        v1_09_hardware,
        v1_09_model,
        v1_09_profile,
        v1_09_selection,
        v1_09_track_id,
        v1_09_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    track_arc_context,
    track_context,
    v1_09_metadata,
    v1_09_profile,
    v1_09_selection,
    v1_09_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 09
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Data Selection
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Marginal Value &middot; Coverage &middot; Label Cost &middot; Residual Risk
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 900px; line-height: 1.65;">
                {v1_09_variant.workload_summary} This lab keeps one shared concept
                sequence across all tracks: data quantity is not data value, and a
                defensible selection policy must account for marginal utility,
                coverage, label cost, and residual downstream risk.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Memo &middot; ~50 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_09_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_09_selection.dataset_unit}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">ICR Saturation</span>
                <span class="badge badge-warn">Coverage Guardrail</span>
                <span class="badge badge-info">Label Frontier</span>
                <span class="badge badge-fail">Risk Memo</span>
            </div>
        </div>
        """),
        track_context(v1_09_profile),
        track_arc_context(v1_09_profile, v1_09_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_09_selection):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
            Learning Objectives
        </div>
        <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
            <div style="margin-bottom: 3px;">1. <strong>Measure saturation:</strong>
                explain why marginal data value falls while collection and compute cost continue to grow.</div>
            <div style="margin-bottom: 3px;">2. <strong>Compare coverage:</strong>
                identify when a smaller diverse cohort beats a larger raw cohort.</div>
            <div style="margin-bottom: 3px;">3. <strong>Budget labels:</strong>
                choose a data policy on a label-quality-cost frontier.</div>
            <div style="margin-bottom: 3px;">4. <strong>Defend risk:</strong>
                record residual bias, rejected alternatives, and downstream validation needs.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                Which {v1_09_selection.dataset_unit} should {v1_09_selection.label}
                collect next, and what residual risk is carried forward by not collecting everything?
            </div>
        </div>
    </div>
    """)
    return


# ===========================================================================
# ZONE B: CONTROLS AND NOTEBOOK-LOCAL SUPPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_09_selection):
    v1_09_value_prediction = mo.ui.radio(
        options={
            "More examples are the best next spend": "quantity",
            "Better labels are the best next spend": "quality",
            "Coverage and rare events are the best next spend": "coverage",
            "Cost or storage will dominate the next spend": "cost",
        },
        label=f"{v1_09_selection.label}: what pressure will dominate the next data increment?",
    )
    v1_09_fraction_multiplier = mo.ui.slider(
        start=0.5,
        stop=1.5,
        value=1.0,
        step=0.05,
        label="Dataset fraction multiplier",
    )
    v1_09_part_a_checkpoint = mo.ui.radio(
        options={
            "Expand the same cohort": "expand",
            "Redirect spend toward higher-value coverage": "redirect",
            "Stop collection until validation explains the gap": "stop",
        },
        label="Part A checkpoint: what should the next spend do?",
    )
    return (
        v1_09_fraction_multiplier,
        v1_09_part_a_checkpoint,
        v1_09_value_prediction,
    )


@app.cell(hide_code=True)
def _(mo, v1_09_selection):
    _policy_options = {policy.label: policy.policy_id for policy in v1_09_selection.policy_options}
    v1_09_coverage_prediction = mo.ui.radio(
        options=_policy_options,
        label=f"{v1_09_selection.label}: which policy will best protect the under-covered cohort?",
    )
    v1_09_policy_choice = mo.ui.dropdown(
        options=_policy_options,
        value=v1_09_selection.policy_options[0].label,
        label="Data policy",
    )
    v1_09_coverage_checkpoint = mo.ui.radio(
        options={
            "Keep the largest policy": "largest",
            "Keep the lowest-risk coverage policy": "lowest_risk",
            "Collect targeted examples for the worst subgroup": "target_worst",
        },
        label="Part B checkpoint: what should coverage evidence change?",
    )
    return (
        v1_09_coverage_checkpoint,
        v1_09_coverage_prediction,
        v1_09_policy_choice,
    )


@app.cell(hide_code=True)
def _(mo, v1_09_selection):
    v1_09_label_prediction = mo.ui.radio(
        options={
            "Label spend": "label spend",
            "Review throughput": "review throughput",
            "Quality floor": "quality floor",
            "Coverage floor": "coverage floor",
            "Rare-event floor": "rare-event floor",
            "Processing or compute": "processing/compute",
            "Storage": "storage",
        },
        label=f"{v1_09_selection.label}: which budget will bind the label-quality frontier?",
    )
    v1_09_label_budget_multiplier = mo.ui.slider(
        start=0.5,
        stop=1.2,
        value=1.0,
        step=0.05,
        label="Label and curation budget multiplier",
    )
    v1_09_part_c_checkpoint = mo.ui.radio(
        options={
            "Choose the cheapest feasible cohort": "cheapest",
            "Choose the highest-coverage feasible cohort": "coverage",
            "Choose the highest-quality feasible cohort": "quality",
            "Reject all candidates and request budget": "request_budget",
        },
        label="Part C checkpoint: which frontier decision belongs in the memo?",
    )
    return (
        v1_09_label_budget_multiplier,
        v1_09_label_prediction,
        v1_09_part_c_checkpoint,
    )


@app.cell(hide_code=True)
def _(mo, v1_09_selection):
    _validation_options = {test: test for test in v1_09_selection.validation_tests}
    v1_09_validation_prediction = mo.ui.radio(
        options={
            "Aggregate validation score is enough": "aggregate score",
            "Subgroup residual risk will gate release": "subgroup risk",
            "Rare-event evidence will gate release": "rare-event evidence",
            "Governance, privacy, or SLA review will gate release": "governance/SLA review",
        },
        label=f"{v1_09_selection.label}: what will gate the release after selection?",
    )
    v1_09_risk_tolerance = mo.ui.slider(
        start=0.0,
        stop=30.0,
        value=8.0,
        step=1.0,
        label="Allowed weighted subgroup risk",
    )
    v1_09_validation_focus = mo.ui.dropdown(
        options=_validation_options,
        value=v1_09_selection.validation_tests[0] if v1_09_selection.validation_tests else None,
        label="Validation evidence focus",
    )
    v1_09_part_d_checkpoint = mo.ui.radio(
        options={
            "Ship with risk memo": "ship",
            "Collect targeted data before release": "collect",
            "Reject the selected policy and redo selection": "redo",
        },
        label="Part D checkpoint: what is the release decision?",
    )
    v1_09_reflection = mo.ui.text_area(
        label="Data selection memo notes",
        placeholder="Record selected cohort, binding budget, rejected alternatives, and carry-forward risk.",
        full_width=True,
    )
    return (
        v1_09_part_d_checkpoint,
        v1_09_reflection,
        v1_09_risk_tolerance,
        v1_09_validation_focus,
        v1_09_validation_prediction,
    )


@app.cell
def _(coverage_profile, selection_utility):
    import math

    def v1_09_policy_by_id(profile, policy_id):
        for policy in profile.policy_options:
            if policy.policy_id == policy_id:
                return policy
        return profile.policy_options[0]

    def v1_09_track_amount_system(profile):
        notes = {
            "iphone": {
                "persona": "Mobile product engineer",
                "amount_system": "privacy-safe on-device cohorts, UX edge cases, consent review, app storage, and collection cost",
                "failure_mode": "broad collection expands trust and storage risk while private local contexts remain under-covered",
                "evidence_emphasis": "privacy-safe cohort replay, consent audit, storage headroom, and UX edge cases",
                "report_frame": "privacy-safe cohort memo",
                "label_cost_factor": 1.25,
                "review_cost_per_k": 0.18,
                "saturation_fraction": 0.26,
                "risk_unit": "weighted context gap",
                "risk_mitigation": "collect consented local hard cases from the under-covered context",
                "carry_forward": "future labs should preserve consent, local replay, storage, and context coverage evidence",
            },
            "oura_ring": {
                "persona": "Wearable firmware engineer",
                "amount_system": "biosignal windows, sensor-contact quality, night/activity cohorts, scarce labels, battery, and OTA storage",
                "failure_mode": "continuous windows look cheap but noisy labels and poor contact dilute physiological coverage",
                "evidence_emphasis": "contact-quality audit, sleep/activity replay, battery regression, and OTA payload check",
                "report_frame": "biosignal selection memo",
                "label_cost_factor": 1.55,
                "review_cost_per_k": 0.10,
                "saturation_fraction": 0.18,
                "risk_unit": "weighted physiology gap",
                "risk_mitigation": "collect balanced windows with contact-quality and delayed health labels",
                "carry_forward": "future labs should preserve signal-quality, battery, OTA, and physiology coverage evidence",
            },
            "robotaxi": {
                "persona": "Safety/perception engineer",
                "amount_system": "rare-event clips, scenario coverage, redaction, expert labels, replay storage, and fallback validation",
                "failure_mode": "random miles raise average validation while rare hazards remain thin",
                "evidence_emphasis": "rare-event replay, construction/weather/road-user coverage, redaction audit, and fallback drill",
                "report_frame": "safety evidence memo",
                "label_cost_factor": 2.40,
                "review_cost_per_k": 1.20,
                "saturation_fraction": 0.12,
                "risk_unit": "weighted safety scenario gap",
                "risk_mitigation": "mine targeted safety-triggered clips and label them with expert review",
                "carry_forward": "future labs should preserve rare-event replay, redaction, fallback, and scenario coverage evidence",
            },
            "cloud_fleet": {
                "persona": "Fleet service owner",
                "amount_system": "traffic/query cohorts, freshness, labeling throughput, cost/request, tenants, languages, regions, and SLA impact",
                "failure_mode": "cheap scale can hide tenant or language failures while increasing compute cost",
                "evidence_emphasis": "subgroup reliability, freshness, quality regression, deletion lineage, and cost/request canary",
                "report_frame": "platform reliability memo",
                "label_cost_factor": 0.90,
                "review_cost_per_k": 0.05,
                "saturation_fraction": 0.20,
                "risk_unit": "weighted reliability gap",
                "risk_mitigation": "sample fresh low-resource language, small-tenant, and burst-demand examples",
                "carry_forward": "future labs should preserve freshness, deletion lineage, subgroup reliability, and SLA canary evidence",
            },
        }
        return notes.get(profile.track_id, notes["iphone"])

    def v1_09_prediction_label(options, value):
        for label, stored in options.items():
            if stored == value:
                return label
        return "Not recorded" if value is None else str(value)

    def v1_09_actual_pressure(result):
        if result.dominant_risk in {"coverage", "rare-event coverage"}:
            return "coverage"
        if result.dominant_risk == "quality":
            return "quality"
        if result.dominant_risk in {"cost", "storage"}:
            return "cost"
        return "quantity"

    def v1_09_saturation_point(profile, policy_id, multiplier):
        policy = v1_09_policy_by_id(profile, policy_id)
        utility = selection_utility(
            profile,
            policy_id=policy.policy_id,
            fraction_multiplier=multiplier,
        )
        amount_system = v1_09_track_amount_system(profile)
        saturation_k = max(1.0, profile.dataset_size_k * amount_system["saturation_fraction"])
        information_score = 100.0 * (1.0 - math.exp(-utility.selected_examples_k / saturation_k))
        compute_cost = max(utility.compute_cost, 1e-9)
        return {
            "policy_id": policy.policy_id,
            "policy_label": policy.label,
            "multiplier": float(multiplier),
            "selected_examples_k": utility.selected_examples_k,
            "information_score": information_score,
            "utility_score": utility.utility_score,
            "total_cost": utility.total_cost,
            "compute_cost": utility.compute_cost,
            "icr_proxy": information_score / compute_cost,
            "feasible": utility.feasible,
            "violations": utility.violations,
        }

    def v1_09_saturation_rows(profile, policy_id):
        rows = []
        previous = None
        for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5):
            point = dict(v1_09_saturation_point(profile, policy_id, multiplier))
            if previous is None:
                point["marginal_signal"] = None
                point["marginal_cost"] = None
                point["marginal_signal_per_cost"] = None
            else:
                marginal_signal = point["information_score"] - previous["information_score"]
                marginal_cost = max(point["compute_cost"] - previous["compute_cost"], 1e-9)
                point["marginal_signal"] = marginal_signal
                point["marginal_cost"] = marginal_cost
                point["marginal_signal_per_cost"] = marginal_signal / marginal_cost
            rows.append(point)
            previous = point
        return tuple(rows)

    def v1_09_saturation_summary(rows, current_point):
        deltas = [row for row in rows if row["marginal_signal"] is not None]
        first_gain = deltas[0]["marginal_signal"] if deltas else 0.0
        last_gain = deltas[-1]["marginal_signal"] if deltas else 0.0
        saturated = bool(first_gain and last_gain < first_gain * 0.55)
        return {
            "first_gain": first_gain,
            "last_gain": last_gain,
            "saturated": saturated,
            "current_icr_proxy": current_point["icr_proxy"],
            "current_information_score": current_point["information_score"],
        }

    def v1_09_coverage_policy_rows(profile, fraction_multiplier):
        rows = []
        for policy in profile.policy_options:
            utility = selection_utility(
                profile,
                policy_id=policy.policy_id,
                fraction_multiplier=fraction_multiplier,
            )
            coverage = coverage_profile(profile, policy_id=policy.policy_id)
            rows.append({
                "policy_id": policy.policy_id,
                "policy_label": policy.label,
                "selected_examples_k": utility.selected_examples_k,
                "quality_score_pct": utility.quality_score_pct,
                "coverage_score_pct": utility.coverage_score_pct,
                "rare_event_score_pct": utility.rare_event_score_pct,
                "total_cost": utility.total_cost,
                "feasible": utility.feasible,
                "worst_subgroup": coverage.worst_subgroup,
                "worst_risk_score": coverage.worst_risk_score,
            })
        return tuple(rows)

    def v1_09_coverage_best_policy(rows):
        return min(rows, key=lambda row: (row["worst_risk_score"], -row["coverage_score_pct"]))

    def v1_09_coverage_largest_policy(rows):
        return max(rows, key=lambda row: row["selected_examples_k"])

    def v1_09_label_frontier_rows(profile, budget_multiplier, fraction_multiplier):
        amount_system = v1_09_track_amount_system(profile)
        budget = profile.cost_budget * float(budget_multiplier)
        rows = []
        for policy in profile.policy_options:
            utility = selection_utility(
                profile,
                policy_id=policy.policy_id,
                fraction_multiplier=fraction_multiplier,
            )
            quality_premium = (
                1.0
                + max(0.0, policy.label_quality_pct - 80.0) / 100.0
                + max(0.0, policy.rare_event_multiplier - 1.0) * 0.18
            )
            label_cost = (
                utility.selected_examples_k
                * profile.cost_per_k
                * amount_system["label_cost_factor"]
                * quality_premium
            )
            review_cost = utility.selected_examples_k * amount_system["review_cost_per_k"]
            process_cost = utility.selected_examples_k * profile.compute_cost_per_k
            total_cost = label_cost + review_cost + process_cost
            ratios = {
                "label spend": label_cost / max(budget, 1e-9),
                "review throughput": review_cost / max(budget, 1e-9),
                "processing/compute": process_cost / max(budget, 1e-9),
                "quality floor": profile.quality_floor_pct / max(utility.quality_score_pct, 1e-9),
                "coverage floor": profile.coverage_floor_pct / max(utility.coverage_score_pct, 1e-9),
                "rare-event floor": profile.rare_event_floor_pct / max(utility.rare_event_score_pct, 1e-9),
                "storage": utility.storage_mb / max(profile.storage_budget_mb, 1e-9),
            }
            binding = max(ratios, key=ratios.get)
            violations = []
            if total_cost > budget:
                violations.append(f"label frontier cost {total_cost:.1f} > budget {budget:.1f}")
            if utility.quality_score_pct < profile.quality_floor_pct:
                violations.append(f"quality {utility.quality_score_pct:.1f}% < floor {profile.quality_floor_pct:.1f}%")
            if utility.coverage_score_pct < profile.coverage_floor_pct:
                violations.append(f"coverage {utility.coverage_score_pct:.1f}% < floor {profile.coverage_floor_pct:.1f}%")
            if utility.rare_event_score_pct < profile.rare_event_floor_pct:
                violations.append(f"rare-event {utility.rare_event_score_pct:.1f}% < floor {profile.rare_event_floor_pct:.1f}%")
            if utility.storage_mb > profile.storage_budget_mb:
                violations.append(f"storage {utility.storage_mb:.1f} MB > budget {profile.storage_budget_mb:.1f} MB")
            rows.append({
                "policy_id": policy.policy_id,
                "policy_label": policy.label,
                "selected_examples_k": utility.selected_examples_k,
                "quality_score_pct": utility.quality_score_pct,
                "coverage_score_pct": utility.coverage_score_pct,
                "rare_event_score_pct": utility.rare_event_score_pct,
                "label_cost": label_cost,
                "review_cost": review_cost,
                "process_cost": process_cost,
                "total_cost": total_cost,
                "budget": budget,
                "binding_constraint": binding,
                "feasible": not violations,
                "violations": tuple(violations),
            })
        return tuple(rows)

    def v1_09_label_selected(rows, policy_id):
        for row in rows:
            if row["policy_id"] == policy_id:
                return row
        return rows[0]

    def v1_09_risk_register(profile, coverage, tolerance):
        amount_system = v1_09_track_amount_system(profile)
        rows = []
        for cell in coverage.cells:
            status = "ok" if cell.risk_score <= tolerance else "action"
            rows.append({
                "subgroup_id": cell.subgroup_id,
                "label": cell.label,
                "coverage_pct": cell.coverage_pct,
                "risk_score": cell.risk_score,
                "status": status,
                "mitigation": amount_system["risk_mitigation"] if status == "action" else "monitor with stratified validation",
            })
        return tuple(rows)

    def v1_09_release_gate(profile, selected_utility, decision, risk_rows, tolerance, validation_focus):
        worst = max(risk_rows, key=lambda row: row["risk_score"]) if risk_rows else None
        aggregate_ok = selected_utility.utility_score >= 75.0 and selected_utility.feasible
        subgroup_ok = all(row["risk_score"] <= tolerance for row in risk_rows)
        rare_ok = selected_utility.rare_event_score_pct >= profile.rare_event_floor_pct
        evidence_ok = bool(validation_focus)
        blockers = []
        if not aggregate_ok:
            blockers.append("aggregate score")
        if not subgroup_ok:
            blockers.append("subgroup risk")
        if not rare_ok:
            blockers.append("rare-event evidence")
        if not evidence_ok:
            blockers.append("governance/SLA review")
        release_ok = not blockers
        if release_ok:
            decision_text = "Ship with risk memo and monitoring."
            binding_gate = "none"
        else:
            decision_text = f"Collect or revise before release; blocked by {', '.join(blockers)}."
            binding_gate = blockers[0]
        return {
            "release_ok": release_ok,
            "decision": decision_text,
            "binding_gate": binding_gate,
            "aggregate_ok": aggregate_ok,
            "subgroup_ok": subgroup_ok,
            "rare_ok": rare_ok,
            "evidence_ok": evidence_ok,
            "worst_subgroup": worst["label"] if worst else decision.worst_subgroup,
            "worst_risk_score": worst["risk_score"] if worst else 0.0,
        }

    def v1_09_bool_text(value):
        return "yes" if value else "no"

    def v1_09_table(headers, rows):
        head = "".join(f"<th>{header}</th>" for header in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        return f"""
        <table class="mlsysbook-table">
          <thead><tr>{head}</tr></thead>
          <tbody>{body}</tbody>
        </table>
        """

    def v1_09_part_banner(letter, title, concept, color):
        return f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part {letter}: {title}</h2></div>
          <div class="mlsysbook-callout" style="border-left-color:{color};">
            <strong>Concept module:</strong> {concept}
          </div>
        </div>
        """

    def v1_09_metric_cards(cards):
        rendered = []
        for label, value, note, color in cards:
            rendered.append(f"""
            <div class="mlsysbook-field" style="border-top:3px solid {color};">
              <strong>{label}</strong>
              <span style="font-size:1.05rem; font-weight:700;">{value}</span>
              <span style="color:#64748b; font-size:0.82rem;">{note}</span>
            </div>
            """)
        return f"""<div class="mlsysbook-grid">{''.join(rendered)}</div>"""

    def v1_09_reveal_card(title, predicted, actual, consequence, ok):
        kind = "success" if ok else "warn"
        color = "#166534" if ok else "#b45309"
        return f"""
        <div class="mlsysbook-panel">
          <h2>{title}</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Prediction</strong>{predicted}</div>
            <div class="mlsysbook-field"><strong>Observed</strong>{actual}</div>
            <div class="mlsysbook-field"><strong>Status</strong>{kind}</div>
          </div>
          <div class="mlsysbook-callout" style="border-left-color:{color};">
            <strong>Consequence:</strong> {consequence}
          </div>
        </div>
        """

    def v1_09_failure_card(title, failures, recovery):
        items = "".join(f"<li>{item}</li>" for item in failures)
        return f"""
        <div class="mlsysbook-panel" style="border-left:4px solid #dc2626;">
          <h2>{title}</h2>
          <ul class="mlsysbook-list">{items}</ul>
          <div class="mlsysbook-callout"><strong>Recovery:</strong> {recovery}</div>
        </div>
        """

    return (
        v1_09_actual_pressure,
        v1_09_bool_text,
        v1_09_coverage_best_policy,
        v1_09_coverage_largest_policy,
        v1_09_coverage_policy_rows,
        v1_09_failure_card,
        v1_09_label_frontier_rows,
        v1_09_label_selected,
        v1_09_metric_cards,
        v1_09_part_banner,
        v1_09_policy_by_id,
        v1_09_prediction_label,
        v1_09_release_gate,
        v1_09_reveal_card,
        v1_09_risk_register,
        v1_09_saturation_point,
        v1_09_saturation_rows,
        v1_09_saturation_summary,
        v1_09_table,
        v1_09_track_amount_system,
    )


@app.cell
def _(
    coverage_profile,
    data_policy_decision,
    selection_frontier,
    selection_utility,
    v1_09_coverage_best_policy,
    v1_09_coverage_largest_policy,
    v1_09_coverage_policy_rows,
    v1_09_fraction_multiplier,
    v1_09_label_budget_multiplier,
    v1_09_label_frontier_rows,
    v1_09_label_selected,
    v1_09_policy_choice,
    v1_09_release_gate,
    v1_09_risk_register,
    v1_09_risk_tolerance,
    v1_09_saturation_point,
    v1_09_saturation_rows,
    v1_09_saturation_summary,
    v1_09_selection,
    v1_09_track_amount_system,
    v1_09_validation_focus,
):
    v1_09_amount_system = v1_09_track_amount_system(v1_09_selection)
    v1_09_frontier = selection_frontier(
        v1_09_selection,
        fraction_multiplier=v1_09_fraction_multiplier.value,
    )
    v1_09_selected_utility = selection_utility(
        v1_09_selection,
        policy_id=v1_09_policy_choice.value,
        fraction_multiplier=v1_09_fraction_multiplier.value,
    )
    v1_09_coverage = coverage_profile(
        v1_09_selection,
        policy_id=v1_09_policy_choice.value,
    )
    v1_09_decision = data_policy_decision(
        v1_09_selection,
        policy_id=v1_09_policy_choice.value,
        fraction_multiplier=v1_09_fraction_multiplier.value,
    )
    v1_09_saturation_points = v1_09_saturation_rows(
        v1_09_selection,
        v1_09_policy_choice.value,
    )
    v1_09_current_saturation = v1_09_saturation_point(
        v1_09_selection,
        v1_09_policy_choice.value,
        v1_09_fraction_multiplier.value,
    )
    v1_09_saturation_evidence = v1_09_saturation_summary(
        v1_09_saturation_points,
        v1_09_current_saturation,
    )
    v1_09_coverage_rows = v1_09_coverage_policy_rows(
        v1_09_selection,
        v1_09_fraction_multiplier.value,
    )
    v1_09_coverage_best = v1_09_coverage_best_policy(v1_09_coverage_rows)
    v1_09_coverage_largest = v1_09_coverage_largest_policy(v1_09_coverage_rows)
    v1_09_label_rows = v1_09_label_frontier_rows(
        v1_09_selection,
        v1_09_label_budget_multiplier.value,
        v1_09_fraction_multiplier.value,
    )
    v1_09_label_selected_row = v1_09_label_selected(
        v1_09_label_rows,
        v1_09_policy_choice.value,
    )
    v1_09_risk_rows = v1_09_risk_register(
        v1_09_selection,
        v1_09_coverage,
        v1_09_risk_tolerance.value,
    )
    v1_09_release = v1_09_release_gate(
        v1_09_selection,
        v1_09_selected_utility,
        v1_09_decision,
        v1_09_risk_rows,
        v1_09_risk_tolerance.value,
        v1_09_validation_focus.value,
    )
    return (
        v1_09_amount_system,
        v1_09_coverage,
        v1_09_coverage_best,
        v1_09_coverage_largest,
        v1_09_coverage_rows,
        v1_09_current_saturation,
        v1_09_decision,
        v1_09_frontier,
        v1_09_label_rows,
        v1_09_label_selected_row,
        v1_09_release,
        v1_09_risk_rows,
        v1_09_saturation_evidence,
        v1_09_saturation_points,
        v1_09_selected_utility,
    )


# ===========================================================================
# ZONE C: CONCEPT MODULES
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    source_trace,
    v1_09_actual_pressure,
    v1_09_amount_system,
    v1_09_bool_text,
    v1_09_coverage,
    v1_09_coverage_best,
    v1_09_coverage_checkpoint,
    v1_09_coverage_largest,
    v1_09_coverage_prediction,
    v1_09_coverage_rows,
    v1_09_current_saturation,
    v1_09_decision,
    v1_09_failure_card,
    v1_09_fraction_multiplier,
    v1_09_label_budget_multiplier,
    v1_09_label_prediction,
    v1_09_label_rows,
    v1_09_label_selected_row,
    v1_09_metric_cards,
    v1_09_part_a_checkpoint,
    v1_09_part_banner,
    v1_09_part_c_checkpoint,
    v1_09_part_d_checkpoint,
    v1_09_policy_choice,
    v1_09_prediction_label,
    v1_09_reflection,
    v1_09_release,
    v1_09_reveal_card,
    v1_09_risk_rows,
    v1_09_risk_tolerance,
    v1_09_saturation_evidence,
    v1_09_saturation_points,
    v1_09_selected_utility,
    v1_09_selection,
    v1_09_table,
    v1_09_validation_focus,
    v1_09_validation_prediction,
    v1_09_value_prediction,
    v1_09_variant,
):
    def build_part_a():
        items = [
            mo.Html(v1_09_part_banner(
                "A",
                "Marginal Data Value Saturates",
                "More examples are not always the best next spend.",
                COLORS["BlueLine"],
            )),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p>{v1_09_amount_system["persona"]} has one more collection window for
              <strong>{v1_09_selection.label}</strong>. The tempting move is to collect
              more of the same {v1_09_selection.dataset_unit}; the chapter asks whether
              the next increment still carries learning signal.</p>
              <div class="mlsysbook-callout"><strong>Track amount system:</strong>
                {v1_09_amount_system["amount_system"]}.</div>
            </div>
            """),
            v1_09_value_prediction,
        ]
        if v1_09_value_prediction.value is None:
            items.append(mo.callout(
                mo.md("Commit to a Part A prediction to reveal the marginal-value instrument."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.extend([v1_09_fraction_multiplier])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[row["selected_examples_k"] for row in v1_09_saturation_points],
            y=[row["information_score"] for row in v1_09_saturation_points],
            mode="lines+markers",
            name="Information proxy",
            line=dict(color=COLORS["BlueLine"], width=3),
        ))
        fig.add_trace(go.Scatter(
            x=[row["selected_examples_k"] for row in v1_09_saturation_points],
            y=[row["icr_proxy"] for row in v1_09_saturation_points],
            mode="lines+markers",
            name="ICR proxy",
            yaxis="y2",
            line=dict(color=COLORS["OrangeLine"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[v1_09_current_saturation["selected_examples_k"]],
            y=[v1_09_current_saturation["information_score"]],
            mode="markers",
            name="current setting",
            marker=dict(size=16, color=COLORS["RedLine"], line=dict(color="white", width=2)),
        ))
        fig.update_layout(
            height=360,
            xaxis=dict(title=f"Selected {v1_09_selection.dataset_unit} (k)", gridcolor="#f1f5f9"),
            yaxis=dict(title="Information proxy", gridcolor="#f1f5f9", range=[0, 105]),
            yaxis2=dict(title="ICR proxy", overlaying="y", side="right", showgrid=False),
            margin=dict(l=60, r=70, t=35, b=60),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        rows = []
        for row in v1_09_saturation_points:
            marginal = "start" if row["marginal_signal"] is None else f"{row['marginal_signal']:.2f}"
            icr = f"{row['icr_proxy']:.3f}"
            rows.append((
                f"{row['multiplier']:.2f}x",
                f"{row['selected_examples_k']:.1f}k",
                f"{row['information_score']:.1f}",
                marginal,
                icr,
                f"{row['total_cost']:.1f}",
                v1_09_bool_text(row["feasible"]),
            ))
        items.append(mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Evidence Table</h2>
          {v1_09_table(("Multiplier", "Selected", "Info proxy", "Marginal gain", "ICR proxy", "Cost", "Feasible"), rows)}
        </div>
        """))

        actual = v1_09_actual_pressure(v1_09_selected_utility)
        predicted = v1_09_prediction_label(
            {
                "More examples are the best next spend": "quantity",
                "Better labels are the best next spend": "quality",
                "Coverage and rare events are the best next spend": "coverage",
                "Cost or storage will dominate the next spend": "cost",
            },
            v1_09_value_prediction.value,
        )
        consequence = (
            f"The current increment has ICR proxy {v1_09_current_saturation['icr_proxy']:.3f}; "
            f"the last planned increment gains {v1_09_saturation_evidence['last_gain']:.2f} information points "
            f"versus {v1_09_saturation_evidence['first_gain']:.2f} early in the curve."
        )
        items.append(mo.Html(v1_09_reveal_card(
            "Prediction vs observed selection pressure",
            predicted,
            actual,
            consequence,
            ok=v1_09_value_prediction.value == actual,
        )))
        if v1_09_current_saturation["violations"]:
            items.append(mo.Html(v1_09_failure_card(
                "Current multiplier crosses a budget",
                v1_09_current_saturation["violations"],
                "Lower the multiplier or redirect collection toward a higher-value cohort.",
            )))
        elif v1_09_saturation_evidence["saturated"]:
            items.append(mo.callout(
                mo.md(
                    "**Boundary found.** Marginal signal has dropped sharply. The next spend "
                    "should be justified by coverage, quality, or risk reduction rather than volume alone."
                ),
                kind="warn",
            ))
        else:
            items.append(mo.callout(
                mo.md("The selected point remains inside the modeled marginal-value envelope, pending coverage and label-cost checks."),
                kind="info",
            ))
        items.extend([
            mo.accordion({
                "Math Peek / Source Model - ICR saturation": mo.md("""
The chapter defines information-compute ratio as:

$$
\\text{ICR} = \\frac{\\Delta I}{\\Delta \\text{FLOPs}}
$$

For redundant data, marginal information decays while compute grows roughly
linearly:

$$
\\text{ICR}(D) \\approx \\frac{1}{O_{sample} \\cdot D}
$$

The lab instrument uses a saturating information proxy and the shared
`selection_utility()` cost fields. The exact values are scenario instrumentation;
the shape is the chapter concept.
""")
            }),
            source_trace({
                "chapter_anchor": "Information-Compute Ratio and ICR Frontier",
                "shared_helper": "selection_utility()",
                "notebook_model": "v1_09_saturation_point()",
                "hardware_ref": v1_09_selection.hardware_ref,
                "model_ref": v1_09_selection.model_ref,
            }, summary="Part A uses chapter ICR decay plus the V1-09 data-selection profile."),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <p>Choose the memo action that follows from the marginal-value evidence.</p>
            </div>
            """),
            v1_09_part_a_checkpoint,
        ])
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(v1_09_part_banner(
                "B",
                "Coverage And Diversity Can Beat Size",
                "The largest cohort can still be the least defensible cohort.",
                COLORS["GreenLine"],
            )),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p>The review asks whether the selected data covers the deployment
              cases that matter for <strong>{v1_09_selection.label}</strong>. A
              raw example count does not answer that question.</p>
              <div class="mlsysbook-callout"><strong>Track failure mode:</strong>
                {v1_09_amount_system["failure_mode"]}.</div>
            </div>
            """),
            v1_09_coverage_prediction,
        ]
        if v1_09_coverage_prediction.value is None:
            items.append(mo.callout(
                mo.md("Commit to a Part B coverage prediction to reveal the cohort comparison."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.extend([v1_09_policy_choice])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[cell.label for cell in v1_09_coverage.cells],
            y=[cell.coverage_pct for cell in v1_09_coverage.cells],
            marker_color=[
                COLORS["GreenLine"] if cell.status == "ok" else COLORS["RedLine"]
                for cell in v1_09_coverage.cells
            ],
            text=[f"{cell.coverage_pct:.1f}%" for cell in v1_09_coverage.cells],
            textposition="outside",
        ))
        fig.add_hline(
            y=v1_09_selection.coverage_floor_pct,
            line_dash="dash",
            line_color=COLORS["RedLine"],
            annotation_text="coverage floor",
            annotation_font_color=COLORS["RedLine"],
        )
        fig.update_layout(
            height=340,
            xaxis=dict(title="Subgroup", gridcolor="#f1f5f9"),
            yaxis=dict(title="Coverage (%)", gridcolor="#f1f5f9", range=[0, 110]),
            margin=dict(l=60, r=20, t=35, b=80),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        policy_rows = []
        for row in v1_09_coverage_rows:
            policy_rows.append((
                row["policy_label"],
                f"{row['selected_examples_k']:.1f}k",
                f"{row['coverage_score_pct']:.1f}%",
                f"{row['rare_event_score_pct']:.1f}%",
                row["worst_subgroup"],
                f"{row['worst_risk_score']:.1f}",
                v1_09_bool_text(row["feasible"]),
            ))
        subgroup_rows = [
            (
                cell.label,
                f"{cell.coverage_pct:.1f}%",
                f"{cell.risk_score:.1f}",
                cell.status,
            )
            for cell in v1_09_coverage.cells
        ]
        items.append(mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Coverage Evidence</h2>
          {v1_09_table(("Policy", "Selected", "Coverage", "Rare events", "Worst subgroup", "Weighted gap", "Feasible"), policy_rows)}
          <h2>Selected Policy Subgroups</h2>
          {v1_09_table(("Subgroup", "Coverage", "Risk score", "Status"), subgroup_rows)}
        </div>
        """))
        predicted = v1_09_prediction_label(
            {policy.label: policy.policy_id for policy in v1_09_selection.policy_options},
            v1_09_coverage_prediction.value,
        )
        consequence = (
            f"Largest policy: {v1_09_coverage_largest['policy_label']} "
            f"({v1_09_coverage_largest['selected_examples_k']:.1f}k). "
            f"Lowest-risk policy: {v1_09_coverage_best['policy_label']} "
            f"(worst gap {v1_09_coverage_best['worst_risk_score']:.1f})."
        )
        items.append(mo.Html(v1_09_reveal_card(
            "Prediction vs lowest-risk coverage policy",
            predicted,
            v1_09_coverage_best["policy_label"],
            consequence,
            ok=v1_09_coverage_prediction.value == v1_09_coverage_best["policy_id"],
        )))
        if v1_09_coverage.worst_risk_score > 0:
            items.append(mo.Html(v1_09_failure_card(
                "Coverage guardrail is still open",
                (
                    f"Worst subgroup: {v1_09_coverage.worst_subgroup}",
                    f"Weighted coverage gap: {v1_09_coverage.worst_risk_score:.1f}",
                ),
                v1_09_amount_system["risk_mitigation"],
            )))
        else:
            items.append(mo.callout(
                mo.md("All selected subgroups clear the coverage floor under this policy."),
                kind="success",
            ))
        items.extend([
            mo.accordion({
                "Math Peek / Source Model - coverage guardrail": mo.md("""
Coreset selection is a coverage problem before it is a sample-count problem:
retain the smallest cohort that preserves deployment-relevant slices.

This lab scores each subgroup as:

$$
\\text{risk} = \\max(0, \\text{coverage floor} - \\text{coverage}) \\cdot \\text{risk weight}
$$

The decision is not "largest dataset wins"; it is "which cohort leaves the
least important deployment gap under the budget?"
""")
            }),
            source_trace({
                "chapter_anchor": "Coreset Selection Algorithms and rare-class pruning pitfall",
                "shared_helper": "coverage_profile()",
                "coverage_floor_pct": v1_09_selection.coverage_floor_pct,
                "track_id": v1_09_selection.track_id,
            }, summary="Part B coverage rows come from the V1-09 profile subgroups and policy adjustments."),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <p>Choose the coverage decision that belongs in the memo.</p>
            </div>
            """),
            v1_09_coverage_checkpoint,
        ])
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(v1_09_part_banner(
                "C",
                "Label Cost And Quality Create A Frontier",
                "Label quality is useful only inside a budgeted amount system.",
                COLORS["OrangeLine"],
            )),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p>The team can label or curate only part of the pool. The selected
              policy must fit the track's labeling, review, processing, storage,
              and quality floors.</p>
              <div class="mlsysbook-callout"><strong>Evidence emphasis:</strong>
                {v1_09_amount_system["evidence_emphasis"]}.</div>
            </div>
            """),
            v1_09_label_prediction,
        ]
        if v1_09_label_prediction.value is None:
            items.append(mo.callout(
                mo.md("Commit to a Part C budget prediction to reveal the label frontier."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.extend([v1_09_label_budget_multiplier, v1_09_policy_choice])
        fig = go.Figure()
        labels = [row["policy_label"] for row in v1_09_label_rows]
        fig.add_trace(go.Bar(
            x=labels,
            y=[row["label_cost"] for row in v1_09_label_rows],
            name="Label cost",
            marker_color=COLORS["BlueLine"],
        ))
        fig.add_trace(go.Bar(
            x=labels,
            y=[row["review_cost"] for row in v1_09_label_rows],
            name="Review cost",
            marker_color=COLORS["OrangeLine"],
        ))
        fig.add_trace(go.Bar(
            x=labels,
            y=[row["process_cost"] for row in v1_09_label_rows],
            name="Process cost",
            marker_color=COLORS["GreenLine"],
        ))
        fig.add_hline(
            y=v1_09_label_selected_row["budget"],
            line_dash="dash",
            line_color=COLORS["RedLine"],
            annotation_text="budget",
            annotation_font_color=COLORS["RedLine"],
        )
        fig.update_layout(
            height=360,
            barmode="stack",
            xaxis=dict(title="Policy", gridcolor="#f1f5f9"),
            yaxis=dict(title="Budget units", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=35, b=80),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        rows = []
        for row in v1_09_label_rows:
            rows.append((
                row["policy_label"],
                f"{row['selected_examples_k']:.1f}k",
                f"{row['quality_score_pct']:.1f}%",
                f"{row['rare_event_score_pct']:.1f}%",
                f"{row['total_cost']:.1f}",
                f"{row['budget']:.1f}",
                row["binding_constraint"],
                v1_09_bool_text(row["feasible"]),
            ))
        items.append(mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Budgeted Frontier</h2>
          {v1_09_table(("Policy", "Selected", "Quality", "Rare events", "Total cost", "Budget", "Binding", "Feasible"), rows)}
        </div>
        """))
        items.append(mo.Html(v1_09_metric_cards((
            (
                "Selected policy",
                v1_09_label_selected_row["policy_label"],
                f"{v1_09_label_selected_row['selected_examples_k']:.1f}k {v1_09_selection.dataset_unit}",
                COLORS["BlueLine"],
            ),
            (
                "Binding budget",
                v1_09_label_selected_row["binding_constraint"],
                f"budget multiplier {v1_09_label_budget_multiplier.value:.2f}x",
                COLORS["OrangeLine"],
            ),
            (
                "Frontier status",
                "feasible" if v1_09_label_selected_row["feasible"] else "blocked",
                f"total {v1_09_label_selected_row['total_cost']:.1f} / budget {v1_09_label_selected_row['budget']:.1f}",
                COLORS["GreenLine"] if v1_09_label_selected_row["feasible"] else COLORS["RedLine"],
            ),
        ))))
        predicted = v1_09_prediction_label(
            {
                "Label spend": "label spend",
                "Review throughput": "review throughput",
                "Quality floor": "quality floor",
                "Coverage floor": "coverage floor",
                "Rare-event floor": "rare-event floor",
                "Processing or compute": "processing/compute",
                "Storage": "storage",
            },
            v1_09_label_prediction.value,
        )
        items.append(mo.Html(v1_09_reveal_card(
            "Prediction vs binding frontier amount",
            predicted,
            v1_09_label_selected_row["binding_constraint"],
            (
                "The binding amount is the largest ratio among label spend, review throughput, "
                "processing, quality, coverage, rare-event, and storage checks."
            ),
            ok=v1_09_label_prediction.value == v1_09_label_selected_row["binding_constraint"],
        )))
        if not v1_09_label_selected_row["feasible"]:
            items.append(mo.Html(v1_09_failure_card(
                "Selected policy is outside the label frontier",
                v1_09_label_selected_row["violations"],
                "Increase budget, lower selected volume, or choose a policy that buys quality and coverage more efficiently.",
            )))
        else:
            items.append(mo.callout(
                mo.md("The selected policy is inside the current budgeted label frontier."),
                kind="success",
            ))
        items.extend([
            mo.accordion({
                "Math Peek / Source Model - total data cost": mo.md("""
The chapter's total data cost model is:

$$
C_{total} = C_{acquire} + C_{label} + C_{store} + C_{process}
$$

The selection engineering gate is:

$$
T_{selection} + T_{train}(D_{subset}) < T_{train}(D_{total})
$$

Part C turns that idea into a track-local label frontier: labels, review,
processing, storage, quality, coverage, and rare-event floors must all pass.
""")
            }),
            source_trace({
                "chapter_anchor": "Cost Modeling and Selection Inequality",
                "shared_helper": "selection_utility()",
                "notebook_model": "v1_09_label_frontier_rows()",
                "track_amount_system": v1_09_amount_system["amount_system"],
            }, summary="Part C cost bars derive from the V1-09 profile costs plus track-specific label/review factors."),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <p>Choose the frontier decision that belongs in the final memo.</p>
            </div>
            """),
            v1_09_part_c_checkpoint,
        ])
        return mo.vstack(items)

    def build_part_d():
        items = [
            mo.Html(v1_09_part_banner(
                "D",
                "Residual Risk Must Be Defended",
                "A validation score is not a release argument by itself.",
                COLORS["RedLine"],
            )),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p>The selected policy has a utility score, but the release reviewer
              asks what bias or downstream harm remains. The answer must be track
              specific and evidence backed.</p>
              <div class="mlsysbook-callout"><strong>Report framing:</strong>
                {v1_09_amount_system["report_frame"]}.</div>
            </div>
            """),
            v1_09_validation_prediction,
        ]
        if v1_09_validation_prediction.value is None:
            items.append(mo.callout(
                mo.md("Commit to a Part D gate prediction to reveal the risk register."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.extend([v1_09_risk_tolerance, v1_09_validation_focus])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[row["label"] for row in v1_09_risk_rows],
            y=[row["risk_score"] for row in v1_09_risk_rows],
            marker_color=[
                COLORS["GreenLine"] if row["status"] == "ok" else COLORS["RedLine"]
                for row in v1_09_risk_rows
            ],
            text=[f"{row['risk_score']:.1f}" for row in v1_09_risk_rows],
            textposition="outside",
        ))
        fig.add_hline(
            y=v1_09_risk_tolerance.value,
            line_dash="dash",
            line_color=COLORS["RedLine"],
            annotation_text="risk tolerance",
            annotation_font_color=COLORS["RedLine"],
        )
        fig.update_layout(
            height=340,
            xaxis=dict(title="Subgroup", gridcolor="#f1f5f9"),
            yaxis=dict(title=v1_09_amount_system["risk_unit"], gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=35, b=80),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        rows = []
        for row in v1_09_risk_rows:
            rows.append((
                row["label"],
                f"{row['coverage_pct']:.1f}%",
                f"{row['risk_score']:.1f}",
                row["status"],
                row["mitigation"],
            ))
        items.append(mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Risk Register</h2>
          {v1_09_table(("Subgroup", "Coverage", "Risk", "Status", "Mitigation"), rows)}
        </div>
        """))
        items.append(mo.Html(v1_09_metric_cards((
            (
                "Aggregate utility",
                f"{v1_09_selected_utility.utility_score:.1f}",
                "not sufficient without slice evidence",
                COLORS["BlueLine"],
            ),
            (
                "Worst subgroup",
                v1_09_release["worst_subgroup"],
                f"risk {v1_09_release['worst_risk_score']:.1f}",
                COLORS["RedLine"] if v1_09_release["worst_risk_score"] > v1_09_risk_tolerance.value else COLORS["GreenLine"],
            ),
            (
                "Release gate",
                "pass" if v1_09_release["release_ok"] else "blocked",
                v1_09_release["decision"],
                COLORS["GreenLine"] if v1_09_release["release_ok"] else COLORS["RedLine"],
            ),
        ))))
        predicted = v1_09_prediction_label(
            {
                "Aggregate validation score is enough": "aggregate score",
                "Subgroup residual risk will gate release": "subgroup risk",
                "Rare-event evidence will gate release": "rare-event evidence",
                "Governance, privacy, or SLA review will gate release": "governance/SLA review",
            },
            v1_09_validation_prediction.value,
        )
        actual_gate = v1_09_release["binding_gate"] if v1_09_release["binding_gate"] != "none" else "no blocking gate"
        items.append(mo.Html(v1_09_reveal_card(
            "Prediction vs release gate",
            predicted,
            actual_gate,
            "Release requires aggregate feasibility, subgroup risk, rare-event evidence, and the selected validation focus.",
            ok=v1_09_validation_prediction.value == v1_09_release["binding_gate"],
        )))
        if not v1_09_release["release_ok"]:
            items.append(mo.Html(v1_09_failure_card(
                "Release cannot be defended yet",
                (v1_09_release["decision"], f"Validation focus: {v1_09_validation_focus.value}"),
                "Collect targeted data for the risk register or choose a lower-risk policy before release.",
            )))
        else:
            items.append(mo.callout(
                mo.md("The selected policy can be defended with a residual-risk memo and continued monitoring."),
                kind="success",
            ))
        items.extend([
            mo.accordion({
                "Math Peek / Source Model - release risk gate": mo.md("""
The lab's release gate is deliberately conjunctive:

$$
release\\_ok = aggregate\\_ok \\land subgroup\\_risk\\_ok \\land rare\\_event\\_ok \\land evidence\\_ok
$$

This follows the chapter warning that PPD, DCR, and aggregate validation can
look strong while deployment edge cases fail. Stratified risk evidence must
travel with the selected data policy.
""")
            }),
            source_trace({
                "chapter_anchor": "Measurement Framework and optimizing selection metrics instead of deployment metrics",
                "shared_helper": "coverage_profile() and data_policy_decision()",
                "notebook_model": "v1_09_release_gate()",
                "validation_focus": v1_09_validation_focus.value,
            }, summary="Part D risk rows reuse coverage evidence and add a track-specific release gate."),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <p>Choose the final release decision, then record the memo notes used in synthesis.</p>
            </div>
            """),
            v1_09_part_d_checkpoint,
            v1_09_reflection,
        ])
        return mo.vstack(items)

    v1_09_tabs = mo.ui.tabs({
        "Part A - Marginal Value": build_part_a(),
        "Part B - Coverage": build_part_b(),
        "Part C - Label Frontier": build_part_c(),
        "Part D - Risk": build_part_d(),
        "Synthesis": mo.md("Use the synthesis memo below after completing Parts A-D."),
    })
    v1_09_tabs
    return (v1_09_tabs,)


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    v1_09_amount_system,
    v1_09_coverage_best,
    v1_09_coverage_checkpoint,
    v1_09_coverage_prediction,
    v1_09_decision,
    v1_09_fraction_multiplier,
    v1_09_label_budget_multiplier,
    v1_09_label_prediction,
    v1_09_label_selected_row,
    v1_09_part_a_checkpoint,
    v1_09_part_c_checkpoint,
    v1_09_part_d_checkpoint,
    v1_09_profile,
    v1_09_reflection,
    v1_09_release,
    v1_09_risk_tolerance,
    v1_09_selected_utility,
    v1_09_selection,
    v1_09_validation_focus,
    v1_09_validation_prediction,
    v1_09_value_prediction,
    v1_09_variant,
):
    _all_predictions = (
        v1_09_value_prediction.value,
        v1_09_coverage_prediction.value,
        v1_09_label_prediction.value,
        v1_09_validation_prediction.value,
    )
    if all(value is not None for value in _all_predictions):
        ledger.save(chapter=9, design={
            "chapter": "v1_09",
            "track_id": v1_09_profile.track_id,
            "scenario_id": v1_09_variant.scenario_id,
            "hardware_ref": v1_09_selection.hardware_ref,
            "model_ref": v1_09_selection.model_ref,
            "completed": True,
            "value_prediction": v1_09_value_prediction.value,
            "coverage_prediction": v1_09_coverage_prediction.value,
            "label_prediction": v1_09_label_prediction.value,
            "validation_prediction": v1_09_validation_prediction.value,
            "fraction_multiplier": v1_09_fraction_multiplier.value,
            "label_budget_multiplier": v1_09_label_budget_multiplier.value,
            "selected_policy": v1_09_decision.selected_id,
            "selected_cohort": v1_09_decision.selected_label,
            "selected_examples_k": v1_09_selected_utility.selected_examples_k,
            "binding_budget": v1_09_label_selected_row["binding_constraint"],
            "worst_subgroup": v1_09_decision.worst_subgroup,
            "residual_risk": v1_09_decision.residual_risk,
            "release_gate": v1_09_release["decision"],
            "carry_forward_risk": v1_09_amount_system["carry_forward"],
        })

    def build_synthesis():
        _rejections = "".join(f"<li>{item}</li>" for item in v1_09_decision.rejected_alternatives)
        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Synthesis: Data Selection Memo</h2>
              <div class="mlsysbook-grid">
                <div class="mlsysbook-field"><strong>Track</strong>{v1_09_selection.label}</div>
                <div class="mlsysbook-field"><strong>Selected cohort</strong>{v1_09_decision.selected_label}</div>
                <div class="mlsysbook-field"><strong>Binding budget</strong>{v1_09_label_selected_row["binding_constraint"]}</div>
                <div class="mlsysbook-field"><strong>Worst subgroup</strong>{v1_09_decision.worst_subgroup}</div>
                <div class="mlsysbook-field"><strong>Release gate</strong>{v1_09_release["decision"]}</div>
                <div class="mlsysbook-field"><strong>Carry-forward risk</strong>{v1_09_decision.residual_risk}</div>
              </div>
              <div class="mlsysbook-callout"><strong>Next data:</strong> {v1_09_decision.next_data}</div>
              <div class="mlsysbook-callout"><strong>Validation requirement:</strong> {v1_09_decision.validation_requirement}</div>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Rejected Alternatives</h2>
              <ul class="mlsysbook-list">{_rejections}</ul>
              <h2>Memo Completeness</h2>
              <ul class="mlsysbook-list">
                <li>Part A checkpoint: {v1_09_part_a_checkpoint.value or "not recorded"}</li>
                <li>Part B checkpoint: {v1_09_coverage_checkpoint.value or "not recorded"}; lowest-risk policy is {v1_09_coverage_best["policy_label"]}</li>
                <li>Part C checkpoint: {v1_09_part_c_checkpoint.value or "not recorded"}; budget multiplier is {v1_09_label_budget_multiplier.value:.2f}x</li>
                <li>Part D checkpoint: {v1_09_part_d_checkpoint.value or "not recorded"}; risk tolerance is {v1_09_risk_tolerance.value:.1f}</li>
                <li>Validation focus: {v1_09_validation_focus.value}</li>
              </ul>
            </div>
            """),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Big Takeaways</h2>
              <ul class="mlsysbook-list">
                <li><strong>Data value is marginal.</strong> Additional examples can add cost after useful signal saturates.</li>
                <li><strong>Coverage is a systems guardrail.</strong> A smaller cohort can be better if it protects the slices that deployment depends on.</li>
                <li><strong>Labels create a frontier.</strong> Quality, curation, review, storage, and processing are budgeted amounts.</li>
                <li><strong>Risk travels forward.</strong> The memo must defend residual bias and downstream validation, not just a score.</li>
              </ul>
            </div>
            """),
            mo.Html(f"""
            <div class="lab-hud">
                <span class="hud-label">LAB</span>
                <span class="hud-value">09 &middot; Data Selection</span>
                <span class="hud-label">TRACK</span>
                <span class="hud-value">{v1_09_profile.label}</span>
                <span style="flex:1;"></span>
                <span class="hud-label">ARTIFACT</span>
                <span class="hud-value">{v1_09_selection.report_artifact}</span>
                <span class="hud-label">STATUS</span>
                <span class="hud-active">ACTIVE</span>
            </div>
            """),
        ])

    build_synthesis()
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    report_export_panel,
    v1_09_amount_system,
    v1_09_coverage,
    v1_09_coverage_best,
    v1_09_coverage_checkpoint,
    v1_09_coverage_prediction,
    v1_09_decision,
    v1_09_fraction_multiplier,
    v1_09_frontier,
    v1_09_label_budget_multiplier,
    v1_09_label_prediction,
    v1_09_label_rows,
    v1_09_label_selected_row,
    v1_09_metadata,
    v1_09_part_a_checkpoint,
    v1_09_part_c_checkpoint,
    v1_09_part_d_checkpoint,
    v1_09_profile,
    v1_09_reflection,
    v1_09_release,
    v1_09_risk_rows,
    v1_09_risk_tolerance,
    v1_09_saturation_evidence,
    v1_09_saturation_points,
    v1_09_selected_utility,
    v1_09_selection,
    v1_09_validation_focus,
    v1_09_validation_prediction,
    v1_09_value_prediction,
    v1_09_variant,
):
    _incomplete = []
    if v1_09_value_prediction.value is None:
        _incomplete.append("Part A marginal-value prediction")
    if v1_09_part_a_checkpoint.value is None:
        _incomplete.append("Part A checkpoint")
    if v1_09_coverage_prediction.value is None:
        _incomplete.append("Part B coverage prediction")
    if v1_09_coverage_checkpoint.value is None:
        _incomplete.append("Part B checkpoint")
    if v1_09_label_prediction.value is None:
        _incomplete.append("Part C label-frontier prediction")
    if v1_09_part_c_checkpoint.value is None:
        _incomplete.append("Part C checkpoint")
    if v1_09_validation_prediction.value is None:
        _incomplete.append("Part D release-gate prediction")
    if v1_09_part_d_checkpoint.value is None:
        _incomplete.append("Part D checkpoint")
    if not str(v1_09_reflection.value or "").strip():
        _incomplete.append("Synthesis memo notes")

    _report = build_lab_report(
        v1_09_metadata,
        track=v1_09_profile.label,
        scenario=v1_09_variant.workload_summary,
        learning_objectives=(
            "Measure when marginal data value saturates while cost keeps growing.",
            "Compare coverage and diversity against raw dataset size.",
            "Choose a label-quality-cost frontier point under a binding budget.",
            "Defend residual bias and downstream risk in a data selection memo.",
        ),
        predictions={
            "marginal_value_pressure": v1_09_value_prediction.value,
            "coverage_policy": v1_09_coverage_prediction.value,
            "binding_label_budget": v1_09_label_prediction.value,
            "release_gate": v1_09_validation_prediction.value,
        },
        knob_settings={
            "fraction_multiplier": v1_09_fraction_multiplier.value,
            "selected_policy": v1_09_decision.selected_id,
            "label_budget_multiplier": v1_09_label_budget_multiplier.value,
            "risk_tolerance": v1_09_risk_tolerance.value,
            "validation_focus": v1_09_validation_focus.value,
        },
        binding_constraints={
            "binding_budget": v1_09_label_selected_row["binding_constraint"],
            "worst_subgroup": v1_09_decision.worst_subgroup,
            "release_gate": v1_09_release["binding_gate"],
        },
        decisions={
            "part_a_checkpoint": v1_09_part_a_checkpoint.value,
            "part_b_checkpoint": v1_09_coverage_checkpoint.value,
            "part_c_checkpoint": v1_09_part_c_checkpoint.value,
            "part_d_checkpoint": v1_09_part_d_checkpoint.value,
            "selected_policy": v1_09_decision.selected_label,
            "memo_decision": v1_09_decision.memo_summary,
        },
        evidence_summary={
            "selected_cohort": v1_09_decision.selected_label,
            "selected_examples_k": v1_09_selected_utility.selected_examples_k,
            "utility_score": v1_09_decision.utility_score,
            "current_icr_proxy": v1_09_saturation_evidence["current_icr_proxy"],
            "coverage_best_policy": v1_09_coverage_best["policy_label"],
            "binding_budget": v1_09_label_selected_row["binding_constraint"],
            "release_gate": v1_09_release["decision"],
            "next_data": v1_09_decision.next_data,
        },
        final_decision={
            "selected_policy": v1_09_decision.selected_label,
            "binding_budget": v1_09_label_selected_row["binding_constraint"],
            "rejected_alternatives": v1_09_decision.rejected_alternatives,
            "carry_forward_risk": v1_09_decision.residual_risk,
        },
        big_takeaways=(
            "Data quantity is not data value; marginal utility saturates.",
            "Coverage and diversity can dominate raw dataset size.",
            "Label cost and quality create a budgeted frontier.",
            "Residual bias and downstream risk must be defended before release.",
        ),
        reflections={
            "student_memo_notes": v1_09_reflection.value,
            "accepted_blind_spot": v1_09_decision.accepted_blind_spot,
            "validation_requirement": v1_09_decision.validation_requirement,
            "track_amount_system": v1_09_amount_system["amount_system"],
            "report_artifact": v1_09_selection.report_artifact,
        },
        residual_risk=v1_09_decision.residual_risk,
        source_trace={
            "track_id": v1_09_profile.track_id,
            "scenario_id": v1_09_variant.scenario_id,
            "hardware_ref": v1_09_variant.hardware_ref,
            "model_ref": v1_09_variant.model_ref,
            "shared_helper": "mlsysbook_labs.selection",
            "notebook_helpers": (
                "v1_09_saturation_point",
                "v1_09_label_frontier_rows",
                "v1_09_release_gate",
            ),
            "source_policy": v1_09_profile.source_policy,
        },
        result_snapshot={
            "selection_profile": v1_09_selection,
            "frontier": v1_09_frontier,
            "coverage": v1_09_coverage,
            "selected_utility": v1_09_selected_utility,
            "decision": v1_09_decision,
            "saturation_points": v1_09_saturation_points,
            "label_rows": v1_09_label_rows,
            "risk_rows": v1_09_risk_rows,
            "release_gate": v1_09_release,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-09 data selection memo is generated locally from the selected "
                "track, your predictions, controls, computed evidence, and synthesis notes."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
