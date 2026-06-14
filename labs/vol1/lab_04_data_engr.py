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
    from mlsysim.labs.components import MathPeek
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        data_pipeline_profile,
        evaluate_pipeline,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        movement_frontier,
        part_workflow,
        pipeline_architecture,
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
        MathPeek,
        apply_plotly_theme,
        build_lab_report,
        data_pipeline_profile,
        evaluate_pipeline,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        mo,
        movement_frontier,
        part_workflow,
        pipeline_architecture,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_04_metadata = get_lab_metadata("vol1/lab_04_data_engr.py")
    return (v1_04_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_04_track_picker = track_selector(default=_default_track)
    v1_04_track_picker
    return (v1_04_track_picker,)


@app.cell
def _(
    data_pipeline_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_04_track_picker,
):
    v1_04_track_id = v1_04_track_picker.value
    v1_04_profile = get_track_profile(v1_04_track_id)
    v1_04_variant = get_lab_track_variant("v1_04_data_gravity", v1_04_profile.track_id)
    v1_04_hardware = resolve_mlsysim_ref(v1_04_variant.hardware_ref)
    v1_04_model = resolve_mlsysim_ref(v1_04_variant.model_ref)
    v1_04_pipeline_profile = data_pipeline_profile(
        v1_04_profile,
        v1_04_variant,
        v1_04_hardware,
        v1_04_model,
    )
    return (
        v1_04_hardware,
        v1_04_model,
        v1_04_pipeline_profile,
        v1_04_profile,
        v1_04_track_id,
        v1_04_variant,
    )


@app.cell
def _():
    import html as _html

    def v1_04_clamp(value, low, high):
        return max(low, min(high, float(value)))

    def v1_04_fmt(value, unit="", precision=1):
        if isinstance(value, str):
            return _html.escape(value)
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            text = f"{value:,}"
        else:
            text = f"{float(value):,.{precision}f}"
        return f"{text} {unit}".strip()

    def v1_04_track_spec(track_id):
        specs = {
            "iphone": {
                "quality_subject": "private camera and app-context feature windows",
                "quality_failure": "private local data passes schema checks but loses coverage or leaks context",
                "base_defect_pct": 6.0,
                "target_defect_pct": 1.5,
                "coverage_floor_pct": 86.0,
                "review_limit_pct": 12.0,
                "true_metric_pct": 87.0,
                "validity_floor_pct": 84.0,
                "required_gap_h": 24.0,
                "leakage_allow_pct": 0.8,
                "freshness_sla_s": 8.0,
                "worker_base": 4.0,
                "worker_label": "on-device preprocessing lanes",
                "throughput_failure": "radio and local preprocessing pressure make the feature stale and battery-expensive",
                "contract_focus": "consent, deletion, feature schema, privacy lineage",
                "contract_threshold": 18.0,
                "debt_base": 9.0,
                "report_artifact": "local/private feature pipeline memo",
                "residual_risk": "rare private failures may remain unobservable if raw samples stay on-device",
            },
            "oura_ring": {
                "quality_subject": "nighttime PPG, temperature, motion, and derived biosignal windows",
                "quality_failure": "sensor dropout looks like clean data unless quality is budgeted by window",
                "base_defect_pct": 8.0,
                "target_defect_pct": 2.0,
                "coverage_floor_pct": 82.0,
                "review_limit_pct": 10.0,
                "true_metric_pct": 85.0,
                "validity_floor_pct": 82.0,
                "required_gap_h": 12.0,
                "leakage_allow_pct": 0.7,
                "freshness_sla_s": 60.0,
                "worker_base": 2.0,
                "worker_label": "firmware processing slots",
                "throughput_failure": "nighttime duty cycle and tiny flash create backlog before the phone syncs",
                "contract_focus": "sensor schema, duty-cycle guardrail, OTA compatibility, health-data lineage",
                "contract_threshold": 16.0,
                "debt_base": 11.0,
                "report_artifact": "nighttime sensor pipeline memo",
                "residual_risk": "summaries can miss waveform anomalies that never trigger snippet retention",
            },
            "robotaxi": {
                "quality_subject": "rare-event labels, scenario tags, and multi-sensor alignment windows",
                "quality_failure": "rare safety cases are under-counted even when aggregate labels look good",
                "base_defect_pct": 4.5,
                "target_defect_pct": 0.6,
                "coverage_floor_pct": 94.0,
                "review_limit_pct": 18.0,
                "true_metric_pct": 93.0,
                "validity_floor_pct": 91.0,
                "required_gap_h": 48.0,
                "leakage_allow_pct": 0.5,
                "freshness_sla_s": 20.0,
                "worker_base": 8.0,
                "worker_label": "triage workers per vehicle batch",
                "throughput_failure": "sensor streams overwhelm local triage and erase long-tail safety evidence",
                "contract_focus": "scenario-label ontology, route/time splits, safety lineage, redaction",
                "contract_threshold": 12.0,
                "debt_base": 13.0,
                "report_artifact": "safety validation data memo",
                "residual_risk": "unknown rare events can be filtered out before upload or review",
            },
            "cloud_fleet": {
                "quality_subject": "object-store shards, feature logs, production feedback, and freshness monitors",
                "quality_failure": "freshness and null drift pass local schemas but poison downstream features",
                "base_defect_pct": 5.5,
                "target_defect_pct": 1.0,
                "coverage_floor_pct": 90.0,
                "review_limit_pct": 15.0,
                "true_metric_pct": 90.0,
                "validity_floor_pct": 88.0,
                "required_gap_h": 24.0,
                "leakage_allow_pct": 0.8,
                "freshness_sla_s": 15.0,
                "worker_base": 16.0,
                "worker_label": "preprocessing worker groups",
                "throughput_failure": "accelerators starve when object-store reads and preprocessing tails backlog",
                "contract_focus": "producer schema, feature freshness, regional retention, access control",
                "contract_threshold": 15.0,
                "debt_base": 12.0,
                "report_artifact": "feature freshness and contract memo",
                "residual_risk": "cached or late-arriving features can bias training and serving comparisons",
            },
        }
        return specs.get(track_id, specs["iphone"])

    def v1_04_fields_html(fields):
        items = []
        for label, value in fields.items():
            items.append(
                f'<div class="mlsysbook-field"><strong>{_html.escape(str(label))}</strong>{_html.escape(str(value))}</div>'
            )
        return "".join(items)

    def v1_04_table_html(headers, rows, *, numeric=()):
        head = "".join(
            f'<th style="text-align:{"right" if index in numeric else "left"};">{_html.escape(str(header))}</th>'
            for index, header in enumerate(headers)
        )
        body_rows = []
        for row in rows:
            cells = []
            for index, value in enumerate(row):
                cells.append(
                    f'<td style="text-align:{"right" if index in numeric else "left"};">{_html.escape(str(value))}</td>'
                )
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        return f"""
        <table style="width:100%; border-collapse:collapse; margin-top:14px; font-size:0.86rem;">
          <thead><tr style="border-bottom:1px solid #D9DEE8; color:#667085;">{head}</tr></thead>
          <tbody>{"".join(body_rows)}</tbody>
        </table>
        """

    def v1_04_callout_html(title, body, *, kind="info"):
        palette = {
            "info": ("#1F4E7A", "#EFF8FF", "#B2DDFF"),
            "ok": ("#247A4D", "#F8FFFB", "#B8D8C6"),
            "warn": ("#B54708", "#FFFAEB", "#FEDF89"),
            "fail": ("#B42318", "#FEF3F2", "#FECDCA"),
        }
        accent, background, border = palette.get(kind, palette["info"])
        return f"""
        <div class="mlsysbook-callout" style="border-left:4px solid {accent}; background:{background};
             border-top:1px solid {border}; border-right:1px solid {border}; border-bottom:1px solid {border};
             padding:12px 14px; border-radius:8px; line-height:1.55;">
          <strong>{_html.escape(str(title))}</strong> {_html.escape(str(body))}
        </div>
        """

    def v1_04_prediction_html(title, prediction_value, actual_value, label_map):
        if prediction_value is None:
            return v1_04_callout_html(
                title,
                "Commit a structured prediction before treating the evidence as a decision.",
                kind="warn",
            )
        predicted = label_map.get(prediction_value, str(prediction_value))
        actual = label_map.get(actual_value, str(actual_value))
        kind = "ok" if prediction_value == actual_value else "warn"
        return v1_04_callout_html(
            title,
            f"Predicted: {predicted}. Measured: {actual}.",
            kind=kind,
        )

    def v1_04_quality_budget(profile, strictness_pct, review_pct):
        spec = v1_04_track_spec(profile.track_id)
        strictness = v1_04_clamp(strictness_pct, 0.0, 100.0)
        review = v1_04_clamp(review_pct, 0.0, 25.0)
        base_defects = spec["base_defect_pct"] / 100.0 * 10_000
        detection_rate = v1_04_clamp(0.18 + 0.0055 * strictness + 0.018 * review, 0.0, 0.94)
        caught = base_defects * detection_rate
        residual = max(0.0, base_defects - caught)
        residual_pct = residual / 100.0
        target = spec["target_defect_pct"] / 100.0 * 10_000
        coverage_loss = (strictness / 100.0) ** 1.35 * 10.5 + review * 0.08
        coverage = max(0.0, 100.0 - coverage_loss)
        review_load = review
        if residual > target:
            actual_failure = "residual_defects"
            mitigation = "raise validation strictness, add semantic monitors, or increase review sampling"
        elif coverage < spec["coverage_floor_pct"]:
            actual_failure = "coverage_loss"
            mitigation = "relax low-value filters and add targeted collection for lost cohorts"
        elif review_load > spec["review_limit_pct"]:
            actual_failure = "review_load"
            mitigation = "route review toward high-entropy or high-risk examples"
        else:
            actual_failure = "inside_budget"
            mitigation = "keep the quality gate and monitor for drift"
        return {
            "subject": spec["quality_subject"],
            "base_defects_per_10k": base_defects,
            "caught_defects_per_10k": caught,
            "residual_defects_per_10k": residual,
            "residual_defect_pct": residual_pct,
            "target_defects_per_10k": target,
            "target_defect_pct": spec["target_defect_pct"],
            "coverage_retained_pct": coverage,
            "coverage_floor_pct": spec["coverage_floor_pct"],
            "review_load_pct": review_load,
            "review_limit_pct": spec["review_limit_pct"],
            "detection_rate_pct": detection_rate * 100.0,
            "pass": actual_failure == "inside_budget",
            "actual_failure": actual_failure,
            "mitigation": mitigation,
            "narrative": spec["quality_failure"],
        }

    def v1_04_split_integrity(profile, leakage_pressure_pct, split_policy, temporal_gap_h):
        spec = v1_04_track_spec(profile.track_id)
        policies = {
            "random_record": ("Random record split", 1.00, "duplicates and related entities cross the boundary"),
            "entity_grouped": ("Entity/session grouped split", 0.35, "related examples are mostly isolated"),
            "time_entity": ("Time-aware entity split", 0.08, "future and entity leakage are both controlled"),
        }
        label, factor, note = policies.get(split_policy, policies["random_record"])
        leakage_pressure = v1_04_clamp(leakage_pressure_pct, 0.0, 25.0)
        gap = v1_04_clamp(temporal_gap_h, 0.0, 168.0)
        time_leak = max(0.0, (spec["required_gap_h"] - gap) / max(spec["required_gap_h"], 1.0)) * 4.0
        effective_leakage = leakage_pressure * factor + time_leak
        inflation = min(12.0, effective_leakage * 0.72 + leakage_pressure * 0.05)
        reported_metric = min(99.5, spec["true_metric_pct"] + inflation)
        adjusted_metric = max(0.0, spec["true_metric_pct"] - max(0.0, effective_leakage - spec["leakage_allow_pct"]) * 0.28)
        valid = effective_leakage <= spec["leakage_allow_pct"] and gap >= spec["required_gap_h"]
        if effective_leakage > spec["leakage_allow_pct"]:
            actual = "invalid_leakage"
            mitigation = "redo the split with entity and time boundaries before using the metric"
        elif gap < spec["required_gap_h"]:
            actual = "invalid_time"
            mitigation = "rebuild features with point-in-time retrieval and a larger temporal gap"
        else:
            actual = "valid"
            mitigation = "the split is defensible for this scenario"
        return {
            "policy_label": label,
            "policy_note": note,
            "effective_leakage_pct": effective_leakage,
            "allowed_leakage_pct": spec["leakage_allow_pct"],
            "temporal_gap_h": gap,
            "required_gap_h": spec["required_gap_h"],
            "reported_metric_pct": reported_metric,
            "adjusted_metric_pct": adjusted_metric,
            "validity_floor_pct": spec["validity_floor_pct"],
            "valid": valid,
            "actual_failure": actual,
            "mitigation": mitigation,
        }

    def v1_04_backlog_model(profile, pipeline_result, worker_count):
        spec = v1_04_track_spec(profile.track_id)
        workers = v1_04_clamp(worker_count, 1.0, 64.0)
        arrival = pipeline_result.effective_rate_mb_s
        base_workers = max(spec["worker_base"], 1.0)
        capacities = {
            "ingest": profile.ingest_capacity_mb_s,
            "preprocess": profile.preprocess_capacity_mb_s * workers / base_workers / 0.85,
            "storage write": profile.storage_capacity_mb_s / 0.65,
            "upload/movement": profile.upload_capacity_mb_s,
        }
        bottleneck_stage = min(capacities, key=capacities.get)
        service = max(0.001, capacities[bottleneck_stage])
        utilization = arrival / service * 100.0
        window_s = 30 * 60
        backlog_mb = max(0.0, arrival - service) * window_s
        freshness_lag_s = backlog_mb / service if service > 0 else 999999.0
        feasible = arrival <= service and freshness_lag_s <= spec["freshness_sla_s"]
        times = [0, 5, 10, 15, 20, 25, 30]
        backlog_gb_series = [max(0.0, arrival - service) * minute * 60 / 1024.0 for minute in times]
        if feasible:
            actual_failure = "no_failure"
            mitigation = "capacity exceeds arrival rate inside the freshness budget"
        else:
            actual_failure = bottleneck_stage
            mitigation = "reduce data arrival, move compute to data, add capacity at the bottleneck, or relax freshness"
        return {
            "arrival_mb_s": arrival,
            "service_mb_s": service,
            "worker_count": workers,
            "worker_label": spec["worker_label"],
            "bottleneck_stage": bottleneck_stage,
            "utilization_pct": utilization,
            "backlog_gb": backlog_mb / 1024.0,
            "freshness_lag_s": freshness_lag_s,
            "freshness_sla_s": spec["freshness_sla_s"],
            "feasible": feasible,
            "actual_failure": actual_failure,
            "mitigation": mitigation,
            "times_min": times,
            "backlog_gb_series": backlog_gb_series,
            "narrative": spec["throughput_failure"],
            "capacities": capacities,
        }

    def v1_04_contract_governance(profile, movement_result, contract_policy, change_pressure_pct):
        spec = v1_04_track_spec(profile.track_id)
        policies = {
            "none": ("No enforced contract", 0.12, False, False),
            "schema": ("Schema-only checks", 0.42, False, False),
            "schema_semantic": ("Schema + semantic checks", 0.68, True, False),
            "lineage_semantic": ("Schema + semantic + lineage", 0.86, True, True),
            "blocking": ("Blocking contract with lineage and freshness SLO", 0.94, True, True),
        }
        label, enforcement, semantic, lineage = policies.get(contract_policy, policies["schema"])
        change_pressure = v1_04_clamp(change_pressure_pct, 0.0, 40.0)
        growth_rate = change_pressure / 100.0
        debt_before = spec["debt_base"] * ((1.0 + growth_rate) ** 3)
        movement_penalty = max(0.0, 100.0 - movement_result.quality_retained_pct) * 0.25
        privacy_penalty = 5.0 if "raw" in movement_result.privacy_risk.lower() else 1.5
        debt_before += movement_penalty + privacy_penalty
        caught_debt = debt_before * enforcement
        silent_debt = max(0.0, debt_before - caught_debt)
        freshness_penalty = min(35.0, movement_result.effective_latency_s / max(spec["freshness_sla_s"], 1.0) * 0.08)
        debt_index = silent_debt + freshness_penalty
        if contract_policy == "none":
            actual_control = "schema"
            mitigation = "add a producer schema contract before downstream consumers depend on this data"
        elif not semantic:
            actual_control = "semantic"
            mitigation = "add semantic distribution and freshness checks, not only schema checks"
        elif not lineage:
            actual_control = "lineage"
            mitigation = "add lineage and point-in-time provenance before approving the downstream contract"
        elif debt_index > spec["contract_threshold"]:
            actual_control = "freshness"
            mitigation = "make freshness and incompatible upstream changes blocking contract failures"
        else:
            actual_control = "contract_ok"
            mitigation = "the contract is strong enough for this scenario; monitor residual debt"
        return {
            "policy_label": label,
            "contract_focus": spec["contract_focus"],
            "enforcement_pct": enforcement * 100.0,
            "lineage": lineage,
            "semantic": semantic,
            "debt_before": debt_before,
            "caught_debt": caught_debt,
            "silent_debt_index": debt_index,
            "contract_threshold": spec["contract_threshold"],
            "actual_control": actual_control,
            "pass": actual_control == "contract_ok",
            "mitigation": mitigation,
            "report_artifact": spec["report_artifact"],
            "track_residual_risk": spec["residual_risk"],
        }

    def v1_04_binding_constraint(quality, split, throughput, contract):
        candidates = []
        candidates.append((
            "quality budget",
            max(0.0, quality["residual_defects_per_10k"] - quality["target_defects_per_10k"]),
            quality["mitigation"],
            quality["pass"],
        ))
        candidates.append((
            "split integrity",
            max(0.0, split["effective_leakage_pct"] - split["allowed_leakage_pct"]) * 100.0,
            split["mitigation"],
            split["valid"],
        ))
        candidates.append((
            "throughput/backlog",
            max(0.0, throughput["utilization_pct"] - 100.0) + throughput["backlog_gb"],
            throughput["mitigation"],
            throughput["feasible"],
        ))
        candidates.append((
            "data contract",
            max(0.0, contract["silent_debt_index"] - contract["contract_threshold"]) * 5.0,
            contract["mitigation"],
            contract["pass"],
        ))
        failing = [item for item in candidates if not item[3]]
        selected = max(failing or candidates, key=lambda item: item[1])
        return {
            "label": selected[0],
            "severity": selected[1],
            "mitigation": selected[2],
            "all_pass": not failing,
        }

    def v1_04_snapshot(quality, split, throughput, contract):
        return {
            "quality": {
                "residual_defects_per_10k": quality["residual_defects_per_10k"],
                "target_defects_per_10k": quality["target_defects_per_10k"],
                "coverage_retained_pct": quality["coverage_retained_pct"],
                "pass": quality["pass"],
            },
            "split": {
                "effective_leakage_pct": split["effective_leakage_pct"],
                "reported_metric_pct": split["reported_metric_pct"],
                "adjusted_metric_pct": split["adjusted_metric_pct"],
                "valid": split["valid"],
            },
            "throughput": {
                "bottleneck_stage": throughput["bottleneck_stage"],
                "utilization_pct": throughput["utilization_pct"],
                "backlog_gb": throughput["backlog_gb"],
                "freshness_lag_s": throughput["freshness_lag_s"],
                "feasible": throughput["feasible"],
            },
            "contract": {
                "policy_label": contract["policy_label"],
                "silent_debt_index": contract["silent_debt_index"],
                "contract_threshold": contract["contract_threshold"],
                "pass": contract["pass"],
            },
        }

    return (
        v1_04_backlog_model,
        v1_04_binding_constraint,
        v1_04_callout_html,
        v1_04_contract_governance,
        v1_04_fields_html,
        v1_04_fmt,
        v1_04_prediction_html,
        v1_04_quality_budget,
        v1_04_snapshot,
        v1_04_split_integrity,
        v1_04_table_html,
        v1_04_track_spec,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    COLORS,
    LAB_CSS,
    mo,
    part_workflow,
    source_trace,
    track_arc_context,
    track_context,
    v1_04_metadata,
    v1_04_pipeline_profile,
    v1_04_profile,
    v1_04_track_spec,
    v1_04_variant,
):
    _spec = v1_04_track_spec(v1_04_profile.track_id)
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 62%, #10233f 100%);
                    padding: 34px 42px; border-radius: 14px; color: white;
                    box-shadow: 0 8px 28px rgba(0,0,0,0.30);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #94a3b8; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 04
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.35rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Data Engineering
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.08rem; font-weight: 650;
                      color: #bfdbfe; letter-spacing: 0.02em;">
                Quality budgets &middot; Split integrity &middot; Flow capacity &middot; Data contracts
            </p>
            <p style="margin: 0 0 20px 0; font-size: 1.0rem; color: #dbeafe;
                      max-width: 900px; line-height: 1.6;">
                {v1_04_variant.workload_summary} The invariant for this lab is that data is infrastructure:
                quality, lineage, throughput, and contracts are amount systems that shape model behavior.
            </p>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <span class="badge badge-info">4 Concept Modules + Synthesis</span>
                <span class="badge badge-warn">{v1_04_profile.label}</span>
                <span class="badge badge-fail">{_spec["report_artifact"]}</span>
            </div>
        </div>
        """),
        track_context(v1_04_profile),
        track_arc_context(v1_04_profile, v1_04_metadata.lab_id),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <div class="mlsysbook-section-label">Chapter Invariant</div>
          <h2>Data Is Infrastructure</h2>
          <p style="line-height:1.65; color:{COLORS['TextSec']};">
            The model inherits the data path. A pipeline decision is not complete until
            the quality budget, split boundary, flow capacity, and governance contract
            are measured for {v1_04_pipeline_profile.label}.
          </p>
        </div>
        """),
        part_workflow(
            "V1-04 Data Engineering Workflow",
            (
                {
                    "part": "Part A",
                    "concept": "Data quality is a measurable budget",
                    "prediction": "Predict which quality budget term fails first.",
                    "controls": "Tune validation strictness and review sampling.",
                    "evidence": "Compare caught defects, residual defects, coverage, and review load.",
                    "decision": "Choose the quality gate you would defend.",
                },
                {
                    "part": "Part B",
                    "concept": "Leakage can invalidate good-looking metrics",
                    "prediction": "Predict whether the metric is valid under the split boundary.",
                    "controls": "Change leakage pressure, split policy, and temporal gap.",
                    "evidence": "Compare reported and leakage-adjusted metric evidence.",
                    "decision": "Choose whether to ship, redo the split, or block the metric.",
                },
                {
                    "part": "Part C",
                    "concept": "Throughput and backlog are physical constraints",
                    "prediction": "Predict the first stage that makes the pipeline fall behind.",
                    "controls": "Adjust arrival pressure and processing workers.",
                    "evidence": "Inspect utilization, backlog growth, and freshness lag.",
                    "decision": "Choose the capacity or demand policy.",
                },
                {
                    "part": "Part D",
                    "concept": "Contracts prevent downstream data debt",
                    "prediction": "Predict which governance control prevents the next failure.",
                    "controls": "Select movement, retention, contract policy, network, and upstream change pressure.",
                    "evidence": "Compare movement frontier, caught violations, and silent debt.",
                    "decision": "Choose the contract gate for the final memo.",
                },
            ),
            scenario=(
                f"You are the {v1_04_variant.stakeholder.lower()} for {v1_04_pipeline_profile.label}; "
                f"the data source is {v1_04_pipeline_profile.data_source}."
            ),
            reflection="The synthesis records one binding data constraint and one residual risk in the Design Ledger.",
        ),
        source_trace(
            {
                "chapter": "vol1/data_engineering/data_engineering.qmd",
                "anchors": (
                    "Dataset Compilation",
                    "Data quality as code",
                    "Data ingestion and backpressure",
                    "Transformation lineage",
                    "Data debt",
                ),
                "shared_helper": "mlsysbook_labs.data_pipeline",
                "scenario_id": v1_04_variant.scenario_id,
            },
            summary="Opening source map",
        ),
    ])
    return


# ===========================================================================
# ZONE B: CONTROLS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo):
    v1_04_quality_prediction = mo.ui.radio(
        options={
            "Residual defects exceed the budget": "residual_defects",
            "Strict filters remove too much coverage": "coverage_loss",
            "Review/audit load exceeds capacity": "review_load",
            "The quality gate stays inside budget": "inside_budget",
        },
        label="Part A prediction: which quality budget term fails first?",
    )
    v1_04_validation_strictness = mo.ui.slider(
        start=20,
        stop=95,
        value=65,
        step=5,
        label="Validation strictness (%)",
    )
    v1_04_review_sample = mo.ui.slider(
        start=0,
        stop=20,
        value=5,
        step=1,
        label="Review/audit sample (%)",
    )
    v1_04_quality_checkpoint = mo.ui.radio(
        options={
            "Tighten semantic validation and rerun before training": "tighten_validation",
            "Increase targeted review for high-risk records": "increase_review",
            "Relax filters and collect missing coverage": "restore_coverage",
            "Accept this quality gate and monitor drift": "accept_quality_gate",
        },
        label="Part A checkpoint: what quality gate goes into the memo?",
    )
    return (
        v1_04_quality_checkpoint,
        v1_04_quality_prediction,
        v1_04_review_sample,
        v1_04_validation_strictness,
    )


@app.cell(hide_code=True)
def _(mo):
    v1_04_split_prediction = mo.ui.radio(
        options={
            "Metric is valid enough to use": "valid",
            "Metric is inflated by leakage": "invalid_leakage",
            "Metric uses future or stale features": "invalid_time",
        },
        label="Part B prediction: is the reported metric valid evidence?",
    )
    v1_04_leakage_pressure = mo.ui.slider(
        start=0,
        stop=20,
        value=8,
        step=1,
        label="Duplicate/entity/future leakage pressure (%)",
    )
    v1_04_split_policy = mo.ui.dropdown(
        options={
            "Random record split": "random_record",
            "Entity/session grouped split": "entity_grouped",
            "Time-aware entity split": "time_entity",
        },
        value="Random record split",
        label="Split policy",
    )
    v1_04_temporal_gap = mo.ui.slider(
        start=0,
        stop=96,
        value=8,
        step=4,
        label="Temporal gap before evaluation (hours)",
    )
    v1_04_split_checkpoint = mo.ui.radio(
        options={
            "Use the metric as release evidence": "use_metric",
            "Redo the split with entity and time boundaries": "redo_split",
            "Block until point-in-time features are fixed": "block_point_in_time",
        },
        label="Part B checkpoint: what happens to the metric?",
    )
    return (
        v1_04_leakage_pressure,
        v1_04_split_checkpoint,
        v1_04_split_policy,
        v1_04_split_prediction,
        v1_04_temporal_gap,
    )


@app.cell(hide_code=True)
def _(mo, v1_04_pipeline_profile):
    v1_04_throughput_prediction = mo.ui.radio(
        options={
            "Ingest is the first flow wall": "ingest",
            "Preprocessing is the first flow wall": "preprocess",
            "Storage write is the first flow wall": "storage write",
            "Upload/movement is the first flow wall": "upload/movement",
            "No flow wall inside the tested envelope": "no_failure",
        },
        label="Part C prediction: which stage creates backlog first?",
    )
    v1_04_flow_multiplier = mo.ui.slider(
        start=v1_04_pipeline_profile.sample_min,
        stop=v1_04_pipeline_profile.sample_max,
        value=v1_04_pipeline_profile.default_sample_multiplier,
        step=v1_04_pipeline_profile.sample_step,
        label="Sampling or traffic multiplier",
    )
    v1_04_worker_count = mo.ui.slider(
        start=1,
        stop=64,
        value=8 if v1_04_pipeline_profile.track_id in {"robotaxi", "cloud_fleet"} else 4,
        step=1,
        label="Processing workers or lanes",
    )
    v1_04_throughput_checkpoint = mo.ui.radio(
        options={
            "Add capacity at the bottleneck": "add_capacity",
            "Reduce collection rate or sampling": "reduce_arrival",
            "Move compute closer to data": "move_compute",
            "Accept the backlog with an explicit freshness risk": "accept_backlog",
        },
        label="Part C checkpoint: what policy controls flow?",
    )
    return (
        v1_04_flow_multiplier,
        v1_04_throughput_checkpoint,
        v1_04_throughput_prediction,
        v1_04_worker_count,
    )


@app.cell(hide_code=True)
def _(mo, v1_04_pipeline_profile):
    _strategy_options = {strategy.label: strategy.strategy_id for strategy in v1_04_pipeline_profile.strategies}
    v1_04_contract_prediction = mo.ui.radio(
        options={
            "Schema checks prevent the next failure": "schema",
            "Semantic distribution checks prevent it": "semantic",
            "Lineage and point-in-time provenance prevent it": "lineage",
            "Freshness SLO enforcement prevents it": "freshness",
            "No new contract is needed": "contract_ok",
        },
        label="Part D prediction: which governance control prevents the next downstream failure?",
    )
    v1_04_strategy = mo.ui.dropdown(
        options=_strategy_options,
        value=v1_04_pipeline_profile.strategies[0].label,
        label="Movement strategy",
    )
    v1_04_dataset_gb = mo.ui.slider(
        start=1,
        stop=5000,
        value=500,
        step=50,
        label="Dataset or event window to move (GB)",
    )
    v1_04_network_gbps = mo.ui.dropdown(
        options={"1 Gbps": 1, "10 Gbps": 10, "25 Gbps": 25, "100 Gbps": 100},
        value="10 Gbps",
        label="Network bandwidth",
    )
    _retention_options = {policy: policy for policy in v1_04_pipeline_profile.retention_options}
    v1_04_retention_policy = mo.ui.dropdown(
        options=_retention_options,
        value=v1_04_pipeline_profile.retention_options[0],
        label="Retention policy",
    )
    v1_04_contract_policy = mo.ui.dropdown(
        options={
            "No enforced contract": "none",
            "Schema-only checks": "schema",
            "Schema + semantic checks": "schema_semantic",
            "Schema + semantic + lineage": "lineage_semantic",
            "Blocking contract with lineage and freshness SLO": "blocking",
        },
        value="Schema + semantic checks",
        label="Contract enforcement",
    )
    v1_04_change_pressure = mo.ui.slider(
        start=0,
        stop=35,
        value=12,
        step=1,
        label="Upstream change pressure (% per release cycle)",
    )
    v1_04_contract_checkpoint = mo.ui.radio(
        options={
            "Block incompatible producer changes": "block_changes",
            "Require lineage before training or serving": "require_lineage",
            "Allow with monitor and remediation budget": "allow_with_monitor",
            "Defer governance and accept data debt": "defer_governance",
        },
        label="Part D checkpoint: what contract gate goes into the memo?",
    )
    return (
        v1_04_change_pressure,
        v1_04_contract_checkpoint,
        v1_04_contract_policy,
        v1_04_contract_prediction,
        v1_04_dataset_gb,
        v1_04_network_gbps,
        v1_04_retention_policy,
        v1_04_strategy,
    )


@app.cell(hide_code=True)
def _(mo):
    v1_04_final_stance = mo.ui.radio(
        options={
            "Proceed with the binding data constraint recorded": "proceed_with_constraint",
            "Redesign the pipeline before launch": "redesign_before_launch",
            "Collect more data evidence before launch": "collect_more_evidence",
        },
        label="Synthesis decision: what is the pipeline stance?",
    )
    v1_04_residual_risk_note = mo.ui.text_area(
        label="Residual risk for the memo",
        placeholder="Name the evidence you still might miss, the downstream model behavior it could affect, and the trigger for revisiting this decision.",
        full_width=True,
    )
    return (v1_04_final_stance, v1_04_residual_risk_note)


@app.cell
def _(
    evaluate_pipeline,
    movement_frontier,
    pipeline_architecture,
    v1_04_backlog_model,
    v1_04_binding_constraint,
    v1_04_change_pressure,
    v1_04_contract_governance,
    v1_04_contract_policy,
    v1_04_dataset_gb,
    v1_04_flow_multiplier,
    v1_04_leakage_pressure,
    v1_04_network_gbps,
    v1_04_pipeline_profile,
    v1_04_quality_budget,
    v1_04_retention_policy,
    v1_04_review_sample,
    v1_04_split_integrity,
    v1_04_split_policy,
    v1_04_strategy,
    v1_04_temporal_gap,
    v1_04_validation_strictness,
    v1_04_worker_count,
):
    v1_04_quality_result = v1_04_quality_budget(
        v1_04_pipeline_profile,
        v1_04_validation_strictness.value,
        v1_04_review_sample.value,
    )
    v1_04_split_result = v1_04_split_integrity(
        v1_04_pipeline_profile,
        v1_04_leakage_pressure.value,
        v1_04_split_policy.value,
        v1_04_temporal_gap.value,
    )
    v1_04_pipeline_result = evaluate_pipeline(
        v1_04_pipeline_profile,
        sample_multiplier=v1_04_flow_multiplier.value,
    )
    v1_04_throughput_result = v1_04_backlog_model(
        v1_04_pipeline_profile,
        v1_04_pipeline_result,
        v1_04_worker_count.value,
    )
    v1_04_movement_result = movement_frontier(
        v1_04_pipeline_profile,
        strategy_id=v1_04_strategy.value,
        dataset_gb=v1_04_dataset_gb.value,
        network_gbps=v1_04_network_gbps.value,
    )
    v1_04_architecture = pipeline_architecture(
        v1_04_pipeline_profile,
        v1_04_pipeline_result,
        v1_04_movement_result,
        retention_policy=v1_04_retention_policy.value,
    )
    v1_04_contract_result = v1_04_contract_governance(
        v1_04_pipeline_profile,
        v1_04_movement_result,
        v1_04_contract_policy.value,
        v1_04_change_pressure.value,
    )
    v1_04_binding_result = v1_04_binding_constraint(
        v1_04_quality_result,
        v1_04_split_result,
        v1_04_throughput_result,
        v1_04_contract_result,
    )
    return (
        v1_04_architecture,
        v1_04_binding_result,
        v1_04_contract_result,
        v1_04_movement_result,
        v1_04_pipeline_result,
        v1_04_quality_result,
        v1_04_split_result,
        v1_04_throughput_result,
    )


# ===========================================================================
# ZONE C: CONCEPT MODULES
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    MathPeek,
    apply_plotly_theme,
    go,
    mo,
    source_trace,
    v1_04_architecture,
    v1_04_binding_result,
    v1_04_callout_html,
    v1_04_change_pressure,
    v1_04_contract_checkpoint,
    v1_04_contract_policy,
    v1_04_contract_prediction,
    v1_04_contract_result,
    v1_04_dataset_gb,
    v1_04_fields_html,
    v1_04_final_stance,
    v1_04_flow_multiplier,
    v1_04_leakage_pressure,
    v1_04_movement_result,
    v1_04_network_gbps,
    v1_04_pipeline_profile,
    v1_04_pipeline_result,
    v1_04_prediction_html,
    v1_04_profile,
    v1_04_quality_checkpoint,
    v1_04_quality_prediction,
    v1_04_quality_result,
    v1_04_residual_risk_note,
    v1_04_retention_policy,
    v1_04_review_sample,
    v1_04_split_checkpoint,
    v1_04_split_policy,
    v1_04_split_prediction,
    v1_04_split_result,
    v1_04_strategy,
    v1_04_table_html,
    v1_04_temporal_gap,
    v1_04_throughput_checkpoint,
    v1_04_throughput_prediction,
    v1_04_throughput_result,
    v1_04_track_spec,
    v1_04_validation_strictness,
    v1_04_variant,
    v1_04_worker_count,
):
    _spec = v1_04_track_spec(v1_04_profile.track_id)

    _quality_fig = go.Figure()
    _quality_fig.add_trace(go.Bar(
        x=["Caught defects", "Residual defects", "Budget"],
        y=[
            v1_04_quality_result["caught_defects_per_10k"],
            v1_04_quality_result["residual_defects_per_10k"],
            v1_04_quality_result["target_defects_per_10k"],
        ],
        marker_color=[COLORS["GreenLine"], COLORS["RedLine"], COLORS["BlueLine"]],
        text=[
            f'{v1_04_quality_result["caught_defects_per_10k"]:.0f}',
            f'{v1_04_quality_result["residual_defects_per_10k"]:.0f}',
            f'{v1_04_quality_result["target_defects_per_10k"]:.0f}',
        ],
        textposition="outside",
    ))
    _quality_fig.update_layout(
        height=310,
        yaxis=dict(title="Records per 10K", gridcolor="#f1f5f9"),
        xaxis=dict(title="Quality budget term", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=30, b=55),
    )
    apply_plotly_theme(_quality_fig)

    _split_fig = go.Figure()
    _split_fig.add_trace(go.Bar(
        x=["Reported metric", "Leakage-adjusted", "Validity floor"],
        y=[
            v1_04_split_result["reported_metric_pct"],
            v1_04_split_result["adjusted_metric_pct"],
            v1_04_split_result["validity_floor_pct"],
        ],
        marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"]],
        text=[
            f'{v1_04_split_result["reported_metric_pct"]:.1f}%',
            f'{v1_04_split_result["adjusted_metric_pct"]:.1f}%',
            f'{v1_04_split_result["validity_floor_pct"]:.1f}%',
        ],
        textposition="outside",
    ))
    _split_fig.update_layout(
        height=310,
        yaxis=dict(title="Metric (%)", range=[0, 105], gridcolor="#f1f5f9"),
        xaxis=dict(title="Evidence view", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=30, b=55),
    )
    apply_plotly_theme(_split_fig)

    _throughput_fig = go.Figure()
    _throughput_fig.add_trace(go.Scatter(
        x=v1_04_throughput_result["times_min"],
        y=v1_04_throughput_result["backlog_gb_series"],
        mode="lines+markers",
        line=dict(color=COLORS["RedLine"], width=3),
        marker=dict(size=7),
        name="Backlog",
    ))
    _throughput_fig.update_layout(
        height=310,
        yaxis=dict(title="Backlog (GB)", gridcolor="#f1f5f9"),
        xaxis=dict(title="Minutes under burst", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=30, b=55),
    )
    apply_plotly_theme(_throughput_fig)

    _stage_fig = go.Figure()
    _stage_fig.add_trace(go.Bar(
        x=[stage.stage for stage in v1_04_pipeline_result.stages],
        y=[stage.utilization_pct for stage in v1_04_pipeline_result.stages],
        marker_color=[
            COLORS["RedLine"] if stage.utilization_pct > 100 else COLORS["GreenLine"]
            for stage in v1_04_pipeline_result.stages
        ],
        text=[f"{stage.utilization_pct:.0f}%" for stage in v1_04_pipeline_result.stages],
        textposition="outside",
    ))
    _stage_fig.add_hline(y=100, line_dash="dash", line_color=COLORS["RedLine"], line_width=1.3)
    _stage_fig.update_layout(
        height=310,
        yaxis=dict(title="Stage utilization (%)", gridcolor="#f1f5f9"),
        xaxis=dict(title="Pipeline stage", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=30, b=65),
    )
    apply_plotly_theme(_stage_fig)

    _contract_fig = go.Figure()
    _contract_fig.add_trace(go.Bar(
        x=["Debt before contract", "Caught by contract", "Silent debt", "Threshold"],
        y=[
            v1_04_contract_result["debt_before"],
            v1_04_contract_result["caught_debt"],
            v1_04_contract_result["silent_debt_index"],
            v1_04_contract_result["contract_threshold"],
        ],
        marker_color=[COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["RedLine"], COLORS["BlueLine"]],
        text=[
            f'{v1_04_contract_result["debt_before"]:.1f}',
            f'{v1_04_contract_result["caught_debt"]:.1f}',
            f'{v1_04_contract_result["silent_debt_index"]:.1f}',
            f'{v1_04_contract_result["contract_threshold"]:.1f}',
        ],
        textposition="outside",
    ))
    _contract_fig.update_layout(
        height=310,
        yaxis=dict(title="Debt index", gridcolor="#f1f5f9"),
        xaxis=dict(title="Governance term", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=30, b=60),
    )
    apply_plotly_theme(_contract_fig)

    _quality_labels = {
        "residual_defects": "residual defects exceed the budget",
        "coverage_loss": "strict filters remove too much coverage",
        "review_load": "review/audit load exceeds capacity",
        "inside_budget": "quality gate is inside budget",
    }
    _split_labels = {
        "valid": "metric is valid enough to use",
        "invalid_leakage": "metric is inflated by leakage",
        "invalid_time": "metric uses future or stale features",
    }
    _throughput_labels = {
        "ingest": "ingest creates backlog",
        "preprocess": "preprocessing creates backlog",
        "storage write": "storage write creates backlog",
        "upload/movement": "upload or movement creates backlog",
        "no_failure": "no flow wall inside the tested envelope",
    }
    _contract_labels = {
        "schema": "schema checks are the missing control",
        "semantic": "semantic distribution checks are the missing control",
        "lineage": "lineage and point-in-time provenance are missing",
        "freshness": "freshness SLO enforcement is missing",
        "contract_ok": "the selected contract is strong enough",
    }

    _part_a = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Data Quality Is A Budget</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            As {v1_04_variant.stakeholder}, you must decide whether {v1_04_quality_result["subject"]}
            can enter training or serving. The budget is explicit: residual defects, coverage retained,
            and review load all have limits.</div>
        </div>
        """),
        mo.hstack([v1_04_quality_prediction], justify="start"),
        mo.hstack([v1_04_validation_strictness, v1_04_review_sample], justify="start", gap="2rem"),
        mo.Html(v1_04_prediction_html(
            "Prediction Check",
            v1_04_quality_prediction.value,
            v1_04_quality_result["actual_failure"],
            _quality_labels,
        )),
        mo.as_html(_quality_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Quality Budget Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_04_fields_html({
                "Track data": v1_04_quality_result["subject"],
                "Detection rate": f'{v1_04_quality_result["detection_rate_pct"]:.1f}%',
                "Residual defects": f'{v1_04_quality_result["residual_defects_per_10k"]:.0f} per 10K',
                "Budget": f'{v1_04_quality_result["target_defects_per_10k"]:.0f} per 10K',
                "Coverage retained": f'{v1_04_quality_result["coverage_retained_pct"]:.1f}% / floor {v1_04_quality_result["coverage_floor_pct"]:.1f}%',
                "Review load": f'{v1_04_quality_result["review_load_pct"]:.1f}% / limit {v1_04_quality_result["review_limit_pct"]:.1f}%',
            })}
          </div>
          {v1_04_table_html(
              ("Term", "Value", "Limit", "Status"),
              (
                  ("Residual defects", f'{v1_04_quality_result["residual_defects_per_10k"]:.0f} per 10K', f'{v1_04_quality_result["target_defects_per_10k"]:.0f} per 10K', "pass" if v1_04_quality_result["residual_defects_per_10k"] <= v1_04_quality_result["target_defects_per_10k"] else "fail"),
                  ("Coverage retained", f'{v1_04_quality_result["coverage_retained_pct"]:.1f}%', f'>= {v1_04_quality_result["coverage_floor_pct"]:.1f}%', "pass" if v1_04_quality_result["coverage_retained_pct"] >= v1_04_quality_result["coverage_floor_pct"] else "fail"),
                  ("Review load", f'{v1_04_quality_result["review_load_pct"]:.1f}%', f'<= {v1_04_quality_result["review_limit_pct"]:.1f}%', "pass" if v1_04_quality_result["review_load_pct"] <= v1_04_quality_result["review_limit_pct"] else "fail"),
              ),
              numeric=(1, 2),
          )}
        </div>
        """),
        mo.Html(v1_04_callout_html(
            "Consequence",
            (
                "Inside the quality budget. "
                if v1_04_quality_result["pass"]
                else "Budget violation. "
            ) + v1_04_quality_result["mitigation"],
            kind="ok" if v1_04_quality_result["pass"] else "fail",
        )),
        MathPeek(
            "residual_defects = base_defects x (1 - detection_rate); pass if residual <= budget and coverage >= floor",
            {
                "base_defects": "Track-specific defect pressure for the selected data source.",
                "detection_rate": "Validation strictness plus review sampling.",
                "coverage": "Useful examples retained after filtering.",
            },
        ),
        source_trace(
            {
                "chapter_anchor": "Data quality as code; Quality through validation and monitoring; Quality debt remediation",
                "formula": "residual defects and budget pass/fail",
                "track_id": v1_04_profile.track_id,
                "scenario_id": v1_04_variant.scenario_id,
            },
            summary="Part A source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_04_quality_checkpoint,
    ])

    _part_b = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Leakage Makes Evidence Invalid</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            {v1_04_pipeline_profile.label} has a strong reported metric. Your decision is whether
            the split boundary actually represents deployment, not familiar entities or future data.</div>
        </div>
        """),
        v1_04_split_prediction,
        mo.hstack([v1_04_leakage_pressure, v1_04_split_policy, v1_04_temporal_gap], justify="start", gap="1.5rem"),
        mo.Html(v1_04_prediction_html(
            "Prediction Check",
            v1_04_split_prediction.value,
            v1_04_split_result["actual_failure"],
            _split_labels,
        )),
        mo.as_html(_split_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Split Integrity Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_04_fields_html({
                "Split policy": v1_04_split_result["policy_label"],
                "Effective leakage": f'{v1_04_split_result["effective_leakage_pct"]:.2f}% / allow {v1_04_split_result["allowed_leakage_pct"]:.2f}%',
                "Temporal gap": f'{v1_04_split_result["temporal_gap_h"]:.0f} h / required {v1_04_split_result["required_gap_h"]:.0f} h',
                "Reported metric": f'{v1_04_split_result["reported_metric_pct"]:.1f}%',
                "Adjusted metric": f'{v1_04_split_result["adjusted_metric_pct"]:.1f}%',
                "Evidence valid": "yes" if v1_04_split_result["valid"] else "no",
            })}
          </div>
          {v1_04_table_html(
              ("Boundary", "Measured", "Required", "Status"),
              (
                  ("Leakage", f'{v1_04_split_result["effective_leakage_pct"]:.2f}%', f'<= {v1_04_split_result["allowed_leakage_pct"]:.2f}%', "pass" if v1_04_split_result["effective_leakage_pct"] <= v1_04_split_result["allowed_leakage_pct"] else "fail"),
                  ("Temporal gap", f'{v1_04_split_result["temporal_gap_h"]:.0f} h', f'>= {v1_04_split_result["required_gap_h"]:.0f} h', "pass" if v1_04_split_result["temporal_gap_h"] >= v1_04_split_result["required_gap_h"] else "fail"),
                  ("Metric adjustment", f'{v1_04_split_result["reported_metric_pct"] - v1_04_split_result["adjusted_metric_pct"]:.1f} pp', "0 pp preferred", "review"),
              ),
              numeric=(1, 2),
          )}
        </div>
        """),
        mo.Html(v1_04_callout_html(
            "Consequence",
            (
                "The metric can support the decision. "
                if v1_04_split_result["valid"]
                else "The metric is not release evidence. "
            ) + v1_04_split_result["mitigation"],
            kind="ok" if v1_04_split_result["valid"] else "fail",
        )),
        MathPeek(
            "reported_metric = true_metric + leakage_inflation; evidence is valid only inside the split boundary",
            {
                "leakage_inflation": "Extra apparent performance from duplicate, entity, augmentation, or future information crossing the boundary.",
                "point-in-time": "Features must be available at prediction time, not reconstructed from the future.",
                "adjusted_metric": "Reported metric after removing the leakage advantage.",
            },
        ),
        source_trace(
            {
                "chapter_anchor": "Dataset Compilation leakage paragraph; Data versioning; Feature stores and point-in-time correctness",
                "formula": "reported metric inflation and split validity",
                "track_id": v1_04_profile.track_id,
                "scenario_id": v1_04_variant.scenario_id,
            },
            summary="Part B source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_04_split_checkpoint,
    ])

    _flow_rows = []
    for _name, _capacity in v1_04_throughput_result["capacities"].items():
        _flow_rows.append((
            _name,
            f'{v1_04_throughput_result["arrival_mb_s"]:.2f} MB/s',
            f'{_capacity:.2f} MB/s',
            "binding" if _name == v1_04_throughput_result["bottleneck_stage"] else "headroom",
        ))

    _part_c = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Throughput And Backlog Are Physical Constraints</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            {v1_04_throughput_result["narrative"]}. The question is whether data arrives
            faster than the track can ingest, preprocess, store, or move it.</div>
        </div>
        """),
        v1_04_throughput_prediction,
        mo.hstack([v1_04_flow_multiplier, v1_04_worker_count], justify="start", gap="2rem"),
        mo.Html(v1_04_prediction_html(
            "Prediction Check",
            v1_04_throughput_prediction.value,
            v1_04_throughput_result["actual_failure"],
            _throughput_labels,
        )),
        mo.hstack([mo.as_html(_stage_fig), mo.as_html(_throughput_fig)], widths="equal"),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Flow Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_04_fields_html({
                "Arrival rate": f'{v1_04_throughput_result["arrival_mb_s"]:.2f} MB/s',
                "Service rate": f'{v1_04_throughput_result["service_mb_s"]:.2f} MB/s',
                "Binding stage": v1_04_throughput_result["bottleneck_stage"],
                "Utilization": f'{v1_04_throughput_result["utilization_pct"]:.1f}%',
                "Backlog after 30 min": f'{v1_04_throughput_result["backlog_gb"]:.2f} GB',
                "Freshness lag": f'{v1_04_throughput_result["freshness_lag_s"]:.1f} s / SLO {v1_04_throughput_result["freshness_sla_s"]:.1f} s',
            })}
          </div>
          {v1_04_table_html(("Stage", "Arrival", "Capacity", "Interpretation"), tuple(_flow_rows), numeric=(1, 2))}
        </div>
        """),
        mo.Html(v1_04_callout_html(
            "Consequence",
            (
                "Flow stays inside the freshness budget. "
                if v1_04_throughput_result["feasible"]
                else "Backlog is a physical constraint. "
            ) + v1_04_throughput_result["mitigation"],
            kind="ok" if v1_04_throughput_result["feasible"] else "fail",
        )),
        MathPeek(
            "backlog(t) = max(0, arrival_rate - service_rate) x t; freshness_lag = backlog / service_rate",
            {
                "arrival_rate": "Track data rate after sampling or traffic multiplier.",
                "service_rate": "The smallest effective capacity across ingest, preprocessing, storage, and movement.",
                "freshness_lag": "How long queued data waits before downstream consumers can use it.",
            },
        ),
        source_trace(
            {
                "chapter_anchor": "The feeding problem; Data ingestion; Batch vs. streaming ingestion; Storage performance",
                "shared_helper": "evaluate_pipeline()",
                "local_helper": "v1_04_backlog_model()",
                "track_id": v1_04_profile.track_id,
            },
            summary="Part C source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_04_throughput_checkpoint,
    ])

    _strategy_rows = []
    for _strategy in v1_04_pipeline_profile.strategies:
        _selected = _strategy.strategy_id == v1_04_movement_result.strategy_id
        _strategy_rows.append((
            _strategy.label,
            f"{v1_04_dataset_gb.value * _strategy.data_reduction_factor:.1f} GB",
            f"{_strategy.quality_factor * 100.0:.1f}%",
            _strategy.privacy_risk,
            "selected" if _selected else "candidate",
        ))

    _part_d = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part D: Contracts Prevent Data Debt</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            Upstream producers keep changing {v1_04_pipeline_profile.data_source}. The movement
            and retention choice must be paired with a contract: {v1_04_contract_result["contract_focus"]}.</div>
        </div>
        """),
        v1_04_contract_prediction,
        mo.hstack([v1_04_strategy, v1_04_network_gbps, v1_04_contract_policy], justify="start", gap="1.5rem"),
        mo.hstack([v1_04_dataset_gb, v1_04_change_pressure], justify="start", gap="2rem"),
        v1_04_retention_policy,
        mo.Html(v1_04_prediction_html(
            "Prediction Check",
            v1_04_contract_prediction.value,
            v1_04_contract_result["actual_control"],
            _contract_labels,
        )),
        mo.as_html(_contract_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Movement And Contract Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_04_fields_html({
                "Selected strategy": v1_04_movement_result.strategy_label,
                "Data moved": f'{v1_04_movement_result.data_moved_gb:.1f} GB',
                "Transfer time": f'{v1_04_movement_result.transfer_hours:.2f} h',
                "Egress cost": f'${v1_04_movement_result.egress_cost:.2f}',
                "Quality retained": f'{v1_04_movement_result.quality_retained_pct:.1f}%',
                "Silent debt": f'{v1_04_contract_result["silent_debt_index"]:.1f} / limit {v1_04_contract_result["contract_threshold"]:.1f}',
            })}
          </div>
          {v1_04_table_html(("Strategy", "Data moved", "Quality", "Governance exposure", "Status"), tuple(_strategy_rows), numeric=(1, 2))}
          {v1_04_table_html(
              ("Contract term", "Value", "Required", "Status"),
              (
                  ("Policy", v1_04_contract_result["policy_label"], "semantic checks plus lineage for production ML", "pass" if v1_04_contract_result["lineage"] and v1_04_contract_result["semantic"] else "gap"),
                  ("Enforcement", f'{v1_04_contract_result["enforcement_pct"]:.1f}%', "high enough to catch incompatible changes", "review"),
                  ("Silent debt", f'{v1_04_contract_result["silent_debt_index"]:.1f}', f'<= {v1_04_contract_result["contract_threshold"]:.1f}', "pass" if v1_04_contract_result["silent_debt_index"] <= v1_04_contract_result["contract_threshold"] else "fail"),
              ),
              numeric=(1, 2),
          )}
        </div>
        """),
        mo.Html(v1_04_callout_html(
            "Consequence",
            (
                "The contract is strong enough for this track. "
                if v1_04_contract_result["pass"]
                else "Downstream data debt remains unmanaged. "
            ) + v1_04_contract_result["mitigation"],
            kind="ok" if v1_04_contract_result["pass"] else "fail",
        )),
        MathPeek(
            "Debt_n = Debt_0 x (1 + r)^n; silent_debt = Debt_n x (1 - enforcement)",
            {
                "r": "Upstream change pressure per release cycle.",
                "enforcement": "Fraction of incompatible changes caught before downstream use.",
                "silent_debt": "Residual governance risk carried by the model pipeline.",
            },
        ),
        source_trace(
            {
                "chapter_anchor": "Data cascades; Transformation lineage; Data debt; Remediation strategies",
                "shared_helpers": "movement_frontier() and pipeline_architecture()",
                "local_helper": "v1_04_contract_governance()",
                "track_id": v1_04_profile.track_id,
            },
            summary="Part D source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_04_contract_checkpoint,
    ])

    _synthesis = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Synthesis: Record The Pipeline Decision</h2></div>
          <div class="mlsysbook-callout"><strong>Invariant:</strong>
            Data is infrastructure. The memo must bind the selected architecture to one measured
            data constraint and one residual risk.</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Decision Record</h2>
          <div class="mlsysbook-grid">
            {v1_04_fields_html({
                "Track": v1_04_pipeline_profile.label,
                "Architecture": v1_04_architecture.memo_summary,
                "Binding data constraint": v1_04_binding_result["label"],
                "Quality budget": "pass" if v1_04_quality_result["pass"] else "fail",
                "Split integrity": "valid" if v1_04_split_result["valid"] else "invalid",
                "Throughput": "feasible" if v1_04_throughput_result["feasible"] else "backlog",
                "Contract": "pass" if v1_04_contract_result["pass"] else "debt risk",
                "Residual risk seed": v1_04_contract_result["track_residual_risk"],
            })}
          </div>
          <div class="mlsysbook-callout"><strong>Binding constraint mitigation:</strong> {v1_04_binding_result["mitigation"]}</div>
        </div>
        """),
        v1_04_final_stance,
        v1_04_residual_risk_note,
        MathPeek(
            "valid_pipeline = quality_budget and split_integrity and throughput_capacity and contract_enforcement",
            {
                "quality_budget": "Residual defects are below target without erasing required coverage.",
                "split_integrity": "Evaluation evidence respects entity, time, and point-in-time boundaries.",
                "throughput_capacity": "Service rate exceeds arrival rate inside the freshness SLO.",
                "contract_enforcement": "Downstream consumers can rely on schema, semantics, freshness, and lineage.",
            },
        ),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">04 &middot; Data Engineering</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v1_04_profile.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">BINDING</span>
            <span class="hud-value">{v1_04_binding_result["label"]}</span>
            <span class="hud-label">STATUS</span>
            <span class="hud-active">ACTIVE</span>
        </div>
        """),
    ])

    def build_synthesis():
        return _synthesis

    _tabs = mo.ui.tabs({
        "Part A: Quality Budget": _part_a,
        "Part B: Split Integrity": _part_b,
        "Part C: Throughput": _part_c,
        "Part D: Contracts": _part_d,
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


# ===========================================================================
# ZONE D: LEDGER AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    ledger,
    mo,
    v1_04_architecture,
    v1_04_binding_result,
    v1_04_contract_checkpoint,
    v1_04_contract_prediction,
    v1_04_contract_result,
    v1_04_final_stance,
    v1_04_movement_result,
    v1_04_pipeline_profile,
    v1_04_profile,
    v1_04_quality_checkpoint,
    v1_04_quality_prediction,
    v1_04_quality_result,
    v1_04_residual_risk_note,
    v1_04_split_checkpoint,
    v1_04_split_prediction,
    v1_04_split_result,
    v1_04_throughput_checkpoint,
    v1_04_throughput_prediction,
    v1_04_throughput_result,
    v1_04_variant,
):
    _risk_note = str(v1_04_residual_risk_note.value or "").strip()
    _ready_for_ledger = all(
        value is not None
        for value in (
            v1_04_quality_prediction.value,
            v1_04_split_prediction.value,
            v1_04_throughput_prediction.value,
            v1_04_contract_prediction.value,
            v1_04_quality_checkpoint.value,
            v1_04_split_checkpoint.value,
            v1_04_throughput_checkpoint.value,
            v1_04_contract_checkpoint.value,
            v1_04_final_stance.value,
        )
    ) and bool(_risk_note)

    if _ready_for_ledger:
        ledger.save(chapter=4, design={
            "chapter": "v1_04",
            "track_id": v1_04_profile.track_id,
            "scenario_id": v1_04_variant.scenario_id,
            "hardware_ref": v1_04_pipeline_profile.hardware_ref,
            "model_ref": v1_04_pipeline_profile.model_ref,
            "completed": True,
            "quality_prediction": v1_04_quality_prediction.value,
            "quality_budget_pass": v1_04_quality_result["pass"],
            "residual_defects_per_10k": v1_04_quality_result["residual_defects_per_10k"],
            "quality_checkpoint": v1_04_quality_checkpoint.value,
            "split_prediction": v1_04_split_prediction.value,
            "effective_leakage_pct": v1_04_split_result["effective_leakage_pct"],
            "adjusted_metric_pct": v1_04_split_result["adjusted_metric_pct"],
            "split_valid": v1_04_split_result["valid"],
            "split_checkpoint": v1_04_split_checkpoint.value,
            "throughput_prediction": v1_04_throughput_prediction.value,
            "actual_bottleneck": v1_04_throughput_result["bottleneck_stage"],
            "utilization_pct": v1_04_throughput_result["utilization_pct"],
            "backlog_gb": v1_04_throughput_result["backlog_gb"],
            "freshness_lag_s": v1_04_throughput_result["freshness_lag_s"],
            "throughput_checkpoint": v1_04_throughput_checkpoint.value,
            "movement_strategy": v1_04_movement_result.strategy_id,
            "retention_policy": v1_04_architecture.retention_policy,
            "contract_prediction": v1_04_contract_prediction.value,
            "contract_policy": v1_04_contract_result["policy_label"],
            "silent_debt_index": v1_04_contract_result["silent_debt_index"],
            "contract_checkpoint": v1_04_contract_checkpoint.value,
            "binding_data_constraint": v1_04_binding_result["label"],
            "final_pipeline_stance": v1_04_final_stance.value,
            "residual_risk": _risk_note,
        })

    mo.Html(f"""
    <div class="mlsysbook-panel">
      <h2>Design Ledger</h2>
      <div class="mlsysbook-grid">
        <div class="mlsysbook-field"><strong>Ready to save</strong>{'yes' if _ready_for_ledger else 'not yet'}</div>
        <div class="mlsysbook-field"><strong>Binding constraint</strong>{v1_04_binding_result["label"]}</div>
        <div class="mlsysbook-field"><strong>Final stance</strong>{v1_04_final_stance.value or 'not recorded'}</div>
        <div class="mlsysbook-field"><strong>Residual risk</strong>{_risk_note or 'not recorded'}</div>
      </div>
      <div style="margin-top:10px; color:{COLORS['TextSec']}; line-height:1.55;">
        The ledger saves only after all four predictions, all four checkpoints, a final stance,
        and a residual risk are recorded.
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    report_export_panel,
    v1_04_architecture,
    v1_04_binding_result,
    v1_04_contract_checkpoint,
    v1_04_contract_prediction,
    v1_04_contract_result,
    v1_04_final_stance,
    v1_04_metadata,
    v1_04_movement_result,
    v1_04_pipeline_profile,
    v1_04_profile,
    v1_04_quality_checkpoint,
    v1_04_quality_prediction,
    v1_04_quality_result,
    v1_04_residual_risk_note,
    v1_04_snapshot,
    v1_04_split_checkpoint,
    v1_04_split_prediction,
    v1_04_split_result,
    v1_04_throughput_checkpoint,
    v1_04_throughput_prediction,
    v1_04_throughput_result,
    v1_04_variant,
):
    _incomplete = []
    _required = (
        ("Part A quality prediction", v1_04_quality_prediction.value),
        ("Part B split prediction", v1_04_split_prediction.value),
        ("Part C throughput prediction", v1_04_throughput_prediction.value),
        ("Part D contract prediction", v1_04_contract_prediction.value),
        ("Part A checkpoint", v1_04_quality_checkpoint.value),
        ("Part B checkpoint", v1_04_split_checkpoint.value),
        ("Part C checkpoint", v1_04_throughput_checkpoint.value),
        ("Part D checkpoint", v1_04_contract_checkpoint.value),
        ("Synthesis final stance", v1_04_final_stance.value),
    )
    for _label, _value in _required:
        if _value is None:
            _incomplete.append(_label)
    _risk_note = str(v1_04_residual_risk_note.value or "").strip()
    if not _risk_note:
        _incomplete.append("Synthesis residual risk")

    _snapshot = v1_04_snapshot(
        v1_04_quality_result,
        v1_04_split_result,
        v1_04_throughput_result,
        v1_04_contract_result,
    )

    _report = build_lab_report(
        v1_04_metadata,
        track=v1_04_profile.label,
        scenario=v1_04_variant.workload_summary,
        learning_objectives=(
            "Treat data quality as a measurable budget with defect, coverage, and review terms.",
            "Diagnose when leakage and split integrity invalidate evaluation evidence.",
            "Quantify pipeline throughput, backlog, and freshness as physical constraints.",
            "Choose data contracts, lineage, movement, and retention policies that limit downstream data debt.",
        ),
        predictions={
            "quality_budget": v1_04_quality_prediction.value,
            "split_integrity": v1_04_split_prediction.value,
            "throughput_backlog": v1_04_throughput_prediction.value,
            "data_contract": v1_04_contract_prediction.value,
        },
        knob_settings={
            "movement_strategy": v1_04_movement_result.strategy_id,
            "retention_policy": v1_04_architecture.retention_policy,
            "contract_policy": v1_04_contract_result["policy_label"],
        },
        binding_constraints={
            "binding_data_constraint": v1_04_binding_result["label"],
            "quality_budget_pass": v1_04_quality_result["pass"],
            "split_valid": v1_04_split_result["valid"],
            "throughput_feasible": v1_04_throughput_result["feasible"],
            "contract_pass": v1_04_contract_result["pass"],
        },
        decisions={
            "quality_checkpoint": v1_04_quality_checkpoint.value,
            "split_checkpoint": v1_04_split_checkpoint.value,
            "throughput_checkpoint": v1_04_throughput_checkpoint.value,
            "contract_checkpoint": v1_04_contract_checkpoint.value,
            "final_stance": v1_04_final_stance.value,
        },
        evidence_summary={
            "hardware_ref": v1_04_pipeline_profile.hardware_ref,
            "model_ref": v1_04_pipeline_profile.model_ref,
            "data_source": v1_04_pipeline_profile.data_source,
            "quality_residual_defects_per_10k": v1_04_quality_result["residual_defects_per_10k"],
            "quality_target_defects_per_10k": v1_04_quality_result["target_defects_per_10k"],
            "effective_leakage_pct": v1_04_split_result["effective_leakage_pct"],
            "adjusted_metric_pct": v1_04_split_result["adjusted_metric_pct"],
            "pipeline_bottleneck": v1_04_throughput_result["bottleneck_stage"],
            "backlog_gb": v1_04_throughput_result["backlog_gb"],
            "freshness_lag_s": v1_04_throughput_result["freshness_lag_s"],
            "data_moved_gb": v1_04_movement_result.data_moved_gb,
            "quality_retained_pct": v1_04_movement_result.quality_retained_pct,
            "silent_debt_index": v1_04_contract_result["silent_debt_index"],
        },
        final_decision={
            "architecture": v1_04_architecture.memo_summary,
            "binding_data_constraint": v1_04_binding_result["label"],
            "final_stance": v1_04_final_stance.value,
        },
        big_takeaways=(
            "Data quality is a budget with measurable defect and coverage terms.",
            "Split leakage can make high metrics invalid evidence.",
            "Pipeline throughput, backlog, and freshness are physical system constraints.",
            "Contracts and lineage prevent unmanaged data debt from propagating downstream.",
        ),
        reflections={
            "residual_risk": _risk_note,
            "privacy_stance": v1_04_pipeline_profile.privacy_stance,
            "report_artifact": v1_04_contract_result["report_artifact"],
        },
        residual_risk=_risk_note,
        source_trace={
            "track_id": v1_04_profile.track_id,
            "scenario_id": v1_04_variant.scenario_id,
            "hardware_ref": v1_04_variant.hardware_ref,
            "model_ref": v1_04_variant.model_ref,
            "shared_helper": "mlsysbook_labs.data_pipeline",
            "local_helpers": (
                "v1_04_quality_budget",
                "v1_04_split_integrity",
                "v1_04_backlog_model",
                "v1_04_contract_governance",
            ),
        },
        result_snapshot={
            "pipeline_profile": v1_04_pipeline_profile,
            "architecture": v1_04_architecture,
            "movement_frontier": v1_04_movement_result,
            "concept_modules": _snapshot,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-04 data pipeline memo is generated locally from the selected track, "
                "your structured predictions, manipulations, evidence, and synthesis decision."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
