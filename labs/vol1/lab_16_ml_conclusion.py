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
    import mlsysim
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        architecture_memo,
        build_lab_report,
        capstone_track_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        replay_ledger,
        report_export_panel,
        resolve_mlsysim_ref,
        sensitivity_audit,
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
        architecture_memo,
        build_lab_report,
        capstone_track_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        mlsysim,
        mo,
        replay_ledger,
        report_export_panel,
        resolve_mlsysim_ref,
        sensitivity_audit,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_16_metadata = get_lab_metadata("vol1/lab_16_ml_conclusion.py")
    return (v1_16_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_16_track_picker = track_selector(default=_default_track)
    v1_16_track_picker
    return (v1_16_track_picker,)


@app.cell
def _(
    capstone_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_16_track_picker,
):
    v1_16_track_id = v1_16_track_picker.value
    v1_16_profile = get_track_profile(v1_16_track_id)
    v1_16_variant = get_lab_track_variant("v1_16_architects_audit", v1_16_profile.track_id)
    v1_16_hardware = resolve_mlsysim_ref(v1_16_variant.hardware_ref)
    v1_16_model = resolve_mlsysim_ref(v1_16_variant.model_ref)
    v1_16_capstone = capstone_track_profile(
        v1_16_profile,
        v1_16_variant,
        v1_16_hardware,
        v1_16_model,
    )
    return (
        v1_16_capstone,
        v1_16_hardware,
        v1_16_model,
        v1_16_profile,
        v1_16_track_id,
        v1_16_variant,
    )


@app.cell
def _():
    def v1_16_track_lens(track_id):
        lenses = {
            "iphone": {
                "persona": "Mobile systems architect",
                "decision": "ship a local feature only if UX, privacy, battery, and thermal limits survive sustained use",
                "amounts": {
                    "Data": {
                        "track_term": "privacy-safe cohort evidence",
                        "base": 30.0,
                        "keywords": ("privacy", "cohort", "accessibility", "telemetry", "drift"),
                        "gap_weight": 0.10,
                    },
                    "Algorithm": {
                        "track_term": "MobileNetV2 operators and quantized quality",
                        "base": 42.0,
                        "keywords": ("model", "compression", "int8", "accuracy", "operator"),
                        "gap_weight": 0.05,
                    },
                    "Machine": {
                        "track_term": "battery, thermal, memory, and accelerator coverage",
                        "base": 72.0,
                        "keywords": ("thermal", "battery", "hardware", "accelerator", "npu", "latency", "memory"),
                        "gap_weight": 0.08,
                    },
                    "System": {
                        "track_term": "mobile UX, rollout, monitoring, and accessibility",
                        "base": 68.0,
                        "keywords": ("ux", "p99", "serving", "monitoring", "rollback", "responsible"),
                        "gap_weight": 0.24,
                    },
                },
                "debt_boundary": 32.0,
                "volume_ii_question": "What changes when privacy-safe mobile evidence must coordinate across a fleet of heterogeneous devices?",
            },
            "oura_ring": {
                "persona": "Wearable firmware architect",
                "decision": "ship firmware only if SRAM, flash, duty cycle, privacy, and sensing quality all fit",
                "amounts": {
                    "Data": {
                        "track_term": "sensor-contact and delayed-label evidence",
                        "base": 35.0,
                        "keywords": ("sensor", "contact", "label", "cohort", "false"),
                        "gap_weight": 0.10,
                    },
                    "Algorithm": {
                        "track_term": "DS-CNN width, INT8 path, and alert quality",
                        "base": 45.0,
                        "keywords": ("model", "int8", "compression", "channels", "alert"),
                        "gap_weight": 0.05,
                    },
                    "Machine": {
                        "track_term": "SRAM, flash, duty cycle, OTA, and radio wake",
                        "base": 84.0,
                        "keywords": ("sram", "flash", "battery", "duty", "ota", "radio", "memory"),
                        "gap_weight": 0.08,
                    },
                    "System": {
                        "track_term": "firmware update, phone relay, privacy, and false-alarm review",
                        "base": 62.0,
                        "keywords": ("serving", "monitor", "phone", "sync", "privacy", "health"),
                        "gap_weight": 0.22,
                    },
                },
                "debt_boundary": 26.0,
                "volume_ii_question": "What changes when many tiny devices must coordinate updates, labels, and recovery through a fleet service?",
            },
            "robotaxi": {
                "persona": "Safety/perception architect",
                "decision": "ship perception only if rare-event evidence, p99/p999 latency, power, and fallback safety pass",
                "amounts": {
                    "Data": {
                        "track_term": "rare-event replay and safety trace coverage",
                        "base": 63.0,
                        "keywords": ("rare", "replay", "safety", "cohort", "weather", "construction"),
                        "gap_weight": 0.20,
                    },
                    "Algorithm": {
                        "track_term": "perception model path and recall after optimization",
                        "base": 52.0,
                        "keywords": ("model", "compression", "recall", "perception", "distill"),
                        "gap_weight": 0.06,
                    },
                    "Machine": {
                        "track_term": "vehicle accelerator, sensor burst, power, and memory margin",
                        "base": 61.0,
                        "keywords": ("hardware", "power", "sensor", "accelerator", "latency", "memory"),
                        "gap_weight": 0.08,
                    },
                    "System": {
                        "track_term": "p99/p999 deadline, fallback, rollback, and safety case",
                        "base": 86.0,
                        "keywords": ("p99", "p999", "fallback", "rollback", "safety", "field", "tail"),
                        "gap_weight": 0.24,
                    },
                },
                "debt_boundary": 22.0,
                "volume_ii_question": "What changes when rare-event evidence and fallback reliability must be coordinated across a city-scale vehicle fleet?",
            },
            "cloud_fleet": {
                "persona": "Cloud fleet systems architect",
                "decision": "ship a service only if p99 SLA, cost/request, utilization, quality, operations, and carbon pass",
                "amounts": {
                    "Data": {
                        "track_term": "quality canary, drift labels, and governance evidence",
                        "base": 44.0,
                        "keywords": ("quality", "drift", "canary", "cohort", "governance"),
                        "gap_weight": 0.12,
                    },
                    "Algorithm": {
                        "track_term": "context length, quantization, and model quality",
                        "base": 49.0,
                        "keywords": ("model", "compression", "context", "quality", "quantize"),
                        "gap_weight": 0.06,
                    },
                    "Machine": {
                        "track_term": "H100 memory bandwidth, KV cache, replicas, and utilization",
                        "base": 70.0,
                        "keywords": ("hardware", "kv", "memory", "replica", "utilization", "gpu"),
                        "gap_weight": 0.10,
                    },
                    "System": {
                        "track_term": "p99 SLA, queueing, cost/request, rollback, and carbon",
                        "base": 86.0,
                        "keywords": ("p99", "sla", "cost", "carbon", "serving", "rollback", "ops"),
                        "gap_weight": 0.22,
                    },
                },
                "debt_boundary": 34.0,
                "volume_ii_question": "What changes when SLA, cost, carbon, and failure are governed by a distributed fleet rather than one service node?",
            },
        }
        return lenses.get(track_id, lenses["cloud_fleet"])

    def v1_16_amount_scores(v1_16_capstone, v1_16_ledger_result, evidence_floor_pct):
        lens = v1_16_track_lens(v1_16_capstone.track_id)
        coverage_gap = max(0.0, float(evidence_floor_pct) - v1_16_ledger_result.coverage_pct)
        rows = []
        for amount, spec in lens["amounts"].items():
            keywords = tuple(str(keyword).lower() for keyword in spec["keywords"])
            matching = []
            preset_hits = 0
            for decision in v1_16_ledger_result.decisions:
                text = " ".join((
                    str(decision.label),
                    str(decision.constraint),
                    str(decision.decision),
                    str(decision.source),
                )).lower()
                if any(keyword in text for keyword in keywords):
                    matching.append(decision.label)
                    if decision.source == "track preset":
                        preset_hits += 1
            score = min(
                100.0,
                float(spec["base"])
                + 4.0 * len(matching)
                + 2.0 * preset_hits
                + coverage_gap * float(spec["gap_weight"]),
            )
            rows.append({
                "amount": amount,
                "track_term": spec["track_term"],
                "score": round(score, 1),
                "evidence": ", ".join(matching[:3]) if matching else "track pressure only",
                "hits": len(matching),
                "preset_hits": preset_hits,
            })

        max_score = max(row["score"] for row in rows) if rows else 0.0
        for row in rows:
            if row["score"] == max_score:
                row["status"] = "BINDING"
            elif row["score"] >= max_score - 8.0:
                row["status"] = "WATCH"
            else:
                row["status"] = "CLEAR"
        return tuple(rows)

    def v1_16_binding_amount(amount_rows):
        if not amount_rows:
            return {"amount": "System", "score": 0.0, "track_term": "no evidence", "status": "BINDING"}
        return max(amount_rows, key=lambda row: row["score"])

    def v1_16_lever_catalog(track_id):
        catalogs = {
            "iphone": (
                {"id": "accelerator_path", "label": "Force supported NPU fast path", "targets": ("Machine",), "relief": 0.82, "base_debt": 8.0, "debt": 0.34, "debt_note": "operator coverage can narrow model choices and require thermal soak"},
                {"id": "distill_model", "label": "Distill or quantize the local model", "targets": ("Algorithm", "Machine"), "relief": 0.70, "base_debt": 11.0, "debt": 0.44, "debt_note": "quality and accessibility cohorts need new validation"},
                {"id": "privacy_telemetry", "label": "Add privacy-safe cohort telemetry", "targets": ("Data", "System"), "relief": 0.62, "base_debt": 7.0, "debt": 0.28, "debt_note": "small cohorts may still be under-sampled"},
                {"id": "selective_offload", "label": "Offload hard cases selectively", "targets": ("System",), "relief": 0.60, "base_debt": 15.0, "debt": 0.55, "debt_note": "privacy, latency, and connectivity debt move downstream"},
            ),
            "oura_ring": (
                {"id": "sampling_cadence", "label": "Reduce sensor sampling cadence", "targets": ("Machine",), "relief": 0.82, "base_debt": 9.0, "debt": 0.42, "debt_note": "signal quality and missed-event evidence become the debt"},
                {"id": "prune_channels", "label": "Prune DS-CNN channels", "targets": ("Algorithm", "Machine"), "relief": 0.66, "base_debt": 12.0, "debt": 0.46, "debt_note": "false-alarm validation must be repeated"},
                {"id": "ota_headroom", "label": "Reserve OTA flash headroom", "targets": ("Machine", "System"), "relief": 0.72, "base_debt": 8.0, "debt": 0.30, "debt_note": "feature scope shrinks but update safety improves"},
                {"id": "phone_relay", "label": "Move summaries through phone relay", "targets": ("System",), "relief": 0.58, "base_debt": 13.0, "debt": 0.48, "debt_note": "privacy and connectivity assumptions move to the phone"},
            ),
            "robotaxi": (
                {"id": "fallback_path", "label": "Add conservative fallback path", "targets": ("System",), "relief": 0.78, "base_debt": 10.0, "debt": 0.38, "debt_note": "fallback validation and power margin become required evidence"},
                {"id": "rare_replay", "label": "Expand rare-event replay coverage", "targets": ("Data", "System"), "relief": 0.70, "base_debt": 8.0, "debt": 0.30, "debt_note": "release pace slows while safety evidence improves"},
                {"id": "distill_perception", "label": "Distill perception path", "targets": ("Algorithm", "Machine"), "relief": 0.64, "base_debt": 16.0, "debt": 0.55, "debt_note": "rare-event recall can regress after optimization"},
                {"id": "tight_p99_gate", "label": "Tighten p99/p999 replay gate", "targets": ("System",), "relief": 0.62, "base_debt": 9.0, "debt": 0.34, "debt_note": "more scenarios fail until model and fallback margin improve"},
            ),
            "cloud_fleet": (
                {"id": "warm_pool", "label": "Increase warm-pool capacity", "targets": ("Machine", "System"), "relief": 0.78, "base_debt": 13.0, "debt": 0.44, "debt_note": "cost and carbon debt grow with idle capacity"},
                {"id": "context_limit", "label": "Reduce maximum context length", "targets": ("Algorithm", "Machine"), "relief": 0.68, "base_debt": 10.0, "debt": 0.36, "debt_note": "quality and user-visible capability may degrade"},
                {"id": "cost_carbon_canary", "label": "Add cost and carbon canary", "targets": ("Data", "System"), "relief": 0.62, "base_debt": 7.0, "debt": 0.28, "debt_note": "operations complexity increases but evidence improves"},
                {"id": "batching_policy", "label": "Tune continuous batching policy", "targets": ("System",), "relief": 0.66, "base_debt": 12.0, "debt": 0.46, "debt_note": "throughput gains can increase p99 tail debt"},
            ),
        }
        return catalogs.get(track_id, catalogs["cloud_fleet"])

    def v1_16_lever_options(track_id):
        return {lever["label"]: lever["label"] for lever in v1_16_lever_catalog(track_id)}

    def v1_16_default_lever(track_id):
        return v1_16_lever_catalog(track_id)[0]["label"]

    def v1_16_lever_audit(track_id, binding_amount, selected_lever, intensity_pct):
        catalog = v1_16_lever_catalog(track_id)
        lever_by_id = {lever["id"]: lever for lever in catalog}
        lever_by_label = {lever["label"]: lever for lever in catalog}
        lever = lever_by_id.get(selected_lever) or lever_by_label.get(selected_lever) or catalog[0]
        lens = v1_16_track_lens(track_id)
        intensity = max(0.0, min(100.0, float(intensity_pct)))
        alignment = 1.0 if binding_amount in lever["targets"] else 0.45
        relief = min(100.0, intensity * float(lever["relief"]) * alignment)
        debt = min(100.0, float(lever["base_debt"]) + intensity * float(lever["debt"]) * (1.2 if alignment < 1.0 else 0.72))
        boundary = float(lens["debt_boundary"])
        net_margin = relief - debt
        status = "PASS" if debt <= boundary and net_margin >= 0.0 and alignment >= 0.9 else "FAIL"
        return {
            "lever_id": lever["id"],
            "lever_label": lever["label"],
            "targets": ", ".join(lever["targets"]),
            "binding_amount": binding_amount,
            "alignment": alignment,
            "relief_pct": round(relief, 1),
            "debt_pct": round(debt, 1),
            "debt_boundary_pct": round(boundary, 1),
            "net_margin_pct": round(net_margin, 1),
            "status": status,
            "debt_note": lever["debt_note"],
        }

    def v1_16_report_audit(v1_16_ledger_result, v1_16_audit, v1_16_memo, rejected_alternative, v1_16_capstone):
        rejected_ok = bool(rejected_alternative) and rejected_alternative != v1_16_memo.revised_decision
        rows = (
            ("Decision", bool(v1_16_memo.revised_decision), v1_16_memo.revised_decision),
            ("Rejected alternative", rejected_ok, rejected_alternative or "missing"),
            ("Ledger replay", v1_16_ledger_result.entries_expected > 0, f"{v1_16_ledger_result.entries_found}/{v1_16_ledger_result.entries_expected} student entries"),
            ("Operating envelope", v1_16_audit.feasible, "PASS" if v1_16_audit.feasible else ", ".join(v1_16_audit.violations)),
            ("Residual risk", bool(v1_16_memo.top_risk), v1_16_memo.top_risk),
            ("Validation evidence", bool(v1_16_memo.mitigation), v1_16_memo.mitigation),
            ("Source trace", bool(v1_16_capstone.source_refs), ", ".join(v1_16_capstone.source_refs)),
        )
        passed = sum(1 for _, ok, _ in rows if ok)
        completeness = passed / len(rows) * 100.0
        status = "PASS" if completeness >= 85.0 and rejected_ok and bool(v1_16_memo.top_risk) else "FAIL"
        return {
            "rows": rows,
            "passed": passed,
            "total": len(rows),
            "completeness_pct": round(completeness, 1),
            "status": status,
        }

    def v1_16_volume_ii_question(track_id):
        return v1_16_track_lens(track_id)["volume_ii_question"]

    return (
        v1_16_amount_scores,
        v1_16_binding_amount,
        v1_16_default_lever,
        v1_16_lever_audit,
        v1_16_lever_catalog,
        v1_16_lever_options,
        v1_16_report_audit,
        v1_16_track_lens,
        v1_16_volume_ii_question,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    track_arc_context,
    v1_16_capstone,
    v1_16_metadata,
    v1_16_profile,
    v1_16_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 16 Capstone
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The Architect's Audit
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Binding Diagnosis &middot; Right Lever &middot; Operating Envelope &middot; Final Report
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 780px; line-height: 1.65;">
                {v1_16_variant.workload_summary} The final artifact is not a
                summary; it is an audit of the architecture implied by earlier decisions.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Memo &middot; ~55 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_16_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_16_capstone.hardware_ref}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Binding Amount</span>
                <span class="badge badge-warn">Downstream Debt</span>
                <span class="badge badge-fail">Residual Risk</span>
            </div>
        </div>
        """),
        track_context(v1_16_profile),
        track_arc_context(v1_16_profile, v1_16_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_16_capstone):
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
                <div style="margin-bottom: 3px;">1. <strong>Diagnose the binding amount:</strong>
                    use ledger evidence to identify the D-A-M/system quantity that constrains the design.</div>
                <div style="margin-bottom: 3px;">2. <strong>Optimize the right lever:</strong>
                    compare binding relief against downstream debt.</div>
                <div style="margin-bottom: 3px;">3. <strong>Deploy inside the envelope:</strong>
                    perturb workload, model, guardrails, and evidence confidence.</div>
                <div style="margin-bottom: 3px;">4. <strong>Defend the report:</strong>
                    revise a decision, reject an alternative, and name residual risk.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 260px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Architecture Goal
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    {v1_16_capstone.architecture_goal}
                </div>
            </div>
            <div style="flex: 0 0 260px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Durable Principle
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    {v1_16_capstone.durable_principle}
                </div>
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** - This is the Volume I capstone. Review your ledger
    entries from Labs 00-15 and the design memos you downloaded from recent labs.
    """), kind="info")
    return


# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_16_capstone):
    partA_pred = mo.ui.radio(
        options={
            "A) Data amount: evidence coverage and cohort trace": "Data",
            "B) Algorithm amount: model size, operators, and quality": "Algorithm",
            "C) Machine amount: memory, energy, thermal, or accelerator headroom": "Machine",
            "D) System amount: p99, monitoring, rollout, responsibility, and operations": "System",
        },
        label=f"{v1_16_capstone.label}: which D-A-M/system amount will bind the final architecture?",
    )
    partA_evidence_floor = mo.ui.slider(
        start=50,
        stop=95,
        value=70,
        step=5,
        label="Evidence floor for report confidence (%)",
    )
    partA_checkpoint = mo.ui.radio(
        options={
            "A) Optimize the binding amount first and label evidence gaps": "binding_first",
            "B) Patch the easiest subsystem first": "easy_patch",
            "C) Ignore preset entries because they are not student evidence": "ignore_presets",
            "D) Make a model-only fix": "model_only",
        },
        label="Part A checkpoint: what should the architecture audit do first?",
    )
    return (partA_checkpoint, partA_evidence_floor, partA_pred)


@app.cell(hide_code=True)
def _(mo, v1_16_capstone, v1_16_default_lever, v1_16_lever_options):
    partB_pred = mo.ui.radio(
        options={
            "A) Pull the lever that relieves the binding amount and check downstream debt": "binding_debt",
            "B) Optimize the easiest local metric because any improvement helps": "local_metric",
            "C) Always shrink the model first": "model_first",
            "D) Add deployment guardrails even if the bottleneck is machine capacity": "guardrail_first",
        },
        label=f"What makes an optimization defensible for {v1_16_capstone.label}?",
    )
    partB_lever = mo.ui.dropdown(
        options=v1_16_lever_options(v1_16_capstone.track_id),
        value=v1_16_default_lever(v1_16_capstone.track_id),
        label="Optimization lever to test",
    )
    partB_intensity = mo.ui.slider(
        start=10,
        stop=80,
        value=45,
        step=5,
        label="Intervention intensity (%)",
    )
    partB_checkpoint = mo.ui.radio(
        options={
            "A) Accept only if binding relief exceeds downstream debt": "accept_if_margin",
            "B) Accept any lever with a local metric gain": "accept_local",
            "C) Reject all changes and keep the original architecture": "reject_all",
            "D) Defer the debt check to production": "defer_debt",
        },
        label="Part B checkpoint: what lever policy belongs in the memo?",
    )
    return (partB_checkpoint, partB_intensity, partB_lever, partB_pred)


@app.cell(hide_code=True)
def _(mo, v1_16_capstone):
    _defaults = v1_16_capstone.sensitivity_defaults
    partC_pred = mo.ui.radio(
        options={
            "A) Workload growth is always the most fragile assumption": "workload",
            "B) The most fragile axis depends on track-specific headroom": "track_specific",
            "C) Evidence confidence is not a systems constraint": "evidence_free",
            "D) Tightening guardrails always improves feasibility": "tighten",
        },
        label=f"Which sensitivity axis is most dangerous for {v1_16_capstone.label}?",
    )
    partC_workload = mo.ui.slider(
        start=0.5,
        stop=4.0,
        value=min(4.0, max(0.5, float(_defaults.get("workload_limit_multiplier", 2.0)))),
        step=0.1,
        label="Workload multiplier",
    )
    partC_model_growth = mo.ui.slider(
        start=0,
        stop=80,
        value=int(float(_defaults.get("model_growth_limit_pct", 30.0))),
        step=5,
        label="Model growth (%)",
    )
    return (partC_model_growth, partC_pred, partC_workload)


@app.cell(hide_code=True)
def _(mo, v1_16_capstone):
    _defaults = v1_16_capstone.sensitivity_defaults
    partC_guardrail = mo.ui.slider(
        start=0,
        stop=60,
        value=int(float(_defaults.get("guardrail_tightening_limit_pct", 25.0))),
        step=5,
        label="Guardrail tightening (%)",
    )
    partC_evidence = mo.ui.slider(
        start=0,
        stop=100,
        value=int(float(_defaults.get("evidence_confidence_floor_pct", 75.0))),
        step=5,
        label="Evidence confidence (%)",
    )
    partC_checkpoint = mo.ui.radio(
        options={
            "A) Ship only if all stress axes pass and validation evidence is named": "inside_envelope",
            "B) Ship if the average risk score looks acceptable": "average_ok",
            "C) Ship now and monitor later": "monitor_later",
            "D) Reject the design without trying a revision": "reject_without_revision",
        },
        label="Part C checkpoint: what release rule follows from the envelope evidence?",
    )

    partD_pred = mo.ui.radio(
        options={
            "A) Revise a decision, reject an alternative, and name validation evidence": "revise",
            "B) Keep every prior decision unchanged": "unchanged",
            "C) Remove residual risk from the memo": "remove_risk",
            "D) Optimize the easiest component rather than the binding one": "easy",
        },
        label="What should the final architecture memo do?",
    )
    return (partC_checkpoint, partC_evidence, partC_guardrail, partD_pred)


@app.cell(hide_code=True)
def _(mo, v1_16_capstone):
    _revision_options = {item: item for item in v1_16_capstone.revision_options}
    _risk_options = {item: item for item in v1_16_capstone.top_risks}
    _mitigations = tuple(
        str(value)
        for key, value in v1_16_capstone.sensitivity_defaults.items()
        if key.endswith("_mitigation")
    )
    _mitigation_options = {item: item for item in _mitigations}
    partD_revision = mo.ui.dropdown(
        options=_revision_options,
        value=v1_16_capstone.revision_options[0],
        label="Decision to revise",
    )
    _rejected_default = (
        v1_16_capstone.revision_options[1]
        if len(v1_16_capstone.revision_options) > 1
        else v1_16_capstone.revision_options[0]
    )
    partD_rejected = mo.ui.dropdown(
        options=_revision_options,
        value=_rejected_default,
        label="Alternative to reject",
    )
    partD_top_risk = mo.ui.dropdown(
        options=_risk_options,
        value=v1_16_capstone.top_risks[0],
        label="Top residual risk",
    )
    partD_mitigation = mo.ui.dropdown(
        options=_mitigation_options,
        value=_mitigations[0],
        label="Mitigation evidence",
    )
    partD_checkpoint = mo.ui.radio(
        options={
            "A) Defensible: decision, rejected alternative, source trace, validation, and residual risk are present": "defensible",
            "B) Needs more validation evidence before release": "more_validation",
            "C) Needs a different optimization lever": "different_lever",
            "D) Not ready to ship": "not_ready",
        },
        label="Part D checkpoint: what is the report decision?",
    )
    return (partD_checkpoint, partD_mitigation, partD_rejected, partD_revision, partD_top_risk)


# ===========================================================================
# ZONE C: MAIN LAB
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    architecture_memo,
    go,
    ledger,
    mo,
    partA_checkpoint,
    partA_evidence_floor,
    partA_pred,
    partB_checkpoint,
    partB_intensity,
    partB_lever,
    partB_pred,
    partC_checkpoint,
    partC_evidence,
    partC_guardrail,
    partC_model_growth,
    partC_pred,
    partC_workload,
    partD_checkpoint,
    partD_mitigation,
    partD_pred,
    partD_rejected,
    partD_revision,
    partD_top_risk,
    replay_ledger,
    sensitivity_audit,
    source_trace,
    v1_16_amount_scores,
    v1_16_binding_amount,
    v1_16_capstone,
    v1_16_lever_audit,
    v1_16_profile,
    v1_16_report_audit,
    v1_16_track_lens,
    v1_16_variant,
    v1_16_volume_ii_question,
):
    def v1_16_metric_card(label, value, detail, color, border=False):
        border_style = f"2px solid {color}" if border else "1px solid #e2e8f0"
        return f"""
        <div style="padding:16px; border:{border_style}; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {color}; flex:1;">
            <div style="color:#64748b; font-size:0.78rem; font-weight:700;">{label}</div>
            <div style="font-size:1.45rem; font-weight:800; color:{color};">{value}</div>
            <div style="font-size:0.72rem; color:#64748b;">{detail}</div>
        </div>
        """

    def v1_16_status_color(status):
        return COLORS["GreenLine"] if status == "PASS" else COLORS["RedLine"]

    _ledger_result = replay_ledger(v1_16_capstone, ledger._state.history)
    _lens = v1_16_track_lens(v1_16_capstone.track_id)
    _amount_rows = v1_16_amount_scores(v1_16_capstone, _ledger_result, partA_evidence_floor.value)
    _binding = v1_16_binding_amount(_amount_rows)
    _lever = v1_16_lever_audit(
        v1_16_capstone.track_id,
        _binding["amount"],
        partB_lever.value,
        partB_intensity.value,
    )
    _audit = sensitivity_audit(
        v1_16_capstone,
        workload_multiplier=partC_workload.value,
        model_growth_pct=partC_model_growth.value,
        guardrail_tightening_pct=partC_guardrail.value,
        evidence_confidence_pct=partC_evidence.value,
    )
    _memo = architecture_memo(
        v1_16_capstone,
        revised_decision=partD_revision.value,
        top_risk=partD_top_risk.value,
        mitigation=partD_mitigation.value,
    )
    _report_audit = v1_16_report_audit(
        _ledger_result,
        _audit,
        _memo,
        partD_rejected.value,
        v1_16_capstone,
    )

    def build_part_a():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Concept Module A &middot; Diagnose the binding amount
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "{v1_16_variant.stakeholder}: use accumulated evidence to decide
                    which D-A-M/system amount is binding before you optimize anything."
                </div>
            </div>
            """),
            mo.md(f"""
## Part A: Diagnose The Binding Amount

**Scenario.** You are the {_lens["persona"]}. Your decision is to
{_lens["decision"]}. The first question is not "what model do we like?" It is
"which amount is binding after Volume I evidence is replayed?"
            """),
            partA_pred,
        ]
        if partA_pred.value is None:
            items.append(mo.callout(mo.md("Select a binding-amount prediction to unlock the ledger replay."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_evidence_floor], widths="equal"))
        _coverage_color = COLORS["GreenLine"] if _ledger_result.coverage_pct >= 70 else COLORS["OrangeLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_16_metric_card("Binding Amount", _binding["amount"], _binding["track_term"], COLORS["BlueLine"], True)}
            {v1_16_metric_card("Binding Score", f"{_binding['score']:.1f}%", "highest D-A-M/system score", COLORS["RedLine"], True)}
            {v1_16_metric_card("Ledger Coverage", f"{_ledger_result.coverage_pct:.0f}%", f"{_ledger_result.entries_found}/{_ledger_result.entries_expected} entries", _coverage_color)}
            {v1_16_metric_card("Missing Chapters", f"{len(_ledger_result.missing_chapters)}", "presets used", COLORS["OrangeLine"])}
        </div>
        """))

        _amount_color = [
            COLORS["RedLine"] if row["status"] == "BINDING"
            else COLORS["OrangeLine"] if row["status"] == "WATCH"
            else COLORS["BlueLine"]
            for row in _amount_rows
        ]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=[row["amount"] for row in _amount_rows],
            y=[row["score"] for row in _amount_rows],
            text=[row["status"] for row in _amount_rows],
            textposition="outside",
            marker_color=_amount_color,
        ))
        _fig.add_hline(
            y=partA_evidence_floor.value,
            line_dash="dash",
            line_color="#64748b",
            annotation_text="selected evidence floor",
        )
        _fig.update_layout(
            height=340,
            yaxis=dict(title="Binding pressure score (%)", gridcolor="#f1f5f9", range=[0, 110]),
            xaxis=dict(title="D-A-M/system amount"),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _amount_rows_md = "\n".join(
            f"| {row['amount']} | {row['track_term']} | {row['score']:.1f}% | {row['status']} | {row['evidence']} | {row['preset_hits']} |"
            for row in _amount_rows
        )
        items.append(mo.md(f"""
**Amount Evidence Table**

| Amount | Track realization | Score | Status | Ledger evidence | Preset hits |
|---|---|---:|---|---|---:|
{_amount_rows_md}
        """))

        _ledger_rows = "\n".join(
            f"| {decision.chapter} | {decision.label} | {decision.constraint} | {decision.source} | {decision.decision} |"
            for decision in _ledger_result.decisions
        )
        items.append(mo.md(f"""
**Ledger Replay Table**

| Lab | Evidence | Constraint | Source | Decision |
|---:|---|---|---|---|
{_ledger_rows}

*Source: `mlsysbook_labs.replay_ledger`; missing entries use typed V1-16 variants.*
        """))

        if _ledger_result.coverage_pct < partA_evidence_floor.value:
            items.append(mo.callout(mo.md(
                f"**Evidence boundary:** coverage is {_ledger_result.coverage_pct:.0f}% but your floor is "
                f"{partA_evidence_floor.value:.0f}%. The final report must label preset evidence debt."
            ), kind="danger"))
        elif _ledger_result.missing_chapters:
            items.append(mo.callout(mo.md(
                f"**Ledger gaps are visible.** Missing chapters: {', '.join(str(ch) for ch in _ledger_result.missing_chapters)}."
            ), kind="warn"))

        items.append(mo.accordion({
            "Math Peek / Source Model - binding amount score": mo.md("""
The local source model scores each amount as:

`score = track pressure + 4 * ledger hits + 2 * preset hits + evidence gap penalty`

The evidence gap penalty is active when ledger coverage falls below the selected
floor. This ties the conclusion's D-A-M and Lighthouse constraint-propagation
claims to the track-specific architecture audit.
            """)
        }))
        items.append(source_trace({
            "chapter_anchor": "Conclusion: Synthesizing ML Systems; Lighthouse models; Thirteen Quantitative Invariants",
            "claim": "A system design starts by finding the amount that actually binds.",
            "helper": "notebook-local v1_16_amount_scores",
            "shared_helper": "mlsysbook_labs.replay_ledger",
            "hardware_ref": v1_16_capstone.hardware_ref,
            "model_ref": v1_16_capstone.model_ref,
        }, summary="Math Peek/source trace: binding quantity diagnosis"))
        items.append(partA_checkpoint)

        if partA_pred.value == _binding["amount"]:
            items.append(mo.callout(mo.md("**Prediction check:** your predicted binding amount matches the ledger-weighted diagnosis."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Prediction check:** the evidence currently binds on **{_binding['amount']}**, not **{partA_pred.value}**. "
                "The point is to let accumulated evidence overrule the easy story."
            ), kind="warn"))
        if partA_checkpoint.value == "binding_first":
            items.append(mo.callout(mo.md("**Checkpoint recorded:** the memo will attack the binding amount first and label evidence gaps."), kind="success"))
        elif partA_checkpoint.value is not None:
            items.append(mo.callout(mo.md("**Checkpoint warning:** a capstone design should not optimize around the binding amount."), kind="warn"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Concept Module B &middot; Optimize the right lever
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "A local improvement only helps if it relieves the binding
                    amount without pushing unpayable debt downstream."
                </div>
            </div>
            """),
            mo.md(f"""
## Part B: Optimize The Right Lever

**Scenario.** The binding amount from Part A is **{_binding["amount"]}**
({_binding["track_term"]}). Choose a track-specific lever and intensity, then
inspect whether the local relief is worth the downstream debt.
            """),
            partB_pred,
        ]
        if partB_pred.value is None:
            items.append(mo.callout(mo.md("Select your optimization prediction to unlock the lever audit."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_lever, partB_intensity], widths="equal"))

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=[_lever["lever_label"]],
            y=[_lever["relief_pct"]],
            name="Binding relief",
            marker_color=COLORS["GreenLine"],
        ))
        _fig.add_trace(go.Bar(
            x=[_lever["lever_label"]],
            y=[_lever["debt_pct"]],
            name="Downstream debt",
            marker_color=COLORS["RedLine"] if _lever["status"] == "FAIL" else COLORS["OrangeLine"],
        ))
        _fig.add_hline(
            y=_lever["debt_boundary_pct"],
            line_dash="dash",
            line_color="#64748b",
            annotation_text="debt boundary",
        )
        _fig.update_layout(
            barmode="group",
            height=340,
            yaxis=dict(title="Percent of design margin", gridcolor="#f1f5f9", range=[0, 110]),
            margin=dict(l=60, r=20, t=40, b=80),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _lever_rows = "\n".join((
            f"| Selected lever | {_lever['lever_label']} |",
            f"| Target amount(s) | {_lever['targets']} |",
            f"| Binding amount | {_lever['binding_amount']} |",
            f"| Binding relief | {_lever['relief_pct']:.1f}% |",
            f"| Downstream debt | {_lever['debt_pct']:.1f}% |",
            f"| Debt boundary | {_lever['debt_boundary_pct']:.1f}% |",
            f"| Net margin | {_lever['net_margin_pct']:.1f}% |",
            f"| Status | {_lever['status']} |",
            f"| Debt explanation | {_lever['debt_note']} |",
        ))
        items.append(mo.md(f"""
**Lever Audit Table**

| Field | Value |
|---|---|
{_lever_rows}
        """))

        if _lever["status"] == "FAIL":
            items.append(mo.callout(mo.md(
                f"**Optimization boundary:** this lever produces {_lever['debt_pct']:.1f}% downstream debt "
                f"against a {_lever['debt_boundary_pct']:.1f}% boundary, or it misses the binding amount."
            ), kind="danger"))

        items.append(mo.accordion({
            "Math Peek / Source Model - local gain and downstream debt": mo.md("""
The local source model uses:

`net margin = binding relief - downstream debt`

Relief is discounted when the selected lever does not target the Part A binding
amount. Debt grows with intervention intensity because complexity is conserved:
a faster local path can become validation, monitoring, cost, or safety debt.
            """)
        }))
        items.append(source_trace({
            "chapter_anchor": "Conclusion: Integrated framework; Fallacies and Pitfalls",
            "claim": "Optimize the term that binds; local speedups can be capped by Amdahl and moved by the Pareto frontier.",
            "helper": "notebook-local v1_16_lever_audit",
            "binding_amount": _binding["amount"],
            "selected_lever": _lever["lever_label"],
        }, summary="Math Peek/source trace: right-lever optimization"))
        items.append(partB_checkpoint)

        if partB_pred.value == "binding_debt":
            items.append(mo.callout(mo.md("**Prediction check:** correct rule. A lever is defensible only after the debt check."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Prediction check:** local gains are not enough. The memo must connect the lever to the binding amount and downstream debt."
            ), kind="warn"))
        if partB_checkpoint.value == "accept_if_margin" and _lever["status"] == "PASS":
            items.append(mo.callout(mo.md("**Checkpoint recorded:** current lever policy is defensible under the debt boundary."), kind="success"))
        elif partB_checkpoint.value == "accept_if_margin":
            items.append(mo.callout(mo.md("**Checkpoint recorded:** the policy is right, but the current lever setting fails the boundary."), kind="warn"))
        elif partB_checkpoint.value is not None:
            items.append(mo.callout(mo.md("**Checkpoint warning:** the memo should not accept a lever without a debt boundary."), kind="warn"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Concept Module C &middot; Deploy inside the operating envelope
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "A deployment claim is valid only inside the envelope you can
                    validate for this track."
                </div>
            </div>
            """),
            mo.md(f"""
## Part C: Deploy Inside The Operating Envelope

**Scenario.** Stress the proposed design for **{v1_16_capstone.label}**.
The envelope is track-specific: {v1_16_capstone.guardrail_metric}.
            """),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the sensitivity audit."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_workload, partC_model_growth, partC_guardrail, partC_evidence], widths="equal"))
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=[axis.name for axis in _audit.axes],
            y=[axis.risk_pct for axis in _audit.axes],
            marker_color=[COLORS["GreenLine"] if axis.status == "PASS" else COLORS["RedLine"] for axis in _audit.axes],
            opacity=0.9,
        ))
        _fig.add_hline(y=100, line_dash="dash", line_color="#64748b", annotation_text="failure boundary")
        _fig.update_layout(
            height=340,
            yaxis=dict(title="Risk score (%)", gridcolor="#f1f5f9", range=[0, 110]),
            margin=dict(l=60, r=20, t=40, b=60),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _status_color = COLORS["GreenLine"] if _audit.feasible else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_16_metric_card("Most Fragile", _audit.most_fragile, "highest sensitivity score", COLORS["OrangeLine"], True)}
            {v1_16_metric_card("Fragility", f"{_audit.fragility_score_pct:.0f}%", "average risk score", COLORS["BlueLine"])}
            {v1_16_metric_card("Status", "PASS" if _audit.feasible else "FAIL", ", ".join(_audit.violations) or "no violations", _status_color, True)}
        </div>
        """))

        _axis_rows = "\n".join(
            f"| {axis.name} | {axis.value:.1f} | {axis.limit:.1f} | {axis.risk_pct:.0f}% | {axis.status} | {axis.mitigation} |"
            for axis in _audit.axes
        )
        items.append(mo.md(f"""
**Sensitivity Table**

| Axis | Value | Limit | Risk | Status | Mitigation |
|---|---:|---:|---:|---|---|
{_axis_rows}

*Source: `mlsysbook_labs.sensitivity_audit`.*
        """))

        if not _audit.feasible:
            items.append(mo.callout(mo.md(
                "**Sensitivity failure:** " + ", ".join(_audit.violations) + ". The memo must revise a decision or add validation evidence."
            ), kind="danger"))

        items.append(mo.accordion({
            "Math Peek / Source Model - operating envelope": mo.md("""
The deployment rule is:

`deployable = all(axis.value <= axis.limit) and evidence_confidence >= floor`

The exact axes come from the track-specific capstone profile and are evaluated
by the existing `mlsysbook_labs.sensitivity_audit` helper.
            """)
        }))
        items.append(source_trace({
            "chapter_anchor": "Conclusion: Deploy invariants; Production reality; Robust AI systems",
            "claim": "A design is only valid inside a measured operating envelope.",
            "shared_helper": "mlsysbook_labs.sensitivity_audit",
            "guardrail_metric": v1_16_capstone.guardrail_metric,
            "most_fragile": _audit.most_fragile,
        }, summary="Math Peek/source trace: operating-envelope validation"))
        items.append(partC_checkpoint)

        if partC_pred.value == "track_specific":
            items.append(mo.callout(mo.md("**Correct.** Fragility depends on the selected track and its headroom."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Fragility is not universal.** The same perturbation stresses iPhone, Oura Ring, RoboTaxi, and Cloud Fleet differently."
            ), kind="warn"))
        if partC_checkpoint.value == "inside_envelope":
            items.append(mo.callout(mo.md(
                f"**Checkpoint recorded:** release is tied to envelope status: {'PASS' if _audit.feasible else 'FAIL'}."
            ), kind="success" if _audit.feasible else "warn"))
        elif partC_checkpoint.value is not None:
            items.append(mo.callout(mo.md("**Checkpoint warning:** average risk or monitor-later logic is not an operating envelope."), kind="warn"))
        return mo.vstack(items)

    def build_part_d():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenLL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Concept Module D &middot; Defend the complete design report
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "A final report is not defensible until it names the chosen
                    design, the rejected alternative, source trace, validation evidence,
                    and residual risk."
                </div>
            </div>
            """),
            mo.md("""
## Part D: Defend The Design Report

**Scenario.** The architecture review board asks for a final Volume I design
report. It must defend one decision and explicitly reject at least one plausible
alternative.
            """),
            partD_pred,
        ]
        if partD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the final memo builder."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_revision, partD_rejected, partD_top_risk, partD_mitigation], widths="equal"))
        items.append(mo.callout(mo.md(f"**Memo summary:** {_memo.memo_summary}"), kind="info"))
        items.append(mo.callout(mo.md(f"**Rejected alternative:** {partD_rejected.value}"), kind="info"))

        _report_rows = "\n".join(
            f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |"
            for name, ok, detail in _report_audit["rows"]
        )
        items.append(mo.md(f"""
**Report Completeness Table**

| Report element | Status | Evidence |
|---|---|---|
{_report_rows}
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_16_metric_card("Completeness", f"{_report_audit['completeness_pct']:.1f}%", f"{_report_audit['passed']}/{_report_audit['total']} elements", v1_16_status_color(_report_audit["status"]), True)}
            {v1_16_metric_card("Report Status", _report_audit["status"], "defensible report gate", v1_16_status_color(_report_audit["status"]), True)}
            {v1_16_metric_card("Residual Risk", "named", _memo.top_risk, COLORS["OrangeLine"])}
        </div>
        """))

        if _report_audit["status"] == "FAIL":
            items.append(mo.callout(mo.md(
                "**Report boundary:** the memo is not yet defensible. Check rejected alternative, operating envelope, source trace, validation evidence, and residual risk."
            ), kind="danger"))

        items.append(mo.md(f"""
**Validation Tests To Attach**

{chr(10).join(f"- {test}" for test in _memo.validation_tests)}

**Durable Principle**

{_memo.durable_principle}
        """))

        items.append(mo.accordion({
            "Math Peek / Source Model - defensible report": mo.md("""
The report gate is:

`defensible = decision + rejected alternative + source trace + residual risk + validation evidence`

The memo can accept residual risk, but it cannot hide that risk or pretend a
rejected alternative was never considered.
            """)
        }))
        items.append(source_trace({
            "chapter_anchor": "Conclusion: Journey Forward; Engineering responsibility; Summary",
            "claim": "The final design report must defend the system, not just the model.",
            "shared_helper": "mlsysbook_labs.architecture_memo",
            "notebook_helper": "v1_16_report_audit",
            "hardware_ref": v1_16_capstone.hardware_ref,
            "model_ref": v1_16_capstone.model_ref,
        }, summary="Math Peek/source trace: final report defense"))
        items.append(partD_checkpoint)

        if partD_pred.value == "revise":
            items.append(mo.callout(mo.md("**Prediction check:** correct. The memo must revise, reject, risk-rank, and validate."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**A capstone memo is not a victory lap.** It is a defensible revision under residual risk."
            ), kind="warn"))
        if partD_checkpoint.value == "defensible" and _report_audit["status"] == "PASS":
            items.append(mo.callout(mo.md("**Checkpoint recorded:** the report has the required evidence packet."), kind="success"))
        elif partD_checkpoint.value == "defensible":
            items.append(mo.callout(mo.md("**Checkpoint warning:** the report decision is defensible only after every required element passes."), kind="warn"))
        elif partD_checkpoint.value is not None:
            items.append(mo.callout(mo.md("**Checkpoint recorded:** the final memo needs revision before being treated as complete."), kind="warn"))
        return mo.vstack(items)

    def build_synthesis():
        _checkpoint_rows = "\n".join((
            f"| Part A binding checkpoint | {partA_checkpoint.value or 'not recorded'} |",
            f"| Part B lever checkpoint | {partB_checkpoint.value or 'not recorded'} |",
            f"| Part C envelope checkpoint | {partC_checkpoint.value or 'not recorded'} |",
            f"| Part D report checkpoint | {partD_checkpoint.value or 'not recorded'} |",
        ))
        return mo.vstack([
            mo.md(f"""
## Synthesis: Volume I Final Report

The Volume I capstone now has one evidence chain:

| Evidence replay | Result |
|---|---|
| Binding amount | {_binding["amount"]} ({_binding["score"]:.1f}%) |
| Optimization lever | {_lever["lever_label"]} ({_lever["status"]}) |
| Operating envelope | {'PASS' if _audit.feasible else 'FAIL'}; most fragile axis: {_audit.most_fragile} |
| Report defense | {_report_audit["status"]}; completeness {_report_audit["completeness_pct"]:.1f}% |
| Residual risk | {_memo.top_risk} |

| Checkpoint | Recorded decision |
|---|---|
{_checkpoint_rows}
            """),
            mo.callout(mo.md(
                f"**1. The selected track controls the envelope.** {v1_16_capstone.label} uses the same concept sequence as every track, but with different constraints and evidence."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. Ledger gaps are evidence gaps.** Presets are useful for teaching, but the final report must label them."
            ), kind="info"),
            mo.callout(mo.md(
                f"**3. Carry-forward question for Volume II:** {v1_16_volume_ii_question(v1_16_capstone.track_id)}"
            ), kind="info"),
            source_trace({
                "chapter_anchor": "Conclusion: Horizon note from node to fleet; Summary",
                "claim": "Volume I names the local binding amount; Volume II asks what changes when the resource boundary moves outward.",
                "binding_amount": _binding["amount"],
                "track": v1_16_profile.label,
                "report_artifact": v1_16_capstone.report_artifact,
            }, summary="Synthesis source trace: Volume I to Volume II handoff"),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Binding Amount": build_part_a(),
        "Part B: Right Lever": build_part_b(),
        "Part C: Operating Envelope": build_part_c(),
        "Part D: Design Report": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


# ===========================================================================
# ZONE D: LEDGER HUD AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    partA_checkpoint,
    partA_evidence_floor,
    partA_pred,
    partB_checkpoint,
    partB_intensity,
    partB_lever,
    partB_pred,
    partC_checkpoint,
    partC_evidence,
    partC_guardrail,
    partC_model_growth,
    partC_pred,
    partC_workload,
    partD_checkpoint,
    partD_mitigation,
    partD_pred,
    partD_rejected,
    partD_revision,
    partD_top_risk,
    replay_ledger,
    sensitivity_audit,
    v1_16_amount_scores,
    v1_16_binding_amount,
    v1_16_capstone,
    v1_16_lever_audit,
    v1_16_profile,
    v1_16_variant,
    v1_16_volume_ii_question,
):
    _ledger_result = replay_ledger(v1_16_capstone, ledger._state.history)
    _amount_rows = v1_16_amount_scores(v1_16_capstone, _ledger_result, partA_evidence_floor.value)
    _binding = v1_16_binding_amount(_amount_rows)
    _lever = v1_16_lever_audit(
        v1_16_capstone.track_id,
        _binding["amount"],
        partB_lever.value,
        partB_intensity.value,
    )
    _audit = sensitivity_audit(
        v1_16_capstone,
        workload_multiplier=partC_workload.value,
        model_growth_pct=partC_model_growth.value,
        guardrail_tightening_pct=partC_guardrail.value,
        evidence_confidence_pct=partC_evidence.value,
    )
    _ready = all(value is not None for value in (
        partA_pred.value,
        partA_checkpoint.value,
        partB_pred.value,
        partB_checkpoint.value,
        partC_pred.value,
        partC_checkpoint.value,
        partD_pred.value,
        partD_checkpoint.value,
    ))
    if _ready:
        ledger.save(chapter=16, design={
            "chapter": "v1_16",
            "track_id": v1_16_profile.track_id,
            "scenario_id": v1_16_variant.scenario_id,
            "hardware_ref": v1_16_capstone.hardware_ref,
            "model_ref": v1_16_capstone.model_ref,
            "architecture_goal": v1_16_capstone.architecture_goal,
            "durable_principle": v1_16_capstone.durable_principle,
            "completed": True,
            "binding_amount_prediction": partA_pred.value,
            "evidence_floor_pct": partA_evidence_floor.value,
            "binding_amount_actual": _binding["amount"],
            "binding_amount_score_pct": _binding["score"],
            "part_a_checkpoint": partA_checkpoint.value,
            "optimization_prediction": partB_pred.value,
            "optimization_lever": _lever["lever_label"],
            "optimization_intensity_pct": partB_intensity.value,
            "optimization_status": _lever["status"],
            "downstream_debt_pct": _lever["debt_pct"],
            "part_b_checkpoint": partB_checkpoint.value,
            "sensitivity_prediction": partC_pred.value,
            "workload_multiplier": partC_workload.value,
            "model_growth_pct": partC_model_growth.value,
            "guardrail_tightening_pct": partC_guardrail.value,
            "evidence_confidence_pct": partC_evidence.value,
            "operating_envelope_status": "PASS" if _audit.feasible else "FAIL",
            "most_fragile_axis": _audit.most_fragile,
            "part_c_checkpoint": partC_checkpoint.value,
            "memo_prediction": partD_pred.value,
            "revised_decision": partD_revision.value,
            "rejected_alternative": partD_rejected.value,
            "top_residual_risk": partD_top_risk.value,
            "mitigation_evidence": partD_mitigation.value,
            "part_d_checkpoint": partD_checkpoint.value,
            "volume_ii_carry_forward_question": v1_16_volume_ii_question(v1_16_capstone.track_id),
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">16 &middot; Architect's Audit</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_16_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">ARTIFACT</span>
        <span class="hud-value">{v1_16_capstone.report_artifact}</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    architecture_memo,
    build_lab_report,
    ledger,
    mo,
    partA_checkpoint,
    partA_evidence_floor,
    partA_pred,
    partB_checkpoint,
    partB_intensity,
    partB_lever,
    partB_pred,
    partC_checkpoint,
    partC_evidence,
    partC_guardrail,
    partC_model_growth,
    partC_pred,
    partC_workload,
    partD_checkpoint,
    partD_mitigation,
    partD_pred,
    partD_rejected,
    partD_revision,
    partD_top_risk,
    replay_ledger,
    report_export_panel,
    sensitivity_audit,
    v1_16_amount_scores,
    v1_16_binding_amount,
    v1_16_capstone,
    v1_16_lever_audit,
    v1_16_metadata,
    v1_16_profile,
    v1_16_report_audit,
    v1_16_variant,
    v1_16_volume_ii_question,
):
    _ledger_result = replay_ledger(v1_16_capstone, ledger._state.history)
    _amount_rows = v1_16_amount_scores(v1_16_capstone, _ledger_result, partA_evidence_floor.value)
    _binding = v1_16_binding_amount(_amount_rows)
    _lever = v1_16_lever_audit(
        v1_16_capstone.track_id,
        _binding["amount"],
        partB_lever.value,
        partB_intensity.value,
    )
    _audit = sensitivity_audit(
        v1_16_capstone,
        workload_multiplier=partC_workload.value,
        model_growth_pct=partC_model_growth.value,
        guardrail_tightening_pct=partC_guardrail.value,
        evidence_confidence_pct=partC_evidence.value,
    )
    _memo = architecture_memo(
        v1_16_capstone,
        revised_decision=partD_revision.value,
        top_risk=partD_top_risk.value,
        mitigation=partD_mitigation.value,
    )
    _report_audit = v1_16_report_audit(
        _ledger_result,
        _audit,
        _memo,
        partD_rejected.value,
        v1_16_capstone,
    )

    _incomplete = []
    if partA_pred.value is None:
        _incomplete.append("Part A binding amount prediction")
    if partA_checkpoint.value is None:
        _incomplete.append("Part A checkpoint")
    if partB_pred.value is None:
        _incomplete.append("Part B optimization prediction")
    if partB_checkpoint.value is None:
        _incomplete.append("Part B checkpoint")
    if partC_pred.value is None:
        _incomplete.append("Part C operating envelope prediction")
    if partC_checkpoint.value is None:
        _incomplete.append("Part C checkpoint")
    if partD_pred.value is None:
        _incomplete.append("Part D design report prediction")
    if partD_checkpoint.value is None:
        _incomplete.append("Part D checkpoint")

    _report = build_lab_report(
        v1_16_metadata,
        track=v1_16_profile.label,
        scenario=v1_16_variant.workload_summary,
        learning_objectives=(
            "Diagnose the binding D-A-M/system amount from accumulated ledger evidence.",
            "Choose an optimization lever that relieves the binding amount without creating downstream debt.",
            "Deploy only inside the measured operating envelope and defend a final report with residual risk.",
        ),
        predictions={
            "binding_amount": partA_pred.value,
            "right_lever": partB_pred.value,
            "operating_envelope": partC_pred.value,
            "design_report": partD_pred.value,
        },
        knob_settings={
            "evidence_floor_pct": partA_evidence_floor.value,
            "optimization_lever": _lever["lever_label"],
            "optimization_intensity_pct": partB_intensity.value,
            "workload_multiplier": partC_workload.value,
            "model_growth_pct": partC_model_growth.value,
            "guardrail_tightening_pct": partC_guardrail.value,
            "evidence_confidence_pct": partC_evidence.value,
            "revised_decision": partD_revision.value,
            "rejected_alternative": partD_rejected.value,
            "top_risk": partD_top_risk.value,
            "mitigation": partD_mitigation.value,
        },
        evidence_summary={
            "hardware_ref": v1_16_capstone.hardware_ref,
            "model_ref": v1_16_capstone.model_ref,
            "architecture_goal": v1_16_capstone.architecture_goal,
            "ledger_coverage_pct": round(_ledger_result.coverage_pct, 3),
            "missing_chapters": _ledger_result.missing_chapters,
            "binding_amount_actual": _binding["amount"],
            "binding_amount_score_pct": _binding["score"],
            "optimization_status": _lever["status"],
            "downstream_debt_pct": _lever["debt_pct"],
            "most_fragile": _audit.most_fragile,
            "sensitivity_feasible": _audit.feasible,
            "sensitivity_violations": _audit.violations,
            "report_completeness_pct": _report_audit["completeness_pct"],
            "report_status": _report_audit["status"],
            "revised_decision": _memo.revised_decision,
            "rejected_alternative": partD_rejected.value,
            "top_risk": _memo.top_risk,
            "volume_ii_carry_forward_question": v1_16_volume_ii_question(v1_16_capstone.track_id),
        },
        final_decision=_memo.memo_summary,
        big_takeaways=(
            "Single-machine ML system design starts by diagnosing the binding D-A-M/system amount.",
            "Local optimization is defensible only when binding relief exceeds downstream debt.",
            "A deployment report must state the operating envelope, rejected alternative, source trace, and residual risk.",
        ),
        reflections={
            "report_artifact": v1_16_capstone.report_artifact,
            "durable_principle": v1_16_capstone.durable_principle,
            "validation_tests": v1_16_capstone.validation_tests,
            "part_a_checkpoint": partA_checkpoint.value,
            "part_b_checkpoint": partB_checkpoint.value,
            "part_c_checkpoint": partC_checkpoint.value,
            "part_d_checkpoint": partD_checkpoint.value,
            "volume_ii_carry_forward_question": v1_16_volume_ii_question(v1_16_capstone.track_id),
        },
        residual_risk=(
            f"{_memo.top_risk}. The rejected alternative is {partD_rejected.value}. "
            f"The mitigation is {_memo.mitigation}; evidence still needs track-specific validation before production use."
        ),
        source_trace={
            "track_id": v1_16_profile.track_id,
            "scenario_id": v1_16_variant.scenario_id,
            "hardware_ref": v1_16_variant.hardware_ref,
            "model_ref": v1_16_variant.model_ref,
            "shared_helper": "mlsysbook_labs.capstone",
            "notebook_helpers": "v1_16_amount_scores, v1_16_lever_audit, v1_16_report_audit",
            "source_policy": v1_16_profile.source_policy,
        },
        result_snapshot={
            "capstone_profile": v1_16_capstone,
            "ledger_replay": _ledger_result,
            "amount_rows": _amount_rows,
            "binding_amount": _binding,
            "lever_audit": _lever,
            "sensitivity_audit": _audit,
            "architecture_memo": _memo,
            "report_audit": _report_audit,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-16 architecture memo is generated locally from the selected track, "
                "ledger entries, your inputs, and the computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
