import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import html as html_lib
    import math
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
        report_export_panel,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v2_07_metadata = get_lab_metadata("vol2/lab_07_fault_tolerance.py")
    return (v2_07_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_07_track_picker = track_selector(default=_default_track)
    v2_07_track_picker
    return (v2_07_track_picker,)


@app.cell
def _(get_lab_track_variant, get_track_profile, v2_07_metadata, v2_07_track_picker):
    v2_07_track_id = v2_07_track_picker.value
    v2_07_profile = get_track_profile(v2_07_track_id)
    v2_07_variant = get_lab_track_variant(v2_07_metadata.lab_id, v2_07_profile.track_id)
    v2_07_defaults = v2_07_variant.defaults
    return v2_07_defaults, v2_07_profile, v2_07_track_id, v2_07_variant


@app.cell
def _():
    v2_07_TRACK_LENSES = {
        "iphone": {
            "scenario": "Approve a staged on-device model rollout while crashes and rollback events accumulate across the installed fleet.",
            "unit_label": "phones",
            "component_label": "phone/app instance",
            "failure_mode": "bad rollout, app crash, or offline fallback miss",
            "component_mtbf_h": 20000.0,
            "fleet_min": 10000,
            "fleet_max": 1000000,
            "fleet_step": 10000,
            "fleet_default": 250000,
            "duration_min_h": 1,
            "duration_max_h": 168,
            "duration_step_h": 1,
            "duration_default_h": 24,
            "clean_target_pct": 95.0,
            "state_gb": 0.8,
            "write_min_default": 4.0,
            "write_min_max": 30.0,
            "interval_default_min": 20.0,
            "interval_max_min": 240.0,
            "tax_limit_pct": 18.0,
            "policy_default": "warm_history",
            "history_default": 2,
            "write_bw_min_gbs": 0.01,
            "write_bw_max_gbs": 5.0,
            "write_bw_step_gbs": 0.01,
            "write_bw_default_gbs": 0.20,
            "write_budget_gbs": 1.0,
            "storage_budget_gb": 4000.0,
            "recovery_objective_min": 12.0,
            "minimum_failover_min": 0.4,
            "single_availability": 0.985,
            "availability_target_pct": 99.90,
            "replica_default": 2,
            "replica_sync_latency_ms": 2.0,
            "orchestration_implication": "V2-08 must schedule staged rollout cohorts, rollback gates, and local fallback capacity before widening release.",
        },
        "oura_ring": {
            "scenario": "Protect an overnight sensing firmware update when rings may drop sync, drain battery, or need rollback away from the phone.",
            "unit_label": "rings",
            "component_label": "ring firmware instance",
            "failure_mode": "battery depletion, sensor dropout, firmware rollback, or sync gap",
            "component_mtbf_h": 30000.0,
            "fleet_min": 5000,
            "fleet_max": 500000,
            "fleet_step": 5000,
            "fleet_default": 100000,
            "duration_min_h": 1,
            "duration_max_h": 168,
            "duration_step_h": 1,
            "duration_default_h": 10,
            "clean_target_pct": 98.0,
            "state_gb": 0.05,
            "write_min_default": 3.0,
            "write_min_max": 60.0,
            "interval_default_min": 45.0,
            "interval_max_min": 360.0,
            "tax_limit_pct": 12.0,
            "policy_default": "warm_history",
            "history_default": 3,
            "write_bw_min_gbs": 0.001,
            "write_bw_max_gbs": 0.5,
            "write_bw_step_gbs": 0.001,
            "write_bw_default_gbs": 0.02,
            "write_budget_gbs": 0.10,
            "storage_budget_gb": 250.0,
            "recovery_objective_min": 30.0,
            "minimum_failover_min": 1.0,
            "single_availability": 0.990,
            "availability_target_pct": 99.95,
            "replica_default": 2,
            "replica_sync_latency_ms": 8.0,
            "orchestration_implication": "V2-08 must schedule OTA waves around radio duty cycle, battery state, and safe-mode rollback cohorts.",
        },
        "robotaxi": {
            "scenario": "Keep perception and planning inside a safety envelope while vehicle-hours expose sensor and compute faults.",
            "unit_label": "vehicles",
            "component_label": "vehicle autonomy stack",
            "failure_mode": "sensor degradation, perception compute fault, or degraded-mode entry",
            "component_mtbf_h": 8000.0,
            "fleet_min": 100,
            "fleet_max": 20000,
            "fleet_step": 100,
            "fleet_default": 5000,
            "duration_min_h": 1,
            "duration_max_h": 72,
            "duration_step_h": 1,
            "duration_default_h": 8,
            "clean_target_pct": 99.9,
            "state_gb": 12.0,
            "write_min_default": 0.8,
            "write_min_max": 20.0,
            "interval_default_min": 6.0,
            "interval_max_min": 60.0,
            "tax_limit_pct": 8.0,
            "policy_default": "replicated_state",
            "history_default": 2,
            "write_bw_min_gbs": 0.1,
            "write_bw_max_gbs": 20.0,
            "write_bw_step_gbs": 0.1,
            "write_bw_default_gbs": 4.0,
            "write_budget_gbs": 8.0,
            "storage_budget_gb": 500.0,
            "recovery_objective_min": 1.5,
            "minimum_failover_min": 0.15,
            "single_availability": 0.995,
            "availability_target_pct": 99.99,
            "replica_default": 2,
            "replica_sync_latency_ms": 3.0,
            "orchestration_implication": "V2-08 must reserve safety-critical standby compute and route degraded vehicles before utilization goals.",
        },
        "cloud_fleet": {
            "scenario": "Run a large accelerator fleet under SLA while node failures, checkpoint storms, and replica loss become routine.",
            "unit_label": "accelerators",
            "component_label": "accelerator",
            "failure_mode": "accelerator/node failure, checkpoint storm, or replica failure",
            "component_mtbf_h": 50000.0,
            "fleet_min": 256,
            "fleet_max": 25000,
            "fleet_step": 256,
            "fleet_default": 10000,
            "duration_min_h": 1,
            "duration_max_h": 2160,
            "duration_step_h": 24,
            "duration_default_h": 168,
            "clean_target_pct": 50.0,
            "state_gb": 1200.0,
            "write_min_default": 5.0,
            "write_min_max": 45.0,
            "interval_default_min": 60.0,
            "interval_max_min": 240.0,
            "tax_limit_pct": 15.0,
            "policy_default": "warm_history",
            "history_default": 2,
            "write_bw_min_gbs": 10.0,
            "write_bw_max_gbs": 500.0,
            "write_bw_step_gbs": 5.0,
            "write_bw_default_gbs": 240.0,
            "write_budget_gbs": 300.0,
            "storage_budget_gb": 10000.0,
            "recovery_objective_min": 20.0,
            "minimum_failover_min": 0.5,
            "single_availability": 0.990,
            "availability_target_pct": 99.95,
            "replica_default": 2,
            "replica_sync_latency_ms": 4.0,
            "orchestration_implication": "V2-08 must co-schedule spare pools, checkpoint I/O windows, and failure-domain-aware replicas.",
        },
    }

    v2_07_POLICY_PROFILES = {
        "retry_only": {
            "label": "Retry or pause only",
            "lost_factor": 2.0,
            "detect_min": 0.5,
            "restart_min": 0.0,
            "load_factor": 0.0,
            "warmup_min": 0.2,
            "storage_mode": "none",
            "write_multiplier": 0.0,
            "coverage": "transient stalls",
            "uncovered": "permanent component loss, silent corruption rollback",
            "validation": "straggler retry replay",
        },
        "single_checkpoint": {
            "label": "Single checkpoint + cold restart",
            "lost_factor": 0.5,
            "detect_min": 1.5,
            "restart_min": 5.0,
            "load_factor": 1.0,
            "warmup_min": 2.0,
            "storage_mode": "single",
            "write_multiplier": 1.0,
            "coverage": "fail-stop component loss",
            "uncovered": "bad checkpoint and correlated outage",
            "validation": "restore-from-checkpoint drill",
        },
        "warm_history": {
            "label": "Checkpoint history + warm restart",
            "lost_factor": 0.35,
            "detect_min": 0.8,
            "restart_min": 1.5,
            "load_factor": 0.25,
            "warmup_min": 0.5,
            "storage_mode": "history",
            "write_multiplier": 1.05,
            "coverage": "transient stalls, fail-stop loss, one bad checkpoint",
            "uncovered": "regional or shared software outage",
            "validation": "warm restart plus older-checkpoint rollback drill",
        },
        "replicated_state": {
            "label": "Replicated state + degraded mode",
            "lost_factor": 0.15,
            "detect_min": 0.4,
            "restart_min": 0.7,
            "load_factor": 0.10,
            "warmup_min": 0.2,
            "storage_mode": "replicated_history",
            "write_multiplier": 1.40,
            "coverage": "fast failover, fail-stop loss, rollback, degraded service",
            "uncovered": "shared software bug or bad upstream data",
            "validation": "failover, degraded-mode, and corruption rollback drill",
        },
    }

    v2_07_PLAN_PROFILES = {
        "local_baseline": {
            "replica_default": 1,
            "recovery_factor": 1.15,
            "cost_step": 0.08,
            "latency_factor": 1.00,
            "frame": "best-effort retry",
        },
        "balanced_policy": {
            "replica_default": 2,
            "recovery_factor": 0.65,
            "cost_step": 0.18,
            "latency_factor": 1.03,
            "frame": "graceful degradation",
        },
        "scale_first": {
            "replica_default": 3,
            "recovery_factor": 0.35,
            "cost_step": 0.38,
            "latency_factor": 1.12,
            "frame": "redundant safety path",
        },
    }

    def v2_07_lens_for(track_id):
        return v2_07_TRACK_LENSES.get(track_id, v2_07_TRACK_LENSES["cloud_fleet"])

    return v2_07_PLAN_PROFILES, v2_07_POLICY_PROFILES, v2_07_TRACK_LENSES, v2_07_lens_for


@app.cell
def _(v2_07_lens_for, v2_07_profile):
    v2_07_lens = v2_07_lens_for(v2_07_profile.track_id)
    return (v2_07_lens,)


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    track_arc_context,
    track_context,
    v2_07_lens,
    v2_07_metadata,
    v2_07_profile,
    v2_07_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #0f172a 100%);
                    padding: 32px 40px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.32);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #9ca3af; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 07
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.35rem; font-weight: 900;
                       color: #f9fafb; line-height: 1.1;">
                When Failure Is Routine
            </h1>
            <p style="margin: 0 0 8px 0; font-size: 1.08rem; font-weight: 600;
                      color: #cbd5e1; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                MTBF &middot; Young-Daly &middot; Lost Work &middot; Recovery Guardrails
            </p>
            <p style="margin: 0 0 20px 0; font-size: 1.0rem; color: #d1d5db;
                      max-width: 820px; line-height: 1.65;">
                {v2_07_lens["scenario"]} The selected track changes the persona,
                thresholds, and memo framing, but every track reasons over the same
                failure amounts.
            </p>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <span class="badge badge-info">{v2_07_profile.label}</span>
                <span class="badge badge-warn">{v2_07_variant.guardrail_metric}</span>
                <span class="badge badge-fail">{v2_07_lens["failure_mode"]}</span>
            </div>
        </div>
        """),
        track_context(v2_07_profile),
        track_arc_context(v2_07_profile, v2_07_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v2_07_lens, v2_07_profile):
    mo.Html(f"""
    <div style="border-left:4px solid {COLORS['BlueLine']}; background:white;
                border-radius:0 12px 12px 0; padding:20px 28px; margin:8px 0 16px 0;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                    text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
            Chapter invariant
        </div>
        <div style="font-size:1.0rem; color:{COLORS['Text']}; line-height:1.65;">
            At fleet scale, failures become routine. MTBF, checkpoint interval, lost work,
            recovery time, and redundancy are design amounts.
        </div>
        <div style="border-top:1px solid {COLORS['Border']}; margin:14px -28px 0 -28px;
                    padding:14px 28px 0 28px;">
            <div style="font-size:0.7rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                Track realization
            </div>
            <div style="font-size:0.95rem; color:{COLORS['TextSec']}; line-height:1.65;">
                <strong>{v2_07_profile.stakeholder}</strong> must plan for
                <strong>{v2_07_lens["unit_label"]}</strong> whose routine failure mode is
                <strong>{v2_07_lens["failure_mode"]}</strong>.
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** - Complete before this lab:

    - **Volume II, Chapter 7: Fault Tolerance and Reliability** - failure analysis
      at scale, MTBF composition, Young-Daly checkpointing, recovery procedures,
      serving redundancy, and graceful degradation.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    v2_07_part_a_prediction = mo.ui.radio(
        options={
            "A) It stays the same because component MTBF is unchanged": "same",
            "B) It falls by about 10x when fleet size grows 10x": "inverse",
            "C) It improves because more devices create redundancy": "improves",
            "D) It cannot be estimated from MTBF": "unknowable",
        },
        label="Part A prediction: if the fleet grows 10x, what happens to aggregate MTBF?",
    )
    v2_07_part_b_prediction = mo.ui.radio(
        options={
            "A) Checkpoint as often as possible": "minimum",
            "B) Checkpoint as rarely as possible": "maximum",
            "C) Choose the interval where save overhead and expected rework balance": "optimum",
            "D) Use the same interval for every fleet": "fixed",
        },
        label="Part B prediction: what checkpoint interval policy is safest?",
    )
    v2_07_part_c_prediction = mo.ui.radio(
        options={
            "A) Lost work since the last checkpoint": "lost_work",
            "B) Storage footprint and write bandwidth": "storage_write",
            "C) Detection, restart, load, and warmup time": "recovery_terms",
            "D) All of these become design amounts": "all_amounts",
        },
        label="Part C prediction: which amount can dominate failure cost?",
    )
    v2_07_part_d_prediction = mo.ui.radio(
        options={
            "A) Recovery time objective": "recovery",
            "B) Cost budget": "cost",
            "C) Latency, quality, or safety guardrails": "performance",
            "D) Any one of these can reject the plan": "any_guardrail",
        },
        label="Part D prediction: which guardrail can reject a fault-tolerance plan?",
    )
    return (
        v2_07_part_a_prediction,
        v2_07_part_b_prediction,
        v2_07_part_c_prediction,
        v2_07_part_d_prediction,
    )


@app.cell(hide_code=True)
def _(mo, v2_07_defaults, v2_07_lens):
    v2_07_fleet_size = mo.ui.slider(
        start=int(v2_07_lens["fleet_min"]),
        stop=int(v2_07_lens["fleet_max"]),
        value=int(v2_07_lens["fleet_default"]),
        step=int(v2_07_lens["fleet_step"]),
        label=f"Fleet size ({v2_07_lens['unit_label']})",
    )
    v2_07_duration_h = mo.ui.slider(
        start=int(v2_07_lens["duration_min_h"]),
        stop=int(v2_07_lens["duration_max_h"]),
        value=int(v2_07_lens["duration_default_h"]),
        step=int(v2_07_lens["duration_step_h"]),
        label="Mission or rollout window (hours)",
    )
    v2_07_write_min = mo.ui.slider(
        start=0.5,
        stop=float(v2_07_lens["write_min_max"]),
        value=float(v2_07_lens["write_min_default"]),
        step=0.5,
        label="Checkpoint or rollback snapshot write time (minutes)",
    )
    v2_07_checkpoint_interval_min = mo.ui.slider(
        start=1.0,
        stop=float(v2_07_lens["interval_max_min"]),
        value=float(v2_07_lens["interval_default_min"]),
        step=1.0,
        label="Chosen checkpoint interval (minutes)",
    )
    v2_07_recovery_policy = mo.ui.radio(
        options={
            "Retry or pause only": "retry_only",
            "Single checkpoint + cold restart": "single_checkpoint",
            "Checkpoint history + warm restart": "warm_history",
            "Replicated state + degraded mode": "replicated_state",
        },
        value=v2_07_POLICY_PROFILES[v2_07_lens["policy_default"]]["label"],
        label="Recovery policy",
    )
    v2_07_checkpoint_history = mo.ui.slider(
        start=1,
        stop=5,
        value=int(v2_07_lens["history_default"]),
        step=1,
        label="Retained checkpoint or rollback history",
    )
    v2_07_write_bandwidth_gbs = mo.ui.slider(
        start=float(v2_07_lens["write_bw_min_gbs"]),
        stop=float(v2_07_lens["write_bw_max_gbs"]),
        value=float(v2_07_lens["write_bw_default_gbs"]),
        step=float(v2_07_lens["write_bw_step_gbs"]),
        label="Available write/read bandwidth (GB/s)",
    )
    v2_07_part_a_checkpoint = mo.ui.radio(
        options={
            "Use component MTBF as the planning amount": "component",
            "Use aggregate fleet MTBF and expected failures": "aggregate",
            "Ignore failures until after the first incident": "ignore",
        },
        value="Use aggregate fleet MTBF and expected failures",
        label="Part A checkpoint: which amount should drive recovery design?",
    )
    v2_07_part_b_checkpoint = mo.ui.radio(
        options={
            "Shorten the interval because expected rework dominates": "shorten",
            "Lengthen the interval because save overhead dominates": "lengthen",
            "Stay near the computed optimum and improve write bandwidth if tax is high": "optimum",
        },
        value="Stay near the computed optimum and improve write bandwidth if tax is high",
        label="Part B checkpoint: what operating move follows from the curve?",
    )
    _plan_options = {
        v2_07_defaults["decision_options"][key]["label"]: key
        for key in ("local_baseline", "balanced_policy", "scale_first")
    }
    v2_07_plan_choice = mo.ui.radio(
        options=_plan_options,
        value=v2_07_defaults["decision_options"]["balanced_policy"]["label"],
        label="Fault-tolerance plan",
    )
    v2_07_replica_count = mo.ui.slider(
        start=1,
        stop=5,
        value=int(v2_07_lens["replica_default"]),
        step=1,
        label="Replica or standby count",
    )
    v2_07_uncovered_failure = mo.ui.radio(
        options={
            "Transient stall": "transient",
            "Permanent component loss": "permanent",
            "Silent corruption or bad checkpoint": "sdc",
            "Correlated failure domain": "correlated",
            "Shared software/configuration bug": "software",
        },
        label="Part C checkpoint: which failure remains uncovered or needs a drill?",
    )
    v2_07_rejected_alternative = mo.ui.radio(
        options=_plan_options,
        value=v2_07_defaults["decision_options"]["scale_first"]["label"],
        label="Synthesis checkpoint: rejected alternative",
    )
    v2_07_student_id = mo.ui.text(label="Student identifier", placeholder="Optional")
    v2_07_memo_decision = mo.ui.text_area(
        label="Reliability memo",
        placeholder="State checkpoint policy, replication plan, binding amount, rejected alternative, and V2-08 orchestration implication.",
        full_width=True,
    )
    return (
        v2_07_checkpoint_history,
        v2_07_checkpoint_interval_min,
        v2_07_duration_h,
        v2_07_fleet_size,
        v2_07_memo_decision,
        v2_07_part_a_checkpoint,
        v2_07_part_b_checkpoint,
        v2_07_plan_choice,
        v2_07_recovery_policy,
        v2_07_rejected_alternative,
        v2_07_replica_count,
        v2_07_student_id,
        v2_07_uncovered_failure,
        v2_07_write_bandwidth_gbs,
        v2_07_write_min,
    )


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    build_lab_report,
    go,
    html_lib,
    ledger,
    math,
    mo,
    report_export_panel,
    v2_07_PLAN_PROFILES,
    v2_07_POLICY_PROFILES,
    v2_07_checkpoint_history,
    v2_07_checkpoint_interval_min,
    v2_07_defaults,
    v2_07_duration_h,
    v2_07_fleet_size,
    v2_07_lens,
    v2_07_memo_decision,
    v2_07_metadata,
    v2_07_part_a_checkpoint,
    v2_07_part_a_prediction,
    v2_07_part_b_checkpoint,
    v2_07_part_b_prediction,
    v2_07_part_c_prediction,
    v2_07_part_d_prediction,
    v2_07_plan_choice,
    v2_07_profile,
    v2_07_recovery_policy,
    v2_07_rejected_alternative,
    v2_07_replica_count,
    v2_07_student_id,
    v2_07_uncovered_failure,
    v2_07_variant,
    v2_07_write_bandwidth_gbs,
    v2_07_write_min,
):
    def v2_07_fmt(value, digits=2):
        if isinstance(value, str):
            return value
        if abs(float(value)) >= 100:
            return f"{float(value):,.0f}"
        if abs(float(value)) >= 10:
            return f"{float(value):,.1f}"
        return f"{float(value):,.{digits}f}"

    def v2_07_pct(value, digits=2):
        return f"{100 * float(value):,.{digits}f}%"

    def v2_07_status(ok):
        return "PASS" if ok else "MISS"

    def v2_07_table(headers, rows):
        _head = "".join(f"<th>{html_lib.escape(str(header))}</th>" for header in headers)
        _body = []
        for row in rows:
            _body.append("<tr>" + "".join(f"<td>{html_lib.escape(str(cell))}</td>" for cell in row) + "</tr>")
        return mo.Html(f"""
        <div style="overflow-x:auto; margin:12px 0;">
          <table style="border-collapse:collapse; width:100%; font-size:0.88rem;">
            <thead><tr style="background:#f8fafc;">{_head}</tr></thead>
            <tbody>{''.join(_body)}</tbody>
          </table>
        </div>
        <style>
          table td, table th {{
            border: 1px solid #e5e7eb;
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
          }}
          table th {{
            font-weight: 700;
            color: #374151;
          }}
        </style>
        """)

    def v2_07_exposure(fleet_size=None, duration_h=None):
        _fleet = max(1, int(fleet_size if fleet_size is not None else v2_07_fleet_size.value))
        _duration = max(0.01, float(duration_h if duration_h is not None else v2_07_duration_h.value))
        _component_mtbf_h = float(v2_07_lens["component_mtbf_h"])
        _system_mtbf_h = _component_mtbf_h / _fleet
        _expected_failures = _duration / _system_mtbf_h
        _clean_probability = 0.0 if _expected_failures > 700 else math.exp(-_expected_failures)
        return {
            "fleet_size": _fleet,
            "duration_h": _duration,
            "component_mtbf_h": _component_mtbf_h,
            "system_mtbf_h": _system_mtbf_h,
            "expected_failures": _expected_failures,
            "clean_probability": _clean_probability,
            "failures_per_day": 24.0 / _system_mtbf_h,
            "target_clean_probability": float(v2_07_lens["clean_target_pct"]) / 100.0,
        }

    def v2_07_checkpoint_result():
        _exposure = v2_07_exposure()
        _mtbf_min = max(0.001, _exposure["system_mtbf_h"] * 60.0)
        _write_min = max(0.001, float(v2_07_write_min.value))
        _interval_min = max(0.001, float(v2_07_checkpoint_interval_min.value))
        _tau_opt = math.sqrt(2.0 * _write_min * _mtbf_min)
        _save_tax = _write_min / _interval_min
        _rework_tax = _interval_min / (2.0 * _mtbf_min)
        _total_tax = _save_tax + _rework_tax
        _side = "save overhead" if _save_tax > _rework_tax else "expected rework"
        _near = abs(_interval_min - _tau_opt) / max(_tau_opt, 0.001) <= 0.20
        return {
            "mtbf_min": _mtbf_min,
            "write_min": _write_min,
            "interval_min": _interval_min,
            "tau_opt_min": _tau_opt,
            "save_tax_pct": 100.0 * _save_tax,
            "rework_tax_pct": 100.0 * _rework_tax,
            "total_tax_pct": 100.0 * _total_tax,
            "dominant_tax": _side,
            "near_optimum": _near,
            "tax_ok": 100.0 * _total_tax <= float(v2_07_lens["tax_limit_pct"]),
        }

    def v2_07_storage_for_policy(policy, history):
        _mode = policy["storage_mode"]
        _state_gb = float(v2_07_lens["state_gb"])
        if _mode == "none":
            return 0.0
        if _mode == "single":
            return _state_gb
        if _mode == "history":
            return _state_gb * int(history)
        return _state_gb * int(history) * 2.0

    def v2_07_policy_result(policy_id=None):
        _policy_id = policy_id or v2_07_recovery_policy.value or v2_07_lens["policy_default"]
        _policy = v2_07_POLICY_PROFILES[_policy_id]
        _interval = max(0.001, float(v2_07_checkpoint_interval_min.value))
        _history = int(v2_07_checkpoint_history.value)
        _available_bw = max(0.001, float(v2_07_write_bandwidth_gbs.value))
        _state_gb = float(v2_07_lens["state_gb"])
        _write_min = max(0.001, float(v2_07_write_min.value))
        _lost_work_min = _interval * float(_policy["lost_factor"])
        _storage_gb = v2_07_storage_for_policy(_policy, _history)
        _required_write_gbs = 0.0
        if _policy["write_multiplier"] > 0:
            _required_write_gbs = (_state_gb * float(_policy["write_multiplier"])) / (_write_min * 60.0)
        _load_min = (_state_gb * float(_policy["load_factor"])) / _available_bw / 60.0
        _recovery_min = (
            _lost_work_min
            + float(_policy["detect_min"])
            + float(_policy["restart_min"])
            + _load_min
            + float(_policy["warmup_min"])
        )
        _rto_ok = _recovery_min <= float(v2_07_lens["recovery_objective_min"])
        _storage_ok = _storage_gb <= float(v2_07_lens["storage_budget_gb"])
        _write_ok = (
            _required_write_gbs <= _available_bw
            and _required_write_gbs <= float(v2_07_lens["write_budget_gbs"])
        )
        return {
            "policy_id": _policy_id,
            "label": _policy["label"],
            "lost_work_min": _lost_work_min,
            "recovery_min": _recovery_min,
            "storage_gb": _storage_gb,
            "required_write_gbs": _required_write_gbs,
            "available_write_gbs": _available_bw,
            "coverage": _policy["coverage"],
            "uncovered": _policy["uncovered"],
            "validation": _policy["validation"],
            "rto_ok": _rto_ok,
            "storage_ok": _storage_ok,
            "write_ok": _write_ok,
            "feasible": _rto_ok and _storage_ok and _write_ok,
        }

    def v2_07_all_policy_results():
        return [v2_07_policy_result(policy_id) for policy_id in v2_07_POLICY_PROFILES]

    def v2_07_plan_result(plan_id=None, replicas=None, recovery=None):
        _plan_id = plan_id or v2_07_plan_choice.value or "balanced_policy"
        _replicas = max(1, int(replicas if replicas is not None else v2_07_replica_count.value))
        _option = v2_07_defaults["decision_options"][_plan_id]
        _profile = v2_07_PLAN_PROFILES[_plan_id]
        _recovery = recovery or v2_07_policy_result()
        _availability = 1.0 - (1.0 - float(v2_07_lens["single_availability"])) ** _replicas
        _recovery_min = max(
            float(v2_07_lens["minimum_failover_min"]),
            _recovery["recovery_min"] * float(_profile["recovery_factor"]) / math.sqrt(_replicas),
        )
        _cost = float(_option["base_cost"]) * (1.0 + max(0, _replicas - 1) * float(_profile["cost_step"]))
        _latency_ms = (
            float(_option["base_latency_ms"]) * float(_profile["latency_factor"])
            + max(0, _replicas - 1) * float(v2_07_lens["replica_sync_latency_ms"])
        )
        _quality_pct = float(_option["quality_pct"])
        _guardrail_pct = float(_option["guardrail_pct"])
        _recovery_ok = _recovery_min <= float(v2_07_lens["recovery_objective_min"])
        _availability_ok = _availability * 100.0 >= float(v2_07_lens["availability_target_pct"])
        _cost_ok = _cost <= float(v2_07_defaults["cost_budget"])
        _latency_ok = _latency_ms <= float(v2_07_defaults["latency_budget_ms"])
        _quality_ok = _quality_pct >= float(v2_07_defaults["quality_floor_pct"])
        _guardrail_ok = _guardrail_pct >= float(v2_07_defaults["guardrail_floor_pct"])
        _misses = []
        if not _recovery_ok:
            _misses.append("recovery objective")
        if not _availability_ok:
            _misses.append("availability target")
        if not _cost_ok:
            _misses.append("cost budget")
        if not _latency_ok:
            _misses.append("latency budget")
        if not _quality_ok:
            _misses.append("quality floor")
        if not _guardrail_ok:
            _misses.append(f"{v2_07_variant.guardrail_metric} floor")
        return {
            "plan_id": _plan_id,
            "label": _option["label"],
            "frame": _profile["frame"],
            "replicas": _replicas,
            "availability_pct": _availability * 100.0,
            "recovery_min": _recovery_min,
            "cost": _cost,
            "latency_ms": _latency_ms,
            "quality_pct": _quality_pct,
            "guardrail_pct": _guardrail_pct,
            "recovery_ok": _recovery_ok,
            "availability_ok": _availability_ok,
            "cost_ok": _cost_ok,
            "latency_ok": _latency_ok,
            "quality_ok": _quality_ok,
            "guardrail_ok": _guardrail_ok,
            "feasible": not _misses,
            "misses": ", ".join(_misses) if _misses else "none",
            "mitigation": _option["mitigation"],
            "validation_requirement": _option["validation_requirement"],
            "residual_risk": _option["residual_risk"],
        }

    def v2_07_default_plan_results(recovery=None):
        return [
            v2_07_plan_result(plan_id, v2_07_PLAN_PROFILES[plan_id]["replica_default"], recovery)
            for plan_id in ("local_baseline", "balanced_policy", "scale_first")
        ]

    def v2_07_binding_summary(exposure, checkpoint, recovery, plan):
        _candidates = [
            {
                "label": "expected failures",
                "value": exposure["expected_failures"],
                "limit": 1.0,
                "unit": "failures/window",
                "severity": exposure["expected_failures"] / 1.0,
                "ok": exposure["expected_failures"] < 1.0,
            },
            {
                "label": "checkpoint tax",
                "value": checkpoint["total_tax_pct"],
                "limit": float(v2_07_lens["tax_limit_pct"]),
                "unit": "percent",
                "severity": checkpoint["total_tax_pct"] / max(float(v2_07_lens["tax_limit_pct"]), 0.001),
                "ok": checkpoint["tax_ok"],
            },
            {
                "label": "recovery time",
                "value": recovery["recovery_min"],
                "limit": float(v2_07_lens["recovery_objective_min"]),
                "unit": "minutes",
                "severity": recovery["recovery_min"] / max(float(v2_07_lens["recovery_objective_min"]), 0.001),
                "ok": recovery["rto_ok"],
            },
            {
                "label": "storage footprint",
                "value": recovery["storage_gb"],
                "limit": float(v2_07_lens["storage_budget_gb"]),
                "unit": "GB",
                "severity": recovery["storage_gb"] / max(float(v2_07_lens["storage_budget_gb"]), 0.001),
                "ok": recovery["storage_ok"],
            },
            {
                "label": "write bandwidth",
                "value": recovery["required_write_gbs"],
                "limit": min(float(v2_07_lens["write_budget_gbs"]), recovery["available_write_gbs"]),
                "unit": "GB/s",
                "severity": recovery["required_write_gbs"] / max(min(float(v2_07_lens["write_budget_gbs"]), recovery["available_write_gbs"]), 0.001),
                "ok": recovery["write_ok"],
            },
            {
                "label": "plan recovery",
                "value": plan["recovery_min"],
                "limit": float(v2_07_lens["recovery_objective_min"]),
                "unit": "minutes",
                "severity": plan["recovery_min"] / max(float(v2_07_lens["recovery_objective_min"]), 0.001),
                "ok": plan["recovery_ok"],
            },
            {
                "label": "plan cost",
                "value": plan["cost"],
                "limit": float(v2_07_defaults["cost_budget"]),
                "unit": "cost units",
                "severity": plan["cost"] / max(float(v2_07_defaults["cost_budget"]), 0.001),
                "ok": plan["cost_ok"],
            },
            {
                "label": "plan latency",
                "value": plan["latency_ms"],
                "limit": float(v2_07_defaults["latency_budget_ms"]),
                "unit": "ms",
                "severity": plan["latency_ms"] / max(float(v2_07_defaults["latency_budget_ms"]), 0.001),
                "ok": plan["latency_ok"],
            },
            {
                "label": v2_07_variant.guardrail_metric,
                "value": plan["guardrail_pct"],
                "limit": float(v2_07_defaults["guardrail_floor_pct"]),
                "unit": "percent floor",
                "severity": float(v2_07_defaults["guardrail_floor_pct"]) / max(plan["guardrail_pct"], 0.001),
                "ok": plan["guardrail_ok"],
            },
        ]
        return max(_candidates, key=lambda item: item["severity"])

    def v2_07_mtbf_chart(exposure):
        _start = int(v2_07_lens["fleet_min"])
        _stop = int(v2_07_lens["fleet_max"])
        _points = []
        for i in range(0, 40):
            _ratio = i / 39
            _points.append(max(1, round(_start * ((_stop / _start) ** _ratio))))
        _mtbf = [float(v2_07_lens["component_mtbf_h"]) / point for point in _points]
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_points,
            y=_mtbf,
            mode="lines",
            name="Aggregate MTBF",
            line=dict(color=COLORS["RedLine"], width=3),
        ))
        _fig.add_trace(go.Scatter(
            x=[exposure["fleet_size"]],
            y=[exposure["system_mtbf_h"]],
            mode="markers",
            name="Current design",
            marker=dict(color=COLORS["BlueLine"], size=12),
        ))
        _fig.add_hline(
            y=exposure["duration_h"],
            line_dash="dash",
            line_color=COLORS["OrangeLine"],
            annotation_text="mission window",
        )
        _fig.update_layout(
            height=360,
            xaxis_title=f"Fleet size ({v2_07_lens['unit_label']})",
            yaxis_title="Aggregate MTBF (hours)",
            xaxis_type="log",
            yaxis_type="log",
            legend=dict(orientation="h", y=1.15, x=0),
            margin=dict(l=65, r=20, t=50, b=45),
        )
        return apply_plotly_theme(_fig)

    def v2_07_young_daly_chart(checkpoint):
        _stop = float(v2_07_lens["interval_max_min"])
        _start = 1.0
        _taus = [_start + (_stop - _start) * i / 79 for i in range(80)]
        _save = [100.0 * checkpoint["write_min"] / tau for tau in _taus]
        _rework = [100.0 * tau / (2.0 * checkpoint["mtbf_min"]) for tau in _taus]
        _total = [_save[i] + _rework[i] for i in range(len(_taus))]
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_taus, y=_save, mode="lines", name="Save overhead"))
        _fig.add_trace(go.Scatter(x=_taus, y=_rework, mode="lines", name="Expected rework"))
        _fig.add_trace(go.Scatter(
            x=_taus,
            y=_total,
            mode="lines",
            name="Total checkpoint tax",
            line=dict(color=COLORS["GreenLine"], width=3),
        ))
        _fig.add_trace(go.Scatter(
            x=[checkpoint["interval_min"]],
            y=[checkpoint["total_tax_pct"]],
            mode="markers",
            name="Chosen interval",
            marker=dict(color=COLORS["RedLine"], size=12),
        ))
        _fig.add_vline(
            x=checkpoint["tau_opt_min"],
            line_dash="dash",
            line_color=COLORS["BlueLine"],
            annotation_text="Young-Daly optimum",
        )
        _fig.update_layout(
            height=360,
            xaxis_title="Checkpoint interval (minutes)",
            yaxis_title="Wasted capacity (%)",
            legend=dict(orientation="h", y=1.15, x=0),
            margin=dict(l=65, r=20, t=50, b=45),
        )
        return apply_plotly_theme(_fig)

    def v2_07_availability_chart(rows):
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=[row["label"] for row in rows],
            y=[row["availability_pct"] for row in rows],
            marker_color=[
                COLORS["GreenLine"] if row["feasible"] else COLORS["RedLine"]
                for row in rows
            ],
            text=[v2_07_status(row["feasible"]) for row in rows],
            textposition="auto",
        ))
        _fig.add_hline(
            y=float(v2_07_lens["availability_target_pct"]),
            line_dash="dash",
            line_color=COLORS["BlueLine"],
            annotation_text="availability target",
        )
        _fig.update_layout(
            height=320,
            yaxis_title="Availability (%)",
            showlegend=False,
            margin=dict(l=60, r=20, t=35, b=70),
        )
        return apply_plotly_theme(_fig)

    def v2_07_math_peek(title, body):
        return mo.accordion({title: mo.md(body)})

    def v2_07_prediction_feedback(selected, correct, explanation):
        if selected is None:
            return mo.callout(mo.md("Select a prediction to unlock the instrument."), kind="warn")
        return mo.callout(
            mo.md(
                f"You predicted `{selected}`; actual result is `{correct}`. "
                + ("Correct. " if selected == correct else "Revise the prior. ")
                + explanation
            ),
            kind="success" if selected == correct else "warn",
        )

    def build_part_a():
        _exposure = v2_07_exposure()
        _clean_ok = _exposure["clean_probability"] >= _exposure["target_clean_probability"]
        _failure_expected = _exposure["expected_failures"] >= 1.0
        _items = [
            mo.md("## Part A - Aggregate MTBF Falls as Fleet Size Grows"),
            mo.callout(mo.md(
                f"{v2_07_profile.stakeholder}: decide whether {v2_07_lens['fleet_default']:,} "
                f"{v2_07_lens['unit_label']} can finish the window without treating "
                f"{v2_07_lens['failure_mode']} as routine."
            ), kind="info"),
            v2_07_part_a_prediction,
        ]
        if v2_07_part_a_prediction.value is None:
            _items.append(v2_07_prediction_feedback(None, "inverse", ""))
            return mo.vstack(_items)
        _items.extend([
            v2_07_prediction_feedback(
                v2_07_part_a_prediction.value,
                "inverse",
                "Aggregate failure rate grows with component count, so aggregate MTBF falls by the same factor.",
            ),
            mo.hstack([v2_07_fleet_size, v2_07_duration_h], justify="start"),
            mo.as_html(v2_07_mtbf_chart(_exposure)),
            v2_07_table(
                ["Amount", "Value", "Decision meaning"],
                [
                    ["Component MTBF", f"{v2_07_fmt(_exposure['component_mtbf_h'])} h", v2_07_lens["component_label"]],
                    ["Aggregate MTBF", f"{v2_07_fmt(_exposure['system_mtbf_h'])} h", "mean time between any fleet-visible failure"],
                    ["Expected failures", v2_07_fmt(_exposure["expected_failures"]), "routine if >= 1 in the window"],
                    ["Clean-run probability", v2_07_pct(_exposure["clean_probability"]), f"target {v2_07_lens['clean_target_pct']}%"],
                    ["Failures/day", v2_07_fmt(_exposure["failures_per_day"]), "sets recovery staffing and automation pressure"],
                ],
            ),
            mo.callout(
                mo.md(
                    f"Boundary: expected failures = **{v2_07_fmt(_exposure['expected_failures'])}** "
                    f"in the window and clean-run probability = **{v2_07_pct(_exposure['clean_probability'])}**. "
                    f"Recovery must be automatic for this track."
                    if (_failure_expected or not _clean_ok)
                    else "The selected window is below one expected failure, but the margin shrinks linearly with fleet size."
                ),
                kind="warn" if (_failure_expected or not _clean_ok) else "success",
            ),
            v2_07_math_peek(
                "Math Peek: Fleet MTBF",
                """
The chapter uses the exponential reliability model:

`R_system(t) = exp(-N * lambda * t)`

For identical independent components:

`MTBF_system = MTBF_component / N`

The track changes what a component is, but the amount-system reasoning is the
same: more exposed units create a shorter time between fleet-visible failures.
                """,
            ),
            v2_07_part_a_checkpoint,
        ])
        return mo.vstack(_items)

    def build_part_b():
        _checkpoint = v2_07_checkpoint_result()
        _items = [
            mo.md("## Part B - Checkpoint Interval Has an Optimum"),
            mo.callout(mo.md(
                f"The {v2_07_profile.label} plan must choose how often to save recoverable state. "
                "The goal is not maximum checkpointing or minimum checkpointing; it is minimum wasted capacity."
            ), kind="info"),
            v2_07_part_b_prediction,
        ]
        if v2_07_part_b_prediction.value is None:
            _items.append(v2_07_prediction_feedback(None, "optimum", ""))
            return mo.vstack(_items)
        _items.extend([
            v2_07_prediction_feedback(
                v2_07_part_b_prediction.value,
                "optimum",
                "Young-Daly gives the minimum of a U-shaped tax: save overhead falls with interval while expected rework rises.",
            ),
            mo.hstack([v2_07_write_min, v2_07_checkpoint_interval_min], justify="start"),
            mo.as_html(v2_07_young_daly_chart(_checkpoint)),
            v2_07_table(
                ["Amount", "Value", "Limit or comparison"],
                [
                    ["System MTBF", f"{v2_07_fmt(_checkpoint['mtbf_min'])} min", "from Part A aggregate MTBF"],
                    ["Write time", f"{v2_07_fmt(_checkpoint['write_min'])} min", "snapshot cost paid every checkpoint"],
                    ["Chosen interval", f"{v2_07_fmt(_checkpoint['interval_min'])} min", "student-controlled policy"],
                    ["Young-Daly optimum", f"{v2_07_fmt(_checkpoint['tau_opt_min'])} min", "minimum total checkpoint tax"],
                    ["Total tax", f"{v2_07_fmt(_checkpoint['total_tax_pct'])}%", f"limit {v2_07_lens['tax_limit_pct']}%"],
                    ["Dominant side", _checkpoint["dominant_tax"], "move interval toward the opposite side"],
                ],
            ),
            mo.callout(
                mo.md(
                    "Near optimum: the current interval is within 20 percent of the Young-Daly interval."
                    if _checkpoint["near_optimum"]
                    else f"Boundary: current interval is dominated by **{_checkpoint['dominant_tax']}**. "
                         f"Move from {v2_07_fmt(_checkpoint['interval_min'])} min toward "
                         f"{v2_07_fmt(_checkpoint['tau_opt_min'])} min, or reduce write time with more bandwidth."
                ),
                kind="success" if _checkpoint["near_optimum"] and _checkpoint["tax_ok"] else "warn",
            ),
            v2_07_math_peek(
                "Math Peek: Young-Daly Checkpoint Tax",
                """
The chapter's first-order checkpoint model is:

`tau_opt = sqrt(2 * T_write * MTBF_system)`

For a chosen interval `tau`, the visible tax is:

`waste = T_write / tau + tau / (2 * MTBF_system)`

Restart time is a per-failure recovery term. It should be budgeted, but not
folded into `T_write` unless a recovery-aware checkpoint model is derived.
                """,
            ),
            v2_07_part_b_checkpoint,
        ])
        return mo.vstack(_items)

    def build_part_c():
        _rows = v2_07_all_policy_results()
        _selected = v2_07_policy_result()
        _items = [
            mo.md("## Part C - Lost Work and Recovery Policy Trade Amounts"),
            mo.callout(mo.md(
                f"A {v2_07_lens['failure_mode']} incident has arrived. Choose the recovery policy that "
                f"meets a {v2_07_lens['recovery_objective_min']}-minute objective without exceeding "
                "storage or write-bandwidth budgets."
            ), kind="info"),
            v2_07_part_c_prediction,
        ]
        if v2_07_part_c_prediction.value is None:
            _items.append(v2_07_prediction_feedback(None, "all_amounts", ""))
            return mo.vstack(_items)
        _items.extend([
            v2_07_prediction_feedback(
                v2_07_part_c_prediction.value,
                "all_amounts",
                "Lost work, storage, bandwidth, and restart terms all become design amounts; the binding one changes by policy.",
            ),
            mo.hstack([v2_07_recovery_policy, v2_07_checkpoint_history, v2_07_write_bandwidth_gbs], justify="start"),
            v2_07_table(
                ["Policy", "Lost work", "Recovery", "Storage", "Write GB/s", "Coverage", "Status"],
                [
                    [
                        row["label"],
                        f"{v2_07_fmt(row['lost_work_min'])} min",
                        f"{v2_07_fmt(row['recovery_min'])} min",
                        f"{v2_07_fmt(row['storage_gb'])} GB",
                        v2_07_fmt(row["required_write_gbs"], 3),
                        row["coverage"],
                        v2_07_status(row["feasible"]),
                    ]
                    for row in _rows
                ],
            ),
            mo.callout(
                mo.md(
                    f"Selected policy: **{_selected['label']}**. Recovery = "
                    f"**{v2_07_fmt(_selected['recovery_min'])} min** against "
                    f"**{v2_07_lens['recovery_objective_min']} min**; storage = "
                    f"**{v2_07_fmt(_selected['storage_gb'])} GB** against "
                    f"**{v2_07_fmt(v2_07_lens['storage_budget_gb'])} GB**; required write bandwidth = "
                    f"**{v2_07_fmt(_selected['required_write_gbs'], 3)} GB/s**. "
                    f"Uncovered: {_selected['uncovered']}."
                ),
                kind="success" if _selected["feasible"] else "warn",
            ),
            v2_07_math_peek(
                "Math Peek: Failure Cost Budget",
                """
The recovery budget decomposes a failure event:

`T_failure = lost_work + T_detect + T_restart + T_load + T_warmup`

Checkpoint history increases storage but protects against a bad latest
checkpoint. Replicated state increases normal-operation cost but reduces lost
work and failover latency. A policy is feasible only when recovery time, storage,
and write bandwidth all fit the selected track.
                """,
            ),
            v2_07_uncovered_failure,
            mo.callout(mo.md(f"Validation drill: **{_selected['validation']}**."), kind="info"),
        ])
        return mo.vstack(_items)

    def build_part_d():
        _recovery = v2_07_policy_result()
        _selected = v2_07_plan_result(recovery=_recovery)
        _rows = v2_07_default_plan_results(_recovery)
        _items = [
            mo.md("## Part D - Recovery Plan Must Pass Guardrails"),
            mo.callout(mo.md(
                f"The {v2_07_profile.label} design review asks for one plan that meets recovery, "
                f"cost, latency, quality, and {v2_07_variant.guardrail_metric} guardrails."
            ), kind="info"),
            v2_07_part_d_prediction,
        ]
        if v2_07_part_d_prediction.value is None:
            _items.append(v2_07_prediction_feedback(None, "any_guardrail", ""))
            return mo.vstack(_items)
        _items.extend([
            v2_07_prediction_feedback(
                v2_07_part_d_prediction.value,
                "any_guardrail",
                "A reliability plan is valid only inside all guardrails at once; a single miss rejects it.",
            ),
            mo.hstack([v2_07_plan_choice, v2_07_replica_count], justify="start"),
            mo.as_html(v2_07_availability_chart(_rows)),
            v2_07_table(
                ["Plan", "Replicas", "Avail.", "Recovery", "Cost", "Latency", "Quality", v2_07_variant.guardrail_metric, "Misses"],
                [
                    [
                        row["label"],
                        row["replicas"],
                        f"{v2_07_fmt(row['availability_pct'], 4)}%",
                        f"{v2_07_fmt(row['recovery_min'])} min",
                        v2_07_fmt(row["cost"]),
                        f"{v2_07_fmt(row['latency_ms'])} ms",
                        f"{v2_07_fmt(row['quality_pct'])}%",
                        f"{v2_07_fmt(row['guardrail_pct'])}%",
                        row["misses"],
                    ]
                    for row in _rows
                ],
            ),
            v2_07_table(
                ["Selected plan amount", "Value", "Limit", "Status"],
                [
                    ["Availability", f"{v2_07_fmt(_selected['availability_pct'], 4)}%", f"{v2_07_lens['availability_target_pct']}%", v2_07_status(_selected["availability_ok"])],
                    ["Recovery time", f"{v2_07_fmt(_selected['recovery_min'])} min", f"{v2_07_lens['recovery_objective_min']} min", v2_07_status(_selected["recovery_ok"])],
                    ["Cost", v2_07_fmt(_selected["cost"]), v2_07_fmt(v2_07_defaults["cost_budget"]), v2_07_status(_selected["cost_ok"])],
                    ["Latency/performance", f"{v2_07_fmt(_selected['latency_ms'])} ms", f"{v2_07_defaults['latency_budget_ms']} ms", v2_07_status(_selected["latency_ok"])],
                    ["Quality", f"{v2_07_fmt(_selected['quality_pct'])}%", f"{v2_07_defaults['quality_floor_pct']}%", v2_07_status(_selected["quality_ok"])],
                    [v2_07_variant.guardrail_metric, f"{v2_07_fmt(_selected['guardrail_pct'])}%", f"{v2_07_defaults['guardrail_floor_pct']}%", v2_07_status(_selected["guardrail_ok"])],
                ],
            ),
            mo.callout(
                mo.md(
                    f"Selected **{_selected['label']}** is feasible. Residual risk: {_selected['residual_risk']}"
                    if _selected["feasible"]
                    else f"Selected **{_selected['label']}** misses: **{_selected['misses']}**. "
                         "Adjust plan, replicas, or the recovery policy before accepting it."
                ),
                kind="success" if _selected["feasible"] else "warn",
            ),
            v2_07_math_peek(
                "Math Peek: Redundancy Guardrail Gate",
                """
For independent replicas, the chapter availability model is:

`A_system = 1 - (1 - A_single)^k`

The design gate is a conjunction:

`feasible = recovery_ok and cost_ok and performance_ok and guardrail_ok`

Replication can improve availability while still failing cost, latency, or
quality. That is why the plan must pass all guardrails, not just one metric.
                """,
            ),
            v2_07_rejected_alternative,
        ])
        return mo.vstack(_items)

    def build_synthesis():
        _exposure = v2_07_exposure()
        _checkpoint = v2_07_checkpoint_result()
        _recovery = v2_07_policy_result()
        _plan = v2_07_plan_result(recovery=_recovery)
        _binding = v2_07_binding_summary(_exposure, _checkpoint, _recovery, _plan)
        _rejected_id = v2_07_rejected_alternative.value or "scale_first"
        _rejected_label = v2_07_defaults["decision_options"][_rejected_id]["label"]
        _decision_text = v2_07_memo_decision.value or (
            f"Use {_recovery['label']} with a {_checkpoint['interval_min']:.1f}-minute checkpoint interval "
            f"and {_plan['label']} at {_plan['replicas']} replicas. Binding amount: "
            f"{_binding['label']} = {_binding['value']:.2f} {_binding['unit']}. "
            f"Reject {_rejected_label}. {v2_07_lens['orchestration_implication']}"
        )
        _snapshot = {
            "track_id": v2_07_profile.track_id,
            "scenario_id": v2_07_variant.scenario_id,
            "fleet_size": _exposure["fleet_size"],
            "duration_h": _exposure["duration_h"],
            "component_mtbf_h": _exposure["component_mtbf_h"],
            "system_mtbf_h": _exposure["system_mtbf_h"],
            "expected_failures": _exposure["expected_failures"],
            "clean_probability": _exposure["clean_probability"],
            "checkpoint_interval_min": _checkpoint["interval_min"],
            "young_daly_interval_min": _checkpoint["tau_opt_min"],
            "checkpoint_tax_pct": _checkpoint["total_tax_pct"],
            "recovery_policy": _recovery["label"],
            "lost_work_min": _recovery["lost_work_min"],
            "recovery_time_min": _recovery["recovery_min"],
            "storage_gb": _recovery["storage_gb"],
            "required_write_gbs": _recovery["required_write_gbs"],
            "replication_plan": _plan["label"],
            "replica_count": _plan["replicas"],
            "availability_pct": _plan["availability_pct"],
            "plan_cost": _plan["cost"],
            "plan_latency_ms": _plan["latency_ms"],
            "binding_failure_amount": _binding,
            "rejected_alternative": _rejected_label,
            "uncovered_failure": v2_07_uncovered_failure.value or _recovery["uncovered"],
            "v2_08_orchestration_implication": v2_07_lens["orchestration_implication"],
        }
        _incomplete = []
        if v2_07_part_a_prediction.value is None:
            _incomplete.append("Part A aggregate MTBF prediction")
        if v2_07_part_b_prediction.value is None:
            _incomplete.append("Part B checkpoint interval prediction")
        if v2_07_part_c_prediction.value is None:
            _incomplete.append("Part C recovery policy prediction")
        if v2_07_part_d_prediction.value is None:
            _incomplete.append("Part D guardrail prediction")
        if v2_07_uncovered_failure.value is None:
            _incomplete.append("Part C uncovered failure checkpoint")

        ledger.save(
            chapter=7,
            design={
                "lab_id": v2_07_metadata.lab_id,
                "track_id": v2_07_profile.track_id,
                "scenario_id": v2_07_variant.scenario_id,
                "decision": _decision_text,
                "binding_constraint": _binding["label"],
                "result_snapshot": _snapshot,
            },
        )
        _report = build_lab_report(
            v2_07_metadata,
            student_id=v2_07_student_id.value or "",
            track=v2_07_profile.label,
            scenario=v2_07_variant.workload_summary,
            learning_objectives=(
                "Calculate aggregate MTBF and clean-completion probability at fleet scale.",
                "Use Young-Daly reasoning to choose a checkpoint interval with an optimum.",
                "Compare recovery policies across lost work, storage, write bandwidth, and resilience.",
                "Select a fault-tolerance plan that satisfies recovery, cost, and performance guardrails.",
            ),
            predictions={
                "aggregate_mtbf": v2_07_part_a_prediction.value,
                "checkpoint_interval": v2_07_part_b_prediction.value,
                "recovery_policy_amount": v2_07_part_c_prediction.value,
                "guardrail_plan": v2_07_part_d_prediction.value,
            },
            knob_settings={
                "fleet_size": v2_07_fleet_size.value,
                "duration_h": v2_07_duration_h.value,
                "part_a_checkpoint": v2_07_part_a_checkpoint.value,
                "write_min": v2_07_write_min.value,
                "checkpoint_interval_min": v2_07_checkpoint_interval_min.value,
                "part_b_checkpoint": v2_07_part_b_checkpoint.value,
                "recovery_policy": v2_07_recovery_policy.value,
                "checkpoint_history": v2_07_checkpoint_history.value,
                "write_bandwidth_gbs": v2_07_write_bandwidth_gbs.value,
                "plan_choice": v2_07_plan_choice.value,
                "replica_count": v2_07_replica_count.value,
            },
            binding_constraints={
                "binding_failure_amount": _binding,
                "selected_policy_feasible": _recovery["feasible"],
                "selected_plan_feasible": _plan["feasible"],
            },
            evidence_summary={
                "system_mtbf_h": round(_exposure["system_mtbf_h"], 4),
                "expected_failures": round(_exposure["expected_failures"], 4),
                "clean_probability": round(_exposure["clean_probability"], 6),
                "young_daly_interval_min": round(_checkpoint["tau_opt_min"], 3),
                "checkpoint_tax_pct": round(_checkpoint["total_tax_pct"], 3),
                "recovery_time_min": round(_recovery["recovery_min"], 3),
                "storage_gb": round(_recovery["storage_gb"], 3),
                "availability_pct": round(_plan["availability_pct"], 5),
                "plan_cost": round(_plan["cost"], 3),
                "plan_latency_ms": round(_plan["latency_ms"], 3),
            },
            decisions={
                "checkpoint_policy": _recovery["label"],
                "replication_policy": _plan["label"],
                "rejected_alternative": _rejected_label,
                "uncovered_failure": v2_07_uncovered_failure.value or _recovery["uncovered"],
            },
            reflections={"reliability_memo": v2_07_memo_decision.value},
            final_decision=_decision_text,
            big_takeaways=(
                "Fleet size turns failures into a routine rate, not a rare anecdote.",
                "Checkpoint interval has an optimum because save overhead and lost work pull in opposite directions.",
                "Recovery policy and redundancy are valid only when recovery objective, cost, and performance guardrails pass together.",
                "V2-08 orchestration must schedule the spare capacity and failure-domain placement implied by the reliability memo.",
            ),
            residual_risk=_plan["residual_risk"],
            source_trace={
                "book_anchor": v2_07_metadata.book_anchor,
                "chapter_formulas": (
                    "R_system(t)=exp(-N lambda t)",
                    "MTBF_system=MTBF_component/N",
                    "tau_opt=sqrt(2*T_write*MTBF_system)",
                    "A_system=1-(1-A_single)^k",
                ),
                "track_id": v2_07_profile.track_id,
                "scenario_id": v2_07_variant.scenario_id,
                "variant_source": "get_lab_track_variant",
                "local_helper_prefix": "v2_07_",
            },
            result_snapshot=_snapshot,
            incomplete_fields=tuple(_incomplete),
        )
        return mo.vstack([
            mo.md("## Synthesis - Reliability Memo"),
            v2_07_student_id,
            v2_07_memo_decision,
            v2_07_table(
                ["Memo field", "Selected evidence"],
                [
                    ["Checkpoint policy", f"{_recovery['label']} every {v2_07_fmt(_checkpoint['interval_min'])} min"],
                    ["Replication policy", f"{_plan['label']} with {_plan['replicas']} replicas"],
                    ["Binding failure amount", f"{_binding['label']}: {v2_07_fmt(_binding['value'])} {_binding['unit']} (limit {v2_07_fmt(_binding['limit'])})"],
                    ["Rejected alternative", _rejected_label],
                    ["V2-08 orchestration implication", v2_07_lens["orchestration_implication"]],
                ],
            ),
            mo.callout(mo.md(_decision_text), kind="success" if _plan["feasible"] else "warn"),
            report_export_panel(_report),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Aggregate MTBF": build_part_a(),
        "Part B: Checkpoint Optimum": build_part_b(),
        "Part C: Recovery Policy": build_part_c(),
        "Part D: Guardrail Plan": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


@app.cell(hide_code=True)
def _(mo, v2_07_metadata, v2_07_profile):
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">{v2_07_metadata.lab_id}</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v2_07_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
