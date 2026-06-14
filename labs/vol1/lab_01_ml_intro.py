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
        action_box,
        build_lab_report,
        diagnose_triad,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        intervention_frontier,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        triad_track_profile,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        action_box,
        COLORS,
        LAB_CSS,
        apply_plotly_theme,
        build_lab_report,
        diagnose_triad,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        intervention_frontier,
        ledger,
        mlsysim,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        triad_track_profile,
    )


@app.cell
def _(get_lab_metadata):
    v1_01_metadata = get_lab_metadata("vol1/lab_01_ml_intro.py")
    return (v1_01_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_01_track_picker = track_selector(default=_default_track)
    v1_01_track_picker
    return (v1_01_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    triad_track_profile,
    v1_01_track_picker,
):
    v1_01_track_id = v1_01_track_picker.value
    v1_01_profile = get_track_profile(v1_01_track_id)
    v1_01_variant = get_lab_track_variant("v1_01_ai_triad", v1_01_profile.track_id)
    v1_01_hardware = resolve_mlsysim_ref(v1_01_variant.hardware_ref)
    v1_01_model = resolve_mlsysim_ref(v1_01_variant.model_ref)
    v1_01_triad = triad_track_profile(
        v1_01_profile,
        v1_01_variant,
        v1_01_hardware,
        v1_01_model,
    )
    return (
        v1_01_hardware,
        v1_01_model,
        v1_01_profile,
        v1_01_track_id,
        v1_01_triad,
        v1_01_variant,
    )


@app.cell
def _():
    def v1_01_track_lens(track_id):
        lenses = {
            "iphone": {
                "silent_metric": "privacy-safe UX quality",
                "quality_initial_pct": 94.0,
                "quality_floor_pct": 88.0,
                "degradation_lambda": 12.0,
                "drift_driver": "lighting, accents, accessibility use, and privacy-limited cohorts",
                "silent_consequence": "crash logs stay clean while users see heat, battery drain, and uneven quality",
                "correct_response": "privacy-safe cohort audit",
                "response_options": (
                    "privacy-safe cohort audit",
                    "code rollback",
                    "buy larger cloud GPUs",
                    "ignore until crash logs fire",
                ),
                "pressure_label": "privacy-limited cohort drift",
                "operating_pressure_label": "interactive request pressure",
                "train_amount_label": "privacy-safe training cohort review",
                "train_unit": "k device-days",
                "train_base": 28.0,
                "train_limit": 55.0,
                "infer_amount_label": "p95 UI latency",
                "infer_unit": "ms/use",
                "infer_base": 62.0,
                "infer_limit": 100.0,
                "guardrail_amount_label": "energy per sustained use",
                "guardrail_unit": "mJ",
                "guardrail_base": 140.0,
                "guardrail_limit": 220.0,
                "deployment_failure": "thermal or visible UX responsiveness miss",
                "evidence_packets": (
                    "10-minute sustained run with p95 UI latency and battery trace",
                    "offline validation accuracy only",
                    "training loss curve only",
                ),
                "correct_evidence": "10-minute sustained run with p95 UI latency and battery trace",
                "risk_options": (
                    "privacy-safe cohorts remain under-sampled after the first fix",
                    "thermal soak passes only on a cold device",
                    "unsupported operators silently fall back to CPU",
                ),
            },
            "oura_ring": {
                "silent_metric": "always-on sensing quality",
                "quality_initial_pct": 92.0,
                "quality_floor_pct": 86.0,
                "degradation_lambda": 10.5,
                "drift_driver": "sensor contact, skin temperature, activity mix, and delayed labels",
                "silent_consequence": "firmware keeps running while sleep and recovery estimates become less reliable",
                "correct_response": "sensor-contact cohort audit",
                "response_options": (
                    "sensor-contact cohort audit",
                    "increase model size first",
                    "raise sensing cadence without budget check",
                    "ignore until OTA failure",
                ),
                "pressure_label": "sensor-contact drift",
                "operating_pressure_label": "sensing cadence pressure",
                "train_amount_label": "labeled biosignal coverage",
                "train_unit": "k nights",
                "train_base": 18.0,
                "train_limit": 38.0,
                "infer_amount_label": "wake time per sensor window",
                "infer_unit": "ms/window",
                "infer_base": 38.0,
                "infer_limit": 50.0,
                "guardrail_amount_label": "duty cycle",
                "guardrail_unit": "%",
                "guardrail_base": 2.4,
                "guardrail_limit": 4.0,
                "deployment_failure": "SRAM/flash, cadence, or duty-cycle violation",
                "evidence_packets": (
                    "24-hour duty-cycle replay with SRAM/flash fit",
                    "offline validation accuracy only",
                    "training loss curve only",
                ),
                "correct_evidence": "24-hour duty-cycle replay with SRAM/flash fit",
                "risk_options": (
                    "sensor-contact cohorts remain thin",
                    "OTA payload exceeds the safe flash image budget",
                    "battery regression appears after cadence changes",
                ),
            },
            "robotaxi": {
                "silent_metric": "rare-hazard recall",
                "quality_initial_pct": 96.0,
                "quality_floor_pct": 94.0,
                "degradation_lambda": 8.0,
                "drift_driver": "weather, construction, emergency vehicles, and sensor burst patterns",
                "silent_consequence": "average accuracy stays strong while rare hazards lose safety margin",
                "correct_response": "rare-event replay",
                "response_options": (
                    "rare-event replay",
                    "optimize average accuracy only",
                    "raise fleet speed cap",
                    "ignore until a dashboard crash",
                ),
                "pressure_label": "rare-hazard distribution drift",
                "operating_pressure_label": "sensor burst pressure",
                "train_amount_label": "rare-event replay coverage",
                "train_unit": "k clips",
                "train_base": 32.0,
                "train_limit": 70.0,
                "infer_amount_label": "p99 perception latency",
                "infer_unit": "ms/frame",
                "infer_base": 7.2,
                "infer_limit": 10.0,
                "guardrail_amount_label": "safety margin consumed",
                "guardrail_unit": "%",
                "guardrail_base": 42.0,
                "guardrail_limit": 70.0,
                "deployment_failure": "safety margin or p99 latency miss",
                "evidence_packets": (
                    "rare-event replay with p99/p999 latency and fallback drill",
                    "offline validation accuracy only",
                    "training loss curve only",
                ),
                "correct_evidence": "rare-event replay with p99/p999 latency and fallback drill",
                "risk_options": (
                    "rare-hazard recall remains below the safety case",
                    "p999 latency fails during sensor bursts",
                    "fallback behavior is untested for new construction scenes",
                ),
            },
            "cloud_fleet": {
                "silent_metric": "served quality at SLA",
                "quality_initial_pct": 91.0,
                "quality_floor_pct": 87.0,
                "degradation_lambda": 9.5,
                "drift_driver": "traffic mix, customer cohorts, prompt/query patterns, and label freshness",
                "silent_consequence": "fleet health stays green while cost/request and quality drift out of policy",
                "correct_response": "quality and cost/request canary",
                "response_options": (
                    "quality and cost/request canary",
                    "add capacity before diagnosis",
                    "optimize validation accuracy only",
                    "ignore until instances crash",
                ),
                "pressure_label": "traffic and cohort drift",
                "operating_pressure_label": "QPS/utilization pressure",
                "train_amount_label": "fresh labeled traffic coverage",
                "train_unit": "M examples",
                "train_base": 42.0,
                "train_limit": 95.0,
                "infer_amount_label": "p99 request latency",
                "infer_unit": "ms/request",
                "infer_base": 180.0,
                "infer_limit": 250.0,
                "guardrail_amount_label": "cost per request",
                "guardrail_unit": "milli-$",
                "guardrail_base": 1.2,
                "guardrail_limit": 2.0,
                "deployment_failure": "SLA breach or negative unit economics",
                "evidence_packets": (
                    "load/SLA canary with cost/request and utilization",
                    "offline validation accuracy only",
                    "training loss curve only",
                ),
                "correct_evidence": "load/SLA canary with cost/request and utilization",
                "risk_options": (
                    "cost/request rises after traffic mix changes",
                    "utilization looks good while p99 SLA misses",
                    "carbon budget is exceeded by the chosen capacity plan",
                ),
            },
        }
        return lenses.get(track_id, lenses["iphone"])

    def v1_01_silent_degradation(lens, drift_pressure_pct, months_in_production, monitoring_cadence):
        cadence_weeks = {
            "weekly": 1.0,
            "monthly": 4.0,
            "quarterly": 13.0,
        }.get(monitoring_cadence, 4.0)
        drift = max(0.0, min(100.0, float(drift_pressure_pct))) / 100.0
        months = max(0.0, min(12.0, float(months_in_production)))
        cadence_penalty = max(0.0, (cadence_weeks - 1.0) * 0.08)
        loss = lens["degradation_lambda"] * drift * (months / 12.0) + cadence_penalty
        quality = max(0.0, lens["quality_initial_pct"] - loss)
        floor = lens["quality_floor_pct"]
        timeline = []
        for month in range(0, 13):
            month_loss = lens["degradation_lambda"] * drift * (month / 12.0) + cadence_penalty
            timeline.append({
                "month": month,
                "quality_pct": max(0.0, lens["quality_initial_pct"] - month_loss),
                "floor_pct": floor,
                "code_health_pct": 100.0,
            })
        return {
            "quality_pct": quality,
            "floor_pct": floor,
            "loss_pct": loss,
            "feasible": quality >= floor,
            "months": months,
            "drift_pct": drift * 100.0,
            "cadence_weeks": cadence_weeks,
            "timeline": tuple(timeline),
        }

    def v1_01_amount_system(lens, model_scale_pct, operating_pressure_pct):
        scale = max(50.0, min(150.0, float(model_scale_pct))) / 100.0
        pressure = max(0.0, min(100.0, float(operating_pressure_pct))) / 100.0
        training_amount = lens["train_base"] * scale * (0.85 + pressure * 0.35)
        inference_amount = lens["infer_base"] * scale * (0.70 + pressure * 0.60)
        guardrail_amount = lens["guardrail_base"] * scale * (0.75 + pressure * 0.70)
        training_pass = training_amount <= lens["train_limit"]
        inference_pass = inference_amount <= lens["infer_limit"]
        guardrail_pass = guardrail_amount <= lens["guardrail_limit"]
        return {
            "training_amount": training_amount,
            "training_limit": lens["train_limit"],
            "training_pass": training_pass,
            "inference_amount": inference_amount,
            "inference_limit": lens["infer_limit"],
            "inference_pass": inference_pass,
            "guardrail_amount": guardrail_amount,
            "guardrail_limit": lens["guardrail_limit"],
            "guardrail_pass": guardrail_pass,
            "deployment_ready": training_pass and inference_pass and guardrail_pass,
        }

    def v1_01_comparison_track_id(track_id):
        return {
            "iphone": "oura_ring",
            "oura_ring": "cloud_fleet",
            "robotaxi": "cloud_fleet",
            "cloud_fleet": "oura_ring",
        }.get(track_id, "cloud_fleet")

    def v1_01_axis_margins(result):
        return {
            "Data": result.data_score_pct - result.data_threshold_pct,
            "Algorithm": result.algorithm_score_pct - result.algorithm_threshold_pct,
            "Machine": result.machine_score_pct - result.machine_threshold_pct,
        }

    return (
        v1_01_amount_system,
        v1_01_axis_margins,
        v1_01_comparison_track_id,
        v1_01_silent_degradation,
        v1_01_track_lens,
    )


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    triad_track_profile,
    v1_01_comparison_track_id,
    v1_01_track_id,
):
    v1_01_comparison_id = v1_01_comparison_track_id(v1_01_track_id)
    v1_01_comparison_profile = get_track_profile(v1_01_comparison_id)
    v1_01_comparison_variant = get_lab_track_variant("v1_01_ai_triad", v1_01_comparison_profile.track_id)
    v1_01_comparison_hardware = resolve_mlsysim_ref(v1_01_comparison_variant.hardware_ref)
    v1_01_comparison_model = resolve_mlsysim_ref(v1_01_comparison_variant.model_ref)
    v1_01_comparison_triad = triad_track_profile(
        v1_01_comparison_profile,
        v1_01_comparison_variant,
        v1_01_comparison_hardware,
        v1_01_comparison_model,
    )
    return (
        v1_01_comparison_id,
        v1_01_comparison_profile,
        v1_01_comparison_triad,
        v1_01_comparison_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    track_arc_context,
    v1_01_metadata,
    v1_01_profile,
    v1_01_triad,
    v1_01_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 01
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The AI Triad
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Data &middot; Algorithm &middot; Machine
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 780px; line-height: 1.65;">
                {v1_01_variant.workload_summary} The first engineering decision
                is not which knob to improve; it is which axis is actually binding.
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
                    {v1_01_profile.label}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Silent Degradation</span>
                <span class="badge badge-warn">Binding Axis</span>
                <span class="badge badge-fail">First Fix</span>
            </div>
        </div>
        """),
        track_context(v1_01_profile),
        track_arc_context(v1_01_profile, v1_01_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_01_triad):
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
            <div style="margin-bottom: 3px;">1. <strong>Explain silent degradation:</strong>
                show how learned behavior changes without a code diff.</div>
            <div style="margin-bottom: 3px;">2. <strong>Diagnose the binding axis:</strong>
                separate data coverage, algorithm design, and machine envelope.</div>
            <div style="margin-bottom: 3px;">3. <strong>Separate evidence systems:</strong>
                distinguish training evidence from inference evidence.</div>
            <div style="margin-bottom: 3px;">4. <strong>Defend the first fix:</strong>
                select an intervention, reject alternatives, and name validation evidence.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                "The behavior changed. Which quantity changed, which axis binds, and what first fix is defensible?"
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** - Complete the Introduction chapter's discussion of
    production ML as a system before starting this lab.
    """), kind="info")
    return


# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================


@app.cell(hide_code=True)
def _(action_box, mo, v1_01_track_id, v1_01_track_lens):
    _lens = v1_01_track_lens(v1_01_track_id)
    partA_pred = mo.ui.radio(
        options={
            "A) No, quality only changes after code changes": "code_only",
            "B) Yes, the input distribution can move under fixed code": "silent_drift",
            "C) Only if the server or device crashes": "infrastructure_only",
            "D) Only if the model file is edited": "weights_only",
        },
        label=f"Part A prediction - can {_lens['silent_metric']} fall while code and infrastructure stay fixed?",
    )
    partA_drift = mo.ui.slider(
        start=0,
        stop=100,
        value=55,
        step=5,
        label=f"{_lens['pressure_label']} (%)",
    )
    partA_months = mo.ui.slider(
        start=0,
        stop=12,
        value=6,
        step=1,
        label="Months since deployment",
    )
    partA_cadence = mo.ui.dropdown(
        options={
            "Weekly monitoring": "weekly",
            "Monthly monitoring": "monthly",
            "Quarterly monitoring": "quarterly",
        },
        value="Monthly monitoring",
        label="Monitoring cadence",
    )
    partA_response = action_box(
        mo.ui.radio(
            options={option: option for option in _lens["response_options"]},
            label="",
        ),
        title="Part A Checkpoint - first response",
        body="Choose the response you would defend after the degradation evidence.",
        name="silent_response",
    )
    return (partA_cadence, partA_drift, partA_months, partA_pred, partA_response)


@app.cell(hide_code=True)
def _(action_box, mo, v1_01_triad):
    partB_pred = mo.ui.radio(
        options={
            "A) Data will bind": "Data",
            "B) Algorithm will bind": "Algorithm",
            "C) Machine will bind": "Machine",
            "D) I need evidence before deciding": "Depends",
        },
        label=f"Part B prediction - for {v1_01_triad.label}, which D-A-M axis do you expect to bind?",
    )
    partB_data = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v1_01_triad.default_data_pct),
        step=5,
        label="Data readiness (%)",
    )
    partB_algorithm = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v1_01_triad.default_algorithm_pct),
        step=5,
        label="Algorithm readiness (%)",
    )
    partB_machine = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v1_01_triad.default_machine_pct),
        step=5,
        label="Machine readiness (%)",
    )
    partB_decision = action_box(
        mo.ui.radio(
            options={
                "Data": "Data",
                "Algorithm": "Algorithm",
                "Machine": "Machine",
            },
            label="",
        ),
        title="Part B Checkpoint - final diagnosis",
        body="Choose the binding axis you would defend after reading the evidence and track comparison.",
        name="diagnosis",
    )
    return (partB_algorithm, partB_data, partB_decision, partB_machine, partB_pred)


@app.cell(hide_code=True)
def _(action_box, mo, v1_01_track_id, v1_01_track_lens):
    _lens = v1_01_track_lens(v1_01_track_id)
    partC_pred = mo.ui.radio(
        options={
            "A) Training convergence is enough to ship": "training_only",
            "B) Offline validation accuracy is enough to ship": "offline_accuracy",
            "C) Runtime inference evidence must authorize deployment": "inference_evidence",
            "D) Hardware evidence matters only after launch": "defer_runtime",
        },
        label="Part C prediction - what evidence should authorize deployment?",
    )
    partC_model_scale = mo.ui.slider(
        start=50,
        stop=150,
        value=100,
        step=5,
        label="Model scale (% of baseline)",
    )
    partC_pressure = mo.ui.slider(
        start=0,
        stop=100,
        value=70,
        step=5,
        label=f"{_lens['operating_pressure_label']} (%)",
    )
    partC_evidence = action_box(
        mo.ui.radio(
            options={packet: packet for packet in _lens["evidence_packets"]},
            label="",
        ),
        title="Part C Checkpoint - evidence packet",
        body="Choose the evidence packet that belongs in the triad diagnosis memo.",
        name="evidence_packet",
    )
    return (partC_evidence, partC_model_scale, partC_pred, partC_pressure)


@app.cell(hide_code=True)
def _(mo, v1_01_triad):
    partD_pred = mo.ui.radio(
        options={
            "A) Spend the whole budget on the weakest axis": "weakest",
            "B) Spend evenly across all three axes": "even",
            "C) Spend on hardware first": "hardware",
            "D) Spend on model architecture first": "model",
        },
        label="Part D prediction - how should a fixed lifecycle budget be allocated?",
    )
    partD_data_budget = mo.ui.slider(start=0, stop=100, value=40, step=5, label="Data budget (%)")
    partD_algorithm_budget = mo.ui.slider(start=0, stop=100, value=30, step=5, label="Algorithm budget (%)")
    partD_machine_budget = mo.ui.slider(start=0, stop=100, value=30, step=5, label="Machine budget (%)")
    partD_selected = mo.ui.dropdown(
        options={axis: axis for axis in v1_01_triad.intervention_options or ("Data", "Algorithm", "Machine")},
        value=(v1_01_triad.intervention_options or ("Data",))[0],
        label="Intervention to defend",
    )
    partD_validation = mo.ui.dropdown(
        options={test: test for test in v1_01_triad.validation_tests},
        value=v1_01_triad.validation_tests[0],
        label="Validation evidence",
    )
    return (
        partD_algorithm_budget,
        partD_data_budget,
        partD_machine_budget,
        partD_pred,
        partD_selected,
        partD_validation,
    )


@app.cell(hide_code=True)
def _(mo, v1_01_track_id, v1_01_track_lens):
    _lens = v1_01_track_lens(v1_01_track_id)
    synthesis_risk = mo.ui.radio(
        options={risk: risk for risk in _lens["risk_options"]},
        label="Carry-forward risk for later labs",
    )
    return (synthesis_risk,)


# ===========================================================================
# ZONE C: MAIN LAB
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    diagnose_triad,
    go,
    intervention_frontier,
    mo,
    partA_cadence,
    partA_drift,
    partA_months,
    partA_pred,
    partA_response,
    partB_algorithm,
    partB_data,
    partB_decision,
    partB_machine,
    partB_pred,
    partC_evidence,
    partC_model_scale,
    partC_pred,
    partC_pressure,
    partD_algorithm_budget,
    partD_data_budget,
    partD_machine_budget,
    partD_pred,
    partD_selected,
    partD_validation,
    source_trace,
    synthesis_risk,
    v1_01_amount_system,
    v1_01_axis_margins,
    v1_01_comparison_profile,
    v1_01_comparison_triad,
    v1_01_profile,
    v1_01_silent_degradation,
    v1_01_track_lens,
    v1_01_triad,
    v1_01_variant,
):
    _lens = v1_01_track_lens(v1_01_profile.track_id)

    def _metric_card(label, value, detail, color, border=False):
        border_style = f"2px solid {color}" if border else "1px solid #e2e8f0"
        return f"""
        <div style="padding:16px; border:{border_style}; border-radius:8px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {color}; flex:1;">
            <div style="color:#64748b; font-size:0.78rem; font-weight:700;">{label}</div>
            <div style="font-size:1.35rem; font-weight:800; color:{color}; line-height:1.15;">{value}</div>
            <div style="font-size:0.72rem; color:#64748b; line-height:1.35;">{detail}</div>
        </div>
        """

    def _module_banner(color, background, label, text):
        return mo.Html(f"""
        <div style="border-left:4px solid {color}; background:{background};
                    border-radius:0 8px 8px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                {label}
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "{text}"
            </div>
        </div>
        """)

    def _readiness_chart(diagnosis):
        axes = ["Data", "Algorithm", "Machine"]
        scores = [diagnosis.data_score_pct, diagnosis.algorithm_score_pct, diagnosis.machine_score_pct]
        thresholds = [diagnosis.data_threshold_pct, diagnosis.algorithm_threshold_pct, diagnosis.machine_threshold_pct]
        colors = [COLORS["RedLine"] if axis == diagnosis.binding_axis else COLORS["BlueLine"] for axis in axes]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=axes, y=scores, name="Readiness", marker_color=colors, opacity=0.9))
        fig.add_trace(go.Scatter(
            x=axes,
            y=thresholds,
            name="Track threshold",
            mode="markers+lines",
            line=dict(color=COLORS["OrangeLine"], dash="dash"),
        ))
        fig.update_layout(
            height=340,
            yaxis=dict(title="Readiness (%)", gridcolor="#f1f5f9", range=[0, 105]),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=60, r=20, t=60, b=40),
        )
        apply_plotly_theme(fig)
        return mo.as_html(fig)

    def build_part_a():
        items = [
            _module_banner(
                COLORS["BlueLine"],
                COLORS["BlueL"],
                f"Operations Page - {v1_01_variant.stakeholder}",
                f"{_lens['silent_metric']} is trending down, but code version, model file, and infrastructure health are unchanged.",
            ),
            mo.md(f"""
## Part A: Model Behavior Is Not Normal Software Behavior

**Concept.** In traditional software, behavior usually changes when code changes.
In ML systems, learned behavior can degrade silently because the current input
distribution drifts away from the training distribution.

**Track lens.** For **{v1_01_profile.label}**, watch **{_lens['silent_metric']}**
under {_lens['drift_driver']}. The guardrail is **{v1_01_triad.guardrail_metric}**.
            """),
            partA_pred,
        ]
        if partA_pred.value is None:
            items.append(mo.callout(mo.md("Make the prediction first, then the degradation instrument unlocks."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_drift, partA_months, partA_cadence], widths="equal"))
        silent = v1_01_silent_degradation(_lens, partA_drift.value, partA_months.value, partA_cadence.value)
        months = [row["month"] for row in silent["timeline"]]
        quality = [row["quality_pct"] for row in silent["timeline"]]
        floors = [row["floor_pct"] for row in silent["timeline"]]
        code_health = [row["code_health_pct"] for row in silent["timeline"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=quality, mode="lines+markers", name=_lens["silent_metric"], line=dict(color=COLORS["BlueLine"], width=3)))
        fig.add_trace(go.Scatter(x=months, y=floors, mode="lines", name="quality floor", line=dict(color=COLORS["RedLine"], dash="dash")))
        fig.add_trace(go.Scatter(x=months, y=code_health, mode="lines", name="code/infrastructure health", line=dict(color=COLORS["GreenLine"], dash="dot")))
        fig.add_trace(go.Scatter(x=[silent["months"]], y=[silent["quality_pct"]], mode="markers", name="current month", marker=dict(size=12, color=COLORS["OrangeLine"])))
        fig.update_layout(
            height=360,
            yaxis=dict(title="Percent", gridcolor="#f1f5f9", range=[70, 102]),
            xaxis=dict(title="Months in production", dtick=1),
            legend=dict(orientation="h", y=1.14, x=0),
            margin=dict(l=60, r=20, t=70, b=50),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        status_color = COLORS["GreenLine"] if silent["feasible"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Observed Quality", f"{silent['quality_pct']:.1f}%", _lens["silent_metric"], status_color, True)}
            {_metric_card("Quality Floor", f"{silent['floor_pct']:.1f}%", "track-specific minimum", COLORS["RedLine"])}
            {_metric_card("Silent Loss", f"{silent['loss_pct']:.1f} pp", "without a code diff", COLORS["OrangeLine"])}
            {_metric_card("Code Health", "100%", "unchanged serving path", COLORS["GreenLine"])}
        </div>
        """))

        items.append(mo.md(f"""
**Evidence Table**

| Quantity | Value | Why it matters |
|---|---:|---|
| Drift pressure | {silent['drift_pct']:.0f}% | {_lens['drift_driver']} |
| Months in production | {silent['months']:.0f} | time lets drift accumulate |
| Monitoring cadence | every {silent['cadence_weeks']:.0f} week(s) | slower cadence delays detection |
| {_lens['silent_metric']} | {silent['quality_pct']:.1f}% | quality can move while code is fixed |
| Code/infrastructure health | 100.0% | conventional dashboards can stay green |
        """))

        if silent["feasible"]:
            items.append(mo.callout(mo.md(
                f"**Boundary not crossed yet.** The system still clears the {_lens['quality_floor_pct']:.1f}% floor, "
                "but the chart shows why monitoring must exist before user complaints become the detector."
            ), kind="info"))
        else:
            items.append(mo.callout(mo.md(
                f"**Reversible failure:** {_lens['silent_metric']} is below the {_lens['quality_floor_pct']:.1f}% floor. "
                f"For {v1_01_profile.label}, the consequence is {_lens['silent_consequence']}."
            ), kind="warn"))

        items.append(source_trace({
            "chapter_anchor": "ML vs. Traditional Software; degradation equation",
            "source_model": "Accuracy(t) ~= Accuracy_0 - lambda * D(P_t || P_0)",
            "local_helper": "v1_01_silent_degradation",
            "track_driver": _lens["drift_driver"],
        }, summary="Math Peek: silent degradation model"))
        items.append(partA_response)
        response_payload = partA_response.value if isinstance(partA_response.value, dict) else {}
        response = response_payload.get("silent_response")
        if response == _lens["correct_response"]:
            items.append(mo.callout(mo.md("**Defensible response.** You chose a monitoring/data response for a behavior change that code logs would not expose."), kind="success"))
        elif response is not None:
            items.append(mo.callout(mo.md(
                f"**Reconsider the failure mode.** The evidence points to **{_lens['correct_response']}** before treating this as a normal code or capacity incident."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            _module_banner(
                COLORS["OrangeLine"],
                COLORS["OrangeL"],
                f"Deployment Review - {v1_01_triad.stakeholder}",
                v1_01_triad.failure_story,
            ),
            mo.md(f"""
## Part B: Data, Algorithm, and Machine Are Coupled

**Concept.** The same symptom can come from Data, Algorithm, or Machine. The
first useful intervention targets the axis with the weakest margin to its
track-specific threshold.

- **Data:** {v1_01_triad.data_axis}
- **Algorithm:** {v1_01_triad.algorithm_axis}
- **Machine:** {v1_01_triad.machine_axis}
            """),
            partB_pred,
        ]
        if partB_pred.value is None:
            items.append(mo.callout(mo.md("Make the D-A-M prediction first. The readiness controls stay hidden until you commit."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_data, partB_algorithm, partB_machine], widths="equal"))
        diag = diagnose_triad(
            v1_01_triad,
            data_score_pct=partB_data.value,
            algorithm_score_pct=partB_algorithm.value,
            machine_score_pct=partB_machine.value,
        )
        compare_diag = diagnose_triad(
            v1_01_comparison_triad,
            data_score_pct=partB_data.value,
            algorithm_score_pct=partB_algorithm.value,
            machine_score_pct=partB_machine.value,
        )
        items.append(_readiness_chart(diag))

        status_color = COLORS["GreenLine"] if diag.feasible else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Binding Axis", diag.binding_axis, "weakest margin to threshold", COLORS["RedLine"], True)}
            {_metric_card("Primary Metric", diag.primary_metric, v1_01_profile.label, COLORS["BlueLine"])}
            {_metric_card("Guardrail", diag.guardrail_metric, "must stay protected", COLORS["OrangeLine"])}
            {_metric_card("Status", "PASS" if diag.feasible else "FAIL", ", ".join(diag.violations) or "no violations", status_color, True)}
        </div>
        """))

        selected_margins = v1_01_axis_margins(diag)
        compare_margins = v1_01_axis_margins(compare_diag)
        items.append(mo.md(f"""
**Diagnosis Table**

| Axis | Score | {v1_01_profile.label} threshold | Margin | Meaning |
|---|---:|---:|---:|---|
| Data | {diag.data_score_pct:.0f}% | {diag.data_threshold_pct:.0f}% | {selected_margins['Data']:+.1f} pp | {v1_01_triad.data_axis} |
| Algorithm | {diag.algorithm_score_pct:.0f}% | {diag.algorithm_threshold_pct:.0f}% | {selected_margins['Algorithm']:+.1f} pp | {v1_01_triad.algorithm_axis} |
| Machine | {diag.machine_score_pct:.0f}% | {diag.machine_threshold_pct:.0f}% | {selected_margins['Machine']:+.1f} pp | {v1_01_triad.machine_axis} |

**Track Comparison: Same Scores, Different Envelope**

| Track | Binding axis | Primary metric | Guardrail | Data margin | Algorithm margin | Machine margin |
|---|---|---|---|---:|---:|---:|
| {v1_01_profile.label} | {diag.binding_axis} | {diag.primary_metric} | {diag.guardrail_metric} | {selected_margins['Data']:+.1f} | {selected_margins['Algorithm']:+.1f} | {selected_margins['Machine']:+.1f} |
| {v1_01_comparison_profile.label} | {compare_diag.binding_axis} | {compare_diag.primary_metric} | {compare_diag.guardrail_metric} | {compare_margins['Data']:+.1f} | {compare_margins['Algorithm']:+.1f} | {compare_margins['Machine']:+.1f} |
        """))

        if diag.binding_axis != compare_diag.binding_axis:
            items.append(mo.callout(mo.md(
                f"**Track context changed the answer.** The selected track binds on **{diag.binding_axis}**, "
                f"while {v1_01_comparison_profile.label} binds on **{compare_diag.binding_axis}** under the same raw scores."
            ), kind="info"))
        else:
            items.append(mo.callout(mo.md(
                f"**Same binding axis, different evidence.** Both tracks bind on **{diag.binding_axis}**, but the primary metric and guardrail language differ."
            ), kind="info"))

        items.append(source_trace({
            "chapter_anchor": "D-A-M taxonomy; samples per dollar",
            "source_model": "Cost proportional to (Model Size * Dataset Size) / Hardware Efficiency",
            "shared_helper": "mlsysbook_labs.triad.diagnose_triad",
            "selected_variant": v1_01_variant.scenario_id,
            "comparison_track": v1_01_comparison_profile.track_id,
        }, summary="Math Peek: coupled D-A-M diagnosis"))

        items.append(partB_decision)
        decision_payload = partB_decision.value if isinstance(partB_decision.value, dict) else {}
        decision = decision_payload.get("diagnosis")
        if decision == diag.binding_axis:
            items.append(mo.callout(mo.md("**Good diagnosis.** Your checkpoint matches the evidence and the selected track envelope."), kind="success"))
        elif decision is not None:
            items.append(mo.callout(mo.md(
                f"**Re-check the margin.** You chose {decision}, but the current binding axis is **{diag.binding_axis}**."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            _module_banner(
                COLORS["GreenLine"],
                COLORS["GreenLL"],
                f"Evidence Gate - {v1_01_triad.stakeholder}",
                "The training run completed, but the deployment owner asks what amount system proves the artifact can run.",
            ),
            mo.md(f"""
## Part C: Training and Inference Create Different Amount Systems

**Concept.** Training evidence is not inference evidence. Training asks whether
the system can learn within a throughput or cost envelope. Inference asks whether
the learned artifact can serve within a request, sensor-window, or device budget.

For **{v1_01_profile.label}**, the inference gate is **{_lens['infer_amount_label']}**
plus **{_lens['guardrail_amount_label']}**.
            """),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Commit to an evidence prediction before changing model scale or operating pressure."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_model_scale, partC_pressure], widths="equal"))
        amounts = v1_01_amount_system(_lens, partC_model_scale.value, partC_pressure.value)
        labels = [
            "Training amount",
            "Inference primary",
            "Inference guardrail",
        ]
        pct_of_limit = [
            100 * amounts["training_amount"] / amounts["training_limit"],
            100 * amounts["inference_amount"] / amounts["inference_limit"],
            100 * amounts["guardrail_amount"] / amounts["guardrail_limit"],
        ]
        colors = [
            COLORS["GreenLine"] if amounts["training_pass"] else COLORS["RedLine"],
            COLORS["GreenLine"] if amounts["inference_pass"] else COLORS["RedLine"],
            COLORS["GreenLine"] if amounts["guardrail_pass"] else COLORS["RedLine"],
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=pct_of_limit, marker_color=colors, opacity=0.9, name="Percent of limit"))
        fig.add_trace(go.Scatter(x=labels, y=[100, 100, 100], mode="lines+markers", name="limit", line=dict(color=COLORS["OrangeLine"], dash="dash")))
        fig.update_layout(
            height=340,
            yaxis=dict(title="% of allowed limit", gridcolor="#f1f5f9", range=[0, max(130, max(pct_of_limit) + 15)]),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=60, r=20, t=60, b=40),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        ready_color = COLORS["GreenLine"] if amounts["deployment_ready"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Training Evidence", "PASS" if amounts["training_pass"] else "FAIL", _lens["train_amount_label"], COLORS["GreenLine"] if amounts["training_pass"] else COLORS["RedLine"], True)}
            {_metric_card("Inference Evidence", "PASS" if amounts["inference_pass"] else "FAIL", _lens["infer_amount_label"], COLORS["GreenLine"] if amounts["inference_pass"] else COLORS["RedLine"], True)}
            {_metric_card("Guardrail", "PASS" if amounts["guardrail_pass"] else "FAIL", _lens["guardrail_amount_label"], COLORS["GreenLine"] if amounts["guardrail_pass"] else COLORS["RedLine"], True)}
            {_metric_card("Deployable", "YES" if amounts["deployment_ready"] else "NO", _lens["deployment_failure"], ready_color, True)}
        </div>
        """))

        items.append(mo.md(f"""
**Amount-System Evidence**

| Stage | Quantity | Value | Limit | Unit | Decision |
|---|---|---:|---:|---|---|
| Training | {_lens['train_amount_label']} | {amounts['training_amount']:.1f} | {amounts['training_limit']:.1f} | {_lens['train_unit']} | {"PASS" if amounts['training_pass'] else "FAIL"} |
| Inference | {_lens['infer_amount_label']} | {amounts['inference_amount']:.1f} | {amounts['inference_limit']:.1f} | {_lens['infer_unit']} | {"PASS" if amounts['inference_pass'] else "FAIL"} |
| Inference guardrail | {_lens['guardrail_amount_label']} | {amounts['guardrail_amount']:.1f} | {amounts['guardrail_limit']:.1f} | {_lens['guardrail_unit']} | {"PASS" if amounts['guardrail_pass'] else "FAIL"} |
        """))

        if partC_pred.value == "inference_evidence":
            items.append(mo.callout(mo.md("**Correct evidence standard.** Deployment needs runtime inference evidence, not only a training or offline score."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Training is not the deployment gate.** For {v1_01_profile.label}, the memo must attach runtime evidence for {_lens['infer_amount_label']} and {_lens['guardrail_amount_label']}."
            ), kind="warn"))

        items.append(source_trace({
            "chapter_anchor": "Inference footnote; Iron Law of ML Systems; training-serving divide",
            "source_model": "T ~= D_vol/BW + O/(R_peak * eta_hw) + L_lat",
            "local_helper": "v1_01_amount_system",
            "track_inference_gate": _lens["infer_amount_label"],
        }, summary="Math Peek: training amount vs inference amount"))

        items.append(partC_evidence)
        evidence_payload = partC_evidence.value if isinstance(partC_evidence.value, dict) else {}
        evidence = evidence_payload.get("evidence_packet")
        if evidence == _lens["correct_evidence"]:
            items.append(mo.callout(mo.md("**Memo-ready evidence.** You selected a packet that measures the deployed artifact in its operating envelope."), kind="success"))
        elif evidence is not None:
            items.append(mo.callout(mo.md(
                f"**Evidence gap.** Attach **{_lens['correct_evidence']}** before treating the artifact as deployable."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_d():
        items = [
            _module_banner(
                COLORS["OrangeLine"],
                COLORS["OrangeL"],
                f"Lifecycle Budget - {v1_01_triad.stakeholder}",
                "The next lifecycle loop has one engineering budget. Choose the first defensible fix and name how it could be invalidated.",
            ),
            mo.md("""
## Part D: Lifecycle Decisions Need a First Defensible Fix

**Concept.** Lifecycle decisions are not generic improvement lists. The first fix
must relieve the current binding axis under the selected track's constraints and
must include validation evidence that could overturn the choice.
            """),
            partD_pred,
        ]
        if partD_pred.value is None:
            items.append(mo.callout(mo.md("Predict the lifecycle budget strategy before opening the intervention frontier."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_data_budget, partD_algorithm_budget, partD_machine_budget, partD_selected], widths="equal"))
        frontier = intervention_frontier(
            v1_01_triad,
            data_budget_pct=partD_data_budget.value,
            algorithm_budget_pct=partD_algorithm_budget.value,
            machine_budget_pct=partD_machine_budget.value,
            selected_intervention=partD_selected.value,
        )
        axes = ["Data", "Algorithm", "Machine"]
        scores = [frontier.data_score_pct, frontier.algorithm_score_pct, frontier.machine_score_pct]
        colors = [COLORS["GreenLine"] if axis == frontier.best_intervention else COLORS["BlueLine"] for axis in axes]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=axes, y=scores, marker_color=colors, opacity=0.9, name="Post-intervention readiness"))
        fig.update_layout(
            height=330,
            yaxis=dict(title="Post-intervention readiness (%)", gridcolor="#f1f5f9", range=[0, 105]),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        status_color = COLORS["GreenLine"] if frontier.feasible else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Selected", frontier.selected_intervention, f"margin {frontier.selected_score_pct:+.1f} pp", COLORS["OrangeLine"], True)}
            {_metric_card("Best Axis", frontier.best_intervention, f"margin {frontier.best_score_pct:+.1f} pp", COLORS["GreenLine"])}
            {_metric_card("Binding After Fix", frontier.binding_axis, "remaining weakest axis", COLORS["RedLine"])}
            {_metric_card("Status", "PASS" if frontier.feasible else "FAIL", "all thresholds met" if frontier.feasible else "constraint remains", status_color, True)}
        </div>
        """))

        items.append(mo.md(f"""
**Frontier Table**

| Axis | Budget | Post-score | Interpretation |
|---|---:|---:|---|
| Data | {frontier.data_budget_pct:.0f}% | {frontier.data_score_pct:.1f}% | {v1_01_triad.data_axis} |
| Algorithm | {frontier.algorithm_budget_pct:.0f}% | {frontier.algorithm_score_pct:.1f}% | {v1_01_triad.algorithm_axis} |
| Machine | {frontier.machine_budget_pct:.0f}% | {frontier.machine_score_pct:.1f}% | {v1_01_triad.machine_axis} |

Rejected alternatives for the selected intervention: {", ".join(frontier.rejected_alternatives)}.
        """))

        if not frontier.feasible:
            items.append(mo.callout(mo.md(
                f"**Reversible lifecycle failure:** this budget leaves **{frontier.binding_axis}** below threshold. "
                "Change the budget or selected intervention until the first fix is defensible."
            ), kind="warn"))
        elif frontier.selected_intervention != frontier.best_intervention:
            items.append(mo.callout(mo.md(
                f"**Defend carefully.** The selected fix is {frontier.selected_intervention}, but the frontier says {frontier.best_intervention} has the strongest margin."
            ), kind="warn"))
        else:
            items.append(mo.callout(mo.md("**Defensible first fix.** The selected intervention matches the strongest current frontier evidence."), kind="success"))

        if partD_pred.value == "weakest":
            items.append(mo.callout(mo.md("**Correct principle.** Spend first where the constraint is binding, then validate that the bottleneck did not migrate."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Budget follows diagnosis.** Even spending, hardware-first spending, or architecture-first spending can miss the binding axis."
            ), kind="warn"))

        items.append(partD_validation)
        items.append(source_trace({
            "chapter_anchor": "ML lifecycle; D-A-M bottlenecks migrate; five-pillar ownership",
            "shared_helper": "mlsysbook_labs.triad.intervention_frontier",
            "validation_evidence": partD_validation.value,
            "track_guardrail": v1_01_triad.guardrail_metric,
        }, summary="Math Peek: first fix inside the lifecycle loop"))
        return mo.vstack(items)

    def build_synthesis():
        silent = v1_01_silent_degradation(_lens, partA_drift.value, partA_months.value, partA_cadence.value)
        diag = diagnose_triad(
            v1_01_triad,
            data_score_pct=partB_data.value,
            algorithm_score_pct=partB_algorithm.value,
            machine_score_pct=partB_machine.value,
        )
        amounts = v1_01_amount_system(_lens, partC_model_scale.value, partC_pressure.value)
        frontier = intervention_frontier(
            v1_01_triad,
            data_budget_pct=partD_data_budget.value,
            algorithm_budget_pct=partD_algorithm_budget.value,
            machine_budget_pct=partD_machine_budget.value,
            selected_intervention=partD_selected.value,
        )
        diagnosis_payload = partB_decision.value if isinstance(partB_decision.value, dict) else {}
        evidence_payload = partC_evidence.value if isinstance(partC_evidence.value, dict) else {}
        response_payload = partA_response.value if isinstance(partA_response.value, dict) else {}
        diagnosis_value = diagnosis_payload.get("diagnosis")
        evidence_value = evidence_payload.get("evidence_packet")
        response_value = response_payload.get("silent_response")
        return mo.vstack([
            mo.md("""
## Synthesis: Triad Diagnosis Memo

Record one memo that connects silent model behavior, D-A-M diagnosis,
training/inference evidence, and the first lifecycle fix.
            """),
            synthesis_risk,
            mo.Html(f"""
            <div style="border:1px solid #d9dee8; border-left:4px solid {COLORS['BlueLine']};
                        border-radius:8px; background:white; padding:18px 22px; margin:14px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
                    Memo Snapshot
                </div>
                <p><strong>Track:</strong> {v1_01_profile.label}</p>
                <p><strong>Silent behavior evidence:</strong> {_lens['silent_metric']} is {silent['quality_pct']:.1f}% against a {silent['floor_pct']:.1f}% floor; first response: {response_value or 'not selected'}.</p>
                <p><strong>D-A-M diagnosis:</strong> current evidence binds on {diag.binding_axis}; checkpoint diagnosis: {diagnosis_value or 'not selected'}.</p>
                <p><strong>Training vs inference:</strong> training evidence is {"PASS" if amounts['training_pass'] else "FAIL"}, inference evidence is {"PASS" if amounts['inference_pass'] else "FAIL"}, guardrail evidence is {"PASS" if amounts['guardrail_pass'] else "FAIL"}; packet: {evidence_value or 'not selected'}.</p>
                <p><strong>First fix:</strong> defend {frontier.selected_intervention}; reject {', '.join(frontier.rejected_alternatives)}; validate with {partD_validation.value}.</p>
                <p><strong>Carry-forward risk:</strong> {synthesis_risk.value or 'not selected'}</p>
            </div>
            """),
            mo.callout(mo.md(
                "**Chapter invariant.** A machine learning system does what its data, arithmetic, and hardware permit. "
                "The memo is defensible only if the selected evidence names which quantity changed and which constraint now binds."
            ), kind="info"),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Silent Behavior": build_part_a(),
        "Part B: D-A-M Diagnosis": build_part_b(),
        "Part C: Training vs Inference": build_part_c(),
        "Part D: First Fix": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


# ===========================================================================
# ZONE D: LEDGER HUD AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    diagnose_triad,
    intervention_frontier,
    ledger,
    mo,
    partA_cadence,
    partA_drift,
    partA_months,
    partA_pred,
    partA_response,
    partB_algorithm,
    partB_data,
    partB_decision,
    partB_machine,
    partB_pred,
    partC_evidence,
    partC_model_scale,
    partC_pred,
    partC_pressure,
    partD_algorithm_budget,
    partD_data_budget,
    partD_machine_budget,
    partD_pred,
    partD_selected,
    partD_validation,
    synthesis_risk,
    v1_01_amount_system,
    v1_01_profile,
    v1_01_silent_degradation,
    v1_01_track_lens,
    v1_01_triad,
    v1_01_variant,
):
    _lens = v1_01_track_lens(v1_01_profile.track_id)
    _silent = v1_01_silent_degradation(_lens, partA_drift.value, partA_months.value, partA_cadence.value)
    _diag = diagnose_triad(
        v1_01_triad,
        data_score_pct=partB_data.value,
        algorithm_score_pct=partB_algorithm.value,
        machine_score_pct=partB_machine.value,
    )
    _amounts = v1_01_amount_system(_lens, partC_model_scale.value, partC_pressure.value)
    _frontier = intervention_frontier(
        v1_01_triad,
        data_budget_pct=partD_data_budget.value,
        algorithm_budget_pct=partD_algorithm_budget.value,
        machine_budget_pct=partD_machine_budget.value,
        selected_intervention=partD_selected.value,
    )
    _silent_payload = partA_response.value if isinstance(partA_response.value, dict) else {}
    _diagnosis_payload = partB_decision.value if isinstance(partB_decision.value, dict) else {}
    _evidence_payload = partC_evidence.value if isinstance(partC_evidence.value, dict) else {}
    _silent_response = _silent_payload.get("silent_response")
    _diagnosis_value = _diagnosis_payload.get("diagnosis")
    _evidence_value = _evidence_payload.get("evidence_packet")
    if (
        partA_pred.value is not None
        and _silent_response is not None
        and partB_pred.value is not None
        and _diagnosis_value is not None
        and partC_pred.value is not None
        and _evidence_value is not None
        and partD_pred.value is not None
        and synthesis_risk.value
    ):
        ledger.save(chapter=1, design={
            "chapter": "v1_01",
            "track_id": v1_01_profile.track_id,
            "scenario_id": v1_01_variant.scenario_id,
            "hardware_ref": v1_01_triad.hardware_ref,
            "model_ref": v1_01_triad.model_ref,
            "completed": True,
            "silent_degradation_prediction": partA_pred.value,
            "silent_failure_response": _silent_response,
            "observed_quality_pct": _silent["quality_pct"],
            "predicted_binding_axis": partB_pred.value,
            "triad_final_diagnosis": _diagnosis_value,
            "computed_binding_axis": _diag.binding_axis,
            "training_inference_prediction": partC_pred.value,
            "selected_evidence_packet": _evidence_value,
            "deployment_ready": _amounts["deployment_ready"],
            "budget_prediction": partD_pred.value,
            "selected_intervention": _frontier.selected_intervention,
            "best_intervention": _frontier.best_intervention,
            "rejected_alternatives": _frontier.rejected_alternatives,
            "validation_evidence": partD_validation.value,
            "carry_forward_risk": synthesis_risk.value,
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">01 &middot; AI Triad</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_01_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">ARTIFACT</span>
        <span class="hud-value">{v1_01_triad.report_artifact}</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    diagnose_triad,
    intervention_frontier,
    mo,
    partA_cadence,
    partA_drift,
    partA_months,
    partA_pred,
    partA_response,
    partB_algorithm,
    partB_data,
    partB_decision,
    partB_machine,
    partB_pred,
    partC_evidence,
    partC_model_scale,
    partC_pred,
    partC_pressure,
    partD_algorithm_budget,
    partD_data_budget,
    partD_machine_budget,
    partD_pred,
    partD_selected,
    partD_validation,
    report_export_panel,
    synthesis_risk,
    v1_01_amount_system,
    v1_01_comparison_profile,
    v1_01_metadata,
    v1_01_profile,
    v1_01_silent_degradation,
    v1_01_track_lens,
    v1_01_triad,
    v1_01_variant,
):
    _lens = v1_01_track_lens(v1_01_profile.track_id)
    _silent = v1_01_silent_degradation(_lens, partA_drift.value, partA_months.value, partA_cadence.value)
    _diag = diagnose_triad(
        v1_01_triad,
        data_score_pct=partB_data.value,
        algorithm_score_pct=partB_algorithm.value,
        machine_score_pct=partB_machine.value,
    )
    _amounts = v1_01_amount_system(_lens, partC_model_scale.value, partC_pressure.value)
    _frontier = intervention_frontier(
        v1_01_triad,
        data_budget_pct=partD_data_budget.value,
        algorithm_budget_pct=partD_algorithm_budget.value,
        machine_budget_pct=partD_machine_budget.value,
        selected_intervention=partD_selected.value,
    )
    _silent_payload = partA_response.value if isinstance(partA_response.value, dict) else {}
    _diagnosis_payload = partB_decision.value if isinstance(partB_decision.value, dict) else {}
    _evidence_payload = partC_evidence.value if isinstance(partC_evidence.value, dict) else {}
    _silent_response = _silent_payload.get("silent_response")
    _diagnosis_value = _diagnosis_payload.get("diagnosis")
    _evidence_value = _evidence_payload.get("evidence_packet")
    _incomplete = []
    if partA_pred.value is None:
        _incomplete.append("Part A silent-degradation prediction")
    if _silent_response is None:
        _incomplete.append("Part A first response")
    if partB_pred.value is None:
        _incomplete.append("Part B D-A-M prediction")
    if _diagnosis_value is None:
        _incomplete.append("Part B final binding-axis diagnosis")
    if partC_pred.value is None:
        _incomplete.append("Part C training/inference prediction")
    if _evidence_value is None:
        _incomplete.append("Part C evidence packet")
    if partD_pred.value is None:
        _incomplete.append("Part D lifecycle-budget prediction")
    if not synthesis_risk.value:
        _incomplete.append("Synthesis carry-forward risk")

    _report = build_lab_report(
        v1_01_metadata,
        track=v1_01_profile.label,
        scenario=v1_01_variant.workload_summary,
        learning_objectives=(
            "Explain why learned model behavior can degrade without code changes.",
            "Diagnose the binding Data, Algorithm, or Machine axis under a track-specific envelope.",
            "Separate training evidence from inference evidence using different amount systems.",
            "Defend a first lifecycle fix with rejected alternatives, validation evidence, and carry-forward risk.",
        ),
        predictions={
            "silent_degradation": partA_pred.value,
            "pre_evidence_binding_axis": partB_pred.value,
            "final_binding_axis_diagnosis": _diagnosis_value,
            "training_vs_inference_evidence": partC_pred.value,
            "lifecycle_budget_strategy": partD_pred.value,
        },
        knob_settings={
            "drift_pressure_pct": partA_drift.value,
            "months_in_production": partA_months.value,
            "monitoring_cadence": partA_cadence.value,
            "data_score_pct": partB_data.value,
            "algorithm_score_pct": partB_algorithm.value,
            "machine_score_pct": partB_machine.value,
            "model_scale_pct": partC_model_scale.value,
            "operating_pressure_pct": partC_pressure.value,
            "data_budget_pct": partD_data_budget.value,
            "algorithm_budget_pct": partD_algorithm_budget.value,
            "machine_budget_pct": partD_machine_budget.value,
            "selected_intervention": partD_selected.value,
            "validation_evidence": partD_validation.value,
            "carry_forward_risk": synthesis_risk.value,
        },
        evidence_summary={
            "hardware_ref": v1_01_triad.hardware_ref,
            "model_ref": v1_01_triad.model_ref,
            "silent_metric": _lens["silent_metric"],
            "observed_quality_pct": _silent["quality_pct"],
            "quality_floor_pct": _silent["floor_pct"],
            "silent_failure_response": _silent_response,
            "binding_axis": _diag.binding_axis,
            "diagnosis_feasible": _diag.feasible,
            "violations": _diag.violations,
            "comparison_track": v1_01_comparison_profile.label,
            "training_amount": _amounts["training_amount"],
            "training_pass": _amounts["training_pass"],
            "inference_amount": _amounts["inference_amount"],
            "inference_pass": _amounts["inference_pass"],
            "guardrail_amount": _amounts["guardrail_amount"],
            "guardrail_pass": _amounts["guardrail_pass"],
            "selected_evidence_packet": _evidence_value,
            "frontier_binding_axis": _frontier.binding_axis,
            "selected_intervention": _frontier.selected_intervention,
            "best_intervention": _frontier.best_intervention,
            "rejected_alternatives": _frontier.rejected_alternatives,
        },
        final_decision=(
            f"Defend {_frontier.selected_intervention} first for {v1_01_triad.label}; "
            f"diagnosis binds on {_diag.binding_axis}; attach {_evidence_value or 'runtime evidence'}; "
            f"validate with {partD_validation.value}; reject {', '.join(_frontier.rejected_alternatives)}."
        ),
        big_takeaways=(
            "ML behavior is learned from data and can silently degrade while code remains fixed.",
            "Data, Algorithm, and Machine are coupled; track thresholds decide which axis binds.",
            "Training success and inference readiness use different units and evidence.",
            "The lifecycle memo should name a first fix, rejected alternatives, validation evidence, and a carry-forward risk.",
        ),
        reflections={
            "report_artifact": v1_01_triad.report_artifact,
            "data_axis": v1_01_triad.data_axis,
            "algorithm_axis": v1_01_triad.algorithm_axis,
            "machine_axis": v1_01_triad.machine_axis,
            "primary_metric": v1_01_triad.primary_metric,
            "guardrail_metric": v1_01_triad.guardrail_metric,
            "validation_tests": v1_01_triad.validation_tests,
        },
        residual_risk=synthesis_risk.value or (
            f"The selected first fix may be invalidated if {partD_validation.value} fails "
            f"or if {_frontier.binding_axis} remains below threshold after implementation."
        ),
        source_trace={
            "track_id": v1_01_profile.track_id,
            "scenario_id": v1_01_variant.scenario_id,
            "hardware_ref": v1_01_variant.hardware_ref,
            "model_ref": v1_01_variant.model_ref,
            "shared_helper": "mlsysbook_labs.triad",
            "local_helpers": ("v1_01_silent_degradation", "v1_01_amount_system"),
            "chapter_formulas": ("Accuracy(t) ~= Accuracy_0 - lambda * D(P_t || P_0)", "T ~= D_vol/BW + O/(R_peak * eta_hw) + L_lat"),
            "source_policy": v1_01_profile.source_policy,
        },
        result_snapshot={
            "triad_profile": v1_01_triad,
            "silent_degradation": _silent,
            "diagnosis": _diag,
            "amount_system": _amounts,
            "intervention_frontier": _frontier,
        },
        incomplete_fields=tuple(_incomplete),
    )

    if _incomplete:
        _status = mo.vstack([
            mo.md("## Report Status"),
            mo.callout(
                mo.md(
                    "Finish the required predictions and decisions before downloading "
                    "the Lab 01 triad diagnosis memo."
                ),
                kind="warn",
            ),
            report_export_panel(_report),
        ])
    else:
        _status = mo.vstack([
            mo.md("## Download Report"),
            mo.callout(
                mo.md(
                    "This V1-01 diagnosis memo is generated locally from the selected track, "
                    "your inputs, and the computed evidence."
                ),
                kind="info",
            ),
            report_export_panel(_report),
        ])

    _status
    return


if __name__ == "__main__":
    app.run()
