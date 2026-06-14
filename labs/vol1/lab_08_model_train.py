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
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        training_frontier,
        training_memory_stack,
        training_plan,
        training_track_profile,
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
        ledger,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        training_frontier,
        training_memory_stack,
        training_plan,
        training_track_profile,
    )


@app.cell
def _(get_lab_metadata):
    v1_08_metadata = get_lab_metadata("vol1/lab_08_model_train.py")
    return (v1_08_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_08_track_picker = track_selector(default=_default_track)
    v1_08_track_picker
    return (v1_08_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    training_track_profile,
    v1_08_track_picker,
):
    v1_08_track_id = v1_08_track_picker.value
    v1_08_profile = get_track_profile(v1_08_track_id)
    v1_08_variant = get_lab_track_variant("v1_08_training_gauntlet", v1_08_profile.track_id)
    v1_08_hardware = resolve_mlsysim_ref(v1_08_variant.hardware_ref)
    v1_08_model = resolve_mlsysim_ref(v1_08_variant.model_ref)
    v1_08_training = training_track_profile(
        v1_08_profile,
        v1_08_variant,
        v1_08_hardware,
        v1_08_model,
    )
    return (
        v1_08_hardware,
        v1_08_model,
        v1_08_profile,
        v1_08_track_id,
        v1_08_training,
        v1_08_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
        track_arc_context,
    v1_08_metadata,
    v1_08_profile,
    v1_08_training,
    v1_08_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 08
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Training Gauntlet
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Weights &middot; Gradients &middot; Optimizer State &middot; Activations &middot; Validation
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 860px; line-height: 1.65;">
                {v1_08_variant.workload_summary} This lab asks where training,
                adaptation, or calibration should happen, then checks whether the memory
                stack and validation plan match the selected track.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Memo &middot; ~45 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_08_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_08_training.workload_label}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Batch Frontier</span>
                <span class="badge badge-warn">Memory Budget</span>
                <span class="badge badge-fail">Precision Evidence</span>
                <span class="badge badge-info">Training Memo</span>
            </div>
        </div>
        """),
        track_context(v1_08_profile),
        track_arc_context(v1_08_profile, v1_08_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_08_training):
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
            <div style="margin-bottom: 3px;">1. <strong>Reason about batch size:</strong>
                compare throughput, memory pressure, and convergence evidence.</div>
            <div style="margin-bottom: 3px;">2. <strong>Build the training memory stack:</strong>
                compare weights, gradients, optimizer state, activations, and data batch memory.</div>
            <div style="margin-bottom: 3px;">3. <strong>Check precision policy:</strong>
                weigh memory/throughput gains against stability evidence.</div>
            <div style="margin-bottom: 3px;">4. <strong>Choose a training plan:</strong>
                satisfy cost, time, memory, validation, and deployment handoff constraints.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                Training is budgeted optimization on {v1_08_training.label}: which
                batch, precision, memory budget, and validation evidence make the
                training or adaptation plan defensible?
            </div>
            <div style="font-size: 0.88rem; color: {COLORS['TextSec']};
                        line-height: 1.6; margin-top: 10px;">
                Every track follows the same four concepts. The selected track changes
                persona, constraints, thresholds, evidence emphasis, failure mode, and
                report framing.
            </div>
        </div>
    </div>
    """)
    return


# ===========================================================================
# ZONE B: CONTROLS AND COMPUTATION
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_08_training):
    v1_08_batch_prediction = mo.ui.radio(
        options={
            "The largest feasible batch is always the best choice": "largest_feasible",
            "The smallest batch is safest because it uses less memory": "smallest_safe",
            "Batch size must balance utilization, memory, and convergence evidence": "balanced",
            "Batch only changes wall-clock speed": "speed_only",
        },
        label=f"{v1_08_training.label}: what will batch size change besides raw speed?",
    )
    v1_08_batch_checkpoint = mo.ui.radio(
        options={
            "Keep the current physical batch": "keep_batch",
            "Lower physical batch and use accumulation": "lower_and_accumulate",
            "Move training upstream and keep local validation": "upstream_local_validation",
            "Block until convergence evidence is available": "block_for_convergence",
        },
        label="Checkpoint: what batch decision belongs in the memo?",
    )
    return (v1_08_batch_checkpoint, v1_08_batch_prediction)


@app.cell(hide_code=True)
def _(mo, v1_08_training):
    v1_08_memory_prediction = mo.ui.radio(
        options={
            "Weights dominate": "weights",
            "Gradients dominate": "gradients",
            "Optimizer state dominates": "optimizer state",
            "Activations dominate": "activations",
            "Data batch dominates": "data batch",
        },
        label=f"{v1_08_training.label}: which training memory component do you expect to dominate?",
    )
    return (v1_08_memory_prediction,)


@app.cell(hide_code=True)
def _(mo, v1_08_training):
    v1_08_batch_size = mo.ui.slider(
        start=v1_08_training.batch_min,
        stop=v1_08_training.batch_max,
        value=v1_08_training.default_batch_size,
        step=v1_08_training.batch_step,
        label="Batch size",
    )
    return (v1_08_batch_size,)


@app.cell(hide_code=True)
def _(mo, v1_08_training):
    _strategy_options = {
        strategy.label: strategy.strategy_id
        for strategy in v1_08_training.strategy_options
    }
    v1_08_strategy_choice = mo.ui.dropdown(
        options=_strategy_options,
        value=v1_08_training.strategy_options[0].label,
        label="Training/adaptation strategy",
    )
    v1_08_memory_mitigation = mo.ui.radio(
        options={
            "Reduce physical batch or use accumulation": "reduce_batch",
            "Reduce trainable fraction or freeze layers": "reduce_trainable",
            "Use checkpointing to trade recompute for activation memory": "checkpoint",
            "Move training upstream and validate locally": "move_upstream",
        },
        label="Checkpoint: which mitigation addresses the binding memory term?",
    )
    v1_08_precision_prediction = mo.ui.radio(
        options={
            "Lower precision is always safe if the memory number improves": "always_safe",
            "Reduced precision improves memory/throughput but requires stability evidence": "needs_evidence",
            "Precision only affects stored model weights": "weights_only",
            "Precision is irrelevant because training is upstream": "irrelevant",
        },
        label="Prediction: what changes when the precision policy changes?",
    )
    v1_08_precision_policy = mo.ui.radio(
        options={
            "FP32 stability baseline": "fp32_stability",
            "FP16 loss-scaled mixed precision": "fp16_loss_scaled",
            "BF16 mixed precision": "bf16_mixed",
            "FP8 experimental path": "fp8_experimental",
        },
        label="Precision policy to test",
    )
    v1_08_precision_checkpoint = mo.ui.radio(
        options={
            "FP32/BF16 parity replay": "parity_replay",
            "loss-scale and NaN telemetry": "loss_scale_telemetry",
            "rare-event or edge-case validation": "rare_event_validation",
            "deployment-conversion numeric diff": "conversion_diff",
        },
        label="Checkpoint: which evidence must accompany reduced precision?",
    )
    v1_08_plan_prediction = mo.ui.radio(
        options={
            "Memory capacity": "memory",
            "Training time, cost, or carbon": "time_cost",
            "Validation/evidence coverage": "validation",
            "Deployment handoff or rollback": "handoff",
        },
        label="Prediction: which constraint will rule out the naive plan?",
    )
    v1_08_plan_checkpoint = mo.ui.radio(
        options={
            "Approve selected plan with listed validation": "approve_with_validation",
            "Reduce scope and keep only adaptation/calibration": "reduce_scope",
            "Move training upstream and keep local validation": "upstream_train_local_validate",
            "Block release until evidence gap closes": "block_for_evidence",
        },
        label="Checkpoint: what should the memo recommend?",
    )
    v1_08_reflection = mo.ui.text_area(
        label="Training plan memo reflection",
        placeholder="Name the binding resource, the evidence number, and the deployment implication that carries forward.",
        full_width=True,
    )
    return (
        v1_08_memory_mitigation,
        v1_08_plan_checkpoint,
        v1_08_plan_prediction,
        v1_08_precision_checkpoint,
        v1_08_precision_policy,
        v1_08_precision_prediction,
        v1_08_reflection,
        v1_08_strategy_choice,
    )


@app.cell
def _():
    def v1_08_strategy_by_id(profile, strategy_id):
        for strategy in profile.strategy_options:
            if strategy.strategy_id == strategy_id:
                return strategy
        return profile.strategy_options[0]

    def v1_08_track_amount_system(profile):
        notes = {
            "iphone": {
                "amount_system": "app memory, thermal headroom, battery drain, privacy-local data, and adapter storage",
                "batch_consequence": "A local batch that looks feasible can still burn thermal and battery budget or overfit one user's contexts.",
                "precision_consequence": "Reduced precision must preserve the converted CoreML behavior and privacy-safe local replay evidence.",
                "plan_consequence": "Full training is upstream; only lightweight local personalization is a defensible device activity.",
                "carry_forward": "Deployment must carry adapter rollback, privacy-safe telemetry, and local validation gates.",
            },
            "oura_ring": {
                "amount_system": "SRAM, flash, OTA payload, duty cycle, battery life, and scarce biosignal labels",
                "batch_consequence": "The ring can collect biosignals, but a training batch competes with always-on sensing and firmware memory.",
                "precision_consequence": "Precision policy is mainly an upstream-training and OTA-artifact choice; firmware must still pass battery and signal replay.",
                "plan_consequence": "Full firmware training is rejected; cloud/offline training plus tiny calibration is the realistic plan.",
                "carry_forward": "Deployment must carry SRAM trace, OTA payload size, and battery-regression evidence.",
            },
            "robotaxi": {
                "amount_system": "rare-event coverage, p99/p999 replay time, safety compute, route evidence, and fallback reliability",
                "batch_consequence": "Bigger fleet batches can improve throughput while diluting or delaying rare-event evidence.",
                "precision_consequence": "Reduced precision needs rare-event replay because numerical drift can hide in safety tails.",
                "plan_consequence": "Training belongs in the fleet/simulation pipeline; the vehicle should validate safety behavior locally.",
                "carry_forward": "Deployment must carry rare-event replay, fallback drill, and route-specific regression evidence.",
            },
            "cloud_fleet": {
                "amount_system": "accelerator HBM, throughput, wall-clock time, cost, utilization, carbon, and serving canary evidence",
                "batch_consequence": "Large batches improve utilization until memory, learning-rate schedule, or convergence evidence becomes binding.",
                "precision_consequence": "Mixed precision can reduce cost and carbon, but FP16/BF16/FP8 choices need stability evidence.",
                "plan_consequence": "Full training is central, but the plan must pay for accelerator memory, recompute, checkpointing, and canary gates.",
                "carry_forward": "Deployment must carry trained-checkpoint promotion, cost/time evidence, and serving rollback gates.",
            },
        }
        return notes.get(profile.track_id, notes["iphone"])

    def v1_08_batch_status(profile, point):
        if not point.feasible:
            return (
                "blocked",
                "Memory or throughput boundary crossed; lower physical batch, accumulate gradients, or change strategy.",
            )
        if point.throughput_samples_s < profile.throughput_budget_samples_s * 1.25:
            return (
                "low-throughput",
                "The batch fits but leaves little utilization margin against the track throughput target.",
            )
        high_batch = point.batch_size >= max(profile.batch_min, int(profile.batch_max * 0.75))
        if high_batch and profile.track_id == "cloud_fleet":
            return (
                "convergence validation",
                "Large effective batch needs learning-rate, warmup, and convergence validation.",
            )
        if high_batch and profile.track_id == "robotaxi":
            return (
                "evidence dilution",
                "Rare-event examples must stay represented when fleet batches grow.",
            )
        if high_batch and profile.track_id in {"iphone", "oura_ring"}:
            return (
                "local-budget caution",
                "Device-side training/adaptation still competes with local data, energy, and update budgets.",
            )
        return ("candidate", "Feasible at this point, pending the track validation evidence.")

    def v1_08_batch_rows(profile, frontier):
        rows = []
        for point in frontier.points:
            status, consequence = v1_08_batch_status(profile, point)
            rows.append({
                "batch_size": point.batch_size,
                "total_mb": point.total_mb,
                "throughput_samples_s": point.throughput_samples_s,
                "feasible": point.feasible,
                "status": status,
                "consequence": consequence,
            })
        return tuple(rows)

    def v1_08_batch_consequence(profile, selected_stack, frontier):
        amount_system = v1_08_track_amount_system(profile)
        if selected_stack.violations:
            consequence = "; ".join(selected_stack.violations)
            mitigation = "Lower physical batch, change strategy, or move training upstream."
        elif (
            frontier.first_infeasible_batch is not None
            and selected_stack.batch_size == frontier.max_feasible_batch
        ):
            consequence = (
                f"Batch {selected_stack.batch_size} is the last feasible point before "
                f"batch {frontier.first_infeasible_batch} crosses the boundary."
            )
            mitigation = "Keep this batch only with validation margin; otherwise use accumulation."
        else:
            consequence = amount_system["batch_consequence"]
            mitigation = "Preserve evidence that the batch choice still satisfies the track guardrail."
        return {
            "consequence": consequence,
            "mitigation": mitigation,
            "amount_system": amount_system["amount_system"],
        }

    def v1_08_precision_policy_rows(profile, selected_stack, strategy):
        policies = (
            {
                "policy_id": "fp32_stability",
                "label": "FP32 stability baseline",
                "bytes": 4,
                "throughput_factor": 0.55,
                "stability_risk": "low",
                "evidence": "FP32 baseline replay and convergence curve",
            },
            {
                "policy_id": "fp16_loss_scaled",
                "label": "FP16 loss-scaled mixed precision",
                "bytes": 2,
                "throughput_factor": 1.18,
                "stability_risk": "medium",
                "evidence": "loss-scale telemetry, NaN checks, and representative replay",
            },
            {
                "policy_id": "bf16_mixed",
                "label": "BF16 mixed precision",
                "bytes": 2,
                "throughput_factor": 1.14,
                "stability_risk": "low-medium",
                "evidence": "BF16 vs FP32 parity replay on representative data",
            },
            {
                "policy_id": "fp8_experimental",
                "label": "FP8 experimental path",
                "bytes": 1,
                "throughput_factor": 1.42,
                "stability_risk": "high",
                "evidence": "per-tensor scaling audit, holdout convergence, and canary replay",
            },
        )
        base_bytes = max(1, strategy.precision_bytes)
        rows = []
        for policy in policies:
            byte_factor = policy["bytes"] / base_bytes
            weights_mb = selected_stack.weights_mb * byte_factor
            gradients_mb = selected_stack.gradients_mb * byte_factor
            activations_mb = selected_stack.activations_mb * byte_factor
            total_mb = (
                weights_mb
                + gradients_mb
                + selected_stack.optimizer_mb
                + activations_mb
                + selected_stack.data_batch_mb
            )
            throughput = selected_stack.throughput_samples_s * policy["throughput_factor"]
            memory_ok = total_mb <= selected_stack.budget_mb
            throughput_ok = throughput >= profile.throughput_budget_samples_s
            support_ok = not (
                policy["policy_id"] == "fp8_experimental"
                and profile.track_id != "cloud_fleet"
            )
            if not support_ok:
                status = "blocked by track/hardware fit"
            elif not memory_ok:
                status = "blocked by memory"
            elif not throughput_ok:
                status = "blocked by throughput"
            elif policy["stability_risk"] == "high":
                status = "candidate only with strong stability evidence"
            else:
                status = "candidate with validation"
            rows.append({
                "policy_id": policy["policy_id"],
                "label": policy["label"],
                "bytes": policy["bytes"],
                "total_mb": total_mb,
                "throughput_samples_s": throughput,
                "memory_ok": memory_ok,
                "throughput_ok": throughput_ok,
                "support_ok": support_ok,
                "status": status,
                "stability_risk": policy["stability_risk"],
                "evidence": policy["evidence"],
            })
        return tuple(rows)

    def v1_08_precision_selection(rows, selected_policy):
        selected = selected_policy or "bf16_mixed"
        for row in rows:
            if row["policy_id"] == selected:
                return row
        return rows[0]

    def v1_08_binding_resource(profile, selected_stack, plan, precision_selected):
        if not plan.feasible:
            if any("memory" in item for item in selected_stack.violations):
                return "training memory budget"
            if any("throughput" in item for item in selected_stack.violations):
                return "throughput/time budget"
            return "training feasibility"
        if "blocked" in precision_selected["status"]:
            return "precision support or stability evidence"
        if profile.track_id == "robotaxi":
            return "safety validation evidence"
        if profile.track_id == "cloud_fleet":
            return "accelerator memory, training time, cost, and carbon"
        if profile.track_id == "oura_ring":
            return "SRAM/OTA deployment budget"
        if profile.track_id == "iphone":
            return "thermal, privacy, and local validation evidence"
        return "deployment evidence"

    def v1_08_memo_evidence_number(selected_stack, precision_selected):
        return (
            f"{selected_stack.total_mb:.2f} MB selected-strategy memory; "
            f"{precision_selected['total_mb']:.2f} MB under {precision_selected['label']}; "
            f"{selected_stack.throughput_samples_s:.2f} samples/s current throughput"
        )

    def v1_08_carry_forward_summary(profile, plan, precision_selected):
        amount_system = v1_08_track_amount_system(profile)
        return (
            f"{amount_system['carry_forward']} Selected plan: {plan.selected_label}. "
            f"Precision evidence: {precision_selected['evidence']}."
        )

    return (
        v1_08_batch_consequence,
        v1_08_batch_rows,
        v1_08_binding_resource,
        v1_08_carry_forward_summary,
        v1_08_memo_evidence_number,
        v1_08_precision_policy_rows,
        v1_08_precision_selection,
        v1_08_strategy_by_id,
        v1_08_track_amount_system,
    )


@app.cell
def _(
    training_frontier,
    training_memory_stack,
    training_plan,
    v1_08_batch_consequence,
    v1_08_batch_rows,
    v1_08_binding_resource,
    v1_08_carry_forward_summary,
    v1_08_memo_evidence_number,
    v1_08_precision_policy,
    v1_08_precision_policy_rows,
    v1_08_precision_selection,
    v1_08_strategy_by_id,
    v1_08_track_amount_system,
    v1_08_batch_size,
    v1_08_strategy_choice,
    v1_08_training,
):
    v1_08_selected_strategy = v1_08_strategy_by_id(
        v1_08_training,
        v1_08_strategy_choice.value,
    )
    v1_08_memory_rows = tuple(
        training_memory_stack(
            v1_08_training,
            strategy_id=strategy.strategy_id,
            batch_size=v1_08_batch_size.value,
        )
        for strategy in v1_08_training.strategy_options
    )
    v1_08_frontier = training_frontier(
        v1_08_training,
        strategy_id=v1_08_strategy_choice.value,
    )
    v1_08_plan = training_plan(
        v1_08_training,
        strategy_id=v1_08_strategy_choice.value,
        batch_size=v1_08_batch_size.value,
    )
    v1_08_selected_stack = next(
        row for row in v1_08_memory_rows
        if row.strategy_id == v1_08_plan.selected_id
    )
    v1_08_batch_rows_current = v1_08_batch_rows(v1_08_training, v1_08_frontier)
    v1_08_batch_consequence_current = v1_08_batch_consequence(
        v1_08_training,
        v1_08_selected_stack,
        v1_08_frontier,
    )
    v1_08_precision_rows = v1_08_precision_policy_rows(
        v1_08_training,
        v1_08_selected_stack,
        v1_08_selected_strategy,
    )
    v1_08_precision_selected = v1_08_precision_selection(
        v1_08_precision_rows,
        v1_08_precision_policy.value,
    )
    v1_08_amount_system = v1_08_track_amount_system(v1_08_training)
    v1_08_binding_resource_current = v1_08_binding_resource(
        v1_08_training,
        v1_08_selected_stack,
        v1_08_plan,
        v1_08_precision_selected,
    )
    v1_08_memo_evidence = v1_08_memo_evidence_number(
        v1_08_selected_stack,
        v1_08_precision_selected,
    )
    v1_08_carry_forward = v1_08_carry_forward_summary(
        v1_08_training,
        v1_08_plan,
        v1_08_precision_selected,
    )
    return (
        v1_08_amount_system,
        v1_08_batch_consequence_current,
        v1_08_batch_rows_current,
        v1_08_binding_resource_current,
        v1_08_carry_forward,
        v1_08_frontier,
        v1_08_memo_evidence,
        v1_08_memory_rows,
        v1_08_plan,
        v1_08_precision_rows,
        v1_08_precision_selected,
        v1_08_selected_stack,
        v1_08_selected_strategy,
    )


# ===========================================================================
# ZONE C: PARTS
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    v1_08_amount_system,
    v1_08_batch_checkpoint,
    v1_08_batch_consequence_current,
    v1_08_batch_prediction,
    v1_08_batch_rows_current,
    v1_08_batch_size,
    v1_08_binding_resource_current,
    v1_08_frontier,
    v1_08_memory_mitigation,
    v1_08_memory_prediction,
    v1_08_memory_rows,
    v1_08_plan,
    v1_08_plan_checkpoint,
    v1_08_plan_prediction,
    v1_08_precision_checkpoint,
    v1_08_precision_policy,
    v1_08_precision_prediction,
    v1_08_precision_rows,
    v1_08_precision_selected,
    v1_08_reflection,
    v1_08_selected_stack,
    v1_08_strategy_choice,
    v1_08_training,
):
    _batch_fig = go.Figure()
    _batch_fig.add_trace(go.Scatter(
        x=[row["batch_size"] for row in v1_08_batch_rows_current],
        y=[row["throughput_samples_s"] for row in v1_08_batch_rows_current],
        mode="lines+markers",
        marker=dict(
            color=[
                COLORS["GreenLine"] if row["feasible"] else COLORS["RedLine"]
                for row in v1_08_batch_rows_current
            ],
            size=[
                11 if row["batch_size"] == v1_08_batch_size.value else 7
                for row in v1_08_batch_rows_current
            ],
        ),
        line=dict(color=COLORS["BlueLine"], width=2.5),
        name="Throughput",
    ))
    _batch_fig.add_hline(
        y=v1_08_training.throughput_budget_samples_s,
        line_dash="dash",
        line_color=COLORS["OrangeLine"],
        annotation_text="throughput target",
        annotation_font_color=COLORS["OrangeLine"],
    )
    _batch_fig.update_layout(
        height=350,
        xaxis=dict(title="Batch size", gridcolor="#f1f5f9"),
        yaxis=dict(title="Throughput (samples/s)", gridcolor="#f1f5f9"),
        margin=dict(l=70, r=20, t=35, b=50),
    )
    apply_plotly_theme(_batch_fig)

    _component_fig = go.Figure()
    _component_fig.add_trace(go.Bar(
        x=["Weights", "Gradients", "Optimizer", "Activations", "Data batch"],
        y=[
            v1_08_selected_stack.weights_mb,
            v1_08_selected_stack.gradients_mb,
            v1_08_selected_stack.optimizer_mb,
            v1_08_selected_stack.activations_mb,
            v1_08_selected_stack.data_batch_mb,
        ],
        marker_color=[
            COLORS["BlueLine"],
            COLORS["OrangeLine"],
            COLORS["RedLine"],
            COLORS["GreenLine"],
            COLORS["Cloud"],
        ],
        text=[
            f"{v1_08_selected_stack.weights_mb:.1f}",
            f"{v1_08_selected_stack.gradients_mb:.1f}",
            f"{v1_08_selected_stack.optimizer_mb:.1f}",
            f"{v1_08_selected_stack.activations_mb:.1f}",
            f"{v1_08_selected_stack.data_batch_mb:.1f}",
        ],
        textposition="outside",
    ))
    _component_fig.add_hline(
        y=v1_08_selected_stack.budget_mb,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        annotation_text="memory budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _component_fig.update_layout(
        height=350,
        xaxis=dict(title="Memory component", gridcolor="#f1f5f9"),
        yaxis=dict(title="Memory (MB)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=60),
    )
    apply_plotly_theme(_component_fig)

    _precision_fig = go.Figure()
    _precision_fig.add_trace(go.Bar(
        x=[row["label"] for row in v1_08_precision_rows],
        y=[row["total_mb"] for row in v1_08_precision_rows],
        marker_color=[
            COLORS["BlueLine"] if row["policy_id"] == v1_08_precision_selected["policy_id"]
            else COLORS["RedLine"] if "blocked" in row["status"]
            else COLORS["Cloud"]
            for row in v1_08_precision_rows
        ],
        text=[f"{row['total_mb']:.1f}" for row in v1_08_precision_rows],
        textposition="outside",
        name="Precision memory",
    ))
    _precision_fig.add_hline(
        y=v1_08_selected_stack.budget_mb,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        annotation_text="memory budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _precision_fig.update_layout(
        height=360,
        xaxis=dict(title="Precision policy", gridcolor="#f1f5f9"),
        yaxis=dict(title="Estimated memory (MB)", gridcolor="#f1f5f9"),
        margin=dict(l=70, r=20, t=35, b=90),
    )
    apply_plotly_theme(_precision_fig)

    _strategy_details = {strategy.strategy_id: strategy for strategy in v1_08_training.strategy_options}
    _memory_rows = "".join(
        f"""
        <tr>
          <td>{row.strategy_label}</td>
          <td>{row.weights_mb:.2f}</td>
          <td>{row.gradients_mb:.2f}</td>
          <td>{row.optimizer_mb:.2f}</td>
          <td>{row.activations_mb:.2f}</td>
          <td>{row.data_batch_mb:.2f}</td>
          <td>{row.total_mb:.2f} MB</td>
          <td>{row.budget_mb:.2f} MB</td>
          <td>{row.memory_utilization_pct:.1f}%</td>
          <td>{row.throughput_samples_s:.2f}</td>
          <td>{'yes' if row.feasible else 'no - violation'}</td>
          <td>{row.dominant_component}</td>
        </tr>
        """
        for row in v1_08_memory_rows
    )

    _batch_table_rows = "".join(
        f"""
        <tr>
          <td>{row["batch_size"]}</td>
          <td>{row["total_mb"]:.2f} MB</td>
          <td>{row["throughput_samples_s"]:.2f}</td>
          <td>{'yes' if row["feasible"] else 'no - violation'}</td>
          <td>{row["status"]}</td>
          <td>{row["consequence"]}</td>
        </tr>
        """
        for row in v1_08_batch_rows_current
    )
    _precision_rows = "".join(
        f"""
        <tr>
          <td>{row["label"]}</td>
          <td>{row["bytes"]}</td>
          <td>{row["total_mb"]:.2f} MB</td>
          <td>{row["throughput_samples_s"]:.2f}</td>
          <td>{row["stability_risk"]}</td>
          <td>{row["status"]}</td>
          <td>{row["evidence"]}</td>
        </tr>
        """
        for row in v1_08_precision_rows
    )
    _plan_rows = "".join(
        f"""
        <tr>
          <td>{'selected' if row.strategy_id == v1_08_plan.selected_id else 'alternative'}</td>
          <td>{row.strategy_label}</td>
          <td>{_strategy_details[row.strategy_id].training_location}</td>
          <td>{_strategy_details[row.strategy_id].validation_location}</td>
          <td>{row.total_mb:.2f} MB</td>
          <td>{row.dominant_component}</td>
          <td>{'yes' if row.feasible else 'no - violation'}</td>
          <td>{_strategy_details[row.strategy_id].hidden_cost}</td>
        </tr>
        """
        for row in v1_08_memory_rows
    )
    _validation_items = "".join(f"<li>{test}</li>" for test in v1_08_training.validation_tests)
    _rejections = "".join(f"<li>{item}</li>" for item in v1_08_plan.rejected_alternatives)
    _prediction_value = lambda widget: widget.value if widget.value is not None else "not selected yet"

    _part_a = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Batch Size Changes Throughput And Convergence</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            What does batch size buy, and what does it put at risk for {v1_08_training.workload_label}?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Scenario</h2>
          <p><strong>{v1_08_training.stakeholder}:</strong> choose a physical batch for {v1_08_training.label}.</p>
          <ul class="mlsysbook-list">
            <li><strong>Amount system:</strong> {v1_08_amount_system["amount_system"]}.</li>
            <li>Batch size affects hardware utilization, activation memory, and convergence evidence.</li>
            <li>The selected strategy max feasible batch is {v1_08_frontier.max_feasible_batch or 'none'}.</li>
          </ul>
        </div>
        """),
        v1_08_batch_prediction,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Structured Prediction</h2>
          <div class="mlsysbook-callout"><strong>Your prediction:</strong> {_prediction_value(v1_08_batch_prediction)}</div>
          <div class="mlsysbook-callout"><strong>Actual instrument:</strong>
            batch {v1_08_batch_size.value} produces {v1_08_selected_stack.throughput_samples_s:.2f}
            samples/s and {v1_08_selected_stack.total_mb:.2f} MB of training memory.</div>
        </div>
        """),
        v1_08_batch_size,
        mo.as_html(_batch_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Evidence Table</h2>
          <table class="mlsysbook-table">
            <thead><tr><th>Batch</th><th>Total memory</th><th>Throughput</th><th>Feasible</th><th>Status</th><th>Consequence</th></tr></thead>
            <tbody>{_batch_table_rows}</tbody>
          </table>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Consequence And Boundary</h2>
          <div class="mlsysbook-callout"><strong>Consequence:</strong> {v1_08_batch_consequence_current["consequence"]}</div>
          <div class="mlsysbook-callout"><strong>Mitigation:</strong> {v1_08_batch_consequence_current["mitigation"]}</div>
          <div class="mlsysbook-callout"><strong>Math Peek:</strong>
            <code>T_train = O / (R_peak * eta_hw)</code>. Batch size can improve
            utilization <code>eta_hw</code>, but it also changes activation memory
            and the convergence evidence required by the track.</div>
        </div>
        """),
        v1_08_batch_checkpoint,
    ])

    _part_b = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Optimizer State And Activations Create A Memory Budget</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which training memory component dominates {v1_08_training.workload_label}?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Scenario</h2>
          <p>{v1_08_training.training_story}</p>
          <ul class="mlsysbook-list">
            <li>Training adds gradients, optimizer state, activations, and data batch memory on top of inference weights.</li>
            <li>Fine-tuning and adaptation reduce the trainable fraction, but they still need validation and rollback plans.</li>
            <li>Full local training can be the wrong activity even when local inference fits.</li>
          </ul>
        </div>
        """),
        v1_08_memory_prediction,
        v1_08_strategy_choice,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Prediction Vs Actual</h2>
          <div class="mlsysbook-callout"><strong>Your prediction:</strong> {_prediction_value(v1_08_memory_prediction)}</div>
          <div class="mlsysbook-callout"><strong>Actual dominant component:</strong>
            {v1_08_selected_stack.dominant_component} at {v1_08_selected_stack.total_mb:.2f} MB total
            against a {v1_08_selected_stack.budget_mb:.2f} MB budget.</div>
        </div>
        """),
        v1_08_batch_size,
        mo.as_html(_component_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Strategy Memory Table</h2>
          <table class="mlsysbook-table">
            <thead>
              <tr>
                <th>Strategy</th><th>Weights</th><th>Gradients</th><th>Optimizer</th><th>Activations</th><th>Data</th>
                <th>Total</th><th>Budget</th><th>Utilization</th>
                <th>Throughput</th><th>Feasible</th><th>Dominant component</th>
              </tr>
            </thead>
            <tbody>{_memory_rows}</tbody>
          </table>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-callout"><strong>Failure boundary:</strong>
            {'; '.join(v1_08_selected_stack.violations) if v1_08_selected_stack.violations else 'No current violation; preserve margin before increasing batch or trainable fraction.'}</div>
          <div class="mlsysbook-callout"><strong>Math Peek:</strong>
            <code>Total Memory = weights + gradients + optimizer + activations + batch data</code>.
            Activations scale with batch; optimizer state scales with trainable parameters.</div>
        </div>
        """),
        v1_08_memory_mitigation,
    ])

    _part_c = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Precision Changes Memory, Throughput, And Stability Evidence</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            What does a lower-precision policy save, and what evidence does it owe?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Scenario</h2>
          <p>{v1_08_amount_system["precision_consequence"]}</p>
          <ul class="mlsysbook-list">
            <li>Precision changes bytes moved and stored for weights, gradients, and activations.</li>
            <li>FP16, BF16, and FP8 differ in exponent range, loss-scaling burden, and validation risk.</li>
            <li>The selected policy is {v1_08_precision_selected["label"]}: {v1_08_precision_selected["status"]}.</li>
          </ul>
        </div>
        """),
        v1_08_precision_prediction,
        v1_08_precision_policy,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Prediction Vs Actual</h2>
          <div class="mlsysbook-callout"><strong>Your prediction:</strong> {_prediction_value(v1_08_precision_prediction)}</div>
          <div class="mlsysbook-callout"><strong>Actual policy evidence:</strong>
            {v1_08_precision_selected["label"]} estimates {v1_08_precision_selected["total_mb"]:.2f} MB,
            {v1_08_precision_selected["throughput_samples_s"]:.2f} samples/s, and requires
            {v1_08_precision_selected["evidence"]}.</div>
        </div>
        """),
        mo.as_html(_precision_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Precision Evidence Table</h2>
          <table class="mlsysbook-table">
            <thead><tr><th>Policy</th><th>Bytes/value</th><th>Total memory</th><th>Throughput</th><th>Stability risk</th><th>Status</th><th>Evidence required</th></tr></thead>
            <tbody>{_precision_rows}</tbody>
          </table>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Consequence And Source Model</h2>
          <div class="mlsysbook-callout"><strong>Consequence:</strong> {v1_08_amount_system["precision_consequence"]}</div>
          <div class="mlsysbook-callout"><strong>Math Peek:</strong>
            Precision changes memory through <code>parameters * bytes_per_value</code>
            and throughput through hardware tensor paths. Stability evidence decides
            whether the savings can become a product artifact.</div>
        </div>
        """),
        v1_08_precision_checkpoint,
    ])

    _part_d = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part D: Training Plan Selection Must Satisfy Multiple Constraints</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which plan satisfies cost, time, memory, validation, and deployment evidence?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Scenario</h2>
          <p>{v1_08_amount_system["plan_consequence"]}</p>
          <ul class="mlsysbook-list">
            <li>A training plan is incomplete unless it names the training location and validation location.</li>
            <li>Centralized training, local adaptation, calibration, and validation are different activities.</li>
            <li>The deployment handoff is part of the risk: a trained checkpoint must still become a safe product artifact.</li>
          </ul>
        </div>
        """),
        v1_08_plan_prediction,
        v1_08_strategy_choice,
        v1_08_batch_size,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Prediction Vs Actual</h2>
          <div class="mlsysbook-callout"><strong>Your prediction:</strong> {_prediction_value(v1_08_plan_prediction)}</div>
          <div class="mlsysbook-callout"><strong>Actual binding resource:</strong> {v1_08_binding_resource_current}</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Plan Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Selected plan</strong>{v1_08_plan.selected_label}</div>
            <div class="mlsysbook-field"><strong>Feasible</strong>{'yes' if v1_08_plan.feasible else 'no - violation'}</div>
            <div class="mlsysbook-field"><strong>Training location</strong>{v1_08_plan.training_location}</div>
            <div class="mlsysbook-field"><strong>Validation location</strong>{v1_08_plan.validation_location}</div>
            <div class="mlsysbook-field"><strong>Total memory</strong>{v1_08_plan.total_memory_mb:.2f} MB</div>
            <div class="mlsysbook-field"><strong>Max feasible batch</strong>{v1_08_plan.max_feasible_batch or 'none'}</div>
          </div>
          <div class="mlsysbook-callout"><strong>Hidden cost:</strong> {v1_08_plan.hidden_cost}</div>
          <div class="mlsysbook-callout"><strong>Memo decision:</strong> {v1_08_plan.memo_summary}</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Plan Comparison Table</h2>
          <table class="mlsysbook-table">
            <thead><tr><th>Role</th><th>Strategy</th><th>Training location</th><th>Validation location</th><th>Total memory</th><th>Dominant component</th><th>Feasible</th><th>Hidden cost</th></tr></thead>
            <tbody>{_plan_rows}</tbody>
          </table>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Rejected Alternatives</h2>
          <ul class="mlsysbook-list">{_rejections}</ul>
          <h2>Validation Tests</h2>
          <ul class="mlsysbook-list">{_validation_items}</ul>
          <div class="mlsysbook-callout"><strong>Math Peek:</strong>
            The plan must satisfy the iron-law budget and the physical ceiling:
            memory, wall-clock time, dataset scale, and deployment evidence all
            have veto power.</div>
        </div>
        """),
        v1_08_plan_checkpoint,
        mo.Html('<div class="mlsysbook-panel"><h2>Report Reflection</h2></div>'),
        v1_08_reflection,
    ])

    mo.ui.tabs({
        "Part A · Batch": _part_a,
        "Part B · Memory": _part_b,
        "Part C · Precision": _part_c,
        "Part D · Plan": _part_d,
        "Synthesis": mo.md("Use the synthesis memo below after completing Parts A-D."),
    })
    return


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    v1_08_batch_checkpoint,
    v1_08_batch_consequence_current,
    v1_08_batch_prediction,
    v1_08_binding_resource_current,
    v1_08_carry_forward,
    v1_08_memo_evidence,
    v1_08_memory_mitigation,
    v1_08_memory_prediction,
    v1_08_plan,
    v1_08_plan_checkpoint,
    v1_08_plan_prediction,
    v1_08_precision_checkpoint,
    v1_08_precision_policy,
    v1_08_precision_prediction,
    v1_08_precision_selected,
    v1_08_profile,
    v1_08_reflection,
    v1_08_selected_stack,
    v1_08_training,
    v1_08_variant,
):
    _required_widgets = (
        v1_08_batch_prediction,
        v1_08_batch_checkpoint,
        v1_08_memory_prediction,
        v1_08_memory_mitigation,
        v1_08_precision_prediction,
        v1_08_precision_policy,
        v1_08_precision_checkpoint,
        v1_08_plan_prediction,
        v1_08_plan_checkpoint,
    )
    _has_progress = any(widget.value is not None for widget in _required_widgets)
    _completed = (
        all(widget.value is not None for widget in _required_widgets)
        and bool(str(v1_08_reflection.value or "").strip())
    )
    if _has_progress:
        ledger.save(chapter=8, design={
            "chapter": "v1_08",
            "track_id": v1_08_profile.track_id,
            "scenario_id": v1_08_variant.scenario_id,
            "hardware_ref": v1_08_training.hardware_ref,
            "model_ref": v1_08_training.model_ref,
            "completed": _completed,
            "batch_prediction": v1_08_batch_prediction.value,
            "batch_decision": v1_08_batch_checkpoint.value,
            "batch_size": v1_08_selected_stack.batch_size,
            "batch_consequence": v1_08_batch_consequence_current["consequence"],
            "memory_prediction": v1_08_memory_prediction.value,
            "memory_mitigation": v1_08_memory_mitigation.value,
            "precision_prediction": v1_08_precision_prediction.value,
            "precision_policy": v1_08_precision_policy.value or v1_08_precision_selected["policy_id"],
            "precision_status": v1_08_precision_selected["status"],
            "precision_total_memory_mb": v1_08_precision_selected["total_mb"],
            "precision_evidence_required": v1_08_precision_selected["evidence"],
            "plan_constraint_prediction": v1_08_plan_prediction.value,
            "plan_checkpoint": v1_08_plan_checkpoint.value,
            "selected_training_plan": v1_08_plan.selected_id,
            "training_location": v1_08_plan.training_location,
            "validation_location": v1_08_plan.validation_location,
            "dominant_component": v1_08_plan.dominant_component,
            "total_memory_mb": v1_08_selected_stack.total_mb,
            "binding_resource": v1_08_binding_resource_current,
            "memo_evidence_number": v1_08_memo_evidence,
            "carry_forward_deployment_implication": v1_08_carry_forward,
        })

    def build_synthesis():
        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Synthesis: Training Plan Memo</h2>
              <div class="mlsysbook-grid">
                <div class="mlsysbook-field"><strong>Track</strong>{v1_08_training.label}</div>
                <div class="mlsysbook-field"><strong>Selected plan</strong>{v1_08_plan.selected_label}</div>
                <div class="mlsysbook-field"><strong>Training location</strong>{v1_08_plan.training_location}</div>
                <div class="mlsysbook-field"><strong>Validation location</strong>{v1_08_plan.validation_location}</div>
                <div class="mlsysbook-field"><strong>Binding resource</strong>{v1_08_binding_resource_current}</div>
                <div class="mlsysbook-field"><strong>Evidence number</strong>{v1_08_memo_evidence}</div>
                <div class="mlsysbook-field"><strong>Precision policy</strong>{v1_08_precision_selected["label"]}</div>
                <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_08_plan.residual_risk}</div>
              </div>
              <div class="mlsysbook-callout"><strong>Carry-forward deployment implication:</strong> {v1_08_carry_forward}</div>
            </div>
            """),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Big Takeaways</h2>
              <ul class="mlsysbook-list">
                <li><strong>Batch is a systems knob.</strong> It changes utilization, memory pressure, convergence risk, and evidence burden.</li>
                <li><strong>Training memory is a stack.</strong> Optimizer state and activations can dominate even when inference weights fit.</li>
                <li><strong>Precision buys resources with evidence debt.</strong> Lower precision must be justified by stability and deployment replay.</li>
                <li><strong>A plan is valid inside constraints.</strong> Cost, time, memory, validation, and deployment handoff all have veto power.</li>
              </ul>
            </div>
            """),
            mo.Html(f"""
            <div class="lab-hud">
                <span class="hud-label">LAB</span>
                <span class="hud-value">08 &middot; Training Gauntlet</span>
                <span class="hud-label">TRACK</span>
                <span class="hud-value">{v1_08_profile.label}</span>
                <span style="flex:1;"></span>
                <span class="hud-label">ARTIFACT</span>
                <span class="hud-value">{v1_08_training.report_artifact}</span>
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
    v1_08_batch_checkpoint,
    v1_08_batch_consequence_current,
    v1_08_batch_prediction,
    v1_08_batch_size,
    v1_08_binding_resource_current,
    v1_08_carry_forward,
    v1_08_frontier,
    v1_08_memo_evidence,
    v1_08_memory_mitigation,
    v1_08_memory_prediction,
    v1_08_memory_rows,
    v1_08_metadata,
    v1_08_plan,
    v1_08_plan_checkpoint,
    v1_08_plan_prediction,
    v1_08_precision_checkpoint,
    v1_08_precision_policy,
    v1_08_precision_prediction,
    v1_08_precision_rows,
    v1_08_precision_selected,
    v1_08_profile,
    v1_08_reflection,
    v1_08_selected_stack,
    v1_08_training,
    v1_08_variant,
):
    _incomplete = []
    if v1_08_batch_prediction.value is None:
        _incomplete.append("Part A batch prediction")
    if v1_08_batch_checkpoint.value is None:
        _incomplete.append("Part A batch checkpoint")
    if v1_08_memory_prediction.value is None:
        _incomplete.append("Part B dominant-memory prediction")
    if v1_08_memory_mitigation.value is None:
        _incomplete.append("Part B memory mitigation")
    if v1_08_precision_prediction.value is None:
        _incomplete.append("Part C precision prediction")
    if v1_08_precision_policy.value is None:
        _incomplete.append("Part C precision policy")
    if v1_08_precision_checkpoint.value is None:
        _incomplete.append("Part C precision evidence checkpoint")
    if v1_08_plan_prediction.value is None:
        _incomplete.append("Part D constraint prediction")
    if v1_08_plan_checkpoint.value is None:
        _incomplete.append("Part D memo checkpoint")
    if not str(v1_08_reflection.value or "").strip():
        _incomplete.append("Synthesis memo reflection")

    _report = build_lab_report(
        v1_08_metadata,
        track=v1_08_profile.label,
        scenario=v1_08_variant.workload_summary,
        learning_objectives=(
            "Explain why batch size changes throughput, memory pressure, and convergence evidence.",
            "Build a training memory stack for weights, gradients, optimizer state, activations, and data batches.",
            "Evaluate precision policy as a memory/throughput/stability trade-off.",
            "Choose a training, adaptation, or calibration plan with cost, time, memory, validation, and deployment handoff constraints.",
        ),
        predictions={
            "batch_size_effect": v1_08_batch_prediction.value,
            "dominant_training_memory": v1_08_memory_prediction.value,
            "precision_effect": v1_08_precision_prediction.value,
            "binding_constraint": v1_08_plan_prediction.value,
        },
        knob_settings={
            "batch_size": v1_08_batch_size.value,
            "selected_strategy": v1_08_plan.selected_id,
            "batch_checkpoint": v1_08_batch_checkpoint.value,
            "memory_mitigation": v1_08_memory_mitigation.value,
            "precision_policy": v1_08_precision_policy.value or v1_08_precision_selected["policy_id"],
            "precision_checkpoint": v1_08_precision_checkpoint.value,
            "plan_checkpoint": v1_08_plan_checkpoint.value,
        },
        evidence_summary={
            "hardware_ref": v1_08_training.hardware_ref,
            "model_ref": v1_08_training.model_ref,
            "model_params_m": v1_08_training.model_params_m,
            "training_budget_mb": v1_08_selected_stack.budget_mb,
            "total_memory_mb": v1_08_selected_stack.total_mb,
            "dominant_component": v1_08_plan.dominant_component,
            "max_feasible_batch": v1_08_frontier.max_feasible_batch,
            "training_location": v1_08_plan.training_location,
            "validation_location": v1_08_plan.validation_location,
            "batch_consequence": v1_08_batch_consequence_current["consequence"],
            "precision_policy": v1_08_precision_selected["label"],
            "precision_total_memory_mb": v1_08_precision_selected["total_mb"],
            "precision_status": v1_08_precision_selected["status"],
            "precision_evidence_required": v1_08_precision_selected["evidence"],
            "binding_resource": v1_08_binding_resource_current,
            "memo_evidence_number": v1_08_memo_evidence,
            "carry_forward_deployment_implication": v1_08_carry_forward,
        },
        final_decision=(
            f"{v1_08_plan.memo_summary} Binding resource: "
            f"{v1_08_binding_resource_current}. Carry forward: {v1_08_carry_forward}"
        ),
        big_takeaways=(
            "Training is budgeted optimization: batch, precision, optimizer state, memory, throughput, convergence cost, and deployment evidence interact.",
            "Batch size changes utilization and convergence evidence, not only speed.",
            "Optimizer state and activations create the memory budget that decides whether training fits.",
            "Precision changes memory and throughput while adding stability evidence requirements.",
            "A training plan must include binding resource, hidden cost, validation location, and carry-forward deployment implication.",
        ),
        reflections={
            "student_reflection": v1_08_reflection.value,
            "hidden_cost": v1_08_plan.hidden_cost,
            "deployment_handoff": v1_08_plan.deployment_handoff,
            "report_artifact": v1_08_training.report_artifact,
            "binding_resource": v1_08_binding_resource_current,
            "memo_evidence_number": v1_08_memo_evidence,
            "carry_forward_deployment_implication": v1_08_carry_forward,
        },
        residual_risk=v1_08_plan.residual_risk,
        source_trace={
            "track_id": v1_08_profile.track_id,
            "scenario_id": v1_08_variant.scenario_id,
            "hardware_ref": v1_08_variant.hardware_ref,
            "model_ref": v1_08_variant.model_ref,
            "shared_helper": "mlsysbook_labs.training",
            "source_policy": v1_08_profile.source_policy,
        },
        result_snapshot={
            "training_profile": v1_08_training,
            "memory_rows": v1_08_memory_rows,
            "frontier": v1_08_frontier,
            "precision_rows": v1_08_precision_rows,
            "precision_selected": v1_08_precision_selected,
            "selected_stack": v1_08_selected_stack,
            "plan": v1_08_plan,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-08 training feasibility plan is generated locally from "
                "the selected track, your inputs, and the computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
