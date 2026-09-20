import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 08: Time to Target · MLSysBook")


@app.cell
async def _():
    import sys
    from pathlib import Path
    import marimo as mo

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.engine.v1_08_experiments import (
        TRACKS, batch_experiment, batch_policy, bottleneck_experiment,
        checkpoint_experiment, memory_outcome, optimizer_experiment, precision_experiment, track_profile,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata, report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, TRACKS, apply_plotly_theme, batch_policy,
        audit_evidence, batch_experiment, bottleneck_experiment, build_lab_report,
        capture_evidence, checkpoint_experiment, get_lab_metadata, go, ledger,
        memory_outcome, mo, optimizer_experiment, precision_experiment,
        report_export_panel, track_profile,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Training context",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(TRACKS, track, track_profile):
    track_id = track.value
    profile = track_profile(track_id)
    track_settings = TRACKS[track_id]
    return profile, track_id, track_settings


@app.cell
def _(batch_policy, mo, track_id):
    _track_key = track_id
    _b_pol = batch_policy(track_id)
    a_optimizer = mo.ui.radio(
        {"Adam": "adam", "AdamW": "adamw"},
        value="AdamW", label="Optimizer conclusion",
    )
    b_strategy = mo.ui.dropdown(
        _b_pol["options"],
        value=_b_pol["default_label"], label="Batch strategy",
    )
    c_precision = mo.ui.dropdown(
        {"BF16 mixed": "bf16", "FP16 mixed": "fp16", "FP8 scenario": "fp8"},
        value="BF16 mixed", label="Precision candidate",
    )
    d_checkpoint = mo.ui.radio(
        {"Selective checkpointing": "selective", "Full checkpointing": "full"},
        value="Full checkpointing", label="Checkpoint policy",
    )
    e_intervention = mo.ui.radio(
        {"Prefetch input": "prefetch", "Faster arithmetic path": "arithmetic", "Checkpoint activations": "checkpoint", "No change / hold for more evidence": "none"},
        value="Prefetch input", label="First systems fix",
    )
    e_rejected = mo.ui.radio(
        {"Prefetch input": "prefetch", "Faster arithmetic": "arithmetic", "Checkpoint activations": "checkpoint"},
        label="Tested alternative if you hold",
    )
    return a_optimizer, b_strategy, c_precision, d_checkpoint, e_intervention, e_rejected


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Weights dominate": "weights", "Gradients dominate": "gradients", "Optimizer state dominates": "optimizer"},
        label="Which allocation is largest under AdamW?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Largest effective batch": "largest", "Resident batch 32": "resident32", "Accumulated middle batch": "middle", "All finish equally": "equal"},
        label="Which strategy reaches target soonest?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Every narrower format": "all", "BF16/FP16 only": "mixed", "Only FP32": "fp32", "Speed alone decides": "speed"},
        label="Which precision evidence remains acceptable?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Memory falls; time rises": "tradeoff", "Both fall": "free", "Only optimizer state changes": "optimizer", "Nothing changes": "none"},
        label="What does activation checkpointing change?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Prefetch input": "prefetch", "Faster arithmetic": "arithmetic", "Checkpoint activations": "checkpoint"},
        label="Which fix should be tested before buying hardware?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Prefetch input": "prefetch", "Faster arithmetic": "arithmetic", "Checkpoint activations": "checkpoint", "No change / hold for more evidence": "none"},
        label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Prefetch input": "prefetch", "Faster arithmetic": "arithmetic", "Checkpoint activations": "checkpoint"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"Target quality is missed": "target_quality", "Training memory exceeds capacity": "memory", "Schedule or cost limit is crossed": "schedule_cost"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Convergence may not transfer": "convergence", "Stage timing may change": "timing", "Replay may miss rare instability": "numerical"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Name the chosen evidence, quantify the rejected alternative, state the limitation, and explain the trigger.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_optimizer, b_strategy, batch_experiment, batch_policy,
    bottleneck_experiment, c_precision, checkpoint_experiment, d_checkpoint,
    e_intervention, e_rejected, memory_outcome, optimizer_experiment,
    precision_experiment, track_id, track_settings,
):
    a_sgd = optimizer_experiment(track_id, optimizer="sgd")
    a_adam = optimizer_experiment(track_id, optimizer="adam")
    a_adamw = optimizer_experiment(track_id, optimizer="adamw")
    a_selected = {"sgd": a_sgd, "adam": a_adam, "adamw": a_adamw}[a_optimizer.value]
    _b_pol = batch_policy(track_id)
    b_physical, b_accumulation = _b_pol["choices"][b_strategy.value]
    b_baseline = batch_experiment(
        track_id,
        physical_batch=_b_pol["baseline"][0],
        accumulation_steps=_b_pol["baseline"][1],
    )
    b_result = batch_experiment(track_id, physical_batch=b_physical, accumulation_steps=b_accumulation)
    b_alternatives = tuple(
        batch_experiment(track_id, physical_batch=physical, accumulation_steps=accumulation)
        for physical, accumulation in _b_pol["choices"].values()
    )
    c_fp32 = precision_experiment(track_id, precision="fp32")
    c_result = precision_experiment(track_id, precision=c_precision.value)
    c_alternatives = tuple(
        precision_experiment(track_id, precision=precision)
        for precision in ("fp32", "bf16", "fp16", "fp8")
    )
    probe_batch = track_settings["checkpoint_probe_batch"]
    d_baseline = memory_outcome(track_id, physical_batch=probe_batch, accumulation_steps=1, checkpointing="none")
    d_result = memory_outcome(track_id, physical_batch=probe_batch, accumulation_steps=1, checkpointing=d_checkpoint.value)
    d_timing = checkpoint_experiment(track_id, checkpointing=d_checkpoint.value)
    e_comparisons = {
        name: bottleneck_experiment(track_id, intervention=name)
        for name in ("prefetch", "arithmetic", "checkpoint")
    }
    if e_intervention.value == "none":
        e_selected = e_comparisons[e_rejected.value] if e_rejected.value is not None else None
    else:
        e_selected = e_comparisons[e_intervention.value]
    return (
        a_adam, a_adamw, a_selected, a_sgd, b_accumulation, b_alternatives,
        b_baseline, b_physical, b_result, c_alternatives, c_fp32, c_result,
        d_baseline, d_result, d_timing, e_comparisons, e_selected, probe_batch,
    )


@app.cell
def _(
    a_adam, a_adamw, a_optimizer, a_prediction, a_selected, a_sgd,
    b_accumulation, b_alternatives, b_baseline, b_physical, b_prediction,
    b_result, c_alternatives, c_fp32, c_precision, c_prediction, c_result,
    capture_evidence, d_baseline, d_checkpoint, d_prediction, d_result,
    d_timing, e_comparisons, e_intervention, e_prediction, e_rejected,
    e_selected, mo,
    probe_batch, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})
    common_upstream = {}
    a_capture = mo.ui.button(
        label="Capture optimizer comparison", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"optimizer_choice": a_optimizer.value}, baseline=a_sgd, result=a_selected,
            alternatives=(a_sgd, a_adam, a_adamw), decision=a_optimizer.value,
            chosen_result=a_selected, result_role="chosen optimizer",
            upstream_inputs=common_upstream, model_key="v1_08_experiments.optimizer_experiment",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture batch comparison", kind="success",
        disabled=b_prediction.value is None or (
            b_physical == b_baseline["timing"]["physical_batch"]
            and b_accumulation == b_baseline["timing"]["accumulation_steps"]
        ),
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"physical_batch": b_physical, "accumulation_steps": b_accumulation},
            baseline=b_baseline, result=b_result, alternatives=b_alternatives,
            decision=f"{b_physical}x{b_accumulation}", upstream_inputs=common_upstream,
            chosen_result=b_result, result_role="chosen batch strategy",
            model_key="v1_08_experiments.batch_experiment",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture precision comparison", kind="success", disabled=c_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"precision": c_precision.value}, baseline=c_fp32, result=c_result,
            alternatives=c_alternatives, decision=c_precision.value,
            chosen_result=c_result, result_role="chosen precision",
            upstream_inputs=common_upstream, model_key="v1_08_experiments.precision_experiment",
        )),
    )
    d_capture = mo.ui.button(
        label="Capture checkpoint comparison", kind="success", disabled=d_prediction.value is None,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"physical_batch": probe_batch, "checkpointing": d_checkpoint.value},
            baseline=d_baseline, result=d_result, alternatives=(d_timing,),
            decision=d_checkpoint.value, upstream_inputs=common_upstream,
            chosen_result=d_result, result_role="chosen checkpoint policy",
            model_key="v1_08_experiments.memory_outcome",
        )),
    )
    e_hold_without_test = e_intervention.value == "none" and e_rejected.value is None
    e_capture = mo.ui.button(
        label="Capture bottleneck decision", kind="success",
        disabled=e_prediction.value is None or e_hold_without_test,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"intervention": e_intervention.value, "rejected_alternative": e_rejected.value},
            baseline=e_selected["baseline"], result=e_selected["result"],
            alternatives=tuple(e_comparisons.values()), decision=e_intervention.value,
            chosen_result=e_selected["baseline"] if e_intervention.value == "none" else e_selected["result"],
            result_role="rejected alternative" if e_intervention.value == "none" else "chosen systems intervention",
            upstream_inputs=common_upstream, model_key="v1_08_experiments.bottleneck_experiment",
        )),
    )
    return a_capture, b_capture, c_capture, common_upstream, d_capture, e_capture


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = """
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:780px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>"""
    header = mo.Html(f"""{css}<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 08</span><span>ABOUT 50–55 MIN</span></div><h1>Train for the Target, Not the Counter</h1><p>Which training design reaches a common quality target within memory, schedule, and cost limits?</p><div class="pilot-meta"><div><b>Track</b><br>{profile['label']}</div><div><b>Context</b><br>{profile['task']}</div><div><b>Teaching workload</b><br>GPT-2 on a development host</div><div><b>Deliverable</b><br>Training design recommendation</div></div></section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, header, track,
        mo.Html('<p class="pilot-note">Device tracks study upstream teacher training or adaptation; GPT-2 is not deployed on the constrained target. Convergence, numerical replay, stage timing, and internal host rates are finite illustrative scenario evidence, not benchmark measurements.</p>'),
    ], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_adam, a_adamw, a_capture, a_optimizer, a_prediction,
    a_selected, a_sgd, apply_plotly_theme, audit_evidence, b_baseline,
    b_capture, b_physical, b_prediction, b_result, b_strategy, c_capture,
    c_fp32, c_precision, c_prediction, c_result, common_upstream,
    d_baseline, d_capture, d_checkpoint, d_prediction, d_result, d_timing,
    e_capture, e_comparisons, e_intervention, e_prediction, e_rejected,
    e_selected, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, go, mo, probe_batch, profile, rationale, track_id,
):
    _captures = get_evidence()
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs={part: common_upstream for part in "ABCDE"},
        contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this part."), kind="danger")
        snapshot = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · prediction: {snapshot["prediction"]}<br><small>The report keeps these captured settings and results.</small></div>')

    def status(result):
        return "PASS" if result["feasible"] else "FAIL · " + ", ".join(result.get("violations", ("memory",)))

    def part_a():
        intro = mo.md(
            f"### A · Why does inference fit while training fails? (8 min)\n"
            f"On the upstream development host ({profile['hardware'].name}) used for {profile['label']} "
            "teacher adaptation, single-batch inference fits easily while full training allocates substantial additional state. "
            "Predict the largest AdamW allocation before opening the training-memory ledger."
        )
        if a_prediction.value is None:
            return mo.vstack([intro, a_prediction])
        rows = [{
            "Optimizer": name.upper(), "Weights (GB)": f"{result['memory']['weights_gb']:.1f}",
            "Gradients (GB)": f"{result['memory']['gradients_gb']:.1f}",
            "Optimizer state (GB)": f"{result['memory']['optimizer_state_gb']:.1f}",
            "Total (GB)": f"{result['memory']['total_memory_gb']:.1f}",
            "Updates to target": result["updates_to_target"],
        } for name, result in (("sgd", a_sgd), ("adam", a_adam), ("adamw", a_adamw))]
        figure = go.Figure()
        for label, key, color in (("Weights", "weights_gb", COLORS["BlueLine"]), ("Gradients", "gradients_gb", COLORS["OrangeLine"]), ("Optimizer", "optimizer_state_gb", COLORS["GreenLine"]), ("Activations", "activations_gb", COLORS["RedLine"])):
            figure.add_bar(name=label, x=["SGD", "Adam", "AdamW"], y=[r["memory"][key] for r in (a_sgd, a_adam, a_adamw)], marker_color=color)
        figure.update_layout(barmode="stack", height=290, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Training memory (GB)", legend_orientation="h")
        return mo.vstack([
            intro, a_prediction, apply_plotly_theme(figure), table(rows), a_optimizer,
            mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. On the {profile['hardware'].name} development host, AdamW optimizer state is **{a_adamw['memory']['optimizer_state_gb']:.1f} GB**; inference weights alone are **{a_adamw['memory']['inference_weights_gb']:.1f} GB**. Your optimizer conclusion is **{a_optimizer.value}**, with {a_selected['updates_to_target']} illustrative updates to target."), kind="info"),
            a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("Training memory sums weights, gradients, optimizer state, retained activations, and buffers. The simulator computes the allocations; convergence traces are supplied scenario observations.")}),
        ])

    def part_b():
        intro = mo.md("### B · Does a larger batch finish sooner? (10 min)\nPhysical batch controls resident activations. Accumulation repeats microsteps before one optimizer update. Compare time to the same target quality.")
        if b_prediction.value is None:
            return mo.vstack([intro, b_prediction])
        rows = [
            {
                "Design": "Baseline",
                "Physical batch": b_baseline["timing"]["physical_batch"],
                "Accumulation": b_baseline["timing"]["accumulation_steps"],
                "Effective batch": b_baseline["timing"]["effective_batch"],
                "Updates": b_baseline["updates_to_target"] if b_baseline["updates_to_target"] is not None else "target missed",
                "Samples presented": b_baseline["samples_presented"] if b_baseline["samples_presented"] is not None else "—",
                "Update time (ms)": f"{b_baseline['timing']['update_time_ms']:.1f}",
                "Total time (s)": f"{b_baseline['total_time_seconds']:.1f}" if b_baseline["total_time_seconds"] is not None else "—",
                "Outcome": status(b_baseline),
            },
            {
                "Design": "Intervention",
                "Physical batch": b_result["timing"]["physical_batch"],
                "Accumulation": b_result["timing"]["accumulation_steps"],
                "Effective batch": b_result["timing"]["effective_batch"],
                "Updates": b_result["updates_to_target"] if b_result["updates_to_target"] is not None else "target missed",
                "Samples presented": b_result["samples_presented"] if b_result["samples_presented"] is not None else "—",
                "Update time (ms)": f"{b_result['timing']['update_time_ms']:.1f}",
                "Total time (s)": f"{b_result['total_time_seconds']:.1f}" if b_result["total_time_seconds"] is not None else "—",
                "Outcome": status(b_result),
            },
        ]
        figure = go.Figure()
        figure.add_scatter(
            x=[point[0] for point in b_baseline["trace"]],
            y=[point[1] for point in b_baseline["trace"]],
            mode="lines+markers",
            name=f"Baseline (eff {b_baseline['timing']['effective_batch']})",
            line=dict(color=COLORS["OrangeLine"], width=2, dash="dot"),
        )
        figure.add_scatter(
            x=[point[0] for point in b_result["trace"]],
            y=[point[1] for point in b_result["trace"]],
            mode="lines+markers",
            name=f"Intervention (eff {b_result['timing']['effective_batch']})",
            line=dict(color=COLORS["BlueLine"], width=3),
        )
        figure.add_hline(y=b_result["target_quality"], line_dash="dash", annotation_text="Common target")
        figure.update_layout(
            height=280, margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Optimizer updates", yaxis_title="Illustrative held-out quality",
            legend_orientation="h",
        )
        b_success = b_result["feasible"] and b_result["reached_target"]
        if not b_result["memory"]["feasible"]:
            b_feedback = (
                f" **Execution failed ({status(b_result)}).** "
                f"Resident physical batch {b_physical} requires {b_result['memory']['total_memory_gb']:.1f} GB, "
                f"exceeding accelerator capacity ({b_result['memory']['available_memory_gb']:.1f} GB) "
                f"with {b_result['memory']['activations_gb']:.1f} GB of resident activations alone (Out-Of-Memory / OOM). "
                "Gradient accumulation enables larger effective batches without blowing up resident activation memory."
            )
        elif not b_result["reached_target"]:
            b_feedback = f" **Target missed ({status(b_result)}).** The design is feasible in memory but failed to reach target quality ({b_result['target_quality']}) within the update horizon."
        elif not b_result["feasible"]:
            b_feedback = f" **Constraint violated ({status(b_result)}).** The design reached target quality but violated schedule or cost limits."
        else:
            b_feedback = " The design is feasible and reached the common target."
        return mo.vstack([
            intro, b_prediction, b_strategy, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. The selected design uses physical batch **{b_physical}** and effective batch **{b_result['timing']['effective_batch']}**. It presents **{b_result['samples_presented'] or 'no completed target run'}** samples before the target result.{b_feedback}"), kind="success" if b_success else "danger"),
            b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md("Update time repeats input + forward + backward + recomputation for every accumulated microbatch, followed by one optimizer step. The simulator multiplies supplied updates-to-target by update time and separately counts sample presentations.")}),
        ])

    def part_c():
        intro = mo.md("### C · When does lower precision help? (10 min)\nNarrower compute can shorten stages and reduce tensors, while high-precision optimizer state remains. Independent numerical and convergence replays decide acceptance.")
        if c_prediction.value is None:
            return mo.vstack([intro, c_prediction])
        rows = [{
            "Format": result["memory"]["precision"].upper(), "Memory (GB)": f"{result['memory']['total_memory_gb']:.1f}",
            "Update time (ms)": f"{result['timing']['update_time_ms']:.1f}", "Optimizer stage (ms)": f"{result['timing']['optimizer_ms']:.1f}",
            "Finite replay": f"{result['numerical_replay']['finite_fraction']:.1%}", "Target": "reached" if result["reached_target"] else "missed",
        } for result in (c_fp32, c_result)]
        figure = go.Figure([go.Bar(x=["FP32", c_precision.value.upper()], y=[c_fp32["timing"]["update_time_ms"], c_result["timing"]["update_time_ms"]], marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]])])
        figure.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Update time (ms)", showlegend=False)
        return mo.vstack([
            intro, c_prediction, c_precision, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. **{c_precision.value.upper()}** {'reaches' if c_result['reached_target'] else 'does not reach'} the common target. Its FP32 optimizer stage remains **{c_result['timing']['optimizer_ms']:.1f} ms**."), kind="success" if c_result["reached_target"] else "danger"),
            c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("Precision selects finite forward/backward observations and tensor widths. Optimizer time is a separate FP32-state fixture. Numerical replay and convergence are not functions of throughput.")}),
        ])

    def part_d():
        intro = mo.md(f"### D · When is recomputation worth paying for? (10 min)\nThe resident physical-batch probe is **{probe_batch}**. Without checkpointing it crosses the memory boundary. Predict both memory and time directions.")
        if d_prediction.value is None:
            return mo.vstack([intro, d_prediction])
        rows = [
            {"Policy": "None", "Activations (GB)": f"{d_baseline['activations_gb']:.1f}", "Total (GB)": f"{d_baseline['total_memory_gb']:.1f}", "Capacity (GB)": f"{d_baseline['available_memory_gb']:.1f}", "Fits": "yes" if d_baseline["feasible"] else "NO"},
            {"Policy": d_checkpoint.value.title(), "Activations (GB)": f"{d_result['activations_gb']:.1f}", "Total (GB)": f"{d_result['total_memory_gb']:.1f}", "Capacity (GB)": f"{d_result['available_memory_gb']:.1f}", "Fits": "yes" if d_result["feasible"] else "NO"},
        ]
        figure = go.Figure([go.Bar(x=["No checkpoint", d_checkpoint.value.title()], y=[d_baseline["total_memory_gb"], d_result["total_memory_gb"]], marker_color=[COLORS["RedLine"], COLORS["GreenLine"]])])
        figure.add_hline(y=d_baseline["available_memory_gb"], line_dash="dash", annotation_text="Capacity")
        figure.update_layout(height=270, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Training memory (GB)", showlegend=False)
        return mo.vstack([
            intro, d_prediction, d_checkpoint, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. The selected policy saves **{d_timing['activation_memory_saved_gb']:.1f} GB** at the default batch and adds **{d_timing['added_update_time_ms']:.1f} ms** per update. The probe changes from **{'fit' if d_baseline['feasible'] else 'OOM'}** to **{'fit' if d_result['feasible'] else 'OOM'}**."), kind="success" if d_result["feasible"] else "danger"),
            d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("The memory model reduces retained activations. The stage model adds supplied recomputation work to each physical microstep; convergence does not improve automatically.")}),
        ])

    def part_e():
        intro = mo.md("### E · What should we fix before buying hardware? (8 min)\nCompare input prefetch, a supplied faster arithmetic path, and activation checkpointing against the identical baseline.")
        if e_prediction.value is None:
            return mo.vstack([intro, e_prediction])
        rows = [{
            "Fix": name.title(), "Update-time change (ms)": f"{comparison['delta_update_time_ms']:+.1f}",
            "Memory change (GB)": f"{comparison['delta_memory_gb']:+.1f}",
            "Total time (s)": f"{comparison['result']['total_time_seconds']:.1f}",
            "Dominant stage": comparison["result"]["timing"]["dominant_stage"], "Outcome": status(comparison["result"]),
        } for name, comparison in e_comparisons.items()]
        figure = go.Figure([go.Bar(x=[name.title() for name in e_comparisons], y=[comparison["delta_update_time_ms"] for comparison in e_comparisons.values()], marker_color=[COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"]])])
        figure.update_layout(height=270, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Change in update time (ms)", showlegend=False)
        if e_selected is None:
            return mo.vstack([intro, e_prediction, e_intervention, e_rejected])
        _held = e_intervention.value == "none"
        _decision_text = "hold the unchanged baseline" if _held else f"choose {e_intervention.value}"
        return mo.vstack([
            intro, e_prediction, e_intervention, apply_plotly_theme(figure), table(rows),
            e_rejected,
            mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. You **{_decision_text}** after testing **{e_selected['intervention']}**, which changes update time by **{e_selected['delta_update_time_ms']:+.1f} ms** and memory by **{e_selected['delta_memory_gb']:+.1f} GB**. Convergence is unchanged."), kind="info"),
            e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md("Each intervention selects one supplied causal path. Prefetch changes visible input time; faster arithmetic changes forward/backward stages; checkpointing changes retained activations and adds recomputation.")}),
        ])

    def synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            rows.append({"Part": part, "Prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING")})
        saved_e = _captures["E"].to_dict() if "E" in _captures else None
        saved_e_decision = saved_e["decision"] if saved_e else None
        saved_e_rejected = saved_e["inputs"].get("rejected_alternative") if saved_e else None
        selected_matches = (final_choice.value == saved_e_decision and (final_choice.value != "none" or final_rejected.value == saved_e_rejected))
        choices_complete = all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        distinct = final_choice.value == "none" or final_choice.value != final_rejected.value
        complete = audit.complete and choices_complete and distinct and selected_matches and bool(rationale.value.strip())
        return mo.vstack([
            mo.md("### Synthesis · Defend a time-to-target decision (5 min)\nChoose one tested fix or hold for more evidence, quantify a rejected tested alternative, name the remaining limitation, and state the reevaluation trigger."),
            table(rows), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale,
            mo.callout(mo.md("**Ready for the local report.**" if complete else "Capture five current contrasts. Match the recommendation to saved Part E. If you hold, match the rejected choice to the alternative tested there."), kind="success" if complete else "warn"),
        ])

    tabs = mo.ui.tabs({"Part A · Memory": part_a(), "Part B · Batch": part_b(), "Part C · Precision": part_c(), "Part D · Checkpoint": part_d(), "Part E · Bottleneck": part_e(), "Synthesis": synthesis()})
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _saved_e = _captures["E"].to_dict() if "E" in _captures else None
    _saved_e_decision = _saved_e["decision"] if _saved_e else None
    _saved_e_rejected = _saved_e["inputs"].get("rejected_alternative") if _saved_e else None
    _choices_complete = all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
    _distinct = final_choice.value == "none" or final_choice.value != final_rejected.value
    _selected_matches = final_choice.value == _saved_e_decision and (final_choice.value != "none" or final_rejected.value == _saved_e_rejected)
    _ready = audit.complete and _choices_complete and _distinct and _selected_matches and bool(rationale.value.strip())
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_08_model_train.py"), track=track_id, scenario=profile["task"],
        learning_objectives=["Decompose training memory and optimizer state", "Compare physical batch, effective batch, and time to target", "Defend precision, checkpointing, and bottleneck interventions with evidence"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "result_role": snapshots[part]["result_role"], "chosen_result": snapshots[part]["chosen_result"], "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        binding_constraints={part: (snapshots[part]["chosen_result"] or snapshots[part]["result"]).get("violations", ["memory"] if not (snapshots[part]["chosen_result"] or snapshots[part]["result"]).get("feasible", True) else []) for part in "ABCDE"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["Training memory includes state inference never retains.", "Step throughput and convergence jointly determine time to target.", "Checkpointing and lower precision need adverse-outcome evidence."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value}, residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": "Illustrative convergence, numerical, timing, and internal-cost fixtures.", "calculations": "Training-memory and stage-time experiment model."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _saved_e = _captures["E"].to_dict() if "E" in _captures else None
    _saved_e_decision = _saved_e["decision"] if _saved_e else None
    _saved_e_rejected = _saved_e["inputs"].get("rejected_alternative") if _saved_e else None
    _choices_complete = all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
    _distinct = final_choice.value == "none" or final_choice.value != final_rejected.value
    _selected_matches = final_choice.value == _saved_e_decision and (final_choice.value != "none" or final_rejected.value == _saved_e_rejected)
    _ready = audit.complete and _choices_complete and _distinct and _selected_matches and bool(rationale.value.strip())
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(chapter=8, design={
                "schema_version": 1, "lab_id": "v1_08", "track_id": track_id,
                "model_id": "v1_08_experiments",
                "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
            })
            await ledger.flush()
        except Exception:
            _status = "LOCAL SAVE FAILED · DOWNLOAD THE REPORT TO KEEP YOUR EVIDENCE"
        else:
            _status = "SAVED"
    mo.Html(f'<div class="lab-hud"><span>LAB 08 · Time to Target · STATUS: {_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
