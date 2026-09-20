import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 01: The Fleet Is the System · MLSysBook")


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
    from mlsysim.engine.v2_01_experiments import (
        compare_failure_semantics,
        evaluate_c3_scaling,
        evaluate_machine_boundary,
        evaluate_partition_policy,
        get_track_scenario,
        serialize_evaluation,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        get_lab_metadata,
        report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        COLORS,
        LAB_CSS,
        apply_plotly_theme,
        audit_evidence,
        build_lab_report,
        capture_evidence,
        compare_failure_semantics,
        evaluate_c3_scaling,
        evaluate_machine_boundary,
        evaluate_partition_policy,
        get_lab_metadata,
        get_track_scenario,
        go,
        ledger,
        mo,
        report_export_panel,
        serialize_evaluation,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML",
        label="Teaching track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_track_scenario, track):
    track_id = track.value
    scenario = get_track_scenario(track_id)
    profile = scenario.profile
    return profile, scenario, track_id


@app.cell
def _(mo, scenario, track_id):
    _track_key = track_id
    a_action = mo.ui.radio(
        {"Add machines": "add_machines", "Reduce state on each machine": "compact_state"},
        value="Add machines",
        label="Capacity intervention",
    )
    a_machine_count = mo.ui.dropdown(
        {f"{value:,} machines": value for value in scenario.machine_count_options},
        value=f"{scenario.machine_count_options[-1]:,} machines",
        label="Machine count",
    )
    b_workers = mo.ui.dropdown(
        {f"{value:,} workers": value for value in scenario.worker_options},
        value=f"{scenario.worker_options[1]:,} workers",
        label="Coupled workers",
    )
    b_bandwidth = mo.ui.dropdown(
        {label.title(): label for label in scenario.bandwidth_options},
        value="Planned",
        label="Network tier",
    )
    b_coordination = mo.ui.dropdown(
        {label.title(): label for label in scenario.coordination_options},
        value="Planned",
        label="Coordination condition",
    )
    c_nodes = mo.ui.dropdown(
        {f"{value:,} nodes": value for value in scenario.fleet_size_options},
        value=f"{scenario.fleet_size_options[1]:,} nodes",
        label="Deployed fleet size",
    )
    c_recovery = mo.ui.dropdown(
        {label.title(): label for label in scenario.recovery_options},
        value="Planned",
        label="Repair path",
    )
    d_duration = mo.ui.dropdown(
        {label.title(): label for label in scenario.partition_duration_options},
        value="Extended",
        label="Partition duration",
    )
    d_fraction = mo.ui.dropdown(
        {label.title(): label for label in scenario.partitioned_fraction_options},
        value="Regional Cohort",
        label="Partitioned scope",
    )
    d_policy = mo.ui.radio(
        {
            "Wait for fresh state": "wait_for_fresh",
            "Serve the last confirmed model": "serve_last_confirmed",
        },
        label="Availability policy",
    )
    return (
        a_action,
        a_machine_count,
        b_bandwidth,
        b_coordination,
        b_workers,
        c_nodes,
        c_recovery,
        d_duration,
        d_fraction,
        d_policy,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {
            "More machines make the original state fit": "fleet_repairs_fit",
            "Only reducing per-machine state repairs fit": "local_repair",
            "Neither intervention satisfies both memory and rate": "neither",
            "The baseline already fits": "already_fits",
        },
        label="Which statement will the capacity test support?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {
            "Compute remains dominant": "compute",
            "Communication becomes dominant": "communication",
            "Coordination becomes dominant": "coordination",
        },
        label="Which C³ term will dominate the changed run?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {
            "One node event stops both systems": "both_stop",
            "Only the coupled job stops": "coupled_stops",
            "Neither system loses useful work": "neither",
        },
        label="What does one node failure do?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {
            "Waiting creates unavailable requests": "wait_unavailable",
            "Serving old state keeps every request fresh": "stale_is_fresh",
            "Both policies drop the same requests": "same_loss",
        },
        label="Which partition consequence do you expect?",
    ).form(submit_button_label="Lock Part D prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    choices = {
        "Hold the tested worker count": "keep_width",
        "Stop at a smaller fleet": "reduce_width",
        "Upgrade the network tier": "upgrade_network",
        "Reduce coordination delay": "reduce_coordination",
    }
    final_choice = mo.ui.radio(choices, label="Recommendation")
    final_rejected = mo.ui.radio(choices, label="Quantified rejected alternative")
    final_trigger = mo.ui.radio(
        {
            "Scaling efficiency falls below the chosen bound": "efficiency",
            "Repair time increases": "repair",
            "The allowed model age decreases": "freshness",
        },
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {
            "Correlated failures": "correlated_failures",
            "Unmodeled traffic bursts": "traffic_bursts",
            "Model-age policy mismatch": "policy_mismatch",
        },
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Fleet memo rationale",
        placeholder="Cite a saved result, quantify the rejected alternative, and explain the trigger.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_action,
    a_machine_count,
    b_bandwidth,
    b_coordination,
    b_workers,
    c_nodes,
    c_recovery,
    compare_failure_semantics,
    d_duration,
    d_fraction,
    evaluate_c3_scaling,
    evaluate_machine_boundary,
    evaluate_partition_policy,
    scenario,
    serialize_evaluation,
):
    a_baseline_inputs = {
        "model_state": scenario.model_state,
        "memory_per_machine": scenario.memory_per_machine,
        "required_rate": scenario.required_rate,
        "rate_per_machine": scenario.rate_per_machine,
        "coupling": scenario.profile.deployed_coupling,
        "machine_count": 1,
    }
    a_result_inputs = {
        **a_baseline_inputs,
        "model_state": scenario.compact_model_state if a_action.value == "compact_state" else scenario.model_state,
        "machine_count": a_machine_count.value if a_action.value == "add_machines" else 1,
    }
    a_baseline_result = evaluate_machine_boundary(**a_baseline_inputs)
    a_changed_result = evaluate_machine_boundary(**a_result_inputs)
    a_baseline = serialize_evaluation(inputs=a_baseline_inputs, result=a_baseline_result)
    a_result = serialize_evaluation(inputs=a_result_inputs, result=a_changed_result)

    b_baseline_inputs = {
        "workers": 1,
        "single_worker_compute_time": scenario.single_worker_compute_time,
        "communication_payload": scenario.communication_payload,
        "effective_bandwidth": scenario.bandwidth_options["planned"],
        "communication_startup": scenario.communication_startup,
        "coordination_base": scenario.coordination_base,
        "coordination_per_added_worker": scenario.coordination_options["planned"],
        "useful_work_per_step": scenario.useful_work_per_step,
        "overlap_fraction": scenario.overlap_fraction,
    }
    b_result_inputs = {
        **b_baseline_inputs,
        "workers": b_workers.value,
        "effective_bandwidth": scenario.bandwidth_options[b_bandwidth.value],
        "coordination_per_added_worker": scenario.coordination_options[b_coordination.value],
    }
    b_baseline_result = evaluate_c3_scaling(**b_baseline_inputs)
    b_changed_result = evaluate_c3_scaling(**b_result_inputs)
    b_baseline = serialize_evaluation(inputs=b_baseline_inputs, result=b_baseline_result)
    b_result = serialize_evaluation(inputs=b_result_inputs, result=b_changed_result)

    c_baseline_inputs = {
        "nodes": scenario.fleet_size_options[0],
        "component_mtbf": scenario.component_mtbf,
        "observation_horizon": scenario.observation_horizon,
        "recovery_time": scenario.recovery_options["planned"],
    }
    c_result_inputs = {
        **c_baseline_inputs,
        "nodes": c_nodes.value,
        "recovery_time": scenario.recovery_options[c_recovery.value],
    }
    c_baseline_result = compare_failure_semantics(**c_baseline_inputs)
    c_changed_result = compare_failure_semantics(**c_result_inputs)
    c_baseline = serialize_evaluation(inputs=c_baseline_inputs, result=c_baseline_result)
    c_result = serialize_evaluation(inputs=c_result_inputs, result=c_changed_result)

    d_common_inputs = {
        "request_rate": scenario.request_rate,
        "observation_horizon": scenario.observation_horizon,
        "partition_duration": scenario.partition_duration_options[d_duration.value],
        "partitioned_fraction": scenario.partitioned_fraction_options[d_fraction.value],
        "last_confirmed_model_age": scenario.last_confirmed_model_age,
    }
    d_wait_inputs = {"policy": "wait_for_fresh", **d_common_inputs}
    d_stale_inputs = {"policy": "serve_last_confirmed", **d_common_inputs}
    d_wait_result = evaluate_partition_policy(**d_wait_inputs)
    d_stale_result = evaluate_partition_policy(**d_stale_inputs)
    d_baseline = serialize_evaluation(inputs=d_wait_inputs, result=d_wait_result)
    d_result = serialize_evaluation(inputs=d_stale_inputs, result=d_stale_result)
    return (
        a_baseline,
        a_baseline_result,
        a_changed_result,
        a_result,
        b_baseline,
        b_baseline_result,
        b_changed_result,
        b_result,
        c_baseline,
        c_baseline_result,
        c_changed_result,
        c_result,
        d_baseline,
        d_result,
        d_stale_result,
        d_wait_result,
    )


@app.cell
def _(
    a_action,
    a_baseline,
    a_machine_count,
    a_prediction,
    a_result,
    b_bandwidth,
    b_baseline,
    b_coordination,
    b_prediction,
    b_result,
    b_workers,
    c_baseline,
    c_nodes,
    c_prediction,
    c_recovery,
    c_result,
    capture_evidence,
    d_baseline,
    d_duration,
    d_fraction,
    d_policy,
    d_prediction,
    d_result,
    mo,
    set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"action": a_action.value, "machine_count": a_machine_count.value}
    b_upstream = {"workers": b_workers.value, "network": b_bandwidth.value, "coordination": b_coordination.value}
    c_upstream = {"nodes": c_nodes.value, "recovery": c_recovery.value}
    d_upstream = {"duration": d_duration.value, "scope": d_fraction.value, "decision": d_policy.value}
    a_capture = mo.ui.button(
        label="Capture capacity contrast",
        kind="success",
        disabled=a_prediction.value is None or a_baseline["inputs"] == a_result["inputs"],
        on_click=lambda _value: store(
            "A",
            capture_evidence(
                track=track_id,
                part="A",
                prediction=a_prediction.value,
                inputs=a_upstream,
                baseline=a_baseline,
                result=a_result,
                alternatives=(a_baseline, a_result),
                decision=a_action.value,
                upstream_inputs=a_upstream,
                model_key="v2_01_experiments.evaluate_machine_boundary",
            ),
        ),
    )
    b_capture = mo.ui.button(
        label="Capture scaling contrast",
        kind="success",
        disabled=b_prediction.value is None or b_baseline["inputs"] == b_result["inputs"],
        on_click=lambda _value: store(
            "B",
            capture_evidence(
                track=track_id,
                part="B",
                prediction=b_prediction.value,
                inputs=b_upstream,
                baseline=b_baseline,
                result=b_result,
                alternatives=(b_baseline, b_result),
                decision=b_workers.value,
                upstream_inputs=b_upstream,
                model_key="v2_01_experiments.evaluate_c3_scaling",
            ),
        ),
    )
    c_capture = mo.ui.button(
        label="Capture failure-semantics contrast",
        kind="success",
        disabled=c_prediction.value is None or c_baseline["inputs"] == c_result["inputs"],
        on_click=lambda _value: store(
            "C",
            capture_evidence(
                track=track_id,
                part="C",
                prediction=c_prediction.value,
                inputs=c_upstream,
                baseline=c_baseline,
                result=c_result,
                alternatives=(c_baseline, c_result),
                decision=c_recovery.value,
                upstream_inputs=c_upstream,
                model_key="v2_01_experiments.compare_failure_semantics",
            ),
        ),
    )
    d_capture = mo.ui.button(
        label="Capture partition trade-off",
        kind="success",
        disabled=d_prediction.value is None or d_policy.value is None,
        on_click=lambda _value: store(
            "D",
            capture_evidence(
                track=track_id,
                part="D",
                prediction=d_prediction.value,
                inputs=d_upstream,
                baseline=d_baseline,
                result=d_result,
                alternatives=(d_baseline, d_result),
                decision=d_policy.value,
                chosen_result=d_baseline if d_policy.value == "wait_for_fresh" else d_result,
                result_role=(
                    "rejected alternative"
                    if d_policy.value == "wait_for_fresh"
                    else "tested intervention"
                ),
                upstream_inputs=d_upstream,
                model_key="v2_01_experiments.evaluate_partition_policy",
            ),
        ),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = """
        <style>
        .fleet-head{background:linear-gradient(135deg,#101827,#164e63);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
        .fleet-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
        .fleet-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.fleet-head p{color:#cffafe;max-width:780px}
        .fleet-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.fleet-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
        .fleet-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
        .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
        @media(max-width:520px){.fleet-head{border-radius:9px;margin-top:30px}.fleet-meta{grid-template-columns:1fr}}
        </style>
        """
    header = mo.Html(
        f"""
        {css}
        <section class="fleet-head">
          <div class="fleet-top"><span>VOLUME II · LAB 01</span><span>ABOUT 45–55 MIN</span></div>
          <h1>The Fleet Is the System</h1>
          <p>When does adding another machine create useful work, and when does fleet coupling erase the gain?</p>
          <div class="fleet-meta">
            <div><b>Track</b><br>{profile.label}</div>
            <div><b>Deployment</b><br>{profile.deployed_system}</div>
            <div><b>Scaling system</b><br>{profile.scaling_system}</div>
            <div><b>Deliverable</b><br>Fleet operating memo</div>
          </div>
        </section>
        """
    )
    note = mo.Html(
        '<p class="fleet-note">Select the track. All values are analytical outputs from explicit illustrative scenarios. Each Calculation Notes panel states the model boundary.</p>'
    )
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, header, track, note], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS,
    a_action,
    a_baseline_result,
    a_capture,
    a_changed_result,
    a_machine_count,
    a_prediction,
    a_upstream,
    apply_plotly_theme,
    audit_evidence,
    b_bandwidth,
    b_baseline_result,
    b_capture,
    b_changed_result,
    b_coordination,
    b_prediction,
    b_upstream,
    b_workers,
    c_baseline_result,
    c_capture,
    c_changed_result,
    c_nodes,
    c_prediction,
    c_recovery,
    c_upstream,
    d_capture,
    d_duration,
    d_fraction,
    d_policy,
    d_prediction,
    d_stale_result,
    d_upstream,
    d_wait_result,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    go,
    mo,
    profile,
    rationale,
    scenario,
    track_id,
):
    _captures = get_evidence()
    upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream}
    audit = audit_evidence(
        _captures,
        track=track_id,
        required_parts=tuple("ABCD"),
        per_part_upstream_inputs=upstream,
        contrast_required_parts=tuple("ABCD"),
    )

    def q(value, unit, digits=2):
        return f"{value.m_as(unit):,.{digits}f} {unit}"

    def pct(value):
        return f"{100 * value:.2f}%"

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this experiment."), kind="danger")
        data = capture.to_dict()
        return mo.Html(
            f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later control changes cannot rewrite this record.</small></div>'
        )

    def part_a():
        setup = mo.md(
            f"### A · When does distribution become necessary? (10 min)\n"
            f"The **{profile.deployed_system}** begin with {q(scenario.model_state, scenario.memory_display_unit, 1)} of state "
            f"({q(scenario.memory_per_machine, scenario.memory_display_unit, 1)} per machine) and {q(scenario.required_rate, 'count/second', 1)} required rate "
            f"({q(scenario.rate_per_machine, 'count/second', 1)} per machine). Both memory and throughput rate constraints must be satisfied. "
            f"Predict whether adding machines, reducing local state, or neither can cross the boundary."
        )
        if a_prediction.value is None:
            return mo.vstack([setup, mo.hstack([a_action, a_machine_count], widths="equal", wrap=True), a_prediction])
        rows = [
            {
                "Run": "Baseline",
                "Machines": a_baseline_result.machine_count,
                "State per machine": q(a_baseline_result.effective_state_per_machine, scenario.memory_display_unit, 1),
                "Memory margin": q(a_baseline_result.effective_memory_margin, scenario.memory_display_unit, 1),
                "Required rate per machine": q(a_baseline_result.effective_required_rate_per_machine, "count/second", 1),
                "Available rate per machine": q(a_baseline_result.rate_per_machine, "count/second", 1),
                "Feasible": "PASS" if a_baseline_result.feasible else "FAIL",
                "Consequence": a_baseline_result.remedy,
            },
            {
                "Run": "Changed",
                "Machines": a_changed_result.machine_count,
                "State per machine": q(a_changed_result.effective_state_per_machine, scenario.memory_display_unit, 1),
                "Memory margin": q(a_changed_result.effective_memory_margin, scenario.memory_display_unit, 1),
                "Required rate per machine": q(a_changed_result.effective_required_rate_per_machine, "count/second", 1),
                "Available rate per machine": q(a_changed_result.rate_per_machine, "count/second", 1),
                "Feasible": "PASS" if a_changed_result.feasible else "FAIL",
                "Consequence": a_changed_result.remedy,
            },
        ]
        consequence = (
            "Independent devices do not pool memory or rate; each device must satisfy both limits."
            if profile.deployed_coupling == "independent_devices"
            else "Coupled workers may shard state and aggregate rate in this first-order boundary."
        )
        return mo.vstack(
            [
                setup,
                mo.hstack([a_action, a_machine_count], widths="equal", wrap=True),
                a_prediction,
                table(rows),
                mo.callout(
                    mo.md(f"**Your prediction:** {a_prediction.value}. **Observed consequence:** {consequence}"),
                    kind="info",
                ),
                a_capture,
                saved("A"),
                mo.accordion(
                    {"Calculation Notes": mo.md("The test compares model-state bytes with memory bytes and required request rate with per-machine rate. Coupled machines may divide state and demand evenly. Independent deployment devices must each satisfy both limits; fleet size is noncausal for local fit.")}
                ),
            ]
        )

    def part_b():
        setup = mo.md(
            f"### B · When does another worker stop helping? (12 min)\nStrong-scale one fixed step on the **{profile.scaling_system}**. Choose a width, network tier, and coordination condition, then predict the dominant term."
        )
        if b_prediction.value is None:
            return mo.vstack([setup, mo.hstack([b_workers, b_bandwidth, b_coordination], widths="equal", wrap=True), b_prediction])
        figure = go.Figure()
        for label, result in (("One worker", b_baseline_result), ("Changed", b_changed_result)):
            for name, value, color in (
                ("Compute", result.compute_time, COLORS["BlueLine"]),
                ("Communication", result.communication_time, COLORS["OrangeLine"]),
                ("Coordination", result.coordination_time, COLORS["GreenLine"]),
            ):
                figure.add_bar(
                    name=name,
                    x=[label],
                    y=[value.m_as("second")],
                    marker_color=color,
                    legendgroup=name,
                    showlegend=label == "One worker",
                )
        figure.update_layout(
            barmode="stack",
            height=285,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Analytical step time (s)",
            legend_orientation="h",
        )
        rows = [
            {
                "Run": "One worker",
                "Step time": q(b_baseline_result.step_time, "second"),
                "Useful throughput": q(b_baseline_result.useful_throughput, "count/second"),
                "Efficiency": pct(b_baseline_result.scaling_efficiency),
                "Dominant": b_baseline_result.dominant_term,
            },
            {
                "Run": f"{b_changed_result.workers:,} workers",
                "Step time": q(b_changed_result.step_time, "second"),
                "Useful throughput": q(b_changed_result.useful_throughput, "count/second"),
                "Efficiency": pct(b_changed_result.scaling_efficiency),
                "Dominant": b_changed_result.dominant_term,
            },
        ]
        return mo.vstack(
            [
                setup,
                mo.hstack([b_workers, b_bandwidth, b_coordination], widths="equal", wrap=True),
                b_prediction,
                apply_plotly_theme(figure),
                table(rows),
                mo.callout(
                    mo.md(f"**Your prediction:** {b_prediction.value}. **Observed consequence:** useful throughput is {q(b_changed_result.useful_throughput, 'count/second')}, while scaling efficiency is {pct(b_changed_result.scaling_efficiency)}."),
                    kind="danger" if b_changed_result.scaling_efficiency < 0.5 else "info",
                ),
                b_capture,
                saved("B"),
                mo.accordion(
                    {"Calculation Notes": mo.md("Fixed work gives compute time = one-worker compute time / workers. Step time adds compute, one disclosed communication transfer, and per-worker coordination, then subtracts bounded compute–communication overlap. Useful throughput is fixed work / step time; scaling efficiency compares it with ideal linear throughput. No collective algorithm is selected or optimized.")}
                ),
            ]
        )

    def part_c():
        independent_system = (
            "an independent serving pool"
            if profile.deployed_coupling == "coupled_job"
            else profile.deployed_system
        )
        setup = mo.md(
            "### C · Does reliable hardware make a reliable fleet? (10 min)\n"
            f"Use the same independent node failure process for a synchronized job and an independently operating deployment ({independent_system}). "
            "Predict whether one event has the same consequence."
        )
        if c_prediction.value is None:
            return mo.vstack([setup, mo.hstack([c_nodes, c_recovery], widths="equal", wrap=True), c_prediction])
        figure = go.Figure(
            [
                go.Bar(
                    name="At least one event over horizon",
                    x=["Baseline fleet", "Changed fleet"],
                    y=[100 * c_baseline_result.probability_any_node_event, 100 * c_changed_result.probability_any_node_event],
                    marker_color=COLORS["OrangeLine"],
                ),
                go.Bar(
                    name="Stationary capacity unavailable",
                    x=["Baseline fleet", "Changed fleet"],
                    y=[
                        100 * c_baseline_result.independent_unavailable_capacity_fraction,
                        100 * c_changed_result.independent_unavailable_capacity_fraction,
                    ],
                    marker_color=COLORS["BlueLine"],
                ),
            ]
        )
        figure.update_layout(
            barmode="group",
            height=285,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Probability or capacity share (%)",
            legend_orientation="h",
        )
        rows = [
            {
                "Run": "Baseline fleet",
                "Nodes": f"{c_baseline_result.nodes:,}",
                "Event probability over horizon": pct(c_baseline_result.probability_any_node_event),
                "Coupled-job effect": "whole job interrupted",
                "Independent capacity available": pct(c_baseline_result.independent_available_capacity_fraction),
            },
            {
                "Run": "Changed fleet",
                "Nodes": f"{c_changed_result.nodes:,}",
                "Event probability over horizon": pct(c_changed_result.probability_any_node_event),
                "Coupled-job effect": "whole job interrupted",
                "Independent capacity available": pct(c_changed_result.independent_available_capacity_fraction),
            },
        ]
        return mo.vstack(
            [
                setup,
                mo.hstack([c_nodes, c_recovery], widths="equal", wrap=True),
                c_prediction,
                apply_plotly_theme(figure),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {c_prediction.value}. **Observed consequence:** a node event interrupts the coupled job. "
                        f"In the independent deployment ({independent_system}), it removes only that node while repair-time availability determines the remaining capacity."
                    ),
                    kind="info",
                ),
                c_capture,
                saved("C"),
                mo.accordion(
                    {"Calculation Notes": mo.md("The event probability is evaluated over the displayed observation horizon, starting with all nodes operational. Independent capacity uses stationary repairable availability A = MTBF / (MTBF + repair time). These are different quantities: a horizon event probability is not instantaneous unavailable capacity. Failures are independent; correlated domains are a stated limitation.")}
                ),
            ]
        )

    def part_d():
        setup = mo.md(
            "### D · What must remain available during a partition? (10 min)\nCompare two explicit policies under the same request rate, duration, and affected scope. Then choose the consequence your deployment can accept."
        )
        if d_prediction.value is None:
            return mo.vstack([setup, mo.hstack([d_duration, d_fraction], widths="equal", wrap=True), d_prediction])
        figure = go.Figure(
            [
                go.Bar(
                    name="Unavailable",
                    x=["Wait for fresh", "Serve last confirmed"],
                    y=[d_wait_result.unavailable_requests.m_as("count"), d_stale_result.unavailable_requests.m_as("count")],
                    marker_color=COLORS["OrangeLine"],
                ),
                go.Bar(
                    name="Stale",
                    x=["Wait for fresh", "Serve last confirmed"],
                    y=[d_wait_result.stale_requests.m_as("count"), d_stale_result.stale_requests.m_as("count")],
                    marker_color=COLORS["BlueLine"],
                ),
            ]
        )
        figure.update_layout(
            barmode="stack",
            height=275,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Affected requests (count)",
            legend_orientation="h",
        )
        rows = [
            {
                "Policy": "Wait for fresh",
                "Unavailable": q(d_wait_result.unavailable_requests, "count", 0),
                "Stale": q(d_wait_result.stale_requests, "count", 0),
                "Maximum model age": q(d_wait_result.maximum_model_age, "minute", 1),
            },
            {
                "Policy": "Serve last confirmed",
                "Unavailable": q(d_stale_result.unavailable_requests, "count", 0),
                "Stale": q(d_stale_result.stale_requests, "count", 0),
                "Maximum model age": q(d_stale_result.maximum_model_age, "minute", 1),
            },
        ]
        return mo.vstack(
            [
                setup,
                mo.hstack([d_duration, d_fraction], widths="equal", wrap=True),
                d_prediction,
                apply_plotly_theme(figure),
                table(rows),
                d_policy,
                mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. **Observed consequence:** both policies see {q(d_wait_result.affected_requests, 'count', 0)} affected requests. Waiting makes them unavailable; serving continues with older state."), kind="danger"),
                d_capture,
                saved("D"),
                mo.accordion(
                    {"Calculation Notes": mo.md("Affected requests = request rate × partition duration × partitioned fraction. Waiting counts them as unavailable. Serving the last confirmed model counts them as stale and adds partition duration to model age. The model assigns no universal score to either policy.")}
                ),
            ]
        )

    def build_synthesis():
        rows = []
        for part in "ABCD":
            capture = _captures.get(part)
            if capture is None:
                state = "MISSING"
                prediction = "—"
            elif part in audit.stale or (part, part) in audit.identical_pairs:
                state = "STALE"
                prediction = capture.to_dict()["prediction"]
            else:
                state = "CURRENT"
                prediction = capture.to_dict()["prediction"]
            rows.append({"Part": part, "Original prediction": prediction, "Evidence": state})
        complete = (
            audit.complete
            and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
            and final_choice.value != final_rejected.value
            and bool(rationale.value.strip())
        )
        status = "**Ready for the local fleet memo.**" if complete else "Capture four current contrasts, choose different recommendation and rejected options, and complete the rationale."
        return mo.vstack(
            [
                mo.md("### Synthesis · Where should scaling stop? (6 min)\nUse the saved capacity, C³, failure, and partition evidence. Choose one option, quantify a rejected alternative in your rationale, name a remaining limitation, and set a reevaluation trigger."),
                table(rows),
                mo.callout(mo.md("Saved snapshots remain fixed while live controls move. Recapture stale evidence before reporting."), kind="info"),
                mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
                mo.hstack([final_trigger, final_risk], widths="equal", wrap=True),
                rationale,
                mo.callout(mo.md(status), kind="success" if complete else "warn"),
            ]
        )

    tabs = mo.ui.tabs(
        {
            "Part A · Capacity": part_a(),
            "Part B · C³ Scaling": part_b(),
            "Part C · Failure": part_c(),
            "Part D · Partition": part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    tabs
    return (audit,)


@app.cell
def _(
    audit,
    build_lab_report,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    get_lab_metadata,
    mo,
    profile,
    rationale,
    report_export_panel,
    track_id,
):
    _captures = get_evidence()
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and final_choice.value != final_rejected.value
        and bool(rationale.value.strip())
    )
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCD"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_01_introduction.py"),
        track=track_id,
        scenario=f"{profile.deployed_system}; {profile.scaling_system}",
        learning_objectives=[
            (
                "Find a per-machine capacity boundary without pooling independent-device memory"
                if profile.deployed_coupling == "independent_devices"
                else "Find a capacity boundary where coupled workers shard state and aggregate rate"
            ),
            "Decompose useful fleet throughput into compute, communication, and coordination",
            "Distinguish coupled-job interruption from repairable independent capacity loss",
            "Quantify unavailable and stale requests during a partition",
        ],
        predictions={part: snapshots[part]["prediction"] for part in "ABCD"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCD"},
        evidence_summary={
            part: {
                "baseline": snapshots[part]["baseline"],
                "result": snapshots[part]["result"],
                "alternatives": snapshots[part]["alternatives"],
            }
            for part in "ABCD"
        },
        binding_constraints={
            "A": snapshots["A"]["result"]["outputs"]["remedy"],
            "B": snapshots["B"]["result"]["outputs"]["dominant_term"],
            "C": "coupled interruption versus repairable independent capacity",
            "D": snapshots["D"]["decision"],
        },
        decisions={
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "reevaluation_trigger": final_trigger.value,
        },
        final_decision={
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "rationale": rationale.value,
        },
        big_takeaways=[
            (
                "Independent devices do not pool memory."
                if profile.deployed_coupling == "independent_devices"
                else "Coupled workers may shard state and aggregate rate in this first-order boundary."
            ),
            "Useful fleet throughput depends on all three C³ time terms.",
            "Failure consequences depend on workload coupling and repair time.",
            "Partition policy trades unavailable requests against older model state.",
        ],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={
            "track": track_id,
            "captures": snapshots,
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "reevaluation_trigger": final_trigger.value,
            "residual_risk": final_risk.value,
        },
        source_trace={
            "scenario": "Explicit illustrative fleet assumptions; analytical rather than measured.",
            "calculations": "Unit-aware fleet capacity, C³, failure, and partition models.",
        },
    )
    mo.vstack([mo.md("## Local fleet evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    ledger,
    mo,
    rationale,
    track_id,
):
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and final_choice.value != final_rejected.value
        and bool(rationale.value.strip())
    )
    status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(
                chapter=1,
                design={
                    "schema_version": 1,
                    "lab_id": "v2_01",
                    "track_id": track_id,
                    "model_id": "v2_01_experiments",
                    "evidence": {part: capture.to_dict() for part, capture in get_evidence().items()},
                    "recommendation": final_choice.value,
                    "rejected_alternative": final_rejected.value,
                    "reevaluation_trigger": final_trigger.value,
                    "residual_risk": final_risk.value,
                    "rationale": rationale.value,
                },
            )
            await ledger.flush()
            status = "SAVED"
        except Exception as error:
            status = f"SAVE FAILED · {type(error).__name__}"
    mo.Html(
        f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">· 01 · The Fleet Is the System</span><span style="flex:1"></span><span class="hud-label">STATUS</span><span class="hud-active">· {status}</span></div>'
    )
    return


if __name__ == "__main__":
    app.run()
