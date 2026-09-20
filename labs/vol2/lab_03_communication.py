import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 03: Network Fabrics · MLSysBook")


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
    from mlsysim.core.units import Q_
    from mlsysim.engine.v2_03_experiments import (
        OVERLAP_WINDOW_OPTIONS, OVERLAP_WINDOW_QUANTITIES, PAYLOAD_OPTIONS,
        PAYLOAD_QUANTITIES, TRAFFIC_PATTERN_OPTIONS, alpha_beta_experiment,
        burst_spacing_params, congestion_experiment, default_overlap_label,
        default_payload_label, default_traffic_pattern_label,
        equal_budget_comparison, get_track_scenario, telemetry_experiment,
        topology_experiment, topology_options,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata,
        report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, OVERLAP_WINDOW_OPTIONS,
        OVERLAP_WINDOW_QUANTITIES, PAYLOAD_OPTIONS, PAYLOAD_QUANTITIES, Q_,
        TRAFFIC_PATTERN_OPTIONS, alpha_beta_experiment, apply_plotly_theme,
        audit_evidence, build_lab_report, burst_spacing_params,
        capture_evidence, congestion_experiment, default_overlap_label,
        default_payload_label, default_traffic_pattern_label,
        equal_budget_comparison, get_lab_metadata, get_track_scenario, go,
        ledger, mo, report_export_panel, telemetry_experiment,
        topology_experiment, topology_options,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="Cloud", label="Deployment track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_track_scenario, track):
    track_id = track.value
    scenario = get_track_scenario(track_id)
    return scenario, track_id


@app.cell
def _(
    OVERLAP_WINDOW_OPTIONS, PAYLOAD_OPTIONS, TRAFFIC_PATTERN_OPTIONS,
    burst_spacing_params, default_overlap_label, default_payload_label,
    default_traffic_pattern_label, mo, scenario, topology_options, track_id,
):
    _track_key = track_id
    link_labels = {profile.label: key for key, profile in scenario.links.items()}
    default_link_label = scenario.links[scenario.default_link_id].label
    link = mo.ui.dropdown(link_labels, value=default_link_label, label="Communication path")
    payload_choice = mo.ui.dropdown(
        PAYLOAD_OPTIONS,
        value=default_payload_label(track_id),
        label="Payload size",
    )
    top_options = topology_options(
        scenario.participants, scenario.links[scenario.default_link_id].bandwidth, track_id=track_id,
    )
    topology_labels = {top.label: key for key, top in top_options.items()}
    topology_choice = mo.ui.dropdown(
        topology_labels, value=top_options["aligned"].label, label="Topology to test",
    )
    isolation = mo.ui.slider(
        0.5, 0.9, value=0.8, step=0.1, label="Foreground capacity reservation",
    )
    sp = burst_spacing_params(track_id)
    burst_spacing_ms = mo.ui.slider(
        sp["start"], sp["stop"], value=sp["value"], step=sp["step"], label=sp["label"],
    )
    fault = mo.ui.dropdown(
        {"Capacity loss": "capacity", "Startup latency": "startup", "Traffic hot spot": "traffic_hotspot"},
        value="Traffic hot spot", label="Suspected fault",
    )
    design_labels = {top.label: key for key, top in top_options.items()}
    first_design = mo.ui.dropdown(design_labels, value=top_options["nonblocking"].label, label="Plan 1")
    second_design = mo.ui.dropdown(design_labels, value=top_options["grouped"].label, label="Plan 2")
    new_mix = mo.ui.dropdown(
        TRAFFIC_PATTERN_OPTIONS,
        value=default_traffic_pattern_label(track_id),
        label="New traffic mix",
    )
    overlap_choice = mo.ui.dropdown(
        OVERLAP_WINDOW_OPTIONS,
        value=default_overlap_label(track_id),
        label="Dependency-ready compute",
    )
    return (
        burst_spacing_ms, fault, first_design, isolation, link, new_mix,
        overlap_choice, payload_choice, second_design, topology_choice,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Startup latency": "startup", "Payload serialization": "serialization"},
        label="Which term dominates the selected transfer?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Non-blocking": "nonblocking", "Aligned rails": "aligned", "Grouped": "grouped", "Oversubscribed": "oversubscribed"},
        label="Which topology has the lowest p95 completion time?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Tail improves": "improves", "Tail worsens": "worsens", "No change": "same"},
        label="What happens after reserving foreground capacity?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Lane busy time and queue": "capacity", "Per-flow startup": "startup", "Concentrated lane bytes": "traffic_hotspot"},
        label="Which counter should identify the selected fault?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Plan 1 remains better": "first", "Plan 2 becomes better": "second", "Neither is feasible": "none"},
        label="Which equal-budget plan survives the new traffic mix?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    choices = {
        "Non-blocking": "nonblocking", "Aligned rails": "aligned",
        "Grouped": "grouped", "Oversubscribed": "oversubscribed",
        "Hold: no feasible plan": "none",
    }
    final_choice = mo.ui.radio(choices, label="Recommended plan")
    final_rejected = mo.ui.radio(
        {key: value for key, value in choices.items() if value != "none"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"p95 exceeds budget": "p95", "Traffic matrix changes": "traffic", "Reservation harms background work": "capacity"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Illustrative traffic assumptions": "traffic_assumptions", "Unmodeled routing dynamics": "routing", "Path measurements may drift": "measurement_drift"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Cite the chosen plan, quantified rejected plan, remaining limitation, and trigger.",
    )
    e_choice = mo.ui.radio(
        {
            "Approve Plan 1": "first", "Approve Plan 2": "second",
            "Hold: no feasible plan": "none",
        },
        label="Decision after reviewing both plans",
    )
    return e_choice, final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    OVERLAP_WINDOW_QUANTITIES, PAYLOAD_QUANTITIES, Q_, alpha_beta_experiment,
    burst_spacing_ms, congestion_experiment, equal_budget_comparison,
    fault, first_design, isolation, link, new_mix, overlap_choice,
    payload_choice, scenario, second_design, telemetry_experiment,
    topology_choice, topology_experiment, track_id,
):
    selected_payload = PAYLOAD_QUANTITIES[payload_choice.value]
    selected_overlap = OVERLAP_WINDOW_QUANTITIES[overlap_choice.value]
    a_baseline = alpha_beta_experiment(track_id, link.value, Q_(1, "byte"))
    a_result = alpha_beta_experiment(track_id, link.value, selected_payload)
    b_rows = topology_experiment(
        track_id, link_id=link.value, participants=scenario.participants,
        payload=selected_payload, traffic_pattern=scenario.traffic_pattern,
    )
    b_by_id = {row["topology_id"]: row for row in b_rows}
    b_baseline = b_by_id["nonblocking"]
    b_result = b_by_id[topology_choice.value]
    c_baseline = congestion_experiment(
        track_id, topology_choice.value, isolation_fraction=0,
        burst_spacing=Q_(burst_spacing_ms.value, "ms"),
    )
    c_result = congestion_experiment(
        track_id, topology_choice.value, isolation_fraction=isolation.value,
        burst_spacing=Q_(burst_spacing_ms.value, "ms"),
    )
    d_evidence = telemetry_experiment(track_id, topology_choice.value, fault=fault.value)
    e_comparison = equal_budget_comparison(
        track_id, first_design.value, second_design.value,
        overlap_window=selected_overlap,
        traffic_pattern=new_mix.value,
    )
    return (
        a_baseline, a_result, b_baseline, b_by_id, b_result, b_rows,
        c_baseline, c_result, d_evidence, e_comparison, selected_overlap,
        selected_payload,
    )


@app.cell
def _(
    a_baseline, a_prediction, a_result, b_baseline, b_by_id, b_prediction,
    b_result, c_baseline, c_prediction, c_result, capture_evidence,
    d_evidence, d_prediction, e_choice, e_comparison, e_prediction, first_design, mo,
    link, payload_choice, second_design, set_evidence, topology_choice, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    stable_upstream = {"track_id": track_id}
    b_upstream = {"track_id": track_id, "link_id": link.value, "payload": payload_choice.value}
    cd_upstream = {"track_id": track_id, "topology_id": topology_choice.value}
    a_capture = mo.ui.button(
        label="Capture crossover evidence", kind="success",
        disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"selected_link": a_result["link_id"]}, baseline=a_baseline,
            result=a_result, upstream_inputs=stable_upstream,
            decision=a_result["binding_term"],
            model_key="v2_03_experiments.alpha_beta_experiment",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture topology comparison", kind="success",
        disabled=b_prediction.value is None or topology_choice.value == "nonblocking",
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"selected_topology": topology_choice.value},
            baseline=b_baseline, result=b_result,
            alternatives=tuple(b_by_id.values()), upstream_inputs=b_upstream,
            decision=topology_choice.value,
            model_key="v2_03_experiments.topology_experiment",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture isolation trade-off", kind="success",
        disabled=c_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"topology_id": topology_choice.value, "reservation": c_result["inputs"]["isolation_fraction"]},
            baseline=c_baseline, result=c_result, upstream_inputs=cd_upstream,
            decision="reserve" if c_result["within_budget"] else "do_not_reserve",
            model_key="v2_03_experiments.congestion_experiment",
        )),
    )
    d_capture = mo.ui.button(
        label="Capture diagnosis test", kind="success",
        disabled=d_prediction.value is None,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs=d_evidence["inputs"], baseline=d_evidence["baseline"],
            result=d_evidence["result"],
            alternatives=({"counter_to_watch": d_evidence["counter_to_watch"]},),
            upstream_inputs=cd_upstream, decision=d_evidence["supports_diagnosis"],
            model_key="v2_03_experiments.telemetry_experiment",
        )),
    )
    e_model_result = e_comparison["winner"] if (
        e_comparison["first"]["valid_plan"] or e_comparison["second"]["valid_plan"]
    ) else "none"
    e_student_decision = (
        first_design.value if e_choice.value == "first"
        else second_design.value if e_choice.value == "second"
        else "none"
    )
    e_chosen_result = (
        e_comparison["first"] if e_choice.value == "first"
        else e_comparison["second"] if e_choice.value == "second"
        else None
    )
    e_capture = mo.ui.button(
        label="Capture equal-budget decision", kind="success",
        disabled=e_prediction.value is None or e_choice.value is None or first_design.value == second_design.value,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs=e_comparison["inputs"], baseline=e_comparison["first"],
            result=e_comparison["second"],
            alternatives=(e_comparison["first"], e_comparison["second"]),
            upstream_inputs=stable_upstream, decision=e_student_decision,
            chosen_result=e_chosen_result, result_role="comparison alternative",
            model_key="v2_03_experiments.equal_budget_comparison",
        )),
    )
    return (
        a_capture, b_capture, b_upstream, c_capture, cd_upstream, d_capture,
        e_capture, e_model_result, stable_upstream,
    )


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, scenario, track):
    css = mo.Html("""
    <style>
    .fabric-head{background:linear-gradient(135deg,#10233d,#185a73);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin:10px 0 14px}
    .fabric-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .fabric-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.fabric-head p{color:#d9f4ff;max-width:780px}
    .fabric-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:9px;margin-top:17px}.fabric-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#10233d!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a9c9d8}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    .saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}
    @media(max-width:520px){.fabric-head{border-radius:9px;margin-top:30px}.fabric-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="fabric-head"><div class="fabric-top"><span>VOLUME II · LAB 03</span><span>ABOUT 50–55 MIN</span></div><h1>When Does the Fabric Become the Computer?</h1><p>Predict which communication constraint binds, test the traffic path, and defend a fleet fabric plan with saved evidence.</p><div class="fabric-meta"><div><b>Fleet unit</b><br>{scenario.fleet_unit}</div><div><b>Output</b><br>Network design review</div><div><b>Method</b><br>Five controlled contrasts</div></div></section>""")
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, track, header]).style({"padding-top": "32px"})
    return


@app.cell
def _(
    audit_evidence, b_upstream, cd_upstream, get_evidence, go, mo,
    stable_upstream, track_id,
):
    captures = get_evidence()
    per_part_upstream = {
        "A": stable_upstream,
        "B": b_upstream,
        "C": cd_upstream,
        "D": cd_upstream,
        "E": stable_upstream,
    }
    audit = audit_evidence(
        captures, track=track_id, required_parts=("A", "B", "C", "D", "E"),
        per_part_upstream_inputs=per_part_upstream,
        contrast_required_parts=("A", "B", "C", "D", "E"),
    )

    def table(rows):
        if not rows:
            return mo.md("No rows.")
        columns = list(rows[0])
        head = "".join(f"<th>{column}</th>" for column in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{row[column]}</td>" for column in columns) + "</tr>"
            for row in rows
        )
        return mo.Html(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')

    def saved(part):
        capture = captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this part."), kind="danger")
        snapshot = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · prediction: {snapshot["prediction"]}<br><small>Track {snapshot["track"]}; later control changes do not rewrite this result.</small></div>')

    def bar_chart(labels, values, title, colors):
        figure = go.Figure([go.Bar(
            x=labels, y=values, marker_color=colors,
            text=[f"{value:,.2f}" for value in values], textposition="outside",
        )])
        figure.update_layout(
            height=285, margin=dict(l=55, r=20, t=25, b=60),
            yaxis_title=title, showlegend=False,
        )
        return figure
    return audit, bar_chart, captures, per_part_upstream, saved, table


@app.cell
def _(
    COLORS, a_baseline, a_capture, a_prediction, a_result,
    apply_plotly_theme, audit, b_capture, b_prediction, b_rows, bar_chart,
    burst_spacing_ms, c_baseline, c_capture, c_prediction, c_result,
    captures, d_capture, d_evidence, d_prediction, e_capture, e_choice,
    e_comparison, e_model_result, e_prediction, fault,
    final_choice, final_rejected, final_risk, final_trigger, first_design,
    isolation, link, mo, new_mix, overlap_choice, payload_choice, rationale, saved,
    second_design, table, topology_choice,
):
    def part_a():
        intro = mo.md("### A · When does bandwidth beat lower startup latency? (8 min)\nChoose a path and payload, then predict which term dominates before revealing the transfer decomposition.")
        controls = mo.hstack([link, payload_choice], widths="equal", wrap=True)
        if a_prediction.value is None:
            return mo.vstack([intro, controls, a_prediction])
        figure = bar_chart(
            ["Startup", "Serialization"],
            [a_result["alpha_ms"], a_result["serialization_ms"]],
            "Transfer time (ms)", [COLORS["BlueLine"], COLORS["OrangeLine"]],
        )
        rows = [
            {"Run": "One byte", "Total": f"{a_baseline['total_ms']:.3f} ms", "Binding": a_baseline["binding_term"]},
            {"Run": f"{a_result['inputs']['payload']['value']:g} {a_result['inputs']['payload']['unit']}", "Total": f"{a_result['total_ms']:.3f} ms", "Binding": a_result["binding_term"]},
        ]
        note = f"startup {a_result['alpha_ms']:.3f} ms; serialization {a_result['serialization_ms']:.3f} ms; crossover {a_result['crossover_kb']:.2f} KB"
        return mo.vstack([intro, controls, a_prediction, apply_plotly_theme(figure), table(rows), mo.callout(mo.md(f"**Prediction:** {a_prediction.value}. **Analytical result:** {note}."), kind="info"), a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("MLSysIM evaluates startup latency plus payload divided by path bandwidth. The crossover is the payload where both terms are equal.")})])

    def part_b():
        intro = mo.md("### B · Why do identical links deliver different performance? (10 min)\nHold link, participants, payload, and traffic fixed. Change topology resources and routing lanes.")
        if b_prediction.value is None:
            return mo.vstack([intro, b_prediction])
        rows = [{
            "Role": row.get("role", "—"),
            "Topology": row["label"], "Cut capacity": f"{row['cut_capacity_mb_s']:,.0f} MB/s",
            "p95": f"{row['p95_ms']:.2f} ms", "Queue": f"{row['max_queue_ms']:.2f} ms",
            "Cost assumption": f"${row['cost_usd']:,.0f}",
            "Budget": "PASS" if row["within_budget"] else "FAIL",
        } for row in b_rows]
        figure = bar_chart(
            [row["label"] for row in b_rows], [row["p95_ms"] for row in b_rows],
            "Flow completion p95 (ms)",
            [COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"], COLORS["RedLine"]],
        )
        return mo.vstack([intro, b_prediction, topology_choice, apply_plotly_theme(figure), table(rows), mo.callout(mo.md("Lower structural cost can remove cut capacity and increase queueing under the same traffic."), kind="info"), b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md("MLSysIM counts links and switches/gateways, routes identical finite flows onto shared lanes, and FIFO-schedules their startup and serialization time. Costs and configurations are illustrative planning assumptions, not measured device specs. The p95 is computed from simulated flow completions.")})])

    def part_c():
        intro = mo.md("### C · When does sharing become unacceptable? (10 min)\nA foreground burst competes with background traffic. Reserve physical lane capacity and observe the protected tail and sacrificed background capacity.")
        controls = mo.hstack([isolation, burst_spacing_ms], widths="equal", wrap=True)
        if c_prediction.value is None:
            return mo.vstack([intro, controls, c_prediction])
        rows = [
            {"Case": "Shared", "Foreground p95": f"{c_baseline['foreground_p95_ms']:.2f} ms", "Finish": f"{c_baseline['foreground_completion_ms']:.2f} ms", "Background capacity": f"{c_baseline['capacity_left_for_background_mb_s']:,.0f} MB/s"},
            {"Case": "Reserved", "Foreground p95": f"{c_result['foreground_p95_ms']:.2f} ms", "Finish": f"{c_result['foreground_completion_ms']:.2f} ms", "Background capacity": f"{c_result['capacity_left_for_background_mb_s']:,.0f} MB/s"},
        ]
        figure = bar_chart(
            ["Shared p95", "Reserved p95"],
            [c_baseline["foreground_p95_ms"], c_result["foreground_p95_ms"]],
            "Foreground p95 (ms)", [COLORS["RedLine"], COLORS["GreenLine"]],
        )
        message = f"Reservation dedicates {c_result['reserved_capacity_mb_s']:,.0f} MB/s and leaves {c_result['capacity_left_for_background_mb_s']:,.0f} MB/s for background work. Isolation does not create bandwidth."
        return mo.vstack([intro, controls, c_prediction, apply_plotly_theme(figure), table(rows), mo.callout(mo.md(message), kind="success" if c_result["within_budget"] else "danger"), c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("The shared run schedules both traffic classes on the same lanes. The reserved run gives foreground flows the selected capacity fraction and reports the complementary capacity unavailable to them.")})])

    def part_d():
        intro = mo.md("### D · Which telemetry identifies the bottleneck? (9 min)\nSelect a suspected fault, commit the expected counter, then test one targeted counterfactual.")
        if d_prediction.value is None:
            return mo.vstack([intro, fault, d_prediction])
        rows = [
            {"Run": "Fault present", "p95": f"{d_evidence['baseline_p95_ms']:.2f} ms", "Max queue": f"{d_evidence['baseline_max_queue_ms']:.2f} ms", "Bottleneck": d_evidence["bottleneck_lane"]},
            {"Run": "Targeted intervention", "p95": f"{d_evidence['counterfactual_p95_ms']:.2f} ms", "Max queue": f"{d_evidence['counterfactual_max_queue_ms']:.2f} ms", "Bottleneck": d_evidence["result"]["bottleneck_lane"]},
        ]
        message = f"Inspect {d_evidence['counter_to_watch']}. The targeted intervention changes p95 by {d_evidence['p95_improvement_ms']:.2f} ms."
        return mo.vstack([intro, fault, d_prediction, table(rows), mo.callout(mo.md(message), kind="success" if d_evidence["supports_diagnosis"] else "danger"), d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("Capacity restoration, startup restoration, and traffic redistribution alter different causal inputs. The diagnosis is supported only when its corresponding intervention changes the completion tail.")})])

    def part_e():
        intro = mo.md("### E · Which plan survives a new traffic mix? (10 min)\nCompare two structures under one capital cap, then bound overlap by dependency-ready compute.")
        plans = mo.hstack([first_design, second_design], widths="equal", wrap=True)
        if e_prediction.value is None:
            return mo.vstack([intro, plans, new_mix, e_prediction])
        if first_design.value == second_design.value:
            return mo.vstack([intro, plans, new_mix, e_prediction, mo.callout(mo.md("Choose two different plans to create a valid contrast."), kind="warn")])
        rows = [
            {"Plan": e_comparison["first"].get("label", first_design.value), "Cut lanes": e_comparison["first"]["cut_lanes"], "Cost": f"${e_comparison['first']['cost_usd']:,.0f}", "Unspent": f"${e_comparison['first']['unspent_usd']:,.0f}", "Exposed": f"{e_comparison['first']['exposed_ms']:.2f} ms", "Outcome": "PASS" if e_comparison["first"]["valid_plan"] else "FAIL"},
            {"Plan": e_comparison["second"].get("label", second_design.value), "Cut lanes": e_comparison["second"]["cut_lanes"], "Cost": f"${e_comparison['second']['cost_usd']:,.0f}", "Unspent": f"${e_comparison['second']['unspent_usd']:,.0f}", "Exposed": f"{e_comparison['second']['exposed_ms']:.2f} ms", "Outcome": "PASS" if e_comparison["second"]["valid_plan"] else "FAIL"},
        ]
        message = f"Model comparison: {e_model_result}. Overlap hides at most the displayed dependency-ready window. Record your own decision, including no feasible plan when justified."
        return mo.vstack([intro, plans, mo.hstack([new_mix, overlap_choice], widths="equal", wrap=True), e_prediction, table(rows), mo.callout(mo.md(message), kind="success" if e_model_result != "none" else "danger"), e_choice, e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md("MLSysIM applies one common illustrative capital cap, purchases integer links, schedules the new traffic matrix, and subtracts only explicitly ready compute. Detailed collective algorithms are deferred.")})])

    def synthesis():
        rows = []
        for part in "ABCDE":
            capture = captures.get(part)
            current = capture is not None and part not in audit.stale and (part, part) not in audit.identical_pairs
            rows.append({"Part": part, "Original prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": "CURRENT" if current else ("STALE" if capture else "MISSING")})
        saved_decision = captures["E"].to_dict()["decision"] if "E" in captures else None
        tested_plans = tuple(captures["E"].to_dict()["inputs"].get(key) for key in ("first_topology_id", "second_topology_id")) if "E" in captures else ()
        ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == saved_decision and final_rejected.value in tested_plans
        prompt = "Ready for the local report." if ready else "Complete five current contrasts, match the recommendation to Part E, reject a different tested plan, and write the rationale."
        return mo.vstack([mo.md("### Synthesis · Defend the network design review (5 min)\nName the chosen plan, quantify a rejected alternative, state the remaining limitation, and identify a reevaluation trigger."), table(rows), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale, mo.callout(mo.md(prompt), kind="success" if ready else "warn")])

    tabs = mo.ui.tabs({
        "Part A": part_a(), "Part B": part_b(), "Part C": part_c(),
        "Part D": part_d(), "Part E": part_e(), "Synthesis": synthesis(),
    })
    tabs
    return


@app.cell
def _(
    audit, build_lab_report, captures, final_choice, final_rejected,
    final_risk, final_trigger, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id,
):
    _saved_decision = captures["E"].to_dict()["decision"] if "E" in captures else None
    _tested_plans = tuple(captures["E"].to_dict()["inputs"].get(key) for key in ("first_topology_id", "second_topology_id")) if "E" in captures else ()
    _ready = audit.complete and all(
        widget.value is not None
        for widget in (final_choice, final_rejected, final_trigger, final_risk)
    ) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_decision and final_rejected.value in _tested_plans
    mo.stop(not _ready)
    snapshots = {part: captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_03_communication.py"), track=track_id,
        scenario=scenario.fleet_unit,
        learning_objectives=[
            "Separate startup and serialization costs",
            "Explain topology and traffic contention with finite flows",
            "Defend a resource-counted fabric plan",
        ],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {
            "baseline": snapshots[part]["baseline"],
            "result": snapshots[part]["result"],
            "alternatives": snapshots[part]["alternatives"],
        } for part in "ABCDE"},
        binding_constraints={
            "A": snapshots["A"]["result"]["binding_term"],
            "B": snapshots["B"]["result"]["bottleneck_lane"],
            "C": snapshots["C"]["result"]["bottleneck_lane"],
            "D": snapshots["D"]["result"]["bottleneck_lane"],
            "E": snapshots["E"]["decision"],
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
            "Message size determines whether startup or serialization binds.",
            "Traffic placement determines which physical lane becomes the bottleneck.",
            "Isolation and topology consume countable capacity and capital.",
        ],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={
            "schema_version": 1, "lab_id": "v2_03", "track_id": track_id,
            "model_id": "v2_03_experiments", "evidence": snapshots,
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "reevaluation_trigger": final_trigger.value,
            "residual_risk": final_risk.value, "rationale": rationale.value,
        },
        source_trace={
            "scenario": "Track values marked illustrative are assumptions, not measurements.",
            "calculations": "MLSysIM v2_03_experiments finite-flow and alpha-beta models.",
        },
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit, captures, final_choice, final_rejected, final_risk, final_trigger,
    ledger, mo, rationale, track_id,
):
    _saved_decision = captures["E"].to_dict()["decision"] if "E" in captures else None
    _tested_plans = tuple(captures["E"].to_dict()["inputs"].get(key) for key in ("first_topology_id", "second_topology_id")) if "E" in captures else ()
    _ready = audit.complete and all(
        widget.value is not None
        for widget in (final_choice, final_rejected, final_trigger, final_risk)
    ) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_decision and final_rejected.value in _tested_plans
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(chapter=3, design={
                "schema_version": 1, "lab_id": "v2_03", "track_id": track_id,
                "model_id": "v2_03_experiments",
                "evidence": {part: capture.to_dict() for part, capture in captures.items()},
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
    mo.Html(f'<div class="lab-hud" style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:#10233d;color:#fff;padding:14px 18px;border-radius:9px;font-family:ui-monospace,monospace"><span>LAB 03 · Network Fabrics · STATUS: {_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
