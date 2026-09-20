import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="full", app_title="Lab 15: Sustainable Fleet Decisions · MLSysBook"
)


@app.cell
async def _():
    import sys
    from pathlib import Path
    import marimo as mo

    if sys.platform == "emscripten":
        import micropip

        await micropip.install(
            ["pydantic", "pint", "plotly", "pandas"], keep_going=False
        )
        await micropip.install(
            "../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False
        )
        await micropip.install(
            "../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False
        )
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap

        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.engine.v2_15_experiments import (
        calculate_energy,
        calculate_lifecycle,
        calculate_rebound,
        check_facility,
        evaluate_mitigation,
        evaluate_placements,
        facility_experiment_args,
        lifecycle_experiment_args,
        lower_lifecycle_option,
        mitigation_experiment_args,
        placement_experiment_args,
        rebound_experiment_args,
        serialize_evaluator_args,
        serialize_experiment_value,
        track_profile,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        get_lab_metadata,
        report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import capture_evidence, audit_evidence

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
        calculate_energy,
        calculate_lifecycle,
        calculate_rebound,
        capture_evidence,
        check_facility,
        evaluate_mitigation,
        evaluate_placements,
        facility_experiment_args,
        get_lab_metadata,
        go,
        ledger,
        lifecycle_experiment_args,
        lower_lifecycle_option,
        mitigation_experiment_args,
        mo,
        placement_experiment_args,
        rebound_experiment_args,
        report_export_panel,
        serialize_evaluator_args,
        serialize_experiment_value,
        track_profile,
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
        label="Deployment track",
        on_change=lambda _value: set_evidence({}),
    )
    track
    return (track,)


@app.cell
def _(track, track_profile):
    track_id = track.value
    profile = track_profile(track_id)
    return profile, track_id


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    is_movable = profile["inference_movable"]
    a_load_scale = mo.ui.slider(
        0.7, 1.35, value=1.2, step=0.05, label="Fleet load scale"
    )
    b_horizon = mo.ui.slider(
        0.5, 5.0, value=2.0, step=0.5, label="Operating horizon (years)"
    )
    c_deadline_scale = mo.ui.slider(
        0.25, 2.0, value=1.0, step=0.25, label="Deadline scale"
    )
    c_capacity_scale = mo.ui.slider(
        0.0,
        1.25,
        value=1.0,
        step=0.25,
        label="Clean-site capacity scale"
        if is_movable
        else "Clean-site capacity scale (inapplicable — fixed local workload)",
        disabled=not is_movable,
    )
    c_choice = mo.ui.radio(
        {
            f"Home site ({profile['home_site']})": "home",
            "Clean region"
            if is_movable
            else "Clean region (inapplicable — fixed local workload)": "clean",
            "No feasible site": "none",
        },
        label="Placement decision",
    )
    d_choice = mo.ui.radio(
        {
            "Reduce operations": "compute",
            "Reduce movement": "movement",
            "Balanced change": "balanced",
            "Keep the tested baseline": "none",
        },
        label="Mitigation decision",
    )
    d_rejected = mo.ui.radio(
        {
            "Reduce operations": "compute",
            "Reduce movement": "movement",
            "Balanced change": "balanced",
        },
        label="Rejected alternative",
    )
    e_demand = mo.ui.slider(
        1.1, 3.0, value=1.5, step=0.1, label="Demand after efficiency (× baseline)"
    )
    return (
        a_load_scale,
        b_horizon,
        c_capacity_scale,
        c_choice,
        c_deadline_scale,
        d_choice,
        d_rejected,
        e_demand,
    )


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {
            "Electrical limit": "electrical",
            "Cooling limit": "cooling",
            "Both limits": "both",
            "Neither limit": "neither",
        },
        label="Which physical boundary will the changed load cross?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Keep current hardware": "keep", "Replace hardware": "replace"},
        label="Which option has lower lifecycle emissions at this horizon?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {
            f"Home site ({profile['home_site']}) is lowest feasible": "home",
            "Clean site is lowest feasible": "clean",
            "No site is feasible": "none",
        },
        label="Which placement will survive every constraint?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {
            "Operations": "compute",
            "Movement": "movement",
            "Balanced": "balanced",
            "None": "none",
        },
        label="Which mitigation saves the most energy while preserving service?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Total emissions fall": "fall", "Total emissions rise": "rise"},
        label="After demand responds, what happens to total emissions?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {
            "Keep current hardware": "keep",
            "Replace hardware": "replace",
            "Reduce operations": "compute",
            "Reduce movement": "movement",
            "Balanced mitigation": "balanced",
            "Hold the current design": "none",
        },
        label="Final recommendation",
    )
    final_rejected = mo.ui.radio(
        {
            "Keep current hardware": "keep",
            "Replace hardware": "replace",
            "Reduce operations": "compute",
            "Reduce movement": "movement",
            "Balanced mitigation": "balanced",
        },
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {
            "Demand reaches rebound boundary": "demand",
            "Power or cooling headroom disappears": "facility",
            "Service deadline or quality fails": "service",
            "Embodied break-even changes": "lifecycle",
        },
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {
            "Grid intensity varies over time": "grid",
            "Demand response is uncertain": "demand",
            "Manufacturing estimate is incomplete": "embodied",
            "Water data is unavailable": "water",
        },
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Evidence-based rationale",
        placeholder="Name your chosen option, a result quantity, the rejected alternative and its quantity, the remaining limitation, and the trigger.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_load_scale,
    b_horizon,
    c_capacity_scale,
    c_deadline_scale,
    calculate_lifecycle,
    calculate_rebound,
    check_facility,
    d_choice,
    e_demand,
    evaluate_mitigation,
    evaluate_placements,
    facility_experiment_args,
    lifecycle_experiment_args,
    lower_lifecycle_option,
    mitigation_experiment_args,
    placement_experiment_args,
    rebound_experiment_args,
    track_id,
):
    a_base_args, a_result_args = facility_experiment_args(
        track_id, load_scale=a_load_scale.value
    )
    a_base = check_facility(**a_base_args)
    a_result = check_facility(**a_result_args)
    b_base_args, b_result_args = lifecycle_experiment_args(
        track_id, horizon_years=b_horizon.value
    )
    b_base = calculate_lifecycle(**b_base_args)
    b_result = calculate_lifecycle(**b_result_args)
    b_decision = lower_lifecycle_option(b_base, b_result)
    c_base_args = placement_experiment_args(
        track_id, deadline_scale=1.0, clean_capacity_scale=1.0
    )
    c_result_args = placement_experiment_args(
        track_id,
        deadline_scale=c_deadline_scale.value,
        clean_capacity_scale=c_capacity_scale.value,
    )
    c_base = evaluate_placements(**c_base_args)
    c_result = evaluate_placements(**c_result_args)
    d_results = {}
    for _action in ("compute", "movement", "balanced"):
        _base_args, _result_args = mitigation_experiment_args(
            track_id, intervention=_action
        )
        d_results[_action] = {
            "base_args": _base_args,
            "result_args": _result_args,
            "base": evaluate_mitigation(**_base_args),
            "result": evaluate_mitigation(**_result_args),
        }
    d_selected = d_choice.value if d_choice.value in d_results else "compute"
    e_base_args, e_result_args = rebound_experiment_args(
        track_id, demand_multiplier=e_demand.value
    )
    e_base = calculate_rebound(**e_base_args)
    e_result = calculate_rebound(**e_result_args)
    e_curve = []
    for _multiplier in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0):
        _curve_base_args, _curve_args = rebound_experiment_args(
            track_id, demand_multiplier=_multiplier
        )
        e_curve.append(calculate_rebound(**_curve_args))
    return (
        a_base,
        a_base_args,
        a_result,
        a_result_args,
        b_base,
        b_base_args,
        b_decision,
        b_result,
        b_result_args,
        c_base,
        c_base_args,
        c_result,
        c_result_args,
        d_results,
        d_selected,
        e_base,
        e_base_args,
        e_curve,
        e_result,
        e_result_args,
    )


@app.cell
def _(
    a_base,
    a_base_args,
    a_load_scale,
    a_prediction,
    a_result,
    a_result_args,
    b_base,
    b_base_args,
    b_decision,
    b_horizon,
    b_prediction,
    b_result,
    b_result_args,
    c_base,
    c_base_args,
    c_capacity_scale,
    c_choice,
    c_deadline_scale,
    c_prediction,
    c_result,
    c_result_args,
    capture_evidence,
    d_choice,
    d_prediction,
    d_rejected,
    d_results,
    d_selected,
    e_base,
    e_base_args,
    e_demand,
    e_prediction,
    e_result,
    e_result_args,
    mo,
    serialize_evaluator_args,
    serialize_experiment_value,
    set_evidence,
    track_id,
):
    def pack(arguments, output):
        return {
            "inputs": serialize_evaluator_args(arguments),
            "outputs": serialize_experiment_value(output),
        }

    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    upstream = {"scenario_version": 1}
    a_capture = mo.ui.button(
        label="Capture physical-limit contrast",
        kind="success",
        disabled=a_prediction.value is None or a_load_scale.value == 1.0,
        on_click=lambda _v: store(
            "A",
            capture_evidence(
                track=track_id,
                part="A",
                prediction=a_prediction.value,
                inputs={"load_scale": a_load_scale.value},
                baseline=pack(a_base_args, a_base),
                result=pack(a_result_args, a_result),
                upstream_inputs=upstream,
                model_key="v2_15_experiments.check_facility",
            ),
        ),
    )
    b_capture = mo.ui.button(
        label="Capture lifecycle comparison",
        kind="success",
        disabled=b_prediction.value is None,
        on_click=lambda _v: store(
            "B",
            capture_evidence(
                track=track_id,
                part="B",
                prediction=b_prediction.value,
                inputs={"horizon_years": b_horizon.value},
                baseline=pack(b_base_args, b_base),
                result=pack(b_result_args, b_result),
                upstream_inputs=upstream,
                decision=b_decision,
                model_key="v2_15_experiments.calculate_lifecycle",
            ),
        ),
    )
    c_capture = mo.ui.button(
        label="Capture placement decision",
        kind="success",
        disabled=c_prediction.value is None
        or c_choice.value is None
        or (c_deadline_scale.value == 1.0 and c_capacity_scale.value == 1.0),
        on_click=lambda _v: store(
            "C",
            capture_evidence(
                track=track_id,
                part="C",
                prediction=c_prediction.value,
                inputs={
                    "deadline_scale": c_deadline_scale.value,
                    "clean_capacity_scale": c_capacity_scale.value,
                    "choice": c_choice.value,
                },
                baseline=pack(c_base_args, c_base),
                result=pack(c_result_args, c_result),
                upstream_inputs=upstream,
                alternatives=serialize_experiment_value(c_result),
                decision=c_choice.value,
                model_key="v2_15_experiments.evaluate_placements",
            ),
        ),
    )
    d_invalid = (
        d_prediction.value is None
        or d_choice.value is None
        or d_rejected.value is None
        or d_choice.value == d_rejected.value
    )
    _d_tested = (
        d_rejected.value
        if d_choice.value == "none" and d_rejected.value
        else d_selected
    )
    d_capture = mo.ui.button(
        label="Capture mitigation decision",
        kind="success",
        disabled=d_invalid,
        on_click=lambda _v: store(
            "D",
            capture_evidence(
                track=track_id,
                part="D",
                prediction=d_prediction.value,
                inputs={"choice": d_choice.value, "rejected": d_rejected.value},
                baseline=pack(
                    d_results[_d_tested]["base_args"], d_results[_d_tested]["base"]
                ),
                result=pack(
                    d_results[_d_tested]["result_args"], d_results[_d_tested]["result"]
                ),
                chosen_result=pack(
                    d_results[_d_tested]["base_args"], d_results[_d_tested]["base"]
                )
                if d_choice.value == "none"
                else None,
                result_role="rejected alternative"
                if d_choice.value == "none"
                else "tested intervention",
                upstream_inputs=upstream,
                alternatives=tuple(
                    {
                        "name": _name,
                        "inputs": serialize_evaluator_args(_comparison["result_args"]),
                        "outputs": serialize_experiment_value(_comparison["result"]),
                    }
                    for _name, _comparison in d_results.items()
                ),
                decision=d_choice.value,
                model_key="v2_15_experiments.evaluate_mitigation",
            ),
        ),
    )
    e_capture = mo.ui.button(
        label="Capture rebound contrast",
        kind="success",
        disabled=e_prediction.value is None or e_demand.value == 1.0,
        on_click=lambda _v: store(
            "E",
            capture_evidence(
                track=track_id,
                part="E",
                prediction=e_prediction.value,
                inputs={"demand_multiplier": e_demand.value},
                baseline=pack(e_base_args, e_base),
                result=pack(e_result_args, e_result),
                upstream_inputs=upstream,
                decision="rise" if e_result.rebound_increases_emissions else "fall",
                model_key="v2_15_experiments.calculate_rebound",
            ),
        ),
    )
    return a_capture, b_capture, c_capture, d_capture, e_capture, upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile):
    css = mo.Html("""
    <style>
    .sustain-head{background:linear-gradient(135deg,#12372a,#176b4d);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}.sustain-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}.sustain-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.sustain-head p{color:#dcfce7;max-width:780px}.sustain-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.sustain-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}.saved{border-left:4px solid #16825d;background:#ecfdf5;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#12372a!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#bbf7d0}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}@media(max-width:520px){.sustain-head{border-radius:9px;margin-top:30px}.sustain-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(
        f"""<section class="sustain-head"><div class="sustain-top"><span>VOLUME II · LAB 15</span><span>ABOUT 50–55 MIN</span></div><h1>Sustainable Fleet Decisions</h1><p>Which fleet design reduces lifecycle emissions while still fitting its power, cooling, locality, quality, and latency constraints?</p><div class="sustain-meta"><div><b>Track</b><br>{profile["display"]}</div><div><b>Fleet context</b><br>{profile["fleet_shape"]}</div><div><b>Deliverable</b><br>Five saved contrasts and one defensible lifecycle recommendation</div></div></section>"""
    )
    mo.vstack(
        [
            ACADEMIC_LAB_CSS,
            LAB_CSS,
            css,
            header,
            mo.md(
                "All profiles are explicit teaching scenarios, not measurements of branded systems. Each part compares the same useful work before and after one change. Carbon accounting follows physical feasibility; it cannot make an overloaded site operable."
            ),
        ]
    )
    return


@app.cell
def _(
    a_base,
    a_capture,
    a_load_scale,
    a_prediction,
    a_result,
    apply_plotly_theme,
    audit_evidence,
    b_base,
    b_capture,
    b_decision,
    b_horizon,
    b_prediction,
    b_result,
    c_capture,
    c_capacity_scale,
    c_choice,
    c_deadline_scale,
    c_prediction,
    c_result,
    d_capture,
    d_choice,
    d_prediction,
    d_rejected,
    d_results,
    e_capture,
    e_curve,
    e_demand,
    e_prediction,
    e_result,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    go,
    mo,
    profile,
    rationale,
    track_id,
    upstream,
):
    def q(value, unit, digits=2):
        return f"{value.to(unit).magnitude:,.{digits}f} {unit}"

    def table(rows):
        if not rows:
            return mo.md("No rows.")
        headers = tuple(rows[0])
        body = "".join(
            "<tr>" + "".join(f"<td>{row[h]}</td>" for h in headers) + "</tr>"
            for row in rows
        )
        head = "".join(f"<th>{h}</th>" for h in headers)
        return mo.Html(
            f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
        )

    def saved(part):
        capture = get_evidence().get(part)
        return (
            mo.Html(
                f'<div class="saved">Saved {part}: prediction and observed values frozen.</div>'
            )
            if capture
            else mo.md("")
        )

    def pass_word(ok):
        return "PASS" if ok else "FAIL"

    def part_a():
        if a_prediction.value is None:
            return mo.vstack(
                [
                    mo.md(
                        "### A · What limits the fleet before carbon does? (9 min)\nChange the common fleet load, then commit a prediction about the first physical boundary."
                    ),
                    a_load_scale,
                    a_prediction,
                ]
            )
        fig = go.Figure()
        fig.add_bar(
            name="Baseline",
            x=["IT load", "Facility draw"],
            y=[
                a_base.it_power.to("kilowatt").magnitude,
                a_base.facility_power.to("kilowatt").magnitude,
            ],
        )
        fig.add_bar(
            name="Changed",
            x=["IT load", "Facility draw"],
            y=[
                a_result.it_power.to("kilowatt").magnitude,
                a_result.facility_power.to("kilowatt").magnitude,
            ],
        )
        fig.update_layout(
            barmode="group", yaxis_title="Power (kW)", legend_title="Condition"
        )
        rows = [
            {
                "Condition": "Baseline",
                "Electrical": pass_word(a_base.electrical_feasible),
                "Cooling": pass_word(a_base.cooling_feasible),
                "Electrical headroom": q(a_base.electrical_headroom, "kilowatt"),
                "Cooling headroom": q(a_base.cooling_headroom, "kilowatt"),
            },
            {
                "Condition": "Changed",
                "Electrical": pass_word(a_result.electrical_feasible),
                "Cooling": pass_word(a_result.cooling_feasible),
                "Electrical headroom": q(a_result.electrical_headroom, "kilowatt"),
                "Cooling headroom": q(a_result.cooling_headroom, "kilowatt"),
            },
        ]
        return mo.vstack(
            [
                mo.md("### A · What limits the fleet before carbon does? (9 min)"),
                a_prediction,
                a_load_scale,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {a_prediction.value}. At **{a_load_scale.value:.2f}×** load, electrical feasibility is **{pass_word(a_result.electrical_feasible)}** and cooling feasibility is **{pass_word(a_result.cooling_feasible)}**. Carbon intensity never enters either test."
                    ),
                    kind="success" if a_result.feasible else "danger",
                ),
                a_capture,
                saved("A"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "MLSysIM computes facility power as IT power × PUE. IT power becomes heat load. Electrical capacity and heat-removal capacity remain separate hard limits."
                        )
                    }
                ),
            ]
        )

    def part_b():
        if b_prediction.value is None:
            return mo.vstack(
                [
                    mo.md(
                        "### B · When should hardware be replaced? (10 min)\nChoose a common horizon and predict whether operational savings repay the additional manufacturing emissions."
                    ),
                    b_horizon,
                    b_prediction,
                ]
            )
        fig = go.Figure(
            go.Bar(
                x=["Keep", "Replace"],
                y=[
                    b_base.total_emissions.to("kilogram").magnitude,
                    b_result.total_emissions.to("kilogram").magnitude,
                ],
                marker_color=["#64748b", "#16825d"],
            )
        )
        fig.update_layout(yaxis_title="Lifecycle emissions (kg CO₂e)", showlegend=False)
        rows = [
            {
                "Option": "Keep",
                "Training": q(b_base.training_facility_energy, "kilowatt_hour"),
                "Inference": q(b_base.inference_facility_energy, "kilowatt_hour"),
                "Operational carbon": q(
                    b_base.operational_emissions, "kilogram"
                ),
                "New manufacture": q(
                    b_base.additional_manufacturing_emissions, "kilogram"
                ),
                "Total": q(b_base.total_emissions, "kilogram"),
            },
            {
                "Option": "Replace",
                "Training": q(b_result.training_facility_energy, "kilowatt_hour"),
                "Inference": q(b_result.inference_facility_energy, "kilowatt_hour"),
                "Operational carbon": q(
                    b_result.operational_emissions, "kilogram"
                ),
                "New manufacture": q(
                    b_result.additional_manufacturing_emissions, "kilogram"
                ),
                "Total": q(b_result.total_emissions, "kilogram"),
            },
        ]
        return mo.vstack(
            [
                mo.md("### B · When should hardware be replaced? (10 min)"),
                b_prediction,
                b_horizon,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {b_prediction.value}. Over **{b_horizon.value:.1f} years**, the lower-emission choice is **{b_decision}**. Existing hardware's prior embodied emissions are sunk; only new manufacturing caused by replacement is added."
                    ),
                    kind="info",
                ),
                b_capture,
                saved("B"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Training, inference, and embodied emissions share one horizon. Inference energy equals request rate × horizon × energy per request. Facility energy applies PUE once; grid intensity converts energy to operational carbon."
                        )
                    }
                ),
            ]
        )

    def part_c():
        if c_prediction.value is None:
            return mo.vstack(
                [
                    mo.md(
                        "### C · Is the cleanest place usable? (10 min)\nChange the deadline or available clean-site capacity, then predict which site remains feasible."
                    ),
                    mo.hstack(
                        [c_deadline_scale, c_capacity_scale], widths="equal", wrap=True
                    ),
                    c_prediction,
                ]
            )
        rows = [
            {
                "Site": result.site,
                "Feasible": pass_word(result.feasible),
                "Emissions": q(result.operational_emissions, "kilogram"),
                "Water": q(result.water_use, "liter")
                if result.water_use is not None
                else "Not modeled",
                "Constraint": "; ".join(result.reasons) if result.reasons else "none",
            }
            for result in c_result
        ]
        fig = go.Figure(
            go.Bar(
                x=[result.site for result in c_result],
                y=[
                    result.operational_emissions.to("kilogram").magnitude
                    for result in c_result
                ],
                marker_color=["#64748b", "#16825d"],
            )
        )
        fig.update_layout(
            yaxis_title="Operational emissions (kg CO₂e)", showlegend=False
        )
        return mo.vstack(
            [
                mo.md("### C · Is the cleanest place usable? (10 min)"),
                c_prediction,
                mo.hstack(
                    [c_deadline_scale, c_capacity_scale], widths="equal", wrap=True
                ),
                apply_plotly_theme(fig),
                table(rows),
                c_choice,
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {c_prediction.value}. A low-carbon site is usable only if capacity, deadline, locality, electrical, and cooling checks all pass. "
                        + (
                            "Regional batch workloads can migrate to clean regions when network deadlines and accelerator capacity permit."
                            if profile["inference_movable"]
                            else f"{profile['display']} inference is bound to {profile['home_site']}; remote placement is inapplicable due to locality and immobility."
                        )
                    ),
                    kind="info",
                ),
                c_capture,
                saved("C"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "MLSysIM evaluates the same job at each site. Optional WUE converts facility kWh to liters only where a scenario supplies grounded WUE; missing water data remains missing."
                        )
                    }
                ),
            ]
        )

    def part_d():
        if d_prediction.value is None:
            return mo.vstack(
                [
                    mo.md(
                        "### D · Which mitigation survives its side effects? (10 min)\nPredict which change reduces facility energy while preserving the supplied quality floor and latency deadline."
                    ),
                    d_prediction,
                ]
            )
        rows = []
        fig = go.Figure()
        for name, comparison in d_results.items():
            result = comparison["result"]
            rows.append(
                {
                    "Action": name.title(),
                    "Facility energy": q(
                        result.energy.facility_energy, "kilowatt_hour", 4
                    ),
                    "Latency": q(result.latency, "millisecond"),
                    "Quality": f"{result.quality:.3f}",
                    "Service": pass_word(result.acceptable),
                }
            )
            fig.add_bar(
                name=name.title(),
                x=["Operations", "Movement", "Other IT", "Facility overhead"],
                y=[
                    result.energy.operation_energy.to("kilowatt_hour").magnitude,
                    result.energy.movement_energy.to("kilowatt_hour").magnitude,
                    result.energy.other_it_energy.to("kilowatt_hour").magnitude,
                    result.energy.facility_overhead_energy.to(
                        "kilowatt_hour"
                    ).magnitude,
                ],
            )
        fig.update_layout(
            barmode="group", yaxis_title="Energy for fixed useful work (kWh)"
        )
        chosen = (
            d_results[d_choice.value]["result"] if d_choice.value in d_results else None
        )
        return mo.vstack(
            [
                mo.md("### D · Which mitigation survives its side effects? (10 min)"),
                d_prediction,
                apply_plotly_theme(fig),
                table(rows),
                mo.hstack([d_choice, d_rejected], widths="equal", wrap=True),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {d_prediction.value}. Choose one tested action and a different rejected alternative. {'The selected action passes both service constraints.' if chosen and chosen.acceptable else 'A no-action conclusion still needs one quantified rejected alternative.'}"
                    ),
                    kind="info",
                ),
                d_capture,
                saved("D"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Operations and moved bytes are explicit energy components. The illustrative quality observations and latency values act as constraints; quality never changes the energy equation."
                        )
                    }
                ),
            ]
        )

    def part_e():
        if e_prediction.value is None:
            return mo.vstack(
                [
                    mo.md(
                        "### E · Can efficiency increase total emissions? (8 min)\nSet explicit post-efficiency demand, then predict which side of the rebound boundary it occupies."
                    ),
                    e_demand,
                    e_prediction,
                ]
            )
        fig = go.Figure()
        fig.add_scatter(
            x=[result.demand_multiplier for result in e_curve],
            y=[
                result.optimized_emissions.to("kilogram").magnitude
                for result in e_curve
            ],
            mode="lines+markers",
            name="After efficiency",
        )
        fig.add_scatter(
            x=[result.demand_multiplier for result in e_curve],
            y=[
                result.baseline_emissions.to("kilogram").magnitude for result in e_curve
            ],
            mode="lines",
            name="Original total",
        )
        fig.update_layout(
            xaxis_title="Demand multiplier", yaxis_title="Total emissions (kg CO₂e)"
        )
        outcome = "rise" if e_result.rebound_increases_emissions else "fall"
        return mo.vstack(
            [
                mo.md("### E · Can efficiency increase total emissions? (8 min)"),
                e_prediction,
                e_demand,
                apply_plotly_theme(fig),
                table(
                    [
                        {
                            "Demand": f"{e_result.demand_multiplier:.2f}×",
                            "Per-request reduction": f"{e_result.energy_reduction_per_request:.1%}",
                            "Break-even demand": f"{e_result.break_even_demand_multiplier:.2f}×",
                            "Original total": q(
                                e_result.baseline_emissions, "kilogram"
                            ),
                            "New total": q(e_result.optimized_emissions, "kilogram"),
                        }
                    ]
                ),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {e_prediction.value}. With demand at **{e_demand.value:.2f}×**, total emissions **{outcome}**. The crossover comes from the explicit demand multiplier, not a policy label."
                    ),
                    kind="danger"
                    if e_result.rebound_increases_emissions
                    else "success",
                ),
                e_capture,
                saved("E"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "The baseline and optimized service share the same horizon, PUE, and grid. Total emissions rise when demand growth exceeds the reciprocal of the remaining per-request energy fraction."
                        )
                    }
                ),
            ]
        )

    _captures = get_evidence()
    audit = audit_evidence(
        _captures,
        track=track_id,
        required_parts=tuple("ABCDE"),
        per_part_upstream_inputs={part: upstream for part in "ABCDE"},
        contrast_required_parts=tuple("ABCDE"),
    )

    def synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            rows.append(
                {
                    "Part": part,
                    "Prediction": capture.to_dict()["prediction"] if capture else "—",
                    "Evidence": "CURRENT"
                    if capture
                    and part not in audit.stale
                    and (part, part) not in audit.identical_pairs
                    else ("STALE" if capture else "MISSING"),
                }
            )
        complete = (
            audit.complete
            and all(
                widget.value is not None
                for widget in (final_choice, final_rejected, final_trigger, final_risk)
            )
            and final_choice.value != final_rejected.value
            and bool(rationale.value.strip())
        )
        return mo.vstack(
            [
                mo.md(
                    "### Synthesis · Defend one lifecycle recommendation (5 min)\nChoose an option supported by the saved experiments. Quantify a rejected alternative, name one remaining limitation, and state the condition that would reverse your decision."
                ),
                table(rows),
                mo.callout(
                    mo.md(
                        "Saved predictions, inputs, baseline outputs, and intervention outputs remain fixed when live controls move. Track changes clear the evidence because the fleet itself changes."
                    ),
                    kind="info",
                ),
                mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
                mo.hstack([final_trigger, final_risk], widths="equal", wrap=True),
                rationale,
                mo.callout(
                    mo.md(
                        "**Ready for the local report and Design Ledger.**"
                        if complete
                        else "Capture five genuine contrasts, choose distinct final and rejected options, and complete the rationale."
                    ),
                    kind="success" if complete else "warn",
                ),
            ]
        )

    mo.ui.tabs(
        {
            "Part A": part_a(),
            "Part B": part_b(),
            "Part C": part_c(),
            "Part D": part_d(),
            "Part E": part_e(),
            "Synthesis": synthesis(),
        }
    )
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
        and all(
            widget.value is not None
            for widget in (final_choice, final_rejected, final_trigger, final_risk)
        )
        and final_choice.value != final_rejected.value
        and bool(rationale.value.strip())
    )
    mo.stop(not _ready)
    _snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    _design = {
        "schema_version": 1,
        "lab_id": "v2_15",
        "track_id": track_id,
        "model_id": "v2_15_experiments",
        "evidence": _snapshots,
        "recommendation": final_choice.value,
        "rejected_alternative": final_rejected.value,
        "reevaluation_trigger": final_trigger.value,
        "residual_risk": final_risk.value,
        "rationale": rationale.value,
    }
    report = build_lab_report(
        get_lab_metadata("vol2/lab_15_sustainable_ai.py"),
        track=track_id,
        scenario=profile["fleet_shape"],
        learning_objectives=[
            "Separate power and cooling feasibility from carbon accounting",
            "Compare operational and additional manufacturing emissions over one horizon",
            "Defend a service-qualified mitigation under explicit demand rebound",
        ],
        predictions={part: _snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: _snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={
            part: {
                "baseline": _snapshots[part]["baseline"],
                "result": _snapshots[part]["result"],
                "alternatives": _snapshots[part]["alternatives"],
            }
            for part in "ABCDE"
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
            "Power and cooling are hard constraints before carbon is counted.",
            "Replacement must repay only the additional manufacturing it causes.",
            "Efficiency lowers total emissions only below an explicit rebound boundary.",
        ],
        reflections={
            "rationale": rationale.value,
            "reevaluation_trigger": final_trigger.value,
        },
        residual_risk=final_risk.value,
        result_snapshot=_design,
        source_trace={
            "scenario": "Illustrative fleet assumptions from MLSysIM; not a hardware benchmark.",
            "calculations": "MLSysIM v2_15_experiments quantity-first evaluators.",
        },
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
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
    _captures = get_evidence()
    _ready = (
        audit.complete
        and all(
            widget.value is not None
            for widget in (final_choice, final_rejected, final_trigger, final_risk)
        )
        and final_choice.value != final_rejected.value
        and bool(rationale.value.strip())
    )
    _save_error = None
    _saved = False
    if _ready:
        try:
            ledger.save(
                chapter=15,
                design={
                    "schema_version": 1,
                    "lab_id": "v2_15",
                    "track_id": track_id,
                    "model_id": "v2_15_experiments",
                    "evidence": {
                        part: capture.to_dict() for part, capture in _captures.items()
                    },
                    "recommendation": final_choice.value,
                    "rejected_alternative": final_rejected.value,
                    "reevaluation_trigger": final_trigger.value,
                    "residual_risk": final_risk.value,
                    "rationale": rationale.value,
                },
            )
            await ledger.flush()
            _saved = True
        except Exception as _exc:
            _save_error = f"{type(_exc).__name__}: {_exc}"
    _status = (
        "SAVED"
        if _saved
        else ("SAVE FAILED" if _save_error else "EVIDENCE IN PROGRESS")
    )
    _hud = mo.Html(
        f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">15 · Sustainable Fleet Decisions</span><span aria-hidden="true">|</span><span style="flex:1"></span><span class="hud-label">STATUS</span><span class="hud-active">{_status}</span></div>'
    )
    if _save_error:
        _footer = mo.vstack(
            [
                _hud,
                mo.callout(
                    mo.md(
                        f"**Design Ledger save failed:** `{_save_error}`. The local report remains available above."
                    ),
                    kind="danger",
                ),
            ]
        )
    else:
        _footer = _hud
    _footer
    return


if __name__ == "__main__":
    app.run()
