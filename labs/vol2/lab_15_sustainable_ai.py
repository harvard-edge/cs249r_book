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
    from mlsysim.labs.experiments.v2_15_experiments import (
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
def _(ledger, mo, set_evidence):
    from mlsysbook_labs.contexts import preferred_context as _preferred_context
    _track_options = {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"}
    track = mo.ui.dropdown(
        _track_options,
        value=next(label for label, context in _track_options.items() if context == _preferred_context(ledger)),
        label="Deployment context",
        on_change=lambda _value: set_evidence({}),
    )
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
    c_cooling = mo.ui.dropdown(
        {"Supplied": "supplied", "Toy dry": "dry",
         "Toy evaporative": "evaporative", "Unknown WUE": "unknown"},
        value="Supplied", label="Clean-site cooling",
    )
    c_water_season = mo.ui.dropdown(
        {"Unassessed": "unassessed", "Toy wet season": "wet",
         "Toy drought": "drought"},
        value="Unassessed", label="Water season",
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
        c_cooling,
        c_water_season,
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
     debounce=False)
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_load_scale,
    b_horizon,
    c_capacity_scale,
    c_deadline_scale,
    c_cooling,
    c_water_season,
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
        clean_cooling=c_cooling.value, water_season=c_water_season.value,
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
    c_cooling,
    c_water_season,
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
    d_upstream = {**upstream, "choice": d_choice.value, "rejected": d_rejected.value}
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
        or (c_deadline_scale.value == 1.0 and c_capacity_scale.value == 1.0
            and c_cooling.value == "supplied" and c_water_season.value == "unassessed"),
        on_click=lambda _v: store(
            "C",
            capture_evidence(
                track=track_id,
                part="C",
                prediction=c_prediction.value,
                inputs={
                    "deadline_scale": c_deadline_scale.value,
                    "clean_capacity_scale": c_capacity_scale.value,
                    "clean_cooling": c_cooling.value, "water_season": c_water_season.value,
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
                upstream_inputs=d_upstream,
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
    return a_capture, b_capture, c_capture, d_capture, e_capture, upstream, d_upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    from mlsysbook_labs.notebook import (
        notebook_header as _course_header, notebook_shell as _course_shell,
        notebook_learning_map as _course_learning_map,
        notebook_scenario as _course_scenario,
    )
    from mlsysbook_labs.learning_paths import get_learning_path as _course_path
    from mlsysbook_labs import get_lab_metadata as _course_metadata
    css = mo.Html("""
    <style>
    .saved{border-left:4px solid #16825d;background:#ecfdf5;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}.table-wrap table{border-collapse:collapse;min-width:100%;font-size:.9rem}.table-wrap th,.table-wrap td{padding:.45rem .6rem;text-align:left;vertical-align:top;min-width:7rem}.table-wrap th{border-bottom:1px solid #cbd5e1}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#12372a!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#bbf7d0}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    </style>""")
    header = _course_header(_course_metadata("vol2/lab_15_sustainable_ai.py"), 'Which fleet design reduces lifecycle emissions while still fitting its power, cooling, locality, quality, and latency constraints?')
    _course_shell(
        [
        ACADEMIC_LAB_CSS,
        LAB_CSS,
        css,
        header,
        track,
        _course_scenario(_course_path('vol2', 15), track=track.value, volume=2),
        mo.md(
                "All profiles are explicit teaching scenarios, not measurements of branded systems. Each part compares the same useful work before and after one change. Carbon accounting follows physical feasibility; it cannot make an overloaded site operable."
            ),
        _course_learning_map(_course_path('vol2', 15))
    ]
    , volume=2)
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
    c_cooling,
    c_water_season,
    c_prediction,
    c_result,
    d_capture,
    d_upstream,
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
    _saved_prediction_widgets = {'A': a_prediction, 'B': b_prediction, 'C': c_prediction, 'D': d_prediction, 'E': e_prediction}
    from mlsysbook_labs.notebook import notebook_prediction_label as _read_prediction
    from mlsysbook_labs.notebook import notebook_tab_hint as _part_tabs_hint, notebook_shell as _lesson_shell
    from mlsysbook_labs.notebook import notebook_part_reflection as _concept_reflection
    from mlsysbook_labs.learning_paths import get_learning_path as _reflection_path
    _reflection_contract = _reflection_path('vol2', 15)
    from mlsysim.fmt import fmt_energy as _fmt_energy
    def q(value, unit, digits=2):
        return f"{value.to(unit).magnitude:,.{digits}f} {unit}"

    def table(rows):
        if not rows:
            return mo.md("No rows.")
        headers = tuple(rows[0])
        cell_style = "padding:.5rem .65rem;text-align:left;vertical-align:top;min-width:7rem"
        body = "".join(
            "<tr>" + "".join(f'<td style="{cell_style}">{row[h]}</td>' for h in headers) + "</tr>"
            for row in rows
        )
        head = "".join(f'<th style="{cell_style};border-bottom:1px solid #cbd5e1">{h}</th>' for h in headers)
        return mo.Html(
            f'<div class="table-wrap" style="max-width:100%;overflow-x:auto"><table style="border-collapse:collapse;min-width:100%;font-size:.9rem"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
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
        return mo.vstack([mo.vstack(
            [
                mo.md("### A · What limits the fleet before carbon does? (9 min)"),
                a_prediction,
                a_load_scale,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {_read_prediction(a_prediction)}. At **{a_load_scale.value:.2f}×** load, electrical feasibility is **{pass_word(a_result.electrical_feasible)}** and cooling feasibility is **{pass_word(a_result.cooling_feasible)}**. Carbon intensity never enters either test."
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
        ), _concept_reflection(_reflection_contract, "A")])

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
        return mo.vstack([mo.vstack(
            [
                mo.md("### B · When should hardware be replaced? (10 min)"),
                b_prediction,
                b_horizon,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {_read_prediction(b_prediction)}. Over **{b_horizon.value:.1f} years**, the lower-emission choice is **{b_decision}**. Existing hardware's prior embodied emissions are sunk; only new manufacturing caused by replacement is added."
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
        ), _concept_reflection(_reflection_contract, "B")])

    def part_c():
        if c_prediction.value is None:
            return mo.vstack(
                [
                    mo.md(
                        "### C · Is the cleanest place usable? (10 min)\nChange deadline, capacity, cooling, or the local water obligation. Predict whether the low-carbon site remains admissible. Dry and evaporative cooling are hypothetical alternatives for the same clean site, not measurements of an actual region."
                    ),
                    mo.hstack(
                        [c_deadline_scale, c_capacity_scale], widths="equal", wrap=True
                    ),
                    mo.hstack([c_cooling, c_water_season], widths="equal", wrap=True),
                    c_prediction,
                ]
            )
        rows = [
            {
                "Site": result.site,
                "Admissible": pass_word(result.feasible),
                "Facility electricity": q(result.facility_energy, "kilowatt_hour"),
                "Emissions": q(result.operational_emissions, "kilogram"),
                "Direct water consumed": q(result.water_use, "liter")
                if result.water_use is not None
                else "Unknown",
                "Local run allowance": q(result.water_allowance, "liter") if result.water_allowance is not None else "Not assessed",
                "Water obligation": (pass_word(result.water_requirement_met) if result.water_requirement_met is not None else ("UNKNOWN — evidence required" if result.water_allowance is not None else "Not assessed")),
                "Watershed / season": result.water_context,
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
        return mo.vstack([mo.vstack(
            [
                mo.md("### C · Is the cleanest place usable? (10 min)"),
                c_prediction,
                mo.hstack(
                    [c_deadline_scale, c_capacity_scale], widths="equal", wrap=True
                ),
                mo.hstack([c_cooling, c_water_season], widths="equal", wrap=True),
                mo.md("**Try this sequence:** select evaporative cooling and the wet season; then change only the season to drought. Finally keep the drought allocation and choose dry cooling. Compare electricity, carbon, direct water consumption, and the local allowance separately. Missing WUE is unknown water use, never zero. The toy run allocation is a stated constraint, not a validated watershed impact score."),
                apply_plotly_theme(fig),
                mo.md("**Compare the separate accounts below.** On a narrow screen, swipe the table to inspect the water obligation and watershed context."),
                table(rows),
                c_choice,
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {_read_prediction(c_prediction)}. A low-carbon site is admissible only if capacity, deadline, locality, electrical, cooling, and any required local water allowance pass. Unknown water evidence cannot establish a required water obligation. "
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
                            "The same fixed IT work runs at each site. Facility electricity = IT kWh × PUE; carbon = facility kWh × grid factor; direct water consumption = IT kWh × WUE. The clean-site dry/evaporative values and wet/drought allocations are explicit teaching assumptions, not actual region measurements. Seasonal allocation changes the local obligation without changing liters consumed. Water withdrawal, indirect electricity/manufacturing water, and validated scarcity characterization are outside this direct-consumption account. No combined environmental score is calculated."
                        )
                    }
                ),
            ]
        ), _concept_reflection(_reflection_contract, "C")])

    def part_d():
        prompt = mo.md("### D · Which mitigation survives its side effects? (10 min)\nPredict which change reduces facility energy while preserving the supplied quality floor and latency deadline.")
        if d_prediction.value is None:
            return mo.vstack([prompt, d_prediction])
        baseline = d_results["compute"]["base"]
        limits = d_results["compute"]["base_args"]
        rows = [{
            "Role": "Held baseline" if d_choice.value == "none" else "Baseline",
            "Action": "No mitigation", "Facility energy": str(_fmt_energy(baseline.energy.facility_energy, precision=4)),
            "Change vs baseline": str(_fmt_energy(baseline.energy.facility_energy - baseline.energy.facility_energy)),
            "Latency": q(baseline.latency, "millisecond"), "Quality": f"{baseline.quality:.3f}",
            "Latency gate": pass_word(baseline.meets_latency), "Quality gate": pass_word(baseline.meets_quality),
        }]
        fig = apply_plotly_theme(go.Figure())
        for name, comparison in d_results.items():
            result = comparison["result"]
            role = "Selected" if name == d_choice.value else "Rejected" if name == d_rejected.value else "Alternative"
            rows.append({
                "Role": role, "Action": name.title(),
                "Facility energy": str(_fmt_energy(result.energy.facility_energy, precision=4)),
                "Change vs baseline": str(_fmt_energy(result.energy.facility_energy - baseline.energy.facility_energy)),
                "Latency": q(result.latency, "millisecond"), "Quality": f"{result.quality:.3f}",
                "Latency gate": pass_word(result.meets_latency), "Quality gate": pass_word(result.meets_quality),
            })
            fig.add_bar(
                name=f"{role} · {name.title()}",
                x=["Operations", "Movement", "Other IT", "Facility<br>overhead"],
                y=[result.energy.operation_energy.to("joule").magnitude,
                   result.energy.movement_energy.to("joule").magnitude,
                   result.energy.other_it_energy.to("joule").magnitude,
                   result.energy.facility_overhead_energy.to("joule").magnitude],
                opacity=1.0 if role == "Selected" else 0.45,
            )
        fig.update_layout(height=360, margin=dict(l=45, r=20, t=90, b=60), barmode="group",
                          legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"),
                          yaxis_title="Energy for fixed useful work (J)")
        chosen = baseline if d_choice.value == "none" else d_results[d_choice.value]["result"] if d_choice.value in d_results else None
        def consequence(label, result):
            return (
                f"**{label}: {str(_fmt_energy(result.energy.facility_energy, precision=4))}**, "
                f"a facility-energy change of **{str(_fmt_energy(result.energy.facility_energy - baseline.energy.facility_energy))}** from the same baseline. "
                f"Latency is **{q(result.latency, 'millisecond')}** against **{q(limits['latency_deadline'], 'millisecond')}** "
                f"(**{pass_word(result.meets_latency)}**); supplied quality is **{result.quality:.3f}** against **{limits['minimum_quality']:.3f}** "
                f"(**{pass_word(result.meets_quality)}**)."
            )
        selected_note = consequence("Held baseline" if d_choice.value == "none" else "Selected " + d_choice.value, chosen) if chosen else "Choose the action to carry forward."
        rejected_note = consequence("Rejected " + d_rejected.value, d_results[d_rejected.value]["result"]) if d_rejected.value in d_results else "Choose a distinct rejected alternative."
        return mo.vstack([mo.vstack([
            prompt, d_prediction, fig, table(rows),
            mo.hstack([d_choice, d_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(f"**Your prediction:** {_read_prediction(d_prediction)}. {selected_note}\n\n{rejected_note}"), kind="success" if chosen and chosen.acceptable else "warn"),
            mo.callout(mo.md(
                f"**Keep the accounting boundary visible.** All actions retain the baseline's **{str(_fmt_energy(baseline.energy.other_it_energy))} of other IT energy**. "
                "The facility total also includes operations, movement, and PUE overhead. A common charge can mask action differences when rounded in kWh; the table retains adaptive energy units and an explicit signed change. "
                "A negative change saves energy for this fixed work. It does not override a failed quality or latency gate."
            ), kind="info"),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md("Operations and moved bytes are explicit energy components. Each displayed change subtracts the common baseline output in physical units; it introduces no additional energy model. The illustrative quality observations and latency values act as constraints; quality never changes the energy equation.")}),
        ]), _concept_reflection(_reflection_contract, "D")])

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
        return mo.vstack([mo.vstack(
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
                        f"**Your prediction:** {_read_prediction(e_prediction)}. With demand at **{e_demand.value:.2f}×**, total emissions **{outcome}**. The crossover comes from the explicit demand multiplier, not a policy label."
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
        ), _concept_reflection(_reflection_contract, "E")])

    _captures = get_evidence()
    audit = audit_evidence(
        _captures,
        track=track_id,
        required_parts=tuple("ABCDE"),
        per_part_upstream_inputs={part: (d_upstream if part == "D" else upstream) for part in "ABCDE"},
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

    _lesson_shell([_part_tabs_hint(), mo.ui.tabs(
        {
            "Part A": part_a(),
            "Part B": part_b(),
            "Part C": part_c(),
            "Part D": part_d(),
            "Part E": part_e(),
            "Synthesis": synthesis(),
        }
    )], volume=2)
    return (audit,)


@app.cell
def _():
    from mlsysbook_labs.notebook import notebook_shell as _submit_shell, notebook_submission as _submit_guide
    from mlsysbook_labs.learning_paths import get_learning_path as _submit_path
    _submit_shell([_submit_guide(_submit_path(2, 15))], volume=2)
    return


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
    from mlsysbook_labs.notebook import notebook_shell as _workspace_shell
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
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not _ready, _download_readiness(volume=2))
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
    _workspace_shell([mo.vstack([mo.md("### Download your discussion"), report_export_panel(report)])], volume=2)
    return (report,)


@app.cell
def _(
    audit, final_choice, final_rejected, final_risk, final_trigger, get_evidence, mo, rationale, track_id
):
    from mlsysbook_labs.fleet_evidence import discussion_fingerprint as _discussion_fingerprint
    _captures = get_evidence()
    _ready = audit.complete and all((widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))) and (final_choice.value != final_rejected.value) and bool(rationale.value.strip())
    fleet_discussion_ready = _ready
    fleet_discussion_design = {'schema_version': 1, 'lab_id': 'v2_15', 'track_id': track_id, 'model_id': 'v2_15_experiments', 'evidence': {part: capture.to_dict() for part, capture in _captures.items()}, 'recommendation': final_choice.value, 'rejected_alternative': final_rejected.value, 'reevaluation_trigger': final_trigger.value, 'residual_risk': final_risk.value, 'rationale': rationale.value} if _ready else None
    fleet_discussion_fingerprint = _discussion_fingerprint(fleet_discussion_design)
    # Recreated when any evidence, track, or discussion field changes.
    save_discussion = mo.ui.run_button(label="Save discussion", disabled=not fleet_discussion_ready)
    return (fleet_discussion_ready, fleet_discussion_design, fleet_discussion_fingerprint, save_discussion)


@app.cell
async def _(fleet_discussion_design, fleet_discussion_fingerprint, fleet_discussion_ready,
            ledger, mo, save_discussion, track_id):
    from mlsysbook_labs.fleet_evidence import save_completed_discussion as _save_completed_discussion
    from mlsysbook_labs.notebook import notebook_shell as _discussion_shell, notebook_status as _notebook_status
    _outcome = await _save_completed_discussion(
        ledger, chapter=15, track=track_id, design=fleet_discussion_design,
        fingerprint=fleet_discussion_fingerprint, ready=fleet_discussion_ready,
        requested=bool(save_discussion.value),
    )
    _discussion_shell([mo.vstack([save_discussion, _notebook_status(_outcome.status, _outcome.detail)])], volume=2)
    return


if __name__ == "__main__":
    app.run()
