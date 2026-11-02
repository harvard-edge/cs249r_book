import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="full", app_title="Lab 04: Evidence Through the Data Pipeline · MLSysBook"
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
            "../../wheels/mlsysim-0.3.0-py3-none-any.whl", keep_going=False
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
    from mlsysim.labs.experiments.v1_04_experiments import (
        capture_experiment,
        contract_inputs,
        freshness_inputs,
        get_track_scenario,
        pipeline_inputs,
        retention_inputs,
        split_inputs,
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

    ledger = DesignLedger(volume="vol1")
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
        capture_experiment,
        contract_inputs,
        freshness_inputs,
        get_lab_metadata,
        get_track_scenario,
        go,
        ledger,
        mo,
        pipeline_inputs,
        report_export_panel,
        retention_inputs,
        split_inputs,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence, ledger):
    from mlsysbook_labs.contexts import preferred_context as _preferred_context
    _track_labels = {"tinyml": "TinyML", "mobile": "Mobile", "edge": "Edge", "cloud": "Cloud"}
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value=_track_labels[_preferred_context(ledger)],
        label="Deployment track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_track_scenario, track):
    track_id = track.value
    scenario = get_track_scenario(track_id)
    profile = {
        "tinyml": ("TinyML", "wearable sensor windows and a phone sync"),
        "mobile": ("Mobile", "private on-device context features"),
        "edge": ("Edge", "rare-event multi-sensor records"),
        "cloud": ("Cloud", "object-store shards and feature logs"),
    }[track_id]
    return profile, scenario, track_id


@app.cell
def _(mo, track_id):
    _track_key = track_id
    b_prediction = mo.ui.radio(
        [
            "The higher accuracy is trustworthy",
            "The higher accuracy is contaminated",
            "Both are equally trustworthy",
        ],
        label="What will the split comparison show?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        [
            "Read will bind",
            "Decode will bind",
            "Transform will bind",
            "The target will be met",
        ],
        label="What will bind after the pipeline change?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        [
            "Batch is fresh enough",
            "Streaming earns its traffic",
            "Local features earn their compute",
        ],
        label="Which freshness policy best fits this track?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        [
            "Schema checks catch the change",
            "Semantic checks are required",
            "No contract is cheaper overall",
        ],
        label="What happens when meaning changes without a type change?",
    ).form(submit_button_label="Lock Part E prediction")
    return b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id, a_task):
    _track_key = track_id
    _task_key = a_task.value
    a_prediction = mo.ui.radio(
        ["Finer labels resolve necessary actions", "Coarse labels suffice and buy more records", "Neither rubric supplies missing raw evidence"],
        label="What will changing label detail buy for this task?",
    ).form(submit_button_label="Lock Part A prediction")
    return (a_prediction,)


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_task = mo.ui.dropdown(
        {"Route a response to the observed condition": "route", "Record only whether an event occurred": "presence"},
        value="Route a response to the observed condition",
        label="Required task",
    )
    a_rubric = mo.ui.dropdown(
        {"Coarse event label": "coarse", "Condition-specific labels": "specific"},
        value="Coarse event label",
        label="Label rubric",
    )
    a_policy = mo.ui.dropdown(
        {
            "Coverage first": "coverage",
            "Deduplicate first": "deduplicate",
            "Newest first": "newest",
        },
        value="Coverage first",
        label="Retention policy",
    )
    b_strategy = mo.ui.dropdown(
        {"Entity-disjoint": "entity", "Record split": "record"},
        value="Entity-disjoint",
        label="Split strategy",
    )
    b_scope = mo.ui.dropdown(
        {"Fit on train only": "train", "Fit before split": "all"},
        value="Fit on train only",
        label="Preprocessing scope",
    )
    c_storage = mo.ui.dropdown(
        {"Native path": "native", "Constrained path": "constrained"},
        value="Native path",
        label="Storage path",
    )
    c_compression = mo.ui.dropdown(
        {"Balanced 4×": "balanced", "Dense 8×": "dense", "Raw": "raw"},
        value="Balanced 4×",
        label="Record format",
    )
    c_lanes = mo.ui.dropdown(
        {"1 lane": 1, "2 lanes": 2, "4 lanes": 4},
        value="2 lanes",
        label="Transform lanes",
    )
    d_policy = mo.ui.dropdown(
        {
            "Stream events": "stream",
            "Compute local features": "local feature",
            "Batch upload": "batch",
        },
        value="Stream events",
        label="Freshness policy",
    )
    a_review = mo.ui.dropdown(
        {"Single annotation": "single", "Review disagreements": "disagreements", "Add gold checks": "gold"},
        value="Single annotation", label="Audit review",
    )
    d_trace = mo.ui.dropdown(
        {"Feature state changes": "changing", "Feature state stays stable": "stable"},
        value="Feature state changes", label="Replay conditions",
    )
    e_level = mo.ui.dropdown(
        {"Semantic contract": "semantic", "Schema contract": "schema"},
        value="Semantic contract",
        label="Contract level",
    )
    final_choice = mo.ui.dropdown(
        {
            "Stream events": "stream",
            "Compute local features": "local feature",
            "Batch upload": "batch",
        },
        value=None,
        allow_select_none=True,
        label="Recommended freshness design",
    )
    final_rejected = mo.ui.dropdown(
        {
            "Batch upload": "batch",
            "Stream events": "stream",
            "Compute local features": "local feature",
        },
        value=None,
        allow_select_none=True,
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.dropdown(
        [
            "Freshness SLA tightens",
            "Traffic budget shrinks",
            "Producer semantics change",
        ],
        value=None,
        allow_select_none=True,
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.dropdown(
        [
            "Unseen cohorts remain",
            "Late records remain",
            "A semantic change can escape",
        ],
        value=None,
        allow_select_none=True,
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Evidence chain",
        full_width=True,
        placeholder="Use saved quantities to explain the recommendation and rejected alternative.",
    )
    return (
        a_policy,
        a_rubric,
        a_review,
        a_task,
        b_scope,
        b_strategy,
        c_compression,
        c_lanes,
        c_storage,
        d_policy,
        d_trace,
        e_level,
        final_choice,
        final_rejected,
        final_risk,
        final_trigger,
        rationale,
    )


@app.cell
def _(
    a_policy,
    a_rubric,
    a_review,
    a_task,
    b_scope,
    b_strategy,
    c_compression,
    c_lanes,
    c_storage,
    capture_experiment,
    contract_inputs,
    d_policy,
    d_trace,
    e_level,
    freshness_inputs,
    pipeline_inputs,
    retention_inputs,
    scenario,
    split_inputs,
):
    a_runs = {
        (p, r): capture_experiment("retention", **retention_inputs(scenario, p, r, a_task.value, a_review.value))
        for p in ("newest", "coverage", "deduplicate")
        for r in ("coarse", "specific")
    }
    a_base = capture_experiment("retention", **retention_inputs(scenario, "newest", "coarse", a_task.value, "single"))
    a_chosen = a_runs[(a_policy.value, a_rubric.value)]
    a_is_baseline = a_policy.value == "newest" and a_rubric.value == "coarse" and a_review.value == "single"
    a_result = a_runs[("newest", "specific")] if a_is_baseline else a_chosen
    b_base = capture_experiment("split", **split_inputs(scenario, "record", "all"))
    b_result = capture_experiment(
        "split", **split_inputs(scenario, b_strategy.value, b_scope.value)
    )
    c_base = capture_experiment(
        "pipeline",
        **pipeline_inputs(
            scenario, storage_path="constrained", compression="raw", transform_lanes=1
        ),
    )
    c_result = capture_experiment(
        "pipeline",
        **pipeline_inputs(
            scenario,
            storage_path=c_storage.value,
            compression=c_compression.value,
            transform_lanes=c_lanes.value,
        ),
    )
    d_runs = {
        p: capture_experiment("freshness", **freshness_inputs(scenario, p, d_trace.value))
        for p in ("batch", "stream", "local feature")
    }
    d_base = d_runs["batch"]
    d_chosen = d_runs[d_policy.value]
    d_result = d_runs["stream"] if d_policy.value == "batch" else d_chosen
    e_base = capture_experiment("contract", **contract_inputs(scenario, "none"))
    e_result = capture_experiment(
        "contract", **contract_inputs(scenario, e_level.value)
    )
    return (
        a_base,
        a_chosen,
        a_result,
        a_runs,
        b_base,
        b_result,
        c_base,
        c_result,
        d_base,
        d_chosen,
        d_result,
        d_runs,
        e_base,
        e_result,
    )


@app.cell
def _(
    a_base,
    a_chosen,
    a_policy,
    a_rubric,
    a_review,
    a_task,
    a_prediction,
    a_result,
    a_runs,
    b_base,
    b_prediction,
    b_result,
    b_scope,
    b_strategy,
    c_base,
    c_compression,
    c_lanes,
    c_prediction,
    c_result,
    c_storage,
    capture_evidence,
    d_base,
    d_policy,
    d_trace,
    d_prediction,
    d_chosen,
    d_result,
    d_runs,
    e_base,
    e_level,
    e_prediction,
    e_result,
    mo,
    set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    def payload(run):
        return {"inputs": run.inputs, "outputs": run.result}

    def capture(
        part,
        prediction,
        inputs,
        baseline,
        result,
        alternatives,
        decision,
        chosen_result,
        result_role="tested intervention",
    ):
        return capture_evidence(
            track=track_id,
            part=part,
            prediction=prediction,
            inputs=inputs,
            baseline=payload(baseline),
            result=payload(result),
            alternatives=tuple(payload(run) for run in alternatives),
            decision=decision,
            model_key="v1_04_experiments",
            upstream_inputs=inputs if part in {"A", "D"} else None,
            chosen_result=payload(chosen_result),
            result_role=result_role,
        )

    a_capture = mo.ui.button(
        label="Capture label and retention contrast",
        kind="success",
        disabled=a_prediction.value is None,
        on_click=lambda _v: store(
            "A",
            capture(
                "A",
                a_prediction.value,
                {"policy": a_policy.value, "rubric": a_rubric.value, "task": a_task.value, "review": a_review.value},
                a_base,
                a_result,
                a_runs.values(),
                a_policy.value,
                a_chosen,
                "rejected alternative"
                if a_policy.value == "newest" and a_rubric.value == "coarse" and a_review.value == "single"
                else "tested intervention",
            ),
        ),
    )
    b_capture = mo.ui.button(
        label="Capture split contrast",
        kind="success",
        disabled=b_prediction.value is None
        or (b_strategy.value == "record" and b_scope.value == "all"),
        on_click=lambda _v: store(
            "B",
            capture(
                "B",
                b_prediction.value,
                {"strategy": b_strategy.value, "preprocessing_scope": b_scope.value},
                b_base,
                b_result,
                (),
                f"{b_strategy.value}/{b_scope.value}",
                b_result,
            ),
        ),
    )
    c_capture = mo.ui.button(
        label="Capture pipeline contrast",
        kind="success",
        disabled=c_prediction.value is None
        or (
            c_storage.value == "constrained"
            and c_compression.value == "raw"
            and c_lanes.value == 1
        ),
        on_click=lambda _v: store(
            "C",
            capture(
                "C",
                c_prediction.value,
                {
                    "storage": c_storage.value,
                    "compression": c_compression.value,
                    "transform_lanes": c_lanes.value,
                },
                c_base,
                c_result,
                (),
                f"{c_storage.value}/{c_compression.value}/{c_lanes.value}",
                c_result,
            ),
        ),
    )
    d_capture = mo.ui.button(
        label="Capture freshness contrast",
        kind="success",
        disabled=d_prediction.value is None,
        on_click=lambda _v: store(
            "D",
            capture(
                "D",
                d_prediction.value,
                {"policy": d_policy.value, "trace": d_trace.value},
                d_base,
                d_result,
                d_runs.values(),
                d_policy.value,
                d_chosen,
                "rejected alternative"
                if d_policy.value == "batch"
                else "tested intervention",
            ),
        ),
    )
    e_capture = mo.ui.button(
        label="Capture contract contrast",
        kind="success",
        disabled=e_prediction.value is None,
        on_click=lambda _v: store(
            "E",
            capture(
                "E",
                e_prediction.value,
                {"contract_level": e_level.value},
                e_base,
                e_result,
                (),
                e_level.value,
                e_result,
            ),
        ),
    )
    return a_capture, b_capture, c_capture, d_capture, e_capture


@app.cell
def _(get_lab_metadata, mo, track, track_id, profile):
    from mlsysbook_labs.notebook import notebook_scenario as _scenario
    from mlsysim import Scenarios as _Scenarios
    from mlsysbook_labs.learning_paths import get_learning_path as _get_learning_path
    from mlsysbook_labs.notebook import (
        notebook_bridge as _bridge, notebook_header as _header,
        notebook_learning_map as _learning_map, notebook_shell as _opening_shell,
    )

    learning_contract = _get_learning_path("vol1", 4)
    _journey = _Scenarios.DeploymentJourneys.get(track_id)
    _episode = _journey.episode(4)
    learning_contract["narrative"] = {
        "project": _journey.mission,
        "scene": _episode.scene,
        "carry_forward": _episode.handoff,
        "scope": _journey.boundary,
    }
    _opening_shell([
        _header(get_lab_metadata("vol1/lab_04_data_engr.py"), _episode.question,
                topics=[part["label"] for part in learning_contract["parts"]]),
        track,
        _scenario(learning_contract, track=track_id, scene=_episode.scene),
        _learning_map(learning_contract),
        mo.md(f"**Reference used in today’s instruments:** {profile[1]}. This controlled case tests the chapter’s mechanism; transfer the reasoning to your project without treating the result as project validation."),
        mo.callout(mo.md("These are supplied, controlled reference studies. Their results do not validate the deployment project. Choose a prediction in Part A to open its live experiment; a wrong prediction still opens the evidence."), kind="info"),
    ])
    return (learning_contract,)


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS,
    a_base,
    a_capture,
    a_chosen,
    a_policy,
    a_rubric,
    a_review,
    a_task,
    a_prediction,
    a_result,
    apply_plotly_theme,
    audit_evidence,
    b_base,
    b_capture,
    b_prediction,
    b_result,
    b_scope,
    b_strategy,
    c_base,
    c_capture,
    c_compression,
    c_lanes,
    c_prediction,
    c_result,
    c_storage,
    d_base,
    d_capture,
    d_chosen,
    d_policy,
    d_trace,
    d_prediction,
    d_result,
    d_runs,
    e_base,
    e_capture,
    e_level,
    e_prediction,
    e_result,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    go,
    mo,
    rationale,
    scenario,
    track_id,
    learning_contract,
):
    from mlsysbook_labs.notebook import notebook_shell as _shell, notebook_learning_part as _learning_part, notebook_tab_hint as _tab_hint
    _route = {part["id"]: part for part in learning_contract["parts"]}


    _captures = get_evidence()
    audit = audit_evidence(
        _captures,
        track=track_id,
        required_parts=tuple("ABCDE"),
        contrast_required_parts=tuple("ABCDE"),
        per_part_upstream_inputs={"A": {"policy": a_policy.value, "rubric": a_rubric.value, "task": a_task.value, "review": a_review.value}, "D": {"policy": d_policy.value, "trace": d_trace.value}},
    )

    def table(rows):
        from mlsysbook_labs.notebook import notebook_table as _compact_table
        return _compact_table(rows)

    def saved(part):
        cap = _captures.get(part)
        if cap is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        data = cap.to_dict()
        if part == "A" and data["inputs"] != {"policy": a_policy.value, "rubric": a_rubric.value, "task": a_task.value, "review": a_review.value}:
            return mo.callout(mo.md("**Saved comparison differs from the current label design.** Live evidence above has updated; the saved snapshot still records its original task, rubric, and retention choice. Capture again to submit the current design."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(
                mo.md(
                    "**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this comparison."
                ),
                kind="danger",
            )
        return mo.Html(
            f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later controls do not change this saved comparison.</small></div>'
        )

    def part_a():
        intro = mo.md(
            "The team has raw candidate records but has not agreed on what a label must distinguish. "
            "Choose the required task, predict, then spend the same annotation budget on coarse or condition-specific labels. "
            "The listed actions are supplied requirements for this controlled reference, not medical or driving recommendations."
        )
        if a_prediction.value is None:
            return mo.vstack([intro, a_task, a_prediction])
        base, result, chosen = a_base.result, a_result.result, a_chosen.result
        fig = go.Figure(
            [
                go.Bar(
                    x=["Newest", result["policy"]],
                    y=[
                        base["represented_cohort_count"],
                        result["represented_cohort_count"],
                    ],
                    marker_color=[COLORS["Grey"], COLORS["BlueLine"]],
                )
            ]
        )
        fig.update_layout(
            height=245,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Represented conditions (count)",
            showlegend=False,
        )
        rows = [
            {
                "Run": "Newest / coarse",
                "Resolvable": base["action_resolvable_records"],
                "Annotation": f"{base['annotation_time']['magnitude']:.0f} min",
                "Conditions": base["represented_cohort_count"],
                "Records": len(base["retained_ids"]),
                "Duplicates": base["duplicate_records"],
                "Task evidence": base["records_with_task_evidence"],
            },
            {
                "Run": f"{result['policy']} / {result['rubric_name']}",
                "Resolvable": result["action_resolvable_records"],
                "Annotation": f"{result['annotation_time']['magnitude']:.0f} min",
                "Conditions": result["represented_cohort_count"],
                "Records": len(result["retained_ids"]),
                "Duplicates": result["duplicate_records"],
                "Task evidence": result["records_with_task_evidence"],
            },
        ]
        return mo.vstack(
            [
                intro,
                a_task,
                a_prediction,
                a_policy,
                a_rubric,
                mo.md("To isolate label detail, select **Newest first**. To isolate retention, select **Coarse event label**. Changing the required task asks for a new prediction."),
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {a_prediction.value}. Compared with newest / coarse, the displayed intervention is **{result['policy']} / {result['rubric_name']}**. "
                        f"Your selected design retains **{chosen['represented_cohort_count']} {'condition' if chosen['represented_cohort_count'] == 1 else 'conditions'}** and **{chosen['records_with_task_evidence']} {'record' if chosen['records_with_task_evidence'] == 1 else 'records'} with task evidence**; "
                        f"**{chosen['action_resolvable_records']} {'record has' if chosen['action_resolvable_records'] == 1 else 'records have'}** both evidence and a label that uniquely specifies the required action. "
                        "If you select the baseline, the displayed contrast uses newest / specific as its rejected alternative. The audit set below stays fixed and has its own labor ledger."
                    ),
                    kind="info",
                ),
                mo.md("**Selected design · inspect each retained record**"),
                table([
                    {"Record": row["record_id"], "Condition": row["condition"], "Assigned label": row["label"],
                     "Required action": row["required_action"], "Actions sharing label": ", ".join(row["possible_actions"]),
                     "Raw evidence": "present" if row["has_task_evidence"] else "missing",
                     "Action specified": "yes" if row["action_resolvable"] else "unresolved"}
                    for row in chosen["label_assignments"]
                ]),
                mo.callout(mo.md(
                    f"The declared rubric has **{chosen['ambiguous_label_count']}** ambiguous labels, where one label merges conditions needing different actions. "
                    "Missing raw evidence remains unresolved even with specific labels. Switch to the presence-only task: does finer annotation still buy a necessary distinction? "
                    "This checks a finite specification; it does not predict learned accuracy or deployment safety."
                ), kind="warn" if chosen["ambiguous_label_count"] else "info"),
                mo.md("**Audit the annotators · does agreement establish correctness?**\nThe following fixed audit set is separate from the retained pool. Predict which error disagreement review will miss, then change the review policy. Its labor is additional to the acquisition budget above. Supplied independent references make this a controlled audit; they are not inferred from agreement."),
                a_review,
                table([{"Record": row["record_id"], "First": row["first"],
                        "Second": row["second"] or "not collected", "Reviewed": row["reviewed"],
                        "Final": row["final"] or "abstain", "Supplied audit reference": row["supplied_reference"] or "unavailable",
                        "Mismatch": row["reference_mismatch"], "Unresolved": row["unresolved"]}
                       for row in chosen["annotation_audit"]["records"]]),
                mo.md(
                    "Only one annotation per record: agreement and kappa are unobserved."
                    if chosen["annotation_audit"]["observed_agreement"] is None else
                    f"Observed pair agreement: **{chosen['annotation_audit']['observed_agreement']:.2f}**; chance agreement from rater marginals: **{chosen['annotation_audit']['chance_agreement']:.2f}**; "
                    + (f"Cohen's kappa: **{chosen['annotation_audit']['cohens_kappa']:.2f}**. " if chosen['annotation_audit']['cohens_kappa'] is not None else "Kappa is undefined because chance agreement is one. ")
                    + "These summarize the supplied annotation pairs before adjudication. Gold review changes final labels, not the original pair agreement."
                ),
                mo.callout(mo.md(
                    f"Audit labor: **{chosen['annotation_audit']['labor_minutes']:.0f} min**, compared with **{base['annotation_audit']['labor_minutes']:.0f} min** for single annotation. "
                    f"Remaining reference mismatches: **{chosen['annotation_audit']['reference_mismatches']}**; unresolved records: **{chosen['annotation_audit']['unresolved_records']}**. "
                    "Double annotation exposes disagreements, but the shared wrong label survives disagreement-only review. Independent gold checks repair that supplied error. Missing raw evidence still requires abstention or new collection. These finite counts are not population error-rate estimates."
                )),
                a_capture,
                saved("A"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Whole records are admitted until the annotation-time budget is exhausted. Condition-specific annotation costs twice the coarse annotation time in this hypothetical reference. "
                            "This price is assumed, not measured. Coverage rotates across observed conditions; deduplication removes explicitly linked duplicates. "
                            "Required actions stay fixed when the rubric changes. A label is action-sufficient only when every condition in its declared vocabulary requires the same action. "
                            "Omitting a condition from the retained set cannot repair the rubric. Selection never uses the supplied task-evidence flags. "
                            "The audit panel replays supplied first/second annotations and independent references on a separate fixed set. Adjudication uses the supplied reference only when available. Its annotation, adjudication, and gold-check costs are hypothetical explicit inputs; it does not estimate learned quality."
                        )
                    }
                ),
            ]
        )

    def part_b():
        intro = mo.md(
            "The baseline overlaps entity keys and fits preprocessing before the split. Change the split, preprocessing scope, or both."
        )
        if b_prediction.value is None:
            return mo.vstack([intro, b_prediction])
        base, result = b_base.result, b_result.result
        fig = go.Figure()
        fig.add_bar(
            name="Key leakage",
            x=["Baseline", "Selected"],
            y=[base["key_leakage_percent"], result["key_leakage_percent"]],
            marker_color=COLORS["OrangeLine"],
        )
        fig.add_bar(
            name="Observed accuracy",
            x=["Baseline", "Selected"],
            y=[base["observed_accuracy_percent"], result["observed_accuracy_percent"]],
            marker_color=COLORS["BlueLine"],
        )
        fig.update_layout(
            barmode="group",
            height=265,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Entity keys or records (%)",
            legend_orientation="h",
        )
        rows = [
            {
                "Run": "Leaky baseline",
                "Overlapping keys": base["overlapping_entity_count"],
                "Test records seen": base["preprocessing_test_records_seen"],
                "Supplied-label accuracy": f"{base['observed_accuracy_percent']:.0f}%",
            },
            {
                "Run": f"{b_strategy.value}/{b_scope.value}",
                "Overlapping keys": result["overlapping_entity_count"],
                "Test records seen": result["preprocessing_test_records_seen"],
                "Supplied-label accuracy": f"{result['observed_accuracy_percent']:.0f}%",
            },
        ]
        return mo.vstack(
            [
                intro,
                b_prediction,
                mo.hstack([b_strategy, b_scope], widths="equal", wrap=True),
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {b_prediction.value}. The selected run has **{result['overlapping_entity_count']} overlapping keys** and sees **{result['preprocessing_test_records_seen']} test records** during preprocessing."
                    ),
                    kind="danger" if result["key_leakage_fraction"] else "success",
                ),
                b_capture,
                saved("B"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Key leakage is overlapping entity IDs divided by distinct test entity IDs. Preprocessing leakage counts test records used to fit transforms. Accuracy is counted from supplied predictions; no leakage-to-quality equation is used."
                        )
                    }
                ),
            ]
        )

    def part_c():
        intro = mo.md(
            "The baseline uses constrained reads, raw records, and one transform lane. Compression reduces bytes and adds decode work. Predict the binding stage."
        )
        if c_prediction.value is None:
            return mo.vstack([intro, c_prediction])
        base, result = c_base.result, c_result.result
        fig = go.Figure()
        for label, run, color in (
            ("Baseline", base, COLORS["Grey"]),
            ("Selected", result, COLORS["BlueLine"]),
        ):
            fig.add_bar(
                name=label,
                x=["Read", "Decode", "Transform"],
                y=[
                    run["read_rate_per_second"],
                    run["decode_rate_per_second"],
                    run["transform_rate_per_second"],
                ],
                marker_color=color,
            )
        fig.add_hline(
            y=result["required_rate_per_second"],
            line_dash="dash",
            annotation_text="Required rate",
        )
        fig.update_layout(
            barmode="group",
            height=275,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Service rate (records/s)",
            legend_orientation="h",
        )
        rows = [
            {
                "Run": "Baseline",
                "Bottleneck": base["bottleneck_stage"],
                "Supply": f"{base['service_rate_per_second']:.1f}/s",
                "Required": f"{base['required_rate_per_second']:.1f}/s",
                "Waiting": f"{base['accelerator_wait_percent']:.1f}%",
                "Outcome": "PASS" if base["meets_required_rate"] else "FAIL",
            },
            {
                "Run": "Selected",
                "Bottleneck": result["bottleneck_stage"],
                "Supply": f"{result['service_rate_per_second']:.1f}/s",
                "Required": f"{result['required_rate_per_second']:.1f}/s",
                "Waiting": f"{result['accelerator_wait_percent']:.1f}%",
                "Outcome": "PASS" if result["meets_required_rate"] else "FAIL",
            },
        ]
        return mo.vstack(
            [
                intro,
                c_prediction,
                mo.hstack(
                    [c_storage, c_compression, c_lanes], widths="equal", wrap=True
                ),
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {c_prediction.value}. The selected bottleneck is **{result['bottleneck_stage']}** at **{result['service_rate_per_second']:.1f} records/s** versus **{result['required_rate_per_second']:.1f} records/s** required."
                    ),
                    kind="success" if result["meets_required_rate"] else "danger",
                ),
                c_capture,
                saved("C"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Read, decode, and transform are independent overlapped stages with enough buffering after warmup. Their minimum rate is steady-state throughput, not one record’s serial latency. Compression changes bytes and decode work."
                        )
                    }
                ),
            ]
        )

    def part_d():
        intro = mo.md(
            "The team must decide whether faster delivery buys a useful decision. Compare policies on the same supplied feature-state history, then hold the feature state stable. Age alone does not establish a behavior loss."
        )
        if d_prediction.value is None:
            return mo.vstack([intro, d_prediction])
        base, result, chosen = d_base.result, d_result.result, d_chosen.result
        fig = go.Figure(
            [
                go.Bar(
                    x=["Batch", result["policy_name"]],
                    y=[
                        base["worst_case_age"]["magnitude"],
                        result["worst_case_age"]["magnitude"],
                    ],
                    marker_color=[COLORS["Grey"], COLORS["GreenLine"]],
                )
            ]
        )
        fig.add_hline(
            y=d_result.inputs["freshness_sla"]["magnitude"],
            line_dash="dash",
            annotation_text="Freshness SLA",
        )
        fig.update_layout(
            height=245,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Worst-case feature age (s)",
            showlegend=False,
        )
        rows = [
            {
                "Policy": name,
                "Worst age": f"{run.result['worst_case_age']['magnitude']:.1f} s",
                "Traffic": f"{run.result['traffic_megabytes']:.1f} MB",
                "Annotation": f"{run.result['annotation_time']['magnitude']:.1f} min",
                "SLA": "PASS" if run.result["meets_freshness_sla"] else "FAIL",
                "Replay mismatches": run.result["stale_decisions"],
            }
            for name, run in d_runs.items()
        ]
        return mo.vstack(
            [
                intro,
                d_prediction,
                d_policy,
                d_trace,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {d_prediction.value}. The tested comparison is **batch versus {result['policy_name']}**. Your chosen **{d_policy.value}** policy gives **{chosen['worst_case_age']['magnitude']:.1f} s** worst-case age and **{chosen['traffic_megabytes']:.1f} MB** traffic."
                    ),
                    kind="success" if chosen["meets_freshness_sla"] else "danger",
                ),
                mo.md("**Decision replay · selected policy**\nEach query uses state available through query time minus the policy's maximum age. This deliberately conservative constant-lag stress test is not an actual batch-arrival schedule. The action rule is supplied, and is held fixed across policies."),
                table([{"Query (s)": row["query_second"], "Evidence through (s)": row["observed_through_second"],
                        "Action from available evidence": row["available_action"], "Currently required": row["current_required_action"],
                        "Mismatch": row["mismatch"]} for row in chosen["decision_trace"]]),
                mo.callout(mo.md(
                    f"The chosen policy produces **{chosen['stale_decisions']} mismatches** on this finite replay. Switch to stable conditions: older evidence can give the same decisions, even when an age SLA fails. "
                    "Faster updates are valuable when changed information changes a required action; this fixture does not estimate learned accuracy, drift rates, or field safety. The local-feature policy assumes it preserves the required state; semantic sufficiency requires separate validation."
                )),
                d_capture,
                saved("D"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Worst-case age sums collection, transport, and feature-compute time. Traffic and annotation time count explicit events over a fixed horizon. The SLA changes acceptance, not physical age or decisions. Decision replay independently compares the latest supplied state with the state available at a conservative constant lag. Changing versus stable state is the counterexample, not an age-to-accuracy curve."
                        )
                    }
                ),
            ]
        )

    def part_e():
        intro = mo.md(
            f"The producer changes **{scenario.semantic_field_name}** from **{scenario.expected_semantic_unit}** while preserving its declared schema version. Compare no enforcement with a contract."
        )
        if e_prediction.value is None:
            return mo.vstack([intro, e_prediction])
        base, result = e_base.result, e_result.result
        fig = go.Figure()
        fig.add_bar(
            name="Escaped errors",
            x=["None", e_level.value],
            y=[base["escaped_semantic_errors"], result["escaped_semantic_errors"]],
            marker_color=COLORS["OrangeLine"],
        )
        fig.add_bar(
            name="Rejected records",
            x=["None", e_level.value],
            y=[base["rejected_records"], result["rejected_records"]],
            marker_color=COLORS["BlueLine"],
        )
        fig.update_layout(
            barmode="group",
            height=255,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Records (count)",
            legend_orientation="h",
        )
        rows = [
            {
                "Run": "No contract",
                "Accepted": base["accepted_records"],
                "Rejected": base["rejected_records"],
                "Escaped": base["escaped_semantic_errors"],
                "Validation": f"{base['validation_time']['magnitude']:.3f} s",
                "Recovery": f"{base['recovery_time']['magnitude']:.0f} min",
            },
            {
                "Run": e_level.value,
                "Accepted": result["accepted_records"],
                "Rejected": result["rejected_records"],
                "Escaped": result["escaped_semantic_errors"],
                "Validation": f"{result['validation_time']['magnitude']:.3f} s",
                "Recovery": f"{result['recovery_time']['magnitude']:.0f} min",
            },
        ]
        return mo.vstack(
            [
                intro,
                e_prediction,
                e_level,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {e_prediction.value}. The contract spends **{result['validation_time']['magnitude']:.3f} s**, rejects **{result['rejected_records']} records**, and lets **{result['escaped_semantic_errors']} semantic errors** escape."
                    ),
                    kind="success"
                    if result["escaped_semantic_errors"] == 0
                    else "danger",
                ),
                e_capture,
                saved("E"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Schema enforcement checks the declared version. Semantic enforcement also checks the consumer’s expected unit. Recovery time is charged only for invalid records that escape."
                        )
                    }
                ),
            ]
        )

    def synthesis():
        rows = []
        for part in "ABCDE":
            cap = _captures.get(part)
            rows.append(
                {
                    "Part": part,
                    "Prediction": cap.to_dict()["prediction"] if cap else "—",
                    "Evidence": "CURRENT"
                    if cap
                    and part not in audit.stale
                    and (part, part) not in audit.identical_pairs
                    else ("STALE" if cap else "MISSING"),
                }
            )
        saved_d = _captures["D"].to_dict()["decision"] if "D" in _captures else None
        complete = (
            audit.complete
            and final_choice.value is not None
            and final_rejected.value is not None
            and final_trigger.value is not None
            and final_risk.value is not None
            and final_choice.value == saved_d
            and final_rejected.value != final_choice.value
            and bool(rationale.value.strip())
        )
        return mo.vstack(
            [
                mo.md(
                    "### Synthesis · Defend one data design (6 min)\nUse the saved chain: retained evidence → split integrity → pipeline supply → freshness cost → producer boundary. Choose the Part D design, reject a quantified alternative from its table, name a limitation, and set a reevaluation trigger."
                ),
                table(rows),
                mo.callout(
                    mo.md(
                        "Saved snapshots preserve original predictions, exact evaluation arguments, and outputs."
                    ),
                    kind="info",
                ),
                mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
                mo.hstack([final_trigger, final_risk], widths="equal", wrap=True),
                rationale,
                mo.callout(
                    mo.md(
                        "**Ready for the local report.**"
                        if complete
                        else "Complete five current contrasts, match the Part D decision, reject another tested policy, and write the evidence chain."
                    ),
                    kind="success" if complete else "warn",
                ),
            ]
        )

    tabs = mo.ui.tabs(
        {
            "Part A": _learning_part(learning_contract, "A", part_a()),
            "Part B": _learning_part(learning_contract, "B", part_b()),
            "Part C": _learning_part(learning_contract, "C", part_c()),
            "Part D": _learning_part(learning_contract, "D", part_d()),
            "Part E": _learning_part(learning_contract, "E", part_e()),
            "Synthesis": synthesis(),
        }
    )
    _shell([_tab_hint(), tabs])
    return (audit,)


@app.cell(hide_code=True)
def _():
    from mlsysbook_labs.learning_paths import get_learning_path as _submission_path
    from mlsysbook_labs.notebook import notebook_shell as _submission_shell, notebook_submission as _submission
    _submission_shell([_submission(_submission_path(1, 4))])
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
    learning_contract,
    mo,
    profile,
    rationale,
    report_export_panel,
    track_id,
):
    _captures = get_evidence()
    _saved_d = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and final_choice.value is not None
        and final_rejected.value is not None
        and final_trigger.value is not None
        and final_risk.value is not None
        and final_choice.value == _saved_d
        and final_rejected.value != final_choice.value
        and bool(rationale.value.strip())
    )
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not _ready, _download_readiness(volume=1))
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_04_data_engr.py"),
        track=track_id,
        scenario=profile[1],
        learning_objectives=learning_contract["learning_objectives"],
        predictions={p: snapshots[p]["prediction"] for p in "ABCDE"},
        knob_settings={p: snapshots[p]["inputs"] for p in "ABCDE"},
        evidence_summary={
            p: {
                "baseline": snapshots[p]["baseline"],
                "result": snapshots[p]["result"],
                "alternatives": snapshots[p]["alternatives"],
                "chosen": snapshots[p]["chosen_result"],
            }
            for p in "ABCDE"
        },
        binding_constraints={
            "split": snapshots["B"]["chosen_result"]["outputs"][
                "overlapping_entity_keys"
            ],
            "pipeline": snapshots["C"]["chosen_result"]["outputs"]["bottleneck_stage"],
            "freshness": snapshots["D"]["chosen_result"]["outputs"][
                "meets_freshness_sla"
            ],
            "contract": snapshots["E"]["chosen_result"]["outputs"][
                "escaped_semantic_errors"
            ],
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
            "Higher evaluation accuracy can hide leakage.",
            "The slowest overlapped stage bounds supply.",
            "Freshness and contracts consume explicit resources.",
        ],
        reflections={
            "rationale": rationale.value,
            "reevaluation_trigger": final_trigger.value,
        },
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
            "scenario": "Illustrative fixed records and supplied outcomes.",
            "calculations": "Deterministic data-pipeline experiment model.",
        },
    )
    from mlsysbook_labs.notebook import notebook_shell as _report_shell
    _report_shell([report_export_panel(report)])
    return (report,)




@app.cell
def _(mo, get_evidence, final_choice, final_rejected, final_risk, final_trigger, rationale, audit):
    from mlsysbook_labs.notebook import notebook_shell as _save_shell
    _draft_identity = (audit.to_dict(), get_evidence(), final_choice.value, final_rejected.value, final_risk.value, final_trigger.value, rationale.value)
    save_discussion = mo.ui.run_button(label="Save discussion")
    _save_shell([
        save_discussion,
        mo.md("Save your completed discussion locally. Editing a decision, rationale, or saved comparison requires saving again. Download remains available independently of local storage."),
    ])
    return (save_discussion,)


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
    save_discussion,
):
    _captures = get_evidence()
    _saved_d = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and final_choice.value is not None
        and final_rejected.value is not None
        and final_trigger.value is not None
        and final_risk.value is not None
        and final_choice.value == _saved_d
        and final_rejected.value != final_choice.value
        and bool(rationale.value.strip())
    )
    _status = "DRAFT READY · SAVE DISCUSSION TO RECORD" if _ready else "EVIDENCE IN PROGRESS"
    if _ready and save_discussion.value:
        try:
            ledger.save(
                chapter=4,
                design={
                    "schema_version": 1,
                    "lab_id": "v1_04",
                    "track_id": track_id,
                    "model_id": "v1_04_experiments",
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
        except Exception:
            _status = "LOCAL SAVE FAILED · DOWNLOAD THE REPORT TO KEEP YOUR EVIDENCE"
        else:
            _status = "SAVED"
    from mlsysbook_labs.notebook import notebook_shell as _status_shell, notebook_status as _status_view
    _status_shell([_status_view("Lab 04", _status)])
    return


if __name__ == "__main__":
    app.run()
