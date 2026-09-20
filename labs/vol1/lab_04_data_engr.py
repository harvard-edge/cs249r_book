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
    from mlsysim.engine.v1_04_experiments import (
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
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML",
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
    a_prediction = mo.ui.radio(
        ["Newest records", "Widest cohort coverage", "Remove duplicates first"],
        label="Which policy preserves the most useful evidence under the fixed budget?",
    ).form(submit_button_label="Lock Part A prediction")
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
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
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
        b_scope,
        b_strategy,
        c_compression,
        c_lanes,
        c_storage,
        d_policy,
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
    b_scope,
    b_strategy,
    c_compression,
    c_lanes,
    c_storage,
    capture_experiment,
    contract_inputs,
    d_policy,
    e_level,
    freshness_inputs,
    pipeline_inputs,
    retention_inputs,
    scenario,
    split_inputs,
):
    a_runs = {
        p: capture_experiment("retention", **retention_inputs(scenario, p))
        for p in ("newest", "coverage", "deduplicate")
    }
    a_base = a_runs["newest"]
    a_chosen = a_runs[a_policy.value]
    a_result = a_runs["coverage"] if a_policy.value == "newest" else a_chosen
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
        p: capture_experiment("freshness", **freshness_inputs(scenario, p))
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
            chosen_result=payload(chosen_result),
            result_role=result_role,
        )

    a_capture = mo.ui.button(
        label="Capture retention contrast",
        kind="success",
        disabled=a_prediction.value is None,
        on_click=lambda _v: store(
            "A",
            capture(
                "A",
                a_prediction.value,
                {"policy": a_policy.value},
                a_base,
                a_result,
                a_runs.values(),
                a_policy.value,
                a_chosen,
                "rejected alternative"
                if a_policy.value == "newest"
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
                {"policy": d_policy.value},
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
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:10px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:780px}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-active{color:#86efac}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(
        f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 04</span><span>ABOUT 50–55 MIN</span></div><h1>Evidence Through the Data Pipeline</h1><p>Which records preserve trustworthy evidence, and can the pipeline deliver them before execution or freshness budgets bind?</p><div class="pilot-meta"><div><b>Context</b><br>{profile[1]}</div><div><b>Investigation</b><br>Five controlled contrasts</div><div><b>Deliverable</b><br>Evidence-backed data design</div></div></section>"""
    )
    mo.vstack(
        [
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            css,
            header,
            track,
            mo.Html(
                '<p class="pilot-note">The track selector sits outside the briefing header. All records and outcomes are illustrative scenario fixtures.</p>'
            ),
        ],
        gap=0.5,
    )
    return


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
):
    _captures = get_evidence()
    audit = audit_evidence(
        _captures,
        track=track_id,
        required_parts=tuple("ABCDE"),
        contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style(
            {"max-width": "100%", "overflow-x": "auto"}
        )

    def saved(part):
        cap = _captures.get(part)
        if cap is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(
                mo.md(
                    "**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this comparison."
                ),
                kind="danger",
            )
        data = cap.to_dict()
        return mo.Html(
            f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later controls do not rewrite this record.</small></div>'
        )

    def part_a():
        intro = mo.md(
            "### A · Which examples are worth retaining? (8 min)\nSix candidates compete for one fixed annotation-time budget. Predict first, then compare coverage, duplicates, and supplied task evidence."
        )
        if a_prediction.value is None:
            return mo.vstack([intro, a_prediction])
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
            yaxis_title="Represented cohorts (count)",
            showlegend=False,
        )
        rows = [
            {
                "Run": "Newest",
                "Annotation": f"{base['annotation_time']['magnitude']:.0f} min",
                "Cohorts": base["represented_cohort_count"],
                "Duplicates": base["duplicate_records"],
                "Task evidence": base["records_with_task_evidence"],
            },
            {
                "Run": result["policy"],
                "Annotation": f"{result['annotation_time']['magnitude']:.0f} min",
                "Cohorts": result["represented_cohort_count"],
                "Duplicates": result["duplicate_records"],
                "Task evidence": result["records_with_task_evidence"],
            },
        ]
        return mo.vstack(
            [
                intro,
                a_prediction,
                a_policy,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {a_prediction.value}. The tested comparison is **newest versus {result['policy']}**. Your chosen **{a_policy.value}** policy retains **{chosen['represented_cohort_count']} cohorts**, **{chosen['duplicate_records']} duplicates**, and **{chosen['records_with_task_evidence']} records with supplied task evidence**."
                    ),
                    kind="info",
                ),
                a_capture,
                saved("A"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Whole records are admitted until the annotation-time budget is exhausted. Coverage rotates across cohorts; deduplication removes records explicitly marked as duplicates. Supplied outcome flags do not influence selection."
                        )
                    }
                ),
            ]
        )

    def part_b():
        intro = mo.md(
            "### B · Can higher evaluation accuracy mean worse evidence? (10 min)\nThe baseline overlaps entity keys and fits preprocessing before the split. Change the split, preprocessing scope, or both."
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
            "### C · Why is execution waiting for data? (10 min)\nThe baseline uses constrained reads, raw records, and one transform lane. Compression reduces bytes and adds decode work. Predict the binding stage."
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
            "### D · How fresh must the evidence be? (8 min)\nCompare batching with streaming or local feature computation. Each policy changes age, traffic, and annotation time over the same horizon."
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
            }
            for name, run in d_runs.items()
        ]
        return mo.vstack(
            [
                intro,
                d_prediction,
                d_policy,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {d_prediction.value}. The tested comparison is **batch versus {result['policy_name']}**. Your chosen **{d_policy.value}** policy gives **{chosen['worst_case_age']['magnitude']:.1f} s** worst-case age and **{chosen['traffic_megabytes']:.1f} MB** traffic."
                    ),
                    kind="success" if chosen["meets_freshness_sla"] else "danger",
                ),
                d_capture,
                saved("D"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Worst-case age sums collection, transport, and feature-compute time. Traffic and annotation time count explicit events over a fixed horizon. The SLA changes acceptance, not physical age."
                        )
                    }
                ),
            ]
        )

    def part_e():
        intro = mo.md(
            f"### E · What happens when the producer changes? (8 min)\nThe producer changes **{scenario.semantic_field_name}** from **{scenario.expected_semantic_unit}** while preserving its declared schema version. Compare no enforcement with a contract."
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
            "Part A": part_a(),
            "Part B": part_b(),
            "Part C": part_c(),
            "Part D": part_d(),
            "Part E": part_e(),
            "Synthesis": synthesis(),
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
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_04_data_engr.py"),
        track=track_id,
        scenario=profile[1],
        learning_objectives=[
            "Allocate fixed annotation time",
            "Detect split leakage",
            "Diagnose supply, freshness, and producer boundaries",
        ],
        predictions={p: snapshots[p]["prediction"] for p in "ABCDE"},
        knob_settings={p: snapshots[p]["inputs"] for p in "ABCDE"},
        evidence_summary={
            p: {
                "baseline": snapshots[p]["baseline"],
                "result": snapshots[p]["result"],
                "alternatives": snapshots[p]["alternatives"],
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
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
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
    mo.Html(
        f'<div class="lab-hud"><span>LAB 04 · Evidence Through the Data Pipeline · STATUS: {_status}</span></div>'
    )
    return


if __name__ == "__main__":
    app.run()
