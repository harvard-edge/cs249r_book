import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="full", app_title="Lab 08: Useful Work from a Shared Fleet · MLSysBook"
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
    from mlsysim.labs.experiments.v2_08_experiments import (
        TRACKS,
        fairness_comparison,
        gang_backfill_comparison,
        preemption_comparison,
        topology_wait_comparison,
        topology_wait_curve,
        track_fixture,
        warm_capacity_comparison,
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
        TRACKS,
        apply_plotly_theme,
        audit_evidence,
        build_lab_report,
        capture_evidence,
        fairness_comparison,
        gang_backfill_comparison,
        get_lab_metadata,
        go,
        ledger,
        mo,
        preemption_comparison,
        report_export_panel,
        topology_wait_comparison,
        topology_wait_curve,
        track_fixture,
        warm_capacity_comparison,
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
def _(TRACKS, track, track_fixture):
    track_id = track.value
    profile = TRACKS[track_id]
    fixture = track_fixture(track_id)
    return fixture, profile, track_id


@app.cell
def _(fixture, mo, track_id):
    _track_key = track_id
    w_cfg = fixture["wait_slider"]
    a_cfg = fixture["arrival_slider"]
    c_cfg = fixture["checkpoint_slider"]
    b_wait = mo.ui.slider(
        w_cfg["min"],
        w_cfg["max"],
        value=w_cfg["value"],
        step=w_cfg["step"],
        label="Wait for local domain (min)",
    )
    b_payload = mo.ui.dropdown(
        {"5 GB": 5, "10 GB": 10, "20 GB": 20},
        value="10 GB",
        label="Bytes exchanged per iteration",
    )
    b_iterations = mo.ui.slider(
        20, 200, value=100, step=20, label="Remaining synchronized iterations"
    )
    c_arrival = mo.ui.slider(
        a_cfg["min"],
        a_cfg["max"],
        value=a_cfg["value"],
        step=a_cfg["step"],
        label="Urgent arrival (min)",
    )
    c_checkpoint = mo.ui.slider(
        c_cfg["min"],
        c_cfg["max"],
        value=c_cfg["value"],
        step=c_cfg["step"],
        label="Checkpoint interval (min)",
    )
    e_reserve = mo.ui.slider(1, 4, value=2, step=1, label="Warm replicas held ready")
    e_demand = mo.ui.slider(0.75, 1.5, value=1.0, step=0.25, label="Demand-ramp scale")
    return b_iterations, b_payload, b_wait, c_arrival, c_checkpoint, e_demand, e_reserve


@app.cell
def _(mo):
    a_reconcile = mo.ui.slider(0, 8, value=3, step=1, label="Reconciliation delay after lease expiry (min)")
    c_resize = mo.ui.dropdown({"Keep 2 slots": 2, "3 slots (invalid increment)": 3, "4 slots": 4, "6 slots": 6, "8 slots": 8}, value="4 slots", label="Requested elastic size")
    c_remaining = mo.ui.slider(1, 60, value=20, step=1, label="Remaining retained target-work units")
    c_transition = mo.ui.slider(0, 15, value=3, step=1, label="Resize transition (min)")
    c_target = mo.ui.checkbox(value=True, label="Same target criterion validated across supported sizes")
    d_replay = mo.ui.slider(0, 24, value=8, step=1, label="Completed steps discarded by replay")
    return a_reconcile, c_resize, c_remaining, c_transition, c_target, d_replay


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    _resource_label = profile["resource_label"]
    a_prediction = mo.ui.radio(
        {
            "Partial hold completes first": "partial",
            "FIFO gang starts the short job first": "fifo",
            "Backfill starts the short job first": "backfill",
        },
        label="Which policy gives the short job its earliest start?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {
            "Wait for the local domain": "wait_for_local",
            "Start across the remote link": "start_remote",
        },
        label="Which option finishes first at the selected wait?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {
            "Urgent wait falls; background loses work": "transfer",
            "Both jobs finish earlier": "both",
            "Checkpoint interval changes urgent start": "urgent_changes",
        },
        label="Who pays for priority preemption?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {
            f"Equal {_resource_label} are fair": "equal_is_fair",
            "Another resource makes them unequal": "dominant_differs",
        },
        label=f"Do equal {_resource_label} counts imply equal resource share?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {
            "Warm reserve lowers shortfall and raises cost": "tradeoff",
            "Warm reserve lowers shortfall and cost": "free",
            "Warm reserve cannot affect the ramp": "no_effect",
        },
        label="What does ready capacity change?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    _resource_label = profile["resource_label"]
    a_decision = mo.ui.radio(
        {"FIFO gang": "fifo_gang", "Backfill": "backfill"},
        label="Policy to carry from Part A",
    )
    b_decision = mo.ui.radio(
        {"Wait for locality": "wait_for_local", "Start remotely": "start_remote"},
        label="Placement decision",
    )
    c_decision = mo.ui.radio(
        {"Keep FIFO": "fifo_gang", "Allow bounded preemption": "priority_preempt"},
        label="Preemption decision",
    )
    d_decision = mo.ui.radio(
        {
            f"Equal {_resource_label} quotas": "equal_accelerators",
            "Dominant-resource fairness": "drf",
        },
        label="Fair-share decision",
    )
    final_choice = mo.ui.radio(
        {
            "Backfill with topology constraints": "topology_backfill",
            "Bounded priority preemption": "bounded_preemption",
            "Keep a warm serving reserve": "warm_reserve",
            "Hold the current fleet policy": "hold",
        },
        label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {
            "Backfill with topology constraints": "topology_backfill",
            "Bounded priority preemption": "bounded_preemption",
            "Keep a warm serving reserve": "warm_reserve",
            "Hold the current policy": "hold",
        },
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {
            "Queue wait exceeds the saved result": "queue_wait",
            "Checkpoint loss exceeds the saved result": "lost_work",
            "Serving shortfall exceeds the saved result": "shortfall",
        },
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {
            "Unseen arrival pattern": "arrival_pattern",
            "Topology changes": "topology_change",
            "Resource demands change": "resource_change",
        },
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Evidence chain",
        placeholder="Name your chosen policy, a saved quantity, the quantified rejected alternative, and what would trigger reevaluation.",
     debounce=False)
    return (
        a_decision,
        b_decision,
        c_decision,
        d_decision,
        final_choice,
        final_rejected,
        final_risk,
        final_trigger,
        rationale,
    )


@app.cell
def _(
    a_reconcile,
    c_resize,
    c_remaining,
    c_transition,
    c_target,
    d_replay,
    b_iterations,
    b_payload,
    b_wait,
    c_arrival,
    c_checkpoint,
    e_demand,
    e_reserve,
    fairness_comparison,
    gang_backfill_comparison,
    preemption_comparison,
    topology_wait_comparison,
    topology_wait_curve,
    track_id,
    warm_capacity_comparison,
):
    a_runs = gang_backfill_comparison(track_id, a_reconcile.value)
    b_runs = topology_wait_comparison(
        track_id, b_wait.value, b_payload.value, b_iterations.value
    )
    b_curve = topology_wait_curve(
        track_id, b_payload.value, b_iterations.value
    )
    c_runs = preemption_comparison(track_id, c_arrival.value, c_checkpoint.value, c_resize.value, c_remaining.value, c_transition.value, c_target.value)
    d_runs = fairness_comparison(track_id, d_replay.value)
    e_runs = warm_capacity_comparison(track_id, e_reserve.value, e_demand.value)
    return a_runs, b_curve, b_runs, c_runs, d_runs, e_runs


@app.cell
def _(
    a_reconcile,
    c_resize,
    c_remaining,
    c_transition,
    c_target,
    d_replay,
    a_decision,
    a_prediction,
    a_runs,
    b_decision,
    b_iterations,
    b_payload,
    b_prediction,
    b_runs,
    b_wait,
    c_arrival,
    c_checkpoint,
    c_decision,
    c_prediction,
    c_runs,
    capture_evidence,
    d_decision,
    d_prediction,
    d_runs,
    e_demand,
    e_prediction,
    e_reserve,
    e_runs,
    mo,
    set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"track_id": track_id, "reconcile_delay_min": a_reconcile.value}
    b_upstream = {
        "track_id": track_id,
        "wait_min": b_wait.value,
        "payload_gb": b_payload.value,
        "iterations": b_iterations.value,
    }
    c_upstream = {
        "track_id": track_id,
        "urgent_arrival_min": c_arrival.value,
        "checkpoint_interval_min": c_checkpoint.value,
        "resize_slots": c_resize.value, "remaining_work": c_remaining.value,
        "transition_min": c_transition.value, "target_validated": c_target.value,
    }
    d_upstream = {"track_id": track_id, "replayed_steps": d_replay.value}
    e_upstream = {
        "track_id": track_id,
        "warm_reserve": e_reserve.value,
        "demand_scale": e_demand.value,
    }

    a_capture = mo.ui.button(
        label="Capture allocation contrast",
        kind="success",
        disabled=a_prediction.value is None or a_decision.value is None,
        on_click=lambda _v: store(
            "A",
            capture_evidence(
                track=track_id,
                part="A",
                prediction=a_prediction.value,
                inputs={"selected_policy": a_decision.value, **a_upstream},
                baseline=a_runs["partial"],
                result=a_runs["backfill"],
                chosen_result=a_runs["fifo"]
                if a_decision.value == "fifo_gang"
                else a_runs["backfill"],
                result_role="rejected alternative"
                if a_decision.value == "fifo_gang"
                else "tested intervention",
                alternatives=(a_runs["fifo"],),
                decision=a_decision.value,
                upstream_inputs=a_upstream,
                model_key="v2_08_experiments",
            ),
        ),
    )
    b_capture = mo.ui.button(
        label="Capture placement crossover",
        kind="success",
        disabled=b_prediction.value is None or b_decision.value is None,
        on_click=lambda _v: store(
            "B",
            capture_evidence(
                track=track_id,
                part="B",
                prediction=b_prediction.value,
                inputs={"decision": b_decision.value, **b_upstream},
                baseline=b_runs["baseline"],
                result=b_runs["result"],
                chosen_result=b_runs["result"]
                if b_decision.value == "wait_for_local"
                else b_runs["baseline"],
                result_role="tested intervention"
                if b_decision.value == "wait_for_local"
                else "rejected alternative",
                alternatives=(b_runs["baseline"],),
                decision=b_decision.value,
                upstream_inputs=b_upstream,
                model_key="v2_08_experiments",
            ),
        ),
    )
    c_capture = mo.ui.button(
        label="Capture preemption consequence",
        kind="success",
        disabled=c_prediction.value is None or c_decision.value is None,
        on_click=lambda _v: store(
            "C",
            capture_evidence(
                track=track_id,
                part="C",
                prediction=c_prediction.value,
                inputs={"decision": c_decision.value, **c_upstream},
                baseline=c_runs["baseline"],
                result=c_runs["result"],
                chosen_result=c_runs["baseline"]
                if c_decision.value == "fifo_gang"
                else c_runs["result"],
                result_role="rejected alternative"
                if c_decision.value == "fifo_gang"
                else "tested intervention",
                decision=c_decision.value,
                upstream_inputs=c_upstream,
                model_key="v2_08_experiments",
            ),
        ),
    )
    d_capture = mo.ui.button(
        label="Capture fairness comparison",
        kind="success",
        disabled=d_prediction.value is None or d_decision.value is None,
        on_click=lambda _v: store(
            "D",
            capture_evidence(
                track=track_id,
                part="D",
                prediction=d_prediction.value,
                inputs={"decision": d_decision.value, **d_upstream},
                baseline=d_runs["baseline"],
                result=d_runs["result"],
                chosen_result=d_runs["baseline"]
                if d_decision.value == "equal_accelerators"
                else d_runs["result"],
                result_role="rejected alternative"
                if d_decision.value == "equal_accelerators"
                else "tested intervention",
                decision=d_decision.value,
                upstream_inputs=d_upstream,
                model_key="v2_08_experiments",
            ),
        ),
    )
    e_capture = mo.ui.button(
        label="Capture capacity tradeoff",
        kind="success",
        disabled=e_prediction.value is None,
        on_click=lambda _v: store(
            "E",
            capture_evidence(
                track=track_id,
                part="E",
                prediction=e_prediction.value,
                inputs=e_upstream,
                baseline=e_runs["baseline"],
                result=e_runs["result"],
                chosen_result=e_runs["result"],
                result_role="tested intervention",
                alternatives=(e_runs["baseline"],),
                decision=e_reserve.value,
                upstream_inputs=e_upstream,
                model_key="v2_08_experiments",
            ),
        ),
    )
    return (
        a_capture,
        a_upstream,
        b_capture,
        b_upstream,
        c_capture,
        c_upstream,
        d_capture,
        d_upstream,
        e_capture,
        e_upstream,
    )


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    from mlsysbook_labs.notebook import (
        notebook_header as _course_header, notebook_shell as _course_shell,
        notebook_learning_map as _course_learning_map,
        notebook_scenario as _course_scenario,
    )
    from mlsysbook_labs.learning_paths import get_learning_path as _course_path
    from mlsysbook_labs import get_lab_metadata as _course_metadata
    css = """
    <style>




    .fleet-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}.lab-hud .hud-error{color:#fecaca}

    </style>"""
    header = _course_header(_course_metadata("vol2/lab_08_fleet_orch.py"), 'When should a scheduler leave capacity idle, delay a job, or interrupt useful work?')
    _course_shell(
        [
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        header,
        track,
        _course_scenario(_course_path('vol2', 8), track=track.value, volume=2),
        mo.Html(
                '<p class="fleet-note">Illustrative deterministic fleet trace, not a production measurement. Each saved contrast preserves its original settings and result.</p>'
            ),
        _course_learning_map(_course_path('vol2', 8))
    ],
        gap=0.5,
     volume=2)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    a_reconcile,
    c_resize,
    c_remaining,
    c_transition,
    c_target,
    d_replay,
    COLORS,
    a_capture,
    a_decision,
    a_prediction,
    a_runs,
    a_upstream,
    apply_plotly_theme,
    audit_evidence,
    b_capture,
    b_curve,
    b_decision,
    b_iterations,
    b_payload,
    b_prediction,
    b_runs,
    b_upstream,
    b_wait,
    c_arrival,
    c_capture,
    c_checkpoint,
    c_decision,
    c_prediction,
    c_runs,
    c_upstream,
    d_capture,
    d_decision,
    d_prediction,
    d_runs,
    d_upstream,
    e_capture,
    e_demand,
    e_prediction,
    e_reserve,
    e_runs,
    e_upstream,
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
):
    _saved_prediction_widgets = {'A': a_prediction, 'B': b_prediction, 'C': c_prediction, 'D': d_prediction, 'E': e_prediction}
    from mlsysbook_labs.notebook import notebook_prediction_label as _read_prediction
    from mlsysbook_labs.notebook import notebook_tab_hint as _part_tabs_hint, notebook_shell as _lesson_shell
    from mlsysbook_labs.notebook import notebook_part_reflection as _concept_reflection
    from mlsysbook_labs.learning_paths import get_learning_path as _reflection_path
    _reflection_contract = _reflection_path('vol2', 8)
    _captures = get_evidence()
    upstream = {
        "A": a_upstream,
        "B": b_upstream,
        "C": c_upstream,
        "D": d_upstream,
        "E": e_upstream,
    }
    audit = audit_evidence(
        _captures,
        track=track_id,
        required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=upstream,
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
                    "**STALE OR NON-CONTRASTING EVIDENCE.** Settings changed, or both saved runs used identical evaluator inputs. Recapture this part."
                ),
                kind="danger",
            )
        data = cap.to_dict()
        return mo.Html(
            f'<div class="saved"><b>Saved snapshot</b> · original prediction: {_read_prediction(_saved_prediction_widgets[part], value=data["prediction"])}<br><small>Track {data["track"]}; later control changes cannot rewrite this result.</small></div>'
        )

    def part_a():
        preface = mo.md(
            f"### A · Why can jobs wait while {profile['resource_label']} are occupied? (11 min)\nTwo rigid jobs each request more than half of the same bounded fleet. Compare partial hold with atomic gangs, then inspect whether a short job can pass a blocked queue head."
        )
        if a_prediction.value is None:
            return mo.vstack([preface, a_prediction])
        rows = [
            {
                "Policy": "Partial hold",
                "Short-job start": "never",
                "Completed jobs": 0,
                "Observed state": "DEADLOCK"
                if a_runs["partial"]["deadlocked"]
                else "progress",
            },
            {
                "Policy": "FIFO gang",
                "Short-job start": f"{a_runs['fifo']['jobs']['short']['start_min']:.1f} min",
                "Completed jobs": a_runs["fifo"]["completed_jobs"],
                "Observed state": "atomic allocation",
            },
            {
                "Policy": "Backfill",
                "Short-job start": f"{a_runs['backfill']['jobs']['short']['start_min']:.1f} min",
                "Completed jobs": a_runs["backfill"]["completed_jobs"],
                "Observed state": "backfill event recorded",
            },
        ]
        return mo.vstack([mo.vstack(
            [
                preface,
                a_prediction,
                table(rows),
                a_decision,
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {_read_prediction(a_prediction)}. **Observed:** partial allocation holds **{a_runs['partial']['idle_held_accelerators']} idle slots** with zero useful work. Backfill starts the short job at **{a_runs['backfill']['jobs']['short']['start_min']:.1f} min**, versus **{a_runs['fifo']['jobs']['short']['start_min']:.1f} min** under FIFO gang scheduling."
                    ),
                    kind="danger" if a_runs["partial"]["deadlocked"] else "info",
                ),
                mo.md("#### A2 · Admission is not proof of exclusive ownership\nA replacement bundle fits, but the previous owner is unobserved. The same separate protocol trace compares immediate reuse against lease expiry followed by an epoch fence. A lease timeout cannot stop a partitioned process."),
                a_reconcile,
                table([{"Policy": label, "Replacement starts (min)": run["replacement_start_min"], "Late old write accepted": run["late_write_accepted"], "Exclusive effect preserved": run["exclusive_effect_preserved"]} for label, run in [("Reuse on lost observation", a_runs["partial"]["control_plane"]), ("Lease and fence", a_runs["backfill"]["control_plane"])]]),
                table(a_runs["backfill"]["control_plane"]["events"]),
                mo.md("**Explain:** Which phase admits the job, which selects its bundle, and which repairs observed state? Increase reconciliation delay: which safety property stays fixed, and what availability cost grows? The fence is enforced by the target of writes, not by trust that the old process stopped."),
                a_capture,
                saved("A"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "The event simulator allocates each rigid gang atomically. Partial hold distributes slots round-robin but a rigid job makes no progress until its full bundle is present. Backfill records a start event only when the later job's complete bundle fits."
                        )
                    }
                ),
            ]
        ), _concept_reflection(_reflection_contract, "A")])

    def part_b():
        preface = mo.md(
            f"### B · When should a job wait for better placement? (10 min)\nThe job can start now across a remote link or wait for one {profile['topology_label']}. Predict which complete path is shorter before revealing the crossover."
        )
        controls = mo.hstack(
            [b_wait, b_payload, b_iterations], widths="equal", wrap=True
        )
        if b_prediction.value is None:
            return mo.vstack([preface, controls, b_prediction])
        fig = go.Figure()
        fig.add_scatter(
            x=[point["inputs"]["wait_min"] for point in b_curve],
            y=[point["result"]["wait_local_total_min"] for point in b_curve],
            mode="lines+markers",
            name="Wait for local domain",
            line=dict(color=COLORS["BlueLine"]),
        )
        fig.add_scatter(
            x=[point["inputs"]["wait_min"] for point in b_curve],
            y=[point["result"]["start_remote_total_min"] for point in b_curve],
            mode="lines",
            name="Start remote now",
            line=dict(color=COLORS["OrangeLine"], dash="dash"),
        )
        fig.update_layout(
            height=285,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Local-domain wait (min)",
            yaxis_title="Time to finish (min)",
            legend_orientation="h",
        )
        result = b_runs["result"]
        rows = [
            {
                "Option": "Wait for local",
                "Complete path": f"{result['wait_local_total_min']:.2f} min",
                "Communication": f"{result['local_communication_min']:.2f} min",
            },
            {
                "Option": "Start remote",
                "Complete path": f"{result['start_remote_total_min']:.2f} min",
                "Communication": f"{result['remote_communication_min']:.2f} min",
            },
        ]
        return mo.vstack([mo.vstack(
            [
                preface,
                controls,
                b_prediction,
                apply_plotly_theme(fig),
                table(rows),
                b_decision,
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {_read_prediction(b_prediction)}. **Observed boundary:** waiting wins only below **{result['crossover_wait_min']:.2f} min** for these bytes, iterations, and links. The selected case favors **{result['decision'].replace('_', ' ')}**."
                    ),
                    kind="info",
                ),
                b_capture,
                saved("B"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Complete time equals compute time plus communication time, with local waiting added only to the local option. Communication time uses exchanged bytes divided by link bandwidth for every remaining iteration. The bandwidths are explicit illustrative fleet assumptions."
                        )
                    }
                ),
            ]
        ), _concept_reflection(_reflection_contract, "B")])

    def part_c():
        preface = mo.md(
            f"### C · Who pays for preemption? (12 min)\nAn urgent {profile['urgent_work']} arrives while {profile['background_work']} occupies the fleet. Compare FIFO and priority preemption from the identical arrival trace."
        )
        controls = mo.hstack([c_arrival, c_checkpoint], widths="equal", wrap=True)
        if c_prediction.value is None:
            return mo.vstack([preface, controls, c_prediction])
        fifo = c_runs["baseline"]
        preempt = c_runs["result"]
        rows = [
            {
                "Policy": "FIFO",
                "Urgent wait": f"{fifo['jobs']['urgent']['wait_min']:.1f} min",
                "Background finish": f"{fifo['jobs']['background']['finish_min']:.1f} min",
                "Lost work": f"{fifo['jobs']['background']['lost_work_min']:.1f} min",
            },
            {
                "Policy": "Priority preemption",
                "Urgent wait": f"{preempt['jobs']['urgent']['wait_min']:.1f} min",
                "Background finish": f"{preempt['jobs']['background']['finish_min']:.1f} min",
                "Lost work": f"{preempt['jobs']['background']['lost_work_min']:.1f} min",
            },
        ]
        return mo.vstack([mo.vstack(
            [
                preface,
                controls,
                c_prediction,
                table(rows),
                c_decision,
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {_read_prediction(c_prediction)}. **Observed transfer:** urgent wait changes from **{fifo['jobs']['urgent']['wait_min']:.1f} to {preempt['jobs']['urgent']['wait_min']:.1f} min**. The interrupted job loses **{preempt['jobs']['background']['lost_work_min']:.1f} min** since its last checkpoint and finishes later."
                    ),
                    kind="info",
                ),
                mo.md("#### C2 · Resize instead of interrupt?\nUse a separate coupled backend-job fixture with a declared size range, two-slot increments, and a supplied diminishing-return retained-work curve. This is a backend training/reference workload for every track; it does not resize an individual device. The fixed target is meaningful only when the same quality criterion remains validated."),
                mo.hstack([c_resize, c_remaining, c_transition], wrap=True),
                c_target,
                table([{"Supported slots": size, "Retained units/min": rate} for size, rate in preempt["elastic"]["contract"]["rate_curve"].items()]),
                mo.callout(mo.md(
                    (f"Keep: **{preempt['elastic']['keep_time_min']:.2f} min**. Resize: **{preempt['elastic']['resize_time_min']:.2f} min**. Decision: **{preempt['elastic']['decision']}**."
                     + (f" Transition is repaid above **{preempt['elastic']['amortization_work']:.2f} retained-work units**." if preempt['elastic']['amortization_work'] is not None else " No speedup at the current size."))
                    if preempt["elastic"]["eligible"] else f"**Resize unavailable:** {preempt['elastic']['reason']}. A faster nominal step rate cannot establish time to the same target."), kind="info" if preempt["elastic"]["eligible"] else "danger"),
                table(preempt["elastic"]["events"]) if preempt["elastic"]["events"] else mo.md("No valid transition is executed."),
                mo.md("**Test a reversal:** shorten remaining work or increase transition time until keeping the allocation wins. Then request an invalid increment or remove target validation. Explain why those are contract failures rather than slow valid executions. The curve is supplied synthetic evidence, not a convergence model."),
                c_capture,
                saved("C"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            f"At preemption, progress since the last completed checkpoint is replayed. Lost {profile['resource_label']}-minutes equal that lost time multiplied by the interrupted allocation. Priority changes allocation events; it does not shorten either job's required useful work."
                        )
                    }
                ),
            ]
        ), _concept_reflection(_reflection_contract, "C")])

    def part_d():
        _resource_label = profile["resource_label"]
        preface = mo.md(
            f"### D · Is equal {_resource_label} allocation fair? (11 min)\nTwo tenants receive equal {_resource_label} counts but consume different CPU, memory, and network shares. Predict whether the dominant shares remain equal."
        )
        if d_prediction.value is None:
            return mo.vstack([preface, d_prediction])
        equal = d_runs["baseline"]["tenants"]
        drf = d_runs["result"]["tenants"]
        fig = go.Figure()
        for resource, color in (
            ("accelerators", COLORS["BlueLine"]),
            ("cpu_cores", COLORS["OrangeLine"]),
            ("memory_gb", COLORS["GreenLine"]),
            ("network_gbps", COLORS["RedLine"]),
        ):
            series_name = (
                _resource_label.title()
                if resource == "accelerators"
                else resource.replace("_", " ").title()
            )
            fig.add_bar(
                name=series_name,
                x=list(equal),
                y=[equal[tenant]["share_pct"][resource] for tenant in equal],
                marker_color=color,
            )
        fig.update_layout(
            barmode="group",
            height=285,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Fleet resource share (%)",
            legend_orientation="h",
        )
        rows = [
            {
                "Allocation": f"Equal {_resource_label}",
                "Compute-heavy dominant share": f"{equal['compute-heavy']['dominant_share_pct']:.1f}%",
                "Network-heavy dominant share": f"{equal['network-heavy']['dominant_share_pct']:.1f}%",
                "Gap": f"{d_runs['baseline']['dominant_share_gap_pct']:.1f} points",
            },
            {
                "Allocation": "DRF progressive filling",
                "Compute-heavy dominant share": f"{drf['compute-heavy']['dominant_share_pct']:.1f}%",
                "Network-heavy dominant share": f"{drf['network-heavy']['dominant_share_pct']:.1f}%",
                "Gap": f"{d_runs['result']['dominant_share_gap_pct']:.1f} points",
            },
        ]
        return mo.vstack([mo.vstack(
            [
                preface,
                d_prediction,
                apply_plotly_theme(fig),
                table(rows),
                d_decision,
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {_read_prediction(d_prediction)}. **Observed:** equal {_resource_label} counts leave a **{d_runs['baseline']['dominant_share_gap_pct']:.1f}-point dominant-share gap**. Progressive filling grants {d_runs['result']['bundle_counts']['compute-heavy']} and {d_runs['result']['bundle_counts']['network-heavy']} feasible bundles and reduces the gap to **{d_runs['result']['dominant_share_gap_pct']:.1f} points**."
                    ),
                    kind="info",
                ),
                mo.md(f"#### D2 · Which utilization is this?\nA separate {d_runs['result']['utilization_trace']['duration_min']:g}-minute runtime record holds {d_runs['result']['utilization_trace']['allocated_slot_min'] / d_runs['result']['utilization_trace']['duration_min']:g} slots in a complete allocation and {d_runs['result']['utilization_trace']['incomplete_held_slots']} in an incomplete gang. Its activity counter records {d_runs['result']['utilization_trace']['active_slot_min']:g} slot-minutes, including communication and replay. It completes {d_runs['result']['utilization_trace']['completed_steps']} steps; discarded steps do not survive as target progress. The fault-free baseline is {d_runs['result']['utilization_trace']['fault_free_steps_per_slot_min']:g} retained steps per slot-minute. This trace is independent of the fairness intervention above."),
                d_replay,
                table([{"Tier": label, "Slot-minutes": d_runs["result"]["utilization_trace"][key], "Fleet capacity (%)": d_runs["result"]["utilization_trace"][pct]} for label, key, pct in [("Allocated: complete groups only", "allocated_slot_min", "allocated_fleet_pct"), ("Active: declared activity counter", "active_slot_min", "active_fleet_pct"), ("Productive: retained / fault-free rate", "productive_slot_min", "productive_fleet_pct")]]),
                mo.md(f"**Retained steps:** {d_runs['result']['utilization_trace']['retained_steps']}. Active and productive capacity share the same fleet-time denominator above. As fractions of the complete allocation they are **{d_runs['result']['utilization_trace']['active_of_allocated_pct']:.1f}%** and **{d_runs['result']['utilization_trace']['productive_of_allocated_pct']:.1f}%**. {d_runs['result']['utilization_trace']['incomplete_held_slots']} incomplete held slots count in none of these allocated groups."),
                mo.md("**Explain:** increase discarded steps while keeping the activity counter fixed. Which dashboard moves? Does the unchanged active counter prove useful progress? Compare the retained-work evidence before recommending more hardware; this counterexample does not identify a production root cause."),
                d_capture,
                saved("D"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "For each tenant, dominant share is the largest allocated fraction of any fleet resource. Progressive filling repeatedly grants the feasible tenant with the lowest current dominant share; it does not manufacture a fairness score."
                        )
                    }
                ),
            ]
        ), _concept_reflection(_reflection_contract, "D")])

    def part_e():
        preface = mo.md(
            "### E · How much serving capacity should stay ready? (8 min)\nReplay one demand ramp. A cold policy pays model-loading and warmup delay; a reserve pays for ready capacity even while idle."
        )
        controls = mo.hstack([e_reserve, e_demand], widths="equal", wrap=True)
        if e_prediction.value is None:
            return mo.vstack([preface, controls, e_prediction])
        cold = e_runs["baseline"]
        warm = e_runs["result"]
        fig = go.Figure()
        fig.add_scatter(
            x=[row["time_min"] for row in warm["timeline"]],
            y=[row["demand_per_min"] for row in warm["timeline"]],
            mode="lines",
            name="Demand",
            line=dict(color=COLORS["OrangeLine"]),
        )
        fig.add_scatter(
            x=[row["time_min"] for row in cold["timeline"]],
            y=[row["served_per_min"] for row in cold["timeline"]],
            mode="lines",
            name="Cold policy served",
            line=dict(color=COLORS["RedLine"], dash="dot"),
        )
        fig.add_scatter(
            x=[row["time_min"] for row in warm["timeline"]],
            y=[row["served_per_min"] for row in warm["timeline"]],
            mode="lines",
            name="Warm reserve served",
            line=dict(color=COLORS["GreenLine"]),
        )
        fig.update_layout(
            height=285,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Trace time (min)",
            yaxis_title="Requests per minute",
            legend_orientation="h",
        )
        rows = [
            {
                "Policy": "Cold",
                "Missed requests": f"{cold['missed_requests']:.1f}",
                "Idle replica-min": f"{cold['idle_replica_minutes']:.1f}",
                "Charged cost": f"${cold['cost_usd']:.2f}",
            },
            {
                "Policy": f"Reserve {e_reserve.value}",
                "Missed requests": f"{warm['missed_requests']:.1f}",
                "Idle replica-min": f"{warm['idle_replica_minutes']:.1f}",
                "Charged cost": f"${warm['cost_usd']:.2f}",
            },
        ]
        return mo.vstack([mo.vstack(
            [
                preface,
                controls,
                e_prediction,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {_read_prediction(e_prediction)}. **Observed tradeoff:** the reserve changes missed requests from **{cold['missed_requests']:.1f} to {warm['missed_requests']:.1f}**, while charged cost changes from **${cold['cost_usd']:.2f} to ${warm['cost_usd']:.2f}**. Loading plus warmup remains **{warm['readiness_delay_min']} min** for newly requested replicas."
                    ),
                    kind="info",
                ),
                e_capture,
                saved("E"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Each minute, demand is compared with ready replicas times per-replica capacity. New replicas enter a loading state and become ready only after loading plus warmup. Charged replica-minutes include ready and loading capacity; the reserve can reduce shortfall only by consuming idle time and cost."
                        )
                    }
                ),
            ]
        ), _concept_reflection(_reflection_contract, "E")])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            state = "MISSING"
            if capture:
                state = (
                    "STALE"
                    if part in audit.stale or (part, part) in audit.identical_pairs
                    else "CURRENT"
                )
            rows.append(
                {
                    "Part": part,
                    "Original prediction": _read_prediction(_saved_prediction_widgets[part], value=(capture.to_dict()["prediction"]
                    if capture
                    else "—")),
                    "Evidence": state,
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
                    "### Synthesis · Defend one fleet policy (5 min)\nUse the saved experiments to name a chosen option, a quantified rejected alternative, the remaining limitation, and the condition that would make you reevaluate."
                ),
                table(rows),
                mo.callout(
                    mo.md(
                        "Saved snapshots preserve predictions, evaluator arguments, and results. Recapture any stale experiment before producing the report."
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
                        else "Complete five current contrasts, choose distinct recommended and rejected options, and write the evidence chain."
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
            "Synthesis": build_synthesis(),
        }
    )
    _lesson_shell([_part_tabs_hint(), tabs], volume=2)
    return (audit,)


@app.cell
def _():
    from mlsysbook_labs.notebook import notebook_shell as _submit_shell, notebook_submission as _submit_guide
    from mlsysbook_labs.learning_paths import get_learning_path as _submit_path
    _submit_shell([_submit_guide(_submit_path(2, 8))], volume=2)
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
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_08_fleet_orch.py"),
        track=track_id,
        scenario=profile["fleet"],
        learning_objectives=[
            "Diagnose gang blocking and backfill from allocation events",
            "Compare topology delay with communication time",
            "Quantify preemption, dominant-share, and warm-capacity consequences",
        ],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={
            part: {
                "baseline": snapshots[part]["baseline"],
                "result": snapshots[part]["result"],
                "chosen_result": snapshots[part]["chosen_result"],
                "result_role": snapshots[part]["result_role"],
                "alternatives": snapshots[part]["alternatives"],
            }
            for part in "ABCDE"
        },
        binding_constraints={
            "A": "gang feasibility",
            "B": "topology wait crossover",
            "C": "checkpoint loss",
            "D": "dominant resource",
            "E": "readiness delay",
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
            "Atomic gangs prevent hold-and-wait deadlock.",
            "Placement delay can cost less than remote communication.",
            "Urgent service, fairness, and ready capacity each transfer cost elsewhere.",
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
            "rejected": final_rejected.value,
            "trigger": final_trigger.value,
            "residual_risk": final_risk.value,
        },
        source_trace={
            "scenario": "Illustrative explicit job, topology, resource, and demand records.",
            "calculations": "Deterministic MLSysIM event schedules and physical-unit calculations.",
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
    fleet_discussion_design = {'schema_version': 1, 'lab_id': 'v2_08', 'track_id': track_id, 'model_id': 'v2_08_experiments', 'evidence': {part: capture.to_dict() for part, capture in _captures.items()}, 'recommendation': final_choice.value, 'rejected_alternative': final_rejected.value, 'reevaluation_trigger': final_trigger.value, 'residual_risk': final_risk.value, 'rationale': rationale.value} if _ready else None
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
        ledger, chapter=8, track=track_id, design=fleet_discussion_design,
        fingerprint=fleet_discussion_fingerprint, ready=fleet_discussion_ready,
        requested=bool(save_discussion.value),
    )
    _discussion_shell([mo.vstack([save_discussion, _notebook_status(_outcome.status, _outcome.detail)])], volume=2)
    return


if __name__ == "__main__":
    app.run()
