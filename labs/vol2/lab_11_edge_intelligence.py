import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 11: Fleet Adaptation · MLSysBook")


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
    from mlsysim.engine.v2_11_experiments import (
        MODEL_ID,
        TRACKS,
        compare_architectures,
        evaluate_adaptation,
        evaluate_architecture,
        evaluate_client_selection,
        evaluate_federation,
        evaluate_replay,
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
        MODEL_ID,
        TRACKS,
        apply_plotly_theme,
        audit_evidence,
        build_lab_report,
        capture_evidence,
        compare_architectures,
        evaluate_adaptation,
        evaluate_architecture,
        evaluate_client_selection,
        evaluate_federation,
        evaluate_replay,
        get_lab_metadata,
        go,
        ledger,
        mo,
        report_export_panel,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="Mobile",
        label="Fleet track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(TRACKS, track):
    track_id = track.value
    profile = TRACKS[track_id]
    return profile, track_id


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {
            "Central upload": "central",
            "Device-local update": "local",
            "Federated update": "federated",
        },
        label="Which shares population learning while keeping raw observations local?",
    ).form(submit_button_label="Lock Part A prediction")
    a_epochs = mo.ui.dropdown(
        {"1 epoch": 1, "2 epochs": 2, "4 epochs": 4},
        value="2 epochs",
        label="Local work per round",
    )
    a_choice = mo.ui.radio(
        {
            "Central": "central",
            "Local only": "local",
            "Federated": "federated",
            "Hold current design": "none",
        },
        label="Architecture to carry forward",
    )
    a_rejected = mo.ui.radio(
        {"Central": "central", "Local only": "local", "Federated": "federated"},
        label="Tested rejected architecture",
    )
    b_prediction = mo.ui.radio(
        {
            "Memory": "memory",
            "Energy": "energy",
            "Update window": "window",
            "Foreground latency": "foreground",
        },
        label="Which check is most likely to reject a full update?",
    ).form(submit_button_label="Lock Part B prediction")
    b_method = mo.ui.dropdown(
        {"Adapter update": "adapter", "Bias-only update": "bias"},
        value="Adapter update",
        label="Reduced update method",
    )
    b_batch = mo.ui.slider(1, 8, value=1, step=1, label="Concurrent examples")
    b_foreground = mo.ui.checkbox(value=True, label="Run with foreground work")
    c_prediction = mo.ui.radio(
        {"No replay": 0.0, "25% replay": 0.25, "50% replay": 0.5, "75% replay": 0.75},
        label="Which allocation best protects the weaker old/new outcome?",
    ).form(submit_button_label="Lock Part C prediction")
    c_fraction = mo.ui.dropdown(
        {"25%": 0.25, "50%": 0.5, "75%": 0.75}, value="50%", label="Memory for replay"
    )
    d_prediction = mo.ui.radio(
        {"1 epoch": 1, "2 epochs": 2, "4 epochs": 4, "8 epochs": 8},
        label="Under high heterogeneity, which sends the fewest bytes to the target?",
    ).form(submit_button_label="Lock Part D prediction")
    d_epochs = mo.ui.dropdown(
        {"2 epochs": 2, "4 epochs": 4, "8 epochs": 8},
        value="4 epochs",
        label="Local epochs",
    )
    d_heterogeneity = mo.ui.dropdown(
        {"Low": "low", "Moderate": "moderate", "High": "high"},
        value="High",
        label="Client heterogeneity",
    )
    e_prediction = mo.ui.radio(
        {"Freshest clients": "freshest", "Cohort coverage": "coverage"},
        label="Which policy represents more cohorts at the same client limit?",
    ).form(submit_button_label="Lock Part E prediction")
    e_policy = mo.ui.radio(
        {"Freshest first": "freshest", "Coverage first": "coverage"},
        label="Selection policy",
    )
    e_freshness = mo.ui.slider(
        4, 24, value=12, step=2, label="Maximum evidence age (hours)"
    )
    _charging_disabled = profile.mains_powered
    e_charging = mo.ui.checkbox(
        value=False,
        label="Require charging" if not _charging_disabled else "Require charging (inapplicable: mains-powered)",
        disabled=_charging_disabled,
    )
    e_clients = mo.ui.slider(2, 6, value=4, step=1, label="Maximum clients")
    return (
        a_choice,
        a_epochs,
        a_prediction,
        a_rejected,
        b_batch,
        b_foreground,
        b_method,
        b_prediction,
        c_fraction,
        c_prediction,
        d_epochs,
        d_heterogeneity,
        d_prediction,
        e_charging,
        e_clients,
        e_freshness,
        e_policy,
        e_prediction,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {
            "Central adaptation": "central",
            "Local-only adaptation": "local",
            "Federated adaptation": "federated",
            "Hold current design": "none",
        },
        label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Central": "central", "Local only": "local", "Federated": "federated"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {
            "Foreground delay exceeds budget": "foreground_delay",
            "Replay no longer fits": "replay_capacity",
            "Cohort coverage falls": "cohort_coverage",
            "Rounds increase": "convergence",
        },
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {
            "Illustrative outcome evidence": "outcome_fixture",
            "Correlated availability": "availability",
            "Stale client evidence": "freshness",
        },
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Cite one saved result, quantify a rejected alternative, and explain the trigger.",
    )
    student_id = mo.ui.text(label="Student ID (optional)")
    return (
        final_choice,
        final_rejected,
        final_risk,
        final_trigger,
        rationale,
        student_id,
    )


@app.cell
def _(
    a_choice,
    a_epochs,
    a_rejected,
    b_batch,
    b_foreground,
    b_method,
    c_fraction,
    compare_architectures,
    d_epochs,
    d_heterogeneity,
    e_charging,
    e_clients,
    e_freshness,
    e_policy,
    evaluate_adaptation,
    evaluate_architecture,
    evaluate_client_selection,
    evaluate_federation,
    evaluate_replay,
    track_id,
):
    a_results = {
        row["architecture"]: row
        for row in compare_architectures(track_id, local_epochs=a_epochs.value)
    }
    a_choice_key = a_choice.value or "federated"
    if a_choice_key == "none":
        a_selected = a_results[a_rejected.value or "federated"]
        a_baseline = evaluate_architecture(
            track_id, "none", local_epochs=a_epochs.value
        )
        a_chosen_result = a_baseline
        a_result_role = "rejected alternative"
    else:
        a_selected = a_results[a_choice_key]
        a_baseline_key = "federated" if a_choice_key == "central" else "central"
        a_baseline = evaluate_architecture(
            track_id, a_baseline_key, local_epochs=a_epochs.value
        )
        a_chosen_result = None
        a_result_role = "tested intervention"
    b_baseline = evaluate_adaptation(
        track_id,
        "full",
        batch_size=b_batch.value,
        concurrent_with_foreground=b_foreground.value,
    )
    b_result = evaluate_adaptation(
        track_id,
        b_method.value,
        batch_size=b_batch.value,
        concurrent_with_foreground=b_foreground.value,
    )
    c_results = {
        fraction: evaluate_replay(track_id, b_method.value, fraction)
        for fraction in (0.0, 0.25, 0.5, 0.75)
    }
    c_baseline, c_result = c_results[0.0], c_results[c_fraction.value]
    d_results = {
        epochs: evaluate_federation(
            track_id, epochs, d_heterogeneity.value, method=b_method.value
        )
        for epochs in (1, 2, 4, 8)
    }
    d_baseline, d_result = d_results[1], d_results[d_epochs.value]
    e_common = dict(
        track_id=track_id,
        max_evidence_age_hours=e_freshness.value,
        require_charging=e_charging.value,
        max_clients=e_clients.value,
    )
    e_results = {
        policy: evaluate_client_selection(**e_common, selection_policy=policy)
        for policy in ("freshest", "coverage")
    }
    e_selected_key = e_policy.value or "coverage"
    e_result = e_results[e_selected_key]
    e_baseline = e_results["coverage" if e_selected_key == "freshest" else "freshest"]
    return (
        a_baseline,
        a_chosen_result,
        a_result_role,
        a_results,
        a_selected,
        b_baseline,
        b_result,
        c_baseline,
        c_result,
        c_results,
        d_baseline,
        d_result,
        d_results,
        e_baseline,
        e_result,
        e_results,
    )


@app.cell
def _(
    a_baseline,
    a_chosen_result,
    a_choice,
    a_epochs,
    a_prediction,
    a_rejected,
    a_result_role,
    a_results,
    a_selected,
    b_baseline,
    b_batch,
    b_foreground,
    b_method,
    b_prediction,
    b_result,
    c_baseline,
    c_fraction,
    c_prediction,
    c_result,
    c_results,
    capture_evidence,
    d_baseline,
    d_epochs,
    d_heterogeneity,
    d_prediction,
    d_result,
    d_results,
    e_baseline,
    e_charging,
    e_clients,
    e_freshness,
    e_policy,
    e_prediction,
    e_result,
    e_results,
    mo,
    set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {
        "local_epochs": a_epochs.value,
        "choice": a_choice.value,
        "rejected": a_rejected.value,
    }
    b_upstream = {
        "method": b_method.value,
        "batch_size": b_batch.value,
        "foreground": b_foreground.value,
    }
    c_upstream = {"method": b_method.value, "replay_fraction": c_fraction.value}
    d_upstream = {
        "method": b_method.value,
        "local_epochs": d_epochs.value,
        "heterogeneity": d_heterogeneity.value,
    }
    e_upstream = {
        "policy": e_policy.value,
        "freshness_hours": e_freshness.value,
        "require_charging": e_charging.value,
        "max_clients": e_clients.value,
    }
    a_capture = mo.ui.button(
        label="Capture architecture contrast",
        kind="success",
        disabled=(
            a_prediction.value is None
            or a_choice.value is None
            or a_rejected.value is None
            or a_choice.value == a_rejected.value
        ),
        on_click=lambda _v: store(
            "A",
            capture_evidence(
                track=track_id,
                part="A",
                prediction=a_prediction.value,
                inputs=a_upstream,
                baseline=a_baseline,
                result=a_selected,
                alternatives=tuple(a_results.values()),
                decision=a_choice.value,
                chosen_result=a_chosen_result,
                result_role=a_result_role,
                upstream_inputs=a_upstream,
                model_key=a_selected["model_key"],
            ),
        ),
    )
    b_capture = mo.ui.button(
        label="Capture admission contrast",
        kind="success",
        disabled=b_prediction.value is None,
        on_click=lambda _v: store(
            "B",
            capture_evidence(
                track=track_id,
                part="B",
                prediction=b_prediction.value,
                inputs=b_upstream,
                baseline=b_baseline,
                result=b_result,
                alternatives=(b_baseline, b_result),
                decision=b_method.value,
                upstream_inputs=b_upstream,
                model_key=b_result["model_key"],
            ),
        ),
    )
    c_capture = mo.ui.button(
        label="Capture replay contrast",
        kind="success",
        disabled=c_prediction.value is None,
        on_click=lambda _v: store(
            "C",
            capture_evidence(
                track=track_id,
                part="C",
                prediction=c_prediction.value,
                inputs=c_upstream,
                baseline=c_baseline,
                result=c_result,
                alternatives=tuple(c_results.values()),
                decision=c_fraction.value,
                upstream_inputs=c_upstream,
                model_key=c_result["model_key"],
            ),
        ),
    )
    d_capture = mo.ui.button(
        label="Capture convergence contrast",
        kind="success",
        disabled=d_prediction.value is None,
        on_click=lambda _v: store(
            "D",
            capture_evidence(
                track=track_id,
                part="D",
                prediction=d_prediction.value,
                inputs=d_upstream,
                baseline=d_baseline,
                result=d_result,
                alternatives=tuple(d_results.values()),
                decision=d_epochs.value,
                upstream_inputs=d_upstream,
                model_key=d_result["model_key"],
            ),
        ),
    )
    e_capture = mo.ui.button(
        label="Capture participation contrast",
        kind="success",
        disabled=e_prediction.value is None or e_policy.value is None,
        on_click=lambda _v: store(
            "E",
            capture_evidence(
                track=track_id,
                part="E",
                prediction=e_prediction.value,
                inputs=e_upstream,
                baseline=e_baseline,
                result=e_result,
                alternatives=tuple(e_results.values()),
                decision=e_policy.value,
                upstream_inputs=e_upstream,
                model_key=e_result["model_key"],
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


@app.cell(hide_code=True)
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .edge-head{background:linear-gradient(135deg,#101827,#164e63);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:10px}.edge-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}.edge-head h1{font-size:clamp(1.65rem,5vw,2.6rem);line-height:1.05;margin:15px 0 8px}.edge-head p{color:#cffafe;max-width:760px;margin:0}.edge-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.edge-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}.track-outside{background:white;border:1px solid #d9dee8;border-radius:9px;padding:10px 14px;margin-bottom:12px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-active{color:#86efac}@media(max-width:520px){.edge-head{border-radius:9px;margin-top:30px}.edge-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(
        f"""<section class="edge-head"><div class="edge-top"><span>VOLUME II · LAB 11</span><span>ABOUT 50 MIN</span></div><h1>Adaptation Across a Fleet</h1><p>Where can a fleet learn without overrunning local resources or excluding the clients that produce its evidence?</p><div class="edge-meta"><div><b>Fleet</b><br>{profile.label}</div><div><b>Worker hardware</b><br>{profile.hardware.name}</div><div><b>Output</b><br>Adaptation decision with saved evidence</div></div></section>"""
    )
    selector = mo.Html(
        '<div class="track-outside"><b>Choose a fleet context.</b> Changing it clears saved evidence.</div>'
    )
    mo.vstack(
        [
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            css,
            header,
            selector,
            track,
            mo.md(
                "Illustrative workload, replay, and learning outcomes are scenario assumptions. Physical resource calculations use MLSysIM."
            ),
        ],
        gap=0.45,
    )
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(
        f"""<div style="border-left:4px solid {COLORS["BlueLine"]};background:white;border-radius:0 12px 12px 0;padding:18px 24px;margin:8px 0 14px;box-shadow:0 1px 4px #00000010"><small><b>LEARNING OBJECTIVES</b></small><div style="line-height:1.7;margin-top:6px">1. <b>Compare</b> central, local, and federated data boundaries.<br>2. <b>Quantify</b> memory, energy, replay, and communication trade-offs.<br>3. <b>Design</b> a cohort-aware participation policy and fleet decision.</div><hr style="border:0;border-top:1px solid #e5e7eb;margin:14px 0"><div style="display:flex;gap:28px;flex-wrap:wrap"><div><small><b>PREREQUISITES</b></small><br>Training state · replay · federated averaging</div><div><small><b>DURATION</b></small><br>Five parts · about 9 minutes each</div></div><hr style="border:0;border-top:1px solid #e5e7eb;margin:14px 0"><small style="color:{COLORS["BlueLine"]}"><b>CORE QUESTION</b></small><div style="font-size:1.03rem;font-weight:600;font-style:italic;margin-top:5px">Can useful fleet adaptation fit local resource limits and still learn from representative, fresh clients?</div></div>"""
    )
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    a_capture,
    a_choice,
    a_epochs,
    a_prediction,
    a_rejected,
    a_results,
    a_upstream,
    apply_plotly_theme,
    audit_evidence,
    b_baseline,
    b_batch,
    b_capture,
    b_foreground,
    b_method,
    b_prediction,
    b_result,
    b_upstream,
    c_capture,
    c_fraction,
    c_prediction,
    c_result,
    c_results,
    c_upstream,
    d_capture,
    d_epochs,
    d_heterogeneity,
    d_prediction,
    d_result,
    d_results,
    d_upstream,
    e_capture,
    e_charging,
    e_clients,
    e_freshness,
    e_policy,
    e_prediction,
    e_result,
    e_results,
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
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(
                mo.md(
                    "**STALE OR NON-CONTRASTING EVIDENCE.** Recapture after changing controls."
                ),
                kind="danger",
            )
        data = capture.to_dict()
        return mo.Html(
            f'<div class="saved"><b>Saved snapshot</b> · prediction: {data["prediction"]}<br><small>The report keeps this result if live controls move.</small></div>'
        )

    def build_part_a():
        intro = mo.md(
            "### A · Where should adaptation occur? (8 min)\nCompare data boundaries and population learning. Transfer totals use the explicitly stated horizon in each row."
        )
        if a_prediction.value is None:
            return mo.vstack([intro, a_prediction])
        rows = [
            {
                "Architecture": k.title(),
                "Raw leaves": "YES" if v["raw_data_leaves_device"] else "NO",
                "Population update": "YES" if v["population_update"] else "NO",
                "Transfer": f"{v['fleet_transfer_mib']:.2f} MiB",
                "Horizon": v["comparison_horizon"],
            }
            for k, v in a_results.items()
        ]
        fig = go.Figure(
            [
                go.Bar(
                    x=[k.title() for k in a_results],
                    y=[v["fleet_transfer_mib"] for v in a_results.values()],
                    marker_color=[
                        COLORS["OrangeLine"],
                        COLORS["BlueLine"],
                        COLORS["GreenLine"],
                    ],
                )
            ]
        )
        fig.update_layout(
            height=270,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Fleet transfer (MiB)",
            showlegend=False,
        )
        reveal = f"You predicted **{a_prediction.value}**. Federated adaptation keeps raw observations local, supports population learning, and transfers **{a_results['federated']['fleet_transfer_mib']:.2f} MiB** over its supplied target trajectory."
        return mo.vstack(
            [
                intro,
                a_prediction,
                a_epochs,
                apply_plotly_theme(fig),
                table(rows),
                mo.hstack([a_choice, a_rejected], widths="equal", wrap=True),
                mo.callout(mo.md(reveal), kind="info"),
                a_capture,
                saved("A"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Central transfer counts one raw-observation collection. Local transfer covers one local update. Federated transfer sums uploads and downloads over rounds to the supplied target; the horizons remain visible because the totals are not interchangeable measurements."
                        )
                    }
                ),
            ]
        )

    def build_part_b():
        intro = mo.md(
            "### B · Can adaptation fit without harming foreground work? (9 min)\nCompare a full update with a reduced method on the same independent fleet worker."
        )
        if b_prediction.value is None:
            return mo.vstack([intro, b_prediction])
        rows = []
        for label, result in (
            ("Full baseline", b_baseline),
            (b_method.value.title(), b_result),
        ):
            rows.append(
                {
                    "Run": label,
                    "Memory": f"{result['total_memory_mib']:.1f}/{result['memory_budget_mib']:.1f} MiB",
                    "Energy": f"{result['energy_j']:.3f} J",
                    "Duration": f"{result['duration_s']:.3f} s",
                    "Added delay": f"{result['foreground_delay_ms']:.2f}/{result['foreground_delay_budget_ms']:.2f} ms",
                    "Outcome": "PASS"
                    if result["admitted"]
                    else "FAIL · " + ", ".join(result["violations"]),
                }
            )
        fig = go.Figure()
        for label, result in (("Full", b_baseline), (b_method.value.title(), b_result)):
            for field, name, color in (
                ("weights_mib", "Weights", COLORS["BlueLine"]),
                ("gradients_mib", "Gradients", COLORS["OrangeLine"]),
                ("optimizer_mib", "Optimizer", "#7c3aed"),
                ("activations_mib", "Activations", COLORS["GreenLine"]),
            ):
                fig.add_bar(
                    name=name,
                    x=[label],
                    y=[result[field]],
                    marker_color=color,
                    legendgroup=name,
                    showlegend=label == "Full",
                )
        fig.update_layout(
            barmode="stack",
            height=290,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Update memory (MiB)",
            legend_orientation="h",
        )
        reveal = f"You predicted **{b_prediction.value}**. Full update: **{'PASS' if b_baseline['admitted'] else 'FAIL'}**. {b_method.value.title()} update: **{'PASS' if b_result['admitted'] else 'FAIL'}**. An infeasible result remains valid evidence."
        return mo.vstack(
            [
                intro,
                b_prediction,
                mo.hstack([b_method, b_batch, b_foreground], wrap=True),
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(reveal), kind="success" if b_result["admitted"] else "danger"
                ),
                b_capture,
                saved("B"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Update memory sums weights, trainable gradients, optimizer state, and activations. Duration uses effective device rate. Energy is active power times duration. Foreground delay has its own admission budget."
                        )
                    }
                ),
            ]
        )

    def build_part_c():
        intro = mo.md(
            "### C · How much past experience should remain? (9 min)\nReplay and adaptation divide finite memory. Outcomes are illustrative evidence for one matched task and population."
        )
        if c_prediction.value is None:
            return mo.vstack([intro, c_prediction])
        rows = [
            {
                "Replay": f"{f:.0%}",
                "Examples": r["retained_examples"],
                "Old": f"{r['old_context_quality_pct']:.1f}%",
                "New": f"{r['new_context_quality_pct']:.1f}%",
                "Weaker": f"{r['balanced_quality_pct']:.1f}%",
                "Fit": "PASS" if r["feasible"] else "FAIL",
            }
            for f, r in c_results.items()
        ]
        fig = go.Figure()
        fig.add_scatter(
            x=list(c_results),
            y=[r["old_context_quality_pct"] for r in c_results.values()],
            mode="lines+markers",
            name="Old context",
            line_color=COLORS["BlueLine"],
        )
        fig.add_scatter(
            x=list(c_results),
            y=[r["new_context_quality_pct"] for r in c_results.values()],
            mode="lines+markers",
            name="New context",
            line_color=COLORS["OrangeLine"],
        )
        fig.update_layout(
            height=285,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Memory fraction for replay",
            xaxis_tickformat=".0%",
            yaxis_title="Illustrative outcome (%)",
            legend_orientation="h",
        )
        reveal = f"You predicted **{c_prediction.value:.0%}**. The selected **{c_fraction.value:.0%}** retains **{c_result['retained_examples']:,} examples**; its weaker supplied outcome is **{c_result['balanced_quality_pct']:.1f}%**."
        return mo.vstack(
            [
                intro,
                c_prediction,
                c_fraction,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(
                    mo.md(reveal), kind="success" if c_result["feasible"] else "danger"
                ),
                c_capture,
                saved("C"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Replay bytes equal the selected fraction of finite memory. Retained examples divide those bytes by stored-example size. Hardware does not generate the supplied old/new outcomes."
                        )
                    }
                ),
            ]
        )

    def build_part_d():
        intro = mo.md(
            "### D · How much local work should precede communication? (10 min)\nEvery row reaches the same 90% target in the illustrative convergence fixture."
        )
        if d_prediction.value is None:
            return mo.vstack([intro, d_prediction])
        rows = [
            {
                "Epochs": e,
                "Rounds": r["rounds_to_target"],
                "Wall time": f"{r['wall_time_s']:.1f} s",
                "Traffic": f"{r['total_communication_mib']:.1f} MiB",
            }
            for e, r in d_results.items()
        ]
        fig = go.Figure(
            [
                go.Scatter(
                    x=list(d_results),
                    y=[r["total_communication_mib"] for r in d_results.values()],
                    mode="lines+markers",
                    line_color=COLORS["OrangeLine"],
                )
            ]
        )
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Local epochs",
            yaxis_title="Traffic to 90% (MiB)",
            showlegend=False,
        )
        reveal = f"You predicted **{d_prediction.value} epoch(s)**. At **{d_heterogeneity.value}** heterogeneity, {d_epochs.value} epochs require **{d_result['rounds_to_target']} rounds**, **{d_result['total_communication_mib']:.1f} MiB**, and **{d_result['wall_time_s']:.1f} s**."
        return mo.vstack(
            [
                intro,
                d_prediction,
                mo.hstack([d_heterogeneity, d_epochs], widths="equal", wrap=True),
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(mo.md(reveal), kind="info"),
                d_capture,
                saved("D"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "A round includes slow-client local work, finite aggregate upload/download time, and coordination. Total bytes sum both directions across clients and supplied rounds to one target."
                        )
                    }
                ),
            ]
        )

    def build_part_e():
        intro = mo.md(
            f"### E · Who can participate without biasing the evidence? (9 min)\n"
            f"Eligibility checks explicit client records before selection ranks them. "
            f"Modeled candidate sample roster: **{len(e_result['completion_seconds'])} candidate nodes** for one round "
            f"(full fleet target: **{profile.clients_per_round} active clients/round**). "
            f"Deployment assumption: *{profile.selection_assumptions}*."
        )
        if e_prediction.value is None:
            return mo.vstack([intro, e_prediction])
        charging_callout = (
            mo.callout(
                mo.md(
                    f"**Mains-powered role:** Charging requirements are physically inapplicable to "
                    f"line-powered {profile.hardware.name} workers and are disabled."
                ),
                kind="info",
            )
            if profile.mains_powered
            else None
        )
        rows = [
            {
                "Policy": p.title(),
                "Eligible": r["eligible_count"],
                "Selected": r["selected_count"],
                "Cohorts": ", ".join(r["selected_cohorts"]) or "none",
                "Coverage": f"{r['cohort_coverage_fraction']:.0%}",
                "Max completion": f"{r['max_completion_seconds']:.1f} s"
                if r["max_completion_seconds"] is not None
                else "none",
            }
            for p, r in e_results.items()
        ]
        completion, selected = (
            e_result["completion_seconds"],
            set(e_result["selected_ids"]),
        )
        fig = go.Figure(
            [
                go.Bar(
                    x=list(completion),
                    y=list(completion.values()),
                    marker_color=[
                        COLORS["GreenLine"] if c in selected else COLORS["Grey"]
                        for c in completion
                    ],
                )
            ]
        )
        fig.update_layout(
            height=275,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Client record",
            yaxis_title="Estimated completion (s)",
            showlegend=False,
        )
        reveal = f"You predicted **{e_prediction.value}**. The chosen policy selects **{e_result['selected_count']} clients** across **{e_result['selected_cohort_count']} cohorts**; coverage is **{e_result['cohort_coverage_fraction']:.0%}**. Resource failures: {e_result['failure_counts'] or 'none'}."
        elements = [intro, e_prediction]
        if charging_callout:
            elements.append(charging_callout)
        elements.extend(
            [
                mo.hstack([e_freshness, e_clients, e_charging], wrap=True),
                e_policy,
                apply_plotly_theme(fig),
                table(rows),
                mo.callout(mo.md(reveal), kind="info"),
                e_capture,
                saved("E"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Completion adds local adapter-update and upload time. Eligibility applies battery, charging (mobile only; bypassed on mains-powered roles), uplink, freshness, availability window, and deadline checks. Modeled sample roster inspects 8 explicit candidate records for one cluster/round; full fleet scale coordinates multiple such candidate pools without fabricating unmeasured full-fleet traces. Coverage is represented cohorts divided by roster cohorts, not a quality score."
                        )
                    }
                ),
            ]
        )
        return mo.vstack(elements)

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            status = (
                "MISSING"
                if capture is None
                else (
                    "CURRENT"
                    if part not in audit.stale
                    and (part, part) not in audit.identical_pairs
                    else "STALE"
                )
            )
            rows.append(
                {
                    "Part": part,
                    "Prediction": capture.to_dict()["prediction"] if capture else "—",
                    "Evidence": status,
                }
            )
        a_decision = _captures["A"].to_dict()["decision"] if "A" in _captures else None
        complete = (
            audit.complete
            and all(
                w.value is not None
                for w in (final_choice, final_rejected, final_trigger, final_risk)
            )
            and final_choice.value != final_rejected.value
            and final_choice.value == a_decision
            and bool(rationale.value.strip())
        )
        prompt = (
            "Ready for the local report."
            if complete
            else "Capture five current contrasts, match the recommendation to Part A, choose a different rejected alternative, select a limitation and trigger, then write the rationale."
        )
        return mo.vstack(
            [
                mo.md(
                    "### Synthesis · Defend one fleet decision (5 min)\nChoose an architecture, quantify a rejected alternative from saved evidence, name the remaining limitation, and state the observation that triggers reevaluation."
                ),
                table(rows),
                mo.callout(
                    mo.md(
                        "Saved snapshots are immutable. Moving a control marks dependent evidence stale instead of rewriting it."
                    ),
                    kind="info",
                ),
                mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
                mo.hstack([final_risk, final_trigger], widths="equal", wrap=True),
                rationale,
                mo.callout(
                    mo.md(f"**{prompt}**"), kind="success" if complete else "warn"
                ),
            ]
        )

    tabs = mo.ui.tabs(
        {
            "Part A · Placement": build_part_a(),
            "Part B · Admission": build_part_b(),
            "Part C · Replay": build_part_c(),
            "Part D · Local work": build_part_d(),
            "Part E · Participation": build_part_e(),
            "Synthesis": build_synthesis(),
        }
    )
    tabs
    return (audit,)


@app.cell(hide_code=True)
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
    student_id,
    track_id,
):
    _captures = get_evidence()
    _a_decision = _captures["A"].to_dict()["decision"] if "A" in _captures else None
    _ready = (
        audit.complete
        and all(
            w.value is not None
            for w in (final_choice, final_rejected, final_trigger, final_risk)
        )
        and final_choice.value != final_rejected.value
        and final_choice.value == _a_decision
        and bool(rationale.value.strip())
    )
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_11_edge_intelligence.py"),
        student_id=student_id.value,
        track=track_id,
        scenario=profile.label,
        learning_objectives=[
            "Compare adaptation data boundaries",
            "Quantify local and fleet resource trade-offs",
            "Design a cohort-aware participation policy",
        ],
        predictions={p: snapshots[p]["prediction"] for p in "ABCDE"},
        knob_settings={p: snapshots[p]["inputs"] for p in "ABCDE"},
        evidence_summary={
            p: {
                "baseline": snapshots[p]["baseline"],
                "result": snapshots[p]["result"],
                "chosen_result": snapshots[p].get("chosen_result"),
                "result_role": snapshots[p].get("result_role"),
                "alternatives": snapshots[p]["alternatives"],
            }
            for p in "ABCDE"
        },
        binding_constraints={
            "adaptation": snapshots["B"]["result"]["violations"],
            "replay_feasible": snapshots["C"]["result"]["feasible"],
            "participation_failures": snapshots["E"]["result"]["failure_counts"],
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
            "Placement changes data exposure and coordination.",
            "Updates must pass memory, energy, time, and foreground checks separately.",
            "Local work and eligibility change communication and fleet evidence.",
        ],
        reflections={
            "rationale": rationale.value,
            "reevaluation_trigger": final_trigger.value,
        },
        residual_risk=final_risk.value,
        result_snapshot={
            "schema_version": 1,
            "lab_id": "v2_11",
            "track_id": track_id,
            "model_id": "v2_11_experiments",
            "evidence": snapshots,
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "reevaluation_trigger": final_trigger.value,
            "residual_risk": final_risk.value,
            "rationale": rationale.value,
        },
        source_trace={
            "scenario": "Illustrative replay and learning outcomes; analytical physical resource calculations.",
            "evidence_boundary": "Saved baseline and result inputs reproduce each experiment exactly.",
        },
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell(hide_code=True)
async def _(
    MODEL_ID,
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
        (get_evidence().get("A") is not None)
        and audit.complete
        and all(
            w.value is not None
            for w in (final_choice, final_rejected, final_trigger, final_risk)
        )
        and final_choice.value != final_rejected.value
        and final_choice.value == get_evidence()["A"].to_dict()["decision"]
        and bool(rationale.value.strip())
    )
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(
                chapter=11,
                design={
                    "schema_version": 1,
                    "lab_id": "v2_11",
                    "track_id": track_id,
                    "model_id": MODEL_ID,
                    "evidence": {p: c.to_dict() for p, c in get_evidence().items()},
                    "recommendation": final_choice.value,
                    "rejected_alternative": final_rejected.value,
                    "reevaluation_trigger": final_trigger.value,
                    "residual_risk": final_risk.value,
                    "rationale": rationale.value,
                },
            )
            await ledger.flush()
        except Exception as error:
            _status = (
                f"SAVE FAILED · {type(error).__name__} · LOCAL REPORT REMAINS AVAILABLE"
            )
        else:
            _status = "SAVED"
    mo.Html(
        f'<div class="lab-hud"><span class="hud-label">LAB</span><span>11 · Fleet Adaptation</span><span aria-hidden="true">|</span><span class="hud-label">STATUS</span><span class="hud-active">{_status}</span></div>'
    )
    return


if __name__ == "__main__":
    app.run()
