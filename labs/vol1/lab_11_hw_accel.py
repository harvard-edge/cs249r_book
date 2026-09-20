import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 11: Acceleration That Survives the Whole Path · MLSysBook")


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
    from mlsysim.engine.v1_11_experiments import (
        MODEL_ID, analyze_execution_path, analyze_roofline, analyze_tile_mapping,
        compare_application_speedup, get_track_scenario, make_replay_packet,
        rank_accelerators, scenario_budgets, scenario_candidates,
        scenario_execution_dimensions, scenario_gemm_demand, scenario_tile_choices,
        to_jsonable,
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
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, MODEL_ID, analyze_execution_path,
        analyze_roofline, analyze_tile_mapping, apply_plotly_theme, audit_evidence,
        build_lab_report, capture_evidence, compare_application_speedup,
        get_lab_metadata, get_track_scenario, go, ledger, make_replay_packet, mo,
        rank_accelerators, report_export_panel, scenario_budgets,
        scenario_candidates, scenario_execution_dimensions, scenario_gemm_demand,
        scenario_tile_choices, to_jsonable,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Deployment track", on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_track_scenario, track):
    track_id = track.value
    scenario = get_track_scenario(track_id)
    return scenario, track_id


@app.cell
def _(mo, track_id):
    _track = track_id
    a_change = mo.ui.dropdown(
        {"Arithmetic": "compute", "Bandwidth": "bandwidth", "Reuse": "reuse"},
        value="Arithmetic", label="Resource change",
    )
    a_factor = mo.ui.slider(2, 8, value=2, step=1, label="Change factor")
    b_tile = mo.ui.dropdown(
        {"Compact": "compact", "Baseline": "baseline", "Oversized": "oversized"},
        value="Compact", label="Tile",
    )
    b_fused = mo.ui.checkbox(value=True, label="Fuse the following operation")
    c_shape = mo.ui.dropdown(
        {"Aligned": "aligned", "Misaligned": "misaligned"}, value="Aligned", label="Shape",
    )
    c_operation = mo.ui.dropdown(
        {"Matrix multiply": "gemm", "Scatter update": "scatter"},
        value="Matrix multiply", label="Operation",
    )
    d_speedup = mo.ui.slider(2, 50, value=10, step=2, label="Kernel-only speedup")
    e_objective = mo.ui.dropdown(
        {"Latency": "latency", "Accelerator energy": "energy", "Operating cost": "cost"},
        value="Latency", label="Objective",
    )
    e_budget = mo.ui.slider(0.05, 1.5, value=1.0, step=0.05, label="Budget scale")
    e_choice = mo.ui.radio(
        {
            "Primary": "primary",
            "Alternative": "alternative",
            "No change / hold for more evidence": "hold",
            "No feasible design": "none",
        },
        label="Deployment decision",
    )
    e_rejected = mo.ui.radio(
        {"Primary": "primary", "Alternative": "alternative"}, label="Rejected tested candidate",
    )
    return a_change, a_factor, b_fused, b_tile, c_operation, c_shape, d_speedup, e_budget, e_choice, e_objective, e_rejected


@app.cell
def _(mo, track_id):
    _track = track_id
    a_prediction = mo.ui.radio(
        {"Latency falls nearly with the factor": "large", "Latency barely changes": "small", "The bottleneck switches": "switch"},
        label="What happens after the resource change?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Fits and moves fewer bytes": "fit_less", "Fits but moves more bytes": "fit_more", "Does not fit local memory": "spill"},
        label="What will the selected mapping do?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Native": "native", "Padded": "padded", "Fallback": "fallback"},
        label="Which execution path will run?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Nearly the kernel speedup": "near", "Less than half": "under_half", "Almost no gain": "small"},
        label="How much kernel speedup survives end to end?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Primary wins": "primary", "Alternative wins": "alternative", "Neither is feasible": "none"},
        label="Which candidate survives the budgets and objective?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    reevaluation_trigger = mo.ui.radio(
        {"Latency budget tightens": "latency", "Energy budget tightens": "energy", "Execution support changes": "support"},
        label="Reevaluation trigger",
    )
    residual_risk = mo.ui.radio(
        {"Deployed kernels differ from the scenario contract": "contract", "Power draw differs from the TDP proxy": "power", "Workload shape mix changes": "shape_mix"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Evidence-backed rationale",
        placeholder="Cite the chosen option, quantify the rejected candidate, name the limitation, and explain the trigger.",
    )
    return rationale, reevaluation_trigger, residual_risk


@app.cell
def _(
    a_change, a_factor, analyze_execution_path, analyze_roofline,
    analyze_tile_mapping, b_fused, b_tile, c_operation, c_shape,
    compare_application_speedup, d_speedup, e_budget, e_objective,
    make_replay_packet, rank_accelerators, scenario, scenario_budgets,
    scenario_candidates, scenario_execution_dimensions, scenario_gemm_demand,
    scenario_tile_choices, track_id,
):
    demand = scenario_gemm_demand(track_id)
    a_base_inputs = dict(
        hardware=scenario.hardware, operations=demand.operations,
        base_bytes_moved=demand.bytes_moved, precision=scenario.precision,
        reuse=1.0, compute_scale=1.0, bandwidth_scale=1.0,
    )
    a_result_inputs = dict(a_base_inputs)
    if a_change.value == "compute":
        a_result_inputs["compute_scale"] = float(a_factor.value)
    elif a_change.value == "bandwidth":
        a_result_inputs["bandwidth_scale"] = float(a_factor.value)
    else:
        a_result_inputs["reuse"] = float(a_factor.value)
    a_base = analyze_roofline(**a_base_inputs)
    a_result = analyze_roofline(**a_result_inputs)
    a_base_packet = make_replay_packet(track=track_id, experiment="roofline", inputs=a_base_inputs)
    a_result_packet = make_replay_packet(track=track_id, experiment="roofline", inputs=a_result_inputs)

    _tile_choices = scenario_tile_choices(track_id)
    _base_tile = _tile_choices["baseline"]
    _selected_tile = _tile_choices[b_tile.value]
    _tile_common = dict(
        hardware=scenario.hardware, m=scenario.dimensions[0], n=scenario.dimensions[1],
        k=scenario.dimensions[2], precision=scenario.precision,
        scratchpad_capacity=scenario.local_capacity,
    )
    b_base_inputs = dict(_tile_common, tile_m=_base_tile[0], tile_n=_base_tile[1], tile_k=_base_tile[2], fused=False)
    b_result_inputs = dict(_tile_common, tile_m=_selected_tile[0], tile_n=_selected_tile[1], tile_k=_selected_tile[2], fused=b_fused.value)
    b_base = analyze_tile_mapping(**b_base_inputs)
    b_result = analyze_tile_mapping(**b_result_inputs)
    b_base_packet = make_replay_packet(track=track_id, experiment="tile_mapping", inputs=b_base_inputs)
    b_result_packet = make_replay_packet(track=track_id, experiment="tile_mapping", inputs=b_result_inputs)

    _base_dims = scenario_execution_dimensions(track_id, aligned=True)
    _selected_dims = scenario_execution_dimensions(track_id, aligned=c_shape.value == "aligned")
    _execution_common = dict(
        hardware=scenario.hardware, fallback_hardware=scenario.fallback,
        contract=scenario.contract, precision=scenario.precision,
        quality_observation="illustrative matched-task evidence held constant",
    )
    c_base_inputs = dict(_execution_common, operation="gemm", m=_base_dims[0], n=_base_dims[1], k=_base_dims[2])
    c_result_inputs = dict(_execution_common, operation=c_operation.value, m=_selected_dims[0], n=_selected_dims[1], k=_selected_dims[2])
    c_base = analyze_execution_path(**c_base_inputs)
    c_result = analyze_execution_path(**c_result_inputs)
    c_base_packet = make_replay_packet(track=track_id, experiment="execution_path", inputs=c_base_inputs)
    c_result_packet = make_replay_packet(track=track_id, experiment="execution_path", inputs=c_result_inputs)

    d_base_inputs = dict(
        hardware=scenario.hardware, baseline_kernel_time=c_result.latency,
        local_speedup=1.0, host_time=scenario.host_time,
        transfer_bytes=scenario.transfer_bytes,
        transfer_fixed_latency=scenario.transfer_fixed_latency,
        launches=scenario.launches, postprocess_time=scenario.postprocess_time,
    )
    d_result_inputs = dict(d_base_inputs, local_speedup=float(d_speedup.value))
    d_base = compare_application_speedup(**d_base_inputs)
    d_result = compare_application_speedup(**d_result_inputs)
    d_base_packet = make_replay_packet(track=track_id, experiment="application_comparison", inputs=d_base_inputs)
    d_result_packet = make_replay_packet(track=track_id, experiment="application_comparison", inputs=d_result_inputs)

    _candidates = scenario_candidates(track_id)
    _latency_budget, _energy_budget, _cost_budget = scenario_budgets(track_id, scale=e_budget.value)
    e_common = dict(
        fallback_hardware=scenario.fallback, operation="gemm", precision=scenario.precision,
        m=scenario.dimensions[0], n=scenario.dimensions[1], k=scenario.dimensions[2],
        host_time=scenario.host_time, transfer_bytes=scenario.transfer_bytes,
        transfer_fixed_latency=scenario.transfer_fixed_latency, launches=scenario.launches,
        postprocess_time=scenario.postprocess_time, objective=e_objective.value,
        latency_budget=_latency_budget, energy_budget=_energy_budget, cost_budget=_cost_budget,
    )
    e_primary_inputs = dict(e_common, candidates=(_candidates[0],))
    e_alternative_inputs = dict(e_common, candidates=(_candidates[1],))
    e_full_inputs = dict(e_common, candidates=_candidates)
    e_primary = rank_accelerators(**e_primary_inputs)
    e_alternative = rank_accelerators(**e_alternative_inputs)
    e_full = rank_accelerators(**e_full_inputs)
    e_primary_packet = make_replay_packet(track=track_id, experiment="accelerator_ranking", inputs=e_primary_inputs)
    e_alternative_packet = make_replay_packet(track=track_id, experiment="accelerator_ranking", inputs=e_alternative_inputs)
    e_full_packet = make_replay_packet(track=track_id, experiment="accelerator_ranking", inputs=e_full_inputs)
    return (
        a_base, a_base_packet, a_result, a_result_packet, b_base, b_base_packet,
        b_result, b_result_packet, c_base, c_base_packet, c_result, c_result_packet,
        d_base, d_base_packet, d_result, d_result_packet, demand, e_alternative,
        e_alternative_packet, e_full, e_full_packet, e_primary, e_primary_packet,
    )


@app.cell
def _(
    a_base, a_base_packet, a_prediction, a_result, a_result_packet, b_base,
    b_base_packet, b_prediction, b_result, b_result_packet, c_base,
    c_base_packet, c_prediction, c_result, c_result_packet, capture_evidence,
    d_base, d_base_packet, d_prediction, d_result, d_result_packet,
    e_alternative, e_alternative_packet, e_choice, e_full, e_full_packet,
    e_prediction, e_primary, e_primary_packet, e_rejected, mo, set_evidence,
    to_jsonable, track_id,
):
    def _store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    def _run(packet, result):
        return {"inputs": packet["inputs"], "outputs": to_jsonable(result)}

    a_upstream = {"baseline": a_base_packet["inputs"], "result": a_result_packet["inputs"]}
    b_upstream = {"baseline": b_base_packet["inputs"], "result": b_result_packet["inputs"]}
    c_upstream = {"baseline": c_base_packet["inputs"], "result": c_result_packet["inputs"]}
    d_upstream = {"baseline": d_base_packet["inputs"], "result": d_result_packet["inputs"]}
    e_upstream = {"ranking": e_full_packet["inputs"], "choice": e_choice.value, "rejected": e_rejected.value}

    a_capture = mo.ui.button(
        label="Capture resource contrast", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _v: _store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value, inputs=a_upstream,
            baseline=_run(a_base_packet, a_base), result=_run(a_result_packet, a_result),
            upstream_inputs=a_upstream, model_key=a_result_packet["model_key"],
        )),
    )
    b_capture = mo.ui.button(
        label="Capture tile contrast", kind="success",
        disabled=b_prediction.value is None or b_base_packet["inputs"] == b_result_packet["inputs"],
        on_click=lambda _v: _store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value, inputs=b_upstream,
            baseline=_run(b_base_packet, b_base), result=_run(b_result_packet, b_result),
            upstream_inputs=b_upstream, model_key=b_result_packet["model_key"],
        )),
    )
    c_capture = mo.ui.button(
        label="Capture execution contrast", kind="success",
        disabled=c_prediction.value is None or c_base_packet["inputs"] == c_result_packet["inputs"],
        on_click=lambda _v: _store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value, inputs=c_upstream,
            baseline=_run(c_base_packet, c_base), result=_run(c_result_packet, c_result),
            upstream_inputs=c_upstream, model_key=c_result_packet["model_key"],
        )),
    )
    d_capture = mo.ui.button(
        label="Capture whole-path contrast", kind="success", disabled=d_prediction.value is None,
        on_click=lambda _v: _store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value, inputs=d_upstream,
            baseline=_run(d_base_packet, d_base), result=_run(d_result_packet, d_result),
            upstream_inputs=d_upstream, model_key=d_result_packet["model_key"],
        )),
    )
    _rows = {row.candidate_id: row for row in e_full.rows}
    _choice_feasible = e_choice.value in _rows and _rows[e_choice.value].feasible
    _hold_valid = e_choice.value == "hold" and _rows["primary"].feasible and e_rejected.value == "alternative"
    _none_valid = e_choice.value == "none" and e_full.recommendation is None
    e_decision_valid = (
        e_prediction.value is not None and e_rejected.value is not None
        and e_choice.value is not None and e_choice.value != e_rejected.value
        and (_choice_feasible or _hold_valid or _none_valid)
    )
    e_capture = mo.ui.button(
        label="Capture accelerator decision", kind="success", disabled=not e_decision_valid,
        on_click=lambda _v: _store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value, inputs=e_upstream,
            baseline=_run(e_primary_packet, e_primary),
            result=_run(e_alternative_packet, e_alternative), upstream_inputs=e_upstream,
            alternatives=(_run(e_full_packet, e_full),),
            decision={"recommendation": e_choice.value, "rejected": e_rejected.value},
            model_key=e_full_packet["model_key"],
            chosen_result=(
                _run(e_primary_packet, e_primary) if e_choice.value in ("primary", "hold")
                else _run(e_alternative_packet, e_alternative) if e_choice.value == "alternative"
                else _run(e_full_packet, e_full)
            ),
            result_role="chosen candidate" if e_choice.value == "alternative" else "rejected alternative",
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_decision_valid, e_upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, scenario, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:760px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}
    .pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 11</span><span>ABOUT 50–55 MIN</span></div><h1>Acceleration That Survives the Whole Path</h1><p>When does specialized hardware improve the application rather than one attractive kernel?</p><div class="pilot-meta"><div><b>Machine</b><br>{scenario.hardware.name}</div><div><b>Track</b><br>{scenario.track.title()}</div><div><b>Deliverable</b><br>Constrained accelerator decision</div></div></section>""")
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, track, header, mo.Html(f'<p class="pilot-note">{scenario.assumption_label.capitalize()}. Hardware supply comes from MLSysIM registry entries; results are analytical, not measured benchmarks.</p>')], gap=0.5).style({"padding-top": "32px"})
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base, a_capture, a_change, a_factor, a_prediction, a_result,
    a_upstream, apply_plotly_theme, audit_evidence, b_base, b_capture,
    b_fused, b_prediction, b_result, b_tile, b_upstream, c_base, c_capture,
    c_operation, c_prediction, c_result, c_shape, c_upstream, d_capture,
    d_prediction, d_result, d_speedup, d_upstream, e_budget, e_capture,
    e_choice, e_decision_valid, e_full, e_objective, e_prediction, e_rejected,
    e_upstream, get_evidence, go, mo, rationale, reevaluation_trigger,
    residual_risk, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def _table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def _saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture after changing dependencies."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · prediction: {data["prediction"]}<br><small>Track {data["track"]}; later controls do not rewrite this record.</small></div>')

    def _part_a():
        intro = mo.md("### A · When does faster arithmetic help? (9 min)\nChange one resource while workload operations and base traffic stay fixed.")
        if a_prediction.value is None:
            return mo.vstack([intro, mo.hstack([a_change, a_factor], widths="equal", wrap=True), a_prediction])
        fig = go.Figure([go.Bar(x=["Baseline", "Changed"], y=[a_base.latency.to("ms").magnitude, a_result.latency.to("ms").magnitude], marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]])])
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Analytical latency (ms)", showlegend=False)
        rows = [{"Run": "Baseline", "Intensity": f"{a_base.arithmetic_intensity.to('flop/byte').magnitude:.1f} FLOP/B", "Bottleneck": a_base.bottleneck, "Latency": f"{a_base.latency.to('ms').magnitude:.4g} ms"}, {"Run": "Changed", "Intensity": f"{a_result.arithmetic_intensity.to('flop/byte').magnitude:.1f} FLOP/B", "Bottleneck": a_result.bottleneck, "Latency": f"{a_result.latency.to('ms').magnitude:.4g} ms"}]
        return mo.vstack([intro, mo.hstack([a_change, a_factor], widths="equal", wrap=True), a_prediction, apply_plotly_theme(fig), _table(rows), mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. The binding limit is **{a_result.bottleneck}** after the change; improving a nonbinding resource cannot lower latency."), kind="info"), a_capture, _saved("A"), mo.accordion({"Calculation Notes": mo.md("Arithmetic intensity = operations / effective off-chip bytes. Reuse lowers traffic. Latency is the larger of compute time and memory time; compute and bandwidth scales change only their own ceilings.")})])

    def _part_b():
        intro = mo.md("### B · How much reuse fits locally? (10 min)\nCompare one blocked matrix mapping with the baseline tile and unfused intermediate.")
        if b_prediction.value is None:
            return mo.vstack([intro, mo.hstack([b_tile, b_fused], widths="equal", wrap=True), b_prediction])
        fig = go.Figure([go.Bar(x=["Baseline", "Selected"], y=[b_base.dram_bytes.to("KiB").magnitude, b_result.dram_bytes.to("KiB").magnitude], marker_color=[COLORS["BlueLine"], COLORS["GreenLine"]])])
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="DRAM traffic (KiB)", showlegend=False)
        rows = [{"Run": "Baseline", "Tile": str(b_base.tile), "Local need": f"{b_base.scratchpad_required.to('KiB').magnitude:.1f} KiB", "Capacity": f"{b_base.scratchpad_capacity.to('KiB').magnitude:.1f} KiB", "Mapping": "FITS" if b_base.fits else "SPILL"}, {"Run": "Selected", "Tile": str(b_result.tile), "Local need": f"{b_result.scratchpad_required.to('KiB').magnitude:.1f} KiB", "Capacity": f"{b_result.scratchpad_capacity.to('KiB').magnitude:.1f} KiB", "Mapping": "FITS" if b_result.fits else "SPILL"}]
        return mo.vstack([intro, mo.hstack([b_tile, b_fused], widths="equal", wrap=True), b_prediction, apply_plotly_theme(fig), _table(rows), mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. The selected tile **{'fits' if b_result.fits else 'does not fit'}**. A failed mapping has no fabricated runtime penalty; it must be remapped."), kind="danger" if not b_result.fits else "success"), b_capture, _saved("B"), mo.accordion({"Calculation Notes": mo.md("Scratchpad occupancy counts A, B, and C tiles. DRAM traffic counts tiled inputs, output, and an explicit write/read intermediate when fusion is off. Movement energy uses registry energy per byte.")})])

    def _part_c():
        intro = mo.md("### C · Can the specialized unit execute this workload? (9 min)\nThe execution contract is an illustrative scenario assumption. Matched-task quality evidence stays fixed and cannot accelerate hardware.")
        if c_prediction.value is None:
            return mo.vstack([intro, mo.hstack([c_shape, c_operation], widths="equal", wrap=True), c_prediction])
        rows = [{"Run": "Baseline", "Operation": c_base.operation, "Original": str(c_base.original_dimensions), "Executed": str(c_base.executed_dimensions), "Path": c_base.path, "Extra work": f"{c_base.extra_operations.to('MFLOP').magnitude:.3g} MFLOP"}, {"Run": "Selected", "Operation": c_result.operation, "Original": str(c_result.original_dimensions), "Executed": str(c_result.executed_dimensions), "Path": c_result.path, "Extra work": f"{c_result.extra_operations.to('MFLOP').magnitude:.3g} MFLOP"}]
        return mo.vstack([intro, mo.hstack([c_shape, c_operation], widths="equal", wrap=True), c_prediction, _table(rows), mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. The selected case uses the **{c_result.path}** path on **{c_result.hardware_name}**. Padding adds counted work; unsupported work names the fallback device."), kind="info"), c_capture, _saved("C"), mo.accordion({"Calculation Notes": mo.md("Native execution requires a supported operation, precision, and aligned shape. Padding rounds each dimension to the contract multiple and recomputes work. Unsupported work executes on the named fallback hardware.")})])

    def _part_d():
        intro = mo.md("### D · Why does kernel speedup disappear end to end? (9 min)\nPreserve host work, transfer, launches, and postprocessing while changing only kernel time.")
        if d_prediction.value is None:
            return mo.vstack([intro, d_speedup, d_prediction])
        baseline, changed = d_result.baseline, d_result.accelerated
        fig = go.Figure()
        for label, path in (("Baseline", baseline), ("Accelerated", changed)):
            for name, value, color in (("Host", path.host_time, COLORS["BlueLine"]), ("Transfer", path.transfer_time, COLORS["OrangeLine"]), ("Launch", path.launch_time, COLORS["GreenLine"]), ("Kernel", path.kernel_time, COLORS["RedLine"]), ("Postprocess", path.postprocess_time, COLORS["TextMuted"])):
                fig.add_bar(name=name, x=[label], y=[value.to("ms").magnitude], marker_color=color, legendgroup=name, showlegend=label == "Baseline")
        fig.update_layout(barmode="stack", height=280, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Application time (ms)", legend_orientation="h")
        return mo.vstack([intro, d_speedup, d_prediction, apply_plotly_theme(fig), mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. A **{d_result.local_speedup:.0f}×** kernel gain becomes **{d_result.end_to_end_speedup:.2f}×** end to end because the other stages remain."), kind="info"), d_capture, _saved("D"), mo.accordion({"Calculation Notes": mo.md("Application time adds host, one-way transfer plus fixed latency, registry dispatch time per launch, kernel time, and postprocessing. Only kernel time is divided by the local speedup.")})])

    def _part_e():
        intro = mo.md("### E · Which feasible accelerator should we choose? (9 min)\nBoth candidates run the same workload and application stages. Filter constraints before optimizing the selected objective.")
        if e_prediction.value is None:
            return mo.vstack([intro, mo.hstack([e_objective, e_budget], widths="equal", wrap=True), e_prediction])
        rows = [{"Candidate": row.candidate_id.title(), "Hardware": row.hardware_name, "Path": row.execution_path, "Latency": f"{row.latency.to('ms').magnitude:.3g} ms", "Energy": f"{row.accelerator_energy.to('mJ').magnitude:.3g} mJ", "Cost": f"${row.operating_cost.to('dollar').magnitude:.3g}", "Outcome": "FEASIBLE" if row.feasible else "FAIL: " + ", ".join(row.violations)} for row in e_full.rows]
        recommendation = e_full.recommendation.candidate_id if e_full.recommendation else "no feasible design"
        return mo.vstack([intro, mo.hstack([e_objective, e_budget], widths="equal", wrap=True), e_prediction, _table(rows), mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. Constrained ranking recommends **{recommendation}** for **{e_full.objective}** under the displayed illustrative budgets."), kind="success" if e_full.recommendation else "danger"), mo.hstack([e_choice, e_rejected], widths="equal", wrap=True), mo.callout(mo.md("Decision is consistent and compares a different tested candidate." if e_decision_valid else "Choose a feasible candidate, hold the feasible primary while rejecting the tested alternative, or choose no feasible design only when both fail."), kind="success" if e_decision_valid else "warn"), e_capture, _saved("E"), mo.accordion({"Calculation Notes": mo.md("Each candidate first passes native support, latency, accelerator energy, and active-time operating cost budgets. Feasible rows are then sorted by the selected objective. Holding preserves the feasible primary baseline while recording why the tested alternative was rejected.")})])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            current = capture and part not in audit.stale and (part, part) not in audit.identical_pairs
            rows.append({"Part": part, "Prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": "CURRENT" if current else ("STALE" if capture else "MISSING")})
        complete = audit.complete and "E" in _captures and all(widget.value is not None for widget in (reevaluation_trigger, residual_risk)) and bool(rationale.value.strip())
        return mo.vstack([mo.md("### Synthesis · Defend the accelerator decision (4 min)\nUse saved experiments to name the chosen option, quantify the rejected candidate, state one limitation, and define a reevaluation trigger."), _table(rows), mo.hstack([reevaluation_trigger, residual_risk], widths="equal", wrap=True), rationale, mo.callout(mo.md("**Ready for the local report.**" if complete else "Capture five current contrasts, then complete the limitation, trigger, and rationale."), kind="success" if complete else "warn")])

    tabs = mo.ui.tabs({"Part A": _part_a(), "Part B": _part_b(), "Part C": _part_c(), "Part D": _part_d(), "Part E": _part_e(), "Synthesis": build_synthesis()})
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, get_evidence, get_lab_metadata, mo, rationale,
    reevaluation_trigger, report_export_panel, residual_risk, scenario, track_id,
):
    _captures = get_evidence()
    _decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and _decision is not None and all(widget.value is not None for widget in (reevaluation_trigger, residual_risk)) and bool(rationale.value.strip())
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_11_hw_accel.py"), track=track_id,
        scenario=f"{scenario.hardware.name} analytical accelerator-fit investigation",
        learning_objectives=["Identify the binding roofline resource", "Test local-memory and execution-support boundaries", "Compare kernel and application speedup", "Rank feasible accelerators under explicit budgets"],
        predictions={part: snapshot["prediction"] for part, snapshot in snapshots.items()},
        knob_settings={part: snapshot["inputs"] for part, snapshot in snapshots.items()},
        evidence_summary={part: {"baseline": snapshot["baseline"], "result": snapshot["result"], "alternatives": snapshot["alternatives"]} for part, snapshot in snapshots.items()},
        binding_constraints={part: (snapshots[part].get("chosen_result") or snapshots[part]["result"])["outputs"] for part in "ABCDE"},
        decisions={"recommendation": _decision["recommendation"], "rejected_alternative": _decision["rejected"], "reevaluation_trigger": reevaluation_trigger.value},
        final_decision={"recommendation": _decision["recommendation"], "rejected_alternative": _decision["rejected"], "rationale": rationale.value},
        big_takeaways=["Peak arithmetic helps only when compute binds.", "Local capacity and execution support can force remapping.", "Host, transfer, and launch stages cap application speedup.", "Selection filters constraints before optimizing an objective."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": reevaluation_trigger.value}, residual_risk=residual_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": _decision["recommendation"], "rejected": _decision["rejected"], "trigger": reevaluation_trigger.value, "residual_risk": residual_risk.value},
        source_trace={"hardware": "MLSysIM registry", "scenario": scenario.assumption_label, "results": "Analytical MLSysIM Chapter 11 experiment models; not measured benchmarks."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(MODEL_ID, audit, get_evidence, ledger, mo, rationale, reevaluation_trigger, residual_risk, track_id):
    _captures = get_evidence()
    _decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and _decision is not None and all(widget.value is not None for widget in (reevaluation_trigger, residual_risk)) and bool(rationale.value.strip())
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(chapter=11, design={
                "schema_version": 1, "lab_id": "v1_11", "track_id": track_id,
                "model_id": MODEL_ID, "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": _decision["recommendation"], "rejected_alternative": _decision["rejected"],
                "reevaluation_trigger": reevaluation_trigger.value, "residual_risk": residual_risk.value,
                "rationale": rationale.value,
            })
            await ledger.flush()
            _status = "SAVED"
        except Exception:
            _status = "SAVE FAILED · REPORT STILL AVAILABLE"
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">11 · Acceleration That Survives the Whole Path · STATUS: </span><span class="hud-active">{_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
