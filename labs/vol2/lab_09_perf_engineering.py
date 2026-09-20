import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 09: Optimize the Path · MLSysBook")


# ZONE A · Opening and setup
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
    from mlsysim.engine.v2_09_experiments import (
        MODEL_ID, TRACKS, compare_configs, evaluate, get_scenario,
        quantity_from_dict, result_to_dict,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata,
        get_lab_track_variant, get_track_profile, report_export_panel,
        track_context,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, MODEL_ID, TRACKS,
        apply_plotly_theme, audit_evidence, build_lab_report, capture_evidence,
        compare_configs, evaluate, get_lab_metadata, get_lab_track_variant,
        get_scenario, get_track_profile, go, ledger, mo, quantity_from_dict,
        report_export_panel, result_to_dict, track_context,
    )


# ZONE B · Widgets and experiment state
@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Fleet track", on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_lab_metadata, get_lab_track_variant, get_scenario, get_track_profile, track):
    track_id = track.value
    scenario = get_scenario(track_id)
    track_profile = get_track_profile(track_id)
    _metadata = get_lab_metadata("vol2/lab_09_perf_engineering.py")
    track_variant = get_lab_track_variant(_metadata.lab_id, track_id)
    return scenario, track_id, track_profile, track_variant


@app.cell
def _(TRACKS, mo, track_id):
    _track_key = track_id
    a_batch = mo.ui.dropdown(
        {"2 items": 2, "4 items": 4, "8 items": 8}, value="4 items",
        label="Items per execution",
    )
    b_lever = mo.ui.radio(
        {"Fuse operators": "fusion", "Tile for reuse": "tiling",
         "Combine fusion and tiling": "combined"},
        value="Fuse operators", label="Path rewrite to test",
    )
    _precision_options = {
        item.precision.upper(): item.precision
        for item in TRACKS[track_id].precision_evidence
    }
    c_precision = mo.ui.dropdown(
        _precision_options, value=TRACKS[track_id].default_precision.upper(),
        label="Execution precision",
    )
    c_reuse = mo.ui.dropdown(
        {"1 run": 1, "100 runs": 100, "100,000 runs": 100_000},
        value="100,000 runs", label="Compiled-region reuse",
    )
    _stress_options = {
        "tinyml": {"8 items": 8, "32 items": 32, "64 items": 64},
        "mobile": {"8 items": 8, "64 items": 64, "1,000 items": 1_000},
        "edge": {"8 items": 8, "64 items": 64, "400 items": 400},
        "cloud": {"8 items": 8, "16 items": 16, "64 items": 64},
    }
    d_batch = mo.ui.dropdown(
        _stress_options[track_id], value=list(_stress_options[track_id])[1],
        label="Fleet-unit batch under stress",
    )
    return a_batch, b_lever, c_precision, c_reuse, d_batch


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Memory traffic": "memory", "Compute": "compute",
         "Launch or compile overhead": "overhead"},
        label="Which term binds the initial batch probe?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Fusion": "fusion", "Tiling": "tiling", "Combined": "combined"},
        label="Which rewrite will deliver the largest end-to-end speedup?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Setup dominates": "setup", "Runtime saving dominates": "runtime",
         "Precision dominates": "precision"},
        label="What determines whether the compiled candidate wins?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Fits per fleet unit": "fit", "Exceeds per-unit memory": "fail"},
        label="Will the stressed batch fit one execution unit?",
    ).form(submit_button_label="Lock Part D prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    c_decision = mo.ui.radio(
        {"Carry compiled precision": "compiled", "Keep the path rewrite": "rewrite"},
        label="Configuration to carry into the boundary test",
    )
    _options = {
        "Adopt tested batch": "batched",
        "Keep prior optimized batch": "optimized",
        "Return to unoptimized selected-batch path": "baseline",
    }
    d_decision = mo.ui.radio(_options, label="Recommendation after the boundary test")
    d_rejected = mo.ui.radio(_options, label="Quantified rejected alternative")
    final_choice = mo.ui.radio(_options, label="Final recommendation")
    final_rejected = mo.ui.radio(_options, label="Rejected alternative")
    final_trigger = mo.ui.radio(
        {"Workload mix changes": "workload", "Per-unit memory shrinks": "memory",
         "Quality evidence changes": "quality"}, label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Illustrative quality fixture": "quality_fixture",
         "Unmodeled communication": "communication",
         "Peak-rate attainability": "attainability"}, label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Concise rationale",
        placeholder="Cite a saved speedup or memory result, then quantify why the alternative loses.",
    )
    return (
        c_decision, d_decision, d_rejected, final_choice, final_rejected,
        final_risk, final_trigger, rationale,
    )


# Experiment orchestration through MLSysIM
@app.cell
def _(
    a_batch, b_lever, c_decision, c_precision, c_reuse, compare_configs,
    d_batch, evaluate, result_to_dict, scenario, track_id,
):
    a_baseline = evaluate(track_id, batch_size=1)
    a_result = evaluate(track_id, batch_size=a_batch.value)
    _rewrite_flags = {
        "fusion": {"fusion": True, "tiling": False},
        "tiling": {"fusion": False, "tiling": True},
        "combined": {"fusion": True, "tiling": True},
    }
    _a_config = {
        "batch_size": a_batch.value, "fleet_units": scenario.default_fleet_units,
        "precision": scenario.default_precision, "fusion": False,
        "tiling": False, "compiled": False, "reuse_count": 1,
    }
    _b_config = {**_a_config, **_rewrite_flags[b_lever.value]}
    b_comparison = compare_configs(
        track_id, baseline_options=_a_config, result_options=_b_config,
    )
    b_alternatives = {
        _name: result_to_dict(evaluate(track_id, **{**_a_config, **_flags}))
        for _name, _flags in _rewrite_flags.items()
    }
    _c_config = {
        **_b_config, "precision": c_precision.value, "compiled": True,
        "reuse_count": c_reuse.value,
    }
    c_comparison = compare_configs(
        track_id, baseline_options=_b_config, result_options=_c_config,
    )
    c_reuse_results = {
        _reuse: evaluate(track_id, **{**_c_config, "reuse_count": _reuse})
        for _reuse in (1, 100, 100_000)
    }
    c_carried = (
        c_comparison["result"] if c_decision.value == "compiled"
        else c_comparison["baseline"]
    )
    _carried_inputs = {
        "batch_size": c_carried.inputs.batch_size,
        "fleet_units": c_carried.inputs.fleet_units,
        "precision": c_carried.inputs.precision,
        "fusion": c_carried.inputs.fusion,
        "tiling": c_carried.inputs.tiling,
        "compiled": c_carried.inputs.compiled,
        "reuse_count": c_carried.inputs.reuse_count,
    }
    _d_config = {**_carried_inputs, "batch_size": d_batch.value}
    d_comparison = compare_configs(
        track_id, baseline_options=_carried_inputs, result_options=_d_config,
    )
    d_alternatives = {
        _batch: result_to_dict(
            evaluate(track_id, **{**_carried_inputs, "batch_size": _batch})
        )
        for _batch in (
            scenario.default_batch_size,
            d_batch.value,
            {"tinyml": 64, "mobile": 1_000, "edge": 400, "cloud": 64}[track_id],
        )
    }
    return (
        a_baseline, a_result, b_alternatives, b_comparison, c_carried,
        c_comparison, c_reuse_results, d_alternatives, d_comparison,
    )


@app.cell
def _(
    MODEL_ID, a_baseline, a_batch, a_prediction, a_result, b_alternatives,
    b_comparison, b_lever, b_prediction, c_comparison, c_decision,
    c_precision, c_reuse, capture_evidence, d_alternatives, d_batch,
    d_comparison, d_decision, d_prediction, d_rejected, mo, result_to_dict,
    set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = None
    b_upstream = {"batch_size": a_batch.value}
    c_upstream = {**b_upstream, "rewrite": b_lever.value}
    d_upstream = {
        **c_upstream, "precision": c_precision.value,
        "reuse_count": c_reuse.value, "carried": c_decision.value,
    }
    a_capture = mo.ui.button(
        label="Capture bottleneck contrast", kind="success",
        disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"batch_size": a_batch.value},
            baseline=result_to_dict(a_baseline), result=result_to_dict(a_result),
            upstream_inputs=a_upstream, decision=a_result.bottleneck,
            model_key=MODEL_ID,
        )),
    )
    b_capture = mo.ui.button(
        label="Capture rewrite comparison", kind="success",
        disabled=b_prediction.value is None,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"rewrite": b_lever.value},
            baseline=result_to_dict(b_comparison["baseline"]),
            result=result_to_dict(b_comparison["result"]),
            alternatives=b_alternatives, upstream_inputs=b_upstream,
            decision=b_lever.value, model_key=MODEL_ID,
        )),
    )
    c_capture = mo.ui.button(
        label="Capture compile and precision evidence", kind="success",
        disabled=c_prediction.value is None or c_decision.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"precision": c_precision.value, "reuse_count": c_reuse.value,
                    "carried": c_decision.value},
            baseline=result_to_dict(c_comparison["baseline"]),
            result=result_to_dict(c_comparison["result"]),
            upstream_inputs=c_upstream, decision=c_decision.value,
            model_key=MODEL_ID,
        )),
    )
    _d_invalid = (
        d_prediction.value is None or c_decision.value is None or d_decision.value is None
        or d_rejected.value is None or d_decision.value == d_rejected.value
        or (d_decision.value != "batched" and d_rejected.value != "batched")
    )
    d_capture = mo.ui.button(
        label="Capture memory-boundary decision", kind="success",
        disabled=_d_invalid,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"stress_batch": d_batch.value, "recommendation": d_decision.value,
                    "rejected": d_rejected.value},
            baseline=result_to_dict(d_comparison["baseline"]),
            result=result_to_dict(d_comparison["result"]),
            chosen_result=(
                result_to_dict(a_result) if d_decision.value == "baseline"
                else result_to_dict(d_comparison["baseline"])
                if d_decision.value == "optimized" else None
            ),
            result_role=(
                "chosen intervention" if d_decision.value == "batched"
                else "rejected alternative"
            ),
            alternatives=d_alternatives, upstream_inputs=d_upstream,
            decision=d_decision.value, model_key=MODEL_ID,
        )),
    )
    return (
        a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream,
        d_capture, d_upstream,
    )


@app.cell
def _(
    ACADEMIC_LAB_CSS, LAB_CSS, mo, scenario, track, track_context,
    track_profile, track_variant,
):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin:30px 0 10px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:760px}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .selector-shell{padding:10px 12px;border:1px solid #dbe4ee;border-radius:9px;background:#fff;margin-bottom:8px;max-width:100%}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .table-wrap{max-width:100%;overflow-x:auto}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-separator{color:#64748b}.lab-hud .hud-active{color:#86efac}.lab-hud .hud-failed{color:#fca5a5}
    @media(max-width:520px){.pilot-head{border-radius:9px}.pilot-meta{grid-template-columns:1fr}.selector-shell{padding:8px}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME II · LAB 09</span><span>ABOUT 50–55 MIN</span></div><h1>Optimize the Path</h1><p>Which physical term should the fleet team change, and when does that optimization stop paying?</p><div class="pilot-meta"><div><b>Stakeholder</b><br>{track_variant.stakeholder}</div><div><b>Fleet unit</b><br>{scenario.fleet_semantics}</div><div><b>Workload</b><br>{scenario.workload}</div><div><b>Output</b><br>Performance engineering memo</div></div></section>""")
    selector = mo.vstack([mo.md("**Choose the fleet context**"), track]).style(
        {
            "max-width": "100%", "padding": "10px 12px",
            "border": "1px solid #dbe4ee", "border-radius": "9px",
            "background": "#fff", "margin-bottom": "8px",
        }
    )
    context = track_context(track_profile)
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css, header, selector, context,
        mo.Html('<p class="pilot-note">Results are simulated analytical bounds. Quality values are supplied scenario evidence, not hardware-derived measurements. Open Calculation Notes in each part for equations and assumptions.</p>'),
    ], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


# ZONE C · Single tab set
@app.cell
def _(
    COLORS, a_baseline, a_batch, a_capture, a_prediction, a_result,
    a_upstream, apply_plotly_theme, audit_evidence, b_alternatives, b_capture,
    b_comparison, b_lever, b_prediction, b_upstream, c_capture, c_comparison,
    c_decision, c_precision, c_reuse, c_reuse_results, c_upstream,
    d_alternatives, d_batch, d_capture, d_comparison, d_decision,
    d_prediction, d_rejected, d_upstream, final_choice, final_rejected,
    final_risk, final_trigger, get_evidence, go, mo, quantity_from_dict,
    rationale, scenario, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream}
    _mem_unit = {"tinyml": "KiB", "mobile": "MiB", "edge": "GiB", "cloud": "GiB"}[track_id]
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCD"),
        per_part_upstream_inputs=_upstream,
        contrast_required_parts=tuple("ABCD"),
    )

    def q(value, unit, digits=2):
        return f"{value.to(unit).magnitude:,.{digits}f} {unit}"

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style(
            {"max-width": "100%", "overflow-x": "auto"}
        )

    def saved(part):
        _capture = _captures.get(part)
        if _capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(
                mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** An upstream choice changed. Recapture this comparison."),
                kind="danger",
            )
        _data = _capture.to_dict()
        return mo.Html(
            f'<div class="saved"><b>Saved snapshot</b> · original prediction: {_data["prediction"]}<br><small>Track {_data["track"]}; later live controls do not rewrite this record.</small></div>'
        )

    def build_part_a():
        _intro = mo.md(f"### A · What actually binds this path? (10 min)\nThe initial probe runs batch **{a_batch.value}**. Predict its dominant exposed term before opening the instrument. Fleet-average utilization cannot identify this local bottleneck.")
        if a_prediction.value is None:
            return mo.vstack([_intro, a_prediction])
        _figure = go.Figure()
        for _run_name, _result in (("Batch 1", a_baseline), (f"Batch {a_batch.value}", a_result)):
            for _label, _term, _color in (
                ("Compute ceiling", _result.compute_time, COLORS["OrangeLine"]),
                ("Movement ceiling", _result.movement_time, COLORS["BlueLine"]),
                ("Launch overhead", _result.launch_time, COLORS["GreenLine"]),
            ):
                _figure.add_bar(name=_label, x=[_run_name], y=[_term.to("ms").magnitude], marker_color=_color, legendgroup=_label, showlegend=_run_name == "Batch 1")
        _figure.update_layout(barmode="group", height=290, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Path term (ms)", legend_orientation="h")
        _rows = [
            {"Run": "Baseline", "Batch": 1, "Latency": q(a_baseline.latency, "ms"), "Intensity": q(a_baseline.arithmetic_intensity, "flop/byte"), "Binding term": a_baseline.bottleneck},
            {"Run": "Changed", "Batch": a_batch.value, "Latency": q(a_result.latency, "ms"), "Intensity": q(a_result.arithmetic_intensity, "flop/byte"), "Binding term": a_result.bottleneck},
        ]
        return mo.vstack([
            _intro, a_prediction, a_batch, apply_plotly_theme(_figure), table(_rows),
            mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. **Observed analytical result:** the changed run is **{a_result.bottleneck}-bound** at **{q(a_result.latency, 'ms')}**. Batching changes fixed-tensor reuse and live state together."), kind="info"),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md("For each phase, compute time is operations divided by the supported effective rate; movement time is bytes divided by registry bandwidth. The exposed kernel time is the larger phase term. Sequential phase times, launches, and compile setup are then added. These are simulated analytical bounds, not measured percentiles.")}),
        ])

    def build_part_b():
        _intro = mo.md("### B · Which rewrite attacks the diagnosed term? (12 min)\nRank fusion, tiling, and their combination before opening the comparison. Every candidate starts from the same batch and precision.")
        if b_prediction.value is None:
            return mo.vstack([_intro, b_prediction])
        _baseline_result = b_comparison["baseline"]
        _figure = go.Figure([go.Bar(
            x=["Baseline", "Fusion", "Tiling", "Combined"],
            y=[_baseline_result.latency.to("ms").magnitude] + [b_alternatives[_name]["latency"]["magnitude"] for _name in ("fusion", "tiling", "combined")],
            marker_color=[COLORS["TextMuted"], COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"]],
        )])
        _figure.update_layout(height=285, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="End-to-end latency (ms)", showlegend=False)
        _candidate = b_comparison["result"]
        _rows = [
            {"Configuration": "Baseline", "Traffic": q(_baseline_result.total_traffic, _mem_unit), "Required memory": q(_baseline_result.memory_required, _mem_unit), "Latency": q(_baseline_result.latency, "ms")},
            {"Configuration": b_lever.value, "Traffic": q(_candidate.total_traffic, _mem_unit), "Required memory": q(_candidate.memory_required, _mem_unit), "Latency": q(_candidate.latency, "ms")},
        ]
        return mo.vstack([
            _intro, b_prediction, b_lever, apply_plotly_theme(_figure), table(_rows),
            mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. **Observed analytical result:** **{b_comparison['speedup']:.2f}×** speedup. The rewrite changes traffic and/or launches, while tiling and fusion reserve additional workspace."), kind="info"),
            b_capture, saved("B"),
            mo.accordion({"Calculation Notes": mo.md("Fusion retains fewer intermediate tensors in off-chip traffic and submits fewer launches. Tiling increases reuse of fixed tensors and adds workspace. The combined case rewrites the common phase path once; its speedup is recomputed and is never the product of standalone speedups.")}),
        ])

    def build_part_c():
        _intro = mo.md("### C · When does compilation pay for itself? (12 min)\nPredict whether setup, repeated runtime savings, or precision will determine the result. For device tracks (TinyML, Mobile, Edge), AOT compilation occurs on an engineering build host prior to deployment; for Cloud, it reflects server graph capture and JIT warm-up. Then select one supported precision and a reuse horizon.")
        if c_prediction.value is None:
            return mo.vstack([_intro, c_prediction])
        _figure = go.Figure([go.Bar(
            x=["1 run", "100 runs", "100,000 runs"],
            y=[c_reuse_results[_reuse].latency.to("ms").magnitude for _reuse in (1, 100, 100_000)],
            marker_color=[COLORS["RedLine"], COLORS["OrangeLine"], COLORS["GreenLine"]],
        )])
        _figure.update_layout(height=285, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Amortized latency per run (ms)", showlegend=False)
        _baseline_result = c_comparison["baseline"]
        _candidate = c_comparison["result"]
        _rows = [
            {"Configuration": "Carried rewrite", "Precision": _baseline_result.precision, "Quality fixture": f"{_baseline_result.quality_pct:.1f}%", "Compile/run": q(_baseline_result.compile_time_per_run, "ms"), "Latency": q(_baseline_result.latency, "ms")},
            {"Configuration": "Compiled candidate", "Precision": _candidate.precision, "Quality fixture": f"{_candidate.quality_pct:.1f}%", "Compile/run": q(_candidate.compile_time_per_run, "ms"), "Latency": q(_candidate.latency, "ms")},
        ]
        _kind = "success" if c_comparison["latency_improved"] else "danger"
        return mo.vstack([
            _intro, c_prediction, mo.hstack([c_precision, c_reuse], widths="equal", wrap=True),
            apply_plotly_theme(_figure), table(_rows), c_decision,
            mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. **Observed analytical result:** AOT/compile setup contributes **{q(_candidate.compile_time_per_run, 'ms')} per run** at this reuse horizon; total latency is **{q(_candidate.latency, 'ms')}**. Precision changes traffic, supported execution rate, and only the supplied quality fixture."), kind=_kind),
            c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md("Compile setup is divided by the declared reuse count and added to each run. For device tracks (TinyML, Mobile, Edge), AOT compilation is performed on an engineering build host prior to deployment; in cloud serving, it reflects one-time graph capture or JIT warmup. Captured graph execution lowers dispatch time but cannot remove kernel work. Precision selects a supported rate and byte width. Its quality value is fixed supplied scenario evidence, not an equation from bit width.")}),
        ])

    def build_part_d():
        _intro = mo.md(f"### D · Where does batching cross the memory boundary? (11 min)\nThe initial stress runs batch **{d_batch.value}**. Predict whether it fits one fleet unit. Fleet size can raise aggregate throughput, but it cannot pool memory across independent units or replicas.")
        if d_prediction.value is None:
            return mo.vstack([_intro, d_prediction])
        _ordered = list(d_alternatives.items())
        _to_mem = lambda p: quantity_from_dict(p).to(_mem_unit).magnitude
        _figure = go.Figure()
        _figure.add_bar(name="Required", x=[str(_batch) for _batch, _payload in _ordered], y=[_to_mem(_payload["memory_required"]) for _batch, _payload in _ordered], marker_color=COLORS["BlueLine"])
        _figure.add_bar(name="Per-unit capacity", x=[str(_batch) for _batch, _payload in _ordered], y=[_to_mem(_payload["memory_capacity"]) for _batch, _payload in _ordered], marker_color=COLORS["TextMuted"])
        _figure.update_layout(barmode="group", height=295, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Batch size", yaxis_title=f"Memory ({_mem_unit})", legend_orientation="h")
        _baseline_result = d_comparison["baseline"]
        _candidate = d_comparison["result"]
        _rows = [
            {"Run": "Carried configuration", "Batch": _baseline_result.batch_size, "Memory": f"{q(_baseline_result.memory_required, _mem_unit)} / {q(_baseline_result.memory_capacity, _mem_unit)}", "Attainable throughput": q(_baseline_result.attainable_per_unit_throughput, "1/s"), "Potential throughput": q(_baseline_result.potential_per_unit_throughput, "1/s"), "Outcome": "FIT" if _baseline_result.memory_feasible else "FAIL"},
            {"Run": "Stress", "Batch": _candidate.batch_size, "Memory": f"{q(_candidate.memory_required, _mem_unit)} / {q(_candidate.memory_capacity, _mem_unit)}", "Attainable throughput": q(_candidate.attainable_per_unit_throughput, "1/s"), "Potential throughput": q(_candidate.potential_per_unit_throughput, "1/s"), "Outcome": "FIT" if _candidate.memory_feasible else "FAIL · per-unit memory"},
        ]
        _kind = "success" if _candidate.memory_feasible else "danger"
        return mo.vstack([
            _intro, d_prediction, d_batch, apply_plotly_theme(_figure), table(_rows),
            mo.hstack([d_decision, d_rejected], widths="equal", wrap=True),
            mo.md("If you keep an earlier configuration, record **Adopt tested batch** as the rejected alternative so the saved contrast and decision agree."),
            mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. **Observed analytical result:** the stressed batch **{'fits' if _candidate.memory_feasible else 'does not fit'}** one {scenario.display_name} execution unit. The tested contrast is saved even when you recommend returning to the unoptimized selected-batch path."), kind=_kind),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md("Required memory is resident tensors plus batch-proportional live state plus fusion and tiling workspace. Attainable throughput is completed batch items divided by latency for memory-feasible runs, and 0 1/s when memory is exceeded; analytical potential throughput reflects hypothetical unconstrained execution. Fleet throughput multiplies the attainable rate by independent units only after each unit passes its own memory constraint.")}),
        ])

    def build_synthesis():
        _rows = []
        for _part in "ABCD":
            _capture = _captures.get(_part)
            _rows.append({
                "Part": _part,
                "Original prediction": _capture.to_dict()["prediction"] if _capture else "—",
                "Evidence": "CURRENT" if _capture and _part not in audit.stale and (_part, _part) not in audit.identical_pairs else ("STALE" if _capture else "MISSING"),
            })
        _d_saved = _captures["D"].to_dict()["decision"] if "D" in _captures else None
        _complete = (
            audit.complete
            and all(_widget.value is not None for _widget in (final_choice, final_rejected, final_trigger, final_risk))
            and bool(rationale.value.strip())
            and final_choice.value != final_rejected.value
            and final_choice.value == _d_saved
        )
        return mo.vstack([
            mo.md("### Synthesis · Defend one fleet recommendation (5 min)\nUse saved evidence to state the chosen configuration, quantify a rejected alternative, name one remaining limitation, and set a trigger for rerunning the experiments."),
            table(_rows),
            mo.callout(mo.md("Saved snapshots preserve original predictions, exact evaluator arguments, and unit-bearing results. Recapture only stale downstream evidence."), kind="info"),
            mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale,
            mo.callout(mo.md("**Ready for the local report.**" if _complete else "Complete four current contrasts, match the final choice to Part D, reject a different option, and add a quantified rationale."), kind="success" if _complete else "warn"),
        ])

    tabs = mo.ui.tabs({
        "Part A": build_part_a(), "Part B": build_part_b(),
        "Part C": build_part_c(), "Part D": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return (audit,)


# ZONE D · Report and volume-isolated ledger
@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id,
):
    _captures = get_evidence()
    _d_saved = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and all(_widget.value is not None for _widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip())
        and final_choice.value != final_rejected.value
        and final_choice.value == _d_saved
    )
    mo.stop(not _ready)
    _snapshots = {_part: _captures[_part].to_dict() for _part in "ABCD"}
    _constraint_results = {
        _part: (_snapshots[_part].get("chosen_result") or _snapshots[_part]["result"])
        for _part in "ABCD"
    }
    report = build_lab_report(
        get_lab_metadata("vol2/lab_09_perf_engineering.py"),
        track=track_id, scenario=scenario.workload,
        learning_objectives=[
            "Diagnose compute, memory, and overhead limits from a phase path.",
            "Compare fusion, tiling, precision, compilation, and batching from shared baselines.",
            "Defend a fleet configuration with per-unit memory and supplied quality evidence.",
        ],
        predictions={_part: _snapshots[_part]["prediction"] for _part in "ABCD"},
        knob_settings={_part: _snapshots[_part]["inputs"] for _part in "ABCD"},
        evidence_summary={_part: {"baseline": _snapshots[_part]["baseline"], "result": _snapshots[_part]["result"], "alternatives": _snapshots[_part]["alternatives"]} for _part in "ABCD"},
        binding_constraints={_part: {"bottleneck": _constraint_results[_part]["bottleneck"], "memory_feasible": _constraint_results[_part]["memory_feasible"]} for _part in "ABCD"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=[
            "The binding phase term determines which local rewrite can improve latency.",
            "Compile setup must be amortized, and combined gains must be recomputed on one path.",
            "Independent fleet units raise aggregate throughput without pooling per-unit memory.",
        ],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": _snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"hardware": scenario.hardware_name, "scenario": scenario.provenance, "calculations": "MLSysIM v2_09_experiments physical phase model"},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    MODEL_ID, audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _d_saved = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and all(_widget.value is not None for _widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip())
        and final_choice.value != final_rejected.value
        and final_choice.value == _d_saved
    )
    _saved = False
    _save_failed = False
    if _ready:
        try:
            ledger.save(chapter=9, design={
                "schema_version": 1, "lab_id": "v2_09", "track_id": track_id,
                "model_id": MODEL_ID,
                "evidence": {_part: _capture.to_dict() for _part, _capture in _captures.items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
            })
            await ledger.flush()
            _saved = True
        except Exception:
            _save_failed = True
    _status = "SAVED" if _saved else "SAVE FAILED · REPORT STILL AVAILABLE" if _save_failed else "EVIDENCE IN PROGRESS"
    _status_class = "hud-failed" if _save_failed else "hud-active"
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">09 · Optimize the Path</span><span class="hud-separator">|</span><span style="flex:1"></span><span class="hud-label">STATUS</span><span class="{_status_class}">{_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
