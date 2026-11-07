import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="The Same-Bits Contract · MLSysBook")


# ZONE A · SETUP
@app.cell
async def _():
    import sys
    from pathlib import Path
    import marimo as mo
    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas", "numpy"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.3.1-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)
    import plotly.graph_objects as go
    from mlsysim.hardware.registry import Hardware
    from mlsysim.labs.experiments.same_bits_experiments import (
        MODEL_ID, ORDERS, addition_orders, batch_contrast, contract_snapshot,
        extra_reduction_adds, get_scenario, integer_agreement, order_snapshot,
        serving_contrast,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, LabMetadata, build_lab_report, report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence
    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, Hardware, LAB_CSS, LabMetadata, MODEL_ID, ORDERS,
        addition_orders, apply_plotly_theme, audit_evidence, batch_contrast,
        build_lab_report, capture_evidence, contract_snapshot, extra_reduction_adds,
        get_scenario, go,
        integer_agreement, ledger, mo, order_snapshot, report_export_panel,
        serving_contrast,
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
def _(get_scenario, track):
    track_id = track.value
    scenario = get_scenario(track_id)
    return scenario, track_id


@app.cell
def _(Hardware):
    track_devices = {
        "TinyML": Hardware.Tiny.OuraRing,
        "Mobile": Hardware.Mobile.iPhone15Pro,
        "Edge": Hardware.Edge.RoboTaxi,
        "Cloud": Hardware.Cloud.H100,
    }
    return (track_devices,)


# ZONE B · WIDGETS
@app.cell
def _(ORDERS, mo, scenario, track_id):
    _track_key = track_id
    _batches = {f"Batch of {b}": b for b in scenario.batch_options}
    _default = f"Batch of {scenario.default_batch}"
    a_order = mo.ui.dropdown(
        {order.capitalize(): order for order in ORDERS},
        value="Pairwise tree", label="Summation order to compare with left to right",
    )
    b_batch = mo.ui.dropdown(_batches, value=_default, label="Batch the row joins")
    c_batch = mo.ui.dropdown(_batches, value=_default, label="Serving batch size")
    d_batch = mo.ui.dropdown(_batches, value=_default, label="Serving batch size")
    e_batch = mo.ui.dropdown(_batches, value=_default, label="Serving batch size")
    return a_order, b_batch, c_batch, d_batch, e_batch


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"One: addition is addition": "one", "Two or three": "few", "Four or five": "many"},
        label="Five orders add the same 4,096 float32 numbers. How many different results?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"None of them": "none", "A few (under 10%)": "few", "Most of them": "most"},
        label="One row of a projection, alone and then inside a batch: how many of its 128 outputs change bits?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"None: temperature 0 is deterministic": "none", "A few": "few", "Most of them": "most"},
        label="64 users, temperature 0, served alone and then batched: how many see different text?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Text still changes": "changes", "No text changes, at no cost": "free", "No text changes, but large batches do extra work": "costly"},
        label="The kernel now uses one fixed reduction split whatever the batch. What happens?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Nothing": "nothing", "Some agreement with the float model": "agreement", "Determinism across batches": "determinism"},
        label="int8 products with exact int32 sums: what do you give up?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    _contracts = {
        "Throughput kernels (accept changing bits)": "throughput",
        "Batch-invariant float kernels": "invariant",
        "Integer arithmetic": "integer",
    }
    final_contract = mo.ui.radio(_contracts, label="Contract you ship")
    final_rejected = mo.ui.radio(_contracts, label="Quantified rejected contract")
    final_trigger = mo.ui.radio(
        {"A kernel or library update changes a reduction order": "kernel",
         "Quality falls below the float reference": "quality",
         "Batches grow past the fixed split's work budget": "budget"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Bits can still differ across hardware generations": "hardware",
         "Quantization error on rare inputs": "quantization",
         "Extra partial-sum work at peak load": "cost"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Cite saved numbers, the rejected contract, the limitation, and the trigger.",
    )
    return final_contract, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_order, b_batch, batch_contrast, c_batch, contract_snapshot, d_batch,
    e_batch, extra_reduction_adds, integer_agreement, order_snapshot, scenario,
    serving_contrast,
):
    _dtype = scenario.dtype
    a_base = order_snapshot("left to right")
    a_result = order_snapshot(a_order.value)
    b_base = batch_contrast(1, dtype=_dtype)
    b_result = batch_contrast(b_batch.value, dtype=_dtype)
    c_base = serving_contrast(1, dtype=_dtype)
    c_result = serving_contrast(c_batch.value, dtype=_dtype)
    c_sweep = [serving_contrast(b, dtype=_dtype) for b in scenario.batch_options]
    d_base = contract_snapshot(d_batch.value, "float", "throughput", _dtype)
    d_result = contract_snapshot(d_batch.value, "float", "invariant", _dtype)
    d_sweep = [
        (b, extra_reduction_adds(b, "throughput"), extra_reduction_adds(b, "invariant"))
        for b in scenario.batch_options
    ]
    e_base = contract_snapshot(e_batch.value, "float", "throughput", _dtype)
    e_result = contract_snapshot(e_batch.value, "integer", "throughput", _dtype)
    e_quality = integer_agreement()
    return (
        a_base, a_result, b_base, b_result, c_base, c_result, c_sweep,
        d_base, d_result, d_sweep, e_base, e_quality, e_result,
    )


@app.cell
def _(
    a_base, a_order, a_prediction, a_result, b_base, b_batch, b_prediction,
    b_result, c_base, c_batch, c_prediction, c_result, capture_evidence,
    d_base, d_batch, d_prediction, d_result, e_base, e_batch, e_prediction,
    e_quality, e_result, mo, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})
    upstream = {"scenario": track_id}
    a_capture = mo.ui.button(
        label="Capture summation-order contrast", kind="success",
        disabled=a_prediction.value is None or a_order.value == "left to right",
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"order": a_order.value}, baseline=a_base, result=a_result,
            upstream_inputs=upstream, model_key="same_bits_experiments.order_snapshot",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture alone-versus-batched contrast", kind="success",
        disabled=b_prediction.value is None or b_batch.value == 1,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"batch": b_batch.value}, baseline=b_base, result=b_result,
            upstream_inputs=upstream, model_key="same_bits_experiments.batch_contrast",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture serving contrast", kind="success",
        disabled=c_prediction.value is None or c_batch.value == 1,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"batch": c_batch.value}, baseline=c_base, result=c_result,
            upstream_inputs=upstream, model_key="same_bits_experiments.serving_contrast",
        )),
    )
    d_capture = mo.ui.button(
        label="Capture batch-invariant contrast", kind="success",
        disabled=d_prediction.value is None or d_batch.value == 1,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"batch": d_batch.value}, baseline=d_base, result=d_result,
            upstream_inputs=upstream, model_key="same_bits_experiments.contract_snapshot",
        )),
    )
    e_capture = mo.ui.button(
        label="Capture integer contrast", kind="success",
        disabled=e_prediction.value is None or e_batch.value == 1,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"batch": e_batch.value}, baseline=e_base,
            result={**e_result, "quality": e_quality},
            upstream_inputs=upstream, model_key="same_bits_experiments.contract_snapshot",
        )),
    )
    return a_capture, b_capture, c_capture, d_capture, e_capture, upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, scenario, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}.pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}.pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:760px}.pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}.pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.stakeholder{border-left:4px solid #1d4f78;background:#f1f5f9;padding:10px 14px;border-radius:7px;line-height:1.5}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;column-gap:14px;row-gap:6px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf;margin-right:2px}.lab-hud .hud-value{color:#fff;margin-right:10px}.lab-hud .hud-active{color:#86efac}@media(max-width:520px){.pilot-head{border-radius:9px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>PROPOSED CO-LAB · INFERENCE</span><span>ABOUT 45–50 MIN</span></div><h1>The Same-Bits Contract</h1><p>You send the same prompt at temperature 0, twice, and get two different answers. Nothing was sampled. Where did the difference come from, and what does it cost to remove it?</p><div class="pilot-meta"><div><b>Device</b><br>{scenario.device_name}</div><div><b>Stakeholder</b><br>{scenario.stakeholder}</div><div><b>Number format</b><br>{scenario.dtype}</div><div><b>Deliverable</b><br>A determinism contract</div></div></section>""")
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, track.style({"margin-top": "30px", "margin-left": "24px", "max-width": "calc(100% - 24px)"}), header, mo.Html('<p class="pilot-note">Every number is computed with real IEEE 754 float16 and float32 arithmetic, rounded after each operation, with every reduction order written out. The language model is a small seeded toy, so counts illustrate the mechanism rather than measure a production model.</p>')], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


# ZONE C · SINGLE TAB SET
@app.cell
def _(
    COLORS, a_base, a_capture, a_order, a_prediction, a_result,
    addition_orders, apply_plotly_theme, audit_evidence, b_base, b_batch,
    b_capture, b_prediction, b_result, c_batch, c_capture, c_prediction,
    c_result, c_sweep, d_base, d_batch, d_capture, d_prediction, d_result,
    d_sweep, e_base, e_batch, e_capture, e_prediction, e_quality, e_result,
    final_contract, final_rejected, final_risk, final_trigger, get_evidence,
    go, mo, rationale, scenario, track_devices, track_id, upstream,
):
    _captures = get_evidence()
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        upstream_inputs=upstream, contrast_required_parts=tuple("ABCDE"),
    )
    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})
    def stakeholder(text):
        return mo.Html(f'<div class="stakeholder"><b>{scenario.stakeholder}:</b> {text}</div>')
    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Review and recapture."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later control changes do not rewrite this record.</small></div>')

    def build_part_a():
        intro = mo.md("### A · One sum, five orders (8 min)\n**Driving question:** Does the order of additions change a floating-point sum?")
        if a_prediction.value is None:
            return mo.vstack([intro, a_prediction])
        _all = addition_orders()
        rows = [{"Order": r["order"], "float32 result": f'{r["value"]:.9g}', "Bits (digest)": r["bits"], "ULPs from left to right": r["ulps_from_left_to_right"]} for r in _all["rows"]]
        return mo.vstack([intro, a_prediction, table(rows), mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. **Measured:** **{_all['distinct']}** distinct float32 results from 5 orders. The float64 sum is {_all['exact']:.9g}; no order is wrong, each rounds differently."), kind="info"), a_order, mo.md(f"Selected: **{a_result['value']:.9g}** ({a_result['ulps_from_left_to_right']} ULPs from left to right, {a_base['value']:.9g})."), a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("Each order rounds to float32 after every addition. Blocked orders sum contiguous chunks, then add the partials left to right, which is the shape of a parallel reduction. A ULP is one step between adjacent representable floats.")})])

    def build_part_b():
        intro = mo.md("### B · The batch picks the kernel (10 min)\n**Driving question:** Does a row's result depend on which other rows share its batch?")
        if b_prediction.value is None:
            return mo.vstack([intro, b_prediction])
        rows = [
            {"Run": "Alone (batch 1)", "Reduction splits": b_base["splits_batched"], "Bits (digest)": b_base["digest_batched"], "Outputs changed": "—"},
            {"Run": f"Batch of {b_batch.value}", "Reduction splits": b_result["splits_batched"], "Bits (digest)": b_result["digest_batched"], "Outputs changed": f'{b_result["outputs_differing"]} / {b_result["outputs"]}'},
        ]
        _changed = b_result["outputs_differing"] > 0
        return mo.vstack([intro, stakeholder(scenario.why_bits_matter), b_batch, b_prediction, table(rows), mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. **Measured:** **{b_result['outputs_differing']} of {b_result['outputs']}** outputs change bits, by up to {b_result['max_ulps']} ULPs. Row 0's numbers did not change; only the kernel the batch selected did."), kind="danger" if _changed else "success"), b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md("The kernel splits the reduction (K) dimension into chunks when the batch is too small to fill the processor, then adds the partial sums. Fewer splits for larger batches is a throughput choice; it also changes the order in which every output is summed.")})])

    def build_part_c():
        intro = mo.md("### C · From bits to words (10 min)\n**Driving question:** Do ULP-sized differences ever change greedy, temperature-0 text?")
        if c_prediction.value is None:
            return mo.vstack([intro, c_prediction])
        fig = go.Figure([go.Bar(x=[f"batch {s['inputs']['batch']}" for s in c_sweep], y=[s["changed"] for s in c_sweep], marker_color=COLORS["RedLine"])])
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=25, b=20), yaxis_title="Users whose text changed (of 64)")
        examples = [{"User": e["user"], "First different token": e["first_divergent_token"], "Alone": e["alone"][:48], "Batched": e["batched"][:48]} for e in c_result["examples"]]
        _bad = c_result["changed"] > 0
        return mo.vstack([intro, c_batch, c_prediction, apply_plotly_theme(fig), table(examples) if examples else mo.md("No user's text changed at this batch size."), mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. **Measured:** **{c_result['changed']} of {c_result['users']}** users get different text at batch {c_batch.value}. {'This is a reproducibility FAIL for the ' + scenario.stakeholder.lower() + '.' if _bad else 'This batch uses the same kernel split as batch 1.'}"), kind="danger" if _bad else "success"), c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("A seeded recurrent toy model decodes 40 tokens greedily for 64 prompts. Only the output projection changes kernel with batch size. Text changes only where two logits are close enough for a few ULPs to swap the argmax; after that token, the sequences diverge.")})])

    def build_part_d():
        intro = mo.md("### D · Batch-invariant kernels (9 min)\n**Driving question:** What does a kernel that ignores batch size cost?")
        if d_prediction.value is None:
            return mo.vstack([intro, d_prediction])
        fig = go.Figure([
            go.Bar(name="Throughput split", x=[f"batch {b}" for b, _t, _i in d_sweep], y=[t for _b, t, _i in d_sweep], marker_color=COLORS["BlueLine"]),
            go.Bar(name="Fixed split", x=[f"batch {b}" for b, _t, _i in d_sweep], y=[i for _b, _t, i in d_sweep], marker_color=COLORS["OrangeLine"]),
        ])
        fig.update_layout(barmode="group", height=280, margin=dict(l=20, r=20, t=25, b=20), yaxis_title="Partial-sum additions per projection", legend_orientation="h")
        rows = [
            {"Kernel": "Throughput split", "Users with changed text": f'{d_base["changed"]} / {d_base["users"]}', "Partial-sum additions": d_base["extra_reduction_adds"]},
            {"Kernel": "Fixed split", "Users with changed text": f'{d_result["changed"]} / {d_result["users"]}', "Partial-sum additions": d_result["extra_reduction_adds"]},
        ]
        return mo.vstack([intro, d_batch, d_prediction, table(rows), apply_plotly_theme(fig), mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. **Measured at batch {d_batch.value}:** changed users fall from **{d_base['changed']}** to **{d_result['changed']}**, while partial-sum additions go from {d_base['extra_reduction_adds']:,} to {d_result['extra_reduction_adds']:,}."), kind="info"), d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("The fixed split uses 4 chunks for every batch, so each output is always summed in the same order. Its cost is the extra additions that combine partial sums at batch sizes where a throughput kernel would not split. Results are exact for this kernel; real kernels also lose some parallelism.")})])

    def build_part_e():
        intro = mo.md("### E · Integer arithmetic (8 min)\n**Driving question:** Can exact integer sums make any batch, and any order, give the same bits?")
        if e_prediction.value is None:
            return mo.vstack([intro, e_prediction])
        rows = [
            {"Arithmetic": f"Float ({scenario.dtype}), throughput split", "Users with changed text": f'{e_base["changed"]} / {e_base["users"]}'},
            {"Arithmetic": "int8 products, exact int32 sums", "Users with changed text": f'{e_result["changed"]} / {e_result["users"]}'},
        ]
        devices = []
        for _label, _hw in track_devices.items():
            _rate = _hw.compute.precision_flops.get("int8")
            devices.append({"Track": _label, "Device": _hw.name, "Listed int8 rate": f"{_rate.m_as('teraflop / second'):g} T ops/s" if _rate is not None else "Not listed"})
        return mo.vstack([intro, e_batch, e_prediction, table(rows), mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. **Measured:** integer arithmetic changes **{e_result['changed']}** users at batch {e_batch.value}, at any split. The price is agreement with the float model: **{e_quality['token_agreement']:.1%}** of tokens and {e_quality['identical_sequences']} of {e_quality['users']} full sequences match a float32 run."), kind="info"), mo.md("**Integer support on each track's device (MLSys·im registry):**"), table(devices), e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md("Hidden states are quantized per row and weights per column to int8. int32 accumulation of int8 products is exact at this width, so addition order cannot change the sum. Agreement is measured against an unbatched float32 decode of the same model.")})])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            rows.append({"Part": part, "Original prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING")})
        _complete = audit.complete and all(w.value is not None for w in (final_contract, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_contract.value != final_rejected.value
        return mo.vstack([mo.md(f"### Synthesis · Choose a determinism contract (5 min)\nAdvise the **{scenario.stakeholder.lower()}**: choose a contract, quantify a rejected one, state a limitation, and define a reevaluation trigger."), stakeholder(scenario.why_bits_matter), table(rows), mo.hstack([final_contract, final_rejected], widths="equal", wrap=True), mo.hstack([final_risk, final_trigger], widths="equal", wrap=True), rationale, mo.callout(mo.md("**Ready for the ledger and report.**" if _complete else "Complete five current captures, reject a different contract, and write the rationale."), kind="success" if _complete else "warn")])

    tabs = mo.ui.tabs({"Part A": build_part_a(), "Part B": build_part_b(), "Part C": build_part_c(), "Part D": build_part_d(), "Part E": build_part_e(), "Synthesis": build_synthesis()})
    tabs
    return (audit,)


@app.cell
def _(
    LabMetadata, MODEL_ID, audit, build_lab_report, final_contract,
    final_rejected, final_risk, final_trigger, get_evidence, mo, rationale,
    report_export_panel, scenario, track_id,
):
    _captures = get_evidence()
    _ready = audit.complete and all(w.value is not None for w in (final_contract, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_contract.value != final_rejected.value
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    metadata = LabMetadata(
        lab_id="proposed_same_bits", title="The Same-Bits Contract",
        volume="Proposed", chapter="Inference", book_anchor="Proposed Co-Lab: inference determinism",
        updated_at="2026-11-04",
    )
    report = build_lab_report(
        metadata, track=track_id, scenario=scenario.stakeholder,
        learning_objectives=["Show that floating-point reduction order changes results", "Trace batch-dependent kernels to changed temperature-0 text", "Choose and defend a determinism contract with its cost"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"]} for part in "ABCDE"},
        binding_constraints={"changed_users_batched": snapshots["C"]["result"]["changed"], "fixed_split_additions": snapshots["D"]["result"]["extra_reduction_adds"], "integer_token_agreement": snapshots["E"]["result"]["quality"]["token_agreement"]},
        decisions={"contract": final_contract.value, "rejected_contract": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"contract": final_contract.value, "rejected_contract": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["Temperature 0 removes sampling, not arithmetic nondeterminism.", "Batch size selects the kernel, and the kernel selects the reduction order.", "Same bits are a contract with a price: extra work or lower agreement with float."],
        reflections={"rationale": rationale.value, "trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"schema_version": 1, "lab_id": "proposed_same_bits", "track_id": track_id, "model_id": MODEL_ID, "evidence": snapshots, "contract": final_contract.value, "rejected_contract": final_rejected.value, "reevaluation_trigger": final_trigger.value, "residual_risk": final_risk.value, "rationale": rationale.value},
        source_trace={"scenario": "Seeded toy model and explicit-order IEEE 754 kernels.", "calculations": "MLSysIM same_bits experiment models."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


# ZONE D · COMPLETION-GATED LEDGER
@app.cell
async def _(
    MODEL_ID, audit, final_contract, final_rejected, final_risk,
    final_trigger, get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _ready = audit.complete and all(w.value is not None for w in (final_contract, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_contract.value != final_rejected.value
    _saved_ok = False
    _save_error = ""
    if _ready:
        try:
            ledger.save(step="same_bits", design={
                "schema_version": 1, "lab_id": "proposed_same_bits", "track_id": track_id,
                "model_id": MODEL_ID,
                "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "contract": final_contract.value,
                "rejected_contract": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
            })
            await ledger.flush()
            _saved_ok = True
        except Exception as exc:
            _save_error = f"SAVE FAILED · {type(exc).__name__}: {exc}"
    _status = "SAVED" if _saved_ok else (_save_error or "EVIDENCE IN PROGRESS")
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">The Same-Bits Contract</span><span aria-hidden="true">|</span><span style="flex:1"></span><span class="hud-label">STATUS</span><span class="hud-active">{_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
