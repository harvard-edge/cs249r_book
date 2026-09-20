import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 02: The Compute Infrastructure Wall · MLSysBook")


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
    from mlsysim.engine.v2_02_experiments import (
        MODEL_ID, compare_systems, compare_transfer_tiers, evaluate_facility,
        evaluate_memory_scale, evaluate_roofline, evaluate_transfer_tier,
        get_facility_count_policy, get_track_config,
        get_transfer_tier_options,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata, report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, MODEL_ID, Q_, apply_plotly_theme,
        audit_evidence, build_lab_report, capture_evidence, compare_systems,
        compare_transfer_tiers, evaluate_facility, get_facility_count_policy,
        get_transfer_tier_options, evaluate_memory_scale,
        evaluate_roofline, evaluate_transfer_tier, get_lab_metadata,
        get_track_config, go, ledger, mo, report_export_panel,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Training-infrastructure track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_track_config, track):
    track_id = track.value
    profile = get_track_config(track_id)
    return profile, track_id


@app.cell
def _(get_facility_count_policy, get_transfer_tier_options, mo, profile, track_id):
    _track_key = track_id
    a_intensity = mo.ui.dropdown(
        {"100 FLOP/byte": 100, "300 FLOP/byte": 300, "600 FLOP/byte": 600, "1,000 FLOP/byte": 1000},
        value="1,000 FLOP/byte", label="Changed arithmetic intensity",
    )
    b_scale = mo.ui.slider(1.1, 2.0, value=1.5, step=0.1, label="Working-set and traffic multiplier")
    _c_tiers = get_transfer_tier_options(track_id)
    c_destination = mo.ui.dropdown(
        _c_tiers,
        value=profile.network_tier_name if profile.network_tier_name in _c_tiers else list(_c_tiers.values())[0],
        label="Changed placement tier",
    )
    _d_policy = get_facility_count_policy(track_id)
    d_count = mo.ui.slider(
        _d_policy["min_count"], _d_policy["max_count"],
        value=_d_policy["default_count"], step=_d_policy["step"],
        label="Accelerators installed",
    )
    e_duty = mo.ui.slider(0.4, 0.9, value=0.8, step=0.1, label="Expected active duty cycle")
    return a_intensity, b_scale, c_destination, d_count, e_duty


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Bandwidth-rich option still wins": "bandwidth", "Higher peak-compute option takes the lead": "compute", "The ranking stays tied": "tie"},
        label="At the changed intensity, which outcome do you expect?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Fits; memory bandwidth sets the floor": "fits_memory", "Fits; compute sets the floor": "fits_compute", "Capacity fails before latency is usable": "capacity"},
        label="What will the enlarged workload expose?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Less than 2× slower": "under_2x", "2–10× slower": "2_10x", "More than 10× slower": "over_10x"},
        label="How much slower will the changed placement be?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Both limits pass": "pass", "Electrical fails first": "electrical", "Cooling fails first": "cooling", "Both fail": "both"},
        label="Which facility boundary will the installation cross?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Lower purchase cost wins": "purchase", "Higher sustained throughput wins": "throughput", "No candidate remains feasible": "none"},
        label="What will dominate the purchase decision?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    _first, _second = profile.procurement_devices
    e_decision = mo.ui.radio({_first: _first, _second: _second, "No feasible purchase": "none"}, label="Purchase decision")
    e_rejected = mo.ui.radio({_first: _first, _second: _second}, label="Quantified rejected alternative")
    final_choice = mo.ui.radio({_first: _first, _second: _second, "No feasible purchase": "none"}, label="Final recommendation")
    final_rejected = mo.ui.radio({_first: _first, _second: _second}, label="Rejected alternative")
    final_trigger = mo.ui.radio(
        {"Working set exceeds memory": "memory", "Facility draw exceeds its limit": "facility", "Duty cycle changes by 20 points": "duty"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Roofline omits software overhead": "software", "Transfer model omits contention": "contention", "Cost omits staffing and construction": "cost_boundary"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Evidence-based rationale",
        placeholder="Cite the selected option, one number for the rejected alternative, the limitation, and the trigger.",
    )
    return e_decision, e_rejected, final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    Q_, a_intensity, b_scale, c_destination, compare_systems,
    compare_transfer_tiers, d_count, e_duty, evaluate_facility,
    evaluate_memory_scale, evaluate_roofline, evaluate_transfer_tier, profile, track_id,
):
    a_base = evaluate_roofline(track_id, Q_("10 flop/byte"))
    a_result = evaluate_roofline(track_id, Q_(a_intensity.value, "flop/byte"))
    b_base = evaluate_memory_scale(track_id, scale=1.0)
    b_result = evaluate_memory_scale(track_id, scale=b_scale.value)
    c_all = compare_transfer_tiers(track_id)
    c_base = evaluate_transfer_tier(track_id, tier=c_all["baseline_tier"])
    c_result = evaluate_transfer_tier(track_id, tier=c_destination.value)
    d_base = evaluate_facility(track_id, accelerator_count=profile.accelerator_count)
    d_result = evaluate_facility(track_id, accelerator_count=d_count.value)
    e_base = compare_systems(track_id, arithmetic_intensity=Q_("1000 flop/byte"), duty_cycle=0.25)
    e_result = compare_systems(track_id, arithmetic_intensity=Q_("1000 flop/byte"), duty_cycle=e_duty.value)
    return a_base, a_result, b_base, b_result, c_all, c_base, c_result, d_base, d_result, e_base, e_result


@app.cell
def _(
    a_base, a_prediction, a_result, b_base, b_prediction, b_result, c_base,
    c_prediction, c_result, capture_evidence, d_base, d_prediction, d_result,
    e_base, e_decision, e_prediction, e_rejected, e_result, mo, profile,
    set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"devices": profile.roofline_devices}
    b_upstream = {"device": profile.memory_device}
    c_upstream = {"device": profile.memory_device, "payload_gib": c_base["payload_gib"]}
    d_upstream = {"device": profile.memory_device, "facility": profile.label}
    e_upstream = {"candidates": profile.procurement_devices, "workload": profile.label}

    def capture(part, prediction, baseline, result, upstream, decision, alternatives=()):
        return capture_evidence(
            track=track_id, part=part, prediction=prediction,
            inputs={"baseline": baseline["inputs"], "result": result["inputs"]},
            baseline=baseline, result=result, upstream_inputs=upstream,
            alternatives=alternatives, decision=decision, model_key=result["model_key"],
        )

    a_capture = mo.ui.button(
        label="Capture Roofline contrast", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture("A", a_prediction.value, a_base, a_result, a_upstream, a_result["winner_device_key"], a_result["rows"])),
    )
    b_capture = mo.ui.button(
        label="Capture memory contrast", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _v: store("B", capture("B", b_prediction.value, b_base, b_result, b_upstream, "fit" if b_result["fits"] else "capacity failure")),
    )
    c_capture = mo.ui.button(
        label="Capture placement contrast", kind="success", disabled=c_prediction.value is None,
        on_click=lambda _v: store("C", capture("C", c_prediction.value, c_base, c_result, c_upstream, c_result["tier"])),
    )
    d_capture = mo.ui.button(
        label="Capture facility boundary", kind="success", disabled=d_prediction.value is None,
        on_click=lambda _v: store("D", capture("D", d_prediction.value, d_base, d_result, d_upstream, "feasible" if d_result["feasible"] else "facility limit")),
    )
    _e_invalid = e_prediction.value is None or e_decision.value is None or e_rejected.value is None or e_decision.value == e_rejected.value
    e_capture = mo.ui.button(
        label="Capture procurement decision", kind="success", disabled=_e_invalid,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"baseline": e_base["inputs"], "result": e_result["inputs"], "decision": e_decision.value, "rejected": e_rejected.value},
            baseline=e_base, result=e_result, upstream_inputs=e_upstream,
            alternatives=e_result["rows"], decision=e_decision.value,
            model_key=e_result["model_key"],
            chosen_result=e_base if e_decision.value == "none" else None,
            result_role="rejected alternative" if e_decision.value == "none" else "tested intervention",
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}.pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}.pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:780px}.pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}.track-outside{border:1px solid #cbd5e1;background:#f8fafc;border-radius:10px;padding:10px 14px;margin-bottom:8px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}@media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME II · LAB 02</span><span>ABOUT 50–55 MIN</span></div><h1>The Compute Infrastructure Wall</h1><p>When does peak compute stop predicting useful capacity, and which accelerator–node–rack design survives the full constraint cascade?</p><div class="pilot-meta"><div><b>Track</b><br>{profile.label}</div><div><b>Fleet context</b><br>{profile.infrastructure_role}</div><div><b>Output</b><br>Evidence-backed infrastructure decision</div></div></section>""")
    note = mo.md("Hardware and transfer-tier specifications come from the course registry. Workload sizes, facility envelopes, duty cycles, and efficiency are **illustrative scenario assumptions**. Each part states omitted effects.")
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, mo.Html('<div class="track-outside"><b>Choose the deployment workload whose training infrastructure you will design.</b></div>'), track, header, note], gap=0.5).style({"padding-top": "32px"})
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base, a_capture, a_intensity, a_prediction, a_result, a_upstream,
    apply_plotly_theme, audit_evidence, b_base, b_capture, b_prediction,
    b_result, b_scale, b_upstream, c_all, c_base, c_capture, c_destination,
    c_prediction, c_result, c_upstream, d_base, d_capture, d_count,
    d_prediction, d_result, d_upstream, e_base, e_capture, e_decision, e_duty,
    e_prediction, e_rejected, e_result, e_upstream, final_choice,
    final_rejected, final_risk, final_trigger, get_evidence, go, mo, profile,
    rationale, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(_captures, track=track_id, required_parts=tuple("ABCDE"), per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCDE"))

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        item = _captures.get(part)
        if item is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this part."), kind="danger")
        snap = item.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {snap["prediction"]}<br><small>Later control changes do not rewrite this evidence.</small></div>')

    def build_part_a():
        intro = mo.md(f"### A · When does the faster accelerator fail to help? (9 min)\nFor the **{profile.infrastructure_role}**, evaluate candidate training devices. The options trade peak compute against memory bandwidth. Predict the changed ranking before opening the instrument.")
        preview = table([{"Candidate": r["device_name"], "Peak compute": f"{r['peak_tflops']:.0f} TFLOP/s", "Memory bandwidth": f"{r['memory_bandwidth_gb_s']:.0f} GB/s"} for r in a_base["rows"]])
        if a_prediction.value is None:
            return mo.vstack([intro, preview, a_intensity, a_prediction])
        fig = go.Figure()
        for label, result in (("10 FLOP/byte", a_base), (f"{a_intensity.value} FLOP/byte", a_result)):
            fig.add_bar(name=label, x=[r["device_name"] for r in result["rows"]], y=[r["useful_tflops"] for r in result["rows"]])
        fig.update_layout(barmode="group", height=285, yaxis_title="Roofline ceiling (TFLOP/s)", legend_orientation="h")
        rows = [{"Run": "Baseline", "Intensity": 10, "Winner": a_base["winner_device_key"]}, {"Run": "Changed", "Intensity": a_intensity.value, "Winner": a_result["winner_device_key"]}]
        reveal = mo.callout(mo.md(f"**Prediction:** {a_prediction.value}. **Analytical result:** {a_base['winner_device_key']} wins at 10 FLOP/byte; {a_result['winner_device_key']} wins at {a_intensity.value} FLOP/byte."), kind="info")
        notes = mo.accordion({"Calculation Notes": mo.md("Roofline throughput = min(peak compute, memory bandwidth × arithmetic intensity). This upper bound excludes software efficiency, launch overhead, and contention.")})
        return mo.vstack([intro, a_intensity, a_prediction, apply_plotly_theme(fig), table(rows), reveal, a_capture, saved("A"), notes])

    def build_part_b():
        b_device = b_base.get("device_name", "the training accelerator")
        intro = mo.md(f"### B · Does the workload fit, and what sets its latency floor? (9 min)\nIn the **{profile.infrastructure_role}**, capacity decides whether execution is possible on {b_device}. The baseline is **{b_base['working_set_gib']:.1f} GiB**; the selected enlargement is **{b_result['working_set_gib']:.1f} GiB** against **{b_base['capacity_gib']:.1f} GiB** of registry memory.")
        if b_prediction.value is None:
            return mo.vstack([intro, b_scale, b_prediction])
        fig = go.Figure([go.Bar(name="Working set", x=["Baseline", "Enlarged"], y=[b_base["working_set_gib"], b_result["working_set_gib"]]), go.Bar(name="Capacity", x=["Baseline", "Enlarged"], y=[b_base["capacity_gib"], b_result["capacity_gib"]])])
        fig.update_layout(barmode="group", height=275, yaxis_title="Memory (GiB)", legend_orientation="h")
        rows = [{"Run": "Baseline", "Fit": b_base["fits"], "Bandwidth floor": f"{b_base['bandwidth_floor_ms']:.1f} ms"}, {"Run": f"{b_scale.value:.1f}×", "Fit": b_result["fits"], "Bandwidth floor": f"{b_result['bandwidth_floor_ms']:.1f} ms"}]
        outcome = f"fits; {b_result['bottleneck']} binds" if b_result["fits"] else f"fails capacity by {-b_result['headroom_gib']:.1f} GiB"
        reveal = mo.callout(mo.md(f"**Prediction:** {b_prediction.value}. **Result:** the enlarged workload {outcome}."), kind="success" if b_result["fits"] else "danger")
        notes = mo.accordion({"Calculation Notes": mo.md("Fit requires working set ≤ capacity. For a fit, latency floor = max(operations ÷ peak compute, bytes moved ÷ memory bandwidth).")})
        return mo.vstack([intro, b_scale, b_prediction, apply_plotly_theme(fig), table(rows), reveal, b_capture, saved("B"), notes])

    def build_part_c():
        c_device = c_all.get("device_name", "the training accelerator")
        intro = mo.md(f"### C · Where should communicating work reside? (9 min)\nFor the **{profile.infrastructure_role}**, evaluate placement for {c_device}. Move the identical payload from baseline **{c_base['tier']}** to another tier. Placement bandwidth is the only changed cause.")
        preview = table([{"Tier": r["tier"], "Bandwidth": f"{r['bandwidth_gb_s']:.1f} GB/s"} for r in c_all["rows"]])
        if c_prediction.value is None:
            return mo.vstack([intro, preview, c_destination, c_prediction])
        rows = [{"Tier": r["tier"], "Bandwidth": f"{r['bandwidth_gb_s']:.1f} GB/s", "Transfer": f"{r['transfer_ms']:.1f} ms"} for r in c_all["rows"]]
        slowdown = c_result["slowdown_vs_baseline"]
        fig = go.Figure([go.Bar(x=[c_base["tier"], c_result["tier"]], y=[c_base["transfer_ms"], c_result["transfer_ms"]], marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]])])
        fig.update_layout(height=275, yaxis_title="Transfer lower bound (ms)", showlegend=False)
        reveal = mo.callout(mo.md(f"**Prediction:** {c_prediction.value}. **Result:** {c_result['tier']} is **{slowdown:.1f}×** slower than {c_base['tier']} for the same {c_result['payload_gib']:.1f} GiB payload."), kind="info")
        omitted = f" Omitted/unmodeled registry links for {c_device}: {', '.join(c_all['omitted_links'])}." if c_all.get("omitted_links") else ""
        notes = mo.accordion({"Calculation Notes": mo.md(f"Transfer time = payload ÷ one-way bandwidth. The streaming lower bound excludes startup, protocol overhead, and contention.{omitted} Network bandwidth is an illustrative scenario assumption.")})
        return mo.vstack([intro, c_destination, c_prediction, apply_plotly_theme(fig), table(rows), reveal, c_capture, saved("C"), notes])

    def build_part_d():
        intro = mo.md(f"### D · Can the facility power and cool this capacity? (9 min)\nFor the **{profile.infrastructure_role}**, hosts and rack equipment add heat around the training accelerators. This scenario supplies **{d_result['electrical_capacity_kw']:.1f} kW electrical** and **{d_result['cooling_capacity_kw']:.1f} kW cooling**; PUE adds electrical draw outside the IT load.")
        if d_prediction.value is None:
            return mo.vstack([intro, d_count, d_prediction])
        fig = go.Figure([go.Bar(x=["IT heat", "Facility electrical"], y=[d_result["it_power_kw"], d_result["facility_power_kw"]])])
        fig.add_hline(y=d_result["cooling_capacity_kw"], line_dash="dash", line_color=COLORS["RedLine"], annotation_text="Cooling limit")
        fig.add_hline(y=d_result["electrical_capacity_kw"], line_dash="dot", line_color=COLORS["GreenLine"], annotation_text="Electrical limit")
        fig.update_layout(height=285, yaxis_title="Power (kW)", showlegend=False)
        rows = [{"Run": f"{d_base['accelerator_count']} accelerator(s) (baseline)", "IT heat": f"{d_base['it_power_kw']:.2f} kW", "Facility draw": f"{d_base['facility_power_kw']:.2f} kW", "Outcome": "PASS" if d_base["feasible"] else "FAIL"}, {"Run": f"{d_count.value} accelerator(s)", "IT heat": f"{d_result['it_power_kw']:.2f} kW", "Facility draw": f"{d_result['facility_power_kw']:.2f} kW", "Outcome": "PASS" if d_result["feasible"] else "FAIL"}]
        failed = [name for name, ok in (("electrical", d_result["electrical_ok"]), ("cooling", d_result["cooling_ok"])) if not ok]
        outcome = "both limits pass" if not failed else " and ".join(failed) + " fails"
        reveal = mo.callout(mo.md(f"**Prediction:** {d_prediction.value}. **Result:** {outcome}; the design occupies {d_result['node_count']} node(s) and {d_result['rack_count']} rack(s)."), kind="success" if d_result["feasible"] else "danger")
        notes = mo.accordion({"Calculation Notes": mo.md("IT power sums accelerator TDP, node overhead, and rack overhead. Cooling handles IT heat; electrical capacity handles IT power × PUE.")})
        return mo.vstack([intro, d_count, d_prediction, apply_plotly_theme(fig), table(rows), reveal, d_capture, saved("D"), notes])

    def build_part_e():
        intro = mo.md(f"### E · Which feasible system should we buy? (9 min)\nFor the **{profile.infrastructure_role}**, compare equal counts and one horizon at low and expected duty. A cheaper system that fails memory or facility limits is not feasible.")
        if e_prediction.value is None:
            return mo.vstack([intro, e_duty, e_prediction])
        rows = []
        for low, expected in zip(e_base["rows"], e_result["rows"]):
            rows.append({"Candidate": expected["device_name"], "Memory": "PASS" if expected["memory_fits"] else "FAIL", "Facility": "PASS" if expected["facility_fits"] else "FAIL", "Sustained": f"{expected['active_sustained_tflops']:.0f} TFLOP/s", "Low-duty $/PFLOP-h": f"${low['usd_per_useful_pflop_hour']:.1f}", "Expected $/PFLOP-h": f"${expected['usd_per_useful_pflop_hour']:.1f}", "Ownership": f"${expected['ownership_cost_usd']:,.0f}"})
        feasible = [r["device_key"] for r in e_result["rows"] if r["feasible"]]
        reveal = mo.callout(mo.md(f"**Prediction:** {e_prediction.value}. **Result:** feasible candidates are {', '.join(feasible) if feasible else 'none'}. Cost per work cannot erase a memory or facility failure."), kind="info" if feasible else "danger")
        notes = mo.accordion({"Calculation Notes": mo.md("Average power blends active and idle draw. Cost includes accelerator capital, explicit added capital, maintenance, and PUE-loaded electricity over one horizon. Staffing and construction remain outside the boundary.")})
        return mo.vstack([intro, e_duty, e_prediction, table(rows), reveal, mo.hstack([e_decision, e_rejected], widths="equal", wrap=True), e_capture, saved("E"), notes])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            item = _captures.get(part)
            state = "MISSING" if item is None else ("STALE" if part in audit.stale or (part, part) in audit.identical_pairs else "CURRENT")
            rows.append({"Part": part, "Prediction": item.to_dict()["prediction"] if item else "—", "Evidence": state})
        saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
        complete = audit.complete and all(w.value is not None for w in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == saved_decision
        message = "Ready for the local evidence report." if complete else "Complete five current contrasts, match the saved Part E decision, choose a different rejected option, and add the limitation, trigger, and rationale."
        return mo.vstack([mo.md("### Synthesis · Defend one infrastructure decision (5 min)\nUse saved evidence: chosen option → quantified rejected alternative → remaining limitation → reevaluation trigger."), table(rows), mo.callout(mo.md("Saved snapshots preserve exact evaluator arguments and cannot be rewritten by later controls."), kind="info"), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale, mo.callout(mo.md(message), kind="success" if complete else "warn")])

    tabs = mo.ui.tabs({"Part A": build_part_a(), "Part B": build_part_b(), "Part C": build_part_c(), "Part D": build_part_d(), "Part E": build_part_e(), "Synthesis": build_synthesis()})
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _saved = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and all(w.value is not None for w in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_02_compute_infra.py"), track=track_id,
        scenario=profile.infrastructure_role,
        learning_objectives=["Find a Roofline ranking reversal", "Separate memory capacity from bandwidth latency", "Compare transfer placement", "Test electrical and cooling limits", "Defend a duty-cycle purchase"],
        predictions={p: snapshots[p]["prediction"] for p in "ABCDE"},
        knob_settings={p: snapshots[p]["inputs"] for p in "ABCDE"},
        evidence_summary={p: {"baseline": snapshots[p]["baseline"], "result": snapshots[p]["result"], "chosen_result": snapshots[p]["chosen_result"], "result_role": snapshots[p]["result_role"], "alternatives": snapshots[p]["alternatives"]} for p in "ABCDE"},
        binding_constraints={"A": snapshots["A"]["decision"], "B": snapshots["B"]["decision"], "C": snapshots["C"]["decision"], "D": snapshots["D"]["decision"], "E": snapshots["E"]["decision"]},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["Peak compute matters only beyond the memory roof.", "Capacity, bandwidth, placement, power, and cooling are separate boundaries.", "Feasibility precedes cost per useful work."],
        reflections={"rationale": rationale.value, "trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": "Illustrative workload and facility assumptions.", "calculations": "Analytical Roofline, transfer, facility, and ownership-cost equations."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    MODEL_ID, audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _saved = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and all(w.value is not None for w in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(chapter=2, design={
                "schema_version": 1, "lab_id": "v2_02", "track_id": track_id,
                "model_id": MODEL_ID,
                "evidence": {part: item.to_dict() for part, item in _captures.items()},
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
    mo.Html(f'<div class="lab-hud"><span>LAB 02 · The Compute Infrastructure Wall · STATUS: {_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
