import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 04: The Data Pipeline Wall · MLSysBook")


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
    from mlsysim.engine.v2_04_experiments import (
        SCENARIO_ASSUMPTION, TIER_PLANS, evaluate_tier_plan,
        evaluate_track_cache, evaluate_track_checkpoint, evaluate_track_demand,
        evaluate_track_layout, make_miss_burst_trace, serialize_for_evidence,
        simulate_cache_prefetch, track_profile,
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
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, Q_, SCENARIO_ASSUMPTION, TIER_PLANS,
        apply_plotly_theme, audit_evidence, build_lab_report, capture_evidence,
        evaluate_tier_plan, evaluate_track_cache, evaluate_track_checkpoint,
        evaluate_track_demand, evaluate_track_layout, get_lab_metadata, go, ledger,
        make_miss_burst_trace, mo, report_export_panel, serialize_for_evidence,
        simulate_cache_prefetch, track_profile,
    )


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
def _(track, track_profile):
    track_id = track.value
    profile = track_profile(track_id)
    return profile, track_id


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_scale = mo.ui.slider(1.2, 3.0, value=2.0, step=0.2, label="Active-consumer multiplier")
    b_spacing = mo.ui.slider(1, 10, value=2, step=1, label="Spacing inside miss burst (ms)")
    b_prefetch = mo.ui.slider(0, 4, value=1, step=1, label="Prefetch lookahead (requests)")
    c_shard = mo.ui.dropdown(
        {"64 samples": 64, "256 samples": 256, "2,000 samples": 2000},
        value="64 samples", label="Samples per shard",
    )
    d_failure = mo.ui.dropdown(
        {"During local copy": "local_copy", "During durable publication": "durable_publication", "After durable publication": "after_publication"},
        value="After durable publication", label="Injected failure",
    )
    e_tested_tier = mo.ui.dropdown(
        {"Sharded regional tier": "balanced", "Performance tier": "performance"},
        value="Sharded regional tier", label="Tier to test against capacity-first",
    )
    e_decision = mo.ui.radio(
        {"Hold current placement": "economy", "Choose sharded regional": "balanced", "Choose performance": "performance", "No feasible design": "no_feasible"},
        label="Placement decision",
    )
    return a_scale, b_prefetch, b_spacing, c_shard, d_failure, e_decision, e_tested_tier


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"No starvation": "full", "75–99% utilization": "75_99", "50–74% utilization": "50_74", "Below 50% utilization": "below_50"},
        label="After adding consumers, where will accelerator utilization land?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Nearly unchanged": "same", "Burst stalls more": "burst_more", "Burst stalls less": "burst_less"},
        label="With the same objects and hit rate, what will correlated misses do?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Storage bandwidth": "storage", "Metadata requests": "metadata", "Preprocessing": "preprocess"},
        label="Which stage binds for one-file-per-sample layout?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Restorable": "restorable", "Not restorable": "not_restorable"},
        label="Will the selected failure leave this checkpoint restorable?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Capacity-first": "economy", "Sharded regional": "balanced", "Performance": "performance"},
        label="Which placement has the lowest complete-job cost?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Hold current placement": "economy", "Choose sharded regional": "balanced", "Choose performance": "performance", "No feasible design": "no_feasible"},
        label="Final recommendation",
    )
    final_rejected = mo.ui.dropdown(
        {"Capacity-first": "economy", "Sharded regional": "balanced", "Performance": "performance"},
        value="Capacity-first", label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"Consumer count rises 25%": "consumer_growth", "Miss burst exceeds prefetch depth": "miss_burst", "Checkpoint deadline tightens": "checkpoint_window", "Storage prices change 20%": "price_change"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Trace may miss rarer bursts": "trace_coverage", "Illustrative prices may drift": "price_uncertainty", "Restore was modeled, not exercised": "restore_validation"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Evidence-backed rationale",
        placeholder="State the chosen tier, quantified rejected alternative, remaining limitation, and trigger.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    Q_, a_scale, b_prefetch, b_spacing, c_shard, d_failure, e_tested_tier,
    evaluate_tier_plan, evaluate_track_cache, evaluate_track_checkpoint,
    evaluate_track_demand, evaluate_track_layout, track_id,
):
    a_base = evaluate_track_demand(track_id, consumer_scale=1.0)
    a_result = evaluate_track_demand(track_id, consumer_scale=a_scale.value)
    b_base = evaluate_track_cache(track_id, cold_spacing=Q_(25, "millisecond"), prefetch_depth=b_prefetch.value)
    b_result = evaluate_track_cache(track_id, cold_spacing=Q_(b_spacing.value, "millisecond"), prefetch_depth=b_prefetch.value)
    b_deeper = evaluate_track_cache(track_id, cold_spacing=Q_(b_spacing.value, "millisecond"), prefetch_depth=4)
    c_base = evaluate_track_layout(track_id, samples_per_object=1)
    c_result = evaluate_track_layout(track_id, samples_per_object=c_shard.value)
    c_bandwidth = evaluate_track_layout(track_id, samples_per_object=1, storage_scale=2.0)
    d_base = evaluate_track_checkpoint(track_id, failure_stage="durable_publication")
    d_result = evaluate_track_checkpoint(track_id, failure_stage=d_failure.value)
    e_plans = {tier: evaluate_tier_plan(track_id, tier_id=tier) for tier in ("economy", "balanced", "performance")}
    e_base = e_plans["economy"]
    e_result = e_plans[e_tested_tier.value]
    return a_base, a_result, b_base, b_deeper, b_result, c_bandwidth, c_base, c_result, d_base, d_result, e_base, e_plans, e_result


@app.cell
def _(
    a_base, a_prediction, a_result, b_base, b_deeper, b_prediction, b_result,
    c_bandwidth, c_base, c_prediction, c_result, capture_evidence, d_base,
    d_prediction, d_result, e_base, e_decision, e_plans, e_prediction, e_result,
    mo, serialize_for_evidence, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    def frozen(value):
        return serialize_for_evidence(value)

    a_inputs = frozen({"baseline": a_base["inputs"], "result": a_result["inputs"]})
    b_inputs = frozen({"baseline": b_base["inputs"], "result": b_result["inputs"]})
    c_inputs = frozen({"baseline": c_base["inputs"], "result": c_result["inputs"]})
    d_inputs = frozen({"baseline": d_base["inputs"], "result": d_result["inputs"]})
    e_inputs = frozen({"baseline": e_base["inputs"], "result": e_result["inputs"], "decision": e_decision.value})
    _e_keeps_baseline = e_decision.value in {"economy", "no_feasible"}
    _e_mismatched_action = e_decision.value in {"balanced", "performance"} and e_decision.value != e_result["tier_id"]
    a_capture = mo.ui.button(
        label="Capture demand boundary", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(track=track_id, part="A", prediction=a_prediction.value, inputs=a_inputs, baseline=frozen(a_base), result=frozen(a_result), upstream_inputs=a_inputs, model_key="v2_04_experiments.evaluate_track_demand")),
    )
    b_capture = mo.ui.button(
        label="Capture miss-trace contrast", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _v: store("B", capture_evidence(track=track_id, part="B", prediction=b_prediction.value, inputs=b_inputs, baseline=frozen(b_base), result=frozen(b_result), alternatives=(frozen(b_deeper),), upstream_inputs=b_inputs, model_key="v2_04_experiments.evaluate_track_cache")),
    )
    c_capture = mo.ui.button(
        label="Capture layout decision", kind="success", disabled=c_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(track=track_id, part="C", prediction=c_prediction.value, inputs=c_inputs, baseline=frozen(c_base), result=frozen(c_result), alternatives=(frozen(c_bandwidth),), upstream_inputs=c_inputs, model_key="v2_04_experiments.evaluate_track_layout")),
    )
    d_capture = mo.ui.button(
        label="Capture checkpoint failure", kind="success", disabled=d_prediction.value is None or d_base["inputs"] == d_result["inputs"],
        on_click=lambda _v: store("D", capture_evidence(track=track_id, part="D", prediction=d_prediction.value, inputs=d_inputs, baseline=frozen(d_base), result=frozen(d_result), upstream_inputs=d_inputs, model_key="v2_04_experiments.evaluate_track_checkpoint")),
    )
    e_capture = mo.ui.button(
        label="Capture placement decision", kind="success", disabled=e_prediction.value is None or e_decision.value is None or _e_mismatched_action,
        on_click=lambda _v: store("E", capture_evidence(track=track_id, part="E", prediction=e_prediction.value, inputs=e_inputs, baseline=frozen(e_base), result=frozen(e_result), chosen_result=frozen(e_base) if _e_keeps_baseline else None, result_role="rejected alternative" if _e_keeps_baseline else "tested intervention", alternatives=tuple(frozen(e_plans[tier]) for tier in e_plans), decision=e_decision.value, upstream_inputs=e_inputs, model_key="v2_04_experiments.evaluate_tier_plan")),
    )
    return a_capture, a_inputs, b_capture, b_inputs, c_capture, c_inputs, d_capture, d_inputs, e_capture, e_inputs


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, SCENARIO_ASSUMPTION, mo, profile, track):
    css = mo.Html("""
    <style>
    .storage-head{background:linear-gradient(135deg,#101827,#164e63);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:12px}.storage-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}.storage-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.storage-head p{color:#dbeafe;max-width:780px}.storage-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.storage-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}.track-row{background:#f8fafc;border:1px solid #dbe3ec;border-radius:9px;padding:10px 13px}.assumption{color:#475569;font-size:.88rem;line-height:1.45;margin:0}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}@media(max-width:520px){.storage-head{border-radius:9px}.storage-top{padding-left:30px}.storage-meta{grid-template-columns:1fr}.track-row{padding:9px}}
    </style>""")
    header = mo.Html(f"""<section class="storage-head"><div class="storage-top"><span>VOLUME II · LAB 04</span><span>ABOUT 50–55 MIN</span></div><h1>The Data Pipeline Wall</h1><p>Which complete storage path keeps a fleet productive, publishes restorable checkpoints, and earns its cost?</p><div class="storage-meta"><div><b>Fleet context</b><br>{profile['fleet_shape']}</div><div><b>Unit of consumption</b><br>{profile['consumer_unit']}</div><div><b>Deliverable</b><br>Tier-placement decision with saved evidence</div></div></section>""")
    selector = mo.vstack([mo.md("**Choose a fleet track**"), track]).style({"max-width": "420px"})
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, header, selector, mo.Html(f'<p class="assumption">{SCENARIO_ASSUMPTION}</p>')], gap=0.5).style({"padding-top": "32px"})
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, TIER_PLANS, a_base, a_capture, a_inputs, a_prediction, a_result,
    a_scale, apply_plotly_theme, audit_evidence, b_base, b_capture, b_deeper,
    b_inputs, b_prediction, b_prefetch, b_result, b_spacing, c_bandwidth,
    c_base, c_capture, c_inputs, c_prediction, c_result, c_shard, d_base,
    d_capture, d_failure, d_inputs, d_prediction, d_result, e_base, e_capture,
    e_decision, e_inputs, e_plans, e_prediction, e_result, e_tested_tier,
    final_choice, final_rejected, final_risk, final_trigger, get_evidence, go,
    mo, profile, rationale, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_inputs, "B": b_inputs, "C": c_inputs, "D": d_inputs, "E": e_inputs}
    audit = audit_evidence(_captures, track=track_id, required_parts=tuple("ABCDE"), per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCDE"))

    def qty(value, unit):
        return float(value.to(unit).magnitude)

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture after changing the live controls."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later control changes do not rewrite this record.</small></div>')

    def part_a():
        setup = mo.md(f"### A · How many consumers can this path feed? (9 min)\nThe **{profile['consumer_unit']}** consumes one local batch every **{qty(profile['step_time'], 'ms'):.0f} ms**. Predict the utilization boundary before increasing active fleet demand.")
        if a_prediction.value is None:
            return mo.vstack([setup, a_scale, a_prediction])
        figure = go.Figure()
        figure.add_bar(name="Required", x=["Baseline", "Added consumers"], y=[qty(a_base["required_batches_per_second"], "count/s"), qty(a_result["required_batches_per_second"], "count/s")], marker_color=COLORS["OrangeLine"])
        figure.add_bar(name="Delivered", x=["Baseline", "Added consumers"], y=[qty(a_base["delivered_batches_per_second"], "count/s"), qty(a_result["delivered_batches_per_second"], "count/s")], marker_color=COLORS["BlueLine"])
        figure.update_layout(barmode="group", height=270, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Batches/s")
        rows = [{"Run": label, "Consumers": run["consumers"], "Required": f"{qty(run['required_bandwidth'], 'GB/s'):.2f} GB/s", "Bottleneck": run["bottleneck"], "Utilization": f"{run['accelerator_utilization']:.1%}"} for label, run in (("Baseline", a_base), ("Added", a_result))]
        return mo.vstack([setup, a_scale, a_prediction, apply_plotly_theme(figure), table(rows), mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. The added-consumer run reaches **{a_result['accelerator_utilization']:.1%} utilization**; **{a_result['bottleneck']}** limits delivery."), kind="danger" if a_result["starved"] else "success"), a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("Required batch rate is active consumers × target utilization ÷ step time. Required byte rate multiplies by batch bytes. Storage, metadata, and preprocessing become batch service rates; their minimum limits delivery. Rates are illustrative assumptions.")})])

    def part_b():
        setup = mo.md(f"### B · Why can a high hit rate still leave accelerators idle? (9 min)\nThe **{profile['consumer_unit']}** consumes local batches from the storage tier ({qty(profile['storage_bandwidth'], 'GB/s'):.1f} GB/s profile bandwidth). Both traces request the same objects and bytes. Only the spacing of five cold misses changes. Commit before replaying them.")
        if b_prediction.value is None:
            return mo.vstack([setup, b_spacing, b_prefetch, b_prediction])
        labels = ["Spread misses", "Burst misses", "Burst + depth 4"]
        runs = [b_base, b_result, b_deeper]
        figure = go.Figure([go.Bar(x=labels, y=[qty(run["total_stall"], "ms") for run in runs], marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"]])])
        figure.update_layout(height=270, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Total consumer stall (ms)", showlegend=False)
        rows = [{"Trace": label, "Hit rate": f"{run['hit_rate']:.1%}", "Misses": run["cache_misses"], "Max stall": f"{qty(run['max_stall'], 'ms'):.1f} ms"} for label, run in zip(labels, runs)]
        return mo.vstack([setup, mo.hstack([b_spacing, b_prefetch], widths="equal", wrap=True), b_prediction, apply_plotly_theme(figure), table(rows), mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. Hit rate remains **{b_result['hit_rate']:.1%}**, while correlated misses produce **{qty(b_result['total_stall'], 'ms'):.1f} ms** total stall. Lookahead changes readiness, not miss count."), kind="info"), b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md(f"The simulator replays timestamped batch requests for the {profile['consumer_unit']} through an LRU cache sized to 4 batches (illustrative scenario assumption) using {qty(profile['storage_bandwidth'], 'GB/s'):.1f} GB/s storage bandwidth. Storage transfers serialize. A consumer stalls until its hit or transfer completes. This deterministic trace is an illustrative fixture, not a production measurement.")})])

    def part_c():
        setup = mo.md("### C · Should we buy bandwidth or change layout? (10 min)\nThe same dataset can use one object per sample or packed shards. Predict the small-file bottleneck before viewing evidence.")
        if c_prediction.value is None:
            return mo.vstack([setup, c_prediction])
        labels = ["Small files", "Selected shards", "2× bandwidth"]
        runs = [c_base, c_result, c_bandwidth]
        figure = go.Figure([go.Bar(x=labels, y=[qty(run["elapsed"], "s") for run in runs], marker_color=[COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["BlueLine"]])])
        figure.update_layout(height=270, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Dataset pass time (s)", showlegend=False)
        rows = [{"Design": label, "Requests": f"{run['request_count']:,}", "Transferred": f"{qty(run['transferred_bytes'], 'GB'):.3f} GB", "Bottleneck": run["bottleneck"]} for label, run in zip(labels, runs)]
        return mo.vstack([setup, c_prediction, c_shard, apply_plotly_theme(figure), table(rows), mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. Small files bind on **{c_base['bottleneck']}**. Shards change request count and bytes; doubling bandwidth leaves small files bound on **{c_bandwidth['bottleneck']}**."), kind="info"), c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("Request count is ceil(samples ÷ samples per object). Bytes include minimum-transfer padding and per-request overhead. Pass time is the maximum of transfer, metadata, and preprocessing time. Oversized final shards may move unused tail bytes.")})])

    def part_d():
        setup = mo.md(f"### D · When is a checkpoint safely recoverable? (8 min)\nCheckpoints belong to the **{profile['consumer_unit']}** in the regional backend training tier ({profile['fleet_shape']}); TinyML and Mobile edge fleets stream data to the backend and never checkpoint 48 GB models locally on an MCU or phone. The checkpoint reaches local storage, then a durable tier publishes it. Choose a failure point and predict restore behavior.")
        if d_prediction.value is None:
            return mo.vstack([setup, d_failure, d_prediction])
        figure = go.Figure()
        figure.add_bar(name="Local copy", y=["Checkpoint"], x=[qty(d_result["local_duration"], "s")], orientation="h", marker_color=COLORS["BlueLine"])
        figure.add_bar(name="Durable publication", y=["Checkpoint"], x=[qty(d_result["durable_duration"], "s")], orientation="h", marker_color=COLORS["OrangeLine"])
        figure.update_layout(barmode="stack", height=220, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Elapsed time (s)", legend_orientation="h")
        outcome = "RESTORABLE" if d_result["recoverable"] else "NOT RESTORABLE"
        rows = [{"Failure": d_result["failure_stage"].replace("_", " "), "Local complete": f"{qty(d_result['local_complete'], 's'):.1f} s", "Durable complete": f"{qty(d_result['durable_complete'], 's'):.1f} s", "Outcome": outcome}]
        d_blocked = [mo.callout(mo.md("Select an injected failure stage distinct from the baseline (durable publication) to produce a distinct contrast before capturing."), kind="warn")] if d_base["inputs"] == d_result["inputs"] else []
        return mo.vstack([setup, d_failure, d_prediction, apply_plotly_theme(figure), table(rows), mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. **Analytical outcome: {outcome}.** Local completion is staging; restore accepts the generation only after durable publication."), kind="success" if d_result["recoverable"] else "danger"), *d_blocked, d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md(f"In TinyML and Mobile tracks, training checkpoints belong strictly to the regional backend training tier ({profile['consumer_unit']}), never constrained MCU or mobile devices. Local completion = bytes ÷ local bandwidth. Durable publication adds bytes ÷ durable bandwidth. Failure strictly before durable completion exposes zero restorable bytes. Atomic manifest and checksum behavior are assumed.")})])

    def part_e():
        setup = mo.md("### E · Which tier placement earns its cost? (10 min)\nCompare complete-job cost for a demanding fleet workload. Cost includes accelerator rental and its idle share, requests, movement, and retained capacity.")
        if e_prediction.value is None:
            return mo.vstack([setup, e_prediction])
        figure = go.Figure()
        for tier, plan in e_plans.items():
            figure.add_bar(name=TIER_PLANS[tier]["label"], x=["Accelerators", "Requests", "Movement", "Capacity"], y=[qty(plan["cost"]["accelerator_cost"], "dollar"), qty(plan["request_cost"], "dollar"), qty(plan["movement_cost"], "dollar"), qty(plan["capacity_cost"], "dollar")])
        figure.update_layout(barmode="group", height=290, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Scenario cost (USD)", legend_orientation="h")
        rows = [{"Tier": plan["label"], "Job time": f"{qty(plan['job_duration'], 'hour'):.1f} h", "Utilization": f"{plan['accelerator_utilization']:.1%}", "Idle accelerator": f"${qty(plan['idle_accelerator_cost'], 'dollar'):,.0f}", "Movement": f"${qty(plan['movement_cost'], 'dollar'):,.2f}", "Total": f"${qty(plan['total_cost'], 'dollar'):,.0f}"} for plan in e_plans.values()]
        _e_mismatched = e_decision.value in {"balanced", "performance"} and e_decision.value != e_result["tier_id"]
        e_blocked = []
        if e_decision.value is None:
            e_blocked.append(mo.callout(mo.md("Select a placement decision to enable capture (deliberately holding current placement or concluding no feasible design is supported)."), kind="info"))
        elif _e_mismatched:
            e_blocked.append(mo.callout(mo.md(f"To capture, make a consistent choice: test the tier you plan to adopt ('{e_decision.value}'), or deliberately hold current placement / choose no feasible design."), kind="warn"))
        return mo.vstack([setup, e_prediction, e_tested_tier, apply_plotly_theme(figure), table(rows), e_decision, mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. **{e_result['label']}** costs **${qty(e_result['total_cost'], 'dollar'):,.0f}** versus **${qty(e_base['total_cost'], 'dollar'):,.0f}** capacity-first. Higher movement price can reduce complete-job cost by avoiding idle hours."), kind="info"), *e_blocked, e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md("Fixed useful-work duration is divided by delivered utilization for complete-job time. Accelerator rental is charged for that time; idle cost is its reported subset, not added twice. All prices are illustrative assumptions.")})])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            status = "MISSING" if capture is None else ("STALE" if part in audit.stale or (part, part) in audit.identical_pairs else "CURRENT")
            rows.append({"Part": part, "Original prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": status})
        _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
        complete = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value == _saved_decision and final_choice.value != final_rejected.value
        return mo.vstack([mo.md("### Synthesis · Defend one storage placement (5 min)\nUse **demand → miss trace → layout → publication → cost**. Choose a placement, quantify a rejected alternative, name a limitation, and set a trigger."), table(rows), mo.callout(mo.md("Snapshots preserve original predictions, exact evaluator inputs, and results. Live changes mark evidence stale; they never rewrite history."), kind="info"), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale, mo.callout(mo.md("**Ready for the local report.**" if complete else "Capture five current contrasts, match the final recommendation to Part E, reject a different tier, and complete the rationale."), kind="success" if complete else "warn")])

    tabs = mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(), "Part D": part_d(), "Part E": part_e(), "Synthesis": build_synthesis()})
    tabs
    return (audit,)


@app.cell
def _(audit, build_lab_report, final_choice, final_rejected, final_risk, final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale, report_export_panel, track_id):
    _captures = get_evidence()
    _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value == _saved_decision and final_choice.value != final_rejected.value
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    _e_chosen = snapshots["E"].get("chosen_result") or snapshots["E"]["result"]
    report = build_lab_report(
        get_lab_metadata("vol2/lab_04_data_storage.py"), track=track_id, scenario=profile["fleet_shape"],
        learning_objectives=["Size a fleet data path from consumption and stage rates", "Explain stalls with explicit misses and finite prefetch", "Compare layout, publication, and lifecycle cost"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"}, knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "chosen_result": snapshots[part].get("chosen_result"), "result_role": snapshots[part].get("result_role"), "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        binding_constraints={"A": snapshots["A"]["result"]["bottleneck"], "B": f"{snapshots['B']['result']['cache_misses']} traced misses", "C": snapshots["C"]["result"]["bottleneck"], "D": snapshots["D"]["result"]["failure_stage"], "E": _e_chosen["bottleneck"]},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["The slowest complete-path service rate limits productivity.", "Hit rate hides when correlated misses drain finite prefetch.", "A checkpoint becomes restorable only after durable publication."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value}, residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": "Illustrative fleet storage assumptions; analytical results, not measurements.", "calculation_notes": "Equations and uncertainty are stated in each saved part."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(audit, final_choice, final_rejected, final_risk, final_trigger, get_evidence, ledger, mo, rationale, track_id):
    _captures = get_evidence()
    _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value == _saved_decision and final_choice.value != final_rejected.value
    _saved = False
    _save_error = None
    if _ready:
        try:
            ledger.save(chapter=4, design={"schema_version": 1, "lab_id": "v2_04", "track_id": track_id, "model_id": "v2_04_experiments", "evidence": {part: capture.to_dict() for part, capture in get_evidence().items()}, "recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value, "residual_risk": final_risk.value, "rationale": rationale.value})
            await ledger.flush()
            _saved = True
        except Exception as _error:
            _save_error = str(_error)
    if _save_error:
        _status = f"SAVE FAILED · {_save_error}"
    elif _saved:
        _status = "SAVED"
    else:
        _status = "EVIDENCE IN PROGRESS"
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">04 · The Data Pipeline Wall</span><span style="flex:1"></span><span class="hud-label">STATUS</span><span class="hud-active">{_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
