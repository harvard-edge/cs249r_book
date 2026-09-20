import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 06: Collective Communication · MLSysBook")


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
    from mlsysim.engine.v2_06_experiments import (
        algorithm_cases, algorithm_crossover, compression_comparison,
        overlap_bucket_options, overlap_timeline, routing_cases,
        semantic_exchange, topology_comparison, track_profile,
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
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, algorithm_cases,
        algorithm_crossover, apply_plotly_theme, audit_evidence,
        build_lab_report, capture_evidence, compression_comparison,
        get_lab_metadata, go, ledger, mo, overlap_bucket_options,
        overlap_timeline, report_export_panel, routing_cases,
        semantic_exchange, topology_comparison, track_profile,
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
def _(algorithm_cases, mo, overlap_bucket_options, routing_cases, track_id):
    _algorithms = algorithm_cases(track_id)
    _algorithm_labels = {
        f"{x['label']} · {x['payload_mb']:.3g} MB": x["case_id"]
        for x in _algorithms["cases"]
    }
    a_payload_case = mo.ui.dropdown(
        _algorithm_labels,
        value=next(label for label, case_id in _algorithm_labels.items() if case_id == "track"),
        label="Payload regime",
    )
    _routes = routing_cases(track_id)
    b_skew = mo.ui.dropdown(
        {x["label"]: x["case_id"] for x in _routes["cases"]},
        value="Moderate hotspot", label="Destination pattern",
    )
    b_oversubscription = mo.ui.dropdown(
        {"Nonblocking": 1.0, "2:1 oversubscribed": 2.0, "4:1 oversubscribed": 4.0},
        value="Nonblocking", label="Fabric pressure",
    )
    c_plan = mo.ui.radio(
        {"Flat ring": "flat", "Hierarchical": "hierarchical", "Hold pending measurement": "hold"},
        label="Topology decision",
    )
    _buckets = overlap_bucket_options(track_id)
    d_bucket = mo.ui.dropdown(
        {x["label"]: x["option_id"] for x in _buckets["options"]},
        value="Medium buckets", label="Bucket policy",
    )
    d_algorithm = mo.ui.dropdown(
        {"Ring": "ring", "Tree": "tree"}, value="Ring", label="Bucket collective",
    )
    e_method = mo.ui.dropdown(
        {"FP8 transport": "fp8", "Top-k with error feedback": "topk_error_feedback", "Naive top-k": "topk_naive"},
        value="FP8 transport", label="Compression candidate",
    )
    e_decision = mo.ui.radio(
        {"Adopt tested candidate": "adopt", "Keep uncompressed baseline": "keep_baseline"},
        label="Compression decision",
    )
    return (
        a_payload_case, b_oversubscription, b_skew, c_plan, d_algorithm,
        d_bucket, e_decision, e_method,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Ring": "ring", "Tree": "tree"}, label="Which algorithm is faster?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Startup": "startup", "Injection": "injection", "Hottest receiver": "hottest_receiver", "Bisection": "bisection"},
        label="What limits destination-specific exchange?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Flat ring": "flat", "Hierarchical": "hierarchical"},
        label="Which calibrated topology is faster?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Mostly hidden": "mostly_hidden", "Mostly exposed": "mostly_exposed"},
        label="Will the selected buckets hide most communication?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Faster to target": "faster", "Slower to target": "slower", "Misses target": "misses"},
        label="What is the compression consequence?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Flat ring": "flat", "Hierarchical": "hierarchical", "Hold for matched measurements": "hold"},
        label="Recommended topology",
    )
    final_rejected = mo.ui.radio(
        {"Flat ring": "flat", "Hierarchical": "hierarchical", "Compression first": "compression"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"Payload crosses boundary": "payload_boundary", "Placement changes": "topology_context", "Quality target is missed": "quality_target"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Calibration mismatch": "calibration", "Network contention": "contention", "Quality evidence transfer": "quality_transfer"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Design rationale",
        placeholder="Connect saved payload, topology, overlap, and quality evidence.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_payload_case, algorithm_cases, algorithm_crossover, b_oversubscription,
    b_skew, compression_comparison, d_algorithm, d_bucket, e_method,
    overlap_bucket_options, overlap_timeline, routing_cases, semantic_exchange,
    topology_comparison, track_id,
):
    a_packet = algorithm_cases(track_id)
    a_choice = next(x for x in a_packet["cases"] if x["case_id"] == a_payload_case.value)
    _a_below = next(x for x in a_packet["cases"] if x["case_id"] == "below")
    a_baseline = algorithm_crossover(track_id, payload_mb=_a_below["payload_mb"])
    a_result = algorithm_crossover(track_id, payload_mb=a_choice["payload_mb"])
    a_alternatives = tuple(
        algorithm_crossover(track_id, payload_mb=x["payload_mb"]) for x in a_packet["cases"]
    )
    b_packet = routing_cases(track_id)
    b_choice = next(x for x in b_packet["cases"] if x["case_id"] == b_skew.value)
    b_result = semantic_exchange(
        track_id, hotspot_fraction=b_choice["hotspot_fraction"],
        oversubscription=b_oversubscription.value,
    )
    c_result = topology_comparison(track_id)
    d_packet = overlap_bucket_options(track_id)
    d_choice = next(x for x in d_packet["options"] if x["option_id"] == d_bucket.value)
    _d_fused = next(x for x in d_packet["options"] if x["option_id"] == "fused")
    d_baseline = overlap_timeline(
        track_id, bucket_mb=_d_fused["bucket_mb"], algorithm=d_algorithm.value,
    )
    d_result = overlap_timeline(
        track_id, bucket_mb=d_choice["bucket_mb"], algorithm=d_algorithm.value,
    )
    e_result = compression_comparison(track_id, method=e_method.value)
    return (
        a_alternatives, a_baseline, a_choice, a_result, b_choice, b_result,
        c_result, d_baseline, d_choice, d_result, e_result,
    )


@app.cell
def _(
    a_alternatives, a_baseline, a_choice, a_prediction, a_result, b_choice,
    b_oversubscription, b_prediction, b_result, b_skew, c_plan, c_prediction,
    c_result, capture_evidence, d_algorithm, d_baseline, d_bucket, d_choice,
    d_prediction, d_result, e_decision, e_method, e_prediction, e_result, mo,
    set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"payload_case": a_choice["case_id"], "payload_mb": a_choice["payload_mb"]}
    b_upstream = {"destination_case": b_skew.value, "hotspot_fraction": b_choice["hotspot_fraction"], "oversubscription": b_oversubscription.value}
    c_upstream = {"fixture": c_result["fixture"], "decision": c_plan.value}
    d_upstream = {"bucket_policy": d_bucket.value, "bucket_mb": d_choice["bucket_mb"], "algorithm": d_algorithm.value}
    e_upstream = {"method": e_method.value, "decision": e_decision.value}

    a_capture = mo.ui.button(
        label="Capture crossover evidence", kind="success",
        disabled=a_prediction.value is None or a_baseline["inputs"] == a_result["inputs"],
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs=a_upstream, baseline=a_baseline, result=a_result,
            alternatives=a_alternatives, decision=a_result["winner"],
            upstream_inputs=a_upstream, model_key="v2_06_experiments.algorithm_crossover",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture semantic contrast", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs=b_upstream, baseline=b_result["reduction"], result=b_result["routed"],
            alternatives=(b_result["reduction"], b_result["routed"]),
            decision=b_result["routed"]["limiting_bound"],
            upstream_inputs=b_upstream, model_key="v2_06_experiments.semantic_exchange",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture topology decision", kind="success",
        disabled=c_prediction.value is None or c_plan.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"selected_plan": c_plan.value, **c_result["inputs"]},
            baseline=c_result["flat"], result=c_result["hierarchical"],
            alternatives=(c_result["flat"], c_result["hierarchical"]),
            decision=c_plan.value, upstream_inputs=c_upstream,
            model_key="v2_06_experiments.topology_comparison",
            chosen_result=(c_result["hierarchical"] if c_plan.value == "hierarchical" else c_result["flat"]),
            result_role=("chosen intervention" if c_plan.value == "hierarchical" else "rejected alternative"),
        )),
    )
    d_capture = mo.ui.button(
        label="Capture overlap timeline", kind="success",
        disabled=d_prediction.value is None or d_baseline["inputs"] == d_result["inputs"],
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs=d_upstream, baseline=d_baseline, result=d_result,
            alternatives=(d_baseline, d_result), decision=d_bucket.value,
            upstream_inputs=d_upstream, model_key="v2_06_experiments.overlap_timeline",
        )),
    )
    e_capture = mo.ui.button(
        label="Capture compression consequence", kind="success",
        disabled=e_prediction.value is None or e_decision.value is None,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs=e_upstream, baseline=e_result["baseline"], result=e_result["result"],
            alternatives=(e_result["baseline"], e_result["result"]),
            decision=e_decision.value, upstream_inputs=e_upstream,
            model_key="v2_06_experiments.compression_comparison",
            chosen_result=(e_result["result"] if e_decision.value == "adopt" else e_result["baseline"]),
            result_role=("chosen intervention" if e_decision.value == "adopt" else "rejected alternative"),
        )),
    )
    return (
        a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream,
        d_capture, d_upstream, e_capture, e_upstream,
    )


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .collective-head{background:linear-gradient(135deg,#101827,#312e81);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:12px}
    .collective-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .collective-head h1{font-size:clamp(1.7rem,5vw,2.7rem);line-height:1.05;margin:16px 0 8px}.collective-head p{color:#e0e7ff;max-width:780px}
    .collective-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:9px;margin-top:17px}.collective-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .collective-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .metric-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:9px}.metric{border:1px solid #dbe4ee;border-radius:9px;padding:10px;background:#fff}.metric b{display:block;font-size:1.15rem;color:#172554}.metric small{color:#64748b}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    .failure-card{border-left:4px solid #CB202D;background:#fff1f2;padding:10px 12px;border-radius:7px}
    @media(max-width:520px){.collective-head{border-radius:9px;margin-top:30px}.collective-meta,.metric-row{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""
    <section class="collective-head">
      <div class="collective-top"><span>VOLUME II · LAB 06</span><span>ABOUT 50 MIN</span></div>
      <h1>Collective Communication</h1>
      <p>When should a fleet change the operation, schedule, topology, bucket timeline, or payload it sends?</p>
      <div class="collective-meta">
        <div><b>Participants</b><br>{profile['participants']} {profile['participant_kind']}</div>
        <div><b>Context</b><br>{profile['fleet_semantics']}</div>
        <div><b>Deliverable</b><br>Communication design review</div>
      </div>
    </section>""")
    _fleet_note = (
        "Endpoint uploads remain separate from gateway or backend collectives."
        if profile["endpoint_count"] > 0
        else "Multi-node accelerator workers synchronize gradients directly without endpoint upload stages."
    )
    note = mo.Html(f'<p class="collective-note">{_fleet_note} Results are analytical or simulated from illustrative fleet workloads; matched calibration and quality fixtures are labeled where used.</p>')
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, header, track, note], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_capture, a_payload_case, a_prediction, a_result, a_upstream, apply_plotly_theme,
    audit_evidence, b_capture, b_oversubscription, b_prediction, b_result,
    b_skew, b_upstream, c_capture, c_plan, c_prediction, c_result, c_upstream,
    d_algorithm, d_bucket, d_capture, d_prediction, d_result, d_upstream,
    e_capture, e_decision, e_method, e_prediction, e_result, e_upstream,
    final_choice, final_rejected, final_risk, final_trigger, get_evidence, go,
    mo, rationale, track_id,
):
    captures = get_evidence()
    upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style(
            {"max-width": "100%", "overflow-x": "auto"}
        )

    def metrics(items):
        cards = "".join(
            f'<div class="metric"><small>{label}</small><b>{value}</b><small>{detail}</small></div>'
            for label, value, detail in items
        )
        return mo.Html(f'<div class="metric-row">{cards}</div>')

    def saved(part):
        capture = captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(
                mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** A dependency changed, or the saved evaluator inputs are identical. Recapture this part."),
                kind="danger",
            )
        snapshot = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {snapshot["prediction"]}<br><small>Track {snapshot["track"]}; later controls do not rewrite this record.</small></div>')

    def build_part_a():
        intro = mo.md("### A · When does ring versus tree preference reverse? (8 min)\nChoose a payload regime, then predict the winner. The instrument separates startup rounds from bytes transferred.")
        if a_prediction.value is None:
            return mo.vstack([intro, a_payload_case, a_prediction])
        fig = go.Figure()
        fig.add_bar(name="Startup", x=["Ring", "Tree"], y=[a_result["ring"]["startup_ms"], a_result["tree"]["startup_ms"]], marker_color=COLORS["OrangeLine"])
        fig.add_bar(name="Transfer", x=["Ring", "Tree"], y=[a_result["ring"]["transfer_ms"], a_result["tree"]["transfer_ms"]], marker_color=COLORS["BlueLine"])
        fig.update_layout(barmode="stack", height=285, yaxis_title="Analytical time (ms)", legend_orientation="h")
        return mo.vstack([
            intro, a_payload_case, a_prediction, apply_plotly_theme(fig),
            metrics([
                ("Payload", f"{a_result['payload_mb']:.3g} MB", "per participant"),
                ("Crossover", f"{a_result['crossover_mb']:.3g} MB", "same process group"),
                ("Winner", a_result["winner"].title(), "analytical comparison"),
            ]),
            mo.callout(mo.md(f"You predicted **{a_prediction.value}**. The selected payload favors **{a_result['winner']}** because startup and transfer use different counts."), kind="info"),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md("Ring uses `2(N−1)` startups and transfers `2(N−1)M/N`. The simple tree uses `2 ceil(log₂N)` startups and transfers a full message at each level. MLSysIM computes their exact intersection.")}),
        ])

    def build_part_b():
        intro = mo.md("### B · Does this workload reduce values or route records? (8 min)\nKeep each participant’s payload fixed. Change destination skew and fabric pressure, then identify the routed exchange’s active bound.")
        if b_prediction.value is None:
            return mo.vstack([intro, b_skew, b_oversubscription, b_prediction])
        fig = go.Figure([go.Bar(
            x=["Reduction", "Routed exchange"],
            y=[b_result["reduction"]["time_ms"], b_result["routed"]["time_ms"]],
            marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]],
        )])
        fig.update_layout(height=270, yaxis_title="Analytical time (ms)", showlegend=False)
        return mo.vstack([
            intro, mo.hstack([b_skew, b_oversubscription], widths="equal", wrap=True),
            b_prediction, apply_plotly_theme(fig),
            table([
                {"Operation": "Reduction", "Result semantics": "Same aggregate everywhere", "Time (ms)": b_result["reduction"]["time_ms"]},
                {"Operation": "Routed exchange", "Result semantics": "Different destination records", "Time (ms)": b_result["routed"]["time_ms"]},
            ]),
            mo.callout(mo.md(f"You predicted **{b_prediction.value}**. The routed case is limited by **{b_result['routed']['limiting_bound']}**. AllReduce cannot produce destination-specific outputs."), kind="info"),
            b_capture, saved("B"),
            mo.accordion({"Calculation Notes": mo.md(f"Routed traffic is bounded by sender injection, the hottest receiver, and bisection capacity. Oversubscription lowers bisection capacity. {b_result['endpoint_note']}")}),
        ])

    def build_part_c():
        intro = mo.md("### C · When does topology change the winning schedule? (9 min)\nCompare a flat inter-group ring with local reduce-scatter, an inter-group ring, and local all-gather. The fixture matches algorithm, group size, and placement.")
        if c_prediction.value is None:
            return mo.vstack([intro, c_prediction])
        fig = go.Figure()
        fig.add_bar(name="Analytical", x=["Flat ring", "Hierarchical"], y=[c_result["flat"]["analytical_ms"], c_result["hierarchical"]["analytical_ms"]], marker_color=COLORS["BlueLine"])
        fig.add_bar(name="Calibrated", x=["Flat ring", "Hierarchical"], y=[c_result["flat"]["calibrated_ms"], c_result["hierarchical"]["calibrated_ms"]], marker_color=COLORS["GreenLine"])
        fig.update_layout(barmode="group", height=285, yaxis_title="Time (ms)", legend_orientation="h")
        unsupported = c_plan.value is not None and c_plan.value != "hold" and c_plan.value != c_result["calibrated_winner"]
        consequence = mo.Html('<div class="failure-card"><b>Decision consequence</b><br>The selected topology is slower in the matched illustrative calibration. The report will preserve this unsupported choice.</div>') if unsupported else mo.callout(mo.md("The plan follows current evidence, or deliberately waits for new matched measurements."), kind="success")
        return mo.vstack([
            intro, c_prediction, apply_plotly_theme(fig),
            metrics([
                ("Analytical winner", c_result["analytical_winner"].title(), "unit-aware model"),
                ("Calibrated winner", c_result["calibrated_winner"].title(), "illustrative fixture"),
                ("Local group", str(c_result["local_group"]), "fast-tier participants"),
            ]),
            c_plan, consequence, c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md("The hierarchical path prices all three phases separately. Calibration factors come from the supplied illustrative fixture, not a live product benchmark.")}),
        ])

    def build_part_d():
        phase_label = d_result.get("compute_phase", "backward pass")
        member_kind = "parameters" if "aggregation" in phase_label else "gradients"
        intro = mo.md(f"### D · How much communication can the {phase_label} actually hide? (9 min)\nSelect a bucket policy and algorithm. Buckets launch only when all member {member_kind} are ready, and every launch competes for one network resource.")
        if d_prediction.value is None:
            return mo.vstack([intro, mo.hstack([d_bucket, d_algorithm], widths="equal", wrap=True), d_prediction])
        fig = go.Figure()
        for event in d_result["events"]:
            fig.add_bar(
                name=f"Bucket {event['bucket']}", y=["Network"],
                x=[event["duration_ms"]], base=[event["start_ms"]], orientation="h",
                text=[", ".join(event["layers"])],
                hovertemplate="%{text}<br>start %{base:.2f} ms<br>duration %{x:.2f} ms<extra></extra>",
            )
        fig.add_vline(x=d_result["backward_end_ms"], line_dash="dash", line_color=COLORS["RedLine"])
        fig.update_layout(height=250, xaxis_title=f"{phase_label.capitalize()} timeline (ms)", showlegend=False, barmode="overlay")
        return mo.vstack([
            intro, mo.hstack([d_bucket, d_algorithm], widths="equal", wrap=True),
            d_prediction, apply_plotly_theme(fig),
            metrics([
                ("Hidden", f"{d_result['hidden_communication_ms']:.2f} ms", f"before {phase_label} completes"),
                ("Exposed", f"{d_result['exposed_communication_ms']:.2f} ms", "extends the step"),
                ("Launches", str(d_result["bucket_count"]), "serialized buckets"),
            ]),
            mo.callout(mo.md(f"You predicted **{d_prediction.value}**. The readiness schedule hides **{d_result['overlap_fraction']:.1%}** of collective work; asynchronous launch alone did not determine this result."), kind="info"),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md(f"Each bucket starts at the later of its final readiness time and network availability. Exposed time is the final network tail beyond {phase_label} completion. Smaller buckets launch earlier but pay more startup rounds.")}),
        ])

    def build_part_e():
        intro = mo.md("### E · When is sending fewer bytes worthwhile? (9 min)\nTest one codec against the uncompressed baseline. The instrument includes encoding, decoding, communication, and supplied steps-to-target evidence.")
        if e_prediction.value is None:
            return mo.vstack([intro, e_method, e_prediction])
        fig = go.Figure()
        fig.add_bar(name="Communication", x=["Uncompressed", e_result["method"]], y=[e_result["baseline_communication_ms"], e_result["communication_ms"]], marker_color=COLORS["BlueLine"])
        fig.add_bar(name="Codec", x=["Uncompressed", e_result["method"]], y=[0, e_result["codec_ms"]], marker_color=COLORS["OrangeLine"])
        fig.update_layout(barmode="stack", height=280, yaxis_title="Per-step overhead (ms)", legend_orientation="h")
        if e_result["quality_target_reached"]:
            consequence = mo.callout(mo.md(f"The supplied scenario reaches target in **{e_result['steps_to_target']} steps**. Compare total time to target, not communication time alone."), kind="success")
        else:
            consequence = mo.Html(f'<div class="failure-card"><b>Quality target missed</b><br>The supplied outcome remains {e_result["quality_gap_pp"]:.1f} percentage points below target. A faster step has no valid time-to-target.</div>')
        prediction_feedback = mo.callout(
            mo.md(f"You predicted **{e_prediction.value}**. The resulting outcome is **{e_result['outcome']}** to target."),
            kind="info" if e_prediction.value == e_result["outcome"] else "warn",
        )
        return mo.vstack([
            intro, e_method, e_prediction, apply_plotly_theme(fig),
            prediction_feedback,
            table([
                {"Plan": "Uncompressed", "Payload (MB)": e_result["payload_mb"], "Step (ms)": e_result["baseline_step_ms"], "Time to target (ms)": e_result["baseline"]["time_to_target_ms"]},
                {"Plan": e_result["method"], "Payload (MB)": e_result["compressed_payload_mb"], "Step (ms)": e_result["step_ms"], "Time to target (ms)": e_result["time_to_target_ms"]},
            ]),
            consequence, e_decision, e_capture, saved("E"),
            mo.accordion({"Calculation Notes": mo.md("Encoding scans the original payload; decoding processes the compressed payload. Steps-to-target and target attainment come from a supplied illustrative outcome fixture, not a hardware score or universal quality equation.")}),
        ])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = captures.get(part)
            state = "MISSING"
            if capture:
                state = "STALE" if part in audit.stale or (part, part) in audit.identical_pairs else "CURRENT"
            rows.append({
                "Part": part,
                "Original prediction": capture.to_dict()["prediction"] if capture else "—",
                "Evidence": state,
            })
        quantified = []
        if "C" in captures:
            topology_snapshot = captures["C"].to_dict()
            quantified.extend([
                {"Alternative": "Flat ring", "Saved consequence": f"{topology_snapshot['baseline']['calibrated_ms']} ms calibrated"},
                {"Alternative": "Hierarchical", "Saved consequence": f"{topology_snapshot['result']['calibrated_ms']} ms calibrated"},
            ])
        if "E" in captures:
            compression_snapshot = captures["E"].to_dict()
            result_time = compression_snapshot["result"]["time_to_target_ms"]
            result_consequence = f"{result_time} ms to target" if result_time is not None else "quality target not reached"
            quantified.extend([
                {"Alternative": "Uncompressed", "Saved consequence": f"{compression_snapshot['baseline']['time_to_target_ms']} ms to target"},
                {"Alternative": "Tested compression", "Saved consequence": result_consequence},
            ])
        ready = (
            audit.complete
            and all(x.value is not None for x in (final_choice, final_rejected, final_trigger, final_risk))
            and bool(rationale.value.strip())
            and final_choice.value != final_rejected.value
            and final_choice.value == c_plan.value
        )
        message = "**Ready for the local communication design review.**" if ready else "Capture five current contrasts, align the recommendation with the saved topology decision, choose a different rejected alternative, and complete the rationale."
        return mo.vstack([
            mo.md("### Synthesis · Defend one fleet communication plan (7 min)\nUse the saved chain: **operation semantics → algorithm boundary → topology → schedulable overlap → time to quality target**."),
            table(rows),
            table(quantified) if quantified else mo.md("Quantified alternatives appear after Parts C and E are captured."),
            mo.callout(mo.md("Saved snapshots preserve the original prediction, exact evaluator inputs, result, and decision. Recapture stale experiments before generating the report."), kind="info"),
            mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_trigger, final_risk], widths="equal", wrap=True),
            rationale, mo.callout(mo.md(message), kind="success" if ready else "warn"),
        ])

    complete = (
        audit.complete
        and all(x.value is not None for x in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip())
        and final_choice.value != final_rejected.value
        and final_choice.value == c_plan.value
    )
    tabs = mo.ui.tabs({
        "A · Algorithm": build_part_a(), "B · Semantics": build_part_b(),
        "C · Topology": build_part_c(), "D · Overlap": build_part_d(),
        "E · Compression": build_part_e(), "Synthesis": build_synthesis(),
    })
    tabs
    return audit, complete


@app.cell
def _(
    build_lab_report, complete, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    mo.stop(
        not complete,
        mo.callout(mo.md("## Local evidence report\nThe report unlocks after all five contrasts are current and synthesis is complete."), kind="warn"),
    )
    snapshots = {part: get_evidence()[part].to_dict() for part in "ABCDE"}
    chosen = {
        part: snapshots[part].get("chosen_result") or snapshots[part]["result"]
        for part in "ABCDE"
    }
    report = build_lab_report(
        get_lab_metadata("vol2/lab_06_collective_communication.py"),
        track=track_id, scenario=profile["fleet_semantics"],
        learning_objectives=[
            "Identify ring versus tree algorithm crossover points",
            "Distinguish reduction semantics from routed exchange limits",
            "Compare flat and hierarchical topology schedules",
            "Quantify backward-pass communication overlap across bucket policies",
            "Evaluate compression codecs against time to quality target",
        ],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {
            "baseline": snapshots[part]["baseline"], "tested_result": snapshots[part]["result"],
            "chosen_result": chosen[part], "result_role": snapshots[part]["result_role"],
            "alternatives": snapshots[part]["alternatives"],
        } for part in "ABCDE"},
        binding_constraints={
            "algorithm": snapshots["A"]["decision"],
            "routed_exchange": snapshots["B"]["decision"],
            "topology": snapshots["C"]["decision"],
            "compression_quality_target": chosen["E"]["quality_target_reached"],
        },
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=[
            "Payload and participant count can reverse ring and tree preference.",
            "Reduction and destination-specific exchange have different semantics and bounds.",
            "Overlap requires a feasible readiness and network-resource schedule.",
        ],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={
            "track": track_id, "captures": snapshots, "recommendation": final_choice.value,
            "rejected": final_rejected.value, "trigger": final_trigger.value,
            "residual_risk": final_risk.value,
        },
        source_trace={
            "scenario": "Illustrative fleet, calibration, and compression outcome fixtures.",
            "calculations": "MLSysIM V2-06 experiment engine with Pint-backed collective formulas.",
        },
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    complete, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    save_status = "EVIDENCE IN PROGRESS"
    if complete:
        try:
            ledger.save(chapter=6, design={
                "schema_version": 1, "lab_id": "v2_06", "track_id": track_id,
                "model_id": "v2_06_experiments",
                "evidence": {part: capture.to_dict() for part, capture in get_evidence().items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
            })
            await ledger.flush()
            save_status = "SAVED"
        except Exception as exc:
            save_status = f"SAVE FAILED · {type(exc).__name__}"
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">06 · Collective Communication</span><span style="flex:1"></span><span class="hud-label">|</span><span class="hud-label">STATUS</span><span class="hud-active">{save_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
