import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 02: Where Should Inference Run? · MLSysBook")


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
    from mlsysim.engine.v1_02_experiments import (
        TRACKS, compare_placements, placement_accounting,
        sustained_operation, workload_feasibility,
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
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, TRACKS, apply_plotly_theme,
        audit_evidence, build_lab_report, capture_evidence, compare_placements,
        get_lab_metadata, go, ledger, mo, placement_accounting,
        report_export_panel, sustained_operation, workload_feasibility,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Deployment track",
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
    payload_options = {f"{value:g} MB": value for value in profile["payload_options_mb"]}
    rate_options = {f"{value:g} Mb/s": value for value in profile["connection_options_mbps"]}
    a_payload = mo.ui.dropdown(
        payload_options, value=f"{profile['payload_options_mb'][-1]:g} MB",
        label="Request payload",
    )
    a_rate = mo.ui.dropdown(
        rate_options, value=f"{profile['connection_mbps']:g} Mb/s",
        label="Connection rate",
    )
    b_memory_scale = mo.ui.slider(
        0.5, 3.0, value=2.0, step=0.25, label="Resident-memory demand"
    )
    b_execution_scale = mo.ui.slider(
        0.5, 20.0, value=2.0, step=0.5, label="Sequential execution work"
    )
    c_duty = mo.ui.slider(
        0.0, 1.0, value=0.75, step=profile.get("duty_cycle_step", 0.05), label="Active observation fraction"
    )
    d_connectivity = mo.ui.checkbox(value=True, label="Remote connection available")
    d_choice = mo.ui.radio(
        {
            "Run locally": "local", "Send raw input": "raw",
            "Filter events, then send": "event_filter",
            "Send feature summary": "feature_summary", "No feasible design": "none",
        }, label="Placement decision",
    )
    d_rejected = mo.ui.radio(
        {
            "Run locally": "local", "Send raw input": "raw",
            "Filter events, then send": "event_filter",
            "Send feature summary": "feature_summary",
        }, label="Rejected alternative",
    )
    return (
        a_payload, a_rate, b_execution_scale, b_memory_scale, c_duty,
        d_choice, d_connectivity, d_rejected,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {
            "Remote stays within the deadline": "pass",
            "Remote crosses the deadline": "fail", "Local and remote tie": "tie",
        }, label="What will the selected payload and link do?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {
            "Memory fails first": "memory", "Execution time fails first": "execution_time",
            "Both fail": "both", "Both pass": "neither",
        }, label="Which local constraint will fail?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {
            "Power budget only": "average_power",
            "Observation requirement only": "observation_time",
            "Both requirements": "both", "Neither requirement": "neither",
        }, label="What will the selected duty cycle violate?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {
            "Local execution": "local", "Raw remote": "raw",
            "Event filtering plus remote": "event_filter",
            "Feature summary plus remote": "feature_summary",
            "No design is feasible": "none",
        }, label="Which placement will satisfy every stated requirement?",
    ).form(submit_button_label="Lock Part D prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {
            "Run locally": "local", "Send raw input": "raw",
            "Filter events, then send": "event_filter",
            "Send feature summary": "feature_summary",
            "Hold: no feasible design": "none",
        }, label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {
            "Run locally": "local", "Send raw input": "raw",
            "Filter events, then send": "event_filter",
            "Send feature summary": "feature_summary",
        }, label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {
            "Payload exceeds saved boundary": "payload",
            "Connection becomes unavailable": "connectivity",
            "Observation or power requirement changes": "duty_cycle",
        }, label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {
            "Filtering evidence may not transfer": "information",
            "Connection rate may vary": "network",
            "Power fixture needs target measurement": "energy",
        }, label="Residual risk",
    )
    rationale = mo.ui.text_area(
        label="Concise rationale",
        placeholder="Use saved latency, resource, duty-cycle, and filtering evidence.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_payload, a_rate, b_execution_scale, b_memory_scale, c_duty,
    compare_placements, d_connectivity, placement_accounting, profile,
    sustained_operation, track_id, workload_feasibility,
):
    a_base = placement_accounting(
        track_id, payload_mb=profile["payload_options_mb"][0],
        connection_mbps=profile["connection_options_mbps"][-1], filter_id="raw",
    )
    a_result = placement_accounting(
        track_id, payload_mb=a_payload.value, connection_mbps=a_rate.value,
        filter_id="raw",
    )
    b_base = workload_feasibility(track_id)
    b_result = workload_feasibility(
        track_id, memory_scale=b_memory_scale.value,
        execution_scale=b_execution_scale.value,
    )
    c_base = sustained_operation(track_id)
    c_result = sustained_operation(track_id, duty_cycle=c_duty.value)
    c_low = sustained_operation(track_id, duty_cycle=0.0)
    d_comparison = compare_placements(
        track_id, payload_mb=a_payload.value, connection_mbps=a_rate.value,
        connectivity_available=d_connectivity.value,
    )
    d_candidates = {
        "local": d_comparison["local"], "raw": d_comparison["remote"]["raw"],
        "event_filter": d_comparison["remote"]["event_filter"],
        "feature_summary": d_comparison["remote"]["feature_summary"],
    }
    return (
        a_base, a_result, b_base, b_result, c_base, c_low, c_result,
        d_candidates, d_comparison,
    )


@app.cell
def _(
    a_base, a_payload, a_prediction, a_rate, a_result, b_base,
    b_execution_scale, b_memory_scale, b_prediction, b_result, c_base,
    c_duty, c_low, c_prediction, c_result, capture_evidence, d_candidates,
    d_choice, d_connectivity, d_prediction, d_rejected, mo, set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"payload_mb": a_payload.value, "connection_mbps": a_rate.value}
    b_upstream = {
        "memory_scale": b_memory_scale.value,
        "execution_scale": b_execution_scale.value,
    }
    c_upstream = {"duty_cycle": c_duty.value}
    d_upstream = {
        **a_upstream, "connectivity_available": d_connectivity.value,
        "choice": d_choice.value, "rejected": d_rejected.value,
    }
    a_capture = mo.ui.button(
        label="Capture placement boundary", kind="success",
        disabled=a_prediction.value is None or a_base["inputs"] == a_result["inputs"],
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs=a_upstream, baseline=a_base, result=a_result,
            upstream_inputs=a_upstream, model_key="v1_02_experiments",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture local feasibility", kind="success",
        disabled=b_prediction.value is None or b_base["inputs"] == b_result["inputs"],
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs=b_upstream, baseline=b_base, result=b_result,
            upstream_inputs=b_upstream, model_key="v1_02_experiments",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture sustained-operation tradeoff", kind="success",
        disabled=c_prediction.value is None or c_base["inputs"] == c_result["inputs"],
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs=c_upstream, baseline=c_base, result=c_result,
            alternatives=(c_low,), upstream_inputs=c_upstream,
            model_key="v1_02_experiments",
        )),
    )
    d_compared_id = d_choice.value if d_choice.value not in (None, "none") else d_rejected.value
    if d_choice.value not in (None, "none"):
        d_baseline_id = d_rejected.value
    else:
        d_baseline_id = "local" if d_compared_id != "local" else "raw"
    d_compared_result = d_candidates.get(d_compared_id, d_candidates["event_filter"])
    d_baseline_result = d_candidates.get(d_baseline_id, d_candidates["local"])
    d_invalid = (
        d_prediction.value is None or d_choice.value is None
        or d_rejected.value is None
        or (d_choice.value != "none" and d_choice.value == d_rejected.value)
    )
    d_capture = mo.ui.button(
        label="Capture placement decision", kind="success", disabled=d_invalid,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={**d_upstream, "compared_design": d_compared_id},
            baseline=d_baseline_result, result=d_compared_result,
            alternatives=tuple(d_candidates.values()), decision=d_choice.value,
            chosen_result=(
                d_candidates[d_choice.value] if d_choice.value != "none" else None
            ),
            result_role=(
                "rejected alternative" if d_choice.value == "none" else "chosen placement"
            ),
            upstream_inputs=d_upstream, model_key="v1_02_experiments",
        )),
    )
    return (
        a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream,
        d_capture, d_upstream,
    )


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:760px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}
    .lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 02</span><span>ABOUT 45–50 MIN</span></div><h1>Where Should Inference Run?</h1><p>Which placement survives latency, memory, power, connectivity, and information requirements?</p><div class="pilot-meta"><div><b>Track</b><br>{profile['display']}</div><div><b>Context</b><br>{profile['scenario']}</div><div><b>Deliverable</b><br>Placement decision with a reversal trigger</div></div></section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css, header, track,
        mo.Html('<p class="pilot-note">All workloads, power envelopes, and retained-information observations are illustrative scenario assumptions. Inspect each Calculation Notes panel before treating a result as evidence.</p>'),
    ], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base, a_capture, a_payload, a_prediction, a_rate, a_result,
    a_upstream, apply_plotly_theme, audit_evidence, b_base, b_capture,
    b_execution_scale, b_memory_scale, b_prediction, b_result, b_upstream,
    c_base, c_capture, c_duty, c_low, c_prediction, c_result, c_upstream,
    d_candidates, d_capture, d_choice, d_connectivity, d_prediction,
    d_rejected, d_upstream, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, go, mo, profile, rationale, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCD"),
        per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCD"),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style(
            {"max-width": "100%", "overflow-x": "auto"}
        )

    def status(feasible, violations):
        return "PASS" if feasible else "FAIL · " + ", ".join(violations)

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(
                mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture after changing this part’s conditions."),
                kind="danger",
            )
        data = capture.to_dict()
        return mo.Html(
            f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later control changes cannot rewrite this record.</small></div>'
        )

    def part_a():
        intro = mo.md(
            f"### A · When does distance rule out remote inference? (10 min)\n"
            f"The remote system stays fixed at **{a_base['remote_execution_ms']:.1f} ms** of execution and **{a_base['propagation_rtt_ms']:.1f} ms** of fiber round-trip time. Change only the request payload and connection rate."
        )
        if a_prediction.value is None:
            return mo.vstack([intro, mo.hstack([a_payload, a_rate], widths="equal", wrap=True), a_prediction])
        figure = go.Figure()
        for label, result in (("Reference", a_base), ("Selected", a_result)):
            for field, name, color in (
                ("request_upload_ms", "Request upload", COLORS["BlueLine"]),
                ("propagation_rtt_ms", "Fiber round trip", COLORS["OrangeLine"]),
                ("remote_execution_ms", "Remote execution", COLORS["GreenLine"]),
                ("response_download_ms", "Response download", COLORS["OrangeLine"]),
            ):
                figure.add_bar(
                    name=name, x=[label], y=[result[field]], marker_color=color,
                    legendgroup=field, showlegend=label == "Reference",
                )
        figure.update_layout(
            barmode="stack", height=275, margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Remote request latency (ms)", legend_orientation="h",
        )
        rows = [
            {
                "Run": label, "Payload": f"{result['raw_payload_mb']:.2f} MB",
                "Link": f"{result['connection_mbps']:.1f} Mb/s",
                "Remote latency": f"{result['remote_latency_ms']:.1f} ms",
                "Deadline": f"{result['deadline_ms']:.1f} ms",
                "Boundary payload": f"{result['deadline_boundary_payload_mb']:.2f} MB",
            }
            for label, result in (("Reference", a_base), ("Selected", a_result))
        ]
        return mo.vstack([
            intro, mo.hstack([a_payload, a_rate], widths="equal", wrap=True),
            a_prediction, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(
                f"**Your prediction:** {a_prediction.value}. **Selected result:** {a_result['remote_latency_ms']:.1f} ms against a {a_result['deadline_ms']:.1f} ms deadline. The missed-deadline boundary at this link is {a_result['deadline_boundary_payload_mb']:.2f} MB."
            ), kind="danger" if "deadline" in a_result["violations"] else "success"),
            a_capture, saved("A"), mo.accordion({
                "Calculation Notes": mo.md(
                    "Remote latency adds request upload, fiber round-trip propagation, fixed remote execution, and response download. The selected rate is assumed symmetric. The model excludes queueing and tail latency."
                )
            }),
        ])

    def part_b():
        intro = mo.md(
            "### B · Does the workload fit here? (10 min)\n"
            "Memory capacity and execution deadline are separate hard constraints. Scale each demand independently."
        )
        if b_prediction.value is None:
            return mo.vstack([intro, mo.hstack([b_memory_scale, b_execution_scale], widths="equal", wrap=True), b_prediction])
        figure = go.Figure()
        for label, result in (("Reference", b_base), ("Selected", b_result)):
            figure.add_bar(
                name="Memory capacity used", x=[label],
                y=[result["memory_utilization_pct"]],
                marker_color=COLORS["BlueLine"], showlegend=label == "Reference",
            )
            figure.add_bar(
                name="Execution budget used", x=[label],
                y=[result["execution_budget_utilization_pct"]],
                marker_color=COLORS["OrangeLine"], showlegend=label == "Reference",
            )
        figure.add_hline(y=100, line_dash="dash", annotation_text="Hard limit")
        figure.update_layout(
            barmode="group", height=270, margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Requirement used (%)", legend_orientation="h",
        )
        rows = [
            {
                "Run": label,
                "Memory": f"{result['required_memory_mb']:.2f} / {result['memory_capacity_mb']:.2f} MB",
                "Execution": f"{result['execution_ms']:.1f} / {result['deadline_ms']:.1f} ms",
                "Outcome": status(result["feasible"], result["violations"]),
            }
            for label, result in (("Reference", b_base), ("Selected", b_result))
        ]
        return mo.vstack([
            intro, mo.hstack([b_memory_scale, b_execution_scale], widths="equal", wrap=True),
            b_prediction, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(
                f"**Your prediction:** {b_prediction.value}. **Selected result:** {status(b_result['feasible'], b_result['violations'])}. Changing memory demand does not change execution time; changing sequential work does not change memory."
            ), kind="success" if b_result["feasible"] else "danger"),
            b_capture, saved("B"), mo.accordion({
                "Calculation Notes": mo.md(
                    "Required memory is compared directly with capacity. Local execution time is compared directly with the request deadline. Passing one check cannot compensate for failing the other."
                )
            }),
        ])

    def part_c():
        intro = mo.md(
            f"### C · Can a fast demonstration run continuously? (10 min)\n"
            f"The supplied mission requires at least **{c_base['minimum_observation_fraction']:.0%}** active observation. Test whether a duty cycle can meet that service requirement and the average-power budget together."
        )
        if c_prediction.value is None:
            return mo.vstack([intro, c_duty, c_prediction])
        figure = go.Figure([go.Bar(
            x=["Reference", "Selected", "Always off"],
            y=[c_base["average_power_w"], c_result["average_power_w"], c_low["average_power_w"]],
            marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"]],
        )])
        figure.add_hline(
            y=c_base["average_power_budget_w"], line_dash="dash",
            annotation_text="Average-power budget",
        )
        figure.update_layout(
            height=260, margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Average power (W)", showlegend=False,
        )
        rows = [
            {
                "Run": label, "Duty cycle": f"{result['duty_cycle']:.0%}",
                "Energy / day": f"{result['energy_wh']:.3f} Wh",
                "Observation": f"{result['active_observation_hours']:.2f} / {result['minimum_observation_hours']:.2f} h",
                "Outcome": status(result["sustained_feasible"], result["violations"]),
            }
            for label, result in (("Reference", c_base), ("Selected", c_result), ("Always off", c_low))
        ]
        return mo.vstack([
            intro, c_duty, c_prediction, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(
                f"**Your prediction:** {c_prediction.value}. **Selected result:** {status(c_result['sustained_feasible'], c_result['violations'])}. Lower duty saves energy, but too little active observation fails the mission."
            ), kind="success" if c_result["sustained_feasible"] else "danger"),
            c_capture, saved("C"), mo.accordion({
                "Calculation Notes": mo.md(
                    "Average power is the duty-weighted active and idle power. Energy is average power over 24 hours. The minimum observation fraction is a supplied mission requirement, not a law of physics. No thermal behavior is simulated."
                )
            }),
        ])

    def part_d():
        intro = mo.md(
            f"### D · Where should the local/remote boundary fall? (12 min)\n"
            f"Compare local execution, raw transfer, event filtering, and a feature summary under request payload **{a_payload.value:g} MB** and link speed **{a_rate.value:g} Mb/s** (configured in Part A). Filtering observations apply only to this matched illustrative task."
        )
        if d_prediction.value is None:
            return mo.vstack([intro, d_connectivity, d_prediction])
        labels = {
            "local": "Local", "raw": "Raw remote",
            "event_filter": "Event filter", "feature_summary": "Feature summary",
        }
        figure = go.Figure([go.Bar(
            x=[labels[key] for key in labels],
            y=[d_candidates[key]["latency_ms"] for key in labels],
            marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["OrangeLine"]],
        )])
        figure.update_layout(
            height=270, margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Request latency (ms)", showlegend=False,
        )
        rows = [
            {
                "Design": labels[key],
                "Latency": "Unavailable" if result["latency_ms"] is None else f"{result['latency_ms']:.1f} ms",
                "Upload": f"{result['uploaded_payload_mb']:.3f} MB",
                "Information retained": f"{result['retained_information_pct']:.1f}%",
                "Outcome": status(result["feasible"], result["violations"]),
            }
            for key, result in d_candidates.items()
        ]
        _chosen = d_candidates.get(d_choice.value)
        _chosen_status = "Choose a decision below." if _chosen is None else status(
            _chosen["feasible"], _chosen["violations"]
        )
        return mo.vstack([
            intro, d_connectivity, d_prediction, apply_plotly_theme(figure), table(rows),
            mo.hstack([d_choice, d_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(
                f"**Your prediction:** {d_prediction.value}. **Selected decision:** {_chosen_status}. Record a different tested alternative; if none is feasible, the saved comparison still uses the rejected tested design."
            ), kind="info"),
            d_capture, saved("D"), mo.accordion({
                "Calculation Notes": mo.md(
                    "Part D evaluates baseline local feasibility (1.0× model demand) alongside remote filtering options under the link conditions set in Part A. Filtering reduces request bytes and adds local preprocessing; remote execution and response bytes stay fixed. Retained-information percentages are supplied observations, not a formula. An unavailable connection is a hard failure for every remote design."
                )
            }),
        ])

    def build_synthesis():
        rows = []
        for part in "ABCD":
            capture = _captures.get(part)
            rows.append({
                "Part": part,
                "Original prediction": capture.to_dict()["prediction"] if capture else "—",
                "Evidence": "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING"),
            })
        _saved_decision = _captures["D"].to_dict()["decision"] if "D" in _captures else None
        complete = (
            audit.complete
            and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
            and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
            and final_choice.value == _saved_decision
        )
        return mo.vstack([
            mo.md(
                "### Synthesis · Defend the placement and its boundary (6 min)\n"
                "Use saved evidence to state one placement, quantify a rejected alternative, name a remaining limitation, and choose the condition that would reverse your decision."
            ), table(rows),
            mo.callout(mo.md(
                "Saved snapshots remain fixed while live controls move. Recapture stale evidence before generating the report."
            ), kind="info"),
            mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_trigger, final_risk], widths="equal", wrap=True),
            rationale,
            mo.callout(mo.md(
                "**Ready for the local report.**" if complete else
                "Complete four current contrasts, match the recommendation to saved Part D, reject a different placement, and add the rationale."
            ), kind="success" if complete else "warn"),
        ])

    tabs = mo.ui.tabs({
        "Part A": part_a(), "Part B": part_b(), "Part C": part_c(),
        "Part D": part_d(), "Synthesis": build_synthesis(),
    })
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _saved_decision = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
        and final_choice.value == _saved_decision
    )
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCD"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_02_ml_systems.py"), track=track_id,
        scenario=profile["scenario"],
        learning_objectives=[
            "Find when transfer and distance make remote inference miss a deadline",
            "Check memory, execution, and sustained-power requirements independently",
            "Defend a local, remote, or filtered placement with a reversal trigger",
        ],
        predictions={part: snapshots[part]["prediction"] for part in "ABCD"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCD"},
        evidence_summary={
            part: {"baseline": snapshots[part]["baseline"],
                   "result": snapshots[part]["result"],
                   "result_role": snapshots[part]["result_role"],
                   "chosen_result": snapshots[part]["chosen_result"],
                   "alternatives": snapshots[part]["alternatives"]}
            for part in "ABCD"
        },
        binding_constraints={
            part: (snapshots[part]["chosen_result"] or snapshots[part]["result"])["violations"]
            for part in "ABCD"
        },
        decisions={
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "reevaluation_trigger": final_trigger.value,
        },
        final_decision={
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value, "rationale": rationale.value,
        },
        big_takeaways=[
            "Remote latency includes transfer, propagation, execution, and response time.",
            "Memory fit, execution time, power, and observation service are separate requirements.",
            "Filtering trades transmitted bytes and latency against supplied information evidence.",
        ],
        reflections={"rationale": rationale.value, "reversal_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={
            "track": track_id, "captures": snapshots,
            "recommendation": final_choice.value, "rejected": final_rejected.value,
            "trigger": final_trigger.value, "residual_risk": final_risk.value,
        },
        source_trace={
            "scenario": "Illustrative matched-task assumptions; not a branded hardware benchmark.",
            "calculations": "Placement path, capacity, and duty-cycle equations with explicit units.",
        },
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _saved_decision = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
        and final_choice.value == _saved_decision
    )
    _saved = False
    _save_error = None
    if _ready:
        try:
            ledger.save(chapter=2, design={
                "schema_version": 1, "lab_id": "v1_02", "track_id": track_id,
                "model_id": "v1_02_experiments",
                "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
            })
            await ledger.flush()
            _saved = True
        except Exception as exc:
            _save_error = str(exc)
    _status = "SAVED" if _saved else (f"SAVE FAILED: {_save_error}" if _save_error else "EVIDENCE IN PROGRESS")
    mo.Html(
        f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">02 · Where Should Inference Run? · STATUS: </span><span class="hud-active">{_status}</span></div>'
    )
    return


if __name__ == "__main__":
    app.run()
