import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 01: Evidence Before Optimization · MLSysBook")


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
    from mlsysim.engine.constraint_explorer import (
        TRACKS, compare, evaluate, margins, monitoring_plan, population_compare,
        speedup, stress_compare,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata, report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import capture_evidence, audit_evidence

    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, TRACKS, apply_plotly_theme,
        audit_evidence, build_lab_report, capture_evidence, compare, evaluate,
        get_lab_metadata, go, ledger, margins, mo, monitoring_plan,
        population_compare, report_export_panel, speedup, stress_compare,
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
def _(TRACKS, mo, track_id):
    _track_key = track_id
    a_shift = mo.ui.slider(0, 30, value=20, step=5, label="Difficult-cohort increase (points)")
    c_budget = mo.ui.slider(0.75, 1.25, value=1.0, step=0.05, label="Deployment-envelope scale")
    c_candidate = mo.ui.dropdown(
        {"Compact": "compact", "Balanced": "balanced", "Large": "large"},
        value="Balanced", label="Candidate to carry forward",
    )
    c_conclusion = mo.ui.radio(
        {"Compact": "compact", "Balanced": "balanced", "Large": "large", "None feasible": "none"},
        label="Deployability conclusion",
    )
    d_choice = mo.ui.radio(
        {"Data": "data", "Model": "model", "Machine": "machine", "No action": "none"},
        label="First action",
    )
    d_rejected = mo.ui.radio(
        {"Data": "data", "Model": "model", "Machine": "machine"},
        label="Rejected alternative",
    )
    e_condition = mo.ui.dropdown(
        {"More serialized work": "demand", "Harder population": "population"},
        value="More serialized work", label="Stress condition",
    )
    e_demand = mo.ui.slider(1.05, 2.0, value=1.5, step=0.05, label="Work multiplier")
    e_shift = mo.ui.slider(5, 30, value=15, step=5, label="Additional difficult share (points)")
    e_interval = mo.ui.slider(20, 240, value=int(TRACKS[track_id]["monitor_interval_ms"]), step=20, label="Monitoring interval (ms)")
    return a_shift, c_budget, c_candidate, c_conclusion, d_choice, d_rejected, e_condition, e_demand, e_interval, e_shift


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Quality passes; runtime passes": "Quality passes; runtime passes", "Quality fails; runtime passes": "Quality fails; runtime passes", "Runtime fails": "Runtime fails"},
        label="After the stated shift, what crosses a requirement?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"1.00–1.24×": "1.00-1.24", "1.25–1.49×": "1.25-1.49", "1.50–1.99×": "1.50-1.99", "Exactly 2.00×": "2.00"},
        label="End-to-end speedup after 2× compute capability",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Compact · quality": "Compact: quality", "Balanced · quality/resource": "Balanced: quality or resources", "Large · memory/latency/energy": "Large: memory, latency, or energy", "No candidate fails": "No candidate fails"},
        label="Which candidate is most likely to fail, and why?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Data": "data", "Model": "model", "Machine": "machine", "None": "none"},
        label="Which equal-budget action should come first?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Quality": "quality", "Memory": "memory", "Latency": "latency", "Energy": "energy", "No crossing": "none"},
        label="Which requirement will the stated stress violate?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Improve data first": "data", "Change model first": "model", "Change machine first": "machine", "No change / hold for more evidence": "none"},
        label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Data": "data", "Model": "model", "Machine": "machine"}, label="Rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"Quality below floor": "quality", "Latency above limit": "latency", "Memory or energy above limit": "resource"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Population coverage": "coverage", "Demand growth": "demand", "Energy assumptions": "energy"},
        label="Residual risk",
    )
    rationale = mo.ui.text_area(
        label="Concise rationale",
        placeholder="Connect requirement, baseline, intervention consequence, and rejected alternative.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(TRACKS, track):
    track_id = track.value
    profile = TRACKS[track_id]
    return profile, track_id


@app.cell
def _(
    a_shift, c_budget, c_candidate, compare, d_choice, e_condition,
    e_demand, e_interval, e_shift, evaluate, monitoring_plan, population_compare,
    stress_compare, track_id,
):
    a_base = evaluate(track_id, candidate="balanced")
    a_result = evaluate(track_id, candidate="balanced", shift_pct=a_shift.value)
    b_base = evaluate(track_id, candidate="balanced")
    b_compute = compare(track_id, baseline=b_base, candidate="balanced", compute_scale=2.0)
    b_movement = compare(track_id, baseline=b_base, candidate="balanced", movement_scale=2.0)
    c_results = {
        name: evaluate(track_id, candidate=name, shift_pct=a_shift.value, budget_scale=c_budget.value)
        for name in ("compact", "balanced", "large")
    }
    d_base = c_results[c_candidate.value]
    d_results = {
        action: compare(
            track_id, baseline=d_base, candidate=c_candidate.value,
            shift_pct=a_shift.value, budget_scale=c_budget.value, intervention=action,
        )
        for action in ("data", "model", "machine")
    }
    carried_action = d_choice.value or "none"
    e_base = evaluate(
        track_id, candidate=c_candidate.value, shift_pct=a_shift.value,
        budget_scale=c_budget.value, intervention=carried_action,
    )
    if e_condition.value == "demand":
        e_comparison = stress_compare(
            track_id, candidate=c_candidate.value, shift_pct=a_shift.value,
            intervention=carried_action, condition="demand", scale=e_demand.value,
            budget_scale=c_budget.value,
        )
        e_setting = e_demand.value
    else:
        e_comparison = population_compare(
            track_id, candidate=c_candidate.value, shift_pct=a_shift.value,
            shift_delta_pct=e_shift.value, budget_scale=c_budget.value,
            intervention=carried_action,
        )
        e_setting = e_shift.value
    monitor = monitoring_plan(
        track_id, demand_scale=e_demand.value if e_condition.value == "demand" else 1.0,
        check_interval_ms=e_interval.value,
    )
    return a_base, a_result, b_base, b_compute, b_movement, c_results, carried_action, d_base, d_results, e_base, e_comparison, e_setting, monitor


@app.cell
def _(
    a_base, a_prediction, a_result, a_shift, b_base, b_compute,
    b_movement, b_prediction, c_budget, c_candidate, c_conclusion,
    c_prediction, c_results, capture_evidence, carried_action, d_base, d_choice,
    d_prediction, d_rejected, d_results, e_base, e_comparison, e_condition,
    e_interval, e_prediction, e_setting, mo, monitor, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"candidate": "balanced", "shift_pct": a_shift.value}
    b_upstream = {"candidate": "balanced", "workload": "fixed", "capability": "compute", "scale": 2.0}
    c_upstream = {"shift_pct": a_shift.value, "budget_scale": c_budget.value, "candidate": c_candidate.value, "conclusion": c_conclusion.value}
    d_upstream = {**c_upstream, "choice": d_choice.value, "rejected": d_rejected.value}
    e_upstream = {**d_upstream, "intervention": carried_action, "condition": e_condition.value, "setting": e_setting, "monitor_interval_ms": e_interval.value}

    a_capture = mo.ui.button(
        label="Capture population contrast", kind="success",
        disabled=a_prediction.value is None or a_shift.value == 0,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"shift_pct": a_shift.value}, baseline=a_base, result=a_result,
            upstream_inputs=a_upstream, model_key="constraint_explorer.evaluate",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture capability contrast", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"capability": "compute", "scale": 2.0}, baseline=b_base, result=b_compute["result"],
            upstream_inputs=b_upstream, alternatives=(b_compute, b_movement), model_key="constraint_explorer.evaluate",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture candidate comparison", kind="success",
        disabled=c_prediction.value is None or c_conclusion.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"conclusion": c_conclusion.value, "carried_candidate": c_candidate.value},
            baseline=c_results["compact"], result=c_results["large"],
            upstream_inputs=c_upstream, alternatives=tuple(c_results.values()),
            decision=c_conclusion.value, model_key="constraint_explorer.evaluate",
            chosen_result=c_results[c_candidate.value],
        )),
    )
    d_invalid = d_prediction.value is None or d_choice.value is None or d_rejected.value is None or d_choice.value == d_rejected.value
    if d_choice.value is None:
        d_selected = d_base
    elif d_choice.value == "none":
        d_selected = d_results[d_rejected.value]["result"] if d_rejected.value in d_results else d_base
    else:
        d_selected = d_results[d_choice.value]["result"]
    d_capture = mo.ui.button(
        label="Capture equal-budget decision", kind="success", disabled=d_invalid,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"choice": d_choice.value, "rejected": d_rejected.value, "tested_action": d_rejected.value if d_choice.value == "none" else d_choice.value},
            baseline=d_base, result=d_selected, upstream_inputs=d_upstream,
            alternatives=tuple(d_results.values()), decision=d_choice.value, model_key="constraint_explorer.evaluate",
            chosen_result=d_base if d_choice.value == "none" else d_selected,
            result_role="rejected alternative" if d_choice.value == "none" else "chosen intervention",
        )),
    )
    e_capture = mo.ui.button(
        label="Capture stress evidence", kind="success",
        disabled=e_prediction.value is None or d_choice.value is None or (e_condition.value == "population" and e_setting == 0),
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"condition": e_condition.value, "setting": e_setting, "monitor_interval_ms": e_interval.value},
            baseline=e_base, result=e_comparison["result"], upstream_inputs=e_upstream,
            alternatives=(monitor,), decision=e_prediction.value, model_key="constraint_explorer.evaluate",
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:760px}
    .pilot-head select{color:#0f172a;background:#fff;min-width:95px;max-width:100%}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}
    .lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 01</span><span>ABOUT 50–55 MIN</span></div><h1>Evidence Before Optimization</h1><p>What should we improve first, and what evidence makes that choice defensible?</p><div class="pilot-meta"><div><b>Track</b><br>{profile['display']}</div><div><b>Context</b><br>{profile['scenario']}</div><div><b>Deliverable</b><br>Evidence-backed recommendation</div></div></section>""")
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, header, track, mo.Html('<p class="pilot-note">All tracks use illustrative workloads and quality rates. Equations and assumptions are available in each part’s Calculation Notes.</p>')], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base, a_capture, a_prediction, a_result, a_shift, a_upstream,
    apply_plotly_theme, audit_evidence, b_base, b_capture, b_compute,
    b_movement, b_prediction, b_upstream, c_budget, c_candidate, c_capture,
    c_conclusion, c_prediction, c_results, c_upstream, carried_action, d_base,
    d_capture, d_choice, d_prediction, d_rejected, d_results, d_upstream,
    e_base, e_capture, e_comparison, e_condition, e_demand, e_interval,
    e_prediction, e_shift, e_upstream, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, go, mo, monitor, profile, rationale, track_id,
):
    _captures = get_evidence()
    upstream = {"A":a_upstream,"B":b_upstream,"C":c_upstream,"D":d_upstream,"E":e_upstream}
    audit = audit_evidence(_captures, track=track_id, required_parts=tuple("ABCDE"), per_part_upstream_inputs=upstream, contrast_required_parts=("A","B","C","D","E"))

    def verdict(r):
        return "PASS" if r["feasible"] else "FAIL · " + ", ".join(r["violations"])
    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width":"100%", "overflow-x":"auto"})
    def saved(part):
        cap = _captures.get(part)
        if cap is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Live dependencies changed, or the saved runs are identical. Recapture."), kind="danger")
        data = cap.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; the report uses this saved result, not later live controls.</small></div>')

    def part_a():
        preface = mo.md(f"### A · Can behavior fail without a code change? (8 min)\nBaseline difficult share: **{a_base['difficult_share_pct']:.0f}%**. Easy/difficult assumed quality: **{a_base['easy_quality_pct']:.0f}% / {a_base['difficult_quality_pct']:.0f}%**. Quality floor: **{a_base['quality_floor_pct']:.0f}%**.")
        if a_prediction.value is None:
            return mo.vstack([preface, a_shift, a_prediction])
        fig=go.Figure([go.Bar(name="Difficult cohort",x=["Baseline","Changed"],y=[a_base["difficult_share_pct"],a_result["difficult_share_pct"]],marker_color=COLORS["OrangeLine"])])
        fig.update_layout(height=245,margin=dict(l=20,r=20,t=20,b=20),yaxis_title="Difficult-cohort share (%)",showlegend=False)
        rows=[{"Run":"Baseline","Weighted quality":f"{a_base['quality_pct']:.1f}%","Floor":f"{a_base['quality_floor_pct']:.1f}%","Runtime":"PASS" if a_base["runtime_feasible"] else "FAIL"},{"Run":"Changed","Weighted quality":f"{a_result['quality_pct']:.1f}%","Floor":f"{a_result['quality_floor_pct']:.1f}%","Runtime":"PASS" if a_result["runtime_feasible"] else "FAIL"}]
        return mo.vstack([preface,a_shift,a_prediction,apply_plotly_theme(fig),table(rows),mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. **Result:** quality {a_base['quality_pct']:.1f}% → {a_result['quality_pct']:.1f}%; runtime remains {'healthy' if a_result['runtime_feasible'] else 'failed'}. Monitoring observes outcomes; it does not repair the model."),kind="info"),a_capture,saved("A"),mo.accordion({"Calculation Notes":mo.md("Quality = easy-cohort share × easy-cohort quality + difficult-cohort share × difficult-cohort quality. These are supplied scenario rates, not a claim that every population change reduces accuracy. The model, weights, and machine stay fixed. Explain which additional cohort evidence you would collect before blaming model capacity.")})])

    def part_b():
        if b_prediction.value is None:
            return mo.vstack([mo.md(f"### B · Which improvement speeds up the whole system? (10 min)\nBaseline terms: movement **{b_base['movement_ms']:.1f} ms**, compute **{b_base['compute_ms']:.1f} ms**, overhead **{b_base['overhead_ms']:.1f} ms**. Predict the result of 2× compute capability."),b_prediction])
        fig=go.Figure()
        for label,r in (("Baseline",b_base),("2× compute",b_compute["result"]),("2× movement",b_movement["result"])):
            for term,color in (("movement_ms",COLORS["BlueLine"]),("compute_ms",COLORS["OrangeLine"]),("overhead_ms",COLORS["GreenLine"])):
                fig.add_bar(name=term[:-3].title(),x=[label],y=[r[term]],marker_color=color,legendgroup=term,showlegend=label=="Baseline")
        fig.update_layout(barmode="stack",height=270,margin=dict(l=20,r=20,t=20,b=20),yaxis_title="Additive latency (ms)",legend_orientation="h")
        rows=[{"Case":"2× compute","Latency":f"{b_compute['result']['latency_ms']:.2f} ms","Speedup":f"{b_compute['speedup']:.2f}×"},{"Case":"2× movement","Latency":f"{b_movement['result']['latency_ms']:.2f} ms","Speedup":f"{b_movement['speedup']:.2f}×"}]
        return mo.vstack([mo.md("### B · Which improvement speeds up the whole system? (10 min)"),b_prediction,apply_plotly_theme(fig),table(rows),mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. **Result:** 2× compute produces **{b_compute['speedup']:.2f}×** end-to-end speedup; the equal capability change to movement produces **{b_movement['speedup']:.2f}×**."),kind="info"),b_capture,saved("B"),mo.accordion({"Calculation Notes":mo.vstack([mo.md("No-overlap assumption: time = bytes moved / bandwidth + operations / effective compute rate + overhead. Doubling capability halves only its term. Speedup = baseline time / changed time. Explain why the same component improvement has a different payoff when another term dominates."), table([{"Bytes moved": f"{b_base['movement_volume_mb']:.2f} MB", "Bandwidth": f"{b_base['bandwidth_mb_per_ms']:.2f} MB/ms", "Operations": f"{b_base['ops_mflop']:.2f} MFLOP", "Effective compute rate": f"{b_base['effective_rate_mflop_per_ms']:.2f} MFLOP/ms"}])])})])

    def part_c():
        preview=table([{"Candidate":n.title(),"Modeled work scale":f"{r['candidate_scale']:.2f}×","Modeled operations":f"{r['ops_mflop']:.1f} MFLOP","Modeled memory":f"{r['memory_mb']:.1f} MB"} for n,r in c_results.items()])
        if c_prediction.value is None:
            return mo.vstack([mo.md("### C · Is the higher-quality model deployable? (10 min)\nUse the visible candidate scale and deployment envelope to predict a likely failure."),preview,c_prediction])
        rows=[{"Candidate":n.title(),"Quality":f"{r['quality_pct']:.1f}/{r['quality_floor_pct']:.1f}%","Memory":f"{r['memory_mb']:.1f}/{r['memory_limit_mb']:.1f} MB","Latency":f"{r['latency_ms']:.1f}/{r['latency_limit_ms']:.1f} ms","Energy":f"{r['energy_mj']:.2f}/{r['energy_limit_mj']:.2f} mJ","Outcome":verdict(r)} for n,r in c_results.items()]
        chosen=c_results[c_candidate.value]
        return mo.vstack([mo.md("### C · Is the higher-quality model deployable? (10 min)\nQuality percentages apply only to this shared task and population."),c_prediction,mo.hstack([c_budget,c_candidate],widths="equal",wrap=True),table(rows),c_conclusion,mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. **Carried candidate:** {c_candidate.value}; {verdict(chosen)}. Higher modeled quality carries visible memory, latency, and energy costs. If none is feasible, carry one candidate into Part D to test whether any action rescues it."),kind="success" if chosen["feasible"] else "danger"),c_capture,saved("C"),mo.accordion({"Calculation Notes":mo.md("The deployment-envelope scale changes the scenario limits. Candidate scale changes model memory, operations, and movement volume; overhead stays fixed. Quality comes from supplied cohort rates for each candidate, not a universal size-to-accuracy law. Every requirement must pass. Explain which requirement rules out the attractive alternative and which assumption you would verify on a real device.")})])

    def part_d():
        if d_prediction.value is None:
            return mo.vstack([mo.md(f"### D · What should we improve first? (10 min)\nAll actions start from **{c_candidate.value}** with a {a_shift.value:.0f}-point population shift and cost one budget unit."),d_prediction])
        rows=[{"Action":"No change","Cost":f"0/{d_base['budget']:.1f}","Quality":f"{d_base['quality_pct']:.1f}%","Latency":f"{d_base['latency_ms']:.1f} ms","Energy":f"{d_base['energy_mj']:.2f} mJ","Outcome":verdict(d_base)}]
        rows += [{"Action":name.title(),"Cost":f"{cmp['result']['intervention_cost']:.1f}/{cmp['result']['budget']:.1f}","Quality":f"{cmp['result']['quality_pct']:.1f}%","Latency":f"{cmp['result']['latency_ms']:.1f} ms","Energy":f"{cmp['result']['energy_mj']:.2f} mJ","Outcome":verdict(cmp['result'])} for name,cmp in d_results.items()]
        chosen=d_base if d_choice.value in (None,"none") else d_results[d_choice.value]["result"]
        return mo.vstack([mo.md("### D · What should we improve first? (10 min)\nCompare all three complete outcomes against one equal-budget baseline."),d_prediction,table(rows),mo.hstack([d_choice,d_rejected],widths="equal",wrap=True),mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. **Chosen live outcome:** {verdict(chosen)}. Record a different rejected action and the requirement it leaves unresolved or worsens."),kind="info"),d_capture,saved("D"),mo.accordion({"Calculation Notes":mo.vstack([mo.md("Actions share the same baseline candidate, population, envelope, and intervention budget. The following assumed outcomes define this experiment; they are not guarantees about an actual optimization. Explain what the selected action improves, what it costs, and why another action is less defensible."), table([{"Action": name.title(), "Scenario assumption": cmp["result"]["intervention_assumption"]} for name, cmp in d_results.items()])])})])

    def part_e():
        stress_control=e_demand if e_condition.value=="demand" else e_shift
        context=mo.md(f"### E · Does the decision survive changed conditions? (7 min)\nPreserved design: **{c_candidate.value}**, population shift **{a_shift.value:.0f} points**, intervention **{carried_action}**. Set one stress before locking the boundary prediction.")
        if e_prediction.value is None:
            return mo.vstack([context,e_condition,stress_control,e_prediction])
        stressed=e_comparison["result"]
        rows=[{"Condition":"Preserved design","Quality":f"{e_base['quality_pct']:.1f}%","Latency":f"{e_base['latency_ms']:.1f} ms","Outcome":verdict(e_base)},{"Condition":"Stress","Quality":f"{stressed['quality_pct']:.1f}%","Latency":f"{stressed['latency_ms']:.1f} ms","Outcome":verdict(stressed)}]
        monitor_rows=[{"Expected delay":f"{monitor['detection_delay_ms']:.0f} ms","Maximum delay":f"{monitor['maximum_detection_delay_ms']:.0f} ms","Checks/window":f"{monitor['checks_per_window']:.1f}","Monitoring effort":f"{monitor['total_check_effort_mj']:.3f} mJ"}]
        return mo.vstack([context,e_condition,stress_control,e_prediction,table(rows),mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. **Stress result:** {verdict(stressed)}. Monitoring leaves quality at {stressed['quality_pct']:.1f}%; it changes detection delay and inspection effort."),kind="danger" if not stressed["feasible"] else "success"),mo.hstack([e_interval,table(monitor_rows)],widths="equal",wrap=True),e_capture,saved("E"),mo.accordion({"Calculation Notes":mo.md("Demand multiplies serialized operations, movement, and overhead; this is not a queue model. Population stress changes cohort weighting while holding the chosen intervention fixed. Monitoring assumes uniform failure arrival between instantaneous checks: expected delay is half the interval and maximum delay is one interval. Shorter intervals cost more checks without repairing quality. Explain which observed signal would justify revisiting the decision, and what evidence is still missing before repair.")})])

    def build_synthesis():
        rows=[]
        for p in "ABCDE":
            cap=_captures.get(p)
            rows.append({"Part":p,"Original prediction":cap.to_dict()["prediction"] if cap else "—","Evidence":"CURRENT" if cap and p not in audit.stale and (p,p) not in audit.identical_pairs else ("STALE" if cap else "MISSING")})
        _d_decision = _captures["D"].to_dict()["decision"] if "D" in _captures else None
        complete=audit.complete and all(w.value is not None for w in (final_choice,final_rejected,final_trigger,final_risk)) and bool(rationale.value.strip()) and final_choice.value!=final_rejected.value and final_choice.value==_d_decision
        return mo.vstack([mo.md("### Synthesis · Defend one recommendation (5 min)\nBuild the chain: **requirement → baseline → hypothesis → intervention → consequence → rejected alternative → remaining constraint → reevaluation trigger**."),table(rows),mo.callout(mo.md("Saved snapshots remain fixed while live controls move. Recapture stale evidence before generating the report."),kind="info"),mo.hstack([final_choice,final_rejected],widths="equal",wrap=True),mo.hstack([final_trigger,final_risk],widths="equal",wrap=True),rationale,mo.callout(mo.md("**Ready for the local report.**" if complete else "Complete five current contrasts, match the recommendation to the saved Part D decision, choose a different rejected action, and add the rationale."),kind="success" if complete else "warn")])

    tabs = mo.ui.tabs({"Part A":part_a(),"Part B":part_b(),"Part C":part_c(),"Part D":part_d(),"Part E":part_e(),"Synthesis":build_synthesis()})
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures=get_evidence()
    _d_decision=_captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready=audit.complete and all(w.value is not None for w in (final_choice,final_rejected,final_trigger,final_risk)) and bool(rationale.value.strip()) and final_choice.value!=final_rejected.value and final_choice.value==_d_decision
    mo.stop(not _ready)
    snapshots={p:_captures[p].to_dict() for p in "ABCDE"}
    report=build_lab_report(
        get_lab_metadata("vol1/lab_01_ml_intro.py"), track=track_id, scenario=profile["scenario"],
        learning_objectives=["Diagnose data, model, and machine constraints with controlled contrasts","Compare component changes from one baseline","Defend a bounded recommendation and reevaluation trigger"],
        predictions={p:snapshots[p]["prediction"] for p in "ABCDE"},
        knob_settings={p:snapshots[p]["inputs"] for p in "ABCDE"},
        evidence_summary={p:{"baseline":snapshots[p]["baseline"],"result":snapshots[p]["result"],"result_role":snapshots[p]["result_role"],"chosen_result":snapshots[p]["chosen_result"],"alternatives":snapshots[p]["alternatives"]} for p in "ABCDE"},
        binding_constraints={p:(snapshots[p]["chosen_result"] or snapshots[p]["result"])["violations"] for p in "ACDE"},
        decisions={"recommendation":final_choice.value,"rejected_alternative":final_rejected.value,"reevaluation_trigger":final_trigger.value},
        final_decision={"recommendation":final_choice.value,"rejected_alternative":final_rejected.value,"rationale":rationale.value},
        big_takeaways=["Population changes can alter learned behavior with fixed code.","Sequential time limits component-only speedups.","Deployability requires quality and operational evidence together."],
        reflections={"rationale":rationale.value,"monitor_trigger":final_trigger.value}, residual_risk=final_risk.value,
        result_snapshot={"track":track_id,"captures":snapshots,"recommendation":final_choice.value,"rejected":final_rejected.value,"trigger":final_trigger.value,"residual_risk":final_risk.value},
        source_trace={"scenario":"Illustrative assumptions; not a hardware benchmark.","calculations":"MLSysIM constraint explorer equations."},
    )
    mo.vstack([mo.md("## Local evidence report"),report_export_panel(report)])
    return (report,)


@app.cell
async def _(audit, final_choice, final_rejected, final_risk, final_trigger, get_evidence, ledger, mo, rationale, track_id):
    _captures=get_evidence()
    _d_decision=_captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready=audit.complete and all(w.value is not None for w in (final_choice,final_rejected,final_trigger,final_risk)) and bool(rationale.value.strip()) and final_choice.value!=final_rejected.value and final_choice.value==_d_decision
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(chapter=1, design={
                "schema_version": 1, "lab_id": "v1_01", "track_id": track_id,
                "model_id": "constraint_explorer",
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
                "evidence": {p: c.to_dict() for p, c in _captures.items()},
            })
            await ledger.flush()
        except Exception:
            _status = "LOCAL SAVE FAILED · DOWNLOAD THE REPORT TO KEEP YOUR EVIDENCE"
        else:
            _status = "SAVED"
    mo.Html(f'<div class="lab-hud" style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:#101827;color:#fff;padding:14px 18px;border-radius:9px;font-family:ui-monospace,monospace"><span>LAB 01 · Evidence Before Optimization · STATUS: {_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
