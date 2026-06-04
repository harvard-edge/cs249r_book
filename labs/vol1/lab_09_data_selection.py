import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ===========================================================================
# ZONE A: SETUP
# ===========================================================================


@app.cell
async def _():
    import marimo as mo
    import sys
    from pathlib import Path

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        coverage_profile,
        data_policy_decision,
        data_selection_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        selection_frontier,
        selection_utility,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        COLORS,
        LAB_CSS,
        apply_plotly_theme,
        build_lab_report,
        coverage_profile,
        data_policy_decision,
        data_selection_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        selection_frontier,
        selection_utility,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_09_metadata = get_lab_metadata("vol1/lab_09_data_selection.py")
    return (v1_09_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_09_track_picker = track_selector(default=_default_track)
    v1_09_track_picker
    return (v1_09_track_picker,)


@app.cell
def _(
    data_selection_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_09_track_picker,
):
    v1_09_track_id = v1_09_track_picker.value
    v1_09_profile = get_track_profile(v1_09_track_id)
    v1_09_variant = get_lab_track_variant("v1_09_selection_paradox", v1_09_profile.track_id)
    v1_09_hardware = resolve_mlsysim_ref(v1_09_variant.hardware_ref)
    v1_09_model = resolve_mlsysim_ref(v1_09_variant.model_ref)
    v1_09_selection = data_selection_profile(
        v1_09_profile,
        v1_09_variant,
        v1_09_hardware,
        v1_09_model,
    )
    return (
        v1_09_hardware,
        v1_09_model,
        v1_09_profile,
        v1_09_selection,
        v1_09_track_id,
        v1_09_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
        track_arc_context,
    v1_09_metadata,
    v1_09_profile,
    v1_09_selection,
    v1_09_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #94a3b8; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 09
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Selection Paradox
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Quantity &middot; Quality &middot; Coverage &middot; Cost &middot; Blind Spots
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 860px; line-height: 1.65;">
                {v1_09_variant.workload_summary} This lab shows why data selection is
                a systems decision: raw volume, label quality, subgroup coverage, rare
                events, privacy, and compute cost all pull in different directions.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    3 Parts + Memo &middot; ~45 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_09_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_09_selection.dataset_unit}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Utility-Cost Frontier</span>
                <span class="badge badge-warn">Coverage Risk</span>
                <span class="badge badge-fail">Data Policy</span>
            </div>
        </div>
        """),
        track_context(v1_09_profile),
        track_arc_context(v1_09_profile, v1_09_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_09_selection):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
            Learning Objectives
        </div>
        <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
            <div style="margin-bottom: 3px;">1. <strong>Compare utility and cost:</strong>
                quantify quality, coverage, rare-event coverage, storage, and compute cost.</div>
            <div style="margin-bottom: 3px;">2. <strong>Inspect subgroup risk:</strong>
                identify the coverage blind spot hidden by aggregate utility.</div>
            <div style="margin-bottom: 3px;">3. <strong>Choose a data policy:</strong>
                name what data to collect next and what bias or blind spot remains.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                Which data should {v1_09_selection.label} collect next, and what
                coverage risk is accepted by not collecting everything?
            </div>
        </div>
    </div>
    """)
    return


# ===========================================================================
# ZONE B: CONTROLS AND COMPUTATION
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_09_selection):
    v1_09_value_prediction = mo.ui.radio(
        options={
            "Quantity is most valuable": "quantity",
            "Label quality is most valuable": "quality",
            "Coverage and rare events are most valuable": "coverage",
            "Cost and storage dominate the choice": "cost",
        },
        label=f"{v1_09_selection.label}: which selection pressure do you expect to matter most?",
    )
    v1_09_value_prediction
    return (v1_09_value_prediction,)


@app.cell(hide_code=True)
def _(mo):
    v1_09_fraction_multiplier = mo.ui.slider(
        start=0.5,
        stop=1.5,
        value=1.0,
        step=0.05,
        label="Dataset fraction multiplier",
    )
    v1_09_fraction_multiplier
    return (v1_09_fraction_multiplier,)


@app.cell(hide_code=True)
def _(mo, v1_09_selection):
    _policy_options = {
        policy.label: policy.policy_id
        for policy in v1_09_selection.policy_options
    }
    v1_09_policy_choice = mo.ui.dropdown(
        options=_policy_options,
        value=v1_09_selection.policy_options[0].label,
        label="Data policy",
    )
    v1_09_reflection = mo.ui.text_area(
        label="Reflection",
        placeholder="Name the data policy, what data to collect next, and the blind spot you accept.",
        full_width=True,
    )
    return (v1_09_policy_choice, v1_09_reflection)


@app.cell
def _(
    coverage_profile,
    data_policy_decision,
    selection_frontier,
    selection_utility,
    v1_09_fraction_multiplier,
    v1_09_policy_choice,
    v1_09_selection,
):
    v1_09_frontier = selection_frontier(
        v1_09_selection,
        fraction_multiplier=v1_09_fraction_multiplier.value,
    )
    v1_09_selected_utility = selection_utility(
        v1_09_selection,
        policy_id=v1_09_policy_choice.value,
        fraction_multiplier=v1_09_fraction_multiplier.value,
    )
    v1_09_coverage = coverage_profile(
        v1_09_selection,
        policy_id=v1_09_policy_choice.value,
    )
    v1_09_decision = data_policy_decision(
        v1_09_selection,
        policy_id=v1_09_policy_choice.value,
        fraction_multiplier=v1_09_fraction_multiplier.value,
    )
    return (
        v1_09_coverage,
        v1_09_decision,
        v1_09_frontier,
        v1_09_selected_utility,
    )


# ===========================================================================
# ZONE C: PARTS
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    source_trace,
    v1_09_coverage,
    v1_09_decision,
    v1_09_fraction_multiplier,
    v1_09_frontier,
    v1_09_policy_choice,
    v1_09_reflection,
    v1_09_selected_utility,
    v1_09_selection,
    v1_09_value_prediction,
):
    _frontier_fig = go.Figure()
    _frontier_fig.add_trace(go.Scatter(
        x=[row.total_cost for row in v1_09_frontier],
        y=[row.utility_score for row in v1_09_frontier],
        mode="markers+text",
        text=[row.policy_label for row in v1_09_frontier],
        textposition="top center",
        marker=dict(
            size=[14 + row.coverage_score_pct / 10 for row in v1_09_frontier],
            color=[COLORS["GreenLine"] if row.feasible else COLORS["RedLine"] for row in v1_09_frontier],
            line=dict(color="#ffffff", width=1.5),
        ),
        name="Policy",
    ))
    _frontier_fig.add_vline(
        x=v1_09_selection.cost_budget,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        annotation_text="cost budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _frontier_fig.update_layout(
        height=360,
        xaxis=dict(title="Total cost", gridcolor="#f1f5f9"),
        yaxis=dict(title="Utility score", gridcolor="#f1f5f9", range=[0, 100]),
        margin=dict(l=60, r=20, t=35, b=60),
    )
    apply_plotly_theme(_frontier_fig)

    _frontier_rows = "".join(
        f"""
        <tr>
          <td>{row.policy_label}</td>
          <td>{row.selected_examples_k:.1f}k</td>
          <td>{row.quality_score_pct:.1f}%</td>
          <td>{row.coverage_score_pct:.1f}%</td>
          <td>{row.rare_event_score_pct:.1f}%</td>
          <td>{row.total_cost:.1f}</td>
          <td>{row.storage_mb:.1f} MB</td>
          <td>{'yes' if row.feasible else 'no - violation'}</td>
          <td>{row.dominant_risk}</td>
        </tr>
        """
        for row in v1_09_frontier
    )

    _coverage_fig = go.Figure()
    _coverage_fig.add_trace(go.Bar(
        x=[cell.label for cell in v1_09_coverage.cells],
        y=[cell.coverage_pct for cell in v1_09_coverage.cells],
        marker_color=[
            COLORS["GreenLine"] if cell.status == "ok" else COLORS["RedLine"]
            for cell in v1_09_coverage.cells
        ],
        text=[f"{cell.coverage_pct:.1f}%" for cell in v1_09_coverage.cells],
        textposition="outside",
    ))
    _coverage_fig.add_hline(
        y=v1_09_selection.coverage_floor_pct,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        annotation_text="coverage floor",
        annotation_font_color=COLORS["RedLine"],
    )
    _coverage_fig.update_layout(
        height=340,
        xaxis=dict(title="Subgroup", gridcolor="#f1f5f9"),
        yaxis=dict(title="Coverage (%)", gridcolor="#f1f5f9", range=[0, 110]),
        margin=dict(l=60, r=20, t=35, b=80),
    )
    apply_plotly_theme(_coverage_fig)

    _coverage_rows = "".join(
        f"""
        <tr>
          <td>{cell.label}</td>
          <td>{cell.coverage_pct:.1f}%</td>
          <td>{cell.risk_score:.1f}</td>
          <td>{cell.status}</td>
        </tr>
        """
        for cell in v1_09_coverage.cells
    )
    _validation_items = "".join(f"<li>{test}</li>" for test in v1_09_selection.validation_tests)
    _rejections = "".join(f"<li>{item}</li>" for item in v1_09_decision.rejected_alternatives)

    _part_a = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Utility-Cost Frontier</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which data policy buys useful quality and coverage without exceeding system budgets?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Quantity can improve utility, but noisy labels, storage, compute, privacy, and retention cost grow too.</li>
            <li>Coverage and rare-event score are separate from average label quality.</li>
            <li>Feasibility checks quality, coverage, rare-event coverage, cost, and storage together.</li>
          </ul>
          <div class="mlsysbook-callout"><strong>Track selection story:</strong> {v1_09_selection.selection_story}</div>
        </div>
        """),
        v1_09_value_prediction,
        v1_09_fraction_multiplier,
        mo.as_html(_frontier_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Table Fallback</h2>
          <table class="mlsysbook-table">
            <thead>
              <tr>
                <th>Policy</th><th>Selected</th><th>Quality</th><th>Coverage</th>
                <th>Rare events</th><th>Cost</th><th>Storage</th><th>Feasible</th><th>Dominant risk</th>
              </tr>
            </thead>
            <tbody>{_frontier_rows}</tbody>
          </table>
        </div>
        """),
    ])

    _part_b = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Coverage Risk</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which subgroup remains under-covered by the selected policy?</div>
        </div>
        """),
        mo.Html("""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>A global utility score can rise while one subgroup gets worse.</li>
            <li>Risk combines coverage shortfall with the track's harm or safety weight for that subgroup.</li>
            <li>The worst subgroup should drive the next-data recommendation.</li>
          </ul>
        </div>
        """),
        mo.as_html(_coverage_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Coverage Table</h2>
          <table class="mlsysbook-table">
            <thead><tr><th>Subgroup</th><th>Coverage</th><th>Risk score</th><th>Status</th></tr></thead>
            <tbody>{_coverage_rows}</tbody>
          </table>
          <div class="mlsysbook-callout"><strong>Worst subgroup:</strong> {v1_09_coverage.worst_subgroup}</div>
        </div>
        """),
    ])

    _part_c = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Data Policy</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            What data should be collected next, and what blind spot remains?</div>
        </div>
        """),
        mo.Html("""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>A data policy should name acquisition, curation, rare-event handling, and accepted blind spots.</li>
            <li>The next data to collect should be tied to the worst subgroup or dominant risk.</li>
            <li>Data policy decisions carry forward into robustness, responsibility, and operations labs.</li>
          </ul>
        </div>
        """),
        v1_09_policy_choice,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Selected policy</strong>{v1_09_decision.selected_label}</div>
            <div class="mlsysbook-field"><strong>Feasible</strong>{'yes' if v1_09_decision.feasible else 'no - violation'}</div>
            <div class="mlsysbook-field"><strong>Utility</strong>{v1_09_decision.utility_score:.1f}</div>
            <div class="mlsysbook-field"><strong>Dominant risk</strong>{v1_09_decision.dominant_risk}</div>
            <div class="mlsysbook-field"><strong>Worst subgroup</strong>{v1_09_decision.worst_subgroup}</div>
            <div class="mlsysbook-field"><strong>Selected examples</strong>{v1_09_selected_utility.selected_examples_k:.1f}k {v1_09_selection.dataset_unit}</div>
          </div>
          <div class="mlsysbook-callout"><strong>Next data:</strong> {v1_09_decision.next_data}</div>
          <div class="mlsysbook-callout"><strong>Accepted blind spot:</strong> {v1_09_decision.accepted_blind_spot}</div>
          <div class="mlsysbook-callout"><strong>Memo decision:</strong> {v1_09_decision.memo_summary}</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Rejected Alternatives</h2>
          <ul class="mlsysbook-list">{_rejections}</ul>
          <h2>Validation Tests</h2>
          <ul class="mlsysbook-list">{_validation_items}</ul>
        </div>
        """),
        mo.Html('<div class="mlsysbook-panel"><h2>Reflection</h2></div>'),
        v1_09_reflection,
    ])

    mo.ui.tabs({
        "Part A · Frontier": _part_a,
        "Part B · Coverage": _part_b,
        "Part C · Policy": _part_c,
    })
    return


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    v1_09_decision,
    v1_09_profile,
    v1_09_selected_utility,
    v1_09_selection,
    v1_09_value_prediction,
    v1_09_variant,
):
    if v1_09_value_prediction.value is not None:
        ledger.save(chapter=9, design={
            "chapter": "v1_09",
            "track_id": v1_09_profile.track_id,
            "scenario_id": v1_09_variant.scenario_id,
            "hardware_ref": v1_09_selection.hardware_ref,
            "model_ref": v1_09_selection.model_ref,
            "completed": True,
            "value_prediction": v1_09_value_prediction.value,
            "selected_policy": v1_09_decision.selected_id,
            "dominant_risk": v1_09_decision.dominant_risk,
            "worst_subgroup": v1_09_decision.worst_subgroup,
            "selected_examples_k": v1_09_selected_utility.selected_examples_k,
        })

    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Synthesis</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Track</strong>{v1_09_selection.label}</div>
            <div class="mlsysbook-field"><strong>Selected policy</strong>{v1_09_decision.selected_label}</div>
            <div class="mlsysbook-field"><strong>Worst subgroup</strong>{v1_09_decision.worst_subgroup}</div>
            <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_09_decision.residual_risk}</div>
          </div>
        </div>
        """),
        mo.Html("""
        <div class="mlsysbook-panel">
          <h2>Big Takeaways</h2>
          <ul class="mlsysbook-list">
            <li><strong>More data is not automatically better.</strong> Quantity can raise cost, storage, noise, privacy burden, or compute without fixing coverage.</li>
            <li><strong>Coverage is a systems metric.</strong> Subgroups, rare events, and blind spots decide whether the data supports deployment.</li>
            <li><strong>A data policy is a decision artifact.</strong> It must name next data, accepted blind spots, and validation tests.</li>
          </ul>
        </div>
        """),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">09 &middot; Selection Paradox</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v1_09_profile.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">ARTIFACT</span>
            <span class="hud-value">{v1_09_selection.report_artifact}</span>
            <span class="hud-label">STATUS</span>
            <span class="hud-active">ACTIVE</span>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    report_export_panel,
    v1_09_coverage,
    v1_09_decision,
    v1_09_fraction_multiplier,
    v1_09_frontier,
    v1_09_metadata,
    v1_09_profile,
    v1_09_reflection,
    v1_09_selected_utility,
    v1_09_selection,
    v1_09_value_prediction,
    v1_09_variant,
):
    _incomplete = []
    if v1_09_value_prediction.value is None:
        _incomplete.append("Part A selection-pressure prediction")
    if not str(v1_09_reflection.value or "").strip():
        _incomplete.append("Part C reflection")

    _report = build_lab_report(
        v1_09_metadata,
        track=v1_09_profile.label,
        scenario=v1_09_variant.workload_summary,
        learning_objectives=(
            "Compare quality, quantity, coverage, rare-event coverage, cost, and storage across data policies.",
            "Inspect subgroup coverage to find hidden blind spots behind aggregate utility.",
            "Choose a data selection policy and name next data, accepted blind spot, and validation requirement.",
        ),
        predictions={
            "dominant_selection_pressure": v1_09_value_prediction.value,
        },
        knob_settings={
            "fraction_multiplier": v1_09_fraction_multiplier.value,
            "selected_policy": v1_09_decision.selected_id,
        },
        evidence_summary={
            "hardware_ref": v1_09_selection.hardware_ref,
            "model_ref": v1_09_selection.model_ref,
            "dataset_unit": v1_09_selection.dataset_unit,
            "selected_examples_k": v1_09_selected_utility.selected_examples_k,
            "utility_score": v1_09_decision.utility_score,
            "dominant_risk": v1_09_decision.dominant_risk,
            "worst_subgroup": v1_09_decision.worst_subgroup,
            "next_data": v1_09_decision.next_data,
        },
        final_decision=v1_09_decision.memo_summary,
        big_takeaways=(
            "Data selection trades quality, quantity, coverage, rare-event evidence, cost, and storage.",
            "A global utility score can hide subgroup or rare-event failure.",
            "A defensible data policy names what to collect next and what blind spot remains.",
        ),
        reflections={
            "student_reflection": v1_09_reflection.value,
            "accepted_blind_spot": v1_09_decision.accepted_blind_spot,
            "validation_requirement": v1_09_decision.validation_requirement,
            "report_artifact": v1_09_selection.report_artifact,
        },
        residual_risk=v1_09_decision.residual_risk,
        source_trace={
            "track_id": v1_09_profile.track_id,
            "scenario_id": v1_09_variant.scenario_id,
            "hardware_ref": v1_09_variant.hardware_ref,
            "model_ref": v1_09_variant.model_ref,
            "shared_helper": "mlsysbook_labs.selection",
            "source_policy": v1_09_profile.source_policy,
        },
        result_snapshot={
            "selection_profile": v1_09_selection,
            "frontier": v1_09_frontier,
            "coverage": v1_09_coverage,
            "selected_utility": v1_09_selected_utility,
            "decision": v1_09_decision,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-09 data selection policy memo is generated locally from "
                "the selected track, your inputs, and the computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
