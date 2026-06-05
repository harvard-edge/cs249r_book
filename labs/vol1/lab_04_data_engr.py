import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ===========================================================================
# ZONE A: OPENING
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
        data_pipeline_profile,
        evaluate_pipeline,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        movement_frontier,
        part_workflow,
        pipeline_architecture,
        report_export_panel,
        resolve_mlsysim_ref,
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
        data_pipeline_profile,
        evaluate_pipeline,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        mo,
        movement_frontier,
        pipeline_architecture,
        part_workflow,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_04_metadata = get_lab_metadata("vol1/lab_04_data_engr.py")
    return (v1_04_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_04_track_picker = track_selector(default=_default_track)
    v1_04_track_picker
    return (v1_04_track_picker,)


@app.cell
def _(
    data_pipeline_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_04_track_picker,
):
    v1_04_track_id = v1_04_track_picker.value
    v1_04_profile = get_track_profile(v1_04_track_id)
    v1_04_variant = get_lab_track_variant("v1_04_data_gravity", v1_04_profile.track_id)
    v1_04_hardware = resolve_mlsysim_ref(v1_04_variant.hardware_ref)
    v1_04_model = resolve_mlsysim_ref(v1_04_variant.model_ref)
    v1_04_pipeline_profile = data_pipeline_profile(
        v1_04_profile,
        v1_04_variant,
        v1_04_hardware,
        v1_04_model,
    )
    return (
        v1_04_hardware,
        v1_04_model,
        v1_04_pipeline_profile,
        v1_04_profile,
        v1_04_track_id,
        v1_04_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
        track_arc_context,
    v1_04_metadata,
    v1_04_pipeline_profile,
    v1_04_profile,
    v1_04_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 04
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Data Gravity
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Ingest &middot; Preprocess &middot; Store &middot; Move &middot; Retain
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 820px; line-height: 1.65;">
                {v1_04_variant.workload_summary} This lab asks where data should be
                processed, moved, cached, and retained for the selected track.
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
                    {v1_04_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_04_pipeline_profile.data_source}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Pipeline Bottleneck</span>
                <span class="badge badge-warn">Movement Frontier</span>
                <span class="badge badge-fail">Retention Risk</span>
            </div>
        </div>
        """),
        track_context(v1_04_profile),
        track_arc_context(v1_04_profile, v1_04_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, part_workflow, v1_04_pipeline_profile):
    mo.vstack([
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
            <div style="margin-bottom: 3px;">1. <strong>Diagnose data feeding:</strong>
                identify which pipeline stage starves the selected track.</div>
            <div style="margin-bottom: 3px;">2. <strong>Compare movement strategies:</strong>
                trade data moved, transfer time, egress cost, quality, and privacy.</div>
            <div style="margin-bottom: 3px;">3. <strong>Choose architecture:</strong>
                define preprocessing and retention policy plus the accepted data risk.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                Should {v1_04_pipeline_profile.label} move raw data, move compute,
                summarize locally, or retain only selected evidence?
            </div>
        </div>
    </div>
    """),
    part_workflow(
        "Data Gravity Workflow",
        (
            {
                "part": "Part A",
                "concept": "Feed The Model",
                "prediction": "Predict which pipeline stage will bottleneck the selected track.",
                "controls": "Change the sampling or traffic multiplier.",
                "evidence": "Inspect stage utilization, effective rate, raw data per day, and storage window.",
                "decision": "Decide which stage must be redesigned first.",
            },
            {
                "part": "Part B",
                "concept": "Data Movement Frontier",
                "prediction": "Predict whether moving raw data, summaries, or compute is best.",
                "controls": "Select a movement strategy, network budget, and dataset size.",
                "evidence": "Compare data moved, transfer time, egress cost, quality, and privacy risk.",
                "decision": "Choose the movement strategy you can defend for the track.",
            },
            {
                "part": "Part C",
                "concept": "Pipeline Architecture",
                "prediction": "Predict what data must be retained for later debugging or governance.",
                "controls": "Select retention policy and architecture stance.",
                "evidence": "Compare retained data, quality retained, bottleneck, and accepted risk.",
                "decision": "Write the data policy and name the evidence you are willing to lose.",
            },
        ),
        scenario=(
            f"{v1_04_pipeline_profile.label} has data from "
            f"{v1_04_pipeline_profile.data_source}; the lab asks where to process, move, and retain it."
        ),
        reflection="Carry one pipeline bottleneck, one movement choice, and one accepted data risk into the report.",
    ),
    ])
    return


# ===========================================================================
# ZONE B: CONTROLS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_04_pipeline_profile):
    v1_04_bottleneck_prediction = mo.ui.radio(
        options={
            "Ingest cannot keep up with bursts": "ingest",
            "Preprocessing cannot feed downstream compute": "preprocess",
            "Storage or retention dominates": "storage",
            "Upload or movement dominates": "movement",
        },
        label=f"{v1_04_pipeline_profile.label}: where will the data pipeline bottleneck?",
    )
    v1_04_bottleneck_prediction
    return (v1_04_bottleneck_prediction,)


@app.cell(hide_code=True)
def _(mo, v1_04_pipeline_profile):
    v1_04_sample_multiplier = mo.ui.slider(
        start=v1_04_pipeline_profile.sample_min,
        stop=v1_04_pipeline_profile.sample_max,
        value=v1_04_pipeline_profile.default_sample_multiplier,
        step=v1_04_pipeline_profile.sample_step,
        label="Sampling or traffic multiplier",
    )
    v1_04_sample_multiplier
    return (v1_04_sample_multiplier,)


@app.cell(hide_code=True)
def _(mo, v1_04_pipeline_profile):
    _strategy_options = {strategy.label: strategy.strategy_id for strategy in v1_04_pipeline_profile.strategies}
    v1_04_strategy = mo.ui.dropdown(
        options=_strategy_options,
        value=v1_04_pipeline_profile.strategies[0].label,
        label="Movement strategy",
    )
    v1_04_dataset_gb = mo.ui.slider(
        start=1,
        stop=5000,
        value=500,
        step=50,
        label="Dataset or event window to move (GB)",
    )
    v1_04_network_gbps = mo.ui.dropdown(
        options={"1 Gbps": 1, "10 Gbps": 10, "25 Gbps": 25, "100 Gbps": 100},
        value="10 Gbps",
        label="Network bandwidth",
    )
    return (v1_04_dataset_gb, v1_04_network_gbps, v1_04_strategy)


@app.cell(hide_code=True)
def _(mo, v1_04_pipeline_profile):
    _retention_options = {policy: policy for policy in v1_04_pipeline_profile.retention_options}
    v1_04_retention_policy = mo.ui.dropdown(
        options=_retention_options,
        value=v1_04_pipeline_profile.retention_options[0],
        label="Retention policy",
    )
    v1_04_reflection = mo.ui.text_area(
        label="Reflection",
        placeholder="Name the data you would keep, the data you would discard, and the risk this creates.",
        full_width=True,
    )
    return (v1_04_reflection, v1_04_retention_policy)


@app.cell
def _(
    evaluate_pipeline,
    movement_frontier,
    pipeline_architecture,
    v1_04_dataset_gb,
    v1_04_network_gbps,
    v1_04_pipeline_profile,
    v1_04_retention_policy,
    v1_04_sample_multiplier,
    v1_04_strategy,
):
    v1_04_pipeline_result = evaluate_pipeline(
        v1_04_pipeline_profile,
        sample_multiplier=v1_04_sample_multiplier.value,
    )
    v1_04_movement_result = movement_frontier(
        v1_04_pipeline_profile,
        strategy_id=v1_04_strategy.value,
        dataset_gb=v1_04_dataset_gb.value,
        network_gbps=v1_04_network_gbps.value,
    )
    v1_04_architecture = pipeline_architecture(
        v1_04_pipeline_profile,
        v1_04_pipeline_result,
        v1_04_movement_result,
        retention_policy=v1_04_retention_policy.value,
    )
    return (v1_04_architecture, v1_04_movement_result, v1_04_pipeline_result)


# ===========================================================================
# ZONE C: PARTS
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    v1_04_bottleneck_prediction,
    v1_04_pipeline_profile,
    v1_04_pipeline_result,
    v1_04_sample_multiplier,
):
    _fig = go.Figure()
    _stage_names = [stage.stage for stage in v1_04_pipeline_result.stages]
    _util = [stage.utilization_pct for stage in v1_04_pipeline_result.stages]
    _colors = [COLORS["RedLine"] if value > 100 else COLORS["GreenLine"] for value in _util]
    _fig.add_trace(go.Bar(
        x=_stage_names,
        y=_util,
        marker_color=_colors,
        text=[f"{value:.0f}%" for value in _util],
        textposition="outside",
    ))
    _fig.add_hline(y=100, line_dash="dash", line_color=COLORS["RedLine"], line_width=1.5)
    _fig.update_layout(
        height=320,
        xaxis=dict(title="Pipeline stage", gridcolor="#f1f5f9"),
        yaxis=dict(title="Utilization (%)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=60),
    )
    apply_plotly_theme(_fig)

    _rows = []
    for stage in v1_04_pipeline_result.stages:
        _color = COLORS["GreenLine"] if stage.feasible else COLORS["RedLine"]
        _rows.append(
            f"""
            <tr>
              <td>{stage.stage}</td>
              <td style="text-align:right;">{stage.demand_mb_s:.3g} MB/s</td>
              <td style="text-align:right;">{stage.capacity_mb_s:.3g} MB/s</td>
              <td style="text-align:right; color:{_color}; font-weight:800;">{stage.utilization_pct:.1f}%</td>
            </tr>
            """
        )
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Feed The Model</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which data stage starves {v1_04_pipeline_profile.label} first?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Data gravity starts where generated data exceeds ingest, preprocessing, storage, or upload capacity.</li>
            <li>Utilization above 100% means that stage cannot keep up with the selected sample rate.</li>
            <li>Retention is a capacity choice: raw data kept too long can dominate even when streaming stages pass.</li>
          </ul>
          <div class="mlsysbook-callout"><strong>Track data source:</strong> {v1_04_pipeline_profile.data_source}</div>
        </div>
        """),
        v1_04_bottleneck_prediction,
        v1_04_sample_multiplier,
        mo.as_html(_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Bottleneck</strong>{v1_04_pipeline_result.bottleneck_stage}</div>
            <div class="mlsysbook-field"><strong>Effective rate</strong>{v1_04_pipeline_result.effective_rate_mb_s:.3g} MB/s</div>
            <div class="mlsysbook-field"><strong>Raw data per day</strong>{v1_04_pipeline_result.daily_raw_gb:.1f} GB</div>
            <div class="mlsysbook-field"><strong>Local storage window</strong>{v1_04_pipeline_result.local_storage_days:.2f} days</div>
          </div>
          <table style="width:100%; border-collapse:collapse; margin-top:14px; font-size:0.88rem;">
            <thead>
              <tr style="border-bottom:1px solid {COLORS['Border']}; color:{COLORS['TextMuted']}; text-align:left;">
                <th>Stage</th><th style="text-align:right;">Demand</th><th style="text-align:right;">Capacity</th><th style="text-align:right;">Utilization</th>
              </tr>
            </thead>
            <tbody>{''.join(_rows)}</tbody>
          </table>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    evaluate_pipeline,
    mo,
    movement_frontier,
    source_trace,
    v1_04_dataset_gb,
    v1_04_movement_result,
    v1_04_network_gbps,
    v1_04_pipeline_profile,
    v1_04_strategy,
):
    _strategy_rows = []
    for strategy in v1_04_pipeline_profile.strategies:
        _movement = movement_frontier(
            v1_04_pipeline_profile,
            strategy_id=strategy.strategy_id,
            dataset_gb=v1_04_dataset_gb.value,
            network_gbps=v1_04_network_gbps.value,
        )
        _color = COLORS["GreenLine"] if strategy.strategy_id == v1_04_movement_result.strategy_id else COLORS["TextSec"]
        _strategy_rows.append(
            f"""
            <tr>
              <td style="color:{_color}; font-weight:800;">{strategy.label}</td>
              <td style="text-align:right;">{_movement.data_moved_gb:.1f} GB</td>
              <td style="text-align:right;">{_movement.transfer_hours:.2f} hr</td>
              <td style="text-align:right;">${_movement.egress_cost:.2f}</td>
              <td style="text-align:right;">{_movement.quality_retained_pct:.1f}%</td>
              <td>{_movement.privacy_risk}</td>
            </tr>
            """
        )
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Data Movement Frontier</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Should the architecture move raw data, summarize locally, cache, shard, or retain only selected evidence?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Moving less data usually reduces latency and cost but can discard debugging or rare-event evidence.</li>
            <li>Privacy risk depends on what leaves the source, not just where the model runs.</li>
            <li>Quality retained is a modeling assumption in the track variant and must be defended in the report.</li>
          </ul>
        </div>
        """),
        mo.hstack([v1_04_strategy, v1_04_network_gbps], justify="start", gap="2rem"),
        v1_04_dataset_gb,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Selected strategy</strong>{v1_04_movement_result.strategy_label}</div>
            <div class="mlsysbook-field"><strong>Data moved</strong>{v1_04_movement_result.data_moved_gb:.1f} GB</div>
            <div class="mlsysbook-field"><strong>Transfer time</strong>{v1_04_movement_result.transfer_hours:.2f} hours</div>
            <div class="mlsysbook-field"><strong>Egress cost</strong>${v1_04_movement_result.egress_cost:.2f}</div>
            <div class="mlsysbook-field"><strong>Quality retained</strong>{v1_04_movement_result.quality_retained_pct:.1f}%</div>
            <div class="mlsysbook-field"><strong>Privacy risk</strong>{v1_04_movement_result.privacy_risk}</div>
          </div>
          <table style="width:100%; border-collapse:collapse; margin-top:14px; font-size:0.82rem;">
            <thead>
              <tr style="border-bottom:1px solid {COLORS['Border']}; text-align:left; color:{COLORS['TextMuted']};">
                <th>Strategy</th><th style="text-align:right;">Moved</th><th style="text-align:right;">Transfer</th>
                <th style="text-align:right;">Cost</th><th style="text-align:right;">Quality</th><th>Privacy risk</th>
              </tr>
            </thead>
            <tbody>{''.join(_strategy_rows)}</tbody>
          </table>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    mo,
    source_trace,
    v1_04_architecture,
    v1_04_pipeline_profile,
    v1_04_reflection,
    v1_04_retention_policy,
):
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Pipeline Architecture</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            What data do you keep, what do you discard, and what risk does that create?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>A retention policy is a model-quality and governance decision, not a storage afterthought.</li>
            <li>The right answer differs by track: privacy, battery, rare events, and accelerator starvation pull in different directions.</li>
            <li>The final memo should name the accepted data loss or bias explicitly.</li>
          </ul>
        </div>
        """),
        v1_04_retention_policy,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Architecture</strong>{v1_04_architecture.strategy_label}</div>
            <div class="mlsysbook-field"><strong>Retention policy</strong>{v1_04_architecture.retention_policy}</div>
            <div class="mlsysbook-field"><strong>Bottleneck</strong>{v1_04_architecture.bottleneck_stage}</div>
            <div class="mlsysbook-field"><strong>Retained data</strong>{v1_04_architecture.retained_gb:.1f} GB</div>
            <div class="mlsysbook-field"><strong>Quality retained</strong>{v1_04_architecture.quality_retained_pct:.1f}%</div>
            <div class="mlsysbook-field"><strong>Accepted risk</strong>{v1_04_architecture.accepted_data_risk}</div>
          </div>
          <div class="mlsysbook-callout"><strong>Memo decision:</strong> {v1_04_architecture.memo_summary}</div>
        </div>
        """),
        mo.Html('<div class="mlsysbook-panel"><h2>Reflection</h2></div>'),
        v1_04_reflection,
    ])
    return


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    ledger,
    mo,
    v1_04_architecture,
    v1_04_bottleneck_prediction,
    v1_04_movement_result,
    v1_04_pipeline_profile,
    v1_04_pipeline_result,
    v1_04_profile,
    v1_04_variant,
):
    if v1_04_bottleneck_prediction.value is not None:
        ledger.save(chapter=4, design={
            "chapter": "v1_04",
            "track_id": v1_04_profile.track_id,
            "scenario_id": v1_04_variant.scenario_id,
            "hardware_ref": v1_04_pipeline_profile.hardware_ref,
            "model_ref": v1_04_pipeline_profile.model_ref,
            "completed": True,
            "bottleneck_prediction": v1_04_bottleneck_prediction.value,
            "actual_bottleneck": v1_04_pipeline_result.bottleneck_stage,
            "movement_strategy": v1_04_movement_result.strategy_id,
            "retention_policy": v1_04_architecture.retention_policy,
            "accepted_data_risk": v1_04_architecture.accepted_data_risk,
        })

    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Synthesis</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Track</strong>{v1_04_pipeline_profile.label}</div>
            <div class="mlsysbook-field"><strong>Bottleneck</strong>{v1_04_pipeline_result.bottleneck_stage}</div>
            <div class="mlsysbook-field"><strong>Strategy</strong>{v1_04_movement_result.strategy_label}</div>
            <div class="mlsysbook-field"><strong>Accepted risk</strong>{v1_04_architecture.accepted_data_risk}</div>
          </div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Big Takeaways</h2>
          <ul class="mlsysbook-list">
            <li><strong>Data has gravity.</strong> The data source, rate, movement path, and retention policy can dominate the model design.</li>
            <li><strong>Moving less data changes evidence.</strong> Summaries and snippets reduce cost but can erase rare failures.</li>
            <li><strong>Retention is part of system design.</strong> Storage, privacy, and quality must be defended together.</li>
          </ul>
        </div>
        """),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">04 &middot; Data Gravity</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v1_04_profile.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">ARTIFACT</span>
            <span class="hud-value">{v1_04_pipeline_profile.report_artifact}</span>
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
    v1_04_architecture,
    v1_04_bottleneck_prediction,
    v1_04_metadata,
    v1_04_movement_result,
    v1_04_pipeline_profile,
    v1_04_pipeline_result,
    v1_04_profile,
    v1_04_reflection,
    v1_04_sample_multiplier,
    v1_04_variant,
):
    _incomplete = []
    if v1_04_bottleneck_prediction.value is None:
        _incomplete.append("Part A bottleneck prediction")
    if not str(v1_04_reflection.value or "").strip():
        _incomplete.append("Part C reflection")

    _report = build_lab_report(
        v1_04_metadata,
        track=v1_04_profile.label,
        scenario=v1_04_variant.workload_summary,
        learning_objectives=(
            "Identify the pipeline stage that bottlenecks the selected track.",
            "Compare data movement strategies across movement, cost, quality, and privacy.",
            "Choose a preprocessing and retention architecture with accepted data risk.",
        ),
        predictions={
            "pipeline_bottleneck": v1_04_bottleneck_prediction.value,
        },
        knob_settings={
            "sample_multiplier": v1_04_sample_multiplier.value,
            "movement_strategy": v1_04_movement_result.strategy_id,
            "retention_policy": v1_04_architecture.retention_policy,
        },
        evidence_summary={
            "hardware_ref": v1_04_pipeline_profile.hardware_ref,
            "model_ref": v1_04_pipeline_profile.model_ref,
            "data_source": v1_04_pipeline_profile.data_source,
            "bottleneck_stage": v1_04_pipeline_result.bottleneck_stage,
            "daily_raw_gb": v1_04_pipeline_result.daily_raw_gb,
            "data_moved_gb": v1_04_movement_result.data_moved_gb,
            "egress_cost": v1_04_movement_result.egress_cost,
            "quality_retained_pct": v1_04_movement_result.quality_retained_pct,
            "accepted_data_risk": v1_04_architecture.accepted_data_risk,
        },
        final_decision=v1_04_architecture.memo_summary,
        big_takeaways=(
            "Data placement is a first-order ML systems design choice.",
            "The selected track changes whether privacy, battery, rare events, or accelerator starvation dominates.",
            "A data architecture must name the evidence it discards.",
        ),
        reflections={
            "student_reflection": v1_04_reflection.value,
            "privacy_stance": v1_04_pipeline_profile.privacy_stance,
            "report_artifact": v1_04_pipeline_profile.report_artifact,
        },
        residual_risk=v1_04_architecture.accepted_data_risk,
        source_trace={
            "track_id": v1_04_profile.track_id,
            "scenario_id": v1_04_variant.scenario_id,
            "hardware_ref": v1_04_variant.hardware_ref,
            "model_ref": v1_04_variant.model_ref,
            "shared_helper": "mlsysbook_labs.data_pipeline",
            "source_policy": v1_04_profile.source_policy,
        },
        result_snapshot={
            "pipeline_profile": v1_04_pipeline_profile,
            "pipeline_result": v1_04_pipeline_result,
            "movement_frontier": v1_04_movement_result,
            "architecture": v1_04_architecture,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-04 data pipeline memo is generated locally from the selected track, "
                "your inputs, and the computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
