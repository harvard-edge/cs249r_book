import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


# ===========================================================================
# ZONE A: OPENING
# ===========================================================================


@app.cell
async def _():
    import marimo as mo
    import html
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
    from mlsysim.labs.components import MathPeek
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        deployment_mitigation,
        deployment_track_profile,
        evaluate_deployment_envelope,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        sweep_deployment_knob,
        track_arc_context,
        track_context,
        track_selector,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        COLORS,
        LAB_CSS,
        MathPeek,
        apply_plotly_theme,
        build_lab_report,
        deployment_mitigation,
        deployment_track_profile,
        evaluate_deployment_envelope,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        html,
        ledger,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        sweep_deployment_knob,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_02_metadata = get_lab_metadata("vol1/lab_02_ml_systems.py")
    return (v1_02_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    v1_02_saved_track = ledger.get_track()
    v1_02_default_track = (
        v1_02_saved_track
        if v1_02_saved_track and v1_02_saved_track != "NONE"
        else "iphone"
    )
    v1_02_track_picker = track_selector(default=v1_02_default_track)
    v1_02_track_picker
    return (v1_02_track_picker,)


@app.cell
def _(
    deployment_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_02_track_picker,
):
    v1_02_track_id = v1_02_track_picker.value
    v1_02_profile = get_track_profile(v1_02_track_id)
    v1_02_variant = get_lab_track_variant(
        "v1_02_physics_of_deployment",
        v1_02_profile.track_id,
    )
    v1_02_hardware = resolve_mlsysim_ref(v1_02_variant.hardware_ref)
    v1_02_model = resolve_mlsysim_ref(v1_02_variant.model_ref)
    v1_02_deployment = deployment_track_profile(
        v1_02_profile,
        v1_02_variant,
        v1_02_hardware,
        v1_02_model,
    )
    return (
        v1_02_deployment,
        v1_02_hardware,
        v1_02_model,
        v1_02_profile,
        v1_02_track_id,
        v1_02_variant,
    )


@app.cell
def _(
    deployment_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
):
    v1_02_track_ids = ("iphone", "oura_ring", "robotaxi", "cloud_fleet")
    v1_02_all_deployments = {}
    for _track_id in v1_02_track_ids:
        track_profile = get_track_profile(_track_id)
        variant = get_lab_track_variant("v1_02_physics_of_deployment", _track_id)
        hardware = resolve_mlsysim_ref(variant.hardware_ref)
        model = resolve_mlsysim_ref(variant.model_ref)
        v1_02_all_deployments[_track_id] = deployment_track_profile(
            track_profile,
            variant,
            hardware,
            model,
        )
    return (v1_02_all_deployments, v1_02_track_ids)


@app.cell
def _(COLORS, html):
    def v1_02_wall_category(wall):
        wall = str(wall or "").lower()
        if "memory" in wall or "flash" in wall or "ota" in wall:
            return "memory_or_flash"
        if "latency" in wall:
            return "latency"
        if "energy" in wall or "power" in wall or "thermal" in wall:
            return "energy_or_power"
        if "bandwidth" in wall or "cost" in wall:
            return "bandwidth_or_cost"
        return "unknown"

    def v1_02_option_label(value, options):
        if not isinstance(options, dict):
            return "not selected" if value is None else str(value)
        for label, option_value in options.items():
            if option_value == value:
                return label
        return "not selected"

    def v1_02_placement_id(raw_value, deployment):
        for option in deployment.placement_options:
            if raw_value in (option.placement_id, option.label):
                return option.placement_id
        return deployment.placement_options[0].placement_id

    def v1_02_mitigation_value(raw_value, deployment):
        for mitigation in deployment.mitigation_options:
            if raw_value == mitigation:
                return mitigation
        return deployment.mitigation_options[0] if deployment.mitigation_options else "No mitigation selected."

    def v1_02_clamp_workload(deployment, value):
        return max(deployment.knob_min, min(deployment.knob_max, float(value)))

    def v1_02_strategy_placement_id(deployment, strategy):
        index_by_strategy = {"primary": 0, "hybrid": 1, "offload": 2}
        index = min(index_by_strategy.get(strategy, 0), len(deployment.placement_options) - 1)
        return deployment.placement_options[index].placement_id

    def v1_02_worst_headroom(result):
        return min(check.headroom_pct for check in result.checks)

    def v1_02_first_wall_check(result):
        for check in result.checks:
            if check.name == result.first_wall:
                return check
        return min(result.checks, key=lambda check: check.headroom_pct)

    def v1_02_latency_terms(deployment, result):
        network_ms = 0.0
        for option in deployment.placement_options:
            if option.placement_id == result.placement_id:
                network_ms = option.network_latency_ms
                break
        local_ms = max(0.001, result.latency_ms - network_ms)
        raw_compute = max(0.05, deployment.model_flops_g / max(deployment.peak_tflops, 0.001))
        raw_memory = max(
            0.05,
            result.memory_required_mb / max(deployment.memory_bandwidth_gbs, 0.001),
        )
        raw_overhead = max(0.10, deployment.latency_ms_at_default * 0.08)
        raw_total = raw_compute + raw_memory + raw_overhead
        scale = local_ms / raw_total
        return {
            "compute": raw_compute * scale,
            "memory/bandwidth": raw_memory * scale,
            "placement/network": network_ms,
            "fixed overhead": raw_overhead * scale,
        }

    def v1_02_upgrade_terms(terms, compute_multiplier, bandwidth_multiplier):
        return {
            "compute": terms["compute"] / max(float(compute_multiplier), 0.001),
            "memory/bandwidth": terms["memory/bandwidth"] / max(float(bandwidth_multiplier), 0.001),
            "placement/network": terms["placement/network"],
            "fixed overhead": terms["fixed overhead"],
        }

    def v1_02_speedup_class(speedup):
        if speedup < 1.0:
            return "worse_due_to_overhead"
        if speedup >= 1.80:
            return "near_2x"
        if speedup >= 1.20:
            return "twenty_to_forty_percent"
        return "less_than_ten_percent"

    def v1_02_part_banner(part, color, title, duration, why):
        return f"""
        <div style="margin: 12px 0 16px 0;">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="background:{color}; color:white; border-radius:50%;
                            width:32px; height:32px; display:inline-flex;
                            align-items:center; justify-content:center; font-size:0.9rem;
                            font-weight:800; flex-shrink:0;">{html.escape(part)}</div>
                <div style="flex:1; height:2px; background:{COLORS['Border']};"></div>
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                            text-transform:uppercase; letter-spacing:0.12em;">
                    Part {html.escape(part)} - {html.escape(duration)}
                </div>
            </div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['Text']};
                        margin-top:8px; line-height:1.2;">{html.escape(title)}</div>
            <div style="color:{COLORS['TextSec']}; font-size:0.92rem; margin-top:6px;
                        line-height:1.55; max-width:760px;">{html.escape(why)}</div>
        </div>
        """

    def v1_02_fields(fields):
        return "".join(
            f"""
            <div class="mlsysbook-field">
              <strong>{html.escape(str(key))}</strong>{html.escape(str(value))}
            </div>
            """
            for key, value in fields.items()
        )

    def v1_02_reveal_card(title, predicted, actual, body, *, tone="info"):
        color = {
            "success": COLORS["GreenLine"],
            "warn": COLORS["OrangeLine"],
            "danger": COLORS["RedLine"],
            "info": COLORS["BlueLine"],
        }.get(tone, COLORS["BlueLine"])
        return f"""
        <div class="mlsysbook-panel" style="border-left:4px solid {color};">
          <div style="font-size:0.72rem; font-weight:800; color:{color};
                      text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px;">
            Prediction vs reality
          </div>
          <h2 style="margin-top:0;">{html.escape(title)}</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>You predicted</strong>{html.escape(str(predicted))}</div>
            <div class="mlsysbook-field"><strong>Instrument measured</strong>{html.escape(str(actual))}</div>
          </div>
          <p style="color:{COLORS['TextSec']}; line-height:1.6; margin-bottom:0;">
            {html.escape(str(body))}
          </p>
        </div>
        """

    def v1_02_checks_table(result):
        rows = []
        for check in result.checks:
            status = "PASS" if check.feasible else "WALL"
            color = COLORS["GreenLine"] if check.feasible else COLORS["RedLine"]
            weight = "900" if check.name == result.first_wall else "700"
            rows.append(
                f"""
                <tr>
                  <td style="font-weight:{weight};">{html.escape(check.name)}</td>
                  <td style="text-align:right;">{check.value:.3g} {html.escape(check.unit)}</td>
                  <td style="text-align:right;">{check.limit:.3g} {html.escape(check.unit)}</td>
                  <td style="text-align:right; color:{color}; font-weight:800;">{check.headroom_pct:.1f}%</td>
                  <td style="color:{color}; font-weight:800;">{status}</td>
                </tr>
                """
            )
        return f"""
        <table style="width:100%; border-collapse:collapse; margin-top:14px; font-size:0.88rem;">
          <thead>
            <tr style="border-bottom:1px solid {COLORS['Border']}; color:{COLORS['TextMuted']}; text-align:left;">
              <th>Constraint</th>
              <th style="text-align:right;">Value</th>
              <th style="text-align:right;">Limit</th>
              <th style="text-align:right;">Headroom</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
        """

    def v1_02_failure_card(result):
        wall = v1_02_first_wall_check(result)
        if result.feasible:
            return f"""
            <div class="mlsysbook-panel" style="border-left:4px solid {COLORS['GreenLine']};">
              <h2>Envelope Status</h2>
              <p style="margin:0; color:{COLORS['TextSec']}; line-height:1.6;">
                Feasible at this setting. The tightest wall is {html.escape(result.first_wall)}
                with {wall.headroom_pct:.1f}% headroom. Move the workload knob upward to find
                the reversible failure boundary.
              </p>
            </div>
            """
        return f"""
        <div class="mlsysbook-panel" style="border-left:4px solid {COLORS['RedLine']};
                    background:#fff5f5;">
          <h2>Reversible Failure State</h2>
          <p style="margin:0; color:{COLORS['Text']}; line-height:1.6;">
            <strong>{html.escape(result.first_wall)} wall:</strong>
            {wall.value:.3g} {html.escape(wall.unit)} exceeds the limit of
            {wall.limit:.3g} {html.escape(wall.unit)}. Pull back the workload
            or choose another placement to recover.
          </p>
        </div>
        """

    return (
        v1_02_checks_table,
        v1_02_clamp_workload,
        v1_02_failure_card,
        v1_02_fields,
        v1_02_first_wall_check,
        v1_02_latency_terms,
        v1_02_mitigation_value,
        v1_02_option_label,
        v1_02_part_banner,
        v1_02_placement_id,
        v1_02_reveal_card,
        v1_02_speedup_class,
        v1_02_strategy_placement_id,
        v1_02_upgrade_terms,
        v1_02_wall_category,
        v1_02_worst_headroom,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    track_arc_context,
    track_context,
    v1_02_deployment,
    v1_02_metadata,
    v1_02_profile,
    v1_02_variant,
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
                Machine Learning Systems - Volume I - Lab 02
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The Physics of Deployment
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Memory - Latency - Energy - Power - Bandwidth - Cost
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 850px; line-height: 1.65;">
                {v1_02_variant.workload_summary} You will test the selected track
                as a physical operating envelope before making a deployment preference.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Concept Modules + Synthesis - 45-55 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_02_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_02_deployment.hardware_ref}
                </span>
            </div>
        </div>
        """),
        track_context(v1_02_profile),
        track_arc_context(v1_02_profile, v1_02_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_02_deployment):
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
            <div style="margin-bottom: 3px;">1. <strong>Identify</strong> which physical quantity binds first for a deployment context.</div>
            <div style="margin-bottom: 3px;">2. <strong>Diagnose</strong> the active Iron Law latency term before choosing an optimization.</div>
            <div style="margin-bottom: 3px;">3. <strong>Defend</strong> a placement and mitigation using measured constraint evidence.</div>
        </div>
        <div style="display:flex; gap:32px; margin-top:16px; flex-wrap:wrap; border-top:1px solid {COLORS['Border']};
                    padding-top:16px;">
            <div style="flex:1; min-width:220px;">
                <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                    Prerequisites
                </div>
                <div style="font-size:0.85rem; color:{COLORS['TextSec']}; line-height:1.65;">
                    Deployment envelopes - Iron Law latency decomposition - unitful resource budgets
                </div>
            </div>
            <div style="flex:0 0 180px;">
                <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                    Duration
                </div>
                <div style="font-size:0.85rem; color:{COLORS['TextSec']}; line-height:1.65;">
                    <strong>45-55 min</strong><br/>4 parts + synthesis
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 16px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                If the model looks good in isolation, which physical amount still decides
                whether {v1_02_deployment.label} can ship it?
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo, v1_02_deployment, v1_02_variant):
    mo.callout(
        mo.md(
            f"""
            **Reading connection.** This lab uses the chapter's ML systems invariant:
            deployment is constrained by physical amounts before preference. The active
            scenario is: {v1_02_variant.workload_summary}

            **Current source profile.** Hardware: `{v1_02_deployment.hardware_ref}`.
            Model: `{v1_02_deployment.model_ref}`. Workload knob:
            `{v1_02_deployment.workload_knob}` in `{v1_02_deployment.workload_unit}`.
            """
        ),
        kind="info",
    )
    return


# ===========================================================================
# ZONE B: WIDGET CELLS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_02_deployment):
    partA_wall_options = {
        "Memory or flash/OTA fit fails first": "memory_or_flash",
        "Latency deadline fails first": "latency",
        "Power or per-inference energy fails first": "energy_or_power",
        "Bandwidth or cost fails first": "bandwidth_or_cost",
    }
    partA_prediction = mo.ui.radio(
        options=partA_wall_options,
        label=f"{v1_02_deployment.label}: which wall fails first at the default workload?",
    )
    partA_prediction
    return (partA_prediction, partA_wall_options)


@app.cell(hide_code=True)
def _(mo, v1_02_deployment):
    partA_workload = mo.ui.slider(
        start=v1_02_deployment.knob_min,
        stop=v1_02_deployment.knob_max,
        value=v1_02_deployment.default_knob,
        step=v1_02_deployment.knob_step,
        label=f"{v1_02_deployment.workload_knob} ({v1_02_deployment.workload_unit})",
    )
    partA_placement = mo.ui.dropdown(
        options={option.label: option.placement_id for option in v1_02_deployment.placement_options},
        value=v1_02_deployment.placement_options[0].label,
        label="Placement option",
    )
    return (partA_placement, partA_workload)


@app.cell(hide_code=True)
def _(mo):
    partB_speedup_options = {
        "Near 2x faster": "near_2x",
        "Only 20-40 percent faster": "twenty_to_forty_percent",
        "Less than 10 percent faster": "less_than_ten_percent",
        "Worse due to placement or fixed overhead": "worse_due_to_overhead",
    }
    partB_prediction = mo.ui.radio(
        options=partB_speedup_options,
        label="If compute throughput improves by 2x, what happens to total latency?",
    )
    partB_prediction
    return (partB_prediction, partB_speedup_options)


@app.cell(hide_code=True)
def _(mo, v1_02_deployment):
    partB_compute_multiplier = mo.ui.slider(
        start=1.0,
        stop=4.0,
        value=2.0,
        step=0.25,
        label="Compute throughput multiplier (x)",
    )
    partB_bandwidth_multiplier = mo.ui.slider(
        start=0.5,
        stop=4.0,
        value=1.0,
        step=0.25,
        label="Memory/bandwidth multiplier (x)",
    )
    partB_placement = mo.ui.dropdown(
        options={option.label: option.placement_id for option in v1_02_deployment.placement_options},
        value=v1_02_deployment.placement_options[0].label,
        label="Placement option",
    )
    return (partB_bandwidth_multiplier, partB_compute_multiplier, partB_placement)


@app.cell(hide_code=True)
def _(mo):
    partC_track_options = {
        "iPhone": "iphone",
        "Oura Ring": "oura_ring",
        "RoboTaxi": "robotaxi",
        "Cloud Fleet": "cloud_fleet",
    }
    partC_prediction = mo.ui.radio(
        options=partC_track_options,
        label="Which track has the least headroom at this normalized workload?",
    )
    partC_prediction
    return (partC_prediction, partC_track_options)


@app.cell(hide_code=True)
def _(mo):
    partC_stress = mo.ui.slider(
        start=50,
        stop=200,
        value=100,
        step=10,
        label="Workload stress (% of each track default)",
    )
    partC_placement_strategy = mo.ui.dropdown(
        options={
            "Primary local/central path": "primary",
            "Middle edge/cache/hybrid path": "hybrid",
            "Offload/batch/fallback path": "offload",
        },
        value="Primary local/central path",
        label="Comparable placement strategy",
    )
    return (partC_placement_strategy, partC_stress)


@app.cell(hide_code=True)
def _(mo, v1_02_deployment):
    partD_prediction = mo.ui.radio(
        options={option.label: option.placement_id for option in v1_02_deployment.placement_options},
        label="Which placement strategy will survive the stress scenario?",
    )
    partD_prediction
    return (partD_prediction,)


@app.cell(hide_code=True)
def _(mo, v1_02_deployment):
    partD_default_stress = min(
        v1_02_deployment.knob_max,
        max(v1_02_deployment.knob_min, v1_02_deployment.default_knob * 1.5),
    )
    partD_workload = mo.ui.slider(
        start=v1_02_deployment.knob_min,
        stop=v1_02_deployment.knob_max,
        value=partD_default_stress,
        step=v1_02_deployment.knob_step,
        label=f"Stress workload ({v1_02_deployment.workload_unit})",
    )
    partD_placement = mo.ui.dropdown(
        options={option.label: option.placement_id for option in v1_02_deployment.placement_options},
        value=v1_02_deployment.placement_options[0].label,
        label="Design placement",
    )
    partD_mitigation = mo.ui.dropdown(
        options={item: item for item in v1_02_deployment.mitigation_options},
        value=v1_02_deployment.mitigation_options[0] if v1_02_deployment.mitigation_options else None,
        label="Mitigation",
    )
    return (partD_mitigation, partD_placement, partD_workload)


# ===========================================================================
# ZONE C: COMPUTED EVIDENCE
# ===========================================================================


@app.cell
def _(
    deployment_mitigation,
    evaluate_deployment_envelope,
    partA_placement,
    partA_workload,
    partB_bandwidth_multiplier,
    partB_compute_multiplier,
    partB_placement,
    partC_placement_strategy,
    partC_stress,
    partD_mitigation,
    partD_placement,
    partD_workload,
    sweep_deployment_knob,
    v1_02_all_deployments,
    v1_02_clamp_workload,
    v1_02_deployment,
    v1_02_latency_terms,
    v1_02_mitigation_value,
    v1_02_placement_id,
    v1_02_speedup_class,
    v1_02_strategy_placement_id,
    v1_02_track_ids,
    v1_02_upgrade_terms,
    v1_02_wall_category,
    v1_02_worst_headroom,
):
    partA_placement_id = v1_02_placement_id(partA_placement.value, v1_02_deployment)
    partA_result = evaluate_deployment_envelope(
        v1_02_deployment,
        workload_value=partA_workload.value,
        placement_id=partA_placement_id,
    )
    partA_sweep = sweep_deployment_knob(
        v1_02_deployment,
        placement_id=partA_placement_id,
        samples=40,
    )
    partA_actual_category = v1_02_wall_category(partA_result.first_wall)

    partB_placement_id = v1_02_placement_id(partB_placement.value, v1_02_deployment)
    partB_result = evaluate_deployment_envelope(
        v1_02_deployment,
        workload_value=v1_02_deployment.default_knob,
        placement_id=partB_placement_id,
    )
    partB_terms = v1_02_latency_terms(v1_02_deployment, partB_result)
    partB_upgraded_terms = v1_02_upgrade_terms(
        partB_terms,
        partB_compute_multiplier.value,
        partB_bandwidth_multiplier.value,
    )
    partB_baseline_latency = sum(partB_terms.values())
    partB_upgraded_latency = sum(partB_upgraded_terms.values())
    partB_actual_speedup = partB_baseline_latency / max(partB_upgraded_latency, 0.001)
    partB_actual_class = v1_02_speedup_class(partB_actual_speedup)
    partB_active_term = max(partB_upgraded_terms, key=lambda key: partB_upgraded_terms[key])

    partC_strategy = partC_placement_strategy.value
    partC_results = {}
    for _track_id in v1_02_track_ids:
        deployment = v1_02_all_deployments[_track_id]
        placement_id = v1_02_strategy_placement_id(deployment, partC_strategy)
        workload = v1_02_clamp_workload(
            deployment,
            deployment.default_knob * (partC_stress.value / 100.0),
        )
        result = evaluate_deployment_envelope(
            deployment,
            workload_value=workload,
            placement_id=placement_id,
        )
        partC_results[_track_id] = {
            "deployment": deployment,
            "placement_id": placement_id,
            "workload": workload,
            "result": result,
            "worst_headroom_pct": v1_02_worst_headroom(result),
        }
    partC_tightest_track = min(
        partC_results,
        key=lambda _track_id: partC_results[_track_id]["worst_headroom_pct"],
    )
    partC_first_walls_by_track = {
        _track_id: data["result"].first_wall for _track_id, data in partC_results.items()
    }
    partC_worst_headroom_by_track = {
        _track_id: round(data["worst_headroom_pct"], 2)
        for _track_id, data in partC_results.items()
    }
    partC_active_placement_id = v1_02_strategy_placement_id(v1_02_deployment, partC_strategy)
    partC_active_sweep = sweep_deployment_knob(
        v1_02_deployment,
        placement_id=partC_active_placement_id,
        samples=32,
    )

    partD_placement_id = v1_02_placement_id(partD_placement.value, v1_02_deployment)
    partD_selected_mitigation = v1_02_mitigation_value(partD_mitigation.value, v1_02_deployment)
    partD_result = evaluate_deployment_envelope(
        v1_02_deployment,
        workload_value=partD_workload.value,
        placement_id=partD_placement_id,
    )
    partD_mitigation_result = deployment_mitigation(
        v1_02_deployment,
        partD_result,
        placement_id=partD_placement_id,
    )
    partD_placement_results = []
    for option in v1_02_deployment.placement_options:
        result = evaluate_deployment_envelope(
            v1_02_deployment,
            workload_value=partD_workload.value,
            placement_id=option.placement_id,
        )
        partD_placement_results.append(
            {
                "option": option,
                "result": result,
                "worst_headroom_pct": v1_02_worst_headroom(result),
            }
        )
    feasible_partD = [row for row in partD_placement_results if row["result"].feasible]
    partD_best_row = (
        max(feasible_partD, key=lambda row: row["worst_headroom_pct"])
        if feasible_partD
        else max(partD_placement_results, key=lambda row: row["worst_headroom_pct"])
    )
    partD_actual_survivor = partD_best_row["option"].placement_id
    partD_survivor_label = partD_best_row["option"].label
    partD_residual_risk = partD_mitigation_result.new_risk
    return (
        partA_actual_category,
        partA_placement_id,
        partA_result,
        partA_sweep,
        partB_actual_class,
        partB_actual_speedup,
        partB_active_term,
        partB_baseline_latency,
        partB_placement_id,
        partB_result,
        partB_terms,
        partB_upgraded_latency,
        partB_upgraded_terms,
        partC_active_sweep,
        partC_first_walls_by_track,
        partC_results,
        partC_tightest_track,
        partC_worst_headroom_by_track,
        partD_actual_survivor,
        partD_best_row,
        partD_mitigation_result,
        partD_placement_id,
        partD_placement_results,
        partD_residual_risk,
        partD_result,
        partD_selected_mitigation,
        partD_survivor_label,
    )


# ===========================================================================
# ZONE D: TABBED CONCEPT MODULES
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    MathPeek,
    apply_plotly_theme,
    go,
    html,
    mo,
    partA_actual_category,
    partA_placement,
    partA_prediction,
    partA_result,
    partA_sweep,
    partA_wall_options,
    partA_workload,
    partB_actual_class,
    partB_actual_speedup,
    partB_active_term,
    partB_bandwidth_multiplier,
    partB_baseline_latency,
    partB_compute_multiplier,
    partB_prediction,
    partB_result,
    partB_speedup_options,
    partB_terms,
    partB_upgraded_latency,
    partB_upgraded_terms,
    partB_placement,
    partC_active_sweep,
    partC_first_walls_by_track,
    partC_placement_strategy,
    partC_prediction,
    partC_results,
    partC_stress,
    partC_tightest_track,
    partC_track_options,
    partC_worst_headroom_by_track,
    partD_actual_survivor,
    partD_mitigation,
    partD_mitigation_result,
    partD_placement,
    partD_placement_results,
    partD_prediction,
    partD_residual_risk,
    partD_result,
    partD_selected_mitigation,
    partD_survivor_label,
    partD_workload,
    source_trace,
    v1_02_checks_table,
    v1_02_deployment,
    v1_02_failure_card,
    v1_02_fields,
    v1_02_first_wall_check,
    v1_02_option_label,
    v1_02_part_banner,
    v1_02_profile,
    v1_02_reveal_card,
    v1_02_variant,
):
    def build_part_a():
        items = [
            mo.Html(v1_02_part_banner(
                "A",
                COLORS["BlueLine"],
                "Physics Before Preference",
                "10-12 min",
                "The preferred placement is irrelevant until memory, latency, energy, power, bandwidth, and cost all fit.",
            )),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <div class="mlsysbook-grid">
                {v1_02_fields({
                    "Stakeholder": v1_02_deployment.stakeholder,
                    "Release request": v1_02_variant.workload_summary,
                    "Default workload": f"{v1_02_deployment.default_knob:g} {v1_02_deployment.workload_unit}",
                    "First question": "Which physical budget fails before preference matters?",
                })}
              </div>
            </div>
            """),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Prediction Lock</h2>
              <p class="mlsysbook-action-note">
                Commit to the first wall before the envelope table appears.
              </p>
            </div>
            """),
            partA_prediction,
        ]
        if partA_prediction.value is None:
            items.append(mo.callout(mo.md("Select your first-wall prediction to unlock Part A evidence."), kind="warn"))
            return mo.vstack(items)

        predicted = v1_02_option_label(partA_prediction.value, partA_wall_options)
        actual = v1_02_option_label(partA_actual_category, partA_wall_options)
        tone = "success" if partA_prediction.value == partA_actual_category else "warn"
        wall = v1_02_first_wall_check(partA_result)
        crossing = (
            f"{partA_sweep.threshold_crossing:.1f} {v1_02_deployment.workload_unit}"
            if partA_sweep.threshold_crossing is not None
            else "not reached in this sweep"
        )
        items.extend([
            mo.hstack([partA_workload, partA_placement], widths="equal"),
            mo.Html(v1_02_reveal_card(
                "The first wall is measured, not chosen.",
                predicted,
                actual,
                f"The active check is {partA_result.first_wall}: {wall.value:.3g} {wall.unit} against a limit of {wall.limit:.3g} {wall.unit}.",
                tone=tone,
            )),
            mo.Html(v1_02_failure_card(partA_result)),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Constraint Headroom</h2>
              <div class="mlsysbook-grid">
                {v1_02_fields({
                    "Placement": partA_result.placement_label,
                    "Workload": f"{partA_result.workload_value:.1f} {v1_02_deployment.workload_unit}",
                    "Feasible": "yes" if partA_result.feasible else "no",
                    "First sweep crossing": crossing,
                })}
              </div>
              {v1_02_checks_table(partA_result)}
            </div>
            """),
            MathPeek(
                "max(value_i / limit_i) <= 1",
                {
                    "value_i": "Measured resource demand for each physical constraint.",
                    "limit_i": "Track-specific deployment budget from the selected profile.",
                    "first wall": "The constraint with the largest normalized value.",
                },
            ),
            source_trace(
                {
                    "api": "evaluate_deployment_envelope() and sweep_deployment_knob()",
                    "hardware_ref": v1_02_deployment.hardware_ref,
                    "model_ref": v1_02_deployment.model_ref,
                    "scenario_id": v1_02_variant.scenario_id,
                },
                summary="Part A source model",
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <div class="mlsysbook-grid">
                {v1_02_fields({
                    "partA_predicted_wall": partA_prediction.value,
                    "partA_actual_wall": partA_result.first_wall,
                    "partA_workload_value": f"{partA_workload.value:.1f}",
                    "partA_placement_id": partA_result.placement_id,
                })}
              </div>
            </div>
            """),
        ])
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(v1_02_part_banner(
                "B",
                COLORS["OrangeLine"],
                "Iron Law And The Bottleneck",
                "10-12 min",
                "A compute upgrade only helps the term it touches; the active latency term can stay binding.",
            )),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p style="color:{COLORS['TextSec']}; line-height:1.6;">
                The stakeholder proposes buying faster compute or offloading the workload.
                Before accepting that plan, decompose latency into compute, memory/bandwidth,
                placement/network, and fixed overhead.
              </p>
            </div>
            """),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Prediction Lock</h2>
              <p class="mlsysbook-action-note">
                Predict whether the 2x compute upgrade produces a 2x system-level win.
              </p>
            </div>
            """),
            partB_prediction,
        ]
        if partB_prediction.value is None:
            items.append(mo.callout(mo.md("Select your speedup prediction to unlock the latency waterfall."), kind="warn"))
            return mo.vstack(items)

        fig = go.Figure(go.Waterfall(
            name="Upgraded latency",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["Compute", "Memory/BW", "Network", "Overhead", "Total"],
            y=[
                partB_upgraded_terms["compute"],
                partB_upgraded_terms["memory/bandwidth"],
                partB_upgraded_terms["placement/network"],
                partB_upgraded_terms["fixed overhead"],
                0,
            ],
            connector={"line": {"color": "#94a3b8"}},
            increasing={"marker": {"color": COLORS["BlueLine"]}},
            totals={"marker": {"color": COLORS["OrangeLine"]}},
        ))
        fig.update_layout(
            height=360,
            yaxis=dict(title="Latency contribution (ms)", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=30, b=60),
            showlegend=False,
        )
        apply_plotly_theme(fig)

        predicted = v1_02_option_label(partB_prediction.value, partB_speedup_options)
        actual = v1_02_option_label(partB_actual_class, partB_speedup_options)
        tone = "success" if partB_prediction.value == partB_actual_class else "warn"
        items.extend([
            mo.hstack([partB_compute_multiplier, partB_bandwidth_multiplier, partB_placement], widths="equal"),
            mo.Html(v1_02_reveal_card(
                "Speedup is limited by the remaining term.",
                predicted,
                actual,
                f"Baseline latency is {partB_baseline_latency:.1f} ms; upgraded latency is {partB_upgraded_latency:.1f} ms, for {partB_actual_speedup:.2f}x speedup. The active term is {partB_active_term}.",
                tone=tone,
            )),
            mo.as_html(fig),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Term Ledger</h2>
              <div class="mlsysbook-grid">
                {v1_02_fields({
                    "Compute term": f"{partB_terms['compute']:.2f} ms -> {partB_upgraded_terms['compute']:.2f} ms",
                    "Memory/BW term": f"{partB_terms['memory/bandwidth']:.2f} ms -> {partB_upgraded_terms['memory/bandwidth']:.2f} ms",
                    "Placement/network": f"{partB_terms['placement/network']:.2f} ms -> {partB_upgraded_terms['placement/network']:.2f} ms",
                    "Fixed overhead": f"{partB_terms['fixed overhead']:.2f} ms -> {partB_upgraded_terms['fixed overhead']:.2f} ms",
                })}
              </div>
            </div>
            """),
            MathPeek(
                "T = D / BW + O / R + L",
                {
                    "D / BW": "Data movement divided by memory or network bandwidth.",
                    "O / R": "Operation count divided by compute rate.",
                    "L": "Placement/network latency plus fixed dispatch overhead.",
                    "speedup": "Old total latency divided by new total latency.",
                },
            ),
            source_trace(
                {
                    "api": "evaluate_deployment_envelope() plus notebook-local latency decomposition",
                    "profile": v1_02_deployment.label,
                    "result_label": "chapter-model approximation, not Engine.solve()",
                    "placement": partB_result.placement_label,
                },
                summary="Part B source model",
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <div class="mlsysbook-grid">
                {v1_02_fields({
                    "partB_predicted_speedup": partB_prediction.value,
                    "partB_actual_speedup": f"{partB_actual_speedup:.2f}",
                    "partB_active_term": partB_active_term,
                })}
              </div>
            </div>
            """),
        ])
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(v1_02_part_banner(
                "C",
                COLORS["GreenLine"],
                "Operating Envelope And First Wall",
                "10-12 min",
                "The best deployment is not universal because each track moves the first wall.",
            )),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p class="mlsysbook-action-note">
                A release review asks whether the same feature can ship across mobile,
                wearable, vehicle, and cloud contexts. Normalize the workload against
                each track's default so the comparison exposes the first wall rather
                than unit names.
              </p>
            </div>
            """),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Prediction Lock</h2>
              <p class="mlsysbook-action-note">
                Pick the context you expect to have the least remaining envelope before
                the all-track table appears.
              </p>
            </div>
            """),
            partC_prediction,
        ]
        if partC_prediction.value is None:
            items.append(mo.callout(mo.md("Select the tightest track to unlock the cross-envelope comparison."), kind="warn"))
            return mo.vstack(items)

        rows = []
        for track_id, data in partC_results.items():
            deployment = data["deployment"]
            result = data["result"]
            color = COLORS["GreenLine"] if result.feasible else COLORS["RedLine"]
            rows.append(f"""
            <tr>
              <td>{html.escape(deployment.label)}</td>
              <td>{data['workload']:.1f} {html.escape(deployment.workload_unit)}</td>
              <td>{html.escape(result.placement_label)}</td>
              <td>{html.escape(result.first_wall)}</td>
              <td style="text-align:right; color:{color}; font-weight:800;">{data['worst_headroom_pct']:.1f}%</td>
              <td style="color:{color}; font-weight:800;">{'yes' if result.feasible else 'no'}</td>
            </tr>
            """)

        fig = go.Figure()
        colors = [COLORS["GreenLine"] if ok else COLORS["RedLine"] for ok in partC_active_sweep.feasible]
        fig.add_trace(go.Scatter(
            x=list(partC_active_sweep.knob_values),
            y=list(partC_active_sweep.worst_headroom_pct),
            mode="lines+markers",
            marker=dict(color=colors, size=7),
            line=dict(color=COLORS["BlueLine"], width=2.5),
            name="Worst headroom",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["RedLine"], line_width=1.5)
        if partC_active_sweep.threshold_crossing is not None:
            fig.add_vline(
                x=partC_active_sweep.threshold_crossing,
                line_dash="dash",
                line_color=COLORS["RedLine"],
                annotation_text=f"first wall: {partC_active_sweep.threshold_wall}",
                annotation_font_color=COLORS["RedLine"],
            )
        fig.update_layout(
            height=320,
            xaxis=dict(title=f"{v1_02_deployment.workload_knob} ({v1_02_deployment.workload_unit})", gridcolor="#f1f5f9"),
            yaxis=dict(title="Worst headroom (%)", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=30, b=55),
        )
        apply_plotly_theme(fig)

        predicted = v1_02_option_label(partC_prediction.value, partC_track_options)
        actual = v1_02_option_label(partC_tightest_track, partC_track_options)
        tone = "success" if partC_prediction.value == partC_tightest_track else "warn"
        tightest_data = partC_results[partC_tightest_track]
        tightest_result = tightest_data["result"]
        tightest_consequence = (
            f"{actual} cannot ship this setting as configured because "
            f"{tightest_result.first_wall} is over budget."
            if not tightest_result.feasible
            else (
                f"{actual} survives, but it has only "
                f"{tightest_data['worst_headroom_pct']:.1f}% worst-case headroom; "
                "that remaining margin is the release risk to carry forward."
            )
        )
        items.extend([
            mo.hstack([partC_stress, partC_placement_strategy], widths="equal"),
            mo.Html(v1_02_reveal_card(
                "The tightest track is context-specific.",
                predicted,
                actual,
                f"At {partC_stress.value}% stress, {actual} has the least normalized headroom. The first walls are {partC_first_walls_by_track}.",
                tone=tone,
            )),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Context Consequence</h2>
              <p style="color:{COLORS['TextSec']}; line-height:1.6; margin-bottom:0;">
                {html.escape(tightest_consequence)}
                The same workload changed the binding axis, so the release memo must
                name the track, first wall, and residual margin rather than a universal
                "best" placement.
              </p>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>All-Track Envelope Table</h2>
              <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                <thead>
                  <tr style="border-bottom:1px solid {COLORS['Border']}; color:{COLORS['TextMuted']}; text-align:left;">
                    <th>Track</th><th>Workload</th><th>Placement</th><th>First wall</th>
                    <th style="text-align:right;">Worst headroom</th><th>Feasible</th>
                  </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
            </div>
            """),
            mo.as_html(fig),
            MathPeek(
                "headroom_i = (limit_i - value_i) / limit_i",
                {
                    "headroom_i": "Remaining fraction of a track-specific budget.",
                    "least headroom": "The most negative or smallest positive value across checks.",
                    "first wall": "The check with the smallest headroom.",
                },
            ),
            source_trace(
                {
                    "api": "get_track_profile(), get_lab_track_variant(), evaluate_deployment_envelope(), sweep_deployment_knob()",
                    "tracks": ", ".join(partC_first_walls_by_track.keys()),
                    "placement_strategy": partC_placement_strategy.value,
                },
                summary="Part C source model",
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <div class="mlsysbook-grid">
                {v1_02_fields({
                    "partC_tightest_track": partC_tightest_track,
                    "partC_first_walls_by_track": partC_first_walls_by_track,
                    "partC_worst_headroom_by_track": partC_worst_headroom_by_track,
                })}
              </div>
            </div>
            """),
        ])
        return mo.vstack(items)

    def build_part_d():
        items = [
            mo.Html(v1_02_part_banner(
                "D",
                COLORS["RedLine"],
                "Placement And Hybrid Design Review",
                "12-15 min",
                "A valid design is a placement plus a mitigation for the binding wall, not simply the fastest path.",
            )),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p style="color:{COLORS['TextSec']}; line-height:1.6;">
                The {v1_02_deployment.stakeholder} asks for one shippable deployment
                choice under stress. Pick the placement you expect to survive, then
                inspect what still fails after your mitigation.
              </p>
            </div>
            """),
            partD_prediction,
        ]
        if partD_prediction.value is None:
            items.append(mo.callout(mo.md("Select the placement you expect to survive before opening the design review."), kind="warn"))
            return mo.vstack(items)

        placement_rows = []
        for row in partD_placement_results:
            option = row["option"]
            result = row["result"]
            color = COLORS["GreenLine"] if result.feasible else COLORS["RedLine"]
            placement_rows.append(f"""
            <tr>
              <td>{html.escape(option.label)}</td>
              <td>{html.escape(result.first_wall)}</td>
              <td style="text-align:right; color:{color}; font-weight:800;">{row['worst_headroom_pct']:.1f}%</td>
              <td style="color:{color}; font-weight:800;">{'yes' if result.feasible else 'no'}</td>
              <td>{html.escape(option.risk)}</td>
            </tr>
            """)

        before_after_rows = []
        for check in partD_result.checks:
            before_status = "PASS" if check.feasible else "WALL"
            if check.name == partD_result.first_wall:
                after_status = (
                    "MITIGATED IN REVIEW"
                    if partD_mitigation_result.feasible_after_mitigation
                    else "STILL BLOCKED"
                )
            else:
                after_status = "UNCHANGED PASS" if check.feasible else "SECONDARY RISK"
            before_after_rows.append(f"""
            <tr>
              <td>{html.escape(check.name)}</td>
              <td>{check.value:.3g} / {check.limit:.3g} {html.escape(check.unit)}</td>
              <td>{before_status}</td>
              <td>{after_status}</td>
            </tr>
            """)

        selected_label = next(
            option.label for option in v1_02_deployment.placement_options
            if option.placement_id == partD_prediction.value
        )
        tone = "success" if partD_prediction.value == partD_actual_survivor else "warn"
        remaining_status = (
            "ready for a canary with risk controls"
            if partD_result.feasible or partD_mitigation_result.feasible_after_mitigation
            else "not shippable until the binding wall is reduced"
        )
        items.extend([
            mo.hstack([partD_workload, partD_placement, partD_mitigation], widths="equal"),
            mo.Html(v1_02_reveal_card(
                "The survivor is the design with the most remaining envelope.",
                selected_label,
                partD_survivor_label,
                f"The selected design is {remaining_status}. Residual risk: {partD_residual_risk}.",
                tone=tone,
            )),
            mo.Html(v1_02_failure_card(partD_result)),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Placement Review</h2>
              <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                <thead>
                  <tr style="border-bottom:1px solid {COLORS['Border']}; color:{COLORS['TextMuted']}; text-align:left;">
                    <th>Placement</th><th>First wall</th><th style="text-align:right;">Worst headroom</th><th>Feasible</th><th>Residual risk</th>
                  </tr>
                </thead>
                <tbody>{''.join(placement_rows)}</tbody>
              </table>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Before And After Mitigation</h2>
              <div class="mlsysbook-callout"><strong>Selected mitigation:</strong> {html.escape(partD_selected_mitigation)}</div>
              <table style="width:100%; border-collapse:collapse; margin-top:12px; font-size:0.88rem;">
                <thead>
                  <tr style="border-bottom:1px solid {COLORS['Border']}; color:{COLORS['TextMuted']}; text-align:left;">
                    <th>Constraint</th><th>Value / limit</th><th>Before</th><th>After review</th>
                  </tr>
                </thead>
                <tbody>{''.join(before_after_rows)}</tbody>
              </table>
            </div>
            """),
            MathPeek(
                "memory_ok and latency_ok and energy_ok and power_ok and bandwidth_ok and cost_ok",
                {
                    "and": "A design ships only when every constraint passes at the same time.",
                    "mitigation": "A targeted change that attacks the binding wall.",
                    "residual risk": "The new operational risk introduced by placement.",
                },
            ),
            source_trace(
                {
                    "api": "evaluate_deployment_envelope() and deployment_mitigation()",
                    "stress_workload": f"{partD_workload.value:.1f} {v1_02_deployment.workload_unit}",
                    "recommended_mitigation": partD_mitigation_result.mitigation,
                    "selected_mitigation": partD_selected_mitigation,
                },
                summary="Part D source model",
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <div class="mlsysbook-grid">
                {v1_02_fields({
                    "partD_placement_id": partD_result.placement_id,
                    "partD_binding_constraint": partD_result.first_wall,
                    "partD_mitigation": partD_selected_mitigation,
                    "residual_risk": partD_residual_risk,
                })}
              </div>
            </div>
            """),
        ])
        return mo.vstack(items)

    def build_synthesis():
        completed = all(
            value is not None
            for value in (
                partA_prediction.value,
                partB_prediction.value,
                partC_prediction.value,
                partD_prediction.value,
            )
        )
        changed = (
            f"You first predicted {v1_02_option_label(partA_prediction.value, partA_wall_options)}; "
            f"the measured first wall was {partA_result.first_wall}."
            if partA_prediction.value is not None
            else "Complete Part A to compare your first prediction with measured evidence."
        )
        status = "complete" if completed else "in progress"
        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Synthesis</h2>
              <div class="mlsysbook-grid">
                {v1_02_fields({
                    "Lab status": status,
                    "Track": v1_02_deployment.label,
                    "First measured wall": partA_result.first_wall,
                    "Final placement": partD_result.placement_label,
                })}
              </div>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Three Takeaways</h2>
              <ul class="mlsysbook-list">
                <li><strong>Physics comes before preference.</strong> Feasibility is the maximum normalized constraint, not a model quality score.</li>
                <li><strong>The Iron Law names the useful optimization.</strong> Your compute multiplier produced {partB_actual_speedup:.2f}x speedup because {partB_active_term} remained in the budget.</li>
                <li><strong>Placement is a design inside an envelope.</strong> {partD_result.placement_label} has to address {partD_result.first_wall} while carrying the residual risk: {html.escape(partD_residual_risk)}.</li>
              </ul>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>What Changed From Your First Prediction?</h2>
              <p style="color:{COLORS['TextSec']}; line-height:1.6;">{html.escape(changed)}</p>
            </div>
            """),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Next Lab</h2>
              <p class="mlsysbook-action-note">
                Lab 03 moves the same constraint thinking into workflow design:
                once a physical wall is visible, the workflow must expose it before
                the release review.
              </p>
            </div>
            """),
        ])

    v1_02_tabs = mo.ui.tabs({
        "Part A - Physics Before Preference": build_part_a(),
        "Part B - Iron Law And Bottleneck": build_part_b(),
        "Part C - Operating Envelope": build_part_c(),
        "Part D - Placement Review": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    v1_02_tabs
    return (v1_02_tabs,)


# ===========================================================================
# ZONE E: LEDGER HUD AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    html,
    ledger,
    mo,
    partA_prediction,
    partA_result,
    partA_workload,
    partB_actual_speedup,
    partB_active_term,
    partB_prediction,
    partC_first_walls_by_track,
    partC_prediction,
    partC_tightest_track,
    partC_worst_headroom_by_track,
    partD_mitigation_result,
    partD_placement_id,
    partD_prediction,
    partD_residual_risk,
    partD_result,
    partD_selected_mitigation,
    v1_02_deployment,
    v1_02_profile,
    v1_02_variant,
):
    v1_02_predictions_complete = all(
        value is not None
        for value in (
            partA_prediction.value,
            partB_prediction.value,
            partC_prediction.value,
            partD_prediction.value,
        )
    )
    v1_02_ledger_design = {
        "track_id": v1_02_profile.track_id,
        "scenario_id": v1_02_variant.scenario_id,
        "partA_predicted_wall": partA_prediction.value,
        "partA_actual_wall": partA_result.first_wall,
        "partA_workload_value": partA_workload.value,
        "partA_placement_id": partA_result.placement_id,
        "partB_predicted_speedup": partB_prediction.value,
        "partB_actual_speedup": round(partB_actual_speedup, 4),
        "partB_active_term": partB_active_term,
        "partC_tightest_track": partC_tightest_track,
        "partC_first_walls_by_track": partC_first_walls_by_track,
        "partC_worst_headroom_by_track": partC_worst_headroom_by_track,
        "partD_placement_id": partD_placement_id,
        "partD_binding_constraint": partD_result.first_wall,
        "partD_mitigation": partD_selected_mitigation,
        "partD_recommended_mitigation": partD_mitigation_result.mitigation,
        "residual_risk": partD_residual_risk,
        "completed": v1_02_predictions_complete,
    }
    if v1_02_predictions_complete:
        ledger.save(track=v1_02_profile.track_id, chapter=2, design=v1_02_ledger_design)

    status = "SAVED" if v1_02_predictions_complete else "IN PROGRESS"
    status_color = COLORS["GreenLine"] if v1_02_predictions_complete else COLORS["OrangeLine"]
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">02 - Physics of Deployment</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{html.escape(v1_02_profile.label)}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">ARTIFACT</span>
        <span class="hud-value">{html.escape(v1_02_deployment.report_artifact)}</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active" style="color:{status_color};">{status}</span>
    </div>
    """)
    return (v1_02_ledger_design, v1_02_predictions_complete)


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    partA_prediction,
    partA_result,
    partB_actual_speedup,
    partB_active_term,
    partB_prediction,
    partC_first_walls_by_track,
    partC_prediction,
    partC_tightest_track,
    partC_worst_headroom_by_track,
    partD_prediction,
    partD_residual_risk,
    partD_result,
    partD_selected_mitigation,
    report_export_panel,
    v1_02_deployment,
    v1_02_ledger_design,
    v1_02_metadata,
    v1_02_predictions_complete,
    v1_02_profile,
    v1_02_variant,
):
    incomplete = []
    if partA_prediction.value is None:
        incomplete.append("Part A prediction")
    if partB_prediction.value is None:
        incomplete.append("Part B prediction")
    if partC_prediction.value is None:
        incomplete.append("Part C prediction")
    if partD_prediction.value is None:
        incomplete.append("Part D prediction")

    final_decision = (
        f"For {v1_02_deployment.label}, use {partD_result.placement_label} with "
        f"{partD_selected_mitigation}. Binding constraint: {partD_result.first_wall}."
    )
    report = build_lab_report(
        v1_02_metadata,
        track=v1_02_profile.label,
        scenario=v1_02_variant.workload_summary,
        learning_objectives=(
            "Identify which physical quantity binds first for a deployment context.",
            "Diagnose the active Iron Law latency term before choosing an optimization.",
            "Defend a placement and mitigation using measured constraint evidence.",
        ),
        predictions={
            "partA_first_wall": partA_prediction.value,
            "partB_speedup": partB_prediction.value,
            "partC_tightest_track": partC_prediction.value,
            "partD_surviving_placement": partD_prediction.value,
        },
        knob_settings={
            "track_id": v1_02_profile.track_id,
            "partA_workload_value": v1_02_ledger_design["partA_workload_value"],
            "partD_placement_id": v1_02_ledger_design["partD_placement_id"],
            "partD_mitigation": v1_02_ledger_design["partD_mitigation"],
        },
        evidence_summary={
            "partA_actual_wall": partA_result.first_wall,
            "partB_actual_speedup": round(partB_actual_speedup, 3),
            "partB_active_term": partB_active_term,
            "partC_tightest_track": partC_tightest_track,
            "partC_first_walls_by_track": partC_first_walls_by_track,
            "partC_worst_headroom_by_track": partC_worst_headroom_by_track,
            "partD_binding_constraint": partD_result.first_wall,
            "partD_feasible": partD_result.feasible,
        },
        final_decision=final_decision if v1_02_predictions_complete else "",
        big_takeaways=(
            "Deployment is constrained by physical amounts before preference.",
            "The useful optimization is the one that attacks the active latency term.",
            "Placement mitigates one wall while introducing residual operational risk.",
        ),
        reflections={
            "residual_risk": partD_residual_risk,
            "report_artifact": v1_02_deployment.report_artifact,
        },
        residual_risk=partD_residual_risk,
        source_trace={
            "track_id": v1_02_profile.track_id,
            "scenario_id": v1_02_variant.scenario_id,
            "hardware_ref": v1_02_variant.hardware_ref,
            "model_ref": v1_02_variant.model_ref,
            "shared_helper": "mlsysbook_labs.deployment",
            "source_policy": v1_02_profile.source_policy,
        },
        result_snapshot=v1_02_ledger_design,
        incomplete_fields=tuple(incomplete),
    )
    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This memo is generated locally from your locked predictions, "
                "track profile, and computed deployment evidence."
            ),
            kind="info",
        ),
        report_export_panel(report),
    ])
    return


if __name__ == "__main__":
    app.run()
