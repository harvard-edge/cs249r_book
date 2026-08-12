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
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        memory_cliff,
        neural_compute_profile,
        operation_ledger,
        operator_design,
        part_workflow,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
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
        apply_plotly_theme,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        memory_cliff,
        mo,
        neural_compute_profile,
        operation_ledger,
        operator_design,
        part_workflow,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_05_metadata = get_lab_metadata("vol1/lab_05_nn_compute.py")
    return (v1_05_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_05_track_picker = track_selector(default=_default_track)
    v1_05_track_picker
    return (v1_05_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    neural_compute_profile,
    resolve_mlsysim_ref,
    v1_05_track_picker,
):
    v1_05_track_id = v1_05_track_picker.value
    v1_05_profile = get_track_profile(v1_05_track_id)
    v1_05_variant = get_lab_track_variant("v1_05_neural_computation", v1_05_profile.track_id)
    v1_05_hardware = resolve_mlsysim_ref(v1_05_variant.hardware_ref)
    v1_05_model = resolve_mlsysim_ref(v1_05_variant.model_ref)
    v1_05_compute = neural_compute_profile(
        v1_05_profile,
        v1_05_variant,
        v1_05_hardware,
        v1_05_model,
    )
    return (
        v1_05_compute,
        v1_05_hardware,
        v1_05_model,
        v1_05_profile,
        v1_05_track_id,
        v1_05_variant,
    )


@app.cell
def _():
    def v1_05_escape(value):
        import html as _html

        return _html.escape(str(value))

    def v1_05_fields_html(fields):
        return "\n".join(
            (
                '<div class="mlsysbook-field">'
                f"<strong>{v1_05_escape(key)}</strong>{v1_05_escape(value)}"
                "</div>"
            )
            for key, value in fields.items()
        )

    def v1_05_table_html(headers, rows, numeric=()):
        header_html = "".join(f"<th>{v1_05_escape(header)}</th>" for header in headers)
        body_rows = []
        numeric_cols = set(numeric)
        for row in rows:
            cells = []
            for idx, value in enumerate(row):
                align = "right" if idx in numeric_cols else "left"
                cells.append(f'<td style="text-align:{align};">{v1_05_escape(value)}</td>')
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        return f"""
        <div style="overflow-x:auto; margin-top:14px;">
          <table class="mlsysbook-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        </div>
        """

    def v1_05_callout_html(title, body, kind="info"):
        palette = {
            "ok": ("#dcfce7", "#166534"),
            "warn": ("#fef3c7", "#92400e"),
            "fail": ("#fee2e2", "#991b1b"),
            "info": ("#e0f2fe", "#075985"),
        }
        background, border = palette.get(kind, palette["info"])
        return f"""
        <div class="mlsysbook-callout" style="background:{background}; border-color:{border};">
          <strong>{v1_05_escape(title)}:</strong> {v1_05_escape(body)}
        </div>
        """

    def v1_05_prediction_html(title, predicted, actual, labels):
        if predicted is None:
            return v1_05_callout_html(
                title,
                "Commit to a structured prediction before using this evidence.",
                "warn",
            )
        predicted_label = labels.get(predicted, predicted)
        actual_label = labels.get(actual, actual)
        if predicted == actual:
            return v1_05_callout_html(
                title,
                f"Prediction matched: {actual_label}.",
                "ok",
            )
        return v1_05_callout_html(
            title,
            f"You predicted {predicted_label}; measured evidence points to {actual_label}.",
            "warn",
        )

    def v1_05_track_story(track_id):
        stories = {
            "iphone": {
                "stakeholder": "mobile product engineer",
                "amount_focus": "local inference latency, activation memory, and power/thermal headroom",
                "failure": "the feature feels slow or heats the phone during repeated local inference",
                "batch_meaning": "batching usually means queued user frames, so latency and thermal pressure matter more than raw throughput",
                "energy_label": "mJ/inference",
                "checkpoint_hint": "protect responsiveness first, then verify the NPU path stays covered",
                "next_lab": "V1-06 should favor mobile architectures that reduce activation traffic before adding width.",
            },
            "oura_ring": {
                "stakeholder": "wearable firmware engineer",
                "amount_focus": "SRAM, flash, wake time, and uJ/window",
                "failure": "the sensing window overflows SRAM or keeps the MCU awake too long",
                "batch_meaning": "batching means holding more windows before sleep, so wake time and SRAM dominate",
                "energy_label": "uJ/window",
                "checkpoint_hint": "stream or summarize before asking for more temporal context",
                "next_lab": "V1-06 should prefer streaming TinyML architectures with bounded buffers.",
            },
            "robotaxi": {
                "stakeholder": "autonomous vehicle platform engineer",
                "amount_focus": "real-time frame deadline, sensor pipeline headroom, and p99 bandwidth",
                "failure": "a bursty sensor frame misses the perception deadline",
                "batch_meaning": "batching means aggregating sensor streams, so tail latency and safety margin dominate",
                "energy_label": "mJ/frame",
                "checkpoint_hint": "bound the worst frame first, then preserve recall at object boundaries",
                "next_lab": "V1-06 should choose architectures with predictable feature-map sizes across sensors.",
            },
            "cloud_fleet": {
                "stakeholder": "fleet service owner",
                "amount_focus": "throughput, accelerator memory, utilization, cost/request, and p99 latency",
                "failure": "a larger batch improves average throughput but raises p99 latency or accelerator memory pressure",
                "batch_meaning": "batching is an economic utilization knob, but activation state still consumes HBM",
                "energy_label": "mJ/request",
                "checkpoint_hint": "increase utilization only while memory and p99 latency remain inside the SLA",
                "next_lab": "V1-06 should connect architecture choices to accelerator occupancy and memory reuse.",
            },
        }
        return stories.get(track_id, stories["iphone"])

    def v1_05_budget_ratios(result, profile):
        return {
            "activation memory": result.activations_mb / max(profile.activation_budget_mb, 1e-9),
            "bandwidth": result.estimated_bandwidth_gbs / max(profile.bandwidth_budget_gbs, 1e-9),
            "latency": result.estimated_latency_ms / max(profile.latency_budget_ms, 1e-9),
            "power": result.estimated_power_w / max(profile.power_budget_w, 1e-9),
        }

    def v1_05_binding_from_ratios(ratios):
        name = max(ratios, key=ratios.get)
        return {"name": name, "ratio": ratios[name]}

    def v1_05_cliff_category(profile, cliff):
        if cliff.threshold_multiplier is None:
            return "no_cliff"
        if cliff.threshold_multiplier < profile.default_shape_multiplier:
            return "before_default"
        if cliff.threshold_multiplier <= profile.default_shape_multiplier * 1.25:
            return "near_default"
        return "later"

    def v1_05_batch_factor(profile, policy):
        if policy == "low_latency":
            return 0.5 if profile.batch > 1 else 1.0
        if policy == "throughput_x2":
            return 2.0
        if policy == "throughput_x4":
            return 4.0
        return 1.0

    def v1_05_precision_bytes(profile, policy):
        if policy == "compact_int8":
            return 1
        if policy == "wide_fp16":
            return 2
        return profile.precision_bytes

    def v1_05_scaled_profile(profile, batch_policy, precision_policy):
        from dataclasses import replace

        batch_factor = v1_05_batch_factor(profile, batch_policy)
        precision_bytes = v1_05_precision_bytes(profile, precision_policy)
        scaled_batch = max(1, int(round(profile.batch * batch_factor)))
        return replace(profile, batch=scaled_batch, precision_bytes=precision_bytes)

    def v1_05_energy_display(track_id, energy_j):
        if track_id == "oura_ring":
            return f"{energy_j * 1_000_000:.1f} uJ/window"
        if track_id == "cloud_fleet":
            return f"{energy_j * 1_000:.2f} mJ/request"
        if track_id == "robotaxi":
            return f"{energy_j * 1_000:.2f} mJ/frame"
        return f"{energy_j * 1_000:.2f} mJ/inference"

    def v1_05_batch_precision_rows(profile, shape_multiplier, selected_batch, selected_precision, operation_ledger):
        specs = [
            ("low_latency", "track_precision"),
            ("default", "track_precision"),
            ("throughput_x2", "compact_int8"),
            ("throughput_x4", "compact_int8"),
            ("throughput_x2", "wide_fp16"),
        ]
        selected_spec = (selected_batch, selected_precision)
        if selected_spec not in specs:
            specs.append(selected_spec)
        labels = {
            "low_latency": "Low-latency batch/window",
            "default": "Track default batch/window",
            "throughput_x2": "Throughput x2 batch/window",
            "throughput_x4": "Aggressive x4 batch/window",
            "track_precision": "Track precision",
            "compact_int8": "Compact INT8",
            "wide_fp16": "Wider FP16",
        }
        rows = []
        for batch_policy, precision_policy in specs:
            scaled_profile = v1_05_scaled_profile(profile, batch_policy, precision_policy)
            result = operation_ledger(scaled_profile, shape_multiplier=shape_multiplier)
            ratios = v1_05_budget_ratios(result, scaled_profile)
            binding = v1_05_binding_from_ratios(ratios)
            throughput_per_s = scaled_profile.batch / max(result.estimated_latency_ms, 1e-9) * 1000.0
            energy_j = result.estimated_power_w * result.estimated_latency_ms / 1000.0
            rows.append(
                {
                    "batch_policy": batch_policy,
                    "precision_policy": precision_policy,
                    "label": f"{labels[batch_policy]} + {labels[precision_policy]}",
                    "batch": scaled_profile.batch,
                    "precision_bytes": scaled_profile.precision_bytes,
                    "activation_mb": result.activations_mb,
                    "latency_ms": result.estimated_latency_ms,
                    "bandwidth_gbs": result.estimated_bandwidth_gbs,
                    "power_w": result.estimated_power_w,
                    "throughput_per_s": throughput_per_s,
                    "energy_j": energy_j,
                    "binding": binding["name"],
                    "binding_ratio": binding["ratio"],
                    "feasible": result.feasible,
                    "selected": (batch_policy, precision_policy) == selected_spec,
                    "violations": result.violations,
                }
            )
        return tuple(rows)

    def v1_05_strategy_category(profile, selected_row):
        if not selected_row["feasible"]:
            return "not_safe"
        if selected_row["precision_bytes"] < profile.precision_bytes:
            return "compact_precision"
        if selected_row["batch"] > profile.batch:
            return "larger_batch"
        return "default"

    def v1_05_roofline_diagnosis(profile, result):
        crossover = profile.peak_tflops * 1000.0 / max(profile.memory_bandwidth_gbs, 1e-9)
        roofline_wall = "memory" if result.arithmetic_intensity < crossover else "compute"
        if result.dominant_resource in {"activation memory", "bandwidth"}:
            diagnosed_wall = "memory"
            recommendation = "reduce activation bytes, tile/stream tensors, or improve reuse"
        elif result.dominant_resource == "latency":
            diagnosed_wall = "compute"
            recommendation = "reduce operation count or use a faster fused kernel"
        elif result.dominant_resource == "power":
            diagnosed_wall = "energy"
            recommendation = "reduce precision, wake time, or sustained data movement"
        else:
            diagnosed_wall = roofline_wall
            recommendation = "match the optimization to the measured binding amount"
        return {
            "crossover": crossover,
            "roofline_wall": roofline_wall,
            "diagnosed_wall": diagnosed_wall,
            "recommendation": recommendation,
        }

    def v1_05_design_alignment(diagnosis, selected_result, all_results):
        highest_activation = max(all_results, key=lambda result: result.activation_mb)
        slowest_latency = max(all_results, key=lambda result: result.latency_ms)
        activation_reduction = selected_result.activation_mb < highest_activation.activation_mb
        latency_reduction = selected_result.latency_ms < slowest_latency.latency_ms
        if diagnosis["diagnosed_wall"] == "memory":
            aligned = activation_reduction
            target = "activation or byte movement reduction"
        elif diagnosis["diagnosed_wall"] == "compute":
            aligned = latency_reduction
            target = "operation or latency reduction"
        else:
            aligned = activation_reduction or latency_reduction
            target = "precision, power, or wake-time reduction"
        return {
            "aligned": aligned,
            "target": target,
            "message": (
                f"Selected design targets {target}."
                if aligned
                else f"Selected design may not attack the diagnosed wall: {target}."
            ),
        }

    return (
        v1_05_batch_precision_rows,
        v1_05_binding_from_ratios,
        v1_05_budget_ratios,
        v1_05_callout_html,
        v1_05_cliff_category,
        v1_05_design_alignment,
        v1_05_energy_display,
        v1_05_fields_html,
        v1_05_prediction_html,
        v1_05_roofline_diagnosis,
        v1_05_strategy_category,
        v1_05_table_html,
        v1_05_track_story,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_arc_context,
    track_context,
    v1_05_compute,
    v1_05_metadata,
    v1_05_profile,
    v1_05_track_story,
    v1_05_variant,
):
    _story = v1_05_track_story(v1_05_profile.track_id)
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #94a3b8; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 05
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Neural Computation Amounts
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Operations &middot; Activations &middot; Memory Traffic &middot; Energy
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 860px; line-height: 1.65;">
                {v1_05_variant.workload_summary} Neural computation is a budget:
                tensor shapes become bounded amounts of operations, activations,
                memory traffic, and energy.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Concept Modules + Synthesis
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_05_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_05_compute.tensor_label}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Shape Growth</span>
                <span class="badge badge-warn">Activation Cliff</span>
                <span class="badge badge-info">Batch/Precision</span>
                <span class="badge badge-fail">Compute-vs-Memory Diagnosis</span>
            </div>
        </div>
        """),
        track_context(v1_05_profile),
        track_arc_context(v1_05_profile, v1_05_metadata.lab_id),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Track Amount Lens</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Stakeholder</strong>{_story["stakeholder"]}</div>
            <div class="mlsysbook-field"><strong>Bounded amounts</strong>{_story["amount_focus"]}</div>
            <div class="mlsysbook-field"><strong>Natural failure</strong>{_story["failure"]}</div>
            <div class="mlsysbook-field"><strong>Operator tensor</strong>{v1_05_compute.tensor_label}</div>
          </div>
          <div class="mlsysbook-callout"><strong>Shared concept sequence:</strong>
            Parts A-D are the same for every track. The selected track changes
            persona, constraints, thresholds, evidence emphasis, failure mode,
            and report framing.</div>
        </div>
        """),
        source_trace(
            {
                "chapter_invariant": "neural computation turns tensors into bounded operation, activation, memory-traffic, and energy amounts",
                "chapter_anchors": "Purpose; Forward pass computation; Memory wall; Batch processing; Arithmetic intensity",
                "hardware_ref": v1_05_variant.hardware_ref,
                "model_ref": v1_05_variant.model_ref,
                "shared_helper": "neural_compute_profile()",
            },
            summary="Opening source map",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, part_workflow, v1_05_compute, v1_05_profile, v1_05_track_story):
    _story = v1_05_track_story(v1_05_profile.track_id)
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
                <div style="margin-bottom: 3px;">1. <strong>Trace tensor shape growth:</strong>
                    connect a changed shape to activation, operation, and byte amounts.</div>
                <div style="margin-bottom: 3px;">2. <strong>Find the binding budget:</strong>
                    identify when activations, bandwidth, latency, or power bind first.</div>
                <div style="margin-bottom: 3px;">3. <strong>Compare batch and precision:</strong>
                    trade throughput against memory, latency, and energy in your track.</div>
                <div style="margin-bottom: 3px;">4. <strong>Diagnose the optimization:</strong>
                    decide whether compute or memory/data movement is the wall.</div>
            </div>
            <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                        padding: 16px 28px 0 28px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Core Question
                </div>
                <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                            line-height: 1.5; font-style: italic;">
                    Which bounded amount controls {v1_05_compute.label}, and which operator
                    choice carries the right architecture implication forward?
                </div>
            </div>
        </div>
        """),
        part_workflow(
            "Neural Computation Amount Workflow",
            (
                {
                    "part": "Part A",
                    "concept": "Tensor shape growth changes activation and operation amounts.",
                    "prediction": "Predict which amount binds first.",
                    "controls": "Adjust the shape multiplier for the active operator.",
                    "evidence": "Compare activation memory, GMACs, bytes moved, intensity, latency, and power.",
                    "decision": "Pick the amount to reduce first.",
                },
                {
                    "part": "Part B",
                    "concept": "Memory and activations can bind before arithmetic does.",
                    "prediction": "Predict where the activation cliff appears.",
                    "controls": "Sweep the shape variable and inspect normalized budgets.",
                    "evidence": "Find the first threshold crossing and exact violations.",
                    "decision": "Choose the largest defensible shape policy.",
                },
                {
                    "part": "Part C",
                    "concept": "Batch and precision trade throughput, memory, latency, and energy.",
                    "prediction": "Predict which batch/precision strategy survives.",
                    "controls": "Change batch/window and precision policies.",
                    "evidence": "Compare feasible and infeasible strategies in a table and scatter plot.",
                    "decision": "Record the track-specific batch/precision policy.",
                },
                {
                    "part": "Part D",
                    "concept": "Compute-vs-memory diagnosis determines the right optimization.",
                    "prediction": "Predict the wall before selecting an operator design.",
                    "controls": "Choose an operator design option.",
                    "evidence": "Compare arithmetic intensity with the hardware crossover and design candidates.",
                    "decision": "Name the optimization family and residual risk.",
                },
            ),
            scenario=(
                f"{v1_05_compute.label} is constrained by {_story['amount_focus']}; "
                f"batch/window choices mean {_story['batch_meaning']}."
            ),
            reflection="The synthesis saves an operator budget note: binding amount, selected design, residual risk, and V1-06 architecture implication.",
        ),
    ])
    return


# ===========================================================================
# ZONE B: CONTROLS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_05_compute):
    v1_05_resource_prediction = mo.ui.radio(
        options={
            "Activation memory binds first": "activation memory",
            "Memory traffic / bandwidth binds first": "bandwidth",
            "Arithmetic latency binds first": "latency",
            "Energy or power binds first": "power",
        },
        label=f"Part A prediction: which bounded amount will dominate {v1_05_compute.label}?",
    )
    v1_05_shape_multiplier = mo.ui.slider(
        start=v1_05_compute.shape_min,
        stop=v1_05_compute.shape_max,
        value=v1_05_compute.default_shape_multiplier,
        step=v1_05_compute.shape_step,
        label="Shape multiplier",
    )
    v1_05_amount_checkpoint = mo.ui.radio(
        options={
            "Reduce activation tensor size first": "reduce_activations",
            "Reduce memory traffic first": "reduce_bytes",
            "Reduce operation count first": "reduce_ops",
            "Reduce sustained power first": "reduce_power",
        },
        label="Part A checkpoint: which amount should the operator budget reduce first?",
    )
    return (v1_05_amount_checkpoint, v1_05_resource_prediction, v1_05_shape_multiplier)


@app.cell(hide_code=True)
def _(mo):
    v1_05_cliff_prediction = mo.ui.radio(
        options={
            "The cliff is already before the default shape": "before_default",
            "The cliff is near the default shape": "near_default",
            "The cliff appears only after more shape growth": "later",
            "No activation cliff appears in the tested envelope": "no_cliff",
        },
        label="Part B prediction: where will the activation budget fail?",
    )
    v1_05_memory_checkpoint = mo.ui.radio(
        options={
            "Ship only below the measured cliff": "below_cliff",
            "Tile or stream the operator before growing shape": "tile_stream",
            "Hold shape and reduce precision first": "reduce_precision",
            "Accept the overage and document the violation": "accept_violation",
        },
        label="Part B checkpoint: what shape policy goes into the budget note?",
    )
    return (v1_05_cliff_prediction, v1_05_memory_checkpoint)


@app.cell(hide_code=True)
def _(mo):
    v1_05_batch_prediction = mo.ui.radio(
        options={
            "Track default is the only defensible policy": "default",
            "Compact precision makes the selected policy safe": "compact_precision",
            "A larger batch/window is safe": "larger_batch",
            "The selected batch/precision policy will fail": "not_safe",
        },
        label="Part C prediction: which batch/precision outcome do you expect?",
    )
    v1_05_batch_policy = mo.ui.dropdown(
        options={
            "Low-latency batch/window": "low_latency",
            "Track default batch/window": "default",
            "Throughput x2 batch/window": "throughput_x2",
            "Aggressive x4 batch/window": "throughput_x4",
        },
        value="Track default batch/window",
        label="Batch/window policy",
    )
    v1_05_precision_policy = mo.ui.dropdown(
        options={
            "Track precision": "track_precision",
            "Compact INT8": "compact_int8",
            "Wider FP16": "wide_fp16",
        },
        value="Track precision",
        label="Precision policy",
    )
    v1_05_batch_checkpoint = mo.ui.radio(
        options={
            "Use default batch/window and protect latency": "default_latency",
            "Use compact precision with validation": "compact_validate",
            "Increase batch/window only with p99 guardrail": "batch_with_guardrail",
            "Reject the selected policy and redesign": "reject_redesign",
        },
        label="Part C checkpoint: what batch/precision policy do you record?",
    )
    return (
        v1_05_batch_checkpoint,
        v1_05_batch_policy,
        v1_05_batch_prediction,
        v1_05_precision_policy,
    )


@app.cell(hide_code=True)
def _(mo, v1_05_compute):
    _design_options = {design.label: design.design_id for design in v1_05_compute.design_options}
    v1_05_diagnosis_prediction = mo.ui.radio(
        options={
            "Memory or data movement is the wall": "memory",
            "Compute latency is the wall": "compute",
            "Energy or thermal power is the wall": "energy",
        },
        label="Part D prediction: what wall should the optimization target?",
    )
    v1_05_design = mo.ui.dropdown(
        options=_design_options,
        value=v1_05_compute.design_options[0].label,
        label="Operator design",
    )
    v1_05_optimization_checkpoint = mo.ui.radio(
        options={
            "Reduce bytes moved and activation residency": "memory_optimization",
            "Reduce operation count or use a fused compute kernel": "compute_optimization",
            "Reduce precision, wake time, or sustained power": "energy_optimization",
            "Keep the baseline and accept the residual risk": "accept_baseline",
        },
        label="Part D checkpoint: which optimization family goes into the memo?",
    )
    return (v1_05_design, v1_05_diagnosis_prediction, v1_05_optimization_checkpoint)


@app.cell(hide_code=True)
def _(mo):
    v1_05_final_decision = mo.ui.radio(
        options={
            "Ship selected operator with measured binding amount": "ship_with_binding",
            "Ship only after reducing the binding amount": "reduce_before_ship",
            "Redesign architecture before this operator can ship": "redesign_architecture",
        },
        label="Synthesis decision: what is the operator budget decision?",
    )
    v1_05_budget_note = mo.ui.text_area(
        label="Operator budget note",
        placeholder=(
            "Name the binding amount, the selected operator design, the evidence number, "
            "the residual risk, and the architecture implication for V1-06."
        ),
        full_width=True,
    )
    return (v1_05_budget_note, v1_05_final_decision)


@app.cell
def _(
    memory_cliff,
    operation_ledger,
    operator_design,
    v1_05_batch_policy,
    v1_05_batch_precision_rows,
    v1_05_binding_from_ratios,
    v1_05_budget_ratios,
    v1_05_cliff_category,
    v1_05_compute,
    v1_05_design,
    v1_05_design_alignment,
    v1_05_precision_policy,
    v1_05_roofline_diagnosis,
    v1_05_shape_multiplier,
    v1_05_strategy_category,
    v1_05_track_story,
):
    v1_05_ledger = operation_ledger(
        v1_05_compute,
        shape_multiplier=v1_05_shape_multiplier.value,
    )
    v1_05_cliff = memory_cliff(v1_05_compute, samples=44)
    v1_05_shape_ledgers = tuple(
        operation_ledger(v1_05_compute, shape_multiplier=value)
        for value in v1_05_cliff.shape_values
    )
    v1_05_design_result = operator_design(
        v1_05_compute,
        design_id=v1_05_design.value,
        shape_multiplier=v1_05_shape_multiplier.value,
    )
    v1_05_design_results = tuple(
        operator_design(
            v1_05_compute,
            design_id=design.design_id,
            shape_multiplier=v1_05_shape_multiplier.value,
        )
        for design in v1_05_compute.design_options
    )
    v1_05_ratios = v1_05_budget_ratios(v1_05_ledger, v1_05_compute)
    v1_05_binding_ratio = v1_05_binding_from_ratios(v1_05_ratios)
    v1_05_cliff_actual = v1_05_cliff_category(v1_05_compute, v1_05_cliff)
    v1_05_batch_rows = v1_05_batch_precision_rows(
        v1_05_compute,
        v1_05_shape_multiplier.value,
        v1_05_batch_policy.value,
        v1_05_precision_policy.value,
        operation_ledger,
    )
    v1_05_selected_batch_row = next(row for row in v1_05_batch_rows if row["selected"])
    v1_05_batch_actual = v1_05_strategy_category(v1_05_compute, v1_05_selected_batch_row)
    v1_05_diagnosis = v1_05_roofline_diagnosis(v1_05_compute, v1_05_ledger)
    v1_05_alignment = v1_05_design_alignment(
        v1_05_diagnosis,
        v1_05_design_result,
        v1_05_design_results,
    )
    v1_05_story = v1_05_track_story(v1_05_compute.track_id)
    return (
        v1_05_alignment,
        v1_05_batch_actual,
        v1_05_batch_rows,
        v1_05_binding_ratio,
        v1_05_cliff,
        v1_05_cliff_actual,
        v1_05_design_result,
        v1_05_design_results,
        v1_05_diagnosis,
        v1_05_ledger,
        v1_05_ratios,
        v1_05_selected_batch_row,
        v1_05_shape_ledgers,
        v1_05_story,
    )


# ===========================================================================
# ZONE C: CONCEPT MODULES
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    source_trace,
    v1_05_alignment,
    v1_05_amount_checkpoint,
    v1_05_batch_actual,
    v1_05_batch_checkpoint,
    v1_05_batch_policy,
    v1_05_batch_prediction,
    v1_05_batch_rows,
    v1_05_binding_ratio,
    v1_05_budget_note,
    v1_05_callout_html,
    v1_05_cliff,
    v1_05_cliff_actual,
    v1_05_cliff_prediction,
    v1_05_compute,
    v1_05_design,
    v1_05_design_result,
    v1_05_design_results,
    v1_05_diagnosis,
    v1_05_diagnosis_prediction,
    v1_05_energy_display,
    v1_05_fields_html,
    v1_05_final_decision,
    v1_05_ledger,
    v1_05_memory_checkpoint,
    v1_05_optimization_checkpoint,
    v1_05_precision_policy,
    v1_05_prediction_html,
    v1_05_ratios,
    v1_05_resource_prediction,
    v1_05_selected_batch_row,
    v1_05_shape_ledgers,
    v1_05_shape_multiplier,
    v1_05_story,
    v1_05_table_html,
    v1_05_variant,
):
    _resource_labels = {
        "activation memory": "activation memory",
        "bandwidth": "memory traffic / bandwidth",
        "latency": "arithmetic latency",
        "power": "energy or power",
    }
    _cliff_labels = {
        "before_default": "before the default shape",
        "near_default": "near the default shape",
        "later": "after more shape growth",
        "no_cliff": "no cliff inside the tested envelope",
    }
    _batch_labels = {
        "default": "track default",
        "compact_precision": "compact precision",
        "larger_batch": "larger batch/window",
        "not_safe": "selected policy fails",
    }
    _diagnosis_labels = {
        "memory": "memory or data movement",
        "compute": "compute latency",
        "energy": "energy or thermal power",
    }

    _growth_fig = go.Figure()
    _growth_fig.add_trace(go.Scatter(
        x=[result.shape_multiplier for result in v1_05_shape_ledgers],
        y=[result.activations_mb for result in v1_05_shape_ledgers],
        mode="lines",
        name="Activation memory (MB)",
        line=dict(color=COLORS["BlueLine"], width=3),
    ))
    _growth_fig.add_trace(go.Scatter(
        x=[result.shape_multiplier for result in v1_05_shape_ledgers],
        y=[result.ops_gmac for result in v1_05_shape_ledgers],
        mode="lines",
        name="Operations (GMAC)",
        yaxis="y2",
        line=dict(color=COLORS["OrangeLine"], width=3),
    ))
    _growth_fig.add_vline(
        x=v1_05_shape_multiplier.value,
        line_dash="dash",
        line_color=COLORS["TextMuted"],
        annotation_text="current",
    )
    _growth_fig.update_layout(
        height=360,
        xaxis=dict(title="Shape multiplier", gridcolor="#f1f5f9"),
        yaxis=dict(title="Activation memory (MB)", gridcolor="#f1f5f9"),
        yaxis2=dict(title="Operations (GMAC)", overlaying="y", side="right", showgrid=False),
        margin=dict(l=60, r=60, t=35, b=55),
        legend=dict(orientation="h", y=-0.22),
    )
    apply_plotly_theme(_growth_fig)

    _cliff_colors = [COLORS["GreenLine"] if ok else COLORS["RedLine"] for ok in v1_05_cliff.feasible]
    _cliff_fig = go.Figure()
    _cliff_fig.add_trace(go.Scatter(
        x=list(v1_05_cliff.shape_values),
        y=list(v1_05_cliff.activation_mb),
        mode="lines+markers",
        marker=dict(color=_cliff_colors, size=7),
        line=dict(color=COLORS["BlueLine"], width=2.5),
        name="Activation memory",
    ))
    _cliff_fig.add_hline(
        y=v1_05_compute.activation_budget_mb,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=1.5,
        annotation_text="activation budget",
    )
    if v1_05_cliff.threshold_multiplier is not None:
        _cliff_fig.add_vline(
            x=v1_05_cliff.threshold_multiplier,
            line_dash="dash",
            line_color=COLORS["RedLine"],
            line_width=1.5,
            annotation_text="first cliff",
        )
    _cliff_fig.update_layout(
        height=360,
        xaxis=dict(title="Shape multiplier", gridcolor="#f1f5f9"),
        yaxis=dict(title="Activation memory (MB)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=55),
    )
    apply_plotly_theme(_cliff_fig)

    _budget_fig = go.Figure()
    _budget_fig.add_trace(go.Bar(
        x=list(v1_05_ratios.values()),
        y=list(v1_05_ratios.keys()),
        orientation="h",
        marker_color=[
            COLORS["RedLine"] if ratio > 1.0 else COLORS["GreenLine"]
            for ratio in v1_05_ratios.values()
        ],
        text=[f"{ratio:.2f}x" for ratio in v1_05_ratios.values()],
        textposition="outside",
    ))
    _budget_fig.add_vline(x=1.0, line_dash="dash", line_color=COLORS["RedLine"])
    _budget_fig.update_layout(
        height=300,
        xaxis=dict(title="Used / budget", gridcolor="#f1f5f9"),
        yaxis=dict(title="Budget amount", gridcolor="#f1f5f9"),
        margin=dict(l=110, r=40, t=25, b=45),
        showlegend=False,
    )
    apply_plotly_theme(_budget_fig)

    _batch_fig = go.Figure()
    for _row in v1_05_batch_rows:
        _color = COLORS["GreenLine"] if _row["feasible"] else COLORS["RedLine"]
        _symbol = "diamond" if _row["selected"] else "circle"
        _batch_fig.add_trace(go.Scatter(
            x=[_row["latency_ms"]],
            y=[_row["activation_mb"]],
            mode="markers+text",
            text=["selected" if _row["selected"] else ""],
            textposition="top center",
            marker=dict(
                color=_color,
                size=max(10, min(28, _row["throughput_per_s"] / 50)),
                symbol=_symbol,
                line=dict(color="#0f172a", width=1),
            ),
            name=_row["label"],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Latency %{x:.3f} ms<br>"
                "Activation %{y:.2f} MB<extra></extra>"
            ),
        ))
    _batch_fig.add_vline(x=v1_05_compute.latency_budget_ms, line_dash="dash", line_color=COLORS["RedLine"])
    _batch_fig.add_hline(y=v1_05_compute.activation_budget_mb, line_dash="dash", line_color=COLORS["RedLine"])
    _batch_fig.update_layout(
        height=380,
        xaxis=dict(title="Latency estimate (ms)", gridcolor="#f1f5f9"),
        yaxis=dict(title="Activation memory (MB)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=60),
        legend=dict(orientation="h", y=-0.25),
    )
    apply_plotly_theme(_batch_fig)

    _roof_fig = go.Figure()
    _roof_fig.add_trace(go.Bar(
        y=["Current intensity", "Hardware crossover"],
        x=[v1_05_ledger.arithmetic_intensity, v1_05_diagnosis["crossover"]],
        orientation="h",
        marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]],
        text=[
            f"{v1_05_ledger.arithmetic_intensity:.2f} ops/byte",
            f"{v1_05_diagnosis['crossover']:.2f} ops/byte",
        ],
        textposition="outside",
    ))
    _roof_fig.update_layout(
        height=260,
        xaxis=dict(title="Arithmetic intensity (ops/byte)", gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="#f1f5f9"),
        margin=dict(l=140, r=40, t=25, b=45),
        showlegend=False,
    )
    apply_plotly_theme(_roof_fig)

    _ledger_rows = (
        ("Weights", f"{v1_05_ledger.weights_mb:.3f} MB", "persistent model bytes"),
        ("Activations", f"{v1_05_ledger.activations_mb:.3f} MB", f"limit {v1_05_compute.activation_budget_mb:.3f} MB"),
        ("Operations", f"{v1_05_ledger.ops_gmac:.3f} GMAC", "shape-dependent arithmetic"),
        ("Bytes moved", f"{v1_05_ledger.bytes_moved_mb:.3f} MB", "memory traffic estimate"),
        ("Arithmetic intensity", f"{v1_05_ledger.arithmetic_intensity:.2f} ops/byte", "reuse diagnostic"),
        ("Estimated latency", f"{v1_05_ledger.estimated_latency_ms:.3f} ms", f"limit {v1_05_compute.latency_budget_ms:.1f} ms"),
        ("Estimated bandwidth", f"{v1_05_ledger.estimated_bandwidth_gbs:.2f} GB/s", f"limit {v1_05_compute.bandwidth_budget_gbs:.2f} GB/s"),
        ("Estimated power", f"{v1_05_ledger.estimated_power_w:.3f} W", f"limit {v1_05_compute.power_budget_w:.3f} W"),
    )
    _budget_rows = tuple(
        (
            name,
            f"{ratio:.2f}x",
            "binding" if name == v1_05_binding_ratio["name"] else "headroom",
        )
        for name, ratio in v1_05_ratios.items()
    )
    _batch_rows = tuple(
        (
            row["label"],
            row["batch"],
            f"{row['precision_bytes']} B",
            f"{row['activation_mb']:.2f} MB",
            f"{row['latency_ms']:.3f} ms",
            f"{row['throughput_per_s']:.1f}/s",
            v1_05_energy_display(v1_05_compute.track_id, row["energy_j"]),
            row["binding"],
            "selected" if row["selected"] else ("pass" if row["feasible"] else "fail"),
        )
        for row in v1_05_batch_rows
    )
    _design_rows = tuple(
        (
            result.design_label,
            f"{result.activation_mb:.2f} MB",
            f"{result.latency_ms:.3f} ms",
            f"{result.bandwidth_gbs:.2f} GB/s",
            "pass" if result.feasible else "fail",
            result.quality_risk,
        )
        for result in v1_05_design_results
    )

    _violations = "; ".join(v1_05_ledger.violations) if v1_05_ledger.violations else "no current budget violation"
    _threshold = (
        f"{v1_05_cliff.threshold_multiplier:.2f}x"
        if v1_05_cliff.threshold_multiplier is not None
        else "not reached"
    )
    _batch_violation = (
        "; ".join(v1_05_selected_batch_row["violations"])
        if v1_05_selected_batch_row["violations"]
        else "selected policy stays inside the modeled envelope"
    )

    _part_a = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Tensor Shape Growth Changes Amounts</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            You are the {v1_05_story["stakeholder"]}. The operator story is:
            {v1_05_compute.operator_story}</div>
        </div>
        """),
        v1_05_resource_prediction,
        v1_05_shape_multiplier,
        mo.Html(v1_05_prediction_html(
            "Prediction Check",
            v1_05_resource_prediction.value,
            v1_05_ledger.dominant_resource,
            _resource_labels,
        )),
        mo.as_html(_growth_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Operation Ledger Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_05_fields_html({
                "Dominant bounded amount": _resource_labels.get(v1_05_ledger.dominant_resource, v1_05_ledger.dominant_resource),
                "Shape multiplier": f"{v1_05_shape_multiplier.value:.2f}x",
                "Activation memory": f"{v1_05_ledger.activations_mb:.2f} MB / {v1_05_compute.activation_budget_mb:.2f} MB",
                "Operations": f"{v1_05_ledger.ops_gmac:.2f} GMAC",
                "Bytes moved": f"{v1_05_ledger.bytes_moved_mb:.2f} MB",
                "Track amount lens": v1_05_story["amount_focus"],
            })}
          </div>
          {v1_05_table_html(("Amount", "Value", "Meaning"), _ledger_rows, numeric=(1,))}
        </div>
        """),
        mo.Html(v1_05_callout_html(
            "Consequence",
            f"For {v1_05_compute.label}, this shape currently makes {v1_05_ledger.dominant_resource} the controlling amount.",
            "info",
        )),
        mo.accordion({
            "Math Peek / Source Model - shape to amount": mo.md("""
            A forward layer transforms `A_prev` through `Z = A_prev W + b` and
            `A = f(Z)`. The tensor dimensions set both activation elements and
            operation count. In this lab the shared helper computes:

            `activation_MB = tensor_elements x bytes_per_value / 1e6`

            `arithmetic_intensity = operation_count / bytes_moved`
            """),
        }),
        source_trace(
            {
                "chapter_anchor": "Forward pass computation; matrix multiplication formulation",
                "shared_helper": "operation_ledger()",
                "hardware_ref": v1_05_variant.hardware_ref,
                "model_ref": v1_05_variant.model_ref,
                "shape_multiplier": f"{v1_05_shape_multiplier.value:.2f}",
            },
            summary="Part A source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_05_amount_checkpoint,
    ])

    _part_b = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Memory And Activations Can Bind First</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            The stakeholder asks how far {v1_05_compute.tensor_label} can grow before
            the track's activation, bandwidth, latency, or power budget fails.</div>
        </div>
        """),
        v1_05_cliff_prediction,
        v1_05_shape_multiplier,
        mo.Html(v1_05_prediction_html(
            "Prediction Check",
            v1_05_cliff_prediction.value,
            v1_05_cliff_actual,
            _cliff_labels,
        )),
        mo.hstack([mo.as_html(_cliff_fig), mo.as_html(_budget_fig)], widths="equal"),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Boundary Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_05_fields_html({
                "First activation cliff": _threshold,
                "Current activation": f"{v1_05_ledger.activations_mb:.2f} MB",
                "Activation budget": f"{v1_05_compute.activation_budget_mb:.2f} MB",
                "Binding budget": f"{v1_05_binding_ratio['name']} at {v1_05_binding_ratio['ratio']:.2f}x",
                "Current feasible": "yes" if v1_05_ledger.feasible else "no",
                "Violations": _violations,
            })}
          </div>
          {v1_05_table_html(("Budget amount", "Used / limit", "Status"), _budget_rows, numeric=(1,))}
        </div>
        """),
        mo.Html(v1_05_callout_html(
            "Consequence",
            (
                f"Current shape is inside the envelope. The measured cliff is {_threshold}."
                if v1_05_ledger.feasible
                else f"Current shape fails: {_violations}. Move the slider down, tile, stream, or reduce precision to recover."
            ),
            "ok" if v1_05_ledger.feasible else "fail",
        )),
        mo.accordion({
            "Math Peek / Source Model - feasible envelope": mo.md("""
            The boundary is not a single FLOP number. Feasibility is a conjunction:

            `activation_ok and bandwidth_ok and latency_ok and power_ok`

            The binding amount is the largest normalized ratio:

            `max(actual_amount / budget_amount)`
            """),
        }),
        source_trace(
            {
                "chapter_anchor": "Memory: training vs. inference; Quick estimation for ML engineers",
                "shared_helpers": "operation_ledger() and memory_cliff()",
                "activation_budget_mb": f"{v1_05_compute.activation_budget_mb:.2f}",
                "threshold_multiplier": _threshold,
                "track_id": v1_05_compute.track_id,
            },
            summary="Part B source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_05_memory_checkpoint,
    ])

    _part_c = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Batch And Precision Trade Amounts</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            Batch/window policy for this track means {v1_05_story["batch_meaning"]}.
            Precision changes bytes per value, memory traffic, and validation risk.</div>
        </div>
        """),
        v1_05_batch_prediction,
        mo.hstack([v1_05_batch_policy, v1_05_precision_policy], justify="start", gap="2rem"),
        mo.Html(v1_05_prediction_html(
            "Prediction Check",
            v1_05_batch_prediction.value,
            v1_05_batch_actual,
            _batch_labels,
        )),
        mo.as_html(_batch_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Batch/Precision Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_05_fields_html({
                "Selected strategy": v1_05_selected_batch_row["label"],
                "Selected batch/window": v1_05_selected_batch_row["batch"],
                "Precision bytes": v1_05_selected_batch_row["precision_bytes"],
                "Throughput estimate": f"{v1_05_selected_batch_row['throughput_per_s']:.1f}/s",
                "Energy amount": v1_05_energy_display(v1_05_compute.track_id, v1_05_selected_batch_row["energy_j"]),
                "Binding amount": f"{v1_05_selected_batch_row['binding']} at {v1_05_selected_batch_row['binding_ratio']:.2f}x",
            })}
          </div>
          {v1_05_table_html(
              ("Strategy", "Batch", "Precision", "Activation", "Latency", "Throughput", "Energy", "Binding", "Status"),
              _batch_rows,
              numeric=(1, 2, 3, 4, 5, 6),
          )}
        </div>
        """),
        mo.Html(v1_05_callout_html(
            "Consequence",
            _batch_violation,
            "ok" if v1_05_selected_batch_row["feasible"] else "fail",
        )),
        mo.accordion({
            "Math Peek / Source Model - batch and precision": mo.md("""
            Batch/window policy changes the number of activation elements held at
            once. Precision changes bytes per activation and bytes per weight.

            `activation_MB = batch x tensor_shape x bytes_per_value / 1e6`

            `energy = estimated_power x latency`

            Larger batches can improve useful work per launch, but they also
            increase activation residency and can move the p99 or wake-time wall.
            """),
        }),
        source_trace(
            {
                "chapter_anchor": "Batch Processing footnote; precision trades against power",
                "shared_helper": "operation_ledger()",
                "local_helper": "v1_05_batch_precision_rows()",
                "track_id": v1_05_compute.track_id,
                "selected_batch_policy": v1_05_batch_policy.value,
                "selected_precision_policy": v1_05_precision_policy.value,
            },
            summary="Part C source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_05_batch_checkpoint,
    ])

    _part_d = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part D: Diagnosis Determines Optimization</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            You need to choose an optimization. A compute-bound operator wants fewer
            operations or a better kernel; a memory-bound operator wants fewer bytes,
            tiling, streaming, or reuse.</div>
        </div>
        """),
        v1_05_diagnosis_prediction,
        v1_05_design,
        mo.Html(v1_05_prediction_html(
            "Prediction Check",
            v1_05_diagnosis_prediction.value,
            v1_05_diagnosis["diagnosed_wall"],
            _diagnosis_labels,
        )),
        mo.as_html(_roof_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Diagnosis Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_05_fields_html({
                "Arithmetic intensity": f"{v1_05_ledger.arithmetic_intensity:.2f} ops/byte",
                "Hardware crossover": f"{v1_05_diagnosis['crossover']:.2f} ops/byte",
                "Roofline wall": _diagnosis_labels.get(v1_05_diagnosis["roofline_wall"], v1_05_diagnosis["roofline_wall"]),
                "Diagnosed wall": _diagnosis_labels.get(v1_05_diagnosis["diagnosed_wall"], v1_05_diagnosis["diagnosed_wall"]),
                "Recommended family": v1_05_diagnosis["recommendation"],
                "Selected design": v1_05_design_result.design_label,
            })}
          </div>
          {v1_05_table_html(
              ("Design", "Activation", "Latency", "Bandwidth", "Feasible", "Quality risk"),
              _design_rows,
              numeric=(1, 2, 3),
          )}
        </div>
        """),
        mo.Html(v1_05_callout_html(
            "Consequence",
            v1_05_alignment["message"],
            "ok" if v1_05_alignment["aligned"] else "warn",
        )),
        mo.Html(v1_05_callout_html(
            "Selected Design Risk",
            f"{v1_05_design_result.quality_risk}; residual risk: {v1_05_design_result.residual_risk}",
            "info",
        )),
        mo.accordion({
            "Math Peek / Source Model - compute vs memory": mo.md("""
            Arithmetic intensity asks how much arithmetic is done for each byte
            moved. The roofline crossover is the point where memory bandwidth and
            peak compute can both be used:

            `crossover = peak_ops_per_second / memory_bytes_per_second`

            Below the crossover, reducing bytes moved is usually more valuable
            than adding compute. Above it, reducing operations or improving the
            kernel is more likely to help.
            """),
        }),
        source_trace(
            {
                "chapter_anchor": "Memory wall; arithmetic intensity of matrix multiply vs. element-wise work",
                "shared_helpers": "operation_ledger() and operator_design()",
                "local_helper": "v1_05_roofline_diagnosis()",
                "track_id": v1_05_compute.track_id,
                "selected_design": v1_05_design.value,
            },
            summary="Part D source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_05_optimization_checkpoint,
    ])

    _synthesis = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Synthesis: Operator Budget Note</h2></div>
          <div class="mlsysbook-callout"><strong>Invariant:</strong>
            Neural computation turns tensors into bounded amounts of operations,
            activations, memory traffic, and energy. The budget note records which
            amount bound this design and what architecture implication carries to V1-06.</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Decision Record</h2>
          <div class="mlsysbook-grid">
            {v1_05_fields_html({
                "Track": v1_05_compute.label,
                "Tensor": v1_05_compute.tensor_label,
                "Binding amount": f"{v1_05_binding_ratio['name']} at {v1_05_binding_ratio['ratio']:.2f}x",
                "Selected design": v1_05_design_result.design_label,
                "Batch/precision": v1_05_selected_batch_row["label"],
                "Diagnosis": _diagnosis_labels.get(v1_05_diagnosis["diagnosed_wall"], v1_05_diagnosis["diagnosed_wall"]),
                "Residual risk": v1_05_design_result.residual_risk,
                "Next-lab implication": v1_05_story["next_lab"],
            })}
          </div>
        </div>
        """),
        v1_05_final_decision,
        v1_05_budget_note,
        mo.accordion({
            "Math Peek / Source Model - operator budget note": mo.md("""
            A complete operator budget note has five fields:

            1. decision
            2. binding amount
            3. evidence number
            4. residual risk
            5. architecture implication

            That record is what later labs can reuse when architecture, training,
            compression, acceleration, and serving decisions change the envelope.
            """),
        }),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">05 &middot; Neural Computation</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v1_05_compute.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">BINDING</span>
            <span class="hud-value">{v1_05_binding_ratio["name"]}</span>
            <span class="hud-label">STATUS</span>
            <span class="hud-active">ACTIVE</span>
        </div>
        """),
    ])

    def build_part_a():
        return _part_a

    def build_part_b():
        return _part_b

    def build_part_c():
        return _part_c

    def build_part_d():
        return _part_d

    def build_synthesis():
        return _synthesis

    v1_05_tabs = mo.ui.tabs({
        "Part A: Shape Amounts": build_part_a(),
        "Part B: Activation Cliff": build_part_b(),
        "Part C: Batch/Precision": build_part_c(),
        "Part D: Diagnosis": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    v1_05_tabs
    return (v1_05_tabs,)


# ===========================================================================
# ZONE D: LEDGER AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    v1_05_alignment,
    v1_05_amount_checkpoint,
    v1_05_batch_checkpoint,
    v1_05_batch_policy,
    v1_05_batch_prediction,
    v1_05_binding_ratio,
    v1_05_budget_note,
    v1_05_cliff,
    v1_05_cliff_prediction,
    v1_05_compute,
    v1_05_design_result,
    v1_05_diagnosis,
    v1_05_diagnosis_prediction,
    v1_05_final_decision,
    v1_05_ledger,
    v1_05_memory_checkpoint,
    v1_05_optimization_checkpoint,
    v1_05_precision_policy,
    v1_05_profile,
    v1_05_resource_prediction,
    v1_05_selected_batch_row,
    v1_05_shape_multiplier,
    v1_05_story,
    v1_05_variant,
):
    if v1_05_resource_prediction.value is not None:
        ledger.save(chapter=5, design={
            "chapter": "v1_05",
            "track_id": v1_05_profile.track_id,
            "scenario_id": v1_05_variant.scenario_id,
            "hardware_ref": v1_05_compute.hardware_ref,
            "model_ref": v1_05_compute.model_ref,
            "completed": v1_05_final_decision.value is not None and bool(str(v1_05_budget_note.value or "").strip()),
            "resource_prediction": v1_05_resource_prediction.value,
            "dominant_resource": v1_05_ledger.dominant_resource,
            "shape_multiplier": v1_05_shape_multiplier.value,
            "activation_memory_mb": v1_05_ledger.activations_mb,
            "ops_gmac": v1_05_ledger.ops_gmac,
            "bytes_moved_mb": v1_05_ledger.bytes_moved_mb,
            "memory_cliff_prediction": v1_05_cliff_prediction.value,
            "memory_cliff_multiplier": v1_05_cliff.threshold_multiplier,
            "amount_checkpoint": v1_05_amount_checkpoint.value,
            "memory_checkpoint": v1_05_memory_checkpoint.value,
            "batch_prediction": v1_05_batch_prediction.value,
            "batch_policy": v1_05_batch_policy.value,
            "precision_policy": v1_05_precision_policy.value,
            "batch_precision_binding": v1_05_selected_batch_row["binding"],
            "throughput_per_s": v1_05_selected_batch_row["throughput_per_s"],
            "energy_j": v1_05_selected_batch_row["energy_j"],
            "diagnosis_prediction": v1_05_diagnosis_prediction.value,
            "diagnosed_wall": v1_05_diagnosis["diagnosed_wall"],
            "operator_design": v1_05_design_result.design_id,
            "optimization_checkpoint": v1_05_optimization_checkpoint.value,
            "design_alignment": v1_05_alignment["aligned"],
            "binding_amount": v1_05_binding_ratio["name"],
            "binding_ratio": v1_05_binding_ratio["ratio"],
            "quality_risk": v1_05_design_result.quality_risk,
            "residual_risk": v1_05_design_result.residual_risk,
            "final_decision": v1_05_final_decision.value,
            "operator_budget_note": v1_05_budget_note.value,
            "next_lab_implication": v1_05_story["next_lab"],
        })

    mo.Html(f"""
    <div class="mlsysbook-panel">
      <h2>Design Ledger</h2>
      <div class="mlsysbook-grid">
        <div class="mlsysbook-field"><strong>Saved track</strong>{v1_05_profile.label}</div>
        <div class="mlsysbook-field"><strong>Binding amount</strong>{v1_05_binding_ratio["name"]}</div>
        <div class="mlsysbook-field"><strong>Selected design</strong>{v1_05_design_result.design_label}</div>
        <div class="mlsysbook-field"><strong>Next lab</strong>{v1_05_story["next_lab"]}</div>
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    report_export_panel,
    v1_05_alignment,
    v1_05_amount_checkpoint,
    v1_05_batch_checkpoint,
    v1_05_batch_policy,
    v1_05_batch_prediction,
    v1_05_batch_rows,
    v1_05_binding_ratio,
    v1_05_budget_note,
    v1_05_cliff,
    v1_05_cliff_prediction,
    v1_05_compute,
    v1_05_design_result,
    v1_05_diagnosis,
    v1_05_diagnosis_prediction,
    v1_05_final_decision,
    v1_05_ledger,
    v1_05_memory_checkpoint,
    v1_05_metadata,
    v1_05_optimization_checkpoint,
    v1_05_precision_policy,
    v1_05_profile,
    v1_05_resource_prediction,
    v1_05_selected_batch_row,
    v1_05_shape_multiplier,
    v1_05_story,
    v1_05_variant,
):
    _incomplete = []
    _required = (
        ("Part A dominant-amount prediction", v1_05_resource_prediction.value),
        ("Part A checkpoint", v1_05_amount_checkpoint.value),
        ("Part B cliff prediction", v1_05_cliff_prediction.value),
        ("Part B checkpoint", v1_05_memory_checkpoint.value),
        ("Part C batch/precision prediction", v1_05_batch_prediction.value),
        ("Part C checkpoint", v1_05_batch_checkpoint.value),
        ("Part D diagnosis prediction", v1_05_diagnosis_prediction.value),
        ("Part D checkpoint", v1_05_optimization_checkpoint.value),
        ("Synthesis decision", v1_05_final_decision.value),
    )
    for label, value in _required:
        if value is None:
            _incomplete.append(label)
    if not str(v1_05_budget_note.value or "").strip():
        _incomplete.append("Synthesis operator budget note")

    _report = build_lab_report(
        v1_05_metadata,
        track=v1_05_profile.label,
        scenario=v1_05_variant.workload_summary,
        learning_objectives=(
            "Trace tensor shape growth into activation memory, operations, bytes moved, and energy.",
            "Find the binding activation, bandwidth, latency, or power budget.",
            "Compare batch and precision choices across throughput, memory, latency, and energy.",
            "Diagnose compute-vs-memory behavior before choosing an operator optimization.",
        ),
        predictions={
            "part_a_dominant_amount": v1_05_resource_prediction.value,
            "part_b_cliff": v1_05_cliff_prediction.value,
            "part_c_batch_precision": v1_05_batch_prediction.value,
            "part_d_diagnosis": v1_05_diagnosis_prediction.value,
        },
        knob_settings={
            "shape_multiplier": v1_05_shape_multiplier.value,
            "batch_policy": v1_05_batch_policy.value,
            "precision_policy": v1_05_precision_policy.value,
            "operator_design": v1_05_design_result.design_id,
        },
        evidence_summary={
            "hardware_ref": v1_05_compute.hardware_ref,
            "model_ref": v1_05_compute.model_ref,
            "dominant_resource": v1_05_ledger.dominant_resource,
            "binding_amount": v1_05_binding_ratio["name"],
            "binding_ratio": v1_05_binding_ratio["ratio"],
            "activation_memory_mb": v1_05_ledger.activations_mb,
            "activation_budget_mb": v1_05_compute.activation_budget_mb,
            "ops_gmac": v1_05_ledger.ops_gmac,
            "bytes_moved_mb": v1_05_ledger.bytes_moved_mb,
            "arithmetic_intensity": v1_05_ledger.arithmetic_intensity,
            "roofline_crossover": v1_05_diagnosis["crossover"],
            "diagnosed_wall": v1_05_diagnosis["diagnosed_wall"],
            "memory_cliff_multiplier": v1_05_cliff.threshold_multiplier,
            "selected_batch_precision": v1_05_selected_batch_row["label"],
            "selected_throughput_per_s": v1_05_selected_batch_row["throughput_per_s"],
            "selected_energy_j": v1_05_selected_batch_row["energy_j"],
            "selected_design": v1_05_design_result.design_label,
            "design_alignment": v1_05_alignment["aligned"],
            "quality_risk": v1_05_design_result.quality_risk,
        },
        final_decision=(
            f"{v1_05_final_decision.value or 'pending'}; "
            f"binding amount is {v1_05_binding_ratio['name']} and selected design is {v1_05_design_result.design_label}."
        ),
        big_takeaways=(
            "Tensor shapes are amount systems: they create operations, activations, bytes moved, and energy.",
            "Activation or memory-traffic budgets can bind before arithmetic throughput is exhausted.",
            "Batch and precision are deployment controls whose value depends on the selected track.",
            "The right optimization follows the measured compute-vs-memory diagnosis.",
        ),
        reflections={
            "part_a_checkpoint": v1_05_amount_checkpoint.value,
            "part_b_checkpoint": v1_05_memory_checkpoint.value,
            "part_c_checkpoint": v1_05_batch_checkpoint.value,
            "part_d_checkpoint": v1_05_optimization_checkpoint.value,
            "operator_budget_note": v1_05_budget_note.value,
            "residual_risk": v1_05_design_result.residual_risk,
            "next_lab_implication": v1_05_story["next_lab"],
        },
        residual_risk=v1_05_design_result.residual_risk,
        source_trace={
            "track_id": v1_05_profile.track_id,
            "scenario_id": v1_05_variant.scenario_id,
            "hardware_ref": v1_05_variant.hardware_ref,
            "model_ref": v1_05_variant.model_ref,
            "shared_helpers": "neural_compute_profile(), operation_ledger(), memory_cliff(), operator_design()",
            "local_helpers": "v1_05_budget_ratios(), v1_05_batch_precision_rows(), v1_05_roofline_diagnosis()",
            "source_policy": v1_05_profile.source_policy,
        },
        result_snapshot={
            "compute_profile": v1_05_compute,
            "operation_ledger": v1_05_ledger,
            "memory_cliff": v1_05_cliff,
            "batch_precision_rows": v1_05_batch_rows,
            "operator_design": v1_05_design_result,
            "diagnosis": v1_05_diagnosis,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-05 operator budget note is generated locally from the selected track, "
                "your predictions, controls, and computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
