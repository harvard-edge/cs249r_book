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
        compile_break_even,
        dispatch_stack,
        framework_track_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        runtime_decision,
        source_trace,
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
        compile_break_even,
        dispatch_stack,
        framework_track_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        runtime_decision,
        source_trace,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_07_metadata = get_lab_metadata("vol1/lab_07_ml_frameworks.py")
    return (v1_07_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_07_track_picker = track_selector(default=_default_track)
    v1_07_track_picker
    return (v1_07_track_picker,)


@app.cell
def _(
    framework_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_07_track_picker,
):
    v1_07_track_id = v1_07_track_picker.value
    v1_07_profile = get_track_profile(v1_07_track_id)
    v1_07_variant = get_lab_track_variant("v1_07_framework_tax", v1_07_profile.track_id)
    v1_07_hardware = resolve_mlsysim_ref(v1_07_variant.hardware_ref)
    v1_07_model = resolve_mlsysim_ref(v1_07_variant.model_ref)
    v1_07_framework = framework_track_profile(
        v1_07_profile,
        v1_07_variant,
        v1_07_hardware,
        v1_07_model,
    )
    return (
        v1_07_framework,
        v1_07_hardware,
        v1_07_model,
        v1_07_profile,
        v1_07_track_id,
        v1_07_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    v1_07_framework,
    v1_07_metadata,
    v1_07_profile,
    v1_07_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 07
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Framework Tax
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Dispatch &middot; Fusion &middot; Compile Break-Even &middot; Unsupported Ops
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 860px; line-height: 1.65;">
                {v1_07_variant.workload_summary} This lab treats the framework as part
                of the system: runtime memory, dispatch, kernel support, and compile
                amortization can decide whether a model ships.
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
                    {v1_07_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_07_framework.workload_label}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Dispatch Stack</span>
                <span class="badge badge-warn">Compile Break-Even</span>
                <span class="badge badge-fail">Runtime Recommendation</span>
            </div>
        </div>
        """),
        track_context(v1_07_profile),
        source_trace(
            {
                "lab_id": v1_07_metadata.lab_id,
                "track_id": v1_07_profile.track_id,
                "hardware_ref": v1_07_variant.hardware_ref,
                "model_ref": v1_07_variant.model_ref,
                "shared_helper": "mlsysbook_labs.frameworks",
                "source_policy": v1_07_profile.source_policy,
            },
            summary="V1-07 evaluates runtime dispatch, compile break-even, and unsupported-op risk through mlsysbook_labs.frameworks.",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_07_framework):
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
            <div style="margin-bottom: 3px;">1. <strong>Quantify dispatch tax:</strong>
                separate useful compute from runtime dispatch, hardware dispatch, transfer, sync, and memory traffic.</div>
            <div style="margin-bottom: 3px;">2. <strong>Find compile break-even:</strong>
                compare upfront compile/delegate cost with per-inference savings under the selected reuse count.</div>
            <div style="margin-bottom: 3px;">3. <strong>Choose a runtime:</strong>
                defend a deployment path and name unsupported-op, portability, or rollback risk.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                Which runtime path fits {v1_07_framework.label}, and when does compile
                or fusion cost actually pay back?
            </div>
        </div>
    </div>
    """)
    return


# ===========================================================================
# ZONE B: CONTROLS AND COMPUTATION
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_07_framework):
    v1_07_overhead_prediction = mo.ui.radio(
        options={
            "Runtime dispatch dominates": "runtime dispatch",
            "Hardware dispatch dominates": "hardware dispatch",
            "Memory traffic dominates": "memory traffic",
            "Unsupported-op fallback dominates": "unsupported ops",
        },
        label=f"{v1_07_framework.label}: which framework overhead do you expect to bind first?",
    )
    v1_07_overhead_prediction
    return (v1_07_overhead_prediction,)


@app.cell(hide_code=True)
def _(mo, v1_07_framework):
    v1_07_op_count = mo.ui.slider(
        start=max(10, int(v1_07_framework.op_count / 4)),
        stop=max(20, int(v1_07_framework.op_count * 4)),
        value=v1_07_framework.op_count,
        step=max(1, int(v1_07_framework.op_count / 20)),
        label="Operation count",
    )
    v1_07_reuse_count = mo.ui.slider(
        start=v1_07_framework.reuse_min,
        stop=v1_07_framework.reuse_max,
        value=v1_07_framework.default_reuse_count,
        step=v1_07_framework.reuse_step,
        label="Expected reuse count",
    )
    return (v1_07_op_count, v1_07_reuse_count)


@app.cell(hide_code=True)
def _(mo, v1_07_framework):
    _runtime_options = {
        runtime.label: runtime.runtime_id
        for runtime in v1_07_framework.runtime_options
    }
    v1_07_runtime_choice = mo.ui.dropdown(
        options=_runtime_options,
        value=v1_07_framework.runtime_options[0].label,
        label="Runtime recommendation",
    )
    v1_07_reflection = mo.ui.text_area(
        label="Reflection",
        placeholder="Name the runtime you recommend, why the compile/delegate cost pays back, and what unsupported-op risk remains.",
        full_width=True,
    )
    return (v1_07_reflection, v1_07_runtime_choice)


@app.cell
def _(
    compile_break_even,
    dispatch_stack,
    runtime_decision,
    v1_07_framework,
    v1_07_op_count,
    v1_07_reuse_count,
    v1_07_runtime_choice,
):
    v1_07_dispatch_rows = tuple(
        dispatch_stack(
            v1_07_framework,
            runtime_id=runtime.runtime_id,
            op_count=v1_07_op_count.value,
        )
        for runtime in v1_07_framework.runtime_options
    )
    v1_07_break_evens = tuple(
        compile_break_even(
            v1_07_framework,
            runtime_id=runtime.runtime_id,
            reuse_count=v1_07_reuse_count.value,
            op_count=v1_07_op_count.value,
        )
        for runtime in v1_07_framework.runtime_options
    )
    v1_07_decision = runtime_decision(
        v1_07_framework,
        runtime_id=v1_07_runtime_choice.value,
        reuse_count=v1_07_reuse_count.value,
        op_count=v1_07_op_count.value,
    )
    v1_07_selected_stack = next(
        item for item in v1_07_dispatch_rows
        if item.runtime_id == v1_07_decision.selected_id
    )
    v1_07_selected_break_even = next(
        item for item in v1_07_break_evens
        if item.runtime_id == v1_07_decision.selected_id
    )
    return (
        v1_07_break_evens,
        v1_07_decision,
        v1_07_dispatch_rows,
        v1_07_selected_break_even,
        v1_07_selected_stack,
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
    v1_07_break_evens,
    v1_07_decision,
    v1_07_dispatch_rows,
    v1_07_framework,
    v1_07_op_count,
    v1_07_overhead_prediction,
    v1_07_reflection,
    v1_07_reuse_count,
    v1_07_runtime_choice,
    v1_07_selected_break_even,
    v1_07_selected_stack,
):
    _labels = [row.runtime_label for row in v1_07_dispatch_rows]
    _latency_fig = go.Figure()
    _latency_fig.add_trace(go.Bar(
        x=_labels,
        y=[row.useful_compute_ms for row in v1_07_dispatch_rows],
        name="Useful compute",
        marker_color=COLORS["GreenLine"],
    ))
    _latency_fig.add_trace(go.Bar(
        x=_labels,
        y=[row.runtime_dispatch_ms for row in v1_07_dispatch_rows],
        name="Runtime dispatch",
        marker_color=COLORS["OrangeLine"],
    ))
    _latency_fig.add_trace(go.Bar(
        x=_labels,
        y=[row.hardware_dispatch_ms + row.transfer_ms + row.sync_ms + row.memory_ms for row in v1_07_dispatch_rows],
        name="Hardware/transfer/sync/memory",
        marker_color=COLORS["BlueLine"],
    ))
    _latency_fig.add_hline(
        y=v1_07_framework.latency_budget_ms,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        annotation_text="latency budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _latency_fig.update_layout(
        barmode="stack",
        height=360,
        xaxis=dict(title="Runtime path", gridcolor="#f1f5f9"),
        yaxis=dict(title="Latency stack (ms)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=80),
    )
    apply_plotly_theme(_latency_fig)

    _dispatch_rows = "".join(
        f"""
        <tr>
          <td>{row.runtime_label}</td>
          <td>{row.total_latency_ms:.2f} ms</td>
          <td>{row.overhead_pct:.1f}%</td>
          <td>{row.footprint_mb:.3f} MB</td>
          <td>{row.kernel_support_pct:.1f}%</td>
          <td>{'yes' if row.feasible else 'no - violation'}</td>
          <td>{row.dominant_overhead}</td>
        </tr>
        """
        for row in v1_07_dispatch_rows
    )

    _break_fig = go.Figure()
    _break_fig.add_trace(go.Bar(
        x=[item.runtime_label for item in v1_07_break_evens],
        y=[item.break_even_inferences or 0 for item in v1_07_break_evens],
        marker_color=[
            COLORS["GreenLine"] if item.pays_back else COLORS["RedLine"]
            for item in v1_07_break_evens
        ],
        text=[
            f"{item.break_even_inferences or 0:,}"
            for item in v1_07_break_evens
        ],
        textposition="outside",
    ))
    _break_fig.add_hline(
        y=v1_07_reuse_count.value,
        line_dash="dash",
        line_color=COLORS["BlueLine"],
        annotation_text="expected reuse",
        annotation_font_color=COLORS["BlueLine"],
    )
    _break_fig.update_layout(
        height=340,
        xaxis=dict(title="Runtime path", gridcolor="#f1f5f9"),
        yaxis=dict(title="Break-even inferences", gridcolor="#f1f5f9"),
        margin=dict(l=70, r=20, t=35, b=80),
    )
    apply_plotly_theme(_break_fig)

    _break_rows = "".join(
        f"""
        <tr>
          <td>{item.runtime_label}</td>
          <td>{item.compile_cost_s:.1f} s</td>
          <td>{item.per_inference_savings_ms:.2f} ms</td>
          <td>{item.break_even_inferences or 'no payback'}</td>
          <td>{'yes' if item.pays_back else 'no'}</td>
        </tr>
        """
        for item in v1_07_break_evens
    )
    _rejections = "".join(f"<li>{item}</li>" for item in v1_07_decision.rejected_alternatives)
    _validation_items = "".join(f"<li>{test}</li>" for test in v1_07_framework.validation_tests)

    _part_a = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Dispatch Stack</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which part of the runtime stack dominates {v1_07_framework.workload_label}?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Useful compute is only one part of runtime latency; dispatch, transfer, synchronization, and memory traffic can dominate.</li>
            <li>Runtime footprint and kernel support can make a path infeasible even when latency looks acceptable.</li>
            <li>Unsupported-op fallback is a failure mode because it changes both performance and validation evidence.</li>
          </ul>
          <div class="mlsysbook-callout"><strong>Track runtime story:</strong> {v1_07_framework.runtime_story}</div>
        </div>
        """),
        v1_07_overhead_prediction,
        v1_07_op_count,
        mo.as_html(_latency_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Table Fallback</h2>
          <table class="mlsysbook-table">
            <thead>
              <tr>
                <th>Runtime</th><th>Total latency</th><th>Overhead</th><th>Footprint</th>
                <th>Kernel support</th><th>Feasible</th><th>Dominant overhead</th>
              </tr>
            </thead>
            <tbody>{_dispatch_rows}</tbody>
          </table>
        </div>
        """),
    ])

    _part_b = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Compile Break-Even</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Does compile or delegate setup pay back for the expected number of uses?</div>
        </div>
        """),
        mo.Html("""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Compilation and graph capture move cost upfront; they only help when reuse amortizes the setup.</li>
            <li>Shape dynamism and graph breaks reduce reuse by forcing recompilation or eager fallback.</li>
            <li>The break-even point should be compared with expected product or fleet volume.</li>
          </ul>
        </div>
        """),
        v1_07_reuse_count,
        mo.as_html(_break_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Break-Even Table</h2>
          <table class="mlsysbook-table">
            <thead>
              <tr><th>Runtime</th><th>Compile cost</th><th>Savings / inference</th><th>Break-even</th><th>Pays back</th></tr>
            </thead>
            <tbody>{_break_rows}</tbody>
          </table>
        </div>
        """),
    ])

    _part_c = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Runtime Choice</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which runtime would you ship, and what fallback risk must be tested?</div>
        </div>
        """),
        mo.Html("""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>The best runtime is the one that satisfies the track constraints, not the most flexible framework.</li>
            <li>Rejected alternatives should name the resource or validation reason they fail.</li>
            <li>Runtime choice creates operational risk: graph breaks, custom kernels, plugin certification, or rollback gaps.</li>
          </ul>
        </div>
        """),
        v1_07_runtime_choice,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Selected runtime</strong>{v1_07_decision.selected_label}</div>
            <div class="mlsysbook-field"><strong>Feasible</strong>{'yes' if v1_07_decision.feasible else 'no - violation'}</div>
            <div class="mlsysbook-field"><strong>Total latency</strong>{v1_07_selected_stack.total_latency_ms:.2f} ms / {v1_07_framework.latency_budget_ms:.1f} ms</div>
            <div class="mlsysbook-field"><strong>Dominant overhead</strong>{v1_07_decision.dominant_overhead}</div>
            <div class="mlsysbook-field"><strong>Break-even</strong>{v1_07_selected_break_even.break_even_inferences or 'no payback'}</div>
            <div class="mlsysbook-field"><strong>Kernel support</strong>{v1_07_selected_stack.kernel_support_pct:.1f}% / {v1_07_framework.kernel_support_floor_pct:.1f}%</div>
          </div>
          <div class="mlsysbook-callout"><strong>Unsupported-op warning:</strong> {v1_07_decision.unsupported_op_warning}</div>
          <div class="mlsysbook-callout"><strong>Memo decision:</strong> {v1_07_decision.memo_summary}</div>
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
        source_trace(
            {
                "helper": "runtime_decision",
                "selected_id": v1_07_decision.selected_id,
                "hardware_ref": v1_07_framework.hardware_ref,
                "model_ref": v1_07_framework.model_ref,
            },
            summary="Runtime decision evidence is computed from runtime options, track budgets, and MLSysIM refs.",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Reflection</h2></div>'),
        v1_07_reflection,
    ])

    mo.ui.tabs({
        "Part A · Dispatch": _part_a,
        "Part B · Break-Even": _part_b,
        "Part C · Runtime": _part_c,
    })
    return


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    v1_07_decision,
    v1_07_framework,
    v1_07_overhead_prediction,
    v1_07_profile,
    v1_07_selected_break_even,
    v1_07_selected_stack,
    v1_07_variant,
):
    if v1_07_overhead_prediction.value is not None:
        ledger.save(chapter=7, design={
            "chapter": "v1_07",
            "track_id": v1_07_profile.track_id,
            "scenario_id": v1_07_variant.scenario_id,
            "hardware_ref": v1_07_framework.hardware_ref,
            "model_ref": v1_07_framework.model_ref,
            "completed": True,
            "overhead_prediction": v1_07_overhead_prediction.value,
            "selected_runtime": v1_07_decision.selected_id,
            "dominant_overhead": v1_07_decision.dominant_overhead,
            "break_even_inferences": v1_07_selected_break_even.break_even_inferences,
            "total_latency_ms": v1_07_selected_stack.total_latency_ms,
        })

    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Synthesis</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Track</strong>{v1_07_framework.label}</div>
            <div class="mlsysbook-field"><strong>Selected runtime</strong>{v1_07_decision.selected_label}</div>
            <div class="mlsysbook-field"><strong>Dominant overhead</strong>{v1_07_decision.dominant_overhead}</div>
            <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_07_decision.residual_risk}</div>
          </div>
        </div>
        """),
        mo.Html("""
        <div class="mlsysbook-panel">
          <h2>Big Takeaways</h2>
          <ul class="mlsysbook-list">
            <li><strong>Frameworks are execution systems.</strong> Runtime dispatch, memory footprint, and kernel support are deployment facts.</li>
            <li><strong>Compilation has a payback period.</strong> A compiled path is only useful when reuse exceeds the break-even point.</li>
            <li><strong>Unsupported ops are a systems risk.</strong> Fallback can invalidate latency, power, safety, or cost evidence.</li>
          </ul>
        </div>
        """),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">07 &middot; Framework Tax</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v1_07_profile.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">ARTIFACT</span>
            <span class="hud-value">{v1_07_framework.report_artifact}</span>
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
    v1_07_break_evens,
    v1_07_decision,
    v1_07_dispatch_rows,
    v1_07_framework,
    v1_07_metadata,
    v1_07_op_count,
    v1_07_overhead_prediction,
    v1_07_profile,
    v1_07_reflection,
    v1_07_reuse_count,
    v1_07_selected_break_even,
    v1_07_selected_stack,
    v1_07_variant,
):
    _incomplete = []
    if v1_07_overhead_prediction.value is None:
        _incomplete.append("Part A overhead prediction")
    if not str(v1_07_reflection.value or "").strip():
        _incomplete.append("Part C reflection")

    _report = build_lab_report(
        v1_07_metadata,
        track=v1_07_profile.label,
        scenario=v1_07_variant.workload_summary,
        learning_objectives=(
            "Separate useful compute from runtime dispatch, hardware dispatch, transfer, sync, and memory traffic.",
            "Calculate compile or delegate break-even from reuse count and per-inference savings.",
            "Choose a runtime path and state unsupported-op, portability, or rollback risk.",
        ),
        predictions={
            "first_overhead_risk": v1_07_overhead_prediction.value,
        },
        knob_settings={
            "operation_count": v1_07_op_count.value,
            "reuse_count": v1_07_reuse_count.value,
            "selected_runtime": v1_07_decision.selected_id,
        },
        evidence_summary={
            "hardware_ref": v1_07_framework.hardware_ref,
            "model_ref": v1_07_framework.model_ref,
            "latency_budget_ms": v1_07_framework.latency_budget_ms,
            "memory_budget_mb": v1_07_framework.memory_budget_mb,
            "selected_runtime": v1_07_decision.selected_label,
            "total_latency_ms": v1_07_selected_stack.total_latency_ms,
            "dominant_overhead": v1_07_decision.dominant_overhead,
            "break_even_inferences": v1_07_selected_break_even.break_even_inferences,
            "unsupported_op_warning": v1_07_decision.unsupported_op_warning,
        },
        final_decision=v1_07_decision.memo_summary,
        big_takeaways=(
            "A framework can make a model infeasible through dispatch, footprint, or unsupported-op fallback.",
            "Compile and fusion paths only help when reuse amortizes setup cost.",
            "Runtime recommendations must include validation requirements, not only latency numbers.",
        ),
        reflections={
            "student_reflection": v1_07_reflection.value,
            "rejected_alternatives": v1_07_decision.rejected_alternatives,
            "validation_requirement": v1_07_decision.validation_requirement,
            "report_artifact": v1_07_framework.report_artifact,
        },
        residual_risk=v1_07_decision.residual_risk,
        source_trace={
            "track_id": v1_07_profile.track_id,
            "scenario_id": v1_07_variant.scenario_id,
            "hardware_ref": v1_07_variant.hardware_ref,
            "model_ref": v1_07_variant.model_ref,
            "shared_helper": "mlsysbook_labs.frameworks",
            "source_policy": v1_07_profile.source_policy,
        },
        result_snapshot={
            "framework_profile": v1_07_framework,
            "dispatch_rows": v1_07_dispatch_rows,
            "break_even_rows": v1_07_break_evens,
            "selected_stack": v1_07_selected_stack,
            "decision": v1_07_decision,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-07 runtime deployment recommendation is generated locally from "
                "the selected track, MLSysIM refs, and shared `mlsysbook_labs.frameworks` calculations."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
