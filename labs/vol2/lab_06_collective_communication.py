import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import html as html_lib
    import sys
    from pathlib import Path
    import numpy as np

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
    import mlsysim
    from mlsysim import Hardware, Systems, ureg
    from mlsysim.physics import (
        calc_hierarchical_allreduce_time,
        calc_ring_allreduce_time,
        calc_tree_allreduce_time,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
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
        Hardware,
        LAB_CSS,
        Systems,
        apply_plotly_theme,
        build_lab_report,
        calc_hierarchical_allreduce_time,
        calc_ring_allreduce_time,
        calc_tree_allreduce_time,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        html_lib,
        ledger,
        mlsysim,
        mo,
        np,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_selector,
        ureg,
    )


@app.cell
def _(get_lab_metadata):
    v2_06_metadata = get_lab_metadata("vol2/lab_06_collective_communication.py")
    return (v2_06_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_06_track_picker = track_selector(default=_default_track)
    v2_06_track_picker
    return (v2_06_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v2_06_track_picker,
):
    v2_06_track_id = v2_06_track_picker.value
    v2_06_profile = get_track_profile(v2_06_track_id)
    v2_06_variant = get_lab_track_variant("v2_06_collective_communication", v2_06_profile.track_id)
    v2_06_hardware = resolve_mlsysim_ref(v2_06_variant.hardware_ref)
    v2_06_model = resolve_mlsysim_ref(v2_06_variant.model_ref)
    v2_06_defaults = v2_06_variant.defaults
    return (
        v2_06_defaults,
        v2_06_hardware,
        v2_06_model,
        v2_06_profile,
        v2_06_track_id,
        v2_06_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    v2_06_defaults,
    v2_06_metadata,
    v2_06_profile,
    v2_06_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 32px 40px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #94a3b8; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 06
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.35rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Collective Communication
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.1rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Alpha-Beta &middot; Ring/Tree/Hierarchy &middot; Overlap &middot; Compression
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 760px; line-height: 1.65;">
                {v2_06_variant.workload_summary} Communication algorithm choice is a
                systems decision, not a library detail.
            </p>
            <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px;">
                <span style="background: rgba(99,102,241,0.18); color:#a5b4fc;
                             padding:5px 14px; border-radius:20px; font-size:0.8rem;
                             font-weight:600; border:1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~45 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color:#fca5a5;
                             padding:5px 14px; border-radius:20px; font-size:0.8rem;
                             font-weight:600; border:1px solid rgba(203,32,45,0.25);">
                    {v2_06_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color:#86efac;
                             padding:5px 14px; border-radius:20px; font-size:0.8rem;
                             font-weight:600; border:1px solid rgba(34,197,94,0.20);">
                    {v2_06_variant.hardware_ref}
                </span>
            </div>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <span class="badge badge-info">{v2_06_defaults["operation"]}</span>
                <span class="badge badge-warn">{v2_06_defaults["topology"]}</span>
                <span class="badge badge-fail">{v2_06_variant.guardrail_metric}</span>
            </div>
        </div>
        """),
        track_context(v2_06_profile),
        source_trace(
            {
                "lab_id": v2_06_metadata.lab_id,
                "track_id": v2_06_profile.track_id,
                "hardware_ref": v2_06_variant.hardware_ref,
                "model_ref": v2_06_variant.model_ref,
                "operation": v2_06_defaults["operation"],
                "topology": v2_06_defaults["topology"],
                "shared_solver": "mlsysim.physics collective communication",
                "source_policy": v2_06_profile.source_policy,
            },
            summary="V2-06 uses typed track variants plus MLSysIM collective physics for ring, tree, and hierarchical timing.",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v2_06_defaults, v2_06_variant):
    mo.Html(f"""
    <div style="border-left:4px solid {COLORS['BlueLine']}; background:white;
                border-radius:0 12px 12px 0; padding:20px 28px; margin:8px 0 16px 0;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                    text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
            Learning Objectives
        </div>
        <div style="font-size:0.9rem; color:{COLORS['TextSec']}; line-height:1.7;">
            <div style="margin-bottom:3px;">1. <strong>Decompose collective cost:</strong>
                distinguish bytes moved, latency rounds, participants, and topology.</div>
            <div style="margin-bottom:3px;">2. <strong>Compare algorithms:</strong>
                ring, tree, and hierarchy change winners as message size and fabric change.</div>
            <div style="margin-bottom:3px;">3. <strong>Make an optimization decision:</strong>
                compression and overlap reduce exposed time but add {v2_06_variant.guardrail_metric} risk.</div>
        </div>
        <div style="border-top:1px solid {COLORS['Border']}; margin:14px -28px 0 -28px;
                    padding:14px 28px 0 28px;">
            <div style="font-size:0.7rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                Starting Scenario
            </div>
            <div style="font-size:1.0rem; color:{COLORS['Text']}; font-weight:600; line-height:1.5;">
                {v2_06_defaults["participants"]} participants, {v2_06_defaults["message_gb"]} GB per participant,
                topology: {v2_06_defaults["topology"]}.
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** - Complete before this lab:

    - **Volume II, Chapter 6: Collective Communication** - AllReduce, AllGather,
      ReduceScatter, alpha-beta models, topology hierarchy, overlap, and compression.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo, v2_06_defaults):
    partA_prediction = mo.ui.radio(
        options={
            "A) The number of model FLOPs": "flops",
            "B) Bytes, rounds, participants, and topology": "bytes_topology",
            "C) The optimizer choice only": "optimizer",
            "D) Dataset size only": "dataset",
        },
        label="What primarily determines collective/coordination time once the operation is fixed?",
    )
    partB_prediction = mo.ui.radio(
        options={
            "A) Ring is always best": "ring",
            "B) Tree is always best": "tree",
            "C) Winner depends on message size, participants, and topology": "depends",
            "D) Algorithm choice only matters in cloud": "cloud_only",
        },
        label="Which collective/coordination strategy should win across all message sizes?",
    )
    partC_prediction = mo.ui.radio(
        options={
            "A) Hierarchy helps when local links are much faster than global links": "hierarchy",
            "B) Hierarchy removes communication": "free",
            "C) Hierarchy is unrelated to topology": "unrelated",
            "D) Hierarchy only helps one participant": "single",
        },
        label=f"When does hierarchy help for {v2_06_defaults['topology']}?",
    )
    partD_prediction = mo.ui.radio(
        options={
            "A) Compression/overlap reduce exposed time but add validation risk": "risk",
            "B) Compression is always free": "free",
            "C) Overlap makes bandwidth irrelevant": "irrelevant",
            "D) Compression only changes optimizer math": "optimizer",
        },
        label="What risk remains after compression or overlap?",
    )
    return partA_prediction, partB_prediction, partC_prediction, partD_prediction


@app.cell(hide_code=True)
def _(mo, v2_06_defaults):
    _fabric_label = {"ib": "InfiniBand NDR", "eth": "100G Ethernet", "nvlink": "NVLink-only"}.get(
        v2_06_defaults["fabric"],
        "InfiniBand NDR",
    )
    n_gpus = mo.ui.slider(
        start=2,
        stop=1024,
        value=int(v2_06_defaults["participants"]),
        step=2,
        label="Participants",
    )
    message_gb = mo.ui.slider(
        start=0.001,
        stop=80.0,
        value=float(v2_06_defaults["message_gb"]),
        step=0.001 if float(v2_06_defaults["message_gb"]) < 0.1 else 0.1,
        label="Message/payload per participant (GB)",
    )
    fabric = mo.ui.dropdown(
        options={"InfiniBand NDR": "ib", "100G Ethernet": "eth", "NVLink-only": "nvlink"},
        value=_fabric_label,
        label="Fabric / link analogy",
    )
    gpus_per_node = mo.ui.slider(
        start=1,
        stop=8,
        value=int(v2_06_defaults["gpus_per_node"]),
        step=1,
        label="Participants per local group",
    )
    overlap_pct = mo.ui.slider(
        start=0,
        stop=80,
        value=int(v2_06_defaults["overlap_pct"]),
        step=5,
        label="Communication hidden by overlap (%)",
    )
    compression_ratio = mo.ui.slider(
        start=1,
        stop=8,
        value=int(v2_06_defaults["compression_ratio"]),
        step=1,
        label="Compression ratio",
    )
    partA_reflection = mo.ui.text_area(label="Part A reflection", placeholder="Name the binding term.", full_width=True)
    partB_reflection = mo.ui.text_area(label="Part B reflection", placeholder="Where does the preferred strategy change?", full_width=True)
    partC_reflection = mo.ui.text_area(label="Part C reflection", placeholder="What hierarchy assumption matters?", full_width=True)
    partD_reflection = mo.ui.text_area(label="Part D reflection", placeholder="What residual risk remains?", full_width=True)
    student_id = mo.ui.text(label="Student identifier", placeholder="Optional")
    decision_text = mo.ui.text_area(label="Final communication decision", placeholder="Choose a strategy and residual risk.", full_width=True)
    return (
        compression_ratio,
        decision_text,
        fabric,
        gpus_per_node,
        message_gb,
        n_gpus,
        overlap_pct,
        partA_reflection,
        partB_reflection,
        partC_reflection,
        partD_reflection,
        student_id,
    )


@app.cell(hide_code=True)
def _(
    COLORS,
    Hardware,
    Systems,
    apply_plotly_theme,
    build_lab_report,
    calc_hierarchical_allreduce_time,
    calc_ring_allreduce_time,
    calc_tree_allreduce_time,
    compression_ratio,
    decision_text,
    fabric,
    go,
    gpus_per_node,
    html_lib,
    ledger,
    message_gb,
    mo,
    n_gpus,
    np,
    overlap_pct,
    partA_prediction,
    partA_reflection,
    partB_prediction,
    partB_reflection,
    partC_prediction,
    partC_reflection,
    partD_prediction,
    partD_reflection,
    report_export_panel,
    student_id,
    ureg,
    v2_06_defaults,
    v2_06_metadata,
    v2_06_profile,
    v2_06_variant,
):
    _h100 = Hardware.Cloud.H100
    _nvlink = _h100.nvlink
    _fabric_map = {
        "ib": Systems.Fabrics.InfiniBand_NDR,
        "eth": Systems.Fabrics.Ethernet_100G,
        "nvlink": _nvlink,
    }
    _fabric_labels = {"ib": "InfiniBand NDR", "eth": "100G Ethernet", "nvlink": "NVLink-only"}

    def _fabric_values():
        active = _fabric_map[fabric.value]
        return active.bandwidth, active.latency

    def _fabric_label():
        return _fabric_labels.get(fabric.value, str(fabric.value))

    def _times(gpus=None, size_gb=None, compression=None, overlap=None):
        _gpus = max(2, int(gpus or n_gpus.value))
        _size_gb = max(0.0001, float(size_gb if size_gb is not None else message_gb.value))
        _compression = max(1, int(compression or compression_ratio.value))
        _overlap = float(overlap if overlap is not None else overlap_pct.value)
        _msg = (_size_gb / _compression) * ureg.GB
        _bw, _lat = _fabric_values()
        _ring = calc_ring_allreduce_time(_msg, _gpus, _bw, _lat).m_as("ms")
        _tree = calc_tree_allreduce_time(_msg, _gpus, _bw, _lat).m_as("ms")
        _local = max(1, int(gpus_per_node.value))
        _nodes = max(1, int(np.ceil(_gpus / _local)))
        _hierarchy = calc_hierarchical_allreduce_time(
            _msg,
            _nodes,
            _local,
            _nvlink.bandwidth,
            _bw,
            _nvlink.latency,
            _lat,
        ).m_as("ms")
        _best_algorithm, _best_ms = min(
            ("Ring", _ring),
            ("Tree", _tree),
            ("Hierarchical", _hierarchy),
            key=lambda item: item[1],
        )
        _exposed = _best_ms * (1 - _overlap / 100)
        _binding = "bandwidth" if _size_gb / _compression >= 1 else "latency/topology"
        return {
            "ring_ms": _ring,
            "tree_ms": _tree,
            "hierarchical_ms": _hierarchy,
            "best_algorithm": _best_algorithm,
            "best_ms": _best_ms,
            "exposed_best_ms": _exposed,
            "message_gb_after_compression": _size_gb / _compression,
            "binding_term": _binding,
            "failure_state": "SLO risk" if _exposed > 1000 else "feasible",
        }

    def _bar_chart(values):
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=["Ring", "Tree", "Hierarchical", "Exposed best"],
            y=[values["ring_ms"], values["tree_ms"], values["hierarchical_ms"], values["exposed_best_ms"]],
            marker_color=[COLORS["RedLine"], COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"]],
        ))
        _fig.update_layout(height=320, yaxis_title="Time (ms)", showlegend=False, margin=dict(l=60, r=20, t=30, b=40))
        return apply_plotly_theme(_fig)

    def _frontier_chart():
        _sizes = np.geomspace(0.001, 80, 40)
        _ring, _tree, _hierarchy = [], [], []
        for _size in _sizes:
            _vals = _times(size_gb=float(_size), compression=1, overlap=0)
            _ring.append(_vals["ring_ms"])
            _tree.append(_vals["tree_ms"])
            _hierarchy.append(_vals["hierarchical_ms"])
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_sizes, y=_ring, mode="lines", name="Ring"))
        _fig.add_trace(go.Scatter(x=_sizes, y=_tree, mode="lines", name="Tree"))
        _fig.add_trace(go.Scatter(x=_sizes, y=_hierarchy, mode="lines", name="Hierarchical"))
        _fig.update_layout(
            height=340,
            xaxis_title="Message/payload per participant (GB)",
            yaxis_title="Time (ms)",
            xaxis_type="log",
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=60, r=20, t=50, b=40),
        )
        return apply_plotly_theme(_fig)

    def _reading_guide(title, steps, interpretation):
        def _inline_markup(text):
            _parts = html_lib.escape(text).split("**")
            return "".join(f"<strong>{_part}</strong>" if _idx % 2 else _part for _idx, _part in enumerate(_parts))

        _items = "".join(f"<li>{_inline_markup(_step)}</li>" for _step in steps)
        return mo.Html(f"""
        <div class="mlsysbook-callout">
          <strong>{html_lib.escape(title)}</strong>
          <ol style="margin:8px 0 8px 20px; padding:0; line-height:1.55;">{_items}</ol>
          <div><strong>How to read the result:</strong> {_inline_markup(interpretation)}</div>
        </div>
        """)

    def build_part_a():
        _vals = _times()
        _items = [
            mo.md("## Part A - Operation Anatomy"),
            mo.callout(mo.md(
                f"{v2_06_variant.stakeholder}: {v2_06_variant.objective}"
            ), kind="info"),
            partA_prediction,
        ]
        if partA_prediction.value is None:
            _items.append(mo.callout(mo.md("Select a prediction to unlock the collective/coordination instruments."), kind="warn"))
            return mo.vstack(_items)
        _items.extend([
            mo.callout(
                mo.md("Correct: bytes, rounds, participants, and topology control the modeled cost." if partA_prediction.value == "bytes_topology" else
                      "The trap is treating communication as one opaque overhead. The cost depends on bytes, rounds, participants, and topology."),
                kind="success" if partA_prediction.value == "bytes_topology" else "warn",
            ),
            _reading_guide(
                "Start from one source-traced scenario",
                (
                    f"Use the selected track defaults: **{v2_06_defaults['participants']} participants** and **{v2_06_defaults['message_gb']} GB**.",
                    "Change the fabric/link analogy first to isolate topology.",
                    "Then change payload size to see whether bytes or latency dominates.",
                ),
                "the shortest bar is the fastest strategy for the selected point; the gap is the penalty for choosing the wrong schedule.",
            ),
            mo.hstack([n_gpus, message_gb, fabric], justify="start"),
            mo.as_html(_bar_chart(_vals)),
            mo.md(
                f"Current best: **{_vals['best_algorithm']}** at **{_vals['best_ms']:.2f} ms** "
                f"before overlap. Binding term: **{_vals['binding_term']}**. "
                f"Failure state: **{_vals['failure_state']}**."
            ),
            partA_reflection,
        ])
        return mo.vstack(_items)

    def build_part_b():
        _items = [
            mo.md("## Part B - Algorithm And Topology Frontier"),
            partB_prediction,
        ]
        if partB_prediction.value is None:
            _items.append(mo.callout(mo.md("Select a prediction before comparing the frontier."), kind="warn"))
            return mo.vstack(_items)
        _items.extend([
            mo.callout(
                mo.md("Correct: the winner moves with message size, participant count, and topology." if partB_prediction.value == "depends" else
                      "No single algorithm dominates every point; read the curve crossings."),
                kind="success" if partB_prediction.value == "depends" else "warn",
            ),
            _reading_guide(
                "Read the frontier as a decision boundary",
                (
                    "Keep participants and fabric fixed.",
                    "Scan left to right across message sizes.",
                    "Look for curve crossings; a crossing means the preferred strategy changes.",
                ),
                "lower curves are better; a flat left side usually exposes latency rounds, while the right side exposes bytes and bandwidth.",
            ),
            mo.hstack([n_gpus, fabric, gpus_per_node], justify="start"),
            mo.as_html(_frontier_chart()),
            partB_reflection,
        ])
        return mo.vstack(_items)

    def build_part_c():
        _vals = _times()
        _gain = _vals["ring_ms"] / _vals["hierarchical_ms"] if _vals["hierarchical_ms"] else 0.0
        _items = [
            mo.md("## Part C - Hierarchy As A Systems Decision"),
            partC_prediction,
        ]
        if partC_prediction.value is None:
            _items.append(mo.callout(mo.md("Select a prediction before inspecting hierarchy."), kind="warn"))
            return mo.vstack(_items)
        _items.extend([
            mo.callout(
                mo.md("Correct: hierarchy helps when local communication is materially cheaper than global communication." if partC_prediction.value == "hierarchy" else
                      "Hierarchy is an assumption about topology. It is useful only when the system has a faster local tier."),
                kind="success" if partC_prediction.value == "hierarchy" else "warn",
            ),
            _reading_guide(
                "Separate local and global communication",
                (
                    "Use the local group slider as the hierarchy assumption.",
                    "If local groups shrink to one participant, hierarchy loses its advantage.",
                    "If the global link is slow, hierarchy can reduce exposed global traffic.",
                ),
                "hierarchical speedup is evidence only if the selected track really has the assumed local/global tiers.",
            ),
            mo.hstack([n_gpus, gpus_per_node, fabric], justify="start"),
            mo.callout(mo.md(f"Hierarchy is **{_gain:.2f}x** the flat-ring time at this point."), kind="success" if _gain > 1 else "warn"),
            mo.as_html(_bar_chart(_vals)),
            partC_reflection,
        ])
        return mo.vstack(_items)

    def build_part_d():
        _vals = _times()
        _items = [
            mo.md("## Part D - Overlap, Compression, And Residual Risk"),
            partD_prediction,
        ]
        if partD_prediction.value is None:
            _items.append(mo.callout(mo.md("Select a prediction before using overlap/compression controls."), kind="warn"))
            return mo.vstack(_items)
        _items.extend([
            mo.callout(
                mo.md("Correct: compression and overlap reduce exposed time, but the design still needs validation." if partD_prediction.value == "risk" else
                      "Optimization is a claim. Compression and overlap must be validated against the track guardrail."),
                kind="success" if partD_prediction.value == "risk" else "warn",
            ),
            _reading_guide(
                "Treat optimization knobs as claims",
                (
                    "Compression changes bytes but can lose fidelity or convergence quality.",
                    "Overlap hides time only when useful work can run at the same time.",
                    "A design review must name the residual risk.",
                ),
                "the exposed-best bar is not free communication; it is the communication left visible after the overlap assumption.",
            ),
            mo.hstack([overlap_pct, compression_ratio], justify="start"),
            mo.callout(
                mo.md(
                    f"Compressed payload: **{_vals['message_gb_after_compression']:.4g} GB**. "
                    f"Residual risk: {v2_06_defaults['residual_risk']}"
                ),
                kind="warn",
            ),
            mo.as_html(_bar_chart(_vals)),
            partD_reflection,
        ])
        return mo.vstack(_items)

    def build_synthesis():
        _vals = _times()
        _decision = decision_text.value or f"Use {_vals['best_algorithm']} and validate {v2_06_defaults['residual_risk']}"
        _snapshot = {
            "track_id": v2_06_profile.track_id,
            "scenario_id": v2_06_variant.scenario_id,
            "operation": v2_06_defaults["operation"],
            "participants": n_gpus.value,
            "message_gb": message_gb.value,
            "fabric": _fabric_label(),
            "gpus_per_node": gpus_per_node.value,
            "overlap_pct": overlap_pct.value,
            "compression_ratio": compression_ratio.value,
            **_vals,
        }
        ledger.save(
            chapter=6,
            design={
                "lab_id": v2_06_metadata.lab_id,
                "track_id": v2_06_profile.track_id,
                "scenario_id": v2_06_variant.scenario_id,
                "decision": _decision,
                "binding_constraint": _vals["binding_term"],
                "result_snapshot": _snapshot,
            },
        )
        _incomplete = []
        if partA_prediction.value is None:
            _incomplete.append("Part A operation anatomy prediction")
        if partB_prediction.value is None:
            _incomplete.append("Part B algorithm frontier prediction")
        if partC_prediction.value is None:
            _incomplete.append("Part C hierarchy prediction")
        if partD_prediction.value is None:
            _incomplete.append("Part D optimization risk prediction")

        _report = build_lab_report(
            v2_06_metadata,
            student_id=student_id.value or "",
            track=v2_06_profile.label,
            scenario=v2_06_variant.workload_summary,
            learning_objectives=(
                "Decompose collective communication into bytes, rounds, participants, and topology.",
                "Compare ring, tree, and hierarchical strategies across a message-size frontier.",
                "Choose compression/overlap with a named residual risk.",
            ),
            predictions={
                "operation_anatomy": partA_prediction.value,
                "algorithm_frontier": partB_prediction.value,
                "hierarchy": partC_prediction.value,
                "overlap_compression": partD_prediction.value,
            },
            knob_settings={
                "participants": n_gpus.value,
                "message_gb": message_gb.value,
                "fabric": _fabric_label(),
                "gpus_per_node": gpus_per_node.value,
                "overlap_pct": overlap_pct.value,
                "compression_ratio": compression_ratio.value,
            },
            binding_constraints={
                "best_algorithm": _vals["best_algorithm"],
                "best_ms": round(_vals["best_ms"], 4),
                "binding_term": _vals["binding_term"],
                "failure_state": _vals["failure_state"],
            },
            evidence_summary={
                "best_algorithm": _vals["best_algorithm"],
                "best_ms": round(_vals["best_ms"], 4),
                "binding_term": _vals["binding_term"],
                "failure_state": _vals["failure_state"],
                "fabric": _fabric_label(),
                "participants": n_gpus.value,
                "message_gb": message_gb.value,
                "overlap_pct": overlap_pct.value,
                "compression_ratio": compression_ratio.value,
            },
            decisions={"communication_strategy": _decision},
            reflections={
                "part_a_operation_anatomy": partA_reflection.value,
                "part_b_algorithm_frontier": partB_reflection.value,
                "part_c_hierarchy": partC_reflection.value,
                "part_d_overlap_compression": partD_reflection.value,
            },
            final_decision=_decision,
            big_takeaways=(
                "Communication strategy depends on message size, participants, fabric, and topology.",
                "Hierarchy is useful only when local and global communication tiers really differ.",
                "Compression and overlap reduce exposed time but must be validated against residual risk.",
            ),
            residual_risk=v2_06_defaults["residual_risk"],
            source_trace={
                "track_id": v2_06_profile.track_id,
                "scenario_id": v2_06_variant.scenario_id,
                "hardware_ref": v2_06_variant.hardware_ref,
                "model_ref": v2_06_variant.model_ref,
                "shared_solver": "mlsysim.physics collective communication",
                "source_policy": v2_06_profile.source_policy,
            },
            result_snapshot=_snapshot,
            incomplete_fields=tuple(_incomplete),
        )
        return mo.vstack([
            mo.md("## Synthesis - Communication Design Review"),
            student_id,
            decision_text,
            mo.callout(mo.md(f"Recommended strategy: **{_vals['best_algorithm']}**. Binding term: **{_vals['binding_term']}**."), kind="success"),
            report_export_panel(_report),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Operation Anatomy": build_part_a(),
        "Part B: Algorithm Frontier": build_part_b(),
        "Part C: Hierarchy": build_part_c(),
        "Part D: Overlap And Compression": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


@app.cell(hide_code=True)
def _(mo, v2_06_metadata, v2_06_profile):
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">{v2_06_metadata.lab_id}</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v2_06_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
