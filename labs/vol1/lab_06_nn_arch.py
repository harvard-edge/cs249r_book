import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ===========================================================================
# ZONE A: SETUP
# ===========================================================================


@app.cell
async def _():
    import html
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
    from mlsysim.labs.components import MathPeek
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        architecture_decision,
        architecture_scaling_curve,
        architecture_signature,
        architecture_track_profile,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
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
        MathPeek,
        apply_plotly_theme,
        architecture_decision,
        architecture_scaling_curve,
        architecture_signature,
        architecture_track_profile,
        build_lab_report,
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
        track_context,
        track_arc_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_06_metadata = get_lab_metadata("vol1/lab_06_nn_arch.py")
    return (v1_06_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_06_track_picker = track_selector(default=_default_track)
    v1_06_track_picker
    return (v1_06_track_picker,)


@app.cell
def _(
    architecture_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_06_track_picker,
):
    v1_06_track_id = v1_06_track_picker.value
    v1_06_profile = get_track_profile(v1_06_track_id)
    v1_06_variant = get_lab_track_variant("v1_06_architecture_tax", v1_06_profile.track_id)
    v1_06_hardware = resolve_mlsysim_ref(v1_06_variant.hardware_ref)
    v1_06_model = resolve_mlsysim_ref(v1_06_variant.model_ref)
    v1_06_architecture = architecture_track_profile(
        v1_06_profile,
        v1_06_variant,
        v1_06_hardware,
        v1_06_model,
    )
    return (
        v1_06_architecture,
        v1_06_hardware,
        v1_06_model,
        v1_06_profile,
        v1_06_track_id,
        v1_06_variant,
    )


@app.cell
def _(COLORS, html):
    def v1_06_e(value):
        return html.escape(str(value))

    def v1_06_fields_html(items):
        return "".join(
            f'<div class="mlsysbook-field"><strong>{v1_06_e(label)}</strong>{v1_06_e(value)}</div>'
            for label, value in items.items()
        )

    def v1_06_table_html(headers, rows, *, numeric=()):
        _head = "".join(f"<th>{v1_06_e(header)}</th>" for header in headers)
        _body = []
        for row in rows:
            _cells = []
            for idx, value in enumerate(row):
                _style = " style='text-align:right; font-variant-numeric: tabular-nums;'" if idx in numeric else ""
                _cells.append(f"<td{_style}>{v1_06_e(value)}</td>")
            _body.append(f"<tr>{''.join(_cells)}</tr>")
        return f"""
        <table class="mlsysbook-table">
          <thead><tr>{_head}</tr></thead>
          <tbody>{''.join(_body)}</tbody>
        </table>
        """

    def v1_06_callout_html(title, message, *, kind="info"):
        _colors = {
            "ok": COLORS["GreenLine"],
            "warn": COLORS["OrangeLine"],
            "fail": COLORS["RedLine"],
            "info": COLORS["BlueLine"],
        }
        _color = _colors.get(kind, COLORS["BlueLine"])
        return f"""
        <div class="mlsysbook-panel" style="border-left:4px solid {_color};">
          <div class="mlsysbook-section-label">{v1_06_e(title)}</div>
          <p style="margin:0; line-height:1.6; color:{COLORS['TextSec']};">{v1_06_e(message)}</p>
        </div>
        """

    def v1_06_prediction_html(title, predicted, actual, labels):
        if predicted is None:
            return v1_06_callout_html(
                title,
                "Commit to a structured prediction before treating the evidence as a design answer.",
                kind="info",
            )
        _same = predicted == actual
        _message = (
            f"You predicted {labels.get(predicted, predicted)}. "
            f"The instrument points to {labels.get(actual, actual)}."
        )
        if _same:
            _message += " The prediction matches the measured constraint."
        else:
            _message += " Use the gap to revise the architecture recommendation."
        return v1_06_callout_html(title, _message, kind="ok" if _same else "warn")

    def v1_06_track_amount_system(profile):
        _systems = {
            "iphone": {
                "amounts": "latency, activation memory, energy, and supported NPU kernels",
                "stake": "local vision/audio UX, privacy, battery, and thermal headroom",
                "failure": "NPU fallback, thermal throttle, or user-visible latency",
                "bias": "local spatial/audio bias is valuable only if it stays on supported kernels",
                "mitigation": "reduce resolution/context, choose mobile-local topology, or prove kernels stay on accelerator",
            },
            "oura_ring": {
                "amounts": "SRAM, flash image size, wake time, duty cycle, and signal quality",
                "stake": "tiny sequence/signal inference without draining the ring battery",
                "failure": "SRAM overflow or duty-cycle violation",
                "bias": "temporal locality is valuable only when state remains SRAM-resident",
                "mitigation": "shorten the window, stream state, or reject attention state that spills from SRAM",
            },
            "robotaxi": {
                "amounts": "p99 latency, activation memory, power, sensor burst margin, and rare-event recall",
                "stake": "vehicle-local perception with a safety case",
                "failure": "tail-latency miss or rare-event safety margin miss",
                "bias": "perception locality is valuable only when rare-event replay and p99 deadlines pass",
                "mitigation": "keep the bounded detector, split global context behind fallback, or reduce sensor scale",
            },
            "cloud_fleet": {
                "amounts": "HBM memory, p99 SLA, utilization, cost/request, and quality",
                "stake": "SLA-compliant transformer service economics",
                "failure": "context/KV memory pressure, queue/SLA breach, or negative cost/request",
                "bias": "flexible attention is valuable only while memory and batching economics hold",
                "mitigation": "cap context, use memory-aware attention, rebatch traffic, or select a smaller serving architecture",
            },
        }
        return _systems.get(profile.track_id, _systems["iphone"])

    def v1_06_family_topology(family):
        _family = family.lower()
        if "transformer" in _family or "vit" in _family:
            return ("all-to-all attention", "gather/reduce over tokens", "low locality")
        if "rnn" in _family or "temporal" in _family:
            return ("recurrent or causal sequence", "streaming state reuse", "state locality")
        if "cnn" in _family or "detector" in _family:
            return ("sliding-window convolution", "spatial reuse and tiling", "high locality")
        if "sparse" in _family or "moe" in _family:
            return ("sparse routing", "scatter/gather across experts", "data-dependent locality")
        return ("dense feed-forward", "GEMM and activation traffic", "batch locality")

    def v1_06_budget_ratios(profile, item):
        return {
            "activation memory": item.activation_mb / max(profile.memory_budget_mb, 1e-9),
            "latency": item.latency_ms / max(profile.latency_budget_ms, 1e-9),
            "power": item.power_w / max(profile.power_budget_w, 1e-9),
            "quality guardrail": profile.quality_floor_pct / max(item.quality_pct, 1e-9),
            "kernel support": profile.kernel_support_floor_pct / max(item.kernel_support_pct, 1e-9),
        }

    def v1_06_topology_summary(profile, signature):
        _rows = []
        for item in signature:
            _topology, _access, _locality = v1_06_family_topology(item.family)
            _ops_per_mb = item.ops_gmac / max(item.activation_mb, 1e-9)
            _ratios = v1_06_budget_ratios(profile, item)
            _rows.append({
                "id": item.architecture_id,
                "label": item.label,
                "family": item.family,
                "topology": _topology,
                "access": _access,
                "locality": _locality,
                "ops_per_mb": _ops_per_mb,
                "memory_pct": 100.0 * _ratios["activation memory"],
                "latency_pct": 100.0 * _ratios["latency"],
                "power_pct": 100.0 * _ratios["power"],
                "dominant": item.dominant_constraint,
                "feasible": item.feasible,
            })
        _best = max(signature, key=lambda item: item.score if item.feasible else item.score - 100.0)
        _topology, _access, _locality = v1_06_family_topology(_best.family)
        if _best.dominant_constraint == "kernel support":
            _actual = "kernel"
        elif _best.dominant_constraint == "quality guardrail":
            _actual = "quality"
        elif "attention" in _topology or "transformer" in _best.family.lower() or "vit" in _best.family.lower():
            _actual = "attention"
        else:
            _actual = "local"
        return {
            "rows": tuple(_rows),
            "actual": _actual,
            "best_label": _best.label,
            "best_topology": _topology,
            "best_access": _access,
            "best_locality": _locality,
            "best_constraint": _best.dominant_constraint,
        }

    def v1_06_attention_wall(profile, signature, curve, scale_value):
        _candidate_by_id = {candidate.architecture_id: candidate for candidate in profile.candidates}
        _attention_items = [
            item for item in signature
            if "transformer" in item.family.lower() or "vit" in item.family.lower()
        ]
        if not _attention_items:
            _attention_items = [max(signature, key=lambda item: _candidate_by_id[item.architecture_id].activation_exponent)]
        _attention = max(
            _attention_items,
            key=lambda item: _candidate_by_id[item.architecture_id].activation_exponent,
        )
        _ratios = v1_06_budget_ratios(profile, _attention)
        _amount = max(
            ("activation memory", "latency", "power", "quality guardrail", "kernel support"),
            key=lambda key: _ratios[key],
        )
        _actual = "memory" if _amount == "activation memory" else "latency" if _amount == "latency" else "power" if _amount == "power" else "quality_kernel"
        _first_failure = curve.first_failure_by_candidate.get(_attention.architecture_id)
        _first_failure_text = (
            f"{_first_failure:.0f} {profile.scaling_unit}" if _first_failure is not None else "not reached in sweep"
        )
        _system = v1_06_track_amount_system(profile)
        _status = "fail" if not _attention.feasible else "warn" if max(_ratios.values()) > 0.8 else "ok"
        _message = (
            f"{_attention.label} is governed by {_amount}: "
            f"activation memory is {_attention.activation_mb:.2f} MB against {profile.memory_budget_mb:.2f} MB, "
            f"latency is {_attention.latency_ms:.2f} ms against {profile.latency_budget_ms:.2f} ms, "
            f"and the first infeasible scale is {_first_failure_text}. "
            f"Mitigation: {_system['mitigation']}."
        )
        return {
            "attention_id": _attention.architecture_id,
            "attention_label": _attention.label,
            "dominant_amount": _amount,
            "actual": _actual,
            "first_failure": _first_failure_text,
            "status": _status,
            "message": _message,
            "scale_value": scale_value,
            "memory_pct": 100.0 * _ratios["activation memory"],
            "latency_pct": 100.0 * _ratios["latency"],
            "power_pct": 100.0 * _ratios["power"],
        }

    def v1_06_bias_match(profile, item):
        _family = item.family.lower()
        if profile.track_id == "cloud_fleet":
            if "transformer" in _family:
                return 1.15
            if "cnn" in _family:
                return 0.68
            return 1.0
        if profile.track_id == "robotaxi":
            if "cnn" in _family or "detector" in _family:
                return 1.12
            if "hybrid" in _family:
                return 0.96
            if "transformer" in _family:
                return 0.78
            return 0.9
        if profile.track_id == "oura_ring":
            if "temporal" in _family or "cnn" in _family or "depthwise" in _family:
                return 1.18
            if "transformer" in _family:
                return 0.64
            return 0.9
        if "cnn" in _family:
            return 1.15
        if "transformer" in _family and "efficient" in item.label.lower():
            return 0.92
        if "transformer" in _family or "vit" in _family:
            return 0.75
        return 0.9

    def v1_06_bias_frontier(profile, signature, data_pressure):
        _rows = []
        for item in signature:
            _match = v1_06_bias_match(profile, item)
            _ratios = v1_06_budget_ratios(profile, item)
            _resource_pressure = max(_ratios["activation memory"], _ratios["latency"], _ratios["power"])
            _data_need = 100.0 * float(data_pressure) / max(_match, 1e-9)
            _deployability = max(
                0.0,
                100.0
                - 40.0 * max(0.0, _resource_pressure - 1.0)
                - 0.20 * max(0.0, _data_need - 100.0)
                - 0.60 * max(0.0, profile.quality_floor_pct - item.quality_pct)
                - 0.50 * max(0.0, profile.kernel_support_floor_pct - item.kernel_support_pct),
            )
            _score = item.quality_pct + 0.25 * _deployability - 0.08 * _data_need
            _rows.append({
                "id": item.architecture_id,
                "label": item.label,
                "family": item.family,
                "bias_match": _match,
                "data_need": _data_need,
                "quality": item.quality_pct,
                "deployability": _deployability,
                "feasible": item.feasible,
                "dominant": item.dominant_constraint,
                "score": _score if item.feasible else _score - 75.0,
            })
        _best = max(_rows, key=lambda row: row["score"])
        _leaderboard = max(signature, key=lambda item: item.quality_pct)
        _best_family = _best["family"].lower()
        if _best["id"] == _leaderboard.architecture_id:
            _actual = "leaderboard"
        elif "transformer" in _best_family and profile.track_id == "cloud_fleet":
            _actual = "flexible_attention"
        elif "cnn" in _best_family or "temporal" in _best_family or "detector" in _best_family:
            _actual = "local_bias"
        else:
            _actual = "track_constraint"
        return {
            "rows": tuple(_rows),
            "best": _best,
            "leaderboard_label": _leaderboard.label,
            "actual": _actual,
        }

    def v1_06_review_summary(profile, signature, decision):
        _selected = next(item for item in signature if item.architecture_id == decision.selected_id)
        _leaderboard = max(signature, key=lambda item: item.quality_pct)
        _ratios = v1_06_budget_ratios(profile, _selected)
        _headroom = min(1.0 / max(_ratios["activation memory"], 1e-9), 1.0 / max(_ratios["latency"], 1e-9), 1.0 / max(_ratios["power"], 1e-9))
        _feasible_items = [item for item in signature if item.feasible]
        _headroom_pick = max(
            _feasible_items or list(signature),
            key=lambda item: min(
                1.0 / max(v1_06_budget_ratios(profile, item)["activation memory"], 1e-9),
                1.0 / max(v1_06_budget_ratios(profile, item)["latency"], 1e-9),
                1.0 / max(v1_06_budget_ratios(profile, item)["power"], 1e-9),
            ),
        )
        if not _selected.feasible:
            _status = "reject"
            _kind = "fail"
            _actual = "guardrail"
            _message = f"Reject {_selected.label}: {', '.join(_selected.violations) or _selected.dominant_constraint}."
        elif _selected.architecture_id == _headroom_pick.architecture_id:
            _status = "approve"
            _kind = "ok"
            _actual = "headroom"
            _message = f"Approve {_selected.label}: it has the strongest track headroom among feasible candidates."
        elif _selected.architecture_id == _leaderboard.architecture_id:
            _status = "approve with mitigation"
            _kind = "warn"
            _actual = "leaderboard"
            _message = f"Approve {_selected.label} only with validation because the leaderboard choice still carries {_selected.dominant_constraint} risk."
        else:
            _status = "approve with mitigation"
            _kind = "warn"
            _actual = "guardrail"
            _message = f"Approve {_selected.label} with mitigation: {_selected.dominant_constraint} is closest to the guardrail."
        return {
            "selected": _selected,
            "leaderboard": _leaderboard,
            "headroom_pick": _headroom_pick,
            "headroom": _headroom,
            "status": _status,
            "kind": _kind,
            "actual": _actual,
            "message": _message,
        }

    return (
        v1_06_attention_wall,
        v1_06_bias_frontier,
        v1_06_callout_html,
        v1_06_fields_html,
        v1_06_prediction_html,
        v1_06_review_summary,
        v1_06_table_html,
        v1_06_topology_summary,
        v1_06_track_amount_system,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    track_arc_context,
    v1_06_architecture,
    v1_06_metadata,
    v1_06_profile,
    v1_06_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 06
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Architecture Tax
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Inductive Bias &middot; Scaling Shape &middot; Kernel Support &middot; Guardrails
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 860px; line-height: 1.65;">
                {v1_06_variant.workload_summary} The goal is not to crown one universal
                architecture; it is to choose the family whose resource signature matches
                the selected track. Architecture choices create different resource
                shapes; inductive bias and scaling laws determine which amount grows first.
                Every track follows the same four concepts; the track changes the persona,
                thresholds, evidence emphasis, failure mode, and report framing.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Concept Modules + Synthesis &middot; ~50 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_06_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_06_architecture.scaling_variable}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Topology Locality</span>
                <span class="badge badge-warn">Attention Memory Wall</span>
                <span class="badge badge-info">Inductive Bias</span>
                <span class="badge badge-fail">Deployment Memo</span>
            </div>
        </div>
        """),
        track_context(v1_06_profile),
        track_arc_context(v1_06_profile, v1_06_metadata.lab_id),
        source_trace({
            "chapter": "vol1/nn_architectures/nn_architectures.qmd",
            "anchors": (
                "Architectural Principles",
                "CNNs: Spatial Pattern Processing",
                "RNNs: Sequential Pattern Processing",
                "Attention: Dynamic Processing",
                "Transformers: Parallel Sequence Processing",
                "Architecture Selection Framework",
                "Fallacies and Pitfalls",
            ),
            "shared_helper": "mlsysbook_labs.architecture",
            "scenario_id": v1_06_variant.scenario_id,
        }, summary="Opening source map"),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_06_architecture):
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
            <div style="margin-bottom: 3px;">1. <strong>Compare architecture signatures:</strong>
                parameters, operations, activation memory, latency, power, quality, and kernel support.</div>
            <div style="margin-bottom: 3px;">2. <strong>Predict scaling failure:</strong>
                sweep {v1_06_architecture.scaling_variable} and identify the first architecture family that breaks.</div>
            <div style="margin-bottom: 3px;">3. <strong>Reason about inductive bias:</strong>
                explain how locality, recurrence, attention, or routing changes data need and deployability.</div>
            <div style="margin-bottom: 3px;">4. <strong>Defend an architecture:</strong>
                recommend one family, reject alternatives, and state the validation requirement.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                Which architecture family fits {v1_06_architecture.label}, and what failure
                appears next as {v1_06_architecture.scaling_variable} grows?
            </div>
        </div>
    </div>
    """)
    return


# ===========================================================================
# ZONE B: CONTROLS AND COMPUTATION
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_06_architecture):
    v1_06_failure_prediction = mo.ui.radio(
        options={
            "Local convolution or streaming state will best match the amount system": "local",
            "Attention/token mixing will best match the amount system": "attention",
            "Kernel support or dispatch will decide before the topology": "kernel",
            "Quality guardrail will decide before resource budgets": "quality",
        },
        label=f"Part A prediction: which topology best matches {v1_06_architecture.label}?",
    )
    v1_06_topology_checkpoint = mo.ui.radio(
        options={
            "Choose the topology with the best locality/headroom": "locality_headroom",
            "Choose the topology with the best quality proxy": "quality_proxy",
            "Defer until kernel and memory profiling is complete": "profile_first",
            "Reject the current candidate set": "reject_set",
        },
        label="Part A checkpoint: what topology decision follows from the signature?",
    )
    return (v1_06_failure_prediction, v1_06_topology_checkpoint)


@app.cell(hide_code=True)
def _(mo, v1_06_architecture):
    v1_06_scale = mo.ui.slider(
        start=v1_06_architecture.scale_min,
        stop=v1_06_architecture.scale_max,
        value=v1_06_architecture.default_scale,
        step=v1_06_architecture.scale_step,
        label=f"{v1_06_architecture.scaling_variable} ({v1_06_architecture.scaling_unit})",
    )
    v1_06_scale
    return (v1_06_scale,)


@app.cell(hide_code=True)
def _(mo):
    v1_06_memory_prediction = mo.ui.radio(
        options={
            "Activation/KV/state memory becomes the hidden wall": "memory",
            "Latency crosses the budget before memory": "latency",
            "Power or duty cycle crosses first": "power",
            "Quality or kernel support fails before scale does": "quality_kernel",
        },
        label="Part B prediction: as the workload scale grows, which amount fails first?",
    )
    v1_06_scaling_checkpoint = mo.ui.radio(
        options={
            "Shorten the context/window/resolution": "shorten_scale",
            "Keep the local or streaming topology": "local_topology",
            "Require memory-aware attention kernels": "memory_kernel",
            "Escalate to a larger deployment envelope": "larger_envelope",
        },
        label="Part B checkpoint: what mitigation would you defend after seeing the wall?",
    )
    return (v1_06_memory_prediction, v1_06_scaling_checkpoint)


@app.cell(hide_code=True)
def _(mo):
    v1_06_bias_prediction = mo.ui.radio(
        options={
            "Structured local bias wins by reducing data and state": "local_bias",
            "Flexible attention wins because quality outweighs resource cost": "flexible_attention",
            "The leaderboard-quality model wins": "leaderboard",
            "The answer changes with the track constraint": "track_constraint",
        },
        label="Part C prediction: which inductive-bias trade-off survives deployment?",
    )
    v1_06_data_pressure = mo.ui.slider(
        start=0.5,
        stop=2.0,
        value=1.0,
        step=0.1,
        label="Data / coverage pressure multiplier",
    )
    v1_06_bias_checkpoint = mo.ui.radio(
        options={
            "Defend the structured-bias architecture": "structured_bias",
            "Defend the flexible attention architecture": "flexible_attention",
            "Collect more data before changing architecture": "more_data",
            "Reject the bias because deployment evidence fails": "reject_bias",
        },
        label="Part C checkpoint: which bias decision would you record?",
    )
    return (v1_06_bias_checkpoint, v1_06_bias_prediction, v1_06_data_pressure)


@app.cell(hide_code=True)
def _(mo, v1_06_architecture):
    _architecture_options = {
        candidate.label: candidate.architecture_id
        for candidate in v1_06_architecture.candidates
    }
    v1_06_review_prediction = mo.ui.radio(
        options={
            "Approve the highest-quality architecture": "leaderboard",
            "Approve the feasible architecture with the most headroom": "headroom",
            "Approve the smallest architecture": "smallest",
            "Let the track guardrail decide": "guardrail",
        },
        label="Part D prediction: which review rule should approve the architecture?",
    )
    v1_06_arch_choice = mo.ui.dropdown(
        options=_architecture_options,
        value=v1_06_architecture.candidates[0].label,
        label="Architecture recommendation",
    )
    v1_06_deployment_checkpoint = mo.ui.radio(
        options={
            "Approve": "approve",
            "Approve only with mitigation": "mitigate",
            "Reject and redesign": "reject",
        },
        label="Part D checkpoint: what should the review record?",
    )
    v1_06_reflection = mo.ui.text_area(
        label="Synthesis memo",
        placeholder="Recommendation, rejected alternatives, measured evidence, residual risk, and validation requirement.",
        full_width=True,
    )
    return (
        v1_06_arch_choice,
        v1_06_deployment_checkpoint,
        v1_06_reflection,
        v1_06_review_prediction,
    )


@app.cell
def _(
    architecture_decision,
    architecture_scaling_curve,
    architecture_signature,
    v1_06_attention_wall,
    v1_06_arch_choice,
    v1_06_architecture,
    v1_06_bias_frontier,
    v1_06_data_pressure,
    v1_06_review_summary,
    v1_06_scale,
    v1_06_topology_summary,
):
    v1_06_signature = architecture_signature(
        v1_06_architecture,
        scale_value=v1_06_scale.value,
    )
    v1_06_curve = architecture_scaling_curve(v1_06_architecture, samples=36)
    v1_06_decision = architecture_decision(
        v1_06_architecture,
        architecture_id=v1_06_arch_choice.value,
        scale_value=v1_06_scale.value,
    )
    v1_06_selected_eval = next(
        item for item in v1_06_signature
        if item.architecture_id == v1_06_decision.selected_id
    )
    v1_06_topology = v1_06_topology_summary(v1_06_architecture, v1_06_signature)
    v1_06_attention = v1_06_attention_wall(
        v1_06_architecture,
        v1_06_signature,
        v1_06_curve,
        v1_06_scale.value,
    )
    v1_06_bias = v1_06_bias_frontier(
        v1_06_architecture,
        v1_06_signature,
        v1_06_data_pressure.value,
    )
    v1_06_review = v1_06_review_summary(
        v1_06_architecture,
        v1_06_signature,
        v1_06_decision,
    )
    return (
        v1_06_attention,
        v1_06_bias,
        v1_06_curve,
        v1_06_decision,
        v1_06_review,
        v1_06_selected_eval,
        v1_06_signature,
        v1_06_topology,
    )


# ===========================================================================
# ZONE C: PARTS
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    MathPeek,
    apply_plotly_theme,
    go,
    mo,
    source_trace,
    v1_06_arch_choice,
    v1_06_architecture,
    v1_06_attention,
    v1_06_bias,
    v1_06_bias_checkpoint,
    v1_06_bias_prediction,
    v1_06_curve,
    v1_06_data_pressure,
    v1_06_decision,
    v1_06_deployment_checkpoint,
    v1_06_failure_prediction,
    v1_06_memory_prediction,
    v1_06_prediction_html,
    v1_06_review,
    v1_06_review_prediction,
    v1_06_scale,
    v1_06_scaling_checkpoint,
    v1_06_selected_eval,
    v1_06_signature,
    v1_06_table_html,
    v1_06_fields_html,
    v1_06_callout_html,
    v1_06_topology,
    v1_06_topology_checkpoint,
    v1_06_track_amount_system,
):
    _labels = [item.label for item in v1_06_signature]
    _signature_fig = go.Figure()
    _signature_fig.add_trace(go.Bar(
        x=_labels,
        y=[100.0 * item.activation_mb / max(v1_06_architecture.memory_budget_mb, 1e-9) for item in v1_06_signature],
        name="Activation memory",
        marker_color=COLORS["BlueLine"],
        text=[f"{100.0 * item.activation_mb / max(v1_06_architecture.memory_budget_mb, 1e-9):.0f}%" for item in v1_06_signature],
        textposition="outside",
    ))
    _signature_fig.add_trace(go.Bar(
        x=_labels,
        y=[100.0 * item.latency_ms / max(v1_06_architecture.latency_budget_ms, 1e-9) for item in v1_06_signature],
        name="Latency",
        marker_color=COLORS["OrangeLine"],
        text=[f"{100.0 * item.latency_ms / max(v1_06_architecture.latency_budget_ms, 1e-9):.0f}%" for item in v1_06_signature],
        textposition="outside",
    ))
    _signature_fig.add_trace(go.Bar(
        x=_labels,
        y=[100.0 * item.power_w / max(v1_06_architecture.power_budget_w, 1e-9) for item in v1_06_signature],
        name="Power / duty cycle",
        marker_color=COLORS["GreenLine"],
        text=[f"{100.0 * item.power_w / max(v1_06_architecture.power_budget_w, 1e-9):.0f}%" for item in v1_06_signature],
        textposition="outside",
    ))
    _signature_fig.add_hline(
        y=100,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=1.5,
        annotation_text="budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _signature_fig.update_layout(
        barmode="group",
        height=360,
        xaxis=dict(title="Candidate architecture", gridcolor="#f1f5f9"),
        yaxis=dict(title="% of track budget", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=80),
    )
    apply_plotly_theme(_signature_fig)

    _signature_rows = tuple(
        (
            item.label,
            item.family,
            f"{item.params_m:.2f}M",
            f"{item.ops_gmac:.2f}",
            f"{item.activation_mb:.2f} MB / {v1_06_architecture.memory_budget_mb:.2f}",
            f"{item.latency_ms:.2f} ms / {v1_06_architecture.latency_budget_ms:.2f}",
            f"{item.power_w:.3f} W / {v1_06_architecture.power_budget_w:.3f}",
            f"{item.quality_pct:.1f}% / {v1_06_architecture.quality_floor_pct:.1f}%",
            f"{item.kernel_support_pct:.1f}% / {v1_06_architecture.kernel_support_floor_pct:.1f}%",
            "yes" if item.feasible else "no - violation",
            item.dominant_constraint,
        )
        for item in v1_06_signature
    )

    _topology_rows = tuple(
        (
            row["label"],
            row["topology"],
            row["access"],
            row["locality"],
            f"{row['ops_per_mb']:.3f}",
            f"{row['memory_pct']:.1f}%",
            f"{row['latency_pct']:.1f}%",
            row["dominant"],
        )
        for row in v1_06_topology["rows"]
    )

    _latency_fig = go.Figure()
    _activation_fig = go.Figure()
    _palette = [COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"], COLORS["RedLine"]]
    for idx, candidate in enumerate(v1_06_architecture.candidates):
        _points = v1_06_curve.points_by_candidate[candidate.architecture_id]
        _latency_fig.add_trace(go.Scatter(
            x=[point.scale_value for point in _points],
            y=[point.latency_ms for point in _points],
            mode="lines+markers",
            name=candidate.label,
            line=dict(color=_palette[idx % len(_palette)], width=2.5),
            marker=dict(size=6),
        ))
        _activation_fig.add_trace(go.Scatter(
            x=[point.scale_value for point in _points],
            y=[point.activation_mb for point in _points],
            mode="lines+markers",
            name=candidate.label,
            line=dict(color=_palette[idx % len(_palette)], width=2.5),
            marker=dict(size=6),
        ))
    _latency_fig.add_hline(
        y=v1_06_architecture.latency_budget_ms,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=1.5,
        annotation_text="latency budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _latency_fig.add_vline(
        x=v1_06_scale.value,
        line_dash="dot",
        line_color=COLORS["TextMuted"],
        line_width=1.3,
        annotation_text="current",
    )
    _latency_fig.update_layout(
        height=360,
        xaxis=dict(
            title=f"{v1_06_architecture.scaling_variable} ({v1_06_architecture.scaling_unit})",
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(title="Latency (ms)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=50),
    )
    apply_plotly_theme(_latency_fig)

    _activation_fig.add_hline(
        y=v1_06_architecture.memory_budget_mb,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=1.5,
        annotation_text="memory budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _activation_fig.add_vline(
        x=v1_06_scale.value,
        line_dash="dot",
        line_color=COLORS["TextMuted"],
        line_width=1.3,
        annotation_text="current",
    )
    _activation_fig.update_layout(
        height=360,
        xaxis=dict(
            title=f"{v1_06_architecture.scaling_variable} ({v1_06_architecture.scaling_unit})",
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(title="Activation / state memory (MB)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=50),
    )
    apply_plotly_theme(_activation_fig)

    _failure_rows = tuple(
        (
            candidate.label,
            candidate.scaling_law,
            (
                f"{v1_06_curve.first_failure_by_candidate[candidate.architecture_id]:.0f} {v1_06_architecture.scaling_unit}"
                if v1_06_curve.first_failure_by_candidate[candidate.architecture_id] is not None
                else "not reached"
            ),
        )
        for candidate in v1_06_architecture.candidates
    )

    _bias_fig = go.Figure()
    _bias_fig.add_trace(go.Scatter(
        x=[row["deployability"] for row in v1_06_bias["rows"]],
        y=[row["quality"] for row in v1_06_bias["rows"]],
        mode="markers+text",
        text=[row["label"] for row in v1_06_bias["rows"]],
        textposition="top center",
        marker=dict(
            size=[max(10, min(28, row["data_need"] / 5.0)) for row in v1_06_bias["rows"]],
            color=[
                COLORS["GreenLine"] if row["feasible"] else COLORS["RedLine"]
                for row in v1_06_bias["rows"]
            ],
            line=dict(color="white", width=1.5),
        ),
        name="Candidate",
    ))
    _bias_fig.add_hline(
        y=v1_06_architecture.quality_floor_pct,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=1.5,
        annotation_text="quality floor",
        annotation_font_color=COLORS["RedLine"],
    )
    _bias_fig.update_layout(
        height=360,
        xaxis=dict(title="Deployability score", range=[0, 105], gridcolor="#f1f5f9"),
        yaxis=dict(title="Quality proxy (%)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=50),
    )
    apply_plotly_theme(_bias_fig)

    _bias_rows = tuple(
        (
            row["label"],
            row["family"],
            f"{row['bias_match']:.2f}x",
            f"{row['data_need']:.1f}",
            f"{row['quality']:.1f}%",
            f"{row['deployability']:.1f}",
            "yes" if row["feasible"] else "no",
            row["dominant"],
        )
        for row in v1_06_bias["rows"]
    )

    _decision_rows = (
        (
            "Selected",
            v1_06_review["selected"].label,
            f"{v1_06_review['selected'].quality_pct:.1f}%",
            f"{v1_06_review['selected'].latency_ms:.2f} ms",
            f"{v1_06_review['selected'].activation_mb:.2f} MB",
            "yes" if v1_06_review["selected"].feasible else "no",
            v1_06_review["selected"].dominant_constraint,
        ),
        (
            "Leaderboard",
            v1_06_review["leaderboard"].label,
            f"{v1_06_review['leaderboard'].quality_pct:.1f}%",
            f"{v1_06_review['leaderboard'].latency_ms:.2f} ms",
            f"{v1_06_review['leaderboard'].activation_mb:.2f} MB",
            "yes" if v1_06_review["leaderboard"].feasible else "no",
            v1_06_review["leaderboard"].dominant_constraint,
        ),
        (
            "Headroom pick",
            v1_06_review["headroom_pick"].label,
            f"{v1_06_review['headroom_pick'].quality_pct:.1f}%",
            f"{v1_06_review['headroom_pick'].latency_ms:.2f} ms",
            f"{v1_06_review['headroom_pick'].activation_mb:.2f} MB",
            "yes" if v1_06_review["headroom_pick"].feasible else "no",
            v1_06_review["headroom_pick"].dominant_constraint,
        ),
    )

    _validation_rows = tuple((test,) for test in v1_06_architecture.validation_tests)
    _rejection_rows = tuple((item,) for item in v1_06_decision.rejected_alternatives)
    _amount_system = v1_06_track_amount_system(v1_06_architecture)

    _part_a_labels = {
        "local": "local convolution or streaming state",
        "attention": "attention/token mixing",
        "kernel": "kernel support or dispatch",
        "quality": "quality guardrail",
    }
    _part_b_labels = {
        "memory": "activation/KV/state memory",
        "latency": "latency",
        "power": "power or duty cycle",
        "quality_kernel": "quality or kernel support",
    }
    _part_c_labels = {
        "local_bias": "structured local bias",
        "flexible_attention": "flexible attention",
        "leaderboard": "leaderboard-quality model",
        "track_constraint": "track constraint",
    }
    _part_d_labels = {
        "leaderboard": "highest-quality architecture",
        "headroom": "feasible headroom model",
        "smallest": "smallest architecture",
        "guardrail": "track guardrail",
    }

    _part_a = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Topology Changes Locality</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            As {v1_06_architecture.stakeholder}, you must choose a topology for
            {v1_06_architecture.workload_label}. The amount system is {_amount_system["amounts"]}.</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Concept</h2>
          <ul class="mlsysbook-list">
            <li>Topology is not decoration: local filters, recurrent state, attention gather/reduce, and sparse routing move different amounts.</li>
            <li>The chart is normalized to the selected track budgets so the binding amount is visible.</li>
            <li>Quality is only one column; locality, memory, latency, power, and kernel support decide deployability.</li>
          </ul>
          <div class="mlsysbook-callout"><strong>Track architecture story:</strong> {v1_06_architecture.architecture_story}</div>
        </div>
        """),
        mo.Html('<div class="mlsysbook-panel"><h2>Prediction</h2></div>'),
        mo.hstack([v1_06_failure_prediction], justify="start"),
        mo.Html(v1_06_prediction_html(
            "Prediction Check",
            v1_06_failure_prediction.value,
            v1_06_topology["actual"],
            _part_a_labels,
        )),
        mo.Html('<div class="mlsysbook-panel"><h2>Manipulation</h2></div>'),
        mo.hstack([v1_06_scale], justify="start"),
        mo.as_html(_signature_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Topology Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_06_fields_html({
                "Best topology": f"{v1_06_topology['best_label']} ({v1_06_topology['best_topology']})",
                "Access pattern": v1_06_topology["best_access"],
                "Locality": v1_06_topology["best_locality"],
                "Dominant amount": v1_06_topology["best_constraint"],
                "Track stake": _amount_system["stake"],
                "Natural failure": _amount_system["failure"],
            })}
          </div>
          {v1_06_table_html(
              ("Architecture", "Topology", "Access pattern", "Locality", "GMAC/MB", "Memory budget", "Latency budget", "Dominant"),
              _topology_rows,
              numeric=(4, 5, 6),
          )}
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Signature Table</h2>
          {v1_06_table_html(
              ("Architecture", "Family", "Params", "GMAC", "Activations", "Latency", "Power", "Quality", "Kernels", "Feasible", "Dominant"),
              _signature_rows,
              numeric=(2, 3, 4, 5, 6, 7, 8),
          )}
        </div>
        """),
        mo.Html(v1_06_callout_html(
            "Consequence",
            (
                f"{v1_06_topology['best_label']} is the current topology match because "
                f"{v1_06_topology['best_constraint']} is the governing amount. "
                "A different family may have a higher quality proxy but create a resource shape the track cannot absorb."
            ),
            kind="ok",
        )),
        MathPeek(
            "dense ~= H*W*C weights; convolution ~= K*K*C weights per filter; attention scores ~= S^2",
            {
                "topology": "The graph structure determines operation and memory locality before runtime optimization.",
                "locality": "Sliding windows reuse nearby data; attention gathers across all tokens; recurrence keeps compact state but serializes time.",
                "amount system": "The selected track decides whether locality appears as latency, SRAM, p99 margin, or cost/request.",
            },
        ),
        source_trace(
            {
                "chapter_anchor": "CNN algorithmic structure; RNN system implications; Computational primitives",
                "formula": "local filters, recurrent state, and attention gather/reduce imply different data movement",
                "track_id": v1_06_architecture.track_id,
            },
            summary="Part A source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_06_topology_checkpoint,
    ])

    _part_b = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Attention And Sequence Scaling Hide The Memory Wall</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            Product asks for a larger {v1_06_architecture.scaling_variable}. The weights may still fit,
            but activation, attention, or state memory can become the hidden wall.</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Concept</h2>
          <ul class="mlsysbook-list">
            <li>Attention scores scale with sequence length squared during full attention.</li>
            <li>Serving state and KV-like caches scale with resident sequence length and concurrency.</li>
            <li>A candidate that fits at the default scale can still be fragile at the next product requirement.</li>
          </ul>
        </div>
        """),
        mo.Html('<div class="mlsysbook-panel"><h2>Prediction</h2></div>'),
        v1_06_memory_prediction,
        mo.Html(v1_06_prediction_html(
            "Prediction Check",
            v1_06_memory_prediction.value,
            v1_06_attention["actual"],
            _part_b_labels,
        )),
        mo.Html('<div class="mlsysbook-panel"><h2>Manipulation</h2></div>'),
        mo.hstack([v1_06_scale], justify="start"),
        mo.hstack([mo.as_html(_latency_fig), mo.as_html(_activation_fig)], widths="equal"),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Scaling Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_06_fields_html({
                "Attention-like candidate": v1_06_attention["attention_label"],
                "Dominant amount": v1_06_attention["dominant_amount"],
                "First infeasible scale": v1_06_attention["first_failure"],
                "Memory pressure": f"{v1_06_attention['memory_pct']:.1f}% of budget",
                "Latency pressure": f"{v1_06_attention['latency_pct']:.1f}% of budget",
                "Power pressure": f"{v1_06_attention['power_pct']:.1f}% of budget",
            })}
          </div>
          {v1_06_table_html(("Architecture", "Scaling shape", "First infeasible scale"), _failure_rows)}
        </div>
        """),
        mo.Html(v1_06_callout_html(
            "Consequence",
            v1_06_attention["message"],
            kind=v1_06_attention["status"],
        )),
        MathPeek(
            "attention_scores = O(S^2); KV/state ~= B x layers x 2 x heads x S x d_head x bytes",
            {
                "S": f"{v1_06_architecture.scaling_variable} expressed as {v1_06_architecture.scaling_unit}.",
                "hidden wall": "Weights can fit while activation/state memory or bandwidth breaks the service.",
                "mitigation": "Reduce scale, use memory-aware attention, or choose a topology with bounded state.",
            },
        ),
        source_trace(
            {
                "chapter_anchor": "Attention system implications; Transformer training compute wall; Transformer inference memory bandwidth wall; KV cache sizing",
                "formula": "quadratic attention memory and linear resident state/KV cache",
                "track_id": v1_06_architecture.track_id,
            },
            summary="Part B source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_06_scaling_checkpoint,
    ])

    _part_c = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Inductive Bias Trades Quality, Data Need, And Deployability</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            The team asks whether a more flexible architecture is worth the data, memory, and validation cost.
            Your answer must fit {_amount_system["stake"]}.</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Concept</h2>
          <ul class="mlsysbook-list">
            <li>Inductive bias narrows the hypothesis space, often reducing data need and resource demand.</li>
            <li>No Free Lunch means the same bias can hurt when the data violates the assumed structure.</li>
            <li>Deployability is part of the bias trade-off, not a post-processing step.</li>
          </ul>
          <div class="mlsysbook-callout"><strong>Track bias statement:</strong> {_amount_system["bias"]}.</div>
        </div>
        """),
        mo.Html('<div class="mlsysbook-panel"><h2>Prediction</h2></div>'),
        v1_06_bias_prediction,
        mo.Html(v1_06_prediction_html(
            "Prediction Check",
            v1_06_bias_prediction.value,
            v1_06_bias["actual"],
            _part_c_labels,
        )),
        mo.Html('<div class="mlsysbook-panel"><h2>Manipulation</h2></div>'),
        mo.hstack([v1_06_data_pressure], justify="start"),
        mo.as_html(_bias_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Bias Frontier Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_06_fields_html({
                "Best bias fit": v1_06_bias["best"]["label"],
                "Leaderboard quality": v1_06_bias["leaderboard_label"],
                "Data pressure": f"{v1_06_data_pressure.value:.1f}x",
                "Best data-need index": f"{v1_06_bias['best']['data_need']:.1f}",
                "Best deployability": f"{v1_06_bias['best']['deployability']:.1f}",
                "Best dominant amount": v1_06_bias["best"]["dominant"],
            })}
          </div>
          {v1_06_table_html(
              ("Architecture", "Family", "Bias match", "Data need index", "Quality", "Deployability", "Feasible", "Dominant"),
              _bias_rows,
              numeric=(2, 3, 4, 5),
          )}
        </div>
        """),
        mo.Html(v1_06_callout_html(
            "Consequence",
            (
                f"{v1_06_bias['best']['label']} is the current bias recommendation. "
                f"It carries a data-need index of {v1_06_bias['best']['data_need']:.1f} and "
                f"a deployability score of {v1_06_bias['best']['deployability']:.1f}. "
                "If data pressure rises, a weaker or mismatched bias must buy its quality with more examples and more system budget."
            ),
            kind="ok" if v1_06_bias["best"]["feasible"] else "warn",
        )),
        MathPeek(
            "effective_data_need ~= data_pressure / bias_match; deployable if quality >= floor and resource ratios <= 1",
            {
                "bias_match": "How well the architecture's structural prior matches the selected track workload.",
                "No Free Lunch": "A bias improves matching tasks and degrades mismatched tasks.",
                "deployability": "A scenario score combining memory, latency, power, quality, and kernel guardrails.",
            },
        ),
        source_trace(
            {
                "chapter_anchor": "Learnability gap; No Free Lunch theorem; Inductive bias hierarchy",
                "formula": "sample/data need changes with architecture bias and workload structure",
                "track_id": v1_06_architecture.track_id,
            },
            summary="Part C source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_06_bias_checkpoint,
    ])

    _part_d = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part D: Architecture Selection Is A Deployment Recommendation</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            You are in architecture review. The recommendation must name a selected family,
            rejected alternatives, guardrails, and the validation test that could overturn the choice.</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Concept</h2>
          <ul class="mlsysbook-list">
            <li>The highest-quality candidate is only valid inside the deployment envelope.</li>
            <li>Review should compare the selected design against the leaderboard and headroom alternatives.</li>
            <li>The memo is incomplete without residual risk and validation evidence.</li>
          </ul>
        </div>
        """),
        mo.Html('<div class="mlsysbook-panel"><h2>Prediction</h2></div>'),
        v1_06_review_prediction,
        mo.Html(v1_06_prediction_html(
            "Prediction Check",
            v1_06_review_prediction.value,
            v1_06_review["actual"],
            _part_d_labels,
        )),
        mo.Html('<div class="mlsysbook-panel"><h2>Manipulation</h2></div>'),
        v1_06_arch_choice,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Review Evidence</h2>
          <div class="mlsysbook-grid">
            {v1_06_fields_html({
                "Selected architecture": v1_06_decision.selected_label,
                "Review status": v1_06_review["status"],
                "Dominant constraint": v1_06_decision.dominant_constraint,
                "Next failure": v1_06_decision.next_failure,
                "Quality": f"{v1_06_selected_eval.quality_pct:.1f}% / floor {v1_06_architecture.quality_floor_pct:.1f}%",
                "Kernel support": f"{v1_06_selected_eval.kernel_support_pct:.1f}% / floor {v1_06_architecture.kernel_support_floor_pct:.1f}%",
            })}
          </div>
          {v1_06_table_html(
              ("Role", "Architecture", "Quality", "Latency", "Activation", "Feasible", "Dominant"),
              _decision_rows,
              numeric=(2, 3, 4),
          )}
        </div>
        """),
        mo.Html(v1_06_callout_html(
            "Consequence",
            v1_06_review["message"],
            kind=v1_06_review["kind"],
        )),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Rejected Alternatives And Validation</h2>
          {v1_06_table_html(("Rejected alternative",), _rejection_rows)}
          {v1_06_table_html(("Validation test",), _validation_rows)}
        </div>
        """),
        MathPeek(
            "valid_architecture = memory<=budget and latency<=budget and power<=budget and quality>=floor and kernels>=floor",
            {
                "leaderboard": "The highest quality proxy is compared against system guardrails.",
                "headroom": "The feasible candidate farthest from memory, latency, and power failure.",
                "residual risk": "The validation condition that could falsify the recommendation.",
            },
        ),
        source_trace(
            {
                "chapter_anchor": "Architecture Selection Framework; Fallacies and Pitfalls",
                "formula": "deployment feasibility is a conjunction of constraints, not a single score",
                "track_id": v1_06_architecture.track_id,
            },
            summary="Part D source model",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Checkpoint</h2></div>'),
        v1_06_deployment_checkpoint,
    ])

    mo.ui.tabs({
        "Part A - Topology": _part_a,
        "Part B - Memory Wall": _part_b,
        "Part C - Inductive Bias": _part_c,
        "Part D - Recommendation": _part_d,
        "Synthesis": mo.md("Use the synthesis memo below after completing Parts A-D."),
    })
    return


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    v1_06_attention,
    v1_06_bias,
    v1_06_bias_checkpoint,
    v1_06_bias_prediction,
    v1_06_callout_html,
    v1_06_deployment_checkpoint,
    ledger,
    mo,
    v1_06_architecture,
    v1_06_decision,
    v1_06_failure_prediction,
    v1_06_fields_html,
    v1_06_memory_prediction,
    v1_06_profile,
    v1_06_reflection,
    v1_06_review,
    v1_06_review_prediction,
    v1_06_scaling_checkpoint,
    v1_06_selected_eval,
    v1_06_table_html,
    v1_06_topology,
    v1_06_topology_checkpoint,
    v1_06_variant,
):
    _memo_text = str(v1_06_reflection.value or "").strip()
    _ready_for_ledger = all(
        value is not None
        for value in (
            v1_06_failure_prediction.value,
            v1_06_memory_prediction.value,
            v1_06_bias_prediction.value,
            v1_06_review_prediction.value,
            v1_06_topology_checkpoint.value,
            v1_06_scaling_checkpoint.value,
            v1_06_bias_checkpoint.value,
            v1_06_deployment_checkpoint.value,
        )
    ) and bool(_memo_text)

    if any(
        value is not None
        for value in (
            v1_06_failure_prediction.value,
            v1_06_memory_prediction.value,
            v1_06_bias_prediction.value,
            v1_06_review_prediction.value,
        )
    ):
        ledger.save(chapter=6, design={
            "chapter": "v1_06",
            "track_id": v1_06_profile.track_id,
            "scenario_id": v1_06_variant.scenario_id,
            "hardware_ref": v1_06_architecture.hardware_ref,
            "model_ref": v1_06_architecture.model_ref,
            "completed": _ready_for_ledger,
            "failure_prediction": v1_06_failure_prediction.value,
            "attention_memory_prediction": v1_06_memory_prediction.value,
            "bias_prediction": v1_06_bias_prediction.value,
            "review_prediction": v1_06_review_prediction.value,
            "topology_checkpoint": v1_06_topology_checkpoint.value,
            "scaling_checkpoint": v1_06_scaling_checkpoint.value,
            "bias_checkpoint": v1_06_bias_checkpoint.value,
            "deployment_checkpoint": v1_06_deployment_checkpoint.value,
            "selected_architecture": v1_06_decision.selected_id,
            "dominant_constraint": v1_06_decision.dominant_constraint,
            "next_failure": v1_06_decision.next_failure,
            "quality_pct": v1_06_selected_eval.quality_pct,
            "kernel_support_pct": v1_06_selected_eval.kernel_support_pct,
            "residual_risk": v1_06_decision.residual_risk,
            "validation_requirement": v1_06_decision.validation_requirement,
        })

    def build_synthesis():
        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel mlsysbook-nugget">
              <div class="mlsysbook-part-title"><h2>Synthesis: Record The Architecture Recommendation Memo</h2></div>
              <div class="mlsysbook-callout"><strong>Invariant:</strong>
                Architecture choices create different resource shapes. The memo is valid only when
                the selected family, rejected alternatives, measured constraint, and residual risk
                are all explicit.</div>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Decision Record</h2>
              <div class="mlsysbook-grid">
                {v1_06_fields_html({
                    "Track": v1_06_architecture.label,
                    "Selected architecture": v1_06_decision.selected_label,
                    "Topology match": v1_06_topology["best_label"],
                    "Attention wall": f"{v1_06_attention['attention_label']} -> {v1_06_attention['dominant_amount']}",
                    "Bias recommendation": v1_06_bias["best"]["label"],
                    "Review status": v1_06_review["status"],
                    "Dominant constraint": v1_06_decision.dominant_constraint,
                    "Next failure": v1_06_decision.next_failure,
                    "Residual risk": v1_06_decision.residual_risk,
                    "Validation requirement": v1_06_decision.validation_requirement,
                })}
              </div>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Memo Fields</h2>
              {v1_06_table_html(
                  ("Required field", "Current value"),
                  (
                      ("Recommendation", v1_06_decision.memo_summary),
                      ("Rejected alternatives", "; ".join(v1_06_decision.rejected_alternatives)),
                      ("Measured evidence", f"{v1_06_selected_eval.latency_ms:.2f} ms, {v1_06_selected_eval.activation_mb:.2f} MB, {v1_06_selected_eval.power_w:.3f} W"),
                      ("Residual risk", v1_06_decision.residual_risk),
                      ("Validation requirement", v1_06_decision.validation_requirement),
                  ),
              )}
            </div>
            """),
            mo.Html('<div class="mlsysbook-panel"><h2>Student Memo</h2></div>'),
            v1_06_reflection,
            mo.Html(v1_06_callout_html(
                "Ledger Status",
                "Ready to save as completed." if _ready_for_ledger else "Complete the predictions, checkpoints, and memo text before treating the ledger entry as complete.",
                kind="ok" if _ready_for_ledger else "warn",
            )),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Big Takeaways</h2>
              <ul class="mlsysbook-list">
                <li><strong>Topology controls locality.</strong> The same quality proxy can hide different operation and memory shapes.</li>
                <li><strong>Scaling shape exposes the next wall.</strong> Attention and sequence state can make memory the binding resource.</li>
                <li><strong>Bias is a deployment trade-off.</strong> Stronger structure can reduce data need and resource cost, but only when the workload matches it.</li>
                <li><strong>Selection is a recommendation.</strong> A defensible architecture memo names rejected alternatives and residual risk.</li>
              </ul>
            </div>
            """),
            mo.Html(f"""
            <div class="lab-hud">
                <span class="hud-label">LAB</span>
                <span class="hud-value">06 &middot; Architecture Tax</span>
                <span class="hud-label">TRACK</span>
                <span class="hud-value">{v1_06_profile.label}</span>
                <span style="flex:1;"></span>
                <span class="hud-label">ARTIFACT</span>
                <span class="hud-value">{v1_06_architecture.report_artifact}</span>
                <span class="hud-label">STATUS</span>
                <span class="hud-active">ACTIVE</span>
            </div>
            """),
        ])

    build_synthesis()
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    report_export_panel,
    v1_06_attention,
    v1_06_architecture,
    v1_06_bias,
    v1_06_bias_checkpoint,
    v1_06_bias_prediction,
    v1_06_curve,
    v1_06_data_pressure,
    v1_06_decision,
    v1_06_deployment_checkpoint,
    v1_06_failure_prediction,
    v1_06_memory_prediction,
    v1_06_metadata,
    v1_06_profile,
    v1_06_reflection,
    v1_06_review,
    v1_06_review_prediction,
    v1_06_scale,
    v1_06_scaling_checkpoint,
    v1_06_selected_eval,
    v1_06_signature,
    v1_06_topology,
    v1_06_topology_checkpoint,
    v1_06_variant,
):
    _incomplete = []
    if v1_06_failure_prediction.value is None:
        _incomplete.append("Part A topology prediction")
    if v1_06_memory_prediction.value is None:
        _incomplete.append("Part B memory-wall prediction")
    if v1_06_bias_prediction.value is None:
        _incomplete.append("Part C inductive-bias prediction")
    if v1_06_review_prediction.value is None:
        _incomplete.append("Part D review prediction")
    if v1_06_topology_checkpoint.value is None:
        _incomplete.append("Part A topology checkpoint")
    if v1_06_scaling_checkpoint.value is None:
        _incomplete.append("Part B scaling checkpoint")
    if v1_06_bias_checkpoint.value is None:
        _incomplete.append("Part C bias checkpoint")
    if v1_06_deployment_checkpoint.value is None:
        _incomplete.append("Part D deployment checkpoint")
    if not str(v1_06_reflection.value or "").strip():
        _incomplete.append("Synthesis recommendation memo")

    _report = build_lab_report(
        v1_06_metadata,
        track=v1_06_profile.label,
        scenario=v1_06_variant.workload_summary,
        learning_objectives=(
            "Explain how topology changes operation and memory locality for the selected track.",
            "Sweep the track-specific scaling variable and identify the memory/latency/power wall.",
            "Reason about inductive bias as a trade-off among quality, data need, and deployability.",
            "Recommend one architecture family and state rejected alternatives plus residual risk.",
        ),
        predictions={
            "topology_prediction": v1_06_failure_prediction.value,
            "memory_wall_prediction": v1_06_memory_prediction.value,
            "bias_prediction": v1_06_bias_prediction.value,
            "review_prediction": v1_06_review_prediction.value,
        },
        knob_settings={
            "scale_value": v1_06_scale.value,
            "scaling_variable": v1_06_architecture.scaling_variable,
            "data_pressure": v1_06_data_pressure.value,
            "selected_architecture": v1_06_decision.selected_id,
        },
        evidence_summary={
            "hardware_ref": v1_06_architecture.hardware_ref,
            "model_ref": v1_06_architecture.model_ref,
            "memory_budget_mb": v1_06_architecture.memory_budget_mb,
            "latency_budget_ms": v1_06_architecture.latency_budget_ms,
            "power_budget_w": v1_06_architecture.power_budget_w,
            "selected_architecture": v1_06_decision.selected_label,
            "dominant_constraint": v1_06_decision.dominant_constraint,
            "next_failure": v1_06_decision.next_failure,
            "selected_latency_ms": v1_06_selected_eval.latency_ms,
            "selected_activation_mb": v1_06_selected_eval.activation_mb,
            "topology_match": v1_06_topology["best_label"],
            "attention_wall": v1_06_attention["dominant_amount"],
            "bias_recommendation": v1_06_bias["best"]["label"],
            "review_status": v1_06_review["status"],
        },
        final_decision=v1_06_decision.memo_summary,
        big_takeaways=(
            "Topology changes locality and therefore the amount system the track must pay.",
            "Attention and sequence scaling can make activation/state memory the hidden wall.",
            "Inductive bias trades quality, data need, and deployability.",
            "A defensible architecture memo names rejected alternatives and the validation risk that remains.",
        ),
        reflections={
            "student_memo": v1_06_reflection.value,
            "topology_checkpoint": v1_06_topology_checkpoint.value,
            "scaling_checkpoint": v1_06_scaling_checkpoint.value,
            "bias_checkpoint": v1_06_bias_checkpoint.value,
            "deployment_checkpoint": v1_06_deployment_checkpoint.value,
            "rejected_alternatives": v1_06_decision.rejected_alternatives,
            "validation_requirement": v1_06_decision.validation_requirement,
            "report_artifact": v1_06_architecture.report_artifact,
        },
        residual_risk=v1_06_decision.residual_risk,
        source_trace={
            "track_id": v1_06_profile.track_id,
            "scenario_id": v1_06_variant.scenario_id,
            "hardware_ref": v1_06_variant.hardware_ref,
            "model_ref": v1_06_variant.model_ref,
            "shared_helper": "mlsysbook_labs.architecture",
            "local_helpers": (
                "v1_06_topology_summary",
                "v1_06_attention_wall",
                "v1_06_bias_frontier",
                "v1_06_review_summary",
            ),
            "source_policy": v1_06_profile.source_policy,
        },
        result_snapshot={
            "architecture_profile": v1_06_architecture,
            "signature": v1_06_signature,
            "scaling_curve": v1_06_curve,
            "topology": v1_06_topology,
            "attention_wall": v1_06_attention,
            "bias_frontier": v1_06_bias,
            "review": v1_06_review,
            "decision": v1_06_decision,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-06 architecture recommendation memo is generated locally from "
                "the selected track, your predictions, checkpoints, and the computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
