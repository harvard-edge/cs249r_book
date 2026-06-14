import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
async def _():
    import html as html_lib
    import sys
    from pathlib import Path

    import marimo as mo
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
        MathPeek,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
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
        Hardware,
        LAB_CSS,
        MathPeek,
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
        track_arc_context,
        track_context,
        track_selector,
        ureg,
    )


@app.cell
def _(get_lab_metadata):
    v2_06_lab_path = "vol2/lab_06_collective_communication.py"
    v2_06_chapter = 6
    v2_06_metadata = get_lab_metadata(v2_06_lab_path)
    return v2_06_chapter, v2_06_lab_path, v2_06_metadata


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
    v2_06_metadata,
    v2_06_track_picker,
):
    v2_06_track_id = v2_06_track_picker.value
    v2_06_profile = get_track_profile(v2_06_track_id)
    v2_06_variant = get_lab_track_variant(v2_06_metadata.lab_id, v2_06_profile.track_id)
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


@app.cell
def _(
    COLORS,
    Hardware,
    Systems,
    apply_plotly_theme,
    calc_hierarchical_allreduce_time,
    calc_ring_allreduce_time,
    calc_tree_allreduce_time,
    go,
    html_lib,
    mo,
    np,
    ureg,
):
    import math

    def v2_06_fmt(value, digits=2):
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            if abs(value) >= 100:
                return f"{value:,.0f}"
            if abs(value) >= 10:
                return f"{value:,.1f}"
            return f"{value:,.{digits}f}"
        return str(value)

    def v2_06_status_label(ok):
        return "PASS" if ok else "FAIL"

    def v2_06_bandwidth(fabric_obj):
        return getattr(fabric_obj, "bandwidth_per_direction", None) or fabric_obj.bandwidth

    _v2_06_nvlink = Hardware.Cloud.H100.nvlink
    _v2_06_fabrics = {
        "ib": {
            "label": "InfiniBand NDR",
            "fabric": Systems.Fabrics.InfiniBand_NDR,
            "source": "MLSysIM Systems.Fabrics.InfiniBand_NDR",
        },
        "eth": {
            "label": "100G Ethernet",
            "fabric": Systems.Fabrics.Ethernet_100G,
            "source": "MLSysIM Systems.Fabrics.Ethernet_100G",
        },
        "nvlink": {
            "label": "NVLink-only",
            "fabric": _v2_06_nvlink,
            "source": "MLSysIM Hardware.Cloud.H100.nvlink",
        },
    }

    def v2_06_fabric_options():
        return {entry["label"]: key for key, entry in _v2_06_fabrics.items()}

    def v2_06_fabric_label(fabric_id):
        return _v2_06_fabrics.get(fabric_id, _v2_06_fabrics["ib"])["label"]

    def v2_06_default_fabric_label(defaults):
        return v2_06_fabric_label(str(defaults.get("fabric", "ib")))

    def v2_06_html_table(rows, columns, caption=""):
        def _esc(value):
            return html_lib.escape(str(value))

        _head = "".join(f"<th>{_esc(label)}</th>" for key, label in columns)
        _body = ""
        for row in rows:
            _body += "<tr>" + "".join(f"<td>{_esc(row.get(key, ''))}</td>" for key, label in columns) + "</tr>"
        _caption = f"<caption>{_esc(caption)}</caption>" if caption else ""
        return mo.Html(
            f"""
            <table style="width:100%; border-collapse:collapse; margin:10px 0 16px 0;
                          font-size:0.88rem; background:white; border:1px solid {COLORS['Border']};">
              {_caption}
              <thead>
                <tr style="background:#f8fafc; color:{COLORS['Text']}; text-align:left;">{_head}</tr>
              </thead>
              <tbody>{_body}</tbody>
            </table>
            <style>
              table th, table td {{
                border-bottom: 1px solid {COLORS['Border']};
                padding: 8px 10px;
                vertical-align: top;
              }}
              table caption {{
                caption-side: bottom;
                text-align: left;
                color: {COLORS['TextMuted']};
                padding-top: 6px;
                font-size: 0.78rem;
              }}
            </style>
            """
        )

    def v2_06_prediction_feedback(value, correct_value, correct_message, miss_message):
        if value is None:
            return mo.callout(mo.md("Commit to a prediction before opening the instrument."), kind="warn")
        if value == correct_value:
            return mo.callout(mo.md(correct_message), kind="success")
        return mo.callout(mo.md(miss_message), kind="warn")

    def v2_06_failure_callout(ok, pass_message, fail_message):
        return mo.callout(mo.md(pass_message if ok else fail_message), kind="success" if ok else "danger")

    def v2_06_track_lens(profile, variant):
        defaults = variant.defaults
        lenses = {
            "iphone": {
                "scenario": "A mobile federated learning engineer is deciding how a phone cohort should aggregate updates without turning secure aggregation into a battery or privacy liability.",
                "decision_frame": "Choose the cohort aggregation algorithm, topology assumption, and optimization policy.",
                "payload_name": "secure update payload",
                "participant_name": "phones in the cohort",
                "budget_ms": 120.0,
                "quality_floor_pct": 88.0,
                "compression_penalty_pct": 2.6,
                "max_compression_ratio": 4,
                "max_overlap_pct": 45,
                "hierarchy_supported": True,
                "requires_local_tier": False,
                "topology_guardrail_text": "Cohort hierarchy is an edge-staging assumption; privacy and radio evidence must support it.",
                "optimization_guardrail_label": "privacy/battery risk",
                "validation_focus": "privacy protocol audit plus battery regression",
                "report_prompt": "Frame the review around cohort aggregation, secure aggregation overhead, and battery cost.",
                "reliability_implication": "V2-07 should test whether coordinator retries or dropped phones corrupt the aggregate or leak privacy metadata.",
            },
            "oura_ring": {
                "scenario": "A wearable systems engineer is deciding how intermittent rings should synchronize tiny summaries through phone-nearby windows.",
                "decision_frame": "Choose the sync aggregation algorithm, topology assumption, and compression policy.",
                "payload_name": "nightly summary payload",
                "participant_name": "rings in a sync cohort",
                "budget_ms": 60.0,
                "quality_floor_pct": 90.0,
                "compression_penalty_pct": 2.3,
                "max_compression_ratio": 4,
                "max_overlap_pct": 25,
                "hierarchy_supported": True,
                "requires_local_tier": False,
                "topology_guardrail_text": "Phone-mediated hierarchy is valid only if the sync window is long enough.",
                "optimization_guardrail_label": "sync reliability risk",
                "validation_focus": "sync-window replay plus payload integrity check",
                "report_prompt": "Frame the review around intermittent connectivity, wakeups, and payload integrity.",
                "reliability_implication": "V2-07 should account for phone absence, retry storms, and partial cohort updates.",
            },
            "robotaxi": {
                "scenario": "An autonomous fleet data platform lead is deciding how vehicle events should aggregate through depot and cloud tiers without losing safety evidence.",
                "decision_frame": "Choose the depot/cloud aggregation topology and optimization policy.",
                "payload_name": "fleet event/update payload",
                "participant_name": "vehicles or depot shards",
                "budget_ms": 180.0,
                "quality_floor_pct": 93.0,
                "compression_penalty_pct": 3.8,
                "max_compression_ratio": 2,
                "max_overlap_pct": 55,
                "hierarchy_supported": True,
                "requires_local_tier": True,
                "topology_guardrail_text": "Depot hierarchy must reduce cloud-facing payload without erasing rare-event evidence.",
                "optimization_guardrail_label": "event fidelity risk",
                "validation_focus": "rare-event fidelity audit plus depot upload replay",
                "report_prompt": "Frame the review around depot hierarchy, rare-event fidelity, and fleet update latency.",
                "reliability_implication": "V2-07 should test whether failed depots, delayed uploads, or corrupt events bias recovery and retraining.",
            },
            "cloud_fleet": {
                "scenario": "A distributed training performance lead is deciding which AllReduce plan should run across NVLink H100 nodes and InfiniBand.",
                "decision_frame": "Choose the collective algorithm, topology assumption, and optimization policy for exposed training step time.",
                "payload_name": "gradient bucket payload",
                "participant_name": "accelerators",
                "budget_ms": 130.0,
                "quality_floor_pct": 94.0,
                "compression_penalty_pct": 5.5,
                "max_compression_ratio": 2,
                "max_overlap_pct": 70,
                "hierarchy_supported": True,
                "requires_local_tier": True,
                "topology_guardrail_text": "A multi-node H100 plan must use local/global topology evidence rather than a flat-ring assumption.",
                "optimization_guardrail_label": "convergence/scheduling risk",
                "validation_focus": "collective profiler plus convergence regression",
                "report_prompt": "Frame the review around training throughput, topology mapping, overlap evidence, and convergence risk.",
                "reliability_implication": "V2-07 should test checkpoint/restart cost, collective hangs, and silent gradient corruption under the selected topology.",
            },
        }
        lens = dict(lenses[profile.track_id])
        lens["operation"] = defaults.get("operation", "collective")
        lens["residual_risk"] = defaults.get("residual_risk", "residual communication risk not specified")
        lens["validation_tests"] = tuple(defaults.get("validation_tests", (lens["validation_focus"],)))
        lens["workload_summary"] = variant.workload_summary
        lens["topology_label"] = defaults.get("topology", "selected topology")
        return lens

    def v2_06_collective_terms(n_gpus, message_gb, fabric_id, local_group=1, compression_ratio=1):
        n = max(2, int(n_gpus))
        local = max(1, min(int(local_group), n))
        compression = max(1, int(compression_ratio))
        size_gb = max(0.0001, float(message_gb))
        effective_gb = size_gb / compression
        fabric_entry = _v2_06_fabrics.get(str(fabric_id), _v2_06_fabrics["ib"])
        fabric_obj = fabric_entry["fabric"]
        bandwidth = v2_06_bandwidth(fabric_obj)
        latency = fabric_obj.latency
        msg = effective_gb * ureg.GB

        ring_total_q = calc_ring_allreduce_time(msg, n, bandwidth, latency)
        ring_alpha_q = (2 * (n - 1) * latency).to(ureg.millisecond)
        ring_total_ms = ring_total_q.m_as("ms")
        ring_alpha_ms = ring_alpha_q.m_as("ms")
        ring_beta_ms = max(0.0, ring_total_ms - ring_alpha_ms)

        tree_total_q = calc_tree_allreduce_time(msg, n, bandwidth, latency)
        tree_steps = 2 * math.ceil(math.log2(max(n, 2)))
        tree_alpha_q = (tree_steps * latency).to(ureg.millisecond)
        tree_total_ms = tree_total_q.m_as("ms")
        tree_alpha_ms = tree_alpha_q.m_as("ms")
        tree_beta_ms = max(0.0, tree_total_ms - tree_alpha_ms)

        n_nodes = max(1, int(math.ceil(n / local)))
        nvlink_bw = v2_06_bandwidth(_v2_06_nvlink)
        hier_total_q = calc_hierarchical_allreduce_time(
            msg,
            n_nodes,
            local,
            nvlink_bw,
            bandwidth,
            _v2_06_nvlink.latency,
            latency,
        )
        hier_total_ms = hier_total_q.m_as("ms")
        crossover_gb = (latency * bandwidth).to(ureg.GB).magnitude

        ring_binding = "alpha latency term" if ring_alpha_ms >= ring_beta_ms else "beta bandwidth term"
        tree_binding = "alpha latency term" if tree_alpha_ms >= tree_beta_ms else "beta bandwidth term"
        hierarchy_binding = "topology/inter-node beta" if effective_gb >= max(crossover_gb, 0.001) else "topology/alpha"

        return {
            "n_gpus": n,
            "local_group": local,
            "n_nodes": n_nodes,
            "message_gb": size_gb,
            "effective_gb": effective_gb,
            "compression_ratio": compression,
            "fabric_id": str(fabric_id),
            "fabric_label": fabric_entry["label"],
            "fabric_source": fabric_entry["source"],
            "bandwidth_gb_s": bandwidth.to(ureg.GB / ureg.second).magnitude,
            "latency_us": latency.to(ureg.microsecond).magnitude,
            "crossover_gb": crossover_gb,
            "ring": {
                "key": "ring",
                "label": "Flat Ring",
                "alpha_ms": ring_alpha_ms,
                "beta_ms": ring_beta_ms,
                "total_ms": ring_total_ms,
                "binding": ring_binding,
            },
            "tree": {
                "key": "tree",
                "label": "Tree",
                "alpha_ms": tree_alpha_ms,
                "beta_ms": tree_beta_ms,
                "total_ms": tree_total_ms,
                "binding": tree_binding,
            },
            "hierarchical": {
                "key": "hierarchical",
                "label": "Hierarchical",
                "alpha_ms": 0.0,
                "beta_ms": hier_total_ms,
                "total_ms": hier_total_ms,
                "binding": hierarchy_binding,
            },
        }

    def v2_06_part_a_rows(lens, terms):
        rows = []
        for key in ("ring", "tree"):
            row = terms[key]
            rows.append(
                {
                    "Algorithm": row["label"],
                    "Alpha": f"{row['alpha_ms']:.3f} ms",
                    "Beta": f"{row['beta_ms']:.3f} ms",
                    "Total": f"{row['total_ms']:.3f} ms",
                    "Binding": row["binding"],
                    "Budget": f"{v2_06_status_label(row['total_ms'] <= lens['budget_ms'])} vs {lens['budget_ms']:.0f} ms",
                }
            )
        return rows

    def v2_06_topology_rows(lens, terms):
        rows = []
        for key in ("ring", "tree", "hierarchical"):
            row = terms[key]
            topology_ok = True
            reason = "topology assumption is valid for this track lens"
            if key == "ring" and lens["requires_local_tier"] and terms["local_group"] > 1 and terms["n_gpus"] > terms["local_group"]:
                topology_ok = False
                reason = "flat ring ignores the fast local tier and scarce global tier"
            if key == "hierarchical":
                topology_ok = lens["hierarchy_supported"] and terms["local_group"] > 1
                if topology_ok:
                    reason = f"local group of {terms['local_group']} shrinks global payload to M/G"
                else:
                    reason = "hierarchy needs a real or justified local aggregation tier"
            rows.append(
                {
                    "key": key,
                    "label": row["label"],
                    "total_ms": row["total_ms"],
                    "binding": row["binding"],
                    "topology_ok": topology_ok,
                    "passes_budget": row["total_ms"] <= lens["budget_ms"],
                    "reason": reason,
                }
            )
        return rows

    def v2_06_best_topology_key(rows):
        feasible = [row for row in rows if row["topology_ok"]]
        candidates = feasible or rows
        return min(candidates, key=lambda row: row["total_ms"])["key"]

    def v2_06_selected_key(part_b_checkpoint_value, topology_rows):
        if part_b_checkpoint_value == "flat_ring":
            return "ring"
        if part_b_checkpoint_value == "tree_schedule":
            return "tree"
        if part_b_checkpoint_value == "hierarchical_schedule":
            return "hierarchical"
        return v2_06_best_topology_key(topology_rows)

    def v2_06_row_by_key(rows, key):
        return next(row for row in rows if row["key"] == key)

    def v2_06_optimization_result(
        lens,
        selected_key,
        n_gpus,
        message_gb,
        fabric_id,
        local_group,
        compression_ratio,
        overlap_pct,
    ):
        raw_terms = v2_06_collective_terms(n_gpus, message_gb, fabric_id, local_group, compression_ratio=1)
        opt_terms = v2_06_collective_terms(
            n_gpus,
            message_gb,
            fabric_id,
            local_group,
            compression_ratio=compression_ratio,
        )
        raw_ms = raw_terms[selected_key]["total_ms"]
        compressed_ms = opt_terms[selected_key]["total_ms"]
        overlap = max(0.0, min(95.0, float(overlap_pct)))
        exposed_ms = compressed_ms * (1 - overlap / 100.0)
        compression = max(1, int(compression_ratio))
        quality_proxy = max(0.0, 100.0 - (compression - 1) * lens["compression_penalty_pct"])
        quality_ok = quality_proxy >= lens["quality_floor_pct"] and compression <= lens["max_compression_ratio"]
        schedule_ok = overlap <= lens["max_overlap_pct"]
        exposed_ok = exposed_ms <= lens["budget_ms"]
        optimization_ok = quality_ok and schedule_ok
        if not exposed_ok:
            risk = "communication remains exposed beyond the step-time guardrail"
        elif not quality_ok:
            risk = lens["optimization_guardrail_label"]
        elif not schedule_ok:
            risk = "overlap assumption exceeds the schedulable work window"
        else:
            risk = f"validate with {lens['validation_focus']}"
        return {
            "selected_key": selected_key,
            "selected_label": opt_terms[selected_key]["label"],
            "raw_ms": raw_ms,
            "compressed_ms": compressed_ms,
            "exposed_ms": exposed_ms,
            "saved_ms": max(0.0, raw_ms - exposed_ms),
            "effective_gb": opt_terms["effective_gb"],
            "quality_proxy": quality_proxy,
            "quality_ok": quality_ok,
            "schedule_ok": schedule_ok,
            "exposed_ok": exposed_ok,
            "optimization_ok": optimization_ok,
            "risk": risk,
            "binding_term": opt_terms[selected_key]["binding"],
        }

    def v2_06_plan_result(
        lens,
        selected_key,
        n_gpus,
        message_gb,
        fabric_id,
        local_group,
        compression_ratio,
        overlap_pct,
    ):
        terms = v2_06_collective_terms(n_gpus, message_gb, fabric_id, local_group, compression_ratio=compression_ratio)
        topology_rows = v2_06_topology_rows(lens, terms)
        selected_row = v2_06_row_by_key(topology_rows, selected_key)
        opt = v2_06_optimization_result(
            lens,
            selected_key,
            n_gpus,
            message_gb,
            fabric_id,
            local_group,
            compression_ratio,
            overlap_pct,
        )
        topology_ok = selected_row["topology_ok"]
        valid_plan = opt["exposed_ok"] and topology_ok and opt["optimization_ok"]
        if not opt["exposed_ok"]:
            binding_guardrail = "exposed step-time"
        elif not topology_ok:
            binding_guardrail = "topology"
        elif not opt["optimization_ok"]:
            binding_guardrail = "optimization risk"
        else:
            binding_guardrail = opt["binding_term"]
        return {
            **opt,
            "topology_ok": topology_ok,
            "topology_reason": selected_row["reason"],
            "valid_plan": valid_plan,
            "binding_guardrail": binding_guardrail,
            "budget_ms": lens["budget_ms"],
            "compression_ratio": max(1, int(compression_ratio)),
            "overlap_pct": max(0.0, min(95.0, float(overlap_pct))),
            "fabric_label": terms["fabric_label"],
            "participant_count": terms["n_gpus"],
            "local_group": terms["local_group"],
            "message_gb": terms["message_gb"],
        }

    def v2_06_alpha_beta_chart(terms, budget_ms):
        fig = go.Figure()
        labels = ["Flat Ring", "Tree"]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[terms["ring"]["alpha_ms"], terms["tree"]["alpha_ms"]],
                name="alpha latency",
                marker_color=COLORS["OrangeLine"],
            )
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[terms["ring"]["beta_ms"], terms["tree"]["beta_ms"]],
                name="beta bandwidth",
                marker_color=COLORS["BlueLine"],
            )
        )
        fig.add_hline(y=budget_ms, line_dash="dash", line_color=COLORS["RedLine"], annotation_text="track budget")
        fig.update_layout(
            barmode="stack",
            height=330,
            yaxis_title="Modeled time (ms)",
            legend=dict(orientation="h", y=1.13, x=0),
            margin=dict(l=60, r=20, t=50, b=45),
        )
        return apply_plotly_theme(fig)

    def v2_06_frontier_chart(lens, n_gpus, fabric_id, local_group, current_message_gb):
        sizes = np.geomspace(0.001, 80, 48)
        series = {"ring": [], "tree": [], "hierarchical": []}
        for size in sizes:
            terms = v2_06_collective_terms(n_gpus, float(size), fabric_id, local_group, compression_ratio=1)
            for key in series:
                series[key].append(terms[key]["total_ms"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sizes, y=series["ring"], mode="lines", name="Flat Ring"))
        fig.add_trace(go.Scatter(x=sizes, y=series["tree"], mode="lines", name="Tree"))
        fig.add_trace(go.Scatter(x=sizes, y=series["hierarchical"], mode="lines", name="Hierarchical"))
        current_terms = v2_06_collective_terms(n_gpus, current_message_gb, fabric_id, local_group, compression_ratio=1)
        for key, color in (("ring", COLORS["RedLine"]), ("tree", COLORS["BlueLine"]), ("hierarchical", COLORS["GreenLine"])):
            fig.add_trace(
                go.Scatter(
                    x=[current_message_gb],
                    y=[current_terms[key]["total_ms"]],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=f"current {current_terms[key]['label']}",
                    showlegend=False,
                )
            )
        fig.add_hline(y=lens["budget_ms"], line_dash="dash", line_color=COLORS["RedLine"], annotation_text="track budget")
        fig.update_layout(
            height=350,
            xaxis_title="Payload per participant (GB)",
            yaxis_title="Modeled time (ms)",
            xaxis_type="log",
            legend=dict(orientation="h", y=1.15, x=0),
            margin=dict(l=60, r=20, t=55, b=45),
        )
        return apply_plotly_theme(fig)

    def v2_06_optimization_chart(result):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Raw selected", "Compressed", "Exposed after overlap"],
                y=[result["raw_ms"], result["compressed_ms"], result["exposed_ms"]],
                marker_color=[COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"]],
                text=[f"{result['raw_ms']:.1f}", f"{result['compressed_ms']:.1f}", f"{result['exposed_ms']:.1f}"],
                textposition="outside",
            )
        )
        fig.add_hline(y=result["budget_ms"], line_dash="dash", line_color=COLORS["RedLine"], annotation_text="track budget")
        fig.update_layout(
            height=330,
            yaxis_title="Time (ms)",
            showlegend=False,
            margin=dict(l=60, r=20, t=35, b=45),
        )
        return apply_plotly_theme(fig)

    def v2_06_candidate_chart(selected, rejected):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Selected plan", "Rejected alternative"],
                y=[selected["exposed_ms"], rejected["exposed_ms"]],
                marker_color=[
                    COLORS["GreenLine"] if selected["valid_plan"] else COLORS["RedLine"],
                    COLORS["GreenLine"] if rejected["valid_plan"] else COLORS["RedLine"],
                ],
                text=[f"{selected['exposed_ms']:.1f} ms", f"{rejected['exposed_ms']:.1f} ms"],
                textposition="outside",
            )
        )
        fig.add_hline(y=selected["budget_ms"], line_dash="dash", line_color=COLORS["RedLine"], annotation_text="track budget")
        fig.update_layout(
            height=320,
            yaxis_title="Exposed time (ms)",
            showlegend=False,
            margin=dict(l=60, r=20, t=35, b=45),
        )
        return apply_plotly_theme(fig)

    return (
        v2_06_alpha_beta_chart,
        v2_06_best_topology_key,
        v2_06_candidate_chart,
        v2_06_collective_terms,
        v2_06_default_fabric_label,
        v2_06_fabric_label,
        v2_06_fabric_options,
        v2_06_failure_callout,
        v2_06_fmt,
        v2_06_frontier_chart,
        v2_06_html_table,
        v2_06_optimization_chart,
        v2_06_optimization_result,
        v2_06_part_a_rows,
        v2_06_plan_result,
        v2_06_prediction_feedback,
        v2_06_row_by_key,
        v2_06_selected_key,
        v2_06_status_label,
        v2_06_topology_rows,
        v2_06_track_lens,
    )


@app.cell
def _(v2_06_profile, v2_06_track_lens, v2_06_variant):
    v2_06_lens = v2_06_track_lens(v2_06_profile, v2_06_variant)
    return (v2_06_lens,)


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    track_arc_context,
    track_context,
    v2_06_defaults,
    v2_06_lens,
    v2_06_metadata,
    v2_06_profile,
    v2_06_variant,
):
    mo.vstack(
        [
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            mo.Html(
                f"""
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
                              max-width: 820px; line-height: 1.65;">
                        {v2_06_lens["scenario"]} Communication algorithm choice is a
                        systems decision, not a library detail.
                    </p>
                    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px;">
                        <span style="background: rgba(99,102,241,0.18); color:#a5b4fc;
                                     padding:5px 14px; border-radius:20px; font-size:0.8rem;
                                     font-weight:600; border:1px solid rgba(99,102,241,0.3);">
                            4 Concept Modules + Synthesis &middot; ~45 min
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
                """
            ),
            track_context(v2_06_profile),
            track_arc_context(v2_06_profile, v2_06_metadata.lab_id),
        ]
    )
    return


@app.cell(hide_code=True)
def _(COLORS, mo, source_trace, v2_06_defaults, v2_06_lens, v2_06_variant):
    mo.vstack(
        [
            mo.Html(
                f"""
                <div style="border-left:4px solid {COLORS['BlueLine']}; background:white;
                            border-radius:0 12px 12px 0; padding:20px 28px; margin:8px 0 16px 0;
                            box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                    <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                                text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                        Chapter Invariant
                    </div>
                    <div style="font-size:0.95rem; color:{COLORS['TextSec']}; line-height:1.7;">
                        Collectives are algorithms whose latency and bandwidth terms depend on topology,
                        payload, overlap, and compression. Your track keeps the same concept sequence
                        while changing the stakeholder, guardrails, evidence emphasis, and report frame.
                    </div>
                    <div style="border-top:1px solid {COLORS['Border']}; margin:14px -28px 0 -28px;
                                padding:14px 28px 0 28px;">
                        <div style="font-size:0.7rem; font-weight:700; color:{COLORS['BlueLine']};
                                    text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                            Starting Scenario
                        </div>
                        <div style="font-size:1.0rem; color:{COLORS['Text']}; font-weight:600; line-height:1.5;">
                            {v2_06_defaults["participants"]} {v2_06_lens["participant_name"]},
                            {v2_06_defaults["message_gb"]} GB per participant,
                            topology: {v2_06_defaults["topology"]}.
                        </div>
                    </div>
                </div>
                """
            ),
            mo.callout(
                mo.md(
                    """
                    **Recommended Reading** - Volume II, Chapter 6: Collective Communication.

                    Focus on alpha-beta cost models, Ring and Tree AllReduce, topology-aware
                    hierarchical communication, gradient compression, error feedback, and
                    communication-computation overlap.
                    """
                ),
                kind="info",
            ),
            source_trace(
                {
                    "chapter anchors": "collective_communication.qmd sections on alpha-beta, AllReduce algorithms, topology, compression, and overlap",
                    "shared solvers": "mlsysim.physics calc_ring_allreduce_time, calc_tree_allreduce_time, calc_hierarchical_allreduce_time",
                    "track defaults": v2_06_variant.scenario_id,
                    "local assumptions": "step-time budgets, quality/fidelity proxies, and overlap guardrails are notebook-local pedagogical assumptions",
                    "validation tests": ", ".join(v2_06_lens["validation_tests"]),
                },
                collapsed=True,
                summary="Source models and local assumptions",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, v2_06_defaults):
    partA_prediction = mo.ui.radio(
        options={
            "A) Ring is always best because it is bandwidth-optimal": "ring_always",
            "B) Tree is always best because it has fewer rounds": "tree_always",
            "C) The winner depends on alpha, beta, N, and payload size": "depends",
            "D) FLOPs dominate once the collective is chosen": "flops",
        },
        label="Part A prediction - which rule chooses between ring and tree costs?",
    )
    partB_prediction = mo.ui.radio(
        options={
            "A) Topology can make a collective infeasible or no longer dominant": "topology",
            "B) Hierarchy always helps regardless of local links": "hierarchy_always",
            "C) Topology changes diagrams but not collective time": "label_only",
            "D) Participant count matters but local grouping does not": "participants_only",
        },
        label=f"Part B prediction - what changes the dominant collective for {v2_06_defaults['topology']}?",
    )
    partC_prediction = mo.ui.radio(
        options={
            "A) Compression and overlap reduce exposed time but carry validation risk": "risk",
            "B) Compression is a free bandwidth multiplier": "free_compression",
            "C) Async overlap makes bandwidth irrelevant": "free_overlap",
            "D) Compression only changes optimizer math, not systems behavior": "optimizer_only",
        },
        label="Part C prediction - what risk remains after hiding or shrinking communication?",
    )
    partD_prediction = mo.ui.radio(
        options={
            "A) Exposed step-time rejects the naive plan": "exposed",
            "B) Topology guardrail rejects the naive plan": "topology",
            "C) Optimization risk rejects the naive plan": "optimization",
            "D) The fastest modeled plan should always be approved": "fastest",
        },
        label="Part D prediction - which guardrail will reject the naive alternative?",
    )

    partA_checkpoint = mo.ui.radio(
        options={
            "Carry ring forward for bandwidth-bound payloads": "ring_family",
            "Carry tree forward for alpha-bound payloads": "tree_family",
            "Do not lock yet; let topology decide": "defer_to_topology",
        },
        label="Part A checkpoint - which algorithm family should the next module test?",
    )
    partB_checkpoint = mo.ui.radio(
        options={
            "Use flat ring on the selected fabric": "flat_ring",
            "Use tree/coordinator scheduling": "tree_schedule",
            "Use hierarchical local-global scheduling": "hierarchical_schedule",
        },
        label="Part B checkpoint - which topology assumption should the plan carry forward?",
    )
    partC_checkpoint = mo.ui.radio(
        options={
            "No compression; only measured overlap": "measured_overlap",
            "Moderate compression with validation": "moderate_compression",
            "Aggressive compression/overlap only after convergence or fidelity evidence": "aggressive_with_evidence",
        },
        label="Part C checkpoint - which optimization policy should the design review use?",
    )
    partD_final_decision = mo.ui.radio(
        options={
            "Approve the selected communication plan": "approve",
            "Revise algorithm, topology, or optimization before approval": "revise",
            "Reject the plan and rerun topology/validation evidence": "reject",
        },
        label="Final decision - how should the stakeholder sign the communication design review?",
    )
    student_id = mo.ui.text(label="Student identifier", placeholder="Optional")
    memo_note = mo.ui.text_area(
        label="Communication design review note",
        placeholder="Name selected algorithm/topology/optimization, binding term, rejected alternative, evidence number, and V2-07 reliability implication.",
        full_width=True,
    )
    return (
        memo_note,
        partA_checkpoint,
        partA_prediction,
        partB_checkpoint,
        partB_prediction,
        partC_checkpoint,
        partC_prediction,
        partD_final_decision,
        partD_prediction,
        student_id,
    )


@app.cell(hide_code=True)
def _(mo, v2_06_default_fabric_label, v2_06_defaults, v2_06_fabric_options):
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
        label="Payload per participant (GB)",
    )
    fabric = mo.ui.dropdown(
        options=v2_06_fabric_options(),
        value=v2_06_default_fabric_label(v2_06_defaults),
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
        stop=95,
        value=int(v2_06_defaults["overlap_pct"]),
        step=5,
        label="Communication hidden by useful work (%)",
    )
    compression_ratio = mo.ui.slider(
        start=1,
        stop=16,
        value=int(v2_06_defaults["compression_ratio"]),
        step=1,
        label="Compression ratio",
    )
    return compression_ratio, fabric, gpus_per_node, message_gb, n_gpus, overlap_pct


@app.cell(hide_code=True)
def _(
    MathPeek,
    build_lab_report,
    compression_ratio,
    fabric,
    gpus_per_node,
    ledger,
    memo_note,
    message_gb,
    mo,
    n_gpus,
    overlap_pct,
    partA_checkpoint,
    partA_prediction,
    partB_checkpoint,
    partB_prediction,
    partC_checkpoint,
    partC_prediction,
    partD_final_decision,
    partD_prediction,
    report_export_panel,
    student_id,
    v2_06_alpha_beta_chart,
    v2_06_best_topology_key,
    v2_06_candidate_chart,
    v2_06_chapter,
    v2_06_collective_terms,
    v2_06_failure_callout,
    v2_06_fmt,
    v2_06_frontier_chart,
    v2_06_html_table,
    v2_06_lens,
    v2_06_metadata,
    v2_06_optimization_chart,
    v2_06_optimization_result,
    v2_06_part_a_rows,
    v2_06_plan_result,
    v2_06_prediction_feedback,
    v2_06_profile,
    v2_06_row_by_key,
    v2_06_selected_key,
    v2_06_status_label,
    v2_06_topology_rows,
    v2_06_variant,
):
    _n = n_gpus.value
    _payload = message_gb.value
    _fabric = fabric.value
    _local = gpus_per_node.value
    _compression = compression_ratio.value
    _overlap = overlap_pct.value

    _terms = v2_06_collective_terms(_n, _payload, _fabric, _local, compression_ratio=1)
    _topology_rows = v2_06_topology_rows(v2_06_lens, _terms)
    _part_b_key = v2_06_selected_key(partB_checkpoint.value, _topology_rows)
    _part_b_row = v2_06_row_by_key(_topology_rows, _part_b_key)
    _part_c = v2_06_optimization_result(
        v2_06_lens,
        _part_b_key,
        _n,
        _payload,
        _fabric,
        _local,
        _compression,
        _overlap,
    )
    _selected = v2_06_plan_result(
        v2_06_lens,
        _part_b_key,
        _n,
        _payload,
        _fabric,
        _local,
        _compression,
        _overlap,
    )
    _rejected = v2_06_plan_result(
        v2_06_lens,
        "ring",
        _n,
        _payload,
        _fabric,
        _local,
        max(8, _compression),
        max(85, _overlap),
    )

    def _part_a_table():
        return v2_06_html_table(
            v2_06_part_a_rows(v2_06_lens, _terms),
            [
                ("Algorithm", "Algorithm"),
                ("Alpha", "Alpha term"),
                ("Beta", "Beta term"),
                ("Total", "Total"),
                ("Binding", "Binding term"),
                ("Budget", "Budget status"),
            ],
            caption="Part A exact alpha/beta evidence table",
        )

    def _part_b_table():
        rows = []
        for row in _topology_rows:
            rows.append(
                {
                    "Candidate": row["label"],
                    "Time": f"{row['total_ms']:.3f} ms",
                    "Binding": row["binding"],
                    "Topology": v2_06_status_label(row["topology_ok"]),
                    "Budget": v2_06_status_label(row["passes_budget"]),
                    "Reason": row["reason"],
                }
            )
        return v2_06_html_table(
            rows,
            [
                ("Candidate", "Candidate"),
                ("Time", "Modeled time"),
                ("Binding", "Binding term"),
                ("Topology", "Topology guard"),
                ("Budget", "Budget"),
                ("Reason", "Interpretation"),
            ],
            caption="Part B topology feasibility and dominance table",
        )

    def _part_c_table():
        return v2_06_html_table(
            [
                {
                    "Metric": "Selected candidate",
                    "Value": _part_c["selected_label"],
                    "Limit or interpretation": "Candidate carried forward from Part B.",
                },
                {
                    "Metric": "Compressed payload",
                    "Value": f"{_part_c['effective_gb']:.4g} GB",
                    "Limit or interpretation": f"Original payload was {_payload:.4g} GB.",
                },
                {
                    "Metric": "Exposed time",
                    "Value": f"{_part_c['exposed_ms']:.3f} ms",
                    "Limit or interpretation": f"Must be <= {v2_06_lens['budget_ms']:.0f} ms.",
                },
                {
                    "Metric": "Quality/fidelity proxy",
                    "Value": f"{_part_c['quality_proxy']:.1f}%",
                    "Limit or interpretation": f"Must be >= {v2_06_lens['quality_floor_pct']:.0f}% and compression <= {v2_06_lens['max_compression_ratio']}x.",
                },
                {
                    "Metric": "Overlap schedule",
                    "Value": f"{_overlap:.0f}%",
                    "Limit or interpretation": f"Notebook-local schedulable limit is {v2_06_lens['max_overlap_pct']:.0f}%.",
                },
                {
                    "Metric": "Residual risk",
                    "Value": _part_c["risk"],
                    "Limit or interpretation": v2_06_lens["residual_risk"],
                },
            ],
            [("Metric", "Metric"), ("Value", "Value"), ("Limit or interpretation", "Limit or interpretation")],
            caption="Part C optimization evidence table",
        )

    def _part_d_table(selected, rejected):
        rows = [
            {
                "Guardrail": "Exposed step-time",
                "Selected": f"{selected['exposed_ms']:.3f} ms ({v2_06_status_label(selected['exposed_ok'])})",
                "Rejected": f"{rejected['exposed_ms']:.3f} ms ({v2_06_status_label(rejected['exposed_ok'])})",
                "Limit": f"<= {selected['budget_ms']:.0f} ms",
            },
            {
                "Guardrail": "Topology",
                "Selected": f"{v2_06_status_label(selected['topology_ok'])}: {selected['topology_reason']}",
                "Rejected": f"{v2_06_status_label(rejected['topology_ok'])}: {rejected['topology_reason']}",
                "Limit": v2_06_lens["topology_guardrail_text"],
            },
            {
                "Guardrail": "Optimization risk",
                "Selected": f"{v2_06_status_label(selected['optimization_ok'])}: {selected['risk']}",
                "Rejected": f"{v2_06_status_label(rejected['optimization_ok'])}: {rejected['risk']}",
                "Limit": v2_06_lens["optimization_guardrail_label"],
            },
        ]
        return v2_06_html_table(
            rows,
            [
                ("Guardrail", "Guardrail"),
                ("Selected", "Selected plan"),
                ("Rejected", "Rejected alternative"),
                ("Limit", "Limit or rule"),
            ],
            caption="Part D simultaneous guardrail table",
        )

    def build_part_a():
        _ring_ms = _terms["ring"]["total_ms"]
        _tree_ms = _terms["tree"]["total_ms"]
        _actual = "depends"
        _best_label = "Flat Ring" if _ring_ms <= _tree_ms else "Tree"
        _best_ms = min(_ring_ms, _tree_ms)
        items = [
            mo.md("## Part A - Concept Module: Ring And Tree Costs Bind Different Alpha/Beta Terms"),
            mo.callout(
                mo.md(
                    f"**Scenario.** {v2_06_variant.stakeholder} must choose a first-pass collective for "
                    f"**{_n} {v2_06_lens['participant_name']}** moving a **{_payload:.4g} GB "
                    f"{v2_06_lens['payload_name']}** over **{_terms['fabric_label']}**."
                ),
                kind="info",
            ),
            partA_prediction,
            v2_06_prediction_feedback(
                partA_prediction.value,
                _actual,
                f"Correct. At this point the lower cost is **{_best_label}** at **{_best_ms:.3f} ms**, but the reason depends on alpha, beta, N, and payload.",
                f"The productive failure is locking to one algorithm. Here **{_best_label}** is lower, but the stacked terms show why the answer can flip.",
            ),
        ]
        if partA_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([n_gpus, message_gb, fabric], justify="start"),
                mo.as_html(v2_06_alpha_beta_chart(_terms, v2_06_lens["budget_ms"])),
                _part_a_table(),
                v2_06_failure_callout(
                    _best_ms <= v2_06_lens["budget_ms"],
                    f"Consequence: the best ring/tree choice fits the {v2_06_lens['budget_ms']:.0f} ms track budget before topology mitigation.",
                    f"Boundary: even the best ring/tree choice is {_best_ms:.3f} ms against a {v2_06_lens['budget_ms']:.0f} ms budget. The next module must test topology, not just algorithm label.",
                ),
                MathPeek(
                    "T_ring=2(N-1)alpha+2((N-1)/N)M/beta; T_tree~=2log2(N)alpha+2log2(N)M/beta",
                    {
                        "N": f"{_n}",
                        "payload M": f"{_payload:.4g} GB",
                        "fabric beta": f"{_terms['bandwidth_gb_s']:.1f} GB/s",
                        "fabric alpha": f"{_terms['latency_us']:.3f} us",
                        "chapter source": "Alpha-beta model and AllReduce algorithm crossover sections",
                    },
                ),
                partA_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_part_b():
        _actual = "topology"
        _best_key = v2_06_best_topology_key(_topology_rows)
        _best_row = v2_06_row_by_key(_topology_rows, _best_key)
        items = [
            mo.md("## Part B - Concept Module: Topology Changes Which Collective Is Feasible Or Dominant"),
            mo.callout(
                mo.md(
                    f"**Scenario.** The Part A amount now has to run through the selected track topology: "
                    f"**{v2_06_lens['topology_label']}**. Test whether local grouping, fabric, and "
                    "topology assumptions change the winning collective."
                ),
                kind="info",
            ),
            partB_prediction,
            v2_06_prediction_feedback(
                partB_prediction.value,
                _actual,
                f"Correct. The feasible dominant candidate is **{_best_row['label']}** at **{_best_row['total_ms']:.3f} ms**.",
                "Topology is not a label. The feasibility table shows when flat, tree, or hierarchy depends on a valid physical or track analogy.",
            ),
        ]
        if partB_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([n_gpus, gpus_per_node, fabric], justify="start"),
                mo.as_html(v2_06_frontier_chart(v2_06_lens, _n, _fabric, _local, _payload)),
                _part_b_table(),
                v2_06_failure_callout(
                    _part_b_row["topology_ok"] and _part_b_row["passes_budget"],
                    f"Consequence: **{_part_b_row['label']}** is a valid topology assumption and reports {_part_b_row['total_ms']:.3f} ms.",
                    f"Boundary: **{_part_b_row['label']}** is not ready. Topology status is {v2_06_status_label(_part_b_row['topology_ok'])}; budget status is {v2_06_status_label(_part_b_row['passes_budget'])}.",
                ),
                MathPeek(
                    "Hierarchical time = local reduce-scatter + inter-node AllReduce(M/G) + local allgather",
                    {
                        "local group G": f"{_local}",
                        "nodes/groups": f"{_terms['n_nodes']}",
                        "current hierarchy time": f"{_terms['hierarchical']['total_ms']:.3f} ms",
                        "topology guardrail": v2_06_lens["topology_guardrail_text"],
                        "chapter source": "Hierarchical AllReduce and topology-aware routing sections",
                    },
                ),
                partB_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_part_c():
        _actual = "risk"
        items = [
            mo.md("## Part C - Concept Module: Overlap And Compression Hide Communication With Risk"),
            mo.callout(
                mo.md(
                    f"**Scenario.** The design now tries to reduce exposed time for **{_part_c['selected_label']}**. "
                    "Compression shrinks the payload; overlap hides only communication that has useful work beside it."
                ),
                kind="info",
            ),
            partC_prediction,
            v2_06_prediction_feedback(
                partC_prediction.value,
                _actual,
                f"Correct. Exposed time is **{_part_c['exposed_ms']:.3f} ms**, but the residual risk is **{_part_c['risk']}**.",
                "The productive failure is treating speedup as proof. The table checks exposed time, quality/fidelity, and scheduling together.",
            ),
        ]
        if partC_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([compression_ratio, overlap_pct], justify="start"),
                mo.as_html(v2_06_optimization_chart({**_part_c, "budget_ms": v2_06_lens["budget_ms"]})),
                _part_c_table(),
                v2_06_failure_callout(
                    _part_c["exposed_ok"] and _part_c["optimization_ok"],
                    f"Consequence: optimization passes the time and validation guardrails; {_part_c['risk']}.",
                    f"Boundary: optimization is not ready. Exposed-time status is {v2_06_status_label(_part_c['exposed_ok'])}; optimization guardrail is {v2_06_status_label(_part_c['optimization_ok'])}.",
                ),
                MathPeek(
                    "exposed = T_compressed*(1-overlap); compression changes M, not convergence proof",
                    {
                        "raw selected time": f"{_part_c['raw_ms']:.3f} ms",
                        "compressed selected time": f"{_part_c['compressed_ms']:.3f} ms",
                        "exposed time": f"{_part_c['exposed_ms']:.3f} ms",
                        "error feedback source model": "e_{t+1}=(g_t+e_t)-v_t",
                        "chapter source": "Gradient compression, error feedback, and overlap limits sections",
                    },
                ),
                partC_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_part_d():
        _actual_map = {
            "exposed step-time": "exposed",
            "topology": "topology",
            "optimization risk": "optimization",
        }
        _actual = _actual_map.get(_rejected["binding_guardrail"], "optimization")
        items = [
            mo.md("## Part D - Concept Module: Communication Plan Guardrails"),
            mo.callout(
                mo.md(
                    f"**Scenario.** The final design review must satisfy exposed step-time, topology, "
                    f"and {v2_06_lens['optimization_guardrail_label']} guardrails for {v2_06_profile.label}. "
                    f"{v2_06_lens['decision_frame']}"
                ),
                kind="info",
            ),
            partD_prediction,
            v2_06_prediction_feedback(
                partD_prediction.value,
                _actual,
                f"Correct. The rejected alternative is blocked by **{_rejected['binding_guardrail']}**.",
                f"The rejected alternative is blocked by **{_rejected['binding_guardrail']}**. Fastest modeled time is not enough evidence to approve a collective plan.",
            ),
        ]
        if partD_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([gpus_per_node, compression_ratio, overlap_pct], justify="start"),
                mo.as_html(v2_06_candidate_chart(_selected, _rejected)),
                _part_d_table(_selected, _rejected),
                v2_06_failure_callout(
                    _selected["valid_plan"],
                    f"Consequence: selected plan passes all guardrails. Binding term is **{_selected['binding_guardrail']}**.",
                    f"Boundary: selected plan fails at least one guardrail. Binding guardrail is **{_selected['binding_guardrail']}**; revise algorithm, topology, compression, or overlap.",
                ),
                MathPeek(
                    "valid = exposed_ms<=budget and topology_guardrail and optimization_guardrail",
                    {
                        "selected exposed": f"{_selected['exposed_ms']:.3f} ms vs {v2_06_fmt(_selected['budget_ms'])} ms",
                        "topology": v2_06_status_label(_selected["topology_ok"]),
                        "optimization": v2_06_status_label(_selected["optimization_ok"]),
                        "rejected alternative": "flat ring with aggressive compression and optimistic overlap",
                        "chapter source": "Fallacies, pitfalls, and chapter summary",
                    },
                ),
                partD_final_decision,
            ]
        )
        return mo.vstack(items)

    def build_synthesis():
        _completed = all(
            value is not None
            for value in (
                partA_prediction.value,
                partA_checkpoint.value,
                partB_prediction.value,
                partB_checkpoint.value,
                partC_prediction.value,
                partC_checkpoint.value,
                partD_prediction.value,
                partD_final_decision.value,
            )
        )
        _selected_optimization = f"{_compression}x compression, {_overlap:.0f}% overlap"
        _memo = memo_note.value or (
            f"Use {_selected['selected_label']} on {_selected['fabric_label']} with local group {_selected['local_group']}; "
            f"optimization: {_selected_optimization}; binding term: {_selected['binding_guardrail']}; "
            f"reject flat ring with aggressive compression/overlap because {_rejected['binding_guardrail']} fails; "
            f"V2-07 implication: {v2_06_lens['reliability_implication']}"
        )
        _snapshot = {
            "track_id": v2_06_profile.track_id,
            "scenario_id": v2_06_variant.scenario_id,
            "participants": _n,
            "payload_gb": _payload,
            "fabric": _selected["fabric_label"],
            "local_group": _local,
            "selected_algorithm": _selected["selected_label"],
            "selected_topology": _part_b_row["reason"],
            "selected_optimization": _selected_optimization,
            "binding_term": _selected["binding_guardrail"],
            "exposed_ms": round(_selected["exposed_ms"], 4),
            "budget_ms": _selected["budget_ms"],
            "topology_guardrail": _selected["topology_ok"],
            "optimization_guardrail": _selected["optimization_ok"],
            "valid_plan": _selected["valid_plan"],
            "rejected_alternative": "flat ring with aggressive compression and optimistic overlap",
            "rejected_binding": _rejected["binding_guardrail"],
            "v2_07_reliability_implication": v2_06_lens["reliability_implication"],
            "completed": _completed,
        }
        _design = {
            "lab_id": v2_06_metadata.lab_id,
            "track_id": v2_06_profile.track_id,
            "scenario_id": v2_06_variant.scenario_id,
            "selected_algorithm": _selected["selected_label"],
            "selected_topology": _part_b_row["reason"],
            "selected_optimization": _selected_optimization,
            "binding_term": _selected["binding_guardrail"],
            "exposed_ms": _selected["exposed_ms"],
            "budget_ms": _selected["budget_ms"],
            "topology_guardrail": _selected["topology_ok"],
            "optimization_guardrail": _selected["optimization_ok"],
            "rejected_alternative": "flat ring with aggressive compression and optimistic overlap",
            "v2_07_reliability_implication": v2_06_lens["reliability_implication"],
            "completed": _completed,
            "result_snapshot": _snapshot,
        }
        ledger.save(track=v2_06_profile.track_id, chapter=v2_06_chapter, design=_design)

        _incomplete = []
        if partA_prediction.value is None:
            _incomplete.append("Part A ring/tree prediction")
        if partA_checkpoint.value is None:
            _incomplete.append("Part A algorithm checkpoint")
        if partB_prediction.value is None:
            _incomplete.append("Part B topology prediction")
        if partB_checkpoint.value is None:
            _incomplete.append("Part B topology checkpoint")
        if partC_prediction.value is None:
            _incomplete.append("Part C optimization prediction")
        if partC_checkpoint.value is None:
            _incomplete.append("Part C optimization checkpoint")
        if partD_prediction.value is None:
            _incomplete.append("Part D guardrail prediction")
        if partD_final_decision.value is None:
            _incomplete.append("Final communication design decision")

        _report = build_lab_report(
            v2_06_metadata,
            student_id=student_id.value or "",
            track=v2_06_profile.label,
            scenario=v2_06_lens["scenario"],
            learning_objectives=(
                "Decompose ring and tree collective costs into alpha and beta terms.",
                "Use topology to decide which collective is feasible or dominant.",
                "Evaluate overlap and compression as conditional optimizations with residual risk.",
                "Approve a communication plan only when exposed time, topology, and optimization guardrails pass.",
            ),
            predictions={
                "partA_ring_tree": partA_prediction.value,
                "partB_topology": partB_prediction.value,
                "partC_overlap_compression": partC_prediction.value,
                "partD_guardrail": partD_prediction.value,
            },
            knob_settings={
                "participants": _n,
                "payload_gb": _payload,
                "fabric": _selected["fabric_label"],
                "local_group": _local,
                "compression_ratio": _compression,
                "overlap_pct": _overlap,
            },
            binding_constraints={
                "binding_term": _selected["binding_guardrail"],
                "exposed_ms": round(_selected["exposed_ms"], 4),
                "budget_ms": _selected["budget_ms"],
                "valid_plan": _selected["valid_plan"],
            },
            evidence_summary={
                "ring_ms": round(_terms["ring"]["total_ms"], 4),
                "tree_ms": round(_terms["tree"]["total_ms"], 4),
                "selected_algorithm": _selected["selected_label"],
                "selected_exposed_ms": round(_selected["exposed_ms"], 4),
                "rejected_binding": _rejected["binding_guardrail"],
                "validation_focus": v2_06_lens["validation_focus"],
            },
            decisions={
                "algorithm_checkpoint": partA_checkpoint.value,
                "topology_checkpoint": partB_checkpoint.value,
                "optimization_checkpoint": partC_checkpoint.value,
                "final_decision": partD_final_decision.value,
            },
            reflections={"communication_design_review_note": memo_note.value},
            final_decision=_memo,
            big_takeaways=(
                "Ring and tree costs differ because their alpha and beta terms scale differently.",
                "Topology changes whether a collective is feasible and whether its modeled advantage is real.",
                "Overlap and compression reduce exposed communication only inside validation guardrails.",
                "The V2-06 decision becomes a V2-07 reliability obligation.",
            ),
            residual_risk=v2_06_lens["residual_risk"],
            source_trace={
                "chapter_anchor": "Volume II, Chapter 6: Collective Communication",
                "source_models": "alpha/beta ring/tree, hierarchical AllReduce, exposed overlap, compression risk",
                "mlsysim_functions": "calc_ring_allreduce_time, calc_tree_allreduce_time, calc_hierarchical_allreduce_time",
                "track_source_policy": v2_06_profile.source_policy,
                "scenario_assumptions": "track budgets, quality proxies, and overlap guardrails are notebook-local pedagogical assumptions",
            },
            result_snapshot=_snapshot,
            incomplete_fields=tuple(_incomplete),
        )

        return mo.vstack(
            [
                mo.md("## Synthesis - Collective Communication Design Review"),
                student_id,
                memo_note,
                mo.callout(
                    mo.md(
                        f"**Selected algorithm/topology/optimization:** {_selected['selected_label']} "
                        f"on {_selected['fabric_label']} with local group {_selected['local_group']}; {_selected_optimization}.\n\n"
                        f"**Binding term or guardrail:** {_selected['binding_guardrail']}.\n\n"
                        f"**Rejected alternative:** flat ring with aggressive compression and optimistic overlap "
                        f"(blocked by {_rejected['binding_guardrail']}).\n\n"
                        f"**V2-07 reliability implication:** {v2_06_lens['reliability_implication']}"
                    ),
                    kind="success" if _selected["valid_plan"] else "warn",
                ),
                partD_final_decision,
                report_export_panel(_report),
            ]
        )

    v2_06_tabs = mo.ui.tabs(
        {
            "Part A: Alpha/Beta": build_part_a(),
            "Part B: Topology": build_part_b(),
            "Part C: Overlap/Compression": build_part_c(),
            "Part D: Guardrails": build_part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    v2_06_tabs
    return


@app.cell(hide_code=True)
def _(mo, v2_06_metadata, v2_06_profile):
    mo.Html(
        f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">{v2_06_metadata.lab_id}</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v2_06_profile.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">STATUS</span>
            <span class="hud-active">ACTIVE</span>
        </div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
