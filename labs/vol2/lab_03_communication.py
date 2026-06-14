import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
async def _():
    import html as html_lib
    import sys
    from pathlib import Path

    import marimo as mo

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
    from mlsysim import Hardware, Systems, ureg
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.physics import (
        calc_alpha_beta_crossover,
        calc_bisection_bandwidth,
        calc_oversubscription_effect,
        calc_point_to_point_time,
    )
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        MathPeek,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
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
        calc_alpha_beta_crossover,
        calc_bisection_bandwidth,
        calc_oversubscription_effect,
        calc_point_to_point_time,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        html_lib,
        ledger,
        mo,
        report_export_panel,
        source_trace,
        track_arc_context,
        track_context,
        track_selector,
        ureg,
    )


@app.cell
def _(get_lab_metadata):
    v2_03_lab_path = "vol2/lab_03_communication.py"
    v2_03_chapter = 3
    v2_03_metadata = get_lab_metadata(v2_03_lab_path)
    return v2_03_chapter, v2_03_lab_path, v2_03_metadata


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_03_track_picker = track_selector(default=_default_track)
    v2_03_track_picker
    return (v2_03_track_picker,)


@app.cell
def _(get_lab_track_variant, get_track_profile, v2_03_metadata, v2_03_track_picker):
    v2_03_track_id = v2_03_track_picker.value
    v2_03_profile = get_track_profile(v2_03_track_id)
    v2_03_variant = get_lab_track_variant(v2_03_metadata.lab_id, v2_03_profile.track_id)
    return v2_03_profile, v2_03_track_id, v2_03_variant


@app.cell
def _(
    Hardware,
    Systems,
    apply_plotly_theme,
    calc_alpha_beta_crossover,
    calc_bisection_bandwidth,
    calc_oversubscription_effect,
    calc_point_to_point_time,
    go,
    html_lib,
    mo,
    ureg,
):
    def v2_03_to_ms(quantity):
        return float(quantity.to(ureg.millisecond).magnitude)

    def v2_03_to_mb_s(quantity):
        return float(quantity.to(ureg.MB / ureg.second).magnitude)

    def v2_03_fmt(value, digits=2):
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            if abs(value) >= 100:
                return f"{value:,.0f}"
            if abs(value) >= 10:
                return f"{value:,.1f}"
            return f"{value:,.{digits}f}"
        return str(value)

    def v2_03_track_lenses(Systems, Hardware, ureg):
        h100_nvlink = Hardware.Cloud.H100.nvlink
        nvlink_alpha = getattr(h100_nvlink, "latency", None) or 0.5 * ureg.microsecond
        ib_ndr = Systems.Fabrics.InfiniBand_NDR
        eth_100g = Systems.Fabrics.Ethernet_100G

        return {
            "iphone": {
                "scenario": "A mobile product team is deciding which payloads can cross the device-edge-cloud path without harming responsiveness or privacy posture.",
                "decision_frame": "Decide what stays local, what can use edge sync, and what must be staged outside the interaction path.",
                "payload_name": "privacy-safe telemetry or offload payload",
                "participant_name": "phones in a regional cohort",
                "message_default_mb": 8,
                "message_min_mb": 1,
                "message_max_mb": 160,
                "message_step_mb": 1,
                "participants_default": 80,
                "participants_min": 4,
                "participants_max": 500,
                "communication_budget_ms": 120,
                "step_budget_ms": 180,
                "utilization_limit": 0.70,
                "failure_mode": "responsiveness or privacy-routing miss",
                "guardrail_text": "Per-interaction payloads should stay edge-local unless they are staged or summarized.",
                "report_prompt": "Frame the memo around local-vs-networked payload policy.",
                "collective_implication": "V2-06 collectives should assume phone-originated payloads are summarized before any fleet aggregation.",
                "links": {
                    "wifi_edge": {
                        "label": "Wi-Fi to edge point of presence",
                        "alpha": 8 * ureg.millisecond,
                        "bandwidth": 90 * ureg.MB / ureg.second,
                        "source": "local scenario assumption for low-latency edge path",
                    },
                    "5g_edge": {
                        "label": "5G uplink to edge",
                        "alpha": 28 * ureg.millisecond,
                        "bandwidth": 18 * ureg.MB / ureg.second,
                        "source": "local scenario assumption for mobile uplink",
                    },
                    "cloud_wan": {
                        "label": "Direct cloud WAN path",
                        "alpha": 70 * ureg.millisecond,
                        "bandwidth": 10 * ureg.MB / ureg.second,
                        "source": "local scenario assumption for cloud round trip",
                    },
                },
                "default_link": "wifi_edge",
                "topology_labels": {
                    "nonblocking": "Edge-local fan-in",
                    "aligned": "Region-sharded edge mesh",
                    "grouped": "Phone-to-edge-to-cloud staging",
                    "oversubscribed": "Single shared cloud ingress",
                },
                "placement_labels": {
                    "affinity": "Keep cohort near its edge region",
                    "balanced": "Balance across nearby edge regions",
                    "spread": "Route by cheapest cloud capacity",
                    "noisy": "Share ingress with analytics upload",
                },
                "hop_latency": 3 * ureg.millisecond,
            },
            "oura_ring": {
                "scenario": "A wearable firmware team must move sync windows and OTA payloads across intermittent ring-phone-cloud connectivity.",
                "decision_frame": "Choose a sync/update policy that survives a short connection window without stealing sensing duty cycle.",
                "payload_name": "BLE sync or OTA payload",
                "participant_name": "rings syncing through phones",
                "message_default_mb": 6,
                "message_min_mb": 1,
                "message_max_mb": 48,
                "message_step_mb": 1,
                "participants_default": 64,
                "participants_min": 4,
                "participants_max": 300,
                "communication_budget_ms": 120000,
                "step_budget_ms": 180000,
                "utilization_limit": 0.55,
                "failure_mode": "sync-window or OTA budget miss",
                "guardrail_text": "Radio transfer must fit an intermittent phone-nearby window.",
                "report_prompt": "Frame the memo around sync cadence, OTA staging, and payload minimization.",
                "collective_implication": "V2-06 collectives should assume wearable updates are staged and delay-tolerant, not synchronous.",
                "links": {
                    "ble_phone": {
                        "label": "BLE ring-to-phone",
                        "alpha": 90 * ureg.millisecond,
                        "bandwidth": 0.18 * ureg.MB / ureg.second,
                        "source": "local scenario assumption for intermittent BLE transfer",
                    },
                    "phone_wifi": {
                        "label": "Phone Wi-Fi relay",
                        "alpha": 35 * ureg.millisecond,
                        "bandwidth": 12 * ureg.MB / ureg.second,
                        "source": "local scenario assumption for phone relay",
                    },
                    "cellular_relay": {
                        "label": "Phone cellular relay",
                        "alpha": 80 * ureg.millisecond,
                        "bandwidth": 4 * ureg.MB / ureg.second,
                        "source": "local scenario assumption for cellular relay",
                    },
                },
                "default_link": "ble_phone",
                "topology_labels": {
                    "nonblocking": "Ring-phone local sync",
                    "aligned": "Phone batches by sleep window",
                    "grouped": "Daily cloud staging group",
                    "oversubscribed": "Fleet-wide OTA push",
                },
                "placement_labels": {
                    "affinity": "Sync only when phone is nearby",
                    "balanced": "Batch by charging/sleep window",
                    "spread": "Opportunistic background relay",
                    "noisy": "OTA and sensing upload collide",
                },
                "hop_latency": 40 * ureg.millisecond,
            },
            "robotaxi": {
                "scenario": "An autonomous vehicle platform team must keep sensor-fabric traffic local while triaging upload of fleet evidence.",
                "decision_frame": "Decide which communication must stay vehicle-local and which evidence can be delayed to depot or cloud.",
                "payload_name": "sensor burst or triaged event payload",
                "participant_name": "vehicles or sensor endpoints",
                "message_default_mb": 96,
                "message_min_mb": 4,
                "message_max_mb": 1024,
                "message_step_mb": 4,
                "participants_default": 32,
                "participants_min": 4,
                "participants_max": 256,
                "communication_budget_ms": 60,
                "step_budget_ms": 90,
                "utilization_limit": 0.62,
                "failure_mode": "safety-latency or event-upload backlog miss",
                "guardrail_text": "Safety-critical sensor exchange must stay on the vehicle-local fabric.",
                "report_prompt": "Frame the memo around vehicle-local placement and delayed fleet upload.",
                "collective_implication": "V2-06 collectives should treat safety data as local-first and fleet evidence as asynchronous.",
                "links": {
                    "vehicle_fabric": {
                        "label": "Vehicle-local sensor fabric",
                        "alpha": 0.7 * ureg.millisecond,
                        "bandwidth": 9500 * ureg.MB / ureg.second,
                        "source": "local scenario assumption for in-vehicle high-speed fabric",
                    },
                    "depot_wifi": {
                        "label": "Depot Wi-Fi offload",
                        "alpha": 18 * ureg.millisecond,
                        "bandwidth": 75 * ureg.MB / ureg.second,
                        "source": "local scenario assumption for depot upload",
                    },
                    "cellular_upload": {
                        "label": "Cellular fleet upload",
                        "alpha": 55 * ureg.millisecond,
                        "bandwidth": 6 * ureg.MB / ureg.second,
                        "source": "local scenario assumption for on-road upload",
                    },
                },
                "default_link": "vehicle_fabric",
                "topology_labels": {
                    "nonblocking": "Vehicle-local switched fabric",
                    "aligned": "Sensor-domain aligned fabric",
                    "grouped": "Depot batch groups",
                    "oversubscribed": "Cloud-first event upload",
                },
                "placement_labels": {
                    "affinity": "Keep safety loop in-vehicle",
                    "balanced": "Upload triaged events at depot",
                    "spread": "Stream selected evidence to cloud",
                    "noisy": "Share link with map and log upload",
                },
                "hop_latency": 0.25 * ureg.millisecond,
            },
            "cloud_fleet": {
                "scenario": "A fleet service owner must choose a GPU fabric and placement policy for synchronous distributed work.",
                "decision_frame": "Choose the topology and placement policy that turns purchased accelerators into useful parallel work.",
                "payload_name": "gradient shard or activation payload",
                "participant_name": "accelerators",
                "message_default_mb": 350,
                "message_min_mb": 4,
                "message_max_mb": 4096,
                "message_step_mb": 4,
                "participants_default": 64,
                "participants_min": 4,
                "participants_max": 1024,
                "communication_budget_ms": 80,
                "step_budget_ms": 120,
                "utilization_limit": 0.80,
                "failure_mode": "SLO breach, queue growth, or idle accelerators",
                "guardrail_text": "Synchronous jobs need topology-aware placement and enough bisection bandwidth.",
                "report_prompt": "Frame the memo around fabric topology, bisection, and placement policy.",
                "collective_implication": "V2-06 collectives should choose algorithms that match the selected bisection and placement envelope.",
                "links": {
                    "ib_ndr": {
                        "label": "InfiniBand NDR",
                        "alpha": Systems.SwitchFabric.AlphaNdr,
                        "bandwidth": ib_ndr.bandwidth,
                        "source": "MLSysIM Systems.Fabrics.InfiniBand_NDR and SwitchFabric.AlphaNdr",
                    },
                    "roce_100g": {
                        "label": "RoCE over 100G Ethernet",
                        "alpha": Systems.SwitchFabric.AlphaRoce,
                        "bandwidth": eth_100g.bandwidth,
                        "source": "MLSysIM Systems.Fabrics.Ethernet_100G and SwitchFabric.AlphaRoce",
                    },
                    "nvlink_local": {
                        "label": "NVLink local domain",
                        "alpha": nvlink_alpha,
                        "bandwidth": h100_nvlink.bandwidth_per_direction,
                        "source": "MLSysIM Hardware.Cloud.H100.nvlink",
                    },
                },
                "default_link": "ib_ndr",
                "topology_labels": {
                    "nonblocking": "Non-blocking fat-tree",
                    "aligned": "Rail-optimized placement",
                    "grouped": "Dragonfly/group-local job",
                    "oversubscribed": "4:1 oversubscribed spine",
                },
                "placement_labels": {
                    "affinity": "Pack job inside one pod/rail group",
                    "balanced": "Topology-aware multi-pod spread",
                    "spread": "Capacity-first scheduler spread",
                    "noisy": "Co-tenant checkpoint traffic",
                },
                "hop_latency": Systems.SwitchFabric.HopLatency,
            },
        }

    def v2_03_topology_options(lens):
        labels = lens["topology_labels"]
        return {
            "nonblocking": {
                "label": labels["nonblocking"],
                "oversubscription": 1.0,
                "bisection_factor": 1.0,
                "cross_fraction": 1.0,
                "hop_count": 2,
                "guard_ok": True,
                "note": "Full bisection for arbitrary communication.",
            },
            "aligned": {
                "label": labels["aligned"],
                "oversubscription": 1.2,
                "bisection_factor": 0.85,
                "cross_fraction": 0.62,
                "hop_count": 1,
                "guard_ok": True,
                "note": "Topology matches the dominant local/same-rank traffic.",
            },
            "grouped": {
                "label": labels["grouped"],
                "oversubscription": 2.2,
                "bisection_factor": 0.58,
                "cross_fraction": 0.78,
                "hop_count": 2,
                "guard_ok": True,
                "note": "Works when the job fits inside groups and avoids global links.",
            },
            "oversubscribed": {
                "label": labels["oversubscribed"],
                "oversubscription": 4.0,
                "bisection_factor": 0.45,
                "cross_fraction": 1.0,
                "hop_count": 4,
                "guard_ok": False,
                "note": "Cheap shared path; narrow cut throttles synchronous traffic.",
            },
        }

    def v2_03_placement_options(lens):
        labels = lens["placement_labels"]
        return {
            "affinity": {
                "label": labels["affinity"],
                "cross_factor": 0.52,
                "background_load": 0.05,
                "guard_ok": True,
                "note": "Keeps most communication inside the natural local domain.",
            },
            "balanced": {
                "label": labels["balanced"],
                "cross_factor": 0.74,
                "background_load": 0.14,
                "guard_ok": True,
                "note": "Uses more fabric while staying topology-aware.",
            },
            "spread": {
                "label": labels["spread"],
                "cross_factor": 1.08,
                "background_load": 0.28,
                "guard_ok": False,
                "note": "Local scheduling looks efficient but crosses the expensive cut.",
            },
            "noisy": {
                "label": labels["noisy"],
                "cross_factor": 1.18,
                "background_load": 0.46,
                "guard_ok": False,
                "note": "A neighboring or secondary flow shares the same bottleneck.",
            },
        }

    def v2_03_option_labels(options):
        return {option["label"]: key for key, option in options.items()}

    def v2_03_link_option_labels(lens):
        return {link["label"]: key for key, link in lens["links"].items()}

    def v2_03_alpha_beta_result(lens, link_id, payload_mb):
        link = lens["links"][link_id]
        payload = float(payload_mb) * ureg.MB
        alpha = link["alpha"]
        beta = link["bandwidth"]
        beta_term = (payload / beta).to(ureg.second)
        total = calc_point_to_point_time(payload, alpha, beta)
        crossover = calc_alpha_beta_crossover(alpha, beta)
        alpha_ms = v2_03_to_ms(alpha)
        beta_ms = v2_03_to_ms(beta_term)
        total_ms = v2_03_to_ms(total)
        binding = "latency (alpha)" if alpha_ms >= beta_ms else "bandwidth (n/beta)"
        return {
            "link_id": link_id,
            "link_label": link["label"],
            "payload_mb": float(payload_mb),
            "alpha_ms": alpha_ms,
            "beta_ms": beta_ms,
            "total_ms": total_ms,
            "bandwidth_mb_s": v2_03_to_mb_s(beta),
            "crossover_kb": float(crossover.to(ureg.KB).magnitude),
            "binding_term": binding,
            "budget_ms": lens["communication_budget_ms"],
            "passes_budget": total_ms <= lens["communication_budget_ms"],
            "source": link["source"],
        }

    def v2_03_evaluate_topologies(lens, link_id, payload_mb, participants):
        link = lens["links"][link_id]
        payload = float(payload_mb) * ureg.MB
        n = max(2, int(participants))
        ports = max(1, n // 2)
        rows = []
        for topology_id, topology in v2_03_topology_options(lens).items():
            effective_oversub = topology["oversubscription"] / topology["bisection_factor"]
            _relative_throughput, throughput_loss = calc_oversubscription_effect(
                0.30,
                effective_oversub,
            )
            bisection = calc_bisection_bandwidth(
                ports,
                link["bandwidth"],
                oversubscription_ratio=effective_oversub,
            )
            cross_cut_data = payload * ports * topology["cross_fraction"]
            hop_penalty = topology["hop_count"] * lens["hop_latency"]
            sync_time = (link["alpha"] + hop_penalty + cross_cut_data / bisection).to(ureg.second)
            sync_ms = v2_03_to_ms(sync_time)
            rows.append(
                {
                    "topology_id": topology_id,
                    "label": topology["label"],
                    "bisection_mb_s": v2_03_to_mb_s(bisection),
                    "sync_ms": sync_ms,
                    "step_budget_ms": lens["step_budget_ms"],
                    "passes_step": sync_ms <= lens["step_budget_ms"],
                    "guard_ok": topology["guard_ok"],
                    "hop_count": topology["hop_count"],
                    "oversubscription": effective_oversub,
                    "throughput_loss": throughput_loss,
                    "note": topology["note"],
                }
            )
        return rows

    def v2_03_selected_topology_result(lens, link_id, payload_mb, participants, topology_id):
        rows = v2_03_evaluate_topologies(lens, link_id, payload_mb, participants)
        return next(row for row in rows if row["topology_id"] == topology_id)

    def v2_03_congestion_result(lens, link_id, payload_mb, participants, topology_id, placement_id, burst_multiplier):
        topology_result = v2_03_selected_topology_result(lens, link_id, payload_mb, participants, topology_id)
        placement = v2_03_placement_options(lens)[placement_id]
        step_seconds = lens["step_budget_ms"] / 1000
        payload_mb = float(payload_mb)
        n = max(2, int(participants))
        offered_mb_s = payload_mb * n * placement["cross_factor"] * float(burst_multiplier) / max(step_seconds, 1e-9)
        capacity_mb_s = max(topology_result["bisection_mb_s"], 1e-9)
        utilization = offered_mb_s / capacity_mb_s + placement["background_load"]
        if utilization < 0.70:
            tail_multiplier = 1 + 0.35 * utilization
        elif utilization < 1.0:
            tail_multiplier = min(8.0, 1 + (utilization - 0.70) / max(1.0 - utilization, 0.02))
        else:
            tail_multiplier = min(20.0, 8.0 + 4.0 * (utilization - 1.0))
        tail_ms = topology_result["sync_ms"] * tail_multiplier
        utilization_limit = lens["utilization_limit"]
        utilization_ok = utilization <= utilization_limit
        topology_ok = topology_result["guard_ok"] and placement["guard_ok"]
        if not topology_ok:
            binding = "placement/topology guardrail"
        elif not utilization_ok:
            binding = "congestion utilization"
        else:
            binding = "bisection headroom"
        return {
            "placement_id": placement_id,
            "placement_label": placement["label"],
            "topology_id": topology_id,
            "topology_label": topology_result["label"],
            "offered_mb_s": offered_mb_s,
            "capacity_mb_s": capacity_mb_s,
            "utilization": utilization,
            "utilization_limit": utilization_limit,
            "utilization_ok": utilization_ok,
            "tail_multiplier": tail_multiplier,
            "tail_ms": tail_ms,
            "topology_ok": topology_ok,
            "placement_guard_ok": placement["guard_ok"],
            "failure_state": "OK" if utilization_ok and topology_ok else lens["failure_mode"],
            "binding_amount": binding,
            "placement_note": placement["note"],
        }

    def v2_03_decision_result(
        lens,
        link_id,
        payload_mb,
        participants,
        topology_id,
        placement_id,
        burst_multiplier,
        payload_reduction_pct,
        overlap_pct,
    ):
        reduced_payload_mb = float(payload_mb) * (1 - float(payload_reduction_pct) / 100)
        reduced_payload_mb = max(reduced_payload_mb, 0.05)
        congestion = v2_03_congestion_result(
            lens,
            link_id,
            reduced_payload_mb,
            participants,
            topology_id,
            placement_id,
            burst_multiplier,
        )
        exposed_ms = congestion["tail_ms"] * (1 - float(overlap_pct) / 100)
        step_ok = exposed_ms <= lens["step_budget_ms"]
        utilization_ok = congestion["utilization_ok"]
        topology_ok = congestion["topology_ok"]
        ratios = {
            "step-time/SLO": exposed_ms / lens["step_budget_ms"],
            "utilization": congestion["utilization"] / lens["utilization_limit"],
            "topology": 0.0 if topology_ok else 1.5,
        }
        binding = max(ratios.items(), key=lambda item: item[1])[0]
        return {
            **congestion,
            "reduced_payload_mb": reduced_payload_mb,
            "payload_reduction_pct": float(payload_reduction_pct),
            "overlap_pct": float(overlap_pct),
            "exposed_ms": exposed_ms,
            "step_budget_ms": lens["step_budget_ms"],
            "step_ok": step_ok,
            "utilization_ok": utilization_ok,
            "topology_ok": topology_ok,
            "valid_plan": step_ok and utilization_ok and topology_ok,
            "binding_guardrail": binding,
            "ratios": ratios,
        }

    def v2_03_status_label(condition):
        return "PASS" if condition else "FAIL"

    def v2_03_html_table(rows, columns, caption=""):
        header = "".join(f"<th>{html_lib.escape(label)}</th>" for label, _key in columns)
        body_rows = []
        for row in rows:
            cells = []
            for _label, key in columns:
                value = row.get(key, "")
                cells.append(f"<td>{html_lib.escape(str(value))}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        caption_html = f"<div style='font-weight:700; margin-bottom:8px;'>{html_lib.escape(caption)}</div>" if caption else ""
        return mo.Html(
            f"""
<div style="overflow-x:auto; margin:12px 0;">
  {caption_html}
  <table style="border-collapse:collapse; min-width:720px; width:100%; font-size:0.9rem;">
    <thead><tr style="background:#f8fafc;">{header}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</div>
<style>
td, th {{ border:1px solid #d9dee8; padding:8px 10px; text-align:left; vertical-align:top; }}
th {{ color:#344054; font-weight:700; }}
</style>
"""
        )

    def v2_03_failure_callout(result, message_ok, message_fail):
        if result:
            return mo.callout(mo.md(message_ok), kind="success")
        return mo.callout(mo.md(message_fail), kind="danger")

    def v2_03_prediction_feedback(predicted, actual, correct, missed):
        if predicted is None:
            return mo.callout(mo.md("Make the structured prediction first; the evidence below is unlocked after you commit."), kind="warn")
        if predicted == actual:
            return mo.callout(mo.md(correct), kind="success")
        return mo.callout(mo.md(missed), kind="warn")

    def v2_03_alpha_beta_chart(colors, result):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Current transfer"],
                y=[result["alpha_ms"]],
                name="alpha startup",
                marker_color=colors["BlueLine"],
            )
        )
        fig.add_trace(
            go.Bar(
                x=["Current transfer"],
                y=[result["beta_ms"]],
                name="n/beta payload",
                marker_color=colors["OrangeLine"],
            )
        )
        fig.add_hline(
            y=result["budget_ms"],
            line_dash="dash",
            line_color=colors["RedLine"],
            annotation_text=f"budget {v2_03_fmt(result['budget_ms'])} ms",
        )
        fig.update_layout(
            barmode="stack",
            height=320,
            yaxis_title="Milliseconds",
            margin=dict(l=60, r=20, t=30, b=40),
            legend=dict(orientation="h", y=1.18, x=0),
        )
        return apply_plotly_theme(fig)

    def v2_03_topology_chart(colors, rows, budget_ms):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[row["label"] for row in rows],
                y=[row["sync_ms"] for row in rows],
                marker_color=[
                    colors["GreenLine"] if row["passes_step"] and row["guard_ok"] else colors["RedLine"]
                    for row in rows
                ],
                text=[
                    "PASS" if row["passes_step"] and row["guard_ok"] else "CHECK"
                    for row in rows
                ],
                textposition="outside",
            )
        )
        fig.add_hline(
            y=budget_ms,
            line_dash="dash",
            line_color=colors["RedLine"],
            annotation_text=f"step/SLO {v2_03_fmt(budget_ms)} ms",
        )
        fig.update_layout(
            height=360,
            yaxis_title="Synchronization time (ms)",
            showlegend=False,
            margin=dict(l=60, r=20, t=30, b=90),
        )
        return apply_plotly_theme(fig)

    def v2_03_congestion_chart(colors, congestion):
        fig = go.Figure()
        util_pct = congestion["utilization"] * 100
        limit_pct = congestion["utilization_limit"] * 100
        fig.add_trace(
            go.Bar(
                x=["Utilization"],
                y=[util_pct],
                name="Utilization",
                marker_color=colors["RedLine"] if not congestion["utilization_ok"] else colors["GreenLine"],
                text=[f"{util_pct:.0f}%"],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Bar(
                x=["Tail time"],
                y=[congestion["tail_ms"]],
                name="Tail time (ms)",
                yaxis="y2",
                marker_color=colors["OrangeLine"],
                text=[f"{congestion['tail_ms']:.1f} ms"],
                textposition="outside",
            )
        )
        fig.add_hline(y=limit_pct, line_dash="dash", line_color=colors["RedLine"], annotation_text="utilization limit")
        fig.update_layout(
            height=330,
            yaxis=dict(title="Utilization (%)"),
            yaxis2=dict(title="Tail time (ms)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.18, x=0),
            margin=dict(l=60, r=60, t=30, b=40),
        )
        return apply_plotly_theme(fig)

    def v2_03_candidate_chart(colors, selected, rejected):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Selected plan", "Rejected alternative"],
                y=[selected["exposed_ms"], rejected["exposed_ms"]],
                marker_color=[
                    colors["GreenLine"] if selected["valid_plan"] else colors["RedLine"],
                    colors["RedLine"],
                ],
                text=[
                    f"{selected['exposed_ms']:.1f} ms",
                    f"{rejected['exposed_ms']:.1f} ms",
                ],
                textposition="outside",
            )
        )
        fig.add_hline(
            y=selected["step_budget_ms"],
            line_dash="dash",
            line_color=colors["RedLine"],
            annotation_text=f"SLO {v2_03_fmt(selected['step_budget_ms'])} ms",
        )
        fig.update_layout(
            height=330,
            yaxis_title="Exposed communication time (ms)",
            showlegend=False,
            margin=dict(l=60, r=20, t=30, b=45),
        )
        return apply_plotly_theme(fig)

    def v2_03_scenario_assumptions(lens):
        link_sources = "; ".join(
            f"{link['label']}: {link['source']}" for link in lens["links"].values()
        )
        return {
            "summary": "V2-03 communication fabric source trace",
            "chapter_anchor": "Volume II, Chapter 3: Network Fabrics",
            "source_models": "T(n)=alpha+n/beta; BW_bisect=(N/2)*beta/oversubscription; utilization=offered_load/capacity",
            "registry_functions": "mlsysim.physics.calc_point_to_point_time, calc_alpha_beta_crossover, calc_bisection_bandwidth, calc_oversubscription_effect",
            "track_scenario_thresholds": (
                f"communication budget {lens['communication_budget_ms']} ms; "
                f"step/SLO budget {lens['step_budget_ms']} ms; "
                f"utilization limit {lens['utilization_limit']:.0%}"
            ),
            "link_sources": link_sources,
            "local_assumptions": "Non-cloud link rates and track guardrail thresholds are notebook-local pedagogical assumptions.",
        }

    return (
        v2_03_alpha_beta_chart,
        v2_03_alpha_beta_result,
        v2_03_candidate_chart,
        v2_03_congestion_chart,
        v2_03_congestion_result,
        v2_03_decision_result,
        v2_03_evaluate_topologies,
        v2_03_failure_callout,
        v2_03_fmt,
        v2_03_html_table,
        v2_03_link_option_labels,
        v2_03_option_labels,
        v2_03_placement_options,
        v2_03_prediction_feedback,
        v2_03_scenario_assumptions,
        v2_03_selected_topology_result,
        v2_03_status_label,
        v2_03_to_mb_s,
        v2_03_to_ms,
        v2_03_topology_chart,
        v2_03_topology_options,
        v2_03_track_lenses,
    )


@app.cell
def _(Hardware, Systems, ureg, v2_03_profile, v2_03_track_lenses):
    v2_03_lenses = v2_03_track_lenses(Systems, Hardware, ureg)
    v2_03_lens = v2_03_lenses[v2_03_profile.track_id]
    return v2_03_lens, v2_03_lenses


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_arc_context,
    track_context,
    v2_03_lens,
    v2_03_metadata,
    v2_03_profile,
    v2_03_scenario_assumptions,
    v2_03_variant,
):
    mo.vstack(
        [
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            mo.Html(
                f"""
<div class="mlsysbook-panel mlsysbook-launch-panel">
  <div class="mlsysbook-section-label">Machine Learning Systems - Volume II - Lab 03</div>
  <h1 style="margin-bottom:8px;">Network Fabric Design</h1>
  <p class="mlsysbook-scenario-narrative" style="font-size:1.02rem;">
    <strong>Chapter invariant:</strong> Network shape governs distributed work.
    Bandwidth, latency, bisection, topology, placement, and congestion turn
    communication into a binding amount.
  </p>
  <div class="mlsysbook-compact-fields is-brief">
    <div><strong>Selected track:</strong> {v2_03_profile.label}</div>
    <div><strong>Stakeholder:</strong> {v2_03_variant.stakeholder}</div>
    <div><strong>Scenario:</strong> {v2_03_lens["scenario"]}</div>
    <div><strong>Report frame:</strong> {v2_03_lens["report_prompt"]}</div>
  </div>
</div>
"""
            ),
            track_context(v2_03_profile),
            track_arc_context(v2_03_profile, v2_03_metadata.lab_id),
            mo.callout(
                mo.md(
                    """
**Recommended reading before this lab**

- Volume II, Chapter 3: Network Fabrics
- Focus sections: alpha/beta performance model, switch and topology,
  fabric behavior, congestion control, monitoring, and summary.
"""
                ),
                kind="info",
            ),
            source_trace(v2_03_scenario_assumptions(v2_03_lens), collapsed=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, v2_03_lens):
    partA_prediction = mo.ui.radio(
        options={
            "A) The fixed startup latency alpha": "latency (alpha)",
            "B) The per-byte bandwidth term n/beta": "bandwidth (n/beta)",
            "C) Link bandwidth alone, regardless of payload": "bandwidth_only",
            "D) Topology only, before payload size is known": "topology_only",
        },
        label="Part A prediction - which term will bind for the current payload and path?",
    )
    partB_prediction = mo.ui.radio(
        options={
            "A) Fast endpoint links are enough": "edge_links",
            "B) The narrowest bisection cut controls feasible parallel work": "bisection",
            "C) More participants always improves synchronization": "participants",
            "D) Hop count matters, but oversubscription does not": "hops_only",
        },
        label="Part B prediction - what decides whether parallel communication remains feasible?",
    )
    partC_prediction = mo.ui.radio(
        options={
            "A) Topology-aware placement keeps utilization below the tail-risk limit": "placement",
            "B) Adaptive routing removes congestion risk": "routing",
            "C) Background traffic is unrelated to synchronous work": "background",
            "D) Local placement choices cannot create fleet-wide bottlenecks": "local_only",
        },
        label="Part C prediction - what prevents a local communication choice from becoming a fleet bottleneck?",
    )
    partD_prediction = mo.ui.radio(
        options={
            "A) Step-time/SLO will reject the naive plan": "step",
            "B) Utilization will reject the naive plan": "utilization",
            "C) Topology/placement guardrails will reject the naive plan": "topology",
            "D) One passing metric is enough to approve the plan": "single_metric",
        },
        label="Part D prediction - which guardrail is most likely to reject the naive plan?",
    )

    partA_checkpoint = mo.ui.radio(
        options={
            "Reduce alpha by shortening the path or hop count": "reduce_alpha",
            "Increase beta by choosing a higher-bandwidth link": "increase_beta",
            "Reduce payload before crossing the network": "reduce_payload",
        },
        label="Part A checkpoint - which lever should this track try first?",
    )
    partB_checkpoint = mo.ui.radio(
        options={
            "Carry forward a topology with full or aligned bisection": "bisection_guard",
            "Accept oversubscription and rely on retries": "accept_oversub",
            "Reduce endpoint count until the topology becomes feasible": "reduce_participants",
        },
        label="Part B checkpoint - what topology assumption should the memo carry forward?",
    )
    partC_checkpoint = mo.ui.radio(
        options={
            "Use topology-aware affinity placement": "affinity",
            "Use balanced placement with explicit utilization headroom": "balanced",
            "Use cheapest available placement and monitor later": "cheap",
        },
        label="Part C checkpoint - what placement mitigation should the final plan use?",
    )
    partD_final_decision = mo.ui.radio(
        options={
            "Approve the selected plan": "approve",
            "Revise payload or overlap before approval": "revise_payload",
            "Reject topology/placement and choose a safer fabric policy": "reject_policy",
        },
        label="Final decision - how should the stakeholder sign the communication plan?",
    )
    student_id = mo.ui.text(label="Student identifier", placeholder="Optional")
    memo_note = mo.ui.text_area(
        label="Communication memo note",
        placeholder="Name the selected topology/placement, binding amount, rejected alternative, and collective-communication implication.",
        full_width=True,
    )

    payload_mb = mo.ui.slider(
        start=v2_03_lens["message_min_mb"],
        stop=v2_03_lens["message_max_mb"],
        value=v2_03_lens["message_default_mb"],
        step=v2_03_lens["message_step_mb"],
        label=f"Payload size ({v2_03_lens['payload_name']}, MB)",
    )
    participants = mo.ui.slider(
        start=v2_03_lens["participants_min"],
        stop=v2_03_lens["participants_max"],
        value=v2_03_lens["participants_default"],
        step=4,
        label=f"Parallel endpoints ({v2_03_lens['participant_name']})",
    )
    burst_multiplier = mo.ui.slider(
        start=0.5,
        stop=4.0,
        value=1.0,
        step=0.25,
        label="Burst / background pressure multiplier",
    )
    payload_reduction_pct = mo.ui.slider(
        start=0,
        stop=80,
        value=25,
        step=5,
        label="Payload reduction before crossing fabric (%)",
    )
    overlap_pct = mo.ui.slider(
        start=0,
        stop=80,
        value=20,
        step=5,
        label="Communication hidden by useful work (%)",
    )
    return (
        burst_multiplier,
        memo_note,
        overlap_pct,
        partA_checkpoint,
        partA_prediction,
        partB_checkpoint,
        partB_prediction,
        partC_checkpoint,
        partC_prediction,
        partD_final_decision,
        partD_prediction,
        participants,
        payload_mb,
        payload_reduction_pct,
        student_id,
    )


@app.cell(hide_code=True)
def _(
    mo,
    v2_03_lens,
    v2_03_link_option_labels,
    v2_03_option_labels,
    v2_03_placement_options,
    v2_03_topology_options,
):
    _link_options = v2_03_link_option_labels(v2_03_lens)
    _topology_options = v2_03_option_labels(v2_03_topology_options(v2_03_lens))
    _placement_options = v2_03_option_labels(v2_03_placement_options(v2_03_lens))
    active_link = mo.ui.dropdown(
        options=_link_options,
        value=v2_03_lens["links"][v2_03_lens["default_link"]]["label"],
        label="Active communication path",
    )
    topology_choice = mo.ui.dropdown(
        options=_topology_options,
        value=v2_03_lens["topology_labels"]["aligned"],
        label="Topology shape",
    )
    placement_choice = mo.ui.dropdown(
        options=_placement_options,
        value=v2_03_lens["placement_labels"]["affinity"],
        label="Placement policy",
    )
    return active_link, placement_choice, topology_choice


@app.cell(hide_code=True)
def _(
    COLORS,
    MathPeek,
    active_link,
    build_lab_report,
    burst_multiplier,
    ledger,
    memo_note,
    mo,
    overlap_pct,
    partA_checkpoint,
    partA_prediction,
    partB_checkpoint,
    partB_prediction,
    partC_checkpoint,
    partC_prediction,
    partD_final_decision,
    partD_prediction,
    participants,
    payload_mb,
    payload_reduction_pct,
    placement_choice,
    report_export_panel,
    student_id,
    topology_choice,
    v2_03_alpha_beta_chart,
    v2_03_alpha_beta_result,
    v2_03_candidate_chart,
    v2_03_chapter,
    v2_03_congestion_chart,
    v2_03_congestion_result,
    v2_03_decision_result,
    v2_03_evaluate_topologies,
    v2_03_failure_callout,
    v2_03_fmt,
    v2_03_html_table,
    v2_03_lens,
    v2_03_metadata,
    v2_03_prediction_feedback,
    v2_03_profile,
    v2_03_status_label,
    v2_03_topology_chart,
    v2_03_variant,
):
    _payload_mb = payload_mb.value
    _participants = participants.value
    _link_id = active_link.value
    _topology_id = topology_choice.value
    _placement_id = placement_choice.value

    _part_a = v2_03_alpha_beta_result(v2_03_lens, _link_id, _payload_mb)
    _actual_part_a = _part_a["binding_term"]
    _topology_rows = v2_03_evaluate_topologies(v2_03_lens, _link_id, _payload_mb, _participants)
    _part_b_selected = next(row for row in _topology_rows if row["topology_id"] == _topology_id)
    _part_c = v2_03_congestion_result(
        v2_03_lens,
        _link_id,
        _payload_mb,
        _participants,
        _topology_id,
        _placement_id,
        burst_multiplier.value,
    )
    _part_d = v2_03_decision_result(
        v2_03_lens,
        _link_id,
        _payload_mb,
        _participants,
        _topology_id,
        _placement_id,
        burst_multiplier.value,
        payload_reduction_pct.value,
        overlap_pct.value,
    )
    _rejected = v2_03_decision_result(
        v2_03_lens,
        _link_id,
        _payload_mb,
        _participants,
        "oversubscribed",
        "spread",
        max(1.5, burst_multiplier.value),
        0,
        0,
    )

    def _part_a_table():
        return v2_03_html_table(
            [
                {
                    "Field": "Active path",
                    "Value": _part_a["link_label"],
                    "Interpretation": _part_a["source"],
                },
                {
                    "Field": "alpha startup",
                    "Value": f"{_part_a['alpha_ms']:.3f} ms",
                    "Interpretation": "Fixed cost before bytes move.",
                },
                {
                    "Field": "n/beta payload term",
                    "Value": f"{_part_a['beta_ms']:.3f} ms",
                    "Interpretation": "Payload cost from message size and bandwidth.",
                },
                {
                    "Field": "Crossover size",
                    "Value": f"{_part_a['crossover_kb']:.1f} KB",
                    "Interpretation": "Below this size alpha dominates; above it bandwidth dominates.",
                },
                {
                    "Field": "Budget status",
                    "Value": v2_03_status_label(_part_a["passes_budget"]),
                    "Interpretation": f"{_part_a['total_ms']:.3f} ms vs {v2_03_fmt(_part_a['budget_ms'])} ms budget.",
                },
            ],
            [("Field", "Field"), ("Value", "Value"), ("Interpretation", "Interpretation")],
            caption="Part A exact evidence table",
        )

    def _part_b_table():
        rows = []
        for row in _topology_rows:
            rows.append(
                {
                    "Topology": row["label"],
                    "BW_bisect": f"{row['bisection_mb_s']:,.0f} MB/s",
                    "Sync time": f"{row['sync_ms']:.2f} ms",
                    "Throughput loss": f"{row['throughput_loss']:.0%}",
                    "Step/SLO": v2_03_status_label(row["passes_step"]),
                    "Topology guard": v2_03_status_label(row["guard_ok"]),
                    "Note": row["note"],
                }
            )
        return v2_03_html_table(
            rows,
            [
                ("Topology", "Topology"),
                ("BW_bisect", "BW_bisect"),
                ("Sync time", "Sync time"),
                ("Throughput loss", "Throughput loss"),
                ("Step/SLO", "Step/SLO"),
                ("Topology guard", "Topology guard"),
                ("Note", "Note"),
            ],
            caption="Part B bisection and topology table",
        )

    def _part_c_table():
        return v2_03_html_table(
            [
                {
                    "Metric": "Placement",
                    "Value": _part_c["placement_label"],
                    "Limit or meaning": _part_c["placement_note"],
                },
                {
                    "Metric": "Offered load",
                    "Value": f"{_part_c['offered_mb_s']:,.0f} MB/s",
                    "Limit or meaning": "Demand injected into the selected bisection window.",
                },
                {
                    "Metric": "Bisection capacity",
                    "Value": f"{_part_c['capacity_mb_s']:,.0f} MB/s",
                    "Limit or meaning": "Effective capacity after topology and oversubscription.",
                },
                {
                    "Metric": "Utilization",
                    "Value": f"{_part_c['utilization']:.0%}",
                    "Limit or meaning": f"Track limit is {v2_03_lens['utilization_limit']:.0%}.",
                },
                {
                    "Metric": "Tail multiplier",
                    "Value": f"{_part_c['tail_multiplier']:.2f}x",
                    "Limit or meaning": "Congestion turns average sync time into tail sync time.",
                },
                {
                    "Metric": "Failure state",
                    "Value": _part_c["failure_state"],
                    "Limit or meaning": _part_c["binding_amount"],
                },
            ],
            [("Metric", "Metric"), ("Value", "Value"), ("Limit or meaning", "Limit or meaning")],
            caption="Part C congestion and placement table",
        )

    def _part_d_table(selected, rejected):
        rows = [
            {
                "Guardrail": "Step-time/SLO",
                "Selected plan": f"{selected['exposed_ms']:.2f} ms ({v2_03_status_label(selected['step_ok'])})",
                "Rejected alternative": f"{rejected['exposed_ms']:.2f} ms ({v2_03_status_label(rejected['step_ok'])})",
                "Limit": f"<= {v2_03_fmt(selected['step_budget_ms'])} ms",
            },
            {
                "Guardrail": "Utilization",
                "Selected plan": f"{selected['utilization']:.0%} ({v2_03_status_label(selected['utilization_ok'])})",
                "Rejected alternative": f"{rejected['utilization']:.0%} ({v2_03_status_label(rejected['utilization_ok'])})",
                "Limit": f"<= {v2_03_lens['utilization_limit']:.0%}",
            },
            {
                "Guardrail": "Topology/placement",
                "Selected plan": v2_03_status_label(selected["topology_ok"]),
                "Rejected alternative": v2_03_status_label(rejected["topology_ok"]),
                "Limit": "must pass topology guardrail",
            },
        ]
        return v2_03_html_table(
            rows,
            [
                ("Guardrail", "Guardrail"),
                ("Selected plan", "Selected plan"),
                ("Rejected alternative", "Rejected alternative"),
                ("Limit", "Limit"),
            ],
            caption="Part D simultaneous guardrail table",
        )

    def build_part_a():
        items = [
            mo.md("## Part A - Concept Module: Alpha/Beta Terms Predict Communication Cost"),
            mo.callout(
                mo.md(
                    f"**Scenario.** {v2_03_variant.stakeholder} must move a "
                    f"**{_part_a['payload_mb']:.0f} MB {v2_03_lens['payload_name']}** "
                    f"over **{_part_a['link_label']}** before the track budget is exhausted."
                ),
                kind="info",
            ),
            partA_prediction,
            v2_03_prediction_feedback(
                partA_prediction.value,
                _actual_part_a,
                f"Correct. The measured binding term is **{_actual_part_a}**.",
                f"The measured binding term is **{_actual_part_a}**. The chart separates alpha from n/beta.",
            ),
        ]
        if partA_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([payload_mb, active_link], justify="start"),
                mo.as_html(v2_03_alpha_beta_chart(COLORS, _part_a)),
                _part_a_table(),
                v2_03_failure_callout(
                    _part_a["passes_budget"],
                    f"Consequence: {_part_a['total_ms']:.2f} ms fits the {v2_03_fmt(_part_a['budget_ms'])} ms communication budget.",
                    f"Boundary: {_part_a['total_ms']:.2f} ms exceeds the {v2_03_fmt(_part_a['budget_ms'])} ms budget. Mitigation must reduce hops/alpha, raise beta, or shrink payload.",
                ),
                MathPeek(
                    "T(n)=alpha+n/beta; n*=alpha*beta",
                    {
                        "alpha": f"{_part_a['alpha_ms']:.3f} ms startup",
                        "n/beta": f"{_part_a['beta_ms']:.3f} ms payload term",
                        "n*": f"{_part_a['crossover_kb']:.1f} KB crossover",
                        "chapter source": "Network Fabrics performance model",
                    },
                ),
                partA_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_part_b():
        _actual = "bisection"
        items = [
            mo.md("## Part B - Concept Module: Topology And Bisection Change Feasible Parallel Work"),
            mo.callout(
                mo.md(
                    f"**Scenario.** The same payload now scales to **{_participants} "
                    f"{v2_03_lens['participant_name']}**. The question is whether the topology "
                    "keeps enough cross-sectional bandwidth for parallel work."
                ),
                kind="info",
            ),
            partB_prediction,
            v2_03_prediction_feedback(
                partB_prediction.value,
                _actual,
                "Correct. The narrowest bisection cut determines feasible global communication.",
                "Endpoint link speed is not enough. The bisection table shows which topology preserves useful parallel work.",
            ),
        ]
        if partB_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([participants, topology_choice, active_link], justify="start"),
                mo.as_html(v2_03_topology_chart(COLORS, _topology_rows, v2_03_lens["step_budget_ms"])),
                _part_b_table(),
                v2_03_failure_callout(
                    _part_b_selected["passes_step"] and _part_b_selected["guard_ok"],
                    f"Consequence: {_part_b_selected['label']} keeps the modeled sync at {_part_b_selected['sync_ms']:.2f} ms.",
                    f"Boundary: {_part_b_selected['label']} is not a safe topology assumption. It reports {_part_b_selected['sync_ms']:.2f} ms against a {v2_03_fmt(v2_03_lens['step_budget_ms'])} ms step/SLO budget and guardrail status {v2_03_status_label(_part_b_selected['guard_ok'])}.",
                ),
                MathPeek(
                    "BW_bisect=(N/2)*beta/oversubscription",
                    {
                        "participants": f"{_participants}",
                        "selected topology": _part_b_selected["label"],
                        "effective bisection": f"{_part_b_selected['bisection_mb_s']:,.0f} MB/s",
                        "chapter source": "Network Fabrics topology and bisection sections",
                    },
                ),
                partB_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_part_c():
        _actual = "placement"
        items = [
            mo.md("## Part C - Concept Module: Congestion And Placement Make Local Choices Fleet-Wide Bottlenecks"),
            mo.callout(
                mo.md(
                    f"**Scenario.** The scheduler chooses **{_part_c['placement_label']}** on "
                    f"**{_part_c['topology_label']}**. Under BSP-style synchronization, the slowest "
                    "congested path paces the whole fleet."
                ),
                kind="info",
            ),
            partC_prediction,
            v2_03_prediction_feedback(
                partC_prediction.value,
                _actual,
                "Correct. Placement and utilization headroom decide whether local choices create fleet-wide tail latency.",
                "The productive failure is treating placement as bookkeeping. The evidence below shows congestion amplification.",
            ),
        ]
        if partC_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([placement_choice, burst_multiplier, topology_choice], justify="start"),
                mo.as_html(v2_03_congestion_chart(COLORS, _part_c)),
                _part_c_table(),
                v2_03_failure_callout(
                    _part_c["utilization_ok"] and _part_c["topology_ok"],
                    f"Consequence: utilization is {_part_c['utilization']:.0%}, below the {v2_03_lens['utilization_limit']:.0%} guardrail.",
                    f"Boundary: utilization is {_part_c['utilization']:.0%} against a {v2_03_lens['utilization_limit']:.0%} limit, with topology guardrail {v2_03_status_label(_part_c['topology_ok'])}. Mitigation must improve locality or reduce burst pressure.",
                ),
                MathPeek(
                    "rho=offered_load/capacity; tail rises as rho approaches 1",
                    {
                        "offered load": f"{_part_c['offered_mb_s']:,.0f} MB/s",
                        "capacity": f"{_part_c['capacity_mb_s']:,.0f} MB/s",
                        "rho": f"{_part_c['utilization']:.2f}",
                        "chapter source": "Network Fabrics fabric behavior and congestion-control sections",
                    },
                ),
                partC_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_part_d():
        _naive_binding = _rejected["binding_guardrail"]
        _naive_prediction_key = {
            "step-time/SLO": "step",
            "utilization": "utilization",
            "topology": "topology",
        }.get(_naive_binding, "topology")
        items = [
            mo.md("## Part D - Concept Module: Communication Plan Guardrails"),
            mo.callout(
                mo.md(
                    f"**Scenario.** The final communication plan must satisfy step-time/SLO, "
                    f"utilization, and topology guardrails for {v2_03_profile.label}. "
                    f"{v2_03_lens['guardrail_text']}"
                ),
                kind="info",
            ),
            partD_prediction,
            v2_03_prediction_feedback(
                partD_prediction.value,
                _naive_prediction_key,
                f"Correct. The rejected alternative is primarily blocked by **{_rejected['binding_guardrail']}**.",
                f"The rejected alternative is primarily blocked by **{_rejected['binding_guardrail']}**. A plan needs all three guardrails, not one good metric.",
            ),
        ]
        if partD_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack(
                    [topology_choice, placement_choice, payload_reduction_pct, overlap_pct],
                    justify="start",
                ),
                mo.as_html(v2_03_candidate_chart(COLORS, _part_d, _rejected)),
                _part_d_table(_part_d, _rejected),
                v2_03_failure_callout(
                    _part_d["valid_plan"],
                    f"Consequence: selected plan passes all guardrails. Binding guardrail is {_part_d['binding_guardrail']}.",
                    f"Boundary: selected plan fails at least one guardrail. Binding guardrail is {_part_d['binding_guardrail']}; revise topology, placement, payload reduction, or overlap.",
                ),
                MathPeek(
                    "valid = exposed_time<=SLO and utilization<=limit and topology_guardrail",
                    {
                        "exposed time": f"{_part_d['exposed_ms']:.2f} ms vs {v2_03_fmt(_part_d['step_budget_ms'])} ms",
                        "utilization": f"{_part_d['utilization']:.0%} vs {v2_03_lens['utilization_limit']:.0%}",
                        "topology guardrail": v2_03_status_label(_part_d["topology_ok"]),
                        "chapter source": "Network Fabrics summary and fallacies",
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
        _binding_amount = _part_d["binding_guardrail"] if not _part_d["valid_plan"] else _part_c["binding_amount"]
        _memo = memo_note.value or (
            f"Use {_part_d['topology_label']} with {_part_d['placement_label']}; "
            f"binding amount: {_binding_amount}; reject raw payload on oversubscribed spread placement; "
            f"{v2_03_lens['collective_implication']}"
        )
        _snapshot = {
            "track_id": v2_03_profile.track_id,
            "scenario_id": v2_03_variant.scenario_id,
            "active_link": _part_a["link_label"],
            "payload_mb": _payload_mb,
            "participants": _participants,
            "selected_topology": _part_d["topology_label"],
            "selected_placement": _part_d["placement_label"],
            "reduced_payload_mb": round(_part_d["reduced_payload_mb"], 3),
            "binding_network_amount": _binding_amount,
            "step_time_ms": round(_part_d["exposed_ms"], 3),
            "utilization": round(_part_d["utilization"], 4),
            "valid_plan": _part_d["valid_plan"],
            "rejected_alternative": "raw payload on oversubscribed spread placement",
            "rejected_exposed_ms": round(_rejected["exposed_ms"], 3),
            "collective_communication_implication": v2_03_lens["collective_implication"],
            "completed": _completed,
        }
        _design = {
            "lab_id": v2_03_metadata.lab_id,
            "track_id": v2_03_profile.track_id,
            "scenario_id": v2_03_variant.scenario_id,
            "selected_topology": _part_d["topology_label"],
            "selected_placement": _part_d["placement_label"],
            "active_link": _part_a["link_label"],
            "payload_mb": _payload_mb,
            "binding_network_amount": _binding_amount,
            "step_time_ms": _part_d["exposed_ms"],
            "utilization": _part_d["utilization"],
            "rejected_alternative": "raw payload on oversubscribed spread placement",
            "collective_communication_implication": v2_03_lens["collective_implication"],
            "completed": _completed,
            "result_snapshot": _snapshot,
        }
        ledger.save(track=v2_03_profile.track_id, chapter=v2_03_chapter, design=_design)

        _incomplete = []
        if partA_prediction.value is None:
            _incomplete.append("Part A alpha/beta prediction")
        if partA_checkpoint.value is None:
            _incomplete.append("Part A checkpoint decision")
        if partB_prediction.value is None:
            _incomplete.append("Part B bisection prediction")
        if partB_checkpoint.value is None:
            _incomplete.append("Part B topology checkpoint")
        if partC_prediction.value is None:
            _incomplete.append("Part C congestion/placement prediction")
        if partC_checkpoint.value is None:
            _incomplete.append("Part C placement checkpoint")
        if partD_prediction.value is None:
            _incomplete.append("Part D guardrail prediction")
        if partD_final_decision.value is None:
            _incomplete.append("Final communication plan decision")

        _report = build_lab_report(
            v2_03_metadata,
            student_id=student_id.value or "",
            track=v2_03_profile.label,
            scenario=v2_03_lens["scenario"],
            learning_objectives=(
                "Use alpha/beta terms to classify communication cost.",
                "Use bisection bandwidth and topology to test feasible parallel work.",
                "Diagnose congestion and placement as fleet-wide bottlenecks.",
                "Approve a communication plan only when SLO, utilization, and topology guardrails pass.",
            ),
            predictions={
                "partA_alpha_beta": partA_prediction.value,
                "partB_bisection": partB_prediction.value,
                "partC_congestion_placement": partC_prediction.value,
                "partD_guardrail": partD_prediction.value,
            },
            knob_settings={
                "payload_mb": _payload_mb,
                "participants": _participants,
                "active_link": _part_a["link_label"],
                "topology": _part_d["topology_label"],
                "placement": _part_d["placement_label"],
                "burst_multiplier": burst_multiplier.value,
                "payload_reduction_pct": payload_reduction_pct.value,
                "overlap_pct": overlap_pct.value,
            },
            binding_constraints={
                "binding_network_amount": _binding_amount,
                "step_time_ms": round(_part_d["exposed_ms"], 3),
                "utilization": round(_part_d["utilization"], 4),
                "valid_plan": _part_d["valid_plan"],
            },
            evidence_summary={
                "alpha_beta_binding": _part_a["binding_term"],
                "topology_sync_ms": round(_part_b_selected["sync_ms"], 3),
                "congestion_tail_ms": round(_part_c["tail_ms"], 3),
                "final_exposed_ms": round(_part_d["exposed_ms"], 3),
                "rejected_exposed_ms": round(_rejected["exposed_ms"], 3),
            },
            decisions={
                "alpha_beta_checkpoint": partA_checkpoint.value,
                "topology_checkpoint": partB_checkpoint.value,
                "placement_checkpoint": partC_checkpoint.value,
                "final_decision": partD_final_decision.value,
            },
            reflections={"communication_memo_note": memo_note.value},
            final_decision=_memo,
            big_takeaways=(
                "Communication cost is an amount-system budget, not a single link-speed number.",
                "Bisection and placement decide how much parallel work remains useful.",
                "A network plan must pass SLO, utilization, and topology guardrails simultaneously.",
            ),
            residual_risk=v2_03_lens["failure_mode"],
            source_trace={
                "chapter_anchor": "Volume II, Chapter 3: Network Fabrics",
                "source_models": "alpha/beta, bisection bandwidth, utilization guardrails",
                "mlsysim_functions": "calc_point_to_point_time, calc_alpha_beta_crossover, calc_bisection_bandwidth",
                "track_source_policy": v2_03_profile.source_policy,
                "scenario_assumptions": "track thresholds and non-cloud link rates are notebook-local pedagogical assumptions",
            },
            result_snapshot=_snapshot,
            incomplete_fields=tuple(_incomplete),
        )

        return mo.vstack(
            [
                mo.md("## Synthesis - Network Communication Memo"),
                student_id,
                memo_note,
                mo.callout(
                    mo.md(
                        f"**Selected policy:** {_part_d['topology_label']} with {_part_d['placement_label']}.\n\n"
                        f"**Binding network amount:** {_binding_amount}.\n\n"
                        f"**Rejected alternative:** raw payload on oversubscribed spread placement "
                        f"({_rejected['exposed_ms']:.2f} ms exposed time).\n\n"
                        f"**Carry-forward:** {v2_03_lens['collective_implication']}"
                    ),
                    kind="success" if _part_d["valid_plan"] else "warn",
                ),
                partD_final_decision,
                report_export_panel(_report),
            ]
        )

    v2_03_tabs = mo.ui.tabs(
        {
            "Part A: Alpha/Beta": build_part_a(),
            "Part B: Topology": build_part_b(),
            "Part C: Congestion": build_part_c(),
            "Part D: Guardrails": build_part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    v2_03_tabs
    return


@app.cell(hide_code=True)
def _(mo, v2_03_metadata, v2_03_profile):
    mo.Html(
        f"""
<div class="lab-hud">
  <span class="hud-label">LAB</span>
  <span class="hud-value">{v2_03_metadata.lab_id}</span>
  <span class="hud-label">TRACK</span>
  <span class="hud-value">{v2_03_profile.label}</span>
  <span style="flex:1;"></span>
  <span class="hud-label">STATUS</span>
  <span class="hud-active">ACTIVE</span>
</div>
"""
    )
    return


if __name__ == "__main__":
    app.run()
