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
    import math
    import html
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
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        batching_tax,
        build_lab_report,
        cache_capacity,
        cold_start_latency,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        queueing_latency,
        report_export_panel,
        resolve_mlsysim_ref,
        serving_track_profile,
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
        batching_tax,
        build_lab_report,
        cache_capacity,
        cold_start_latency,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        html,
        ledger,
        math,
        mlsysim,
        mo,
        np,
        queueing_latency,
        report_export_panel,
        resolve_mlsysim_ref,
        serving_track_profile,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_13_metadata = get_lab_metadata("vol1/lab_13_model_serving.py")
    return (v1_13_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v1_13_track_picker = track_selector(default=_default_track)
    v1_13_track_picker
    return (v1_13_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    serving_track_profile,
    v1_13_track_picker,
):
    v1_13_track_id = v1_13_track_picker.value
    v1_13_profile = get_track_profile(v1_13_track_id)
    v1_13_variant = get_lab_track_variant("v1_13_tail_latency_trap", v1_13_profile.track_id)
    v1_13_hardware = resolve_mlsysim_ref(v1_13_variant.hardware_ref)
    v1_13_model = resolve_mlsysim_ref(v1_13_variant.model_ref)
    v1_13_serving = serving_track_profile(
        v1_13_profile,
        v1_13_variant,
        v1_13_hardware,
        v1_13_model,
    )
    return (
        v1_13_hardware,
        v1_13_model,
        v1_13_profile,
        v1_13_serving,
        v1_13_track_id,
        v1_13_variant,
    )


@app.cell
def _(batching_tax, cache_capacity, cold_start_latency, html, math, queueing_latency):
    def v1_13_track_packet(track_id, serving, variant):
        common = {
            "stakeholder": variant.stakeholder,
            "amount_focus": "requests, tail latency, capacity headroom, and recurring cost",
            "replica_label": "serving replicas",
            "batch_amount": "requests per batch",
            "queue_failure": "p99 crosses the SLO before the mean looks alarming",
            "cost_label": "Cost/request",
            "cost_unit": "USD/request",
            "cost_limit": 0.00055,
            "hourly_cost_per_device": 3.50,
            "warm_pool_floor": 0.25,
            "cold_allowance_ms": serving.slo_ms * 20,
            "policy_guardrail": variant.guardrail_metric,
            "first_mitigation": "reduce utilization with replicas, smaller batches, or load shedding",
            "ops_risks": {
                "arrival_shift": "Arrival traces shift after launch, invalidating the p99 frontier.",
                "cold_start": "Scale-out cold starts expose users before warm capacity is ready.",
                "state_growth": f"Live {serving.state_kind} grows faster than capacity planning assumed.",
            },
        }
        overrides = {
            "iphone": {
                "amount_focus": "local requests/sec, app p99, energy/request, thermal headroom, and fallback share",
                "replica_label": "local lanes / bounded fallback paths",
                "batch_amount": "UI requests per local batch",
                "queue_failure": "background work makes the app miss its responsiveness budget",
                "cost_label": "Energy/request",
                "cost_unit": "mJ/request",
                "cost_limit": 650.0,
                "hourly_cost_per_device": 0.0,
                "warm_pool_floor": 0.0,
                "cold_allowance_ms": serving.slo_ms * 8,
                "first_mitigation": "prefer batch size 1-2, cap fallback, and shed noninteractive work",
                "ops_risks": {
                    "thermal": "Thermal throttling lengthens service time after sustained use.",
                    "fallback": "Cloud fallback share grows under burst load and harms privacy or battery.",
                    "arrival_shift": "Background app work changes the local arrival distribution.",
                },
            },
            "oura_ring": {
                "amount_focus": "sensor windows, wake duty cycle, data freshness, state bytes, and battery/day",
                "replica_label": "serving windows / phone-mediated sync lanes",
                "batch_amount": "sensor windows per wake",
                "queue_failure": "a sparse queue still misses the sensing cadence when wake windows slip",
                "cost_label": "Wake energy/window",
                "cost_unit": "mJ/window",
                "cost_limit": 45.0,
                "hourly_cost_per_device": 0.0,
                "warm_pool_floor": 0.0,
                "cold_allowance_ms": serving.slo_ms * 8,
                "first_mitigation": "defer nonurgent sync, keep batch size 1, and preserve duty-cycle slack",
                "ops_risks": {
                    "freshness": "Deferred phone sync creates stale summaries before upload.",
                    "battery": "Radio wakeups dominate the expected energy/window.",
                    "state_growth": "Sensor window buffers exceed the firmware memory reserve.",
                },
            },
            "robotaxi": {
                "amount_focus": "frames/sec, p99/p999 deadline, warm spare margin, and power per frame",
                "replica_label": "perception lanes / warm safety spares",
                "batch_amount": "sensor frames per scheduler batch",
                "queue_failure": "a sensor burst crosses the perception deadline and consumes safety margin",
                "cost_label": "Power cost/frame",
                "cost_unit": "J/frame",
                "cost_limit": 2.4,
                "hourly_cost_per_device": 0.0,
                "warm_pool_floor": 0.5,
                "cold_allowance_ms": serving.slo_ms * 4,
                "first_mitigation": "use batch size 1, bounded queues, priority lanes, and warm fallback",
                "ops_risks": {
                    "p999": "Rare sensor bursts push p999 past the perception deadline.",
                    "fallback": "Fallback perception path lacks enough warm safety margin.",
                    "power": "Extra warm lanes exceed the vehicle power envelope.",
                },
            },
            "cloud_fleet": {
                "amount_focus": "QPS, batch size, replicas, utilization, p99 SLA, and cost/request",
                "replica_label": "API replicas",
                "batch_amount": "requests per accelerator batch",
                "queue_failure": "multi-tenant demand drives replicas through the queueing knee",
                "cost_label": "Cost/request",
                "cost_unit": "USD/request",
                "cost_limit": 0.00055,
                "hourly_cost_per_device": 3.50,
                "warm_pool_floor": 0.4,
                "cold_allowance_ms": serving.slo_ms * 20,
                "first_mitigation": "autoscale earlier, keep a warm pool, or shed/degrade excess requests",
                "ops_risks": {
                    "arrival_shift": "Customer traffic shifts from Poisson-like load to synchronized bursts.",
                    "cold_start": "Scale-out cold starts lengthen p99 during the same event that needs capacity.",
                    "state_growth": "KV cache growth lowers live concurrency below the admission target.",
                },
            },
        }
        base_risks = dict(common["ops_risks"])
        override = overrides.get(track_id, {})
        override_risks = dict(override.get("ops_risks", {}))
        packet = dict(common)
        packet.update(override)
        packet["ops_risks"] = {**base_risks, **override_risks}
        return packet

    def v1_13_fmt_amount(value, unit, precision=2):
        if not math.isfinite(value):
            return "unstable"
        if abs(value) >= 100:
            return f"{value:,.0f} {unit}"
        if abs(value) >= 10:
            return f"{value:,.1f} {unit}"
        return f"{value:,.{precision}f} {unit}"

    def v1_13_metric_card(label, value, detail, color, border=False):
        border_style = f"2px solid {color}" if border else "1px solid #e2e8f0"
        return f"""
        <div style="padding:16px; border:{border_style}; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {color}; flex:1;">
            <div style="color:#64748b; font-size:0.78rem; font-weight:700;">{html.escape(str(label))}</div>
            <div style="font-size:1.55rem; font-weight:800; color:{color};">{html.escape(str(value))}</div>
            <div style="font-size:0.72rem; color:#64748b;">{html.escape(str(detail))}</div>
        </div>
        """

    def v1_13_html_table(headers, rows):
        header_html = "".join(
            f"<th style='text-align:left; padding:8px 10px; border-bottom:1px solid #cbd5e1;'>{html.escape(str(h))}</th>"
            for h in headers
        )
        row_html = ""
        for row in rows:
            row_html += "<tr>"
            for cell in row:
                row_html += (
                    "<td style='padding:8px 10px; border-bottom:1px solid #e2e8f0; "
                    f"vertical-align:top;'>{html.escape(str(cell))}</td>"
                )
            row_html += "</tr>"
        return f"""
        <div style="overflow-x:auto; margin:14px 0;">
            <table style="width:100%; border-collapse:collapse; background:white; font-size:0.86rem;">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{row_html}</tbody>
            </table>
        </div>
        """

    def v1_13_request_cost(serving, packet, *, arrival_qps, service_ms, replicas, warm_pool, buffer_pct, throughput_gain):
        arrival = max(0.0001, float(arrival_qps))
        planned_replicas = max(1.0, float(replicas) * (1.0 + max(0.0, buffer_pct) / 100.0))
        warm = max(0.0, float(warm_pool))
        if packet["cost_unit"] == "USD/request":
            hourly = packet["hourly_cost_per_device"]
            devices = max(1, serving.default_devices_per_replica)
            standing_units = planned_replicas + 0.35 * warm
            value = standing_units * devices * hourly / (arrival * 3600.0)
        else:
            active_j = serving.tdp_w * (service_ms / 1000.0) / max(1.0, throughput_gain)
            standby_j = serving.tdp_w * 0.04 * max(0.0, planned_replicas + warm - 1.0) / arrival
            value_j = active_j + standby_j
            value = value_j if packet["cost_unit"] == "J/frame" else value_j * 1000.0
        return {
            "value": value,
            "label": packet["cost_label"],
            "unit": packet["cost_unit"],
            "limit": packet["cost_limit"],
            "ok": value <= packet["cost_limit"],
        }

    def v1_13_policy_candidates(serving, track_id):
        default_batch = max(1, int(serving.batch_size))
        default_replicas = max(1, int(serving.replicas))
        default_warm = max(0, int(serving.warm_pool_replicas))
        candidates = {
            "default": {
                "label": "Track default policy",
                "batch_size": default_batch,
                "replicas": default_replicas,
                "buffer_pct": 0,
                "warm_pool": default_warm,
                "demand_multiplier": 1.0,
                "rationale": "Use the variant defaults as the baseline launch posture.",
            },
            "batch_efficiency": {
                "label": "Batch for throughput",
                "batch_size": min(64, max(2, default_batch * 2 if default_batch > 1 else 4)),
                "replicas": default_replicas,
                "buffer_pct": 10,
                "warm_pool": default_warm,
                "demand_multiplier": 1.0,
                "rationale": "Spend some latency budget to improve throughput per serving unit.",
            },
            "tail_headroom": {
                "label": "Scale for tail headroom",
                "batch_size": max(1, default_batch // 2),
                "replicas": max(default_replicas + 1, math.ceil(default_replicas * 1.5)),
                "buffer_pct": 30,
                "warm_pool": max(default_warm, 1),
                "demand_multiplier": 1.0,
                "rationale": "Buy lower utilization and p99 headroom with more standing capacity.",
            },
            "fallback_defer": {
                "label": "Bounded fallback/defer path",
                "batch_size": 1,
                "replicas": default_replicas,
                "buffer_pct": 15,
                "warm_pool": default_warm,
                "demand_multiplier": 0.8,
                "rationale": "Admit only the urgent path locally and defer or route noncritical work.",
            },
        }
        if track_id == "robotaxi":
            candidates["fallback_defer"].update({
                "label": "Priority safety path",
                "replicas": max(2, default_replicas + 1),
                "buffer_pct": 35,
                "warm_pool": max(1, default_warm),
                "demand_multiplier": 1.0,
                "rationale": "Keep batch size 1 and reserve a warm safety lane for bursts.",
            })
        elif track_id == "oura_ring":
            candidates["fallback_defer"].update({
                "label": "Deferred phone sync",
                "demand_multiplier": 0.65,
                "rationale": "Serve the local window and defer nonurgent phone-mediated sync.",
            })
        elif track_id == "cloud_fleet":
            candidates["fallback_defer"].update({
                "label": "Admission control + warm pool",
                "replicas": default_replicas,
                "warm_pool": max(2, default_warm),
                "demand_multiplier": 0.9,
                "rationale": "Shed/degrade excess traffic while keeping warm replicas for scale-out.",
            })
        return candidates

    def v1_13_evaluate_policy(
        serving,
        model,
        packet,
        candidate,
        *,
        cost_multiplier=1.0,
        capacity_reserve_pct=20,
    ):
        batch_size = max(1, int(candidate["batch_size"]))
        visible_replicas = max(1, int(candidate["replicas"]))
        buffer_pct = max(0.0, float(candidate["buffer_pct"]))
        planned_replicas = max(1, math.ceil(visible_replicas * (1 + buffer_pct / 100.0)))
        arrival_qps = max(0.0001, serving.arrival_qps * float(candidate["demand_multiplier"]))
        batching = batching_tax(
            batch_size=batch_size,
            arrival_qps=arrival_qps,
            service_ms=serving.service_ms,
            slo_ms=serving.slo_ms,
            efficiency_gain=serving.batch_efficiency_gain,
            replicas=planned_replicas,
            service_cv=serving.service_cv,
        )
        queue = queueing_latency(
            arrival_qps=arrival_qps / batch_size,
            service_ms=batching.batched_service_ms,
            replicas=planned_replicas,
            service_cv=serving.service_cv,
            slo_ms=serving.slo_ms,
        )
        capacity = cache_capacity(
            serving,
            model,
            context_tokens=serving.context_tokens,
            precision_bytes=serving.precision_bytes,
            devices_per_replica=serving.default_devices_per_replica,
            kv_precision_bytes=serving.kv_precision_bytes,
        )
        p99_ms = batching.total_p99_ms
        live_required = math.inf if not math.isfinite(p99_ms) else max(1, math.ceil(arrival_qps * p99_ms / 1000.0))
        live_with_reserve = math.inf if not math.isfinite(live_required) else math.ceil(live_required * (1 + capacity_reserve_pct / 100.0))
        capacity_ok = (not capacity.oom) and capacity.max_concurrent >= live_with_reserve
        cost = v1_13_request_cost(
            serving,
            packet,
            arrival_qps=arrival_qps,
            service_ms=batching.batched_service_ms,
            replicas=planned_replicas,
            warm_pool=int(candidate["warm_pool"]),
            buffer_pct=0,
            throughput_gain=batching.throughput_gain,
        )
        cost_limit = cost["limit"] * max(0.1, float(cost_multiplier))
        cost_ok = cost["value"] <= cost_limit
        cold = cold_start_latency(
            serving,
            precision_bytes=serving.precision_bytes,
            warm_pool_replicas=int(candidate["warm_pool"]),
            scale_out_replicas=max(1, planned_replicas - serving.replicas + 1),
        )
        warm_ok = (
            cold.protected_fraction >= packet["warm_pool_floor"]
            or cold.exposed_first_request_ms <= packet["cold_allowance_ms"]
        )
        checks = [
            ("p99/SLO", batching.slo_ok, f"{p99_ms:.1f} ms <= {serving.slo_ms:g} ms"),
            ("capacity/state", capacity_ok, f"{live_with_reserve} live <= {capacity.max_concurrent} fit"),
            (
                packet["cost_label"],
                cost_ok,
                f"{v1_13_fmt_amount(cost['value'], cost['unit'])} <= {v1_13_fmt_amount(cost_limit, cost['unit'])}",
            ),
            ("warm scale-out", warm_ok, f"{cold.protected_fraction*100:.0f}% protected"),
        ]
        binding = "none - all guardrails pass"
        for name, ok, detail in checks:
            if not ok:
                binding = f"{name}: {detail}"
                break
        return {
            "candidate": candidate,
            "arrival_qps": arrival_qps,
            "batching": batching,
            "queue": queue,
            "capacity": capacity,
            "cold": cold,
            "cost": cost,
            "cost_limit": cost_limit,
            "planned_replicas": planned_replicas,
            "live_required": live_required,
            "live_with_reserve": live_with_reserve,
            "capacity_ok": capacity_ok,
            "cost_ok": cost_ok,
            "warm_ok": warm_ok,
            "feasible": batching.slo_ok and capacity_ok and cost_ok and warm_ok,
            "binding": binding,
            "checks": checks,
        }

    return (
        v1_13_evaluate_policy,
        v1_13_fmt_amount,
        v1_13_html_table,
        v1_13_metric_card,
        v1_13_policy_candidates,
        v1_13_request_cost,
        v1_13_track_packet,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
        track_arc_context,
    v1_13_metadata,
    v1_13_profile,
    v1_13_serving,
    v1_13_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 13
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The Tail Latency Trap
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Batching &middot; Queueing &middot; Replicas &middot; Launch Policy
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 760px; line-height: 1.65;">
                {v1_13_variant.workload_summary} The average path can look healthy
                while p99 violates {v1_13_variant.guardrail_metric}.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~50 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_13_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_13_serving.hardware_ref}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">M/M/c Queue</span>
                <span class="badge badge-warn">Batching Tax</span>
                <span class="badge badge-fail">{v1_13_serving.state_kind}</span>
            </div>
        </div>
        """),
        track_context(v1_13_profile),
        track_arc_context(v1_13_profile, v1_13_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_13_serving, v1_13_variant):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Measure batching as a trade:</strong>
                    throughput gain and formation delay land in the same p99 budget.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose queueing tails:</strong>
                    arrival rate and utilization can fail p99 while mean latency looks fine.</div>
                <div style="margin-bottom: 3px;">3. <strong>Launch with guardrails:</strong>
                    replicas, warm capacity, {v1_13_serving.state_kind}, and cost/energy must pass together.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Queueing theory from the Model Serving chapter &middot;
                    Memory and state accounting from Hardware Acceleration and Inference at Scale
                </div>
            </div>
            <div style="flex: 0 0 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Track Defaults
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    {v1_13_serving.arrival_qps:g} QPS &middot;
                    {v1_13_serving.service_ms:g} ms service &middot;
                    {v1_13_serving.slo_ms:g} ms p99 SLO
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 12px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                "For {v1_13_serving.label}, can a serving policy meet
                {v1_13_serving.slo_ms:g} ms p99 while protecting
                {v1_13_variant.guardrail_metric}?"
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** - Complete before this lab:

    - **The Model Serving chapter** - queuing theory, batching strategies,
      live state/cache management, autoscaling, and cold start latency.
    """), kind="info")
    return


# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    partA_pred = mo.ui.radio(
        options={
            "A) Batching only improves throughput": "throughput_only",
            "B) Batching improves throughput but can spend latency": "tradeoff",
            "C) Batching only changes memory": "memory_only",
            "D) Batching removes queueing": "removes_queueing",
        },
        label=(
            f"{v1_13_serving.label}: if we batch {v1_13_serving.batch_size} request(s), "
            "what changes first?"
        ),
    )
    return (partA_pred,)


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    partA_batch = mo.ui.slider(
        start=1,
        stop=64,
        value=v1_13_serving.batch_size,
        step=1,
        label="Batch size",
    )
    if v1_13_serving.arrival_qps < 1:
        _arr_start = 0.01
        _arr_stop = 1.0
        _arr_step = 0.01
        _arr_value = v1_13_serving.arrival_qps
    else:
        _arr_start = max(1.0, v1_13_serving.arrival_qps * 0.25)
        _arr_stop = max(10.0, v1_13_serving.arrival_qps * 4)
        _arr_step = max(1.0, v1_13_serving.arrival_qps / 20)
        _arr_value = v1_13_serving.arrival_qps
    partA_arr = mo.ui.slider(
        start=_arr_start,
        stop=_arr_stop,
        value=_arr_value,
        step=_arr_step,
        label="Arrival rate (QPS)",
    )
    partA_slo = mo.ui.slider(
        start=10,
        stop=max(500, int(v1_13_serving.slo_ms * 3)),
        value=int(v1_13_serving.slo_ms),
        step=5,
        label="SLO budget (ms)",
    )
    partA_checkpoint = mo.ui.radio(
        options={
            "Keep this batch size": "keep",
            "Reduce batching for latency": "reduce",
            "Reject batching for this track": "reject",
        },
        label="Checkpoint: what batching decision belongs in the launch memo?",
    )
    return (partA_arr, partA_batch, partA_checkpoint, partA_slo)


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    partB_pred = mo.ui.radio(
        options={
            "A) Mean latency proves the service is healthy": "mean",
            "B) Utilization and p99 reveal the failure": "tail",
            "C) Arrival rate only affects cost": "cost",
            "D) Replicas remove all queueing": "replicas",
        },
        label=(
            f"{v1_13_serving.label}: average service is {v1_13_serving.service_ms:g} ms. "
            "Which evidence decides whether traffic is safe?"
        ),
    )
    return (partB_pred,)


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    if v1_13_serving.arrival_qps < 1:
        _arr_start = 0.001
        _arr_stop = max(0.5, v1_13_serving.arrival_qps * 20)
        _arr_step = 0.001
        _arr_value = v1_13_serving.arrival_qps
    else:
        _arr_start = max(1.0, v1_13_serving.arrival_qps * 0.25)
        _arr_stop = max(10.0, v1_13_serving.arrival_qps * 4)
        _arr_step = max(1.0, v1_13_serving.arrival_qps / 25)
        _arr_value = v1_13_serving.arrival_qps
    partB_arr = mo.ui.slider(
        start=_arr_start,
        stop=_arr_stop,
        value=_arr_value,
        step=_arr_step,
        label="Arrival rate (QPS)",
    )
    partB_svc = mo.ui.slider(
        start=1.0,
        stop=max(200.0, v1_13_serving.service_ms * 3),
        value=v1_13_serving.service_ms,
        step=1.0,
        label="Service time (ms)",
    )
    partB_slo = mo.ui.slider(
        start=10,
        stop=max(500, int(v1_13_serving.slo_ms * 3)),
        value=int(v1_13_serving.slo_ms),
        step=5,
        label="SLO budget (ms)",
    )
    partB_checkpoint = mo.ui.radio(
        options={
            "Operate below the queueing knee": "headroom",
            "Watch mean latency only": "mean",
            "Accept p99 violations at peak": "accept_tail",
        },
        label="Checkpoint: what queueing rule should carry forward?",
    )
    return (partB_arr, partB_checkpoint, partB_slo, partB_svc)


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    partC_pred = mo.ui.radio(
        options={
            "A) More replicas always solve the serving problem": "always",
            "B) Replicas trade p99 headroom against cost and warm capacity": "tradeoff",
            "C) Replicas change only memory": "memory",
            "D) Autoscaling has no p99 effect": "none",
        },
        label=(
            f"{v1_13_serving.label}: should we scale beyond "
            f"{v1_13_serving.replicas} serving unit(s)?"
        ),
    )
    return (partC_pred,)


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    partC_replicas = mo.ui.slider(
        start=1,
        stop=max(16, v1_13_serving.replicas * 3),
        value=v1_13_serving.replicas,
        step=1,
        label="Visible serving units",
    )
    partC_buffer = mo.ui.slider(
        start=0,
        stop=100,
        value=25,
        step=5,
        label="Autoscale headroom (%)",
    )
    partC_warm_pool = mo.ui.slider(
        start=0,
        stop=max(8, v1_13_serving.scale_out_replicas * 2),
        value=v1_13_serving.warm_pool_replicas,
        step=1,
        label="Warm pool units",
    )
    partC_checkpoint = mo.ui.radio(
        options={
            "Minimum policy that passes p99": "minimum_pass",
            "Overprovision for safety": "overprovision",
            "Keep default and accept risk": "accept_risk",
        },
        label="Checkpoint: which replica/autoscale posture should the report use?",
    )
    return (partC_buffer, partC_checkpoint, partC_replicas, partC_warm_pool)


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    partD_pred = mo.ui.radio(
        options={
            "A) Fastest p99 is automatically launchable": "fastest",
            "B) Lowest cost/request is automatically launchable": "cheapest",
            "C) Launch requires all SLO, capacity, cost, and guardrails to pass": "all_guardrails",
            "D) Warm pools are only an operations detail": "warm_later",
        },
        label=(
            f"{v1_13_serving.label}: what makes a serving policy valid enough to launch?"
        ),
    )
    return (partD_pred,)


@app.cell(hide_code=True)
def _(mo, v1_13_policy_candidates, v1_13_serving, v1_13_track_id):
    _candidates = v1_13_policy_candidates(v1_13_serving, v1_13_track_id)
    _options = {spec["label"]: key for key, spec in _candidates.items()}
    partD_policy = mo.ui.dropdown(
        options=_options,
        value="Track default policy",
        label="Selected launch policy",
    )
    partD_cost_multiplier = mo.ui.slider(
        start=0.5,
        stop=1.5,
        value=1.0,
        step=0.05,
        label="Cost/energy guardrail multiplier",
    )
    partD_capacity_reserve = mo.ui.slider(
        start=0,
        stop=100,
        value=20,
        step=5,
        label="Live-state reserve (%)",
    )
    partD_reject = mo.ui.dropdown(
        options=_options,
        value="Batch for throughput",
        label="Rejected alternative",
    )
    partD_ops_risk = mo.ui.radio(
        options={
            "Arrival trace shifts after launch": "arrival_shift",
            "Warm capacity is not ready during scale-out": "cold_start",
            f"Live {v1_13_serving.state_kind} grows faster than planned": "state_growth",
        },
        label="Carry-forward operations risk",
    )
    return (
        partD_capacity_reserve,
        partD_cost_multiplier,
        partD_ops_risk,
        partD_policy,
        partD_reject,
    )


# ===========================================================================
# ZONE C: MAIN LAB
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    batching_tax,
    cache_capacity,
    cold_start_latency,
    go,
    math,
    mo,
    np,
    partA_arr,
    partA_batch,
    partA_checkpoint,
    partA_pred,
    partA_slo,
    partB_arr,
    partB_checkpoint,
    partB_pred,
    partB_slo,
    partB_svc,
    partC_buffer,
    partC_checkpoint,
    partC_pred,
    partC_replicas,
    partC_warm_pool,
    partD_capacity_reserve,
    partD_cost_multiplier,
    partD_ops_risk,
    partD_policy,
    partD_pred,
    partD_reject,
    queueing_latency,
    v1_13_evaluate_policy,
    v1_13_fmt_amount,
    v1_13_html_table,
    v1_13_metric_card,
    v1_13_policy_candidates,
    v1_13_request_cost,
    v1_13_track_id,
    v1_13_track_packet,
    v1_13_model,
    v1_13_profile,
    v1_13_serving,
    v1_13_variant,
):
    v1_13_packet = v1_13_track_packet(v1_13_track_id, v1_13_serving, v1_13_variant)
    v1_13_candidates = v1_13_policy_candidates(v1_13_serving, v1_13_track_id)

    def v1_13_part_banner(part, title, color, body):
        return mo.Html(f"""
        <div style="border-left:4px solid {color}; background:white;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;
                    box-shadow:0 1px 4px rgba(0,0,0,0.06);">
            <div style="font-size:0.72rem; font-weight:700; color:{color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Part {part} Concept Module &middot; {v1_13_packet['stakeholder']}
            </div>
            <div style="font-weight:800; font-size:1.1rem; color:#0f172a; margin-bottom:6px;">
                {title}
            </div>
            <div style="font-size:0.94rem; color:#334155; line-height:1.6;">{body}</div>
        </div>
        """)

    def v1_13_prediction_feedback(actual_key, selected, correct_text, miss_text):
        if selected == actual_key:
            return mo.callout(mo.md(correct_text), kind="success")
        return mo.callout(mo.md(miss_text), kind="warn")

    def build_part_a():
        items = [
            v1_13_part_banner(
                "A",
                "Batching Changes Throughput And Latency Together",
                COLORS["BlueLine"],
                (
                    f"{v1_13_serving.label} is deciding whether the default batch policy can launch. "
                    f"The amount system is {v1_13_packet['amount_focus']}."
                ),
            ),
            mo.md(f"""
## Concept: Batching Is A Trade, Not A Free Win

A batch can improve throughput per serving unit, but the first request in the
batch waits while the batch forms. For {v1_13_serving.label}, that wait is paid
inside the same {v1_13_serving.slo_ms:g} ms p99 budget as model service time.
            """),
            partA_pred,
        ]
        if partA_pred.value is None:
            items.append(mo.callout(mo.md("Commit to a batching prediction before opening the latency decomposition."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_batch, partA_arr, partA_slo], widths="equal"))
        batch = partA_batch.value
        arrival = partA_arr.value
        slo = partA_slo.value
        result = batching_tax(
            batch_size=batch,
            arrival_qps=arrival,
            service_ms=v1_13_serving.service_ms,
            slo_ms=slo,
            efficiency_gain=v1_13_serving.batch_efficiency_gain,
            replicas=v1_13_serving.replicas,
            service_cv=v1_13_serving.service_cv,
        )
        terms = [
            ("Formation delay", result.formation_delay_ms, COLORS["OrangeLine"]),
            ("Batched service", result.batched_service_ms, COLORS["BlueLine"]),
            ("Queue tail", result.queue_p99_ms, COLORS["RedLine"]),
        ]
        binding_term = max(terms, key=lambda item: item[1])[0]
        fig = go.Figure()
        for name, value, color in terms:
            fig.add_trace(go.Bar(
                x=["Total p99"],
                y=[value],
                name=name,
                marker_color=color,
                hovertemplate=f"{name}: %{{y:.1f}} ms<extra></extra>",
            ))
        fig.add_hline(y=slo, line_dash="dash", line_color=COLORS["GreenLine"], annotation_text=f"SLO = {slo:g} ms")
        fig.update_layout(
            barmode="stack",
            height=330,
            yaxis=dict(title="Latency contribution (ms)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.16, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        total_color = COLORS["GreenLine"] if result.slo_ok else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_13_metric_card("Formation Delay", f"{result.formation_delay_ms:.1f} ms", f"{result.formation_slo_pct:.0f}% of SLO", COLORS["OrangeLine"])}
            {v1_13_metric_card("Throughput Gain", f"{result.throughput_gain:.1f}x", "amortized service", COLORS["BlueLine"])}
            {v1_13_metric_card("Batch Utilization", f"{result.utilization:.2f}", "effective batch queue", COLORS["OrangeLine"])}
            {v1_13_metric_card("Total P99", f"{result.total_p99_ms:.1f} ms", binding_term, total_color, True)}
        </div>
        """))
        items.append(mo.Html(v1_13_html_table(
            ("Term", "Amount", "Decision meaning"),
            (
                ("Batch size", batch, v1_13_packet["batch_amount"]),
                ("Formation delay", f"{result.formation_delay_ms:.1f} ms", "waiting before inference starts"),
                ("Batched service", f"{result.batched_service_ms:.1f} ms", "service time after throughput gain"),
                ("Queue p99", f"{result.queue_p99_ms:.1f} ms", "remaining tail delay"),
                ("Total p99", f"{result.total_p99_ms:.1f} ms", "PASS" if result.slo_ok else "FAIL"),
            ),
        )))
        if result.slo_ok:
            items.append(mo.callout(mo.md(
                f"**Boundary status:** total p99 is {result.total_p99_ms:.1f} ms within the {slo:g} ms SLO. "
                f"The binding amount is {binding_term.lower()}."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**SLO violated.** Total p99 is {result.total_p99_ms:.1f} ms against {slo:g} ms. "
                f"Mitigation: reduce batch size, increase arrival-adaptive scheduling, or reserve more capacity."
            ), kind="danger"))
        items.append(v1_13_prediction_feedback(
            "tradeoff",
            partA_pred.value,
            "**Correct.** Batching changes throughput and latency at the same time.",
            "**The batching trap:** throughput improves only after the request pays formation delay and queueing tax.",
        ))
        items.append(mo.accordion({
            "Math Peek / Source Model - batching tax": mo.md(f"""
```
formation_delay_ms = (batch_size - 1) / (2 * arrival_qps) * 1000
batched_service_ms = service_ms * (1 + 0.08*log2(batch_size)) / efficiency_gain
total_p99_ms       = formation_delay + batched_service + queue_tail
```

Current values: batch `{batch}`, arrival `{arrival:g}` QPS, total p99
`{result.total_p99_ms:.1f}` ms. Source helper: `mlsysbook_labs.batching_tax`.
Chapter anchor: Traffic-Aware Batching Strategy and the batching-tax formula.
            """)
        }))
        items.append(partA_checkpoint)
        if partA_checkpoint.value is None:
            items.append(mo.callout(mo.md("Checkpoint: choose how the batching decision should appear in the launch memo."), kind="info"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            v1_13_part_banner(
                "B",
                "Utilization And Arrival Rate Create Tail-Latency Failure",
                COLORS["OrangeLine"],
                (
                    f"{v1_13_variant.stakeholder} sees average latency below budget. "
                    f"The failure to test is: {v1_13_packet['queue_failure']}."
                ),
            ),
            mo.md("""
## Concept: Queueing Makes Capacity Planning Nonlinear

Requests wait behind other requests. Once arrival rate pushes utilization toward
1.0, mean latency rises and p99/p999 rise faster. The decision evidence is the
latency distribution, not the average alone.
            """),
            partB_pred,
        ]
        if partB_pred.value is None:
            items.append(mo.callout(mo.md("Commit to a queueing prediction before opening the arrival-rate sweep."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_arr, partB_svc, partB_slo], widths="equal"))
        arrival = partB_arr.value
        svc = partB_svc.value
        slo = partB_slo.value
        selected = queueing_latency(
            arrival_qps=arrival,
            service_ms=svc,
            replicas=v1_13_serving.replicas,
            service_cv=v1_13_serving.service_cv,
            slo_ms=slo,
        )
        if v1_13_serving.arrival_qps < 1:
            low = max(0.001, arrival * 0.2)
            high = max(0.5, arrival * 5)
        else:
            low = max(0.1, arrival * 0.25)
            high = max(arrival * 2.5, v1_13_serving.replicas * (1000.0 / svc) * 1.05)
        xs = np.linspace(low, high, 60)
        means = []
        p95s = []
        p99s = []
        p999s = []
        utils = []
        for x in xs:
            q = queueing_latency(
                arrival_qps=float(x),
                service_ms=svc,
                replicas=v1_13_serving.replicas,
                service_cv=v1_13_serving.service_cv,
                slo_ms=slo,
            )
            means.append(q.mean_latency_ms if math.isfinite(q.mean_latency_ms) else None)
            p95s.append(q.p95_latency_ms if math.isfinite(q.p95_latency_ms) else None)
            p99s.append(q.p99_latency_ms if math.isfinite(q.p99_latency_ms) else None)
            p999s.append(q.p999_latency_ms if math.isfinite(q.p999_latency_ms) else None)
            utils.append(q.utilization)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=means, mode="lines", name="Mean", line=dict(color=COLORS["BlueLine"], width=2)))
        fig.add_trace(go.Scatter(x=xs, y=p95s, mode="lines", name="P95", line=dict(color=COLORS["OrangeLine"], width=2)))
        fig.add_trace(go.Scatter(x=xs, y=p99s, mode="lines", name="P99", line=dict(color=COLORS["RedLine"], width=3)))
        if v1_13_track_id == "robotaxi":
            fig.add_trace(go.Scatter(x=xs, y=p999s, mode="lines", name="P999", line=dict(color="#7f1d1d", width=2, dash="dot")))
        fig.add_hline(y=slo, line_dash="dash", line_color=COLORS["GreenLine"], annotation_text=f"SLO = {slo:g} ms")
        fig.add_trace(go.Scatter(
            x=[arrival],
            y=[selected.p99_latency_ms if math.isfinite(selected.p99_latency_ms) else None],
            mode="markers",
            name="selected p99",
            marker=dict(color=COLORS["RedLine"], size=13, symbol="diamond"),
        ))
        fig.update_layout(
            height=370,
            xaxis=dict(title="Arrival rate (QPS)"),
            yaxis=dict(title="Latency (ms)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.15, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        p99_color = COLORS["GreenLine"] if selected.slo_ok else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_13_metric_card("Utilization", f"{selected.utilization:.2f}", f"{v1_13_serving.replicas} replica(s)", COLORS["OrangeLine"])}
            {v1_13_metric_card("Mean", f"{selected.mean_latency_ms:.1f} ms", "wait + service", COLORS["BlueLine"])}
            {v1_13_metric_card("P99", f"{selected.p99_latency_ms:.1f} ms", f"SLO {slo:g} ms", p99_color, True)}
            {v1_13_metric_card("P999", f"{selected.p999_latency_ms:.1f} ms", "rare tail", COLORS["RedLine"])}
        </div>
        """))
        rows = []
        for factor in (0.75, 1.0, 1.25):
            x = max(0.0001, arrival * factor)
            q = queueing_latency(
                arrival_qps=x,
                service_ms=svc,
                replicas=v1_13_serving.replicas,
                service_cv=v1_13_serving.service_cv,
                slo_ms=slo,
            )
            rows.append((f"{x:g} QPS", f"{q.utilization:.2f}", f"{q.mean_latency_ms:.1f} ms", f"{q.p99_latency_ms:.1f} ms", "PASS" if q.slo_ok else "FAIL"))
        items.append(mo.Html(v1_13_html_table(("Arrival", "Utilization", "Mean", "P99", "SLO"), rows)))
        if selected.slo_ok:
            items.append(mo.callout(mo.md(
                f"**Queueing boundary holds.** P99 is {selected.p99_latency_ms:.1f} ms, but the margin depends on staying below utilization {selected.utilization:.2f}."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Tail failure.** P99 is {selected.p99_latency_ms:.1f} ms > {slo:g} ms at utilization {selected.utilization:.2f}. "
                f"First mitigation: {v1_13_packet['first_mitigation']}."
            ), kind="danger"))
        items.append(v1_13_prediction_feedback(
            "tail",
            partB_pred.value,
            "**Correct.** Utilization and p99 expose the failure that the mean hides.",
            "**Average latency is incomplete evidence.** The serving chapter uses p95/p99 because queueing tails determine user-visible failure.",
        ))
        items.append(mo.accordion({
            "Math Peek / Source Model - M/M/c tail": mo.md(f"""
```
mu       = 1000 / service_ms
rho      = arrival_qps / (replicas * mu)
p99      = service_ms + queue_tail(rho, replicas, service_cv)
Little's Law: live_requests ~= arrival_qps * latency_seconds
```

Current values: arrival `{arrival:g}` QPS, service `{svc:g}` ms,
replicas `{v1_13_serving.replicas}`, rho `{selected.utilization:.2f}`,
p99 `{selected.p99_latency_ms:.1f}` ms. Source helper:
`mlsysbook_labs.queueing_latency`. Chapter anchor: Queuing Theory for Capacity
Planning and Tail Latency and Headroom.
            """)
        }))
        items.append(partB_checkpoint)
        if partB_checkpoint.value is None:
            items.append(mo.callout(mo.md("Checkpoint: choose the queueing rule that should constrain Part C."), kind="info"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            v1_13_part_banner(
                "C",
                "Replicas And Autoscaling Trade P99, Utilization, And Cost",
                COLORS["GreenLine"],
                (
                    f"The operations review can add {v1_13_packet['replica_label']}, but every extra unit "
                    f"has a {v1_13_packet['cost_label'].lower()} and warm-capacity consequence."
                ),
            ),
            mo.md("""
## Concept: Scale-Out Buys Headroom, Not A Free Guarantee

Replicas reduce utilization and usually lower p99. They also raise the standing
capacity bill and create a warm-pool/cold-start obligation during scale-out.
            """),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Commit to a replica/autoscaling prediction before opening the frontier."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_replicas, partC_buffer, partC_warm_pool], widths="equal"))
        batch = max(1, partA_batch.value)
        arrival = max(0.0001, partB_arr.value)
        service = partB_svc.value
        slo = partB_slo.value
        frontier = []
        max_visible = max(4, min(32, max(v1_13_serving.replicas * 3, partC_replicas.value + 4)))
        for visible in range(1, max_visible + 1):
            planned = max(1, math.ceil(visible * (1 + partC_buffer.value / 100.0)))
            b = batching_tax(
                batch_size=batch,
                arrival_qps=arrival,
                service_ms=service,
                slo_ms=slo,
                efficiency_gain=v1_13_serving.batch_efficiency_gain,
                replicas=planned,
                service_cv=v1_13_serving.service_cv,
            )
            q = queueing_latency(
                arrival_qps=arrival / batch,
                service_ms=b.batched_service_ms,
                replicas=planned,
                service_cv=v1_13_serving.service_cv,
                slo_ms=slo,
            )
            cost = v1_13_request_cost(
                v1_13_serving,
                v1_13_packet,
                arrival_qps=arrival,
                service_ms=b.batched_service_ms,
                replicas=planned,
                warm_pool=partC_warm_pool.value,
                buffer_pct=0,
                throughput_gain=b.throughput_gain,
            )
            cold = cold_start_latency(
                v1_13_serving,
                precision_bytes=v1_13_serving.precision_bytes,
                warm_pool_replicas=partC_warm_pool.value,
                scale_out_replicas=max(1, planned - v1_13_serving.replicas + 1),
            )
            p999 = b.formation_delay_ms + b.batched_service_ms + max(0.0, q.p999_latency_ms - b.batched_service_ms)
            status = "PASS" if b.slo_ok and cost["value"] <= cost["limit"] else "FAIL"
            frontier.append({
                "visible": visible,
                "planned": planned,
                "batching": b,
                "queue": q,
                "cost": cost,
                "cold": cold,
                "p999": p999,
                "status": status,
            })
        selected = frontier[partC_replicas.value - 1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[row["planned"] for row in frontier],
            y=[row["batching"].total_p99_ms for row in frontier],
            mode="lines+markers",
            name="P99 latency",
            line=dict(color=COLORS["RedLine"], width=3),
        ))
        fig.add_trace(go.Scatter(
            x=[row["planned"] for row in frontier],
            y=[row["cost"]["value"] for row in frontier],
            mode="lines+markers",
            name=v1_13_packet["cost_label"],
            yaxis="y2",
            line=dict(color=COLORS["BlueLine"], width=2),
        ))
        fig.add_hline(y=slo, line_dash="dash", line_color=COLORS["GreenLine"], annotation_text=f"SLO = {slo:g} ms")
        fig.add_trace(go.Scatter(
            x=[selected["planned"]],
            y=[selected["batching"].total_p99_ms],
            mode="markers",
            name="selected",
            marker=dict(color=COLORS["RedLine"], size=14, symbol="diamond"),
        ))
        fig.update_layout(
            height=370,
            xaxis=dict(title="Planned serving units after autoscale buffer"),
            yaxis=dict(title="P99 latency (ms)", gridcolor="#f1f5f9"),
            yaxis2=dict(title=v1_13_packet["cost_unit"], overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.15, x=0),
            margin=dict(l=50, r=70, t=60, b=40),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        cost_ok = selected["cost"]["value"] <= selected["cost"]["limit"]
        p99_ok = selected["batching"].slo_ok
        warm_gap = max(0.0, v1_13_packet["warm_pool_floor"] - selected["cold"].protected_fraction)
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_13_metric_card("Planned Units", str(selected["planned"]), f"{partC_buffer.value}% headroom", COLORS["GreenLine"])}
            {v1_13_metric_card("Utilization", f"{selected['batching'].utilization:.2f}", "after batching", COLORS["OrangeLine"])}
            {v1_13_metric_card("P99", f"{selected['batching'].total_p99_ms:.1f} ms", f"SLO {slo:g} ms", COLORS["GreenLine"] if p99_ok else COLORS["RedLine"], True)}
            {v1_13_metric_card(v1_13_packet["cost_label"], v1_13_fmt_amount(selected["cost"]["value"], selected["cost"]["unit"]), f"limit {v1_13_fmt_amount(selected['cost']['limit'], selected['cost']['unit'])}", COLORS["GreenLine"] if cost_ok else COLORS["RedLine"])}
        </div>
        """))
        table_rows = []
        for row in frontier[: min(8, len(frontier))]:
            tail_value = f"{row['p999']:.1f} ms" if v1_13_track_id == "robotaxi" else f"{row['queue'].p95_latency_ms:.1f} ms p95"
            table_rows.append((
                row["visible"],
                row["planned"],
                f"{row['batching'].utilization:.2f}",
                f"{row['batching'].total_p99_ms:.1f} ms",
                tail_value,
                v1_13_fmt_amount(row["cost"]["value"], row["cost"]["unit"]),
                f"{row['cold'].protected_fraction*100:.0f}% warm",
                row["status"],
            ))
        items.append(mo.Html(v1_13_html_table(
            ("Visible", "Planned", "Util", "P99", "Tail evidence", v1_13_packet["cost_label"], "Warm pool", "Status"),
            table_rows,
        )))
        if p99_ok and cost_ok and warm_gap <= 0:
            items.append(mo.callout(mo.md(
                "**Replica frontier passes.** The selected scale posture has p99, cost/energy, and warm-capacity evidence inside guardrails."
            ), kind="success"))
        else:
            failures = []
            if not p99_ok:
                failures.append(f"p99 {selected['batching'].total_p99_ms:.1f} ms > {slo:g} ms")
            if not cost_ok:
                failures.append(f"{v1_13_packet['cost_label']} above limit")
            if warm_gap > 0:
                failures.append(f"warm-pool coverage short by {warm_gap*100:.0f} percentage points")
            items.append(mo.callout(mo.md("**Scale policy not launch-ready:** " + "; ".join(failures) + "."), kind="danger"))
        items.append(v1_13_prediction_feedback(
            "tradeoff",
            partC_pred.value,
            "**Correct.** Replicas lower utilization only by spending capacity, cost, and warm-pool budget.",
            "**Replica intuition is incomplete.** Scale-out can lower p99, but it can also violate cost/energy or cold-start guardrails.",
        ))
        items.append(mo.accordion({
            "Math Peek / Source Model - replica frontier": mo.md(f"""
```
planned_replicas = ceil(visible_replicas * (1 + autoscale_buffer_pct/100))
capacity_qps     = planned_replicas * 1000 / service_ms
rho              = arrival_qps / capacity_qps
cost_per_request = standing_capacity_cost / (arrival_qps * 3600)
```

Current values: visible `{partC_replicas.value}`, planned `{selected['planned']}`,
rho `{selected['batching'].utilization:.2f}`, p99 `{selected['batching'].total_p99_ms:.1f}` ms,
{v1_13_packet['cost_label']} `{v1_13_fmt_amount(selected['cost']['value'], selected['cost']['unit'])}`.
Source helpers: `queueing_latency`, `batching_tax`, and `cold_start_latency`; cost coefficients are V1-13 scenario assumptions.
            """)
        }))
        items.append(partC_checkpoint)
        if partC_checkpoint.value is None:
            items.append(mo.callout(mo.md("Checkpoint: choose the scale posture that should feed the policy gate."), kind="info"))
        return mo.vstack(items)

    def build_part_d():
        items = [
            v1_13_part_banner(
                "D",
                "A Serving Policy Must Pass SLO, Capacity, And Cost Guardrails",
                COLORS["RedLine"],
                (
                    f"The launch memo for {v1_13_serving.label} must pick one policy, reject one alternative, "
                    f"and protect {v1_13_variant.guardrail_metric}."
                ),
            ),
            mo.md("""
## Concept: Feasibility Is A Conjunction

A policy is not launchable because one metric looks good. It must pass p99/SLA,
live-state capacity, cost or energy, and warm/fallback guardrails under the same
track workload.
            """),
            partD_pred,
        ]
        if partD_pred.value is None:
            items.append(mo.callout(mo.md("Commit to a launch-policy prediction before opening the candidate gate."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_policy, partD_reject, partD_cost_multiplier, partD_capacity_reserve], widths="equal"))
        items.append(partD_ops_risk)
        selected_key = partD_policy.value
        rejected_key = partD_reject.value
        selected = v1_13_evaluate_policy(
            v1_13_serving,
            v1_13_model,
            v1_13_packet,
            v1_13_candidates[selected_key],
            cost_multiplier=partD_cost_multiplier.value,
            capacity_reserve_pct=partD_capacity_reserve.value,
        )
        rejected = v1_13_evaluate_policy(
            v1_13_serving,
            v1_13_model,
            v1_13_packet,
            v1_13_candidates[rejected_key],
            cost_multiplier=partD_cost_multiplier.value,
            capacity_reserve_pct=partD_capacity_reserve.value,
        )
        candidate_rows = []
        for key, candidate in v1_13_candidates.items():
            ev = v1_13_evaluate_policy(
                v1_13_serving,
                v1_13_model,
                v1_13_packet,
                candidate,
                cost_multiplier=partD_cost_multiplier.value,
                capacity_reserve_pct=partD_capacity_reserve.value,
            )
            candidate_rows.append((
                candidate["label"],
                f"batch {candidate['batch_size']}, repl {ev['planned_replicas']}",
                f"{ev['batching'].total_p99_ms:.1f} ms",
                f"{ev['batching'].utilization:.2f}",
                f"{ev['live_with_reserve']} <= {ev['capacity'].max_concurrent}",
                v1_13_fmt_amount(ev["cost"]["value"], ev["cost"]["unit"]),
                "PASS" if ev["feasible"] else ev["binding"],
            ))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[v1_13_candidates[key]["label"] for key in v1_13_candidates],
            y=[v1_13_evaluate_policy(v1_13_serving, v1_13_model, v1_13_packet, cand, cost_multiplier=partD_cost_multiplier.value, capacity_reserve_pct=partD_capacity_reserve.value)["batching"].total_p99_ms for cand in v1_13_candidates.values()],
            marker_color=[COLORS["GreenLine"] if v1_13_evaluate_policy(v1_13_serving, v1_13_model, v1_13_packet, cand, cost_multiplier=partD_cost_multiplier.value, capacity_reserve_pct=partD_capacity_reserve.value)["feasible"] else COLORS["RedLine"] for cand in v1_13_candidates.values()],
            name="candidate p99",
        ))
        fig.add_hline(y=v1_13_serving.slo_ms, line_dash="dash", line_color=COLORS["BlueLine"], annotation_text="p99 SLO")
        fig.update_layout(
            height=330,
            xaxis=dict(title="Policy candidate"),
            yaxis=dict(title="P99 latency (ms)", gridcolor="#f1f5f9"),
            margin=dict(l=50, r=20, t=60, b=80),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        items.append(mo.Html(v1_13_html_table(
            ("Policy", "Config", "P99", "Util", "Live state", v1_13_packet["cost_label"], "Gate"),
            candidate_rows,
        )))
        selected_color = COLORS["GreenLine"] if selected["feasible"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {v1_13_metric_card("Selected Policy", selected['candidate']['label'], selected['candidate']['rationale'], selected_color, True)}
            {v1_13_metric_card("Binding", selected['binding'], "release gate", selected_color)}
            {v1_13_metric_card("Rejected", rejected['candidate']['label'], rejected['binding'], COLORS["OrangeLine"])}
        </div>
        """))
        if selected["feasible"]:
            items.append(mo.callout(mo.md(
                f"**Launch gate passes.** {selected['candidate']['label']} satisfies p99, capacity, {v1_13_packet['cost_label'].lower()}, and warm/fallback guardrails."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Launch gate fails.** Binding constraint: {selected['binding']}. First mitigation: {v1_13_packet['first_mitigation']}."
            ), kind="danger"))
        items.append(v1_13_prediction_feedback(
            "all_guardrails",
            partD_pred.value,
            "**Correct.** A serving policy launches only when all guardrails pass together.",
            "**Single-metric launch decisions are unsafe.** Fastest or cheapest policies can still fail capacity, warm-start, or track guardrails.",
        ))
        risk_text = v1_13_packet["ops_risks"].get(partD_ops_risk.value, "Select a carry-forward operations risk.")
        items.append(mo.accordion({
            "Math Peek / Source Model - constrained policy gate": mo.md(f"""
```
feasible = p99_ok and capacity_ok and cost_ok and warm_start_ok
live_required = ceil(arrival_qps * p99_ms / 1000 * (1 + reserve_pct/100))
```

Selected policy: `{selected['candidate']['label']}`. Binding constraint:
`{selected['binding']}`. Rejected alternative: `{rejected['candidate']['label']}`.
Carry-forward risk: {risk_text}

Source helpers: `batching_tax`, `queueing_latency`, `cache_capacity`, and
`cold_start_latency`; V1-13 notebook-local policy scoring supplies the conjunction
and track-specific cost/energy labels.
            """)
        }))
        return mo.vstack(items)

    def build_synthesis():
        selected_key = partD_policy.value
        rejected_key = partD_reject.value
        selected = v1_13_evaluate_policy(
            v1_13_serving,
            v1_13_model,
            v1_13_packet,
            v1_13_candidates[selected_key],
            cost_multiplier=partD_cost_multiplier.value,
            capacity_reserve_pct=partD_capacity_reserve.value,
        )
        rejected = v1_13_evaluate_policy(
            v1_13_serving,
            v1_13_model,
            v1_13_packet,
            v1_13_candidates[rejected_key],
            cost_multiplier=partD_cost_multiplier.value,
            capacity_reserve_pct=partD_capacity_reserve.value,
        )
        risk_text = v1_13_packet["ops_risks"].get(partD_ops_risk.value, "No carry-forward risk selected yet.")
        memo_rows = (
            ("Selected policy", selected["candidate"]["label"], selected["candidate"]["rationale"]),
            ("Binding constraint", selected["binding"], "all guardrails pass" if selected["feasible"] else "mitigation required"),
            ("Rejected alternative", rejected["candidate"]["label"], rejected["binding"]),
            ("P99/SLO evidence", f"{selected['batching'].total_p99_ms:.1f} ms / {v1_13_serving.slo_ms:g} ms", "PASS" if selected["batching"].slo_ok else "FAIL"),
            ("Utilization", f"{selected['batching'].utilization:.2f}", "after batching and replicas"),
            (v1_13_packet["cost_label"], v1_13_fmt_amount(selected["cost"]["value"], selected["cost"]["unit"]), f"limit {v1_13_fmt_amount(selected['cost_limit'], selected['cost']['unit'])}"),
            ("Capacity/state", f"{selected['live_with_reserve']} live <= {selected['capacity'].max_concurrent} fit", selected["capacity"].state_kind),
            ("Carry-forward ops risk", risk_text, "feeds Lab 14 operations planning"),
        )
        return mo.vstack([
            mo.md("## Serving Launch Memo"),
            mo.callout(mo.md(
                "**Chapter invariant:** serving is a latency distribution plus a capacity system. "
                "Batching, queueing, replicas, live state, warm capacity, and cost must be judged together."
            ), kind="info"),
            mo.Html(v1_13_html_table(("Memo field", "Decision/evidence", "Report framing"), memo_rows)),
            mo.callout(mo.md(
                f"**Final report frame for {v1_13_serving.label}:** choose `{selected['candidate']['label']}`, "
                f"name `{selected['binding']}` as the binding constraint, reject `{rejected['candidate']['label']}`, "
                f"and carry forward: {risk_text}"
            ), kind="success" if selected["feasible"] else "warn"),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Batching Trade-off": build_part_a(),
        "Part B: Queueing Tail": build_part_b(),
        "Part C: Replica Frontier": build_part_c(),
        "Part D: Policy Gate": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


# ===========================================================================
# ZONE D: LEDGER HUD
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    batching_tax,
    ledger,
    mo,
    partA_arr,
    partA_batch,
    partA_checkpoint,
    partA_pred,
    partA_slo,
    partB_arr,
    partB_checkpoint,
    partB_pred,
    partB_slo,
    partB_svc,
    partC_buffer,
    partC_checkpoint,
    partC_pred,
    partC_replicas,
    partC_warm_pool,
    partD_capacity_reserve,
    partD_cost_multiplier,
    partD_ops_risk,
    partD_policy,
    partD_pred,
    partD_reject,
    queueing_latency,
    v1_13_evaluate_policy,
    v1_13_policy_candidates,
    v1_13_track_id,
    v1_13_track_packet,
    v1_13_model,
    v1_13_profile,
    v1_13_serving,
    v1_13_variant,
):
    _packet = v1_13_track_packet(v1_13_track_id, v1_13_serving, v1_13_variant)
    _candidates = v1_13_policy_candidates(v1_13_serving, v1_13_track_id)
    _batching = batching_tax(
        batch_size=partA_batch.value,
        arrival_qps=partA_arr.value,
        service_ms=v1_13_serving.service_ms,
        slo_ms=partA_slo.value,
        efficiency_gain=v1_13_serving.batch_efficiency_gain,
        replicas=v1_13_serving.replicas,
        service_cv=v1_13_serving.service_cv,
    )
    _queue = queueing_latency(
        arrival_qps=partB_arr.value,
        service_ms=partB_svc.value,
        replicas=v1_13_serving.replicas,
        service_cv=v1_13_serving.service_cv,
        slo_ms=partB_slo.value,
    )
    _selected = v1_13_evaluate_policy(
        v1_13_serving,
        v1_13_model,
        _packet,
        _candidates[partD_policy.value],
        cost_multiplier=partD_cost_multiplier.value,
        capacity_reserve_pct=partD_capacity_reserve.value,
    )
    _rejected = v1_13_evaluate_policy(
        v1_13_serving,
        v1_13_model,
        _packet,
        _candidates[partD_reject.value],
        cost_multiplier=partD_cost_multiplier.value,
        capacity_reserve_pct=partD_capacity_reserve.value,
    )

    if partA_pred.value is not None and partB_pred.value is not None and partC_pred.value is not None and partD_pred.value is not None:
        ledger.save(chapter=13, design={
            "chapter": "v1_13",
            "track_id": v1_13_profile.track_id,
            "scenario_id": v1_13_variant.scenario_id,
            "hardware_ref": v1_13_serving.hardware_ref,
            "model_ref": v1_13_serving.model_ref,
            "completed": True,
            "part_a_batch_prediction": partA_pred.value,
            "part_a_batch_decision": partA_checkpoint.value,
            "batch_size": partA_batch.value,
            "batch_total_p99_ms": round(_batching.total_p99_ms, 3),
            "batch_throughput_gain": round(_batching.throughput_gain, 3),
            "part_b_queue_prediction": partB_pred.value,
            "part_b_queue_decision": partB_checkpoint.value,
            "arrival_qps": round(partB_arr.value, 3),
            "utilization": round(_queue.utilization, 4),
            "queue_p99_ms": round(_queue.p99_latency_ms, 3),
            "queue_slo_ok": _queue.slo_ok,
            "part_c_replica_prediction": partC_pred.value,
            "part_c_replica_decision": partC_checkpoint.value,
            "visible_replicas": partC_replicas.value,
            "autoscale_buffer_pct": partC_buffer.value,
            "warm_pool_units": partC_warm_pool.value,
            "part_d_policy_prediction": partD_pred.value,
            "selected_policy": _selected["candidate"]["label"],
            "selected_policy_feasible": _selected["feasible"],
            "binding_constraint": _selected["binding"],
            "rejected_alternative": _rejected["candidate"]["label"],
            "rejected_alternative_binding": _rejected["binding"],
            "carry_forward_ops_risk": _packet["ops_risks"].get(partD_ops_risk.value, partD_ops_risk.value),
            "cost_label": _packet["cost_label"],
            "cost_per_request": round(_selected["cost"]["value"], 8),
            "cost_unit": _selected["cost"]["unit"],
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">13 &middot; Model Serving</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_13_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">SLO</span>
        <span class="hud-value">{v1_13_serving.slo_ms:g} ms p99</span>
        <span class="hud-label">POLICY</span>
        <span class="hud-active">{_selected['candidate']['label']}</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    batching_tax,
    build_lab_report,
    mo,
    partA_arr,
    partA_batch,
    partA_checkpoint,
    partA_pred,
    partA_slo,
    partB_arr,
    partB_checkpoint,
    partB_pred,
    partB_slo,
    partB_svc,
    partC_buffer,
    partC_checkpoint,
    partC_pred,
    partC_replicas,
    partC_warm_pool,
    partD_capacity_reserve,
    partD_cost_multiplier,
    partD_ops_risk,
    partD_policy,
    partD_pred,
    partD_reject,
    queueing_latency,
    report_export_panel,
    v1_13_evaluate_policy,
    v1_13_fmt_amount,
    v1_13_metadata,
    v1_13_model,
    v1_13_policy_candidates,
    v1_13_profile,
    v1_13_serving,
    v1_13_track_id,
    v1_13_track_packet,
    v1_13_variant,
):
    _packet = v1_13_track_packet(v1_13_track_id, v1_13_serving, v1_13_variant)
    _candidates = v1_13_policy_candidates(v1_13_serving, v1_13_track_id)
    _batching = batching_tax(
        batch_size=partA_batch.value,
        arrival_qps=partA_arr.value,
        service_ms=v1_13_serving.service_ms,
        slo_ms=partA_slo.value,
        efficiency_gain=v1_13_serving.batch_efficiency_gain,
        replicas=v1_13_serving.replicas,
        service_cv=v1_13_serving.service_cv,
    )
    _queue = queueing_latency(
        arrival_qps=partB_arr.value,
        service_ms=partB_svc.value,
        replicas=v1_13_serving.replicas,
        service_cv=v1_13_serving.service_cv,
        slo_ms=partB_slo.value,
    )
    _selected = v1_13_evaluate_policy(
        v1_13_serving,
        v1_13_model,
        _packet,
        _candidates[partD_policy.value],
        cost_multiplier=partD_cost_multiplier.value,
        capacity_reserve_pct=partD_capacity_reserve.value,
    )
    _rejected = v1_13_evaluate_policy(
        v1_13_serving,
        v1_13_model,
        _packet,
        _candidates[partD_reject.value],
        cost_multiplier=partD_cost_multiplier.value,
        capacity_reserve_pct=partD_capacity_reserve.value,
    )
    _risk_text = _packet["ops_risks"].get(partD_ops_risk.value, "No carry-forward operations risk selected yet.")

    _incomplete = []
    if partA_pred.value is None:
        _incomplete.append("Part A batching prediction")
    if partA_checkpoint.value is None:
        _incomplete.append("Part A batching checkpoint")
    if partB_pred.value is None:
        _incomplete.append("Part B queueing prediction")
    if partB_checkpoint.value is None:
        _incomplete.append("Part B queueing checkpoint")
    if partC_pred.value is None:
        _incomplete.append("Part C replica prediction")
    if partC_checkpoint.value is None:
        _incomplete.append("Part C replica checkpoint")
    if partD_pred.value is None:
        _incomplete.append("Part D policy prediction")
    if partD_ops_risk.value is None:
        _incomplete.append("Synthesis carry-forward risk")

    _report = build_lab_report(
        v1_13_metadata,
        track=v1_13_profile.label,
        scenario=v1_13_variant.workload_summary,
        learning_objectives=(
            "Explain why batching changes throughput and latency at the same time.",
            "Use arrival rate and utilization to diagnose queueing and p99 failures.",
            "Choose replicas/autoscaling with p99, capacity, and cost evidence.",
            "Write a serving launch memo with selected policy, binding constraint, rejected alternative, and ops risk.",
        ),
        predictions={
            "batching_tradeoff": partA_pred.value,
            "queueing_tail_failure": partB_pred.value,
            "replica_autoscale_tradeoff": partC_pred.value,
            "launch_policy_validity": partD_pred.value,
        },
        knob_settings={
            "part_a_batch_size": partA_batch.value,
            "part_a_arrival_qps": partA_arr.value,
            "part_a_slo_ms": partA_slo.value,
            "part_a_checkpoint": partA_checkpoint.value,
            "part_b_arrival_qps": partB_arr.value,
            "part_b_service_ms": partB_svc.value,
            "part_b_slo_ms": partB_slo.value,
            "part_b_checkpoint": partB_checkpoint.value,
            "part_c_visible_replicas": partC_replicas.value,
            "part_c_autoscale_buffer_pct": partC_buffer.value,
            "part_c_warm_pool_units": partC_warm_pool.value,
            "part_c_checkpoint": partC_checkpoint.value,
            "part_d_selected_policy": _selected["candidate"]["label"],
            "part_d_rejected_alternative": _rejected["candidate"]["label"],
            "part_d_cost_multiplier": partD_cost_multiplier.value,
            "part_d_capacity_reserve_pct": partD_capacity_reserve.value,
            "part_d_ops_risk": _risk_text,
        },
        evidence_summary={
            "hardware_ref": v1_13_serving.hardware_ref,
            "model_ref": v1_13_serving.model_ref,
            "batch_total_p99_ms": round(_batching.total_p99_ms, 3),
            "batch_throughput_gain": round(_batching.throughput_gain, 3),
            "batch_slo_ok": _batching.slo_ok,
            "queue_utilization": round(_queue.utilization, 4),
            "queue_p99_ms": round(_queue.p99_latency_ms, 3),
            "queue_p999_ms": round(_queue.p999_latency_ms, 3),
            "queue_slo_ok": _queue.slo_ok,
            "selected_policy_p99_ms": round(_selected["batching"].total_p99_ms, 3),
            "selected_policy_utilization": round(_selected["batching"].utilization, 4),
            "selected_policy_capacity_fit": f"{_selected['live_with_reserve']} <= {_selected['capacity'].max_concurrent}",
            "selected_policy_cost": v1_13_fmt_amount(_selected["cost"]["value"], _selected["cost"]["unit"]),
            "selected_policy_feasible": _selected["feasible"],
            "binding_constraint": _selected["binding"],
            "rejected_alternative": _rejected["candidate"]["label"],
            "rejected_alternative_binding": _rejected["binding"],
            "carry_forward_ops_risk": _risk_text,
        },
        final_decision=(
            f"Use `{_selected['candidate']['label']}` for {v1_13_serving.label} only if the launch gate remains "
            f"consistent with binding constraint `{_selected['binding']}`. Reject `{_rejected['candidate']['label']}` because `{_rejected['binding']}`."
        ),
        big_takeaways=(
            "Batching follows traffic and spends latency while buying throughput.",
            "Arrival rate and utilization make p99/p999 the serving evidence, not the mean.",
            "Replicas and autoscaling buy headroom by spending cost, energy, or warm spare capacity.",
            "A launch policy is feasible only when SLO, capacity, cost, and track guardrails pass together.",
        ),
        reflections={
            "track_policy": v1_13_serving.serving_policy,
            "selected_policy": _selected["candidate"]["label"],
            "binding_constraint": _selected["binding"],
            "rejected_alternative": _rejected["candidate"]["label"],
            "validation_tests": v1_13_serving.validation_tests,
            "carry_forward_ops_risk": _risk_text,
        },
        residual_risk=(
            "The helper models are teaching estimates. Production deployment still needs real arrival traces, "
            "service-time histograms, p99/p999 replay, hardware counters, capacity telemetry, cost validation, and warm-pool drills."
        ),
        source_trace={
            "track_id": v1_13_profile.track_id,
            "scenario_id": v1_13_variant.scenario_id,
            "hardware_ref": v1_13_variant.hardware_ref,
            "model_ref": v1_13_variant.model_ref,
            "shared_helpers": ("batching_tax", "queueing_latency", "cache_capacity", "cold_start_latency"),
            "notebook_local_helpers": ("v1_13_track_packet", "v1_13_policy_candidates", "v1_13_evaluate_policy"),
            "source_policy": v1_13_profile.source_policy,
        },
        result_snapshot={
            "serving_profile": v1_13_serving,
            "batching": _batching,
            "queueing": _queue,
            "selected_policy": _selected,
            "rejected_policy": _rejected,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-13 report is generated locally from the selected track, MLSysIM hardware/model refs, "
                "shared serving helpers, and notebook-local V1-13 policy scoring."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
