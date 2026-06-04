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
                Queuing &middot; Batching &middot; State/Cache &middot; Cold Start
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify tail amplification:</strong>
                    utilization and service-time variance turn a normal service time into a p99 SLO risk.</div>
                <div style="margin-bottom: 3px;">2. <strong>Separate throughput wins from latency taxes:</strong>
                    static batching can improve throughput while spending the SLO before inference starts.</div>
                <div style="margin-bottom: 3px;">3. <strong>Size state and scale-out policy:</strong>
                    {v1_13_serving.state_kind}, model weights, and cold starts determine safe concurrency.</div>
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
            "A) Mean latency is enough": "mean",
            "B) P99 is roughly 2x service time": "2x",
            "C) P99 explodes near saturation": "tail",
            "D) Replicas remove queueing entirely": "replicas",
        },
        label=(
            f"{v1_13_serving.label}: {v1_13_serving.arrival_qps:g} QPS, "
            f"{v1_13_serving.service_ms:g} ms service, "
            f"{v1_13_serving.replicas} replica(s). What happens to p99?"
        ),
    )
    return (partA_pred,)


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    _capacity_qps = 1000 / v1_13_serving.service_ms * v1_13_serving.replicas
    _raw_util = min(0.95, max(0.05, v1_13_serving.arrival_qps / _capacity_qps))
    _default_util = round(round(_raw_util / 0.05) * 0.05, 2)
    partA_rho = mo.ui.slider(
        start=0.05,
        stop=0.95,
        value=_default_util,
        step=0.05,
        label="Server utilization (rho)",
    )
    partA_svc = mo.ui.slider(
        start=1.0,
        stop=max(200.0, v1_13_serving.service_ms * 3),
        value=v1_13_serving.service_ms,
        step=1.0,
        label="Service time (ms)",
    )
    partA_slo = mo.ui.slider(
        start=10.0,
        stop=max(500.0, v1_13_serving.slo_ms * 3),
        value=v1_13_serving.slo_ms,
        step=10.0,
        label="SLO budget (ms)",
    )

    partB_pred = mo.ui.radio(
        options={
            "A) Batching always improves p99": "always",
            "B) Formation delay can consume the SLO": "delay",
            "C) Batching only changes memory": "memory",
            "D) Batching removes cold starts": "cold",
        },
        label=(
            f"Batch {v1_13_serving.batch_size} at "
            f"{v1_13_serving.arrival_qps:g} QPS. What is the first latency tax?"
        ),
    )
    return (partA_rho, partA_slo, partA_svc, partB_pred)


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    partB_batch = mo.ui.slider(
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
    partB_arr = mo.ui.slider(
        start=_arr_start,
        stop=_arr_stop,
        value=_arr_value,
        step=_arr_step,
        label="Arrival rate (QPS)",
    )
    partB_slo = mo.ui.slider(
        start=10,
        stop=max(500, int(v1_13_serving.slo_ms * 3)),
        value=int(v1_13_serving.slo_ms),
        step=5,
        label="SLO budget (ms)",
    )

    partC_pred = mo.ui.radio(
        options={
            "A) Peak compute sets concurrency": "compute",
            "B) Weights plus live state/cache set concurrency": "memory",
            "C) Replicas make memory irrelevant": "replicas",
            "D) Context/window length does not matter": "context",
        },
        label=(
            f"{v1_13_serving.model_name} on {v1_13_serving.hardware_name}: "
            "what limits live requests?"
        ),
    )
    return (partB_arr, partB_batch, partB_slo, partC_pred)


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    partC_model = mo.ui.dropdown(
        options={v1_13_serving.model_name: v1_13_serving.model_ref},
        value=v1_13_serving.model_name,
        label="Model ref",
    )
    partC_prec = mo.ui.dropdown(
        options={"FP16 (2B)": 2.0, "INT8 (1B)": 1.0, "INT4 (0.5B)": 0.5},
        value={2.0: "FP16 (2B)", 1.0: "INT8 (1B)", 0.5: "INT4 (0.5B)"}.get(
            v1_13_serving.precision_bytes,
            "FP16 (2B)",
        ),
        label="Weight precision (bytes)",
    )
    partC_ctx = mo.ui.slider(
        start=256,
        stop=max(131072, v1_13_serving.context_tokens),
        value=v1_13_serving.context_tokens,
        step=256 if v1_13_serving.context_tokens <= 4096 else 2048,
        label="Context/window length (tokens)",
    )
    partC_gpus = mo.ui.dropdown(
        options={"1 device": 1, "2 devices": 2, "4 devices": 4, "8 devices": 8},
        value={1: "1 device", 2: "2 devices", 4: "4 devices", 8: "8 devices"}.get(
            v1_13_serving.default_devices_per_replica,
            "1 device",
        ),
        label="Devices per serving unit",
    )

    partD_pred = mo.ui.radio(
        options={
            "A) Cold start is close to normal service time": "service",
            "B) Data movement dominates first-request latency": "movement",
            "C) Warm pools only affect throughput": "throughput",
            "D) Cold starts are unrelated to SLO": "unrelated",
        },
        label=(
            f"{v1_13_serving.label}: scale out {v1_13_serving.model_name}. "
            "What dominates the first uncached request?"
        ),
    )
    return (partC_ctx, partC_gpus, partC_model, partC_prec, partD_pred)


@app.cell(hide_code=True)
def _(mo, v1_13_serving):
    partD_scaleout = mo.ui.slider(
        start=1,
        stop=16,
        value=v1_13_serving.scale_out_replicas,
        step=1,
        label="Scale-out replicas",
    )
    partD_stor = mo.ui.dropdown(
        options={
            "Registry/default storage": "default",
            "Slow network storage": "nfs",
            "Cached in host RAM": "ram",
        },
        value="Registry/default storage",
        label="Storage type",
    )
    partD_warm_pool = mo.ui.slider(
        start=0,
        stop=8,
        value=v1_13_serving.warm_pool_replicas,
        step=1,
        label="Warm pool replicas",
    )
    return (partD_scaleout, partD_stor, partD_warm_pool)


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
    partA_pred,
    partA_rho,
    partA_slo,
    partA_svc,
    partB_arr,
    partB_batch,
    partB_pred,
    partB_slo,
    partC_ctx,
    partC_gpus,
    partC_model,
    partC_prec,
    partC_pred,
    partD_pred,
    partD_scaleout,
    partD_stor,
    partD_warm_pool,
    queueing_latency,
    v1_13_model,
    v1_13_profile,
    v1_13_serving,
    v1_13_variant,
):
    def _metric_card(label, value, detail, color, border=False):
        border_style = f"2px solid {color}" if border else "1px solid #e2e8f0"
        return f"""
        <div style="padding:16px; border:{border_style}; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {color}; flex:1;">
            <div style="color:#64748b; font-size:0.78rem; font-weight:700;">{label}</div>
            <div style="font-size:1.55rem; font-weight:800; color:{color};">{value}</div>
            <div style="font-size:0.72rem; color:#64748b;">{detail}</div>
        </div>
        """

    def build_part_a():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Incoming Message &middot; {v1_13_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "The average request looks fine on {v1_13_serving.label}. Can we trust
                    that number, or will p99 violate {v1_13_serving.slo_ms:g} ms?"
                </div>
            </div>
            """),
            mo.md("""
## Queueing Turns Utilization Into Tail Latency

The serving wall is not only service time. Once utilization rises, requests wait
behind other requests. This lab uses the shared `mlsysbook_labs.queueing_latency`
helper, an M/M/c queue with a service-variability adjustment.

```
rho = arrival_rate / (replicas * service_rate)
p99 = service_time + queue_tail(rho, replicas, service_cv)
```
            """),
            partA_pred,
        ]
        if partA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the p99 instruments."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_rho, partA_svc, partA_slo], widths="equal"))

        _svc = partA_svc.value
        _rho = partA_rho.value
        _slo = partA_slo.value
        _replicas = v1_13_serving.replicas
        _arrival = _rho * _replicas * (1000 / _svc)
        _queue = queueing_latency(
            arrival_qps=_arrival,
            service_ms=_svc,
            replicas=_replicas,
            service_cv=v1_13_serving.service_cv,
            slo_ms=_slo,
        )

        _rhos = np.linspace(0.05, 0.95, 50)
        _means = []
        _p99s = []
        for _r in _rhos:
            _q = queueing_latency(
                arrival_qps=_r * _replicas * (1000 / _svc),
                service_ms=_svc,
                replicas=_replicas,
                service_cv=v1_13_serving.service_cv,
                slo_ms=_slo,
            )
            _means.append(_q.mean_latency_ms if math.isfinite(_q.mean_latency_ms) else None)
            _p99s.append(_q.p99_latency_ms if math.isfinite(_q.p99_latency_ms) else None)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_rhos,
            y=_means,
            mode="lines",
            name="Mean latency",
            line=dict(color=COLORS["BlueLine"], width=2),
        ))
        _fig.add_trace(go.Scatter(
            x=_rhos,
            y=_p99s,
            mode="lines",
            name="P99 latency",
            line=dict(color=COLORS["RedLine"], width=3),
        ))
        _fig.add_hline(y=_slo, line_dash="dash", line_color=COLORS["GreenLine"], annotation_text=f"SLO = {_slo:.0f} ms")
        _fig.add_trace(go.Scatter(
            x=[_rho],
            y=[_queue.p99_latency_ms],
            mode="markers",
            name="selected p99",
            marker=dict(color=COLORS["RedLine"], size=14, symbol="diamond"),
        ))
        _fig.update_layout(
            height=380,
            xaxis=dict(title="Utilization (rho)", range=[0, 1]),
            yaxis=dict(title="Latency (ms)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _p99_color = COLORS["RedLine"] if not _queue.slo_ok else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Utilization", f"{_queue.utilization:.2f}", f"{_arrival:.2f} QPS arrival", COLORS["OrangeLine"])}
            {_metric_card("Mean Latency", f"{_queue.mean_latency_ms:.1f} ms", "wait + service", COLORS["BlueLine"])}
            {_metric_card("P99 Latency", f"{_queue.p99_latency_ms:.1f} ms", f"{_queue.queue_amplifier:.1f}x service", _p99_color, True)}
            {_metric_card("Wait Probability", f"{_queue.queue_wait_probability*100:.0f}%", "Erlang-C", COLORS["PurpleLine"] if "PurpleLine" in COLORS else COLORS["BlueLine"])}
        </div>
        """))

        if not _queue.slo_ok:
            items.append(mo.callout(mo.md(
                f"**SLO VIOLATED.** P99 = {_queue.p99_latency_ms:.1f} ms > {_slo:.0f} ms. "
                f"The mean ({_queue.mean_latency_ms:.1f} ms) hides the one-in-100 experience."
            ), kind="danger"))

        items.append(mo.md(f"""
**Queueing - Live Calculation**

```
replicas     = {_replicas}
service time = {_svc:.1f} ms
arrival      = {_arrival:.2f} QPS
rho          = {_queue.utilization:.2f}
mean         = {_queue.mean_latency_ms:.1f} ms
p95 / p99    = {_queue.p95_latency_ms:.1f} / {_queue.p99_latency_ms:.1f} ms
```
*Source: `mlsysbook_labs.queueing_latency`, track `{v1_13_profile.track_id}`.*
        """))

        if partA_pred.value == "tail":
            items.append(mo.callout(mo.md("**Correct.** Tail latency is the binding metric once utilization rises."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**The trap is trusting the average.** Replicas help only if they reduce utilization enough; "
                "p99 still needs direct measurement and SLO headroom."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Escalation &middot; Serving Performance Engineer
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Throughput improves when we batch. Does that automatically improve
                    {v1_13_serving.label} p99?"
                </div>
            </div>
            """),
            mo.md("""
## Batching Has a Formation Delay Tax

Static batching waits for requests to arrive before work starts. That waiting
time is charged to the user's end-to-end latency.

```
formation_delay = (batch_size - 1) / (2 * arrival_rate)
total_p99       = formation_delay + batched_service + queue_tail
```
            """),
            partB_pred,
        ]
        if partB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the batching instruments."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_batch, partB_arr, partB_slo], widths="equal"))
        _batch = partB_batch.value
        _arrival = partB_arr.value
        _slo = partB_slo.value
        _batching = batching_tax(
            batch_size=_batch,
            arrival_qps=_arrival,
            service_ms=v1_13_serving.service_ms,
            slo_ms=_slo,
            efficiency_gain=v1_13_serving.batch_efficiency_gain,
            replicas=v1_13_serving.replicas,
            service_cv=v1_13_serving.service_cv,
        )

        _fig = go.Figure()
        for _name, _value, _color in [
            ("Formation delay", _batching.formation_delay_ms, COLORS["OrangeLine"]),
            ("Batched service", _batching.batched_service_ms, COLORS["BlueLine"]),
            ("Queue p99", _batching.queue_p99_ms, COLORS["RedLine"]),
        ]:
            _fig.add_trace(go.Bar(
                name=_name,
                x=[_name],
                y=[_value],
                marker_color=_color,
                hovertemplate="%{x}: %{y:.1f} ms<extra></extra>",
            ))
        _fig.add_hline(y=_slo, line_dash="dash", line_color=COLORS["GreenLine"], annotation_text=f"SLO = {_slo} ms")
        _fig.update_layout(
            height=340,
            yaxis=dict(title="Latency contribution (ms)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _total_color = COLORS["GreenLine"] if _batching.slo_ok else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Formation Delay", f"{_batching.formation_delay_ms:.1f} ms", f"{_batching.formation_slo_pct:.0f}% of SLO", COLORS["OrangeLine"])}
            {_metric_card("Throughput Gain", f"{_batching.throughput_gain:.1f}x", "service amortization", COLORS["BlueLine"])}
            {_metric_card("Batch Utilization", f"{_batching.utilization:.2f}", "batch queue rho", COLORS["OrangeLine"])}
            {_metric_card("Total P99", f"{_batching.total_p99_ms:.1f} ms", "formation + service + queue", _total_color, True)}
        </div>
        """))

        if not _batching.slo_ok:
            items.append(mo.callout(mo.md(
                f"**SLO VIOLATED.** Total p99 = {_batching.total_p99_ms:.1f} ms > {_slo:g} ms. "
                f"Formation delay alone consumes {_batching.formation_slo_pct:.0f}% of the SLO."
            ), kind="danger"))

        items.append(mo.md(f"""
**Batching Tax - Live Calculation**

```
batch size       = {_batch}
arrival          = {_arrival:g} QPS
formation delay  = {_batching.formation_delay_ms:.1f} ms
batched service  = {_batching.batched_service_ms:.1f} ms
queue p99        = {_batching.queue_p99_ms:.1f} ms
total p99        = {_batching.total_p99_ms:.1f} ms
```
*Source: `mlsysbook_labs.batching_tax`.*
        """))

        if partB_pred.value == "delay":
            items.append(mo.callout(mo.md("**Correct.** Throughput batching has a latency bill: waiting for the batch to form."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Batching is not free.** It can be the right serving policy, but only if formation delay and p99 stay inside the SLO."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Capacity Review &middot; {v1_13_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Compute says the accelerator is fast enough. Before we approve the
                    rollout, how many live requests actually fit in memory?"
                </div>
            </div>
            """),
            mo.md(f"""
## Live State/Cache Capacity Sets Concurrency

For this track, the live state is **{v1_13_serving.state_kind}**. A serving unit
must fit model weights and one state/cache allocation per live request.

```
weights + live_requests * state_per_request <= device_memory
```
            """),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the state/cache calculator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_model, partC_prec, partC_ctx, partC_gpus], widths="equal"))
        _precision = partC_prec.value
        _context = partC_ctx.value
        _devices = partC_gpus.value
        _capacity = cache_capacity(
            v1_13_serving,
            v1_13_model,
            context_tokens=_context,
            precision_bytes=_precision,
            devices_per_replica=_devices,
            kv_precision_bytes=v1_13_serving.kv_precision_bytes,
        )

        _max_show = max(3, min(_capacity.max_concurrent + 3, 20))
        _batches = list(range(1, _max_show + 1))
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            name="Weights",
            x=[str(_b) for _b in _batches],
            y=[_capacity.weight_gb] * len(_batches),
            marker_color=COLORS["BlueLine"],
        ))
        _fig.add_trace(go.Bar(
            name=_capacity.state_kind,
            x=[str(_b) for _b in _batches],
            y=[_capacity.state_per_request_gb * _b for _b in _batches],
            marker_color=COLORS["OrangeLine"],
        ))
        _fig.add_hline(
            y=_capacity.total_memory_gb,
            line_dash="dash",
            line_color=COLORS["RedLine"],
            annotation_text=f"memory = {_capacity.total_memory_gb:.2f} GB",
        )
        _fig.update_layout(
            barmode="stack",
            height=380,
            xaxis=dict(title="Live requests"),
            yaxis=dict(title="Memory (GB)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _max_color = COLORS["RedLine"] if _capacity.oom else (COLORS["OrangeLine"] if _capacity.max_concurrent <= 2 else COLORS["GreenLine"])
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Weights", f"{_capacity.weight_gb:.3g} GB", f"{_precision:g} B/param", COLORS["BlueLine"])}
            {_metric_card("State / Request", f"{_capacity.state_per_request_gb:.3g} GB", _capacity.state_kind, COLORS["OrangeLine"])}
            {_metric_card("Available Memory", f"{_capacity.available_gb:.3g} GB", f"{_devices} device(s)", COLORS["BlueLine"])}
            {_metric_card("Max Live Requests", f"{_capacity.max_concurrent}", "memory-bound", _max_color, True)}
        </div>
        """))

        if _capacity.oom:
            items.append(mo.callout(mo.md(
                f"**OOM.** Weights ({_capacity.weight_gb:.3g} GB) plus one "
                f"{_capacity.state_kind} allocation ({_capacity.state_per_request_gb:.3g} GB) "
                f"exceed {_capacity.total_memory_gb:.3g} GB."
            ), kind="danger"))
        elif _capacity.max_concurrent <= 2:
            items.append(mo.callout(mo.md(
                f"**Severe concurrency limit.** Only {_capacity.max_concurrent} live request(s) fit. "
                "Throughput planning must account for memory, not just compute."
            ), kind="warn"))

        items.append(mo.md(f"""
**State/Cache Capacity - Live Calculation**

```
hardware        = {v1_13_serving.hardware_ref}
model           = {v1_13_serving.model_ref}
context/window  = {_context:,}
total memory    = {_capacity.total_memory_gb:.3g} GB
weights         = {_capacity.weight_gb:.3g} GB
state/request   = {_capacity.state_per_request_gb:.3g} GB
max live reqs   = {_capacity.max_concurrent}
```
*Source: `mlsysbook_labs.cache_capacity`, with MLSysIM hardware and model refs.*
        """))

        if partC_pred.value == "memory":
            items.append(mo.callout(mo.md("**Correct.** Serving concurrency is a memory/state problem before it is a peak-compute problem."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Compute is incomplete evidence.** A serving unit must reserve model weights and live request state/cache."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_d():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Scale-Out Drill &middot; Operations Lead
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Traffic spikes. New serving units need the model, runtime, and warmup.
                    Which users see the cold path?"
                </div>
            </div>
            """),
            mo.md("""
## Cold Start Is Data Movement Plus Runtime Initialization

```
cold_start = weights / min(storage_bw, interconnect_bw)
           + weights / deserialize_bw
           + runtime_init + warmup
```

Warm pools do not make cold starts disappear. They reduce the fraction of
scale-out replicas whose first request is exposed to the cold path.
            """),
            partD_pred,
        ]
        if partD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the cold-start calculator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_scaleout, partD_stor, partD_warm_pool], widths="equal"))
        _storage_override = {"default": None, "nfs": 1.25, "ram": 50.0}[partD_stor.value]
        _storage_label = {
            "default": "registry/default storage",
            "nfs": "slow network storage",
            "ram": "cached in host RAM",
        }[partD_stor.value]
        _cold = cold_start_latency(
            v1_13_serving,
            precision_bytes=v1_13_serving.precision_bytes,
            storage_bandwidth_gbs=_storage_override,
            warm_pool_replicas=partD_warm_pool.value,
            scale_out_replicas=partD_scaleout.value,
        )

        _fig = go.Figure()
        _base = 0.0
        for _name, _duration_ms, _color in [
            ("Read weights", _cold.read_ms, COLORS["BlueLine"]),
            ("Deserialize", _cold.deserialize_ms, COLORS["OrangeLine"]),
            ("Runtime init", _cold.runtime_init_ms, "#64748b"),
            ("Warmup", _cold.warmup_ms, COLORS["GreenLine"]),
        ]:
            _fig.add_trace(go.Bar(
                name=_name,
                x=[_duration_ms / 1000],
                y=["Cold start"],
                orientation="h",
                marker_color=_color,
                base=_base / 1000,
                hovertemplate="%{fullData.name}: %{x:.2f} s<extra></extra>",
            ))
            _base += _duration_ms
        _fig.add_vline(x=v1_13_serving.slo_ms / 1000, line_dash="dash", line_color=COLORS["RedLine"], annotation_text="p99 SLO")
        _fig.update_layout(
            height=260,
            barmode="stack",
            xaxis=dict(title="Seconds", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.25, x=0),
            margin=dict(l=90, r=20, t=50, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _cold_color = COLORS["RedLine"] if _cold.exceeds_slo else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Model Weights", f"{_cold.model_weight_gb:.3g} GB", f"{v1_13_serving.precision_bytes:g} B/param", COLORS["BlueLine"])}
            {_metric_card("Effective BW", f"{_cold.effective_bandwidth_gbs:.2f} GB/s", _storage_label, COLORS["OrangeLine"])}
            {_metric_card("Protected", f"{_cold.protected_fraction*100:.0f}%", "warm-pool coverage", COLORS["GreenLine"])}
            {_metric_card("Exposed First Request", f"{_cold.exposed_first_request_ms/1000:.2f} s", "service + cold path share", _cold_color, True)}
        </div>
        """))

        if _cold.exceeds_slo:
            items.append(mo.callout(mo.md(
                f"**Cold-start SLO risk.** Exposed first-request latency is "
                f"{_cold.exposed_first_request_ms/1000:.2f} s, above the "
                f"{v1_13_serving.slo_ms:g} ms p99 SLO."
            ), kind="danger"))

        items.append(mo.md(f"""
**Cold Start - Live Calculation**

```
weights       = {_cold.model_weight_gb:.3g} GB
storage path  = {_storage_label}
effective BW  = min({_cold.storage_bandwidth_gbs:.2f}, {_cold.interconnect_bandwidth_gbs:.2f}) = {_cold.effective_bandwidth_gbs:.2f} GB/s
read          = {_cold.read_ms/1000:.2f} s
deserialize   = {_cold.deserialize_ms/1000:.2f} s
init + warmup = {(_cold.runtime_init_ms + _cold.warmup_ms)/1000:.2f} s
cold path     = {_cold.cold_start_ms/1000:.2f} s
```
*Source: `mlsysbook_labs.cold_start_latency`.*
        """))

        if partD_pred.value == "movement":
            items.append(mo.callout(mo.md("**Correct.** Large cold starts are dominated by moving and preparing model state."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Cold start is a serving SLO problem.** Runtime and data movement can exceed normal inference latency by orders of magnitude."
            ), kind="warn"))
        return mo.vstack(items)

    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                f"**1. Tail latency is the serving metric for {v1_13_serving.label}.** "
                "Mean latency is not enough evidence for an interactive, safety, wearable, or SLA-bound system."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. Batching is a policy, not a free win.** Formation delay, queueing, and SLO budget decide whether it helps."
            ), kind="info"),
            mo.callout(mo.md(
                f"**3. Live state and cold starts are source-of-truth calculations.** "
                f"This lab uses the selected track inputs to compute {v1_13_serving.state_kind}, memory, and scale-out evidence."
            ), kind="info"),
            mo.Html(f"""
            <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        What's Next
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        <strong>Lab 14: ML Operations</strong> - your model shipped. Now
                        the challenge is detecting drift, regressions, and operational failure
                        before dashboards create false confidence.
                    </div>
                </div>
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        Report Focus
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        Your report should name the selected track, p99 SLO, batching policy,
                        memory-bound concurrency, warm-pool coverage, and residual risk.
                    </div>
                </div>
            </div>
            """),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Queueing Explosion": build_part_a(),
        "Part B: Batching Tax": build_part_b(),
        "Part C: State/Cache Wall": build_part_c(),
        "Part D: Cold Start": build_part_d(),
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
    ledger,
    mo,
    partA_pred,
    partB_pred,
    partC_pred,
    partD_pred,
    v1_13_profile,
    v1_13_serving,
    v1_13_variant,
):
    if partA_pred.value is not None and partB_pred.value is not None and partC_pred.value is not None and partD_pred.value is not None:
        ledger.save(chapter=13, design={
            "chapter": "v1_13",
            "track_id": v1_13_profile.track_id,
            "scenario_id": v1_13_variant.scenario_id,
            "hardware_ref": v1_13_serving.hardware_ref,
            "model_ref": v1_13_serving.model_ref,
            "completed": True,
            "p99_latency_prediction": partA_pred.value,
            "batching_tax_prediction": partB_pred.value,
            "state_cache_prediction": partC_pred.value,
            "cold_start_prediction": partD_pred.value,
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
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    batching_tax,
    build_lab_report,
    cache_capacity,
    cold_start_latency,
    mo,
    partA_pred,
    partA_rho,
    partA_slo,
    partA_svc,
    partB_arr,
    partB_batch,
    partB_pred,
    partB_slo,
    partC_ctx,
    partC_gpus,
    partC_prec,
    partC_pred,
    partD_pred,
    partD_scaleout,
    partD_stor,
    partD_warm_pool,
    queueing_latency,
    report_export_panel,
    v1_13_metadata,
    v1_13_model,
    v1_13_profile,
    v1_13_serving,
    v1_13_variant,
):
    _arrival = partA_rho.value * v1_13_serving.replicas * (1000 / partA_svc.value)
    _queue = queueing_latency(
        arrival_qps=_arrival,
        service_ms=partA_svc.value,
        replicas=v1_13_serving.replicas,
        service_cv=v1_13_serving.service_cv,
        slo_ms=partA_slo.value,
    )
    _batching = batching_tax(
        batch_size=partB_batch.value,
        arrival_qps=partB_arr.value,
        service_ms=v1_13_serving.service_ms,
        slo_ms=partB_slo.value,
        efficiency_gain=v1_13_serving.batch_efficiency_gain,
        replicas=v1_13_serving.replicas,
        service_cv=v1_13_serving.service_cv,
    )
    _capacity = cache_capacity(
        v1_13_serving,
        v1_13_model,
        context_tokens=partC_ctx.value,
        precision_bytes=partC_prec.value,
        devices_per_replica=partC_gpus.value,
        kv_precision_bytes=v1_13_serving.kv_precision_bytes,
    )
    _storage_override = {"default": None, "nfs": 1.25, "ram": 50.0}[partD_stor.value]
    _cold = cold_start_latency(
        v1_13_serving,
        precision_bytes=v1_13_serving.precision_bytes,
        storage_bandwidth_gbs=_storage_override,
        warm_pool_replicas=partD_warm_pool.value,
        scale_out_replicas=partD_scaleout.value,
    )

    _incomplete = []
    if partA_pred.value is None:
        _incomplete.append("Part A queueing prediction")
    if partB_pred.value is None:
        _incomplete.append("Part B batching prediction")
    if partC_pred.value is None:
        _incomplete.append("Part C state/cache prediction")
    if partD_pred.value is None:
        _incomplete.append("Part D cold-start prediction")

    _report = build_lab_report(
        v1_13_metadata,
        track=v1_13_profile.label,
        scenario=v1_13_variant.workload_summary,
        learning_objectives=(
            "Quantify how utilization and service variability amplify p99 latency.",
            "Evaluate batching as a throughput/latency trade-off rather than a free win.",
            "Compute live state/cache capacity and cold-start exposure for the selected track.",
        ),
        predictions={
            "queueing_tail_latency": partA_pred.value,
            "batching_tax": partB_pred.value,
            "state_cache_wall": partC_pred.value,
            "cold_start_tax": partD_pred.value,
        },
        knob_settings={
            "rho": partA_rho.value,
            "service_ms": partA_svc.value,
            "partA_slo_ms": partA_slo.value,
            "batch_size": partB_batch.value,
            "batch_arrival_qps": partB_arr.value,
            "batch_slo_ms": partB_slo.value,
            "context_tokens": partC_ctx.value,
            "precision_bytes": partC_prec.value,
            "devices_per_serving_unit": partC_gpus.value,
            "storage_mode": partD_stor.value,
            "warm_pool_replicas": partD_warm_pool.value,
            "scale_out_replicas": partD_scaleout.value,
        },
        evidence_summary={
            "hardware_ref": v1_13_serving.hardware_ref,
            "model_ref": v1_13_serving.model_ref,
            "arrival_qps": round(_arrival, 3),
            "queue_p99_ms": round(_queue.p99_latency_ms, 3),
            "queue_slo_ok": _queue.slo_ok,
            "batch_total_p99_ms": round(_batching.total_p99_ms, 3),
            "formation_delay_ms": round(_batching.formation_delay_ms, 3),
            "state_kind": _capacity.state_kind,
            "state_per_request_gb": round(_capacity.state_per_request_gb, 6),
            "max_live_requests": _capacity.max_concurrent,
            "cold_start_ms": round(_cold.cold_start_ms, 3),
            "exposed_first_request_ms": round(_cold.exposed_first_request_ms, 3),
        },
        final_decision=(
            f"Use the {v1_13_variant.assumptions.get('serving_policy', 'selected serving policy')} "
            f"only if p99, batching delay, {_capacity.state_kind}, and warm-pool evidence "
            f"protect {v1_13_variant.guardrail_metric}."
        ),
        big_takeaways=(
            "Average latency is not a serving SLO.",
            "Batching improves throughput only after paying a formation-delay tax.",
            "Weights, live state/cache, and cold starts must be included in capacity plans.",
        ),
        reflections={
            "track_policy": v1_13_serving.serving_policy,
            "validation_tests": v1_13_serving.validation_tests,
            "residual_question": "Which production traces or hardware counters would you collect before launch?",
        },
        residual_risk=(
            "The helper models are teaching estimates. Production deployment still needs real arrival traces, "
            "service-time histograms, p99/p999 replay, hardware counters, memory telemetry, and warm-pool drills."
        ),
        source_trace={
            "track_id": v1_13_profile.track_id,
            "scenario_id": v1_13_variant.scenario_id,
            "hardware_ref": v1_13_variant.hardware_ref,
            "model_ref": v1_13_variant.model_ref,
            "shared_helper": "mlsysbook_labs.serving",
            "source_policy": v1_13_profile.source_policy,
        },
        result_snapshot={
            "serving_profile": v1_13_serving,
            "queueing": _queue,
            "batching": _batching,
            "capacity": _capacity,
            "cold_start": _cold,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-13 report is generated locally from the selected track, MLSysIM hardware/model refs, "
                "and shared `mlsysbook_labs.serving` calculations."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
