import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


# CELL 0: SETUP

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
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        amdahl_speedup,
        benchmark_track_profile,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        metric_gate,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        sustained_benchmark,
        tail_latency,
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
        amdahl_speedup,
        apply_plotly_theme,
        benchmark_track_profile,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        math,
        metric_gate,
        mo,
        np,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        sustained_benchmark,
        tail_latency,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_12_metadata = get_lab_metadata("vol1/lab_12_perf_bench.py")
    return (v1_12_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_12_track_picker = track_selector(default=_default_track)
    v1_12_track_picker
    return (v1_12_track_picker,)


@app.cell
def _(
    benchmark_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_12_track_picker,
):
    v1_12_track_id = v1_12_track_picker.value
    v1_12_profile = get_track_profile(v1_12_track_id)
    v1_12_variant = get_lab_track_variant("v1_12_benchmarking_trap", v1_12_profile.track_id)
    v1_12_hardware = resolve_mlsysim_ref(v1_12_variant.hardware_ref)
    v1_12_model = resolve_mlsysim_ref(v1_12_variant.model_ref)
    v1_12_benchmark = benchmark_track_profile(
        v1_12_profile,
        v1_12_variant,
        v1_12_hardware,
        v1_12_model,
    )
    return (
        v1_12_benchmark,
        v1_12_hardware,
        v1_12_model,
        v1_12_profile,
        v1_12_track_id,
        v1_12_variant,
    )


# NOTEBOOK-LOCAL SUPPORT

@app.cell
def _(math, np, sustained_benchmark, tail_latency):
    def v1_12_track_packet(benchmark, variant):
        common = {
            "stakeholder": variant.stakeholder,
            "headline_scope": "headline benchmark",
            "score_threshold": 75,
            "cv_limit_pct": 5.0,
            "ci_limit_pct": 5.0,
            "source_note": (
                "Track identity and hardware/model refs come from the lab registry. "
                "Protocol scores are notebook-local teaching estimates."
            ),
        }
        tracks = {
            "iphone": {
                "production_question": "Can this on-device feature survive a sustained user session?",
                "easy_scope": "cold-run latency on a freshly started phone",
                "production_scope": "10-minute sustained device run with battery and thermal evidence",
                "selected_metric": "sustained p95 UX latency",
                "guardrail_label": "battery drain and thermal headroom",
                "amount_line_a": "The amount system is p95 latency in ms, battery percent/hour, and thermal headroom.",
                "amount_line_b": "Discard cold/cache warmup, then measure enough sustained samples to see thermal drift.",
                "amount_line_c": "A mean latency win is blocked if p95 UX latency or battery/thermal limits fail.",
                "amount_line_d": "Runs are comparable only with the same device class, ambient, run length, battery state, and p95 evidence.",
                "warmup_floor": 10,
                "sample_floor": 20,
                "failure_unit": "bad UI frames per second",
                "accepted_comparison": "same-device sustained run",
                "rejected_comparison": "short cold-run latency claim",
                "conditions": (
                    ("same_workload", "same sustained user workload"),
                    ("same_warmup", "same warmup discard"),
                    ("same_samples", "same sample count and duration"),
                    ("same_hardware", "same phone, ambient, and battery state"),
                    ("same_guardrail", "same p95, battery, and thermal guardrails"),
                ),
            },
            "oura_ring": {
                "production_question": "Can always-on sensing fit the overnight energy and memory window?",
                "easy_scope": "isolated single-inference loop",
                "production_scope": "24-hour duty-cycle replay with SRAM, flash, OTA, and energy accounting",
                "selected_metric": "energy per sensing window",
                "guardrail_label": "SRAM/flash fit and battery days",
                "amount_line_a": "The amount system is uJ/window, wake windows/day, SRAM KB, flash KB, and OTA payload.",
                "amount_line_b": "Many small windows are needed because each measurement is tiny and noise can dominate.",
                "amount_line_c": "A latency win is blocked if energy/window, SRAM, or flash exceeds the wearable envelope.",
                "amount_line_d": "Runs are comparable only with the same sensing cadence, memory accounting, and energy boundary.",
                "warmup_floor": 8,
                "sample_floor": 30,
                "failure_unit": "energy budget multiplier",
                "accepted_comparison": "same duty-cycle replay",
                "rejected_comparison": "isolated inference latency claim",
                "conditions": (
                    ("same_workload", "same duty-cycle replay"),
                    ("same_warmup", "same sensor/window warmup discard"),
                    ("same_samples", "same number of replay windows"),
                    ("same_hardware", "same SRAM, flash, and firmware build"),
                    ("same_guardrail", "same energy, OTA, SRAM, and flash guardrails"),
                ),
            },
            "robotaxi": {
                "production_question": "Can perception meet the deadline during bursts and rare events?",
                "easy_scope": "average FPS on ordinary frames",
                "production_scope": "synchronized sensor-burst and rare-event replay",
                "selected_metric": "p99/p999 perception deadline",
                "guardrail_label": "rare-event recall and safety margin",
                "amount_line_a": "The amount system is p99/p999 latency, frame deadline, burst multiplier, and replay count.",
                "amount_line_b": "Rare-event and p999 claims need more samples than average-frame claims.",
                "amount_line_c": "Average FPS is blocked if p99/p999 misses the deadline or rare-event recall drops.",
                "amount_line_d": "Runs are comparable only with the same sensor mix, deadline, rare-event count, and recall floor.",
                "warmup_floor": 12,
                "sample_floor": 50,
                "failure_unit": "frames over deadline",
                "accepted_comparison": "same sensor-burst replay",
                "rejected_comparison": "average-frame FPS claim",
                "conditions": (
                    ("same_workload", "same synchronized sensor burst"),
                    ("same_warmup", "same perception stack warmup"),
                    ("same_samples", "same rare-event replay count"),
                    ("same_hardware", "same vehicle compute and sensor configuration"),
                    ("same_guardrail", "same p99/p999 deadline and recall guardrail"),
                ),
            },
            "cloud_fleet": {
                "production_question": "Can the fleet satisfy load, SLA, utilization, and cost/request?",
                "easy_scope": "peak Offline throughput at the largest batch",
                "production_scope": "Server-style load test with p99, utilization, quality, and cost/request",
                "selected_metric": "SLA-compliant p99 at production load",
                "guardrail_label": "utilization, quality, and cost/request",
                "amount_line_a": "The amount system is QPS, p99 latency, utilization, replicas, and dollars/request.",
                "amount_line_b": "Load tests need enough request samples to estimate p99 and utilization with confidence.",
                "amount_line_c": "Peak throughput is blocked if p99, utilization, quality, or cost/request fails.",
                "amount_line_d": "Runs are comparable only with the same demand trace, replica budget, SLA, and cost model.",
                "warmup_floor": 10,
                "sample_floor": 40,
                "failure_unit": "SLA misses per second",
                "accepted_comparison": "same production load trace",
                "rejected_comparison": "peak Offline throughput claim",
                "conditions": (
                    ("same_workload", "same load trace and arrival model"),
                    ("same_warmup", "same warmup and cache state"),
                    ("same_samples", "same request count"),
                    ("same_hardware", "same replica budget and accelerator class"),
                    ("same_guardrail", "same p99, quality, utilization, and cost guardrails"),
                ),
            },
        }
        packet = dict(common)
        packet.update(tracks.get(benchmark.track_id, tracks["iphone"]))
        return packet

    def v1_12_metric_cards(COLORS, cards):
        html = ['<div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">']
        for label, value, note, color in cards:
            html.append(
                f"""
                <div style="padding:14px 16px; border:1px solid #e2e8f0; border-radius:8px;
                            background:white; border-top:3px solid {color}; flex:1; min-width:180px;">
                    <div style="color:#64748b; font-size:0.74rem; font-weight:700;
                                text-transform:uppercase; letter-spacing:0.08em;">{label}</div>
                    <div style="font-size:1.35rem; font-weight:800; color:{color}; margin-top:4px;">{value}</div>
                    <div style="font-size:0.76rem; color:#64748b; margin-top:4px;">{note}</div>
                </div>
                """
            )
        html.append("</div>")
        return "".join(html)

    def v1_12_html_table(rows, columns):
        def escape(value):
            return (
                str(value)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        header = "".join(f"<th>{escape(column)}</th>" for column in columns)
        body_rows = []
        for row in rows:
            body_rows.append(
                "<tr>" + "".join(f"<td>{escape(row.get(column, ''))}</td>" for column in columns) + "</tr>"
            )
        body = "".join(body_rows)
        return f"""
        <table style="width:100%; border-collapse:collapse; margin:12px 0; font-size:0.86rem;">
            <thead>
                <tr style="background:#f8fafc; color:#334155; text-align:left;">{header}</tr>
            </thead>
            <tbody>{body}</tbody>
        </table>
        <style>
        table th, table td {{
            border: 1px solid #e2e8f0;
            padding: 8px 10px;
            vertical-align: top;
        }}
        </style>
        """

    def v1_12_validity_result(benchmark, packet, workload_match_pct, metric_choice):
        workload_score = max(0.0, min(1.0, workload_match_pct / 100))
        metric_scores = {"headline": 0.35, "production": 1.0, "guardrail": 0.85}
        metric_score = metric_scores.get(metric_choice or "headline", 0.35)
        duration_score = 0.25 + 0.75 * workload_score
        guardrail_score = 1.0 if metric_choice in {"production", "guardrail"} else 0.30
        validity = 100 * (
            0.35 * workload_score
            + 0.30 * metric_score
            + 0.20 * duration_score
            + 0.15 * guardrail_score
        )
        valid = validity >= packet["score_threshold"]
        easy_score = 42
        tail = tail_latency(
            base_ms=benchmark.tail_base_ms,
            sigma=benchmark.tail_sigma,
            slo_ms=benchmark.tail_slo_ms,
        )
        sustained = sustained_benchmark(
            peak_value=benchmark.burst_value,
            tdp_w=benchmark.tdp_w,
            duration_s=benchmark.default_duration_s,
            ambient_c=benchmark.default_ambient_c,
            cooling=benchmark.default_cooling,
        )
        track = benchmark.track_id
        if track == "iphone":
            production_value = (
                f"p95 {tail.p95_ms:.0f} ms, sustained {sustained.sustained_value:.0f} {benchmark.metric_unit}, "
                f"battery {3.5 + 0.045 * workload_match_pct:.1f}%/h"
            )
        elif track == "oura_ring":
            production_value = f"{70 + 0.55 * workload_match_pct:.0f} uJ/window, 236 KB SRAM"
        elif track == "robotaxi":
            production_value = (
                f"p99 {tail.p99_ms:.0f} ms, p999 {tail.p999_ms:.0f} ms, "
                f"sustained {sustained.sustained_value:.0f} {benchmark.metric_unit}"
            )
        else:
            production_value = (
                f"p99 {tail.p99_ms:.0f} ms, sustained {sustained.sustained_value:.0f} {benchmark.metric_unit}, "
                f"utilization {55 + 0.30 * workload_match_pct:.0f}%"
            )
        selected_metric = {
            "headline": benchmark.benchmark_claim,
            "production": packet["selected_metric"],
            "guardrail": packet["guardrail_label"],
        }.get(metric_choice or "headline", benchmark.benchmark_claim)
        rows = (
            {
                "Benchmark": "Headline",
                "Workload": packet["easy_scope"],
                "Metric": benchmark.benchmark_claim,
                "Measured amount": f"{benchmark.burst_value:g} {benchmark.metric_unit}",
                "Decision support": "Reject",
            },
            {
                "Benchmark": "Production-like",
                "Workload": packet["production_scope"],
                "Metric": packet["selected_metric"],
                "Measured amount": production_value,
                "Decision support": "Accept" if valid else "Incomplete",
            },
        )
        return {
            "easy_score": easy_score,
            "validity_score": validity,
            "valid": valid,
            "selected_metric": selected_metric,
            "production_value": production_value,
            "rows": rows,
        }

    def v1_12_confidence_result(benchmark, packet, warmup, samples, jitter_pct):
        sample_count = max(1, int(samples))
        total = min(120, max(warmup + sample_count + 8, 12))
        base = max(1.0, float(benchmark.tail_base_ms))
        drift_map = {
            "iphone": 0.035,
            "oura_ring": 0.004,
            "robotaxi": 0.025,
            "cloud_fleet": 0.018,
        }
        drift_per_iter = drift_map.get(benchmark.track_id, 0.01)
        values = []
        for i in range(total):
            warmup_penalty = 0.22 * math.exp(-i / 6.0)
            jitter = (jitter_pct / 100.0) * (math.sin(1.7 * i) + 0.35 * math.cos(0.4 * i))
            drift = drift_per_iter * i
            values.append(base * (1 + warmup_penalty + jitter) + drift)
        start = min(max(0, int(warmup)), total - 1)
        stop = min(total, start + sample_count)
        measured = values[start:stop]
        n = len(measured)
        mean = float(np.mean(measured))
        std = float(np.std(measured, ddof=1)) if n > 1 else 0.0
        cv_pct = (std / mean * 100) if mean else 0.0
        ci_half = 1.96 * std / math.sqrt(n) if n else 0.0
        ci_pct = (ci_half / mean * 100) if mean else 0.0
        run_rule_ok = warmup >= packet["warmup_floor"] and n >= packet["sample_floor"]
        confidence_ok = (
            run_rule_ok
            and cv_pct <= packet["cv_limit_pct"]
            and ci_pct <= packet["ci_limit_pct"]
        )
        rows = (
            {"Check": "Warmup discard", "Measured": f"{warmup} iterations", "Limit": f">= {packet['warmup_floor']}", "Verdict": "PASS" if warmup >= packet["warmup_floor"] else "FAIL"},
            {"Check": "Measured samples", "Measured": f"{n}", "Limit": f">= {packet['sample_floor']}", "Verdict": "PASS" if n >= packet["sample_floor"] else "FAIL"},
            {"Check": "Coefficient of variation", "Measured": f"{cv_pct:.1f}%", "Limit": f"<= {packet['cv_limit_pct']:.1f}%", "Verdict": "PASS" if cv_pct <= packet["cv_limit_pct"] else "FAIL"},
            {"Check": "95% CI half-width", "Measured": f"+/- {ci_half:.1f} ms ({ci_pct:.1f}%)", "Limit": f"<= {packet['ci_limit_pct']:.1f}% of mean", "Verdict": "PASS" if ci_pct <= packet["ci_limit_pct"] else "FAIL"},
        )
        return {
            "values": values,
            "measured": measured,
            "mean": mean,
            "std": std,
            "cv_pct": cv_pct,
            "ci_half": ci_half,
            "ci_pct": ci_pct,
            "confidence_ok": confidence_ok,
            "run_rule_ok": run_rule_ok,
            "rows": rows,
        }

    def v1_12_guardrail_result(benchmark, packet, sigma, stress):
        tail = tail_latency(
            base_ms=benchmark.tail_base_ms,
            sigma=sigma,
            slo_ms=benchmark.tail_slo_ms,
        )
        rows = [
            {"Metric": "Mean latency", "Measured": f"{tail.mean_ms:.0f} ms", "Limit": "not sufficient alone", "Verdict": "INFO"},
            {"Metric": "P95 latency", "Measured": f"{tail.p95_ms:.0f} ms", "Limit": "track context", "Verdict": "INFO"},
            {"Metric": "P99 latency", "Measured": f"{tail.p99_ms:.0f} ms", "Limit": f"<= {benchmark.tail_slo_ms:g} ms", "Verdict": "PASS" if tail.p99_ms <= benchmark.tail_slo_ms else "FAIL"},
            {"Metric": "P99.9 latency", "Measured": f"{tail.p999_ms:.0f} ms", "Limit": "tail evidence", "Verdict": "INFO"},
        ]
        blocked_metrics = []
        track = benchmark.track_id
        if track == "iphone":
            p95_limit = benchmark.p99_max_ms
            battery = 3.2 + 4.1 * stress + max(0, tail.p95_ms - p95_limit) * 0.02
            thermal = 34 + 10 * stress + max(0, sigma - 0.6) * 10
            extra = (
                ("p95 UX latency", tail.p95_ms, p95_limit, "ms"),
                ("battery drain", battery, 7.0, "%/h"),
                ("skin temperature proxy", thermal, 43.0, "C"),
            )
            failure_rate = f"{tail.violation_pct / 100 * 60:.1f} bad UI frames/s at 60 FPS"
        elif track == "oura_ring":
            energy = 75 * stress * (1 + sigma / 2)
            sram = 180 + 60 * stress + 12 * sigma
            flash = 720 + 180 * stress
            extra = (
                ("energy per window", energy, 120.0, "uJ"),
                ("SRAM footprint", sram, 256.0, "KB"),
                ("flash plus OTA payload", flash, 1024.0, "KB"),
            )
            failure_rate = f"{energy / 120.0:.2f}x energy/window budget"
        elif track == "robotaxi":
            recall = 96.5 - 2.5 * stress - max(0, sigma - 0.55) * 6
            extra = (
                ("p99 deadline", tail.p99_ms, benchmark.p99_max_ms, "ms"),
                ("p999 deadline", tail.p999_ms, benchmark.tail_slo_ms, "ms"),
                ("rare-event recall", recall, 92.0, "% min"),
            )
            failure_rate = f"{tail.violation_pct:.2f}% frames over {benchmark.tail_slo_ms:g} ms"
        else:
            utilization = min(0.99, 0.55 + 0.25 * stress + max(0, sigma - 0.6) * 0.10)
            cost = 0.0011 * stress * (1 + max(0, sigma - 0.6) * 0.35)
            quality = 91.8 - max(0, stress - 1) * 2
            extra = (
                ("p99 SLA latency", tail.p99_ms, benchmark.p99_max_ms, "ms"),
                ("utilization", utilization * 100, 82.0, "%"),
                ("cost/request", cost, 0.0018, "$"),
                ("quality floor", quality, 90.0, "% min"),
            )
            failure_rate = f"{tail.violation_pct / 100 * benchmark.throughput_min:.1f} SLA misses/s"
        for label, measured, limit, unit in extra:
            if "min" in unit:
                ok = measured >= limit
                limit_text = f">= {limit:g} {unit.replace(' min', '')}"
            else:
                ok = measured <= limit
                limit_text = f"<= {limit:g} {unit}"
            if not ok:
                blocked_metrics.append(label)
            rows.append(
                {
                    "Metric": label,
                    "Measured": f"{measured:.3f} {unit}" if unit == "$" else f"{measured:.1f} {unit.replace(' min', '')}",
                    "Limit": limit_text,
                    "Verdict": "PASS" if ok else "FAIL",
                }
            )
        blocked = bool(blocked_metrics or tail.p99_ms > benchmark.tail_slo_ms)
        return {
            "tail": tail,
            "blocked": blocked,
            "blocked_metrics": tuple(blocked_metrics),
            "failure_rate": failure_rate,
            "rows": tuple(rows),
        }

    def v1_12_fairness_result(packet, controls):
        conditions = packet["conditions"]
        missing = tuple(label for key, label in conditions if not controls.get(key, False))
        controlled = len(conditions) - len(missing)
        index = 100 * controlled / len(conditions)
        reportable = not missing
        rows = (
            {
                "Comparison": packet["rejected_comparison"],
                "Reported win": "larger headline number",
                "Run-rule state": "missing production evidence",
                "Reportable": "No",
            },
            {
                "Comparison": packet["accepted_comparison"],
                "Reported win": "smaller but controlled improvement",
                "Run-rule state": f"{controlled}/{len(conditions)} controls satisfied",
                "Reportable": "Yes" if reportable else "Not yet",
            },
        )
        return {
            "index": index,
            "missing": missing,
            "reportable": reportable,
            "rows": rows,
            "accepted": packet["accepted_comparison"] if reportable else "No comparison is reportable yet",
            "rejected": packet["rejected_comparison"],
        }

    return (
        v1_12_confidence_result,
        v1_12_fairness_result,
        v1_12_guardrail_result,
        v1_12_html_table,
        v1_12_metric_cards,
        v1_12_track_packet,
        v1_12_validity_result,
    )


@app.cell
def _(v1_12_benchmark, v1_12_track_packet, v1_12_variant):
    v1_12_packet = v1_12_track_packet(v1_12_benchmark, v1_12_variant)
    return (v1_12_packet,)


# CELL 1: HEADER

@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    track_arc_context,
    track_context,
    v1_12_benchmark,
    v1_12_metadata,
    v1_12_packet,
    v1_12_profile,
    v1_12_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #111827 0%, #1f2937 58%, #0f3b3e 100%);
                    padding: 34px 42px; border-radius: 14px; color: white;
                    box-shadow: 0 8px 30px rgba(0,0,0,0.30);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #9ca3af; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 12
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.35rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Performance Benchmarking
            </h1>
            <p style="margin: 0 0 10px 0; font-size: 1.05rem; font-weight: 600;
                      color: #a7f3d0; letter-spacing: 0.03em; font-family: 'SF Mono', monospace;">
                Workload validity &middot; Confidence &middot; Tail guardrails &middot; Fair comparison
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #d1d5db;
                      max-width: 760px; line-height: 1.65;">
                {v1_12_variant.workload_summary} Your job is to turn the benchmark claim
                <strong>{v1_12_benchmark.benchmark_claim}</strong> into reportable evidence
                for: {v1_12_packet["production_question"]}
            </p>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Shared A/B/C/D concept sequence</span>
                <span class="badge badge-warn">{v1_12_profile.label}</span>
                <span class="badge badge-fail">{v1_12_benchmark.hidden_failure_metric}</span>
            </div>
        </div>
        """),
        track_context(v1_12_profile),
        track_arc_context(v1_12_profile, v1_12_metadata.lab_id),
    ])
    return


# CELL 2: BRIEFING

@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
            Chapter Invariant
        </div>
        <div style="font-size: 1.02rem; color: {COLORS['Text']}; line-height: 1.65; font-weight: 600;">
            Measurement changes decisions. A benchmark is only valid when workload,
            warmup, variance, tail behavior, and comparison rules match the deployment question.
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 16px -28px 0 -28px;
                    padding: 16px 28px 0 28px; display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
                    gap:16px; font-size:0.88rem; color:{COLORS['TextSec']}; line-height:1.55;">
            <div><strong>Part A:</strong> benchmark validity depends on matching the production workload and metric.</div>
            <div><strong>Part B:</strong> warmup, variance, and sample size determine confidence.</div>
            <div><strong>Part C:</strong> averages can hide tail and guardrail failures.</div>
            <div><strong>Part D:</strong> fair comparison requires controlled conditions and reportable evidence.</div>
        </div>
    </div>
    """)
    return


# CELL 3: READING

@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete these chapter anchors before the lab:

    - **Benchmarks as proxies** and **Benchmarking Granularity** for workload and metric validity.
    - **Micro-benchmarking rules** and **Statistical and methodological issues** for warmup, variance, and confidence.
    - **Latency and tail latency** plus **Fallacies and Pitfalls** for average-vs-tail failures.
    - **Benchmark Components**, **Run Rules**, and **MLPerf execution scenarios** for fair comparison.
    """), kind="info")
    return


# CELL 4: WIDGETS

@app.cell(hide_code=True)
def _(mo, v1_12_benchmark, v1_12_packet):
    pA_pred = mo.ui.radio(
        options={
            "A) Accept the headline result because it is standardized": "accept_headline",
            "B) Reject until workload and metric match deployment": "match_deployment",
            "C) Use the highest throughput metric because it is easiest to compare": "throughput",
            "D) Cannot decide without a model accuracy score": "accuracy_only",
        },
        label=(
            f"Part A prediction: {v1_12_benchmark.benchmark_claim} is available. "
            f"Can it answer: {v1_12_packet['production_question']}"
        ),
    )
    pA_workload = mo.ui.slider(
        start=30,
        stop=100,
        value=55,
        step=5,
        label="Production workload match (%)",
    )
    pA_metric = mo.ui.radio(
        options={
            f"Headline metric: {v1_12_benchmark.benchmark_claim}": "headline",
            f"Production metric: {v1_12_packet['selected_metric']}": "production",
            f"Guardrail metric: {v1_12_packet['guardrail_label']}": "guardrail",
        },
        value=f"Production metric: {v1_12_packet['selected_metric']}",
        label="Metric used for the deployment decision",
    )
    pA_decision = mo.ui.radio(
        options={
            "Report the production-like benchmark only": "production",
            "Report the headline benchmark": "headline",
            "Report both without qualifying scope": "both",
        },
        label="Checkpoint: which result belongs in the report?",
    )
    return (pA_decision, pA_metric, pA_pred, pA_workload)


@app.cell(hide_code=True)
def _(mo, v1_12_packet):
    pB_pred = mo.ui.radio(
        options={
            "A) One clean run is enough": "one",
            "B) Five runs with no warmup discard are enough": "five_no_warmup",
            "C) Discard warmup and report repeated-run confidence": "warmup_confidence",
            "D) Skip offline confidence and rely on production incidents": "prod_only",
        },
        label="Part B prediction: what run rule makes the benchmark claim believable?",
    )
    pB_warmup = mo.ui.slider(
        start=0,
        stop=40,
        value=v1_12_packet["warmup_floor"],
        step=2,
        label="Warmup iterations discarded",
    )
    pB_samples = mo.ui.slider(
        start=3,
        stop=80,
        value=max(20, v1_12_packet["sample_floor"]),
        step=1,
        label="Measured sample count",
    )
    pB_jitter = mo.ui.slider(
        start=1,
        stop=20,
        value=6,
        step=1,
        label="Environment jitter (%)",
    )
    pB_decision = mo.ui.radio(
        options={
            "Claim is confident enough for release": "confident",
            "Claim is underpowered or noisy": "noisy",
            "Claim needs a different production metric": "wrong_metric",
        },
        label="Checkpoint: what confidence verdict goes in the report?",
    )
    return (pB_decision, pB_jitter, pB_pred, pB_samples, pB_warmup)


@app.cell(hide_code=True)
def _(mo, v1_12_benchmark, v1_12_packet):
    pC_pred = mo.ui.radio(
        options={
            "A) The mean is enough if it is under budget": "mean_enough",
            "B) Tail percentile evidence is required": "tail_required",
            "C) The guardrail can be checked later": "guardrail_later",
            "D) Only accuracy can block deployment": "accuracy_only",
        },
        label=(
            f"Part C prediction: the mean looks healthy. Does that prove "
            f"{v1_12_packet['guardrail_label']} is safe?"
        ),
    )
    pC_sigma = mo.ui.slider(
        start=0.1,
        stop=1.5,
        value=v1_12_benchmark.tail_sigma,
        step=0.05,
        label="Tail heaviness (sigma)",
    )
    pC_stress = mo.ui.slider(
        start=0.6,
        stop=1.8,
        value=1.0,
        step=0.1,
        label="Guardrail stress multiplier",
    )
    pC_decision = mo.ui.radio(
        options={
            "Block release on tail or guardrail evidence": "block",
            "Approve because the mean passed": "approve_mean",
            "Approve with no guardrail in scope": "no_guardrail",
        },
        label="Checkpoint: what blocks or approves the release?",
    )
    return (pC_decision, pC_pred, pC_sigma, pC_stress)


@app.cell(hide_code=True)
def _(mo, v1_12_packet):
    pD_pred = mo.ui.radio(
        options={
            "A) The larger headline number wins": "headline",
            "B) The controlled comparison wins even if the gain is smaller": "controlled",
            "C) Both comparisons are reportable": "both",
            "D) Neither comparison is useful for deployment": "neither",
        },
        label="Part D prediction: which comparison is fair enough to report?",
    )
    pD_same_workload = mo.ui.checkbox(value=False, label=v1_12_packet["conditions"][0][1])
    pD_same_warmup = mo.ui.checkbox(value=True, label=v1_12_packet["conditions"][1][1])
    pD_same_samples = mo.ui.checkbox(value=False, label=v1_12_packet["conditions"][2][1])
    pD_same_hardware = mo.ui.checkbox(value=True, label=v1_12_packet["conditions"][3][1])
    pD_same_guardrail = mo.ui.checkbox(value=False, label=v1_12_packet["conditions"][4][1])
    pD_decision = mo.ui.radio(
        options={
            "Report only the controlled comparison": "controlled",
            "Report the headline-only comparison": "headline",
            "Withhold the comparison until missing evidence is collected": "withhold",
        },
        label="Checkpoint: which comparison enters the benchmark report?",
    )
    return (
        pD_decision,
        pD_pred,
        pD_same_guardrail,
        pD_same_hardware,
        pD_same_samples,
        pD_same_warmup,
        pD_same_workload,
    )


# CELL 5: TABS

@app.cell(hide_code=True)
def _(
    COLORS,
    amdahl_speedup,
    apply_plotly_theme,
    go,
    mo,
    np,
    pA_decision,
    pA_metric,
    pA_pred,
    pA_workload,
    pB_decision,
    pB_jitter,
    pB_pred,
    pB_samples,
    pB_warmup,
    pC_decision,
    pC_pred,
    pC_sigma,
    pC_stress,
    pD_decision,
    pD_pred,
    pD_same_guardrail,
    pD_same_hardware,
    pD_same_samples,
    pD_same_warmup,
    pD_same_workload,
    v1_12_benchmark,
    v1_12_confidence_result,
    v1_12_fairness_result,
    v1_12_guardrail_result,
    v1_12_html_table,
    v1_12_metric_cards,
    v1_12_packet,
    v1_12_validity_result,
    v1_12_variant,
):
    def build_part_a():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Part A Concept Module &middot; Benchmark validity
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;{v1_12_variant.stakeholder}: The claim is {v1_12_benchmark.benchmark_claim}.
                    I need to know whether it answers: {v1_12_packet['production_question']}&rdquo;
                </div>
            </div>
            """),
            mo.md(f"""
            ## Valid Benchmarks Match Workload And Metric

            A benchmark is not a universal number. It is evidence for one workload,
            one metric, and one decision boundary. For this track, the production
            boundary is: **{v1_12_packet['production_scope']}**.
            """),
            pA_pred,
        ]
        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Commit to a validity prediction before opening the evidence."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pA_workload, pA_metric], justify="start"))
        metric_choice = pA_metric.value or "headline"
        result = v1_12_validity_result(
            v1_12_benchmark,
            v1_12_packet,
            pA_workload.value,
            metric_choice,
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Headline", "Production-like"],
            y=[result["easy_score"], result["validity_score"]],
            marker_color=[COLORS["OrangeLine"], COLORS["BlueLine"] if result["valid"] else COLORS["RedLine"]],
            text=[f"{result['easy_score']:.0f}", f"{result['validity_score']:.0f}"],
            textposition="outside",
        ))
        fig.add_hline(
            y=v1_12_packet["score_threshold"],
            line_dash="dash",
            line_color=COLORS["GreenLine"],
            annotation_text=f"Report threshold: {v1_12_packet['score_threshold']}",
        )
        fig.update_layout(height=320, yaxis=dict(title="Validity score", range=[0, 105]), showlegend=False)
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        amdahl = amdahl_speedup(
            component_speedup=v1_12_benchmark.default_speedup,
            serial_pct=v1_12_benchmark.default_serial_pct,
        )
        items.append(mo.Html(v1_12_metric_cards(COLORS, (
            ("Selected Metric", result["selected_metric"], "metric used for the deployment decision", COLORS["BlueLine"]),
            ("Validity Score", f"{result['validity_score']:.0f}/100", "weighted workload/metric/run-rule overlap", COLORS["GreenLine"] if result["valid"] else COLORS["RedLine"]),
            ("Amdahl Cross-Check", f"{amdahl.system_speedup:.2f}x", f"{v1_12_benchmark.default_speedup:g}x component speedup after pipeline overhead", COLORS["OrangeLine"]),
        ))))
        items.append(mo.Html(v1_12_html_table(result["rows"], ("Benchmark", "Workload", "Metric", "Measured amount", "Decision support"))))
        items.append(mo.callout(
            mo.md(f"You predicted `{pA_pred.value}`; actual benchmark evidence must match deployment workload and metric."),
            kind="info",
        ))

        if pA_pred.value == "match_deployment":
            items.append(mo.callout(mo.md("**Correct.** The valid benchmark is the one whose workload and metric match the deployment question."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Consequence:** the headline number misses {v1_12_benchmark.hidden_failure_metric}. "
                "A benchmark can be precise and still answer the wrong question."
            ), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Validity Overlap": mo.md(f"""
            The local validity score weights four overlaps:

            ```
            validity = 0.35*workload + 0.30*metric + 0.20*duration + 0.15*guardrail
            ```

            Workload match is `{pA_workload.value}%`. The selected metric is
            `{result["selected_metric"]}`. This mirrors the chapter claim that
            benchmark results are proxies whose boundaries must be named.

            Source model: Chapter sections on benchmarks as proxies, Benchmarking
            Granularity, MLPerf execution scenarios, and `mlsysbook_labs.amdahl_speedup`
            for the component-vs-end-to-end cross-check.
            """)
        }))
        items.append(pA_decision)
        if pA_decision.value is None:
            items.append(mo.callout(mo.md("Checkpoint: choose which benchmark result belongs in the final report."), kind="info"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Part B Concept Module &middot; Confidence
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;The benchmark number looks precise. Is the run rule strong enough to
                    publish it for {v1_12_benchmark.label}?&rdquo;
                </div>
            </div>
            """),
            mo.md(f"""
            ## Warmup, Variance, And Sample Size Set Confidence

            {v1_12_packet["amount_line_b"]} A reportable benchmark needs warmup
            discard, repeated samples, coefficient of variation, and a confidence
            interval rather than a single impressive run.
            """),
            pB_pred,
        ]
        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Commit to a confidence prediction before opening the run-rule controls."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pB_warmup, pB_samples, pB_jitter], justify="start"))
        result = v1_12_confidence_result(
            v1_12_benchmark,
            v1_12_packet,
            pB_warmup.value,
            pB_samples.value,
            pB_jitter.value,
        )
        xs = list(range(len(result["values"])))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs,
            y=result["values"],
            mode="lines+markers",
            line=dict(color=COLORS["BlueLine"], width=2),
            marker=dict(size=5),
            name="observed run value",
        ))
        if pB_warmup.value > 0:
            fig.add_vrect(
                x0=-0.5,
                x1=pB_warmup.value - 0.5,
                fillcolor=COLORS["OrangeLine"],
                opacity=0.14,
                line_width=0,
                annotation_text="discarded warmup",
            )
        fig.add_hline(y=result["mean"], line_dash="dash", line_color=COLORS["GreenLine"], annotation_text=f"measured mean {result['mean']:.1f}")
        fig.update_layout(height=330, xaxis=dict(title="Iteration"), yaxis=dict(title="Measured latency proxy (ms)"))
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        items.append(mo.Html(v1_12_metric_cards(COLORS, (
            ("Measured Samples", f"{pB_samples.value}", f"floor: {v1_12_packet['sample_floor']}", COLORS["BlueLine"]),
            ("CV", f"{result['cv_pct']:.1f}%", f"limit: {v1_12_packet['cv_limit_pct']:.1f}%", COLORS["GreenLine"] if result["cv_pct"] <= v1_12_packet["cv_limit_pct"] else COLORS["RedLine"]),
            ("95% CI", f"+/- {result['ci_half']:.1f} ms", f"{result['ci_pct']:.1f}% of mean", COLORS["GreenLine"] if result["confidence_ok"] else COLORS["RedLine"]),
        ))))
        items.append(mo.Html(v1_12_html_table(result["rows"], ("Check", "Measured", "Limit", "Verdict"))))

        if result["confidence_ok"]:
            items.append(mo.callout(mo.md("**Confidence gate passes.** The run rule is strong enough to support a bounded benchmark claim."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Confidence gate fails.** A precise-looking number can be underpowered, noisy, or polluted by warmup artifacts."
            ), kind="danger"))

        if pB_pred.value == "warmup_confidence":
            items.append(mo.callout(mo.md("**Correct.** Warmup plus repeated-run confidence is the minimum credible rule."), kind="success"))
        else:
            items.append(mo.callout(mo.md("The chapter's warmup and variance rules reject single-run claims."), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: CV And Confidence": mo.md(f"""
            ```
            CV = sigma_run / mean_run
            95% CI half-width = 1.96 * sigma_run / sqrt(n)
            ```

            Current values:

            ```
            mean = {result["mean"]:.2f} ms
            sigma = {result["std"]:.2f} ms
            n = {pB_samples.value}
            CV = {result["cv_pct"]:.2f}%
            CI = +/- {result["ci_half"]:.2f} ms
            ```

            Source model: Chapter micro-benchmarking rules and statistical
            confidence discussion.
            """)
        }))
        items.append(pB_decision)
        if pB_decision.value is None:
            items.append(mo.callout(mo.md("Checkpoint: choose the confidence verdict for the benchmark report."), kind="info"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Part C Concept Module &middot; Tail and guardrails
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;The mean looks good, but production incidents mention
                    {v1_12_benchmark.hidden_failure_metric}. What did the average hide?&rdquo;
                </div>
            </div>
            """),
            mo.md(f"""
            ## Averages Hide Tail And Guardrail Failures

            {v1_12_packet["amount_line_c"]} This module keeps the concept
            shared across tracks: the mean can pass while the deployment guardrail fails.
            """),
            pC_pred,
        ]
        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Commit to a mean-vs-tail prediction before opening the distribution."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pC_sigma, pC_stress], justify="start"))
        result = v1_12_guardrail_result(v1_12_benchmark, v1_12_packet, pC_sigma.value, pC_stress.value)
        tail = result["tail"]
        rng = np.random.default_rng(12012)
        samples = v1_12_benchmark.tail_base_ms * np.exp(rng.normal(0, pC_sigma.value, 6000))
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=samples.tolist(),
            nbinsx=90,
            marker_color=COLORS["BlueLine"],
            opacity=0.72,
            name="latency samples",
        ))
        fig.add_vline(x=tail.mean_ms, line_dash="dash", line_color=COLORS["BlueLine"], annotation_text=f"mean {tail.mean_ms:.0f} ms")
        fig.add_vline(x=tail.p99_ms, line_dash="solid", line_color=COLORS["RedLine"], annotation_text=f"p99 {tail.p99_ms:.0f} ms")
        fig.add_vline(x=v1_12_benchmark.tail_slo_ms, line_dash="dot", line_color=COLORS["GreenLine"], annotation_text=f"limit {v1_12_benchmark.tail_slo_ms:g} ms")
        fig.update_layout(
            height=340,
            xaxis=dict(title="Latency sample (ms)", range=[0, min(max(tail.p999_ms * 1.15, tail.mean_ms * 4), 1200)]),
            yaxis=dict(title="Count"),
            showlegend=False,
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        verdict_color = COLORS["RedLine"] if result["blocked"] else COLORS["GreenLine"]
        items.append(mo.Html(v1_12_metric_cards(COLORS, (
            ("Mean", f"{tail.mean_ms:.0f} ms", "headline average", COLORS["BlueLine"]),
            ("P99", f"{tail.p99_ms:.0f} ms", f"limit {v1_12_benchmark.tail_slo_ms:g} ms", COLORS["RedLine"] if tail.p99_ms > v1_12_benchmark.tail_slo_ms else COLORS["GreenLine"]),
            ("Guardrail", "BLOCK" if result["blocked"] else "PASS", result["failure_rate"], verdict_color),
        ))))
        items.append(mo.Html(v1_12_html_table(result["rows"], ("Metric", "Measured", "Limit", "Verdict"))))

        if result["blocked"]:
            items.append(mo.callout(mo.md(
                f"**Release blocked.** Mean latency does not cover {v1_12_packet['guardrail_label']}. "
                f"Failure consequence: {result['failure_rate']}."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md("**Guardrails pass under this setting.** The report still needs tail evidence, not only the mean."), kind="success"))

        if pC_pred.value == "tail_required":
            items.append(mo.callout(mo.md("**Correct.** Percentile and guardrail evidence decides production viability."), kind="success"))
        else:
            items.append(mo.callout(mo.md("The average is not a release gate when the deployment question is percentile or guardrail based."), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Log-Normal Tail": mo.md(f"""
            ```
            p_q = base_latency * exp(z_q * sigma)
            p99 uses z = 2.326
            p99.9 uses z = 3.09
            ```

            Current source model:

            ```
            base = {v1_12_benchmark.tail_base_ms:.1f} ms
            sigma = {pC_sigma.value:.2f}
            mean = {tail.mean_ms:.1f} ms
            p99 = {tail.p99_ms:.1f} ms
            p99.9 = {tail.p999_ms:.1f} ms
            ```

            Source helper: `mlsysbook_labs.tail_latency`. Chapter anchor:
            latency and tail latency plus single-metric fallacies.
            """)
        }))
        items.append(pC_decision)
        if pC_decision.value is None:
            items.append(mo.callout(mo.md("Checkpoint: choose the tail/guardrail decision for the report."), kind="info"))
        return mo.vstack(items)

    def build_part_d():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Part D Concept Module &middot; Fair comparison
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;Two benchmark submissions disagree. Which comparison can be reported
                    without overstating the evidence?&rdquo;
                </div>
            </div>
            """),
            mo.md(f"""
            ## Fair Comparison Requires Controlled Conditions

            {v1_12_packet["amount_line_d"]} A larger number loses if it is
            measured under easier rules or omits the guardrail evidence.
            """),
            pD_pred,
        ]
        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Commit to a comparison prediction before opening the fairness audit."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.vstack([
            mo.hstack([pD_same_workload, pD_same_warmup, pD_same_samples], justify="start"),
            mo.hstack([pD_same_hardware, pD_same_guardrail], justify="start"),
        ]))
        controls = {
            "same_workload": pD_same_workload.value,
            "same_warmup": pD_same_warmup.value,
            "same_samples": pD_same_samples.value,
            "same_hardware": pD_same_hardware.value,
            "same_guardrail": pD_same_guardrail.value,
        }
        result = v1_12_fairness_result(v1_12_packet, controls)
        labels = [label for _, label in v1_12_packet["conditions"]]
        values = [1 if controls[key] else 0 for key, _ in v1_12_packet["conditions"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker_color=[COLORS["GreenLine"] if value else COLORS["RedLine"] for value in values],
            text=["controlled" if value else "missing" for value in values],
            textposition="outside",
        ))
        fig.update_layout(height=340, yaxis=dict(title="Control present", range=[0, 1.25]), showlegend=False)
        fig.update_xaxes(tickangle=-20)
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))

        items.append(mo.Html(v1_12_metric_cards(COLORS, (
            ("Fairness Index", f"{result['index']:.0f}%", "controlled conditions / required conditions", COLORS["GreenLine"] if result["reportable"] else COLORS["RedLine"]),
            ("Accepted", result["accepted"], "comparison eligible for report", COLORS["BlueLine"]),
            ("Rejected", result["rejected"], "headline-only comparison", COLORS["OrangeLine"]),
        ))))
        items.append(mo.Html(v1_12_html_table(result["rows"], ("Comparison", "Reported win", "Run-rule state", "Reportable"))))

        if result["reportable"]:
            items.append(mo.callout(mo.md(
                f"**Comparison is reportable.** The headline-only comparison is still rejected; "
                f"the report can use {v1_12_packet['accepted_comparison']}."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Comparison withheld.** Missing evidence: " + ", ".join(result["missing"])
            ), kind="danger"))

        if pD_pred.value == "controlled":
            items.append(mo.callout(mo.md("**Correct.** Fair comparisons are governed by run rules, not headline size."), kind="success"))
        else:
            items.append(mo.callout(mo.md("The faster number is not reportable unless the workload, run rules, and guardrails match."), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Fair Comparison Index": mo.md(f"""
            ```
            fair_comparison_index = controlled_conditions / required_conditions
            ```

            Current index:

            ```
            controlled = {len(v1_12_packet["conditions"]) - len(result["missing"])}
            required = {len(v1_12_packet["conditions"])}
            index = {result["index"]:.0f}%
            ```

            Source model: Chapter Benchmark Components, System Specifications,
            Run Rules, and MLPerf reference-vs-submission validation.
            """)
        }))
        items.append(pD_decision)
        if pD_decision.value is None:
            items.append(mo.callout(mo.md("Checkpoint: choose the comparison decision for the final report."), kind="info"))
        return mo.vstack(items)

    def build_synthesis():
        a_result = v1_12_validity_result(
            v1_12_benchmark,
            v1_12_packet,
            pA_workload.value,
            pA_metric.value or "headline",
        )
        b_result = v1_12_confidence_result(
            v1_12_benchmark,
            v1_12_packet,
            pB_warmup.value,
            pB_samples.value,
            pB_jitter.value,
        )
        c_result = v1_12_guardrail_result(
            v1_12_benchmark,
            v1_12_packet,
            pC_sigma.value,
            pC_stress.value,
        )
        d_result = v1_12_fairness_result(
            v1_12_packet,
            {
                "same_workload": pD_same_workload.value,
                "same_warmup": pD_same_warmup.value,
                "same_samples": pD_same_samples.value,
                "same_hardware": pD_same_hardware.value,
                "same_guardrail": pD_same_guardrail.value,
            },
        )
        incomplete = []
        for label, widget in (
            ("Part A prediction", pA_pred),
            ("Part A checkpoint", pA_decision),
            ("Part B prediction", pB_pred),
            ("Part B checkpoint", pB_decision),
            ("Part C prediction", pC_pred),
            ("Part C checkpoint", pC_decision),
            ("Part D prediction", pD_pred),
            ("Part D checkpoint", pD_decision),
        ):
            if widget.value is None:
                incomplete.append(label)
        rows = (
            {"Report field": "Selected metric", "Evidence": a_result["selected_metric"]},
            {"Report field": "Confidence", "Evidence": "confident" if b_result["confidence_ok"] else f"not yet confident: CV {b_result['cv_pct']:.1f}%, CI {b_result['ci_pct']:.1f}%"},
            {"Report field": "Tail/guardrail", "Evidence": "blocked" if c_result["blocked"] else "passes under selected stress"},
            {"Report field": "Rejected comparison", "Evidence": d_result["rejected"]},
            {"Report field": "Claim scope", "Evidence": v1_12_packet["production_scope"]},
        )
        return mo.vstack([
            mo.md("""
            ## Synthesis: Benchmark Report

            The durable lesson is not that one metric is always best. The report is
            valid only when metric, confidence, tail/guardrail evidence, and
            comparison rules match the deployment question.
            """),
            mo.Html(v1_12_html_table(rows, ("Report field", "Evidence"))),
            mo.callout(
                mo.md(
                    "Incomplete fields: " + ", ".join(incomplete)
                    if incomplete
                    else "All concept-module predictions and checkpoints are complete."
                ),
                kind="warn" if incomplete else "success",
            ),
            mo.Html(f"""
            <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                        border-radius:8px; padding:18px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
                    Track-specific report frame
                </div>
                <div style="font-size:0.92rem; color:{COLORS['Text']}; line-height:1.7;">
                    For <strong>{v1_12_benchmark.label}</strong>, accept the benchmark claim only
                    within the scope <strong>{v1_12_packet['production_scope']}</strong>, with
                    selected metric <strong>{a_result['selected_metric']}</strong>, confidence
                    verdict <strong>{"confident" if b_result["confidence_ok"] else "not confident"}</strong>,
                    guardrail verdict <strong>{"blocked" if c_result["blocked"] else "passes"}</strong>,
                    and rejected comparison <strong>{d_result['rejected']}</strong>.
                </div>
            </div>
            """),
        ])

    tabs = mo.ui.tabs({
        "Part A: Validity": build_part_a(),
        "Part B: Confidence": build_part_b(),
        "Part C: Tail/Guardrail": build_part_c(),
        "Part D: Fairness": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


# CELL 6: LEDGER HUD

@app.cell(hide_code=True)
def _(
    COLORS,
    ledger,
    mo,
    pA_decision,
    pA_metric,
    pA_pred,
    pA_workload,
    pB_decision,
    pB_jitter,
    pB_pred,
    pB_samples,
    pB_warmup,
    pC_decision,
    pC_pred,
    pC_sigma,
    pC_stress,
    pD_decision,
    pD_pred,
    pD_same_guardrail,
    pD_same_hardware,
    pD_same_samples,
    pD_same_warmup,
    pD_same_workload,
    v1_12_benchmark,
    v1_12_confidence_result,
    v1_12_fairness_result,
    v1_12_guardrail_result,
    v1_12_packet,
    v1_12_profile,
    v1_12_validity_result,
    v1_12_variant,
):
    _a_result = v1_12_validity_result(v1_12_benchmark, v1_12_packet, pA_workload.value, pA_metric.value or "headline")
    _b_result = v1_12_confidence_result(v1_12_benchmark, v1_12_packet, pB_warmup.value, pB_samples.value, pB_jitter.value)
    _c_result = v1_12_guardrail_result(v1_12_benchmark, v1_12_packet, pC_sigma.value, pC_stress.value)
    _d_result = v1_12_fairness_result(v1_12_packet, {
        "same_workload": pD_same_workload.value,
        "same_warmup": pD_same_warmup.value,
        "same_samples": pD_same_samples.value,
        "same_hardware": pD_same_hardware.value,
        "same_guardrail": pD_same_guardrail.value,
    })
    completed = all(widget.value is not None for widget in (
        pA_pred, pA_decision, pB_pred, pB_decision, pC_pred, pC_decision, pD_pred, pD_decision
    ))
    if completed:
        ledger.save(chapter=12, design={
            "lab": "perf_bench",
            "track_id": v1_12_profile.track_id,
            "scenario_id": v1_12_variant.scenario_id,
            "hardware_ref": v1_12_benchmark.hardware_ref,
            "model_ref": v1_12_benchmark.model_ref,
            "completed": True,
            "selected_metric": _a_result["selected_metric"],
            "validity_score": round(_a_result["validity_score"], 2),
            "confidence_ok": _b_result["confidence_ok"],
            "cv_pct": round(_b_result["cv_pct"], 2),
            "ci_half_width": round(_b_result["ci_half"], 2),
            "tail_guardrail_blocked": _c_result["blocked"],
            "failure_rate": _c_result["failure_rate"],
            "fair_comparison_index": round(_d_result["index"], 2),
            "accepted_comparison": _d_result["accepted"],
            "rejected_comparison": _d_result["rejected"],
        })
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">12 &middot; Performance Benchmarking</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_12_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">{"COMPLETE" if completed else "IN PROGRESS"}</span>
    </div>
    """)
    return


# DOWNLOADABLE TRACK REPORT

@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    pA_decision,
    pA_metric,
    pA_pred,
    pA_workload,
    pB_decision,
    pB_jitter,
    pB_pred,
    pB_samples,
    pB_warmup,
    pC_decision,
    pC_pred,
    pC_sigma,
    pC_stress,
    pD_decision,
    pD_pred,
    pD_same_guardrail,
    pD_same_hardware,
    pD_same_samples,
    pD_same_warmup,
    pD_same_workload,
    report_export_panel,
    v1_12_benchmark,
    v1_12_confidence_result,
    v1_12_fairness_result,
    v1_12_guardrail_result,
    v1_12_metadata,
    v1_12_packet,
    v1_12_profile,
    v1_12_validity_result,
    v1_12_variant,
):
    a_result = v1_12_validity_result(v1_12_benchmark, v1_12_packet, pA_workload.value, pA_metric.value or "headline")
    b_result = v1_12_confidence_result(v1_12_benchmark, v1_12_packet, pB_warmup.value, pB_samples.value, pB_jitter.value)
    c_result = v1_12_guardrail_result(v1_12_benchmark, v1_12_packet, pC_sigma.value, pC_stress.value)
    d_result = v1_12_fairness_result(v1_12_packet, {
        "same_workload": pD_same_workload.value,
        "same_warmup": pD_same_warmup.value,
        "same_samples": pD_same_samples.value,
        "same_hardware": pD_same_hardware.value,
        "same_guardrail": pD_same_guardrail.value,
    })

    incomplete = []
    for label, widget in (
        ("Part A prediction", pA_pred),
        ("Part A checkpoint", pA_decision),
        ("Part B prediction", pB_pred),
        ("Part B checkpoint", pB_decision),
        ("Part C prediction", pC_pred),
        ("Part C checkpoint", pC_decision),
        ("Part D prediction", pD_pred),
        ("Part D checkpoint", pD_decision),
    ):
        if widget.value is None:
            incomplete.append(label)

    report = build_lab_report(
        v1_12_metadata,
        track=v1_12_profile.label,
        scenario=v1_12_variant.workload_summary,
        learning_objectives=(
            "Select a benchmark metric that matches the deployment workload.",
            "Use warmup, variance, and sample size to judge confidence.",
            "Reject mean-only evidence when tail or guardrail metrics fail.",
            "Report only comparisons with controlled run rules.",
        ),
        predictions={
            "validity": pA_pred.value,
            "confidence": pB_pred.value,
            "tail_guardrail": pC_pred.value,
            "fairness": pD_pred.value,
        },
        knob_settings={
            "workload_match_pct": pA_workload.value,
            "metric_choice": pA_metric.value,
            "warmup_discard": pB_warmup.value,
            "sample_count": pB_samples.value,
            "jitter_pct": pB_jitter.value,
            "tail_sigma": pC_sigma.value,
            "guardrail_stress": pC_stress.value,
            "fairness_controls": {
                "same_workload": pD_same_workload.value,
                "same_warmup": pD_same_warmup.value,
                "same_samples": pD_same_samples.value,
                "same_hardware": pD_same_hardware.value,
                "same_guardrail": pD_same_guardrail.value,
            },
        },
        binding_constraints={
            "production_question": v1_12_packet["production_question"],
            "selected_metric": a_result["selected_metric"],
            "guardrail_metric": v1_12_packet["guardrail_label"],
            "confidence_gate": b_result["confidence_ok"],
            "tail_guardrail_blocked": c_result["blocked"],
            "fair_comparison_reportable": d_result["reportable"],
        },
        evidence_summary={
            "validity_score": round(a_result["validity_score"], 3),
            "production_value": a_result["production_value"],
            "cv_pct": round(b_result["cv_pct"], 3),
            "ci_half_width": round(b_result["ci_half"], 3),
            "tail_p99_ms": round(c_result["tail"].p99_ms, 3),
            "tail_p999_ms": round(c_result["tail"].p999_ms, 3),
            "failure_rate": c_result["failure_rate"],
            "fair_comparison_index": round(d_result["index"], 3),
            "rejected_comparison": d_result["rejected"],
        },
        final_decision=(
            f"Accept the benchmark claim only for {v1_12_packet['production_scope']} "
            f"using {a_result['selected_metric']}, confidence evidence, tail/guardrail evidence, "
            f"and rejection of {d_result['rejected']}."
        ),
        big_takeaways=(
            "Benchmark validity is a workload-and-metric claim.",
            "Confidence requires warmup, repeated samples, and variance reporting.",
            "Tail and guardrail failures can be hidden by averages.",
            "Fair comparisons need controlled run rules and reportable evidence.",
        ),
        reflections={
            "report_scope": v1_12_packet["production_scope"],
            "rejected_comparison": d_result["rejected"],
            "residual_risk": "Teaching estimates still need production traces and hardware counters before launch.",
        },
        residual_risk=(
            "Notebook calculations are scenario estimates. A production benchmark still needs real traces, "
            "instrumented hardware, repeated runs, and a signed-off run-rule protocol."
        ),
        source_trace={
            "track_id": v1_12_profile.track_id,
            "scenario_id": v1_12_variant.scenario_id,
            "hardware_ref": v1_12_variant.hardware_ref,
            "model_ref": v1_12_variant.model_ref,
            "shared_helpers": ("benchmark_track_profile", "amdahl_speedup", "sustained_benchmark", "tail_latency"),
            "notebook_local_helpers": (
                "v1_12_validity_result",
                "v1_12_confidence_result",
                "v1_12_guardrail_result",
                "v1_12_fairness_result",
            ),
            "source_policy": v1_12_profile.source_policy,
        },
        result_snapshot={
            "validity": a_result,
            "confidence": b_result,
            "tail_guardrail": c_result,
            "fairness": d_result,
        },
        incomplete_fields=tuple(incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-12 report is generated locally from the selected track, MLSysIM hardware/model refs, "
                "shared benchmarking helpers, and notebook-local `v1_12_` protocol calculations."
            ),
            kind="info",
        ),
        report_export_panel(report),
    ])
    return


if __name__ == "__main__":
    app.run()
