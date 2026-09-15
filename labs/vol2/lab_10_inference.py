import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 10: The Inference Economy · MLSysBook")


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
    from mlsysim.labs.components import DecisionLog
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        MathPeek,
        batching_result,
        big_takeaways,
        build_lab_report,
        cost_crossover,
        gated_hypothesis_card,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        inference_economy_profile,
        instrumentation_console,
        paged_attention_fragmentation,
        report_export_panel,
        resolve_mlsysim_ref,
        serving_plan,
        state_capacity,
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
        MathPeek,
        apply_plotly_theme,
        batching_result,
        big_takeaways,
        build_lab_report,
        cost_crossover,
        gated_hypothesis_card,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        inference_economy_profile,
        instrumentation_console,
        ledger,
        math,
        mo,
        np,
        paged_attention_fragmentation,
        report_export_panel,
        resolve_mlsysim_ref,
        serving_plan,
        state_capacity,
    )


@app.cell
def _(get_lab_metadata):
    v2_10_metadata = get_lab_metadata("vol2/lab_10_inference.py")
    return (v2_10_metadata,)


@app.cell(hide_code=True)
def _(mo):
    v2_10_track_picker = mo.ui.dropdown(
        options={
            "⚡ TinyML Track (ARM Cortex-M55 / SRAM Limits & Duty-Cycle Budget)": "oura_ring",
            "📱 Mobile Track (Apple Silicon / Unified Memory & 5W Battery Limits)": "iphone",
            "🤖 Edge & Embodied Track (NVIDIA Jetson AGX Orin & Real-Time Loops)": "robotaxi",
            "☁️ Cloud Supercomputing Track (H100 Clusters & Continuous Batching vs KV Cache)": "cloud_fleet",
        },
        value="☁️ Cloud Supercomputing Track (H100 Clusters & Continuous Batching vs KV Cache)",
        label="Select Course / Industry Track",
    )
    v2_10_track_picker
    return (v2_10_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    inference_economy_profile,
    resolve_mlsysim_ref,
    v2_10_track_picker,
):
    v2_10_track_id = v2_10_track_picker.value
    v2_10_profile = get_track_profile(v2_10_track_id)
    v2_10_variant = get_lab_track_variant("v2_10_inference_economy", v2_10_profile.track_id)
    v2_10_hardware = resolve_mlsysim_ref(v2_10_variant.hardware_ref)
    v2_10_model = resolve_mlsysim_ref(v2_10_variant.model_ref)
    v2_10_inference = inference_economy_profile(
        v2_10_profile,
        v2_10_variant,
        v2_10_hardware,
        v2_10_model,
    )
    return v2_10_inference, v2_10_model, v2_10_profile, v2_10_variant


@app.cell
def _(COLORS, math, mo):
    def v2_10_fmt_currency(value, unit):
        if unit == "USD":
            return f"${value:,.0f}"
        if abs(value) >= 1000:
            return f"{value:,.0f} {unit}"
        if abs(value) >= 1:
            return f"{value:,.2f} {unit}"
        return f"{value:.4f} {unit}"

    def v2_10_fmt_duration_days(days):
        if not math.isfinite(days):
            return "never in the first year"
        if days < 14:
            return f"{days:.1f} days"
        if days < 365:
            return f"{days / 7:.1f} weeks"
        return f"{days / 365:.1f} years"

    def v2_10_fmt_ms(value):
        if not math.isfinite(value):
            return "not feasible"
        if abs(value) >= 1000:
            return f"{value / 1000:.2f} s"
        return f"{value:.1f} ms"

    def v2_10_crossover_bucket(days):
        if not math.isfinite(days) or days > 365:
            return "never"
        if days <= 14:
            return "days"
        if days <= 70:
            return "weeks"
        return "months"

    def v2_10_prediction_bucket_days(bucket):
        return {"days": 7.0, "weeks": 35.0, "months": 180.0, "never": 365.0}.get(bucket, 35.0)

    def v2_10_capacity_bucket(max_concurrent):
        if max_concurrent <= 1:
            return "one"
        if max_concurrent <= 4:
            return "few"
        if max_concurrent <= 32:
            return "modest"
        return "many"

    def v2_10_precision_label(precision_bytes):
        labels = {2.0: "FP16", 1.0: "INT8", 0.5: "INT4"}
        return labels.get(float(precision_bytes), f"{precision_bytes:g} bytes")

    def v2_10_schedule_label(policy):
        return {
            "no_batching": "No batching / immediate serving",
            "static": "Static batching",
            "dynamic": "Dynamic batching",
            "continuous": "Continuous batching",
        }.get(policy, str(policy))

    def v2_10_phase_label(phase):
        return {
            "prefill": "Prefill / input-pass amount",
            "decode": "Decode / output-loop amount",
        }.get(phase, str(phase))

    def v2_10_phase_amounts(profile, prefill_tokens, decode_tokens, precision_bytes):
        weight_gb = max(0.001, profile.model_params_b * precision_bytes)
        bandwidth_gbs = max(1.0, profile.memory_bandwidth_gbs)
        if profile.track_id == "cloud_fleet":
            prefill_ms = (
                max(1.0, 0.10 * profile.slo_ms)
                + 0.015 * (prefill_tokens / 1024) ** 2 * max(1.0, profile.model_params_b / 7)
            )
            tpot_ms = weight_gb / bandwidth_gbs * 1000
            decode_ms = decode_tokens * tpot_ms
            input_unit = "prompt tokens"
            output_unit = "generated tokens"
        else:
            track_scalars = {
                "iphone": (0.010, 0.022),
                "oura_ring": (0.090, 0.180),
                "robotaxi": (0.004, 0.035),
            }
            input_scale, output_scale = track_scalars.get(profile.track_id, (0.010, 0.030))
            prefill_ms = max(1.0, 0.18 * profile.slo_ms) + prefill_tokens * input_scale
            decode_ms = max(0.5, decode_tokens * output_scale + decode_tokens * weight_gb / bandwidth_gbs * 12)
            tpot_ms = decode_ms / max(1, decode_tokens)
            input_unit = "input/window units"
            output_unit = "output/decision steps"
        binding_phase = "prefill" if prefill_ms >= decode_ms else "decode"
        return {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "tpot_ms": tpot_ms,
            "weight_gb": weight_gb,
            "decode_read_gb": decode_tokens * weight_gb,
            "binding_phase": binding_phase,
            "input_unit": input_unit,
            "output_unit": output_unit,
        }

    def v2_10_policy_multiplier(track_id, policy):
        table = {
            "cloud_fleet": {"no_batching": 0.8, "static": 1.0, "dynamic": 1.8, "continuous": 3.0},
            "iphone": {"no_batching": 1.0, "static": 1.1, "dynamic": 1.5, "continuous": 1.4},
            "oura_ring": {"no_batching": 0.8, "static": 1.0, "dynamic": 1.35, "continuous": 1.05},
            "robotaxi": {"no_batching": 1.0, "static": 0.95, "dynamic": 1.15, "continuous": 1.05},
        }
        return table.get(track_id, table["cloud_fleet"]).get(policy, 1.0)

    def v2_10_track_schedule_note(track_id):
        if track_id == "robotaxi":
            return "For RoboTaxi, batching is a deadline-sensitive streaming admission decision, not a broad LLM-style batching win."
        if track_id == "oura_ring":
            return "For Oura Ring, the schedule is a duty-cycle and handoff policy; always-on scheduling can waste the battery budget."
        if track_id == "iphone":
            return "For iPhone, small dynamic batches can help background work, but interactive paths often prefer immediate local inference."
        return "For Cloud Fleet, continuous batching can help when mixed request lengths and sufficient volume justify scheduler overhead."

    def v2_10_scheduling_rows(track_id, batch_size, avg_len, max_len, fill_factor, slo_ms):
        ratio = max(1.0, max_len / max(1, avg_len))
        padding_waste = max(0.0, min(1.0, 1 - avg_len / max(1, max_len)))
        base_ms = max(1.0, 0.25 * slo_ms)
        overhead = {
            "cloud_fleet": {"no_batching": 2, "static": 12, "dynamic": 18, "continuous": 26},
            "iphone": {"no_batching": 1, "static": 8, "dynamic": 12, "continuous": 24},
            "oura_ring": {"no_batching": 8, "static": 18, "dynamic": 28, "continuous": 72},
            "robotaxi": {"no_batching": 0, "static": 10, "dynamic": 6, "continuous": 18},
        }.get(track_id, {"no_batching": 2, "static": 12, "dynamic": 18, "continuous": 26})

        rows = []
        raw_rows = (
            ("no_batching", 1.0, 0.0, base_ms * 1.1 + overhead["no_batching"]),
            ("static", float(batch_size), padding_waste, base_ms * ratio + overhead["static"] + batch_size * 0.5),
            ("dynamic", batch_size * (1 + padding_waste * 0.70) * min(0.95, fill_factor + 0.10), padding_waste * 0.45, base_ms * math.sqrt(ratio) + overhead["dynamic"]),
            ("continuous", batch_size * ratio * fill_factor, 0.0, base_ms * (1.10 if ratio > 2 else 1.25) + overhead["continuous"] + batch_size * 0.25),
        )
        for policy, throughput, waste, tail_ms in raw_rows:
            risk_pct = max(0.0, (tail_ms - slo_ms) / max(1.0, slo_ms) * 100)
            score = throughput / (1.0 + risk_pct / 35.0)
            if track_id == "robotaxi" and policy == "continuous":
                score *= 0.55
            if track_id == "robotaxi" and policy == "no_batching":
                score *= 1.35
            if track_id == "oura_ring" and policy == "continuous":
                score *= 0.55
            if track_id == "oura_ring" and policy == "dynamic":
                score *= 1.25
            rows.append({
                "policy": policy,
                "label": v2_10_schedule_label(policy),
                "throughput": throughput,
                "waste_pct": waste * 100,
                "tail_ms": tail_ms,
                "risk_pct": risk_pct,
                "score": score,
            })
        winner = max(rows, key=lambda row: row["score"])["policy"]
        return rows, winner

    def v2_10_serving_latency_ms(track_id, slo_ms, policy, utilization, oom=False):
        if oom:
            return math.inf
        overhead = {
            "no_batching": 0.04,
            "static": 0.42,
            "dynamic": 0.22,
            "continuous": 0.14,
        }.get(policy, 0.22)
        if track_id == "robotaxi" and policy in ("static", "continuous"):
            overhead += 0.30
        if track_id == "oura_ring" and policy == "continuous":
            overhead += 0.25
        return slo_ms * (0.35 + 0.25 * min(0.99, utilization) + overhead)

    def v2_10_binding_constraint(memory_ok, slo_ok, cost_ok, quality_ok):
        if not memory_ok:
            return "memory/state"
        if not slo_ok:
            return "SLO/deadline"
        if not quality_ok:
            return "quality guardrail"
        if not cost_ok:
            return "recurring cost"
        return "none - all guardrails pass"

    def v2_10_rejected_alternative(track_id, binding_constraint):
        if binding_constraint == "memory/state":
            return "Rejected single-unit FP16 long-context policy: state/cache memory does not fit beside weights."
        if binding_constraint == "SLO/deadline":
            return "Rejected minimal static policy: queueing and scheduler delay spend the p99/deadline budget."
        if binding_constraint == "quality guardrail":
            return "Rejected lowest-precision policy: quality risk is too high for this track."
        if binding_constraint == "recurring cost":
            return "Rejected overprovisioned FP16 policy: it passes latency by spending too much recurring cost or power."
        defaults = {
            "cloud_fleet": "Rejected single-unit FP16 static policy: KV/cache headroom is fragile and p99 risk is high.",
            "iphone": "Rejected always-local overprovisioned policy: battery and thermal cost are not justified.",
            "oura_ring": "Rejected always-on continuous schedule: duty cycle and battery budget are too fragile.",
            "robotaxi": "Rejected broad continuous batching policy: the safety path needs deadline-first streaming behavior.",
        }
        return defaults.get(track_id, "Rejected alternative: it fails the selected track's binding guardrail.")

    def v2_10_edge_implication(track_id, binding_constraint):
        if track_id == "cloud_fleet":
            return (
                "V2-11 asks which parts of this policy can move toward edge devices when HBM, "
                "NVLink, and always-on connectivity disappear."
            )
        if track_id == "iphone":
            return "V2-11 turns the same policy into an on-device privacy, battery, and intermittent-offload problem."
        if track_id == "oura_ring":
            return "V2-11 tightens the same policy around SRAM, duty cycle, phone handoff, and local sensing quality."
        if track_id == "robotaxi":
            return "V2-11 keeps the safety loop local and asks what fleet/cloud work can tolerate intermittent handoff."
        return f"V2-11 moves the binding amount, currently {binding_constraint}, into a smaller edge envelope."

    def v2_10_part_banner(letter, title, duration, why, color):
        return mo.Html(f"""
        <div style="margin:12px 0 16px 0;">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="background:{color}; color:white; border-radius:50%;
                            width:32px; height:32px; display:inline-flex; align-items:center;
                            justify-content:center; font-size:0.9rem; font-weight:800;
                            flex-shrink:0;">{letter}</div>
                <div style="flex:1; height:2px; background:{COLORS['Border']};"></div>
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                            text-transform:uppercase; letter-spacing:0.12em;">
                    Part {letter} &middot; {duration}</div>
            </div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['Text']};
                        margin-top:8px; line-height:1.2;">{title}</div>
            <div style="color:{COLORS['TextSec']}; font-size:0.92rem; margin-top:6px;
                        line-height:1.55; max-width:760px;">{why}</div>
        </div>
        """)

    def v2_10_stakeholder_card(persona, quote, color, background):
        return mo.Html(f"""
        <div style="border-left:4px solid {color}; background:{background};
                    border-radius:0 8px 8px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; {persona}
            </div>
            <div style="font-style:italic; font-size:1rem; color:#1e293b; line-height:1.65;">
                "{quote}"
            </div>
        </div>
        """)

    def v2_10_metric_card(label, value, subvalue="", color=None):
        _color = color or COLORS["BlueLine"]
        return mo.Html(f"""
        <div style="padding:16px 18px; border:1px solid {COLORS['Border']}; border-radius:8px;
                    min-width:150px; text-align:center; background:white;">
            <div style="color:{COLORS['TextMuted']}; font-size:0.76rem; font-weight:700;
                        text-transform:uppercase;">{label}</div>
            <div style="font-size:1.7rem; font-weight:800; color:{_color}; font-family:monospace;
                        line-height:1.35;">{value}</div>
            <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">{subvalue}</div>
        </div>
        """)

    def v2_10_reveal_card(title, prediction, actual, detail, kind="info"):
        palette = {
            "success": (COLORS["GreenLine"], COLORS["GreenLL"]),
            "warn": (COLORS["OrangeLine"], COLORS["OrangeLL"]),
            "danger": (COLORS["RedLine"], COLORS["RedLL"]),
            "info": (COLORS["BlueLine"], COLORS["BlueLL"]),
        }
        color, background = palette.get(kind, palette["info"])
        return mo.Html(f"""
        <div style="background:{background}; border:1px solid {color}; border-left:5px solid {color};
                    border-radius:8px; padding:14px 18px; margin:12px 0;">
            <div style="font-size:0.82rem; font-weight:800; color:{color};
                        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
                {title}
            </div>
            <div style="font-size:0.9rem; color:{COLORS['Text']}; line-height:1.65;">
                You predicted <strong>{prediction}</strong>. Actual: <strong>{actual}</strong>. {detail}
            </div>
        </div>
        """)

    def v2_10_failure_card(active, title, detail, recovery):
        if active:
            return mo.callout(
                mo.md(f"**{title}**  \n{detail}  \n\nRecovery path: {recovery}"),
                kind="danger",
            )
        return mo.callout(
            mo.md(f"**Recovered boundary: {title}**  \nCurrent settings pass. Boundary to watch: {detail}"),
            kind="success",
        )

    def v2_10_math_peek(title, body):
        return mo.accordion({title: mo.md(body)})

    def v2_10_table_html(headers, rows):
        _head = "".join(f"<th>{header}</th>" for header in headers)
        _rows = []
        for row in rows:
            _cells = "".join(f"<td>{cell}</td>" for cell in row)
            _rows.append(f"<tr>{_cells}</tr>")
        return mo.Html(f"""
        <div style="overflow-x:auto; margin:10px 0;">
        <table style="width:100%; border-collapse:collapse; font-size:0.86rem;">
            <thead><tr style="background:{COLORS['Surface2']}; color:{COLORS['Text']};">{_head}</tr></thead>
            <tbody>{''.join(_rows)}</tbody>
        </table>
        </div>
        <style>
            table td, table th {{
                border:1px solid {COLORS['Border']};
                padding:8px 10px;
                text-align:left;
                vertical-align:top;
            }}
        </style>
        """)

    return (
        v2_10_binding_constraint,
        v2_10_capacity_bucket,
        v2_10_crossover_bucket,
        v2_10_edge_implication,
        v2_10_failure_card,
        v2_10_fmt_currency,
        v2_10_fmt_duration_days,
        v2_10_fmt_ms,
        v2_10_metric_card,
        v2_10_part_banner,
        v2_10_phase_amounts,
        v2_10_phase_label,
        v2_10_policy_multiplier,
        v2_10_precision_label,
        v2_10_prediction_bucket_days,
        v2_10_rejected_alternative,
        v2_10_reveal_card,
        v2_10_schedule_label,
        v2_10_scheduling_rows,
        v2_10_serving_latency_ms,
        v2_10_stakeholder_card,
        v2_10_table_html,
        v2_10_track_schedule_note,
    )


@app.cell(hide_code=True)
def _(ACADEMIC_LAB_CSS, mo, v2_10_inference, v2_10_profile, v2_10_variant):
    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell">
      <div class="mlsysbook-lab-header" style="--mlsysbook-accent: #A51C30;">
        <div class="mlsysbook-meta">
          ML SYSTEMS TEXTBOOK &middot; VOLUME II &middot; CHAPTER 10 &middot; LAB 10
        </div>
        <h1 style="margin: 8px 0 4px 0; color: #0F172A; font-weight: 800; font-size: 1.85rem; letter-spacing: -0.02em;">
          The Inference Economy: Prefill, Decode, KV-Cache & Continuous Batching
        </h1>
        <p style="margin: 0 0 14px 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">
          Model recurring inference economics, navigate the state/KV-cache memory wall,
          quantify batching under sequence length variance, and balance multi-objective serving guardrails.
        </p>
        <div class="mlsysbook-chip-row" style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;">
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Track:</strong> {v2_10_profile.label}
          </span>
          <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155;">
            <strong>Stakeholder:</strong> {v2_10_variant.stakeholder}
          </span>
          <span class="mlsysbook-chip" style="background: #F8FAFC; color: #475569;">
            <strong>Hardware:</strong> {v2_10_inference.hardware_name}
          </span>
          <span class="mlsysbook-chip" style="background: #F8FAFC; color: #475569;">
            <strong>Model:</strong> {v2_10_inference.model_name}
          </span>
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Primary Metric:</strong> {v2_10_variant.primary_metric}
          </span>
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Guardrail:</strong> {v2_10_variant.guardrail_metric}
          </span>
        </div>
      </div>

      <div class="mlsysbook-panel" style="margin-bottom: 20px;">
        <h3 style="margin: 0 0 8px 0; color: #0F172A; font-size: 1.15rem;">
          System Scenario: {v2_10_profile.label} Production Serving Architecture
        </h3>
        <p style="margin: 0 0 12px 0; font-size: 0.92rem; color: #334155; line-height: 1.55;">
          {v2_10_variant.workload_summary} Production inference is a coupled queueing, memory/state, and economics optimization problem. High average throughput does not guarantee a viable deployment without strict guarantees on memory footprint, tail latency (TTFT/TPOT), quality degradation, and cost.
        </p>
        <div style="background: #F8FAFC; border-left: 4px solid #006395; padding: 12px 16px; border-radius: 4px; font-size: 0.9rem; color: #1E293B;">
          <strong>The Architectural Invariants of Inference Systems:</strong>
          <ul class="mlsysbook-list" style="margin: 8px 0 4px 0;">
            <li><strong>Prefill vs Decode Duality:</strong> Prefill is compute-bound matrix multiplication (<i>O</i>(N<sup>2</sup>) attention, high arithmetic intensity); decode is memory-bandwidth-bound autoregressive weight and KV-cache streaming (<i>O</i>(1) step arithmetic intensity).</li>
            <li><strong>The KV-Cache Memory Wall (M<sub>total</sub> = M<sub>weights</sub> + B &middot; M<sub>KV</sub>):</strong> Serving capacity is bounded by memory footprint, not FLOPs: each active context stores key-value tensors (2 &middot; L &middot; H &middot; S &middot; bytes) that quickly starve available accelerator memory.</li>
            <li><strong>Continuous vs Static Batching:</strong> Static batching incurs severe padding waste (1 - L&#772; / L<sub>max</sub>); iteration-level (continuous) batching reclaims idle execution slots but introduces scheduler overhead and tail latency variance under tight SLOs.</li>
            <li><strong>Multi-Objective Guardrail Invariant:</strong> A serving policy ships only if memory (no OOM), tail latency (p99 &le; SLO), fidelity floor, and cost/power budgets are satisfied simultaneously.</li>
          </ul>
        </div>
      </div>
    </div>
    """)
    mo.vstack([ACADEMIC_LAB_CSS, header_html])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v2_10_inference, v2_10_variant):
    mo.Html(f"""
    <div style="border-left:4px solid {COLORS['BlueLine']};
                background:white; border-radius:0 8px 8px 0;
                padding:20px 28px; margin:8px 0 16px 0;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="margin-bottom:16px;">
            <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                Learning Objectives
            </div>
            <div style="font-size:0.9rem; color:{COLORS['TextSec']}; line-height:1.7;">
                <div>1. <strong>Quantify recurring serving cost:</strong> calculate when {v2_10_inference.cost_label} exceeds the setup budget.</div>
                <div>2. <strong>Diagnose state memory:</strong> compute how {v2_10_inference.state_kind} caps concurrent sessions.</div>
                <div>3. <strong>Compare scheduling policies:</strong> explain why batching depends on variance, volume, and {v2_10_variant.guardrail_metric}.</div>
                <div>4. <strong>Design a serving policy:</strong> choose precision, scheduling, routing, and serving units that satisfy all guardrails.</div>
            </div>
        </div>
        <div style="border-top:1px solid {COLORS['Border']}; margin:0 -28px; padding:0 28px;"></div>
        <div style="display:flex; gap:32px; margin-top:16px; margin-bottom:16px; flex-wrap:wrap;">
            <div style="flex:1; min-width:220px;">
                <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                    Prerequisites
                </div>
                <div style="font-size:0.85rem; color:{COLORS['TextSec']}; line-height:1.65;">
                    Inference economics &middot; state/KV cache memory &middot; static, dynamic, and iteration-level scheduling &middot; p99/SLO guardrails
                </div>
            </div>
            <div style="flex:0 0 180px;">
                <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                    Duration
                </div>
                <div style="font-size:0.85rem; color:{COLORS['TextSec']}; line-height:1.65;">
                    <strong>45-55 min</strong><br/>
                    4 parts &middot; ~10-15 min each
                </div>
            </div>
        </div>
        <div style="border-top:1px solid {COLORS['Border']}; margin:0 -28px; padding:0 28px;"></div>
        <div style="margin-top:16px;">
            <div style="font-size:0.7rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                Core Question
            </div>
            <div style="font-size:1.05rem; color:{COLORS['Text']}; font-weight:600;
                        line-height:1.5; font-style:italic;">
                "If average throughput looks good, what still prevents {v2_10_inference.label}
                from shipping under {v2_10_variant.guardrail_metric}?"
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete before this lab:

    - The Inference at Scale chapter sections on serving economics and state/KV cache scaling.
    - The continuous batching section contrasting static padding with iteration-level scheduling.
    - The serving policy discussion on SLOs, cost/request, quality, and operational guardrails.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo, v2_10_inference):
    partA_prediction = mo.ui.radio(
        options={
            "A) Days -- recurring serving crosses almost immediately": "days",
            "B) Weeks -- serving overtakes setup during early rollout": "weeks",
            "C) Months -- setup cost dominates for a quarter or more": "months",
            "D) Never in the first year -- setup remains dominant": "never",
        },
        value=None,
        label=(
            f"{v2_10_inference.label}: when does cumulative {v2_10_inference.cost_label} "
            "exceed the one-time setup/training budget?"
        ),
    )
    partA_phase_prediction = mo.ui.radio(
        options={
            "A) Prefill / input pass -- prompt or window processing controls the first response": "prefill",
            "B) Decode / output loop -- repeated output steps control the live service": "decode",
        },
        value=None,
        label=(
            f"{v2_10_inference.label}: which amount system is more likely to bind "
            "after the request is live?"
        ),
    )
    _qps = v2_10_inference.demand_qps
    _unit_cost = v2_10_inference.cost_per_event
    _prefill_default = max(16, min(v2_10_inference.context_tokens, 8192))
    _prefill_step = max(1, _prefill_default // 16)
    _decode_default = max(1, min(512, max(8, v2_10_inference.context_tokens // 8)))
    _decode_step = max(1, _decode_default // 16)
    partA_qps = mo.ui.number(
        start=0.0,
        stop=max(_qps * 10, 1.0),
        value=_qps,
        step=max(_qps / 20, 0.01),
        label="Demand rate (events/s)",
    )
    partA_cost_per_event = mo.ui.number(
        start=0.0,
        stop=max(_unit_cost * 10, 0.001),
        value=_unit_cost,
        step=max(_unit_cost / 20, 0.000001),
        label=f"Cost per event ({v2_10_inference.cost_unit})",
    )
    partA_horizon_weeks = mo.ui.slider(
        start=1,
        stop=52,
        value=v2_10_inference.horizon_weeks,
        step=1,
        label="Planning horizon (weeks)",
    )
    partA_optimization_pct = mo.ui.slider(
        start=0,
        stop=50,
        value=10,
        step=5,
        label="Recurring optimization (%)",
    )
    partA_prefill_tokens = mo.ui.slider(
        start=max(1, _prefill_step),
        stop=max(_prefill_default * 4, _prefill_default + _prefill_step),
        value=_prefill_default,
        step=_prefill_step,
        label="Prefill/input tokens or window units",
    )
    partA_decode_tokens = mo.ui.slider(
        start=1,
        stop=max(_decode_default * 4, 16),
        value=_decode_default,
        step=_decode_step,
        label="Decode/output tokens or decision steps",
    )
    partA_checkpoint = mo.ui.radio(
        options={
            "A) Ignore inference cost until next year's budget cycle": "ignore",
            "B) Track cost/event and prioritize recurring efficiency": "optimize",
            "C) Only reduce launch setup cost": "setup_only",
            "D) Reduce demand even if quality and access suffer": "throttle",
        },
        value="B) Track cost/event and prioritize recurring efficiency",
        label="Checkpoint: what design lever should the platform owner carry forward?",
    )
    return (
        partA_checkpoint,
        partA_cost_per_event,
        partA_decode_tokens,
        partA_horizon_weeks,
        partA_optimization_pct,
        partA_phase_prediction,
        partA_prediction,
        partA_prefill_tokens,
        partA_qps,
    )


@app.cell(hide_code=True)
def _(mo, v2_10_inference, v2_10_variant):
    partB_prediction = mo.ui.radio(
        options={
            "A) Many sessions -- if weights fit, serving fits": "many",
            "B) Tens of sessions -- state is visible but not binding": "modest",
            "C) A few sessions -- state/cache is the first wall": "few",
            "D) One or zero sessions -- memory fails immediately": "one",
        },
        value=None,
        label=(
            f"{v2_10_inference.model_name} on {v2_10_inference.hardware_name}: "
            "how many concurrent sessions fit before memory fails?"
        ),
    )
    _precision_default = float(v2_10_variant.defaults.get("precision_bytes", 2.0))
    _precision_label_default = {
        2.0: "FP16 (2 bytes)",
        1.0: "INT8 (1 byte)",
        0.5: "INT4 (0.5 bytes)",
    }.get(_precision_default, "FP16 (2 bytes)")
    partB_precision = mo.ui.dropdown(
        options={"FP16 (2 bytes)": 2.0, "INT8 (1 byte)": 1.0, "INT4 (0.5 bytes)": 0.5},
        value=_precision_label_default,
        label="Weight precision",
    )
    _context = v2_10_inference.context_tokens
    _context_step = max(128, min(4096, _context // 8 if _context >= 1024 else _context))
    partB_context_tokens = mo.ui.slider(
        start=max(128, _context_step),
        stop=max(_context * 2, _context_step * 2),
        value=_context,
        step=_context_step,
        label="Context/state window",
    )
    partB_devices = mo.ui.slider(
        start=1,
        stop=max(8, v2_10_inference.default_devices_per_replica),
        value=v2_10_inference.default_devices_per_replica,
        step=1,
        label="Devices per serving unit",
    )
    partB_checkpoint = mo.ui.radio(
        options={
            "A) Lower weight precision to free live state memory": "precision",
            "B) Add compute-only accelerators": "compute",
            "C) Ignore state because the model weights fit": "ignore_state",
            "D) Spill state/cache to slow off-device memory first": "spill",
        },
        value="A) Lower weight precision to free live state memory",
        label=f"Checkpoint: what is the first capacity lever for {v2_10_inference.state_kind}?",
    )
    return (
        partB_checkpoint,
        partB_context_tokens,
        partB_devices,
        partB_precision,
        partB_prediction,
    )


@app.cell(hide_code=True)
def _(mo, v2_10_inference):
    partC_prediction = mo.ui.radio(
        options={
            "A) No batching / immediate serving": "no_batching",
            "B) Static batching": "static",
            "C) Dynamic batching": "dynamic",
            "D) Continuous batching": "continuous",
        },
        value=None,
        label="Which scheduling policy wins for this workload after variance and SLO risk are counted?",
    )
    _context = max(64, v2_10_inference.context_tokens)
    _avg_default = max(64, min(4096, _context))
    _max_default = max(_avg_default * 4, _context)
    partC_avg_len = mo.ui.slider(
        start=max(16, _avg_default // 8),
        stop=max(_avg_default * 8, 512),
        value=_avg_default,
        step=max(16, _avg_default // 16),
        label="Average request/window length",
    )
    partC_max_len = mo.ui.slider(
        start=max(64, _avg_default),
        stop=max(_max_default * 4, _avg_default * 2),
        value=_max_default,
        step=max(64, _max_default // 16),
        label="Max request/window length",
    )
    partC_batch_size = mo.ui.slider(
        start=1,
        stop=32,
        value=8,
        step=1,
        label="Batch size / live slots",
    )
    partC_fill_factor = mo.ui.slider(
        start=0.50,
        stop=0.95,
        value=v2_10_inference.batching_fill_factor,
        step=0.05,
        label="Scheduler fill factor",
    )
    partC_slo_ms = mo.ui.slider(
        start=max(5, int(v2_10_inference.slo_ms * 0.25)),
        stop=max(10, int(v2_10_inference.slo_ms * 3)),
        value=int(v2_10_inference.slo_ms),
        step=max(1, int(v2_10_inference.slo_ms / 20)),
        label="SLO / deadline (ms)",
    )
    partC_checkpoint = mo.ui.radio(
        options={
            "A) Use the same continuous scheduler for every workload": "always_continuous",
            "B) Pick the simplest policy that meets SLO and uses capacity well": "conditional",
            "C) Always avoid batching because it adds latency": "never_batch",
            "D) Maximize batch size and ignore tail latency": "max_batch",
        },
        value="B) Pick the simplest policy that meets SLO and uses capacity well",
        label="Checkpoint: what rule should be used in production?",
    )
    return (
        partC_avg_len,
        partC_batch_size,
        partC_checkpoint,
        partC_fill_factor,
        partC_max_len,
        partC_prediction,
        partC_slo_ms,
    )


@app.cell(hide_code=True)
def _(mo, v2_10_inference, v2_10_variant):
    partD_prediction = mo.ui.radio(
        options={
            "A) Overprovisioned FP16 static policy -- safe but expensive": "overprovisioned",
            "B) Single-unit FP16 long-context policy -- memory fails": "memory_fail",
            "C) Minimal static policy -- SLO/deadline fails": "slo_fail",
            "D) Guardrail-balanced policy -- best feasible option": "best_feasible",
        },
        value=None,
        label="Release review: which serving policy survives all guardrails?",
    )
    _target = v2_10_inference.demand_qps
    partD_target_qps = mo.ui.number(
        start=0.01,
        stop=max(_target * 10, 1.0),
        value=_target,
        step=max(_target / 20, 0.01),
        label="Target events/s",
    )
    _precision_default = float(v2_10_variant.defaults.get("precision_bytes", 2.0))
    _precision_label_default = {
        2.0: "FP16 (2 bytes)",
        1.0: "INT8 (1 byte)",
        0.5: "INT4 (0.5 bytes)",
    }.get(_precision_default, "FP16 (2 bytes)")
    if v2_10_inference.track_id == "cloud_fleet":
        _precision_label_default = "INT4 (0.5 bytes)"
    partD_precision = mo.ui.dropdown(
        options={"FP16 (2 bytes)": 2.0, "INT8 (1 byte)": 1.0, "INT4 (0.5 bytes)": 0.5},
        value=_precision_label_default,
        label="Release precision",
    )
    _default_policy = "Continuous batching" if v2_10_inference.track_id == "cloud_fleet" else "Dynamic batching"
    if v2_10_inference.track_id == "robotaxi":
        _default_policy = "No batching / immediate serving"
    partD_schedule = mo.ui.dropdown(
        options={
            "No batching / immediate serving": "no_batching",
            "Static batching": "static",
            "Dynamic batching": "dynamic",
            "Continuous batching": "continuous",
        },
        value=_default_policy,
        label="Scheduling policy",
    )
    partD_devices = mo.ui.slider(
        start=1,
        stop=max(8, v2_10_inference.default_devices_per_replica),
        value=v2_10_inference.default_devices_per_replica,
        step=1,
        label="Devices per serving unit",
    )
    partD_routing = mo.ui.dropdown(
        options={
            "Single route": "single_route",
            "Priority routing": "priority_routing",
            "Warm pool / handoff": "warm_pool",
        },
        value="Priority routing",
        label="Routing / handoff",
    )
    partD_checkpoint = mo.ui.radio(
        options={
            "A) Minimize replicas even if p99 rises": "replicas",
            "B) Minimize recurring cost subject to every guardrail": "guardrails",
            "C) Maximize batch size first": "batch",
            "D) Use the largest serving unit available": "largest",
        },
        value="B) Minimize recurring cost subject to every guardrail",
        label="Checkpoint: what objective should the release review enforce?",
    )
    v2_10_student_id = mo.ui.text(label="Student identifier", placeholder="Optional")
    v2_10_memo_decision = mo.ui.text_area(
        label="Inference deployment memo",
        placeholder="State serving policy, precision, devices per replica, routing strategy, binding constraint, and edge implications.",
        full_width=True,
    )
    return (
        partD_checkpoint,
        partD_devices,
        partD_precision,
        partD_prediction,
        partD_routing,
        partD_schedule,
        partD_target_qps,
        v2_10_memo_decision,
        v2_10_student_id,
    )


@app.cell(hide_code=True)
def _(
    COLORS,
    MathPeek,
    apply_plotly_theme,
    batching_result,
    big_takeaways,
    build_lab_report,
    cost_crossover,
    gated_hypothesis_card,
    go,
    instrumentation_console,
    ledger,
    math,
    mo,
    np,
    partA_checkpoint,
    partA_cost_per_event,
    partA_decode_tokens,
    partA_horizon_weeks,
    partA_optimization_pct,
    partA_phase_prediction,
    partA_prediction,
    partA_prefill_tokens,
    partA_qps,
    partB_checkpoint,
    partB_context_tokens,
    partB_devices,
    partB_precision,
    partB_prediction,
    partC_avg_len,
    partC_batch_size,
    partC_checkpoint,
    partC_fill_factor,
    partC_max_len,
    partC_prediction,
    partC_slo_ms,
    partD_checkpoint,
    partD_devices,
    partD_precision,
    partD_prediction,
    partD_routing,
    partD_schedule,
    partD_target_qps,
    paged_attention_fragmentation,
    report_export_panel,
    serving_plan,
    state_capacity,
    v2_10_binding_constraint,
    v2_10_capacity_bucket,
    v2_10_crossover_bucket,
    v2_10_edge_implication,
    v2_10_failure_card,
    v2_10_fmt_currency,
    v2_10_fmt_duration_days,
    v2_10_fmt_ms,
    v2_10_inference,
    v2_10_memo_decision,
    v2_10_metadata,
    v2_10_metric_card,
    v2_10_model,
    v2_10_part_banner,
    v2_10_phase_amounts,
    v2_10_phase_label,
    v2_10_policy_multiplier,
    v2_10_precision_label,
    v2_10_prediction_bucket_days,
    v2_10_profile,
    v2_10_rejected_alternative,
    v2_10_reveal_card,
    v2_10_schedule_label,
    v2_10_scheduling_rows,
    v2_10_serving_latency_ms,
    v2_10_stakeholder_card,
    v2_10_student_id,
    v2_10_table_html,
    v2_10_track_schedule_note,
    v2_10_variant,
):
    def build_part_a():
        _items = [
            v2_10_part_banner(
                "A",
                "Prefill/Decode Amount Split",
                "10-12 min",
                "A live request is not one amount. Prefill/input work, decode/output work, and recurring event cost bind different parts of the service.",
                COLORS["BlueLine"],
            ),
            v2_10_stakeholder_card(
                "Platform finance lead",
                f"{v2_10_variant.stakeholder}: separate the first-response input work from the recurring output loop, then tell me when the recurring side becomes larger.",
                COLORS["BlueLine"],
                COLORS["BlueLL"],
            ),
            gated_hypothesis_card(
                partA_prediction,
                title="1. Formulate Recurring Cost Crossover Hypothesis",
                subtitle=(
                    f"Scenario: The selected track ({v2_10_profile.label}) runs a continuing inference loop. "
                    f"Setup or training budget is {v2_10_fmt_currency(v2_10_inference.setup_cost, v2_10_inference.cost_unit)}, "
                    f"while each event costs {v2_10_inference.cost_per_event:g} {v2_10_inference.cost_unit} before optimization. "
                    "Predict when cumulative serving cost overtakes one-time setup cost."
                ),
            ),
            gated_hypothesis_card(
                partA_phase_prediction,
                title="2. Formulate Phase Bottleneck Hypothesis",
                subtitle=(
                    "Scenario: An inference request separates into prefill/prompt ingestion and decode/token generation. "
                    "Predict which phase amount binds the live service."
                ),
            ),
        ]
        if partA_prediction.value is None or partA_phase_prediction.value is None:
            return mo.vstack(_items)

        _default_cost = cost_crossover(
            setup_cost=v2_10_inference.setup_cost,
            demand_qps=v2_10_inference.demand_qps,
            cost_per_event=v2_10_inference.cost_per_event,
        )
        _cost = cost_crossover(
            setup_cost=v2_10_inference.setup_cost,
            demand_qps=partA_qps.value,
            cost_per_event=partA_cost_per_event.value,
            optimization_pct=partA_optimization_pct.value,
        )
        _weeks = partA_horizon_weeks.value
        _week_range = np.arange(0, _weeks + 1)
        _setup_line = [v2_10_inference.setup_cost] * len(_week_range)
        _serving_line = [week * _cost.weekly_cost for week in _week_range]
        _phase = v2_10_phase_amounts(
            v2_10_inference,
            partA_prefill_tokens.value,
            partA_decode_tokens.value,
            float(v2_10_variant.defaults.get("precision_bytes", 2.0)),
        )

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_week_range,
            y=_setup_line,
            mode="lines",
            name="Setup/training budget",
            line=dict(color=COLORS["BlueLine"], width=2.5, dash="dash"),
            hovertemplate="Week %{x}: %{y:,.0f}<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=_week_range,
            y=_serving_line,
            mode="lines",
            name="Cumulative serving cost",
            line=dict(color=COLORS["RedLine"], width=2.5),
            fill="tonexty",
            fillcolor="rgba(203,32,45,0.10)",
            hovertemplate="Week %{x}: %{y:,.0f}<extra></extra>",
        ))
        if _cost.crossover_weeks <= _weeks:
            _fig.add_vline(
                x=_cost.crossover_weeks,
                line=dict(color=COLORS["OrangeLine"], width=2, dash="dot"),
                annotation_text=f"crossover {v2_10_fmt_duration_days(_cost.crossover_days)}",
                annotation_position="top left",
            )
        _fig.update_layout(
            height=340,
            xaxis=dict(title="Weeks since deployment"),
            yaxis=dict(title=f"Cumulative cost ({v2_10_inference.cost_unit})"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=70, r=20),
        )
        apply_plotly_theme(_fig)

        _phase_fig = go.Figure()
        _phase_fig.add_trace(go.Bar(
            x=["Prefill / input pass", "Decode / output loop"],
            y=[_phase["prefill_ms"], _phase["decode_ms"]],
            marker_color=[
                COLORS["BlueLine"] if _phase["binding_phase"] == "prefill" else COLORS["BlueLL"],
                COLORS["RedLine"] if _phase["binding_phase"] == "decode" else COLORS["RedLL"],
            ],
            text=[v2_10_fmt_ms(_phase["prefill_ms"]), v2_10_fmt_ms(_phase["decode_ms"])],
            textposition="auto",
            hovertemplate="%{x}: %{y:.2f} ms<extra></extra>",
        ))
        _phase_fig.update_layout(
            height=280,
            xaxis=dict(title="Request phase amount"),
            yaxis=dict(title="Teaching latency proxy (ms)"),
            margin=dict(t=30, b=55, l=70, r=20),
        )
        apply_plotly_theme(_phase_fig)

        _actual_bucket = v2_10_crossover_bucket(_default_cost.crossover_days)
        _pred_days = v2_10_prediction_bucket_days(partA_prediction.value)
        _gap = _default_cost.crossover_days / _pred_days if _pred_days > 0 and math.isfinite(_default_cost.crossover_days) else math.inf
        _gap_text = "The category matches the default case." if partA_prediction.value == _actual_bucket else f"The default crossover is about {_gap:.1f}x your representative estimate."
        _phase_detail = (
            f"Prefill/input uses {partA_prefill_tokens.value:,} {_phase['input_unit']}; "
            f"decode/output uses {partA_decode_tokens.value:,} {_phase['output_unit']} "
            f"and about {_phase['decode_read_gb']:,.1f} GB of model-byte reads in this proxy."
        )

        _items.extend([
            instrumentation_console(
                mo.vstack([
                    mo.hstack([partA_qps, partA_cost_per_event, partA_horizon_weeks, partA_optimization_pct], justify="start", gap=1.0),
                    mo.hstack([partA_prefill_tokens, partA_decode_tokens], justify="start", gap=1.0),
                ]),
                title="Inference Economics & Phase Workload Console",
                subtitle="Manipulate demand rate, per-event cost, planning horizon, and prefill/decode balance",
            ),
            mo.hstack([
                v2_10_metric_card("Crossover", v2_10_fmt_duration_days(_cost.crossover_days), "serving > setup", COLORS["OrangeLine"]),
                v2_10_metric_card("Daily recurring", v2_10_fmt_currency(_cost.daily_cost, v2_10_inference.cost_unit), "after optimization", COLORS["RedLine"]),
                v2_10_metric_card("Annual savings", v2_10_fmt_currency(_cost.annual_savings, v2_10_inference.cost_unit), f"{partA_optimization_pct.value}% optimization", COLORS["GreenLine"]),
            ], justify="center", gap=1),
            mo.hstack([
                v2_10_metric_card("Binding phase", _phase["binding_phase"].title(), "current phase controls", COLORS["BlueLine"] if _phase["binding_phase"] == "prefill" else COLORS["RedLine"]),
                v2_10_metric_card("Prefill/input", v2_10_fmt_ms(_phase["prefill_ms"]), f"{partA_prefill_tokens.value:,} {_phase['input_unit']}", COLORS["BlueLine"]),
                v2_10_metric_card("Decode/output", v2_10_fmt_ms(_phase["decode_ms"]), f"{partA_decode_tokens.value:,} {_phase['output_unit']}", COLORS["RedLine"]),
            ], justify="center", gap=1),
            mo.ui.plotly(_fig),
            mo.ui.plotly(_phase_fig),
            v2_10_reveal_card(
                "Prediction vs actual",
                partA_prediction.value.replace("_", " "),
                f"{_actual_bucket} ({v2_10_fmt_duration_days(_default_cost.crossover_days)})",
                _gap_text,
                "success" if partA_prediction.value == _actual_bucket else "warn",
            ),
            v2_10_reveal_card(
                "Phase prediction vs actual",
                v2_10_phase_label(partA_phase_prediction.value),
                v2_10_phase_label(_phase["binding_phase"]),
                _phase_detail,
                "success" if partA_phase_prediction.value == _phase["binding_phase"] else "warn",
            ),
            MathPeek(
                "C_total(t) = C_setup + C_event * QPS * 86400 * t; t_crossover = C_setup / C_serving_day",
                {
                    "setup/training budget": v2_10_fmt_currency(v2_10_inference.setup_cost, v2_10_inference.cost_unit),
                    "demand rate": f"{partA_qps.value:g} events/s",
                    "cost per event": f"{partA_cost_per_event.value:g} {v2_10_inference.cost_unit}",
                    "crossover point": v2_10_fmt_duration_days(_cost.crossover_days),
                    "annual savings": v2_10_fmt_currency(_cost.annual_savings, v2_10_inference.cost_unit),
                    "binding phase": v2_10_phase_label(_phase["binding_phase"]),
                    "chapter source": "Inference Economics & Prefill/Decode Duality",
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part A Operating Lever</h4>
                <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #475569;">
                    What design lever should the platform owner carry forward?
                </p>
                {partA_checkpoint}
            </div>
            """),
        ])
        if partA_checkpoint.value == "optimize":
            _items.append(mo.callout(mo.md("Checkpoint saved: recurring cost/event is the lever to carry forward."), kind="success"))
        elif partA_checkpoint.value is not None:
            _items.append(mo.callout(mo.md("Cost inversion means the recurring term deserves first-class design attention."), kind="warn"))
        return mo.vstack(_items)

    def build_part_b():
        _items = [
            v2_10_part_banner(
                "B",
                "State/KV Cache Wall",
                "10-12 min",
                "Fitting model weights is not the same as fitting a live service. State grows with context and concurrency.",
                COLORS["GreenLine"],
            ),
            v2_10_stakeholder_card(
                "Inference platform lead",
                f"{v2_10_variant.stakeholder}: the model artifact loads, but live {v2_10_inference.state_kind} grows beside the weights. How many sessions fit before memory becomes the wall?",
                COLORS["GreenLine"],
                COLORS["GreenLL"],
            ),
            gated_hypothesis_card(
                partB_prediction,
                title="3. Formulate State/KV-Cache Capacity Hypothesis",
                subtitle=(
                    f"Scenario: For transformer serving, state is KV cache; for device tracks, it is runtime buffers. "
                    f"Model weights load on {v2_10_inference.hardware_name}, but live {v2_10_inference.state_kind} expands per session. "
                    "Predict how many concurrent sessions can fit before reaching the memory wall."
                ),
            ),
        ]
        if partB_prediction.value is None:
            return mo.vstack(_items)

        _state = state_capacity(
            v2_10_inference,
            v2_10_model,
            context_tokens=partB_context_tokens.value,
            precision_bytes=partB_precision.value,
            devices_per_replica=partB_devices.value,
        )
        _actual_bucket = v2_10_capacity_bucket(_state.max_concurrent)
        _oom = _state.oom
        _chart_slots = list(range(0, max(3, min(_state.max_concurrent + 4, 24))))
        _weight_vals = [_state.weight_gb] * len(_chart_slots)
        _state_vals = [slot * _state.state_per_request_gb for slot in _chart_slots]
        _total_vals = [w + s for w, s in zip(_weight_vals, _state_vals)]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=_chart_slots,
            y=_weight_vals,
            name="Model weights",
            marker_color=COLORS["BlueLine"],
            hovertemplate="%{x} sessions: %{y:.2f} GB weights<extra></extra>",
        ))
        _fig.add_trace(go.Bar(
            x=_chart_slots,
            y=_state_vals,
            name=_state.state_kind,
            marker_color=[COLORS["GreenLine"] if total <= _state.total_memory_gb else COLORS["RedLine"] for total in _total_vals],
            hovertemplate="%{x} sessions: %{y:.2f} GB state<extra></extra>",
        ))
        _fig.add_hline(
            y=_state.total_memory_gb,
            line=dict(color=COLORS["RedLine"], width=2, dash="dash"),
            annotation_text=f"memory limit {_state.total_memory_gb:.2g} GB",
            annotation_position="top right",
        )
        _fig.update_layout(
            height=320,
            barmode="stack",
            xaxis=dict(title="Concurrent sessions"),
            yaxis=dict(title="Memory (GB)", range=[0, max(_state.total_memory_gb * 1.15, max(_total_vals or [0]) * 1.10)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=55, r=20),
        )
        apply_plotly_theme(_fig)

        _detail = (
            f"{_state.weight_gb:.3g} GB weights + one {_state.state_kind} "
            f"({_state.state_per_request_gb:.3g} GB) must fit inside {_state.total_memory_gb:.3g} GB."
        )
        _gap_detail = "The bucket matches the computed capacity." if partB_prediction.value == _actual_bucket else f"Computed max concurrency is {_state.max_concurrent}, so the memory wall is {_actual_bucket}."
        _paged = paged_attention_fragmentation(
            avg_seq_len=int(partB_context_tokens.value // 4) if partB_context_tokens.value > 64 else 64,
            max_seq_len=int(partB_context_tokens.value),
            page_size_tokens=16,
        )

        _items.extend([
            instrumentation_console(
                mo.hstack([partB_precision, partB_context_tokens, partB_devices], justify="start", gap=1.0),
                title="State Memory & Accelerator Topology Console",
                subtitle="Adjust weight precision, context window length, and devices per serving replica",
            ),
            v2_10_failure_card(
                _oom,
                "OOM - state/cache wall reached",
                f"{_detail} Available memory after weights is {_state.available_gb:.3g} GB.",
                "reduce context length, lower weight precision, or add devices per serving unit",
            ),
            mo.hstack([
                v2_10_metric_card("Max concurrent", str(_state.max_concurrent), "sessions", COLORS["RedLine"] if _oom else COLORS["GreenLine"]),
                v2_10_metric_card("Weights", f"{_state.weight_gb:.2g} GB", v2_10_precision_label(partB_precision.value), COLORS["BlueLine"]),
                v2_10_metric_card("State/session", f"{_state.state_per_request_gb:.2g} GB", _state.state_kind, COLORS["OrangeLine"]),
                v2_10_metric_card("Available", f"{_state.available_gb:.2g} GB", "after weights", COLORS["GreenLine"] if _state.available_gb > 0 else COLORS["RedLine"]),
            ], justify="center", gap=1),
            mo.ui.plotly(_fig),
            v2_10_reveal_card(
                "Prediction vs actual",
                partB_prediction.value.replace("_", " "),
                f"{_actual_bucket} ({_state.max_concurrent} sessions)",
                _gap_detail,
                "success" if partB_prediction.value == _actual_bucket else "warn",
            ),
            MathPeek(
                "M_KV = 2 * layers * hidden_dim * sequence * bytes * batch; max_concurrent = floor((M_device - M_weights) / M_state)",
                {
                    "device memory": f"{_state.total_memory_gb:.2f} GB",
                    "model weights": f"{_state.weight_gb:.2f} GB ({v2_10_precision_label(partB_precision.value)})",
                    "state per session": f"{_state.state_per_request_gb:.4f} GB ({_state.state_kind})",
                    "max concurrent sessions": str(_state.max_concurrent),
                    "available headroom": f"{_state.available_gb:.2f} GB",
                    "pagedattention recovered memory": f"{_paged.memory_recovered_pct:.1f}% recovered (contiguous usable: {_paged.contiguous_usable_memory_pct:.1f}% vs paged: {_paged.paged_usable_memory_pct:.1f}%)",
                    "paged concurrency gain": f"{_paged.concurrency_gain_multiplier:.1f}x ({_paged.block_table_entries_per_req} virtual blocks/req)",
                    "chapter source": "KV-Cache Scaling & PagedAttention Virtual Memory (§10.4.3)",
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part B Memory Headroom Strategy</h4>
                <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #475569;">
                    When live state threatens capacity, what is the primary architecture lever?
                </p>
                {partB_checkpoint}
            </div>
            """),
        ])
        if partB_checkpoint.value == "precision":
            _items.append(mo.callout(mo.md("Checkpoint saved: reduce memory pressure before adding compute-only capacity."), kind="success"))
        elif partB_checkpoint.value is not None:
            _items.append(mo.callout(mo.md("The first question is whether the live state term fits beside weights; compute is secondary when memory is binding."), kind="warn"))
        return mo.vstack(_items)

    def build_part_c():
        _items = [
            v2_10_part_banner(
                "C",
                "Batching Under Variance",
                "10-12 min",
                "Continuous batching is a policy with overheads and prerequisites, not a universal production answer.",
                COLORS["OrangeLine"],
            ),
            v2_10_stakeholder_card(
                "Serving scheduler owner",
                f"{v2_10_variant.stakeholder}: requests arrive with different lengths and deadline pressure. Which scheduling policy wins after padding waste and tail risk are both counted?",
                COLORS["OrangeLine"],
                COLORS["OrangeLL"],
            ),
            mo.md(f"{v2_10_track_schedule_note(v2_10_inference.track_id)}"),
            gated_hypothesis_card(
                partC_prediction,
                title="4. Formulate Batching Policy Hypothesis",
                subtitle=(
                    f"Scenario: Requests arrive with length variance and deadline pressure. Predict which scheduling policy "
                    f"maximizes effective throughput while respecting {v2_10_variant.guardrail_metric}."
                ),
            ),
        ]
        if partC_prediction.value is None:
            return mo.vstack(_items)

        _avg = partC_avg_len.value
        _max = max(partC_max_len.value, _avg)
        _batch = partC_batch_size.value
        _fill = partC_fill_factor.value
        _slo = partC_slo_ms.value
        _batching = batching_result(avg_len=_avg, max_len=_max, batch_size=_batch, fill_factor=_fill)
        _rows, _winner = v2_10_scheduling_rows(v2_10_inference.track_id, _batch, _avg, _max, _fill, _slo)
        _winner_row = next(row for row in _rows if row["policy"] == _winner)

        _ratios = np.linspace(0.05, 1.0, 50)
        _static_tp = [_batch for _ in _ratios]
        _continuous_tp = [_batch * (1 / ratio) * _fill for ratio in _ratios]
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_ratios * 100,
            y=_static_tp,
            mode="lines",
            name="Static batching",
            line=dict(color=COLORS["BlueLine"], width=2.5),
            hovertemplate="%{x:.0f}% avg/max: %{y:.1f}<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=_ratios * 100,
            y=_continuous_tp,
            mode="lines",
            name="Continuous batching formula",
            line=dict(color=COLORS["GreenLine"], width=2.5),
            hovertemplate="%{x:.0f}% avg/max: %{y:.1f}<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=[_avg / _max * 100],
            y=[_batching.continuous_throughput],
            mode="markers",
            marker=dict(size=14, color=COLORS["OrangeLine"], symbol="diamond"),
            name="Current workload",
            hovertemplate="%{x:.0f}% avg/max: %{y:.1f}<extra></extra>",
        ))
        _fig.update_layout(
            height=310,
            xaxis=dict(title="Average length / max length (%)"),
            yaxis=dict(title="Relative throughput units"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=55, r=20),
        )
        apply_plotly_theme(_fig)

        _table_rows = []
        for row in _rows:
            _status = "SLO risk" if row["risk_pct"] > 0 else "passes"
            _color = COLORS["RedLine"] if row["risk_pct"] > 0 else COLORS["GreenLine"]
            _winner_badge = "best score" if row["policy"] == _winner else ""
            _table_rows.append((
                row["label"],
                f"{row['throughput']:.1f}",
                f"{row['waste_pct']:.1f}%",
                f"{row['tail_ms']:.1f} ms",
                f"<span style='color:{_color}; font-weight:700;'>{_status}</span> {_winner_badge}",
            ))

        _continuous_row = next(row for row in _rows if row["policy"] == "continuous")
        _continuous_overhead_failure = _continuous_row["risk_pct"] > 0
        _static_waste_failure = _batching.padding_waste_pct > 50 and _winner != "static"
        _detail = (
            f"waste = 1 - avg_len / max_len = {_batching.padding_waste_pct:.1f}%; "
            f"TP_continuous = batch_size * (max_len / avg_len) * fill_factor = {_batching.continuous_throughput:.1f}."
        )
        _prediction_detail = "That policy wins for the current settings." if partC_prediction.value == _winner else "The score changes because throughput, padding waste, scheduler overhead, and deadline risk interact."

        _items.extend([
            instrumentation_console(
                mo.hstack([
                    mo.vstack([partC_avg_len, partC_max_len]),
                    mo.vstack([partC_batch_size, partC_fill_factor, partC_slo_ms]),
                ], justify="start", gap=1.0),
                title="Batch Scheduling & Variance Console",
                subtitle="Tune request length variance, batch size, fill factor, and latency SLO",
            ),
            v2_10_failure_card(
                _continuous_overhead_failure,
                "SLO/deadline violation - scheduler overhead dominates",
                f"Continuous batching estimates {_continuous_row['tail_ms']:.1f} ms against a {_slo:g} ms guardrail.",
                "use dynamic or immediate scheduling, reduce batch size, or relax the deadline only if product requirements allow",
            ),
            v2_10_failure_card(
                _static_waste_failure,
                "Padding waste - static batch leaves capacity idle",
                f"Static padding waste is {_batching.padding_waste_pct:.1f}% for avg={_avg:,} and max={_max:,}.",
                "bucket by length, use dynamic batching, or use continuous batching when volume justifies it",
            ),
            mo.hstack([
                v2_10_metric_card("Winner", v2_10_schedule_label(_winner), "current workload", COLORS["GreenLine"]),
                v2_10_metric_card("Speedup", f"{_batching.speedup:.1f}x", "continuous formula vs static", COLORS["OrangeLine"]),
                v2_10_metric_card("Static waste", f"{_batching.padding_waste_pct:.1f}%", "1 - avg/max", COLORS["RedLine"] if _batching.padding_waste_pct > 50 else COLORS["BlueLine"]),
                v2_10_metric_card("Tail estimate", f"{_winner_row['tail_ms']:.1f} ms", f"{_slo:g} ms SLO", COLORS["GreenLine"] if _winner_row["risk_pct"] == 0 else COLORS["RedLine"]),
            ], justify="center", gap=1),
            mo.ui.plotly(_fig),
            v2_10_table_html(
                ("Policy", "Throughput", "Waste", "Tail estimate", "Status"),
                _table_rows,
            ),
            v2_10_reveal_card(
                "Prediction vs actual",
                v2_10_schedule_label(partC_prediction.value),
                v2_10_schedule_label(_winner),
                _prediction_detail,
                "success" if partC_prediction.value == _winner else "warn",
            ),
            MathPeek(
                "padding_waste = 1 - avg_len / max_len; TP_continuous = batch_size * (max_len / avg_len) * fill_factor",
                {
                    "average length": f"{_avg:,}",
                    "maximum length": f"{_max:,}",
                    "static padding waste": f"{_batching.padding_waste_pct:.1f}%",
                    "continuous speedup formula": f"{_batching.speedup:.1f}x",
                    "policy winner": v2_10_schedule_label(_winner),
                    "target SLO": f"{_slo:g} ms",
                    "chapter source": "Continuous vs Static Batching Under Skew",
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part C Production Scheduling Rule</h4>
                <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #475569;">
                    What operational rule governs continuous batching adoption?
                </p>
                {partC_checkpoint}
            </div>
            """),
        ])
        if partC_checkpoint.value == "conditional":
            _items.append(mo.callout(mo.md("Checkpoint saved: batching policy is workload- and SLO-dependent."), kind="success"))
        elif partC_checkpoint.value is not None:
            _items.append(mo.callout(mo.md("The production rule is conditional: use the simplest policy that satisfies SLO and recovers enough capacity."), kind="warn"))
        return mo.vstack(_items)

    def build_part_d():
        _items = [
            v2_10_part_banner(
                "D",
                "Serving Design Challenge",
                "12-15 min",
                "A serving policy ships only when memory, latency, quality, power or cost, and routing guardrails pass together.",
                COLORS["RedLine"],
            ),
            v2_10_stakeholder_card(
                "Release review chair",
                f"{v2_10_variant.stakeholder}: choose one policy for {v2_10_inference.label}. It must meet target demand, {v2_10_variant.guardrail_metric}, memory, quality, and recurring-cost guardrails.",
                COLORS["RedLine"],
                COLORS["RedLL"],
            ),
            gated_hypothesis_card(
                partD_prediction,
                title="5. Formulate Serving Release Policy Hypothesis",
                subtitle=(
                    f"Scenario: Choose a release policy for {v2_10_inference.label}. It must meet target QPS, "
                    f"{v2_10_variant.guardrail_metric}, memory headroom, fidelity floors, and cost budgets simultaneously. "
                    "Predict which policy profile satisfies all guardrails."
                ),
            ),
        ]
        if partD_prediction.value is None:
            return mo.vstack(_items)

        _target = partD_target_qps.value
        _precision = partD_precision.value
        _policy = partD_schedule.value
        _devices = partD_devices.value
        _multiplier = v2_10_policy_multiplier(v2_10_inference.track_id, _policy)
        _plan = serving_plan(
            v2_10_inference,
            v2_10_model,
            target_qps=_target,
            precision_bytes=_precision,
            batching_multiplier=_multiplier,
            devices_per_replica=_devices,
            context_tokens=v2_10_inference.context_tokens,
        )
        _state = state_capacity(
            v2_10_inference,
            v2_10_model,
            context_tokens=v2_10_inference.context_tokens,
            precision_bytes=_precision,
            devices_per_replica=_devices,
        )
        _capacity = max(1e-9, _plan.replicas_needed * _plan.per_replica_qps)
        _utilization = min(0.999, _target / _capacity)
        _p99_ms = v2_10_serving_latency_ms(v2_10_inference.track_id, v2_10_inference.slo_ms, _policy, _utilization, _plan.oom)
        _memory_ok = not _plan.oom
        _slo_ok = _p99_ms <= v2_10_inference.slo_ms
        _quality_ok = not (_precision == 0.5 and v2_10_inference.track_id == "robotaxi")
        _cost_ok = _plan.daily_cost <= _plan.baseline_daily_cost * 1.05
        _binding = v2_10_binding_constraint(_memory_ok, _slo_ok, _cost_ok, _quality_ok)
        _slo_margin = v2_10_inference.slo_ms - _p99_ms if math.isfinite(_p99_ms) else -math.inf

        def _candidate(label, precision, policy, devices, note):
            multiplier = v2_10_policy_multiplier(v2_10_inference.track_id, policy)
            plan = serving_plan(
                v2_10_inference,
                v2_10_model,
                target_qps=_target,
                precision_bytes=precision,
                batching_multiplier=multiplier,
                devices_per_replica=devices,
                context_tokens=v2_10_inference.context_tokens,
            )
            capacity = max(1e-9, plan.replicas_needed * plan.per_replica_qps)
            util = min(0.999, _target / capacity)
            p99 = v2_10_serving_latency_ms(v2_10_inference.track_id, v2_10_inference.slo_ms, policy, util, plan.oom)
            memory_ok = not plan.oom
            slo_ok = p99 <= v2_10_inference.slo_ms
            quality_ok = not (precision == 0.5 and v2_10_inference.track_id == "robotaxi")
            cost_ok = plan.daily_cost <= plan.baseline_daily_cost * 1.05
            binding = v2_10_binding_constraint(memory_ok, slo_ok, cost_ok, quality_ok)
            return {
                "label": label,
                "precision": precision,
                "policy": policy,
                "devices": devices,
                "plan": plan,
                "p99": p99,
                "memory_ok": memory_ok,
                "slo_ok": slo_ok,
                "quality_ok": quality_ok,
                "cost_ok": cost_ok,
                "binding": binding,
                "note": note,
            }

        _candidates = [
            _candidate("A) Overprovisioned FP16 static", 2.0, "static", max(v2_10_inference.default_devices_per_replica * 2, 2), "usually safe but pays for excess capacity"),
            _candidate("B) Single-unit FP16 long-context", 2.0, "static", 1, "primary cloud-fleet memory failure case"),
            _candidate("C) Minimal static schedule", 0.5, "static", max(1, v2_10_inference.default_devices_per_replica), "cheap shape that risks deadline or quality"),
            _candidate("D) Guardrail-balanced policy", _precision, _policy, _devices, "your selected release knobs"),
        ]
        _candidate_rows = []
        for candidate in _candidates:
            status = "PASS" if all((candidate["memory_ok"], candidate["slo_ok"], candidate["quality_ok"], candidate["cost_ok"])) else f"BLOCKED: {candidate['binding']}"
            color = COLORS["GreenLine"] if status == "PASS" else COLORS["RedLine"]
            p99_text = "OOM" if not math.isfinite(candidate["p99"]) else f"{candidate['p99']:.1f} ms"
            _candidate_rows.append((
                candidate["label"],
                f"{v2_10_precision_label(candidate['precision'])}, {v2_10_schedule_label(candidate['policy'])}, {candidate['devices']} device(s)",
                f"{candidate['plan'].replicas_needed}",
                v2_10_fmt_currency(candidate["plan"].daily_cost, v2_10_inference.cost_unit),
                p99_text,
                f"<span style='color:{color}; font-weight:800;'>{status}</span><br><span style='color:{COLORS['TextMuted']};'>{candidate['note']}</span>",
            ))

        _costs = [candidate["plan"].daily_cost for candidate in _candidates]
        _names = [candidate["label"].split(") ", 1)[0] for candidate in _candidates]
        _colors = [COLORS["OrangeLine"], COLORS["RedLine"], COLORS["RedLine"], COLORS["BlueLine"]]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=_names,
            y=[cost / 1000 for cost in _costs],
            marker_color=_colors,
            text=[f"{cost / 1000:.1f}K" for cost in _costs],
            textposition="auto",
            hovertemplate="Policy %{x}: %{y:.2f}K/day<extra></extra>",
        ))
        _fig.update_layout(
            height=310,
            xaxis=dict(title="Policy candidate"),
            yaxis=dict(title=f"Daily cost (K {v2_10_inference.cost_unit})"),
            margin=dict(t=30, b=55, l=70, r=20),
        )
        apply_plotly_theme(_fig)

        _prediction_actual = "best feasible" if all((_memory_ok, _slo_ok, _quality_ok, _cost_ok)) else f"blocked by {_binding}"
        _prediction_detail = "The release policy must pass every guardrail, not just cost or throughput."

        _items.extend([
            instrumentation_console(
                mo.hstack([
                    mo.vstack([partD_target_qps, partD_precision]),
                    mo.vstack([partD_schedule, partD_devices, partD_routing]),
                ], justify="start", gap=1.0),
                title="Multi-Objective Serving Release Console",
                subtitle="Configure precision, batching engine, replica topology, and routing policy",
            ),
            v2_10_failure_card(
                not _memory_ok,
                "OOM - memory guardrail failed",
                f"Weights plus one {_state.state_kind} require more than {_state.total_memory_gb:.3g} GB.",
                "lower precision, add devices per serving unit, shorten context, or pick a smaller model",
            ),
            v2_10_failure_card(
                not _slo_ok,
                "SLA/deadline violation",
                f"Estimated p99 is {'OOM' if not math.isfinite(_p99_ms) else f'{_p99_ms:.1f} ms'} against {v2_10_inference.slo_ms:g} ms.",
                "reduce utilization, change scheduling, add reserve capacity, or route priority traffic differently",
            ),
            v2_10_failure_card(
                not _cost_ok,
                "Cost/power guardrail violation",
                f"Daily recurring cost is {v2_10_fmt_currency(_plan.daily_cost, v2_10_inference.cost_unit)} versus baseline {v2_10_fmt_currency(_plan.baseline_daily_cost, v2_10_inference.cost_unit)}.",
                "improve per-unit throughput, right-size replicas, or revisit the policy target",
            ),
            mo.hstack([
                v2_10_metric_card("Replicas", str(_plan.replicas_needed), f"{_plan.total_devices} total devices", COLORS["BlueLine"]),
                v2_10_metric_card("Daily cost", v2_10_fmt_currency(_plan.daily_cost, v2_10_inference.cost_unit), f"{_plan.savings_pct:.0f}% vs baseline", COLORS["GreenLine"] if _cost_ok else COLORS["RedLine"]),
                v2_10_metric_card("SLO margin", "OOM" if not math.isfinite(_slo_margin) else f"{_slo_margin:.1f} ms", f"{v2_10_inference.slo_ms:g} ms guardrail", COLORS["GreenLine"] if _slo_ok else COLORS["RedLine"]),
                v2_10_metric_card("Binding", _binding, "release review", COLORS["GreenLine"] if _binding.startswith("none") else COLORS["RedLine"]),
            ], justify="center", gap=1),
            mo.ui.plotly(_fig),
            v2_10_table_html(
                ("Candidate", "Config", "Replicas", "Cost/day", "p99 estimate", "Guardrail status"),
                _candidate_rows,
            ),
            v2_10_reveal_card(
                "Prediction vs release review",
                partD_prediction.value.replace("_", " "),
                _prediction_actual,
                _prediction_detail,
                "success" if partD_prediction.value == "best_feasible" and _binding.startswith("none") else "warn",
            ),
            MathPeek(
                "feasible = memory_ok and slo_ok and quality_ok and cost_ok; replicas = ceil(target_qps / (max_batch * base_qps * multiplier))",
                {
                    "target QPS": f"{_target:g} req/s",
                    "selected precision": v2_10_precision_label(_precision),
                    "scheduling engine": v2_10_schedule_label(_policy),
                    "replicas needed": str(_plan.replicas_needed),
                    "total accelerators": str(_plan.total_devices),
                    "daily cost": v2_10_fmt_currency(_plan.daily_cost, v2_10_inference.cost_unit),
                    "estimated p99": f"{_p99_ms:.1f} ms" if math.isfinite(_p99_ms) else "OOM",
                    "binding constraint": _binding,
                    "chapter source": "End-to-End Serving Policy Feasibility",
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part D Production Release Objective</h4>
                <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #475569;">
                    What design objective governs production serving release?
                </p>
                {partD_checkpoint}
            </div>
            """),
        ])
        if partD_checkpoint.value == "guardrails":
            _items.append(mo.callout(mo.md("Checkpoint saved: the objective is constrained cost minimization under all guardrails."), kind="success"))
        elif partD_checkpoint.value is not None:
            _items.append(mo.callout(mo.md("A release design is feasible only when every guardrail passes simultaneously."), kind="warn"))
        return mo.vstack(_items)

    def build_synthesis():
        _cost = cost_crossover(
            setup_cost=v2_10_inference.setup_cost,
            demand_qps=partA_qps.value,
            cost_per_event=partA_cost_per_event.value,
            optimization_pct=partA_optimization_pct.value,
        )
        _phase = v2_10_phase_amounts(
            v2_10_inference,
            partA_prefill_tokens.value,
            partA_decode_tokens.value,
            float(v2_10_variant.defaults.get("precision_bytes", 2.0)),
        )
        _state = state_capacity(
            v2_10_inference,
            v2_10_model,
            context_tokens=partB_context_tokens.value,
            precision_bytes=partB_precision.value,
            devices_per_replica=partB_devices.value,
        )
        _max_len = max(partC_max_len.value, partC_avg_len.value)
        _batching = batching_result(
            avg_len=partC_avg_len.value,
            max_len=_max_len,
            batch_size=partC_batch_size.value,
            fill_factor=partC_fill_factor.value,
        )
        _rows, _winner = v2_10_scheduling_rows(
            v2_10_inference.track_id,
            partC_batch_size.value,
            partC_avg_len.value,
            _max_len,
            partC_fill_factor.value,
            partC_slo_ms.value,
        )
        _plan = serving_plan(
            v2_10_inference,
            v2_10_model,
            target_qps=partD_target_qps.value,
            precision_bytes=partD_precision.value,
            batching_multiplier=v2_10_policy_multiplier(v2_10_inference.track_id, partD_schedule.value),
            devices_per_replica=partD_devices.value,
            context_tokens=v2_10_inference.context_tokens,
        )
        _capacity = max(1e-9, _plan.replicas_needed * _plan.per_replica_qps)
        _utilization = min(0.999, partD_target_qps.value / _capacity)
        _p99_ms = v2_10_serving_latency_ms(v2_10_inference.track_id, v2_10_inference.slo_ms, partD_schedule.value, _utilization, _plan.oom)
        _memory_ok = not _plan.oom
        _slo_ok = _p99_ms <= v2_10_inference.slo_ms
        _quality_ok = not (partD_precision.value == 0.5 and v2_10_inference.track_id == "robotaxi")
        _cost_ok = _plan.daily_cost <= _plan.baseline_daily_cost * 1.05
        _binding = v2_10_binding_constraint(_memory_ok, _slo_ok, _cost_ok, _quality_ok)
        _slo_margin = v2_10_inference.slo_ms - _p99_ms if math.isfinite(_p99_ms) else -math.inf
        _rejected = v2_10_rejected_alternative(v2_10_inference.track_id, _binding)
        _edge_implication = v2_10_edge_implication(v2_10_inference.track_id, _binding)

        _custom_note = v2_10_memo_decision.value if isinstance(v2_10_memo_decision.value, str) and v2_10_memo_decision.value.strip() else None
        _decision_text = _custom_note or (
            f"Deploy {v2_10_precision_label(partD_precision.value)} with {v2_10_schedule_label(partD_schedule.value)} "
            f"across {_plan.replicas_needed} serving units ({_plan.total_devices} total devices) using {partD_routing.value}. "
            f"Binding amount: {_binding}; daily cost {v2_10_fmt_currency(_plan.daily_cost, v2_10_inference.cost_unit)}. "
            f"Reject: {_rejected}. {v2_10_variant.guardrail_metric} margin: {'OOM' if not math.isfinite(_slo_margin) else f'{_slo_margin:.1f} ms'}."
        )

        _incomplete = []
        for label, widget in (
            ("Part A cost inversion prediction", partA_prediction),
            ("Part A prefill/decode phase prediction", partA_phase_prediction),
            ("Part A checkpoint", partA_checkpoint),
            ("Part B state/KV prediction", partB_prediction),
            ("Part B checkpoint", partB_checkpoint),
            ("Part C scheduling prediction", partC_prediction),
            ("Part C checkpoint", partC_checkpoint),
            ("Part D policy prediction", partD_prediction),
            ("Part D checkpoint", partD_checkpoint),
        ):
            if widget.value is None:
                _incomplete.append(label)

        _complete = len(_incomplete) == 0
        _residual_risk = (
            "Validate teaching estimates with production traces: real token distributions, "
            "thermal/power replay, p99 load tests, quality canaries, routing behavior, and current pricing."
        )
        _ledger_design = {
            "track_id": v2_10_profile.track_id,
            "scenario_id": v2_10_variant.scenario_id,
            "partA_predicted_crossover": partA_prediction.value or "no_selection",
            "partA_actual_crossover_days": round(_cost.crossover_days, 3),
            "partA_phase_prediction": partA_phase_prediction.value or "no_selection",
            "partA_binding_phase": _phase["binding_phase"],
            "partA_prefill_tokens": partA_prefill_tokens.value,
            "partA_decode_tokens": partA_decode_tokens.value,
            "partB_max_concurrent": _state.max_concurrent,
            "partB_context_tokens": partB_context_tokens.value,
            "partB_precision_bytes": float(partB_precision.value),
            "partB_oom": bool(_state.oom),
            "partC_scheduling_policy": partC_prediction.value or "no_selection",
            "partC_speedup": round(_batching.speedup, 3),
            "partC_padding_waste_pct": round(_batching.padding_waste_pct, 3),
            "partD_selected_precision": v2_10_precision_label(partD_precision.value),
            "partD_selected_policy": partD_schedule.value,
            "partD_selected_routing": partD_routing.value,
            "partD_replicas_needed": _plan.replicas_needed,
            "partD_cost_per_day": round(_plan.daily_cost, 6),
            "partD_binding_constraint": _binding,
            "partD_rejected_alternative": _rejected,
            "v2_11_edge_implication": _edge_implication,
            "decision": _decision_text,
        }
        if _complete:
            ledger.save(chapter=10, design=_ledger_design)

        _report = build_lab_report(
            v2_10_metadata,
            student_id=v2_10_student_id.value or "",
            track=v2_10_profile.label,
            scenario=v2_10_variant.workload_summary,
            learning_objectives=(
                "Separate prefill/input work from decode/output work before sizing the service.",
                "Quantify when recurring serving cost exceeds setup or training cost.",
                "Compute the selected track's state/KV or runtime-state memory wall.",
                "Compare scheduling policies under request variance and SLO pressure.",
                "Choose a release policy under memory, latency, quality, and cost guardrails.",
            ),
            predictions={
                "partA_predicted_crossover": partA_prediction.value,
                "partA_predicted_phase": partA_phase_prediction.value,
                "partB_predicted_capacity": partB_prediction.value,
                "partC_predicted_policy": partC_prediction.value,
                "partD_predicted_release_policy": partD_prediction.value,
            },
            knob_settings={
                "demand_qps": partA_qps.value,
                "cost_per_event": partA_cost_per_event.value,
                "optimization_pct": partA_optimization_pct.value,
                "prefill_tokens": partA_prefill_tokens.value,
                "decode_tokens": partA_decode_tokens.value,
                "context_tokens": partB_context_tokens.value,
                "precision_bytes_partB": partB_precision.value,
                "avg_len": partC_avg_len.value,
                "max_len": _max_len,
                "batch_size": partC_batch_size.value,
                "fill_factor": partC_fill_factor.value,
                "release_precision": v2_10_precision_label(partD_precision.value),
                "release_policy": v2_10_schedule_label(partD_schedule.value),
                "release_devices": partD_devices.value,
                "release_routing": partD_routing.value,
            },
            evidence_summary={
                "actual_crossover_days": round(_cost.crossover_days, 3),
                "annual_savings": v2_10_fmt_currency(_cost.annual_savings, v2_10_inference.cost_unit),
                "binding_phase": v2_10_phase_label(_phase["binding_phase"]),
                "prefill_ms": round(_phase["prefill_ms"], 3),
                "decode_ms": round(_phase["decode_ms"], 3),
                "max_concurrent": _state.max_concurrent,
                "state_per_request_gb": round(_state.state_per_request_gb, 6),
                "padding_waste_pct": round(_batching.padding_waste_pct, 3),
                "continuous_speedup": round(_batching.speedup, 3),
                "replicas_needed": _plan.replicas_needed,
                "daily_cost": v2_10_fmt_currency(_plan.daily_cost, v2_10_inference.cost_unit),
                "slo_margin_ms": round(_slo_margin, 3) if math.isfinite(_slo_margin) else "not_feasible",
                "binding_constraint": _binding,
                "rejected_alternative": _rejected,
            },
            final_decision={
                "selected_precision": v2_10_precision_label(partD_precision.value),
                "selected_policy": partD_schedule.value,
                "selected_routing": partD_routing.value,
                "replicas_needed": _plan.replicas_needed,
                "cost_per_day": round(_plan.daily_cost, 6),
                "binding_constraint": _binding,
                "rejected_alternative": _rejected,
                "v2_11_edge_implication": _edge_implication,
            },
            big_takeaways=(
                "Prefill/input and decode/output are different amount systems: compute-bound vs memory-bandwidth bound.",
                "Recurring serving cost compounds quickly: cost-per-event optimization soon overtakes setup/training investments.",
                "State/KV cache memory caps concurrency: weights are only a static baseline; dynamic context state dominates memory at load.",
                "Batching policy is workload- and SLO-dependent: continuous batching excels under variance but pays scheduler overhead under strict latency budgets.",
                "A serving policy ships only if every guardrail passes: memory, latency, quality floor, and cost must all clear simultaneously.",
            ),
            reflections={
                "cost_inversion_checkpoint": partA_checkpoint.value,
                "phase_amount_prediction": partA_phase_prediction.value,
                "state_wall_checkpoint": partB_checkpoint.value,
                "batching_checkpoint": partC_checkpoint.value,
                "release_checkpoint": partD_checkpoint.value,
                "student_justification": str(v2_10_memo_decision.value or ""),
            },
            residual_risk=_residual_risk,
            source_trace={
                "track_id": v2_10_profile.track_id,
                "scenario_id": v2_10_variant.scenario_id,
                "hardware_ref": v2_10_variant.hardware_ref,
                "model_ref": v2_10_variant.model_ref,
                "shared_helper": "mlsysbook_labs.inference",
                "helper_apis": ("cost_crossover", "state_capacity", "batching_result", "serving_plan"),
                "source_policy": v2_10_profile.source_policy,
            },
            result_snapshot={
                "cost_crossover": _cost,
                "state_capacity": _state,
                "batching": _batching,
                "serving_plan": _plan,
            },
            incomplete_fields=tuple(_incomplete),
        )

        return mo.vstack([
            mo.md("## Synthesis - Inference Architecture Memo"),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #1F407A; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">STUDENT MEMO & REFLECTIONS</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Inference Architecture Deployment Memo</h4>
                {v2_10_student_id}
                <div style="margin-top: 12px;">{v2_10_memo_decision}</div>
            </div>
            """),
            v2_10_table_html(
                ("Memo Field", "Selected Evidence & Architecture Decision"),
                [
                    ("Serving Policy", f"{v2_10_precision_label(partD_precision.value)} with {v2_10_schedule_label(partD_schedule.value)}, {partD_devices.value} dev/replica, {partD_routing.value}"),
                    ("Cluster Footprint", f"{_plan.replicas_needed} serving unit(s) ({_plan.total_devices} devices total), {v2_10_fmt_currency(_plan.daily_cost, v2_10_inference.cost_unit)}/day"),
                    ("Latency & Margin", f"p99 estimate: {'OOM' if not math.isfinite(_p99_ms) else f'{_p99_ms:.1f} ms'} (guardrail: {v2_10_inference.slo_ms:g} ms, margin: {'N/A' if not math.isfinite(_slo_margin) else f'{_slo_margin:.1f} ms'})"),
                    ("Binding Constraint", _binding),
                    ("Rejected Alternative", _rejected),
                    ("V2-11 Edge Implication", _edge_implication),
                ],
            ),
            mo.callout(mo.md(_decision_text), kind="success" if all((_memory_ok, _slo_ok, _quality_ok, _cost_ok)) else "warn"),
            big_takeaways([
                "Prefill/input and decode/output are different amount systems: compute-bound vs memory-bandwidth bound.",
                "Recurring serving cost compounds quickly: cost-per-event optimization soon overtakes setup/training investments.",
                "State/KV cache memory caps concurrency: weights are only a static baseline; dynamic context state dominates memory at load.",
                "Batching policy is workload- and SLO-dependent: continuous batching excels under variance but pays scheduler overhead under strict latency budgets.",
                "A serving policy ships only if every guardrail passes: memory, latency, quality floor, and cost must all clear simultaneously.",
            ]),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #A51C30; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">FINAL VERIFICATION & SIGN-OFF</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Lead Architect Authorization</h4>
                <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #475569;">
                    Confirm your serving architecture policy and export your engineering audit record.
                </p>
            </div>
            """),
            report_export_panel(_report),
        ])

    tabs = mo.ui.tabs({
        "Part A -- Cost Inversion": build_part_a(),
        "Part B -- State/KV Wall": build_part_b(),
        "Part C -- Batching Variance": build_part_c(),
        "Part D -- Design Challenge": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(mo, v2_10_metadata, v2_10_profile):
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">{v2_10_metadata.lab_id}</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v2_10_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
