import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


# -----------------------------------------------------------------------------
# LAB V2-09: THE OPTIMIZATION TRAP
#
# Chapter invariant: optimization is measurement-driven. The same Part A/B/C/D
# concept sequence is realized by each track through different thresholds,
# evidence emphasis, failure modes, and report framing.
# -----------------------------------------------------------------------------


@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    import html as html_lib
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
        LAB_CSS,
        apply_plotly_theme,
        build_lab_report,
        get_lab_metadata,
        get_track_profile,
        go,
        html_lib,
        ledger,
        math,
        mo,
        report_export_panel,
        source_trace,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v2_09_lab_path = "vol2/lab_09_perf_engineering.py"
    v2_09_chapter = 9
    v2_09_metadata = get_lab_metadata(v2_09_lab_path)
    return v2_09_chapter, v2_09_lab_path, v2_09_metadata


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_09_track_picker = track_selector(default=_default_track)
    v2_09_track_picker
    return (v2_09_track_picker,)


@app.cell
def _(get_track_profile, v2_09_track_picker):
    v2_09_track_id = v2_09_track_picker.value
    v2_09_profile = get_track_profile(v2_09_track_id)
    return v2_09_profile, v2_09_track_id


@app.cell
def _(COLORS, html_lib, math, mo):
    def v2_09_pct(value, digits=1):
        return f"{value * 100:.{digits}f}%"

    def v2_09_ms(value, digits=1):
        if not math.isfinite(value):
            return "unbounded"
        return f"{value:.{digits}f} ms"

    def v2_09_num(value, digits=1):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{digits}f}"

    def v2_09_money(value):
        if value >= 1000:
            return f"${value:,.0f}"
        if value >= 10:
            return f"${value:,.1f}"
        return f"${value:,.2f}"

    def v2_09_optimization_label(lever):
        return {
            "none": "No local optimization",
            "data_path": "Reduce data movement",
            "compute_path": "Improve compute path",
            "overhead_path": "Reduce launch/coordination overhead",
            "capacity_path": "Add serving capacity",
            "bottleneck_fix": "Fix the measured bottleneck",
        }.get(lever, str(lever))

    def v2_09_bottleneck_label(bottleneck):
        return {
            "data": "Data movement / bandwidth",
            "compute": "Compute path",
            "overhead": "Launch or coordination overhead",
            "capacity": "Fleet capacity",
        }.get(bottleneck, str(bottleneck))

    def v2_09_candidate_label(candidate_id):
        return {
            "targeted": "Target measured bottleneck",
            "scale": "Buy more serving units",
            "precision": "Aggressive precision/cache compression",
            "algorithmic": "Speculative or algorithmic path",
        }.get(candidate_id, str(candidate_id))

    def v2_09_track_packet(profile):
        shared = {
            "track_id": profile.track_id,
            "label": profile.label,
            "stakeholder": profile.stakeholder,
            "track_source": profile.source_policy,
            "hardware_ref": profile.hardware_ref,
            "system_ref": profile.system_ref or "track-local deployment envelope",
            "quality_budget_pct": 2.0,
            "regression_budget_pct": 4.0,
            "reality_tax_pct": 12.0,
            "headroom_target_pct": 25.0,
            "cost_ceiling_multiplier": 1.35,
        }
        track_configs = {
            "iphone": {
                "scenario_id": "iphone_sustained_local_assistant",
                "workload": "sustained on-device assistant requests after a feature rollout",
                "report_frame": "Mobile performance release memo",
                "amount_unit": "requests/min across enrolled phones",
                "serving_unit": "release shard",
                "failure_mode": "thermal or battery budget miss after a cold run looked fine",
                "constraint_note": "battery, thermal headroom, memory pressure, and responsiveness",
                "base_terms_ms": {"data": 78.0, "compute": 40.0, "overhead": 20.0},
                "tail_multiplier": 1.28,
                "slo_ms": 150.0,
                "base_demand": 4200.0,
                "base_capacity_per_unit": 900.0,
                "default_units": 5,
                "max_units": 12,
                "unit_cost_day": 180.0,
                "default_cv_pct": 5.5,
                "quality_budget_pct": 2.0,
            },
            "oura_ring": {
                "scenario_id": "oura_sleep_feature_wake_window",
                "workload": "overnight sensing windows that must finish inside a duty-cycle wake slot",
                "report_frame": "Wearable firmware performance memo",
                "amount_unit": "sensing windows/hour",
                "serving_unit": "firmware cohort",
                "failure_mode": "duty-cycle energy or SRAM/flash budget miss",
                "constraint_note": "SRAM, flash, wake time, energy, and OTA payload",
                "base_terms_ms": {"data": 16.0, "compute": 9.0, "overhead": 7.0},
                "tail_multiplier": 1.25,
                "slo_ms": 35.0,
                "base_demand": 9600.0,
                "base_capacity_per_unit": 1800.0,
                "default_units": 6,
                "max_units": 16,
                "unit_cost_day": 45.0,
                "default_cv_pct": 8.0,
                "quality_budget_pct": 1.5,
            },
            "robotaxi": {
                "scenario_id": "robotaxi_perception_tail_deadline",
                "workload": "perception frames during dense urban driving",
                "report_frame": "Safety-critical performance memo",
                "amount_unit": "frames/s across active vehicles",
                "serving_unit": "compute lane",
                "failure_mode": "p99 deadline or safety margin miss",
                "constraint_note": "p99/p999 latency, recall, power, and sensor bandwidth",
                "base_terms_ms": {"data": 42.0, "compute": 58.0, "overhead": 12.0},
                "tail_multiplier": 1.45,
                "slo_ms": 95.0,
                "base_demand": 84.0,
                "base_capacity_per_unit": 32.0,
                "default_units": 3,
                "max_units": 10,
                "unit_cost_day": 620.0,
                "default_cv_pct": 4.0,
                "quality_budget_pct": 1.0,
            },
            "cloud_fleet": {
                "scenario_id": "cloud_llm_serving_efficiency",
                "workload": "LLM serving traffic during the weekday peak",
                "report_frame": "Fleet capacity and optimization memo",
                "amount_unit": "tokens/s at peak",
                "serving_unit": "GPU slice",
                "failure_mode": "SLA breach, queueing pressure, or negative ROI",
                "constraint_note": "SLA p99, throughput, cost/request, utilization, and headroom",
                "base_terms_ms": {"data": 96.0, "compute": 54.0, "overhead": 26.0},
                "tail_multiplier": 1.35,
                "slo_ms": 180.0,
                "base_demand": 19000.0,
                "base_capacity_per_unit": 4500.0,
                "default_units": 5,
                "max_units": 18,
                "unit_cost_day": 980.0,
                "default_cv_pct": 6.0,
                "quality_budget_pct": 2.0,
            },
        }
        packet = dict(shared)
        packet.update(track_configs.get(profile.track_id, track_configs["cloud_fleet"]))
        packet["headroom_target"] = packet["headroom_target_pct"] / 100.0
        packet["reality_tax"] = packet["reality_tax_pct"] / 100.0
        packet["quality_budget"] = packet["quality_budget_pct"] / 100.0
        packet["regression_budget"] = packet["regression_budget_pct"] / 100.0
        return packet

    def v2_09_best_lever(bottleneck):
        return {
            "data": "data_path",
            "compute": "compute_path",
            "overhead": "overhead_path",
        }.get(bottleneck, "data_path")

    def v2_09_profile_result(packet, optimization="none", pressure=1.0):
        terms = dict(packet["base_terms_ms"])
        terms["data"] *= 0.88 + 0.18 * pressure
        terms["compute"] *= 0.90 + 0.12 * pressure
        terms["overhead"] *= 0.80 + 0.22 * pressure

        effects = {
            "none": {"data": 1.0, "compute": 1.0, "overhead": 1.0, "quality": 0.0, "complexity": 0.0},
            "data_path": {"data": 0.58, "compute": 1.03, "overhead": 0.96, "quality": 0.012, "complexity": 0.35},
            "compute_path": {"data": 1.00, "compute": 0.62, "overhead": 1.03, "quality": 0.004, "complexity": 0.30},
            "overhead_path": {"data": 0.98, "compute": 1.00, "overhead": 0.45, "quality": 0.001, "complexity": 0.25},
            "capacity_path": {"data": 0.96, "compute": 0.96, "overhead": 0.98, "quality": 0.0, "complexity": 0.10},
        }
        effect = effects.get(optimization, effects["none"])
        for key in ("data", "compute", "overhead"):
            terms[key] *= effect[key]

        p50_ms = max(terms["data"], terms["compute"]) + terms["overhead"]
        load_tail = 1.0 + max(0.0, pressure - 1.0) * 0.30
        p99_ms = p50_ms * packet["tail_multiplier"] * load_tail
        bottleneck = max(terms, key=terms.get)
        return {
            "terms": terms,
            "p50_ms": p50_ms,
            "p99_ms": p99_ms,
            "slo_ok": p99_ms <= packet["slo_ms"],
            "bottleneck": bottleneck,
            "quality_risk": effect["quality"],
            "complexity": effect["complexity"],
        }

    def v2_09_regression_result(packet, sample_count, cv_pct, candidate_delta_pct):
        baseline = v2_09_profile_result(packet, v2_09_best_lever(v2_09_profile_result(packet)["bottleneck"]))
        baseline_ms = baseline["p99_ms"]
        candidate_ms = baseline_ms * (1.0 + candidate_delta_pct / 100.0)
        sigma_ms = baseline_ms * cv_pct / 100.0
        ci_diff_ms = 1.96 * math.sqrt(2.0) * sigma_ms / math.sqrt(max(1, sample_count))
        mde_pct = ci_diff_ms / baseline_ms * 100.0
        detectable = abs(candidate_delta_pct) >= mde_pct
        regression = candidate_delta_pct > packet["regression_budget_pct"]
        guardrail_fail = candidate_ms > packet["slo_ms"]
        if guardrail_fail and detectable:
            decision = "block"
            reason = "candidate p99 breaches the track guardrail and the effect is detectable"
        elif regression and detectable:
            decision = "block"
            reason = "regression exceeds the release budget and is detectable"
        elif regression or guardrail_fail:
            decision = "hold"
            reason = "effect is policy-relevant but hidden by variance; collect more canary evidence"
        elif candidate_delta_pct < -mde_pct:
            decision = "ship"
            reason = "improvement is detectable and guardrails hold"
        else:
            decision = "hold"
            reason = "change is inside the noise band; do not claim a speedup"
        return {
            "baseline_ms": baseline_ms,
            "candidate_ms": candidate_ms,
            "sigma_ms": sigma_ms,
            "ci_diff_ms": ci_diff_ms,
            "mde_pct": mde_pct,
            "detectable": detectable,
            "regression": regression,
            "guardrail_fail": guardrail_fail,
            "decision": decision,
            "reason": reason,
        }

    def v2_09_capacity_result(packet, demand_multiplier, units, optimization):
        base_profile = v2_09_profile_result(packet, "none")
        if optimization == "bottleneck_fix":
            optimization = v2_09_best_lever(base_profile["bottleneck"])
        profile = v2_09_profile_result(packet, optimization)
        service_rate = packet["base_capacity_per_unit"] * base_profile["p99_ms"] / max(1e-6, profile["p99_ms"])
        service_rate *= 1.0 - packet["reality_tax"]
        demand = packet["base_demand"] * demand_multiplier
        capacity = service_rate * units
        utilization = demand / max(1e-9, capacity)
        if utilization >= 1.0:
            queue_multiplier = 3.0 + 4.0 * min(1.0, utilization - 1.0)
        else:
            queue_multiplier = 1.0 + 0.35 * (utilization * utilization) / max(0.05, 1.0 - utilization)
        p99_ms = profile["p99_ms"] * queue_multiplier
        headroom = capacity / max(1e-9, demand) - 1.0
        daily_cost = units * packet["unit_cost_day"]
        failures = []
        if utilization >= 1.0:
            failures.append("demand exceeds measured capacity")
        if p99_ms > packet["slo_ms"]:
            failures.append("p99 exceeds track guardrail")
        if headroom < packet["headroom_target"]:
            failures.append("headroom below target")
        return {
            "optimization": optimization,
            "profile": profile,
            "demand": demand,
            "service_rate": service_rate,
            "capacity": capacity,
            "units": units,
            "utilization": utilization,
            "queue_multiplier": queue_multiplier,
            "p99_ms": p99_ms,
            "headroom": headroom,
            "daily_cost": daily_cost,
            "feasible": not failures,
            "failures": failures,
        }

    def v2_09_tradeoff_candidates(packet, demand_multiplier, units, risk_budget_pct, cost_ceiling):
        base = v2_09_profile_result(packet)
        targeted_lever = v2_09_best_lever(base["bottleneck"])
        quality_by_track = {
            "iphone": {"targeted": 0.8, "scale": 0.1, "precision": 1.9, "algorithmic": 0.9},
            "oura_ring": {"targeted": 0.7, "scale": 0.1, "precision": 2.4, "algorithmic": 1.4},
            "robotaxi": {"targeted": 0.4, "scale": 0.1, "precision": 4.8, "algorithmic": 2.4},
            "cloud_fleet": {"targeted": 0.7, "scale": 0.1, "precision": 1.3, "algorithmic": 1.0},
        }.get(packet["track_id"], {})
        specs = [
            {
                "id": "targeted",
                "optimization": targeted_lever,
                "unit_multiplier": 1.0,
                "cost_multiplier": 1.10,
                "p99_multiplier": 1.0,
                "reason": f"targets {v2_09_bottleneck_label(base['bottleneck'])}",
            },
            {
                "id": "scale",
                "optimization": "capacity_path",
                "unit_multiplier": 1.5,
                "cost_multiplier": 1.35,
                "p99_multiplier": 0.94,
                "reason": "buys queue headroom but raises recurring cost",
            },
            {
                "id": "precision",
                "optimization": "data_path",
                "unit_multiplier": 1.0,
                "cost_multiplier": 0.90,
                "p99_multiplier": 0.95,
                "reason": "cheap when bandwidth binds but carries quality and validation risk",
            },
            {
                "id": "algorithmic",
                "optimization": "compute_path" if base["bottleneck"] == "compute" else "overhead_path",
                "unit_multiplier": 1.0,
                "cost_multiplier": 1.20,
                "p99_multiplier": 0.82,
                "reason": "can cut latency but adds scheduler and validation complexity",
            },
        ]
        rows = []
        for spec in specs:
            cand_units = max(1, math.ceil(units * spec["unit_multiplier"]))
            capacity = v2_09_capacity_result(packet, demand_multiplier, cand_units, spec["optimization"])
            p99_ms = capacity["p99_ms"] * spec["p99_multiplier"]
            daily_cost = capacity["daily_cost"] * spec["cost_multiplier"]
            quality_loss_pct = quality_by_track.get(spec["id"], 1.0)
            failures = []
            if p99_ms > packet["slo_ms"]:
                failures.append("p99")
            if capacity["headroom"] < packet["headroom_target"]:
                failures.append("headroom")
            if quality_loss_pct > risk_budget_pct:
                failures.append("quality")
            if daily_cost > cost_ceiling:
                failures.append("cost")
            score = (
                p99_ms / packet["slo_ms"]
                + max(0.0, packet["headroom_target"] - capacity["headroom"]) * 2.0
                + daily_cost / max(1.0, cost_ceiling) * 0.45
                + quality_loss_pct / max(0.1, risk_budget_pct) * 0.30
                + (3.0 if failures else 0.0)
            )
            rows.append({
                "id": spec["id"],
                "label": v2_09_candidate_label(spec["id"]),
                "optimization": spec["optimization"],
                "units": cand_units,
                "p99_ms": p99_ms,
                "headroom": capacity["headroom"],
                "daily_cost": daily_cost,
                "quality_loss_pct": quality_loss_pct,
                "feasible": not failures,
                "failures": failures,
                "score": score,
                "reason": spec["reason"],
            })
        feasible = [row for row in rows if row["feasible"]]
        selected = min(feasible or rows, key=lambda row: row["score"])
        rejected_pool = [row for row in rows if row["id"] != selected["id"]]
        rejected = max(rejected_pool, key=lambda row: (not row["feasible"], -row["p99_ms"], row["daily_cost"]))
        if rejected["feasible"]:
            rejected = min(rejected_pool, key=lambda row: row["score"])
        return rows, selected, rejected

    def v2_09_part_banner(letter, title, why, color):
        return mo.Html(f"""
        <div style="margin:12px 0 16px 0;">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="background:{color}; color:white; border-radius:50%;
                            width:32px; height:32px; display:inline-flex; align-items:center;
                            justify-content:center; font-size:0.9rem; font-weight:800;
                            flex-shrink:0;">{html_lib.escape(letter)}</div>
                <div style="flex:1; height:2px; background:{COLORS['Border']};"></div>
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                            text-transform:uppercase; letter-spacing:0.12em;">
                    Concept module</div>
            </div>
            <div style="font-size:1.45rem; font-weight:800; color:{COLORS['Text']};
                        margin-top:8px; line-height:1.2;">{html_lib.escape(title)}</div>
            <div style="color:{COLORS['TextSec']}; font-size:0.92rem; margin-top:6px;
                        line-height:1.55; max-width:820px;">{html_lib.escape(why)}</div>
        </div>
        """)

    def v2_09_metric_card(label, value, subvalue="", color=None, danger=False):
        _color = color or COLORS["BlueLine"]
        _border = COLORS["RedLine"] if danger else COLORS["Border"]
        return f"""
        <div style="padding:14px 16px; border:1px solid {_border}; border-radius:8px;
                    min-width:148px; text-align:center; background:white;">
            <div style="color:{COLORS['TextMuted']}; font-size:0.72rem; font-weight:700;
                        text-transform:uppercase;">{html_lib.escape(label)}</div>
            <div style="font-size:1.45rem; font-weight:800; color:{_color}; font-family:monospace;
                        line-height:1.35;">{html_lib.escape(str(value))}</div>
            <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">{html_lib.escape(str(subvalue))}</div>
        </div>
        """

    def v2_09_table(headers, rows):
        head = "".join(f"<th>{html_lib.escape(str(header))}</th>" for header in headers)
        body_rows = []
        for row in rows:
            cells = "".join(f"<td>{html_lib.escape(str(cell))}</td>" for cell in row)
            body_rows.append(f"<tr>{cells}</tr>")
        return mo.Html(f"""
        <div style="overflow-x:auto; margin:10px 0;">
        <table style="width:100%; border-collapse:collapse; font-size:0.86rem;">
            <thead><tr style="background:{COLORS['Surface2']}; color:{COLORS['Text']};">{head}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
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

    def v2_09_math_peek(title, body):
        return mo.accordion({title: mo.md(body)})

    def v2_09_feedback(predicted, actual, success, miss):
        if predicted is None:
            return mo.callout(mo.md("Commit to the structured prediction before using this evidence."), kind="warn")
        if predicted == actual:
            return mo.callout(mo.md(success), kind="success")
        return mo.callout(mo.md(miss), kind="warn")

    def v2_09_failure_card(active, title, detail, recovery):
        if active:
            return mo.callout(mo.md(f"**{title}**  \n{detail}  \n\nRecovery path: {recovery}"), kind="danger")
        return mo.callout(mo.md(f"**Boundary holds: {title}**  \n{detail}"), kind="success")

    return (
        v2_09_best_lever,
        v2_09_bottleneck_label,
        v2_09_candidate_label,
        v2_09_capacity_result,
        v2_09_failure_card,
        v2_09_feedback,
        v2_09_math_peek,
        v2_09_metric_card,
        v2_09_money,
        v2_09_ms,
        v2_09_num,
        v2_09_optimization_label,
        v2_09_part_banner,
        v2_09_pct,
        v2_09_profile_result,
        v2_09_regression_result,
        v2_09_table,
        v2_09_track_packet,
        v2_09_tradeoff_candidates,
    )


@app.cell
def _(v2_09_profile, v2_09_track_packet):
    v2_09_packet = v2_09_track_packet(v2_09_profile)
    return (v2_09_packet,)


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    COLORS,
    LAB_CSS,
    mo,
    source_trace,
    track_arc_context,
    track_context,
    v2_09_metadata,
    v2_09_packet,
    v2_09_profile,
):
    reading_rows = """
    <tr><td>Part A</td><td>Iron law and profiling hierarchy</td><td>Localize the active bottleneck before choosing a lever.</td></tr>
    <tr><td>Part B</td><td>Profiling feedback loop and scaling regressions</td><td>Use baseline, variance, and detectability evidence.</td></tr>
    <tr><td>Part C</td><td>Measurement at scale and fleet efficiency</td><td>Convert measured service rate into headroom and cost.</td></tr>
    <tr><td>Part D</td><td>Efficiency frontier and optimization playbook</td><td>Defend a trade-off and reject an alternative.</td></tr>
    """
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background:linear-gradient(135deg, {COLORS['Surface0']} 0%, {COLORS['Surface1']} 100%);
                    border-radius:16px; padding:32px 40px; margin-bottom:8px;
                    border:1px solid #2d3748;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
                <div>
                    <div style="font-size:0.72rem; font-weight:700; color:#94a3b8;
                                text-transform:uppercase; letter-spacing:0.14em; margin-bottom:8px;">
                        Vol 2 &middot; Lab 09 &middot; Performance Engineering
                    </div>
                    <div style="font-size:2rem; font-weight:800; color:#f1f5f9; line-height:1.15; margin-bottom:10px;">
                        The Optimization Trap
                    </div>
                    <div style="font-size:0.95rem; color:#94a3b8; max-width:760px; line-height:1.6;">
                        {v2_09_packet['stakeholder']} must improve {v2_09_packet['workload']}.
                        The invariant is measurement-driven optimization: localize, detect, plan, and defend.
                    </div>
                </div>
                <div style="display:flex; flex-direction:column; gap:8px; flex-shrink:0;">
                    <span class="badge badge-info">{v2_09_profile.label}</span>
                    <span class="badge badge-info">{v2_09_packet['hardware_ref']}</span>
                    <span class="badge badge-warn">45-55 minutes &middot; 4 Parts + Synthesis</span>
                </div>
            </div>
        </div>
        """),
        track_context(v2_09_profile),
        track_arc_context(v2_09_profile, v2_09_metadata.lab_id),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <div class="mlsysbook-section-label">Shared Concept Sequence</div>
          <h2>One Sequence, Track-Specific Evidence</h2>
          <p>All tracks run the same modules: bottleneck localization, regression detectability,
          capacity planning, trade-off reporting, and a synthesis memo for V2-10 inference.</p>
          <div style="overflow-x:auto;">
          <table style="width:100%; border-collapse:collapse; font-size:0.86rem;">
            <thead><tr style="background:{COLORS['Surface2']};"><th>Module</th><th>Reading anchor</th><th>Student decision</th></tr></thead>
            <tbody>{reading_rows}</tbody>
          </table>
          </div>
        </div>
        """),
        source_trace({
            "chapter": "Volume II, Chapter 9: Performance Engineering",
            "anchors": (
                "The iron law of ML performance",
                "System Profiling",
                "Profiling feedback loop",
                "Measurement at Scale",
                "Detecting scaling regressions",
                "The Optimization Playbook",
                "Fallacies and Pitfalls",
            ),
            "local_models": "Notebook-local v2_09_* teaching models encode the track thresholds and calculations.",
            "track_source": v2_09_packet["track_source"],
        }, summary="Source trace: V2-09 concept-module evidence"),
    ])
    return


@app.cell(hide_code=True)
def _(mo, v2_09_packet):
    partA_prediction = mo.ui.radio(
        options={
            "Data movement / memory bandwidth": "data",
            "Compute path": "compute",
            "Launch or coordination overhead": "overhead",
            "Fleet capacity, not local code": "capacity",
        },
        label=f"Part A prediction: what is the active bottleneck for {v2_09_packet['label']}?",
    )
    partA_pressure = mo.ui.slider(
        start=0.75,
        stop=1.40,
        value=1.00,
        step=0.05,
        label="Workload pressure multiplier",
    )
    partA_lever = mo.ui.radio(
        options={
            "Reduce data movement": "data_path",
            "Improve compute path": "compute_path",
            "Reduce launch/coordination overhead": "overhead_path",
            "Add serving capacity": "capacity_path",
        },
        label="Optimization lever to try first",
    )
    partA_checkpoint = mo.ui.radio(
        options={
            "Authorize data-movement work": "data_path",
            "Authorize compute-kernel work": "compute_path",
            "Authorize overhead/graph work": "overhead_path",
            "Do not optimize locally; plan capacity instead": "capacity_path",
        },
        label="Checkpoint: which first experiment should the memo authorize?",
    )
    return partA_checkpoint, partA_lever, partA_prediction, partA_pressure


@app.cell(hide_code=True)
def _(mo, v2_09_packet):
    partB_prediction = mo.ui.radio(
        options={
            "Ship: the change is clearly safe": "ship",
            "Hold: the effect is inside measurement noise": "hold",
            "Block: this is a detectable regression": "block",
            "Ignore variance and use the latest run": "ignore",
        },
        label="Part B prediction: what release decision will the evidence support?",
    )
    partB_samples = mo.ui.slider(start=3, stop=50, value=8, step=1, label="Repeated runs per variant")
    partB_cv = mo.ui.slider(
        start=2.0,
        stop=20.0,
        value=float(v2_09_packet["default_cv_pct"]),
        step=0.5,
        label="Run-to-run CV (%)",
    )
    partB_delta = mo.ui.slider(
        start=-8.0,
        stop=12.0,
        value=5.0,
        step=0.5,
        label="Candidate p99 delta vs baseline (%)",
    )
    partB_checkpoint = mo.ui.radio(
        options={
            "Ship after canary": "ship",
            "Hold and collect more samples": "hold",
            "Block and revert": "block",
            "Shadow traffic only": "shadow",
        },
        label="Checkpoint: what should the release owner do?",
    )
    return partB_checkpoint, partB_cv, partB_delta, partB_prediction, partB_samples


@app.cell(hide_code=True)
def _(mo, v2_09_packet):
    partC_prediction = mo.ui.radio(
        options={
            "Under-capacity: headroom will fail": "under",
            "Enough headroom: launch as sized": "enough",
            "Overbought: too much cost for the demand": "overbuy",
            "Unknown until p99 is included": "unknown",
        },
        label="Part C prediction: what will the capacity plan reveal?",
    )
    partC_demand = mo.ui.slider(
        start=0.60,
        stop=2.40,
        value=1.00,
        step=0.05,
        label="Demand forecast multiplier",
    )
    partC_units = mo.ui.slider(
        start=1,
        stop=int(v2_09_packet["max_units"]),
        value=int(v2_09_packet["default_units"]),
        step=1,
        label=f"Serving units ({v2_09_packet['serving_unit']})",
    )
    partC_optimization = mo.ui.radio(
        options={
            "No optimization": "none",
            "Fix measured bottleneck": "bottleneck_fix",
            "Reduce data movement": "data_path",
            "Improve compute path": "compute_path",
            "Reduce launch/coordination overhead": "overhead_path",
        },
        value="Fix measured bottleneck",
        label="Local profile used for capacity planning",
    )
    partC_checkpoint = mo.ui.radio(
        options={
            "Add serving units": "add_units",
            "Optimize measured bottleneck before buying capacity": "optimize_first",
            "Reduce rollout scope": "reduce_scope",
            "Launch with current headroom": "launch",
        },
        label="Checkpoint: which capacity decision should go into the memo?",
    )
    return partC_checkpoint, partC_demand, partC_optimization, partC_prediction, partC_units


@app.cell(hide_code=True)
def _(mo, v2_09_money, v2_09_packet):
    partD_prediction = mo.ui.radio(
        options={
            "Target measured bottleneck": "targeted",
            "Buy more serving units": "scale",
            "Aggressive precision/cache compression": "precision",
            "Speculative or algorithmic path": "algorithmic",
        },
        label="Part D prediction: which candidate will survive the report guardrails?",
    )
    partD_risk_budget = mo.ui.slider(
        start=0.5,
        stop=6.0,
        value=float(v2_09_packet["quality_budget_pct"]),
        step=0.5,
        label="Allowed quality/safety risk (%)",
    )
    _default_cost = (
        v2_09_packet["default_units"]
        * v2_09_packet["unit_cost_day"]
        * v2_09_packet["cost_ceiling_multiplier"]
    )
    partD_cost_ceiling = mo.ui.number(
        start=0.0,
        stop=max(_default_cost * 3.0, 1.0),
        value=_default_cost,
        step=max(_default_cost / 50.0, 1.0),
        label=f"Daily cost ceiling ({v2_09_money(_default_cost)} default)",
    )
    partD_checkpoint = mo.ui.radio(
        options={
            "Target measured bottleneck": "targeted",
            "Buy more serving units": "scale",
            "Aggressive precision/cache compression": "precision",
            "Speculative or algorithmic path": "algorithmic",
        },
        label="Checkpoint: which candidate should the memo defend?",
    )
    partD_rejected = mo.ui.radio(
        options={
            "Target measured bottleneck": "targeted",
            "Buy more serving units": "scale",
            "Aggressive precision/cache compression": "precision",
            "Speculative or algorithmic path": "algorithmic",
        },
        label="Checkpoint: which tempting alternative should the memo reject?",
    )
    return partD_checkpoint, partD_cost_ceiling, partD_prediction, partD_rejected, partD_risk_budget


@app.cell(hide_code=True)
def _(mo):
    v2_09_next_implication = mo.ui.radio(
        options={
            "Carry measured p99 into V2-10 serving SLO policy": "measured_p99",
            "Carry capacity headroom into V2-10 batching/admission": "capacity_headroom",
            "Carry residual bottleneck into V2-10 optimization risk": "residual_bottleneck",
            "Carry regression canary into V2-10 rollout guardrails": "regression_canary",
        },
        label="Synthesis: what evidence should V2-10 inference inherit first?",
    )
    v2_09_student_id = mo.ui.text(label="Student or team ID")
    return v2_09_next_implication, v2_09_student_id


@app.cell
def _(
    partA_lever,
    partA_pressure,
    partB_cv,
    partB_delta,
    partB_samples,
    partC_demand,
    partC_optimization,
    partC_units,
    partD_cost_ceiling,
    partD_risk_budget,
    v2_09_best_lever,
    v2_09_capacity_result,
    v2_09_packet,
    v2_09_profile_result,
    v2_09_regression_result,
    v2_09_tradeoff_candidates,
):
    v2_09_partA_base = v2_09_profile_result(v2_09_packet, "none", partA_pressure.value)
    v2_09_partA_best_lever = v2_09_best_lever(v2_09_partA_base["bottleneck"])
    v2_09_partA_selected_lever = partA_lever.value or v2_09_partA_best_lever
    v2_09_partA_after = v2_09_profile_result(
        v2_09_packet,
        v2_09_partA_selected_lever,
        partA_pressure.value,
    )
    v2_09_partA_speedup = v2_09_partA_base["p99_ms"] / max(1e-9, v2_09_partA_after["p99_ms"])
    v2_09_partB = v2_09_regression_result(
        v2_09_packet,
        partB_samples.value,
        partB_cv.value,
        partB_delta.value,
    )
    v2_09_partC = v2_09_capacity_result(
        v2_09_packet,
        partC_demand.value,
        partC_units.value,
        partC_optimization.value or "bottleneck_fix",
    )
    v2_09_partD_candidates, v2_09_partD_selected, v2_09_partD_rejected = v2_09_tradeoff_candidates(
        v2_09_packet,
        partC_demand.value,
        partC_units.value,
        partD_risk_budget.value,
        partD_cost_ceiling.value,
    )
    return (
        v2_09_partA_after,
        v2_09_partA_base,
        v2_09_partA_best_lever,
        v2_09_partA_selected_lever,
        v2_09_partA_speedup,
        v2_09_partB,
        v2_09_partC,
        v2_09_partD_candidates,
        v2_09_partD_rejected,
        v2_09_partD_selected,
    )


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    build_lab_report,
    go,
    ledger,
    mo,
    partA_checkpoint,
    partA_lever,
    partA_prediction,
    partA_pressure,
    partB_checkpoint,
    partB_cv,
    partB_delta,
    partB_prediction,
    partB_samples,
    partC_checkpoint,
    partC_demand,
    partC_optimization,
    partC_prediction,
    partC_units,
    partD_checkpoint,
    partD_cost_ceiling,
    partD_prediction,
    partD_rejected,
    partD_risk_budget,
    report_export_panel,
    v2_09_best_lever,
    v2_09_bottleneck_label,
    v2_09_candidate_label,
    v2_09_capacity_result,
    v2_09_chapter,
    v2_09_failure_card,
    v2_09_feedback,
    v2_09_math_peek,
    v2_09_metadata,
    v2_09_metric_card,
    v2_09_money,
    v2_09_ms,
    v2_09_next_implication,
    v2_09_num,
    v2_09_optimization_label,
    v2_09_packet,
    v2_09_partA_after,
    v2_09_partA_base,
    v2_09_partA_best_lever,
    v2_09_partA_selected_lever,
    v2_09_partA_speedup,
    v2_09_partB,
    v2_09_partC,
    v2_09_partD_candidates,
    v2_09_partD_rejected,
    v2_09_partD_selected,
    v2_09_part_banner,
    v2_09_pct,
    v2_09_profile,
    v2_09_table,
    v2_09_student_id,
):
    def v2_09_opening():
        return mo.vstack([
            mo.md(f"""
## Lab Brief

You are the performance engineer for **{v2_09_packet['label']}**. The stakeholder
has a concrete launch decision for **{v2_09_packet['workload']}**.

The shared sequence is:

1. Localize the bottleneck before optimizing.
2. Prove regressions with baseline, variance, and detectability.
3. Convert measured bottlenecks into capacity headroom and cost.
4. Defend an optimization trade-off and reject an alternative.
5. Save a memo that V2-10 inference can inherit.
            """),
            mo.callout(mo.md(
                f"**Track consequence:** for {v2_09_packet['label']}, the natural failure is "
                f"{v2_09_packet['failure_mode']}. Track constraints emphasize "
                f"{v2_09_packet['constraint_note']}."
            ), kind="info"),
            v2_09_table(
                ("Part", "Concept", "Evidence saved"),
                (
                    ("A", "Localize bottleneck", "active term, selected first lever, p99 speedup"),
                    ("B", "Regression detectability", "n, CV, MDE, canary decision"),
                    ("C", "Capacity planning", "demand, capacity, utilization, headroom, daily cost"),
                    ("D", "Trade-off report", "selected candidate, rejected alternative, feasibility reason"),
                    ("Synthesis", "Performance memo", "V2-10 inference implication"),
                ),
            ),
        ])

    def build_part_a():
        items = [
            v2_09_part_banner(
                "A",
                "Localize The Bottleneck Before Optimizing",
                f"{v2_09_packet['stakeholder']} needs to decide which experiment is worth engineering time first.",
                COLORS["BlueLine"],
            ),
            mo.md("""
### Scenario

A quick optimization request arrives. Before changing kernels, precision, graph
capture, or capacity, commit to the bottleneck you think is active.
            """),
            partA_prediction,
        ]
        if partA_prediction.value is None:
            items.append(mo.callout(mo.md("Commit to the bottleneck prediction before revealing the profile."), kind="warn"))
            return mo.vstack(items)

        items.extend([mo.hstack([partA_pressure, partA_lever], widths="equal")])
        labels = ["Data movement", "Compute", "Overhead"]
        term_keys = ["data", "compute", "overhead"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=[v2_09_partA_base["terms"][key] for key in term_keys],
            name="Measured baseline",
            marker_color=COLORS["BlueLine"],
        ))
        fig.add_trace(go.Bar(
            x=labels,
            y=[v2_09_partA_after["terms"][key] for key in term_keys],
            name=v2_09_optimization_label(v2_09_partA_selected_lever),
            marker_color=COLORS["OrangeLine"],
        ))
        fig.update_layout(
            barmode="group",
            height=350,
            yaxis=dict(title="Exposed time term (ms)", gridcolor="#edf2f7"),
            legend=dict(orientation="h", y=1.14, x=0),
            margin=dict(l=60, r=20, t=55, b=50),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        miss = v2_09_partA_selected_lever != v2_09_partA_best_lever
        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:14px 0;">
          {v2_09_metric_card("Actual bottleneck", v2_09_bottleneck_label(v2_09_partA_base['bottleneck']), "largest exposed term", COLORS["BlueLine"])}
          {v2_09_metric_card("Baseline p99", v2_09_ms(v2_09_partA_base['p99_ms']), f"SLO {v2_09_ms(v2_09_packet['slo_ms'], 0)}", COLORS["OrangeLine"], not v2_09_partA_base["slo_ok"])}
          {v2_09_metric_card("After p99", v2_09_ms(v2_09_partA_after['p99_ms']), f"{v2_09_partA_speedup:.2f}x speedup", COLORS["GreenLine"], not v2_09_partA_after["slo_ok"])}
          {v2_09_metric_card("First lever", v2_09_optimization_label(v2_09_partA_best_lever), "bottleneck-targeted", COLORS["GreenLine"])}
        </div>
        """))
        items.append(v2_09_table(
            ("Term", "Baseline", "After selected lever", "Interpretation"),
            (
                ("Data movement", v2_09_ms(v2_09_partA_base["terms"]["data"]), v2_09_ms(v2_09_partA_after["terms"]["data"]), "bandwidth/cache/precision/fusion path"),
                ("Compute", v2_09_ms(v2_09_partA_base["terms"]["compute"]), v2_09_ms(v2_09_partA_after["terms"]["compute"]), "kernel math/Tensor Core path"),
                ("Overhead", v2_09_ms(v2_09_partA_base["terms"]["overhead"]), v2_09_ms(v2_09_partA_after["terms"]["overhead"]), "launch, scheduler, coordination, or communication gaps"),
                ("p99", v2_09_ms(v2_09_partA_base["p99_ms"]), v2_09_ms(v2_09_partA_after["p99_ms"]), "track guardrail evidence"),
            ),
        ))
        items.append(v2_09_feedback(
            partA_prediction.value,
            v2_09_partA_base["bottleneck"],
            "**Prediction matched the measured bottleneck.** The first optimization now has a measured target.",
            f"**Optimization trap:** the profile says the active bottleneck is {v2_09_bottleneck_label(v2_09_partA_base['bottleneck'])}, not the predicted target.",
        ))
        items.append(v2_09_failure_card(
            miss or not v2_09_partA_after["slo_ok"],
            "Wrong lever or remaining guardrail miss",
            (
                f"Selected lever: {v2_09_optimization_label(v2_09_partA_selected_lever)}. "
                f"Targeted lever: {v2_09_optimization_label(v2_09_partA_best_lever)}. "
                f"After p99: {v2_09_ms(v2_09_partA_after['p99_ms'])} against SLO {v2_09_ms(v2_09_packet['slo_ms'], 0)}."
            ),
            "switch to the bottleneck-targeted lever, then reprofile before claiming a win",
        ))
        items.append(v2_09_math_peek(
            "Math Peek / Source Model - iron-law bottleneck",
            f"""
```
T_p50 ~= max(data_movement_ms, compute_ms) + overhead_ms
T_p99 ~= T_p50 * track_tail_multiplier
speedup = baseline_p99 / optimized_p99
first_lever = lever_that_targets(max(data, compute, overhead))
```

Current profile: data `{v2_09_ms(v2_09_partA_base['terms']['data'])}`,
compute `{v2_09_ms(v2_09_partA_base['terms']['compute'])}`, overhead
`{v2_09_ms(v2_09_partA_base['terms']['overhead'])}`. Chapter anchor:
the iron law of ML performance and the profiling hierarchy.
            """,
        ))
        items.append(partA_checkpoint)
        return mo.vstack(items)

    def build_part_b():
        items = [
            v2_09_part_banner(
                "B",
                "Regressions Need Baseline, Variance, And Detectability",
                "A before/after number is not release evidence until variance and sample size say the effect is detectable.",
                COLORS["OrangeLine"],
            ),
            mo.md("""
### Scenario

A candidate patch claims to improve performance, but production-like runs vary.
Decide whether the release owner should ship, hold for more evidence, or block.
            """),
            partB_prediction,
        ]
        if partB_prediction.value is None:
            items.append(mo.callout(mo.md("Commit to the regression decision prediction before revealing the run evidence."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_samples, partB_cv, partB_delta], widths="equal"))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Baseline", "Candidate"],
            y=[v2_09_partB["baseline_ms"], v2_09_partB["candidate_ms"]],
            error_y=dict(type="data", array=[v2_09_partB["ci_diff_ms"] / 2.0, v2_09_partB["ci_diff_ms"] / 2.0], visible=True),
            marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]],
            name="p99 with uncertainty",
        ))
        fig.add_hline(
            y=v2_09_packet["slo_ms"],
            line_dash="dash",
            line_color=COLORS["RedLine"],
            annotation_text=f"SLO {v2_09_ms(v2_09_packet['slo_ms'], 0)}",
        )
        fig.update_layout(
            height=340,
            yaxis=dict(title="p99 latency / time budget (ms)", gridcolor="#edf2f7"),
            showlegend=False,
            margin=dict(l=60, r=20, t=50, b=50),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:14px 0;">
          {v2_09_metric_card("MDE", f"{v2_09_partB['mde_pct']:.1f}%", "minimum detectable effect", COLORS["BlueLine"])}
          {v2_09_metric_card("Candidate delta", f"{partB_delta.value:.1f}%", f"budget {v2_09_packet['regression_budget_pct']:.1f}%", COLORS["OrangeLine"], v2_09_partB["regression"])}
          {v2_09_metric_card("Detectable", "yes" if v2_09_partB["detectable"] else "no", f"n={partB_samples.value}, CV={partB_cv.value:.1f}%", COLORS["GreenLine"] if v2_09_partB["detectable"] else COLORS["OrangeLine"])}
          {v2_09_metric_card("Decision", v2_09_partB["decision"], v2_09_partB["reason"], COLORS["GreenLine"] if v2_09_partB["decision"] == "ship" else COLORS["RedLine"] if v2_09_partB["decision"] == "block" else COLORS["OrangeLine"])}
        </div>
        """))
        items.append(v2_09_table(
            ("Evidence", "Value", "Why it matters"),
            (
                ("Baseline p99", v2_09_ms(v2_09_partB["baseline_ms"]), "reference distribution after the Part A bottleneck fix"),
                ("Candidate p99", v2_09_ms(v2_09_partB["candidate_ms"]), "patched distribution under the same scenario"),
                ("95% diff half-width", v2_09_ms(v2_09_partB["ci_diff_ms"]), "uncertainty around before/after comparison"),
                ("Minimum detectable effect", f"{v2_09_partB['mde_pct']:.1f}%", "smallest effect this experiment can distinguish"),
                ("Release decision", v2_09_partB["decision"], v2_09_partB["reason"]),
            ),
        ))
        items.append(v2_09_feedback(
            partB_prediction.value,
            v2_09_partB["decision"],
            "**Prediction matched the evidence.** The release decision is supported by variance-aware data.",
            "**Regression evidence gap:** one run is not enough; the baseline, CV, sample count, and MDE drive the decision.",
        ))
        items.append(v2_09_failure_card(
            v2_09_partB["decision"] != "ship",
            "Regression evidence is not launch-safe",
            (
                f"Decision `{v2_09_partB['decision']}` because {v2_09_partB['reason']}. "
                f"The measured delta is {partB_delta.value:.1f}% and the MDE is {v2_09_partB['mde_pct']:.1f}%."
            ),
            "increase sample count, reduce measurement noise, or block/revert if the regression is detectable",
        ))
        items.append(v2_09_math_peek(
            "Math Peek / Source Model - baseline variance and detectability",
            f"""
```
sigma_ms = baseline_ms * CV
CI_diff  = 1.96 * sqrt(2) * sigma_ms / sqrt(n)
MDE_pct  = CI_diff / baseline_ms
```

Current values: n `{partB_samples.value}`, CV `{partB_cv.value:.1f}%`,
MDE `{v2_09_partB['mde_pct']:.1f}%`, candidate delta `{partB_delta.value:.1f}%`.
Chapter anchors: profiling feedback loop, profiling at scale, and detecting
scaling regressions.
            """,
        ))
        items.append(partB_checkpoint)
        return mo.vstack(items)

    def build_part_c():
        items = [
            v2_09_part_banner(
                "C",
                "Capacity Planning Converts Bottlenecks Into Headroom And Cost",
                "The local profile becomes a fleet amount system: demand, service rate, units, utilization, headroom, p99, and cost.",
                COLORS["GreenLine"],
            ),
            mo.md(f"""
### Scenario

The launch forecast is expressed as **{v2_09_packet['amount_unit']}**. Use the
measured profile to decide whether the selected **{v2_09_packet['serving_unit']}**
count is enough.
            """),
            partC_prediction,
        ]
        if partC_prediction.value is None:
            items.append(mo.callout(mo.md("Commit to the capacity prediction before opening the amount-system planner."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_demand, partC_units, partC_optimization], widths="equal"))
        x_units = list(range(1, int(v2_09_packet["max_units"]) + 1))
        y_capacity = [
            v2_09_capacity_result(v2_09_packet, partC_demand.value, units, partC_optimization.value or "bottleneck_fix")["capacity"]
            for units in x_units
        ]
        demand_line = [v2_09_partC["demand"] for _ in x_units]
        headroom_line = [v2_09_partC["demand"] * (1.0 + v2_09_packet["headroom_target"]) for _ in x_units]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_units, y=y_capacity, mode="lines+markers", name="Measured capacity", line=dict(color=COLORS["BlueLine"], width=3)))
        fig.add_trace(go.Scatter(x=x_units, y=demand_line, mode="lines", name="Forecast demand", line=dict(color=COLORS["OrangeLine"], dash="dash")))
        fig.add_trace(go.Scatter(x=x_units, y=headroom_line, mode="lines", name="Demand + headroom target", line=dict(color=COLORS["GreenLine"], dash="dot")))
        fig.add_trace(go.Scatter(x=[partC_units.value], y=[v2_09_partC["capacity"]], mode="markers", name="selected plan", marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond")))
        fig.update_layout(
            height=360,
            xaxis=dict(title=v2_09_packet["serving_unit"]),
            yaxis=dict(title=v2_09_packet["amount_unit"], gridcolor="#edf2f7"),
            legend=dict(orientation="h", y=1.17, x=0),
            margin=dict(l=70, r=20, t=55, b=50),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:14px 0;">
          {v2_09_metric_card("Demand", v2_09_num(v2_09_partC['demand']), v2_09_packet["amount_unit"], COLORS["OrangeLine"])}
          {v2_09_metric_card("Capacity", v2_09_num(v2_09_partC['capacity']), f"{partC_units.value} {v2_09_packet['serving_unit']}", COLORS["BlueLine"])}
          {v2_09_metric_card("Utilization", v2_09_pct(v2_09_partC['utilization']), "demand / capacity", COLORS["OrangeLine"], v2_09_partC["utilization"] >= 1.0)}
          {v2_09_metric_card("Headroom", v2_09_pct(v2_09_partC['headroom']), f"target {v2_09_packet['headroom_target_pct']:.0f}%", COLORS["GreenLine"], v2_09_partC["headroom"] < v2_09_packet["headroom_target"])}
          {v2_09_metric_card("Daily cost", v2_09_money(v2_09_partC['daily_cost']), "serving-unit cost", COLORS["BlueLine"])}
        </div>
        """))
        items.append(v2_09_table(
            ("Amount-system field", "Value", "Decision use"),
            (
                ("Measured service rate", f"{v2_09_num(v2_09_partC['service_rate'])} {v2_09_packet['amount_unit']} per {v2_09_packet['serving_unit']}", "capacity per unit after reality tax"),
                ("Forecast demand", f"{v2_09_num(v2_09_partC['demand'])} {v2_09_packet['amount_unit']}", "what the launch must absorb"),
                ("Total capacity", f"{v2_09_num(v2_09_partC['capacity'])} {v2_09_packet['amount_unit']}", "serving units times measured rate"),
                ("p99 under load", v2_09_ms(v2_09_partC["p99_ms"]), f"track SLO {v2_09_ms(v2_09_packet['slo_ms'], 0)}"),
                ("Failures", ", ".join(v2_09_partC["failures"]) or "none", "launch gate"),
            ),
        ))
        expected_capacity = "under" if not v2_09_partC["feasible"] else "enough"
        items.append(v2_09_feedback(
            partC_prediction.value,
            expected_capacity,
            "**Prediction matched the capacity evidence.** The amount system makes the launch decision explicit.",
            "**Capacity planning trap:** average speed is not enough; headroom and p99 determine whether the plan is launchable.",
        ))
        items.append(v2_09_failure_card(
            not v2_09_partC["feasible"],
            "Capacity plan misses the operating envelope",
            (
                f"Failures: {', '.join(v2_09_partC['failures']) or 'none'}. "
                f"p99 is {v2_09_ms(v2_09_partC['p99_ms'])}; headroom is {v2_09_pct(v2_09_partC['headroom'])}."
            ),
            "add measured capacity, lower demand, or apply the bottleneck fix before launch",
        ))
        items.append(v2_09_math_peek(
            "Math Peek / Source Model - capacity headroom",
            f"""
```
service_rate = baseline_capacity_per_unit * baseline_p99 / optimized_p99
capacity     = serving_units * service_rate * (1 - reality_tax)
utilization  = demand / capacity
headroom     = capacity / demand - 1
p99_load     = measured_p99 * queue_pressure(utilization)
```

Current values: demand `{v2_09_num(v2_09_partC['demand'])}`,
capacity `{v2_09_num(v2_09_partC['capacity'])}`, utilization
`{v2_09_pct(v2_09_partC['utilization'])}`, headroom
`{v2_09_pct(v2_09_partC['headroom'])}`. Chapter anchors: Measurement at
Scale, fleet efficiency, and benchmark reality tax.
            """,
        ))
        items.append(partC_checkpoint)
        return mo.vstack(items)

    def build_part_d():
        items = [
            v2_09_part_banner(
                "D",
                "Optimization Reports Must Defend A Trade-Off",
                "The final memo must explain why one optimization survives p99, cost, quality/risk, and headroom while another is rejected.",
                COLORS["RedLine"],
            ),
            mo.md("""
### Scenario

The engineering lead does not want a benchmark screenshot. They want the
optimization you recommend, the trade-off it accepts, and the tempting
alternative it rejects.
            """),
            partD_prediction,
        ]
        if partD_prediction.value is None:
            items.append(mo.callout(mo.md("Commit to the candidate prediction before opening the frontier."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_risk_budget, partD_cost_ceiling], widths="equal"))
        fig = go.Figure()
        for row in v2_09_partD_candidates:
            fig.add_trace(go.Scatter(
                x=[row["daily_cost"]],
                y=[row["p99_ms"]],
                mode="markers+text",
                text=[row["label"]],
                textposition="top center",
                name=row["label"],
                marker=dict(
                    size=16,
                    color=COLORS["GreenLine"] if row["feasible"] else COLORS["RedLine"],
                    symbol="circle" if row["feasible"] else "x",
                ),
                hovertemplate="%{text}<br>cost=%{x:.1f}<br>p99=%{y:.1f} ms<extra></extra>",
            ))
        fig.add_hline(y=v2_09_packet["slo_ms"], line_dash="dash", line_color=COLORS["RedLine"], annotation_text="p99 guardrail")
        fig.add_vline(x=partD_cost_ceiling.value, line_dash="dash", line_color=COLORS["OrangeLine"], annotation_text="cost ceiling")
        fig.update_layout(
            height=390,
            xaxis=dict(title="Daily cost", gridcolor="#edf2f7"),
            yaxis=dict(title="p99 latency / time budget (ms)", gridcolor="#edf2f7"),
            showlegend=False,
            margin=dict(l=70, r=30, t=50, b=55),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        rows = []
        for row in v2_09_partD_candidates:
            rows.append((
                row["label"],
                v2_09_ms(row["p99_ms"]),
                v2_09_pct(row["headroom"]),
                v2_09_money(row["daily_cost"]),
                f"{row['quality_loss_pct']:.1f}%",
                "PASS" if row["feasible"] else "FAIL: " + ", ".join(row["failures"]),
                row["reason"],
            ))
        items.append(v2_09_table(
            ("Candidate", "p99", "Headroom", "Daily cost", "Quality/risk", "Gate", "Trade-off"),
            rows,
        ))
        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:14px 0;">
          {v2_09_metric_card("Selected", v2_09_partD_selected['label'], "best feasible score", COLORS["GreenLine"], not v2_09_partD_selected["feasible"])}
          {v2_09_metric_card("Rejected", v2_09_partD_rejected['label'], ", ".join(v2_09_partD_rejected["failures"]) or "dominated", COLORS["RedLine"])}
          {v2_09_metric_card("Risk budget", f"{partD_risk_budget.value:.1f}%", "quality/safety loss", COLORS["OrangeLine"])}
          {v2_09_metric_card("Cost ceiling", v2_09_money(partD_cost_ceiling.value), "daily", COLORS["BlueLine"])}
        </div>
        """))
        items.append(v2_09_feedback(
            partD_prediction.value,
            v2_09_partD_selected["id"],
            "**Prediction matched the frontier.** The chosen option survives all report guardrails.",
            "**Trade-off trap:** fastest, cheapest, or most familiar is not enough; feasibility is a conjunction of p99, headroom, quality/risk, and cost.",
        ))
        items.append(v2_09_failure_card(
            not v2_09_partD_selected["feasible"],
            "No candidate satisfies all report guardrails",
            (
                f"Best candidate is {v2_09_partD_selected['label']}, but failures are "
                f"{', '.join(v2_09_partD_selected['failures']) or 'none'}."
            ),
            "relax scope, collect more evidence, or change the capacity plan before signing the memo",
        ))
        items.append(v2_09_math_peek(
            "Math Peek / Source Model - trade-off feasibility",
            f"""
```
feasible =
  p99 <= track_slo
  and headroom >= target_headroom
  and quality_loss <= risk_budget
  and daily_cost <= cost_ceiling

score = p99/SLO + headroom_penalty + cost_penalty + quality_penalty
```

Selected candidate: `{v2_09_partD_selected['label']}`.
Rejected alternative: `{v2_09_partD_rejected['label']}`. Chapter anchors:
efficiency frontier, combining techniques, and case-study lessons.
            """,
        ))
        items.extend([partD_checkpoint, partD_rejected])
        return mo.vstack(items)

    def build_synthesis():
        incomplete = []
        required = {
            "Part A prediction": partA_prediction.value,
            "Part A checkpoint": partA_checkpoint.value,
            "Part B prediction": partB_prediction.value,
            "Part B checkpoint": partB_checkpoint.value,
            "Part C prediction": partC_prediction.value,
            "Part C checkpoint": partC_checkpoint.value,
            "Part D prediction": partD_prediction.value,
            "Part D checkpoint": partD_checkpoint.value,
            "Part D rejected alternative": partD_rejected.value,
            "V2-10 implication": v2_09_next_implication.value,
        }
        for label, value in required.items():
            if value is None:
                incomplete.append(label)
        implication_text = {
            "measured_p99": "V2-10 must use measured p99 as the serving SLO input, not peak throughput.",
            "capacity_headroom": "V2-10 batching and admission controls must preserve the measured capacity headroom.",
            "residual_bottleneck": "V2-10 should treat the residual bottleneck as an inference rollout risk.",
            "regression_canary": "V2-10 rollout must include the regression canary because variance can hide tail changes.",
        }.get(v2_09_next_implication.value, "V2-10 implication not selected yet.")
        selected_id = partD_checkpoint.value or v2_09_partD_selected["id"]
        rejected_id = partD_rejected.value or v2_09_partD_rejected["id"]
        final_summary = (
            f"For {v2_09_packet['label']}, localize {v2_09_bottleneck_label(v2_09_partA_base['bottleneck'])}; "
            f"use {v2_09_partB['decision']} as the regression gate; plan "
            f"{partC_units.value} {v2_09_packet['serving_unit']} with "
            f"{v2_09_pct(v2_09_partC['headroom'])} headroom; defend "
            f"{v2_09_candidate_label(selected_id)} and reject {v2_09_candidate_label(rejected_id)}."
        )
        ledger_design = {
            "lab_id": v2_09_metadata.lab_id,
            "track_id": v2_09_profile.track_id,
            "scenario_id": v2_09_packet["scenario_id"],
            "stakeholder": v2_09_packet["stakeholder"],
            "report_frame": v2_09_packet["report_frame"],
            "partA_predicted_bottleneck": partA_prediction.value or "no_selection",
            "partA_actual_bottleneck": v2_09_partA_base["bottleneck"],
            "partA_selected_first_lever": partA_checkpoint.value or "no_selection",
            "partA_profile_p99_ms": round(v2_09_partA_base["p99_ms"], 4),
            "partA_selected_speedup": round(v2_09_partA_speedup, 4),
            "partB_prediction": partB_prediction.value or "no_selection",
            "partB_samples": partB_samples.value,
            "partB_cv_pct": round(partB_cv.value, 4),
            "partB_candidate_delta_pct": round(partB_delta.value, 4),
            "partB_mde_pct": round(v2_09_partB["mde_pct"], 4),
            "partB_decision": partB_checkpoint.value or v2_09_partB["decision"],
            "partC_prediction": partC_prediction.value or "no_selection",
            "partC_demand_amount": round(v2_09_partC["demand"], 4),
            "partC_serving_units": partC_units.value,
            "partC_utilization_pct": round(v2_09_partC["utilization"] * 100.0, 4),
            "partC_headroom_pct": round(v2_09_partC["headroom"] * 100.0, 4),
            "partC_daily_cost": round(v2_09_partC["daily_cost"], 4),
            "partC_decision": partC_checkpoint.value or "no_selection",
            "partD_prediction": partD_prediction.value or "no_selection",
            "partD_selected_candidate": selected_id,
            "partD_rejected_candidate": rejected_id,
            "partD_computed_selected": v2_09_partD_selected["id"],
            "partD_computed_rejected": v2_09_partD_rejected["id"],
            "partD_risk_budget_pct": partD_risk_budget.value,
            "partD_cost_ceiling": partD_cost_ceiling.value,
            "v2_10_implication": implication_text,
            "final_summary": final_summary,
        }
        if not incomplete:
            ledger.save(track=v2_09_profile.track_id, chapter=v2_09_chapter, design=ledger_design)
        report = build_lab_report(
            v2_09_metadata,
            student_id=v2_09_student_id.value or "",
            track=v2_09_profile.label,
            scenario=v2_09_packet["workload"],
            learning_objectives=(
                "Localize the active bottleneck before optimizing.",
                "Use baseline, variance, and detectability evidence for regressions.",
                "Convert measured performance into capacity headroom and cost.",
                "Defend an optimization trade-off and reject an alternative.",
            ),
            predictions={
                "part_a_bottleneck": partA_prediction.value,
                "part_b_regression": partB_prediction.value,
                "part_c_capacity": partC_prediction.value,
                "part_d_tradeoff": partD_prediction.value,
            },
            knob_settings={
                "part_a_pressure": partA_pressure.value,
                "part_a_lever": partA_lever.value,
                "part_b_samples": partB_samples.value,
                "part_b_cv_pct": partB_cv.value,
                "part_b_delta_pct": partB_delta.value,
                "part_c_demand_multiplier": partC_demand.value,
                "part_c_units": partC_units.value,
                "part_c_optimization": partC_optimization.value,
                "part_d_risk_budget_pct": partD_risk_budget.value,
                "part_d_cost_ceiling": partD_cost_ceiling.value,
            },
            binding_constraints={
                "part_a_bottleneck": v2_09_bottleneck_label(v2_09_partA_base["bottleneck"]),
                "part_b_decision": v2_09_partB["decision"],
                "part_c_failures": v2_09_partC["failures"],
                "part_d_selected_failures": v2_09_partD_selected["failures"],
            },
            decisions={
                "part_a_checkpoint": partA_checkpoint.value,
                "part_b_checkpoint": partB_checkpoint.value,
                "part_c_checkpoint": partC_checkpoint.value,
                "part_d_selected": selected_id,
                "part_d_rejected": rejected_id,
                "v2_10_implication": implication_text,
            },
            reflections={"performance_engineering_memo": final_summary},
            evidence_summary={
                "actual_bottleneck": v2_09_bottleneck_label(v2_09_partA_base["bottleneck"]),
                "mde_pct": round(v2_09_partB["mde_pct"], 3),
                "capacity_headroom_pct": round(v2_09_partC["headroom"] * 100.0, 3),
                "capacity_daily_cost": round(v2_09_partC["daily_cost"], 3),
                "selected_candidate": v2_09_candidate_label(selected_id),
                "rejected_candidate": v2_09_candidate_label(rejected_id),
            },
            final_decision={
                "memo": final_summary,
                "computed_selected": v2_09_partD_selected["label"],
                "computed_rejected": v2_09_partD_rejected["label"],
            },
            big_takeaways=(
                "Performance optimization starts with bottleneck localization.",
                "Regression evidence needs baseline variance and detectability.",
                "Capacity planning is an amount system, not an average-speed claim.",
                "A credible optimization report names the trade-off and rejected alternative.",
            ),
            residual_risk=(
                "Teaching constants must be checked against production traces, current hardware counters, "
                "quality canaries, and real workload distributions before launch."
            ),
            source_trace={
                "book_anchor": v2_09_metadata.book_anchor,
                "chapter_sections": (
                    "The iron law of ML performance",
                    "System Profiling",
                    "Profiling feedback loop",
                    "Measurement at Scale",
                    "The Optimization Playbook",
                    "Fallacies and Pitfalls",
                ),
                "local_solver": "v2_09_profile_result, v2_09_regression_result, v2_09_capacity_result, v2_09_tradeoff_candidates",
                "hardware_ref": v2_09_packet["hardware_ref"],
                "system_ref": v2_09_packet["system_ref"],
            },
            result_snapshot=ledger_design,
            incomplete_fields=tuple(incomplete),
        )
        status_kind = "success" if not incomplete else "warn"
        status_text = "Saved to Design Ledger" if not incomplete else "Complete all prediction and checkpoint controls to save."
        return mo.vstack([
            mo.md("## Synthesis - Performance Engineering Memo"),
            v2_09_student_id,
            v2_09_next_implication,
            mo.callout(mo.md(f"**Memo summary:** {final_summary}  \n\n**V2-10 implication:** {implication_text}"), kind=status_kind),
            mo.callout(mo.md(status_text), kind=status_kind),
            v2_09_table(
                ("Memo field", "Evidence", "Report decision"),
                (
                    ("Bottleneck", v2_09_bottleneck_label(v2_09_partA_base["bottleneck"]), v2_09_optimization_label(partA_checkpoint.value or v2_09_partA_best_lever)),
                    ("Regression", f"MDE {v2_09_partB['mde_pct']:.1f}%, decision {v2_09_partB['decision']}", partB_checkpoint.value or "not selected"),
                    ("Capacity", f"{v2_09_pct(v2_09_partC['headroom'])} headroom, {v2_09_money(v2_09_partC['daily_cost'])}/day", partC_checkpoint.value or "not selected"),
                    ("Trade-off", f"computed select {v2_09_partD_selected['label']}; reject {v2_09_partD_rejected['label']}", f"student select {v2_09_candidate_label(selected_id)}; reject {v2_09_candidate_label(rejected_id)}"),
                    ("Next lab", implication_text, "carry into V2-10 inference"),
                ),
            ),
            report_export_panel(report),
        ])

    tabs = mo.ui.tabs({
        "Opening": v2_09_opening(),
        "Part A: Bottleneck Localization": build_part_a(),
        "Part B: Regression Evidence": build_part_b(),
        "Part C: Capacity Planning": build_part_c(),
        "Part D: Trade-off Report": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    mo,
    v2_09_bottleneck_label,
    v2_09_metadata,
    v2_09_money,
    v2_09_packet,
    v2_09_partA_base,
    v2_09_partC,
    v2_09_partD_selected,
    v2_09_profile,
    v2_09_pct,
):
    _status = "PASS" if v2_09_partC["feasible"] and v2_09_partD_selected["feasible"] else "WATCH"
    _status_color = COLORS["GreenLine"] if _status == "PASS" else COLORS["OrangeLine"]
    mo.Html(f"""
    <div class="lab-hud">
      <span class="hud-label">LAB</span>
      <span class="hud-value">{v2_09_metadata.lab_id}</span>
      <span class="hud-label">TRACK</span>
      <span class="hud-value">{v2_09_profile.label}</span>
      <span class="hud-label">BOTTLENECK</span>
      <span class="hud-value">{v2_09_bottleneck_label(v2_09_partA_base['bottleneck'])}</span>
      <span class="hud-label">HEADROOM</span>
      <span class="hud-value">{v2_09_pct(v2_09_partC['headroom'])}</span>
      <span class="hud-label">COST</span>
      <span class="hud-value">{v2_09_money(v2_09_partC['daily_cost'])}/day</span>
      <span class="hud-label">SLO</span>
      <span class="hud-value">{v2_09_packet['slo_ms']:.0f} ms</span>
      <span style="flex:1;"></span>
      <span class="hud-label">STATUS</span>
      <span style="color:{_status_color}; font-family:var(--font-mono);">{_status}</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
