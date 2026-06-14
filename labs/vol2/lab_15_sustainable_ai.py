import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# -----------------------------------------------------------------------------
# LAB V2-15: THE CARBON BUDGET
#
# Chapter invariant: sustainable AI is an amount system. Energy, carbon
# intensity, utilization, embodied carbon, quality, latency, cost, reliability,
# and governance all have to fit the selected track's operating envelope.
#
# Packet modules:
#   Part A - Carbon Is An Amount Stack
#   Part B - Placement And Utilization Change The Carbon Bill
#   Part C - Mitigation Must Preserve Guardrails
#   Part D - Carbon-Aware Policy Is A Guardrail Bundle
#   Synthesis
# -----------------------------------------------------------------------------


@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    import html
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
        LAB_CSS,
        apply_plotly_theme,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        html,
        ledger,
        math,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v2_15_metadata = get_lab_metadata("vol2/lab_15_sustainable_ai.py")
    v2_15_chapter = 15
    return v2_15_chapter, v2_15_metadata


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_15_track_picker = track_selector(default=_default_track)
    v2_15_track_picker
    return (v2_15_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v2_15_track_picker,
):
    v2_15_track_id = v2_15_track_picker.value
    v2_15_profile = get_track_profile(v2_15_track_id)
    v2_15_variant = get_lab_track_variant("v2_15_carbon_budget", v2_15_profile.track_id)
    v2_15_hardware = resolve_mlsysim_ref(v2_15_variant.hardware_ref)
    v2_15_model = resolve_mlsysim_ref(v2_15_variant.model_ref)
    v2_15_system = resolve_mlsysim_ref(v2_15_variant.system_ref) if v2_15_variant.system_ref else None
    return (
        v2_15_hardware,
        v2_15_model,
        v2_15_profile,
        v2_15_system,
        v2_15_track_id,
        v2_15_variant,
    )


@app.cell
def _(COLORS, html, math, mo):
    def v2_15_color(name, fallback):
        return COLORS.get(name, fallback)

    def v2_15_qty(value, unit, default=0.0):
        try:
            return float(value.to(unit).magnitude)
        except Exception:
            try:
                return float(value)
            except Exception:
                return float(default)

    def v2_15_model_params_m(model):
        return v2_15_qty(getattr(model, "parameters", 0.0), "param", 0.0) / 1_000_000

    def v2_15_model_gflops(model):
        return v2_15_qty(getattr(model, "inference_flops", 0.0), "flop", 0.0) / 1_000_000_000

    def v2_15_num(value, digits=1):
        if not math.isfinite(float(value)):
            return "not feasible"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:,.1f}"
        return f"{value:,.{digits}f}"

    def v2_15_pct(value, digits=0):
        return f"{value * 100:.{digits}f}%"

    def v2_15_status(ok):
        return "PASS" if ok else "FAIL"

    def v2_15_status_html(ok):
        color = v2_15_color("GreenLine", "#047857") if ok else v2_15_color("RedLine", "#b42318")
        bg = v2_15_color("GreenLL", "#ecfdf3") if ok else v2_15_color("RedLL", "#fef3f2")
        return (
            f"<span style='display:inline-block; min-width:54px; text-align:center; "
            f"border:1px solid {color}; background:{bg}; color:{color}; border-radius:999px; "
            "padding:2px 8px; font-size:0.72rem; font-weight:800;'>"
            f"{v2_15_status(ok)}</span>"
        )

    def v2_15_table(headers, rows):
        head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
        body_rows = []
        for row in rows:
            cells = "".join(f"<td>{cell}</td>" for cell in row)
            body_rows.append(f"<tr>{cells}</tr>")
        return mo.Html(
            f"""
<div style="overflow-x:auto; margin:12px 0;">
  <table style="width:100%; border-collapse:collapse; font-size:0.86rem;">
    <thead>
      <tr style="background:{v2_15_color('Surface2', '#f8fafc')}; color:{v2_15_color('Text', '#1f2937')};">
        {head}
      </tr>
    </thead>
    <tbody>{"".join(body_rows)}</tbody>
  </table>
</div>
<style>
  table td, table th {{
    border:1px solid {v2_15_color('Border', '#d9dee8')};
    padding:8px 10px;
    text-align:left;
    vertical-align:top;
  }}
</style>
"""
        )

    def v2_15_metric_card(label, value, subvalue="", color=None):
        accent = color or v2_15_color("BlueLine", "#2563eb")
        return mo.Html(
            f"""
<div style="padding:15px 17px; border:1px solid {v2_15_color('Border', '#d9dee8')};
            border-radius:8px; min-width:150px; text-align:center; background:white;">
  <div style="color:{v2_15_color('TextMuted', '#64748b')}; font-size:0.76rem;
              font-weight:800; text-transform:uppercase;">{html.escape(label)}</div>
  <div style="font-size:1.55rem; font-weight:850; color:{accent};
              font-family:ui-monospace, SFMono-Regular, Consolas, monospace; line-height:1.35;">
    {html.escape(str(value))}
  </div>
  <div style="font-size:0.72rem; color:{v2_15_color('TextMuted', '#64748b')};">
    {html.escape(str(subvalue))}
  </div>
</div>
"""
        )

    def v2_15_part_banner(letter, title, why, color):
        return mo.Html(
            f"""
<div style="margin:18px 0 14px 0;">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="background:{color}; color:white; border-radius:50%; width:34px; height:34px;
                display:inline-flex; align-items:center; justify-content:center; font-size:0.92rem;
                font-weight:850; flex-shrink:0;">{letter}</div>
    <div style="flex:1; height:2px; background:{v2_15_color('Border', '#d9dee8')};"></div>
    <div style="font-size:0.72rem; font-weight:800; color:{v2_15_color('TextMuted', '#64748b')};
                text-transform:uppercase; letter-spacing:0.12em;">Part {letter}</div>
  </div>
  <div style="font-size:1.48rem; font-weight:850; color:{v2_15_color('Text', '#172033')};
              margin-top:8px; line-height:1.2;">{html.escape(title)}</div>
  <div style="color:{v2_15_color('TextSec', '#475467')}; font-size:0.93rem; margin-top:6px;
              line-height:1.55; max-width:820px;">{html.escape(why)}</div>
</div>
"""
        )

    def v2_15_reveal_card(title, prediction, actual, detail, kind="info"):
        palette = {
            "success": (v2_15_color("GreenLine", "#047857"), v2_15_color("GreenLL", "#ecfdf3")),
            "warn": (v2_15_color("OrangeLine", "#d97706"), v2_15_color("OrangeLL", "#fffbeb")),
            "danger": (v2_15_color("RedLine", "#b42318"), v2_15_color("RedLL", "#fef3f2")),
            "info": (v2_15_color("BlueLine", "#2563eb"), v2_15_color("BlueLL", "#eff6ff")),
        }
        color, background = palette.get(kind, palette["info"])
        return mo.Html(
            f"""
<div style="background:{background}; border:1px solid {color}; border-left:5px solid {color};
            border-radius:8px; padding:14px 18px; margin:12px 0;">
  <div style="font-size:0.82rem; font-weight:850; color:{color};
              text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
    {html.escape(title)}
  </div>
  <div style="font-size:0.9rem; color:{v2_15_color('Text', '#172033')}; line-height:1.65;">
    You predicted <strong>{html.escape(str(prediction))}</strong>. Actual:
    <strong>{html.escape(str(actual))}</strong>. {html.escape(str(detail))}
  </div>
</div>
"""
        )

    def v2_15_failure_card(active, title, detail, recovery):
        if active:
            return mo.callout(
                mo.md(f"**{title}**  \n{detail}  \n\nRecovery path: {recovery}"),
                kind="danger",
            )
        return mo.callout(
            mo.md(f"**Boundary recovered: {title}**  \nCurrent settings pass. Watch: {detail}"),
            kind="success",
        )

    def v2_15_math_peek(title, body):
        return mo.accordion({title: mo.md(body)})

    return (
        v2_15_color,
        v2_15_failure_card,
        v2_15_math_peek,
        v2_15_metric_card,
        v2_15_model_gflops,
        v2_15_model_params_m,
        v2_15_num,
        v2_15_part_banner,
        v2_15_pct,
        v2_15_qty,
        v2_15_reveal_card,
        v2_15_status,
        v2_15_status_html,
        v2_15_table,
    )


@app.cell
def _(v2_15_model_gflops, v2_15_model_params_m, v2_15_qty):
    def v2_15_track_packet(profile, variant, hardware, model, system):
        tdp_w = v2_15_qty(getattr(hardware, "tdp", 0.0), "watt", 1.0)
        battery_wh = v2_15_qty(getattr(hardware, "battery_capacity", 0.0), "watt_hour", 0.0)
        system_units = getattr(system, "total_accelerators", None) if system is not None else None
        embodied_registry = getattr(hardware, "embodied_carbon_kg", None)

        base = {
            "iphone": {
                "label": "iPhone local assistant feature",
                "workload_unit": "local assistant sessions/day",
                "fleet_units": 25000,
                "active_hours_day": 0.50,
                "avg_power_w": max(0.8, min(tdp_w * 0.42, 2.2)),
                "idle_power_w": 0.12,
                "pue": 1.00,
                "baseline_region": "US_Avg",
                "embodied_kg_per_unit": 70.0,
                "embodied_lifetime_years": 3.0,
                "energy_budget_kwh_day": max(1.0, 25000 * battery_wh * 0.03 / 1000.0),
                "carbon_budget_kg_day": 18.0,
                "embodied_budget_kg_day": 1250.0,
                "latency_slo_ms": 180.0,
                "freshness_delay_limit_h": 2.0,
                "quality_floor_pct": 86.0,
                "reliability_floor_pct": 95.0,
                "cost_budget_day": 120.0,
                "service_name": "interactive local feature",
                "governance_need": "privacy-safe battery and thermal audit",
                "failure_story": "local sustainability fails when battery drain or embodied fleet carbon is hidden by per-session averages",
                "v2_16_implication": "Responsible AI review must include who pays the battery, privacy, and accessibility cost of the policy.",
            },
            "oura_ring": {
                "label": "Oura Ring always-on sensing",
                "workload_unit": "sensing windows/day",
                "fleet_units": 100000,
                "active_hours_day": 6.0,
                "avg_power_w": max(0.003, min(tdp_w * 0.30, 0.008)),
                "idle_power_w": 0.0008,
                "pue": 1.00,
                "baseline_region": "US_Avg",
                "embodied_kg_per_unit": 3.0,
                "embodied_lifetime_years": 3.0,
                "energy_budget_kwh_day": max(0.5, 100000 * battery_wh * 0.45 / 1000.0),
                "carbon_budget_kg_day": 2.5,
                "embodied_budget_kg_day": 210.0,
                "latency_slo_ms": 500.0,
                "freshness_delay_limit_h": 8.0,
                "quality_floor_pct": 80.0,
                "reliability_floor_pct": 94.0,
                "cost_budget_day": 55.0,
                "service_name": "battery-safe sensing cadence",
                "governance_need": "health-adjacent sensing, comfort, and battery review",
                "failure_story": "wearable sustainability fails when duty-cycle savings erase signal quality or battery life",
                "v2_16_implication": "Responsible AI review must account for false alerts, missed signals, comfort, and battery trade-offs.",
            },
            "robotaxi": {
                "label": "RoboTaxi noncritical perception replay",
                "workload_unit": "fleet replay windows/day",
                "fleet_units": 400,
                "active_hours_day": 9.0,
                "avg_power_w": max(25.0, min(tdp_w * 0.75, 55.0)),
                "idle_power_w": 9.0,
                "pue": 1.00,
                "baseline_region": "US_Avg",
                "embodied_kg_per_unit": 250.0,
                "embodied_lifetime_years": 5.0,
                "energy_budget_kwh_day": 145.0,
                "carbon_budget_kg_day": 65.0,
                "embodied_budget_kg_day": 70.0,
                "latency_slo_ms": 55.0,
                "freshness_delay_limit_h": 0.25,
                "quality_floor_pct": 92.0,
                "reliability_floor_pct": 99.0,
                "cost_budget_day": 240.0,
                "service_name": "safety-bounded fleet replay",
                "governance_need": "safety-case traceability and noncritical deferral review",
                "failure_story": "autonomy sustainability fails if carbon savings weaken safety margin or replay freshness",
                "v2_16_implication": "Responsible AI must justify which work is deferrable without hiding rare-event safety risk.",
            },
            "cloud_fleet": {
                "label": "Cloud Fleet inference and evaluation service",
                "workload_unit": "service batches/day",
                "fleet_units": int(system_units or 64),
                "active_hours_day": 20.0,
                "avg_power_w": max(250.0, min(tdp_w * 0.55, 420.0)),
                "idle_power_w": 65.0,
                "pue": 1.12,
                "baseline_region": "US_Avg",
                "embodied_kg_per_unit": float(embodied_registry or 164.0),
                "embodied_lifetime_years": 4.0,
                "energy_budget_kwh_day": 560.0,
                "carbon_budget_kg_day": 190.0,
                "embodied_budget_kg_day": 10.0,
                "latency_slo_ms": 130.0,
                "freshness_delay_limit_h": 4.0,
                "quality_floor_pct": 88.0,
                "reliability_floor_pct": 97.0,
                "cost_budget_day": 520.0,
                "service_name": "SLA-bound production service",
                "governance_need": "carbon cap, SLA, quality canary, and carbon-price review",
                "failure_story": "cloud sustainability fails when high utilization or cheap dirty regions break carbon or p99 budgets",
                "v2_16_implication": "Responsible AI must audit carbon caps alongside subgroup quality, appealability, and explanation overhead.",
            },
        }[profile.track_id]

        defaults = variant.defaults
        base["quality_floor_pct"] = float(defaults.get("quality_floor_pct", base["quality_floor_pct"]))
        base["latency_slo_ms"] = float(defaults.get("latency_budget_ms", base["latency_slo_ms"]))
        base["cost_budget_day"] = float(defaults.get("cost_budget", base["cost_budget_day"]))
        base["hardware_name"] = getattr(hardware, "name", variant.hardware_ref)
        base["model_name"] = getattr(model, "name", variant.model_ref)
        base["model_params_m"] = v2_15_model_params_m(model)
        base["model_gflops"] = v2_15_model_gflops(model)
        base["hardware_ref"] = variant.hardware_ref
        base["model_ref"] = variant.model_ref
        base["system_ref"] = variant.system_ref or "device fleet"
        base["track_id"] = profile.track_id
        base["track_label"] = profile.label
        base["stakeholder"] = variant.stakeholder
        base["scenario"] = variant.workload_summary
        base["objective"] = variant.objective
        base["source_policy"] = profile.source_policy
        return base

    def v2_15_region_catalog():
        return {
            "US_Avg": {
                "label": "US average grid",
                "carbon_g_kwh": 429.0,
                "pue": 1.12,
                "cost_usd_kwh": 0.105,
                "latency_add_ms": 0.0,
                "reliability_pct": 98.0,
            },
            "Quebec": {
                "label": "Quebec hydro-heavy grid",
                "carbon_g_kwh": 20.0,
                "pue": 1.06,
                "cost_usd_kwh": 0.073,
                "latency_add_ms": 22.0,
                "reliability_pct": 97.0,
            },
            "Iowa": {
                "label": "Iowa mixed grid",
                "carbon_g_kwh": 680.0,
                "pue": 1.12,
                "cost_usd_kwh": 0.075,
                "latency_add_ms": 12.0,
                "reliability_pct": 98.0,
            },
            "Poland": {
                "label": "Poland coal-heavy grid",
                "carbon_g_kwh": 820.0,
                "pue": 1.58,
                "cost_usd_kwh": 0.090,
                "latency_add_ms": 45.0,
                "reliability_pct": 96.0,
            },
        }

    return v2_15_region_catalog, v2_15_track_packet


@app.cell
def _(v2_15_hardware, v2_15_model, v2_15_profile, v2_15_system, v2_15_track_packet, v2_15_variant):
    v2_15_packet = v2_15_track_packet(
        v2_15_profile,
        v2_15_variant,
        v2_15_hardware,
        v2_15_model,
        v2_15_system,
    )
    return (v2_15_packet,)


@app.cell
def _(math, v2_15_region_catalog):
    def v2_15_part_a_result(packet, workload_mult, utilization_pct):
        utilization = max(0.20, utilization_pct / 100.0)
        active_energy = (
            packet["fleet_units"]
            * packet["avg_power_w"]
            * packet["active_hours_day"]
            * workload_mult
            / 1000.0
        )
        idle_hours = max(0.0, 24.0 - packet["active_hours_day"])
        idle_energy = packet["fleet_units"] * packet["idle_power_w"] * idle_hours * (1.0 - utilization) / 1000.0
        it_energy = active_energy + idle_energy
        facility_energy = it_energy * packet["pue"]
        region = v2_15_region_catalog()[packet["baseline_region"]]
        operational_kg = facility_energy * region["carbon_g_kwh"] / 1000.0
        embodied_kg_day = (
            packet["fleet_units"]
            * packet["embodied_kg_per_unit"]
            / max(1.0, packet["embodied_lifetime_years"] * 365.0)
        )
        lifecycle_kg = operational_kg + embodied_kg_day
        ratios = {
            "energy": facility_energy / max(1e-9, packet["energy_budget_kwh_day"]),
            "carbon intensity": operational_kg / max(1e-9, packet["carbon_budget_kg_day"]),
            "embodied carbon": embodied_kg_day / max(1e-9, packet["embodied_budget_kg_day"]),
        }
        if packet["track_id"] in ("iphone", "oura_ring"):
            battery_wh = packet["energy_budget_kwh_day"] * 1000.0 / max(1, packet["fleet_units"])
            actual_wh = packet["avg_power_w"] * packet["active_hours_day"] * workload_mult
            ratios["device energy"] = actual_wh / max(1e-9, battery_wh)
        binding = max(ratios, key=ratios.get)
        return {
            "active_energy_kwh": active_energy,
            "idle_energy_kwh": idle_energy,
            "it_energy_kwh": it_energy,
            "facility_energy_kwh": facility_energy,
            "operational_kg": operational_kg,
            "embodied_kg_day": embodied_kg_day,
            "lifecycle_kg_day": lifecycle_kg,
            "ratios": ratios,
            "binding": binding,
            "fails": any(value > 1.0 for value in ratios.values()),
            "region": region,
            "utilization": utilization,
        }

    def v2_15_part_b_result(packet, workload_mult, utilization_pct, region_id, schedule_id):
        regions = v2_15_region_catalog()
        region = regions[region_id]
        utilization = max(0.20, utilization_pct / 100.0)
        schedule = {
            "immediate": {
                "label": "Immediate serving",
                "carbon_multiplier": 1.00,
                "delay_h": 0.0,
                "latency_multiplier": 1.00,
                "reliability_penalty": 0.0,
                "energy_multiplier": 1.00,
            },
            "clean_window": {
                "label": "Wait for a cleaner grid window",
                "carbon_multiplier": 0.60,
                "delay_h": 6.0,
                "latency_multiplier": 1.08,
                "reliability_penalty": 1.0,
                "energy_multiplier": 1.01,
            },
            "region_shift": {
                "label": "Route flexible work to selected region",
                "carbon_multiplier": 0.92,
                "delay_h": 1.0,
                "latency_multiplier": 1.00,
                "reliability_penalty": 0.5,
                "energy_multiplier": 1.04,
            },
            "demand_cap": {
                "label": "Cap nonurgent demand",
                "carbon_multiplier": 0.74,
                "delay_h": 0.5,
                "latency_multiplier": 0.88,
                "reliability_penalty": 0.2,
                "energy_multiplier": 0.78,
            },
        }[schedule_id]
        active_energy = (
            packet["fleet_units"]
            * packet["avg_power_w"]
            * packet["active_hours_day"]
            * workload_mult
            / 1000.0
        )
        idle_overhead = 0.38 / utilization
        it_energy = active_energy * (0.72 + idle_overhead) * schedule["energy_multiplier"]
        facility_energy = it_energy * region["pue"]
        carbon_kg = facility_energy * region["carbon_g_kwh"] * schedule["carbon_multiplier"] / 1000.0
        queue_pressure = (utilization * utilization) / max(0.04, 1.0 - utilization)
        p99_ms = (
            packet["latency_slo_ms"]
            * (0.38 + 0.12 * queue_pressure)
            * schedule["latency_multiplier"]
            + region["latency_add_ms"]
        )
        reliability_pct = region["reliability_pct"] - schedule["reliability_penalty"] - max(0.0, utilization - 0.82) * 18.0
        delay_ok = schedule["delay_h"] <= packet["freshness_delay_limit_h"]
        p99_ok = p99_ms <= packet["latency_slo_ms"]
        reliability_ok = reliability_pct >= packet["reliability_floor_pct"]
        carbon_ok = carbon_kg <= packet["carbon_budget_kg_day"]
        service_ok = p99_ok and delay_ok and reliability_ok
        binding_scores = {
            "carbon": carbon_kg / max(1e-9, packet["carbon_budget_kg_day"]),
            "p99/freshness": max(
                p99_ms / max(1e-9, packet["latency_slo_ms"]),
                schedule["delay_h"] / max(1e-9, packet["freshness_delay_limit_h"]),
            ),
            "reliability": packet["reliability_floor_pct"] / max(1e-9, reliability_pct),
            "utilization": utilization / 0.85,
        }
        binding = max(binding_scores, key=binding_scores.get)
        return {
            "region_id": region_id,
            "region": region,
            "schedule_id": schedule_id,
            "schedule": schedule,
            "utilization": utilization,
            "facility_energy_kwh": facility_energy,
            "carbon_kg": carbon_kg,
            "p99_ms": p99_ms,
            "reliability_pct": reliability_pct,
            "delay_h": schedule["delay_h"],
            "carbon_ok": carbon_ok,
            "p99_ok": p99_ok,
            "delay_ok": delay_ok,
            "reliability_ok": reliability_ok,
            "service_ok": service_ok,
            "binding": binding,
            "binding_scores": binding_scores,
            "cost_usd": facility_energy * region["cost_usd_kwh"],
        }

    def v2_15_strategy_candidates(packet, part_b, intensity_pct, governance_ack):
        intensity = intensity_pct / 100.0
        base_quality = float(packet["quality_floor_pct"]) + 5.0
        base_cost = part_b["cost_usd"] + 0.20 * packet["cost_budget_day"]
        base_latency = part_b["p99_ms"]
        base_reliability = part_b["reliability_pct"]
        base_carbon = part_b["carbon_kg"]
        base_embodied = (
            packet["fleet_units"]
            * packet["embodied_kg_per_unit"]
            / max(1.0, packet["embodied_lifetime_years"] * 365.0)
        )
        specs = {
            "model_efficiency": {
                "label": "Model efficiency",
                "energy_mult": 1.0 - 0.42 * intensity,
                "carbon_mult": 1.0 - 0.42 * intensity,
                "embodied_mult": 1.0,
                "quality_delta": -2.0 * intensity,
                "latency_mult": 1.0 - 0.20 * intensity,
                "cost_mult": 1.0 - 0.20 * intensity,
                "reliability_delta": -0.4 * intensity,
                "governance_required": True,
                "rejected_reason": "quality regression can erase the sustainability win",
            },
            "carbon_aware_schedule": {
                "label": "Carbon-aware schedule",
                "energy_mult": 1.0,
                "carbon_mult": 1.0 - 0.45 * intensity,
                "embodied_mult": 1.0,
                "quality_delta": -0.2 * intensity,
                "latency_mult": 1.0 + 0.18 * intensity,
                "cost_mult": 1.0 + 0.06 * intensity,
                "reliability_delta": -1.2 * intensity,
                "governance_required": False,
                "rejected_reason": "freshness or p99 can fail when flexible work is delayed",
            },
            "utilization_consolidation": {
                "label": "Utilization consolidation",
                "energy_mult": 1.0 - 0.30 * intensity,
                "carbon_mult": 1.0 - 0.30 * intensity,
                "embodied_mult": 1.0,
                "quality_delta": 0.0,
                "latency_mult": 1.0 + 0.24 * intensity,
                "cost_mult": 1.0 - 0.18 * intensity,
                "reliability_delta": -1.8 * intensity,
                "governance_required": False,
                "rejected_reason": "high utilization can turn saved idle power into p99 or reliability risk",
            },
            "lifecycle_extension": {
                "label": "Extend hardware lifetime",
                "energy_mult": 1.0 + 0.08 * intensity,
                "carbon_mult": 1.0 + 0.08 * intensity,
                "embodied_mult": 1.0 - 0.35 * intensity,
                "quality_delta": -0.5 * intensity,
                "latency_mult": 1.0 + 0.07 * intensity,
                "cost_mult": 1.0 - 0.10 * intensity,
                "reliability_delta": -1.0 * intensity,
                "governance_required": True,
                "rejected_reason": "older hardware can trade embodied savings for reliability and efficiency loss",
            },
            "demand_governance": {
                "label": "Demand governance",
                "energy_mult": 1.0 - 0.34 * intensity,
                "carbon_mult": 1.0 - 0.34 * intensity,
                "embodied_mult": 1.0,
                "quality_delta": -0.8 * intensity,
                "latency_mult": 0.90,
                "cost_mult": 1.0 - 0.28 * intensity,
                "reliability_delta": 0.4 * intensity,
                "governance_required": True,
                "rejected_reason": "usage caps need accountable policy because they decide who receives less service",
            },
        }
        candidates = []
        for strategy_id, spec in specs.items():
            carbon_kg = base_carbon * spec["carbon_mult"]
            embodied_kg = base_embodied * spec["embodied_mult"]
            quality_pct = base_quality + spec["quality_delta"]
            latency_ms = base_latency * spec["latency_mult"]
            cost_usd = base_cost * spec["cost_mult"]
            reliability_pct = base_reliability + spec["reliability_delta"]
            governance_ok = (not spec["governance_required"]) or bool(governance_ack)
            checks = {
                "carbon": carbon_kg <= packet["carbon_budget_kg_day"],
                "quality": quality_pct >= packet["quality_floor_pct"],
                "latency": latency_ms <= packet["latency_slo_ms"],
                "cost": cost_usd <= packet["cost_budget_day"],
                "reliability": reliability_pct >= packet["reliability_floor_pct"],
                "governance": governance_ok,
            }
            failed = tuple(name for name, ok in checks.items() if not ok)
            if failed:
                binding = failed[0]
            else:
                margins = {
                    "carbon": carbon_kg / max(1e-9, packet["carbon_budget_kg_day"]),
                    "quality": packet["quality_floor_pct"] / max(1e-9, quality_pct),
                    "latency": latency_ms / max(1e-9, packet["latency_slo_ms"]),
                    "cost": cost_usd / max(1e-9, packet["cost_budget_day"]),
                    "reliability": packet["reliability_floor_pct"] / max(1e-9, reliability_pct),
                }
                binding = max(margins, key=margins.get)
            candidates.append({
                "strategy_id": strategy_id,
                "label": spec["label"],
                "carbon_kg": carbon_kg,
                "embodied_kg": embodied_kg,
                "lifecycle_kg": carbon_kg + embodied_kg,
                "quality_pct": quality_pct,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "reliability_pct": reliability_pct,
                "governance_required": spec["governance_required"],
                "governance_ok": governance_ok,
                "checks": checks,
                "passes": all(checks.values()),
                "failed": failed,
                "binding": binding,
                "rejected_reason": spec["rejected_reason"],
            })
        recommended = min(
            (candidate for candidate in candidates if candidate["passes"]),
            key=lambda candidate: candidate["lifecycle_kg"],
            default=min(candidates, key=lambda candidate: len(candidate["failed"])),
        )
        rejected = max(
            (candidate for candidate in candidates if candidate["strategy_id"] != recommended["strategy_id"]),
            key=lambda candidate: (not candidate["passes"], candidate["lifecycle_kg"]),
        )
        return {"candidates": candidates, "recommended": recommended, "rejected": rejected}

    def v2_15_policy_candidates(packet, part_b, strategy_packet, carbon_price):
        recommended = strategy_packet["recommended"]
        base = {
            "carbon_kg": part_b["carbon_kg"],
            "lifecycle_kg": part_b["carbon_kg"]
            + packet["fleet_units"] * packet["embodied_kg_per_unit"] / max(1.0, packet["embodied_lifetime_years"] * 365.0),
            "latency_ms": part_b["p99_ms"],
            "quality_pct": float(packet["quality_floor_pct"]) + 5.0,
            "reliability_pct": part_b["reliability_pct"],
            "cost_usd": part_b["cost_usd"] + 0.18 * packet["cost_budget_day"],
        }
        specs = {
            "throughput_first": {
                "label": "Max-throughput baseline",
                "carbon_mult": 1.25,
                "lifecycle_mult": 1.20,
                "latency_mult": 1.24,
                "quality_delta": 1.0,
                "reliability_delta": -1.4,
                "cost_mult": 1.05,
                "governance": False,
                "memo": "rejects carbon accounting until after launch",
            },
            "efficiency_guardrail": {
                "label": "Efficiency with quality canary",
                "carbon_mult": max(0.55, recommended["carbon_kg"] / max(1e-9, part_b["carbon_kg"])),
                "lifecycle_mult": 0.78,
                "latency_mult": 0.92,
                "quality_delta": -0.6,
                "reliability_delta": -0.2,
                "cost_mult": 0.86,
                "governance": True,
                "memo": "uses model efficiency but keeps quality and rollback checks",
            },
            "carbon_guardrail": {
                "label": "Carbon guardrail scheduler",
                "carbon_mult": 0.50 if packet["track_id"] == "cloud_fleet" else 0.62,
                "lifecycle_mult": 0.68,
                "latency_mult": 1.05,
                "quality_delta": -0.4,
                "reliability_delta": -0.4,
                "cost_mult": 0.95,
                "governance": True,
                "memo": "uses region, time, and admission guardrails before consuming carbon budget",
            },
            "lifecycle_guarded": {
                "label": "Lifecycle guardrail policy",
                "carbon_mult": 0.64,
                "lifecycle_mult": 0.58 if packet["track_id"] in ("iphone", "oura_ring") else 0.74,
                "latency_mult": 1.02,
                "quality_delta": -0.5,
                "reliability_delta": -0.6,
                "cost_mult": 0.92,
                "governance": True,
                "memo": "combines carbon-aware operation with hardware lifetime and reuse evidence",
            },
        }
        candidates = []
        for policy_id, spec in specs.items():
            carbon_kg = base["carbon_kg"] * spec["carbon_mult"]
            lifecycle_kg = base["lifecycle_kg"] * spec["lifecycle_mult"]
            latency_ms = base["latency_ms"] * spec["latency_mult"]
            quality_pct = base["quality_pct"] + spec["quality_delta"]
            reliability_pct = base["reliability_pct"] + spec["reliability_delta"]
            cost_usd = base["cost_usd"] * spec["cost_mult"] + lifecycle_kg / 1000.0 * carbon_price
            rebound_ok = policy_id != "throughput_first"
            checks = {
                "carbon": carbon_kg <= packet["carbon_budget_kg_day"],
                "quality": quality_pct >= packet["quality_floor_pct"],
                "latency": latency_ms <= packet["latency_slo_ms"],
                "cost": cost_usd <= packet["cost_budget_day"],
                "reliability": reliability_pct >= packet["reliability_floor_pct"],
                "governance": spec["governance"],
                "rebound": rebound_ok,
            }
            failed = tuple(name for name, ok in checks.items() if not ok)
            binding = failed[0] if failed else max(
                {
                    "carbon": carbon_kg / max(1e-9, packet["carbon_budget_kg_day"]),
                    "quality": packet["quality_floor_pct"] / max(1e-9, quality_pct),
                    "latency": latency_ms / max(1e-9, packet["latency_slo_ms"]),
                    "cost": cost_usd / max(1e-9, packet["cost_budget_day"]),
                    "reliability": packet["reliability_floor_pct"] / max(1e-9, reliability_pct),
                },
                key=lambda key: {
                    "carbon": carbon_kg / max(1e-9, packet["carbon_budget_kg_day"]),
                    "quality": packet["quality_floor_pct"] / max(1e-9, quality_pct),
                    "latency": latency_ms / max(1e-9, packet["latency_slo_ms"]),
                    "cost": cost_usd / max(1e-9, packet["cost_budget_day"]),
                    "reliability": packet["reliability_floor_pct"] / max(1e-9, reliability_pct),
                }[key],
            )
            candidates.append({
                "policy_id": policy_id,
                "label": spec["label"],
                "memo": spec["memo"],
                "carbon_kg": carbon_kg,
                "lifecycle_kg": lifecycle_kg,
                "latency_ms": latency_ms,
                "quality_pct": quality_pct,
                "cost_usd": cost_usd,
                "reliability_pct": reliability_pct,
                "checks": checks,
                "passes": all(checks.values()),
                "failed": failed,
                "binding": binding,
            })
        launchable = [candidate for candidate in candidates if candidate["passes"]]
        selected_default = min(launchable, key=lambda candidate: candidate["lifecycle_kg"]) if launchable else min(
            candidates,
            key=lambda candidate: len(candidate["failed"]),
        )
        rejected = max(
            (candidate for candidate in candidates if candidate["policy_id"] != selected_default["policy_id"]),
            key=lambda candidate: (not candidate["passes"], candidate["lifecycle_kg"]),
        )
        return {"candidates": candidates, "recommended": selected_default, "rejected": rejected}

    def v2_15_prediction_key_for_part_b(result):
        if not result["carbon_ok"]:
            return "carbon"
        if not result["p99_ok"] or not result["delay_ok"]:
            return "service"
        if not result["reliability_ok"]:
            return "reliability"
        return "utilization"

    return (
        v2_15_part_a_result,
        v2_15_part_b_result,
        v2_15_policy_candidates,
        v2_15_prediction_key_for_part_b,
        v2_15_strategy_candidates,
    )


@app.cell(hide_code=True)
def _(mo, v2_15_packet):
    partA_prediction = mo.ui.radio(
        options={
            "Device/facility energy becomes the first budget": "energy",
            "Grid carbon intensity dominates the result": "carbon intensity",
            "Embodied carbon from hardware dominates": "embodied carbon",
            "Service quality or latency will bind first": "service guardrail",
        },
        label=f"Before measuring {v2_15_packet['label']}, which sustainability amount do you expect to bind?",
    )
    partA_workload = mo.ui.slider(
        start=0.50,
        stop=2.50,
        value=1.00,
        step=0.05,
        label="Workload scale (x baseline)",
    )
    partA_utilization = mo.ui.slider(
        start=25,
        stop=95,
        value=62,
        step=1,
        label="Average useful utilization (%)",
    )
    partA_checkpoint = mo.ui.radio(
        options={
            "Energy budget is the carry-forward amount": "energy",
            "Carbon intensity is the carry-forward amount": "carbon intensity",
            "Embodied carbon is the carry-forward amount": "embodied carbon",
            "The service guardrail blocks sustainability claims": "service guardrail",
        },
        label="Checkpoint: which amount should the policy carry forward?",
    )
    return partA_checkpoint, partA_prediction, partA_utilization, partA_workload


@app.cell(hide_code=True)
def _(mo):
    partB_prediction = mo.ui.radio(
        options={
            "Carbon budget will still bind": "carbon",
            "Service latency or freshness will bind": "service",
            "Reliability will bind": "reliability",
            "Utilization headroom will be the main lever": "utilization",
        },
        label="After placement and scheduling, which amount do you expect to limit the plan?",
    )
    partB_region = mo.ui.dropdown(
        options={
            "US average grid": "US_Avg",
            "Quebec hydro-heavy grid": "Quebec",
            "Iowa mixed grid": "Iowa",
            "Poland coal-heavy grid": "Poland",
        },
        value="US average grid",
        label="Execution or offload region",
    )
    partB_schedule = mo.ui.dropdown(
        options={
            "Immediate serving": "immediate",
            "Wait for cleaner grid window": "clean_window",
            "Route flexible work to selected region": "region_shift",
            "Cap nonurgent demand": "demand_cap",
        },
        value="Immediate serving",
        label="Scheduling policy",
    )
    partB_utilization = mo.ui.slider(
        start=35,
        stop=96,
        value=72,
        step=1,
        label="Target useful utilization (%)",
    )
    partB_checkpoint = mo.ui.radio(
        options={
            "Move flexible work to a cleaner region": "region",
            "Wait for cleaner hours only for nonurgent work": "schedule",
            "Lower utilization to protect p99 and reliability": "utilization",
            "Cap demand because efficiency savings rebound": "demand",
        },
        label="Checkpoint: what is the next operational lever?",
    )
    return partB_checkpoint, partB_prediction, partB_region, partB_schedule, partB_utilization


@app.cell(hide_code=True)
def _(mo):
    partC_prediction = mo.ui.radio(
        options={
            "Quality regression will reject it": "quality",
            "Latency or freshness will reject it": "latency",
            "Cost will reject it": "cost",
            "Reliability or governance will reject it": "reliability",
            "Carbon will still reject it": "carbon",
        },
        label="Which guardrail is most likely to reject an aggressive mitigation?",
    )
    partC_strategy = mo.ui.dropdown(
        options={
            "Model efficiency": "model_efficiency",
            "Carbon-aware schedule": "carbon_aware_schedule",
            "Utilization consolidation": "utilization_consolidation",
            "Extend hardware lifetime": "lifecycle_extension",
            "Demand governance": "demand_governance",
        },
        value="Model efficiency",
        label="Mitigation strategy",
    )
    partC_intensity = mo.ui.slider(
        start=10,
        stop=100,
        value=65,
        step=5,
        label="Mitigation intensity (%)",
    )
    partC_governance = mo.ui.checkbox(
        value=False,
        label="Attach governance review and validation evidence",
    )
    partC_checkpoint = mo.ui.radio(
        options={
            "Use the selected mitigation as the primary lever": "selected",
            "Use the recommended passing mitigation instead": "recommended",
            "Reject mitigation until a quality canary is added": "quality_canary",
            "Reject mitigation until carbon and demand caps are explicit": "carbon_cap",
        },
        label="Checkpoint: which mitigation should enter the final policy review?",
    )
    return (
        partC_checkpoint,
        partC_governance,
        partC_intensity,
        partC_prediction,
        partC_strategy,
    )


@app.cell(hide_code=True)
def _(mo):
    partD_prediction = mo.ui.radio(
        options={
            "Max-throughput baseline": "throughput_first",
            "Efficiency with quality canary": "efficiency_guardrail",
            "Carbon guardrail scheduler": "carbon_guardrail",
            "Lifecycle guardrail policy": "lifecycle_guarded",
        },
        label="Which policy do you expect to pass all guardrails?",
    )
    partD_policy = mo.ui.dropdown(
        options={
            "Max-throughput baseline": "throughput_first",
            "Efficiency with quality canary": "efficiency_guardrail",
            "Carbon guardrail scheduler": "carbon_guardrail",
            "Lifecycle guardrail policy": "lifecycle_guarded",
        },
        value="Carbon guardrail scheduler",
        label="Selected sustainability policy",
    )
    partD_carbon_price = mo.ui.slider(
        start=0,
        stop=250,
        value=100,
        step=10,
        label="Internal carbon price ($/ton CO2e)",
    )
    partD_checkpoint = mo.ui.radio(
        options={
            "Approve selected policy with guardrails": "approve",
            "Revise selected policy before launch": "revise",
            "Use recommended launchable policy instead": "recommended",
            "Escalate to governance because trade-offs remain unresolved": "escalate",
        },
        label="Checkpoint: what should the launch review do?",
    )
    decision_input = mo.ui.text_area(
        label="Engineering memo note",
        placeholder="One sentence: selected policy, binding amount, rejected alternative, residual risk.",
    )
    return partD_carbon_price, partD_checkpoint, partD_policy, partD_prediction, decision_input


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_arc_context,
    track_context,
    v2_15_metadata,
    v2_15_packet,
    v2_15_profile,
    v2_15_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(
            f"""
<div style="background:linear-gradient(135deg, #0f172a 0%, #1f2937 100%);
            border-radius:16px; padding:32px 40px; margin-bottom:8px; color:white;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
    <div>
      <div style="font-size:0.72rem; font-weight:800; color:#cbd5e1; text-transform:uppercase;
                  letter-spacing:0.14em; margin-bottom:8px;">
        Vol 2 - Lab 15 - Sustainable AI
      </div>
      <div style="font-size:2rem; font-weight:850; line-height:1.15; margin-bottom:10px;">
        The Carbon Budget
      </div>
      <div style="font-size:0.96rem; color:#d1d5db; max-width:760px; line-height:1.6;">
        Sustainability is an amount system. You will measure energy, carbon intensity,
        utilization, embodied carbon, and guardrails before choosing a carbon-aware
        policy for {v2_15_packet['track_label']}.
      </div>
    </div>
    <div style="display:flex; flex-direction:column; gap:8px; flex-shrink:0;">
      <span class="badge badge-info">{v2_15_packet['track_label']}</span>
      <span class="badge badge-info">{v2_15_packet['hardware_ref']}</span>
      <span class="badge badge-info">{v2_15_packet['model_ref']}</span>
      <span class="badge badge-warn">4 Parts + Synthesis</span>
    </div>
  </div>
</div>
"""
        ),
        track_context(v2_15_profile),
        track_arc_context(v2_15_profile, v2_15_metadata.lab_id),
        source_trace(
            {
                "track_id": v2_15_profile.track_id,
                "scenario_id": v2_15_variant.scenario_id,
                "hardware_ref": v2_15_variant.hardware_ref,
                "model_ref": v2_15_variant.model_ref,
                "system_ref": v2_15_variant.system_ref or "device fleet",
                "chapter_sources": (
                    "The Energy Ceiling; Carbon footprint analysis; Geographic and temporal optimization; "
                    "Google 4 Ms; Fallacies and Pitfalls"
                ),
                "notebook_local_helpers": "v2_15_* sustainability amount model",
                "local_assumptions": (
                    "track fleet sizes, non-H100 embodied estimates, electricity prices, "
                    "service budgets, and mitigation multipliers"
                ),
            },
            collapsed=False,
            summary="Registry-backed track and hardware context plus notebook-local sustainability assumptions.",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    build_lab_report,
    decision_input,
    go,
    ledger,
    mo,
    partA_checkpoint,
    partA_prediction,
    partA_utilization,
    partA_workload,
    partB_checkpoint,
    partB_prediction,
    partB_region,
    partB_schedule,
    partB_utilization,
    partC_checkpoint,
    partC_governance,
    partC_intensity,
    partC_prediction,
    partC_strategy,
    partD_carbon_price,
    partD_checkpoint,
    partD_policy,
    partD_prediction,
    report_export_panel,
    source_trace,
    v2_15_chapter,
    v2_15_color,
    v2_15_failure_card,
    v2_15_math_peek,
    v2_15_metadata,
    v2_15_metric_card,
    v2_15_num,
    v2_15_packet,
    v2_15_part_a_result,
    v2_15_part_b_result,
    v2_15_part_banner,
    v2_15_pct,
    v2_15_policy_candidates,
    v2_15_prediction_key_for_part_b,
    v2_15_profile,
    v2_15_region_catalog,
    v2_15_reveal_card,
    v2_15_status_html,
    v2_15_strategy_candidates,
    v2_15_table,
    v2_15_variant,
):
    def build_part_a():
        result = v2_15_part_a_result(v2_15_packet, partA_workload.value, partA_utilization.value)
        predicted = partA_prediction.value or "no prediction yet"
        actual = result["binding"]

        stack_fig = go.Figure()
        stack_fig.add_trace(go.Bar(
            x=["Operational", "Embodied"],
            y=[result["operational_kg"], result["embodied_kg_day"]],
            marker_color=[v2_15_color("BlueLine", "#2563eb"), v2_15_color("OrangeLine", "#d97706")],
            hovertemplate="%{x}: %{y:.2f} kg CO2e/day<extra></extra>",
        ))
        stack_fig.add_hline(
            y=v2_15_packet["carbon_budget_kg_day"],
            line=dict(color=v2_15_color("RedLine", "#b42318"), width=2, dash="dash"),
            annotation_text="operational carbon budget",
            annotation_position="top right",
        )
        stack_fig.update_layout(
            height=330,
            xaxis=dict(title="Lifecycle term"),
            yaxis=dict(title="kg CO2e/day"),
            margin=dict(t=45, b=55, l=60, r=20),
        )
        apply_plotly_theme(stack_fig)

        ratios = result["ratios"]
        rows = [
            ("Facility energy", f"{v2_15_num(result['facility_energy_kwh'], 2)} kWh/day", f"budget {v2_15_num(v2_15_packet['energy_budget_kwh_day'], 2)}", v2_15_status_html(ratios["energy"] <= 1.0)),
            ("Operational carbon", f"{v2_15_num(result['operational_kg'], 2)} kg/day", f"budget {v2_15_num(v2_15_packet['carbon_budget_kg_day'], 2)}", v2_15_status_html(ratios["carbon intensity"] <= 1.0)),
            ("Embodied carbon", f"{v2_15_num(result['embodied_kg_day'], 2)} kg/day", f"budget {v2_15_num(v2_15_packet['embodied_budget_kg_day'], 2)}", v2_15_status_html(ratios["embodied carbon"] <= 1.0)),
            ("Binding amount", actual, f"{v2_15_num(ratios[actual], 2)}x budget", v2_15_status_html(ratios[actual] <= 1.0)),
        ]

        return mo.vstack([
            v2_15_part_banner(
                "A",
                "Carbon Is An Amount Stack",
                "Measure operational energy, grid carbon, and embodied carbon before choosing a mitigation.",
                v2_15_color("BlueLine", "#2563eb"),
            ),
            mo.callout(
                mo.md(
                    f"**Scenario:** {v2_15_packet['stakeholder']} is asked to approve "
                    f"{v2_15_packet['service_name']} for {v2_15_packet['track_label']}. "
                    "The decision cannot rely on model quality alone; it needs an amount stack."
                ),
                kind="info",
            ),
            partA_prediction,
            mo.hstack([partA_workload, partA_utilization], justify="center", gap=2),
            v2_15_failure_card(
                result["fails"],
                f"Binding amount: {actual}",
                (
                    f"{actual} is at {v2_15_num(ratios[actual], 2)}x its budget. "
                    f"Operational carbon is {v2_15_num(result['operational_kg'], 2)} kg/day; "
                    f"embodied carbon is {v2_15_num(result['embodied_kg_day'], 2)} kg/day."
                ),
                "reduce workload, improve useful utilization, change region, or extend hardware lifetime",
            ),
            mo.hstack([
                v2_15_metric_card("Energy", f"{v2_15_num(result['facility_energy_kwh'], 1)} kWh", "facility/day", v2_15_color("BlueLine", "#2563eb")),
                v2_15_metric_card("Operational", f"{v2_15_num(result['operational_kg'], 1)} kg", "CO2e/day", v2_15_color("OrangeLine", "#d97706")),
                v2_15_metric_card("Embodied", f"{v2_15_num(result['embodied_kg_day'], 1)} kg", "amortized/day", v2_15_color("PurpleLine", "#7c3aed")),
                v2_15_metric_card("Binding", actual, "highest budget ratio", v2_15_color("RedLine", "#b42318") if result["fails"] else v2_15_color("GreenLine", "#047857")),
            ], justify="center", gap=1),
            mo.ui.plotly(stack_fig),
            v2_15_table(("Amount", "Measured value", "Budget or meaning", "Status"), rows),
            v2_15_reveal_card(
                "Prediction vs actual",
                predicted,
                actual,
                "The binding amount is the largest normalized budget ratio, not necessarily the largest raw number.",
                "success" if predicted == actual else "warn",
            ),
            v2_15_math_peek(
                "Math Peek / Source Model - energy, carbon, and lifecycle boundary",
                f"""
```
IT_energy_kWh        = units * average_workload_power_W * active_hours / 1000
facility_energy_kWh  = IT_energy_kWh * PUE
operational_CO2e_kg  = facility_energy_kWh * grid_intensity_g_per_kWh / 1000
embodied_CO2e_day    = units * embodied_kg_per_unit / lifetime_days
lifecycle_CO2e_day   = operational_CO2e_kg + embodied_CO2e_day
```

Chapter connection: `Carbon footprint analysis` and the lifecycle-estimation
callout require operational and embodied carbon in the same accounting boundary.
The hardware/model identity comes from the selected track registry; track fleet
size and non-H100 embodied values are notebook-local teaching assumptions.
"""
            ),
            partA_checkpoint,
        ])

    def build_part_b():
        result = v2_15_part_b_result(
            v2_15_packet,
            partA_workload.value,
            partB_utilization.value,
            partB_region.value,
            partB_schedule.value,
        )
        actual_key = v2_15_prediction_key_for_part_b(result)
        region_rows = []
        for region_id, region in v2_15_region_catalog().items():
            probe = v2_15_part_b_result(
                v2_15_packet,
                partA_workload.value,
                partB_utilization.value,
                region_id,
                partB_schedule.value,
            )
            region_rows.append((
                region["label"],
                f"{v2_15_num(region['carbon_g_kwh'], 0)} g/kWh",
                f"{v2_15_num(region['pue'], 2)} PUE",
                f"{v2_15_num(probe['carbon_kg'], 1)} kg/day",
                v2_15_status_html(probe["carbon_ok"] and probe["service_ok"]),
            ))

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=[v2_15_region_catalog()[rid]["label"] for rid in v2_15_region_catalog()],
            y=[
                v2_15_part_b_result(v2_15_packet, partA_workload.value, partB_utilization.value, rid, partB_schedule.value)["carbon_kg"]
                for rid in v2_15_region_catalog()
            ],
            marker_color=[
                v2_15_color("GreenLine", "#047857") if rid == partB_region.value else v2_15_color("BlueLine", "#2563eb")
                for rid in v2_15_region_catalog()
            ],
            hovertemplate="%{x}: %{y:.2f} kg CO2e/day<extra></extra>",
        ))
        bar_fig.add_hline(
            y=v2_15_packet["carbon_budget_kg_day"],
            line=dict(color=v2_15_color("RedLine", "#b42318"), dash="dash", width=2),
            annotation_text="carbon budget",
            annotation_position="top right",
        )
        bar_fig.update_layout(height=330, yaxis=dict(title="kg CO2e/day"), margin=dict(t=45, b=80, l=60, r=20))
        apply_plotly_theme(bar_fig)

        return mo.vstack([
            v2_15_part_banner(
                "B",
                "Placement And Utilization Change The Carbon Bill",
                "Cleaner grids help only if utilization, freshness, p99 latency, and reliability remain inside the track envelope.",
                v2_15_color("GreenLine", "#047857"),
            ),
            mo.callout(
                mo.md(
                    f"**Scenario:** operations wants to run {v2_15_packet['workload_unit']} at higher utilization. "
                    "You must decide whether region, schedule, or demand governance changes the carbon bill without breaking service."
                ),
                kind="info",
            ),
            partB_prediction,
            mo.hstack([partB_region, partB_schedule, partB_utilization], justify="center", gap=2),
            v2_15_failure_card(
                (not result["carbon_ok"]) or (not result["service_ok"]),
                f"Placement/utilization boundary: {result['binding']}",
                (
                    f"{result['region']['label']} with {result['schedule']['label']} produces "
                    f"{v2_15_num(result['carbon_kg'], 2)} kg/day, p99/freshness "
                    f"{v2_15_num(result['p99_ms'], 1)} ms and {v2_15_num(result['delay_h'], 1)} h delay, "
                    f"reliability {v2_15_num(result['reliability_pct'], 1)}%."
                ),
                "choose a lower-carbon region, lower target utilization, or apply demand governance only to nonurgent work",
            ),
            mo.hstack([
                v2_15_metric_card("Region carbon", f"{v2_15_num(result['region']['carbon_g_kwh'], 0)} g/kWh", result["region"]["label"], v2_15_color("GreenLine", "#047857")),
                v2_15_metric_card("Utilization", v2_15_pct(result["utilization"]), "useful work", v2_15_color("BlueLine", "#2563eb")),
                v2_15_metric_card("Carbon", f"{v2_15_num(result['carbon_kg'], 1)} kg", "CO2e/day", v2_15_color("OrangeLine", "#d97706")),
                v2_15_metric_card("p99/deadline", f"{v2_15_num(result['p99_ms'], 1)} ms", f"limit {v2_15_num(v2_15_packet['latency_slo_ms'], 0)}", v2_15_color("RedLine", "#b42318") if not result["p99_ok"] else v2_15_color("GreenLine", "#047857")),
            ], justify="center", gap=1),
            mo.ui.plotly(bar_fig),
            v2_15_table(("Region", "Grid intensity", "Facility overhead", "Carbon under current controls", "Status"), region_rows),
            v2_15_reveal_card(
                "Prediction vs actual",
                partB_prediction.value or "no prediction yet",
                actual_key,
                "Utilization, region, and schedule are evaluated together; the cleanest grid is not launchable if service guardrails fail.",
                "success" if partB_prediction.value == actual_key else "warn",
            ),
            v2_15_math_peek(
                "Math Peek / Source Model - utilization and carbon-aware placement",
                f"""
```
useful_utilization = useful_work / available_capacity
IT_energy          = active_energy * (0.72 + 0.38 / useful_utilization)
carbon_kg          = IT_energy * PUE_region * carbon_intensity_region * schedule_multiplier
service_ok         = p99 <= SLO and delay <= freshness_limit and reliability >= floor
```

Chapter connection: the `Geographic and temporal optimization` section says
where and when a job runs can dominate algorithmic efficiency gains, but the
service-level guardrails still determine whether the schedule is usable.
"""
            ),
            partB_checkpoint,
        ])

    def build_part_c():
        part_b = v2_15_part_b_result(
            v2_15_packet,
            partA_workload.value,
            partB_utilization.value,
            partB_region.value,
            partB_schedule.value,
        )
        packet = v2_15_strategy_candidates(v2_15_packet, part_b, partC_intensity.value, partC_governance.value)
        selected = next(item for item in packet["candidates"] if item["strategy_id"] == partC_strategy.value)
        actual_guardrail = selected["binding"]

        rows = []
        for item in packet["candidates"]:
            rows.append((
                item["label"],
                f"{v2_15_num(item['lifecycle_kg'], 1)} kg/day",
                f"{v2_15_num(item['quality_pct'], 1)}%",
                f"{v2_15_num(item['latency_ms'], 1)} ms",
                f"${v2_15_num(item['cost_usd'], 1)}",
                f"{v2_15_num(item['reliability_pct'], 1)}%",
                item["binding"],
                v2_15_status_html(item["passes"]),
            ))

        frontier_fig = go.Figure()
        frontier_fig.add_trace(go.Scatter(
            x=[item["lifecycle_kg"] for item in packet["candidates"]],
            y=[item["quality_pct"] for item in packet["candidates"]],
            mode="markers+text",
            text=[item["label"] for item in packet["candidates"]],
            textposition="top center",
            marker=dict(
                size=14,
                color=[
                    v2_15_color("GreenLine", "#047857") if item["passes"] else v2_15_color("RedLine", "#b42318")
                    for item in packet["candidates"]
                ],
            ),
            hovertemplate="%{text}<br>%{x:.2f} kg/day<br>%{y:.2f}% quality<extra></extra>",
        ))
        frontier_fig.add_hline(
            y=v2_15_packet["quality_floor_pct"],
            line=dict(color=v2_15_color("RedLine", "#b42318"), dash="dash", width=2),
            annotation_text="quality floor",
            annotation_position="bottom right",
        )
        frontier_fig.update_layout(
            height=340,
            xaxis=dict(title="Lifecycle carbon (kg CO2e/day)"),
            yaxis=dict(title="Quality (%)"),
            margin=dict(t=45, b=55, l=60, r=20),
        )
        apply_plotly_theme(frontier_fig)

        return mo.vstack([
            v2_15_part_banner(
                "C",
                "Mitigation Must Preserve Guardrails",
                "A sustainability mitigation only counts when it changes the binding amount and still passes quality, latency, cost, reliability, and governance.",
                v2_15_color("OrangeLine", "#d97706"),
            ),
            mo.callout(
                mo.md(
                    f"**Scenario:** a review board asks which mitigation should be attached to "
                    f"{v2_15_packet['service_name']}. You must name the guardrail that can reject it."
                ),
                kind="info",
            ),
            partC_prediction,
            mo.hstack([partC_strategy, partC_intensity, partC_governance], justify="center", gap=2),
            v2_15_failure_card(
                not selected["passes"],
                f"Mitigation guardrail: {selected['binding']}",
                (
                    f"{selected['label']} has lifecycle carbon {v2_15_num(selected['lifecycle_kg'], 2)} kg/day, "
                    f"quality {v2_15_num(selected['quality_pct'], 1)}%, latency {v2_15_num(selected['latency_ms'], 1)} ms, "
                    f"cost ${v2_15_num(selected['cost_usd'], 1)}, reliability {v2_15_num(selected['reliability_pct'], 1)}%."
                ),
                "lower intensity, add governance evidence, or choose the recommended passing mitigation",
            ),
            mo.hstack([
                v2_15_metric_card("Selected", selected["label"], "strategy", v2_15_color("BlueLine", "#2563eb")),
                v2_15_metric_card("Binding", selected["binding"], "guardrail", v2_15_color("RedLine", "#b42318") if not selected["passes"] else v2_15_color("GreenLine", "#047857")),
                v2_15_metric_card("Recommended", packet["recommended"]["label"], "lowest passing carbon", v2_15_color("GreenLine", "#047857")),
                v2_15_metric_card("Rejected", packet["rejected"]["label"], "alternative", v2_15_color("OrangeLine", "#d97706")),
            ], justify="center", gap=1),
            mo.ui.plotly(frontier_fig),
            v2_15_table(("Strategy", "Lifecycle carbon", "Quality", "Latency", "Cost/day", "Reliability", "Binding", "Status"), rows),
            v2_15_reveal_card(
                "Prediction vs actual",
                partC_prediction.value or "no prediction yet",
                actual_guardrail,
                "The selected strategy is checked against every guardrail, so the blocker may not be carbon.",
                "success" if partC_prediction.value == actual_guardrail else "warn",
            ),
            v2_15_math_peek(
                "Math Peek / Source Model - mitigation guardrail predicate",
                f"""
```
mitigated_carbon   = carbon_base * strategy_carbon_multiplier
mitigated_quality  = quality_base + strategy_quality_delta
mitigated_latency  = latency_base * strategy_latency_multiplier
mitigated_cost     = cost_base * strategy_cost_multiplier
launchable_strategy =
  carbon <= budget and quality >= floor and latency <= SLO and
  cost <= budget and reliability >= floor and governance_ok
```

Chapter connection: the Google 4 Ms and engineering guidelines reduce different
terms. The lab makes the conjunction explicit so a local efficiency win cannot
hide quality, reliability, or governance debt.
"""
            ),
            partC_checkpoint,
        ])

    def build_part_d_and_report():
        part_a = v2_15_part_a_result(v2_15_packet, partA_workload.value, partA_utilization.value)
        part_b = v2_15_part_b_result(
            v2_15_packet,
            partA_workload.value,
            partB_utilization.value,
            partB_region.value,
            partB_schedule.value,
        )
        strategy_packet = v2_15_strategy_candidates(v2_15_packet, part_b, partC_intensity.value, partC_governance.value)
        policy_packet = v2_15_policy_candidates(v2_15_packet, part_b, strategy_packet, partD_carbon_price.value)
        selected = next(item for item in policy_packet["candidates"] if item["policy_id"] == partD_policy.value)
        launch_policy = selected if selected["passes"] else policy_packet["recommended"]
        rejected = policy_packet["rejected"]

        rows = []
        for item in policy_packet["candidates"]:
            rows.append((
                item["label"],
                f"{v2_15_num(item['carbon_kg'], 1)} kg/day",
                f"{v2_15_num(item['lifecycle_kg'], 1)} kg/day",
                f"{v2_15_num(item['quality_pct'], 1)}%",
                f"{v2_15_num(item['latency_ms'], 1)} ms",
                f"${v2_15_num(item['cost_usd'], 1)}",
                item["binding"],
                v2_15_status_html(item["passes"]),
            ))

        policy_fig = go.Figure()
        policy_fig.add_trace(go.Bar(
            x=[item["label"] for item in policy_packet["candidates"]],
            y=[item["lifecycle_kg"] for item in policy_packet["candidates"]],
            marker_color=[
                v2_15_color("GreenLine", "#047857") if item["passes"] else v2_15_color("RedLine", "#b42318")
                for item in policy_packet["candidates"]
            ],
            hovertemplate="%{x}: %{y:.2f} kg CO2e/day<extra></extra>",
        ))
        policy_fig.add_hline(
            y=v2_15_packet["carbon_budget_kg_day"],
            line=dict(color=v2_15_color("OrangeLine", "#d97706"), dash="dash", width=2),
            annotation_text="operational carbon budget reference",
            annotation_position="top right",
        )
        policy_fig.update_layout(height=330, yaxis=dict(title="Lifecycle kg CO2e/day"), margin=dict(t=45, b=95, l=60, r=20))
        apply_plotly_theme(policy_fig)

        _incomplete = []
        for label, widget in (
            ("Part A prediction", partA_prediction),
            ("Part A checkpoint", partA_checkpoint),
            ("Part B prediction", partB_prediction),
            ("Part B checkpoint", partB_checkpoint),
            ("Part C prediction", partC_prediction),
            ("Part C checkpoint", partC_checkpoint),
            ("Part D prediction", partD_prediction),
            ("Part D checkpoint", partD_checkpoint),
        ):
            if widget.value is None:
                _incomplete.append(label)
        if not str(decision_input.value).strip():
            _incomplete.append("Engineering memo note")

        ledger_design = {
            "track_id": v2_15_profile.track_id,
            "selected_policy": launch_policy["label"],
            "student_selected_policy": selected["label"],
            "binding_amount": part_a["binding"],
            "binding_policy_guardrail": launch_policy["binding"],
            "rejected_alternative": rejected["label"],
            "operational_carbon_kg_day": round(part_a["operational_kg"], 4),
            "embodied_carbon_kg_day": round(part_a["embodied_kg_day"], 4),
            "region": part_b["region"]["label"],
            "schedule": part_b["schedule"]["label"],
            "utilization": round(part_b["utilization"], 4),
            "residual_risk": v2_15_packet["failure_story"],
            "v2_16_responsible_ai_implication": v2_15_packet["v2_16_implication"],
        }
        if not _incomplete:
            ledger.save(track=v2_15_profile.track_id, chapter=v2_15_chapter, design=ledger_design)

        report = build_lab_report(
            v2_15_metadata,
            track=v2_15_profile.label,
            scenario=v2_15_variant.workload_summary,
            learning_objectives=(
                "Model operational energy, carbon intensity, utilization, and embodied carbon as separate amounts.",
                "Find the selected track's binding sustainability amount before optimizing.",
                "Compare placement and scheduling choices under carbon and service-level guardrails.",
                "Choose a mitigation strategy that preserves quality, latency, cost, reliability, and governance.",
                "Export a carbon-aware policy memo with residual risk and V2-16 responsibility implication.",
            ),
            predictions={
                "partA_binding_prediction": partA_prediction.value,
                "partB_limiting_amount_prediction": partB_prediction.value,
                "partC_guardrail_prediction": partC_prediction.value,
                "partD_policy_prediction": partD_prediction.value,
            },
            knob_settings={
                "workload_multiplier": partA_workload.value,
                "partA_utilization_pct": partA_utilization.value,
                "region": partB_region.value,
                "schedule": partB_schedule.value,
                "partB_utilization_pct": partB_utilization.value,
                "strategy": partC_strategy.value,
                "strategy_intensity_pct": partC_intensity.value,
                "governance_review_attached": bool(partC_governance.value),
                "policy": partD_policy.value,
                "carbon_price_usd_per_ton": partD_carbon_price.value,
            },
            evidence_summary={
                "partA_binding_amount": part_a["binding"],
                "facility_energy_kwh_day": round(part_a["facility_energy_kwh"], 4),
                "operational_carbon_kg_day": round(part_a["operational_kg"], 4),
                "embodied_carbon_kg_day": round(part_a["embodied_kg_day"], 4),
                "partB_region": part_b["region"]["label"],
                "partB_schedule": part_b["schedule"]["label"],
                "partB_carbon_kg_day": round(part_b["carbon_kg"], 4),
                "partB_p99_ms": round(part_b["p99_ms"], 4),
                "partB_service_ok": part_b["service_ok"],
                "partC_recommended_strategy": strategy_packet["recommended"]["label"],
                "partC_selected_strategy": next(item for item in strategy_packet["candidates"] if item["strategy_id"] == partC_strategy.value)["label"],
                "partD_selected_policy": launch_policy["label"],
                "partD_student_selected_policy": selected["label"],
                "partD_binding_guardrail": launch_policy["binding"],
                "rejected_alternative": rejected["label"],
            },
            final_decision={
                "selected_policy": launch_policy["label"],
                "binding_amount": part_a["binding"],
                "binding_policy_guardrail": launch_policy["binding"],
                "rejected_alternative": rejected["label"],
                "residual_risk": v2_15_packet["failure_story"],
                "v2_16_responsible_ai_implication": v2_15_packet["v2_16_implication"],
            },
            big_takeaways=(
                "Sustainability requires amount accounting before optimization.",
                "Carbon intensity and utilization can dominate per-operation efficiency.",
                "Embodied carbon and operational carbon trade places across tracks.",
                "A mitigation is valid only if service and governance guardrails still pass.",
                "Carbon-aware policy needs demand governance to avoid rebound.",
            ),
            reflections={
                "partA_checkpoint": partA_checkpoint.value,
                "partB_checkpoint": partB_checkpoint.value,
                "partC_checkpoint": partC_checkpoint.value,
                "partD_checkpoint": partD_checkpoint.value,
                "student_memo_note": str(decision_input.value),
            },
            residual_risk=(
                f"{v2_15_packet['failure_story']} Validate these teaching estimates with measured workload power, "
                "current grid data, hardware product carbon footprints, quality canaries, and governance review."
            ),
            source_trace={
                "track_id": v2_15_profile.track_id,
                "scenario_id": v2_15_variant.scenario_id,
                "hardware_ref": v2_15_variant.hardware_ref,
                "model_ref": v2_15_variant.model_ref,
                "system_ref": v2_15_variant.system_ref or "device fleet",
                "shared_helpers": ("get_lab_track_variant", "get_track_profile", "build_lab_report", "report_export_panel"),
                "notebook_local_helpers": "v2_15_* amount model",
                "chapter_sections": (
                    "Energy Ceiling",
                    "Carbon footprint analysis",
                    "Geographic and temporal optimization",
                    "Google 4 Ms",
                    "Fallacies and Pitfalls",
                ),
            },
            result_snapshot={
                "track_packet": v2_15_packet,
                "part_a": part_a,
                "part_b": part_b,
                "part_c": strategy_packet,
                "part_d": policy_packet,
                "ledger_design": ledger_design,
            },
            incomplete_fields=tuple(_incomplete),
        )

        return mo.vstack([
            v2_15_part_banner(
                "D",
                "Carbon-Aware Policy Is A Guardrail Bundle",
                "A launch policy must pass carbon, quality, latency, cost, reliability, governance, and rebound checks together.",
                v2_15_color("PurpleLine", "#7c3aed"),
            ),
            mo.callout(
                mo.md(
                    f"**Scenario:** launch review requires a selected policy, rejected alternative, residual risk, "
                    f"and the V2-16 responsibility implication for {v2_15_packet['track_label']}."
                ),
                kind="info",
            ),
            partD_prediction,
            mo.hstack([partD_policy, partD_carbon_price], justify="center", gap=2),
            v2_15_failure_card(
                not selected["passes"],
                f"Policy guardrail: {selected['binding']}",
                (
                    f"{selected['label']} has lifecycle carbon {v2_15_num(selected['lifecycle_kg'], 2)} kg/day, "
                    f"quality {v2_15_num(selected['quality_pct'], 1)}%, latency {v2_15_num(selected['latency_ms'], 1)} ms, "
                    f"cost ${v2_15_num(selected['cost_usd'], 1)}, reliability {v2_15_num(selected['reliability_pct'], 1)}%."
                ),
                f"use {policy_packet['recommended']['label']} or revise failed guardrails before launch",
            ),
            mo.hstack([
                v2_15_metric_card("Selected", launch_policy["label"], "launch policy", v2_15_color("GreenLine", "#047857") if launch_policy["passes"] else v2_15_color("RedLine", "#b42318")),
                v2_15_metric_card("Binding", launch_policy["binding"], "policy guardrail", v2_15_color("OrangeLine", "#d97706")),
                v2_15_metric_card("Rejected", rejected["label"], "alternative", v2_15_color("RedLine", "#b42318")),
                v2_15_metric_card("V2-16", "responsibility", "next implication", v2_15_color("BlueLine", "#2563eb")),
            ], justify="center", gap=1),
            mo.ui.plotly(policy_fig),
            v2_15_table(("Policy", "Operational carbon", "Lifecycle carbon", "Quality", "Latency", "Cost/day", "Binding", "Status"), rows),
            v2_15_reveal_card(
                "Prediction vs actual",
                partD_prediction.value or "no prediction yet",
                policy_packet["recommended"]["policy_id"],
                "The recommended policy is the lowest lifecycle-carbon option that passes every launch guardrail.",
                "success" if partD_prediction.value == policy_packet["recommended"]["policy_id"] else "warn",
            ),
            v2_15_math_peek(
                "Math Peek / Source Model - policy launch predicate",
                f"""
```
launchable =
  carbon <= carbon_budget and quality >= quality_floor and latency <= SLO and
  cost + carbon_price * tonnes <= cost_budget and reliability >= floor and
  governance_ok and rebound_guardrail
```

Chapter connection: policy changes the objective function. The carbon price
makes carbon visible in cost, while the rebound guardrail prevents efficiency
savings from being spent as unconstrained demand growth.
"""
            ),
            partD_checkpoint,
            decision_input,
            mo.callout(
                mo.md(
                    f"**Synthesis memo:** Use `{launch_policy['label']}`. Binding amount from Part A is "
                    f"`{part_a['binding']}`; current policy guardrail is `{launch_policy['binding']}`. "
                    f"Reject `{rejected['label']}`. V2-16 implication: {v2_15_packet['v2_16_implication']}"
                ),
                kind="success" if not _incomplete else "warn",
            ),
            source_trace(
                {
                    "selected_track": v2_15_profile.track_id,
                    "selected_policy": launch_policy["label"],
                    "binding_amount": part_a["binding"],
                    "rejected_alternative": rejected["label"],
                    "ledger_save": "enabled after required predictions, checkpoints, and memo note are complete",
                    "report_artifact": "carbon-aware engineering memo",
                },
                collapsed=True,
                summary="Final memo source trace and ledger handoff.",
            ),
            mo.md("## Download Report"),
            report_export_panel(report),
        ])

    def build_synthesis():
        return build_part_d_and_report()

    mo.vstack([
        build_part_a(),
        build_part_b(),
        build_part_c(),
        build_synthesis(),
        mo.Html(
            f"""
<div class="lab-hud">
  <span class="hud-label">LAB</span>
  <span class="hud-value">15 &middot; Sustainable AI</span>
  <span class="hud-label">TRACK</span>
  <span class="hud-value">{v2_15_profile.label}</span>
</div>
"""
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
