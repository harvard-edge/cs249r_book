import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 15: The Carbon Budget · MLSysBook")


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
        MathPeek,
        big_takeaways,
        build_lab_report,
        gated_hypothesis_card,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        instrumentation_console,
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
        MathPeek,
        apply_plotly_theme,
        big_takeaways,
        build_lab_report,
        gated_hypothesis_card,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        html,
        instrumentation_console,
        ledger,
        math,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_arc_context,
        track_context,
    )


@app.cell
def _(get_lab_metadata):
    v2_15_metadata = get_lab_metadata("vol2/lab_15_sustainable_ai.py")
    v2_15_chapter = 15
    return v2_15_chapter, v2_15_metadata


@app.cell(hide_code=True)
def _(mo):
    v2_15_track_picker = mo.ui.dropdown(
        options={
            "⚡ TinyML Track (ARM Cortex-M55 / ESP32-S3 & Low-Power Carbon Budgeting)": "oura_ring",
            "📱 Mobile Track (Apple Silicon / Snapdragon & Thermal Envelope / Embodied Carbon)": "iphone",
            "🤖 Edge & Embodied Track (NVIDIA Jetson AGX Orin & Real-Time Carbon Footprint)": "robotaxi",
            "☁️ Cloud Supercomputing Track (H100/B200 Clusters & Grid Intensity / Hyperscale PUE)": "cloud_fleet",
        },
        value="☁️ Cloud Supercomputing Track (H100/B200 Clusters & Grid Intensity / Hyperscale PUE)",
        label="Select Course / Industry Track",
    )
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
        v2_15_metric_card,
        v2_15_model_gflops,
        v2_15_model_params_m,
        v2_15_num,
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
def _(
    v2_15_hardware,
    v2_15_model,
    v2_15_profile,
    v2_15_system,
    v2_15_track_packet,
    v2_15_variant,
):
    v2_15_packet = v2_15_track_packet(
        v2_15_profile,
        v2_15_variant,
        v2_15_hardware,
        v2_15_model,
        v2_15_system,
    )
    return (v2_15_packet,)


@app.cell
def _(v2_15_region_catalog):
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
    return (
        partA_checkpoint,
        partA_prediction,
        partA_utilization,
        partA_workload,
    )


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
    return (
        partB_checkpoint,
        partB_prediction,
        partB_region,
        partB_schedule,
        partB_utilization,
    )


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
    v2_15_student_id = mo.ui.text(
        label="Student / Engineer ID",
        placeholder="e.g. MLSYS-ENG-9042",
        value="",
    )
    decision_input = mo.ui.text_area(
        label="Engineering memo note",
        placeholder="One sentence: selected policy, binding amount, rejected alternative, residual risk.",
    )
    return (
        decision_input,
        partD_carbon_price,
        partD_checkpoint,
        partD_policy,
        partD_prediction,
        v2_15_student_id,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    COLORS,
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
    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell">
      <div class="mlsysbook-lab-header" style="--mlsysbook-accent: #A51C30;">
        <div class="mlsysbook-meta">
          ML SYSTEMS TEXTBOOK &middot; VOLUME II &middot; CHAPTER 15 &middot; LAB 15
        </div>
        <h1 style="margin: 8px 0 4px 0; color: #0F172A; font-weight: 800; font-size: 1.85rem; letter-spacing: -0.02em;">
          Sustainable AI: The Carbon Budget, Energy Ceiling &amp; Operational Guardrails
        </h1>
        <p style="margin: 0 0 14px 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">
          Treat sustainability as a closed amount system: measure energy, grid carbon intensity, utilization, embodied carbon, and hardware lifecycles while enforcing latency, cost, and quality operational guardrails.
        </p>
        <div class="mlsysbook-chip-row" style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;">
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Track:</strong> {v2_15_profile.label}
          </span>
          <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155;">
            <strong>Stakeholder:</strong> {v2_15_packet['stakeholder']}
          </span>
          <span class="mlsysbook-chip" style="background: #F8FAFC; color: #475569;">
            <strong>Hardware:</strong> {v2_15_packet['hardware_ref']}
          </span>
          <span class="mlsysbook-chip" style="background: #F8FAFC; color: #475569;">
            <strong>Model:</strong> {v2_15_packet['model_ref']}
          </span>
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Primary Metric:</strong> kg CO2e/day
          </span>
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Guardrail:</strong> p99 &le; SLO &amp; Quality &ge; Floor
          </span>
        </div>
      </div>

      <div class="mlsysbook-panel" style="margin-bottom: 20px;">
        <h3 style="margin: 0 0 8px 0; color: #0F172A; font-size: 1.15rem;">
          System Scenario: {v2_15_profile.label} Carbon Budget &amp; Operating Envelope
        </h3>
        <p style="margin: 0 0 12px 0; font-size: 0.92rem; color: #334155; line-height: 1.55;">
          {v2_15_variant.workload_summary} Sustainability is an engineering amount stack: operational energy,
          grid carbon intensity, hardware embodied carbon, and utilization overheads must fit within fixed physical envelopes.
          Greenwashing occurs when energy gains are reported without lifecycle embodied carbon or when rebound effects increase aggregate fleet power.
        </p>
        <div style="background: #F8FAFC; border-left: 4px solid #006395; padding: 12px 16px; border-radius: 4px; font-size: 0.9rem; color: #1E293B;">
          <strong>The Architectural Invariants of Sustainable ML Systems:</strong>
          <ul class="mlsysbook-list" style="margin: 8px 0 4px 0;">
            <li><strong>The Energy &amp; Facility Invariant:</strong> Compute energy expands by datacenter infrastructure: <i>E</i><sub>facility</sub> = <i>PUE</i> &middot; <i>E</i><sub>IT</sub>. On battery-powered mobile and wearables, energy capacity is finite and irreversible.</li>
            <li><strong>The Lifecycle Carbon Accounting Law:</strong> Sustainability accounts for the full physical asset: <i>C</i><sub>lifecycle</sub> = <i>C</i><sub>operational</sub> + <i>C</i><sub>embodied</sub>. Extending hardware retirement schedules amortizes manufacturing emissions.</li>
            <li><strong>Geographic &amp; Temporal Carbon Optimization:</strong> Carbon intensity of electric grids fluctuates by geography and hour: <i>C</i><sub>op</sub> = &int; <i>P</i>(<i>t</i>) &middot; <i>I</i><sub>grid</sub>(<i>t</i>) <i>dt</i>. Flexible workloads migrate spatially to low-carbon grids or temporally to peak renewable windows.</li>
            <li><strong>The Rebound Principle / Jevons Guardrail:</strong> Algorithmic and hardware efficiency gains must not be silently consumed by unconstrained service request scaling. Carbon budgeting requires an explicit demand ceiling.</li>
            <li><strong>Conjunctive Operational Release Gate:</strong> Carbon reduction is deployable only when all service guardrails hold simultaneously: Launchable = Carbon<sub>ok</sub> &and; Quality<sub>ok</sub> &and; Latency<sub>ok</sub> &and; Cost<sub>ok</sub> &and; Reliability<sub>ok</sub>.</li>
          </ul>
        </div>
      </div>
    </div>
    """)

    objectives_html = mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <ul class="mlsysbook-list" style="margin: 0; padding-left: 1.25rem; font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.6;">
                <li><strong>Model the lifecycle carbon stack:</strong> quantify operational energy, regional grid carbon intensity, and amortized embodied carbon for {v2_15_profile.label}.</li>
                <li><strong>Identify the binding sustainability amount:</strong> determine whether facility energy, carbon intensity, or embodied hardware dominates the bill.</li>
                <li><strong>Evaluate spatial and temporal optimization:</strong> exploit cleaner regional grids and deferrable scheduling windows while respecting p99 latency and freshness limits.</li>
                <li><strong>Enforce conjunctive operational guardrails:</strong> reject greenwashing mitigations that violate quality floors, latency SLOs, cost caps, or reliability thresholds.</li>
                <li><strong>Author an authorized sustainability memo:</strong> commit a carbon-aware policy with explicit residual risk and governance sign-off into the system ledger.</li>
            </ul>
        </div>
        <div style="display: flex; gap: 24px; padding-top: 12px; border-top: 1px solid {COLORS['Border']}; font-size: 0.82rem; color: {COLORS['TextMuted']};">
            <div>
                <strong>PREREQUISITES:</strong> Energy modeling &middot; PUE calculations &middot; Grid carbon intensity &middot; Embodied carbon &middot; SLO guardrails
            </div>
            <div style="margin-left: auto;">
                <strong>DURATION:</strong> ~50 min <span style="color: {COLORS['TextSec']};">(A: 12 &middot; B: 12 &middot; C: 12 &middot; D: 14 min)</span>
            </div>
        </div>
        <div style="margin-top: 14px; padding: 10px 14px; background: {COLORS['BlueLL']}; border-left: 3px solid {COLORS['BlueLine']}; border-radius: 0 6px 6px 0; font-size: 0.85rem; color: {COLORS['Text']};">
            <strong>CORE QUESTION:</strong> <em>&ldquo;How much carbon can this system eliminate before operational guardrails reject the mitigation?&rdquo;</em>
        </div>
    </div>
    """)

    reading_html = mo.Html(f"""
    <div style="border: 1px solid {COLORS['Border']}; background: #FAFAFA; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
      <div style="font-size: 0.85rem; font-weight: 700; color: #1E293B; margin-bottom: 6px;">Recommended Reading</div>
      <p style="font-size: 0.82rem; color: #475569; margin: 0 0 8px 0;">
        Complete these foundational readings before beginning this lab:
      </p>
      <ul class="mlsysbook-list" style="margin: 0; padding-left: 1.25rem; font-size: 0.82rem; color: #334155; line-height: 1.5;">
        <li><strong>The Energy Ceiling &amp; PUE:</strong> datacenter power usage effectiveness, cooling overheads, and hardware thermal design power.</li>
        <li><strong>Embodied Carbon &amp; Lifecycle Accounting:</strong> silicon fabrication emissions, server lifecycle amortization, and hardware turnover rates.</li>
        <li><strong>Geographic &amp; Temporal Optimization:</strong> regional grid carbon intensity, marginal vs. average emissions, and diurnal clean energy windows.</li>
        <li><strong>Carbon Shadow Pricing &amp; Jevons Paradox:</strong> internal carbon tariffs, demand capping, and rebound prevention in AI systems.</li>
      </ul>
    </div>
    """)

    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        header_html,
        objectives_html,
        reading_html,
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
            collapsed=True,
            summary="Registry-backed track and hardware context plus notebook-local sustainability assumptions.",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(
    MathPeek,
    apply_plotly_theme,
    big_takeaways,
    build_lab_report,
    decision_input,
    gated_hypothesis_card,
    go,
    instrumentation_console,
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
    v2_15_metadata,
    v2_15_metric_card,
    v2_15_num,
    v2_15_packet,
    v2_15_part_a_result,
    v2_15_part_b_result,
    v2_15_pct,
    v2_15_policy_candidates,
    v2_15_prediction_key_for_part_b,
    v2_15_profile,
    v2_15_region_catalog,
    v2_15_reveal_card,
    v2_15_status,
    v2_15_status_html,
    v2_15_strategy_candidates,
    v2_15_student_id,
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

        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {v2_15_color('BlueLine', '#2563eb')}; background:{v2_15_color('BlueLL', '#eff6ff')};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{v2_15_color('BlueLine', '#2563eb')};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Lifecycle Carbon Briefing &middot; {v2_15_packet['stakeholder']}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;{v2_15_packet['service_name']} cannot deploy on accuracy alone. Measure operational energy, grid carbon intensity, and amortized embodied carbon to determine which amount binds the system.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; {v2_15_packet['stakeholder']} &middot; {v2_15_packet['track_label']}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                partA_prediction,
                title="1. Formulate Binding Sustainability Amount Hypothesis",
                subtitle=f"Predict which sustainability amount will bind {v2_15_packet['track_label']} before running the lifecycle simulator.",
            ),
        ]
        if partA_prediction.value is None:
            return mo.vstack(items)

        items.extend([
            instrumentation_console(
                mo.hstack([partA_workload, partA_utilization], justify="center", gap=2),
                title="Workload &amp; Utilization Operating Knobs",
                subtitle="Scale request demand and average hardware utilization percentage",
            ),
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
            MathPeek(
                r"E_{\text{facility}} = E_{\text{IT}} \cdot PUE, \quad C_{\text{op}} = E_{\text{fac}} \cdot I_{\text{grid}}, \quad C_{\text{emb}} = \frac{N \cdot C_{\text{unit}}}{T_{\text{lifetime}}}, \quad C_{\text{total}} = C_{\text{op}} + C_{\text{emb}}",
                {
                    "facility energy": f"{result['facility_energy_kwh']:.2f} kWh/day",
                    "operational carbon": f"{result['operational_kg']:.2f} kg CO2e/day",
                    "embodied carbon": f"{result['embodied_kg_day']:.2f} kg CO2e/day",
                    "binding amount": actual,
                    "chapter source": "Volume II, Chapter 15: The Energy Ceiling & Carbon Footprint Analysis",
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part A Lifecycle Carbon Decision</h4>
                <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                    Binding sustainability amount: <code>{actual}</code> ({v2_15_num(ratios[actual], 2)}x budget).
                    Which amount should the policy carry forward into placement and scheduling?
                </p>
                {partA_checkpoint}
            </div>
            """),
        ])
        return mo.vstack(items)

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

        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {v2_15_color('GreenLine', '#047857')}; background:{v2_15_color('GreenLL', '#ecfdf3')};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{v2_15_color('GreenLine', '#047857')};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Spatial &amp; Temporal Optimization Briefing &middot; Operations Lead
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;Cleaner regional grids and delayed scheduling slash operational emissions, but higher utilization degrades tail latency and queueing stability. Keep latency and freshness inside the track SLO.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; Operations Lead &middot; {v2_15_packet['track_label']}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                partB_prediction,
                title="2. Formulate Placement &amp; Scheduling Limit Hypothesis",
                subtitle="Predict which operational factor will limit carbon optimization after spatial and temporal shifts.",
            ),
        ]
        if partB_prediction.value is None:
            return mo.vstack(items)

        items.extend([
            instrumentation_console(
                mo.hstack([partB_region, partB_schedule, partB_utilization], justify="center", gap=2),
                title="Placement, Scheduling &amp; Target Utilization",
                subtitle="Select execution grid, temporal batch window, and server target utilization",
            ),
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
                "Grid carbon intensity only dominates if tail latency, freshness limits, and reliability remain inside SLA boundaries.",
                "success" if partB_prediction.value == actual_key else "warn",
            ),
            MathPeek(
                r"I_{\text{eff}} = \frac{I_{\text{grid}}}{PUE}, \quad T_{\text{queue}} \approx \frac{\rho}{1-\rho} \cdot \frac{T_{\text{service}}}{2}, \quad C_{\text{saved}} = E \cdot (I_{\text{base}} - I_{\text{clean}})",
                {
                    "selected grid intensity": f"{result['region']['carbon_g_kwh']:.0f} g/kWh",
                    "p99 latency / deadline": f"{result['p99_ms']:.1f} ms",
                    "freshness delay": f"{result['delay_h']:.2f} hours",
                    "limiting constraint": actual_key,
                    "chapter source": "Volume II, Chapter 15: Geographic and Temporal Optimization & Queueing",
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part B Spatial/Temporal Optimization Decision</h4>
                <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                    Operating in <code>{result['region']['label']}</code> with <code>{result['schedule']['label']}</code>.
                    Commit your scheduling rule:
                </p>
                {partB_checkpoint}
            </div>
            """),
        ])
        return mo.vstack(items)

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

        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {v2_15_color('OrangeLine', '#d97706')}; background:{v2_15_color('OrangeLL', '#fffbeb')};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{v2_15_color('OrangeLine', '#d97706')};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Mitigation Guardrail Briefing &middot; Technical Review Board
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;A sustainability mitigation only counts if it reduces the binding amount while passing quality, latency, cost, reliability, and governance. Aggressive compression or voltage scaling must not hide silent accuracy collapse.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; Technical Review Board &middot; {v2_15_packet['track_label']}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                partC_prediction,
                title="3. Formulate Mitigation Guardrail Hypothesis",
                subtitle="Predict which operational constraint will bind first when mitigation aggressiveness increases.",
            ),
        ]
        if partC_prediction.value is None:
            return mo.vstack(items)

        items.extend([
            instrumentation_console(
                mo.hstack([partC_strategy, partC_intensity, partC_governance], justify="center", gap=2),
                title="Mitigation Strategy &amp; Governance Attachment",
                subtitle="Select mitigation family, aggressive compression intensity, and formal governance trace",
            ),
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
            MathPeek(
                r"C_{\text{mit}} = C_{\text{base}} \cdot \mu_{\text{strategy}}, \quad Q_{\text{mit}} = Q_{\text{base}} + \Delta Q, \quad T_{\text{mit}} = T_{\text{base}} + \Delta T, \quad \text{Valid} = \bigwedge_i \left(M_i \in \text{Envelope}_i\right)",
                {
                    "mitigated lifecycle carbon": f"{selected['lifecycle_kg']:.2f} kg/day",
                    "quality score": f"{selected['quality_pct']:.1f}%",
                    "latency": f"{selected['latency_ms']:.1f} ms",
                    "binding guardrail": actual_guardrail,
                    "chapter source": "Volume II, Chapter 15: Mitigation Guardrails & The Google 4 Ms",
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part C Mitigation Guardrail Decision</h4>
                <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                    Evaluated mitigation: <code>{selected['label']}</code> &mdash; Status: <b>{v2_15_status(selected['passes'])}</b>.
                    What guardrail policy should carry forward into Part D?
                </p>
                {partC_checkpoint}
            </div>
            """),
        ])
        return mo.vstack(items)

    def build_part_d():
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

        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {v2_15_color('RedLine', '#b42318')}; background:{v2_15_color('RedLL', '#fef3f2')};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{v2_15_color('RedLine', '#b42318')};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Policy Release Briefing &middot; Lead Sustainability Architect
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    &ldquo;Sustainability policy changes the objective function. A carbon price penalizes fossil emissions while the rebound guardrail blocks Jevons paradox rebound. Enforce the full conjunction.&rdquo;
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                    &mdash; Lead Architect &middot; {v2_15_packet['track_label']}
                </div>
            </div>
            """),
            gated_hypothesis_card(
                partD_prediction,
                title="4. Formulate Carbon-Aware Policy Gate Hypothesis",
                subtitle="Predict which policy candidate will satisfy the full conjunction of carbon and operational guardrails.",
            ),
        ]
        if partD_prediction.value is None:
            return mo.vstack(items)

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

        items.extend([
            instrumentation_console(
                mo.hstack([partD_policy, partD_carbon_price], justify="center", gap=2),
                title="Policy Candidate &amp; Carbon Shadow Price Controls",
                subtitle="Select deployment policy rule and internal carbon price per ton",
            ),
            v2_15_failure_card(
                not selected["passes"],
                f"Policy guardrail: {selected['binding']}",
                (
                    f"{selected['label']} produces lifecycle carbon {v2_15_num(selected['lifecycle_kg'], 2)} kg/day, "
                    f"quality {v2_15_num(selected['quality_pct'], 1)}%, latency {v2_15_num(selected['latency_ms'], 1)} ms, "
                    f"cost ${v2_15_num(selected['cost_usd'], 1)} under ${partD_carbon_price.value}/ton."
                ),
                "raise carbon price, switch to guarded policy, or adopt the recommended passing configuration",
            ),
            mo.hstack([
                v2_15_metric_card("Selected", selected["label"], "policy", v2_15_color("BlueLine", "#2563eb")),
                v2_15_metric_card("Binding", selected["binding"], "guardrail", v2_15_color("RedLine", "#b42318") if not selected["passes"] else v2_15_color("GreenLine", "#047857")),
                v2_15_metric_card("Recommended", policy_packet["recommended"]["label"], "policy", v2_15_color("GreenLine", "#047857")),
                v2_15_metric_card("Rejected", rejected["label"], "alternative", v2_15_color("RedLine", "#b42318")),
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
            MathPeek(
                r"\text{Launchable} = (C \le B_C) \wedge (Q \ge Q_{\min}) \wedge (T \le \text{SLO}) \wedge (\text{Cost} + P_C \cdot \text{Tons} \le B_{\$}) \wedge (R \ge R_{\min}) \wedge \text{Gov}_{ok} \wedge \text{Rebound}_{ok}",
                {
                    "selected policy": selected["label"],
                    "binding guardrail": selected["binding"],
                    "carbon price": f"${partD_carbon_price.value}/ton",
                    "lifecycle carbon": f"{selected['lifecycle_kg']:.2f} kg/day",
                    "chapter source": "Volume II, Chapter 15: Policy Formulation, Carbon Shadow Prices & Fallacies",
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part D Policy Authorization Decision</h4>
                <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                    Selected policy: <code>{selected['label']}</code> &mdash; Status: <b>{v2_15_status(selected['passes'])}</b>.
                    What action should the launch review take?
                </p>
                {partD_checkpoint}
            </div>
            """),
        ])
        return mo.vstack(items)

    def build_synthesis():
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

        complete_widgets = (
            ("Part A prediction", partA_prediction),
            ("Part A checkpoint", partA_checkpoint),
            ("Part B prediction", partB_prediction),
            ("Part B checkpoint", partB_checkpoint),
            ("Part C prediction", partC_prediction),
            ("Part C checkpoint", partC_checkpoint),
            ("Part D prediction", partD_prediction),
            ("Part D checkpoint", partD_checkpoint),
        )
        incomplete = [label for label, widget in complete_widgets if widget.value is None]
        if not str(decision_input.value).strip():
            incomplete.append("Engineering memo note")

        passed = launch_policy["passes"]
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
        if not incomplete:
            ledger.save(track=v2_15_profile.track_id, chapter=v2_15_chapter, design=ledger_design)

        report = build_lab_report(
            v2_15_metadata,
            student_id=str(v2_15_student_id.value).strip(),
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
            incomplete_fields=tuple(incomplete),
        )

        status_text = "SAVED TO LEDGER" if not incomplete else "INCOMPLETE"
        status_kind = "success" if not incomplete else "warn"

        items = [
            mo.md("## Synthesis &mdash; Carbon-Aware Engineering Memo &amp; Governance Authorization"),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #1F407A; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">STUDENT MEMO &amp; REFLECTIONS</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Sustainability Engineering Memo</h4>
                {v2_15_student_id}
                <div style="margin-top: 12px;">{decision_input}</div>
            </div>
            """),
            mo.callout(
                mo.md(
                    f"**Synthesis memo:** Selected `{launch_policy['label']}`. Binding amount from Part A is "
                    f"`{part_a['binding']}`; current policy guardrail is `{launch_policy['binding']}`. "
                    f"Reject `{rejected['label']}`.  \n\n"
                    f"**V2-16 implication:** {v2_15_packet['v2_16_implication']}"
                ),
                kind=status_kind,
            ),
            mo.callout(
                mo.md(
                    f"**Status:** {status_text}. "
                    + (
                        "Complete all predictions, checkpoints, and your engineering memo note before final save."
                        if incomplete
                        else "Ledger snapshot successfully recorded for downstream labs."
                    )
                ),
                kind=status_kind,
            ),
            mo.Html(f"""
            <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
                <div style="flex:1; min-width:220px; background:white; border:1px solid {v2_15_color('Border', '#d9dee8')};
                            border-radius:10px; padding:16px; border-top:3px solid {v2_15_color('GreenLine', '#047857')};">
                    <div style="font-size:0.72rem; font-weight:700; color:{v2_15_color('TextMuted', '#64748b')}; text-transform:uppercase;">
                        Selected release policy</div>
                    <div style="font-size:1.05rem; font-weight:800; color:{v2_15_color('Text', '#1f2937')}; margin-top:5px;">
                        {launch_policy['label']}</div>
                </div>
                <div style="flex:1; min-width:220px; background:white; border:1px solid {v2_15_color('Border', '#d9dee8')};
                            border-radius:10px; padding:16px; border-top:3px solid {v2_15_color('OrangeLine', '#d97706')};">
                    <div style="font-size:0.72rem; font-weight:700; color:{v2_15_color('TextMuted', '#64748b')}; text-transform:uppercase;">
                        Binding amount</div>
                    <div style="font-size:1.05rem; font-weight:800; color:{v2_15_color('Text', '#1f2937')}; margin-top:5px;">
                        {part_a['binding']}</div>
                </div>
                <div style="flex:1; min-width:220px; background:white; border:1px solid {v2_15_color('Border', '#d9dee8')};
                            border-radius:10px; padding:16px; border-top:3px solid {v2_15_color('RedLine', '#b42318')};">
                    <div style="font-size:0.72rem; font-weight:700; color:{v2_15_color('TextMuted', '#64748b')}; text-transform:uppercase;">
                        Rejected alternative</div>
                    <div style="font-size:1.05rem; font-weight:800; color:{v2_15_color('Text', '#1f2937')}; margin-top:5px;">
                        {rejected['label']}</div>
                </div>
            </div>
            """),
            big_takeaways([
                "Sustainability requires full amount accounting (energy, grid carbon, embodied carbon) before optimization.",
                "Regional grid carbon intensity and temporal scheduling can dominate per-operation computational efficiency.",
                "Embodied hardware carbon and operational power trade places across wearables, mobile, edge, and cloud tracks.",
                "A mitigation is valid only inside a conjunctive guardrail envelope preserving quality, latency, cost, and reliability.",
                "Carbon shadow prices and demand governance are mandatory to prevent Jevons paradox rebound.",
            ]),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #A51C30; margin: 18px 0; background: #FFFDFD;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #A51C30; text-transform: uppercase; margin-bottom: 6px;">LEAD ARCHITECT AUTHORIZATION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Lead Architect Authorization: {v2_15_packet['stakeholder']}</h4>
                <div style="display: flex; gap: 12px; align-items: center; margin-top: 8px;">
                    <span style="display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 800; font-size: 0.8rem; background: {'#ECFDF5' if passed else '#FEF2F2'}; color: {'#065F46' if passed else '#991B1B'}; border: 1px solid {'#A7F3D0' if passed else '#FECACA'};">
                        {'APPROVED FOR DEPLOYMENT' if passed else 'BLOCKED BY GUARDRAIL CONJUNCTION'}
                    </span>
                    <span style="font-size: 0.85rem; color: #475569;">
                        Policy: <code>{launch_policy['label']}</code> &middot; Binding: <code>{launch_policy['binding']}</code>
                    </span>
                </div>
            </div>
            """),
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
            mo.Html(f"""
            <div style="border: 1px solid #CBD5E1; border-radius: 8px; padding: 16px 20px; margin-top: 20px; background: #F8FAFC;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
                    What's Next &middot; Volume II Synthesis Pipeline
                </div>
                <h4 style="margin: 0 0 6px 0; color: #0F172A; font-size: 1.05rem;">
                    Next Lab: Volume II, Chapter 16 &mdash; Responsible AI: Fairness, Bias, Safety &amp; Auditability
                </h4>
                <p style="margin: 0; font-size: 0.88rem; color: #334155; line-height: 1.5;">
                    Now that your sustainability budget and energy ceiling are established, investigate how compute and data pruning
                    impact demographic subgroups, model fairness, safety boundaries, and explainability overhead in Chapter 16.
                </p>
            </div>
            """),
        ]
        return mo.vstack(items)

    tabs = mo.ui.tabs({
        "Part A -- Carbon Stack": build_part_a(),
        "Part B -- Placement & Timing": build_part_b(),
        "Part C -- Mitigation Guardrails": build_part_c(),
        "Part D -- Policy Gate": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


@app.cell
def _(mo, v2_15_packet):
    mo.Html(
        f"""
    <div class="lab-hud">
      <span class="hud-label">LAB</span>
      <span class="hud-value">Vol2 &middot; Lab 15</span>
      <span class="hud-label">TRACK</span>
      <span class="hud-value">{v2_15_packet['track_label']}</span>
      <span class="hud-label">METRIC</span>
      <span class="hud-value">kg CO2e/day</span>
    </div>
    """
    )
    return


if __name__ == "__main__":
    app.run()
