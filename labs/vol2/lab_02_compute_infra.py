import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
async def _():
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
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
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
        ledger,
        mo,
        report_export_panel,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v2_02_metadata = get_lab_metadata("vol2/lab_02_compute_infra.py")
    return (v2_02_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_02_track_picker = track_selector(default=_default_track)
    v2_02_track_picker
    return (v2_02_track_picker,)


@app.cell
def _(get_lab_track_variant, get_track_profile, v2_02_track_picker):
    v2_02_track_id = v2_02_track_picker.value
    v2_02_profile = get_track_profile(v2_02_track_id)
    v2_02_variant = get_lab_track_variant("v2_02_compute_wall", v2_02_profile.track_id)
    return (v2_02_profile, v2_02_track_id, v2_02_variant)


@app.cell
def _():
    def v2_02_escape(value):
        import html as _html

        return _html.escape(str(value))

    def v2_02_track_packet(profile, variant):
        shared = {
            "iphone": {
                "asset": "device thermal tray",
                "accelerator_label": "A17-class NPU slice",
                "accelerator_unit": "device",
                "accel_power_kw": 0.006,
                "rack_overhead_kw": 0.080,
                "default_accels_per_rack": 24,
                "default_racks": 4,
                "max_accels_per_rack": 80,
                "max_racks": 12,
                "site_power_kw": 1.80,
                "cooling_kw": {
                    "Air / passive": 0.18,
                    "Direct liquid / active": 0.42,
                    "High density liquid": 0.78,
                },
                "peak_tflops_per_accel": 35.0,
                "mfu": 0.42,
                "baseline_demand_tflops": 820.0,
                "electricity_usd_kwh": 0.18,
                "capex_per_accel_usd": 1150.0,
                "min_util_pct": 38,
                "max_util_pct": 82,
                "target_util_pct": 62,
                "hourly_cost_guardrail": 4.2,
                "carbon_guardrail_kg_hr": 0.75,
                "memory_margin_pct": 18,
                "carry_forward": "Offload fallback needs a privacy-aware network path and a small on-device cache.",
                "report_frame": "minimum supported device tier and offload boundary",
                "failure_noun": "thermal tray",
            },
            "oura_ring": {
                "asset": "wearable charging/programming tray",
                "accelerator_label": "TinyML MCU inference slot",
                "accelerator_unit": "ring",
                "accel_power_kw": 0.00018,
                "rack_overhead_kw": 0.006,
                "default_accels_per_rack": 36,
                "default_racks": 3,
                "max_accels_per_rack": 120,
                "max_racks": 10,
                "site_power_kw": 0.15,
                "cooling_kw": {
                    "Air / passive": 0.010,
                    "Direct liquid / active": 0.030,
                    "High density liquid": 0.060,
                },
                "peak_tflops_per_accel": 0.08,
                "mfu": 0.36,
                "baseline_demand_tflops": 2.4,
                "electricity_usd_kwh": 0.20,
                "capex_per_accel_usd": 220.0,
                "min_util_pct": 28,
                "max_util_pct": 68,
                "target_util_pct": 46,
                "hourly_cost_guardrail": 0.38,
                "carbon_guardrail_kg_hr": 0.035,
                "memory_margin_pct": 7,
                "carry_forward": "Phone assist needs buffered storage and a radio schedule that preserves duty cycle.",
                "report_frame": "MCU, flash, battery, and phone/cloud assist envelope",
                "failure_noun": "duty-cycle tray",
            },
            "robotaxi": {
                "asset": "vehicle compute bay",
                "accelerator_label": "Orin-class safety compute module",
                "accelerator_unit": "module",
                "accel_power_kw": 0.075,
                "rack_overhead_kw": 0.45,
                "default_accels_per_rack": 8,
                "default_racks": 6,
                "max_accels_per_rack": 24,
                "max_racks": 16,
                "site_power_kw": 8.5,
                "cooling_kw": {
                    "Air / passive": 0.95,
                    "Direct liquid / active": 1.65,
                    "High density liquid": 2.60,
                },
                "peak_tflops_per_accel": 275.0,
                "mfu": 0.48,
                "baseline_demand_tflops": 5200.0,
                "electricity_usd_kwh": 0.16,
                "capex_per_accel_usd": 1800.0,
                "min_util_pct": 35,
                "max_util_pct": 74,
                "target_util_pct": 55,
                "hourly_cost_guardrail": 5.6,
                "carbon_guardrail_kg_hr": 2.4,
                "memory_margin_pct": 14,
                "carry_forward": "Sensor replay and map cache need deterministic local storage before network upload.",
                "report_frame": "vehicle-local compute and safety margin",
                "failure_noun": "vehicle power bay",
            },
            "cloud_fleet": {
                "asset": "42U accelerator rack",
                "accelerator_label": "H100-class accelerator",
                "accelerator_unit": "GPU",
                "accel_power_kw": 0.700,
                "rack_overhead_kw": 11.0,
                "default_accels_per_rack": 32,
                "default_racks": 8,
                "max_accels_per_rack": 48,
                "max_racks": 40,
                "site_power_kw": 1250.0,
                "cooling_kw": {
                    "Air / passive": 30.0,
                    "Direct liquid / active": 45.0,
                    "High density liquid": 80.0,
                },
                "peak_tflops_per_accel": 1979.0,
                "mfu": 0.45,
                "baseline_demand_tflops": 142000.0,
                "electricity_usd_kwh": 0.11,
                "capex_per_accel_usd": 32000.0,
                "min_util_pct": 52,
                "max_util_pct": 86,
                "target_util_pct": 68,
                "hourly_cost_guardrail": 520.0,
                "carbon_guardrail_kg_hr": 430.0,
                "memory_margin_pct": 22,
                "carry_forward": "The next design must reserve nonblocking fabric and local NVMe staging for the selected rack count.",
                "report_frame": "accelerator tier, rack count, utilization, power, and TCO plan",
                "failure_noun": "rack",
            },
        }[profile.track_id]

        packet = dict(shared)
        packet.update(
            {
                "track_id": profile.track_id,
                "track_label": profile.label,
                "stakeholder": variant.stakeholder,
                "hardware_ref": variant.hardware_ref,
                "model_ref": variant.model_ref,
                "scenario_id": variant.scenario_id,
                "workload_summary": variant.workload_summary,
                "objective": variant.objective,
                "source_policy": profile.source_policy,
                "primary_metric": variant.primary_metric,
                "guardrail_metric": variant.guardrail_metric,
            }
        )
        return packet

    def v2_02_part_a_state(packet, accelerators_per_rack, rack_count, cooling_tier):
        rack_power_kw = (
            packet["rack_overhead_kw"]
            + accelerators_per_rack * packet["accel_power_kw"]
        )
        cooling_limit_kw = packet["cooling_kw"][cooling_tier]
        total_accelerators = accelerators_per_rack * rack_count
        fleet_power_kw = rack_power_kw * rack_count
        sustained_tflops = (
            total_accelerators
            * packet["peak_tflops_per_accel"]
            * packet["mfu"]
        )
        demand_tflops = packet["baseline_demand_tflops"]
        cooling_ok = rack_power_kw <= cooling_limit_kw
        site_ok = fleet_power_kw <= packet["site_power_kw"]
        capacity_ok = sustained_tflops >= demand_tflops
        if not cooling_ok:
            binding = "cooling"
        elif not site_ok:
            binding = "site power"
        elif not capacity_ok:
            binding = "sustained capacity"
        else:
            binding = "none"
        return {
            "accelerators_per_rack": accelerators_per_rack,
            "rack_count": rack_count,
            "cooling_tier": cooling_tier,
            "rack_power_kw": rack_power_kw,
            "cooling_limit_kw": cooling_limit_kw,
            "total_accelerators": total_accelerators,
            "fleet_power_kw": fleet_power_kw,
            "sustained_tflops": sustained_tflops,
            "demand_tflops": demand_tflops,
            "cooling_ok": cooling_ok,
            "site_ok": site_ok,
            "capacity_ok": capacity_ok,
            "feasible": cooling_ok and site_ok and capacity_ok,
            "binding": binding,
            "cooling_headroom_kw": cooling_limit_kw - rack_power_kw,
            "site_headroom_kw": packet["site_power_kw"] - fleet_power_kw,
            "capacity_headroom_tflops": sustained_tflops - demand_tflops,
        }

    def v2_02_part_b_state(packet, part_a, utilization_pct, demand_multiplier):
        utilization = utilization_pct / 100
        adjusted_demand = part_a["demand_tflops"] * demand_multiplier
        useful_tflops = part_a["sustained_tflops"] * utilization
        idle_tflops = max(part_a["sustained_tflops"] - useful_tflops, 0)
        pue = 1.18 if packet["track_id"] == "cloud_fleet" else 1.08
        carbon_intensity = 0.38
        carbon_kg_hr = part_a["fleet_power_kw"] * pue * carbon_intensity
        capex_hour = (
            part_a["total_accelerators"]
            * packet["capex_per_accel_usd"]
            / (3 * 365 * 24)
        )
        energy_hour = part_a["fleet_power_kw"] * pue * packet["electricity_usd_kwh"]
        hourly_cost = capex_hour + energy_hour
        idle_cost = hourly_cost * (1 - utilization)
        idle_carbon = carbon_kg_hr * (1 - utilization)
        demand_ok = useful_tflops >= adjusted_demand
        waste = utilization_pct < packet["min_util_pct"]
        saturation = utilization_pct > packet["max_util_pct"] or not demand_ok
        if waste:
            verdict = "waste"
        elif saturation:
            verdict = "saturation"
        else:
            verdict = "balanced"
        return {
            "utilization_pct": utilization_pct,
            "utilization": utilization,
            "demand_multiplier": demand_multiplier,
            "adjusted_demand_tflops": adjusted_demand,
            "useful_tflops": useful_tflops,
            "idle_tflops": idle_tflops,
            "hourly_cost": hourly_cost,
            "idle_cost": idle_cost,
            "carbon_kg_hr": carbon_kg_hr,
            "idle_carbon_kg_hr": idle_carbon,
            "demand_ok": demand_ok,
            "waste": waste,
            "saturation": saturation,
            "verdict": verdict,
            "pue": pue,
        }

    def v2_02_candidate_rows(packet, part_a, part_b, placement, role_emphasis):
        placement_effects = {
            "Rack-local / device-local": {
                "throughput": 1.00,
                "cost": 1.00,
                "carbon": 1.00,
                "risk": "low",
                "penalty": 0,
            },
            "Same facility / phone assist": {
                "throughput": 0.94,
                "cost": 1.04,
                "carbon": 0.96,
                "risk": "medium",
                "penalty": 5,
            },
            "Regional pool": {
                "throughput": 0.86,
                "cost": 0.94,
                "carbon": 0.78,
                "risk": "medium",
                "penalty": 12,
            },
            "Remote assist / cloud burst": {
                "throughput": 0.72,
                "cost": 0.82,
                "carbon": 0.62,
                "risk": "high",
                "penalty": 24,
            },
        }[placement]
        role_effects = {
            "Balanced serving/training": {"premium": 1.00, "mixed": 1.04, "efficient": 0.98, "offload": 0.92},
            "Training-heavy": {"premium": 1.12, "mixed": 1.02, "efficient": 0.84, "offload": 0.72},
            "Inference-heavy": {"premium": 0.96, "mixed": 1.08, "efficient": 1.04, "offload": 0.95},
        }[role_emphasis]
        specs = [
            {
                "id": "premium",
                "label": "All premium accelerators",
                "throughput": 1.18,
                "power": 1.20,
                "cost": 1.35,
                "carbon": 1.18,
                "memory_bonus": 12,
                "note": "fastest chips, highest rack pressure",
            },
            {
                "id": "mixed",
                "label": "Mixed train/serve fleet",
                "throughput": 1.00,
                "power": 1.00,
                "cost": 1.00,
                "carbon": 0.94,
                "memory_bonus": 6,
                "note": "premium capacity reserved for the work that needs it",
            },
            {
                "id": "efficient",
                "label": "Efficiency-skewed fleet",
                "throughput": 0.82,
                "power": 0.72,
                "cost": 0.68,
                "carbon": 0.62,
                "memory_bonus": -8,
                "note": "lower energy and cost, less memory and peak headroom",
            },
            {
                "id": "offload",
                "label": "Remote/offload assist",
                "throughput": 0.65,
                "power": 0.58,
                "cost": 0.80,
                "carbon": 0.55,
                "memory_bonus": 2,
                "note": "local pressure drops but placement dependency rises",
            },
        ]
        rows = []
        for spec in specs:
            throughput = (
                part_b["useful_tflops"]
                * spec["throughput"]
                * role_effects[spec["id"]]
                * placement_effects["throughput"]
            )
            cost = part_b["hourly_cost"] * spec["cost"] * placement_effects["cost"]
            carbon = part_b["carbon_kg_hr"] * spec["carbon"] * placement_effects["carbon"]
            memory_margin = packet["memory_margin_pct"] + spec["memory_bonus"] - placement_effects["penalty"] / 3
            power_kw = part_a["fleet_power_kw"] * spec["power"]
            violations = []
            if throughput < part_b["adjusted_demand_tflops"]:
                violations.append("throughput")
            if memory_margin < 0:
                violations.append("memory")
            if cost > packet["hourly_cost_guardrail"]:
                violations.append("cost")
            if placement_effects["risk"] == "high" and packet["track_id"] in {"robotaxi", "oura_ring"}:
                violations.append("placement")
            feasible = not violations
            score = (
                (throughput / max(part_b["adjusted_demand_tflops"], 1)) * 40
                + max(memory_margin, -20) * 0.9
                - (cost / max(packet["hourly_cost_guardrail"], 0.01)) * 18
                - (carbon / max(packet["carbon_guardrail_kg_hr"], 0.01)) * 10
                - placement_effects["penalty"] * 0.55
            )
            rows.append(
                {
                    "id": spec["id"],
                    "label": spec["label"],
                    "throughput_tflops": throughput,
                    "cost_usd_hr": cost,
                    "carbon_kg_hr": carbon,
                    "power_kw": power_kw,
                    "memory_margin_pct": memory_margin,
                    "placement": placement,
                    "role_emphasis": role_emphasis,
                    "placement_risk": placement_effects["risk"],
                    "violations": tuple(violations),
                    "feasible": feasible,
                    "score": score,
                    "note": spec["note"],
                }
            )
        feasible_rows = [row for row in rows if row["feasible"]]
        recommended = max(feasible_rows or rows, key=lambda row: row["score"])
        rejected = min(rows, key=lambda row: (row["feasible"], row["score"]))
        return rows, recommended, rejected

    def v2_02_part_d_state(packet, part_a, part_b, selected, margin_pct, carbon_region, procurement_stance):
        region_intensity = {
            "Low-carbon region": 0.08,
            "Average grid": 0.38,
            "Constrained fossil grid": 0.62,
        }[carbon_region]
        procurement = {
            "Buy fixed capacity": {"cost": 0.95, "util_shift": -4, "risk": "depreciation"},
            "Reserved cloud capacity": {"cost": 1.12, "util_shift": 0, "risk": "placement"},
            "Hybrid burst": {"cost": 1.04, "util_shift": -7, "risk": "network/storage handoff"},
        }[procurement_stance]
        margin = margin_pct / 100
        effective_power_kw = selected["power_kw"] * (1 + margin)
        effective_rack_kw = part_a["rack_power_kw"] * (
            selected["power_kw"] / max(part_a["fleet_power_kw"], 0.001)
        ) * part_a["rack_count"]
        utilization_after_margin = max(0, part_b["utilization_pct"] + procurement["util_shift"]) / (1 + margin)
        cost_usd_hr = selected["cost_usd_hr"] * (1 + margin) * procurement["cost"]
        carbon_kg_hr = (
            effective_power_kw
            * part_b["pue"]
            * region_intensity
            * (selected["carbon_kg_hr"] / max(part_b["carbon_kg_hr"], 0.001))
        )
        power_ok = effective_power_kw <= packet["site_power_kw"]
        cooling_ok = effective_rack_kw / max(part_a["rack_count"], 1) <= part_a["cooling_limit_kw"]
        util_ok = packet["min_util_pct"] <= utilization_after_margin <= packet["max_util_pct"]
        cost_ok = cost_usd_hr <= packet["hourly_cost_guardrail"]
        carbon_ok = carbon_kg_hr <= packet["carbon_guardrail_kg_hr"]
        checks = [
            ("power", power_ok),
            ("cooling", cooling_ok),
            ("utilization", util_ok),
            ("cost", cost_ok),
            ("carbon", carbon_ok),
        ]
        failed = [name for name, ok in checks if not ok]
        binding = failed[0] if failed else "none"
        feasible = not failed
        if feasible:
            verdict = "approve"
        elif len(failed) <= 2:
            verdict = "revise"
        else:
            verdict = "reject"
        return {
            "carbon_region": carbon_region,
            "region_intensity": region_intensity,
            "procurement_stance": procurement_stance,
            "procurement_risk": procurement["risk"],
            "capacity_margin_pct": margin_pct,
            "effective_power_kw": effective_power_kw,
            "effective_rack_kw": effective_rack_kw,
            "utilization_after_margin_pct": utilization_after_margin,
            "cost_usd_hr": cost_usd_hr,
            "carbon_kg_hr": carbon_kg_hr,
            "power_ok": power_ok,
            "cooling_ok": cooling_ok,
            "utilization_ok": util_ok,
            "cost_ok": cost_ok,
            "carbon_ok": carbon_ok,
            "failed": tuple(failed),
            "binding": binding,
            "feasible": feasible,
            "verdict": verdict,
        }

    def v2_02_status_badge(ok, label):
        tone = "#15803d" if ok else "#b91c1c"
        text = "PASS" if ok else "FAIL"
        return (
            f"<span style='display:inline-block; padding:2px 8px; border-radius:999px; "
            f"background:{tone}; color:white; font-size:0.72rem; font-weight:700;'>"
            f"{text}: {v2_02_escape(label)}</span>"
        )

    def v2_02_fields_html(fields):
        return "\n".join(
            (
                "<div class='mlsysbook-field'>"
                f"<strong>{v2_02_escape(key)}</strong>{v2_02_escape(value)}"
                "</div>"
            )
            for key, value in fields.items()
        )

    def v2_02_metric_cards_html(cards):
        chunks = []
        for title, value, detail, color in cards:
            chunks.append(
                f"""
<div style="flex:1; min-width:170px; background:white; border:1px solid #e2e8f0;
            border-top:3px solid {color}; border-radius:8px; padding:14px 16px;">
  <div style="font-size:0.72rem; color:#64748b; font-weight:700; text-transform:uppercase;">
    {v2_02_escape(title)}
  </div>
  <div style="font-size:1.35rem; font-weight:800; color:{color}; margin-top:4px;">
    {v2_02_escape(value)}
  </div>
  <div style="font-size:0.78rem; color:#475569; line-height:1.35;">{v2_02_escape(detail)}</div>
</div>
"""
            )
        return "<div style='display:flex; flex-wrap:wrap; gap:12px; margin:14px 0;'>" + "\n".join(chunks) + "</div>"

    def v2_02_markdown_table(headers, rows):
        header = "| " + " | ".join(headers) + " |"
        sep = "| " + " | ".join("---" for _ in headers) + " |"
        body = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
        return "\n".join([header, sep] + body)

    def v2_02_select_row(rows, selected_id):
        for row in rows:
            if row["id"] == selected_id:
                return row
        return max([row for row in rows if row["feasible"]] or rows, key=lambda row: row["score"])

    def v2_02_rejected_row(rows, rejected_id, selected_id):
        for row in rows:
            if row["id"] == rejected_id:
                return row
        candidates = [row for row in rows if row["id"] != selected_id]
        return min(candidates or rows, key=lambda row: (row["feasible"], row["score"]))

    return (
        v2_02_candidate_rows,
        v2_02_escape,
        v2_02_fields_html,
        v2_02_markdown_table,
        v2_02_metric_cards_html,
        v2_02_part_a_state,
        v2_02_part_b_state,
        v2_02_part_d_state,
        v2_02_rejected_row,
        v2_02_select_row,
        v2_02_status_badge,
        v2_02_track_packet,
    )


@app.cell
def _(v2_02_profile, v2_02_track_packet, v2_02_variant):
    v2_02_packet = v2_02_track_packet(v2_02_profile, v2_02_variant)
    return (v2_02_packet,)


@app.cell(hide_code=True)
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, track_arc_context, track_context, v2_02_metadata, v2_02_packet, v2_02_profile):
    mo.vstack(
        [
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            mo.Html(
                f"""
<div style="background:#0f172a; color:white; padding:32px 40px; border-radius:12px; margin-bottom:10px;">
  <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.16em; color:#94a3b8; text-transform:uppercase;">
    Machine Learning Systems - Volume II - Lab 02
  </div>
  <h1 style="margin:8px 0 8px 0; color:#f8fafc; font-size:2.15rem; line-height:1.1;">
    The Compute Infrastructure Wall
  </h1>
  <p style="margin:0; color:#cbd5e1; max-width:850px; line-height:1.55;">
    Datacenter compute is constrained infrastructure. Power, cooling, accelerator mix,
    placement, utilization, cost, and carbon are coupled budgets; peak FLOPs matter
    only after those budgets can sustain the plan.
  </p>
  <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:18px;">
    <span class="badge badge-info">{v2_02_profile.label}</span>
    <span class="badge badge-warn">{v2_02_packet['accelerator_label']}</span>
    <span class="badge badge-fail">{v2_02_packet['hardware_ref']}</span>
  </div>
</div>
"""
            ),
            track_context(v2_02_profile),
            track_arc_context(v2_02_profile, v2_02_metadata.lab_id),
            mo.Html(
                """
<div class="mlsysbook-panel">
  <div class="mlsysbook-section-label">Shared Concept Sequence</div>
  <div class="mlsysbook-compact-fields">
    <div class="mlsysbook-field"><strong>Part A</strong>Rack, power, and cooling budgets constrain accelerators before peak FLOPs do.</div>
    <div class="mlsysbook-field"><strong>Part B</strong>Utilization converts capacity into economics and waste.</div>
    <div class="mlsysbook-field"><strong>Part C</strong>Accelerator mix and placement change throughput, memory, cost, and carbon.</div>
    <div class="mlsysbook-field"><strong>Part D</strong>The recommendation must satisfy power, utilization, cost, and carbon guardrails.</div>
  </div>
</div>
"""
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, v2_02_packet):
    pA_pred = mo.ui.radio(
        options={
            "Peak FLOPs should bind first": "compute",
            "Power delivery should bind first": "power",
            "Cooling should bind first": "cooling",
            "Cost should bind first": "cost",
        },
        label=f"Part A prediction: what will reject the first {v2_02_packet['report_frame']}?",
    )
    pA_accels = mo.ui.slider(
        start=1,
        stop=v2_02_packet["max_accels_per_rack"],
        value=v2_02_packet["default_accels_per_rack"],
        step=1,
        label=f"{v2_02_packet['accelerator_unit'].title()}s per {v2_02_packet['asset']}",
    )
    pA_racks = mo.ui.slider(
        start=1,
        stop=v2_02_packet["max_racks"],
        value=v2_02_packet["default_racks"],
        step=1,
        label=f"{v2_02_packet['asset'].title()} count",
    )
    pA_cooling = mo.ui.dropdown(
        options=tuple(v2_02_packet["cooling_kw"].keys()),
        value="Air / passive",
        label="Cooling / thermal tier",
    )
    pA_checkpoint = mo.ui.radio(
        options={
            "Carry forward cooling as the binding budget": "cooling",
            "Carry forward site power as the binding budget": "site power",
            "Carry forward sustained capacity as the binding budget": "sustained capacity",
            "Carry forward no binding budget": "none",
        },
        label="Part A checkpoint: which infrastructure budget belongs in the memo?",
    )
    return (pA_accels, pA_checkpoint, pA_cooling, pA_pred, pA_racks)


@app.cell(hide_code=True)
def _(mo, v2_02_packet):
    pB_pred = mo.ui.radio(
        options={
            "Maximum utilization is always best": "max",
            "Low utilization is safer and therefore best": "low",
            "A guarded utilization band is the defensible target": "band",
            "Utilization does not affect cost or carbon": "irrelevant",
        },
        label="Part B prediction: how should utilization be treated?",
    )
    pB_util = mo.ui.slider(
        start=10,
        stop=98,
        value=v2_02_packet["target_util_pct"],
        step=1,
        label="Utilization target (%)",
    )
    pB_demand = mo.ui.slider(
        start=0.5,
        stop=1.8,
        value=1.0,
        step=0.05,
        label="Demand multiplier",
    )
    pB_checkpoint = mo.ui.radio(
        options={
            "Use this target because it stays inside the utilization band": "accept",
            "Revise lower to reduce saturation risk": "revise_lower",
            "Revise higher to reduce idle waste": "revise_higher",
        },
        label="Part B checkpoint: what utilization decision feeds Part D?",
    )
    return (pB_checkpoint, pB_demand, pB_pred, pB_util)


@app.cell(hide_code=True)
def _(mo):
    pC_pred = mo.ui.radio(
        options={
            "All premium accelerators": "premium",
            "Mixed train/serve fleet": "mixed",
            "Efficiency-skewed fleet": "efficient",
            "Remote/offload assist": "offload",
        },
        label="Part C prediction: which accelerator mix/placement will survive the guardrails?",
    )
    pC_placement = mo.ui.dropdown(
        options=(
            "Rack-local / device-local",
            "Same facility / phone assist",
            "Regional pool",
            "Remote assist / cloud burst",
        ),
        value="Rack-local / device-local",
        label="Placement assumption",
    )
    pC_role = mo.ui.dropdown(
        options=("Balanced serving/training", "Training-heavy", "Inference-heavy"),
        value="Balanced serving/training",
        label="Workload role emphasis",
    )
    pC_choice = mo.ui.radio(
        options={
            "Select all premium accelerators": "premium",
            "Select mixed train/serve fleet": "mixed",
            "Select efficiency-skewed fleet": "efficient",
            "Select remote/offload assist": "offload",
        },
        label="Part C checkpoint: selected capacity mix",
    )
    pC_reject = mo.ui.radio(
        options={
            "Reject all premium accelerators": "premium",
            "Reject mixed train/serve fleet": "mixed",
            "Reject efficiency-skewed fleet": "efficient",
            "Reject remote/offload assist": "offload",
        },
        label="Part C checkpoint: rejected alternative",
    )
    return (pC_choice, pC_placement, pC_pred, pC_reject, pC_role)


@app.cell(hide_code=True)
def _(mo):
    pD_pred = mo.ui.radio(
        options={
            "Power or cooling will still bind": "power",
            "Utilization will still bind": "utilization",
            "Cost will still bind": "cost",
            "Carbon will still bind": "carbon",
        },
        label="Part D prediction: which guardrail will reject the recommendation if any?",
    )
    pD_margin = mo.ui.slider(
        start=0,
        stop=35,
        value=12,
        step=1,
        label="Capacity reserve margin (%)",
    )
    pD_region = mo.ui.dropdown(
        options=("Low-carbon region", "Average grid", "Constrained fossil grid"),
        value="Average grid",
        label="Carbon region",
    )
    pD_procurement = mo.ui.dropdown(
        options=("Buy fixed capacity", "Reserved cloud capacity", "Hybrid burst"),
        value="Reserved cloud capacity",
        label="Procurement stance",
    )
    pD_decision = mo.ui.radio(
        options={
            "Approve the plan": "approve",
            "Revise the plan before approval": "revise",
            "Reject the plan and restart sizing": "reject",
        },
        label="Part D checkpoint: final infrastructure recommendation",
    )
    return (pD_decision, pD_margin, pD_pred, pD_procurement, pD_region)


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    pA_accels,
    pA_checkpoint,
    pA_cooling,
    pA_pred,
    pA_racks,
    pB_checkpoint,
    pB_demand,
    pB_pred,
    pB_util,
    pC_choice,
    pC_placement,
    pC_pred,
    pC_reject,
    pC_role,
    pD_decision,
    pD_margin,
    pD_pred,
    pD_procurement,
    pD_region,
    v2_02_candidate_rows,
    v2_02_fields_html,
    v2_02_markdown_table,
    v2_02_metric_cards_html,
    v2_02_packet,
    v2_02_part_a_state,
    v2_02_part_b_state,
    v2_02_part_d_state,
    v2_02_rejected_row,
    v2_02_select_row,
    v2_02_status_badge,
):
    def build_part_a():
        items = [
            mo.Html(
                f"""
<div class="mlsysbook-panel">
  <div class="mlsysbook-section-label">Part A - Concept Module</div>
  <h2>Rack, Power, And Cooling Bind Before Peak FLOPs</h2>
  <p>{v2_02_packet['stakeholder']} needs a feasible {v2_02_packet['report_frame']}.
  The first question is not how many peak FLOPs the plan advertises; it is whether
  the {v2_02_packet['asset']} can receive power and reject heat.</p>
  <div class="mlsysbook-compact-fields">
    {v2_02_fields_html({
        "Chapter claim": "Selecting the fastest accelerator is counterproductive if cooling cannot remove the heat.",
        "Your decision": "Choose the binding infrastructure budget to carry forward.",
        "Track consequence": v2_02_packet["failure_noun"] + " failure changes what enough compute means.",
    })}
  </div>
</div>
"""
            ),
            mo.md("### 1. Structured Prediction"),
            pA_pred,
        ]
        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Commit to the binding-budget prediction to unlock the rack instrument."), kind="warn"))
            return mo.vstack(items)

        state = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
        items += [
            mo.md("### 2. Manipulate The Physical Envelope"),
            mo.hstack([pA_accels, pA_racks, pA_cooling], gap="1rem"),
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Rack power", "Cooling limit", "Site power", "Fleet power"],
                y=[
                    state["rack_power_kw"],
                    state["cooling_limit_kw"],
                    v2_02_packet["site_power_kw"],
                    state["fleet_power_kw"],
                ],
                marker_color=[COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["BlueLine"], COLORS["RedLine"]],
                text=[
                    f"{state['rack_power_kw']:.2f} kW",
                    f"{state['cooling_limit_kw']:.2f} kW",
                    f"{v2_02_packet['site_power_kw']:.2f} kW",
                    f"{state['fleet_power_kw']:.2f} kW",
                ],
                textposition="auto",
            )
        )
        fig.update_layout(height=360, yaxis_title="kW", margin=dict(l=50, r=20, t=30, b=40))
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        items.append(
            mo.Html(
                v2_02_metric_cards_html(
                    [
                        ("Accelerators", f"{state['total_accelerators']}", f"{pA_racks.value} x {pA_accels.value}", COLORS["BlueLine"]),
                        ("Rack power", f"{state['rack_power_kw']:.2f} kW", f"limit {state['cooling_limit_kw']:.2f} kW", COLORS["OrangeLine"]),
                        ("Sustained capacity", f"{state['sustained_tflops']:.0f}", "TFLOP/s-equivalent", COLORS["GreenLine"]),
                        ("Binding budget", state["binding"], "after power/cooling/capacity checks", COLORS["RedLine"] if not state["feasible"] else COLORS["GreenLine"]),
                    ]
                )
            )
        )
        rows = [
            ("Cooling", f"{state['rack_power_kw']:.3f} kW/rack", f"{state['cooling_limit_kw']:.3f} kW/rack", "pass" if state["cooling_ok"] else "fail"),
            ("Site power", f"{state['fleet_power_kw']:.3f} kW", f"{v2_02_packet['site_power_kw']:.3f} kW", "pass" if state["site_ok"] else "fail"),
            ("Sustained capacity", f"{state['sustained_tflops']:.1f}", f"{state['demand_tflops']:.1f}", "pass" if state["capacity_ok"] else "fail"),
        ]
        items.append(mo.md(v2_02_markdown_table(("Budget", "Current", "Limit / Need", "Status"), rows)))
        if state["feasible"]:
            items.append(mo.callout(mo.md(f"**Recovered envelope.** The current plan satisfies rack cooling, site power, and sustained capacity with `{state['binding']}` as the binding budget."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"**Infrastructure violation:** `{state['binding']}` rejects the plan. Change cooling tier, reduce density, or adjust rack count until the physical budget recovers."), kind="danger"))
        if pA_pred.value == state["binding"] or (pA_pred.value == "power" and state["binding"] == "site power"):
            items.append(mo.callout(mo.md("**Prediction check:** your prior matched the binding infrastructure budget."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"**Prediction check:** the instrument found `{state['binding']}`, not `{pA_pred.value}`. Peak capacity is not the first question when the physical envelope fails."), kind="warn"))
        items += [
            mo.accordion(
                {
                    "Math Peek / Source Model - rack power and cooling": mo.md(
                        f"""
**Formula**

`rack_power_kw = rack_overhead_kw + accelerators_per_rack * accelerator_power_kw`

`fleet_power_kw = rack_power_kw * rack_count`

`sustained_capacity = accelerators * peak_per_accelerator * MFU`

**Chapter anchor**

`#sec-compute-rack` and `#sec-compute-power-wall` explain why dense AI racks
cross air-cooling and power-delivery envelopes before advertised peak FLOPs
settle the plan.

**Source model**

Hardware identity comes from `{v2_02_packet['hardware_ref']}`. Track thresholds
are notebook-local teaching envelopes for this Wave 5 lab and are recorded in
the report source trace.
"""
                    )
                }
            ),
            mo.md("### 7. Checkpoint"),
            pA_checkpoint,
        ]
        return mo.vstack(items)

    def build_part_b():
        part_a = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
        state = v2_02_part_b_state(v2_02_packet, part_a, pB_util.value, pB_demand.value)
        items = [
            mo.Html(
                f"""
<div class="mlsysbook-panel">
  <div class="mlsysbook-section-label">Part B - Concept Module</div>
  <h2>Utilization Converts Capacity Into Economics And Waste</h2>
  <p>The same physical capacity from Part A can be wasteful, healthy, or saturated.
  Utilization is the conversion factor between bought infrastructure and useful work.</p>
</div>
"""
            ),
            mo.md("### 1. Structured Prediction"),
            pB_pred,
        ]
        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Commit to a utilization prediction to unlock the economics instrument."), kind="warn"))
            return mo.vstack(items)

        items += [mo.md("### 2. Manipulate Utilization And Demand"), mo.hstack([pB_util, pB_demand], gap="1rem")]
        xs = list(range(15, 99, 5))
        useful = []
        idle_cost = []
        carbon_waste = []
        for pct in xs:
            row = v2_02_part_b_state(v2_02_packet, part_a, pct, pB_demand.value)
            useful.append(row["useful_tflops"])
            idle_cost.append(row["idle_cost"])
            carbon_waste.append(row["idle_carbon_kg_hr"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=useful, mode="lines+markers", name="Useful throughput", line=dict(color=COLORS["BlueLine"])))
        fig.add_trace(go.Scatter(x=xs, y=idle_cost, mode="lines+markers", name="Idle cost ($/hr)", yaxis="y2", line=dict(color=COLORS["OrangeLine"])))
        fig.add_vrect(x0=v2_02_packet["min_util_pct"], x1=v2_02_packet["max_util_pct"], fillcolor="rgba(34,197,94,0.12)", line_width=0)
        fig.add_vline(x=pB_util.value, line_dash="dash", line_color=COLORS["RedLine"])
        fig.update_layout(
            height=380,
            xaxis_title="Utilization target (%)",
            yaxis=dict(title="Useful TFLOP/s-equivalent"),
            yaxis2=dict(title="Idle cost ($/hr)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.18),
            margin=dict(l=55, r=55, t=30, b=55),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        items.append(
            mo.Html(
                v2_02_metric_cards_html(
                    [
                        ("Useful throughput", f"{state['useful_tflops']:.0f}", f"demand {state['adjusted_demand_tflops']:.0f}", COLORS["BlueLine"]),
                        ("Idle cost", f"${state['idle_cost']:.2f}/hr", "capacity paid for but unused", COLORS["OrangeLine"]),
                        ("Carbon waste", f"{state['idle_carbon_kg_hr']:.2f} kg/hr", "idle share of facility emissions", COLORS["RedLine"]),
                        ("Verdict", state["verdict"], f"healthy band {v2_02_packet['min_util_pct']}-{v2_02_packet['max_util_pct']}%", COLORS["GreenLine"] if state["verdict"] == "balanced" else COLORS["RedLine"]),
                    ]
                )
            )
        )
        rows = [
            ("Utilization target", f"{state['utilization_pct']:.0f}%", f"{v2_02_packet['min_util_pct']}-{v2_02_packet['max_util_pct']}%", state["verdict"]),
            ("Useful capacity", f"{state['useful_tflops']:.1f}", f"{state['adjusted_demand_tflops']:.1f}", "pass" if state["demand_ok"] else "shortfall"),
            ("Hourly cost", f"${state['hourly_cost']:.2f}", f"${v2_02_packet['hourly_cost_guardrail']:.2f}", "context"),
            ("Idle carbon", f"{state['idle_carbon_kg_hr']:.2f} kg/hr", "lower is better", "waste" if state["waste"] else "bounded"),
        ]
        items.append(mo.md(v2_02_markdown_table(("Amount", "Current", "Reference", "Status"), rows)))
        if state["verdict"] == "balanced":
            items.append(mo.callout(mo.md("**Utilization target is defensible.** The plan turns capacity into useful work without erasing operating headroom."), kind="success"))
        elif state["verdict"] == "waste":
            items.append(mo.callout(mo.md("**Waste boundary reached.** Bought accelerators are powered and cooled while too little useful work is extracted."), kind="danger"))
        else:
            items.append(mo.callout(mo.md("**Saturation boundary reached.** The plan may look efficient, but queue/capacity headroom is gone."), kind="danger"))
        if pB_pred.value == "band" and state["verdict"] == "balanced":
            items.append(mo.callout(mo.md("**Prediction check:** correct. Utilization is a guarded band, not a one-way maximize knob."), kind="success"))
        elif pB_pred.value in {"max", "low", "irrelevant"}:
            items.append(mo.callout(mo.md("**Prediction check:** the evidence rejects one-sided utilization rules. Economics, carbon, and queue headroom move together."), kind="warn"))
        items += [
            mo.accordion(
                {
                    "Math Peek / Source Model - useful capacity and waste": mo.md(
                        """
**Formula**

`useful_capacity = sustained_capacity * utilization`

`idle_cost = hourly_cost * (1 - utilization)`

`idle_carbon = facility_power_kw * PUE * carbon_intensity * (1 - utilization)`

**Chapter anchor**

`#sec-compute-infrastructure-peak-vs-sustained-throughput-625a` and
`#sec-compute-summary` frame sustained throughput and utilization as first-order
capacity-planning quantities.
"""
                    )
                }
            ),
            mo.md("### 7. Checkpoint"),
            pB_checkpoint,
        ]
        return mo.vstack(items)

    def build_part_c():
        part_a = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
        part_b = v2_02_part_b_state(v2_02_packet, part_a, pB_util.value, pB_demand.value)
        rows, recommended, rejected_default = v2_02_candidate_rows(v2_02_packet, part_a, part_b, pC_placement.value, pC_role.value)
        selected = v2_02_select_row(rows, pC_choice.value)
        rejected = v2_02_rejected_row(rows, pC_reject.value, selected["id"])
        items = [
            mo.Html(
                """
<div class="mlsysbook-panel">
  <div class="mlsysbook-section-label">Part C - Concept Module</div>
  <h2>Accelerator Mix And Placement Change The Plan</h2>
  <p>A fleet is not just a larger accelerator. Training, inference, memory bandwidth,
  placement, cost, and carbon can prefer different capacity mixes.</p>
</div>
"""
            ),
            mo.md("### 1. Structured Prediction"),
            pC_pred,
        ]
        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Choose a mix prediction to unlock the placement comparison."), kind="warn"))
            return mo.vstack(items)

        items += [mo.md("### 2. Manipulate Mix Context"), mo.hstack([pC_placement, pC_role], gap="1rem")]
        fig = go.Figure()
        for row in rows:
            color = COLORS["GreenLine"] if row["feasible"] else COLORS["RedLine"]
            fig.add_trace(
                go.Scatter(
                    x=[row["cost_usd_hr"]],
                    y=[row["throughput_tflops"]],
                    mode="markers+text",
                    text=[row["id"]],
                    textposition="top center",
                    marker=dict(size=max(10, min(34, row["carbon_kg_hr"] / max(v2_02_packet["carbon_guardrail_kg_hr"], 0.01) * 20)), color=color),
                    name=row["label"],
                )
            )
        fig.add_vline(x=v2_02_packet["hourly_cost_guardrail"], line_dash="dash", line_color=COLORS["OrangeLine"])
        fig.add_hline(y=part_b["adjusted_demand_tflops"], line_dash="dash", line_color=COLORS["BlueLine"])
        fig.update_layout(
            height=390,
            xaxis_title="Cost ($/hr)",
            yaxis_title="Useful TFLOP/s-equivalent",
            showlegend=False,
            margin=dict(l=55, r=20, t=30, b=45),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        table_rows = []
        for row in rows:
            table_rows.append(
                (
                    row["label"],
                    f"{row['throughput_tflops']:.0f}",
                    f"{row['memory_margin_pct']:.1f}%",
                    f"${row['cost_usd_hr']:.2f}",
                    f"{row['carbon_kg_hr']:.2f}",
                    "pass" if row["feasible"] else ", ".join(row["violations"]),
                )
            )
        items.append(mo.md(v2_02_markdown_table(("Candidate", "Throughput", "Memory", "Cost/hr", "Carbon/hr", "Status"), table_rows)))
        items.append(
            mo.Html(
                v2_02_metric_cards_html(
                    [
                        ("Recommended", recommended["label"], recommended["note"], COLORS["GreenLine"] if recommended["feasible"] else COLORS["OrangeLine"]),
                        ("Selected", selected["label"], "student checkpoint or recommended default", COLORS["BlueLine"]),
                        ("Rejected", rejected["label"], ", ".join(rejected["violations"]) or rejected["note"], COLORS["RedLine"]),
                    ]
                )
            )
        )
        if pC_pred.value == recommended["id"]:
            items.append(mo.callout(mo.md("**Prediction check:** your predicted mix matches the current recommended candidate."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"**Prediction check:** current evidence favors `{recommended['label']}` under this placement and workload role."), kind="warn"))
        if not selected["feasible"]:
            items.append(mo.callout(mo.md(f"**Selected mix fails:** {', '.join(selected['violations'])}. A plan can be locally appealing and still fail the amount-system guardrails."), kind="danger"))
        else:
            items.append(mo.callout(mo.md(f"**Selected mix passes Part C.** It still needs Part D's simultaneous power, utilization, cost, and carbon guardrails."), kind="success"))
        items += [
            mo.accordion(
                {
                    "Math Peek / Source Model - mix and placement score": mo.md(
                        """
**Formula**

Each candidate transforms the Part B capacity:

`candidate_throughput = useful_capacity * mix_factor * role_factor * placement_factor`

`candidate_cost = hourly_cost * mix_cost_factor * placement_cost_factor`

`candidate_carbon = carbon_kg_hr * mix_carbon_factor * placement_carbon_factor`

**Chapter anchor**

`#sec-compute-accelerator-selection`, `#sec-compute-bandwidth-hierarchy`, and
`#sec-compute-fallacies-pitfalls` warn against treating one accelerator-hour as
interchangeable with another.
"""
                    )
                }
            ),
            mo.md("### 7. Checkpoint"),
            mo.hstack([pC_choice, pC_reject], gap="1rem"),
        ]
        return mo.vstack(items)

    def build_part_d():
        part_a = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
        part_b = v2_02_part_b_state(v2_02_packet, part_a, pB_util.value, pB_demand.value)
        rows, recommended, _ = v2_02_candidate_rows(v2_02_packet, part_a, part_b, pC_placement.value, pC_role.value)
        selected = v2_02_select_row(rows, pC_choice.value or recommended["id"])
        state = v2_02_part_d_state(v2_02_packet, part_a, part_b, selected, pD_margin.value, pD_region.value, pD_procurement.value)
        items = [
            mo.Html(
                """
<div class="mlsysbook-panel">
  <div class="mlsysbook-section-label">Part D - Concept Module</div>
  <h2>Recommendation Under Simultaneous Guardrails</h2>
  <p>The winning mix from Part C is not launch-ready until power, cooling,
  utilization, cost, and carbon all pass together.</p>
</div>
"""
            ),
            mo.md("### 1. Structured Prediction"),
            pD_pred,
        ]
        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Predict the rejecting guardrail to unlock the recommendation scorecard."), kind="warn"))
            return mo.vstack(items)

        items += [mo.md("### 2. Manipulate Final Guardrails"), mo.hstack([pD_margin, pD_region, pD_procurement], gap="1rem")]
        guardrail_rows = [
            ("Power", state["effective_power_kw"], v2_02_packet["site_power_kw"], state["power_ok"]),
            ("Rack cooling", state["effective_rack_kw"] / max(part_a["rack_count"], 1), part_a["cooling_limit_kw"], state["cooling_ok"]),
            ("Utilization", state["utilization_after_margin_pct"], v2_02_packet["max_util_pct"], state["utilization_ok"]),
            ("Cost", state["cost_usd_hr"], v2_02_packet["hourly_cost_guardrail"], state["cost_ok"]),
            ("Carbon", state["carbon_kg_hr"], v2_02_packet["carbon_guardrail_kg_hr"], state["carbon_ok"]),
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[row[0] for row in guardrail_rows],
                y=[row[1] / max(row[2], 0.001) for row in guardrail_rows],
                marker_color=[COLORS["GreenLine"] if row[3] else COLORS["RedLine"] for row in guardrail_rows],
                text=[f"{row[1] / max(row[2], 0.001):.2f}x" for row in guardrail_rows],
                textposition="auto",
            )
        )
        fig.add_hline(y=1, line_dash="dash", line_color=COLORS["OrangeLine"])
        fig.update_layout(height=360, yaxis_title="Current / limit ratio", margin=dict(l=55, r=20, t=30, b=45))
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        status_html = " ".join(v2_02_status_badge(ok, label) for label, _, _, ok in guardrail_rows)
        items.append(mo.Html(f"<div style='display:flex; flex-wrap:wrap; gap:8px; margin:12px 0;'>{status_html}</div>"))
        table_rows = [
            ("Power", f"{state['effective_power_kw']:.2f} kW", f"{v2_02_packet['site_power_kw']:.2f} kW", "pass" if state["power_ok"] else "fail"),
            ("Cooling", f"{state['effective_rack_kw'] / max(part_a['rack_count'], 1):.2f} kW/enclosure", f"{part_a['cooling_limit_kw']:.2f}", "pass" if state["cooling_ok"] else "fail"),
            ("Utilization", f"{state['utilization_after_margin_pct']:.1f}%", f"{v2_02_packet['min_util_pct']}-{v2_02_packet['max_util_pct']}%", "pass" if state["utilization_ok"] else "fail"),
            ("Cost", f"${state['cost_usd_hr']:.2f}/hr", f"${v2_02_packet['hourly_cost_guardrail']:.2f}/hr", "pass" if state["cost_ok"] else "fail"),
            ("Carbon", f"{state['carbon_kg_hr']:.2f} kg/hr", f"{v2_02_packet['carbon_guardrail_kg_hr']:.2f} kg/hr", "pass" if state["carbon_ok"] else "fail"),
        ]
        items.append(mo.md(v2_02_markdown_table(("Guardrail", "Current", "Limit", "Status"), table_rows)))
        if state["feasible"]:
            items.append(mo.callout(mo.md("**Recommendation passes.** The selected capacity plan satisfies all simultaneous guardrails."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"**Recommendation not ready:** `{state['binding']}` is the first failed guardrail. The memo should revise or reject the plan."), kind="danger"))
        if pD_pred.value == state["binding"] or (pD_pred.value == "power" and state["binding"] in {"power", "cooling"}):
            items.append(mo.callout(mo.md("**Prediction check:** your final guardrail prediction matches the scorecard."), kind="success"))
        elif state["binding"] == "none":
            items.append(mo.callout(mo.md("**Prediction check:** no guardrail failed under the current controls. The memo can approve if the residual risk is named."), kind="info"))
        else:
            items.append(mo.callout(mo.md(f"**Prediction check:** `{state['binding']}` rejected the plan first, not `{pD_pred.value}`."), kind="warn"))
        items += [
            mo.accordion(
                {
                    "Math Peek / Source Model - simultaneous guardrail model": mo.md(
                        """
**Formula**

`feasible = power_ok and cooling_ok and utilization_ok and cost_ok and carbon_ok`

`carbon_kg_hr = power_kw * PUE * region_carbon_intensity * mix_carbon_factor`

`cost_usd_hr = selected_cost * capacity_margin * procurement_factor`

**Chapter anchor**

`#sec-compute-summary` states the constraint cascade: accelerator power
determines rack cooling, rack density determines pod layout, and scaling
efficiency determines whether economics close.
"""
                    )
                }
            ),
            mo.md("### 7. Checkpoint"),
            pD_decision,
        ]
        return mo.vstack(items)

    def build_synthesis():
        part_a = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
        part_b = v2_02_part_b_state(v2_02_packet, part_a, pB_util.value, pB_demand.value)
        rows, recommended, rejected_default = v2_02_candidate_rows(v2_02_packet, part_a, part_b, pC_placement.value, pC_role.value)
        selected = v2_02_select_row(rows, pC_choice.value or recommended["id"])
        rejected = v2_02_rejected_row(rows, pC_reject.value or rejected_default["id"], selected["id"])
        part_d = v2_02_part_d_state(v2_02_packet, part_a, part_b, selected, pD_margin.value, pD_region.value, pD_procurement.value)
        chosen_plan = (
            f"{selected['label']} across {part_a['rack_count']} {v2_02_packet['asset']}(s), "
            f"{part_a['total_accelerators']} {v2_02_packet['accelerator_unit']}(s), "
            f"{pB_util.value}% target utilization, {pD_margin.value}% reserve margin"
        )
        memo_rows = [
            ("Chosen capacity plan", chosen_plan),
            ("Binding infrastructure budget", part_d["binding"] if part_d["binding"] != "none" else part_a["binding"]),
            ("Rejected alternative", f"{rejected['label']} ({', '.join(rejected['violations']) or rejected['note']})"),
            ("Power/cooling evidence", f"{part_d['effective_power_kw']:.2f} kW fleet power; {part_a['rack_power_kw']:.2f} kW per enclosure"),
            ("Utilization/cost/carbon evidence", f"{part_d['utilization_after_margin_pct']:.1f}% util; ${part_d['cost_usd_hr']:.2f}/hr; {part_d['carbon_kg_hr']:.2f} kg/hr"),
            ("Carry-forward implication", v2_02_packet["carry_forward"]),
        ]
        verdict_text = {
            "approve": "Approve the plan",
            "revise": "Revise before approval",
            "reject": "Reject and restart sizing",
        }.get(pD_decision.value or part_d["verdict"], "Decision not recorded")
        return mo.vstack(
            [
                mo.Html(
                    f"""
<div class="mlsysbook-panel">
  <div class="mlsysbook-section-label">Synthesis - Compute Infrastructure Memo</div>
  <h2>{verdict_text}</h2>
  <p>The memo must state the selected capacity plan, the binding infrastructure
  budget, a rejected alternative, and the network/storage implication that
  carries into the next labs.</p>
  <div class="mlsysbook-compact-fields">{v2_02_fields_html(dict(memo_rows))}</div>
</div>
"""
                ),
                mo.callout(
                    mo.md(
                        f"**Durable invariant:** {v2_02_packet['track_label']} infrastructure is an amount system: "
                        "chips become power, heat, useful throughput, cost, carbon, and network/storage obligations."
                    ),
                    kind="info",
                ),
                mo.accordion(
                    {
                        "Math Peek / Source Model - memo evidence packet": mo.md(
                            """
The final report is defensible only if it carries forward all four evidence types:

1. Physical envelope: rack/enclosure power and cooling.
2. Operating envelope: utilization target and demand headroom.
3. Procurement envelope: accelerator mix and rejected alternative.
4. Governance envelope: cost, carbon, and the next network/storage implication.
"""
                        )
                    }
                ),
            ]
        )

    v2_02_tabs = mo.ui.tabs(
        {
            "Part A - Rack/Power/Cooling": build_part_a(),
            "Part B - Utilization": build_part_b(),
            "Part C - Mix/Placement": build_part_c(),
            "Part D - Guardrails": build_part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    v2_02_tabs
    return (v2_02_tabs,)


@app.cell(hide_code=True)
def _(
    COLORS,
    ledger,
    mo,
    pA_accels,
    pA_checkpoint,
    pA_cooling,
    pA_pred,
    pA_racks,
    pB_checkpoint,
    pB_demand,
    pB_pred,
    pB_util,
    pC_choice,
    pC_placement,
    pC_pred,
    pC_reject,
    pC_role,
    pD_decision,
    pD_margin,
    pD_pred,
    pD_procurement,
    pD_region,
    v2_02_candidate_rows,
    v2_02_packet,
    v2_02_part_a_state,
    v2_02_part_b_state,
    v2_02_part_d_state,
    v2_02_profile,
    v2_02_rejected_row,
    v2_02_select_row,
):
    _part_a = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
    _part_b = v2_02_part_b_state(v2_02_packet, _part_a, pB_util.value, pB_demand.value)
    _rows, _recommended, _rejected_default = v2_02_candidate_rows(v2_02_packet, _part_a, _part_b, pC_placement.value, pC_role.value)
    _selected = v2_02_select_row(_rows, pC_choice.value or _recommended["id"])
    _rejected = v2_02_rejected_row(_rows, pC_reject.value or _rejected_default["id"], _selected["id"])
    _part_d = v2_02_part_d_state(v2_02_packet, _part_a, _part_b, _selected, pD_margin.value, pD_region.value, pD_procurement.value)
    _complete = all(
        value is not None
        for value in (
            pA_pred.value,
            pB_pred.value,
            pC_pred.value,
            pD_pred.value,
            pD_decision.value,
        )
    )
    ledger.save(
        chapter=2,
        design={
            "chapter": "v2_02",
            "track_id": v2_02_profile.track_id,
            "scenario_id": v2_02_packet["scenario_id"],
            "hardware_ref": v2_02_packet["hardware_ref"],
            "model_ref": v2_02_packet["model_ref"],
            "completed": _complete,
            "partA_prediction": pA_pred.value,
            "accelerators_per_enclosure": pA_accels.value,
            "enclosure_count": pA_racks.value,
            "cooling_tier": pA_cooling.value,
            "rack_power_kw": round(_part_a["rack_power_kw"], 4),
            "fleet_power_kw": round(_part_a["fleet_power_kw"], 4),
            "binding_infrastructure_budget": _part_a["binding"],
            "partA_checkpoint": pA_checkpoint.value,
            "partB_prediction": pB_pred.value,
            "utilization_target_pct": pB_util.value,
            "demand_multiplier": pB_demand.value,
            "useful_tflops": round(_part_b["useful_tflops"], 4),
            "idle_cost_usd_hr": round(_part_b["idle_cost"], 4),
            "idle_carbon_kg_hr": round(_part_b["idle_carbon_kg_hr"], 4),
            "utilization_verdict": _part_b["verdict"],
            "partB_checkpoint": pB_checkpoint.value,
            "partC_prediction": pC_pred.value,
            "placement": pC_placement.value,
            "role_emphasis": pC_role.value,
            "selected_mix": _selected["label"],
            "selected_mix_feasible": _selected["feasible"],
            "rejected_alternative": _rejected["label"],
            "rejected_alternative_reason": ", ".join(_rejected["violations"]) or _rejected["note"],
            "partD_prediction": pD_pred.value,
            "capacity_margin_pct": pD_margin.value,
            "carbon_region": pD_region.value,
            "procurement_stance": pD_procurement.value,
            "final_binding_guardrail": _part_d["binding"],
            "final_verdict": _part_d["verdict"],
            "final_decision": pD_decision.value,
            "carry_forward_network_storage": v2_02_packet["carry_forward"],
        },
    )
    _status = "COMPLETE" if _complete else "IN PROGRESS"
    mo.Html(
        f"""
<div class="lab-hud" style="background:#0f172a; border-radius:10px; padding:16px 22px; margin-top:24px;
            font-family:SFMono-Regular, Consolas, monospace;">
  <div style="color:#94a3b8; font-size:0.72rem; font-weight:700; letter-spacing:0.12em;
              text-transform:uppercase;">Design Ledger - Lab V2-02 Saved</div>
  <div style="color:#cbd5e1; font-size:0.84rem; line-height:1.75; margin-top:8px;">
    <span style="color:#64748b;">track:</span> <span style="color:{COLORS['BlueLine']};">{v2_02_profile.label}</span><br/>
    <span style="color:#64748b;">binding_budget:</span> <span style="color:{COLORS['OrangeLine']};">{_part_d['binding']}</span><br/>
    <span style="color:#64748b;">selected_mix:</span> <span style="color:{COLORS['GreenLine']};">{_selected['label']}</span><br/>
    <span style="color:#64748b;">status:</span> <span style="color:{COLORS['RedLine'] if not _complete else COLORS['GreenLine']};">{_status}</span>
  </div>
</div>
"""
    )
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    pA_accels,
    pA_cooling,
    pA_pred,
    pA_racks,
    pB_demand,
    pB_pred,
    pB_util,
    pC_choice,
    pC_placement,
    pC_pred,
    pC_reject,
    pC_role,
    pD_decision,
    pD_margin,
    pD_pred,
    pD_procurement,
    pD_region,
    report_export_panel,
    v2_02_candidate_rows,
    v2_02_metadata,
    v2_02_packet,
    v2_02_part_a_state,
    v2_02_part_b_state,
    v2_02_part_d_state,
    v2_02_profile,
    v2_02_rejected_row,
    v2_02_select_row,
):
    _part_a = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
    _part_b = v2_02_part_b_state(v2_02_packet, _part_a, pB_util.value, pB_demand.value)
    _rows, _recommended, _rejected_default = v2_02_candidate_rows(v2_02_packet, _part_a, _part_b, pC_placement.value, pC_role.value)
    _selected = v2_02_select_row(_rows, pC_choice.value or _recommended["id"])
    _rejected = v2_02_rejected_row(_rows, pC_reject.value or _rejected_default["id"], _selected["id"])
    _part_d = v2_02_part_d_state(v2_02_packet, _part_a, _part_b, _selected, pD_margin.value, pD_region.value, pD_procurement.value)
    _incomplete = []
    if pA_pred.value is None:
        _incomplete.append("Part A prediction")
    if pB_pred.value is None:
        _incomplete.append("Part B prediction")
    if pC_pred.value is None:
        _incomplete.append("Part C prediction")
    if pD_pred.value is None:
        _incomplete.append("Part D prediction")
    if pD_decision.value is None:
        _incomplete.append("Final infrastructure decision")

    _report = build_lab_report(
        v2_02_metadata,
        track=v2_02_profile.label,
        scenario=v2_02_packet["workload_summary"],
        learning_objectives=(
            "Explain why power and cooling can reject an accelerator plan before peak FLOPs do.",
            "Use utilization to connect capacity with cost, waste, and carbon.",
            "Compare accelerator mix and placement using throughput, memory, cost, and carbon evidence.",
            "Write an infrastructure recommendation that satisfies power, utilization, cost, and carbon guardrails.",
        ),
        predictions={
            "partA_binding_budget": pA_pred.value,
            "partB_utilization_rule": pB_pred.value,
            "partC_mix": pC_pred.value,
            "partD_final_guardrail": pD_pred.value,
        },
        knob_settings={
            "accelerators_per_enclosure": pA_accels.value,
            "enclosure_count": pA_racks.value,
            "cooling_tier": pA_cooling.value,
            "utilization_target_pct": pB_util.value,
            "demand_multiplier": pB_demand.value,
            "placement": pC_placement.value,
            "role_emphasis": pC_role.value,
            "capacity_margin_pct": pD_margin.value,
            "carbon_region": pD_region.value,
            "procurement_stance": pD_procurement.value,
        },
        evidence_summary={
            "rack_power_kw": round(_part_a["rack_power_kw"], 4),
            "fleet_power_kw": round(_part_a["fleet_power_kw"], 4),
            "sustained_tflops": round(_part_a["sustained_tflops"], 4),
            "binding_infrastructure_budget": _part_a["binding"],
            "utilization_verdict": _part_b["verdict"],
            "useful_tflops": round(_part_b["useful_tflops"], 4),
            "idle_cost_usd_hr": round(_part_b["idle_cost"], 4),
            "idle_carbon_kg_hr": round(_part_b["idle_carbon_kg_hr"], 4),
            "selected_mix": _selected["label"],
            "selected_mix_feasible": _selected["feasible"],
            "final_binding_guardrail": _part_d["binding"],
            "final_feasible": _part_d["feasible"],
        },
        final_decision={
            "decision": pD_decision.value or _part_d["verdict"],
            "chosen_capacity_plan": (
                f"{_selected['label']} across {_part_a['rack_count']} {v2_02_packet['asset']}(s) "
                f"at {pB_util.value}% utilization with {pD_margin.value}% reserve"
            ),
            "binding_infrastructure_budget": _part_d["binding"] if _part_d["binding"] != "none" else _part_a["binding"],
            "rejected_alternative": _rejected["label"],
            "carry_forward_network_storage": v2_02_packet["carry_forward"],
        },
        big_takeaways=(
            "Compute infrastructure is a coupled amount-system, not a peak-FLOPs purchase.",
            "Utilization controls whether capacity becomes useful work, waste, or saturation.",
            "Accelerator mix and placement must be defended with throughput, memory, cost, carbon, and failure evidence.",
        ),
        reflections={
            "diagnosis": f"The first final guardrail is {_part_d['binding']}.",
            "tradeoff": f"{_selected['label']} rejects {_rejected['label']} because {', '.join(_rejected['violations']) or _rejected['note']}.",
            "residual_risk": "Scenario thresholds are teaching envelopes; production needs measured power, cooling, utilization, and carbon traces.",
        },
        residual_risk=(
            "Notebook-local thresholds model the chapter concepts but are not measured production hardware traces. "
            "Validate the selected plan against facility telemetry, accelerator utilization traces, and region carbon data."
        ),
        source_trace={
            "track_id": v2_02_profile.track_id,
            "scenario_id": v2_02_packet["scenario_id"],
            "hardware_ref": v2_02_packet["hardware_ref"],
            "model_ref": v2_02_packet["model_ref"],
            "source_policy": v2_02_packet["source_policy"],
            "chapter_anchors": (
                "#sec-compute-rack",
                "#sec-compute-infrastructure-peak-vs-sustained-throughput-625a",
                "#sec-compute-accelerator-selection",
                "#sec-compute-summary",
            ),
            "local_solver": "v2_02_* notebook-local amount-system helpers",
        },
        result_snapshot={
            "part_a": _part_a,
            "part_b": _part_b,
            "candidate_rows": _rows,
            "selected": _selected,
            "rejected": _rejected,
            "part_d": _part_d,
        },
        incomplete_fields=tuple(_incomplete),
    )
    mo.vstack(
        [
            mo.md("## Download Report"),
            mo.callout(
                mo.md(
                    "This V2-02 report is generated from the selected track, current controls, "
                    "computed evidence, final recommendation, and residual risk."
                ),
                kind="info",
            ),
            report_export_panel(_report),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
