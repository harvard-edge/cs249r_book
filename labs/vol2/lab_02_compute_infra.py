import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 02: The Compute Wall · MLSysBook")


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

    import html
    import plotly.graph_objects as go
    import mlsysim
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
        instrumentation_console,
        ledger,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_arc_context,
        track_context,
    )


@app.cell
def _(get_lab_metadata):
    v2_02_metadata = get_lab_metadata("vol2/lab_02_compute_infra.py")
    return (v2_02_metadata,)


@app.cell(hide_code=True)
def _(mo):
    v2_02_track_picker = mo.ui.dropdown(
        options={
            "⚡ TinyML Track (Microcontrollers & Wearables / Oura Ring & Cortex-M55)": "oura_ring",
            "📱 Mobile Track (On-Device Personal AI / iPhone & Apple Silicon M4)": "iphone",
            "🤖 Edge & Embodied Track (Robotics & Drones / Robotaxi & Jetson Orin)": "robotaxi",
            "☁️ Cloud Supercomputing Track (NVIDIA H100 / B200 Clusters & 3D Parallelism / Scale)": "cloud_fleet",
        },
        value="☁️ Cloud Supercomputing Track (NVIDIA H100 / B200 Clusters & 3D Parallelism / Scale)",
        label="Select Course / Industry Track",
    )
    v2_02_track_picker
    return (v2_02_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v2_02_metadata,
    v2_02_track_picker,
):
    v2_02_track_id = v2_02_track_picker.value
    v2_02_profile = get_track_profile(v2_02_track_id)
    v2_02_variant = get_lab_track_variant(v2_02_metadata.lab_id, v2_02_profile.track_id)
    v2_02_hardware = resolve_mlsysim_ref(v2_02_variant.hardware_ref)
    v2_02_model = resolve_mlsysim_ref(v2_02_variant.model_ref)
    # Cross-tier hardware targets: Hardware.Tiny.CortexM55, Hardware.Mobile.AppleM4, Hardware.Edge.JetsonOrin, Hardware.Cloud.H100
    return v2_02_profile, v2_02_variant


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
                "<div class='mlsysbook-field' style='margin-bottom:6px;'>"
                f"<strong style='color:#1E293B;'>{v2_02_escape(key)}:</strong> <span style='color:#475569;'>{v2_02_escape(value)}</span>"
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
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    track_arc_context,
    track_context,
    v2_02_metadata,
    v2_02_packet,
    v2_02_profile,
):
    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell" style="margin-bottom: 20px;">
        <div style="border-left: 4px solid #A51C30; padding: 12px 18px; background: white; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #A51C30; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
                ML Systems Textbook &middot; Volume II &middot; Chapter 2 &middot; Foundational Lab 02
            </div>
            <h1 style="margin: 0 0 6px 0; color: #0F172A; font-size: 1.85rem; font-weight: 800;">
                Compute Infrastructure: The Compute Wall
            </h1>
            <p style="margin: 0 0 12px 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">
                Datacenter and edge compute are physically constrained infrastructure. Power delivery, heat dissipation, accelerator mix,
                placement, utilization, cost, and carbon form tightly coupled budgets. Peak FLOPs matter only after physical constraints sustain the plan.
            </p>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="mlsysbook-chip" style="background: #FEE2E2; color: #991B1B; font-weight: 700;">Track: {v2_02_profile.label}</span>
                <span class="mlsysbook-chip" style="background: #E0F2FE; color: #0369A1;">Stakeholder: {v2_02_packet['stakeholder']}</span>
                <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155;">Hardware: {v2_02_packet['hardware_ref']}</span>
                <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155;">Model: {v2_02_packet['model_ref']}</span>
                <span class="mlsysbook-chip" style="background: #FEF3C7; color: #92400E;">Focus: Power Walls &amp; Rooflines</span>
                <span class="mlsysbook-chip" style="background: #EDE9FE; color: #5B21B6;">Deliverable: Compute Sizing Memo</span>
            </div>
        </div>
    </div>
    """)

    scenario_panel = mo.Html(f"""
    <div class="mlsysbook-panel" style="margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #0F172A; font-size: 1.1rem;">System Scenario: {v2_02_profile.label} Compute Sizing</h3>
        <p style="color: #334155; font-size: 0.9rem; line-height: 1.6;">
            {v2_02_packet['workload_summary']} You are operating as the <strong>{v2_02_packet['stakeholder']}</strong>
            evaluating compute deployments for <strong>{v2_02_packet['asset']}</strong> infrastructure. Sizing must balance thermal envelopes,
            power density, arithmetic intensity, and Model FLOPs Utilization (MFU) against hard operational guardrails.
        </p>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-top: 12px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;">The Architectural Invariants of Compute Infrastructure:</div>
            <ul class="mlsysbook-list" style="margin: 0; padding-left: 1.25rem; font-size: 0.85rem; color: #334155; line-height: 1.6;">
                <li><strong>The Thermal &amp; Power Density Limit:</strong> Compute throughput is physically bounded by heat dissipation and site delivery: <i>P</i><sub>total</sub> = <i>N</i> &middot; <i>P</i><sub>accel</sub> + <i>P</i><sub>facility</sub>. Racks melt before algorithms finish if cooling capacity is exceeded.</li>
                <li><strong>The Roofline Model:</strong> Kernel performance is bounded by the min of peak compute and memory bandwidth: GFLOPS = min(&pi;, &beta; &middot; <i>I</i>). Scaling compute cores yields zero speedup when arithmetic intensity falls below machine balance.</li>
                <li><strong>The MFU Invariant:</strong> Model FLOPs Utilization (MFU) measures true efficiency: MFU = Observed TFLOPS / Theoretical Peak TFLOPS. Amortizing cluster CapEx requires keeping sustained MFU high across varying batch and sequence regimes.</li>
                <li><strong>The Guardrail Envelope:</strong> Cost, thermals, and power cannot be treated as downstream afterthoughts; they are hard mathematical constraints in cluster sizing.</li>
            </ul>
        </div>
    </div>
    """)

    objectives_panel = mo.Html(f"""
    <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">LEARNING OBJECTIVES</div>
        <ul class="mlsysbook-list" style="margin: 0; padding-left: 1.25rem; font-size: 0.88rem; color: #334155; line-height: 1.6;">
            <li><strong>Analyze physical rack, power, and cooling limits:</strong> Determine whether thermal dissipation or site delivery rejects an accelerator topology before FLOPs do.</li>
            <li><strong>Map utilization to economics and carbon:</strong> Connect target MFU with idle cost waste, operational saturation, and grid carbon emissions.</li>
            <li><strong>Optimize heterogeneous mix and placement:</strong> Compare homogeneous vs. tier-specialized fleets across throughput, memory margins, and placement penalties.</li>
            <li><strong>Defend an authorized infrastructure recommendation:</strong> Satisfy multi-constraint guardrails and synthesize a durable hand-off memo for downstream networking and storage.</li>
        </ul>
        <div style="margin-top: 10px; font-size: 0.84rem; color: #0284C7; font-style: italic;">
            CORE QUESTION: "How do physical thermal and power boundaries constrain compute choices, and how do we co-design accelerators for high sustained MFU?"
        </div>
    </div>
    """)

    track_mission_card = track_context(v2_02_profile)
    arc_card = track_arc_context(v2_02_profile, v2_02_metadata.lab_id)

    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        header_html,
        scenario_panel,
        objectives_panel,
        track_mission_card,
        arc_card,
    ])
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
    return pA_accels, pA_checkpoint, pA_cooling, pA_pred, pA_racks


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
    return pB_checkpoint, pB_demand, pB_pred, pB_util


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
    return pC_choice, pC_placement, pC_pred, pC_reject, pC_role


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
    return pD_decision, pD_margin, pD_pred, pD_procurement, pD_region


@app.cell(hide_code=True)
def _(
    COLORS,
    MathPeek,
    apply_plotly_theme,
    big_takeaways,
    gated_hypothesis_card,
    go,
    instrumentation_console,
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
    source_trace,
    v2_02_candidate_rows,
    v2_02_fields_html,
    v2_02_markdown_table,
    v2_02_metric_cards_html,
    v2_02_packet,
    v2_02_part_a_state,
    v2_02_part_b_state,
    v2_02_part_d_state,
    v2_02_profile,
    v2_02_rejected_row,
    v2_02_select_row,
    v2_02_status_badge,
):
    def build_part_a():
        state = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
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

        rows = [
            ("Cooling", f"{state['rack_power_kw']:.3f} kW/rack", f"{state['cooling_limit_kw']:.3f} kW/rack", "pass" if state["cooling_ok"] else "fail"),
            ("Site power", f"{state['fleet_power_kw']:.3f} kW", f"{v2_02_packet['site_power_kw']:.3f} kW", "pass" if state["site_ok"] else "fail"),
            ("Sustained capacity", f"{state['sustained_tflops']:.1f}", f"{state['demand_tflops']:.1f}", "pass" if state["capacity_ok"] else "fail"),
        ]

        pred_check = (
            mo.callout(mo.md("Commit to a structured prediction before finalizing the memo."), kind="warn")
            if pA_pred.value is None
            else (
                mo.callout(mo.md("**Prediction check:** your prior matched the binding infrastructure budget."), kind="success")
                if (pA_pred.value == state["binding"] or (pA_pred.value == "power" and state["binding"] == "site power"))
                else mo.callout(mo.md(f"**Prediction check:** the instrument found `{state['binding']}`, not `{pA_pred.value}`. Peak capacity is not the first question when the physical envelope fails."), kind="warn")
            )
        )

        boundary_callout = (
            mo.callout(mo.md(f"**Recovered envelope.** The current plan satisfies rack cooling, site power, and sustained capacity with `{state['binding']}` as the binding budget."), kind="success")
            if state["feasible"]
            else mo.callout(mo.md(f"**Infrastructure violation:** `{state['binding']}` rejects the plan. Change cooling tier, reduce density, or adjust rack count until the physical budget recovers."), kind="danger")
        )

        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <div class="mlsysbook-section-label">Part A - Concept Module</div>
              <h2 style="margin:8px 0 6px 0; color:#0F172A;">Rack, Power, and Cooling Bind Before Peak FLOPs</h2>
              <p style="color:#475569; font-size:0.92rem; line-height:1.55;">
                {v2_02_packet['stakeholder']} needs a feasible {v2_02_packet['report_frame']}.
                The first question is not how many peak FLOPs the plan advertises; it is whether
                the {v2_02_packet['asset']} can receive power and reject heat.
              </p>
              <div class="mlsysbook-compact-fields">
                {v2_02_fields_html({
                    "Chapter claim": "Selecting the fastest accelerator is counterproductive if cooling cannot remove the heat.",
                    "Your decision": "Choose the binding infrastructure budget to carry forward.",
                    "Track consequence": v2_02_packet["failure_noun"] + " failure changes what enough compute means.",
                })}
              </div>
            </div>
            """),
            gated_hypothesis_card(
                pA_pred,
                title="1. Formulate Physical Envelope Hypothesis",
                subtitle=f"Predict which infrastructure budget rejects the first {v2_02_packet['report_frame']}:",
                gate_label="Hypothesis Gate A",
            ),
            instrumentation_console(
                mo.hstack([pA_accels, pA_racks, pA_cooling], gap="1rem"),
                title="Physical Rack & Power Envelope Controls",
                subtitle=f"Adjust accelerator density per {v2_02_packet['asset']}, rack count, and cooling tier:",
            ),
            mo.md("### Empirical Evidence"),
            mo.as_html(fig),
            mo.Html(
                v2_02_metric_cards_html([
                    ("Accelerators", f"{state['total_accelerators']}", f"{pA_racks.value} x {pA_accels.value}", COLORS["BlueLine"]),
                    ("Rack power", f"{state['rack_power_kw']:.2f} kW", f"limit {state['cooling_limit_kw']:.2f} kW", COLORS["OrangeLine"]),
                    ("Sustained capacity", f"{state['sustained_tflops']:.0f}", "TFLOP/s-equivalent", COLORS["GreenLine"]),
                    ("Binding budget", state["binding"], "after power/cooling/capacity checks", COLORS["RedLine"] if not state["feasible"] else COLORS["GreenLine"]),
                ])
            ),
            mo.md(v2_02_markdown_table(("Budget", "Current", "Limit / Need", "Status"), rows)),
            boundary_callout,
            pred_check,
            MathPeek(
                formula="P_{\\text{rack}} = P_{\\text{overhead}} + N_{\\text{accel}} \\cdot P_{\\text{accel}}; \\quad P_{\\text{fleet}} = P_{\\text{rack}} \\cdot N_{\\text{rack}}",
                variables={
                    "AcceleratorsPerRack": f"{pA_accels.value}",
                    "RackPower": f"{state['rack_power_kw']:.2f} kW",
                    "CoolingLimit": f"{state['cooling_limit_kw']:.2f} kW",
                    "FleetPower": f"{state['fleet_power_kw']:.2f} kW",
                    "SiteLimit": f"{v2_02_packet['site_power_kw']:.2f} kW",
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid {COLORS['BlueLine']}; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT A: BINDING INFRASTRUCTURE BUDGET</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Physical Budget Carry-Forward</h4>
                {pA_checkpoint}
            </div>
            """),
        ])

    def build_part_b():
        part_a = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
        state = v2_02_part_b_state(v2_02_packet, part_a, pB_util.value, pB_demand.value)

        xs = list(range(15, 99, 5))
        useful = []
        idle_cost = []
        for pct in xs:
            row = v2_02_part_b_state(v2_02_packet, part_a, pct, pB_demand.value)
            useful.append(row["useful_tflops"])
            idle_cost.append(row["idle_cost"])

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

        rows = [
            ("Utilization target", f"{state['utilization_pct']:.0f}%", f"{v2_02_packet['min_util_pct']}-{v2_02_packet['max_util_pct']}%", state["verdict"]),
            ("Useful capacity", f"{state['useful_tflops']:.1f}", f"{state['adjusted_demand_tflops']:.1f}", "pass" if state["demand_ok"] else "shortfall"),
            ("Hourly cost", f"${state['hourly_cost']:.2f}", f"${v2_02_packet['hourly_cost_guardrail']:.2f}", "context"),
            ("Idle carbon", f"{state['idle_carbon_kg_hr']:.2f} kg/hr", "lower is better", "waste" if state["waste"] else "bounded"),
        ]

        verdict_callout = (
            mo.callout(mo.md("**Utilization target is defensible.** The plan turns capacity into useful work without erasing operating headroom."), kind="success")
            if state["verdict"] == "balanced"
            else (
                mo.callout(mo.md("**Waste boundary reached.** Bought accelerators are powered and cooled while too little useful work is extracted."), kind="danger")
                if state["verdict"] == "waste"
                else mo.callout(mo.md("**Saturation boundary reached.** The plan may look efficient, but queue/capacity headroom is gone."), kind="danger")
            )
        )

        pred_check = (
            mo.callout(mo.md("Commit to a utilization prediction to unlock the economics instrument."), kind="warn")
            if pB_pred.value is None
            else (
                mo.callout(mo.md("**Prediction check:** correct. Utilization is a guarded band, not a one-way maximize knob."), kind="success")
                if (pB_pred.value == "band" and state["verdict"] == "balanced")
                else mo.callout(mo.md("**Prediction check:** the evidence rejects one-sided utilization rules. Economics, carbon, and queue headroom move together."), kind="warn")
            )
        )

        return mo.vstack([
            mo.Html("""
            <div class="mlsysbook-panel">
              <div class="mlsysbook-section-label">Part B - Concept Module</div>
              <h2 style="margin:8px 0 6px 0; color:#0F172A;">Utilization Converts Capacity Into Economics And Waste</h2>
              <p style="color:#475569; font-size:0.92rem; line-height:1.55;">
                The same physical capacity from Part A can be wasteful, healthy, or saturated.
                Utilization is the conversion factor between bought infrastructure and useful work.
              </p>
            </div>
            """),
            gated_hypothesis_card(
                pB_pred,
                title="2. Formulate Utilization & MFU Hypothesis",
                subtitle="Predict how accelerator utilization should be governed in operational capacity planning:",
                gate_label="Hypothesis Gate B",
            ),
            instrumentation_console(
                mo.hstack([pB_util, pB_demand], gap="1rem"),
                title="Utilization & Workload Demand Controls",
                subtitle="Sweep target utilization percentage and demand multiplier:",
            ),
            mo.md("### Empirical Evidence"),
            mo.as_html(fig),
            mo.Html(
                v2_02_metric_cards_html([
                    ("Useful throughput", f"{state['useful_tflops']:.0f}", f"demand {state['adjusted_demand_tflops']:.0f}", COLORS["BlueLine"]),
                    ("Idle cost", f"${state['idle_cost']:.2f}/hr", "capacity paid for but unused", COLORS["OrangeLine"]),
                    ("Carbon waste", f"{state['idle_carbon_kg_hr']:.2f} kg/hr", "idle share of facility emissions", COLORS["RedLine"]),
                    ("Verdict", state["verdict"], f"healthy band {v2_02_packet['min_util_pct']}-{v2_02_packet['max_util_pct']}%", COLORS["GreenLine"] if state["verdict"] == "balanced" else COLORS["RedLine"]),
                ])
            ),
            mo.md(v2_02_markdown_table(("Amount", "Current", "Reference", "Status"), rows)),
            verdict_callout,
            pred_check,
            MathPeek(
                formula="\\text{UsefulTFLOPS} = \\text{SustainedTFLOPS} \\cdot U; \\quad \\text{IdleCost} = \\text{Cost}_{\\text{hr}} \\cdot (1 - U)",
                variables={
                    "TargetUtilization": f"{pB_util.value}%",
                    "UsefulTFLOPS": f"{state['useful_tflops']:.1f}",
                    "IdleCost": f"${state['idle_cost']:.2f}/hr",
                    "IdleCarbon": f"{state['idle_carbon_kg_hr']:.2f} kg/hr",
                    "Verdict": state["verdict"],
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid {COLORS['BlueLine']}; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT B: UTILIZATION DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Operational Target Confirmation</h4>
                {pB_checkpoint}
            </div>
            """),
        ])

    def build_part_c():
        part_a = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
        part_b = v2_02_part_b_state(v2_02_packet, part_a, pB_util.value, pB_demand.value)
        rows, recommended, rejected_default = v2_02_candidate_rows(v2_02_packet, part_a, part_b, pC_placement.value, pC_role.value)
        selected = v2_02_select_row(rows, pC_choice.value)
        rejected = v2_02_rejected_row(rows, pC_reject.value, selected["id"])

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

        table_rows = [
            (
                row["label"],
                f"{row['throughput_tflops']:.0f}",
                f"{row['memory_margin_pct']:.1f}%",
                f"${row['cost_usd_hr']:.2f}",
                f"{row['carbon_kg_hr']:.2f}",
                "pass" if row["feasible"] else ", ".join(row["violations"]),
            )
            for row in rows
        ]

        pred_check = (
            mo.callout(mo.md("Choose a mix prediction to unlock the placement comparison."), kind="warn")
            if pC_pred.value is None
            else (
                mo.callout(mo.md("**Prediction check:** your predicted mix matches the current recommended candidate."), kind="success")
                if pC_pred.value == recommended["id"]
                else mo.callout(mo.md(f"**Prediction check:** current evidence favors `{recommended['label']}` under this placement and workload role."), kind="warn")
            )
        )

        selected_callout = (
            mo.callout(mo.md(f"**Selected mix passes Part C.** It still needs Part D's simultaneous power, utilization, cost, and carbon guardrails."), kind="success")
            if selected["feasible"]
            else mo.callout(mo.md(f"**Selected mix fails:** {', '.join(selected['violations'])}. A plan can be locally appealing and still fail the amount-system guardrails."), kind="danger")
        )

        return mo.vstack([
            mo.Html("""
            <div class="mlsysbook-panel">
              <div class="mlsysbook-section-label">Part C - Concept Module</div>
              <h2 style="margin:8px 0 6px 0; color:#0F172A;">Accelerator Mix And Placement Change The Plan</h2>
              <p style="color:#475569; font-size:0.92rem; line-height:1.55;">
                A fleet is not just a larger accelerator. Training, inference, memory bandwidth,
                placement, cost, and carbon can prefer different capacity mixes.
              </p>
            </div>
            """),
            gated_hypothesis_card(
                pC_pred,
                title="3. Formulate Fleet Heterogeneity Hypothesis",
                subtitle="Predict which accelerator mix and placement topology survives operational guardrails:",
                gate_label="Hypothesis Gate C",
            ),
            instrumentation_console(
                mo.hstack([pC_placement, pC_role], gap="1rem"),
                title="Heterogeneous Mix & Placement Controls",
                subtitle="Configure placement topology and workload role emphasis:",
            ),
            mo.md("### Empirical Evidence"),
            mo.as_html(fig),
            mo.md(v2_02_markdown_table(("Candidate", "Throughput", "Memory", "Cost/hr", "Carbon/hr", "Status"), table_rows)),
            mo.Html(
                v2_02_metric_cards_html([
                    ("Recommended", recommended["label"], recommended["note"], COLORS["GreenLine"] if recommended["feasible"] else COLORS["OrangeLine"]),
                    ("Selected", selected["label"], "student checkpoint or recommended default", COLORS["BlueLine"]),
                    ("Rejected", rejected["label"], ", ".join(rejected["violations"]) or rejected["note"], COLORS["RedLine"]),
                ])
            ),
            selected_callout,
            pred_check,
            MathPeek(
                formula="\\text{Score} = 40 \\cdot \\frac{\\text{TFLOPS}}{\\text{Demand}} + 0.9 \\cdot \\text{MemMargin} - 18 \\cdot \\frac{\\text{Cost}}{\\text{Limit}} - 10 \\cdot \\frac{\\text{Carbon}}{\\text{Limit}} - \\text{Penalty}",
                variables={
                    "Placement": pC_placement.value,
                    "RoleEmphasis": pC_role.value,
                    "SelectedMix": selected["label"],
                    "SelectedFeasible": str(selected["feasible"]),
                    "Recommended": recommended["label"],
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid {COLORS['BlueLine']}; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT C: SELECTION &amp; REJECTION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Document Capacity Mix Decision and Justify Rejection</h4>
                {mo.hstack([pC_choice, pC_reject], gap="1rem")}
            </div>
            """),
        ])

    def build_part_d():
        part_a = v2_02_part_a_state(v2_02_packet, pA_accels.value, pA_racks.value, pA_cooling.value)
        part_b = v2_02_part_b_state(v2_02_packet, part_a, pB_util.value, pB_demand.value)
        rows, recommended, _ = v2_02_candidate_rows(v2_02_packet, part_a, part_b, pC_placement.value, pC_role.value)
        selected = v2_02_select_row(rows, pC_choice.value or recommended["id"])
        state = v2_02_part_d_state(v2_02_packet, part_a, part_b, selected, pD_margin.value, pD_region.value, pD_procurement.value)

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

        status_html = " ".join(v2_02_status_badge(ok, label) for label, _, _, ok in guardrail_rows)

        table_rows = [
            ("Power", f"{state['effective_power_kw']:.2f} kW", f"{v2_02_packet['site_power_kw']:.2f} kW", "pass" if state["power_ok"] else "fail"),
            ("Cooling", f"{state['effective_rack_kw'] / max(part_a['rack_count'], 1):.2f} kW/enclosure", f"{part_a['cooling_limit_kw']:.2f}", "pass" if state["cooling_ok"] else "fail"),
            ("Utilization", f"{state['utilization_after_margin_pct']:.1f}%", f"{v2_02_packet['min_util_pct']}-{v2_02_packet['max_util_pct']}%", "pass" if state["utilization_ok"] else "fail"),
            ("Cost", f"${state['cost_usd_hr']:.2f}/hr", f"${v2_02_packet['hourly_cost_guardrail']:.2f}/hr", "pass" if state["cost_ok"] else "fail"),
            ("Carbon", f"{state['carbon_kg_hr']:.2f} kg/hr", f"{v2_02_packet['carbon_guardrail_kg_hr']:.2f} kg/hr", "pass" if state["carbon_ok"] else "fail"),
        ]

        pred_check = (
            mo.callout(mo.md("Predict the rejecting guardrail to unlock the recommendation scorecard."), kind="warn")
            if pD_pred.value is None
            else (
                mo.callout(mo.md("**Prediction check:** your final guardrail prediction matches the scorecard."), kind="success")
                if (pD_pred.value == state["binding"] or (pD_pred.value == "power" and state["binding"] in {"power", "cooling"}))
                else (
                    mo.callout(mo.md("**Prediction check:** no guardrail failed under the current controls. The memo can approve if the residual risk is named."), kind="info")
                    if state["binding"] == "none"
                    else mo.callout(mo.md(f"**Prediction check:** `{state['binding']}` rejected the plan first, not `{pD_pred.value}`."), kind="warn")
                )
            )
        )

        guardrail_callout = (
            mo.callout(mo.md("**Recommendation passes.** The selected capacity plan satisfies all simultaneous guardrails."), kind="success")
            if state["feasible"]
            else mo.callout(mo.md(f"**Recommendation not ready:** `{state['binding']}` is the first failed guardrail. The memo should revise or reject the plan."), kind="danger")
        )

        return mo.vstack([
            mo.Html("""
            <div class="mlsysbook-panel">
              <div class="mlsysbook-section-label">Part D - Concept Module</div>
              <h2 style="margin:8px 0 6px 0; color:#0F172A;">Recommendation Under Simultaneous Guardrails</h2>
              <p style="color:#475569; font-size:0.92rem; line-height:1.55;">
                The winning mix from Part C is not launch-ready until power, cooling,
                utilization, cost, and carbon all pass together.
              </p>
            </div>
            """),
            gated_hypothesis_card(
                pD_pred,
                title="4. Formulate Multi-Constraint Guardrail Hypothesis",
                subtitle="Predict which physical, economic, or environmental guardrail rejects the final sizing recommendation:",
                gate_label="Hypothesis Gate D",
            ),
            instrumentation_console(
                mo.hstack([pD_margin, pD_region, pD_procurement], gap="1rem"),
                title="Operational Guardrail & Procurement Controls",
                subtitle="Set capacity reserve margin, carbon intensity grid region, and procurement posture:",
            ),
            mo.md("### Empirical Evidence"),
            mo.as_html(fig),
            mo.Html(f"<div style='display:flex; flex-wrap:wrap; gap:8px; margin:12px 0;'>{status_html}</div>"),
            mo.md(v2_02_markdown_table(("Guardrail", "Current", "Limit", "Status"), table_rows)),
            guardrail_callout,
            pred_check,
            MathPeek(
                formula="\\text{Feasible} = \\text{PowerOK} \\land \\text{CoolingOK} \\land \\text{UtilOK} \\land \\text{CostOK} \\land \\text{CarbonOK}",
                variables={
                    "ReserveMargin": f"{pD_margin.value}%",
                    "Region": pD_region.value,
                    "Procurement": pD_procurement.value,
                    "EffectivePower": f"{state['effective_power_kw']:.2f} kW",
                    "HourlyCost": f"${state['cost_usd_hr']:.2f}/hr",
                    "BindingGuardrail": state["binding"],
                },
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid {COLORS['BlueLine']}; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT D: FINAL ARCHITECTURAL RECOMMENDATION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Authorize, Revise, or Reject the Compute Sizing Memo</h4>
                {pD_decision}
            </div>
            """),
        ])

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
        passed = part_d["feasible"]

        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <div class="mlsysbook-section-label">Synthesis - Compute Infrastructure Memo</div>
              <h2 style="margin:8px 0 6px 0; color:#0F172A;">{verdict_text}</h2>
              <p style="color:#475569; font-size:0.92rem; line-height:1.55;">
                The memo must state the selected capacity plan, the binding infrastructure
                budget, a rejected alternative, and the network/storage implication that
                carries into the next labs.
              </p>
              <div class="mlsysbook-compact-fields">{v2_02_fields_html(dict(memo_rows))}</div>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #A51C30; margin: 18px 0; background: #FFFDFD;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #A51C30; text-transform: uppercase; margin-bottom: 6px;">LEAD ARCHITECT AUTHORIZATION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Compute Infrastructure Sign-Off: {v2_02_packet['stakeholder']}</h4>
                <div style="display: flex; gap: 12px; align-items: center; margin-top: 8px;">
                    <span style="display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 800; font-size: 0.8rem; background: {'#ECFDF5' if passed else '#FEF2F2'}; color: {'#065F46' if passed else '#991B1B'}; border: 1px solid {'#A7F3D0' if passed else '#FECACA'};">
                        {'APPROVED FOR PROCUREMENT' if passed else 'REVISION REQUIRED &mdash; GUARDRAIL BREACH'}
                    </span>
                    <span style="font-size: 0.85rem; color: #475569;">
                        Verdict: <code>{verdict_text}</code> &middot; Binding: <code>{part_d['binding']}</code> &middot; Fleet Power: <code>{part_d['effective_power_kw']:.2f} kW</code>
                    </span>
                </div>
            </div>
            """),
            mo.Html(f"""
            <div style="background:#0f172a; color:#e2e8f0; border-radius:10px; padding:20px 24px; margin:16px 0;">
                <div style="font-size:0.72rem; font-weight:800; color:#93c5fd;
                            text-transform:uppercase; letter-spacing:0.12em;">Compute Infrastructure Sizing Memo</div>
                <h3 style="margin:8px 0 10px 0; color:white;">Track: {v2_02_profile.label}</h3>
                <p style="line-height:1.6; margin:0 0 12px 0;">
                    Selected <strong>{selected['label']}</strong> deployed across <strong>{part_a['rack_count']} {v2_02_packet['asset']}(s)</strong>
                    ({part_a['total_accelerators']} {v2_02_packet['accelerator_unit']}(s)).
                    Binding budget: <strong>{part_d['binding'] if part_d['binding'] != 'none' else part_a['binding']}</strong>.
                    Operating at <strong>{pB_util.value}%</strong> utilization with <strong>{pD_margin.value}%</strong> reserve margin.
                    Fleet power is <strong>{part_d['effective_power_kw']:.2f} kW</strong> and hourly TCO is <strong>${part_d['cost_usd_hr']:.2f}/hr</strong>.
                </p>
                <div style="border-top:1px solid #334155; padding-top:12px; color:#bfdbfe;">
                    <strong>Downstream Carry-Forward:</strong> {v2_02_packet['carry_forward']}
                </div>
            </div>
            """),
            big_takeaways([
                "Compute infrastructure is a coupled amount-system, not a peak-FLOPs purchase.",
                "Selecting the fastest accelerator is counterproductive if rack cooling cannot remove the heat.",
                "Model FLOPs Utilization (MFU) is a guarded operating band: over-allocation wastes capital, while saturation destroys latency.",
                "Heterogeneous placement must balance arithmetic intensity against interconnect bandwidth and offload latency.",
                "Cost, power delivery, and carbon intensity form simultaneous non-negotiable operational guardrails.",
            ]),
            source_trace(
                {
                    "Report builder": "mlsysbook_labs.build_lab_report",
                    "Report export": "mlsysbook_labs.report_export_panel",
                    "Ledger": "DesignLedger.save(chapter=2)",
                    "Hardware ref": v2_02_packet["hardware_ref"],
                    "Model ref": v2_02_packet["model_ref"],
                },
                collapsed=True,
                summary="The sizing memo and ledger snapshot are generated from local controls and source-traced helpers.",
            ),
            mo.Html("""
            <div style="border: 1px solid #CBD5E1; border-radius: 8px; padding: 16px 20px; margin-top: 20px; background: #F8FAFC;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
                    What's Next &middot; Volume II Curriculum Continuum
                </div>
                <h4 style="margin: 0 0 6px 0; color: #0F172A; font-size: 1.05rem;">
                    Next Lab: Volume II, Chapter 3 &mdash; Interconnect &amp; Communication Topologies
                </h4>
                <p style="margin: 0; font-size: 0.88rem; color: #334155; line-height: 1.5;">
                    Carry your compute infrastructure sizing into Chapter 3, where you will wire these compute nodes into high-bandwidth
                    Clos, torus, and dragonfly fabrics to sustain all-reduce collective communication.
                </p>
            </div>
            """),
        ])

    v2_02_tabs = mo.ui.tabs({
        "Part A -- Rack, Power & Cooling": build_part_a(),
        "Part B -- Compute Utilization & MFU": build_part_b(),
        "Part C -- Heterogeneous Placement": build_part_c(),
        "Part D -- Cost & Operational Guardrails": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    v2_02_tabs
    return


@app.cell(hide_code=True)
def _(
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
    ledger.save(chapter=2, design={
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
    })

    _passed = _part_d["feasible"]
    mo.Html(
        f"""
    <div class="lab-hud">
      <span class="hud-label">LAB</span>
      <span class="hud-value">Vol2 &middot; Lab 02</span>
      <span class="hud-label">TRACK</span>
      <span class="hud-value">{v2_02_profile.label}</span>
      <span class="hud-label">STATUS</span>
      <span class="hud-value" style="color: {'#10B981' if _passed else '#F59E0B'};">{'PASS' if _passed else 'REVIEW'}</span>
      <span class="hud-label">BINDING</span>
      <span class="hud-value">{_part_d['binding']}</span>
      <span class="hud-label">FLEET POWER</span>
      <span class="hud-value">{_part_d['effective_power_kw']:.2f} kW</span>
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
