import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import sys
    import math
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
    import mlsysim
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        get_track_profile,
        render_system_design_lab,
        system_design_context,
        system_design_controls,
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
        get_track_profile,
        go,
        ledger,
        math,
        mlsysim,
        mo,
        render_system_design_lab,
        system_design_context,
        system_design_controls,
        track_context,
        track_selector,
    )


@app.cell
def _():
    chapter = 1
    lab_path = "vol2/lab_01_introduction.py"
    return chapter, lab_path


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_01_track_picker = track_selector(default=_default_track)
    return (v2_01_track_picker,)


@app.cell
def _(lab_path, system_design_context, v2_01_track_picker):
    v2_01_context = system_design_context(lab_path=lab_path, track_id=v2_01_track_picker.value)
    v2_01_profile = v2_01_context.profile
    v2_01_track = v2_01_context.track
    return v2_01_context, v2_01_profile, v2_01_track


@app.cell
def _(math):
    v2_01_TRACK_LENSES = {
        "iphone": {
            "stakeholder": "Mobile release lead",
            "scale_unit": "phones in rollout",
            "single_unit": "one phone",
            "workload": "on-device assistant release",
            "failure_mode": "thermal regressions and privacy-limited telemetry",
            "report_name": "mobile rollout envelope memo",
            "fleet_min": 50_000,
            "fleet_default": 1_000_000,
            "fleet_max": 5_000_000,
            "fleet_step": 50_000,
            "unit_reliability": 0.9999995,
            "health_floor_pct": 90.0,
            "coord_threshold": 30_000_000,
            "metric_prompt": "Which fleet-level amount belongs in the memo first?",
            "budget_label": "usable memory and thermal envelope",
            "memory_budget_gb": 6.0,
            "model_min": 0.25,
            "model_default": 3.0,
            "model_max": 12.0,
            "model_step": 0.25,
            "state_options": {
                "FP16 local weights": 2.0,
                "Quantized local weights": 1.0,
                "Training/adaptation state": 8.0,
            },
            "state_default": "FP16 local weights",
            "unit_capacity": 1.0,
            "demand": 12.0,
            "capacity_unit": "rollout cohorts",
            "part_b_action": "tier the rollout by device class",
            "c3_min": 1_000,
            "c3_default": 100_000,
            "c3_max": 1_000_000,
            "c3_step": 1_000,
            "c3_base_compute_s": 4_000.0,
            "c3_payload_gb": 18.0,
            "c3_network_gbps": 160.0,
            "c3_base_sync_s": 0.20,
            "c3_sync_log_s": 0.035,
            "c3_overlap_pct": 35.0,
            "c3_coord_default_pct": 18,
            "d_min": 50_000,
            "d_default": 1_000_000,
            "d_max": 5_000_000,
            "d_step": 50_000,
            "mtbf_default": 1_500_000,
            "mtbf_min": 200_000,
            "mtbf_max": 5_000_000,
            "mtbf_step": 50_000,
            "recovery_default": 25,
            "routine_failures_per_day": 20.0,
            "failure_event": "support-impacting device issue",
        },
        "oura_ring": {
            "stakeholder": "Wearable firmware lead",
            "scale_unit": "rings in firmware cohort",
            "single_unit": "one ring",
            "workload": "sleep-stage model update",
            "failure_mode": "flash overflow, battery miss, and intermittent sync",
            "report_name": "wearable firmware scale memo",
            "fleet_min": 10_000,
            "fleet_default": 250_000,
            "fleet_max": 1_000_000,
            "fleet_step": 10_000,
            "unit_reliability": 0.999998,
            "health_floor_pct": 92.0,
            "coord_threshold": 8_000_000,
            "metric_prompt": "Which fleet amount should gate the firmware release?",
            "budget_label": "flash/SRAM envelope",
            "memory_budget_gb": 0.016,
            "model_min": 0.001,
            "model_default": 0.012,
            "model_max": 0.08,
            "model_step": 0.001,
            "state_options": {
                "INT8 inference image": 1.0,
                "FP16 debug image": 2.0,
                "Local adaptation state": 6.0,
            },
            "state_default": "INT8 inference image",
            "unit_capacity": 0.25,
            "demand": 2.0,
            "capacity_unit": "nightly cohorts",
            "part_b_action": "simplify the model and stage OTA cohorts",
            "c3_min": 1_000,
            "c3_default": 25_000,
            "c3_max": 500_000,
            "c3_step": 1_000,
            "c3_base_compute_s": 900.0,
            "c3_payload_gb": 2.5,
            "c3_network_gbps": 24.0,
            "c3_base_sync_s": 0.30,
            "c3_sync_log_s": 0.045,
            "c3_overlap_pct": 20.0,
            "c3_coord_default_pct": 24,
            "d_min": 10_000,
            "d_default": 250_000,
            "d_max": 1_000_000,
            "d_step": 10_000,
            "mtbf_default": 800_000,
            "mtbf_min": 100_000,
            "mtbf_max": 3_000_000,
            "mtbf_step": 25_000,
            "recovery_default": 45,
            "routine_failures_per_day": 10.0,
            "failure_event": "battery or firmware incident",
        },
        "robotaxi": {
            "stakeholder": "Safety operations lead",
            "scale_unit": "vehicles on road",
            "single_unit": "one vehicle computer",
            "workload": "perception and map-update release",
            "failure_mode": "tail-latency miss and incident review overload",
            "report_name": "safety operations scale memo",
            "fleet_min": 100,
            "fleet_default": 2_000,
            "fleet_max": 20_000,
            "fleet_step": 100,
            "unit_reliability": 0.9999,
            "health_floor_pct": 95.0,
            "coord_threshold": 220_000,
            "metric_prompt": "Which amount should gate the next geography?",
            "budget_label": "redundant compute and p999 latency envelope",
            "memory_budget_gb": 64.0,
            "model_min": 1.0,
            "model_default": 12.0,
            "model_max": 80.0,
            "model_step": 1.0,
            "state_options": {
                "FP16 perception weights": 2.0,
                "Redundant safety pair": 4.0,
                "Debug/training capture state": 10.0,
            },
            "state_default": "Redundant safety pair",
            "unit_capacity": 2.0,
            "demand": 24.0,
            "capacity_unit": "regional safety slices",
            "part_b_action": "split capacity by geography and redundancy tier",
            "c3_min": 100,
            "c3_default": 2_000,
            "c3_max": 20_000,
            "c3_step": 100,
            "c3_base_compute_s": 600.0,
            "c3_payload_gb": 12.0,
            "c3_network_gbps": 80.0,
            "c3_base_sync_s": 0.10,
            "c3_sync_log_s": 0.050,
            "c3_overlap_pct": 25.0,
            "c3_coord_default_pct": 20,
            "d_min": 100,
            "d_default": 2_000,
            "d_max": 20_000,
            "d_step": 100,
            "mtbf_default": 100_000,
            "mtbf_min": 10_000,
            "mtbf_max": 500_000,
            "mtbf_step": 5_000,
            "recovery_default": 20,
            "routine_failures_per_day": 2.0,
            "failure_event": "vehicle intervention or safety review",
        },
        "cloud_fleet": {
            "stakeholder": "Fleet platform owner",
            "scale_unit": "accelerators in the job",
            "single_unit": "one H100-class accelerator",
            "workload": "frontier training slice",
            "failure_mode": "SLO breach, AllReduce stall, and restart churn",
            "report_name": "fleet operating envelope memo",
            "fleet_min": 64,
            "fleet_default": 8_192,
            "fleet_max": 32_768,
            "fleet_step": 64,
            "unit_reliability": 0.9995,
            "health_floor_pct": 90.0,
            "coord_threshold": 100_000,
            "metric_prompt": "Which fleet amount should the SRE page first?",
            "budget_label": "HBM capacity and useful accelerator time",
            "memory_budget_gb": 80.0,
            "model_min": 7.0,
            "model_default": 175.0,
            "model_max": 400.0,
            "model_step": 1.0,
            "state_options": {
                "FP16 inference weights": 2.0,
                "FP16 gradients": 4.0,
                "Training state with Adam": 12.0,
            },
            "state_default": "Training state with Adam",
            "unit_capacity": 8.0,
            "demand": 128.0,
            "capacity_unit": "job capacity units",
            "part_b_action": "shard model state before adding replicas",
            "c3_min": 64,
            "c3_default": 8_192,
            "c3_max": 16_384,
            "c3_step": 64,
            "c3_base_compute_s": 10_000.0,
            "c3_payload_gb": 350.0,
            "c3_network_gbps": 400.0,
            "c3_base_sync_s": 0.05,
            "c3_sync_log_s": 0.020,
            "c3_overlap_pct": 45.0,
            "c3_coord_default_pct": 16,
            "d_min": 64,
            "d_default": 8_192,
            "d_max": 32_768,
            "d_step": 64,
            "mtbf_default": 50_000,
            "mtbf_min": 5_000,
            "mtbf_max": 200_000,
            "mtbf_step": 5_000,
            "recovery_default": 15,
            "routine_failures_per_day": 1.0,
            "failure_event": "accelerator or node interruption",
        },
    }

    def v2_01_lens_for(track_id):
        return v2_01_TRACK_LENSES.get(track_id, v2_01_TRACK_LENSES["cloud_fleet"])

    def v2_01_fmt_count(value):
        if abs(value) >= 1_000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:,.1f}".rstrip("0").rstrip(".")
        return f"{value:,.3f}".rstrip("0").rstrip(".")

    def v2_01_fmt_pct(value):
        return f"{100 * value:.1f}%"

    def v2_01_table(headers, rows):
        header_html = "".join(
            f"<th style='text-align:left; padding:8px 10px; border-bottom:1px solid #e2e8f0;'>{header}</th>"
            for header in headers
        )
        row_html = ""
        for row in rows:
            row_html += "<tr>"
            for cell in row:
                row_html += f"<td style='padding:8px 10px; border-bottom:1px solid #eef2f7;'>{cell}</td>"
            row_html += "</tr>"
        return (
            "<div style='overflow-x:auto; margin:12px 0; border:1px solid #e2e8f0; "
            "border-radius:8px; background:white;'><table style='border-collapse:collapse; "
            f"width:100%; font-size:0.9rem;'><thead><tr>{header_html}</tr></thead>"
            f"<tbody>{row_html}</tbody></table></div>"
        )

    def v2_01_metric_card(label, value, note, color):
        return f"""
        <div style="flex:1; min-width:170px; padding:14px 16px; border:1px solid #e2e8f0;
                    border-radius:8px; background:white; border-top:3px solid {color};">
            <div style="font-size:0.72rem; font-weight:800; color:#64748b;
                        text-transform:uppercase; letter-spacing:0.06em;">{label}</div>
            <div style="font-size:1.35rem; font-weight:850; color:{color}; margin:5px 0;">{value}</div>
            <div style="font-size:0.78rem; color:#64748b; line-height:1.35;">{note}</div>
        </div>
        """

    def v2_01_part_a_amounts(lens, fleet_size):
        n = max(1.0, float(fleet_size))
        min_n = max(1.0, float(lens["fleet_min"]))
        p_unit = float(lens["unit_reliability"])
        health = p_unit**n
        issue_probability = 1.0 - health
        coord = n * math.log2(max(2.0, n))
        min_coord = min_n * math.log2(max(2.0, min_n))
        capacity_growth = n / min_n
        coord_growth = coord / min_coord
        issue_growth = max(issue_probability, 1e-12) / max(1.0 - (p_unit**min_n), 1e-12)
        normalized = {
            "capacity": capacity_growth,
            "coordination": coord_growth,
            "failure": issue_growth,
        }
        first_order = max(normalized, key=normalized.get)
        boundary = health * 100.0 < lens["health_floor_pct"] or coord > lens["coord_threshold"]
        return {
            "n": n,
            "health": health,
            "issue_probability": issue_probability,
            "coord_index": coord,
            "capacity_index": n,
            "coord_pressure": coord / lens["coord_threshold"],
            "normalized": normalized,
            "first_order": first_order,
            "boundary": boundary,
        }

    def v2_01_part_b_capacity(lens, model_b, state_bytes_per_param):
        model_b = max(float(model_b), 0.000001)
        state_gb = model_b * float(state_bytes_per_param)
        memory_units = max(1, math.ceil(state_gb / lens["memory_budget_gb"]))
        throughput_units = max(1, math.ceil(lens["demand"] / lens["unit_capacity"]))
        required_units = max(memory_units, throughput_units)
        if memory_units >= throughput_units:
            binding = "memory/capacity"
        else:
            binding = "throughput/latency"
        return {
            "state_gb": state_gb,
            "memory_units": memory_units,
            "throughput_units": throughput_units,
            "required_units": required_units,
            "binding": binding,
            "single_unit_feasible": required_units <= 1,
        }

    def v2_01_part_c_fleet_law(lens, fleet_width, comm_reduction_pct, coordination_pct):
        n = max(1.0, float(fleet_width))
        reduction = max(0.0, min(95.0, float(comm_reduction_pct))) / 100.0
        visible_coord = max(0.0, min(95.0, float(coordination_pct))) / 100.0
        compute_s = lens["c3_base_compute_s"] / n
        ring_factor = 0.0 if n <= 1 else 2.0 * (n - 1.0) / n
        raw_comm_s = (lens["c3_payload_gb"] * 8.0 * ring_factor) / lens["c3_network_gbps"]
        comm_s = raw_comm_s * (1.0 - reduction)
        sync_s = lens["c3_base_sync_s"] + lens["c3_sync_log_s"] * math.log2(max(2.0, n)) + compute_s * visible_coord
        overlap_s = min(comm_s * 0.60, compute_s * lens["c3_overlap_pct"] / 100.0)
        step_s = max(0.001, compute_s + comm_s + sync_s - overlap_s)
        compute_fraction = compute_s / step_s
        comm_fraction = comm_s / step_s
        sync_fraction = sync_s / step_s
        goodput = max(0.0, 1.0 - sync_fraction)
        shares = {
            "compute": compute_fraction,
            "communication": comm_fraction,
            "coordination": sync_fraction,
        }
        dominant = max(shares, key=shares.get)
        red_state = comm_fraction > 0.40 or goodput < 0.75
        return {
            "n": n,
            "compute_s": compute_s,
            "comm_s": comm_s,
            "sync_s": sync_s,
            "overlap_s": overlap_s,
            "step_s": step_s,
            "compute_fraction": compute_fraction,
            "comm_fraction": comm_fraction,
            "sync_fraction": sync_fraction,
            "goodput": goodput,
            "dominant": dominant,
            "red_state": red_state,
        }

    def v2_01_part_d_reliability(lens, fleet_size, component_mtbf_hours, recovery_minutes):
        n = max(1.0, float(fleet_size))
        component_mtbf = max(1.0, float(component_mtbf_hours))
        system_mtbf = component_mtbf / n
        failures_per_day = 24.0 / system_mtbf
        lost_hours_per_day = failures_per_day * float(recovery_minutes) / 60.0
        goodput = max(0.0, 1.0 - lost_hours_per_day / 24.0)
        p_failure_shift = 1.0 - math.exp(-8.0 / system_mtbf)
        routine = failures_per_day >= lens["routine_failures_per_day"] or goodput < 0.90
        return {
            "n": n,
            "system_mtbf": system_mtbf,
            "failures_per_day": failures_per_day,
            "lost_hours_per_day": lost_hours_per_day,
            "goodput": goodput,
            "p_failure_shift": p_failure_shift,
            "routine": routine,
        }

    return (
        v2_01_TRACK_LENSES,
        v2_01_fmt_count,
        v2_01_fmt_pct,
        v2_01_lens_for,
        v2_01_metric_card,
        v2_01_part_a_amounts,
        v2_01_part_b_capacity,
        v2_01_part_c_fleet_law,
        v2_01_part_d_reliability,
        v2_01_table,
    )


@app.cell
def _(v2_01_lens_for, v2_01_profile):
    v2_01_lens = v2_01_lens_for(v2_01_profile.track_id)
    return (v2_01_lens,)


@app.cell(hide_code=True)
def _(ACADEMIC_LAB_CSS, COLORS, LAB_CSS, mo, v2_01_lens, v2_01_profile, v2_01_track_picker):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div class="mlsysbook-lab-header" style="--mlsysbook-accent:{COLORS['BlueLine']};">
            <div class="mlsysbook-meta">Machine Learning Systems at Scale - Volume II - Lab 01</div>
            <h1>Scale Changes the Unit</h1>
            <p>
                Work through one shared concept sequence: the fleet is the unit,
                single-node limits force distributed capacity, C3 determines useful
                work, and routine failure sets the operating envelope.
            </p>
            <div class="mlsysbook-chip-row">
                <span class="mlsysbook-chip">Track: {v2_01_profile.label}</span>
                <span class="mlsysbook-chip">Stakeholder: {v2_01_lens['stakeholder']}</span>
                <span class="mlsysbook-chip">Fleet unit: {v2_01_lens['scale_unit']}</span>
            </div>
        </div>
        """),
        mo.callout(mo.md(
            "**Chapter invariant:** scale changes the unit of analysis from a single "
            "machine to a fleet. Coordination, communication, capacity, and failure "
            "cadence become first-order amounts."
        ), kind="info"),
        mo.md("### Track selector"),
        v2_01_track_picker,
        mo.md("""
### Reading Map

- **Part A:** Scale Moment and Machine Learning Fleet.
- **Part B:** Single-node stack to distributed fleet stack.
- **Part C:** C3 taxonomy and the fleet law.
- **Part D:** Reliability gap and routine failure.
- **Synthesis:** Fleet operating envelope for the next compute-infrastructure decision.
"""),
    ])
    return


@app.cell(hide_code=True)
def _(mo, v2_01_lens):
    v2_01_part_a_prediction = mo.ui.radio(
        options={
            "A) Raw capacity grows fastest; the fleet is mostly more units": "capacity",
            "B) Coordination surface grows fastest; the fleet must be managed as one system": "coordination",
            "C) Failure opportunity grows fastest; reliability becomes the first-order amount": "failure",
        },
        label="Part A prediction: which amount changes the scale story first?",
    )
    v2_01_part_a_fleet = mo.ui.slider(
        start=v2_01_lens["fleet_min"],
        stop=v2_01_lens["fleet_max"],
        value=v2_01_lens["fleet_default"],
        step=v2_01_lens["fleet_step"],
        label=f"Fleet size ({v2_01_lens['scale_unit']})",
    )
    v2_01_part_a_checkpoint = mo.ui.radio(
        options={
            "A) Keep reporting the single-unit metric": "single_unit",
            "B) Report fleet health and coordination pressure": "fleet_amounts",
            "C) Defer scale until after model quality improves": "defer",
        },
        label=v2_01_lens["metric_prompt"],
    )
    return v2_01_part_a_checkpoint, v2_01_part_a_fleet, v2_01_part_a_prediction


@app.cell(hide_code=True)
def _(mo, v2_01_lens):
    v2_01_part_b_prediction = mo.ui.radio(
        options={
            "A) One unit can carry it; optimize locally first": "single",
            "B) Memory/capacity fails first; shard the state": "memory",
            "C) Throughput/latency fails first; replicate service capacity": "throughput",
            "D) Operational headroom fails first; reduce scope or specialize": "headroom",
        },
        label="Part B prediction: what breaks before distribution?",
    )
    v2_01_part_b_model_b = mo.ui.slider(
        start=v2_01_lens["model_min"],
        stop=v2_01_lens["model_max"],
        value=v2_01_lens["model_default"],
        step=v2_01_lens["model_step"],
        label="Workload amount (billion-parameter equivalent)",
    )
    v2_01_part_b_state = mo.ui.dropdown(
        options=v2_01_lens["state_options"],
        value=v2_01_lens["state_default"],
        label="State multiplier (bytes per parameter equivalent)",
    )
    v2_01_part_b_choice = mo.ui.radio(
        options={
            "A) Shard model/state across units": "shard",
            "B) Replicate for demand and tail latency": "replicate",
            "C) Specialize/tier the rollout": "specialize",
            "D) Refuse this scale until the workload changes": "refuse",
        },
        label="Checkpoint: what capacity choice survives the single-node limit?",
    )
    return (
        v2_01_part_b_choice,
        v2_01_part_b_model_b,
        v2_01_part_b_prediction,
        v2_01_part_b_state,
    )


@app.cell(hide_code=True)
def _(mo, v2_01_lens):
    v2_01_part_c_prediction = mo.ui.radio(
        options={
            "A) Compute - buy or tune faster local math": "compute",
            "B) Communication - reduce, overlap, or upgrade movement": "communication",
            "C) Coordination - reduce barriers, recovery, and scheduling tax": "coordination",
        },
        label="Part C prediction: which C3 axis will dominate the scaled step?",
    )
    v2_01_part_c_width = mo.ui.slider(
        start=v2_01_lens["c3_min"],
        stop=v2_01_lens["c3_max"],
        value=v2_01_lens["c3_default"],
        step=v2_01_lens["c3_step"],
        label=f"Fleet width ({v2_01_lens['scale_unit']})",
    )
    v2_01_part_c_comm_reduction = mo.ui.slider(
        start=0,
        stop=80,
        value=20,
        step=5,
        label="Communication reduction or overlap (%)",
    )
    v2_01_part_c_coordination = mo.ui.slider(
        start=0,
        stop=50,
        value=v2_01_lens["c3_coord_default_pct"],
        step=2,
        label="Visible coordination overhead (%)",
    )
    v2_01_part_c_mitigation = mo.ui.radio(
        options={
            "A) Faster accelerators": "compute",
            "B) Communication compression/overlap or fabric upgrade": "communication",
            "C) Async checkpointing, elastic recovery, or fewer barriers": "coordination",
            "D) Shrink the fleet width until the envelope is green": "shrink",
        },
        label="Checkpoint: which mitigation targets the measured bottleneck?",
    )
    return (
        v2_01_part_c_comm_reduction,
        v2_01_part_c_coordination,
        v2_01_part_c_mitigation,
        v2_01_part_c_prediction,
        v2_01_part_c_width,
    )


@app.cell(hide_code=True)
def _(mo, v2_01_lens):
    v2_01_part_d_prediction = mo.ui.radio(
        options={
            "A) Rare - manual response is still plausible": "rare",
            "B) Routine - automation is required": "routine",
            "C) Dominant - the workload should be split into smaller failure domains": "dominant",
        },
        label="Part D prediction: what does rare per-unit failure become at fleet scale?",
    )
    v2_01_part_d_fleet = mo.ui.slider(
        start=v2_01_lens["d_min"],
        stop=v2_01_lens["d_max"],
        value=v2_01_lens["d_default"],
        step=v2_01_lens["d_step"],
        label=f"Failure-domain size ({v2_01_lens['scale_unit']})",
    )
    v2_01_part_d_mtbf = mo.ui.slider(
        start=v2_01_lens["mtbf_min"],
        stop=v2_01_lens["mtbf_max"],
        value=v2_01_lens["mtbf_default"],
        step=v2_01_lens["mtbf_step"],
        label="Per-unit MTBF or incident interval (hours)",
    )
    v2_01_part_d_recovery = mo.ui.slider(
        start=1,
        stop=180,
        value=v2_01_lens["recovery_default"],
        step=1,
        label="Recovery time per event (minutes)",
    )
    v2_01_part_d_policy = mo.ui.radio(
        options={
            "A) Manual restart or manual support triage": "manual",
            "B) Scheduled checkpoints and runbook automation": "checkpoint",
            "C) Elastic recovery with smaller blast radius": "elastic",
            "D) Reduce the failure domain before rollout": "reduce_domain",
        },
        label="Checkpoint: what recovery policy matches the measured cadence?",
    )
    return (
        v2_01_part_d_fleet,
        v2_01_part_d_mtbf,
        v2_01_part_d_policy,
        v2_01_part_d_prediction,
        v2_01_part_d_recovery,
    )


@app.cell(hide_code=True)
def _(mo):
    v2_01_synthesis_envelope = mo.ui.radio(
        options={
            "A) Conservative - smaller blast radius and high headroom": "conservative",
            "B) Balanced - accept some C3 tax with explicit mitigations": "balanced",
            "C) Aggressive - maximize scale and rely on recovery machinery": "aggressive",
        },
        label="Synthesis: select the operating envelope for the memo",
    )
    v2_01_synthesis_question = mo.ui.radio(
        options={
            "A) How much HBM capacity is needed per unit?": "hbm",
            "B) What interconnect bandwidth keeps communication below red?": "interconnect",
            "C) What power/cooling envelope sustains the selected fleet?": "power",
            "D) What failure domain size should compute infrastructure expose?": "failure_domain",
        },
        label="Carry-forward question for Compute Infrastructure",
    )
    return v2_01_synthesis_envelope, v2_01_synthesis_question


@app.cell
def _(
    v2_01_lens,
    v2_01_part_a_amounts,
    v2_01_part_a_fleet,
    v2_01_part_b_capacity,
    v2_01_part_b_model_b,
    v2_01_part_b_state,
    v2_01_part_c_comm_reduction,
    v2_01_part_c_coordination,
    v2_01_part_c_fleet_law,
    v2_01_part_c_width,
    v2_01_part_d_fleet,
    v2_01_part_d_mtbf,
    v2_01_part_d_recovery,
    v2_01_part_d_reliability,
):
    v2_01_a = v2_01_part_a_amounts(v2_01_lens, v2_01_part_a_fleet.value)
    v2_01_b = v2_01_part_b_capacity(
        v2_01_lens,
        v2_01_part_b_model_b.value,
        v2_01_part_b_state.value,
    )
    v2_01_c = v2_01_part_c_fleet_law(
        v2_01_lens,
        v2_01_part_c_width.value,
        v2_01_part_c_comm_reduction.value,
        v2_01_part_c_coordination.value,
    )
    v2_01_d = v2_01_part_d_reliability(
        v2_01_lens,
        v2_01_part_d_fleet.value,
        v2_01_part_d_mtbf.value,
        v2_01_part_d_recovery.value,
    )
    return v2_01_a, v2_01_b, v2_01_c, v2_01_d


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    v2_01_a,
    v2_01_b,
    v2_01_c,
    v2_01_d,
    v2_01_fmt_count,
    v2_01_fmt_pct,
    v2_01_lens,
    v2_01_metric_card,
    v2_01_part_a_checkpoint,
    v2_01_part_a_fleet,
    v2_01_part_a_prediction,
    v2_01_part_b_choice,
    v2_01_part_b_model_b,
    v2_01_part_b_prediction,
    v2_01_part_b_state,
    v2_01_part_c_comm_reduction,
    v2_01_part_c_coordination,
    v2_01_part_c_mitigation,
    v2_01_part_c_prediction,
    v2_01_part_c_width,
    v2_01_part_d_fleet,
    v2_01_part_d_mtbf,
    v2_01_part_d_policy,
    v2_01_part_d_prediction,
    v2_01_part_d_recovery,
    v2_01_profile,
    v2_01_synthesis_envelope,
    v2_01_synthesis_question,
    v2_01_table,
):
    def v2_01_section_header(letter, title, concept):
        return mo.Html(f"""
        <div style="margin:24px 0 12px 0;">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="background:{COLORS['BlueLine']}; color:white; border-radius:50%;
                            width:32px; height:32px; display:inline-flex; align-items:center;
                            justify-content:center; font-weight:850;">{letter}</div>
                <div style="flex:1; height:2px; background:{COLORS['Border']};"></div>
                <div style="font-size:0.72rem; font-weight:800; color:{COLORS['TextMuted']};
                            text-transform:uppercase; letter-spacing:0.12em;">Concept Module</div>
            </div>
            <h2 style="margin:10px 0 4px 0;">{title}</h2>
            <div style="color:#475569; line-height:1.55; max-width:850px;">{concept}</div>
        </div>
        """)

    def v2_01_prediction_callout(predicted, actual, labels):
        if predicted is None:
            return mo.callout(mo.md("Commit to a structured prediction before treating the evidence as complete."), kind="warn")
        if predicted == actual:
            return mo.callout(mo.md(f"Prediction check: correct. The measured result is **{labels[actual]}**."), kind="success")
        return mo.callout(mo.md(
            f"Prediction check: you chose **{labels.get(predicted, predicted)}**, but the measured result is "
            f"**{labels[actual]}**. Update the memo around the measured amount."
        ), kind="warn")

    def v2_01_build_part_a():
        labels = {
            "capacity": "raw capacity",
            "coordination": "coordination surface",
            "failure": "failure opportunity",
        }
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Capacity", "Coordination", "Failure opportunity"],
            y=[
                v2_01_a["normalized"]["capacity"],
                v2_01_a["normalized"]["coordination"],
                v2_01_a["normalized"]["failure"],
            ],
            marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["RedLine"]],
            name="growth vs minimum fleet",
        ))
        fig.update_layout(
            height=340,
            yaxis_title="Growth multiple vs smallest shown fleet",
            showlegend=False,
            margin=dict(l=55, r=20, t=24, b=50),
        )
        apply_plotly_theme(fig)
        table = v2_01_table(
            ["Amount", "Current value", "Why it matters"],
            [
                ["Fleet capacity", v2_01_fmt_count(v2_01_a["capacity_index"]), "linear useful capacity"],
                ["Coordination index", v2_01_fmt_count(v2_01_a["coord_index"]), "surface that must be operated"],
                ["Healthy-fleet probability", v2_01_fmt_pct(v2_01_a["health"]), "all units remain healthy in the window"],
                ["At least one issue", v2_01_fmt_pct(v2_01_a["issue_probability"]), "single-unit issue becomes fleet event"],
            ],
        )
        boundary = mo.callout(
            mo.md(
                f"**Boundary reached:** {v2_01_lens['scale_unit']} must be managed as a fleet. "
                f"Health is {v2_01_fmt_pct(v2_01_a['health'])} and coordination pressure is "
                f"{v2_01_a['coord_pressure']:.2f}x the track threshold."
            ),
            kind="danger",
        ) if v2_01_a["boundary"] else mo.callout(
            mo.md("The current point is inside the introductory fleet envelope, but the unit of analysis is still the fleet."),
            kind="success",
        )
        return mo.vstack([
            v2_01_section_header(
                "A",
                "Fleet Is The Unit",
                f"You are the {v2_01_lens['stakeholder']} for a {v2_01_lens['workload']}. "
                f"The question is no longer whether {v2_01_lens['single_unit']} works. "
                f"The question is what happens across {v2_01_lens['scale_unit']}."
            ),
            mo.md("### Prediction"),
            v2_01_part_a_prediction,
            mo.md("### Manipulation"),
            v2_01_part_a_fleet,
            mo.md("### Evidence"),
            mo.as_html(fig),
            mo.Html(table),
            mo.Html(f"""
            <div style="display:flex; gap:12px; flex-wrap:wrap; margin:12px 0;">
                {v2_01_metric_card("Fleet size", v2_01_fmt_count(v2_01_a["n"]), v2_01_lens["scale_unit"], COLORS["BlueLine"])}
                {v2_01_metric_card("First-order amount", labels[v2_01_a["first_order"]], "largest normalized shift", COLORS["OrangeLine"])}
                {v2_01_metric_card("Fleet health", v2_01_fmt_pct(v2_01_a["health"]), f"floor {v2_01_lens['health_floor_pct']:.0f}%", COLORS["RedLine"] if v2_01_a["boundary"] else COLORS["GreenLine"])}
            </div>
            """),
            v2_01_prediction_callout(v2_01_part_a_prediction.value, v2_01_a["first_order"], labels),
            boundary,
            mo.accordion({
                "Math Peek / Source Model": mo.md("""
Capacity alone is linear:

$$ Capacity_{fleet} = N * Capacity_{unit} $$

The coordination surface grows faster in this introductory model:

$$ CoordinationIndex = N * log_2(N) $$

Fleet health compounds the per-unit survival probability:

$$ P(healthy fleet) = p_{unit}^{N} $$

The chapter claim is not that these simple equations are the final simulator.
The claim is that the unit changed: the report must carry fleet amounts.
""")
            }),
            mo.md("### Checkpoint"),
            v2_01_part_a_checkpoint,
        ])

    def v2_01_build_part_b():
        actual = "single" if v2_01_b["single_unit_feasible"] else ("memory" if v2_01_b["binding"] == "memory/capacity" else "throughput")
        labels = {
            "single": "single-unit feasible",
            "memory": "memory/capacity limit",
            "throughput": "throughput/latency limit",
            "headroom": "operational headroom",
        }
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Required state", "Single-unit budget"],
            y=[v2_01_b["state_gb"], v2_01_lens["memory_budget_gb"]],
            marker_color=[COLORS["RedLine"] if v2_01_b["state_gb"] > v2_01_lens["memory_budget_gb"] else COLORS["BlueLine"], COLORS["GreenLine"]],
            name=v2_01_lens["budget_label"],
        ))
        fig.update_layout(
            height=330,
            yaxis_title="GB-equivalent amount",
            showlegend=False,
            margin=dict(l=55, r=20, t=24, b=50),
        )
        apply_plotly_theme(fig)
        table = v2_01_table(
            ["Capacity term", "Amount", "Distributed units"],
            [
                ["State footprint", f"{v2_01_b['state_gb']:.2f} GB", v2_01_fmt_count(v2_01_b["memory_units"])],
                ["Single-unit budget", f"{v2_01_lens['memory_budget_gb']:.3g} GB", "1"],
                ["Demand envelope", f"{v2_01_lens['demand']:.2f} {v2_01_lens['capacity_unit']}", v2_01_fmt_count(v2_01_b["throughput_units"])],
                ["Required distributed units", v2_01_fmt_count(v2_01_b["required_units"]), v2_01_b["binding"]],
            ],
        )
        boundary = mo.callout(
            mo.md(
                f"**Single-unit violation:** {v2_01_lens['single_unit']} cannot carry this amount. "
                f"The smallest distributed envelope is {v2_01_fmt_count(v2_01_b['required_units'])} units, "
                f"binding on **{v2_01_b['binding']}**. A plausible track move is to {v2_01_lens['part_b_action']}."
            ),
            kind="danger",
        ) if not v2_01_b["single_unit_feasible"] else mo.callout(
            mo.md("This specific amount still fits one unit, but the capacity calculation is now explicit."),
            kind="success",
        )
        return mo.vstack([
            v2_01_section_header(
                "B",
                "Single-Node Limits Force Distribution",
                f"The selected track realizes the same concept through {v2_01_lens['budget_label']}. "
                "The activity asks for the first point where local optimization stops being the right unit of work."
            ),
            mo.md("### Prediction"),
            v2_01_part_b_prediction,
            mo.md("### Manipulation"),
            mo.hstack([v2_01_part_b_model_b, v2_01_part_b_state], gap=2),
            mo.md("### Evidence"),
            mo.as_html(fig),
            mo.Html(table),
            mo.Html(f"""
            <div style="display:flex; gap:12px; flex-wrap:wrap; margin:12px 0;">
                {v2_01_metric_card("Workload amount", f"{v2_01_part_b_model_b.value:.3g}B", "billion-parameter equivalent", COLORS["BlueLine"])}
                {v2_01_metric_card("State footprint", f"{v2_01_b['state_gb']:.2f} GB", v2_01_lens["budget_label"], COLORS["RedLine"] if v2_01_b["state_gb"] > v2_01_lens["memory_budget_gb"] else COLORS["GreenLine"])}
                {v2_01_metric_card("Required units", v2_01_fmt_count(v2_01_b["required_units"]), v2_01_b["binding"], COLORS["OrangeLine"])}
            </div>
            """),
            v2_01_prediction_callout(v2_01_part_b_prediction.value, actual, labels),
            boundary,
            mo.accordion({
                "Math Peek / Source Model": mo.md("""
The single-unit question is an amount check, not a preference:

$$ StateGB = ParamsB * bytesPerParam $$

$$ Units_{memory} = ceil(StateGB / UnitBudgetGB) $$

$$ Units_{demand} = ceil(Demand / UnitCapacity) $$

$$ Units_{required} = max(Units_{memory}, Units_{demand}) $$

When required units exceed one, distribution is no longer optional. The
remaining design question is which form of distribution matches the binding
amount.
""")
            }),
            mo.md("### Checkpoint"),
            v2_01_part_b_choice,
        ])

    def v2_01_build_part_c():
        labels = {
            "compute": "Compute",
            "communication": "Communication",
            "coordination": "Coordination",
        }
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Current step"],
            y=[v2_01_c["compute_s"]],
            name="Compute",
            marker_color=COLORS["BlueLine"],
        ))
        fig.add_trace(go.Bar(
            x=["Current step"],
            y=[v2_01_c["comm_s"]],
            name="Communication",
            marker_color=COLORS["OrangeLine"],
        ))
        fig.add_trace(go.Bar(
            x=["Current step"],
            y=[v2_01_c["sync_s"]],
            name="Coordination",
            marker_color=COLORS["RedLine"],
        ))
        fig.add_trace(go.Bar(
            x=["Current step"],
            y=[-v2_01_c["overlap_s"]],
            name="Hidden by overlap",
            marker_color=COLORS["GreenLine"],
        ))
        fig.update_layout(
            barmode="relative",
            height=360,
            yaxis_title="Seconds per step or control cycle",
            margin=dict(l=55, r=20, t=24, b=50),
            legend=dict(orientation="h", y=-0.18),
        )
        apply_plotly_theme(fig)
        table = v2_01_table(
            ["Fleet-law term", "Seconds", "Share"],
            [
                ["Compute", f"{v2_01_c['compute_s']:.3f}", v2_01_fmt_pct(v2_01_c["compute_fraction"])],
                ["Communication", f"{v2_01_c['comm_s']:.3f}", v2_01_fmt_pct(v2_01_c["comm_fraction"])],
                ["Coordination", f"{v2_01_c['sync_s']:.3f}", v2_01_fmt_pct(v2_01_c["sync_fraction"])],
                ["Overlap credit", f"-{v2_01_c['overlap_s']:.3f}", "hidden time"],
                ["Effective step", f"{v2_01_c['step_s']:.3f}", "100%"],
                ["Goodput proxy", v2_01_fmt_pct(v2_01_c["goodput"]), "red below 75%"],
            ],
        )
        boundary = mo.callout(
            mo.md(
                f"**C3 red state:** communication share is {v2_01_fmt_pct(v2_01_c['comm_fraction'])} "
                f"and goodput is {v2_01_fmt_pct(v2_01_c['goodput'])}. "
                "The next lever must target the dominant term, not the most familiar component."
            ),
            kind="danger",
        ) if v2_01_c["red_state"] else mo.callout(
            mo.md("The current C3 envelope is green by the introductory traffic-light thresholds."),
            kind="success",
        )
        return mo.vstack([
            v2_01_section_header(
                "C",
                "C3 Trade-Off At Scale",
                "The fleet law turns the scaled system into an amount budget: local compute, communication, "
                "coordination, and overlap. The track changes what those terms mean, not the concept."
            ),
            mo.md("### Prediction"),
            v2_01_part_c_prediction,
            mo.md("### Manipulation"),
            mo.vstack([
                v2_01_part_c_width,
                mo.hstack([v2_01_part_c_comm_reduction, v2_01_part_c_coordination], gap=2),
            ]),
            mo.md("### Evidence"),
            mo.as_html(fig),
            mo.Html(table),
            mo.Html(f"""
            <div style="display:flex; gap:12px; flex-wrap:wrap; margin:12px 0;">
                {v2_01_metric_card("Dominant C3 axis", labels[v2_01_c["dominant"]], "largest term in the measured step", COLORS["RedLine"] if v2_01_c["red_state"] else COLORS["BlueLine"])}
                {v2_01_metric_card("Compute fraction", v2_01_fmt_pct(v2_01_c["compute_fraction"]), "useful local math share", COLORS["BlueLine"])}
                {v2_01_metric_card("Goodput", v2_01_fmt_pct(v2_01_c["goodput"]), "coordination-loss proxy", COLORS["RedLine"] if v2_01_c["goodput"] < 0.75 else COLORS["GreenLine"])}
            </div>
            """),
            v2_01_prediction_callout(v2_01_part_c_prediction.value, v2_01_c["dominant"], labels),
            boundary,
            mo.accordion({
                "Math Peek / Source Model": mo.md("""
The chapter's fleet law decomposes each distributed step:

$$ T_{step}(N) = T_{compute}/N + T_{comm}(N) + T_{sync}(N) - T_{overlap} $$

The diagnostic rule is to optimize the dominant term:

- high compute fraction -> local math or accelerator efficiency
- high communication fraction -> compression, overlap, topology, or fabric
- low goodput -> checkpointing, elastic recovery, fewer barriers, scheduling

The introductory red thresholds follow the C3 traffic-light pattern:
communication above 40% or goodput below 75% deserves immediate action.
""")
            }),
            mo.md("### Checkpoint"),
            v2_01_part_c_mitigation,
        ])

    def v2_01_build_part_d():
        actual = "routine" if v2_01_d["routine"] else "rare"
        if v2_01_d["failures_per_day"] >= 5 * v2_01_lens["routine_failures_per_day"]:
            actual = "dominant"
        labels = {
            "rare": "rare enough for manual response",
            "routine": "routine enough to require automation",
            "dominant": "dominant enough to shrink failure domains",
        }
        sizes = [
            v2_01_lens["d_min"],
            max(v2_01_lens["d_min"], v2_01_lens["d_default"] // 2),
            v2_01_lens["d_default"],
            min(v2_01_lens["d_max"], max(v2_01_lens["d_default"] * 2, v2_01_lens["d_default"] + v2_01_lens["d_step"])),
            v2_01_lens["d_max"],
        ]
        sizes = sorted(set(int(size) for size in sizes))
        failures = [24.0 / (v2_01_part_d_mtbf.value / max(1, size)) for size in sizes]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sizes,
            y=failures,
            mode="lines+markers",
            name="failures/day",
            line=dict(color=COLORS["RedLine"], width=3),
        ))
        fig.add_hline(
            y=v2_01_lens["routine_failures_per_day"],
            line_dash="dash",
            line_color=COLORS["BlueLine"],
            annotation_text="routine threshold",
        )
        fig.add_trace(go.Scatter(
            x=[v2_01_d["n"]],
            y=[v2_01_d["failures_per_day"]],
            mode="markers",
            marker=dict(size=14, color=COLORS["OrangeLine"], line=dict(color="white", width=2)),
            name="current",
        ))
        fig.update_layout(
            height=350,
            xaxis_title=v2_01_lens["scale_unit"],
            yaxis_title="Expected events per day",
            margin=dict(l=55, r=20, t=24, b=50),
        )
        apply_plotly_theme(fig)
        table = v2_01_table(
            ["Reliability amount", "Current value", "Decision meaning"],
            [
                ["System MTBF", f"{v2_01_d['system_mtbf']:.2f} hours", "time between fleet events"],
                ["Failures per day", f"{v2_01_d['failures_per_day']:.2f}", v2_01_lens["failure_event"]],
                ["Lost time per day", f"{v2_01_d['lost_hours_per_day']:.2f} hours", "recovery load"],
                ["Goodput after recovery", v2_01_fmt_pct(v2_01_d["goodput"]), "useful operating time"],
                ["Failure probability in 8 hours", v2_01_fmt_pct(v2_01_d["p_failure_shift"]), "shift-level exposure"],
            ],
        )
        boundary = mo.callout(
            mo.md(
                f"**Routine-failure boundary:** this fleet expects {v2_01_d['failures_per_day']:.2f} "
                f"{v2_01_lens['failure_event']} events per day. Manual response is no longer an architecture."
            ),
            kind="danger",
        ) if v2_01_d["routine"] else mo.callout(
            mo.md("The selected failure domain remains below the routine-failure threshold."),
            kind="success",
        )
        return mo.vstack([
            v2_01_section_header(
                "D",
                "Rare Events Become Routine",
                f"The failure mode for this track is {v2_01_lens['failure_mode']}. "
                "The activity turns per-unit reliability into fleet failure cadence and recovery load."
            ),
            mo.md("### Prediction"),
            v2_01_part_d_prediction,
            mo.md("### Manipulation"),
            mo.vstack([
                v2_01_part_d_fleet,
                mo.hstack([v2_01_part_d_mtbf, v2_01_part_d_recovery], gap=2),
            ]),
            mo.md("### Evidence"),
            mo.as_html(fig),
            mo.Html(table),
            mo.Html(f"""
            <div style="display:flex; gap:12px; flex-wrap:wrap; margin:12px 0;">
                {v2_01_metric_card("System MTBF", f"{v2_01_d['system_mtbf']:.2f} h", "component MTBF divided by fleet size", COLORS["BlueLine"])}
                {v2_01_metric_card("Events/day", f"{v2_01_d['failures_per_day']:.2f}", v2_01_lens["failure_event"], COLORS["RedLine"] if v2_01_d["routine"] else COLORS["GreenLine"])}
                {v2_01_metric_card("Goodput", v2_01_fmt_pct(v2_01_d["goodput"]), "after recovery load", COLORS["RedLine"] if v2_01_d["goodput"] < 0.90 else COLORS["GreenLine"])}
            </div>
            """),
            v2_01_prediction_callout(v2_01_part_d_prediction.value, actual, labels),
            boundary,
            mo.accordion({
                "Math Peek / Source Model": mo.md("""
For independent components, the fleet-level interruption clock shrinks with N:

$$ MTBF_{system} = MTBF_{component} / N $$

The probability of at least one event during a window t is:

$$ P(event before t) = 1 - exp(-t / MTBF_{system}) $$

This is why the chapter says failure becomes steady state. The design target is
not perfect hardware; it is recovery speed, checkpoint quality, and blast-radius
control.
""")
            }),
            mo.md("### Checkpoint"),
            v2_01_part_d_policy,
        ])

    def build_synthesis():
        envelope = v2_01_synthesis_envelope.value or "balanced"
        question = v2_01_synthesis_question.value or "interconnect"
        question_labels = {
            "hbm": "How much HBM capacity is needed per unit?",
            "interconnect": "What interconnect bandwidth keeps communication below red?",
            "power": "What power/cooling envelope sustains the selected fleet?",
            "failure_domain": "What failure domain size should compute infrastructure expose?",
        }
        return mo.vstack([
            v2_01_section_header(
                "S",
                "Fleet Scale Memo",
                "The synthesis converts the four concept modules into one operating-envelope decision for the next infrastructure chapter."
            ),
            v2_01_synthesis_envelope,
            v2_01_synthesis_question,
            mo.Html(f"""
            <div style="background:#0f172a; color:#e2e8f0; border-radius:10px; padding:20px 24px; margin:16px 0;">
                <div style="font-size:0.72rem; font-weight:800; color:#93c5fd;
                            text-transform:uppercase; letter-spacing:0.12em;">{v2_01_lens['report_name']}</div>
                <h3 style="margin:8px 0 10px 0; color:white;">Selected envelope: {envelope}</h3>
                <p style="line-height:1.6; margin:0 0 12px 0;">
                    For <strong>{v2_01_profile.label}</strong>, the memo treats
                    <strong>{v2_01_lens['scale_unit']}</strong> as the unit of analysis.
                    Part A found <strong>{v2_01_a['first_order']}</strong> as the largest
                    scale shift. Part B requires <strong>{v2_01_fmt_count(v2_01_b['required_units'])}</strong>
                    distributed units, binding on <strong>{v2_01_b['binding']}</strong>.
                    Part C is dominated by <strong>{v2_01_c['dominant']}</strong>
                    with goodput <strong>{v2_01_fmt_pct(v2_01_c['goodput'])}</strong>.
                    Part D estimates <strong>{v2_01_d['failures_per_day']:.2f}</strong>
                    routine events per day.
                </p>
                <div style="border-top:1px solid #334155; padding-top:12px; color:#bfdbfe;">
                    Carry-forward question: {question_labels[question]}
                </div>
            </div>
            """),
            mo.callout(mo.md(
                "A complete fleet memo names the unit, the selected operating envelope, "
                "the binding amount, the C3 bottleneck, the recovery policy, and the next "
                "infrastructure question."
            ), kind="info"),
        ])

    v2_01_tabs = mo.ui.tabs({
        "Part A - Fleet Unit": v2_01_build_part_a(),
        "Part B - Distributed Capacity": v2_01_build_part_b(),
        "Part C - C3 Trade-Off": v2_01_build_part_c(),
        "Part D - Routine Failure": v2_01_build_part_d(),
        "Synthesis": build_synthesis(),
    })
    v2_01_tabs
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    chapter,
    ledger,
    mo,
    v2_01_a,
    v2_01_b,
    v2_01_c,
    v2_01_d,
    v2_01_fmt_count,
    v2_01_fmt_pct,
    v2_01_lens,
    v2_01_part_a_checkpoint,
    v2_01_part_a_fleet,
    v2_01_part_a_prediction,
    v2_01_part_b_choice,
    v2_01_part_b_model_b,
    v2_01_part_b_prediction,
    v2_01_part_b_state,
    v2_01_part_c_comm_reduction,
    v2_01_part_c_coordination,
    v2_01_part_c_mitigation,
    v2_01_part_c_prediction,
    v2_01_part_c_width,
    v2_01_part_d_fleet,
    v2_01_part_d_mtbf,
    v2_01_part_d_policy,
    v2_01_part_d_prediction,
    v2_01_part_d_recovery,
    v2_01_profile,
    v2_01_synthesis_envelope,
    v2_01_synthesis_question,
):
    v2_01_ledger_design = {
        "chapter": "v2_01",
        "track_id": v2_01_profile.track_id,
        "track_label": v2_01_profile.label,
        "scale_unit": v2_01_lens["scale_unit"],
        "partA_prediction": v2_01_part_a_prediction.value,
        "partA_fleet_size": v2_01_part_a_fleet.value,
        "partA_first_order_amount": v2_01_a["first_order"],
        "partA_fleet_health_pct": round(100 * v2_01_a["health"], 3),
        "partA_checkpoint": v2_01_part_a_checkpoint.value,
        "partB_prediction": v2_01_part_b_prediction.value,
        "partB_model_b": v2_01_part_b_model_b.value,
        "partB_state_bytes_per_param": v2_01_part_b_state.value,
        "partB_required_units": v2_01_b["required_units"],
        "partB_binding_limit": v2_01_b["binding"],
        "partB_capacity_choice": v2_01_part_b_choice.value,
        "partC_prediction": v2_01_part_c_prediction.value,
        "partC_fleet_width": v2_01_part_c_width.value,
        "partC_comm_reduction_pct": v2_01_part_c_comm_reduction.value,
        "partC_coordination_pct": v2_01_part_c_coordination.value,
        "partC_dominant_axis": v2_01_c["dominant"],
        "partC_goodput_pct": round(100 * v2_01_c["goodput"], 3),
        "partC_mitigation": v2_01_part_c_mitigation.value,
        "partD_prediction": v2_01_part_d_prediction.value,
        "partD_failure_domain": v2_01_part_d_fleet.value,
        "partD_component_mtbf_hours": v2_01_part_d_mtbf.value,
        "partD_recovery_minutes": v2_01_part_d_recovery.value,
        "partD_system_mtbf_hours": round(v2_01_d["system_mtbf"], 4),
        "partD_failures_per_day": round(v2_01_d["failures_per_day"], 4),
        "partD_recovery_policy": v2_01_part_d_policy.value,
        "operating_envelope": v2_01_synthesis_envelope.value,
        "carry_forward_question": v2_01_synthesis_question.value,
    }
    ledger.save(track=v2_01_profile.track_id, chapter=chapter, design=v2_01_ledger_design)

    mo.Html(f"""
    <div class="lab-hud" style="background:#0f172a; border-radius:10px; padding:18px 24px;
                margin-top:26px; font-family:'SF Mono', 'Fira Code', monospace;">
        <div style="color:#94a3b8; font-size:0.7rem; font-weight:800;
                    text-transform:uppercase; letter-spacing:0.14em; margin-bottom:10px;">
            Design Ledger - Lab V2-01 Saved
        </div>
        <div style="color:#cbd5e1; font-size:0.82rem; line-height:1.8;">
            <span style="color:#64748b;">track:</span>
            <span style="color:{COLORS['BlueLine']};">{v2_01_profile.label}</span><br/>
            <span style="color:#64748b;">fleet unit:</span> {v2_01_lens['scale_unit']}<br/>
            <span style="color:#64748b;">Part A amount:</span> {v2_01_a['first_order']}
            at {v2_01_fmt_count(v2_01_part_a_fleet.value)} units<br/>
            <span style="color:#64748b;">Part B required units:</span>
            {v2_01_fmt_count(v2_01_b['required_units'])} ({v2_01_b['binding']})<br/>
            <span style="color:#64748b;">Part C axis:</span> {v2_01_c['dominant']}
            with goodput {v2_01_fmt_pct(v2_01_c['goodput'])}<br/>
            <span style="color:#64748b;">Part D cadence:</span>
            {v2_01_d['failures_per_day']:.2f} events/day
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
