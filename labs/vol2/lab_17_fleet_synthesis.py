import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 17: The Fleet Synthesis · MLSysBook")


@app.cell
async def _():
    import html as html_lib
    import math
    import sys
    from pathlib import Path

    import marimo as mo

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
        lab_header,
        learning_objectives,
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
    # MLSysBook reference tiers: Hardware.Tiny.CortexM55, Hardware.Mobile.AppleM4, Hardware.Edge.JetsonOrin, Hardware.Cloud.H100
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
        html_lib,
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
    v2_17_lab_path = "vol2/lab_17_fleet_synthesis.py"
    v2_17_chapter = 17
    v2_17_metadata = get_lab_metadata(v2_17_lab_path)
    return v2_17_chapter, v2_17_metadata


@app.cell(hide_code=True)
def _(mo):
    v2_17_track_picker = mo.ui.dropdown(
        options={
            "⚡ TinyML Track (ARM Cortex-M55 / ESP32-S3 & Wearable Firmware / Battery Limits)": "oura_ring",
            "📱 Mobile Track (Apple Silicon / Snapdragon & On-Device Thermal / Privacy Limits)": "iphone",
            "🤖 Edge & Embodied Track (NVIDIA Jetson AGX Orin & Safety-Critical Autonomy Limits)": "robotaxi",
            "☁️ Cloud Supercomputing Track (NVIDIA H100 / B200 Clusters & 3D Parallelism / Scale)": "cloud_fleet",
        },
        value="☁️ Cloud Supercomputing Track (NVIDIA H100 / B200 Clusters & 3D Parallelism / Scale)",
        label="Select Course / Industry Track",
    )
    v2_17_track_picker
    return (v2_17_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v2_17_metadata,
    v2_17_track_picker,
):
    v2_17_track_id = v2_17_track_picker.value
    v2_17_profile = get_track_profile(v2_17_track_id)
    v2_17_variant = get_lab_track_variant(v2_17_metadata.lab_id, v2_17_profile.track_id)
    v2_17_hardware = resolve_mlsysim_ref(v2_17_variant.hardware_ref)
    v2_17_model = resolve_mlsysim_ref(v2_17_variant.model_ref)
    return v2_17_hardware, v2_17_model, v2_17_profile, v2_17_variant


@app.cell
def _(html_lib, math):
    def v2_17_escape(value):
        return html_lib.escape(str(value))

    def v2_17_quantity_to_float(value, unit, default=0.0):
        if value is None:
            return default
        if hasattr(value, "m_as"):
            try:
                return float(value.m_as(unit))
            except Exception:
                return default
        if hasattr(value, "to"):
            try:
                return float(value.to(unit).magnitude)
            except Exception:
                return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def v2_17_axis_label(axis):
        labels = {
            "capacity": "Capacity",
            "communication": "Communication",
            "storage": "Storage/Data",
            "coordination": "Coordination",
            "reliability": "Reliability",
            "serving": "Serving/Performance",
            "edge": "Edge/Placement",
            "security_privacy": "Security/Privacy",
            "robustness": "Robustness",
            "carbon": "Carbon",
            "governance": "Governance",
        }
        return labels.get(axis, str(axis).replace("_", " ").title())

    def v2_17_axis_options():
        axes = (
            "capacity",
            "communication",
            "reliability",
            "security_privacy",
            "robustness",
            "carbon",
            "governance",
        )
        return {v2_17_axis_label(axis): axis for axis in axes}

    def v2_17_track_lens(track_id):
        lenses = {
            "iphone": {
                "report_frame": "Mobile ML fleet release review",
                "architecture_goal": "ship a privacy-preserving local model across device and OS cohorts",
                "natural_failure": "thermal or battery regression reaches a device cohort before privacy-safe evidence is sufficient",
                "evidence_emphasis": "battery, thermal headroom, local p95/p99 latency, privacy-safe telemetry, staged rollout",
                "close": "The iPhone fleet can ship only inside the device support, thermal, privacy, and rollout evidence envelope.",
                "components": (
                    "device support matrix",
                    "local runtime and fallback path",
                    "privacy-safe telemetry",
                    "staged app rollout",
                    "model update channel",
                    "support escalation loop",
                ),
                "axis_pressure": {
                    "capacity": 66.0,
                    "communication": 44.0,
                    "storage": 52.0,
                    "coordination": 58.0,
                    "reliability": 62.0,
                    "serving": 72.0,
                    "edge": 84.0,
                    "security_privacy": 86.0,
                    "robustness": 68.0,
                    "carbon": 50.0,
                    "governance": 74.0,
                },
                "risk_options": (
                    "device cohort silently falls back to a hotter path",
                    "privacy-safe telemetry under-samples affected users",
                    "battery regression appears after broad rollout",
                    "OS or hardware heterogeneity invalidates the measured envelope",
                ),
            },
            "oura_ring": {
                "report_frame": "Wearable firmware fleet review",
                "architecture_goal": "ship sensing and inference updates without breaking duty cycle, OTA, or health-adjacent trust",
                "natural_failure": "firmware and model payload pass nominal tests but exceed SRAM, flash, duty-cycle, or false-alert limits",
                "evidence_emphasis": "SRAM, flash, battery life, sensing quality, OTA payload, privacy",
                "close": "The wearable fleet can ship only when inference, sensing, OTA, and privacy budgets pass together.",
                "components": (
                    "sensor and MCU runtime",
                    "firmware plus model package",
                    "duty-cycle controller",
                    "phone/cloud sync",
                    "OTA update path",
                    "health-adjacent review loop",
                ),
                "axis_pressure": {
                    "capacity": 76.0,
                    "communication": 52.0,
                    "storage": 78.0,
                    "coordination": 64.0,
                    "reliability": 76.0,
                    "serving": 62.0,
                    "edge": 88.0,
                    "security_privacy": 90.0,
                    "robustness": 72.0,
                    "carbon": 46.0,
                    "governance": 82.0,
                },
                "risk_options": (
                    "SRAM or flash headroom disappears under a firmware update",
                    "battery duty cycle regresses during overnight sensing",
                    "health-adjacent false alerts increase for a cohort",
                    "OTA rollback evidence is incomplete",
                ),
            },
            "robotaxi": {
                "report_frame": "Safety-critical fleet deployment case",
                "architecture_goal": "ship a perception and fallback architecture with rare-event evidence and accountable operations",
                "natural_failure": "a tail-latency, fallback, or rare-event recall miss reaches live fleet operation",
                "evidence_emphasis": "p99/p999 latency, safety margin, rare-event replay, fallback reliability, accountable operations",
                "close": "The RoboTaxi fleet can ship only when the safety case, fallback path, and operations evidence all pass.",
                "components": (
                    "sensor stack",
                    "local accelerator path",
                    "fallback controller",
                    "rare-event replay system",
                    "safety operations gate",
                    "fleet learning review",
                ),
                "axis_pressure": {
                    "capacity": 78.0,
                    "communication": 64.0,
                    "storage": 68.0,
                    "coordination": 76.0,
                    "reliability": 92.0,
                    "serving": 94.0,
                    "edge": 90.0,
                    "security_privacy": 78.0,
                    "robustness": 96.0,
                    "carbon": 64.0,
                    "governance": 90.0,
                },
                "risk_options": (
                    "rare-event recall evidence is too thin for the rollout geography",
                    "fallback latency violates the safety margin",
                    "weather or road-user shift invalidates validation coverage",
                    "incident accountability chain is incomplete",
                ),
            },
            "cloud_fleet": {
                "report_frame": "Distributed ML fleet architecture review",
                "architecture_goal": "run training and serving under SLA, cost, capacity, carbon, security, and governance constraints",
                "natural_failure": "SLA, communication, failure recovery, carbon, or audit pressure binds after scale-up",
                "evidence_emphasis": "throughput, p99 latency, utilization, cost/request, carbon, security, governance evidence",
                "close": "The cloud fleet can ship only when capacity, fabric, storage, serving, recovery, carbon, and governance pass together.",
                "components": (
                    "accelerator pools",
                    "high-bisection fabric",
                    "checkpoint and object storage",
                    "training and serving scheduler",
                    "observability control loop",
                    "security, carbon, and governance gates",
                ),
                "axis_pressure": {
                    "capacity": 88.0,
                    "communication": 92.0,
                    "storage": 82.0,
                    "coordination": 86.0,
                    "reliability": 86.0,
                    "serving": 86.0,
                    "edge": 54.0,
                    "security_privacy": 82.0,
                    "robustness": 76.0,
                    "carbon": 86.0,
                    "governance": 80.0,
                },
                "risk_options": (
                    "communication bottleneck erases capacity gains",
                    "failure recovery burns the service or training window",
                    "carbon-aware scheduling conflicts with SLA headroom",
                    "audit or privacy evidence arrives after launch pressure",
                ),
            },
        }
        return lenses[track_id]

    def v2_17_volume_layer_specs(profile, lens):
        label = profile.label
        return (
            {
                "chapter": 2,
                "lab_id": "v2_02_compute_wall",
                "layer": "Compute infrastructure",
                "concept": "physical capacity, power, cooling, and accelerator fit",
                "axis": "capacity",
                "weight": 8.0,
                "preset": f"{label}: size the compute envelope around {profile.primary_metrics[0]} and headroom",
                "constraint": profile.dominant_constraints[0],
                "realization": lens["components"][0],
            },
            {
                "chapter": 3,
                "lab_id": "v2_03_network_fabric_design",
                "layer": "Network fabric",
                "concept": "bandwidth, latency, topology, and congestion",
                "axis": "communication",
                "weight": 8.0,
                "preset": f"{label}: preserve fabric margin for synchronization and telemetry paths",
                "constraint": "network bandwidth and tail latency",
                "realization": lens["components"][1],
            },
            {
                "chapter": 4,
                "lab_id": "v2_04_data_pipeline_wall",
                "layer": "Data and storage",
                "concept": "storage throughput, locality, consistency, and lifecycle",
                "axis": "storage",
                "weight": 7.0,
                "preset": f"{label}: keep data movement and retention matched to the operating path",
                "constraint": "data movement and retention",
                "realization": lens["components"][2],
            },
            {
                "chapter": 5,
                "lab_id": "v2_05_parallelism_design",
                "layer": "Distributed training",
                "concept": "parallelism trades compute for communication and coordination",
                "axis": "coordination",
                "weight": 7.0,
                "preset": f"{label}: choose the parallelism or update plan that fits the fleet envelope",
                "constraint": "parallel efficiency and memory",
                "realization": "training, update, or validation flow",
            },
            {
                "chapter": 6,
                "lab_id": "v2_06_collective_communication",
                "layer": "Collective communication",
                "concept": "collectives are algorithms with topology costs",
                "axis": "communication",
                "weight": 8.0,
                "preset": f"{label}: expose synchronization and aggregation cost instead of hiding it in throughput",
                "constraint": "collective payload and topology",
                "realization": "synchronization or aggregation path",
            },
            {
                "chapter": 7,
                "lab_id": "v2_07_failure_budget_engineering",
                "layer": "Fault tolerance",
                "concept": "failure is routine at scale",
                "axis": "reliability",
                "weight": 9.0,
                "preset": f"{label}: design recovery before failures become incidents",
                "constraint": "MTBF, checkpoint, rollback, and lost work",
                "realization": "recovery and rollback path",
            },
            {
                "chapter": 8,
                "lab_id": "v2_08_fleet_orchestration",
                "layer": "Fleet orchestration",
                "concept": "schedulers allocate scarce resources under priorities",
                "axis": "coordination",
                "weight": 8.0,
                "preset": f"{label}: reserve scheduler headroom for the highest-risk workload slice",
                "constraint": "queueing, fairness, and placement",
                "realization": "scheduler and rollout control",
            },
            {
                "chapter": 9,
                "lab_id": "v2_09_optimization_trap",
                "layer": "Performance engineering",
                "concept": "optimization is measurement-driven and can displace cost",
                "axis": "serving",
                "weight": 7.0,
                "preset": f"{label}: reject local speedups that hide downstream debt",
                "constraint": "measured bottleneck and regression risk",
                "realization": "performance review loop",
            },
            {
                "chapter": 10,
                "lab_id": "v2_10_inference_economy",
                "layer": "Inference",
                "concept": "serving couples queues, state, cache, latency, and cost",
                "axis": "serving",
                "weight": 9.0,
                "preset": f"{label}: keep serving policy inside latency, quality, and cost guardrails",
                "constraint": profile.guardrail_metrics[0],
                "realization": "serving and fallback path",
            },
            {
                "chapter": 11,
                "lab_id": "v2_11_edge_thermodynamics",
                "layer": "Edge intelligence",
                "concept": "placement moves constraints between device, edge, and cloud",
                "axis": "edge",
                "weight": 7.0,
                "preset": f"{label}: place work where the track envelope can actually sustain it",
                "constraint": "placement, connectivity, and local limits",
                "realization": "device, edge, or central placement",
            },
            {
                "chapter": 12,
                "lab_id": "v2_12_silent_fleet",
                "layer": "Operations at scale",
                "concept": "operations are control loops over error budgets and blast radius",
                "axis": "reliability",
                "weight": 8.0,
                "preset": f"{label}: connect rollout, telemetry, incident response, and rollback",
                "constraint": "error budget, blast radius, and response time",
                "realization": "observability and incident loop",
            },
            {
                "chapter": 13,
                "lab_id": "v2_13_price_of_privacy",
                "layer": "Security and privacy",
                "concept": "adversaries and privacy controls change the architecture",
                "axis": "security_privacy",
                "weight": 9.0,
                "preset": f"{label}: preserve privacy and security evidence without blinding operations",
                "constraint": "attack surface, access, and privacy budget",
                "realization": "security and privacy controls",
            },
            {
                "chapter": 14,
                "lab_id": "v2_14_robustness_budget",
                "layer": "Robust AI",
                "concept": "robustness is distributional and operational",
                "axis": "robustness",
                "weight": 9.0,
                "preset": f"{label}: validate the failure slice that matters most for the track",
                "constraint": "drift, uncertainty, and fallback",
                "realization": "robustness validation loop",
            },
            {
                "chapter": 15,
                "lab_id": "v2_15_carbon_budget",
                "layer": "Sustainable AI",
                "concept": "energy and carbon are first-order constraints",
                "axis": "carbon",
                "weight": 8.0,
                "preset": f"{label}: treat carbon and power as launch guardrails, not after-the-fact reports",
                "constraint": "energy, carbon, and placement",
                "realization": "carbon and power gate",
            },
            {
                "chapter": 16,
                "lab_id": "v2_16_fairness_budget",
                "layer": "Responsible AI",
                "concept": "responsibility constrains design",
                "axis": "governance",
                "weight": 9.0,
                "preset": f"{label}: make fairness, accountability, and audit evidence part of the operating path",
                "constraint": "fairness, accountability, and auditability",
                "realization": "governance review path",
            },
        )

    def v2_17_track_packet(profile, variant, hardware, model):
        lens = v2_17_track_lens(profile.track_id)
        memory = getattr(hardware, "memory", None)
        storage = getattr(hardware, "storage", None)
        hardware_facts = {
            "hardware_name": getattr(hardware, "name", variant.hardware_ref),
            "model_name": getattr(model, "name", variant.model_ref),
            "memory_gb": round(v2_17_quantity_to_float(getattr(memory, "capacity", None), "GB", 0.0), 2),
            "storage_tb": round(v2_17_quantity_to_float(getattr(storage, "capacity", None), "TB", 0.0), 2),
            "tdp_w": round(v2_17_quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0), 2),
            "battery_wh": round(v2_17_quantity_to_float(getattr(hardware, "battery_capacity", None), "Wh", 0.0), 2),
            "model_params_m": round(v2_17_quantity_to_float(getattr(model, "parameters", None), "param", 0.0) / 1_000_000.0, 3),
            "model_flops_g": round(v2_17_quantity_to_float(getattr(model, "inference_flops", None), "flop", 0.0) / 1_000_000_000.0, 3),
        }
        plan_factors = {
            "local_baseline": {
                "communication_factor": 0.96,
                "recovery_score": 0.70,
                "security_score": 0.72,
                "robustness_score": 0.70,
                "carbon_factor": 0.78,
                "governance_score": 0.72,
                "evidence_score": 0.64,
            },
            "balanced_policy": {
                "communication_factor": 0.82,
                "recovery_score": 0.90,
                "security_score": 0.90,
                "robustness_score": 0.88,
                "carbon_factor": 0.80,
                "governance_score": 0.88,
                "evidence_score": 0.86,
            },
            "scale_first": {
                "communication_factor": 1.15,
                "recovery_score": 0.82,
                "security_score": 0.78,
                "robustness_score": 0.83,
                "carbon_factor": 1.08,
                "governance_score": 0.78,
                "evidence_score": 0.72,
            },
        }
        plan_options = {}
        for key, option in dict(variant.defaults["decision_options"]).items():
            plan_options[key] = {
                "option_id": key,
                "label": str(option["label"]),
                "emphasis": str(option["emphasis"]),
                "capacity": float(option["capacity"]),
                "base_load": float(option["base_load"]),
                "base_latency_ms": float(option["base_latency_ms"]),
                "base_cost": float(option["base_cost"]),
                "quality_pct": float(option["quality_pct"]),
                "guardrail_pct": float(option["guardrail_pct"]),
                "mitigation": str(option["mitigation"]),
                "validation_requirement": str(option["validation_requirement"]),
                "residual_risk": str(option["residual_risk"]),
                **plan_factors[key],
            }
        validation_packages = {
            "stress_replay": {
                "label": "Capacity, communication, and failure stress replay",
                "covers": ("capacity", "communication", "reliability"),
            },
            "privacy_security_audit": {
                "label": "Privacy/security and governance audit",
                "covers": ("security_privacy", "governance"),
            },
            "robustness_redteam": {
                "label": "Robustness and fallback red-team replay",
                "covers": ("robustness", "security_privacy", "reliability"),
            },
            "carbon_governance_review": {
                "label": "Carbon, cost, and governance review",
                "covers": ("carbon", "governance", "capacity"),
            },
            "full_board_packet": {
                "label": "Full deployment review packet",
                "covers": ("all",),
            },
        }
        return {
            "track_id": profile.track_id,
            "label": profile.label,
            "stakeholder": variant.stakeholder,
            "scenario": variant.workload_summary,
            "hardware_ref": variant.hardware_ref,
            "model_ref": variant.model_ref,
            "primary_metric": variant.primary_metric,
            "guardrail_metric": variant.guardrail_metric,
            "source_policy": profile.source_policy,
            "report_artifact": variant.assumptions.get("report_artifact", "final fleet architecture report"),
            "validation_tests": tuple(variant.defaults.get("validation_tests", ())),
            "lens": lens,
            "hardware_facts": hardware_facts,
            "layer_specs": v2_17_volume_layer_specs(profile, lens),
            "axis_pressure": dict(lens["axis_pressure"]),
            "plan_options": plan_options,
            "validation_packages": validation_packages,
            "revision_actions": {
                "relax_noncritical": "Relax a noncritical target and record the risk",
                "monitor_gate": "Monitor and gate the risky slice",
                "defer_scope": "Defer scope until the guardrail has margin",
                "redesign_architecture": "Redesign the architecture around the binding guardrail",
            },
            "board_outcomes": {
                "APPROVE": "Approve with named validation and residual risk",
                "PROVISIONAL": "Provisional approval; validation or margin is incomplete",
                "REJECT": "Reject until the board packet is repaired",
            },
        }

    def v2_17_history_entry(history, spec):
        if not isinstance(history, dict):
            return None
        for key in (spec["chapter"], str(spec["chapter"])):
            entry = history.get(key)
            if isinstance(entry, dict):
                return entry
        chapter_tag = f"v2_{int(spec['chapter']):02d}"
        for entry in history.values():
            if not isinstance(entry, dict):
                continue
            entry_lab = str(entry.get("lab_id") or entry.get("lab") or entry.get("chapter") or "")
            if entry_lab == spec["lab_id"] or entry_lab.startswith(chapter_tag):
                return entry
        return None

    def v2_17_entry_summary(entry, spec):
        if not isinstance(entry, dict):
            return spec["preset"], spec["constraint"]
        decision_keys = (
            "decision",
            "final_decision",
            "selected_plan",
            "selected_policy",
            "selected_rollout_incident_policy",
            "selected_option",
            "selected_strategy",
            "selected_placement_policy",
            "selected_lifecycle_policy",
            "binding_constraint",
            "dominant_risk",
        )
        decision = ""
        for key in decision_keys:
            value = entry.get(key)
            if value not in (None, "", False):
                decision = str(value)
                break
        if not decision and entry.get("completed") is True:
            decision = "completed ledger entry"
        if not decision:
            decision = spec["preset"]
        constraint = (
            entry.get("binding_constraint")
            or entry.get("dominant_risk")
            or entry.get("binding_guardrail")
            or entry.get("constraint")
            or spec["constraint"]
        )
        return decision, str(constraint)

    def v2_17_ledger_replay(packet, history, evidence_floor_pct, preset_weight_pct):
        rows = []
        found = 0
        axis_evidence = {axis: 0.0 for axis in packet["axis_pressure"]}
        for spec in packet["layer_specs"]:
            entry = v2_17_history_entry(history, spec)
            decision, constraint = v2_17_entry_summary(entry, spec)
            source = "track preset"
            confidence = float(preset_weight_pct)
            if isinstance(entry, dict):
                entry_track = entry.get("track_id")
                if entry_track and entry_track != packet["track_id"]:
                    source = "other-track ledger"
                    confidence = max(45.0, float(preset_weight_pct))
                else:
                    source = "student ledger"
                    confidence = 100.0
                    found += 1
            axis = spec["axis"]
            axis_evidence[axis] = axis_evidence.get(axis, 0.0) + spec["weight"] * (confidence / 100.0)
            rows.append(
                {
                    "chapter": spec["chapter"],
                    "layer": spec["layer"],
                    "concept": spec["concept"],
                    "amount": v2_17_axis_label(axis),
                    "track_realization": spec["realization"],
                    "constraint": constraint,
                    "source": source,
                    "confidence_pct": round(confidence, 1),
                    "decision": decision,
                }
            )
        expected = len(packet["layer_specs"])
        coverage = found / expected * 100.0 if expected else 0.0
        evidence_gap = max(0.0, float(evidence_floor_pct) - coverage)
        score_rows = []
        for axis, pressure in packet["axis_pressure"].items():
            penalty = evidence_gap * 0.30 if axis in ("governance", "reliability") else 0.0
            score = min(100.0, float(pressure) + axis_evidence.get(axis, 0.0) + penalty)
            score_rows.append(
                {
                    "axis": axis,
                    "amount": v2_17_axis_label(axis),
                    "track_pressure": round(float(pressure), 1),
                    "evidence_points": round(axis_evidence.get(axis, 0.0), 2),
                    "evidence_debt_penalty": round(penalty, 2),
                    "score_pct": round(score, 2),
                    "status": "BINDING" if score >= 90.0 else "WATCH" if score >= 75.0 else "MARGIN",
                }
            )
        binding = max(score_rows, key=lambda row: row["score_pct"])
        return {
            "rows": rows,
            "score_rows": score_rows,
            "entries_found": found,
            "entries_expected": expected,
            "coverage_pct": round(coverage, 2),
            "evidence_gap_pct": round(evidence_gap, 2),
            "binding": binding,
            "architecture_summary": (
                f"{packet['label']} replay uses {found}/{expected} matching ledger entries; "
                f"missing entries use lower-confidence track presets."
            ),
        }

    def v2_17_pressure_factor(packet, axis):
        return 0.86 + float(packet["axis_pressure"].get(axis, 70.0)) / 500.0

    def v2_17_constraint_review(
        packet,
        ledger_replay,
        *,
        plan_key,
        demand_multiplier,
        communication_fanout,
        failure_multiplier,
        privacy_security_depth,
        carbon_cap_pct,
        governance_depth,
    ):
        plan = packet["plan_options"][plan_key]
        demand = max(0.1, float(demand_multiplier))
        fanout = max(0.5, float(communication_fanout))
        failures = max(0.1, float(failure_multiplier))
        security_depth = max(1.0, float(privacy_security_depth))
        carbon_cap = max(1.0, float(carbon_cap_pct))
        governance = max(1.0, float(governance_depth))
        coverage_gap = float(ledger_replay["evidence_gap_pct"])

        values = {
            "capacity": plan["base_load"] * demand * v2_17_pressure_factor(packet, "capacity") / max(plan["capacity"], 0.1),
            "communication": demand
            * fanout
            * plan["communication_factor"]
            * (1.0 + 0.03 * security_depth)
            * v2_17_pressure_factor(packet, "communication"),
            "reliability": failures
            * demand
            * (1.35 - plan["recovery_score"])
            * v2_17_pressure_factor(packet, "reliability")
            / 0.58,
            "security_privacy": (0.55 + 0.11 * security_depth + 0.002 * coverage_gap)
            * v2_17_pressure_factor(packet, "security_privacy")
            / max(plan["security_score"], 0.1),
            "robustness": (0.52 + 0.08 * demand + 0.07 * failures + 0.03 * security_depth)
            * v2_17_pressure_factor(packet, "robustness")
            / max(plan["robustness_score"], 0.1),
            "carbon": demand
            * plan["carbon_factor"]
            * 100.0
            * v2_17_pressure_factor(packet, "carbon")
            / carbon_cap,
            "governance": (0.54 + 0.08 * governance + 0.0022 * coverage_gap)
            * v2_17_pressure_factor(packet, "governance")
            / max(plan["governance_score"], 0.1),
        }
        mitigations = {
            "capacity": "reduce demand, add headroom, or narrow launch scope",
            "communication": "change topology, reduce fanout, overlap work, or defer distributed coupling",
            "reliability": "increase recovery automation, checkpoint margin, or rollback capacity",
            "security_privacy": "preserve detection signals while tightening access and privacy controls",
            "robustness": "add fallback, red-team replay, and drift-specific validation",
            "carbon": "shift placement, schedule carbon-aware work, or reduce absolute demand",
            "governance": "add audit evidence, ownership, and release gates before launch",
        }
        rows = []
        for axis, value in values.items():
            ratio = max(0.0, float(value))
            rows.append(
                {
                    "axis": axis,
                    "guardrail": v2_17_axis_label(axis),
                    "value": round(ratio, 3),
                    "limit": 1.0,
                    "ratio": round(ratio, 3),
                    "status": "PASS" if ratio <= 1.0 else "FAIL",
                    "mitigation": mitigations[axis],
                }
            )
        binding = max(rows, key=lambda row: row["ratio"])
        violations = tuple(row["axis"] for row in rows if row["status"] == "FAIL")
        return {
            "plan_key": plan_key,
            "plan_label": plan["label"],
            "axis_rows": rows,
            "binding_axis": binding["axis"],
            "binding_guardrail": binding["guardrail"],
            "binding_ratio": binding["ratio"],
            "violations": violations,
            "feasible": not violations,
            "controls": {
                "demand_multiplier": demand,
                "communication_fanout": fanout,
                "failure_multiplier": failures,
                "privacy_security_depth": security_depth,
                "carbon_cap_pct": carbon_cap,
                "governance_depth": governance,
            },
        }

    def v2_17_recommended_revision(review):
        binding = review["binding_axis"]
        ratio = float(review["binding_ratio"])
        if binding in ("security_privacy", "governance", "robustness"):
            return "monitor_gate"
        if binding in ("capacity", "communication", "carbon") and ratio <= 1.18:
            return "defer_scope"
        return "redesign_architecture"

    def v2_17_revision_review(packet, review, *, target_axis, action, intensity_pct):
        before = {row["axis"]: float(row["ratio"]) for row in review["axis_rows"]}
        after = dict(before)
        intensity = max(0.0, min(100.0, float(intensity_pct))) / 100.0
        target = review["binding_axis"] if target_axis == "auto" else target_axis
        residual = 28.0 + 12.0 * len(review["violations"])
        note = ""

        if action == "relax_noncritical":
            after[target] = max(0.0, after.get(target, 0.0) * (1.0 - 0.22 * intensity))
            after["governance"] = after.get("governance", 0.0) * (1.0 + 0.10 * intensity)
            residual += 34.0 * intensity
            note = "Relaxation creates explicit residual risk and may increase governance scrutiny."
        elif action == "monitor_gate":
            for axis in ("security_privacy", "robustness", "governance", target):
                after[axis] = max(0.0, after.get(axis, 0.0) * (1.0 - 0.16 * intensity))
            after["capacity"] = after.get("capacity", 0.0) * (1.0 + 0.04 * intensity)
            residual += 14.0 * intensity
            note = "Monitoring reduces unknown risk but spends capacity and operational attention."
        elif action == "defer_scope":
            for axis in ("capacity", "communication", "carbon", "serving", target):
                if axis in after:
                    after[axis] = max(0.0, after[axis] * (1.0 - 0.24 * intensity))
            residual += 10.0 * intensity
            note = "Deferring scope buys margin by reducing launch value and coverage."
        else:
            for axis in (target, "communication", "reliability", "capacity"):
                if axis in after:
                    after[axis] = max(0.0, after[axis] * (1.0 - 0.30 * intensity))
            after["carbon"] = after.get("carbon", 0.0) * (1.0 + 0.04 * intensity)
            residual += 18.0 * intensity
            note = "Redesign attacks the binding guardrail but adds schedule, carbon, and validation cost."

        rows = []
        for axis in before:
            before_ratio = before[axis]
            after_ratio = after[axis]
            rows.append(
                {
                    "axis": axis,
                    "guardrail": v2_17_axis_label(axis),
                    "before_ratio": round(before_ratio, 3),
                    "after_ratio": round(after_ratio, 3),
                    "delta": round(after_ratio - before_ratio, 3),
                    "status": "PASS" if after_ratio <= 1.0 else "FAIL",
                }
            )
        remaining = tuple(row["axis"] for row in rows if row["status"] == "FAIL")
        residual_risk_pct = min(100.0, residual)
        feasible = not remaining and residual_risk_pct <= 75.0
        status = "PASS" if feasible else "PROVISIONAL" if not remaining else "FAIL"
        return {
            "target_axis": target,
            "target_guardrail": v2_17_axis_label(target),
            "action": action,
            "action_label": packet["revision_actions"][action],
            "intensity_pct": round(intensity * 100.0, 1),
            "rows": rows,
            "remaining_violations": remaining,
            "residual_risk_pct": round(residual_risk_pct, 2),
            "status": status,
            "feasible": feasible,
            "note": note,
            "recommended_action": v2_17_recommended_revision(review),
        }

    def v2_17_board_review(
        packet,
        *,
        selected_plan_key,
        rejected_plan_key,
        validation_key,
        residual_risk_key,
        ledger_replay,
        review,
        revision,
    ):
        validation = packet["validation_packages"][validation_key]
        covered = "all" in validation["covers"] or revision["target_axis"] in validation["covers"]
        selected = packet["plan_options"][selected_plan_key]
        rejected = packet["plan_options"][rejected_plan_key]
        risk_options = packet["lens"]["risk_options"]
        residual_risk = risk_options[int(residual_risk_key)]
        criteria = (
            {
                "criterion": "Ledger replay covers enough Volume II evidence",
                "status": ledger_replay["coverage_pct"] >= 50.0,
                "evidence": f"{ledger_replay['coverage_pct']:.0f}% matching ledger coverage",
            },
            {
                "criterion": "Binding fleet amount is named",
                "status": bool(ledger_replay["binding"]["axis"]),
                "evidence": ledger_replay["binding"]["amount"],
            },
            {
                "criterion": "Multi-constraint review was run",
                "status": bool(review["axis_rows"]),
                "evidence": f"Binding guardrail: {review['binding_guardrail']} at {review['binding_ratio']:.2f}x",
            },
            {
                "criterion": "Revision leaves launch guardrails defensible",
                "status": revision["status"] in ("PASS", "PROVISIONAL"),
                "evidence": f"{revision['status']}; residual risk {revision['residual_risk_pct']:.0f}%",
            },
            {
                "criterion": "Rejected alternative is distinct",
                "status": selected_plan_key != rejected_plan_key,
                "evidence": f"Selected {selected['label']}; rejected {rejected['label']}",
            },
            {
                "criterion": "Validation package covers the active guardrail",
                "status": covered,
                "evidence": validation["label"],
            },
            {
                "criterion": "Residual risk is named",
                "status": bool(residual_risk),
                "evidence": residual_risk,
            },
        )
        passed = sum(1 for item in criteria if item["status"])
        completeness = passed / len(criteria) * 100.0
        blocking_fail = any(
            not item["status"]
            for item in criteria
            if item["criterion"]
            in (
                "Rejected alternative is distinct",
                "Validation package covers the active guardrail",
                "Revision leaves launch guardrails defensible",
            )
        )
        if blocking_fail or revision["status"] == "FAIL":
            outcome = "REJECT"
        elif completeness < 100.0 or revision["status"] == "PROVISIONAL":
            outcome = "PROVISIONAL"
        else:
            outcome = "APPROVE"
        return {
            "criteria": tuple(
                {
                    "criterion": item["criterion"],
                    "status": "PASS" if item["status"] else "FAIL",
                    "evidence": item["evidence"],
                }
                for item in criteria
            ),
            "selected_plan_key": selected_plan_key,
            "selected_plan": selected["label"],
            "rejected_plan_key": rejected_plan_key,
            "rejected_plan": rejected["label"],
            "validation_key": validation_key,
            "validation_label": validation["label"],
            "residual_risk": residual_risk,
            "completeness_pct": round(completeness, 2),
            "outcome": outcome,
            "outcome_label": packet["board_outcomes"][outcome],
        }

    def v2_17_markdown_table(rows, columns):
        def clean(value):
            if isinstance(value, float):
                text = f"{value:.3f}".rstrip("0").rstrip(".")
            else:
                text = str(value)
            return text.replace("|", "/").replace("\n", " ")

        header = "| " + " | ".join(title for title, _key in columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        body = []
        for row in rows:
            body.append("| " + " | ".join(clean(row.get(key, "")) for _title, key in columns) + " |")
        return "\n".join([header, sep, *body])

    def v2_17_score_fig(go, apply_plotly_theme, colors, replay):
        names = [row["amount"] for row in replay["score_rows"]]
        values = [row["score_pct"] for row in replay["score_rows"]]
        marker = [
            colors["RedLine"] if row["axis"] == replay["binding"]["axis"] else colors["BlueLine"]
            for row in replay["score_rows"]
        ]
        fig = go.Figure()
        fig.add_bar(x=names, y=values, marker_color=marker, name="Binding score")
        fig.add_hline(y=75, line_dash="dash", line_color=colors["OrangeLine"], annotation_text="watch threshold")
        fig.update_layout(title="Part A binding amount score", yaxis_title="Score (%)", xaxis_title="Fleet amount")
        fig.update_yaxes(range=[0, max(105, math.ceil(max(values) / 10.0) * 10)])
        return apply_plotly_theme(fig)

    def v2_17_review_fig(go, apply_plotly_theme, colors, review):
        rows = review["axis_rows"]
        fig = go.Figure()
        fig.add_bar(
            x=[row["guardrail"] for row in rows],
            y=[row["ratio"] for row in rows],
            marker_color=[colors["RedLine"] if row["status"] == "FAIL" else colors["GreenLine"] for row in rows],
            name="Spend / guardrail",
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color=colors["BlueLine"], annotation_text="launch limit")
        fig.update_layout(title="Part B multi-constraint review", yaxis_title="Ratio to guardrail", xaxis_title="Guardrail")
        return apply_plotly_theme(fig)

    def v2_17_revision_fig(go, apply_plotly_theme, colors, revision):
        rows = revision["rows"]
        fig = go.Figure()
        fig.add_bar(
            x=[row["guardrail"] for row in rows],
            y=[row["before_ratio"] for row in rows],
            marker_color=colors["OrangeLine"],
            name="Before revision",
        )
        fig.add_bar(
            x=[row["guardrail"] for row in rows],
            y=[row["after_ratio"] for row in rows],
            marker_color=colors["BlueLine"],
            name="After revision",
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color=colors["RedLine"], annotation_text="guardrail")
        fig.update_layout(barmode="group", title="Part C revision moves risk", yaxis_title="Ratio to guardrail")
        return apply_plotly_theme(fig)

    def v2_17_board_fig(go, apply_plotly_theme, colors, board):
        fig = go.Figure()
        fig.add_bar(
            x=[row["criterion"] for row in board["criteria"]],
            y=[1 if row["status"] == "PASS" else 0 for row in board["criteria"]],
            marker_color=[colors["GreenLine"] if row["status"] == "PASS" else colors["RedLine"] for row in board["criteria"]],
            name="Readiness",
        )
        fig.update_layout(title="Part D board readiness", yaxis_title="Pass flag", xaxis_title="Review criterion")
        fig.update_yaxes(range=[0, 1.1], tickvals=[0, 1], ticktext=["Fail", "Pass"])
        return apply_plotly_theme(fig)

    return (
        v2_17_axis_options,
        v2_17_board_fig,
        v2_17_board_review,
        v2_17_constraint_review,
        v2_17_ledger_replay,
        v2_17_markdown_table,
        v2_17_review_fig,
        v2_17_revision_fig,
        v2_17_revision_review,
        v2_17_score_fig,
        v2_17_track_packet,
    )


@app.cell
def _(
    v2_17_hardware,
    v2_17_model,
    v2_17_profile,
    v2_17_track_packet,
    v2_17_variant,
):
    v2_17_packet = v2_17_track_packet(v2_17_profile, v2_17_variant, v2_17_hardware, v2_17_model)
    return (v2_17_packet,)


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_arc_context,
    track_context,
    v2_17_metadata,
    v2_17_packet,
    v2_17_profile,
):
    _shell = mo.vstack(
        [
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            mo.Html(f"""
            <div class="mlsysbook-lab-shell">
                <div class="lab-shell-header" style="border-left: 5px solid #A51C30; padding-left: 16px; margin-bottom: 20px;">
                    <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.72rem; font-weight: 800; color: #A51C30; text-transform: uppercase; letter-spacing: 0.08em;">
                            ML Systems Textbook &middot; Volume II &middot; Chapter 17 &middot; Capstone Lab 17
                        </span>
                    </div>
                    <h1 style="margin: 0 0 10px 0; color: #0F172A; font-size: 1.85rem; font-weight: 800; line-height: 1.2;">
                        The Fleet Synthesis: Co-Designing Distributed Systems at Scale
                    </h1>
                    <p style="margin: 0 0 12px 0; color: #475569; font-size: 0.95rem; line-height: 1.5; max-width: 850px;">
                        Synthesize the entire Volume II distributed curriculum into an executive deployment review board decision.
                        Follow the active C3 constraint across hardware, networking, storage, coordination, reliability, serving, and governance.
                    </p>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        <span class="mlsysbook-chip" style="background: #FFF1F2; color: #A51C30; border: 1px solid #FECDD3; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700;">
                            Track: {v2_17_profile.label}
                        </span>
                        <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;">
                            Stakeholder: {v2_17_packet['stakeholder']}
                        </span>
                        <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;">
                            Hardware: {v2_17_packet['hardware_ref']}
                        </span>
                        <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;">
                            Model: {v2_17_packet['model_ref']}
                        </span>
                        <span class="mlsysbook-chip" style="background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700;">
                            Focus: Conjunctive Fleet Guardrails
                        </span>
                        <span class="mlsysbook-chip" style="background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700;">
                            Deliverable: Deployment Review Memo
                        </span>
                    </div>
                </div>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #0F172A; font-size: 1.1rem;">System Scenario: {v2_17_packet['label']} Fleet Synthesis</h3>
                <p style="color: #334155; font-size: 0.9rem; line-height: 1.6;">
                    The fleet, not the model, is the true object of engineering. A defensible deployment review follows the binding C3
                    term across infrastructure, communication, coordination, serving, operations, and governance until the displaced cost is visible.
                </p>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-top: 12px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;">The Architectural Invariants of Fleet Synthesis:</div>
                    <ul class="mlsysbook-list" style="margin: 0; padding-left: 1.25rem; font-size: 0.85rem; color: #334155; line-height: 1.6;">
                        <li><strong>The Whole-Fleet Invariant:</strong> Optimization in one layer (e.g. quantization, kernel fusion) that degrades another (e.g. tail latency, auditability, recovery MTTR) is technical debt, not progress.</li>
                        <li><strong>Conjunctive Gate Invariant:</strong> Production release requires: Launch = &bigwedge;<sub><i>i</i></sub> (Metric<sub><i>i</i></sub> &le; Guardrail<sub><i>i</i></sub>). A near-perfect aggregate score cannot mask a single critical boundary violation.</li>
                        <li><strong>Displaced Cost Invariant:</strong> Cost and latency cannot be destroyed; they are displaced into queues, energy, retry storms, or governance debt.</li>
                        <li><strong>Defense In Depth:</strong> Every deployment must name an explicit rejected alternative, a primary validation suite, and a named residual risk owner.</li>
                    </ul>
                </div>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-bottom: 20px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">LEARNING OBJECTIVES</div>
                <ul class="mlsysbook-list" style="margin: 0; padding-left: 1.25rem; font-size: 0.88rem; color: #334155; line-height: 1.6;">
                    <li><strong>Assemble fleet ledger:</strong> Replay earlier Volume II architectural decisions and identify the binding amount.</li>
                    <li><strong>Execute multi-constraint stress review:</strong> Evaluate the candidate plan across capacity, communication, reliability, security, robustness, carbon, and governance.</li>
                    <li><strong>Resolve guardrail conflicts:</strong> Defensibly revise the architecture by relaxing, deferring, monitoring, or redesigning, tracking where risk moves.</li>
                    <li><strong>Author executive deployment board decision:</strong> Sign off on deployment readiness with a defensible rejected alternative and named residual risks.</li>
                </ul>
                <div style="background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 6px; padding: 10px 14px; margin-top: 14px; font-size: 0.85rem; color: #0369A1;">
                    <strong>CORE QUESTION:</strong> <em>\"How do all distributed systems dimensions cohere into a single, defensible production fleet deployment?\"</em>
                </div>
            </div>
            """),
            track_context(v2_17_profile),
            track_arc_context(v2_17_profile, v2_17_metadata.lab_id),
            source_trace(
                {
                    "book_anchor": v2_17_metadata.book_anchor,
                    "chapter_claim": "The fleet is the object; follow the active C3 constraint across layers.",
                    "hardware_ref": v2_17_packet["hardware_ref"],
                    "model_ref": v2_17_packet["model_ref"],
                    "source_policy": v2_17_packet["source_policy"],
                },
                summary="Opening source trace: chapter invariant and track registries",
            ),
        ]
    )
    _shell
    return


@app.cell(hide_code=True)
def _(mo):
    v2_17_partA_pred = mo.ui.radio(
        options={
            "Capacity and compute will bind first": "capacity",
            "Communication and coordination will bind first": "communication",
            "Reliability and recovery will bind first": "reliability",
            "Serving/performance will bind first": "serving",
            "Security/privacy will bind first": "security_privacy",
            "Carbon will bind first": "carbon",
            "Governance and auditability will bind first": "governance",
        },
        label="Part A prediction: which fleet amount will dominate the deployment review?",
    )
    v2_17_partA_evidence_floor = mo.ui.slider(
        start=40,
        stop=100,
        value=75,
        step=5,
        label="Required matching ledger coverage (%)",
    )
    v2_17_partA_preset_weight = mo.ui.slider(
        start=20,
        stop=80,
        value=45,
        step=5,
        label="Confidence for track presets when ledger entries are missing (%)",
    )
    v2_17_partA_checkpoint = mo.ui.radio(
        options={
            "Attack the binding fleet amount first": "attack_binding",
            "Fill evidence gaps before approving architecture": "fill_gaps",
            "Accept lower-confidence track presets explicitly": "accept_presets",
            "Use a model-only summary": "model_only",
        },
        label="Part A checkpoint: what should the architecture review do with this ledger?",
    )
    return (
        v2_17_partA_checkpoint,
        v2_17_partA_evidence_floor,
        v2_17_partA_pred,
        v2_17_partA_preset_weight,
    )


@app.cell(hide_code=True)
def _(mo, v2_17_packet):
    _plan_options = {plan["label"]: key for key, plan in v2_17_packet["plan_options"].items()}
    _balanced_plan_label = v2_17_packet["plan_options"]["balanced_policy"]["label"]
    v2_17_partB_plan = mo.ui.dropdown(
        options=_plan_options,
        value=_balanced_plan_label,
        label="Candidate architecture plan",
    )
    v2_17_partB_pred = mo.ui.radio(
        options={
            "Capacity fails first": "capacity",
            "Communication fails first": "communication",
            "Reliability fails first": "reliability",
            "Security/privacy fails first": "security_privacy",
            "Robustness fails first": "robustness",
            "Carbon fails first": "carbon",
            "Governance fails first": "governance",
        },
        label="Part B prediction: which guardrail fails first under stress?",
    )
    v2_17_demand_multiplier = mo.ui.slider(
        start=0.8,
        stop=2.5,
        value=1.1,
        step=0.05,
        label="Demand multiplier",
    )
    v2_17_communication_fanout = mo.ui.slider(
        start=0.8,
        stop=2.0,
        value=1.05,
        step=0.05,
        label="Communication / fanout multiplier",
    )
    v2_17_failure_multiplier = mo.ui.slider(
        start=0.5,
        stop=2.5,
        value=1.1,
        step=0.05,
        label="Failure-rate multiplier",
    )
    v2_17_privacy_security_depth = mo.ui.slider(
        start=1,
        stop=4,
        value=2,
        step=1,
        label="Privacy/security control depth",
    )
    v2_17_carbon_cap_pct = mo.ui.slider(
        start=60,
        stop=130,
        value=90,
        step=5,
        label="Carbon cap (% of baseline allowance)",
    )
    v2_17_governance_depth = mo.ui.slider(
        start=1,
        stop=4,
        value=2,
        step=1,
        label="Governance depth",
    )
    v2_17_partB_checkpoint = mo.ui.radio(
        options={
            "Approve because every guardrail passes": "approve_all_pass",
            "Revise the binding guardrail before approval": "revise_binding",
            "Narrow rollout scope and retest": "narrow_scope",
            "Reject the candidate plan": "reject_plan",
        },
        label="Part B checkpoint: what does the guardrail review imply?",
    )
    return (
        v2_17_carbon_cap_pct,
        v2_17_communication_fanout,
        v2_17_demand_multiplier,
        v2_17_failure_multiplier,
        v2_17_governance_depth,
        v2_17_partB_checkpoint,
        v2_17_partB_plan,
        v2_17_partB_pred,
        v2_17_privacy_security_depth,
    )


@app.cell(hide_code=True)
def _(mo, v2_17_axis_options, v2_17_packet):
    _axis_options = {"Auto: current binding guardrail": "auto", **v2_17_axis_options()}
    v2_17_partC_pred = mo.ui.radio(
        options={
            "Relax a noncritical target and disclose the risk": "relax_noncritical",
            "Monitor and gate the risky slice": "monitor_gate",
            "Defer scope until margin improves": "defer_scope",
            "Redesign architecture around the binding guardrail": "redesign_architecture",
        },
        label="Part C prediction: which revision mode is defensible?",
    )
    v2_17_revision_axis = mo.ui.dropdown(
        options=_axis_options,
        value="Auto: current binding guardrail",
        label="Target guardrail for revision",
    )
    v2_17_revision_action = mo.ui.dropdown(
        options={label: key for key, label in v2_17_packet["revision_actions"].items()},
        value=v2_17_packet["revision_actions"]["redesign_architecture"],
        label="Revision action",
    )
    v2_17_revision_intensity = mo.ui.slider(
        start=10,
        stop=90,
        value=45,
        step=5,
        label="Revision intensity (%)",
    )
    v2_17_partC_checkpoint = mo.ui.radio(
        options={
            "Use the revised plan if all guardrails are defensible": "use_revision",
            "Monitor residual risk before broad rollout": "monitor_residual",
            "Defer launch scope until the binding guardrail has margin": "defer_launch",
            "Reject because risk was only hidden, not managed": "reject_hidden_risk",
        },
        label="Part C checkpoint: what revision rule goes into the memo?",
    )
    return (
        v2_17_partC_checkpoint,
        v2_17_partC_pred,
        v2_17_revision_action,
        v2_17_revision_axis,
        v2_17_revision_intensity,
    )


@app.cell(hide_code=True)
def _(mo, v2_17_packet):
    _plan_options = {plan["label"]: key for key, plan in v2_17_packet["plan_options"].items()}
    _risk_options = {risk: str(index) for index, risk in enumerate(v2_17_packet["lens"]["risk_options"])}
    _validation_options = {
        package["label"]: key for key, package in v2_17_packet["validation_packages"].items()
    }
    _balanced_plan_label = v2_17_packet["plan_options"]["balanced_policy"]["label"]
    _local_baseline_label = v2_17_packet["plan_options"]["local_baseline"]["label"]
    v2_17_partD_pred = mo.ui.radio(
        options={
            "Selected plan, rejected alternative, validation evidence, and residual risk": "complete_packet",
            "Selected plan and one success metric": "metric_only",
            "All prior decisions copied without rejection": "copy_ledger",
            "Approval without residual risk": "no_risk",
        },
        label="Part D prediction: what is the board's minimum evidence packet?",
    )
    v2_17_selected_plan = mo.ui.dropdown(
        options=_plan_options,
        value=_balanced_plan_label,
        label="Selected deployment plan",
    )
    v2_17_rejected_plan = mo.ui.dropdown(
        options=_plan_options,
        value=_local_baseline_label,
        label="Rejected alternative",
    )
    v2_17_validation_package = mo.ui.dropdown(
        options=_validation_options,
        value=v2_17_packet["validation_packages"]["full_board_packet"]["label"],
        label="Validation evidence package",
    )
    v2_17_residual_risk = mo.ui.dropdown(
        options=_risk_options,
        value=v2_17_packet["lens"]["risk_options"][0],
        label="Top residual risk",
    )
    v2_17_partD_checkpoint = mo.ui.radio(
        options={
            "Approve the selected plan with validation and residual risk": "approve",
            "Provisional approval pending validation": "provisional",
            "Reject until the binding guardrail is repaired": "reject",
            "Escalate to architecture redesign": "redesign",
        },
        label="Part D checkpoint: what is the board decision?",
    )
    return (
        v2_17_partD_checkpoint,
        v2_17_partD_pred,
        v2_17_rejected_plan,
        v2_17_residual_risk,
        v2_17_selected_plan,
        v2_17_validation_package,
    )


@app.cell(hide_code=True)
def _(mo):
    v2_17_student_id = mo.ui.text(label="Student or team ID", placeholder="Optional")
    v2_17_final_note = mo.ui.text_area(
        label="Final memo note",
        placeholder="State the one trade-off the board must keep visible after launch.",
        full_width=True,
    )
    return v2_17_final_note, v2_17_student_id


@app.cell
def _(
    ledger,
    v2_17_carbon_cap_pct,
    v2_17_communication_fanout,
    v2_17_constraint_review,
    v2_17_demand_multiplier,
    v2_17_failure_multiplier,
    v2_17_governance_depth,
    v2_17_ledger_replay,
    v2_17_packet,
    v2_17_partA_evidence_floor,
    v2_17_partA_preset_weight,
    v2_17_partB_plan,
    v2_17_privacy_security_depth,
    v2_17_revision_action,
    v2_17_revision_axis,
    v2_17_revision_intensity,
    v2_17_revision_review,
):
    v2_17_ledger_result = v2_17_ledger_replay(
        v2_17_packet,
        ledger._state.history,
        v2_17_partA_evidence_floor.value,
        v2_17_partA_preset_weight.value,
    )
    v2_17_review = v2_17_constraint_review(
        v2_17_packet,
        v2_17_ledger_result,
        plan_key=v2_17_partB_plan.value,
        demand_multiplier=v2_17_demand_multiplier.value,
        communication_fanout=v2_17_communication_fanout.value,
        failure_multiplier=v2_17_failure_multiplier.value,
        privacy_security_depth=v2_17_privacy_security_depth.value,
        carbon_cap_pct=v2_17_carbon_cap_pct.value,
        governance_depth=v2_17_governance_depth.value,
    )
    v2_17_revision = v2_17_revision_review(
        v2_17_packet,
        v2_17_review,
        target_axis=v2_17_revision_axis.value,
        action=v2_17_revision_action.value,
        intensity_pct=v2_17_revision_intensity.value,
    )
    return v2_17_ledger_result, v2_17_review, v2_17_revision


@app.cell
def _(
    v2_17_board_review,
    v2_17_ledger_result,
    v2_17_packet,
    v2_17_rejected_plan,
    v2_17_residual_risk,
    v2_17_review,
    v2_17_revision,
    v2_17_selected_plan,
    v2_17_validation_package,
):
    v2_17_board = v2_17_board_review(
        v2_17_packet,
        selected_plan_key=v2_17_selected_plan.value,
        rejected_plan_key=v2_17_rejected_plan.value,
        validation_key=v2_17_validation_package.value,
        residual_risk_key=v2_17_residual_risk.value,
        ledger_replay=v2_17_ledger_result,
        review=v2_17_review,
        revision=v2_17_revision,
    )
    return (v2_17_board,)


@app.cell(hide_code=True)
def _(
    COLORS,
    MathPeek,
    apply_plotly_theme,
    big_takeaways,
    build_lab_report,
    gated_hypothesis_card,
    go,
    instrumentation_console,
    ledger,
    mo,
    report_export_panel,
    source_trace,
    v2_17_board,
    v2_17_board_fig,
    v2_17_carbon_cap_pct,
    v2_17_chapter,
    v2_17_communication_fanout,
    v2_17_demand_multiplier,
    v2_17_failure_multiplier,
    v2_17_final_note,
    v2_17_governance_depth,
    v2_17_ledger_result,
    v2_17_markdown_table,
    v2_17_metadata,
    v2_17_packet,
    v2_17_partA_checkpoint,
    v2_17_partA_evidence_floor,
    v2_17_partA_pred,
    v2_17_partA_preset_weight,
    v2_17_partB_checkpoint,
    v2_17_partB_plan,
    v2_17_partB_pred,
    v2_17_partC_checkpoint,
    v2_17_partC_pred,
    v2_17_partD_checkpoint,
    v2_17_partD_pred,
    v2_17_privacy_security_depth,
    v2_17_rejected_plan,
    v2_17_residual_risk,
    v2_17_review,
    v2_17_review_fig,
    v2_17_revision,
    v2_17_revision_action,
    v2_17_revision_axis,
    v2_17_revision_fig,
    v2_17_revision_intensity,
    v2_17_score_fig,
    v2_17_selected_plan,
    v2_17_student_id,
    v2_17_validation_package,
    v2_17_variant,
):
    def v2_17_prediction_callout(label, predicted, actual, success_text, correction_text):
        if predicted is None:
            return mo.callout(mo.md(f"**Prediction needed:** choose a Part {label} prediction before using this as final evidence."), kind="warn")
        if predicted == actual:
            return mo.callout(mo.md(success_text), kind="success")
        return mo.callout(mo.md(correction_text), kind="warn")

    def build_part_a():
        binding = v2_17_ledger_result["binding"]
        ledger_table = v2_17_markdown_table(
            v2_17_ledger_result["rows"],
            (
                ("Lab", "chapter"),
                ("Layer", "layer"),
                ("Amount", "amount"),
                ("Track realization", "track_realization"),
                ("Source", "source"),
                ("Confidence", "confidence_pct"),
                ("Decision or preset", "decision"),
            ),
        )
        score_table = v2_17_markdown_table(
            v2_17_ledger_result["score_rows"],
            (
                ("Amount", "amount"),
                ("Track pressure", "track_pressure"),
                ("Evidence", "evidence_points"),
                ("Debt penalty", "evidence_debt_penalty"),
                ("Score", "score_pct"),
                ("Status", "status"),
            ),
        )
        boundary = (
            mo.callout(
                mo.md(
                    f"**Evidence-debt boundary:** matching ledger coverage is {v2_17_ledger_result['coverage_pct']:.0f}% "
                    f"against your {v2_17_partA_evidence_floor.value:.0f}% floor. The report must label lower-confidence presets."
                ),
                kind="danger",
            )
            if v2_17_ledger_result["evidence_gap_pct"] > 0
            else mo.callout(
                mo.md("**Ledger coverage passes the selected floor.** Presets may still appear, but they are not the binding evidence debt."),
                kind="success",
            )
        )
        return mo.vstack(
            [
                mo.md(
                    f"""
    ## Part A - Assemble The Fleet Architecture Ledger

    **Scenario.** {v2_17_packet['stakeholder']} is preparing a deployment review
    for **{v2_17_packet['label']}**. The board asks for one architecture ledger
    rather than disconnected prior-lab answers.

    **Track architecture goal:** {v2_17_packet['lens']['architecture_goal']}
                    """
                ),
                gated_hypothesis_card(
                    v2_17_partA_pred,
                    gate_label="Hypothesis Gate A",
                    title="1. Formulate Binding Fleet Amount Hypothesis",
                    subtitle=f"Predict which fleet dimension dominates the deployment review for {v2_17_packet['label']}.",
                ),
                instrumentation_console(
                    mo.hstack([v2_17_partA_evidence_floor, v2_17_partA_preset_weight], widths="equal"),
                    title="Ledger Replay Controls",
                    subtitle="Adjust ledger coverage threshold and default preset confidence weight.",
                ),
                v2_17_score_fig(go, apply_plotly_theme, COLORS, v2_17_ledger_result),
                mo.md(f"**Binding amount:** `{binding['amount']}` with score `{binding['score_pct']:.1f}%`."),
                mo.md("**Binding Score Table**\n\n" + score_table),
                mo.md("**Volume II Ledger Replay**\n\n" + ledger_table),
                boundary,
                MathPeek(
                    r"\text{score} = \text{track\_pressure} + \text{ledger\_evidence} + \text{preset\_evidence} + \text{debt\_penalty}",
                    {
                        "track pressure": f"{binding['track_pressure']:.1f}",
                        "ledger evidence": f"{binding['evidence_points']:.1f}",
                        "evidence debt penalty": f"{binding['evidence_debt_penalty']:.1f}",
                        "binding score": f"{binding['score_pct']:.1f}%",
                        "diagnostic rule": "start with observed evidence, map to binding C3 term, name layer that owns intervention",
                    },
                ),
                source_trace(
                    {
                        "chapter_anchor": "Conclusion: Six Principles; Complete Production System",
                        "claim": "Any one fleet layer can bind the whole production system.",
                        "helper": "notebook-local v2_17_ledger_replay",
                        "hardware_ref": v2_17_packet["hardware_ref"],
                        "model_ref": v2_17_packet["model_ref"],
                    },
                    summary="Part A source trace: fleet ledger replay",
                ),
                v2_17_partA_checkpoint,
                v2_17_prediction_callout(
                    "A",
                    v2_17_partA_pred.value,
                    binding["axis"],
                    "**Prediction check:** your predicted binding amount matches the ledger-weighted diagnosis.",
                    f"**Prediction check:** current evidence binds on **{binding['amount']}**, not the selected prediction. The capstone lets evidence overrule the easy story.",
                ),
            ]
        )

    def build_part_b():
        review_table = v2_17_markdown_table(
            v2_17_review["axis_rows"],
            (
                ("Guardrail", "guardrail"),
                ("Value", "value"),
                ("Limit", "limit"),
                ("Ratio", "ratio"),
                ("Status", "status"),
                ("Mitigation", "mitigation"),
            ),
        )
        failure = (
            mo.callout(
                mo.md(
                    f"**Launch failure:** `{v2_17_review['binding_guardrail']}` is at "
                    f"`{v2_17_review['binding_ratio']:.2f}x` its guardrail. Revise the plan, scope, or controls."
                ),
                kind="danger",
            )
            if not v2_17_review["feasible"]
            else mo.callout(mo.md("**All guardrails pass under the current stress settings.** Keep the validation evidence in the final memo."), kind="success")
        )
        return mo.vstack(
            [
                mo.md(
                    """
    ## Part B - Run A Multi-Constraint Fleet Review

    The board now stresses the candidate architecture. The launch rule is conjunctive:
    capacity, communication, reliability, security/privacy, robustness, carbon, and
    governance must all remain inside their guardrails.
                    """
                ),
                v2_17_partB_pred,
                mo.hstack([v2_17_partB_plan, v2_17_demand_multiplier, v2_17_communication_fanout], widths="equal"),
                mo.hstack([v2_17_failure_multiplier, v2_17_privacy_security_depth], widths="equal"),
                mo.hstack([v2_17_carbon_cap_pct, v2_17_governance_depth], widths="equal"),
                v2_17_review_fig(go, apply_plotly_theme, COLORS, v2_17_review),
                mo.md("**Guardrail Review Table**\n\n" + review_table),
                failure,
                MathPeek(
                    r"\text{feasible} = \bigwedge_{i} (\text{ratio}_i \le 1.0),\quad \text{ratio}_i = \frac{\text{observed\_amount}_i}{\text{guardrail\_limit}_i}",
                    {
                        "binding guardrail": v2_17_review['binding_guardrail'],
                        "binding ratio": f"{v2_17_review['binding_ratio']:.2f}x",
                        "feasible": "YES" if v2_17_review['feasible'] else "NO",
                        "correlations": "higher demand compounds communication; failures compound MTTR",
                    },
                ),
                source_trace(
                    {
                        "chapter_anchor": "Conclusion: Complete Production System; Fallacies and Pitfalls",
                        "formula": "feasible = all(guardrail_ratio <= 1.0)",
                        "helper": "notebook-local v2_17_constraint_review",
                        "scenario_constants": "demand, fanout, failure, privacy/security depth, carbon cap, governance depth",
                    },
                    summary="Part B source trace: multi-constraint guardrail review",
                ),
                v2_17_partB_checkpoint,
                v2_17_prediction_callout(
                    "B",
                    v2_17_partB_pred.value,
                    v2_17_review["binding_axis"],
                    "**Prediction check:** your predicted first-failing guardrail matches the stress review.",
                    f"**Prediction check:** the current binding guardrail is **{v2_17_review['binding_guardrail']}**. Use the table to explain why the review changed.",
                ),
            ]
        )

    def build_part_c():
        revision_table = v2_17_markdown_table(
            v2_17_revision["rows"],
            (
                ("Guardrail", "guardrail"),
                ("Before", "before_ratio"),
                ("After", "after_ratio"),
                ("Delta", "delta"),
                ("Status", "status"),
            ),
        )
        status_kind = "success" if v2_17_revision["status"] == "PASS" else "warn" if v2_17_revision["status"] == "PROVISIONAL" else "danger"
        status_card = mo.callout(
            mo.md(
                f"**Revision status:** `{v2_17_revision['status']}`. Target guardrail is "
                f"`{v2_17_revision['target_guardrail']}`; residual risk is "
                f"`{v2_17_revision['residual_risk_pct']:.0f}%`. {v2_17_revision['note']}"
            ),
            kind=status_kind,
        )
        return mo.vstack(
            [
                mo.md(
                    """
    ## Part C - Revise When Guardrails Conflict

    The board will not accept a hidden average score. Choose what to relax, monitor,
    defer, or redesign, then inspect where the risk moved.
                    """
                ),
                gated_hypothesis_card(
                    v2_17_partC_pred,
                    gate_label="Hypothesis Gate C",
                    title="3. Guardrail Conflict Revision Hypothesis",
                    subtitle=f"Predict the defensible revision strategy when guardrails conflict for {v2_17_packet['label']}.",
                ),
                instrumentation_console(
                    mo.hstack([v2_17_revision_axis, v2_17_revision_action, v2_17_revision_intensity], widths="equal"),
                    title="Trade-off Revision Controls",
                    subtitle="Select target guardrail axis, choose revision action, and set intervention intensity.",
                ),
                v2_17_revision_fig(go, apply_plotly_theme, COLORS, v2_17_revision),
                mo.md("**Before/After Guardrail Table**\n\n" + revision_table),
                status_card,
                MathPeek(
                    r"\text{after\_ratio} = \text{before\_ratio} - \text{relief} + \text{displaced\_overhead}",
                    {
                        "target guardrail": v2_17_revision['target_guardrail'],
                        "revision action": v2_17_revision['action_label'],
                        "residual risk": f"{v2_17_revision['residual_risk_pct']:.0f}%",
                        "displacement law": "overhead cannot be eliminated; it is displaced into queues, schedules, or governance debt",
                    },
                ),
                source_trace(
                    {
                        "chapter_anchor": "Conclusion: Fallacies and Pitfalls",
                        "claim": "Overhead cannot be eliminated; it is displaced and must stay visible.",
                        "helper": "notebook-local v2_17_revision_review",
                        "recommended_action": v2_17_revision["recommended_action"],
                    },
                    summary="Part C source trace: trade-off revision",
                ),
                v2_17_partC_checkpoint,
                v2_17_prediction_callout(
                    "C",
                    v2_17_partC_pred.value,
                    v2_17_revision["recommended_action"],
                    "**Prediction check:** your predicted revision mode matches the current binding guardrail.",
                    f"**Prediction check:** this stress case recommends `{v2_17_revision['recommended_action']}`. Defend any different choice with the before/after evidence.",
                ),
            ]
        )

    def build_part_d():
        board_table = v2_17_markdown_table(
            v2_17_board["criteria"],
            (
                ("Criterion", "criterion"),
                ("Status", "status"),
                ("Evidence", "evidence"),
            ),
        )
        outcome_kind = "success" if v2_17_board["outcome"] == "APPROVE" else "warn" if v2_17_board["outcome"] == "PROVISIONAL" else "danger"
        return mo.vstack(
            [
                mo.md(
                    """
    ## Part D - Defend The Deployment Review Board Decision

    Now convert the evidence into a board packet: selected plan, rejected
    alternative, validation package, source trace, and residual risk.
                    """
                ),
                gated_hypothesis_card(
                    v2_17_partD_pred,
                    gate_label="Hypothesis Gate D",
                    title="4. Deployment Board Authorization Hypothesis",
                    subtitle=f"Predict the minimum packet required for executive sign-off on {v2_17_packet['label']}.",
                ),
                instrumentation_console(
                    mo.vstack([
                        mo.hstack([v2_17_selected_plan, v2_17_rejected_plan], widths="equal"),
                        mo.hstack([v2_17_validation_package, v2_17_residual_risk], widths="equal"),
                    ]),
                    title="Board Packet Controls",
                    subtitle="Select deployment plan, specify rejected alternative, select validation test suite, and acknowledge residual risk.",
                ),
                v2_17_board_fig(go, apply_plotly_theme, COLORS, v2_17_board),
                mo.md("**Board Readiness Table**\n\n" + board_table),
                mo.callout(
                    mo.md(
                        f"**Board outcome:** `{v2_17_board['outcome']}` - {v2_17_board['outcome_label']} "
                        f"Completeness is `{v2_17_board['completeness_pct']:.0f}%`."
                    ),
                    kind=outcome_kind,
                ),
                MathPeek(
                    r"\text{packet} = \text{selected\_plan} \land \text{rejected\_alternative} \land \text{validation\_suite} \land (\text{residual\_risk} \ne \varnothing)",
                    {
                        "selected plan": v2_17_board['selected_plan'],
                        "rejected alternative": v2_17_board['rejected_plan'],
                        "validation suite": v2_17_board['validation_label'],
                        "residual risk": v2_17_board['residual_risk'],
                        "board requirement": "defensible review board decision requires explicit rejected alternative and named residual owner",
                    },
                ),
                source_trace(
                    {
                        "chapter_anchor": "Conclusion: Competencies Mastered; The Fleet Stack as Discipline",
                        "helper": "notebook-local v2_17_board_review",
                        "validation_package": v2_17_board["validation_label"],
                        "report_frame": v2_17_packet["lens"]["report_frame"],
                    },
                    summary="Part D source trace: deployment review board decision",
                ),
                v2_17_partD_checkpoint,
                v2_17_prediction_callout(
                    "D",
                    v2_17_partD_pred.value,
                    "complete_packet",
                    "**Prediction check:** the selected evidence packet matches the board's minimum bar.",
                    "**Prediction check:** the board requires selected plan, rejected alternative, validation evidence, and residual risk together.",
                ),
            ]
        )

    def build_synthesis():
        required_widgets = (
            ("Part A binding prediction", v2_17_partA_pred),
            ("Part A checkpoint", v2_17_partA_checkpoint),
            ("Part B guardrail prediction", v2_17_partB_pred),
            ("Part B checkpoint", v2_17_partB_checkpoint),
            ("Part C revision prediction", v2_17_partC_pred),
            ("Part C checkpoint", v2_17_partC_checkpoint),
            ("Part D board prediction", v2_17_partD_pred),
            ("Part D checkpoint", v2_17_partD_checkpoint),
        )
        incomplete = [label for label, widget in required_widgets if widget.value is None]
        if v2_17_selected_plan.value == v2_17_rejected_plan.value:
            incomplete.append("Part D rejected alternative must differ from selected plan")
        selected_label = v2_17_board["selected_plan"]
        rejected_label = v2_17_board["rejected_plan"]
        final_decision = {
            "selected_plan": selected_label,
            "rejected_alternative": rejected_label,
            "board_outcome": v2_17_board["outcome"],
            "binding_fleet_amount": v2_17_ledger_result["binding"]["amount"],
            "binding_guardrail": v2_17_review["binding_guardrail"],
            "revision": v2_17_revision["action_label"],
            "validation_package": v2_17_board["validation_label"],
            "residual_risk": v2_17_board["residual_risk"],
        }
        snapshot = {
            "track_id": v2_17_packet["track_id"],
            "scenario_id": v2_17_variant.scenario_id,
            "hardware_ref": v2_17_packet["hardware_ref"],
            "model_ref": v2_17_packet["model_ref"],
            "hardware_facts": v2_17_packet["hardware_facts"],
            "ledger_replay": v2_17_ledger_result,
            "multi_constraint_review": v2_17_review,
            "revision_review": v2_17_revision,
            "board_review": v2_17_board,
            "final_note": v2_17_final_note.value or "",
        }
        report = build_lab_report(
            v2_17_metadata,
            student_id=v2_17_student_id.value or "",
            track=v2_17_packet["label"],
            scenario=v2_17_packet["scenario"],
            learning_objectives=(
                "Assemble a fleet architecture ledger from prior Volume II evidence.",
                "Run a simultaneous guardrail review across technical, operational, and governance constraints.",
                "Revise a conflicting architecture by naming what risk moves where.",
                "Defend a deployment board decision with selected plan, rejected alternative, validation, and residual risk.",
            ),
            predictions={
                "part_a_binding_amount": v2_17_partA_pred.value,
                "part_b_first_failing_guardrail": v2_17_partB_pred.value,
                "part_c_revision_mode": v2_17_partC_pred.value,
                "part_d_board_packet": v2_17_partD_pred.value,
            },
            knob_settings={
                "evidence_floor_pct": v2_17_partA_evidence_floor.value,
                "preset_weight_pct": v2_17_partA_preset_weight.value,
                "candidate_plan": v2_17_partB_plan.value,
                "demand_multiplier": v2_17_demand_multiplier.value,
                "communication_fanout": v2_17_communication_fanout.value,
                "failure_multiplier": v2_17_failure_multiplier.value,
                "privacy_security_depth": v2_17_privacy_security_depth.value,
                "carbon_cap_pct": v2_17_carbon_cap_pct.value,
                "governance_depth": v2_17_governance_depth.value,
                "revision_axis": v2_17_revision_axis.value,
                "revision_action": v2_17_revision_action.value,
                "revision_intensity_pct": v2_17_revision_intensity.value,
                "selected_plan": v2_17_selected_plan.value,
                "rejected_plan": v2_17_rejected_plan.value,
                "validation_package": v2_17_validation_package.value,
            },
            binding_constraints={
                "binding_fleet_amount": v2_17_ledger_result["binding"]["amount"],
                "ledger_coverage_pct": v2_17_ledger_result["coverage_pct"],
                "binding_guardrail": v2_17_review["binding_guardrail"],
                "binding_guardrail_ratio": v2_17_review["binding_ratio"],
                "revision_status": v2_17_revision["status"],
                "board_outcome": v2_17_board["outcome"],
            },
            decisions={
                "part_a_checkpoint": v2_17_partA_checkpoint.value,
                "part_b_checkpoint": v2_17_partB_checkpoint.value,
                "part_c_checkpoint": v2_17_partC_checkpoint.value,
                "part_d_checkpoint": v2_17_partD_checkpoint.value,
                **final_decision,
            },
            reflections={
                "report_frame": v2_17_packet["lens"]["report_frame"],
                "track_closure": v2_17_packet["lens"]["close"],
                "final_note": v2_17_final_note.value or "Not recorded.",
            },
            residual_risk=(
                f"{v2_17_board['residual_risk']}. The board outcome is {v2_17_board['outcome']}; "
                f"validation package: {v2_17_board['validation_label']}."
            ),
            evidence_summary={
                "ledger_summary": v2_17_ledger_result["architecture_summary"],
                "binding_amount": v2_17_ledger_result["binding"]["amount"],
                "binding_guardrail": f"{v2_17_review['binding_guardrail']} at {v2_17_review['binding_ratio']:.2f}x",
                "revision_status": f"{v2_17_revision['status']} with residual risk {v2_17_revision['residual_risk_pct']:.0f}%",
                "board_outcome": f"{v2_17_board['outcome']} ({v2_17_board['completeness_pct']:.0f}% complete)",
            },
            final_decision=final_decision,
            big_takeaways=(
                "The fleet is the object of engineering; prior decisions become one architecture ledger.",
                "Launch feasibility is conjunctive across capacity, communication, reliability, security, robustness, carbon, and governance.",
                "A revision is only defensible when it names what risk was relaxed, monitored, deferred, or redesigned.",
                "A deployment review board decision needs a selected plan, rejected alternative, validation evidence, and residual risk.",
            ),
            source_trace={
                "book_anchor": v2_17_metadata.book_anchor,
                "hardware_ref": v2_17_packet["hardware_ref"],
                "model_ref": v2_17_packet["model_ref"],
                "track_source": v2_17_packet["source_policy"],
                "variant_defaults": "mlsysbook_labs variants for plan labels, thresholds, refs, and validation tests",
                "notebook_helpers": (
                    "v2_17_ledger_replay",
                    "v2_17_constraint_review",
                    "v2_17_revision_review",
                    "v2_17_board_review",
                ),
            },
            result_snapshot=snapshot,
            incomplete_fields=tuple(incomplete),
        )
        if not incomplete:
            ledger.save(
                chapter=v2_17_chapter,
                design={
                    "lab_id": v2_17_metadata.lab_id,
                    "track_id": v2_17_packet["track_id"],
                    "scenario_id": v2_17_variant.scenario_id,
                    "selected_plan": selected_label,
                    "rejected_alternative": rejected_label,
                    "binding_fleet_amount": v2_17_ledger_result["binding"]["amount"],
                    "binding_guardrail": v2_17_review["binding_guardrail"],
                    "revision_status": v2_17_revision["status"],
                    "validation_package": v2_17_board["validation_label"],
                    "residual_risk": v2_17_board["residual_risk"],
                    "board_outcome": v2_17_board["outcome"],
                    "result_snapshot": snapshot,
                },
            )
        status_kind = "success" if not incomplete else "warn"
        status_text = "Ledger snapshot saved." if not incomplete else "Complete all required predictions, checkpoints, and board choices before final save."
        passed = v2_17_board["outcome"] == "APPROVE"
        return mo.vstack(
            [
                mo.Html(f"""
                <div class="mlsysbook-panel" style="border-left: 4px solid #1F407A; margin-top: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">LEAD ARCHITECT MEMO & REFLECTIONS</div>
                    <h4 style="margin: 0 0 8px 0; color: #0F172A;">Fleet Synthesis Architecture Memorandum</h4>
                    {v2_17_student_id}
                    <div style="margin-top: 12px;">{v2_17_final_note}</div>
                </div>
                """),
                mo.callout(
                    mo.md(
                        f"**Deployment Review Summary:** Selected `{selected_label}` (rejected alternative: `{rejected_label}`). "
                        f"Binding fleet amount: `{v2_17_ledger_result['binding']['amount']}`. "
                        f"Active guardrail: `{v2_17_review['binding_guardrail']}`.  \n\n"
                        f"**Validation Evidence:** `{v2_17_board['validation_label']}` &middot; **Residual Risk:** {v2_17_board['residual_risk']}  \n\n"
                        f"**Track Closure:** {v2_17_packet['lens']['close']}"
                    ),
                    kind=status_kind,
                ),
                mo.callout(mo.md(f"**Report status:** {status_text}"), kind=status_kind),
                mo.Html(f"""
                <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
                    <div style="flex:1; min-width:220px; background:white; border:1px solid #d9dee8;
                                border-radius:10px; padding:16px; border-top:3px solid {COLORS['GreenLine']};">
                        <div style="font-size:0.72rem; font-weight:700; color:#64748b; text-transform:uppercase;">
                            Selected deployment plan</div>
                        <div style="font-size:1.05rem; font-weight:800; color:#1f2937; margin-top:5px;">
                            {selected_label}</div>
                    </div>
                    <div style="flex:1; min-width:220px; background:white; border:1px solid #d9dee8;
                                border-radius:10px; padding:16px; border-top:3px solid {COLORS['OrangeLine']};">
                        <div style="font-size:0.72rem; font-weight:700; color:#64748b; text-transform:uppercase;">
                            Binding fleet amount</div>
                        <div style="font-size:1.05rem; font-weight:800; color:#1f2937; margin-top:5px;">
                            {v2_17_ledger_result['binding']['amount']}</div>
                    </div>
                    <div style="flex:1; min-width:220px; background:white; border:1px solid #d9dee8;
                                border-radius:10px; padding:16px; border-top:3px solid {COLORS['RedLine']};">
                        <div style="font-size:0.72rem; font-weight:700; color:#64748b; text-transform:uppercase;">
                            Rejected alternative</div>
                        <div style="font-size:1.05rem; font-weight:800; color:#1f2937; margin-top:5px;">
                            {rejected_label}</div>
                    </div>
                </div>
                """),
                big_takeaways([
                    "The fleet is the object of engineering; prior decisions cohere into one architecture ledger.",
                    "Launch feasibility is conjunctive across capacity, communication, reliability, security, robustness, carbon, and governance.",
                    "A revision is only defensible when it names what risk was relaxed, monitored, deferred, or redesigned.",
                    "Every production release requires a selected plan, rejected alternative, validation evidence, and named residual risk owner.",
                    "Scale cannot eliminate physical limits; the discipline of MLSys is keeping trade-offs visible and governed.",
                ]),
                mo.Html(f"""
                <div class="mlsysbook-panel" style="border-left: 4px solid #A51C30; margin: 18px 0; background: #FFFDFD;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #A51C30; text-transform: uppercase; margin-bottom: 6px;">LEAD ARCHITECT AUTHORIZATION</div>
                    <h4 style="margin: 0 0 8px 0; color: #0F172A;">Deployment Board Authorization: {v2_17_packet['stakeholder']}</h4>
                    <div style="display: flex; gap: 12px; align-items: center; margin-top: 8px;">
                        <span style="display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 800; font-size: 0.8rem; background: {'#ECFDF5' if passed else '#FEF2F2'}; color: {'#065F46' if passed else '#991B1B'}; border: 1px solid {'#A7F3D0' if passed else '#FECACA'};">
                            {'APPROVED BY DEPLOYMENT BOARD' if passed else 'PROVISIONAL / BLOCKED BY BOARD REVIEW'}
                        </span>
                        <span style="font-size: 0.85rem; color: #475569;">
                            Plan: <code>{selected_label}</code> &middot; Binding: <code>{v2_17_ledger_result['binding']['amount']}</code>
                        </span>
                    </div>
                </div>
                """),
                mo.md("## Download Report"),
                report_export_panel(report),
                mo.Html(f"""
                <div style="border: 1px solid #CBD5E1; border-radius: 8px; padding: 16px 20px; margin-top: 20px; background: #F8FAFC;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
                        Volume II Capstone Complete &middot; Systems at Scale
                    </div>
                    <h4 style="margin: 0 0 6px 0; color: #0F172A; font-size: 1.05rem;">
                        Congratulations! You have completed the Volume II Distributed Systems &amp; Fleet Architecture Curriculum.
                    </h4>
                    <p style="margin: 0; font-size: 0.88rem; color: #334155; line-height: 1.5;">
                        You have mastered the end-to-end discipline of ML systems: sizing compute, provisioning networking fabrics,
                        handling routine faults, orchestrating diverse workloads, serving under tight SLOs, securing private pipelines,
                        bounding carbon intensity, and enforcing sociotechnical fairness.
                    </p>
                </div>
                """),
            ]
        )

    v2_17_tabs = mo.ui.tabs({
        "Part A -- Architecture Ledger": build_part_a(),
        "Part B -- Multi-Constraint Review": build_part_b(),
        "Part C -- Guardrail Revision": build_part_c(),
        "Part D -- Board Authorization": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    v2_17_tabs
    return


@app.cell
def _(COLORS, mo, v2_17_board, v2_17_packet):
    _status_color = COLORS["GreenLine"] if v2_17_board["outcome"] == "APPROVE" else COLORS["OrangeLine"] if v2_17_board["outcome"] == "PROVISIONAL" else COLORS["RedLine"]
    mo.Html(
        f"""
    <div class="lab-hud">
      <span class="hud-label">LAB</span>
      <span class="hud-value">Vol2 &middot; Lab 17</span>
      <span class="hud-label">TRACK</span>
      <span class="hud-value">{v2_17_packet['label']}</span>
      <span class="hud-label">STATUS</span>
      <span class="hud-value" style="color:{_status_color}; font-weight:800;">{v2_17_board['outcome']}</span>
      <span class="hud-label">METRIC</span>
      <span class="hud-value">Fleet Readiness</span>
    </div>
    """
    )
    return


if __name__ == "__main__":
    app.run()
