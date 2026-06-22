import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# -----------------------------------------------------------------------------
# LAB V2-14: THE ROBUSTNESS BUDGET
#
# Chapter invariant: robustness is an amount system. A deployed model buys
# bounded behavior under shift by spending finite stress coverage, retraining,
# monitoring, hardening, and fallback capacity. Under-spend and the failure stays
# silent; over-spend and latency, cost, energy, clean quality, and sustainability
# become the new binding constraints.
#
# Packet modules:
#   Part A - Shift Exposure And Failure Cost
#   Part B - Robustness Budget Allocation
#   Part C - Robustness Tax Frontier
#   Part D - Robustness Policy Gate
#   Synthesis
# -----------------------------------------------------------------------------


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
    from mlsysim.labs.components import DecisionLog
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
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
        DecisionLog,
        LAB_CSS,
        apply_plotly_theme,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
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
    v2_14_metadata = get_lab_metadata("vol2/lab_14_robust_ai.py")
    return (v2_14_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "robotaxi"
    v2_14_track_picker = track_selector(default=_default_track)
    v2_14_track_picker
    return (v2_14_track_picker,)


@app.cell
def _(get_lab_track_variant, get_track_profile, v2_14_track_picker):
    v2_14_track_id = v2_14_track_picker.value
    v2_14_profile = get_track_profile(v2_14_track_id)
    v2_14_variant = get_lab_track_variant("v2_14_robustness_budget", v2_14_profile.track_id)
    return v2_14_profile, v2_14_track_id, v2_14_variant


@app.cell
def _(COLORS, math, mo):
    def v2_14_track_params(track_id, variant):
        defaults = variant.defaults
        base = {
            "iphone": {
                "label": "iPhone",
                "base_bins": [0.42, 0.24, 0.16, 0.11, 0.07],
                "bin_labels": ["Bright", "Indoor", "Low light", "Motion", "Rare context"],
                "default_shift": "covariate_shift",
                "failure_unit": "UX harm points per 10k sessions",
                "failure_value": 14.0,
                "base_failures": 8.0,
                "latency_base_ratio": 0.54,
                "energy_budget": 95.0,
                "energy_base": 42.0,
                "residual_limit": 22.0,
                "coverage_floor": 62.0,
                "fallback_floor": 52.0,
                "regression_limit": 10.0,
                "budget_defaults": (28, 18, 26, 24),
                "budget_needs": {"coverage": 28, "retraining": 18, "monitoring": 26, "fallback": 24},
                "likely_failure": "confident wrong action under lighting or user-context shift",
                "sustainability": "robust local checks raise battery energy before V2-15 carbon accounting.",
                "residual_cases": {
                    "tail_context": "Rare accessibility or lighting context still below validation coverage.",
                    "battery_guard": "Fallback path protects quality but can drain battery during long sessions.",
                    "privacy_gap": "Privacy-safe telemetry may hide the slice that is drifting fastest.",
                },
            },
            "oura_ring": {
                "label": "Oura Ring",
                "base_bins": [0.46, 0.22, 0.15, 0.10, 0.07],
                "bin_labels": ["Good contact", "Sleep", "Exercise", "Loose fit", "Rare physiology"],
                "default_shift": "sensor_shift",
                "failure_unit": "health-risk points per 10k windows",
                "failure_value": 18.0,
                "base_failures": 10.0,
                "latency_base_ratio": 0.38,
                "energy_budget": 32.0,
                "energy_base": 13.0,
                "residual_limit": 20.0,
                "coverage_floor": 64.0,
                "fallback_floor": 58.0,
                "regression_limit": 8.0,
                "budget_defaults": (30, 24, 24, 20),
                "budget_needs": {"coverage": 30, "retraining": 24, "monitoring": 24, "fallback": 20},
                "likely_failure": "sensor-contact drift creates plausible but wrong wellness summaries",
                "sustainability": "always-on robustness checks become duty-cycle and battery-life inputs in V2-15.",
                "residual_cases": {
                    "delayed_labels": "Health labels arrive late, so retraining can chase stale drift.",
                    "sensor_contact": "Loose fit or skin-contact changes are not fully covered by lab validation.",
                    "battery_guard": "Fallback summaries reduce harm but spend the tiny energy budget.",
                },
            },
            "robotaxi": {
                "label": "RoboTaxi",
                "base_bins": [0.50, 0.20, 0.13, 0.10, 0.07],
                "bin_labels": ["Clear day", "Night", "Rain", "Construction", "Snow/rare object"],
                "default_shift": "covariate_shift",
                "failure_unit": "safety-risk points per 10k scenes",
                "failure_value": 70.0,
                "base_failures": 6.0,
                "latency_base_ratio": 0.58,
                "energy_budget": 210.0,
                "energy_base": 104.0,
                "residual_limit": 12.0,
                "coverage_floor": 82.0,
                "fallback_floor": 78.0,
                "regression_limit": 5.0,
                "budget_defaults": (38, 18, 18, 30),
                "budget_needs": {"coverage": 38, "retraining": 18, "monitoring": 18, "fallback": 30},
                "likely_failure": "rare-object or weather miss inside a green perception pipeline",
                "sustainability": "redundant perception and fallback compute become vehicle energy constraints in V2-15.",
                "residual_cases": {
                    "physical_patch": "A physical-world artifact survives digital stress tests.",
                    "weather_tail": "Snow or construction scenes remain under-covered in the replay suite.",
                    "deadline_fallback": "Fallback is correct but consumes the p99 decision deadline.",
                },
            },
            "cloud_fleet": {
                "label": "Cloud Fleet",
                "base_bins": [0.44, 0.23, 0.15, 0.11, 0.07],
                "bin_labels": ["Known tenant", "New tenant", "Long tail", "Abuse pattern", "Model update"],
                "default_shift": "concept_drift",
                "failure_unit": "business-risk points per 10k requests",
                "failure_value": 24.0,
                "base_failures": 16.0,
                "latency_base_ratio": 0.52,
                "energy_budget": 260.0,
                "energy_base": 118.0,
                "residual_limit": 24.0,
                "coverage_floor": 68.0,
                "fallback_floor": 56.0,
                "regression_limit": 9.0,
                "budget_defaults": (24, 28, 34, 18),
                "budget_needs": {"coverage": 24, "retraining": 28, "monitoring": 34, "fallback": 18},
                "likely_failure": "tenant or abuse distribution moves while aggregate SLO stays green",
                "sustainability": "continuous monitors and rollback capacity add datacenter energy and carbon load in V2-15.",
                "residual_cases": {
                    "tenant_tail": "A small tenant cohort drifts without moving global metrics.",
                    "abuse_adaptation": "Attackers adapt to the monitor and stay below alert thresholds.",
                    "rollback_blast": "Rollback contains quality risk but doubles serving load during recovery.",
                },
            },
        }[track_id]
        base["latency_budget_ms"] = float(defaults["latency_budget_ms"])
        base["cost_budget"] = float(defaults["cost_budget"]) * 0.72
        base["quality_floor"] = float(defaults["quality_floor_pct"])
        base["guardrail_floor"] = float(defaults["guardrail_floor_pct"])
        base["base_quality"] = min(98.0, base["quality_floor"] + 6.0)
        base["base_latency_ms"] = base["latency_budget_ms"] * base["latency_base_ratio"]
        base["base_cost"] = base["cost_budget"] * 0.48
        return base

    def v2_14_shift_catalog(track_id):
        track_examples = {
            "iphone": {
                "covariate_shift": "lighting, camera motion, locale, and accessibility context",
                "sensor_shift": "microphone/camera calibration and OS-level preprocessing changes",
                "concept_drift": "user behavior or app workflow changes after launch",
                "adversarial_input": "crafted photo, audio, or prompt-like input inside normal permissions",
                "system_fault": "preprocessing or runtime update silently changes inputs",
            },
            "oura_ring": {
                "covariate_shift": "seasonal physiology, activity mix, and sleep-environment changes",
                "sensor_shift": "loose fit, skin contact, sweat, and firmware sampling changes",
                "concept_drift": "health-label relationship changes as cohorts or habits change",
                "adversarial_input": "spoofed sensor pattern or manipulative repeated behavior",
                "system_fault": "firmware or feature extraction bug corrupts windows",
            },
            "robotaxi": {
                "covariate_shift": "weather, lighting, roadwork, geography, and rare objects",
                "sensor_shift": "camera/lidar occlusion, calibration drift, or dirty sensors",
                "concept_drift": "traffic behavior changes after local policy or road changes",
                "adversarial_input": "physical-world patch, sign manipulation, or staged scene",
                "system_fault": "preprocessing parity bug or late sensor packet",
            },
            "cloud_fleet": {
                "covariate_shift": "tenant mix, geography, language, workload, and traffic seasonality",
                "sensor_shift": "schema, feature logging, retrieval, or context-window changes",
                "concept_drift": "label, intent, abuse, or business process changes over time",
                "adversarial_input": "prompt injection, crafted queries, or abuse patterns",
                "system_fault": "pipeline, model routing, dependency, or rollback bug",
            },
        }[track_id]
        severity = {
            "iphone": (0.085, 1.10, 1.05),
            "oura_ring": (0.090, 1.20, 1.10),
            "robotaxi": (0.110, 1.55, 1.25),
            "cloud_fleet": (0.095, 1.30, 1.15),
        }[track_id]
        base_movement, cost_scale, monitor_scale = severity
        return {
            "covariate_shift": {
                "label": "Environmental / covariate shift",
                "example": track_examples["covariate_shift"],
                "movement": base_movement,
                "cost_scale": cost_scale,
                "monitor_scale": monitor_scale,
                "best_account": "coverage",
            },
            "sensor_shift": {
                "label": "Sensor or feature-channel shift",
                "example": track_examples["sensor_shift"],
                "movement": base_movement * 0.88,
                "cost_scale": cost_scale * 0.90,
                "monitor_scale": monitor_scale * 1.20,
                "best_account": "monitoring",
            },
            "concept_drift": {
                "label": "Concept drift / label relationship change",
                "example": track_examples["concept_drift"],
                "movement": base_movement * 1.05,
                "cost_scale": cost_scale * 1.15,
                "monitor_scale": monitor_scale * 1.05,
                "best_account": "retraining",
            },
            "adversarial_input": {
                "label": "Adversarial or manipulated input",
                "example": track_examples["adversarial_input"],
                "movement": base_movement * 1.18,
                "cost_scale": cost_scale * 1.30,
                "monitor_scale": monitor_scale * 0.90,
                "best_account": "coverage",
            },
            "system_fault": {
                "label": "System or preprocessing fault",
                "example": track_examples["system_fault"],
                "movement": base_movement * 0.78,
                "cost_scale": cost_scale * 1.35,
                "monitor_scale": monitor_scale * 1.35,
                "best_account": "fallback",
            },
        }

    def v2_14_shift_options(track_id):
        return {
            item["label"]: key
            for key, item in v2_14_shift_catalog(track_id).items()
        }

    def v2_14_default_shift_label(track_id, params):
        for label, value in v2_14_shift_options(track_id).items():
            if value == params["default_shift"]:
                return label
        return next(iter(v2_14_shift_options(track_id)))

    def v2_14_psi(base, current):
        return sum((p - q) * math.log(p / q) for p, q in zip(base, current) if p > 0 and q > 0)

    def v2_14_shift_distribution(base_bins, movement):
        movement = max(0.0, min(0.34, movement))
        current = [
            max(0.01, base_bins[0] - movement),
            max(0.01, base_bins[1] - movement * 0.30),
            max(0.01, base_bins[2] + movement * 0.05),
            max(0.01, base_bins[3] + movement * 0.45),
            max(0.01, base_bins[4] + movement * 0.80),
        ]
        total = sum(current)
        return [value / total for value in current]

    def v2_14_psi_tier(psi):
        if psi < 0.10:
            return "negligible", "continue monitoring"
        if psi < 0.20:
            return "minor", "investigate root cause"
        if psi < 0.25:
            return "moderate", "consider retraining"
        return "major", "retrain or fallback"

    def v2_14_shift_exposure(track_id, shift_id, exposure, failure_cost_multiplier, variant):
        params = v2_14_track_params(track_id, variant)
        shift = v2_14_shift_catalog(track_id)[shift_id]
        movement = shift["movement"] * max(0.25, exposure)
        current = v2_14_shift_distribution(params["base_bins"], movement)
        psi = v2_14_psi(params["base_bins"], current)
        tier, action = v2_14_psi_tier(psi)
        failures = params["base_failures"] * shift["cost_scale"] * (1 + psi * 4.0) * max(0.2, exposure) ** 0.85
        expected_loss = failures * params["failure_value"] * failure_cost_multiplier
        risk_score = min(100.0, psi * 145.0 + failures * 0.85 + expected_loss / max(1.0, params["failure_value"]) * 0.35)
        if expected_loss > params["failure_value"] * 50 and action == "continue monitoring":
            action = "investigate root cause"
        if expected_loss > params["failure_value"] * 110 and action != "retrain or fallback":
            action = "consider retraining"
        return {
            "params": params,
            "shift": shift,
            "base": params["base_bins"],
            "current": current,
            "psi": psi,
            "tier": tier,
            "action": action,
            "failures_per_10k": failures,
            "expected_loss": expected_loss,
            "risk_score": risk_score,
            "best_account": shift["best_account"],
            "shift_id": shift_id,
            "exposure": exposure,
            "failure_cost_multiplier": failure_cost_multiplier,
        }

    def v2_14_budget_result(shift_result, coverage, retraining, monitoring, fallback):
        params = shift_result["params"]
        needs = params["budget_needs"]
        allocations = {
            "coverage": float(coverage),
            "retraining": float(retraining),
            "monitoring": float(monitoring),
            "fallback": float(fallback),
        }
        total = sum(allocations.values())
        over_budget = max(0.0, total - 100.0)
        under_budget = max(0.0, 100.0 - total)
        ratios = {key: allocations[key] / max(1.0, needs[key]) for key in allocations}
        underfunded = min(ratios, key=ratios.get)
        best_account = shift_result["best_account"]
        benefits = {
            "coverage": 0.30 * (1 - math.exp(-allocations["coverage"] / max(1.0, needs["coverage"]))),
            "retraining": 0.26 * (1 - math.exp(-allocations["retraining"] / max(1.0, needs["retraining"]))),
            "monitoring": 0.22 * (1 - math.exp(-allocations["monitoring"] / max(1.0, needs["monitoring"]))),
            "fallback": 0.24 * (1 - math.exp(-allocations["fallback"] / max(1.0, needs["fallback"]))),
        }
        benefits[best_account] *= 1.22
        total_benefit = min(0.86, sum(benefits.values()))
        residual_risk = max(0.0, shift_result["risk_score"] * (1.0 - total_benefit) + over_budget * 0.18 + under_budget * 0.05)
        detection_delay = max(0.5, 10.0 / (1.0 + allocations["monitoring"] / 22.0 + allocations["coverage"] / 34.0))
        fallback_coverage = min(96.0, allocations["fallback"] * 1.45 + allocations["monitoring"] * 0.22)
        stress_coverage = min(98.0, allocations["coverage"] * 1.55 + allocations["retraining"] * 0.24)
        binding = "budget overrun" if over_budget > 0 else underfunded
        marginal = {
            key: (needs[key] - allocations[key]) / max(1.0, needs[key])
            for key in allocations
        }
        actual_best = max(marginal, key=marginal.get)
        if best_account in marginal and marginal[best_account] > -0.20:
            actual_best = best_account
        return {
            "allocations": allocations,
            "benefits": benefits,
            "total_budget": total,
            "over_budget": over_budget,
            "under_budget": under_budget,
            "underfunded": underfunded,
            "binding": binding,
            "actual_best": actual_best,
            "residual_risk": residual_risk,
            "detection_delay": detection_delay,
            "fallback_coverage": fallback_coverage,
            "stress_coverage": stress_coverage,
            "total_benefit": total_benefit,
        }

    def v2_14_defense_catalog(track_id):
        robotaxi_latency = 1.18 if track_id == "robotaxi" else 1.0
        tiny_energy = 1.25 if track_id == "oura_ring" else 1.0
        return {
            "input_monitoring": {
                "label": "Input monitor + output guardrail",
                "gain": 18.0,
                "latency_pct": 0.10 * robotaxi_latency,
                "cost_pct": 0.14,
                "energy_pct": 0.12 * tiny_energy,
                "quality_tax": 0.8,
                "regression": 3.0,
                "fit": "monitoring",
            },
            "targeted_retraining": {
                "label": "Targeted stress coverage + retraining",
                "gain": 28.0,
                "latency_pct": 0.08,
                "cost_pct": 0.30,
                "energy_pct": 0.24 * tiny_energy,
                "quality_tax": 1.8,
                "regression": 5.0,
                "fit": "retraining",
            },
            "adversarial_training": {
                "label": "Adversarial / robust training",
                "gain": 34.0,
                "latency_pct": 0.18 * robotaxi_latency,
                "cost_pct": 0.54,
                "energy_pct": 0.46 * tiny_energy,
                "quality_tax": 8.5,
                "regression": 8.0,
                "fit": "coverage",
            },
            "ensemble_uq": {
                "label": "Ensemble or uncertainty fallback",
                "gain": 31.0,
                "latency_pct": 0.32 * robotaxi_latency,
                "cost_pct": 0.46,
                "energy_pct": 0.38 * tiny_energy,
                "quality_tax": 2.6,
                "regression": 4.0,
                "fit": "fallback",
            },
            "fallback_first": {
                "label": "Fallback-first policy",
                "gain": 22.0,
                "latency_pct": 0.22 * robotaxi_latency,
                "cost_pct": 0.24,
                "energy_pct": 0.25 * tiny_energy,
                "quality_tax": 1.2,
                "regression": 2.0,
                "fit": "fallback",
            },
        }

    def v2_14_defense_options(track_id):
        return {
            item["label"]: key
            for key, item in v2_14_defense_catalog(track_id).items()
        }

    def v2_14_defense_result(track_id, defense_id, strength, uq_samples, budget_result, shift_result, variant):
        params = v2_14_track_params(track_id, variant)
        defense = v2_14_defense_catalog(track_id)[defense_id]
        s = max(0.0, min(1.0, strength / 100.0))
        sample_tax = max(0, int(uq_samples) - 1)
        fit_bonus = 1.12 if defense["fit"] == shift_result["best_account"] else 0.96
        gain = defense["gain"] * (1 - math.exp(-2.2 * s)) * fit_bonus + min(8.0, sample_tax * 1.4)
        residual_after = max(0.0, budget_result["residual_risk"] - gain)
        latency_ms = params["base_latency_ms"] * (1 + defense["latency_pct"] * s + 0.055 * sample_tax)
        cost = params["base_cost"] * (1 + defense["cost_pct"] * s + 0.035 * sample_tax)
        energy = params["energy_base"] * (1 + defense["energy_pct"] * s + 0.040 * sample_tax)
        quality = params["base_quality"] - defense["quality_tax"] * (s ** 1.08) - 0.18 * sample_tax
        regression_risk = max(0.0, defense["regression"] * s + max(0.0, params["quality_floor"] - quality) * 2.0)
        robustness_score = max(0.0, min(100.0, 100.0 - residual_after))
        tax_checks = {
            "latency": latency_ms <= params["latency_budget_ms"],
            "cost": cost <= params["cost_budget"],
            "energy": energy <= params["energy_budget"],
            "clean_quality": quality >= params["quality_floor"],
            "regression": regression_risk <= params["regression_limit"],
        }
        if not tax_checks["latency"]:
            binding_tax = "latency"
        elif not tax_checks["clean_quality"]:
            binding_tax = "clean_quality"
        elif not tax_checks["cost"]:
            binding_tax = "cost"
        elif not tax_checks["energy"]:
            binding_tax = "energy"
        elif not tax_checks["regression"]:
            binding_tax = "regression"
        else:
            binding_tax = "none"
        return {
            "defense": defense,
            "strength": strength,
            "uq_samples": uq_samples,
            "gain": gain,
            "residual_after": residual_after,
            "robustness_score": robustness_score,
            "latency_ms": latency_ms,
            "cost": cost,
            "energy": energy,
            "quality": quality,
            "regression_risk": regression_risk,
            "tax_checks": tax_checks,
            "binding_tax": binding_tax,
        }

    def v2_14_policy_catalog():
        return {
            "monitor_only": "Monitor-only guardrail",
            "targeted_budget": "Targeted robustness budget",
            "safety_first": "Safety-first defense-in-depth",
            "adversarial_only": "Adversarial hardening only",
        }

    def v2_14_policy_result(track_id, policy_id, strictness_pct, residual_case, shift_result, budget_result, defense_result, variant):
        params = v2_14_track_params(track_id, variant)
        strictness = max(0.75, strictness_pct / 100.0)
        base = {
            "monitor_only": {
                "coverage": budget_result["stress_coverage"] * 0.55,
                "fallback": budget_result["fallback_coverage"] * 0.55,
                "residual": budget_result["residual_risk"] * 1.12,
                "latency": params["base_latency_ms"] * 1.05,
                "cost": params["base_cost"] * 1.08,
                "energy": params["energy_base"] * 1.06,
                "quality": params["base_quality"],
            },
            "targeted_budget": {
                "coverage": max(budget_result["stress_coverage"], defense_result["robustness_score"] * 0.70),
                "fallback": budget_result["fallback_coverage"],
                "residual": min(budget_result["residual_risk"], defense_result["residual_after"]),
                "latency": defense_result["latency_ms"],
                "cost": defense_result["cost"],
                "energy": defense_result["energy"],
                "quality": defense_result["quality"],
            },
            "safety_first": {
                "coverage": max(88.0, defense_result["robustness_score"] * 0.92),
                "fallback": max(82.0, budget_result["fallback_coverage"]),
                "residual": defense_result["residual_after"] * 0.62,
                "latency": defense_result["latency_ms"] * 1.18,
                "cost": defense_result["cost"] * 1.22,
                "energy": defense_result["energy"] * 1.18,
                "quality": defense_result["quality"] - 0.8,
            },
            "adversarial_only": {
                "coverage": 76.0 if shift_result["shift_id"] == "adversarial_input" else 44.0,
                "fallback": budget_result["fallback_coverage"] * 0.35,
                "residual": defense_result["residual_after"] + (0.0 if shift_result["shift_id"] == "adversarial_input" else 16.0),
                "latency": defense_result["latency_ms"] * 1.05,
                "cost": defense_result["cost"] * 1.12,
                "energy": defense_result["energy"] * 1.08,
                "quality": defense_result["quality"] - 1.5,
            },
        }[policy_id]
        residual_penalty = {
            "tail_context": 4.0,
            "battery_guard": 3.0,
            "privacy_gap": 4.0,
            "delayed_labels": 5.0,
            "sensor_contact": 4.0,
            "physical_patch": 7.0,
            "weather_tail": 6.0,
            "deadline_fallback": 7.0,
            "tenant_tail": 4.0,
            "abuse_adaptation": 5.0,
            "rollback_blast": 5.0,
        }.get(residual_case, 3.0)
        metrics = dict(base)
        metrics["residual"] += residual_penalty
        limits = {
            "residual": params["residual_limit"] / strictness,
            "latency": params["latency_budget_ms"] / max(0.85, strictness * 0.95),
            "cost": params["cost_budget"] / max(0.80, strictness * 0.92),
            "energy": params["energy_budget"] / max(0.85, strictness * 0.95),
            "quality": params["quality_floor"] + max(0.0, strictness - 1.0) * 2.0,
            "coverage": params["coverage_floor"] * strictness,
            "fallback": params["fallback_floor"] * strictness,
        }
        checks = {
            "residual": metrics["residual"] <= limits["residual"],
            "latency": metrics["latency"] <= limits["latency"],
            "cost": metrics["cost"] <= limits["cost"],
            "energy": metrics["energy"] <= limits["energy"],
            "quality": metrics["quality"] >= limits["quality"],
            "coverage": metrics["coverage"] >= limits["coverage"],
            "fallback": metrics["fallback"] >= limits["fallback"],
        }
        if not checks["residual"]:
            binding = "residual risk"
        elif not checks["coverage"]:
            binding = "stress coverage"
        elif not checks["fallback"]:
            binding = "fallback capacity"
        elif not checks["latency"]:
            binding = "latency"
        elif not checks["quality"]:
            binding = "clean quality"
        elif not checks["cost"]:
            binding = "cost"
        elif not checks["energy"]:
            binding = "energy"
        else:
            binding = "none - all guardrails pass"
        rejected = v2_14_rejected_alternative(track_id, policy_id, binding)
        return {
            "policy_id": policy_id,
            "policy_label": v2_14_policy_catalog()[policy_id],
            "metrics": metrics,
            "limits": limits,
            "checks": checks,
            "binding": binding,
            "rejected": rejected,
            "residual_case": residual_case,
            "residual_case_text": params["residual_cases"].get(residual_case, "Residual case not recorded."),
            "sustainability": v2_14_sustainability_implication(track_id, metrics, params),
        }

    def v2_14_rejected_alternative(track_id, policy_id, binding):
        if binding == "none - all guardrails pass":
            defaults = {
                "monitor_only": "Rejected safety-first expansion: current risk does not justify the extra serving tax.",
                "targeted_budget": "Rejected monitor-only policy: it leaves fallback and stress coverage too thin.",
                "safety_first": "Rejected adversarial-only policy: it misses drift and system-fault residuals.",
                "adversarial_only": "Rejected broad defense stack: threat model is narrow enough to avoid the extra tax.",
            }
            return defaults.get(policy_id, "Rejected alternative did not improve the binding amount enough.")
        if binding in ("latency", "energy", "cost"):
            return "Rejected safety-first defense-in-depth: the robustness tax exceeds the operating envelope."
        if binding in ("stress coverage", "fallback capacity", "residual risk"):
            return "Rejected monitor-only baseline: it observes the problem without enough coverage or fallback."
        if binding == "clean quality":
            return "Rejected adversarial-heavy hardening: clean-quality regression violates the release floor."
        return "Rejected alternative: it fails the selected track's guardrail conjunction."

    def v2_14_sustainability_implication(track_id, metrics, params):
        extra_energy = max(0.0, metrics["energy"] - params["energy_base"])
        labels = {
            "iphone": "battery energy per session",
            "oura_ring": "always-on duty-cycle energy",
            "robotaxi": "vehicle compute energy",
            "cloud_fleet": "datacenter energy and carbon",
        }
        return (
            f"V2-15 should carry +{extra_energy:.1f} energy units of robustness overhead into "
            f"{labels.get(track_id, 'energy and carbon')} accounting."
        )

    def v2_14_fmt(value, unit="", precision=1):
        if isinstance(value, str):
            return value
        if abs(value) >= 100:
            text = f"{value:,.0f}"
        else:
            text = f"{value:.{precision}f}"
        return f"{text} {unit}".strip()

    def v2_14_pct(value):
        return f"{value:.1f}%"

    def v2_14_account_label(account):
        return {
            "coverage": "stress coverage",
            "retraining": "retraining",
            "monitoring": "monitoring",
            "fallback": "fallback",
            "budget overrun": "budget overrun",
        }.get(account, str(account).replace("_", " "))

    def v2_14_tax_label(tax):
        return {
            "clean_quality": "clean quality",
            "cost_energy": "cost / energy",
            "latency": "latency",
            "cost": "cost",
            "energy": "energy",
            "regression": "regression risk",
            "none": "none",
        }.get(tax, str(tax).replace("_", " "))

    def v2_14_part_banner(letter, title, duration, why, color):
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
                        line-height:1.55; max-width:780px;">{why}</div>
        </div>
        """)

    def v2_14_stakeholder_card(persona, quote, color, background):
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

    def v2_14_metric_card(label, value, subvalue="", color=None):
        _color = color or COLORS["BlueLine"]
        return mo.Html(f"""
        <div style="padding:16px 18px; border:1px solid {COLORS['Border']}; border-radius:8px;
                    min-width:150px; text-align:center; background:white;">
            <div style="color:{COLORS['TextMuted']}; font-size:0.76rem; font-weight:700;
                        text-transform:uppercase;">{label}</div>
            <div style="font-size:1.65rem; font-weight:800; color:{_color}; font-family:monospace;
                        line-height:1.35;">{value}</div>
            <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">{subvalue}</div>
        </div>
        """)

    def v2_14_reveal_card(title, prediction, actual, detail, kind="info"):
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

    def v2_14_failure_card(active, title, detail, recovery):
        if active:
            return mo.callout(
                mo.md(f"**{title}**  \n{detail}  \n\nRecovery path: {recovery}"),
                kind="danger",
            )
        return mo.callout(
            mo.md(f"**Recovered boundary: {title}**  \nCurrent settings pass. Boundary to watch: {detail}"),
            kind="success",
        )

    def v2_14_math_peek(title, body):
        return mo.accordion({title: mo.md(body)})

    def v2_14_table_html(headers, rows):
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
        v2_14_account_label,
        v2_14_budget_result,
        v2_14_default_shift_label,
        v2_14_defense_catalog,
        v2_14_defense_options,
        v2_14_defense_result,
        v2_14_failure_card,
        v2_14_fmt,
        v2_14_math_peek,
        v2_14_metric_card,
        v2_14_part_banner,
        v2_14_pct,
        v2_14_policy_catalog,
        v2_14_policy_result,
        v2_14_reveal_card,
        v2_14_shift_catalog,
        v2_14_shift_exposure,
        v2_14_shift_options,
        v2_14_stakeholder_card,
        v2_14_table_html,
        v2_14_tax_label,
        v2_14_track_params,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    COLORS,
    LAB_CSS,
    mo,
    track_arc_context,
    track_context,
    v2_14_metadata,
    v2_14_profile,
    v2_14_track_params,
    v2_14_variant,
):
    _params = v2_14_track_params(v2_14_profile.track_id, v2_14_variant)
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
                        Vol 2 &middot; Lab 14 &middot; Robust AI
                    </div>
                    <div style="font-size:2rem; font-weight:800; color:#f1f5f9; line-height:1.15; margin-bottom:10px;">
                        The Robustness Budget
                    </div>
                    <div style="font-size:0.95rem; color:#94a3b8; max-width:710px; line-height:1.6;">
                        {v2_14_variant.workload_summary} You will turn robustness into quantities:
                        shift exposure, stress coverage, hardening spend, residual failure, and policy guardrails.
                    </div>
                </div>
                <div style="display:flex; flex-direction:column; gap:8px; flex-shrink:0;">
                    <span class="badge badge-info">{v2_14_profile.label}</span>
                    <span class="badge badge-info">{v2_14_variant.hardware_ref}</span>
                    <span class="badge badge-info">{v2_14_variant.model_ref}</span>
                    <span class="badge badge-warn">45-55 minutes &middot; 4 Parts + Synthesis</span>
                </div>
            </div>
        </div>
        """),
        track_context(v2_14_profile),
        track_arc_context(v2_14_profile, v2_14_metadata.lab_id),
        mo.callout(
            mo.md(
                f"**Track consequence:** {_params['likely_failure']}. "
                "Every part uses the same concept sequence; the selected track changes thresholds, costs, and residual-risk framing."
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div style="border-left:4px solid {COLORS['BlueLine']};
                background:white; border-radius:0 8px 8px 0;
                padding:20px 28px; margin:8px 0 16px 0;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                    text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
            Learning Objectives
        </div>
        <div style="font-size:0.9rem; color:{COLORS['TextSec']}; line-height:1.7;">
            <div>1. <strong>Measure shift exposure:</strong> connect likely distribution shifts to failure consequence and PSI-style evidence.</div>
            <div>2. <strong>Allocate fixed robustness budget:</strong> split spend across coverage, retraining, monitoring, and fallback.</div>
            <div>3. <strong>Test the tax frontier:</strong> compare robustness improvement against latency, cost, energy, clean quality, and regression risk.</div>
            <div>4. <strong>Write a policy memo:</strong> select guardrails, reject an alternative, and name residual failure.</div>
        </div>
        <div style="border-top:1px solid {COLORS['Border']}; margin:16px -28px 0 -28px; padding:16px 28px 0 28px;">
            <div style="font-size:0.7rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">
                Core Question
            </div>
            <div style="font-size:1.05rem; color:{COLORS['Text']}; font-weight:600;
                        line-height:1.5; font-style:italic;">
                "How much robustness should this system buy, and which residual failure will still remain?"
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md("""
**Recommended Reading** -- Complete before this lab:

- Robust AI: silent failures, environmental shifts, adversarial inputs, and system faults.
- Quantitative drift detection: PSI, KL, KS thresholds, and retraining decision framework.
- Adversarial defenses: robustness tax, certified defenses, uncertainty, guardrails, and fallback.
"""),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(
    mo,
    v2_14_default_shift_label,
    v2_14_shift_options,
    v2_14_track_id,
    v2_14_track_params,
    v2_14_variant,
):
    _params = v2_14_track_params(v2_14_track_id, v2_14_variant)
    partA_shift_prediction = mo.ui.radio(
        options={
            "A) Environmental / covariate shift": "covariate_shift",
            "B) Sensor or feature-channel shift": "sensor_shift",
            "C) Concept drift changes the label relationship": "concept_drift",
            "D) Adversarial input or system fault": "adversarial_or_fault",
        },
        label="Part A prediction: which shift family is most likely to create silent failure for this track?",
    )
    partA_cost_prediction = mo.ui.radio(
        options={
            "A) Latency or SLO miss": "latency",
            "B) Quality, safety, or user-harm cost": "quality",
            "C) Cost or energy overhead": "cost_energy",
            "D) Fallback or blast-radius cost": "fallback",
        },
        label="Part A prediction: which failure cost will matter most?",
    )
    partA_shift = mo.ui.dropdown(
        options=v2_14_shift_options(v2_14_track_id),
        value=v2_14_default_shift_label(v2_14_track_id, _params),
        label="Shift scenario to test",
    )
    partA_exposure = mo.ui.slider(
        start=0.50,
        stop=2.50,
        value=1.00,
        step=0.05,
        label="Stress exposure multiplier",
    )
    partA_failure_cost = mo.ui.slider(
        start=0.50,
        stop=3.00,
        value=1.00,
        step=0.05,
        label="Failure-cost multiplier",
    )
    partA_checkpoint = mo.ui.radio(
        options={
            "A) Keep serving and monitor only": "monitor",
            "B) Investigate root cause before spending training budget": "investigate",
            "C) Retrain immediately": "retrain",
            "D) Trigger fallback first, then diagnose": "fallback",
        },
        label="Checkpoint: what response follows from the shift and failure cost?",
    )
    return (
        partA_checkpoint,
        partA_cost_prediction,
        partA_exposure,
        partA_failure_cost,
        partA_shift,
        partA_shift_prediction,
    )


@app.cell(hide_code=True)
def _(mo, v2_14_track_id, v2_14_track_params, v2_14_variant):
    _params = v2_14_track_params(v2_14_track_id, v2_14_variant)
    _coverage, _retraining, _monitoring, _fallback = _params["budget_defaults"]
    partB_prediction = mo.ui.radio(
        options={
            "A) Stress coverage / replay set": "coverage",
            "B) Retraining and adaptation": "retraining",
            "C) Monitoring and detection": "monitoring",
            "D) Fallback and abstention capacity": "fallback",
        },
        label="Part B prediction: which account will reduce residual failure most for the Part A shift?",
    )
    partB_coverage = mo.ui.slider(start=0, stop=80, value=_coverage, step=2, label="Stress coverage points")
    partB_retraining = mo.ui.slider(start=0, stop=80, value=_retraining, step=2, label="Retraining points")
    partB_monitoring = mo.ui.slider(start=0, stop=80, value=_monitoring, step=2, label="Monitoring points")
    partB_fallback = mo.ui.slider(start=0, stop=80, value=_fallback, step=2, label="Fallback points")
    partB_checkpoint = mo.ui.radio(
        options={
            "A) Fund the weakest required account first": "weakest",
            "B) Maximize retraining because it fixes the model": "retraining",
            "C) Maximize monitoring because visibility is enough": "monitoring",
            "D) Keep all accounts low to avoid tax": "underfund",
        },
        label="Checkpoint: what allocation rule should be carried forward?",
    )
    return (
        partB_checkpoint,
        partB_coverage,
        partB_fallback,
        partB_monitoring,
        partB_prediction,
        partB_retraining,
    )


@app.cell(hide_code=True)
def _(mo, v2_14_defense_options, v2_14_track_id):
    partC_prediction = mo.ui.radio(
        options={
            "A) Latency / deadline": "latency",
            "B) Cost or energy": "cost_energy",
            "C) Clean quality": "clean_quality",
            "D) Regression risk or none": "regression",
        },
        label="Part C prediction: which tax will bind first when robustness hardening increases?",
    )
    partC_defense = mo.ui.dropdown(
        options=v2_14_defense_options(v2_14_track_id),
        value="Targeted stress coverage + retraining",
        label="Hardening strategy",
    )
    partC_strength = mo.ui.slider(start=0, stop=100, value=58, step=2, label="Hardening strength")
    partC_uq_samples = mo.ui.slider(start=1, stop=8, value=2, step=1, label="Uncertainty / ensemble samples")
    partC_checkpoint = mo.ui.radio(
        options={
            "A) Push hardening until residual risk is minimal": "min_residual",
            "B) Use the strongest policy that still passes tax guardrails": "guardrail",
            "C) Prefer monitoring-only because it is cheap": "cheap",
            "D) Defer robustness until incidents occur": "defer",
        },
        label="Checkpoint: what release rule should handle robustness tax?",
    )
    return (
        partC_checkpoint,
        partC_defense,
        partC_prediction,
        partC_strength,
        partC_uq_samples,
    )


@app.cell(hide_code=True)
def _(mo, v2_14_policy_catalog, v2_14_track_id, v2_14_track_params, v2_14_variant):
    _params = v2_14_track_params(v2_14_track_id, v2_14_variant)
    partD_prediction = mo.ui.radio(
        options={
            "A) Monitor-only guardrail": "monitor_only",
            "B) Targeted robustness budget": "targeted_budget",
            "C) Safety-first defense-in-depth": "safety_first",
            "D) Adversarial hardening only": "adversarial_only",
        },
        label="Part D prediction: which policy survives all guardrails?",
    )
    partD_policy = mo.ui.dropdown(
        options={label: key for key, label in v2_14_policy_catalog().items()},
        value="Targeted robustness budget",
        label="Policy candidate to submit",
    )
    partD_strictness = mo.ui.slider(
        start=80,
        stop=125,
        value=100,
        step=5,
        label="Guardrail strictness (%)",
    )
    partD_residual_case = mo.ui.dropdown(
        options={value: key for key, value in _params["residual_cases"].items()},
        value=next(iter(_params["residual_cases"].values())),
        label="Residual failure case to disclose",
    )
    partD_checkpoint = mo.ui.radio(
        options={
            "A) Ship only if every guardrail passes": "all_guardrails",
            "B) Ship if robustness score improved": "score_only",
            "C) Ship if cost is lowest": "cost_only",
            "D) Ship monitor-only and accept residual risk": "accept_risk",
        },
        label="Checkpoint: what policy objective should the review enforce?",
    )
    return (
        partD_checkpoint,
        partD_policy,
        partD_prediction,
        partD_residual_case,
        partD_strictness,
    )


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    partA_checkpoint,
    partA_cost_prediction,
    partA_exposure,
    partA_failure_cost,
    partA_shift,
    partA_shift_prediction,
    partB_checkpoint,
    partB_coverage,
    partB_fallback,
    partB_monitoring,
    partB_prediction,
    partB_retraining,
    partC_checkpoint,
    partC_defense,
    partC_prediction,
    partC_strength,
    partC_uq_samples,
    partD_checkpoint,
    partD_policy,
    partD_prediction,
    partD_residual_case,
    partD_strictness,
    source_trace,
    v2_14_account_label,
    v2_14_budget_result,
    v2_14_defense_catalog,
    v2_14_defense_result,
    v2_14_failure_card,
    v2_14_fmt,
    v2_14_math_peek,
    v2_14_metric_card,
    v2_14_part_banner,
    v2_14_pct,
    v2_14_policy_catalog,
    v2_14_policy_result,
    v2_14_profile,
    v2_14_reveal_card,
    v2_14_shift_catalog,
    v2_14_shift_exposure,
    v2_14_stakeholder_card,
    v2_14_table_html,
    v2_14_tax_label,
    v2_14_track_id,
    v2_14_track_params,
    v2_14_variant,
):
    def build_part_a():
        items = [
            v2_14_part_banner(
                "A",
                "Shift Exposure Has A Cost",
                "10-12 min",
                "Robustness starts by naming the likely distribution shift and the cost of being confidently wrong while the service still looks healthy.",
                COLORS["BlueLine"],
            ),
            v2_14_stakeholder_card(
                v2_14_variant.stakeholder,
                "The dashboard is green. Tell me which shift will break the model first, how much harm it creates, and whether we monitor, investigate, retrain, or fall back.",
                COLORS["BlueLine"],
                COLORS["BlueLL"],
            ),
            mo.md("Commit to the likely shift and failure cost before manipulating the stress exposure."),
            partA_shift_prediction,
            partA_cost_prediction,
        ]
        if partA_shift_prediction.value is None or partA_cost_prediction.value is None:
            items.append(mo.callout(mo.md("Select both Part A predictions to unlock the shift instrument."), kind="warn"))
            return mo.vstack(items)

        result = v2_14_shift_exposure(
            v2_14_track_id,
            partA_shift.value,
            partA_exposure.value,
            partA_failure_cost.value,
            v2_14_variant,
        )
        params = result["params"]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=params["bin_labels"],
            y=[value * 100 for value in result["base"]],
            name="Training baseline",
            marker_color=COLORS["BlueLine"],
            hovertemplate="%{x}: %{y:.1f}% baseline<extra></extra>",
        ))
        _fig.add_trace(go.Bar(
            x=params["bin_labels"],
            y=[value * 100 for value in result["current"]],
            name="Current traffic",
            marker_color=COLORS["OrangeLine"] if result["psi"] < 0.25 else COLORS["RedLine"],
            hovertemplate="%{x}: %{y:.1f}% current<extra></extra>",
        ))
        _fig.add_hline(
            y=0,
            line=dict(color=COLORS["Border"], width=1),
        )
        _fig.update_layout(
            height=330,
            barmode="group",
            xaxis=dict(title="Track-specific cohort or stress bucket"),
            yaxis=dict(title="Share of observations (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=60, l=60, r=20),
        )
        apply_plotly_theme(_fig)

        _rows = []
        for label, base, current in zip(params["bin_labels"], result["base"], result["current"]):
            contribution = (base - current) * __import__("math").log(base / current)
            _rows.append((
                label,
                v2_14_pct(base * 100),
                v2_14_pct(current * 100),
                f"{contribution:.4f}",
            ))
        _actual_prediction = result["shift_id"]
        if _actual_prediction in ("adversarial_input", "system_fault"):
            _actual_prediction = "adversarial_or_fault"
        _prediction_detail = (
            "The selected track's shift profile matches your prediction."
            if partA_shift_prediction.value == _actual_prediction
            else f"The tested scenario is {result['shift']['label']}; the notebook treats it as the dominant current stress."
        )
        _cost_actual = "quality"
        if v2_14_track_id == "cloud_fleet":
            _cost_actual = "fallback" if result["shift_id"] == "system_fault" else "cost_energy"
        if v2_14_track_id == "iphone":
            _cost_actual = "cost_energy" if partA_failure_cost.value > 1.6 else "quality"
        if v2_14_track_id == "robotaxi":
            _cost_actual = "quality"
        _kind = "success" if result["psi"] < 0.20 else "warn" if result["psi"] < 0.25 else "danger"

        items.extend([
            mo.md("### Manipulate the shift and failure cost"),
            mo.hstack([
                mo.vstack([partA_shift, partA_exposure]),
                mo.vstack([partA_failure_cost]),
            ], justify="center", gap=2),
            v2_14_failure_card(
                result["psi"] >= 0.25 or result["expected_loss"] > params["failure_value"] * 110,
                "Major silent-failure exposure",
                f"PSI is {result['psi']:.3f} ({result['tier']}) and expected loss is {result['expected_loss']:.1f} {params['failure_unit']}.",
                "reduce exposure, add stress coverage, raise monitoring sensitivity, or activate fallback",
            ),
            mo.hstack([
                v2_14_metric_card("PSI", f"{result['psi']:.3f}", result["tier"], COLORS["RedLine"] if result["psi"] >= 0.25 else COLORS["OrangeLine"]),
                v2_14_metric_card("Action", result["action"], "chapter threshold", COLORS["BlueLine"]),
                v2_14_metric_card("Failures", f"{result['failures_per_10k']:.1f}", "per 10k units", COLORS["OrangeLine"]),
                v2_14_metric_card("Expected loss", f"{result['expected_loss']:.0f}", params["failure_unit"], COLORS["RedLine"]),
            ], justify="center", gap=1),
            mo.ui.plotly(_fig),
            v2_14_table_html(("Bucket", "Baseline", "Current", "PSI contribution"), _rows),
            v2_14_reveal_card(
                "Shift prediction vs scenario",
                partA_shift_prediction.value.replace("_", " "),
                result["shift"]["label"],
                _prediction_detail,
                "success" if partA_shift_prediction.value == _actual_prediction else "warn",
            ),
            v2_14_reveal_card(
                "Cost prediction vs track consequence",
                v2_14_tax_label(partA_cost_prediction.value),
                v2_14_tax_label(_cost_actual),
                f"For {v2_14_profile.label}, failure consequence is framed as {params['failure_unit']}.",
                "success" if partA_cost_prediction.value == _cost_actual else "warn",
            ),
            v2_14_math_peek(
                "Math Peek / Source Model - PSI and expected failure cost",
                f"""
```
PSI = sum_i (p_i - q_i) * ln(p_i / q_i)
expected_loss = failures_per_10k * consequence_value * failure_cost_multiplier
```

Chapter source: Robust AI quantitative drift detection and PSI threshold bands.
Current PSI = {result['psi']:.3f}; action tier = {result['action']}.
The cohort shares are notebook-local teaching distributions for {v2_14_profile.label},
with track-specific buckets and consequence values.
""",
            ),
            source_trace(
                {
                    "chapter": "book/quarto/contents/vol2/robust_ai/robust_ai.qmd",
                    "formula": "PSI = sum_i (p_i - q_i) * ln(p_i / q_i)",
                    "helper": "v2_14_shift_exposure",
                    "track_id": v2_14_track_id,
                    "scenario_id": v2_14_variant.scenario_id,
                },
                summary="Part A source trace: PSI and failure-cost proxy",
            ),
            partA_checkpoint,
        ])
        if partA_checkpoint.value in ("investigate", "retrain", "fallback"):
            items.append(mo.callout(mo.md("Checkpoint saved: the response treats robustness as a measurable operating condition."), kind="success"))
        elif partA_checkpoint.value is not None:
            items.append(mo.callout(mo.md("Monitor-only is safe only for negligible shift and low consequence. Recheck the PSI tier and expected loss."), kind="warn"))
        return mo.vstack(items)

    def build_part_b():
        shift_result = v2_14_shift_exposure(
            v2_14_track_id,
            partA_shift.value,
            partA_exposure.value,
            partA_failure_cost.value,
            v2_14_variant,
        )
        result = v2_14_budget_result(
            shift_result,
            partB_coverage.value,
            partB_retraining.value,
            partB_monitoring.value,
            partB_fallback.value,
        )
        items = [
            v2_14_part_banner(
                "B",
                "Robustness Budget Is Allocated, Not Added",
                "10-12 min",
                "A fixed robustness budget has to buy different capabilities. Coverage, retraining, monitoring, and fallback are not interchangeable.",
                COLORS["GreenLine"],
            ),
            v2_14_stakeholder_card(
                "Robustness review owner",
                "You have 100 budget points. I need to know which account is underfunded and what residual failure remains.",
                COLORS["GreenLine"],
                COLORS["GreenLL"],
            ),
            partB_prediction,
        ]
        if partB_prediction.value is None:
            items.append(mo.callout(mo.md("Select your budget-account prediction to unlock Part B."), kind="warn"))
            return mo.vstack(items)

        accounts = ["coverage", "retraining", "monitoring", "fallback"]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=[v2_14_account_label(account) for account in accounts],
            y=[result["allocations"][account] for account in accounts],
            marker_color=[
                COLORS["RedLine"] if account == result["underfunded"] else COLORS["GreenLine"]
                for account in accounts
            ],
            text=[f"{result['allocations'][account]:.0f}" for account in accounts],
            textposition="auto",
            hovertemplate="%{x}: %{y:.0f} points<extra></extra>",
        ))
        _fig.add_hline(
            y=25,
            line=dict(color=COLORS["Border"], width=1, dash="dot"),
            annotation_text="even split reference",
        )
        _fig.update_layout(
            height=310,
            xaxis=dict(title="Robustness spending account"),
            yaxis=dict(title="Budget points"),
            margin=dict(t=40, b=70, l=55, r=20),
        )
        apply_plotly_theme(_fig)

        _rows = []
        for account in accounts:
            need = shift_result["params"]["budget_needs"][account]
            status = "under target" if result["allocations"][account] < need else "funded"
            color = COLORS["RedLine"] if status == "under target" else COLORS["GreenLine"]
            _rows.append((
                v2_14_account_label(account),
                f"{result['allocations'][account]:.0f}",
                f"{need:.0f}",
                f"{result['benefits'][account] * 100:.1f}%",
                f"<span style='color:{color}; font-weight:700;'>{status}</span>",
            ))
        _prediction_detail = (
            "That account is the current highest-value account for this shift."
            if partB_prediction.value == result["actual_best"]
            else f"The Part A shift points first to {v2_14_account_label(result['actual_best'])}."
        )
        items.extend([
            mo.md("### Allocate exactly 100 robustness points"),
            mo.hstack([
                mo.vstack([partB_coverage, partB_retraining]),
                mo.vstack([partB_monitoring, partB_fallback]),
            ], justify="center", gap=2),
            v2_14_failure_card(
                result["over_budget"] > 0,
                "Budget overrun",
                f"Current allocation spends {result['total_budget']:.0f} points, {result['over_budget']:.0f} over the 100-point budget.",
                "move points out of a lower-value account until total spend is at or below 100",
            ),
            v2_14_failure_card(
                result["residual_risk"] > shift_result["params"]["residual_limit"],
                "Residual risk remains above track limit",
                f"Residual risk is {result['residual_risk']:.1f} versus limit {shift_result['params']['residual_limit']:.1f}. Underfunded account: {v2_14_account_label(result['underfunded'])}.",
                "rebalance toward the underfunded account or lower the Part A exposure",
            ),
            mo.hstack([
                v2_14_metric_card("Total spend", f"{result['total_budget']:.0f}", "100 point budget", COLORS["RedLine"] if result["over_budget"] else COLORS["GreenLine"]),
                v2_14_metric_card("Residual risk", f"{result['residual_risk']:.1f}", "lower is safer", COLORS["RedLine"] if result["residual_risk"] > shift_result["params"]["residual_limit"] else COLORS["GreenLine"]),
                v2_14_metric_card("Detection delay", f"{result['detection_delay']:.1f}", "weeks proxy", COLORS["OrangeLine"]),
                v2_14_metric_card("Binding", v2_14_account_label(result["binding"]), "budget account", COLORS["RedLine"] if result["binding"] != "none" else COLORS["GreenLine"]),
            ], justify="center", gap=1),
            mo.ui.plotly(_fig),
            v2_14_table_html(("Account", "Spend", "Target need", "Risk reduction", "Status"), _rows),
            v2_14_reveal_card(
                "Budget prediction vs marginal account",
                v2_14_account_label(partB_prediction.value),
                v2_14_account_label(result["actual_best"]),
                _prediction_detail,
                "success" if partB_prediction.value == result["actual_best"] else "warn",
            ),
            v2_14_math_peek(
                "Math Peek / Source Model - fixed budget and diminishing returns",
                f"""
```
total_budget = coverage + retraining + monitoring + fallback
benefit_i = weight_i * (1 - exp(-spend_i / need_i))
residual_risk = shift_risk * (1 - sum_i benefit_i) + overrun_penalty
```

The four accounts map to the chapter's defense-in-depth stack: stress tests,
adaptation, continuous monitoring, and graceful degradation. The selected shift
boosts the account that best matches its failure mode.
""",
            ),
            partB_checkpoint,
        ])
        if partB_checkpoint.value == "weakest":
            items.append(mo.callout(mo.md("Checkpoint saved: protect the weakest required account before increasing the strongest one."), kind="success"))
        elif partB_checkpoint.value is not None:
            items.append(mo.callout(mo.md("A single strong account can still leave a silent residual failure in the underfunded account."), kind="warn"))
        return mo.vstack(items)

    def build_part_c():
        shift_result = v2_14_shift_exposure(
            v2_14_track_id,
            partA_shift.value,
            partA_exposure.value,
            partA_failure_cost.value,
            v2_14_variant,
        )
        budget = v2_14_budget_result(
            shift_result,
            partB_coverage.value,
            partB_retraining.value,
            partB_monitoring.value,
            partB_fallback.value,
        )
        result = v2_14_defense_result(
            v2_14_track_id,
            partC_defense.value,
            partC_strength.value,
            partC_uq_samples.value,
            budget,
            shift_result,
            v2_14_variant,
        )
        items = [
            v2_14_part_banner(
                "C",
                "Robustness Has A Tax Frontier",
                "10-12 min",
                "Hardening improves worst-case behavior, but every defense spends latency, cost, energy, clean quality, or regression risk.",
                COLORS["OrangeLine"],
            ),
            v2_14_stakeholder_card(
                "Release performance lead",
                "Show me the robustness gain and the tax side by side. A defense that breaks our operating envelope is not shippable.",
                COLORS["OrangeLine"],
                COLORS["OrangeLL"],
            ),
            partC_prediction,
        ]
        if partC_prediction.value is None:
            items.append(mo.callout(mo.md("Select your tax prediction to unlock Part C."), kind="warn"))
            return mo.vstack(items)

        params = shift_result["params"]
        candidates = []
        for defense_id in v2_14_defense_catalog(v2_14_track_id):
            candidates.append(v2_14_defense_result(
                v2_14_track_id,
                defense_id,
                partC_strength.value,
                partC_uq_samples.value,
                budget,
                shift_result,
                v2_14_variant,
            ))
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=[candidate["latency_ms"] for candidate in candidates],
            y=[candidate["robustness_score"] for candidate in candidates],
            mode="markers+text",
            text=[candidate["defense"]["label"].split()[0] for candidate in candidates],
            textposition="top center",
            marker=dict(
                size=[max(10, candidate["cost"] / max(1.0, params["base_cost"]) * 10) for candidate in candidates],
                color=[COLORS["RedLine"] if candidate["binding_tax"] != "none" else COLORS["GreenLine"] for candidate in candidates],
                line=dict(color="white", width=1),
            ),
            hovertemplate="Latency %{x:.1f} ms<br>Robustness %{y:.1f}<extra></extra>",
        ))
        _fig.add_vline(
            x=params["latency_budget_ms"],
            line=dict(color=COLORS["RedLine"], width=2, dash="dash"),
            annotation_text="latency guardrail",
        )
        _fig.add_hline(
            y=100.0 - params["residual_limit"],
            line=dict(color=COLORS["GreenLine"], width=2, dash="dot"),
            annotation_text="residual target",
        )
        _fig.update_layout(
            height=330,
            xaxis=dict(title="Latency with defense (ms)"),
            yaxis=dict(title="Robustness score (100 - residual risk)", range=[0, 105]),
            margin=dict(t=45, b=55, l=70, r=20),
        )
        apply_plotly_theme(_fig)

        _rows = []
        for candidate in candidates:
            status = "PASS" if candidate["binding_tax"] == "none" else f"TAX: {v2_14_tax_label(candidate['binding_tax'])}"
            color = COLORS["GreenLine"] if candidate["binding_tax"] == "none" else COLORS["RedLine"]
            _rows.append((
                candidate["defense"]["label"],
                f"+{candidate['gain']:.1f}",
                f"{candidate['latency_ms']:.1f} ms",
                f"{candidate['cost']:.1f}",
                f"{candidate['energy']:.1f}",
                f"{candidate['quality']:.1f}%",
                f"<span style='color:{color}; font-weight:700;'>{status}</span>",
            ))
        _actual_tax = result["binding_tax"]
        if _actual_tax in ("cost", "energy"):
            _actual_tax_group = "cost_energy"
        elif _actual_tax == "none":
            _actual_tax_group = "regression"
        else:
            _actual_tax_group = _actual_tax
        _prediction_detail = (
            "Your tax prediction matches the selected hardening strategy."
            if partC_prediction.value == _actual_tax_group
            else f"The selected strategy currently binds on {v2_14_tax_label(_actual_tax)}."
        )
        items.extend([
            mo.md("### Tune hardening strength and uncertainty sampling"),
            mo.hstack([
                mo.vstack([partC_defense, partC_strength]),
                mo.vstack([partC_uq_samples]),
            ], justify="center", gap=2),
            v2_14_failure_card(
                result["binding_tax"] != "none",
                "Robustness tax violates a guardrail",
                f"Binding tax: {v2_14_tax_label(result['binding_tax'])}. Latency {result['latency_ms']:.1f}/{params['latency_budget_ms']:.1f} ms, quality {result['quality']:.1f}/{params['quality_floor']:.1f}%.",
                "lower hardening strength, reduce UQ samples, or choose a defense whose tax matches the track envelope",
            ),
            mo.hstack([
                v2_14_metric_card("Robustness gain", f"+{result['gain']:.1f}", "risk points removed", COLORS["GreenLine"]),
                v2_14_metric_card("Latency", f"{result['latency_ms']:.1f} ms", f"limit {params['latency_budget_ms']:.1f}", COLORS["RedLine"] if not result["tax_checks"]["latency"] else COLORS["BlueLine"]),
                v2_14_metric_card("Clean quality", f"{result['quality']:.1f}%", f"floor {params['quality_floor']:.1f}%", COLORS["RedLine"] if not result["tax_checks"]["clean_quality"] else COLORS["GreenLine"]),
                v2_14_metric_card("Binding tax", v2_14_tax_label(result["binding_tax"]), "release guardrail", COLORS["GreenLine"] if result["binding_tax"] == "none" else COLORS["RedLine"]),
            ], justify="center", gap=1),
            mo.ui.plotly(_fig),
            v2_14_table_html(("Defense", "Gain", "Latency", "Cost", "Energy", "Quality", "Status"), _rows),
            v2_14_reveal_card(
                "Tax prediction vs selected defense",
                v2_14_tax_label(partC_prediction.value),
                v2_14_tax_label(_actual_tax),
                _prediction_detail,
                "success" if partC_prediction.value == _actual_tax_group else "warn",
            ),
            v2_14_math_peek(
                "Math Peek / Source Model - robustness tax",
                f"""
```
robustness_gain ~= defense_gain * (1 - exp(-2.2 * strength))
tax_checks = latency_ok and cost_ok and energy_ok and clean_quality_ok and regression_ok
robust objective: min_theta max_||delta||<=epsilon L(f_theta(x + delta), y)
```

Chapter source: Robust AI adversarial-defense workflow and robustness-tax example.
The chapter reports a large clean-accuracy tax for strong adversarial robustness;
this lab generalizes the same idea across latency, cost, energy, quality, and
regression-risk guardrails.
""",
            ),
            partC_checkpoint,
        ])
        if partC_checkpoint.value == "guardrail":
            items.append(mo.callout(mo.md("Checkpoint saved: choose the strongest defense that still passes the operating envelope."), kind="success"))
        elif partC_checkpoint.value is not None:
            items.append(mo.callout(mo.md("The frontier matters: maximum robustness and minimum cost are rarely the same feasible point."), kind="warn"))
        return mo.vstack(items)

    def build_part_d():
        shift_result = v2_14_shift_exposure(
            v2_14_track_id,
            partA_shift.value,
            partA_exposure.value,
            partA_failure_cost.value,
            v2_14_variant,
        )
        budget = v2_14_budget_result(
            shift_result,
            partB_coverage.value,
            partB_retraining.value,
            partB_monitoring.value,
            partB_fallback.value,
        )
        defense = v2_14_defense_result(
            v2_14_track_id,
            partC_defense.value,
            partC_strength.value,
            partC_uq_samples.value,
            budget,
            shift_result,
            v2_14_variant,
        )
        result = v2_14_policy_result(
            v2_14_track_id,
            partD_policy.value,
            partD_strictness.value,
            partD_residual_case.value,
            shift_result,
            budget,
            defense,
            v2_14_variant,
        )
        items = [
            v2_14_part_banner(
                "D",
                "A Robustness Policy Has Guardrails And Residual Failure",
                "12-15 min",
                "A shippable robustness policy is not a slogan. It is a threat-model-bound guardrail conjunction with a rejected alternative and a disclosed residual failure.",
                COLORS["RedLine"],
            ),
            v2_14_stakeholder_card(
                "Governance review chair",
                "Submit one policy. It must state guardrails, what you rejected, and which failure case remains after the robustness budget is spent.",
                COLORS["RedLine"],
                COLORS["RedLL"],
            ),
            partD_prediction,
        ]
        if partD_prediction.value is None:
            items.append(mo.callout(mo.md("Select your policy prediction to unlock Part D."), kind="warn"))
            return mo.vstack(items)

        policy_rows = []
        for policy_id in v2_14_policy_catalog():
            candidate = v2_14_policy_result(
                v2_14_track_id,
                policy_id,
                partD_strictness.value,
                partD_residual_case.value,
                shift_result,
                budget,
                defense,
                v2_14_variant,
            )
            passed = all(candidate["checks"].values())
            color = COLORS["GreenLine"] if passed else COLORS["RedLine"]
            status = "PASS" if passed else f"BLOCKED: {candidate['binding']}"
            metrics = candidate["metrics"]
            policy_rows.append((
                candidate["policy_label"],
                f"{metrics['residual']:.1f} / {candidate['limits']['residual']:.1f}",
                f"{metrics['coverage']:.1f}% / {candidate['limits']['coverage']:.1f}%",
                f"{metrics['fallback']:.1f}% / {candidate['limits']['fallback']:.1f}%",
                f"{metrics['latency']:.1f} ms / {candidate['limits']['latency']:.1f}",
                f"{metrics['quality']:.1f}% / {candidate['limits']['quality']:.1f}%",
                f"<span style='color:{color}; font-weight:700;'>{status}</span>",
            ))
        passed = all(result["checks"].values())
        prediction_detail = (
            "The selected policy passes the guardrail conjunction."
            if passed
            else f"The selected policy is blocked by {result['binding']}."
        )
        items.extend([
            mo.md("### Select the policy, strictness, and residual failure"),
            mo.hstack([
                mo.vstack([partD_policy, partD_strictness]),
                mo.vstack([partD_residual_case]),
            ], justify="center", gap=2),
            v2_14_failure_card(
                not passed,
                "Policy gate failed",
                f"Binding amount: {result['binding']}. A robustness memo cannot ship until every guardrail passes or the exception is explicit.",
                "relax strictness with justification, rebalance budget, lower hardening tax, or choose a better-matched policy",
            ),
            mo.hstack([
                v2_14_metric_card("Policy", result["policy_label"], "selected", COLORS["BlueLine"]),
                v2_14_metric_card("Residual", f"{result['metrics']['residual']:.1f}", f"limit {result['limits']['residual']:.1f}", COLORS["GreenLine"] if result["checks"]["residual"] else COLORS["RedLine"]),
                v2_14_metric_card("Coverage", v2_14_pct(result["metrics"]["coverage"]), f"floor {result['limits']['coverage']:.1f}%", COLORS["GreenLine"] if result["checks"]["coverage"] else COLORS["RedLine"]),
                v2_14_metric_card("Binding", result["binding"], "guardrail gate", COLORS["GreenLine"] if passed else COLORS["RedLine"]),
            ], justify="center", gap=1),
            v2_14_table_html(("Policy", "Residual", "Coverage", "Fallback", "Latency", "Quality", "Status"), policy_rows),
            v2_14_reveal_card(
                "Policy prediction vs guardrail result",
                v2_14_policy_catalog().get(partD_prediction.value, partD_prediction.value),
                result["policy_label"],
                prediction_detail,
                "success" if partD_prediction.value == partD_policy.value and passed else "warn",
            ),
            mo.Html(f"""
            <div style="background:white; border:1px solid {COLORS['Border']};
                        border-radius:8px; padding:20px 24px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:800; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px;">
                    Policy Memo Draft
                </div>
                <div style="font-size:0.9rem; color:{COLORS['TextSec']}; line-height:1.7;">
                    <div><strong>Selected policy:</strong> {result['policy_label']}.</div>
                    <div><strong>Binding amount:</strong> {result['binding']}.</div>
                    <div><strong>Rejected alternative:</strong> {result['rejected']}</div>
                    <div><strong>Residual failure case:</strong> {result['residual_case_text']}</div>
                    <div><strong>V2-15 implication:</strong> {result['sustainability']}</div>
                </div>
            </div>
            """),
            v2_14_math_peek(
                "Math Peek / Source Model - policy feasibility",
                f"""
```
deployable = residual_ok and coverage_ok and fallback_ok
             and latency_ok and cost_ok and energy_ok and quality_ok
```

Chapter source: defense-in-depth workflow, fallacies and pitfalls, and summary.
The selected policy is explicitly bounded by a threat model and operating
envelope. It must include a rejected alternative and residual failure case.
""",
            ),
            partD_checkpoint,
        ])
        if partD_checkpoint.value == "all_guardrails":
            items.append(mo.callout(mo.md("Checkpoint saved: the policy is a guardrail conjunction, not a single robustness score."), kind="success"))
        elif partD_checkpoint.value is not None:
            items.append(mo.callout(mo.md("A robustness policy needs every guardrail plus a residual-risk disclosure."), kind="warn"))
        return mo.vstack(items)

    def build_synthesis():
        shift_result = v2_14_shift_exposure(
            v2_14_track_id,
            partA_shift.value,
            partA_exposure.value,
            partA_failure_cost.value,
            v2_14_variant,
        )
        budget = v2_14_budget_result(
            shift_result,
            partB_coverage.value,
            partB_retraining.value,
            partB_monitoring.value,
            partB_fallback.value,
        )
        defense = v2_14_defense_result(
            v2_14_track_id,
            partC_defense.value,
            partC_strength.value,
            partC_uq_samples.value,
            budget,
            shift_result,
            v2_14_variant,
        )
        policy = v2_14_policy_result(
            v2_14_track_id,
            partD_policy.value,
            partD_strictness.value,
            partD_residual_case.value,
            shift_result,
            budget,
            defense,
            v2_14_variant,
        )
        return mo.vstack([
            mo.Html(f"""
            <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                        border-radius:8px; padding:24px 28px; margin:16px 0;">
                <div style="font-size:0.7rem; font-weight:700; color:{COLORS['TextMuted']};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:12px;">
                    Key Takeaways
                </div>
                <div style="font-size:0.92rem; color:{COLORS['Text']}; line-height:1.75;">
                    <div><strong>1. Shift has an amount.</strong> Current PSI is {shift_result['psi']:.3f}, tier {shift_result['tier']}, with {shift_result['expected_loss']:.1f} expected loss units.</div>
                    <div><strong>2. Robustness spend is allocated.</strong> Budget spend is {budget['total_budget']:.0f}/100; underfunded account is {v2_14_account_label(budget['underfunded'])}.</div>
                    <div><strong>3. Hardening has a tax.</strong> {defense['defense']['label']} removes {defense['gain']:.1f} risk points and binds on {v2_14_tax_label(defense['binding_tax'])}.</div>
                    <div><strong>4. Policy is a guardrail conjunction.</strong> Selected policy is {policy['policy_label']} with binding amount {policy['binding']}.</div>
                    <div><strong>5. Residual failure remains.</strong> {policy['residual_case_text']}</div>
                </div>
            </div>
            """),
            mo.Html(f"""
            <div style="background:white; border:1px solid {COLORS['Border']};
                        border-radius:8px; padding:22px 26px; margin:8px 0 16px 0;">
                <div style="font-size:0.7rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px;">
                    Robustness Budget Memo
                </div>
                <div style="font-size:0.9rem; color:{COLORS['TextSec']}; line-height:1.7;">
                    <div><strong>Selected policy:</strong> {policy['policy_label']} for {v2_14_profile.label}.</div>
                    <div><strong>Binding amount:</strong> {policy['binding']}.</div>
                    <div><strong>Residual risk:</strong> {policy['metrics']['residual']:.1f} risk points after disclosed case: {policy['residual_case_text']}</div>
                    <div><strong>Rejected alternative:</strong> {policy['rejected']}</div>
                    <div><strong>V2-15 sustainability implication:</strong> {policy['sustainability']}</div>
                </div>
            </div>
            """),
            mo.accordion({
                "Self-Assessment": mo.md("""
1. Which distribution shift or stress case is most likely for your selected track?
2. Which account was underfunded when you allocated the 100-point robustness budget?
3. Which robustness tax became binding?
4. What residual failure case remains after your selected policy?
5. What energy or carbon implication should V2-15 inherit?
""")
            }),
        ])

    tabs = mo.ui.tabs({
        "Part A -- Shift Exposure": build_part_a(),
        "Part B -- Budget Allocation": build_part_b(),
        "Part C -- Tax Frontier": build_part_c(),
        "Part D -- Policy Gate": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(DecisionLog):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    COLORS,
    decision_input,
    decision_ui,
    ledger,
    mo,
    partA_checkpoint,
    partA_cost_prediction,
    partA_exposure,
    partA_failure_cost,
    partA_shift,
    partA_shift_prediction,
    partB_checkpoint,
    partB_coverage,
    partB_fallback,
    partB_monitoring,
    partB_prediction,
    partB_retraining,
    partC_checkpoint,
    partC_defense,
    partC_prediction,
    partC_strength,
    partC_uq_samples,
    partD_checkpoint,
    partD_policy,
    partD_prediction,
    partD_residual_case,
    partD_strictness,
    v2_14_account_label,
    v2_14_budget_result,
    v2_14_defense_result,
    v2_14_policy_result,
    v2_14_profile,
    v2_14_shift_exposure,
    v2_14_tax_label,
    v2_14_variant,
):
    _shift = v2_14_shift_exposure(
        v2_14_profile.track_id,
        partA_shift.value,
        partA_exposure.value,
        partA_failure_cost.value,
        v2_14_variant,
    )
    _budget = v2_14_budget_result(
        _shift,
        partB_coverage.value,
        partB_retraining.value,
        partB_monitoring.value,
        partB_fallback.value,
    )
    _defense = v2_14_defense_result(
        v2_14_profile.track_id,
        partC_defense.value,
        partC_strength.value,
        partC_uq_samples.value,
        _budget,
        _shift,
        v2_14_variant,
    )
    _policy = v2_14_policy_result(
        v2_14_profile.track_id,
        partD_policy.value,
        partD_strictness.value,
        partD_residual_case.value,
        _shift,
        _budget,
        _defense,
        v2_14_variant,
    )
    _complete = all(
        widget.value is not None
        for widget in (
            partA_shift_prediction,
            partA_cost_prediction,
            partA_checkpoint,
            partB_prediction,
            partB_checkpoint,
            partC_prediction,
            partC_checkpoint,
            partD_prediction,
            partD_checkpoint,
        )
    )
    _ledger_design = {
        "track_id": v2_14_profile.track_id,
        "scenario_id": v2_14_variant.scenario_id,
        "partA_shift_prediction": partA_shift_prediction.value or "no_selection",
        "partA_cost_prediction": partA_cost_prediction.value or "no_selection",
        "partA_selected_shift": partA_shift.value,
        "partA_psi": round(_shift["psi"], 4),
        "partA_action": _shift["action"],
        "partA_expected_loss": round(_shift["expected_loss"], 3),
        "partB_prediction": partB_prediction.value or "no_selection",
        "partB_total_budget": round(_budget["total_budget"], 3),
        "partB_underfunded": _budget["underfunded"],
        "partB_residual_risk": round(_budget["residual_risk"], 3),
        "partC_prediction": partC_prediction.value or "no_selection",
        "partC_defense": _defense["defense"]["label"],
        "partC_strength": partC_strength.value,
        "partC_binding_tax": _defense["binding_tax"],
        "partC_robustness_gain": round(_defense["gain"], 3),
        "partD_prediction": partD_prediction.value or "no_selection",
        "partD_policy": _policy["policy_label"],
        "partD_binding_amount": _policy["binding"],
        "partD_rejected_alternative": _policy["rejected"],
        "partD_residual_failure_case": _policy["residual_case_text"],
        "v2_15_sustainability_implication": _policy["sustainability"],
        "student_justification": str(decision_input.value),
    }
    if _complete:
        ledger.save(chapter=14, design=_ledger_design)

    _status = "SAVED" if _complete else "INCOMPLETE"
    _status_color = COLORS["GreenLine"] if _complete else COLORS["OrangeLine"]
    decision_ui
    mo.Html(f"""
    <div class="lab-hud">
        <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab 14</span></div>
        <div><span class="hud-label">TRACK</span> <span class="hud-value">{v2_14_profile.label}</span></div>
        <div><span class="hud-label">PSI</span> <span class="hud-value">{_shift['psi']:.3f}</span></div>
        <div><span class="hud-label">BUDGET</span> <span class="hud-value">{_budget['total_budget']:.0f}/100</span></div>
        <div><span class="hud-label">TAX</span> <span class="hud-value">{v2_14_tax_label(_defense['binding_tax'])}</span></div>
        <div><span class="hud-label">STATUS</span> <span style="color:{_status_color}; font-family:var(--font-mono);">{_status}</span></div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    decision_input,
    mo,
    partA_checkpoint,
    partA_cost_prediction,
    partA_exposure,
    partA_failure_cost,
    partA_shift,
    partA_shift_prediction,
    partB_checkpoint,
    partB_coverage,
    partB_fallback,
    partB_monitoring,
    partB_prediction,
    partB_retraining,
    partC_checkpoint,
    partC_defense,
    partC_prediction,
    partC_strength,
    partC_uq_samples,
    partD_checkpoint,
    partD_policy,
    partD_prediction,
    partD_residual_case,
    partD_strictness,
    report_export_panel,
    v2_14_account_label,
    v2_14_budget_result,
    v2_14_defense_result,
    v2_14_metadata,
    v2_14_policy_result,
    v2_14_profile,
    v2_14_shift_exposure,
    v2_14_tax_label,
    v2_14_variant,
):
    _shift = v2_14_shift_exposure(
        v2_14_profile.track_id,
        partA_shift.value,
        partA_exposure.value,
        partA_failure_cost.value,
        v2_14_variant,
    )
    _budget = v2_14_budget_result(
        _shift,
        partB_coverage.value,
        partB_retraining.value,
        partB_monitoring.value,
        partB_fallback.value,
    )
    _defense = v2_14_defense_result(
        v2_14_profile.track_id,
        partC_defense.value,
        partC_strength.value,
        partC_uq_samples.value,
        _budget,
        _shift,
        v2_14_variant,
    )
    _policy = v2_14_policy_result(
        v2_14_profile.track_id,
        partD_policy.value,
        partD_strictness.value,
        partD_residual_case.value,
        _shift,
        _budget,
        _defense,
        v2_14_variant,
    )
    _incomplete = []
    for label, widget in (
        ("Part A shift prediction", partA_shift_prediction),
        ("Part A cost prediction", partA_cost_prediction),
        ("Part A checkpoint", partA_checkpoint),
        ("Part B budget prediction", partB_prediction),
        ("Part B checkpoint", partB_checkpoint),
        ("Part C tax prediction", partC_prediction),
        ("Part C checkpoint", partC_checkpoint),
        ("Part D policy prediction", partD_prediction),
        ("Part D checkpoint", partD_checkpoint),
    ):
        if widget.value is None:
            _incomplete.append(label)

    _report = build_lab_report(
        v2_14_metadata,
        track=v2_14_profile.label,
        scenario=v2_14_variant.workload_summary,
        learning_objectives=(
            "Identify likely distribution shifts and failure costs for the selected track.",
            "Allocate a fixed robustness budget across stress coverage, retraining, monitoring, and fallback.",
            "Compare robustness improvement against latency, cost, energy, clean quality, and regression-risk tax.",
            "Select a robustness policy with guardrails, a rejected alternative, and a residual failure case.",
            "Carry the robustness overhead and residual risk into V2-15 sustainability reasoning.",
        ),
        predictions={
            "partA_likely_shift": partA_shift_prediction.value,
            "partA_failure_cost": partA_cost_prediction.value,
            "partB_high_value_account": partB_prediction.value,
            "partC_binding_tax": partC_prediction.value,
            "partD_policy": partD_prediction.value,
        },
        knob_settings={
            "shift_scenario": partA_shift.value,
            "stress_exposure": partA_exposure.value,
            "failure_cost_multiplier": partA_failure_cost.value,
            "coverage_points": partB_coverage.value,
            "retraining_points": partB_retraining.value,
            "monitoring_points": partB_monitoring.value,
            "fallback_points": partB_fallback.value,
            "hardening_strategy": _defense["defense"]["label"],
            "hardening_strength": partC_strength.value,
            "uq_samples": partC_uq_samples.value,
            "policy": _policy["policy_label"],
            "guardrail_strictness_pct": partD_strictness.value,
            "residual_case": _policy["residual_case_text"],
        },
        evidence_summary={
            "psi": round(_shift["psi"], 4),
            "psi_tier": _shift["tier"],
            "recommended_action": _shift["action"],
            "expected_loss": round(_shift["expected_loss"], 3),
            "budget_total": round(_budget["total_budget"], 3),
            "underfunded_account": v2_14_account_label(_budget["underfunded"]),
            "residual_risk_after_budget": round(_budget["residual_risk"], 3),
            "robustness_gain": round(_defense["gain"], 3),
            "binding_tax": v2_14_tax_label(_defense["binding_tax"]),
            "policy_binding_amount": _policy["binding"],
            "rejected_alternative": _policy["rejected"],
        },
        final_decision={
            "selected_policy": _policy["policy_label"],
            "binding_amount": _policy["binding"],
            "residual_risk": round(_policy["metrics"]["residual"], 3),
            "residual_failure_case": _policy["residual_case_text"],
            "rejected_alternative": _policy["rejected"],
            "v2_15_sustainability_implication": _policy["sustainability"],
        },
        big_takeaways=(
            "Robustness is measured against shift and consequence, not average accuracy alone.",
            "Coverage, retraining, monitoring, and fallback are different budget accounts.",
            "Hardening improves worst-case behavior by paying taxes in other system resources.",
            "A policy is deployable only inside a guardrail conjunction.",
            "Residual robustness overhead becomes a sustainability input in the next lab.",
        ),
        reflections={
            "partA_checkpoint": partA_checkpoint.value,
            "partB_checkpoint": partB_checkpoint.value,
            "partC_checkpoint": partC_checkpoint.value,
            "partD_checkpoint": partD_checkpoint.value,
            "student_justification": str(decision_input.value),
        },
        residual_risk=_policy["residual_case_text"],
        source_trace={
            "track_id": v2_14_profile.track_id,
            "scenario_id": v2_14_variant.scenario_id,
            "hardware_ref": v2_14_variant.hardware_ref,
            "model_ref": v2_14_variant.model_ref,
            "chapter_source": "book/quarto/contents/vol2/robust_ai/robust_ai.qmd",
            "notebook_local_helpers": (
                "v2_14_shift_exposure",
                "v2_14_budget_result",
                "v2_14_defense_result",
                "v2_14_policy_result",
            ),
            "formula_refs": ("PSI", "robust minimax objective", "guardrail conjunction"),
            "source_policy": v2_14_profile.source_policy,
        },
        result_snapshot={
            "shift": _shift,
            "budget": _budget,
            "defense": _defense,
            "policy": _policy,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This report is generated locally from the selected track, typed lab variant metadata, "
                "and notebook-local `v2_14_` robustness-budget helpers."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
