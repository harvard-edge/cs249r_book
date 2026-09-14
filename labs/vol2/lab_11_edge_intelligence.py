import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


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
        adaptation_storage,
        big_takeaways,
        build_lab_report,
        edge_device_profile,
        energy_drain,
        federated_communication,
        gated_hypothesis_card,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        instrumentation_console,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        training_memory_breakdown,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        COLORS,
        MathPeek,
        apply_plotly_theme,
        big_takeaways,
        build_lab_report,
        edge_device_profile,
        energy_drain,
        federated_communication,
        gated_hypothesis_card,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        instrumentation_console,
        ledger,
        math,
        mo,
        np,
        report_export_panel,
        resolve_mlsysim_ref,
        training_memory_breakdown,
    )


@app.cell
def _(get_lab_metadata):
    v2_11_metadata = get_lab_metadata("vol2/lab_11_edge_intelligence.py")
    return (v2_11_metadata,)


@app.cell(hide_code=True)
def _(mo):
    v2_11_track_picker = mo.ui.dropdown(
        options={
            "⚡ TinyML Track (ARM Cortex-M55 / ESP32-S3 & Micro-Watt Duty-Cycled Sensing)": "oura_ring",
            "📱 Mobile Track (Apple Silicon / Snapdragon & Thermal Throttling vs Local LoRA)": "iphone",
            "🤖 Edge & Embodied Track (Jetson AGX Orin / DRIVE & Autonomous Perception-Actuation Energy)": "robotaxi",
            "☁️ Cloud Supercomputing Track (Federated Aggregation & Global Edge Fleet Coordination)": "cloud_fleet",
        },
        value="📱 Mobile Track (Apple Silicon / Snapdragon & Thermal Throttling vs Local LoRA)",
        label="Select Course / Industry Track",
    )
    v2_11_track_picker
    return (v2_11_track_picker,)


@app.cell
def _(
    edge_device_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v2_11_track_picker,
):
    v2_11_track_id = v2_11_track_picker.value
    v2_11_profile = get_track_profile(v2_11_track_id)
    v2_11_variant = get_lab_track_variant("v2_11_edge_thermodynamics", v2_11_track_id)
    v2_11_hardware = resolve_mlsysim_ref(v2_11_variant.hardware_ref)
    v2_11_device = edge_device_profile(v2_11_profile, v2_11_variant, v2_11_hardware)
    return v2_11_device, v2_11_profile, v2_11_variant


@app.cell(hide_code=True)
def _(ACADEMIC_LAB_CSS, mo, v2_11_device, v2_11_profile, v2_11_variant):
    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell">
      <div class="mlsysbook-lab-header" style="--mlsysbook-accent: #A51C30;">
        <div class="mlsysbook-meta">
          ML SYSTEMS TEXTBOOK &middot; VOLUME II &middot; CHAPTER 11 &middot; LAB 11
        </div>
        <h1 style="margin: 8px 0 4px 0; color: #0F172A; font-weight: 800; font-size: 1.85rem; letter-spacing: -0.02em;">
          Edge Intelligence: Local Adaptation, Thermodynamics &amp; Federated Scaling
        </h1>
        <p style="margin: 0 0 14px 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">
          Navigate the fundamental thermodynamic and communication limits of edge intelligence: quantify on-device adaptation memory and energy budgets,
          balance federated learning communication against non-IID staleness, extract fresh evidence from intermittent duty-cycled windows, and enforce conjunctive edge deployment guardrails.
        </p>
        <div class="mlsysbook-chip-row" style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;">
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Track:</strong> {v2_11_profile.label}
          </span>
          <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155;">
            <strong>Stakeholder:</strong> {v2_11_variant.stakeholder}
          </span>
          <span class="mlsysbook-chip" style="background: #F8FAFC; color: #475569;">
            <strong>Hardware:</strong> {v2_11_device.hardware_ref}
          </span>
          <span class="mlsysbook-chip" style="background: #F8FAFC; color: #475569;">
            <strong>Memory Budget:</strong> {v2_11_device.available_memory_mb:g} MB
          </span>
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Primary Metric:</strong> {v2_11_variant.primary_metric}
          </span>
          <span class="mlsysbook-chip" style="background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5;">
            <strong>Guardrail:</strong> {v2_11_variant.guardrail_metric}
          </span>
        </div>
      </div>

      <div class="mlsysbook-panel" style="margin-bottom: 20px;">
        <h3 style="margin: 0 0 8px 0; color: #0F172A; font-size: 1.15rem;">
          System Scenario: {v2_11_profile.label} Edge Adaptation Envelope
        </h3>
        <p style="margin: 0 0 12px 0; font-size: 0.92rem; color: #334155; line-height: 1.55;">
          {v2_11_variant.workload_summary} Moving intelligence outward to the edge creates severe thermodynamic and computational tensions. Local fine-tuning amplifies memory and power demands by 4&ndash;12&times; over inference due to gradient storage and optimizer states, federated updates face severe bandwidth and staleness cliffs under heterogeneous client availability, and intermittent duty cycles throttle evidence collection.
        </p>
        <div style="background: #F8FAFC; border-left: 4px solid #006395; padding: 12px 16px; border-radius: 4px; font-size: 0.9rem; color: #1E293B;">
          <strong>The Architectural Invariants of Edge Intelligence:</strong>
          <ul class="mlsysbook-list" style="margin: 8px 0 4px 0;">
            <li><strong>The Thermodynamic Law of Edge Learning:</strong> On-device adaptation is strictly bounded by thermal dissipation and battery capacity (<i>P</i><sub>dynamic</sub> + <i>P</i><sub>leak</sub> &le; <i>P</i><sub>TDP</sub>). Gradient computation easily triggers thermal throttling, collapsing clock frequencies by 40&ndash;50%.</li>
            <li><strong>Federated Communication vs. Computation Law:</strong> Increasing local training epochs (<i>E</i>) reduces global communication rounds (<i>R</i>) but increases client drift under non-IID distributions: <i>T</i><sub>total</sub> = <i>R</i> &middot; (<i>E</i> &middot; <i>T</i><sub>comp</sub> + <i>T</i><sub>comm</sub>).</li>
            <li><strong>Intermittent Evidence Invariant:</strong> Usable edge updates are governed by the intersection of duty-cycle wake windows, charging state, and unmetered network availability: <i>W</i><sub>usable</sub> = <i>D</i><sub>charge</sub> &cap; <i>D</i><sub>wifi</sub> &cap; <i>D</i><sub>idle</sub>.</li>
            <li><strong>Conjunctive Edge Guardrail Bundle:</strong> Viable edge deployment policies must simultaneously satisfy on-device memory footprints, battery drain budgets (&le; 2&ndash;5%/day), differential privacy guarantees (&epsilon; &le; &epsilon;<sub>target</sub>), update freshness thresholds, and model quality floors.</li>
          </ul>
        </div>
      </div>
    </div>
    """)
    mo.vstack([ACADEMIC_LAB_CSS, header_html])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v2_11_profile):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <ul class="mlsysbook-list" style="margin: 0; font-size: 0.9rem; color: {COLORS['TextSec']};">
                <li><strong>Find the on-device boundary:</strong> calculate whether local learning fits the selected track's memory and energy envelope.</li>
                <li><strong>Tune the federated update loop:</strong> compare local epochs, communication, privacy overhead, staleness, and convergence.</li>
                <li><strong>Measure intermittent evidence:</strong> turn duty cycle and connectivity into usable update windows and evidence freshness.</li>
                <li><strong>Write a guardrailed edge policy:</strong> choose a policy that satisfies memory, energy, privacy, update, and quality thresholds.</li>
            </ul>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Edge resource amplification &middot; FedAvg algorithm &middot; client scheduling &middot;
                    production edge guardrails
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~55 min</strong><br/>
                    A: 10 &middot; B: 10 &middot; C: 10 &middot; D: 15 min
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;When intelligence moves to {v2_11_profile.label}, which amount becomes
                binding first: local memory, local energy, federated communication,
                privacy overhead, intermittent evidence, or quality guardrails?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(f"""
    <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-bottom: 20px;">
        <h4 style="margin: 0 0 8px 0; color: #0F172A; font-size: 1.05rem;">
            Recommended Reading &mdash; Complete before this lab:
        </h4>
        <ul class="mlsysbook-list" style="margin: 0; font-size: 0.9rem; color: #334155;">
            <li><strong>Design Constraints:</strong> the 4&ndash;12&times; resource amplification from gradients, optimizer state, activations, bandwidth, and energy.</li>
            <li><strong>Federated Learning Algorithms:</strong> FedAvg, non-IID drift, local epochs, update compression, and privacy-preserving aggregation.</li>
            <li><strong>Federated Systems at Scale:</strong> client scheduling, over-selection, intermittent availability, and stale updates.</li>
            <li><strong>Production Integration:</strong> monitoring, validation, rollback, compliance, and guardrail policies for adaptive edge systems.</li>
        </ul>
    </div>
    """)
    return


@app.cell
def _(energy_drain, federated_communication, training_memory_breakdown):
    def v2_11_track_thresholds(track_id):
        _thresholds = {
            "iphone": {
                "freshness_hours": 24.0,
                "daily_energy_pct": 5.0,
                "quality_floor": 0.90,
                "privacy_epsilon": 8.0,
                "min_updates": 2.0,
            },
            "oura_ring": {
                "freshness_hours": 12.0,
                "daily_energy_pct": 2.0,
                "quality_floor": 0.86,
                "privacy_epsilon": 5.0,
                "min_updates": 1.0,
            },
            "robotaxi": {
                "freshness_hours": 6.0,
                "daily_energy_pct": 3.0,
                "quality_floor": 0.95,
                "privacy_epsilon": 4.0,
                "min_updates": 4.0,
            },
            "cloud_fleet": {
                "freshness_hours": 4.0,
                "daily_energy_pct": 10.0,
                "quality_floor": 0.92,
                "privacy_epsilon": 10.0,
                "min_updates": 6.0,
            },
        }
        return _thresholds.get(track_id, _thresholds["iphone"])

    def v2_11_intermittency_result(track_id, duty_pct, connectivity_pct, windows_per_day, target):
        _success_by_target = {"cpu": 0.72, "gpu": 0.82, "npu": 0.90}
        _threshold = v2_11_track_thresholds(track_id)
        _eligible = windows_per_day * (duty_pct / 100.0) * (connectivity_pct / 100.0)
        _success = _eligible * _success_by_target.get(target, 0.80)
        _missed = max(windows_per_day - _success, 0.0)
        _evidence_age = 24.0 / _success if _success > 0 else float("inf")
        _stale = _evidence_age > _threshold["freshness_hours"]
        _decision = "promote" if (not _stale and _success >= _threshold["min_updates"]) else "defer"
        return {
            "eligible_windows": _eligible,
            "usable_updates": _success,
            "missed_windows": _missed,
            "evidence_age_hours": _evidence_age,
            "freshness_limit_hours": _threshold["freshness_hours"],
            "min_updates": _threshold["min_updates"],
            "stale": _stale,
            "decision": _decision,
        }

    def v2_11_policy_candidates(
        device,
        track_id,
        params_m,
        batch_size,
        energy_target,
        duty_pct,
        connectivity_pct,
        windows_per_day,
        beta,
        local_epochs,
        compression,
        privacy_epsilon,
    ):
        _threshold = v2_11_track_thresholds(track_id)
        _intermittent = v2_11_intermittency_result(
            track_id,
            duty_pct,
            connectivity_pct,
            windows_per_day,
            energy_target,
        )
        _comm = federated_communication(
            device,
            beta=beta,
            local_epochs=local_epochs,
            compression=compression,
        )
        _energy = energy_drain(device, target=energy_target)
        _policy_specs = [
            {
                "policy_id": "full_local",
                "label": "Full local fine-tune",
                "strategy": "full",
                "federated": False,
                "privacy_mode": "local_raw_private",
                "quality": 0.97,
                "energy_multiplier": 1.8,
                "update_age_factor": 1.4,
                "rejected_reason": "full fine-tuning spends the largest local memory and energy amounts",
            },
            {
                "policy_id": "lora_federated",
                "label": "LoRA local + compressed FL",
                "strategy": "lora",
                "federated": True,
                "privacy_mode": "secure_agg_dp",
                "quality": 0.92,
                "energy_multiplier": 1.0,
                "update_age_factor": 1.0,
                "rejected_reason": "compressed federation is usually viable but can still miss freshness",
            },
            {
                "policy_id": "bias_local",
                "label": "Bias-only local fallback",
                "strategy": "bias",
                "federated": False,
                "privacy_mode": "local_only",
                "quality": 0.84,
                "energy_multiplier": 0.45,
                "update_age_factor": 0.8,
                "rejected_reason": "bias-only updates protect resources but may miss the quality floor",
            },
            {
                "policy_id": "cloud_raw",
                "label": "Cloud raw-data retrain",
                "strategy": "inference",
                "federated": False,
                "privacy_mode": "raw_upload",
                "quality": 0.96,
                "energy_multiplier": 0.25,
                "update_age_factor": 0.6,
                "rejected_reason": "raw upload violates the edge privacy premise for sensitive tracks",
            },
        ]

        _rows = []
        for _spec in _policy_specs:
            _strategy = _spec["strategy"] if _spec["strategy"] != "inference" else "bias"
            _memory = training_memory_breakdown(
                params_m=params_m,
                batch_size=batch_size,
                strategy=_strategy,
                available_memory_mb=device.available_memory_mb,
            )
            _memory_mb = _memory.inference_mb if _spec["strategy"] == "inference" else _memory.total_mb
            _daily_energy_pct = _energy.budget_used_pct * _intermittent["usable_updates"] * _spec["energy_multiplier"]
            _evidence_age = _intermittent["evidence_age_hours"] * _spec["update_age_factor"]
            _privacy_ok = (
                _spec["privacy_mode"] in ("local_raw_private", "local_only")
                or (_spec["privacy_mode"] == "secure_agg_dp" and privacy_epsilon <= _threshold["privacy_epsilon"])
            )
            _comm_ok = True
            if _spec["federated"]:
                _comm_ok = _comm.noniid_rounds <= 8 * device.iid_rounds
            _checks = {
                "memory": _memory_mb <= device.available_memory_mb,
                "energy": _daily_energy_pct <= _threshold["daily_energy_pct"],
                "privacy": _privacy_ok,
                "update": _evidence_age <= _threshold["freshness_hours"] and _comm_ok,
                "quality": _spec["quality"] >= _threshold["quality_floor"],
            }
            _order = ["memory", "energy", "privacy", "update", "quality"]
            _failed = [name for name in _order if not _checks[name]]
            _binding = _failed[0] if _failed else min(
                {
                    "memory": device.available_memory_mb - _memory_mb,
                    "energy": _threshold["daily_energy_pct"] - _daily_energy_pct,
                    "privacy": _threshold["privacy_epsilon"] - privacy_epsilon if _spec["federated"] else 99.0,
                    "update": _threshold["freshness_hours"] - _evidence_age,
                    "quality": _spec["quality"] - _threshold["quality_floor"],
                },
                key=lambda name: {
                    "memory": (device.available_memory_mb - _memory_mb) / max(device.available_memory_mb, 1e-9),
                    "energy": (_threshold["daily_energy_pct"] - _daily_energy_pct) / max(_threshold["daily_energy_pct"], 1e-9),
                    "privacy": (_threshold["privacy_epsilon"] - privacy_epsilon) / max(_threshold["privacy_epsilon"], 1e-9) if _spec["federated"] else 99.0,
                    "update": (_threshold["freshness_hours"] - _evidence_age) / max(_threshold["freshness_hours"], 1e-9),
                    "quality": (_spec["quality"] - _threshold["quality_floor"]) / max(_threshold["quality_floor"], 1e-9),
                }[name],
            )
            _rows.append({
                **_spec,
                "memory_mb": _memory_mb,
                "daily_energy_pct": _daily_energy_pct,
                "privacy_epsilon": privacy_epsilon if _spec["federated"] else 0.0,
                "evidence_age_hours": _evidence_age,
                "noniid_rounds": _comm.noniid_rounds if _spec["federated"] else 0.0,
                "total_communication_mb": _comm.total_communication_mb if _spec["federated"] else 0.0,
                "passes": all(_checks.values()),
                "checks": _checks,
                "binding": _binding,
                "failed_guardrails": _failed,
                "threshold": _threshold,
            })
        return {
            "threshold": _threshold,
            "intermittent": _intermittent,
            "communication": _comm,
            "energy": _energy,
            "policies": _rows,
        }

    return (
        v2_11_intermittency_result,
        v2_11_policy_candidates,
        v2_11_track_thresholds,
    )


@app.cell(hide_code=True)
def _(mo, v2_11_device):
    pA_pred = mo.ui.radio(
        options={
            "Memory, because backward pass state multiplies the inference footprint": "memory",
            "Energy, because every local update drains the device budget": "energy",
            "Privacy, because raw data cannot leave the device": "privacy",
            "None; if inference runs locally, learning should fit too": "none",
        },
        value="Memory, because backward pass state multiplies the inference footprint",
        label=(
            f"A {v2_11_device.default_model_params_m:g}M-parameter model is being adapted "
            f"for {v2_11_device.label}. Which on-device amount do you expect to bind first?"
        ),
    )
    return (pA_pred,)


@app.cell(hide_code=True)
def _(mo, v2_11_device):
    pA_params = mo.ui.slider(
        start=0.01, stop=500, value=v2_11_device.default_model_params_m, step=0.01,
        label="Model parameters (millions)",
    )
    pA_batch = mo.ui.slider(
        start=1, stop=32, value=v2_11_device.default_batch_size, step=1,
        label="Batch size",
    )
    pA_strategy = mo.ui.dropdown(
        options={"Full Fine-Tuning": "full", "LoRA (rank-16)": "lora", "Bias-Only": "bias"},
        value="Full Fine-Tuning",
        label="Adaptation strategy",
    )

    pB_pred = mo.ui.radio(
        options={
            "More local epochs will always reduce total communication": "more_epochs",
            "Compression will reduce bytes but can add convergence rounds": "compression_tradeoff",
            "Secure aggregation and DP protect privacy at no systems cost": "free_privacy",
            "Stale and non-IID updates can dominate the federation plan": "staleness",
        },
        value="Compression will reduce bytes but can add convergence rounds",
        label=(
            "The fleet cannot send raw data to the cloud. Which federated-update "
            "trade-off will shape the policy first?"
        ),
    )
    return pA_batch, pA_params, pA_strategy, pB_pred


@app.cell(hide_code=True)
def _(mo, v2_11_device):
    pB_contexts = mo.ui.slider(
        start=1, stop=20, value=5, step=1,
        label="Privacy budget epsilon (lower = stronger DP)",
    )

    pC_pred = mo.ui.number(
        start=0.0, stop=24.0, value=6.0, step=0.1,
        label=(
            f"{v2_11_device.label} has 8 candidate update windows per day. "
            "With intermittent duty cycle and connectivity, how many usable evidence windows survive?"
        ),
    )
    return pB_contexts, pC_pred


@app.cell(hide_code=True)
def _(mo, v2_11_device):
    pC_duty = mo.ui.slider(
        start=1, stop=100, value=35, step=1,
        label="Duty-cycle eligibility (%)",
    )
    pC_connectivity = mo.ui.slider(
        start=1, stop=100, value=60, step=1,
        label="Connectivity availability (%)",
    )
    pC_windows = mo.ui.slider(
        start=1, stop=24, value=8, step=1,
        label="Candidate update windows/day",
    )
    pC_target = mo.ui.radio(
        options={"CPU": "cpu", "GPU": "gpu", v2_11_device.accelerator_label: "npu"},
        value=v2_11_device.accelerator_label,
        label="Local execution target",
        inline=True,
    )

    pD_pred = mo.ui.radio(
        options={
            "Memory will reject the policy": "memory",
            "Energy or duty cycle will reject the policy": "energy",
            "Privacy or update freshness will reject the policy": "privacy_update",
            "Quality will reject the policy": "quality",
        },
        value="Memory will reject the policy",
        label=(
            "Before opening the policy table, which guardrail do you expect to bind first?"
        ),
    )
    return pC_connectivity, pC_duty, pC_target, pC_windows, pD_pred


@app.cell(hide_code=True)
def _(mo):
    pD_beta = mo.ui.slider(
        start=0.1, stop=2.0, value=0.5, step=0.1,
        label="Data heterogeneity (beta) -- lower = more non-IID",
    )
    pD_epochs = mo.ui.slider(
        start=1, stop=20, value=3, step=1,
        label="Local epochs (E)",
    )
    pD_compress = mo.ui.dropdown(
        options={
            "No compression": "none",
            "INT8 quantized (4x reduction)": "int8",
            "INT4 quantized (8x reduction)": "int4",
            "Top-K sparse (10x reduction)": "topk",
        },
        value="INT8 quantized (4x reduction)",
        label="Federated update compression",
    )
    pD_policy = mo.ui.dropdown(
        options={
            "Full local fine-tune": "full_local",
            "LoRA local + compressed FL": "lora_federated",
            "Bias-only local fallback": "bias_local",
            "Cloud raw-data retrain": "cloud_raw",
        },
        value="LoRA local + compressed FL",
        label="Candidate deployment policy",
    )
    return pD_beta, pD_compress, pD_epochs, pD_policy


@app.cell(hide_code=True)
def _(mo):
    v2_11_student_id = mo.ui.text(value="", placeholder="e.g. edge-arch-42", label="Student ID / Reviewer Handle")
    v2_11_next_implication = mo.ui.radio(
        options={
            "Monitor eligibility & duty cycles": "duty_cycle",
            "Track privacy budget composition": "privacy_budget",
            "Automate silent rollback triggers": "rollback",
            "Enforce cohort quality gates": "quality_gate",
        },
        value="Monitor eligibility & duty cycles",
        label="Implication for V2-12 Silent Fleet Operations",
    )
    v2_11_memo_decision = mo.ui.text_area(
        value="",
        placeholder="Write your deployment memo justifying the selected edge policy, binding constraint, and residual risk...",
        label="Deployment Authorization Memo",
    )
    return v2_11_memo_decision, v2_11_next_implication, v2_11_student_id


@app.cell(hide_code=True)
def _(
    COLORS,
    MathPeek,
    apply_plotly_theme,
    big_takeaways,
    build_lab_report,
    energy_drain,
    federated_communication,
    gated_hypothesis_card,
    go,
    instrumentation_console,
    ledger,
    math,
    mo,
    np,
    pA_batch,
    pA_params,
    pA_pred,
    pA_strategy,
    pB_contexts,
    pB_pred,
    pC_connectivity,
    pC_duty,
    pC_pred,
    pC_target,
    pC_windows,
    pD_beta,
    pD_compress,
    pD_epochs,
    pD_policy,
    pD_pred,
    report_export_panel,
    training_memory_breakdown,
    v2_11_device,
    v2_11_intermittency_result,
    v2_11_memo_decision,
    v2_11_metadata,
    v2_11_next_implication,
    v2_11_policy_candidates,
    v2_11_profile,
    v2_11_student_id,
    v2_11_track_thresholds,
    v2_11_variant,
):
    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER -- On-Device Limits Bound Local Learning
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
    <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; {v2_11_variant.stakeholder}</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;{v2_11_variant.objective} The active memory budget is
        {v2_11_device.available_memory_mb:g} MB. Can the training graph fit, or do
        optimizer states and activations make local adaptation impossible?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; {v2_11_variant.stakeholder} &middot; {v2_11_profile.label}</div>
    </div>
    """))

        # Banner
        items.append(mo.Html(f"""
        <div id="part-a" style="margin: 20px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['BlueLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">A</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Part A &middot; 10 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                On-Device Limits Bound Local Learning
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                A model that comfortably runs inference on one deployment target may not learn
                on that same target. Local learning spends active memory, energy, and privacy
                budget at the edge before federation ever begins.
            </div>
        </div>
        """))

        # Prediction
        items.append(gated_hypothesis_card(
            pA_pred,
            title="1. Formulate On-Device Learning Limits Hypothesis",
            subtitle=(
                f"Scenario: A {v2_11_device.default_model_params_m:g}M-parameter model is being adapted "
                f"for {v2_11_device.label}. Predict which on-device resource limit will bind first during training."
            ),
        ))

        if pA_pred.value is None:
            return mo.vstack(items)

        # Controls
        items.append(instrumentation_console(
            mo.hstack([pA_params, pA_batch, pA_strategy], gap="1.5rem"),
            title="Local Training Memory & Compute Controls",
            subtitle=f"Tune model parameters, batch size, and fine-tuning strategy for {v2_11_profile.label}",
        ))

        # Computation
        _params_m = pA_params.value
        _batch = pA_batch.value
        _strategy = pA_strategy.value
        _memory = training_memory_breakdown(
            params_m=_params_m,
            batch_size=_batch,
            strategy=_strategy,
            available_memory_mb=v2_11_device.available_memory_mb,
        )
        _weights_mb = _memory.weights_mb
        _grads_mb = _memory.gradients_mb
        _optim_mb = _memory.optimizer_mb
        _activ_mb = _memory.activations_mb
        _total_mb = _memory.total_mb
        _infer_mb = _memory.inference_mb
        _amplification = _memory.amplification
        _oom = not _memory.fits_memory
        _threshold = v2_11_track_thresholds(v2_11_profile.track_id)
        _cpu_energy = energy_drain(v2_11_device, target="cpu")
        _binding_actual = (
            "memory"
            if _oom else
            "energy"
            if _cpu_energy.budget_used_pct > _threshold["daily_energy_pct"] else
            "none"
        )

        # Stacked bar chart
        _fig = go.Figure()
        _segments = [
            ("Weights", _weights_mb, COLORS["BlueLine"]),
            ("Gradients", _grads_mb, COLORS["OrangeLine"]),
            ("Optimizer State", _optim_mb, "#7c3aed"),
            ("Activations", _activ_mb, COLORS["GreenLine"]),
        ]
        for _name, _val, _color in _segments:
            _fig.add_trace(go.Bar(
                name=_name, x=["Training Memory"], y=[_val],
                marker_color=_color, opacity=0.85,
                text=[f"{_val:.0f} MB"], textposition="inside",
            ))
        # RAM ceiling
        _fig.add_hline(y=v2_11_device.available_memory_mb, line_dash="dash", line_color=COLORS["RedLine"],
                       annotation_text=f"{v2_11_profile.label} active budget: {v2_11_device.available_memory_mb:g} MB",
                       annotation_position="top right")
        # Inference reference
        _fig.add_trace(go.Bar(
            name="Inference Only", x=["Inference"], y=[_infer_mb],
            marker_color=COLORS["BlueLine"], opacity=0.5,
            text=[f"{_infer_mb:.0f} MB"], textposition="inside",
        ))
        _fig.update_layout(
            barmode="stack", height=380,
            yaxis=dict(title="Memory (MB)"),
            margin=dict(l=50, r=20, t=30, b=40),
            legend=dict(orientation="h", y=-0.15, x=0),
        )
        apply_plotly_theme(_fig)

        items.append(mo.as_html(_fig))

        _evidence_rows = [
            ("Inference footprint", f"{_infer_mb:.2f} MB", "Model weights for local inference."),
            ("Training footprint", f"{_total_mb:.2f} MB", "Weights + gradients + optimizer state + activations."),
            ("Active memory budget", f"{v2_11_device.available_memory_mb:g} MB", f"{v2_11_profile.label} track envelope."),
            ("CPU energy/session", f"{_cpu_energy.budget_used_pct:.2f}%", f"Share of {v2_11_device.energy_budget_label} per local session."),
            ("Daily energy guardrail", f"{_threshold['daily_energy_pct']:.1f}%", "Policy limit used again in Part D."),
        ]
        _evidence_html = "".join(
            f"<tr><td>{name}</td><td><strong>{value}</strong></td><td>{note}</td></tr>"
            for name, value, note in _evidence_rows
        )
        items.append(mo.Html(f"""
        <table style="width:100%; border-collapse:collapse; margin:14px 0; font-size:0.88rem;">
            <thead>
                <tr style="background:#f8fafc; color:#475569;">
                    <th style="text-align:left; padding:9px; border:1px solid #e2e8f0;">Amount</th>
                    <th style="text-align:left; padding:9px; border:1px solid #e2e8f0;">Measured value</th>
                    <th style="text-align:left; padding:9px; border:1px solid #e2e8f0;">Why it matters</th>
                </tr>
            </thead>
            <tbody>{_evidence_html}</tbody>
        </table>
        """))

        # OOM banner
        if _oom:
            items.append(mo.callout(mo.md(
                f"**OOM -- Training infeasible on this device.** "
                f"Required: {_total_mb:.2f} MB | Available: {v2_11_device.available_memory_mb:g} MB. "
                f"Switch to LoRA or reduce model size."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"Training fits within the {v2_11_profile.label} active memory budget "
                f"({_total_mb:.2f} MB used, {v2_11_device.available_memory_mb - _total_mb:.2f} MB headroom)."
            ), kind="success"))

        # Cards
        _amp_color = COLORS["RedLine"] if _amplification > 5 else COLORS["OrangeLine"] if _amplification > 2 else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Inference Memory</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_infer_mb:.0f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_amp_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Training Memory</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_amp_color};">{_total_mb:.0f} MB</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_amp_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Amplification</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_amp_color};">{_amplification:.1f}x</div>
            </div>
        </div>"""))

        items.append(mo.md(f"""
    **Training Memory Breakdown** ({_params_m}M params, batch={_batch}, {_strategy})

    ```
    Weights       = {_weights_mb:.2f} MB
    Gradients     = {_grads_mb:.2f} MB
    Optimizer     = {_optim_mb:.2f} MB
    Activations   = {_activ_mb:.2f} MB
    Total         = {_total_mb:.2f} MB ({_amplification:.1f}x inference)
    ```
    *Source: `mlsysbook_labs.training_memory_breakdown`, using `{v2_11_device.hardware_ref}`.*
    """))

        # Reveal
        if pA_pred.value == _binding_actual:
            _msg = (
                f"**Correct binding amount for this setting: {_binding_actual}.** "
                "The device boundary is an amount-system boundary, not a label. "
                "A feasible edge policy must clear this local constraint before it can join the fleet."
            )
            _kind = "success"
        elif pA_pred.value == "none":
            _msg = (
                "**Inference feasibility did not prove learning feasibility.** "
                "Training adds gradients, optimizer state, activation storage, and sustained energy draw. "
                f"Actual binding amount for the current setting: {_binding_actual}."
            )
            _kind = "warn"
        else:
            _msg = (
                f"**Actual binding amount for this setting: {_binding_actual}.** "
                f"Training uses {_total_mb:.2f} MB against a {v2_11_device.available_memory_mb:g} MB "
                f"active memory budget; CPU adaptation uses {_cpu_energy.budget_used_pct:.2f}% "
                f"of the {v2_11_device.energy_budget_label} per session."
            )
            _kind = "warn"

        items.append(mo.callout(mo.md(_msg), kind=_kind))

        items.append(mo.Html(f"""
        <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part A On-Device Feasibility Boundary</h4>
            <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                Carry <code>{_strategy}</code> into the policy review. The current local boundary is <code>{_binding_actual}</code>
                with {_total_mb:.2f} MB of training memory and {_amplification:.1f}x amplification over inference.
            </p>
        </div>
        """))

        items.append(MathPeek(
            r"M_{\text{train}} = M_{\text{weights}} + M_{\text{grad}} + M_{\text{opt}} + M_{\text{act}} \approx 5\text{--}9 \times M_{\text{inf}}",
            {
                "weights storage": f"{_weights_mb:.2f} MB",
                "gradients storage": f"{_grads_mb:.2f} MB",
                "optimizer state (Adam FP32)": f"{_optim_mb:.2f} MB",
                "activations cache": f"{_activ_mb:.2f} MB",
                "total training memory": f"{_total_mb:.2f} MB",
                "amplification factor": f"{_amplification:.1f}x",
                "available memory budget": f"{v2_11_device.available_memory_mb:g} MB",
                "chapter source": "Volume II, Chapter 11: Edge Intelligence & Memory Amplification",
            },
        ))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER -- Federated Updates Trade Communication, Staleness, Privacy, and Convergence
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
    <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; {v2_11_variant.stakeholder}</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;Legal will not let raw data leave {v2_11_profile.label}, but product still wants
        population learning. If we use federated updates, how many rounds and bytes do we spend
        after non-IID drift, compression loss, privacy overhead, and stale updates?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; {v2_11_variant.stakeholder} &middot; {v2_11_profile.label}</div>
    </div>
    """))

        # Banner
        items.append(mo.Html(f"""
        <div id="part-b" style="margin: 20px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['GreenLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">B</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Part B &middot; 15 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                Federated Updates Are Not Free Privacy
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                Federated learning protects raw data locality, but the update loop still spends
                communication rounds, upload bytes, privacy overhead, and freshness budget. More
                local work can reduce communication until non-IID drift makes convergence worse.
            </div>
        </div>
        """))

        # Prediction
        items.append(gated_hypothesis_card(
            pB_pred,
            title="2. Formulate Federated Update Trade-off Hypothesis",
            subtitle=(
                f"Scenario: Fleet cannot send raw data from {v2_11_profile.label} to the cloud. "
                "Predict which federated trade-off (staleness, compression, privacy overhead, or local epochs) shapes the policy first."
            ),
        ))

        if pB_pred.value is None:
            return mo.vstack(items)

        # Controls
        items.append(instrumentation_console(
            mo.hstack([pD_beta, pD_epochs, pD_compress, pB_contexts], gap="1.2rem"),
            title="Federated Learning & Privacy Console",
            subtitle="Adjust data heterogeneity (beta), local epochs (E), compression scheme, and privacy budget epsilon",
        ))

        _beta = pD_beta.value
        _E = pD_epochs.value
        _compress = pD_compress.value
        _epsilon = pB_contexts.value
        _threshold = v2_11_track_thresholds(v2_11_profile.track_id)
        _comm = federated_communication(
            v2_11_device,
            beta=_beta,
            local_epochs=_E,
            compression=_compress,
        )
        _iid_rounds = _comm.iid_rounds
        _noniid_rounds = _comm.noniid_rounds
        _compressed_rounds = _comm.compressed_rounds
        _secure_agg_overhead = 1.10
        _privacy_multiplier = 1.0 + max((_threshold["privacy_epsilon"] / max(_epsilon, 0.1)) - 1.0, 0.0) * 0.20
        _stale_rounds = _compressed_rounds * _privacy_multiplier
        _bytes_per_round = _comm.compressed_bytes_per_round_mb * _secure_agg_overhead
        _total_comm_mb = _stale_rounds * _bytes_per_round
        _freshness_round_limit = 8 * _iid_rounds
        _privacy_ok = _epsilon <= _threshold["privacy_epsilon"]
        _fresh_enough = _stale_rounds <= _freshness_round_limit

        _round_max = int(min(max(_stale_rounds * 1.25, 200), 2500))
        _round_range = np.arange(1, _round_max + 1)
        _iid_acc = np.clip(0.90 * (1 - np.exp(-_round_range / _iid_rounds * 3)), 0, 0.95)
        _noniid_rate = _iid_rounds / max(_noniid_rounds, 1e-9)
        _noniid_acc = np.clip(0.90 * (1 - np.exp(-_round_range / (_iid_rounds / _noniid_rate) * 3)), 0, 0.92)
        _privacy_rate = _iid_rounds / max(_stale_rounds, 1e-9)
        _privacy_acc = np.clip(0.90 * (1 - np.exp(-_round_range / (_iid_rounds / _privacy_rate) * 3)), 0, 0.91)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_round_range, y=_iid_acc, mode="lines",
            name="IID baseline", line=dict(color=COLORS["GreenLine"], width=3),
        ))
        _fig.add_trace(go.Scatter(
            x=_round_range, y=_noniid_acc, mode="lines",
            name=f"Non-IID beta={_beta}", line=dict(color=COLORS["RedLine"], width=3),
        ))
        _fig.add_trace(go.Scatter(
            x=_round_range, y=_privacy_acc, mode="lines",
            name=f"{_comm.compression_label} + privacy overhead",
            line=dict(color=COLORS["BlueLine"], width=2, dash="dash"),
        ))
        _fig.add_vline(
            x=_freshness_round_limit,
            line_dash="dot",
            line_color=COLORS["OrangeLine"],
            annotation_text="freshness budget",
            annotation_position="top right",
        )
        _fig.add_hline(y=0.90, line_dash="dot", line_color="#94a3b8",
                       annotation_text="Target accuracy: 90%")
        _fig.update_layout(
            height=380,
            xaxis=dict(title="Communication Rounds"),
            yaxis=dict(title="Accuracy", range=[0, 1]),
            legend=dict(orientation="h", y=-0.2, font_size=11),
            margin=dict(l=50, r=20, t=30, b=80),
        )
        apply_plotly_theme(_fig)

        _round_ratio = _stale_rounds / max(_iid_rounds, 1e-9)
        _r_color = COLORS["RedLine"] if _round_ratio > 8 else COLORS["OrangeLine"] if _round_ratio > 4 else COLORS["GreenLine"]

        items.append(mo.as_html(_fig))
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">IID Rounds</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_iid_rounds}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_r_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Stale-Aware Rounds</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_r_color};">{_stale_rounds:.0f}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total Upload</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_total_comm_mb/1024:.1f} GB</div>
            </div>
        </div>"""))

        _fed_rows = [
            ("Non-IID rounds", f"{_noniid_rounds:.0f}", "Extra rounds from heterogeneous local data."),
            ("Compressed rounds", f"{_compressed_rounds:.0f}", f"After {_comm.compression_label} update compression."),
            ("Privacy multiplier", f"{_privacy_multiplier:.2f}x", "Stronger DP noise slows useful convergence."),
            ("Bytes per round", f"{_bytes_per_round:.1f} MB", "Compressed update plus secure aggregation overhead."),
            ("Freshness budget", f"{_freshness_round_limit:.0f} rounds", "Track policy limit before updates are too stale."),
        ]
        _fed_table = "".join(
            f"<tr><td>{name}</td><td><strong>{value}</strong></td><td>{note}</td></tr>"
            for name, value, note in _fed_rows
        )
        items.append(mo.Html(f"""
        <table style="width:100%; border-collapse:collapse; margin:14px 0; font-size:0.88rem;">
            <thead>
                <tr style="background:#f8fafc; color:#475569;">
                    <th style="text-align:left; padding:9px; border:1px solid #e2e8f0;">Federated amount</th>
                    <th style="text-align:left; padding:9px; border:1px solid #e2e8f0;">Value</th>
                    <th style="text-align:left; padding:9px; border:1px solid #e2e8f0;">Interpretation</th>
                </tr>
            </thead>
            <tbody>{_fed_table}</tbody>
        </table>
        """))

        if not _privacy_ok:
            items.append(mo.callout(mo.md(
                f"**Privacy guardrail miss.** Epsilon {_epsilon:.1f} exceeds the "
                f"{v2_11_profile.label} limit of {_threshold['privacy_epsilon']:.1f}. "
                "Keeping raw data local is not enough; update privacy must also be bounded."
            ), kind="danger"))
        elif not _fresh_enough:
            items.append(mo.callout(mo.md(
                f"**Freshness miss.** The stale-aware plan needs {_stale_rounds:.0f} rounds, "
                f"above the {_freshness_round_limit:.0f}-round freshness budget. Reduce local epochs, "
                "change compression, or keep the policy local until more representative clients arrive."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"**Federated update is inside the teaching envelope.** The plan spends "
                f"{_stale_rounds:.0f} stale-aware rounds and {_total_comm_mb/1024:.1f} GB of uploads "
                f"while satisfying epsilon <= {_threshold['privacy_epsilon']:.1f}."
            ), kind="success"))

        # Reveal
        _actual_tradeoff = "staleness" if not _fresh_enough else "compression_tradeoff"
        if pB_pred.value == _actual_tradeoff:
            _msg = (
                f"**Correct.** The current federation plan is governed by `{_actual_tradeoff}`. "
                f"It spends {_stale_rounds:.0f} stale-aware rounds, {_bytes_per_round:.1f} MB/round, "
                f"and {_total_comm_mb/1024:.1f} GB total upload."
            )
            _kind = "success"
        elif pB_pred.value == "free_privacy":
            _msg = (
                "**Federation is not free privacy.** Secure aggregation adds protocol overhead, "
                "differential privacy adds noise that can require more rounds, and weak epsilon "
                "values can fail the track policy."
            )
            _kind = "warn"
        else:
            _msg = (
                f"**Actual governing trade-off: `{_actual_tradeoff}`.** Local epochs, compression, "
                "privacy, and stale evidence interact; improving one amount can move the cost to another."
            )
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))

        items.append(mo.Html(f"""
        <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part B Federated Communication Boundary</h4>
            <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                Carry <code>{_comm.compression_label}</code> updates with E={_E}, beta={_beta},
                epsilon={_epsilon:.1f}, and {_stale_rounds:.0f} stale-aware rounds into the final policy review.
            </p>
        </div>
        """))

        items.append(MathPeek(
            r"\theta^{t+1} = \sum_k \frac{n_k}{n}\theta_k^{t+1}, \quad R_{\text{edge}} \approx R_{\text{IID}} \cdot \alpha_{\text{non-IID}} \cdot \alpha_{\text{compress}} \cdot \alpha_{\text{DP}}",
            {
                "IID baseline rounds": f"{_iid_rounds}",
                "non-IID rounds": f"{_noniid_rounds:.0f}",
                "compressed rounds": f"{_compressed_rounds:.0f}",
                "privacy multiplier": f"{_privacy_multiplier:.2f}x",
                "stale-aware rounds": f"{_stale_rounds:.0f}",
                "upload bytes per round": f"{_bytes_per_round:.1f} MB",
                "total upload": f"{_total_comm_mb/1024:.1f} GB",
                "chapter source": "Volume II, Chapter 11: Federated Averaging & Communication Bounds",
            },
        ))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER -- Duty Cycle And Connectivity Make Intermittent Evidence First-Order
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
    <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; {v2_11_variant.stakeholder}</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;The local update ran once in the lab, but production only sees updates when the device
        is awake, connected, eligible, and not stale. How much usable evidence reaches the fleet
        before the next policy decision?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; {v2_11_variant.stakeholder} &middot; {v2_11_profile.label}</div>
    </div>
    """))

        # Banner
        items.append(mo.Html(f"""
        <div id="part-c" style="margin: 20px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['OrangeLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">C</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Part C &middot; 10 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                Intermittent Evidence Becomes First-Order
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                Edge learning is not continuous just because the code can run. Duty cycle,
                connectivity, execution success, and straggler cutoffs decide whether fresh
                local evidence reaches the coordinator.
            </div>
        </div>
        """))

        # Prediction
        items.append(gated_hypothesis_card(
            pC_pred,
            title="3. Formulate Intermittent Evidence Survival Hypothesis",
            subtitle=(
                f"Scenario: {v2_11_device.label} has 8 candidate update windows per day. "
                "With intermittent duty cycle, connectivity availability, and execution targets, predict how many usable evidence windows survive."
            ),
        ))

        if pC_pred.value is None:
            return mo.vstack(items)

        # Controls
        items.append(instrumentation_console(
            mo.hstack([pC_duty, pC_connectivity, pC_windows, pC_target], gap="1.2rem"),
            title="Intermittency & Duty Cycle Console",
            subtitle="Adjust duty-cycle eligibility, network availability, update windows, and local execution target",
        ))

        _target = pC_target.value
        _result = v2_11_intermittency_result(
            v2_11_profile.track_id,
            pC_duty.value,
            pC_connectivity.value,
            pC_windows.value,
            _target,
        )
        _energy = energy_drain(v2_11_device, target=_target)
        _daily_energy_pct = _energy.budget_used_pct * _result["usable_updates"]
        _duty_values = np.arange(1, 101)
        _usable_curve = [
            v2_11_intermittency_result(
                v2_11_profile.track_id,
                int(_duty),
                pC_connectivity.value,
                pC_windows.value,
                _target,
            )["usable_updates"]
            for _duty in _duty_values
        ]
        _age_curve = [
            min(
                v2_11_intermittency_result(
                    v2_11_profile.track_id,
                    int(_duty),
                    pC_connectivity.value,
                    pC_windows.value,
                    _target,
                )["evidence_age_hours"],
                72,
            )
            for _duty in _duty_values
        ]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_duty_values,
            y=_usable_curve,
            mode="lines",
            name="Usable updates/day",
            line=dict(color=COLORS["GreenLine"], width=3),
            yaxis="y",
        ))
        _fig.add_trace(go.Scatter(
            x=_duty_values,
            y=_age_curve,
            mode="lines",
            name="Evidence age (hours)",
            line=dict(color=COLORS["RedLine"], width=3),
            yaxis="y2",
        ))
        _fig.add_vline(
            x=pC_duty.value,
            line_dash="dash",
            line_color=COLORS["BlueLine"],
            annotation_text="current duty cycle",
        )
        _fig.add_hline(
            y=_result["min_updates"],
            line_dash="dot",
            line_color=COLORS["GreenLine"],
            annotation_text="min usable updates",
        )
        _fig.update_layout(
            height=380,
            xaxis=dict(title="Duty-cycle eligibility (%)"),
            yaxis=dict(title="Usable updates/day"),
            yaxis2=dict(title="Evidence age (hours)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=50, r=50, t=30, b=80),
        )
        apply_plotly_theme(_fig)

        items.append(mo.as_html(_fig))

        _age = _result["evidence_age_hours"]
        _age_display = "infinite" if not math.isfinite(_age) else f"{_age:.1f} h"
        _age_color = COLORS["RedLine"] if _result["stale"] else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Eligible Windows</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_result['eligible_windows']:.2f}/day</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Usable Updates</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_result['usable_updates']:.2f}/day</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_age_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Evidence Age</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_age_color};">{_age_display}</div>
            </div>
        </div>"""))

        _rows = [
            ("Candidate windows/day", f"{pC_windows.value}", "How often the scheduler could attempt an update."),
            ("Duty-cycle eligibility", f"{pC_duty.value}%", "Fraction of windows where the device may spend energy."),
            ("Connectivity availability", f"{pC_connectivity.value}%", "Fraction of windows with usable network state."),
            ("Missed windows/day", f"{_result['missed_windows']:.2f}", "Windows lost to sleep, link loss, or execution failure."),
            ("Daily local energy", f"{_daily_energy_pct:.2f}%", f"Share of {v2_11_device.energy_budget_label} spent by usable updates."),
        ]
        _table = "".join(
            f"<tr><td>{name}</td><td><strong>{value}</strong></td><td>{note}</td></tr>"
            for name, value, note in _rows
        )
        items.append(mo.Html(f"""
        <table style="width:100%; border-collapse:collapse; margin:14px 0; font-size:0.88rem;">
            <thead>
                <tr style="background:#f8fafc; color:#475569;">
                    <th style="text-align:left; padding:9px; border:1px solid #e2e8f0;">Intermittent amount</th>
                    <th style="text-align:left; padding:9px; border:1px solid #e2e8f0;">Value</th>
                    <th style="text-align:left; padding:9px; border:1px solid #e2e8f0;">Consequence</th>
                </tr>
            </thead>
            <tbody>{_table}</tbody>
        </table>
        """))

        _prediction_error = abs((pC_pred.value or 0) - _result["usable_updates"])
        if _result["stale"]:
            items.append(mo.callout(mo.md(
                f"**Stale evidence failure.** Evidence age is {_age_display}, above the "
                f"{_result['freshness_limit_hours']:.1f} h freshness limit. The policy should defer "
                "promotion or improve eligibility before trusting this evidence."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"**Fresh enough.** Evidence age is {_age_display} against a "
                f"{_result['freshness_limit_hours']:.1f} h limit. Your prediction missed by "
                f"{_prediction_error:.2f} usable update windows/day."
            ), kind="success"))

        items.append(mo.Html(f"""
        <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part C Evidence Readiness Decision</h4>
            <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                Decision: <code>{_result['decision']}</code> with {_result['usable_updates']:.2f} usable updates/day,
                {_age_display} evidence age, and {_daily_energy_pct:.2f}% daily {v2_11_device.energy_budget_label} use.
            </p>
        </div>
        """))

        items.append(MathPeek(
            r"U = W \cdot d \cdot c \cdot s, \quad A_{\text{evidence}} = \frac{24}{U}",
            {
                "candidate windows (W)": f"{pC_windows.value}/day",
                "duty-cycle eligibility (d)": f"{pC_duty.value}%",
                "connectivity availability (c)": f"{pC_connectivity.value}%",
                "execution target": str(_target),
                "usable updates/day (U)": f"{_result['usable_updates']:.2f}",
                "evidence age (A)": str(_age_display),
                "daily energy used": f"{_daily_energy_pct:.2f}%",
                "chapter source": "Volume II, Chapter 11: Intermittent Evidence & Client Scheduling",
            },
        ))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER -- Edge Deployment Policy Is A Guardrail Bundle
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
    <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; {v2_11_variant.stakeholder}</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;We have local feasibility, a federated update plan, and intermittent evidence. Now pick one
        deployment policy we can defend: it must satisfy memory, energy, privacy, update freshness,
        and quality guardrails at the same time.&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; {v2_11_variant.stakeholder} &middot; {v2_11_profile.label}</div>
    </div>
    """))

        # Banner
        items.append(mo.Html(f"""
        <div id="part-d" style="margin: 20px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: #7c3aed; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">D</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Part D &middot; 15 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                Edge Deployment Policy Is A Guardrail Bundle
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                A policy is deployable only if every guardrail passes together. A high-quality
                policy can still fail memory, energy, privacy, update freshness, or operational
                rollback requirements.
            </div>
        </div>
        """))

        # Prediction
        items.append(gated_hypothesis_card(
            pD_pred,
            title="4. Formulate Edge Deployment Guardrail Hypothesis",
            subtitle=(
                f"Scenario: Evaluating deployment candidate policies for {v2_11_profile.label}. "
                "Predict which guardrail (memory, energy, privacy/freshness, or quality) will reject the candidate policy first."
            ),
        ))

        if pD_pred.value is None:
            return mo.vstack(items)

        # Controls
        items.append(instrumentation_console(
            mo.hstack([pD_policy, pA_strategy, pC_target], gap="1.2rem"),
            title="Deployment Policy Selection Console",
            subtitle="Select candidate deployment policy, local adaptation strategy, and hardware execution target",
        ))

        _policy_packet = v2_11_policy_candidates(
            v2_11_device,
            v2_11_profile.track_id,
            pA_params.value,
            pA_batch.value,
            pC_target.value,
            pC_duty.value,
            pC_connectivity.value,
            pC_windows.value,
            pD_beta.value,
            pD_epochs.value,
            pD_compress.value,
            pB_contexts.value,
        )
        _policies = _policy_packet["policies"]
        _selected = next(
            (_policy for _policy in _policies if _policy["policy_id"] == pD_policy.value),
            _policies[1],
        )
        _passing = [_policy for _policy in _policies if _policy["passes"]]
        _recommended = _passing[0] if _passing else min(_policies, key=lambda policy: len(policy["failed_guardrails"]))
        _rejected = next(
            (_policy for _policy in _policies if _policy["policy_id"] != _selected["policy_id"] and not _policy["passes"]),
            _policies[0],
        )

        _guardrail_names = ["memory", "energy", "privacy", "update", "quality"]
        _chart = go.Figure()
        for _policy in _policies:
            _chart.add_trace(go.Bar(
                x=_guardrail_names,
                y=[1 if _policy["checks"][_name] else 0 for _name in _guardrail_names],
                name=_policy["label"],
            ))
        _chart.update_layout(
            barmode="group",
            height=340,
            yaxis=dict(title="Guardrail pass", tickvals=[0, 1], ticktext=["fail", "pass"], range=[0, 1.2]),
            xaxis=dict(title="Guardrail"),
            legend=dict(orientation="h", y=-0.25, font_size=10),
            margin=dict(l=50, r=20, t=30, b=90),
        )
        apply_plotly_theme(_chart)
        items.append(mo.as_html(_chart))

        def _status_cell(_ok):
            return "<strong style='color:#16a34a;'>pass</strong>" if _ok else "<strong style='color:#dc2626;'>fail</strong>"

        _rows = []
        for _policy in _policies:
            _rows.append(
                "<tr>"
                f"<td><strong>{_policy['label']}</strong></td>"
                f"<td>{_policy['memory_mb']:.1f} MB / {v2_11_device.available_memory_mb:g} MB<br/>{_status_cell(_policy['checks']['memory'])}</td>"
                f"<td>{_policy['daily_energy_pct']:.2f}% / {_policy['threshold']['daily_energy_pct']:.1f}%<br/>{_status_cell(_policy['checks']['energy'])}</td>"
                f"<td>{_policy['privacy_mode']}<br/>{_status_cell(_policy['checks']['privacy'])}</td>"
                f"<td>{_policy['evidence_age_hours']:.1f} h / {_policy['threshold']['freshness_hours']:.1f} h<br/>{_status_cell(_policy['checks']['update'])}</td>"
                f"<td>{_policy['quality']:.2f} / {_policy['threshold']['quality_floor']:.2f}<br/>{_status_cell(_policy['checks']['quality'])}</td>"
                f"<td>{_policy['binding']}</td>"
                "</tr>"
            )
        _table = "".join(_rows)
        items.append(mo.Html(f"""
        <table style="width:100%; border-collapse:collapse; margin:14px 0; font-size:0.82rem;">
            <thead>
                <tr style="background:#f8fafc; color:#475569;">
                    <th style="text-align:left; padding:8px; border:1px solid #e2e8f0;">Policy</th>
                    <th style="text-align:left; padding:8px; border:1px solid #e2e8f0;">Memory</th>
                    <th style="text-align:left; padding:8px; border:1px solid #e2e8f0;">Energy/day</th>
                    <th style="text-align:left; padding:8px; border:1px solid #e2e8f0;">Privacy</th>
                    <th style="text-align:left; padding:8px; border:1px solid #e2e8f0;">Update</th>
                    <th style="text-align:left; padding:8px; border:1px solid #e2e8f0;">Quality</th>
                    <th style="text-align:left; padding:8px; border:1px solid #e2e8f0;">Binding</th>
                </tr>
            </thead>
            <tbody>{_table}</tbody>
        </table>
        """))

        _selected_status = "passes" if _selected["passes"] else "fails"
        _binding_for_prediction = "privacy_update" if _selected["binding"] in ("privacy", "update") else _selected["binding"]
        if pD_pred.value == _binding_for_prediction:
            _msg = (
                f"**Correct.** `{_selected['label']}` {_selected_status} with "
                f"`{_selected['binding']}` as the binding guardrail."
            )
            _kind = "success"
        else:
            _msg = (
                f"**Actual binding guardrail: `{_selected['binding']}`.** "
                f"`{_selected['label']}` {_selected_status}; the recommended policy is "
                f"`{_recommended['label']}`."
            )
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))

        if _selected["passes"]:
            items.append(mo.callout(mo.md(
                f"**Launchable teaching policy.** Select `{_selected['label']}` and reject "
                f"`{_rejected['label']}` because {_rejected['rejected_reason']}."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Policy not launchable yet.** `{_selected['label']}` fails "
                f"{', '.join(_selected['failed_guardrails'])}. Use `{_recommended['label']}` "
                "or revise earlier controls until all guardrails pass."
            ), kind="danger"))

        items.append(mo.Html(f"""
        <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-top: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT DECISION</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Part D Guardrail Authorization Decision</h4>
            <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #475569;">
                Memo decision: <code>{_recommended['label']}</code>; binding edge amount: <code>{_recommended['binding']}</code>;
                rejected alternative: <code>{_rejected['label']}</code>; V2-12 implication: monitor eligibility, rollback, privacy budget, and stale evidence as fleet operations signals.
            </p>
        </div>
        """))

        items.append(MathPeek(
            r"\text{launchable} = (M \le M_{\max}) \land (E_{\text{day}} \le E_{\max}) \land (\varepsilon \le \varepsilon_{\max}) \land (A_{\text{evidence}} \le A_{\max}) \land (Q \ge Q_{\min})",
            {
                "memory check": f"{_selected['memory_mb']:.1f} MB <= {v2_11_device.available_memory_mb:g} MB",
                "energy check": f"{_selected['daily_energy_pct']:.2f}% <= {_selected['threshold']['daily_energy_pct']:.1f}%",
                "privacy mode": str(_selected['privacy_mode']),
                "evidence age": f"{_selected['evidence_age_hours']:.1f} h <= {_selected['threshold']['freshness_hours']:.1f} h",
                "quality floor": f"{_selected['quality']:.2f} >= {_selected['threshold']['quality_floor']:.2f}",
                "binding guardrail": str(_selected['binding']),
                "recommended policy": str(_recommended['label']),
                "chapter source": "Volume II, Chapter 11: Production Integration & Guardrail Bundles",
            },
        ))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        items = []
        _policy_packet = v2_11_policy_candidates(
            v2_11_device,
            v2_11_profile.track_id,
            pA_params.value,
            pA_batch.value,
            pC_target.value,
            pC_duty.value,
            pC_connectivity.value,
            pC_windows.value,
            pD_beta.value,
            pD_epochs.value,
            pD_compress.value,
            pB_contexts.value,
        )
        _policies = _policy_packet["policies"]
        _selected = next(
            (_policy for _policy in _policies if _policy["policy_id"] == pD_policy.value),
            _policies[1],
        )
        _passing = [_policy for _policy in _policies if _policy["passes"]]
        _recommended = _passing[0] if _passing else min(_policies, key=lambda policy: len(policy["failed_guardrails"]))
        _rejected = next(
            (_policy for _policy in _policies if _policy["policy_id"] != _recommended["policy_id"] and not _policy["passes"]),
            _policies[0],
        )

        _incomplete = []
        if pA_pred.value is None:
            _incomplete.append("Part A prediction")
        if pB_pred.value is None:
            _incomplete.append("Part B prediction")
        if pC_pred.value is None:
            _incomplete.append("Part C prediction")
        if pD_pred.value is None:
            _incomplete.append("Part D prediction")

        _implication_text = {
            "duty_cycle": "V2-12 must monitor real-world client eligibility and wake windows to prevent fleet stragglers.",
            "privacy_budget": "V2-12 privacy accounting must track cumulative DP composition across federated rounds.",
            "rollback": "V2-12 silent fleet operations must implement automated rollbacks when local accuracy degrades.",
            "quality_gate": "V2-12 admission control must enforce cohort-level validation gates before model updates merge.",
        }.get(v2_11_next_implication.value, "V2-12 implication not selected yet.")

        _decision_text = v2_11_memo_decision.value or (
            f"For {v2_11_profile.label}, recommend deployment policy {_recommended['label']} "
            f"with binding constraint {_recommended['binding']}; formally reject {_rejected['label']} "
            f"due to {_rejected['rejected_reason']}. "
            f"V2-12 downstream implication: {_implication_text}"
        )

        _memory = training_memory_breakdown(
            params_m=pA_params.value,
            batch_size=pA_batch.value,
            strategy=pA_strategy.value,
            available_memory_mb=v2_11_device.available_memory_mb,
        )
        _energy = energy_drain(v2_11_device, target=pC_target.value)
        _comm = federated_communication(
            v2_11_device,
            beta=pD_beta.value,
            local_epochs=pD_epochs.value,
            compression=pD_compress.value,
        )
        _intermittent = v2_11_intermittency_result(
            v2_11_profile.track_id,
            pC_duty.value,
            pC_connectivity.value,
            pC_windows.value,
            pC_target.value,
        )

        ledger_design = {
            "chapter": "v2_11",
            "track_id": v2_11_profile.track_id,
            "scenario_id": v2_11_variant.scenario_id,
            "hardware_ref": v2_11_device.hardware_ref,
            "partA_local_boundary_prediction": pA_pred.value,
            "partA_adaptation_strategy": pA_strategy.value,
            "partB_federation_prediction": pB_pred.value,
            "partB_privacy_epsilon": pB_contexts.value,
            "partB_compression_choice": pD_compress.value,
            "partC_usable_window_prediction": pC_pred.value,
            "partC_execution_target": pC_target.value,
            "partC_duty_cycle_pct": pC_duty.value,
            "partC_connectivity_pct": pC_connectivity.value,
            "partC_windows_per_day": pC_windows.value,
            "partD_guardrail_prediction": pD_pred.value,
            "selected_edge_policy": _recommended["policy_id"],
            "binding_edge_amount": _recommended["binding"],
            "rejected_alternative": _rejected["policy_id"],
            "v2_12_ops_implication": _implication_text,
            "memo_decision": _decision_text,
        }
        if not _incomplete:
            ledger.save(track=v2_11_profile.track_id, chapter=11, design=ledger_design)

        report = build_lab_report(
            v2_11_metadata,
            student_id=v2_11_student_id.value or "",
            track=v2_11_profile.label,
            scenario=v2_11_variant.workload_summary,
            learning_objectives=(
                "Find the local memory and energy boundary for the selected edge track.",
                "Explain the federated trade-off among communication, staleness, privacy, and convergence.",
                "Convert duty cycle and connectivity into usable intermittent evidence.",
                "Select an edge/federated deployment policy that satisfies guardrails.",
            ),
            predictions={
                "local_binding_amount": pA_pred.value,
                "federated_update_tradeoff": pB_pred.value,
                "usable_evidence_windows_per_day": pC_pred.value,
                "guardrail_binding_prediction": pD_pred.value,
            },
            knob_settings={
                "model_params_m": pA_params.value,
                "batch_size": pA_batch.value,
                "adaptation_strategy": pA_strategy.value,
                "privacy_epsilon": pB_contexts.value,
                "execution_target": pC_target.value,
                "duty_cycle_pct": pC_duty.value,
                "connectivity_pct": pC_connectivity.value,
                "candidate_windows_per_day": pC_windows.value,
                "heterogeneity_beta": pD_beta.value,
                "local_epochs": pD_epochs.value,
                "compression": pD_compress.value,
                "candidate_policy": pD_policy.value,
            },
            evidence_summary={
                "hardware_ref": v2_11_device.hardware_ref,
                "model_ref": v2_11_variant.model_ref,
                "active_memory_budget_mb": v2_11_device.available_memory_mb,
                "training_memory_mb": round(_memory.total_mb, 3),
                "training_fits": _memory.fits_memory,
                "energy_budget_used_pct_per_update": round(_energy.budget_used_pct, 3),
                "noniid_rounds": round(_comm.noniid_rounds, 1),
                "total_communication_mb": round(_comm.total_communication_mb, 3),
                "usable_updates_per_day": round(_intermittent["usable_updates"], 3),
                "evidence_age_hours": round(_intermittent["evidence_age_hours"], 3),
                "selected_edge_policy": _recommended["policy_id"],
                "binding_edge_amount": _recommended["binding"],
                "rejected_alternative": _rejected["policy_id"],
            },
            final_decision=(
                f"Use {_recommended['label']} for {v2_11_profile.label}; binding edge amount is "
                f"{_recommended['binding']}. Reject {_rejected['label']} because "
                f"{_rejected['rejected_reason']}."
            ),
            big_takeaways=(
                "Edge feasibility is a joint memory, energy, privacy, update, and quality decision.",
                "Federated learning preserves raw-data locality but spends rounds, bytes, and privacy budget.",
                "Intermittent evidence determines whether an apparently feasible policy can be trusted in production.",
            ),
            reflections={
                "edge_deployment_memo": _decision_text,
                "diagnosis": (
                    f"{v2_11_profile.label} is constrained first by "
                    f"{', '.join(v2_11_profile.dominant_constraints)}."
                ),
                "tradeoff": (
                    f"The selected path optimizes {v2_11_variant.primary_metric} while guarding "
                    f"{v2_11_variant.guardrail_metric}; policy table selected {_recommended['label']}."
                ),
                "residual_risk": (
                    "V2-12 operations must monitor eligibility, stale evidence, privacy-budget composition, "
                    "rollback triggers, and cohort quality."
                ),
            },
            residual_risk=(
                "The report records first-order teaching estimates, not measured hardware traces. "
                "Validate on representative devices and carry the policy into V2-12 fleet operations."
            ),
            source_trace={
                "track_id": v2_11_profile.track_id,
                "scenario_id": v2_11_variant.scenario_id,
                "hardware_ref": v2_11_variant.hardware_ref,
                "model_ref": v2_11_variant.model_ref,
                "shared_helper": "mlsysbook_labs.edge",
                "source_policy": v2_11_profile.source_policy,
            },
            result_snapshot={
                "device": v2_11_device,
                "memory": _memory,
                "energy": _energy,
                "communication": _comm,
                "intermittent": _intermittent,
                "policy": _policy_packet,
            },
            incomplete_fields=tuple(_incomplete),
        )

        items.append(mo.md("## Synthesis &mdash; Edge Intelligence Deployment Memo"))
        items.append(mo.Html(f"""
        <div class="mlsysbook-panel" style="border-left: 4px solid #1F407A; margin-top: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">STUDENT MEMO &amp; REFLECTIONS</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Edge Intelligence Deployment Memo</h4>
            {v2_11_student_id}
            <div style="margin-top: 12px;">{v2_11_next_implication}</div>
            <div style="margin-top: 12px;">{v2_11_memo_decision}</div>
        </div>
        """))

        _status_kind = "success" if not _incomplete else "warn"
        _status_text = "Saved to Design Ledger" if not _incomplete else "Complete all prediction and checkpoint controls to save."
        items.append(mo.callout(mo.md(f"**Memo summary:** {_decision_text}  \n\n**V2-12 implication:** {_implication_text}"), kind=_status_kind))
        items.append(mo.callout(mo.md(_status_text), kind=_status_kind))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="flex:1; min-width:220px; background:white; border:1px solid {COLORS['Border']};
                        border-radius:10px; padding:16px; border-top:3px solid {COLORS['GreenLine']};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']}; text-transform:uppercase;">
                    Selected policy</div>
                <div style="font-size:1.05rem; font-weight:800; color:{COLORS['Text']}; margin-top:5px;">
                    {_recommended['label']}</div>
            </div>
            <div style="flex:1; min-width:220px; background:white; border:1px solid {COLORS['Border']};
                        border-radius:10px; padding:16px; border-top:3px solid {COLORS['OrangeLine']};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']}; text-transform:uppercase;">
                    Binding edge amount</div>
                <div style="font-size:1.05rem; font-weight:800; color:{COLORS['Text']}; margin-top:5px;">
                    {_recommended['binding']}</div>
            </div>
            <div style="flex:1; min-width:220px; background:white; border:1px solid {COLORS['Border']};
                        border-radius:10px; padding:16px; border-top:3px solid {COLORS['RedLine']};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']}; text-transform:uppercase;">
                    Rejected alternative</div>
                <div style="font-size:1.05rem; font-weight:800; color:{COLORS['Text']}; margin-top:5px;">
                    {_rejected['label']}</div>
            </div>
        </div>
        """))

        items.append(big_takeaways([
            "On-device learning is bounded by local amounts (memory, thermal/battery energy, execution target).",
            "Federated updates move rather than eliminate cost: rounds, bytes, staleness, and DP noise dictate convergence.",
            "Intermittent evidence determines whether an apparently viable policy can survive real-world deployment.",
            "Edge deployment policy is a conjunctive guardrail bundle, not a single model artifact.",
        ]))

        items.append(mo.Html(f"""
        <div class="mlsysbook-panel" style="border-left: 4px solid #A51C30; margin-top: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">FINAL VERIFICATION &amp; SIGN-OFF</div>
            <h4 style="margin: 0 0 8px 0; color: #0F172A;">Lead Edge AI Architect Authorization</h4>
            <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #475569;">
                Confirm your edge intelligence memo, verify all conjunctive guardrails, and export the official engineering audit record.
            </p>
        </div>
        """))
        items.append(report_export_panel(report))

        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab V2-12: The Silent Fleet</strong> &mdash; Carry forward the
                    selected edge/federated policy, binding amount, rejected alternative, and
                    operations implication. The next problem is keeping that heterogeneous fleet
                    observable, rollback-ready, and within policy as devices drift.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> the Edge Intelligence chapter for full derivations.<br/>
                    <strong>Build:</strong> TinyTorch federated averaging module &mdash;
                    implement FedAvg with non-IID data simulation.
                </div>
            </div>
        </div>
        """))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A: On-Device Limits":            build_part_a(),
        "Part B: Federated Updates":           build_part_b(),
        "Part C: Intermittent Evidence":       build_part_c(),
        "Part D: Guardrailed Edge Policy":     build_part_d(),
        "Synthesis":                           build_synthesis(),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    pA_batch,
    pA_params,
    pB_contexts,
    pC_connectivity,
    pC_duty,
    pC_target,
    pC_windows,
    pD_beta,
    pD_compress,
    pD_epochs,
    pD_policy,
    v2_11_device,
    v2_11_policy_candidates,
    v2_11_profile,
):
    _packet = v2_11_policy_candidates(
        v2_11_device,
        v2_11_profile.track_id,
        pA_params.value,
        pA_batch.value,
        pC_target.value,
        pC_duty.value,
        pC_connectivity.value,
        pC_windows.value,
        pD_beta.value,
        pD_epochs.value,
        pD_compress.value,
        pB_contexts.value,
    )
    _policies = _packet["policies"]
    _selected = next((_policy for _policy in _policies if _policy["policy_id"] == pD_policy.value), _policies[1])
    _status = "PASS" if _selected["passes"] else "FAIL"
    _status_color = COLORS["GreenLine"] if _status == "PASS" else COLORS["RedLine"]
    return


if __name__ == "__main__":
    app.run()
