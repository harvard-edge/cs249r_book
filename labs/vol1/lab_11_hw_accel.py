import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


# ===========================================================================
# CELL 0: SETUP
# ===========================================================================


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
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        fusion_traffic,
        gemm_workload,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        hardware_roofline_profile,
        part_workflow,
        report_export_panel,
        resolve_mlsysim_ref,
        roofline_point,
        track_context,
        track_arc_context,
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
        fusion_traffic,
        gemm_workload,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        hardware_roofline_profile,
        ledger,
        math,
        mo,
        np,
        part_workflow,
        report_export_panel,
        resolve_mlsysim_ref,
        roofline_point,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_11_metadata = get_lab_metadata("vol1/lab_11_hw_accel.py")
    return (v1_11_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_11_track_picker = track_selector(default=_default_track)
    v1_11_track_picker
    return (v1_11_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    hardware_roofline_profile,
    resolve_mlsysim_ref,
    v1_11_track_picker,
):
    v1_11_track_id = v1_11_track_picker.value
    v1_11_profile = get_track_profile(v1_11_track_id)
    v1_11_variant = get_lab_track_variant("v1_11_hardware_roofline", v1_11_track_id)
    v1_11_hardware = resolve_mlsysim_ref(v1_11_variant.hardware_ref)
    v1_11_roofline = hardware_roofline_profile(v1_11_profile, v1_11_variant, v1_11_hardware)
    return (
        v1_11_hardware,
        v1_11_profile,
        v1_11_roofline,
        v1_11_track_id,
        v1_11_variant,
    )


# ===========================================================================
# CELL 1: NOTEBOOK-LOCAL V1-11 HELPERS
# ===========================================================================


@app.cell
def _(
    fusion_traffic,
    gemm_workload,
    math,
    v1_11_hardware,
    v1_11_profile,
    v1_11_roofline,
    v1_11_variant,
):
    def v1_11_quantity_to_float(value, unit, default=0.0):
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

    v1_11_track_configs = {
        "iphone": {
            "stakeholder": "mobile accelerator engineer",
            "amount_focus": "local latency, mJ/inference, sustained watts, thermal headroom",
            "memory_name": "LPDDR/unified memory",
            "local_memory_name": "Neural Engine SRAM/cache",
            "local_buffer_kb": 512,
            "movement_budget_us": 280,
            "hidden_dim": 2048,
            "default_batch": 4,
            "movement_power_w": 3.5,
            "shape_multiple": 16,
            "supported_precisions": ("fp16", "int8"),
            "quality_tolerance_pct": 1.5,
            "quality_loss_pct": {"fp32": 0.0, "fp16": 0.2, "int8": 1.1},
            "target_latency_ms": 18.0,
            "power_budget_w": 5.0,
            "secondary_budget_label": "energy/inference",
            "secondary_budget_limit": 70.0,
            "secondary_budget_unit": "mJ",
            "report_frame": "Ship on device only if the Neural Engine path stays supported and thermal headroom remains.",
            "failure_mode": "unsupported NPU op fallback or thermal throttle",
            "accelerator_paths": (
                {"id": "neural_engine", "label": "Neural Engine", "speed": 1.0, "power_w": 3.8, "extra_ms": 0.0, "cost": 0.0, "carbon_g": 0.0, "validation_rank": 2},
                {"id": "gpu_shaders", "label": "GPU shaders", "speed": 0.55, "power_w": 6.2, "extra_ms": 1.2, "cost": 0.0, "carbon_g": 0.0, "validation_rank": 1},
                {"id": "cpu_fallback", "label": "CPU fallback", "speed": 0.08, "power_w": 2.7, "extra_ms": 8.0, "cost": 0.0, "carbon_g": 0.0, "validation_rank": 1},
            ),
        },
        "oura_ring": {
            "stakeholder": "wearable firmware engineer",
            "amount_focus": "SRAM KB, wake time, uJ/window, duty cycle, flash image size",
            "memory_name": "external flash/DRAM staging",
            "local_memory_name": "MCU SRAM/scratchpad",
            "local_buffer_kb": 96,
            "movement_budget_us": 1600,
            "hidden_dim": 256,
            "default_batch": 8,
            "movement_power_w": 0.018,
            "shape_multiple": 8,
            "supported_precisions": ("int8",),
            "quality_tolerance_pct": 1.5,
            "quality_loss_pct": {"fp32": 0.0, "fp16": 0.4, "int8": 0.9},
            "target_latency_ms": 50.0,
            "power_budget_w": 0.025,
            "secondary_budget_label": "energy/window",
            "secondary_budget_limit": 0.8,
            "secondary_budget_unit": "mJ",
            "report_frame": "Run on ring only if tiles fit SRAM and duty-cycle energy stays inside the nightly budget.",
            "failure_mode": "SRAM spill, duty-cycle miss, or battery regression",
            "accelerator_paths": (
                {"id": "dsp_int8", "label": "DSP-like int8 kernel", "speed": 1.0, "power_w": 0.018, "extra_ms": 0.0, "cost": 0.0, "carbon_g": 0.0, "validation_rank": 2},
                {"id": "scalar_mcu", "label": "Scalar MCU", "speed": 0.18, "power_w": 0.012, "extra_ms": 3.0, "cost": 0.0, "carbon_g": 0.0, "validation_rank": 1},
                {"id": "phone_offload", "label": "Phone offload", "speed": 2.0, "power_w": 0.045, "extra_ms": 35.0, "cost": 0.0, "carbon_g": 0.0, "validation_rank": 2},
            ),
        },
        "robotaxi": {
            "stakeholder": "autonomous vehicle platform engineer",
            "amount_focus": "deterministic p99 ms, safety margin, watts, sensor-burst headroom",
            "memory_name": "vehicle accelerator memory",
            "local_memory_name": "edge accelerator SRAM/L2",
            "local_buffer_kb": 4096,
            "movement_budget_us": 450,
            "hidden_dim": 4096,
            "default_batch": 8,
            "movement_power_w": 55.0,
            "shape_multiple": 16,
            "supported_precisions": ("fp16", "int8"),
            "quality_tolerance_pct": 0.5,
            "quality_loss_pct": {"fp32": 0.0, "fp16": 0.2, "int8": 0.9},
            "target_latency_ms": 35.0,
            "power_budget_w": 65.0,
            "secondary_budget_label": "safety validation gap",
            "secondary_budget_limit": 0.5,
            "secondary_budget_unit": "% recall loss",
            "report_frame": "Approve only with deterministic edge latency and explicit fallback risk.",
            "failure_mode": "p99 deadline miss, power-envelope violation, or rare-event recall risk",
            "accelerator_paths": (
                {"id": "vehicle_accel", "label": "Vehicle accelerator", "speed": 1.0, "power_w": 58.0, "extra_ms": 0.0, "cost": 0.0, "carbon_g": 0.0, "validation_rank": 2},
                {"id": "gpu_fallback", "label": "GPU fallback", "speed": 0.72, "power_w": 82.0, "extra_ms": 2.0, "cost": 0.0, "carbon_g": 0.0, "validation_rank": 2},
                {"id": "cloud_fallback", "label": "Cloud fallback", "speed": 1.3, "power_w": 12.0, "extra_ms": 55.0, "cost": 0.0, "carbon_g": 0.0, "validation_rank": 2},
            ),
        },
        "cloud_fleet": {
            "stakeholder": "GPU performance engineer",
            "amount_focus": "MFU, HBM bandwidth, cost/request, p99 SLA, gCO2e/request",
            "memory_name": "HBM",
            "local_memory_name": "SM shared memory/L2",
            "local_buffer_kb": 16384,
            "movement_budget_us": 380,
            "hidden_dim": 8192,
            "default_batch": 32,
            "movement_power_w": 620.0,
            "shape_multiple": 16,
            "supported_precisions": ("fp16", "int8"),
            "quality_tolerance_pct": 0.8,
            "quality_loss_pct": {"fp32": 0.0, "fp16": 0.2, "int8": 0.7},
            "target_latency_ms": 80.0,
            "power_budget_w": 720.0,
            "secondary_budget_label": "cost/request",
            "secondary_budget_limit": 0.0025,
            "secondary_budget_unit": "USD",
            "report_frame": "Use accelerators only when utilization and cost/carbon beat alternatives under SLA.",
            "failure_mode": "low utilization, SLO breach, negative unit economics, or carbon waste",
            "accelerator_paths": (
                {"id": "h100_tensor", "label": "H100 tensor cores", "speed": 1.0, "power_w": 700.0, "extra_ms": 0.0, "cost": 0.0018, "carbon_g": 0.045, "validation_rank": 2},
                {"id": "a100_pool", "label": "A100 pool", "speed": 0.46, "power_w": 400.0, "extra_ms": 0.4, "cost": 0.0015, "carbon_g": 0.038, "validation_rank": 1},
                {"id": "cpu_fleet", "label": "CPU fleet", "speed": 0.055, "power_w": 180.0, "extra_ms": 4.0, "cost": 0.0042, "carbon_g": 0.085, "validation_rank": 1},
            ),
        },
    }

    v1_11_lens = dict(v1_11_track_configs[v1_11_profile.track_id])
    v1_11_lens.update(
        {
            "track_id": v1_11_profile.track_id,
            "track_label": v1_11_profile.label,
            "variant_stakeholder": v1_11_variant.stakeholder,
            "accelerator_path": v1_11_variant.assumptions.get("accelerator_path", v1_11_profile.label),
            "hardware_ref": v1_11_variant.hardware_ref,
            "model_ref": v1_11_variant.model_ref,
        }
    )

    def v1_11_roofline_result(dimension, precision):
        workload = gemm_workload(dimension=int(dimension), precision=str(precision).lower())
        point = roofline_point(v1_11_roofline, workload.arithmetic_intensity)
        latency_ms = workload.flops / (max(point.attainable_gflops, 1e-12) * 1e9) * 1000
        boundary = v1_11_boundary_dimension(precision)
        return {
            "workload": workload,
            "point": point,
            "latency_ms": latency_ms,
            "boundary_dimension": boundary,
            "actual_key": "compute" if point.regime == "Compute-bound" else "memory",
        }

    def v1_11_boundary_dimension(precision):
        bytes_per_element = {"fp32": 4, "fp16": 2, "int8": 1}[str(precision).lower()]
        raw_n = v1_11_roofline.ridge_flop_per_byte * 3 * bytes_per_element / 2
        return max(128, int(math.ceil(raw_n / 128) * 128))

    def v1_11_memory_result(mode, batch, workspace_kb):
        mode = str(mode)
        batch = int(batch)
        workspace_kb = float(workspace_kb)
        elements = batch * int(v1_11_lens["hidden_dim"])
        bytes_per_element = 2
        fusion = fusion_traffic(
            elements=elements,
            bytes_per_element=bytes_per_element,
            bandwidth_gbs=v1_11_roofline.bandwidth_gbs,
            eager_reads=3,
            eager_writes=3,
            fused_reads=1,
            fused_writes=1,
        )
        selected_bytes = fusion.eager_bytes if mode == "eager" else fusion.fused_bytes
        selected_time_us = fusion.eager_time_us if mode == "eager" else fusion.fused_time_us
        local_required_kb = elements * bytes_per_element * (3 if mode == "eager" else 1) / 1024
        available_kb = min(float(v1_11_lens["local_buffer_kb"]), workspace_kb)
        spills = local_required_kb > available_kb
        movement_time_us = selected_time_us * (1.65 if spills else 1.0)
        movement_miss = movement_time_us > float(v1_11_lens["movement_budget_us"])
        movement_energy_mj = float(v1_11_lens["movement_power_w"]) * movement_time_us / 1000
        return {
            "fusion": fusion,
            "mode": mode,
            "batch": batch,
            "elements": elements,
            "selected_bytes": selected_bytes,
            "selected_time_us": movement_time_us,
            "raw_time_us": selected_time_us,
            "local_required_kb": local_required_kb,
            "available_kb": available_kb,
            "spills": spills,
            "movement_miss": movement_miss,
            "movement_energy_mj": movement_energy_mj,
        }

    def v1_11_precision_peak_tflops(precision):
        precision = str(precision).lower()
        precision_map = getattr(v1_11_hardware.compute, "precision_flops", {})
        if precision == "fp32":
            peak = precision_map.get("fp32", precision_map.get("fp32_cuda", None))
        elif precision == "fp16":
            peak = precision_map.get("fp16", precision_map.get("bf16", None))
        elif precision == "int8":
            peak = precision_map.get("int8", None)
        else:
            peak = None
        if peak is None:
            peak = v1_11_hardware.compute.peak_flops
        return v1_11_quantity_to_float(peak, "TFLOPs/s", v1_11_roofline.peak_tflops)

    def v1_11_precision_result(dimension, precision):
        dimension = int(dimension)
        selected_precision = str(precision).lower()
        rows = []
        for candidate in ("fp32", "fp16", "int8"):
            workload = gemm_workload(dimension=dimension, precision=candidate)
            peak_tflops = v1_11_precision_peak_tflops(candidate)
            supported = candidate in v1_11_lens["supported_precisions"]
            aligned = dimension % int(v1_11_lens["shape_multiple"]) == 0
            quality_delta = float(v1_11_lens["quality_loss_pct"][candidate])
            quality_ok = quality_delta <= float(v1_11_lens["quality_tolerance_pct"])
            fast_path = bool(supported and aligned and quality_ok)
            if fast_path:
                effective_peak_tflops = peak_tflops
                status = "fast path"
                reason = "format, shape, and quality pass"
            elif not supported:
                effective_peak_tflops = max(peak_tflops * 0.18, 0.000001)
                status = "fallback"
                reason = f"{candidate.upper()} is not on the {v1_11_lens['accelerator_path']} fast path"
            elif not aligned:
                effective_peak_tflops = max(peak_tflops * 0.55, 0.000001)
                status = "padding/fallback"
                reason = f"dimension is not a multiple of {v1_11_lens['shape_multiple']}"
            else:
                effective_peak_tflops = max(peak_tflops * 0.40, 0.000001)
                status = "quality fail"
                reason = f"quality delta {quality_delta:.1f}% exceeds {v1_11_lens['quality_tolerance_pct']:.1f}%"
            peak_gflops = effective_peak_tflops * 1000
            attainable_gflops = min(peak_gflops, v1_11_roofline.bandwidth_gbs * workload.arithmetic_intensity)
            latency_ms = workload.flops / (max(attainable_gflops, 1e-12) * 1e9) * 1000
            rows.append(
                {
                    "precision": candidate,
                    "supported": supported,
                    "aligned": aligned,
                    "quality_delta_pct": quality_delta,
                    "quality_ok": quality_ok,
                    "fast_path": fast_path,
                    "status": status,
                    "reason": reason,
                    "effective_peak_tflops": effective_peak_tflops,
                    "attainable_gflops": attainable_gflops,
                    "latency_ms": latency_ms,
                    "arithmetic_intensity": workload.arithmetic_intensity,
                }
            )
        selected = next(row for row in rows if row["precision"] == selected_precision)
        return {"rows": tuple(rows), "selected": selected, "dimension": dimension}

    def v1_11_deployment_result(path_id, validation_level, a_result, b_result, c_result):
        validation_rank = {"prototype": 0, "profiled": 1, "full": 2}[validation_level]
        base_latency_ms = a_result["latency_ms"] + b_result["selected_time_us"] / 1000
        precision_penalty = 1.0 if c_result["selected"]["fast_path"] else 2.4
        rows = []
        for path in v1_11_lens["accelerator_paths"]:
            latency_ms = base_latency_ms * precision_penalty / path["speed"] + path["extra_ms"]
            energy_mj = path["power_w"] * latency_ms
            if v1_11_lens["track_id"] == "cloud_fleet":
                secondary_value = path["cost"] * max(1.0, latency_ms / v1_11_lens["target_latency_ms"])
            elif v1_11_lens["track_id"] == "robotaxi":
                secondary_value = c_result["selected"]["quality_delta_pct"]
            else:
                secondary_value = energy_mj
            failures = []
            if latency_ms > v1_11_lens["target_latency_ms"]:
                failures.append(f"latency {latency_ms:.2f} ms > {v1_11_lens['target_latency_ms']:.2f} ms")
            if path["power_w"] > v1_11_lens["power_budget_w"]:
                failures.append(f"power {path['power_w']:.2f} W > {v1_11_lens['power_budget_w']:.2f} W")
            if secondary_value > v1_11_lens["secondary_budget_limit"]:
                failures.append(
                    f"{v1_11_lens['secondary_budget_label']} {secondary_value:.4g} > "
                    f"{v1_11_lens['secondary_budget_limit']:.4g} {v1_11_lens['secondary_budget_unit']}"
                )
            if b_result["spills"]:
                failures.append(f"{v1_11_lens['local_memory_name']} spill")
            if not c_result["selected"]["fast_path"]:
                failures.append(c_result["selected"]["reason"])
            if validation_rank < path["validation_rank"]:
                failures.append("validation evidence is incomplete")
            rows.append(
                {
                    "path_id": path["id"],
                    "label": path["label"],
                    "latency_ms": latency_ms,
                    "power_w": path["power_w"],
                    "energy_mj": energy_mj,
                    "secondary_value": secondary_value,
                    "cost": path["cost"],
                    "carbon_g": path["carbon_g"],
                    "passes": len(failures) == 0,
                    "reason": "passes all current constraints" if not failures else "; ".join(failures),
                }
            )
        selected = next(row for row in rows if row["path_id"] == path_id)
        passing = [row for row in rows if row["passes"]]
        recommendation = passing[0] if passing else min(rows, key=lambda row: len(row["reason"].split("; ")))
        return {
            "rows": tuple(rows),
            "selected": selected,
            "recommendation": recommendation,
            "validation_rank": validation_rank,
            "base_latency_ms": base_latency_ms,
        }

    return (
        v1_11_boundary_dimension,
        v1_11_deployment_result,
        v1_11_lens,
        v1_11_memory_result,
        v1_11_precision_result,
        v1_11_roofline_result,
    )


# ===========================================================================
# CELL 2: HEADER AND BRIEFING
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    track_arc_context,
    track_context,
    v1_11_lens,
    v1_11_metadata,
    v1_11_profile,
    v1_11_roofline,
    v1_11_variant,
):
    mo.vstack(
        [
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            mo.Html(
                f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #94a3b8; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 11
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.35rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Hardware Acceleration Fit
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 650;
                      color: #cbd5e1; letter-spacing: 0.02em;">
                Roofline diagnosis &middot; Memory movement &middot; Precision contracts &middot; Deployment recommendation
            </p>
            <p style="margin: 0 0 22px 0; font-size: 0.98rem; color: #cbd5e1;
                      max-width: 820px; line-height: 1.65;">
                {v1_11_variant.workload_summary} The chapter invariant is that accelerators
                expose bottlenecks: speedup appears only when arithmetic intensity, precision,
                memory hierarchy, and hardware capability match the deployment envelope.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;">
                <span class="badge badge-info">4 concept modules &middot; ~50 min</span>
                <span class="badge badge-warn">{v1_11_profile.label}</span>
                <span class="badge badge-fail">{v1_11_roofline.hardware_ref}</span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Ridge {v1_11_roofline.ridge_flop_per_byte:.1f} FLOP/B</span>
                <span class="badge badge-warn">{v1_11_lens['amount_focus']}</span>
                <span class="badge badge-fail">{v1_11_variant.guardrail_metric}</span>
            </div>
        </div>
        """
            ),
            track_context(v1_11_profile),
            track_arc_context(v1_11_profile, v1_11_metadata.lab_id),
        ]
    )
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_11_lens):
    mo.Html(
        f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
            Lab Design Contract
        </div>
        <div style="font-size: 0.92rem; color: {COLORS['TextSec']}; line-height: 1.7;">
            <div>1. <strong>Part A:</strong> classify the active roofline regime.</div>
            <div>2. <strong>Part B:</strong> measure memory movement and local-buffer fit.</div>
            <div>3. <strong>Part C:</strong> test whether precision and shape satisfy the accelerator contract.</div>
            <div>4. <strong>Part D:</strong> recommend a deployment path under cost, power, validation, and residual-risk constraints.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 14px 28px 0 28px; color: {COLORS['Text']}; font-weight: 650;">
            Track amount system: {v1_11_lens['amount_focus']}.
        </div>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            """
**Recommended reading**

- Chapter 11, **Roofline Model** and **Hardware ridge points**.
- Chapter 11, **AI Memory Systems**, **Memory hierarchy**, and **Host-accelerator communication**.
- Chapter 11, **Tensor Cores**, **Numerics in AI acceleration**, and the **Tensor Core contract**.
- Chapter 11, **Heterogeneous SoC Design**, **Hardware Sustainability**, and **Feasibility assessment**.
"""
        ),
        kind="info",
    )
    return


# ===========================================================================
# CELL 3: WIDGETS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_11_variant):
    _default_dim = int(v1_11_variant.defaults.get("matrix_dim", 512))
    _default_precision = str(v1_11_variant.defaults.get("precision", "fp16")).lower()
    _precision_labels = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}
    pA_pred = mo.ui.radio(
        options={
            "Memory bandwidth is the first ceiling": "memory",
            "Compute throughput is the first ceiling": "compute",
            "The accelerator path is unsupported": "unsupported",
            "Thermal or power is the first ceiling": "power",
        },
        label="Part A prediction: why might peak throughput not appear?",
    )
    pA_dim = mo.ui.slider(start=128, stop=8192, value=_default_dim, step=128, label="Matrix dimension N")
    pA_prec = mo.ui.radio(
        options={"FP32": "fp32", "FP16": "fp16", "INT8": "int8"},
        value=_precision_labels.get(_default_precision, "FP16"),
        label="Arithmetic format for the roofline point",
        inline=True,
    )
    pA_action = mo.ui.radio(
        options={
            "Increase reuse or batch before buying hardware": "increase_reuse",
            "Reduce memory traffic first": "reduce_bytes",
            "Switch precision and retest": "switch_precision",
            "Reject this accelerator path": "reject_path",
        },
        label="Part A checkpoint: first action after the roofline diagnosis",
    )
    return (pA_action, pA_dim, pA_prec, pA_pred)


@app.cell(hide_code=True)
def _(mo, v1_11_lens):
    pB_pred = mo.ui.radio(
        options={
            "Minor, because FLOPs stay the same": "minor",
            "About 2x, because half the movement disappears": "2x",
            "About 3-5x, because round-trips disappear": "3_5x",
            "About 10x, because compute becomes free": "10x",
        },
        label="Part B prediction: how much can fusion or tiling help movement-heavy kernels?",
    )
    pB_mode = mo.ui.radio(
        options={"Eager separate kernels": "eager", "Fused single kernel": "fused"},
        value="Eager separate kernels",
        label="Execution mode",
        inline=True,
    )
    pB_batch = mo.ui.slider(
        start=1,
        stop=128,
        value=int(v1_11_lens["default_batch"]),
        step=1,
        label="Batch/windows processed together",
    )
    _workspace_stop = int(max(128, float(v1_11_lens["local_buffer_kb"]) * 2))
    pB_workspace = mo.ui.slider(
        start=16,
        stop=_workspace_stop,
        value=int(v1_11_lens["local_buffer_kb"]),
        step=16,
        label="Available local workspace (KB)",
    )
    pB_action = mo.ui.radio(
        options={
            "Fuse adjacent kernels": "fuse",
            "Tile into local memory": "tile",
            "Lower precision to reduce bytes": "lower_precision",
            "Shrink the batch/window": "shrink_window",
            "Reject the accelerator path": "reject_path",
        },
        label="Part B checkpoint: memory tactic",
    )
    return (pB_action, pB_batch, pB_mode, pB_pred, pB_workspace)


@app.cell(hide_code=True)
def _(mo, v1_11_variant):
    _default_dim = int(v1_11_variant.defaults.get("matrix_dim", 1024))
    _default_precision = str(v1_11_variant.defaults.get("precision", "fp16")).lower()
    _precision_labels = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}
    pC_pred = mo.ui.radio(
        options={
            "Lower precision always wins": "always_wins",
            "It can fail if the format is unsupported": "unsupported",
            "It can fail if the shape is misaligned": "misaligned",
            "It can fail if validation rejects the numeric change": "quality",
        },
        label="Part C prediction: what can prevent tensor-core or low-precision speedup?",
    )
    pC_dim = mo.ui.slider(start=128, stop=4096, value=_default_dim, step=64, label="Tensor dimension")
    pC_prec = mo.ui.radio(
        options={"FP32": "fp32", "FP16": "fp16", "INT8": "int8"},
        value=_precision_labels.get(_default_precision, "FP16"),
        label="Candidate precision",
        inline=True,
    )
    pC_action = mo.ui.radio(
        options={
            "Ship this precision path": "ship",
            "Pad or reshape tensors": "reshape",
            "Keep higher precision": "higher_precision",
            "Add numeric validation before sign-off": "validate",
        },
        label="Part C checkpoint: precision decision",
    )
    return (pC_action, pC_dim, pC_prec, pC_pred)


@app.cell(hide_code=True)
def _(mo, v1_11_lens):
    pD_pred = mo.ui.radio(
        options={
            "The highest peak path will win": "highest_peak",
            "Latency or p99 will reject it": "latency",
            "Power, energy, cost, or carbon will reject it": "amount_budget",
            "Validation evidence will reject it": "validation",
        },
        label="Part D prediction: what is most likely to reject the naive accelerator choice?",
    )
    _path_options = {path["label"]: path["id"] for path in v1_11_lens["accelerator_paths"]}
    pD_path = mo.ui.radio(
        options=_path_options,
        value=v1_11_lens["accelerator_paths"][0]["label"],
        label="Candidate deployment path",
    )
    pD_validation = mo.ui.radio(
        options={
            "Prototype trace only": "prototype",
            "Profiler plus load or replay test": "profiled",
            "Full validation gate": "full",
        },
        value="Profiler plus load or replay test",
        label="Validation evidence level",
    )
    pD_action = mo.ui.radio(
        options={
            "Recommend selected path": "recommend_selected",
            "Recommend first passing alternative": "recommend_passing",
            "Defer until validation completes": "defer_validation",
            "Reject acceleration for this release": "reject_release",
        },
        label="Part D checkpoint: deployment recommendation",
    )
    return (pD_action, pD_path, pD_pred, pD_validation)


# ===========================================================================
# CELL 4: CONCEPT MODULES
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    np,
    pA_action,
    pA_dim,
    pA_prec,
    pA_pred,
    pB_action,
    pB_batch,
    pB_mode,
    pB_pred,
    pB_workspace,
    pC_action,
    pC_dim,
    pC_prec,
    pC_pred,
    pD_action,
    pD_path,
    pD_pred,
    pD_validation,
    part_workflow,
    v1_11_deployment_result,
    v1_11_lens,
    v1_11_memory_result,
    v1_11_precision_result,
    v1_11_profile,
    v1_11_roofline,
    v1_11_roofline_result,
    v1_11_variant,
):
    def v1_11_table(headers, rows):
        _head = "".join(f"<th>{header}</th>" for header in headers)
        _rows = []
        for row in rows:
            _rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
        return mo.Html(
            f"""
        <table style="width:100%; border-collapse:collapse; font-size:0.86rem; margin:12px 0;
                      background:white; border:1px solid {COLORS['Border']};">
            <thead><tr style="background:{COLORS['Surface2']}; color:{COLORS['Text']};">{_head}</tr></thead>
            <tbody>{''.join(_rows)}</tbody>
        </table>
        <style>
        table th, table td {{
            padding: 8px 10px;
            border-bottom: 1px solid {COLORS['Border']};
            text-align: left;
            vertical-align: top;
        }}
        </style>
        """
        )

    def v1_11_cards(cards):
        _cards = []
        for label, value, sub, color in cards:
            _cards.append(
                f"""
            <div style="padding:15px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        background:white; border-top:3px solid {color}; flex:1; min-width:170px;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.76rem; font-weight:650;">{label}</div>
                <div style="font-size:1.35rem; font-weight:850; color:{color};">{value}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">{sub}</div>
            </div>
            """
            )
        return mo.Html(f"<div style='display:flex; gap:12px; flex-wrap:wrap; margin:14px 0;'>{''.join(_cards)}</div>")

    def v1_11_scenario(title, body, color):
        return mo.Html(
            f"""
        <div style="border-left:4px solid {color}; background:white; border-radius:0 10px 10px 0;
                    padding:16px 22px; margin:12px 0; box-shadow:0 1px 4px rgba(0,0,0,0.06);">
            <div style="font-size:0.72rem; font-weight:700; color:{color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">{title}</div>
            <div style="font-style:italic; font-size:1.0rem; color:{COLORS['Text']}; line-height:1.65;">{body}</div>
        </div>
        """
        )

    def v1_11_draw_roofline(result):
        _workload = result["workload"]
        _point = result["point"]
        _ais = np.logspace(-1, 4, 220)
        _roof = [min(v1_11_roofline.peak_tflops * 1000, v1_11_roofline.bandwidth_gbs * ai) for ai in _ais]
        _fig = go.Figure()
        _fig.add_trace(
            go.Scatter(
                x=_ais.tolist(),
                y=_roof,
                mode="lines",
                line=dict(color=COLORS["BlueLine"], width=3),
                name=f"{v1_11_profile.label} roofline",
                hovertemplate="AI %{x:.1f} FLOP/B: %{y:,.0f} GFLOP/s<extra></extra>",
            )
        )
        _fig.add_vline(
            x=v1_11_roofline.ridge_flop_per_byte,
            line_dash="dot",
            line_color=COLORS["OrangeLine"],
            annotation_text=f"ridge {v1_11_roofline.ridge_flop_per_byte:.1f}",
        )
        _fig.add_trace(
            go.Scatter(
                x=[_workload.arithmetic_intensity],
                y=[_point.attainable_gflops],
                mode="markers+text",
                marker=dict(size=15, color=COLORS["RedLine"], symbol="diamond"),
                text=[f"N={_workload.dimension}"],
                textposition="top right",
                name="current workload",
                hovertemplate="AI %{x:.1f}: %{y:,.0f} GFLOP/s<extra></extra>",
            )
        )
        _fig.update_layout(
            height=380,
            xaxis=dict(title="Arithmetic intensity (FLOP/byte)", type="log", range=[-1, 4]),
            yaxis=dict(title="Attainable performance (GFLOP/s)", type="log"),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        return mo.as_html(_fig)

    def v1_11_workflow(part, concept, prediction, controls, evidence, decision):
        return part_workflow(
            f"{part} Concept Module",
            (
                {
                    "part": part,
                    "concept": concept,
                    "prediction": prediction,
                    "controls": controls,
                    "evidence": evidence,
                    "decision": decision,
                },
            ),
            scenario=f"{v1_11_lens['variant_stakeholder']} must make a {v1_11_profile.label} deployment decision.",
            reflection="Use the numeric evidence to update the recommendation, not just the label of the bottleneck.",
        )

    def v1_11_build_part_a():
        items = [
            v1_11_scenario(
                "Incoming message - accelerator triage",
                (
                    f"The {v1_11_lens['stakeholder']} sees low utilization on {v1_11_roofline.hardware_name}. "
                    f"The question is whether this is a code failure, a roofline ceiling, or a {v1_11_lens['failure_mode']}."
                ),
                COLORS["BlueLine"],
            ),
            v1_11_workflow(
                "Part A",
                "Roofline separates compute-bound from memory-bound regimes.",
                "Commit to the most likely bottleneck before seeing the chart.",
                "Move matrix size and precision to change arithmetic intensity.",
                "Compare AI, ridge, attainable throughput, and MFU.",
                "Choose the first engineering action after the boundary is visible.",
            ),
            mo.md(
                f"""
### Concept

Peak throughput is only the flat roof. A workload below the ridge point is limited by
{v1_11_lens['memory_name']} bandwidth; above it, the compute ceiling is active.
"""
            ),
            pA_pred,
        ]
        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Select a prediction to unlock the roofline evidence."), kind="warn"))
            return mo.vstack(items)

        result = v1_11_roofline_result(pA_dim.value, pA_prec.value)
        point = result["point"]
        workload = result["workload"]
        boundary = result["boundary_dimension"]
        items.extend([mo.hstack([pA_dim, pA_prec], justify="start"), v1_11_draw_roofline(result)])
        items.append(
            v1_11_cards(
                (
                    ("Arithmetic intensity", f"{workload.arithmetic_intensity:.1f} FLOP/B", f"ridge {v1_11_roofline.ridge_flop_per_byte:.1f}", COLORS["BlueLine"]),
                    ("Actual regime", point.regime, f"prediction: {pA_pred.value}", COLORS["GreenLine"] if point.regime == "Compute-bound" else COLORS["OrangeLine"]),
                    ("MFU", f"{point.mfu_pct:.1f}%", f"{point.attainable_gflops:,.0f} of {point.peak_gflops:,.0f} GFLOP/s", COLORS["RedLine"]),
                    ("Boundary", f"N >= {boundary}", f"for {pA_prec.value.upper()} on this roof", COLORS["OrangeLine"]),
                )
            )
        )
        items.append(
            v1_11_table(
                ("Quantity", "Value", "Interpretation"),
                (
                    ("FLOPs", f"{workload.flops:.3e}", "work performed by the GEMM"),
                    ("Bytes moved", f"{workload.bytes_moved:.3e}", f"{pA_prec.value.upper()} read A/read B/write C traffic"),
                    ("AI vs ridge", f"{workload.arithmetic_intensity:.1f} vs {v1_11_roofline.ridge_flop_per_byte:.1f}", point.regime),
                    ("Track amount at risk", v1_11_lens["amount_focus"], v1_11_variant.guardrail_metric),
                ),
            )
        )
        _prediction_correct = pA_pred.value == result["actual_key"]
        items.append(
            mo.callout(
                mo.md(
                    (
                        f"Prediction matched the measured regime: AI {workload.arithmetic_intensity:.1f} "
                        f"against ridge {v1_11_roofline.ridge_flop_per_byte:.1f} is {point.regime.lower()}."
                    )
                    if _prediction_correct
                    else (
                        f"The evidence points to {point.regime.lower()}, not `{pA_pred.value}`. "
                        f"At this point the boundary is N >= {boundary} for {pA_prec.value.upper()}."
                    )
                ),
                kind="success" if _prediction_correct else "warn",
            )
        )
        items.append(
            mo.accordion(
                {
                    "Math Peek and source model": mo.md(
                        f"""
Formula:

```text
AI = FLOPs / bytes
ridge = peak FLOP/s / memory bandwidth
R_attainable = min(peak FLOP/s, memory bandwidth * AI)
```

Source model: `mlsysbook_labs.gemm_workload` and `mlsysbook_labs.roofline_point`
with `{v1_11_roofline.hardware_ref}` from MLSysIM. The chapter claim is the
Roofline Model: the plot reveals whether the active ceiling is compute or bandwidth.
"""
                    )
                }
            )
        )
        items.append(pA_action)
        return mo.vstack(items)

    def v1_11_build_part_b():
        items = [
            v1_11_scenario(
                "Incoming message - memory movement review",
                (
                    f"The same model alternates matrix kernels with elementwise work. The {v1_11_lens['stakeholder']} "
                    f"needs to know whether {v1_11_lens['local_memory_name']} can keep data close enough."
                ),
                COLORS["OrangeLine"],
            ),
            v1_11_workflow(
                "Part B",
                "Memory hierarchy and data movement can dominate accelerator performance.",
                "Predict the speedup from eliminating memory round-trips.",
                "Change execution mode, batch/window count, and local workspace.",
                "Inspect traffic, movement time, energy, and spill status.",
                "Choose the memory tactic to carry into deployment.",
            ),
            mo.md(
                f"""
### Concept

Fusion and tiling do not reduce the mathematical operation count. They reduce traffic through
{v1_11_lens['memory_name']} and keep reused values in {v1_11_lens['local_memory_name']}.
"""
            ),
            pB_pred,
        ]
        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Select a prediction to unlock the memory movement evidence."), kind="warn"))
            return mo.vstack(items)

        result = v1_11_memory_result(pB_mode.value, pB_batch.value, pB_workspace.value)
        fusion = result["fusion"]
        items.append(mo.hstack([pB_mode, pB_batch, pB_workspace], justify="start"))
        _fig = go.Figure()
        _fig.add_trace(
            go.Bar(
                x=["Eager bytes", "Fused bytes", "Selected bytes"],
                y=[fusion.eager_bytes / 1024, fusion.fused_bytes / 1024, result["selected_bytes"] / 1024],
                marker_color=[COLORS["RedLine"], COLORS["GreenLine"], COLORS["BlueLine"]],
                text=[f"{fusion.eager_bytes/1024:.1f} KB", f"{fusion.fused_bytes/1024:.1f} KB", f"{result['selected_bytes']/1024:.1f} KB"],
                textposition="outside",
                hovertemplate="%{x}: %{y:.1f} KB<extra></extra>",
            )
        )
        _fig.update_layout(height=320, yaxis=dict(title="Traffic through slower memory (KB)"), showlegend=False)
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))
        items.append(
            v1_11_cards(
                (
                    ("Fusion ratio", f"{fusion.speedup:.1f}x", "traffic-only upper bound", COLORS["GreenLine"]),
                    ("Movement time", f"{result['selected_time_us']:.2f} us", f"budget {v1_11_lens['movement_budget_us']} us", COLORS["OrangeLine"]),
                    ("Local workspace", f"{result['local_required_kb']:.1f} KB", f"available {result['available_kb']:.1f} KB", COLORS["BlueLine"]),
                    ("Movement energy", f"{result['movement_energy_mj']:.4g} mJ", v1_11_lens["secondary_budget_label"], COLORS["RedLine"]),
                )
            )
        )
        items.append(
            v1_11_table(
                ("Check", "Value", "Status"),
                (
                    ("Selected mode", result["mode"], "fused removes intermediate writes" if result["mode"] == "fused" else "eager pays every round-trip"),
                    ("Local fit", f"{result['local_required_kb']:.1f} KB <= {result['available_kb']:.1f} KB", "spill" if result["spills"] else "fits"),
                    ("Movement budget", f"{result['selected_time_us']:.2f} us <= {v1_11_lens['movement_budget_us']} us", "miss" if result["movement_miss"] else "passes"),
                    ("Failure mode", v1_11_lens["failure_mode"], "recover by fusing, tiling, or shrinking the window"),
                ),
            )
        )
        if result["spills"] or result["movement_miss"]:
            _why = "local workspace spills" if result["spills"] else "movement time exceeds the budget"
            items.append(mo.callout(mo.md(f"Boundary hit: {_why}. This is reversible by moving the controls."), kind="danger"))
        else:
            items.append(mo.callout(mo.md("The selected memory plan stays inside the current local-buffer and movement-time budgets."), kind="success"))
        items.append(
            mo.accordion(
                {
                    "Math Peek and source model": mo.md(
                        f"""
Fusion traffic model:

```text
eager bytes = (reads + writes) * tensor bytes
fused bytes = (one read + one write) * tensor bytes
movement time = selected bytes / bandwidth
```

Source model: `mlsysbook_labs.fusion_traffic`, using `{v1_11_roofline.bandwidth_gbs:g} GB/s`
from `{v1_11_roofline.hardware_ref}`. Chapter connection: Memory hierarchy and host-accelerator
communication explain why eliminated round-trips can dominate speed and energy.
"""
                    )
                }
            )
        )
        items.append(pB_action)
        return mo.vstack(items)

    def v1_11_build_part_c():
        items = [
            v1_11_scenario(
                "Incoming message - tensor path review",
                (
                    f"The {v1_11_lens['stakeholder']} wants to enable a faster numeric path. "
                    "The question is whether the format, tensor shape, and validation envelope all pass."
                ),
                COLORS["GreenLine"],
            ),
            v1_11_workflow(
                "Part C",
                "Tensor cores and precision accelerate only supported shapes and tolerable numeric formats.",
                "Predict which contract term breaks first.",
                "Change precision and tensor dimension.",
                "Inspect fast-path support, alignment, quality delta, and latency.",
                "Choose the precision path and validation note.",
            ),
            mo.md(
                f"""
### Concept

Reduced precision changes both sides of the roofline: fewer bytes move and specialized datapaths
can raise peak throughput. That only helps when `{v1_11_lens['accelerator_path']}` supports the
format, dimension multiples of {v1_11_lens['shape_multiple']} are respected, and quality loss stays
within {v1_11_lens['quality_tolerance_pct']:.1f}%.
"""
            ),
            pC_pred,
        ]
        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Select a prediction to unlock the tensor/precision evidence."), kind="warn"))
            return mo.vstack(items)

        result = v1_11_precision_result(pC_dim.value, pC_prec.value)
        selected = result["selected"]
        items.append(mo.hstack([pC_dim, pC_prec], justify="start"))
        _fig = go.Figure()
        _fig.add_trace(
            go.Bar(
                x=[row["precision"].upper() for row in result["rows"]],
                y=[row["latency_ms"] for row in result["rows"]],
                marker_color=[COLORS["GreenLine"] if row["fast_path"] else COLORS["RedLine"] for row in result["rows"]],
                text=[row["status"] for row in result["rows"]],
                textposition="outside",
                hovertemplate="%{x}: %{y:.4f} ms<extra></extra>",
            )
        )
        _fig.update_layout(height=320, yaxis=dict(title="Estimated kernel latency (ms)"), showlegend=False)
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))
        items.append(
            v1_11_table(
                ("Precision", "AI", "Fast path", "Quality delta", "Latency", "Reason"),
                tuple(
                    (
                        row["precision"].upper(),
                        f"{row['arithmetic_intensity']:.1f}",
                        "yes" if row["fast_path"] else row["status"],
                        f"{row['quality_delta_pct']:.1f}% / {v1_11_lens['quality_tolerance_pct']:.1f}%",
                        f"{row['latency_ms']:.4f} ms",
                        row["reason"],
                    )
                    for row in result["rows"]
                ),
            )
        )
        items.append(
            v1_11_cards(
                (
                    ("Selected precision", selected["precision"].upper(), selected["status"], COLORS["GreenLine"] if selected["fast_path"] else COLORS["RedLine"]),
                    ("Shape alignment", "passes" if selected["aligned"] else "fails", f"multiple of {v1_11_lens['shape_multiple']}", COLORS["BlueLine"]),
                    ("Quality gate", "passes" if selected["quality_ok"] else "fails", f"delta {selected['quality_delta_pct']:.1f}%", COLORS["OrangeLine"]),
                    ("Effective peak", f"{selected['effective_peak_tflops']:.3g} TFLOP/s", "after fallback penalties", COLORS["RedLine"]),
                )
            )
        )
        if selected["fast_path"]:
            items.append(mo.callout(mo.md(f"{selected['precision'].upper()} satisfies the current tensor/precision contract."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"Contract failure: {selected['reason']}. Speedup is not valid until this is fixed."), kind="danger"))
        items.append(
            mo.accordion(
                {
                    "Math Peek and source model": mo.md(
                        f"""
Precision affects both bytes and peak throughput:

```text
AI_GEMM = 2N^3 / (3N^2 * bytes_per_element)
fast path = supported_format and aligned_shape and quality_delta <= tolerance
```

Source model: `mlsysbook_labs.gemm_workload` plus notebook-local `v1_11_precision_result`.
Chapter connection: Tensor cores require supported precision and shape contracts; mixed precision
must still pass validation.
"""
                    )
                }
            )
        )
        items.append(pC_action)
        return mo.vstack(items)

    def v1_11_build_part_d():
        a_result = v1_11_roofline_result(pA_dim.value, pA_prec.value)
        b_result = v1_11_memory_result(pB_mode.value, pB_batch.value, pB_workspace.value)
        c_result = v1_11_precision_result(pC_dim.value, pC_prec.value)
        d_result = v1_11_deployment_result(pD_path.value, pD_validation.value, a_result, b_result, c_result)
        selected = d_result["selected"]
        recommendation = d_result["recommendation"]
        items = [
            v1_11_scenario(
                "Incoming message - deployment sign-off",
                (
                    f"The {v1_11_lens['stakeholder']} now needs a recommendation, not a benchmark. "
                    f"The memo must respect {v1_11_variant.guardrail_metric} and name residual risk."
                ),
                COLORS["RedLine"],
            ),
            v1_11_workflow(
                "Part D",
                "Accelerator fit is a deployment recommendation with cost, power, and validation constraints.",
                "Predict what rejects the naive highest-peak path.",
                "Choose a candidate path and validation evidence level.",
                "Compare latency, power/energy/cost/carbon, validation, and failure reasons.",
                "Commit the recommendation and rejected alternatives.",
            ),
            mo.md(
                f"""
### Concept

The deployable accelerator is the one that passes the amount system for this track:
{v1_11_lens['amount_focus']}. The chapter feasibility check is practical: compute time,
memory fit, power or cost, and validation must all pass at the same time.
"""
            ),
            pD_pred,
        ]
        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Select a prediction to unlock the deployment comparison."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pD_path, pD_validation], justify="start"))
        items.append(
            v1_11_table(
                ("Path", "Latency", "Power", v1_11_lens["secondary_budget_label"], "Pass?", "Reason"),
                tuple(
                    (
                        row["label"],
                        f"{row['latency_ms']:.3f} ms",
                        f"{row['power_w']:.3g} W",
                        f"{row['secondary_value']:.4g} {v1_11_lens['secondary_budget_unit']}",
                        "yes" if row["passes"] else "no",
                        row["reason"],
                    )
                    for row in d_result["rows"]
                ),
            )
        )
        items.append(
            v1_11_cards(
                (
                    ("Selected path", selected["label"], "student-controlled candidate", COLORS["BlueLine"]),
                    ("Selected status", "passes" if selected["passes"] else "fails", selected["reason"], COLORS["GreenLine"] if selected["passes"] else COLORS["RedLine"]),
                    ("Recommended path", recommendation["label"], "first passing path or least-bad fallback", COLORS["OrangeLine"]),
                    ("Base model+movement", f"{d_result['base_latency_ms']:.4f} ms", "before path/fallback factors", COLORS["RedLine"]),
                )
            )
        )
        if selected["passes"]:
            items.append(mo.callout(mo.md(f"The selected path passes the current deployment constraints: {selected['label']}."), kind="success"))
        else:
            items.append(mo.callout(mo.md(f"Deployment failure for {selected['label']}: {selected['reason']}."), kind="danger"))
        items.append(
            mo.accordion(
                {
                    "Math Peek and source model": mo.md(
                        """
Feasibility model:

```text
T_process = operations / attainable_throughput
movement_time = bytes_moved / bandwidth
path_latency = (T_process + movement_time) / path_speed + transfer_or_runtime_overhead
deployment_pass = latency_ok and power_or_cost_ok and memory_ok and precision_ok and validation_ok
```

Source model: Part A uses the shared roofline helpers, Part B uses shared fusion traffic,
and Part C/D use notebook-local `v1_11_` scenario assumptions for deployment budgets.
Chapter connection: the feasibility assessment asks whether the workload can run inside
memory, bandwidth, compute, power, cost, and validation limits.
"""
                    )
                }
            )
        )
        items.append(pD_action)
        return mo.vstack(items)

    def v1_11_build_synthesis():
        a_result = v1_11_roofline_result(pA_dim.value, pA_prec.value)
        b_result = v1_11_memory_result(pB_mode.value, pB_batch.value, pB_workspace.value)
        c_result = v1_11_precision_result(pC_dim.value, pC_prec.value)
        d_result = v1_11_deployment_result(pD_path.value, pD_validation.value, a_result, b_result, c_result)
        selected = d_result["selected"]
        recommendation = d_result["recommendation"]
        rejected = "; ".join(
            f"{row['label']}: {row['reason']}" for row in d_result["rows"] if row["path_id"] != recommendation["path_id"]
        )
        incomplete = []
        for label, widget in (
            ("Part A prediction", pA_pred),
            ("Part B prediction", pB_pred),
            ("Part C prediction", pC_pred),
            ("Part D prediction", pD_pred),
            ("Part A checkpoint", pA_action),
            ("Part B checkpoint", pB_action),
            ("Part C checkpoint", pC_action),
            ("Part D checkpoint", pD_action),
        ):
            if widget.value is None:
                incomplete.append(label)
        items = [
            mo.md("## Synthesis: Hardware Acceleration Memo"),
            mo.callout(
                mo.md(
                    f"""
**Decision:** {v1_11_lens['report_frame']}

**Bottleneck diagnosis:** Part A measured AI = {a_result['workload'].arithmetic_intensity:.1f}
FLOP/B against ridge = {v1_11_roofline.ridge_flop_per_byte:.1f}, so the active regime is
{a_result['point'].regime.lower()}.

**Memory movement evidence:** Part B selected `{b_result['mode']}` execution with
{b_result['selected_bytes']/1024:.1f} KB of slower-memory traffic, {b_result['selected_time_us']:.2f} us
movement time, and {"a local-memory spill" if b_result['spills'] else "no local-memory spill"}.

**Selected accelerator/precision:** Part C selected {c_result['selected']['precision'].upper()}
with status `{c_result['selected']['status']}`. Part D selected `{selected['label']}` and recommends
`{recommendation['label']}`.

**Rejected alternatives:** {rejected if rejected else "No alternative was rejected by the current constraints."}

**Residual risk:** validate profiler counters, supported-operator coverage, thermal or duty-cycle behavior,
and production p99 before treating the recommendation as release evidence.
"""
                ),
                kind="info",
            ),
            v1_11_table(
                ("Memo field", "Value"),
                (
                    ("Track", v1_11_profile.label),
                    ("Hardware ref", v1_11_roofline.hardware_ref),
                    ("Model ref", v1_11_variant.model_ref),
                    ("Recommended path", recommendation["label"]),
                    ("Selected precision", c_result["selected"]["precision"].upper()),
                    ("Deployment pass", "yes" if recommendation["passes"] else "no"),
                    ("Residual risk", v1_11_lens["failure_mode"]),
                ),
            ),
        ]
        if incomplete:
            items.append(mo.callout(mo.md("Complete these fields before the ledger save is meaningful: " + ", ".join(incomplete)), kind="warn"))
        return mo.vstack(items)

    def build_synthesis():
        return v1_11_build_synthesis()

    _tabs = mo.ui.tabs(
        {
            "Part A: Roofline Regime": v1_11_build_part_a(),
            "Part B: Memory Movement": v1_11_build_part_b(),
            "Part C: Precision Contract": v1_11_build_part_c(),
            "Part D: Deployment Fit": v1_11_build_part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    _tabs
    return


# ===========================================================================
# CELL 5: LEDGER HUD
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    ledger,
    mo,
    pA_action,
    pA_dim,
    pA_prec,
    pA_pred,
    pB_action,
    pB_batch,
    pB_mode,
    pB_pred,
    pB_workspace,
    pC_action,
    pC_dim,
    pC_prec,
    pC_pred,
    pD_action,
    pD_path,
    pD_pred,
    pD_validation,
    v1_11_deployment_result,
    v1_11_lens,
    v1_11_memory_result,
    v1_11_precision_result,
    v1_11_profile,
    v1_11_roofline,
    v1_11_roofline_result,
    v1_11_variant,
):
    _a = v1_11_roofline_result(pA_dim.value, pA_prec.value)
    _b = v1_11_memory_result(pB_mode.value, pB_batch.value, pB_workspace.value)
    _c = v1_11_precision_result(pC_dim.value, pC_prec.value)
    _d = v1_11_deployment_result(pD_path.value, pD_validation.value, _a, _b, _c)
    _complete = all(
        widget.value is not None
        for widget in (pA_pred, pB_pred, pC_pred, pD_pred, pA_action, pB_action, pC_action, pD_action)
    )
    if _complete:
        ledger.save(
            chapter=11,
            design={
                "lab": "hw_accel",
                "track_id": v1_11_profile.track_id,
                "scenario_id": v1_11_variant.scenario_id,
                "hardware_ref": v1_11_roofline.hardware_ref,
                "model_ref": v1_11_variant.model_ref,
                "completed": True,
                "part_a_prediction": pA_pred.value,
                "part_a_action": pA_action.value,
                "workload_ai": round(_a["workload"].arithmetic_intensity, 3),
                "ridge_flop_per_byte": round(v1_11_roofline.ridge_flop_per_byte, 3),
                "roofline_regime": _a["point"].regime,
                "mfu_pct": round(_a["point"].mfu_pct, 3),
                "ridge_boundary_dimension": _a["boundary_dimension"],
                "part_b_prediction": pB_pred.value,
                "part_b_action": pB_action.value,
                "memory_mode": pB_mode.value,
                "memory_batch": pB_batch.value,
                "local_workspace_kb": round(_b["local_required_kb"], 3),
                "local_buffer_kb": round(_b["available_kb"], 3),
                "memory_spill": _b["spills"],
                "movement_time_us": round(_b["selected_time_us"], 3),
                "part_c_prediction": pC_pred.value,
                "part_c_action": pC_action.value,
                "precision_choice": pC_prec.value,
                "shape_dimension": pC_dim.value,
                "shape_aligned": _c["selected"]["aligned"],
                "precision_supported": _c["selected"]["supported"],
                "quality_delta_pct": _c["selected"]["quality_delta_pct"],
                "quality_tolerance_pct": v1_11_lens["quality_tolerance_pct"],
                "precision_fast_path": _c["selected"]["fast_path"],
                "part_d_prediction": pD_pred.value,
                "part_d_action": pD_action.value,
                "accelerator_choice": pD_path.value,
                "validation_level": pD_validation.value,
                "deployment_pass": _d["selected"]["passes"],
                "deployment_rejection_reason": _d["selected"]["reason"],
                "recommended_path": _d["recommendation"]["label"],
                "residual_risk": v1_11_lens["failure_mode"],
            },
        )
    mo.Html(
        f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">11 &middot; Hardware Acceleration</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_11_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">CH</span>
        <span class="hud-value">11</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">{'SAVED' if _complete else 'IN PROGRESS'}</span>
    </div>
    """
    )
    return


# ===========================================================================
# CELL 6: DOWNLOADABLE REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    pA_action,
    pA_dim,
    pA_prec,
    pA_pred,
    pB_action,
    pB_batch,
    pB_mode,
    pB_pred,
    pB_workspace,
    pC_action,
    pC_dim,
    pC_prec,
    pC_pred,
    pD_action,
    pD_path,
    pD_pred,
    pD_validation,
    report_export_panel,
    v1_11_deployment_result,
    v1_11_lens,
    v1_11_memory_result,
    v1_11_metadata,
    v1_11_precision_result,
    v1_11_profile,
    v1_11_roofline,
    v1_11_roofline_result,
    v1_11_variant,
):
    _a = v1_11_roofline_result(pA_dim.value, pA_prec.value)
    _b = v1_11_memory_result(pB_mode.value, pB_batch.value, pB_workspace.value)
    _c = v1_11_precision_result(pC_dim.value, pC_prec.value)
    _d = v1_11_deployment_result(pD_path.value, pD_validation.value, _a, _b, _c)
    _selected = _d["selected"]
    _recommendation = _d["recommendation"]
    _incomplete = []
    for _label, _widget in (
        ("Part A prediction", pA_pred),
        ("Part B prediction", pB_pred),
        ("Part C prediction", pC_pred),
        ("Part D prediction", pD_pred),
        ("Part A checkpoint", pA_action),
        ("Part B checkpoint", pB_action),
        ("Part C checkpoint", pC_action),
        ("Part D checkpoint", pD_action),
    ):
        if _widget.value is None:
            _incomplete.append(_label)

    _rejected = tuple(
        f"{row['label']}: {row['reason']}" for row in _d["rows"] if row["path_id"] != _recommendation["path_id"]
    )
    _report = build_lab_report(
        v1_11_metadata,
        track=v1_11_profile.label,
        scenario=v1_11_variant.workload_summary,
        learning_objectives=(
            "Diagnose whether the workload is memory-bound or compute-bound using a roofline.",
            "Explain how memory hierarchy and data movement dominate accelerator performance.",
            "Validate precision and tensor-shape contracts before claiming tensor-core speedup.",
            "Recommend an accelerator path under latency, cost, power, validation, and residual-risk constraints.",
        ),
        predictions={
            "part_a_roofline": pA_pred.value,
            "part_b_memory": pB_pred.value,
            "part_c_precision": pC_pred.value,
            "part_d_deployment": pD_pred.value,
        },
        knob_settings={
            "matrix_dim": pA_dim.value,
            "roofline_precision": pA_prec.value,
            "memory_mode": pB_mode.value,
            "memory_batch": pB_batch.value,
            "local_workspace_kb": pB_workspace.value,
            "precision_dim": pC_dim.value,
            "precision_choice": pC_prec.value,
            "accelerator_choice": pD_path.value,
            "validation_level": pD_validation.value,
            "part_a_action": pA_action.value,
            "part_b_action": pB_action.value,
            "part_c_action": pC_action.value,
            "part_d_action": pD_action.value,
        },
        evidence_summary={
            "hardware_ref": v1_11_roofline.hardware_ref,
            "model_ref": v1_11_variant.model_ref,
            "peak_tflops": round(v1_11_roofline.peak_tflops, 6),
            "bandwidth_gbs": round(v1_11_roofline.bandwidth_gbs, 6),
            "ridge_flop_per_byte": round(v1_11_roofline.ridge_flop_per_byte, 3),
            "workload_ai": round(_a["workload"].arithmetic_intensity, 3),
            "primary_regime": _a["point"].regime,
            "primary_mfu_pct": round(_a["point"].mfu_pct, 3),
            "movement_time_us": round(_b["selected_time_us"], 3),
            "memory_spill": _b["spills"],
            "precision_fast_path": _c["selected"]["fast_path"],
            "precision_status": _c["selected"]["status"],
            "deployment_selected_path": _selected["label"],
            "deployment_selected_pass": _selected["passes"],
            "recommended_path": _recommendation["label"],
            "rejected_alternatives": _rejected,
        },
        final_decision=(
            f"Recommend {_recommendation['label']} for {v1_11_profile.label} only with "
            f"{pC_prec.value.upper()} precision evidence, {pD_validation.value} validation, "
            f"and explicit residual risk: {v1_11_lens['failure_mode']}."
        ),
        big_takeaways=(
            "Peak TOPS is not performance; arithmetic intensity decides which ceiling is active.",
            "Memory hierarchy matters because eliminated movement can dominate speed and energy.",
            "Precision speedups are contracts: supported format, aligned shape, and tolerated quality loss.",
            "Deployment recommendations must reject alternatives and name residual validation risk.",
        ),
        reflections={
            "bottleneck_diagnosis": (
                f"AI={_a['workload'].arithmetic_intensity:.1f} FLOP/B versus ridge="
                f"{v1_11_roofline.ridge_flop_per_byte:.1f} gives {_a['point'].regime.lower()} behavior."
            ),
            "selected_accelerator_precision": (
                f"{_recommendation['label']} with {pC_prec.value.upper()} precision; "
                f"precision status is {_c['selected']['status']}."
            ),
            "rejected_alternatives": "; ".join(_rejected) if _rejected else "No rejected alternative.",
            "residual_risk": v1_11_lens["failure_mode"],
        },
        residual_risk=(
            "Roofline and scenario budgets are first-order evidence. Validate with profiler counters, "
            "supported-op coverage, numerical checks, thermal or duty-cycle tests, and production p99."
        ),
        source_trace={
            "track_id": v1_11_profile.track_id,
            "scenario_id": v1_11_variant.scenario_id,
            "hardware_ref": v1_11_variant.hardware_ref,
            "model_ref": v1_11_variant.model_ref,
            "shared_helper": "mlsysbook_labs.roofline",
            "notebook_local_helpers": (
                "v1_11_memory_result",
                "v1_11_precision_result",
                "v1_11_deployment_result",
            ),
            "source_policy": v1_11_profile.source_policy,
        },
        result_snapshot={
            "roofline": _a,
            "memory": _b,
            "precision": _c,
            "deployment": _d,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack(
        [
            mo.md("## Download Report"),
            mo.callout(
                mo.md(
                    "This V1-11 report is generated locally from the selected track, MLSysIM hardware refs, "
                    "shared roofline calculations, and notebook-local deployment scenario budgets."
                ),
                kind="info",
            ),
            report_export_panel(_report),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
