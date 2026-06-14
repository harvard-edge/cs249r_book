import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


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
    from mlsysim import Hardware, Models, Systems
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        MathPeek,
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
        Hardware,
        LAB_CSS,
        MathPeek,
        Models,
        Systems,
        apply_plotly_theme,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
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
    v2_05_lab_path = "vol2/lab_05_dist_train.py"
    v2_05_chapter = 5
    v2_05_metadata = get_lab_metadata(v2_05_lab_path)
    return v2_05_chapter, v2_05_lab_path, v2_05_metadata


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_05_track_picker = track_selector(default=_default_track)
    v2_05_track_picker
    return (v2_05_track_picker,)


@app.cell
def _(get_lab_track_variant, get_track_profile, v2_05_metadata, v2_05_track_picker):
    v2_05_track_id = v2_05_track_picker.value
    v2_05_profile = get_track_profile(v2_05_track_id)
    v2_05_variant = get_lab_track_variant(v2_05_metadata.lab_id, v2_05_profile.track_id)
    return v2_05_profile, v2_05_track_id, v2_05_variant


@app.cell
def _(Hardware, Models, Systems, apply_plotly_theme, go, html_lib, math, mo):
    def v2_05_qty_to_float(value, unit, default):
        if value is None:
            return float(default)
        if hasattr(value, "m_as"):
            try:
                return float(value.m_as(unit))
            except Exception:
                return float(default)
        if hasattr(value, "to"):
            try:
                return float(value.to(unit).magnitude)
            except Exception:
                return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def v2_05_escape(value):
        return html_lib.escape(str(value))

    def v2_05_fmt(value, digits=1, suffix=""):
        value = float(value)
        if abs(value) >= 1000:
            text = f"{value:,.{digits}f}"
        else:
            text = f"{value:.{digits}f}"
        if digits == 0:
            text = text.split(".")[0]
        return f"{text}{suffix}"

    def v2_05_pct(value, digits=0):
        return f"{100 * float(value):.{digits}f}%"

    def v2_05_status(condition):
        return "PASS" if condition else "FAIL"

    def v2_05_safe_div(num, den, default=0.0):
        if den == 0:
            return default
        return num / den

    def v2_05_bar_color(colors, condition):
        return colors["GreenLine"] if condition else colors["RedLine"]

    def v2_05_table(rows, columns, caption=""):
        header = "".join(f"<th>{v2_05_escape(label)}</th>" for label, _key in columns)
        body = []
        for row in rows:
            cells = []
            for _label, key in columns:
                value = row.get(key, "")
                cells.append(f"<td>{v2_05_escape(value)}</td>")
            body.append(f"<tr>{''.join(cells)}</tr>")
        caption_html = f"<div class='v2-05-table-caption'>{v2_05_escape(caption)}</div>" if caption else ""
        return mo.Html(
            f"""
<div style="overflow-x:auto; margin:12px 0;">
  {caption_html}
  <table class="v2-05-table">
    <thead><tr>{header}</tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</div>
<style>
.v2-05-table {{
  border-collapse: collapse;
  min-width: 780px;
  width: 100%;
  font-size: 0.9rem;
}}
.v2-05-table td, .v2-05-table th {{
  border: 1px solid #d9dee8;
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}}
.v2-05-table th {{
  color: #344054;
  font-weight: 700;
  background: #f8fafc;
}}
.v2-05-table-caption {{
  font-weight: 700;
  margin-bottom: 8px;
}}
</style>
"""
        )

    def v2_05_callout(ok, pass_text, fail_text):
        return mo.callout(mo.md(pass_text if ok else fail_text), kind="success" if ok else "danger")

    def v2_05_prediction_feedback(predicted, actual, correct_text, miss_text):
        if predicted is None:
            return mo.callout(
                mo.md("Commit to a structured prediction before using the instrument."),
                kind="warn",
            )
        if predicted == actual:
            return mo.callout(mo.md(correct_text), kind="success")
        return mo.callout(mo.md(miss_text), kind="warn")

    def v2_05_model_constants():
        return {
            "gpt2_b": v2_05_qty_to_float(Models.Language.GPT2.parameters, "param", 1.5e9) / 1e9,
            "llama2_7b": v2_05_qty_to_float(Models.Language.Llama2_7B.parameters, "param", 7.0e9) / 1e9,
            "gpt3_b": v2_05_qty_to_float(Models.Language.GPT3.parameters, "param", 175.0e9) / 1e9,
            "a100_gib": v2_05_qty_to_float(Hardware.Cloud.A100.memory.capacity, "GiB", 80.0),
            "h100_gib": v2_05_qty_to_float(Hardware.Cloud.H100.memory.capacity, "GiB", 80.0),
            "h100_nvlink_gb_s": v2_05_qty_to_float(Hardware.Cloud.H100.nvlink.bandwidth_per_direction, "GB/s", 450.0),
            "ib_ndr_gb_s": v2_05_qty_to_float(Systems.Fabrics.InfiniBand_NDR.bandwidth, "GB/s", 50.0),
            "ethernet_10g_gb_s": v2_05_qty_to_float(Systems.Fabrics.Ethernet_10G.bandwidth, "GB/s", 1.25),
            "gpus_per_node": int(getattr(Systems.Nodes.DGX_H100, "accelerators_per_node", 8)),
        }

    def v2_05_track_lenses():
        constants = v2_05_model_constants()
        return {
            "iphone": {
                "scenario": "A mobile ML product lead must decide where personalization training happens for an on-device feature.",
                "stakeholder": "mobile ML product lead",
                "decision_frame": "Keep the phone responsive while using consented evidence to improve the model.",
                "training_location": "backend fine-tune with on-device adapter update",
                "model_name": "1.5B personalization backbone",
                "model_b": constants["gpt2_b"],
                "memory_cap_gb": 6.0,
                "activation_base_gb": 2.5,
                "default_strategy": "adapter_offdevice",
                "default_scale": 8,
                "max_scale": 64,
                "part_b_default_gpus": 8,
                "part_b_max_gpus": 64,
                "step_budget_ms": 1800.0,
                "time_target_hours": 8.0,
                "baseline_hours": 54.0,
                "efficiency_floor": 0.62,
                "comm_share_limit": 0.30,
                "bubble_limit": 0.18,
                "critical_batch": 16384,
                "local_batch_default": 2,
                "local_batch_max": 16,
                "accum_default": 8,
                "evidence_floor": 0.82,
                "failure_mode": "local memory, battery/radio, or privacy evidence miss",
                "rejected_alternative": "full on-device training",
                "report_prompt": "Frame the memo around where personalization happens and what evidence leaves the phone.",
                "collective_implication": "V2-06 should treat phone-originated updates as staged summaries before any backend AllReduce.",
            },
            "oura_ring": {
                "scenario": "A TinyML firmware lead must package training updates for a wearable with intermittent sync.",
                "stakeholder": "TinyML firmware lead",
                "decision_frame": "Train off-device while keeping the wearable update small enough for flash, SRAM, and duty cycle.",
                "training_location": "off-device training plus tiny calibration package",
                "model_name": "750M sleep-stage teacher with tiny deployed student",
                "model_b": 0.75,
                "memory_cap_gb": 0.45,
                "activation_base_gb": 0.9,
                "default_strategy": "adapter_offdevice",
                "default_scale": 4,
                "max_scale": 32,
                "part_b_default_gpus": 4,
                "part_b_max_gpus": 32,
                "step_budget_ms": 2400.0,
                "time_target_hours": 12.0,
                "baseline_hours": 36.0,
                "efficiency_floor": 0.58,
                "comm_share_limit": 0.22,
                "bubble_limit": 0.16,
                "critical_batch": 8192,
                "local_batch_default": 1,
                "local_batch_max": 8,
                "accum_default": 8,
                "evidence_floor": 0.88,
                "failure_mode": "SRAM, OTA, or intermittent-sync miss",
                "rejected_alternative": "ring-local full training",
                "report_prompt": "Frame the memo around off-device training and the update artifact that can reach the ring.",
                "collective_implication": "V2-06 should assume wearable updates are asynchronous and delay tolerant, not synchronous collectives.",
            },
            "robotaxi": {
                "scenario": "A safety/perception platform lead must turn fleet evidence into a central training run without weakening validation.",
                "stakeholder": "safety/perception platform lead",
                "decision_frame": "Use fleet data centrally while preserving vehicle-local validation and release evidence.",
                "training_location": "central fleet-data training with edge validation",
                "model_name": "7B perception foundation model",
                "model_b": constants["llama2_7b"],
                "memory_cap_gb": constants["h100_gib"],
                "activation_base_gb": 18.0,
                "default_strategy": "fsdp_hybrid",
                "default_scale": 32,
                "max_scale": 128,
                "part_b_default_gpus": 32,
                "part_b_max_gpus": 128,
                "step_budget_ms": 2200.0,
                "time_target_hours": 10.0,
                "baseline_hours": 210.0,
                "efficiency_floor": 0.70,
                "comm_share_limit": 0.34,
                "bubble_limit": 0.20,
                "critical_batch": 32768,
                "local_batch_default": 4,
                "local_batch_max": 32,
                "accum_default": 4,
                "evidence_floor": 0.92,
                "failure_mode": "safety validation evidence or training turnaround miss",
                "rejected_alternative": "raw scale-out without validation gate",
                "report_prompt": "Frame the memo around training throughput plus safety evidence before deployment.",
                "collective_implication": "V2-06 should inspect the AllReduce path for central training and the delayed evidence-upload path separately.",
            },
            "cloud_fleet": {
                "scenario": "A training platform owner must schedule a large-model run on a GPU fleet without wasting accelerators.",
                "stakeholder": "training platform owner",
                "decision_frame": "Choose the 3D/sharded strategy that meets time, HBM, and network guardrails.",
                "training_location": "multi-node GPU cluster",
                "model_name": "175B frontier language model",
                "model_b": constants["gpt3_b"],
                "memory_cap_gb": constants["h100_gib"],
                "activation_base_gb": 46.0,
                "default_strategy": "fsdp_hybrid",
                "default_scale": 64,
                "max_scale": 512,
                "part_b_default_gpus": 128,
                "part_b_max_gpus": 512,
                "step_budget_ms": 3200.0,
                "time_target_hours": 24.0,
                "baseline_hours": 6200.0,
                "efficiency_floor": 0.68,
                "comm_share_limit": 0.38,
                "bubble_limit": 0.24,
                "critical_batch": 1_000_000,
                "local_batch_default": 4,
                "local_batch_max": 32,
                "accum_default": 16,
                "evidence_floor": 0.86,
                "failure_mode": "HBM, communication wall, or pipeline bubble miss",
                "rejected_alternative": "pure data parallel scale-out",
                "report_prompt": "Frame the memo around the scheduled 3D/sharded plan and the bottleneck that remains.",
                "collective_implication": "V2-06 should focus on gradient AllReduce plus FSDP AllGather/ReduceScatter placement.",
            },
        }

    def v2_05_strategy_specs(lens):
        scale = int(lens["default_scale"])
        return {
            "local_full": {
                "label": "Full local/single-replica training",
                "family": "single",
                "dp": 1,
                "tp": 1,
                "pp": 1,
                "scale": 1,
                "evidence": 0.52,
                "note": "Useful baseline; often invalid for device tracks and large models.",
            },
            "data_parallel": {
                "label": "Pure data parallel",
                "family": "data",
                "dp": max(2, scale),
                "tp": 1,
                "pp": 1,
                "scale": max(2, scale),
                "evidence": 0.70,
                "note": "Replicates state and synchronizes gradients once per optimizer step.",
            },
            "tensor_parallel": {
                "label": "Tensor parallel",
                "family": "tensor",
                "dp": 1,
                "tp": max(2, min(8, scale)),
                "pp": 1,
                "scale": max(2, min(8, scale)),
                "evidence": 0.74,
                "note": "Splits layers but puts frequent AllReduce on NVLink-class links.",
            },
            "pipeline_parallel": {
                "label": "Pipeline parallel",
                "family": "pipeline",
                "dp": 1,
                "tp": 1,
                "pp": max(2, min(32, scale)),
                "scale": max(2, min(32, scale)),
                "evidence": 0.76,
                "note": "Splits layers and pays fill/drain bubble cost.",
            },
            "fsdp_hybrid": {
                "label": "FSDP plus tensor/pipeline hybrid",
                "family": "hybrid",
                "dp": max(1, scale // 8),
                "tp": min(8, max(2, scale)),
                "pp": max(1, min(16, scale // 8)),
                "scale": max(8, scale),
                "evidence": 0.88,
                "note": "Combines sharding and 3D placement to buy memory with multiple collective paths.",
            },
            "adapter_offdevice": {
                "label": "Off-device training plus adapter/update",
                "family": "adapter",
                "dp": max(2, min(16, scale)),
                "tp": 1,
                "pp": 1,
                "scale": max(2, min(16, scale)),
                "evidence": 0.90,
                "note": "Track device receives a small update while backend training uses staged data parallelism.",
            },
        }

    def v2_05_update_spec_scale(spec, scale):
        spec = dict(spec)
        scale = max(1, int(scale))
        family = spec["family"]
        if family == "data":
            spec.update({"dp": scale, "tp": 1, "pp": 1, "scale": scale})
        elif family == "tensor":
            tp = max(2, min(8, scale))
            spec.update({"dp": 1, "tp": tp, "pp": 1, "scale": tp})
        elif family == "pipeline":
            pp = max(2, min(32, scale))
            spec.update({"dp": 1, "tp": 1, "pp": pp, "scale": pp})
        elif family == "hybrid":
            tp = min(8, max(2, scale))
            pp = max(1, min(16, scale // max(tp, 1)))
            dp = max(1, scale // max(tp * pp, 1))
            spec.update({"dp": dp, "tp": tp, "pp": pp, "scale": max(1, dp * tp * pp)})
        elif family == "adapter":
            dp = max(2, min(16, scale))
            spec.update({"dp": dp, "tp": 1, "pp": 1, "scale": dp})
        else:
            spec.update({"dp": 1, "tp": 1, "pp": 1, "scale": 1})
        return spec

    def v2_05_ring_factor(n):
        n = max(1, int(n))
        return 0.0 if n == 1 else 2 * (n - 1) / n

    def v2_05_strategy_result(lens, strategy_id, scale, microbatches=None, bandwidth_gb_s=None):
        specs = v2_05_strategy_specs(lens)
        spec = v2_05_update_spec_scale(specs[strategy_id], scale)
        constants = v2_05_model_constants()
        model_b = float(lens["model_b"])
        state_gb_full = model_b * 16.0
        weight_gb = model_b * 2.0
        activation_gb = float(lens["activation_base_gb"])
        microbatches = int(microbatches or 16)
        bandwidth_gb_s = float(bandwidth_gb_s or constants["ib_ndr_gb_s"])
        family = spec["family"]

        if family == "single":
            memory_gb = state_gb_full + activation_gb
            comm_ms = 0.0
            bubble_pct = 0.0
            convergence_risk = 0.10
        elif family == "data":
            memory_gb = state_gb_full + activation_gb
            comm_gb = weight_gb * v2_05_ring_factor(spec["dp"])
            comm_ms = comm_gb / max(bandwidth_gb_s, 0.1) * 1000.0
            bubble_pct = 0.0
            convergence_risk = min(0.55, spec["dp"] / max(lens["critical_batch"] / 256.0, 1.0))
        elif family == "tensor":
            memory_gb = state_gb_full / spec["tp"] + activation_gb / max(spec["tp"], 1)
            comm_gb = activation_gb * 0.45 * spec["tp"]
            comm_ms = comm_gb / max(constants["h100_nvlink_gb_s"], 0.1) * 1000.0
            bubble_pct = 0.0
            convergence_risk = 0.12
        elif family == "pipeline":
            memory_gb = state_gb_full / spec["pp"] + activation_gb / max(spec["pp"], 1)
            comm_gb = activation_gb * 0.35 * max(spec["pp"] - 1, 1) / max(spec["pp"], 1)
            comm_ms = comm_gb / max(bandwidth_gb_s, 0.1) * 1000.0
            bubble_pct = (spec["pp"] - 1) / (microbatches + spec["pp"] - 1)
            convergence_risk = 0.16
        elif family == "hybrid":
            shards = max(1, spec["dp"] * spec["tp"] * spec["pp"])
            memory_gb = state_gb_full / shards + activation_gb / max(spec["pp"], 1) + weight_gb / max(spec["tp"], 1) * 0.15
            dp_comm = weight_gb * v2_05_ring_factor(spec["dp"]) / max(bandwidth_gb_s, 0.1) * 1000.0
            fsdp_comm = weight_gb * 1.5 / max(bandwidth_gb_s, 0.1) * 1000.0
            tp_comm = activation_gb * 0.25 * spec["tp"] / max(constants["h100_nvlink_gb_s"], 0.1) * 1000.0
            comm_ms = 0.35 * dp_comm + 0.20 * fsdp_comm + tp_comm
            bubble_pct = (spec["pp"] - 1) / (microbatches + spec["pp"] - 1) if spec["pp"] > 1 else 0.0
            convergence_risk = 0.18
        else:
            memory_gb = min(lens["memory_cap_gb"] * 0.35, 1.2)
            comm_ms = 80.0 + 6.0 * model_b
            bubble_pct = 0.0
            convergence_risk = 0.20

        memory_ok = memory_gb <= lens["memory_cap_gb"]
        comm_ok = comm_ms <= lens["step_budget_ms"] * lens["comm_share_limit"]
        bubble_ok = bubble_pct <= lens["bubble_limit"]
        ratios = {
            "memory": v2_05_safe_div(memory_gb, lens["memory_cap_gb"], 99.0),
            "communication": v2_05_safe_div(comm_ms, lens["step_budget_ms"] * lens["comm_share_limit"], 99.0),
            "pipeline bubble": v2_05_safe_div(bubble_pct, lens["bubble_limit"], 99.0),
            "convergence": convergence_risk,
        }
        binding_amount = max(ratios.items(), key=lambda item: item[1])[0]
        if family == "adapter" and memory_ok and comm_ok:
            binding_amount = "evidence/update communication"
        return {
            **spec,
            "strategy_id": strategy_id,
            "memory_gb": memory_gb,
            "state_gb_full": state_gb_full,
            "comm_ms": comm_ms,
            "bubble_pct": bubble_pct,
            "convergence_risk": convergence_risk,
            "memory_ok": memory_ok,
            "comm_ok": comm_ok,
            "bubble_ok": bubble_ok,
            "feasible": memory_ok and comm_ok and bubble_ok,
            "binding_amount": binding_amount,
            "ratios": ratios,
        }

    def v2_05_network_options():
        constants = v2_05_model_constants()
        return {
            "nvlink": {
                "label": "NVLink island",
                "bandwidth_gb_s": constants["h100_nvlink_gb_s"],
                "latency_ms": 0.004,
                "overlap": 0.70,
            },
            "ib_ndr": {
                "label": "InfiniBand NDR",
                "bandwidth_gb_s": constants["ib_ndr_gb_s"],
                "latency_ms": 0.030,
                "overlap": 0.55,
            },
            "ethernet_10g": {
                "label": "10G Ethernet / staged edge",
                "bandwidth_gb_s": constants["ethernet_10g_gb_s"],
                "latency_ms": 0.600,
                "overlap": 0.15,
            },
        }

    def v2_05_step_for_n(lens, strategy_id, n_gpus, network_id, microbatches):
        network = v2_05_network_options()[network_id]
        strategy = v2_05_strategy_result(
            lens,
            strategy_id,
            n_gpus,
            microbatches=microbatches,
            bandwidth_gb_s=network["bandwidth_gb_s"],
        )
        n_gpus = max(1, int(n_gpus))
        compute_ms = lens["step_budget_ms"] * 0.72 / n_gpus
        exposed_comm_ms = strategy["comm_ms"] * (1 - network["overlap"])
        sync_ms = network["latency_ms"] * max(1.0, math.log2(n_gpus + 1)) * (1 + 0.05 * n_gpus)
        bubble_ms = compute_ms * strategy["bubble_pct"]
        step_ms = compute_ms + exposed_comm_ms + sync_ms + bubble_ms
        t_compute = lens["step_budget_ms"] * 0.72
        efficiency = v2_05_safe_div(t_compute, n_gpus * step_ms, 0.0)
        useful_speedup = n_gpus * efficiency
        comm_share = v2_05_safe_div(exposed_comm_ms + sync_ms, step_ms, 0.0)
        bubble_share = v2_05_safe_div(bubble_ms, step_ms, 0.0)
        return {
            **strategy,
            "n_gpus": n_gpus,
            "network_id": network_id,
            "network_label": network["label"],
            "compute_ms": compute_ms,
            "exposed_comm_ms": exposed_comm_ms,
            "sync_ms": sync_ms,
            "bubble_ms": bubble_ms,
            "step_ms": step_ms,
            "efficiency": efficiency,
            "useful_speedup": useful_speedup,
            "comm_share": comm_share,
            "bubble_share": bubble_share,
            "scaling_ok": efficiency >= lens["efficiency_floor"] and bubble_share <= lens["bubble_limit"],
        }

    def v2_05_sharding_options():
        return {
            "ddp": {"label": "DDP replicated state", "bpp": 16.0, "overhead": 0.00, "evidence": 0.72},
            "zero1": {"label": "ZeRO-1 optimizer sharding", "bpp": 7.0, "overhead": 0.04, "evidence": 0.78},
            "zero2": {"label": "ZeRO-2 grad + optimizer sharding", "bpp": 4.0, "overhead": 0.08, "evidence": 0.83},
            "zero3": {"label": "FSDP / ZeRO-3 full sharding", "bpp": 1.2, "overhead": 0.18, "evidence": 0.90},
        }

    def v2_05_batch_result(lens, n_gpus, batch_per_device, accumulation, sharding_id, step_reference_ms):
        sharding = v2_05_sharding_options()[sharding_id]
        model_b = lens["model_b"]
        state_gb = model_b * sharding["bpp"]
        activation_gb = lens["activation_base_gb"] * (batch_per_device / max(lens["local_batch_default"], 1)) * 0.45
        memory_gb = state_gb + activation_gb
        global_batch = int(n_gpus) * int(batch_per_device) * int(accumulation)
        batch_ratio = global_batch / lens["critical_batch"]
        convergence_ok = batch_ratio <= 1.0
        memory_ok = memory_gb <= lens["memory_cap_gb"]
        step_ms = step_reference_ms * (1 + sharding["overhead"]) * (1 + 0.015 * max(accumulation - 1, 0))
        step_ok = step_ms <= lens["step_budget_ms"] * 1.35
        if not memory_ok:
            binding = "memory"
        elif not convergence_ok:
            binding = "convergence"
        elif not step_ok:
            binding = "step time"
        else:
            binding = "optimizer/evidence headroom"
        return {
            "sharding_id": sharding_id,
            "sharding_label": sharding["label"],
            "batch_per_device": int(batch_per_device),
            "accumulation": int(accumulation),
            "global_batch": global_batch,
            "critical_batch": lens["critical_batch"],
            "batch_ratio": batch_ratio,
            "state_gb": state_gb,
            "activation_gb": activation_gb,
            "memory_gb": memory_gb,
            "memory_ok": memory_ok,
            "convergence_ok": convergence_ok,
            "step_ms": step_ms,
            "step_ok": step_ok,
            "binding_amount": binding,
            "evidence": sharding["evidence"],
        }

    def v2_05_plan_result(lens, candidate_id, strategy_id, scale, network_id, microbatches, batch_result, overlap_pct, evidence_mode):
        evidence_thresholds = {"minimum": 0.76, "release": lens["evidence_floor"], "audit": min(0.96, lens["evidence_floor"] + 0.05)}
        candidate_adjust = {
            "naive_scaleout": {"strategy": "data_parallel", "scale_mult": 2.0, "evidence_delta": -0.16, "name": "Naive pure data-parallel scale-out"},
            "student_strategy": {"strategy": strategy_id, "scale_mult": 1.0, "evidence_delta": 0.00, "name": "Student-selected Part A strategy"},
            "memory_first_fsdp": {"strategy": "fsdp_hybrid", "scale_mult": 1.0, "evidence_delta": 0.03, "name": "Memory-first FSDP/hybrid plan"},
            "bandwidth_matched": {"strategy": "fsdp_hybrid" if lens["model_b"] >= 7 else "adapter_offdevice", "scale_mult": 0.75, "evidence_delta": 0.08, "name": "Bandwidth-matched staged plan"},
        }[candidate_id]
        candidate_scale = max(1, int(scale * candidate_adjust["scale_mult"]))
        step = v2_05_step_for_n(lens, candidate_adjust["strategy"], candidate_scale, network_id, microbatches)
        exposed_step_ms = step["step_ms"] * (1 - float(overlap_pct) / 100.0)
        training_hours = lens["baseline_hours"] / max(step["useful_speedup"], 0.1)
        memory_ok = batch_result["memory_ok"] if candidate_id != "naive_scaleout" else step["memory_ok"]
        time_ok = training_hours <= lens["time_target_hours"]
        comm_ok = step["comm_share"] <= lens["comm_share_limit"] and step["efficiency"] >= lens["efficiency_floor"]
        evidence_score = max(0.0, min(1.0, step["evidence"] + candidate_adjust["evidence_delta"]))
        evidence_ok = evidence_score >= evidence_thresholds[evidence_mode]
        ratios = {
            "time": v2_05_safe_div(training_hours, lens["time_target_hours"], 99.0),
            "memory": 0.80 if memory_ok else 1.35,
            "communication": v2_05_safe_div(step["comm_share"], lens["comm_share_limit"], 99.0),
            "evidence": v2_05_safe_div(evidence_thresholds[evidence_mode], max(evidence_score, 0.01), 99.0),
        }
        binding = max(ratios.items(), key=lambda item: item[1])[0]
        return {
            **step,
            "candidate_id": candidate_id,
            "candidate_label": candidate_adjust["name"],
            "candidate_strategy": candidate_adjust["strategy"],
            "candidate_scale": candidate_scale,
            "exposed_step_ms": exposed_step_ms,
            "training_hours": training_hours,
            "time_ok": time_ok,
            "memory_ok": memory_ok,
            "comm_ok": comm_ok,
            "evidence_score": evidence_score,
            "evidence_threshold": evidence_thresholds[evidence_mode],
            "evidence_ok": evidence_ok,
            "valid_plan": time_ok and memory_ok and comm_ok and evidence_ok,
            "binding_guardrail": binding,
        }

    def v2_05_strategy_chart(colors, result):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["memory ratio", "comm ratio", "bubble ratio", "convergence risk"],
                y=[
                    result["ratios"]["memory"],
                    result["ratios"]["communication"],
                    result["ratios"]["pipeline bubble"],
                    result["ratios"]["convergence"],
                ],
                marker_color=[
                    v2_05_bar_color(colors, result["memory_ok"]),
                    v2_05_bar_color(colors, result["comm_ok"]),
                    v2_05_bar_color(colors, result["bubble_ok"]),
                    colors["OrangeLine"],
                ],
                text=[
                    f"{result['ratios']['memory']:.2f}x",
                    f"{result['ratios']['communication']:.2f}x",
                    f"{result['ratios']['pipeline bubble']:.2f}x",
                    f"{result['ratios']['convergence']:.2f}",
                ],
                textposition="auto",
            )
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color=colors["RedLine"], annotation_text="failure boundary")
        fig.update_layout(
            title="Part A - shifted amount ratios",
            yaxis_title="ratio to track limit",
            xaxis_title="amount system term",
            height=360,
            showlegend=False,
        )
        apply_plotly_theme(fig)
        return fig

    def v2_05_scaling_chart(colors, rows, current_n):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[row["n_gpus"] for row in rows],
                y=[row["efficiency"] for row in rows],
                mode="lines+markers",
                name="scaling efficiency",
                line={"color": colors["BlueLine"]},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["n_gpus"] for row in rows],
                y=[row["comm_share"] for row in rows],
                mode="lines+markers",
                name="communication share",
                line={"color": colors["OrangeLine"]},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["n_gpus"] for row in rows],
                y=[row["bubble_share"] for row in rows],
                mode="lines+markers",
                name="bubble share",
                line={"color": colors.get("VioletLine", colors["BlueLine"])},
            )
        )
        fig.add_vline(x=current_n, line_dash="dot", line_color=colors["Text"], annotation_text="current")
        fig.update_layout(
            title="Part B - efficiency falls as overhead takes over",
            xaxis_title="accelerators",
            yaxis_title="fraction of ideal or step",
            height=380,
        )
        apply_plotly_theme(fig)
        return fig

    def v2_05_memory_chart(colors, batch_result, lens):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["state", "activations", "headroom"],
                y=[
                    batch_result["state_gb"],
                    batch_result["activation_gb"],
                    max(lens["memory_cap_gb"] - batch_result["memory_gb"], 0.0),
                ],
                marker_color=[colors["BlueLine"], colors["OrangeLine"], colors["GreenLine"]],
                text=[
                    f"{batch_result['state_gb']:.1f} GB",
                    f"{batch_result['activation_gb']:.1f} GB",
                    f"{max(lens['memory_cap_gb'] - batch_result['memory_gb'], 0.0):.1f} GB",
                ],
                textposition="auto",
            )
        )
        fig.add_hline(y=lens["memory_cap_gb"], line_dash="dash", line_color=colors["RedLine"], annotation_text="device cap")
        fig.update_layout(
            title="Part C - per-device memory ledger",
            yaxis_title="GB",
            height=360,
            showlegend=False,
        )
        apply_plotly_theme(fig)
        return fig

    def v2_05_plan_chart(colors, selected, rejected):
        labels = ["time", "memory", "communication", "evidence"]
        selected_values = [
            selected["training_hours"] / selected["time_target_hours"] if "time_target_hours" in selected else 0,
            0.80 if selected["memory_ok"] else 1.30,
            selected["comm_share"] / max(selected["comm_share_limit"] if "comm_share_limit" in selected else 1, 0.01),
            selected["evidence_threshold"] / max(selected["evidence_score"], 0.01),
        ]
        rejected_values = [
            rejected["training_hours"] / max(rejected.get("time_target_hours", selected.get("time_target_hours", 1)), 1),
            0.80 if rejected["memory_ok"] else 1.30,
            rejected["comm_share"] / max(rejected.get("comm_share_limit", selected.get("comm_share_limit", 1)), 0.01),
            rejected["evidence_threshold"] / max(rejected["evidence_score"], 0.01),
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=selected_values, name="selected", marker_color=colors["GreenLine"]))
        fig.add_trace(go.Bar(x=labels, y=rejected_values, name="rejected", marker_color=colors["RedLine"]))
        fig.add_hline(y=1.0, line_dash="dash", line_color=colors["Text"], annotation_text="guardrail")
        fig.update_layout(
            title="Part D - simultaneous guardrail ratios",
            yaxis_title="ratio to limit",
            barmode="group",
            height=380,
        )
        apply_plotly_theme(fig)
        return fig

    return (
        v2_05_batch_result,
        v2_05_callout,
        v2_05_escape,
        v2_05_fmt,
        v2_05_memory_chart,
        v2_05_network_options,
        v2_05_pct,
        v2_05_plan_chart,
        v2_05_plan_result,
        v2_05_prediction_feedback,
        v2_05_scaling_chart,
        v2_05_status,
        v2_05_strategy_chart,
        v2_05_strategy_result,
        v2_05_strategy_specs,
        v2_05_step_for_n,
        v2_05_table,
        v2_05_track_lenses,
    )


@app.cell
def _(v2_05_profile, v2_05_track_lenses):
    _lenses = v2_05_track_lenses()
    v2_05_lens = _lenses.get(v2_05_profile.track_id, _lenses["cloud_fleet"])
    return (v2_05_lens,)


@app.cell
def _(mo, v2_05_lens, v2_05_network_options, v2_05_strategy_specs):
    _strategy_options = {value["label"]: key for key, value in v2_05_strategy_specs(v2_05_lens).items()}
    _network_options = {value["label"]: key for key, value in v2_05_network_options().items()}
    _sharding_options = {
        "DDP replicated state": "ddp",
        "ZeRO-1 optimizer sharding": "zero1",
        "ZeRO-2 grad + optimizer sharding": "zero2",
        "FSDP / ZeRO-3 full sharding": "zero3",
    }
    _candidate_options = {
        "Naive pure data-parallel scale-out": "naive_scaleout",
        "Student-selected Part A strategy": "student_strategy",
        "Memory-first FSDP/hybrid plan": "memory_first_fsdp",
        "Bandwidth-matched staged plan": "bandwidth_matched",
    }

    partA_prediction = mo.ui.radio(
        options={
            "memory": "Memory becomes binding",
            "communication": "Communication becomes binding",
            "pipeline bubble": "Pipeline bubbles become binding",
            "evidence/update communication": "Evidence or update traffic becomes binding",
        },
        label="Prediction: which amount will the first strategy shift into?",
    )
    partA_strategy = mo.ui.dropdown(
        options=_strategy_options,
        value=v2_05_strategy_specs(v2_05_lens)[v2_05_lens["default_strategy"]]["label"],
        label="Strategy to test",
    )
    partA_scale = mo.ui.slider(
        start=1,
        stop=int(v2_05_lens["max_scale"]),
        step=1,
        value=int(v2_05_lens["default_scale"]),
        label="parallel workers/stages",
    )
    partA_checkpoint = mo.ui.radio(
        options={
            "carry": "Carry this strategy forward",
            "revise": "Revise before scaling",
            "reject": "Reject for this track",
        },
        label="Checkpoint decision",
    )

    partB_prediction = mo.ui.radio(
        options={
            "communication": "Communication wall",
            "pipeline bubble": "Pipeline bubble",
            "synchronization": "Synchronization latency",
            "compute": "Compute remains dominant",
        },
        label="Prediction: which term drops scaling efficiency first?",
    )
    partB_gpus = mo.ui.slider(
        start=1,
        stop=int(v2_05_lens["part_b_max_gpus"]),
        step=1,
        value=int(v2_05_lens["part_b_default_gpus"]),
        label="accelerators",
    )
    partB_network = mo.ui.dropdown(options=_network_options, value="InfiniBand NDR", label="communication tier")
    partB_microbatches = mo.ui.slider(start=1, stop=64, step=1, value=16, label="pipeline microbatches")
    partB_checkpoint = mo.ui.radio(
        options={
            "scale": "Scale out",
            "accumulate": "Stay smaller and accumulate",
            "change_parallelism": "Change parallelism mix",
        },
        label="Checkpoint decision",
    )

    partC_prediction = mo.ui.radio(
        options={
            "memory": "Optimizer state or activations exceed memory",
            "convergence": "Global batch exceeds useful convergence range",
            "step time": "Sharding overhead pushes step time over budget",
            "optimizer/evidence headroom": "All constraints keep headroom",
        },
        label="Prediction: what binds after batch and sharding choices?",
    )
    partC_batch = mo.ui.slider(
        start=1,
        stop=int(v2_05_lens["local_batch_max"]),
        step=1,
        value=int(v2_05_lens["local_batch_default"]),
        label="per-device batch",
    )
    partC_accumulation = mo.ui.slider(
        start=1,
        stop=64,
        step=1,
        value=int(v2_05_lens["accum_default"]),
        label="gradient accumulation steps",
    )
    partC_sharding = mo.ui.dropdown(options=_sharding_options, value="ZeRO-2 grad + optimizer sharding", label="optimizer/sharding policy")
    partC_checkpoint = mo.ui.radio(
        options={
            "keep": "Keep batch and sharding",
            "reduce_batch": "Reduce batch/accumulation",
            "deepen_shard": "Increase sharding",
        },
        label="Checkpoint decision",
    )

    partD_prediction = mo.ui.radio(
        options={
            "time": "Time-to-train rejects the naive plan",
            "memory": "Memory rejects the naive plan",
            "communication": "Communication efficiency rejects the naive plan",
            "evidence": "Evidence threshold rejects the naive plan",
        },
        label="Prediction: which guardrail rejects the naive plan?",
    )
    partD_candidate = mo.ui.dropdown(options=_candidate_options, value="Bandwidth-matched staged plan", label="candidate plan")
    partD_overlap = mo.ui.slider(start=0, stop=80, step=5, value=35, label="extra communication overlap (%)")
    partD_evidence = mo.ui.dropdown(
        options={
            "Minimum lab evidence": "minimum",
            "Release review evidence": "release",
            "Audit-grade evidence": "audit",
        },
        value="Release review evidence",
        label="evidence threshold",
    )
    partD_final = mo.ui.radio(
        options={
            "approve": "Approve selected plan",
            "revise": "Revise and remeasure",
            "reject": "Reject for this track",
        },
        label="Final decision",
    )

    student_id = mo.ui.text(label="Optional student/team id")
    memo_note = mo.ui.text_area(
        label="Memo note",
        placeholder="Selected parallelism, binding bottleneck, rejected alternative, and V2-06 implication.",
    )
    return (
        memo_note,
        partA_checkpoint,
        partA_prediction,
        partA_scale,
        partA_strategy,
        partB_checkpoint,
        partB_gpus,
        partB_microbatches,
        partB_network,
        partB_prediction,
        partC_accumulation,
        partC_batch,
        partC_checkpoint,
        partC_prediction,
        partC_sharding,
        partD_candidate,
        partD_evidence,
        partD_final,
        partD_overlap,
        partD_prediction,
        student_id,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    COLORS,
    LAB_CSS,
    MathPeek,
    build_lab_report,
    ledger,
    memo_note,
    mo,
    partA_checkpoint,
    partA_prediction,
    partA_scale,
    partA_strategy,
    partB_checkpoint,
    partB_gpus,
    partB_microbatches,
    partB_network,
    partB_prediction,
    partC_accumulation,
    partC_batch,
    partC_checkpoint,
    partC_prediction,
    partC_sharding,
    partD_candidate,
    partD_evidence,
    partD_final,
    partD_overlap,
    partD_prediction,
    report_export_panel,
    source_trace,
    student_id,
    track_arc_context,
    track_context,
    v2_05_batch_result,
    v2_05_callout,
    v2_05_chapter,
    v2_05_fmt,
    v2_05_lens,
    v2_05_memory_chart,
    v2_05_metadata,
    v2_05_network_options,
    v2_05_pct,
    v2_05_plan_chart,
    v2_05_plan_result,
    v2_05_prediction_feedback,
    v2_05_profile,
    v2_05_scaling_chart,
    v2_05_status,
    v2_05_strategy_chart,
    v2_05_strategy_result,
    v2_05_strategy_specs,
    v2_05_step_for_n,
    v2_05_table,
    v2_05_variant,
):
    _network = v2_05_network_options()[partB_network.value]
    _part_a = v2_05_strategy_result(
        v2_05_lens,
        partA_strategy.value,
        partA_scale.value,
        microbatches=partB_microbatches.value,
        bandwidth_gb_s=_network["bandwidth_gb_s"],
    )
    _part_b = v2_05_step_for_n(
        v2_05_lens,
        partA_strategy.value,
        partB_gpus.value,
        partB_network.value,
        partB_microbatches.value,
    )
    _part_c = v2_05_batch_result(
        v2_05_lens,
        partB_gpus.value,
        partC_batch.value,
        partC_accumulation.value,
        partC_sharding.value,
        _part_b["step_ms"],
    )
    _part_d = v2_05_plan_result(
        v2_05_lens,
        partD_candidate.value,
        partA_strategy.value,
        partB_gpus.value,
        partB_network.value,
        partB_microbatches.value,
        _part_c,
        partD_overlap.value,
        partD_evidence.value,
    )
    _rejected = v2_05_plan_result(
        v2_05_lens,
        "naive_scaleout",
        partA_strategy.value,
        partB_gpus.value,
        partB_network.value,
        partB_microbatches.value,
        _part_c,
        0,
        partD_evidence.value,
    )
    _part_d["time_target_hours"] = v2_05_lens["time_target_hours"]
    _part_d["comm_share_limit"] = v2_05_lens["comm_share_limit"]
    _rejected["time_target_hours"] = v2_05_lens["time_target_hours"]
    _rejected["comm_share_limit"] = v2_05_lens["comm_share_limit"]

    _scale_values = sorted(
        set(
            [
                1,
                max(2, int(v2_05_lens["part_b_max_gpus"]) // 16),
                max(2, int(v2_05_lens["part_b_max_gpus"]) // 8),
                max(2, int(v2_05_lens["part_b_max_gpus"]) // 4),
                max(2, int(v2_05_lens["part_b_max_gpus"]) // 2),
                int(v2_05_lens["part_b_max_gpus"]),
                int(partB_gpus.value),
            ]
        )
    )
    _scaling_rows = [
        v2_05_step_for_n(v2_05_lens, partA_strategy.value, value, partB_network.value, partB_microbatches.value)
        for value in _scale_values
    ]

    def _part_a_table():
        rows = [
            {
                "Metric": "Strategy",
                "Value": _part_a["label"],
                "Limit or meaning": _part_a["note"],
            },
            {
                "Metric": "Per-device memory",
                "Value": f"{_part_a['memory_gb']:.1f} GB ({v2_05_status(_part_a['memory_ok'])})",
                "Limit or meaning": f"<= {v2_05_lens['memory_cap_gb']:.1f} GB",
            },
            {
                "Metric": "Communication time",
                "Value": f"{_part_a['comm_ms']:.1f} ms ({v2_05_status(_part_a['comm_ok'])})",
                "Limit or meaning": f"<= {v2_05_lens['step_budget_ms'] * v2_05_lens['comm_share_limit']:.0f} ms exposed budget",
            },
            {
                "Metric": "Pipeline bubble",
                "Value": f"{v2_05_pct(_part_a['bubble_pct'], 1)} ({v2_05_status(_part_a['bubble_ok'])})",
                "Limit or meaning": f"<= {v2_05_pct(v2_05_lens['bubble_limit'], 0)}",
            },
            {
                "Metric": "Binding amount",
                "Value": _part_a["binding_amount"],
                "Limit or meaning": v2_05_lens["failure_mode"],
            },
        ]
        return v2_05_table(
            rows,
            [("Metric", "Metric"), ("Value", "Value"), ("Limit or meaning", "Limit or meaning")],
            caption="Part A strategy evidence table",
        )

    def _part_b_table():
        rows = [
            {
                "Metric": "Accelerators",
                "Value": _part_b["n_gpus"],
                "Limit or meaning": _part_b["network_label"],
            },
            {
                "Metric": "Step time",
                "Value": f"{_part_b['step_ms']:.1f} ms",
                "Limit or meaning": "compute/N + exposed communication + synchronization + bubble",
            },
            {
                "Metric": "Scaling efficiency",
                "Value": f"{v2_05_pct(_part_b['efficiency'], 1)} ({v2_05_status(_part_b['efficiency'] >= v2_05_lens['efficiency_floor'])})",
                "Limit or meaning": f">= {v2_05_pct(v2_05_lens['efficiency_floor'], 0)}",
            },
            {
                "Metric": "Communication share",
                "Value": f"{v2_05_pct(_part_b['comm_share'], 1)}",
                "Limit or meaning": f"guardrail <= {v2_05_pct(v2_05_lens['comm_share_limit'], 0)}",
            },
            {
                "Metric": "Bubble share",
                "Value": f"{v2_05_pct(_part_b['bubble_share'], 1)}",
                "Limit or meaning": f"guardrail <= {v2_05_pct(v2_05_lens['bubble_limit'], 0)}",
            },
        ]
        return v2_05_table(
            rows,
            [("Metric", "Metric"), ("Value", "Value"), ("Limit or meaning", "Limit or meaning")],
            caption="Part B scaling evidence table",
        )

    def _part_c_table():
        rows = [
            {
                "Metric": "Sharding policy",
                "Value": _part_c["sharding_label"],
                "Limit or meaning": "memory is bought with additional collectives",
            },
            {
                "Metric": "Per-device memory",
                "Value": f"{_part_c['memory_gb']:.1f} GB ({v2_05_status(_part_c['memory_ok'])})",
                "Limit or meaning": f"<= {v2_05_lens['memory_cap_gb']:.1f} GB",
            },
            {
                "Metric": "Global batch",
                "Value": f"{_part_c['global_batch']:,} ({v2_05_status(_part_c['convergence_ok'])})",
                "Limit or meaning": f"critical batch ~= {_part_c['critical_batch']:,}",
            },
            {
                "Metric": "Critical-batch ratio",
                "Value": f"{_part_c['batch_ratio']:.2f}x",
                "Limit or meaning": "above 1.0 means diminishing convergence returns",
            },
            {
                "Metric": "Binding amount",
                "Value": _part_c["binding_amount"],
                "Limit or meaning": v2_05_lens["failure_mode"],
            },
        ]
        return v2_05_table(
            rows,
            [("Metric", "Metric"), ("Value", "Value"), ("Limit or meaning", "Limit or meaning")],
            caption="Part C memory, batch, and convergence table",
        )

    def _part_d_table(selected, rejected):
        rows = [
            {
                "Guardrail": "Time-to-train",
                "Selected plan": f"{selected['training_hours']:.1f} h ({v2_05_status(selected['time_ok'])})",
                "Rejected alternative": f"{rejected['training_hours']:.1f} h ({v2_05_status(rejected['time_ok'])})",
                "Limit": f"<= {v2_05_lens['time_target_hours']:.1f} h",
            },
            {
                "Guardrail": "Memory",
                "Selected plan": v2_05_status(selected["memory_ok"]),
                "Rejected alternative": v2_05_status(rejected["memory_ok"]),
                "Limit": f"<= {v2_05_lens['memory_cap_gb']:.1f} GB",
            },
            {
                "Guardrail": "Communication",
                "Selected plan": f"{v2_05_pct(selected['comm_share'], 1)} ({v2_05_status(selected['comm_ok'])})",
                "Rejected alternative": f"{v2_05_pct(rejected['comm_share'], 1)} ({v2_05_status(rejected['comm_ok'])})",
                "Limit": f"<= {v2_05_pct(v2_05_lens['comm_share_limit'], 0)} and efficiency >= {v2_05_pct(v2_05_lens['efficiency_floor'], 0)}",
            },
            {
                "Guardrail": "Evidence",
                "Selected plan": f"{selected['evidence_score']:.2f} ({v2_05_status(selected['evidence_ok'])})",
                "Rejected alternative": f"{rejected['evidence_score']:.2f} ({v2_05_status(rejected['evidence_ok'])})",
                "Limit": f">= {selected['evidence_threshold']:.2f}",
            },
        ]
        return v2_05_table(
            rows,
            [
                ("Guardrail", "Guardrail"),
                ("Selected plan", "Selected plan"),
                ("Rejected alternative", "Rejected alternative"),
                ("Limit", "Limit"),
            ],
            caption="Part D simultaneous guardrail matrix",
        )

    def build_opening():
        return mo.vstack(
            [
                LAB_CSS,
                ACADEMIC_LAB_CSS,
                mo.md("# V2-05 Distributed Training: The Parallelism Puzzle"),
                track_context(v2_05_profile),
                track_arc_context(v2_05_profile, v2_05_metadata.lab_id),
                mo.callout(
                    mo.md(
                        f"**Chapter invariant.** Distributed training trades compute for communication, "
                        f"memory, synchronization, and convergence cost. For **{v2_05_profile.label}**, "
                        f"you are acting as the {v2_05_lens['stakeholder']} deciding: "
                        f"{v2_05_lens['decision_frame']}"
                    ),
                    kind="info",
                ),
                mo.md(
                    f"""
**Scenario.** {v2_05_lens['scenario']}

**Track constraint lens.** Model/update workload: **{v2_05_lens['model_name']}**. Training location: **{v2_05_lens['training_location']}**. Natural failure: **{v2_05_lens['failure_mode']}**.

**Concept sequence.**

1. Part A: data/model/pipeline parallelism shifts the binding amount.
2. Part B: scaling efficiency falls when communication or bubbles dominate.
3. Part C: batch size and optimizer state change memory, convergence, and step time.
4. Part D: a distributed training plan must satisfy time, memory, communication, and evidence constraints.
"""
                ),
            ]
        )

    def build_part_a():
        _actual = _part_a["binding_amount"]
        items = [
            mo.md("## Part A - Concept Module: Parallelism Moves The Binding Amount"),
            mo.callout(
                mo.md(
                    f"**Scenario.** The {v2_05_lens['stakeholder']} tries **{_part_a['label']}** "
                    f"for **{v2_05_lens['model_name']}**. The decision is not whether more devices "
                    "exist; it is which amount becomes binding after the split."
                ),
                kind="info",
            ),
            partA_prediction,
            v2_05_prediction_feedback(
                partA_prediction.value,
                _actual,
                f"Correct. The modeled binding amount is **{_actual}**.",
                f"The modeled binding amount is **{_actual}**. Parallelism moved the pressure instead of removing it.",
            ),
        ]
        if partA_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([partA_strategy, partA_scale], justify="start"),
                mo.as_html(v2_05_strategy_chart(COLORS, _part_a)),
                _part_a_table(),
                v2_05_callout(
                    _part_a["feasible"],
                    f"Consequence: {_part_a['label']} stays inside the current memory, communication, and bubble thresholds.",
                    f"Boundary: {_part_a['label']} violates a track threshold. Binding amount: {_part_a['binding_amount']}. Mitigation is to change the split, reduce scale, or move training off the constrained endpoint.",
                ),
                MathPeek(
                    "N_total=d*p*t; each axis changes state placement and traffic",
                    {
                        "data parallel": "replicate model, shard data, AllReduce gradients",
                        "tensor parallel": "split layers, frequent intra-layer collectives",
                        "pipeline parallel": "split layer depth, pay bubble=(p-1)/(m+p-1)",
                        "chapter source": "3D parallelism cube and strategy decision tree",
                    },
                ),
                partA_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_part_b():
        if _part_b["comm_share"] >= max(_part_b["bubble_share"], 0.08):
            _actual = "communication"
        elif _part_b["bubble_share"] > 0.08:
            _actual = "pipeline bubble"
        elif _part_b["sync_ms"] > _part_b["compute_ms"] * 0.25:
            _actual = "synchronization"
        else:
            _actual = "compute"
        items = [
            mo.md("## Part B - Concept Module: Scaling Efficiency Falls When Overhead Dominates"),
            mo.callout(
                mo.md(
                    f"**Scenario.** The team asks whether **{_part_b['n_gpus']} accelerators** "
                    f"on **{_part_b['network_label']}** still buy useful work for {v2_05_profile.label}."
                ),
                kind="info",
            ),
            partB_prediction,
            v2_05_prediction_feedback(
                partB_prediction.value,
                _actual,
                f"Correct. The dominant scaling limiter is **{_actual}**.",
                f"The dominant scaling limiter is **{_actual}**. The fleet law shows why useful speedup diverges from raw GPU count.",
            ),
        ]
        if partB_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([partB_gpus, partB_network, partB_microbatches], justify="start"),
                mo.as_html(v2_05_scaling_chart(COLORS, _scaling_rows, _part_b["n_gpus"])),
                _part_b_table(),
                v2_05_callout(
                    _part_b["scaling_ok"],
                    f"Consequence: scaling efficiency is {v2_05_pct(_part_b['efficiency'], 1)}, above the {v2_05_pct(v2_05_lens['efficiency_floor'], 0)} track threshold.",
                    f"Boundary: scaling efficiency is {v2_05_pct(_part_b['efficiency'], 1)} against a {v2_05_pct(v2_05_lens['efficiency_floor'], 0)} threshold, or bubble share is too high. Mitigation is fewer workers, faster fabric, more microbatches, or a different parallelism axis.",
                ),
                MathPeek(
                    "T_step(N)=T_compute/N+T_comm(N)+T_sync(N)-T_overlap; eta=T_compute/(N*T_step)",
                    {
                        "compute term": f"{_part_b['compute_ms']:.1f} ms",
                        "exposed communication": f"{_part_b['exposed_comm_ms'] + _part_b['sync_ms']:.1f} ms",
                        "bubble term": f"{_part_b['bubble_ms']:.1f} ms",
                        "chapter source": "scaling efficiency and Amdahl with communication",
                    },
                ),
                partB_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_part_c():
        _actual = _part_c["binding_amount"]
        items = [
            mo.md("## Part C - Concept Module: Batch And Optimizer State Couple Memory, Convergence, And Step Time"),
            mo.callout(
                mo.md(
                    f"**Scenario.** The strategy now needs an optimizer setup. "
                    f"Per-device batch **{_part_c['batch_per_device']}**, accumulation **{_part_c['accumulation']}**, "
                    f"and **{_part_c['sharding_label']}** create a global batch of **{_part_c['global_batch']:,}**."
                ),
                kind="info",
            ),
            partC_prediction,
            v2_05_prediction_feedback(
                partC_prediction.value,
                _actual,
                f"Correct. The modeled binding amount is **{_actual}**.",
                f"The modeled binding amount is **{_actual}**. Memory fit, convergence, and step time are coupled.",
            ),
        ]
        if partC_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([partC_batch, partC_accumulation, partC_sharding], justify="start"),
                mo.as_html(v2_05_memory_chart(COLORS, _part_c, v2_05_lens)),
                _part_c_table(),
                v2_05_callout(
                    _part_c["memory_ok"] and _part_c["convergence_ok"] and _part_c["step_ok"],
                    f"Consequence: memory is {_part_c['memory_gb']:.1f} GB and critical-batch ratio is {_part_c['batch_ratio']:.2f}x.",
                    f"Boundary: {_part_c['binding_amount']} fails first. Memory {_part_c['memory_gb']:.1f} GB vs {v2_05_lens['memory_cap_gb']:.1f} GB, critical-batch ratio {_part_c['batch_ratio']:.2f}x, step {_part_c['step_ms']:.1f} ms.",
                ),
                MathPeek(
                    "M_state ~= params*bytes_per_param; B_global=N*b*accum; B* ~= tr(Sigma)/||grad L||^2",
                    {
                        "state": f"{_part_c['state_gb']:.1f} GB",
                        "activations": f"{_part_c['activation_gb']:.1f} GB",
                        "global batch": f"{_part_c['global_batch']:,}",
                        "chapter source": "ZeRO/FSDP memory and critical batch size sections",
                    },
                ),
                partC_checkpoint,
            ]
        )
        return mo.vstack(items)

    def build_part_d():
        _actual = _rejected["binding_guardrail"]
        items = [
            mo.md("## Part D - Concept Module: Training Plan Guardrails"),
            mo.callout(
                mo.md(
                    f"**Scenario.** The final plan must pass time, memory, communication, and evidence guardrails. "
                    f"{v2_05_lens['report_prompt']}"
                ),
                kind="info",
            ),
            partD_prediction,
            v2_05_prediction_feedback(
                partD_prediction.value,
                _actual,
                f"Correct. The rejected alternative is primarily blocked by **{_actual}**.",
                f"The rejected alternative is primarily blocked by **{_actual}**. A throughput-looking plan still needs every guardrail.",
            ),
        ]
        if partD_prediction.value is None:
            return mo.vstack(items)
        items.extend(
            [
                mo.hstack([partD_candidate, partD_overlap, partD_evidence], justify="start"),
                mo.as_html(v2_05_plan_chart(COLORS, _part_d, _rejected)),
                _part_d_table(_part_d, _rejected),
                v2_05_callout(
                    _part_d["valid_plan"],
                    f"Consequence: selected plan passes all guardrails. Binding guardrail is {_part_d['binding_guardrail']}.",
                    f"Boundary: selected plan fails at least one guardrail. Binding guardrail is {_part_d['binding_guardrail']}; revise parallelism, evidence threshold, or overlap.",
                ),
                MathPeek(
                    "valid=time_ok and memory_ok and communication_ok and evidence_ok",
                    {
                        "time": f"{_part_d['training_hours']:.1f} h vs {v2_05_lens['time_target_hours']:.1f} h",
                        "communication": f"{v2_05_pct(_part_d['comm_share'], 1)} vs {v2_05_pct(v2_05_lens['comm_share_limit'], 0)}",
                        "evidence": f"{_part_d['evidence_score']:.2f} vs {_part_d['evidence_threshold']:.2f}",
                        "V2-06 implication": v2_05_lens["collective_implication"],
                    },
                ),
                partD_final,
            ]
        )
        return mo.vstack(items)

    def build_synthesis():
        _completed = all(
            value is not None
            for value in (
                partA_prediction.value,
                partA_checkpoint.value,
                partB_prediction.value,
                partB_checkpoint.value,
                partC_prediction.value,
                partC_checkpoint.value,
                partD_prediction.value,
                partD_final.value,
            )
        )
        _binding = _part_d["binding_guardrail"] if not _part_d["valid_plan"] else _part_c["binding_amount"]
        _memo = memo_note.value or (
            f"Select {_part_d['candidate_label']} for {v2_05_profile.label}; "
            f"binding bottleneck: {_binding}; reject {v2_05_lens['rejected_alternative']}; "
            f"{v2_05_lens['collective_implication']}"
        )
        _snapshot = {
            "track_id": v2_05_profile.track_id,
            "scenario_id": v2_05_variant.scenario_id,
            "selected_parallelism": _part_d["candidate_label"],
            "training_location": v2_05_lens["training_location"],
            "binding_bottleneck": _binding,
            "memory_per_device_gb": round(_part_c["memory_gb"], 3),
            "scaling_efficiency": round(_part_b["efficiency"], 4),
            "critical_batch_ratio": round(_part_c["batch_ratio"], 4),
            "rejected_alternative": v2_05_lens["rejected_alternative"],
            "collective_implication": v2_05_lens["collective_implication"],
            "completed": _completed,
        }
        _design = {
            "lab_id": v2_05_metadata.lab_id,
            "track_id": v2_05_profile.track_id,
            "scenario_id": v2_05_variant.scenario_id,
            "selected_parallelism": _part_d["candidate_label"],
            "training_location": v2_05_lens["training_location"],
            "binding_bottleneck": _binding,
            "memory_per_device_gb": _part_c["memory_gb"],
            "scaling_efficiency": _part_b["efficiency"],
            "critical_batch_ratio": _part_c["batch_ratio"],
            "rejected_alternative": v2_05_lens["rejected_alternative"],
            "collective_implication": v2_05_lens["collective_implication"],
            "completed": _completed,
            "result_snapshot": _snapshot,
        }
        ledger.save(track=v2_05_profile.track_id, chapter=v2_05_chapter, design=_design)

        _incomplete = []
        if partA_prediction.value is None:
            _incomplete.append("Part A binding prediction")
        if partA_checkpoint.value is None:
            _incomplete.append("Part A strategy checkpoint")
        if partB_prediction.value is None:
            _incomplete.append("Part B scaling prediction")
        if partB_checkpoint.value is None:
            _incomplete.append("Part B scaling checkpoint")
        if partC_prediction.value is None:
            _incomplete.append("Part C batch/sharding prediction")
        if partC_checkpoint.value is None:
            _incomplete.append("Part C optimizer checkpoint")
        if partD_prediction.value is None:
            _incomplete.append("Part D guardrail prediction")
        if partD_final.value is None:
            _incomplete.append("Final distributed training decision")

        _report = build_lab_report(
            v2_05_metadata,
            student_id=student_id.value or "",
            track=v2_05_profile.label,
            scenario=v2_05_lens["scenario"],
            learning_objectives=(
                "Diagnose how data, tensor, and pipeline parallelism shift the binding amount.",
                "Use scaling efficiency and bubble terms to decide whether more accelerators help.",
                "Connect batch size and optimizer state to memory, convergence, and step time.",
                "Approve a distributed training plan only when all guardrails pass.",
            ),
            predictions={
                "partA_binding": partA_prediction.value,
                "partB_scaling": partB_prediction.value,
                "partC_batch_optimizer": partC_prediction.value,
                "partD_guardrail": partD_prediction.value,
            },
            knob_settings={
                "partA_strategy": partA_strategy.value,
                "partA_scale": partA_scale.value,
                "partB_gpus": partB_gpus.value,
                "partB_network": partB_network.value,
                "partB_microbatches": partB_microbatches.value,
                "partC_batch": partC_batch.value,
                "partC_accumulation": partC_accumulation.value,
                "partC_sharding": partC_sharding.value,
                "partD_candidate": partD_candidate.value,
                "partD_overlap_pct": partD_overlap.value,
                "partD_evidence": partD_evidence.value,
            },
            binding_constraints={
                "partA_binding_amount": _part_a["binding_amount"],
                "partB_efficiency": f"{v2_05_pct(_part_b['efficiency'], 1)}",
                "partC_binding_amount": _part_c["binding_amount"],
                "partD_binding_guardrail": _part_d["binding_guardrail"],
            },
            decisions={
                "partA_checkpoint": partA_checkpoint.value,
                "partB_checkpoint": partB_checkpoint.value,
                "partC_checkpoint": partC_checkpoint.value,
                "partD_final": partD_final.value,
            },
            residual_risk="Teaching estimates should be replaced with target workload traces, measured fabric bandwidth, memory profiling, convergence pilots, and current cluster reliability before production scheduling.",
            evidence_summary={
                "memory_per_device_gb": round(_part_c["memory_gb"], 3),
                "scaling_efficiency": round(_part_b["efficiency"], 4),
                "global_batch": _part_c["global_batch"],
                "critical_batch_ratio": round(_part_c["batch_ratio"], 4),
                "selected_training_hours": round(_part_d["training_hours"], 3),
                "selected_evidence_score": round(_part_d["evidence_score"], 3),
            },
            final_decision=_memo,
            big_takeaways=(
                "Parallelism relocates overhead; it does not erase it.",
                "Scaling efficiency falls when exposed communication or pipeline bubbles consume useful compute.",
                "Batch and sharding choices are systems choices because they affect memory, step time, and convergence.",
                "V2-06 must inspect the collective pattern implied by the chosen V2-05 plan.",
            ),
            reflections={
                "binding_bottleneck": _binding,
                "rejected_alternative": v2_05_lens["rejected_alternative"],
                "collective_implication": v2_05_lens["collective_implication"],
                "residual_risk": "Teaching estimates need workload traces, fabric measurements, memory profiling, and convergence pilots before production scheduling.",
            },
            source_trace={
                "chapter": "Distributed Training, sections on step-time law, parallelism strategies, ZeRO/FSDP, critical batch size, and summary.",
                "formulas": "T_step(N), eta_scaling, bubble=(p-1)/(m+p-1), B*=tr(Sigma)/||grad L||^2, Adam state bytes/parameter.",
                "local assumptions": "Track budgets, evidence scores, and edge/mobile update thresholds are notebook-local teaching assumptions.",
            },
            result_snapshot=_snapshot,
            incomplete_fields=tuple(_incomplete),
        )

        _status = "SAVED" if _completed else "INCOMPLETE"
        _status_kind = "success" if _completed else "warn"
        return mo.vstack(
            [
                mo.md("## Synthesis - Distributed Training Memo"),
                mo.callout(mo.md(f"**Status:** {_status}. Complete all predictions and checkpoints before final submission."), kind=_status_kind),
                memo_note,
                mo.md(
                    f"""
**Memo draft**

{_memo}

**Evidence to cite**

- Selected plan: **{_part_d['candidate_label']}**
- Binding bottleneck: **{_binding}**
- Rejected alternative: **{v2_05_lens['rejected_alternative']}**
- Memory per device: **{_part_c['memory_gb']:.1f} GB**
- Scaling efficiency: **{v2_05_pct(_part_b['efficiency'], 1)}**
- Critical-batch ratio: **{_part_c['batch_ratio']:.2f}x**
- V2-06 implication: **{v2_05_lens['collective_implication']}**
"""
                ),
                source_trace(
                    {
                        "Distributed step-time law": "T_step(N)=T_compute/N+T_comm(N)+T_sync(N)-T_overlap",
                        "Scaling efficiency": "eta_scaling=T_compute/(N*T_step(N))",
                        "Pipeline bubble": "bubble=(p-1)/(m+p-1)",
                        "Critical batch": "B* ~= tr(Sigma)/||grad L||^2",
                        "Scenario thresholds": "Notebook-local teaching assumptions by track.",
                    },
                    summary="Formula and source trace for the distributed training memo.",
                ),
                report_export_panel(_report),
            ]
        )

    _tabs = mo.ui.tabs(
        {
            "Opening": build_opening(),
            "Part A - Binding Amount": build_part_a(),
            "Part B - Scaling Tax": build_part_b(),
            "Part C - Batch And State": build_part_c(),
            "Part D - Guardrails": build_part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    _tabs
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    ledger,
    mo,
    partA_prediction,
    partB_gpus,
    partB_prediction,
    partC_prediction,
    partD_final,
    partD_prediction,
    v2_05_chapter,
    v2_05_lens,
    v2_05_profile,
):
    _complete = all(
        value is not None
        for value in (
            partA_prediction.value,
            partB_prediction.value,
            partC_prediction.value,
            partD_prediction.value,
            partD_final.value,
        )
    )
    _status = "SAVED" if _complete else "INCOMPLETE"
    _status_color = COLORS["GreenLine"] if _complete else COLORS["OrangeLine"]
    mo.Html(
        f"""
<div class="lab-hud">
  <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab {v2_05_chapter}</span></div>
  <div><span class="hud-label">TRACK</span> <span class="hud-value">{v2_05_profile.label}</span></div>
  <div><span class="hud-label">TRAINING</span> <span class="hud-value">{v2_05_lens['training_location']}</span></div>
  <div><span class="hud-label">SCALE</span> <span class="hud-value">{partB_gpus.value} accelerators</span></div>
  <div><span class="hud-label">STATUS</span> <span style="color:{_status_color}; font-family:var(--font-mono);">{_status}</span></div>
</div>
"""
    )
    return


if __name__ == "__main__":
    app.run()
