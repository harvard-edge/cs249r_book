import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    import html as html_lib
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
        LAB_CSS,
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
        np,
        report_export_panel,
        source_trace,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v2_08_lab_path = "vol2/lab_08_fleet_orch.py"
    v2_08_chapter = 8
    v2_08_metadata = get_lab_metadata(v2_08_lab_path)
    return v2_08_chapter, v2_08_lab_path, v2_08_metadata


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "cloud_fleet"
    v2_08_track_picker = track_selector(default=_default_track)
    v2_08_track_picker
    return (v2_08_track_picker,)


@app.cell
def _(get_lab_track_variant, get_track_profile, v2_08_metadata, v2_08_track_picker):
    v2_08_track_id = v2_08_track_picker.value
    v2_08_profile = get_track_profile(v2_08_track_id)
    v2_08_variant = get_lab_track_variant(v2_08_metadata.lab_id, v2_08_track_id)
    return v2_08_profile, v2_08_track_id, v2_08_variant


@app.cell
def _(html_lib, math, np):
    def v2_08_track_packet(profile, variant):
        base = {
            "track_id": profile.track_id,
            "label": profile.label,
            "stakeholder": variant.stakeholder,
            "mission": profile.narrative,
            "scenario": variant.workload_summary,
            "objective": variant.objective,
            "primary_metric": variant.primary_metric,
            "guardrail_metric": variant.guardrail_metric,
            "hardware_ref": variant.hardware_ref,
            "model_ref": variant.model_ref,
            "system_ref": variant.system_ref or profile.system_ref or "track-local resource pool",
            "arrival_unit": "work units/min",
            "latency_unit": "ms",
            "base_arrival": 5.0,
            "base_capacity": 8.0,
            "service_amount": 40.0,
            "queue_slo": float(variant.defaults.get("latency_budget_ms", 150.0)),
            "arrival_default": 1.0,
            "heavy_mix_default": 30,
            "variability_default": 1.3,
            "util_target_pct": 72,
            "min_util_pct": 45,
            "max_util_pct": 82,
            "fairness_floor_pct": 80,
            "starvation_guard_h": 4.0,
            "topology_limit_pct": 18.0,
            "priority_base_latency": 80.0,
            "urgent_share_default": 35,
            "preempt_default": 35,
            "checkpoint_default": 20,
            "reload_min": 12,
            "warmup_min": 6,
            "preempted_slots": 8,
            "starvation_base_h": 1.4,
            "domain_label": "placement domain",
            "resource_unit": "accelerator slots",
            "urgent_work": "interactive requests",
            "background_work": "batch work",
            "queue_failure": "the queue crosses the SLO even while average utilization still looks efficient",
            "preemption_failure": "urgent work improves by transferring recovery tax and starvation risk",
            "placement_failure": "global free slots cannot form a topology-valid allocation",
            "report_frame": "Fleet scheduler memo",
            "v2_09_implication": "Performance tuning is interpretable only after orchestration hands it stable, topology-valid capacity.",
        }
        overrides = {
            "iphone": {
                "arrival_unit": "tasks/min",
                "latency_unit": "ms",
                "base_arrival": 5.5,
                "base_capacity": 8.8,
                "service_amount": 34.0,
                "queue_slo": 180.0,
                "util_target_pct": 68,
                "min_util_pct": 38,
                "max_util_pct": 78,
                "fairness_floor_pct": 76,
                "starvation_guard_h": 2.0,
                "topology_limit_pct": 14.0,
                "priority_base_latency": 125.0,
                "urgent_share_default": 48,
                "preempt_default": 30,
                "checkpoint_default": 10,
                "reload_min": 4,
                "warmup_min": 2,
                "preempted_slots": 2,
                "starvation_base_h": 0.7,
                "domain_label": "local execution lane",
                "resource_unit": "local compute slots",
                "urgent_work": "foreground interaction",
                "background_work": "background ML refresh",
                "queue_failure": "background ML makes foreground UI miss the responsiveness budget",
                "preemption_failure": "foreground work stays smooth but background refreshes never complete",
                "placement_failure": "local lanes are fragmented across CPU, GPU, and neural-engine windows",
                "report_frame": "App scheduler memo",
                "v2_09_implication": "V2-09 kernel tuning should profile only the foreground path that the scheduler protects.",
            },
            "oura_ring": {
                "arrival_unit": "wake windows/min",
                "latency_unit": "ms",
                "base_arrival": 2.4,
                "base_capacity": 4.2,
                "service_amount": 70.0,
                "queue_slo": 500.0,
                "util_target_pct": 58,
                "min_util_pct": 30,
                "max_util_pct": 70,
                "fairness_floor_pct": 86,
                "starvation_guard_h": 4.0,
                "topology_limit_pct": 12.0,
                "priority_base_latency": 220.0,
                "urgent_share_default": 40,
                "preempt_default": 25,
                "checkpoint_default": 30,
                "reload_min": 2,
                "warmup_min": 1,
                "preempted_slots": 1,
                "starvation_base_h": 1.2,
                "domain_label": "duty-cycle window",
                "resource_unit": "wake slots",
                "urgent_work": "sensing window",
                "background_work": "phone sync and summaries",
                "queue_failure": "deferred sync and inference overrun the sensing cadence",
                "preemption_failure": "sensing cadence wins by starving sync and stale summaries grow",
                "placement_failure": "wake windows are fragmented across sensing, inference, and radio slots",
                "report_frame": "Duty-cycle scheduler memo",
                "v2_09_implication": "V2-09 optimization should target the kernels in the protected sensing window, not deferred sync.",
            },
            "robotaxi": {
                "arrival_unit": "frames/s",
                "latency_unit": "ms",
                "base_arrival": 22.0,
                "base_capacity": 34.0,
                "service_amount": 9.0,
                "queue_slo": 55.0,
                "util_target_pct": 55,
                "min_util_pct": 35,
                "max_util_pct": 68,
                "fairness_floor_pct": 90,
                "starvation_guard_h": 0.12,
                "topology_limit_pct": 8.0,
                "priority_base_latency": 38.0,
                "urgent_share_default": 62,
                "preempt_default": 45,
                "checkpoint_default": 5,
                "reload_min": 1,
                "warmup_min": 1,
                "preempted_slots": 4,
                "starvation_base_h": 0.035,
                "domain_label": "perception lane",
                "resource_unit": "safety compute slots",
                "urgent_work": "perception/control frame",
                "background_work": "mapping and logging",
                "queue_failure": "sensor bursts consume the perception deadline",
                "preemption_failure": "safety lanes meet deadline by repeatedly evicting mapping and logging",
                "placement_failure": "safety tasks land across lanes with too much topology delay",
                "report_frame": "Safety scheduler memo",
                "v2_09_implication": "V2-09 must optimize the worst exposed perception kernels after bounded preemption stabilizes the trace.",
            },
            "cloud_fleet": {
                "arrival_unit": "jobs/min",
                "latency_unit": "min",
                "base_arrival": 6.5,
                "base_capacity": 10.5,
                "service_amount": 4.0,
                "queue_slo": 35.0,
                "util_target_pct": 76,
                "min_util_pct": 55,
                "max_util_pct": 85,
                "fairness_floor_pct": 80,
                "starvation_guard_h": 12.0,
                "topology_limit_pct": 20.0,
                "priority_base_latency": 18.0,
                "urgent_share_default": 32,
                "preempt_default": 35,
                "checkpoint_default": 30,
                "reload_min": 20,
                "warmup_min": 10,
                "preempted_slots": 64,
                "starvation_base_h": 3.5,
                "domain_label": "GPU node",
                "resource_unit": "GPU slots",
                "urgent_work": "interactive debug or serving fix",
                "background_work": "long training run",
                "queue_failure": "high utilization creates a queue wall and tenant SLO breach",
                "preemption_failure": "interactive work improves while long training jobs starve and reload",
                "placement_failure": "free GPUs are stranded by node and topology fragmentation",
                "report_frame": "Fleet accelerator policy memo",
                "v2_09_implication": "V2-09 performance work should profile topology-aware placements, because scattered jobs hide kernel gains behind communication.",
            },
        }
        packet = dict(base)
        packet.update(overrides.get(profile.track_id, {}))
        packet["source_policy"] = profile.source_policy
        return packet

    def v2_08_fmt_amount(value, unit="", precision=2):
        if value is None:
            return "not available"
        if isinstance(value, float) and not math.isfinite(value):
            return "unstable"
        if isinstance(value, (int, float)):
            if abs(value) >= 100:
                text = f"{value:,.0f}"
            elif abs(value) >= 10:
                text = f"{value:,.1f}"
            else:
                text = f"{value:,.{precision}f}"
        else:
            text = str(value)
        return f"{text} {unit}".strip()

    def v2_08_fmt_hours(hours):
        if hours < 0.2:
            return f"{hours * 60:.1f} min"
        return f"{hours:.1f} h"

    def v2_08_html_table(headers, rows):
        header_html = "".join(
            f"<th style='text-align:left; padding:8px 10px; border-bottom:1px solid #cbd5e1;'>{html_lib.escape(str(header))}</th>"
            for header in headers
        )
        row_html = ""
        for row in rows:
            row_html += "<tr>"
            for cell in row:
                row_html += (
                    "<td style='padding:8px 10px; border-bottom:1px solid #e2e8f0; "
                    f"vertical-align:top;'>{html_lib.escape(str(cell))}</td>"
                )
            row_html += "</tr>"
        return f"""
        <div style="overflow-x:auto; margin:14px 0;">
          <table style="width:100%; border-collapse:collapse; background:white; font-size:0.86rem;">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{row_html}</tbody>
          </table>
        </div>
        """

    def v2_08_metric_card(label, value, detail, color, border=False):
        border_style = f"2px solid {color}" if border else "1px solid #e2e8f0"
        return f"""
        <div style="padding:14px; border:{border_style}; border-radius:8px;
                    min-width:155px; text-align:center; background:white;
                    border-top:3px solid {color}; flex:1;">
          <div style="color:#64748b; font-size:0.76rem; font-weight:700;">{html_lib.escape(str(label))}</div>
          <div style="font-size:1.45rem; font-weight:800; color:{color};">{html_lib.escape(str(value))}</div>
          <div style="font-size:0.72rem; color:#64748b;">{html_lib.escape(str(detail))}</div>
        </div>
        """

    def v2_08_clamp(value, low, high):
        return max(low, min(high, value))

    def v2_08_queue_model(packet, arrival_multiplier, heavy_mix_pct, service_cv):
        mix = max(0.0, min(100.0, float(heavy_mix_pct))) / 100.0
        arrival = packet["base_arrival"] * float(arrival_multiplier) * (1.0 + 0.45 * mix)
        capacity = packet["base_capacity"]
        rho = arrival / max(capacity, 1e-9)
        cs = max(0.2, float(service_cv) + 0.75 * mix)
        service = packet["service_amount"] * (1.0 + 0.22 * mix)
        stable_rho = min(rho, 0.985)
        wait_multiplier = stable_rho / (1.0 - stable_rho) * (1.0 + cs**2) / 2.0
        if rho >= 1.0:
            wait_multiplier = 80.0 + (rho - 1.0) * 160.0
        mean_latency = service * (1.0 + wait_multiplier)
        p95_latency = mean_latency * (1.25 + 0.10 * cs)
        queue_depth = arrival * (mean_latency / max(service, 1e-9))
        failure = rho >= 1.0 or p95_latency > packet["queue_slo"]
        binding = "arrival rate" if rho >= 0.85 else "service-time variability" if cs >= 1.8 else "headroom"
        return {
            "arrival": arrival,
            "capacity": capacity,
            "rho": rho,
            "service_amount": service,
            "cs": cs,
            "wait_multiplier": wait_multiplier,
            "mean_latency": mean_latency,
            "p95_latency": p95_latency,
            "queue_depth": queue_depth,
            "failure": failure,
            "binding": binding,
        }

    def v2_08_priority_model(packet, preempt_pct, urgent_share_pct, checkpoint_min):
        preempt = max(0.0, min(100.0, float(preempt_pct))) / 100.0
        urgent_share = max(0.0, min(100.0, float(urgent_share_pct))) / 100.0
        checkpoint = max(1.0, float(checkpoint_min))
        urgent_latency = packet["priority_base_latency"] * (1.08 - 0.58 * preempt) * (1.0 + max(0.0, urgent_share - 0.45) * 0.55)
        lost_work_min = checkpoint / 2.0
        recovery_min = lost_work_min + packet["reload_min"] + packet["warmup_min"]
        preemption_events_h = 0.12 + preempt * (0.6 + urgent_share * 2.4)
        preemption_tax_slot_min_h = preemption_events_h * recovery_min * packet["preempted_slots"]
        starvation_wait_h = packet["starvation_base_h"] * (1.0 + preempt * 3.8 + urgent_share * 1.4) * (1.0 + packet["checkpoint_default"] / checkpoint * 0.10)
        urgent_ok = urgent_latency <= packet["queue_slo"] * 0.75
        starvation_ok = starvation_wait_h <= packet["starvation_guard_h"]
        binding = "starvation" if not starvation_ok else "urgent latency" if not urgent_ok else "preemption tax"
        return {
            "urgent_latency": urgent_latency,
            "starvation_wait_h": starvation_wait_h,
            "preemption_tax_slot_min_h": preemption_tax_slot_min_h,
            "preemption_events_h": preemption_events_h,
            "recovery_min": recovery_min,
            "urgent_ok": urgent_ok,
            "starvation_ok": starvation_ok,
            "failure": not (urgent_ok and starvation_ok),
            "binding": binding,
        }

    def v2_08_base_grid():
        return np.array([
            [1, 1, 0, 0, 2, 2, 2, 0],
            [3, 3, 3, 3, 0, 0, 4, 4],
            [5, 5, 0, 6, 6, 6, 6, 6],
            [7, 7, 7, 0, 0, 8, 8, 8],
            [9, 9, 0, 0, 0, 10, 10, 10],
            [11, 11, 11, 11, 0, 0, 12, 12],
            [13, 13, 13, 13, 13, 0, 0, 0],
            [14, 14, 14, 14, 15, 0, 0, 0],
        ])

    def v2_08_contiguous_free(row):
        best = 0
        run = 0
        for value in row:
            if value == 0:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    def v2_08_fill_contiguous(row, job_size):
        run = 0
        start = None
        for idx, value in enumerate(row):
            if value == 0:
                if run == 0:
                    start = idx
                run += 1
                if run >= job_size:
                    return list(range(start, start + job_size))
            else:
                run = 0
                start = None
        return []

    def v2_08_placement_model(packet, job_size, policy, topology_weight):
        grid = v2_08_base_grid()
        marked = grid.copy()
        total_slots = int(grid.size)
        free_slots = int(np.count_nonzero(grid == 0))
        allocated_before = total_slots - free_slots
        row_free = [int(np.count_nonzero(row == 0)) for row in grid]
        row_contig = [v2_08_contiguous_free(row) for row in grid]
        max_contiguous = max(row_contig)
        rows_used = []
        feasible = False
        locality_cost = 0.0
        preemption_tax = 0
        failure_reason = ""

        if policy == "greedy_split":
            feasible = free_slots >= job_size
            remaining = job_size
            for row_idx in range(marked.shape[0]):
                for col_idx in range(marked.shape[1]):
                    if remaining and marked[row_idx, col_idx] == 0:
                        marked[row_idx, col_idx] = 99
                        rows_used.append(row_idx)
                        remaining -= 1
                if remaining == 0:
                    break
            unique_rows = sorted(set(rows_used))
            for left, row_i in enumerate(unique_rows):
                for row_j in unique_rows[left + 1:]:
                    same_rack = row_i // 4 == row_j // 4
                    locality_cost += 1 if same_rack else 10
            failure_reason = "scattered placement" if feasible and len(unique_rows) > 1 else ""
        elif policy in ("best_fit", "topology_aware"):
            candidates = []
            for row_idx, row in enumerate(marked):
                slots = v2_08_fill_contiguous(row, job_size)
                if slots:
                    candidates.append((row_free[row_idx] - job_size, row_idx, slots))
            if candidates:
                _, row_idx, slots = min(candidates)
                feasible = True
                rows_used = [row_idx] * job_size
                for col_idx in slots:
                    marked[row_idx, col_idx] = 99
            else:
                failure_reason = "no contiguous domain"
        elif policy == "defrag_backfill":
            feasible = free_slots >= job_size
            if feasible:
                row_idx = int(np.argmax(row_free))
                rows_used = [row_idx] * job_size
                preemption_tax = max(0, job_size - max_contiguous)
                placed = 0
                for col_idx in range(marked.shape[1]):
                    if marked[row_idx, col_idx] == 0 and placed < min(job_size, row_free[row_idx]):
                        marked[row_idx, col_idx] = 99
                        placed += 1
                for col_idx in range(marked.shape[1]):
                    if placed < job_size:
                        marked[row_idx, col_idx] = 99
                        placed += 1
                locality_cost = 1.5 * preemption_tax
                failure_reason = "defragmentation tax" if preemption_tax else ""
            else:
                failure_reason = "not enough free slots"

        if not feasible and not failure_reason:
            failure_reason = "not enough free slots"
        fragmentation_index = 0.0 if free_slots == 0 else max(0.0, (free_slots - max_contiguous) / free_slots)
        topology_penalty_pct = min(35.0, float(topology_weight) * (locality_cost * 2.4 + preemption_tax * 1.5))
        utilization_after = (allocated_before + (job_size if feasible else 0)) / total_slots * 100.0
        return {
            "grid": marked,
            "free_slots": free_slots,
            "max_contiguous": max_contiguous,
            "fragmentation_index": fragmentation_index,
            "feasible": feasible,
            "locality_cost": locality_cost,
            "topology_penalty_pct": topology_penalty_pct,
            "utilization_after": utilization_after,
            "rows_used": sorted(set(rows_used)),
            "preemption_tax_slots": preemption_tax,
            "failure_reason": failure_reason or "none",
        }

    def v2_08_policy_candidates(packet):
        return {
            "greedy_utilization": {
                "label": "Greedy utilization",
                "util_delta": 10.0,
                "fair_delta": -14.0,
                "slo_mult": 1.22,
                "starve_mult": 1.35,
                "topology_mult": 1.25,
                "rationale": "Fill every open slot and accept queueing or topology side effects.",
            },
            "priority_preempt": {
                "label": "Priority preemption",
                "util_delta": 5.0,
                "fair_delta": -8.0,
                "slo_mult": 0.82,
                "starve_mult": 1.70,
                "topology_mult": 1.05,
                "rationale": "Protect urgent work by evicting lower-priority work.",
            },
            "topology_fair_share": {
                "label": "Topology-aware fair-share",
                "util_delta": -2.0,
                "fair_delta": 8.0,
                "slo_mult": 0.92,
                "starve_mult": 0.72,
                "topology_mult": 0.55,
                "rationale": "Reserve topology-valid domains and age waiting tenants.",
            },
            "conservative_reserve": {
                "label": "Conservative reservation",
                "util_delta": -12.0,
                "fair_delta": 3.0,
                "slo_mult": 0.78,
                "starve_mult": 0.55,
                "topology_mult": 0.75,
                "rationale": "Keep spare capacity and wait for clean placements.",
            },
        }

    def v2_08_evaluate_policy(packet, key, util_target_pct, fairness_floor_pct, starvation_guard_h, queue_result, priority_result, placement_result):
        candidate = v2_08_policy_candidates(packet)[key]
        queue_util = min(99.0, queue_result["rho"] * 100.0)
        utilization_pct = v2_08_clamp(
            0.55 * float(util_target_pct) + 0.45 * queue_util + candidate["util_delta"],
            0.0,
            99.0,
        )
        fairness_pct = v2_08_clamp(packet["fairness_floor_pct"] + candidate["fair_delta"], 0.0, 100.0)
        p95_latency = queue_result["p95_latency"] * candidate["slo_mult"] * max(0.65, utilization_pct / max(packet["util_target_pct"], 1))
        starvation_h = priority_result["starvation_wait_h"] * candidate["starve_mult"]
        topology_penalty = placement_result["topology_penalty_pct"] * candidate["topology_mult"]
        utilization_ok = packet["min_util_pct"] <= utilization_pct <= packet["max_util_pct"]
        fairness_ok = fairness_pct >= float(fairness_floor_pct)
        slo_ok = p95_latency <= packet["queue_slo"]
        starvation_ok = starvation_h <= float(starvation_guard_h)
        topology_ok = topology_penalty <= packet["topology_limit_pct"] and placement_result["feasible"]
        checks = {
            "utilization": utilization_ok,
            "fairness": fairness_ok,
            "SLO": slo_ok,
            "starvation": starvation_ok,
            "topology": topology_ok,
        }
        binding = next((name for name, ok in checks.items() if not ok), "all guardrails pass")
        return {
            "candidate": candidate,
            "key": key,
            "utilization_pct": utilization_pct,
            "fairness_pct": fairness_pct,
            "p95_latency": p95_latency,
            "starvation_h": starvation_h,
            "topology_penalty_pct": topology_penalty,
            "checks": checks,
            "feasible": all(checks.values()),
            "binding": binding,
        }

    def v2_08_policy_label(key, packet=None):
        candidates = v2_08_policy_candidates(packet or {})
        return candidates.get(key, {}).get("label", str(key).replace("_", " "))

    return (
        v2_08_base_grid,
        v2_08_clamp,
        v2_08_contiguous_free,
        v2_08_evaluate_policy,
        v2_08_fill_contiguous,
        v2_08_fmt_amount,
        v2_08_fmt_hours,
        v2_08_html_table,
        v2_08_metric_card,
        v2_08_placement_model,
        v2_08_policy_candidates,
        v2_08_policy_label,
        v2_08_priority_model,
        v2_08_queue_model,
        v2_08_track_packet,
    )


@app.cell
def _(v2_08_profile, v2_08_track_packet, v2_08_variant):
    v2_08_packet = v2_08_track_packet(v2_08_profile, v2_08_variant)
    return (v2_08_packet,)


@app.cell
def _(mo, v2_08_packet):
    v2_08_partA_pred = mo.ui.radio(
        options={
            "Arrival pressure and job mix will create the queue wall.": "queue_pressure",
            "Raw capacity alone decides whether the queue is healthy.": "raw_capacity",
            "High utilization is enough evidence that scheduling works.": "utilization_only",
        },
        label="Part A prediction",
    )
    v2_08_arrival_multiplier = mo.ui.slider(
        start=0.45,
        stop=1.9,
        step=0.05,
        value=float(v2_08_packet["arrival_default"]),
        label=f"Arrival multiplier ({v2_08_packet['arrival_unit']})",
    )
    v2_08_heavy_mix = mo.ui.slider(
        start=0,
        stop=75,
        step=5,
        value=int(v2_08_packet["heavy_mix_default"]),
        label="Heavy or long-running work mix (%)",
    )
    v2_08_service_cv = mo.ui.slider(
        start=0.4,
        stop=3.2,
        step=0.1,
        value=float(v2_08_packet["variability_default"]),
        label="Service-time variability Cs",
    )
    v2_08_partA_checkpoint = mo.ui.radio(
        options={
            "Hold headroom below the queueing knee.": "hold_headroom",
            "Admit all work until capacity is fully used.": "admit_all",
            "Ignore job mix and tune only average service time.": "average_only",
        },
        label="Part A checkpoint",
    )

    v2_08_partB_pred = mo.ui.radio(
        options={
            "Preemption lowers urgent latency but raises starvation and recovery tax.": "latency_starvation_trade",
            "Preemption is free if checkpoints exist.": "preemption_free",
            "Priority affects only the urgent class.": "urgent_only",
        },
        label="Part B prediction",
    )
    v2_08_preempt_pct = mo.ui.slider(
        start=0,
        stop=90,
        step=5,
        value=int(v2_08_packet["preempt_default"]),
        label="Preemption aggressiveness (%)",
    )
    v2_08_urgent_share = mo.ui.slider(
        start=10,
        stop=85,
        step=5,
        value=int(v2_08_packet["urgent_share_default"]),
        label="Urgent work share (%)",
    )
    v2_08_checkpoint_min = mo.ui.slider(
        start=5,
        stop=60,
        step=5,
        value=int(v2_08_packet["checkpoint_default"]),
        label="Checkpoint interval (min)",
    )
    v2_08_partB_checkpoint = mo.ui.radio(
        options={
            "Use bounded preemption with aging for waiting work.": "bounded_preempt",
            "Let urgent work preempt without a churn budget.": "unbounded_preempt",
            "Disable preemption and accept urgent latency misses.": "no_preempt",
        },
        label="Part B checkpoint",
    )

    v2_08_partC_pred = mo.ui.radio(
        options={
            "A) Total free slots can still be unusable when placement is fragmented.": "fragmentation_matters",
            "B) Any free slots are equivalent once the job count fits.": "free_slots_only",
            "C) Topology affects speed but not scheduling feasibility.": "topology_only_speed",
        },
        label="Part C prediction",
    )
    v2_08_job_size = mo.ui.slider(
        start=2,
        stop=8,
        step=1,
        value=4,
        label=f"Pending job size ({v2_08_packet['resource_unit']})",
    )
    v2_08_placement_policy = mo.ui.dropdown(
        options={
            "Greedy split placement": "greedy_split",
            "Best-fit contiguous placement": "best_fit",
            "Topology-aware placement": "topology_aware",
            "Defrag plus backfill": "defrag_backfill",
        },
        value="Topology-aware placement",
        label="Placement policy",
    )
    v2_08_topology_weight = mo.ui.slider(
        start=0.5,
        stop=3.0,
        step=0.25,
        value=1.5,
        label="Topology sensitivity",
    )
    v2_08_partC_checkpoint = mo.ui.radio(
        options={
            "Wait for or create a topology-valid placement.": "topology_valid",
            "Split across any free slots to maximize utilization.": "split_anywhere",
            "Preempt immediately to compact the pool.": "compact_now",
        },
        label="Part C checkpoint",
    )

    v2_08_partD_pred = mo.ui.radio(
        options={
            "The launch decision is the conjunction of all guardrails.": "all_guardrails",
            "The highest utilization policy should win.": "utilization_wins",
            "The lowest urgent latency policy should win.": "latency_wins",
            "The fairest policy should win even if SLO fails.": "fairness_wins",
        },
        label="Part D prediction",
    )
    v2_08_policy = mo.ui.dropdown(
        options={
            "Greedy utilization": "greedy_utilization",
            "Priority preemption": "priority_preempt",
            "Topology-aware fair-share": "topology_fair_share",
            "Conservative reservation": "conservative_reserve",
        },
        value="Topology-aware fair-share",
        label="Selected scheduler policy",
    )
    v2_08_rejected_policy = mo.ui.dropdown(
        options={
            "Greedy utilization": "greedy_utilization",
            "Priority preemption": "priority_preempt",
            "Topology-aware fair-share": "topology_fair_share",
            "Conservative reservation": "conservative_reserve",
        },
        value="Greedy utilization",
        label="Rejected alternative",
    )
    v2_08_util_target = mo.ui.slider(
        start=35,
        stop=95,
        step=1,
        value=int(v2_08_packet["util_target_pct"]),
        label="Utilization target (%)",
    )
    v2_08_fairness_floor = mo.ui.slider(
        start=50,
        stop=98,
        step=1,
        value=int(v2_08_packet["fairness_floor_pct"]),
        label="Fairness floor (%)",
    )
    v2_08_starvation_guard = mo.ui.slider(
        start=0.05,
        stop=max(1.0, float(v2_08_packet["starvation_guard_h"]) * 2.5),
        step=0.05 if float(v2_08_packet["starvation_guard_h"]) < 1.0 else 0.5,
        value=float(v2_08_packet["starvation_guard_h"]),
        label="Starvation guardrail (hours)",
    )
    v2_08_v2_09_implication = mo.ui.dropdown(
        options={
            "Profile only topology-valid placements in V2-09.": "topology_valid_profiles",
            "Relieve queue pressure before interpreting V2-09 kernel traces.": "queue_before_kernel",
            "Bound preemption churn before V2-09 performance tuning.": "bound_churn",
        },
        value="Profile only topology-valid placements in V2-09.",
        label="V2-09 performance implication",
    )
    v2_08_student_id = mo.ui.text(label="Student identifier", placeholder="Optional")
    return (
        v2_08_arrival_multiplier,
        v2_08_checkpoint_min,
        v2_08_fairness_floor,
        v2_08_heavy_mix,
        v2_08_job_size,
        v2_08_partA_checkpoint,
        v2_08_partA_pred,
        v2_08_partB_checkpoint,
        v2_08_partB_pred,
        v2_08_partC_checkpoint,
        v2_08_partC_pred,
        v2_08_partD_pred,
        v2_08_placement_policy,
        v2_08_policy,
        v2_08_preempt_pct,
        v2_08_rejected_policy,
        v2_08_service_cv,
        v2_08_starvation_guard,
        v2_08_student_id,
        v2_08_topology_weight,
        v2_08_urgent_share,
        v2_08_util_target,
        v2_08_v2_09_implication,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    COLORS,
    LAB_CSS,
    apply_plotly_theme,
    build_lab_report,
    go,
    ledger,
    mo,
    np,
    report_export_panel,
    source_trace,
    track_arc_context,
    track_context,
    v2_08_arrival_multiplier,
    v2_08_chapter,
    v2_08_checkpoint_min,
    v2_08_evaluate_policy,
    v2_08_fairness_floor,
    v2_08_fmt_amount,
    v2_08_fmt_hours,
    v2_08_heavy_mix,
    v2_08_html_table,
    v2_08_job_size,
    v2_08_metadata,
    v2_08_metric_card,
    v2_08_packet,
    v2_08_partA_checkpoint,
    v2_08_partA_pred,
    v2_08_partB_checkpoint,
    v2_08_partB_pred,
    v2_08_partC_checkpoint,
    v2_08_partC_pred,
    v2_08_partD_pred,
    v2_08_placement_model,
    v2_08_placement_policy,
    v2_08_policy,
    v2_08_policy_candidates,
    v2_08_policy_label,
    v2_08_preempt_pct,
    v2_08_priority_model,
    v2_08_profile,
    v2_08_queue_model,
    v2_08_rejected_policy,
    v2_08_service_cv,
    v2_08_starvation_guard,
    v2_08_student_id,
    v2_08_topology_weight,
    v2_08_urgent_share,
    v2_08_util_target,
    v2_08_v2_09_implication,
    v2_08_variant,
):
    v2_08_queue = v2_08_queue_model(
        v2_08_packet,
        v2_08_arrival_multiplier.value,
        v2_08_heavy_mix.value,
        v2_08_service_cv.value,
    )
    v2_08_priority = v2_08_priority_model(
        v2_08_packet,
        v2_08_preempt_pct.value,
        v2_08_urgent_share.value,
        v2_08_checkpoint_min.value,
    )
    v2_08_placement = v2_08_placement_model(
        v2_08_packet,
        v2_08_job_size.value,
        v2_08_placement_policy.value,
        v2_08_topology_weight.value,
    )

    def v2_08_part_banner(part, title, color, body):
        return mo.Html(f"""
        <div style="border-left:4px solid {color}; background:white;
                    border-radius:0 8px 8px 0; padding:16px 22px; margin:12px 0;
                    box-shadow:0 1px 4px rgba(0,0,0,0.06);">
          <div style="font-size:0.72rem; font-weight:700; color:{color};
                      text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
            Part {part} Concept Module - {v2_08_packet['stakeholder']}
          </div>
          <div style="font-weight:800; font-size:1.08rem; color:#0f172a; margin-bottom:6px;">
            {title}
          </div>
          <div style="font-size:0.94rem; color:#334155; line-height:1.6;">{body}</div>
        </div>
        """)

    def v2_08_prediction_feedback(selected, correct_key, correct, miss):
        if selected == correct_key:
            return mo.callout(mo.md(f"You predicted `{selected}`; actual result is `{correct_key}`. {correct}"), kind="success")
        return mo.callout(mo.md(f"You predicted `{selected}`; actual result is `{correct_key}`. {miss}"), kind="warn")

    def v2_08_opening():
        reading_rows = (
            ("Part A", "Queueing pressure", "Pollaczek-Khinchine wait multiplier and the utilization knee."),
            ("Part B", "Priority/preemption", "Preemption tax and starvation guardrails."),
            ("Part C", "Placement/bin packing", "Fragmented free slots and topology locality score."),
            ("Part D", "Policy gate", "Guardrail conjunction across utilization, fairness, SLO, starvation, and topology."),
        )
        return mo.vstack([
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            mo.md(f"""
# V2-08 - The Scheduling Trap

**Chapter invariant:** schedulers allocate scarce resources. Queueing,
priorities, placement/bin packing, and fairness/utilization interact, so the
policy is valid only when the same workload passes all guardrails.
            """),
            track_context(v2_08_profile),
            track_arc_context(v2_08_profile, v2_08_metadata.lab_id),
            mo.callout(mo.md(
                f"**Scenario:** {v2_08_packet['stakeholder']} must choose an orchestration policy for "
                f"{v2_08_packet['label']}. The amount system is {v2_08_packet['arrival_unit']}, "
                f"{v2_08_packet['resource_unit']}, p95 wait/latency in {v2_08_packet['latency_unit']}, "
                "fair-share percent, and starvation time."
            ), kind="info"),
            mo.Html(v2_08_html_table(("Module", "Concept", "Reading connection"), reading_rows)),
            source_trace({
                "chapter": "Volume II, Chapter 8: Fleet Orchestration",
                "anchors": (
                    "Scheduling objectives and conflicts",
                    "The queuing theory of GPU clusters",
                    "Bin packing and topology-aware placement",
                    "Priority preemption cascades",
                    "Hierarchical fair-share",
                    "From orchestration to optimization",
                ),
                "local_models": "Notebook-local v2_08_* teaching models document scenario constants in Math Peek blocks.",
                "track_source": v2_08_packet["source_policy"],
            }, summary="Source trace: V2-08 concept-module evidence"),
        ])

    def v2_08_build_part_a():
        items = [
            v2_08_part_banner(
                "A",
                "Queueing Pressure Emerges From Job Mix And Arrival Rate",
                COLORS["BlueLine"],
                (
                    f"{v2_08_packet['stakeholder']} is setting admission headroom for "
                    f"{v2_08_packet['urgent_work']} plus {v2_08_packet['background_work']}."
                ),
            ),
            mo.md("""
## Concept: Utilization Is Not Queue Health

The same capacity can feel healthy or broken depending on arrival pressure and
service-time variability. Commit to a prediction, then move the workload mix
until the queue crosses the track guardrail.
            """),
            v2_08_partA_pred,
        ]
        if v2_08_partA_pred.value is None:
            items.append(mo.callout(mo.md("Commit to the queueing prediction before opening the instrument."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([v2_08_arrival_multiplier, v2_08_heavy_mix, v2_08_service_cv], widths="equal"))
        sweep_x = np.linspace(0.45, 1.9, 70)
        p95_values = []
        rho_values = []
        for x in sweep_x:
            result = v2_08_queue_model(v2_08_packet, float(x), v2_08_heavy_mix.value, v2_08_service_cv.value)
            p95_values.append(min(result["p95_latency"], v2_08_packet["queue_slo"] * 3.5))
            rho_values.append(min(result["rho"] * 100.0, 120.0))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep_x,
            y=p95_values,
            mode="lines",
            name="p95 wait/latency",
            line=dict(color=COLORS["RedLine"], width=3),
        ))
        fig.add_trace(go.Scatter(
            x=sweep_x,
            y=rho_values,
            mode="lines",
            name="utilization (%)",
            yaxis="y2",
            line=dict(color=COLORS["OrangeLine"], width=2),
        ))
        fig.add_hline(
            y=v2_08_packet["queue_slo"],
            line_dash="dash",
            line_color=COLORS["GreenLine"],
            annotation_text=f"guardrail {v2_08_packet['queue_slo']:g} {v2_08_packet['latency_unit']}",
        )
        fig.add_trace(go.Scatter(
            x=[v2_08_arrival_multiplier.value],
            y=[min(v2_08_queue["p95_latency"], v2_08_packet["queue_slo"] * 3.5)],
            mode="markers",
            name="selected point",
            marker=dict(color=COLORS["BlueLine"], size=14, symbol="diamond"),
        ))
        fig.update_layout(
            height=360,
            xaxis=dict(title="Arrival multiplier"),
            yaxis=dict(title=f"p95 wait/latency ({v2_08_packet['latency_unit']})", gridcolor="#f1f5f9"),
            yaxis2=dict(title="utilization (%)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.15, x=0),
            margin=dict(l=60, r=70, t=55, b=45),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        status_color = COLORS["RedLine"] if v2_08_queue["failure"] else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
          {v2_08_metric_card("Utilization", f"{v2_08_queue['rho']*100:.1f}%", "arrival / capacity", COLORS["OrangeLine"])}
          {v2_08_metric_card("Wait Multiplier", f"{v2_08_queue['wait_multiplier']:.1f}x", "Wq / service", COLORS["BlueLine"])}
          {v2_08_metric_card("Queue Depth", f"{v2_08_queue['queue_depth']:.1f}", v2_08_packet["arrival_unit"], COLORS["OrangeLine"])}
          {v2_08_metric_card("P95", f"{v2_08_queue['p95_latency']:.1f} {v2_08_packet['latency_unit']}", f"limit {v2_08_packet['queue_slo']:g}", status_color, True)}
        </div>
        """))
        table_rows = []
        for multiplier in (max(0.45, v2_08_arrival_multiplier.value - 0.25), v2_08_arrival_multiplier.value, min(1.9, v2_08_arrival_multiplier.value + 0.25)):
            result = v2_08_queue_model(v2_08_packet, multiplier, v2_08_heavy_mix.value, v2_08_service_cv.value)
            table_rows.append((
                f"{multiplier:.2f}x",
                f"{result['arrival']:.2f} {v2_08_packet['arrival_unit']}",
                f"{result['rho']*100:.1f}%",
                f"{result['wait_multiplier']:.1f}x",
                f"{result['p95_latency']:.1f} {v2_08_packet['latency_unit']}",
                "FAIL" if result["failure"] else "PASS",
            ))
        items.append(mo.Html(v2_08_html_table(
            ("Arrival", "Effective load", "Utilization", "Wait multiplier", "P95", "Boundary"),
            table_rows,
        )))
        if v2_08_queue["failure"]:
            items.append(mo.callout(mo.md(
                f"**Boundary crossed:** {v2_08_packet['queue_failure']}. "
                f"P95 is {v2_08_queue['p95_latency']:.1f} {v2_08_packet['latency_unit']} "
                f"against a {v2_08_packet['queue_slo']:g} {v2_08_packet['latency_unit']} guardrail. "
                "Recover by lowering admitted arrivals, reducing long-job mix, or reserving headroom."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"**Boundary holds:** the selected queue has {v2_08_queue['rho']*100:.1f}% utilization "
                f"and p95 {v2_08_queue['p95_latency']:.1f} {v2_08_packet['latency_unit']}."
            ), kind="success"))
        items.append(v2_08_prediction_feedback(
            v2_08_partA_pred.value,
            "queue_pressure",
            "**Correct.** Queueing pressure is a joint result of arrival rate, capacity, and service-time variability.",
            "**Queueing trap:** raw capacity and high utilization are incomplete evidence when job durations are heavy-tailed.",
        ))
        items.append(mo.accordion({
            "Math Peek / Source Model - queueing pressure": mo.md(f"""
```
rho              = lambda_effective / mu_capacity
Wq / E[S]        = rho/(1-rho) * (1 + Cs^2)/2
p95_guardrail_ok = p95_wait_or_latency <= track_guardrail
```

Current values: rho `{v2_08_queue['rho']:.3f}`, Cs `{v2_08_queue['cs']:.2f}`,
wait multiplier `{v2_08_queue['wait_multiplier']:.2f}x`, p95
`{v2_08_queue['p95_latency']:.1f} {v2_08_packet['latency_unit']}`.

Chapter anchor: the M/G/1 queueing callout in Fleet Orchestration. Local
scenario constants encode the selected track's arrival unit, service amount,
and guardrail.
            """)
        }))
        items.append(v2_08_partA_checkpoint)
        if v2_08_partA_checkpoint.value is None:
            items.append(mo.callout(mo.md("Checkpoint: choose the admission rule that should constrain priority decisions."), kind="info"))
        return mo.vstack(items)

    def v2_08_build_part_b():
        items = [
            v2_08_part_banner(
                "B",
                "Priority And Preemption Trade Latency For Starvation Risk",
                COLORS["OrangeLine"],
                (
                    f"The scheduler must protect {v2_08_packet['urgent_work']} without turning "
                    f"{v2_08_packet['background_work']} into permanent backlog."
                ),
            ),
            mo.md("""
## Concept: Priority Moves Pain, It Does Not Remove It

Preemption can make urgent work fast, but every eviction pays lost work,
reload, warmup, and queue churn. The policy question is whether that tax is
bounded.
            """),
            v2_08_partB_pred,
        ]
        if v2_08_partB_pred.value is None:
            items.append(mo.callout(mo.md("Commit to the preemption prediction before opening the trade-off chart."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([v2_08_preempt_pct, v2_08_urgent_share, v2_08_checkpoint_min], widths="equal"))
        sweep = np.linspace(0, 90, 46)
        urgent_curve = []
        starve_curve = []
        tax_curve = []
        for pct in sweep:
            result = v2_08_priority_model(v2_08_packet, pct, v2_08_urgent_share.value, v2_08_checkpoint_min.value)
            urgent_curve.append(result["urgent_latency"])
            starve_curve.append(result["starvation_wait_h"])
            tax_curve.append(result["preemption_tax_slot_min_h"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep,
            y=urgent_curve,
            mode="lines",
            name=f"urgent latency ({v2_08_packet['latency_unit']})",
            line=dict(color=COLORS["BlueLine"], width=3),
        ))
        fig.add_trace(go.Scatter(
            x=sweep,
            y=starve_curve,
            mode="lines",
            name="starvation wait (h)",
            yaxis="y2",
            line=dict(color=COLORS["RedLine"], width=3),
        ))
        fig.add_hline(
            y=v2_08_packet["queue_slo"] * 0.75,
            line_dash="dash",
            line_color=COLORS["GreenLine"],
            annotation_text="urgent latency guardrail",
        )
        fig.add_trace(go.Scatter(
            x=[v2_08_preempt_pct.value],
            y=[v2_08_priority["urgent_latency"]],
            mode="markers",
            name="selected urgent point",
            marker=dict(color=COLORS["BlueLine"], size=13, symbol="diamond"),
        ))
        fig.update_layout(
            height=360,
            xaxis=dict(title="Preemption aggressiveness (%)"),
            yaxis=dict(title=f"Urgent latency ({v2_08_packet['latency_unit']})", gridcolor="#f1f5f9"),
            yaxis2=dict(title="Starvation wait (h)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.15, x=0),
            margin=dict(l=60, r=70, t=55, b=45),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        status_color = COLORS["RedLine"] if v2_08_priority["failure"] else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
          {v2_08_metric_card("Urgent Latency", f"{v2_08_priority['urgent_latency']:.1f} {v2_08_packet['latency_unit']}", "after priority", COLORS["BlueLine"])}
          {v2_08_metric_card("Starvation Wait", v2_08_fmt_hours(v2_08_priority['starvation_wait_h']), f"limit {v2_08_fmt_hours(v2_08_packet['starvation_guard_h'])}", status_color, True)}
          {v2_08_metric_card("Recovery Tax", f"{v2_08_priority['preemption_tax_slot_min_h']:.0f}", "slot-min/h", COLORS["OrangeLine"])}
          {v2_08_metric_card("Events", f"{v2_08_priority['preemption_events_h']:.2f}/h", "expected churn", COLORS["RedLine"])}
        </div>
        """))
        table_rows = []
        for pct in (0, v2_08_preempt_pct.value, 75):
            result = v2_08_priority_model(v2_08_packet, pct, v2_08_urgent_share.value, v2_08_checkpoint_min.value)
            table_rows.append((
                f"{pct:.0f}%",
                f"{result['urgent_latency']:.1f} {v2_08_packet['latency_unit']}",
                v2_08_fmt_hours(result["starvation_wait_h"]),
                f"{result['preemption_tax_slot_min_h']:.0f} slot-min/h",
                "PASS" if not result["failure"] else f"FAIL: {result['binding']}",
            ))
        items.append(mo.Html(v2_08_html_table(
            ("Preemption", "Urgent latency", "Starvation wait", "Recovery tax", "Boundary"),
            table_rows,
        )))
        if v2_08_priority["failure"]:
            items.append(mo.callout(mo.md(
                f"**Preemption boundary crossed:** {v2_08_packet['preemption_failure']}. "
                f"Binding guardrail: {v2_08_priority['binding']}. Recovery tax is "
                f"{v2_08_priority['preemption_tax_slot_min_h']:.0f} slot-min/h."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"**Priority rule is bounded:** urgent latency is {v2_08_priority['urgent_latency']:.1f} "
                f"{v2_08_packet['latency_unit']} and starvation wait is {v2_08_fmt_hours(v2_08_priority['starvation_wait_h'])}."
            ), kind="success"))
        items.append(v2_08_prediction_feedback(
            v2_08_partB_pred.value,
            "latency_starvation_trade",
            "**Correct.** Preemption improves one class by spending recovery and fairness budget from another.",
            "**Priority trap:** a fast urgent path can still be an unhealthy scheduler if waiting work never ages into service.",
        ))
        items.append(mo.accordion({
            "Math Peek / Source Model - preemption tax": mo.md(f"""
```
recovery_min = checkpoint_interval/2 + reload_min + warmup_min
tax          = preemption_events_per_hour * recovery_min * preempted_slots
starvation_ok = waiting_time_h <= starvation_guard_h
```

Current values: recovery `{v2_08_priority['recovery_min']:.1f}` min/event,
tax `{v2_08_priority['preemption_tax_slot_min_h']:.0f}` slot-min/h,
starvation wait `{v2_08_fmt_hours(v2_08_priority['starvation_wait_h'])}`.

Chapter anchor: Priority preemption cascades and preemption-tax discussion.
The local model exposes the chapter consequence without claiming a production
scheduler simulator.
            """)
        }))
        items.append(v2_08_partB_checkpoint)
        if v2_08_partB_checkpoint.value is None:
            items.append(mo.callout(mo.md("Checkpoint: choose the priority rule that should constrain placement."), kind="info"))
        return mo.vstack(items)

    def v2_08_build_part_c():
        items = [
            v2_08_part_banner(
                "C",
                "Placement And Bin Packing Change Utilization And Topology Cost",
                COLORS["GreenLine"],
                (
                    f"A pending job needs {v2_08_job_size.value} {v2_08_packet['resource_unit']}. "
                    f"The pool has free amount, but it may not have a valid {v2_08_packet['domain_label']}."
                ),
            ),
            mo.md("""
## Concept: Free Capacity Is Spatial

Global free amount is not the same as schedulable amount. Placement decides
whether slots are contiguous, whether topology is valid, and whether the
resulting job pays a communication or migration penalty.
            """),
            v2_08_partC_pred,
        ]
        if v2_08_partC_pred.value is None:
            items.append(mo.callout(mo.md("Commit to the placement prediction before opening the topology heatmap."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([v2_08_job_size, v2_08_placement_policy, v2_08_topology_weight], widths="equal"))
        heat = v2_08_placement["grid"]
        colorscale = [
            [0.0, "#e2e8f0"],
            [0.48, "#e2e8f0"],
            [0.49, "#94a3b8"],
            [0.90, "#94a3b8"],
            [0.91, COLORS["GreenLine"]],
            [1.0, COLORS["GreenLine"]],
        ]
        z = np.where(heat == 0, 0, np.where(heat == 99, 2, 1))
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[f"slot {idx}" for idx in range(8)],
            y=[f"domain {idx}" for idx in range(8)],
            colorscale=colorscale,
            showscale=False,
            hovertemplate="%{y}, %{x}<extra></extra>",
        ))
        fig.update_layout(
            height=360,
            xaxis=dict(title="Local slot"),
            yaxis=dict(title=v2_08_packet["domain_label"]),
            margin=dict(l=80, r=20, t=40, b=50),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        status_color = COLORS["GreenLine"] if v2_08_placement["feasible"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
          {v2_08_metric_card("Free Slots", v2_08_placement['free_slots'], "global free amount", COLORS["BlueLine"])}
          {v2_08_metric_card("Max Contiguous", v2_08_placement['max_contiguous'], v2_08_packet["domain_label"], COLORS["OrangeLine"])}
          {v2_08_metric_card("Topology Penalty", f"{v2_08_placement['topology_penalty_pct']:.1f}%", f"limit {v2_08_packet['topology_limit_pct']:.0f}%", status_color, True)}
          {v2_08_metric_card("Utilization", f"{v2_08_placement['utilization_after']:.1f}%", "after placement if feasible", COLORS["GreenLine"])}
        </div>
        """))
        placement_rows = (
            ("Requested amount", f"{v2_08_job_size.value} {v2_08_packet['resource_unit']}", "gang or topology unit"),
            ("Global free amount", f"{v2_08_placement['free_slots']} slots", "sum across all domains"),
            ("Largest contiguous domain", f"{v2_08_placement['max_contiguous']} slots", "what a topology-aware job can use without compaction"),
            ("Fragmentation index", f"{v2_08_placement['fragmentation_index']:.2f}", "free capacity outside the largest block"),
            ("Locality cost", f"{v2_08_placement['locality_cost']:.1f}", "weighted topology distance"),
            ("Placement result", "PASS" if v2_08_placement["feasible"] else "FAIL", v2_08_placement["failure_reason"]),
        )
        items.append(mo.Html(v2_08_html_table(("Quantity", "Value", "Meaning"), placement_rows)))
        if not v2_08_placement["feasible"]:
            items.append(mo.callout(mo.md(
                f"**Placement failure:** {v2_08_packet['placement_failure']}. "
                f"There are {v2_08_placement['free_slots']} free slots, but the largest contiguous "
                f"{v2_08_packet['domain_label']} has only {v2_08_placement['max_contiguous']} slots."
            ), kind="danger"))
        elif v2_08_placement["topology_penalty_pct"] > v2_08_packet["topology_limit_pct"]:
            items.append(mo.callout(mo.md(
                f"**Topology boundary crossed:** placement is feasible, but the topology penalty is "
                f"{v2_08_placement['topology_penalty_pct']:.1f}% against a "
                f"{v2_08_packet['topology_limit_pct']:.0f}% guardrail."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"**Placement passes:** the selected policy places the job with locality cost "
                f"{v2_08_placement['locality_cost']:.1f} and topology penalty "
                f"{v2_08_placement['topology_penalty_pct']:.1f}%."
            ), kind="success"))
        items.append(v2_08_prediction_feedback(
            v2_08_partC_pred.value,
            "fragmentation_matters",
            "**Correct.** The scheduler needs the right shape and location of capacity, not only a global count.",
            "**Placement trap:** global free slots can be stranded by bin packing and topology constraints.",
        ))
        items.append(mo.accordion({
            "Math Peek / Source Model - locality and fragmentation": mo.md(f"""
```
fragmentation_index = (free_slots - largest_contiguous_block) / free_slots
Cost_locality       = sum_{{i<j}} w(d(g_i, g_j))
topology_ok         = feasible and topology_penalty <= limit
```

Current values: free slots `{v2_08_placement['free_slots']}`, largest
contiguous block `{v2_08_placement['max_contiguous']}`, locality cost
`{v2_08_placement['locality_cost']:.1f}`, topology penalty
`{v2_08_placement['topology_penalty_pct']:.1f}%`.

Chapter anchor: bin packing, fragmentation, and topology-aware placement.
            """)
        }))
        items.append(v2_08_partC_checkpoint)
        if v2_08_partC_checkpoint.value is None:
            items.append(mo.callout(mo.md("Checkpoint: choose how placement should be framed in the policy memo."), kind="info"))
        return mo.vstack(items)

    def v2_08_build_part_d():
        candidates = v2_08_policy_candidates(v2_08_packet)
        selected = v2_08_evaluate_policy(
            v2_08_packet,
            v2_08_policy.value,
            v2_08_util_target.value,
            v2_08_fairness_floor.value,
            v2_08_starvation_guard.value,
            v2_08_queue,
            v2_08_priority,
            v2_08_placement,
        )
        rejected = v2_08_evaluate_policy(
            v2_08_packet,
            v2_08_rejected_policy.value,
            v2_08_util_target.value,
            v2_08_fairness_floor.value,
            v2_08_starvation_guard.value,
            v2_08_queue,
            v2_08_priority,
            v2_08_placement,
        )
        items = [
            v2_08_part_banner(
                "D",
                "Policy Must Pass Utilization, Fairness, SLO, And Starvation Guardrails",
                COLORS["RedLine"],
                (
                    f"The {v2_08_packet['report_frame']} must pick one scheduler policy, reject one alternative, "
                    "and state which amount is binding."
                ),
            ),
            mo.md("""
## Concept: Feasibility Is A Conjunction

No single metric launches an orchestration policy. Utilization, fairness, SLO,
starvation, and topology must all pass under the same workload.
            """),
            v2_08_partD_pred,
        ]
        if v2_08_partD_pred.value is None:
            items.append(mo.callout(mo.md("Commit to the policy-gate prediction before opening the candidate table."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([v2_08_policy, v2_08_rejected_policy], widths="equal"))
        items.append(mo.hstack([v2_08_util_target, v2_08_fairness_floor, v2_08_starvation_guard], widths="equal"))
        policy_rows = []
        labels = []
        pass_counts = []
        colors = []
        for key, candidate in candidates.items():
            result = v2_08_evaluate_policy(
                v2_08_packet,
                key,
                v2_08_util_target.value,
                v2_08_fairness_floor.value,
                v2_08_starvation_guard.value,
                v2_08_queue,
                v2_08_priority,
                v2_08_placement,
            )
            labels.append(candidate["label"])
            pass_counts.append(sum(1 for ok in result["checks"].values() if ok))
            colors.append(COLORS["GreenLine"] if result["feasible"] else COLORS["RedLine"])
            policy_rows.append((
                candidate["label"],
                f"{result['utilization_pct']:.1f}%",
                f"{result['fairness_pct']:.1f}%",
                f"{result['p95_latency']:.1f} {v2_08_packet['latency_unit']}",
                v2_08_fmt_hours(result["starvation_h"]),
                f"{result['topology_penalty_pct']:.1f}%",
                "PASS" if result["feasible"] else f"FAIL: {result['binding']}",
            ))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=pass_counts,
            marker_color=colors,
            text=[f"{count}/5" for count in pass_counts],
            textposition="outside",
            name="passed guardrails",
        ))
        fig.update_layout(
            height=340,
            yaxis=dict(title="Guardrails passed", range=[0, 5.5], dtick=1, gridcolor="#f1f5f9"),
            xaxis=dict(title="Scheduler policy"),
            showlegend=False,
            margin=dict(l=60, r=20, t=45, b=80),
        )
        apply_plotly_theme(fig)
        items.append(mo.as_html(fig))
        items.append(mo.Html(v2_08_html_table(
            ("Policy", "Utilization", "Fairness", "P95/SLO", "Starvation", "Topology", "Gate"),
            policy_rows,
        )))
        selected_color = COLORS["GreenLine"] if selected["feasible"] else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
          {v2_08_metric_card("Selected", selected['candidate']['label'], selected['candidate']['rationale'], selected_color, True)}
          {v2_08_metric_card("Binding", selected['binding'], "launch gate", selected_color)}
          {v2_08_metric_card("Rejected", rejected['candidate']['label'], rejected['binding'], COLORS["OrangeLine"])}
        </div>
        """))
        if selected["feasible"]:
            items.append(mo.callout(mo.md(
                f"**Policy gate passes.** {selected['candidate']['label']} satisfies utilization, fairness, "
                "SLO, starvation, and topology guardrails together."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Policy gate fails.** Binding guardrail: {selected['binding']}. "
                "Revise the selected scheduler policy or loosen a justified operating threshold."
            ), kind="danger"))
        items.append(v2_08_prediction_feedback(
            v2_08_partD_pred.value,
            "all_guardrails",
            "**Correct.** The scheduler policy launches only when every guardrail passes.",
            "**Single-metric trap:** fastest, fairest, or most utilized can still be non-launchable.",
        ))
        items.append(mo.accordion({
            "Math Peek / Source Model - guardrail conjunction": mo.md(f"""
```
feasible = utilization_ok and fairness_ok and slo_ok and starvation_ok and topology_ok
utilization_ok = min_util <= utilization_pct <= max_util
fairness_ok    = fairness_pct >= fairness_floor
slo_ok         = p95 <= track_slo
starvation_ok  = starvation_h <= starvation_guard_h
topology_ok    = placement_feasible and topology_penalty <= topology_limit
```

Selected policy: `{selected['candidate']['label']}`. Binding guardrail:
`{selected['binding']}`. Rejected alternative:
`{rejected['candidate']['label']}`.
            """)
        }))
        return mo.vstack(items)

    def build_synthesis():
        selected = v2_08_evaluate_policy(
            v2_08_packet,
            v2_08_policy.value,
            v2_08_util_target.value,
            v2_08_fairness_floor.value,
            v2_08_starvation_guard.value,
            v2_08_queue,
            v2_08_priority,
            v2_08_placement,
        )
        rejected = v2_08_evaluate_policy(
            v2_08_packet,
            v2_08_rejected_policy.value,
            v2_08_util_target.value,
            v2_08_fairness_floor.value,
            v2_08_starvation_guard.value,
            v2_08_queue,
            v2_08_priority,
            v2_08_placement,
        )
        implication_text = {
            "topology_valid_profiles": "V2-09 should profile only topology-valid placements so communication noise does not hide kernel effects.",
            "queue_before_kernel": "V2-09 kernel traces are interpretable only after queue pressure is relieved.",
            "bound_churn": "V2-09 tuning needs bounded preemption churn so profiles represent real execution rather than reload storms.",
        }.get(v2_08_v2_09_implication.value, v2_08_packet["v2_09_implication"])
        memo_rows = (
            ("Selected scheduler policy", selected["candidate"]["label"], selected["candidate"]["rationale"]),
            ("Binding resource/guardrail", selected["binding"], "all guardrails pass" if selected["feasible"] else "mitigation required"),
            ("Rejected alternative", rejected["candidate"]["label"], rejected["binding"]),
            ("Queueing evidence", f"rho {v2_08_queue['rho']:.2f}, p95 {v2_08_queue['p95_latency']:.1f} {v2_08_packet['latency_unit']}", v2_08_queue["binding"]),
            ("Preemption evidence", f"{v2_08_priority['preemption_tax_slot_min_h']:.0f} slot-min/h", v2_08_fmt_hours(v2_08_priority["starvation_wait_h"])),
            ("Placement evidence", f"fragmentation {v2_08_placement['fragmentation_index']:.2f}", f"topology penalty {v2_08_placement['topology_penalty_pct']:.1f}%"),
            ("V2-09 implication", implication_text, "performance engineering handoff"),
        )
        incomplete = []
        if v2_08_partA_pred.value is None:
            incomplete.append("Part A queueing prediction")
        if v2_08_partB_pred.value is None:
            incomplete.append("Part B priority/preemption prediction")
        if v2_08_partC_pred.value is None:
            incomplete.append("Part C placement prediction")
        if v2_08_partD_pred.value is None:
            incomplete.append("Part D policy prediction")
        if v2_08_partA_checkpoint.value is None:
            incomplete.append("Part A checkpoint")
        if v2_08_partB_checkpoint.value is None:
            incomplete.append("Part B checkpoint")
        if v2_08_partC_checkpoint.value is None:
            incomplete.append("Part C checkpoint")

        snapshot = {
            "track_id": v2_08_profile.track_id,
            "scenario_id": v2_08_variant.scenario_id,
            "selected_scheduler_policy": selected["candidate"]["label"],
            "rejected_policy": rejected["candidate"]["label"],
            "binding_guardrail": selected["binding"],
            "policy_feasible": selected["feasible"],
            "queue": {
                "arrival_multiplier": v2_08_arrival_multiplier.value,
                "heavy_mix_pct": v2_08_heavy_mix.value,
                "service_variability": v2_08_service_cv.value,
                "rho": round(v2_08_queue["rho"], 4),
                "wait_multiplier": round(v2_08_queue["wait_multiplier"], 4),
                "p95": round(v2_08_queue["p95_latency"], 4),
                "unit": v2_08_packet["latency_unit"],
            },
            "priority": {
                "preemption_aggressiveness_pct": v2_08_preempt_pct.value,
                "urgent_share_pct": v2_08_urgent_share.value,
                "checkpoint_interval_min": v2_08_checkpoint_min.value,
                "urgent_latency": round(v2_08_priority["urgent_latency"], 4),
                "starvation_wait_h": round(v2_08_priority["starvation_wait_h"], 4),
                "preemption_tax_slot_min_h": round(v2_08_priority["preemption_tax_slot_min_h"], 4),
            },
            "placement": {
                "job_size": v2_08_job_size.value,
                "placement_policy": v2_08_placement_policy.value,
                "fragmentation_index": round(v2_08_placement["fragmentation_index"], 4),
                "locality_cost": round(v2_08_placement["locality_cost"], 4),
                "topology_penalty_pct": round(v2_08_placement["topology_penalty_pct"], 4),
                "feasible": v2_08_placement["feasible"],
            },
            "policy": {
                "utilization_target_pct": v2_08_util_target.value,
                "fairness_floor_pct": v2_08_fairness_floor.value,
                "starvation_guard_h": v2_08_starvation_guard.value,
                "utilization_pct": round(selected["utilization_pct"], 4),
                "fairness_pct": round(selected["fairness_pct"], 4),
                "p95": round(selected["p95_latency"], 4),
                "topology_penalty_pct": round(selected["topology_penalty_pct"], 4),
            },
            "v2_09_implication": implication_text,
        }
        if not incomplete:
            ledger.save(chapter=v2_08_chapter, design={
                "lab_id": v2_08_metadata.lab_id,
                "track_id": v2_08_profile.track_id,
                "scenario_id": v2_08_variant.scenario_id,
                "selected_scheduler_policy": selected["candidate"]["label"],
                "rejected_policy": rejected["candidate"]["label"],
                "binding_guardrail": selected["binding"],
                "policy_feasible": selected["feasible"],
                "v2_09_implication": implication_text,
                "result_snapshot": snapshot,
            })
        report = build_lab_report(
            v2_08_metadata,
            student_id=v2_08_student_id.value or "",
            track=v2_08_profile.label,
            scenario=v2_08_variant.workload_summary,
            learning_objectives=(
                "Explain why queueing pressure emerges from job mix and arrival rate.",
                "Quantify the priority/preemption trade-off between urgent latency and starvation risk.",
                "Use placement evidence to distinguish global free capacity from topology-valid capacity.",
                "Choose a scheduler policy by applying utilization, fairness, SLO, starvation, and topology guardrails together.",
            ),
            predictions={
                "part_a_queueing": v2_08_partA_pred.value,
                "part_b_priority_preemption": v2_08_partB_pred.value,
                "part_c_placement": v2_08_partC_pred.value,
                "part_d_policy_gate": v2_08_partD_pred.value,
            },
            knob_settings={
                "arrival_multiplier": v2_08_arrival_multiplier.value,
                "heavy_mix_pct": v2_08_heavy_mix.value,
                "service_variability": v2_08_service_cv.value,
                "preemption_aggressiveness_pct": v2_08_preempt_pct.value,
                "urgent_share_pct": v2_08_urgent_share.value,
                "checkpoint_interval_min": v2_08_checkpoint_min.value,
                "job_size": v2_08_job_size.value,
                "placement_policy": v2_08_placement_policy.value,
                "utilization_target_pct": v2_08_util_target.value,
                "fairness_floor_pct": v2_08_fairness_floor.value,
                "starvation_guard_h": v2_08_starvation_guard.value,
            },
            binding_constraints={
                "queue_binding": v2_08_queue["binding"],
                "preemption_binding": v2_08_priority["binding"],
                "placement_reason": v2_08_placement["failure_reason"],
                "policy_binding": selected["binding"],
            },
            decisions={
                "part_a_checkpoint": v2_08_partA_checkpoint.value,
                "part_b_checkpoint": v2_08_partB_checkpoint.value,
                "part_c_checkpoint": v2_08_partC_checkpoint.value,
                "selected_scheduler_policy": selected["candidate"]["label"],
                "rejected_policy": rejected["candidate"]["label"],
                "v2_09_implication": implication_text,
            },
            reflections={
                "policy_memo": f"Select {selected['candidate']['label']}; reject {rejected['candidate']['label']}; binding guardrail: {selected['binding']}.",
            },
            evidence_summary={
                "queue_p95": f"{v2_08_queue['p95_latency']:.1f} {v2_08_packet['latency_unit']}",
                "preemption_tax": f"{v2_08_priority['preemption_tax_slot_min_h']:.0f} slot-min/h",
                "fragmentation_index": round(v2_08_placement["fragmentation_index"], 3),
                "policy_feasible": selected["feasible"],
            },
            final_decision={
                "selected_scheduler_policy": selected["candidate"]["label"],
                "binding_guardrail": selected["binding"],
                "rejected_alternative": rejected["candidate"]["label"],
                "v2_09_implication": implication_text,
            },
            big_takeaways=(
                "High utilization can be the cause of queueing failure rather than proof of health.",
                "Preemption is a trade between urgent latency and starvation/recovery tax.",
                "Placement is performance because topology and fragmentation change usable capacity.",
                "A scheduler policy launches only as a conjunction of guardrails.",
            ),
            residual_risk=selected["binding"] if not selected["feasible"] else "Policy assumptions must be revisited as workload mix and topology change.",
            source_trace={
                "book_anchor": v2_08_metadata.book_anchor,
                "chapter_sections": (
                    "Scheduling objectives and conflicts",
                    "Queuing theory of GPU clusters",
                    "Bin packing",
                    "Topology-aware placement",
                    "Priority preemption cascades",
                    "Hierarchical fair-share",
                ),
                "hardware_ref": v2_08_packet["hardware_ref"],
                "model_ref": v2_08_packet["model_ref"],
                "system_ref": v2_08_packet["system_ref"],
                "local_solver": "v2_08_queue_model, v2_08_priority_model, v2_08_placement_model, v2_08_evaluate_policy",
            },
            result_snapshot=snapshot,
            incomplete_fields=tuple(incomplete),
        )
        return mo.vstack([
            mo.md("## Synthesis - Orchestration Policy Memo"),
            v2_08_student_id,
            v2_08_v2_09_implication,
            mo.callout(mo.md(
                f"**Memo frame for {v2_08_packet['label']}:** select "
                f"`{selected['candidate']['label']}`, name `{selected['binding']}` as the binding guardrail, "
                f"reject `{rejected['candidate']['label']}`, and carry forward: {implication_text}"
            ), kind="success" if selected["feasible"] else "warn"),
            mo.Html(v2_08_html_table(("Memo field", "Decision/evidence", "Report framing"), memo_rows)),
            report_export_panel(report),
        ])

    tabs = mo.ui.tabs({
        "Opening": v2_08_opening(),
        "Part A: Queueing Pressure": v2_08_build_part_a(),
        "Part B: Priority And Starvation": v2_08_build_part_b(),
        "Part C: Placement And Topology": v2_08_build_part_c(),
        "Part D: Policy Gate": v2_08_build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(mo, v2_08_metadata, v2_08_profile):
    mo.Html(f"""
    <div class="lab-hud">
      <span class="hud-label">LAB</span>
      <span class="hud-value">{v2_08_metadata.lab_id}</span>
      <span class="hud-label">TRACK</span>
      <span class="hud-value">{v2_08_profile.label}</span>
      <span style="flex:1;"></span>
      <span class="hud-label">STATUS</span>
      <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
