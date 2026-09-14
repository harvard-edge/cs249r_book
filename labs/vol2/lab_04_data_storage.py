import marimo

__generated_with = "0.23.3"
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
    from mlsysim import Hardware, Models, ReferenceStats
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
        Hardware,
        LAB_CSS,
        MathPeek,
        Models,
        ReferenceStats,
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
    v2_04_metadata = get_lab_metadata("vol2/lab_04_data_storage.py")
    v2_04_lab_id = v2_04_metadata.lab_id
    return v2_04_lab_id, v2_04_metadata


@app.cell(hide_code=True)
def _(mo):
    v2_04_track_picker = mo.ui.dropdown(
        options={
            "⚡ TinyML Track (Microcontrollers & Wearables / Oura Ring & Cortex-M55)": "oura_ring",
            "📱 Mobile Track (On-Device Personal AI / iPhone & Apple Silicon M4)": "iphone",
            "🤖 Edge & Embodied Track (Robotics & Drones / Robotaxi & Jetson Orin)": "robotaxi",
            "☁️ Cloud Supercomputing Track (NVIDIA H100 / B200 Clusters & 3D Parallelism / Scale)": "cloud_fleet",
        },
        value="☁️ Cloud Supercomputing Track (NVIDIA H100 / B200 Clusters & 3D Parallelism / Scale)",
        label="Select Course / Industry Track",
    )
    v2_04_track_picker
    return (v2_04_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v2_04_lab_id,
    v2_04_track_picker,
):
    v2_04_track_id = v2_04_track_picker.value
    v2_04_profile = get_track_profile(v2_04_track_id)
    v2_04_variant = get_lab_track_variant(v2_04_lab_id, v2_04_profile.track_id)
    v2_04_hardware = resolve_mlsysim_ref(v2_04_variant.hardware_ref)
    v2_04_model = resolve_mlsysim_ref(v2_04_variant.model_ref)
    # Cross-tier hardware targets: Hardware.Tiny.CortexM55, Hardware.Mobile.AppleM4, Hardware.Edge.JetsonOrin, Hardware.Cloud.H100
    return v2_04_profile, v2_04_variant


@app.cell
def _():
    def v2_04_qty_to_float(value, unit, default):
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

    def v2_04_escape(value):
        import html as _html

        return _html.escape(str(value))

    def v2_04_format(value, digits=1, suffix=""):
        value = float(value)
        if abs(value) >= 1000:
            text = f"{value:,.{digits}f}"
        else:
            text = f"{value:.{digits}f}"
        if digits == 0:
            text = text.split(".")[0]
        return f"{text}{suffix}"

    def v2_04_status_text(feasible):
        return "PASS" if feasible else "FAIL"

    def v2_04_fields_html(fields):
        return "\n".join(
            (
                '<div class="mlsysbook-field">'
                f"<strong>{v2_04_escape(key)}</strong>{v2_04_escape(value)}"
                "</div>"
            )
            for key, value in fields.items()
        )

    def v2_04_table_html(headers, rows, numeric=()):
        numeric_cols = set(numeric)
        header_html = "".join(f"<th>{v2_04_escape(header)}</th>" for header in headers)
        body_rows = []
        for row in rows:
            cells = []
            for idx, value in enumerate(row):
                align = "right" if idx in numeric_cols else "left"
                cells.append(f'<td style="text-align:{align};">{v2_04_escape(value)}</td>')
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        return f"""
        <div style="overflow-x:auto; margin-top:14px;">
          <table class="mlsysbook-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        </div>
        """

    def v2_04_callout_html(title, body, kind="info"):
        palette = {
            "ok": ("#ECFDF5", "#008F45"),
            "warn": ("#FFF7ED", "#CC5500"),
            "fail": ("#FEF2F2", "#CB202D"),
            "info": ("#EBF4FA", "#006395"),
        }
        background, border = palette.get(kind, palette["info"])
        return f"""
        <div class="mlsysbook-callout" style="background:{background}; border-color:{border};">
          <strong>{v2_04_escape(title)}:</strong> {v2_04_escape(body)}
        </div>
        """

    def v2_04_prediction_html(title, predicted, actual, labels):
        if predicted is None:
            return v2_04_callout_html(
                title,
                "Commit to a structured prediction before using this evidence.",
                "warn",
            )
        predicted_label = labels.get(predicted, predicted)
        actual_label = labels.get(actual, actual)
        if predicted == actual:
            return v2_04_callout_html(title, f"Prediction matched: {actual_label}.", "ok")
        return v2_04_callout_html(
            title,
            f"You predicted {predicted_label}; measured evidence points to {actual_label}.",
            "warn",
        )

    def v2_04_option_labels(options):
        return {details["label"]: option_id for option_id, details in options.items()}

    def v2_04_first_feasible(rows):
        feasible = [row for row in rows if row["feasible"]]
        candidates = feasible if feasible else rows
        return min(candidates, key=lambda row: row["score"])

    def v2_04_ratio_label(ratio):
        return f"{ratio:.2f}x"

    return (
        v2_04_callout_html,
        v2_04_fields_html,
        v2_04_format,
        v2_04_option_labels,
        v2_04_prediction_html,
        v2_04_qty_to_float,
        v2_04_ratio_label,
        v2_04_status_text,
        v2_04_table_html,
    )


@app.cell
def _(Hardware, Models, ReferenceStats, v2_04_profile, v2_04_qty_to_float):
    def v2_04_storage_profile(track_profile):
        h100_nvme_gb_s = v2_04_qty_to_float(Hardware.Cloud.H100.storage.bandwidth, "GB/s", 7.0)
        h100_hbm_tb_s = v2_04_qty_to_float(Hardware.Cloud.H100.memory.bandwidth, "TB/s", 3.35)
        h100_checkpoint_gb = v2_04_qty_to_float(
            Models.Language.GPT3.parameters * ReferenceStats.StorageTrainingCorpus.CheckpointBytesPerParameter,
            "GB",
            1750.0,
        )
        tokenized_corpus_tb = v2_04_qty_to_float(
            ReferenceStats.StorageTrainingCorpus.TokenizedText,
            "TB",
            6.0,
        )
        oura_flash_mb_s = v2_04_qty_to_float(Hardware.Tiny.OuraRing.memory.flash_bandwidth, "MB/s", 40.0)
        robotaxi_storage_gb_s = v2_04_qty_to_float(Hardware.Edge.RoboTaxi.storage.bandwidth, "GB/s", 1.0)

        profiles = {
            "iphone": {
                "scenario": "A mobile camera feature keeps local serving responsive while uploading consented failure evidence for retraining.",
                "stakeholder_pressure": "avoid a privacy and radio backlog without slowing the on-device feature",
                "consumer_label": "local feature windows",
                "training_link": "consented failures feed retraining",
                "serving_link": "local cache feeds interactive serving",
                "consumer_count": 1,
                "record_mb": 6.0,
                "iteration_s": 1.5,
                "target_utilization": 1.0,
                "demand_default": 1.0,
                "demand_min": 0.4,
                "demand_max": 3.0,
                "demand_step": 0.1,
                "stages": (
                    {"name": "local flash read", "capacity_mb_s": 450.0, "factor": 1.0, "group": "storage_throughput"},
                    {"name": "privacy preprocessing", "capacity_mb_s": 10.0, "factor": 0.90, "group": "preprocess"},
                    {"name": "local evidence cache", "capacity_mb_s": 45.0, "factor": 0.55, "group": "storage_throughput"},
                    {"name": "consented upload", "capacity_mb_s": 2.0, "factor": 0.65, "group": "network"},
                ),
                "placement_request_mb": 6.0,
                "latency_budget_ms": 180.0,
                "movement_budget_gb_day": 120.0,
                "egress_budget_day": 4.0,
                "freshness_budget_min": 120.0,
                "default_placement": "remote_source",
                "placements": {
                    "remote_source": {"label": "Cloud raw upload", "class": "remote", "base_latency_ms": 90.0, "tail_ms": 180.0, "bandwidth_gbps": 0.025, "cache_hit": 0.05, "egress_per_gb": 0.08, "freshness_min": 15.0, "risk": "raw personal data and radio energy dominate"},
                    "regional_cache": {"label": "Phone summary cache", "class": "tiered", "base_latency_ms": 28.0, "tail_ms": 55.0, "bandwidth_gbps": 0.12, "cache_hit": 0.72, "egress_per_gb": 0.02, "freshness_min": 45.0, "risk": "debugging evidence is summarized"},
                    "local_cache": {"label": "On-device raw buffer", "class": "local", "base_latency_ms": 4.0, "tail_ms": 12.0, "bandwidth_gbps": 3.6, "cache_hit": 0.92, "egress_per_gb": 0.00, "freshness_min": 180.0, "risk": "local retention and deletion policy must be enforced"},
                },
                "state_gb": 8.0,
                "node_count": 1,
                "local_write_gb_s": 0.45,
                "durable_write_gb_s": 0.025,
                "pause_budget_s": 2.0,
                "restore_budget_min": 20.0,
                "write_storm_budget_gb_h": 90.0,
                "evidence_floor_pct": 90.0,
                "checkpoint_default": "sync_barrier",
                "interval_min": 2.0,
                "interval_max": 60.0,
                "interval_default": 10.0,
                "interval_step": 1.0,
                "lifecycle_daily_gb": 42.0,
                "min_retention_days": 2.0,
                "monthly_budget": 45.0,
                "durability_floor_pct": 98.0,
                "freshness_min": 15.0,
                "freshness_max": 720.0,
                "freshness_default": 120.0,
                "freshness_step": 15.0,
                "tier_prices": {"hot": 0.09, "warm": 0.035, "cold": 0.012},
                "next_lab": "V2-05 should avoid distributed training plans that require raw mobile data to move continuously.",
            },
            "oura_ring": {
                "scenario": "A wearable firmware team must decide which biosignal windows stay in flash and which summaries sync through the phone.",
                "stakeholder_pressure": "preserve health evidence without overflowing flash or spending the BLE duty cycle",
                "consumer_label": "biosignal windows",
                "training_link": "summaries and rare snippets feed model updates",
                "serving_link": "fresh local features feed always-on inference",
                "consumer_count": 1,
                "record_mb": 0.006,
                "iteration_s": 0.40,
                "target_utilization": 1.0,
                "demand_default": 1.0,
                "demand_min": 0.3,
                "demand_max": 5.0,
                "demand_step": 0.1,
                "stages": (
                    {"name": "sensor window write", "capacity_mb_s": min(0.030, oura_flash_mb_s), "factor": 1.0, "group": "storage_throughput"},
                    {"name": "feature extraction", "capacity_mb_s": 0.024, "factor": 1.10, "group": "preprocess"},
                    {"name": "flash retention", "capacity_mb_s": 0.018, "factor": 0.80, "group": "storage_throughput"},
                    {"name": "phone BLE sync", "capacity_mb_s": 0.012, "factor": 1.00, "group": "network"},
                ),
                "placement_request_mb": 0.006,
                "latency_budget_ms": 2000.0,
                "movement_budget_gb_day": 0.75,
                "egress_budget_day": 0.25,
                "freshness_budget_min": 720.0,
                "default_placement": "remote_source",
                "placements": {
                    "remote_source": {"label": "Raw phone/cloud sync", "class": "remote", "base_latency_ms": 950.0, "tail_ms": 1800.0, "bandwidth_gbps": 0.001, "cache_hit": 0.05, "egress_per_gb": 0.08, "freshness_min": 60.0, "risk": "raw physiology overwhelms sync and retention"},
                    "regional_cache": {"label": "Phone-side summaries", "class": "tiered", "base_latency_ms": 220.0, "tail_ms": 420.0, "bandwidth_gbps": 0.006, "cache_hit": 0.70, "egress_per_gb": 0.02, "freshness_min": 240.0, "risk": "phone availability becomes part of the data path"},
                    "local_cache": {"label": "Ring rare snippets", "class": "local", "base_latency_ms": 25.0, "tail_ms": 80.0, "bandwidth_gbps": 0.040, "cache_hit": 0.90, "egress_per_gb": 0.00, "freshness_min": 720.0, "risk": "snippet trigger quality controls the evidence set"},
                },
                "state_gb": 0.020,
                "node_count": 1,
                "local_write_gb_s": 0.040,
                "durable_write_gb_s": 0.002,
                "pause_budget_s": 0.80,
                "restore_budget_min": 90.0,
                "write_storm_budget_gb_h": 0.45,
                "evidence_floor_pct": 85.0,
                "checkpoint_default": "sync_barrier",
                "interval_min": 5.0,
                "interval_max": 180.0,
                "interval_default": 30.0,
                "interval_step": 5.0,
                "lifecycle_daily_gb": 1.55,
                "min_retention_days": 30.0,
                "monthly_budget": 5.0,
                "durability_floor_pct": 95.0,
                "freshness_min": 60.0,
                "freshness_max": 1440.0,
                "freshness_default": 720.0,
                "freshness_step": 60.0,
                "tier_prices": {"hot": 0.20, "warm": 0.06, "cold": 0.018},
                "next_lab": "V2-05 should train from summarized windows and rare snippets, not a continuous raw wearable stream.",
            },
            "robotaxi": {
                "scenario": "A vehicle platform team must retain enough sensor evidence for rare-event replay while preventing depot upload from backing up.",
                "stakeholder_pressure": "keep safety evidence local and move only the highest-value logs off vehicle",
                "consumer_label": "sensor evidence bundles",
                "training_link": "rare-event logs feed offline perception training",
                "serving_link": "local triage feeds the safety path and incident replay",
                "consumer_count": 8,
                "record_mb": 200.0,
                "iteration_s": 1.0,
                "target_utilization": 1.0,
                "demand_default": 1.0,
                "demand_min": 0.2,
                "demand_max": 3.0,
                "demand_step": 0.1,
                "stages": (
                    {"name": "sensor ingest", "capacity_mb_s": 1400.0, "factor": 1.00, "group": "storage_throughput"},
                    {"name": "local event triage", "capacity_mb_s": 900.0, "factor": 0.65, "group": "preprocess"},
                    {"name": "vehicle NVMe write", "capacity_mb_s": robotaxi_storage_gb_s * 1000.0, "factor": 0.35, "group": "storage_throughput"},
                    {"name": "depot upload", "capacity_mb_s": 80.0, "factor": 0.10, "group": "network"},
                ),
                "placement_request_mb": 1600.0,
                "latency_budget_ms": 80.0,
                "movement_budget_gb_day": 9500.0,
                "egress_budget_day": 900.0,
                "freshness_budget_min": 1440.0,
                "default_placement": "remote_source",
                "placements": {
                    "remote_source": {"label": "Full cloud upload", "class": "remote", "base_latency_ms": 700.0, "tail_ms": 2500.0, "bandwidth_gbps": 0.64, "cache_hit": 0.02, "egress_per_gb": 0.08, "freshness_min": 240.0, "risk": "raw location and bystander data leave the vehicle"},
                    "regional_cache": {"label": "Depot event cache", "class": "tiered", "base_latency_ms": 120.0, "tail_ms": 420.0, "bandwidth_gbps": 8.0, "cache_hit": 0.68, "egress_per_gb": 0.02, "freshness_min": 720.0, "risk": "depot bandwidth and route coverage set the replay delay"},
                    "local_cache": {"label": "Vehicle-local triage", "class": "local", "base_latency_ms": 8.0, "tail_ms": 25.0, "bandwidth_gbps": 16.0, "cache_hit": 0.94, "egress_per_gb": 0.00, "freshness_min": 1440.0, "risk": "triage rules can discard unknown rare events"},
                },
                "state_gb": 512.0,
                "node_count": 32,
                "local_write_gb_s": 1.0,
                "durable_write_gb_s": 8.0,
                "pause_budget_s": 30.0,
                "restore_budget_min": 180.0,
                "write_storm_budget_gb_h": 9500.0,
                "evidence_floor_pct": 95.0,
                "checkpoint_default": "sync_barrier",
                "interval_min": 5.0,
                "interval_max": 120.0,
                "interval_default": 20.0,
                "interval_step": 5.0,
                "lifecycle_daily_gb": 48000.0,
                "min_retention_days": 7.0,
                "monthly_budget": 12000.0,
                "durability_floor_pct": 99.0,
                "freshness_min": 60.0,
                "freshness_max": 2880.0,
                "freshness_default": 1440.0,
                "freshness_step": 60.0,
                "tier_prices": {"hot": 0.07, "warm": 0.024, "cold": 0.006},
                "next_lab": "V2-05 should train from event-mined shards and keep rare-event replay local until upload catches up.",
            },
            "cloud_fleet": {
                "scenario": "A fleet SRE must keep H100 training and serving jobs fed from object storage, preprocessing workers, local cache, and checkpoint tiers.",
                "stakeholder_pressure": "prevent accelerator starvation while avoiding checkpoint storms and runaway lifecycle cost",
                "consumer_label": "H100 training steps",
                "training_link": "object shards and checkpoints feed distributed training",
                "serving_link": "model weights and fresh features feed serving scale-up",
                "consumer_count": 64,
                "record_mb": 38.4,
                "iteration_s": 0.2,
                "target_utilization": 1.0,
                "demand_default": 1.0,
                "demand_min": 0.2,
                "demand_max": 2.5,
                "demand_step": 0.05,
                "stages": (
                    {"name": "object store read", "capacity_mb_s": 5200.0, "factor": 1.00, "group": "storage_throughput"},
                    {"name": "preprocessing workers", "capacity_mb_s": 9000.0, "factor": 0.75, "group": "preprocess"},
                    {"name": "parallel file stage", "capacity_mb_s": 11000.0, "factor": 0.90, "group": "storage_throughput"},
                    {"name": "local NVMe cache", "capacity_mb_s": h100_nvme_gb_s * 1000.0 * 4.0, "factor": 0.55, "group": "storage_throughput"},
                ),
                "placement_request_mb": 2457.6,
                "latency_budget_ms": 250.0,
                "movement_budget_gb_day": 450000.0,
                "egress_budget_day": 18000.0,
                "freshness_budget_min": 120.0,
                "default_placement": "remote_source",
                "placements": {
                    "remote_source": {"label": "Remote object-store read", "class": "remote", "base_latency_ms": 85.0, "tail_ms": 220.0, "bandwidth_gbps": 40.0, "cache_hit": 0.10, "egress_per_gb": 0.04, "freshness_min": 10.0, "risk": "object-store tails starve expensive accelerators"},
                    "regional_cache": {"label": "Regional shard cache", "class": "tiered", "base_latency_ms": 18.0, "tail_ms": 45.0, "bandwidth_gbps": 160.0, "cache_hit": 0.76, "egress_per_gb": 0.010, "freshness_min": 35.0, "risk": "cache invalidation and regional governance become correctness risks"},
                    "local_cache": {"label": "Local NVMe pre-stage", "class": "local", "base_latency_ms": 4.0, "tail_ms": 12.0, "bandwidth_gbps": 224.0, "cache_hit": 0.91, "egress_per_gb": 0.002, "freshness_min": 90.0, "risk": "pre-stage misses and skewed shards create tail stalls"},
                },
                "state_gb": h100_checkpoint_gb,
                "node_count": 256,
                "local_write_gb_s": h100_nvme_gb_s,
                "durable_write_gb_s": 100.0,
                "pause_budget_s": 15.0,
                "restore_budget_min": 60.0,
                "write_storm_budget_gb_h": 12000.0,
                "evidence_floor_pct": 95.0,
                "checkpoint_default": "sync_barrier",
                "interval_min": 2.0,
                "interval_max": 60.0,
                "interval_default": 10.0,
                "interval_step": 1.0,
                "lifecycle_daily_gb": 8000.0,
                "min_retention_days": 14.0,
                "monthly_budget": 35000.0,
                "durability_floor_pct": 99.9,
                "freshness_min": 15.0,
                "freshness_max": 720.0,
                "freshness_default": 120.0,
                "freshness_step": 15.0,
                "tier_prices": {"hot": 0.023, "warm": 0.012, "cold": 0.004},
                "next_lab": "V2-05 should include storage bandwidth and checkpoint write storms in the distributed-training parallelism plan.",
                "chapter_amount_note": f"H100 HBM is {h100_hbm_tb_s:.2f} TB/s while one local NVMe path is {h100_nvme_gb_s:.1f} GB/s; the tokenized running corpus is {tokenized_corpus_tb:.1f} TB.",
            },
        }

        selected = dict(profiles[track_profile.track_id])
        selected["track_id"] = track_profile.track_id
        selected["track_label"] = track_profile.label
        selected["stakeholder"] = track_profile.stakeholder
        selected["hardware_ref"] = track_profile.hardware_ref
        selected["system_ref"] = track_profile.system_ref or "not applicable"
        selected["source_policy"] = track_profile.source_policy
        selected.setdefault(
            "chapter_amount_note",
            "Scenario thresholds are notebook-local lab assumptions tied to the chapter storage hierarchy.",
        )
        return selected

    def v2_04_checkpoint_policies():
        return {
            "sync_barrier": {
                "label": "Synchronous durable barrier",
                "class": "sync",
                "fraction": 1.00,
                "write_amp": 1.00,
                "evidence_pct": 98.0,
                "restore_multiplier": 1.00,
                "risk": "training or serving pauses on the slowest durable write",
            },
            "async_local": {
                "label": "Async local staging",
                "class": "async",
                "fraction": 1.00,
                "write_amp": 1.15,
                "evidence_pct": 88.0,
                "restore_multiplier": 1.25,
                "risk": "fast pause, but evidence is incomplete until durable copy is verified",
            },
            "incremental_verified": {
                "label": "Incremental verified checkpoints",
                "class": "incremental",
                "fraction": 0.25,
                "write_amp": 0.35,
                "evidence_pct": 96.0,
                "restore_multiplier": 1.45,
                "risk": "restore must replay an incremental chain",
            },
            "fast_unverified": {
                "label": "Fast unverified snapshots",
                "class": "unverified",
                "fraction": 0.12,
                "write_amp": 0.18,
                "evidence_pct": 70.0,
                "restore_multiplier": 2.20,
                "risk": "low pause does not prove a restorable checkpoint",
            },
        }

    def v2_04_lifecycle_policies(storage_profile):
        min_days = storage_profile["min_retention_days"]
        hot_recent = min(2.0, min_days)
        warm_rest = max(0.0, min_days - hot_recent)
        return {
            "raw_hot": {
                "label": "Full raw hot retention",
                "class": "hot",
                "hot_days": min_days,
                "warm_days": 0.0,
                "cold_days": 0.0,
                "reduction": 1.00,
                "lag_min": max(15.0, storage_profile["freshness_min"]),
                "durability_pct": 99.0,
                "risk": "fresh evidence is expensive and can overflow close storage",
            },
            "tiered_policy": {
                "label": "Tiered hot/warm/cold policy",
                "class": "tiered",
                "hot_days": hot_recent,
                "warm_days": warm_rest,
                "cold_days": max(min_days, 7.0),
                "reduction": 0.38,
                "lag_min": min(storage_profile["freshness_default"], storage_profile["freshness_budget_min"]),
                "durability_pct": 99.9,
                "risk": "cache invalidation and restore drills must be operational",
            },
            "summary_first": {
                "label": "Summaries plus rare raw snippets",
                "class": "summary",
                "hot_days": min(0.5, min_days),
                "warm_days": min_days,
                "cold_days": 0.0,
                "reduction": 0.08,
                "lag_min": min(storage_profile["freshness_max"], storage_profile["freshness_default"] * 1.5),
                "durability_pct": 97.0,
                "risk": "raw failures can be lost if the trigger misses them",
            },
            "cold_archive": {
                "label": "Cold archive after ingest",
                "class": "archive",
                "hot_days": 0.0,
                "warm_days": 0.0,
                "cold_days": min_days * 3.0,
                "reduction": 1.00,
                "lag_min": storage_profile["freshness_max"] * 1.5,
                "durability_pct": 99.99,
                "risk": "cheap retention fails freshness and restore-time expectations",
            },
        }

    v2_04_storage = v2_04_storage_profile(v2_04_profile)
    v2_04_checkpoint_policy_specs = v2_04_checkpoint_policies()
    v2_04_lifecycle_policy_specs = v2_04_lifecycle_policies(v2_04_storage)
    return (
        v2_04_checkpoint_policy_specs,
        v2_04_lifecycle_policy_specs,
        v2_04_storage,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    track_arc_context,
    track_context,
    v2_04_profile,
    v2_04_storage,
    v2_04_variant,
):
    header_html = mo.Html(f"""
    <div class="mlsysbook-lab-shell" style="margin-bottom: 20px;">
        <div style="border-left: 4px solid #A51C30; padding: 12px 18px; background: white; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #A51C30; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
                ML Systems Textbook &middot; Volume II &middot; Chapter 4 &middot; Foundational Lab 04
            </div>
            <h1 style="margin: 0 0 6px 0; color: #0F172A; font-size: 1.85rem; font-weight: 800;">
                The Data Storage Fuel Line: Ingestion, Locality, and Checkpointing
            </h1>
            <p style="margin: 0 0 12px 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">
                Storage feeds training and serving. Throughput, locality, consistency, checkpoint storms, and lifecycle policy shape
                system capacity because the consumer only benefits from data that arrives in the right place, at the right time,
                with enough recovery evidence.
            </p>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="mlsysbook-chip" style="background: #FEE2E2; color: #991B1B; font-weight: 700;">Track: {v2_04_profile.label}</span>
                <span class="mlsysbook-chip" style="background: #E0F2FE; color: #0369A1;">Stakeholder: {v2_04_storage['stakeholder']}</span>
                <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155;">Hardware: {v2_04_storage['hardware_ref']}</span>
                <span class="mlsysbook-chip" style="background: #F1F5F9; color: #334155;">System: {v2_04_storage['system_ref']}</span>
                <span class="mlsysbook-chip" style="background: #FEF3C7; color: #92400E;">Focus: Storage Fuel Line &amp; I/O Walls</span>
                <span class="mlsysbook-chip" style="background: #EDE9FE; color: #5B21B6;">Deliverable: Storage Architecture Memo</span>
            </div>
        </div>
    </div>
    """)

    scenario_panel = mo.Html(f"""
    <div class="mlsysbook-panel" style="margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #0F172A; font-size: 1.1rem;">System Scenario: {v2_04_profile.label} Data Engine</h3>
        <p style="color: #334155; font-size: 0.9rem; line-height: 1.6;">
            {v2_04_storage['scenario']} You are operating as the <strong>{v2_04_storage['stakeholder']}</strong>
            managing data flows for <strong>{v2_04_storage['consumer_label']}</strong>. Your pipeline must feed
            distributed training (<i>{v2_04_storage['training_link']}</i>) while sustaining low-latency serving
            (<i>{v2_04_storage['serving_link']}</i>).
        </p>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-top: 12px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;">The Architectural Invariants of Data Storage:</div>
            <ul class="mlsysbook-list" style="margin: 0; padding-left: 1.25rem; font-size: 0.85rem; color: #334155; line-height: 1.6;">
                <li><strong>The Fuel Line Invariant:</strong> Compute throughput is strictly bounded by data arrival rate: <i>B</i><sub>req</sub> = <i>N</i> &middot; &eta; &middot; <i>D</i><sub>step</sub> / <i>T</i><sub>iter</sub>. Accelerators starved of data waste expensive CapEx.</li>
                <li><strong>The Locality &amp; Bandwidth Hierarchy:</strong> Bandwidth drops by orders of magnitude across storage tiers (HBM &gg; Local NVMe &gg; Remote Object Store). Working sets must be aggressively staged or compressed to avoid pipeline stalls.</li>
                <li><strong>The Checkpoint Trade-Off:</strong> Checkpointing trades synchronous pause time and write amplification against lost work during failure recovery: <i>T</i><sub>pause</sub> = State / BW<sub>write</sub>. Fast unverified snapshots save runtime but compromise restore guarantees.</li>
                <li><strong>The Lifecycle Guardrail:</strong> Retention economics compound over fleet operational lifetime: Cost<sub>monthly</sub> = &sum; Footprint<sub>i</sub> &times; <i>P</i><sub>i</sub>. Data tiering must satisfy freshness, cost, and durability simultaneously.</li>
            </ul>
        </div>
    </div>
    """)

    objectives_panel = mo.Html(f"""
    <div class="mlsysbook-panel" style="border-left: 4px solid #006395; margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">LEARNING OBJECTIVES</div>
        <ul class="mlsysbook-list" style="margin: 0; padding-left: 1.25rem; font-size: 0.88rem; color: #334155; line-height: 1.6;">
            <li><strong>Size the data fuel line:</strong> Use the pipeline equation to identify ingestion bottlenecks, queue backlogs, and consumer starvation.</li>
            <li><strong>Analyze locality and placement hierarchies:</strong> Quantify trade-offs across latency, data movement cost, egress bandwidth, and staleness.</li>
            <li><strong>Evaluate checkpointing protocols:</strong> Balance synchronous pause latency and write storms against verifiable recovery evidence.</li>
            <li><strong>Defend a durable lifecycle policy:</strong> Optimize multi-tier retention economics while strictly adhering to durability and freshness SLAs.</li>
        </ul>
        <div style="margin-top: 10px; font-size: 0.84rem; color: #0284C7; font-style: italic;">
            CORE QUESTION: "How does storage hierarchy and placement constrain end-to-end ML throughput, and how do we co-design checkpointing and data lifecycle policies?"
        </div>
    </div>
    """)

    track_mission_card = track_context(v2_04_profile)
    arc_card = track_arc_context(v2_04_profile, v2_04_variant.lab_id)

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
def _(
    mo,
    v2_04_checkpoint_policy_specs,
    v2_04_lifecycle_policy_specs,
    v2_04_option_labels,
    v2_04_storage,
):
    v2_04_a_prediction = mo.ui.radio(
        options={
            "Storage/read throughput will bind": "storage_throughput",
            "Preprocessing will bind": "preprocess",
            "Network or upload will bind": "network",
            "No backlog appears in this envelope": "no_backlog",
        },
        label="Part A prediction: which amount blocks the consumer first?",
    )
    v2_04_demand_multiplier = mo.ui.slider(
        start=v2_04_storage["demand_min"],
        stop=v2_04_storage["demand_max"],
        value=v2_04_storage["demand_default"],
        step=v2_04_storage["demand_step"],
        label="Data demand pressure",
    )
    v2_04_a_checkpoint = mo.ui.radio(
        options={
            "Add throughput or prefetch at the bottleneck": "add_throughput",
            "Reduce data demand before storage": "reduce_demand",
            "Move the consumer closer to storage": "move_closer",
            "Accept the backlog and document the violation": "accept_backlog",
        },
        label="Part A checkpoint: what decision follows from the backlog evidence?",
    )

    v2_04_b_prediction = mo.ui.radio(
        options={
            "Remote placement survives": "remote",
            "Tiered regional/cache placement survives": "tiered",
            "Local placement survives": "local",
            "No placement satisfies the guardrails": "none",
        },
        label="Part B prediction: which placement class can satisfy the guardrails?",
    )
    v2_04_working_set_pressure = mo.ui.slider(
        start=0.5,
        stop=2.0,
        value=1.0,
        step=0.1,
        label="Working-set pressure",
    )
    _placement_options = v2_04_option_labels(v2_04_storage["placements"])
    _placement_default_label = next(
        _label for _label, _value in _placement_options.items() if _value == v2_04_storage["default_placement"]
    )
    v2_04_placement_policy = mo.ui.dropdown(
        options=_placement_options,
        value=_placement_default_label,
        label="Placement policy",
    )
    v2_04_b_checkpoint = mo.ui.radio(
        options={
            "Use the selected placement with measured guardrails": "use_selected",
            "Reject remote reads and stage closer to the consumer": "reject_remote",
            "Keep data local and summarize before movement": "local_summary",
            "Escalate because no placement is feasible": "escalate",
        },
        label="Part B checkpoint: what placement goes into the memo?",
    )

    v2_04_c_prediction = mo.ui.radio(
        options={
            "Synchronous durable barrier": "sync_barrier",
            "Async local staging": "async_local",
            "Incremental verified checkpoints": "incremental_verified",
            "Fast unverified snapshots": "fast_unverified",
        },
        label="Part C prediction: which consistency/checkpoint policy survives?",
    )
    _checkpoint_options = v2_04_option_labels(v2_04_checkpoint_policy_specs)
    _checkpoint_default_label = next(
        _label for _label, _value in _checkpoint_options.items() if _value == v2_04_storage["checkpoint_default"]
    )
    v2_04_checkpoint_policy = mo.ui.dropdown(
        options=_checkpoint_options,
        value=_checkpoint_default_label,
        label="Checkpoint or evidence policy",
    )
    v2_04_checkpoint_interval = mo.ui.slider(
        start=v2_04_storage["interval_min"],
        stop=v2_04_storage["interval_max"],
        value=v2_04_storage["interval_default"],
        step=v2_04_storage["interval_step"],
        label="Checkpoint interval in minutes",
    )
    v2_04_c_checkpoint = mo.ui.radio(
        options={
            "Use verified incremental or sharded checkpoints": "verified_incremental",
            "Use async local staging plus restore drills": "async_with_restore",
            "Keep synchronous barrier despite pause cost": "sync_with_cost",
            "Reject selected policy because restore evidence is weak": "reject_weak_evidence",
        },
        label="Part C checkpoint: what recovery evidence is required?",
    )

    v2_04_d_prediction = mo.ui.radio(
        options={
            "Full raw hot retention": "raw_hot",
            "Tiered hot/warm/cold policy": "tiered_policy",
            "Summaries plus rare raw snippets": "summary_first",
            "Cold archive after ingest": "cold_archive",
        },
        label="Part D prediction: which lifecycle policy satisfies all guardrails?",
    )
    _lifecycle_options = v2_04_option_labels(v2_04_lifecycle_policy_specs)
    _lifecycle_default_label = next(
        _label for _label, _value in _lifecycle_options.items() if _value == "raw_hot"
    )
    v2_04_lifecycle_policy = mo.ui.dropdown(
        options=_lifecycle_options,
        value=_lifecycle_default_label,
        label="Lifecycle policy",
    )
    v2_04_freshness_target = mo.ui.slider(
        start=v2_04_storage["freshness_min"],
        stop=v2_04_storage["freshness_max"],
        value=v2_04_storage["freshness_default"],
        step=v2_04_storage["freshness_step"],
        label="Freshness target in minutes",
    )
    v2_04_d_checkpoint = mo.ui.radio(
        options={
            "Use selected lifecycle policy with guardrail evidence": "use_selected_lifecycle",
            "Spend more on hot retention to protect freshness": "spend_for_freshness",
            "Summarize earlier to protect cost and privacy": "summarize_earlier",
            "Reject policy until reliability evidence improves": "reject_reliability",
        },
        label="Part D checkpoint: what lifecycle policy goes into the memo?",
    )

    v2_04_final_decision = mo.ui.radio(
        options={
            "Adopt selected placement and lifecycle policy": "adopt_selected",
            "Adopt only after relieving the binding storage amount": "relieve_binding_first",
            "Reject architecture and redesign the storage path": "redesign_storage",
        },
        label="Synthesis decision: what is the storage architecture decision?",
    )
    v2_04_architecture_memo = mo.ui.text_area(
        label="Storage architecture memo",
        placeholder=(
            "Name the selected placement, lifecycle policy, binding storage amount, "
            "rejected alternative, checkpoint evidence, and distributed-training implication."
        ),
        full_width=True,
    )
    return (
        v2_04_a_checkpoint,
        v2_04_a_prediction,
        v2_04_architecture_memo,
        v2_04_b_checkpoint,
        v2_04_b_prediction,
        v2_04_c_checkpoint,
        v2_04_c_prediction,
        v2_04_checkpoint_interval,
        v2_04_checkpoint_policy,
        v2_04_d_checkpoint,
        v2_04_d_prediction,
        v2_04_demand_multiplier,
        v2_04_final_decision,
        v2_04_freshness_target,
        v2_04_lifecycle_policy,
        v2_04_placement_policy,
        v2_04_working_set_pressure,
    )


@app.cell
def _():
    def v2_04_evaluate_throughput(storage_profile, demand_multiplier):
        base_demand = (
            storage_profile["consumer_count"]
            * storage_profile["record_mb"]
            / max(storage_profile["iteration_s"], 1e-9)
            * storage_profile["target_utilization"]
            * demand_multiplier
        )
        rows = []
        for stage in storage_profile["stages"]:
            demand = base_demand * stage["factor"]
            capacity = stage["capacity_mb_s"]
            utilization = demand / max(capacity, 1e-9)
            backlog_gb_h = max(0.0, demand - capacity) * 3600.0 / 1024.0
            rows.append(
                {
                    "stage": stage["name"],
                    "group": stage["group"],
                    "demand_mb_s": demand,
                    "capacity_mb_s": capacity,
                    "utilization_pct": utilization * 100.0,
                    "backlog_gb_h": backlog_gb_h,
                    "feasible": utilization <= 1.0,
                }
            )
        bottleneck = max(rows, key=lambda row: row["utilization_pct"])
        consumer_utilization = min(100.0, 100.0 / max(1.0, bottleneck["utilization_pct"] / 100.0))
        actual = "no_backlog" if all(row["feasible"] for row in rows) else bottleneck["group"]
        return {
            "base_demand_mb_s": base_demand,
            "rows": rows,
            "bottleneck": bottleneck,
            "actual_prediction": actual,
            "feasible": all(row["feasible"] for row in rows),
            "starvation_pct": max(0.0, 100.0 - consumer_utilization),
            "binding_ratio": bottleneck["utilization_pct"] / 100.0,
        }

    def v2_04_evaluate_placements(storage_profile, throughput_result, pressure):
        daily_gb = throughput_result["base_demand_mb_s"] * 86400.0 / 1024.0 * pressure
        request_mb = storage_profile["placement_request_mb"] * pressure
        rows = []
        for option_id, option in storage_profile["placements"].items():
            miss_fraction = max(0.0, min(1.0, 1.0 - option["cache_hit"]))
            moved_gb_day = daily_gb * miss_fraction
            transfer_ms = request_mb * miss_fraction * 8.0 / max(option["bandwidth_gbps"], 1e-9)
            latency_ms = option["base_latency_ms"] + option["tail_ms"] + transfer_ms
            cost_day = moved_gb_day * option["egress_per_gb"]
            ratios = {
                "latency": latency_ms / storage_profile["latency_budget_ms"],
                "movement": moved_gb_day / storage_profile["movement_budget_gb_day"],
                "cost": cost_day / storage_profile["egress_budget_day"],
                "freshness": option["freshness_min"] / storage_profile["freshness_budget_min"],
            }
            feasible = all(value <= 1.0 for value in ratios.values())
            rows.append(
                {
                    "option_id": option_id,
                    "label": option["label"],
                    "class": option["class"],
                    "latency_ms": latency_ms,
                    "moved_gb_day": moved_gb_day,
                    "cost_day": cost_day,
                    "freshness_min": option["freshness_min"],
                    "risk": option["risk"],
                    "feasible": feasible,
                    "ratios": ratios,
                    "score": sum(ratios.values()),
                    "binding_ratio": max(ratios.values()),
                }
            )
        best = min([row for row in rows if row["feasible"]] or rows, key=lambda row: row["score"])
        actual = best["class"] if best["feasible"] else "none"
        return {"rows": rows, "best": best, "actual_prediction": actual}

    def v2_04_evaluate_checkpoints(storage_profile, policy_specs, interval_min):
        writes_per_hour = 60.0 / max(interval_min, 1e-9)
        rows = []
        for policy_id, policy in policy_specs.items():
            state_gb = storage_profile["state_gb"] * policy["fraction"]
            per_node_gb = state_gb / max(storage_profile["node_count"], 1)
            if policy_id == "sync_barrier":
                pause_s = storage_profile["state_gb"] / max(storage_profile["durable_write_gb_s"], 1e-9)
            else:
                pause_s = per_node_gb / max(storage_profile["local_write_gb_s"], 1e-9)
            durable_delay_min = state_gb / max(storage_profile["durable_write_gb_s"], 1e-9) / 60.0
            write_storm_gb_h = state_gb * policy["write_amp"] * writes_per_hour
            lost_work_min = interval_min / 2.0 + durable_delay_min * max(policy["restore_multiplier"] - 1.0, 0.0)
            restore_min = durable_delay_min * policy["restore_multiplier"] + 5.0
            ratios = {
                "pause": pause_s / storage_profile["pause_budget_s"],
                "write_storm": write_storm_gb_h / storage_profile["write_storm_budget_gb_h"],
                "restore_evidence": storage_profile["evidence_floor_pct"] / max(policy["evidence_pct"], 1e-9),
                "restore_time": restore_min / storage_profile["restore_budget_min"],
            }
            feasible = all(value <= 1.0 for value in ratios.values())
            rows.append(
                {
                    "policy_id": policy_id,
                    "label": policy["label"],
                    "class": policy["class"],
                    "pause_s": pause_s,
                    "durable_delay_min": durable_delay_min,
                    "write_storm_gb_h": write_storm_gb_h,
                    "lost_work_min": lost_work_min,
                    "restore_evidence_pct": policy["evidence_pct"],
                    "restore_min": restore_min,
                    "risk": policy["risk"],
                    "ratios": ratios,
                    "binding_ratio": max(ratios.values()),
                    "score": sum(ratios.values()),
                    "feasible": feasible,
                }
            )
        best = min([row for row in rows if row["feasible"]] or rows, key=lambda row: row["score"])
        return {"rows": rows, "best": best, "actual_prediction": best["policy_id"] if best["feasible"] else "fast_unverified"}

    def v2_04_evaluate_lifecycle(storage_profile, lifecycle_specs, freshness_target):
        daily_gb = storage_profile["lifecycle_daily_gb"]
        prices = storage_profile["tier_prices"]
        rows = []
        for policy_id, policy in lifecycle_specs.items():
            hot_gb = daily_gb * policy["reduction"] * policy["hot_days"]
            warm_gb = daily_gb * policy["reduction"] * policy["warm_days"]
            cold_gb = daily_gb * policy["reduction"] * policy["cold_days"]
            retained_gb = hot_gb + warm_gb + cold_gb
            monthly_cost = (
                hot_gb * prices["hot"]
                + warm_gb * prices["warm"]
                + cold_gb * prices["cold"]
            )
            retention_days = policy["hot_days"] + policy["warm_days"] + policy["cold_days"]
            ratios = {
                "freshness": policy["lag_min"] / freshness_target,
                "retention": storage_profile["min_retention_days"] / max(retention_days, 1e-9),
                "cost": monthly_cost / storage_profile["monthly_budget"],
                "reliability": storage_profile["durability_floor_pct"] / max(policy["durability_pct"], 1e-9),
            }
            feasible = all(value <= 1.0 for value in ratios.values())
            rows.append(
                {
                    "policy_id": policy_id,
                    "label": policy["label"],
                    "class": policy["class"],
                    "hot_gb": hot_gb,
                    "warm_gb": warm_gb,
                    "cold_gb": cold_gb,
                    "retained_gb": retained_gb,
                    "monthly_cost": monthly_cost,
                    "freshness_lag_min": policy["lag_min"],
                    "retention_days": retention_days,
                    "durability_pct": policy["durability_pct"],
                    "risk": policy["risk"],
                    "ratios": ratios,
                    "binding_ratio": max(ratios.values()),
                    "score": sum(ratios.values()),
                    "feasible": feasible,
                }
            )
        best = min([row for row in rows if row["feasible"]] or rows, key=lambda row: row["score"])
        return {"rows": rows, "best": best, "actual_prediction": best["policy_id"] if best["feasible"] else "cold_archive"}

    def v2_04_selected_row(rows, key, selected_id):
        return next((row for row in rows if row[key] == selected_id), rows[0])

    def v2_04_binding_summary(items):
        return max(items, key=lambda item: item["ratio"])

    return (
        v2_04_binding_summary,
        v2_04_evaluate_checkpoints,
        v2_04_evaluate_lifecycle,
        v2_04_evaluate_placements,
        v2_04_evaluate_throughput,
        v2_04_selected_row,
    )


@app.cell
def _(
    v2_04_binding_summary,
    v2_04_checkpoint_interval,
    v2_04_checkpoint_policy,
    v2_04_checkpoint_policy_specs,
    v2_04_demand_multiplier,
    v2_04_evaluate_checkpoints,
    v2_04_evaluate_lifecycle,
    v2_04_evaluate_placements,
    v2_04_evaluate_throughput,
    v2_04_freshness_target,
    v2_04_lifecycle_policy,
    v2_04_lifecycle_policy_specs,
    v2_04_placement_policy,
    v2_04_selected_row,
    v2_04_storage,
    v2_04_working_set_pressure,
):
    v2_04_a_result = v2_04_evaluate_throughput(v2_04_storage, v2_04_demand_multiplier.value)
    v2_04_b_result = v2_04_evaluate_placements(
        v2_04_storage,
        v2_04_a_result,
        v2_04_working_set_pressure.value,
    )
    v2_04_b_selected = v2_04_selected_row(
        v2_04_b_result["rows"],
        "option_id",
        v2_04_placement_policy.value,
    )
    v2_04_c_result = v2_04_evaluate_checkpoints(
        v2_04_storage,
        v2_04_checkpoint_policy_specs,
        v2_04_checkpoint_interval.value,
    )
    v2_04_c_selected = v2_04_selected_row(
        v2_04_c_result["rows"],
        "policy_id",
        v2_04_checkpoint_policy.value,
    )
    v2_04_d_result = v2_04_evaluate_lifecycle(
        v2_04_storage,
        v2_04_lifecycle_policy_specs,
        v2_04_freshness_target.value,
    )
    v2_04_d_selected = v2_04_selected_row(
        v2_04_d_result["rows"],
        "policy_id",
        v2_04_lifecycle_policy.value,
    )
    v2_04_binding = v2_04_binding_summary(
        (
            {
                "module": "Part A",
                "amount": f"throughput at {v2_04_a_result['bottleneck']['stage']}",
                "ratio": v2_04_a_result["binding_ratio"],
                "evidence": f"{v2_04_a_result['bottleneck']['utilization_pct']:.1f}% utilization",
            },
            {
                "module": "Part B",
                "amount": f"locality for {v2_04_b_selected['label']}",
                "ratio": v2_04_b_selected["binding_ratio"],
                "evidence": f"{v2_04_b_selected['latency_ms']:.1f} ms and ${v2_04_b_selected['cost_day']:.2f}/day",
            },
            {
                "module": "Part C",
                "amount": f"checkpoint policy {v2_04_c_selected['label']}",
                "ratio": v2_04_c_selected["binding_ratio"],
                "evidence": f"{v2_04_c_selected['pause_s']:.1f}s pause, {v2_04_c_selected['write_storm_gb_h']:.1f} GB/h writes",
            },
            {
                "module": "Part D",
                "amount": f"lifecycle policy {v2_04_d_selected['label']}",
                "ratio": v2_04_d_selected["binding_ratio"],
                "evidence": f"${v2_04_d_selected['monthly_cost']:.0f}/month, {v2_04_d_selected['freshness_lag_min']:.0f} min lag",
            },
        )
    )
    v2_04_rejected_alternative = (
        v2_04_b_result["rows"][0]["label"]
        if v2_04_b_result["rows"][0]["label"] != v2_04_b_selected["label"]
        else v2_04_d_result["rows"][0]["label"]
    )
    return (
        v2_04_a_result,
        v2_04_b_result,
        v2_04_b_selected,
        v2_04_binding,
        v2_04_c_result,
        v2_04_c_selected,
        v2_04_d_result,
        v2_04_d_selected,
        v2_04_rejected_alternative,
    )


@app.cell
def _(
    COLORS,
    apply_plotly_theme,
    go,
    v2_04_a_result,
    v2_04_b_result,
    v2_04_c_result,
    v2_04_d_result,
):
    v2_04_a_fig = go.Figure()
    _a_rows = v2_04_a_result["rows"]
    v2_04_a_fig.add_trace(
        go.Bar(
            name="Demand",
            x=[row["stage"] for row in _a_rows],
            y=[row["demand_mb_s"] for row in _a_rows],
            marker_color=COLORS["OrangeLine"],
        )
    )
    v2_04_a_fig.add_trace(
        go.Bar(
            name="Capacity",
            x=[row["stage"] for row in _a_rows],
            y=[row["capacity_mb_s"] for row in _a_rows],
            marker_color=COLORS["BlueLine"],
        )
    )
    v2_04_a_fig.update_layout(
        title="Stage demand versus capacity",
        yaxis_title="MB/s",
        barmode="group",
    )
    apply_plotly_theme(v2_04_a_fig)

    v2_04_b_fig = go.Figure()
    _b_rows = v2_04_b_result["rows"]
    v2_04_b_fig.add_trace(
        go.Scatter(
            x=[row["moved_gb_day"] for row in _b_rows],
            y=[row["latency_ms"] for row in _b_rows],
            mode="markers+text",
            text=[row["label"] for row in _b_rows],
            textposition="top center",
            marker=dict(
                size=[18 if row["feasible"] else 13 for row in _b_rows],
                color=[COLORS["GreenLine"] if row["feasible"] else COLORS["RedLine"] for row in _b_rows],
            ),
            name="Placement",
        )
    )
    v2_04_b_fig.update_layout(
        title="Placement frontier",
        xaxis_title="GB moved per day",
        yaxis_title="request latency (ms)",
    )
    apply_plotly_theme(v2_04_b_fig)

    v2_04_c_fig = go.Figure()
    _c_rows = v2_04_c_result["rows"]
    v2_04_c_fig.add_trace(
        go.Bar(
            name="Pause seconds",
            x=[row["label"] for row in _c_rows],
            y=[row["pause_s"] for row in _c_rows],
            marker_color=[COLORS["GreenLine"] if row["feasible"] else COLORS["RedLine"] for row in _c_rows],
        )
    )
    v2_04_c_fig.update_layout(
        title="Checkpoint pause by policy",
        yaxis_title="seconds",
    )
    apply_plotly_theme(v2_04_c_fig)

    v2_04_d_fig = go.Figure()
    _d_rows = v2_04_d_result["rows"]
    for _tier_key, _tier_label, _tier_color in (
        ("hot_gb", "Hot footprint", COLORS["RedLine"]),
        ("warm_gb", "Warm footprint", COLORS["OrangeLine"]),
        ("cold_gb", "Cold footprint", COLORS["BlueLine"]),
    ):
        v2_04_d_fig.add_trace(
            go.Bar(
                name=_tier_label,
                x=[row["label"] for row in _d_rows],
                y=[row[_tier_key] for row in _d_rows],
                marker_color=_tier_color,
            )
        )
    v2_04_d_fig.update_layout(
        title="Retained footprint by lifecycle tier",
        yaxis_title="GB retained",
        barmode="stack",
    )
    apply_plotly_theme(v2_04_d_fig)
    return v2_04_a_fig, v2_04_b_fig, v2_04_c_fig, v2_04_d_fig


@app.cell(hide_code=True)
def _(
    COLORS,
    MathPeek,
    big_takeaways,
    gated_hypothesis_card,
    instrumentation_console,
    mo,
    source_trace,
    v2_04_a_checkpoint,
    v2_04_a_fig,
    v2_04_a_prediction,
    v2_04_a_result,
    v2_04_architecture_memo,
    v2_04_b_checkpoint,
    v2_04_b_fig,
    v2_04_b_prediction,
    v2_04_b_result,
    v2_04_b_selected,
    v2_04_binding,
    v2_04_c_checkpoint,
    v2_04_c_fig,
    v2_04_c_prediction,
    v2_04_c_result,
    v2_04_c_selected,
    v2_04_callout_html,
    v2_04_checkpoint_interval,
    v2_04_checkpoint_policy,
    v2_04_d_checkpoint,
    v2_04_d_fig,
    v2_04_d_prediction,
    v2_04_d_result,
    v2_04_d_selected,
    v2_04_demand_multiplier,
    v2_04_fields_html,
    v2_04_final_decision,
    v2_04_format,
    v2_04_freshness_target,
    v2_04_lifecycle_policy,
    v2_04_placement_policy,
    v2_04_prediction_html,
    v2_04_profile,
    v2_04_ratio_label,
    v2_04_rejected_alternative,
    v2_04_status_text,
    v2_04_storage,
    v2_04_table_html,
    v2_04_variant,
    v2_04_working_set_pressure,
):
    _a_labels = {
        "storage_throughput": "storage/read throughput",
        "preprocess": "preprocessing",
        "network": "network or upload",
        "no_backlog": "no backlog",
    }
    _b_labels = {
        "remote": "remote placement",
        "tiered": "tiered regional/cache placement",
        "local": "local placement",
        "none": "no feasible placement",
    }
    _c_labels = {
        "sync_barrier": "synchronous durable barrier",
        "async_local": "async local staging",
        "incremental_verified": "incremental verified checkpoints",
        "fast_unverified": "fast unverified snapshots",
    }
    _d_labels = {
        "raw_hot": "full raw hot retention",
        "tiered_policy": "tiered hot/warm/cold policy",
        "summary_first": "summaries plus rare raw snippets",
        "cold_archive": "cold archive after ingest",
    }

    _a_rows = [
        (
            row["stage"],
            v2_04_format(row["demand_mb_s"], 2),
            v2_04_format(row["capacity_mb_s"], 2),
            v2_04_format(row["utilization_pct"], 1, "%"),
            v2_04_format(row["backlog_gb_h"], 2),
            v2_04_status_text(row["feasible"]),
        )
        for row in v2_04_a_result["rows"]
    ]
    _a_failure = not v2_04_a_result["feasible"]
    _a_consequence = (
        f"{v2_04_a_result['bottleneck']['stage']} is over capacity; backlog grows "
        f"at {v2_04_a_result['bottleneck']['backlog_gb_h']:.2f} GB/hour and "
        f"consumer starvation is {v2_04_a_result['starvation_pct']:.1f}%."
        if _a_failure
        else "Every stage is inside its capacity envelope for the current demand pressure."
    )

    _b_rows = [
        (
            row["label"],
            row["class"],
            v2_04_format(row["latency_ms"], 1),
            v2_04_format(row["moved_gb_day"], 1),
            f"${row['cost_day']:.2f}",
            v2_04_format(row["freshness_min"], 0),
            v2_04_status_text(row["feasible"]),
        )
        for row in v2_04_b_result["rows"]
    ]
    _b_failure = not v2_04_b_selected["feasible"]
    _b_consequence = (
        f"{v2_04_b_selected['label']} violates at least one placement guardrail; "
        f"binding ratio is {v2_04_b_selected['binding_ratio']:.2f}x."
        if _b_failure
        else f"{v2_04_b_selected['label']} satisfies the current locality, latency, cost, and freshness guardrails."
    )

    _c_rows = [
        (
            row["label"],
            v2_04_format(row["pause_s"], 2),
            v2_04_format(row["durable_delay_min"], 1),
            v2_04_format(row["write_storm_gb_h"], 1),
            v2_04_format(row["restore_evidence_pct"], 0, "%"),
            v2_04_format(row["lost_work_min"], 1),
            v2_04_status_text(row["feasible"]),
        )
        for row in v2_04_c_result["rows"]
    ]
    _c_failure = not v2_04_c_selected["feasible"]
    _c_consequence = (
        f"{v2_04_c_selected['label']} is not defensible at this interval; pause, write storm, restore time, or evidence is outside the guardrail."
        if _c_failure
        else f"{v2_04_c_selected['label']} keeps pause and write storm inside the guardrails while preserving restore evidence."
    )

    _d_rows = [
        (
            row["label"],
            v2_04_format(row["retained_gb"], 1),
            f"${row['monthly_cost']:.0f}",
            v2_04_format(row["freshness_lag_min"], 0),
            v2_04_format(row["retention_days"], 1),
            v2_04_format(row["durability_pct"], 2, "%"),
            v2_04_status_text(row["feasible"]),
        )
        for row in v2_04_d_result["rows"]
    ]
    _d_failure = not v2_04_d_selected["feasible"]
    _d_consequence = (
        f"{v2_04_d_selected['label']} fails at least one lifecycle guardrail; binding ratio is {v2_04_d_selected['binding_ratio']:.2f}x."
        if _d_failure
        else f"{v2_04_d_selected['label']} satisfies freshness, retention, cost, and reliability guardrails."
    )

    def build_part_a():
        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <div class="mlsysbook-section-label">Part A - Concept Module</div>
              <h2 style="margin:8px 0 6px 0; color:#0F172A;">Throughput Must Match Consumer Demand</h2>
              <p style="color:#475569; font-size:0.92rem; line-height:1.55;">
                You are operating as the <strong>{v2_04_storage["stakeholder"]}</strong>. Your consumer is
                <strong>{v2_04_storage["consumer_label"]}</strong>, and the storage path must feed both:
                {v2_04_storage["training_link"]}; {v2_04_storage["serving_link"]}.
              </p>
              <div class="mlsysbook-compact-fields">
                {v2_04_fields_html({
                    "Chapter claim": "Storage capacity is not storage throughput; starving accelerators destroys system MFU.",
                    "Your decision": "Identify the primary fuel line bottleneck and size prefetch/bandwidth accordingly.",
                    "Track consequence": f"Consumer starvation is {v2_04_a_result['starvation_pct']:.1f}% under current demand.",
                })}
              </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_04_a_prediction,
                title="1. Formulate Fuel Line Bottleneck Hypothesis",
                subtitle="Predict which stage in the storage pipeline blocks consumer demand first:",
                gate_label="Hypothesis Gate A",
            ),
            instrumentation_console(
                v2_04_demand_multiplier,
                title="Data Demand & Ingestion Controls",
                subtitle="Sweep consumer demand pressure to observe pipeline stage saturation:",
            ),
            mo.md("### Empirical Evidence"),
            mo.Html(
                v2_04_prediction_html(
                    "Prediction Check",
                    v2_04_a_prediction.value,
                    v2_04_a_result["actual_prediction"],
                    _a_labels,
                )
            ),
            mo.as_html(v2_04_a_fig),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Throughput Evidence</h2>
              <div class="mlsysbook-grid">
                {v2_04_fields_html({
                    "Base demand": f"{v2_04_a_result['base_demand_mb_s']:.2f} MB/s",
                    "Bottleneck stage": v2_04_a_result["bottleneck"]["stage"],
                    "Binding ratio": v2_04_ratio_label(v2_04_a_result["binding_ratio"]),
                    "Backlog": f"{v2_04_a_result['bottleneck']['backlog_gb_h']:.2f} GB/hour",
                    "Starvation": f"{v2_04_a_result['starvation_pct']:.1f}%",
                    "Consumer count": v2_04_storage["consumer_count"],
                })}
              </div>
              {v2_04_table_html(("Stage", "Demand MB/s", "Capacity MB/s", "Utilization", "Backlog GB/h", "Status"), _a_rows, numeric=(1, 2, 3, 4))}
            </div>
            """),
            mo.Html(v2_04_callout_html("Consequence", _a_consequence, "fail" if _a_failure else "ok")),
            MathPeek(
                formula="\\text{BW}_{\\text{req}} = N \\cdot \\eta \\cdot \\frac{D_{\\text{step}}}{T_{\\text{iter}}}; \\quad \\text{Backlog} = \\max(0, D_{\\text{stage}} - C_{\\text{stage}}) \\cdot \\Delta t",
                variables={
                    "BaseDemand": f"{v2_04_a_result['base_demand_mb_s']:.2f} MB/s",
                    "Bottleneck": v2_04_a_result["bottleneck"]["stage"],
                    "BacklogRate": f"{v2_04_a_result['bottleneck']['backlog_gb_h']:.2f} GB/h",
                    "ConsumerStarvation": f"{v2_04_a_result['starvation_pct']:.1f}%",
                },
            ),
            source_trace(
                {
                    "chapter_anchor": "Data Pipeline Equation and I/O Wall",
                    "formula": "BW_required = N x eta x D_step / T_iteration",
                    "hardware_ref": v2_04_storage["hardware_ref"],
                    "scenario_assumption": v2_04_storage["chapter_amount_note"],
                    "track_id": v2_04_profile.track_id,
                },
                summary="Part A source model",
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid {COLORS['BlueLine']}; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT A: PIPELINE BACKLOG RESOLUTION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Throughput Remediation Selection</h4>
                {v2_04_a_checkpoint}
            </div>
            """),
        ])

    def build_part_b():
        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <div class="mlsysbook-section-label">Part B - Concept Module</div>
              <h2 style="margin:8px 0 6px 0; color:#0F172A;">Locality And Placement Change Cost And Latency</h2>
              <p style="color:#475569; font-size:0.92rem; line-height:1.55;">
                Move the same data through remote, tiered, or local placement. The concept is
                unchanged; the <strong>{v2_04_profile.label}</strong> track changes latency, movement cost,
                freshness, and residual risk.
              </p>
              <div class="mlsysbook-compact-fields">
                {v2_04_fields_html({
                    "Chapter claim": "Placement changes the amount system: cache hits lower bandwidth but increase staleness risk.",
                    "Your decision": "Select a placement tier that satisfies latency, movement, cost, and freshness budgets.",
                    "Selected status": "PASS" if v2_04_b_selected["feasible"] else "FAIL",
                })}
              </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_04_b_prediction,
                title="2. Formulate Data Placement Hypothesis",
                subtitle="Predict which data placement class satisfies all operational guardrails:",
                gate_label="Hypothesis Gate B",
            ),
            instrumentation_console(
                mo.hstack([v2_04_placement_policy, v2_04_working_set_pressure], gap="1rem"),
                title="Locality & Working-Set Controls",
                subtitle="Select placement architecture and scale working-set pressure:",
            ),
            mo.md("### Empirical Evidence"),
            mo.Html(
                v2_04_prediction_html(
                    "Prediction Check",
                    v2_04_b_prediction.value,
                    v2_04_b_result["actual_prediction"],
                    _b_labels,
                )
            ),
            mo.as_html(v2_04_b_fig),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Placement Evidence</h2>
              <div class="mlsysbook-grid">
                {v2_04_fields_html({
                    "Selected placement": v2_04_b_selected["label"],
                    "Latency": f"{v2_04_b_selected['latency_ms']:.1f} ms / {v2_04_storage['latency_budget_ms']:.0f} ms",
                    "Data moved": f"{v2_04_b_selected['moved_gb_day']:.1f} GB/day",
                    "Movement cost": f"${v2_04_b_selected['cost_day']:.2f}/day",
                    "Freshness lag": f"{v2_04_b_selected['freshness_min']:.0f} min",
                    "Residual risk": v2_04_b_selected["risk"],
                })}
              </div>
              {v2_04_table_html(("Placement", "Class", "Latency ms", "GB/day", "Cost/day", "Freshness min", "Status"), _b_rows, numeric=(2, 3, 5))}
            </div>
            """),
            mo.Html(v2_04_callout_html("Consequence", _b_consequence, "fail" if _b_failure else "ok")),
            MathPeek(
                formula="T_{\\text{latency}} = T_{\\text{base}} + T_{\\text{tail}} + \\frac{\\text{Req}_{\\text{MB}} \\cdot 8}{\\text{BW}_{\\text{Gbps}}}; \\quad \\text{Cost} = \\text{Moved}_{\\text{GB}} \\cdot P_{\\text{egress}}",
                variables={
                    "SelectedPlacement": v2_04_b_selected["label"],
                    "Latency": f"{v2_04_b_selected['latency_ms']:.1f} ms",
                    "DataMoved": f"{v2_04_b_selected['moved_gb_day']:.1f} GB/day",
                    "EgressCost": f"${v2_04_b_selected['cost_day']:.2f}/day",
                    "FreshnessLag": f"{v2_04_b_selected['freshness_min']:.0f} min",
                },
            ),
            source_trace(
                {
                    "chapter_anchor": "ML Storage Hierarchy; Data locality and placement",
                    "hardware_ref": v2_04_storage["hardware_ref"],
                    "system_ref": v2_04_storage["system_ref"],
                    "selected_policy": v2_04_b_selected["label"],
                    "track_source_policy": v2_04_storage["source_policy"],
                },
                summary="Part B source model",
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid {COLORS['BlueLine']}; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT B: DATA PLACEMENT DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Placement Tier Confirmation</h4>
                {v2_04_b_checkpoint}
            </div>
            """),
        ])

    def build_part_c():
        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <div class="mlsysbook-section-label">Part C - Concept Module</div>
              <h2 style="margin:8px 0 6px 0; color:#0F172A;">Checkpointing Trades Evidence Against Write Storms</h2>
              <p style="color:#475569; font-size:0.92rem; line-height:1.55;">
                Pick a consistency and checkpoint policy. The decision is valid only if it
                controls pause time and write storms while proving that the state can be
                restored.
              </p>
              <div class="mlsysbook-compact-fields">
                {v2_04_fields_html({
                    "Chapter claim": "A checkpoint is only valid if restorable; unverified snapshots create silent recovery failure.",
                    "Your decision": "Balance checkpoint pause, write amplification, and recovery verification.",
                    "Selected status": "PASS" if v2_04_c_selected["feasible"] else "FAIL",
                })}
              </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_04_c_prediction,
                title="3. Formulate Checkpoint Protocol Hypothesis",
                subtitle="Predict which checkpointing strategy survives operational write storms and recovery SLAs:",
                gate_label="Hypothesis Gate C",
            ),
            instrumentation_console(
                mo.hstack([v2_04_checkpoint_policy, v2_04_checkpoint_interval], gap="1rem"),
                title="Checkpointing & Recovery Controls",
                subtitle="Select checkpoint protocol and configure snapshot interval:",
            ),
            mo.md("### Empirical Evidence"),
            mo.Html(
                v2_04_prediction_html(
                    "Prediction Check",
                    v2_04_c_prediction.value,
                    v2_04_c_result["actual_prediction"],
                    _c_labels,
                )
            ),
            mo.as_html(v2_04_c_fig),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Checkpoint Evidence</h2>
              <div class="mlsysbook-grid">
                {v2_04_fields_html({
                    "Selected policy": v2_04_c_selected["label"],
                    "State amount": f"{v2_04_storage['state_gb']:.2f} GB",
                    "Pause": f"{v2_04_c_selected['pause_s']:.2f}s / {v2_04_storage['pause_budget_s']:.1f}s",
                    "Write storm": f"{v2_04_c_selected['write_storm_gb_h']:.1f} GB/hour",
                    "Restore evidence": f"{v2_04_c_selected['restore_evidence_pct']:.0f}%",
                    "Lost-work exposure": f"{v2_04_c_selected['lost_work_min']:.1f} min",
                })}
              </div>
              {v2_04_table_html(("Policy", "Pause s", "Durable delay min", "Write GB/h", "Evidence", "Lost work min", "Status"), _c_rows, numeric=(1, 2, 3, 4, 5))}
            </div>
            """),
            mo.Html(v2_04_callout_html("Consequence", _c_consequence, "fail" if _c_failure else "ok")),
            MathPeek(
                formula="T_{\\text{pause}} = \\frac{\\text{State}}{\\text{BW}_{\\text{local}}}; \\quad \\text{Storm} = \\text{State} \\cdot \\alpha_{\\text{amp}} \\cdot \\frac{60}{T_{\\text{interval}}}",
                variables={
                    "SelectedPolicy": v2_04_c_selected["label"],
                    "StateSize": f"{v2_04_storage['state_gb']:.2f} GB",
                    "PauseTime": f"{v2_04_c_selected['pause_s']:.2f}s",
                    "WriteStorm": f"{v2_04_c_selected['write_storm_gb_h']:.1f} GB/h",
                    "RestoreEvidence": f"{v2_04_c_selected['restore_evidence_pct']:.0f}%",
                },
            ),
            source_trace(
                {
                    "chapter_anchor": "Checkpoint Storage; Distributed checkpoint coordination",
                    "model_ref": "Models.Language.GPT3 for cloud-fleet; track-specific evidence snapshots elsewhere",
                    "checkpoint_state_gb": f"{v2_04_storage['state_gb']:.2f}",
                    "node_count": v2_04_storage["node_count"],
                    "selected_policy": v2_04_c_selected["label"],
                },
                summary="Part C source model",
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid {COLORS['BlueLine']}; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT C: RECOVERY GUARANTEE DECISION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Checkpoint Protocol Confirmation</h4>
                {v2_04_c_checkpoint}
            </div>
            """),
        ])

    def build_part_d():
        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <div class="mlsysbook-section-label">Part D - Concept Module</div>
              <h2 style="margin:8px 0 6px 0; color:#0F172A;">Lifecycle Policy Must Satisfy Simultaneous Guardrails</h2>
              <p style="color:#475569; font-size:0.92rem; line-height:1.55;">
                Decide what remains hot, what is tiered, what is summarized, and what is
                discarded. A cheap policy that fails freshness, retention, or reliability is
                not a valid storage design.
              </p>
              <div class="mlsysbook-compact-fields">
                {v2_04_fields_html({
                    "Chapter claim": "Storage lifecycle is a multi-constraint problem: price, retention, durability, and freshness must close together.",
                    "Your decision": "Balance hot/warm/cold footprint against operational budget and durability floor.",
                    "Selected status": "PASS" if v2_04_d_selected["feasible"] else "FAIL",
                })}
              </div>
            </div>
            """),
            gated_hypothesis_card(
                v2_04_d_prediction,
                title="4. Formulate Storage Lifecycle Hypothesis",
                subtitle="Predict which tiering and retention policy satisfies simultaneous SLA and cost guardrails:",
                gate_label="Hypothesis Gate D",
            ),
            instrumentation_console(
                mo.hstack([v2_04_lifecycle_policy, v2_04_freshness_target], gap="1rem"),
                title="Lifecycle & Tiering Controls",
                subtitle="Configure retention tiering policy and maximum tolerable freshness lag:",
            ),
            mo.md("### Empirical Evidence"),
            mo.Html(
                v2_04_prediction_html(
                    "Prediction Check",
                    v2_04_d_prediction.value,
                    v2_04_d_result["actual_prediction"],
                    _d_labels,
                )
            ),
            mo.as_html(v2_04_d_fig),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Lifecycle Evidence</h2>
              <div class="mlsysbook-grid">
                {v2_04_fields_html({
                    "Selected policy": v2_04_d_selected["label"],
                    "Retained footprint": f"{v2_04_d_selected['retained_gb']:.1f} GB",
                    "Monthly cost": f"${v2_04_d_selected['monthly_cost']:.0f} / ${v2_04_storage['monthly_budget']:.0f}",
                    "Freshness": f"{v2_04_d_selected['freshness_lag_min']:.0f} min / {v2_04_freshness_target.value:.0f} min",
                    "Retention": f"{v2_04_d_selected['retention_days']:.1f} days / {v2_04_storage['min_retention_days']:.1f} days",
                    "Reliability": f"{v2_04_d_selected['durability_pct']:.2f}% / {v2_04_storage['durability_floor_pct']:.2f}%",
                })}
              </div>
              {v2_04_table_html(("Policy", "Retained GB", "Monthly cost", "Freshness min", "Retention days", "Durability", "Status"), _d_rows, numeric=(1, 3, 4))}
            </div>
            """),
            mo.Html(v2_04_callout_html("Consequence", _d_consequence, "fail" if _d_failure else "ok")),
            MathPeek(
                formula="\\text{Cost}_{\\text{month}} = \\sum_{i \\in \\{h,w,c\\}} \\text{Footprint}_i \\cdot P_i; \\quad \\text{Retained}_{\\text{GB}} = D_{\\text{daily}} \\cdot \\text{Days} \\cdot \\beta_{\\text{red}}",
                variables={
                    "SelectedPolicy": v2_04_d_selected["label"],
                    "RetainedGB": f"{v2_04_d_selected['retained_gb']:.1f} GB",
                    "MonthlyCost": f"${v2_04_d_selected['monthly_cost']:.0f}",
                    "Durability": f"{v2_04_d_selected['durability_pct']:.2f}%",
                    "FreshnessLag": f"{v2_04_d_selected['freshness_lag_min']:.0f} min",
                },
            ),
            source_trace(
                {
                    "chapter_anchor": "Storage Economics; Tiering strategies; Fallacies and Pitfalls",
                    "daily_ingest_gb": f"{v2_04_storage['lifecycle_daily_gb']:.1f}",
                    "freshness_target_min": f"{v2_04_freshness_target.value:.0f}",
                    "selected_policy": v2_04_d_selected["label"],
                    "scenario_assumption": "Lifecycle prices and thresholds are notebook-local track guardrails.",
                },
                summary="Part D source model",
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid {COLORS['BlueLine']}; margin-top: 16px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">CHECKPOINT D: LIFECYCLE POLICY AUTHORIZATION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Retention Policy Sign-Off</h4>
                {v2_04_d_checkpoint}
            </div>
            """),
        ])

    def build_synthesis():
        passed = (
            v2_04_a_result["feasible"]
            and v2_04_b_selected["feasible"]
            and v2_04_c_selected["feasible"]
            and v2_04_d_selected["feasible"]
        )
        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <div class="mlsysbook-section-label">Synthesis - Storage Architecture Memo</div>
              <h2 style="margin:8px 0 6px 0; color:#0F172A;">Storage Fuel Line Sign-Off: {v2_04_profile.label}</h2>
              <p style="color:#475569; font-size:0.92rem; line-height:1.55;">
                Select placement and lifecycle policy for <strong>{v2_04_profile.label}</strong>, name the
                binding storage amount, reject one alternative, and carry the implication
                into Volume II Chapter 5 distributed training.
              </p>
              <div class="mlsysbook-compact-fields">
                {v2_04_fields_html({
                    "Selected placement": v2_04_b_selected["label"],
                    "Selected lifecycle": v2_04_d_selected["label"],
                    "Checkpoint evidence": v2_04_c_selected["label"],
                    "Binding storage amount": f"{v2_04_binding['module']} - {v2_04_binding['amount']} ({v2_04_binding['ratio']:.2f}x)",
                    "Rejected alternative": v2_04_rejected_alternative,
                    "Carry forward": v2_04_storage["next_lab"],
                })}
              </div>
            </div>
            """),
            mo.Html(f"""
            <div class="mlsysbook-panel" style="border-left: 4px solid #A51C30; margin: 18px 0; background: #FFFDFD;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #A51C30; text-transform: uppercase; margin-bottom: 6px;">LEAD ARCHITECT AUTHORIZATION</div>
                <h4 style="margin: 0 0 8px 0; color: #0F172A;">Data Storage Architecture Sign-Off: {v2_04_storage['stakeholder']}</h4>
                <div style="display: flex; gap: 12px; align-items: center; margin-top: 8px;">
                    <span style="display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 800; font-size: 0.8rem; background: {'#ECFDF5' if passed else '#FEF2F2'}; color: {'#065F46' if passed else '#991B1B'}; border: 1px solid {'#A7F3D0' if passed else '#FECACA'};">
                        {'APPROVED FOR FLEET DEPLOYMENT' if passed else 'PROVISIONAL &mdash; STORAGE BOUNDARY BREACH'}
                    </span>
                    <span style="font-size: 0.85rem; color: #475569;">
                        Binding: <code>{v2_04_binding['amount']}</code> &middot; Placement: <code>{v2_04_b_selected['label']}</code> &middot; Cost: <code>${v2_04_d_selected['monthly_cost']:.0f}/mo</code>
                    </span>
                </div>
            </div>
            """),
            mo.Html(f"""
            <div style="background:#0f172a; color:#e2e8f0; border-radius:10px; padding:20px 24px; margin:16px 0;">
                <div style="font-size:0.72rem; font-weight:800; color:#93c5fd;
                            text-transform:uppercase; letter-spacing:0.12em;">Storage Architecture Memo Summary</div>
                <h3 style="margin:8px 0 10px 0; color:white;">Track: {v2_04_profile.label}</h3>
                <p style="line-height:1.6; margin:0 0 12px 0;">
                    Operating as <strong>{v2_04_storage['stakeholder']}</strong>.
                    Placement <strong>{v2_04_b_selected['label']}</strong> provides <strong>{v2_04_b_selected['latency_ms']:.1f} ms</strong> latency
                    at <strong>${v2_04_b_selected['cost_day']:.2f}/day</strong> movement cost.
                    Recovery uses <strong>{v2_04_c_selected['label']}</strong> with <strong>{v2_04_c_selected['pause_s']:.2f}s</strong> pause.
                    Lifecycle policy <strong>{v2_04_d_selected['label']}</strong> retains <strong>{v2_04_d_selected['retained_gb']:.1f} GB</strong>
                    at <strong>${v2_04_d_selected['monthly_cost']:.0f}/month</strong>.
                </p>
                <div style="border-top:1px solid #334155; padding-top:12px; color:#bfdbfe;">
                    <strong>Downstream Carry-Forward:</strong> {v2_04_storage['next_lab']}
                </div>
            </div>
            """),
            instrumentation_console(
                mo.vstack([v2_04_final_decision, v2_04_architecture_memo]),
                title="Synthesis Governance & Downstream Hand-Off",
                subtitle="Designate final storage architecture decision and document carry-forward rationale:",
            ),
            big_takeaways([
                "Storage capacity is not storage throughput: fuel line starvation destroys accelerator MFU.",
                "Data placement hierarchy determines latency, movement energy, and egress cost across physical tiers.",
                "A checkpoint only counts if verified restorable: fast unverified snapshots risk unrecoverable state loss.",
                "Storage lifecycle policy must balance hot, warm, and cold tiers against freshness, durability, and cost guardrails.",
                "Downstream Chapter 5 Distributed Training must co-design batch staging and checkpoint write storms with storage infrastructure.",
            ]),
            source_trace(
                {
                    "chapter_anchor": "Data Storage summary",
                    "report_artifact": v2_04_variant.assumptions["report_artifact"],
                    "selected_track": v2_04_profile.track_id,
                    "binding_amount": v2_04_binding["amount"],
                    "distributed_training_implication": v2_04_storage["next_lab"],
                },
                collapsed=True,
                summary="Synthesis source model",
            ),
            mo.Html("""
            <div style="border: 1px solid #CBD5E1; border-radius: 8px; padding: 16px 20px; margin-top: 20px; background: #F8FAFC;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
                    What's Next &middot; Volume II Curriculum Continuum
                </div>
                <h4 style="margin: 0 0 6px 0; color: #0F172A; font-size: 1.05rem;">
                    Next Lab: Volume II, Chapter 5 &mdash; Distributed Training: 3D Parallelism &amp; Sharding
                </h4>
                <p style="margin: 0; font-size: 0.88rem; color: #334155; line-height: 1.5;">
                    Carry your storage pipeline and checkpoint write storm budgets into Chapter 5, where you will size data, tensor, and pipeline parallelism across distributed nodes.
                </p>
            </div>
            """),
        ])

    v2_04_tabs = mo.ui.tabs({
        "Part A -- Ingestion & Storage Bandwidth": build_part_a(),
        "Part B -- Locality & Working Set Placement": build_part_b(),
        "Part C -- Checkpointing & Recovery Protocols": build_part_c(),
        "Part D -- Lifecycle & Retention Economics": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    v2_04_tabs
    return


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    v2_04_a_checkpoint,
    v2_04_a_prediction,
    v2_04_a_result,
    v2_04_architecture_memo,
    v2_04_b_checkpoint,
    v2_04_b_prediction,
    v2_04_b_selected,
    v2_04_binding,
    v2_04_c_checkpoint,
    v2_04_c_prediction,
    v2_04_c_selected,
    v2_04_checkpoint_interval,
    v2_04_checkpoint_policy,
    v2_04_d_checkpoint,
    v2_04_d_prediction,
    v2_04_d_selected,
    v2_04_demand_multiplier,
    v2_04_final_decision,
    v2_04_freshness_target,
    v2_04_lifecycle_policy,
    v2_04_placement_policy,
    v2_04_profile,
    v2_04_rejected_alternative,
    v2_04_storage,
    v2_04_variant,
    v2_04_working_set_pressure,
):
    _complete = (
        v2_04_a_prediction.value is not None
        and v2_04_b_prediction.value is not None
        and v2_04_c_prediction.value is not None
        and v2_04_d_prediction.value is not None
        and v2_04_final_decision.value is not None
    )
    ledger.save(chapter=4, design={
        "chapter": "v2_04",
        "track_id": v2_04_profile.track_id,
        "scenario_id": v2_04_variant.scenario_id,
        "hardware_ref": v2_04_storage["hardware_ref"],
        "system_ref": v2_04_storage["system_ref"],
        "completed": _complete,
        "part_a_prediction": v2_04_a_prediction.value,
        "demand_multiplier": v2_04_demand_multiplier.value,
        "throughput_bottleneck": v2_04_a_result["bottleneck"]["stage"],
        "backlog_gb_per_hour": v2_04_a_result["bottleneck"]["backlog_gb_h"],
        "starvation_pct": v2_04_a_result["starvation_pct"],
        "part_a_checkpoint": v2_04_a_checkpoint.value,
        "part_b_prediction": v2_04_b_prediction.value,
        "placement_policy": v2_04_placement_policy.value,
        "working_set_pressure": v2_04_working_set_pressure.value,
        "placement_latency_ms": v2_04_b_selected["latency_ms"],
        "placement_cost_per_day": v2_04_b_selected["cost_day"],
        "part_b_checkpoint": v2_04_b_checkpoint.value,
        "part_c_prediction": v2_04_c_prediction.value,
        "checkpoint_policy": v2_04_checkpoint_policy.value,
        "checkpoint_interval_min": v2_04_checkpoint_interval.value,
        "checkpoint_pause_s": v2_04_c_selected["pause_s"],
        "write_storm_gb_per_hour": v2_04_c_selected["write_storm_gb_h"],
        "restore_evidence_pct": v2_04_c_selected["restore_evidence_pct"],
        "part_c_checkpoint": v2_04_c_checkpoint.value,
        "part_d_prediction": v2_04_d_prediction.value,
        "lifecycle_policy": v2_04_lifecycle_policy.value,
        "freshness_target_min": v2_04_freshness_target.value,
        "monthly_cost": v2_04_d_selected["monthly_cost"],
        "retention_days": v2_04_d_selected["retention_days"],
        "durability_pct": v2_04_d_selected["durability_pct"],
        "part_d_checkpoint": v2_04_d_checkpoint.value,
        "selected_placement_policy": v2_04_b_selected["label"],
        "selected_lifecycle_policy": v2_04_d_selected["label"],
        "binding_storage_amount": v2_04_binding["amount"],
        "binding_storage_ratio": v2_04_binding["ratio"],
        "rejected_alternative": v2_04_rejected_alternative,
        "distributed_training_implication": v2_04_storage["next_lab"],
        "final_decision": v2_04_final_decision.value,
        "architecture_memo": v2_04_architecture_memo.value,
    })

    _passed = (
        v2_04_a_result["feasible"]
        and v2_04_b_selected["feasible"]
        and v2_04_c_selected["feasible"]
        and v2_04_d_selected["feasible"]
    )
    mo.Html(
        f"""
    <div class="lab-hud">
      <span class="hud-label">LAB</span>
      <span class="hud-value">Vol2 &middot; Lab 04</span>
      <span class="hud-label">TRACK</span>
      <span class="hud-value">{v2_04_profile.label}</span>
      <span class="hud-label">STATUS</span>
      <span class="hud-value" style="color: {'#10B981' if _passed else '#F59E0B'};">{'PASS' if _passed else 'REVIEW'}</span>
      <span class="hud-label">BINDING</span>
      <span class="hud-value">{v2_04_binding['amount']}</span>
      <span class="hud-label">PLACEMENT</span>
      <span class="hud-value">{v2_04_b_selected['label']}</span>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    report_export_panel,
    v2_04_a_checkpoint,
    v2_04_a_prediction,
    v2_04_a_result,
    v2_04_architecture_memo,
    v2_04_b_checkpoint,
    v2_04_b_prediction,
    v2_04_b_selected,
    v2_04_binding,
    v2_04_c_checkpoint,
    v2_04_c_prediction,
    v2_04_c_selected,
    v2_04_checkpoint_interval,
    v2_04_checkpoint_policy,
    v2_04_d_checkpoint,
    v2_04_d_prediction,
    v2_04_d_selected,
    v2_04_demand_multiplier,
    v2_04_final_decision,
    v2_04_freshness_target,
    v2_04_lifecycle_policy,
    v2_04_metadata,
    v2_04_placement_policy,
    v2_04_profile,
    v2_04_rejected_alternative,
    v2_04_storage,
    v2_04_variant,
    v2_04_working_set_pressure,
):
    _incomplete = []
    _required = (
        ("Part A prediction", v2_04_a_prediction.value),
        ("Part A checkpoint", v2_04_a_checkpoint.value),
        ("Part B prediction", v2_04_b_prediction.value),
        ("Part B checkpoint", v2_04_b_checkpoint.value),
        ("Part C prediction", v2_04_c_prediction.value),
        ("Part C checkpoint", v2_04_c_checkpoint.value),
        ("Part D prediction", v2_04_d_prediction.value),
        ("Part D checkpoint", v2_04_d_checkpoint.value),
        ("Synthesis decision", v2_04_final_decision.value),
    )
    for _required_label, _required_value in _required:
        if _required_value is None:
            _incomplete.append(_required_label)
    if not str(v2_04_architecture_memo.value or "").strip():
        _incomplete.append("Storage architecture memo")

    v2_04_report = build_lab_report(
        v2_04_metadata,
        track=v2_04_profile.label,
        scenario=v2_04_storage["scenario"],
        learning_objectives=(
            "Use the pipeline equation to identify throughput backlog and consumer starvation.",
            "Explain how locality and placement change latency, bandwidth movement, cost, and freshness.",
            "Compare checkpoint and consistency policies by pause time, write storm, and restore evidence.",
            "Select a lifecycle policy that satisfies freshness, retention, cost, and reliability guardrails.",
        ),
        predictions={
            "part_a_throughput": v2_04_a_prediction.value,
            "part_b_placement": v2_04_b_prediction.value,
            "part_c_checkpointing": v2_04_c_prediction.value,
            "part_d_lifecycle": v2_04_d_prediction.value,
        },
        knob_settings={
            "demand_multiplier": v2_04_demand_multiplier.value,
            "placement_policy": v2_04_placement_policy.value,
            "working_set_pressure": v2_04_working_set_pressure.value,
            "checkpoint_policy": v2_04_checkpoint_policy.value,
            "checkpoint_interval_min": v2_04_checkpoint_interval.value,
            "lifecycle_policy": v2_04_lifecycle_policy.value,
            "freshness_target_min": v2_04_freshness_target.value,
        },
        binding_constraints={
            "throughput_bottleneck": v2_04_a_result["bottleneck"]["stage"],
            "binding_storage_amount": v2_04_binding["amount"],
            "binding_ratio": v2_04_binding["ratio"],
            "binding_module": v2_04_binding["module"],
        },
        decisions={
            "part_a_checkpoint": v2_04_a_checkpoint.value,
            "part_b_checkpoint": v2_04_b_checkpoint.value,
            "part_c_checkpoint": v2_04_c_checkpoint.value,
            "part_d_checkpoint": v2_04_d_checkpoint.value,
            "final_decision": v2_04_final_decision.value,
            "architecture_memo": v2_04_architecture_memo.value,
        },
        reflections={
            "selected_placement": v2_04_b_selected["label"],
            "selected_lifecycle": v2_04_d_selected["label"],
            "rejected_alternative": v2_04_rejected_alternative,
            "carry_forward": v2_04_storage["next_lab"],
        },
        residual_risk=f"{v2_04_b_selected['risk']} {v2_04_c_selected['risk']} {v2_04_d_selected['risk']}",
        evidence_summary={
            "backlog_gb_per_hour": v2_04_a_result["bottleneck"]["backlog_gb_h"],
            "starvation_pct": v2_04_a_result["starvation_pct"],
            "placement_latency_ms": v2_04_b_selected["latency_ms"],
            "placement_cost_day": v2_04_b_selected["cost_day"],
            "checkpoint_pause_s": v2_04_c_selected["pause_s"],
            "write_storm_gb_per_hour": v2_04_c_selected["write_storm_gb_h"],
            "restore_evidence_pct": v2_04_c_selected["restore_evidence_pct"],
            "monthly_lifecycle_cost": v2_04_d_selected["monthly_cost"],
            "lifecycle_retention_days": v2_04_d_selected["retention_days"],
            "lifecycle_durability_pct": v2_04_d_selected["durability_pct"],
        },
        final_decision={
            "decision": v2_04_final_decision.value,
            "selected_placement_policy": v2_04_b_selected["label"],
            "selected_lifecycle_policy": v2_04_d_selected["label"],
            "binding_storage_amount": v2_04_binding["amount"],
            "rejected_alternative": v2_04_rejected_alternative,
        },
        big_takeaways=(
            "Storage capacity is not storage throughput; backlog grows when consumer demand exceeds a stage capacity.",
            "Placement changes latency, movement cost, staleness, and governance even for the same bytes.",
            "A checkpoint only counts when it is restorable; low pause without verification is weak evidence.",
            "Lifecycle policy is valid only when freshness, retention, cost, and reliability all pass.",
        ),
        source_trace={
            "book_anchor": v2_04_metadata.book_anchor,
            "chapter_sources": "data_storage.qmd and data_storage_concepts.yml",
            "hardware_ref": v2_04_storage["hardware_ref"],
            "system_ref": v2_04_storage["system_ref"],
            "scenario_id": v2_04_variant.scenario_id,
        },
        result_snapshot={
            "track_id": v2_04_profile.track_id,
            "scenario_id": v2_04_variant.scenario_id,
            "throughput": v2_04_a_result,
            "placement": v2_04_b_selected,
            "checkpoint": v2_04_c_selected,
            "lifecycle": v2_04_d_selected,
            "binding": v2_04_binding,
            "distributed_training_implication": v2_04_storage["next_lab"],
        },
        incomplete_fields=tuple(_incomplete),
    )

    report_export_panel(v2_04_report)
    return


if __name__ == "__main__":
    app.run()
