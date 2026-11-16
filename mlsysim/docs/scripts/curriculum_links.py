"""Render validated cross-chapter, lab, and slide links for MLSys·im docs.

Reads ``mlsysim/docs/config/curriculum_links.yml`` and emits clean HTML/Markdown
components that respect the ecosystem ``.mls-unreleased`` release gate:
Volumes I and II render everywhere; Volumes III and IV carry ``class="mls-unreleased"``
so they appear on ``cs249r_book_dev`` and localhost previews while hiding cleanly
with zero broken links or orphaned separators on live ``mlsysbook.ai``.
"""

from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _find_docs_dir() -> Path:
    here = Path(__file__).resolve()
    if (here.parent.parent / "config" / "curriculum_links.yml").exists():
        return here.parent.parent
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        for rel in [Path("config/curriculum_links.yml"), Path("mlsysim/docs/config/curriculum_links.yml")]:
            target = candidate / rel
            if target.exists():
                return target.parent.parent
    raise FileNotFoundError("Could not locate mlsysim/docs/config/curriculum_links.yml")


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    docs_dir = _find_docs_dir()
    return yaml.safe_load((docs_dir / "config" / "curriculum_links.yml").read_text(encoding="utf-8"))


def resolve_chapter(ref: str) -> dict[str, Any]:
    """Resolve ``volN.NN_slug`` into URLs, labels, and release status."""
    manifest = load_manifest()
    vol_key, slug = ref.split(".", 1)
    vol = manifest["volumes"][vol_key]
    ch = vol["chapters"][slug]
    num = int(ch["num"])
    return {
        "ref": ref,
        "vol_key": vol_key,
        "vol_label": vol["label"],
        "vol_full_title": vol["full_title"],
        "released": bool(vol["released"]),
        "slug": slug,
        "num": num,
        "title": ch["title"],
        "book_url": f"{vol['book_base']}/{slug}/{slug}.html",
        "lab_slug": ch["lab"],
        "lab_url": f"{vol['labs_base']}/{ch['lab']}/",
        "slides_url": f"{vol['slides_page']}#ch-{num:02d}",
    }


def _qmd_to_html_href(href: str) -> str:
    """Convert internal ``.qmd`` targets to ``.html`` for raw ``{=html}`` blocks."""
    if "#" in href:
        path_part, frag = href.split("#", 1)
        if path_part.endswith(".qmd"):
            return f"{path_part[:-4]}.html#{frag}"
        return href
    if href.endswith(".qmd"):
        return f"{href[:-4]}.html"
    return href


@lru_cache(maxsize=512)
def resolve_solver_api_href(solver_name: str, docs_prefix: str = "") -> str:
    """Resolve a concrete solver class name to its generated ``api/*.html`` page + anchor."""
    docs_dir = _find_docs_dir()
    direct_qmd = docs_dir / "api" / f"solvers.{solver_name}.qmd"
    if direct_qmd.exists():
        return f"{docs_prefix}api/solvers.{solver_name}.html"
    try:
        import mlsysim.solvers as _solvers

        cls = getattr(_solvers, solver_name, None)
        if cls is not None:
            mod_full = cls.__module__
            mod_rel = mod_full.removeprefix("mlsysim.")
            if (docs_dir / "api" / f"{mod_rel}.qmd").exists():
                return f"{docs_prefix}api/{mod_rel}.html#{mod_full}.{solver_name}"
    except Exception:
        pass
    return f"{docs_prefix}api/index.html"


def render_tutorial_links(tutorial_key: str, *, map_href: str = "../curriculum-map.qmd") -> str:
    """Render a 'Where This Clicks in the Textbook' grid for a tutorial page."""
    manifest = load_manifest()
    entry = manifest["tutorials"].get(tutorial_key)
    if not entry:
        raise KeyError(f"Unknown tutorial key in curriculum_links.yml: {tutorial_key!r}")

    cards: list[str] = []
    for item in entry["links"]:
        ch = resolve_chapter(item["ref"])
        role = html.escape(item.get("role", ""))
        why = html.escape(item.get("why", ""))
        title = html.escape(ch["title"])
        badge_text = html.escape(f"{ch['vol_label']} · Ch {ch['num']}")
        unreleased_cls = "" if ch["released"] else " mls-unreleased"
        vol_cls = f"sim-vol-{ch['vol_key']}"
        role_html = f'<span class="sim-curr-role">{role}</span>' if role else ""
        cards.append(
            f'<div class="sim-curr-card{unreleased_cls}">'
            f'<div class="sim-curr-meta">'
            f'<span class="sim-vol-pill {vol_cls}">{badge_text}</span>'
            f"{role_html}"
            f"</div>"
            f'<div class="sim-curr-title">'
            f'<a href="{ch["book_url"]}">{title}</a>'
            f"</div>"
            f'<p class="sim-curr-why">{why}</p>'
            f'<div class="sim-curr-actions">'
            f'<a href="{ch["book_url"]}">Read Chapter →</a>'
            f'<span class="sim-curr-dot">·</span>'
            f'<a href="{ch["lab_url"]}">Interactive Lab</a>'
            f'<span class="sim-curr-dot">·</span>'
            f'<a href="{ch["slides_url"]}">Slides</a>'
            f"</div>"
            f"</div>"
        )

    cards_html = "\n".join(cards)
    html_map_href = _qmd_to_html_href(map_href)
    return (
        "```{=html}\n"
        '<div class="sim-curr-wrap">\n'
        f'<div class="sim-curr-grid">\n{cards_html}\n</div>\n'
        f'<p class="sim-curr-footer">See how these constraints connect across the full series in the '
        f'<a href="{html.escape(html_map_href)}">Cross-Volume Textbook Map →</a></p>\n'
        "</div>\n"
        "```\n"
    )


def render_threads_cards(*, docs_prefix: str = "") -> str:
    """Render the 7 cross-volume physics threads for curriculum-map.qmd."""
    manifest = load_manifest()
    blocks: list[str] = []
    for thread in manifest["threads"]:
        t_title = html.escape(thread["title"])
        t_sum = html.escape(thread["summary"])
        solvers_html = ", ".join(
            f'<a href="{html.escape(resolve_solver_api_href(s, docs_prefix))}"><code>{html.escape(s)}</code></a>'
            for s in thread.get("solvers", [])
        )
        tuts_html = " · ".join(
            f'<a href="{docs_prefix}{html.escape(_qmd_to_html_href(t["href"]))}">{html.escape(t["title"])}</a>'
            for t in thread.get("tutorials", [])
        )
        stops_html: list[str] = []
        for stop in thread["stops"]:
            ch = resolve_chapter(stop["ref"])
            unreleased_cls = "" if ch["released"] else " mls-unreleased"
            vol_cls = f"sim-vol-{ch['vol_key']}"
            badge = html.escape(f"{ch['vol_label']} · Ch {ch['num']}")
            note = html.escape(stop["note"])
            stops_html.append(
                f'<div class="sim-thread-stop{unreleased_cls}">'
                f'<div class="sim-curr-meta">'
                f'<span class="sim-vol-pill {vol_cls}">{badge}</span>'
                f'<a class="sim-thread-ch" href="{ch["book_url"]}">{html.escape(ch["title"])}</a>'
                f"</div>"
                f'<p class="sim-curr-why">{note}</p>'
                f'<div class="sim-curr-actions">'
                f'<a href="{ch["book_url"]}">Chapter</a>'
                f'<span class="sim-curr-dot">·</span>'
                f'<a href="{ch["lab_url"]}">Lab</a>'
                f'<span class="sim-curr-dot">·</span>'
                f'<a href="{ch["slides_url"]}">Slides</a>'
                f"</div>"
                f"</div>"
            )
        stops_joined = "\n".join(stops_html)
        blocks.append(
            f'<div class="sim-thread-card" id="thread-{html.escape(thread["id"])}">\n'
            f"<h3>{t_title}</h3>\n"
            f'<p class="sim-thread-summary">{t_sum}</p>\n'
            f'<div class="sim-thread-stops">\n{stops_joined}\n</div>\n'
            f'<div class="sim-thread-tools">'
            f"<span><strong>Solvers:</strong> {solvers_html}</span>"
            f'<span><strong>Tutorials:</strong> {tuts_html}</span>'
            f"</div>\n"
            f"</div>"
        )
    return "```{=html}\n" + "\n\n".join(blocks) + "\n```\n"


# Companion MLSys·im tools per chapter for the chapter-by-chapter lookup table.
_CHAPTER_TOOLS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    # Volume I
    "vol1.01_introduction": ("SingleNodeModel", [("0 · Hello, Roofline", "tutorials/00_hello_roofline.qmd")]),
    "vol1.02_system_paradigms": ("SynthesisSolver", [("12 · Design Space Exploration", "tutorials/12_design_space_exploration.qmd")]),
    "vol1.03_ml_lifecycle": ("EconomicsModel", [("8 · The $9M Question", "tutorials/08_nine_million_dollar.qmd")]),
    "vol1.04_data_engineering": ("DataModel, TransformationModel", [("4 · Starving the GPU", "tutorials/04_starving_the_gpu.qmd")]),
    "vol1.05_nn_computation": ("DenseNetworkModel, SingleNodeModel", [("0 · Hello, Roofline", "tutorials/00_hello_roofline.qmd")]),
    "vol1.06_nn_architectures": ("KvAttentionDesignModel, ServingModel", [("2 · Two Phases", "tutorials/02_two_phases.qmd")]),
    "vol1.07_frameworks": ("EpochOverheadModel", [("Composable Pipelines", "tutorials/01_pipeline_callbacks.qmd")]),
    "vol1.08_training": ("TrainingMemoryModel, ScalingModel", [("11 · Memory, Capacity & MoE", "tutorials/11_training_memory_capacity_moe.qmd")]),
    "vol1.09_data_selection": ("ScalingModel", [("Datasets Zoo", "zoo/datasets.qmd")]),
    "vol1.10_model_compression": ("CompressionModel", [("5 · Quantization", "tutorials/05_quantization.qmd")]),
    "vol1.11_hw_acceleration": ("SingleNodeModel, WeightStreamingModel", [("1 · The Memory Wall", "tutorials/01_memory_wall.qmd"), ("10 · GPU vs. Wafer", "tutorials/10_gpu_vs_wafer.qmd")]),
    "vol1.12_benchmarking": ("SensitivitySolver, SynthesisSolver", [("9 · Sensitivity Analysis", "tutorials/09_sensitivity.qmd")]),
    "vol1.13_model_serving": ("ServingModel, TailLatencyModel", [("2 · Two Phases", "tutorials/02_two_phases.qmd"), ("3 · KV-Cache Tax", "tutorials/03_kv_cache.qmd")]),
    "vol1.14_ml_ops": ("SequentialCanaryModel, TraceRetentionModel", [("Ops Zoo", "zoo/ops.qmd")]),
    "vol1.15_responsible_engr": ("ResponsibleEngineeringModel", [("7 · Geography Matters", "tutorials/07_geography.qmd")]),
    "vol1.16_conclusion": ("SingleNodeModel, EconomicsModel", [("13 · Full-Stack Audit", "tutorials/12_full_stack_audit.qmd")]),
    # Volume II
    "vol2.01_introduction": ("DistributedModel, EconomicsModel", [("13 · Full-Stack Audit", "tutorials/12_full_stack_audit.qmd")]),
    "vol2.02_compute_infrastructure": ("EconomicsModel, FacilityPowerModel", [("8 · The $9M Question", "tutorials/08_nine_million_dollar.qmd"), ("Silicon Zoo", "zoo/hardware.qmd")]),
    "vol2.03_network_fabrics": ("TopologyModel, TransferLadderModel", [("Distributed Training", "tutorials/distributed.qmd")]),
    "vol2.04_data_storage": ("StorageFootprintModel, TieredCheckpointModel", [("4 · Starving the GPU", "tutorials/04_starving_the_gpu.qmd")]),
    "vol2.05_collective_communication": ("AllReduceAlgorithmModel, HierarchicalAllReduceModel", [("Distributed Training", "tutorials/distributed.qmd")]),
    "vol2.06_distributed_training": ("DistributedModel, MoERoutingModel, ZeroShardingModel", [("6 · Scaling to 1000 GPUs", "tutorials/06_scaling_1000_gpus.qmd"), ("11 · Memory, Capacity & MoE", "tutorials/11_training_memory_capacity_moe.qmd")]),
    "vol2.07_fault_tolerance": ("ReliabilityModel, CheckpointModel", [("6 · Scaling to 1000 GPUs", "tutorials/06_scaling_1000_gpus.qmd")]),
    "vol2.08_fleet_orchestration": ("OrchestrationModel, PreemptionTaxModel", [("12 · Design Space Exploration", "tutorials/12_design_space_exploration.qmd")]),
    "vol2.09_performance_engineering": ("EfficiencyModel, GoodputModel", [("9 · Sensitivity Analysis", "tutorials/09_sensitivity.qmd"), ("Differential Explainer", "tutorials/02_differential_explainer.qmd")]),
    "vol2.10_inference": ("ContinuousBatchingModel, ServingCapacityModel", [("3 · KV-Cache Tax", "tutorials/03_kv_cache.qmd"), ("11 · Memory, Capacity & MoE", "tutorials/11_training_memory_capacity_moe.qmd")]),
    "vol2.11_edge_intelligence": ("CompressionModel, PrecisionDividendModel", [("5 · Quantization", "tutorials/05_quantization.qmd"), ("Platforms Zoo", "zoo/platforms.qmd")]),
    "vol2.12_ops_scale": ("SpanCriticalPathModel, OutageCostModel", [("Ops Zoo", "zoo/ops.qmd")]),
    "vol2.13_security_privacy": ("ResponsibleEngineeringModel, PolicyComparisonModel", [("Ops Zoo", "zoo/ops.qmd")]),
    "vol2.14_robust_ai": ("SeriesFailureModel, TailLatencyModel", [("9 · Sensitivity Analysis", "tutorials/09_sensitivity.qmd")]),
    "vol2.15_sustainable_ai": ("SustainabilityModel, WaterUseModel, PueRetrofitModel", [("7 · Geography Matters", "tutorials/07_geography.qmd")]),
    "vol2.16_responsible_ai": ("ResponsibleEngineeringModel", [("Infrastructure Zoo", "zoo/infra.qmd")]),
    "vol2.17_conclusion": ("DistributedModel, SustainabilityModel, EconomicsModel", [("13 · Full-Stack Audit", "tutorials/12_full_stack_audit.qmd")]),
    # Volume III
    "vol3.01_introduction": ("HorizonCompoundingModel, AcceptedTaskCostModel", [("Agent Zoo", "zoo/agents.qmd")]),
    "vol3.02_foundation_model": ("ModelCallModel, ContextAdmissionModel", [("2 · Two Phases", "tutorials/02_two_phases.qmd")]),
    "vol3.03_test_time_compute": ("InferenceScalingModel, BestOfNPoolModel", [("8 · The $9M Question", "tutorials/08_nine_million_dollar.qmd")]),
    "vol3.04_context_engineering": ("ContextBudgetModel, CompactionCadenceModel", [("3 · KV-Cache Tax", "tutorials/03_kv_cache.qmd")]),
    "vol3.05_kv_cache": ("PagedAllocationModel, PrefixRebuildModel", [("3 · KV-Cache Tax", "tutorials/03_kv_cache.qmd")]),
    "vol3.06_long_term_memory": ("RetrievalPayloadModel, AnnFootprintModel", [("Agent Scenarios", "api/scenario-domain-agents.qmd")]),
    "vol3.07_tool_calling": ("BlindRetryModel, ToolCatalogSelectionModel", [("Agent Scenarios", "api/scenario-domain-agents.qmd")]),
    "vol3.08_sandboxes": ("SyscallTaxModel, SandboxPoolReserveModel", [("Agent Scenarios", "api/scenario-domain-agents.qmd")]),
    "vol3.09_security": ("InjectionExposureModel, PolicyComparisonModel", [("Agent Scenarios", "api/scenario-domain-agents.qmd")]),
    "vol3.10_agent_harness": ("ApprovalEscrowModel, TrajectoryBudgetMeterModel", [("Agent Zoo", "zoo/agents.qmd")]),
    "vol3.11_durable_execution": ("GroupCommitModel, CheckpointTraceModel", [("6 · Scaling to 1000 GPUs", "tutorials/06_scaling_1000_gpus.qmd")]),
    "vol3.12_failure_recovery": ("RepairBreakEvenModel, RetryAmplificationModel", [("Agent Scenarios", "api/scenario-domain-agents.qmd")]),
    "vol3.13_evaluation": ("BinomialIntervalsModel, PairedComparisonModel", [("Statistics Scenarios", "api/scenario-domain-statistics.qmd")]),
    "vol3.14_agent_operations": ("SpanCriticalPathModel, SequentialCanaryModel", [("Ops Zoo", "zoo/ops.qmd")]),
    "vol3.15_trajectory_curation": ("FailureTriageModel, RunSplitLeakageModel", [("Agent Scenarios", "api/scenario-domain-agents.qmd")]),
    "vol3.16_fine_tuning": ("TargetMaskModel, SequencePackingModel", [("Training Scenarios", "api/scenario-domain-training.qmd")]),
    "vol3.17_rlvr": ("RewardCompositionModel, ContinuationForkCostModel", [("Agent Scenarios", "api/scenario-domain-agents.qmd")]),
    "vol3.18_multi_agent": ("DeficitRoundRobinModel, CohortAttemptCostModel", [("Agent Zoo", "zoo/agents.qmd")]),
    "vol3.19_agent_economics": ("AcceptedTaskCostModel, PrefixCacheBillModel", [("8 · The $9M Question", "tutorials/08_nine_million_dollar.qmd")]),
    "vol3.20_conclusion": ("AcceptedTaskCostModel, HorizonCompoundingModel", [("Agent Zoo", "zoo/agents.qmd")]),
    # Volume IV
    "vol4.01_boundary": ("StoppingBudgetModel, PhaseLagBudgetModel", [("Embodied Zoo", "zoo/embodied.qmd")]),
    "vol4.02_body": ("BrakingCeilingsModel, HubDriveElectricalModel", [("Actuators Zoo", "zoo/actuators.qmd")]),
    "vol4.03_brain": ("ProposerTrafficModel, DecoderFamiliesModel", [("Embodied Zoo", "zoo/embodied.qmd")]),
    "vol4.04_nervous": ("CanFdBusModel, EtherCatCycleModel", [("Robotics Scenarios", "api/scenario-domain-robotics.qmd")]),
    "vol4.05_data": ("ReplayBufferModel, InputDemandModel", [("4 · Starving the GPU", "tutorials/04_starving_the_gpu.qmd")]),
    "vol4.06_training": ("TrainingMemoryModel, DistributedModel", [("11 · Memory, Capacity & MoE", "tutorials/11_training_memory_capacity_moe.qmd")]),
    "vol4.07_evaluation": ("VlaEvaluationModel, OnsetTrialModel", [("Robotics Scenarios", "api/scenario-domain-robotics.qmd")]),
    "vol4.08_perception": ("ObservationAgeModel, LidarSweepAgeModel", [("Sensors Zoo", "zoo/sensors.qmd")]),
    "vol4.09_memory": ("HostMapSizingModel, RaymarchTrafficModel", [("Robotics Scenarios", "api/scenario-domain-robotics.qmd")]),
    "vol4.10_intent": ("IntentCadenceModel, IntentLeaseExpiryModel", [("Robotics Scenarios", "api/scenario-domain-robotics.qmd")]),
    "vol4.11_planning": ("TwoLinkJointRateModel, JerkLimitedStopModel", [("Robotics Scenarios", "api/scenario-domain-robotics.qmd")]),
    "vol4.12_enforcement": ("StoppingBudgetModel, TractionLossModel", [("Embodied Zoo", "zoo/embodied.qmd")]),
    "vol4.13_placement": ("ResponseTimeModel, ProposerTrafficModel", [("12 · Design Space Exploration", "tutorials/12_design_space_exploration.qmd")]),
    "vol4.14_intervention": ("TakeoverBudgetModel, SupervisorQueueModel", [("Robotics Scenarios", "api/scenario-domain-robotics.qmd")]),
    "vol4.15_verification": ("DetectionShortfallModel, EstimatorConsistencyModel", [("Robotics Scenarios", "api/scenario-domain-robotics.qmd")]),
    "vol4.16_release": ("SequentialCanaryModel, BinomialIntervalsModel", [("Robotics Scenarios", "api/scenario-domain-robotics.qmd")]),
    "vol4.17_responsibility": ("StoppingBudgetModel, ResponsibleEngineeringModel", [("Embodied Zoo", "zoo/embodied.qmd")]),
    "vol4.18_frontier": ("StoppingBudgetModel, PhaseLagBudgetModel", [("Embodied Zoo", "zoo/embodied.qmd")]),
}


def render_volume_tables(*, docs_prefix: str = "") -> str:
    """Render chapter-by-chapter MLSys·im lookup tables for Volumes I–IV."""
    manifest = load_manifest()
    sections: list[str] = []
    for vol_key, vol_data in manifest["volumes"].items():
        released = bool(vol_data["released"])
        rows: list[str] = []
        for slug in vol_data["chapters"]:
            ref = f"{vol_key}.{slug}"
            ch = resolve_chapter(ref)
            solvers_csv, guides = _CHAPTER_TOOLS.get(ref, ("SingleNodeModel", []))
            solver_links = ", ".join(
                f'<a href="{html.escape(resolve_solver_api_href(s.strip(), docs_prefix))}"><code>{html.escape(s.strip())}</code></a>'
                for s in solvers_csv.split(",")
                if s.strip()
            )
            guide_links = " · ".join(
                f'<a href="{docs_prefix}{html.escape(_qmd_to_html_href(href))}">{html.escape(label)}</a>'
                for label, href in guides
            )
            rows.append(
                f"<tr>"
                f"<td><code>{ch['num']:02d}</code></td>"
                f'<td><a href="{ch["book_url"]}"><strong>{html.escape(ch["title"])}</strong></a></td>'
                f'<td><a href="{ch["lab_url"]}">Lab {ch["num"]:02d}</a> · '
                f'<a href="{ch["slides_url"]}">Slides</a></td>'
                f"<td>{solver_links}</td>"
                f"<td>{guide_links}</td>"
                f"</tr>"
            )
        table_html = (
            f'<h3>{html.escape(vol_data["full_title"])}</h3>\n'
            '<table class="table table-sm">\n'
            "<thead><tr>"
            "<th>Ch</th><th>Textbook Chapter</th><th>Lab &amp; Slides</th>"
            "<th>Key Solvers</th><th>MLSys·im Guide / Zoo</th>"
            "</tr></thead>\n"
            "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"
        )
        if not released:
            table_html = f'<div class="mls-unreleased">\n{table_html}\n</div>'
        sections.append(table_html)
    return "```{=html}\n" + "\n\n".join(sections) + "\n```\n"

