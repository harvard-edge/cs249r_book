"""Pinned historical study policies and baselines; not general vocabulary standards."""
from __future__ import annotations

from pathlib import Path

_PRE_SIDECAR_VERSIONS = {"0.1-draft", "0.2-draft", "0.3-draft", "0.4-draft"}


_ALLOWED_PROV_ALIAS_TYPES = {"synonym", "quasi-synonym", "acronym", "upward-posting"}


_ALLOWED_OP_PREFIXES = (
    "add:",
    "update:",
    "retire:",
    "move:",
    "edge:",
    "rel_update:",
    "rel_remove:",
    "gap_add:",
    "gap_update:",
    "gap_remove:",
    "row:",
    "pe:",
    "guides:",
)


_ALLOWED_GUIDES_OP_PREFIXES = (
    "role_add:",
    "role_update:",
    "symptom_add:",
    "symptom_update:",
    "path_add:",
    "path_update:",
)


GENESIS_SNAPSHOT_REL_PATH = Path("taxonomy/method-study/genesis-0.4-baseline.json.gz")


BURNED_CLUSTERS_REL_PATH = Path("taxonomy/method-study/burned-clusters.json")


REPO_ROOT_PATH = Path(__file__).resolve().parents[4]


GENESIS_PRE_SIDECAR_TAXONOMY_SHA256 = "a3eefe42c766e031c43430aba061a56605c94117decf459b83962213f01de39e"


GENESIS_PRE_SIDECAR_NON_GUIDES_SHA256 = "1fd44b93987fea8ec835414c04cb8417cf127ae5d0ff4fbfb7fb9ade6ba934cc"


GENESIS_SNAPSHOT_GZ_SHA256 = "2b49f250a20c758d4cd46b32ab37b149baf395aaae50b1d5ab7d982a21f6ff09"


GENESIS_CANONICAL_JSON_SHA256 = "22b889e18a707601365572779bedff904f8a609d007a79d1ec62aadb7281a056"


GENESIS_HISTORICAL_PHASE3_UNITS_SHA256 = "da9c70a8dd6a4cb9272806ff4748b72a7da46e66b2a38b5c91144091451db279"


PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS: dict[str, tuple[str, ...]] = {
    "mlsys.safety-case": ("assurance case",),
    "mlsys.tree-ensemble": ("XGBoost",),
    "mlsys.constrained-decoding": ("grammar-guided decoding",),
    "mlsys.state-sharding": ("ZeRO", "FSDP"),
    "mlsys.paged-kv": ("PagedAttention",),
    "mlsys.graph-replay": ("CUDA Graphs",),
    "mlsys.device-streams": ("CUDA streams",),
}


BOOTSTRAP_GRANDFATHERED_ALIAS_EXCEPTIONS: dict[str, dict[str, object]] = {
    "mlsys.safety-case": {
        "aliases": ["assurance case"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft broader cross-domain term queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.tree-ensemble": {
        "aliases": ["XGBoost"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft named implementation entry term queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.constrained-decoding": {
        "aliases": ["grammar-guided decoding"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft narrower grammar-automaton subtype entry term queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.state-sharding": {
        "aliases": ["ZeRO", "FSDP"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft named sharded-optimizer/parameter implementations queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.paged-kv": {
        "aliases": ["PagedAttention"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft named paper/kernel entry term queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.graph-replay": {
        "aliases": ["CUDA Graphs"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft vendor-specific graph capture entry term queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.device-streams": {
        "aliases": ["CUDA streams"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft vendor-specific execution stream entry term queued for pre-1.0 legacy alias audit.",
    },
}


_REQUIRED_WITNESS_KEYS = ("compared_concept", "criterion_a", "criterion_b", "criterion_c", "scenario_a", "scenario_b")


_INTERNAL_LINEAGE_PREFIXES = ("pf_mlsysbook_", "pf_staffml_", "pf_internal_")


GENESIS_CLUSTER_MERGES: dict[str, str] = {
    "pf_cov_courses_2026_AIE": "pf_huyen_dmls",
    "pf_cov_courses_2026_C868": "pf_cmu_llmsys",
    "pf_cov_courses_2026_DMLS": "pf_huyen_dmls",
    "pf_cov_courses_2026_HAR": "pf_internal_mlsysbook",
    "pf_cov_courses_2026_M594": "pf_mit_han_tinyml",
    "pf_cov_courses_2026_MLS26": "pf_cov_mlsys_25_26_mlsys2026",
    "pf_cov_courses_2026_NIST": "pf_nist_ai_rmf",
    "pf_cov_courses_2026_S329": "pf_huyen_dmls",
    "pf_cov_courses_2026_S336": "pf_stanford_cs336",
    "pf_cov_courses_2026_UT378": "pf_ut_cs378",
}


GENESIS_EDITOR_AFFILIATED_CLUSTERS: frozenset[str] = frozenset({"pf_mlperf_inference"})


GENESIS_EDITOR_COAUTHOR_5YR_CLUSTERS: frozenset[str] = frozenset(
    {"pf_barroso_wsc", "pf_mlperf_inference", "pf_sglang_neurips24"}
)


GENESIS_INTERNAL_CLUSTERS: frozenset[str] = frozenset(
    {
        "pf_cov_courses_2026_HAR",
        "pf_internal_mlsysbook",
        "pf_mlsysbook_v1_v2_vol1/02_system_paradigms",
        "pf_mlsysbook_v1_v2_vol1/05_nn_computation",
        "pf_mlsysbook_v1_v2_vol1/06_nn_architectures",
        "pf_mlsysbook_v1_v2_vol1/12_benchmarking",
        "pf_mlsysbook_v1_v2_vol1/13_model_serving",
        "pf_mlsysbook_v1_v2_vol2/08_fleet_orchestration",
        "pf_mlsysbook_v1_v2_vol2/10_inference",
        "pf_mlsysbook_v1_v2_vol2/11_edge_intelligence",
        "pf_mlsysbook_v1_v2_vol2/14_robust_ai",
        "pf_mlsysbook_v1_v2_vol2/16_responsible_ai",
        "pf_mlsysbook_v3_v4_vol3/02_foundation_model",
        "pf_mlsysbook_v3_v4_vol3/03_test_time_compute",
        "pf_mlsysbook_v3_v4_vol3/04_context_engineering",
        "pf_mlsysbook_v3_v4_vol3/05_kv_cache",
        "pf_mlsysbook_v3_v4_vol3/06_long_term_memory",
        "pf_mlsysbook_v3_v4_vol3/10_agent_harness",
        "pf_mlsysbook_v3_v4_vol3/15_trajectory_curation",
        "pf_mlsysbook_v3_v4_vol3/17_rlvr",
        "pf_mlsysbook_v3_v4_vol3/19_agent_economics",
        "pf_mlsysbook_v3_v4_vol4/01_boundary",
        "pf_mlsysbook_v3_v4_vol4/02_body",
        "pf_mlsysbook_v3_v4_vol4/03_brain",
        "pf_mlsysbook_v3_v4_vol4/04_nervous",
        "pf_mlsysbook_v3_v4_vol4/05_data",
        "pf_mlsysbook_v3_v4_vol4/06_training",
        "pf_mlsysbook_v3_v4_vol4/07_evaluation",
        "pf_mlsysbook_v3_v4_vol4/08_perception",
        "pf_mlsysbook_v3_v4_vol4/10_intent",
        "pf_mlsysbook_v3_v4_vol4/11_planning",
        "pf_mlsysbook_v3_v4_vol4/12_enforcement",
        "pf_mlsysbook_v3_v4_vol4/13_placement",
        "pf_mlsysbook_v3_v4_vol4/14_intervention",
        "pf_mlsysbook_v3_v4_vol4/15_verification",
        "pf_mlsysbook_v3_v4_vol4/16_release",
        "pf_mlsysbook_v3_v4_vol4/18_frontier",
        "pf_staffml_heldout_p4",
        "pf_staffml_heldout_q4",
        "pf_staffml_heldout_q4_ext",
    }
)


GENESIS_BURNED_CLUSTER_IDS: frozenset[str] = frozenset(
    {
        "pf_amodei_concrete_safety",
        "pf_barroso_wsc",
        "pf_berkeley_aisafety",
        "pf_cmu_llmsys",
        "pf_cov_courses_2026_AIE",
        "pf_cov_courses_2026_B294",
        "pf_cov_courses_2026_BAG",
        "pf_cov_courses_2026_C442",
        "pf_cov_courses_2026_C779",
        "pf_cov_courses_2026_C868",
        "pf_cov_courses_2026_DLS",
        "pf_cov_courses_2026_DMLS",
        "pf_cov_courses_2026_HAR",
        "pf_cov_courses_2026_K17",
        "pf_cov_courses_2026_M594",
        "pf_cov_courses_2026_MLDP",
        "pf_cov_courses_2026_MLS26",
        "pf_cov_courses_2026_NIST",
        "pf_cov_courses_2026_OSDI",
        "pf_cov_courses_2026_RAT",
        "pf_cov_courses_2026_S329",
        "pf_cov_courses_2026_S336",
        "pf_cov_courses_2026_SEI",
        "pf_cov_courses_2026_USC",
        "pf_cov_courses_2026_UT378",
        "pf_cov_mlsys_25_26_mlsys2025",
        "pf_cov_mlsys_25_26_mlsys2026",
        "pf_cov_sysvenues_2025_asplos2025",
        "pf_cov_sysvenues_2025_eurosys2025",
        "pf_cov_sysvenues_2025_nsdi2025",
        "pf_cov_sysvenues_2025_osdi2025",
        "pf_cov_sysvenues_2025_sosp2025",
        "pf_cov_sysvenues_2026_heldout_asplos2026",
        "pf_cov_sysvenues_2026_heldout_eurosys2026",
        "pf_cov_sysvenues_2026_heldout_nsdi2026",
        "pf_cov_sysvenues_2026_heldout_osdi2026",
        "pf_distserve_osdi24",
        "pf_frontier_scaling_policies",
        "pf_huyen_dmls",
        "pf_internal_mlsysbook",
        "pf_iso42001_pas8800",
        "pf_iso_automotive_ul4600",
        "pf_k8s_ray_serving",
        "pf_kalra_statistical_safety",
        "pf_katz_verification",
        "pf_leveson_stpa",
        "pf_mit_han_tinyml",
        "pf_mlperf_inference",
        "pf_mlsysbook_v1_v2_vol1/02_system_paradigms",
        "pf_mlsysbook_v1_v2_vol1/05_nn_computation",
        "pf_mlsysbook_v1_v2_vol1/06_nn_architectures",
        "pf_mlsysbook_v1_v2_vol1/12_benchmarking",
        "pf_mlsysbook_v1_v2_vol1/13_model_serving",
        "pf_mlsysbook_v1_v2_vol2/08_fleet_orchestration",
        "pf_mlsysbook_v1_v2_vol2/10_inference",
        "pf_mlsysbook_v1_v2_vol2/11_edge_intelligence",
        "pf_mlsysbook_v1_v2_vol2/14_robust_ai",
        "pf_mlsysbook_v1_v2_vol2/16_responsible_ai",
        "pf_mlsysbook_v3_v4_vol3/02_foundation_model",
        "pf_mlsysbook_v3_v4_vol3/03_test_time_compute",
        "pf_mlsysbook_v3_v4_vol3/04_context_engineering",
        "pf_mlsysbook_v3_v4_vol3/05_kv_cache",
        "pf_mlsysbook_v3_v4_vol3/06_long_term_memory",
        "pf_mlsysbook_v3_v4_vol3/10_agent_harness",
        "pf_mlsysbook_v3_v4_vol3/15_trajectory_curation",
        "pf_mlsysbook_v3_v4_vol3/17_rlvr",
        "pf_mlsysbook_v3_v4_vol3/19_agent_economics",
        "pf_mlsysbook_v3_v4_vol4/01_boundary",
        "pf_mlsysbook_v3_v4_vol4/02_body",
        "pf_mlsysbook_v3_v4_vol4/03_brain",
        "pf_mlsysbook_v3_v4_vol4/04_nervous",
        "pf_mlsysbook_v3_v4_vol4/05_data",
        "pf_mlsysbook_v3_v4_vol4/06_training",
        "pf_mlsysbook_v3_v4_vol4/07_evaluation",
        "pf_mlsysbook_v3_v4_vol4/08_perception",
        "pf_mlsysbook_v3_v4_vol4/10_intent",
        "pf_mlsysbook_v3_v4_vol4/11_planning",
        "pf_mlsysbook_v3_v4_vol4/12_enforcement",
        "pf_mlsysbook_v3_v4_vol4/13_placement",
        "pf_mlsysbook_v3_v4_vol4/14_intervention",
        "pf_mlsysbook_v3_v4_vol4/15_verification",
        "pf_mlsysbook_v3_v4_vol4/16_release",
        "pf_mlsysbook_v3_v4_vol4/18_frontier",
        "pf_nist_ai_rmf",
        "pf_onnx_runtime_docs",
        "pf_orca_osdi22",
        "pf_sarathi_osdi24",
        "pf_sglang_neurips24",
        "pf_simplex_cbf_control",
        "pf_slora_mlsys24",
        "pf_specdec_icml23",
        "pf_staffml_heldout_p4",
        "pf_staffml_heldout_q4",
        "pf_staffml_heldout_q4_ext",
        "pf_stanford_cs329t",
        "pf_stanford_cs336",
        "pf_ut_cs378",
        "pf_vllm_sosp23",
    }
)


PINNED_BASELINE_SPLIT_HIERARCHY_EDGES: tuple[str, ...] = (
    "rel.activation-sparsity.sparsity",
    "rel.coherent-interconnect.host-device-interconnect",
    "rel.concept-drift.distribution-shift",
    "rel.confidential-computing.isolation",
    "rel.early-stopping.regularization",
    "rel.efficiency-frontier.pareto-frontier",
    "rel.feature-cross.features",
    "rel.fmea.hazard-analysis",
    "rel.llm-call-tracing.distributed-tracing",
    "rel.mode-collapse.training-instability",
    "rel.model-flops-utilization.utilization",
    "rel.neural-architecture-search.hyperparameter-search",
    "rel.pipeline-jungle.technical-debt",
    "rel.simt.accelerator-execution-architecture",
    "rel.transformer-variants.transformer",
    "rel.wafer-scale-acceleration.domain-specific-architecture",
)


PINNED_BASELINE_ZERO_WARRANT_CONCEPTS: tuple[str, ...] = (
    "mlsys.agent-training-environment",
    "mlsys.ai-bill-of-materials",
    "mlsys.ai-incident-reporting",
    "mlsys.artifact-archival",
    "mlsys.attention-ffn-disaggregation",
    "mlsys.carbon-accounting-method",
    "mlsys.content-provenance",
    "mlsys.coordination-failure",
    "mlsys.data-model-licensing",
    "mlsys.diffusion-feature-caching",
    "mlsys.fault-tree-analysis",
    "mlsys.filtered-vector-search",
    "mlsys.frontier-safety-framework",
    "mlsys.game-day",
    "mlsys.hyperparameter-transfer",
    "mlsys.instruction-hierarchy",
    "mlsys.label-shift-estimation",
    "mlsys.leader-election",
    "mlsys.model-deprecation",
    "mlsys.model-directed-control-flow",
    "mlsys.model-merging",
    "mlsys.nearest-neighbor",
    "mlsys.off-policy-evaluation",
    "mlsys.over-refusal",
    "mlsys.representational-bias-evaluation",
    "mlsys.rl-training-colocation",
    "mlsys.unsupervised-learning",
)


_STAGE2_PAIR_ORDER: tuple[str, ...] = ("F", "S", "P", "Q", "T1", "T2")


_STAGE2_ANCHOR_ROW_QUOTAS: dict[str, dict[str, int]] = {
    "Q": {"F": 8, "S": 7, "P": 7, "T1": 7, "T2": 7},
    "P": {"F": 6, "S": 6, "Q": 6, "T1": 6, "T2": 6},
    "S": {"F": 4, "P": 5, "Q": 5, "T1": 5, "T2": 5},
    "F": {"S": 5, "P": 5, "Q": 5, "T1": 1, "T2": 2},
    "T1": {"F": 1, "S": 1, "P": 2, "Q": 2, "T2": 0},
    "T2": {"F": 1, "S": 1, "P": 1, "Q": 2, "T1": 1},
}


_SPECIALIST_SLOTS_EXPECTED = frozenset({
    "F3", "S4", "P1", "P2", "P3", "P5", "Q1", "Q2", "Q3", "Q4a", "Q4b", "Q5", "Q6",
})


_CHAIR_SIGN_OFF_AREAS_TO_SEAT: dict[str, str] = {
    "F1": "F",
    "F2": "F",
    "S1": "S",
    "S2": "S",
    "S3": "S",
    "P4": "P",
    "T1": "T",
    "T2": "T",
}


_BOARD_SEATS_EXPECTED = frozenset({"F", "S", "P", "Q", "T", "V1", "V2"})


_EXPECTED_STANDING_SLOTS = frozenset({
    *(
        (seat, role)
        for seat in ("F", "S", "P", "Q", "V1", "V2")
        for role in ("chair", "alternate")
    ),
    ("T", "t1_co_chair"),
    ("T", "t2_co_chair"),
    ("T", "t1_alternate"),
    ("T", "t2_alternate"),
})


_PRIMARY_BOOTSTRAP_CI_KEYS: tuple[str, ...] = (
    "alpha_subject",
    "kappa_bar_subject",
    "cond_hooper",
    "all_hooper",
    "cond_rolling",
    "all_rolling",
    "alpha_masi",
)


_ANCHOR_BOOTSTRAP_CI_KEYS: tuple[str, ...] = (
    "alpha_subject_anchor",
    "alpha_masi_anchor",
    "kappa_fleiss_anchor",
)
