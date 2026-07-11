from collections import Counter
from pathlib import Path

from mlperf.assets import asset_dossier, has_asset_dossier
from mlperf.registry import (
    EDU_SCENARIOS,
    PROFILES,
    QUALITY_TARGET_BASES,
    REFERENCE_PROTOCOL_FIELDS,
    PUBLIC_RESULT_SCENARIOS,
    PUBLIC_STATUSES,
    STARTER_WORKLOAD_ORDER,
    STARTER_WORKLOADS,
    RESEARCH_WORKLOAD_ORDER,
    RESEARCH_WORKLOADS,
    STANDARD_WORKLOAD_ORDER,
    STANDARD_WORKLOADS,
    load_registry,
    public_contract_report,
    select_workloads,
)


def test_profiles_are_min_max_pro():
    assert PROFILES == ("min", "max", "pro")


def test_registry_loads_current_workloads():
    workloads = load_registry()
    assert len(workloads) >= 20
    assert STARTER_WORKLOADS.issubset(workloads)
    assert STANDARD_WORKLOADS.issubset(workloads)
    assert RESEARCH_WORKLOADS.issubset(workloads)


def test_packaged_registry_copy_matches_flat_registry():
    repo_root = Path(__file__).resolve().parents[1]
    flat = repo_root / "workloads.yaml"
    packaged = repo_root / "src" / "mlperf_edu" / "workloads.yaml"
    assert packaged.read_text() == flat.read_text()


def test_registry_loads_native_suite_workload_variant_layout(tmp_path):
    language = tmp_path / "suites" / "language"
    language.mkdir(parents=True)
    (language / "nanogpt-train.yaml").write_text(
        """
model: nanogpt-12m
dataset: tinyshakespeare
quality_target:
  metric: cross_entropy_loss
  value: 2.3
scenario: single_stream
public:
  status: systems-only
  rationale: Native flat workload fixture.
runner:
  min: mlperf.runners.nanogpt:run_min
  max: mlperf.runners.nanogpt:run_max
""".lstrip()
    )

    workload_dir = tmp_path / "suites" / "slm" / "smollm2-chat-inference"
    variants = workload_dir / "variants"
    variants.mkdir(parents=True)
    (workload_dir / "workload.yaml").write_text(
        """
id: smollm2-chat-inference
model: SmolLM2-135M-Instruct
dataset: prompt-suite-local
quality_target:
  metric: generated_tokens
  value: 8
functional_check:
  metric: generated_tokens
  condition: generated_tokens >= 8
scenario: single_stream
public:
  status: performance-bearing
  rationale: Native canonical workload fixture.
runner:
  min: mlperf.runners.slm:run_decode_min
  max: mlperf.runners.slm:run_decode_max
""".lstrip()
    )
    (variants / "baseline.yaml").write_text(
        """
id: slm-decode
default_variant: true
""".lstrip()
    )
    (variants / "quantized-int8.yaml").write_text(
        """
id: slm-quantized-decode
model: SmolLM2-135M-Instruct dynamic-int8
runner:
  min: mlperf.runners.slm:run_quantized_decode_min
  max: mlperf.runners.slm:run_quantized_decode_max
""".lstrip()
    )

    workloads = load_registry(tmp_path, validate=False)

    assert sorted(workloads) == ["nanogpt-train", "slm-decode", "slm-quantized-decode"]
    assert workloads["nanogpt-train"].suite == "language"
    assert workloads["slm-decode"].canonical_workload == "smollm2-chat-inference"
    assert workloads["slm-decode"].variant == "baseline"
    assert workloads["slm-decode"].default_variant is True
    assert workloads["slm-quantized-decode"].variant == "quantized-int8"
    assert (
        workloads["slm-quantized-decode"].raw["runner"]["max"]
        == "mlperf.runners.slm:run_quantized_decode_max"
    )


def test_canonical_workloads_and_variants_live_in_registry():
    workloads = load_registry()

    slm_variants = {
        workload.variant: workload.id
        for workload in workloads.values()
        if workload.canonical_workload == "smollm2-chat-inference"
    }
    nanogpt_variants = {
        workload.variant: workload.id
        for workload in workloads.values()
        if workload.canonical_workload == "nanogpt-inference"
    }

    assert slm_variants == {
        "baseline": "slm-decode",
        "quantized-int8": "slm-quantized-decode",
        "batched-b4": "slm-batched-decode",
        "long-context": "slm-long-context-decode",
    }
    assert nanogpt_variants == {
        "prefill": "nanogpt-prefill",
        "decode": "nanogpt-decode",
        "fp32-b16": "nanogpt-decode-fp32-b16",
        "fp16-b16": "nanogpt-decode-fp16-b16",
        "speculative": "nanogpt-decode-spec",
    }
    assert workloads["slm-decode"].default_variant is True
    assert workloads["nanogpt-decode"].default_variant is True


def test_all_workloads_declare_min_and_max_runners():
    workloads = load_registry()

    assert workloads
    for workload in workloads.values():
        runner = workload.raw.get("runner", {})
        assert "min" in runner, workload.id
        assert "max" in runner, workload.id


def test_all_workloads_declare_public_contract_metadata():
    workloads = load_registry()
    counts = Counter(workload.public_status for workload in workloads.values())

    assert set(counts).issubset(PUBLIC_STATUSES)
    assert counts == {
        "score-bearing": 5,
        "performance-bearing": 3,
        "systems-only": 22,
    }
    assert all(workload.public_rationale for workload in workloads.values())


def test_public_result_workloads_use_educational_mlcommons_scenarios():
    workloads = load_registry()

    assert {workload.scenario for workload in workloads.values()}.issubset(
        EDU_SCENARIOS
    )
    for workload in workloads.values():
        if workload.public_status in {"score-bearing", "performance-bearing"}:
            assert workload.scenario in PUBLIC_RESULT_SCENARIOS, workload.id
        if workload.public_status == "score-bearing":
            assert workload.scenario == "training", workload.id
        if workload.public_status == "performance-bearing":
            assert workload.scenario in {"single_stream", "offline", "server"}, (
                workload.id
            )


def test_score_bearing_workloads_are_public_quality_contracts():
    workloads = load_registry()
    selected = select_workloads(workloads, public_status="score-bearing")

    assert {workload.id for workload in selected} == {
        "anomaly-ae-train",
        "micro-dlrm-train",
        "mobilenetv2-train",
        "nanogpt-train",
        "resnet18-train",
    }
    for workload in selected:
        assert workload.dataset, workload.id
        assert workload.raw.get("dataset_source"), workload.id
        assert has_asset_dossier(workload.dataset), workload.id
        assert workload.quality_metric, workload.id
        assert workload.quality_value is not None, workload.id
        assert workload.quality_direction in {"higher", "lower"}, workload.id
        assert workload.quality_target_basis in QUALITY_TARGET_BASES, workload.id
        assert workload.quality_reference_runs, workload.id
        assert workload.quality_variance_summary, workload.id
        assert (
            workload.quality_variance_summary["runs"] == workload.quality_reference_runs
        ), workload.id
        assert workload.quality_variance_summary["statistic"], workload.id
        assert workload.quality_variance_summary["acceptance_rule"], workload.id
        assert isinstance(workload.quality_reference_protocol, dict), workload.id
        for field in REFERENCE_PROTOCOL_FIELDS:
            assert workload.quality_reference_protocol.get(field), (workload.id, field)
        assert (
            len(workload.quality_reference_protocol["seeds"])
            == workload.quality_reference_runs
        ), workload.id
        assert workload.quality_reviewer_notes, workload.id
        assert workload.raw.get("verified_baseline"), workload.id


def test_performance_bearing_workloads_have_functional_contracts():
    workloads = load_registry()
    selected = select_workloads(workloads, public_status="performance-bearing")

    assert {workload.id for workload in selected} == {
        "nanogpt-decode",
        "nanogpt-prefill",
        "slm-decode",
    }
    for workload in selected:
        functional_check = workload.raw.get("functional_check")
        assert isinstance(functional_check, dict), workload.id
        assert functional_check.get("metric"), workload.id
        assert functional_check.get("condition"), workload.id
        assert "reference_runs" not in functional_check, workload.id
        reference_protocol = workload.raw.get("performance_reference_protocol")
        assert isinstance(reference_protocol, dict), workload.id
        assert reference_protocol.get("profile") == "max", workload.id
        assert reference_protocol.get("reference_runs") == 5, workload.id
        assert reference_protocol.get("seeds") == [0, 1, 2, 3, 4], workload.id
        for field in (
            "backend",
            "machine_class",
            "dataset_mode",
            "aggregation",
            "functional_acceptance",
            "artifact_policy",
            "rerun_policy",
        ):
            assert reference_protocol.get(field), (workload.id, field)
        protocol = workload.raw.get("measurement_protocol")
        assert isinstance(protocol, dict), workload.id
        assert int(protocol.get("warmup_runs", 0)) >= 1, workload.id
        assert int(protocol.get("measured_runs", 0)) >= 3, workload.id
        assert protocol.get("latency_statistics") == ["median", "p90", "p99"], (
            workload.id
        )
        assert protocol.get("timing_scope"), workload.id
        assert (
            workload.raw.get("model_source")
            or workload.raw.get("shared_checkpoint")
            or workload.dataset
        ), workload.id
        model_source = workload.raw.get("model_source")
        if isinstance(model_source, dict):
            assert model_source.get("license"), workload.id
        if workload.raw.get("shared_checkpoint"):
            assert workload.raw.get("quality_dependency"), workload.id
        if workload.dataset:
            assert workload.raw.get("dataset_source"), workload.id
            assert has_asset_dossier(workload.dataset), workload.id


def test_public_inference_contracts_pin_quality_and_provenance():
    workloads = load_registry()
    prefill = workloads["nanogpt-prefill"].raw
    decode = workloads["nanogpt-decode"].raw
    slm = workloads["slm-decode"].raw

    assert prefill["measurement_protocol"]["warmup_runs"] == 3
    assert prefill["measurement_protocol"]["measured_runs"] == 10
    assert prefill["measurement_protocol"]["primary_metric"] == "prefill_tokens_per_sec"
    assert prefill["checkpoint_contract"]["source_workload"] == "nanogpt-train"
    assert decode["measurement_protocol"]["warmup_runs"] == 1
    assert decode["measurement_protocol"]["measured_runs"] == 5
    assert decode["measurement_protocol"]["decode_steps_per_request"] == 64
    assert decode["measurement_protocol"]["primary_metric"] == "output_tokens_per_sec"
    assert decode["checkpoint_contract"]["digest"] == "sha256"

    model_source = slm["model_source"]
    assert model_source["revision"] == "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    quality = slm["quality_evaluation"]
    assert quality["suite"] == "mlperf-edu-slm-quality/0.1"
    assert (
        quality["asset_sha256"]
        == "5fa25872d0b7dc986b12137256b16fd6329267d1640f03e4e04f1dc4e8c8ed5f"
    )
    assert quality["maximum"] == 10.0
    assert slm["measurement_protocol"]["primary_metric"] == "output_tokens_per_sec"
    assert slm["calibration_observation"]["evidence_status"] == (
        "local-calibration-awaiting-committed-review-artifact"
    )


def test_systems_only_rows_do_not_claim_uncommitted_verified_baselines():
    workloads = load_registry()

    for workload in workloads.values():
        if workload.public_status != "systems-only":
            continue
        assert "verified_baseline" not in workload.raw, workload.id
        calibration = workload.raw.get("calibration_observation")
        if calibration:
            assert calibration.get("evidence_status") == (
                "historical-unverified-no-committed-artifact"
            ), workload.id


def test_systems_only_max_paths_declare_their_execution_boundaries():
    workloads = load_registry()
    selected = [
        workload
        for workload in workloads.values()
        if workload.public_status == "systems-only"
    ]

    assert len(selected) == 22
    for workload in selected:
        execution = workload.raw.get("max_execution")
        assert isinstance(execution, dict), workload.id
        assert execution["data_mode"], workload.id
        assert isinstance(execution["quality_target_enforced"], bool), workload.id
        assert isinstance(execution["fetched_assets_used"], bool), workload.id
        assert isinstance(execution["declared_dataset_used"], bool), workload.id
        assert execution["scope"] == "systems-only", workload.id
        assert execution["note"], workload.id

    extended = {
        workload.id
        for workload in selected
        if str((workload.raw.get("runner") or {}).get("max", "")).startswith(
            "mlperf.runners.extended:"
        )
    }
    assert extended == {
        "micro-bert-train",
        "micro-diffusion-train",
        "micro-gnn-train",
        "micro-lstm-train",
        "micro-rl-train",
        "nano-lora-finetune",
        "nano-moe-train",
        "nanogpt-decode-fp16-b16",
        "nanogpt-decode-fp32-b16",
        "nanogpt-decode-spec",
    }
    assert {
        workloads[workload_id].raw["max_execution"]["data_mode"]
        for workload_id in extended
    } == {"synthetic-micro-shard"}


def test_public_contract_report_has_no_blockers():
    issues = public_contract_report(load_registry())

    assert all(not workload_issues for workload_issues in issues.values())


def test_public_asset_dossiers_include_size_and_hash_policy():
    tiny = asset_dossier("tinyshakespeare")
    movielens = asset_dossier("movielens-100k")
    mnist = asset_dossier("mnist")
    cifar = asset_dossier("cifar100")
    fashion = asset_dossier("fashion-mnist")
    prompt = asset_dossier("prompt-suite-local")

    assert tiny["expected_download_bytes"] == 5_600_000
    assert tiny["expected_unpacked_bytes"] == 1_115_394
    assert tiny["hash_policy"]
    assert tiny["license_status"] == "public-domain-us"
    assert tiny["public_release_status"] == "public-ok-fetch-only"
    assert movielens["expected_unpacked_bytes"] == 16_100_896
    assert movielens["license_status"] == "noncommercial-research-education"
    assert movielens["public_release_status"] == "restricted-needs-approval"
    assert mnist["license_status"] == "cc-by-sa-3.0"
    assert mnist["public_release_status"] == "public-ok-with-attribution"
    assert cifar["license_status"] == "source-citation-no-license"
    assert cifar["public_release_status"] == "needs-release-decision"
    assert fashion["license_status"] == "mit"
    assert fashion["license_spdx"] == "MIT"
    assert fashion["public_release_status"] == "public-ok-with-attribution"
    assert prompt["expected_download_bytes"] == 0
    assert prompt["public_result_use"] == "performance-bearing functional check"
    assert prompt["public_release_status"] == "public-ok-bundled"


def test_starter_selection_uses_workload_collection():
    workloads = load_registry()
    selected = select_workloads(workloads, collection="starter")
    selected_ids = {workload.id for workload in selected}

    assert selected_ids == STARTER_WORKLOADS
    assert [workload.id for workload in selected] == list(STARTER_WORKLOAD_ORDER)


def test_standard_selection_uses_workload_collection():
    workloads = load_registry()
    selected = select_workloads(workloads, collection="standard")

    assert {workload.id for workload in selected} == STANDARD_WORKLOADS
    assert [workload.id for workload in selected] == list(STANDARD_WORKLOAD_ORDER)


def test_research_selection_uses_workload_collection():
    workloads = load_registry()
    selected = select_workloads(workloads, collection="research")

    assert {workload.id for workload in selected} == RESEARCH_WORKLOADS
    assert [workload.id for workload in selected] == list(RESEARCH_WORKLOAD_ORDER)


def test_nanogpt_train_declares_min_runner():
    workloads = load_registry()
    runner = workloads["nanogpt-train"].raw.get("runner", {})

    assert runner["min"] == "mlperf.runners.nanogpt:run_min"
    assert runner["max"] == "mlperf.runners.nanogpt:run_max"


def test_nano_moe_declares_min_runner():
    workloads = load_registry()
    runner = workloads["nano-moe-train"].raw.get("runner", {})

    assert workloads["nano-moe-train"].suite == "language"
    assert runner["min"] == "mlperf.runners.extended:run_nano_moe_min"
    assert runner["max"] == "mlperf.runners.extended:run_nano_moe_max"


def test_small_research_training_workloads_declare_min_runners():
    workloads = load_registry()
    expected = {
        "micro-bert-train": ("language", "mlperf.runners.extended:run_micro_bert_min"),
        "micro-diffusion-train": (
            "vision",
            "mlperf.runners.extended:run_micro_diffusion_min",
        ),
        "micro-gnn-train": ("graph", "mlperf.runners.extended:run_micro_gnn_min"),
        "micro-lstm-train": (
            "timeseries",
            "mlperf.runners.extended:run_micro_lstm_min",
        ),
        "micro-rl-train": ("rl", "mlperf.runners.extended:run_micro_rl_min"),
    }

    for workload_id, (suite, runner_name) in expected.items():
        assert workloads[workload_id].suite == suite
        assert workloads[workload_id].raw.get("runner", {})["min"] == runner_name
        assert workloads[workload_id].raw.get("runner", {})[
            "max"
        ] == runner_name.replace("_min", "_max")


def test_language_research_lora_and_serving_workloads_declare_min_runners():
    workloads = load_registry()
    expected = {
        "nano-lora-finetune": "mlperf.runners.extended:run_nano_lora_min",
        "nanogpt-decode-fp32-b16": "mlperf.runners.extended:run_nanogpt_decode_fp32_min",
        "nanogpt-decode-fp16-b16": "mlperf.runners.extended:run_nanogpt_decode_fp16_min",
        "nanogpt-decode-spec": "mlperf.runners.extended:run_nanogpt_decode_spec_min",
    }

    for workload_id, runner_name in expected.items():
        assert workloads[workload_id].suite == "language"
        assert workloads[workload_id].raw.get("runner", {})["min"] == runner_name
        assert workloads[workload_id].raw.get("runner", {})[
            "max"
        ] == runner_name.replace("_min", "_max")


def test_nanogpt_prefill_declares_min_runner():
    workloads = load_registry()
    runner = workloads["nanogpt-prefill"].raw.get("runner", {})

    assert runner["min"] == "mlperf.runners.nanogpt:run_prefill_min"
    assert runner["max"] == "mlperf.runners.nanogpt:run_prefill_max"


def test_nanogpt_decode_declares_min_runner():
    workloads = load_registry()
    runner = workloads["nanogpt-decode"].raw.get("runner", {})

    assert runner["min"] == "mlperf.runners.nanogpt:run_decode_min"
    assert runner["max"] == "mlperf.runners.nanogpt:run_decode_max"


def test_micro_dlrm_declares_min_runner():
    workloads = load_registry()
    runner = workloads["micro-dlrm-train"].raw.get("runner", {})

    assert runner["min"] == "mlperf.runners.dlrm:run_min"
    assert runner["max"] == "mlperf.runners.dlrm:run_max"


def test_recommender_suite_declares_dram_runner():
    workloads = load_registry()
    selected = select_workloads(workloads, suite="recommender")
    dram_runner = workloads["micro-dlrm-dram-train"].raw.get("runner", {})

    assert [workload.id for workload in selected] == [
        "micro-dlrm-train",
        "micro-dlrm-dram-train",
    ]
    assert dram_runner["min"] == "mlperf.runners.dlrm:run_dram_min"
    assert dram_runner["max"] == "mlperf.runners.dlrm:run_dram_max"


def test_distributed_suite_declares_ddp_runner():
    workloads = load_registry()
    selected = select_workloads(workloads, suite="distributed")
    distributed_runner = workloads["micro-dlrm-distributed"].raw.get("runner", {})

    assert [workload.id for workload in selected] == ["micro-dlrm-distributed"]
    assert distributed_runner["min"] == "mlperf.runners.dlrm:run_distributed_min"
    assert distributed_runner["max"] == "mlperf.runners.dlrm:run_distributed_max"


def test_resnet18_declares_min_runner():
    workloads = load_registry()
    runner = workloads["resnet18-train"].raw.get("runner", {})

    assert runner["min"] == "mlperf.runners.vision:run_resnet18_min"
    assert runner["max"] == "mlperf.runners.vision:run_resnet18_max"


def test_vision_suite_declares_mobilenet_min_runners():
    workloads = load_registry()
    selected = select_workloads(workloads, suite="vision")
    mobilenet = workloads["mobilenetv2-train"].raw.get("runner", {})
    composed = workloads["mobilenet-cifar100-composed-fp16"].raw.get("runner", {})

    assert [workload.id for workload in selected] == [
        "resnet18-train",
        "mobilenetv2-train",
        "mobilenet-cifar100-composed-fp16",
        "micro-diffusion-train",
    ]
    assert mobilenet["min"] == "mlperf.runners.vision:run_mobilenetv2_min"
    assert mobilenet["max"] == "mlperf.runners.vision:run_mobilenetv2_max"
    assert composed["min"] == "mlperf.runners.vision:run_mobilenet_composed_min"
    assert composed["max"] == "mlperf.runners.vision:run_mobilenet_composed_max"


def test_anomaly_ae_declares_min_runner():
    workloads = load_registry()
    runner = workloads["anomaly-ae-train"].raw.get("runner", {})

    assert runner["min"] == "mlperf.runners.tiny:run_anomaly_ae_min"
    assert runner["max"] == "mlperf.runners.tiny:run_anomaly_ae_max"


def test_tiny_suite_declares_min_runners():
    workloads = load_registry()
    selected = select_workloads(workloads, suite="tiny")
    dscnn = workloads["dscnn-kws-train"].raw.get("runner", {})
    wake = workloads["wake-vision-vww"].raw.get("runner", {})

    assert [workload.id for workload in selected] == [
        "anomaly-ae-train",
        "dscnn-kws-train",
        "wake-vision-vww",
    ]
    assert dscnn["min"] == "mlperf.runners.tiny:run_dscnn_kws_min"
    assert dscnn["max"] == "mlperf.runners.tiny:run_dscnn_kws_max"
    assert wake["min"] == "mlperf.runners.tiny:run_wake_vision_min"
    assert wake["max"] == "mlperf.runners.tiny:run_wake_vision_max"


def test_agent_suite_declares_min_runners():
    workloads = load_registry()
    selected = select_workloads(workloads, suite="agent")
    codegen = workloads["nano-codegen-agent"].raw.get("runner", {})
    rag = workloads["nano-rag-agent"].raw.get("runner", {})
    react = workloads["nano-react-agent"].raw.get("runner", {})
    toolcall = workloads["nano-toolcall-agent"].raw.get("runner", {})

    assert [workload.id for workload in selected] == [
        "nano-rag-agent",
        "nano-codegen-agent",
        "nano-react-agent",
        "nano-toolcall-agent",
    ]
    assert codegen["min"] == "mlperf.runners.agent:run_codegen_min"
    assert codegen["max"] == "mlperf.runners.agent:run_codegen_max"
    assert rag["min"] == "mlperf.runners.agent:run_rag_min"
    assert rag["max"] == "mlperf.runners.agent:run_rag_max"
    assert react["min"] == "mlperf.runners.agent:run_react_min"
    assert react["max"] == "mlperf.runners.agent:run_react_max"
    assert toolcall["min"] == "mlperf.runners.agent:run_toolcall_min"
    assert toolcall["max"] == "mlperf.runners.agent:run_toolcall_max"


def test_slm_decode_declares_transformers_runner():
    workloads = load_registry()
    selected = select_workloads(workloads, suite="slm")
    runner = workloads["slm-decode"].raw.get("runner", {})
    quantized_runner = workloads["slm-quantized-decode"].raw.get("runner", {})
    batched_runner = workloads["slm-batched-decode"].raw.get("runner", {})
    long_context_runner = workloads["slm-long-context-decode"].raw.get("runner", {})

    assert [workload.id for workload in selected] == [
        "slm-decode",
        "slm-quantized-decode",
        "slm-batched-decode",
        "slm-long-context-decode",
    ]
    assert workloads["slm-decode"].suite == "slm"
    assert workloads["slm-quantized-decode"].suite == "slm"
    assert workloads["slm-batched-decode"].suite == "slm"
    assert workloads["slm-long-context-decode"].suite == "slm"
    assert runner["min"] == "mlperf.runners.slm:run_decode_min"
    assert runner["max"] == "mlperf.runners.slm:run_decode_max"
    assert quantized_runner["min"] == "mlperf.runners.slm:run_quantized_decode_min"
    assert quantized_runner["max"] == "mlperf.runners.slm:run_quantized_decode_max"
    assert batched_runner["min"] == "mlperf.runners.slm:run_batched_decode_min"
    assert batched_runner["max"] == "mlperf.runners.slm:run_batched_decode_max"
    assert (
        long_context_runner["min"] == "mlperf.runners.slm:run_long_context_decode_min"
    )
    assert (
        long_context_runner["max"] == "mlperf.runners.slm:run_long_context_decode_max"
    )


def test_unknown_workload_rejected():
    workloads = load_registry()
    try:
        select_workloads(workloads, workload_id="does-not-exist")
    except ValueError as exc:
        assert "unknown workload" in str(exc)
    else:
        raise AssertionError("unknown workload should fail")
