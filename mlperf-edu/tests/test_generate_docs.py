from __future__ import annotations

from collections import Counter
import re
from pathlib import Path

import pytest
import yaml

from mlperf.registry import load_registry
from tools import generate_docs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def generated_outputs() -> dict[Path, str]:
    return generate_docs.build_outputs()


def test_dataset_catalog_exactly_covers_registry_datasets():
    workloads = load_registry(ROOT / "registry")
    usage = generate_docs.dataset_usage(workloads)
    catalog = generate_docs.load_dataset_catalog()

    generate_docs.validate_dataset_catalog(catalog, usage)
    assert set(catalog) == set(usage)
    assert len(catalog) == 14


def test_generated_stats_report_exact_candidate_status_counts(generated_outputs):
    workloads = load_registry(ROOT / "registry")
    counts = Counter(workload.public_status for workload in workloads.values())
    stats = generated_outputs[ROOT / "site" / "_stats.qmd"]

    assert counts == {
        "score-bearing": 5,
        "performance-bearing": 3,
        "systems-only": 22,
    }
    assert "**30 workloads**" in stats
    assert "3 performance-bearing" in stats
    assert "5 score-bearing" in stats
    assert "22 systems-only" in stats


def test_tinyshakespeare_catalog_uses_project_gutenberg():
    catalog = yaml.safe_load((ROOT / "datasets.yaml").read_text())["datasets"]
    source = catalog["tinyshakespeare"]["uri"]

    assert source == "https://www.gutenberg.org/files/100/100-0.txt"
    assert "karpathy" not in source.lower()


def test_dataset_page_surfaces_release_boundaries_and_dossier_status():
    workloads = load_registry(ROOT / "registry")
    page = generate_docs.datasets_page(workloads)

    assert "not an official MLCommons benchmark" in page
    assert "restricted-needs-approval" in page
    assert "public-ok-fetch-only" in page
    assert "Project Gutenberg" in page
    for dataset in generate_docs.dataset_usage(workloads):
        assert f"`{dataset}`" in page


def test_generated_benchmark_pages_disclose_candidate_status(generated_outputs):
    visitor_pages = [
        content
        for path, content in generated_outputs.items()
        if "benchmarks" in path.parts or path.name == "datasets.qmd"
    ]

    assert visitor_pages
    for content in visitor_pages:
        assert "not an official MLCommons benchmark" in content
        assert "Candidate result status" in content or path_is_dataset(content)


def test_family_pages_use_default_variant_and_disclose_mixed_metadata(
    generated_outputs,
):
    workloads = load_registry(ROOT / "registry")
    members = generate_docs.group_families(workloads)["nanogpt-inference"]

    lead = generate_docs.family_lead(members)
    assert lead.id == "nanogpt-decode"

    index_row = generate_docs.family_row("nanogpt-inference", members, depth=1)
    assert "nanogpt-12m (default; variants differ)" in index_row
    assert "11.5M (default; variants differ)" in index_row
    assert "nanogpt-small-86m" not in index_row
    assert (
        "performance-bearing (default); other variants include systems-only"
        in index_row
    )

    page = generated_outputs[
        ROOT / "site" / "benchmarks" / "language" / "nanogpt-inference.qmd"
    ]
    assert "Every variant shares the family's model and dataset contract" not in page
    assert (
        "| **Variant** | **Workload ID** | **Model** | **Params** | **Dataset** "
        "| **Scenario** | **Candidate result status** |"
    ) in page
    assert (
        "| [fp32-b16](#variant-fp32-b16) | `nanogpt-decode-fp32-b16` "
        "| nanogpt-small-86m | 88.3M | prompt-suite-local | server | systems-only |"
    ) in page
    assert (
        "| [decode](#variant-decode) | `nanogpt-decode` | nanogpt-12m "
        "| 11.5M | prompt-suite-local | server | performance-bearing |"
    ) in page


def test_benchmark_index_tables_never_emit_empty_cells(generated_outputs):
    benchmark_root = ROOT / "site" / "benchmarks"
    index_pages = {
        path: content
        for path, content in generated_outputs.items()
        if path.name == "index.qmd" and path.is_relative_to(benchmark_root)
    }

    assert index_pages
    for path, content in index_pages.items():
        for line in content.splitlines():
            if not line.startswith("|"):
                continue
            cells = line.split("|")[1:-1]
            assert cells
            assert all(cell.strip() for cell in cells), f"{path}: {line}"

    language_index = index_pages[benchmark_root / "language" / "index.qmd"]
    lora_row = next(
        line for line in language_index.splitlines() if "[`nano-lora-finetune`]" in line
    )
    assert "| — | — |" in lora_row


def test_public_candidate_pages_disclose_committed_reference_evidence(
    generated_outputs,
):
    workloads = load_registry(ROOT / "registry")

    for workload in workloads.values():
        if workload.public_status not in {"score-bearing", "performance-bearing"}:
            continue
        baseline = workload.raw.get("verified_baseline") or {}
        assert baseline.get("review_eligible") is True
        assert baseline.get("evidence_status") == "committed-reference-summary"
        assert baseline.get("evidence_tier") == "public-candidate"
        assert baseline.get("evidence_file", "").startswith("reference_results/")
        assert len(baseline.get("evidence_sha256", "")) == 64
        assert baseline.get("reference_package_availability") == "local-handoff"
        assert baseline.get("external_publication_status") == "pending"
        assert baseline.get("seeds") == [0, 1, 2, 3, 4]
        assert len(baseline.get("metric_values_by_seed") or []) == 5

        family = workload.canonical_workload or workload.id
        page = generated_outputs[
            ROOT / "site" / "benchmarks" / workload.suite / f"{family}.qmd"
        ]
        assert "| **Evidence status** | committed-reference-summary |" in page
        assert "| **Review eligible** | True |" in page
        assert "| **Evidence file** | reference_results/" in page
        assert "Recorded Project Reference Baseline" in page
        assert "not an MLCommons-verified result" in page
        if workload.public_status == "score-bearing":
            assert workload.scenario == "training"
            assert "not an official MLPerf Inference scenario" in page


def test_performance_bearing_pages_disclose_both_protocol_layers(generated_outputs):
    workloads = load_registry(ROOT / "registry")

    for workload in workloads.values():
        if workload.public_status != "performance-bearing":
            continue
        assert workload.scenario in {"single_stream", "offline", "server"}
        reference = workload.raw.get("performance_reference_protocol") or {}
        measurement = workload.raw.get("measurement_protocol") or {}
        assert reference.get("reference_runs") == 5
        assert reference.get("seeds") == [0, 1, 2, 3, 4]
        assert int(measurement.get("warmup_runs", 0)) >= 1
        assert int(measurement.get("measured_runs", 0)) >= 3

        family = workload.canonical_workload or workload.id
        page = generated_outputs[
            ROOT / "site" / "benchmarks" / workload.suite / f"{family}.qmd"
        ]
        assert "Candidate Performance Contract" in page
        assert "**Functional acceptance:**" in page
        assert "**Five-seed reference protocol:**" in page
        assert "**Within-run measurement protocol:**" in page


def test_systems_only_pages_disclose_the_current_max_execution_boundary():
    workloads = load_registry(ROOT / "registry")
    systems_only = [
        workload
        for workload in workloads.values()
        if workload.public_status == "systems-only"
    ]
    source_paths = generate_docs.index_source_paths()

    assert len(systems_only) == 22
    for workload in systems_only:
        execution = workload.raw["max_execution"]
        body = generate_docs.render_workload_body(
            workload, source_paths, heading_shift=False
        )
        assert "Current `max` execution boundary" in body, workload.id
        assert f"Reported data mode: `{execution['data_mode']}`" in body, workload.id
        expected_quality = (
            "yes" if execution["quality_target_enforced"] is True else "no"
        )
        assert f"Candidate quality target enforced: **{expected_quality}**" in body, (
            workload.id
        )
        if execution["fetched_assets_used"] is False:
            assert "no asset fetch is required" in body, workload.id
        if workload.quality_metric and not execution["quality_target_enforced"]:
            assert "Not enforced by the current `max` runner" in body, workload.id


def test_shared_checkpoint_pages_run_training_before_inference():
    workloads = load_registry(ROOT / "registry")
    for workload_id in ("nanogpt-prefill", "nanogpt-decode"):
        section = generate_docs.section_how_to_run(workloads[workload_id])
        training = "run --workload nanogpt-train --profile max"
        inference = "run --workload nanogpt-inference"
        assert training in section
        assert inference in section
        assert section.index(training) < section.index(inference)
        assert section.count('--output-dir "$OUTPUT_DIR"') >= 2
        assert "fetch --workload nanogpt-inference" not in section


def test_site_install_commands_use_the_source_checkout(generated_outputs):
    authored = [
        ROOT / "site" / "index.qmd",
        ROOT / "site" / "getting-started.qmd",
        *(ROOT / "site" / "guide").glob("*.qmd"),
    ]
    content = "\n".join(path.read_text() for path in authored)
    content += "\n" + "\n".join(
        generated
        for path, generated in generated_outputs.items()
        if path.name != "cli.qmd"
    )

    assert "uv tool install mlperf-edu" not in content
    commands = (
        r"doctor|init|list|show|info|fetch|run|verify|report|package|audit|"
        r"validate|grade|cache"
    )
    assert not re.search(rf"(?m)^mlperf (?:{commands})\b", content)
    assert "uv sync --locked --extra dev" in content


def test_site_describes_the_actual_lab_and_sut_scope(generated_outputs):
    sut_page = (ROOT / "site" / "guide" / "sut-plugins.qmd").read_text()
    about = (ROOT / "site" / "about.qmd").read_text()
    instructors = (ROOT / "site" / "guide" / "instructors.qmd").read_text()

    assert "Lab 2 is the only shipped lab" in sut_page
    assert "does not expose a generic `--sut` option" in sut_page
    assert "Lab 1 | `lab1_optimization.py` | Standalone" in sut_page
    assert "Lab 3 | `lab3_arch_comparison.py` | Standalone" in sut_page
    assert "run --workload nanogpt-train --profile max" in sut_page
    assert (
        "Fetching assets alone does\nnot create a quality-approved checkpoint"
        in sut_page
    )
    assert "only [Lab 2]" in about
    assert "product CLI has no generic\n`--sut` plugin-loading option" in instructors
    assert "--sut" not in generated_outputs[ROOT / "site" / "reference" / "cli.qmd"]

    for content in (about, instructors):
        assert "implement an optimization as a" not in content
        assert "Optimizations live in isolated" not in content


def path_is_dataset(content: str) -> bool:
    return 'title: "Dataset Catalog"' in content
