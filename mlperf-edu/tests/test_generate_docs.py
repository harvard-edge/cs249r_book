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

    assert counts == {"experimental": 14}
    assert "**14 workloads**" in stats
    assert "14 experimental" in stats


def test_tinyshakespeare_catalog_uses_pinned_upstream_corpus():
    catalog = yaml.safe_load((ROOT / "datasets.yaml").read_text())["datasets"]
    source = catalog["tinyshakespeare"]["uri"]

    assert "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e" in source
    assert "karpathy/char-rnn" in source.lower()


def test_dataset_page_surfaces_release_boundaries_and_dossier_status():
    workloads = load_registry(ROOT / "registry")
    page = generate_docs.datasets_page(workloads)

    assert "not an official MLCommons benchmark" in page
    assert "public-ok-fetch-only" in page
    assert "mit-repository-public-domain-text" in page
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


def test_generated_workload_pages_render_structured_provenance(generated_outputs):
    workload_pages = [
        content
        for path, content in generated_outputs.items()
        if "benchmarks" in path.parts and path.name != "index.qmd"
    ]
    assert workload_pages
    for content in workload_pages:
        assert "## Authoritative Sources and Adaptation" in content
        assert "**Adaptation boundary:**" in content
        assert "{'authority':" not in content
        assert "[open source](https://" in content


def test_generated_workload_pages_disclose_indexed_reference_evidence(
    generated_outputs,
):
    workload_pages = [
        content
        for path, content in generated_outputs.items()
        if "benchmarks" in path.parts and path.name != "index.qmd"
    ]

    assert len(generate_docs.load_provisional_reference_results()) == 9
    assert sum("## Draft Reference Results" in page for page in workload_pages) == 9
    assert sum("## Draft Reference Results" not in page for page in workload_pages) == 5
    combined = "\n".join(workload_pages)
    assert "Five-run verified" in combined
    assert "One-run provisional" in combined
    assert "Two-run provisional" in combined
    assert "None are MLCommons-verified results" in combined
    assert "do not establish repeatability" in combined
    assert "CV 5.19%; **diagnostic fail**" in combined


def test_quality_conformance_pages_disclose_result_boundary(generated_outputs):
    workload_pages = [
        content
        for path, content in generated_outputs.items()
        if "benchmarks" in path.parts and path.name != "index.qmd"
    ]

    assert sum("## Readiness Spiral Status" in page for page in workload_pages) == 5
    combined = "\n".join(workload_pages)
    assert "The `max` runner implements the authoritative quality contract" in combined
    assert "The `min` path remains a functional probe" in combined
    assert "No draft reference result or public baseline is claimed" in combined


def test_canonical_pages_do_not_expose_retired_synthetic_max_boundaries():
    workloads = load_registry(ROOT / "registry")
    systems_only = [
        workload
        for workload in workloads.values()
        if workload.public_status == "systems-only"
    ]
    assert systems_only == []
    assert all("max_execution" not in workload.raw for workload in workloads.values())


def test_consolidated_language_page_runs_training_before_inference():
    workloads = load_registry(ROOT / "registry")
    section = generate_docs.section_how_to_run(workloads["causal-language-modeling"])
    training = "run --workload causal-language-modeling --mode training --profile max"
    prefill = "run --workload causal-language-modeling --mode inference --phase prefill"
    decode = "run --workload causal-language-modeling --mode inference --phase decode"
    assert training in section
    assert prefill in section
    assert decode in section
    assert section.index(training) < section.index(prefill)
    assert section.count('--output-dir "$OUTPUT_DIR"') >= 4
    assert "nanogpt-prefill" not in section


def test_site_install_commands_use_the_source_checkout(generated_outputs):
    authored = [
        ROOT / "site" / "index.qmd",
        ROOT / "site" / "getting-started.qmd",
        ROOT / "site" / "readiness.qmd",
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
    assert (
        "run --workload causal-language-modeling --mode training --profile max"
        in sut_page
    )
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


def test_authored_site_avoids_retired_fixed_portfolio_counts():
    authored = [
        ROOT / "site" / "index.qmd",
        ROOT / "site" / "getting-started.qmd",
        ROOT / "site" / "readiness.qmd",
        ROOT / "site" / "about.qmd",
        *(ROOT / "site" / "guide").glob("*.qmd"),
    ]
    content = "\n".join(path.read_text().lower() for path in authored)

    retired_claims = (
        "seven workloads",
        "all seven",
        "ten-case",
        "ten evidence",
        "ten paths",
    )
    assert all(claim not in content for claim in retired_claims)


def path_is_dataset(content: str) -> bool:
    return 'title: "Dataset Catalog"' in content
