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


def test_generated_pages_surface_portfolio_selection_reasoning(generated_outputs):
    workload_pages = [
        content
        for path, content in generated_outputs.items()
        if "benchmarks" in path.parts and path.name != "index.qmd"
    ]
    assert len(workload_pages) == 14
    for content in workload_pages:
        assert "## Why This Benchmark Is Included" in content
        assert "**Classroom value**" in content
        assert "**Systems behavior**" in content
        assert "**Benchmark lineage**" in content
        assert "**Quality metric**" in content
        assert "**Alternative rejected**" in content
        assert "**Target kind**" in content

    index = generated_outputs[ROOT / "site" / "benchmarks" / "index.qmd"]
    assert "## Portfolio Design" in index
    assert "## Deliberate Exclusions" in index
    assert "`end-to-end-rag`" in index
    assert "`react-agent`" in index
    assert "`distributed-training`" in index


def test_selection_ledger_exactly_covers_registered_workloads():
    workloads = load_registry(ROOT / "registry")
    selection = generate_docs.load_selection_ledger(workloads)
    entries = selection["workloads"]
    selected = {
        name
        for name, entry in entries.items()
        if entry["status"] in {"admitted", "candidate"}
    }

    assert selected == set(workloads)


def test_generated_workload_pages_send_readers_to_their_own_run(generated_outputs):
    """The site explains and instructs. It is not a results scoreboard."""
    workload_pages = [
        content
        for path, content in generated_outputs.items()
        if "benchmarks" in path.parts and path.name != "index.qmd"
    ]

    assert len(generate_docs.load_provisional_reference_results()) == 9
    assert len(workload_pages) == 14
    for page in workload_pages:
        assert "## Results" in page
        assert "read your own report" in page

    combined = "\n".join(workload_pages)
    # Evidence taxonomy, repeatability statistics, and verdicts are properties
    # of a run and belong to its artifact, not to a page every reader sees.
    for banned in (
        "## Draft Reference Results",
        "## Measured Systems Regime",
        "Five-run verified",
        "One-run provisional",
        "Two-run provisional",
        "do not establish repeatability",
        "CV 5.19%",
    ):
        assert banned not in combined, f"site must not publish {banned!r}"


def test_illustrative_numbers_are_not_presented_as_targets_or_scores(
    generated_outputs,
):
    """A single observation may calibrate the metric; it may not read as a score."""
    page = generated_outputs[
        ROOT / "site" / "benchmarks" / "tiny" / "anomaly-detection.qmd"
    ]

    assert "To make the metric concrete" in page
    assert "not a target and not a score" in page
    assert "your hardware will produce different" in page

    # The superseded 0.292929 gate may appear only as the reviewer note
    # recording why it was withdrawn. It must never reappear as a live target.
    timeseries = generated_outputs[
        ROOT / "site" / "benchmarks" / "timeseries" / "time-series-forecasting.qmd"
    ]
    assert "policy-derived gate was removed" in timeseries
    assert "target ≤ 0.2929" not in timeseries
    assert "0.2929; **pass**" not in timeseries


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


def test_no_workload_page_leads_with_a_preflight_instead_of_a_run():
    """Every workload page can tell the reader to just run it.

    This test used to assert the opposite for the gated set: that pages for
    workloads which could not execute locally led with doctor and a handoff
    rather than a max run. That set is now empty. Recommendation left it when
    its contract moved to NCF on MovieLens-20M, and reinforcement learning left
    it when the PyTorch adapter replaced the MiniGo container. The invariant
    worth holding is the inverse, so a workload that stops running locally
    fails here rather than quietly regrowing a preflight page.
    """
    workloads = load_registry(ROOT / "registry")
    gated = [
        workload_id
        for workload_id, workload in workloads.items()
        if (workload.raw.get("canonical_max_contract") or {}).get("execution_status")
        == "environment-gated-quality-conformance"
    ]
    assert not gated, f"these workloads no longer run locally: {gated}"

    for workload_id, workload in workloads.items():
        section = generate_docs.section_how_to_run(workload)
        assert "## Current Preflight and Handoff" not in section, workload_id
        # Mode-bearing workloads spell the command as
        # `run --workload X --mode training --profile max`, so match the two
        # halves rather than one fixed string.
        assert f"run --workload {workload_id}" in section, workload_id
        assert "--profile max" in section, workload_id


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
