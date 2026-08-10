#!/usr/bin/env python3
"""Generate the MLPerf EDU documentation site from the workload registry.

Single source of truth
----------------------
Generated benchmark and reference facts come from these project contracts:

  * ``registry/suites/**``      - per-workload / per-variant benchmark metadata
  * ``registry/suites.yaml``    - suite-level titles and summaries
  * ``registry/selection-ledger.yaml`` - portfolio admission and rationale
  * ``datasets.yaml``           - dataset catalog
  * ``src/mlperf/assets.py``    - structured public-asset dossiers
  * ``provisional_results/``    - verified draft reference-result records
  * the live ``mlperf`` CLI     - ``--help`` text for the command reference

Nothing on these pages is hand-written. To change a page, change the
registry (or the CLI help text) and regenerate:

    python3 tools/generate_docs.py            # rewrite generated pages
    python3 tools/generate_docs.py --check    # CI drift gate (no writes)

Generated outputs (the generator owns these paths entirely):

    site/benchmarks/**            one page per workload family + indexes
    site/reference/cli.qmd        CLI reference from --help output
    site/reference/datasets.qmd   dataset catalog + usage map
    site/_stats.qmd               include partial (landing-page stats)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from mlperf.assets import asset_dossier, has_asset_dossier  # noqa: E402
from mlperf.registry import (  # noqa: E402
    Workload,
    baseline_is_current_review_evidence,
    baseline_is_protocol_superseded,
    load_registry,
    quality_target_satisfied,
)

GITHUB_BLOB = "https://github.com/harvard-edge/cs249r_book/blob/main/mlperf-edu"
GITHUB_TREE = "https://github.com/harvard-edge/cs249r_book/tree/main/mlperf-edu"
CHECKOUT_COMMAND = "uv run mlperf"

GENERATED_NOTE = (
    "<!-- GENERATED FILE - do not edit by hand.\n"
    "     Sources: registry/ + selection ledger + datasets.yaml + asset dossiers + "
    "draft reference results + the mlperf CLI.\n"
    "     Regenerate with: python3 tools/generate_docs.py -->\n"
)

PREVIEW_CALLOUT = (
    "::: {.callout-warning}\n"
    "**Independent preview.** MLPerf EDU is not an official MLCommons benchmark "
    "and is not endorsed by MLCommons. Registry result labels are candidate "
    "classifications for review, not accepted MLPerf result categories.\n"
    ":::\n"
)

CLI_COMMANDS = [
    "doctor",
    "init",
    "list",
    "show",
    "info",
    "fetch",
    "run",
    "verify",
    "report",
    "package",
    "audit",
    "validate",
    "grade",
    "cache",
]


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------


def esc(value: Any) -> str:
    """Escape a scalar for use inside a markdown pipe table cell."""
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|")


def table_cell(value: Any) -> str:
    """Render a nonempty markdown table cell."""
    if value is None:
        return "—"
    rendered = esc(value)
    return rendered if rendered else "—"


def badge(status: str) -> str:
    slug = status.replace(" ", "-")
    return f'<span class="badge status-{slug}">{status}</span>'


def kv_table(rows: list[tuple[str, Any]]) -> str:
    """Render a two-column field table, skipping empty values."""
    kept = [(k, v) for k, v in rows if v not in (None, "", [], {})]
    if not kept:
        return ""
    out = ["| **Field** | **Value** |", "|:---|:---|"]
    for key, value in kept:
        out.append(f"| **{esc(key)}** | {esc(value)} |")
    return "\n".join(out) + "\n"


def bullet_block(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    lines = [f"**{title}:**", ""]
    lines += [f"- {esc(item)}" for item in items]
    return "\n".join(lines) + "\n"


def mapping_table(mapping: Any) -> str:
    """Render a registry mapping as a readable two-column table."""
    if not isinstance(mapping, dict) or not mapping:
        return ""
    rows: list[tuple[str, Any]] = []
    for key, value in mapping.items():
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        rows.append((key.replace("_", " ").capitalize(), value))
    return kv_table(rows)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_provisional_reference_results() -> dict[str, list[dict[str, Any]]]:
    """Load and verify the draft reference-result index used by the site."""
    result_root = ROOT / "provisional_results"
    index_path = result_root / "index.json"
    index = json.loads(index_path.read_text())
    if index.get("schema") != "mlperf-edu-provisional-reference-index/0.1":
        raise ValueError("unsupported provisional reference-result index schema")

    cases = index.get("cases")
    if not isinstance(cases, list) or len(cases) != index.get("case_count"):
        raise ValueError("provisional reference-result case count is inconsistent")

    by_workload: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for entry in cases:
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative.startswith(
            "provisional_results/"
        ):
            raise ValueError("invalid provisional reference-result path")
        path = (ROOT / relative).resolve()
        if result_root.resolve() not in path.parents:
            raise ValueError("provisional reference-result path escapes its root")
        if _sha256(path) != entry.get("sha256"):
            raise ValueError(
                f"provisional reference-result digest mismatch: {relative}"
            )

        record = json.loads(path.read_text())
        case_id = record.get("case_id")
        if (
            record.get("schema") != "mlperf-edu-provisional-reference-result/0.1"
            or case_id != entry.get("case_id")
            or record.get("workload") != entry.get("workload")
            or case_id in seen
        ):
            raise ValueError(f"invalid provisional reference-result record: {relative}")
        seen.add(case_id)
        by_workload.setdefault(record["workload"], []).append(record)

    if len(by_workload) != index.get("workload_count"):
        raise ValueError("provisional reference-result workload count is inconsistent")
    return by_workload


def provenance_value(key: str, value: Any) -> Any:
    """Render trusted registry provenance without exposing Python containers."""
    if not isinstance(value, str):
        return value
    if value.startswith(("https://", "http://")):
        return f"[open source]({value})"
    if key.endswith(("commit", "revision")):
        return f"`{esc(value)}`"
    if key == "path":
        return f"`{esc(value)}`"
    return value


def workload_href(suite: str, family: str, depth: int) -> str:
    prefix = "../" * depth
    return f"{prefix}benchmarks/{suite}/{family}.qmd"


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def load_suite_meta() -> dict[str, dict[str, str]]:
    path = ROOT / "registry" / "suites.yaml"
    with path.open("r") as handle:
        data = yaml.safe_load(handle)
    suites = data.get("suites", {})
    if not isinstance(suites, dict):
        raise ValueError("registry/suites.yaml must define a 'suites' mapping")
    return suites


def load_selection_ledger(
    workloads: dict[str, Workload],
) -> dict[str, Any]:
    """Load the portfolio decision record and require exact registry coverage."""
    path = ROOT / "registry" / "selection-ledger.yaml"
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if data.get("schema") != "mlperf-edu-workload-selection/0.1":
        raise ValueError("unsupported selection-ledger schema")
    entries = data.get("workloads")
    if not isinstance(entries, dict):
        raise ValueError("selection ledger must define a 'workloads' mapping")

    selected = {
        name
        for name, entry in entries.items()
        if isinstance(entry, dict) and entry.get("status") in {"admitted", "candidate"}
    }
    registered = set(workloads)
    if selected != registered:
        missing = sorted(registered - selected)
        stale = sorted(selected - registered)
        raise ValueError(
            "selection ledger and workload registry disagree: "
            f"missing={missing}, stale={stale}"
        )
    return data


def index_source_paths() -> dict[str, str]:
    """Map workload id -> registry source path (repo-relative)."""
    paths: dict[str, str] = {}
    suites_dir = ROOT / "registry" / "suites"
    for suite_dir in sorted(p for p in suites_dir.iterdir() if p.is_dir()):
        for item in sorted(suite_dir.iterdir()):
            if item.is_file() and item.suffix in {".yaml", ".yml"}:
                raw = yaml.safe_load(item.read_text()) or {}
                workload_id = str(raw.get("id") or item.stem)
                paths[workload_id] = str(item.relative_to(ROOT))
            elif item.is_dir():
                base_path = item / "workload.yaml"
                base = (
                    yaml.safe_load(base_path.read_text()) if base_path.is_file() else {}
                )
                canonical = str((base or {}).get("id") or item.name)
                variants_dir = item / "variants"
                if variants_dir.is_dir():
                    for variant_path in sorted(variants_dir.glob("*.y*ml")):
                        raw = yaml.safe_load(variant_path.read_text()) or {}
                        variant_name = str(raw.get("variant") or variant_path.stem)
                        workload_id = str(
                            raw.get("id") or f"{canonical}-{variant_name}"
                        )
                        paths[workload_id] = str(variant_path.relative_to(ROOT))
                else:
                    paths[canonical] = str(base_path.relative_to(ROOT))
    return paths


def group_families(
    workloads: dict[str, Workload],
) -> "OrderedDict[str, list[Workload]]":
    """Group workloads into families keyed by canonical workload id."""
    families: OrderedDict[str, list[Workload]] = OrderedDict()
    for workload in workloads.values():
        family = workload.canonical_workload or workload.id
        families.setdefault(family, []).append(workload)
    return families


def family_lead(members: list[Workload]) -> Workload:
    """Return the declared default variant, falling back to the first member."""
    if not members:
        raise ValueError("workload family must contain at least one member")
    return next((member for member in members if member.default_variant), members[0])


def family_metadata_cell(lead: Workload, value: Any, values: list[Any]) -> str:
    """Render lead metadata and disclose when variants carry different values."""
    normalized = {
        "" if candidate is None else str(candidate).strip() for candidate in values
    }
    rendered = table_cell(value)
    if len(normalized) <= 1:
        return rendered
    role = "default" if lead.default_variant else "lead"
    return f"{rendered} ({role}; variants differ)"


def family_status_cell(members: list[Workload], lead: Workload) -> str:
    """Lead with the default status while retaining mixed-family disclosure."""
    lead_status = table_cell(lead.public_status)
    other_statuses = sorted(
        {
            member.public_status
            for member in members
            if member.public_status and member.public_status != lead.public_status
        }
    )
    if not other_statuses:
        return lead_status
    role = "default" if lead.default_variant else "lead"
    return (
        f"{lead_status} ({role}); other variants include "
        f"{', '.join(table_cell(status) for status in other_statuses)}"
    )


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def section_at_a_glance(w: Workload) -> str:
    raw = w.raw
    rows = [
        ("Suite", w.suite),
        ("Model", w.model),
        ("Parameters", raw.get("params")),
        ("Dataset", w.dataset),
        ("Dataset source", raw.get("dataset_source")),
        ("Scenario", w.scenario),
        ("Maturity", w.maturity),
        ("Candidate result status", w.public_status),
    ]
    body = kv_table(rows)
    note = raw.get("params_note")
    if note:
        body += f"\n> **Parameter count note:** {esc(note)}\n"
    if w.scenario == "training":
        body += (
            "\n> **Scenario note:** `training` is a proposed MLPerf EDU label "
            "for train-then-quality workloads. It is not an official MLPerf "
            "Inference scenario.\n"
        )
    return f"## At a Glance\n\n{body}"


def section_spiral_status(w: Workload) -> str:
    spiral = w.raw.get("spiral")
    if not isinstance(spiral, dict) or not spiral:
        return ""
    body = mapping_table(
        {
            "stage": spiral.get("stage"),
            "functional_ready": spiral.get("functional_ready"),
            "quality_conformant": spiral.get("quality_conformant"),
            "repeatability_verified": spiral.get("repeatability_verified"),
            "promotion_ready": spiral.get("promotion_ready"),
            "next_gate": spiral.get("next_gate"),
        }
    )
    if spiral.get("stage") == "functional":
        boundary = (
            "**Functional-stage boundary.** The current `min` and `max` runners are "
            "bounded integration probes. They validate execution, reporting, and "
            "provenance, but they do not run the complete authoritative quality "
            "contract and must not be used as benchmark baselines."
        )
    else:
        boundary = (
            "**Quality-conformance boundary.** The `max` runner implements the "
            "authoritative quality contract. A result remains experimental until a "
            "complete run meets the unchanged target and its provenance verifies. "
            "The `min` path remains a functional probe."
        )
    body += f"\n::: {{.callout-caution}}\n{boundary}\n:::"
    return f"## Readiness Spiral Status\n\n{body}\n"


def section_selection_rationale(w: Workload, selection: dict[str, Any]) -> str:
    """Explain why a workload belongs in the portfolio."""
    entry = (selection.get("workloads") or {}).get(w.id)
    if not isinstance(entry, dict):
        raise ValueError(f"selection ledger is missing workload '{w.id}'")
    rationale = entry.get("rationale")
    upstream = entry.get("upstream")
    if not isinstance(rationale, dict) or not isinstance(upstream, dict):
        raise ValueError(f"selection rationale is incomplete for '{w.id}'")

    body = esc(rationale.get("task_significance", "")) + "\n\n"
    body += kv_table(
        [
            ("Classroom value", rationale.get("classroom_value")),
            ("Systems behavior", rationale.get("systems_behavior")),
            ("Benchmark lineage", rationale.get("benchmark_lineage")),
            ("Model choice", rationale.get("reason_for_model")),
            ("Dataset choice", rationale.get("reason_for_dataset")),
            ("Quality metric", rationale.get("reason_for_metric")),
            ("Published baseline", upstream.get("published_baseline")),
            ("Alternative rejected", rationale.get("alternatives_rejected")),
        ]
    )
    return f"## Why This Benchmark Is Included\n\n{body}"


def section_provenance(w: Workload) -> str:
    provenance = w.raw.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        return ""
    adaptation = provenance.get("adaptation")
    rows = [
        (
            key.replace("_", " ").capitalize(),
            provenance_value(key, value),
        )
        for key, value in provenance.items()
        if key != "adaptation"
    ]
    body = kv_table(rows)
    if adaptation:
        body += f"\n**Adaptation boundary:** {esc(adaptation)}\n"
    return f"## Authoritative Sources and Adaptation\n\n{body}"


def section_execution_boundary(w: Workload) -> str:
    execution = w.raw.get("max_execution")
    if not isinstance(execution, dict) or not execution:
        return ""

    quality = "yes" if execution.get("quality_target_enforced") is True else "no"
    fetched = "yes" if execution.get("fetched_assets_used") is True else "no"
    declared = "yes" if execution.get("declared_dataset_used") is True else "no"
    return "\n".join(
        [
            "::: {.callout-caution}",
            "**Current `max` execution boundary.** " + esc(execution.get("note", "")),
            "",
            f"- Reported data mode: `{esc(execution.get('data_mode'))}`",
            f"- Candidate quality target enforced: **{quality}**",
            f"- Fetched assets used by this runner: **{fetched}**",
            f"- Declared dataset used by this runner: **{declared}**",
            "",
            "This path is systems-only. Its measurements are not a public score or "
            "performance baseline unless the workload is promoted through a reviewed "
            "result contract.",
            ":::",
        ]
    )


def section_how_to_run(w: Workload) -> str:
    target = (
        f"--workload {w.canonical_workload} --variant {w.variant}"
        if w.variant
        else f"--workload {w.id}"
    )
    canonical = w.raw.get("canonical_max_contract") or {}
    if canonical.get("execution_status") == "environment-gated-quality-conformance":
        next_gate = ((w.raw.get("spiral") or {}).get("next_gate") or "").strip()
        lines = [
            "## Current Preflight and Handoff",
            "",
            "::: {.callout-warning}",
            "**The quality path is not a general local run yet.** " + esc(next_gate),
            "The `max` and `pro` runners fail closed until the declared assets, "
            "hardware, and runtime are available. Use `doctor` to inspect the "
            "machine-readable environment handoff.",
            ":::",
            "",
            "```bash",
            "# inspect the authoritative environment handoff",
            f"{CHECKOUT_COMMAND} doctor {target} --profile max --format json",
            "",
            "# preview asset requirements without downloading",
            f"{CHECKOUT_COMMAND} fetch {target} --profile max --dry-run",
            "",
            "# the bounded functional path runs on supported local hardware",
            f"{CHECKOUT_COMMAND} run {target} --profile min",
            "```",
            "",
            "A successful `min` result verifies the local plumbing only. It does not "
            "evaluate the authoritative quality target. See the "
            "[running guide](../../guide/running.qmd) for the current local-execution "
            "boundary and recovery status.",
        ]
        return "\n".join(lines) + "\n"

    lines = ["## How to Run", "", "```bash"]
    shared_checkpoint = w.raw.get("shared_checkpoint")
    max_execution = w.raw.get("max_execution") or {}
    implemented_modes = set(w.raw.get("implemented_modes") or [])
    inference_phases = (w.raw.get("phases") or {}).get("inference") or []
    consolidated_training_inference = {"training", "inference"}.issubset(
        implemented_modes
    )
    if consolidated_training_inference:
        lines.append("# prepare and quality-check the training checkpoint")
        lines.append(f'OUTPUT_DIR="submissions/{w.id}-max"')
        lines.append(f"{CHECKOUT_COMMAND} fetch {target} --profile max")
        lines.append(
            f'{CHECKOUT_COMMAND} run {target} --mode training --profile max --output-dir "$OUTPUT_DIR"'
        )
        for phase in inference_phases:
            lines.append(
                f'{CHECKOUT_COMMAND} run {target} --mode inference --phase {phase} --profile max --output-dir "$OUTPUT_DIR"'
            )
    elif shared_checkpoint:
        lines.append("# prepare and quality-check the shared training checkpoint")
        lines.append('OUTPUT_DIR="submissions/nanogpt-inference-max"')
        lines.append(
            f"{CHECKOUT_COMMAND} fetch --workload {shared_checkpoint} --profile max"
        )
        lines.append(
            f'{CHECKOUT_COMMAND} run --workload {shared_checkpoint} --profile max --output-dir "$OUTPUT_DIR"'
        )
        lines.append("")
        lines.append(
            "# checkpoint-backed benchmark run (reuses the same output directory)"
        )
        lines.append(
            f'{CHECKOUT_COMMAND} run {target} --profile max --output-dir "$OUTPUT_DIR"'
        )
    else:
        if max_execution.get("fetched_assets_used") is False:
            lines.append("# no asset fetch is required by this workload contract")
        else:
            lines.append("# one-time asset preparation")
            lines.append(f"{CHECKOUT_COMMAND} fetch {target} --profile max")
        lines.append("")
        lines.append(
            "# benchmark run (writes JSON/HTML/CSV reports + .provd.json provenance)"
        )
        lines.append(f"{CHECKOUT_COMMAND} run {target} --profile max")
    lines.append("")
    lines.append("# quick smoke pass")
    lines.append(f"{CHECKOUT_COMMAND} run {target} --profile min")
    if consolidated_training_inference:
        for phase in inference_phases:
            lines.append(
                f"{CHECKOUT_COMMAND} run {target} --mode inference --phase {phase} --profile min"
            )
    lines.append("")
    lines.append("# research envelope")
    pro_output = (
        ' --output-dir "$OUTPUT_DIR"'
        if shared_checkpoint or consolidated_training_inference
        else ""
    )
    lines.append(f"{CHECKOUT_COMMAND} run {target} --profile pro{pro_output}")
    lines.append("```")
    lines.append("")
    lines.append(
        "See the [running guide](../../guide/running.qmd) for profile semantics, "
        "report handling, and power measurement flags."
    )
    return "\n".join(lines) + "\n"


def section_quality_target(w: Workload) -> str:
    if not w.quality_metric:
        return ""
    rows = [
        ("Metric", w.quality_metric),
        ("Target", w.quality_value),
        ("Direction", w.quality_direction),
        (
            "Target kind",
            str(w.quality_target_kind or "").replace("_", " "),
        ),
        ("Target basis", w.quality_target_basis),
        ("Tolerance", w.quality_tolerance),
        ("Reference runs", w.quality_reference_runs),
    ]
    body = kv_table(rows)
    max_execution = w.raw.get("max_execution") or {}
    if max_execution.get("quality_target_enforced") is False:
        body += (
            "\n> **Not enforced by the current `max` runner.** This target is "
            "research context for future promotion; the current path remains systems-only.\n"
        )

    variance = w.quality_variance_summary
    if isinstance(variance, dict) and variance:
        body += "\n**Variance summary:**\n\n" + kv_table(
            [(k.replace("_", " ").capitalize(), v) for k, v in variance.items()]
        )

    protocol = w.quality_reference_protocol
    if isinstance(protocol, dict) and protocol:
        rows = []
        for key, value in protocol.items():
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            rows.append((key.replace("_", " ").capitalize(), value))
        body += "\n**Reference protocol:**\n\n" + kv_table(rows)

    if w.quality_reviewer_notes:
        body += "\n" + bullet_block("Reviewer notes", list(w.quality_reviewer_notes))
    return f"## Candidate Quality Target\n\n{body}"


def section_performance_contract(w: Workload) -> str:
    raw = w.raw
    functional = raw.get("functional_check")
    reference = raw.get("performance_reference_protocol")
    measurement = raw.get("measurement_protocol")
    checkpoint = raw.get("checkpoint_contract")
    quality = raw.get("quality_evaluation")
    if not any(
        isinstance(item, dict) and item
        for item in (functional, reference, measurement, checkpoint, quality)
    ):
        return ""

    title = (
        "Candidate Performance Contract"
        if w.public_status == "performance-bearing"
        else "Systems Experiment Execution Contract"
    )
    parts = [f"## {title}", ""]
    if isinstance(functional, dict) and functional:
        notes = functional.get("reviewer_notes")
        visible = {
            key: value for key, value in functional.items() if key != "reviewer_notes"
        }
        parts += ["**Functional acceptance:**", "", mapping_table(visible)]
        if isinstance(notes, list) and notes:
            parts += [bullet_block("Reviewer notes", notes)]
    if isinstance(reference, dict) and reference:
        parts += ["**Five-seed reference protocol:**", "", mapping_table(reference)]
    if isinstance(measurement, dict) and measurement:
        parts += [
            "**Within-run measurement protocol:**",
            "",
            mapping_table(measurement),
        ]
    if isinstance(checkpoint, dict) and checkpoint:
        parts += ["**Checkpoint contract:**", "", mapping_table(checkpoint)]
    if isinstance(quality, dict) and quality:
        parts += ["**Task-quality evaluation:**", "", mapping_table(quality)]
    return "\n".join(parts).rstrip() + "\n"


def section_verified_baseline(w: Workload) -> str:
    baseline = w.raw.get("verified_baseline")
    if not isinstance(baseline, dict) or not baseline:
        return ""
    note = baseline.get("baseline_note")
    visible = {key: value for key, value in baseline.items() if key != "baseline_note"}
    body = mapping_table(visible)
    if note:
        body += f"\n> {esc(note)}\n"
    if baseline_is_protocol_superseded(baseline):
        title = "Historical Project Reference (Protocol Superseded)"
        disclosure = (
            "This content-addressed packet is retained for historical traceability. "
            "It does not validate the current benchmark contract, is not review "
            "eligible, and must be replaced by a clean reference sweep before promotion. "
            "It is not an MLCommons-verified result."
        )
    elif baseline_is_current_review_evidence(baseline):
        title = "Recorded Project Reference Baseline"
        disclosure = (
            "This is a project reference baseline, not an MLCommons-verified result."
        )
    else:
        title = "Development Calibration (Not Review Eligible)"
        disclosure = (
            "This is a project development calibration, not an MLCommons-verified "
            "result or a review-eligible reference package."
        )
    body += f"\n> {disclosure}\n"
    return f"## {title}\n\n{body}"


def _result_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    magnitude = abs(float(value))
    if magnitude >= 1000:
        return f"{value:,.1f}"
    if magnitude >= 10:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _quality_result(record: dict[str, Any], workload: Workload) -> str:
    quality = record.get("quality")
    if not isinstance(quality, dict):
        return "Performance-only case"
    source_gate = quality.get("gate") or {}
    direction = workload.quality_direction
    comparison = "≥" if direction == "higher" else "≤" if direction == "lower" else "="
    mean = (quality.get("aggregate") or {}).get("mean")
    target = workload.quality_value
    tolerance = workload.quality_tolerance or 0.0
    values = quality.get("values") or []
    compatible_metric = quality.get("metric") == workload.quality_metric
    status = "fail"
    if (
        compatible_metric
        and isinstance(target, (int, float))
        and direction in {"higher", "lower"}
        and values
    ):
        status = (
            "pass"
            if all(
                isinstance(value, (int, float))
                and quality_target_satisfied(
                    float(value),
                    float(target),
                    direction=direction,
                    tolerance=float(tolerance),
                )
                for value in values
            )
            else "fail"
        )
    current_gate = {
        "metric": workload.quality_metric,
        "target": target,
        "direction": direction,
        "tolerance": tolerance,
    }
    source_gate_comparable = {key: source_gate.get(key) for key in current_gate}
    disclosure = (
        " *(recomputed with the current registry contract)*"
        if source_gate_comparable != current_gate
        else ""
    )
    return (
        f"`{workload.quality_metric}` {_result_number(mean)} mean; "
        f"target {comparison} {_result_number(target)}; **{status}**{disclosure}"
    )


def _repeatability_result(record: dict[str, Any]) -> str:
    repeatability = record.get("repeatability") or {}
    cv = repeatability.get("coefficient_of_variation")
    if cv is None:
        return "Not established"
    status = "pass" if repeatability.get("passed") else "diagnostic fail"
    return f"CV {float(cv):.2%}; **{status}**"


def section_reference_results(workload: Workload, records: list[dict[str, Any]]) -> str:
    if not records:
        return ""
    evidence_labels = {
        "five-run-verified": "Repeated timing",
        "single-run-provisional": "One measurement",
        "two-run-provisional": "Two measurements",
    }
    lines = [
        "## Draft Reference Results",
        "",
        "These project-generated measurements demonstrate the current `max` paths on "
        "the disclosed reference system. Records measured more than once report a "
        "timing spread; a single measurement is reported as one measurement and "
        "makes no repeatability claim. Quality decisions are recomputed "
        "against the current registry contract so a preserved historical target cannot "
        "make a stale pass claim. None are MLCommons-verified results.",
        "",
        "| **Case** | **Evidence** | **Runs** | **Reference measurement** | **Quality** | **Repeatability** | **System** |",
        "|:---|:---|---:|:---|:---|:---|:---|",
    ]
    for record in records:
        phase = record.get("phase")
        case = str(record.get("mode")) + (f" / {phase}" if phase else "")
        measurement = record.get("measurement") or {}
        metric = measurement.get("primary_metric")
        mean = (measurement.get("aggregate") or {}).get("mean")
        reference = f"`{metric}` {_result_number(mean)} mean"
        execution = record.get("execution") or {}
        devices = ", ".join(execution.get("executed_devices") or [])
        chips = ", ".join(execution.get("hardware_chips") or [])
        system = f"{chips}; {devices}" if devices else chips
        evidence = evidence_labels.get(
            record.get("evidence_class"), esc(record.get("evidence_class"))
        )
        lines.append(
            f"| {esc(case)} | {evidence} | {measurement.get('run_count', '—')} "
            f"| {reference} | {_quality_result(record, workload)} "
            f"| {_repeatability_result(record)} | {esc(system)} |"
        )
    lines += [
        "",
        "> Quality is accepted from one complete run. Timing is reported from the "
        "measurements a case actually has, and repeated timing is reported where it "
        "exists rather than treated as a separate class of result.",
    ]
    return "\n".join(lines) + "\n"


def section_calibration_observation(w: Workload) -> str:
    calibration = w.raw.get("calibration_observation")
    if not isinstance(calibration, dict) or not calibration:
        return ""
    body = mapping_table(calibration)
    body += (
        "\n> This observation is local development context. It is not a "
        "review-eligible reference result.\n"
    )
    return f"## Local Calibration Observation\n\n{body}"


def section_results_pointer(w: Workload, records: list[dict[str, Any]]) -> str:
    """Show one illustrative measurement, then send the reader to their own run.

    The site is not a scoreboard. It carries just enough of a number to make
    the metric and the time cost concrete, explicitly labelled as one machine's
    observation. Evidence classes, repeatability statistics, pass/fail verdicts,
    and promotion eligibility belong to the run artifact, not to a page every
    reader sees regardless of their hardware.
    """
    lines = ["## Results", ""]

    example = next(
        (r for r in records if (r.get("quality") or {}).get("aggregate")), None
    )
    if example is not None:
        quality = example["quality"]
        metric = esc(str(quality.get("metric") or w.quality_metric))
        observed = _result_number((quality.get("aggregate") or {}).get("mean"))
        measurement = example.get("measurement") or {}
        cost_metric = esc(str(measurement.get("primary_metric") or ""))
        cost = _result_number((measurement.get("aggregate") or {}).get("mean"))
        chips = ", ".join((example.get("execution") or {}).get("hardware_chips") or [])

        lines += [
            "To make the metric concrete, one run of this benchmark on "
            f"{esc(chips) or 'the project reference machine'} observed "
            f"`{metric}` of {observed}, taking {cost} `{cost_metric}`.",
            "",
            "That is an illustration, not a target and not a score. It comes "
            "from a single machine and your hardware will produce different "
            "timing. The number that matters is the one your own run reports.",
            "",
        ]

    lines += [
        "Run the benchmark and read your own report:",
        "",
        "```bash",
        f"{CHECKOUT_COMMAND} run --workload {w.id} --profile max",
        "```",
        "",
        "Every run writes an HTML dashboard for reading, JSON for machine use, "
        "CSV for spreadsheets, and a `.provd.json` provenance manifest. Those "
        "artifacts carry the quality decision, timing distribution, hardware "
        "and software fingerprint, asset digests, and model lineage for the run "
        "you performed.",
        "",
        "See [Reading Results](../../guide/results.qmd) for how to interpret "
        "them.",
        "",
    ]
    return "\n".join(lines)


def section_regime(w: Workload) -> str:
    regime = w.raw.get("regime")
    if not isinstance(regime, dict) or not regime:
        return ""
    parts = ["## Measured Systems Regime", ""]
    labels = {
        "working_set": "Working set",
        "arithmetic_intensity": "Arithmetic intensity",
        "dispatch": "Dispatch",
    }
    for key, label in labels.items():
        entry = regime.get(key)
        if not isinstance(entry, dict):
            continue
        rows = []
        for field, value in entry.items():
            rows.append((field.replace("_", " ").capitalize(), value))
        parts.append(f"**{label}:**\n")
        parts.append(kv_table(rows))
    return "\n".join(parts)


def section_model_source(w: Workload) -> str:
    source = w.raw.get("model_source")
    if not isinstance(source, dict) or not source:
        return ""
    rows = [
        ("Type", source.get("type")),
        ("Model", source.get("repo_id")),
        ("Pinned revision", source.get("revision")),
        ("License", source.get("license")),
    ]
    body = kv_table(rows)
    for key in ("selection_rationale", "size_rationale", "backend_rationale"):
        if source.get(key):
            title = key.replace("_", " ").capitalize()
            body += f"\n**{title}:** {esc(source[key])}\n"
    return f"## Model Source\n\n{body}"


def section_runner(w: Workload, source_paths: dict[str, str]) -> str:
    runner = w.raw.get("runner")
    parts = ["## Implementation", ""]
    if isinstance(runner, dict) and runner:
        rows = [(profile, f"`{entry}`") for profile, entry in runner.items()]
        parts.append("| **Profile** | **Runner entry point** |\n|:---|:---|")
        for profile, entry in rows:
            parts.append(f"| {esc(profile)} | {entry} |")
        parts.append("")
    source = source_paths.get(w.id)
    if source:
        parts.append(f"Registry source: [`{source}`]({GITHUB_BLOB}/{source})")
        parts.append("")
    return "\n".join(parts)


def render_workload_body(
    w: Workload,
    source_paths: dict[str, str],
    reference_results: dict[str, list[dict[str, Any]]],
    selection: dict[str, Any],
    heading_shift: bool,
) -> str:
    sections = [
        section_at_a_glance(w),
        section_spiral_status(w),
        section_selection_rationale(w, selection),
        section_provenance(w),
        section_execution_boundary(w),
        section_how_to_run(w),
        section_quality_target(w),
        section_performance_contract(w),
        # Measurements are deliberately absent from the website. The site
        # describes what a workload is and how to run it; every number belongs
        # to the report a run produces on the reader's own machine. See
        # section_results_pointer.
        section_results_pointer(w, reference_results.get(w.id, [])),
        section_model_source(w),
        section_runner(w, source_paths),
    ]
    body = "\n".join(s for s in sections if s)
    if heading_shift:
        body = body.replace("\n## ", "\n### ").replace("## ", "### ", 1)
    return body


def public_line(w: Workload) -> str:
    if w.raw.get("promotion_scope", True) is not True:
        return (
            f"{badge(w.public_status)}\n\n> **Readiness status.** "
            f"{esc(w.public_rationale)} No draft reference result or public baseline "
            "is claimed for this workload.\n"
        )
    return (
        f"{badge(w.public_status)}\n\n> **Source-locked promotion rationale.** "
        f"{esc(w.public_rationale)} Current draft evidence is listed below.\n"
        if w.public_rationale
        else badge(w.public_status) + "\n"
    )


def family_page(
    family: str,
    members: list[Workload],
    source_paths: dict[str, str],
    reference_results: dict[str, list[dict[str, Any]]],
    selection: dict[str, Any],
) -> str:
    lead = family_lead(members)
    suite = lead.suite
    lines = [
        "---",
        f'title: "{family}"',
        f'subtitle: "{suite} suite"',
        "---",
        "",
        GENERATED_NOTE,
        PREVIEW_CALLOUT,
    ]
    if len(members) == 1:
        w = members[0]
        lines.append(public_line(w))
        lines.append(
            render_workload_body(
                w,
                source_paths,
                reference_results,
                selection,
                heading_shift=False,
            )
        )
        return "\n".join(lines) + "\n"

    lines.append(
        f"This workload family exposes **{len(members)} measured variants** "
        f"of `{family}`. Each variant is independently runnable and reported. "
        "The table records model and dataset contracts per variant because "
        "those contracts can differ within a family."
    )
    lines.append("")
    lines.append(
        "| **Variant** | **Workload ID** | **Model** | **Params** | **Dataset** "
        "| **Scenario** | **Candidate result status** |"
    )
    lines.append("|:---|:---|:---|:---|:---|:---|:---|")
    for member in members:
        anchor = f"#variant-{member.variant}"
        lines.append(
            f"| [{esc(member.variant)}]({anchor}) | `{esc(member.id)}` "
            f"| {table_cell(member.model)} | {table_cell(member.raw.get('params'))} "
            f"| {table_cell(member.dataset)} | {table_cell(member.scenario)} "
            f"| {table_cell(member.public_status)} |"
        )
    lines.append("")
    for member in members:
        lines.append(f"## Variant: {member.variant} {{#variant-{member.variant}}}")
        lines.append("")
        lines.append(public_line(member))
        lines.append(
            render_workload_body(
                member,
                source_paths,
                reference_results,
                selection,
                heading_shift=True,
            )
        )
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Index pages
# ---------------------------------------------------------------------------


def family_row(family: str, members: list[Workload], depth: int) -> str:
    lead = family_lead(members)
    variants = f"{len(members)}" if len(members) > 1 else "—"
    href = workload_href(lead.suite, family, depth)
    model = family_metadata_cell(lead, lead.model, [member.model for member in members])
    params = family_metadata_cell(
        lead,
        lead.raw.get("params"),
        [member.raw.get("params") for member in members],
    )
    dataset = family_metadata_cell(
        lead, lead.dataset, [member.dataset for member in members]
    )
    return (
        f"| [`{family}`]({href}) | {table_cell(lead.suite)} | {model} "
        f"| {params} | {dataset} | {variants} "
        f"| {family_status_cell(members, lead)} |"
    )


FAMILY_TABLE_HEADER = (
    "| **Workload** | **Suite** | **Model** | **Params** | **Dataset** "
    "| **Variants** | **Candidate result status** |\n"
    "|:---|:---|:---|:---|:---|:---|:---|"
)


def benchmarks_index(
    families: "OrderedDict[str, list[Workload]]",
    suite_meta: dict[str, dict[str, str]],
    workloads: dict[str, Workload],
    selection: dict[str, Any],
) -> str:
    by_suite: OrderedDict[str, list[str]] = OrderedDict()
    for suite in sorted(suite_meta):
        by_suite[suite] = []
    for family, members in families.items():
        by_suite.setdefault(family_lead(members).suite, []).append(family)

    lines = [
        "---",
        'title: "Benchmark Registry"',
        "toc: true",
        "---",
        "",
        GENERATED_NOTE,
        PREVIEW_CALLOUT,
        f"MLPerf EDU currently registers **{len(workloads)} workloads** in "
        f"**{len(families)} families** across **{sum(1 for s in by_suite.values() if s)} suites**. "
        "Every page in this section is generated from the "
        f"[workload registry]({GITHUB_TREE}/registry); the registry YAML is the "
        "single source of truth for models, datasets, quality targets, and "
        "public-result status.",
        "",
        "## Portfolio Design",
        "",
        esc(selection.get("admission_rule", "")),
        "",
        "The portfolio covers distinct learning tasks and systems behaviors rather "
        "than collecting interchangeable model examples. Each quality contract comes "
        "from the named upstream authority and remains unchanged when a local result "
        "misses it.",
        "",
        "| **Workload** | **Why It Is Included** | **Systems Behavior** | **Quality Authority** |",
        "|:---|:---|:---|:---|",
    ]
    entries = selection.get("workloads") or {}
    for family, members in families.items():
        lead = family_lead(members)
        entry = entries.get(lead.id) or {}
        rationale = entry.get("rationale") or {}
        upstream = entry.get("upstream") or {}
        href = workload_href(lead.suite, family, depth=1)
        authority = (
            f"{upstream.get('authority', '')}. "
            f"{upstream.get('published_baseline', '')}"
        )
        lines.append(
            f"| [`{esc(family)}`]({href}) "
            f"| {esc(rationale.get('classroom_value', ''))} "
            f"| {esc(rationale.get('systems_behavior', ''))} "
            f"| {esc(authority)} |"
        )
    lines += [
        "",
        "## Deliberate Exclusions",
        "",
        "The selection ledger also records attractive ideas that do not yet meet the "
        "suite's stable, authoritative, single-node contract.",
        "",
        "| **Proposal** | **Reason Excluded** |",
        "|:---|:---|",
    ]
    for name, entry in entries.items():
        if isinstance(entry, dict) and entry.get("status") == "rejected":
            lines.append(f"| `{esc(name)}` | {esc(entry.get('reason', ''))} |")
    lines.append("")
    for suite, suite_families in by_suite.items():
        if not suite_families:
            continue
        meta = suite_meta.get(suite, {})
        title = meta.get("title", suite)
        lines.append(f"## {title} (`{suite}`)")
        lines.append("")
        if meta.get("summary"):
            lines.append(esc(meta["summary"]))
            lines.append("")
        lines.append(FAMILY_TABLE_HEADER)
        for family in suite_families:
            lines.append(family_row(family, families[family], depth=1))
        lines.append("")
    return "\n".join(lines) + "\n"


def suite_index(
    suite: str,
    meta: dict[str, str],
    suite_families: list[tuple[str, list[Workload]]],
) -> str:
    lines = [
        "---",
        f'title: "{meta.get("title", suite)}"',
        f'subtitle: "`{suite}` suite"',
        "---",
        "",
        GENERATED_NOTE,
        PREVIEW_CALLOUT,
    ]
    if meta.get("summary"):
        lines += [esc(meta["summary"]), ""]
    if meta.get("focus"):
        lines += [f"**Systems focus:** {esc(meta['focus'])}", ""]
    lines.append(FAMILY_TABLE_HEADER)
    for family, members in suite_families:
        lines.append(family_row(family, members, depth=2))
    lines.append("")
    lines.append("```bash")
    lines.append(f"# run every {suite} workload in the max profile")
    lines.append(f"{CHECKOUT_COMMAND} fetch --suite {suite} --profile max")
    lines.append(f"{CHECKOUT_COMMAND} run --suite {suite} --profile max")
    lines.append("```")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Datasets and CLI reference
# ---------------------------------------------------------------------------


def load_dataset_catalog() -> dict[str, dict[str, Any]]:
    with (ROOT / "datasets.yaml").open("r") as handle:
        catalog = (yaml.safe_load(handle) or {}).get("datasets", {})
    if not isinstance(catalog, dict):
        raise ValueError("datasets.yaml must define a 'datasets' mapping")
    for name, entry in catalog.items():
        if not isinstance(entry, dict):
            raise ValueError(f"dataset '{name}' must be a mapping")
    return catalog


def dataset_usage(workloads: dict[str, Workload]) -> dict[str, list[Workload]]:
    usage: dict[str, list[Workload]] = {}
    for workload in workloads.values():
        datasets = {workload.dataset} if workload.dataset else set()
        mode_contracts = workload.raw.get("mode_contracts") or {}
        for contract in mode_contracts.values():
            if isinstance(contract, dict) and contract.get("dataset"):
                datasets.add(str(contract["dataset"]))
        for dataset in sorted(datasets):
            usage.setdefault(dataset, []).append(workload)
    return usage


def validate_dataset_catalog(
    catalog: dict[str, dict[str, Any]],
    usage: dict[str, list[Workload]],
) -> None:
    missing = sorted(set(usage) - set(catalog))
    stale = sorted(set(catalog) - set(usage))
    problems: list[str] = []
    if missing:
        problems.append(f"registry datasets missing from datasets.yaml: {missing}")
    if stale:
        problems.append(f"datasets.yaml entries unused by the registry: {stale}")

    required = {
        "description",
        "uri",
        "estimated_size_mb",
        "split",
        "license",
        "license_status",
        "public_release_status",
    }
    for name, entry in sorted(catalog.items()):
        absent = sorted(key for key in required if entry.get(key) in (None, ""))
        if absent:
            problems.append(f"dataset '{name}' is missing fields: {absent}")
        if has_asset_dossier(name):
            dossier = asset_dossier(name)
            for field in ("license_status", "public_release_status"):
                if entry.get(field) != dossier.get(field):
                    problems.append(
                        f"dataset '{name}' field '{field}' disagrees with its "
                        f"asset dossier: {entry.get(field)!r} != {dossier.get(field)!r}"
                    )

    tiny_uri = str(catalog.get("tinyshakespeare", {}).get("uri", ""))
    pinned_char_rnn_commit = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
    if pinned_char_rnn_commit not in tiny_uri:
        problems.append("tinyshakespeare must use the pinned char-rnn corpus revision")

    if problems:
        raise ValueError(
            "dataset catalog is inconsistent:\n  - " + "\n  - ".join(problems)
        )


def dataset_source_cell(source: Any) -> str:
    text = str(source or "")
    if text.startswith(("https://", "http://")):
        return f"[upstream]({text})"
    return f"`{esc(text)}`" if text else ""


def datasets_page(workloads: dict[str, Workload]) -> str:
    catalog = load_dataset_catalog()
    usage = dataset_usage(workloads)
    validate_dataset_catalog(catalog, usage)

    lines = [
        "---",
        'title: "Dataset Catalog"',
        "---",
        "",
        GENERATED_NOTE,
        PREVIEW_CALLOUT,
        "The catalog is generated from the workload registry, "
        f"[`datasets.yaml`]({GITHUB_BLOB}/datasets.yaml), and the structured "
        "public-asset dossiers used by the harness. Every registry dataset is "
        "listed, and unresolved release decisions are marked explicitly.",
        "",
        "::: {.callout-note}",
        "**Release boundary.** Dataset status covers the named asset only. The "
        "MLPerf EDU component license, package-index publication, dataset rights, "
        "and MLCommons review remain separate release gates. Fetch-only status "
        "does not authorize redistribution.",
        ":::",
        "",
        "| **Dataset** | **Purpose** | **Size (MB)** | **Source** | **License status** | **Release status** | **Used by** |",
        "|:---|:---|---:|:---|:---|:---|:---|",
    ]
    for name in sorted(usage):
        entry = catalog[name]
        users = usage.get(name, [])
        dossier = (
            asset_dossier(
                name,
                declared_source=users[0].raw.get("dataset_source") if users else None,
            )
            if has_asset_dossier(name)
            else {}
        )
        links = ", ".join(
            f"[`{w.id}`](../benchmarks/{w.suite}/{w.canonical_workload or w.id}.qmd)"
            for w in sorted(users, key=lambda item: item.id)
        )
        source = dossier.get("source_url") or entry.get("uri", "")
        license_status = dossier.get("license_status") or entry.get(
            "license_status", ""
        )
        release_status = dossier.get("public_release_status") or entry.get(
            "public_release_status", ""
        )
        lines.append(
            f"| `{esc(name)}` | {esc(entry.get('description', ''))} "
            f"| {esc(entry.get('estimated_size_mb', ''))} "
            f"| {dataset_source_cell(source)} | `{esc(license_status)}` "
            f"| `{esc(release_status)}` | {links or '—'} |"
        )
    lines += [
        "",
        "Fetch and inspect assets from a source checkout:",
        "",
        "```bash",
        f"{CHECKOUT_COMMAND} fetch --profile max --dry-run   # show what would be downloaded",
        f"{CHECKOUT_COMMAND} info --dataset tinyshakespeare  # one dataset's dossier",
        f"{CHECKOUT_COMMAND} cache list                      # inspect the local cache",
        "```",
    ]
    return "\n".join(lines) + "\n"


def capture_help(args: list[str]) -> str:
    env = dict(os.environ, COLUMNS="80", PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "mlperf_edu", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=True,
    )
    return result.stdout.strip()


def cli_page() -> str:
    lines = [
        "---",
        'title: "CLI Reference"',
        "toc: true",
        "---",
        "",
        GENERATED_NOTE,
        "The `mlperf` command is the entire user surface. This reference is "
        "generated directly from the CLI's own `--help` output, so it cannot "
        "drift from the implementation.",
        "",
        "## mlperf",
        "",
        "```text",
        capture_help(["--help"]),
        "```",
        "",
    ]
    for command in CLI_COMMANDS:
        lines += [
            f"## mlperf {command}",
            "",
            "```text",
            capture_help([command, "--help"]),
            "```",
            "",
        ]
    return "\n".join(lines) + "\n"


def stats_partial(
    workloads: dict[str, Workload],
    families: "OrderedDict[str, list[Workload]]",
    suite_meta: dict[str, dict[str, str]],
) -> str:
    status_counts: dict[str, int] = {}
    for w in workloads.values():
        status_counts[w.public_status] = status_counts.get(w.public_status, 0) + 1
    suites_used = {family_lead(members).suite for members in families.values()}
    status_text = ", ".join(
        f"{count} {status}" for status, count in sorted(status_counts.items())
    )
    lines = [
        GENERATED_NOTE,
        f"**{len(workloads)} workloads** · **{len(families)} families** · "
        f"**{len(suites_used)} suites** · {status_text}",
        "",
        "| **Suite** | **What it measures** | **Workloads** |",
        "|:---|:---|:---|",
    ]
    for suite in sorted(suites_used):
        meta = suite_meta.get(suite, {})
        count = sum(
            1 for members in families.values() if family_lead(members).suite == suite
        )
        lines.append(
            f"| [{esc(meta.get('title', suite))}](benchmarks/{suite}/index.qmd) "
            f"| {esc(meta.get('summary', ''))} | {count} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def normalize(content: str) -> str:
    """Match the repo pre-commit contract: no trailing whitespace on any
    line, exactly one trailing newline. The generator must emit hook-clean
    text or every regeneration would fight the end-of-file/whitespace
    fixers."""
    lines = [line.rstrip() for line in content.split("\n")]
    return "\n".join(lines).rstrip("\n") + "\n"


def build_outputs() -> dict[Path, str]:
    workloads = load_registry(ROOT / "registry")
    suite_meta = load_suite_meta()
    selection = load_selection_ledger(workloads)
    source_paths = index_source_paths()
    reference_results = load_provisional_reference_results()
    families = group_families(workloads)

    unknown_results = set(reference_results) - set(workloads)
    if unknown_results:
        raise ValueError(
            f"reference results name unknown workloads: {sorted(unknown_results)}"
        )

    unknown = {family_lead(members).suite for members in families.values()} - set(
        suite_meta
    )
    if unknown:
        raise ValueError(f"suites missing from registry/suites.yaml: {sorted(unknown)}")

    site = ROOT / "site"
    outputs: dict[Path, str] = {}
    outputs[site / "benchmarks" / "index.qmd"] = benchmarks_index(
        families, suite_meta, workloads, selection
    )
    per_suite: OrderedDict[str, list[tuple[str, list[Workload]]]] = OrderedDict()
    for family, members in families.items():
        per_suite.setdefault(family_lead(members).suite, []).append((family, members))
    for suite, suite_families in per_suite.items():
        outputs[site / "benchmarks" / suite / "index.qmd"] = suite_index(
            suite, suite_meta.get(suite, {}), suite_families
        )
        for family, members in suite_families:
            outputs[site / "benchmarks" / suite / f"{family}.qmd"] = family_page(
                family, members, source_paths, reference_results, selection
            )
    outputs[site / "reference" / "datasets.qmd"] = datasets_page(workloads)
    outputs[site / "reference" / "cli.qmd"] = cli_page()
    outputs[site / "_stats.qmd"] = stats_partial(workloads, families, suite_meta)
    return {path: normalize(content) for path, content in outputs.items()}


def managed_existing(site: Path) -> set[Path]:
    managed: set[Path] = set()
    for pattern in ("benchmarks/**/*.qmd", "_stats.qmd", "_generated/*.qmd"):
        managed.update(site.glob(pattern))
    for name in ("reference/cli.qmd", "reference/datasets.qmd"):
        candidate = site / name
        if candidate.exists():
            managed.add(candidate)
    return managed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated pages are current without writing.",
    )
    args = parser.parse_args()

    outputs = build_outputs()
    site = ROOT / "site"
    stale = managed_existing(site) - set(outputs)

    if args.check:
        problems: list[str] = []
        for path, content in outputs.items():
            if not path.exists():
                problems.append(f"missing: {path.relative_to(ROOT)}")
            elif path.read_text() != content:
                problems.append(f"out of date: {path.relative_to(ROOT)}")
        problems += [f"stale generated file: {p.relative_to(ROOT)}" for p in stale]
        if problems:
            print("generated docs are out of sync with the registry:")
            for problem in problems:
                print(f"  - {problem}")
            print("run: python3 tools/generate_docs.py")
            return 1
        print(f"generated docs are current ({len(outputs)} pages).")
        return 0

    for path in stale:
        path.unlink()
        print(f"removed stale {path.relative_to(ROOT)}")
    written = 0
    for path, content in sorted(outputs.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
            written += 1
    print(f"wrote {written} of {len(outputs)} generated pages under site/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
