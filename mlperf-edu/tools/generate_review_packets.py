from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from mlperf.assets import asset_dossier, huggingface_model_dossier
from mlperf.edu_cli import public_audit_warnings, workload_run_selector
from mlperf.registry import (
    Workload,
    baseline_is_current_review_evidence,
    baseline_is_protocol_superseded,
    load_registry,
    select_workloads,
)


PUBLIC_REVIEW_STATUSES = ("score-bearing", "performance-bearing")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate MLPerf EDU workload review packets."
    )
    parser.add_argument(
        "--output-dir",
        default="review_packets",
        help="Directory for generated Markdown packets.",
    )
    parser.add_argument(
        "--status",
        choices=PUBLIC_REVIEW_STATUSES,
        default=None,
        help="Limit packets to one public status.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated packets are current without writing files.",
    )
    args = parser.parse_args()

    workloads = load_registry()
    selected = [
        workload
        for status in PUBLIC_REVIEW_STATUSES
        for workload in select_workloads(workloads, public_status=status)
        if args.status in (None, status)
    ]

    output_dir = Path(args.output_dir)
    expected = expected_packets(selected, output_dir, workloads)
    if args.check:
        return check_packets(expected, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in expected.items():
        path.write_text(content, encoding="utf-8")
    for path in output_dir.glob("*.md"):
        if path not in expected:
            path.unlink()
    print(f"wrote {len(expected) - 1} packet(s) to {output_dir}")
    return 0


def expected_packets(
    selected: list[Workload], output_dir: Path, workloads: dict[str, Workload]
) -> dict[Path, str]:
    paths = [output_dir / f"{packet_slug(workload)}.md" for workload in selected]
    expected = {
        path: render_packet(workload, workloads)
        for workload, path in zip(selected, paths)
    }
    expected[output_dir / "README.md"] = render_index(selected, paths)
    return expected


def check_packets(expected: dict[Path, str], output_dir: Path) -> int:
    problems: list[str] = []
    for path, content in expected.items():
        if not path.exists():
            problems.append(f"missing {path}")
            continue
        if path.read_text(encoding="utf-8") != content:
            problems.append(f"stale {path}")

    if output_dir.exists():
        extra = sorted(path for path in output_dir.glob("*.md") if path not in expected)
        problems.extend(f"extra {path}" for path in extra)

    if problems:
        print("review packets are not current:")
        for problem in problems:
            print(f"- {problem}")
        print("run: python3 tools/generate_review_packets.py")
        return 1

    print(f"review packets are current ({len(expected) - 1} packet(s))")
    return 0


def packet_slug(workload: Workload) -> str:
    selector = workload_run_selector(workload)
    selector = selector.replace(" --variant ", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", selector).strip("-")


def render_index(workloads: list[Workload], paths: list[Path]) -> str:
    rows = [
        "| Workload | Internal ID | Suite | Public status | Packet |",
        "|---|---|---|---|---|",
    ]
    for workload, path in zip(workloads, paths):
        rows.append(
            f"| `{workload_run_selector(workload)}` | `{workload.id}` | `{workload.suite}` | `{workload.public_status}` | [{path.name}]({path.name}) |"
        )
    return "\n".join(
        [
            "# MLPerf EDU Review Packets",
            "",
            "These packets are generated from the native registry and structured asset dossiers.",
            "They are intended for MLCommons, instructor, and artifact-review feedback.",
            "A committed evidence summary can pass repository CI while the packet continues to flag a raw-package publication blocker.",
            "",
            *rows,
            "",
        ]
    )


def render_packet(workload: Workload, workloads: dict[str, Workload]) -> str:
    raw = workload.raw
    sections = [
        f"# MLPerf EDU Review Packet: `{workload_run_selector(workload)}`",
        "",
        "## Summary",
        "",
        summary_table(workload),
        "",
        "## Reviewer Commands",
        "",
        command_block(workload),
        "",
        quality_section(workload),
        "",
        evidence_contract_section(workload),
        "",
        taxonomy_section(workload),
        "",
        asset_section(workload),
        "",
        checkpoint_section(workload, workloads),
        "",
        "## Public Review Notes",
        "",
        bullet_list(public_review_notes(workload, workloads)),
        "",
        "## Source Provenance",
        "",
        bullet_list(
            [
                f"Registry provenance: {raw.get('provenance', 'not declared')}",
                f"Runner min: {(raw.get('runner') or {}).get('min', 'missing')}",
                f"Runner max: {(raw.get('runner') or {}).get('max', 'missing')}",
            ]
        ),
        "",
    ]
    return "\n".join(section for section in sections if section is not None)


def summary_table(workload: Workload) -> str:
    rows = [
        ("Internal ID", workload.id),
        ("Run selector", workload_run_selector(workload)),
        ("Suite", workload.suite),
        ("Public status", workload.public_status),
        ("Scenario", workload.scenario or ""),
        ("Model", workload.model),
        ("Dataset", workload.dataset or ""),
    ]
    if workload.canonical_workload:
        rows.extend(
            [
                ("Canonical workload", workload.canonical_workload),
                ("Variant", workload.variant or ""),
            ]
        )
    rows = [(field, value) for field, value in rows if value not in (None, "")]
    return markdown_table(("Field", "Value"), rows)


def command_block(workload: Workload) -> str:
    selector = workload_run_selector(workload)
    if " --variant " in selector:
        workload_name, variant = selector.split(" --variant ", 1)
        args = f"--workload {workload_name} --variant {variant}"
    else:
        args = f"--workload {selector}"
    slug = packet_slug(workload)
    commands = [
        "```bash",
        f'OUTPUT_DIR="submissions/review-{slug}"',
    ]
    shared_checkpoint = workload.raw.get("shared_checkpoint")
    if shared_checkpoint:
        commands.extend(
            [
                f"mlperf fetch --workload {shared_checkpoint} --profile max",
                f'mlperf run --workload {shared_checkpoint} --profile max --output-dir "$OUTPUT_DIR"',
            ]
        )
    commands.extend(
        [
            f"mlperf fetch {args} --profile max",
            f'mlperf run {args} --profile max --output-dir "$OUTPUT_DIR"',
            'for manifest in "$OUTPUT_DIR"/*.provd.json; do mlperf verify "$manifest"; done',
            'mlperf grade "$OUTPUT_DIR" --output "$OUTPUT_DIR/grade.json"',
            "```",
        ]
    )
    return "\n".join(commands)


def quality_section(workload: Workload) -> str:
    if workload.public_status == "score-bearing":
        quality = workload.raw.get("quality_target") or {}
        rows = [
            ("Metric", quality.get("metric", "")),
            ("Target", quality.get("value", "")),
            ("Direction", quality.get("direction", "")),
            ("Target basis", quality.get("target_basis", "")),
            ("Reference runs", quality.get("reference_runs", "")),
            (
                "Acceptance rule",
                (quality.get("variance_summary") or {}).get("acceptance_rule", ""),
            ),
            ("Reference protocol", compact_dict(quality.get("reference_protocol"))),
        ]
        rows = [(field, value) for field, value in rows if value not in (None, "")]
        return "\n".join(
            ["## Quality Contract", "", markdown_table(("Field", "Value"), rows)]
        )

    functional = workload.raw.get("functional_check") or {}
    reference_protocol = workload.raw.get("performance_reference_protocol") or {}
    rows = [
        ("Functional metric", functional.get("metric", "")),
        ("Condition", functional.get("condition", "")),
        ("Independent reference runs", reference_protocol.get("reference_runs", "")),
        (
            "Reviewer notes",
            "; ".join(str(note) for note in functional.get("reviewer_notes", [])),
        ),
    ]
    rows = [(field, value) for field, value in rows if value not in (None, "")]
    return "\n".join(
        ["## Functional Contract", "", markdown_table(("Field", "Value"), rows)]
    )


def evidence_contract_section(workload: Workload) -> str:
    raw = workload.raw
    baseline = raw.get("verified_baseline") or {}
    calibration = raw.get("calibration_observation") or {}
    if baseline_is_protocol_superseded(baseline):
        baseline_role = "historical-protocol-superseded"
        baseline_disclosure = (
            "Retained for historical traceability only; it does not validate the current "
            "contract and is not an MLCommons-verified result."
        )
    elif baseline_is_current_review_evidence(baseline):
        baseline_role = "current-review-evidence"
        baseline_disclosure = (
            "Project reference evidence; not an MLCommons-verified result."
        )
    else:
        baseline_role = "development-only"
        baseline_disclosure = (
            "Development evidence only; not an MLCommons-verified result."
        )
    rows = [
        ("Reference protocol", compact_dict(raw.get("performance_reference_protocol"))),
        ("Measurement protocol", compact_dict(raw.get("measurement_protocol"))),
        ("Checkpoint contract", compact_dict(raw.get("checkpoint_contract"))),
        ("Task-quality evaluation", compact_dict(raw.get("quality_evaluation"))),
        ("Baseline record", compact_dict(baseline)),
        ("Baseline record role", baseline_role),
        ("Baseline disclosure", baseline_disclosure),
        ("Baseline evidence status", baseline.get("evidence_status", "not declared")),
        ("Baseline review eligible", baseline.get("review_eligible", "not declared")),
        ("Baseline evidence file", baseline.get("evidence_file", "not declared")),
        (
            "Reference package availability",
            baseline.get("reference_package_availability", "not declared"),
        ),
        (
            "External publication status",
            baseline.get("external_publication_status", "not declared"),
        ),
        (
            "External publication URL",
            baseline.get("external_publication_url", "not declared"),
        ),
        ("Calibration observation", compact_dict(calibration)),
    ]
    rows = [(field, value) for field, value in rows if value not in (None, "")]
    return "\n".join(
        [
            "## Measurement and Evidence Contract",
            "",
            markdown_table(("Field", "Value"), rows),
        ]
    )


def taxonomy_section(workload: Workload) -> str:
    regime = workload.raw.get("regime") or {}
    rows = []
    for axis in ("working_set", "arithmetic_intensity", "dispatch"):
        block = regime.get(axis) or {}
        value = block.get("value", "missing")
        sidecar = block.get("evidence_sidecar", "none")
        digest = block.get("evidence_sha256", "none")
        note = block.get("note", "")
        rows.append(
            (axis, f"value={value}; evidence={sidecar}; sha256={digest}; note={note}")
        )
    return "\n".join(
        [
            "## Taxonomy Evidence",
            "",
            markdown_table(("Axis", "Claim and evidence"), rows),
        ]
    )


def asset_section(workload: Workload) -> str:
    dataset = asset_dossier(
        workload.dataset, declared_source=workload.raw.get("dataset_source")
    )
    rows = [
        ("Dataset asset", dataset.get("id", "")),
        ("Dataset source", dataset.get("source_url", "")),
        ("Dataset license status", dataset.get("license_status", "")),
        ("Dataset release status", dataset.get("public_release_status", "")),
        ("Dataset release next step", dataset.get("release_next_step", "")),
        ("Dataset citation", dataset.get("citation", "")),
    ]
    model_source = workload.raw.get("model_source")
    if (
        isinstance(model_source, dict)
        and model_source.get("type") == "huggingface-pinned"
    ):
        model = huggingface_model_dossier(
            model_source,
            model_name=workload.model,
            model_id=str(model_source.get("repo_id") or workload.model),
        )
        rows.extend(
            [
                ("Model source", model.get("source_url", "")),
                ("Model license", model.get("license", "")),
                ("Model rationale", model.get("selection_rationale", "")),
            ]
        )
    rows = [(field, value) for field, value in rows if value not in (None, "")]
    return "\n".join(["## Assets", "", markdown_table(("Field", "Value"), rows)])


def checkpoint_section(workload: Workload, workloads: dict[str, Workload]) -> str:
    if not workload.raw.get("shared_checkpoint"):
        return "## Checkpoint Lineage\n\n- No shared checkpoint dependency declared."
    source = workloads.get(str(workload.raw.get("shared_checkpoint")))
    rows = [
        ("Shared checkpoint", workload.raw.get("shared_checkpoint", "")),
        ("Quality dependency", workload.raw.get("quality_dependency", "")),
        (
            "Source run selector",
            workload_run_selector(source)
            if source
            else workload.raw.get("shared_checkpoint", ""),
        ),
        ("Source quality", source_quality_summary(source) if source else ""),
        (
            "Source baseline record",
            compact_dict(source.raw.get("verified_baseline")) if source else "",
        ),
        (
            "Policy",
            "Preserve the source training report and .provd.json alongside checkpoint-backed inference results.",
        ),
    ]
    return "\n".join(
        ["## Checkpoint Lineage", "", markdown_table(("Field", "Value"), rows)]
    )


def source_quality_summary(workload: Workload | None) -> str:
    if workload is None or not workload.quality_metric:
        return ""
    parts = [str(workload.quality_metric)]
    if workload.quality_direction:
        parts.append(str(workload.quality_direction))
    if workload.quality_value is not None:
        parts.append(str(workload.quality_value))
    if workload.quality_target_basis:
        parts.append(f"basis={workload.quality_target_basis}")
    return " ".join(parts)


def public_review_notes(
    workload: Workload, workloads: dict[str, Workload]
) -> list[str]:
    warnings = public_audit_warnings(workload)
    baseline = workload.raw.get("verified_baseline")
    if workload.public_status in {"score-bearing", "performance-bearing"}:
        evidence_status = (
            baseline.get("evidence_status") if isinstance(baseline, dict) else None
        )
        if evidence_status != "committed-reference-summary":
            warnings.append(
                f"{workload.public_status} baseline is not backed by a committed reference summary; "
                f"evidence status is {evidence_status or 'not declared'}"
            )
        if baseline_is_protocol_superseded(baseline):
            reason = str(baseline.get("superseded_reason") or "").strip()
            detail = f" Reason: {reason}" if reason else ""
            warnings.append(
                "replacement blocker: the committed packet is historical and uses a "
                "protocol superseded by the current benchmark contract; a clean reference "
                f"sweep is required before promotion.{detail}"
            )
    calibration = workload.raw.get("calibration_observation")
    if (
        workload.public_status == "performance-bearing"
        and isinstance(calibration, dict)
        and not (
            isinstance(baseline, dict)
            and baseline.get("evidence_status") == "committed-reference-summary"
        )
    ):
        warnings.append(
            "calibration values are informational and are not a review baseline; "
            f"evidence status is {calibration.get('evidence_status', 'not declared')}"
        )
    checkpoint_id = workload.raw.get("shared_checkpoint")
    checkpoint_source = workloads.get(str(checkpoint_id)) if checkpoint_id else None
    if checkpoint_source is not None:
        source_baseline = checkpoint_source.raw.get("verified_baseline") or {}
        source_status = (
            source_baseline.get("evidence_status")
            if isinstance(source_baseline, dict)
            else None
        )
        if source_status != "committed-reference-summary":
            warnings.append(
                f"shared checkpoint source {checkpoint_source.id} is not backed by a committed "
                f"reference summary; evidence status is {source_status or 'not declared'}"
            )
        else:
            if baseline_is_protocol_superseded(source_baseline):
                warnings.append(
                    f"replacement blocker: shared checkpoint source {checkpoint_source.id} "
                    "has only protocol-superseded historical evidence"
                )
            if source_baseline.get("reference_package_availability") != "published":
                warnings.append(
                    f"external-publication blocker: the raw reference package for shared checkpoint "
                    f"source {checkpoint_source.id} is not yet publicly retrievable"
                )
    if warnings:
        return warnings
    return ["No public-release warning from the current structured audit."]


def markdown_table(headers: tuple[str, str], rows: list[tuple[str, Any]]) -> str:
    output = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    for key, value in rows:
        output.append(
            f"| {escape_cell(str(key))} | {escape_cell(format_value(value))} |"
        )
    return "\n".join(output)


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def compact_dict(value: Any) -> str:
    if not isinstance(value, dict):
        return format_value(value)
    parts = []
    for key, item in value.items():
        if isinstance(item, list):
            item = ", ".join(str(entry) for entry in item)
        parts.append(f"{key}={item}")
    return "; ".join(parts)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return compact_dict(value)
    return str(value)


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
