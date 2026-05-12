#!/usr/bin/env python3
"""Merge reference-aware and blind-need cross-reference reports.

The output is a decision queue for the second-pass editor. It does not edit book
source files.
"""

from __future__ import annotations

import json
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "book" / "quarto" / "contents").exists():
            return parent
    raise RuntimeError("Could not locate repository root containing book/quarto/contents")


ROOT = find_repo_root()
BASE = ROOT / "review" / "cross-references"
AWARE_DIR = BASE / "chapter-reports"
BLIND_DIR = BASE / "blind-need-reports"
QUEUE_PATH = BASE / "merged-decision-queue.yml"
SUMMARY_PATH = BASE / "merged-decision-summary.md"


def configure_base(base: Path) -> None:
    global BASE, AWARE_DIR, BLIND_DIR, QUEUE_PATH, SUMMARY_PATH

    BASE = base
    AWARE_DIR = BASE / "chapter-reports"
    BLIND_DIR = BASE / "blind-need-reports"
    QUEUE_PATH = BASE / "merged-decision-queue.yml"
    SUMMARY_PATH = BASE / "merged-decision-summary.md"

ACTION_MAP = {
    "no-op": "none",
    "needs-human-review": "needs-human-review",
}
EDIT_ACTIONS = {"add", "retarget", "remove", "localize"}


@dataclass
class Finding:
    source: str
    report: str
    file: str
    line: int | None
    action: str
    status: str | None
    current_reference: str | None
    recommended_target: str | None
    confidence: str | None
    priority: str | None
    rationale: str | None
    context: str | None
    proposed_edit: str | None
    id: str | None


def norm_action(value: Any) -> str:
    action = str(value or "unknown")
    return ACTION_MAP.get(action, action)


def norm_ref(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[1:] if text.startswith("@") else text


def scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return data


def chapter_file(data: dict[str, Any]) -> str:
    chapter = data.get("chapter") or data.get("chapter_report") or {}
    if isinstance(chapter, dict):
        return str(chapter.get("file") or data.get("file") or "")
    return str(data.get("file") or "")


def extract_aware(path: Path) -> list[Finding]:
    data = read_yaml(path)
    chapter = data.get("chapter_report") or {}
    file = chapter_file(data)
    findings = chapter.get("findings") or data.get("findings") or []
    out: list[Finding] = []
    for item in findings:
        if not isinstance(item, dict):
            continue
        action = norm_action(item.get("action"))
        if action not in EDIT_ACTIONS and action != "needs-human-review":
            continue
        out.append(
            Finding(
                source="reference-aware",
                report=path.name,
                file=file,
                line=item.get("line"),
                action=action,
                status=item.get("status"),
                current_reference=norm_ref(item.get("current_reference") or item.get("current_ref")),
                recommended_target=norm_ref(item.get("recommended_target") or item.get("target_anchor")),
                confidence=item.get("confidence"),
                priority=item.get("priority"),
                rationale=item.get("rationale"),
                context=item.get("context"),
                proposed_edit=item.get("proposed_edit"),
                id=item.get("id"),
            )
        )
    return out


def extract_blind(path: Path) -> list[Finding]:
    data = read_yaml(path)
    file = chapter_file(data)
    items = (
        data.get("needs")
        or data.get("findings")
        or (data.get("chapter_report") or {}).get("needs")
        or (data.get("chapter_report") or {}).get("findings")
        or []
    )
    out: list[Finding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = norm_action(item.get("proposed_action") or item.get("action"))
        status = item.get("status")
        if action not in EDIT_ACTIONS and action != "needs-human-review":
            continue
        out.append(
            Finding(
                source="blind-need",
                report=path.name,
                file=file,
                line=item.get("line"),
                action=action,
                status=status,
                current_reference=norm_ref(item.get("existing_reference") or item.get("current_reference")),
                recommended_target=norm_ref(item.get("recommended_target")),
                confidence=item.get("confidence"),
                priority=item.get("priority"),
                rationale=item.get("rationale"),
                context=item.get("reader_need") or item.get("context") or item.get("concept"),
                proposed_edit=item.get("proposed_edit"),
                id=item.get("id"),
            )
        )
    return out


def load_findings() -> tuple[list[Finding], list[Finding]]:
    aware = []
    for path in sorted(AWARE_DIR.glob("*.yml")):
        aware.extend(extract_aware(path))
    blind = []
    for path in sorted(BLIND_DIR.glob("*.blind.yml")):
        blind.extend(extract_blind(path))
    return aware, blind


def compatible(left: Finding, right: Finding) -> bool:
    if left.file != right.file or left.action != right.action:
        return False
    if left.line is None or right.line is None:
        return False
    if abs(int(left.line) - int(right.line)) > 5:
        return False
    if left.action in {"add", "retarget"}:
        return bool(left.recommended_target and right.recommended_target and left.recommended_target == right.recommended_target)
    if left.action == "remove":
        return (left.current_reference or "") == (right.current_reference or "") or not left.current_reference or not right.current_reference
    if left.action == "localize":
        return True
    return False


def confidence_rank(value: str | None) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(value or "").lower(), 0)


def queue_status(findings: list[Finding]) -> str:
    actions = {f.action for f in findings}
    if "needs-human-review" in actions:
        return "needs-human-review"
    if len(findings) >= 2:
        if all(confidence_rank(f.confidence) >= 2 for f in findings):
            return "strong-candidate"
        return "review"
    only = findings[0]
    if only.action == "remove" and only.confidence == "high":
        return "candidate"
    if only.action in {"retarget", "localize"} and confidence_rank(only.confidence) >= 2:
        return "candidate"
    if only.action == "add" and only.confidence == "high":
        return "candidate"
    return "needs-human-review"


def merge_findings(aware: list[Finding], blind: list[Finding]) -> list[dict[str, Any]]:
    used_blind: set[int] = set()
    records: list[list[Finding]] = []

    for aware_item in aware:
        match_idx = None
        for idx, blind_item in enumerate(blind):
            if idx in used_blind:
                continue
            if compatible(aware_item, blind_item):
                match_idx = idx
                break
        if match_idx is None:
            records.append([aware_item])
        else:
            used_blind.add(match_idx)
            records.append([aware_item, blind[match_idx]])

    for idx, blind_item in enumerate(blind):
        if idx not in used_blind:
            records.append([blind_item])

    out: list[dict[str, Any]] = []
    for seq, findings in enumerate(records, start=1):
        primary = sorted(
            findings,
            key=lambda f: (confidence_rank(f.confidence), 1 if f.source == "blind-need" else 0),
            reverse=True,
        )[0]
        out.append(
            {
                "id": f"xref-decision-{seq:04d}",
                "status": queue_status(findings),
                "action": primary.action,
                "file": primary.file,
                "line": primary.line,
                "current_reference": primary.current_reference,
                "recommended_target": primary.recommended_target,
                "confidence": primary.confidence,
                "priority": primary.priority,
                "sources": [f.source for f in findings],
                "source_reports": [f.report for f in findings],
                "source_ids": [f.id for f in findings],
                "rationales": [f.rationale for f in findings if f.rationale],
                "contexts": [f.context for f in findings if f.context],
                "proposed_edits": [f.proposed_edit for f in findings if f.proposed_edit],
            }
        )

    return sorted(out, key=lambda r: (r["file"], r["line"] or 0, r["action"], r["id"]))


def write_queue(records: list[dict[str, Any]]) -> None:
    lines = [
        'schema_version: "crossref-merged-decision-queue/v1"',
        "description: \"Merged queue from reference-aware and blind-need cross-reference analyses.\"",
        "decisions:",
    ]
    for record in records:
        lines.append(f"  - id: {scalar(record['id'])}")
        lines.append(f"    status: {scalar(record['status'])}")
        lines.append(f"    action: {scalar(record['action'])}")
        lines.append(f"    file: {scalar(record['file'])}")
        lines.append(f"    line: {record['line'] if record['line'] is not None else 'null'}")
        lines.append(f"    current_reference: {scalar(record['current_reference'])}")
        lines.append(f"    recommended_target: {scalar(record['recommended_target'])}")
        lines.append(f"    confidence: {scalar(record['confidence'])}")
        lines.append(f"    priority: {scalar(record['priority'])}")
        lines.append("    sources:")
        for source in record["sources"]:
            lines.append(f"      - {scalar(source)}")
        lines.append("    source_reports:")
        for report in record["source_reports"]:
            lines.append(f"      - {scalar(report)}")
        lines.append("    source_ids:")
        for source_id in record["source_ids"]:
            lines.append(f"      - {scalar(source_id)}")
        lines.append("    rationales:")
        for rationale in record["rationales"]:
            lines.append(f"      - {scalar(rationale)}")
        if not record["rationales"]:
            lines.append("      []")
        lines.append("    contexts:")
        for context in record["contexts"]:
            lines.append(f"      - {scalar(context)}")
        if not record["contexts"]:
            lines.append("      []")
        lines.append("    proposed_edits:")
        for edit in record["proposed_edits"]:
            lines.append(f"      - {scalar(edit)}")
        if not record["proposed_edits"]:
            lines.append("      []")
    QUEUE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(records: list[dict[str, Any]], aware: list[Finding], blind: list[Finding]) -> None:
    by_status = Counter(r["status"] for r in records)
    by_action = Counter(r["action"] for r in records)
    by_sources = Counter("+".join(r["sources"]) for r in records)
    by_file = defaultdict(Counter)
    for r in records:
        by_file[r["file"]][r["action"]] += 1

    lines = [
        "# Merged Cross-Reference Decision Summary",
        "",
        "Generated by `review/cross-references/scripts/merge_crossref_reports.py`.",
        "",
        "## Inputs",
        "",
        f"- Reference-aware findings considered: `{len(aware)}`",
        f"- Blind-need findings considered: `{len(blind)}`",
        f"- Merged decisions: `{len(records)}`",
        "",
        "## By Status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for key, count in sorted(by_status.items()):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## By Action", "", "| Action | Count |", "|---|---:|"])
    for key, count in sorted(by_action.items()):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## By Source", "", "| Source Set | Count |", "|---|---:|"])
    for key, count in sorted(by_sources.items()):
        lines.append(f"| `{key}` | {count} |")

    lines.extend(["", "## Strong Candidates", ""])
    strong = [r for r in records if r["status"] == "strong-candidate"]
    if not strong:
        lines.append("No strong candidates found.")
    else:
        lines.append("| Action | Source | Target | Rationale |")
        lines.append("|---|---|---|---|")
        for r in strong[:80]:
            source = f"`{r['file']}:{r['line']}`"
            target = r["recommended_target"] or r["current_reference"] or ""
            rationale = (r["rationales"][0] if r["rationales"] else "").replace("|", "\\|")
            lines.append(f"| `{r['action']}` | {source} | `{target}` | {rationale} |")
        if len(strong) > 80:
            lines.append("")
            lines.append(f"Showing first 80 of {len(strong)} strong candidates.")

    lines.extend(["", "## File Hotspots", "", "| File | Add | Retarget | Remove | Localize | Review |", "|---|---:|---:|---:|---:|---:|"])
    for file, counts in sorted(by_file.items(), key=lambda item: (-sum(item[1].values()), item[0]))[:60]:
        review = sum(1 for r in records if r["file"] == file and r["status"] == "needs-human-review")
        lines.append(
            f"| `{file}` | {counts['add']} | {counts['retarget']} | {counts['remove']} | {counts['localize']} | {review} |"
        )
    lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge cross-reference YAML reports.")
    parser.add_argument(
        "--base-dir",
        default="review/cross-references",
        help="Review directory relative to the repository root, or an absolute path.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    configure_base(base_dir if base_dir.is_absolute() else ROOT / base_dir)
    aware, blind = load_findings()
    records = merge_findings(aware, blind)
    write_queue(records)
    write_summary(records, aware, blind)
    print(f"Wrote {QUEUE_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"reference-aware findings: {len(aware)}")
    print(f"blind-need findings: {len(blind)}")
    print(f"merged decisions: {len(records)}")
    print(json.dumps(Counter(r["status"] for r in records), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
