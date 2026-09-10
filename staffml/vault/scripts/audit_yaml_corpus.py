#!/usr/bin/env python3
"""Read-only deterministic audit for StaffML question YAML files.

This script intentionally ignores historical LLM audit artifacts. It validates
the current YAML corpus against the local schema and authoring conventions, then
writes a JSONL issue log plus a short Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = VAULT_DIR.parents[1]
QUESTIONS_DIR = VAULT_DIR / "questions"
SCHEMA_DIR = VAULT_DIR / "schema"

if str(VAULT_DIR) not in sys.path:
    sys.path.insert(0, str(VAULT_DIR))
if str(SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEMA_DIR))

from schema import Question  # noqa: E402
from enums import ZONE_BLOOM_AFFINITY  # noqa: E402


TITLE_MARKDOWN_RE = re.compile(r"(\*\*|__|`|\[[^\]]+\]\(|\$[^$]+\$|\\[A-Za-z]+|<[^>]+>)")
HTML_RE = re.compile(r"<[^>]+>")
GENERIC_TITLE_RE = re.compile(
    r"^(question|q\d+|cloud q\d+|edge q\d+|mobile q\d+|tinyml q\d+|global q\d+|"
    r"memory question|latency question|compute question|draft|todo|tbd)$",
    re.IGNORECASE,
)
COMMON_MISTAKE_MARKERS = (
    "**The Pitfall:**",
    "**The Rationale:**",
    "**The Consequence:**",
)
NAPKIN_MATH_MARKERS = (
    "**Assumptions",
    "**Calculations:**",
    "**Conclusion",
)


@dataclass(frozen=True)
class Issue:
    severity: str
    category: str
    code: str
    path: str
    qid: str | None
    message: str


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def add_issue(
    issues: list[Issue],
    severity: str,
    category: str,
    code: str,
    path: Path,
    qid: str | None,
    message: str,
) -> None:
    issues.append(Issue(severity, category, code, rel(path), qid, message))


def ordered_markers_missing(text: str, markers: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    cursor = 0
    for marker in markers:
        idx = text.find(marker, cursor)
        if idx < 0:
            missing.append(marker)
        else:
            cursor = idx + len(marker)
    return missing


def load_chains() -> set[str]:
    path = VAULT_DIR / "chains.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except Exception:
        return set()

    ids: set[str] = set()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.add(item["id"])
    elif isinstance(data, dict):
        if isinstance(data.get("chains"), list):
            for item in data["chains"]:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    ids.add(item["id"])
        ids.update(k for k in data.keys() if isinstance(k, str))
    return ids


def validate_path_invariants(path: Path, data: dict[str, Any], issues: list[Issue]) -> None:
    qid = data.get("id") if isinstance(data.get("id"), str) else None
    parts = path.relative_to(QUESTIONS_DIR).parts
    if len(parts) != 3:
        add_issue(
            issues,
            "error",
            "filesystem",
            "bad_question_path",
            path,
            qid,
            "Question YAML must live at questions/<track>/<competency_area>/<id>.yaml",
        )
        return

    track_dir, area_dir, filename = parts
    stem = Path(filename).stem
    if qid and stem != qid:
        add_issue(
            issues,
            "error",
            "filesystem",
            "id_filename_mismatch",
            path,
            qid,
            f"id {qid!r} does not match filename stem {stem!r}",
        )
    if data.get("track") and data.get("track") != track_dir:
        add_issue(
            issues,
            "error",
            "filesystem",
            "track_path_mismatch",
            path,
            qid,
            f"track {data.get('track')!r} does not match path track {track_dir!r}",
        )
    if data.get("competency_area") and data.get("competency_area") != area_dir:
        add_issue(
            issues,
            "warning",
            "filesystem",
            "area_path_mismatch",
            path,
            qid,
            "competency_area does not match path folder "
            f"({data.get('competency_area')!r} vs {area_dir!r})",
        )


def validate_content_rules(
    path: Path,
    data: dict[str, Any],
    issues: list[Issue],
    chain_ids: set[str],
) -> None:
    qid = data.get("id") if isinstance(data.get("id"), str) else None
    status = data.get("status")
    title = data.get("title", "")
    scenario = data.get("scenario", "")
    question = data.get("question")
    details = data.get("details") if isinstance(data.get("details"), dict) else {}

    if isinstance(title, str):
        if len(title) > 120:
            add_issue(issues, "error", "title", "title_too_long", path, qid, "title exceeds 120 chars")
        if title.endswith("."):
            add_issue(issues, "warning", "title", "title_trailing_period", path, qid, "title ends with a period")
        if "_" in title:
            add_issue(issues, "error", "title", "title_underscore", path, qid, "title contains underscore")
        if TITLE_MARKDOWN_RE.search(title):
            add_issue(issues, "error", "title", "title_markup", path, qid, "title contains markdown, HTML, LaTeX, or code markup")
        if GENERIC_TITLE_RE.match(title.strip()):
            add_issue(issues, "warning", "title", "generic_title", path, qid, "title appears generic or placeholder-like")

    if isinstance(scenario, str) and HTML_RE.search(scenario):
        add_issue(issues, "error", "content", "scenario_html", path, qid, "scenario contains HTML markup")

    if isinstance(question, str):
        if len(question) > 200:
            add_issue(issues, "warning", "content", "question_too_long", path, qid, "question exceeds 200 chars")
        if TITLE_MARKDOWN_RE.search(question):
            add_issue(issues, "warning", "content", "question_markup", path, qid, "question contains markdown, HTML, LaTeX, or code markup")
        if status == "published" and "?" not in question:
            add_issue(issues, "warning", "content", "question_not_interrogative", path, qid, "published question has no question mark")
    elif status == "published":
        add_issue(issues, "warning", "content", "missing_question", path, qid, "published question is missing the top-level question field")

    realistic_solution = details.get("realistic_solution")
    common_mistake = details.get("common_mistake", "")
    napkin_math = details.get("napkin_math", "")
    if status == "published":
        for field_name, value in (
            ("details.realistic_solution", realistic_solution),
            ("details.common_mistake", common_mistake),
            ("details.napkin_math", napkin_math),
        ):
            if not isinstance(value, str) or not value.strip():
                add_issue(
                    issues,
                    "error",
                    "content",
                    "published_missing_required_body",
                    path,
                    qid,
                    f"published question has empty {field_name}",
                )

    if isinstance(common_mistake, str) and common_mistake.strip():
        missing = ordered_markers_missing(common_mistake, COMMON_MISTAKE_MARKERS)
        if missing:
            add_issue(
                issues,
                "error" if status == "published" else "warning",
                "format",
                "common_mistake_markers",
                path,
                qid,
                "common_mistake markers missing or out of order: " + ", ".join(missing),
            )

    if isinstance(napkin_math, str) and napkin_math.strip():
        missing = ordered_markers_missing(napkin_math, NAPKIN_MATH_MARKERS)
        if missing:
            add_issue(
                issues,
                "error" if status == "published" else "warning",
                "format",
                "napkin_math_markers",
                path,
                qid,
                "napkin_math markers missing or out of order: " + ", ".join(missing),
            )
    options = details.get("options")
    correct_index = details.get("correct_index")
    if options is None and correct_index is not None:
        add_issue(issues, "error", "schema", "correct_index_without_options", path, qid, "correct_index is present but options is missing")

    if data.get("status") == "deleted" and not data.get("deletion_reason"):
        add_issue(issues, "error", "workflow", "deleted_missing_reason", path, qid, "deleted question lacks deletion_reason")
    if data.get("status") != "deleted" and data.get("deletion_reason"):
        add_issue(issues, "warning", "workflow", "deletion_reason_on_active", path, qid, "non-deleted question has deletion_reason")

    bloom = data.get("bloom_level")
    zone = data.get("zone")
    if isinstance(zone, str) and isinstance(bloom, str) and bloom:
        allowed = ZONE_BLOOM_AFFINITY.get(zone)
        if allowed and bloom not in allowed:
            add_issue(
                issues,
                "error" if status == "published" else "warning",
                "classification",
                "zone_bloom_mismatch",
                path,
                qid,
                f"zone {zone!r} does not admit bloom_level {bloom!r}",
            )

    visual = data.get("visual")
    track = data.get("track")
    if isinstance(visual, dict) and isinstance(visual.get("path"), str) and isinstance(track, str):
        visual_path = VAULT_DIR / "visuals" / track / visual["path"]
        if not visual_path.exists():
            add_issue(issues, "error", "visual", "missing_visual_asset", path, qid, f"visual asset does not exist: {rel(visual_path)}")

    chains = data.get("chains")
    if isinstance(chains, list) and chain_ids:
        for chain in chains:
            if isinstance(chain, dict) and isinstance(chain.get("id"), str) and chain["id"] not in chain_ids:
                add_issue(issues, "error", "chains", "unknown_chain_id", path, qid, f"chain id {chain['id']!r} is not present in chains.json")


def audit_file(path: Path, chain_ids: set[str]) -> tuple[dict[str, Any] | None, list[Issue]]:
    issues: list[Issue] = []
    try:
        raw = path.read_text()
    except Exception as exc:
        add_issue(issues, "error", "yaml", "read_error", path, None, str(exc))
        return None, issues

    try:
        data = yaml.safe_load(raw)
    except Exception as exc:
        add_issue(issues, "error", "yaml", "parse_error", path, None, str(exc))
        return None, issues

    if not isinstance(data, dict):
        add_issue(issues, "error", "yaml", "not_mapping", path, None, "YAML root is not a mapping")
        return None, issues

    qid = data.get("id") if isinstance(data.get("id"), str) else None
    try:
        Question(**data)
    except Exception as exc:
        add_issue(issues, "error", "schema", "pydantic_validation", path, qid, str(exc))

    validate_path_invariants(path, data, issues)
    validate_content_rules(path, data, issues, chain_ids)
    return data, issues


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary(path: Path, stats: dict[str, Any], issues: list[Issue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_severity = Counter(issue.severity for issue in issues)
    by_code = Counter(issue.code for issue in issues)
    by_category = Counter(issue.category for issue in issues)
    lines = [
        "# StaffML YAML Corpus Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        f"- Files scanned: {stats['files_scanned']}",
        f"- Parsed question records: {stats['records']}",
        f"- Unique qids: {stats['unique_qids']}",
        f"- Duplicate qids: {stats['duplicate_qids']}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(stats["status_counts"].items()):
        lines.append(f"- {status}: {count}")

    lines.extend(["", "## Issue Counts", ""])
    for severity in ("error", "warning"):
        lines.append(f"- {severity}: {by_severity.get(severity, 0)}")

    lines.extend(["", "## Issues By Category", ""])
    for category, count in by_category.most_common():
        lines.append(f"- {category}: {count}")

    lines.extend(["", "## Top Issue Codes", ""])
    for code, count in by_code.most_common(20):
        lines.append(f"- {code}: {count}")

    lines.extend(["", "## First 50 Issues", ""])
    for issue in issues[:50]:
        lines.append(
            f"- `{issue.severity}` `{issue.code}` [{issue.path}] "
            f"{issue.qid or '<unknown>'}: {issue.message}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_manifests(out_dir: Path, issues: list[Issue]) -> None:
    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        grouped[f"code-{issue.code}"].append(issue)
        grouped[f"severity-{issue.severity}"].append(issue)
        grouped[f"category-{issue.category}"].append(issue)

    def clean_cell(value: str) -> str:
        return " ".join(value.split())

    for name, group in grouped.items():
        rows = sorted(
            {
                f"{clean_cell(issue.path)}\t{clean_cell(issue.qid or '<unknown>')}\t"
                f"{clean_cell(issue.severity)}\t{clean_cell(issue.category)}\t"
                f"{clean_cell(issue.code)}\t{clean_cell(issue.message)}"
                for issue in group
            }
        )
        (manifest_dir / f"{name}.tsv").write_text("\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=VAULT_DIR / "audit" / "fresh-yaml-audit",
    )
    args = parser.parse_args()

    question_paths = sorted(args.questions_dir.glob("*/*/*.yaml"))
    chain_ids = load_chains()
    issues: list[Issue] = []
    records: list[dict[str, Any]] = []
    parsed_records: list[tuple[Path, dict[str, Any]]] = []

    for path in question_paths:
        data, file_issues = audit_file(path, chain_ids)
        issues.extend(file_issues)
        if data is not None:
            records.append(data)
            parsed_records.append((path, data))

    qid_to_paths: dict[str, list[Path]] = defaultdict(list)
    for path, data in parsed_records:
        qid = data.get("id")
        if isinstance(qid, str):
            qid_to_paths[qid].append(path)
    for qid, paths in qid_to_paths.items():
        if len(paths) > 1:
            for path in paths:
                add_issue(
                    issues,
                    "error",
                    "schema",
                    "duplicate_qid",
                    path,
                    qid,
                    f"qid appears in {len(paths)} files",
                )

    status_counts = Counter(str(record.get("status", "<missing>")) for record in records)
    stats = {
        "files_scanned": len(question_paths),
        "records": len(records),
        "unique_qids": len(qid_to_paths),
        "duplicate_qids": sum(1 for paths in qid_to_paths.values() if len(paths) > 1),
        "status_counts": dict(status_counts),
    }

    issue_rows = [asdict(issue) for issue in issues]
    write_jsonl(args.out_dir / "issues.jsonl", issue_rows)
    write_jsonl(args.out_dir / "stats.jsonl", [stats])
    write_summary(args.out_dir / "summary.md", stats, issues)
    write_manifests(args.out_dir, issues)

    print(f"Scanned {stats['files_scanned']} YAML files.")
    print(f"Issues: {Counter(issue.severity for issue in issues)}")
    print(f"Wrote {rel(args.out_dir / 'summary.md')}")
    print(f"Wrote {rel(args.out_dir / 'issues.jsonl')}")
    return 1 if any(issue.severity == "error" for issue in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
