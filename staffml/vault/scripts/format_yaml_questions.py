#!/usr/bin/env python3
"""Canonical formatter for StaffML question YAML files."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = VAULT_DIR.parents[1]
QUESTIONS_DIR = VAULT_DIR / "questions"

TOP_LEVEL_ORDER = [
    "schema_version",
    "id",
    "track",
    "level",
    "zone",
    "topic",
    "competency_area",
    "bloom_level",
    "phase",
    "title",
    "scenario",
    "question",
    "visual",
    "details",
    "status",
    "provenance",
    "requires_explanation",
    "expected_time_minutes",
    "deletion_reason",
    "chains",
    "validated",
    "validation_status",
    "validation_date",
    "validation_model",
    "validation_issues",
    "validation_status_pro",
    "validation_issues_pro",
    "math_verified",
    "math_status",
    "math_date",
    "math_model",
    "math_issues",
    "human_reviewed",
    "classification_review",
    "tags",
    "created_at",
    "updated_at",
    "last_modified",
]

DETAILS_ORDER = [
    "realistic_solution",
    "common_mistake",
    "napkin_math",
    "options",
    "correct_index",
    "resources",
]

VISUAL_ORDER = ["kind", "path", "alt", "caption"]
RESOURCE_ORDER = ["name", "url"]
HUMAN_REVIEW_ORDER = ["status", "by", "date", "notes"]
CHAIN_ORDER = ["id", "position"]


class LiteralString(str):
    pass


class StaffMLDumper(yaml.SafeDumper):
    pass


def str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.nodes.ScalarNode:
    if isinstance(data, LiteralString):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def increase_indent(self: StaffMLDumper, flow: bool = False, indentless: bool = False) -> Any:
    return yaml.SafeDumper.increase_indent(self, flow, False)


StaffMLDumper.add_representer(str, str_representer)
StaffMLDumper.add_representer(LiteralString, str_representer)
StaffMLDumper.add_multi_representer(LiteralString, str_representer)
StaffMLDumper.increase_indent = increase_indent  # type: ignore[method-assign]


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def ordered_mapping(data: dict[str, Any], order: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in order:
        if key in data:
            result[key] = data[key]
    for key in sorted(k for k in data if k not in result):
        result[key] = data[key]
    return result


def literal_if_multiline(value: Any) -> Any:
    if isinstance(value, str) and "\n" in value:
        return LiteralString(value.rstrip() + "\n")
    return value


def normalize_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_nested(literal_if_multiline(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_nested(literal_if_multiline(v)) for v in value]
    return literal_if_multiline(value)


def normalize_question(data: dict[str, Any]) -> dict[str, Any]:
    normalized = ordered_mapping(dict(data), TOP_LEVEL_ORDER)

    details = normalized.get("details")
    if isinstance(details, dict):
        details = ordered_mapping(details, DETAILS_ORDER)
        for key in ("common_mistake", "napkin_math"):
            if isinstance(details.get(key), str) and details[key].strip():
                details[key] = LiteralString(details[key].rstrip() + "\n")
        normalized["details"] = details

    visual = normalized.get("visual")
    if isinstance(visual, dict):
        normalized["visual"] = ordered_mapping(visual, VISUAL_ORDER)

    human_reviewed = normalized.get("human_reviewed")
    if isinstance(human_reviewed, dict):
        normalized["human_reviewed"] = ordered_mapping(human_reviewed, HUMAN_REVIEW_ORDER)

    chains = normalized.get("chains")
    if isinstance(chains, list):
        normalized["chains"] = [
            ordered_mapping(item, CHAIN_ORDER) if isinstance(item, dict) else item
            for item in chains
        ]

    details = normalized.get("details")
    if isinstance(details, dict) and isinstance(details.get("resources"), list):
        details["resources"] = [
            ordered_mapping(item, RESOURCE_ORDER) if isinstance(item, dict) else item
            for item in details["resources"]
        ]

    return normalize_nested(normalized)


def dump_question(data: dict[str, Any]) -> str:
    text = yaml.dump(
        normalize_question(data),
        Dumper=StaffMLDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=1000,
    )
    return text.rstrip() + "\n"


def format_file(path: Path) -> tuple[bool, str | None]:
    original = path.read_text()
    try:
        data = yaml.safe_load(original)
    except Exception as exc:
        return False, f"parse error: {exc}"
    if not isinstance(data, dict):
        return False, "YAML root is not a mapping"
    formatted = dump_question(data)
    return original == formatted, formatted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Specific YAML files to format")
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument("--write", action="store_true", help="Rewrite files in canonical form")
    parser.add_argument("--diff", action="store_true", help="Print unified diffs for noncanonical files")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files processed, useful for dry runs")
    args = parser.parse_args()

    if args.paths:
        paths = [path if path.is_absolute() else Path.cwd() / path for path in args.paths]
    else:
        paths = sorted(args.questions_dir.glob("*/*/*.yaml"))
    if args.max_files is not None:
        paths = paths[: args.max_files]

    changed: list[Path] = []
    errors: list[tuple[Path, str]] = []

    for path in paths:
        ok, result = format_file(path)
        if result is None or result.startswith("parse error:") or result == "YAML root is not a mapping":
            errors.append((path, result or "unknown error"))
            continue
        if ok:
            continue
        changed.append(path)
        if args.diff:
            original = path.read_text().splitlines(keepends=True)
            formatted = result.splitlines(keepends=True)
            sys.stdout.writelines(
                difflib.unified_diff(
                    original,
                    formatted,
                    fromfile=rel(path),
                    tofile=rel(path) + " (formatted)",
                )
            )
        if args.write:
            path.write_text(result)

    for path, message in errors:
        print(f"ERROR {rel(path)}: {message}", file=sys.stderr)

    if changed:
        action = "Reformatted" if args.write else "Would reformat"
        print(f"{action} {len(changed)} file(s).")
        for path in changed[:50]:
            print(f"- {rel(path)}")
        if len(changed) > 50:
            print(f"... and {len(changed) - 50} more")
    else:
        print("All checked YAML files are canonical.")

    return 1 if errors or (changed and not args.write) else 0


if __name__ == "__main__":
    raise SystemExit(main())
