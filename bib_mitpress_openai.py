#!/usr/bin/env python3
"""Standalone MIT Press BibTeX validator for MLSysBook.

This is the single command-line surface for the MIT Press bibliography sweep.
It validates BibTeX entries with the repo's canonical parser/rules in
``book/tools/bib_lint.py`` and adds a small MIT Press production layer for
fields that should not ship in the final bibliography.

OpenAI is optional and only runs when ``--smart-fix`` is passed. The normal
``--check`` path is deterministic and does not call the network or modify files.

Usage:
    python3 bib_mitpress_openai.py --check book/quarto/contents/vol1/backmatter/references.bib
    python3 bib_mitpress_openai.py --all --check
    python3 bib_mitpress_openai.py --fix path/to/references.bib
    python3 bib_mitpress_openai.py --smart-fix path/to/references.bib --limit 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
BIB_LINT_PATH = REPO_ROOT / "book" / "tools" / "bib_lint.py"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REPORT_DIR = REPO_ROOT / "book" / "tools" / "audit" / "out"

EXCLUDE_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "_build",
    "build",
    "dist",
    "node_modules",
    "venv",
}

MITPRESS_FORBIDDEN_FIELDS = {
    "abstract": "Drop abstracts from production bibliography entries.",
    "address": "MIT Press bibliography style does not include publisher locations.",
    "file": "Local file attachments must not appear in production bibliography entries.",
    "keywords": "Keyword metadata must not appear in production bibliography entries.",
    "location": "MIT Press bibliography style does not include publisher locations.",
    "month": "Use year/date metadata consistently; month fields add copyedit noise.",
    "organization": "Use publisher instead of organization for conference proceedings.",
    "timestamp": "Tool timestamps must not appear in production bibliography entries.",
}


@dataclass
class Finding:
    file: str
    key: str
    line: int
    severity: str
    rule: str
    message: str

    def to_json(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "key": self.key,
            "line": self.line,
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
        }


@dataclass
class FileResult:
    path: Path
    entries: int = 0
    parsed_entries: list[Any] = field(default_factory=list, repr=False)
    findings: list[Finding] = field(default_factory=list)
    parse_error: str = ""
    changed: bool = False

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "error") + int(bool(self.parse_error))

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "info")


def load_bib_lint() -> Any:
    """Load the repo parser/validator without requiring package installation."""
    if not BIB_LINT_PATH.exists():
        raise RuntimeError(f"Cannot find {BIB_LINT_PATH}")
    module_name = "_mlsysbook_bib_lint"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, BIB_LINT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BIB_LINT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def short_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover_bib_files() -> list[Path]:
    """Find production-facing .bib files while skipping caches/build output."""
    files = []
    for path in REPO_ROOT.rglob("*.bib"):
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def resolve_files(args: argparse.Namespace) -> list[Path]:
    aliases = {
        "vol1": REPO_ROOT / "book/quarto/contents/vol1/backmatter/references.bib",
        "vol2": REPO_ROOT / "book/quarto/contents/vol2/backmatter/references.bib",
    }
    if args.all:
        return discover_bib_files()
    files = []
    for item in args.files:
        path = aliases.get(item)
        if path is None:
            path = Path(item)
            if not path.is_absolute():
                path = REPO_ROOT / path
        files.append(path)
    return files


def normalize_doi(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I)
    value = re.sub(r"^doi:\s*", "", value, flags=re.I)
    return value.strip()


def entry_field(entry: Any, name: str) -> Any | None:
    return entry.get(name)


def extra_mitpress_findings(path: Path, entry: Any) -> list[Finding]:
    """Checks that are specific to the MIT Press production pass."""
    findings: list[Finding] = []
    rel = short_path(path)

    for field_obj in entry.fields:
        name = field_obj.name.lower()
        value = field_obj.value.strip()
        if name in MITPRESS_FORBIDDEN_FIELDS:
            severity = "error" if name in {"address", "location", "organization"} else "warning"
            findings.append(Finding(
                rel,
                entry.key,
                entry.start_line,
                severity,
                "mitpress-forbidden-field",
                f"`{name}`: {MITPRESS_FORBIDDEN_FIELDS[name]}",
            ))

        if name == "doi" and value != normalize_doi(value):
            findings.append(Finding(
                rel,
                entry.key,
                entry.start_line,
                "error",
                "doi-not-bare",
                "DOI must be a bare identifier, not a doi.org URL.",
            ))

        if name == "author" and re.search(r"\bet\s+al\.?\b", value, re.I):
            findings.append(Finding(
                rel,
                entry.key,
                entry.start_line,
                "error",
                "author-et-al",
                "Author lists must be expanded in .bib entries; do not use et al.",
            ))

        if name == "pages":
            if re.search(r"\b\d{2,}--\d{1,2}\b", value):
                findings.append(Finding(
                    rel,
                    entry.key,
                    entry.start_line,
                    "warning",
                    "short-page-range",
                    "Page ranges should use all digits, for example 175--185.",
                ))
            if "-" in value and "--" not in value:
                findings.append(Finding(
                    rel,
                    entry.key,
                    entry.start_line,
                    "warning",
                    "single-dash-pages",
                    "Page ranges should use BibTeX double dash, for example 175--185.",
                ))

        if "%" in value:
            findings.append(Finding(
                rel,
                entry.key,
                entry.start_line,
                "warning",
                "percent-symbol",
                "Spell out percent in bibliography prose fields.",
            ))

    verified_by = entry_field(entry, "x-verified-by")
    verified = entry_field(entry, "x-verified")
    source = entry_field(entry, "x-verified-source")
    if verified_by or verified or source:
        missing = []
        for stamp_name in ("x-verified", "x-verified-by", "x-verified-source"):
            if entry_field(entry, stamp_name) is None:
                missing.append(stamp_name)
        if missing:
            findings.append(Finding(
                rel,
                entry.key,
                entry.start_line,
                "warning",
                "incomplete-verification-stamp",
                "Verification stamp is incomplete; missing " + ", ".join(missing) + ".",
            ))

    return findings


def lint_findings(path: Path, entry: Any, bib_lint: Any) -> list[Finding]:
    rel = short_path(path)
    findings = []
    for violation in bib_lint.validate_entry(entry):
        findings.append(Finding(
            rel,
            violation.entry_key,
            violation.entry_line,
            violation.severity,
            violation.rule,
            violation.message,
        ))
    findings.extend(extra_mitpress_findings(path, entry))
    return findings


def apply_mechanical_fixes(text: str, bib_lint: Any) -> str:
    """Use bib_lint formatting, then apply tiny value-normalization fixes."""
    formatted, _ = bib_lint.format_file(text)
    entries, preamble = bib_lint.parse_bib(formatted)
    changed = False

    for entry in entries:
        for field_obj in entry.fields:
            name = field_obj.name.lower()
            value = field_obj.value
            fixed = value
            if name == "doi":
                fixed = normalize_doi(fixed)
            if name == "pages":
                fixed = re.sub(r"(?<=\d)-(?=\d)", "--", fixed)
            if fixed != value:
                field_obj.value = fixed
                changed = True

    if not changed:
        return formatted

    output: list[str] = []
    if preamble and preamble[0].strip():
        output.append(preamble[0].rstrip() + "\n\n")
    for entry in entries:
        output.append(bib_lint.format_entry(entry))
        output.append("\n\n")
    return "".join(output).rstrip() + "\n"


def validate_file(path: Path, bib_lint: Any, fix: bool = False) -> FileResult:
    result = FileResult(path=path)
    if not path.exists():
        result.parse_error = "file not found"
        return result

    text = path.read_text(encoding="utf-8")
    try:
        entries, _ = bib_lint.parse_bib(text)
    except Exception as exc:
        result.parse_error = str(exc)
        return result

    result.entries = len(entries)
    result.parsed_entries = entries
    for entry in entries:
        result.findings.extend(lint_findings(path, entry, bib_lint))

    if fix:
        new_text = apply_mechanical_fixes(text, bib_lint)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            result.changed = True

    return result


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


SMART_FIX_RULES = {
    "missing-required-field",
    "article-missing-volume",
    "abbreviated-journal",
    "author-initials-only",
    "mitpress-forbidden-field",
    "doi-not-bare",
    "author-et-al",
}


def process_file(path: Path, bib_lint: Any, fix: bool) -> FileResult:
    return validate_file(path, bib_lint, fix=fix)


def build_smart_fix_prompt(path: Path, findings: list[Finding], entries: list[Any]) -> str:

    return f"""You are repairing MIT Press bibliography metadata for MLSysBook.

Anti-hallucination contract:
- Use canonical publication pages only: DOI landing pages, Crossref, DBLP, arXiv, ACM, IEEE, USENIX, PMLR, OpenReview, publisher pages.
- Do not use Google Scholar or search-result snippets as evidence.
- Return a field only if you can provide a source URL for it.
- Do not rename citekeys.
- Return only JSON.

File: {short_path(path)}
Findings:
{json.dumps([finding.to_json() for finding in findings], indent=2)}

Entries:
{"\n\n".join(entry.raw.strip() for entry in entries)}

Return this JSON shape:
{{
  "fixes": [
    {{
      "key": "citekey",
      "fields": {{
        "field-name": "verified BibTeX value",
        "x-verified": "{datetime.now().strftime('%Y-%m-%d')}",
        "x-verified-by": "openai-MODEL",
        "x-verified-source": "source URL"
      }},
      "evidence": "short quote or exact source fact"
    }}
  ]
}}
"""


def run_smart_fix_batch(path: Path, findings: list[Finding], entries: list[Any], model: str) -> tuple[Path, dict[str, Any] | None]:
    prompt = build_smart_fix_prompt(path, findings, entries)
    return path, call_openai(prompt, model)


def call_openai(prompt: str, model: str) -> dict[str, Any] | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return None
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"Could not import openai client: {exc}", file=sys.stderr)
        return None

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return only JSON. Follow the user's formatting contract exactly."},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:
        print(f"OpenAI request failed: {exc}", file=sys.stderr)
        return None

    text = (response.choices[0].message.content or "").strip()
    first = text.find("{")
    last = text.rfind("}")
    if first < 0 or last <= first:
        print("OpenAI returned no JSON object.", file=sys.stderr)
        return None
    try:
        return json.loads(text[first:last + 1])
    except json.JSONDecodeError as exc:
        print(f"OpenAI JSON parse failed: {exc}", file=sys.stderr)
        return None


def apply_smart_fixes(path: Path, payloads: list[dict[str, Any]], bib_lint: Any, model: str) -> int:
    text = path.read_text(encoding="utf-8")
    entries, _ = bib_lint.parse_bib(text)
    entry_types = {entry.key: entry.entry_type for entry in entries}
    applied = 0

    for payload in payloads:
        fixes = payload.get("fixes", [])
        if not isinstance(fixes, list):
            continue
        for fix in fixes:
            key = fix.get("key")
            fields = fix.get("fields") or {}
            evidence = fix.get("evidence", "")
            if not key or key not in entry_types or not isinstance(fields, dict):
                continue
            if not fields.get("x-verified-source") or not evidence:
                print(f"Skipping {key}: missing source/evidence.", file=sys.stderr)
                continue
            fields["x-verified-by"] = fields.get("x-verified-by") or f"openai-{model}"
            fields["x-verified"] = fields.get("x-verified") or datetime.now().strftime("%Y-%m-%d")
            new_fields = [(str(name), str(value)) for name, value in fields.items()]
            text = bib_lint.apply_fields(
                text,
                key,
                entry_types[key],
                new_fields,
                replace_existing=True,
            )
            applied += 1

    if applied:
        path.write_text(text, encoding="utf-8")
    return applied


def print_result(result: FileResult, verbose: bool = False) -> None:
    label = short_path(result.path)
    if result.parse_error:
        print(f"{label}: PARSE ERROR: {result.parse_error}")
        return
    status = "OK" if result.error_count == 0 and result.warning_count == 0 else "ISSUES"
    changed = " changed" if result.changed else ""
    print(
        f"{label}: {status} entries={result.entries} "
        f"errors={result.error_count} warnings={result.warning_count} info={result.info_count}{changed}"
    )
    for finding in result.findings:
        if not verbose and finding.severity == "info":
            continue
        print(
            f"  {finding.severity.upper():7} line {finding.line:<5} "
            f"{finding.key}: {finding.rule}: {finding.message}"
        )


def write_report(results: list[FileResult], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"bib_mitpress_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "files": [
            {
                "file": short_path(result.path),
                "entries": result.entries,
                "parse_error": result.parse_error,
                "changed": result.changed,
                "errors": result.error_count,
                "warnings": result.warning_count,
                "info": result.info_count,
                "findings": [finding.to_json() for finding in result.findings],
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", help=".bib files or aliases: vol1, vol2")
    parser.add_argument("--all", action="store_true", help="Validate every .bib file in the repo")
    parser.add_argument("--check", action="store_true", help="Validate only; do not edit files")
    parser.add_argument("--fix", action="store_true", help="Apply deterministic mechanical fixes")
    parser.add_argument("--smart-fix", action="store_true", help="Ask OpenAI for source-backed metadata fixes")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model for --smart-fix (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=0, help="Limit smart-fix entries sent per file to OpenAI")
    parser.add_argument("--chunk-size", type=int, default=8, help="Number of entries per smart-fix chunk")
    parser.add_argument("--json", action="store_true", help="Write a JSON report under book/tools/audit/out")
    parser.add_argument("--verbose", action="store_true", help="Include info-level findings in console output")
    parser.add_argument("--max-parallel", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Maximum number of files to process in parallel")
    args = parser.parse_args()

    if args.fix and args.check:
        parser.error("--fix and --check are mutually exclusive")
    if args.smart_fix and not args.files and not args.all:
        parser.error("--smart-fix requires files or --all")
    if not args.files and not args.all:
        parser.error("provide one or more .bib files, aliases, or --all")

    bib_lint = load_bib_lint()
    files = resolve_files(args)
    missing = [path for path in files if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing file: {path}", file=sys.stderr)
        return 2

    results: list[FileResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        futures = [
            executor.submit(process_file, path, bib_lint, args.fix)
            for path in files
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: short_path(r.path))
    for result in results:
        print_result(result, verbose=args.verbose)

    if args.smart_fix:
        smart_tasks: list[tuple[Path, list[Finding], list[Any]]] = []
        for result in results:
            if result.parse_error or not result.parsed_entries:
                continue
            target_keys = sorted({
                finding.key
                for finding in result.findings
                if finding.rule in SMART_FIX_RULES
            })
            if not target_keys:
                continue
            entries_by_key = {entry.key: entry for entry in result.parsed_entries}
            target_entries = [entries_by_key[key] for key in target_keys if key in entries_by_key]
            if args.limit:
                target_entries = target_entries[: args.limit]
            if not target_entries:
                continue
            for entry_batch in chunked(target_entries, max(1, args.chunk_size)):
                batch_keys = {entry.key for entry in entry_batch}
                batch_findings = [finding for finding in result.findings if finding.key in batch_keys]
                smart_tasks.append((result.path, batch_findings, entry_batch))

        payloads_by_path: dict[Path, list[dict[str, Any]]] = {}
        if smart_tasks:
            with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
                futures = [
                    executor.submit(run_smart_fix_batch, path, findings, entries, args.model)
                    for path, findings, entries in smart_tasks
                ]
                for future in as_completed(futures):
                    path, payload = future.result()
                    if payload:
                        payloads_by_path.setdefault(path, []).append(payload)

        for result in results:
            payloads = payloads_by_path.get(result.path, [])
            if payloads:
                applied = apply_smart_fixes(result.path, payloads, bib_lint, args.model)
                if applied:
                    result.changed = True
                    print(f"{short_path(result.path)}: smart-fix applied {applied} entries")

    if args.json:
        report = write_report(results, DEFAULT_REPORT_DIR)
        print(f"JSON report: {short_path(report)}")

    total_errors = sum(result.error_count for result in results)
    total_warnings = sum(result.warning_count for result in results)
    print(f"\nSummary: files={len(results)} errors={total_errors} warnings={total_warnings}")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
