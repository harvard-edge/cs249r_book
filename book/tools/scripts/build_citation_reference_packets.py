#!/usr/bin/env python3
"""Build chapter packets for semantic citation-reference validation.

The existing bibliography checks prove that citekeys resolve. These packets
support the harder editorial question: whether the cited source actually backs
the sentence or paragraph where it is used.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "book" / "quarto" / "contents").exists():
            return parent
    raise RuntimeError("Could not locate repository root containing book/quarto/contents")


ROOT = find_repo_root()

SCOPES = [
    {
        "name": "vol1",
        "paths": ["book/quarto/contents/vol1/"],
        "bibs": ["book/quarto/contents/references.bib"],
    },
    {
        "name": "vol2",
        "paths": ["book/quarto/contents/vol2/"],
        "bibs": ["book/quarto/contents/references.bib"],
    },
    {
        "name": "book-shared",
        "paths": [
            "book/quarto/contents/frontmatter/",
            "book/quarto/contents/backmatter/",
        ],
        "bibs": ["book/quarto/contents/references.bib"],
    },
    {
        "name": "interviews",
        "paths": ["interviews/"],
        "bibs": ["interviews/paper/references.bib"],
    },
    {
        "name": "tinytorch",
        "paths": ["tinytorch/"],
        "bibs": ["tinytorch/paper/references.bib"],
    },
    {
        "name": "mlsysim",
        "paths": ["mlsysim/"],
        "bibs": ["mlsysim/paper/references.bib", "mlsysim/docs/references.bib"],
    },
    {
        "name": "design-grammar",
        "paths": ["design-grammar/"],
        "bibs": ["design-grammar/paper/references.bib"],
    },
]

QMD_ROOTS = [
    "book/quarto/contents",
    "interviews",
    "tinytorch",
    "mlsysim",
    "design-grammar",
]

EXCLUDE_PARTS = {
    "_build",
    "_site",
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".aiconfigs-local",
}

NON_CITE_PREFIXES = (
    "sec-",
    "fig-",
    "tbl-",
    "eq-",
    "lst-",
    "exr-",
    "exm-",
    "thm-",
    "cor-",
    "cnj-",
    "def-",
    "prp-",
    "rem-",
    "prf-",
    "alg-",
    "algo-",
)

KNOWN_FALSE_POSITIVE_KEYS = {
    "media",
    "keyframes",
    "import",
    "supports",
    "page",
    "font-face",
    "charset",
    "namespace",
    "document",
    "grad",
    "staticmethod",
    "classmethod",
    "property",
    "abstractmethod",
    "dataclass",
    "cached_property",
    "wraps",
    "eecs",
}

CITE_RE = re.compile(r"(?<![=,(])\[?@([A-Za-z][\w:.-]*)\b")
ENTRY_START_RE = re.compile(r"^@(?P<entry_type>\w+)\s*\{\s*(?P<key>[\w:.-]+)\s*,", re.M)
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SINGLE_LETTER_RE = re.compile(r"^[A-Z](\.[A-Z])*$")

FIELDS_TO_SUMMARIZE = (
    "author",
    "title",
    "booktitle",
    "journal",
    "publisher",
    "institution",
    "year",
    "pages",
    "doi",
    "url",
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def find_scope(rel_path: str) -> dict[str, Any] | None:
    for scope in SCOPES:
        if any(rel_path.startswith(prefix) for prefix in scope["paths"]):
            return scope
    return None


def discover_qmds(all_qmd: bool) -> list[Path]:
    roots = [ROOT] if all_qmd else [ROOT / root for root in QMD_ROOTS]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.qmd"):
            if any(part in EXCLUDE_PARTS for part in path.parts):
                continue
            if path.name.startswith("_shelved"):
                continue
            if not all_qmd and find_scope(rel(path)) is None:
                continue
            files.append(path)
    return sorted(set(files))


def strip_inline_protected(line: str) -> str:
    line = re.sub(r"`[^`]*`", "", line)
    line = re.sub(r"<!--.*?-->", "", line)
    return line


def should_skip_key(key: str) -> bool:
    key = key.rstrip(".,;:)")
    lower_key = key.lower()
    return (
        not key
        or lower_key.startswith(NON_CITE_PREFIXES)
        or lower_key in KNOWN_FALSE_POSITIVE_KEYS
        or bool(SINGLE_LETTER_RE.match(key))
        or bool(re.match(r"^\d+\.\d+", key))
    )


def paragraph_at(lines: list[str], line_no: int, max_chars: int) -> str:
    idx = max(0, min(line_no - 1, len(lines) - 1))
    start = idx
    while start > 0 and lines[start - 1].strip():
        start -= 1
    end = idx
    while end + 1 < len(lines) and lines[end + 1].strip():
        end += 1
    text = " ".join(line.strip() for line in lines[start : end + 1]).strip()
    return text[:max_chars]


def citation_occurrences(path: Path, max_context_chars: int) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    occurrences: list[dict[str, Any]] = []
    in_yaml = bool(lines and lines[0].strip() == "---")
    in_fence = False
    in_html_comment = False
    in_raw_block: str | None = None

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()

        if line_no > 1 and in_yaml and stripped == "---":
            in_yaml = False
            continue
        if in_yaml:
            continue

        if in_html_comment:
            if "-->" in line:
                in_html_comment = False
            continue
        if "<!--" in line and "-->" not in line:
            in_html_comment = True
            continue

        if in_raw_block:
            if f"</{in_raw_block}>" in line.lower():
                in_raw_block = None
            continue
        lower = line.lower()
        if "<style" in lower and "</style>" not in lower:
            in_raw_block = "style"
            continue
        if "<script" in lower and "</script>" not in lower:
            in_raw_block = "script"
            continue

        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        scan_line = strip_inline_protected(line)
        for match in CITE_RE.finditer(scan_line):
            key = match.group(1).rstrip(".,;:)")
            if should_skip_key(key):
                continue
            occurrences.append(
                {
                    "occurrence_id": f"{rel(path)}:{line_no}:{match.start()}:{key}",
                    "key": key,
                    "line": line_no,
                    "column": match.start() + 1,
                    "line_text": line.strip(),
                    "context": paragraph_at(lines, line_no, max_context_chars),
                }
            )

    return occurrences


def extract_bib_value(entry_text: str, field: str) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(field)}\s*=\s*([{{\"])", entry_text)
    if not match:
        return None

    opener = match.group(1)
    start = match.end()
    if opener == '"':
        end = start
        escaped = False
        while end < len(entry_text):
            char = entry_text[end]
            if char == '"' and not escaped:
                break
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
            end += 1
        raw = entry_text[start:end]
    else:
        depth = 1
        end = start
        while end < len(entry_text) and depth:
            char = entry_text[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            end += 1
        raw = entry_text[start : max(start, end - 1)]

    raw = re.sub(r"\s+", " ", raw).strip()
    return raw.replace("{", "").replace("}", "") or None


def parse_bib_entries(bib_paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bib_rel in bib_paths:
        bib_path = ROOT / bib_rel
        if not bib_path.exists():
            continue
        text = bib_path.read_text(encoding="utf-8", errors="replace")
        starts = list(ENTRY_START_RE.finditer(text))
        for i, match in enumerate(starts):
            start = match.start()
            end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
            entry_text = text[start:end].strip()
            key = match.group("key")
            fields = {
                field: value
                for field in FIELDS_TO_SUMMARIZE
                if (value := extract_bib_value(entry_text, field)) is not None
            }
            by_key[key].append(
                {
                    "key": key,
                    "entry_type": match.group("entry_type"),
                    "defined_in": bib_rel,
                    "fields": fields,
                    "bibtex": entry_text,
                }
            )
    return by_key


def slug(value: str) -> str:
    return value.strip("/").replace("/", "__").replace(".", "_")


def qmd_packet_name(rel_path: str) -> str:
    name = rel_path.removesuffix(".qmd").replace("/", "__")
    return f"{name}.citation-packet.json"


def audit_unit_for_path(rel_path: str, granularity: str) -> dict[str, str]:
    if granularity == "qmd":
        unit_id = rel_path.removesuffix(".qmd")
        return {
            "id": unit_id,
            "label": rel_path,
            "root": rel_path,
            "kind": "qmd",
        }

    parts = rel_path.split("/")
    if parts[:3] == ["book", "quarto", "contents"] and len(parts) >= 5:
        if parts[3] in {"vol1", "vol2"}:
            volume = parts[3]
            section = parts[4].removesuffix(".qmd")

            if section in {"frontmatter", "backmatter", "parts"}:
                leaf = parts[-1].removesuffix(".qmd")
                if parts[-1].startswith("_"):
                    leaf = f"includes/{leaf}"
                elif section == "backmatter" and len(parts) >= 7:
                    leaf = parts[5]
                unit_id = f"{volume}/{section}/{leaf}"
                return {
                    "id": unit_id,
                    "label": unit_id,
                    "root": rel_path,
                    "kind": "book-section",
                }

            root = f"book/quarto/contents/{volume}/{section}/"
            return {
                "id": f"{volume}/{section}",
                "label": f"{volume}/{section}",
                "root": root,
                "kind": "book-chapter",
            }
        if parts[3] in {"frontmatter", "backmatter"}:
            section = parts[3]
            chapter = parts[4].removesuffix(".qmd")
            root = f"book/quarto/contents/{section}/{parts[4]}"
            if not parts[4].endswith(".qmd"):
                root += "/"
            return {
                "id": f"book-shared/{section}/{chapter}",
                "label": f"book-shared/{section}/{chapter}",
                "root": root,
                "kind": "book-shared-section",
            }

    unit_id = rel_path.removesuffix(".qmd")
    return {
        "id": unit_id,
        "label": rel_path,
        "root": rel_path,
        "kind": "qmd",
    }


def packet_name_for_unit(unit: dict[str, str], granularity: str) -> str:
    if granularity == "qmd":
        return qmd_packet_name(unit["root"])
    return f"{slug(unit['id'])}.citation-packet.json"


def build_packets(out_dir: Path, *, all_qmd: bool, granularity: str, max_context_chars: int) -> dict[str, Any]:
    all_bibs = sorted({bib for scope in SCOPES for bib in scope["bibs"]})
    bib_entries = parse_bib_entries(all_bibs)
    packet_dir = out_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    for stale_packet in packet_dir.glob("*.citation-packet.json"):
        stale_packet.unlink()

    grouped_qmds: dict[str, dict[str, Any]] = {}
    for qmd in discover_qmds(all_qmd):
        rel_path = rel(qmd)
        unit = audit_unit_for_path(rel_path, granularity)
        grouped_qmds.setdefault(unit["id"], {"audit_unit": unit, "qmds": []})["qmds"].append(qmd)

    summary_packets: list[dict[str, Any]] = []
    total_occurrences = 0
    total_source_files = 0
    all_keys: set[str] = set()

    for unit_id, group in sorted(grouped_qmds.items()):
        unit = group["audit_unit"]
        citations: list[dict[str, Any]] = []
        source_files: list[dict[str, Any]] = []
        packet_allowed_bibs: set[str] = set()

        for qmd in sorted(group["qmds"]):
            rel_path = rel(qmd)
            occurrences = citation_occurrences(qmd, max_context_chars)
            if not occurrences:
                continue

            scope = find_scope(rel_path)
            allowed_bibs = set(scope["bibs"] if scope else all_bibs)
            packet_allowed_bibs.update(allowed_bibs)
            for occurrence in occurrences:
                key = occurrence["key"]
                entries = bib_entries.get(key, [])
                scoped_entries = [entry for entry in entries if entry["defined_in"] in allowed_bibs]
                selected_entries = scoped_entries or entries
                occurrence["source_file"] = rel_path
                occurrence["source_scope"] = scope["name"] if scope else "unscoped"
                occurrence["allowed_bibliographies"] = sorted(allowed_bibs)
                occurrence["reference_status"] = "resolved" if selected_entries else "missing"
                occurrence["reference_entries"] = selected_entries
                citations.append(occurrence)
                all_keys.add(key)

            source_files.append(
                {
                    "path": rel_path,
                    "scope": scope["name"] if scope else "unscoped",
                    "citation_occurrences": len(occurrences),
                    "unique_citekeys": sorted({item["key"] for item in occurrences}),
                }
            )

        if not citations:
            continue

        packet = {
            "schema_version": "citation-reference-validation-packet/v1",
            "packet_granularity": granularity,
            "audit_unit": unit,
            "source_files": source_files,
            "allowed_bibliographies": sorted(packet_allowed_bibs),
            "citation_occurrences": len(citations),
            "unique_citekeys": sorted({item["key"] for item in citations}),
            "instructions": [
                "Validate only the citation usages in this packet; this packet is one audit unit for one subagent.",
                "Core question: does the cited source support the local claim the citation is attached to?",
                "Use the embedded BibTeX first, but inspect the canonical source when metadata is not enough.",
                "Canonical source checks may require DOI/publisher pages, arXiv, OpenReview, official docs, project pages, or web search.",
                "Search-result snippets are not evidence; read enough of the source to judge the claim.",
                "Do not require one citation to support unrelated nearby prose.",
                "Do not flag broad synthesis just because the source does not use identical wording.",
                "Flag only high-confidence unsupported, overbroad, stale, misplaced, or source-access issues.",
                "Return only actionable issues plus a concise coverage summary.",
            ],
            "expected_finding_schema": {
                "key": "citekey",
                "source_file": "repo-relative qmd path",
                "line": "source line number",
                "status": "unsupported | overbroad | misplaced | stale | needs_human | missing_canonical | bib_metadata",
                "claim": "short paraphrase of what the local text asks the source to support",
                "reason": "why the source does or does not support the claim",
                "suggested_fix": "replace citation | add source | narrow claim | move citation | remove citation | fix metadata",
                "evidence": "brief source evidence, external source URL/identifier, or note that human review is needed",
            },
            "citations": citations,
        }
        packet_path = packet_dir / packet_name_for_unit(unit, granularity)
        packet_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        total_occurrences += len(citations)
        total_source_files += len(source_files)
        summary_packets.append(
            {
                "audit_unit": unit,
                "packet": display_path(packet_path),
                "source_files": [item["path"] for item in source_files],
                "citation_occurrences": len(citations),
                "unique_citekeys": len(packet["unique_citekeys"]),
                "allowed_bibliographies": sorted(packet_allowed_bibs),
            }
        )

    summary = {
        "schema_version": "citation-reference-validation-summary/v1",
        "packet_granularity": granularity,
        "audit_units": len(summary_packets),
        "source_files": total_source_files,
        "citation_occurrences": total_occurrences,
        "unique_citekeys": len(all_keys),
        "packet_dir": display_path(packet_dir),
        "packets": summary_packets,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_agent_manifest(out_dir, summary)
    return summary


def write_agent_manifest(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Citation Reference Audit Agent Manifest",
        "",
        "Spawn one subagent per packet. Each subagent owns the whole audit unit",
        "listed for its packet and should not edit source files.",
        "",
        "## Subagent Task",
        "",
        "Audit the assigned chapter packet. For every citation occurrence, answer:",
        "does the cited source support the local claim the citation is attached to?",
        "",
        "Use the embedded BibTeX metadata first. If that is not enough, search for",
        "and inspect the canonical source: DOI or publisher page, arXiv, OpenReview,",
        "DBLP plus linked paper, official documentation, project page, standards",
        "document, or another primary source. Search-result snippets are not",
        "evidence. Read enough of the paper/source to judge the claim.",
        "",
        "Return only actionable findings plus a concise coverage summary. Do not",
        "flag a citation merely because it does not support unrelated nearby prose",
        "or because the source uses different wording. If the source cannot be",
        "inspected or the fit remains ambiguous after reasonable source lookup,",
        "return `needs_human` and say exactly what must be checked.",
        "",
        "Finding schema:",
        "",
        "```json",
        "{",
        '  "key": "citekey",',
        '  "source_file": "repo-relative qmd path",',
        '  "line": 123,',
        '  "status": "unsupported | overbroad | misplaced | stale | needs_human | missing_canonical | bib_metadata",',
        '  "claim": "short paraphrase of the local claim",',
        '  "reason": "why the cited source does or does not support it",',
        '  "suggested_fix": "replace citation | add source | narrow claim | move citation | remove citation | fix metadata",',
        '  "evidence": "brief source evidence and URL/identifier, or what needs human review"',
        "}",
        "```",
        "",
        "## Packets",
        "",
    ]

    for packet in summary["packets"]:
        unit = packet["audit_unit"]
        lines.extend(
            [
                f"- `{packet['packet']}`",
                f"  - audit unit: `{unit['label']}` ({unit['kind']})",
                f"  - citations: {packet['citation_occurrences']}",
                f"  - source files: {len(packet['source_files'])}",
                f"  - bibliographies: {', '.join(f'`{bib}`' for bib in packet['allowed_bibliographies'])}",
            ]
        )

    (out_dir / "agent_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build citation-reference validation packets.")
    parser.add_argument(
        "--out-dir",
        default="review/citation-reference-validation",
        help="Output directory, relative to repository root or absolute.",
    )
    parser.add_argument(
        "--all-qmd",
        action="store_true",
        help="Scan every .qmd in the repository instead of the scoped bibliography trees.",
    )
    parser.add_argument(
        "--granularity",
        choices=("chapter", "qmd"),
        default="chapter",
        help="Packet sharding strategy. Default: chapter.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=1800,
        help="Maximum paragraph context characters per citation occurrence.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_packets(
        out_dir,
        all_qmd=args.all_qmd,
        granularity=args.granularity,
        max_context_chars=args.max_context_chars,
    )
    print(
        "Wrote "
        f"{summary['audit_units']} {summary['packet_granularity']} packets covering "
        f"{summary['source_files']} source files with "
        f"{summary['citation_occurrences']} citation occurrences to "
        f"{Path(summary['packet_dir'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
