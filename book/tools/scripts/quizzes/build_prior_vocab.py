#!/usr/bin/env python3
"""Build the cumulative prior-vocabulary context for a target chapter.

Walks the Vol1 → Vol2 reading order and, for chapter N, produces the
union of every definition term first introduced in chapters 1..N-1. The
resulting JSON is passed to that chapter's quiz-generation sub-agent so
it knows which terms it may assume vs. which it must treat as novel to
this chapter.

Usage
-----
    python3 build_prior_vocab.py vol1 training
    python3 build_prior_vocab.py vol2 distributed_training > _context/vol2/distributed_training/prior_vocab.json

The reading order is extracted from
``book/quarto/config/_quarto-html-vol{1,2}.yml`` and must match the
actual sidebar sequence; to update, rerun ``extract_reading_order.sh``
(or edit :data:`READING_ORDER` below by hand).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Source: extracted from book/quarto/config/_quarto-html-vol{1,2}.yml on
# 2026-04-23 (git sha 5eaaa7642). If the sidebar is reordered, update
# this list and regenerate all prior_vocab.json files.
READING_ORDER: list[tuple[str, str]] = [
    # Vol1 (16 content chapters)
    ("vol1", "introduction"),
    ("vol1", "ml_systems"),
    ("vol1", "ml_workflow"),
    ("vol1", "data_engineering"),
    ("vol1", "nn_computation"),
    ("vol1", "nn_architectures"),
    ("vol1", "frameworks"),
    ("vol1", "training"),
    ("vol1", "data_selection"),
    ("vol1", "model_compression"),
    ("vol1", "hw_acceleration"),
    ("vol1", "benchmarking"),
    ("vol1", "model_serving"),
    ("vol1", "ml_ops"),
    ("vol1", "responsible_engr"),
    ("vol1", "conclusion"),
    # Vol2 (17 content chapters) — inherits all of Vol1's prior vocab
    ("vol2", "introduction"),
    ("vol2", "compute_infrastructure"),
    ("vol2", "network_fabrics"),
    ("vol2", "data_storage"),
    ("vol2", "distributed_training"),
    ("vol2", "collective_communication"),
    ("vol2", "fault_tolerance"),
    ("vol2", "fleet_orchestration"),
    ("vol2", "performance_engineering"),
    ("vol2", "inference"),
    ("vol2", "edge_intelligence"),
    ("vol2", "ops_scale"),
    ("vol2", "security_privacy"),
    ("vol2", "robust_ai"),
    ("vol2", "sustainable_ai"),
    ("vol2", "responsible_ai"),
    ("vol2", "conclusion"),
]

BASE = Path(__file__).resolve().parents[3] / "quarto" / "contents"


def _chapter_qmd_path(vol: str, chap: str) -> Path | None:
    """Return the main QMD path for a chapter, or ``None`` if absent.

    The directory name and QMD stem normally match; the fallback handles
    the case where they diverge.
    """
    chap_dir = BASE / vol / chap
    direct = chap_dir / f"{chap}.qmd"
    if direct.is_file():
        return direct
    candidates = sorted(
        path for path in chap_dir.glob("*.qmd") if not path.name.startswith("_")
    )
    return candidates[0] if candidates else None


_DEFINITION_CALLOUT_RE = re.compile(
    r"^:{3,4}\s+\{\.callout-definition(?P<attrs>[^\n]*)\}\s*\n(?P<body>.*?)(?=^:{3,4}\s*$)",
    re.MULTILINE | re.DOTALL,
)
_TITLE_RE = re.compile(r'title="([^"]+)"')
_BOLD_TERM_RE = re.compile(r"\*\*\*([^*\n]+?)\*\*\*")
_INDEX_RE = re.compile(r"\\index\{[^{}]*\}")
_ATTR_RE = re.compile(r"\{#[^{}]+\}")


def _clean_markdown(text: str) -> str:
    text = _INDEX_RE.sub("", text)
    text = _ATTR_RE.sub("", text)
    text = text.replace("***", "").replace("**", "").replace("*", "")
    return " ".join(text.split()).strip()


def _first_paragraph(text: str) -> str:
    paragraph: list[str] = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            if paragraph:
                break
            continue
        if line.startswith((":::", "```")):
            if paragraph:
                break
            continue
        paragraph.append(line)
    return _clean_markdown(" ".join(paragraph))


def _terms_from_chapter_qmd(path: Path) -> list[dict[str, str]]:
    """Extract defined terms from definition callouts in a chapter QMD."""
    text = path.read_text(encoding="utf-8")
    entries: list[dict[str, str]] = []
    for match in _DEFINITION_CALLOUT_RE.finditer(text):
        body = match.group("body")
        title_match = _TITLE_RE.search(match.group("attrs"))
        bold_match = _BOLD_TERM_RE.search(body)
        term = bold_match.group(1) if bold_match else (
            title_match.group(1) if title_match else ""
        )
        term = _clean_markdown(term)
        if not term:
            continue
        entries.append(
            {
                "term": term,
                "definition": _first_paragraph(body),
            }
        )
    return entries


def prior_vocab_for(target_idx: int) -> list[dict]:
    """Union of every definition term introduced before ``target_idx``.

    Deduplicates by lowercased term; the first chapter to introduce a
    term wins, and later re-definitions are ignored.
    """
    terms: list[dict] = []
    seen: set[str] = set()
    for vol, chap in READING_ORDER[:target_idx]:
        qmd = _chapter_qmd_path(vol, chap)
        if qmd is None:
            continue
        for entry in _terms_from_chapter_qmd(qmd):
            term = entry.get("term", "").strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(
                {
                    "term": term,
                    "definition": entry.get("definition", "").strip(),
                    "first_seen": f"{vol}/{chap}",
                }
            )
    return terms


def build(vol: str, chap: str) -> dict:
    try:
        target_idx = next(
            i for i, (v, c) in enumerate(READING_ORDER) if v == vol and c == chap
        )
    except StopIteration:
        raise SystemExit(f"error: {vol}/{chap} not in reading order")
    terms = prior_vocab_for(target_idx)
    return {
        "target_chapter": f"{vol}/{chap}",
        "position_in_reading_order": target_idx + 1,
        "total_chapters": len(READING_ORDER),
        "prior_term_count": len(terms),
        "terms": terms,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: build_prior_vocab.py <vol1|vol2> <chapter>", file=sys.stderr)
        return 2
    vol, chap = argv[1], argv[2]
    print(json.dumps(build(vol, chap), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
