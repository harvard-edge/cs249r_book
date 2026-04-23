#!/usr/bin/env python3
"""Programmatic locator for Phase B1.

Replaces the per-chapter LLM subagents with a deterministic, reproducible
checker that judges every ``needs_locator`` annotation against the live
QMD source on the dev branch.

For each entry the locator picks an evidence string ("expected after"
text) derived from the annotation type, normalises whitespace and
typographic punctuation (curly quotes, em-/en-dashes, NBSPs), and
searches the QMD for it.  Outcomes:

* ``applied``        — evidence found in QMD.
* ``cannot-locate``  — annotation has no usable post-edit text (typical
  for short PDF parsing fragments).
* ``missed``         — evidence had ≥ a configurable minimum signal yet
  is absent from QMD.
* ``skipped-with-reason`` — annotation type known to be cosmetic /
  non-applicable (highlight-only with no insert/strike, ``Comp:``
  query stubs already covered elsewhere, etc.).

The output mirrors the LLM subagent contract and is written one JSON
file per bin to ``scripts/locator-output/``, so downstream Phase D
aggregation does not need to special-case origin.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

REPO_ROOT = Path("/Users/VJ/GitHub/MLSysBook-release-audit")
INPUT_DIR = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/scripts/locator-input"
OUTPUT_DIR = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/scripts/locator-output"

# Global fallback search root — annotations whose recorded chapter does
# not contain the evidence may have been mis-attributed during PDF
# extraction (e.g. ``ch14_end`` rolled into Ch15). We confirm by
# searching across all Vol I QMDs before declaring "missed".
VOL1_GLOBAL = "book/quarto/contents/vol1"

# Evidence shorter than this is too noisy to count as a meaningful match.
MIN_EVIDENCE_LEN = 6

# Skip-with-reason lookup for buckets that are cosmetic or covered by a
# different ledger.
SKIP_BUCKETS = {
    "highlight-only": "Highlight without insert/strike — not a textual edit.",
    "au-query": "AU/Comp query — coverage handled by au_query_coverage ledger.",
}


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200b": "",
    "\u2026": "...",
}


def normalise(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


# ---------------------------------------------------------------------------
# Evidence selection
# ---------------------------------------------------------------------------
def evidence_for(entry: dict) -> tuple[str, str]:
    """Return ``(evidence, source_field)`` to look for in the QMD.

    The chosen evidence is the post-edit text the entry should produce.
    """

    typ = (entry.get("type") or "").lower()
    insert = (entry.get("insert_text") or "").strip()
    struck = (entry.get("struck_text") or "").strip()
    nearby = (entry.get("nearby_context") or "").strip()
    content = (entry.get("content") or "").strip()
    bucket = entry.get("bucket") or ""

    # Cosmetic typesetter markers ("[space]", "space", "sp", "ws") have
    # no in-text evidence the editor can verify against the QMD.
    cosmetic = {"[space]", "space", "sp", "ws", "[sp]", "[nbsp]"}
    if insert.lower() in cosmetic or content.lower() in cosmetic:
        return "", "typesetter-spacing"

    # Typesetter notes like "[en dash]", "[em dash]", "closed em dash",
    # "open em dash", "[minus sign]", "[dpace]" describe a typographic
    # substitution rather than literal text. They have no QMD-locatable
    # evidence; report under the typesetter ledger.
    text_norm = (insert or content).strip().lower()
    typesetter_keywords = {
        "en dash", "em dash", "minus sign", "dpace", "pace", "thin space",
        "narrow space", "hair space", "non-breaking space", "nbsp",
    }
    if (text_norm.startswith("[") and text_norm.endswith("]")) or any(
        kw in text_norm for kw in typesetter_keywords
    ):
        return "", "typesetter-note"

    # AU/Comp query stubs are tracked separately in the AU coverage
    # ledger; they do not represent in-text edits.
    text_for_query = (content or insert or "").strip().lower()
    if (
        text_for_query.startswith("au:")
        or text_for_query.startswith("comp:")
        or text_for_query.startswith("au ")
    ):
        return "", "au-or-comp-query"

    # Caret/insert: the inserted text should be in QMD.
    if typ in {"caret", "insert"} and insert:
        return insert, "insert_text"

    # Replace: prefer insert_text (post-edit form).
    if typ == "replace" and insert:
        return insert, "insert_text"

    # StrikeOut: nothing replaces it. Use nearby_context tokens minus the
    # struck text so we can confirm the surrounding sentence still
    # exists. If that yields too short a fragment we cannot judge.
    if typ in {"strikeout", "strike"}:
        if nearby and struck:
            cleaned = nearby.replace(struck, "").strip()
            return cleaned, "nearby_minus_struck"
        return nearby or "", "nearby_context"

    # Highlight without insert/strike — cosmetic / queryable.
    if typ == "highlight" and not (insert or struck):
        return "", "highlight-only"

    # Comments / Free-text annotations (CE notes) are usually queries —
    # they map to AU/Comp ledgers, not to in-text changes.
    if typ in {"text", "freetext", "highlight"} and content and not insert:
        return "", "query-only"

    # Generic fallback — try insert, then content, then nearby.
    for candidate, source in [
        (insert, "insert_text"),
        (content, "content"),
        (nearby, "nearby_context"),
    ]:
        if candidate:
            return candidate, source
    return "", "no-evidence"


# ---------------------------------------------------------------------------
# QMD search helpers
# ---------------------------------------------------------------------------
class QmdIndex:
    """In-memory cache mapping QMD file -> normalised text + line index."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._cache: dict[str, tuple[str, list[tuple[int, str]]]] = {}

    def _load(self, rel_path: str) -> tuple[str, list[tuple[int, str]]]:
        if rel_path in self._cache:
            return self._cache[rel_path]
        target = self.repo_root / rel_path
        text = ""
        line_index: list[tuple[int, str]] = []
        if target.is_dir():
            for qmd in sorted(target.rglob("*.qmd")):
                raw = qmd.read_text(errors="ignore")
                for i, line in enumerate(raw.splitlines(), 1):
                    line_index.append((i, normalise(line)))
                text += "\n" + raw
        elif target.is_file():
            raw = target.read_text(errors="ignore")
            for i, line in enumerate(raw.splitlines(), 1):
                line_index.append((i, normalise(line)))
            text = raw
        norm_full = normalise(text)
        self._cache[rel_path] = (norm_full, line_index)
        return self._cache[rel_path]

    def find(self, rel_path: str, evidence: str) -> tuple[bool, int | None]:
        if not evidence:
            return False, None
        norm_full, line_index = self._load(rel_path)
        target = normalise(evidence)
        if not target:
            return False, None
        if target not in norm_full:
            return False, None
        # Find a representative line number for evidence.
        for line_no, line_norm in line_index:
            if target in line_norm:
                return True, line_no
        return True, None


# ---------------------------------------------------------------------------
# Bin processor
# ---------------------------------------------------------------------------
def process_bin(bin_path: Path, qmd_index: QmdIndex) -> dict:
    bin_data = json.loads(bin_path.read_text())
    qmd_target = bin_data["qmd_target"]
    judgments: list[dict] = []
    counts: Counter[str] = Counter()

    for entry in bin_data.get("entries", []):
        bucket = entry.get("bucket", "")
        if bucket in SKIP_BUCKETS:
            judgments.append({
                "annotation_id": entry["annotation_id"],
                "status": "skipped-with-reason",
                "evidence": "",
                "qmd_line": None,
                "rationale": SKIP_BUCKETS[bucket],
            })
            counts["skipped-with-reason"] += 1
            continue

        evidence, source = evidence_for(entry)

        # Catch query-only / no-evidence first.
        if source in {
            "highlight-only",
            "query-only",
            "no-evidence",
            "typesetter-spacing",
            "typesetter-note",
            "au-or-comp-query",
        }:
            judgments.append({
                "annotation_id": entry["annotation_id"],
                "status": "skipped-with-reason",
                "evidence": "",
                "qmd_line": None,
                "rationale": f"{source} (type={entry.get('type')}, bucket={bucket})",
            })
            counts["skipped-with-reason"] += 1
            continue

        # Too-short evidence -> cannot confidently judge.
        if len(normalise(evidence)) < MIN_EVIDENCE_LEN:
            judgments.append({
                "annotation_id": entry["annotation_id"],
                "status": "cannot-locate",
                "evidence": evidence,
                "qmd_line": None,
                "rationale": (
                    "Evidence too short for unique match "
                    f"(len={len(normalise(evidence))}, source={source})."
                ),
            })
            counts["cannot-locate"] += 1
            continue

        found, line_no = qmd_index.find(qmd_target, evidence)
        if found:
            judgments.append({
                "annotation_id": entry["annotation_id"],
                "status": "applied",
                "evidence": evidence,
                "qmd_line": line_no,
                "rationale": f"Match for {source} at line {line_no}.",
            })
            counts["applied"] += 1
            continue

        # Fall back to a global Vol I search to catch chapter
        # mis-attribution from the PDF extractor.
        global_found, _ = qmd_index.find(VOL1_GLOBAL, evidence)
        if global_found:
            judgments.append({
                "annotation_id": entry["annotation_id"],
                "status": "applied",
                "evidence": evidence,
                "qmd_line": None,
                "rationale": (
                    f"Match for {source} found via Vol I global fallback "
                    f"(annotation chapter likely mis-attributed)."
                ),
            })
            counts["applied"] += 1
        else:
            judgments.append({
                "annotation_id": entry["annotation_id"],
                "status": "missed",
                "evidence": evidence,
                "qmd_line": None,
                "rationale": f"{source} not found in {qmd_target} or Vol I.",
            })
            counts["missed"] += 1

    return {
        "bin_id": bin_data.get("chapter", bin_path.stem),
        "qmd_target": qmd_target,
        "judged_count": len(judgments),
        "by_status": dict(counts),
        "judgments": judgments,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qmd_index = QmdIndex(REPO_ROOT)

    bins = sorted(p for p in INPUT_DIR.glob("bin-*.json"))
    print(f"Processing {len(bins)} bin(s)…")

    grand = Counter()
    for bin_path in bins:
        result = process_bin(bin_path, qmd_index)
        out = OUTPUT_DIR / bin_path.name
        out.write_text(json.dumps(result, indent=2))
        grand.update(result["by_status"])
        print(
            f"  {bin_path.name:<45s} "
            f"applied={result['by_status'].get('applied', 0):>4d}  "
            f"missed={result['by_status'].get('missed', 0):>4d}  "
            f"cannot-locate={result['by_status'].get('cannot-locate', 0):>4d}  "
            f"skipped={result['by_status'].get('skipped-with-reason', 0):>4d}"
        )

    print("\nGrand totals:", dict(grand))
    summary_path = OUTPUT_DIR / "_LOCATOR_SUMMARY.json"
    summary_path.write_text(json.dumps({"by_status": dict(grand)}, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
