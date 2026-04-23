#!/usr/bin/env python3
"""Phase A3 — alt-text coverage diff.

Reads ``07_alt_text/data/alt_text_edits_fixed.json`` (212 entries with a
``label`` like ``fig-ai-timeline`` and the editor-approved ``alt_text``
string), then walks ``book/quarto/contents/vol1/**/*.qmd`` looking for
``fig-alt="..."`` (or ``fig-alt: "..."``) attributes attached to a figure
with ``#fig-<label>`` or matching label in a ``label:`` block. Compares
the current alt text to the expected one (whitespace-normalised) and
emits per-figure status: ``applied`` / ``differs`` / ``missing-alt`` /
``fig-not-found``.

Output: ``ledgers/vol1-alt-text-coverage.json``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

INPUT = Path.home() / "Desktop/MIT_Press_Feedback/07_alt_text/data/alt_text_edits_fixed.json"
QUARTO_ROOT = Path("/Users/VJ/GitHub/MLSysBook-release-audit/book/quarto/contents/vol1")
OUT_DIR = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/ledgers"

# Match fig-alt attribute in a Pandoc-style figure attribute block.
#   ![Caption](path){#fig-foo fig-alt="..." width=...}
# or in a YAML block (markdown image with ``label:`` and ``fig-alt:``).
ALT_INLINE_RE = re.compile(r'fig-alt\s*=\s*"([^"]*)"')
ALT_YAML_RE = re.compile(r'^\s*fig-alt:\s*(?:"([^"]*)"|([^\n]+))', re.MULTILINE)
LABEL_INLINE_RE = re.compile(r'\{#(fig-[A-Za-z0-9_-]+)[^}]*\}')
LABEL_YAML_RE = re.compile(r'^\s*label:\s*(fig-[A-Za-z0-9_-]+)', re.MULTILINE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def collect_qmd_alt_text(root: Path) -> dict[str, list[dict]]:
    """Return {label: [{file, line, alt}]} for every fig label found in QMDs."""
    result: dict[str, list[dict]] = {}
    qmds = sorted(root.rglob("*.qmd"))

    for qmd in qmds:
        text = qmd.read_text(errors="replace")
        # Inline form: scan line by line so we can capture line numbers.
        lines = text.splitlines()
        # Inline pattern: a single line carrying both label and fig-alt or split
        # across the same figure attribute block. We allow up to 6 lines of
        # adjacency.
        for i, line in enumerate(lines):
            label_match = LABEL_INLINE_RE.search(line)
            if not label_match:
                continue
            label = label_match.group(1)
            # search the same line and the surrounding lines for fig-alt=
            window = "\n".join(lines[max(0, i - 2):min(len(lines), i + 4)])
            alt_match = ALT_INLINE_RE.search(window)
            if alt_match:
                result.setdefault(label, []).append({
                    "file": str(qmd.relative_to(root.parent)),
                    "line": i + 1,
                    "alt": alt_match.group(1),
                    "form": "inline",
                })

        # YAML form: figure declared in a callout-figure or block-figure with
        # explicit ``label:`` and ``fig-alt:`` keys (rare in this codebase but
        # covered for completeness).
        for m in LABEL_YAML_RE.finditer(text):
            label = m.group(1)
            # search for fig-alt within 30 lines after the label declaration
            label_pos = m.start()
            tail = text[label_pos: label_pos + 4000]
            alt_m = ALT_YAML_RE.search(tail)
            if alt_m:
                alt = alt_m.group(1) or alt_m.group(2) or ""
                line_no = text[:label_pos].count("\n") + 1
                # avoid duplicate insertion for inline-already-found labels
                seen = result.setdefault(label, [])
                if not any(s["file"].endswith(qmd.name) and s["line"] == line_no for s in seen):
                    seen.append({
                        "file": str(qmd.relative_to(root.parent)),
                        "line": line_no,
                        "alt": alt.strip().strip('"'),
                        "form": "yaml",
                    })
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    expected = json.loads(INPUT.read_text())
    print(f"Loaded {len(expected)} expected alt-text entries from {INPUT.name}")

    qmd_index = collect_qmd_alt_text(QUARTO_ROOT)
    print(f"Found {len(qmd_index)} unique fig labels with fig-alt in vol1 QMDs")

    results = []
    status_counts: Counter[str] = Counter()

    for entry in expected:
        label = entry.get("label") or ""
        expected_alt = (entry.get("alt_text") or "").strip()
        figure_ref = entry.get("figure_ref") or ""

        hits = qmd_index.get(label, [])
        if not hits:
            status = "fig-not-found"
            current_alt = None
            qmd_file = None
            qmd_line = None
        else:
            # Use the first hit; report how many duplicates were seen.
            hit = hits[0]
            current_alt = hit["alt"]
            qmd_file = hit["file"]
            qmd_line = hit["line"]
            if not current_alt:
                status = "missing-alt"
            elif normalize(current_alt) == normalize(expected_alt):
                status = "applied"
            else:
                status = "differs"

        results.append({
            "label": label,
            "figure_ref": figure_ref,
            "expected_alt": expected_alt,
            "current_alt": current_alt,
            "status": status,
            "qmd_file": qmd_file,
            "qmd_line": qmd_line,
            "extra_hits": len(hits) - 1 if hits else 0,
        })
        status_counts[status] += 1

    summary = {
        "expected_total": len(expected),
        "qmd_labels_with_fig_alt": len(qmd_index),
        "status_counts": dict(status_counts),
        "discrepancy_note": (
            "212 alt-text entries in alt_text_edits_fixed.json vs the often-"
            "cited '213 figures' in vol1; one figure was either dropped from "
            "the editor's list or merged into another."
        ),
    }

    out_path = OUT_DIR / "vol1-alt-text-coverage.json"
    out_path.write_text(json.dumps({"summary": summary, "entries": results}, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
