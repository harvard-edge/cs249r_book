#!/usr/bin/env python3
"""Phase A5 — figure audit inventory (pointer-only).

The figure-caption verification work is being run separately by the
author. This script just records the inventory of HIGH/MEDIUM/LOW
figures per volume and chapter so the coverage report can quote the
baseline numbers without re-running the audit.

Source: ``book/quarto/_build/figure_triage_priorities.json`` lives in
the user's primary worktree, not in this audit worktree (the ``_build``
directory is git-ignored), so we read it via absolute path.

Output: ``ledgers/vol2-figure-audit-pointers.json`` (and a vol1 sibling
for completeness).
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

TRIAGE = Path("/Users/VJ/GitHub/MLSysBook/book/quarto/_build/figure_triage_priorities.json")
QUARTO_ROOT = Path("/Users/VJ/GitHub/MLSysBook-release-audit/book/quarto/contents")
OUT_DIR = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/ledgers"


def find_label_volume_chapter(label: str) -> tuple[str | None, str | None, str | None]:
    """Return (volume, chapter_dir, qmd_file_path_relative) for a fig label."""
    pattern = re.compile(r"\{#" + re.escape(label) + r"[}\s]")
    # Search both vol1 and vol2 directly.
    for vol_dir in ("vol1", "vol2"):
        vol_root = QUARTO_ROOT / vol_dir
        if not vol_root.exists():
            continue
        for qmd in vol_root.rglob("*.qmd"):
            try:
                txt = qmd.read_text(errors="replace")
            except Exception:
                continue
            if pattern.search(txt):
                rel = qmd.relative_to(QUARTO_ROOT.parent)
                # chapter directory is the path component immediately under vol_dir
                chapter_dir = qmd.relative_to(vol_root).parts[0]
                return (vol_dir, chapter_dir, str(rel))
    return (None, None, None)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRIAGE.exists():
        raise FileNotFoundError(
            f"Figure triage file missing at {TRIAGE}. The audit worktree's "
            f"_build dir is gitignored; this file is read via absolute path "
            f"from the primary worktree."
        )

    triage = json.loads(TRIAGE.read_text())
    print(f"Loaded {len(triage)} figures from {TRIAGE}")

    per_volume: dict[str, Counter[str]] = defaultdict(Counter)
    per_chapter: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    unresolved: list[str] = []
    entries = []

    for label, meta in sorted(triage.items()):
        priority = meta.get("priority", "UNKNOWN")
        reason = meta.get("reason", "")
        vol, chapter, qmd = find_label_volume_chapter(label)
        if vol is None:
            unresolved.append(label)
            entries.append({
                "label": label,
                "priority": priority,
                "reason": reason,
                "volume": None,
                "chapter": None,
                "qmd_file": None,
                "found": False,
            })
            continue
        per_volume[vol][priority] += 1
        per_chapter[vol][chapter][priority] += 1
        entries.append({
            "label": label,
            "priority": priority,
            "reason": reason,
            "volume": vol,
            "chapter": chapter,
            "qmd_file": qmd,
            "found": True,
        })

    summary = {
        "source_triage_file": str(TRIAGE),
        "total_figures_in_triage": len(triage),
        "unresolved_labels": len(unresolved),
        "per_volume_priority": {
            v: dict(sorted(c.items())) for v, c in per_volume.items()
        },
        "per_chapter_priority": {
            v: {ch: dict(sorted(c.items())) for ch, c in chapters.items()}
            for v, chapters in per_chapter.items()
        },
        "verification_status": (
            "Figure caption verification is being run separately by the "
            "author. See .claude/_reviews/figure_audit/FIX_LIST_BY_CHAPTER.md "
            "in the primary worktree (note: file may not exist in audit "
            "worktree because _reviews is gitignored)."
        ),
        "unresolved_labels_sample": unresolved[:20],
    }

    out_path = OUT_DIR / "vol2-figure-audit-pointers.json"
    out_path.write_text(json.dumps({"summary": summary, "entries": entries}, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
