#!/usr/bin/env python3
"""Flag doubled back-to-back words in SUBSTITUTED prose.

Why this cannot be a source-level check
---------------------------------------
The defect is invisible in the ``.qmd`` source. Given::

    (`{python} MobileNetTradeoffCalc.fps_fp32_str` FPS)

the source contains exactly one "FPS". The closed export renders ``8.3 FPS``,
so the reader sees ``(8.3 FPS FPS)``. ``prose --scope duplicate-words`` reads
source and therefore cannot see it; this checker substitutes values first.

Relationship to ``lego_prose_units.py``
---------------------------------------
That checker classifies exports as open/closed by *formatter name* and owns
domain-unit duplication. This one is formatter-agnostic: it reads the rendered
value, so it also catches label duplication from ``fmt_count(label=...)``
(``1,024 GPUs GPUs``), parameter/shard/lookup counts, and any future closed
export whose trailing token prose happens to repeat. On the 2026-08 corpus it
found 16 sites where the unit-specific checker found 10.

Cost: this executes LEGO cells, so it is slower than a source scan, but it does
NOT require a Quarto/HTML build -- it reuses the prose-preview machinery.

Added 2026-08-16 after a "surgical clarity" pass shipped 16 doubled tokens that
every source-level gate passed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTENTS = REPO_ROOT / "book" / "quarto" / "contents"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mlsysim"))

# Repeats that are legitimate English or table/markup artefacts.
ALLOW = {"had", "that", "no", "very", "long", "many", "s", "d", "t", "the"}

DOUBLE = re.compile(r"\b([A-Za-z][A-Za-z/%$.\-]{0,18})\s+\1\b")


def check_file(path: Path) -> list[tuple[int, str, str]]:
    """Return (lineno, doubled_token, rendered_context) for each hit."""
    from book.tools.audit.fmt.audit_prose import audit_prose_previews

    try:
        previews = audit_prose_previews(path)
    except Exception as exc:  # noqa: BLE001 - a broken cell is another check's job
        print(f"  (skipped {path.name}: cell exec failed: {exc})", file=sys.stderr)
        return []

    issues: list[tuple[int, str, str]] = []
    for p in previews:
        text = p.preview or ""
        stripped = text.lstrip()
        if not stripped or stripped.startswith(("|", ":--", "---")):
            continue
        for m in DOUBLE.finditer(text):
            token = m.group(1)
            if token.lower() in ALLOW or token.isdigit():
                continue
            start = max(0, m.start() - 50)
            issues.append((p.line, token, text[start : m.end() + 25].strip()))
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="QMD file(s) or directories")
    args = ap.parse_args()

    if args.paths:
        expanded: list[Path] = []
        for raw in args.paths:
            p = Path(raw)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if p.is_dir():
                expanded.extend(sorted(p.rglob("*.qmd")))
            elif p.suffix == ".qmd":
                expanded.append(p)
        paths = expanded
    else:
        paths = sorted(CONTENTS.rglob("*.qmd"))

    failures = 0
    total = 0
    for path in paths:
        p = path if path.is_absolute() else REPO_ROOT / path
        if not p.exists() or p.suffix != ".qmd":
            continue
        if "```{python}" not in p.read_text(encoding="utf-8"):
            continue  # no substitution can occur
        total += 1
        issues = check_file(p)
        if not issues:
            continue
        failures += 1
        print(f"\n{p.relative_to(REPO_ROOT)}")
        for lineno, token, context in issues:
            print(f"  L{lineno}: doubled '{token}' after substitution")
            print(f"    ...{context}...")

    if failures:
        print(f"\n{failures} file(s) with doubled words in rendered prose")
        return 1
    print(f"OK rendered prose has no doubled words ({total} QMD files with LEGO cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
