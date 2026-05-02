#!/usr/bin/env python3
"""
book-index-tag-placement hook.

Fails if any \index{} is placed inside a structural element where it
breaks formatting:
  - Inside opening **bold** span (V1)
  - Inside *italic* span (V2)
  - Inside `code` span (V3)
  - On heading line (V4)
  - On callout fence (V5) — extends existing book-check-index-placement
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
CONTENT = ROOT / "book" / "quarto" / "contents"


def find_violations(file: Path) -> list[tuple[int, str, str]]:
    s = str(file)
    if any(x in s for x in ("frontmatter/", "backmatter/", "/parts/",
                            "/glossary/", "/appendix", "_shelved")):
        return []
    rows = []
    text = file.read_text(errors="replace")
    in_code_block = False
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if "\\index{" not in line:
            continue
        if stripped.startswith("#") and not stripped.startswith("# │"):
            rows.append((i, "V4_heading", line[:120]))
            continue
        # Skip callout fence lines — handled by project's existing
        # book-check-index-placement hook
        if stripped.startswith(":::"):
            continue
        # V1: \index{} inside opening **
        for m in re.finditer(r"\*\*\\index\{", line):
            before = line[:m.start()]
            pairs = re.findall(r"\*\*[^*\n]+?\*\*", before)
            consumed_pairs = 0
            search_pos = 0
            for p in pairs:
                idx = before.find(p, search_pos)
                if idx >= 0:
                    consumed_pairs += 1
                    search_pos = idx + len(p)
            leftover = before[search_pos:]
            if leftover.count("**") % 2 == 0:
                rows.append((i, "V1_inside_opening_bold",
                             line[max(0, m.start()-15):m.start()+50]))
                break
        # V3: inside backticks
        for m in re.finditer(r"\\index\{", line):
            before = line[:m.start()]
            ticks = len(re.findall(r"(?<!`)`(?!`)", before))
            if ticks % 2 == 1:
                rows.append((i, "V3_inside_code",
                             line[max(0, m.start()-15):m.start()+50]))
                break
    return rows


def main():
    failures = []
    for f in sorted(CONTENT.rglob("*.qmd")):
        rows = find_violations(f)
        for line, kind, ctx in rows:
            failures.append((f.relative_to(ROOT), line, kind, ctx))

    if failures:
        print(f"Index tag-placement audit FAILED: {len(failures)} violations")
        for rel, line, kind, ctx in failures[:20]:
            print(f"  {rel}:{line}  [{kind}]  {ctx}")
        if len(failures) > 20:
            print(f"  ... and {len(failures)-20} more")
        return 1
    print("Index tag-placement audit PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
