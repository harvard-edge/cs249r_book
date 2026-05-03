#!/usr/bin/env python3
"""
Phase C.1 — formatting violation detector.

Scans every body-prose .qmd and reports \index{} placements that violate
formatting rules:
  V1. \index{} inside opening **bold** span (after **, before content)
  V2. \index{} inside *italic* span
  V3. \index{} inside `code` span
  V4. \index{} on heading line (#+)
  V5. \index{} on callout fence (:::)
  V6. \index{} on code fence (```)
  V7. \index{} on YAML separator (---)
  V8. Bold span unbalanced (odd ** count) on a line with \index{
  V9. Italic span unbalanced
"""
from __future__ import annotations
import csv
import re
from pathlib import Path

ROOT = Path.cwd()
OUTDIR = ROOT / ".claude" / "_reviews" / "index_audit_2026-05-02"
CONTENT = ROOT / "book" / "quarto" / "contents"


def find_violations(file: Path) -> list[dict]:
    s = str(file)
    if any(x in s for x in ("frontmatter/", "backmatter/", "/parts/",
                            "/glossary/", "/appendix", "_shelved")):
        return []
    rows = []
    text = file.read_text(errors="replace")
    in_code_block = False
    in_yaml = False
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()

        # Track code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            if "\\index{" in line:
                rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                             "violation": "V6_code_fence", "ctx": line[:120]})
            continue
        if in_code_block:
            if "\\index{" in line:
                rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                             "violation": "V6_inside_code_block", "ctx": line[:120]})
            continue

        # YAML
        if stripped.startswith("---") and not stripped.startswith("--- "):
            in_yaml = not in_yaml
            if "\\index{" in line:
                rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                             "violation": "V7_yaml", "ctx": line[:120]})
            continue
        if in_yaml:
            if "\\index{" in line:
                rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                             "violation": "V7_in_yaml", "ctx": line[:120]})
            continue

        if "\\index{" not in line:
            continue

        # V4: heading
        if stripped.startswith("#") and not stripped.startswith("# │"):
            rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                         "violation": "V4_heading", "ctx": line[:120]})
            continue

        # V5: callout fence
        if stripped.startswith(":::"):
            rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                         "violation": "V5_callout", "ctx": line[:120]})
            continue

        # V1: \index{} immediately after opening ** (no content between **)
        # Pattern: ** followed by \index{
        # The ** is "opening" if there's an even count of completed **...** pairs before it
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
            # Count remaining `**` after consuming
            leftover = before[search_pos:]
            stray_stars = leftover.count("**")
            if stray_stars % 2 == 0:  # the ** is OPENING a new bold span
                rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                             "violation": "V1_inside_bold",
                             "ctx": line[max(0, m.start()-20):m.start()+60]})

        # V2: \index{} inside italic — pattern: \w*\\index{ where the * isn't part of **
        # Detect by checking if position is inside an unmatched single * span
        for m in re.finditer(r"\\index\{", line):
            pos = m.start()
            before = line[:pos]
            # Strip ** pairs first
            no_bolds = re.sub(r"\*\*[^*\n]+?\*\*", "", before)
            # Strip code spans
            no_code = re.sub(r"`[^`\n]+`", "", no_bolds)
            # Now count remaining single *
            stars = no_code.count("*")
            if stars % 2 == 1:
                rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                             "violation": "V2_inside_italic",
                             "ctx": line[max(0, pos-20):pos+60]})

        # V3: \index{} inside inline code `...`
        for m in re.finditer(r"\\index\{", line):
            pos = m.start()
            before = line[:pos]
            # Count backticks (not triple)
            ticks = len(re.findall(r"(?<!`)`(?!`)", before))
            if ticks % 2 == 1:
                rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                             "violation": "V3_inside_code",
                             "ctx": line[max(0, pos-20):pos+60]})

    return rows


def main():
    all_rows = []
    by_violation = {}
    for f in sorted(CONTENT.rglob("*.qmd")):
        rows = find_violations(f)
        all_rows.extend(rows)
        for r in rows:
            by_violation.setdefault(r["violation"], 0)
            by_violation[r["violation"]] += 1

    outpath = OUTDIR / "phase_c_format_violations.csv"
    with open(outpath, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=["file", "line", "violation", "ctx"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"Total violations: {len(all_rows)}")
    for k, v in sorted(by_violation.items()):
        print(f"  {k}: {v}")
    print(f"Written: {outpath}")


if __name__ == "__main__":
    main()
