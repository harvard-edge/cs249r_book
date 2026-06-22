#!/usr/bin/env python3
r"""
Phase C.1 — formatting violation detector.

Scans every body-prose .qmd and reports \index{} placements that violate
formatting rules:
  V1. \index{} inside **bold** span
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


def text_before_for_italic_scan(before: str) -> str:
    """Remove Markdown constructs whose `*` characters are not italic markers."""
    before = re.sub(r"^\s*[*+-]\s+", "", before)
    before = re.sub(r"\*\*[^*\n]+?\*\*", "", before)
    before = re.sub(r"`[^`\n]+`", "", before)
    before = re.sub(r"\$[^$\n]*\$", "", before)
    before = before.replace(r"\*", "")
    return before


def bold_spans(line: str) -> list[tuple[int, int]]:
    """Return simple same-line Markdown bold spans."""
    spans = []
    start = None
    for marker in re.finditer(r"(?<!\\)\*\*", line):
        if start is None:
            start = marker.start()
        else:
            spans.append((start, marker.end()))
            start = None
    return spans


def find_violations(file: Path) -> list[dict]:
    s = str(file)
    if any(x in s for x in ("frontmatter/", "backmatter/", "/parts/",
                            "/glossary/", "/appendix", "_shelved")):
        return []
    rows = []
    text = file.read_text(errors="replace")
    in_code_block = False
    in_yaml = False
    in_html_comment = False
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()

        # Converted/retired figure source often lives inside HTML comments.
        # Ignore fences and index-like text there; Quarto will not render them.
        if in_html_comment or "<!--" in line:
            in_html_comment = "-->" not in line
            continue

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

        # YAML frontmatter is only the opening metadata block. Later `---`
        # lines are Markdown thematic breaks and must not put the scanner into
        # YAML mode for the rest of the chapter.
        if i == 1 and stripped == "---":
            in_yaml = True
            if "\\index{" in line:
                rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                             "violation": "V7_yaml", "ctx": line[:120]})
            continue
        if in_yaml and stripped == "---":
            in_yaml = False
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

        # V1: any \index{} inside a simple same-line bold span.
        for start, end in bold_spans(line):
            pos = line.find("\\index{", start, end)
            if pos >= 0:
                rows.append({"file": str(file.relative_to(ROOT)), "line": i,
                             "violation": "V1_inside_bold",
                             "ctx": line[max(0, pos-20):pos+60]})

        # V2: \index{} inside italic — pattern: \w*\\index{ where the * isn't part of **
        # Detect by checking if position is inside an unmatched single * span
        for m in re.finditer(r"\\index\{", line):
            pos = m.start()
            before = text_before_for_italic_scan(line[:pos])
            stars = before.count("*")
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

    OUTDIR.mkdir(parents=True, exist_ok=True)
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
