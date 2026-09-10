#!/usr/bin/env python3
"""Preprocess QMD textbook chapters into clean prose for taxonomy extraction.

Strips code blocks, LaTeX, TikZ, figures, tables, and Quarto markup.
Keeps section headers, paragraph prose, and lists.

Usage:
    python3 preprocess.py                          # Process all chapters
    python3 preprocess.py book/.../chapter.qmd     # Process one chapter
    python3 preprocess.py --test                   # Run self-test
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).parent.parent / "book" / "quarto" / "contents"
PROSE_DIR = Path(__file__).parent / "_prose"

SKIP_DIRS = {"frontmatter", "backmatter", "parts"}


def extract_prose(qmd_path: str | Path) -> str:
    """Extract clean teaching prose from a QMD chapter file.

    Returns text with section headers and paragraph prose only.
    """
    text = Path(qmd_path).read_text(encoding="utf-8")

    # 1. YAML frontmatter
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)

    # 2. Code blocks (```python ... ``` and ```{python} ... ```)
    text = re.sub(r"```\{?[a-zA-Z]*\}?.*?```", "", text, flags=re.DOTALL)

    # 3. TikZ environments
    text = re.sub(
        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", text, flags=re.DOTALL
    )

    # 4. Display math ($$...$$) → keep a marker
    text = re.sub(r"\$\$.*?\$\$", "[EQUATION]", text, flags=re.DOTALL)

    # 5. Figure divs (::: {#fig-...} ... :::)
    text = re.sub(r":::\s*\{#fig-.*?\}.*?:::", "", text, flags=re.DOTALL)

    # 6. Other div blocks (callouts, column-margin, etc.)
    text = re.sub(r":::+\s*\{[^}]*\}.*?:::+", "", text, flags=re.DOTALL)
    text = re.sub(r":::+", "", text)

    # 7. HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # 8. Image references
    text = re.sub(r"!\[.*?\]\(.*?\)(\{[^}]*\})?", "", text)

    # 9. LaTeX commands (strip command, keep content)
    text = re.sub(r"\\(chapterminitoc|noindent|newpage|clearpage|pagebreak)", "", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # 10. Footnote definitions
    text = re.sub(r"^\[\^[^\]]+\]:.*$", "", text, flags=re.MULTILINE)

    # 11. Quarto cross-references (strip but note them)
    text = re.sub(r"@(fig|tbl|lst|eq)-[\w-]+", "", text)
    text = re.sub(r"@sec-[\w-]+", "", text)

    # 12. Inline Python refs
    text = re.sub(r"`\{python\}[^`]*`", "[VALUE]", text)

    # 13. Quarto attributes on headers and blocks
    text = re.sub(r"\{[^}]*\}", "", text)

    # 14. Margin figure commands
    text = re.sub(r"^\s*marginfigure.*$", "", text, flags=re.MULTILINE)

    # 15. Table rows (pipe tables)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s|:-]+$", "", text, flags=re.MULTILINE)

    # 16. Clean up
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Filter lines
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        # Skip artifact lines
        if stripped in (":::", "::::", "{}", "[]"):
            continue
        if len(stripped) < 3 and not stripped.startswith("#"):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def get_chapters() -> list[tuple[str, Path]]:
    """Find all content chapters in Vol1 and Vol2."""
    chapters = []
    for vol in ["vol1", "vol2"]:
        vol_dir = BOOK_ROOT / vol
        if not vol_dir.exists():
            continue
        for d in sorted(vol_dir.iterdir()):
            if not d.is_dir() or d.name in SKIP_DIRS:
                continue
            qmd = d / f"{d.name}.qmd"
            if qmd.exists():
                name = f"{vol}_{d.name}"
                chapters.append((name, qmd))
    return chapters


def preprocess_all() -> dict[str, str]:
    """Process all chapters, save to _prose/ directory."""
    PROSE_DIR.mkdir(exist_ok=True)
    chapters = get_chapters()
    results = {}

    for name, qmd_path in chapters:
        prose = extract_prose(qmd_path)
        out_path = PROSE_DIR / f"{name}.txt"
        out_path.write_text(prose, encoding="utf-8")
        results[name] = prose

        original_kb = qmd_path.stat().st_size // 1024
        prose_kb = len(prose) // 1024
        pct = 100 - (len(prose) * 100 // qmd_path.stat().st_size) if qmd_path.stat().st_size > 0 else 0
        print(f"  {name}: {original_kb}KB → {prose_kb}KB ({pct}% reduction)")

    return results


def self_test():
    """Test the preprocessor on known chapters."""
    print("═══ Preprocessor Self-Test ═══\n")

    test_chapters = [
        ("vol1", "nn_computation"),
        ("vol1", "hw_acceleration"),
        ("vol2", "inference"),
    ]

    for vol, ch in test_chapters:
        qmd = BOOK_ROOT / vol / ch / f"{ch}.qmd"
        if not qmd.exists():
            print(f"  SKIP: {qmd} not found")
            continue

        prose = extract_prose(qmd)
        original = qmd.stat().st_size

        # Verify
        has_headers = bool(re.search(r"^##", prose, re.MULTILINE))
        has_prose = len(prose) > 1000
        no_code = "```" not in prose
        no_tikz = "tikzpicture" not in prose
        no_yaml = "---\n" not in prose[:50]
        no_html = "<div" not in prose and "<img" not in prose

        status = "✅" if all([has_headers, has_prose, no_code, no_tikz, no_yaml, no_html]) else "❌"

        # Count sections
        sections = re.findall(r"^##\s+(.+)$", prose, re.MULTILINE)

        print(f"  {status} {vol}/{ch}:")
        print(f"      Size: {original // 1024}KB → {len(prose) // 1024}KB ({100 - len(prose) * 100 // original}%)")
        print(f"      Sections: {len(sections)}")
        print(f"      Has headers: {has_headers}, Has prose: {has_prose}")
        print(f"      No code: {no_code}, No TikZ: {no_tikz}, No YAML: {no_yaml}, No HTML: {no_html}")
        print()

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        self_test()
    elif len(sys.argv) > 1:
        prose = extract_prose(sys.argv[1])
        print(prose)
    else:
        print("═══ Preprocessing All Chapters ═══\n")
        results = preprocess_all()
        print(f"\n  Total: {len(results)} chapters processed")
        total_kb = sum(len(v) for v in results.values()) // 1024
        print(f"  Total prose: {total_kb}KB")
