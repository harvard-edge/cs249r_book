#!/usr/bin/env python3
"""
Check that figures use div syntax (fig-cap and fig-alt on the div), not old-style
markdown image or chunk options.

We standardize on:
  ::: {#fig-xxx fig-env="figure" fig-pos="htb" fig-cap="..." fig-alt="..."}
  ![](path)   OR   ```{python} / ```{.tikz} block
  :::

This script fails if it finds:
1. Markdown image figures: ![Caption](path){#fig-...} (caption/alt on image; no wrapper div)
2. Chunk options: #| fig-cap= or #| fig-alt= (YAML options on a code chunk instead of div)

Usage (from repo root or book/):
  python3 book/tools/scripts/content/check_figure_div_syntax.py
  python3 tools/scripts/content/check_figure_div_syntax.py -d quarto/contents/

Pre-commit: run from book/ with -d quarto/contents/ (default).

Exit: 0 if no violations, 1 if any (with message to use div syntax).
"""

import argparse
import re
import sys
from pathlib import Path

# Default: when run from book/, scan quarto/contents/
DEFAULT_DIR = Path("quarto/contents")

# Markdown image with #fig- on the image (old style). Caption in brackets, path in parens, then {#fig-...}
MARKDOWN_IMAGE_FIG = re.compile(r"!\[.*\]\s*\([^)]+\)\s*\{#fig-")

# Chunk option fig-cap or fig-alt (we use div attributes only)
CHUNK_FIG_OPTION = re.compile(r"^#\|\s*(fig-cap|fig-alt)\s*[:=]")


def scan_file(qmd_path: Path, contents_dir: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_1based, kind, line_stripped) for violations."""
    violations = []
    try:
        text = qmd_path.read_text(encoding="utf-8")
    except OSError:
        return violations
    for i, line in enumerate(text.splitlines(), start=1):
        if MARKDOWN_IMAGE_FIG.search(line):
            violations.append((i, "markdown-image-fig", line.strip()[:80]))
        elif CHUNK_FIG_OPTION.search(line):
            violations.append((i, "chunk-fig-option", line.strip()[:80]))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce figure div syntax (no ![](){#fig-}, no #| fig-cap/fig-alt)."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Directory containing .qmd files (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only exit 1; minimal output",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    # Pre-commit runs from repo root; manual run may be from book/. Find book/quarto/contents.
    if (cwd / "book" / "quarto" / "contents").is_dir():
        base = cwd / "book"
    elif (cwd / "quarto" / "contents").is_dir():
        base = cwd
    else:
        base = cwd
    content_dir = (base / args.directory).resolve()
    if not content_dir.is_dir():
        if not args.quiet:
            print(f"Directory not found: {content_dir}", file=sys.stderr)
        return 1

    all_violations: list[tuple[Path, list[tuple[int, str, str]]]] = []
    for qmd in sorted(content_dir.rglob("*.qmd")):
        v = scan_file(qmd, content_dir)
        if v:
            all_violations.append((qmd, v))

    if not all_violations:
        return 0

    if args.quiet:
        return 1

    print("Figure div syntax check failed: use div syntax for all figures.", file=sys.stderr)
    print("  Use: ::: {#fig-xxx fig-env=\"figure\" fig-pos=\"htb\" fig-cap=\"...\" fig-alt=\"...\"}", file=sys.stderr)
    print("       <content: ![](path) or code block>", file=sys.stderr)
    print("  :::", file=sys.stderr)
    print("  Do NOT use: ![Caption](path){#fig-...} or #| fig-cap= / #| fig-alt=", file=sys.stderr)
    print("  See .claude/rules/book-prose.md Section 6 (Visuals & Assets).", file=sys.stderr)
    print(file=sys.stderr)
    for qmd, violations in all_violations:
        try:
            rel = qmd.relative_to(base)
        except ValueError:
            rel = qmd
        print(f"  {rel}", file=sys.stderr)
        for line_no, kind, snippet in violations:
            label = "markdown-image" if kind == "markdown-image-fig" else "chunk fig-cap/fig-alt"
            print(f"    L{line_no} ({label}): {snippet}...", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
