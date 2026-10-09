"""Cross-check notation_strict.extract_math against Pandoc's own parse.

Pandoc is the parser the book build uses. For every .qmd under books/vol*/,
the multiset of math strings our extractor finds must equal the multiset of
Math nodes in Pandoc's JSON AST. Run directly (`python3 -m pytest` or
`python3 binder/tests/test_notation_strict_extraction.py [files...]`).
"""

from __future__ import annotations

import collections
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "binder" / "cli" / "checks"))
from notation_strict import extract_math  # noqa: E402

PANDOC = shutil.which("pandoc") or "/Applications/quarto/bin/tools/aarch64/pandoc"
PANDOC_FORMAT = "markdown+tex_math_dollars-tex_math_single_backslash-raw_tex"


def _strip_for_pandoc(src: str) -> str:
    # Quarto-only syntax Pandoc's markdown reader would misread as text, not math:
    # inline python refs and shortcodes. Replacing them keeps math boundaries intact.
    src = re.sub(r"`\{python\}[^`\n]*`", "X", src)
    src = re.sub(r"\{\{<[^>\n]*>\}\}", "X", src)
    # Quarto runs executable cells itself; Pandoc would not recognize the
    # `{python}` fence info, so present them as ordinary fenced code.
    src = re.sub(r"^(\s*)(`{3,}|~{3,})\{(\w+)[^}\n]*\}", r"\1\2\3", src, flags=re.M)
    return src


def pandoc_math(src: str) -> collections.Counter:
    out = subprocess.run([PANDOC, "-f", PANDOC_FORMAT, "-t", "json"],
                         input=_strip_for_pandoc(src), capture_output=True, text=True, check=True).stdout
    found: collections.Counter = collections.Counter()

    def walk(node):
        if isinstance(node, dict):
            if node.get("t") == "Math":
                found[" ".join(node["c"][1].split())] += 1
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(json.loads(out))
    return found


def ours(src: str) -> collections.Counter:
    return collections.Counter(" ".join(s.text.split()) for s in extract_math(_strip_for_pandoc(src), include_captions=False))


def compare(path: Path):
    src = path.read_text(encoding="utf-8", errors="replace")
    a, b = ours(src), pandoc_math(src)
    return a - b, b - a


def test_matches_pandoc():
    if not Path(PANDOC).exists():
        return
    files = sorted((ROOT / "books").glob("vol*/**/*.qmd"))
    bad = {}
    for f in files:
        extra, missing = compare(f)
        if extra or missing:
            bad[str(f.relative_to(ROOT))] = (sum(extra.values()), sum(missing.values()))
    assert not bad, bad


if __name__ == "__main__":
    targets = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "books").glob("vol*/**/*.qmd"))
    total_extra = total_missing = 0
    for f in targets:
        extra, missing = compare(f)
        total_extra += sum(extra.values())
        total_missing += sum(missing.values())
        if extra or missing:
            print(f"{f}: +{sum(extra.values())} only-ours, -{sum(missing.values())} only-pandoc")
            for k, n in list(extra.items())[:3]:
                print(f"    only ours   x{n}: {k[:90]!r}")
            for k, n in list(missing.items())[:3]:
                print(f"    only pandoc x{n}: {k[:90]!r}")
    print(f"files={len(targets)} only-ours={total_extra} only-pandoc={total_missing}")
