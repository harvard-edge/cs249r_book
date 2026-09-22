#!/usr/bin/env python3
"""Fail clearly when the narrative PDF's code-glyph fonts are unavailable."""
import shutil
import subprocess
import sys


def main():
    if not shutil.which("lualatex") or not shutil.which("kpsewhich"):
        sys.exit("Narrative PDF requires LuaLaTeX and kpsewhich on PATH (TeX Live or TinyTeX).")
    missing = []
    for font in ("DejaVuSans.ttf", "NotoEmoji-Regular.ttf", "NotoColorEmoji.ttf"):
        result = subprocess.run(["kpsewhich", font], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode or not result.stdout.strip():
            missing.append(font)
    if missing:
        sys.exit("Narrative PDF code-glyph fonts missing: " + ", ".join(missing)
                 + ". Install into the TeX distribution on PATH: tlmgr install dejavu noto-emoji")


if __name__ == "__main__":
    main()
