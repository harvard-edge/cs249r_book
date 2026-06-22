#!/usr/bin/env python3
"""Repair training.qmd {python} refs broken by over-aggressive ×/unit stripping."""

from __future__ import annotations

import re
from pathlib import Path

QMD = Path(__file__).resolve().parents[3] / "book/quarto/contents/vol1/training/training.qmd"

# `{python} Class.field_str <word — missing closing backtick before prose word
BROKEN_REF = re.compile(
    r"`\{python\}\s+((?:[A-Za-z_][\w]*\.)?[A-Za-z_]\w*_str)\s+(?![`$])([A-Za-z\"~])"
)


def main() -> None:
    text = QMD.read_text(encoding="utf-8")
    n = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return f"`{{python}} {m.group(1)}` {m.group(2)}"

    text = BROKEN_REF.sub(repl, text)

    # `_str.` at sentence end without backtick
    text2, n2 = re.subn(
        r"`\{python\}\s+((?:[A-Za-z_][\w]*\.)?[A-Za-z_]\w*_str)\.",
        r"`{python} \1`.",
        text,
    )

    QMD.write_text(text2, encoding="utf-8")
    print(f"Repaired {n} broken refs, {n2} sentence-end refs")


if __name__ == "__main__":
    main()
