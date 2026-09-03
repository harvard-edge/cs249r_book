#!/usr/bin/env python3
"""
visible_text.py — render a prose line to the *characters a reader sees*.

Used by the fmt migration's equivalence check (ASSESSMENT.md §0a, Regime 2):
a multiplier migration moves the glyph from the string into the prose
(`"6×"` → `"6"` + prose `$\\times$`), so comparing raw strings is the wrong
test. Normalizing both sides to visible text makes the composite
(value ⊕ surrounding prose) comparable:

    to_visible("...is 6× the...")           == "...is 6× the..."
    to_visible("...is 6$\\times$ the...")    == "...is 6× the..."   ✓ equal

This is NOT a LaTeX engine. It maps the small, closed set of inline-math glyphs
the migration actually relocates (and a few common neighbors) to Unicode, drops
spacing macros, unwraps text wrappers, and strips ``$`` delimiters — leaving any
genuinely complex math as a *stable* residue (identical before/after when
untouched, which is all the diff needs).
"""
from __future__ import annotations

import re

# LaTeX command → visible Unicode glyph (longest names first to avoid prefixes)
_GLYPHS = [
    (r"\times", "×"),
    (r"\cdot", "·"),
    (r"\approx", "≈"),
    (r"\leq", "≤"), (r"\geq", "≥"), (r"\neq", "≠"),
    (r"\le", "≤"), (r"\ge", "≥"),
    (r"\pm", "±"), (r"\mp", "∓"),
    (r"\sim", "~"),
    (r"\rightarrow", "→"), (r"\to", "→"), (r"\leftarrow", "←"),
    (r"\Rightarrow", "⇒"),
    (r"\ll", "≪"), (r"\gg", "≫"),
    (r"\infty", "∞"),
    (r"\mu", "μ"), (r"\eta", "η"), (r"\alpha", "α"), (r"\beta", "β"),
    (r"\sqrt", "√"),
]

_SENTINEL_USD = "\x00USD\x00"
_TEXT_WRAP = re.compile(r"\\(?:text|mathrm|mathbf|mathit|mathsf|operatorname)\{([^{}]*)\}")
_SPACING = re.compile(r"\\[,;:! ]|\\quad|\\qquad")
_THINNBSP = "\u00a0\u202f\u2009"  # nbsp, narrow nbsp, thin space → normal space


def to_visible(s: str) -> str:
    """Return the reader-visible characters of a (ref-substituted) prose line."""
    # 1. protect escaped currency so the $-delimiter strip can't eat it
    s = s.replace(r"\$", _SENTINEL_USD)
    # 2. unwrap \text{...}/\mathrm{...} → inner text
    for _ in range(3):  # a few nesting levels
        s, n = _TEXT_WRAP.subn(r"\1", s)
        if not n:
            break
    # 3. escaped percent → visible percent
    s = s.replace(r"\%", "%")
    # 4. glyph commands → Unicode (word-boundary so \tox doesn't match \to)
    for cmd, glyph in _GLYPHS:
        s = re.sub(re.escape(cmd) + r"(?![A-Za-z])", glyph, s)
    # 5. drop spacing macros
    s = _SPACING.sub("", s)
    # 6. strip math delimiters now that inline glyphs are Unicode
    s = s.replace("$$", "").replace("$", "")
    # 7. restore currency
    s = s.replace(_SENTINEL_USD, "$")
    # 8. normalize whitespace (incl. LaTeX/markdown nbsp + thin spaces)
    for ch in _THINNBSP:
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        print(to_visible(line.rstrip("\n")))
