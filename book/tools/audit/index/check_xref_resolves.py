#!/usr/bin/env python3
"""
book-index-xref-resolves hook.

Fails if any \index{X|see{Y}} or \index{X|seealso{Y}} target Y doesn't
resolve to a real main entry in the corpus.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
CONTENT = ROOT / "book" / "quarto" / "contents"
INDEX_RE = re.compile(r"\\index\{([^{}]*(?:\{[^{}]*\}[^{}]*)?)\}")
SEEREF_RE = re.compile(r"^([^|]+)\|see(?:also)?\{([^}]+)\}$")


def main():
    main_heads = set()
    see_refs = []
    for f in sorted(CONTENT.rglob("*.qmd")):
        s = str(f)
        if any(x in s for x in ("frontmatter/", "backmatter/", "/parts/",
                                "/glossary/", "/appendix", "_shelved")):
            continue
        text = f.read_text(errors="replace")
        for m in INDEX_RE.finditer(text):
            k = m.group(1)
            sm = SEEREF_RE.match(k)
            if sm:
                see_refs.append((f.relative_to(ROOT), sm.group(1).strip(),
                                 sm.group(2).strip()))
            else:
                # Main entry — record headword
                h = k.split("!", 1)[0]
                if "@" in h:
                    h = h.split("@", 1)[1]
                main_heads.add(h)

    broken = [(rel, src, tgt) for rel, src, tgt in see_refs if tgt not in main_heads]
    if broken:
        print(f"Index cross-reference audit FAILED: {len(broken)} unresolved targets")
        for rel, src, tgt in broken[:15]:
            print(f"  {rel}: '{src}' -> '{tgt}' (target not found)")
        return 1
    print(f"Index cross-reference audit PASSED ({len(see_refs)} cross-refs, all resolve)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
