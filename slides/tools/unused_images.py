#!/usr/bin/env python3
"""List the files in a deck's images/ folder that the deck does not use.

A file counts as used when its stem, or its stem with a trailing step number
removed (name_step3 -> name_step, name3 -> name), appears in the deck's .tex
or in images/make_figs.py. Decks reference step figures by prefix through
\\mlsyssteps{name_step}{n}, so a full-name match alone would wrongly report
them unused. Generator outputs, SOURCES.yml, and data folders are kept.

Usage: unused_images.py slides/vol1/<NN_dir>
Prints one path per line. It does not delete anything; after removing files,
rebuild the deck and confirm the page count and fit check are unchanged.
"""

import re
import sys
from pathlib import Path

KEEP = {"make_figs.py", "SOURCES.yml"}


def main():
    deck = Path(sys.argv[1])
    tex = next(deck.glob("*.tex")).read_text(errors="replace")
    gen = deck / "images" / "make_figs.py"
    src = tex + (gen.read_text(errors="replace") if gen.exists() else "")
    for f in sorted((deck / "images").iterdir()):
        if f.is_dir() or f.name in KEEP:
            continue
        stem = f.stem
        prefix = re.sub(r"\d+$", "", stem)
        # A whole-name match: "cover_x" inside "cover_x_blueprint" is not a use.
        whole = re.search(re.escape(stem) + r"(?![A-Za-z0-9_])", src)
        if whole or (prefix and prefix != stem and prefix in src):
            continue
        print(f)


if __name__ == "__main__":
    main()
