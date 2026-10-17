#!/usr/bin/env python3
"""Write each volume page's deck list from the decks themselves.

For every deck folder with a .tex file, one row: the lecture number, the
title, and links to the PDF, the PowerPoint file, and the source. The title
comes from the deck's lecture_map.yml when it has one, else from the .tex
(`chaptertitle=` or `\\title{}`). Decks named <NN>b_<name> are the second
lecture on a chapter and are listed as <NN>b.

The list is written between the markers
    <!-- decks:start -->  and  <!-- decks:end -->
in volN.qmd, so the page never carries a hand-typed deck list.

Usage: python3 tools/site_decks.py            (from slides/)
"""

import html
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/harvard-edge/cs249r_book/releases/download/slides-latest"
SOURCE = "https://github.com/harvard-edge/cs249r_book/tree/dev/slides"


def deck_title(deck: Path, tex: Path) -> str:
    m = deck / "lecture_map.yml"
    if m.exists():
        t = (yaml.safe_load(m.read_text()) or {}).get("title")
        if t:
            return str(t)
    src = tex.read_text(errors="replace")
    for pat in (r"chaptertitle=\{([^}]*)\}", r"\\title\{([^}]*)\}"):
        found = re.search(pat, src)
        if found:
            return found.group(1)
    return deck.name


def rows(vol: str) -> list:
    out = []
    for deck in sorted((ROOT / vol).iterdir()):
        tex = deck / f"{deck.name}.tex"
        if not deck.is_dir() or not tex.exists():
            continue
        num = re.match(r"(\d+b?)_", deck.name)
        label = (num.group(1).lstrip("0") or "0") if num else ""
        name = f"{vol}_{deck.name}"
        out.append(
            f'    <tr><td class="num">{label}</td>'
            f"<td>{html.escape(deck_title(deck, tex))}</td>"
            f'<td class="links"><a href="{RELEASE}/{name}.pdf">PDF</a>'
            f' · <a href="{RELEASE}/{name}.pptx">PowerPoint</a>'
            f' · <a href="{SOURCE}/{vol}/{deck.name}">Source</a></td></tr>'
        )
    return out


def main():
    for page in sorted(ROOT.glob("vol[0-9].qmd")):
        vol = page.stem
        body = rows(vol)
        if not body:
            continue
        block = "\n".join(
            ["<!-- decks:start -->", "```{=html}", '<table class="deck-list">',
             '  <thead><tr><th>Lecture</th><th>Title</th><th class="links">Download</th></tr></thead>',
             "  <tbody>", *body, "  </tbody>", "</table>", "```", "<!-- decks:end -->"]
        )
        text = page.read_text()
        new = re.sub(r"<!-- decks:start -->.*?<!-- decks:end -->", block, text,
                     flags=re.S)
        if new != text:
            page.write_text(new)
        print(f"{page.name}: {len(body)} decks")


if __name__ == "__main__":
    main()
