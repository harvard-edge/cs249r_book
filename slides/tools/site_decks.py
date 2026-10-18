#!/usr/bin/env python3
"""Write each volume page's lecture list from the decks themselves.

One row per chapter, grouped under the book's parts. A chapter taught in one
lecture has one line; a chapter taught in two (deck folders <NN>_<name> and
<NN>b_<name>) lists Part 1 and Part 2 under the chapter title. Each line links
to the deck's PDF, its PowerPoint file, and its source folder.

Titles come from each deck's lecture_map.yml (`title`, `subtitle`), falling
back to the .tex (`chaptertitle=` or `\\title{}`).

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

# The book's parts, by first chapter number; chapters before the first entry
# are listed under "Introduction". Taken from books/config/_quarto-pdf-volN.yml.
PARTS = {
    "vol1": [(0, "Introduction"), (2, "Part I: Foundations"),
             (5, "Part II: Development"), (9, "Part III: Optimization"),
             (13, "Part IV: Deployment")],
    "vol2": [(0, "Introduction"), (2, "Part I: The Fleet"),
             (5, "Part II: Distributed ML"), (9, "Part III: Deployment at Scale"),
             (13, "Part IV: The Responsible Fleet")],
}


def deck_info(deck: Path, tex: Path) -> tuple:
    title, subtitle = "", ""
    m = deck / "lecture_map.yml"
    if m.exists():
        data = yaml.safe_load(m.read_text()) or {}
        title = str(data.get("title") or "")
        subtitle = str(data.get("subtitle") or "")
    if not title:
        src = tex.read_text(errors="replace")
        for pat in (r"chaptertitle=\{([^}]*)\}", r"\\title\{([^}]*)\}"):
            found = re.search(pat, src)
            if found:
                title = found.group(1)
                break
    title = title or deck.name
    if not subtitle and ": " in title:
        title, subtitle = title.split(": ", 1)
    return title, subtitle


def links(vol: str, deck: str) -> str:
    name = f"{vol}_{deck}"
    return (
        '<span class="lec-links">'
        f'<a href="{RELEASE}/{name}.pdf">PDF</a>'
        f'<a href="{RELEASE}/{name}.pptx">PowerPoint</a>'
        f'<a href="{SOURCE}/{vol}/{deck}">Source</a>'
        "</span>"
    )


def chapters(vol: str) -> dict:
    """Map chapter number -> list of (deck folder, title, subtitle)."""
    out = {}
    for deck in sorted((ROOT / vol).iterdir()):
        tex = deck / f"{deck.name}.tex"
        num = re.match(r"(\d+)(b?)_", deck.name)
        if not deck.is_dir() or not tex.exists() or not num:
            continue
        title, subtitle = deck_info(deck, tex)
        out.setdefault(int(num.group(1)), []).append((deck.name, title, subtitle))
    return out


def chapter_html(vol: str, n: int, decks: list) -> list:
    esc = html.escape
    rows = [f'  <div class="ch"><div class="ch-num">{n}</div><div class="ch-body">']
    if len(decks) == 1:
        deck, title, subtitle = decks[0]
        sub = f'<span class="lec-sub">{esc(subtitle)}</span>' if subtitle else ""
        rows.append(f'    <div class="lec"><span class="lec-name"><span class="ch-title">'
                    f'{esc(title)}</span>{sub}</span>{links(vol, deck)}</div>')
    else:
        chapter = re.split(r", Part \d", decks[0][1])[0]
        rows.append(f'    <div class="ch-title">{esc(chapter)}</div>')
        for k, (deck, _title, subtitle) in enumerate(decks, start=1):
            sub = f'<span class="lec-sub">{esc(subtitle)}</span>' if subtitle else ""
            rows.append(f'    <div class="lec lec-part"><span class="lec-name">'
                        f'<span class="part-tag">Part {k}</span>{sub}</span>'
                        f'{links(vol, deck)}</div>')
    rows.append("  </div></div>")
    return rows


def page_block(vol: str) -> tuple:
    chs = chapters(vol)
    parts = PARTS.get(vol, [(0, "")])
    body, n_decks = [], 0
    for i, (start, label) in enumerate(parts):
        end = parts[i + 1][0] if i + 1 < len(parts) else 10**6
        nums = [n for n in sorted(chs) if start <= n < end]
        if not nums:
            continue
        if label:
            body.append(f'  <div class="part-head">{html.escape(label)}</div>')
        for n in nums:
            body += chapter_html(vol, n, chs[n])
            n_decks += len(chs[n])
    return body, n_decks


def main():
    for page in sorted(ROOT.glob("vol[0-9].qmd")):
        vol = page.stem
        body, n = page_block(vol)
        if not body:
            continue
        block = "\n".join(["<!-- decks:start -->", "```{=html}",
                           '<div class="lecture-list">', *body, "</div>",
                           "```", "<!-- decks:end -->"])
        text = page.read_text()
        new = re.sub(r"<!-- decks:start -->.*?<!-- decks:end -->", lambda _: block,
                     text, flags=re.S)
        if new != text:
            page.write_text(new)
        print(f"{page.name}: {n} decks")


if __name__ == "__main__":
    main()
