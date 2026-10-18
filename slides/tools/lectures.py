#!/usr/bin/env python3
"""Link the lecture decks and the book chapters they teach, from one source.

The source of truth is the deck folders themselves: slides/volN/<deck>/ and
its lecture_map.yml (`chapter`, `lecture`, `title`, `subtitle`). A deck
without a lecture_map.yml falls back to its folder name for the chapter and
to the .tex (`chaptertitle=` or `\\title{}`) for the title. Deck folders
<NN>_<name> and <NN>b_<name> are Part 1 and Part 2 of chapter NN. Lecture 0
(the course overview) has no book chapter.

Each chapter's book URL comes from the book's own HTML config,
books/config/_quarto-html-volN.yml: Quarto renders a chapter listed there as
volN/<chapter>/<chapter>.qmd to volN/<chapter>/<chapter>.html under the
output directory, and the publish step (.github/scripts/flatten-vol-urls.sh)
moves volN/ up to the site root, so the page is served at
<site-url><chapter>/<chapter>.html. A deck that names a chapter the config
does not list stops the build.

Two outputs, both written at build time:

1. Slides site (default). The lecture list between the markers
       <!-- decks:start -->  and  <!-- decks:end -->
   in slides/volN.qmd: one row per chapter, grouped under the book's parts,
   each lecture linking to its PDF, its PowerPoint file, and its source
   folder, and each chapter row linking to the chapter in the book. Runs as
   the slides site's Quarto pre-render, so the page never carries a
   hand-typed deck list.

2. Book site (--manifest PATH). A JSON manifest, per volume and chapter
   folder, of the slides page anchor for that chapter and its lectures. Runs
   as a pre-render of each books/config/_quarto-html-volN.yml; the HTML
   filter books/shared/filters/inject_slides.lua reads it.

Usage: python3 tools/lectures.py                         (from slides/)
       python3 ../slides/tools/lectures.py --manifest _build/slides/lectures.json
                                                         (from books/)
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT.parent / "books"
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


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def site_url(config: dict) -> str:
    url = str((config.get("website") or {}).get("site-url") or "")
    return url if url.endswith("/") else url + "/"


def book_chapters(vol: str) -> dict:
    """Map chapter folder -> book page URL, for every chapter the config lists."""
    config_file = BOOKS / "config" / f"_quarto-html-{vol}.yml"
    if not config_file.exists():
        return {}
    config = read_yaml(config_file)
    base = site_url(config)
    chapter = re.compile(rf"^{vol}/((\d+_[^/]+)/\2)\.qmd$")
    out = {}
    for qmd in (config.get("project") or {}).get("render") or []:
        found = chapter.match(str(qmd))
        if found:
            out[found.group(2)] = f"{base}{found.group(1)}.html"
    return out


def deck_info(deck: Path, tex: Path) -> dict:
    data = {}
    m = deck / "lecture_map.yml"
    if m.exists():
        data = read_yaml(m)
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
    chapter = str(data.get("chapter") or re.sub(r"^(\d+)b_", r"\1_", deck.name))
    return {"deck": deck.name, "title": title, "subtitle": subtitle,
            "chapter": chapter, "lecture": str(data.get("lecture", deck.name.split("_")[0]))}


def lecture_links(vol: str, deck: str) -> dict:
    name = f"{vol}_{deck}"
    return {"pdf": f"{RELEASE}/{name}.pdf", "pptx": f"{RELEASE}/{name}.pptx",
            "source": f"{SOURCE}/{vol}/{deck}"}


def volume(vol: str, slides_page: str) -> list:
    """The chapters of one volume in order, each with its book URL and lectures."""
    book = book_chapters(vol)
    by_num = {}
    for deck in sorted((ROOT / vol).iterdir()):
        tex = deck / f"{deck.name}.tex"
        num = re.match(r"(\d+)(b?)_", deck.name)
        if not deck.is_dir() or not tex.exists() or not num:
            continue
        info = deck_info(deck, tex)
        n = int(num.group(1))
        if n == 0:
            info["chapter"] = ""
        elif info["chapter"] not in book:
            sys.exit(f"lectures.py: {vol}/{deck.name} names chapter "
                     f"'{info['chapter']}', which books/config/_quarto-html-{vol}.yml "
                     "does not list.")
        by_num.setdefault(n, []).append(info)

    chapters = []
    for n in sorted(by_num):
        decks = by_num[n]
        anchor = f"ch-{n:02d}"
        if len(decks) == 1:
            title = decks[0]["title"]
        else:
            title = re.split(r", Part \d", decks[0]["title"])[0]
        lectures = []
        for k, info in enumerate(decks, start=1):
            lec_anchor = f"lec-{n:02d}" + ("b" if k > 1 else "")
            lectures.append({**info, **lecture_links(vol, info["deck"]),
                             "label": f"Part {k}" if len(decks) > 1 else "",
                             "anchor": lec_anchor,
                             "slides_url": f"{slides_page}#{lec_anchor}"})
        chapters.append({"num": n, "chapter": decks[0]["chapter"], "title": title,
                         "book_url": book.get(decks[0]["chapter"], ""),
                         "anchor": anchor, "slides_url": f"{slides_page}#{anchor}",
                         "lectures": lectures})
    return chapters


def link_html(lec: dict) -> str:
    return (
        '<span class="lec-links">'
        f'<a href="{lec["pdf"]}">PDF</a>'
        f'<a href="{lec["pptx"]}">PowerPoint</a>'
        f'<a href="{lec["source"]}">Source</a>'
        "</span>"
    )


def chapter_html(ch: dict) -> list:
    esc = html.escape
    read = (f'<a class="ch-read" href="{ch["book_url"]}">Read the chapter</a>'
            if ch["book_url"] else "")
    rows = [f'  <div class="ch" id="{ch["anchor"]}"><div class="ch-num">{ch["num"]}</div>'
            '<div class="ch-body">']
    lectures = ch["lectures"]
    if len(lectures) == 1:
        lec = lectures[0]
        sub = f'<span class="lec-sub">{esc(lec["subtitle"])}</span>' if lec["subtitle"] else ""
        rows.append(f'    <div class="lec" id="{lec["anchor"]}"><span class="lec-name">'
                    f'<span class="ch-title">{esc(ch["title"])}</span>{sub}{read}</span>'
                    f'{link_html(lec)}</div>')
    else:
        rows.append(f'    <div class="ch-head"><span class="ch-title">{esc(ch["title"])}</span>'
                    f'{read}</div>')
        for lec in lectures:
            sub = f'<span class="lec-sub">{esc(lec["subtitle"])}</span>' if lec["subtitle"] else ""
            rows.append(f'    <div class="lec lec-part" id="{lec["anchor"]}"><span class="lec-name">'
                        f'<span class="part-tag">{lec["label"]}</span>{sub}</span>'
                        f'{link_html(lec)}</div>')
    rows.append("  </div></div>")
    return rows


def page_block(chapters: list, vol: str) -> list:
    parts = PARTS.get(vol, [(0, "")])
    body = []
    for i, (start, label) in enumerate(parts):
        end = parts[i + 1][0] if i + 1 < len(parts) else 10**6
        group = [ch for ch in chapters if start <= ch["num"] < end]
        if not group:
            continue
        if label:
            body.append(f'  <div class="part-head">{html.escape(label)}</div>')
        for ch in group:
            body += chapter_html(ch)
    return body


def volumes() -> dict:
    base = site_url(read_yaml(ROOT / "_quarto.yml"))
    out = {}
    for page in sorted(ROOT.glob("vol[0-9].qmd")):
        vol = page.stem
        if (ROOT / vol).is_dir():
            out[vol] = volume(vol, f"{base}{vol}.html")
    return out


def write_pages(vols: dict) -> None:
    for vol, chapters in vols.items():
        page = ROOT / f"{vol}.qmd"
        body = page_block(chapters, vol)
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
        n = sum(len(ch["lectures"]) for ch in chapters)
        print(f"{page.name}: {n} decks")


def write_manifest(vols: dict, path: Path) -> None:
    manifest = {
        vol: {ch["chapter"]: {"title": ch["title"], "slides_url": ch["slides_url"],
                              "lectures": [{"label": lec["label"], "url": lec["slides_url"]}
                                           for lec in ch["lectures"]]}
              for ch in chapters if ch["chapter"]}
        for vol, chapters in vols.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    n = sum(len(v) for v in manifest.values())
    print(f"lectures.py: {n} chapters with lectures -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path,
                        help="write the book's JSON manifest here instead of the slides pages")
    args = parser.parse_args()
    vols = volumes()
    if args.manifest:
        write_manifest(vols, args.manifest)
    else:
        write_pages(vols)


if __name__ == "__main__":
    main()
