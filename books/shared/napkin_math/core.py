"""Napkin Math sheets: shared page model, HTML rendering, and PDF printing.

A volume's sheet is a Python module (vol1.py ... vol4.py) whose build()
returns a Sheet. Every number on a sheet is computed in that module from the
mlsysim registry or scenarios, or declared as an example input and tagged
with assumed(). This module only lays the content out.

Layout contract: two US Letter landscape pages. Each page has a header, a
notation strip, an optional full-width band, three columns of items, and an
optional bottom table. render() + to_pdf() report any item that runs past its
page or any equation wider than its column; a sheet ships only with none.
"""
from __future__ import annotations

import html as _html
import math
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- markup helpers


def v(name: str, sub: str | None = None, sup: str | None = None) -> str:
    """A math variable in italic serif, with optional subscript/superscript."""
    s = f"<i>{name}</i>"
    if sub:
        s += f"<sub>{sub}</sub>"
    if sup:
        s += f"<sup>{sup}</sup>"
    return s


def frac(num: str, den: str) -> str:
    return f'<span class="frac"><span>{num}</span><span>{den}</span></span>'


def n(text: str, warn: bool = False) -> str:
    """A result number, highlighted."""
    return f'<span class="n{" warn" if warn else ""}">{text}</span>'


def assumed() -> str:
    """Tag placed immediately before an example input chosen for the example."""
    return '<span class="tag">assumed</span>'


def fmt(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}"


def sig(x: float, digits: int = 2) -> str:
    """Round to significant digits, with thousands separators."""
    if x == 0:
        return "0"
    d = max(0, digits - 1 - int(math.floor(math.log10(abs(x)))))
    return f"{round(x, d):,.{d}f}"


# ---------------------------------------------------------------- page model


@dataclass
class Card:
    title: str
    eq: str | list[str] | None = None      # one or more equation lines (HTML)
    holds: str | None = None               # when it holds / what the symbols mean
    ex: str | None = None                  # worked example with computed numbers
    chapter: str | None = None             # e.g. "Ch 8"


@dataclass
class Table:
    head: list[str]
    rows: list[list[str]]
    caption: str | None = None


@dataclass
class Chart:
    svg: str
    caption: str | None = None


@dataclass
class Note:
    title: str
    text: str
    warn: bool = False


@dataclass
class Know:
    """A number to know: value and what it is. Shown in the strip on page 1."""
    value: str
    label: str


@dataclass
class Col:
    title: str
    items: list
    chapters: str = ""


@dataclass
class Page:
    question: str              # e.g. "Will it run?"
    sub: str                   # one-line description under the title
    notation: list[tuple[str, str]]   # (group label, HTML)
    cols: list[Col]
    know: list[Know] = field(default_factory=list)
    band: str | None = None    # full-width HTML band under the notation
    bottom: Table | None = None
    meta: str = ""


@dataclass
class Sheet:
    volume: int
    book: str                  # e.g. "Introduction to Machine Learning Systems"
    scope: str                 # e.g. "One Node"
    pages: list[Page]
    sources: str               # footer provenance line


# ---------------------------------------------------------------- rendering

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}

# Each sheet takes its volume's print accent (books/volN/tex/theme-colors.tex),
# so a printed sheet matches its book. "warn" marks results that fail a limit
# and must stay distinct from the accent hue.
THEME = {
    1: {"accent": "#A51C30", "warn": "#B45309"},   # Harvard crimson; amber warn
    2: {"accent": "#1F407A", "warn": "#B4472F"},   # ETH blue
    3: {"accent": "#581C87", "warn": "#B4472F"},   # Royal amethyst
    4: {"accent": "#1A4D3E", "warn": "#B4472F"},   # Emerald pine
}


def _tint(hex_color: str, share: float) -> str:
    """Mix a color with white; share is the fraction of the color kept."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: round(255 - (255 - c) * share)
    return "#{:02X}{:02X}{:02X}".format(mix(r), mix(g), mix(b))


def theme_css(volume: int) -> str:
    t = THEME[volume]
    return (f":root {{ --accent: {t['accent']}; --accent-soft: {_tint(t['accent'], 0.09)}; "
            f"--accent-mid: {_tint(t['accent'], 0.35)}; --warn: {t['warn']}; }}")
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600;700"
         "&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500"
         "&family=STIX+Two+Text:ital@0;1&display=swap")
CSS_PATH = Path(__file__).parent / "sheet.css"


def _item(it) -> str:
    if isinstance(it, Card):
        eqs = it.eq if isinstance(it.eq, list) else ([it.eq] if it.eq else [])
        ch = f'<span class="ch">{it.chapter}</span>' if it.chapter else ""
        out = [f'<div class="item"><h3><span>{it.title}</span>{ch}</h3>']
        out += [f'<div class="eq">{e}</div>' for e in eqs]
        if it.holds:
            out.append(f'<p class="holds">{it.holds}</p>')
        if it.ex:
            out.append(f'<p class="ex">{it.ex}</p>')
        out.append("</div>")
        return "".join(out)
    if isinstance(it, Table):
        cap = f"<caption>{it.caption}</caption>" if it.caption else ""
        head = "".join(f"<th>{h}</th>" for h in it.head)
        rows = "".join("<tr>" + "".join(
            f'<td class="l">{c}</td>' if k == 0 else f"<td>{c}</td>" for k, c in enumerate(r)) + "</tr>" for r in it.rows)
        return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
    if isinstance(it, Chart):
        cap = f'<p class="figcap">{it.caption}</p>' if it.caption else ""
        return it.svg + cap
    if isinstance(it, Note):
        return f'<div class="callout{" warn" if it.warn else ""}"><b>{it.title}</b> {it.text}</div>'
    if isinstance(it, str):
        return it
    raise TypeError(type(it))


def render(sheet: Sheet) -> str:
    vol = ROMAN[sheet.volume]
    pages = []
    for k, pg in enumerate(sheet.pages, 1):
        rows = ["auto", "auto"]
        parts = [f'''<div class="head"><div>
<h1>Napkin Math: {sheet.scope} <span>· {pg.question}</span></h1>
<div class="sub">{pg.sub}</div></div>
<div class="meta">{sheet.book} · Volume {vol}<br>{pg.meta} · {k} / {len(sheet.pages)}</div></div>''']
        notation = "".join(f"<span><b>{g}</b>{h}</span>" for g, h in pg.notation)
        parts.append(f'<div class="notation">{notation}</div>')
        if pg.know:
            rows.append("auto")
            parts.append('<div class="know"><div class="know-h">Numbers to know</div>' + "".join(
                f"<div><b>{x.value}</b><span>{x.label}</span></div>" for x in pg.know) + "</div>")
        if pg.band:
            rows.append("auto")
            parts.append(f'<div class="band">{pg.band}</div>')
        rows.append("1fr")
        cols = []
        for c in pg.cols:
            ch = f"<small>{c.chapters}</small>" if c.chapters else ""
            cols.append(f'<div class="col"><h2><span>{c.title}</span>{ch}</h2>' + "".join(_item(i) for i in c.items) + "</div>")
        parts.append('<div class="board">' + "".join(cols) + "</div>")
        rows.append("auto")
        if pg.bottom:
            parts.append('<div class="bottom">' + _item(pg.bottom) + "</div>")
        else:
            parts.append(f'<div class="foot"><span>{sheet.sources}</span><span>Machine Learning Systems · mlsysbook.ai</span></div>')
        pages.append(f'<section class="page" style="grid-template-rows:{" ".join(rows)}">' + "".join(parts) + "</section>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Napkin Math: {sheet.scope}</title>'
            f'<link rel="stylesheet" href="{FONTS}"><style>{CSS_PATH.read_text()}{theme_css(sheet.volume)}</style></head><body>' + "".join(pages) + "</body></html>")


LAYOUT_CHECK = """() => {
  const issues = [];
  document.querySelectorAll('.page').forEach((pg, k) => {
    const pr = pg.getBoundingClientRect();
    const lim = pg.querySelector('.foot, .bottom');
    const stop = lim ? lim.getBoundingClientRect().top : pr.bottom;
    pg.querySelectorAll('.col > *').forEach(el => {
      const r = el.getBoundingClientRect();
      if (r.bottom > stop + 0.5) issues.push(`p${k+1} runs past the page: ${(el.textContent||'').trim().slice(0,50)}`);
    });
    pg.querySelectorAll('.eq, table, .notation, .band, .know').forEach(el => {
      if (el.scrollWidth > el.clientWidth + 1) issues.push(`p${k+1} too wide: ${(el.textContent||'').trim().slice(0,50)}`);
    });
    pg.querySelectorAll('*').forEach(el => { const r = el.getBoundingClientRect();
      if (r.width && (r.right > pr.right + 0.5 || r.bottom > pr.bottom + 0.5)) issues.push(`p${k+1} off page: ${el.tagName} ${(el.textContent||'').trim().slice(0,30)}`); });
    const free = [...pg.querySelectorAll('.col')].map(c => Math.round(stop - c.lastElementChild.getBoundingClientRect().bottom));
    issues.push(`p${k+1} free px per column: ${free.join(', ')}`);
  });
  return issues;
}"""


def to_pdf(html_path: Path, pdf_path: Path) -> list[str]:
    """Print a rendered sheet to PDF with headless Chromium; return the layout report."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1100, "height": 900})
        pg.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        pg.wait_for_timeout(600)
        report = pg.evaluate(LAYOUT_CHECK)
        pg.pdf(path=str(pdf_path), width="11in", height="8.5in", print_background=True, prefer_css_page_size=True)
        b.close()
    return report


# ---------------------------------------------------------------- simple SVG charts


class Plot:
    """A small log/linear plot in a fixed viewBox, coordinates computed from data."""

    def __init__(self, x_range, y_range, xlog=False, ylog=False, W=340, H=140, left=36, right=18, top=12, bottom=26):
        self.W, self.H, self.l, self.r, self.t, self.b = W, H, left, right, top, bottom
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.xlog, self.ylog = xlog, ylog
        self.parts: list[str] = []

    def X(self, x):
        f = (math.log10(x) - math.log10(self.x0)) / (math.log10(self.x1) - math.log10(self.x0)) if self.xlog \
            else (x - self.x0) / (self.x1 - self.x0)
        return self.l + f * (self.W - self.l - self.r)

    def Y(self, y):
        y = min(max(y, self.y0), self.y1)
        f = (math.log10(y) - math.log10(self.y0)) / (math.log10(self.y1) - math.log10(self.y0)) if self.ylog \
            else (y - self.y0) / (self.y1 - self.y0)
        return self.H - self.b - f * (self.H - self.t - self.b)

    def grid(self, xticks, yticks, xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}"):
        for x in xticks:
            self.parts.append(f'<line x1="{self.X(x):.1f}" x2="{self.X(x):.1f}" y1="{self.t}" y2="{self.H-self.b}" class="grid"/>'
                              f'<text x="{self.X(x):.1f}" y="{self.H-self.b+11}" class="tick" text-anchor="middle">{xfmt(x)}</text>')
        for y in yticks:
            self.parts.append(f'<line x1="{self.l}" x2="{self.W-self.r}" y1="{self.Y(y):.1f}" y2="{self.Y(y):.1f}" class="grid"/>'
                              f'<text x="{self.l-4}" y="{self.Y(y)+3:.1f}" class="tick" text-anchor="end">{yfmt(y)}</text>')

    def line(self, pts, cls="line"):
        self.parts.append(f'<polyline points="{" ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in pts)}" class="{cls}"/>')

    def point(self, x, y, label=None, anchor="start", dx=5, dy=4):
        self.parts.append(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="3" class="pt"/>')
        if label:
            self.text(self.X(x) + dx, self.Y(y) + dy, label, anchor)

    def text(self, px, py, s, anchor="start", cls="lab"):
        self.parts.append(f'<text x="{px:.1f}" y="{py:.1f}" class="{cls}" text-anchor="{anchor}">{_html.escape(s)}</text>')

    def axes(self, xlabel, ylabel):
        self.text((self.l + self.W - self.r) / 2, self.H - 3, xlabel, "middle", "ax")
        cy = (self.t + self.H - self.b) / 2
        self.parts.append(f'<text x="9" y="{cy:.1f}" class="ax" text-anchor="middle" transform="rotate(-90 9 {cy:.1f})">{_html.escape(ylabel)}</text>')

    def svg(self, label: str) -> str:
        return f'<svg viewBox="0 0 {self.W} {self.H}" class="chart" role="img" aria-label="{_html.escape(label)}">' + "".join(self.parts) + "</svg>"
