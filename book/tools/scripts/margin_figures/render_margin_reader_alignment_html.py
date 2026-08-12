#!/usr/bin/env python3
"""Render a readable HTML dashboard for margin-figure reader alignment.

The page is an editor-facing companion to the markdown reader-link audit. It
shows every placed margin figure with its SVG, caption, fig-alt evidence,
source QMD location, strongest local prose anchor, and expandable before/after
context. Use it when checking whether a student can read the paragraph, glance
at the margin figure, and see the same point in the caption.
"""

from __future__ import annotations

import argparse
import base64
import html
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from audit_margin_caption_alignment import ROOT, CaptionPacket, collect
from render_margin_reader_link_audit import (
    asset_path,
    collapse,
    rel_link,
    source_excerpt,
    strongest_anchor,
    truncate,
)


DEFAULT_OUTPUT = ROOT / "book/tools/audit/margin_figure_reader_alignment.html"
DEFAULT_VERDICTS = ROOT / "book/tools/audit/margin_figure_reader_alignment_verdicts.md"


@dataclass(frozen=True)
class VerdictInfo:
    verdict: str
    reader_link: str
    strict_reviewed: bool


@dataclass(frozen=True)
class CardStatus:
    key: str
    label: str
    note: str


def esc(text: str | None) -> str:
    return html.escape(text or "", quote=True)


def rel_href(target: Path, output: Path) -> str:
    return rel_link(target, output).replace(os.sep, "/")


def image_data_uri(path: Path) -> str:
    if path.suffix.lower() == ".svg":
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^\s*<\?xml[^>]*>\s*", "", text)
        text = re.sub(r"(?s)^\s*<!DOCTYPE.*?>\s*", "", text)
        data = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{data}"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
    return f"data:{mime};base64,{data}"


def packet_key(packet: CaptionPacket) -> tuple[str, int, str]:
    return (packet.qmd_path, packet.line, packet.asset)


def parse_verdicts(path: Path) -> dict[tuple[str, int, str], VerdictInfo]:
    if not path.exists():
        return {}

    verdicts: dict[tuple[str, int, str], VerdictInfo] = {}
    row_re = re.compile(
        r"^\|\s*\d+\s*\|\s*`(?P<source>[^`]+)`\s*\|\s*`(?P<asset>[^`]+)`"
        r"\s*\|\s*(?P<caption>.*?)\s*\|\s*(?P<verdict>[^|]+?)\s*\|"
        r"\s*(?P<link>.*)\|\s*$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = row_re.match(line)
        if not match:
            continue
        source = match.group("source")
        if ":" not in source:
            continue
        qmd_path, line_text = source.rsplit(":", 1)
        try:
            line_no = int(line_text)
        except ValueError:
            continue
        reader_link = match.group("link").strip()
        verdicts[(qmd_path, line_no, match.group("asset"))] = VerdictInfo(
            verdict=collapse(match.group("verdict")),
            reader_link=reader_link,
            strict_reviewed="Strict-review candidate" in reader_link,
        )
    return verdicts


def strict_review_candidate(packet: CaptionPacket, threshold: float) -> bool:
    return packet.status != "pass-triage" or packet.overlap < threshold


def card_status(
    packet: CaptionPacket,
    verdicts: dict[tuple[str, int, str], VerdictInfo],
    threshold: float,
) -> CardStatus:
    verdict = verdicts.get(packet_key(packet))
    if verdict:
        normalized = verdict.verdict.lower()
        if "pass after" in normalized:
            return CardStatus(
                key="bridge",
                label=verdict.verdict,
                note="Caption and visual passed after a local prose bridge was added.",
            )
        if normalized == "pass":
            if verdict.strict_reviewed or strict_review_candidate(packet, threshold):
                return CardStatus(
                    key="reviewed",
                    label="Reviewed pass",
                    note="Low lexical overlap or title-like signal; manually checked against local prose.",
                )
            return CardStatus(
                key="pass",
                label="Pass",
                note="Caption, visual evidence, and local prose align.",
            )
        return CardStatus(
            key="needs-fix",
            label=verdict.verdict,
            note="Verdict file marks this placement for follow-up.",
        )

    if packet.status.startswith("fix"):
        return CardStatus(
            key="needs-fix",
            label="Needs fix",
            note="The automated audit found a missing or invalid caption.",
        )
    if strict_review_candidate(packet, threshold):
        return CardStatus(
            key="review",
            label="Review candidate",
            note="No manual verdict was found for this low-overlap placement.",
        )
    return CardStatus(
        key="pass",
        label="Pass",
        note="Caption, visual evidence, and local prose align by triage.",
    )


def chapter_label(chapter: str) -> str:
    volume, name = chapter.split("/", 1)
    words = name.replace("_", " ").replace("-", " ").title()
    return f"{volume.upper()} / {words}"


def render_summary(
    packets: list[CaptionPacket],
    statuses: list[CardStatus],
    threshold: float,
    verdict_count: int,
) -> str:
    counts = Counter(status.key for status in statuses)
    strict_count = sum(strict_review_candidate(packet, threshold) for packet in packets)
    pass_count = counts["pass"] + counts["reviewed"] + counts["bridge"]
    remaining = counts["review"] + counts["needs-fix"]
    cards = [
        ("Figures audited", str(len(packets))),
        ("Reader-alignment passes", str(pass_count)),
        ("Strict-reviewed", str(strict_count)),
        ("Prose bridge", str(counts["bridge"])),
        ("Remaining fixes", str(remaining)),
        ("Manual verdict rows", str(verdict_count)),
    ]
    return "\n".join(
        f'<div class="metric"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'
        for label, value in cards
    )


def render_controls(packets: list[CaptionPacket]) -> str:
    chapter_counts = Counter(packet.chapter for packet in packets)
    options = ['<option value="all">All chapters</option>']
    for chapter in sorted(chapter_counts):
        options.append(
            f'<option value="{esc(chapter)}">{esc(chapter_label(chapter))} '
            f'({chapter_counts[chapter]})</option>'
        )
    return f"""
      <section class="controls" aria-label="Audit filters">
        <label class="search-label" for="search">Search</label>
        <input id="search" type="search" placeholder="caption, asset, prose, chapter..." />
        <label class="select-label" for="chapterFilter">Chapter</label>
        <select id="chapterFilter">
          {''.join(options)}
        </select>
        <label class="check"><input id="reviewedOnly" type="checkbox" /> strict-reviewed only</label>
        <label class="check"><input id="bridgeOnly" type="checkbox" /> prose bridge only</label>
        <button id="resetFilters" type="button">Reset</button>
        <span id="visibleCount" class="visible-count">{len(packets)} shown</span>
      </section>
    """


def render_sidebar(packets: list[CaptionPacket]) -> str:
    chapter_counts = Counter(packet.chapter for packet in packets)
    buttons = [
        '<button type="button" data-chapter-filter="all">'
        '<span>All chapters</span><strong>{}</strong></button>'.format(len(packets))
    ]
    for chapter in sorted(chapter_counts):
        buttons.append(
            '<button type="button" data-chapter-filter="{chapter}">'
            '<span>{label}</span><strong>{count}</strong></button>'.format(
                chapter=esc(chapter),
                label=esc(chapter_label(chapter)),
                count=chapter_counts[chapter],
            )
        )
    return f"""
      <aside class="sidebar" aria-label="Chapter navigation">
        <h2>Chapters</h2>
        <div class="chapter-list">
          {''.join(buttons)}
        </div>
      </aside>
    """


def render_context_block(label: str, text: str) -> str:
    if not text:
        text = "No adjacent prose captured."
    return f"""
      <div class="context-block">
        <h4>{esc(label)}</h4>
        <p>{esc(truncate(text, 1200))}</p>
      </div>
    """


def render_card(
    index: int,
    packet: CaptionPacket,
    status: CardStatus,
    output: Path,
    threshold: float,
) -> str:
    qmd = ROOT / packet.qmd_path
    qmd_href = rel_href(qmd, output)
    asset = asset_path(packet)
    asset_href = rel_href(asset, output)
    image_src = image_data_uri(asset)
    anchor = strongest_anchor(packet) or packet.before or packet.after
    excerpt = source_excerpt(packet)
    search_blob = " ".join(
        [
            packet.chapter,
            packet.qmd_path,
            packet.asset,
            packet.caption,
            packet.fig_alt,
            packet.before,
            packet.after,
            anchor,
        ]
    ).lower()
    review_flag = strict_review_candidate(packet, threshold)
    source_label = f"{packet.qmd_path}:{packet.line}"
    chapter = chapter_label(packet.chapter)

    return f"""
      <article
        class="figure-card"
        id="figure-{index:03d}"
        data-chapter="{esc(packet.chapter)}"
        data-status="{esc(status.key)}"
        data-review="{str(review_flag).lower()}"
        data-search="{esc(search_blob)}"
      >
        <header class="card-head">
          <div>
            <div class="eyebrow">
              <a href="#figure-{index:03d}">#{index:03d}</a>
              <span>{esc(chapter)}</span>
              <span>overlap {packet.overlap:.2f}</span>
            </div>
            <h2>{esc(packet.caption or "Missing caption")}</h2>
            <p class="source">
              <a href="{esc(qmd_href)}">{esc(source_label)}</a>
              <span>{esc(packet.asset)}</span>
            </p>
          </div>
          <div class="status-wrap">
            <span class="chip chip-{esc(status.key)}">{esc(status.label)}</span>
            <p>{esc(status.note)}</p>
          </div>
        </header>
        <div class="card-body">
          <figure class="figure-preview">
            <img src="{esc(image_src)}" alt="{esc(packet.fig_alt or packet.caption)}" />
            <figcaption><a href="{esc(asset_href)}">{esc(packet.asset)}</a></figcaption>
          </figure>
          <section class="reader-link">
            <h3>Reader Link</h3>
            <div class="evidence">
              <h4>Strongest prose anchor</h4>
              <blockquote>{esc(truncate(anchor, 850))}</blockquote>
            </div>
            <div class="evidence-pair">
              <div>
                <h4>Caption takeaway</h4>
                <p>{esc(packet.caption or "Missing caption.")}</p>
              </div>
              <div>
                <h4>Figure evidence</h4>
                <p>{esc(packet.fig_alt or "Missing fig-alt.")}</p>
              </div>
            </div>
            <details>
              <summary>Placement context and markdown</summary>
              <div class="details-grid">
                {render_context_block("Paragraph before", packet.before)}
                {render_context_block("Paragraph after", packet.after)}
              </div>
              <h4>Source markdown excerpt</h4>
              <pre><code>{esc(excerpt)}</code></pre>
            </details>
          </section>
        </div>
      </article>
    """


def render_cards(
    packets: list[CaptionPacket],
    statuses: list[CardStatus],
    output: Path,
    threshold: float,
) -> str:
    return "\n".join(
        render_card(index, packet, status, output, threshold)
        for index, (packet, status) in enumerate(zip(packets, statuses), 1)
    )


def render_html(
    packets: list[CaptionPacket],
    output: Path,
    threshold: float,
    verdicts: dict[tuple[str, int, str], VerdictInfo],
) -> str:
    statuses = [card_status(packet, verdicts, threshold) for packet in packets]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Margin Figure Reader Alignment Audit</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #17212f;
      --muted: #607287;
      --line: #d9e1ea;
      --blue: #0069a8;
      --green: #147d4f;
      --amber: #9c5c00;
      --red: #b42335;
      --shadow: 0 1px 2px rgba(21, 31, 44, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .page {{
      width: min(1500px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    .top {{
      margin-bottom: 18px;
    }}
    .top h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .top p {{
      max-width: 980px;
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .how-to {{
      margin: 18px 0;
      padding: 14px 16px;
      background: #eef5fa;
      border: 1px solid #c9ddeb;
      border-radius: 8px;
      color: #263b4f;
      font-size: 14px;
    }}
    .how-to strong {{ color: var(--ink); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(140px, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .metric {{
      min-height: 78px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric strong {{
      display: block;
      margin-top: 6px;
      font-size: 26px;
    }}
    .controls {{
      position: sticky;
      top: 0;
      z-index: 5;
      display: grid;
      grid-template-columns: auto minmax(260px, 1fr) auto minmax(220px, 300px) auto auto auto auto;
      align-items: center;
      gap: 10px;
      margin: 18px 0 20px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    .controls label {{
      font-size: 13px;
      font-weight: 700;
      color: #324459;
    }}
    .controls input[type="search"],
    .controls select {{
      width: 100%;
      min-height: 36px;
      border: 1px solid #c9d4df;
      border-radius: 6px;
      padding: 7px 10px;
      color: var(--ink);
      background: #fff;
      font: inherit;
    }}
    .controls .check {{
      display: flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      font-weight: 600;
      color: var(--muted);
    }}
    .controls button {{
      min-height: 36px;
      border: 1px solid #c9d4df;
      border-radius: 6px;
      padding: 7px 12px;
      background: #fff;
      color: var(--ink);
      font-weight: 700;
      cursor: pointer;
    }}
    .visible-count {{
      min-width: 86px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-align: right;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 250px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .sidebar {{
      position: sticky;
      top: 78px;
      max-height: calc(100vh - 96px);
      overflow: auto;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .sidebar h2 {{
      margin: 0 0 10px;
      font-size: 15px;
    }}
    .chapter-list {{
      display: grid;
      gap: 6px;
    }}
    .chapter-list button {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      width: 100%;
      min-height: 34px;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 7px 8px;
      background: #f8fafc;
      color: var(--ink);
      cursor: pointer;
      text-align: left;
    }}
    .chapter-list button:hover {{
      border-color: #b8ccd9;
      background: #eef5fa;
    }}
    .chapter-list span {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
    }}
    .chapter-list strong {{
      color: var(--muted);
      font-size: 12px;
    }}
    .cards {{
      display: grid;
      gap: 16px;
    }}
    .figure-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .card-head {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 230px;
      gap: 18px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .eyebrow {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .card-head h2 {{
      margin: 0;
      max-width: 980px;
      font-size: 20px;
      line-height: 1.3;
      letter-spacing: 0;
    }}
    .source {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .status-wrap {{
      display: grid;
      justify-items: end;
      align-content: start;
      gap: 7px;
      text-align: right;
    }}
    .status-wrap p {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border-radius: 999px;
      padding: 5px 10px;
      color: #fff;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.02em;
    }}
    .chip-pass {{ background: var(--green); }}
    .chip-reviewed {{ background: var(--blue); }}
    .chip-bridge {{ background: var(--amber); }}
    .chip-review {{ background: #6f5f00; }}
    .chip-needs-fix {{ background: var(--red); }}
    .card-body {{
      display: grid;
      grid-template-columns: minmax(260px, 360px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
    }}
    .figure-preview {{
      margin: 0;
      display: grid;
      gap: 8px;
      align-content: start;
    }}
    .figure-preview img {{
      width: 100%;
      max-height: 320px;
      object-fit: contain;
      border: 1px solid #e4eaf0;
      border-radius: 8px;
      background: #fff;
    }}
    .figure-preview figcaption {{
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .reader-link h3 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .evidence,
    .evidence-pair > div,
    .context-block {{
      border: 1px solid #e1e8ef;
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .evidence h4,
    .evidence-pair h4,
    .context-block h4,
    details h4 {{
      margin: 0 0 7px;
      color: #47596d;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    blockquote {{
      margin: 0;
      padding-left: 12px;
      border-left: 4px solid #9fc6dd;
      color: #263b4f;
    }}
    .evidence-pair {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 10px 0;
    }}
    .evidence-pair p,
    .context-block p {{
      margin: 0;
      color: #263b4f;
    }}
    details {{
      margin-top: 10px;
      border: 1px solid #e1e8ef;
      border-radius: 8px;
      padding: 10px 12px;
      background: #fbfcfe;
    }}
    summary {{
      cursor: pointer;
      color: var(--blue);
      font-weight: 800;
    }}
    .details-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0;
    }}
    pre {{
      max-height: 320px;
      overflow: auto;
      margin: 0;
      border: 1px solid #d9e1ea;
      border-radius: 8px;
      padding: 12px;
      background: #111827;
      color: #e5edf6;
      font-size: 12px;
      line-height: 1.45;
    }}
    .empty {{
      display: none;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 1000px) {{
      .page {{ width: min(100vw - 24px, 900px); padding-top: 20px; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .controls {{
        position: static;
        grid-template-columns: 1fr;
        align-items: stretch;
      }}
      .visible-count {{ text-align: left; }}
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; max-height: none; }}
      .card-head,
      .card-body,
      .evidence-pair,
      .details-grid {{
        grid-template-columns: 1fr;
      }}
      .status-wrap {{
        justify-items: start;
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="top">
      <h1>Margin Figure Reader Alignment Audit</h1>
      <p>Every placed SVG margin figure is shown with the nearby prose that should make it click for a student reader.</p>
    </header>
    <section class="how-to">
      <strong>How to read this:</strong> the source line is the actual QMD margin placement. Compare the strongest prose anchor, the caption takeaway, and the figure evidence. A good margin figure makes the same point in all three places.
    </section>
    <section class="metrics" aria-label="Audit summary">
      {render_summary(packets, statuses, threshold, len(verdicts))}
    </section>
    {render_controls(packets)}
    <div class="layout">
      {render_sidebar(packets)}
      <main>
        <div id="empty" class="empty">No figures match the current filters.</div>
        <section id="cards" class="cards" aria-label="Margin figure audit cards">
          {render_cards(packets, statuses, output, threshold)}
        </section>
      </main>
    </div>
  </div>
  <script>
    const search = document.getElementById('search');
    const chapterFilter = document.getElementById('chapterFilter');
    const reviewedOnly = document.getElementById('reviewedOnly');
    const bridgeOnly = document.getElementById('bridgeOnly');
    const resetFilters = document.getElementById('resetFilters');
    const visibleCount = document.getElementById('visibleCount');
    const empty = document.getElementById('empty');
    const cards = Array.from(document.querySelectorAll('.figure-card'));
    const chapterButtons = Array.from(document.querySelectorAll('[data-chapter-filter]'));

    function applyFilters(scrollToFirst) {{
      const query = search.value.trim().toLowerCase();
      const chapter = chapterFilter.value;
      let shown = 0;
      let firstVisible = null;

      for (const card of cards) {{
        const matchesQuery = !query || card.dataset.search.includes(query);
        const matchesChapter = chapter === 'all' || card.dataset.chapter === chapter;
        const matchesReviewed = !reviewedOnly.checked || card.dataset.review === 'true';
        const matchesBridge = !bridgeOnly.checked || card.dataset.status === 'bridge';
        const visible = matchesQuery && matchesChapter && matchesReviewed && matchesBridge;
        card.hidden = !visible;
        if (visible) {{
          shown += 1;
          if (!firstVisible) firstVisible = card;
        }}
      }}

      visibleCount.textContent = `${{shown}} shown`;
      empty.style.display = shown ? 'none' : 'block';
      if (scrollToFirst && firstVisible) {{
        firstVisible.scrollIntoView({{ block: 'start' }});
      }}
    }}

    search.addEventListener('input', () => applyFilters(false));
    chapterFilter.addEventListener('change', () => applyFilters(true));
    reviewedOnly.addEventListener('change', () => applyFilters(false));
    bridgeOnly.addEventListener('change', () => applyFilters(false));
    resetFilters.addEventListener('click', () => {{
      search.value = '';
      chapterFilter.value = 'all';
      reviewedOnly.checked = false;
      bridgeOnly.checked = false;
      applyFilters(false);
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }});
    for (const button of chapterButtons) {{
      button.addEventListener('click', () => {{
        chapterFilter.value = button.dataset.chapterFilter;
        applyFilters(true);
      }});
    }}
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="HTML output path.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.30,
        help="Lexical overlap below which a placement is treated as strict-review material.",
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=DEFAULT_VERDICTS,
        help="Reader-alignment verdict markdown to reflect in the HTML status chips.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    verdict_path = args.verdicts if args.verdicts.is_absolute() else ROOT / args.verdicts
    packets = collect(args.review_threshold)
    verdicts = parse_verdicts(verdict_path)
    html_text = render_html(packets, output, args.review_threshold, verdicts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(line.rstrip() for line in html_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    statuses = [card_status(packet, verdicts, args.review_threshold) for packet in packets]
    counts = Counter(status.key for status in statuses)
    print(
        "captions={captions} pass={passed} reviewed={reviewed} bridge={bridge} "
        "remaining={remaining} output={output}".format(
            captions=len(packets),
            passed=counts["pass"] + counts["reviewed"] + counts["bridge"],
            reviewed=counts["reviewed"],
            bridge=counts["bridge"],
            remaining=counts["review"] + counts["needs-fix"],
            output=output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
