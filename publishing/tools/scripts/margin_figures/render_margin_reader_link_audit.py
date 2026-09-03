#!/usr/bin/env python3
"""Render an inspectable reader-link audit for margin figures.

This script creates a markdown packet for editor/LLM review. Each entry shows
where the margin figure appears in QMD source, embeds the referenced SVG, shows
the caption and figure alt text, and records the nearest prose before and after
the ``.column-margin`` block. The packet is meant to answer: "What point in the
text is this margin figure supporting, and how can I verify that from markdown?"
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from audit_margin_caption_alignment import ROOT, CaptionPacket, collect, strip_markup, words


DEFAULT_OUTPUT = ROOT / "book/tools/audit/margin_figure_reader_link_audit.md"


def md_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|")


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, limit: int) -> str:
    text = collapse(text)
    if len(text) <= limit:
        return text
    cut = text[: max(limit - 1, 0)].rstrip()
    split = cut.rfind(" ")
    if split > max(24, limit // 2):
        cut = cut[:split]
    return cut.rstrip(" ,;:") + "..."


def quote_block(text: str, limit: int = 900) -> str:
    text = truncate(text, limit)
    if not text:
        return "> _No adjacent prose captured._\n"
    return "\n".join(f"> {line}" for line in text.splitlines()) + "\n"


def sentences(text: str) -> list[str]:
    cleaned = strip_markup(text)
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [collapse(part) for part in parts if collapse(part)]


def strongest_anchor(packet: CaptionPacket) -> str:
    caption_words = words(packet.caption)
    candidates = sentences(packet.before) + sentences(packet.after)
    if not candidates:
        return ""
    if not caption_words:
        return candidates[0]
    scored: list[tuple[int, int, str]] = []
    for sentence in candidates:
        sentence_words = words(sentence)
        overlap = len(caption_words & sentence_words)
        scored.append((overlap, len(sentence_words), sentence))
    scored.sort(reverse=True)
    return scored[0][2]


def asset_path(packet: CaptionPacket) -> Path:
    qmd = ROOT / packet.qmd_path
    for kind in ("svg", "png"):
        candidate = qmd.parent / f"images/{kind}" / packet.asset
        if candidate.exists():
            return candidate
    return qmd.parent / "images/svg" / packet.asset


def rel_link(target: Path, output: Path) -> str:
    return os.path.relpath(target, output.parent)


def source_link(packet: CaptionPacket, output: Path) -> str:
    qmd = ROOT / packet.qmd_path
    rel = rel_link(qmd, output)
    return f"{rel}:{packet.line}"


def source_excerpt(packet: CaptionPacket, before: int = 2, after: int = 5) -> str:
    qmd = ROOT / packet.qmd_path
    lines = qmd.read_text(encoding="utf-8").splitlines()
    start = max(1, packet.line - before)
    end = min(len(lines), packet.line + after)
    width = len(str(end))
    return "\n".join(
        f"{line_no:>{width}}  {lines[line_no - 1]}".rstrip()
        for line_no in range(start, end + 1)
    )


def verdict(packet: CaptionPacket, strict_threshold: float) -> str:
    if packet.status.startswith("fix"):
        return "Needs fix"
    if packet.status.startswith("review") or packet.overlap < strict_threshold:
        return "Manual review candidate"
    return "Pass"


def write_entry(handle, index: int, packet: CaptionPacket, output: Path, strict_threshold: float) -> None:
    asset = asset_path(packet)
    anchor = strongest_anchor(packet)
    status = verdict(packet, strict_threshold)
    asset_rel = rel_link(asset, output)
    source = source_link(packet, output)

    handle.write(
        f"### {index:03d}. {packet.chapter} @ line {packet.line}: "
        f"{md_escape(packet.caption)}\n\n"
    )
    handle.write(f"- **Source QMD:** `{source}`\n")
    handle.write(f"- **Asset:** `{asset_rel}`\n")
    handle.write(f"- **Audit status:** `{status}`; lexical overlap `{packet.overlap:.2f}`\n")
    handle.write(f"- **Caption:** {md_escape(packet.caption)}\n")
    handle.write(f"- **Figure evidence (`fig-alt`):** {md_escape(packet.fig_alt or '_Missing fig-alt._')}\n\n")
    handle.write(f"![{md_escape(packet.caption)}]({asset_rel})\n\n")
    handle.write("**Source Markdown Excerpt**\n\n")
    handle.write("```markdown\n")
    handle.write(source_excerpt(packet))
    handle.write("\n```\n\n")
    handle.write("**Strongest Prose Anchor**\n\n")
    handle.write(quote_block(anchor, 650))
    handle.write("\n**Placement Context**\n\n")
    handle.write("_Paragraph before the margin block:_\n\n")
    handle.write(quote_block(packet.before, 650))
    handle.write("\n_Paragraph after the margin block:_\n\n")
    handle.write(quote_block(packet.after, 650))
    handle.write("\n**Reader-Link Check**\n\n")
    handle.write(
        "- Source markdown: the excerpt above shows the `.column-margin` block "
        "and the exact caption beside the prose.\n"
    )
    handle.write(
        "- The prose anchor is the text an editor should compare against the "
        "caption.\n"
    )
    handle.write(
        "- The `fig-alt` describes what the visual marks encode; the caption "
        "should state the reader takeaway from those marks.\n\n"
    )


def write_markdown(packets: list[CaptionPacket], output: Path, strict_threshold: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    review_count = sum(
        packet.status != "pass-triage" or packet.overlap < strict_threshold
        for packet in packets
    )
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# Margin Figure Reader-Link Audit\n\n")
        handle.write(
            "This packet is an inspectable editor/LLM audit of margin-figure "
            "correspondence. It is not just a score table: each entry embeds the "
            "figure, shows the caption, shows the objective `fig-alt`, and shows "
            "the nearest prose before and after the `.column-margin` block.\n\n"
        )
        handle.write("## How To Read The QMD\n\n")
        handle.write(
            "A margin figure corresponds to the point in the text because it is "
            "placed directly in document flow inside a `.column-margin` block:\n\n"
        )
        handle.write("```markdown\n")
        handle.write("::: {.column-margin}\n")
        handle.write("![](images/svg/example.svg){width=\"100%\" fig-alt=\"...\"}\n\n")
        handle.write("*Caption states the reader takeaway.*\n")
        handle.write(":::\n")
        handle.write("```\n\n")
        handle.write(
            "When inspecting source markdown, use the `Source QMD` line below, "
            "then read the paragraph immediately before and after that margin "
            "block. Those paragraphs are the prose anchor. If the figure sits "
            "inside a callout or notebook, the local callout content is the "
            "anchor. The caption is good only when the prose anchor, visual "
            "marks, and caption all make the same point.\n\n"
        )
        handle.write("## Summary\n\n")
        handle.write(f"- Margin figures audited: `{len(packets)}`\n")
        handle.write(f"- Strict review threshold: `{strict_threshold:.2f}`\n")
        handle.write(f"- Entries marked for manual review by the packet: `{review_count}`\n")
        handle.write("- Manual review standard: prose claim + visual evidence + caption takeaway click together.\n\n")
        handle.write("## Entries\n\n")
        for index, packet in enumerate(packets, 1):
            write_entry(handle, index, packet, output, strict_threshold)
    output.write_text(output.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown output path.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.30,
        help="Lexical overlap below which an entry is marked for manual review.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    packets = collect(args.review_threshold)
    write_markdown(packets, output, args.review_threshold)
    review_count = sum(packet.status != "pass-triage" for packet in packets)
    print(f"captions={len(packets)} review={review_count} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
