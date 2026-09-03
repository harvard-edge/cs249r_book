#!/usr/bin/env python3
"""Audit margin-figure captions against nearby prose.

This is an editorial support tool. It inventories QMD ``.column-margin`` blocks,
captures the nearest prose before and after each figure, and emits a markdown
packet for judgment. The score is only a triage signal; final pass/fix decisions
must be made by reading the caption with the local paragraph.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CONTENTS = ROOT / "book/quarto/contents"

COLUMN_START_RE = re.compile(r"^:{3,}\s*\{[^}]*\.column-margin\b")
DIV_CLOSE_RE = re.compile(r"^:{3,}\s*$")
IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\((?P<src>images/(?P<kind>svg|png)/[^)]+)\)"
    r"(?:\{(?P<attrs>[^}]*)\})?"
)
ATTR_RE = re.compile(r"(?P<key>[A-Za-z0-9_-]+)\s*=\s*\"(?P<value>[^\"]*)\"")

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "the",
    "to",
    "with",
    "without",
}


@dataclass(frozen=True)
class CaptionPacket:
    chapter: str
    qmd_path: str
    line: int
    asset: str
    caption: str
    fig_alt: str
    before: str
    after: str
    overlap: float
    status: str


def attrs_from(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {match.group("key"): match.group("value") for match in ATTR_RE.finditer(raw)}


def clean_caption(raw: str) -> str:
    caption = raw.strip()
    if caption.startswith("*") and caption.endswith("*"):
        caption = caption.strip("*").strip()
    return re.sub(r"\s+", " ", caption)


def strip_markup(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)(?:\{[^}]*\})?", " ", text)
    text = re.sub(r"\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"[@#][A-Za-z0-9_.:-]+", " ", text)
    text = re.sub(r"\\[A-Za-z]+(?:\{[^}]*\})?", " ", text)
    text = text.replace("*", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", strip_markup(text).lower())
        if word not in STOPWORDS
    }


def overlap_score(caption: str, before: str, after: str) -> float:
    caption_words = words(caption)
    if not caption_words:
        return 0.0
    context_words = words(before) | words(after)
    return len(caption_words & context_words) / len(caption_words)


def paragraph_before(lines: list[str], start_index: int) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in lines[:start_index]:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped.startswith((":::", "|", "#", "```")):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if stripped.startswith("[^") or stripped.startswith("![]("):
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    return strip_markup(paragraphs[-1]) if paragraphs else ""


def paragraph_after(lines: list[str], end_index: int) -> str:
    current: list[str] = []
    in_fence = False
    for line in lines[end_index:]:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or stripped.startswith((":::", "|", "#", "```")):
            if current:
                break
            continue
        if not stripped:
            if current:
                break
            continue
        if stripped.startswith("[^") or stripped.startswith("![]("):
            continue
        current.append(stripped)
    return strip_markup(" ".join(current))


def caption_after(block: list[tuple[int, str]], image_index: int) -> str:
    for _, line in block[image_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if IMAGE_RE.search(stripped) or stripped.startswith(":::"):
            return ""
        return clean_caption(stripped)
    return ""


def iter_margin_blocks(lines: list[str]) -> list[tuple[int, int, list[tuple[int, str]]]]:
    blocks: list[tuple[int, int, list[tuple[int, str]]]] = []
    in_margin = False
    start_index = 0
    current: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not in_margin and COLUMN_START_RE.match(stripped):
            in_margin = True
            start_index = index
            current = []
            continue
        if in_margin and DIV_CLOSE_RE.match(stripped):
            blocks.append((start_index, index + 1, current))
            in_margin = False
            current = []
            continue
        if in_margin:
            current.append((index + 1, line))
    return blocks


def status_for(caption: str, score: float, review_threshold: float) -> str:
    if not caption:
        return "fix-missing-caption"
    if score < review_threshold:
        return "review-low-overlap"
    if len(caption.split()) <= 3:
        return "review-title-like"
    return "pass-triage"


def collect(review_threshold: float = 0.18) -> list[CaptionPacket]:
    packets: list[CaptionPacket] = []
    for qmd in sorted(CONTENTS.glob("vol*/*/*.qmd")):
        chapter = qmd.parent.relative_to(CONTENTS).as_posix()
        lines = qmd.read_text(encoding="utf-8").splitlines()
        for block_start, block_end, block in iter_margin_blocks(lines):
            before = paragraph_before(lines, block_start)
            after = paragraph_after(lines, block_end)
            for idx, (line_no, line) in enumerate(block):
                match = IMAGE_RE.search(line.strip())
                if not match:
                    continue
                attrs = attrs_from(match.group("attrs"))
                caption = caption_after(block, idx)
                score = overlap_score(caption, before, after)
                packets.append(
                    CaptionPacket(
                        chapter=chapter,
                        qmd_path=str(qmd.relative_to(ROOT)),
                        line=line_no,
                        asset=Path(match.group("src")).name,
                        caption=caption,
                        fig_alt=attrs.get("fig-alt", ""),
                        before=before,
                        after=after,
                        overlap=score,
                        status=status_for(caption, score, review_threshold),
                    )
                )
    return packets


def write_csv(packets: list[CaptionPacket], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CaptionPacket.__dataclass_fields__))
        writer.writeheader()
        for packet in packets:
            writer.writerow(packet.__dict__)


def md_cell(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "..."
    return text.replace("|", "\\|")


def write_markdown(packets: list[CaptionPacket], output: Path) -> None:
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# Margin Caption Alignment Packet\n\n")
        handle.write("Generated from current QMD `.column-margin` blocks.\n\n")
        handle.write("| Status | Chapter:line | Asset | Caption | Before prose | After prose |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for packet in packets:
            handle.write(
                f"| {packet.status} ({packet.overlap:.2f}) "
                f"| {packet.qmd_path}:{packet.line} "
                f"| {md_cell(packet.asset, 80)} "
                f"| {md_cell(packet.caption)} "
                f"| {md_cell(packet.before)} "
                f"| {md_cell(packet.after)} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, help="Write a markdown audit packet.")
    parser.add_argument("--csv", type=Path, help="Write a CSV audit packet.")
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Emit only non-pass triage rows in markdown output.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.18,
        help=(
            "Minimum caption/context overlap score before a row is treated as "
            "pass-triage. The default is tuned for missing obvious links; use "
            "0.30 for a stricter narrative-click editorial pass."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packets = collect(args.review_threshold)
    output_packets = [p for p in packets if p.status != "pass-triage"] if args.review_only else packets
    if args.markdown:
        write_markdown(output_packets, args.markdown)
    if args.csv:
        write_csv(packets, args.csv)
    print(f"captions={len(packets)} review={sum(p.status != 'pass-triage' for p in packets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
