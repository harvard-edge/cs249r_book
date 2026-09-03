#!/usr/bin/env python3
"""Render MLSysBook margin SVGs into a contact sheet for visual QA.

The generator outlines text into SVG paths, so this script is primarily for
human checks: margin-scale legibility, line weight, collisions, and whether the
rendered result looks publication-clean. It uses ``rsvg-convert`` so the raster
preview exercises the same SVG renderer family used by Quarto/PDF workflows.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Pillow is required: python3 -m pip install Pillow") from exc


ROOT = Path(__file__).resolve().parents[4]
CONTENTS = ROOT / "book/quarto/contents"
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((images/svg/[^)]+\.svg)\)")


@dataclass(frozen=True)
class Entry:
    chapter: str
    svg: Path
    label: str


def _chapter_matches(chapter: str, filters: list[str]) -> bool:
    if not filters:
        return True
    short = chapter.split("/", 1)[-1]
    return any(chapter == item or short == item for item in filters)


def referenced_margin_svgs(volume: str, chapters: list[str]) -> list[Entry]:
    entries: list[Entry] = []
    seen: set[Path] = set()
    for qmd in sorted(CONTENTS.glob("vol*/*/*.qmd")):
        chapter = qmd.parent.relative_to(CONTENTS).as_posix()
        if volume != "all" and not chapter.startswith(f"{volume}/"):
            continue
        if not _chapter_matches(chapter, chapters):
            continue

        in_margin = False
        for line in qmd.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(":::") and ".column-margin" in stripped:
                in_margin = True
                continue
            if in_margin and stripped == ":::":
                in_margin = False
                continue
            if not in_margin:
                continue
            match = IMAGE_RE.search(stripped)
            if not match:
                continue
            svg = (qmd.parent / match.group(1)).resolve()
            if svg in seen:
                continue
            seen.add(svg)
            entries.append(Entry(chapter=chapter, svg=svg, label=f"{chapter}/{svg.name}"))
    return entries


def explicit_svgs(paths: list[str]) -> list[Entry]:
    entries: list[Entry] = []
    for raw in paths:
        svg = Path(raw)
        if not svg.is_absolute():
            svg = ROOT / svg
        svg = svg.resolve()
        try:
            chapter = svg.relative_to(CONTENTS).parts[:2]
            chapter_label = "/".join(chapter)
        except ValueError:
            chapter_label = "manual"
        entries.append(Entry(chapter=chapter_label, svg=svg, label=svg.name))
    return entries


def render_svg(rsvg_convert: str, entry: Entry, out_dir: Path, width: int, index: int) -> Path:
    if not entry.svg.exists():
        raise FileNotFoundError(entry.svg)
    out = out_dir / f"{index:04d}.png"
    subprocess.run(
        [rsvg_convert, "-w", str(width), "-o", str(out), str(entry.svg)],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def _fit_label(label: str, max_chars: int = 34) -> str:
    return label if len(label) <= max_chars else label[: max_chars - 1] + "..."


def compose_sheet(entries: list[Entry], pngs: list[Path], output: Path, columns: int, width: int) -> None:
    images = [Image.open(path).convert("RGB") for path in pngs]
    label_h = 46
    pad = 12
    tile_w = width + pad * 2
    max_img_h = max(image.height for image in images)
    tile_h = max_img_h + label_h + pad
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, (entry, image) in enumerate(zip(entries, images)):
        row, col = divmod(idx, columns)
        x = col * tile_w + pad + (width - image.width) // 2
        y = row * tile_h + pad
        sheet.paste(image, (x, y))
        label_y = row * tile_h + max_img_h + pad + 4
        draw.text((col * tile_w + pad, label_y), _fit_label(entry.chapter), fill=(70, 70, 70))
        draw.text((col * tile_w + pad, label_y + 14), _fit_label(entry.svg.name), fill=(20, 20, 20))

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/mlsysbook-margin-contact-sheet.png")
    parser.add_argument("--volume", choices=("all", "vol1", "vol2"), default="all")
    parser.add_argument("--chapter", action="append", default=[], help="Filter to a chapter, e.g. vol2/inference or inference.")
    parser.add_argument("--svg", action="append", default=[], help="Render an explicit SVG path. Repeat for multiple SVGs.")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--width", type=int, default=260, help="Rendered width of each margin SVG in pixels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rsvg_convert = shutil.which("rsvg-convert")
    if not rsvg_convert:
        print("rsvg-convert is required for margin contact-sheet rendering.", file=sys.stderr)
        return 2

    entries = explicit_svgs(args.svg) if args.svg else referenced_margin_svgs(args.volume, args.chapter)
    if not entries:
        print("No margin SVGs matched.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="mlsysbook-margin-render-") as tmp:
        tmp_dir = Path(tmp)
        pngs = [render_svg(rsvg_convert, entry, tmp_dir, args.width, idx) for idx, entry in enumerate(entries)]
        compose_sheet(entries, pngs, Path(args.output), args.columns, args.width)

    print(f"rendered {len(entries)} margin SVGs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
