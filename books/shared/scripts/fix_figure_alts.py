#!/usr/bin/env python3
"""Repair Volume IV rendered image descriptions and the logo-only home link.

Some Quarto figure floats place ``fig-alt`` on a surrounding ``div`` while
leaving its only ``img`` without alternate text. This pass changes only the
affected opening tags, preserving the rest of the rendered HTML verbatim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re


_ATTRIBUTE = re.compile(r'\s+([^\s=/>]+)(?:\s*=\s*("[^"]*"|\'[^\']*\'|[^\s>]+))?')


def _attribute(tag: str, name: str) -> tuple[int, int, str] | None:
    """Return the source span and raw value of an opening-tag attribute."""
    tag_name = re.match(r"<[^\s/>]+", tag)
    if tag_name is None:
        return None
    offset = tag_name.end()
    for match in _ATTRIBUTE.finditer(tag, offset):
        if match.group(1).lower() != name:
            continue
        value = match.group(2)
        if value is None:
            return None
        if value[:1] in ('"', "'"):
            value = value[1:-1]
        return match.start(), match.end(), value
    return None


@dataclass
class _Figure:
    depth: int
    start: int
    tag: str
    images: list[tuple[int, str]] = field(default_factory=list)
    nested_figure: bool = False


class _FigureAltParser(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_starts = [0]
        for line in source.splitlines(keepends=True):
            self.line_starts.append(self.line_starts[-1] + len(line))
        self.div_depth = 0
        self.figures: list[_Figure] = []
        self.edits: list[tuple[int, int, str]] = []
        self.transferred = 0
        self.wrappers_cleaned = 0
        self.fallback_labeled = 0

    def _position(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        start = self._position()
        raw = self.get_starttag_text()
        if tag == "div":
            self.div_depth += 1
            classes = (dict(attrs).get("class") or "").split()
            if "quarto-figure" in classes and _attribute(raw, "alt"):
                for figure in self.figures:
                    figure.nested_figure = True
                self.figures.append(_Figure(self.div_depth, start, raw))
        elif tag == "img":
            for figure in self.figures:
                figure.images.append((start, raw))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag == "div":
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag != "div":
            return
        while self.figures and self.figures[-1].depth == self.div_depth:
            self._close_figure(self.figures.pop())
        self.div_depth = max(0, self.div_depth - 1)

    def _close_figure(self, figure: _Figure) -> None:
        wrapper_alt = _attribute(figure.tag, "alt")
        if wrapper_alt is None:
            return
        alt_start, alt_end, alt_value = wrapper_alt
        raw_alt = figure.tag[alt_start:alt_end]
        if len(figure.images) == 1 and not figure.nested_figure:
            image_start, image_tag = figure.images[0]
            image_alt = _attribute(image_tag, "alt")
            if image_alt is None:
                insert_at = image_start + len(image_tag.rstrip()) - 1
                if image_tag.rstrip().endswith("/>"):
                    insert_at -= 1
                self.edits.append((insert_at, insert_at, raw_alt))
                self.transferred += 1
            elif not unescape(image_alt[2]).strip():
                self.edits.append((image_start + image_alt[0],
                                   image_start + image_alt[1], raw_alt))
                self.transferred += 1
            self.edits.append((figure.start + alt_start,
                               figure.start + alt_end, ""))
            self.wrappers_cleaned += 1
        else:
            # A single image is the only safe target for a transfer. Keep a
            # description on unusual multi-image or image-free figures.
            wrapper_label = _attribute(figure.tag, "aria-label")
            if wrapper_label is None:
                replacement = raw_alt.replace("alt", "aria-label", 1)
                self.fallback_labeled += 1
            elif unescape(wrapper_label[2]).strip():
                replacement = ""
            else:
                self.edits.append((figure.start + wrapper_label[0],
                                   figure.start + wrapper_label[1],
                                   raw_alt.replace("alt", "aria-label", 1)))
                replacement = ""
                self.fallback_labeled += 1
            self.edits.append((figure.start + alt_start,
                               figure.start + alt_end, replacement))
            self.wrappers_cleaned += 1


def repair_figure_alts(source: str) -> tuple[str, dict[str, int]]:
    """Repair Quarto figure floats and report the number of changes."""
    parser = _FigureAltParser(source)
    parser.feed(source)
    parser.close()
    output = source
    for start, end, replacement in sorted(parser.edits, reverse=True):
        output = output[:start] + replacement + output[end:]
    return output, {
        "transferred": parser.transferred,
        "wrappers_cleaned": parser.wrappers_cleaned,
        "fallback_labeled": parser.fallback_labeled,
    }


class _NavbarLogoParser(HTMLParser):
    """Find an otherwise unnamed Quarto logo-only home link."""

    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.line_starts = [0]
        for line in source.splitlines(keepends=True):
            self.line_starts.append(self.line_starts[-1] + len(line))
        self.candidate: tuple[int, str] | None = None
        self.images = 0
        self.other_content = False
        self.edits: list[tuple[int, int, str]] = []

    def _position(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            if self.candidate is not None:
                self.other_content = True
                return
            anchor_attrs = dict(attrs)
            classes = set((anchor_attrs.get("class") or "").split())
            if ({"navbar-brand", "navbar-brand-logo"} <= classes
                    and (anchor_attrs.get("href") or "").endswith("index.html")):
                self.candidate = (self._position(), self.get_starttag_text())
                self.images = 0
                self.other_content = False
        elif self.candidate is not None:
            if tag == "img":
                image_attrs = dict(attrs)
                classes = set((image_attrs.get("class") or "").split())
                if "navbar-logo" in classes and image_attrs.get("alt") == "":
                    self.images += 1
                else:
                    self.other_content = True
            else:
                self.other_content = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if self.candidate is not None and data.strip():
            self.other_content = True

    def handle_entityref(self, name: str) -> None:
        if self.candidate is not None:
            self.other_content = True

    def handle_charref(self, name: str) -> None:
        if self.candidate is not None:
            self.other_content = True

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self.candidate is None:
            return
        start, raw = self.candidate
        if self.images and not self.other_content:
            label = _attribute(raw, "aria-label")
            if label is None:
                insert_at = start + len(raw) - 1
                self.edits.append((insert_at, insert_at,
                                   ' aria-label="Machine Learning Systems home"'))
            elif not unescape(label[2]).strip():
                self.edits.append((start + label[0], start + label[1],
                                   ' aria-label="Machine Learning Systems home"'))
        self.candidate = None


def repair_navbar_logo_links(source: str) -> tuple[str, int]:
    """Name only logo home links whose image alternatives are decorative."""
    parser = _NavbarLogoParser(source)
    parser.feed(source)
    parser.close()
    output = source
    for start, end, replacement in sorted(parser.edits, reverse=True):
        output = output[:start] + replacement + output[end:]
    return output, len(parser.edits)


def repair_file(path: Path) -> dict[str, int]:
    source = path.read_text(encoding="utf-8")
    output, counts = repair_figure_alts(source)
    output, counts["navbar_links_labeled"] = repair_navbar_logo_links(output)
    if output != source:
        path.write_text(output, encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "_build" / "html-vol4")
    args = parser.parse_args()
    totals = {"transferred": 0, "wrappers_cleaned": 0,
              "fallback_labeled": 0, "navbar_links_labeled": 0}
    files_changed = 0
    for path in sorted(args.build_dir.rglob("*.html")):
        counts = repair_file(path)
        if counts["wrappers_cleaned"] or counts["navbar_links_labeled"]:
            files_changed += 1
        for key in totals:
            totals[key] += counts[key]
    print(f"[figure-alt] {totals['transferred']} image alt(s) transferred; "
          f"{totals['wrappers_cleaned']} wrapper alt(s) removed "
          f"in {files_changed} file(s); "
          f"{totals['fallback_labeled']} unusual float(s) retained an aria-label; "
          f"{totals['navbar_links_labeled']} logo home link(s) named.")


if __name__ == "__main__":
    main()
