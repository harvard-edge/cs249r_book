#!/usr/bin/env python3
"""Check a built deck for content that does not fit.

Two checks, both reported per page with the frame title:

1. LaTeX overflow from the .log: vertical overflow over 10 pt and horizontal
   overflow over 2 pt, mapped to the frame that contains the source line.
2. Rendered bleed: each page is rendered and inspected for ink in the side
   margins and for content touching the footer rule. LaTeX cannot see these:
   a TikZ overlay, a credit line on the footer, or a figure drawn past the
   frame edge.

This does not replace looking at every page. It catches the failures that
recur, so the visual review can concentrate on everything else.

Usage:
    python3 fitcheck.py path/to/deck.tex [--pdf path/to/deck.pdf] [--log path/to/deck.log]
Exit status 1 if any page fails.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

VBOX_LIMIT_PT = 10.0
HBOX_LIMIT_PT = 2.0
DPI = 100
INK = 225            # a pixel darker than this (any channel) counts as ink
SIDE_STRIP_CM = 0.22  # content must stay this far from the left and right edges


def frames_by_line(tex_lines):
    """Map each source line to (frame start line, frame title)."""
    spans, start, title = [], None, ""
    for i, line in enumerate(tex_lines, 1):
        m = re.search(r"\\begin\{frame\}(?:<[^>]*>)?(?:\[[^\]]*\])?(?:\{([^}]*)\})?", line)
        if m:
            start, title = i, (m.group(1) or "").strip()
        if start and re.search(r"\\frametitle\{([^}]*)\}", line) and not title:
            title = re.search(r"\\frametitle\{([^}]*)\}", line).group(1)
        if start and "\\end{frame}" in line:
            spans.append((start, i, title))
            start = None
    return spans


def frame_for(spans, line_no):
    for a, b, t in spans:
        if a <= line_no <= b:
            return f"line {a}: {t or '(untitled)'}"
    return f"line {line_no}: (outside a frame)"


def log_overflow(tex, log):
    spans = frames_by_line(tex.read_text(errors="replace").splitlines())
    problems = []
    text = log.read_text(errors="replace")
    for m in re.finditer(r"Overfull \\vbox \(([\d.]+)pt too high\)(?:[^\n]*?at lines? (\d+))?", text):
        pt = float(m.group(1))
        if pt > VBOX_LIMIT_PT:
            where = frame_for(spans, int(m.group(2))) if m.group(2) else "(no line)"
            problems.append(f"vbox {pt:.1f}pt too high  {where}")
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*?lines? (\d+)", text):
        pt = float(m.group(1))
        if pt > HBOX_LIMIT_PT:
            problems.append(f"hbox {pt:.1f}pt too wide   {frame_for(spans, int(m.group(2)))}")
    return problems


NOTE_WORD_LIMIT = 220   # the notes page silently cuts off longer notes (plain layout, footnotesize)


def long_notes(tex):
    """Speaker notes the notes PDF would truncate without warning."""
    src = tex.read_text(errors="replace")
    out = []
    for m in re.finditer(r"\\(?:mlsys)?note\{(.*?)\}\s*\n", src):
        body = re.sub(r"\\[A-Za-z]+|[{}$]", " ", m.group(1))
        n = len(body.split())
        if n > NOTE_WORD_LIMIT:
            line = src.count("\n", 0, m.start()) + 1
            out.append(f"note at line {line} has {n} words (limit {NOTE_WORD_LIMIT}; the notes PDF truncates it)")
    return out


def is_ink(px):
    return min(px[:3]) < INK


def footer_rule_row(img):
    """The footer starts at a thin full-width colored rule near the bottom."""
    w, h = img.size
    for y in range(h - 1, int(h * 0.8), -1):
        row = [img.getpixel((x, y)) for x in range(int(w * 0.3), int(w * 0.7), 7)]
        if all(is_ink(p) and max(p[:3]) - min(p[:3]) > 40 for p in row):
            return y
    return None


def page_bleed(img):
    w, h = img.size
    strip = int(SIDE_STRIP_CM / 2.54 * DPI)
    rule = footer_rule_row(img)
    header = int(h * 0.06)   # the navigation bar
    bottom = rule if rule else int(h * 0.92)
    issues = []
    for side, xs in (("left", range(0, strip)), ("right", range(w - strip, w))):
        hits = sum(1 for x in xs for y in range(header, bottom, 2) if is_ink(img.getpixel((x, y))))
        if hits > 3:
            issues.append(f"ink in the {side} margin ({hits} px)")
    if rule:
        band = [img.getpixel((x, y)) for y in range(rule - 3, rule) for x in range(strip, w - strip)]
        hits = sum(1 for p in band if is_ink(p))
        if hits > 6:
            issues.append(f"content touches the footer rule ({hits} px)")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", type=Path)
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--log", type=Path)
    a = ap.parse_args()
    pdf = a.pdf or a.tex.with_suffix(".pdf")
    log = a.log or a.tex.with_suffix(".log")
    failed = False
    if log.exists():
        for p in log_overflow(a.tex, log):
            print("✗", p)
            failed = True
    else:
        print("⚠ no log found; skipped the overflow check")
    for p in long_notes(a.tex):
        print("✗", p)
        failed = True
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-png", "-r", str(DPI), str(pdf), f"{d}/p"], check=True)
        pages = sorted(Path(d).glob("p-*.png"), key=lambda p: int(p.stem.split("-")[1]))
        for i, p in enumerate(pages, 1):
            if i == 1:
                continue   # title page has its own layout
            img = Image.open(p).convert("RGB")
            if is_ink(img.getpixel((img.size[0] // 2, img.size[1] // 2 + 40))) and \
                    is_ink(img.getpixel((5, img.size[1] // 2))):
                continue   # dark focus slide: its own full-bleed layout
            for issue in page_bleed(img):
                print(f"✗ page {i}: {issue}")
                failed = True
    print("✓ fit check passed" if not failed else "fit check found problems")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
