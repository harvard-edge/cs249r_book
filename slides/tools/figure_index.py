#!/usr/bin/env python3
"""Index a volume's book figures from its rendered HTML, for reuse in slides.

Book figures are the first choice for slides (students recognize them from
the reading). The rendered HTML holds every figure as an image, including
TikZ and Python-generated ones. This script finds each numbered figure,
copies its image into a figure pack, renders a PNG preview, and writes an
index with the figure id, title, and source file.

Usage:
    figure_index.py BUILD_HTML_DIR OUT_DIR
    e.g. figure_index.py books/_build/html-vol1/chapters /tmp/bookfigs-vol1

Writes OUT_DIR/<chapter>/<fig-id>.<ext>, OUT_DIR/<chapter>/<fig-id>.png
(preview), OUT_DIR/index.md, and OUT_DIR/<chapter>/sheet.png.
"""

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def chapters(build):
    for d in sorted(build.iterdir()):
        if d.is_dir() and re.match(r"\d\d_", d.name):
            f = next(d.rglob(f"{d.name}.html"), None)
            if f:
                yield d.name, f


def figures(page):
    text = page.read_text(errors="replace")
    for m in re.finditer(r'<div id="(fig-[^"]+)"(.*?)</figure>', text, re.S):
        fid, body = m.group(1), m.group(2)
        img = re.search(r'<img[^>]+src="([^"]+)"', body) or re.search(r'<embed[^>]+src="([^"]+)"', body)
        cap = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", body, re.S)
        title = ""
        if cap:
            s = html.unescape(re.sub(r"<[^>]+>", "", cap.group(1)))
            s = re.sub(r"^\s*Figure\s+[\d.]+:\s*", "", " ".join(s.split()))
            title = s.split(":")[0][:90]
        if img:
            yield fid, title, (page.parent / html.unescape(img.group(1))).resolve()
    # Margin figures: small numbered-less illustrations (ladders, bars) that
    # often make good one-number slides. Their alt text is the description.
    seen = set()
    for m in re.finditer(r'<img[^>]+src="([^"]*margin[^"]*\.svg)"[^>]*?alt="([^"]*)"', text):
        src, alt = html.unescape(m.group(1)), html.unescape(m.group(2))
        if "margin_stack" in src or src in seen:
            continue
        seen.add(src)
        yield "margin-" + Path(src).stem.split("_")[-1], alt[:90], (page.parent / src).resolve()


def preview(src, dst):
    if src.suffix == ".svg":
        subprocess.run(["rsvg-convert", "-b", "white", "-w", "1100", str(src), "-o", str(dst)],
                       check=False, capture_output=True)
    else:
        im = Image.open(src).convert("RGB")
        im.thumbnail((1100, 1100))
        im.save(dst)


def sheet(items, dst, cols=4, w=360, h=240):
    rows = (len(items) + cols - 1) // cols
    s = Image.new("RGB", (cols * w, rows * (h + 18)), "#dddddd")
    d = ImageDraw.Draw(s)
    for i, (fid, png) in enumerate(items):
        x, y = (i % cols) * w, (i // cols) * (h + 18)
        d.text((x + 4, y + 3), fid[:48], fill="black")
        try:
            im = Image.open(png).convert("RGB")
            im.thumbnail((w - 8, h - 6))
            s.paste(im, (x + 4, y + 16))
        except Exception:
            d.text((x + 4, y + 40), "(no preview)", fill="red")
    s.save(dst)


def main():
    build, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Book figure index\n",
             "| Chapter | Figure | Title | File |", "|---|---|---|---|"]
    total = 0
    for ch, page in chapters(build):
        items = []
        (out / ch).mkdir(exist_ok=True)
        for fid, title, src in figures(page):
            if not src.exists():
                continue
            dst = out / ch / f"{fid}{src.suffix}"
            shutil.copy(src, dst)
            png = out / ch / f"{fid}.png" if src.suffix != ".png" else dst
            if src.suffix != ".png":
                preview(src, png)
            items.append((fid, png))
            lines.append(f"| {ch} | `{fid}` | {title} | `{ch}/{dst.name}` |")
            total += 1
        if items:
            sheet(items, out / ch / "sheet.png")
    (out / "index.md").write_text("\n".join(lines) + "\n")
    print(f"{total} figures indexed in {out}")


if __name__ == "__main__":
    main()
