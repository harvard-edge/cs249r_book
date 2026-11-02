#!/usr/bin/env python3
"""Make the soft-focus Volume III and IV covers for the landing-page teaser.

The live site teases the unreleased volumes without revealing them. The title
band is painted out of each cover before blurring, so no title, not even its
shape, ships in the image a browser downloads. Rerun whenever a source cover
changes:

    python3 site/scripts/make_teaser_covers.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

COVERS = Path(__file__).resolve().parent.parent / "covers"
SIZE = (160, 205)      # 2x the 80x102 display size
TITLE_TOP = 0.64       # the title band starts at 64% of the cover height
BLUR_RADIUS = 9        # art stays recognizable, detail does not


def main() -> None:
    for vol in (3, 4):
        cover = Image.open(COVERS / f"cover-hardcover-book-vol{vol}.webp").convert("RGB")
        w, h = cover.size
        paper = cover.getpixel((w // 2, int(h * 0.60)))
        ImageDraw.Draw(cover).rectangle([0, int(h * TITLE_TOP), w, h], fill=paper)
        teaser = cover.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)).resize(SIZE, Image.LANCZOS)
        out = COVERS / f"teaser-vol{vol}.webp"
        teaser.save(out, quality=82)
        print(f"wrote {out.relative_to(COVERS.parent)}")


if __name__ == "__main__":
    main()
