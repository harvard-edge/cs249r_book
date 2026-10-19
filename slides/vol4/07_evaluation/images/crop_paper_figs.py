#!/usr/bin/env python3
"""Crop published figures and report excerpts for Lecture 7 (Closed-Loop Evaluation).

Each crop is an exact region of a public document, rendered with pdftoppm and
cut at the fractions below (x0, y0, x1, y1 of the page). Sources and URLs are
in SOURCES.yml. Download the PDFs into one folder first, then run:

    python3 crop_paper_figs.py <folder with the PDFs>

Expected file names in that folder (URL in SOURCES.yml):
    nhtsa_pe16007.pdf  NHTSA ODI closing resume, PE 16-007 (January 2017)
    simpler.pdf        Li et al., SIMPLER, arXiv 2405.05941
    tri_lbm.pdf        TRI LBM Team, arXiv 2507.05331
    mandlekar.pdf      Mandlekar et al., CoRL 2021, arXiv 2108.03298
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

OUT = Path(__file__).parent

# (pdf, page, dpi, box as page fractions, output name)
CROPS = [
    ("nhtsa_pe16007.pdf", 12, 200, (0.215, 0.06, 0.77, 0.326), "nhtsa_fig11.png"),
    ("simpler.pdf", 1, 200, (0.08, 0.26, 0.92, 0.512), "simpler_fig1.png"),
    ("tri_lbm.pdf", 11, 200, (0.08, 0.075, 0.92, 0.225), "tri_initial_overlay.png"),
    ("tri_lbm.pdf", 14, 200, (0.08, 0.19, 0.92, 0.40), "tri_platform.png"),
    ("tri_lbm.pdf", 14, 200, (0.522, 0.19, 0.89, 0.40), "tri_settings.png"),
    ("mandlekar.pdf", 30, 250, (0.17, 0.55, 0.485, 0.635), "mandlekar_g1.png"),
]


def render(pdf, page, dpi, tmp):
    stem = Path(tmp) / f"{pdf.stem}_{page}"
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", str(page),
                    "-l", str(page), "-singlefile", str(pdf), str(stem)], check=True)
    return Path(str(stem) + ".png")


def main(src):
    src = Path(src)
    with tempfile.TemporaryDirectory() as tmp:
        for name, page, dpi, box, out in CROPS:
            im = Image.open(render(src / name, page, dpi, tmp)).convert("RGB")
            w, h = im.size
            crop = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
            crop.save(OUT / out)
            print(f"  {out}: {crop.size[0]} x {crop.size[1]} px")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
