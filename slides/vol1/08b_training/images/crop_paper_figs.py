#!/usr/bin/env python3
"""Crop the two panels of FlashAttention's Figure 1 shown in Lecture 8b.

Source: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness," NeurIPS 2022, arXiv 2205.14135 (PDF from
https://arxiv.org/pdf/2205.14135), page 2, Figure 1. The page is rendered
at 400 dpi with pdftoppm; the boxes below are in points at 100 dpi.

Usage (download the PDF into an empty folder first):
  python3 crop_paper_figs.py /path/to/2205.14135.pdf
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

pdf = Path(sys.argv[1])
out = Path(__file__).parent
BOXES = {"fa_fig1_hierarchy.png": (140, 148, 333, 282),   # left panel
         "fa_fig1_gpt2.png": (551, 119, 705, 281)}         # right panel
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["pdftoppm", "-f", "2", "-l", "2", "-r", "400", "-png", str(pdf),
                    f"{tmp}/p"], check=True)
    page = Image.open(next(Path(tmp).glob("p*.png"))).convert("RGB")
    s = 4
    for name, (x0, y0, x1, y1) in BOXES.items():
        page.crop((x0 * s, y0 * s, x1 * s, y1 * s)).save(out / name)
        print("wrote", name)
