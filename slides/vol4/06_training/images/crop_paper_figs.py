#!/usr/bin/env python3
"""Crop the published figures this deck shows from the papers' arXiv PDFs.

Usage: python3 crop_paper_figs.py <folder holding the downloaded PDFs>
Download each PDF first as <arXiv id>.pdf from https://arxiv.org/pdf/<id>.
Each crop is rendered at 400 dpi, then shrunk with `sips -Z 1600`.
Clip rectangles are in PDF points (page, x0, y0, x1, y1).
"""
import sys
from pathlib import Path

import pymupdf

OUT = Path(__file__).parent
JOBS = [
    # Chi et al., Diffusion Policy, Figure 3 (multimodal behavior, Push-T)
    ("2303.04137", 5, (44, 40, 302, 134), "paper_dp_multimodal"),
    # Luo et al., HIL-SERL, Figure 5, top row (success rate, two tasks)
    ("2410.21845", 14, (60, 100, 296, 189), "paper_hilserl_curves"),
    # Rudin et al., Figure 1 (simulated ANYmals) and Figure 7 (physical robot)
    ("2109.11978", 1, (118, 436, 494, 581), "paper_anymal_sim"),
    ("2109.11978", 8, (110, 82, 502, 155), "paper_anymal_real"),
    # Zhao et al., ACT/ALOHA, Figure 1, left photo (teleoperation)
    ("2304.13705", 1, (57, 164, 252, 318), "paper_aloha_teleop"),
    # Open X-Embodiment Collaboration, Fig. 1
    ("2310.08864", 1, (54, 488, 558, 642), "paper_oxe_fig1"),
]

if __name__ == "__main__":
    src = Path(sys.argv[1])
    for arxiv, page, rect, name in JOBS:
        pg = pymupdf.open(src / f"{arxiv}.pdf")[page - 1]
        pg.get_pixmap(dpi=400, clip=pymupdf.Rect(*rect)).save(OUT / f"{name}.png")
        print(name)
