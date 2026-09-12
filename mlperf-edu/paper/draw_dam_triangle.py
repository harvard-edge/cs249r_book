#!/usr/bin/env python3
"""Generate publication-grade DAM Taxonomy Triangle diagram for MLPerf EDU paper.

Designed for single-column IEEE/ACM/MLSys format (~3.3-3.5 in print width).
Uses exact mathematical bounding-box centering, no overlapping cards,
and crisp typography.
"""

from pathlib import Path
from generate_paper_plots import draw_dam_triangle

if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_dam_triangle(
        str(out_dir / 'fig_dam_triangle.pdf'),
        str(out_dir / 'fig_dam_triangle.png'),
        str(out_dir / 'fig_dam_triangle.svg'),
    )
