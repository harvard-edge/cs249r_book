#!/usr/bin/env python3
"""Generate publication-grade DAM Taxonomy Triangle diagram for MLPerf EDU paper.

Designed for single-column IEEE/ACM/MLSys format (~3.3-3.5 in print width).
Features an unmistakable geometric triangle with vertex hubs (D, A, M),
orthogonal edge-anchored pairwise intersection callouts, and an unobstructed
central non-decomposability core.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Polygon, Circle

def draw_dam_triangle(out_pdf, out_png, out_svg=None):
    # Configure clean sans-serif math font
    plt.rcParams['mathtext.fontset'] = 'stixsans'
    
    # Proportions: 9.4 x 7.8 gives a spacious, perfectly balanced layout
    fig, ax = plt.subplots(figsize=(9.4, 7.8), dpi=300)
    
    # Coordinate limits with generous margins
    ax.set_xlim(-7.4, 7.4)
    ax.set_ylim(-4.6, 5.4)
    ax.set_aspect('equal')
    ax.axis('off')

    # -------------------------------------------------------------
    # 1. PRIMARY GEOMETRIC TRIANGLE FRAME
    # -------------------------------------------------------------
    v_d = (0.0, 3.60)      # Top apex (Data)
    v_a = (-4.60, -2.40)   # Bottom-left vertex (Algorithm)
    v_m = (4.60, -2.40)    # Bottom-right vertex (Machine)

    # Shaded triangle interior with clean architectural border
    triangle_frame = Polygon([v_d, v_a, v_m], closed=True,
                             facecolor='#F8FAFC', edgecolor='#334155',
                             linewidth=3.0, zorder=1)
    ax.add_patch(triangle_frame)

    # Helper to draw formatted cards
    def draw_card(cx, cy, w, h, bg_color, border_color, lw=1.5, zorder=5):
        left = cx - w / 2.0
        bottom = cy - h / 2.0
        box = FancyBboxPatch((left, bottom), w, h,
                             boxstyle="round,pad=0.04,rounding_size=0.10",
                             facecolor=bg_color, edgecolor=border_color,
                             linewidth=lw, zorder=zorder)
        ax.add_patch(box)
        return box

    # Helper to draw pill badges
    def draw_badge(cx, cy, w, h, bg_color, border_color='none', lw=1.0, zorder=6):
        left = cx - w / 2.0
        bottom = cy - h / 2.0
        badge = FancyBboxPatch((left, bottom), w, h,
                               boxstyle="round,pad=0.03,rounding_size=0.10",
                               facecolor=bg_color, edgecolor=border_color,
                               linewidth=lw, zorder=zorder)
        ax.add_patch(badge)
        return badge

    # -------------------------------------------------------------
    # 2. VERTEX HUBS (D, A, M)
    # -------------------------------------------------------------
    # Corner circular anchor dots at triangle tips
    ax.add_patch(Circle(v_d, 0.20, facecolor='#2563EB', edgecolor='#1E3A8A', lw=2.2, zorder=6))
    ax.add_patch(Circle(v_a, 0.20, facecolor='#7C3AED', edgecolor='#5B21B6', lw=2.2, zorder=6))
    ax.add_patch(Circle(v_m, 0.20, facecolor='#059669', edgecolor='#065F46', lw=2.2, zorder=6))

    # Vertical stems from vertices to cards
    ax.plot([0.0, 0.0], [3.60, 3.97], color='#2563EB', lw=2.2, zorder=4)
    ax.plot([-4.60, -4.60], [-2.40, -3.05], color='#7C3AED', lw=2.2, zorder=4)
    ax.plot([4.60, 4.60], [-2.40, -3.05], color='#059669', lw=2.2, zorder=4)

    # Top Vertex Card: DATA (D)
    cx_d, cy_d = 0.0, 4.55
    w_d, h_d = 4.5, 1.15
    draw_card(cx_d, cy_d, w_d, h_d, '#F0F7FF', '#2563EB', lw=2.0)
    ax.text(cx_d, cy_d + 0.30, "DATA (D)", ha='center', va='center',
            fontsize=11.5, fontweight='bold', color='#1E40AF', zorder=7)
    ax.text(cx_d, cy_d - 0.04, "Sample Pruning, Scaling & Augmentation", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color='#1D4ED8', zorder=7)
    ax.text(cx_d, cy_d - 0.34, "Dataset pruning floors • Quality-to-budget scaling", ha='center', va='center',
            fontsize=7.2, fontstyle='italic', color='#475569', zorder=7)

    # Bottom-Left Vertex Card: ALGORITHM (A)
    cx_a, cy_a = -4.85, -3.65
    w_a, h_a = 3.6, 1.20
    draw_card(cx_a, cy_a, w_a, h_a, '#F5F3FF', '#7C3AED', lw=2.0)
    ax.text(cx_a, cy_a + 0.32, "ALGORITHM (A)", ha='center', va='center',
            fontsize=11.0, fontweight='bold', color='#6B21A8', zorder=7)
    ax.text(cx_a, cy_a - 0.02, "Capacity, Precision & Quantization", ha='center', va='center',
            fontsize=7.8, fontweight='bold', color='#7E22CE', zorder=7)
    ax.text(cx_a, cy_a - 0.34, "FP16 vs. BF16 bifurcation • Dynamic INT8", ha='center', va='center',
            fontsize=7.0, fontstyle='italic', color='#475569', zorder=7)

    # Bottom-Right Vertex Card: MACHINE (M)
    cx_m, cy_m = 4.85, -3.65
    w_m, h_m = 3.6, 1.20
    draw_card(cx_m, cy_m, w_m, h_m, '#F0FDF4', '#059669', lw=2.0)
    ax.text(cx_m, cy_m + 0.32, "MACHINE (M)", ha='center', va='center',
            fontsize=11.0, fontweight='bold', color='#065F46', zorder=7)
    ax.text(cx_m, cy_m - 0.02, "Execution Backends & Kernel Limits", ha='center', va='center',
            fontsize=7.8, fontweight='bold', color='#047857', zorder=7)
    ax.text(cx_m, cy_m - 0.34, "100× UMA acceleration spread • Roofline OI", ha='center', va='center',
            fontsize=7.0, fontstyle='italic', color='#475569', zorder=7)

    # -------------------------------------------------------------
    # 3. PAIRWISE INTERSECTION CALLOUTS & EDGE BADGES
    # -------------------------------------------------------------
    # Left Edge Badge (D ∩ A) sitting right on the diagonal frame
    b_da_x, b_da_y = -2.07, 0.90
    draw_badge(b_da_x, b_da_y, 1.30, 0.40, '#4338CA', '#312E81', lw=1.2, zorder=6)
    ax.text(b_da_x, b_da_y, r"$\mathbf{D} \; \boldsymbol{\cap} \; \mathbf{A}$", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#FFFFFF', zorder=7)

    # Left Flank Card (D ∩ A) - positioned cleanly outside the triangle, leveled with badge
    cx_da, cy_da = -5.10, 0.90
    w_da, h_da = 3.40, 1.25
    draw_card(cx_da, cy_da, w_da, h_da, '#FFFFFF', '#6366F1', lw=1.6)
    # Perfectly horizontal leader line from card right edge to badge left edge
    ax.plot([-3.40, b_da_x - 0.65], [0.90, 0.90], color='#6366F1', lw=1.6, linestyle='-', zorder=4)
    ax.text(cx_da, cy_da + 0.34, "Floors vs. Capacity", ha='center', va='center',
            fontsize=8.6, fontweight='bold', color='#4338CA', zorder=7)
    ax.text(cx_da, cy_da + 0.06, "• Data deficit floor binds universally", ha='center', va='center',
            fontsize=7.0, fontweight='bold', color='#1E293B', zorder=7)
    ax.text(cx_da, cy_da - 0.18, "• Capacity cannot rescue data deficit", ha='center', va='center',
            fontsize=6.8, color='#475569', zorder=7)
    ax.text(cx_da, cy_da - 0.40, "• GCN: 50% data drops acc below floor", ha='center', va='center',
            fontsize=6.8, fontstyle='italic', color='#64748B', zorder=7)

    # Right Edge Badge (D ∩ M) sitting right on the diagonal frame
    b_dm_x, b_dm_y = 2.07, 0.90
    draw_badge(b_dm_x, b_dm_y, 1.30, 0.40, '#0F766E', '#134E4A', lw=1.2, zorder=6)
    ax.text(b_dm_x, b_dm_y, r"$\mathbf{D} \; \boldsymbol{\cap} \; \mathbf{M}$", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#FFFFFF', zorder=7)

    # Right Flank Card (D ∩ M) - positioned cleanly outside the triangle, leveled with badge
    cx_dm, cy_dm = 5.10, 0.90
    w_dm, h_dm = 3.40, 1.25
    draw_card(cx_dm, cy_dm, w_dm, h_dm, '#FFFFFF', '#0D9488', lw=1.6)
    # Perfectly horizontal leader line from badge right edge to card left edge
    ax.plot([b_dm_x + 0.65, 3.40], [0.90, 0.90], color='#0D9488', lw=1.6, linestyle='-', zorder=4)
    ax.text(cx_dm, cy_dm + 0.34, "Working Set vs. Bus", ha='center', va='center',
            fontsize=8.6, fontweight='bold', color='#0F766E', zorder=7)
    ax.text(cx_dm, cy_dm + 0.06, "• 90 MB working set > 8 MB L2 cache", ha='center', va='center',
            fontsize=7.0, fontweight='bold', color='#1E293B', zorder=7)
    ax.text(cx_dm, cy_dm - 0.18, "• Invariant 2.38× hardware speedup", ha='center', va='center',
            fontsize=6.8, color='#475569', zorder=7)
    ax.text(cx_dm, cy_dm - 0.40, "• 150 GB/s bus binds pointer-chasing", ha='center', va='center',
            fontsize=6.8, fontstyle='italic', color='#64748B', zorder=7)

    # Bottom Edge Badge (A ∩ M) sitting right on the horizontal frame
    b_am_x, b_am_y = 0.0, -2.40
    draw_badge(b_am_x, b_am_y, 1.30, 0.40, '#B45309', '#78350F', lw=1.2, zorder=6)
    ax.text(b_am_x, b_am_y, r"$\mathbf{A} \; \boldsymbol{\cap} \; \mathbf{M}$", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#FFFFFF', zorder=7)

    # Bottom Flank Card (A ∩ M) - positioned cleanly below the triangle
    cx_am, cy_am = 0.0, -3.65
    w_am, h_am = 4.40, 1.20
    draw_card(cx_am, cy_am, w_am, h_am, '#FFFFFF', '#D97706', lw=1.6)
    # Perfectly vertical stem line
    ax.plot([0.0, 0.0], [-2.60, -3.05], color='#D97706', lw=1.6, linestyle='-', zorder=4)
    ax.text(cx_am, cy_am + 0.34, "Quantization & Kernel Regressions", ha='center', va='center',
            fontsize=8.6, fontweight='bold', color='#B45309', zorder=7)
    ax.text(cx_am, cy_am + 0.08, "• INT8 MPS fallback: 8.3×–11.0× slowdown", ha='center', va='center',
            fontsize=7.0, fontweight='bold', color='#1E293B', zorder=7)
    ax.text(cx_am, cy_am - 0.14, "• Forward-pass call-pattern paradox:", ha='center', va='center',
            fontsize=6.8, color='#475569', zorder=7)
    ax.text(cx_am, cy_am - 0.36, "  DistilBERT (+10% ops/s) vs. ColBERT (-48% ops/s)", ha='center', va='center',
            fontsize=6.8, fontweight='bold', color='#B45309', zorder=7)

    # -------------------------------------------------------------
    # 4. CENTRAL CORE: THREE-WAY NON-DECOMPOSABILITY (D ∩ A ∩ M)
    # -------------------------------------------------------------
    # Centered with generous breathing room inside the triangle
    cx_core, cy_core = 0.0, -0.85
    w_core, h_core = 4.30, 2.00
    draw_card(cx_core, cy_core, w_core, h_core, '#FFFDF7', '#D97706', lw=2.2, zorder=5)

    # Core Header Pill Badge
    badge_w, badge_h = 4.10, 0.32
    draw_badge(cx_core, cy_core + 0.78, badge_w, badge_h, '#D97706', '#B45309', lw=1.2, zorder=6)
    ax.text(cx_core, cy_core + 0.78, r"$\mathbf{D} \; \boldsymbol{\cap} \; \mathbf{A} \; \boldsymbol{\cap} \; \mathbf{M}$: NON-DECOMPOSABILITY",
            ha='center', va='center', fontsize=8.4, fontweight='bold', color='#FFFFFF', zorder=7)

    # Central Theorem
    ax.text(cx_core, cy_core + 0.48, r"$\mathbf{(D, A, M)^* \ne (D^*, A^*, M^*)}$",
            ha='center', va='center', fontsize=10.2, fontweight='bold', color='#92400E', zorder=6)

    ax.text(cx_core, cy_core + 0.25, "Full 2×2×2 Factorial on Graph Classification",
            ha='center', va='center', fontsize=7.2, fontstyle='italic', color='#78350F', zorder=6)

    # Subtle separator line
    ax.plot([cx_core - 1.8, cx_core + 1.8], [cy_core + 0.10, cy_core + 0.10],
            color='#FDE68A', lw=1.2, zorder=6)

    # Empirical Results: Naive vs True Optimum
    ax.text(cx_core, cy_core - 0.08,
            "Naive Search (D=125, A=64, M=MPS): 30.26× speedup",
            ha='center', va='center', fontsize=6.8, fontweight='bold', color='#DC2626', zorder=6)
    ax.text(cx_core, cy_core - 0.26,
            "Accuracy collapses to 61.80% (-9.94% drop)  ✗ REJECTED",
            ha='center', va='center', fontsize=6.8, color='#B91C1C', zorder=6)

    ax.text(cx_core, cy_core - 0.54,
            "Joint Optimum (D=500, A=64, M=MPS): 6.95× speedup",
            ha='center', va='center', fontsize=6.8, fontweight='bold', color='#16A34A', zorder=6)
    ax.text(cx_core, cy_core - 0.72,
            "Accuracy: 71.62% ≥ 71.74% floor (tolerance)  ✓ ADMITTED",
            ha='center', va='center', fontsize=6.8, color='#15803D', zorder=6)

    # -------------------------------------------------------------
    # 5. CONVERGENCE VECTORS (Pairwise Edge Badges -> Central Core Boundary)
    # -------------------------------------------------------------
    # Left: D∩A badge -> Central Core top-left boundary
    ax.annotate('', xy=(-1.75, 0.18), xytext=(b_da_x + 0.65, b_da_y - 0.12),
                arrowprops=dict(arrowstyle='->', color='#D97706', lw=1.6, linestyle='--', shrinkA=2, shrinkB=2), zorder=4)
    # Right: D∩M badge -> Central Core top-right boundary
    ax.annotate('', xy=(1.75, 0.18), xytext=(b_dm_x - 0.65, b_dm_y - 0.12),
                arrowprops=dict(arrowstyle='->', color='#D97706', lw=1.6, linestyle='--', shrinkA=2, shrinkB=2), zorder=4)
    # Bottom: A∩M badge -> Central Core bottom boundary
    ax.annotate('', xy=(0.0, -1.86), xytext=(0.0, -2.18),
                arrowprops=dict(arrowstyle='->', color='#D97706', lw=1.6, linestyle='--', shrinkA=2, shrinkB=2), zorder=4)

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', metadata={'CreationDate': None})
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    if out_svg:
        plt.savefig(out_svg, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated:\n  - {out_pdf}\n  - {out_png}")

if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_dam_triangle(
        str(out_dir / 'fig_dam_triangle.pdf'),
        str(out_dir / 'fig_dam_triangle.png'),
        str(out_dir / 'fig_dam_triangle.svg'),
    )

