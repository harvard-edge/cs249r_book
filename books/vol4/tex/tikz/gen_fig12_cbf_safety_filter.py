#!/usr/bin/env python3
"""Generate fig12_cbf_safety_filter as a matplotlib figure."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

# ── Book style ──────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["mathtext.default"] = "regular"

# ── Colors ──────────────────────────────────────────────────────────────
harvardcrimson = "#A51C30"
safegreen      = "#059669"
ethdarkblue    = "#1F407A"
ink            = "#1A202C"
paleink        = "#A0AEC0"
softink        = "#4A5568"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# ════════════════════════════════════════════════════════════════════════
# Panel (a): State space — viable boundary
# ════════════════════════════════════════════════════════════════════════
ax = ax1
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect("equal")
ax.axis("off")

# Filled regions
# Viable (green, upper-left of boundary)
xs = np.linspace(-0.5, 6.5, 300)
boundary_y = 0.15 * xs**2 + 0.3 * xs + 0.5  # quadratic boundary
ax.fill_between(xs, boundary_y, 6.5, color=safegreen, alpha=0.10, zorder=0)
ax.fill_between(xs, -0.5, boundary_y, color=harvardcrimson, alpha=0.10, zorder=0)

# Boundary ∂C
ax.plot(xs, boundary_y, color=harvardcrimson, linewidth=2.0, zorder=2)
ax.text(5.2, 4.8, r"$\partial\mathcal{C}$", fontsize=13, fontweight="bold",
        color=harvardcrimson, ha="center", va="center")

# Inset boundary (shifted up)
inset_y = 0.15 * xs**2 + 0.3 * xs + 1.5
ax.plot(xs, inset_y, color=ethdarkblue, linewidth=1.5, linestyle="--", zorder=2)
ax.text(0.7, 3.0, "Inset\nBoundary", fontsize=8, color=ethdarkblue,
        ha="center", va="center", rotation=28)

# Margin δ
ax.annotate("", xy=(3.0, 0.15*9+0.9+0.5), xytext=(3.0, 0.15*9+0.9+1.5),
            arrowprops=dict(arrowstyle="<->", color=ethdarkblue, lw=1.2))
ax.text(3.25, (0.15*9+0.9+0.5 + 0.15*9+0.9+1.5)/2, r"$\delta$",
        fontsize=11, color=ethdarkblue, ha="left", va="center")

# Set labels
ax.text(1.2, 5.5, r"Viable Set $\mathcal{C}$", fontsize=11, fontweight="bold",
        color=safegreen, ha="left")
ax.text(4.8, 0.8, "Forbidden", fontsize=11, fontweight="bold",
        color=harvardcrimson, ha="center")

# State point x
x_pt = np.array([2.0, 4.5])
ax.plot(*x_pt, "o", color=ink, markersize=6, zorder=5)
ax.text(x_pt[0] - 0.25, x_pt[1] + 0.2, r"$x$", fontsize=11, color=ink,
        ha="right", va="bottom")

# Unsafe vector f(x, u_nom) → toward boundary
u_nom_dir = np.array([1.8, -1.5])
ax.annotate("", xy=x_pt + u_nom_dir, xytext=x_pt,
            arrowprops=dict(arrowstyle="-|>", color=harvardcrimson,
                           lw=2.0, linestyle="--"))
ax.text(x_pt[0] + u_nom_dir[0] + 0.15, x_pt[1] + u_nom_dir[1] - 0.15,
        r"$f(x, u_{\mathrm{nom}})$", fontsize=9, color=harvardcrimson,
        ha="left", va="top")

# Safe vector f(x, u*) → away from boundary
u_safe_dir = np.array([0.3, 1.5])
ax.annotate("", xy=x_pt + u_safe_dir, xytext=x_pt,
            arrowprops=dict(arrowstyle="-|>", color=safegreen, lw=2.0))
ax.text(x_pt[0] + u_safe_dir[0] + 0.15, x_pt[1] + u_safe_dir[1] + 0.1,
        r"$f(x, u^*)$", fontsize=9, color=safegreen, ha="left", va="bottom")

# Border
ax.add_patch(Rectangle((-0.5, -0.5), 7, 7, fill=False, edgecolor=paleink,
                        linewidth=0.8, zorder=1))

ax.set_title("(a)  Schematic Viable Boundary", fontsize=10, fontweight="bold",
             color=ink, pad=8)

# ════════════════════════════════════════════════════════════════════════
# Panel (b): Control space — projection
# ════════════════════════════════════════════════════════════════════════
ax = ax2
ax.set_xlim(-0.8, 5.5)
ax.set_ylim(-0.8, 5.5)
ax.set_aspect("equal")

# Admissible half-space fill (u1 + u2 <= 4)
xs_fill = np.linspace(-0.8, 4.8, 100)
ys_fill = 4.0 - xs_fill
ax.fill_between(xs_fill, -0.8, np.minimum(ys_fill, 5.5),
                where=(ys_fill > -0.8), color=ethdarkblue, alpha=0.06, zorder=0)

# Constraint hyperplane: u1 + u2 = 4
ax.plot([-0.5, 4.5], [4.5, -0.5], color=ethdarkblue, linewidth=2.0, zorder=2)
ax.text(0.5, 4.0, r"$a^{\mathsf{T}} u = b$", fontsize=10, fontweight="bold",
        color=ethdarkblue, ha="center", va="bottom", rotation=-45)

# Axes
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r"$u_1$", fontsize=11, color=ink, labelpad=-8)
ax.xaxis.set_label_coords(1.02, -0.02)
ax.set_ylabel(r"$u_2$", fontsize=11, color=ink, labelpad=-8, rotation=0)
ax.yaxis.set_label_coords(-0.02, 1.02)

# Origin
origin = np.array([0, 0])

# u_nom — outside (upper right, infeasible)
u_nom = np.array([3.8, 3.6])
ax.annotate("", xy=u_nom, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=harvardcrimson,
                           lw=1.8, linestyle="--"))
ax.plot(*u_nom, "o", color=harvardcrimson, markersize=5, zorder=5)
ax.text(u_nom[0] + 0.15, u_nom[1] + 0.15, r"$u_{\mathrm{nom}}$",
        fontsize=10, fontweight="bold", color=harvardcrimson, ha="left")

# u* — projection onto constraint line (orthogonal: (3.8+3.6)/2 = 3.7 → (1.9, 2.1))
# For u1+u2=4, projection of (3.8,3.6): u* = (3.8,3.6) - ((3.8+3.6-4)/2)(1,1) = (2.1,1.9)
u_star = np.array([2.1, 1.9])
ax.annotate("", xy=u_star, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=safegreen, lw=2.0))
ax.plot(*u_star, "o", color=safegreen, markersize=5, zorder=5)
ax.text(u_star[0] - 0.3, u_star[1] - 0.35, r"$u^*$",
        fontsize=10, fontweight="bold", color=safegreen, ha="right")

# Δu projection line
ax.plot([u_nom[0], u_star[0]], [u_nom[1], u_star[1]],
        color=ethdarkblue, linewidth=1.3, linestyle="--", zorder=3)
mid_du = (u_nom + u_star) / 2
ax.text(mid_du[0] + 0.2, mid_du[1] + 0.15, r"$\Delta u$",
        fontsize=10, color=ethdarkblue, ha="left")

# Right-angle mark at u*
# Perpendicular indicator
perp_size = 0.2
# Direction along constraint: (1,-1)/√2
d_along = np.array([1, -1]) / np.sqrt(2)
# Direction toward u_nom: (1,1)/√2
d_perp = np.array([1, 1]) / np.sqrt(2)
p1 = u_star + perp_size * d_along
p2 = p1 + perp_size * d_perp
p3 = u_star + perp_size * d_perp
ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]],
        color=ethdarkblue, linewidth=1.0, zorder=4)

# u_clip — component-wise clamping (wrong answer)
u_clip = np.array([2.0, 3.6])
ax.annotate("", xy=u_clip, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=softink, lw=1.2,
                           linestyle=(0, (4, 3))))
ax.plot(*u_clip, "o", color=softink, markersize=4, zorder=5)
ax.text(u_clip[0] - 0.15, u_clip[1] + 0.2, r"$u_{\mathrm{clip}}$",
        fontsize=8.5, color=softink, ha="right")

# Guide from u_nom to u_clip
ax.plot([u_nom[0], u_clip[0]], [u_nom[1], u_clip[1]],
        color=paleink, linewidth=0.8, linestyle=":", zorder=2)

ax.set_title("(b)  Control Space Projection", fontsize=10, fontweight="bold",
             color=ink, pad=8)

plt.tight_layout(w_pad=2.0)

# Output paths relative to this script
_here = __import__("pathlib").Path(__file__).resolve().parent
_root = _here.parents[1]  # books/vol4/
(_root / "12_enforcement" / "images" / "png").mkdir(parents=True, exist_ok=True)
(_root / "12_enforcement" / "images" / "svg").mkdir(parents=True, exist_ok=True)
plt.savefig(_root / "12_enforcement" / "images" / "png" / "fig12_cbf_safety_filter.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(_root / "12_enforcement" / "images" / "svg" / "fig12_cbf_safety_filter.svg",
            bbox_inches="tight", facecolor="white")
print("OK: fig12_cbf_safety_filter")
