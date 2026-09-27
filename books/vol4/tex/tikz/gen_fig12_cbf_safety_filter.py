#!/usr/bin/env python3
"""Generate fig12_cbf_safety_filter as a matplotlib figure."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["mathtext.default"] = "regular"

harvardcrimson = "#A51C30"
safegreen      = "#059669"
ethdarkblue    = "#1F407A"
ink            = "#1A202C"
paleink        = "#A0AEC0"
softink        = "#4A5568"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# ════════════════════════════════════════════════════════════════════════
# Panel (a): State space
# ════════════════════════════════════════════════════════════════════════
ax = ax1
ax.set_xlim(0, 7)
ax.set_ylim(0, 7)
ax.set_aspect("equal")
ax.axis("off")

xs = np.linspace(0, 7, 300)
boundary_y = 0.12 * xs**2 + 0.2 * xs + 1.0
ax.fill_between(xs, boundary_y, 7, color=safegreen, alpha=0.10, zorder=0)
ax.fill_between(xs, 0, boundary_y, color=harvardcrimson, alpha=0.10, zorder=0)

ax.plot(xs, boundary_y, color=harvardcrimson, linewidth=2.2, zorder=2)
ax.text(6.3, 7.0, r"$\partial\mathcal{C}$", fontsize=14, fontweight="bold",
        color=harvardcrimson, ha="center", va="top", clip_on=False)

inset_y = 0.12 * xs**2 + 0.2 * xs + 2.2
ax.plot(xs, inset_y, color=ethdarkblue, linewidth=1.5, linestyle="--", zorder=2)
ax.text(0.15, 3.2, "Inset\nBoundary", fontsize=7.5, color=ethdarkblue,
        ha="left", va="center", style="italic")

# δ margin at x=3.8
xm = 3.8
ym_lo = 0.12 * xm**2 + 0.2 * xm + 1.0
ym_hi = 0.12 * xm**2 + 0.2 * xm + 2.2
ax.annotate("", xy=(xm, ym_lo + 0.05), xytext=(xm, ym_hi - 0.05),
            arrowprops=dict(arrowstyle="<->", color=ethdarkblue, lw=1.2))
ax.text(xm + 0.2, (ym_lo + ym_hi) / 2, r"$\delta$",
        fontsize=12, color=ethdarkblue, ha="left", va="center")

ax.text(3.0, 5.8, r"Viable Set $\mathcal{C}$", fontsize=11, fontweight="bold",
        color=safegreen)
ax.text(5.2, 0.5, "Forbidden", fontsize=11, fontweight="bold",
        color=harvardcrimson, ha="center")

x_pt = np.array([2.0, 5.0])
ax.plot(*x_pt, "o", color=ink, markersize=6, zorder=5)
ax.text(x_pt[0] - 0.2, x_pt[1] + 0.15, r"$x$", fontsize=12, color=ink,
        ha="right", va="bottom")

u_nom_dir = np.array([1.6, -1.8])
ax.annotate("", xy=x_pt + u_nom_dir, xytext=x_pt,
            arrowprops=dict(arrowstyle="-|>", color=harvardcrimson,
                           lw=2.0, linestyle="--"))
ax.text(x_pt[0] + u_nom_dir[0] + 0.1, x_pt[1] + u_nom_dir[1] - 0.2,
        r"$f(x,\, u_{\mathrm{nom}})$", fontsize=9, color=harvardcrimson, ha="left")

u_safe_dir = np.array([-0.3, 1.4])
ax.annotate("", xy=x_pt + u_safe_dir, xytext=x_pt,
            arrowprops=dict(arrowstyle="-|>", color=safegreen, lw=2.2))
ax.text(x_pt[0] + u_safe_dir[0] + 0.1, x_pt[1] + u_safe_dir[1] + 0.15,
        r"$f(x,\, u^*)$", fontsize=9, color=safegreen, ha="left")

ax.add_patch(Rectangle((0, 0), 7, 7, fill=False, edgecolor=paleink,
                        linewidth=0.8, zorder=1))
ax.set_title("(a)  State Space", fontsize=10, fontweight="bold", color=ink, pad=10)

# ════════════════════════════════════════════════════════════════════════
# Panel (b): Control space
# Constraint: u1 + 2*u2 = 5  (shallow slope ~-0.5)
# This gives good visual separation between vectors and the line.
# ════════════════════════════════════════════════════════════════════════
ax = ax2
ax.set_xlim(-0.5, 6)
ax.set_ylim(-0.5, 5)
ax.set_aspect("equal")

# Admissible half-space (below the line)
fill_x = np.array([-0.5, -0.5, 5.0, 6.0])
fill_y = np.array([2.75, -0.5, -0.5, -0.5])
ax.fill(fill_x, fill_y, color=ethdarkblue, alpha=0.06, zorder=0, clip_on=True)

# Constraint line: u2 = (5 - u1)/2
cx = np.linspace(-0.3, 5.5, 100)
cy = (5 - cx) / 2
ax.plot(cx, cy, color=ethdarkblue, linewidth=2.0, zorder=2)

# Label for constraint — on upper portion of line, clear of vectors
ax.text(0.7, 2.5, r"$a^\top u = b$", fontsize=9, fontweight="bold",
        color=ethdarkblue, rotation=-27, ha="center", va="bottom")

# Axes arrows
ax.annotate("", xy=(5.8, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=paleink, lw=1.0))
ax.annotate("", xy=(0, 4.8), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=paleink, lw=1.0))
ax.text(5.8, -0.3, r"$u_1$", fontsize=11, color=ink, ha="center", va="top")
ax.text(-0.3, 4.8, r"$u_2$", fontsize=11, color=ink, ha="center", va="bottom")

ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

origin = np.array([0, 0])

# u_nom — infeasible, above the constraint
u_nom = np.array([3.0, 3.5])
ax.annotate("", xy=u_nom, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=harvardcrimson,
                           lw=1.8, linestyle="--"))
ax.plot(*u_nom, "o", color=harvardcrimson, markersize=5, zorder=5)
ax.text(u_nom[0] + 0.15, u_nom[1] + 0.15, r"$u_{\mathrm{nom}}$",
        fontsize=10, fontweight="bold", color=harvardcrimson)

# u* — orthogonal projection onto u1 + 2*u2 = 5
# a=(1,2), aT*u_nom = 3+7=10, excess = 10-5=5, |a|^2=5
# u* = (3,3.5) - (5/5)*(1,2) = (2.0, 1.5)
u_star = np.array([2.0, 1.5])
ax.annotate("", xy=u_star, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=safegreen, lw=2.0))
ax.plot(*u_star, "o", color=safegreen, markersize=5, zorder=5)
ax.text(u_star[0] - 0.5, u_star[1] - 0.1, r"$u^*$",
        fontsize=10, fontweight="bold", color=safegreen, ha="right")

# Δu projection line
ax.plot([u_nom[0], u_star[0]], [u_nom[1], u_star[1]],
        color=ethdarkblue, linewidth=1.3, linestyle="--", zorder=3)
mid = (u_nom + u_star) / 2
ax.text(mid[0] + 0.3, mid[1] + 0.05, r"$\Delta u$",
        fontsize=10, color=ethdarkblue, ha="left")

# Right-angle mark at u*
ps = 0.18
d_along = np.array([2, -1]) / np.sqrt(5)   # along constraint (u1+2u2=5)
d_perp  = np.array([1,  2]) / np.sqrt(5)    # normal
p1 = u_star + ps * d_along
p2 = p1 + ps * d_perp
p3 = u_star + ps * d_perp
ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]],
        color=ethdarkblue, linewidth=1.0, zorder=4)

# u_clip — per-axis clamping (different from QP projection, suboptimal)
u_clip = np.array([4.5, 1.0])
ax.annotate("", xy=u_clip, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=softink, lw=1.0,
                           linestyle=(0, (4, 3))))
ax.plot(*u_clip, "o", color=softink, markersize=4, zorder=5)
ax.text(u_clip[0] + 0.15, u_clip[1] + 0.15, r"$u_{\mathrm{clip}}$",
        fontsize=8.5, color=softink)

# Guide from u_nom to u_clip
ax.plot([u_nom[0], u_clip[0]], [u_nom[1], u_clip[1]],
        color=paleink, linewidth=0.8, linestyle=":", zorder=2)

ax.set_title("(b)  Control Space Projection", fontsize=10, fontweight="bold",
             color=ink, pad=10)

plt.tight_layout(w_pad=2.5)

_here = __import__("pathlib").Path(__file__).resolve().parent
_root = _here.parents[1]
(_root / "12_enforcement" / "images" / "png").mkdir(parents=True, exist_ok=True)
(_root / "12_enforcement" / "images" / "svg").mkdir(parents=True, exist_ok=True)
plt.savefig(_root / "12_enforcement" / "images" / "png" / "fig12_cbf_safety_filter.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(_root / "12_enforcement" / "images" / "svg" / "fig12_cbf_safety_filter.svg",
            bbox_inches="tight", facecolor="white")
print("OK: fig12_cbf_safety_filter")
