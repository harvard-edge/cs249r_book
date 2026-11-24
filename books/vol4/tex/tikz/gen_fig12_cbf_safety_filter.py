#!/usr/bin/env python3
"""Generate fig12_cbf_safety_filter (schematic state-space CBF and torque-space QP projection)."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from scipy.optimize import minimize

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["svg.fonttype"] = "path"

harvardcrimson = "#A51C30"
safegreen = "#059669"
ethdarkblue = "#1F407A"
ink = "#1A202C"
paleink = "#94A3B8"
softink = "#4A5568"
amber = "#B45309"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.6))

# ════════════════════════════════════════════════════════════════════════
# Panel (a): State space (schematic kinetic barrier inset)
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
ax.text(
    6.05, 5.85, r"$\partial\mathcal{C}$", fontsize=12, fontweight="bold",
    color=harvardcrimson, ha="left", va="center",
)

inset_y = 0.12 * xs**2 + 0.2 * xs + 2.2
ax.plot(xs, inset_y, color=ethdarkblue, linewidth=1.5, linestyle="--", zorder=2)
ax.text(
    0.20, 3.30, "Inset\nboundary", fontsize=9.5, color=ethdarkblue,
    ha="left", va="center", style="italic",
)

xm = 3.8
ym_lo = 0.12 * xm**2 + 0.2 * xm + 1.0
ym_hi = 0.12 * xm**2 + 0.2 * xm + 2.2
ax.annotate(
    "", xy=(xm, ym_lo + 0.05), xytext=(xm, ym_hi - 0.05),
    arrowprops=dict(arrowstyle="<->", color=ethdarkblue, lw=1.2),
)
ax.text(
    xm + 0.2, (ym_lo + ym_hi) / 2, r"$\delta$",
    fontsize=11, color=ethdarkblue, ha="left", va="center",
)

ax.text(2.8, 5.85, r"Viable Set $\mathcal{C}$", fontsize=11, fontweight="bold", color=safegreen)
ax.text(5.2, 0.55, "Forbidden", fontsize=11, fontweight="bold", color=harvardcrimson, ha="center")

x_pt = np.array([2.0, 5.0])
ax.plot(*x_pt, "o", color=ink, markersize=6, zorder=5)
ax.text(x_pt[0] - 0.2, x_pt[1] + 0.15, r"$x$", fontsize=11.5, color=ink, ha="right", va="bottom")

u_nom_dir = np.array([1.6, -1.8])
ax.annotate(
    "", xy=x_pt + u_nom_dir, xytext=x_pt,
    arrowprops=dict(arrowstyle="-|>", color=harvardcrimson, lw=2.0, linestyle="--"),
)
ax.text(
    x_pt[0] + u_nom_dir[0] + 0.1, x_pt[1] + u_nom_dir[1] - 0.25,
    r"$f(x,\, u_{\mathrm{nom}})$", fontsize=9.5, fontweight="bold", color=harvardcrimson, ha="left",
)

u_safe_dir = np.array([-0.3, 1.4])
ax.annotate(
    "", xy=x_pt + u_safe_dir, xytext=x_pt,
    arrowprops=dict(arrowstyle="-|>", color=safegreen, lw=2.2),
)
ax.text(
    x_pt[0] + u_safe_dir[0] + 0.1, x_pt[1] + u_safe_dir[1] + 0.15,
    r"$f(x,\, u^*)$", fontsize=9.5, fontweight="bold", color=safegreen, ha="left",
)

ax.add_patch(Rectangle((0, 0), 7, 7, fill=False, edgecolor=paleink, linewidth=0.8, zorder=1))
ax.set_title("(a)  State Space", fontsize=10.5, fontweight="bold", color=ink, pad=10)

# ════════════════════════════════════════════════════════════════════════
# Panel (b): Control space (actuator box U = [-u_max, u_max]^2 and CBF half-space)
# Schematic values: u_max = 3.0, a_cbf = [1, 2], b_cbf = 4.5, u_nom = [3.0, 3.75].
# Componentwise clamp: u_clip = clamp(u_nom, -u_max, u_max) = [3.0, 3.0] on corner of U,
# which still violates a_cbf^T u <= b_cbf (9.0 > 4.5).
# True QP projection onto U_safe: u* = [1.80, 1.35].
# ════════════════════════════════════════════════════════════════════════
ax = ax2
u_max = 3.0
a_cbf = np.array([1.0, 2.0])
b_cbf = 4.5
u_nom = np.array([3.0, 3.75])
u_clip = np.clip(u_nom, -u_max, u_max)

res = minimize(
    lambda u: float(np.sum((u - u_nom) ** 2)),
    x0=np.array([1.5, 1.5]),
    bounds=[(-u_max, u_max), (-u_max, u_max)],
    constraints=[{"type": "ineq", "fun": lambda u: b_cbf - float(a_cbf @ u)}],
)
assert res.success, "QP projection failed"
u_star = res.x
assert np.all(np.abs(u_clip) <= np.abs(u_nom) + 1e-12), "Clamp increased magnitude"
assert float(a_cbf @ u_clip) > b_cbf, "u_clip must violate CBF half-space"
assert np.allclose(u_star, np.array([1.80, 1.35]), atol=1e-6), f"Unexpected u*: {u_star}"

ax.set_xlim(-0.55, 5.15)
ax.set_ylim(-0.55, 4.45)
ax.set_aspect("equal")

# Positive quadrant of U_safe: vertices (0, 0), (3, 0), (3, 0.75), (0, 2.25)
y_at_0 = b_cbf / a_cbf[1]
y_at_umax = (b_cbf - a_cbf[0] * u_max) / a_cbf[1]
safe_poly = np.array([[0.0, 0.0], [u_max, 0.0], [u_max, y_at_umax], [0.0, y_at_0]])
ax.add_patch(Polygon(safe_poly, closed=True, facecolor=safegreen, alpha=0.11, edgecolor="none", zorder=0))

# Unsafe portion of U above CBF line (light amber tint to make U's corner visually clear)
unsafe_box_poly = np.array([[0.0, y_at_0], [u_max, y_at_umax], [u_max, u_max], [0.0, u_max]])
ax.add_patch(Polygon(unsafe_box_poly, closed=True, facecolor=harvardcrimson, alpha=0.04, edgecolor="none", zorder=0))

# Actuator box U boundary in first quadrant (x <= u_max, y <= u_max)
ax.plot(
    [0.0, u_max, u_max], [u_max, u_max, 0.0],
    color=softink, linewidth=1.3, linestyle=(0, (5, 3)), zorder=2,
)
ax.text(1.35, u_max + 0.12, r"Actuator box $\mathcal{U}$ ($|u_i| \leq u_{\max}$)",
        fontsize=9.2, color=softink, ha="center", va="bottom")

# Tick labels for u_max on both axes
ax.plot([u_max, u_max], [-0.08, 0.08], color=softink, lw=1.0, zorder=3)
ax.text(u_max, -0.16, r"$u_{\max}$", fontsize=9.5, color=softink, ha="center", va="top")
ax.plot([-0.08, 0.08], [u_max, u_max], color=softink, lw=1.0, zorder=3)
ax.text(-0.14, u_max, r"$u_{\max}$", fontsize=9.5, color=softink, ha="right", va="center")

# Feasible region label inside U_safe (upper-left open region)
ax.text(0.68, 1.40, r"$\mathcal{U}_{\mathrm{safe}}$",
        fontsize=10.0, fontweight="bold", color=safegreen, ha="center", va="center")

# CBF half-space boundary line: u1 + 2*u2 = 4.5
cx = np.linspace(-0.25, 4.65, 100)
cy = (b_cbf - a_cbf[0] * cx) / a_cbf[1]
mask = cy >= -0.20
ax.plot(cx[mask], cy[mask], color=ethdarkblue, linewidth=2.0, zorder=2)

ax.text(
    3.92, 0.52, r"$\mathbf{a}_{\mathrm{cbf}}^\top \mathbf{u} = b_{\mathrm{cbf}}$",
    fontsize=9.5, color=ethdarkblue, rotation=-26.5, ha="center", va="bottom",
)

# Coordinate axes
ax.annotate("", xy=(4.95, 0), xytext=(-0.35, 0), arrowprops=dict(arrowstyle="-|>", color=paleink, lw=1.0))
ax.annotate("", xy=(0, 4.30), xytext=(0, -0.35), arrowprops=dict(arrowstyle="-|>", color=paleink, lw=1.0))
ax.text(4.95, -0.18, r"$u_1$", fontsize=11, color=ink, ha="center", va="top")
ax.text(-0.18, 4.25, r"$u_2$", fontsize=11, color=ink, ha="right", va="center")

for sp in ("left", "bottom", "right", "top"):
    ax.spines[sp].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

origin = np.array([0.0, 0.0])

# u_nom (outside actuator box U on u_2)
ax.annotate(
    "", xy=u_nom, xytext=origin,
    arrowprops=dict(arrowstyle="-|>", color=harvardcrimson, lw=1.8, linestyle="--"),
)
ax.plot(*u_nom, "o", color=harvardcrimson, markersize=5.5, zorder=5)
ax.text(
    u_nom[0] + 0.14, u_nom[1] + 0.06, r"$u_{\mathrm{nom}}$",
    fontsize=10.5, fontweight="bold", color=harvardcrimson, ha="left", va="bottom",
)

# Componentwise clamp u_clip = clamp(u_nom, -u_max, u_max) = (3.0, 3.0)
ax.annotate(
    "", xy=u_clip, xytext=origin,
    arrowprops=dict(arrowstyle="-|>", color=amber, lw=1.3, linestyle=(0, (4, 3))),
)
# Vertical clamp drop arrow from u_nom down onto corner of U
ax.annotate(
    "", xy=(u_clip[0], u_clip[1] + 0.07), xytext=(u_nom[0], u_nom[1] - 0.07),
    arrowprops=dict(arrowstyle="-|>", color=amber, lw=1.2, linestyle=":"),
)
ax.plot(*u_clip, "s", color=amber, markersize=5.2, zorder=5)
ax.text(
    u_clip[0] + 0.15, u_clip[1] - 0.04, r"$u_{\mathrm{clip}}$ (clamp, unsafe)",
    fontsize=9.2, fontweight="bold", color=amber, ha="left", va="center",
)

# True QP projection u* onto U_safe
ax.annotate(
    "", xy=u_star, xytext=origin,
    arrowprops=dict(arrowstyle="-|>", color=safegreen, lw=2.1),
)
ax.plot(*u_star, "o", color=safegreen, markersize=5.5, zorder=5)
ax.text(
    u_star[0] - 0.06, u_star[1] - 0.28, r"$u^*$",
    fontsize=11, fontweight="bold", color=safegreen, ha="center", va="top",
)

# Euclidean projection segment Delta u = u* - u_nom
ax.plot(
    [u_nom[0], u_star[0]], [u_nom[1], u_star[1]],
    color=ethdarkblue, linewidth=1.4, linestyle="--", zorder=3,
)
ax.text(
    2.22, 1.92, r"$\Delta u$",
    fontsize=10, fontweight="bold", color=ethdarkblue, ha="left", va="center",
)

# Orthogonal right-angle mark at u*
ps = 0.17
d_along = np.array([2.0, -1.0]) / np.sqrt(5.0)
d_perp = a_cbf / np.linalg.norm(a_cbf)
p1 = u_star + ps * d_along
p2 = p1 + ps * d_perp
p3 = u_star + ps * d_perp
ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color=ethdarkblue, linewidth=1.0, zorder=4)

ax.set_title("(b)  Control Space Projection", fontsize=10.5, fontweight="bold", color=ink, pad=10)

plt.tight_layout(w_pad=2.2)

_here = Path(__file__).resolve().parent
_root = _here.parents[1]
(_root / "12_enforcement" / "images" / "png").mkdir(parents=True, exist_ok=True)
(_root / "12_enforcement" / "images" / "svg").mkdir(parents=True, exist_ok=True)
plt.savefig(
    _root / "12_enforcement" / "images" / "png" / "fig12_cbf_safety_filter.png",
    dpi=300, bbox_inches="tight", facecolor="white",
)
_svg_out = _root / "12_enforcement" / "images" / "svg" / "fig12_cbf_safety_filter.svg"
plt.savefig(
    _svg_out,
    bbox_inches="tight", facecolor="white",
)
_svg_out.write_text("\n".join(line.rstrip() for line in _svg_out.read_text().splitlines()) + "\n")
print("OK: fig12_cbf_safety_filter")
